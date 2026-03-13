"""
Download Qwen3-VL-8B-Thinking, load to GPU (bf16 + flash attention),
run inference on Mind2Web test_website, save predictions + metrics.

Usage
-----
    python run_inference.py
    python run_inference.py --n 100          # first 100 examples
    python run_inference.py --split test_task
    python run_inference.py --no-scoring     # generation mode instead of candidate scoring
"""

import argparse
import json
import os
import re
import time
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from mind2web_dataset import get_dataloader, Mind2WebDataset
from mind2web_metrics import evaluate


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT = {
    "model_id":          "Qwen/Qwen3-VL-8B-Thinking",
    "split":             "test_website",
    "max_html_chars":    12_000,
    "max_new_tokens":    512,
    "scoring_max_new_tokens": 10,
    "use_candidate_scoring":  True,   # log-prob scoring when n_cands <= scoring_threshold
    "scoring_threshold":      50,     # raise or set to 9999 to always score
    "output_dir":        "outputs",
}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str):
    """Load Qwen3-VL with bf16 + flash attention, fall back to sdpa."""
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, attn_implementation="flash_attention_2", **kwargs
        )
        print("Loaded with flash_attention_2")
    except Exception:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, attn_implementation="sdpa", **kwargs
        )
        print("Loaded with sdpa (flash_attention_2 unavailable)")

    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


# ── Prompt / message helpers ──────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a web navigation agent. You are given a screenshot of a webpage, "
    "the cleaned HTML of the page, and a list of candidate actions. "
    "Your job is to select the single best action that accomplishes the user's task. "
    "Do NOT invent new actions. You MUST choose from the provided candidates only.\n\n"
    "After your reasoning, you MUST end your response with exactly:\nAnswer: <index>\n"
    "where <index> is the zero-based integer index of your chosen action."
)


def build_messages(example: dict, max_html_chars: int) -> list:
    html = example["cleaned_html"]
    if len(html) > max_html_chars:
        half = max_html_chars // 2
        html = html[:half] + "\n... [TRUNCATED] ...\n" + html[-half:]

    candidates = example["action_reprs"]
    cand_str = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))

    text = (
        f"Task: {example['confirmed_task']}\n\n"
        f"Cleaned HTML:\n{html}\n\n"
        f"Candidate Actions:\n{cand_str}\n\n"
        f"Select the correct action index. Respond with:\nAnswer: <index>"
    )

    content = []
    img = example["screenshot"]
    if img is not None:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": text})

    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": content},
    ]


def apply_chat_template(processor, model, messages: list) -> dict:
    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        )
    except TypeError:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
    return {k: v.to(model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()}


# ── Inference modes ───────────────────────────────────────────────────────────

def score_candidates(model, processor, cached_inputs: dict, n_candidates: int,
                     max_new_tokens: int) -> tuple[int, list[int], list[float]]:
    """
    Log-prob candidate scoring with KV caching.

    The prompt is run through the model exactly once to build the KV cache.
    Each candidate's answer tokens are then scored using only the cached
    key-value pairs, avoiding O(N × prompt_len²) repeated attention.

    Returns (top1_idx, top3_indices, scores).
    """
    # ── Step 1: run prompt once, cache KV pairs ───────────────────────────────
    prompt_fwd = dict(cached_inputs)
    prompt_fwd["use_cache"] = True

    with torch.no_grad():
        prompt_out = model(**prompt_fwd)

    past_kv   = prompt_out.past_key_values          # cached prompt KV pairs
    prompt_len = cached_inputs["input_ids"].shape[1]

    # Extend attention mask shape helper
    base_mask = cached_inputs["attention_mask"]      # (1, prompt_len)

    # ── Step 2: score each candidate using only its answer tokens ─────────────
    scores = []
    for c_idx in range(n_candidates):
        answer_text = f"Answer: {c_idx}"
        answer_ids  = processor.tokenizer.encode(answer_text, add_special_tokens=False)
        ans_tensor  = torch.tensor([answer_ids], device=model.device)

        # Attention mask must cover prompt + answer tokens
        ext_mask  = torch.ones(1, ans_tensor.shape[1], device=model.device,
                               dtype=base_mask.dtype)
        full_mask = torch.cat([base_mask, ext_mask], dim=1)

        with torch.no_grad():
            out = model(
                input_ids      = ans_tensor,
                attention_mask = full_mask,
                past_key_values= past_kv,
                use_cache      = False,   # no need to extend cache further
            )

        logits = out.logits   # (1, answer_len, vocab)
        log_probs = []
        for t, tok in enumerate(answer_ids):
            # position t in logits predicts token t+1;
            # for t=0 the prompt's last hidden state (position -1) predicts answer[0]
            lp = torch.log_softmax(logits[0, t, :], dim=-1)
            log_probs.append(lp[tok].item())

        scores.append(sum(log_probs) / len(log_probs))

    ranked = sorted(range(n_candidates), key=lambda i: scores[i], reverse=True)
    return ranked[0], ranked[:3], scores


def generate_answer(model, processor, inputs: dict,
                    max_new_tokens: int) -> str:
    """Free-form generation. Returns the raw decoded output string."""
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    out_ids = generated[0][inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(out_ids, skip_special_tokens=True)
    return raw


def parse_answer(text: str, n_candidates: int) -> int:
    for pat in [r"Answer:\s*(\d+)", r"answer:\s*(\d+)", r"\b(\d+)\s*$"]:
        m = re.search(pat, text)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < n_candidates:
                return idx
    for n_str in reversed(re.findall(r"\b(\d+)\b", text)):
        idx = int(n_str)
        if 0 <= idx < n_candidates:
            return idx
    return -1


# ── Main inference loop ───────────────────────────────────────────────────────

def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Loading model: {args.model_id}")
    model, processor = load_model(args.model_id)

    print(f"Loading dataset: split={args.split}")
    dataset = Mind2WebDataset(
        split=args.split,
        max_html_chars=args.max_html_chars,
    )
    total = len(dataset) if args.n is None else min(args.n, len(dataset))

    predictions = []
    use_scoring = args.use_scoring

    pbar = tqdm(range(total), desc="Inference", dynamic_ncols=True)
    for i in pbar:
        example  = dataset[i]
        cands    = example["action_reprs"]
        gold_idx = example["target_action_index"]
        gold_repr = example["target_action_reprs"]

        try:
            messages = build_messages(example, args.max_html_chars)
            inputs   = apply_chat_template(processor, model, messages)

            if use_scoring and len(cands) <= args.scoring_threshold:
                pred_idx, top3, scores = score_candidates(
                    model, processor, inputs, len(cands),
                    args.scoring_max_new_tokens,
                )
                raw_output = json.dumps(
                    {"scores": {str(ci): round(s, 4) for ci, s in enumerate(scores)}}
                )
            else:
                raw_output = generate_answer(model, processor, inputs,
                                             args.max_new_tokens)
                pred_idx   = parse_answer(raw_output, len(cands))
                top3       = [pred_idx]

        except Exception as e:
            pred_idx   = -1
            top3       = [-1]
            raw_output = f"ERROR: {e}"

        pred_repr  = cands[pred_idx] if 0 <= pred_idx < len(cands) else "INVALID"
        top3_reprs = [cands[j] if 0 <= j < len(cands) else "INVALID" for j in top3]

        predictions.append({
            # required by mind2web_metrics.evaluate()
            "candidate_actions":        cands,
            "gold_target_index":        gold_idx,
            "gold_target_action":       gold_repr,
            "predicted_index":          pred_idx,
            "top3_predicted_indices":   top3,
            "top3_predicted_actions":   top3_reprs,
            "task_id":                  example["annotation_id"],
            # extra context
            "example_index":            i,
            "action_uid":               example["action_uid"],
            "website":                  example["website"],
            "confirmed_task":           example["confirmed_task"],
            "predicted_action":         pred_repr,
            "raw_model_output":         raw_output[:2000],
        })

        correct = pred_idx == gold_idx
        pbar.set_postfix({"correct": correct, "pred": pred_idx, "gold": gold_idx})

    # ── Metrics ───────────────────────────────────────────────────────────────
    results = evaluate(predictions)
    print(f"\n{results}")

    # ── Save ──────────────────────────────────────────────────────────────────
    pred_path    = os.path.join(args.output_dir, f"predictions_{timestamp}.json")
    metrics_path = os.path.join(args.output_dir, f"metrics_{timestamp}.json")

    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    with open(metrics_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"\nPredictions : {pred_path}")
    print(f"Metrics     : {metrics_path}")
    return predictions, results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",    default=DEFAULT["model_id"])
    p.add_argument("--split",       default=DEFAULT["split"],
                   choices=["train", "test_website", "test_task", "test_domain"])
    p.add_argument("--n",           type=int, default=None,
                   help="Limit to first N examples (default: full split)")
    p.add_argument("--max_html_chars", type=int, default=DEFAULT["max_html_chars"])
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT["max_new_tokens"])
    p.add_argument("--scoring_max_new_tokens", type=int,
                   default=DEFAULT["scoring_max_new_tokens"])
    p.add_argument("--no-scoring",  dest="use_scoring", action="store_false",
                   help="Use generation mode instead of log-prob scoring")
    p.add_argument("--scoring_threshold", type=int, default=DEFAULT["scoring_threshold"],
                   help="Max candidates to still use scoring mode (default 50, set to 9999 to always score)")
    p.add_argument("--output_dir",  default=DEFAULT["output_dir"])
    p.set_defaults(use_scoring=DEFAULT["use_candidate_scoring"])

    run(p.parse_args())
