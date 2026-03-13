"""
Inference on Multimodal Mind2Web using Qwen3-VL-8B-Thinking.

Modes
-----
  default        HTML + Image
  --use_axtree   AXTree + Image   (run generate_axtrees.py first)
  --cot          HTML + Image + Chain-of-Thought (enable_thinking)

Usage
-----
    python run_inference.py --load_in_4bit --n 50
    python run_inference.py --use_axtree --load_in_4bit --n 50
    python run_inference.py --cot --load_in_4bit --n 50
"""

import argparse
import json
import os
import re
import traceback
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from mind2web_dataset import get_dataloader

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT = {
    "model_id":           "Qwen/Qwen3-VL-8B-Thinking",
    "dataset_id":         "osunlp/Multimodal-Mind2Web",
    "split":              "test_website",
    "axtree_dir":         "axtrees",
    "max_html_chars":     8_000,
    "max_new_tokens":     32,    # just "Answer: N" — no need for more
    "cot_max_new_tokens": 1024,  # CoT needs room to think
    "output_dir":         "outputs",
}

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_HTML = (
    "You are a web navigation agent. Given a webpage screenshot, its HTML, "
    "and a numbered list of candidate actions, select the single best action "
    "that accomplishes the task. Do NOT invent actions outside the list.\n\n"
    "End your response with exactly:\nAnswer: <index>\n"
    "where <index> is the zero-based integer index of your chosen action."
)

SYSTEM_AXTREE = (
    "You are a web navigation agent. Given a webpage screenshot, its accessibility "
    "tree (AXTree), and a numbered list of candidate actions, select the single best "
    "action that accomplishes the task. Do NOT invent actions outside the list.\n\n"
    "End your response with exactly:\nAnswer: <index>\n"
    "where <index> is the zero-based integer index of your chosen action."
)

SYSTEM_COT = (
    "You are a web navigation agent. Given a webpage screenshot, its HTML, "
    "and a numbered list of candidate actions:\n"
    "1. Identify the user's goal.\n"
    "2. Locate relevant elements in the HTML and screenshot.\n"
    "3. Evaluate each candidate against the goal.\n"
    "4. Pick the best one.\n\n"
    "Do NOT invent actions outside the list.\n\n"
    "End your response with exactly:\nAnswer: <index>\n"
    "where <index> is the zero-based integer index of your chosen action."
)

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, load_in_4bit: bool = False):
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs = dict(quantization_config=quant_cfg, device_map="auto", low_cpu_mem_usage=True)
    else:
        kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, attn_implementation="flash_attention_2", **kwargs
        )
        print("Loaded with flash_attention_2")
    except Exception:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, attn_implementation="sdpa", **kwargs
        )
        print("Loaded with sdpa")

    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

# ── Inference ─────────────────────────────────────────────────────────────────

def _cand_str(candidates):
    return "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))


def build_messages(task, context, context_type, candidates, screenshot, cot):
    if cot:
        system, label = SYSTEM_COT, "Cleaned HTML"
    elif context_type == "axtree":
        system, label = SYSTEM_AXTREE, "Accessibility Tree"
    else:
        system, label = SYSTEM_HTML, "Cleaned HTML"

    text = (
        f"Task: {task}\n\n"
        f"{label}:\n{context}\n\n"
        f"Candidate Actions:\n{_cand_str(candidates)}\n\n"
        f"Respond with:\nAnswer: <index>"
    )
    content = []
    if screenshot is not None:
        content.append({"type": "image", "image": screenshot})
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user",   "content": content},
    ]


def generate(model, processor, messages, max_new_tokens, enable_thinking):
    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            enable_thinking=enable_thinking,
        )
    except TypeError:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
    inputs    = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    full = processor.tokenizer.decode(out[0][input_len:], skip_special_tokens=False)

    # Split thinking from answer
    think_m  = re.search(r"<think>(.*?)</think>", full, re.DOTALL)
    thinking = think_m.group(1).strip() if think_m else ""
    answer   = re.sub(r"<think>.*?</think>", "", full, flags=re.DOTALL)
    answer   = re.sub(r"<[^>]+>", "", answer).strip()
    return answer, thinking


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

# ── Main loop ─────────────────────────────────────────────────────────────────

def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "cot" if args.cot else ("axtree" if args.use_axtree else "html")
    print(f"Mode: {mode} | Model: {args.model_id} | 4bit: {args.load_in_4bit}")

    loader = get_dataloader(
        split=args.split,
        dataset_id=args.dataset_id,
        max_html_chars=args.max_html_chars,
        use_axtree=args.use_axtree,
        axtree_dir=args.axtree_dir,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    total = min(args.n, len(loader.dataset)) if args.n else len(loader.dataset)
    print(f"Dataset: {total} examples")

    model, processor = load_model(args.model_id, load_in_4bit=args.load_in_4bit)

    predictions = []
    pbar = tqdm(loader, total=total, desc=f"[{mode}]", dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        if args.n and step >= args.n:
            break

        task       = batch["confirmed_task"][0]
        context    = batch["page_context"][0]
        ctx_type   = batch["context_type"][0]
        cands      = batch["action_reprs"][0]
        gold_idx   = batch["target_action_index"].item()
        gold_repr  = batch["target_action_reprs"][0]
        screenshot = batch["screenshot"][0]
        ann_id     = batch["annotation_id"][0]
        act_uid    = batch["action_uid"][0]
        website    = batch["website"][0]

        try:
            messages             = build_messages(task, context, ctx_type, cands,
                                                  screenshot, cot=args.cot)
            max_tok              = args.cot_max_new_tokens if args.cot else args.max_new_tokens
            answer_text, thinking = generate(model, processor, messages,
                                             max_tok, enable_thinking=args.cot)
            pred_idx             = parse_answer(answer_text, len(cands))

        except Exception as e:
            traceback.print_exc()
            pred_idx     = -1
            answer_text  = f"ERROR: {e}"
            thinking     = ""

        pred_repr = cands[pred_idx] if 0 <= pred_idx < len(cands) else "INVALID"

        predictions.append({
            "example_index":          step,
            "annotation_id":          ann_id,
            "task_id":                ann_id,
            "action_uid":             act_uid,
            "website":                website,
            "confirmed_task":         task,
            "context_type":           ctx_type,
            "candidate_actions":      cands,
            "gold_target_index":      gold_idx,
            "gold_target_action":     gold_repr,
            "predicted_index":        pred_idx,
            "predicted_action":       pred_repr,
            "top3_predicted_indices": [pred_idx],
            "top3_predicted_actions": [pred_repr],
            "thinking":               thinking[:3000],
            "raw_model_output":       answer_text[:2000],
        })

        pbar.set_postfix({"pred": pred_idx, "gold": gold_idx,
                          "ok": pred_idx == gold_idx})

    pred_path = os.path.join(args.output_dir, f"{mode}_predictions_{timestamp}.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)

    print(f"\nSaved {len(predictions)} predictions → {pred_path}")
    print(f"Evaluate: python evaluate_predictions.py {pred_path}")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",       default=DEFAULT["model_id"])
    p.add_argument("--dataset_id",     default=DEFAULT["dataset_id"])
    p.add_argument("--split",          default=DEFAULT["split"],
                   choices=["train", "test_website", "test_task", "test_domain"])
    p.add_argument("--n",              type=int, default=None)
    p.add_argument("--use_axtree",     action="store_true")
    p.add_argument("--axtree_dir",     default=DEFAULT["axtree_dir"])
    p.add_argument("--cot",            action="store_true",
                   help="Enable chain-of-thought (Qwen3 thinking mode)")
    p.add_argument("--max_html_chars", type=int, default=DEFAULT["max_html_chars"])
    p.add_argument("--max_new_tokens",     type=int, default=DEFAULT["max_new_tokens"])
    p.add_argument("--cot_max_new_tokens", type=int, default=DEFAULT["cot_max_new_tokens"])
    p.add_argument("--load_in_4bit",   action="store_true",
                   help="4-bit quantization (requires bitsandbytes)")
    p.add_argument("--output_dir",     default=DEFAULT["output_dir"])
    run(p.parse_args())
