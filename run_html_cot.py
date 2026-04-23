"""
Ablation: HTML + Image + Chain-of-Thought
Passes cleaned_html + screenshot to Qwen3-VL-8B-Thinking with thinking
enabled (enable_thinking=True), letting the model reason before answering.
Uses free-form generation (not candidate scoring) so the full CoT is captured.

Usage
-----
    python run_html_cot.py
    python run_html_cot.py --n 50
    python run_html_cot.py --max_new_tokens 1024
"""

import argparse
import json
import os
import re
import traceback
from datetime import datetime
from io import BytesIO

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from html_pruning import prune_html_dom
from mind2web_metrics import evaluate
from run_inference import load_model, parse_answer

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT = {
    "model_id":       "Qwen/Qwen3-VL-8B-Thinking",
    "dataset_id":     "osunlp/Multimodal-Mind2Web",
    "split":          "test_website",
    "max_html_chars": 12_000,
    "max_new_tokens": 1024,   # higher budget for CoT reasoning
    "output_dir":     "outputs",
}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_HTML_COT = (
    "You are a web navigation agent. You are given a screenshot of a webpage, "
    "the cleaned HTML of the page, and a numbered list of candidate actions.\n\n"
    "Think step by step:\n"
    "1. Identify the user's goal from the task description.\n"
    "2. Scan the HTML and screenshot to locate relevant elements.\n"
    "3. Evaluate each candidate action against the goal.\n"
    "4. Select the single best action.\n\n"
    "Do NOT invent new actions. You MUST choose from the provided candidates only.\n\n"
    "End your response with exactly:\nAnswer: <index>\n"
    "where <index> is the zero-based integer index of your chosen action."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_image(raw):
    if raw is None:
        return None
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict) and raw.get("bytes"):
        return Image.open(BytesIO(raw["bytes"])).convert("RGB")
    if isinstance(raw, dict) and raw.get("path"):
        return Image.open(raw["path"]).convert("RGB")
    return None


def _truncate_html(html: str, max_chars: int):
    if len(html) <= max_chars:
        return html
    half = max_chars // 2
    return html[:half] + "\n... [TRUNCATED] ...\n" + html[-half:]


def build_messages(row, max_html_chars, prune_html=False):
    html = str(row["cleaned_html"] or "")
    if prune_html:
        html = prune_html_dom(html)
    html     = _truncate_html(html, max_html_chars)
    cand_str = "\n".join(f"[{i}] {c}" for i, c in enumerate(row["action_reprs"]))
    text = (
        f"Task: {row['confirmed_task']}\n\n"
        f"Cleaned HTML:\n{html}\n\n"
        f"Candidate Actions:\n{cand_str}\n\n"
        f"Think through this carefully, then respond with:\nAnswer: <index>"
    )
    content = []
    img = _load_image(row["screenshot"])
    if img is not None:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_HTML_COT}]},
        {"role": "user",   "content": content},
    ]


def apply_chat_template_cot(processor, model, messages):
    """Apply chat template with thinking enabled for CoT."""
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=True,   # ← enables <think>...</think> CoT
        )
    except TypeError:
        # older processor versions don't support enable_thinking
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    return {k: v.to(model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()}


def generate_with_thinking(model, processor, inputs, max_new_tokens):
    """Generate and split out the <think> block from the final answer."""
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    out_ids = generated[0][inputs["input_ids"].shape[1]:]
    full_text = processor.tokenizer.decode(out_ids, skip_special_tokens=False)

    # Extract thinking and answer separately
    think_match = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL)
    thinking    = think_match.group(1).strip() if think_match else ""
    answer_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
    answer_text = re.sub(r"<[^>]+>", "", answer_text).strip()   # remove any remaining tags

    return full_text, thinking, answer_text


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Loading dataset: {args.dataset_id}  split={args.split}")
    dataset = load_dataset(args.dataset_id, split=args.split)
    total = len(dataset) if args.n is None else min(args.n, len(dataset))
    print(f"{total} examples to evaluate")

    print(f"\nLoading model: {args.model_id}")
    model, processor = load_model(args.model_id)

    predictions = []
    pbar = tqdm(range(total), desc="HTML+Image+CoT", dynamic_ncols=True)
    for i in pbar:
        row       = dataset[i]
        cands     = row["action_reprs"]
        gold_idx  = int(row["target_action_index"])
        gold_repr = row["target_action_reprs"]

        try:
            messages = build_messages(row, args.max_html_chars, prune_html=args.prune_html)
            inputs   = apply_chat_template_cot(processor, model, messages)
            full_text, thinking, answer_text = generate_with_thinking(
                model, processor, inputs, args.max_new_tokens
            )
            pred_idx = parse_answer(answer_text, len(cands))

        except Exception as e:
            traceback.print_exc()
            pred_idx    = -1
            thinking    = ""
            answer_text = f"ERROR: {e}"
            full_text   = answer_text

        pred_repr = cands[pred_idx] if 0 <= pred_idx < len(cands) else "INVALID"

        predictions.append({
            "candidate_actions":      cands,
            "gold_target_index":      gold_idx,
            "gold_target_action":     gold_repr,
            "predicted_index":        pred_idx,
            "top3_predicted_indices": [pred_idx],
            "top3_predicted_actions": [pred_repr],
            "task_id":                row["annotation_id"],
            "example_index":          i,
            "action_uid":             row["action_uid"],
            "website":                row.get("website", ""),
            "confirmed_task":         row["confirmed_task"],
            "predicted_action":       pred_repr,
            "thinking":               thinking[:3000],   # stored for analysis
            "raw_model_output":       answer_text[:2000],
        })

        pbar.set_postfix({"pred": pred_idx, "gold": gold_idx,
                          "✓" if pred_idx == gold_idx else "✗": ""})

    results = evaluate(predictions)
    print(f"\n{results}")

    pred_path    = os.path.join(args.output_dir, f"html_cot_predictions_{timestamp}.json")
    metrics_path = os.path.join(args.output_dir, f"html_cot_metrics_{timestamp}.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    with open(metrics_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"\nPredictions : {pred_path}")
    print(f"Metrics     : {metrics_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",       default=DEFAULT["model_id"])
    p.add_argument("--dataset_id",     default=DEFAULT["dataset_id"])
    p.add_argument("--split",          default=DEFAULT["split"],
                   choices=["train", "test_website", "test_task", "test_domain"])
    p.add_argument("--n",              type=int, default=None)
    p.add_argument("--max_html_chars", type=int, default=DEFAULT["max_html_chars"])
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT["max_new_tokens"])
    p.add_argument("--prune_html",     action="store_true")
    p.add_argument("--output_dir",     default=DEFAULT["output_dir"])
    main(p.parse_args())
