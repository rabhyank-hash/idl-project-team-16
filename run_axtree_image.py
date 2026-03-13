"""
Ablation: AXTree + Image
Converts cleaned_html → AXTree, passes it with the screenshot to
Qwen3-VL-8B-Thinking, uses KV-cached candidate scoring.

Usage
-----
    python run_axtree_image.py
    python run_axtree_image.py --n 50
    python run_axtree_image.py --scoring_threshold 9999
"""

import argparse
import json
import os
import traceback
from datetime import datetime
from io import BytesIO

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from mind2web_downloader import html_to_axtree
from mind2web_metrics import evaluate
from run_inference import (
    load_model,
    apply_chat_template,
    score_candidates,
    generate_answer,
    parse_answer,
)

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT = {
    "model_id":               "Qwen/Qwen3-VL-8B-Thinking",
    "dataset_id":             "osunlp/Multimodal-Mind2Web",
    "split":                  "test_website",
    "max_axtree_nodes":       300,
    "max_new_tokens":         512,
    "scoring_max_new_tokens": 10,
    "scoring_threshold":      50,
    "output_dir":             "outputs",
}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_AXTREE = (
    "You are a web navigation agent. You are given a screenshot of a webpage "
    "and an accessibility tree (AXTree) that describes the interactive elements "
    "and structure of the page. You are also given a numbered list of candidate actions.\n\n"
    "Use the AXTree to understand which elements are on the page and how they relate "
    "to the user's task. Then select the single best candidate action.\n\n"
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


def build_messages(row, max_axtree_nodes):
    tree = html_to_axtree(row["cleaned_html"], max_nodes=max_axtree_nodes)
    cand_str = "\n".join(f"[{i}] {c}" for i, c in enumerate(row["action_reprs"]))
    text = (
        f"Task: {row['confirmed_task']}\n\n"
        f"Accessibility Tree:\n{tree}\n\n"
        f"Candidate Actions:\n{cand_str}\n\n"
        f"Respond with:\nAnswer: <index>"
    )
    content = []
    img = _load_image(row["screenshot"])
    if img is not None:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_AXTREE}]},
        {"role": "user",   "content": content},
    ]


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
    pbar = tqdm(range(total), desc="AXTree+Image", dynamic_ncols=True)
    for i in pbar:
        row      = dataset[i]
        cands    = row["action_reprs"]
        gold_idx = int(row["target_action_index"])
        gold_repr = row["target_action_reprs"]

        try:
            messages = build_messages(row, args.max_axtree_nodes)
            inputs   = apply_chat_template(processor, model, messages)

            if len(cands) <= args.scoring_threshold:
                pred_idx, top3, scores = score_candidates(
                    model, processor, inputs, len(cands), args.scoring_max_new_tokens
                )
                raw_output = json.dumps(
                    {"scores": {str(ci): round(s, 4) for ci, s in enumerate(scores)}}
                )
            else:
                raw_output = generate_answer(model, processor, inputs, args.max_new_tokens)
                pred_idx   = parse_answer(raw_output, len(cands))
                top3       = [pred_idx]
                scores     = []

        except Exception as e:
            traceback.print_exc()
            pred_idx   = -1
            top3       = [-1]
            raw_output = f"ERROR: {e}"
            scores     = []

        pred_repr  = cands[pred_idx] if 0 <= pred_idx < len(cands) else "INVALID"
        top3_reprs = [cands[j] if 0 <= j < len(cands) else "INVALID" for j in top3]

        predictions.append({
            "candidate_actions":       cands,
            "gold_target_index":       gold_idx,
            "gold_target_action":      gold_repr,
            "predicted_index":         pred_idx,
            "top3_predicted_indices":  top3,
            "top3_predicted_actions":  top3_reprs,
            "task_id":                 row["annotation_id"],
            "example_index":           i,
            "action_uid":              row["action_uid"],
            "website":                 row.get("website", ""),
            "confirmed_task":          row["confirmed_task"],
            "predicted_action":        pred_repr,
            "raw_model_output":        raw_output[:2000],
        })

        pbar.set_postfix({"pred": pred_idx, "gold": gold_idx,
                          "✓" if pred_idx == gold_idx else "✗": ""})

    results = evaluate(predictions)
    print(f"\n{results}")

    pred_path    = os.path.join(args.output_dir, f"axtree_image_predictions_{timestamp}.json")
    metrics_path = os.path.join(args.output_dir, f"axtree_image_metrics_{timestamp}.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    with open(metrics_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"\nPredictions : {pred_path}")
    print(f"Metrics     : {metrics_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",               default=DEFAULT["model_id"])
    p.add_argument("--dataset_id",             default=DEFAULT["dataset_id"])
    p.add_argument("--split",                  default=DEFAULT["split"],
                   choices=["train", "test_website", "test_task", "test_domain"])
    p.add_argument("--n",                      type=int, default=None)
    p.add_argument("--max_axtree_nodes",       type=int, default=DEFAULT["max_axtree_nodes"])
    p.add_argument("--max_new_tokens",         type=int, default=DEFAULT["max_new_tokens"])
    p.add_argument("--scoring_max_new_tokens", type=int, default=DEFAULT["scoring_max_new_tokens"])
    p.add_argument("--scoring_threshold",      type=int, default=DEFAULT["scoring_threshold"])
    p.add_argument("--output_dir",             default=DEFAULT["output_dir"])
    main(p.parse_args())
