"""
End-to-end pipeline: download → AXTree → inference (2 baselines) → metrics.

Baselines
---------
  1. HTML  + Image  (cleaned_html as-is)
  2. AXTree + Image (accessibility tree generated from cleaned_html)

Both use the same model, candidate-scoring inference, and metric set.

Usage
-----
    python run_pipeline.py                  # full test_website split
    python run_pipeline.py --n 50           # quick smoke-test
    python run_pipeline.py --scoring_threshold 9999   # always score all candidates
"""

import argparse
import json
import os
import traceback
from datetime import datetime

import torch
from datasets import load_dataset
from tqdm import tqdm

# ── local modules ─────────────────────────────────────────────────────────────
from mind2web_downloader import html_to_axtree
from mind2web_metrics import evaluate, compare
from run_inference import (
    load_model,
    apply_chat_template,
    score_candidates,
    generate_answer,
    parse_answer,
)

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT = {
    "model_id":           "Qwen/Qwen3-VL-8B-Thinking",
    "dataset_id":         "osunlp/Multimodal-Mind2Web",
    "split":              "test_website",
    "max_html_chars":     12_000,
    "max_axtree_nodes":   300,
    "max_new_tokens":     512,
    "scoring_max_new_tokens": 10,
    "scoring_threshold":  50,
    "output_dir":         "outputs",
}

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_HTML = (
    "You are a web navigation agent. You are given a screenshot of a webpage, "
    "the cleaned HTML of the page, and a list of candidate actions. "
    "Your job is to select the single best action that accomplishes the user's task. "
    "Do NOT invent new actions. You MUST choose from the provided candidates only.\n\n"
    "End your response with exactly:\nAnswer: <index>\n"
    "where <index> is the zero-based integer index of your chosen action."
)

SYSTEM_AXTREE = (
    "You are a web navigation agent. You are given a screenshot of a webpage, "
    "an accessibility tree (AXTree) describing the page structure, "
    "and a list of candidate actions. "
    "Your job is to select the single best action that accomplishes the user's task. "
    "Do NOT invent new actions. You MUST choose from the provided candidates only.\n\n"
    "End your response with exactly:\nAnswer: <index>\n"
    "where <index> is the zero-based integer index of your chosen action."
)

# ── Dataset helpers ───────────────────────────────────────────────────────────

def _load_image(raw):
    from PIL import Image
    from io import BytesIO
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
        return html, False
    half = max_chars // 2
    return html[:half] + "\n... [TRUNCATED] ...\n" + html[-half:], True

def _candidate_str(candidates):
    return "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))

# ── Message builders ──────────────────────────────────────────────────────────

def build_html_messages(row, max_html_chars):
    html, truncated = _truncate_html(row["cleaned_html"], max_html_chars)
    trunc_note = f" (truncated)" if truncated else ""
    text = (
        f"Task: {row['confirmed_task']}\n\n"
        f"Cleaned HTML{trunc_note}:\n{html}\n\n"
        f"Candidate Actions:\n{_candidate_str(row['action_reprs'])}\n\n"
        f"Respond with:\nAnswer: <index>"
    )
    content = []
    img = _load_image(row["screenshot"])
    if img is not None:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_HTML}]},
        {"role": "user",   "content": content},
    ]


def build_axtree_messages(row, max_axtree_nodes):
    tree = html_to_axtree(row["cleaned_html"], max_nodes=max_axtree_nodes)
    text = (
        f"Task: {row['confirmed_task']}\n\n"
        f"Accessibility Tree:\n{tree}\n\n"
        f"Candidate Actions:\n{_candidate_str(row['action_reprs'])}\n\n"
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

# ── Core inference loop ───────────────────────────────────────────────────────

def run_baseline(
    name: str,
    dataset,
    message_builder,
    model,
    processor,
    total: int,
    scoring_threshold: int,
    scoring_max_new_tokens: int,
    max_new_tokens: int,
):
    print(f"\n{'='*60}")
    print(f"  Baseline: {name}")
    print(f"{'='*60}")

    predictions = []

    pbar = tqdm(range(total), desc=name, dynamic_ncols=True)
    for i in pbar:
        row       = dataset[i]
        cands     = row["action_reprs"]
        gold_idx  = int(row["target_action_index"])
        gold_repr = row["target_action_reprs"]

        try:
            messages = message_builder(row)
            inputs   = apply_chat_template(processor, model, messages)

            if len(cands) <= scoring_threshold:
                pred_idx, top3, scores = score_candidates(
                    model, processor, inputs, len(cands), scoring_max_new_tokens
                )
                raw_output = json.dumps(
                    {"scores": {str(ci): round(s, 4) for ci, s in enumerate(scores)}}
                )
            else:
                raw_output = generate_answer(model, processor, inputs, max_new_tokens)
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
            # fields required by mind2web_metrics.evaluate()
            "candidate_actions":        cands,
            "gold_target_index":        gold_idx,
            "gold_target_action":       gold_repr,
            "predicted_index":          pred_idx,
            "top3_predicted_indices":   top3,
            "top3_predicted_actions":   top3_reprs,
            "task_id":                  row["annotation_id"],
            # extra context
            "example_index":            i,
            "action_uid":               row["action_uid"],
            "website":                  row.get("website", ""),
            "confirmed_task":           row["confirmed_task"],
            "predicted_action":         pred_repr,
            "raw_model_output":         raw_output[:2000],
        })

        pbar.set_postfix({
            "✓" if pred_idx == gold_idx else "✗": f"{pred_idx}/{gold_idx}",
        })

    results = evaluate(predictions)
    return predictions, results

# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 1. Download dataset ───────────────────────────────────────────────────
    print(f"[1/4] Downloading dataset: {args.dataset_id}  split={args.split}")
    dataset = load_dataset(args.dataset_id, split=args.split)
    total   = len(dataset) if args.n is None else min(args.n, len(dataset))
    print(f"      {total} examples to evaluate")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print(f"\n[2/4] Loading model: {args.model_id}")
    model, processor = load_model(args.model_id)

    # ── 3. Run both baselines ─────────────────────────────────────────────────
    print("\n[3/4] Running baselines")

    html_preds, html_results = run_baseline(
        name                   = "HTML + Image",
        dataset                = dataset,
        message_builder        = lambda row: build_html_messages(row, args.max_html_chars),
        model                  = model,
        processor              = processor,
        total                  = total,
        scoring_threshold      = args.scoring_threshold,
        scoring_max_new_tokens = args.scoring_max_new_tokens,
        max_new_tokens         = args.max_new_tokens,
    )

    axtree_preds, axtree_results = run_baseline(
        name                   = "AXTree + Image",
        dataset                = dataset,
        message_builder        = lambda row: build_axtree_messages(row, args.max_axtree_nodes),
        model                  = model,
        processor              = processor,
        total                  = total,
        scoring_threshold      = args.scoring_threshold,
        scoring_max_new_tokens = args.scoring_max_new_tokens,
        max_new_tokens         = args.max_new_tokens,
    )

    # ── 4. Metrics & save ─────────────────────────────────────────────────────
    print("\n[4/4] Results\n")
    print(compare({"HTML + Image": html_results, "AXTree + Image": axtree_results}))

    def _save(name_slug, preds, results):
        pred_path    = os.path.join(args.output_dir, f"{name_slug}_predictions_{timestamp}.json")
        metrics_path = os.path.join(args.output_dir, f"{name_slug}_metrics_{timestamp}.json")
        with open(pred_path, "w") as f:
            json.dump(preds, f, indent=2, default=str)
        with open(metrics_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"  {name_slug:12s}  predictions → {pred_path}")
        print(f"  {name_slug:12s}  metrics     → {metrics_path}")

    _save("html_image",   html_preds,   html_results)
    _save("axtree_image", axtree_preds, axtree_results)

    summary_path = os.path.join(args.output_dir, f"summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump({
            "html_image":   html_results.to_dict(),
            "axtree_image": axtree_results.to_dict(),
        }, f, indent=2)
    print(f"\n  summary → {summary_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",    default=DEFAULT["model_id"])
    p.add_argument("--dataset_id",  default=DEFAULT["dataset_id"])
    p.add_argument("--split",       default=DEFAULT["split"],
                   choices=["train", "test_website", "test_task", "test_domain"])
    p.add_argument("--n",           type=int, default=None,
                   help="Limit to first N examples (omit for full split)")
    p.add_argument("--max_html_chars",     type=int, default=DEFAULT["max_html_chars"])
    p.add_argument("--max_axtree_nodes",   type=int, default=DEFAULT["max_axtree_nodes"])
    p.add_argument("--max_new_tokens",     type=int, default=DEFAULT["max_new_tokens"])
    p.add_argument("--scoring_max_new_tokens", type=int,
                   default=DEFAULT["scoring_max_new_tokens"])
    p.add_argument("--scoring_threshold",  type=int, default=DEFAULT["scoring_threshold"],
                   help="Max candidates for scoring mode (9999 = always score)")
    p.add_argument("--output_dir",  default=DEFAULT["output_dir"])
    main(p.parse_args())
