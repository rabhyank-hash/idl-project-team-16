"""
Mind2Web evaluation metrics.

Metrics
-------
- Element Accuracy  (Ele. Acc)  : correct element selected
- Operation F1      (Op. F1)    : char-level F1 on action type + value
- Step Success Rate (Step SR)   : element correct AND Op. F1 >= 0.9
- Task Success Rate (Task SR)   : all steps in a task pass Step SR
- Top-3 Accuracy    (Top3 Acc)  : gold index in top-3 predictions

Usage
-----
    from mind2web_metrics import evaluate, step_metrics

    # Single step
    m = step_metrics(pred_idx=2, gold_idx=2,
                     pred_repr="-> CLICK Submit",
                     gold_repr="-> CLICK Submit",
                     top3=[2, 0, 1])

    # Full dataset (list of prediction dicts)
    results = evaluate(predictions)
    print(results)
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional


# ── Character-level F1 ───────────────────────────────────────────────────────

def char_f1(pred: str, gold: str) -> float:
    """Character-level F1 between two strings."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    gold_remaining = list(gold)
    common = 0
    for c in pred:
        if c in gold_remaining:
            common += 1
            gold_remaining.remove(c)
    precision = common / len(pred)
    recall    = common / len(gold)
    denom = precision + recall
    return (2 * precision * recall / denom) if denom > 0 else 0.0


# ── Action repr parsing ───────────────────────────────────────────────────────

_OP_PATTERN = re.compile(r"->\s*(CLICK|TYPE|SELECT)\s*(.*)", re.IGNORECASE)


def parse_action_repr(action_repr: str):
    """Return (op_type, value) from an action_repr string."""
    m = _OP_PATTERN.search(action_repr)
    if m:
        return m.group(1).strip().upper(), m.group(2).strip()
    return action_repr.strip(), ""


# ── Per-step metrics ─────────────────────────────────────────────────────────

@dataclass
class StepResult:
    ele_acc:  int    # 1 if correct element selected, else 0
    op_f1:    float  # character-level F1 on op type + value
    step_sr:  int    # 1 if ele_acc==1 and op_f1 >= 0.9, else 0
    top3_acc: int    # 1 if gold index in top-3 predictions, else 0


def step_metrics(
    pred_idx:  int,
    gold_idx:  int,
    pred_repr: str,
    gold_repr: str,
    top3:      Optional[List[int]] = None,
    op_f1_threshold: float = 0.9,
) -> StepResult:
    """
    Compute all per-step metrics.

    Parameters
    ----------
    pred_idx  : predicted candidate index
    gold_idx  : ground-truth candidate index
    pred_repr : predicted action representation string
    gold_repr : ground-truth action representation string
    top3      : list of top-3 predicted indices (defaults to [pred_idx])
    op_f1_threshold : minimum Op. F1 required for Step SR (default 0.9)
    """
    if top3 is None:
        top3 = [pred_idx]

    ele_acc = int(pred_idx == gold_idx)

    pred_op, pred_val = parse_action_repr(pred_repr)
    gold_op, gold_val = parse_action_repr(gold_repr)
    op_f1 = char_f1(
        f"{pred_op} {pred_val}".strip(),
        f"{gold_op} {gold_val}".strip(),
    )

    step_sr  = int(ele_acc == 1 and op_f1 >= op_f1_threshold)
    top3_acc = int(gold_idx in top3)

    return StepResult(ele_acc=ele_acc, op_f1=op_f1,
                      step_sr=step_sr, top3_acc=top3_acc)


# ── Aggregate evaluation ─────────────────────────────────────────────────────

@dataclass
class EvalResults:
    ele_acc:   float  # Element Accuracy
    op_f1:     float  # Operation F1
    step_sr:   float  # Step Success Rate  ← primary metric
    task_sr:   float  # Task Success Rate
    top3_acc:  float  # Top-3 Accuracy
    n_steps:   int
    n_tasks:   int
    # Per-step breakdown stored back into prediction dicts (side-effect)
    per_step:  List[StepResult] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"{'Metric':<28} {'Value':>8}",
            "-" * 38,
            f"{'Element Accuracy':<28} {self.ele_acc:>8.4f}",
            f"{'Operation F1':<28} {self.op_f1:>8.4f}",
            f"{'Step Success Rate':<28} {self.step_sr:>8.4f}",
            f"{'Task Success Rate':<28} {self.task_sr:>8.4f}",
            f"{'Top-3 Accuracy':<28} {self.top3_acc:>8.4f}",
            "-" * 38,
            f"{'Steps evaluated':<28} {self.n_steps:>8}",
            f"{'Tasks evaluated':<28} {self.n_tasks:>8}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "element_accuracy":  self.ele_acc,
            "operation_f1":      self.op_f1,
            "step_success_rate": self.step_sr,
            "task_success_rate": self.task_sr,
            "top3_accuracy":     self.top3_acc,
            "n_steps":           self.n_steps,
            "n_tasks":           self.n_tasks,
        }


def evaluate(
    predictions: List[dict],
    op_f1_threshold: float = 0.9,
) -> EvalResults:
    """
    Aggregate metrics over a list of prediction dicts.

    Each dict must contain:
        candidate_actions       : List[str]
        gold_target_index       : int
        gold_target_action      : str
        predicted_index         : int
        top3_predicted_indices  : List[int]  (optional; falls back to [predicted_index])
        task_id                 : str        (optional; used for Task SR)

    Metric values are also written back into each dict as:
        metric_ele_acc, metric_op_f1, metric_step_sr, metric_top3_acc
    """
    if not predictions:
        return EvalResults(0, 0, 0, 0, 0, 0, 0)

    sum_ele = sum_f1 = sum_sr = sum_top3 = 0
    task_steps: dict[str, list] = defaultdict(list)
    per_step: List[StepResult] = []

    for p in predictions:
        candidates = p["candidate_actions"]
        gold_idx   = p["gold_target_index"]
        pred_idx   = p["predicted_index"]
        gold_repr  = p["gold_target_action"]
        pred_repr  = (candidates[pred_idx]
                      if 0 <= pred_idx < len(candidates) else "")
        top3       = p.get("top3_predicted_indices", [pred_idx])

        sr = step_metrics(pred_idx, gold_idx, pred_repr, gold_repr,
                          top3=top3, op_f1_threshold=op_f1_threshold)

        sum_ele  += sr.ele_acc
        sum_f1   += sr.op_f1
        sum_sr   += sr.step_sr
        sum_top3 += sr.top3_acc

        # Write back into the dict for easy downstream inspection
        p["metric_ele_acc"]  = sr.ele_acc
        p["metric_op_f1"]    = round(sr.op_f1, 4)
        p["metric_step_sr"]  = sr.step_sr
        p["metric_top3_acc"] = sr.top3_acc

        task_id = p.get("task_id", "unknown")
        task_steps[task_id].append(sr.step_sr)
        per_step.append(sr)

    n = len(predictions)
    n_tasks = len(task_steps)
    task_sr = sum(
        int(all(s == 1 for s in steps))
        for steps in task_steps.values()
    )

    return EvalResults(
        ele_acc  = sum_ele  / n,
        op_f1    = sum_f1   / n,
        step_sr  = sum_sr   / n,
        task_sr  = task_sr  / n_tasks if n_tasks else 0.0,
        top3_acc = sum_top3 / n,
        n_steps  = n,
        n_tasks  = n_tasks,
        per_step = per_step,
    )


# ── Compare multiple ablations ───────────────────────────────────────────────

def compare(ablations: dict[str, EvalResults]) -> str:
    """
    Pretty-print a comparison table.

    Parameters
    ----------
    ablations : {name: EvalResults}

    Example
    -------
        print(compare({"Zero-Shot": r1, "Few-Shot": r2, "CoT": r3}))
    """
    keys   = list(ablations.keys())
    col_w  = max(18, *(len(k) + 2 for k in keys))
    header = f"{'Metric':<28}" + "".join(f"{k:>{col_w}}" for k in keys)
    sep    = "-" * len(header)

    ROWS = [
        ("Element Accuracy",  "ele_acc"),
        ("Operation F1",      "op_f1"),
        ("Step Success Rate", "step_sr"),
        ("Task Success Rate", "task_sr"),
        ("Top-3 Accuracy",    "top3_acc"),
    ]

    lines = [header, sep]
    for label, attr in ROWS:
        row = f"{label:<28}"
        for r in ablations.values():
            row += f"{getattr(r, attr):>{col_w}.4f}"
        lines.append(row)

    lines.append(sep)
    for label, attr in [("Steps", "n_steps"), ("Tasks", "n_tasks")]:
        row = f"{label:<28}"
        for r in ablations.values():
            row += f"{getattr(r, attr):>{col_w}}"
        lines.append(row)

    return "\n".join(lines)
