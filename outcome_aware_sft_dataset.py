"""
Dataset and collator utilities for outcome-aware SFT on Multimodal Mind2Web.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from html_pruning import prune_html_dom

DEFAULT_SYSTEM_PROMPT = (
    "You are a web navigation agent. You are given a webpage screenshot, pruned HTML, "
    "the user's task, and a numbered list of candidate actions. Select the single best "
    "candidate action and predict the immediate outcome after taking it. Do NOT invent "
    "new actions.\n\n"
    "Respond with exactly:\n"
    "Answer: <index>\n"
    'Outcome: {"transition_type":"...","changed_region":"...","change_magnitude":"...",'
    '"confidence":"..."}\n\n'
    'Use only the provided candidate indices. If nothing meaningful changes, use "none" '
    'for transition_type, changed_region, and change_magnitude.'
)


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n... [PRUNED HTML TRUNCATED] ...\n" + text[-tail:]


def _load_image(raw: Any) -> Image.Image | None:
    if raw is None:
        return None
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, bytes):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(raw, dict):
        if raw.get("bytes"):
            return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        if raw.get("path"):
            return Image.open(raw["path"]).convert("RGB")
    return None


def _parse_action_reprs(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        return [raw]
    return []


def build_candidate_block(candidates: list[str]) -> str:
    return "\n".join(f"[{idx}] {candidate}" for idx, candidate in enumerate(candidates))


def build_user_prompt(task: str, pruned_html: str, candidates: list[str]) -> str:
    return (
        f"Task: {task}\n\n"
        f"Pruned HTML:\n{pruned_html}\n\n"
        f"Candidate Actions:\n{build_candidate_block(candidates)}\n\n"
        "Choose the best candidate action and predict the immediate webpage change caused "
        "by taking that action.\n\n"
        "Return exactly:\n"
        "Answer: <index>\n"
        'Outcome: {"transition_type":"...","changed_region":"...","change_magnitude":"...",'
        '"confidence":"..."}'
    )


def analyze_outcome_targets(outcome_path: str | Path) -> dict[str, Any]:
    outcome_path = Path(outcome_path)
    counts = {
        "total_rows": 0,
        "rows_with_outcome": 0,
        "target_chars_min": None,
        "target_chars_max": 0,
        "target_chars_avg": 0.0,
        "target_words_min": None,
        "target_words_max": 0,
        "target_words_avg": 0.0,
        "outcome_json_chars_min": None,
        "outcome_json_chars_max": 0,
        "outcome_json_chars_avg": 0.0,
        "suggested_max_new_tokens": 96,
    }
    char_total = 0
    word_total = 0
    outcome_char_total = 0

    with outcome_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            target = str(row.get("target") or "")
            outcome_start = target.find("Outcome:")
            outcome_json = target[outcome_start + len("Outcome:") :].strip() if outcome_start >= 0 else ""

            target_chars = len(target)
            target_words = len(target.split())
            counts["total_rows"] += 1
            char_total += target_chars
            word_total += target_words
            counts["target_chars_max"] = max(counts["target_chars_max"], target_chars)
            counts["target_words_max"] = max(counts["target_words_max"], target_words)
            counts["target_chars_min"] = (
                target_chars if counts["target_chars_min"] is None else min(counts["target_chars_min"], target_chars)
            )
            counts["target_words_min"] = (
                target_words if counts["target_words_min"] is None else min(counts["target_words_min"], target_words)
            )

            if outcome_json:
                outcome_chars = len(outcome_json)
                counts["rows_with_outcome"] += 1
                outcome_char_total += outcome_chars
                counts["outcome_json_chars_max"] = max(counts["outcome_json_chars_max"], outcome_chars)
                counts["outcome_json_chars_min"] = (
                    outcome_chars
                    if counts["outcome_json_chars_min"] is None
                    else min(counts["outcome_json_chars_min"], outcome_chars)
                )

    if counts["total_rows"]:
        counts["target_chars_avg"] = round(char_total / counts["total_rows"], 2)
        counts["target_words_avg"] = round(word_total / counts["total_rows"], 2)
    if counts["rows_with_outcome"]:
        counts["outcome_json_chars_avg"] = round(outcome_char_total / counts["rows_with_outcome"], 2)
    return counts


def _load_outcome_rows(outcome_path: str | Path) -> dict[int, dict[str, Any]]:
    path = Path(outcome_path)
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            dataset_index = int(row["dataset_index"])
            if dataset_index in rows:
                raise ValueError(f"Duplicate dataset_index={dataset_index} in {path}")
            rows[dataset_index] = row
    return rows


def _load_pruned_html_rows(pruned_html_path: str | Path | None) -> dict[int, dict[str, Any]]:
    if not pruned_html_path:
        return {}

    path = Path(pruned_html_path)
    if not path.exists():
        return {}

    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            dataset_index = int(row["index"])
            rows[dataset_index] = row
    return rows


class OutcomeAwareMind2WebSFTDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        dataset_id: str = "osunlp/Multimodal-Mind2Web",
        data_cache_dir: str | Path | None = None,
        pruned_html_path: str | Path | None = None,
        outcome_path: str | Path = "sft_data/sft_data/outcome_aware_sft.jsonl",
        max_html_chars: int = 8_000,
        hf_dataset=None,
    ) -> None:
        self.split = split
        self.dataset_id = dataset_id
        self.max_html_chars = max_html_chars

        if hf_dataset is not None:
            self._dataset = hf_dataset
        else:
            load_kwargs: dict[str, Any] = {"split": split}
            if data_cache_dir is not None:
                load_kwargs["cache_dir"] = str(data_cache_dir)
                load_kwargs["download_mode"] = "reuse_dataset_if_exists"
            self._dataset = load_dataset(dataset_id, **load_kwargs)

        self.outcome_rows = _load_outcome_rows(outcome_path)
        self.pruned_html_rows = _load_pruned_html_rows(pruned_html_path)

        dataset_size = len(self._dataset)
        outcome_index_set = set(self.outcome_rows)
        self.valid_indices = [idx for idx in range(dataset_size) if idx in outcome_index_set]

        missing_from_outcome = [idx for idx in range(dataset_size) if idx not in outcome_index_set]
        pruned_html_overlap = sum(1 for idx in self.valid_indices if idx in self.pruned_html_rows)
        self.dataset_report = {
            "split": split,
            "dataset_id": dataset_id,
            "train_split_size": dataset_size,
            "outcome_rows": len(self.outcome_rows),
            "matched_rows": len(self.valid_indices),
            "missing_from_outcome_rows": len(missing_from_outcome),
            "all_train_samples_accounted_for": len(self.valid_indices) == dataset_size,
            "first_missing_dataset_indices": missing_from_outcome[:25],
            "pruned_html_path": str(pruned_html_path) if pruned_html_path else None,
            "pruned_html_rows_loaded": len(self.pruned_html_rows),
            "matched_rows_with_preprocessed_pruned_html": pruned_html_overlap,
            "matched_rows_without_preprocessed_pruned_html": len(self.valid_indices) - pruned_html_overlap,
        }

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_index = self.valid_indices[index]
        row = self._dataset[dataset_index]
        outcome_row = self.outcome_rows[dataset_index]
        pruned_html_row = self.pruned_html_rows.get(dataset_index)

        if pruned_html_row is not None:
            pruned_html = str(pruned_html_row.get("pruned_html") or "")
        else:
            pruned_html = prune_html_dom(str(row.get("cleaned_html") or ""))

        pruned_html = _truncate_text(pruned_html, self.max_html_chars)
        candidates = _parse_action_reprs(row.get("action_reprs"))
        task = str(row.get("confirmed_task") or "")

        return {
            "dataset_index": dataset_index,
            "annotation_id": str(row.get("annotation_id") or outcome_row.get("annotation_id") or ""),
            "website": str(row.get("website") or ""),
            "confirmed_task": task,
            "candidate_actions": candidates,
            "screenshot": _load_image(row.get("screenshot")),
            "pruned_html": pruned_html,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "user_prompt": build_user_prompt(task, pruned_html, candidates),
            "assistant_text": str(outcome_row.get("target") or ""),
            "operation_type": str(outcome_row.get("operation_type") or ""),
        }


class OutcomeAwareSFTCollator:
    def __init__(self, processor, max_length: int = 4_096) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompt_texts: list[str] = []
        full_texts: list[str] = []
        images: list[Image.Image | None] = []

        for feature in features:
            user_content: list[dict[str, Any]] = []
            if feature["screenshot"] is not None:
                user_content.append({"type": "image", "image": feature["screenshot"]})
            user_content.append({"type": "text", "text": feature["user_prompt"]})

            prompt_messages = [
                {"role": "system", "content": [{"type": "text", "text": feature["system_prompt"]}]},
                {"role": "user", "content": user_content},
            ]
            full_messages = prompt_messages + [
                {"role": "assistant", "content": [{"type": "text", "text": feature["assistant_text"]}]}
            ]

            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            images.append(feature["screenshot"])

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        for row_idx in range(labels.shape[0]):
            prompt_len = int(prompt_batch["attention_mask"][row_idx].sum().item())
            prompt_len = min(prompt_len, labels.shape[1])
            labels[row_idx, :prompt_len] = -100

        batch["labels"] = labels
        return batch
