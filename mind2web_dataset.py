"""
PyTorch Dataset and DataLoader for osunlp/Multimodal-Mind2Web.

Each item contains:
  - screenshot          : PIL.Image (or None)
  - confirmed_task      : str
  - page_context        : str  — AXTree text if use_axtree=True, else cleaned HTML
  - context_type        : str  — "axtree" or "html"
  - action_reprs        : List[str]
  - target_action_index : int
  - target_action_reprs : str
  - action_uid          : str
  - annotation_id       : str
  - website             : str
"""

import io
import json
import os
from typing import List, Optional

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ── Helpers ───────────────────────────────────────────────────────────────────

def _truncate_html(html: str, max_chars: int) -> str:
    if len(html) <= max_chars:
        return html
    half = max_chars // 2
    return html[:half] + "\n... [HTML truncated] ...\n" + html[-half:]


def _load_image(raw) -> Optional[Image.Image]:
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


def _parse_action_reprs(raw) -> List[str]:
    if isinstance(raw, list):
        return [str(a) for a in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(a) for a in parsed]
        except json.JSONDecodeError:
            pass
        return [raw]
    return []


def _parse_target_index(raw) -> int:
    if isinstance(raw, (int, float)):
        return int(raw)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


# ── Dataset ───────────────────────────────────────────────────────────────────

class Mind2WebDataset(Dataset):
    """
    Args:
        split        : HF split — "train", "test_website", "test_task", "test_domain"
        dataset_id   : HuggingFace repo ID
        max_html_chars : max chars for cleaned_html (ignored when use_axtree=True)
        use_axtree   : if True, load pre-generated AXTree from axtree_dir
        axtree_dir   : root folder where generate_axtrees.py wrote files
                       (expects axtree_dir/{split}/{idx:06d}.txt)
        hf_dataset   : pass an already-loaded HF dataset to skip download
    """

    VALID_SPLITS = ("train", "test_website", "test_task", "test_domain")

    def __init__(
        self,
        split: str = "test_website",
        dataset_id: str = "osunlp/Multimodal-Mind2Web",
        max_html_chars: int = 12_000,
        use_axtree: bool = False,
        axtree_dir: str = "axtrees",
        hf_dataset=None,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{split}'")

        self.split = split
        self.max_html_chars = max_html_chars
        self.use_axtree = use_axtree
        self.axtree_split_dir = os.path.join(axtree_dir, split)

        if use_axtree and not os.path.isdir(self.axtree_split_dir):
            raise FileNotFoundError(
                f"AXTree directory not found: {self.axtree_split_dir}\n"
                f"Run `python generate_axtrees.py --split {split}` first."
            )

        if hf_dataset is not None:
            self._data = hf_dataset
        else:
            print(f"[Mind2WebDataset] Downloading '{dataset_id}' split='{split}' …")
            self._data = load_dataset(dataset_id, split=split)

        context = f"AXTree ({self.axtree_split_dir})" if use_axtree else "HTML"
        print(f"[Mind2WebDataset] {len(self._data)} examples | context={context}")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        row = self._data[idx]

        if self.use_axtree:
            path = os.path.join(self.axtree_split_dir, f"{idx:06d}.txt")
            with open(path, "r", encoding="utf-8") as f:
                page_context = f.read()
            context_type = "axtree"
        else:
            page_context = _truncate_html(
                str(row.get("cleaned_html") or ""), self.max_html_chars
            )
            context_type = "html"

        return {
            "screenshot":           _load_image(row.get("screenshot")),
            "confirmed_task":       str(row.get("confirmed_task") or ""),
            "page_context":         page_context,
            "context_type":         context_type,
            "action_reprs":         _parse_action_reprs(row.get("action_reprs")),
            "target_action_index":  _parse_target_index(row.get("target_action_index")),
            "target_action_reprs":  str(row.get("target_action_reprs") or ""),
            "action_uid":           str(row.get("action_uid") or ""),
            "annotation_id":        str(row.get("annotation_id") or ""),
            "website":              str(row.get("website") or ""),
        }


# ── Collate ───────────────────────────────────────────────────────────────────

def mind2web_collate_fn(batch: List[dict]) -> dict:
    return {
        "screenshot":          [item["screenshot"] for item in batch],
        "confirmed_task":      [item["confirmed_task"] for item in batch],
        "page_context":        [item["page_context"] for item in batch],
        "context_type":        [item["context_type"] for item in batch],
        "action_reprs":        [item["action_reprs"] for item in batch],
        "target_action_index": torch.tensor(
            [item["target_action_index"] for item in batch], dtype=torch.long
        ),
        "target_action_reprs": [item["target_action_reprs"] for item in batch],
        "action_uid":          [item["action_uid"] for item in batch],
        "annotation_id":       [item["annotation_id"] for item in batch],
        "website":             [item["website"] for item in batch],
    }


# ── Factory ───────────────────────────────────────────────────────────────────

def get_dataloader(
    split: str = "test_website",
    dataset_id: str = "osunlp/Multimodal-Mind2Web",
    max_html_chars: int = 12_000,
    use_axtree: bool = False,
    axtree_dir: str = "axtrees",
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    hf_dataset=None,
) -> DataLoader:
    dataset = Mind2WebDataset(
        split=split,
        dataset_id=dataset_id,
        max_html_chars=max_html_chars,
        use_axtree=use_axtree,
        axtree_dir=axtree_dir,
        hf_dataset=hf_dataset,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=mind2web_collate_fn,
    )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--use_axtree", action="store_true")
    p.add_argument("--axtree_dir", default="axtrees")
    args = p.parse_args()

    loader = get_dataloader(
        split="test_website", batch_size=2,
        use_axtree=args.use_axtree, axtree_dir=args.axtree_dir,
    )
    batch = next(iter(loader))
    print("Keys         :", list(batch.keys()))
    print("Context type :", batch["context_type"])
    print("Tasks        :", [t[:60] for t in batch["confirmed_task"]])
    print("Gold indices :", batch["target_action_index"].tolist())
    print("# candidates :", [len(a) for a in batch["action_reprs"]])
    print("Context len  :", [len(c) for c in batch["page_context"]])
