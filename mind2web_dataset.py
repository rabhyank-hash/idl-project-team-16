"""
PyTorch Dataset and DataLoader for osunlp/Multimodal-Mind2Web.

Each item contains:
  - screenshot        : PIL.Image (or None if unavailable)
  - confirmed_task    : str  – natural-language task description
  - cleaned_html      : str  – DOM snapshot (truncated to max_html_chars)
  - action_reprs      : List[str] – candidate actions
  - target_action_index : int – gold label
  - target_action_reprs : str – gold action representation
  - action_uid        : str
  - annotation_id     : str
  - website           : str
"""

import io
import json
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Helpers (mirrors the notebook utilities)
# ---------------------------------------------------------------------------

def truncate_html(html: str, max_chars: int = 12_000) -> str:
    """Keep first and last quarters, replace middle with an ellipsis marker."""
    if len(html) <= max_chars:
        return html
    keep = max_chars // 2
    return html[:keep] + "\n... [HTML truncated] ...\n" + html[-keep:]


def _load_image(raw) -> Optional[Image.Image]:
    """Convert whatever HF gives us into a PIL Image (or None)."""
    if raw is None:
        return None
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, bytes):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(raw, dict):
        # HF image feature stores {"bytes": ..., "path": ...}
        if raw.get("bytes"):
            return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
    return None


def _parse_action_reprs(raw) -> List[str]:
    """Normalise candidate actions to a plain list of strings."""
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
    """Coerce target_action_index to int."""
    if isinstance(raw, (int, float)):
        return int(raw)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Mind2WebDataset(Dataset):
    """
    Wraps either a pre-loaded HuggingFace dataset split or loads one on demand.

    Args:
        split: HF split name – "train", "test_website", "test_task", "test_domain"
        dataset_id: HuggingFace dataset identifier
        max_html_chars: Maximum characters kept in cleaned_html
        hf_dataset: Pass an already-loaded HF dataset to avoid re-downloading
    """

    VALID_SPLITS = ("train", "test_website", "test_task", "test_domain")

    def __init__(
        self,
        split: str = "test_website",
        dataset_id: str = "osunlp/Multimodal-Mind2Web",
        max_html_chars: int = 12_000,
        hf_dataset=None,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{split}'")

        self.split = split
        self.max_html_chars = max_html_chars

        if hf_dataset is not None:
            self._data = hf_dataset
        else:
            print(f"[Mind2WebDataset] Loading '{dataset_id}' split='{split}' …")
            self._data = load_dataset(dataset_id, split=split)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        row = self._data[idx]

        screenshot = _load_image(row.get("screenshot"))
        html = truncate_html(str(row.get("cleaned_html") or ""), self.max_html_chars)
        action_reprs = _parse_action_reprs(row.get("action_reprs"))
        target_idx = _parse_target_index(row.get("target_action_index"))

        return {
            "screenshot": screenshot,                          # PIL.Image | None
            "confirmed_task": str(row.get("confirmed_task") or ""),
            "cleaned_html": html,
            "action_reprs": action_reprs,                      # List[str]
            "target_action_index": target_idx,                 # int
            "target_action_reprs": str(row.get("target_action_reprs") or ""),
            "action_uid": str(row.get("action_uid") or ""),
            "annotation_id": str(row.get("annotation_id") or ""),
            "website": str(row.get("website") or ""),
        }


# ---------------------------------------------------------------------------
# Collate – keeps variable-length lists intact, PIL images in a plain list
# ---------------------------------------------------------------------------

def mind2web_collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate that handles:
      - PIL Images  → kept as a Python list (not tensorised)
      - List[str]   → kept as a Python list of lists
      - int fields  → stacked into a torch.LongTensor
      - str fields  → kept as a Python list of strings
    """
    return {
        "screenshot": [item["screenshot"] for item in batch],
        "confirmed_task": [item["confirmed_task"] for item in batch],
        "cleaned_html": [item["cleaned_html"] for item in batch],
        "action_reprs": [item["action_reprs"] for item in batch],
        "target_action_index": torch.tensor(
            [item["target_action_index"] for item in batch], dtype=torch.long
        ),
        "target_action_reprs": [item["target_action_reprs"] for item in batch],
        "action_uid": [item["action_uid"] for item in batch],
        "annotation_id": [item["annotation_id"] for item in batch],
        "website": [item["website"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_dataloader(
    split: str = "test_website",
    dataset_id: str = "osunlp/Multimodal-Mind2Web",
    max_html_chars: int = 12_000,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    hf_dataset=None,
) -> DataLoader:
    """
    Build a DataLoader for Multimodal-Mind2Web.

    Args:
        split: Dataset split ("train", "test_website", "test_task", "test_domain")
        dataset_id: HuggingFace dataset repo ID
        max_html_chars: Max characters for cleaned_html (middle-truncated if longer)
        batch_size: Samples per batch
        shuffle: Whether to shuffle (set True for training)
        num_workers: Parallel workers for data loading (0 = main process)
        hf_dataset: Optional pre-loaded HF dataset (avoids re-download)

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = Mind2WebDataset(
        split=split,
        dataset_id=dataset_id,
        max_html_chars=max_html_chars,
        hf_dataset=hf_dataset,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=mind2web_collate_fn,
    )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = get_dataloader(split="test_website", batch_size=2)
    batch = next(iter(loader))

    print("Batch keys  :", list(batch.keys()))
    print("Tasks       :", batch["confirmed_task"])
    print("Gold indices:", batch["target_action_index"])
    print("# candidates:", [len(a) for a in batch["action_reprs"]])
    print("Screenshots :", [type(s).__name__ for s in batch["screenshot"]])
