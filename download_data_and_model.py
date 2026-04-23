"""
Download the Multimodal Mind2Web dataset into `data/` and the
Qwen3-VL-8B-Thinking model into `assets/`.

Examples
--------
    python download_data_and_model.py
    python download_data_and_model.py --dataset-only
    python download_data_and_model.py --model-only
    python download_data_and_model.py --splits test_website test_domain
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATASET_ID = "osunlp/Multimodal-Mind2Web"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_SPLITS = ("train", "test_website", "test_task", "test_domain")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Download Mind2Web into data/ and Qwen3-VL-8B-Thinking into assets/."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data")
    parser.add_argument("--assets-dir", type=Path, default=repo_root / "assets")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to materialize locally.",
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Download only the dataset.",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Download only the model.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force a fresh download instead of reusing local cached files.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def download_dataset(
    dataset_id: str,
    splits: list[str],
    data_dir: Path,
    force_redownload: bool,
) -> dict:
    from datasets import load_dataset

    cache_dir = data_dir / "multimodal_mind2web"
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_sizes: dict[str, int] = {}
    cache_files: dict[str, list[str]] = {}

    download_mode = "force_redownload" if force_redownload else "reuse_dataset_if_exists"

    print(f"Downloading dataset {dataset_id} into {cache_dir}")
    for split in splits:
        print(f"  - materializing split: {split}")
        ds = load_dataset(
            dataset_id,
            split=split,
            cache_dir=str(cache_dir),
            download_mode=download_mode,
        )
        split_sizes[split] = len(ds)
        cache_files[split] = [entry["filename"] for entry in ds.cache_files]

    manifest = {
        "dataset_id": dataset_id,
        "splits": splits,
        "split_sizes": split_sizes,
        "cache_dir": str(cache_dir),
        "cache_files": cache_files,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(data_dir / "mind2web_download_manifest.json", manifest)
    return manifest


def download_model(model_id: str, assets_dir: Path, force_redownload: bool) -> dict:
    from huggingface_hub import snapshot_download

    model_dir = assets_dir / model_id.split("/")[-1]
    cache_dir = assets_dir / ".hf-cache"
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model {model_id} into {model_dir}")
    snapshot_path = snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        cache_dir=str(cache_dir),
        force_download=force_redownload,
        resume_download=not force_redownload,
    )

    manifest = {
        "model_id": model_id,
        "model_dir": str(model_dir),
        "snapshot_path": str(snapshot_path),
        "cache_dir": str(cache_dir),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(assets_dir / "model_download_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()

    if args.dataset_only and args.model_only:
        raise SystemExit("Choose only one of --dataset-only or --model-only.")

    data_dir = args.data_dir.resolve()
    assets_dir = args.assets_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    should_download_dataset = not args.model_only
    should_download_model = not args.dataset_only

    if should_download_dataset:
        dataset_manifest = download_dataset(
            dataset_id=args.dataset_id,
            splits=args.splits,
            data_dir=data_dir,
            force_redownload=args.force_redownload,
        )
        print(
            "Dataset ready:",
            f"{dataset_manifest['dataset_id']} -> {dataset_manifest['cache_dir']}",
        )

    if should_download_model:
        model_manifest = download_model(
            model_id=args.model_id,
            assets_dir=assets_dir,
            force_redownload=args.force_redownload,
        )
        print(
            "Model ready:",
            f"{model_manifest['model_id']} -> {model_manifest['model_dir']}",
        )


if __name__ == "__main__":
    main()
