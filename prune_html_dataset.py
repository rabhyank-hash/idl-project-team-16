"""
Materialize lightly pruned HTML from Multimodal Mind2Web into JSONL files.

Examples
--------
    python prune_html_dataset.py
    python prune_html_dataset.py --splits test_website
    python prune_html_dataset.py --limit 100
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from html_pruning import prune_html_dom


DEFAULT_DATASET_ID = "osunlp/Multimodal-Mind2Web"
DEFAULT_SPLITS = ("train", "test_website", "test_task", "test_domain")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Write pruned Mind2Web HTML to data/pruned_html/*.jsonl."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to prune.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows per split.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force re-download of the source dataset instead of reusing cache.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    data_dir = args.data_dir.resolve()
    cache_dir = data_dir / "multimodal_mind2web"
    output_dir = data_dir / "pruned_html"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    download_mode = "force_redownload" if args.force_redownload else "reuse_dataset_if_exists"
    manifest = {
        "dataset_id": args.dataset_id,
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "splits": {},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    for split in args.splits:
        print(f"Loading split={split} from {args.dataset_id}")
        ds = load_dataset(
            args.dataset_id,
            split=split,
            cache_dir=str(cache_dir),
            download_mode=download_mode,
        )

        total_rows = len(ds) if args.limit is None else min(args.limit, len(ds))
        out_path = output_dir / f"{split}.jsonl"

        print(f"Writing {total_rows} pruned rows -> {out_path}")
        with out_path.open("w", encoding="utf-8") as f:
            for idx in range(total_rows):
                row = ds[idx]
                cleaned_html = str(row.get("cleaned_html") or "")
                pruned_html = prune_html_dom(cleaned_html)
                payload = {
                    "index": idx,
                    "annotation_id": str(row.get("annotation_id") or ""),
                    "action_uid": str(row.get("action_uid") or ""),
                    "website": str(row.get("website") or ""),
                    "confirmed_task": str(row.get("confirmed_task") or ""),
                    "cleaned_html_chars": len(cleaned_html),
                    "pruned_html_chars": len(pruned_html),
                    "pruned_html": pruned_html,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        manifest["splits"][split] = {
            "rows_written": total_rows,
            "output_path": str(out_path),
        }

    _write_json(output_dir / "manifest.json", manifest)
    print(f"Pruned HTML manifest -> {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
