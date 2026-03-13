"""
Pre-generate AXTree files from cleaned_html for a dataset split.
Saves one .txt per example: axtrees/{split}/{idx:06d}.txt

Run this once before inference when using --use_axtree.

Usage
-----
    python generate_axtrees.py
    python generate_axtrees.py --split test_website --max_nodes 300
"""

import argparse
import os

from datasets import load_dataset
from tqdm import tqdm

from mind2web_downloader import html_to_axtree

DEFAULT = {
    "dataset_id": "osunlp/Multimodal-Mind2Web",
    "split":      "test_website",
    "max_nodes":  300,
    "output_dir": "axtrees",
}


def main(args):
    print(f"Loading {args.dataset_id}  split={args.split}")
    ds = load_dataset(args.dataset_id, split=args.split)

    out_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing {len(ds)} AXTrees → {out_dir}/")

    for i in tqdm(range(len(ds)), desc="AXTree"):
        row  = ds[i]
        tree = html_to_axtree(row["cleaned_html"], max_nodes=args.max_nodes)
        path = os.path.join(out_dir, f"{i:06d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(tree)

    print(f"Done — {len(ds)} files in {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_id", default=DEFAULT["dataset_id"])
    p.add_argument("--split",      default=DEFAULT["split"],
                   choices=["train", "test_website", "test_task", "test_domain"])
    p.add_argument("--max_nodes",  type=int, default=DEFAULT["max_nodes"])
    p.add_argument("--output_dir", default=DEFAULT["output_dir"])
    main(p.parse_args())
