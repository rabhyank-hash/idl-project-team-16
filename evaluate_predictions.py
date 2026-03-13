"""
Evaluate a predictions JSON file produced by run_inference.py.
Computes Element Accuracy, Operation F1, Step SR, Task SR, Top-3 Accuracy.

Usage
-----
    python evaluate_predictions.py outputs/html_predictions_20240101_120000.json
    python evaluate_predictions.py outputs/axtree_predictions_*.json   # glob OK via shell
    python evaluate_predictions.py a.json b.json c.json                # compare multiple
"""

import argparse
import json
import os
import sys

from mind2web_metrics import evaluate, compare


def load_and_evaluate(path: str):
    with open(path) as f:
        predictions = json.load(f)
    print(f"\nEvaluating: {path}  ({len(predictions)} examples)")
    results = evaluate(predictions)
    print(results)

    # Save metrics alongside predictions
    metrics_path = path.replace("_predictions_", "_metrics_")
    if metrics_path == path:                      # fallback if no _predictions_ in name
        metrics_path = path.replace(".json", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"Metrics saved → {metrics_path}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions", nargs="+", help="Path(s) to predictions JSON file(s)")
    args = p.parse_args()

    if len(args.predictions) == 1:
        load_and_evaluate(args.predictions[0])
    else:
        ablations = {}
        for path in args.predictions:
            name = os.path.splitext(os.path.basename(path))[0]
            ablations[name] = load_and_evaluate(path)
        print("\n" + "=" * 60)
        print("Comparison")
        print("=" * 60)
        print(compare(ablations))


if __name__ == "__main__":
    main()
