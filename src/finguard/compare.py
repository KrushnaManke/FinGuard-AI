"""Train comparable fraud models on the same chronological split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finguard.train import train


def compare_models(data_path: str | Path, output_dir: str | Path, algorithms) -> dict:
    if len(algorithms) < 2:
        raise ValueError("Provide at least two algorithms for comparison.")
    if len(set(algorithms)) != len(algorithms):
        raise ValueError("Algorithms must be unique.")
    root = Path(output_dir)
    reports = {
        algorithm: train(data_path, root / algorithm, algorithm=algorithm)
        for algorithm in algorithms
    }
    comparison = {
        "evaluation": "same chronological train/validation/test protocol and validation alert fraction",
        "models": {
            algorithm: {
                "test_metrics": report["test_metrics"],
                "test_metrics_at_alert_budgets": report["test_metrics_at_alert_budgets"],
                "test_all_negative_baseline": report["test_all_negative_baseline"],
                "threshold": report["threshold"],
                "threshold_selection": report["threshold_selection"],
            }
            for algorithm, report in reports.items()
        },
        "ranked_by_test_average_precision": sorted(
            algorithms,
            key=lambda algorithm: reports[algorithm]["test_metrics"]["average_precision"],
            reverse=True,
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/creditcard.csv")
    parser.add_argument("--output-dir", default="models/comparison")
    parser.add_argument("--algorithms", nargs="+", choices=["logistic", "xgboost", "lightgbm"],
                        default=["logistic", "xgboost"])
    args = parser.parse_args()
    result = compare_models(args.data, args.output_dir, args.algorithms)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
