"""Train, evaluate, and save the first fraud detection model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from finguard.data import BASE_FEATURES, MODEL_FEATURES, TARGET, load_transactions, transaction_summary
from finguard.features import engineer_transaction_features
from finguard.metrics import (
    choose_alert_rate_threshold,
    choose_f1_threshold,
    classification_metrics,
    metrics_at_alert_budgets,
)


def chronological_split(frame):
    """Split by timestamp quantiles without placing equal timestamps in multiple sets."""
    times = frame["Time"].to_numpy()
    train_cutoff, validation_cutoff = (float(value) for value in np.quantile(times, [0.70, 0.85]))
    train_mask = frame["Time"] <= train_cutoff
    validation_mask = (frame["Time"] > train_cutoff) & (frame["Time"] <= validation_cutoff)
    test_mask = frame["Time"] > validation_cutoff
    splits = {
        "train": frame.loc[train_mask],
        "validation": frame.loc[validation_mask],
        "test": frame.loc[test_mask],
    }
    if any(split.empty or split[TARGET].nunique() != 2 for split in splits.values()):
        raise ValueError("Chronological split must contain both classes in all three partitions.")
    return splits, train_cutoff, validation_cutoff


def build_classifier(algorithm: str, random_state: int, positive_weight: float):
    if algorithm == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    if algorithm == "xgboost":
        if importlib.util.find_spec("xgboost") is None:
            raise RuntimeError("XGBoost is optional; install with `pip install -e '.[boosted]'`.")
        try:
            from xgboost import XGBClassifier
        except Exception as error:
            raise RuntimeError(
                "XGBoost is installed but its native runtime could not load. On macOS install "
                "OpenMP (for example, `brew install libomp`), then retry."
            ) from error

        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=positive_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    if algorithm == "lightgbm":
        if importlib.util.find_spec("lightgbm") is None:
            raise RuntimeError("LightGBM is optional; install with `pip install -e '.[boosted]'`.")
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unknown model {algorithm!r}; choose logistic, xgboost, or lightgbm.")


def train(
    data_path: str | Path,
    model_dir: str | Path,
    random_state: int = 42,
    algorithm: str = "logistic",
    threshold_objective: str = "alert-rate",
    validation_alert_fraction: float = 0.005,
) -> dict:
    training_started = time.perf_counter()
    frame = load_transactions(data_path)
    raw_splits, train_cutoff, validation_cutoff = chronological_split(frame)
    split_features = {
        name: engineer_transaction_features(part[BASE_FEATURES])[MODEL_FEATURES]
        for name, part in raw_splits.items()
    }
    split_labels = {name: part[TARGET] for name, part in raw_splits.items()}
    x_train, x_validation, x_test = (
        split_features[name] for name in ("train", "validation", "test")
    )
    y_train, y_validation, y_test = (
        split_labels[name] for name in ("train", "validation", "test")
    )

    model = build_classifier(algorithm, random_state, float((y_train == 0).sum() / y_train.sum()))
    model.fit(x_train, y_train)

    validation_scores = model.predict_proba(x_validation)[:, 1]
    if threshold_objective == "alert-rate":
        threshold, validation_alert_budget = choose_alert_rate_threshold(
            validation_scores, validation_alert_fraction
        )
        validation_threshold_description = (
            f"top validation alert fraction {validation_alert_fraction:.4%} "
            f"(target {validation_alert_budget} alerts)"
        )
    elif threshold_objective == "max-f1":
        threshold, _ = choose_f1_threshold(y_validation.to_numpy(), validation_scores)
        validation_alert_budget = int((validation_scores >= threshold).sum())
        validation_threshold_description = "maximum fraud-class F1 on validation set"
    else:
        raise ValueError("threshold_objective must be 'alert-rate' or 'max-f1'.")
    validation_metrics = classification_metrics(
        y_validation.to_numpy(),
        (validation_scores >= threshold).astype("int8"),
        validation_scores,
    )
    test_scores = model.predict_proba(x_test)[:, 1]
    test_metrics = classification_metrics(
        y_test.to_numpy(),
        (test_scores >= threshold).astype("int8"),
        test_scores,
    )
    all_negative_predictions = y_test.to_numpy() * 0
    baseline_metrics = classification_metrics(
        y_test.to_numpy(),
        all_negative_predictions,
        all_negative_predictions.astype(float),
    )

    anomaly_model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "detector",
                IsolationForest(
                    n_estimators=100,
                    contamination="auto",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    anomaly_model.fit(x_train)
    test_anomaly_scores = -anomaly_model.decision_function(x_test)

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "fraud_model.joblib")
    joblib.dump(anomaly_model, output_dir / "anomaly_model.joblib")

    report = {
        "model": algorithm,
        "target": TARGET,
        "features": MODEL_FEATURES,
        "threshold": threshold,
        "threshold_objective": threshold_objective,
        "threshold_selection": validation_threshold_description,
        "validation_alert_budget": {
            "target_count": validation_alert_budget,
            "actual_count": int((validation_scores >= threshold).sum()),
            "fraction_of_validation_rows": float((validation_scores >= threshold).mean()),
        },
        "split": {
            "strategy": "chronological, timestamp-quantile 70/15/15 split",
            "train_max_time": float(raw_splits["train"]["Time"].max()),
            "validation_min_time": float(raw_splits["validation"]["Time"].min()),
            "validation_max_time": float(raw_splits["validation"]["Time"].max()),
            "test_min_time": float(raw_splits["test"]["Time"].min()),
            "train_cutoff": train_cutoff,
            "validation_cutoff": validation_cutoff,
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_validation)),
            "test_rows": int(len(x_test)),
            "train_fraud": int(y_train.sum()),
            "validation_fraud": int(y_validation.sum()),
            "test_fraud": int(y_test.sum()),
            "random_state": random_state,
        },
        "dataset": transaction_summary(frame),
        "training_duration_seconds": time.perf_counter() - training_started,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_all_negative_baseline": baseline_metrics,
        "test_metrics_at_alert_budgets": metrics_at_alert_budgets(
            y_test.to_numpy(), test_scores
        ),
        "anomaly_test_metrics": {
            "average_precision": float(average_precision_score(y_test, test_anomaly_scores)),
            "roc_auc": float(roc_auc_score(y_test, test_anomaly_scores)),
            "score_description": "Higher values indicate greater anomaly; this is not a fraud probability.",
        },
    }
    if algorithm in {"xgboost", "lightgbm"}:
        report["feature_importance"] = {
            feature: float(value)
            for feature, value in zip(MODEL_FEATURES, model.feature_importances_)
        }
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/creditcard.csv", help="Path to input CSV")
    parser.add_argument("--model-dir", default="models", help="Directory for model artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--algorithm", choices=["logistic", "xgboost", "lightgbm"], default="logistic")
    parser.add_argument("--threshold-objective", choices=["alert-rate", "max-f1"], default="alert-rate")
    parser.add_argument("--validation-alert-fraction", type=float, default=0.005)
    parser.add_argument("--track", choices=["mlflow", "wandb"])
    args = parser.parse_args()

    report = train(
        args.data,
        args.model_dir,
        args.random_state,
        args.algorithm,
        args.threshold_objective,
        args.validation_alert_fraction,
    )
    if args.track:
        from finguard.tracking import log_experiment

        log_experiment(report, args.track)
    print(json.dumps({
        "model": report["model"],
        "threshold": report["threshold"],
        "split": report["split"],
        "test_metrics": report["test_metrics"],
        "test_all_negative_baseline": report["test_all_negative_baseline"],
        "test_metrics_at_alert_budgets": report["test_metrics_at_alert_budgets"],
        "anomaly_test_metrics": report["anomaly_test_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
