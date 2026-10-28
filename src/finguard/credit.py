"""Independent tabular credit-default training entry point."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_credit_model(
    csv_path: str | Path,
    target: str,
    output_dir: str | Path = "models/credit",
    random_state: int = 42,
    algorithm: str = "logistic",
) -> dict:
    """Train a credit-default baseline from a user-supplied labeled CSV."""
    frame = pd.read_csv(csv_path)
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} is not present in the input CSV.")
    if frame[target].isna().any() or frame[target].nunique() != 2:
        raise ValueError("Credit target must be complete and contain exactly two classes.")

    y = frame[target]
    X = frame.drop(columns=[target])
    if X.empty:
        raise ValueError("Credit dataset must include at least one predictor column.")
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    transformers = []
    if numeric:
        transformers.append(
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric)
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers)
    if algorithm == "logistic":
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
    elif algorithm == "lightgbm":
        if importlib.util.find_spec("lightgbm") is None:
            raise RuntimeError("LightGBM is optional; install with `pip install -e '.[boosted]'`.")
        from lightgbm import LGBMClassifier

        classifier = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    else:
        raise ValueError("Credit model must be 'logistic' or 'lightgbm'.")
    model = Pipeline([("preprocess", preprocessor), ("classifier", classifier)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    report = {
        "target": target,
        "model": algorithm,
        "features": X.columns.tolist(),
        "classes": [str(value) for value in model.classes_],
        "positive_class": str(model.classes_[1]),
        "rows": len(frame),
        "test_average_precision": float(average_precision_score(y_test == model.classes_[1], scores)),
        "test_roc_auc": float(roc_auc_score(y_test == model.classes_[1], scores)),
        "test_classification_report": classification_report(y_test, model.predict(X_test), output_dict=True),
        "warning": "Educational baseline only; validate fairness, calibration, and legal requirements before use.",
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination / "credit_model.joblib")
    (destination / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to a credit dataset CSV")
    parser.add_argument("--target", required=True, help="Binary credit-default label column")
    parser.add_argument("--output-dir", default="models/credit")
    parser.add_argument("--algorithm", choices=["logistic", "lightgbm"], default="logistic")
    args = parser.parse_args()
    result = train_credit_model(args.data, args.target, args.output_dir, algorithm=args.algorithm)
    print(json.dumps({key: result[key] for key in ("target", "rows", "test_average_precision", "test_roc_auc")}, indent=2))


if __name__ == "__main__":
    main()
