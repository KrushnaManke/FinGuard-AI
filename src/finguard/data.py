"""Dataset loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "Class"
BASE_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
ENGINEERED_FEATURES = ["LogAmount", "ElapsedCycleSin", "ElapsedCycleCos"]
MODEL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
REQUIRED_COLUMNS = BASE_FEATURES + [TARGET]


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load the labeled credit-card dataset after validating its schema."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    frame = pd.read_csv(data_path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    if frame.empty:
        raise ValueError("Dataset contains no transactions.")

    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    if frame.isnull().any().any():
        null_columns = frame.columns[frame.isnull().any()].tolist()
        raise ValueError(f"Dataset contains missing values in: {', '.join(null_columns)}")

    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    labels = set(frame[TARGET].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only 0 and 1; found {sorted(labels)}")
    if frame[TARGET].nunique() != 2:
        raise ValueError("Dataset must contain both non-fraud and fraud examples.")
    frame[TARGET] = frame[TARGET].astype("int8")
    return frame


def transaction_summary(frame: pd.DataFrame) -> dict:
    """Return descriptive counts for a validated transaction dataset."""
    fraud_count = int(frame[TARGET].sum())
    total = int(len(frame))
    return {
        "transactions": total,
        "duplicate_rows": int(frame.duplicated().sum()),
        "fraud_transactions": fraud_count,
        "non_fraud_transactions": total - fraud_count,
        "fraud_rate": fraud_count / total if total else 0.0,
        "amount_min": float(frame["Amount"].min()),
        "amount_max": float(frame["Amount"].max()),
        "time_min": float(frame["Time"].min()),
        "time_max": float(frame["Time"].max()),
    }
