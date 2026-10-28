"""Reproducible, source-data-only transaction feature engineering."""

import numpy as np
import pandas as pd


def engineer_transaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append amount and elapsed-cycle features without implying calendar time."""
    features = frame.copy()
    if "Amount" in features:
        features["LogAmount"] = np.log1p(features["Amount"].clip(lower=0))
    if "Time" in features:
        seconds_in_day = 24 * 60 * 60
        phase = 2 * np.pi * (features["Time"] % seconds_in_day) / seconds_in_day
        features["ElapsedCycleSin"] = np.sin(phase)
        features["ElapsedCycleCos"] = np.cos(phase)
    return features
