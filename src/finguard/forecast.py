"""Simple baseline forecasting for timestamped alert counts."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error


def forecast_alert_counts(
    frame: pd.DataFrame,
    timestamp_column: str,
    alert_column: str,
    horizon: int = 7,
) -> dict:
    """Forecast daily alert counts using a seven-day seasonal-naive baseline."""
    if timestamp_column not in frame or alert_column not in frame:
        raise ValueError("Timestamp and alert columns must exist in the supplied data.")
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one day.")
    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError("Timestamp column contains invalid or missing values.")
    alert_values = pd.to_numeric(frame[alert_column], errors="coerce")
    if alert_values.isna().any() or not alert_values.isin([0, 1]).all():
        raise ValueError("Alert column must contain only 0 and 1.")
    data = pd.DataFrame({"date": timestamps.dt.floor("D"), "alert": alert_values.astype(int)})
    if not data["alert"].isin([0, 1]).all():
        raise ValueError("Alert column must contain only 0 and 1.")
    series = data.groupby("date")["alert"].sum().asfreq("D", fill_value=0)
    if len(series) < 14:
        raise ValueError("At least 14 daily observations are required for a weekly baseline.")

    test_days = min(7, max(1, len(series) // 5))
    train, test = series.iloc[:-test_days], series.iloc[-test_days:]
    baseline = train.iloc[-7:].to_numpy()
    validation_predictions = [int(baseline[index % len(baseline)]) for index in range(len(test))]
    mae = float(mean_absolute_error(test, validation_predictions))
    future_values = [int(baseline[index % len(baseline)]) for index in range(horizon)]
    start = series.index[-1] + pd.Timedelta(days=1)
    return {
        "method": "seven-day seasonal-naive baseline",
        "observed_days": int(len(series)),
        "test_days": int(test_days),
        "validation_mae": mae,
        "forecast": [
            {"date": (start + pd.Timedelta(days=index)).date().isoformat(), "alerts": count}
            for index, count in enumerate(future_values)
        ],
        "caveat": "Baseline forecast, not a financial decision signal; requires adequate historical data.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--alert-column", required=True)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--output", default="reports/forecast.json")
    args = parser.parse_args()
    result = forecast_alert_counts(pd.read_csv(args.data), args.timestamp, args.alert_column, args.horizon)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
