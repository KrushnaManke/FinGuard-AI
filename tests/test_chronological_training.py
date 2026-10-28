import numpy as np
import pandas as pd

from finguard.data import BASE_FEATURES, TARGET
from finguard.train import chronological_split, train


def chronological_fixture(rows=200):
    frame = pd.DataFrame(
        {
            column: np.linspace(0.1, 4.0, rows) if column != "Time" else np.arange(rows, dtype=float)
            for column in BASE_FEATURES
        }
    )
    frame[TARGET] = np.tile([0, 0, 0, 1], rows // 4)
    return frame


def test_chronological_splits_do_not_overlap_time_and_have_both_classes():
    frame = chronological_fixture()

    splits, train_cutoff, validation_cutoff = chronological_split(frame)

    assert splits["train"]["Time"].max() < splits["validation"]["Time"].min()
    assert splits["validation"]["Time"].max() < splits["test"]["Time"].min()
    assert train_cutoff < validation_cutoff
    assert all(part[TARGET].nunique() == 2 for part in splits.values())


def test_training_uses_validation_alert_capacity_and_reports_later_test(tmp_path):
    source = tmp_path / "transactions.csv"
    chronological_fixture().to_csv(source, index=False)

    report = train(source, tmp_path / "models", validation_alert_fraction=0.1)

    assert report["split"]["strategy"].startswith("chronological")
    assert report["threshold_objective"] == "alert-rate"
    assert report["validation_alert_budget"]["target_count"] == 3
    assert report["test_metrics"]["true_positives"] >= 0
    assert len(report["test_metrics_at_alert_budgets"]) == 3
    assert report["test_all_negative_baseline"]["recall"] == 0
    assert (tmp_path / "models" / "fraud_model.joblib").is_file()
