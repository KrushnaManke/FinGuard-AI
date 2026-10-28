import numpy as np

from finguard.metrics import (
    choose_alert_rate_threshold,
    choose_f1_threshold,
    classification_metrics,
    metrics_at_alert_budgets,
)


def test_metrics_counts_positive_class_errors():
    result = classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 0]),
        np.array([0.1, 0.8, 0.9, 0.2]),
    )

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["f1"] == 0.5


def test_threshold_selection_finds_perfect_separator():
    threshold, result = choose_f1_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
    )

    assert 0.2 < threshold <= 0.8
    assert result["f1"] == 1.0


def test_alert_rate_threshold_and_fixed_budget_metrics():
    labels = np.array([0, 1, 0, 1, 0, 0])
    scores = np.array([0.1, 0.99, 0.8, 0.7, 0.3, 0.2])

    threshold, target = choose_alert_rate_threshold(scores, 1 / 3)
    budget_metrics = metrics_at_alert_budgets(labels, scores, budgets=(2,))

    assert target == 2
    assert threshold == 0.8
    assert budget_metrics[0]["true_positives"] == 1
    assert budget_metrics[0]["recall"] == 0.5
    assert budget_metrics[0]["alert_budget"] == 2
