"""Shared evaluation metrics and alert-threshold selection."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_f1_threshold(y_true, scores) -> tuple[float, dict]:
    """Choose the validation threshold with the best fraud-class F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    denominator = precision[:-1] + recall[:-1]
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    best_index = int(np.argmax(f1_scores))
    threshold = float(thresholds[best_index])
    predictions = (np.asarray(scores) >= threshold).astype("int8")
    return threshold, classification_metrics(y_true, predictions, scores)


def choose_alert_rate_threshold(scores, alert_fraction: float) -> tuple[float, int]:
    """Choose a validation threshold near a configured review-capacity fraction."""
    if not 0 < alert_fraction <= 1:
        raise ValueError("alert_fraction must be greater than 0 and at most 1.")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty one-dimensional array of finite values.")
    target_alerts = max(1, int(np.ceil(len(values) * alert_fraction)))
    descending = np.sort(values)[::-1]
    return float(descending[target_alerts - 1]), target_alerts


def metrics_at_alert_budgets(y_true, scores, budgets=(100, 500, 1000)) -> list[dict]:
    """Report top-ranked recall/precision at fixed review counts."""
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    if len(labels) != len(values) or len(labels) == 0:
        raise ValueError("Labels and scores must have the same non-zero length.")
    order = np.argsort(values)[::-1]
    total_fraud = int(labels.sum())
    prevalence = total_fraud / len(labels)
    result = []
    for requested_budget in budgets:
        budget = min(int(requested_budget), len(labels))
        if budget < 1:
            raise ValueError("Alert budgets must be positive.")
        true_positives = int(labels[order[:budget]].sum())
        result.append(
            {
                "alert_budget": budget,
                "requested_budget": int(requested_budget),
                "true_positives": true_positives,
                "precision": true_positives / budget,
                "recall": true_positives / total_fraud if total_fraud else 0.0,
                "random_baseline_expected_precision": prevalence,
                "random_baseline_expected_recall": budget / len(labels),
            }
        )
    return result


def classification_metrics(y_true, predictions, scores) -> dict:
    """Calculate metrics for the positive (fraud) class."""
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "confusion_matrix": matrix.tolist(),
        "true_negatives": int(matrix[0, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_positives": int(matrix[1, 1]),
    }
