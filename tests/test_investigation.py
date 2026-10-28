import pytest

from finguard.investigation import investigate
from finguard.rag import ComplianceRetriever
from finguard.synthetic import make_synthetic_context


def test_synthetic_customer_context_is_repeatable_and_separately_labeled():
    first = make_synthetic_context("TXN-000123", 850.0)
    second = make_synthetic_context("TXN-000123", 850.0)

    assert first == second
    assert first["data_classification"] == "SYNTHETIC_DEMO_ONLY"
    assert first["current_amount"] == 850.0
    assert "fictional" in first["notice"]


def test_investigation_keeps_synthetic_evidence_out_of_model_score(tmp_path):
    transaction = {
        "transaction_id": "TXN-000123",
        "transaction_source": "benchmark row",
        "fraud_score": 0.9,
        "anomaly_score": 0.2,
        "threshold": 0.8,
    }
    context = make_synthetic_context("TXN-000123", 850.0)

    report = investigate(transaction, ComplianceRetriever(tmp_path), synthetic_context=context)

    assert report["data_lineage"]["synthetic_context_attached"] is True
    assert report["data_lineage"]["synthetic_context_used_for_model_score"] is False
    assert any(item["is_synthetic"] for item in report["evidence"])
    assert "No compliance policy conclusion" in report["draft"]
    assert report["decision"] is None


def test_investigation_rejects_context_without_synthetic_label(tmp_path):
    with pytest.raises(ValueError, match="explicitly classified as synthetic"):
        investigate(
            {"fraud_score": 0.9, "threshold": 0.8},
            ComplianceRetriever(tmp_path),
            synthetic_context={"customer": "real?"},
        )
