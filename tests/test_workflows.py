import pandas as pd
import pytest

from finguard.forecast import forecast_alert_counts
from finguard.rag_evaluation import evaluate_scenarios
from finguard.monitoring import population_stability_index
from finguard.rag import ComplianceRetriever
from finguard.storage import AuditStore


def test_forecast_uses_later_holdout_and_returns_future_days():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="D"),
            "alert": [int(index % 3 == 0) for index in range(30)],
        }
    )

    result = forecast_alert_counts(frame, "timestamp", "alert", horizon=3)

    assert result["test_days"] == 6
    assert result["validation_mae"] >= 0
    assert len(result["forecast"]) == 3


def test_forecast_requires_sufficient_daily_history():
    frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=4), "alert": [0, 1, 0, 1]})

    with pytest.raises(ValueError, match="14 daily observations"):
        forecast_alert_counts(frame, "time", "alert")


def test_forecast_rejects_non_binary_alert_values():
    frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=20), "alert": [0.5] * 20})

    with pytest.raises(ValueError, match="only 0 and 1"):
        forecast_alert_counts(frame, "time", "alert")


def test_rag_returns_cited_sources_and_empty_library_is_explicit(tmp_path):
    (tmp_path / "policy.md").write_text("Escalate suspicious transactions to a human reviewer.", encoding="utf-8")
    retriever = ComplianceRetriever(tmp_path)

    matches = retriever.search("suspicious transaction escalation")

    assert matches[0]["source"] == "policy.md"
    assert "human reviewer" in matches[0]["text"]
    assert not ComplianceRetriever(tmp_path / "empty").search("anything")


def test_rag_does_not_index_its_setup_readme(tmp_path):
    (tmp_path / "README.md").write_text("Add policies in this folder.", encoding="utf-8")

    assert not ComplianceRetriever(tmp_path).search("policies in folder")


def test_rag_relevance_threshold_rejects_weak_keyword_match(tmp_path):
    (tmp_path / "policy.md").write_text(
        "A model score is not proof of fraud. An authorized analyst reviews available evidence.",
        encoding="utf-8",
    )
    retriever = ComplianceRetriever(tmp_path)

    assert not retriever.search("customer home address and password")


def test_audit_store_separates_prediction_and_review(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3")

    prediction_id = store.log_prediction("demo-1", 0.7, 0.2, "model-v1")
    review_id = store.save_review("demo-1", "analyst", "escalated", "Needs review")
    investigation_id = store.save_investigation(
        "demo-1",
        {"data_lineage": {"synthetic_context_attached": True}, "sources": []},
    )

    assert prediction_id == 1
    assert review_id == 1
    assert investigation_id == 1
    with pytest.raises(ValueError, match="Unsupported"):
        store.save_review("demo-1", "analyst", "block_account")


def test_audit_store_exposes_dashboard_counts_and_recent_records(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3")
    store.log_prediction("demo-1", 0.7, 0.2, "model-v1")
    store.save_review("demo-1", "analyst", "open")
    store.save_investigation(
        "demo-1",
        {"data_lineage": {"synthetic_context_attached": False}, "sources": []},
    )

    assert store.dashboard_summary() == {
        "predictions": 1,
        "reviews": 1,
        "investigations": 1,
    }
    assert store.recent_records("predictions")[0]["transaction_id"] == "demo-1"
    assert store.recent_records("investigations")[0]["data_classification"] == "BENCHMARK_ONLY"
    with pytest.raises(ValueError, match="Unsupported"):
        store.recent_records("sqlite_master")
    with pytest.raises(ValueError, match="Limit"):
        store.recent_records("predictions", limit=101)


def test_psi_is_zero_for_identical_distribution_and_positive_on_shift():
    reference = pd.Series(range(100))

    assert population_stability_index(reference, reference) == pytest.approx(0)
    assert population_stability_index(reference, reference + 100) > 0


def test_rag_evaluation_reports_citations_missing_info_and_cost(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "policy.md").write_text(
        "A model score is not proof of fraud. An authorized analyst reviews available evidence.",
        encoding="utf-8",
    )
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(
        """[
          {"scenario_id":"SYN-1","query":"model score proof fraud authorized analyst review",
           "expected_sources":["policy.md"],"expect_missing":false},
          {"scenario_id":"SYN-2","query":"unicorn nebula telescope coordinates",
           "expected_sources":[],"expect_missing":true}
        ]""",
        encoding="utf-8",
    )

    report = evaluate_scenarios(scenarios, documents)

    assert report["synthetic_scenarios"] is True
    assert report["metrics"]["citation_recall"] == 1.0
    assert report["metrics"]["missing_information_accuracy"] == 1.0
    assert report["metrics"]["evidence_faithfulness"] == 1.0
    assert report["metrics"]["failure_rate"] == 0.0
    assert report["metrics"]["estimated_llm_cost_usd"] == 0.0
