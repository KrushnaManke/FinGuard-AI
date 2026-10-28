import pytest
import numpy as np

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from finguard import api
from finguard.data import BASE_FEATURES, MODEL_FEATURES


class MockModel:
    def predict_proba(self, frame):
        assert list(frame.columns) == MODEL_FEATURES
        return np.array([[0.1, 0.9]])


class MockAnomalyModel:
    def decision_function(self, frame):
        return [-0.25]


def test_prediction_api_validates_features_and_logs_prediction(tmp_path, monkeypatch):
    monkeypatch.setenv("FINGUARD_AUDIT_DB", str(tmp_path / "audit.db"))
    monkeypatch.delenv("FINGUARD_API_KEY", raising=False)
    monkeypatch.setattr(
        api,
        "get_artifacts",
        lambda: (MockModel(), MockAnomalyModel(), {"threshold": 0.5, "model": "test"}),
    )
    client = TestClient(api.app)
    valid_features = {name: 0.1 for name in BASE_FEATURES}

    response = client.post(
        "/v1/predict",
        json={"transaction_id": "demo-1", "features": valid_features},
    )

    assert response.status_code == 200
    assert response.json()["review_alert"] is True
    assert response.json()["decision"] == "analyst_review_required"
    invalid = client.post("/v1/predict", json={"transaction_id": "demo-2", "features": {"Amount": 1}})
    assert invalid.status_code == 422


def test_investigation_api_keeps_optional_synthetic_context_out_of_score(tmp_path, monkeypatch):
    monkeypatch.setenv("FINGUARD_AUDIT_DB", str(tmp_path / "audit.db"))
    monkeypatch.setenv("FINGUARD_COMPLIANCE_DIR", str(tmp_path / "empty"))
    monkeypatch.delenv("FINGUARD_API_KEY", raising=False)
    monkeypatch.setattr(
        api,
        "get_artifacts",
        lambda: (MockModel(), MockAnomalyModel(), {"threshold": 0.5, "model": "test"}),
    )
    client = TestClient(api.app)
    request = {
        "transaction_id": "demo-investigation",
        "features": {name: 0.1 for name in BASE_FEATURES},
        "question": "policy requirements",
        "include_synthetic_demo_context": True,
    }

    response = client.post("/v1/investigations", json=request)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["fraud_score"] == 0.9
    assert body["synthetic_context"]["data_classification"] == "SYNTHETIC_DEMO_ONLY"
    assert body["investigation"]["data_lineage"]["synthetic_context_used_for_model_score"] is False
    assert body["investigation"]["sources"] == []
    assert body["investigation_id"] == 1


def test_production_api_requires_configured_key(tmp_path, monkeypatch):
    monkeypatch.setenv("FINGUARD_ENV", "production")
    monkeypatch.delenv("FINGUARD_API_KEY", raising=False)

    with pytest.raises(Exception) as error:
        api.require_api_key(authorization=None)
    assert getattr(error.value, "status_code", None) == 503
