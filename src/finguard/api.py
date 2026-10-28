"""FastAPI prediction service for the trained fraud model."""

from __future__ import annotations

import os
import hmac
import sqlite3
from functools import lru_cache
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi import Depends, Header
from pydantic import BaseModel, Field
import numpy as np
from dotenv import load_dotenv

from finguard.data import BASE_FEATURES, MODEL_FEATURES
from finguard.features import engineer_transaction_features
from finguard.investigation import investigate
from finguard.paths import project_path
from finguard.rag import ComplianceRetriever
from finguard.storage import AuditStore
from finguard.synthetic import make_synthetic_context

load_dotenv()
MODEL_DIR = project_path("FINGUARD_MODEL_DIR", "models")
app = FastAPI(title="FinGuard AI API", version="0.1.0")


class PredictionRequest(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=128)
    features: dict[str, float]


class ReviewRequest(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=128)
    reviewer: str = Field(min_length=1, max_length=128)
    status: str
    notes: str = Field(default="", max_length=2000)


class InvestigationRequest(PredictionRequest):
    question: str = Field(default="", max_length=1000)
    include_synthetic_demo_context: bool = False


def require_api_key(authorization: Optional[str] = Header(default=None)):
    expected = os.getenv("FINGUARD_API_KEY")
    production = os.getenv("FINGUARD_ENV", "development").lower() == "production"
    if not expected:
        if production:
            raise HTTPException(status_code=503, detail="API authentication is not configured.")
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Valid bearer API key required.")
    supplied = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Valid bearer API key required.")


@lru_cache(maxsize=1)
def get_artifacts():
    model_path = MODEL_DIR / "fraud_model.joblib"
    anomaly_path = MODEL_DIR / "anomaly_model.joblib"
    report_path = MODEL_DIR / "evaluation.json"
    if not all(path.is_file() for path in (model_path, anomaly_path, report_path)):
        raise FileNotFoundError("Train the models before starting the API.")
    import json

    return (
        joblib.load(model_path),
        joblib.load(anomaly_path),
        json.loads(report_path.read_text(encoding="utf-8")),
    )


@app.get("/health")
def health():
    try:
        get_artifacts()
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "ok"}


def score_transaction(request: PredictionRequest):
    try:
        model, anomaly_model, report = get_artifacts()
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    missing = sorted(set(BASE_FEATURES) - set(request.features))
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required features: {', '.join(missing)}")
    values = {feature: request.features[feature] for feature in BASE_FEATURES}
    if not all(np.isfinite(value) for value in values.values()):
        raise HTTPException(status_code=422, detail="Features must be finite numeric values.")
    try:
        engineered = engineer_transaction_features(pd.DataFrame([values]))[MODEL_FEATURES]
        fraud_score = float(model.predict_proba(engineered)[0, 1])
        anomaly_score = float(-anomaly_model.decision_function(engineered)[0])
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid feature values: {error}") from error

    threshold = float(report["threshold"])
    store = AuditStore(project_path("FINGUARD_AUDIT_DB", "data/processed/finguard.sqlite3"))
    store.log_prediction(request.transaction_id, fraud_score, anomaly_score, str(report["model"]))
    return report, {
        "transaction_id": request.transaction_id,
        "fraud_score": fraud_score,
        "anomaly_score": anomaly_score,
        "threshold": threshold,
        "review_alert": fraud_score >= threshold,
        "model_version": str(report["model"]),
        "decision": "analyst_review_required",
    }


@app.post("/v1/predict", dependencies=[Depends(require_api_key)])
def predict(request: PredictionRequest):
    _, result = score_transaction(request)
    return result


@app.post("/v1/investigations", dependencies=[Depends(require_api_key)])
def create_investigation(request: InvestigationRequest):
    _, scored = score_transaction(request)
    compliance_dir = project_path("FINGUARD_COMPLIANCE_DIR", "data/compliance")
    backend = os.getenv("FINGUARD_RAG_BACKEND", "tfidf")
    try:
        retriever = ComplianceRetriever(compliance_dir, backend)
        synthetic_context = (
            make_synthetic_context(request.transaction_id, request.features["Amount"])
            if request.include_synthetic_demo_context
            else None
        )
        report = investigate(
            {
                **scored,
                "transaction_source": "API-submitted transaction features",
            },
            retriever,
            request.question,
            synthetic_context,
        )
        investigation_id = AuditStore(
            project_path("FINGUARD_AUDIT_DB", "data/processed/finguard.sqlite3")
        ).save_investigation(request.transaction_id, report)
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=f"Investigation workflow failed: {error}") from error
    return {
        "prediction": scored,
        "investigation": report,
        "synthetic_context": synthetic_context,
        "investigation_id": investigation_id,
    }


@app.post("/v1/reviews", dependencies=[Depends(require_api_key)])
def review(request: ReviewRequest):
    try:
        review_id = AuditStore(project_path("FINGUARD_AUDIT_DB", "data/processed/finguard.sqlite3")).save_review(
            request.transaction_id,
            request.reviewer,
            request.status,
            request.notes,
        )
    except (sqlite3.Error, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"review_id": review_id, "status": request.status}
