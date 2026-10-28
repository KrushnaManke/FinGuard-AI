"""FinGuard AI analyst dashboard."""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from finguard.data import BASE_FEATURES, MODEL_FEATURES, load_transactions
from finguard.features import engineer_transaction_features
from finguard.investigation import investigate
from finguard.paths import PROJECT_ROOT, project_path
from finguard.rag import ComplianceRetriever
from finguard.storage import AuditStore
from finguard.synthetic import make_synthetic_context

load_dotenv()
FULL_DATA_PATH = project_path("FINGUARD_DATA_PATH", "data/raw/creditcard.csv")
DEMO_DATA_PATH = PROJECT_ROOT / "data" / "demo" / "creditcard-demo.csv"
DATA_PATH = FULL_DATA_PATH if FULL_DATA_PATH.is_file() else DEMO_DATA_PATH
MODEL_DIR = project_path("FINGUARD_MODEL_DIR", "models")
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_model.joblib"
REPORT_PATH = MODEL_DIR / "evaluation.json"
AUDIT_PATH = project_path("FINGUARD_AUDIT_DB", "data/processed/finguard.sqlite3")
RAG_EVALUATION_PATH = PROJECT_ROOT / "reports" / "rag_evaluation.json"

st.set_page_config(
    page_title="FinGuard AI", page_icon="🛡️", layout="wide",
)
st.markdown("""
<style>
:root { --burgundy: #8f1538; --wine: #68102b; --ink: #08090c; --panel: #15171c; }
[data-testid="stAppViewContainer"] { background: radial-gradient(circle at 85% 5%, #2a0715 0, #08090c 32%, #08090c 100%); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #0d0e12; border-right: 1px solid #2b2025; }
[data-testid="stSidebar"] [role="radiogroup"] label { border: 1px solid transparent; border-radius: 8px; padding: 5px 8px; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #211219; }
[data-testid="stSidebar"] [aria-checked="true"] { color: #fff; border-color: #8f1538; background: #68102b40; }
.block-container { padding: 1.25rem 2.5rem 3rem; max-width: 1500px; }
[data-testid="stMetric"] { background: linear-gradient(145deg, #17171c, #111218); border: 1px solid #34242a; border-radius: 12px; padding: 14px 16px; }
[data-testid="stMetricLabel"] { color: #b9b6ba; }
[data-testid="stMetricValue"] { color: #f7f4f5; }
[data-testid="stDataFrame"] { border: 1px solid #30272c; border-radius: 10px; overflow: hidden; }
div[data-testid="stAlert"] { border-radius: 9px; border-left-width: 3px; }
[data-testid="stExpander"] { border: 1px solid #30272c; border-radius: 10px; background: #111218; }
hr { border-color: #30272c; }
button[kind="primary"] { background: var(--burgundy) !important; border-color: var(--burgundy) !important; }
button[kind="primary"]:hover { background: var(--wine) !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Scoring the benchmark transactions with the saved models...")
def get_scored_transactions(
    data_path: str,
    data_modified_ns: int,
    model_path: str,
    model_modified_ns: int,
    anomaly_path: str,
    anomaly_modified_ns: int,
) -> pd.DataFrame:
    frame = load_transactions(data_path).copy()
    model = joblib.load(model_path)
    anomaly_model = joblib.load(anomaly_path)
    inputs = engineer_transaction_features(frame[BASE_FEATURES])[MODEL_FEATURES]
    frame.insert(0, "Transaction ID", [f"TXN-{index:06d}" for index in range(len(frame))])
    frame["Fraud score"] = model.predict_proba(inputs)[:, 1]
    frame["Anomaly score"] = -anomaly_model.decision_function(inputs)
    return frame


@st.cache_data(show_spinner=False)
def read_json(path: str, modified_ns: int) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def load_report(path: Path, label: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        return read_json(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not read {label}: {error}")
        return None


def show_model_metrics(metrics: dict, prefix: str = "") -> None:
    columns = st.columns(5)
    for column, label, key, format_value in zip(
        columns,
        ("PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"),
        ("average_precision", "roc_auc", "precision", "recall", "f1"),
        (".3f", ".3f", ".1%", ".1%", ".3f"),
    ):
        column.metric(f"{prefix}{label}", format(metrics[key], format_value))


def metric_comparison_chart(model_metrics: dict, baseline_metrics: dict) -> go.Figure:
    keys = ("average_precision", "roc_auc", "precision", "recall", "f1")
    labels = ("PR-AUC", "ROC-AUC", "Precision", "Recall", "F1")
    return go.Figure(
        [
            go.Bar(name="Fraud model", x=labels, y=[model_metrics[key] for key in keys], marker_color="#8f1538"),
            go.Bar(
                name="All-negative baseline",
                x=labels,
                y=[baseline_metrics[key] for key in keys],
                marker_color="#4d5058",
            ),
        ]
    ).update_layout(
        barmode="group", yaxis_title="Score", yaxis_range=[0, 1],
        margin=dict(l=10, r=10, t=25, b=10), legend=dict(orientation="h", y=1.12),
    )


def confusion_matrix_chart(metrics: dict) -> go.Figure:
    return px.imshow(
        metrics["confusion_matrix"],
        text_auto=True,
        x=["Predicted legitimate", "Predicted fraud"],
        y=["Actual legitimate", "Actual fraud"],
        color_continuous_scale=["#15171c", "#8f1538"],
        aspect="auto", labels={"x": "", "y": "", "color": "Transactions"},
    ).update_layout(margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)


def api_health(api_url: str) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{api_url.rstrip('/')}/health", timeout=1.5) as response:
            body = json.loads(response.read().decode("utf-8"))
            return response.status == 200 and body.get("status") == "ok", body.get("status", "unknown")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as error:
        return False, str(error)


st.title("FinGuard AI")
st.caption("Risk operations · Human review · Decision support")
st.warning("Educational benchmark prototype. Scores do not prove fraud or authorize transaction blocking.")

report = load_report(REPORT_PATH, "the saved fraud evaluation report")
if report is None:
    if not REPORT_PATH.is_file():
        st.error("Fraud evaluation report is missing. Train the fraud model to generate verified outputs.")
        st.code("python -m finguard.train")
    st.stop()

threshold = float(report["threshold"])
data_available = DATA_PATH.is_file()
models_available = MODEL_PATH.is_file() and ANOMALY_MODEL_PATH.is_file()
transactions = None
if data_available and models_available:
    try:
        transactions = get_scored_transactions(
            str(DATA_PATH),
            DATA_PATH.stat().st_mtime_ns,
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime_ns,
            str(ANOMALY_MODEL_PATH),
            ANOMALY_MODEL_PATH.stat().st_mtime_ns,
        )
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not score transactions with the saved models: {error}")
elif not data_available:
    st.info(f"Benchmark CSV is not present at `{DATA_PATH}`. Held-out model metrics remain available.")
else:
    st.info("One or more trained model artifacts are missing. Run the fraud training command.")

page = st.sidebar.radio(
    "Workspace", [
        "Overview", "Model results", "Investigation", "Audit & services", "Other models",
    ],
)
st.sidebar.markdown(
    f"""<div style="margin-top:16px;color:#8f1538;font-size:.72rem;font-weight:700;letter-spacing:.12em">MODEL</div>
    <div style="margin:4px 0 10px;font-weight:700">{report.get('model', 'unknown')}</div>
    <div style="color:#a9a6aa">Threshold <b style="color:#fff">{threshold:.4f}</b></div>
    <div style="color:#a9a6aa">Data <code style="color:#d6a8b6">{DATA_PATH.name}</code></div>""",
    unsafe_allow_html=True,
)

if page == "Overview":
    st.header("Portfolio overview")
    if transactions is None:
        st.metric("Held-out test transactions", f"{report['split']['test_rows']:,}")
        show_model_metrics(report["test_metrics"])
        st.info("Live benchmark scoring is unavailable until both the dataset and model artifacts are present.")
    else:
        flagged = transactions["Fraud score"] >= threshold
        high_risk = transactions["Fraud score"] >= 0.90
        values = (
            ("Transactions scored", f"{len(transactions):,}"),
            ("Threshold alerts", f"{int(flagged.sum()):,}"),
            ("Scores ≥ 0.90", f"{int(high_risk.sum()):,}"),
            ("Held-out PR-AUC", f"{report['test_metrics']['average_precision']:.3f}"),
            ("Held-out precision", f"{report['test_metrics']['precision']:.1%}"),
            ("Held-out recall", f"{report['test_metrics']['recall']:.1%}"),
        )
        for row in range(0, len(values), 3):
            for column, (label, value) in zip(st.columns(3), values[row:row + 3]):
                column.metric(label, value)
        st.caption(
            "Counts are computed by applying the saved model to the complete local benchmark CSV. "
            "Precision and recall are from the separate chronological test partition."
        )

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.subheader("Fraud score distribution")
            figure = px.histogram(
                transactions,
                x="Fraud score",
                color=(transactions["Fraud score"] >= threshold).map({True: "At/above threshold", False: "Below threshold"}),
                nbins=50,
                labels={"color": "Threshold status", "x": "Model fraud score"},
                color_discrete_map={"At/above threshold": "#8f1538", "Below threshold": "#5f626d"},
            )
            st.plotly_chart(figure)
        with chart_right:
            st.subheader("Observed volume by elapsed hour")
            hourly = (
                transactions.assign(Hour=transactions["Time"].floordiv(3600))
                .groupby("Hour", as_index=False)
                .agg(
                    Transactions=("Time", "size"),
                    Alerts=("Fraud score", lambda values: int((values >= threshold).sum())),
                )
            )
            figure = px.line(hourly, x="Hour", y=["Transactions", "Alerts"], markers=False)
            figure.update_layout(xaxis_title="Elapsed hour in source data", yaxis_title="Count")
            st.plotly_chart(figure)

        st.subheader("Current review queue")
        minimum_score = st.slider("Minimum fraud score", 0.0, 1.0, threshold, 0.01)
        queue = transactions.loc[transactions["Fraud score"] >= minimum_score].sort_values(
            "Fraud score", ascending=False
        )
        st.dataframe(
            queue[["Transaction ID", "Amount", "Time", "Fraud score", "Anomaly score"]].head(1000),
            width="stretch",
            hide_index=True,
        )
        st.caption(f"Displaying up to 1,000 of {len(queue):,} matching rows; source currency is unspecified.")

elif page == "Model results":
    st.header("Model results & validation")
    st.caption(
        f"Saved artifact report · algorithm `{report.get('model', 'unknown')}` · "
        f"threshold {threshold:.6f} · objective `{report.get('threshold_selection', 'not recorded')}`"
    )
    st.subheader("Fraud classifier · chronological held-out test")
    show_model_metrics(report["test_metrics"])
    matrix_left, matrix_right = st.columns([1, 1.5])
    with matrix_left:
        st.markdown("**Confusion matrix**")
        st.plotly_chart(confusion_matrix_chart(report["test_metrics"]))
    with matrix_right:
        st.markdown("**Model vs all-negative baseline**")
        st.plotly_chart(
            metric_comparison_chart(report["test_metrics"], report["test_all_negative_baseline"]),
        )
    metric_details = {
        "threshold": threshold,
        "threshold_selection": report.get("threshold_selection"),
        "held_out_test_metrics": report["test_metrics"],
        "all_negative_test_baseline": report["test_all_negative_baseline"],
        "split_and_counts": report["split"],
    }
    with st.expander("Full fraud test report and chronological split details"):
        st.json(metric_details)

    st.subheader("Investigation capacity trade-off")
    budgets = pd.DataFrame(report["test_metrics_at_alert_budgets"])
    st.dataframe(budgets, width="stretch", hide_index=True)
    if not budgets.empty:
        figure = px.line(
            budgets,
            x="alert_budget",
            y=["precision", "recall", "random_baseline_expected_recall"],
            markers=True,
            labels={"alert_budget": "Alerts analysts can review", "value": "Rate", "variable": "Metric"},
        )
        figure.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(figure)
    st.caption("Threshold selection uses validation data; test results are reported only after selection.")

    st.subheader("Anomaly detector · separate unusualness signal")
    anomaly = report["anomaly_test_metrics"]
    anomaly_columns = st.columns(3)
    anomaly_columns[0].metric("Anomaly PR-AUC", f"{anomaly['average_precision']:.3f}")
    anomaly_columns[1].metric("Anomaly ROC-AUC", f"{anomaly['roc_auc']:.3f}")
    anomaly_columns[2].metric("Threshold", "Not a fraud probability")
    st.info(anomaly.get("score_description", "Higher scores indicate greater anomaly; this is not a fraud probability."))

    st.subheader("Training-data partitions")
    split = report["split"]
    partition_frame = pd.DataFrame([
        {"Partition": name, "Rows": split[f"{key}_rows"], "Fraud": split[f"{key}_fraud"]}
        for name, key in (("Train", "train"), ("Validation", "validation"), ("Test", "test"))
    ])
    st.dataframe(partition_frame, width="stretch", hide_index=True)
    st.caption(f"Split protocol: {split['strategy']}. Equal timestamps remain in the same partition.")

    model_comparison_path = MODEL_DIR / "comparison" / "comparison.json"
    comparison = load_report(model_comparison_path, "the saved model comparison")
    if comparison:
        st.subheader("Saved model comparison")
        st.write("Ranking by held-out average precision:", comparison["ranked_by_test_average_precision"])
        st.json(comparison["models"])
    else:
        if not model_comparison_path.is_file():
            st.info("No classifier comparison is saved. The active model is the logistic baseline.")

elif page == "Investigation":
    st.header("Transaction investigation")
    st.caption("Scored facts, retrieved references, synthetic context, and missing evidence are separated.")
    if transactions is None:
        st.warning("Transaction inspection requires the local dataset and both fraud/anomaly model artifacts.")
    else:
        candidates = (
            transactions.loc[transactions["Fraud score"] >= threshold]
            .nlargest(1000, "Fraud score")["Transaction ID"]
            .tolist()
        )
        if not candidates:
            st.info("No transaction meets the model's current review threshold.")
        else:
            selected_id = st.selectbox("Select a threshold alert", candidates)
            selected = transactions.loc[transactions["Transaction ID"] == selected_id].iloc[0]
            detail_columns = st.columns(4)
            detail_columns[0].metric("Fraud score", f"{selected['Fraud score']:.4f}")
            detail_columns[1].metric("Review threshold", f"{threshold:.4f}")
            detail_columns[2].metric("Anomaly score", f"{selected['Anomaly score']:.4f}")
            detail_columns[3].metric("Source amount", f"{selected['Amount']:,.2f}")
            st.caption("Anomaly score is an independent unusualness signal. Amount currency is unspecified.")
            with st.expander("Inspect source model inputs"):
                st.dataframe(
                    selected[BASE_FEATURES].rename("Value").to_frame(),
                    width="stretch",
                )
            if st.button("Explain this score with SHAP", type="secondary"):
                try:
                    from finguard.explain import explain_instance

                    model_inputs = engineer_transaction_features(transactions[BASE_FEATURES])[MODEL_FEATURES]
                    row_index = int(selected_id.removeprefix("TXN-"))
                    model = joblib.load(MODEL_PATH)
                    contributions = explain_instance(
                        model,
                        model_inputs.iloc[[row_index]],
                        model_inputs.iloc[: min(100, len(model_inputs))],
                    )
                except (RuntimeError, ValueError, ImportError, OSError) as error:
                    st.error(f"Could not calculate SHAP explanation: {error}")
                else:
                    st.caption("Contributions explain this model output only; anonymized V-features have no business labels.")
                    st.dataframe(pd.DataFrame(contributions), width="stretch", hide_index=True)

            question = st.text_input(
                "Compliance retrieval question",
                value="transaction review and escalation requirements",
            )
            attach_synthetic = st.checkbox(
                "Attach a clearly labeled synthetic demo customer-history scenario",
                value=False,
            )
            if st.button("Build and save evidence report", type="primary"):
                compliance_dir = project_path("FINGUARD_COMPLIANCE_DIR", "data/compliance")
                try:
                    retriever = ComplianceRetriever(
                        compliance_dir,
                        os.getenv("FINGUARD_RAG_BACKEND", "tfidf"),
                    )
                    synthetic_context = (
                        make_synthetic_context(selected_id, float(selected["Amount"]))
                        if attach_synthetic
                        else None
                    )
                    investigation = investigate(
                        {
                            "transaction_id": selected_id,
                            "fraud_score": float(selected["Fraud score"]),
                            "anomaly_score": float(selected["Anomaly score"]),
                            "threshold": threshold,
                            "transaction_source": "creditcard.csv benchmark",
                        },
                        retriever,
                        question,
                        synthetic_context,
                    )
                    record_id = AuditStore(AUDIT_PATH).save_investigation(selected_id, investigation)
                except (OSError, sqlite3.Error, ValueError, RuntimeError) as error:
                    st.error(f"Investigation failed or could not be saved: {error}")
                else:
                    st.session_state["investigation_result"] = investigation
                    st.session_state["investigation_record_id"] = record_id
                    st.session_state["investigation_transaction_id"] = selected_id
                    st.session_state["llm_draft"] = None

            investigation = st.session_state.get("investigation_result")
            if investigation and st.session_state.get("investigation_transaction_id") == selected_id:
                st.markdown("---")
                st.subheader("Evidence-supported investigation report")
                st.caption(f"Persisted audit record: {st.session_state['investigation_record_id']}")
                if investigation["data_lineage"]["synthetic_context_attached"]:
                    st.warning(
                        "SYNTHETIC DEMO ONLY — fictional history is for UI demonstration and was not used "
                        "to train or score the model."
                    )
                st.info(investigation["draft"])
                st.markdown("**Evidence ledger**")
                st.dataframe(pd.DataFrame(investigation["evidence"]), width="stretch", hide_index=True)
                if investigation["missing_information"]:
                    st.markdown("**Missing / unavailable information**")
                    for missing_item in investigation["missing_information"]:
                        st.write(f"- {missing_item}")
                if investigation["sources"]:
                    st.markdown("**Retrieved document excerpts**")
                    for source in investigation["sources"]:
                        with st.expander(f"{source['source']} · relevance {source['relevance']:.3f}"):
                            st.write(source["excerpt"])
                else:
                    st.warning("No relevant compliance source was retrieved; no policy conclusion is made.")
                st.download_button(
                    "Download investigation report (JSON)",
                    data=json.dumps(investigation, indent=2),
                    file_name=f"{selected_id}-investigation.json",
                    mime="application/json",
                )

                if os.getenv("OPENAI_API_KEY") and investigation["sources"]:
                    if st.button("Draft with configured LLM"):
                        try:
                            from finguard.llm import draft_investigation

                            st.session_state["llm_draft"] = draft_investigation(investigation)
                        except (RuntimeError, ValueError) as error:
                            st.error(f"LLM draft failed: {error}")
                    llm_draft = st.session_state.get("llm_draft")
                    if llm_draft:
                        st.write(llm_draft["text"])
                        if llm_draft["citation_validation"]["valid"]:
                            st.success("Citation filenames match retrieved sources; human review is still required.")
                        else:
                            st.error("Citation validation failed. Do not rely on this draft.")
                        estimated_cost = llm_draft["estimated_cost_usd"]
                        cost_summary = (
                            f"${estimated_cost:.6f}"
                            if estimated_cost is not None
                            else "unavailable; configure token price rates"
                        )
                        st.caption(
                            f"Model: {llm_draft['model']} · Latency: {llm_draft['latency_seconds']:.2f}s · "
                            f"Tokens: {llm_draft['input_tokens']} in / {llm_draft['output_tokens']} out · "
                            f"Estimated cost: {cost_summary}"
                        )

            with st.form("analyst_review"):
                st.markdown("**Record analyst review**")
                reviewer = st.text_input("Reviewer")
                review_status = st.selectbox(
                    "Review outcome",
                    ["open", "confirmed_fraud", "legitimate", "escalated"],
                )
                review_notes = st.text_area("Notes")
                submitted = st.form_submit_button("Save analyst review")
            if submitted:
                if not reviewer.strip():
                    st.error("Enter a reviewer name to save a review.")
                else:
                    try:
                        review_id = AuditStore(AUDIT_PATH).save_review(
                            selected_id,
                            reviewer.strip(),
                            review_status,
                            review_notes,
                        )
                    except (OSError, sqlite3.Error, ValueError) as error:
                        st.error(f"Could not save analyst review: {error}")
                    else:
                        st.success(f"Review saved (ID {review_id}); the model prediction was not changed.")

elif page == "Audit & services":
    st.header("Audit trail & service health")
    api_url = os.getenv("FINGUARD_API_URL", "http://127.0.0.1:8000")
    api_ok, api_status = api_health(api_url)
    health_columns = st.columns(3)
    health_columns[0].metric("Prediction API", "Online" if api_ok else "Offline")
    health_columns[1].metric("Model artifact", "Loaded" if MODEL_PATH.is_file() else "Missing")
    health_columns[2].metric("Benchmark data", "Available" if data_available else "Missing")
    if api_ok:
        st.success(f"API health endpoint responded at `{api_url.rstrip('/')}/health`.")
    else:
        st.info(f"API is not reachable at `{api_url}` ({api_status}). Start it with `uvicorn finguard.api:app`.")

    st.subheader("Shared SQLite audit database")
    st.caption(f"Database path: `{AUDIT_PATH}`")
    try:
        store = AuditStore(AUDIT_PATH)
        counts = store.dashboard_summary()
        audit_columns = st.columns(3)
        for column, (label, count) in zip(audit_columns, counts.items()):
            column.metric(f"Persisted {label}", f"{count:,}")
    except (OSError, sqlite3.Error, ValueError) as error:
        st.error(f"Could not read the audit database: {error}")
    else:
        record_type = st.selectbox("Inspect persisted records", ["predictions", "investigations", "reviews"])
        try:
            records = store.recent_records(record_type, limit=50)
        except (OSError, sqlite3.Error, ValueError) as error:
            st.error(f"Could not read {record_type}: {error}")
        else:
            if records:
                if record_type == "investigations":
                    for record in records:
                        with st.expander(
                            f"#{record['id']} · {record['transaction_id']} · "
                            f"{record['data_classification']} · {record['created_at']}"
                        ):
                            st.json(json.loads(record["report_json"]))
                else:
                    st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
            else:
                st.info(f"No {record_type} have been recorded yet.")
        st.caption(
            "Dashboard investigations and API predictions/reviews use the configured SQLite file. "
            "API requests are persisted when the API is running and receives requests."
        )

    st.subheader("Artifacts")
    artifacts = {"Fraud model": MODEL_PATH, "Anomaly model": ANOMALY_MODEL_PATH,
                 "Evaluation": REPORT_PATH, "RAG evaluation": RAG_EVALUATION_PATH}
    st.dataframe(pd.DataFrame([
        {"Artifact": name, "Status": "Available" if path.is_file() else "Missing"}
        for name, path in artifacts.items()
    ]), width="stretch", hide_index=True)

elif page == "Other models":
    st.header("Optional model outputs")
    st.caption(
        "These extensions need their own suitable data and evaluation. Missing artifacts are reported as "
        "not available, never replaced with simulated performance."
    )
    credit_report_path = MODEL_DIR / "credit" / "evaluation.json"
    credit_report = load_report(credit_report_path, "the credit model report")
    if credit_report:
        st.subheader(f"Credit default · {credit_report['model']}")
        st.warning(credit_report.get("warning", "Separate credit data; not part of the fraud benchmark."))
        credit_columns = st.columns(4)
        for column, (label, value) in zip(
            credit_columns,
            (
                ("Test PR-AUC", f"{credit_report['test_average_precision']:.3f}"),
                ("Test ROC-AUC", f"{credit_report['test_roc_auc']:.3f}"),
                ("Rows", f"{credit_report['rows']:,}"),
                ("Target", credit_report["target"]),
            ),
        ):
            column.metric(label, value)
        with st.expander("Credit report, class metrics, features"):
            st.json(credit_report)
    elif not credit_report_path.is_file():
        st.info("Credit-default model output is unavailable: no separately labeled credit dataset has been trained.")

    forecast_path = PROJECT_ROOT / "reports" / "forecast.json"
    forecast_report = load_report(forecast_path, "the forecast report")
    if forecast_report:
        st.subheader("Alert volume forecast")
        st.metric("Holdout MAE", f"{forecast_report['validation_mae']:.3f}")
        st.dataframe(pd.DataFrame(forecast_report["forecast"]), width="stretch", hide_index=True)
        st.caption(f"{forecast_report['method']} · {forecast_report['caveat']}")
    elif not forecast_path.is_file():
        st.info("Forecast output is unavailable: no timestamped daily alert history has been configured and evaluated.")

    rag_report = load_report(RAG_EVALUATION_PATH, "the RAG evaluation")
    if rag_report:
        st.subheader("Retrieval & investigation evaluation")
        st.caption(f"{rag_report['evaluation_type']} · {rag_report['scenarios']} scenarios")
        rag_metrics = rag_report["metrics"]
        metric_names = ("evidence_faithfulness", "citation_precision", "citation_recall", "missing_information_accuracy")
        rag_columns = st.columns(4)
        for column, name in zip(rag_columns, metric_names):
            value = rag_metrics.get(name)
            column.metric(name.replace("_", " ").title(), "N/A" if value is None else f"{value:.1%}")
        st.caption(
            f"Failure rate: {rag_metrics['failure_rate']:.1%} · "
            f"Mean local retrieval latency: {rag_metrics['mean_latency_ms']:.2f} ms · "
            "LLM provider cost is not measured by this offline fixture."
        )
        st.warning(
            "Evaluation scenarios and policy fixture are synthetic. Faithfulness checks evidence-ledger "
            "provenance; it is not a human assessment of LLM truthfulness."
        )
        with st.expander("RAG scenario-level results and limitations"):
            st.dataframe(pd.DataFrame(rag_report["results"]), width="stretch", hide_index=True)
            st.json(rag_report["limitations"])
    elif not RAG_EVALUATION_PATH.is_file():
        st.info("RAG evaluation output is not available. Run `python -m finguard.rag_evaluation`.")
