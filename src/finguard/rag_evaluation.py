"""Offline evaluation for source retrieval and deterministic investigation evidence."""

from __future__ import annotations

import json
import time
from pathlib import Path

from finguard.investigation import investigate
from finguard.rag import ComplianceRetriever
from finguard.synthetic import make_synthetic_context


def _claim_is_supported(claim: dict, transaction: dict, synthetic_context: dict) -> bool:
    source_type = claim["source_type"]
    text = claim["claim"]
    if source_type == "model_output":
        return (
            f"{float(transaction['fraud_score']):.4f}" in text
            and f"{float(transaction['threshold']):.4f}" in text
            and claim["source_ref"] == transaction["transaction_id"]
            and not claim["is_synthetic"]
        )
    if source_type == "anomaly_model_output":
        return (
            f"{float(transaction['anomaly_score']):.4f}" in text
            and claim["source_ref"] == transaction["transaction_id"]
            and not claim["is_synthetic"]
        )
    if source_type == "workflow_policy":
        return (
            text == "Model output is a review signal and does not establish fraud."
            and claim["source_ref"] == "human-review-required"
            and not claim["is_synthetic"]
        )
    if source_type == "synthetic_demo_context":
        return (
            claim["is_synthetic"]
            and claim["source_ref"] == synthetic_context["scenario_id"]
            and synthetic_context["data_classification"] == "SYNTHETIC_DEMO_ONLY"
            and (
                str(synthetic_context["historical_transaction_count"]) in text
                or f"{synthetic_context['amount_vs_history_ratio']:.2f}" in text
                or (
                    "previously observed" in text
                    and (
                        ("not previously observed" in text)
                        == (not synthetic_context["destination_seen_before"])
                    )
                )
            )
        )
    return False


def evaluate_scenarios(scenarios_path: str | Path, documents_dir: str | Path) -> dict:
    scenarios = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("RAG evaluation scenarios must be a non-empty JSON list.")
    retriever = ComplianceRetriever(documents_dir, backend="tfidf")
    results = []
    failures = 0
    supported_claims = 0
    total_claims = 0
    expected_citations = 0
    correct_citations = 0
    retrieved_citations = 0
    missing_cases = 0
    correctly_missing = 0

    for scenario in scenarios:
        started = time.perf_counter()
        try:
            question = scenario["query"]
            expected_sources = set(scenario.get("expected_sources", []))
            transaction = {
                "transaction_id": scenario["scenario_id"],
                "transaction_source": "synthetic evaluation fixture",
                "fraud_score": 0.82,
                "anomaly_score": 0.31,
                "threshold": 0.75,
            }
            synthetic_context = make_synthetic_context(scenario["scenario_id"], amount=425.0)
            report = investigate(
                transaction,
                retriever,
                question,
                synthetic_context,
            )
            returned_sources = {source["source"] for source in report["sources"]}
            claims = report["evidence"]
            claim_support = [
                _claim_is_supported(claim, transaction, synthetic_context) for claim in claims
            ]
            supported_claims += sum(claim_support)
            total_claims += len(claim_support)
            expected_citations += len(expected_sources)
            correct_citations += len(returned_sources & expected_sources)
            retrieved_citations += len(returned_sources)
            expects_missing = bool(scenario.get("expect_missing", not expected_sources))
            if expects_missing:
                missing_cases += 1
                correctly_missing += int(
                    not returned_sources
                    and bool(report["missing_information"])
                    and "no relevant source was retrieved" in report["draft"].lower()
                )
            result = {
                "scenario_id": scenario["scenario_id"],
                "status": "ok",
                "expected_sources": sorted(expected_sources),
                "returned_sources": sorted(returned_sources),
                "citation_correct": (
                    bool(returned_sources & expected_sources) if expected_sources else not returned_sources
                ),
                "missing_information_handled": (
                    not returned_sources
                    and bool(report["missing_information"])
                    and "no relevant source was retrieved" in report["draft"].lower()
                    if expects_missing
                    else None
                ),
                "evidence_claims": len(claims),
                "supported_evidence_claims": sum(claim_support),
            }
        except Exception as error:
            failures += 1
            result = {
                "scenario_id": scenario.get("scenario_id", "unknown"),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        result["latency_ms"] = (time.perf_counter() - started) * 1000
        results.append(result)

    total_scenarios = len(scenarios)
    return {
        "evaluation_type": "offline deterministic TF-IDF retrieval and evidence-ledger evaluation",
        "synthetic_scenarios": True,
        "scenarios": total_scenarios,
        "metrics": {
            "evidence_faithfulness": supported_claims / total_claims if total_claims else 0.0,
            "citation_precision": correct_citations / retrieved_citations if retrieved_citations else (
                1.0 if expected_citations == 0 else 0.0
            ),
            "citation_recall": correct_citations / expected_citations if expected_citations else 1.0,
            "missing_information_accuracy": correctly_missing / missing_cases if missing_cases else None,
            "failure_rate": failures / total_scenarios,
            "mean_latency_ms": sum(item["latency_ms"] for item in results) / total_scenarios,
            "estimated_llm_cost_usd": 0.0,
        },
        "limitations": [
            "Scenario corpus and policy fixture are synthetic and are not legal or compliance guidance.",
            "Faithfulness measures provenance completeness of the deterministic evidence ledger, not human factuality judgments.",
            "LLM latency and cost are not measured because no LLM is called by this offline evaluation.",
        ],
        "results": results,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="data/evaluation/rag_scenarios.json")
    parser.add_argument("--documents", default="data/evaluation/rag_fixtures")
    parser.add_argument("--output", default="reports/rag_evaluation.json")
    args = parser.parse_args()
    report = evaluate_scenarios(args.scenarios, args.documents)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
