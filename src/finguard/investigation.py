"""Evidence-first investigation workflow; generated text is only a review draft."""

from __future__ import annotations

from finguard.rag import ComplianceRetriever


def investigate(
    transaction: dict,
    retriever: ComplianceRetriever,
    question: str = "",
    synthetic_context: dict | None = None,
) -> dict:
    score = float(transaction["fraud_score"])
    threshold = float(transaction["threshold"])
    sources = retriever.search(question or "transaction fraud review and escalation")
    evidence = [
        {
            "claim": f"Fraud model score is {score:.4f}; configured review threshold is {threshold:.4f}.",
            "source_type": "model_output",
            "source_ref": str(transaction.get("transaction_id", "scored_transaction")),
            "is_synthetic": False,
        },
        {
            "claim": "Model output is a review signal and does not establish fraud.",
            "source_type": "workflow_policy",
            "source_ref": "human-review-required",
            "is_synthetic": False,
        },
    ]
    if transaction.get("anomaly_score") is not None:
        evidence.append(
            {
                "claim": (
                    f"Anomaly score is {float(transaction['anomaly_score']):.4f}; "
                    "it is a separate unusualness signal, not fraud probability."
                ),
                "source_type": "anomaly_model_output",
                "source_ref": str(transaction.get("transaction_id", "scored_transaction")),
                "is_synthetic": False,
            }
        )
    if synthetic_context is not None:
        if synthetic_context.get("data_classification") != "SYNTHETIC_DEMO_ONLY":
            raise ValueError("Investigation context must be explicitly classified as synthetic demo data.")
        evidence.extend(
            [
                {
                    "claim": (
                        f"Synthetic customer {synthetic_context['synthetic_customer_id']} has "
                        f"{synthetic_context['historical_transaction_count']} fictional prior transactions."
                    ),
                    "source_type": "synthetic_demo_context",
                    "source_ref": synthetic_context["scenario_id"],
                    "is_synthetic": True,
                },
                {
                    "claim": (
                        "Fictional transaction amount is "
                        f"{synthetic_context['amount_vs_history_ratio']:.2f} times the fictional "
                        "historical median amount."
                    ),
                    "source_type": "synthetic_demo_context",
                    "source_ref": synthetic_context["scenario_id"],
                    "is_synthetic": True,
                },
                {
                    "claim": (
                        "Fictional destination was previously observed."
                        if synthetic_context["destination_seen_before"]
                        else "Fictional destination was not previously observed."
                    ),
                    "source_type": "synthetic_demo_context",
                    "source_ref": synthetic_context["scenario_id"],
                    "is_synthetic": True,
                },
            ]
        )
    if not sources:
        missing_information = [
            "No relevant compliance source was found in the configured document library.",
            "The benchmark dataset contains no real customer history, merchant, destination, or customer identifier.",
        ]
    else:
        missing_information = [
            "Retrieved documents are reference material and do not establish facts about this transaction."
        ]
    draft = " ".join(
        [
            f"Transaction {transaction.get('transaction_id', 'unknown')} has a fraud score of {score:.4f} "
            f"against a review threshold of {threshold:.4f}.",
            "Analyst review is required; this score is not a finding of fraud.",
        ]
    )
    if synthetic_context is not None:
        draft += (
            " A separate synthetic demonstration scenario is attached for UI illustration only; "
            "it was not used by the fraud model."
        )
    if not sources:
        draft += " No compliance policy conclusion is available because no relevant source was retrieved."
    return {
        "status": "analyst_review_required",
        "draft": draft,
        "evidence": evidence,
        "missing_information": missing_information,
        "data_lineage": {
            "model_transaction_source": transaction.get("transaction_source", "creditcard.csv benchmark"),
            "synthetic_context_attached": synthetic_context is not None,
            "synthetic_context_used_for_model_score": False,
        },
        "sources": [
            {"source": source["source"], "excerpt": source["text"], "relevance": source["score"]}
            for source in sources
        ],
        "decision": None,
    }
