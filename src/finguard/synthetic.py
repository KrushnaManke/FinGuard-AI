"""Deterministic, explicitly fictional customer context for UI demonstrations."""

import hashlib

import numpy as np


def make_synthetic_context(transaction_id: str, amount: float) -> dict:
    """Create illustrative customer-history facts that never enter the ML pipeline."""
    digest = hashlib.sha256(transaction_id.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big")
    generator = np.random.default_rng(seed)
    historical_median = float(np.round(generator.lognormal(mean=3.5, sigma=0.8), 2))
    history_count = int(generator.integers(20, 250))
    destination_seen = bool(generator.integers(0, 2))
    return {
        "data_classification": "SYNTHETIC_DEMO_ONLY",
        "scenario_id": f"SYN-{seed % 1_000_000:06d}",
        "synthetic_customer_id": f"SYN-CUST-{seed % 100_000:05d}",
        "source": "deterministic synthetic scenario generator",
        "historical_transaction_count": history_count,
        "historical_median_amount": historical_median,
        "current_amount": float(amount),
        "amount_vs_history_ratio": float(amount / historical_median) if historical_median else None,
        "destination_seen_before": destination_seen,
        "notice": (
            "All customer and history fields in this context are fictional demo data. "
            "The fraud model was trained without these fields; they were not used to calculate its score."
        ),
    }
