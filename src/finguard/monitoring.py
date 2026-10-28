"""Lightweight data-drift metrics for monitoring incoming feature batches."""

import numpy as np
import pandas as pd


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate PSI using reference quantile bins; smooth empty bins."""
    if bins < 2:
        raise ValueError("At least two bins are required.")
    reference = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Reference and current samples must both contain numeric values.")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        raise ValueError("Reference feature has too few distinct values for drift analysis.")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, bins=edges)[0].astype(float) + 1e-6
    cur_counts = np.histogram(current, bins=edges)[0].astype(float) + 1e-6
    ref_share, cur_share = ref_counts / ref_counts.sum(), cur_counts / cur_counts.sum()
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))
