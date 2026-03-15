"""Monte Carlo simulation for scenario analysis.

Replaces deterministic what-if scenarios with probabilistic distributions
of potential savings, using Beta-distributed parameter uncertainty.

Usage
-----
    from src.analysis.monte_carlo import monte_carlo_scenario
    result = monte_carlo_scenario(df, sev_pct=20, lit_pct=25, close_pct=20)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import RANDOM_SEED

logger = logging.getLogger(__name__)


def monte_carlo_scenario(
    df: pd.DataFrame,
    sev_pct: float = 20,
    lit_pct: float = 25,
    close_pct: float = 20,
    n_sim: int = 5000,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Run Monte Carlo simulation for combined scenario savings.

    Instead of assuming fixed percentage reductions, each simulation
    draw samples parameter uncertainty from Beta distributions centered
    on the specified reduction percentages.

    Parameters
    ----------
    df : pd.DataFrame
        Claims summary DataFrame with *paid_amount*, *incurred_amount*,
        *severity_level*, *litigation_flag*, *days_to_close*.
    sev_pct : float
        Target severity reduction percentage (high -> medium shift).
    lit_pct : float
        Target litigation rate reduction percentage.
    close_pct : float
        Target close time reduction percentage.
    n_sim : int
        Number of Monte Carlo simulations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``mean_savings``, ``median_savings``, ``p5``, ``p95``,
        ``prob_positive``, ``distribution``, ``var_95``.
    """
    rng = np.random.RandomState(seed)

    if df.empty:
        return _empty_result()

    # Baseline metrics
    total_incurred = pd.to_numeric(df.get("incurred_amount", pd.Series(dtype=float)), errors="coerce").sum()
    total_paid = pd.to_numeric(df.get("paid_amount", pd.Series(dtype=float)), errors="coerce").sum()
    baseline = max(total_incurred, total_paid, 1.0)

    # High severity cost differential
    sev_col = pd.to_numeric(df.get("severity_level", pd.Series(dtype=float)), errors="coerce")
    paid_col = pd.to_numeric(df.get("paid_amount", pd.Series(dtype=float)), errors="coerce").fillna(0)
    high_sev_mask = sev_col >= 4
    high_sev_paid = paid_col[high_sev_mask].sum()
    med_sev_avg = paid_col[(sev_col >= 2) & (sev_col <= 3)].mean()
    high_sev_avg = paid_col[high_sev_mask].mean()

    if pd.isna(med_sev_avg):
        med_sev_avg = 0
    if pd.isna(high_sev_avg) or high_sev_avg == 0:
        high_sev_avg = 1

    sev_savings_per_claim = max(high_sev_avg - med_sev_avg, 0)
    n_high_sev = high_sev_mask.sum()

    # Litigation cost
    lit_col = pd.to_numeric(df.get("litigation_flag", pd.Series(dtype=float)), errors="coerce").fillna(0)
    lit_rate = lit_col.mean()
    lit_claims = lit_col.sum()
    lit_cost_premium = paid_col[lit_col == 1].mean() - paid_col[lit_col == 0].mean()
    if pd.isna(lit_cost_premium):
        lit_cost_premium = 0

    # Close time cost proxy
    close_col = pd.to_numeric(df.get("days_to_close", pd.Series(dtype=float)), errors="coerce")
    avg_close = close_col.mean()
    if pd.isna(avg_close) or avg_close == 0:
        avg_close = 100

    # Simulate
    savings_dist = np.zeros(n_sim)

    for i in range(n_sim):
        # Sample reduction parameters from Beta distributions
        # Beta(a, b) centered around target pct/100 with some spread
        if sev_pct > 0:
            a_sev = max(sev_pct / 100 * 10, 1)
            b_sev = max((1 - sev_pct / 100) * 10, 1)
            sev_reduction = rng.beta(a_sev, b_sev)
        else:
            sev_reduction = 0.0

        if lit_pct > 0:
            a_lit = max(lit_pct / 100 * 10, 1)
            b_lit = max((1 - lit_pct / 100) * 10, 1)
            lit_reduction = rng.beta(a_lit, b_lit)
        else:
            lit_reduction = 0.0

        if close_pct > 0:
            a_close = max(close_pct / 100 * 10, 1)
            b_close = max((1 - close_pct / 100) * 10, 1)
            close_reduction = rng.beta(a_close, b_close)
        else:
            close_reduction = 0.0

        # Severity shift savings
        n_shifted = int(n_high_sev * sev_reduction)
        sev_saving = n_shifted * sev_savings_per_claim

        # Litigation reduction savings
        claims_removed = lit_claims * lit_reduction
        lit_saving = claims_removed * max(lit_cost_premium, 0)

        # Close time savings (proxy: reduced ALAE/expenses proportional to days)
        close_saving = baseline * (close_reduction * 0.05)  # ~5% of incurred is close-related

        savings_dist[i] = sev_saving + lit_saving + close_saving

    mean_savings = float(np.mean(savings_dist))
    median_savings = float(np.median(savings_dist))
    p5 = float(np.percentile(savings_dist, 5))
    p95 = float(np.percentile(savings_dist, 95))
    prob_positive = float((savings_dist > 0).mean() * 100)
    var_95 = float(np.percentile(savings_dist, 5))  # VaR at 95% confidence = 5th percentile

    logger.info(
        "Monte Carlo: mean=$%.0f, median=$%.0f, P(>0)=%.1f%%, 95%% VaR=$%.0f",
        mean_savings, median_savings, prob_positive, var_95,
    )

    return {
        "mean_savings": mean_savings,
        "median_savings": median_savings,
        "p5": p5,
        "p95": p95,
        "prob_positive": prob_positive,
        "distribution": savings_dist.tolist(),
        "var_95": var_95,
    }


def _empty_result() -> Dict[str, Any]:
    return {
        "mean_savings": 0.0,
        "median_savings": 0.0,
        "p5": 0.0,
        "p95": 0.0,
        "prob_positive": 0.0,
        "distribution": [],
        "var_95": 0.0,
    }
