"""Statistical utility functions: bootstrap confidence intervals and Wilson score intervals.

Provides non-parametric bootstrap for arbitrary statistics and Wilson
score intervals for proportions, used across KPI cards and analysis modules.

Usage
-----
    from src.analysis.statistical_utils import bootstrap_ci, proportional_ci
    result = bootstrap_ci(data, statistic_fn=np.mean, n_boot=1000, ci=0.95)
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    data : array-like
        1-D array of observations.
    statistic_fn : callable
        Function that takes an array and returns a scalar (e.g. ``np.mean``).
    n_boot : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g. 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``estimate``, ``ci_lower``, ``ci_upper``, ``margin_of_error``.
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "margin_of_error": 0.0}

    rng = np.random.RandomState(seed)
    estimate = float(statistic_fn(data))

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats[i] = statistic_fn(sample)

    alpha = 1 - ci
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    margin = (ci_upper - ci_lower) / 2

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "margin_of_error": margin,
    }


def proportional_ci(
    successes: int,
    trials: int,
    ci: float = 0.95,
) -> Dict[str, Any]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because it handles edge cases
    (p near 0 or 1, small samples) without producing impossible bounds.

    Parameters
    ----------
    successes : int
        Number of successes (e.g. litigation count).
    trials : int
        Total number of trials (e.g. total claims).
    ci : float
        Confidence level.

    Returns
    -------
    dict
        Keys: ``estimate``, ``ci_lower``, ``ci_upper``, ``margin_of_error``.
    """
    if trials == 0:
        return {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "margin_of_error": 0.0}

    from scipy.stats import norm

    z = norm.ppf(1 - (1 - ci) / 2)
    p_hat = successes / trials
    n = trials

    denominator = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denominator
    margin = (z / denominator) * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))

    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)

    return {
        "estimate": float(p_hat),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "margin_of_error": float(margin),
    }
