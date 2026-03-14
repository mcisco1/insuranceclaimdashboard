"""Statistical distribution helpers for synthetic claims generation.

All functions accept a ``numpy.random.Generator`` (not the legacy API) so
that every call chain is fully reproducible from a single seed.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd

from config import (
    LOSS_MAX,
    LOSS_MIN,
    LOSS_MU,
    LOSS_SIGMA,
    NEGATIVE_PAID_RATE,
    MIXED_CASE_STATE_RATE,
    TEXT_SEVERITY_RATE,
    SWAPPED_DATE_RATE,
    NULL_DIAGNOSIS_RATE,
    OUTLIER_COUNT,
)
from src.generation.reference_data import (
    HIGH_LITIGATION_STATES,
    SPECIALTY_SEVERITY_MAP,
)

# ── Severity ──────────────────────────────────────────────────────────────

def severity_for_specialty(specialty: str, rng: np.random.Generator) -> int:
    """Draw a severity score (1-5) using the specialty's weight profile.

    Falls back to a uniform distribution when the specialty is not in
    ``SPECIALTY_SEVERITY_MAP``.
    """
    weights = SPECIALTY_SEVERITY_MAP.get(specialty)
    if weights is None:
        return int(rng.integers(1, 6))
    return int(rng.choice([1, 2, 3, 4, 5], p=weights))


def severity_for_specialty_batch(
    specialties: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Vectorised severity draw for an array of specialty names."""
    severities = np.empty(len(specialties), dtype=np.int32)
    for i, spec in enumerate(specialties):
        severities[i] = severity_for_specialty(spec, rng)
    return severities


# ── Loss Amounts ──────────────────────────────────────────────────────────

_SEVERITY_LOSS_MULTIPLIER = {1: 0.4, 2: 0.7, 3: 1.0, 4: 1.6, 5: 2.5}


def lognormal_losses(
    n: int,
    severity: int | np.ndarray,
    rng: np.random.Generator,
    *,
    mu: float = LOSS_MU,
    sigma: float = LOSS_SIGMA,
) -> np.ndarray:
    """Generate *n* loss amounts from a log-normal distribution.

    The raw draws are scaled by a severity multiplier and clipped to
    [``LOSS_MIN``, ``LOSS_MAX``].

    Parameters
    ----------
    n : int
        Number of values to draw.
    severity : int or array of int
        Severity level(s) (1-5).  Scalar is broadcast; array must have
        length *n*.
    rng : numpy.random.Generator
        PRNG instance for reproducibility.
    mu, sigma : float
        Log-normal parameters (defaults from config).
    """
    raw = rng.lognormal(mean=mu, sigma=sigma, size=n)

    if np.isscalar(severity):
        mult = _SEVERITY_LOSS_MULTIPLIER.get(int(severity), 1.0)
    else:
        mult = np.array(
            [_SEVERITY_LOSS_MULTIPLIER.get(int(s), 1.0) for s in severity]
        )

    scaled = raw * mult
    return np.clip(scaled, LOSS_MIN, LOSS_MAX)


# ── Seasonal Modifier ────────────────────────────────────────────────────

def seasonal_modifier(date: pd.Timestamp | np.datetime64) -> float:
    """Return a multiplicative seasonal factor for a given date.

    The modifier follows a sinusoidal curve that peaks in winter
    (December-February) and troughs in summer (June-August), ranging
    approximately from 0.8 to 1.2.
    """
    if isinstance(date, np.datetime64):
        date = pd.Timestamp(date)
    day_of_year = date.day_of_year
    # Shift so the peak lands near Jan 15 (day ~15).
    phase = 2.0 * math.pi * (day_of_year - 15) / 365.0
    return 1.0 + 0.2 * math.cos(phase)


def seasonal_modifier_array(dates: pd.Series) -> np.ndarray:
    """Vectorised seasonal modifier for a Series of datetime values."""
    day_of_year = dates.dt.day_of_year.values.astype(float)
    phase = 2.0 * np.pi * (day_of_year - 15.0) / 365.0
    return 1.0 + 0.2 * np.cos(phase)


# ── Reporting Lag ─────────────────────────────────────────────────────────

def days_to_report(
    severity: int | np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Reporting lag in days (log-normal, median ~7 days).

    Higher severity claims tend to be reported slightly faster because
    they are more conspicuous.
    """
    n = 1 if np.isscalar(severity) else len(severity)
    # Base: log-normal with median exp(mu) ≈ 7 days
    mu_base = math.log(7.0)
    sigma = 0.8

    if np.isscalar(severity):
        # Faster reporting for higher severity (reduce mu slightly)
        mu = mu_base - 0.08 * (int(severity) - 1)
        days = rng.lognormal(mean=mu, sigma=sigma, size=1)
    else:
        sev = np.asarray(severity, dtype=float)
        mu = mu_base - 0.08 * (sev - 1.0)
        days = np.array(
            [rng.lognormal(mean=m, sigma=sigma) for m in mu]
        )

    return np.maximum(np.round(days), 0).astype(int)


# ── Closure Lag ───────────────────────────────────────────────────────────

_SEVERITY_CLOSE_MEDIAN = {1: 30, 2: 60, 3: 120, 4: 210, 5: 300}


def days_to_close(
    severity: int | np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Days from report to closure (log-normal, correlated with severity).

    Severity 1 → median ~30 days, severity 5 → median ~300 days.
    """
    sigma = 0.6

    if np.isscalar(severity):
        median = _SEVERITY_CLOSE_MEDIAN.get(int(severity), 120)
        mu = math.log(median)
        days = rng.lognormal(mean=mu, sigma=sigma, size=1)
    else:
        days_list: List[float] = []
        for s in severity:
            median = _SEVERITY_CLOSE_MEDIAN.get(int(s), 120)
            mu = math.log(median)
            days_list.append(rng.lognormal(mean=mu, sigma=sigma))
        days = np.array(days_list)

    return np.maximum(np.round(days), 1).astype(int)


# ── Litigation Probability ───────────────────────────────────────────────

def litigation_probability(
    severity: int, state_code: str
) -> float:
    """Probability that a claim enters litigation.

    Base probability ramps with severity.  High-litigation states
    receive an additive bump.
    """
    base = {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.40, 5: 0.60}.get(
        int(severity), 0.15
    )
    if state_code in HIGH_LITIGATION_STATES:
        base += 0.15
    return min(base, 1.0)


def litigation_flags(
    severities: np.ndarray,
    state_codes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorised litigation flag (bool) for arrays of severity / state."""
    probs = np.array(
        [
            litigation_probability(int(s), sc)
            for s, sc in zip(severities, state_codes)
        ]
    )
    return rng.random(len(probs)) < probs


# ── Data-Quality Issue Injection ──────────────────────────────────────────

_SEVERITY_TEXT = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def inject_quality_issues(
    df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Mutate *df* **in place** to embed realistic data-quality problems.

    The function returns *df* for convenience but modifies the original.

    Issues injected (approximate rates pulled from ``config.py``):
    * Negative paid amounts
    * Mixed-case state codes (e.g. 'Ny')
    * Severity stored as English text instead of int
    * ``report_date`` set before ``incident_date`` (swapped)
    * NULL diagnosis
    * A handful of extreme outlier losses (> $10 M)
    """
    n = len(df)
    idx = df.index.to_numpy()

    # 1. Negative paid amounts (~2%)
    mask = rng.random(n) < NEGATIVE_PAID_RATE
    neg_idx = idx[mask]
    if len(neg_idx) > 0 and "paid_amount" in df.columns:
        df.loc[neg_idx, "paid_amount"] = -df.loc[neg_idx, "paid_amount"].abs()

    # 2. Mixed-case state codes (~3%)
    mask = rng.random(n) < MIXED_CASE_STATE_RATE
    mc_idx = idx[mask]
    if len(mc_idx) > 0 and "state_code" in df.columns:
        df.loc[mc_idx, "state_code"] = df.loc[mc_idx, "state_code"].apply(
            lambda s: s[0].upper() + s[1:].lower() if isinstance(s, str) and len(s) == 2 else s
        )

    # 3. Severity as text (~1.5%)
    mask = rng.random(n) < TEXT_SEVERITY_RATE
    txt_idx = idx[mask]
    if len(txt_idx) > 0 and "severity" in df.columns:
        df["severity"] = df["severity"].astype(object)  # allow mixed types
        df.loc[txt_idx, "severity"] = df.loc[txt_idx, "severity"].map(
            lambda x: _SEVERITY_TEXT.get(int(x), str(x)) if pd.notna(x) else x
        )

    # 4. report_date before incident_date (~2%)
    mask = rng.random(n) < SWAPPED_DATE_RATE
    swap_idx = idx[mask]
    if (
        len(swap_idx) > 0
        and "report_date" in df.columns
        and "incident_date" in df.columns
    ):
        orig_report = df.loc[swap_idx, "report_date"].copy()
        df.loc[swap_idx, "report_date"] = df.loc[swap_idx, "incident_date"] - pd.to_timedelta(
            rng.integers(1, 30, size=len(swap_idx)), unit="D"
        )

    # 5. NULL diagnosis (~4%)
    mask = rng.random(n) < NULL_DIAGNOSIS_RATE
    null_idx = idx[mask]
    if len(null_idx) > 0 and "diagnosis_id" in df.columns:
        df.loc[null_idx, "diagnosis_id"] = np.nan

    # 6. Outliers at $10M+ (exactly OUTLIER_COUNT)
    if "paid_amount" in df.columns and OUTLIER_COUNT > 0:
        outlier_idx = rng.choice(idx, size=min(OUTLIER_COUNT, n), replace=False)
        df.loc[outlier_idx, "paid_amount"] = rng.uniform(
            10_000_000, 25_000_000, size=len(outlier_idx)
        )

    return df
