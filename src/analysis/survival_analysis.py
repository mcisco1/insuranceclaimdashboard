"""Kaplan-Meier survival analysis for claim time-to-close.

Handles open claims as right-censored observations, which is the
actuarially correct treatment. Provides overall KM curve, curves
stratified by claim type, and log-rank tests for significance.

Usage
-----
    from src.analysis.survival_analysis import analyze_survival
    results = analyze_survival(df)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_survival(df: pd.DataFrame) -> Dict[str, Any]:
    """Fit Kaplan-Meier survival curves for time-to-close.

    Open claims are treated as right-censored: their duration is
    measured from report_date to today, and the event (close) has
    not yet occurred.

    Parameters
    ----------
    df : pd.DataFrame
        Claims summary DataFrame with columns: *status*, *days_to_close*,
        *report_date* (or *incident_date*), *claim_type*.

    Returns
    -------
    dict
        Keys: ``overall_curve``, ``curves_by_type``, ``logrank_tests``,
        ``median_survival_time``.
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        logger.warning("lifelines not installed; survival analysis skipped.")
        return _empty_result()

    if df.empty:
        return _empty_result()

    work = df.copy()

    # Build duration and event columns
    work["_status_lower"] = work["status"].astype(str).str.lower()
    work["event_observed"] = (work["_status_lower"] == "closed").astype(int)

    # Duration: days_to_close for closed; days since report_date for open
    work["days_to_close"] = pd.to_numeric(work["days_to_close"], errors="coerce")

    today = pd.Timestamp.now().normalize()
    for date_col in ("report_date", "incident_date"):
        if date_col in work.columns:
            work[date_col] = pd.to_datetime(work[date_col], errors="coerce")

    if "report_date" in work.columns:
        ref_date = work["report_date"]
    elif "incident_date" in work.columns:
        ref_date = work["incident_date"]
    else:
        ref_date = pd.Series(pd.NaT, index=work.index)

    open_duration = (today - ref_date).dt.days
    work["duration"] = np.where(
        work["event_observed"] == 1,
        work["days_to_close"],
        open_duration,
    )
    work["duration"] = pd.to_numeric(work["duration"], errors="coerce")
    work = work.dropna(subset=["duration"])
    work = work[work["duration"] > 0]

    if work.empty:
        return _empty_result()

    T = work["duration"].values
    E = work["event_observed"].values

    # Overall KM curve
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E, label="Overall")

    overall_curve = {
        "timeline": kmf.survival_function_.index.tolist(),
        "survival": kmf.survival_function_.iloc[:, 0].tolist(),
        "ci_lower": kmf.confidence_interval_survival_function_.iloc[:, 0].tolist(),
        "ci_upper": kmf.confidence_interval_survival_function_.iloc[:, 1].tolist(),
    }

    median_survival = float(kmf.median_survival_time_) if not np.isinf(kmf.median_survival_time_) else None

    # Curves by claim type
    curves_by_type = {}
    claim_types = []
    if "claim_type" in work.columns:
        claim_types = sorted(work["claim_type"].dropna().unique())
        for ct in claim_types:
            mask = work["claim_type"] == ct
            if mask.sum() < 5:
                continue
            kmf_ct = KaplanMeierFitter()
            kmf_ct.fit(T[mask], event_observed=E[mask], label=ct)
            curves_by_type[ct] = {
                "timeline": kmf_ct.survival_function_.index.tolist(),
                "survival": kmf_ct.survival_function_.iloc[:, 0].tolist(),
                "ci_lower": kmf_ct.confidence_interval_survival_function_.iloc[:, 0].tolist(),
                "ci_upper": kmf_ct.confidence_interval_survival_function_.iloc[:, 1].tolist(),
                "median": float(kmf_ct.median_survival_time_) if not np.isinf(kmf_ct.median_survival_time_) else None,
            }

    # Log-rank tests between claim types
    logrank_tests = {}
    type_list = list(curves_by_type.keys())
    for i in range(len(type_list)):
        for j in range(i + 1, len(type_list)):
            ct_a, ct_b = type_list[i], type_list[j]
            mask_a = work["claim_type"] == ct_a
            mask_b = work["claim_type"] == ct_b
            try:
                result = logrank_test(
                    T[mask_a], T[mask_b],
                    event_observed_A=E[mask_a],
                    event_observed_B=E[mask_b],
                )
                logrank_tests[f"{ct_a} vs {ct_b}"] = {
                    "test_statistic": float(result.test_statistic),
                    "p_value": float(result.p_value),
                    "significant": result.p_value < 0.05,
                }
            except Exception:
                pass

    logger.info(
        "Survival analysis: %d observations, median=%.0f days, %d claim types",
        len(work),
        median_survival or 0,
        len(curves_by_type),
    )

    return {
        "overall_curve": overall_curve,
        "curves_by_type": curves_by_type,
        "logrank_tests": logrank_tests,
        "median_survival_time": median_survival,
    }


def _empty_result() -> Dict[str, Any]:
    return {
        "overall_curve": {},
        "curves_by_type": {},
        "logrank_tests": {},
        "median_survival_time": None,
    }
