"""Patient-safety benchmarking (AHRQ PSI-style indicators).

Computes facility-level patient safety indicators, compares them against
dataset-wide averages ("national benchmarks"), and tracks quarterly trends.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── PSI indicator definitions ────────────────────────────────────────────
# Each indicator is a (label, numerator_filter, denominator_filter) tuple.
# Filters are callables that accept a DataFrame and return a boolean mask.

def _surgical_complication_num(df: pd.DataFrame) -> pd.Series:
    """Surgical procedures with severity >= 4."""
    is_surgical = df["procedure_category"].str.contains(
        "Surg", case=False, na=False
    )
    high_sev = pd.to_numeric(df["severity_level"], errors="coerce") >= 4
    return is_surgical & high_sev


def _surgical_complication_den(df: pd.DataFrame) -> pd.Series:
    """All surgical procedures."""
    return df["procedure_category"].str.contains("Surg", case=False, na=False)


def _fall_num(df: pd.DataFrame) -> pd.Series:
    """Claims with root_cause containing 'Fall'."""
    if "root_cause_category" in df.columns:
        return df["root_cause_category"].str.contains("Fall", case=False, na=False)
    if "diagnosis_category" in df.columns:
        return df["diagnosis_category"].str.contains("Fall", case=False, na=False)
    return pd.Series(False, index=df.index)


def _medication_error_num(df: pd.DataFrame) -> pd.Series:
    """Claims with root_cause or diagnosis involving medication error."""
    mask = pd.Series(False, index=df.index)
    if "root_cause_category" in df.columns:
        mask = mask | df["root_cause_category"].str.contains(
            "Medication", case=False, na=False
        )
    if "diagnosis_category" in df.columns:
        mask = mask | df["diagnosis_category"].str.contains(
            "Medication", case=False, na=False
        )
    return mask


def _diagnostic_error_num(df: pd.DataFrame) -> pd.Series:
    """Claims with diagnostic error root cause or misdiagnosis."""
    mask = pd.Series(False, index=df.index)
    if "root_cause_category" in df.columns:
        mask = mask | df["root_cause_category"].str.contains(
            "Diagnostic", case=False, na=False
        )
    if "diagnosis_category" in df.columns:
        mask = mask | df["diagnosis_category"].str.contains(
            "Misdiagnos", case=False, na=False
        )
    return mask


def _hai_num(df: pd.DataFrame) -> pd.Series:
    """Hospital-acquired infection indicators."""
    mask = pd.Series(False, index=df.index)
    if "diagnosis_category" in df.columns:
        mask = mask | df["diagnosis_category"].str.contains(
            "Infection|Sepsis|Pneumonia|Catheter", case=False, na=False
        )
    return mask


# Full indicator registry
_INDICATORS: List[Tuple[str, Any, Any]] = [
    ("surgical_complication_rate", _surgical_complication_num, _surgical_complication_den),
    ("fall_rate", _fall_num, None),
    ("medication_error_rate", _medication_error_num, None),
    ("diagnostic_error_rate", _diagnostic_error_num, None),
    ("hospital_acquired_infection_rate", _hai_num, None),
]


def calculate_benchmarks(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate AHRQ PSI-style benchmarks and facility comparisons.

    Parameters
    ----------
    df : pd.DataFrame
        Claims summary DataFrame.  Expected columns include
        *procedure_category*, *severity_level*, *root_cause_category*,
        *diagnosis_category*, *department*, *incident_date*.

    Returns
    -------
    dict
        Keys: psi_indicators, facility_vs_benchmark, benchmark_trends.
    """
    if df.empty:
        return _empty_result()

    df = df.copy()
    df["severity_level"] = pd.to_numeric(df.get("severity_level"), errors="coerce")
    df["incident_date"] = pd.to_datetime(df.get("incident_date"), errors="coerce")

    result: Dict[str, Any] = {}

    # ── PSI indicators (overall) ─────────────────────────────────────────
    psi: Dict[str, Dict[str, Any]] = {}
    total_claims = len(df)

    for name, num_fn, den_fn in _INDICATORS:
        try:
            numerator_mask = num_fn(df)
            numerator = int(numerator_mask.sum())

            if den_fn is not None:
                denominator_mask = den_fn(df)
                denominator = int(denominator_mask.sum())
            else:
                denominator = total_claims

            rate = round(numerator / denominator * 1000, 2) if denominator > 0 else 0.0
            psi[name] = {
                "numerator": numerator,
                "denominator": denominator,
                "rate_per_1000": rate,
            }
        except Exception:
            psi[name] = {
                "numerator": 0, "denominator": 0, "rate_per_1000": 0.0,
            }

    result["psi_indicators"] = psi

    # ── Facility vs benchmark (by department) ────────────────────────────
    facility_col = "department"
    if facility_col not in df.columns:
        result["facility_vs_benchmark"] = {}
    else:
        facility_benchmark: Dict[str, Dict[str, Any]] = {}

        for dept, dept_df in df.groupby(facility_col):
            dept_metrics: Dict[str, Any] = {"claim_count": len(dept_df)}

            for name, num_fn, den_fn in _INDICATORS:
                try:
                    num_mask = num_fn(dept_df)
                    num = int(num_mask.sum())

                    if den_fn is not None:
                        den_mask = den_fn(dept_df)
                        den = int(den_mask.sum())
                    else:
                        den = len(dept_df)

                    dept_rate = round(num / den * 1000, 2) if den > 0 else 0.0

                    # National benchmark = overall rate
                    national_rate = psi[name]["rate_per_1000"]

                    # Variance from benchmark
                    if national_rate > 0:
                        variance_pct = round(
                            (dept_rate - national_rate) / national_rate * 100, 2
                        )
                    else:
                        variance_pct = 0.0

                    dept_metrics[name] = {
                        "dept_rate_per_1000": dept_rate,
                        "benchmark_rate_per_1000": national_rate,
                        "variance_pct": variance_pct,
                        "status": (
                            "above_benchmark"
                            if dept_rate > national_rate * 1.1
                            else (
                                "below_benchmark"
                                if dept_rate < national_rate * 0.9
                                else "at_benchmark"
                            )
                        ),
                    }
                except Exception:
                    dept_metrics[name] = {
                        "dept_rate_per_1000": 0.0,
                        "benchmark_rate_per_1000": 0.0,
                        "variance_pct": 0.0,
                        "status": "at_benchmark",
                    }

            facility_benchmark[str(dept)] = dept_metrics

        result["facility_vs_benchmark"] = facility_benchmark

    # ── Quarterly PSI trends ─────────────────────────────────────────────
    dated = df.dropna(subset=["incident_date"]).copy()
    if not dated.empty:
        dated["quarter"] = dated["incident_date"].dt.to_period("Q")
        quarters = sorted(dated["quarter"].unique())

        trends: Dict[str, Dict[str, float]] = {name: {} for name, _, _ in _INDICATORS}

        for q in quarters:
            q_df = dated[dated["quarter"] == q]
            q_total = len(q_df)
            q_str = str(q)

            for name, num_fn, den_fn in _INDICATORS:
                try:
                    num = int(num_fn(q_df).sum())
                    if den_fn is not None:
                        den = int(den_fn(q_df).sum())
                    else:
                        den = q_total

                    rate = round(num / den * 1000, 2) if den > 0 else 0.0
                    trends[name][q_str] = rate
                except Exception:
                    trends[name][q_str] = 0.0

        result["benchmark_trends"] = trends
    else:
        result["benchmark_trends"] = {name: {} for name, _, _ in _INDICATORS}

    return result


def _empty_result() -> Dict[str, Any]:
    return {
        "psi_indicators": {
            name: {"numerator": 0, "denominator": 0, "rate_per_1000": 0.0}
            for name, _, _ in _INDICATORS
        },
        "facility_vs_benchmark": {},
        "benchmark_trends": {name: {} for name, _, _ in _INDICATORS},
    }
