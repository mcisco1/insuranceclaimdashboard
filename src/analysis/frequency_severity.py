"""Frequency and severity analysis for insurance claims.

Accepts a claims-summary DataFrame (from v_claims_summary) and returns
a dict of analytic results the dashboard can consume directly.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from config import SLA_THRESHOLDS


def analyze_frequency_severity(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute frequency/severity breakdowns and loss-ratio metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Claims summary DataFrame (one row per claim).  Expected columns
        include *incident_date*, *claim_type*, *severity_level*, *specialty*,
        *paid_amount*, *incurred_amount*, and *accident_year*.

    Returns
    -------
    dict
        Keys: monthly_counts, quarterly_counts, yearly_counts,
        severity_distribution, severity_by_category, severity_by_specialty,
        frequency_severity_scatter, loss_ratio, yoy_metrics.
    """
    if df.empty:
        return _empty_result()

    # Ensure proper types
    df = df.copy()
    df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
    df["severity_level"] = pd.to_numeric(df["severity_level"], errors="coerce")
    df["paid_amount"] = pd.to_numeric(df["paid_amount"], errors="coerce").fillna(0)
    df["incurred_amount"] = pd.to_numeric(df["incurred_amount"], errors="coerce").fillna(0)

    # Derived period columns
    df["incident_month"] = df["incident_date"].dt.to_period("M")
    df["incident_quarter"] = df["incident_date"].dt.to_period("Q")
    df["incident_year"] = df["incident_date"].dt.year

    result: Dict[str, Any] = {}

    # ── Monthly counts with claim_type breakdown ─────────────────────────
    if "claim_type" in df.columns:
        monthly = (
            df.groupby(["incident_month", "claim_type"])
            .size()
            .unstack(fill_value=0)
        )
        monthly.index = monthly.index.astype(str)
        monthly["total"] = monthly.sum(axis=1)
        result["monthly_counts"] = monthly.to_dict()
    else:
        monthly = df.groupby("incident_month").size()
        monthly.index = monthly.index.astype(str)
        result["monthly_counts"] = {"total": monthly.to_dict()}

    # ── Quarterly counts ─────────────────────────────────────────────────
    quarterly = df.groupby("incident_quarter").size()
    quarterly.index = quarterly.index.astype(str)
    result["quarterly_counts"] = quarterly.to_dict()

    # ── Yearly counts with YoY change ────────────────────────────────────
    yearly = df.groupby("incident_year").size().sort_index()
    yearly_dict = yearly.to_dict()
    yoy_change: Dict[int, float | None] = {}
    years_sorted = sorted(yearly_dict.keys())
    for i, yr in enumerate(years_sorted):
        if i == 0:
            yoy_change[yr] = None
        else:
            prev = yearly_dict[years_sorted[i - 1]]
            yoy_change[yr] = round(
                (yearly_dict[yr] - prev) / prev * 100, 2
            ) if prev else None
    result["yearly_counts"] = {
        "counts": yearly_dict,
        "yoy_change_pct": yoy_change,
    }

    # ── Severity distribution ────────────────────────────────────────────
    sev_valid = df.dropna(subset=["severity_level"])
    if not sev_valid.empty:
        sev_counts = sev_valid["severity_level"].value_counts().sort_index()
        sev_pcts = sev_valid["severity_level"].value_counts(normalize=True).sort_index()
        result["severity_distribution"] = {
            "counts": {int(k): int(v) for k, v in sev_counts.items()},
            "percentages": {int(k): round(v * 100, 2) for k, v in sev_pcts.items()},
            "mean": round(float(sev_valid["severity_level"].mean()), 2),
            "median": float(sev_valid["severity_level"].median()),
        }
    else:
        result["severity_distribution"] = {
            "counts": {}, "percentages": {}, "mean": None, "median": None,
        }

    # ── Severity by claim_type ───────────────────────────────────────────
    if "claim_type" in df.columns and not sev_valid.empty:
        sev_by_cat = (
            sev_valid.groupby(["claim_type", "severity_level"])
            .size()
            .unstack(fill_value=0)
        )
        sev_by_cat.columns = [int(c) for c in sev_by_cat.columns]
        result["severity_by_category"] = sev_by_cat.to_dict(orient="index")
    else:
        result["severity_by_category"] = {}

    # ── Severity by specialty ────────────────────────────────────────────
    if "specialty" in df.columns and not sev_valid.empty:
        sev_by_spec = (
            sev_valid.groupby(["specialty", "severity_level"])
            .size()
            .unstack(fill_value=0)
        )
        sev_by_spec.columns = [int(c) for c in sev_by_spec.columns]
        result["severity_by_specialty"] = sev_by_spec.to_dict(orient="index")
    else:
        result["severity_by_specialty"] = {}

    # ── Frequency vs severity scatter (per specialty) ────────────────────
    if "specialty" in df.columns and not sev_valid.empty:
        spec_agg = sev_valid.groupby("specialty").agg(
            claim_count=("claim_id", "count"),
            avg_severity=("severity_level", "mean"),
            total_paid=("paid_amount", "sum"),
        )
        spec_agg["avg_severity"] = spec_agg["avg_severity"].round(2)
        spec_agg["total_paid"] = spec_agg["total_paid"].round(2)
        result["frequency_severity_scatter"] = spec_agg.to_dict(orient="index")
    else:
        result["frequency_severity_scatter"] = {}

    # ── Loss ratio by category ───────────────────────────────────────────
    if "claim_type" in df.columns:
        lr_group = df.groupby("claim_type").agg(
            total_paid=("paid_amount", "sum"),
            total_incurred=("incurred_amount", "sum"),
        )
        lr_group["loss_ratio"] = np.where(
            lr_group["total_incurred"] > 0,
            (lr_group["total_paid"] / lr_group["total_incurred"]).round(4),
            None,
        )
        result["loss_ratio"] = lr_group.to_dict(orient="index")
    else:
        total_paid = df["paid_amount"].sum()
        total_incurred = df["incurred_amount"].sum()
        result["loss_ratio"] = {
            "overall": {
                "total_paid": round(total_paid, 2),
                "total_incurred": round(total_incurred, 2),
                "loss_ratio": round(total_paid / total_incurred, 4)
                if total_incurred > 0
                else None,
            }
        }

    # ── Year-over-year metrics ───────────────────────────────────────────
    yoy_df = df.groupby("incident_year").agg(
        claim_count=("claim_id", "count"),
        total_paid=("paid_amount", "sum"),
        total_incurred=("incurred_amount", "sum"),
        avg_severity=("severity_level", "mean"),
    ).sort_index()

    yoy_metrics: Dict[int, Dict[str, Any]] = {}
    years_list = list(yoy_df.index)
    for i, yr in enumerate(years_list):
        row = yoy_df.loc[yr]
        entry: Dict[str, Any] = {
            "claim_count": int(row["claim_count"]),
            "total_paid": round(float(row["total_paid"]), 2),
            "total_incurred": round(float(row["total_incurred"]), 2),
            "avg_severity": round(float(row["avg_severity"]), 2)
            if pd.notna(row["avg_severity"])
            else None,
        }
        if i > 0:
            prev = yoy_df.loc[years_list[i - 1]]
            for metric in ["claim_count", "total_paid", "total_incurred"]:
                prev_val = float(prev[metric])
                curr_val = float(row[metric])
                entry[f"{metric}_yoy_pct"] = (
                    round((curr_val - prev_val) / prev_val * 100, 2)
                    if prev_val != 0
                    else None
                )
        else:
            for metric in ["claim_count", "total_paid", "total_incurred"]:
                entry[f"{metric}_yoy_pct"] = None

        yoy_metrics[int(yr)] = entry

    result["yoy_metrics"] = yoy_metrics

    return result


def _empty_result() -> Dict[str, Any]:
    """Return a skeleton result for empty input."""
    return {
        "monthly_counts": {},
        "quarterly_counts": {},
        "yearly_counts": {"counts": {}, "yoy_change_pct": {}},
        "severity_distribution": {
            "counts": {}, "percentages": {}, "mean": None, "median": None,
        },
        "severity_by_category": {},
        "severity_by_specialty": {},
        "frequency_severity_scatter": {},
        "loss_ratio": {},
        "yoy_metrics": {},
    }
