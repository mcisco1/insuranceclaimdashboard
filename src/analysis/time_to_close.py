"""Time-to-close analysis for insurance claims.

Measures claim lifecycle duration, SLA compliance, backlog aging,
and close-time trends.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from config import SLA_THRESHOLDS


# ── Aging buckets for open-claim backlog ─────────────────────────────────
_AGING_BUCKETS: List[tuple[str, int, int]] = [
    ("0-30", 0, 30),
    ("31-90", 31, 90),
    ("91-180", 91, 180),
    ("181-365", 181, 365),
    ("365+", 366, 999_999),
]


def analyze_time_to_close(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse claim closure durations and backlog.

    Parameters
    ----------
    df : pd.DataFrame
        Claims summary DataFrame.  Expected columns include
        *days_to_close*, *severity_level*, *specialty*, *region*,
        *claim_type*, *status*, *incident_date*, *report_date*.

    Returns
    -------
    dict
        Keys: close_time_distribution, median_by_severity,
        median_by_specialty, median_by_region, sla_compliance,
        backlog_aging, monthly_close_trend.
    """
    if df.empty:
        return _empty_result()

    df = df.copy()
    df["days_to_close"] = pd.to_numeric(df.get("days_to_close"), errors="coerce")
    df["severity_level"] = pd.to_numeric(df.get("severity_level"), errors="coerce")
    df["incident_date"] = pd.to_datetime(df.get("incident_date"), errors="coerce")
    df["report_date"] = pd.to_datetime(df.get("report_date"), errors="coerce")

    closed = df[df["days_to_close"].notna()].copy()

    result: Dict[str, Any] = {}

    # ── Close-time histogram ─────────────────────────────────────────────
    if not closed.empty:
        bins = [0, 30, 60, 90, 120, 180, 270, 365, 545, 730, np.inf]
        labels = [
            "0-30", "31-60", "61-90", "91-120", "121-180",
            "181-270", "271-365", "366-545", "546-730", "730+",
        ]
        closed["close_bucket"] = pd.cut(
            closed["days_to_close"], bins=bins, labels=labels, right=True,
        )
        hist = closed["close_bucket"].value_counts().sort_index()
        result["close_time_distribution"] = {
            "bins": labels,
            "counts": {str(k): int(v) for k, v in hist.items()},
            "median_overall": round(float(closed["days_to_close"].median()), 1),
            "mean_overall": round(float(closed["days_to_close"].mean()), 1),
            "p95": round(float(closed["days_to_close"].quantile(0.95)), 1),
        }
    else:
        result["close_time_distribution"] = {
            "bins": [], "counts": {},
            "median_overall": None, "mean_overall": None, "p95": None,
        }

    # ── Median by severity ───────────────────────────────────────────────
    result["median_by_severity"] = _median_by(closed, "severity_level")

    # ── Median by specialty ──────────────────────────────────────────────
    result["median_by_specialty"] = _median_by(closed, "specialty")

    # ── Median by region ─────────────────────────────────────────────────
    result["median_by_region"] = _median_by(closed, "region")

    # ── SLA compliance ───────────────────────────────────────────────────
    sla: Dict[str, Dict[str, float]] = {}
    group_col = "claim_type" if "claim_type" in closed.columns else None

    if not closed.empty:
        if group_col:
            for cat, grp in closed.groupby(group_col):
                cat_sla: Dict[str, float] = {}
                total = len(grp)
                for threshold in SLA_THRESHOLDS:
                    within = (grp["days_to_close"] <= threshold).sum()
                    cat_sla[f"within_{threshold}_days_pct"] = round(
                        within / total * 100, 2
                    )
                cat_sla["claim_count"] = total
                sla[str(cat)] = cat_sla
        # Overall
        total = len(closed)
        overall_sla: Dict[str, Any] = {"claim_count": total}
        for threshold in SLA_THRESHOLDS:
            within = (closed["days_to_close"] <= threshold).sum()
            overall_sla[f"within_{threshold}_days_pct"] = round(
                within / total * 100, 2
            )
        sla["overall"] = overall_sla

    result["sla_compliance"] = sla

    # ── Backlog aging (open claims) ──────────────────────────────────────
    open_claims = df[df.get("status", pd.Series(dtype=str)).isin(["open", "reopened"])].copy()
    backlog: Dict[str, Any] = {}

    if not open_claims.empty:
        # Calculate age from report_date to today
        today = pd.Timestamp.now().normalize()
        reference_date = open_claims["report_date"].fillna(open_claims["incident_date"])
        open_claims["age_days"] = (today - reference_date).dt.days

        bucket_counts: Dict[str, int] = {}
        for label, lo, hi in _AGING_BUCKETS:
            count = int(
                ((open_claims["age_days"] >= lo) & (open_claims["age_days"] <= hi)).sum()
            )
            bucket_counts[label] = count

        backlog = {
            "total_open": len(open_claims),
            "buckets": bucket_counts,
            "median_age_days": round(float(open_claims["age_days"].median()), 1),
            "mean_age_days": round(float(open_claims["age_days"].mean()), 1),
        }
    else:
        backlog = {
            "total_open": 0,
            "buckets": {label: 0 for label, _, _ in _AGING_BUCKETS},
            "median_age_days": None,
            "mean_age_days": None,
        }

    result["backlog_aging"] = backlog

    # ── Monthly close-time trend ─────────────────────────────────────────
    if not closed.empty and closed["incident_date"].notna().any():
        closed["incident_month"] = closed["incident_date"].dt.to_period("M")
        trend = (
            closed.groupby("incident_month")["days_to_close"]
            .median()
            .sort_index()
        )
        trend.index = trend.index.astype(str)
        result["monthly_close_trend"] = {
            str(k): round(float(v), 1) for k, v in trend.items()
        }
    else:
        result["monthly_close_trend"] = {}

    return result


# ── Helpers ──────────────────────────────────────────────────────────────

def _median_by(closed_df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Compute median and mean days_to_close grouped by *col*."""
    if closed_df.empty or col not in closed_df.columns:
        return {}

    valid = closed_df.dropna(subset=[col, "days_to_close"])
    if valid.empty:
        return {}

    agg = valid.groupby(col)["days_to_close"].agg(["median", "mean", "count"])
    agg["median"] = agg["median"].round(1)
    agg["mean"] = agg["mean"].round(1)
    agg["count"] = agg["count"].astype(int)
    return agg.to_dict(orient="index")


def _empty_result() -> Dict[str, Any]:
    return {
        "close_time_distribution": {
            "bins": [], "counts": {},
            "median_overall": None, "mean_overall": None, "p95": None,
        },
        "median_by_severity": {},
        "median_by_specialty": {},
        "median_by_region": {},
        "sla_compliance": {},
        "backlog_aging": {
            "total_open": 0,
            "buckets": {label: 0 for label, _, _ in _AGING_BUCKETS},
            "median_age_days": None, "mean_age_days": None,
        },
        "monthly_close_trend": {},
    }
