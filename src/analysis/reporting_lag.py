"""Reporting-lag analysis for insurance claims.

Measures the delay between incident occurrence and claim reporting,
segmented by specialty, region, severity, and individual providers.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def analyze_reporting_lag(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse reporting lag (days between incident and report).

    Parameters
    ----------
    df : pd.DataFrame
        Claims summary DataFrame.  Expected columns include
        *days_to_report*, *specialty*, *region*, *severity_level*,
        *provider_id*, *provider_name*, *incident_date*.

    Returns
    -------
    dict
        Keys: lag_distribution, lag_by_specialty, lag_by_region,
        lag_by_severity, slow_reporters, monthly_lag_trend.
    """
    if df.empty:
        return _empty_result()

    df = df.copy()
    df["days_to_report"] = pd.to_numeric(df.get("days_to_report"), errors="coerce")
    df["severity_level"] = pd.to_numeric(df.get("severity_level"), errors="coerce")
    df["incident_date"] = pd.to_datetime(df.get("incident_date"), errors="coerce")

    lag_valid = df.dropna(subset=["days_to_report"])
    if lag_valid.empty:
        return _empty_result()

    result: Dict[str, Any] = {}

    # ── Lag distribution histogram ───────────────────────────────────────
    bins = [0, 7, 14, 30, 60, 90, 180, 365, np.inf]
    labels = ["0-7", "8-14", "15-30", "31-60", "61-90", "91-180", "181-365", "365+"]
    lag_valid = lag_valid.copy()
    lag_valid["lag_bucket"] = pd.cut(
        lag_valid["days_to_report"], bins=bins, labels=labels, right=True,
    )
    hist = lag_valid["lag_bucket"].value_counts().sort_index()
    result["lag_distribution"] = {
        "bins": labels,
        "counts": {str(k): int(v) for k, v in hist.items()},
        "median": round(float(lag_valid["days_to_report"].median()), 1),
        "mean": round(float(lag_valid["days_to_report"].mean()), 1),
        "p90": round(float(lag_valid["days_to_report"].quantile(0.90)), 1),
        "p95": round(float(lag_valid["days_to_report"].quantile(0.95)), 1),
    }

    # ── Lag by specialty ─────────────────────────────────────────────────
    result["lag_by_specialty"] = _lag_by_group(lag_valid, "specialty")

    # ── Lag by region ────────────────────────────────────────────────────
    result["lag_by_region"] = _lag_by_group(lag_valid, "region")

    # ── Lag by severity ──────────────────────────────────────────────────
    sev_valid = lag_valid.dropna(subset=["severity_level"])
    result["lag_by_severity"] = _lag_by_group(sev_valid, "severity_level")

    # ── Slow reporters ───────────────────────────────────────────────────
    # Providers with average lag > 2x the overall median and at least 5 claims
    overall_median = float(lag_valid["days_to_report"].median())
    threshold = overall_median * 2

    if "provider_id" in lag_valid.columns:
        prov_agg = lag_valid.groupby("provider_id").agg(
            claim_count=("days_to_report", "count"),
            mean_lag=("days_to_report", "mean"),
            median_lag=("days_to_report", "median"),
        )
        # Add provider_name if available
        if "provider_name" in lag_valid.columns:
            name_map = (
                lag_valid.dropna(subset=["provider_name"])
                .drop_duplicates("provider_id")
                .set_index("provider_id")["provider_name"]
            )
            prov_agg = prov_agg.join(name_map, how="left")

        slow = prov_agg[
            (prov_agg["claim_count"] >= 5) & (prov_agg["mean_lag"] > threshold)
        ].sort_values("mean_lag", ascending=False)

        slow["mean_lag"] = slow["mean_lag"].round(1)
        slow["median_lag"] = slow["median_lag"].round(1)
        result["slow_reporters"] = {
            "threshold_days": round(threshold, 1),
            "overall_median": round(overall_median, 1),
            "providers": slow.head(50).to_dict(orient="index"),
        }
    else:
        result["slow_reporters"] = {
            "threshold_days": round(threshold, 1),
            "overall_median": round(overall_median, 1),
            "providers": {},
        }

    # ── Monthly lag trend ────────────────────────────────────────────────
    dated = lag_valid.dropna(subset=["incident_date"])
    if not dated.empty:
        dated = dated.copy()
        dated["incident_month"] = dated["incident_date"].dt.to_period("M")
        trend = (
            dated.groupby("incident_month")["days_to_report"]
            .agg(["median", "mean", "count"])
            .sort_index()
        )
        trend.index = trend.index.astype(str)
        trend["median"] = trend["median"].round(1)
        trend["mean"] = trend["mean"].round(1)
        trend["count"] = trend["count"].astype(int)
        result["monthly_lag_trend"] = trend.to_dict(orient="index")
    else:
        result["monthly_lag_trend"] = {}

    return result


# ── Helpers ──────────────────────────────────────────────────────────────

def _lag_by_group(data: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Compute mean/median reporting lag grouped by *col*."""
    if data.empty or col not in data.columns:
        return {}

    valid = data.dropna(subset=[col, "days_to_report"])
    if valid.empty:
        return {}

    agg = valid.groupby(col)["days_to_report"].agg(
        ["mean", "median", "count", "std"]
    )
    agg["mean"] = agg["mean"].round(1)
    agg["median"] = agg["median"].round(1)
    agg["std"] = agg["std"].round(1)
    agg["count"] = agg["count"].astype(int)

    # Convert index to string keys for JSON serialisation
    result = {}
    for idx, row in agg.iterrows():
        key = str(int(idx)) if isinstance(idx, (float, np.floating)) else str(idx)
        result[key] = row.to_dict()

    return result


def _empty_result() -> Dict[str, Any]:
    return {
        "lag_distribution": {
            "bins": [], "counts": {},
            "median": None, "mean": None, "p90": None, "p95": None,
        },
        "lag_by_specialty": {},
        "lag_by_region": {},
        "lag_by_severity": {},
        "slow_reporters": {
            "threshold_days": None, "overall_median": None, "providers": {},
        },
        "monthly_lag_trend": {},
    }
