"""Repeat-incident and reopen analysis for insurance claims.

Examines claim reopening patterns, correlating reopen rates with
providers, root causes, specialties, and litigation.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def analyze_repeat_incidents(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse claim reopening and repeat-event patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Claims summary DataFrame.  Expected columns include *status*,
        *provider_id*, *provider_name*, *root_cause_category*, *specialty*,
        *litigation_flag*, *repeat_event_flag*.

    Returns
    -------
    dict
        Keys: overall_reopen_rate, reopen_by_provider, reopen_by_root_cause,
        reopen_by_specialty, reopen_litigation_correlation,
        repeat_event_summary.
    """
    if df.empty:
        return _empty_result()

    df = df.copy()

    # Normalise flag columns
    for col in ["litigation_flag", "repeat_event_flag"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Binary reopen indicator
    if "status" in df.columns:
        df["is_reopened"] = (df["status"].str.lower() == "reopened").astype(int)
    else:
        df["is_reopened"] = 0

    total_claims = len(df)
    reopened_count = int(df["is_reopened"].sum())

    result: Dict[str, Any] = {}

    # ── Overall reopen rate ──────────────────────────────────────────────
    result["overall_reopen_rate"] = {
        "total_claims": total_claims,
        "reopened_claims": reopened_count,
        "reopen_rate_pct": round(reopened_count / total_claims * 100, 2)
        if total_claims > 0
        else 0.0,
    }

    # ── Reopen by provider (min 5 claims) ────────────────────────────────
    if "provider_id" in df.columns:
        prov_agg = df.groupby("provider_id").agg(
            claim_count=("claim_id", "count") if "claim_id" in df.columns
            else ("is_reopened", "count"),
            reopened=("is_reopened", "sum"),
        )

        # Add provider_name and specialty if available
        if "provider_name" in df.columns:
            name_map = (
                df.dropna(subset=["provider_name"])
                .drop_duplicates("provider_id")
                .set_index("provider_id")["provider_name"]
            )
            prov_agg = prov_agg.join(name_map, how="left")

        if "specialty" in df.columns:
            spec_map = (
                df.dropna(subset=["specialty"])
                .drop_duplicates("provider_id")
                .set_index("provider_id")["specialty"]
            )
            prov_agg = prov_agg.join(spec_map, how="left")

        # Filter to providers with at least 5 claims
        prov_qualified = prov_agg[prov_agg["claim_count"] >= 5].copy()
        prov_qualified["reopen_rate_pct"] = (
            prov_qualified["reopened"] / prov_qualified["claim_count"] * 100
        ).round(2)
        prov_qualified = prov_qualified.sort_values(
            "reopen_rate_pct", ascending=False
        )
        prov_qualified["reopened"] = prov_qualified["reopened"].astype(int)
        prov_qualified["claim_count"] = prov_qualified["claim_count"].astype(int)

        result["reopen_by_provider"] = prov_qualified.head(50).to_dict(orient="index")
    else:
        result["reopen_by_provider"] = {}

    # ── Reopen by root cause ─────────────────────────────────────────────
    result["reopen_by_root_cause"] = _reopen_by_group(df, "root_cause_category")

    # ── Reopen by specialty ──────────────────────────────────────────────
    result["reopen_by_specialty"] = _reopen_by_group(df, "specialty")

    # ── Reopen–litigation correlation ────────────────────────────────────
    if "litigation_flag" in df.columns:
        lit_groups = df.groupby("is_reopened").agg(
            claim_count=("is_reopened", "count"),
            litigation_rate_pct=("litigation_flag", "mean"),
        )
        lit_groups["litigation_rate_pct"] = (
            lit_groups["litigation_rate_pct"] * 100
        ).round(2)
        lit_groups.index = lit_groups.index.map({0: "not_reopened", 1: "reopened"})

        # Also compute reopen rate for litigated vs non-litigated
        lit_reopen = df.groupby("litigation_flag").agg(
            claim_count=("litigation_flag", "count"),
            reopen_rate_pct=("is_reopened", "mean"),
        )
        lit_reopen["reopen_rate_pct"] = (
            lit_reopen["reopen_rate_pct"] * 100
        ).round(2)
        lit_reopen.index = lit_reopen.index.map({0: "not_litigated", 1: "litigated"})

        result["reopen_litigation_correlation"] = {
            "litigation_rate_by_reopen_status": lit_groups.to_dict(orient="index"),
            "reopen_rate_by_litigation_status": lit_reopen.to_dict(orient="index"),
        }
    else:
        result["reopen_litigation_correlation"] = {}

    # ── Repeat-event summary ─────────────────────────────────────────────
    if "repeat_event_flag" in df.columns:
        repeat_count = int(df["repeat_event_flag"].sum())
        repeat_rate = round(repeat_count / total_claims * 100, 2) if total_claims > 0 else 0.0

        repeat_summary: Dict[str, Any] = {
            "total_repeat_events": repeat_count,
            "repeat_rate_pct": repeat_rate,
        }

        # Repeat events by severity
        if "severity_level" in df.columns:
            df["severity_level"] = pd.to_numeric(
                df["severity_level"], errors="coerce"
            )
            sev_repeat = df.dropna(subset=["severity_level"]).groupby(
                "severity_level"
            ).agg(
                claim_count=("repeat_event_flag", "count"),
                repeat_count=("repeat_event_flag", "sum"),
            )
            sev_repeat["repeat_rate_pct"] = np.where(
                sev_repeat["claim_count"] > 0,
                (sev_repeat["repeat_count"] / sev_repeat["claim_count"] * 100).round(2),
                0.0,
            )
            sev_repeat["claim_count"] = sev_repeat["claim_count"].astype(int)
            sev_repeat["repeat_count"] = sev_repeat["repeat_count"].astype(int)
            repeat_summary["by_severity"] = {
                str(int(k)): v for k, v in sev_repeat.to_dict(orient="index").items()
            }

        # Repeat events by claim_type
        if "claim_type" in df.columns:
            ct_repeat = df.groupby("claim_type").agg(
                claim_count=("repeat_event_flag", "count"),
                repeat_count=("repeat_event_flag", "sum"),
            )
            ct_repeat["repeat_rate_pct"] = np.where(
                ct_repeat["claim_count"] > 0,
                (ct_repeat["repeat_count"] / ct_repeat["claim_count"] * 100).round(2),
                0.0,
            )
            ct_repeat["claim_count"] = ct_repeat["claim_count"].astype(int)
            ct_repeat["repeat_count"] = ct_repeat["repeat_count"].astype(int)
            repeat_summary["by_claim_type"] = ct_repeat.to_dict(orient="index")

        # Overlap: claims that are both repeat events and reopened
        both = int(
            ((df["repeat_event_flag"] == 1) & (df["is_reopened"] == 1)).sum()
        )
        repeat_summary["repeat_and_reopened"] = both

        result["repeat_event_summary"] = repeat_summary
    else:
        result["repeat_event_summary"] = {
            "total_repeat_events": 0,
            "repeat_rate_pct": 0.0,
        }

    return result


# ── Helpers ──────────────────────────────────────────────────────────────

def _reopen_by_group(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Compute reopen rate grouped by *col*."""
    if col not in df.columns or df.empty:
        return {}

    valid = df.dropna(subset=[col])
    if valid.empty:
        return {}

    agg = valid.groupby(col).agg(
        claim_count=("is_reopened", "count"),
        reopened=("is_reopened", "sum"),
    )
    agg["reopen_rate_pct"] = np.where(
        agg["claim_count"] > 0,
        (agg["reopened"] / agg["claim_count"] * 100).round(2),
        0.0,
    )
    agg["reopened"] = agg["reopened"].astype(int)
    agg["claim_count"] = agg["claim_count"].astype(int)
    agg = agg.sort_values("reopen_rate_pct", ascending=False)

    return agg.to_dict(orient="index")


def _empty_result() -> Dict[str, Any]:
    return {
        "overall_reopen_rate": {
            "total_claims": 0, "reopened_claims": 0, "reopen_rate_pct": 0.0,
        },
        "reopen_by_provider": {},
        "reopen_by_root_cause": {},
        "reopen_by_specialty": {},
        "reopen_litigation_correlation": {},
        "repeat_event_summary": {
            "total_repeat_events": 0, "repeat_rate_pct": 0.0,
        },
    }
