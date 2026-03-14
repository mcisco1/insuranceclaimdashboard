"""What-if scenario analysis for insurance claims.

Models the financial and operational impact of hypothetical interventions
such as reducing close times, shifting severity distributions, and
lowering litigation rates.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ── Scenario parameters ─────────────────────────────────────────────────
_CLOSE_TIME_REDUCTION_PCTS = [10, 20, 30]
_SEVERITY_REDUCTION_PCTS = [10, 20, 30]
_LITIGATION_REDUCTION_PCTS = [25, 50]
_LITIGATION_COST_MULTIPLIER = 3.0  # litigated claims cost ~3x non-litigated


def run_scenarios(df: pd.DataFrame) -> Dict[str, Any]:
    """Run what-if scenario analyses on the claims portfolio.

    Parameters
    ----------
    df : pd.DataFrame
        Claims summary DataFrame.  Expected columns include
        *days_to_close*, *severity_level*, *incurred_amount*,
        *paid_amount*, *litigation_flag*, *status*.

    Returns
    -------
    dict
        Keys: close_time_reduction, severity_reduction,
        litigation_reduction, combined_scenario.
        Each scenario contains baseline, new values, and deltas.
    """
    if df.empty:
        return _empty_result()

    df = df.copy()
    df["days_to_close"] = pd.to_numeric(df.get("days_to_close"), errors="coerce")
    df["severity_level"] = pd.to_numeric(df.get("severity_level"), errors="coerce")
    df["incurred_amount"] = pd.to_numeric(df.get("incurred_amount"), errors="coerce").fillna(0)
    df["paid_amount"] = pd.to_numeric(df.get("paid_amount"), errors="coerce").fillna(0)
    df["litigation_flag"] = pd.to_numeric(df.get("litigation_flag"), errors="coerce").fillna(0).astype(int)

    result: Dict[str, Any] = {}

    # ── Scenario 1: Close-time reduction ─────────────────────────────────
    result["close_time_reduction"] = _close_time_scenarios(df)

    # ── Scenario 2: Severity reduction ───────────────────────────────────
    result["severity_reduction"] = _severity_scenarios(df)

    # ── Scenario 3: Litigation reduction ─────────────────────────────────
    result["litigation_reduction"] = _litigation_scenarios(df)

    # ── Combined scenario ────────────────────────────────────────────────
    result["combined_scenario"] = _combined_scenario(df)

    return result


# ── Scenario implementations ────────────────────────────────────────────

def _close_time_scenarios(df: pd.DataFrame) -> Dict[str, Any]:
    """What-if: reduce average close time by 10/20/30%.

    Impact measured on:
    - Average and median close time
    - Number of claims exceeding SLA thresholds
    - Estimated reserve savings (shorter tail = lower reserves)
    """
    closed = df[df["days_to_close"].notna()].copy()
    if closed.empty:
        return {"scenarios": {}, "baseline": {}}

    baseline_mean = float(closed["days_to_close"].mean())
    baseline_median = float(closed["days_to_close"].median())

    # Count open/reopened claims (backlog)
    if "status" in df.columns:
        open_count = int(df["status"].isin(["open", "reopened"]).sum())
    else:
        open_count = int(df["days_to_close"].isna().sum())

    baseline_over_365 = int((closed["days_to_close"] > 365).sum())
    baseline_over_180 = int((closed["days_to_close"] > 180).sum())

    baseline = {
        "mean_close_days": round(baseline_mean, 1),
        "median_close_days": round(baseline_median, 1),
        "claims_over_180_days": baseline_over_180,
        "claims_over_365_days": baseline_over_365,
        "open_backlog": open_count,
    }

    scenarios: Dict[str, Dict[str, Any]] = {}
    for pct in _CLOSE_TIME_REDUCTION_PCTS:
        factor = 1 - pct / 100
        new_close = closed["days_to_close"] * factor
        new_mean = float(new_close.mean())
        new_median = float(new_close.median())
        new_over_365 = int((new_close > 365).sum())
        new_over_180 = int((new_close > 180).sum())

        # Estimate backlog reduction: proportional to close-time reduction
        estimated_backlog = max(0, int(open_count * factor))

        scenarios[f"{pct}_pct_reduction"] = {
            "new_mean_close_days": round(new_mean, 1),
            "new_median_close_days": round(new_median, 1),
            "new_claims_over_180_days": new_over_180,
            "new_claims_over_365_days": new_over_365,
            "estimated_backlog": estimated_backlog,
            "absolute_change": {
                "mean_close_days": round(new_mean - baseline_mean, 1),
                "claims_over_365_days": new_over_365 - baseline_over_365,
                "backlog": estimated_backlog - open_count,
            },
            "pct_change": {
                "mean_close_days": round(-pct, 2),
                "claims_over_365_days": round(
                    (new_over_365 - baseline_over_365) / baseline_over_365 * 100, 2
                ) if baseline_over_365 > 0 else 0.0,
                "backlog": round(
                    (estimated_backlog - open_count) / open_count * 100, 2
                ) if open_count > 0 else 0.0,
            },
        }

    return {"baseline": baseline, "scenarios": scenarios}


def _severity_scenarios(df: pd.DataFrame) -> Dict[str, Any]:
    """What-if: shift 10/20/30% of severity 4-5 claims to severity 3.

    Impact measured on incurred_amount (since higher severity = higher cost).
    """
    sev_valid = df.dropna(subset=["severity_level"]).copy()
    if sev_valid.empty:
        return {"scenarios": {}, "baseline": {}}

    # Compute baseline cost by severity
    high_sev_mask = sev_valid["severity_level"].isin([4, 5])
    high_sev = sev_valid[high_sev_mask]
    sev3 = sev_valid[sev_valid["severity_level"] == 3]

    baseline_total_incurred = float(sev_valid["incurred_amount"].sum())
    baseline_high_sev_count = len(high_sev)
    baseline_high_sev_incurred = float(high_sev["incurred_amount"].sum())

    # Average cost per severity level
    avg_cost_high = (
        float(high_sev["incurred_amount"].mean()) if not high_sev.empty else 0.0
    )
    avg_cost_sev3 = (
        float(sev3["incurred_amount"].mean()) if not sev3.empty else 0.0
    )

    baseline = {
        "total_incurred": round(baseline_total_incurred, 2),
        "high_severity_count": baseline_high_sev_count,
        "high_severity_incurred": round(baseline_high_sev_incurred, 2),
        "avg_cost_severity_4_5": round(avg_cost_high, 2),
        "avg_cost_severity_3": round(avg_cost_sev3, 2),
    }

    scenarios: Dict[str, Dict[str, Any]] = {}
    for pct in _SEVERITY_REDUCTION_PCTS:
        claims_shifted = int(baseline_high_sev_count * pct / 100)

        # Cost savings: shifted claims move from avg high-sev cost to avg sev-3 cost
        cost_savings = claims_shifted * (avg_cost_high - avg_cost_sev3)
        new_total_incurred = baseline_total_incurred - cost_savings

        scenarios[f"{pct}_pct_shift"] = {
            "claims_shifted": claims_shifted,
            "remaining_high_severity": baseline_high_sev_count - claims_shifted,
            "baseline_total_incurred": round(baseline_total_incurred, 2),
            "new_total_incurred": round(new_total_incurred, 2),
            "absolute_change": round(-cost_savings, 2),
            "pct_change": round(
                -cost_savings / baseline_total_incurred * 100, 2
            ) if baseline_total_incurred > 0 else 0.0,
        }

    return {"baseline": baseline, "scenarios": scenarios}


def _litigation_scenarios(df: pd.DataFrame) -> Dict[str, Any]:
    """What-if: reduce litigation rate by 25/50%.

    Litigated claims typically cost ~3x non-litigated (defense costs,
    settlements).  Reducing litigation produces direct cost savings.
    """
    if "litigation_flag" not in df.columns:
        return {"scenarios": {}, "baseline": {}}

    total = len(df)
    litigated = df[df["litigation_flag"] == 1]
    non_litigated = df[df["litigation_flag"] == 0]

    litigated_count = len(litigated)
    non_litigated_count = len(non_litigated)
    litigation_rate = litigated_count / total * 100 if total > 0 else 0.0

    avg_cost_litigated = (
        float(litigated["incurred_amount"].mean()) if not litigated.empty else 0.0
    )
    avg_cost_non_litigated = (
        float(non_litigated["incurred_amount"].mean())
        if not non_litigated.empty
        else 0.0
    )
    total_litigation_cost = float(litigated["incurred_amount"].sum())
    baseline_total = float(df["incurred_amount"].sum())

    baseline = {
        "total_claims": total,
        "litigated_claims": litigated_count,
        "litigation_rate_pct": round(litigation_rate, 2),
        "avg_cost_litigated": round(avg_cost_litigated, 2),
        "avg_cost_non_litigated": round(avg_cost_non_litigated, 2),
        "total_litigation_cost": round(total_litigation_cost, 2),
        "total_incurred": round(baseline_total, 2),
        "cost_multiplier": _LITIGATION_COST_MULTIPLIER,
    }

    scenarios: Dict[str, Dict[str, Any]] = {}
    for pct in _LITIGATION_REDUCTION_PCTS:
        claims_removed = int(litigated_count * pct / 100)
        new_litigated = litigated_count - claims_removed

        # Savings: removed litigated claims revert to non-litigated cost level
        cost_savings = claims_removed * (avg_cost_litigated - avg_cost_non_litigated)
        new_total = baseline_total - cost_savings

        scenarios[f"{pct}_pct_reduction"] = {
            "claims_no_longer_litigated": claims_removed,
            "remaining_litigated": new_litigated,
            "new_litigation_rate_pct": round(
                new_litigated / total * 100, 2
            ) if total > 0 else 0.0,
            "baseline_total_incurred": round(baseline_total, 2),
            "new_total_incurred": round(new_total, 2),
            "absolute_change": round(-cost_savings, 2),
            "pct_change": round(
                -cost_savings / baseline_total * 100, 2
            ) if baseline_total > 0 else 0.0,
        }

    return {"baseline": baseline, "scenarios": scenarios}


def _combined_scenario(df: pd.DataFrame) -> Dict[str, Any]:
    """Combined scenario: 20% close-time + 20% severity shift + 25% litigation reduction."""
    sev_valid = df.dropna(subset=["severity_level"]).copy()

    baseline_total_incurred = float(df["incurred_amount"].sum())

    # --- Severity component (20% shift) ---
    high_sev = sev_valid[sev_valid["severity_level"].isin([4, 5])]
    sev3 = sev_valid[sev_valid["severity_level"] == 3]

    avg_cost_high = float(high_sev["incurred_amount"].mean()) if not high_sev.empty else 0.0
    avg_cost_sev3 = float(sev3["incurred_amount"].mean()) if not sev3.empty else 0.0
    claims_shifted = int(len(high_sev) * 0.20)
    severity_savings = claims_shifted * (avg_cost_high - avg_cost_sev3)

    # --- Litigation component (25% reduction) ---
    litigation_savings = 0.0
    if "litigation_flag" in df.columns:
        litigated = df[df["litigation_flag"] == 1]
        non_litigated = df[df["litigation_flag"] == 0]
        avg_lit = float(litigated["incurred_amount"].mean()) if not litigated.empty else 0.0
        avg_non = float(non_litigated["incurred_amount"].mean()) if not non_litigated.empty else 0.0
        lit_removed = int(len(litigated) * 0.25)
        litigation_savings = lit_removed * (avg_lit - avg_non)

    # --- Close-time component (20% reduction) ---
    # Estimate: faster closure reduces tail reserves by ~5% of total incurred
    # (conservative actuarial rule of thumb)
    close_time_savings = baseline_total_incurred * 0.20 * 0.05

    total_savings = severity_savings + litigation_savings + close_time_savings
    new_total = baseline_total_incurred - total_savings

    # --- Backlog component ---
    if "status" in df.columns:
        open_count = int(df["status"].isin(["open", "reopened"]).sum())
    else:
        open_count = int(df["days_to_close"].isna().sum())
    new_backlog = max(0, int(open_count * 0.80))

    return {
        "baseline_total_incurred": round(baseline_total_incurred, 2),
        "new_total_incurred": round(new_total, 2),
        "total_savings": round(total_savings, 2),
        "components": {
            "severity_shift_savings": round(severity_savings, 2),
            "litigation_reduction_savings": round(litigation_savings, 2),
            "close_time_reduction_savings": round(close_time_savings, 2),
        },
        "absolute_change": round(-total_savings, 2),
        "pct_change": round(
            -total_savings / baseline_total_incurred * 100, 2
        ) if baseline_total_incurred > 0 else 0.0,
        "backlog": {
            "baseline_open": open_count,
            "estimated_new_open": new_backlog,
            "reduction": open_count - new_backlog,
        },
        "assumptions": {
            "severity_shift_pct": 20,
            "litigation_reduction_pct": 25,
            "close_time_reduction_pct": 20,
            "close_time_reserve_impact_factor": 0.05,
        },
    }


def _empty_result() -> Dict[str, Any]:
    empty_scenario = {"scenarios": {}, "baseline": {}}
    return {
        "close_time_reduction": empty_scenario,
        "severity_reduction": empty_scenario,
        "litigation_reduction": empty_scenario,
        "combined_scenario": {
            "baseline_total_incurred": 0.0,
            "new_total_incurred": 0.0,
            "total_savings": 0.0,
            "components": {},
            "absolute_change": 0.0,
            "pct_change": 0.0,
            "backlog": {},
            "assumptions": {},
        },
    }
