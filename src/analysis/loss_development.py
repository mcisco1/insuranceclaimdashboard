"""Loss-development triangle and chain-ladder reserve estimation.

Builds an accident-year x development-year triangle of cumulative paid
amounts, computes age-to-age factors, and projects ultimate losses and
IBNR reserves using the standard chain-ladder method.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def build_loss_triangle(df: pd.DataFrame) -> Dict[str, Any]:
    """Construct a loss triangle and estimate IBNR reserves.

    Parameters
    ----------
    df : pd.DataFrame
        Claims summary DataFrame.  Required columns:
        *accident_year*, *development_year*, *paid_amount*.

    Returns
    -------
    dict
        Keys: triangle, age_to_age_factors, cumulative_factors,
        ultimate_losses, ibnr_reserves.
    """
    required = {"accident_year", "development_year", "paid_amount"}
    if df.empty or not required.issubset(df.columns):
        return _empty_result()

    df = df.copy()
    df["accident_year"] = pd.to_numeric(df["accident_year"], errors="coerce")
    df["development_year"] = pd.to_numeric(df["development_year"], errors="coerce")
    df["paid_amount"] = pd.to_numeric(df["paid_amount"], errors="coerce").fillna(0)

    # Drop rows with invalid year data
    df = df.dropna(subset=["accident_year", "development_year"])
    if df.empty:
        return _empty_result()

    df["accident_year"] = df["accident_year"].astype(int)
    df["development_year"] = df["development_year"].astype(int)

    # ── Build incremental triangle ───────────────────────────────────────
    incremental = (
        df.groupby(["accident_year", "development_year"])["paid_amount"]
        .sum()
        .reset_index()
    )

    # Pivot to matrix form: rows = accident_year, cols = development_year
    inc_pivot = incremental.pivot_table(
        index="accident_year",
        columns="development_year",
        values="paid_amount",
        aggfunc="sum",
        fill_value=0,
    )

    # Ensure development years are sorted and start from 0
    dev_years = sorted(inc_pivot.columns)
    acc_years = sorted(inc_pivot.index)
    inc_pivot = inc_pivot.reindex(index=acc_years, columns=dev_years, fill_value=0)

    # ── Cumulative triangle ──────────────────────────────────────────────
    cum_triangle = inc_pivot.cumsum(axis=1)

    # Mask future development periods (upper-right of triangle)
    max_dev = max(dev_years) if dev_years else 0
    triangle_masked = cum_triangle.copy()
    for ay in acc_years:
        # The most recent accident year has the fewest development periods
        # available.  For a calendar-year triangle the latest observable
        # dev-year for accident year *ay* is  max_calendar_year - ay.
        max_calendar = max(acc_years) + max_dev  # upper bound
        # Simple heuristic: keep dev periods actually observed in data
        observed_devs = set(
            df.loc[df["accident_year"] == ay, "development_year"].unique()
        )
        for dy in dev_years:
            if dy not in observed_devs:
                triangle_masked.loc[ay, dy] = np.nan

    # Serialise triangle for output
    triangle_dict: Dict[str, Dict[str, Optional[float]]] = {}
    for ay in acc_years:
        row: Dict[str, Optional[float]] = {}
        for dy in dev_years:
            val = triangle_masked.loc[ay, dy]
            row[str(dy)] = round(float(val), 2) if pd.notna(val) else None
        triangle_dict[str(ay)] = row

    # ── Age-to-age (link) factors ────────────────────────────────────────
    # For each consecutive pair of dev years, compute the weighted-average
    # link factor using all accident years that have data for both periods.
    link_factors: Dict[str, Optional[float]] = {}
    for i in range(len(dev_years) - 1):
        d_curr = dev_years[i]
        d_next = dev_years[i + 1]

        # Only use rows where both periods are observed
        both = triangle_masked[[d_curr, d_next]].dropna()
        if both.empty or both[d_curr].sum() == 0:
            link_factors[f"{d_curr}-{d_next}"] = None
            continue

        # Volume-weighted average
        factor = float(both[d_next].sum() / both[d_curr].sum())
        link_factors[f"{d_curr}-{d_next}"] = round(factor, 6)

    # ── Cumulative development factors (tail) ────────────────────────────
    # CDF for period k = product of link factors from k to the end.
    cum_factors: Dict[str, Optional[float]] = {}
    # Build list of factors in order
    factor_values = []
    for i in range(len(dev_years) - 1):
        key = f"{dev_years[i]}-{dev_years[i+1]}"
        factor_values.append(link_factors.get(key))

    for i in range(len(dev_years)):
        # CDF from dev_years[i] to ultimate
        remaining = factor_values[i:]
        if any(f is None for f in remaining):
            # If any factor is missing, assume 1.0 (fully developed)
            product = 1.0
            for f in remaining:
                product *= f if f is not None else 1.0
        else:
            product = 1.0
            for f in remaining:
                product *= f
        cum_factors[str(dev_years[i])] = round(product, 6)

    # ── Ultimate losses ──────────────────────────────────────────────────
    # For each accident year, apply the CDF to the latest known cumulative
    # paid amount to project the ultimate loss.
    ultimate_losses: Dict[str, Dict[str, Optional[float]]] = {}
    for ay in acc_years:
        # Find latest dev year with data
        row = triangle_masked.loc[ay].dropna()
        if row.empty:
            ultimate_losses[str(ay)] = {
                "current_paid": 0.0,
                "latest_dev_year": None,
                "cdf_applied": None,
                "ultimate": None,
            }
            continue

        latest_dev = int(row.index.max())
        current_paid = float(row.iloc[-1])
        cdf = cum_factors.get(str(latest_dev), 1.0)
        ultimate = current_paid * cdf if cdf is not None else current_paid

        ultimate_losses[str(ay)] = {
            "current_paid": round(current_paid, 2),
            "latest_dev_year": latest_dev,
            "cdf_applied": cdf,
            "ultimate": round(ultimate, 2),
        }

    # ── IBNR reserves ────────────────────────────────────────────────────
    ibnr_reserves: Dict[str, Dict[str, Optional[float]]] = {}
    total_ibnr = 0.0
    total_ultimate = 0.0
    total_current = 0.0

    for ay_str, ult_data in ultimate_losses.items():
        current = ult_data["current_paid"] or 0.0
        ultimate = ult_data["ultimate"]
        if ultimate is not None:
            ibnr = ultimate - current
        else:
            ibnr = None

        ibnr_reserves[ay_str] = {
            "current_paid": round(current, 2),
            "ultimate": round(ultimate, 2) if ultimate is not None else None,
            "ibnr": round(ibnr, 2) if ibnr is not None else None,
        }

        total_current += current
        if ultimate is not None:
            total_ultimate += ultimate
            total_ibnr += (ultimate - current)

    ibnr_reserves["total"] = {
        "current_paid": round(total_current, 2),
        "ultimate": round(total_ultimate, 2),
        "ibnr": round(total_ibnr, 2),
    }

    return {
        "triangle": triangle_dict,
        "age_to_age_factors": link_factors,
        "cumulative_factors": cum_factors,
        "ultimate_losses": ultimate_losses,
        "ibnr_reserves": ibnr_reserves,
    }


def _empty_result() -> Dict[str, Any]:
    return {
        "triangle": {},
        "age_to_age_factors": {},
        "cumulative_factors": {},
        "ultimate_losses": {},
        "ibnr_reserves": {},
    }
