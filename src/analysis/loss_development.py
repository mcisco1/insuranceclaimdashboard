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

    # ── Mack standard errors ──────────────────────────────────────────
    mack_se = _mack_standard_errors(
        cum_triangle, triangle_masked, link_factors,
        cum_factors, dev_years, acc_years, ultimate_losses,
    )

    return {
        "triangle": triangle_dict,
        "age_to_age_factors": link_factors,
        "cumulative_factors": cum_factors,
        "ultimate_losses": ultimate_losses,
        "ibnr_reserves": ibnr_reserves,
        "mack_standard_errors": mack_se,
    }


def _mack_standard_errors(
    cum_triangle: pd.DataFrame,
    triangle_masked: pd.DataFrame,
    link_factors: Dict[str, Optional[float]],
    cum_factors: Dict[str, Optional[float]],
    dev_years: list,
    acc_years: list,
    ultimate_losses: Dict[str, Dict[str, Optional[float]]],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Implement Mack (1993) variance estimators for chain-ladder prediction intervals.

    Returns per-accident-year IBNR estimate, standard error, and 95% CI.
    """
    n_dev = len(dev_years)
    n_acc = len(acc_years)

    if n_dev < 2 or n_acc < 2:
        return {}

    # Compute sigma^2_k -- Mack's variance of individual link ratios
    # sigma^2_k = (1/(n_k - 1)) * sum over i of C_{i,k} * (f_{i,k+1}/C_{i,k} - f_k)^2
    sigma_sq = {}
    for k_idx in range(n_dev - 1):
        d_curr = dev_years[k_idx]
        d_next = dev_years[k_idx + 1]
        key = f"{d_curr}-{d_next}"
        f_k = link_factors.get(key)

        if f_k is None:
            sigma_sq[k_idx] = None
            continue

        # Get accident years that have both periods observed
        both = triangle_masked[[d_curr, d_next]].dropna()
        if len(both) <= 1:
            sigma_sq[k_idx] = 0.0
            continue

        c_ik = both[d_curr].values
        c_ik_next = both[d_next].values

        # Individual link ratios
        individual_factors = c_ik_next / np.where(c_ik == 0, np.nan, c_ik)
        valid = ~np.isnan(individual_factors)

        if valid.sum() <= 1:
            sigma_sq[k_idx] = 0.0
            continue

        residuals = c_ik[valid] * (individual_factors[valid] - f_k) ** 2
        sigma_sq[k_idx] = float(residuals.sum() / max(valid.sum() - 1, 1))

    # For the last period, use extrapolation: sigma^2_{n-2} = min(sigma^2_{n-3}^2 / sigma^2_{n-4}, ...)
    # Simplified: use the last available sigma_sq
    if n_dev >= 3 and sigma_sq.get(n_dev - 2) is None:
        sigma_sq[n_dev - 2] = sigma_sq.get(n_dev - 3, 0.0)

    # Compute Mack SE for each accident year's IBNR
    mack_results = {}
    for ay in acc_years:
        ay_str = str(ay)
        ult_data = ultimate_losses.get(ay_str, {})
        current_paid = ult_data.get("current_paid", 0.0)
        ultimate = ult_data.get("ultimate")
        latest_dev = ult_data.get("latest_dev_year")

        if ultimate is None or latest_dev is None:
            mack_results[ay_str] = {
                "ibnr": 0.0,
                "se": None,
                "ci_lower": None,
                "ci_upper": None,
            }
            continue

        ibnr = ultimate - current_paid

        # Find the index of latest_dev in dev_years
        if latest_dev not in dev_years:
            mack_results[ay_str] = {
                "ibnr": round(ibnr, 2),
                "se": None,
                "ci_lower": None,
                "ci_upper": None,
            }
            continue

        dev_idx = dev_years.index(latest_dev)

        # Mack's formula for process variance of the ultimate:
        # Var(R_i) = C_{i,ultimate}^2 * sum_{k=I_i}^{n-1} (sigma^2_k / f_k^2) * (1/C_{i,k} + 1/S_k)
        # where S_k = sum of C_{j,k} for accident years with data at period k
        variance = 0.0
        c_i_current = current_paid

        for k_idx in range(dev_idx, n_dev - 1):
            d_curr = dev_years[k_idx]
            d_next = dev_years[k_idx + 1]
            key = f"{d_curr}-{d_next}"
            f_k = link_factors.get(key)
            s2_k = sigma_sq.get(k_idx)

            if f_k is None or f_k == 0 or s2_k is None:
                continue

            # S_k = sum of C_{j,k} for all accident years with data at period k
            col_data = triangle_masked[d_curr].dropna()
            s_k = float(col_data.sum()) if not col_data.empty else 0.0

            if s_k == 0:
                continue

            # Accumulate variance term
            inv_c = 1.0 / c_i_current if c_i_current > 0 else 0.0
            inv_s = 1.0 / s_k
            variance += (s2_k / (f_k ** 2)) * (inv_c + inv_s)

            # Update c_i_current through the chain
            c_i_current *= f_k

        se = float(np.sqrt(variance)) * ultimate if variance > 0 else 0.0

        ci_lower = max(0, ibnr - 1.96 * se)
        ci_upper = ibnr + 1.96 * se

        mack_results[ay_str] = {
            "ibnr": round(ibnr, 2),
            "se": round(se, 2),
            "ci_lower": round(ci_lower, 2),
            "ci_upper": round(ci_upper, 2),
        }

    return mack_results


def _empty_result() -> Dict[str, Any]:
    return {
        "triangle": {},
        "age_to_age_factors": {},
        "cumulative_factors": {},
        "ultimate_losses": {},
        "ibnr_reserves": {},
        "mack_standard_errors": {},
    }
