"""Time-series forecasting for insurance claim volumes.

Uses Holt-Winters (triple exponential smoothing) to forecast monthly
claim counts.  Falls back to simple exponential smoothing when fewer
than 24 months of history are available.

Usage
-----
    from src.models.forecasting import forecast_claims
    results = forecast_claims(df, periods=12)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing,
    SimpleExpSmoothing,
)

from config import FORECAST_PERIODS

logger = logging.getLogger(__name__)

# Minimum months required for Holt-Winters with seasonality
_MIN_MONTHS_HW = 24
# Seasonal period (monthly data)
_SEASONAL_PERIOD = 12


# ── Helpers ──────────────────────────────────────────────────────────────

def _build_monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate claims into monthly counts.

    Parameters
    ----------
    df : pd.DataFrame
        Claims DataFrame with an *incident_date* column.

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame (``claim_count``) indexed by month-end
        ``DatetimeIndex``.
    """
    work = df.copy()
    work["incident_date"] = pd.to_datetime(work["incident_date"], errors="coerce")
    work = work.dropna(subset=["incident_date"])

    if work.empty:
        raise ValueError("No valid incident dates found in the DataFrame.")

    monthly = (
        work
        .set_index("incident_date")
        .resample("ME")
        .size()
        .rename("claim_count")
        .to_frame()
    )

    # Fill any interior gaps with zero (exterior gaps are unlikely)
    full_range = pd.date_range(
        start=monthly.index.min(),
        end=monthly.index.max(),
        freq="ME",
    )
    monthly = monthly.reindex(full_range, fill_value=0)
    monthly.index.name = "month"

    return monthly


def _compute_confidence_intervals(
    residuals: np.ndarray,
    forecast: np.ndarray,
    periods: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Generate confidence intervals using residual standard deviation.

    Parameters
    ----------
    residuals : np.ndarray
        In-sample residuals from the fitted model.
    forecast : np.ndarray
        Point forecast values.
    periods : int
        Number of forecast steps.

    Returns
    -------
    dict
        ``confidence_80`` and ``confidence_95``, each containing
        ``lower`` and ``upper`` arrays.
    """
    resid_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    # Scale uncertainty by sqrt of forecast horizon step
    horizon_scale = np.sqrt(np.arange(1, periods + 1))

    z_80 = 1.2816  # ~80% CI
    z_95 = 1.9600  # ~95% CI

    margin_80 = z_80 * resid_std * horizon_scale
    margin_95 = z_95 * resid_std * horizon_scale

    return {
        "confidence_80": {
            "lower": np.maximum(forecast - margin_80, 0),
            "upper": forecast + margin_80,
        },
        "confidence_95": {
            "lower": np.maximum(forecast - margin_95, 0),
            "upper": forecast + margin_95,
        },
    }


# ── Main Forecasting Function ───────────────────────────────────────────

def forecast_claims(
    df: pd.DataFrame,
    periods: int = FORECAST_PERIODS,
) -> Dict[str, Any]:
    """Forecast monthly claim counts using exponential smoothing.

    Uses Holt-Winters (additive trend + additive seasonality, period=12)
    when at least 24 months of data are available.  Falls back to simple
    exponential smoothing otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        Claims-summary DataFrame with an *incident_date* column.
    periods : int
        Number of future months to forecast (default from config).

    Returns
    -------
    dict
        Keys:
        - ``monthly_actuals``: historical monthly claim counts (DataFrame)
        - ``forecast``: ndarray of forecasted values
        - ``forecast_dates``: DatetimeIndex for forecast periods
        - ``confidence_80``: dict with ``lower`` / ``upper`` arrays
        - ``confidence_95``: dict with ``lower`` / ``upper`` arrays
        - ``model_params``: dict of fitted smoothing parameters
        - ``aic``: float -- model AIC
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # ── Build monthly time series ────────────────────────────────────
    monthly = _build_monthly_counts(df)
    n_months = len(monthly)
    logger.info("Monthly time series has %d observations.", n_months)

    series = monthly["claim_count"].astype(float)

    # ── Fit model ────────────────────────────────────────────────────
    use_hw = n_months >= _MIN_MONTHS_HW

    if use_hw:
        logger.info(
            "Using Holt-Winters (additive trend + additive seasonality, "
            "period=%d).",
            _SEASONAL_PERIOD,
        )
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=_SEASONAL_PERIOD,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)

        model_params = {
            "alpha": float(fitted.params.get("smoothing_level", np.nan)),
            "beta": float(fitted.params.get("smoothing_trend", np.nan)),
            "gamma": float(fitted.params.get("smoothing_seasonal", np.nan)),
            "method": "holt_winters",
        }
    else:
        logger.warning(
            "Only %d months of data (< %d). Falling back to simple "
            "exponential smoothing.",
            n_months,
            _MIN_MONTHS_HW,
        )
        model = SimpleExpSmoothing(
            series,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)

        model_params = {
            "alpha": float(fitted.params.get("smoothing_level", np.nan)),
            "beta": None,
            "gamma": None,
            "method": "simple_exponential_smoothing",
        }

    # ── Generate forecast ────────────────────────────────────────────
    forecast_values = fitted.forecast(periods)
    forecast_arr = np.array(forecast_values, dtype=float)

    # Build date index for forecast periods
    last_date = monthly.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.offsets.MonthEnd(1),
        periods=periods,
        freq="ME",
    )

    # ── Confidence intervals ─────────────────────────────────────────
    residuals = np.array(fitted.resid, dtype=float)
    # Remove NaN residuals (common at the start of the series)
    residuals = residuals[~np.isnan(residuals)]

    ci = _compute_confidence_intervals(residuals, forecast_arr, periods)

    # ── AIC ──────────────────────────────────────────────────────────
    try:
        aic_value = float(fitted.aic)
    except (AttributeError, TypeError):
        # SimpleExpSmoothing may not expose AIC in all statsmodels versions
        aic_value = np.nan

    logger.info(
        "Forecast generated for %d periods. AIC=%.2f", periods, aic_value,
    )

    return {
        "monthly_actuals": monthly,
        "forecast": forecast_arr,
        "forecast_dates": forecast_dates,
        "confidence_80": ci["confidence_80"],
        "confidence_95": ci["confidence_95"],
        "model_params": model_params,
        "aic": aic_value,
    }
