"""Anomaly detection for insurance claims using Isolation Forest.

Identifies unusual claims based on financial and operational features.
Provides anomaly flags, scores, and summary statistics comparing
anomalous claims to the normal population.

Usage
-----
    from src.models.anomaly_detection import detect_anomalies
    results = detect_anomalies(df)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import ANOMALY_CONTAMINATION, RANDOM_SEED

logger = logging.getLogger(__name__)

# ── Feature configuration ────────────────────────────────────────────────

_FEATURE_SPECS: List[Dict[str, Any]] = [
    {"source": "paid_amount",    "output": "log_paid_amount", "transform": "log1p"},
    {"source": "days_to_close",  "output": "days_to_close",  "transform": "numeric"},
    {"source": "days_to_report", "output": "days_to_report", "transform": "numeric"},
    {"source": "severity_level", "output": "severity_level", "transform": "numeric"},
    {"source": "litigation_flag","output": "litigation_flag", "transform": "numeric"},
]

# Fallback column names for common naming variations
_COLUMN_ALIASES: Dict[str, List[str]] = {
    "severity_level": ["severity_level", "severity"],
    "days_to_report": ["days_to_report", "report_lag_days"],
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _resolve_column_name(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Find the actual column name in *df* for the given canonical name."""
    if canonical in df.columns:
        return canonical
    for alias in _COLUMN_ALIASES.get(canonical, []):
        if alias in df.columns:
            return alias
    return None


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """Construct the feature matrix for anomaly detection.

    Missing values are imputed with the column median.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        The feature matrix (same index as *df*) and the list of feature
        names actually used.
    """
    feat = pd.DataFrame(index=df.index)
    features_used: List[str] = []

    for spec in _FEATURE_SPECS:
        col = _resolve_column_name(df, spec["source"])
        if col is None:
            logger.warning(
                "Column '%s' not found in DataFrame; skipping.", spec["source"],
            )
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        if spec["transform"] == "log1p":
            values = np.log1p(values.clip(lower=0))

        # Impute missing with median
        median_val = values.median()
        if pd.isna(median_val):
            median_val = 0.0
        values = values.fillna(median_val)

        out_name = spec["output"]
        feat[out_name] = values
        features_used.append(out_name)

    return feat, features_used


def _summarize_differences(
    df: pd.DataFrame,
    anomaly_mask: pd.Series,
) -> pd.DataFrame:
    """Compare anomalous vs normal claims across numeric columns.

    Returns a DataFrame with columns: feature, normal_mean, anomaly_mean,
    normal_median, anomaly_median, pct_diff_mean.
    """
    numeric_cols = [
        "paid_amount", "days_to_close", "days_to_report",
        "severity_level", "litigation_flag",
    ]
    # Resolve actual column names
    resolved = []
    for col in numeric_cols:
        actual = _resolve_column_name(df, col)
        if actual is not None:
            resolved.append(actual)

    if not resolved:
        return pd.DataFrame()

    normal = df.loc[~anomaly_mask, resolved].apply(
        pd.to_numeric, errors="coerce",
    )
    anomalous = df.loc[anomaly_mask, resolved].apply(
        pd.to_numeric, errors="coerce",
    )

    records = []
    for col in resolved:
        n_mean = float(normal[col].mean()) if not normal.empty else 0.0
        a_mean = float(anomalous[col].mean()) if not anomalous.empty else 0.0
        n_med = float(normal[col].median()) if not normal.empty else 0.0
        a_med = float(anomalous[col].median()) if not anomalous.empty else 0.0

        pct_diff = (
            ((a_mean - n_mean) / n_mean * 100) if n_mean != 0 else 0.0
        )

        records.append({
            "feature": col,
            "normal_mean": round(n_mean, 2),
            "anomaly_mean": round(a_mean, 2),
            "normal_median": round(n_med, 2),
            "anomaly_median": round(a_med, 2),
            "pct_diff_mean": round(pct_diff, 2),
        })

    return pd.DataFrame(records)


def _compute_trend_breaks(
    df: pd.DataFrame,
    anomaly_mask: pd.Series,
) -> pd.DataFrame:
    """Compute monthly anomaly rate to help identify trend breaks.

    Returns a DataFrame indexed by month with columns: total_claims,
    anomaly_count, anomaly_rate.
    """
    date_col = None
    for candidate in ("incident_date", "report_date"):
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        return pd.DataFrame()

    work = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "is_anomaly": anomaly_mask.astype(int),
    })
    work = work.dropna(subset=["date"])

    if work.empty:
        return pd.DataFrame()

    work["month"] = work["date"].dt.to_period("M")

    monthly = work.groupby("month").agg(
        total_claims=("is_anomaly", "count"),
        anomaly_count=("is_anomaly", "sum"),
    )
    monthly["anomaly_rate"] = (
        monthly["anomaly_count"] / monthly["total_claims"] * 100
    ).round(2)

    monthly.index = monthly.index.astype(str)

    return monthly


# ── Main Detection Function ─────────────────────────────────────────────

def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = ANOMALY_CONTAMINATION,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Detect anomalous claims using Isolation Forest.

    Parameters
    ----------
    df : pd.DataFrame
        Claims-summary DataFrame.  Expected columns include
        *paid_amount*, *days_to_close*, *days_to_report*
        (or *report_lag_days*), *severity_level* (or *severity*),
        and *litigation_flag*.
    contamination : float
        Expected proportion of anomalies (default from config).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys:
        - ``anomaly_flags``: boolean Series aligned to *df* index
        - ``anomaly_count``: int
        - ``anomaly_rate``: float (percentage)
        - ``anomaly_claims``: DataFrame of flagged claims with scores
        - ``anomaly_scores``: Series of anomaly scores for all claims
        - ``feature_summary``: DataFrame comparing anomalous vs normal
        - ``trend_breaks``: DataFrame of monthly anomaly rates
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # ── Build feature matrix ─────────────────────────────────────────
    X_raw, features_used = _build_feature_matrix(df)

    if X_raw.shape[1] == 0:
        raise ValueError(
            "No valid features could be built.  Ensure the DataFrame "
            "contains at least some of: paid_amount, days_to_close, "
            "days_to_report, severity_level, litigation_flag."
        )

    logger.info(
        "Anomaly detection using %d features on %d claims.",
        len(features_used),
        len(X_raw),
    )

    # ── Scale features ───────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ── Fit Isolation Forest ─────────────────────────────────────────
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=seed,
        n_estimators=200,
        n_jobs=-1,
    )
    predictions = iso_forest.fit_predict(X_scaled)
    raw_scores = iso_forest.decision_function(X_scaled)

    # Isolation Forest returns -1 for anomalies, 1 for normal
    anomaly_flags = pd.Series(predictions == -1, index=df.index, name="is_anomaly")
    anomaly_scores = pd.Series(raw_scores, index=df.index, name="anomaly_score")

    anomaly_count = int(anomaly_flags.sum())
    anomaly_rate = round(anomaly_count / len(df) * 100, 2) if len(df) > 0 else 0.0

    logger.info(
        "Detected %d anomalies (%.2f%%) out of %d claims.",
        anomaly_count,
        anomaly_rate,
        len(df),
    )

    # ── Build anomaly claims DataFrame ───────────────────────────────
    anomaly_claims = df.loc[anomaly_flags].copy()
    anomaly_claims["anomaly_score"] = anomaly_scores.loc[anomaly_flags]
    anomaly_claims = anomaly_claims.sort_values("anomaly_score", ascending=True)

    # ── Feature summary ──────────────────────────────────────────────
    feature_summary = _summarize_differences(df, anomaly_flags)

    # ── Trend breaks ─────────────────────────────────────────────────
    trend_breaks = _compute_trend_breaks(df, anomaly_flags)

    return {
        "anomaly_flags": anomaly_flags,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
        "anomaly_claims": anomaly_claims,
        "anomaly_scores": anomaly_scores,
        "feature_summary": feature_summary,
        "trend_breaks": trend_breaks,
    }
