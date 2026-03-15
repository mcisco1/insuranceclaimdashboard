"""Tests for the ML models (src.models)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.severity_predictor import train_severity_model
from src.models.forecasting import forecast_claims
from src.models.anomaly_detection import detect_anomalies


# ── Severity predictor ────────────────────────────────────────────────────


class TestSeverityPredictorAccuracy:
    """Accuracy > 0.2 (better than random for 5 classes)."""

    def test_rf_accuracy_above_random(self, sample_claims_df):
        result = train_severity_model(sample_claims_df, seed=42)
        assert result["rf_accuracy"] >= 0.2, (
            f"RF accuracy {result['rf_accuracy']:.3f} should be at least random (0.2)"
        )

    def test_lr_accuracy_above_random(self, sample_claims_df):
        result = train_severity_model(sample_claims_df, seed=42)
        assert result["lr_accuracy"] >= 0.2, (
            f"LR accuracy {result['lr_accuracy']:.3f} should be at least random (0.2)"
        )

    def test_result_has_models(self, sample_claims_df):
        result = train_severity_model(sample_claims_df, seed=42)
        assert result["rf_model"] is not None
        assert result["lr_model"] is not None

    def test_result_has_reports(self, sample_claims_df):
        result = train_severity_model(sample_claims_df, seed=42)
        assert isinstance(result["rf_report"], dict)
        assert isinstance(result["lr_report"], dict)

    def test_confusion_matrix_shape(self, sample_claims_df):
        result = train_severity_model(sample_claims_df, seed=42)
        cm = result["confusion_matrix"]
        assert cm.shape == (5, 5)


class TestSeverityPredictorFeatures:
    """Feature importances sum to ~1.0."""

    def test_importances_sum_to_one(self, sample_claims_df):
        result = train_severity_model(sample_claims_df, seed=42)
        imp_df = result["feature_importances"]
        total = imp_df["importance"].sum()
        assert abs(total - 1.0) < 0.01, (
            f"Feature importances sum to {total:.4f}, expected ~1.0"
        )

    def test_importances_non_negative(self, sample_claims_df):
        result = train_severity_model(sample_claims_df, seed=42)
        imp_df = result["feature_importances"]
        assert (imp_df["importance"] >= 0).all()

    def test_features_used_non_empty(self, sample_claims_df):
        result = train_severity_model(sample_claims_df, seed=42)
        assert len(result["features_used"]) > 0


# ── Forecasting ───────────────────────────────────────────────────────────


class TestForecastOutputLength:
    """Forecast has correct number of periods."""

    def test_default_periods(self, sample_claims_df):
        result = forecast_claims(sample_claims_df)
        # Default is FORECAST_PERIODS = 12
        assert len(result["forecast"]) == 12

    def test_custom_periods(self, sample_claims_df):
        result = forecast_claims(sample_claims_df, periods=6)
        assert len(result["forecast"]) == 6

    def test_forecast_dates_match_length(self, sample_claims_df):
        periods = 8
        result = forecast_claims(sample_claims_df, periods=periods)
        assert len(result["forecast_dates"]) == periods
        assert len(result["forecast"]) == periods

    def test_result_has_actuals(self, sample_claims_df):
        result = forecast_claims(sample_claims_df)
        assert "monthly_actuals" in result
        assert isinstance(result["monthly_actuals"], pd.DataFrame)

    def test_result_has_confidence_intervals(self, sample_claims_df):
        result = forecast_claims(sample_claims_df)
        assert "confidence_80" in result
        assert "confidence_95" in result
        for key in ("confidence_80", "confidence_95"):
            assert "lower" in result[key]
            assert "upper" in result[key]


class TestForecastPositive:
    """Forecast values are positive."""

    def test_forecast_values_positive(self, sample_claims_df):
        result = forecast_claims(sample_claims_df)
        forecast = result["forecast"]
        # Forecasted claim counts should be non-negative
        assert (forecast >= 0).all(), (
            f"Forecast has negative values: min={forecast.min()}"
        )

    def test_confidence_lower_non_negative(self, sample_claims_df):
        result = forecast_claims(sample_claims_df)
        assert (result["confidence_80"]["lower"] >= 0).all()
        assert (result["confidence_95"]["lower"] >= 0).all()


# ── Anomaly detection ─────────────────────────────────────────────────────


class TestAnomalyDetectionRate:
    """Anomaly rate is close to contamination parameter."""

    def test_anomaly_rate_near_contamination(self, sample_claims_df):
        contamination = 0.05
        result = detect_anomalies(
            sample_claims_df, contamination=contamination, seed=42
        )
        rate = result["anomaly_rate"] / 100.0  # convert from pct
        # Allow generous tolerance due to small sample size
        assert abs(rate - contamination) < 0.15, (
            f"Anomaly rate {rate:.3f} too far from contamination {contamination}"
        )

    def test_anomaly_count_positive(self, sample_claims_df):
        result = detect_anomalies(sample_claims_df, contamination=0.10, seed=42)
        assert result["anomaly_count"] > 0

    def test_anomaly_count_less_than_total(self, sample_claims_df):
        result = detect_anomalies(sample_claims_df, contamination=0.05, seed=42)
        assert result["anomaly_count"] < len(sample_claims_df)

    def test_result_keys(self, sample_claims_df):
        result = detect_anomalies(sample_claims_df, seed=42)
        expected = {
            "anomaly_flags",
            "anomaly_count",
            "anomaly_rate",
            "anomaly_claims",
            "anomaly_scores",
            "feature_summary",
            "trend_breaks",
        }
        assert expected == set(result.keys())


class TestAnomalyFlagsType:
    """Flags are boolean."""

    def test_flags_are_boolean(self, sample_claims_df):
        result = detect_anomalies(sample_claims_df, seed=42)
        flags = result["anomaly_flags"]
        assert flags.dtype == bool

    def test_flags_length_matches_input(self, sample_claims_df):
        result = detect_anomalies(sample_claims_df, seed=42)
        assert len(result["anomaly_flags"]) == len(sample_claims_df)

    def test_scores_are_numeric(self, sample_claims_df):
        result = detect_anomalies(sample_claims_df, seed=42)
        scores = result["anomaly_scores"]
        assert np.issubdtype(scores.dtype, np.floating)

    def test_anomaly_claims_subset_of_input(self, sample_claims_df):
        result = detect_anomalies(sample_claims_df, seed=42)
        anomaly_ids = set(result["anomaly_claims"].index)
        input_ids = set(sample_claims_df.index)
        assert anomaly_ids.issubset(input_ids)
