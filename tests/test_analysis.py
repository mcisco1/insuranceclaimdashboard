"""Tests for the analysis modules (src.analysis)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.frequency_severity import analyze_frequency_severity
from src.analysis.time_to_close import analyze_time_to_close
from src.analysis.loss_development import build_loss_triangle
from src.analysis.reporting_lag import analyze_reporting_lag
from src.analysis.repeat_incidents import analyze_repeat_incidents
from src.analysis.benchmarks import calculate_benchmarks
from src.analysis.scenario_analysis import run_scenarios


# ── Frequency / severity ──────────────────────────────────────────────────


class TestFrequencySeverityKeys:
    """Output has all expected keys."""

    def test_all_keys_present(self, sample_claims_df):
        result = analyze_frequency_severity(sample_claims_df)
        expected_keys = {
            "monthly_counts",
            "quarterly_counts",
            "yearly_counts",
            "severity_distribution",
            "severity_by_category",
            "severity_by_specialty",
            "frequency_severity_scatter",
            "loss_ratio",
            "yoy_metrics",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    def test_severity_distribution_subkeys(self, sample_claims_df):
        result = analyze_frequency_severity(sample_claims_df)
        sd = result["severity_distribution"]
        assert "counts" in sd
        assert "percentages" in sd
        assert "mean" in sd
        assert "median" in sd

    def test_yearly_counts_subkeys(self, sample_claims_df):
        result = analyze_frequency_severity(sample_claims_df)
        yc = result["yearly_counts"]
        assert "counts" in yc
        assert "yoy_change_pct" in yc

    def test_empty_df(self):
        result = analyze_frequency_severity(pd.DataFrame())
        assert isinstance(result, dict)
        assert "monthly_counts" in result


# ── Time to close ─────────────────────────────────────────────────────────


class TestTimeToCloseKeys:
    """Output has all expected keys."""

    def test_all_keys_present(self, sample_claims_df):
        result = analyze_time_to_close(sample_claims_df)
        expected_keys = {
            "close_time_distribution",
            "median_by_severity",
            "median_by_specialty",
            "median_by_region",
            "sla_compliance",
            "backlog_aging",
            "monthly_close_trend",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    def test_close_time_distribution_subkeys(self, sample_claims_df):
        result = analyze_time_to_close(sample_claims_df)
        ctd = result["close_time_distribution"]
        assert "bins" in ctd
        assert "counts" in ctd
        assert "median_overall" in ctd
        assert "mean_overall" in ctd

    def test_backlog_has_total_open(self, sample_claims_df):
        result = analyze_time_to_close(sample_claims_df)
        backlog = result["backlog_aging"]
        assert "total_open" in backlog
        assert "buckets" in backlog

    def test_empty_df(self):
        result = analyze_time_to_close(pd.DataFrame())
        assert isinstance(result, dict)
        assert "close_time_distribution" in result


# ── Loss triangle ─────────────────────────────────────────────────────────


class TestLossTriangleStructure:
    """Triangle is a proper structure with accident years as index."""

    def test_triangle_keys(self, sample_claims_df):
        result = build_loss_triangle(sample_claims_df)
        expected_keys = {
            "triangle",
            "age_to_age_factors",
            "cumulative_factors",
            "ultimate_losses",
            "ibnr_reserves",
        }
        assert expected_keys == set(result.keys())

    def test_triangle_has_accident_years(self, sample_claims_df):
        result = build_loss_triangle(sample_claims_df)
        triangle = result["triangle"]
        assert len(triangle) > 0, "Triangle should have at least one accident year"
        # Keys should be stringified years
        for key in triangle.keys():
            assert key.isdigit(), f"Triangle key '{key}' should be a year string"

    def test_triangle_values_are_numeric_or_none(self, sample_claims_df):
        result = build_loss_triangle(sample_claims_df)
        triangle = result["triangle"]
        for ay, row in triangle.items():
            for dy, val in row.items():
                assert val is None or isinstance(val, (int, float))

    def test_ibnr_has_total(self, sample_claims_df):
        result = build_loss_triangle(sample_claims_df)
        ibnr = result["ibnr_reserves"]
        assert "total" in ibnr
        assert "current_paid" in ibnr["total"]

    def test_empty_df(self):
        result = build_loss_triangle(pd.DataFrame())
        assert result["triangle"] == {}


# ── Reporting lag ─────────────────────────────────────────────────────────


class TestReportingLagValues:
    """Lag values are non-negative."""

    def test_lag_distribution_present(self, sample_claims_df):
        result = analyze_reporting_lag(sample_claims_df)
        assert "lag_distribution" in result
        ld = result["lag_distribution"]
        assert "median" in ld
        assert "mean" in ld

    def test_lag_values_nonnegative(self, sample_claims_df):
        result = analyze_reporting_lag(sample_claims_df)
        ld = result["lag_distribution"]
        assert ld["median"] >= 0
        assert ld["mean"] >= 0

    def test_lag_by_severity_present(self, sample_claims_df):
        result = analyze_reporting_lag(sample_claims_df)
        assert "lag_by_severity" in result

    def test_slow_reporters_present(self, sample_claims_df):
        result = analyze_reporting_lag(sample_claims_df)
        sr = result["slow_reporters"]
        assert "threshold_days" in sr
        assert "overall_median" in sr

    def test_empty_df(self):
        result = analyze_reporting_lag(pd.DataFrame())
        assert result["lag_distribution"]["median"] is None


# ── Repeat incidents ──────────────────────────────────────────────────────


class TestRepeatIncidentsRate:
    """Reopen rate is between 0 and 1 (expressed as percentage)."""

    def test_overall_reopen_rate_bounded(self, sample_claims_df):
        result = analyze_repeat_incidents(sample_claims_df)
        rate = result["overall_reopen_rate"]["reopen_rate_pct"]
        assert 0 <= rate <= 100

    def test_overall_reopen_rate_keys(self, sample_claims_df):
        result = analyze_repeat_incidents(sample_claims_df)
        orr = result["overall_reopen_rate"]
        assert "total_claims" in orr
        assert "reopened_claims" in orr
        assert "reopen_rate_pct" in orr

    def test_repeat_event_summary_present(self, sample_claims_df):
        result = analyze_repeat_incidents(sample_claims_df)
        res = result["repeat_event_summary"]
        assert "total_repeat_events" in res
        assert "repeat_rate_pct" in res

    def test_all_keys_present(self, sample_claims_df):
        result = analyze_repeat_incidents(sample_claims_df)
        expected = {
            "overall_reopen_rate",
            "reopen_by_provider",
            "reopen_by_root_cause",
            "reopen_by_specialty",
            "reopen_litigation_correlation",
            "repeat_event_summary",
        }
        assert expected.issubset(result.keys())

    def test_empty_df(self):
        result = analyze_repeat_incidents(pd.DataFrame())
        assert result["overall_reopen_rate"]["reopen_rate_pct"] == 0.0


# ── Benchmarks ────────────────────────────────────────────────────────────


class TestBenchmarksIndicators:
    """All PSI indicators present."""

    _EXPECTED_INDICATORS = [
        "surgical_complication_rate",
        "fall_rate",
        "medication_error_rate",
        "diagnostic_error_rate",
        "hospital_acquired_infection_rate",
    ]

    def test_all_psi_indicators_present(self, sample_claims_df):
        result = calculate_benchmarks(sample_claims_df)
        psi = result["psi_indicators"]
        for name in self._EXPECTED_INDICATORS:
            assert name in psi, f"Missing PSI indicator: {name}"

    def test_psi_indicator_keys(self, sample_claims_df):
        result = calculate_benchmarks(sample_claims_df)
        for name in self._EXPECTED_INDICATORS:
            ind = result["psi_indicators"][name]
            assert "numerator" in ind
            assert "denominator" in ind
            assert "rate_per_1000" in ind

    def test_facility_vs_benchmark_present(self, sample_claims_df):
        result = calculate_benchmarks(sample_claims_df)
        assert "facility_vs_benchmark" in result

    def test_benchmark_trends_present(self, sample_claims_df):
        result = calculate_benchmarks(sample_claims_df)
        assert "benchmark_trends" in result
        for name in self._EXPECTED_INDICATORS:
            assert name in result["benchmark_trends"]

    def test_empty_df(self):
        result = calculate_benchmarks(pd.DataFrame())
        assert "psi_indicators" in result


# ── Scenario analysis ────────────────────────────────────────────────────


class TestScenarioAnalysisFormat:
    """Scenarios have baseline, projected, change values."""

    def test_all_scenario_keys(self, sample_claims_df):
        result = run_scenarios(sample_claims_df)
        expected = {
            "close_time_reduction",
            "severity_reduction",
            "litigation_reduction",
            "combined_scenario",
        }
        assert expected == set(result.keys())

    def test_close_time_has_baseline_and_scenarios(self, sample_claims_df):
        result = run_scenarios(sample_claims_df)
        ctr = result["close_time_reduction"]
        assert "baseline" in ctr
        assert "scenarios" in ctr

    def test_severity_has_baseline_and_scenarios(self, sample_claims_df):
        result = run_scenarios(sample_claims_df)
        sr = result["severity_reduction"]
        assert "baseline" in sr
        assert "scenarios" in sr

    def test_litigation_has_baseline_and_scenarios(self, sample_claims_df):
        result = run_scenarios(sample_claims_df)
        lr = result["litigation_reduction"]
        assert "baseline" in lr
        assert "scenarios" in lr

    def test_combined_scenario_has_components(self, sample_claims_df):
        result = run_scenarios(sample_claims_df)
        combined = result["combined_scenario"]
        assert "baseline_total_incurred" in combined
        assert "new_total_incurred" in combined
        assert "total_savings" in combined
        assert "components" in combined
        assert "absolute_change" in combined
        assert "pct_change" in combined

    def test_combined_scenario_savings_positive(self, sample_claims_df):
        result = run_scenarios(sample_claims_df)
        combined = result["combined_scenario"]
        assert combined["total_savings"] >= 0, "Combined savings should be non-negative"

    def test_empty_df(self):
        result = run_scenarios(pd.DataFrame())
        assert "close_time_reduction" in result
