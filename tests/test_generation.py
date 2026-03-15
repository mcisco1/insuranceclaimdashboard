"""Tests for the data generation layer (src.generation)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.reference_data import (
    CLAIM_TYPES,
    DEPARTMENTS,
    DIAGNOSES,
    PROCEDURE_CATEGORIES,
    ROOT_CAUSES,
    SPECIALTIES,
    SPECIALTY_SEVERITY_MAP,
    STATES,
)
from src.generation.distributions import (
    days_to_close,
    days_to_report,
    inject_quality_issues,
    lognormal_losses,
    seasonal_modifier,
    severity_for_specialty,
)
from src.generation.synthetic_claims import generate_claims


# ── Reference data completeness ────────────────────────────────────────────


class TestReferenceDataCompleteness:
    """All reference lists are non-empty and have required keys."""

    def test_specialties_non_empty_and_keys(self):
        assert len(SPECIALTIES) > 0
        for s in SPECIALTIES:
            assert "specialty_id" in s
            assert "name" in s
            assert "risk_tier" in s

    def test_states_non_empty_and_keys(self):
        assert len(STATES) > 0
        for s in STATES:
            assert "state_code" in s
            assert "name" in s
            assert "region" in s
            assert "urban_rural" in s

    def test_diagnoses_non_empty_and_keys(self):
        assert len(DIAGNOSES) > 0
        for d in DIAGNOSES:
            assert "diagnosis_id" in d
            assert "code" in d
            assert "category" in d
            assert "severity_weight" in d

    def test_root_causes_non_empty_and_keys(self):
        assert len(ROOT_CAUSES) > 0
        for r in ROOT_CAUSES:
            assert "root_cause_id" in r
            assert "category" in r
            assert "preventability" in r

    def test_departments_non_empty(self):
        assert len(DEPARTMENTS) > 0

    def test_claim_types_non_empty(self):
        assert len(CLAIM_TYPES) > 0

    def test_procedure_categories_non_empty(self):
        assert len(PROCEDURE_CATEGORIES) > 0

    def test_specialty_severity_map_covers_specialties(self):
        for s in SPECIALTIES:
            assert s["name"] in SPECIALTY_SEVERITY_MAP, (
                f"Specialty '{s['name']}' missing from SPECIALTY_SEVERITY_MAP"
            )


# ── Severity distribution ──────────────────────────────────────────────────


class TestSeverityDistribution:
    """severity_for_specialty returns values in {1, 2, 3, 4, 5}."""

    @pytest.mark.parametrize("specialty", list(SPECIALTY_SEVERITY_MAP.keys()))
    def test_severity_in_range(self, specialty):
        rng = np.random.default_rng(42)
        values = [severity_for_specialty(specialty, rng) for _ in range(200)]
        assert all(1 <= v <= 5 for v in values)

    def test_unknown_specialty_returns_valid(self):
        rng = np.random.default_rng(42)
        values = [severity_for_specialty("UnknownSpec", rng) for _ in range(200)]
        assert all(1 <= v <= 5 for v in values)


# ── Lognormal losses ───────────────────────────────────────────────────────


class TestLognormalLosses:
    """Output is within expected range [500, 5M]."""

    def test_scalar_severity(self):
        rng = np.random.default_rng(42)
        losses = lognormal_losses(1000, 3, rng)
        assert losses.min() >= 500
        assert losses.max() <= 5_000_000

    def test_array_severity(self):
        rng = np.random.default_rng(42)
        severities = np.array([1, 2, 3, 4, 5] * 200)
        losses = lognormal_losses(1000, severities, rng)
        assert losses.min() >= 500
        assert losses.max() <= 5_000_000

    def test_output_length(self):
        rng = np.random.default_rng(42)
        losses = lognormal_losses(50, 2, rng)
        assert len(losses) == 50


# ── Seasonal modifier ──────────────────────────────────────────────────────


class TestSeasonalModifier:
    """Values between 0.7 and 1.3."""

    def test_range_across_year(self):
        for month in range(1, 13):
            for day in [1, 15, 28]:
                date = pd.Timestamp(f"2023-{month:02d}-{day:02d}")
                mod = seasonal_modifier(date)
                assert 0.7 <= mod <= 1.3, f"Out of range for {date}: {mod}"

    def test_winter_peak(self):
        winter = seasonal_modifier(pd.Timestamp("2023-01-15"))
        summer = seasonal_modifier(pd.Timestamp("2023-07-15"))
        assert winter > summer, "Winter modifier should exceed summer"

    def test_numpy_datetime_input(self):
        dt = np.datetime64("2023-06-15")
        mod = seasonal_modifier(dt)
        assert 0.7 <= mod <= 1.3


# ── Days to report ──────────────────────────────────────────────────────────


class TestDaysToReport:
    """Positive values, median roughly ~7."""

    def test_positive_values_scalar(self):
        rng = np.random.default_rng(42)
        vals = days_to_report(3, rng)
        assert (vals >= 0).all()

    def test_positive_values_array(self):
        rng = np.random.default_rng(42)
        severities = np.array([1, 2, 3, 4, 5] * 200)
        vals = days_to_report(severities, rng)
        assert (vals >= 0).all()

    def test_median_roughly_seven(self):
        rng = np.random.default_rng(42)
        severities = np.array([3] * 5000)
        vals = days_to_report(severities, rng)
        median = np.median(vals)
        assert 3 <= median <= 15, f"Median {median} not near 7"


# ── Days to close ──────────────────────────────────────────────────────────


class TestDaysToClose:
    """Positive values, correlated with severity."""

    def test_positive_values(self):
        rng = np.random.default_rng(42)
        severities = np.array([1, 2, 3, 4, 5] * 200)
        vals = days_to_close(severities, rng)
        assert (vals >= 1).all()

    def test_correlated_with_severity(self):
        rng = np.random.default_rng(42)
        n = 3000
        low_sev = days_to_close(np.ones(n, dtype=int), rng)
        high_sev = days_to_close(np.full(n, 5, dtype=int), rng)
        assert np.median(high_sev) > np.median(low_sev), (
            "Higher severity should have longer close times"
        )


# ── Inject quality issues ──────────────────────────────────────────────────


class TestInjectQualityIssues:
    """Issues are actually injected (negatives, mixed case, etc.)."""

    def test_negative_paid_injected(self, sample_claims_df):
        rng = np.random.default_rng(42)
        df = sample_claims_df.copy()
        # Ensure all positive before injection
        df["paid_amount"] = df["paid_amount"].abs()
        inject_quality_issues(df, rng)
        assert (df["paid_amount"] < 0).any(), "No negative paid amounts injected"

    def test_mixed_case_state_injected(self, sample_claims_df):
        rng = np.random.default_rng(42)
        df = sample_claims_df.copy()
        df["state_code"] = df["state_code"].str.upper()
        inject_quality_issues(df, rng)
        non_upper = (df["state_code"] != df["state_code"].str.upper()).sum()
        assert non_upper > 0, "No mixed-case state codes injected"

    def test_text_severity_injected(self, sample_claims_df):
        rng = np.random.default_rng(42)
        df = sample_claims_df.copy()
        df["severity"] = df["severity"].astype(int)
        inject_quality_issues(df, rng)
        text_mask = pd.to_numeric(df["severity"], errors="coerce").isna() & df["severity"].notna()
        assert text_mask.any(), "No text severity values injected"

    def test_null_diagnosis_injected(self, sample_claims_df):
        rng = np.random.default_rng(42)
        df = sample_claims_df.copy()
        df["diagnosis_id"] = df["diagnosis_id"].fillna("DX01")
        inject_quality_issues(df, rng)
        assert df["diagnosis_id"].isna().any(), "No null diagnosis_id injected"


# ── Generate claims creates files ───────────────────────────────────────────


class TestGenerateClaimsCreatesFiles:
    """Generates CSVs in the expected directory."""

    def test_output_csvs_exist(self, tmp_dir, monkeypatch):
        """generate_claims writes five CSVs to data/raw/."""
        # Monkeypatch RAW_DIR to use the tmp directory
        import config
        monkeypatch.setattr(config, "RAW_DIR", tmp_dir)

        # Also need to monkeypatch the module-level import in synthetic_claims
        import src.generation.synthetic_claims as sc_module
        monkeypatch.setattr(sc_module, "RAW_DIR", tmp_dir)

        result = generate_claims(n=50, seed=42)

        expected_files = [
            "claims.csv",
            "providers.csv",
            "regions.csv",
            "diagnoses.csv",
            "root_causes.csv",
        ]
        for fname in expected_files:
            assert (tmp_dir / fname).exists(), f"{fname} not found in output dir"

        # Verify returned dict has same keys
        assert set(result.keys()) == {
            "claims", "providers", "regions", "diagnoses", "root_causes"
        }

    def test_generated_claims_row_count(self, tmp_dir, monkeypatch):
        import config
        monkeypatch.setattr(config, "RAW_DIR", tmp_dir)

        import src.generation.synthetic_claims as sc_module
        monkeypatch.setattr(sc_module, "RAW_DIR", tmp_dir)

        result = generate_claims(n=100, seed=99)
        assert len(result["claims"]) == 100
