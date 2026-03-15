"""Tests for the cleaning pipeline (src.cleaning)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cleaning.cleaner import (
    _cap_outlier_amounts,
    _convert_text_severity,
    _fill_null_diagnosis,
    _fix_negative_paid_amounts,
    _fix_swapped_dates,
    _standardize_state_codes,
    clean_claims,
)
from src.cleaning.validators import validate_claims


# ── Fix negative amounts ───────────────────────────────────────────────────


class TestFixNegativeAmounts:
    """Negatives become positive."""

    def test_negatives_become_positive(self, raw_claims_df):
        df = raw_claims_df.copy()
        assert (df["paid_amount"] < 0).any(), "Precondition: negatives present"

        _fix_negative_paid_amounts(df)

        assert (df["paid_amount"] >= 0).all(), "Some paid_amount still negative"

    def test_positive_values_unchanged(self, sample_claims_df):
        df = sample_claims_df.copy()
        original = df["paid_amount"].copy()
        _fix_negative_paid_amounts(df)
        pd.testing.assert_series_equal(df["paid_amount"], original)

    def test_returns_count(self, raw_claims_df):
        df = raw_claims_df.copy()
        neg_before = (df["paid_amount"] < 0).sum()
        count = _fix_negative_paid_amounts(df)
        assert count == neg_before


# ── Standardize state codes ────────────────────────────────────────────────


class TestStandardizeStateCodes:
    """All uppercase after cleaning."""

    def test_all_uppercase(self, raw_claims_df):
        df = raw_claims_df.copy()
        assert (df["state_code"] != df["state_code"].str.upper()).any(), (
            "Precondition: mixed-case present"
        )

        _standardize_state_codes(df)

        assert (df["state_code"] == df["state_code"].str.upper()).all()

    def test_already_uppercase_unchanged(self, sample_claims_df):
        df = sample_claims_df.copy()
        count = _standardize_state_codes(df)
        assert count == 0


# ── Convert text severity ──────────────────────────────────────────────────


class TestConvertTextSeverity:
    """'one', 'two', etc. become 1, 2, etc."""

    def test_text_to_numeric(self, raw_claims_df):
        df = raw_claims_df.copy()
        text_mask = pd.to_numeric(df["severity"], errors="coerce").isna() & df["severity"].notna()
        assert text_mask.any(), "Precondition: text severity values present"

        _convert_text_severity(df)

        assert df["severity"].dropna().apply(
            lambda x: isinstance(x, (int, np.integer)) or pd.notna(pd.to_numeric(x, errors="coerce"))
        ).all()

    def test_known_mappings(self):
        df = pd.DataFrame({"severity": ["one", "two", "three", "four", "five"]})
        _convert_text_severity(df)
        expected = [1, 2, 3, 4, 5]
        for i, val in enumerate(expected):
            assert int(df.at[i, "severity"]) == val

    def test_numeric_preserved(self, sample_claims_df):
        df = sample_claims_df.copy()
        original = df["severity"].astype(int).tolist()
        _convert_text_severity(df)
        result = df["severity"].astype(int).tolist()
        assert result == original


# ── Fix swapped dates ──────────────────────────────────────────────────────


class TestFixSwappedDates:
    """report_date >= incident_date after cleaning."""

    def test_no_swapped_after_fix(self, raw_claims_df):
        df = raw_claims_df.copy()
        df["incident_date"] = pd.to_datetime(df["incident_date"])
        df["report_date"] = pd.to_datetime(df["report_date"])
        assert (df["report_date"] < df["incident_date"]).any(), (
            "Precondition: swapped dates present"
        )

        _fix_swapped_dates(df)

        assert (df["report_date"] >= df["incident_date"]).all()

    def test_correct_dates_unchanged(self, sample_claims_df):
        df = sample_claims_df.copy()
        df["incident_date"] = pd.to_datetime(df["incident_date"])
        df["report_date"] = pd.to_datetime(df["report_date"])
        orig_incident = df["incident_date"].copy()
        orig_report = df["report_date"].copy()

        _fix_swapped_dates(df)

        pd.testing.assert_series_equal(df["incident_date"], orig_incident)
        pd.testing.assert_series_equal(df["report_date"], orig_report)


# ── Fill null diagnosis ────────────────────────────────────────────────────


class TestFillNullDiagnosis:
    """No nulls in diagnosis after cleaning."""

    def test_no_nulls_after_fill(self, raw_claims_df):
        df = raw_claims_df.copy()
        assert df["diagnosis_id"].isna().any(), "Precondition: nulls present"

        _fill_null_diagnosis(df)

        assert df["diagnosis_id"].isna().sum() == 0

    def test_filled_with_unknown(self, raw_claims_df):
        df = raw_claims_df.copy()
        null_indices = df.index[df["diagnosis_id"].isna()]
        _fill_null_diagnosis(df)
        for idx in null_indices:
            assert df.at[idx, "diagnosis_id"] == "UNKNOWN"


# ── Cap outliers ───────────────────────────────────────────────────────────


class TestCapOutliers:
    """No values > $5M after cleaning."""

    def test_no_values_over_cap(self, raw_claims_df):
        df = raw_claims_df.copy()
        # Fix negatives first so abs() comparison is clean
        df["paid_amount"] = df["paid_amount"].abs()
        assert (df["paid_amount"] > 5_000_000).any(), "Precondition: outliers present"

        _cap_outlier_amounts(df)

        assert (df["paid_amount"].abs() <= 5_000_000).all()

    def test_values_at_cap(self, raw_claims_df):
        df = raw_claims_df.copy()
        df["paid_amount"] = df["paid_amount"].abs()
        _cap_outlier_amounts(df)
        assert df["paid_amount"].max() <= 5_000_000


# ── Quality report generated ──────────────────────────────────────────────


class TestQualityReportGenerated:
    """Report file is created by the clean_claims pipeline."""

    def test_report_file_created(self, tmp_dir, sample_claims_df, monkeypatch):
        """Run a full cleaning pipeline and check that the report is written."""
        import config

        raw_dir = tmp_dir / "raw"
        cleaned_dir = tmp_dir / "cleaned"
        reports_dir = tmp_dir / "reports"
        raw_dir.mkdir()
        cleaned_dir.mkdir()
        reports_dir.mkdir()

        # Write sample data as CSV to raw_dir
        sample_claims_df.to_csv(raw_dir / "claims.csv", index=False)

        monkeypatch.setattr(config, "REPORTS_DIR", reports_dir)

        # Also patch REPORTS_DIR inside the cleaner module
        import src.cleaning.cleaner as cleaner_mod
        monkeypatch.setattr(cleaner_mod, "REPORTS_DIR", reports_dir)

        report = clean_claims(input_dir=raw_dir, output_dir=cleaned_dir)

        report_path = reports_dir / "data_quality_report.md"
        assert report_path.exists(), "Quality report file not created"
        assert report_path.stat().st_size > 0, "Quality report is empty"


# ── Validate claims ───────────────────────────────────────────────────────


class TestValidateClaims:
    """Validator catches known issues."""

    def test_catches_negative_paid(self, raw_claims_df):
        report = validate_claims(raw_claims_df)
        categories = report.category_counts
        assert "Negative Paid Amounts" in categories
        assert categories["Negative Paid Amounts"] > 0

    def test_catches_mixed_case(self, raw_claims_df):
        report = validate_claims(raw_claims_df)
        categories = report.category_counts
        assert "Mixed-Case State Codes" in categories
        assert categories["Mixed-Case State Codes"] > 0

    def test_catches_text_severity(self, raw_claims_df):
        report = validate_claims(raw_claims_df)
        categories = report.category_counts
        assert "Text Severity Values" in categories
        assert categories["Text Severity Values"] > 0

    def test_catches_swapped_dates(self, raw_claims_df):
        report = validate_claims(raw_claims_df)
        categories = report.category_counts
        assert "Swapped Dates" in categories
        assert categories["Swapped Dates"] > 0

    def test_catches_null_diagnosis(self, raw_claims_df):
        report = validate_claims(raw_claims_df)
        categories = report.category_counts
        assert "Null Diagnosis Codes" in categories
        assert categories["Null Diagnosis Codes"] > 0

    def test_catches_outliers(self, raw_claims_df):
        report = validate_claims(raw_claims_df)
        categories = report.category_counts
        assert "Extreme Outliers" in categories
        assert categories["Extreme Outliers"] > 0

    def test_clean_data_has_no_issues(self, sample_claims_df):
        report = validate_claims(sample_claims_df)
        # Clean data should have zero or very few issues
        # (only potential "Future Dates" depending on when tests run)
        for cat in [
            "Negative Paid Amounts",
            "Mixed-Case State Codes",
            "Text Severity Values",
            "Swapped Dates",
            "Null Diagnosis Codes",
            "Extreme Outliers",
        ]:
            assert report.category_counts.get(cat, 0) == 0, (
                f"Clean data should not have '{cat}' issues"
            )
