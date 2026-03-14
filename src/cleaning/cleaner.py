"""Cleaning pipeline for raw insurance claims data.

Reads CSVs from ``data/raw/``, applies deterministic cleaning steps,
validates before and after, saves cleaned output to ``data/cleaned/``,
and writes a professional data-quality report.

Usage
-----
    python -m src.cleaning.cleaner          # from project root
    # or
    from src.cleaning.cleaner import clean_claims
    report = clean_claims()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import CLEANED_DIR, RAW_DIR, RANDOM_SEED, REPORTS_DIR
from src.cleaning.validators import DataQualityReport, validate_claims


# ── Severity text-to-numeric mapping ──────────────────────────────────────

_TEXT_TO_NUMERIC: Dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

# Maximum allowable paid_amount (used to cap outliers)
_PAID_AMOUNT_CAP = 5_000_000


# ── Individual Cleaning Steps ─────────────────────────────────────────────

def _fix_negative_paid_amounts(df: pd.DataFrame) -> int:
    """Convert negative paid_amount values to their absolute value.

    Returns the number of rows affected.
    """
    mask = df["paid_amount"] < 0
    count = int(mask.sum())
    if count > 0:
        df.loc[mask, "paid_amount"] = df.loc[mask, "paid_amount"].abs()
    return count


def _standardize_state_codes(df: pd.DataFrame) -> int:
    """Uppercase all state_code values.

    Returns the number of rows that were changed.
    """
    original = df["state_code"].copy()
    df["state_code"] = df["state_code"].str.upper()
    changed = int((original != df["state_code"]).sum())
    return changed


def _convert_text_severity(df: pd.DataFrame) -> int:
    """Map text severity labels ('one' .. 'five') to integers 1-5.

    Returns the number of rows converted.
    """
    # Force column to object dtype so we can mix types during conversion
    sev = df["severity"].astype(object)
    text_mask = pd.to_numeric(sev, errors="coerce").isna() & sev.notna()
    count = int(text_mask.sum())
    if count > 0:
        mapped = (
            sev[text_mask]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(_TEXT_TO_NUMERIC)
        )
        sev.loc[text_mask] = mapped
    # Ensure the entire column is numeric
    df["severity"] = pd.to_numeric(sev, errors="coerce").astype("Int64")
    return count


def _fix_swapped_dates(df: pd.DataFrame) -> int:
    """Where report_date < incident_date, swap the two columns.

    Returns the number of rows corrected.
    """
    df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

    swapped_mask = df["report_date"] < df["incident_date"]
    count = int(swapped_mask.sum())
    if count > 0:
        orig_incident = df.loc[swapped_mask, "incident_date"].copy()
        df.loc[swapped_mask, "incident_date"] = df.loc[swapped_mask, "report_date"]
        df.loc[swapped_mask, "report_date"] = orig_incident
    return count


def _fill_null_diagnosis(df: pd.DataFrame) -> int:
    """Replace missing diagnosis_id values with 'UNKNOWN'.

    Returns the number of rows filled.
    """
    null_mask = df["diagnosis_id"].isna()
    count = int(null_mask.sum())
    if count > 0:
        df.loc[null_mask, "diagnosis_id"] = "UNKNOWN"
    return count


def _cap_outlier_amounts(df: pd.DataFrame, cap: float = _PAID_AMOUNT_CAP) -> int:
    """Cap paid_amount at the given threshold.

    Returns the number of rows capped.
    """
    outlier_mask = df["paid_amount"].abs() > cap
    count = int(outlier_mask.sum())
    if count > 0:
        df.loc[outlier_mask, "paid_amount"] = np.sign(
            df.loc[outlier_mask, "paid_amount"]
        ) * cap
    return count


def _remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop full duplicate rows (keeping the first occurrence).

    Returns the deduplicated DataFrame and the number of rows removed.
    """
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    removed = before - len(df)
    return df, removed


def _enforce_dtypes(df: pd.DataFrame) -> None:
    """Ensure proper dtypes for date and numeric columns (in place)."""
    # Date columns
    for col in ("incident_date", "report_date", "close_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Float columns
    for col in ("paid_amount", "reserve_amount", "incurred_amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # Integer columns (nullable Int64 to tolerate NaN)
    for col in (
        "patient_age",
        "litigation_flag",
        "repeat_event_flag",
        "report_lag_days",
        "accident_year",
        "report_year",
        "development_year",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")


# ── Before/After Comparison ───────────────────────────────────────────────

def _build_comparison_report(
    before: DataQualityReport,
    after: DataQualityReport,
    actions: Dict[str, str],
) -> DataQualityReport:
    """Merge before/after validation into a single professional report.

    The returned report contains the *before* issues annotated with the
    actions taken, plus a before/after comparison table.
    """
    report = DataQualityReport()
    report.total_records = before.total_records
    report.column_null_counts = after.column_null_counts
    report.column_ranges = after.column_ranges

    # Carry over before-issues with action annotations
    before_cats = before.category_counts
    after_cats = after.category_counts

    for issue in before.issues:
        action = actions.get(issue.category, "")
        report.add_issue(
            category=issue.category,
            description=issue.description,
            count=issue.count,
            action_taken=action,
        )

    # Store comparison data for rendering
    report._before_counts = before_cats  # type: ignore[attr-defined]
    report._after_counts = after_cats  # type: ignore[attr-defined]

    return report


def _render_final_report(
    report: DataQualityReport,
    before_cats: Dict[str, int],
    after_cats: Dict[str, int],
    cleaning_time: float,
) -> str:
    """Render the complete data-quality report including before/after table."""
    lines: List[str] = []

    lines.append("# Data Quality Report")
    lines.append(f"Generated: {report.timestamp}")
    lines.append("")

    # Summary
    total_before = sum(before_cats.values())
    total_after = sum(after_cats.values())
    resolved = total_before - total_after

    lines.append("## Summary")
    lines.append(f"- Total records examined: {report.total_records:,}")
    lines.append(f"- Issues found: {total_before:,}")
    lines.append(f"- Issues resolved: {resolved:,}")
    lines.append(f"- Issues remaining: {total_after:,}")
    lines.append(f"- Cleaning time: {cleaning_time:.2f}s")
    lines.append("")

    # Issues Detail
    if report.issues:
        lines.append("## Issues Detail")
        for issue in report.issues:
            lines.append(f"### {issue.category}")
            lines.append(f"- Found: {issue.count:,} records — {issue.description}")
            if issue.action_taken:
                lines.append(f"- Action: {issue.action_taken}")
            after_val = after_cats.get(issue.category, 0)
            lines.append(f"- Remaining: {after_val:,}")
            lines.append("")

    # Before/After Comparison Table
    all_categories = sorted(
        set(list(before_cats.keys()) + list(after_cats.keys()))
    )
    if all_categories:
        lines.append("## Before/After Comparison")
        lines.append("| Metric | Before | After |")
        lines.append("|--------|--------|-------|")
        for cat in all_categories:
            b = before_cats.get(cat, 0)
            a = after_cats.get(cat, 0)
            lines.append(f"| {cat} | {b:,} | {a:,} |")
        lines.append(
            f"| **Total** | **{total_before:,}** | **{total_after:,}** |"
        )
        lines.append("")

    # Column Null Counts (post-cleaning)
    if report.column_null_counts:
        non_zero = {
            k: v for k, v in report.column_null_counts.items() if v > 0
        }
        if non_zero:
            lines.append("## Column Null Counts (After Cleaning)")
            lines.append("| Column | Null Count |")
            lines.append("|--------|------------|")
            for col, cnt in sorted(non_zero.items(), key=lambda x: -x[1]):
                lines.append(f"| {col} | {cnt:,} |")
            lines.append("")

    # Numeric Ranges (post-cleaning)
    if report.column_ranges:
        lines.append("## Numeric Column Ranges (After Cleaning)")
        lines.append("| Column | Min | Max |")
        lines.append("|--------|-----|-----|")
        for col, rng in sorted(report.column_ranges.items()):
            lines.append(
                f"| {col} | {rng['min']:,.2f} | {rng['max']:,.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


# ── Main Pipeline ─────────────────────────────────────────────────────────

def clean_claims(
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    seed: int = RANDOM_SEED,
) -> DataQualityReport:
    """Run the full cleaning pipeline on raw claims data.

    Parameters
    ----------
    input_dir : Path, optional
        Directory containing raw CSVs (default: ``data/raw/``).
    output_dir : Path, optional
        Destination for cleaned CSVs (default: ``data/cleaned/``).
    seed : int
        Random seed for any stochastic steps (currently unused but
        reserved for imputation strategies).

    Returns
    -------
    DataQualityReport
        The quality report covering both before and after states.
    """
    input_dir = Path(input_dir) if input_dir else RAW_DIR
    output_dir = Path(output_dir) if output_dir else CLEANED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    np.random.seed(seed)  # noqa: NPY002 — reproducibility for any future stochastic steps

    # ── Load raw data ─────────────────────────────────────────────────
    raw_files = sorted(input_dir.glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(
            f"No CSV files found in {input_dir}. Run the data generation "
            "step first: python -m src.generation.synthetic_claims"
        )

    dataframes: Dict[str, pd.DataFrame] = {}
    for csv_path in raw_files:
        name = csv_path.stem
        dataframes[name] = pd.read_csv(csv_path)

    if "claims" not in dataframes:
        raise FileNotFoundError(
            f"Expected claims.csv in {input_dir} but it was not found."
        )

    claims = dataframes["claims"]
    print(f"\n{'=' * 60}")
    print("  DATA CLEANING PIPELINE")
    print(f"{'=' * 60}")
    print(f"\n  Loaded {len(claims):,} claims from {input_dir / 'claims.csv'}")

    # ── Validate BEFORE cleaning ──────────────────────────────────────
    before_report = validate_claims(claims)
    before_cats = before_report.category_counts
    print(f"\n  Issues detected before cleaning: {before_report.total_issues:,}")
    for cat, cnt in sorted(before_cats.items()):
        print(f"    {cat:<30s}  {cnt:>6,}")

    # ── Cleaning Steps ────────────────────────────────────────────────
    actions: Dict[str, str] = {}
    print(f"\n{'-' * 60}")
    print("  Applying cleaning steps...")
    print(f"{'-' * 60}")

    # 1. Negative paid_amounts -> absolute value
    fixed = _fix_negative_paid_amounts(claims)
    actions["Negative Paid Amounts"] = "Converted to absolute values"
    print(f"  [1/8] Fixed negative paid_amounts:     {fixed:>6,}")

    # 2. Standardize state codes -> uppercase
    fixed = _standardize_state_codes(claims)
    actions["Mixed-Case State Codes"] = "Converted to uppercase"
    print(f"  [2/8] Standardized state codes:         {fixed:>6,}")

    # 3. Convert text severity -> numeric
    fixed = _convert_text_severity(claims)
    actions["Text Severity Values"] = "Mapped text labels to integers 1-5"
    print(f"  [3/8] Converted text severity values:   {fixed:>6,}")

    # 4. Fix swapped dates
    fixed = _fix_swapped_dates(claims)
    actions["Swapped Dates"] = "Swapped incident_date and report_date back to correct order"
    print(f"  [4/8] Fixed swapped dates:              {fixed:>6,}")

    # 5. Fill null diagnosis codes
    fixed = _fill_null_diagnosis(claims)
    actions["Null Diagnosis Codes"] = "Filled with 'UNKNOWN'"
    print(f"  [5/8] Filled null diagnosis codes:      {fixed:>6,}")

    # 6. Cap outlier paid_amounts at $5M
    fixed = _cap_outlier_amounts(claims)
    actions["Extreme Outliers"] = f"Capped at ${_PAID_AMOUNT_CAP:,.0f}"
    print(f"  [6/8] Capped outlier paid_amounts:      {fixed:>6,}")

    # 7. Remove true duplicates
    claims, removed = _remove_duplicates(claims)
    actions["Duplicate Claim IDs"] = "Removed duplicate rows"
    print(f"  [7/8] Removed duplicate rows:           {removed:>6,}")

    # 8. Ensure proper dtypes
    _enforce_dtypes(claims)
    print(f"  [8/8] Enforced proper column dtypes")

    # Update the reference in case dedup changed the DataFrame
    dataframes["claims"] = claims

    # ── Validate AFTER cleaning ───────────────────────────────────────
    after_report = validate_claims(claims)
    after_cats = after_report.category_counts

    elapsed = time.perf_counter() - t0

    # ── Build combined report ─────────────────────────────────────────
    combined = _build_comparison_report(before_report, after_report, actions)
    report_md = _render_final_report(combined, before_cats, after_cats, elapsed)

    # Override the default generate_report with our custom render
    combined._rendered_report = report_md  # type: ignore[attr-defined]
    _original_generate = combined.generate_report
    combined.generate_report = lambda: report_md  # type: ignore[method-assign]

    # ── Save cleaned CSVs ─────────────────────────────────────────────
    for name, frame in dataframes.items():
        dest = output_dir / f"{name}.csv"
        frame.to_csv(dest, index=False)

    # ── Save report ───────────────────────────────────────────────────
    report_path = REPORTS_DIR / "data_quality_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")

    # ── Print summary ─────────────────────────────────────────────────
    total_before = sum(before_cats.values())
    total_after = sum(after_cats.values())
    resolved = total_before - total_after

    print(f"\n{'-' * 60}")
    print("  Cleaning Complete")
    print(f"{'-' * 60}")
    print(f"  Issues found:    {total_before:>6,}")
    print(f"  Issues resolved: {resolved:>6,}")
    print(f"  Issues remaining:{total_after:>6,}")
    print(f"  Cleaning time:   {elapsed:>8.2f}s")
    print(f"\n  Cleaned data saved to:  {output_dir}")
    print(f"  Quality report saved to: {report_path}")
    print(f"{'=' * 60}\n")

    return combined


# ── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    clean_claims()
