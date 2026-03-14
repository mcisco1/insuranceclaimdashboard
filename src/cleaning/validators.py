"""Data quality validation and report generation for insurance claims.

Provides a ``DataQualityReport`` collector and a ``validate_claims`` function
that runs a comprehensive suite of checks against a claims DataFrame.

Usage
-----
    from src.cleaning.validators import validate_claims
    report = validate_claims(df)
    print(report.generate_report())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import REPORTS_DIR


# ── Issue Record ──────────────────────────────────────────────────────────

@dataclass
class _Issue:
    """Single data-quality issue discovered during validation."""

    category: str
    description: str
    count: int
    action_taken: str


# ── DataQualityReport ─────────────────────────────────────────────────────

class DataQualityReport:
    """Accumulates data-quality issues found during validation.

    Attributes
    ----------
    issues : list[_Issue]
        All issues recorded via :meth:`add_issue`.
    total_records : int
        Number of rows examined (set externally after validation).
    column_null_counts : dict[str, int]
        Per-column null counts (set externally after validation).
    column_ranges : dict[str, dict]
        Per-column min/max for numeric columns.
    timestamp : str
        ISO-format timestamp when the report object was created.
    """

    def __init__(self) -> None:
        self.issues: List[_Issue] = []
        self.total_records: int = 0
        self.column_null_counts: Dict[str, int] = {}
        self.column_ranges: Dict[str, dict] = {}
        self.timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._category_counts: Dict[str, int] = {}

    # ── Recording helpers ─────────────────────────────────────────────

    def add_issue(
        self,
        category: str,
        description: str,
        count: int,
        action_taken: str = "",
    ) -> None:
        """Register one data-quality issue.

        Parameters
        ----------
        category : str
            Short label grouping related issues (e.g. "Negative Paid Amounts").
        description : str
            Human-readable explanation of the problem.
        count : int
            Number of affected rows.
        action_taken : str, optional
            Remediation applied (or planned). Empty string when validating
            *before* cleaning.
        """
        self.issues.append(
            _Issue(
                category=category,
                description=description,
                count=count,
                action_taken=action_taken,
            )
        )
        self._category_counts[category] = (
            self._category_counts.get(category, 0) + count
        )

    @property
    def total_issues(self) -> int:
        """Total number of affected rows across all categories."""
        return sum(issue.count for issue in self.issues)

    @property
    def category_counts(self) -> Dict[str, int]:
        """Issue counts keyed by category label."""
        return dict(self._category_counts)

    # ── Report Generation ─────────────────────────────────────────────

    def generate_report(self) -> str:
        """Return a professional Markdown-formatted quality report."""
        lines: List[str] = []

        lines.append("# Data Quality Report")
        lines.append(f"Generated: {self.timestamp}")
        lines.append("")

        # Summary section
        lines.append("## Summary")
        lines.append(f"- Total records examined: {self.total_records:,}")
        lines.append(f"- Issues found: {self.total_issues:,}")
        issues_resolved = sum(
            i.count for i in self.issues if i.action_taken
        )
        lines.append(f"- Issues resolved: {issues_resolved:,}")
        lines.append("")

        # Per-issue detail
        if self.issues:
            lines.append("## Issues Detail")
            for issue in self.issues:
                lines.append(f"### {issue.category}")
                lines.append(f"- Found: {issue.count:,} records — {issue.description}")
                if issue.action_taken:
                    lines.append(f"- Action: {issue.action_taken}")
                lines.append("")

        # Column null counts
        if self.column_null_counts:
            lines.append("## Column Null Counts")
            lines.append("| Column | Null Count |")
            lines.append("|--------|------------|")
            for col, cnt in sorted(
                self.column_null_counts.items(), key=lambda x: -x[1]
            ):
                if cnt > 0:
                    lines.append(f"| {col} | {cnt:,} |")
            lines.append("")

        # Numeric ranges
        if self.column_ranges:
            lines.append("## Numeric Column Ranges")
            lines.append("| Column | Min | Max |")
            lines.append("|--------|-----|-----|")
            for col, rng in sorted(self.column_ranges.items()):
                lines.append(
                    f"| {col} | {rng['min']:,.2f} | {rng['max']:,.2f} |"
                )
            lines.append("")

        return "\n".join(lines)

    def save_report(self, path: Optional[Path] = None) -> Path:
        """Write the Markdown report to disk.

        Parameters
        ----------
        path : Path, optional
            Destination file.  Defaults to
            ``reports/data_quality_report.md``.

        Returns
        -------
        Path
            The file that was written.
        """
        if path is None:
            path = REPORTS_DIR / "data_quality_report.md"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generate_report(), encoding="utf-8")
        return path


# ── Validation Suite ──────────────────────────────────────────────────────

def validate_claims(df: pd.DataFrame) -> DataQualityReport:
    """Run all data-quality checks on a claims DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (or cleaned) claims data.  Expected columns include
        ``paid_amount``, ``state_code``, ``severity``, ``incident_date``,
        ``report_date``, ``diagnosis_id``, ``claim_id``, among others.

    Returns
    -------
    DataQualityReport
        Populated report object ready for rendering or comparison.
    """
    report = DataQualityReport()
    report.total_records = len(df)

    # ── 1. Negative paid amounts ──────────────────────────────────────
    if "paid_amount" in df.columns:
        neg_count = int((df["paid_amount"] < 0).sum())
        if neg_count > 0:
            report.add_issue(
                category="Negative Paid Amounts",
                description="records with negative paid_amount values",
                count=neg_count,
            )

    # ── 2. Mixed-case state codes ─────────────────────────────────────
    if "state_code" in df.columns:
        non_upper = df["state_code"].dropna()
        mixed_count = int((non_upper != non_upper.str.upper()).sum())
        if mixed_count > 0:
            report.add_issue(
                category="Mixed-Case State Codes",
                description="state codes not in uppercase format",
                count=mixed_count,
            )

    # ── 3. Text severity values ───────────────────────────────────────
    if "severity" in df.columns:
        text_mask = pd.to_numeric(df["severity"], errors="coerce").isna() & df["severity"].notna()
        text_count = int(text_mask.sum())
        if text_count > 0:
            report.add_issue(
                category="Text Severity Values",
                description="severity values stored as text instead of numeric",
                count=text_count,
            )

    # ── 4. Swapped dates (report_date before incident_date) ──────────
    if "report_date" in df.columns and "incident_date" in df.columns:
        rd = pd.to_datetime(df["report_date"], errors="coerce")
        id_ = pd.to_datetime(df["incident_date"], errors="coerce")
        swapped_count = int((rd < id_).sum())
        if swapped_count > 0:
            report.add_issue(
                category="Swapped Dates",
                description="report_date earlier than incident_date",
                count=swapped_count,
            )

    # ── 5. Null / missing diagnosis codes ─────────────────────────────
    if "diagnosis_id" in df.columns:
        null_diag = int(df["diagnosis_id"].isna().sum())
        if null_diag > 0:
            report.add_issue(
                category="Null Diagnosis Codes",
                description="missing diagnosis_id values",
                count=null_diag,
            )

    # ── 6. Extreme outliers (paid_amount > $5M) ──────────────────────
    if "paid_amount" in df.columns:
        outlier_count = int((df["paid_amount"].abs() > 5_000_000).sum())
        if outlier_count > 0:
            report.add_issue(
                category="Extreme Outliers",
                description="paid_amount exceeding $5,000,000",
                count=outlier_count,
            )

    # ── 7. Duplicate claim_ids ────────────────────────────────────────
    if "claim_id" in df.columns:
        dup_count = int(df["claim_id"].duplicated().sum())
        if dup_count > 0:
            report.add_issue(
                category="Duplicate Claim IDs",
                description="duplicate claim_id entries",
                count=dup_count,
            )

    # ── 8. Future dates ───────────────────────────────────────────────
    today = pd.Timestamp.now().normalize()
    for date_col in ("incident_date", "report_date", "close_date"):
        if date_col in df.columns:
            dt = pd.to_datetime(df[date_col], errors="coerce")
            future_count = int((dt > today).sum())
            if future_count > 0:
                report.add_issue(
                    category="Future Dates",
                    description=f"{date_col} values set in the future",
                    count=future_count,
                )

    # ── Summary statistics ────────────────────────────────────────────
    # Null counts per column
    report.column_null_counts = df.isnull().sum().to_dict()

    # Numeric range checks
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0:
            report.column_ranges[col] = {
                "min": float(series.min()),
                "max": float(series.max()),
            }

    return report
