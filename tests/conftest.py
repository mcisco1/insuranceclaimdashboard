"""Shared pytest fixtures for the insurance-claims-dashboard test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.reference_data import (
    CLAIM_TYPES,
    DEPARTMENTS,
    DIAGNOSES,
    ROOT_CAUSES,
    SPECIALTIES,
    STATES,
)


# ── Constants for fixture generation ────────────────────────────────────────

_SEED = 42
_N_ROWS = 100

_STATUSES = ["closed", "open", "reopened"]
_STATUS_WEIGHTS = [0.75, 0.18, 0.07]

_SEVERITIES = [1, 2, 3, 4, 5]

_STATE_CODES = [s["state_code"] for s in STATES]
_REGIONS = {s["state_code"]: s["region"] for s in STATES}
_SPECIALTY_NAMES = [s["name"] for s in SPECIALTIES]
_DEPARTMENT_NAMES = DEPARTMENTS
_DIAGNOSIS_IDS = [d["diagnosis_id"] for d in DIAGNOSES]
_DIAGNOSIS_CATEGORIES = {d["diagnosis_id"]: d["category"] for d in DIAGNOSES}
_ROOT_CAUSE_IDS = [r["root_cause_id"] for r in ROOT_CAUSES]
_ROOT_CAUSE_CATEGORIES = {r["root_cause_id"]: r["category"] for r in ROOT_CAUSES}
_PROCEDURE_CATEGORIES = [
    "Surgical - Major",
    "Surgical - Minor",
    "Diagnostic Imaging",
    "Laboratory/Pathology",
    "Medication Administration",
    "Therapeutic Procedure",
    "Obstetric Delivery",
    "Anesthesia",
    "Emergency Intervention",
    "Rehabilitation/Physical Therapy",
]


def _age_band(age: int) -> str:
    if age < 18:
        return "0-17"
    if age < 35:
        return "18-34"
    if age < 50:
        return "35-49"
    if age < 65:
        return "50-64"
    return "65+"


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_claims_df() -> pd.DataFrame:
    """Small clean DataFrame (~100 rows) with realistic claims data.

    Includes all columns from the fact table plus joined dimension fields
    (specialty, department, region, state_code, diagnosis_category,
    root_cause_category, etc.).  Uses a fixed seed for reproducibility.
    """
    rng = np.random.default_rng(_SEED)
    n = _N_ROWS

    # Dates
    start = pd.Timestamp("2022-01-01")
    offsets = rng.integers(0, 365 * 3, size=n)
    incident_dates = pd.to_datetime(
        [start + pd.Timedelta(days=int(d)) for d in offsets]
    )

    report_lag = rng.integers(1, 30, size=n)
    report_dates = pd.to_datetime(
        [inc + pd.Timedelta(days=int(lag)) for inc, lag in zip(incident_dates, report_lag)]
    )

    severities = rng.choice(_SEVERITIES, size=n)
    statuses = rng.choice(_STATUSES, size=n, p=_STATUS_WEIGHTS)

    close_lag = rng.integers(30, 400, size=n)
    close_dates = pd.to_datetime(
        [
            rep + pd.Timedelta(days=int(cl)) if st == "closed" else pd.NaT
            for rep, cl, st in zip(report_dates, close_lag, statuses)
        ]
    )

    days_to_close = np.where(
        statuses == "closed",
        close_lag,
        np.nan,
    )

    state_codes = rng.choice(_STATE_CODES, size=n)
    regions = [_REGIONS[sc] for sc in state_codes]

    specialty_names = rng.choice(_SPECIALTY_NAMES, size=n)
    departments = rng.choice(_DEPARTMENT_NAMES, size=n)

    diagnosis_ids = rng.choice(_DIAGNOSIS_IDS, size=n)
    diagnosis_categories = [_DIAGNOSIS_CATEGORIES[did] for did in diagnosis_ids]

    root_cause_ids = rng.choice(_ROOT_CAUSE_IDS, size=n)
    root_cause_categories = [_ROOT_CAUSE_CATEGORIES[rid] for rid in root_cause_ids]

    claim_types = rng.choice(CLAIM_TYPES, size=n)
    procedure_categories = rng.choice(_PROCEDURE_CATEGORIES, size=n)

    paid = np.round(rng.lognormal(mean=9.0, sigma=1.0, size=n).clip(500, 5_000_000), 2)
    reserve = np.round(paid * (1 + rng.exponential(0.3, size=n)), 2)
    incurred = np.round(paid + reserve, 2)

    patient_ages = rng.integers(0, 96, size=n)
    age_bands = [_age_band(int(a)) for a in patient_ages]

    litigation_flags = rng.choice([0, 1], size=n, p=[0.75, 0.25]).astype(int)
    repeat_flags = rng.choice([0, 1], size=n, p=[0.92, 0.08]).astype(int)

    accident_years = pd.Series(incident_dates).dt.year.values
    report_years = pd.Series(report_dates).dt.year.values
    development_years = report_years - accident_years

    provider_ids = [f"PRV{rng.integers(1, 50):04d}" for _ in range(n)]

    df = pd.DataFrame(
        {
            "claim_id": [f"CLM{i + 1:06d}" for i in range(n)],
            "incident_date": incident_dates,
            "report_date": report_dates,
            "close_date": close_dates,
            "accident_year": accident_years,
            "report_year": report_years,
            "development_year": development_years,
            "provider_id": provider_ids,
            "state_code": state_codes,
            "region": regions,
            "specialty": specialty_names,
            "department": departments,
            "diagnosis_id": diagnosis_ids,
            "diagnosis_category": diagnosis_categories,
            "root_cause_id": root_cause_ids,
            "root_cause_category": root_cause_categories,
            "claim_type": claim_types,
            "procedure_category": procedure_categories,
            "severity": severities,
            "severity_level": severities,
            "status": statuses,
            "paid_amount": paid,
            "reserve_amount": reserve,
            "incurred_amount": incurred,
            "patient_age": patient_ages,
            "patient_age_band": age_bands,
            "patient_risk_segment": rng.choice(
                ["low", "medium", "high", "critical"], size=n
            ),
            "litigation_flag": litigation_flags,
            "repeat_event_flag": repeat_flags,
            "report_lag_days": report_lag,
            "days_to_report": report_lag,
            "days_to_close": days_to_close.astype("float64"),
        }
    )

    return df


@pytest.fixture()
def raw_claims_df(sample_claims_df: pd.DataFrame) -> pd.DataFrame:
    """Version of sample_claims_df with intentional data-quality issues.

    Issues injected:
    - Negative paid_amount values
    - Mixed-case state codes
    - Text severity values
    - Swapped dates (report_date < incident_date)
    - Null diagnosis_id
    - Extreme outliers (> $5M)
    """
    df = sample_claims_df.copy()

    # 1. Negative paid amounts (rows 0-4)
    df.loc[0:4, "paid_amount"] = -df.loc[0:4, "paid_amount"].abs()

    # 2. Mixed-case state codes (rows 5-9)
    df.loc[5:9, "state_code"] = df.loc[5:9, "state_code"].apply(
        lambda s: s[0].upper() + s[1:].lower()
    )

    # 3. Text severity values (rows 10-14)
    text_map = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    df["severity"] = df["severity"].astype(object)
    for idx in range(10, 15):
        df.at[idx, "severity"] = text_map.get(int(df.at[idx, "severity"]), "one")

    # 4. Swapped dates: report_date < incident_date (rows 15-19)
    for idx in range(15, 20):
        df.at[idx, "report_date"] = df.at[idx, "incident_date"] - pd.Timedelta(days=5)

    # 5. Null diagnosis_id (rows 20-24)
    df.loc[20:24, "diagnosis_id"] = np.nan

    # 6. Extreme outliers > $5M (rows 25-27)
    df.loc[25:27, "paid_amount"] = 12_000_000.0

    return df


@pytest.fixture()
def tmp_dir(tmp_path):
    """Temporary directory for file-based tests."""
    return tmp_path


@pytest.fixture()
def db_engine():
    """In-memory SQLite engine for database tests."""
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()
