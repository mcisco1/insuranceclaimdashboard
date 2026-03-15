"""Tests for the database layer (src.database)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import inspect, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.models import (
    Base,
    DimDiagnosis,
    DimProvider,
    DimRegion,
    DimRootCause,
    FactClaim,
)
from src.generation.reference_data import DIAGNOSES, ROOT_CAUSES, SPECIALTIES, STATES


# ── Create tables ──────────────────────────────────────────────────────────


class TestCreateTables:
    """All 5 tables are created."""

    def test_all_tables_created(self, db_engine):
        Base.metadata.create_all(db_engine)
        inspector = inspect(db_engine)
        table_names = set(inspector.get_table_names())
        expected = {"dim_provider", "dim_region", "dim_diagnosis", "dim_root_cause", "fact_claim"}
        assert expected.issubset(table_names), (
            f"Missing tables: {expected - table_names}"
        )

    def test_table_count(self, db_engine):
        Base.metadata.create_all(db_engine)
        inspector = inspect(db_engine)
        tables = inspector.get_table_names()
        assert len(tables) >= 5


# ── Load dimensions ───────────────────────────────────────────────────────


class TestLoadDimensions:
    """Dimension tables have correct row counts."""

    def test_load_providers(self, db_engine):
        Base.metadata.create_all(db_engine)

        providers_data = []
        for i, s in enumerate(SPECIALTIES[:5]):
            providers_data.append({
                "provider_id": f"PRV{i+1:04d}",
                "provider_name": f"Provider {i+1}",
                "specialty": s["name"],
                "department": "Emergency Department",
                "risk_tier": s["risk_tier"],
            })

        df = pd.DataFrame(providers_data)
        df.to_sql("dim_provider", db_engine, if_exists="append", index=False)

        with db_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM dim_provider")).scalar()
        assert count == 5

    def test_load_regions(self, db_engine):
        Base.metadata.create_all(db_engine)

        regions_data = []
        for s in STATES[:10]:
            regions_data.append({
                "state_code": s["state_code"],
                "state_name": s["name"],
                "region": s["region"],
                "urban_rural": s["urban_rural"],
            })

        df = pd.DataFrame(regions_data)
        df.to_sql("dim_region", db_engine, if_exists="append", index=False)

        with db_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM dim_region")).scalar()
        assert count == 10

    def test_load_diagnoses(self, db_engine):
        Base.metadata.create_all(db_engine)

        diag_data = []
        for d in DIAGNOSES:
            diag_data.append({
                "diagnosis_code": d["diagnosis_id"],
                "diagnosis_category": d["category"],
                "severity_weight": d["severity_weight"],
            })

        df = pd.DataFrame(diag_data)
        df.to_sql("dim_diagnosis", db_engine, if_exists="append", index=False)

        with db_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM dim_diagnosis")).scalar()
        assert count == len(DIAGNOSES)

    def test_load_root_causes(self, db_engine):
        Base.metadata.create_all(db_engine)

        rc_data = []
        for r in ROOT_CAUSES:
            rc_data.append({
                "root_cause_code": r["root_cause_id"],
                "root_cause_category": r["category"],
                "preventability": r["preventability"],
            })

        df = pd.DataFrame(rc_data)
        df.to_sql("dim_root_cause", db_engine, if_exists="append", index=False)

        with db_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM dim_root_cause")).scalar()
        assert count == len(ROOT_CAUSES)


# ── Load facts ─────────────────────────────────────────────────────────────


class TestLoadFacts:
    """Fact table loads with proper FK relationships."""

    def _setup_dimensions(self, engine):
        """Insert minimal dimension data and return surrogate key maps."""
        Base.metadata.create_all(engine)

        # Provider
        pd.DataFrame([{
            "provider_id": "PRV0001",
            "provider_name": "Test Provider",
            "specialty": "Cardiology",
            "department": "Emergency Department",
            "risk_tier": "medium",
        }]).to_sql("dim_provider", engine, if_exists="append", index=False)

        # Region
        pd.DataFrame([{
            "state_code": "NY",
            "state_name": "New York",
            "region": "Northeast",
            "urban_rural": "urban",
        }]).to_sql("dim_region", engine, if_exists="append", index=False)

        # Diagnosis
        pd.DataFrame([{
            "diagnosis_code": "DX01",
            "diagnosis_category": "Surgical Site Infection",
            "severity_weight": 1.0,
        }]).to_sql("dim_diagnosis", engine, if_exists="append", index=False)

        # Root cause
        pd.DataFrame([{
            "root_cause_code": "RC01",
            "root_cause_category": "Communication Failure",
            "preventability": "preventable",
        }]).to_sql("dim_root_cause", engine, if_exists="append", index=False)

        # Retrieve surrogate keys
        with engine.connect() as conn:
            pk = conn.execute(text("SELECT provider_key FROM dim_provider")).scalar()
            rk = conn.execute(text("SELECT region_key FROM dim_region")).scalar()
            dk = conn.execute(text("SELECT diagnosis_key FROM dim_diagnosis")).scalar()
            rck = conn.execute(text("SELECT root_cause_key FROM dim_root_cause")).scalar()

        return pk, rk, dk, rck

    def test_fact_row_inserted(self, db_engine):
        pk, rk, dk, rck = self._setup_dimensions(db_engine)

        fact_data = pd.DataFrame([{
            "claim_id": "CLM000001",
            "incident_date": "2023-06-15",
            "report_date": "2023-06-20",
            "close_date": "2023-09-15",
            "provider_key": pk,
            "region_key": rk,
            "diagnosis_key": dk,
            "root_cause_key": rck,
            "claim_type": "medical_malpractice",
            "procedure_category": "Surgical - Major",
            "severity_level": 3,
            "status": "closed",
            "paid_amount": 50000.0,
            "incurred_amount": 80000.0,
            "reserved_amount": 30000.0,
            "days_to_close": 87,
            "days_to_report": 5,
            "patient_age_band": "35-49",
            "patient_risk_segment": "medium",
            "repeat_event_flag": 0,
            "litigation_flag": 0,
            "accident_year": 2023,
            "development_year": 0,
        }])
        fact_data.to_sql("fact_claim", db_engine, if_exists="append", index=False)

        with db_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM fact_claim")).scalar()
        assert count == 1

    def test_fk_relationship(self, db_engine):
        pk, rk, dk, rck = self._setup_dimensions(db_engine)

        fact_data = pd.DataFrame([{
            "claim_id": "CLM000002",
            "incident_date": "2023-07-01",
            "report_date": "2023-07-05",
            "close_date": None,
            "provider_key": pk,
            "region_key": rk,
            "diagnosis_key": dk,
            "root_cause_key": rck,
            "claim_type": "general_liability",
            "procedure_category": "Diagnostic Imaging",
            "severity_level": 2,
            "status": "open",
            "paid_amount": 10000.0,
            "incurred_amount": 20000.0,
            "reserved_amount": 10000.0,
            "days_to_close": None,
            "days_to_report": 4,
            "patient_age_band": "50-64",
            "patient_risk_segment": "high",
            "repeat_event_flag": 0,
            "litigation_flag": 1,
            "accident_year": 2023,
            "development_year": 0,
        }])
        fact_data.to_sql("fact_claim", db_engine, if_exists="append", index=False)

        with db_engine.connect() as conn:
            row = conn.execute(text(
                "SELECT f.claim_id, p.specialty, r.state_code "
                "FROM fact_claim f "
                "JOIN dim_provider p ON f.provider_key = p.provider_key "
                "JOIN dim_region r ON f.region_key = r.region_key "
                "WHERE f.claim_id = 'CLM000002'"
            )).fetchone()

        assert row is not None
        assert row[0] == "CLM000002"
        assert row[1] == "Cardiology"
        assert row[2] == "NY"


# ── Query claims ──────────────────────────────────────────────────────────


class TestQueryClaims:
    """Basic query returns expected results."""

    def test_query_by_status(self, db_engine):
        Base.metadata.create_all(db_engine)

        # Insert minimal dimension data
        pd.DataFrame([{
            "provider_id": "PRV0001",
            "provider_name": "Test",
            "specialty": "Cardiology",
            "department": "ED",
            "risk_tier": "medium",
        }]).to_sql("dim_provider", db_engine, if_exists="append", index=False)

        pd.DataFrame([{
            "state_code": "CA",
            "state_name": "California",
            "region": "West",
            "urban_rural": "urban",
        }]).to_sql("dim_region", db_engine, if_exists="append", index=False)

        pd.DataFrame([{
            "diagnosis_code": "DX01",
            "diagnosis_category": "Test",
            "severity_weight": 1.0,
        }]).to_sql("dim_diagnosis", db_engine, if_exists="append", index=False)

        pd.DataFrame([{
            "root_cause_code": "RC01",
            "root_cause_category": "Test",
            "preventability": "preventable",
        }]).to_sql("dim_root_cause", db_engine, if_exists="append", index=False)

        with db_engine.connect() as conn:
            pk = conn.execute(text("SELECT provider_key FROM dim_provider")).scalar()
            rk = conn.execute(text("SELECT region_key FROM dim_region")).scalar()
            dk = conn.execute(text("SELECT diagnosis_key FROM dim_diagnosis")).scalar()
            rck = conn.execute(text("SELECT root_cause_key FROM dim_root_cause")).scalar()

        claims = []
        for i, status in enumerate(["closed", "closed", "open"]):
            claims.append({
                "claim_id": f"CLM{i:06d}",
                "incident_date": "2023-01-01",
                "report_date": "2023-01-05",
                "close_date": "2023-03-01" if status == "closed" else None,
                "provider_key": pk,
                "region_key": rk,
                "diagnosis_key": dk,
                "root_cause_key": rck,
                "claim_type": "medical_malpractice",
                "procedure_category": "Surgical - Major",
                "severity_level": 3,
                "status": status,
                "paid_amount": 50000.0,
                "incurred_amount": 80000.0,
                "reserved_amount": 30000.0,
                "days_to_close": 55 if status == "closed" else None,
                "days_to_report": 4,
                "patient_age_band": "35-49",
                "patient_risk_segment": "medium",
                "repeat_event_flag": 0,
                "litigation_flag": 0,
                "accident_year": 2023,
                "development_year": 0,
            })

        pd.DataFrame(claims).to_sql("fact_claim", db_engine, if_exists="append", index=False)

        with db_engine.connect() as conn:
            closed = conn.execute(
                text("SELECT COUNT(*) FROM fact_claim WHERE status = 'closed'")
            ).scalar()
            open_count = conn.execute(
                text("SELECT COUNT(*) FROM fact_claim WHERE status = 'open'")
            ).scalar()

        assert closed == 2
        assert open_count == 1
