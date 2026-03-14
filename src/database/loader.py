"""ETL loader -- populates the SQLite warehouse from cleaned CSVs.

Reads the five cleaned CSV files produced by the cleaning pipeline,
maps them into the star-schema ORM models, and bulk-loads them into
the database.  Dimension tables are loaded first so that foreign-key
lookups succeed when the fact table is inserted.

Usage
-----
    python -m src.database.loader          # from project root
    # or
    from src.database.loader import load_database
    load_database()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sqlalchemy import inspect, text

from config import CLEANED_DIR, DB_URL, WAREHOUSE_DIR
from src.database.connection import get_engine
from src.database.models import (
    Base,
    DimDiagnosis,
    DimProvider,
    DimRegion,
    DimRootCause,
    FactClaim,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV with sensible defaults."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def _print_table_counts(engine) -> None:
    """Print row counts for every table in the database."""
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    print(f"\n  {'Table':<20s}  {'Rows':>10s}")
    print(f"  {'-' * 20}  {'-' * 10}")
    with engine.connect() as conn:
        for table in sorted(table_names):
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  {table:<20s}  {count:>10,}")


# ── Dimension Loaders ────────────────────────────────────────────────────

def _load_dim_provider(engine, cleaned_dir: Path) -> Dict[str, int]:
    """Load dim_provider and return a mapping provider_id -> provider_key."""
    df = _read_csv(cleaned_dir / "providers.csv")

    records = []
    for _, row in df.iterrows():
        records.append({
            "provider_id": row["provider_id"],
            "provider_name": f"Provider {row['provider_id']}",
            "specialty": row["specialty_name"],
            "department": row["department"],
            "risk_tier": row["risk_tier"],
        })

    insert_df = pd.DataFrame(records)
    insert_df.to_sql("dim_provider", engine, if_exists="append", index=False)

    # Build lookup: provider_id -> provider_key
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT provider_key, provider_id FROM dim_provider")
        ).fetchall()
    return {r[1]: r[0] for r in rows}


def _load_dim_region(engine, cleaned_dir: Path) -> Dict[str, int]:
    """Load dim_region and return a mapping state_code -> region_key."""
    df = _read_csv(cleaned_dir / "regions.csv")

    records = []
    for _, row in df.iterrows():
        records.append({
            "state_code": row["state_code"],
            "state_name": row["state_name"],
            "region": row["region"],
            "urban_rural": row["urban_rural"],
        })

    insert_df = pd.DataFrame(records)
    insert_df.to_sql("dim_region", engine, if_exists="append", index=False)

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT region_key, state_code FROM dim_region")
        ).fetchall()
    return {r[1]: r[0] for r in rows}


def _load_dim_diagnosis(engine, cleaned_dir: Path) -> Dict[str, int]:
    """Load dim_diagnosis and return a mapping diagnosis_id -> diagnosis_key."""
    df = _read_csv(cleaned_dir / "diagnoses.csv")

    records = []
    for _, row in df.iterrows():
        records.append({
            "diagnosis_code": row["diagnosis_id"],
            "diagnosis_category": row["category"],
            "severity_weight": float(row["severity_weight"]),
        })

    # Add the UNKNOWN sentinel used for null diagnoses
    if not any(r["diagnosis_code"] == "UNKNOWN" for r in records):
        records.append({
            "diagnosis_code": "UNKNOWN",
            "diagnosis_category": "Unknown",
            "severity_weight": 1.0,
        })

    insert_df = pd.DataFrame(records)
    insert_df.to_sql("dim_diagnosis", engine, if_exists="append", index=False)

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT diagnosis_key, diagnosis_code FROM dim_diagnosis")
        ).fetchall()
    return {r[1]: r[0] for r in rows}


def _load_dim_root_cause(engine, cleaned_dir: Path) -> Dict[str, int]:
    """Load dim_root_cause and return a mapping root_cause_id -> root_cause_key."""
    df = _read_csv(cleaned_dir / "root_causes.csv")

    records = []
    for _, row in df.iterrows():
        records.append({
            "root_cause_code": row["root_cause_id"],
            "root_cause_category": row["category"],
            "preventability": row["preventability"],
        })

    insert_df = pd.DataFrame(records)
    insert_df.to_sql("dim_root_cause", engine, if_exists="append", index=False)

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT root_cause_key, root_cause_code FROM dim_root_cause")
        ).fetchall()
    return {r[1]: r[0] for r in rows}


# ── Fact Loader ──────────────────────────────────────────────────────────

def _load_fact_claims(
    engine,
    cleaned_dir: Path,
    provider_map: Dict[str, int],
    region_map: Dict[str, int],
    diagnosis_map: Dict[str, int],
    root_cause_map: Dict[str, int],
) -> int:
    """Load fact_claim using FK lookups from the dimension maps.

    Returns the number of rows loaded.
    """
    df = pd.read_csv(cleaned_dir / "claims.csv")

    # Map natural keys to surrogate keys
    df["provider_key"] = df["provider_id"].map(provider_map)
    df["region_key"] = df["state_code"].map(region_map)
    df["diagnosis_key"] = df["diagnosis_id"].map(diagnosis_map)
    df["root_cause_key"] = df["root_cause_id"].map(root_cause_map)

    # Check for unmapped keys
    for col, name in [
        ("provider_key", "provider"),
        ("region_key", "region"),
        ("diagnosis_key", "diagnosis"),
        ("root_cause_key", "root_cause"),
    ]:
        unmapped = df[col].isna().sum()
        if unmapped > 0:
            print(f"  WARNING: {unmapped:,} claims have unmapped {name} keys")

    # Drop rows that failed FK mapping (should be zero in clean data)
    fk_cols = ["provider_key", "region_key", "diagnosis_key", "root_cause_key"]
    before = len(df)
    df = df.dropna(subset=fk_cols)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  WARNING: Dropped {dropped:,} rows with unmapped foreign keys")

    # Cast FK columns to int
    for col in fk_cols:
        df[col] = df[col].astype(int)

    # Compute days_to_close and days_to_report if not already present
    df["incident_date"] = pd.to_datetime(df["incident_date"])
    df["report_date"] = pd.to_datetime(df["report_date"])
    df["close_date"] = pd.to_datetime(df["close_date"], errors="coerce")

    if "report_lag_days" in df.columns:
        df["days_to_report"] = df["report_lag_days"].fillna(0).astype(int)
    else:
        df["days_to_report"] = (df["report_date"] - df["incident_date"]).dt.days

    if "close_date" in df.columns:
        mask = df["close_date"].notna()
        df.loc[mask, "days_to_close"] = (
            df.loc[mask, "close_date"] - df.loc[mask, "incident_date"]
        ).dt.days
    else:
        df["days_to_close"] = None

    # Rename CSV columns to match ORM model
    rename_map = {
        "severity": "severity_level",
        "reserve_amount": "reserved_amount",
    }
    df = df.rename(columns=rename_map)

    # Select only the columns that exist in the fact table
    fact_columns = [
        "claim_id",
        "incident_date",
        "report_date",
        "close_date",
        "provider_key",
        "region_key",
        "diagnosis_key",
        "root_cause_key",
        "claim_type",
        "procedure_category",
        "severity_level",
        "status",
        "paid_amount",
        "incurred_amount",
        "reserved_amount",
        "days_to_close",
        "days_to_report",
        "patient_age_band",
        "patient_risk_segment",
        "repeat_event_flag",
        "litigation_flag",
        "accident_year",
        "development_year",
    ]

    fact_df = df[fact_columns].copy()

    # Ensure boolean flags are integers (SQLite stores as 0/1)
    for flag in ("repeat_event_flag", "litigation_flag"):
        fact_df[flag] = pd.to_numeric(fact_df[flag], errors="coerce").fillna(0).astype(int)

    # Format dates as strings for SQLite storage
    for date_col in ("incident_date", "report_date", "close_date"):
        fact_df[date_col] = pd.to_datetime(fact_df[date_col], errors="coerce")
        fact_df[date_col] = fact_df[date_col].dt.strftime("%Y-%m-%d")
        fact_df[date_col] = fact_df[date_col].replace("NaT", None)

    # Bulk insert via pandas
    fact_df.to_sql(
        "fact_claim",
        engine,
        if_exists="append",
        index=False,
        chunksize=5000,
    )

    return len(fact_df)


# ── Main Entry Point ────────────────────────────────────────────────────

def load_database(
    cleaned_dir: Optional[str] = None,
    db_url: Optional[str] = None,
) -> None:
    """Build and populate the claims data warehouse.

    Parameters
    ----------
    cleaned_dir : str or None
        Path to the directory containing cleaned CSVs.  Defaults to
        ``config.CLEANED_DIR``.
    db_url : str or None
        SQLAlchemy database URL.  Defaults to ``config.DB_URL``.
    """
    cleaned_path = Path(cleaned_dir) if cleaned_dir else CLEANED_DIR
    url = db_url or DB_URL

    # Validate that cleaned data exists
    required_files = [
        "claims.csv",
        "providers.csv",
        "regions.csv",
        "diagnoses.csv",
        "root_causes.csv",
    ]
    missing = [f for f in required_files if not (cleaned_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing cleaned CSV files in {cleaned_path}: {missing}. "
            "Run the cleaning pipeline first: python -m src.cleaning.cleaner"
        )

    # Ensure warehouse directory exists
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  DATABASE LOADER")
    print(f"{'=' * 60}")
    print(f"\n  Source:  {cleaned_path}")
    print(f"  Target:  {url}")

    t0 = time.perf_counter()

    engine = get_engine(url)

    # Drop existing tables and recreate (idempotent full reload)
    print(f"\n  Creating schema...")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    # Load dimensions first, capturing surrogate-key mappings
    print(f"\n  Loading dimension tables...")
    provider_map = _load_dim_provider(engine, cleaned_path)
    print(f"    dim_provider:    {len(provider_map):>6,} rows")

    region_map = _load_dim_region(engine, cleaned_path)
    print(f"    dim_region:      {len(region_map):>6,} rows")

    diagnosis_map = _load_dim_diagnosis(engine, cleaned_path)
    print(f"    dim_diagnosis:   {len(diagnosis_map):>6,} rows")

    root_cause_map = _load_dim_root_cause(engine, cleaned_path)
    print(f"    dim_root_cause:  {len(root_cause_map):>6,} rows")

    # Load fact table
    print(f"\n  Loading fact table...")
    n_claims = _load_fact_claims(
        engine, cleaned_path,
        provider_map, region_map, diagnosis_map, root_cause_map,
    )
    print(f"    fact_claim:      {n_claims:>6,} rows")

    elapsed = time.perf_counter() - t0

    # Summary
    _print_table_counts(engine)

    print(f"\n  Load time: {elapsed:.2f}s")
    print(f"{'=' * 60}\n")


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    load_database()
