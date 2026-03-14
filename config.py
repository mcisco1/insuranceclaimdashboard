"""Central configuration for paths, seeds, and constants."""

import os
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
EXPORTS_DIR = DATA_DIR / "exports"
REPORTS_DIR = BASE_DIR / "reports"
SQL_DIR = BASE_DIR / "sql"

DB_PATH = WAREHOUSE_DIR / "claims.db"
DB_URL = f"sqlite:///{DB_PATH}"

# ── Reproducibility ────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Data Generation ────────────────────────────────────────────────────
N_CLAIMS = 20_000
DATE_START = "2022-01-01"
DATE_END = "2025-12-31"
N_PROVIDERS = 250

# ── Loss Distribution ──────────────────────────────────────────────────
LOSS_MU = 9.5
LOSS_SIGMA = 1.2
LOSS_MIN = 500
LOSS_MAX = 5_000_000

# ── Status Distribution ────────────────────────────────────────────────
STATUS_WEIGHTS = {"closed": 0.75, "open": 0.18, "reopened": 0.07}

# ── Data Quality (intentional issues) ──────────────────────────────────
NEGATIVE_PAID_RATE = 0.02
MIXED_CASE_STATE_RATE = 0.03
TEXT_SEVERITY_RATE = 0.015
SWAPPED_DATE_RATE = 0.02
NULL_DIAGNOSIS_RATE = 0.04
OUTLIER_COUNT = 8

# ── SLA Thresholds (days) ──────────────────────────────────────────────
SLA_THRESHOLDS = [90, 180, 365]

# ── ML ──────────────────────────────────────────────────────────────────
TEST_SIZE = 0.2
ANOMALY_CONTAMINATION = 0.05
FORECAST_PERIODS = 12

# ── Dashboard ───────────────────────────────────────────────────────────
DASH_HOST = "127.0.0.1"
DASH_PORT = 8050
DASH_DEBUG = True

# Ensure directories exist
for d in [RAW_DIR, CLEANED_DIR, WAREHOUSE_DIR, EXPORTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
