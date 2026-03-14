-- ============================================================================
-- Insurance Claims Data Warehouse -- Star Schema DDL
-- ============================================================================
-- Database: SQLite
-- Description: Kimball-style star schema with one fact table (fact_claim)
--              and four dimension tables (dim_provider, dim_region,
--              dim_diagnosis, dim_root_cause).
--
-- This file is provided for documentation and showcase purposes.
-- The authoritative schema is defined in src/database/models.py (SQLAlchemy
-- ORM) and is created programmatically by the loader.
-- ============================================================================

PRAGMA foreign_keys = ON;

-- ────────────────────────────────────────────────────────────────────────────
-- Dimension: Provider / Specialty
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_provider (
    provider_key   INTEGER      PRIMARY KEY AUTOINCREMENT,
    provider_id    VARCHAR(20)  NOT NULL UNIQUE,       -- natural key (e.g. PRV0001)
    provider_name  VARCHAR(100),                        -- display name
    specialty      VARCHAR(80)  NOT NULL,               -- clinical specialty
    department     VARCHAR(80)  NOT NULL,               -- hospital department
    risk_tier      VARCHAR(20)  NOT NULL                -- low / medium / high
);

CREATE INDEX IF NOT EXISTS ix_dim_provider_id ON dim_provider (provider_id);

-- ────────────────────────────────────────────────────────────────────────────
-- Dimension: Region / Geography
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_region (
    region_key   INTEGER      PRIMARY KEY AUTOINCREMENT,
    state_code   VARCHAR(2)   NOT NULL UNIQUE,          -- e.g. CA, NY
    state_name   VARCHAR(50)  NOT NULL,                 -- full state name
    region       VARCHAR(30)  NOT NULL,                 -- Northeast, Southeast, etc.
    urban_rural  VARCHAR(20)  NOT NULL                  -- urban / suburban / rural
);

CREATE INDEX IF NOT EXISTS ix_dim_region_state ON dim_region (state_code);

-- ────────────────────────────────────────────────────────────────────────────
-- Dimension: Diagnosis
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_diagnosis (
    diagnosis_key      INTEGER      PRIMARY KEY AUTOINCREMENT,
    diagnosis_code     VARCHAR(20)  NOT NULL UNIQUE,    -- e.g. DX01, UNKNOWN
    diagnosis_category VARCHAR(80)  NOT NULL,            -- human-readable category
    severity_weight    REAL         NOT NULL DEFAULT 1.0 -- relative severity multiplier
);

CREATE INDEX IF NOT EXISTS ix_dim_diagnosis_code ON dim_diagnosis (diagnosis_code);

-- ────────────────────────────────────────────────────────────────────────────
-- Dimension: Root Cause
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_root_cause (
    root_cause_key      INTEGER      PRIMARY KEY AUTOINCREMENT,
    root_cause_code     VARCHAR(20)  NOT NULL UNIQUE,   -- e.g. RC01
    root_cause_category VARCHAR(80)  NOT NULL,           -- descriptive category
    preventability      VARCHAR(30)  NOT NULL            -- preventable / non-preventable
);

CREATE INDEX IF NOT EXISTS ix_dim_root_cause_code ON dim_root_cause (root_cause_code);

-- ────────────────────────────────────────────────────────────────────────────
-- Fact: Claim
-- ────────────────────────────────────────────────────────────────────────────
-- Central fact table -- one row per insurance claim.  Contains all monetary
-- measures, timing metrics, and foreign keys to each dimension.
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fact_claim (
    -- Primary key (natural: claim identifier)
    claim_id             VARCHAR(20)  PRIMARY KEY,

    -- Date attributes
    incident_date        DATE         NOT NULL,
    report_date          DATE         NOT NULL,
    close_date           DATE,                          -- NULL if still open

    -- Foreign keys to dimension tables
    provider_key         INTEGER      NOT NULL REFERENCES dim_provider(provider_key),
    region_key           INTEGER      NOT NULL REFERENCES dim_region(region_key),
    diagnosis_key        INTEGER      NOT NULL REFERENCES dim_diagnosis(diagnosis_key),
    root_cause_key       INTEGER      NOT NULL REFERENCES dim_root_cause(root_cause_key),

    -- Categorical attributes
    claim_type           VARCHAR(40)  NOT NULL,          -- e.g. general_liability
    procedure_category   VARCHAR(80)  NOT NULL,          -- e.g. Surgery, Rehabilitation
    severity_level       SMALLINT     NOT NULL CHECK (severity_level BETWEEN 1 AND 5),
    status               VARCHAR(20)  NOT NULL,          -- open / closed / reopened

    -- Monetary measures
    paid_amount          REAL         NOT NULL DEFAULT 0 CHECK (paid_amount >= 0),
    incurred_amount      REAL         NOT NULL DEFAULT 0 CHECK (incurred_amount >= 0),
    reserved_amount      REAL         NOT NULL DEFAULT 0,

    -- Timing measures (pre-computed for query performance)
    days_to_close        INTEGER,                        -- NULL if still open
    days_to_report       INTEGER      NOT NULL,          -- report_date - incident_date

    -- Patient demographics
    patient_age_band     VARCHAR(20)  NOT NULL,          -- e.g. 18-29, 30-39
    patient_risk_segment VARCHAR(20)  NOT NULL,          -- low / medium / high

    -- Boolean flags (stored as 0/1 in SQLite)
    repeat_event_flag    INTEGER      NOT NULL DEFAULT 0,
    litigation_flag      INTEGER      NOT NULL DEFAULT 0,

    -- Time-intelligence columns for loss triangles
    accident_year        INTEGER      NOT NULL,
    development_year     INTEGER      NOT NULL
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS ix_fact_claim_provider   ON fact_claim (provider_key);
CREATE INDEX IF NOT EXISTS ix_fact_claim_region     ON fact_claim (region_key);
CREATE INDEX IF NOT EXISTS ix_fact_claim_diagnosis  ON fact_claim (diagnosis_key);
CREATE INDEX IF NOT EXISTS ix_fact_claim_root_cause ON fact_claim (root_cause_key);
CREATE INDEX IF NOT EXISTS ix_fact_claim_status     ON fact_claim (status);
CREATE INDEX IF NOT EXISTS ix_fact_claim_acc_year   ON fact_claim (accident_year);
CREATE INDEX IF NOT EXISTS ix_fact_claim_inc_date   ON fact_claim (incident_date);
