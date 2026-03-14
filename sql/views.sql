-- ============================================================================
-- Insurance Claims Data Warehouse -- Reusable Views
-- ============================================================================
-- Database: SQLite
-- Description: Four analytical views built on top of the star schema.
--              These views simplify downstream queries by pre-joining
--              dimensions and pre-computing common aggregations.
-- ============================================================================

-- ────────────────────────────────────────────────────────────────────────────
-- View 1: v_claims_summary
-- ────────────────────────────────────────────────────────────────────────────
-- Joins fact_claim with all four dimension tables and adds derived fields:
--   - loss_ratio         = paid_amount / NULLIF(incurred_amount, 0)
--   - reserve_adequacy   = reserved_amount - paid_amount
--   - incident_month     = YYYY-MM extracted from incident_date
--   - is_high_severity   = 1 when severity_level >= 4
--   - is_long_tail       = 1 when days_to_close > 365
-- ────────────────────────────────────────────────────────────────────────────
DROP VIEW IF EXISTS v_claims_summary;
CREATE VIEW v_claims_summary AS
SELECT
    f.claim_id,
    f.incident_date,
    f.report_date,
    f.close_date,
    f.claim_type,
    f.procedure_category,
    f.severity_level,
    f.status,
    f.paid_amount,
    f.incurred_amount,
    f.reserved_amount,
    f.days_to_close,
    f.days_to_report,
    f.patient_age_band,
    f.patient_risk_segment,
    f.repeat_event_flag,
    f.litigation_flag,
    f.accident_year,
    f.development_year,

    -- Provider dimension
    p.provider_id,
    p.provider_name,
    p.specialty,
    p.department,
    p.risk_tier,

    -- Region dimension
    r.state_code,
    r.state_name,
    r.region,
    r.urban_rural,

    -- Diagnosis dimension
    d.diagnosis_code,
    d.diagnosis_category,
    d.severity_weight,

    -- Root cause dimension
    rc.root_cause_code,
    rc.root_cause_category,
    rc.preventability,

    -- Derived fields
    ROUND(f.paid_amount / NULLIF(f.incurred_amount, 0), 4)
        AS loss_ratio,
    ROUND(f.reserved_amount - f.paid_amount, 2)
        AS reserve_adequacy,
    SUBSTR(f.incident_date, 1, 7)
        AS incident_month,
    CASE WHEN f.severity_level >= 4 THEN 1 ELSE 0 END
        AS is_high_severity,
    CASE WHEN f.days_to_close > 365 THEN 1 ELSE 0 END
        AS is_long_tail

FROM fact_claim f
JOIN dim_provider   p  ON f.provider_key   = p.provider_key
JOIN dim_region     r  ON f.region_key     = r.region_key
JOIN dim_diagnosis  d  ON f.diagnosis_key  = d.diagnosis_key
JOIN dim_root_cause rc ON f.root_cause_key = rc.root_cause_key;


-- ────────────────────────────────────────────────────────────────────────────
-- View 2: v_monthly_metrics
-- ────────────────────────────────────────────────────────────────────────────
-- Monthly aggregates useful for time-series dashboards.
-- Columns: incident_month, claim_count, total_paid, total_incurred,
--          avg_severity, avg_days_to_close, litigation_count, reopen_count.
-- ────────────────────────────────────────────────────────────────────────────
DROP VIEW IF EXISTS v_monthly_metrics;
CREATE VIEW v_monthly_metrics AS
SELECT
    SUBSTR(f.incident_date, 1, 7)           AS incident_month,
    COUNT(*)                                 AS claim_count,
    ROUND(SUM(f.paid_amount), 2)            AS total_paid,
    ROUND(SUM(f.incurred_amount), 2)        AS total_incurred,
    ROUND(AVG(f.severity_level), 2)         AS avg_severity,
    ROUND(AVG(f.days_to_close), 1)          AS avg_days_to_close,
    SUM(f.litigation_flag)                   AS litigation_count,
    SUM(CASE WHEN f.status = 'reopened' THEN 1 ELSE 0 END)
                                             AS reopen_count
FROM fact_claim f
GROUP BY SUBSTR(f.incident_date, 1, 7)
ORDER BY incident_month;


-- ────────────────────────────────────────────────────────────────────────────
-- View 3: v_loss_triangle
-- ────────────────────────────────────────────────────────────────────────────
-- Accident-year x development-year loss triangle.  Each cell contains the
-- cumulative paid amount for that combination using a correlated subquery.
-- This format is the standard actuarial layout for reserve analysis.
-- ────────────────────────────────────────────────────────────────────────────
DROP VIEW IF EXISTS v_loss_triangle;
CREATE VIEW v_loss_triangle AS
SELECT
    a.accident_year,
    a.development_year,
    a.incremental_paid,
    (
        SELECT ROUND(SUM(b.paid_amount), 2)
        FROM fact_claim b
        WHERE b.accident_year = a.accident_year
          AND b.development_year <= a.development_year
    ) AS cumulative_paid,
    a.claim_count
FROM (
    SELECT
        accident_year,
        development_year,
        ROUND(SUM(paid_amount), 2) AS incremental_paid,
        COUNT(*)                    AS claim_count
    FROM fact_claim
    GROUP BY accident_year, development_year
) a
ORDER BY a.accident_year, a.development_year;


-- ────────────────────────────────────────────────────────────────────────────
-- View 4: v_provider_scorecard
-- ────────────────────────────────────────────────────────────────────────────
-- Per-provider performance scorecard with key risk indicators.
-- Designed to feed provider comparison dashboards and risk-tiering models.
-- ────────────────────────────────────────────────────────────────────────────
DROP VIEW IF EXISTS v_provider_scorecard;
CREATE VIEW v_provider_scorecard AS
SELECT
    p.provider_id,
    p.provider_name,
    p.specialty,
    p.department,
    p.risk_tier,
    COUNT(*)                                           AS claim_count,
    ROUND(AVG(f.severity_level), 2)                   AS avg_severity,
    ROUND(AVG(f.paid_amount), 2)                      AS avg_paid,
    ROUND(SUM(f.paid_amount), 2)                      AS total_paid,
    ROUND(
        CAST(SUM(f.litigation_flag) AS REAL)
        / NULLIF(COUNT(*), 0), 4
    )                                                  AS litigation_rate,
    ROUND(AVG(f.days_to_close), 1)                    AS avg_days_to_close,
    ROUND(AVG(f.days_to_report), 1)                   AS avg_days_to_report,
    SUM(CASE WHEN f.status = 'reopened' THEN 1 ELSE 0 END)
                                                       AS reopen_count,
    SUM(f.repeat_event_flag)                           AS repeat_event_count
FROM fact_claim f
JOIN dim_provider p ON f.provider_key = p.provider_key
GROUP BY
    p.provider_id,
    p.provider_name,
    p.specialty,
    p.department,
    p.risk_tier
ORDER BY total_paid DESC;
