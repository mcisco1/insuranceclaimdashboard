-- ============================================================================
-- Insurance Claims Data Warehouse -- Analytical Queries
-- ============================================================================
-- Database: SQLite
-- Description: 17 showcase SQL queries demonstrating analytical capabilities
--              of the star-schema warehouse.  Each query is self-contained,
--              commented, and uses valid SQLite syntax.
-- ============================================================================


-- ────────────────────────────────────────────────────────────────────────────
-- Query 1: Monthly Claim Frequency with Running Total
-- ────────────────────────────────────────────────────────────────────────────
-- Uses a window function to compute a cumulative running total of claims
-- alongside each month's individual count.  Useful for monitoring whether
-- claim volume is accelerating or decelerating over time.
-- ────────────────────────────────────────────────────────────────────────────
SELECT
    incident_month,
    claim_count,
    SUM(claim_count) OVER (
        ORDER BY incident_month
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM v_monthly_metrics
ORDER BY incident_month;


-- ────────────────────────────────────────────────────────────────────────────
-- Query 2: Severity Distribution by Specialty
-- ────────────────────────────────────────────────────────────────────────────
-- Conditional aggregation to pivot severity levels into columns per
-- specialty.  Helps identify which specialties skew toward higher severity.
-- ────────────────────────────────────────────────────────────────────────────
SELECT
    p.specialty,
    COUNT(*)                                                    AS total_claims,
    SUM(CASE WHEN f.severity_level = 1 THEN 1 ELSE 0 END)     AS sev_1,
    SUM(CASE WHEN f.severity_level = 2 THEN 1 ELSE 0 END)     AS sev_2,
    SUM(CASE WHEN f.severity_level = 3 THEN 1 ELSE 0 END)     AS sev_3,
    SUM(CASE WHEN f.severity_level = 4 THEN 1 ELSE 0 END)     AS sev_4,
    SUM(CASE WHEN f.severity_level = 5 THEN 1 ELSE 0 END)     AS sev_5,
    ROUND(AVG(f.severity_level), 2)                            AS avg_severity
FROM fact_claim f
JOIN dim_provider p ON f.provider_key = p.provider_key
GROUP BY p.specialty
ORDER BY avg_severity DESC;


-- ────────────────────────────────────────────────────────────────────────────
-- Query 3: Top 10 Highest-Cost Claims with Provider Details
-- ────────────────────────────────────────────────────────────────────────────
-- Simple JOIN + ORDER BY to find the most expensive individual claims,
-- enriched with provider and diagnosis information.
-- ────────────────────────────────────────────────────────────────────────────
SELECT
    f.claim_id,
    f.incident_date,
    f.paid_amount,
    f.incurred_amount,
    f.severity_level,
    f.status,
    p.provider_id,
    p.specialty,
    p.risk_tier,
    d.diagnosis_category,
    r.state_code,
    r.region
FROM fact_claim f
JOIN dim_provider  p ON f.provider_key  = p.provider_key
JOIN dim_diagnosis d ON f.diagnosis_key = d.diagnosis_key
JOIN dim_region    r ON f.region_key    = r.region_key
ORDER BY f.paid_amount DESC
LIMIT 10;


-- ────────────────────────────────────────────────────────────────────────────
-- Query 4: Year-over-Year Claim Count Comparison
-- ────────────────────────────────────────────────────────────────────────────
-- CTE to aggregate by accident year, then LAG to compare each year with the
-- prior year.  The pct_change column shows year-over-year growth.
-- ────────────────────────────────────────────────────────────────────────────
WITH yearly AS (
    SELECT
        accident_year,
        COUNT(*)                    AS claim_count,
        ROUND(SUM(paid_amount), 2) AS total_paid
    FROM fact_claim
    GROUP BY accident_year
)
SELECT
    accident_year,
    claim_count,
    total_paid,
    LAG(claim_count) OVER (ORDER BY accident_year)  AS prev_year_count,
    LAG(total_paid)  OVER (ORDER BY accident_year)  AS prev_year_paid,
    ROUND(
        (CAST(claim_count AS REAL)
         - LAG(claim_count) OVER (ORDER BY accident_year))
        / NULLIF(LAG(claim_count) OVER (ORDER BY accident_year), 0)
        * 100, 2
    ) AS pct_change
FROM yearly
ORDER BY accident_year;


-- ────────────────────────────────────────────────────────────────────────────
-- Query 5: Moving 3-Month Average of Paid Amounts
-- ────────────────────────────────────────────────────────────────────────────
-- Window function with a 3-row sliding frame to smooth out monthly paid
-- amounts.  Useful for trend lines on dashboards.
-- ────────────────────────────────────────────────────────────────────────────
SELECT
    incident_month,
    total_paid,
    ROUND(
        AVG(total_paid) OVER (
            ORDER BY incident_month
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ), 2
    ) AS moving_avg_3m
FROM v_monthly_metrics
ORDER BY incident_month;


-- ────────────────────────────────────────────────────────────────────────────
-- Query 6: Claims by Status and Aging Bucket
-- ────────────────────────────────────────────────────────────────────────────
-- CASE expression to bucket claims into aging tiers (0-30, 31-90, 91-180,
-- 181-365, 365+ days) and cross-tabulate with claim status.
-- ────────────────────────────────────────────────────────────────────────────
SELECT
    status,
    CASE
        WHEN days_to_close IS NULL          THEN 'Still Open'
        WHEN days_to_close <= 30            THEN '0-30 days'
        WHEN days_to_close <= 90            THEN '31-90 days'
        WHEN days_to_close <= 180           THEN '91-180 days'
        WHEN days_to_close <= 365           THEN '181-365 days'
        ELSE '365+ days'
    END AS aging_bucket,
    COUNT(*)                                AS claim_count,
    ROUND(SUM(paid_amount), 2)             AS total_paid,
    ROUND(AVG(paid_amount), 2)             AS avg_paid
FROM fact_claim
GROUP BY status, aging_bucket
ORDER BY status, MIN(COALESCE(days_to_close, 9999));


-- ────────────────────────────────────────────────────────────────────────────
-- Query 7: Loss Development Triangle
-- ────────────────────────────────────────────────────────────────────────────
-- Pivot-style query producing an actuarial loss triangle with accident years
-- as rows and development years as columns.  Uses conditional aggregation
-- to place cumulative paid amounts into the correct development-year column.
-- ────────────────────────────────────────────────────────────────────────────
SELECT
    accident_year,
    ROUND(SUM(CASE WHEN development_year = 0 THEN paid_amount END), 2) AS dev_0,
    ROUND(SUM(CASE WHEN development_year = 1 THEN paid_amount END), 2) AS dev_1,
    ROUND(SUM(CASE WHEN development_year = 2 THEN paid_amount END), 2) AS dev_2,
    ROUND(SUM(CASE WHEN development_year = 3 THEN paid_amount END), 2) AS dev_3,
    ROUND(SUM(CASE WHEN development_year = 4 THEN paid_amount END), 2) AS dev_4,
    ROUND(SUM(paid_amount), 2)                                          AS total_paid
FROM fact_claim
GROUP BY accident_year
ORDER BY accident_year;


-- ────────────────────────────────────────────────────────────────────────────
-- Query 8: Provider Risk Scorecard
-- ────────────────────────────────────────────────────────────────────────────
-- Multi-metric aggregation per provider.  Combines volume, severity, cost,
-- litigation risk, and speed-to-close into a single scorecard row.  This
-- powers the provider benchmarking dashboard panel.
-- ────────────────────────────────────────────────────────────────────────────
SELECT
    p.provider_id,
    p.specialty,
    p.risk_tier,
    COUNT(*)                                                      AS claim_count,
    ROUND(AVG(f.severity_level), 2)                              AS avg_severity,
    ROUND(AVG(f.paid_amount), 2)                                 AS avg_paid,
    ROUND(SUM(f.paid_amount), 2)                                 AS total_paid,
    ROUND(100.0 * SUM(f.litigation_flag) / COUNT(*), 2)          AS litigation_pct,
    ROUND(AVG(f.days_to_close), 1)                               AS avg_days_to_close,
    SUM(f.repeat_event_flag)                                      AS repeat_events,
    ROUND(100.0
        * SUM(CASE WHEN f.status = 'reopened' THEN 1 ELSE 0 END)
        / COUNT(*), 2
    )                                                             AS reopen_pct
FROM fact_claim f
JOIN dim_provider p ON f.provider_key = p.provider_key
GROUP BY p.provider_id, p.specialty, p.risk_tier
HAVING COUNT(*) >= 5                       -- exclude low-volume providers
ORDER BY total_paid DESC;


-- ────────────────────────────────────────────────────────────────────────────
-- Query 9: Regional Severity Comparison
-- ────────────────────────────────────────────────────────────────────────────
-- Groups claims by region and filters to regions where the average severity
-- exceeds a threshold.  HAVING clause keeps only high-severity regions.
-- ────────────────────────────────────────────────────────────────────────────
SELECT
    r.region,
    r.urban_rural,
    COUNT(*)                               AS claim_count,
    ROUND(AVG(f.severity_level), 2)       AS avg_severity,
    ROUND(SUM(f.paid_amount), 2)          AS total_paid,
    ROUND(AVG(f.paid_amount), 2)          AS avg_paid,
    SUM(f.litigation_flag)                 AS litigation_count
FROM fact_claim f
JOIN dim_region r ON f.region_key = r.region_key
GROUP BY r.region, r.urban_rural
HAVING AVG(f.severity_level) > 2.5
ORDER BY avg_severity DESC;


-- ────────────────────────────────────────────────────────────────────────────
-- Query 10: Diagnosis Category Pareto Analysis
-- ────────────────────────────────────────────────────────────────────────────
-- Ranks diagnosis categories by total paid and computes a running percentage
-- of total losses.  The classic 80/20 analysis to identify the vital few
-- diagnosis categories driving the majority of costs.
-- ────────────────────────────────────────────────────────────────────────────
WITH ranked AS (
    SELECT
        d.diagnosis_category,
        COUNT(*)                          AS claim_count,
        ROUND(SUM(f.paid_amount), 2)     AS total_paid
    FROM fact_claim f
    JOIN dim_diagnosis d ON f.diagnosis_key = d.diagnosis_key
    GROUP BY d.diagnosis_category
)
SELECT
    diagnosis_category,
    claim_count,
    total_paid,
    ROUND(
        100.0 * SUM(total_paid) OVER (
            ORDER BY total_paid DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) / SUM(total_paid) OVER (), 2
    ) AS cumulative_pct
FROM ranked
ORDER BY total_paid DESC;


-- ────────────────────────────────────────────────────────────────────────────
-- Query 11: Reporting Lag Analysis by Specialty
-- ────────────────────────────────────────────────────────────────────────────
-- Examines how quickly claims are reported after incidents, grouped by
-- specialty.  Uses AVG for the mean lag and a percentile proxy (median via
-- subquery) to handle skewed distributions.
-- ────────────────────────────────────────────────────────────────────────────
SELECT
    p.specialty,
    COUNT(*)                                        AS claim_count,
    ROUND(AVG(f.days_to_report), 1)                AS avg_report_lag,
    ROUND(MIN(f.days_to_report), 1)                AS min_report_lag,
    ROUND(MAX(f.days_to_report), 1)                AS max_report_lag,
    -- Median proxy: use the middle value via a window-based approach
    (
        SELECT f2.days_to_report
        FROM fact_claim f2
        JOIN dim_provider p2 ON f2.provider_key = p2.provider_key
        WHERE p2.specialty = p.specialty
        ORDER BY f2.days_to_report
        LIMIT 1
        OFFSET (
            SELECT COUNT(*) / 2
            FROM fact_claim f3
            JOIN dim_provider p3 ON f3.provider_key = p3.provider_key
            WHERE p3.specialty = p.specialty
        )
    ) AS median_report_lag
FROM fact_claim f
JOIN dim_provider p ON f.provider_key = p.provider_key
GROUP BY p.specialty
ORDER BY avg_report_lag DESC;


-- ────────────────────────────────────────────────────────────────────────────
-- Query 12: Litigation Rate by Severity and Region
-- ────────────────────────────────────────────────────────────────────────────
-- Cross-tabulation of litigation rate across severity levels and regions.
-- Reveals whether litigation correlates more strongly with severity or
-- geography.
-- ────────────────────────────────────────────────────────────────────────────
SELECT
    r.region,
    f.severity_level,
    COUNT(*)                                                      AS claim_count,
    SUM(f.litigation_flag)                                        AS litigation_count,
    ROUND(100.0 * SUM(f.litigation_flag) / COUNT(*), 2)          AS litigation_pct
FROM fact_claim f
JOIN dim_region r ON f.region_key = r.region_key
GROUP BY r.region, f.severity_level
ORDER BY r.region, f.severity_level;


-- ────────────────────────────────────────────────────────────────────────────
-- Query 13: Reopened Claims Analysis
-- ────────────────────────────────────────────────────────────────────────────
-- Filtered aggregation focusing on reopened claims to understand their
-- cost impact relative to the overall portfolio.  Compares reopened vs.
-- non-reopened claims on key metrics.
-- ────────────────────────────────────────────────────────────────────────────
SELECT
    CASE WHEN status = 'reopened' THEN 'Reopened' ELSE 'Not Reopened' END
        AS claim_group,
    COUNT(*)                               AS claim_count,
    ROUND(AVG(paid_amount), 2)            AS avg_paid,
    ROUND(SUM(paid_amount), 2)            AS total_paid,
    ROUND(AVG(severity_level), 2)         AS avg_severity,
    ROUND(AVG(days_to_close), 1)          AS avg_days_to_close,
    ROUND(100.0 * SUM(litigation_flag) / COUNT(*), 2)
                                           AS litigation_pct,
    ROUND(
        100.0 * COUNT(*)
        / (SELECT COUNT(*) FROM fact_claim), 2
    )                                      AS pct_of_total
FROM fact_claim
GROUP BY claim_group
ORDER BY claim_group;


-- ────────────────────────────────────────────────────────────────────────────
-- Query 14: Seasonal Trend Analysis
-- ────────────────────────────────────────────────────────────────────────────
-- Extracts the month component from incident_date and aggregates to reveal
-- seasonal patterns in claim frequency, severity, and cost.
-- ────────────────────────────────────────────────────────────────────────────
SELECT
    CAST(STRFTIME('%m', incident_date) AS INTEGER) AS incident_month_num,
    CASE CAST(STRFTIME('%m', incident_date) AS INTEGER)
        WHEN 1  THEN 'January'
        WHEN 2  THEN 'February'
        WHEN 3  THEN 'March'
        WHEN 4  THEN 'April'
        WHEN 5  THEN 'May'
        WHEN 6  THEN 'June'
        WHEN 7  THEN 'July'
        WHEN 8  THEN 'August'
        WHEN 9  THEN 'September'
        WHEN 10 THEN 'October'
        WHEN 11 THEN 'November'
        WHEN 12 THEN 'December'
    END                                             AS month_name,
    COUNT(*)                                        AS claim_count,
    ROUND(AVG(severity_level), 2)                  AS avg_severity,
    ROUND(SUM(paid_amount), 2)                     AS total_paid,
    ROUND(AVG(paid_amount), 2)                     AS avg_paid
FROM fact_claim
GROUP BY incident_month_num
ORDER BY incident_month_num;


-- ────────────────────────────────────────────────────────────────────────────
-- Query 15: Data Quality Check via SQL
-- ────────────────────────────────────────────────────────────────────────────
-- A comprehensive data quality audit that counts NULLs in critical columns,
-- checks value ranges, and flags potential issues -- all in a single query.
-- ────────────────────────────────────────────────────────────────────────────
SELECT
    'Total rows'                    AS check_name,
    COUNT(*)                        AS result
FROM fact_claim

UNION ALL
SELECT 'NULL incident_date',       COUNT(*) FROM fact_claim WHERE incident_date IS NULL
UNION ALL
SELECT 'NULL report_date',         COUNT(*) FROM fact_claim WHERE report_date IS NULL
UNION ALL
SELECT 'NULL close_date',          COUNT(*) FROM fact_claim WHERE close_date IS NULL
UNION ALL
SELECT 'NULL provider_key',        COUNT(*) FROM fact_claim WHERE provider_key IS NULL
UNION ALL
SELECT 'NULL region_key',          COUNT(*) FROM fact_claim WHERE region_key IS NULL
UNION ALL
SELECT 'NULL diagnosis_key',       COUNT(*) FROM fact_claim WHERE diagnosis_key IS NULL
UNION ALL
SELECT 'Severity out of range',    COUNT(*) FROM fact_claim WHERE severity_level NOT BETWEEN 1 AND 5
UNION ALL
SELECT 'Negative paid_amount',     COUNT(*) FROM fact_claim WHERE paid_amount < 0
UNION ALL
SELECT 'Negative incurred_amount', COUNT(*) FROM fact_claim WHERE incurred_amount < 0
UNION ALL
SELECT 'Paid > Incurred',          COUNT(*) FROM fact_claim WHERE paid_amount > incurred_amount
UNION ALL
SELECT 'Report before incident',   COUNT(*) FROM fact_claim WHERE report_date < incident_date
UNION ALL
SELECT 'Negative days_to_report',  COUNT(*) FROM fact_claim WHERE days_to_report < 0
UNION ALL
SELECT 'Negative days_to_close',   COUNT(*) FROM fact_claim WHERE days_to_close < 0
UNION ALL
SELECT 'Future incident_date',     COUNT(*) FROM fact_claim WHERE incident_date > DATE('now')
UNION ALL
SELECT 'Unknown status values',    COUNT(*) FROM fact_claim WHERE status NOT IN ('open', 'closed', 'reopened')
UNION ALL
SELECT 'Orphan provider keys',     COUNT(*)
    FROM fact_claim f
    LEFT JOIN dim_provider p ON f.provider_key = p.provider_key
    WHERE p.provider_key IS NULL
UNION ALL
SELECT 'Orphan region keys',       COUNT(*)
    FROM fact_claim f
    LEFT JOIN dim_region r ON f.region_key = r.region_key
    WHERE r.region_key IS NULL;


-- ────────────────────────────────────────────────────────────────────────────
-- Query 16: Cohort Analysis by Accident Year
-- ────────────────────────────────────────────────────────────────────────────
-- Retention-style cohort analysis showing how each accident-year cohort
-- develops over time.  Each row is an accident year; columns show the
-- percentage of total ultimate losses recognized at each development stage.
-- ────────────────────────────────────────────────────────────────────────────
WITH cohort_totals AS (
    SELECT
        accident_year,
        SUM(paid_amount) AS ultimate_paid
    FROM fact_claim
    GROUP BY accident_year
),
dev_stage AS (
    SELECT
        f.accident_year,
        f.development_year,
        SUM(f.paid_amount) AS stage_paid
    FROM fact_claim f
    GROUP BY f.accident_year, f.development_year
)
SELECT
    d.accident_year,
    d.development_year,
    ROUND(d.stage_paid, 2)                                       AS stage_paid,
    ROUND(c.ultimate_paid, 2)                                    AS ultimate_paid,
    ROUND(100.0 * d.stage_paid / NULLIF(c.ultimate_paid, 0), 2) AS pct_of_ultimate,
    ROUND(
        100.0 * SUM(d.stage_paid) OVER (
            PARTITION BY d.accident_year
            ORDER BY d.development_year
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) / NULLIF(c.ultimate_paid, 0), 2
    ) AS cumulative_pct
FROM dev_stage d
JOIN cohort_totals c ON d.accident_year = c.accident_year
ORDER BY d.accident_year, d.development_year;


-- ────────────────────────────────────────────────────────────────────────────
-- Query 17: Outlier Identification Using Statistical Thresholds
-- ────────────────────────────────────────────────────────────────────────────
-- Identifies claims where paid_amount exceeds the population mean + 3
-- standard deviations.  Uses a subquery to compute the statistical
-- thresholds and then filters the fact table.
-- ────────────────────────────────────────────────────────────────────────────
WITH stats AS (
    SELECT
        AVG(paid_amount)                          AS mean_paid,
        AVG(paid_amount * paid_amount)
            - AVG(paid_amount) * AVG(paid_amount) AS var_paid
    FROM fact_claim
),
thresholds AS (
    SELECT
        mean_paid,
        -- SQLite lacks SQRT(); approximate via power function workaround.
        -- For exact results, use the application layer.  Here we use the
        -- identity: sqrt(x) = exp(0.5 * ln(x)) which works in SQLite.
        mean_paid + 3.0 * EXP(0.5 * LN(MAX(var_paid, 0.0001))) AS upper_bound,
        mean_paid - 3.0 * EXP(0.5 * LN(MAX(var_paid, 0.0001))) AS lower_bound
    FROM stats
)
SELECT
    f.claim_id,
    f.paid_amount,
    f.severity_level,
    f.status,
    p.specialty,
    r.region,
    d.diagnosis_category,
    ROUND(t.mean_paid, 2)   AS population_mean,
    ROUND(t.upper_bound, 2) AS upper_threshold,
    ROUND(
        (f.paid_amount - t.mean_paid)
        / EXP(0.5 * LN(MAX((SELECT var_paid FROM stats), 0.0001))),
        2
    ) AS z_score
FROM fact_claim f
JOIN dim_provider  p ON f.provider_key  = p.provider_key
JOIN dim_region    r ON f.region_key    = r.region_key
JOIN dim_diagnosis d ON f.diagnosis_key = d.diagnosis_key
CROSS JOIN thresholds t
WHERE f.paid_amount > t.upper_bound
ORDER BY f.paid_amount DESC;
