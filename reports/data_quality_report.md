# Data Quality Report
Generated: 2026-03-14 04:32:16

## Summary
- Total records examined: 20,000
- Issues found: 3,715
- Issues resolved: 2,515
- Issues remaining: 1,200
- Cleaning time: 0.20s

## Issues Detail
### Negative Paid Amounts
- Found: 383 records — records with negative paid_amount values
- Action: Converted to absolute values
- Remaining: 0

### Mixed-Case State Codes
- Found: 616 records — state codes not in uppercase format
- Action: Converted to uppercase
- Remaining: 0

### Text Severity Values
- Found: 303 records — severity values stored as text instead of numeric
- Action: Mapped text labels to integers 1-5
- Remaining: 0

### Swapped Dates
- Found: 412 records — report_date earlier than incident_date
- Action: Swapped incident_date and report_date back to correct order
- Remaining: 0

### Null Diagnosis Codes
- Found: 793 records — missing diagnosis_id values
- Action: Filled with 'UNKNOWN'
- Remaining: 0

### Extreme Outliers
- Found: 8 records — paid_amount exceeding $5,000,000
- Action: Capped at $5,000,000
- Remaining: 0

### Future Dates
- Found: 1 records — report_date values set in the future
- Remaining: 1,200

### Future Dates
- Found: 1,199 records — close_date values set in the future
- Remaining: 1,200

## Before/After Comparison
| Metric | Before | After |
|--------|--------|-------|
| Extreme Outliers | 8 | 0 |
| Future Dates | 1,200 | 1,200 |
| Mixed-Case State Codes | 616 | 0 |
| Negative Paid Amounts | 383 | 0 |
| Null Diagnosis Codes | 793 | 0 |
| Swapped Dates | 412 | 0 |
| Text Severity Values | 303 | 0 |
| **Total** | **3,715** | **1,200** |

## Column Null Counts (After Cleaning)
| Column | Null Count |
|--------|------------|
| close_date | 5,105 |

## Numeric Column Ranges (After Cleaning)
| Column | Min | Max |
|--------|-----|-----|
| accident_year | 2,022.00 | 2,025.00 |
| development_year | 0.00 | 1.00 |
| incurred_amount | 1,002.35 | 8,393,393.72 |
| litigation_flag | 0.00 | 1.00 |
| paid_amount | 500.00 | 5,000,000.00 |
| patient_age | 0.00 | 95.00 |
| repeat_event_flag | 0.00 | 1.00 |
| report_lag_days | 0.00 | 324.00 |
| report_year | 2,022.00 | 2,026.00 |
| reserve_amount | 502.35 | 4,722,888.12 |
| severity | 1.00 | 5.00 |
