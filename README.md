# Insurance Claims & Clinical Risk Intelligence Dashboard

A full-stack actuarial analytics platform that transforms raw insurance claims data into actionable risk intelligence. The project covers every stage of the data lifecycle -- synthetic data generation, cleaning, star-schema warehousing, statistical analysis, machine learning, and an interactive 8-page dashboard -- providing the kind of end-to-end analytical capability that risk management teams rely on to reduce loss costs and improve patient safety outcomes.

Built as a self-contained demonstration of insurance analytics, this project generates its own realistic dataset (20,000 claims across 250 providers, 50 states, and 4 years), cleans it, loads it into a dimensional warehouse, runs seven analytical modules plus three ML models, and surfaces everything through a Plotly Dash dashboard with export pipelines for Power BI and Tableau.

## Features

- **End-to-end data pipeline** -- generation, cleaning, warehousing, analysis, export, and visualization in a single `python run.py` command
- **Star schema data warehouse** in SQLite with a central fact table and four dimension tables
- **7 analytical modules** covering frequency/severity, time-to-close, loss development, reporting lag, repeat incidents, benchmarking, and scenario analysis
- **3 ML models** -- severity prediction (Random Forest), time series forecasting (Holt-Winters exponential smoothing), and anomaly detection (Isolation Forest)
- **8-page interactive Plotly Dash dashboard** with cross-filtering, drill-downs, and KPI cards
- **Chain-ladder loss development triangle** with age-to-age factors and IBNR reserve estimation
- **What-if scenario analysis** with quantified financial impact of hypothetical interventions
- **AHRQ PSI-style benchmarking** comparing facility-level patient safety indicators against dataset-wide averages
- **Export-ready outputs** for Power BI and Tableau (CSV, formatted multi-sheet Excel)
- **130+ automated tests** across all pipeline stages with pytest

## Screenshots

> Screenshots will be available after running the dashboard. Launch the pipeline with `python run.py` and navigate to `http://localhost:8050` to explore the interactive pages.

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd insurance-claims-dashboard

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (generate data -> clean -> load DB -> analyze -> export -> launch dashboard)
python run.py

# Open http://localhost:8050 in your browser
```

The full pipeline takes about 30-60 seconds depending on your machine. On the first run it creates the `data/` subdirectories, generates 20,000 synthetic claims, builds the SQLite warehouse, runs all analyses, exports BI-ready files, and launches the dashboard.

## Usage

You can run any individual pipeline step on its own:

```bash
python run.py --step generate    # Generate 20,000 synthetic claims
python run.py --step clean       # Clean and validate raw data
python run.py --step load        # Load into SQLite star-schema warehouse
python run.py --step analyze     # Run all 7 analysis modules + 3 ML models
python run.py --step export      # Generate Power BI / Tableau exports
python run.py --step dashboard   # Launch the Dash dashboard only
```

Running without `--step` executes the full pipeline in sequence.

## Project Structure

```
insurance-claims-dashboard/
├── config.py                        # Central configuration (paths, seeds, thresholds)
├── run.py                           # Pipeline runner (CLI entry point)
├── requirements.txt
│
├── src/
│   ├── generation/                  # Synthetic data generation
│   │   ├── synthetic_claims.py      #   Faker-based claim generator
│   │   ├── distributions.py         #   Loss amount & severity distributions
│   │   └── reference_data.py        #   Provider, region, diagnosis lookups
│   ├── cleaning/                    # Data cleaning & validation
│   │   ├── cleaner.py               #   Main cleaning pipeline
│   │   └── validators.py            #   Field-level validation rules
│   ├── database/                    # SQLAlchemy models & loader
│   │   ├── models.py                #   ORM models (star schema)
│   │   ├── loader.py                #   ETL into SQLite warehouse
│   │   └── connection.py            #   Engine management
│   ├── analysis/                    # 7 analytical modules
│   │   ├── frequency_severity.py
│   │   ├── time_to_close.py
│   │   ├── loss_development.py
│   │   ├── reporting_lag.py
│   │   ├── repeat_incidents.py
│   │   ├── benchmarks.py
│   │   └── scenario_analysis.py
│   ├── models/                      # ML models
│   │   ├── severity_predictor.py    #   Random Forest severity prediction
│   │   ├── forecasting.py           #   Holt-Winters time series forecasting
│   │   └── anomaly_detection.py     #   Isolation Forest anomaly detection
│   └── exports/
│       └── exporters.py             # CSV & Excel export for BI tools
│
├── dashboard/
│   ├── app.py                       # Dash application factory & DataStore
│   ├── nav.py                       # Navigation sidebar
│   ├── components/                  # Reusable UI components
│   │   ├── kpi_card.py
│   │   ├── filter_panel.py
│   │   └── download_button.py
│   └── pages/                       # 8 dashboard pages
│       ├── executive_overview.py
│       ├── loss_trends.py
│       ├── loss_development.py
│       ├── geographic_risk.py
│       ├── operational_efficiency.py
│       ├── forecasting_anomalies.py
│       ├── scenario_analysis.py
│       └── recommendations.py
│
├── sql/
│   ├── schema.sql                   # DDL reference (star schema)
│   ├── views.sql                    # Analytical views
│   └── analytical_queries.sql       # Standalone SQL queries
│
├── data/
│   ├── raw/                         # Generated synthetic CSV
│   ├── cleaned/                     # Validated & cleaned CSV
│   ├── warehouse/                   # SQLite database (claims.db)
│   └── exports/                     # BI-ready CSV & Excel files
│
├── reports/
│   ├── executive_memo.md            # Executive risk intelligence memo
│   └── data_quality_report.md       # Cleaning audit trail
│
└── tests/
    ├── conftest.py                  # Shared fixtures
    ├── test_generation.py
    ├── test_cleaning.py
    ├── test_database.py
    ├── test_analysis.py
    └── test_models.py
```

## Data Model

The warehouse uses a Kimball-style star schema optimised for analytical queries. One central fact table stores all claim measures and links to four dimension tables through surrogate keys.

```
                    ┌──────────────┐
                    │ dim_provider │
                    │──────────────│
                    │ provider_key │◄──┐
                    │ provider_id  │   │
                    │ specialty    │   │
                    │ department   │   │
                    │ risk_tier    │   │
                    └──────────────┘   │
                                      │
┌──────────────┐    ┌─────────────────┴────────────────┐    ┌────────────────┐
│  dim_region  │    │            fact_claim             │    │ dim_diagnosis  │
│──────────────│    │──────────────────────────────────│    │────────────────│
│ region_key   │◄───│ claim_id  (PK)                   │───►│ diagnosis_key  │
│ state_code   │    │ incident_date, report_date        │    │ diagnosis_code │
│ state_name   │    │ provider_key  (FK)                │    │ diagnosis_cat  │
│ region       │    │ region_key    (FK)                │    │ severity_weight│
│ urban_rural  │    │ diagnosis_key (FK)                │    └────────────────┘
└──────────────┘    │ root_cause_key(FK)                │
                    │ paid_amount, incurred, reserved   │    ┌────────────────┐
                    │ severity_level, status             │    │ dim_root_cause │
                    │ days_to_close, days_to_report     │    │────────────────│
                    │ accident_year, development_year   │───►│ root_cause_key │
                    │ litigation_flag, repeat_event_flag│    │ root_cause_code│
                    └──────────────────────────────────┘    │ root_cause_cat │
                                                            │ preventability │
                                                            └────────────────┘
```

## Analytical Modules

| Module | Description |
|--------|-------------|
| **Frequency & Severity** | Claim counts and average loss by specialty, claim type, region, and time period; severity distribution analysis |
| **Time to Close** | Claim lifecycle metrics, SLA compliance rates (90/180/365-day thresholds), backlog aging |
| **Loss Development** | Chain-ladder triangle with age-to-age factors; projects ultimate losses and IBNR reserves by accident year |
| **Reporting Lag** | Days between incident and report date, segmented by specialty, region, and provider |
| **Repeat Incidents** | Reopen patterns correlated with providers, root causes, specialties, and litigation outcomes |
| **Benchmarks** | AHRQ PSI-style patient safety indicators with facility-vs-average comparisons and quarterly trends |
| **Scenario Analysis** | What-if modeling of interventions (faster closure, severity reduction, litigation mitigation) with dollar-impact estimates |

**ML Models:**

| Model | Technique | Purpose |
|-------|-----------|---------|
| Severity Predictor | Random Forest | Predicts claim severity level from claim attributes |
| Forecasting | Holt-Winters (exponential smoothing) | 12-month ahead claim volume and cost forecasts |
| Anomaly Detection | Isolation Forest | Flags unusual claims for investigation |

## Dashboard Pages

| Page | Route | Description |
|------|-------|-------------|
| **Executive Overview** | `/` | High-level KPIs, status breakdowns, and trend sparklines for leadership reporting |
| **Loss Trends** | `/loss-trends` | Claim frequency and severity over time, loss ratio analysis, year-over-year comparisons |
| **Loss Development** | `/loss-development` | Interactive loss triangle heatmap, age-to-age factors, IBNR projections by accident year |
| **Geographic Risk** | `/geographic-risk` | State-level maps, regional heatmaps, and urban/rural risk comparisons |
| **Operational Efficiency** | `/operational-efficiency` | SLA compliance tracking, close-time distributions, reporting lag analysis, backlog views |
| **Forecasting & Anomalies** | `/forecasting` | Time series forecasts with confidence intervals, anomaly scatter plots, flagged claim details |
| **Scenario Analysis** | `/scenario-analysis` | Interactive sliders for what-if interventions, projected savings waterfall charts, benchmark comparisons |
| **Recommendations** | `/recommendations` | Prioritised risk reduction actions, ROI estimates, and implementation roadmap |

## Technology Stack

- **Python 3.10+**
- **pandas / NumPy** -- data manipulation and numerical computation
- **scikit-learn** -- Random Forest severity model, Isolation Forest anomaly detection
- **SciPy** -- statistical tests and distribution fitting
- **statsmodels** -- Holt-Winters exponential smoothing forecasts
- **SQLAlchemy + SQLite** -- ORM-based star schema warehouse
- **Plotly Dash** -- interactive multi-page dashboard
- **Faker** -- realistic synthetic data generation
- **openpyxl** -- formatted Excel exports
- **pytest** -- test suite

## Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run tests for a specific module
pytest tests/test_analysis.py -v
pytest tests/test_models.py -v

# Run with coverage (if pytest-cov is installed)
pytest tests/ --cov=src --cov-report=term-missing
```

## Exports

The export step generates BI-ready files in `data/exports/`:

| File | Contents |
|------|----------|
| `claims_detail.csv` | Full denormalised claims dataset (fact table joined with all dimensions) |
| `loss_triangle.csv` | Cumulative paid amounts pivoted by accident year and development year |
| `monthly_metrics.csv` | Monthly time series of claim counts, paid/incurred totals, severity, and litigation |
| `provider_scorecard.csv` | Per-provider risk metrics (claim count, avg severity, total paid, litigation rate, SLA performance) |
| `claims_dashboard.xlsx` | Multi-sheet Excel workbook with formatted headers, auto-filters, and number formatting |

**Using with Power BI:** Import the CSV files as data sources or connect directly to `data/warehouse/claims.db` via ODBC. The star schema translates naturally into a Power BI data model.

**Using with Tableau:** Drop the CSV files into Tableau or use the Excel workbook. The `claims_detail.csv` file is fully denormalised and ready for drag-and-drop analysis.

## Reports

The pipeline generates two written reports in the `reports/` directory:

- **`executive_memo.md`** -- A stakeholder-ready executive memo summarising key findings across the portfolio: frequency trends, severity migration, IBNR exposure, SLA performance, and prioritised recommendations with estimated savings.
- **`data_quality_report.md`** -- An audit trail of every data quality issue found during cleaning (negative amounts, mixed-case codes, swapped dates, null diagnoses, outliers), what action was taken, and what remains.

## License

MIT
