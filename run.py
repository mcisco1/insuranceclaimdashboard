"""
Insurance Claims & Clinical Risk Intelligence Dashboard
Single entry point for the full data pipeline.

Usage:
    python run.py                    # Run full pipeline
    python run.py --step generate    # Run only data generation
    python run.py --step clean       # Run only cleaning
    python run.py --step load        # Run only database loading
    python run.py --step analyze     # Run analysis (prints summary)
    python run.py --step export      # Run exports
    python run.py --step dashboard   # Launch dashboard only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import webbrowser
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so all imports resolve regardless of
# how this script is invoked.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Banner
# ══════════════════════════════════════════════════════════════════════════════

_BANNER = """
================================================================
   Insurance Claims & Clinical Risk Intelligence Dashboard
                     Data Pipeline Runner
================================================================
"""


def _print_banner() -> None:
    print(_BANNER)


# ══════════════════════════════════════════════════════════════════════════════
# Individual pipeline steps
# ══════════════════════════════════════════════════════════════════════════════

def step_generate() -> None:
    """Step 1: Generate synthetic claims data."""
    from src.generation.synthetic_claims import generate_claims

    logger.info("STEP: Generate synthetic data")
    t0 = time.perf_counter()
    try:
        generate_claims()
        elapsed = time.perf_counter() - t0
        logger.info("Data generation completed in %.2fs", elapsed)
    except Exception as exc:
        logger.error("ERROR during data generation: %s", exc)
        raise


def step_clean() -> None:
    """Step 2: Clean raw data."""
    from src.cleaning.cleaner import clean_claims

    logger.info("STEP: Clean raw data")
    t0 = time.perf_counter()
    try:
        clean_claims()
        elapsed = time.perf_counter() - t0
        logger.info("Data cleaning completed in %.2fs", elapsed)
    except Exception as exc:
        logger.error("ERROR during data cleaning: %s", exc)
        raise


def step_load() -> None:
    """Step 3: Load cleaned data into the database warehouse."""
    from src.database.loader import load_database

    logger.info("STEP: Load database")
    t0 = time.perf_counter()
    try:
        load_database()
        elapsed = time.perf_counter() - t0
        logger.info("Database loading completed in %.2fs", elapsed)
    except Exception as exc:
        logger.error("ERROR during database loading: %s", exc)
        raise


def step_analyze() -> None:
    """Step 4: Run all analysis modules and print summary results."""
    import pandas as pd
    from sqlalchemy import text

    from config import DB_URL
    from src.database.connection import get_engine
    from src.analysis import (
        analyze_frequency_severity,
        analyze_time_to_close,
        build_loss_triangle,
        analyze_reporting_lag,
        analyze_repeat_incidents,
        calculate_benchmarks,
        run_scenarios,
    )
    from src.models import (
        train_severity_model,
        forecast_claims,
        detect_anomalies,
    )

    logger.info("STEP: Run analyses")
    t0 = time.perf_counter()

    try:
        engine = get_engine(DB_URL)
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM v_claims_summary"), conn)
        logger.info("Loaded %s claims from warehouse", f"{len(df):,}")
    except Exception as exc:
        logger.error("ERROR loading data for analysis: %s", exc)
        raise

    # -- Analysis modules --
    analyses = [
        ("Frequency & Severity", lambda: analyze_frequency_severity(df)),
        ("Time to Close", lambda: analyze_time_to_close(df)),
        ("Loss Triangle", lambda: build_loss_triangle(df)),
        ("Reporting Lag", lambda: analyze_reporting_lag(df)),
        ("Repeat Incidents", lambda: analyze_repeat_incidents(df)),
        ("Benchmarks", lambda: calculate_benchmarks(df)),
        ("Scenario Analysis", lambda: run_scenarios(df)),
    ]

    results = {}
    for name, func in analyses:
        try:
            ta = time.perf_counter()
            result = func()
            tb = time.perf_counter()
            results[name] = result
            logger.info("%-25s  OK  (%.2fs)", name, tb - ta)
        except Exception as exc:
            logger.error("%-25s  FAILED: %s", name, exc)

    # -- ML models --
    ml_modules = [
        ("Severity Predictor", lambda: train_severity_model(df)),
        ("Forecasting", lambda: forecast_claims(df)),
        ("Anomaly Detection", lambda: detect_anomalies(df)),
    ]

    for name, func in ml_modules:
        try:
            ta = time.perf_counter()
            result = func()
            tb = time.perf_counter()
            results[name] = result
            logger.info("%-25s  OK  (%.2fs)", name, tb - ta)
        except Exception as exc:
            logger.error("%-25s  FAILED: %s", name, exc)

    # -- Print summary --
    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("  ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info("  Modules executed:  %d", len(results))
    logger.info("  Total time:        %.2fs", elapsed)

    for name, result in results.items():
        if isinstance(result, dict):
            keys_preview = ", ".join(list(result.keys())[:5])
            logger.info("  %s: keys=[%s]", name, keys_preview)
        else:
            logger.info("  %s: %s", name, type(result).__name__)

    logger.info("=" * 60)


def step_export() -> None:
    """Step 5: Export data for BI tools."""
    from src.exports.exporters import export_all

    logger.info("STEP: Export data")
    t0 = time.perf_counter()
    try:
        paths = export_all()
        elapsed = time.perf_counter() - t0
        logger.info("%d exports completed in %.2fs", len(paths), elapsed)
    except Exception as exc:
        logger.error("ERROR during export: %s", exc)
        raise


def step_dashboard(no_browser: bool = False) -> None:
    """Step 6: Launch the Dash dashboard."""
    from config import DASH_HOST, DASH_PORT, DASH_DEBUG
    from dashboard.app import DataStore, create_app

    logger.info("STEP: Launch dashboard")
    t0 = time.perf_counter()
    try:
        store = DataStore.get_instance()
        app = create_app(store)
        elapsed = time.perf_counter() - t0
        logger.info("DataStore ready in %.2fs", elapsed)
        url = f"http://{DASH_HOST if DASH_HOST != '0.0.0.0' else '127.0.0.1'}:{DASH_PORT}"
        logger.info("Starting server at %s", url)
        if not no_browser:
            threading.Timer(1.5, webbrowser.open, args=[url]).start()
        app.run(host=DASH_HOST, port=DASH_PORT, debug=DASH_DEBUG)
    except Exception as exc:
        logger.error("ERROR launching dashboard: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# Step registry
# ══════════════════════════════════════════════════════════════════════════════

_STEPS = {
    "generate": step_generate,
    "clean": step_clean,
    "load": step_load,
    "analyze": step_analyze,
    "export": step_export,
    "dashboard": step_dashboard,
}

_FULL_PIPELINE_ORDER = [
    "generate",
    "clean",
    "load",
    "analyze",
    "export",
    "dashboard",
]


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insurance Claims & Clinical Risk Intelligence Dashboard -- pipeline runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step",
        choices=list(_STEPS.keys()),
        default=None,
        help="Run a single pipeline step. If omitted, the full pipeline runs sequentially.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=False,
        help="Do not auto-open the browser when launching the dashboard.",
    )

    args = parser.parse_args()

    setup_logging()
    _print_banner()

    if args.step:
        # Run a single step
        steps_to_run = [args.step]
    else:
        # Run the full pipeline
        steps_to_run = _FULL_PIPELINE_ORDER
        logger.info("Running full pipeline: %s", " -> ".join(steps_to_run))

    pipeline_t0 = time.perf_counter()

    for step_name in steps_to_run:
        try:
            if step_name == "dashboard":
                step_dashboard(no_browser=args.no_browser)
            else:
                _STEPS[step_name]()
        except Exception:
            logger.error("Pipeline halted at step '%s' due to error.", step_name)
            sys.exit(1)

    pipeline_elapsed = time.perf_counter() - pipeline_t0
    logger.info("Pipeline finished in %.2fs", pipeline_elapsed)


if __name__ == "__main__":
    main()
