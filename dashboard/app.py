"""Dash application factory and centralized data store."""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
from typing import Optional

import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from sqlalchemy import text

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so that ``config`` and ``src`` imports
# resolve correctly regardless of how the dashboard is launched.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DB_URL, DASH_HOST, DASH_PORT, DASH_DEBUG  # noqa: E402
from src.database.connection import get_engine  # noqa: E402
from src.analysis import (  # noqa: E402
    analyze_frequency_severity,
    analyze_time_to_close,
    build_loss_triangle,
    analyze_reporting_lag,
    analyze_repeat_incidents,
    calculate_benchmarks,
    run_scenarios,
)
from src.models import (  # noqa: E402
    train_severity_model,
    forecast_claims,
    detect_anomalies,
)

from dashboard.nav import create_sidebar  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# DataStore -- singleton that loads all data at startup
# ══════════════════════════════════════════════════════════════════════════════

class DataStore:
    """Singleton data store that loads claims data and runs all analyses.

    Call ``DataStore.get_instance()`` to obtain the shared instance.  The
    first call triggers data loading; subsequent calls return the cached
    instance immediately.
    """

    _instance: Optional["DataStore"] = None
    _lock = threading.Lock()

    # ── singleton accessor ────────────────────────────────────────────────
    @classmethod
    def get_instance(cls, db_url: Optional[str] = None) -> "DataStore":
        """Return the singleton *DataStore*, creating it on first call."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = cls(db_url=db_url)
        return cls._instance

    # ── initializer ───────────────────────────────────────────────────────
    def __init__(self, db_url: Optional[str] = None) -> None:
        t0 = time.perf_counter()

        # Connect to the warehouse
        url = db_url or DB_URL
        engine = get_engine(url)

        # Load the full claims summary (fact + all dimensions).
        # Try the v_claims_summary view first; if the DB file or the view
        # does not exist yet, fall back to an inline join.  If even the
        # underlying tables are missing, surface a clear error message.
        _FALLBACK_SQL = """
        SELECT
            f.claim_id, f.incident_date, f.report_date, f.close_date,
            f.claim_type, f.procedure_category, f.severity_level,
            f.status, f.paid_amount, f.incurred_amount, f.reserved_amount,
            f.days_to_close, f.days_to_report, f.patient_age_band,
            f.patient_risk_segment, f.repeat_event_flag, f.litigation_flag,
            f.accident_year, f.development_year,
            p.provider_id, p.provider_name, p.specialty, p.department, p.risk_tier,
            r.state_code, r.state_name, r.region, r.urban_rural,
            d.diagnosis_code, d.diagnosis_category, d.severity_weight,
            rc.root_cause_code, rc.root_cause_category, rc.preventability,
            ROUND(f.paid_amount / NULLIF(f.incurred_amount, 0), 4) AS loss_ratio,
            ROUND(f.reserved_amount - f.paid_amount, 2) AS reserve_adequacy,
            SUBSTR(f.incident_date, 1, 7) AS incident_month,
            CASE WHEN f.severity_level >= 4 THEN 1 ELSE 0 END AS is_high_severity,
            CASE WHEN f.days_to_close > 365 THEN 1 ELSE 0 END AS is_long_tail
        FROM fact_claim f
        JOIN dim_provider   p  ON f.provider_key   = p.provider_key
        JOIN dim_region     r  ON f.region_key     = r.region_key
        JOIN dim_diagnosis  d  ON f.diagnosis_key  = d.diagnosis_key
        JOIN dim_root_cause rc ON f.root_cause_key = rc.root_cause_key
        """

        try:
            with engine.connect() as conn:
                self.claims_df: pd.DataFrame = pd.read_sql(
                    text("SELECT * FROM v_claims_summary"), conn
                )
        except Exception:
            try:
                with engine.connect() as conn:
                    self.claims_df = pd.read_sql(text(_FALLBACK_SQL), conn)
            except Exception as exc:
                raise RuntimeError(
                    "Could not load claims data. The database or required "
                    "tables/views do not exist yet. Run the full pipeline "
                    "first:  python run.py  (or at minimum: "
                    "python run.py --step generate && "
                    "python run.py --step clean && "
                    "python run.py --step load)"
                ) from exc

        print(
            f"[DataStore] Loaded {len(self.claims_df):,} claims "
            f"from {url}"
        )

        # ── Analysis modules ──────────────────────────────────────────────
        df = self.claims_df

        self.freq_sev = analyze_frequency_severity(df)
        self.time_close = analyze_time_to_close(df)
        self.loss_tri = build_loss_triangle(df)
        self.report_lag = analyze_reporting_lag(df)
        self.repeat = analyze_repeat_incidents(df)
        self.benchmarks = calculate_benchmarks(df)
        self.scenarios = run_scenarios(df)

        # ── ML models ─────────────────────────────────────────────────────
        self.severity_model = train_severity_model(df)
        self.forecast = forecast_claims(df)
        self.anomalies = detect_anomalies(df)

        elapsed = time.perf_counter() - t0
        print(f"[DataStore] All analyses complete in {elapsed:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# Application factory
# ══════════════════════════════════════════════════════════════════════════════

def create_app(data_store: Optional[DataStore] = None) -> Dash:
    """Create and configure the Dash application.

    Parameters
    ----------
    data_store : DataStore, optional
        Pre-loaded data store.  When *None* the singleton is fetched (and
        created on the first call).

    Returns
    -------
    Dash
        Fully configured Dash application ready to be served.
    """
    if data_store is None:
        data_store = DataStore.get_instance()

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        assets_folder=str(Path(__file__).resolve().parent / "assets"),
        title="Insurance Claims Risk Intelligence",
    )

    # ── Layout ────────────────────────────────────────────────────────────
    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            create_sidebar(),
            html.Div(id="page-content", className="page-content"),
        ],
        className="app-container",
    )

    # ── URL-based routing ─────────────────────────────────────────────────
    # Uses the centralized PAGE_MODULES registry from dashboard.pages so
    # that adding a new page only requires updating the registry.
    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
    )
    def _route(pathname: str):
        from dashboard.pages import PAGE_MODULES  # noqa: F811

        # Build routes dict: path -> layout attribute of each module
        routes = {
            path: module.layout
            for path, module in PAGE_MODULES.items()
        }

        page_factory = routes.get(pathname, routes["/"])

        # Page layouts may be either a component or a callable that
        # accepts the data store and returns a component.
        if callable(page_factory):
            return page_factory(data_store)
        return page_factory

    # ── Register page-level callbacks ─────────────────────────────────────
    _register_page_callbacks(app, data_store)

    return app


def _register_page_callbacks(app: Dash, data_store: DataStore) -> None:
    """Import every page module and call its ``register_callbacks`` hook.

    Uses the centralized PAGE_MODULES registry so new pages are
    automatically picked up.
    """
    from dashboard.pages import PAGE_MODULES  # noqa: F811

    for module in PAGE_MODULES.values():
        if hasattr(module, "register_callbacks"):
            module.register_callbacks(app, data_store)


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    store = DataStore.get_instance()
    application = create_app(store)
    application.run(host=DASH_HOST, port=DASH_PORT, debug=DASH_DEBUG)
