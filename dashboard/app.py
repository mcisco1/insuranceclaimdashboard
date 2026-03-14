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

        # Load the full claims summary (fact + all dimensions)
        query = text("SELECT * FROM v_claims_summary")
        with engine.connect() as conn:
            self.claims_df: pd.DataFrame = pd.read_sql(query, conn)

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
    # Page modules are imported lazily inside the callback to avoid circular
    # imports (each page module may reference components from this package).
    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
    )
    def _route(pathname: str):
        from dashboard.pages import (  # noqa: F811
            executive_overview,
            loss_trends,
            geographic_risk,
            operational_efficiency,
            loss_development,
            forecasting_anomalies,
            scenario_analysis,
            recommendations,
        )

        routes = {
            "/": executive_overview.layout,
            "/loss-trends": loss_trends.layout,
            "/geographic-risk": geographic_risk.layout,
            "/operational-efficiency": operational_efficiency.layout,
            "/loss-development": loss_development.layout,
            "/forecasting": forecasting_anomalies.layout,
            "/scenario-analysis": scenario_analysis.layout,
            "/recommendations": recommendations.layout,
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
    """Import every page module and call its ``register_callbacks`` hook."""
    from dashboard.pages import (  # noqa: F811
        executive_overview,
        loss_trends,
        geographic_risk,
        operational_efficiency,
        loss_development,
        forecasting_anomalies,
        scenario_analysis,
        recommendations,
    )

    for module in (
        executive_overview,
        loss_trends,
        geographic_risk,
        operational_efficiency,
        loss_development,
        forecasting_anomalies,
        scenario_analysis,
        recommendations,
    ):
        if hasattr(module, "register_callbacks"):
            module.register_callbacks(app, data_store)


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    store = DataStore.get_instance()
    application = create_app(store)
    application.run(host=DASH_HOST, port=DASH_PORT, debug=DASH_DEBUG)
