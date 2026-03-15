"""Executive Overview page -- high-level KPIs, status breakdown, trends.

Route: /
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html, dash_table
from dash.exceptions import PreventUpdate

from dashboard.components import create_download_button, create_filter_panel, create_kpi_card, create_info_tooltip
from src.analysis.statistical_utils import bootstrap_ci, proportional_ci

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
PRIMARY = "#0f3460"
SECONDARY = "#16213e"
ACCENT = "#e94560"
SUCCESS = "#00b894"
WARNING = "#fdcb6e"
SEVERITY_COLORS = {1: "#00b894", 2: "#55efc4", 3: "#fdcb6e", 4: "#e17055", 5: "#d63031"}

_FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

_COMMON_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family=_FONT_FAMILY, size=12, color=SECONDARY),
    margin=dict(l=40, r=20, t=40, b=40),
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_currency(val: float) -> str:
    """Format a dollar value as a compact string."""
    if abs(val) >= 1_000_000:
        return f"${val / 1_000_000:,.1f}M"
    if abs(val) >= 1_000:
        return f"${val / 1_000:,.1f}K"
    return f"${val:,.0f}"


def _fmt_pct(val: float) -> str:
    return f"{val:.1f}%"


def _yoy_delta(current: float, previous: float) -> tuple[str, str]:
    """Return (delta_text, delta_color) for year-over-year comparison."""
    if previous == 0:
        return ("N/A", PRIMARY)
    pct = (current - previous) / previous * 100
    sign = "+" if pct >= 0 else ""
    text = f"{sign}{pct:.1f}% YoY"
    color = ACCENT if pct > 0 else SUCCESS
    return text, color


def _apply_filters(
    df: pd.DataFrame,
    years: Optional[List[int]],
    claim_types: Optional[List[str]],
    severities: Optional[List[int]],
    regions: Optional[List[str]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Return a filtered copy of the claims DataFrame."""
    filtered = df.copy()
    if years:
        filtered = filtered[filtered["accident_year"].isin(years)]
    if claim_types:
        filtered = filtered[filtered["claim_type"].isin(claim_types)]
    if severities:
        filtered = filtered[filtered["severity_level"].isin(severities)]
    if regions:
        filtered = filtered[filtered["region"].isin(regions)]
    if start_date:
        filtered = filtered[pd.to_datetime(filtered["incident_date"], errors="coerce") >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered[pd.to_datetime(filtered["incident_date"], errors="coerce") <= pd.to_datetime(end_date)]
    return filtered


def _monthly_sparkline(df: pd.DataFrame, value_col: Optional[str] = None) -> List[float]:
    """Build monthly aggregated sparkline data from *df*."""
    if df.empty:
        return []
    tmp = df.copy()
    tmp["_month"] = pd.to_datetime(tmp["incident_date"], errors="coerce").dt.to_period("M")
    tmp = tmp.dropna(subset=["_month"])
    if tmp.empty:
        return []
    if value_col:
        series = tmp.groupby("_month")[value_col].sum().sort_index()
    else:
        series = tmp.groupby("_month").size().sort_index()
    return series.tolist()


# ═══════════════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════════════

def layout(data_store) -> html.Div:
    """Return the Executive Overview page layout."""
    df = data_store.claims_df

    years = sorted(df["accident_year"].dropna().unique())
    claim_types = sorted(df["claim_type"].dropna().unique())
    severities = sorted(df["severity_level"].dropna().unique())
    regions = sorted(df["region"].dropna().unique())

    dates = pd.to_datetime(df["incident_date"], errors="coerce").dropna()
    min_date = str(dates.min().date()) if not dates.empty else None
    max_date = str(dates.max().date()) if not dates.empty else None

    return html.Div(
        [
            html.H2(
                [
                    "Executive Overview",
                    create_info_tooltip(
                        "Bootstrap CI (1,000 resamples). Wilson interval for proportions.",
                        "exec-tooltip",
                    ),
                ],
                className="page-title",
            ),

            # Filter panel
            create_filter_panel(years, claim_types, severities, regions, min_date=min_date, max_date=max_date),

            # KPI row
            html.Div(id="exec-kpi-row", className="kpi-row"),

            # Row 2: Status donut + Top 10 root causes
            html.Div(
                [
                    html.Div(
                        dcc.Loading(dcc.Graph(id="exec-status-donut", config={"displayModeBar": False}), type="circle"),
                        className="card half-card",
                    ),
                    html.Div(
                        dcc.Loading(dcc.Graph(id="exec-root-causes-bar", config={"displayModeBar": False}), type="circle"),
                        className="card half-card",
                    ),
                ],
                className="chart-row",
            ),

            # Row 3: Monthly trend + Severity distribution
            html.Div(
                [
                    html.Div(
                        dcc.Loading(dcc.Graph(id="exec-monthly-trend", config={"displayModeBar": False}), type="circle"),
                        className="card half-card",
                    ),
                    html.Div(
                        dcc.Loading(dcc.Graph(id="exec-severity-bar", config={"displayModeBar": False}), type="circle"),
                        className="card half-card",
                    ),
                ],
                className="chart-row",
            ),

            # Download
            create_download_button("exec-overview"),
        ],
        className="page-container",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════

def register_callbacks(app, data_store) -> None:  # noqa: C901
    """Register all Executive Overview callbacks."""

    # ------------------------------------------------------------------
    # Reset filters
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("filter-year", "value"),
            Output("filter-claim-type", "value"),
            Output("filter-severity", "value"),
            Output("filter-region", "value"),
        ],
        Input("filter-reset-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_filters(_n):
        return [None, None, None, None]

    # ------------------------------------------------------------------
    # KPI row
    # ------------------------------------------------------------------
    @app.callback(
        Output("exec-kpi-row", "children"),
        [
            Input("filter-date-range", "start_date"),
            Input("filter-date-range", "end_date"),
            Input("filter-year", "value"),
            Input("filter-claim-type", "value"),
            Input("filter-severity", "value"),
            Input("filter-region", "value"),
        ],
    )
    def _update_kpis(start_date, end_date, years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions, start_date, end_date)

        # Determine latest and prior years for YoY comparison
        all_years = sorted(df["accident_year"].dropna().unique())
        if len(all_years) >= 2:
            latest_yr = all_years[-1]
            prior_yr = all_years[-2]
            df_latest = df[df["accident_year"] == latest_yr]
            df_prior = df[df["accident_year"] == prior_yr]
        else:
            df_latest = df
            df_prior = pd.DataFrame(columns=df.columns)

        # KPI 1: Total Claims
        total = len(df)
        total_latest = len(df_latest)
        total_prior = len(df_prior) if not df_prior.empty else 0
        delta_text, delta_color = _yoy_delta(total_latest, total_prior)
        sparkline_claims = _monthly_sparkline(df)

        kpi_total = create_kpi_card(
            "Total Claims",
            f"{total:,}",
            delta=delta_text,
            delta_color=delta_color,
            sparkline_data=sparkline_claims,
        )

        # KPI 2: Total Paid
        total_paid = df["paid_amount"].sum()
        paid_latest = df_latest["paid_amount"].sum()
        paid_prior = df_prior["paid_amount"].sum() if not df_prior.empty else 0
        d_text, d_color = _yoy_delta(paid_latest, paid_prior)
        sparkline_paid = _monthly_sparkline(df, "paid_amount")

        # Bootstrap CI for Total Paid
        paid_values = df["paid_amount"].dropna().values
        if len(paid_values) > 10:
            ci_result = bootstrap_ci(paid_values, np.sum)
            ci_paid_text = f"95% CI: {_fmt_currency(ci_result['ci_lower'])} - {_fmt_currency(ci_result['ci_upper'])}"
        else:
            ci_paid_text = None

        kpi_paid = create_kpi_card(
            "Total Paid ($)",
            _fmt_currency(total_paid),
            delta=d_text,
            delta_color=d_color,
            sparkline_data=sparkline_paid,
            ci_text=ci_paid_text,
        )

        # KPI 3: Avg Severity
        avg_sev = df["severity_level"].mean() if not df.empty else 0
        avg_sev_latest = df_latest["severity_level"].mean() if not df_latest.empty else 0
        avg_sev_prior = df_prior["severity_level"].mean() if not df_prior.empty else 0
        d_text, d_color = _yoy_delta(avg_sev_latest, avg_sev_prior)

        # Bootstrap CI for Avg Severity
        sev_values = df["severity_level"].dropna().values
        if len(sev_values) > 10:
            ci_sev = bootstrap_ci(sev_values, np.mean)
            ci_sev_text = f"95% CI: {ci_sev['ci_lower']:.2f} - {ci_sev['ci_upper']:.2f}"
        else:
            ci_sev_text = None

        kpi_sev = create_kpi_card(
            "Avg Severity",
            f"{avg_sev:.2f}",
            delta=d_text,
            delta_color=d_color,
            ci_text=ci_sev_text,
        )

        # KPI 4: Avg Days to Close
        closed = df.dropna(subset=["days_to_close"])
        avg_close = closed["days_to_close"].mean() if not closed.empty else 0
        closed_l = df_latest.dropna(subset=["days_to_close"])
        closed_p = df_prior.dropna(subset=["days_to_close"])
        avg_close_l = closed_l["days_to_close"].mean() if not closed_l.empty else 0
        avg_close_p = closed_p["days_to_close"].mean() if not closed_p.empty else 0
        d_text, d_color = _yoy_delta(avg_close_l, avg_close_p)
        # For days to close, lower is better -- invert colour
        if d_color == SUCCESS:
            d_color = ACCENT
        elif d_color == ACCENT:
            d_color = SUCCESS

        kpi_close = create_kpi_card(
            "Avg Days to Close",
            f"{avg_close:.0f}",
            delta=d_text,
            delta_color=d_color,
        )

        # KPI 5: Litigation Rate
        lit_rate = df["litigation_flag"].mean() * 100 if not df.empty else 0
        lit_l = df_latest["litigation_flag"].mean() * 100 if not df_latest.empty else 0
        lit_p = df_prior["litigation_flag"].mean() * 100 if not df_prior.empty else 0
        d_text, d_color = _yoy_delta(lit_l, lit_p)

        # Wilson CI for Litigation Rate
        lit_successes = int(df["litigation_flag"].sum())
        lit_trials = len(df)
        if lit_trials > 10:
            ci_lit = proportional_ci(lit_successes, lit_trials)
            ci_lit_text = f"95% CI: {ci_lit['ci_lower']*100:.1f}% - {ci_lit['ci_upper']*100:.1f}%"
        else:
            ci_lit_text = None

        kpi_lit = create_kpi_card(
            "Litigation Rate (%)",
            _fmt_pct(lit_rate),
            delta=d_text,
            delta_color=d_color,
            ci_text=ci_lit_text,
        )

        # KPI 6: Open Claims
        open_count = len(df[df["status"].isin(["open", "reopened"])])
        open_l = len(df_latest[df_latest["status"].isin(["open", "reopened"])]) if not df_latest.empty else 0
        open_p = len(df_prior[df_prior["status"].isin(["open", "reopened"])]) if not df_prior.empty else 0
        d_text, d_color = _yoy_delta(open_l, open_p)

        kpi_open = create_kpi_card(
            "Open Claims",
            f"{open_count:,}",
            delta=d_text,
            delta_color=d_color,
        )

        return [kpi_total, kpi_paid, kpi_sev, kpi_close, kpi_lit, kpi_open]

    # ------------------------------------------------------------------
    # Status donut chart
    # ------------------------------------------------------------------
    @app.callback(
        Output("exec-status-donut", "figure"),
        [
            Input("filter-date-range", "start_date"),
            Input("filter-date-range", "end_date"),
            Input("filter-year", "value"),
            Input("filter-claim-type", "value"),
            Input("filter-severity", "value"),
            Input("filter-region", "value"),
        ],
    )
    def _update_status_donut(start_date, end_date, years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions, start_date, end_date)
        status_counts = df["status"].value_counts()

        color_map = {"closed": SUCCESS, "open": WARNING, "reopened": ACCENT}
        colors = [color_map.get(s, PRIMARY) for s in status_counts.index]

        fig = go.Figure(
            go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
        )
        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text="Claim Status Breakdown", font=dict(size=14)),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            height=380,
        )
        return fig

    # ------------------------------------------------------------------
    # Top 10 root causes
    # ------------------------------------------------------------------
    @app.callback(
        Output("exec-root-causes-bar", "figure"),
        [
            Input("filter-date-range", "start_date"),
            Input("filter-date-range", "end_date"),
            Input("filter-year", "value"),
            Input("filter-claim-type", "value"),
            Input("filter-severity", "value"),
            Input("filter-region", "value"),
        ],
    )
    def _update_root_causes(start_date, end_date, years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions, start_date, end_date)
        top = df["root_cause_category"].value_counts().head(10).sort_values(ascending=True)

        fig = go.Figure(
            go.Bar(
                x=top.values,
                y=top.index,
                orientation="h",
                marker_color=PRIMARY,
                hovertemplate="<b>%{y}</b><br>Claims: %{x:,}<extra></extra>",
            )
        )
        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text="Top 10 Root Causes", font=dict(size=14)),
            xaxis_title="Claim Count",
            yaxis_title="",
            height=380,
            showlegend=False,
        )
        return fig

    # ------------------------------------------------------------------
    # Monthly claims trend (by claim_type)
    # ------------------------------------------------------------------
    @app.callback(
        Output("exec-monthly-trend", "figure"),
        [
            Input("filter-date-range", "start_date"),
            Input("filter-date-range", "end_date"),
            Input("filter-year", "value"),
            Input("filter-claim-type", "value"),
            Input("filter-severity", "value"),
            Input("filter-region", "value"),
        ],
    )
    def _update_monthly_trend(start_date, end_date, years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions, start_date, end_date)
        df = df.copy()
        df["month"] = pd.to_datetime(df["incident_date"], errors="coerce").dt.to_period("M")
        df = df.dropna(subset=["month"])

        monthly = df.groupby(["month", "claim_type"]).size().reset_index(name="count")
        monthly["month"] = monthly["month"].astype(str)

        types = sorted(monthly["claim_type"].unique())
        type_colors = {t: px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)] for i, t in enumerate(types)}

        fig = go.Figure()
        for ct in types:
            sub = monthly[monthly["claim_type"] == ct]
            fig.add_trace(
                go.Scatter(
                    x=sub["month"],
                    y=sub["count"],
                    mode="lines+markers",
                    name=ct,
                    line=dict(width=2, color=type_colors[ct]),
                    marker=dict(size=4),
                    hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:,}<extra></extra>",
                )
            )
        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text="Monthly Claims Trend by Type", font=dict(size=14)),
            xaxis_title="Month",
            yaxis_title="Claim Count",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # Severity distribution stacked bar
    # ------------------------------------------------------------------
    @app.callback(
        Output("exec-severity-bar", "figure"),
        [
            Input("filter-date-range", "start_date"),
            Input("filter-date-range", "end_date"),
            Input("filter-year", "value"),
            Input("filter-claim-type", "value"),
            Input("filter-severity", "value"),
            Input("filter-region", "value"),
        ],
    )
    def _update_severity_bar(start_date, end_date, years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions, start_date, end_date)
        df = df.copy()
        df["year"] = pd.to_datetime(df["incident_date"], errors="coerce").dt.year
        df = df.dropna(subset=["year", "severity_level"])

        pivot = df.groupby(["year", "severity_level"]).size().reset_index(name="count")
        sev_levels = sorted(pivot["severity_level"].unique())

        fig = go.Figure()
        for sev in sev_levels:
            sub = pivot[pivot["severity_level"] == sev]
            fig.add_trace(
                go.Bar(
                    x=sub["year"],
                    y=sub["count"],
                    name=f"Level {int(sev)}",
                    marker_color=SEVERITY_COLORS.get(int(sev), PRIMARY),
                    hovertemplate="<b>Year %{x}</b><br>Level %{fullData.name}: %{y:,}<extra></extra>",
                )
            )
        fig.update_layout(
            **_COMMON_LAYOUT,
            barmode="stack",
            title=dict(text="Severity Distribution by Year", font=dict(size=14)),
            xaxis_title="Year",
            yaxis_title="Claim Count",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # Download CSV
    # ------------------------------------------------------------------
    @app.callback(
        Output("exec-overview-download", "data"),
        Input("exec-overview-btn", "n_clicks"),
        [
            State("filter-date-range", "start_date"),
            State("filter-date-range", "end_date"),
            State("filter-year", "value"),
            State("filter-claim-type", "value"),
            State("filter-severity", "value"),
            State("filter-region", "value"),
        ],
        prevent_initial_call=True,
    )
    def _download_csv(n_clicks, start_date, end_date, years, claim_types, severities, regions):
        if not n_clicks:
            raise PreventUpdate
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions, start_date, end_date)
        export_cols = [
            "claim_id", "incident_date", "claim_type", "severity_level",
            "status", "paid_amount", "incurred_amount", "root_cause_category",
            "region", "state_name", "days_to_close", "litigation_flag",
        ]
        cols = [c for c in export_cols if c in df.columns]
        return dcc.send_data_frame(df[cols].to_csv, "executive_overview.csv", index=False)
