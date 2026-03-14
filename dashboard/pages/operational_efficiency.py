"""Operational Efficiency page -- close times, SLA compliance, reporting lag.

Route: /operational-efficiency
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from dashboard.components import create_download_button, create_filter_panel, create_kpi_card

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

def _apply_filters(
    df: pd.DataFrame,
    years: Optional[List[int]],
    claim_types: Optional[List[str]],
    severities: Optional[List[int]],
    regions: Optional[List[str]],
) -> pd.DataFrame:
    filtered = df.copy()
    if years:
        filtered = filtered[filtered["accident_year"].isin(years)]
    if claim_types:
        filtered = filtered[filtered["claim_type"].isin(claim_types)]
    if severities:
        filtered = filtered[filtered["severity_level"].isin(severities)]
    if regions:
        filtered = filtered[filtered["region"].isin(regions)]
    return filtered


def _fmt_pct(val: float) -> str:
    return f"{val:.1f}%"


# ═══════════════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════════════

def layout(data_store) -> html.Div:
    """Return the Operational Efficiency page layout."""
    df = data_store.claims_df

    years = sorted(df["accident_year"].dropna().unique())
    claim_types = sorted(df["claim_type"].dropna().unique())
    severities = sorted(df["severity_level"].dropna().unique())
    regions = sorted(df["region"].dropna().unique())

    return html.Div(
        [
            html.H2("Operational Efficiency", className="page-title"),
            create_filter_panel(years, claim_types, severities, regions),

            # Row 1: KPI cards
            html.Div(id="ops-kpi-row", className="kpi-row"),

            # Row 2: Close time histogram | Aging bucket bar
            html.Div(
                [
                    html.Div(
                        dcc.Graph(id="ops-close-histogram", config={"displayModeBar": False}),
                        className="card half-card",
                    ),
                    html.Div(
                        dcc.Graph(id="ops-aging-bar", config={"displayModeBar": False}),
                        className="card half-card",
                    ),
                ],
                className="chart-row",
            ),

            # Row 3: Reporting lag histogram | Monthly reporting lag trend
            html.Div(
                [
                    html.Div(
                        dcc.Graph(id="ops-lag-histogram", config={"displayModeBar": False}),
                        className="card half-card",
                    ),
                    html.Div(
                        dcc.Graph(id="ops-lag-trend", config={"displayModeBar": False}),
                        className="card half-card",
                    ),
                ],
                className="chart-row",
            ),

            # Row 4: SLA compliance table
            html.Div(
                html.Div(
                    [
                        html.H4("SLA Compliance by Claim Type", className="table-title"),
                        html.Div(id="ops-sla-table-container"),
                    ],
                    className="card full-card",
                ),
                className="chart-row",
            ),

            create_download_button("ops"),
        ],
        className="page-container",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════

def register_callbacks(app, data_store) -> None:  # noqa: C901
    """Register all Operational Efficiency callbacks."""

    _filter_inputs = [
        Input("filter-year", "value"),
        Input("filter-claim-type", "value"),
        Input("filter-severity", "value"),
        Input("filter-region", "value"),
    ]

    # ------------------------------------------------------------------
    # KPI cards
    # ------------------------------------------------------------------
    @app.callback(Output("ops-kpi-row", "children"), _filter_inputs)
    def _update_kpis(years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)

        closed = df.dropna(subset=["days_to_close"])

        # Avg Days to Close
        avg_close = closed["days_to_close"].mean() if not closed.empty else 0

        # SLA compliance percentages
        total_closed = len(closed) if not closed.empty else 1
        sla_90 = (closed["days_to_close"] <= 90).sum() / total_closed * 100 if not closed.empty else 0
        sla_180 = (closed["days_to_close"] <= 180).sum() / total_closed * 100 if not closed.empty else 0
        sla_365 = (closed["days_to_close"] <= 365).sum() / total_closed * 100 if not closed.empty else 0

        # Colour logic: green if good, red/accent if bad
        def _sla_color(pct: float) -> str:
            if pct >= 80:
                return SUCCESS
            if pct >= 60:
                return WARNING
            return ACCENT

        kpi_avg = create_kpi_card(
            "Avg Days to Close",
            f"{avg_close:.0f}",
            delta=f"Median: {closed['days_to_close'].median():.0f}" if not closed.empty else None,
            delta_color=PRIMARY,
        )

        kpi_90 = create_kpi_card(
            "SLA 90-Day %",
            _fmt_pct(sla_90),
            delta=f"{(closed['days_to_close'] <= 90).sum():,} of {total_closed:,}",
            delta_color=_sla_color(sla_90),
        )

        kpi_180 = create_kpi_card(
            "SLA 180-Day %",
            _fmt_pct(sla_180),
            delta=f"{(closed['days_to_close'] <= 180).sum():,} of {total_closed:,}",
            delta_color=_sla_color(sla_180),
        )

        kpi_365 = create_kpi_card(
            "SLA 365-Day %",
            _fmt_pct(sla_365),
            delta=f"{(closed['days_to_close'] <= 365).sum():,} of {total_closed:,}",
            delta_color=_sla_color(sla_365),
        )

        return [kpi_avg, kpi_90, kpi_180, kpi_365]

    # ------------------------------------------------------------------
    # Close time histogram (by severity)
    # ------------------------------------------------------------------
    @app.callback(Output("ops-close-histogram", "figure"), _filter_inputs)
    def _close_histogram(years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        closed = df.dropna(subset=["days_to_close", "severity_level"]).copy()
        if closed.empty:
            return go.Figure().update_layout(**_COMMON_LAYOUT, title="Close Time Distribution")

        # Cap at 99th percentile for better visualisation
        cap = closed["days_to_close"].quantile(0.99)
        closed["days_capped"] = closed["days_to_close"].clip(upper=cap)

        sev_levels = sorted(closed["severity_level"].unique())

        fig = go.Figure()
        for sev in sev_levels:
            sub = closed[closed["severity_level"] == sev]
            fig.add_trace(
                go.Histogram(
                    x=sub["days_capped"],
                    name=f"Level {int(sev)}",
                    marker_color=SEVERITY_COLORS.get(int(sev), PRIMARY),
                    opacity=0.7,
                    nbinsx=30,
                    hovertemplate="Days: %{x:.0f}<br>Count: %{y:,}<extra>Level " + str(int(sev)) + "</extra>",
                )
            )
        fig.update_layout(
            **_COMMON_LAYOUT,
            barmode="overlay",
            title=dict(text="Close Time Distribution by Severity", font=dict(size=14)),
            xaxis_title="Days to Close",
            yaxis_title="Claim Count",
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
            height=400,
        )

        # Add SLA reference lines
        for threshold, dash_style in [(90, "dash"), (180, "dot"), (365, "dashdot")]:
            fig.add_vline(
                x=threshold,
                line_dash=dash_style,
                line_color="rgba(15,52,96,0.4)",
                annotation_text=f"{threshold}d SLA",
                annotation_position="top",
                annotation_font_size=10,
            )

        return fig

    # ------------------------------------------------------------------
    # Aging bucket bar chart for open claims
    # ------------------------------------------------------------------
    @app.callback(Output("ops-aging-bar", "figure"), _filter_inputs)
    def _aging_bar(years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        open_claims = df[df["status"].isin(["open", "reopened"])].copy()

        if open_claims.empty:
            return go.Figure().update_layout(**_COMMON_LAYOUT, title="Open Claims Aging")

        # Calculate age from report_date to today
        today = pd.Timestamp.now().normalize()
        open_claims["report_date"] = pd.to_datetime(open_claims["report_date"], errors="coerce")
        open_claims["incident_date"] = pd.to_datetime(open_claims["incident_date"], errors="coerce")
        reference = open_claims["report_date"].fillna(open_claims["incident_date"])
        open_claims["age_days"] = (today - reference).dt.days

        # Bucket the ages
        buckets = [
            ("0-30", 0, 30),
            ("31-90", 31, 90),
            ("91-180", 91, 180),
            ("181-365", 181, 365),
            ("365+", 366, 999_999),
        ]

        bucket_labels = []
        bucket_counts = []
        bucket_colors = []
        color_scale = [SUCCESS, "#55efc4", WARNING, "#e17055", ACCENT]

        for i, (label, lo, hi) in enumerate(buckets):
            count = int(((open_claims["age_days"] >= lo) & (open_claims["age_days"] <= hi)).sum())
            bucket_labels.append(label)
            bucket_counts.append(count)
            bucket_colors.append(color_scale[i])

        fig = go.Figure(
            go.Bar(
                x=bucket_labels,
                y=bucket_counts,
                marker_color=bucket_colors,
                hovertemplate="<b>%{x} days</b><br>Open Claims: %{y:,}<extra></extra>",
            )
        )
        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text="Open Claims Aging Buckets", font=dict(size=14)),
            xaxis_title="Age Bucket (Days)",
            yaxis_title="Open Claims",
            showlegend=False,
            height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # Reporting lag distribution histogram
    # ------------------------------------------------------------------
    @app.callback(Output("ops-lag-histogram", "figure"), _filter_inputs)
    def _lag_histogram(years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        df = df.dropna(subset=["days_to_report"]).copy()
        if df.empty:
            return go.Figure().update_layout(**_COMMON_LAYOUT, title="Reporting Lag Distribution")

        # Cap at 99th percentile for visualisation
        cap = df["days_to_report"].quantile(0.99)
        df["lag_capped"] = df["days_to_report"].clip(upper=cap)

        median_lag = df["days_to_report"].median()
        mean_lag = df["days_to_report"].mean()

        fig = go.Figure(
            go.Histogram(
                x=df["lag_capped"],
                nbinsx=40,
                marker_color=PRIMARY,
                opacity=0.85,
                hovertemplate="Days: %{x:.0f}<br>Count: %{y:,}<extra></extra>",
            )
        )

        # Add median reference line
        fig.add_vline(
            x=median_lag,
            line_dash="dash",
            line_color=ACCENT,
            annotation_text=f"Median: {median_lag:.0f}d",
            annotation_position="top right",
            annotation_font_size=11,
        )

        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text="Reporting Lag Distribution", font=dict(size=14)),
            xaxis_title="Days from Incident to Report",
            yaxis_title="Claim Count",
            showlegend=False,
            height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # Monthly reporting lag trend
    # ------------------------------------------------------------------
    @app.callback(Output("ops-lag-trend", "figure"), _filter_inputs)
    def _lag_trend(years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        df = df.copy()
        df["_dt"] = pd.to_datetime(df["incident_date"], errors="coerce")
        df["month"] = df["_dt"].dt.to_period("M")
        df = df.dropna(subset=["month", "days_to_report"])
        if df.empty:
            return go.Figure().update_layout(**_COMMON_LAYOUT, title="Monthly Reporting Lag Trend")

        monthly = df.groupby("month")["days_to_report"].agg(
            ["median", "mean", "count"]
        ).sort_index()
        monthly.index = monthly.index.astype(str)

        fig = go.Figure()

        # Median line
        fig.add_trace(
            go.Scatter(
                x=monthly.index,
                y=monthly["median"],
                mode="lines+markers",
                name="Median Lag",
                line=dict(width=2, color=PRIMARY),
                marker=dict(size=4),
                hovertemplate="<b>%{x}</b><br>Median: %{y:.1f} days<extra></extra>",
            )
        )

        # Mean line
        fig.add_trace(
            go.Scatter(
                x=monthly.index,
                y=monthly["mean"],
                mode="lines",
                name="Mean Lag",
                line=dict(width=2, color=ACCENT, dash="dash"),
                hovertemplate="<b>%{x}</b><br>Mean: %{y:.1f} days<extra></extra>",
            )
        )

        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text="Monthly Reporting Lag Trend", font=dict(size=14)),
            xaxis_title="Month",
            yaxis_title="Days to Report",
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
            height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # SLA compliance table (by claim_type)
    # ------------------------------------------------------------------
    @app.callback(Output("ops-sla-table-container", "children"), _filter_inputs)
    def _sla_table(years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        closed = df.dropna(subset=["days_to_close"])

        if closed.empty:
            return html.P("No closed claims available for the selected filters.")

        rows = []
        for ct, grp in closed.groupby("claim_type"):
            total = len(grp)
            avg_close = grp["days_to_close"].mean()
            median_close = grp["days_to_close"].median()
            sla_90 = (grp["days_to_close"] <= 90).sum() / total * 100
            sla_180 = (grp["days_to_close"] <= 180).sum() / total * 100
            sla_365 = (grp["days_to_close"] <= 365).sum() / total * 100
            rows.append({
                "Claim Type": ct,
                "Claims": total,
                "Avg Close (days)": round(avg_close, 1),
                "Median Close (days)": round(median_close, 1),
                "SLA 90d (%)": round(sla_90, 1),
                "SLA 180d (%)": round(sla_180, 1),
                "SLA 365d (%)": round(sla_365, 1),
            })

        # Add overall row
        total = len(closed)
        rows.append({
            "Claim Type": "OVERALL",
            "Claims": total,
            "Avg Close (days)": round(closed["days_to_close"].mean(), 1),
            "Median Close (days)": round(closed["days_to_close"].median(), 1),
            "SLA 90d (%)": round((closed["days_to_close"] <= 90).sum() / total * 100, 1),
            "SLA 180d (%)": round((closed["days_to_close"] <= 180).sum() / total * 100, 1),
            "SLA 365d (%)": round((closed["days_to_close"] <= 365).sum() / total * 100, 1),
        })

        table_df = pd.DataFrame(rows)

        return dash_table.DataTable(
            id="ops-sla-datatable",
            columns=[{"name": c, "id": c} for c in table_df.columns],
            data=table_df.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": PRIMARY,
                "color": "white",
                "fontWeight": "bold",
                "fontFamily": _FONT_FAMILY,
                "fontSize": "13px",
                "textAlign": "center",
                "padding": "10px 8px",
            },
            style_cell={
                "textAlign": "center",
                "fontFamily": _FONT_FAMILY,
                "fontSize": "13px",
                "padding": "8px 12px",
                "minWidth": "100px",
            },
            style_data={
                "backgroundColor": "rgba(0,0,0,0)",
            },
            style_data_conditional=[
                # Highlight overall row
                {
                    "if": {"filter_query": '{Claim Type} = "OVERALL"'},
                    "backgroundColor": "rgba(15, 52, 96, 0.08)",
                    "fontWeight": "bold",
                },
                # Conditional colouring for SLA columns
                {
                    "if": {
                        "filter_query": "{SLA 90d (%)} >= 80",
                        "column_id": "SLA 90d (%)",
                    },
                    "color": SUCCESS,
                    "fontWeight": "bold",
                },
                {
                    "if": {
                        "filter_query": "{SLA 90d (%)} < 60",
                        "column_id": "SLA 90d (%)",
                    },
                    "color": ACCENT,
                    "fontWeight": "bold",
                },
                {
                    "if": {
                        "filter_query": "{SLA 180d (%)} >= 80",
                        "column_id": "SLA 180d (%)",
                    },
                    "color": SUCCESS,
                    "fontWeight": "bold",
                },
                {
                    "if": {
                        "filter_query": "{SLA 180d (%)} < 60",
                        "column_id": "SLA 180d (%)",
                    },
                    "color": ACCENT,
                    "fontWeight": "bold",
                },
                {
                    "if": {
                        "filter_query": "{SLA 365d (%)} >= 80",
                        "column_id": "SLA 365d (%)",
                    },
                    "color": SUCCESS,
                    "fontWeight": "bold",
                },
                {
                    "if": {
                        "filter_query": "{SLA 365d (%)} < 60",
                        "column_id": "SLA 365d (%)",
                    },
                    "color": ACCENT,
                    "fontWeight": "bold",
                },
            ],
            sort_action="native",
            page_size=20,
        )

    # ------------------------------------------------------------------
    # Download CSV
    # ------------------------------------------------------------------
    @app.callback(
        Output("ops-download", "data"),
        Input("ops-btn", "n_clicks"),
        [
            State("filter-year", "value"),
            State("filter-claim-type", "value"),
            State("filter-severity", "value"),
            State("filter-region", "value"),
        ],
        prevent_initial_call=True,
    )
    def _download_csv(n_clicks, years, claim_types, severities, regions):
        if not n_clicks:
            raise PreventUpdate
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        export_cols = [
            "claim_id", "incident_date", "report_date", "close_date",
            "claim_type", "severity_level", "status", "days_to_close",
            "days_to_report", "region", "state_name",
        ]
        cols = [c for c in export_cols if c in df.columns]
        return dcc.send_data_frame(df[cols].to_csv, "operational_efficiency.csv", index=False)
