"""Loss Trends page -- claim frequency, severity trends, loss ratios.

Route: /loss-trends
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
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

# Consistent line colours per claim_type (up to 8 types)
_TYPE_PALETTE = px.colors.qualitative.Set2


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
    return filtered


def _prepare_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Add a string *month* column derived from *incident_date*."""
    out = df.copy()
    out["_dt"] = pd.to_datetime(out["incident_date"], errors="coerce")
    out["month"] = out["_dt"].dt.to_period("M")
    return out.dropna(subset=["month"])


def _type_color_map(types: list) -> dict:
    return {t: _TYPE_PALETTE[i % len(_TYPE_PALETTE)] for i, t in enumerate(sorted(types))}


# ═══════════════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════════════

def layout(data_store) -> html.Div:
    """Return the Loss Trends page layout."""
    df = data_store.claims_df

    years = sorted(df["accident_year"].dropna().unique())
    claim_types = sorted(df["claim_type"].dropna().unique())
    severities = sorted(df["severity_level"].dropna().unique())
    regions = sorted(df["region"].dropna().unique())

    return html.Div(
        [
            html.H2("Loss Trends", className="page-title"),
            create_filter_panel(years, claim_types, severities, regions),

            # Row 1: Monthly counts by type | Monthly counts by severity (area)
            html.Div(
                [
                    html.Div(
                        dcc.Graph(id="trends-monthly-type", config={"displayModeBar": False}),
                        className="card half-card",
                    ),
                    html.Div(
                        dcc.Graph(id="trends-monthly-severity-area", config={"displayModeBar": False}),
                        className="card half-card",
                    ),
                ],
                className="chart-row",
            ),

            # Row 2: Paid vs Incurred | Loss ratio trend
            html.Div(
                [
                    html.Div(
                        dcc.Graph(id="trends-paid-incurred", config={"displayModeBar": False}),
                        className="card half-card",
                    ),
                    html.Div(
                        dcc.Graph(id="trends-loss-ratio", config={"displayModeBar": False}),
                        className="card half-card",
                    ),
                ],
                className="chart-row",
            ),

            # Row 3: Severity mix 100% stacked | Claim type treemap
            html.Div(
                [
                    html.Div(
                        dcc.Graph(id="trends-severity-mix", config={"displayModeBar": False}),
                        className="card half-card",
                    ),
                    html.Div(
                        dcc.Graph(id="trends-treemap", config={"displayModeBar": False}),
                        className="card half-card",
                    ),
                ],
                className="chart-row",
            ),

            create_download_button("trends"),
        ],
        className="page-container",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════

def register_callbacks(app, data_store) -> None:
    """Register all Loss Trends callbacks."""

    _filter_inputs = [
        Input("filter-year", "value"),
        Input("filter-claim-type", "value"),
        Input("filter-severity", "value"),
        Input("filter-region", "value"),
    ]

    # ------------------------------------------------------------------
    # Monthly claim counts by type (line chart)
    # ------------------------------------------------------------------
    @app.callback(Output("trends-monthly-type", "figure"), _filter_inputs)
    def _monthly_type(years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        df = _prepare_monthly(df)
        if df.empty:
            return go.Figure().update_layout(**_COMMON_LAYOUT, title="Monthly Claims by Type")

        monthly = df.groupby(["month", "claim_type"]).size().reset_index(name="count")
        monthly["month"] = monthly["month"].astype(str)
        types = sorted(monthly["claim_type"].unique())
        cmap = _type_color_map(types)

        fig = go.Figure()
        for ct in types:
            sub = monthly[monthly["claim_type"] == ct].sort_values("month")
            fig.add_trace(
                go.Scatter(
                    x=sub["month"],
                    y=sub["count"],
                    mode="lines+markers",
                    name=ct,
                    line=dict(width=2, color=cmap[ct]),
                    marker=dict(size=4),
                    hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:,}<extra></extra>",
                )
            )
        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text="Monthly Claim Counts by Type", font=dict(size=14)),
            xaxis_title="Month",
            yaxis_title="Claims",
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
            height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # Monthly claim counts by severity (stacked area)
    # ------------------------------------------------------------------
    @app.callback(Output("trends-monthly-severity-area", "figure"), _filter_inputs)
    def _monthly_severity_area(years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        df = _prepare_monthly(df)
        df = df.dropna(subset=["severity_level"])
        if df.empty:
            return go.Figure().update_layout(**_COMMON_LAYOUT, title="Monthly Claims by Severity")

        monthly = df.groupby(["month", "severity_level"]).size().reset_index(name="count")
        monthly["month"] = monthly["month"].astype(str)
        sev_levels = sorted(monthly["severity_level"].unique())

        fig = go.Figure()
        for sev in sev_levels:
            sub = monthly[monthly["severity_level"] == sev].sort_values("month")
            fig.add_trace(
                go.Scatter(
                    x=sub["month"],
                    y=sub["count"],
                    mode="lines",
                    name=f"Level {int(sev)}",
                    stackgroup="one",
                    line=dict(width=0.5, color=SEVERITY_COLORS.get(int(sev), PRIMARY)),
                    fillcolor=SEVERITY_COLORS.get(int(sev), PRIMARY),
                    hovertemplate="<b>%{x}</b><br>Level %{fullData.name}: %{y:,}<extra></extra>",
                )
            )
        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text="Monthly Claims by Severity (Stacked)", font=dict(size=14)),
            xaxis_title="Month",
            yaxis_title="Claims",
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
            height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # Paid vs Incurred trend
    # ------------------------------------------------------------------
    @app.callback(Output("trends-paid-incurred", "figure"), _filter_inputs)
    def _paid_incurred(years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        df = _prepare_monthly(df)
        if df.empty:
            return go.Figure().update_layout(**_COMMON_LAYOUT, title="Paid vs Incurred Trend")

        monthly = df.groupby("month").agg(
            paid=("paid_amount", "sum"),
            incurred=("incurred_amount", "sum"),
        ).sort_index()
        monthly.index = monthly.index.astype(str)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly.index,
                y=monthly["paid"],
                mode="lines+markers",
                name="Paid",
                line=dict(width=2, color=ACCENT),
                marker=dict(size=4),
                hovertemplate="<b>%{x}</b><br>Paid: $%{y:,.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly.index,
                y=monthly["incurred"],
                mode="lines+markers",
                name="Incurred",
                line=dict(width=2, color=PRIMARY),
                marker=dict(size=4),
                hovertemplate="<b>%{x}</b><br>Incurred: $%{y:,.0f}<extra></extra>",
            )
        )
        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text="Paid vs Incurred Amount Trend", font=dict(size=14)),
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
            height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # Loss ratio trend by quarter
    # ------------------------------------------------------------------
    @app.callback(Output("trends-loss-ratio", "figure"), _filter_inputs)
    def _loss_ratio_trend(years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        df = df.copy()
        df["_dt"] = pd.to_datetime(df["incident_date"], errors="coerce")
        df["quarter"] = df["_dt"].dt.to_period("Q")
        df = df.dropna(subset=["quarter"])
        if df.empty:
            return go.Figure().update_layout(**_COMMON_LAYOUT, title="Loss Ratio Trend")

        quarterly = df.groupby("quarter").agg(
            paid=("paid_amount", "sum"),
            incurred=("incurred_amount", "sum"),
        ).sort_index()
        quarterly["loss_ratio"] = np.where(
            quarterly["incurred"] > 0,
            quarterly["paid"] / quarterly["incurred"],
            0,
        )
        quarterly.index = quarterly.index.astype(str)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=quarterly.index,
                y=quarterly["loss_ratio"],
                mode="lines+markers",
                name="Loss Ratio",
                line=dict(width=2, color=PRIMARY),
                marker=dict(size=6),
                hovertemplate="<b>%{x}</b><br>Loss Ratio: %{y:.3f}<extra></extra>",
            )
        )
        # Reference line at 1.0
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color=ACCENT,
            annotation_text="1.0 (break-even)",
            annotation_position="top right",
        )
        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text="Quarterly Loss Ratio Trend", font=dict(size=14)),
            xaxis_title="Quarter",
            yaxis_title="Loss Ratio (Paid / Incurred)",
            showlegend=False,
            height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # Severity mix over time (100% stacked area)
    # ------------------------------------------------------------------
    @app.callback(Output("trends-severity-mix", "figure"), _filter_inputs)
    def _severity_mix(years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        df = df.copy()
        df["_dt"] = pd.to_datetime(df["incident_date"], errors="coerce")
        df["quarter"] = df["_dt"].dt.to_period("Q")
        df = df.dropna(subset=["quarter", "severity_level"])
        if df.empty:
            return go.Figure().update_layout(**_COMMON_LAYOUT, title="Severity Mix Over Time")

        pivot = df.groupby(["quarter", "severity_level"]).size().unstack(fill_value=0)
        # Normalise to percentages
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        pivot_pct.index = pivot_pct.index.astype(str)
        sev_levels = sorted(pivot_pct.columns)

        fig = go.Figure()
        for sev in sev_levels:
            if sev not in pivot_pct.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=pivot_pct.index,
                    y=pivot_pct[sev],
                    mode="lines",
                    name=f"Level {int(sev)}",
                    stackgroup="one",
                    groupnorm="percent",
                    line=dict(width=0.5),
                    fillcolor=SEVERITY_COLORS.get(int(sev), PRIMARY),
                    hovertemplate="<b>%{x}</b><br>Level "
                    + str(int(sev))
                    + ": %{y:.1f}%<extra></extra>",
                )
            )
        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text="Severity Mix Over Time (100% Stacked)", font=dict(size=14)),
            xaxis_title="Quarter",
            yaxis_title="Share (%)",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
            height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # Claim type composition treemap
    # ------------------------------------------------------------------
    @app.callback(Output("trends-treemap", "figure"), _filter_inputs)
    def _treemap(years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        if df.empty:
            return go.Figure().update_layout(**_COMMON_LAYOUT, title="Claim Type Composition")

        agg = df.groupby("claim_type").agg(
            count=("claim_id", "count"),
            total_paid=("paid_amount", "sum"),
        ).reset_index()

        fig = px.treemap(
            agg,
            path=["claim_type"],
            values="count",
            color="total_paid",
            color_continuous_scale=[SUCCESS, WARNING, ACCENT],
            hover_data={"count": ":,", "total_paid": ":$,.0f"},
        )
        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text="Claim Type Composition", font=dict(size=14)),
            height=400,
            coloraxis_colorbar=dict(title="Total Paid ($)"),
        )
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Claims: %{value:,}<br>Total Paid: $%{color:,.0f}<extra></extra>",
        )
        return fig

    # ------------------------------------------------------------------
    # Download CSV
    # ------------------------------------------------------------------
    @app.callback(
        Output("trends-download", "data"),
        Input("trends-btn", "n_clicks"),
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
            "claim_id", "incident_date", "claim_type", "severity_level",
            "paid_amount", "incurred_amount", "region", "state_name",
        ]
        cols = [c for c in export_cols if c in df.columns]
        return dcc.send_data_frame(df[cols].to_csv, "loss_trends.csv", index=False)
