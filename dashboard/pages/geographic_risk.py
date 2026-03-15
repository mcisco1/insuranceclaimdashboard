"""Geographic & Department Risk page -- maps, heatmaps, regional analysis.

Route: /geographic-risk
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


# ═══════════════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════════════

def layout(data_store) -> html.Div:
    """Return the Geographic & Department Risk page layout."""
    df = data_store.claims_df

    years = sorted(df["accident_year"].dropna().unique())
    claim_types = sorted(df["claim_type"].dropna().unique())
    severities = sorted(df["severity_level"].dropna().unique())
    regions = sorted(df["region"].dropna().unique())

    return html.Div(
        [
            html.H2("Geographic & Department Risk", className="page-title"),
            create_filter_panel(years, claim_types, severities, regions),

            # Hidden store for map click -> cross-filter
            dcc.Store(id="geo-selected-state", data=None),

            # Row 1: US choropleth map (full width)
            html.Div(
                html.Div(
                    dcc.Loading(dcc.Graph(id="geo-choropleth", config={"displayModeBar": False}), type="circle"),
                    className="card full-card",
                ),
                className="chart-row",
            ),

            # Row 2: Department severity heatmap | Region x claim_type heatmap
            html.Div(
                [
                    html.Div(
                        dcc.Loading(dcc.Graph(id="geo-dept-heatmap", config={"displayModeBar": False}), type="circle"),
                        className="card half-card",
                    ),
                    html.Div(
                        dcc.Loading(dcc.Graph(id="geo-region-type-heatmap", config={"displayModeBar": False}), type="circle"),
                        className="card half-card",
                    ),
                ],
                className="chart-row",
            ),

            # Row 3: Top 10 states | Urban vs Rural comparison
            html.Div(
                [
                    html.Div(
                        dcc.Loading(dcc.Graph(id="geo-top-states", config={"displayModeBar": False}), type="circle"),
                        className="card half-card",
                    ),
                    html.Div(
                        dcc.Loading(dcc.Graph(id="geo-urban-rural", config={"displayModeBar": False}), type="circle"),
                        className="card half-card",
                    ),
                ],
                className="chart-row",
            ),

            create_download_button("geo"),
        ],
        className="page-container",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════

def register_callbacks(app, data_store) -> None:
    """Register all Geographic Risk callbacks."""

    _filter_inputs = [
        Input("filter-year", "value"),
        Input("filter-claim-type", "value"),
        Input("filter-severity", "value"),
        Input("filter-region", "value"),
    ]

    # ------------------------------------------------------------------
    # Capture map click -> store selected state
    # ------------------------------------------------------------------
    @app.callback(
        Output("geo-selected-state", "data"),
        Input("geo-choropleth", "clickData"),
        prevent_initial_call=True,
    )
    def _capture_state_click(click_data):
        if click_data and click_data.get("points"):
            location = click_data["points"][0].get("location")
            return location
        return None

    # ------------------------------------------------------------------
    # US Choropleth map
    # ------------------------------------------------------------------
    @app.callback(Output("geo-choropleth", "figure"), _filter_inputs)
    def _choropleth(years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        if df.empty:
            return go.Figure().update_layout(**_COMMON_LAYOUT, title="Claims by State")

        state_agg = df.groupby("state_code").agg(
            total_paid=("paid_amount", "sum"),
            claim_count=("claim_id", "count"),
            avg_severity=("severity_level", "mean"),
        ).reset_index()
        state_agg["avg_severity"] = state_agg["avg_severity"].round(2)

        fig = go.Figure(
            go.Choropleth(
                locations=state_agg["state_code"],
                z=state_agg["total_paid"],
                locationmode="USA-states",
                colorscale=[
                    [0.0, "#dfe6e9"],
                    [0.25, "#74b9ff"],
                    [0.5, WARNING],
                    [0.75, "#e17055"],
                    [1.0, ACCENT],
                ],
                colorbar=dict(
                    title="Total Paid ($)",
                    tickformat="$,.0f",
                ),
                customdata=np.stack(
                    [state_agg["claim_count"], state_agg["avg_severity"]], axis=-1
                ),
                hovertemplate=(
                    "<b>%{location}</b><br>"
                    "Total Paid: $%{z:,.0f}<br>"
                    "Claims: %{customdata[0]:,}<br>"
                    "Avg Severity: %{customdata[1]:.2f}"
                    "<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text="Total Paid Amount by State", font=dict(size=14)),
            geo=dict(
                scope="usa",
                bgcolor="rgba(0,0,0,0)",
                lakecolor="rgba(0,0,0,0)",
                landcolor="rgba(240,240,240,0.5)",
                showlakes=True,
            ),
            height=480,
        )
        return fig

    # ------------------------------------------------------------------
    # Department severity heatmap
    # ------------------------------------------------------------------
    @app.callback(
        Output("geo-dept-heatmap", "figure"),
        _filter_inputs + [Input("geo-selected-state", "data")],
    )
    def _dept_heatmap(years, claim_types, severities, regions, selected_state):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        if selected_state:
            df = df[df["state_code"] == selected_state]
        if df.empty:
            return go.Figure().update_layout(**_COMMON_LAYOUT, title="Department x Severity")

        pivot = df.groupby(["department", "severity_level"]).size().unstack(fill_value=0)
        pivot = pivot.reindex(columns=sorted(pivot.columns))

        fig = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=[f"Level {int(c)}" for c in pivot.columns],
                y=pivot.index,
                colorscale=[
                    [0, "#dfe6e9"],
                    [0.5, WARNING],
                    [1, ACCENT],
                ],
                hovertemplate=(
                    "<b>%{y}</b> | %{x}<br>"
                    "Claims: %{z:,}<extra></extra>"
                ),
                colorbar=dict(title="Claims"),
            )
        )
        title_suffix = f" ({selected_state})" if selected_state else ""
        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text=f"Department x Severity Heatmap{title_suffix}", font=dict(size=14)),
            xaxis_title="Severity Level",
            yaxis_title="",
            height=420,
        )
        return fig

    # ------------------------------------------------------------------
    # Region x Claim Type heatmap
    # ------------------------------------------------------------------
    @app.callback(
        Output("geo-region-type-heatmap", "figure"),
        _filter_inputs + [Input("geo-selected-state", "data")],
    )
    def _region_type_heatmap(years, claim_types, severities, regions, selected_state):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        if selected_state:
            df = df[df["state_code"] == selected_state]
        if df.empty:
            return go.Figure().update_layout(**_COMMON_LAYOUT, title="Region x Claim Type")

        pivot = df.groupby(["region", "claim_type"]).size().unstack(fill_value=0)

        fig = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[
                    [0, "#dfe6e9"],
                    [0.5, "#74b9ff"],
                    [1, PRIMARY],
                ],
                hovertemplate=(
                    "<b>%{y}</b> | %{x}<br>"
                    "Claims: %{z:,}<extra></extra>"
                ),
                colorbar=dict(title="Claims"),
            )
        )
        title_suffix = f" ({selected_state})" if selected_state else ""
        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text=f"Region x Claim Type Heatmap{title_suffix}", font=dict(size=14)),
            xaxis_title="Claim Type",
            yaxis_title="",
            height=420,
        )
        return fig

    # ------------------------------------------------------------------
    # Top 10 states by claim volume
    # ------------------------------------------------------------------
    @app.callback(Output("geo-top-states", "figure"), _filter_inputs)
    def _top_states(years, claim_types, severities, regions):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        if df.empty:
            return go.Figure().update_layout(**_COMMON_LAYOUT, title="Top 10 States")

        state_counts = (
            df.groupby("state_code")
            .agg(count=("claim_id", "count"), total_paid=("paid_amount", "sum"))
            .sort_values("count", ascending=False)
            .head(10)
            .sort_values("count", ascending=True)
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=state_counts["count"],
                y=state_counts.index,
                orientation="h",
                marker_color=PRIMARY,
                customdata=state_counts["total_paid"],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Claims: %{x:,}<br>"
                    "Total Paid: $%{customdata:,.0f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text="Top 10 States by Claim Volume", font=dict(size=14)),
            xaxis_title="Claim Count",
            yaxis_title="",
            showlegend=False,
            height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # Urban vs Rural comparison
    # ------------------------------------------------------------------
    @app.callback(
        Output("geo-urban-rural", "figure"),
        _filter_inputs + [Input("geo-selected-state", "data")],
    )
    def _urban_rural(years, claim_types, severities, regions, selected_state):
        df = _apply_filters(data_store.claims_df, years, claim_types, severities, regions)
        if selected_state:
            df = df[df["state_code"] == selected_state]
        if df.empty or "urban_rural" not in df.columns:
            return go.Figure().update_layout(**_COMMON_LAYOUT, title="Urban vs Rural")

        agg = df.groupby("urban_rural").agg(
            avg_severity=("severity_level", "mean"),
            avg_paid=("paid_amount", "mean"),
            claim_count=("claim_id", "count"),
        ).reset_index()
        agg["avg_severity"] = agg["avg_severity"].round(2)
        agg["avg_paid"] = agg["avg_paid"].round(0)

        categories = agg["urban_rural"].tolist()
        bar_colors = [PRIMARY, ACCENT] if len(categories) >= 2 else [PRIMARY]

        fig = go.Figure()

        # Avg severity bars
        fig.add_trace(
            go.Bar(
                x=agg["urban_rural"],
                y=agg["avg_severity"],
                name="Avg Severity",
                marker_color=PRIMARY,
                yaxis="y",
                hovertemplate="<b>%{x}</b><br>Avg Severity: %{y:.2f}<extra></extra>",
            )
        )

        # Avg paid bars
        fig.add_trace(
            go.Bar(
                x=agg["urban_rural"],
                y=agg["avg_paid"],
                name="Avg Paid ($)",
                marker_color=ACCENT,
                yaxis="y2",
                hovertemplate="<b>%{x}</b><br>Avg Paid: $%{y:,.0f}<extra></extra>",
            )
        )

        title_suffix = f" ({selected_state})" if selected_state else ""
        fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text=f"Urban vs Rural Comparison{title_suffix}", font=dict(size=14)),
            xaxis_title="",
            yaxis=dict(title="Avg Severity", side="left", showgrid=False),
            yaxis2=dict(
                title="Avg Paid ($)",
                side="right",
                overlaying="y",
                showgrid=False,
                tickformat="$,.0f",
            ),
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # Download CSV
    # ------------------------------------------------------------------
    @app.callback(
        Output("geo-download", "data"),
        Input("geo-btn", "n_clicks"),
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
            "claim_id", "incident_date", "state_code", "state_name", "region",
            "urban_rural", "department", "claim_type", "severity_level",
            "paid_amount", "incurred_amount",
        ]
        cols = [c for c in export_cols if c in df.columns]
        return dcc.send_data_frame(df[cols].to_csv, "geographic_risk.csv", index=False)
