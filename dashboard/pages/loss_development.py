"""Loss Development Triangle & IBNR Analysis page.

Route: /loss-development
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from dashboard.components import create_download_button, create_kpi_card, create_info_tooltip

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


def _build_triangle_df(triangle_dict: Dict) -> pd.DataFrame:
    """Convert the triangle dict (accident_year -> {dev_year: value}) to a DataFrame."""
    if not triangle_dict:
        return pd.DataFrame()
    df = pd.DataFrame(triangle_dict).T
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    df = df.sort_index(axis=0).sort_index(axis=1)
    return df


def _get_current_diagonal(triangle_df: pd.DataFrame) -> Dict[int, float]:
    """Extract the latest known cumulative paid for each accident year (diagonal)."""
    diagonal = {}
    for ay in triangle_df.index:
        row = triangle_df.loc[ay].dropna()
        if not row.empty:
            diagonal[ay] = float(row.iloc[-1])
        else:
            diagonal[ay] = 0.0
    return diagonal


# ═══════════════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════════════

def layout(data_store) -> html.Div:
    """Return the Loss Development Triangle & IBNR Analysis page layout."""
    loss_tri = data_store.loss_tri

    triangle_dict = loss_tri.get("triangle", {})
    triangle_df = _build_triangle_df(triangle_dict)
    ultimate_losses = loss_tri.get("ultimate_losses", {})
    ibnr_reserves = loss_tri.get("ibnr_reserves", {})
    age_to_age = loss_tri.get("age_to_age_factors", {})
    mack_se = loss_tri.get("mack_standard_errors", {})

    # ── Build heatmap figure ──────────────────────────────────────────
    if not triangle_df.empty:
        acc_years = triangle_df.index.tolist()
        dev_years = triangle_df.columns.tolist()
        z_values = triangle_df.values.astype(float)

        # Build text annotations (formatted dollar amounts)
        text_values = []
        for row in z_values:
            text_row = []
            for val in row:
                if np.isnan(val):
                    text_row.append("")
                else:
                    text_row.append(_fmt_currency(val))
            text_values.append(text_row)

        heatmap_fig = go.Figure(
            go.Heatmap(
                z=z_values,
                x=[str(d) for d in dev_years],
                y=[str(a) for a in acc_years],
                text=text_values,
                texttemplate="%{text}",
                textfont=dict(size=11),
                colorscale=[
                    [0, "#dfe6e9"],
                    [0.3, "#74b9ff"],
                    [0.6, WARNING],
                    [1, ACCENT],
                ],
                hovertemplate=(
                    "<b>AY %{y} | DY %{x}</b><br>"
                    "Cumulative Paid: %{text}<extra></extra>"
                ),
                colorbar=dict(
                    title="Cumulative Paid ($)",
                    tickformat="$,.0f",
                ),
                connectgaps=False,
            )
        )
        heatmap_fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(
                text="Loss Development Triangle (Cumulative Paid)",
                font=dict(size=14),
            ),
            xaxis_title="Development Year",
            yaxis_title="Accident Year",
            yaxis=dict(autorange="reversed"),
            height=450,
        )
    else:
        heatmap_fig = go.Figure()
        heatmap_fig.update_layout(**_COMMON_LAYOUT, title="No triangle data available")

    # ── Age-to-age factors bar chart ──────────────────────────────────
    if age_to_age:
        factor_labels = list(age_to_age.keys())
        factor_values = [v if v is not None else 0 for v in age_to_age.values()]

        ata_fig = go.Figure(
            go.Bar(
                x=factor_labels,
                y=factor_values,
                marker_color=PRIMARY,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Factor: %{y:.4f}<extra></extra>"
                ),
            )
        )
        # Reference line at 1.0
        ata_fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color=ACCENT,
            annotation_text="1.0 (fully developed)",
            annotation_position="top right",
            annotation_font_size=10,
        )
        ata_fig.update_layout(
            **_COMMON_LAYOUT,
            title=dict(text="Age-to-Age Development Factors", font=dict(size=14)),
            xaxis_title="Development Period",
            yaxis_title="Link Factor",
            showlegend=False,
            height=400,
        )
    else:
        ata_fig = go.Figure()
        ata_fig.update_layout(**_COMMON_LAYOUT, title="No development factors available")

    # ── Ultimate vs Current Paid grouped bar ──────────────────────────
    if ultimate_losses and not triangle_df.empty:
        diagonal = _get_current_diagonal(triangle_df)
        bar_years = []
        current_vals = []
        ultimate_vals = []

        for ay_str, data in sorted(ultimate_losses.items()):
            ay = int(ay_str)
            current = data.get("current_paid", 0.0) or 0.0
            ultimate = data.get("ultimate") or current
            bar_years.append(str(ay))
            current_vals.append(current)
            ultimate_vals.append(ultimate)

        ultimate_bar_fig = go.Figure()
        ultimate_bar_fig.add_trace(
            go.Bar(
                x=bar_years,
                y=current_vals,
                name="Current Paid",
                marker_color=PRIMARY,
                hovertemplate="<b>AY %{x}</b><br>Current Paid: $%{y:,.0f}<extra></extra>",
            )
        )
        # Compute error bars from Mack SEs
        error_bars = []
        for ay_str in bar_years:
            mack_data = mack_se.get(ay_str, {})
            se = mack_data.get("se")
            error_bars.append(1.96 * se if se is not None else 0)

        ultimate_bar_fig.add_trace(
            go.Bar(
                x=bar_years,
                y=ultimate_vals,
                name="Ultimate Loss",
                marker_color=ACCENT,
                error_y=dict(type="data", array=error_bars, visible=True, color="#636e72"),
                hovertemplate="<b>AY %{x}</b><br>Ultimate: $%{y:,.0f}<extra></extra>",
            )
        )
        ultimate_bar_fig.update_layout(
            **_COMMON_LAYOUT,
            barmode="group",
            title=dict(text="Ultimate Losses vs Current Paid", font=dict(size=14)),
            xaxis_title="Accident Year",
            yaxis_title="Amount ($)",
            yaxis=dict(tickformat="$,.0f"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            height=400,
        )
    else:
        ultimate_bar_fig = go.Figure()
        ultimate_bar_fig.update_layout(**_COMMON_LAYOUT, title="No ultimate loss data available")

    # ── IBNR reserves table data ──────────────────────────────────────
    ibnr_rows = []
    if ibnr_reserves:
        for ay_str, data in sorted(ibnr_reserves.items()):
            current = data.get("current_paid", 0.0) or 0.0
            ultimate = data.get("ultimate")
            ibnr = data.get("ibnr")
            mack_data = mack_se.get(ay_str, {})
            se = mack_data.get("se")
            ci_lo = mack_data.get("ci_lower")
            ci_hi = mack_data.get("ci_upper")

            ibnr_rows.append({
                "Accident Year": ay_str.upper() if ay_str == "total" else ay_str,
                "Paid to Date ($)": f"${current:,.0f}",
                "Ultimate ($)": f"${ultimate:,.0f}" if ultimate is not None else "N/A",
                "IBNR Reserve ($)": f"${ibnr:,.0f}" if ibnr is not None else "N/A",
                "SE ($)": f"${se:,.0f}" if se is not None else "N/A",
                "95% CI ($)": f"${ci_lo:,.0f} - ${ci_hi:,.0f}" if ci_lo is not None and ci_hi is not None else "N/A",
            })

    ibnr_table_df = pd.DataFrame(ibnr_rows) if ibnr_rows else pd.DataFrame(
        columns=["Accident Year", "Paid to Date ($)", "Ultimate ($)", "IBNR Reserve ($)", "SE ($)", "95% CI ($)"]
    )

    return html.Div(
        [
            html.H2(
                [
                    "Loss Development Triangle & IBNR Analysis",
                    create_info_tooltip(
                        "Chain-ladder cumulative development factors. Mack (1993) standard errors "
                        "provide prediction intervals for IBNR reserves.",
                        "lossdev-tooltip",
                    ),
                ],
                className="page-title",
            ),

            # Row 1: Heatmap (full width)
            html.Div(
                html.Div(
                    dcc.Graph(
                        id="lossdev-heatmap",
                        figure=heatmap_fig,
                        config={"displayModeBar": False},
                    ),
                    className="card full-card",
                ),
                className="chart-row",
            ),

            # Row 2: ATA factors | Ultimate vs Current
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            id="lossdev-ata-bar",
                            figure=ata_fig,
                            config={"displayModeBar": False},
                        ),
                        className="card half-card",
                    ),
                    html.Div(
                        dcc.Graph(
                            id="lossdev-ultimate-bar",
                            figure=ultimate_bar_fig,
                            config={"displayModeBar": False},
                        ),
                        className="card half-card",
                    ),
                ],
                className="chart-row",
            ),

            # Row 3: IBNR Reserves table
            html.Div(
                html.Div(
                    [
                        html.H4("IBNR Reserves by Accident Year", className="table-title"),
                        dash_table.DataTable(
                            id="lossdev-ibnr-table",
                            columns=[{"name": c, "id": c} for c in ibnr_table_df.columns],
                            data=ibnr_table_df.to_dict("records"),
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
                                "minWidth": "140px",
                            },
                            style_data={
                                "backgroundColor": "rgba(0,0,0,0)",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"filter_query": '{Accident Year} = "TOTAL"'},
                                    "backgroundColor": "rgba(15, 52, 96, 0.08)",
                                    "fontWeight": "bold",
                                },
                                {
                                    "if": {
                                        "column_id": "IBNR Reserve ($)",
                                        "filter_query": '{Accident Year} != "TOTAL"',
                                    },
                                    "color": ACCENT,
                                    "fontWeight": "600",
                                },
                                {
                                    "if": {
                                        "column_id": "Ultimate ($)",
                                        "filter_query": '{Accident Year} != "TOTAL"',
                                    },
                                    "color": PRIMARY,
                                    "fontWeight": "600",
                                },
                            ],
                            sort_action="native",
                            page_size=20,
                        ),
                    ],
                    className="card full-card",
                ),
                className="chart-row",
            ),

            # Download button
            create_download_button("lossdev-triangle"),
        ],
        className="page-container",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════

def register_callbacks(app, data_store) -> None:
    """Register Loss Development page callbacks."""

    # ------------------------------------------------------------------
    # Download triangle CSV
    # ------------------------------------------------------------------
    @app.callback(
        Output("lossdev-triangle-download", "data"),
        Input("lossdev-triangle-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _download_triangle(n_clicks):
        if not n_clicks:
            raise PreventUpdate

        triangle_dict = data_store.loss_tri.get("triangle", {})
        triangle_df = _build_triangle_df(triangle_dict)

        if triangle_df.empty:
            raise PreventUpdate

        # Reset index for export
        export_df = triangle_df.copy()
        export_df.index.name = "accident_year"
        export_df.columns = [f"dev_year_{c}" for c in export_df.columns]
        export_df = export_df.reset_index()

        return dcc.send_data_frame(
            export_df.to_csv, "loss_development_triangle.csv", index=False
        )
