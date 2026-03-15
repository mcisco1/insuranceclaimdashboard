"""Data Quality Dashboard page -- data provenance, completeness metrics, distributions.

Route: /data-quality
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html

from dashboard.components import create_kpi_card, create_info_tooltip

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
PRIMARY = "#0f3460"
SECONDARY = "#16213e"
ACCENT = "#e94560"
SUCCESS = "#00b894"
WARNING = "#fdcb6e"

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

def _fmt_number(val: float) -> str:
    if abs(val) >= 1_000_000:
        return f"{val / 1_000_000:,.1f}M"
    if abs(val) >= 1_000:
        return f"{val / 1_000:,.1f}K"
    return f"{val:,.0f}"


# ═══════════════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════════════

def layout(data_store) -> html.Div:
    """Return the Data Quality page layout."""
    df = data_store.claims_df

    total_records = len(df)
    total_cells = total_records * len(df.columns)
    null_cells = int(df.isnull().sum().sum())
    completeness = ((total_cells - null_cells) / total_cells * 100) if total_cells > 0 else 100

    # Data freshness
    for date_col in ("incident_date", "report_date"):
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            latest = dates.max()
            if pd.notna(latest):
                freshness = f"{latest.strftime('%Y-%m-%d')}"
                break
    else:
        freshness = "N/A"

    # KPIs
    kpi_records = create_kpi_card("Total Records", _fmt_number(total_records))
    kpi_completeness = create_kpi_card(
        "Overall Completeness",
        f"{completeness:.1f}%",
        delta_color=SUCCESS if completeness >= 95 else WARNING,
    )
    kpi_freshness = create_kpi_card("Data Freshness", freshness)
    kpi_columns = create_kpi_card("Columns", str(len(df.columns)))

    # Column null rates
    null_counts = df.isnull().sum().sort_values(ascending=True)
    null_pct = (null_counts / total_records * 100).round(2)
    non_zero_nulls = null_pct[null_pct > 0].sort_values(ascending=True)

    null_bar_fig = go.Figure()
    if not non_zero_nulls.empty:
        colors = [ACCENT if v > 10 else WARNING if v > 5 else SUCCESS for v in non_zero_nulls.values]
        null_bar_fig.add_trace(
            go.Bar(
                x=non_zero_nulls.values,
                y=non_zero_nulls.index,
                orientation="h",
                marker_color=colors,
                hovertemplate="<b>%{y}</b><br>Null Rate: %{x:.1f}%<extra></extra>",
            )
        )
    null_bar_fig.update_layout(
        **_COMMON_LAYOUT,
        title=dict(text="Column Null Rates (%)", font=dict(size=14)),
        xaxis_title="Null %",
        yaxis_title="",
        showlegend=False,
        height=max(300, len(non_zero_nulls) * 25 + 100),
    )

    # Paid amount distribution
    paid = pd.to_numeric(df.get("paid_amount", pd.Series(dtype=float)), errors="coerce").dropna()
    paid_fig = go.Figure()
    if not paid.empty:
        p99 = paid.quantile(0.99)
        paid_capped = paid.clip(upper=p99)
        paid_fig.add_trace(
            go.Histogram(
                x=paid_capped,
                nbinsx=50,
                marker_color=PRIMARY,
                opacity=0.85,
                hovertemplate="$%{x:,.0f}<br>Count: %{y:,}<extra></extra>",
            )
        )
        paid_fig.add_vline(
            x=paid.median(),
            line_dash="dash",
            line_color=ACCENT,
            annotation_text=f"Median: ${paid.median():,.0f}",
            annotation_position="top right",
            annotation_font_size=10,
        )
    paid_fig.update_layout(
        **_COMMON_LAYOUT,
        title=dict(text="Paid Amount Distribution (capped at 99th pctl)", font=dict(size=14)),
        xaxis_title="Paid Amount ($)",
        xaxis=dict(tickformat="$,.0f"),
        yaxis_title="Count",
        showlegend=False,
        height=380,
    )

    # Column profile table
    profile_rows = []
    for col in df.columns:
        col_data = df[col]
        null_count = int(col_data.isnull().sum())
        unique_count = int(col_data.nunique())
        dtype_str = str(col_data.dtype)

        row = {
            "Column": col,
            "Type": dtype_str,
            "Nulls": null_count,
            "Unique": unique_count,
        }

        numeric = pd.to_numeric(col_data, errors="coerce")
        if numeric.notna().sum() > 0 and numeric.notna().sum() > len(col_data) * 0.5:
            row["Min"] = f"{numeric.min():,.2f}"
            row["Max"] = f"{numeric.max():,.2f}"
        else:
            row["Min"] = "N/A"
            row["Max"] = "N/A"

        profile_rows.append(row)

    profile_df = pd.DataFrame(profile_rows)

    return html.Div(
        [
            html.H2(
                [
                    "Data Quality",
                    create_info_tooltip(
                        "Data provenance and quality metrics. Shows completeness, "
                        "null rates, and distributions for the claims warehouse.",
                        "dq-tooltip",
                    ),
                ],
                className="page-title",
            ),

            # KPI row
            html.Div(
                [kpi_records, kpi_completeness, kpi_freshness, kpi_columns],
                className="kpi-row",
            ),

            # Row 2: Null rates | Paid amount distribution
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            id="dq-null-bar",
                            figure=null_bar_fig,
                            config={"displayModeBar": False},
                        ),
                        className="card half-card",
                    ),
                    html.Div(
                        dcc.Graph(
                            id="dq-paid-dist",
                            figure=paid_fig,
                            config={"displayModeBar": False},
                        ),
                        className="card half-card",
                    ),
                ],
                className="chart-row",
            ),

            # Row 3: Column profile table
            html.Div(
                html.Div(
                    [
                        html.H4("Column Profile", className="table-title"),
                        dash_table.DataTable(
                            id="dq-profile-table",
                            columns=[{"name": c, "id": c} for c in profile_df.columns],
                            data=profile_df.to_dict("records"),
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
                                "minWidth": "80px",
                            },
                            style_data={
                                "backgroundColor": "rgba(0,0,0,0)",
                            },
                            style_data_conditional=[
                                {
                                    "if": {
                                        "filter_query": "{Nulls} > 0",
                                        "column_id": "Nulls",
                                    },
                                    "color": ACCENT,
                                    "fontWeight": "bold",
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

            # Pipeline metadata card
            html.Div(
                html.Div(
                    [
                        html.H4("Pipeline Metadata", className="table-title"),
                        html.Div(
                            [
                                html.P(f"Total records: {total_records:,}", style={"fontSize": "13px"}),
                                html.P(f"Total columns: {len(df.columns)}", style={"fontSize": "13px"}),
                                html.P(f"Total null cells: {null_cells:,}", style={"fontSize": "13px"}),
                                html.P(f"Overall completeness: {completeness:.2f}%", style={"fontSize": "13px"}),
                                html.P(f"Latest data point: {freshness}", style={"fontSize": "13px"}),
                            ],
                            style={"padding": "8px 0"},
                        ),
                    ],
                    className="card full-card",
                    style={"padding": "24px"},
                ),
                className="chart-row",
            ),
        ],
        className="page-container",
    )
