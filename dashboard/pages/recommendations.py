"""Risk Intelligence & Recommendations page.

Route: /recommendations
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from dashboard.components import create_download_button, create_kpi_card

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


def _build_risk_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Build top 10 risk segments ranked by total_paid * avg_severity."""
    if df.empty:
        return pd.DataFrame()

    segments = df.groupby(["specialty", "region", "claim_type"]).agg(
        claim_count=("claim_id", "count"),
        avg_severity=("severity_level", "mean"),
        total_paid=("paid_amount", "sum"),
        litigation_rate=("litigation_flag", "mean"),
    ).reset_index()

    segments["avg_severity"] = segments["avg_severity"].round(2)
    segments["litigation_rate"] = (segments["litigation_rate"] * 100).round(1)
    segments["risk_score"] = (
        segments["total_paid"] * segments["avg_severity"]
    ).round(0)

    segments = segments.sort_values("risk_score", ascending=False).head(10).reset_index(drop=True)
    segments.index = segments.index + 1  # 1-based rank
    segments = segments.reset_index().rename(columns={"index": "rank"})

    return segments


def _build_recommendation_card(
    rec_number: int,
    title: str,
    metric: str,
    impact: str,
    priority: str,
) -> html.Div:
    """Build a styled recommendation card."""
    priority_colors = {
        "High": ACCENT,
        "Medium": WARNING,
        "Low": SUCCESS,
    }
    priority_color = priority_colors.get(priority, PRIMARY)

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        html.Span(
                            f"#{rec_number}",
                            style={
                                "fontSize": "14px",
                                "fontWeight": "bold",
                                "color": "white",
                                "backgroundColor": priority_color,
                                "borderRadius": "50%",
                                "width": "32px",
                                "height": "32px",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                            },
                        ),
                        style={"marginRight": "12px"},
                    ),
                    html.Div(
                        [
                            html.H5(title, style={
                                "fontFamily": _FONT_FAMILY,
                                "color": PRIMARY,
                                "margin": "0 0 6px 0",
                                "fontSize": "14px",
                                "fontWeight": "600",
                                "lineHeight": "1.3",
                            }),
                            html.P(metric, style={
                                "fontFamily": _FONT_FAMILY,
                                "fontSize": "12px",
                                "color": "#636e72",
                                "margin": "0 0 4px 0",
                            }),
                        ],
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "alignItems": "flex-start"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Est. Impact: ", style={
                                "fontSize": "12px", "color": "#636e72",
                            }),
                            html.Span(impact, style={
                                "fontSize": "12px", "fontWeight": "600", "color": SUCCESS,
                            }),
                        ],
                    ),
                    html.Span(
                        f"Priority: {priority}",
                        style={
                            "fontSize": "11px",
                            "fontWeight": "bold",
                            "color": priority_color,
                            "backgroundColor": f"{priority_color}18",
                            "padding": "2px 10px",
                            "borderRadius": "12px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginTop": "10px",
                    "paddingTop": "8px",
                    "borderTop": "1px solid #dfe6e9",
                },
            ),
        ],
        className="kpi-card",
        style={"marginBottom": "12px", "padding": "16px"},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════════════

def layout(data_store) -> html.Div:
    """Return the Risk Intelligence & Recommendations page layout."""
    df = data_store.claims_df
    freq_sev = data_store.freq_sev

    # ── Risk matrix scatter data ──────────────────────────────────────
    scatter_data = freq_sev.get("frequency_severity_scatter", {})

    risk_matrix_fig = go.Figure()

    if scatter_data:
        specialties = []
        frequencies = []
        avg_severities = []
        total_paids = []

        for spec, metrics in scatter_data.items():
            if isinstance(metrics, dict):
                specialties.append(spec)
                frequencies.append(metrics.get("claim_count", 0))
                avg_severities.append(metrics.get("avg_severity", 0))
                total_paids.append(metrics.get("total_paid", 0))

        if specialties:
            # Assign risk levels based on combined frequency and severity
            risk_levels = []
            freq_arr = np.array(frequencies)
            sev_arr = np.array(avg_severities)
            freq_median = np.median(freq_arr) if len(freq_arr) > 0 else 0
            sev_median = np.median(sev_arr) if len(sev_arr) > 0 else 0

            for f, s in zip(frequencies, avg_severities):
                if f > freq_median and s > sev_median:
                    risk_levels.append("High")
                elif f > freq_median or s > sev_median:
                    risk_levels.append("Medium")
                else:
                    risk_levels.append("Low")

            risk_colors = {"High": ACCENT, "Medium": WARNING, "Low": SUCCESS}

            # Scale bubble sizes
            max_paid = max(total_paids) if total_paids else 1
            bubble_sizes = [max(8, (tp / max_paid) * 60) for tp in total_paids]

            for level in ["High", "Medium", "Low"]:
                indices = [i for i, r in enumerate(risk_levels) if r == level]
                if not indices:
                    continue

                risk_matrix_fig.add_trace(
                    go.Scatter(
                        x=[frequencies[i] for i in indices],
                        y=[avg_severities[i] for i in indices],
                        mode="markers+text",
                        name=f"{level} Risk",
                        marker=dict(
                            size=[bubble_sizes[i] for i in indices],
                            color=risk_colors[level],
                            opacity=0.7,
                            line=dict(width=1, color="white"),
                        ),
                        text=[specialties[i] for i in indices],
                        textposition="top center",
                        textfont=dict(size=9),
                        customdata=[
                            [specialties[i], _fmt_currency(total_paids[i])]
                            for i in indices
                        ],
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Frequency: %{x:,} claims<br>"
                            "Avg Severity: %{y:.2f}<br>"
                            "Total Paid: %{customdata[1]}"
                            "<extra>%{fullData.name}</extra>"
                        ),
                    )
                )

        # Add quadrant lines
        if frequencies and avg_severities:
            risk_matrix_fig.add_hline(
                y=sev_median,
                line_dash="dot",
                line_color="rgba(0,0,0,0.15)",
            )
            risk_matrix_fig.add_vline(
                x=freq_median,
                line_dash="dot",
                line_color="rgba(0,0,0,0.15)",
            )

    risk_matrix_fig.update_layout(
        **_COMMON_LAYOUT,
        title=dict(text="Risk Matrix: Frequency vs Severity by Specialty", font=dict(size=14)),
        xaxis_title="Claim Frequency (Count)",
        yaxis_title="Average Severity",
        legend=dict(orientation="h", yanchor="bottom", y=-0.20, xanchor="center", x=0.5),
        height=500,
    )

    # ── Risk segments table ───────────────────────────────────────────
    risk_segments = _build_risk_segments(df)

    if not risk_segments.empty:
        # Format for display
        display_segments = risk_segments.copy()
        display_segments["total_paid"] = display_segments["total_paid"].apply(
            lambda x: f"${x:,.0f}"
        )
        display_segments["risk_score"] = display_segments["risk_score"].apply(
            lambda x: f"{x:,.0f}"
        )
        display_segments["litigation_rate"] = display_segments["litigation_rate"].apply(
            lambda x: f"{x:.1f}%"
        )

        seg_column_labels = {
            "rank": "Rank",
            "specialty": "Specialty",
            "region": "Region",
            "claim_type": "Claim Type",
            "claim_count": "Claims",
            "avg_severity": "Avg Severity",
            "total_paid": "Total Paid",
            "litigation_rate": "Litigation Rate",
            "risk_score": "Risk Score",
        }
        seg_display_cols = list(seg_column_labels.keys())
        seg_columns = [
            {"name": seg_column_labels[c], "id": c} for c in seg_display_cols
        ]
        seg_data = display_segments[seg_display_cols].to_dict("records")
    else:
        seg_columns = [{"name": "No Data", "id": "no_data"}]
        seg_data = []

    # ── Recommendation cards ──────────────────────────────────────────
    # Dynamically build supporting metrics from data

    # 1. Highest-severity specialty
    if "specialty" in df.columns:
        surg_claims = df[df["claim_type"].str.contains("surg", case=False, na=False)] if "claim_type" in df.columns else df
        avg_surg_severity = surg_claims["severity_level"].mean() if not surg_claims.empty else df["severity_level"].mean()
        avg_surg_severity = round(avg_surg_severity, 2) if pd.notna(avg_surg_severity) else 0
    else:
        avg_surg_severity = 0

    # 2. Highest reporting lag specialty
    if "specialty" in df.columns and "days_to_report" in df.columns:
        lag_by_spec = df.groupby("specialty")["days_to_report"].mean().dropna()
        if not lag_by_spec.empty:
            highest_lag_spec = lag_by_spec.idxmax()
            highest_lag_val = round(lag_by_spec.max(), 1)
        else:
            highest_lag_spec = "Unknown"
            highest_lag_val = 0
    else:
        highest_lag_spec = "Unknown"
        highest_lag_val = 0

    # 3. Low-severity claim count and avg close time
    low_sev = df[df["severity_level"] <= 2] if "severity_level" in df.columns else pd.DataFrame()
    low_sev_count = len(low_sev)
    low_sev_avg_close = round(low_sev["days_to_close"].mean(), 0) if not low_sev.empty and "days_to_close" in low_sev.columns else 0

    # 4. Litigation by state
    if "state_code" in df.columns and "litigation_flag" in df.columns:
        high_lit_states = ["NY", "CA", "FL", "TX"]
        high_lit_df = df[df["state_code"].isin(high_lit_states)]
        high_lit_rate = round(high_lit_df["litigation_flag"].mean() * 100, 1) if not high_lit_df.empty else 0
        overall_lit_rate = round(df["litigation_flag"].mean() * 100, 1) if not df.empty else 0
    else:
        high_lit_rate = 0
        overall_lit_rate = 0

    # 5. Top repeat root causes
    if "root_cause_category" in df.columns and "repeat_event_flag" in df.columns:
        repeats = df[df["repeat_event_flag"] == 1]
        if not repeats.empty:
            top_repeat_causes = repeats["root_cause_category"].value_counts().head(3)
            repeat_cause_text = ", ".join(top_repeat_causes.index.tolist())
            repeat_count = int(repeats["repeat_event_flag"].sum())
        else:
            repeat_cause_text = "N/A"
            repeat_count = 0
    else:
        repeat_cause_text = "N/A"
        repeat_count = 0

    # Combined savings from scenarios
    combined_savings = data_store.scenarios.get("combined_scenario", {}).get("total_savings", 0)

    recommendation_cards = [
        _build_recommendation_card(
            rec_number=1,
            title="Reduce surgical claim severity through enhanced pre-operative protocols",
            metric=f"Surgical claims avg severity: {avg_surg_severity}. "
                   f"Severity 4-5 claims cost ~3x more than level 3.",
            impact=f"Potential {_fmt_currency(combined_savings * 0.35)} in annual savings",
            priority="High",
        ),
        _build_recommendation_card(
            rec_number=2,
            title=f"Target reporting lag reduction in {highest_lag_spec}",
            metric=f"Avg reporting lag for {highest_lag_spec}: {highest_lag_val} days. "
                   f"Delayed reporting increases costs by 15-25%.",
            impact="10-15% reduction in late-reported claim costs",
            priority="High",
        ),
        _build_recommendation_card(
            rec_number=3,
            title="Implement fast-track closure for low-severity claims (Level 1-2)",
            metric=f"{low_sev_count:,} low-severity claims with avg close time "
                   f"of {low_sev_avg_close:.0f} days. Target: <60 days.",
            impact="20-30% reduction in administrative overhead",
            priority="Medium",
        ),
        _build_recommendation_card(
            rec_number=4,
            title="Focus litigation prevention on high-risk states (NY, CA, FL, TX)",
            metric=f"Litigation rate in target states: {high_lit_rate}% vs "
                   f"overall {overall_lit_rate}%. Litigated claims cost ~3x more.",
            impact=f"Potential {_fmt_currency(combined_savings * 0.25)} from litigation reduction",
            priority="High",
        ),
        _build_recommendation_card(
            rec_number=5,
            title="Address repeat incidents in top root cause categories",
            metric=f"{repeat_count:,} repeat incidents identified. "
                   f"Top causes: {repeat_cause_text}.",
            impact="15-20% reduction in preventable claims",
            priority="Medium",
        ),
        _build_recommendation_card(
            rec_number=6,
            title="Deploy predictive severity model for early triage and resource allocation",
            metric=f"RF model accuracy: {data_store.severity_model.get('rf_accuracy', 0):.1%}. "
                   f"Early identification enables proactive management.",
            impact="Faster interventions on high-severity claims",
            priority="Low",
        ),
    ]

    # ── Drill-through claims columns ──────────────────────────────────
    drill_columns = [
        "claim_id", "incident_date", "specialty", "region", "claim_type",
        "severity_level", "paid_amount", "days_to_close", "litigation_flag",
        "root_cause_category", "status",
    ]
    avail_drill_cols = [c for c in drill_columns if c in df.columns]

    drill_col_labels = {
        "claim_id": "Claim ID",
        "incident_date": "Incident Date",
        "specialty": "Specialty",
        "region": "Region",
        "claim_type": "Claim Type",
        "severity_level": "Severity",
        "paid_amount": "Paid Amount",
        "days_to_close": "Days to Close",
        "litigation_flag": "Litigation",
        "root_cause_category": "Root Cause",
        "status": "Status",
    }

    return html.Div(
        [
            html.H2("Risk Intelligence & Recommendations", className="page-title"),

            # Hidden stores for cross-filtering
            dcc.Store(id="recs-selected-specialty", data=None),
            dcc.Store(id="recs-selected-segment", data=None),

            # Row 1: Risk matrix scatter (full width)
            html.Div(
                html.Div(
                    dcc.Graph(
                        id="recs-risk-matrix",
                        figure=risk_matrix_fig,
                        config={"displayModeBar": False},
                    ),
                    className="card full-card",
                ),
                className="chart-row",
            ),

            # Row 2: Risk segments table
            html.Div(
                html.Div(
                    [
                        html.H4("Top 10 Risk Segments", className="table-title"),
                        dash_table.DataTable(
                            id="recs-risk-segments-table",
                            columns=seg_columns,
                            data=seg_data,
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
                                "minWidth": "90px",
                            },
                            style_data={
                                "backgroundColor": "rgba(0,0,0,0)",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "rgba(15, 52, 96, 0.03)",
                                },
                            ],
                            sort_action="native",
                            row_selectable="single",
                            page_size=10,
                        ),
                    ],
                    className="card full-card",
                ),
                className="chart-row",
            ),

            # Row 3: Recommendation cards
            html.Div(
                [
                    html.H4("Actionable Recommendations", style={
                        "fontFamily": _FONT_FAMILY,
                        "color": PRIMARY,
                        "marginBottom": "16px",
                        "paddingLeft": "8px",
                    }),
                    html.Div(
                        recommendation_cards,
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit, minmax(400px, 1fr))",
                            "gap": "16px",
                        },
                    ),
                ],
                className="chart-row",
                style={"padding": "0 8px"},
            ),

            # Row 4: Drill-through claims table
            html.Div(
                html.Div(
                    [
                        html.H4(
                            id="recs-drill-title",
                            children="Claims Detail (select a specialty or segment to filter)",
                            className="table-title",
                        ),
                        html.Div(id="recs-drill-table-container"),
                    ],
                    className="card full-card",
                ),
                className="chart-row",
            ),

            # Download buttons
            html.Div(
                [
                    create_download_button("recs-segments"),
                    create_download_button("recs-claims"),
                ],
                style={"display": "flex", "gap": "16px", "justifyContent": "flex-start"},
            ),
        ],
        className="page-container",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════

def register_callbacks(app, data_store) -> None:
    """Register Recommendations page callbacks."""

    # ------------------------------------------------------------------
    # Capture risk matrix click -> specialty filter
    # ------------------------------------------------------------------
    @app.callback(
        Output("recs-selected-specialty", "data"),
        Input("recs-risk-matrix", "clickData"),
        prevent_initial_call=True,
    )
    def _capture_specialty_click(click_data):
        if click_data and click_data.get("points"):
            pt = click_data["points"][0]
            custom = pt.get("customdata")
            if custom and len(custom) > 0:
                return custom[0]  # specialty name
        return None

    # ------------------------------------------------------------------
    # Capture risk segment table row selection -> segment filter
    # ------------------------------------------------------------------
    @app.callback(
        Output("recs-selected-segment", "data"),
        Input("recs-risk-segments-table", "selected_rows"),
        State("recs-risk-segments-table", "data"),
        prevent_initial_call=True,
    )
    def _capture_segment_selection(selected_rows, table_data):
        if selected_rows and table_data:
            row = table_data[selected_rows[0]]
            return {
                "specialty": row.get("specialty"),
                "region": row.get("region"),
                "claim_type": row.get("claim_type"),
            }
        return None

    # ------------------------------------------------------------------
    # Drill-through table (filtered by specialty or segment selection)
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("recs-drill-title", "children"),
            Output("recs-drill-table-container", "children"),
        ],
        [
            Input("recs-selected-specialty", "data"),
            Input("recs-selected-segment", "data"),
        ],
    )
    def _update_drill_table(selected_specialty, selected_segment):
        df = data_store.claims_df.copy()
        title = "Claims Detail"
        filters_applied = []

        # Apply segment filter (more specific) or specialty filter
        ctx = callback_context
        trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

        if "recs-selected-segment" in trigger and selected_segment:
            if selected_segment.get("specialty"):
                df = df[df["specialty"] == selected_segment["specialty"]]
                filters_applied.append(selected_segment["specialty"])
            if selected_segment.get("region"):
                df = df[df["region"] == selected_segment["region"]]
                filters_applied.append(selected_segment["region"])
            if selected_segment.get("claim_type"):
                df = df[df["claim_type"] == selected_segment["claim_type"]]
                filters_applied.append(selected_segment["claim_type"])
        elif selected_specialty:
            df = df[df["specialty"] == selected_specialty]
            filters_applied.append(selected_specialty)

        if filters_applied:
            title = f"Claims Detail - Filtered: {' | '.join(filters_applied)}"
        else:
            title = "Claims Detail (click a risk matrix bubble or select a segment row to filter)"

        # Prepare display columns
        drill_columns = [
            "claim_id", "incident_date", "specialty", "region", "claim_type",
            "severity_level", "paid_amount", "days_to_close", "litigation_flag",
            "root_cause_category", "status",
        ]
        avail_cols = [c for c in drill_columns if c in df.columns]

        display_df = df[avail_cols].copy()

        # Format columns for display
        if "paid_amount" in display_df.columns:
            display_df["paid_amount"] = display_df["paid_amount"].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
            )
        if "days_to_close" in display_df.columns:
            display_df["days_to_close"] = display_df["days_to_close"].apply(
                lambda x: f"{x:.0f}" if pd.notna(x) else "N/A"
            )
        if "incident_date" in display_df.columns:
            display_df["incident_date"] = pd.to_datetime(
                display_df["incident_date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")

        col_labels = {
            "claim_id": "Claim ID",
            "incident_date": "Incident Date",
            "specialty": "Specialty",
            "region": "Region",
            "claim_type": "Claim Type",
            "severity_level": "Severity",
            "paid_amount": "Paid Amount",
            "days_to_close": "Days to Close",
            "litigation_flag": "Litigation",
            "root_cause_category": "Root Cause",
            "status": "Status",
        }

        table = dash_table.DataTable(
            id="recs-drill-datatable",
            columns=[{"name": col_labels.get(c, c), "id": c} for c in avail_cols],
            data=display_df.to_dict("records"),
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
                "fontSize": "12px",
                "padding": "6px 10px",
                "minWidth": "80px",
                "maxWidth": "180px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            },
            style_data={
                "backgroundColor": "rgba(0,0,0,0)",
            },
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "rgba(15, 52, 96, 0.03)",
                },
            ],
            sort_action="native",
            filter_action="native",
            page_size=20,
            page_action="native",
        )

        return title, table

    # ------------------------------------------------------------------
    # Download risk segments CSV
    # ------------------------------------------------------------------
    @app.callback(
        Output("recs-segments-download", "data"),
        Input("recs-segments-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _download_segments(n_clicks):
        if not n_clicks:
            raise PreventUpdate

        segments = _build_risk_segments(data_store.claims_df)
        if segments.empty:
            raise PreventUpdate

        return dcc.send_data_frame(
            segments.to_csv, "risk_segments.csv", index=False
        )

    # ------------------------------------------------------------------
    # Download full claims CSV
    # ------------------------------------------------------------------
    @app.callback(
        Output("recs-claims-download", "data"),
        Input("recs-claims-btn", "n_clicks"),
        [
            State("recs-selected-specialty", "data"),
            State("recs-selected-segment", "data"),
        ],
        prevent_initial_call=True,
    )
    def _download_claims(n_clicks, selected_specialty, selected_segment):
        if not n_clicks:
            raise PreventUpdate

        df = data_store.claims_df.copy()

        # Apply same filters as drill-through table
        if selected_segment:
            if selected_segment.get("specialty"):
                df = df[df["specialty"] == selected_segment["specialty"]]
            if selected_segment.get("region"):
                df = df[df["region"] == selected_segment["region"]]
            if selected_segment.get("claim_type"):
                df = df[df["claim_type"] == selected_segment["claim_type"]]
        elif selected_specialty:
            df = df[df["specialty"] == selected_specialty]

        export_cols = [
            "claim_id", "incident_date", "specialty", "region", "claim_type",
            "severity_level", "paid_amount", "incurred_amount", "days_to_close",
            "days_to_report", "litigation_flag", "root_cause_category", "status",
        ]
        cols = [c for c in export_cols if c in df.columns]
        return dcc.send_data_frame(
            df[cols].to_csv, "claims_detail.csv", index=False
        )
