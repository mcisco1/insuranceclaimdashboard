"""What-If Scenario Analysis & Benchmarks page.

Route: /scenario-analysis
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
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


def _build_scenario_card(
    title: str,
    baseline_label: str,
    baseline_val: str,
    projected_label: str,
    projected_val: str,
    savings_val: str,
    pct_change: str,
) -> html.Div:
    """Build a styled before/after comparison card."""
    return html.Div(
        [
            html.H5(title, style={
                "fontFamily": _FONT_FAMILY,
                "color": PRIMARY,
                "marginBottom": "12px",
                "fontSize": "14px",
                "fontWeight": "600",
            }),
            html.Div(
                [
                    html.Div(
                        [
                            html.P(baseline_label, style={
                                "fontSize": "11px", "color": "#636e72", "margin": "0",
                            }),
                            html.P(baseline_val, style={
                                "fontSize": "18px", "fontWeight": "bold", "color": SECONDARY,
                                "margin": "4px 0",
                            }),
                        ],
                        style={"flex": "1", "textAlign": "center"},
                    ),
                    html.Div(
                        html.Span(
                            "\u2192",
                            style={"fontSize": "24px", "color": "#b2bec3"},
                        ),
                        style={"display": "flex", "alignItems": "center", "padding": "0 8px"},
                    ),
                    html.Div(
                        [
                            html.P(projected_label, style={
                                "fontSize": "11px", "color": "#636e72", "margin": "0",
                            }),
                            html.P(projected_val, style={
                                "fontSize": "18px", "fontWeight": "bold", "color": SUCCESS,
                                "margin": "4px 0",
                            }),
                        ],
                        style={"flex": "1", "textAlign": "center"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
            ),
            html.Div(
                [
                    html.Span(f"Savings: {savings_val}", style={
                        "fontSize": "13px", "color": SUCCESS, "fontWeight": "600",
                    }),
                    html.Span(f"  ({pct_change})", style={
                        "fontSize": "12px", "color": "#636e72", "marginLeft": "6px",
                    }),
                ],
                style={"textAlign": "center"},
            ),
        ],
        className="kpi-card",
        style={"flex": "1", "minWidth": "260px"},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════════════

def layout(data_store) -> html.Div:
    """Return the Scenario Analysis & Benchmarks page layout."""
    scenarios = data_store.scenarios
    monte_carlo = data_store.monte_carlo
    benchmarks = data_store.benchmarks

    # ── Build PSI benchmark bar chart ─────────────────────────────────
    facility_data = benchmarks.get("facility_vs_benchmark", {})
    psi_indicators = benchmarks.get("psi_indicators", {})

    benchmark_bar_fig = go.Figure()

    if facility_data and isinstance(facility_data, dict):
        # Get all indicator names
        indicator_names = list(psi_indicators.keys()) if psi_indicators else []
        departments = sorted(facility_data.keys())

        if indicator_names and departments:
            # For each indicator, build grouped bar
            indicator_labels = [n.replace("_", " ").title() for n in indicator_names]

            for i, indicator in enumerate(indicator_names):
                dept_rates = []
                benchmark_rates = []
                dept_names = []

                for dept in departments:
                    dept_data = facility_data[dept]
                    if isinstance(dept_data, dict) and indicator in dept_data:
                        ind_data = dept_data[indicator]
                        if isinstance(ind_data, dict):
                            dept_rates.append(ind_data.get("dept_rate_per_1000", 0))
                            benchmark_rates.append(ind_data.get("benchmark_rate_per_1000", 0))
                            dept_names.append(dept)

                if i == 0 and dept_names:
                    # Add benchmark trace (only once, for first indicator)
                    # We'll show department vs benchmark per indicator instead
                    pass

            # Show as grouped bar: each department, facility rate vs benchmark
            # Pick the most important indicators (first 3)
            display_indicators = indicator_names[:3]
            colors = [PRIMARY, ACCENT, WARNING, SUCCESS, "#74b9ff"]

            for idx, indicator in enumerate(display_indicators):
                dept_rates_list = []
                dept_labels = []

                for dept in departments:
                    dept_data = facility_data[dept]
                    if isinstance(dept_data, dict) and indicator in dept_data:
                        ind_data = dept_data[indicator]
                        if isinstance(ind_data, dict):
                            dept_rates_list.append(ind_data.get("dept_rate_per_1000", 0))
                            if idx == 0:
                                dept_labels.append(dept)

                label = indicator.replace("_", " ").title()
                benchmark_bar_fig.add_trace(
                    go.Bar(
                        x=departments[:len(dept_rates_list)],
                        y=dept_rates_list,
                        name=label,
                        marker_color=colors[idx % len(colors)],
                        hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}} per 1000<extra></extra>",
                    )
                )

            # Add benchmark reference lines for each indicator
            for idx, indicator in enumerate(display_indicators):
                national_rate = psi_indicators.get(indicator, {}).get("rate_per_1000", 0)
                if national_rate > 0:
                    label = indicator.replace("_", " ").title()
                    benchmark_bar_fig.add_hline(
                        y=national_rate,
                        line_dash="dash",
                        line_color=colors[idx % len(colors)],
                        annotation_text=f"{label} Avg: {national_rate:.1f}",
                        annotation_position="top right" if idx == 0 else "bottom right",
                        annotation_font_size=9,
                        opacity=0.6,
                    )

    benchmark_bar_fig.update_layout(
        **_COMMON_LAYOUT,
        barmode="group",
        title=dict(text="PSI Benchmark: Facility vs National Average", font=dict(size=14)),
        xaxis_title="Department",
        yaxis_title="Rate per 1,000 Claims",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        height=420,
    )

    # ── Quarterly PSI trend chart ─────────────────────────────────────
    benchmark_trends = benchmarks.get("benchmark_trends", {})

    trend_fig = go.Figure()

    if isinstance(benchmark_trends, dict) and benchmark_trends:
        trend_colors = [PRIMARY, ACCENT, SUCCESS, WARNING, "#74b9ff"]

        for idx, (indicator, quarters_dict) in enumerate(benchmark_trends.items()):
            if isinstance(quarters_dict, dict) and quarters_dict:
                q_labels = sorted(quarters_dict.keys())
                q_values = [quarters_dict[q] for q in q_labels]

                label = indicator.replace("_", " ").title()
                trend_fig.add_trace(
                    go.Scatter(
                        x=q_labels,
                        y=q_values,
                        mode="lines+markers",
                        name=label,
                        line=dict(width=2, color=trend_colors[idx % len(trend_colors)]),
                        marker=dict(size=5),
                        hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}} per 1000<extra></extra>",
                    )
                )

    trend_fig.update_layout(
        **_COMMON_LAYOUT,
        title=dict(text="Quarterly PSI Indicator Trends", font=dict(size=14)),
        xaxis_title="Quarter",
        yaxis_title="Rate per 1,000 Claims",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        height=420,
    )

    # ── Combined scenario summary values ──────────────────────────────
    combined = scenarios.get("combined_scenario", {})
    combined_baseline = combined.get("baseline_total_incurred", 0)
    combined_new = combined.get("new_total_incurred", 0)
    combined_savings = combined.get("total_savings", 0)
    combined_pct = combined.get("pct_change", 0)
    components = combined.get("components", {})

    # Monte Carlo histogram
    mc_dist = monte_carlo.get("distribution", [])
    mc_fig = go.Figure()
    if mc_dist:
        mc_fig.add_trace(go.Histogram(
            x=mc_dist,
            nbinsx=60,
            marker_color=PRIMARY,
            opacity=0.85,
            hovertemplate="Savings: $%{x:,.0f}<br>Count: %{y:,}<extra></extra>",
        ))
        # Percentile lines
        p5 = monte_carlo.get("p5", 0)
        p95 = monte_carlo.get("p95", 0)
        mc_fig.add_vline(x=p5, line_dash="dash", line_color=ACCENT,
                         annotation_text=f"5th pctl: {_fmt_currency(p5)}",
                         annotation_position="top left", annotation_font_size=10)
        mc_fig.add_vline(x=p95, line_dash="dash", line_color=SUCCESS,
                         annotation_text=f"95th pctl: {_fmt_currency(p95)}",
                         annotation_position="top right", annotation_font_size=10)
        median_val = monte_carlo.get("median_savings", 0)
        mc_fig.add_vline(x=median_val, line_dash="solid", line_color=SECONDARY,
                         annotation_text=f"Median: {_fmt_currency(median_val)}",
                         annotation_position="top", annotation_font_size=10)
    mc_fig.update_layout(
        **_COMMON_LAYOUT,
        title=dict(text="Monte Carlo Simulation: Savings Distribution (5,000 simulations)", font=dict(size=14)),
        xaxis_title="Potential Savings ($)",
        xaxis=dict(tickformat="$,.0f"),
        yaxis_title="Frequency",
        showlegend=False,
        height=420,
    )

    mc_var = monte_carlo.get("var_95", 0)
    mc_prob = monte_carlo.get("prob_positive", 0)
    kpi_var = create_kpi_card(
        "95% VaR",
        _fmt_currency(mc_var),
        delta="5th percentile of savings",
        delta_color=WARNING,
    )
    kpi_prob = create_kpi_card(
        "P(Savings > 0)",
        f"{mc_prob:.1f}%",
        delta="Monte Carlo estimate",
        delta_color=SUCCESS if mc_prob > 90 else WARNING,
    )

    return html.Div(
        [
            html.H2(
                [
                    "What-If Scenario Analysis & Benchmarks",
                    create_info_tooltip(
                        "Monte Carlo simulation (5,000 draws) with Beta-distributed parameter uncertainty. "
                        "Deterministic scenarios provide point estimates; MC provides distributions.",
                        "scenario-tooltip",
                    ),
                ],
                className="page-title",
            ),

            # Row 1: Sliders
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Scenario Parameters", style={
                                "fontFamily": _FONT_FAMILY,
                                "color": PRIMARY,
                                "marginBottom": "20px",
                            }),
                            # Close time reduction slider
                            html.Div(
                                [
                                    html.Label("Close Time Reduction", style={
                                        "fontFamily": _FONT_FAMILY,
                                        "fontWeight": "600",
                                        "color": SECONDARY,
                                        "fontSize": "13px",
                                    }),
                                    dcc.Slider(
                                        id="scenario-close-slider",
                                        min=0,
                                        max=30,
                                        step=None,
                                        marks={0: "0%", 10: "10%", 20: "20%", 30: "30%"},
                                        value=0,
                                        className="scenario-slider",
                                    ),
                                ],
                                style={"marginBottom": "24px"},
                            ),
                            # Severity reduction slider
                            html.Div(
                                [
                                    html.Label("Severity Reduction (High \u2192 Medium)", style={
                                        "fontFamily": _FONT_FAMILY,
                                        "fontWeight": "600",
                                        "color": SECONDARY,
                                        "fontSize": "13px",
                                    }),
                                    dcc.Slider(
                                        id="scenario-severity-slider",
                                        min=0,
                                        max=30,
                                        step=None,
                                        marks={0: "0%", 10: "10%", 20: "20%", 30: "30%"},
                                        value=0,
                                        className="scenario-slider",
                                    ),
                                ],
                                style={"marginBottom": "24px"},
                            ),
                            # Litigation reduction slider
                            html.Div(
                                [
                                    html.Label("Litigation Rate Reduction", style={
                                        "fontFamily": _FONT_FAMILY,
                                        "fontWeight": "600",
                                        "color": SECONDARY,
                                        "fontSize": "13px",
                                    }),
                                    dcc.Slider(
                                        id="scenario-litigation-slider",
                                        min=0,
                                        max=50,
                                        step=None,
                                        marks={0: "0%", 25: "25%", 50: "50%"},
                                        value=0,
                                        className="scenario-slider",
                                    ),
                                ],
                                style={"marginBottom": "12px"},
                            ),
                        ],
                        className="card full-card",
                        style={"padding": "24px"},
                    ),
                ],
                className="chart-row",
            ),

            # Row 2: Before/After comparison cards (dynamic)
            html.Div(id="scenario-comparison-cards", className="kpi-row"),

            # Row 3: Combined scenario summary
            html.Div(
                html.Div(
                    [
                        html.H4("Combined Scenario Summary", style={
                            "fontFamily": _FONT_FAMILY,
                            "color": PRIMARY,
                            "marginBottom": "16px",
                        }),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P("Baseline Total Incurred", style={
                                            "fontSize": "12px", "color": "#636e72", "margin": "0",
                                        }),
                                        html.H3(
                                            _fmt_currency(combined_baseline),
                                            style={"color": SECONDARY, "margin": "4px 0"},
                                        ),
                                    ],
                                    style={"flex": "1", "textAlign": "center"},
                                ),
                                html.Div(
                                    [
                                        html.P("Projected Total", style={
                                            "fontSize": "12px", "color": "#636e72", "margin": "0",
                                        }),
                                        html.H3(
                                            _fmt_currency(combined_new),
                                            style={"color": SUCCESS, "margin": "4px 0"},
                                        ),
                                    ],
                                    style={"flex": "1", "textAlign": "center"},
                                ),
                                html.Div(
                                    [
                                        html.P("Total Potential Savings", style={
                                            "fontSize": "12px", "color": "#636e72", "margin": "0",
                                        }),
                                        html.H3(
                                            _fmt_currency(combined_savings),
                                            style={"color": SUCCESS, "margin": "4px 0"},
                                        ),
                                        html.Span(
                                            f"{abs(combined_pct):.1f}% reduction",
                                            style={"fontSize": "13px", "color": SUCCESS},
                                        ),
                                    ],
                                    style={"flex": "1", "textAlign": "center"},
                                ),
                            ],
                            style={"display": "flex", "gap": "20px"},
                        ),
                        # Component breakdown
                        html.Hr(style={"margin": "16px 0", "borderColor": "#dfe6e9"}),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P("Severity Shift Savings", style={
                                            "fontSize": "12px", "color": "#636e72", "margin": "0",
                                        }),
                                        html.P(
                                            _fmt_currency(components.get("severity_shift_savings", 0)),
                                            style={"fontSize": "16px", "fontWeight": "bold",
                                                   "color": PRIMARY, "margin": "4px 0"},
                                        ),
                                    ],
                                    style={"flex": "1", "textAlign": "center"},
                                ),
                                html.Div(
                                    [
                                        html.P("Litigation Reduction Savings", style={
                                            "fontSize": "12px", "color": "#636e72", "margin": "0",
                                        }),
                                        html.P(
                                            _fmt_currency(components.get("litigation_reduction_savings", 0)),
                                            style={"fontSize": "16px", "fontWeight": "bold",
                                                   "color": PRIMARY, "margin": "4px 0"},
                                        ),
                                    ],
                                    style={"flex": "1", "textAlign": "center"},
                                ),
                                html.Div(
                                    [
                                        html.P("Close Time Reduction Savings", style={
                                            "fontSize": "12px", "color": "#636e72", "margin": "0",
                                        }),
                                        html.P(
                                            _fmt_currency(components.get("close_time_reduction_savings", 0)),
                                            style={"fontSize": "16px", "fontWeight": "bold",
                                                   "color": PRIMARY, "margin": "4px 0"},
                                        ),
                                    ],
                                    style={"flex": "1", "textAlign": "center"},
                                ),
                            ],
                            style={"display": "flex", "gap": "20px"},
                        ),
                    ],
                    className="card full-card",
                    style={"padding": "24px"},
                ),
                className="chart-row",
            ),

            # Monte Carlo KPIs
            html.Div(
                [kpi_var, kpi_prob],
                className="kpi-row",
            ),

            # Monte Carlo histogram
            html.Div(
                html.Div(
                    dcc.Graph(
                        id="scenario-mc-histogram",
                        figure=mc_fig,
                        config={"displayModeBar": False},
                    ),
                    className="card full-card",
                ),
                className="chart-row",
            ),

            # Row 4: PSI benchmark comparison
            html.Div(
                html.Div(
                    dcc.Graph(
                        id="scenario-benchmark-bar",
                        figure=benchmark_bar_fig,
                        config={"displayModeBar": False},
                    ),
                    className="card full-card",
                ),
                className="chart-row",
            ),

            # Row 5: Quarterly PSI trends
            html.Div(
                html.Div(
                    dcc.Graph(
                        id="scenario-benchmark-trends",
                        figure=trend_fig,
                        config={"displayModeBar": False},
                    ),
                    className="card full-card",
                ),
                className="chart-row",
            ),

            # Download button
            create_download_button("scenario"),
        ],
        className="page-container",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════

def register_callbacks(app, data_store) -> None:
    """Register Scenario Analysis page callbacks."""

    # ------------------------------------------------------------------
    # Slider-driven scenario comparison cards
    # ------------------------------------------------------------------
    @app.callback(
        Output("scenario-comparison-cards", "children"),
        [
            Input("scenario-close-slider", "value"),
            Input("scenario-severity-slider", "value"),
            Input("scenario-litigation-slider", "value"),
        ],
    )
    def _update_comparison_cards(close_pct, severity_pct, litigation_pct):
        scenarios = data_store.scenarios
        cards = []

        # ── Close Time Reduction card ─────────────────────────────────
        close_data = scenarios.get("close_time_reduction", {})
        close_baseline = close_data.get("baseline", {})
        close_scenarios = close_data.get("scenarios", {})

        baseline_close_days = close_baseline.get("mean_close_days", 0)

        if close_pct == 0:
            proj_close_days = baseline_close_days
            close_savings = 0
            close_change = "0.0%"
        else:
            key = f"{close_pct}_pct_reduction"
            scenario = close_scenarios.get(key, {})
            proj_close_days = scenario.get("new_mean_close_days", baseline_close_days)
            abs_change = scenario.get("absolute_change", {})
            if isinstance(abs_change, dict):
                close_savings = abs_change.get("mean_close_days", 0)
            else:
                close_savings = baseline_close_days - proj_close_days
            close_change = f"{close_pct}%"

        cards.append(
            _build_scenario_card(
                title=f"Close Time Reduction ({close_pct}%)",
                baseline_label="Baseline Avg Days",
                baseline_val=f"{baseline_close_days:.0f} days",
                projected_label="Projected Avg Days",
                projected_val=f"{proj_close_days:.0f} days",
                savings_val=f"{abs(baseline_close_days - proj_close_days):.0f} days",
                pct_change=f"-{close_change}" if close_pct > 0 else close_change,
            )
        )

        # ── Severity Reduction card ───────────────────────────────────
        sev_data = scenarios.get("severity_reduction", {})
        sev_baseline = sev_data.get("baseline", {})
        sev_scenarios = sev_data.get("scenarios", {})

        baseline_incurred = sev_baseline.get("total_incurred", 0)

        if severity_pct == 0:
            proj_incurred = baseline_incurred
            sev_pct_change = "0.0%"
        else:
            key = f"{severity_pct}_pct_shift"
            scenario = sev_scenarios.get(key, {})
            proj_incurred = scenario.get("new_total_incurred", baseline_incurred)
            pct_ch = scenario.get("pct_change", 0)
            if isinstance(pct_ch, dict):
                pct_ch = 0
            sev_pct_change = f"{pct_ch:+.1f}%"

        sev_savings = baseline_incurred - proj_incurred

        cards.append(
            _build_scenario_card(
                title=f"Severity Shift ({severity_pct}%)",
                baseline_label="Baseline Incurred",
                baseline_val=_fmt_currency(baseline_incurred),
                projected_label="Projected Incurred",
                projected_val=_fmt_currency(proj_incurred),
                savings_val=_fmt_currency(sev_savings),
                pct_change=sev_pct_change,
            )
        )

        # ── Litigation Reduction card ─────────────────────────────────
        lit_data = scenarios.get("litigation_reduction", {})
        lit_baseline = lit_data.get("baseline", {})
        lit_scenarios = lit_data.get("scenarios", {})

        baseline_lit_incurred = lit_baseline.get("total_incurred", 0)

        if litigation_pct == 0:
            proj_lit_incurred = baseline_lit_incurred
            lit_pct_change = "0.0%"
        else:
            key = f"{litigation_pct}_pct_reduction"
            scenario = lit_scenarios.get(key, {})
            proj_lit_incurred = scenario.get("new_total_incurred", baseline_lit_incurred)
            pct_ch = scenario.get("pct_change", 0)
            if isinstance(pct_ch, dict):
                pct_ch = 0
            lit_pct_change = f"{pct_ch:+.1f}%"

        lit_savings = baseline_lit_incurred - proj_lit_incurred

        cards.append(
            _build_scenario_card(
                title=f"Litigation Reduction ({litigation_pct}%)",
                baseline_label="Baseline Incurred",
                baseline_val=_fmt_currency(baseline_lit_incurred),
                projected_label="Projected Incurred",
                projected_val=_fmt_currency(proj_lit_incurred),
                savings_val=_fmt_currency(lit_savings),
                pct_change=lit_pct_change,
            )
        )

        return cards

    # ------------------------------------------------------------------
    # Download scenario summary CSV
    # ------------------------------------------------------------------
    @app.callback(
        Output("scenario-download", "data"),
        Input("scenario-btn", "n_clicks"),
        [
            State("scenario-close-slider", "value"),
            State("scenario-severity-slider", "value"),
            State("scenario-litigation-slider", "value"),
        ],
        prevent_initial_call=True,
    )
    def _download_scenario(n_clicks, close_pct, severity_pct, litigation_pct):
        if not n_clicks:
            raise PreventUpdate

        scenarios = data_store.scenarios
        rows = []

        # Close time
        close_data = scenarios.get("close_time_reduction", {})
        close_bl = close_data.get("baseline", {})
        rows.append({
            "Scenario": "Close Time Reduction",
            "Parameter": f"{close_pct}%",
            "Baseline Avg Close Days": close_bl.get("mean_close_days", ""),
            "Baseline Total Incurred": "",
            "Note": "Days-based metric",
        })

        # Severity
        sev_data = scenarios.get("severity_reduction", {})
        sev_bl = sev_data.get("baseline", {})
        sev_scen = sev_data.get("scenarios", {}).get(f"{severity_pct}_pct_shift", {})
        rows.append({
            "Scenario": "Severity Shift",
            "Parameter": f"{severity_pct}%",
            "Baseline Avg Close Days": "",
            "Baseline Total Incurred": sev_bl.get("total_incurred", ""),
            "Note": f"Projected: {sev_scen.get('new_total_incurred', '')}",
        })

        # Litigation
        lit_data = scenarios.get("litigation_reduction", {})
        lit_bl = lit_data.get("baseline", {})
        lit_scen = lit_data.get("scenarios", {}).get(f"{litigation_pct}_pct_reduction", {})
        rows.append({
            "Scenario": "Litigation Reduction",
            "Parameter": f"{litigation_pct}%",
            "Baseline Avg Close Days": "",
            "Baseline Total Incurred": lit_bl.get("total_incurred", ""),
            "Note": f"Projected: {lit_scen.get('new_total_incurred', '')}",
        })

        # Combined
        combined = scenarios.get("combined_scenario", {})
        rows.append({
            "Scenario": "Combined (20% sev + 25% lit + 20% close)",
            "Parameter": "Combined",
            "Baseline Avg Close Days": "",
            "Baseline Total Incurred": combined.get("baseline_total_incurred", ""),
            "Note": f"Total Savings: {combined.get('total_savings', '')}",
        })

        export_df = pd.DataFrame(rows)
        return dcc.send_data_frame(
            export_df.to_csv, "scenario_analysis.csv", index=False
        )
