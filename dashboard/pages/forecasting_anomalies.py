"""Claims Forecasting & Anomaly Detection page.

Route: /forecasting
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

def _fmt_number(val: float) -> str:
    """Format a number as a compact string."""
    if abs(val) >= 1_000_000:
        return f"{val / 1_000_000:,.1f}M"
    if abs(val) >= 1_000:
        return f"{val / 1_000:,.1f}K"
    return f"{val:,.0f}"


# ═══════════════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════════════

def layout(data_store) -> html.Div:
    """Return the Forecasting & Anomaly Detection page layout."""
    forecast_data = data_store.forecast
    anomaly_data = data_store.anomalies
    severity_model = data_store.severity_model

    # ── KPI values ────────────────────────────────────────────────────
    forecast_arr = forecast_data.get("forecast", np.array([]))
    forecast_next = float(forecast_arr[0]) if len(forecast_arr) > 0 else 0
    forecast_total_12m = float(forecast_arr.sum()) if len(forecast_arr) > 0 else 0
    anomaly_rate = anomaly_data.get("anomaly_rate", 0.0)
    model_aic = forecast_data.get("aic", np.nan)

    kpi_forecast_next = create_kpi_card(
        "Forecast Next Month",
        _fmt_number(forecast_next),
        delta="Claims expected",
        delta_color=PRIMARY,
    )
    kpi_forecast_total = create_kpi_card(
        "12-Month Forecast Total",
        _fmt_number(forecast_total_12m),
        delta=f"{len(forecast_arr)} months ahead",
        delta_color=PRIMARY,
    )
    kpi_anomaly_rate = create_kpi_card(
        "Anomaly Rate",
        f"{anomaly_rate:.1f}%",
        delta=f"{anomaly_data.get('anomaly_count', 0):,} flagged claims",
        delta_color=ACCENT if anomaly_rate > 5 else SUCCESS,
    )
    kpi_aic = create_kpi_card(
        "Model AIC",
        f"{model_aic:,.1f}" if not np.isnan(model_aic) else "N/A",
        delta=forecast_data.get("model_params", {}).get("method", "").replace("_", " ").title(),
        delta_color=PRIMARY,
    )

    # Backtest MAPE KPI
    backtest = forecast_data.get("backtest", {})
    backtest_mape = backtest.get("mape")
    kpi_backtest = create_kpi_card(
        "Backtest MAPE",
        f"{backtest_mape:.1f}%" if backtest_mape is not None else "N/A",
        delta="6-month holdout" if backtest_mape is not None else "Insufficient data",
        delta_color=SUCCESS if backtest_mape is not None and backtest_mape < 15 else WARNING,
    )

    # ── Forecast chart ────────────────────────────────────────────────
    monthly_actuals = forecast_data.get("monthly_actuals", pd.DataFrame())
    forecast_dates = forecast_data.get("forecast_dates", pd.DatetimeIndex([]))
    conf_80 = forecast_data.get("confidence_80", {})
    conf_95 = forecast_data.get("confidence_95", {})

    forecast_fig = go.Figure()

    if not monthly_actuals.empty:
        # Historical actuals
        actual_dates = monthly_actuals.index
        actual_values = monthly_actuals.iloc[:, 0].values

        forecast_fig.add_trace(
            go.Scatter(
                x=actual_dates,
                y=actual_values,
                mode="lines+markers",
                name="Historical Actuals",
                line=dict(width=2, color=PRIMARY),
                marker=dict(size=3),
                hovertemplate="<b>%{x|%Y-%m}</b><br>Actual: %{y:,.0f}<extra></extra>",
            )
        )

    if len(forecast_arr) > 0 and len(forecast_dates) > 0:
        # 95% confidence band (lighter)
        if "upper" in conf_95 and "lower" in conf_95:
            forecast_fig.add_trace(
                go.Scatter(
                    x=list(forecast_dates) + list(forecast_dates)[::-1],
                    y=list(conf_95["upper"]) + list(conf_95["lower"])[::-1],
                    fill="toself",
                    fillcolor="rgba(15, 52, 96, 0.08)",
                    line=dict(width=0),
                    name="95% CI",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

        # 80% confidence band (darker)
        if "upper" in conf_80 and "lower" in conf_80:
            forecast_fig.add_trace(
                go.Scatter(
                    x=list(forecast_dates) + list(forecast_dates)[::-1],
                    y=list(conf_80["upper"]) + list(conf_80["lower"])[::-1],
                    fill="toself",
                    fillcolor="rgba(15, 52, 96, 0.18)",
                    line=dict(width=0),
                    name="80% CI",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

        # Forecast line (dashed)
        forecast_fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_arr,
                mode="lines+markers",
                name="Forecast",
                line=dict(width=2, color=ACCENT, dash="dash"),
                marker=dict(size=4),
                hovertemplate="<b>%{x|%Y-%m}</b><br>Forecast: %{y:,.0f}<extra></extra>",
            )
        )

        # Connect actuals to forecast with a linking segment
        if not monthly_actuals.empty:
            last_actual_date = actual_dates[-1]
            last_actual_val = float(actual_values[-1])
            first_forecast_date = forecast_dates[0]
            first_forecast_val = float(forecast_arr[0])

            forecast_fig.add_trace(
                go.Scatter(
                    x=[last_actual_date, first_forecast_date],
                    y=[last_actual_val, first_forecast_val],
                    mode="lines",
                    line=dict(width=1.5, color=ACCENT, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    forecast_fig.update_layout(
        **_COMMON_LAYOUT,
        title=dict(text="Monthly Claims Forecast with Confidence Intervals", font=dict(size=14)),
        xaxis_title="Month",
        yaxis_title="Claim Count",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=440,
    )

    # ── Model comparison chart ─────────────────────────────────────────
    model_comparison = severity_model.get("model_comparison", {})
    model_comp_fig = go.Figure()
    rankings = model_comparison.get("rankings", [])
    if rankings:
        names = [r["name"] for r in rankings]
        f1_means = [r["f1_mean"] for r in rankings]
        f1_stds = [r["f1_std"] for r in rankings]
        model_comp_fig.add_trace(go.Bar(
            x=names, y=f1_means,
            error_y=dict(type="data", array=f1_stds, visible=True),
            marker_color=[PRIMARY, ACCENT, SUCCESS][:len(names)],
            hovertemplate="<b>%{x}</b><br>F1-macro: %{y:.4f}<extra></extra>",
        ))
        # Annotate statistical test
        stat_test = model_comparison.get("statistical_test", {})
        if stat_test:
            sig = "significant" if stat_test.get("significant") else "not significant"
            annotation = f"Paired t-test: {stat_test.get('model_a', '')} vs {stat_test.get('model_b', '')}: p={stat_test.get('p_value', 0):.4f} ({sig})"
            model_comp_fig.add_annotation(
                text=annotation, xref="paper", yref="paper",
                x=0.5, y=-0.18, showarrow=False, font=dict(size=10, color="#64748b"),
            )
    model_comp_fig.update_layout(
        **_COMMON_LAYOUT,
        title=dict(text="Model Comparison: 5-Fold CV F1-Macro (+/- Std)", font=dict(size=14)),
        xaxis_title="Model",
        yaxis_title="F1-Macro Score",
        showlegend=False,
        height=400,
    )

    # ── Calibration curve ─────────────────────────────────────────────
    calibration_curves = model_comparison.get("calibration_curves", {})
    cal_fig = go.Figure()
    cal_colors = [PRIMARY, ACCENT, SUCCESS]
    for idx, (model_name, cal_data) in enumerate(calibration_curves.items()):
        fractions = cal_data.get("fractions", [])
        means = cal_data.get("mean_predicted", [])
        if fractions and means:
            # Average across classes for the reliability diagram
            avg_frac = np.mean(fractions, axis=0) if len(fractions) > 0 else []
            avg_mean = np.mean(means, axis=0) if len(means) > 0 else []
            if len(avg_frac) > 0:
                cal_fig.add_trace(go.Scatter(
                    x=avg_mean, y=avg_frac,
                    mode="lines+markers",
                    name=model_name,
                    line=dict(width=2, color=cal_colors[idx % len(cal_colors)]),
                    marker=dict(size=6),
                ))
    # Perfect calibration line
    cal_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", name="Perfect",
        line=dict(width=1, color="#94a3b8", dash="dash"),
        showlegend=True,
    ))
    cal_fig.update_layout(
        **_COMMON_LAYOUT,
        title=dict(text="Calibration Curve (Reliability Diagram)", font=dict(size=14)),
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=400,
    )

    # ── Backtest actual vs forecast ───────────────────────────────────
    backtest_fig = go.Figure()
    avf = backtest.get("actual_vs_forecast", [])
    if avf:
        dates_bt = [r["date"] for r in avf]
        actuals_bt = [r["actual"] for r in avf]
        forecasts_bt = [r["forecast"] for r in avf]
        backtest_fig.add_trace(go.Scatter(
            x=dates_bt, y=actuals_bt,
            mode="lines+markers", name="Actual",
            line=dict(width=2, color=PRIMARY),
        ))
        backtest_fig.add_trace(go.Scatter(
            x=dates_bt, y=forecasts_bt,
            mode="lines+markers", name="Forecast",
            line=dict(width=2, color=ACCENT, dash="dash"),
        ))
    backtest_fig.update_layout(
        **_COMMON_LAYOUT,
        title=dict(text="Backtest: Actual vs Forecast (6-Month Holdout)", font=dict(size=14)),
        xaxis_title="Month",
        yaxis_title="Claim Count",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=350,
    )

    # ── SHAP summary chart ────────────────────────────────────────────
    shap_vals = anomaly_data.get("shap_values")
    shap_feature_names = anomaly_data.get("feature_names", [])
    shap_fig = go.Figure()
    if shap_vals is not None and len(shap_feature_names) > 0:
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        sorted_idx = np.argsort(mean_abs_shap)
        sorted_names = [shap_feature_names[i].replace("_", " ").title() for i in sorted_idx]
        sorted_values = mean_abs_shap[sorted_idx]
        shap_fig.add_trace(go.Bar(
            x=sorted_values, y=sorted_names,
            orientation="h",
            marker_color=PRIMARY,
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        ))
    shap_fig.update_layout(
        **_COMMON_LAYOUT,
        title=dict(text="SHAP Feature Importance for Anomaly Detection", font=dict(size=14)),
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        showlegend=False,
        height=350,
    )

    # ── Anomaly scatter plot ──────────────────────────────────────────
    claims_df = data_store.claims_df
    anomaly_flags = anomaly_data.get("anomaly_flags", pd.Series(dtype=bool))

    anomaly_scatter_fig = go.Figure()

    if not anomaly_flags.empty and "paid_amount" in claims_df.columns and "days_to_close" in claims_df.columns:
        plot_df = claims_df[["paid_amount", "days_to_close", "claim_id"]].copy()
        plot_df["is_anomaly"] = anomaly_flags
        plot_df = plot_df.dropna(subset=["paid_amount", "days_to_close"])

        # Sample down for performance if needed (keep all anomalies)
        normal = plot_df[~plot_df["is_anomaly"]]
        anomalous = plot_df[plot_df["is_anomaly"]]

        if len(normal) > 3000:
            normal = normal.sample(n=3000, random_state=42)

        # Normal points
        anomaly_scatter_fig.add_trace(
            go.Scatter(
                x=normal["days_to_close"],
                y=normal["paid_amount"],
                mode="markers",
                name="Normal",
                marker=dict(size=4, color=PRIMARY, opacity=0.4),
                hovertemplate=(
                    "<b>Normal Claim</b><br>"
                    "Days to Close: %{x:,.0f}<br>"
                    "Paid: $%{y:,.0f}<extra></extra>"
                ),
            )
        )

        # Anomaly points
        anomaly_scatter_fig.add_trace(
            go.Scatter(
                x=anomalous["days_to_close"],
                y=anomalous["paid_amount"],
                mode="markers",
                name="Anomaly",
                marker=dict(size=6, color=ACCENT, opacity=0.8, symbol="diamond"),
                hovertemplate=(
                    "<b>Anomaly</b><br>"
                    "Days to Close: %{x:,.0f}<br>"
                    "Paid: $%{y:,.0f}<extra></extra>"
                ),
            )
        )

    anomaly_scatter_fig.update_layout(
        **_COMMON_LAYOUT,
        title=dict(text="Anomaly Detection: Paid Amount vs Days to Close", font=dict(size=14)),
        xaxis_title="Days to Close",
        yaxis_title="Paid Amount ($)",
        yaxis=dict(tickformat="$,.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=420,
    )

    # ── Monthly anomaly rate trend ────────────────────────────────────
    trend_breaks = anomaly_data.get("trend_breaks", pd.DataFrame())

    anomaly_trend_fig = go.Figure()

    if isinstance(trend_breaks, pd.DataFrame) and not trend_breaks.empty:
        months = trend_breaks.index.tolist()
        rates = trend_breaks["anomaly_rate"].tolist() if "anomaly_rate" in trend_breaks.columns else []

        if rates:
            anomaly_trend_fig.add_trace(
                go.Scatter(
                    x=months,
                    y=rates,
                    mode="lines+markers",
                    name="Anomaly Rate",
                    line=dict(width=2, color=ACCENT),
                    marker=dict(size=4),
                    hovertemplate="<b>%{x}</b><br>Anomaly Rate: %{y:.1f}%<extra></extra>",
                )
            )

            # Add overall average line
            avg_rate = anomaly_data.get("anomaly_rate", 0.0)
            anomaly_trend_fig.add_hline(
                y=avg_rate,
                line_dash="dash",
                line_color=PRIMARY,
                annotation_text=f"Overall: {avg_rate:.1f}%",
                annotation_position="top right",
                annotation_font_size=10,
            )

    anomaly_trend_fig.update_layout(
        **_COMMON_LAYOUT,
        title=dict(text="Monthly Anomaly Rate Trend", font=dict(size=14)),
        xaxis_title="Month",
        yaxis_title="Anomaly Rate (%)",
        showlegend=False,
        height=420,
    )

    # ── Flagged claims table ──────────────────────────────────────────
    anomaly_claims = anomaly_data.get("anomaly_claims", pd.DataFrame())
    table_columns = ["claim_id", "paid_amount", "severity_level", "days_to_close", "specialty", "anomaly_score", "top_reason"]

    if not anomaly_claims.empty:
        # Ensure columns exist
        available_cols = [c for c in table_columns if c in anomaly_claims.columns]
        table_data = anomaly_claims[available_cols].head(50).copy()

        # Format for display
        if "paid_amount" in table_data.columns:
            table_data["paid_amount"] = table_data["paid_amount"].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
            )
        if "anomaly_score" in table_data.columns:
            table_data["anomaly_score"] = table_data["anomaly_score"].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            )
        if "days_to_close" in table_data.columns:
            table_data["days_to_close"] = table_data["days_to_close"].apply(
                lambda x: f"{x:.0f}" if pd.notna(x) else "N/A"
            )

        column_labels = {
            "claim_id": "Claim ID",
            "paid_amount": "Paid Amount",
            "severity_level": "Severity",
            "days_to_close": "Days to Close",
            "specialty": "Specialty",
            "anomaly_score": "Anomaly Score",
            "top_reason": "Top Reason",
        }
        display_columns = [{"name": column_labels.get(c, c), "id": c} for c in available_cols]
        display_data = table_data.to_dict("records")
    else:
        available_cols = table_columns
        display_columns = [{"name": c.replace("_", " ").title(), "id": c} for c in available_cols]
        display_data = []

    # ── Feature importance bar chart ──────────────────────────────────
    feature_importances = severity_model.get("feature_importances", pd.DataFrame())

    feat_importance_fig = go.Figure()

    if isinstance(feature_importances, pd.DataFrame) and not feature_importances.empty:
        top_features = feature_importances.head(15).sort_values("importance", ascending=True)

        feat_importance_fig.add_trace(
            go.Bar(
                x=top_features["importance"],
                y=top_features["feature"].apply(
                    lambda f: f.replace("_", " ").title() if len(f) < 30 else f[:27] + "..."
                ),
                orientation="h",
                marker_color=PRIMARY,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Importance: %{x:.4f}<extra></extra>"
                ),
            )
        )

    feat_importance_fig.update_layout(
        **_COMMON_LAYOUT,
        title=dict(text="Severity Model: Top 15 Feature Importances", font=dict(size=14)),
        xaxis_title="Importance",
        yaxis_title="",
        showlegend=False,
        height=450,
    )

    return html.Div(
        [
            html.H2(
                [
                    "Claims Forecasting & Anomaly Detection",
                    create_info_tooltip(
                        "Isolation Forest (contamination=5%). SHAP values explain individual flags. "
                        "Walk-forward backtest validates forecast accuracy.",
                        "forecast-tooltip",
                    ),
                ],
                className="page-title",
            ),

            # Row 1: KPI cards
            html.Div(
                [kpi_forecast_next, kpi_forecast_total, kpi_anomaly_rate, kpi_aic, kpi_backtest],
                className="kpi-row",
            ),

            # Row 2: Forecast chart (full width)
            html.Div(
                html.Div(
                    dcc.Graph(
                        id="forecast-main-chart",
                        figure=forecast_fig,
                        config={"displayModeBar": False},
                    ),
                    className="card full-card",
                ),
                className="chart-row",
            ),

            # Row 3: Anomaly scatter | Monthly anomaly trend
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            id="forecast-anomaly-scatter",
                            figure=anomaly_scatter_fig,
                            config={"displayModeBar": False},
                        ),
                        className="card half-card",
                    ),
                    html.Div(
                        dcc.Graph(
                            id="forecast-anomaly-trend",
                            figure=anomaly_trend_fig,
                            config={"displayModeBar": False},
                        ),
                        className="card half-card",
                    ),
                ],
                className="chart-row",
            ),

            # Row 4: Flagged claims table
            html.Div(
                html.Div(
                    [
                        html.H4(
                            "Top 50 Flagged Anomalous Claims",
                            className="table-title",
                        ),
                        dash_table.DataTable(
                            id="forecast-anomaly-table",
                            columns=display_columns,
                            data=display_data,
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
                            sort_action="native",
                            page_size=15,
                        ),
                    ],
                    className="card full-card",
                ),
                className="chart-row",
            ),

            # Row 5: Feature importance chart (full width)
            html.Div(
                html.Div(
                    dcc.Graph(
                        id="forecast-feature-importance",
                        figure=feat_importance_fig,
                        config={"displayModeBar": False},
                    ),
                    className="card full-card",
                ),
                className="chart-row",
            ),

            # Download buttons
            html.Div(
                [
                    create_download_button("forecast-data"),
                    create_download_button("forecast-anomalies"),
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
    """Register Forecasting & Anomaly Detection page callbacks."""

    # ------------------------------------------------------------------
    # Download forecast CSV
    # ------------------------------------------------------------------
    @app.callback(
        Output("forecast-data-download", "data"),
        Input("forecast-data-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _download_forecast(n_clicks):
        if not n_clicks:
            raise PreventUpdate

        forecast_data = data_store.forecast
        monthly_actuals = forecast_data.get("monthly_actuals", pd.DataFrame())
        forecast_arr = forecast_data.get("forecast", np.array([]))
        forecast_dates = forecast_data.get("forecast_dates", pd.DatetimeIndex([]))

        rows = []

        # Historical actuals
        if not monthly_actuals.empty:
            for idx, row in monthly_actuals.iterrows():
                rows.append({
                    "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
                    "type": "actual",
                    "claim_count": int(row.iloc[0]),
                })

        # Forecast
        for i, dt in enumerate(forecast_dates):
            rows.append({
                "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "type": "forecast",
                "claim_count": round(float(forecast_arr[i]), 1),
            })

        if not rows:
            raise PreventUpdate

        export_df = pd.DataFrame(rows)
        return dcc.send_data_frame(export_df.to_csv, "forecast_data.csv", index=False)

    # ------------------------------------------------------------------
    # Download anomaly list CSV
    # ------------------------------------------------------------------
    @app.callback(
        Output("forecast-anomalies-download", "data"),
        Input("forecast-anomalies-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _download_anomalies(n_clicks):
        if not n_clicks:
            raise PreventUpdate

        anomaly_claims = data_store.anomalies.get("anomaly_claims", pd.DataFrame())
        if anomaly_claims.empty:
            raise PreventUpdate

        export_cols = [
            "claim_id", "incident_date", "paid_amount", "severity_level",
            "days_to_close", "days_to_report", "specialty", "region",
            "claim_type", "litigation_flag", "anomaly_score",
        ]
        cols = [c for c in export_cols if c in anomaly_claims.columns]
        return dcc.send_data_frame(
            anomaly_claims[cols].to_csv, "anomaly_claims.csv", index=False
        )
