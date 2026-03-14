"""Reusable KPI card component for the insurance claims dashboard."""

from __future__ import annotations

from typing import List, Optional

from dash import dcc, html
import plotly.graph_objects as go


def create_kpi_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: Optional[str] = None,
    sparkline_data: Optional[List[float]] = None,
) -> html.Div:
    """Build a professional KPI card.

    Parameters
    ----------
    title : str
        Short descriptor shown above the value (e.g. "Total Claims").
    value : str
        The primary metric, already formatted (e.g. "$1.2M").
    delta : str, optional
        Change indicator text (e.g. "+12.3%").
    delta_color : str, optional
        CSS colour for the delta text.  Common values: ``"green"`` for
        favourable, ``"red"`` for unfavourable.  Defaults to the theme
        accent colour when omitted.
    sparkline_data : list[float], optional
        A short series of values rendered as a tiny trend line inside the
        card.  Omit to hide the sparkline.

    Returns
    -------
    html.Div
        A styled card component.
    """
    children = [
        html.P(title, className="kpi-title"),
        html.H3(value, className="kpi-value"),
    ]

    # Optional delta indicator
    if delta is not None:
        # Determine arrow direction from the delta string
        if delta.startswith("+"):
            arrow = "\u25B2 "  # up triangle
        elif delta.startswith("-"):
            arrow = "\u25BC "  # down triangle
        else:
            arrow = ""

        color = delta_color or "#0f3460"
        children.append(
            html.Span(
                f"{arrow}{delta}",
                className="kpi-delta",
                style={"color": color},
            )
        )

    # Optional sparkline
    if sparkline_data and len(sparkline_data) >= 2:
        fig = go.Figure(
            go.Scatter(
                y=sparkline_data,
                mode="lines",
                line=dict(color="#0f3460", width=2),
                fill="tozeroy",
                fillcolor="rgba(15, 52, 96, 0.08)",
            )
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=40,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False,
        )
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)

        children.append(
            dcc.Graph(
                figure=fig,
                config={"displayModeBar": False},
                className="kpi-sparkline",
            )
        )

    return html.Div(children, className="kpi-card")
