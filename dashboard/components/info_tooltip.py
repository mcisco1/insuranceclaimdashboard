"""Info tooltip component for methodology explanations."""

from __future__ import annotations

from dash import html


def create_info_tooltip(text: str, tooltip_id: str) -> html.Span:
    """Render an (i) icon with a tooltip on hover.

    Parameters
    ----------
    text : str
        Methodology explanation text shown on hover.
    tooltip_id : str
        Unique identifier for the tooltip element.

    Returns
    -------
    html.Span
        A styled info icon with a CSS-only tooltip.
    """
    return html.Span(
        [
            html.Span("i", className="info-icon", id=tooltip_id),
            html.Span(text, className="info-tooltip-text"),
        ],
        className="info-tooltip-wrapper",
    )
