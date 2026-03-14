"""Reusable download button component for the insurance claims dashboard."""

from __future__ import annotations

from dash import dcc, html


def create_download_button(
    button_id: str,
    label: str = "Download CSV",
) -> html.Div:
    """Create a download button paired with a ``dcc.Download`` trigger.

    Parameters
    ----------
    button_id : str
        Unique identifier prefix.  The button element receives
        ``id="{button_id}-btn"`` and the ``dcc.Download`` component
        receives ``id="{button_id}-download"``.
    label : str
        Button label text.

    Returns
    -------
    html.Div
        A container holding the visible button and the hidden
        ``dcc.Download`` component.  Page-level callbacks should wire
        the button's ``n_clicks`` to the download component's ``data``
        property.
    """
    return html.Div(
        [
            html.Button(
                label,
                id=f"{button_id}-btn",
                className="download-btn",
            ),
            dcc.Download(id=f"{button_id}-download"),
        ],
        className="download-container",
    )
