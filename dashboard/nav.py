"""Sidebar navigation component for the insurance claims dashboard."""

from __future__ import annotations

from dash import dcc, html, callback
from dash.dependencies import Input, Output


# ── Navigation structure ──────────────────────────────────────────────────────

_NAV_ITEMS = [
    {"path": "/", "label": "Executive Overview", "icon": "H"},
    {"path": "/loss-trends", "label": "Loss Trends", "icon": "T"},
    {"path": "/geographic-risk", "label": "Geographic & Department Risk", "icon": "G"},
    {"path": "/operational-efficiency", "label": "Operational Efficiency", "icon": "O"},
    {"path": "/loss-development", "label": "Loss Development", "icon": "L"},
    {"path": "/forecasting", "label": "Forecasting & Anomalies", "icon": "F"},
    {"path": "/scenario-analysis", "label": "Scenario Analysis", "icon": "S"},
    {"path": "/recommendations", "label": "Recommendations", "icon": "R"},
    {"path": "/data-quality", "label": "Data Quality", "icon": "Q"},
]


# ── Sidebar builder ──────────────────────────────────────────────────────────

def create_sidebar() -> html.Div:
    """Return a styled sidebar ``html.Div`` with navigation links.

    The sidebar includes the dashboard title/logo area at the top and a
    vertical list of ``dcc.Link`` elements for each page.  Active page
    highlighting is handled by a Dash callback that adds an ``active``
    CSS class to the current link.
    """
    nav_links = []
    for item in _NAV_ITEMS:
        nav_links.append(
            dcc.Link(
                html.Div(
                    [
                        html.Span(item["icon"], className="nav-icon"),
                        html.Span(item["label"], className="nav-label"),
                    ],
                    className="nav-link-inner",
                ),
                href=item["path"],
                className="nav-link",
                id=f"nav-link-{item['path'].strip('/') or 'home'}",
            )
        )

    sidebar = html.Div(
        [
            # Logo / title area
            html.Div(
                [
                    html.H2("Claims Risk", className="sidebar-title"),
                    html.P("Intelligence Dashboard", className="sidebar-subtitle"),
                ],
                className="sidebar-header",
            ),
            html.Hr(className="sidebar-divider"),
            # Navigation
            html.Nav(nav_links, className="sidebar-nav"),
            # Footer
            html.Div(
                html.P(
                    "v1.0  |  Insurance Analytics",
                    className="sidebar-footer-text",
                ),
                className="sidebar-footer",
            ),
        ],
        className="sidebar",
    )

    return sidebar


# ── Active-page highlighting callback ────────────────────────────────────────

@callback(
    [
        Output(f"nav-link-{item['path'].strip('/') or 'home'}", "className")
        for item in _NAV_ITEMS
    ],
    Input("url", "pathname"),
)
def _highlight_active_link(pathname: str):
    """Set the ``active`` CSS class on the link matching the current URL."""
    classes = []
    for item in _NAV_ITEMS:
        if pathname == item["path"]:
            classes.append("nav-link active")
        else:
            classes.append("nav-link")
    return classes
