"""Horizontal filter panel component for the insurance claims dashboard."""

from __future__ import annotations

from typing import List

from dash import dcc, html


def create_filter_panel(
    years: List[int],
    claim_types: List[str],
    severities: List[int],
    regions: List[str],
) -> html.Div:
    """Build a compact horizontal filter bar.

    Each dropdown is a Dash ``dcc.Dropdown`` with multi-select enabled.  A
    *Reset Filters* button clears all selections.

    Parameters
    ----------
    years : list[int]
        Available accident / incident years.
    claim_types : list[str]
        Distinct claim-type labels.
    severities : list[int]
        Distinct severity levels (typically 1-5).
    regions : list[str]
        Distinct region names.

    Returns
    -------
    html.Div
        A styled filter bar ready to embed at the top of any page.
    """
    year_options = [{"label": str(y), "value": y} for y in sorted(years)]
    type_options = [{"label": t, "value": t} for t in sorted(claim_types)]
    sev_options = [
        {"label": f"Level {s}", "value": s} for s in sorted(severities)
    ]
    region_options = [{"label": r, "value": r} for r in sorted(regions)]

    return html.Div(
        [
            # Year selector
            html.Div(
                [
                    html.Label("Year", className="filter-label"),
                    dcc.Dropdown(
                        id="filter-year",
                        options=year_options,
                        multi=True,
                        placeholder="All Years",
                        className="filter-dropdown",
                    ),
                ],
                className="filter-group",
            ),
            # Claim type selector
            html.Div(
                [
                    html.Label("Claim Type", className="filter-label"),
                    dcc.Dropdown(
                        id="filter-claim-type",
                        options=type_options,
                        multi=True,
                        placeholder="All Types",
                        className="filter-dropdown",
                    ),
                ],
                className="filter-group",
            ),
            # Severity selector
            html.Div(
                [
                    html.Label("Severity", className="filter-label"),
                    dcc.Dropdown(
                        id="filter-severity",
                        options=sev_options,
                        multi=True,
                        placeholder="All Levels",
                        className="filter-dropdown",
                    ),
                ],
                className="filter-group",
            ),
            # Region selector
            html.Div(
                [
                    html.Label("Region", className="filter-label"),
                    dcc.Dropdown(
                        id="filter-region",
                        options=region_options,
                        multi=True,
                        placeholder="All Regions",
                        className="filter-dropdown",
                    ),
                ],
                className="filter-group",
            ),
            # Reset button
            html.Div(
                html.Button(
                    "Reset Filters",
                    id="filter-reset-btn",
                    className="filter-reset-btn",
                ),
                className="filter-group filter-group-btn",
            ),
        ],
        className="filter-panel",
    )
