"""Dashboard page modules and URL-to-module mapping.

``PAGE_MODULES`` maps each URL path to its corresponding page module.
This central registry is consumed by ``dashboard.app`` for routing and
callback registration, so adding a new page only requires an entry here.
"""

from dashboard.pages import (
    executive_overview,
    loss_trends,
    geographic_risk,
    operational_efficiency,
    loss_development,
    forecasting_anomalies,
    scenario_analysis,
    recommendations,
)

PAGE_MODULES = {
    '/': executive_overview,
    '/loss-trends': loss_trends,
    '/geographic-risk': geographic_risk,
    '/operational-efficiency': operational_efficiency,
    '/loss-development': loss_development,
    '/forecasting': forecasting_anomalies,
    '/scenario-analysis': scenario_analysis,
    '/recommendations': recommendations,
}
