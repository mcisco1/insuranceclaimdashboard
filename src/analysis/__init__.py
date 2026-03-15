"""Analysis modules for the insurance claims dashboard.

Each module accepts a claims-summary DataFrame (from v_claims_summary)
and returns a dict of results that the dashboard can consume directly.
"""

from src.analysis.frequency_severity import analyze_frequency_severity
from src.analysis.time_to_close import analyze_time_to_close
from src.analysis.loss_development import build_loss_triangle
from src.analysis.reporting_lag import analyze_reporting_lag
from src.analysis.repeat_incidents import analyze_repeat_incidents
from src.analysis.benchmarks import calculate_benchmarks
from src.analysis.scenario_analysis import run_scenarios
from src.analysis.statistical_utils import bootstrap_ci, proportional_ci
from src.analysis.survival_analysis import analyze_survival

__all__ = [
    "analyze_frequency_severity",
    "analyze_time_to_close",
    "build_loss_triangle",
    "analyze_reporting_lag",
    "analyze_repeat_incidents",
    "calculate_benchmarks",
    "run_scenarios",
    "bootstrap_ci",
    "proportional_ci",
    "analyze_survival",
]
