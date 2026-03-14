"""ML models for the insurance claims dashboard.

Provides severity prediction, time-series forecasting, and anomaly
detection.  Each module accepts a claims-summary DataFrame and returns
a dict of results that the dashboard can consume directly.
"""

from src.models.severity_predictor import train_severity_model
from src.models.forecasting import forecast_claims
from src.models.anomaly_detection import detect_anomalies

__all__ = [
    "train_severity_model",
    "forecast_claims",
    "detect_anomalies",
]
