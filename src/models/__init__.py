"""ML models for the insurance claims dashboard.

Provides severity prediction, time-series forecasting, and anomaly
detection.  Each module accepts a claims-summary DataFrame and returns
a dict of results that the dashboard can consume directly.
"""

from src.models.severity_predictor import train_severity_model
from src.models.forecasting import forecast_claims
from src.models.anomaly_detection import detect_anomalies
from src.models.model_evaluation import evaluate_model, compare_models

__all__ = [
    "train_severity_model",
    "forecast_claims",
    "detect_anomalies",
    "evaluate_model",
    "compare_models",
]
