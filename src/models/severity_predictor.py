"""Severity prediction models for insurance claims.

Trains RandomForest and LogisticRegression classifiers to predict claim
severity level (1-5) from engineered features.  Designed to consume the
claims-summary DataFrame produced by the dashboard's data pipeline.

Usage
-----
    from src.models.severity_predictor import train_severity_model
    results = train_severity_model(df)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED, TEST_SIZE
from src.models.model_evaluation import compare_models

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

_TOP_N_SPECIALTIES = 10
_SEVERITY_COL_CANDIDATES = ("severity_level", "severity")
_SPECIALTY_COL_CANDIDATES = ("specialty", "specialty_name")


# ── Feature Engineering ─────────────────────────────────────────────────

def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    """Return the first column name from *candidates* that exists in *df*."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """Build the feature matrix from a claims DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        The feature matrix and the ordered list of feature names.
    """
    feat = pd.DataFrame(index=df.index)

    # ── claim_type (one-hot) ─────────────────────────────────────────
    if "claim_type" in df.columns:
        ct_dummies = pd.get_dummies(df["claim_type"], prefix="claim_type")
        feat = pd.concat([feat, ct_dummies], axis=1)

    # ── specialty (one-hot, top N + "other") ─────────────────────────
    spec_col = _resolve_column(df, _SPECIALTY_COL_CANDIDATES)
    if spec_col is not None:
        spec = df[spec_col].copy()
        top_specs = spec.value_counts().nlargest(_TOP_N_SPECIALTIES).index
        spec = spec.where(spec.isin(top_specs), other="other")
        spec_dummies = pd.get_dummies(spec, prefix="specialty")
        feat = pd.concat([feat, spec_dummies], axis=1)

    # ── region (one-hot) ─────────────────────────────────────────────
    if "region" in df.columns:
        region_dummies = pd.get_dummies(df["region"], prefix="region")
        feat = pd.concat([feat, region_dummies], axis=1)

    # ── patient_age_band (one-hot) ───────────────────────────────────
    if "patient_age_band" in df.columns:
        age_dummies = pd.get_dummies(df["patient_age_band"], prefix="age_band")
        feat = pd.concat([feat, age_dummies], axis=1)

    # ── paid_amount (log-transformed) ────────────────────────────────
    if "paid_amount" in df.columns:
        paid = pd.to_numeric(df["paid_amount"], errors="coerce").fillna(0)
        feat["log_paid_amount"] = np.log1p(paid.clip(lower=0))

    # ── days_to_report ───────────────────────────────────────────────
    report_col = _resolve_column(df, ("days_to_report", "report_lag_days"))
    if report_col is not None:
        feat["days_to_report"] = (
            pd.to_numeric(df[report_col], errors="coerce").fillna(0)
        )

    # ── litigation_flag (binary) ─────────────────────────────────────
    if "litigation_flag" in df.columns:
        feat["litigation_flag"] = (
            pd.to_numeric(df["litigation_flag"], errors="coerce").fillna(0).astype(int)
        )

    # ── repeat_event_flag (binary) ───────────────────────────────────
    if "repeat_event_flag" in df.columns:
        feat["repeat_event_flag"] = (
            pd.to_numeric(df["repeat_event_flag"], errors="coerce").fillna(0).astype(int)
        )

    # Ensure all columns are numeric (safety net for any object leftovers)
    feat = feat.astype(float)
    feature_names = list(feat.columns)

    return feat, feature_names


# ── Filtering ────────────────────────────────────────────────────────────

def _filter_valid_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only closed claims with a valid severity level in [1, 5]."""
    severity_col = _resolve_column(df, _SEVERITY_COL_CANDIDATES)
    if severity_col is None:
        raise ValueError(
            f"DataFrame must contain one of {_SEVERITY_COL_CANDIDATES} columns."
        )

    out = df.copy()

    # Standardise column name to severity_level
    if severity_col != "severity_level":
        out = out.rename(columns={severity_col: "severity_level"})

    out["severity_level"] = pd.to_numeric(out["severity_level"], errors="coerce")

    # Filter to closed claims
    if "status" in out.columns:
        out = out[out["status"].str.lower() == "closed"]

    # Keep only valid severity values 1-5
    out = out[out["severity_level"].isin([1, 2, 3, 4, 5])].copy()
    out["severity_level"] = out["severity_level"].astype(int)

    return out


# ── Main Training Function ──────────────────────────────────────────────

def train_severity_model(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Train severity prediction models on claims data.

    Parameters
    ----------
    df : pd.DataFrame
        Claims-summary DataFrame.  Expected columns include
        *severity_level* (or *severity*), *status*, *claim_type*,
        *specialty* (or *specialty_name*), *region*,
        *patient_age_band*, *paid_amount*, *days_to_report*
        (or *report_lag_days*), *litigation_flag*, *repeat_event_flag*.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys:
        - ``rf_model``: trained RandomForestClassifier
        - ``lr_model``: trained LogisticRegression
        - ``rf_accuracy``: float -- test-set accuracy (RF)
        - ``lr_accuracy``: float -- test-set accuracy (LR)
        - ``rf_report``: dict -- sklearn classification_report (RF)
        - ``lr_report``: dict -- sklearn classification_report (LR)
        - ``feature_importances``: DataFrame of feature names + importances
        - ``confusion_matrix``: ndarray -- RF confusion matrix
        - ``features_used``: list of feature names
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # ── Filter to valid, closed claims ───────────────────────────────
    filtered = _filter_valid_claims(df)
    if len(filtered) < 50:
        raise ValueError(
            f"Only {len(filtered)} valid closed claims with severity 1-5. "
            "Need at least 50 to train a meaningful model."
        )

    logger.info("Training on %d closed claims with valid severity.", len(filtered))

    # ── Engineer features ────────────────────────────────────────────
    X, feature_names = _engineer_features(filtered)
    y = filtered["severity_level"]

    if X.shape[1] == 0:
        raise ValueError(
            "No features could be engineered.  Check that the DataFrame "
            "contains the expected columns (claim_type, specialty, etc.)."
        )

    # ── Train / test split ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y,
    )

    # ── Random Forest pipeline ───────────────────────────────────────
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )),
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    rf_accuracy = float((rf_preds == y_test).mean())
    rf_cls_report = classification_report(
        y_test, rf_preds, output_dict=True, zero_division=0,
    )
    rf_cm = confusion_matrix(y_test, rf_preds, labels=[1, 2, 3, 4, 5])

    # ── Logistic Regression pipeline ─────────────────────────────────
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        )),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    lr_accuracy = float((lr_preds == y_test).mean())
    lr_cls_report = classification_report(
        y_test, lr_preds, output_dict=True, zero_division=0,
    )

    # ── Feature importances (from RF) ────────────────────────────────
    rf_clf = rf_pipeline.named_steps["clf"]
    importances = rf_clf.feature_importances_
    importance_df = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # ── Gradient Boosting pipeline ─────────────────────────────────
    gbt_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=seed,
        )),
    ])
    gbt_pipeline.fit(X_train, y_train)
    gbt_preds = gbt_pipeline.predict(X_test)
    gbt_accuracy = float((gbt_preds == y_test).mean())
    gbt_cls_report = classification_report(
        y_test, gbt_preds, output_dict=True, zero_division=0,
    )

    logger.info(
        "RF accuracy: %.4f | LR accuracy: %.4f | GBT accuracy: %.4f",
        rf_accuracy, lr_accuracy, gbt_accuracy,
    )

    # ── Cross-validated model comparison ───────────────────────────
    comparison_models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=15,
                min_samples_split=5, min_samples_leaf=2,
                class_weight="balanced", random_state=seed, n_jobs=-1,
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, solver="lbfgs",
                class_weight="balanced", random_state=seed,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1,
                min_samples_split=5, min_samples_leaf=2, random_state=seed,
            )),
        ]),
    }

    model_comparison = compare_models(X, y, comparison_models, cv=5, seed=seed)

    return {
        "rf_model": rf_pipeline,
        "lr_model": lr_pipeline,
        "gbt_model": gbt_pipeline,
        "rf_accuracy": rf_accuracy,
        "lr_accuracy": lr_accuracy,
        "gbt_accuracy": gbt_accuracy,
        "rf_report": rf_cls_report,
        "lr_report": lr_cls_report,
        "gbt_report": gbt_cls_report,
        "feature_importances": importance_df,
        "confusion_matrix": rf_cm,
        "features_used": feature_names,
        "model_comparison": model_comparison,
    }
