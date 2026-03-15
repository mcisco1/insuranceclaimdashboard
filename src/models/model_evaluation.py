"""ML model evaluation framework: cross-validation, calibration, and comparison.

Provides stratified k-fold cross-validation, calibration curve data,
and statistical model comparison using paired t-tests.

Usage
-----
    from src.models.model_evaluation import evaluate_model, compare_models
    results = compare_models(X, y, models_dict)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy import stats

from config import RANDOM_SEED

logger = logging.getLogger(__name__)


def evaluate_model(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
    cv: int = 5,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Evaluate a single model pipeline using stratified k-fold CV.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Unfitted model pipeline.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    name : str
        Human-readable model name.
    cv : int
        Number of cross-validation folds.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``name``, ``cv_f1_scores``, ``cv_f1_mean``, ``cv_f1_std``,
        ``calibration_data``.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    f1_scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)

    # Fit on full data for calibration curve
    pipeline.fit(X, y)
    if hasattr(pipeline, "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X)
            # For multi-class, use one-vs-rest average calibration
            classes = pipeline.classes_
            n_classes = len(classes)
            cal_fractions = []
            cal_means = []

            for c_idx in range(min(n_classes, 5)):
                y_binary = (y == classes[c_idx]).astype(int)
                prob_c = y_prob[:, c_idx]
                try:
                    fraction, mean_pred = calibration_curve(
                        y_binary, prob_c, n_bins=10, strategy="uniform"
                    )
                    cal_fractions.append(fraction.tolist())
                    cal_means.append(mean_pred.tolist())
                except Exception:
                    pass

            calibration_data = {
                "fractions": cal_fractions,
                "mean_predicted": cal_means,
                "classes": [int(c) for c in classes[:5]],
            }
        except Exception:
            calibration_data = {}
    else:
        calibration_data = {}

    return {
        "name": name,
        "cv_f1_scores": f1_scores.tolist(),
        "cv_f1_mean": float(f1_scores.mean()),
        "cv_f1_std": float(f1_scores.std()),
        "calibration_data": calibration_data,
    }


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    models_dict: Dict[str, Any],
    cv: int = 5,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Evaluate and compare multiple models.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target.
    models_dict : dict
        Mapping of model name to unfitted sklearn Pipeline.
    cv : int
        Number of CV folds.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``rankings``, ``cv_results``, ``calibration_curves``,
        ``statistical_test``.
    """
    cv_results = {}
    calibration_curves = {}

    for name, pipeline in models_dict.items():
        result = evaluate_model(pipeline, X, y, name, cv=cv, seed=seed)
        cv_results[name] = result
        if result["calibration_data"]:
            calibration_curves[name] = result["calibration_data"]

    # Rank by F1-macro mean
    rankings = sorted(
        cv_results.values(),
        key=lambda r: r["cv_f1_mean"],
        reverse=True,
    )
    ranking_list = [
        {"rank": i + 1, "name": r["name"], "f1_mean": r["cv_f1_mean"], "f1_std": r["cv_f1_std"]}
        for i, r in enumerate(rankings)
    ]

    # Paired t-test between top 2 models
    statistical_test = {}
    if len(rankings) >= 2:
        top1 = rankings[0]
        top2 = rankings[1]
        scores1 = np.array(top1["cv_f1_scores"])
        scores2 = np.array(top2["cv_f1_scores"])
        if len(scores1) == len(scores2) and len(scores1) > 1:
            t_stat, p_val = stats.ttest_rel(scores1, scores2)
            statistical_test = {
                "model_a": top1["name"],
                "model_b": top2["name"],
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "significant": p_val < 0.05,
            }

    logger.info(
        "Model comparison complete. Top model: %s (F1=%.4f +/- %.4f)",
        rankings[0]["name"] if rankings else "N/A",
        rankings[0]["cv_f1_mean"] if rankings else 0,
        rankings[0]["cv_f1_std"] if rankings else 0,
    )

    return {
        "rankings": ranking_list,
        "cv_results": {name: {k: v for k, v in r.items() if k != "calibration_data"} for name, r in cv_results.items()},
        "calibration_curves": calibration_curves,
        "statistical_test": statistical_test,
    }
