"""Metrics and stratified cross-validation for binary diabetes classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from diabetes_prediction.config import FEATURE_COLUMNS, N_SPLITS, RANDOM_SEED
from diabetes_prediction.preprocess import selected_feature_names


METRIC_ORDER = [
    "accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
]


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the required classification metrics for one set of predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    if y_proba is not None and len(np.unique(y_true)) >= 2:
        proba = np.asarray(y_proba, dtype=float)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, proba))
        except ValueError:
            metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def predict_proba_positive(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=float)
        # Convert unbounded scores to (0, 1) for PR/ROC plotting only.
        return 1.0 / (1.0 + np.exp(-scores))
    pred = np.asarray(estimator.predict(X), dtype=float)
    return pred


def cross_validate_model(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_SEED,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Stratified CV with out-of-fold predictions. Fits a clone per fold."""
    names = list(feature_names) if feature_names is not None else list(FEATURE_COLUMNS)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_array = np.asarray(y).astype(int)
    oof_pred = np.zeros(len(y_array), dtype=int)
    oof_proba = np.zeros(len(y_array), dtype=float)
    fold_rows: list[dict[str, Any]] = []
    selected_by_fold: list[list[str]] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_array), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        model = clone(pipeline)
        model.fit(X_train, y_train)
        pred = np.asarray(model.predict(X_test)).astype(int)
        proba = predict_proba_positive(model, X_test)
        oof_pred[test_idx] = pred
        oof_proba[test_idx] = proba
        metrics = classification_metrics(y_test, pred, proba)
        metrics["fold"] = fold
        fold_rows.append(metrics)
        selected_by_fold.append(selected_feature_names(model, names))

    oof_metrics = classification_metrics(y_array, oof_pred, oof_proba)
    fold_frame = pd.DataFrame(fold_rows)
    summary = {
        metric: {
            "mean": float(fold_frame[metric].mean()),
            "std": float(fold_frame[metric].std(ddof=1)) if len(fold_frame) > 1 else 0.0,
        }
        for metric in METRIC_ORDER
    }
    return {
        "n_splits": n_splits,
        "random_state": random_state,
        "fold_metrics": fold_rows,
        "summary": summary,
        "oof_metrics": oof_metrics,
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
        "y_true": y_array,
        "selected_features_by_fold": selected_by_fold,
        "confusion_matrix_oof": confusion_matrix(
            y_array, oof_pred, labels=[0, 1]
        ).tolist(),
    }


def mean_std_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per experiment with mean ± std for each metric."""
    rows = []
    for name, result in results.items():
        row: dict[str, Any] = {"experiment": name}
        for metric in METRIC_ORDER:
            stats = result["summary"][metric]
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        row["oof_roc_auc"] = result["oof_metrics"]["roc_auc"]
        row["oof_pr_auc"] = result["oof_metrics"]["pr_auc"]
        rows.append(row)
    return pd.DataFrame(rows)


def format_metric(mean: float, std: float) -> str:
    if mean != mean:  # NaN
        return "NA"
    return f"{mean:.3f} ± {std:.3f}"
