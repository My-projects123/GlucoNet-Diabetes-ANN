"""Model definitions and the leakage-safe comparison experiment."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from diabetes_prediction.config import (
    FEATURE_COLUMNS,
    N_SPLITS,
    RANDOM_SEED,
    SELECT_K,
)
from diabetes_prediction.evaluate import cross_validate_model
from diabetes_prediction.preprocess import build_pipeline


def make_estimators(random_state: int = RANDOM_SEED) -> dict[str, Any]:
    """Small, fixed-hyperparameter models. No test-set tuning.

    The MLP is a two-hidden-layer feed-forward network. It is kept small
    because the dataset has only a few hundred rows.
    """
    return {
        "dummy_most_frequent": DummyClassifier(
            strategy="most_frequent",
            random_state=random_state,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=1,
            random_state=random_state,
        ),
        "mlp_ann": MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=32,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=random_state,
        ),
    }


def experiment_grid(
    *,
    include_dummy: bool = True,
    feature_selection_modes: tuple[bool, ...] = (False, True),
    resample_modes: tuple[bool, ...] = (False, True),
    random_state: int = RANDOM_SEED,
    k: int = SELECT_K,
    estimators: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Configs compared in the main run.

    DummyClassifier is a sanity check only (always predict the majority
    class). The fair comparison is logistic regression vs random forest vs MLP.
    """
    estimators = estimators if estimators is not None else make_estimators(random_state)
    configs = []
    for name, estimator in estimators.items():
        if name == "dummy_most_frequent":
            if not include_dummy:
                continue
            configs.append(
                {
                    "name": "dummy_most_frequent",
                    "estimator": estimator,
                    "feature_selection": False,
                    "resample": False,
                    "role": "sanity_baseline",
                }
            )
            continue
        for use_select in feature_selection_modes:
            for use_resample in resample_modes:
                select_tag = f"kbest{k}" if use_select else "all_features"
                resample_tag = "oversample" if use_resample else "no_resample"
                configs.append(
                    {
                        "name": f"{name}__{select_tag}__{resample_tag}",
                        "estimator": estimator,
                        "feature_selection": use_select,
                        "resample": use_resample,
                        "role": "comparison",
                    }
                )
    return configs


def run_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_SEED,
    k: int = SELECT_K,
    include_dummy: bool = True,
    feature_selection_modes: tuple[bool, ...] = (False, True),
    resample_modes: tuple[bool, ...] = (False, True),
    estimators: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit every config with stratified CV. Selection/resample stay inside folds."""
    results: dict[str, Any] = {}
    configs = experiment_grid(
        include_dummy=include_dummy,
        feature_selection_modes=feature_selection_modes,
        resample_modes=resample_modes,
        random_state=random_state,
        k=k,
        estimators=estimators,
    )
    for config in configs:
        pipeline = build_pipeline(
            config["estimator"],
            feature_selection=config["feature_selection"],
            k=k,
            resample=config["resample"],
            random_state=random_state,
        )
        cv_result = cross_validate_model(
            pipeline,
            X,
            y,
            n_splits=n_splits,
            random_state=random_state,
            feature_names=list(FEATURE_COLUMNS),
        )
        cv_result["role"] = config["role"]
        cv_result["feature_selection"] = config["feature_selection"]
        cv_result["resample"] = config["resample"]
        results[config["name"]] = cv_result
    return results
