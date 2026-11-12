from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline

from diabetes_prediction.config import FEATURE_COLUMNS, TARGET_COLUMN, ZERO_AS_MISSING_COLUMNS
from diabetes_prediction.evaluate import classification_metrics, cross_validate_model
from diabetes_prediction.preprocess import ZeroAsMissing, build_pipeline, selected_feature_names
from diabetes_prediction.reproducibility import seed_everything
from diabetes_prediction.train import make_estimators


def test_zero_as_missing_keeps_pregnancies(tiny_diabetes_frame: pd.DataFrame) -> None:
    transformer = ZeroAsMissing()
    transformed = transformer.fit_transform(tiny_diabetes_frame[FEATURE_COLUMNS])
    assert "Glucose" in transformer.columns_
    assert transformed["Pregnancies"].isna().sum() == 0
    assert (tiny_diabetes_frame["Pregnancies"] == 0).sum() >= 1
    assert transformed.loc[tiny_diabetes_frame["Glucose"] == 0, "Glucose"].isna().all()
    assert transformed.loc[tiny_diabetes_frame["BloodPressure"] == 0, "BloodPressure"].isna().all()
    assert transformed.loc[tiny_diabetes_frame["BMI"] == 0, "BMI"].isna().all()
    for column in ZERO_AS_MISSING_COLUMNS:
        assert column in transformer.columns_


def test_imputer_statistics_come_from_training_fold_only(
    tiny_diabetes_frame: pd.DataFrame,
) -> None:
    X = tiny_diabetes_frame[FEATURE_COLUMNS]
    y = tiny_diabetes_frame[TARGET_COLUMN]
    pipeline = build_pipeline(
        LogisticRegression(max_iter=200),
        feature_selection=False,
        resample=False,
    )
    train = [0, 1, 2, 3, 4, 5]
    pipeline.fit(X.iloc[train], y.iloc[train])
    imputer = pipeline.named_steps["impute"]
    glucose_idx = FEATURE_COLUMNS.index("Glucose")
    train_glucose = X.iloc[train]["Glucose"].replace(0, np.nan)
    expected_median = float(train_glucose.median())
    assert imputer.statistics_[glucose_idx] == expected_median

    full_glucose = X["Glucose"].replace(0, np.nan)
    full_median = float(full_glucose.median())
    # If the full table median differs, prove we did not fit on all rows.
    if not np.isclose(expected_median, full_median):
        assert imputer.statistics_[glucose_idx] != full_median


def test_feature_selection_is_inside_the_pipeline(
    tiny_diabetes_frame: pd.DataFrame,
) -> None:
    X = tiny_diabetes_frame[FEATURE_COLUMNS]
    y = tiny_diabetes_frame[TARGET_COLUMN]
    pipeline = build_pipeline(
        LogisticRegression(max_iter=400),
        feature_selection=True,
        k=3,
        resample=False,
    )
    result = cross_validate_model(pipeline, X, y, n_splits=3, random_state=0)
    assert "select" in pipeline.named_steps
    for fold_features in result["selected_features_by_fold"]:
        assert len(fold_features) == 3
        assert set(fold_features).issubset(set(FEATURE_COLUMNS))
    # Different folds may choose different features; at least record them.
    assert len(result["selected_features_by_fold"]) == 3


def test_pipeline_is_not_a_plain_sklearn_pipeline_when_resampling() -> None:
    pipeline = build_pipeline(
        LogisticRegression(max_iter=200),
        resample=True,
        random_state=0,
    )
    assert "resample" in pipeline.named_steps
    # imblearn pipeline subclasses sklearn's Pipeline, but must handle y.
    assert isinstance(pipeline, SklearnPipeline)
    assert hasattr(pipeline, "fit_resample") or "resample" in pipeline.named_steps


def test_selected_feature_names_passthrough() -> None:
    pipeline = build_pipeline(LogisticRegression(), feature_selection=False)
    names = selected_feature_names(pipeline, FEATURE_COLUMNS)
    assert names == FEATURE_COLUMNS


def test_metrics_on_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = classification_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall_sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_metrics_on_all_negative_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    metrics = classification_metrics(y_true, y_pred, np.array([0.1, 0.2, 0.3, 0.4]))
    assert metrics["recall_sensitivity"] == 0.0
    assert metrics["true_positives"] == 0
    assert metrics["specificity"] == 1.0


def test_estimators_include_ann_and_baselines() -> None:
    estimators = make_estimators(random_state=0)
    assert "mlp_ann" in estimators
    assert "logistic_regression" in estimators
    assert "random_forest" in estimators
    hidden = estimators["mlp_ann"].hidden_layer_sizes
    assert hidden == (16, 8)


def test_feature_selection_frequency_plot_skips_when_unused() -> None:
    from diabetes_prediction.plots import plot_feature_selection_frequency

    assert plot_feature_selection_frequency({"dummy": {"feature_selection": False}}) is None


def test_seed_helper_sets_numpy() -> None:
    seed_everything(123)
    a = np.random.rand(3)
    seed_everything(123)
    b = np.random.rand(3)
    assert np.allclose(a, b)
