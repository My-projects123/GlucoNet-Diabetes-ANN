"""Leakage-safe preprocessing for the diabetes pipeline.

Imputation, scaling, feature selection, and resampling are sklearn/imblearn
estimators so they can be fitted on training folds only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from diabetes_prediction.config import (
    FEATURE_COLUMNS,
    RANDOM_SEED,
    SELECT_K,
    ZERO_AS_MISSING_COLUMNS,
)


class ZeroAsMissing(BaseEstimator, TransformerMixin):
    """Replace 0 with NaN only in columns where 0 is not a valid value.

    Pregnancies is never included: a value of 0 means 'never pregnant'.
    """

    def __init__(self, columns: list[str] | None = None):
        # Keep the constructor argument unchanged so sklearn.clone() works.
        self.columns = columns

    def fit(self, X, y=None):
        frame = self._to_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = int(self.feature_names_in_.shape[0])
        self.columns_ = (
            list(self.columns)
            if self.columns is not None
            else list(ZERO_AS_MISSING_COLUMNS)
        )
        return self

    def transform(self, X):
        if not hasattr(self, "columns_"):
            self.fit(X)
        frame = self._to_frame(X).copy()
        for column in self.columns_:
            if column in frame.columns:
                numeric = pd.to_numeric(frame[column], errors="coerce")
                frame[column] = numeric.mask(numeric == 0)
        return frame

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray(input_features, dtype=object)

    def _to_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        columns = getattr(self, "feature_names_in_", None)
        if columns is None:
            array = np.asarray(X)
            columns = [f"x{i}" for i in range(array.shape[1])]
        return pd.DataFrame(X, columns=columns)


def build_pipeline(
    estimator: BaseEstimator,
    *,
    feature_selection: bool = False,
    k: int = SELECT_K,
    resample: bool = False,
    random_state: int = RANDOM_SEED,
    zero_as_missing_columns: list[str] | None = None,
) -> ImbPipeline:
    """Build a full model pipeline.

    Step order:
      1. Recode medically implausible zeros as missing
      2. Median impute (fit on train fold only)
      3. Optional ANOVA F-test feature selection (train fold only)
      4. Standardize
      5. Optional random oversampling of the minority class (train fold only)
      6. Classifier
    """
    n_features = len(FEATURE_COLUMNS)
    k_use = max(1, min(int(k), n_features))
    steps: list[tuple[str, Any]] = [
        (
            "zero_as_missing",
            ZeroAsMissing(columns=zero_as_missing_columns),
        ),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if feature_selection:
        steps.append(("select", SelectKBest(score_func=f_classif, k=k_use)))
    else:
        steps.append(("select", "passthrough"))
    steps.append(("scale", StandardScaler()))
    if resample:
        steps.append(
            (
                "resample",
                RandomOverSampler(sampling_strategy="auto", random_state=random_state),
            )
        )
    steps.append(("clf", clone(estimator)))
    return ImbPipeline(steps)


def selected_feature_names(pipeline: ImbPipeline, feature_names: list[str] | None = None) -> list[str]:
    """Return the feature names kept by SelectKBest, or all names if unused."""
    names = list(feature_names) if feature_names is not None else list(FEATURE_COLUMNS)
    select = pipeline.named_steps.get("select")
    if select is None or select == "passthrough":
        return names
    mask = select.get_support()
    incoming = getattr(select, "feature_names_in_", None)
    if incoming is not None:
        names = [str(n) for n in incoming]
    return [name for name, kept in zip(names, mask) if kept]
