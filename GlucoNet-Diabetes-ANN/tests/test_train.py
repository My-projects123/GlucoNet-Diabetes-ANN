from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from diabetes_prediction.config import FEATURE_COLUMNS, TARGET_COLUMN
from diabetes_prediction.evaluate import METRIC_ORDER
from diabetes_prediction.train import experiment_grid, run_comparison


def test_experiment_names_use_requested_k() -> None:
    configs = experiment_grid(
        k=3,
        resample_modes=(False,),
        feature_selection_modes=(True,),
        random_state=0,
    )
    names = [config["name"] for config in configs]
    assert any("kbest3" in name for name in names)
    assert not any("kbest5" in name for name in names)


def test_run_comparison_on_tiny_data(tiny_diabetes_frame) -> None:
    """Full comparison on synthetic rows; MLP skips early stopping so n is tiny."""
    X = tiny_diabetes_frame[FEATURE_COLUMNS]
    y = tiny_diabetes_frame[TARGET_COLUMN]
    estimators = {
        "dummy_most_frequent": DummyClassifier(
            strategy="most_frequent", random_state=0
        ),
        "logistic_regression": LogisticRegression(
            max_iter=400, solver="liblinear", random_state=0
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=8, max_depth=3, random_state=0
        ),
        "mlp_ann": MLPClassifier(
            hidden_layer_sizes=(8,),
            max_iter=250,
            random_state=0,
        ),
    }
    results = run_comparison(
        X,
        y,
        n_splits=2,
        random_state=0,
        k=3,
        include_dummy=True,
        feature_selection_modes=(False, True),
        resample_modes=(False,),
        estimators=estimators,
    )
    assert "dummy_most_frequent" in results
    assert any(name.startswith("mlp_ann") for name in results)
    assert any("kbest3" in name for name in results)
    for result in results.values():
        for metric in METRIC_ORDER:
            assert metric in result["summary"]
            mean = result["summary"][metric]["mean"]
            assert mean == mean  # not NaN for this balanced toy table
        if result.get("feature_selection"):
            for feats in result["selected_features_by_fold"]:
                assert len(feats) == 3
