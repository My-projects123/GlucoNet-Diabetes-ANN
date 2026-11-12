"""Final-review experiments for the Diabetes Prediction System.

Runs, on the local Pima table only:
  1. the original seeded 5-fold comparison (same configs as run_experiment.py);
  2. 10 x 5 repeated stratified CV with corrected resampled t-tests;
  3. an ablation study of the proposed ANN pipeline;
  4. leakage experiments (what happens when preprocessing escapes the fold);
  5. nested vs non-nested hyperparameter selection;
  6. seed stability, learning curve, loss curve, threshold sweep;
  7. a single stratified 80/20 hold-out test with bootstrap intervals and
     permutation importance.

Every learned step (imputation, selection, scaling, oversampling) is fitted on
training rows only, except in the experiments explicitly labelled "leaky".

Usage:
    cd diabetes_ann && .venv/bin/python final_report/experiments.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from diabetes_prediction.config import (  # noqa: E402
    FEATURE_COLUMNS,
    RANDOM_SEED,
    ZERO_AS_MISSING_COLUMNS,
)
from diabetes_prediction.data import load_dataset, split_xy  # noqa: E402
from diabetes_prediction.evaluate import (  # noqa: E402
    METRIC_ORDER,
    classification_metrics,
    predict_proba_positive,
)
from diabetes_prediction.preprocess import ZeroAsMissing  # noqa: E402
from diabetes_prediction.reproducibility import configure_runtime  # noqa: E402
from diabetes_prediction.train import run_comparison  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CV_MAIN = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
CV_ABLATION = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)
CV_REPEATED = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_SEED)


# --------------------------------------------------------------------------
# Model and pipeline factories
# --------------------------------------------------------------------------

MLP_DEFAULTS: dict[str, Any] = dict(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    alpha=1e-3,
    batch_size=32,
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=RANDOM_SEED,
)


def make_mlp(**overrides: Any) -> MLPClassifier:
    params = dict(MLP_DEFAULTS)
    params.update(overrides)
    return MLPClassifier(**params)


def make_lr() -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="liblinear", random_state=RANDOM_SEED
    )


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=1,
        random_state=RANDOM_SEED,
    )


def make_pipeline(
    clf,
    *,
    zero_cols: list[str] | None = None,
    imputer: str = "median",
    select_k: int | None = None,
    scale: bool = True,
    resample: str | None = "ros",
    seed: int = RANDOM_SEED,
) -> ImbPipeline:
    """Proposed pipeline by default: recode -> median -> scale -> ROS -> clf."""
    zero_cols = list(ZERO_AS_MISSING_COLUMNS) if zero_cols is None else list(zero_cols)
    steps: list[tuple[str, Any]] = []
    if zero_cols:
        steps.append(("zero_as_missing", ZeroAsMissing(columns=zero_cols)))
    if imputer == "median":
        steps.append(("impute", SimpleImputer(strategy="median")))
    elif imputer == "mean":
        steps.append(("impute", SimpleImputer(strategy="mean")))
    elif imputer == "knn":
        steps.append(("impute", KNNImputer(n_neighbors=5)))
    else:
        raise ValueError(imputer)
    if select_k:
        steps.append(("select", SelectKBest(score_func=f_classif, k=select_k)))
    if scale:
        steps.append(("scale", StandardScaler()))
    if resample == "ros":
        steps.append(("resample", RandomOverSampler(random_state=seed)))
    elif resample == "smote":
        steps.append(("resample", SMOTE(random_state=seed)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------


def cv_rows(pipe, X: pd.DataFrame, y: np.ndarray, cv) -> list[dict[str, float]]:
    rows = []
    for split, (tr, te) in enumerate(cv.split(X, y)):
        model = clone(pipe)
        model.fit(X.iloc[tr], y[tr])
        pred = np.asarray(model.predict(X.iloc[te])).astype(int)
        proba = predict_proba_positive(model, X.iloc[te])
        metrics = classification_metrics(y[te], pred, proba)
        metrics["split"] = split
        metrics["n_train"] = int(len(tr))
        metrics["n_test"] = int(len(te))
        rows.append(metrics)
    return rows


def cv_oof(pipe, X: pd.DataFrame, y: np.ndarray, cv) -> tuple[np.ndarray, np.ndarray]:
    proba = np.zeros(len(y))
    pred = np.zeros(len(y), dtype=int)
    for tr, te in cv.split(X, y):
        model = clone(pipe)
        model.fit(X.iloc[tr], y[tr])
        pred[te] = np.asarray(model.predict(X.iloc[te])).astype(int)
        proba[te] = predict_proba_positive(model, X.iloc[te])
    return pred, proba


def summarise(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(rows)
    return {
        metric: {
            "mean": float(frame[metric].mean()),
            "std": float(frame[metric].std(ddof=1)),
        }
        for metric in METRIC_ORDER
    }


def corrected_t_test(a: list[float], b: list[float], n_train: int, n_test: int) -> dict[str, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated CV."""
    diff = np.asarray(a) - np.asarray(b)
    j = len(diff)
    mean = float(diff.mean())
    var = float(diff.var(ddof=1))
    denom = np.sqrt((1.0 / j + n_test / n_train) * var)
    t = mean / denom if denom > 0 else 0.0
    p = float(2 * stats.t.sf(abs(t), df=j - 1))
    return {"mean_diff": mean, "t": float(t), "p": p, "df": j - 1}


def log(msg: str) -> None:
    print(msg, flush=True)


SHORT_METRIC = {
    "accuracy": "Acc",
    "precision": "Prec",
    "recall_sensitivity": "Sens",
    "specificity": "Spec",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


def fmt(summary: dict[str, dict[str, float]], metric: str) -> str:
    mean, std = summary[metric]["mean"], summary[metric]["std"]
    return f"{mean:.3f}±{std:.3f}".replace("0.", ".")


def short_main_name(name: str) -> str:
    if name == "dummy_most_frequent":
        return "Dummy (majority)"
    model, feats, resample = name.split("__")
    model = {"logistic_regression": "LogReg", "random_forest": "RandForest", "mlp_ann": "MLP-ANN"}[model]
    feats = "all 8" if feats == "all_features" else "k=5"
    resample = "+ROS" if resample == "oversample" else "-"
    return f"{model:<10} {feats:<5} {resample}"


def print_table(title: str, entries: dict[str, dict[str, dict[str, float]]], width: int = 30) -> None:
    log("")
    log(title)
    header = f"{'configuration':<{width}}" + "".join(f"{SHORT_METRIC[m]:>11}" for m in METRIC_ORDER)
    log(header)
    log("-" * len(header))
    for name, summary in entries.items():
        log(f"{name[:width]:<{width}}" + "".join(f"{fmt(summary, m):>11}" for m in METRIC_ORDER))


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


def main() -> int:
    configure_runtime(RANDOM_SEED)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    t0 = time.time()
    df = load_dataset()
    X, y_series = split_xy(df)
    y = y_series.to_numpy().astype(int)
    out: dict[str, Any] = {"seed": RANDOM_SEED, "n_rows": int(len(df))}
    arrays: dict[str, np.ndarray] = {"y": y}

    log("Diabetes Prediction System — final experiments")
    log(f"Loaded {len(df)} rows x {len(FEATURE_COLUMNS)} predictors from the local CSV.")
    log(f"Class counts: 0 -> {(y == 0).sum()}, 1 -> {(y == 1).sum()}  (seed {RANDOM_SEED})")

    # 1. Original seeded 5-fold comparison (identical configs to run_experiment.py)
    log("\n[1/8] Seeded 5-fold comparison (13 configurations)...")
    main_results = run_comparison(X, y_series)
    out["main_cv"] = {}
    for name, res in main_results.items():
        out["main_cv"][name] = {
            "summary": res["summary"],
            "oof_metrics": {k: float(v) for k, v in res["oof_metrics"].items()},
            "confusion_matrix_oof": res["confusion_matrix_oof"],
            "selected_features_by_fold": res["selected_features_by_fold"],
            "fold_metrics": [
                {k: float(v) for k, v in row.items()} for row in res["fold_metrics"]
            ],
        }
        arrays[f"oof_proba__{name}"] = res["oof_proba"]
        arrays[f"oof_pred__{name}"] = res["oof_pred"]
    print_table(
        "5-fold CV (mean±std over folds)",
        {short_main_name(k): v["summary"] for k, v in out["main_cv"].items()},
    )

    # 2. Repeated CV with corrected t-tests
    log("\n[2/8] 10 x 5 repeated stratified CV for the main models...")
    repeated_models = {
        "Dummy (majority)": ImbPipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        "Logistic regression": make_pipeline(make_lr(), resample=None),
        "Random forest": make_pipeline(make_rf(), resample=None),
        "MLP ANN (no rebalancing)": make_pipeline(make_mlp(), resample=None),
        "Proposed MLP ANN (+ROS)": make_pipeline(make_mlp(), resample="ros"),
    }
    out["repeated_cv"] = {}
    rep_rows: dict[str, list[dict[str, float]]] = {}
    for name, pipe in repeated_models.items():
        rows = cv_rows(pipe, X, y, CV_REPEATED)
        rep_rows[name] = rows
        out["repeated_cv"][name] = {"summary": summarise(rows), "n_splits": len(rows)}
    print_table(
        "Repeated CV, 50 train/test splits (mean±std)",
        {k: v["summary"] for k, v in out["repeated_cv"].items()},
    )
    n_train = rep_rows["Proposed MLP ANN (+ROS)"][0]["n_train"]
    n_test = rep_rows["Proposed MLP ANN (+ROS)"][0]["n_test"]
    tests: dict[str, Any] = {}
    for other in ["Logistic regression", "Random forest", "MLP ANN (no rebalancing)"]:
        tests[other] = {}
        for metric in ["roc_auc", "pr_auc", "f1", "recall_sensitivity", "accuracy"]:
            a = [r[metric] for r in rep_rows["Proposed MLP ANN (+ROS)"]]
            b = [r[metric] for r in rep_rows[other]]
            tests[other][metric] = corrected_t_test(a, b, n_train, n_test)
    out["corrected_t_tests"] = tests
    log("\nCorrected resampled t-tests, proposed minus comparator (difference, p-value):")
    for other, per_metric in tests.items():
        cells = "  ".join(
            f"{SHORT_METRIC[m]} {per_metric[m]['mean_diff']:+.3f} (p={per_metric[m]['p']:.3f})"
            for m in ["roc_auc", "f1", "recall_sensitivity"]
        )
        log(f"  vs {other:<25} {cells}")

    # 3. Ablation study (3 x 5-fold, same splits for every row)
    log("\n[3/8] Ablation study (15 splits per configuration)...")
    ablation_specs: dict[str, dict[str, Any]] = {}

    def add(group: str, label: str, pipe) -> None:
        ablation_specs[f"{group}::{label}"] = {"group": group, "label": label, "pipe": pipe}

    proposed = make_pipeline(make_mlp())
    add("component", "Proposed (full pipeline)", proposed)
    add("component", "− zero recoding (zeros kept)", make_pipeline(make_mlp(), zero_cols=[]))
    add(
        "component",
        "Recode Pregnancies = 0 as well",
        make_pipeline(make_mlp(), zero_cols=["Pregnancies"] + list(ZERO_AS_MISSING_COLUMNS)),
    )
    add("component", "Median → mean imputation", make_pipeline(make_mlp(), imputer="mean"))
    add("component", "Median → KNN imputation", make_pipeline(make_mlp(), imputer="knn"))
    add("component", "− standard scaling", make_pipeline(make_mlp(), scale=False))
    add("component", "− oversampling (imbalanced)", make_pipeline(make_mlp(), resample=None))
    add("component", "ROS → SMOTE", make_pipeline(make_mlp(), resample="smote"))
    add("component", "+ SelectKBest (k = 5)", make_pipeline(make_mlp(), select_k=5))
    add(
        "component",
        "− early stopping",
        make_pipeline(make_mlp(early_stopping=False, max_iter=500)),
    )
    for hidden in [(8,), (16,), (32,), (16, 8), (32, 16), (64, 32), (64, 32, 16)]:
        label = " → ".join(str(h) for h in hidden)
        add("architecture", label, make_pipeline(make_mlp(hidden_layer_sizes=hidden)))
    for act in ["relu", "tanh", "logistic"]:
        add("activation", act, make_pipeline(make_mlp(activation=act)))
    add("optimiser", "Adam", make_pipeline(make_mlp()))
    add(
        "optimiser",
        "SGD + momentum",
        make_pipeline(make_mlp(solver="sgd", learning_rate_init=0.01, momentum=0.9)),
    )
    add("optimiser", "L-BFGS", make_pipeline(make_mlp(solver="lbfgs", max_iter=500)))
    for alpha in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        add("alpha", f"{alpha:g}", make_pipeline(make_mlp(alpha=alpha)))
    for k in [3, 4, 5, 6, 7, 8]:
        add("k_features", str(k), make_pipeline(make_mlp(), select_k=k))

    out["ablation"] = {}
    cache: dict[str, list[dict[str, float]]] = {}
    for key, spec in ablation_specs.items():
        cache_key = repr(spec["pipe"].get_params(deep=True).get("steps"))
        if cache_key not in cache:
            cache[cache_key] = cv_rows(spec["pipe"], X, y, CV_ABLATION)
        rows = cache[cache_key]
        out["ablation"][key] = {
            "group": spec["group"],
            "label": spec["label"],
            "summary": summarise(rows),
        }
    for group in ["component", "architecture", "activation", "optimiser", "alpha", "k_features"]:
        print_table(
            f"Ablation — {group}",
            {
                v["label"]: v["summary"]
                for v in out["ablation"].values()
                if v["group"] == group
            },
        )

    # 4. Leakage experiments
    log("\n[4/8] Leakage experiments...")
    leakage: dict[str, Any] = {}
    for model_name, clf_factory in [("MLP", make_mlp), ("RF", make_rf)]:
        safe = make_pipeline(clf_factory())
        leakage[f"{model_name}::Leakage-safe (proposed)"] = summarise(
            cv_rows(safe, X, y, CV_ABLATION)
        )

        pre = Pipeline(
            [
                ("zero_as_missing", ZeroAsMissing(columns=list(ZERO_AS_MISSING_COLUMNS))),
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        ).fit(X)
        X_pre = pd.DataFrame(pre.transform(X), columns=FEATURE_COLUMNS)
        leaky_pre = ImbPipeline(
            [("resample", RandomOverSampler(random_state=RANDOM_SEED)), ("clf", clf_factory())]
        )
        leakage[f"{model_name}::Impute+scale on all rows"] = summarise(
            cv_rows(leaky_pre, X_pre, y, CV_ABLATION)
        )

        selector = SelectKBest(f_classif, k=5).fit(X_pre, y)
        X_sel = pd.DataFrame(selector.transform(X_pre))
        leakage[f"{model_name}::+SelectKBest on all rows"] = summarise(
            cv_rows(leaky_pre, X_sel, y, CV_ABLATION)
        )

        X_os, y_os = RandomOverSampler(random_state=RANDOM_SEED).fit_resample(X, y)
        X_os = pd.DataFrame(X_os, columns=FEATURE_COLUMNS)
        no_ros = make_pipeline(clf_factory(), resample=None)
        leakage[f"{model_name}::Oversample before CV"] = summarise(
            cv_rows(no_ros, X_os, np.asarray(y_os), CV_ABLATION)
        )

        X_pre_sm, y_sm = SMOTE(random_state=RANDOM_SEED).fit_resample(X_pre, y)
        no_ros_pre = ImbPipeline([("clf", clf_factory())])
        leakage[f"{model_name}::SMOTE before CV"] = summarise(
            cv_rows(no_ros_pre, pd.DataFrame(X_pre_sm), np.asarray(y_sm), CV_ABLATION)
        )
    out["leakage"] = leakage
    print_table(
        "Leakage experiments (15 splits each)",
        {k.replace("::", ": "): v for k, v in leakage.items()},
    )

    # 5. Nested vs non-nested hyperparameter selection
    log("\n[5/8] Nested vs non-nested hyperparameter selection (ROC-AUC)...")
    grid = {
        "clf__hidden_layer_sizes": [(8,), (16,), (16, 8), (32, 16)],
        "clf__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    }
    flat_scores = {}
    for hidden in grid["clf__hidden_layer_sizes"]:
        for alpha in grid["clf__alpha"]:
            pipe = make_pipeline(make_mlp(hidden_layer_sizes=hidden, alpha=alpha))
            rows = cv_rows(pipe, X, y, CV_MAIN)
            flat_scores[f"{hidden}|{alpha:g}"] = float(np.mean([r["roc_auc"] for r in rows]))
    best_key = max(flat_scores, key=flat_scores.get)
    nested_scores, chosen = [], []
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + 1)
    for tr, te in CV_MAIN.split(X, y):
        search = GridSearchCV(make_pipeline(make_mlp()), grid, scoring="roc_auc", cv=inner, n_jobs=1)
        search.fit(X.iloc[tr], y[tr])
        proba = predict_proba_positive(search.best_estimator_, X.iloc[te])
        nested_scores.append(float(roc_auc_score(y[te], proba)))
        chosen.append(
            {
                "hidden": str(search.best_params_["clf__hidden_layer_sizes"]),
                "alpha": float(search.best_params_["clf__alpha"]),
                "inner_auc": float(search.best_score_),
            }
        )
    out["nested"] = {
        "grid_scores_non_nested": flat_scores,
        "best_non_nested_key": best_key,
        "best_non_nested_auc": flat_scores[best_key],
        "nested_auc_mean": float(np.mean(nested_scores)),
        "nested_auc_std": float(np.std(nested_scores, ddof=1)),
        "nested_fold_auc": nested_scores,
        "chosen_per_outer_fold": chosen,
    }
    log(f"  Best non-nested grid point {best_key}: ROC-AUC {flat_scores[best_key]:.3f}")
    log(
        f"  Nested CV estimate: ROC-AUC {np.mean(nested_scores):.3f} ± "
        f"{np.std(nested_scores, ddof=1):.3f}"
    )
    for i, c in enumerate(chosen, 1):
        log(f"  outer fold {i}: chose hidden={c['hidden']} alpha={c['alpha']:g}")

    # 6. Seed stability
    log("\n[6/8] Seed stability of the proposed ANN (10 seeds, fixed folds)...")
    seed_rows = []
    for s in range(10):
        pipe = make_pipeline(make_mlp(random_state=s), seed=s)
        summ = summarise(cv_rows(pipe, X, y, CV_MAIN))
        seed_rows.append({m: summ[m]["mean"] for m in METRIC_ORDER} | {"seed": s})
    seed_frame = pd.DataFrame(seed_rows)
    out["seed_stability"] = {
        "per_seed": seed_rows,
        "summary": {
            m: {"mean": float(seed_frame[m].mean()), "std": float(seed_frame[m].std(ddof=1)),
                "min": float(seed_frame[m].min()), "max": float(seed_frame[m].max())}
            for m in METRIC_ORDER
        },
    }
    for m in METRIC_ORDER:
        s = out["seed_stability"]["summary"][m]
        log(
            f"  {SHORT_METRIC[m]:<8} mean {s['mean']:.3f}  std {s['std']:.3f}  "
            f"range [{s['min']:.3f}, {s['max']:.3f}]"
        )

    # 7. Learning curve, loss curve, threshold sweep
    log("\n[7/8] Learning curve, loss curve, and threshold sweep...")
    sizes, train_sc, test_sc = learning_curve(
        make_pipeline(make_mlp()),
        X,
        y,
        cv=CV_MAIN,
        scoring="roc_auc",
        train_sizes=np.linspace(0.2, 1.0, 6),
        shuffle=True,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    out["learning_curve"] = {
        "train_sizes": sizes.tolist(),
        "train_mean": train_sc.mean(axis=1).tolist(),
        "train_std": train_sc.std(axis=1).tolist(),
        "test_mean": test_sc.mean(axis=1).tolist(),
        "test_std": test_sc.std(axis=1).tolist(),
    }
    for n, a, b in zip(sizes, train_sc.mean(axis=1), test_sc.mean(axis=1)):
        log(f"  n_train={n:4d}  train ROC-AUC {a:.3f}  validation ROC-AUC {b:.3f}")

    tr, te = next(iter(CV_MAIN.split(X, y)))
    fold_model = make_pipeline(make_mlp()).fit(X.iloc[tr], y[tr])
    clf = fold_model.named_steps["clf"]
    out["loss_curve"] = {
        "loss": [float(v) for v in clf.loss_curve_],
        "val_accuracy": [float(v) for v in clf.validation_scores_],
        "n_iter": int(clf.n_iter_),
        "best_val_accuracy": float(clf.best_validation_score_),
        "n_parameters": int(sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_)),
    }
    log(
        f"  Fold-1 training stopped after {clf.n_iter_} epochs; "
        f"trainable parameters = {out['loss_curve']['n_parameters']}"
    )

    proposed_name = "mlp_ann__all_features__oversample"
    oof = arrays[f"oof_proba__{proposed_name}"]
    sweep = []
    for thr in np.round(np.arange(0.05, 0.951, 0.025), 3):
        pred = (oof >= thr).astype(int)
        m = classification_metrics(y, pred, oof)
        sweep.append({"threshold": float(thr), **{k: float(m[k]) for k in
                      ["accuracy", "precision", "recall_sensitivity", "specificity", "f1"]}})
    sweep_frame = pd.DataFrame(sweep)
    sweep_frame["youden_j"] = sweep_frame["recall_sensitivity"] + sweep_frame["specificity"] - 1
    best = sweep_frame.loc[sweep_frame["youden_j"].idxmax()]
    out["threshold_sweep"] = {"rows": sweep_frame.to_dict(orient="records"),
                              "youden_best": best.to_dict()}
    log(
        f"  Youden-optimal OOF threshold {best['threshold']:.3f}: sensitivity "
        f"{best['recall_sensitivity']:.3f}, specificity {best['specificity']:.3f}"
    )

    # 8. Hold-out test
    log("\n[8/8] Stratified 80/20 hold-out test (fit once, test once)...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    holdout_models = {
        "Logistic regression": make_pipeline(make_lr(), resample=None),
        "Random forest": make_pipeline(make_rf(), resample=None),
        "MLP ANN (no rebalancing)": make_pipeline(make_mlp(), resample=None),
        "Proposed MLP ANN (+ROS)": make_pipeline(make_mlp(), resample="ros"),
    }
    out["holdout"] = {"n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                      "test_positives": int(y_te.sum()), "models": {}}
    rng = np.random.default_rng(RANDOM_SEED)
    boot_idx = [rng.integers(0, len(y_te), len(y_te)) for _ in range(2000)]
    fitted = {}
    for name, pipe in holdout_models.items():
        model = clone(pipe).fit(X_tr, y_tr)
        fitted[name] = model
        pred = np.asarray(model.predict(X_te)).astype(int)
        proba = predict_proba_positive(model, X_te)
        metrics = classification_metrics(y_te, pred, proba)
        boots = {"accuracy": [], "roc_auc": [], "recall_sensitivity": [], "specificity": [], "f1": []}
        for idx in boot_idx:
            if len(np.unique(y_te[idx])) < 2:
                continue
            bm = classification_metrics(y_te[idx], pred[idx], proba[idx])
            for k in boots:
                boots[k].append(bm[k])
        ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
        fpr, tpr, _ = roc_curve(y_te, proba)
        prec, rec, _ = precision_recall_curve(y_te, proba)
        out["holdout"]["models"][name] = {
            "metrics": {k: float(v) for k, v in metrics.items()},
            "ci95": ci,
            "confusion_matrix": confusion_matrix(y_te, pred, labels=[0, 1]).tolist(),
            "report": classification_report(
                y_te, pred, target_names=["No diabetes", "Diabetes"], digits=3
            ),
        }
        arrays[f"holdout_proba__{name}"] = proba
        arrays[f"holdout_roc__{name}"] = np.vstack([fpr, tpr])
        arrays[f"holdout_pr__{name}"] = np.vstack([rec, prec])
    arrays["holdout_y"] = y_te
    for name, res in out["holdout"]["models"].items():
        m = res["metrics"]
        log(
            f"  {name:<25} Acc {m['accuracy']:.3f}  Sens {m['recall_sensitivity']:.3f}  "
            f"Spec {m['specificity']:.3f}  F1 {m['f1']:.3f}  ROC {m['roc_auc']:.3f}  "
            f"PR {m['pr_auc']:.3f}"
        )
    log("\nClassification report — proposed MLP ANN on the 154-row hold-out test set:")
    log(out["holdout"]["models"]["Proposed MLP ANN (+ROS)"]["report"])

    perm = permutation_importance(
        fitted["Proposed MLP ANN (+ROS)"],
        X_te,
        y_te,
        scoring="roc_auc",
        n_repeats=30,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    out["permutation_importance"] = {
        feat: {"mean": float(perm.importances_mean[i]), "std": float(perm.importances_std[i])}
        for i, feat in enumerate(FEATURE_COLUMNS)
    }
    log("Permutation importance (mean drop in hold-out ROC-AUC, 30 repeats):")
    for feat, v in sorted(out["permutation_importance"].items(), key=lambda kv: -kv[1]["mean"]):
        log(f"  {feat:<26} {v['mean']:+.4f} ± {v['std']:.4f}")

    model_path = RESULTS / "proposed_mlp_pipeline.joblib"
    joblib.dump(
        {
            "pipeline": fitted["Proposed MLP ANN (+ROS)"],
            "feature_columns": FEATURE_COLUMNS,
            "test_index": X_te.index.to_list(),
        },
        model_path,
    )

    out["runtime_seconds"] = round(time.time() - t0, 1)
    (RESULTS / "final_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez_compressed(RESULTS / "final_arrays.npz", **arrays)
    log(f"\nSaved {RESULTS / 'final_results.json'}")
    log(f"Saved {model_path}")
    log(f"Total runtime: {out['runtime_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
