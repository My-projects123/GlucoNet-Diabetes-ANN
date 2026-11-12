"""Exploratory and evaluation plots. Saved under reports/figures/."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from diabetes_prediction.config import (
    DISCLAIMER,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ZERO_AS_MISSING_COLUMNS,
    figures_dir,
)


sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_outcome_balance(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or figures_dir()
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["No diabetes (0)", "Diabetes (1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C78A8", "#F58518"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Number of rows")
    ax.set_title("Outcome class balance")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
        )
    fig.text(0.01, 0.01, DISCLAIMER, fontsize=7, wrap=True)
    return _save(fig, out_dir / "outcome_balance.png")


def plot_feature_distributions(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or figures_dir()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    plotted = df.copy()
    plotted["Outcome label"] = plotted[TARGET_COLUMN].map(
        {0: "No diabetes", 1: "Diabetes"}
    )
    for ax, column in zip(axes.ravel(), FEATURE_COLUMNS):
        sns.histplot(
            data=plotted,
            x=column,
            hue="Outcome label",
            bins=20,
            kde=True,
            element="step",
            ax=ax,
            palette=["#4C78A8", "#F58518"],
            legend=False,
        )
        ax.set_title(column)
    handles = [
        plt.Line2D([0], [0], color="#4C78A8", lw=3, label="No diabetes"),
        plt.Line2D([0], [0], color="#F58518", lw=3, label="Diabetes"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Feature distributions by outcome", y=1.02)
    fig.text(0.01, 0.01, DISCLAIMER, fontsize=7, wrap=True)
    return _save(fig, out_dir / "feature_distributions.png")


def plot_outliers(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or figures_dir()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    plotted = df.copy()
    plotted["Outcome label"] = plotted[TARGET_COLUMN].map(
        {0: "No diabetes", 1: "Diabetes"}
    )
    for ax, column in zip(axes.ravel(), FEATURE_COLUMNS):
        sns.boxplot(
            data=plotted,
            x="Outcome label",
            y=column,
            hue="Outcome label",
            ax=ax,
            palette=["#4C78A8", "#F58518"],
            legend=False,
        )
        ax.set_xlabel("")
        ax.set_title(column)
    fig.suptitle("Outliers and spread by outcome (raw values)")
    fig.text(0.01, 0.01, DISCLAIMER, fontsize=7, wrap=True)
    return _save(fig, out_dir / "outliers_boxplots.png")


def plot_correlations(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or figures_dir()
    corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        ax=ax,
        square=True,
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Pearson correlations (raw values)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.text(0.01, 0.01, DISCLAIMER, fontsize=7, wrap=True)
    return _save(fig, out_dir / "feature_correlations.png")


def plot_zero_counts(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or figures_dir()
    zeros = [(col, int((df[col] == 0).sum())) for col in FEATURE_COLUMNS]
    labels, values = zip(*zeros)
    display_labels = [
        "DPF" if col == "DiabetesPedigreeFunction" else col for col in labels
    ]
    colors = [
        "#E45756" if col in ZERO_AS_MISSING_COLUMNS else "#4C78A8" for col in labels
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(display_labels, values, color=colors)
    ax.set_ylabel("Zero count")
    ax.set_title("Raw zero counts (red = treated as missing; DPF = pedigree function)")
    ax.tick_params(axis="x", rotation=25)
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.text(0.01, 0.01, DISCLAIMER, fontsize=7, wrap=True)
    return _save(fig, out_dir / "zero_counts.png")


def plot_confusion_matrices(
    results: dict[str, dict],
    out_dir: Path | None = None,
    names: list[str] | None = None,
) -> Path:
    out_dir = out_dir or figures_dir()
    names = names or list(results)
    n = len(names)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows))
    axes_list = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes_list, names):
        y_true = results[name]["y_true"]
        y_pred = results[name]["oof_pred"]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["No diabetes", "Diabetes"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name.replace("__", "\n"), fontsize=10)
    for ax in axes_list[n:]:
        ax.axis("off")
    fig.suptitle("Out-of-fold confusion matrices")
    fig.text(0.01, 0.01, DISCLAIMER, fontsize=7, wrap=True)
    return _save(fig, out_dir / "confusion_matrices.png")


def plot_roc_pr(
    results: dict[str, dict],
    out_dir: Path | None = None,
    names: list[str] | None = None,
) -> tuple[Path, Path]:
    out_dir = out_dir or figures_dir()
    names = names or list(results)

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for name in names:
        y_true = results[name]["y_true"]
        proba = results[name]["oof_proba"]
        RocCurveDisplay.from_predictions(y_true, proba, name=name, ax=ax_roc)
        PrecisionRecallDisplay.from_predictions(y_true, proba, name=name, ax=ax_pr)
    ax_roc.set_title("Out-of-fold ROC curves")
    ax_pr.set_title("Out-of-fold precision-recall curves")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    prevalence = float(np.mean(results[names[0]]["y_true"]))
    ax_pr.axhline(prevalence, linestyle="--", color="gray", linewidth=1, label="prevalence")
    fig_roc.text(0.01, 0.01, DISCLAIMER, fontsize=7, wrap=True)
    fig_pr.text(0.01, 0.01, DISCLAIMER, fontsize=7, wrap=True)
    roc_path = _save(fig_roc, out_dir / "roc_curves.png")
    pr_path = _save(fig_pr, out_dir / "pr_curves.png")
    return roc_path, pr_path


def plot_metric_bars(
    table: pd.DataFrame,
    out_dir: Path | None = None,
    metric: str = "roc_auc",
) -> Path:
    out_dir = out_dir or figures_dir()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(table))))
    means = table[f"{metric}_mean"]
    stds = table[f"{metric}_std"]
    labels = table["experiment"]
    y_pos = np.arange(len(table))
    ax.barh(y_pos, means, xerr=stds, color="#4C78A8", alpha=0.85, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"{metric} (mean ± std over folds)")
    ax.set_xlim(0, 1)
    ax.set_title(f"Stratified CV {metric}")
    ax.invert_yaxis()
    fig.text(0.01, 0.01, DISCLAIMER, fontsize=7, wrap=True)
    return _save(fig, out_dir / f"cv_{metric}.png")


def plot_feature_selection_frequency(
    results: dict[str, dict],
    out_dir: Path | None = None,
) -> Path | None:
    """Bar chart of how often each feature was kept by SelectKBest across folds."""
    out_dir = out_dir or figures_dir()
    counts: Counter[str] = Counter()
    n_folds = 0
    for result in results.values():
        if not result.get("feature_selection"):
            continue
        for feats in result.get("selected_features_by_fold", []):
            n_folds += 1
            counts.update(feats)
    if not counts or n_folds == 0:
        return None

    labels = [name for name, _ in counts.most_common()]
    values = [counts[name] / n_folds for name in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of training folds that kept the feature")
    ax.set_title("SelectKBest frequency (fit inside each training fold)")
    ax.tick_params(axis="x", rotation=25)
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.text(0.01, 0.01, DISCLAIMER, fontsize=7, wrap=True)
    return _save(fig, out_dir / "feature_selection_frequency.png")


def write_eda_plots(df: pd.DataFrame, out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or figures_dir()
    return [
        plot_outcome_balance(df, out_dir),
        plot_feature_distributions(df, out_dir),
        plot_outliers(df, out_dir),
        plot_correlations(df, out_dir),
        plot_zero_counts(df, out_dir),
    ]
