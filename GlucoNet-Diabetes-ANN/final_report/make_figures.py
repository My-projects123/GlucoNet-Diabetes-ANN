"""Figures for the final report and slides.

Reads results/final_results.json and results/final_arrays.npz written by
experiments.py, plus the local CSV for EDA. Writes PNGs to figures/.

Usage:
    cd diabetes_ann && .venv/bin/python final_report/make_figures.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402
from sklearn.feature_selection import f_classif  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from diabetes_prediction.config import (  # noqa: E402
    FEATURE_COLUMNS,
    RANDOM_SEED,
    ZERO_AS_MISSING_COLUMNS,
)
from diabetes_prediction.data import load_dataset  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results"

NAVY = "#0E2A47"
TEAL = "#1F6F8B"
ACCENT = "#C45C26"
GOLD = "#C9A227"
GREEN = "#4F7F3A"
GREY = "#9AA5B1"
LIGHT = "#E8EEF4"
RED = "#A32D2D"

MODEL_COLORS = {
    "Dummy (majority)": "#BBBBBB",
    "Logistic regression": TEAL,
    "Random forest": GREEN,
    "MLP ANN (no rebalancing)": GREY,
    "Proposed MLP ANN (+ROS)": ACCENT,
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    }
)

PRETTY = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose (mg/dL)",
    "BloodPressure": "Diastolic BP (mm Hg)",
    "SkinThickness": "Triceps skinfold (mm)",
    "Insulin": "2-h insulin (µU/mL)",
    "BMI": "BMI (kg/m²)",
    "DiabetesPedigreeFunction": "Pedigree function",
    "Age": "Age (years)",
}
SHORT = {
    "Pregnancies": "Preg",
    "Glucose": "Glucose",
    "BloodPressure": "BP",
    "SkinThickness": "Skin",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "DPF",
    "Age": "Age",
}


def save(fig, name: str) -> None:
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"wrote figures/{name}")


def recoded(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ZERO_AS_MISSING_COLUMNS:
        out[c] = out[c].mask(out[c] == 0)
    return out


# --------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------


def fig_class_balance(df):
    counts = df["Outcome"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bars = ax.bar(["No diabetes (0)", "Diabetes (1)"], counts.values, color=[TEAL, ACCENT], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v} ({v / len(df):.1%})", ha="center", fontsize=11)
    ax.axhline(len(df) / 2, color="k", ls=":", lw=1)
    ax.text(1.32, len(df) / 2 + 6, "balanced = 384", fontsize=9, ha="right")
    ax.set_ylabel("Number of rows")
    ax.set_ylim(0, 580)
    ax.set_title("Outcome class balance (n = 768)")
    save(fig, "fig_class_balance.png")


def fig_zero_counts(df):
    zeros = (df[FEATURE_COLUMNS] == 0).sum()
    colors = [ACCENT if c in ZERO_AS_MISSING_COLUMNS else TEAL for c in FEATURE_COLUMNS]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar([SHORT[c] for c in FEATURE_COLUMNS], zeros.values, color=colors)
    for b, v in zip(bars, zeros.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 6, f"{v}\n({v / len(df):.1%})", ha="center", fontsize=9)
    ax.set_ylabel("Rows with value 0")
    ax.set_ylim(0, 450)
    ax.set_title("Zero values per feature in the raw file")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color=ACCENT, label="0 is impossible → recoded as missing"),
                 Patch(color=TEAL, label="0 is valid (or absent) → kept")],
        loc="upper left",
        frameon=False,
    )
    save(fig, "fig_zero_counts.png")


def fig_distributions(df):
    rc = recoded(df)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, col in zip(axes.ravel(), FEATURE_COLUMNS):
        a = rc.loc[rc.Outcome == 0, col].dropna()
        b = rc.loc[rc.Outcome == 1, col].dropna()
        bins = np.histogram_bin_edges(pd.concat([a, b]), bins=22)
        ax.hist(a, bins=bins, density=True, alpha=0.55, color=TEAL, label="No diabetes")
        ax.hist(b, bins=bins, density=True, alpha=0.55, color=ACCENT, label="Diabetes")
        ax.axvline(a.median(), color=TEAL, lw=1.4, ls="--")
        ax.axvline(b.median(), color=ACCENT, lw=1.4, ls="--")
        ax.set_title(PRETTY[col], fontsize=11)
        ax.set_yticks([])
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Feature distributions by outcome after recoding impossible zeros (dashed = class median)", y=1.0)
    fig.tight_layout()
    save(fig, "fig_feature_distributions.png")


def fig_boxplots(df):
    rc = recoded(df)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, col in zip(axes.ravel(), FEATURE_COLUMNS):
        data = [rc.loc[rc.Outcome == k, col].dropna() for k in (0, 1)]
        bp = ax.boxplot(data, labels=["0", "1"], patch_artist=True, widths=0.55,
                        flierprops=dict(marker="o", markersize=2.5, alpha=0.5))
        for patch, color in zip(bp["boxes"], [TEAL, ACCENT]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("k")
        ax.set_title(PRETTY[col], fontsize=11)
        ax.set_xlabel("Outcome")
    fig.suptitle("Box plots by outcome (missing values excluded; points beyond 1.5×IQR shown)", y=1.0)
    fig.tight_layout()
    save(fig, "fig_boxplots.png")


def fig_correlation(df):
    rc = recoded(df)
    corr = rc[FEATURE_COLUMNS + ["Outcome"]].corr()
    labels = [SHORT.get(c, c) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(v) > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Pearson correlation after recoding (pairwise complete)")
    save(fig, "fig_correlation.png")
    return corr


def fig_anova_and_selection(df, results):
    X = recoded(df)[FEATURE_COLUMNS]
    y = df["Outcome"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for tr, _ in cv.split(X, y):
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
        F, _ = f_classif(imp.transform(X.iloc[tr]), y[tr])
        scores.append(F)
    scores = np.array(scores)
    sel = results["main_cv"]["mlp_ann__kbest5__oversample"]["selected_features_by_fold"]
    freq = Counter(f for fold in sel for f in fold)

    order = np.argsort(-scores.mean(axis=0))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    names = [SHORT[FEATURE_COLUMNS[i]] for i in order]
    axes[0].bar(names, scores.mean(axis=0)[order], yerr=scores.std(axis=0)[order], color=NAVY, capsize=3)
    axes[0].set_ylabel("ANOVA F statistic")
    axes[0].set_title("ANOVA F-scores on each training fold (mean ± sd)")
    vals = [freq.get(FEATURE_COLUMNS[i], 0) for i in order]
    colors = [ACCENT if v == 5 else (GOLD if v > 0 else GREY) for v in vals]
    axes[1].bar(names, vals, color=colors)
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.08, f"{v}/5", ha="center", fontsize=10)
    axes[1].set_ylim(0, 5.8)
    axes[1].set_ylabel("Folds in which selected")
    axes[1].set_title("SelectKBest (k = 5) selection frequency")
    fig.tight_layout()
    save(fig, "fig_feature_selection.png")
    return {FEATURE_COLUMNS[i]: (float(scores.mean(axis=0)[i]), float(scores.std(axis=0)[i])) for i in range(8)}


# --------------------------------------------------------------------------
# Method diagrams
# --------------------------------------------------------------------------


def _box(ax, x, y, w, h, text, fc="white", ec=NAVY, fs=10, bold_first=True, tc="black", lw=1.4):
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                           fc=fc, ec=ec, lw=lw)
    ax.add_patch(patch)
    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        ax.text(x, y + h * 0.18, lines[0], ha="center", va="center", fontsize=fs, fontweight="bold", color=tc)
        ax.text(x, y - h * 0.14, "\n".join(lines[1:]), ha="center", va="center", fontsize=fs - 1, color=tc,
                linespacing=1.15)
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold_first else None,
                color=tc)


def _arrow(ax, p1, p2, color=NAVY, lw=1.6, label=None, label_offset=(0, 0.18), fs=9):
    ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + label_offset[0], (p1[1] + p2[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs, style="italic", color=color)


def _polyline_arrow(ax, pts, color=NAVY, lw=1.6):
    xs, ys = zip(*pts[:-1])
    ax.plot(list(xs) + [pts[-2][0]], list(ys) + [pts[-2][1]], color=color, lw=lw, solid_capstyle="round")
    _arrow(ax, pts[-2], pts[-1], color=color, lw=lw)


def fig_block_schematic():
    fig, ax = plt.subplots(figsize=(15, 8.6))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    # Stage A — data
    ya, ha = 8.35, 1.15
    _box(ax, 1.9, ya, 3.2, ha, "Local dataset\nPima CSV · 768 rows\n8 predictors + Outcome", fc=LIGHT)
    _box(ax, 5.8, ya, 3.2, ha, "Profiling & EDA\nbalance · zeros · IQR\ndistributions · correlation", fc=LIGHT)
    _box(ax, 9.7, ya, 3.2, ha, "Stratified K-fold split\nk = 5, shuffle, seed 42\n(repeated 10× for tests)", fc=LIGHT)
    _box(ax, 13.6, ya, 3.0, ha, "Test fold (≈154 rows)\nheld out: never used\nfor any fitting", fc="#FBF0D9", ec=GOLD)
    _arrow(ax, (3.5, ya), (4.2, ya))
    _arrow(ax, (7.4, ya), (8.1, ya))
    _arrow(ax, (11.3, ya), (12.1, ya), color=GOLD)

    # Stage B — training-fold pipeline
    dash = Rectangle((0.3, 4.25), 15.2, 2.75, fill=False, ls="--", lw=1.4, ec=TEAL)
    ax.add_patch(dash)
    ax.text(2.2, 6.78, "Leakage-safe pipeline — every learned step is fitted on the TRAINING fold (≈614 rows) only",
            fontsize=11, fontweight="bold", color=TEAL, va="center")
    yb, hb, wb = 5.35, 1.45, 2.3
    xs = [1.6, 4.1, 6.6, 9.1, 11.6, 14.1]
    labels = [
        "1. Zero → NaN\nGlucose, BP, Skin,\nInsulin, BMI\n(Pregnancies kept)",
        "2. Median impute\nmedians learned\nfrom train fold",
        "3. SelectKBest\n(optional, k = 5)\nANOVA F-test",
        "4. Standardise\n$z = (x-\\mu)/\\sigma$\n$\\mu, \\sigma$ from train fold",
        "5. Oversampling\nRandomOverSampler\nminority class,\ntrain fold only",
        "6. MLP ANN\n8 → 16 → 8 → 1\nReLU · sigmoid\nAdam · early stop",
    ]
    for i, (x, lab) in enumerate(zip(xs, labels)):
        fc = "#FDE9DC" if i == 5 else "white"
        ec = ACCENT if i == 5 else NAVY
        _box(ax, x, yb, wb, hb, lab, fc=fc, ec=ec, fs=10)
        if i:
            _arrow(ax, (xs[i - 1] + wb / 2, yb), (x - wb / 2, yb))
    _polyline_arrow(ax, [(9.7, ya - ha / 2), (9.7, 7.35), (1.6, 7.35), (1.6, yb + hb / 2)])
    ax.text(5.6, 7.5, "training fold", fontsize=9, style="italic", color=NAVY, ha="center")

    # Stage C — test-fold inference & evaluation (right to left)
    yc, hc = 2.85, 1.25
    xc = [14.1, 10.6, 7.1, 3.4]
    labs_c = [
        "Fitted pipeline\ntransform only:\nno refit, no resampling",
        "ANN forward pass\n$\\hat{p} = \\sigma(W^{3}a^{2} + b^{3})$\nP(diabetes | x)",
        "Decision rule\n$\\hat{y} = 1$ if $\\hat{p} \\geq 0.5$\n(threshold sweep)",
        "Evaluation per fold\nconfusion matrix · Acc · Prec\nSens · Spec · F1 · ROC-AUC · PR-AUC",
    ]
    widths = [3.0, 3.0, 2.9, 4.3]
    for x, lab, w in zip(xc, labs_c, widths):
        _box(ax, x, yc, w, hc, lab, fc="white", ec=NAVY, fs=10)
    for i in range(1, 4):
        _arrow(ax, (xc[i - 1] - widths[i - 1] / 2, yc), (xc[i] + widths[i] / 2, yc))
    _arrow(ax, (14.1, yb - hb / 2), (14.1, yc + hc / 2), color=ACCENT,
           label="fitted $\\theta^{*}$, medians, $\\mu$, $\\sigma$",
           label_offset=(-1.25, 0.0))
    _polyline_arrow(ax, [(15.1, ya), (16.2, ya), (16.2, yc), (15.6, yc)], color=GOLD)
    ax.text(16.35, 5.6, "test fold", rotation=90, fontsize=9, style="italic", color=GOLD, va="center")

    # Stage D — analysis
    yd = 0.85
    _box(ax, 7.9, yd, 15.2, 1.05,
         "Comparison & analysis\nDummy · Logistic regression · Random forest baselines on identical folds  |  "
         "10×5 repeated CV + corrected t-test  |  ablation study  |  leakage experiments  |  hold-out test",
         fc=LIGHT, fs=10.5)
    _arrow(ax, (3.4, yc - hc / 2), (3.4, yd + 0.53))

    for y, lab in [(ya, "A"), (yb, "B"), (yc, "C"), (yd, "D")]:
        ax.text(0.02, y, lab, fontsize=13, fontweight="bold", color=ACCENT, va="center")
    save(fig, "fig_block_schematic.png")


def fig_ann_architecture():
    layers = [8, 16, 8, 1]
    xs = [1.0, 3.6, 6.2, 8.4]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set_xlim(-1.6, 10.6)
    ax.set_ylim(-0.9, 9.0)
    ax.axis("off")
    pos = []
    for n, x in zip(layers, xs):
        span = 8.0 if n > 1 else 0
        ys = np.linspace(4 + span / 2, 4 - span / 2, n) if n > 1 else np.array([4.0])
        pos.append([(x, y) for y in ys])
    for a, b in zip(pos[:-1], pos[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#B8C2CC", lw=0.45, zorder=1)
    colors = [TEAL, NAVY, NAVY, ACCENT]
    for layer, col in zip(pos, colors):
        for p in layer:
            ax.add_patch(Circle(p, 0.2, fc=col, ec="white", lw=1.2, zorder=3))
    for p, f in zip(pos[0], FEATURE_COLUMNS):
        ax.text(p[0] - 0.32, p[1], SHORT[f] if f != "DiabetesPedigreeFunction" else "Pedigree",
                ha="right", va="center", fontsize=10)
    ax.text(pos[3][0][0] + 0.35, 4.0, "P(Outcome = 1 | x)", va="center", fontsize=11)
    heads = [
        ("Input layer", "8 standardised features\n$x \\in \\mathbb{R}^{8}$"),
        ("Hidden layer 1", "16 units, ReLU\n$W^{1} \\in \\mathbb{R}^{16\\times 8},\\ b^{1} \\in \\mathbb{R}^{16}$"),
        ("Hidden layer 2", "8 units, ReLU\n$W^{2} \\in \\mathbb{R}^{8\\times 16},\\ b^{2} \\in \\mathbb{R}^{8}$"),
        ("Output layer", "1 unit, sigmoid\n$W^{3} \\in \\mathbb{R}^{1\\times 8},\\ b^{3} \\in \\mathbb{R}$"),
    ]
    for x, (h, sub) in zip(xs, heads):
        ax.text(x, 8.75, h, ha="center", fontsize=11.5, fontweight="bold", color=NAVY)
        ax.text(x, -0.55, sub, ha="center", va="center", fontsize=10)
    ax.text(4.9, 8.3, "parameters: (8·16+16) + (16·8+8) + (8·1+1) = 144 + 136 + 9 = 289",
            ha="center", fontsize=10.5, style="italic")
    save(fig, "fig_ann_architecture.png")


def fig_cv_protocol():
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.6, 5.6)
    ax.axis("off")
    for i in range(5):
        y = 4.6 - i
        ax.text(0.15, y + 0.3, f"Fold {i + 1}", va="center", fontsize=11)
        for j in range(5):
            test = i == j
            ax.add_patch(Rectangle((1.2 + j * 1.25, y), 1.15, 0.6, fc=ACCENT if test else TEAL,
                                   alpha=0.9 if test else 0.55, ec="white"))
            ax.text(1.2 + j * 1.25 + 0.575, y + 0.3, "test" if test else "train", ha="center", va="center",
                    fontsize=9.5, color="white")
    ax.text(7.75, 4.4,
            "Inside each fold:\n"
            "1. fit zero-recode, imputer, selector,\n    scaler, ROS and ANN on train blocks\n"
            "2. transform the test block with the\n    fitted objects (no refit)\n"
            "3. score the test block; store $\\hat{p}$\n"
            "Every model sees the same 5 splits.\n"
            "Repeated 10× with new shuffles for\n"
            "the corrected t-test (50 splits).",
            va="top", fontsize=10.5, linespacing=1.3)
    save(fig, "fig_cv_protocol.png")


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

METRICS = ["accuracy", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]
METRIC_LABEL = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall_sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


def fig_repeated_cv_bars(res):
    rep = res["repeated_cv"]
    names = list(rep)
    fig, ax = plt.subplots(figsize=(12, 4.4))
    width = 0.16
    x = np.arange(len(METRICS))
    for k, name in enumerate(names):
        means = [rep[name]["summary"][m]["mean"] for m in METRICS]
        stds = [rep[name]["summary"][m]["std"] for m in METRICS]
        ax.bar(x + (k - 2) * width, means, width, yerr=stds, capsize=2.5, color=MODEL_COLORS[name], label=name,
               edgecolor="white", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABEL[m] for m in METRICS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (mean ± sd, 50 splits)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False, fontsize=9.5)
    save(fig, "fig_repeated_cv_bars.png")


def fig_roc_pr(res, arrays):
    y = arrays["y"]
    keys = [
        ("Dummy (majority)", "dummy_most_frequent"),
        ("Logistic regression", "logistic_regression__all_features__no_resample"),
        ("Random forest", "random_forest__all_features__no_resample"),
        ("MLP ANN (no rebalancing)", "mlp_ann__all_features__no_resample"),
        ("Proposed MLP ANN (+ROS)", "mlp_ann__all_features__oversample"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, key in keys:
        p = arrays[f"oof_proba__{key}"]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ls = "--" if label.startswith("Dummy") else "-"
        lw = 2.4 if label.startswith("Proposed") else 1.5
        axes[0].plot(fpr, tpr, ls=ls, lw=lw, color=MODEL_COLORS[label],
                     label=f"{label} (AUC {roc_auc_score(y, p):.3f})")
        axes[1].plot(rec, prec, ls=ls, lw=lw, color=MODEL_COLORS[label],
                     label=f"{label} (AP {average_precision_score(y, p):.3f})")
    axes[0].set_xlabel("False positive rate (1 − specificity)")
    axes[0].set_ylabel("True positive rate (sensitivity)")
    axes[0].set_title("ROC curves (out-of-fold, 5-fold CV)")
    axes[1].set_xlabel("Recall (sensitivity)")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–recall curves (out-of-fold)")
    axes[1].axhline(y.mean(), color="#BBBBBB", ls=":", lw=1)
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left", fontsize=8.8)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.02)
    fig.tight_layout()
    save(fig, "fig_roc_pr_oof.png")


def _cm(ax, cm, title, cmap="Blues"):
    cm = np.asarray(cm)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.sum(axis=1).max())
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            pct = v / cm[i].sum()
            ax.text(j, i, f"{v}\n({pct:.1%})", ha="center", va="center", fontsize=11,
                    color="white" if v > cm.max() * 0.55 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_confusion_oof(res):
    keys = [
        ("Logistic regression", "logistic_regression__all_features__no_resample"),
        ("Random forest", "random_forest__all_features__no_resample"),
        ("MLP ANN (no rebalancing)", "mlp_ann__all_features__no_resample"),
        ("Proposed MLP ANN (+ROS)", "mlp_ann__all_features__oversample"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (label, key) in zip(axes, keys):
        _cm(ax, res["main_cv"][key]["confusion_matrix_oof"], label, cmap="Oranges" if "Proposed" in label else "Blues")
    fig.tight_layout()
    save(fig, "fig_confusion_oof.png")


def fig_loss_curve(res):
    lc = res["loss_curve"]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ep = np.arange(1, len(lc["loss"]) + 1)
    ax1.plot(ep, lc["loss"], color=NAVY, lw=2, label="Training loss (cross-entropy + L2)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color=NAVY)
    ax2 = ax1.twinx()
    ax2.plot(ep, lc["val_accuracy"], color=ACCENT, lw=1.8, marker="o", ms=3, label="Validation accuracy")
    ax2.set_ylabel("Validation accuracy (15% of train fold)", color=ACCENT)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    best = int(np.argmax(lc["val_accuracy"])) + 1
    ax1.axvline(best, color="k", ls=":", lw=1)
    ax1.text(best + 0.4, min(lc["loss"]) + 0.02, f"best validation epoch = {best}", fontsize=9.5)
    ax1.set_title(f"ANN training on fold 1 — stopped at epoch {lc['n_iter']} (patience 20)")
    lines = ax1.get_lines()[:1] + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=9.5)
    save(fig, "fig_loss_curve.png")


def fig_learning_curve(res):
    lc = res["learning_curve"]
    n = np.array(lc["train_sizes"])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for key, color, label in [("train", NAVY, "Training ROC-AUC"), ("test", ACCENT, "Validation ROC-AUC")]:
        m, s = np.array(lc[f"{key}_mean"]), np.array(lc[f"{key}_std"])
        ax.plot(n, m, marker="o", color=color, label=label)
        ax.fill_between(n, m - s, m + s, color=color, alpha=0.15)
    ax.set_xlabel("Training rows used (per fold)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.7, 0.95)
    ax.set_title("Learning curve of the proposed ANN (5-fold CV)")
    ax.legend()
    save(fig, "fig_learning_curve.png")


def fig_threshold(res):
    rows = pd.DataFrame(res["threshold_sweep"]["rows"])
    best = res["threshold_sweep"]["youden_best"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for col, color, label in [
        ("recall_sensitivity", ACCENT, "Sensitivity"),
        ("specificity", TEAL, "Specificity"),
        ("precision", GOLD, "Precision"),
        ("f1", NAVY, "F1"),
    ]:
        ax.plot(rows["threshold"], rows[col], color=color, lw=2, label=label)
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.text(0.51, 0.05, "default 0.5", fontsize=9.5)
    ax.axvline(best["threshold"], color=RED, ls=":", lw=1.3)
    ax.text(best["threshold"] - 0.01, 0.12, f"Youden J max\n@ {best['threshold']:.2f}", fontsize=9.5, ha="right",
            color=RED)
    ax.set_xlabel("Decision threshold on $\\hat{p}$")
    ax.set_ylabel("Score (out-of-fold)")
    ax.set_title("Operating-point trade-off of the proposed ANN")
    ax.legend(loc="center right")
    save(fig, "fig_threshold_sweep.png")


def fig_holdout(res, arrays):
    ho = res["holdout"]["models"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    for name in ho:
        fpr, tpr = arrays[f"holdout_roc__{name}"]
        lw = 2.4 if name.startswith("Proposed") else 1.5
        axes[0].plot(fpr, tpr, color=MODEL_COLORS[name], lw=lw,
                     label=f"{name} (AUC {ho[name]['metrics']['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], color="#BBBBBB", ls="--", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("Hold-out ROC curves (154 unseen rows)")
    axes[0].legend(loc="lower right", fontsize=9)
    _cm(axes[1], ho["Proposed MLP ANN (+ROS)"]["confusion_matrix"], "Proposed ANN — hold-out confusion matrix",
        cmap="Oranges")
    fig.tight_layout()
    save(fig, "fig_holdout.png")


def fig_permutation(res):
    pi = res["permutation_importance"]
    items = sorted(pi.items(), key=lambda kv: kv[1]["mean"])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh([SHORT[k] if k != "DiabetesPedigreeFunction" else "Pedigree" for k, _ in items],
            [v["mean"] for _, v in items], xerr=[v["std"] for _, v in items],
            color=[ACCENT if v["mean"] > 0.05 else TEAL for _, v in items], capsize=3)
    ax.set_xlabel("Mean drop in hold-out ROC-AUC when the feature is shuffled")
    ax.set_title("Permutation importance of the proposed ANN (30 repeats)")
    save(fig, "fig_permutation_importance.png")


def fig_ablation_components(res):
    ab = [v for v in res["ablation"].values() if v["group"] == "component"]
    labels = [v["label"] for v in ab]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, metric in zip(axes, ["f1", "roc_auc"]):
        means = [v["summary"][metric]["mean"] for v in ab]
        stds = [v["summary"][metric]["std"] for v in ab]
        colors = [ACCENT if i == 0 else (RED if means[i] < means[0] - 0.02 else TEAL) for i in range(len(ab))]
        y = np.arange(len(ab))[::-1]
        ax.barh(y, means, xerr=stds, color=colors, capsize=2.5, alpha=0.9)
        ax.axvline(means[0], color=ACCENT, ls="--", lw=1)
        for yi, m in zip(y, means):
            ax.text(m + 0.003, yi + 0.28, f"{m:.3f}", fontsize=8.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        lo = 0.5 if metric == "f1" else 0.65
        ax.set_xlim(lo, 0.9)
        ax.set_xlabel(f"{METRIC_LABEL[metric]} (mean ± sd, 15 splits)")
    axes[0].set_title("Component ablation — F1")
    axes[1].set_title("Component ablation — ROC-AUC")
    fig.tight_layout()
    save(fig, "fig_ablation_components.png")


def fig_ablation_hyper(res):
    groups = [("architecture", "Hidden layers"), ("activation", "Activation"), ("optimiser", "Optimiser"),
              ("alpha", "L2 penalty α"), ("k_features", "Features kept (k)")]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.2), sharey=True)
    for ax, (g, title) in zip(axes, groups):
        items = [v for v in res["ablation"].values() if v["group"] == g]
        labels = [v["label"] for v in items]
        x = np.arange(len(items))
        for off, metric, color in [(-0.2, "roc_auc", NAVY), (0.2, "f1", ACCENT)]:
            ax.bar(x + off, [v["summary"][metric]["mean"] for v in items], 0.38,
                   yerr=[v["summary"][metric]["std"] for v in items], color=color, capsize=2,
                   label=METRIC_LABEL[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40 if g in ("architecture", "optimiser") else 0,
                           ha="right" if g in ("architecture", "optimiser") else "center", fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0.45, 0.92)
    axes[0].set_ylabel("Score (mean ± sd, 15 splits)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_ablation_hyper.png")


def fig_leakage(res):
    lk = res["leakage"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, model in zip(axes, ["MLP", "RF"]):
        items = [(k.split("::")[1], v) for k, v in lk.items() if k.startswith(model + "::")]
        x = np.arange(len(items))
        for off, metric, color in [(-0.27, "f1", ACCENT), (0, "roc_auc", NAVY), (0.27, "pr_auc", GOLD)]:
            vals = [v[metric]["mean"] for _, v in items]
            bars = ax.bar(x + off, vals, 0.26, yerr=[v[metric]["std"] for _, v in items], capsize=2, color=color,
                          label=METRIC_LABEL[metric])
            for i, b in enumerate(bars):
                if i >= 3:
                    b.set_hatch("//")
                    b.set_edgecolor("white")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" (proposed)", "\n(proposed)").replace(" on all rows", "\non all rows")
                            .replace(" before CV", "\nbefore CV") for n, _ in items], fontsize=9)
        ax.set_title("ANN (MLP)" if model == "MLP" else "Random forest")
        ax.set_ylim(0.55, 0.95)
    axes[0].set_ylabel("Cross-validated score (15 splits)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_leakage.png")


def fig_seed_stability(res):
    rows = pd.DataFrame(res["seed_stability"]["per_seed"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for metric, color in [("roc_auc", NAVY), ("recall_sensitivity", ACCENT), ("f1", GOLD), ("accuracy", TEAL)]:
        ax.plot(rows["seed"], rows[metric], marker="o", color=color, label=METRIC_LABEL[metric])
        ax.axhline(rows[metric].mean(), color=color, ls=":", lw=1)
    ax.set_xticks(rows["seed"])
    ax.set_xlabel("Random seed of the ANN initialisation and oversampler")
    ax.set_ylabel("5-fold mean score")
    ax.set_title("Seed stability of the proposed ANN (fixed folds)")
    ax.legend(ncol=4, fontsize=9, loc="lower center")
    ax.set_ylim(0.55, 0.9)
    save(fig, "fig_seed_stability.png")


# --------------------------------------------------------------------------
# Console screenshots (rendered from the captured stdout files)
# --------------------------------------------------------------------------


def render_console(lines: list[str], command: str, name: str, fs: float = 9.0) -> None:
    lines = [l.rstrip("\n") for l in lines]
    all_lines = [f"$ {command}"] + lines
    maxlen = max(len(l) for l in all_lines)
    char_w = fs * 0.602 / 72
    line_h = fs * 1.38 / 72
    width = maxlen * char_w + 0.5
    bar = 0.32
    height = len(all_lines) * line_h + bar + 0.3
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), width, height - bar, fc="#1E1E1E", ec="none"))
    ax.add_patch(Rectangle((0, height - bar), width, bar, fc="#3A3A3A", ec="none"))
    for i, c in enumerate(["#FF5F56", "#FFBD2E", "#27C93F"]):
        ax.add_patch(Circle((0.18 + i * 0.2, height - bar / 2), 0.055, fc=c, ec="none"))
    ax.text(width / 2, height - bar / 2, "Terminal — diabetes_ann (zsh)", color="#DDDDDD", ha="center",
            va="center", fontsize=fs, family="sans-serif")
    y = height - bar - 0.18
    for i, line in enumerate(all_lines):
        color = "#7EC699" if i == 0 else "#D4D4D4"
        ax.text(0.25, y, line, family="Menlo", fontsize=fs, color=color, va="top")
        y -= line_h
    fig.savefig(FIG / name, dpi=220, facecolor="#1E1E1E")
    plt.close(fig)
    print(f"wrote figures/{name}")


def console_figures():
    text = (RESULTS / "console_experiments.txt").read_text(encoding="utf-8").splitlines()

    def section(start: str, end: str | None) -> list[str]:
        i = next(k for k, l in enumerate(text) if l.startswith(start))
        j = next((k for k, l in enumerate(text) if end and l.startswith(end) and k > i), len(text))
        block = text[i:j]
        while block and not block[-1].strip():
            block.pop()
        return block

    cmd = "python final_report/experiments.py"
    render_console(text[:4] + [""] + section("[1/8]", "[2/8]"), cmd, "console_cv_main.png")
    render_console(section("[2/8]", "[3/8]"), cmd + "   # (continued)", "console_repeated_cv.png")
    abl = section("[3/8]", "[4/8]")
    cut = next(k for k, l in enumerate(abl) if l.startswith("Ablation — architecture"))
    render_console(abl[:cut], cmd + "   # (continued)", "console_ablation.png")
    render_console(section("[4/8]", "[6/8]"), cmd + "   # (continued)", "console_leakage_nested.png")
    render_console(section("[8/8]", "Saved"), cmd + "   # (continued)", "console_holdout.png")
    demo = (RESULTS / "console_predict_demo.txt").read_text(encoding="utf-8").splitlines()
    demo = [l if len(l) < 125 else l[:122] + "..." for l in demo]
    render_console(demo, "python final_report/predict_demo.py", "console_predict_demo.png")
    pytest_path = RESULTS / "console_pytest.txt"
    if pytest_path.exists():
        lines = [l for l in pytest_path.read_text(encoding="utf-8").splitlines()
                 if "PASSED" in l or "passed" in l or l.startswith("platform") or l.startswith("collected")]
        lines = [l.replace(str(ROOT.parent) + "/", "") for l in lines]
        render_console(lines, "python -m pytest -v", "console_pytest.png")


def main() -> int:
    res = json.loads((RESULTS / "final_results.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(RESULTS / "final_arrays.npz"))
    df = load_dataset()

    fig_class_balance(df)
    fig_zero_counts(df)
    fig_distributions(df)
    fig_boxplots(df)
    corr = fig_correlation(df)
    anova = fig_anova_and_selection(df, res)
    fig_block_schematic()
    fig_ann_architecture()
    fig_cv_protocol()
    fig_repeated_cv_bars(res)
    fig_roc_pr(res, arrays)
    fig_confusion_oof(res)
    fig_loss_curve(res)
    fig_learning_curve(res)
    fig_threshold(res)
    fig_holdout(res, arrays)
    fig_permutation(res)
    fig_ablation_components(res)
    fig_ablation_hyper(res)
    fig_leakage(res)
    fig_seed_stability(res)
    console_figures()

    rc = recoded(df)
    eda = {
        "corr_with_outcome": corr["Outcome"].drop("Outcome").round(3).to_dict(),
        "anova_train_folds": anova,
        "class_means_recoded": rc.groupby("Outcome")[FEATURE_COLUMNS].mean().round(2).to_dict(),
        "class_medians_recoded": rc.groupby("Outcome")[FEATURE_COLUMNS].median().round(2).to_dict(),
        "describe_recoded": rc[FEATURE_COLUMNS].describe().round(2).to_dict(),
    }
    (RESULTS / "eda_summary.json").write_text(json.dumps(eda, indent=2), encoding="utf-8")
    print("wrote results/eda_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
