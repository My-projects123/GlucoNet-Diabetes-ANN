"""Command-line entry point: profile, plot, train, and write reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diabetes_prediction.config import (
    DISCLAIMER,
    FEATURE_COLUMNS,
    N_SPLITS,
    RANDOM_SEED,
    SELECT_K,
    default_data_candidates,
    figures_dir,
    project_root,
    reports_dir,
)
from diabetes_prediction.data import (
    DatasetNotFoundError,
    convert_arff_to_csv,
    dataset_instructions,
    load_dataset,
    split_xy,
)
from diabetes_prediction.evaluate import METRIC_ORDER, format_metric, mean_std_table
from diabetes_prediction.plots import (
    plot_confusion_matrices,
    plot_feature_selection_frequency,
    plot_metric_bars,
    plot_roc_pr,
    write_eda_plots,
)
from diabetes_prediction.profile import profile_dataframe, profile_to_markdown
from diabetes_prediction.reproducibility import configure_runtime
from diabetes_prediction.train import run_comparison


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _strip_arrays(result: dict[str, Any]) -> dict[str, Any]:
    skip = {"oof_pred", "oof_proba", "y_true"}
    return {k: _json_ready(v) for k, v in result.items() if k not in skip}


def write_model_report(
    results: dict[str, Any],
    table: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Model comparison",
        "",
        f"> {DISCLAIMER}",
        "",
        "Metrics are **mean ± standard deviation** over stratified folds.",
        "Out-of-fold (OOF) ROC-AUC / PR-AUC use concatenated fold predictions.",
        "Feature selection and resampling, when enabled, are fit on the training",
        "fold only. Numbers below are computed results, not a performance claim",
        "made before training.",
        "",
        "## Summary table",
        "",
        "| Experiment | Accuracy | Precision | Recall/Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        cells = [row["experiment"]]
        for metric in METRIC_ORDER:
            cells.append(format_metric(row[f"{metric}_mean"], row[f"{metric}_std"]))
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Out-of-fold confusion matrices", ""]
    for name, result in results.items():
        cm = result["confusion_matrix_oof"]
        lines += [
            f"### {name}",
            "",
            "```",
            f"              Pred 0    Pred 1",
            f"Actual 0   {cm[0][0]:8d} {cm[0][1]:8d}",
            f"Actual 1   {cm[1][0]:8d} {cm[1][1]:8d}",
            "```",
            "",
        ]

    lines += ["", "## Features selected within each training fold", ""]
    for name, result in results.items():
        if not result.get("feature_selection"):
            continue
        lines.append(f"### {name}")
        for fold, feats in enumerate(result["selected_features_by_fold"], start=1):
            lines.append(f"- Fold {fold}: {', '.join(feats)}")
        counts = Counter(
            feat
            for fold_feats in result["selected_features_by_fold"]
            for feat in fold_feats
        )
        lines.append(
            "- Selection frequency: "
            + ", ".join(f"{feat} ({count})" for feat, count in counts.most_common())
        )
        lines.append("")

    lines += [
        "## How to read these metrics",
        "",
        "- **Accuracy** can look high if the model mostly predicts the majority class.",
        "- **Recall / sensitivity** is the share of true diabetes cases found.",
        "- **Specificity** is the share of true non-diabetes cases correctly kept negative.",
        "- **PR-AUC** is often more informative than ROC-AUC on imbalanced data.",
        "- The dummy most-frequent model is a sanity floor, not a clinical baseline.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diabetes prediction with a small ANN, logistic regression, "
            "and a random-forest baseline. Local CSV only — no download."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to a local CSV or ARFF file. Optional if the default file exists.",
    )
    parser.add_argument(
        "--convert-arff",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Convert a local ARFF file to CSV and exit. Pass a path, or omit "
            "the path to convert data/diabetes.arff if it exists."
        ),
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--select-k", type=int, default=SELECT_K)
    parser.add_argument(
        "--skip-resample",
        action="store_true",
        help="Skip the oversampling comparison (faster).",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Write the data profile and EDA plots, then stop.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for reports (default: diabetes_ann/reports).",
    )
    return parser.parse_args(argv)


def _find_arff() -> Path | None:
    for candidate in default_data_candidates():
        if candidate.suffix.lower() == ".arff" and candidate.exists():
            return candidate
    return None


def convert_and_exit(arff_arg: str) -> int:
    if arff_arg == "auto":
        arff_path = _find_arff()
        if arff_path is None:
            print("No local ARFF file was found.")
            print(dataset_instructions())
            return 1
    else:
        arff_path = Path(arff_arg).expanduser().resolve()
        if not arff_path.exists():
            print(dataset_instructions(arff_path))
            return 1
    csv_path = project_root() / "data" / "pima_indians_diabetes.csv"
    convert_arff_to_csv(arff_path, csv_path)
    print(f"Wrote {csv_path} from {arff_path}")
    print("This converted a file already on disk. Nothing was downloaded.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.convert_arff is not None:
        return convert_and_exit(args.convert_arff)

    configure_runtime(args.seed)
    out_dir = args.output_dir or reports_dir()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures" if args.output_dir else figures_dir()
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_dataset(args.data)
    except DatasetNotFoundError as exc:
        print(str(exc))
        return 1

    print(DISCLAIMER)
    print()
    print(f"Loaded {len(df)} rows from a local file.")
    print(f"Features: {', '.join(FEATURE_COLUMNS)}")

    profile = profile_dataframe(df)
    (out_dir / "data_profile.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )
    (out_dir / "data_profile.md").write_text(
        profile_to_markdown(profile), encoding="utf-8"
    )
    plot_paths = write_eda_plots(df, fig_dir)
    print(f"Wrote data profile to {out_dir / 'data_profile.md'}")
    print("Wrote EDA figures:")
    for path in plot_paths:
        print(f"  {path}")

    if args.profile_only:
        print("Profile-only run complete. No models were trained.")
        return 0

    X, y = split_xy(df)
    resample_modes: tuple[bool, ...] = (False,) if args.skip_resample else (False, True)
    print(
        f"Running stratified {args.splits}-fold CV "
        f"(seed={args.seed}, select_k={args.select_k}, "
        f"resample_modes={resample_modes})..."
    )
    results = run_comparison(
        X,
        y,
        n_splits=args.splits,
        random_state=args.seed,
        k=args.select_k,
        resample_modes=resample_modes,
    )
    table = mean_std_table(results)
    table.to_csv(out_dir / "metrics.csv", index=False)
    compact = {name: _strip_arrays(result) for name, result in results.items()}
    (out_dir / "metrics.json").write_text(
        json.dumps(compact, indent=2), encoding="utf-8"
    )
    write_model_report(results, table, out_dir / "model_comparison.md")

    # Plot a readable subset: dummy + the three models with all features, no resample.
    plot_names = [
        name
        for name in results
        if name == "dummy_most_frequent" or name.endswith("all_features__no_resample")
    ]
    if not plot_names:
        plot_names = list(results)
    plot_confusion_matrices(results, fig_dir, plot_names)
    plot_roc_pr(results, fig_dir, plot_names)
    plot_metric_bars(table, fig_dir, "roc_auc")
    plot_metric_bars(table, fig_dir, "pr_auc")
    plot_metric_bars(table, fig_dir, "f1")
    plot_feature_selection_frequency(results, fig_dir)

    print()
    print(table.to_string(index=False))
    print()
    print(f"Wrote {out_dir / 'model_comparison.md'}")
    print(f"Wrote {out_dir / 'metrics.csv'}")
    print()
    print("Reproduce this run:")
    extra = ""
    if args.skip_resample:
        extra = " --skip-resample"
    print(
        "  cd diabetes_ann && python run_experiment.py "
        f"--seed {args.seed} --splits {args.splits} --select-k {args.select_k}{extra}"
    )
    print()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
