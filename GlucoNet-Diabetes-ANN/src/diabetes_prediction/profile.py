"""Raw-data profiling: balance, duplicates, missing values, and suspicious zeros."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from diabetes_prediction.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    VALID_ZERO_COLUMNS,
    ZERO_AS_MISSING_COLUMNS,
    ZERO_AS_MISSING_JUSTIFICATION,
)


def _iqr_outlier_counts(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in FEATURE_COLUMNS:
        series = pd.to_numeric(df[column], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            counts[column] = 0
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        counts[column] = int(((series < lower) | (series > upper)).sum())
    return counts


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-serializable profile of the standardized diabetes table."""
    n_rows = int(len(df))
    n_cols = int(df.shape[1])
    duplicate_rows = int(df.duplicated().sum())

    outcome_counts = df[TARGET_COLUMN].value_counts().sort_index()
    n_negative = int(outcome_counts.get(0, 0))
    n_positive = int(outcome_counts.get(1, 0))
    positive_rate = float(n_positive / n_rows) if n_rows else float("nan")

    missing_true_nan = {col: int(df[col].isna().sum()) for col in FEATURE_COLUMNS}
    zero_counts = {
        col: int((pd.to_numeric(df[col], errors="coerce") == 0).sum())
        for col in FEATURE_COLUMNS
    }
    zeros_to_missing = {col: zero_counts[col] for col in ZERO_AS_MISSING_COLUMNS}
    valid_zeros = {col: zero_counts[col] for col in VALID_ZERO_COLUMNS}

    recoded = df.copy()
    for col in ZERO_AS_MISSING_COLUMNS:
        recoded[col] = recoded[col].mask(recoded[col] == 0)
    missing_after_recode = {
        col: int(recoded[col].isna().sum()) for col in FEATURE_COLUMNS
    }

    describe = df[FEATURE_COLUMNS].describe().to_dict()
    # numpy types -> plain python for JSON
    describe_plain = {
        col: {stat: _to_plain(value) for stat, value in stats.items()}
        for col, stats in describe.items()
    }

    return {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "duplicate_rows": duplicate_rows,
        "class_counts": {"no_diabetes_0": n_negative, "diabetes_1": n_positive},
        "positive_rate": positive_rate,
        "class_balance_note": (
            "The minority class is Outcome=1 (diabetes). Accuracy alone can look "
            "optimistic because predicting 'no diabetes' for every row would "
            f"already score about {n_negative / n_rows:.1%} on this table."
            if n_rows
            else "Empty table."
        ),
        "true_nan_missing": missing_true_nan,
        "zero_counts": zero_counts,
        "zeros_treated_as_missing": zeros_to_missing,
        "zeros_kept_as_valid": valid_zeros,
        "missing_after_zero_recode": missing_after_recode,
        "zero_as_missing_justification": ZERO_AS_MISSING_JUSTIFICATION,
        "pregnancies_note": (
            "Pregnancies=0 is kept. It means the person has never been pregnant, "
            "which is a valid value in this dataset of adult women."
        ),
        "iqr_outlier_counts": _iqr_outlier_counts(df),
        "summary_stats": describe_plain,
        "high_missingness_after_recode": [
            col
            for col, count in missing_after_recode.items()
            if n_rows and count / n_rows >= 0.20
        ],
    }


def _to_plain(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def profile_to_markdown(profile: dict[str, Any]) -> str:
    class_counts = profile["class_counts"]
    lines = [
        "# Data profile — Pima Indians Diabetes",
        "",
        profile_disclaimer(),
        "",
        "## Shape and duplicates",
        "",
        f"- Rows: {profile['n_rows']}",
        f"- Columns: {profile['n_columns']}",
        f"- Duplicate rows: {profile['duplicate_rows']}",
        "",
        "## Class balance",
        "",
        f"- Outcome=0 (no diabetes): {class_counts['no_diabetes_0']}",
        f"- Outcome=1 (diabetes): {class_counts['diabetes_1']}",
        f"- Positive rate: {profile['positive_rate']:.3f}",
        f"- {profile['class_balance_note']}",
        "",
        "## Missing values (true NaN in the file)",
        "",
    ]
    for col, count in profile["true_nan_missing"].items():
        lines.append(f"- {col}: {count}")

    lines += ["", "## Zero counts (raw file)", ""]
    for col, count in profile["zero_counts"].items():
        lines.append(f"- {col}: {count}")

    lines += [
        "",
        "## Zeros recoded as missing (justified)",
        "",
        "These fields use 0 as a missing-value code. Recoding happens inside the",
        "training pipeline so the rule is applied the same way on every fold.",
        "",
    ]
    for col, count in profile["zeros_treated_as_missing"].items():
        why = profile["zero_as_missing_justification"][col]
        lines.append(f"- **{col}**: {count} zeros. {why}")

    lines += [
        "",
        "## Zeros kept",
        "",
        profile["pregnancies_note"],
        "",
    ]
    for col, count in profile["zeros_kept_as_valid"].items():
        lines.append(f"- {col}: {count} zeros kept")

    lines += ["", "## Missingness after zero recoding", ""]
    for col, count in profile["missing_after_zero_recode"].items():
        pct = count / profile["n_rows"] * 100 if profile["n_rows"] else 0
        lines.append(f"- {col}: {count} ({pct:.1f}%)")

    high = profile["high_missingness_after_recode"]
    if high:
        lines += [
            "",
            "Columns with at least 20% missing after recoding: "
            + ", ".join(high)
            + ". Median imputation will fill these, but the imputed values",
            "are a modeling convenience, not recovered lab measurements.",
        ]

    lines += [
        "",
        "## IQR outlier counts (1.5×IQR on raw values)",
        "",
        "These counts use the raw file, so coded-missing zeros (especially",
        "BloodPressure and Insulin) can appear as outliers. That is expected.",
        "",
    ]
    for col, count in profile["iqr_outlier_counts"].items():
        lines.append(f"- {col}: {count}")

    lines += ["", "## Summary statistics (raw values, zeros still present)", "", "```"]
    stats_df = pd.DataFrame(profile["summary_stats"])
    lines.append(stats_df.to_string())
    lines += ["```", ""]
    return "\n".join(lines)


def profile_disclaimer() -> str:
    from diabetes_prediction.config import DISCLAIMER

    return f"> {DISCLAIMER}"
