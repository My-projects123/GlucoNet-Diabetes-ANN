"""Load the Pima Indians Diabetes dataset from a local file.

This module never downloads data. If the CSV (or a local ARFF original) is
missing, it raises a clear error with obtain-and-placement instructions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from diabetes_prediction.config import (
    ALL_COLUMNS,
    ARFF_FILENAME,
    COLUMN_ALIASES,
    CSV_FILENAME,
    FEATURE_COLUMNS,
    NEGATIVE_LABELS,
    POSITIVE_LABELS,
    TARGET_COLUMN,
    default_data_candidates,
)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the local diabetes CSV/ARFF cannot be found."""


def dataset_instructions(missing: Path | None = None) -> str:
    missing_line = f"Looked for: {missing}\n\n" if missing else ""
    return (
        f"{missing_line}"
        "This project does not download the dataset.\n\n"
        "How to obtain the Pima Indians Diabetes dataset:\n"
        "  1. UCI Machine Learning Repository — historically titled\n"
        "     'Pima Indians Diabetes Database' (originally from NIDDK).\n"
        "  2. Kaggle — search for 'Pima Indians Diabetes Database'.\n"
        "  3. This repository already includes a Weka ARFF copy at\n"
        f"     data/{ARFF_FILENAME}. Convert it instead of downloading:\n"
        "       python -m diabetes_prediction --convert-arff\n\n"
        "Place a CSV here (either path works):\n"
        f"  diabetes_ann/data/{CSV_FILENAME}\n"
        f"  data/{CSV_FILENAME}\n\n"
        "Expected columns:\n"
        f"  {', '.join(ALL_COLUMNS)}\n"
        "Outcome should be 0 (no diabetes) or 1 (diabetes). A header row is\n"
        "preferred; a headerless 9-column CSV is also accepted.\n"
    )


def _normalize_name(name: str) -> str:
    cleaned = str(name).strip().strip("'\"")
    key = cleaned.lower().replace(" ", "").replace("-", "")
    key = key.replace("(", "").replace(")", "")
    return COLUMN_ALIASES.get(key, cleaned)


def _map_outcome(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Outcome contains a missing value.")
    if isinstance(value, (int, float)) and float(value) in (0.0, 1.0):
        return int(value)
    text = str(value).strip().lower()
    if text in POSITIVE_LABELS:
        return 1
    if text in NEGATIVE_LABELS:
        return 0
    raise ValueError(f"Unrecognized Outcome value: {value!r}")


def _standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_name(c) for c in df.columns]

    if list(df.columns) == list(range(len(df.columns))) or (
        len(df.columns) == 9 and TARGET_COLUMN not in df.columns
    ):
        if df.shape[1] != 9:
            raise ValueError(
                f"Headerless CSV must have 9 columns; found {df.shape[1]}."
            )
        df.columns = ALL_COLUMNS

    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    extra_ok = [c for c in df.columns if c not in ALL_COLUMNS]
    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            f"{missing}. Extra columns found: {extra_ok}.\n"
            + dataset_instructions()
        )

    out = df[ALL_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[TARGET_COLUMN] = out[TARGET_COLUMN].map(_map_outcome).astype(int)
    return out.reset_index(drop=True)


def read_arff_table(path: Path) -> pd.DataFrame:
    """Minimal ARFF reader for the local Pima file (no extra dependency)."""
    rows: list[list[str]] = []
    in_data = False
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not in_data:
                if line.lower().startswith("@data"):
                    in_data = True
                continue
            if not line or line.startswith("%"):
                continue
            rows.append([part.strip() for part in line.split(",")])
    if not rows:
        raise ValueError(f"No @data rows found in {path}")
    df = pd.DataFrame(rows)
    if df.shape[1] != 9:
        raise ValueError(f"Expected 9 ARFF columns, found {df.shape[1]} in {path}")
    df.columns = ALL_COLUMNS
    return _standardize_frame(df)


def convert_arff_to_csv(arff_path: Path, csv_path: Path) -> Path:
    """Write a standardized CSV from a local ARFF file."""
    df = read_arff_table(arff_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def resolve_dataset_path(path: str | Path | None = None) -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise DatasetNotFoundError(dataset_instructions(resolved))
        return resolved

    for candidate in default_data_candidates():
        if candidate.exists():
            return candidate.resolve()
    raise DatasetNotFoundError(dataset_instructions(default_data_candidates()[0]))


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the local diabetes table as a standardized DataFrame.

    Accepts a CSV with the expected headers, a headerless 9-column CSV, or
    the Weka ARFF file already in this repository.
    """
    resolved = resolve_dataset_path(path)
    if resolved.suffix.lower() == ".arff":
        return read_arff_table(resolved)
    header = None if not _csv_has_named_header(resolved) else "infer"
    df = pd.read_csv(resolved, header=header)
    return _standardize_frame(df)


def _csv_has_named_header(path: Path) -> bool:
    with path.open(encoding="utf-8") as handle:
        first = handle.readline()
    parts = [part.strip().strip("'\"") for part in first.split(",")]
    mapped = [_normalize_name(part) for part in parts]
    return sum(name in ALL_COLUMNS for name in mapped) >= 6


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    y.name = TARGET_COLUMN
    return X, y
