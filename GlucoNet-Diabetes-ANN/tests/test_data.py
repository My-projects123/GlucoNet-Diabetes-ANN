from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from diabetes_prediction.config import FEATURE_COLUMNS, TARGET_COLUMN
from diabetes_prediction.data import (
    DatasetNotFoundError,
    convert_arff_to_csv,
    load_dataset,
    split_xy,
)
from diabetes_prediction.profile import profile_dataframe


def test_missing_file_raises_with_instructions(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(DatasetNotFoundError) as exc:
        load_dataset(missing)
    message = str(exc.value)
    assert "does not download" in message
    assert "UCI" in message
    assert "Kaggle" in message
    assert str(missing) in message


def test_load_standard_csv(tiny_csv: Path) -> None:
    df = load_dataset(tiny_csv)
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert set(df[TARGET_COLUMN].unique()) <= {0, 1}
    X, y = split_xy(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == 12


def test_load_headerless_csv(tmp_path: Path, tiny_diabetes_frame: pd.DataFrame) -> None:
    path = tmp_path / "headerless.csv"
    tiny_diabetes_frame.to_csv(path, index=False, header=False)
    df = load_dataset(path)
    assert df.shape == tiny_diabetes_frame.shape
    assert int(df[TARGET_COLUMN].sum()) == int(tiny_diabetes_frame[TARGET_COLUMN].sum())


def test_load_alias_columns(tmp_path: Path) -> None:
    path = tmp_path / "aliases.csv"
    path.write_text(
        "preg,plas,pres,skin,insu,mass,pedi,age,class\n"
        "0,120,70,20,80,30.0,0.4,25,tested_negative\n"
        "1,150,80,25,100,33.0,0.6,40,tested_positive\n"
    )
    df = load_dataset(path)
    assert df.loc[0, TARGET_COLUMN] == 0
    assert df.loc[1, TARGET_COLUMN] == 1
    assert df.loc[0, "Pregnancies"] == 0


def test_profile_counts_zeros_and_duplicates(tiny_diabetes_frame: pd.DataFrame) -> None:
    profile = profile_dataframe(tiny_diabetes_frame)
    assert profile["n_rows"] == 12
    assert profile["duplicate_rows"] == 0
    assert profile["class_counts"]["diabetes_1"] == 6
    assert profile["zeros_treated_as_missing"]["Glucose"] >= 1
    assert profile["zeros_kept_as_valid"]["Pregnancies"] >= 1
    assert "Pregnancies" not in profile["zeros_treated_as_missing"]


def test_real_pima_csv_if_present() -> None:
    """Use the local CSV when it exists; skip instead of downloading."""
    from diabetes_prediction.config import CSV_FILENAME, project_root, workspace_root

    candidates = [
        project_root() / "data" / CSV_FILENAME,
        workspace_root() / "data" / CSV_FILENAME,
    ]
    existing = next((path for path in candidates if path.exists()), None)
    if existing is None:
        pytest.skip("local Pima CSV is not in the workspace")
    df = load_dataset(existing)
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert set(df[TARGET_COLUMN].unique()) <= {0, 1}
    assert df["Pregnancies"].min() >= 0
    # Canonical extract is 768 rows; other local copies may differ.
    if len(df) == 768:
        assert int((df[TARGET_COLUMN] == 0).sum()) == 500
        assert int((df[TARGET_COLUMN] == 1).sum()) == 268


def test_convert_arff(tmp_path: Path) -> None:
    arff = tmp_path / "mini.arff"
    arff.write_text(
        "@relation pima_diabetes\n"
        "@attribute preg numeric\n"
        "@attribute plas numeric\n"
        "@attribute pres numeric\n"
        "@attribute skin numeric\n"
        "@attribute insu numeric\n"
        "@attribute mass numeric\n"
        "@attribute pedi numeric\n"
        "@attribute age numeric\n"
        "@attribute class {tested_negative, tested_positive}\n"
        "@data\n"
        "0,100,70,20,80,28.0,0.2,21,tested_negative\n"
        "1,160,88,30,120,34.0,0.7,44,tested_positive\n"
    )
    csv_path = tmp_path / "out.csv"
    convert_arff_to_csv(arff, csv_path)
    df = pd.read_csv(csv_path)
    assert list(df.columns)[0] == "Pregnancies"
    assert df.loc[0, "Outcome"] == 0
    assert df.loc[1, "Outcome"] == 1
