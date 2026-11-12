from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from diabetes_prediction.config import ALL_COLUMNS


@pytest.fixture
def tiny_diabetes_frame() -> pd.DataFrame:
    """Tiny synthetic table with valid zeros, implausible zeros, and both classes."""
    rows = [
        # Pregnancies=0 must remain valid; Glucose=0 should become missing.
        [0, 0, 70, 20, 80, 32.0, 0.4, 25, 0],
        [1, 140, 80, 30, 120, 28.0, 0.5, 30, 1],
        [2, 90, 0, 18, 0, 24.0, 0.2, 22, 0],
        [3, 160, 88, 0, 200, 35.0, 0.8, 45, 1],
        [0, 100, 72, 22, 90, 0.0, 0.3, 28, 0],
        [4, 180, 90, 33, 150, 31.0, 1.1, 50, 1],
        [1, 110, 68, 19, 70, 26.5, 0.25, 33, 0],
        [5, 155, 76, 28, 130, 34.2, 0.7, 41, 1],
        [2, 95, 64, 21, 60, 22.1, 0.15, 24, 0],
        [6, 170, 92, 31, 190, 36.8, 0.9, 48, 1],
        [0, 88, 60, 15, 50, 21.0, 0.12, 21, 0],
        [3, 148, 74, 25, 110, 30.5, 0.55, 36, 1],
    ]
    return pd.DataFrame(rows, columns=ALL_COLUMNS)


@pytest.fixture
def tiny_csv(tmp_path: Path, tiny_diabetes_frame: pd.DataFrame) -> Path:
    path = tmp_path / "tiny_diabetes.csv"
    tiny_diabetes_frame.to_csv(path, index=False)
    return path
