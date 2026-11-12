"""Seed helpers so reruns with the same inputs match."""

from __future__ import annotations

import os
import random
import warnings

import numpy as np

from diabetes_prediction.config import RANDOM_SEED


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy, and hash randomization as far as this stack allows.

    scikit-learn models in this project also receive the same `random_state`.
    TensorFlow / PyTorch are not used, so they are not seeded.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def configure_runtime(seed: int = RANDOM_SEED) -> None:
    """Seed RNGs and hide a known NumPy 2 / scikit-learn matmul warning flood."""
    seed_everything(seed)
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=r".*(divide by zero|overflow|invalid value) encountered in matmul",
    )
    # imbalanced-learn 0.12 still calls sklearn 1.6 private helpers when a
    # newer sklearn is installed. Prefer sklearn<1.6; this keeps older mixes usable.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"sklearn\.(utils\.deprecation|base)",
    )
