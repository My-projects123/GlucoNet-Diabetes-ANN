"""Diabetes prediction with a small feed-forward neural network."""

from diabetes_prediction.config import DISCLAIMER, RANDOM_SEED
from diabetes_prediction.data import load_dataset, split_xy
from diabetes_prediction.profile import profile_dataframe

__all__ = [
    "DISCLAIMER",
    "RANDOM_SEED",
    "load_dataset",
    "profile_dataframe",
    "split_xy",
]
