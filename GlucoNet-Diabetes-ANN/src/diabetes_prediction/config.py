"""Project constants, paths, and reproducibility settings.

Change RANDOM_SEED here (and keep the rest of the pipeline unchanged) to
re-run the experiment with a different but still fully specified seed.
"""

from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 42
N_SPLITS = 5
SELECT_K = 5

# Standard Pima Indians Diabetes column names used throughout this project.
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COLUMN = "Outcome"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# 0 is a historically used missing-value code for these measurements.
# Pregnancies is intentionally excluded: 0 pregnancies is medically valid.
ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

ZERO_AS_MISSING_JUSTIFICATION = {
    "Glucose": (
        "A recorded plasma glucose of 0 mg/dL is not a plausible OGTT result "
        "and is treated as a missing measurement in this dataset."
    ),
    "BloodPressure": (
        "A diastolic blood pressure of 0 mm Hg is not a valid clinical reading "
        "in living adults, so 0 is treated as missing."
    ),
    "SkinThickness": (
        "A triceps skinfold of 0 mm is not physiologically plausible, so 0 is "
        "treated as a missing measurement."
    ),
    "Insulin": (
        "A 2-hour serum insulin of 0 is the missing-value code used in this "
        "dataset; it is not interpreted as a true lab result of zero."
    ),
    "BMI": (
        "A body-mass index of 0 is not a valid measurement, so 0 is treated "
        "as missing."
    ),
}

VALID_ZERO_COLUMNS = ["Pregnancies", "DiabetesPedigreeFunction", "Age"]

COLUMN_ALIASES = {
    "pregnancies": "Pregnancies",
    "preg": "Pregnancies",
    "glucose": "Glucose",
    "plas": "Glucose",
    "bloodpressure": "BloodPressure",
    "blood_pressure": "BloodPressure",
    "pres": "BloodPressure",
    "skinthickness": "SkinThickness",
    "skin_thickness": "SkinThickness",
    "skin": "SkinThickness",
    "insulin": "Insulin",
    "insu": "Insulin",
    "bmi": "BMI",
    "mass": "BMI",
    "diabetespedigreefunction": "DiabetesPedigreeFunction",
    "diabetes_pedigree_function": "DiabetesPedigreeFunction",
    "dpf": "DiabetesPedigreeFunction",
    "pedi": "DiabetesPedigreeFunction",
    "age": "Age",
    "outcome": "Outcome",
    "class": "Outcome",
    "diabetes": "Outcome",
}

POSITIVE_LABELS = {
    "1",
    "true",
    "yes",
    "positive",
    "tested_positive",
    "diabetes",
}
NEGATIVE_LABELS = {
    "0",
    "false",
    "no",
    "negative",
    "tested_negative",
}

CSV_FILENAME = "pima_indians_diabetes.csv"
ARFF_FILENAME = "diabetes.arff"

DISCLAIMER = (
    "Educational model only — not a medical diagnostic tool. The Pima Indians "
    "Diabetes dataset is small (768 rows) and limited to adult women (age 21+) "
    "of Pima heritage. Results do not generalize to other populations, men, "
    "children, or clinical decision-making."
)


def project_root() -> Path:
    """Return the repository root (this project folder)."""
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    """Return the repository root. Kept as an alias for older call sites."""
    return project_root()


def default_data_candidates() -> list[Path]:
    """Local CSV/ARFF locations we may use. Never downloads data."""
    root = project_root()
    workspace = workspace_root()
    return [
        root / "data" / CSV_FILENAME,
        workspace / "data" / CSV_FILENAME,
        workspace / "data" / ARFF_FILENAME,
        root / "data" / ARFF_FILENAME,
    ]


def reports_dir() -> Path:
    path = project_root() / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir() -> Path:
    path = reports_dir() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path
