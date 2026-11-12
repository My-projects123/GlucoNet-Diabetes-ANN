"""Score held-out rows with the saved proposed ANN pipeline.

The pipeline was fitted on the 614-row training split by experiments.py and
has never seen these test rows. Educational output only, not a diagnosis.

Usage:
    cd diabetes_ann && .venv/bin/python final_report/predict_demo.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from diabetes_prediction.config import DISCLAIMER  # noqa: E402
from diabetes_prediction.data import load_dataset  # noqa: E402

warnings.filterwarnings("ignore")

SHORT = {
    "Pregnancies": "Preg",
    "Glucose": "Gluc",
    "BloodPressure": "BP",
    "SkinThickness": "Skin",
    "Insulin": "Insu",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "DPF",
    "Age": "Age",
}


def main() -> int:
    bundle = joblib.load(ROOT / "results" / "proposed_mlp_pipeline.joblib")
    pipe = bundle["pipeline"]
    features = bundle["feature_columns"]
    df = load_dataset()
    test = df.loc[bundle["test_index"]]
    sample = test.head(10)

    proba = pipe.predict_proba(sample[features])[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("Proposed ANN: 8 -> 16 (ReLU) -> 8 (ReLU) -> 1 (sigmoid), trained on 614 rows")
    print(f"Loaded pipeline steps: {' -> '.join(pipe.named_steps)}")
    print()
    header = " ".join(f"{SHORT[c]:>6}" for c in features)
    print(f"{'Row':>4} {header} {'P(diab)':>8} {'Pred':>5} {'True':>5}  Result")
    print("-" * (4 + 7 * len(features) + 30))
    decimals = {"DiabetesPedigreeFunction": 3, "BMI": 1}
    for (idx, row), p, yhat in zip(sample.iterrows(), proba, pred):
        values = " ".join(f"{row[c]:>6.{decimals.get(c, 0)}f}" for c in features)
        truth = int(row["Outcome"])
        verdict = "correct" if yhat == truth else ("missed case" if truth == 1 else "false alarm")
        print(f"{idx:>4} {values} {p:>8.3f} {yhat:>5d} {truth:>5d}  {verdict}")

    all_proba = pipe.predict_proba(test[features])[:, 1]
    all_pred = (all_proba >= 0.5).astype(int)
    correct = int((all_pred == test["Outcome"].to_numpy()).sum())
    print()
    print(f"Hold-out rows scored: {len(test)}; correct at threshold 0.5: {correct} "
          f"({correct / len(test):.1%})")

    example = pd.DataFrame(
        [
            {"Pregnancies": 2, "Glucose": 168, "BloodPressure": 74, "SkinThickness": 0,
             "Insulin": 0, "BMI": 36.4, "DiabetesPedigreeFunction": 0.62, "Age": 47},
            {"Pregnancies": 1, "Glucose": 94, "BloodPressure": 68, "SkinThickness": 22,
             "Insulin": 80, "BMI": 24.1, "DiabetesPedigreeFunction": 0.21, "Age": 26},
        ]
    )[features]
    print()
    print("Illustrative new inputs (zeros in Skin/Insulin are treated as missing):")
    for i, p in enumerate(pipe.predict_proba(example)[:, 1], 1):
        label = "Outcome = 1 (higher risk)" if p >= 0.5 else "Outcome = 0 (lower risk)"
        print(f"  Record {i}: P(diabetes) = {p:.3f} -> {label}")
    print()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
