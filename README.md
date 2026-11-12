# GlucoNet — Diabetes Prediction using an Artificial Neural Network

GlucoNet is a complete academic project for **diabetes risk ranking** from eight routine clinical measurements. A small feed-forward neural network (**8 → 16 → 8 → 1**, 289 trainable weights) is trained on the Pima Indians Diabetes dataset inside a leakage-safe preprocessing pipeline and compared with logistic regression and a random forest on identical folds. The repository includes the source code, trained model, interactive web demonstration, executed notebook, written report, and slides.

This tool is for **education and examination only**. It is not a medical device and does not diagnose diabetes.

| | |
|---|---|
| **Course** | 21CSE326T Artificial Neural Networks |
| **Programme** | B.Tech. CSE (AIML), SRM Institute of Science and Technology, Tiruchirappalli |
| **Team** | Shanmugapriyan J (RA2411026050011), Deepak V F (RA2411026050015) |

## Objective

Predict the probability that a record belongs to the diabetic class (`Outcome = 1`) from eight biomarkers: pregnancies, plasma glucose, diastolic blood pressure, triceps skinfold, serum insulin, body-mass index, diabetes pedigree function, and age. The network is kept small because the table has only 768 rows. Imputation, scaling, and oversampling are fitted on the training fold only so test rows never leak into the model.

## Dataset

[Pima Indians Diabetes Database](data/pima_indians_diabetes.csv) (NIDDK; UCI extract). 768 examinations of adult women, age 21+, Pima heritage. Target: WHO diabetes criteria from the original study (`0` = no diabetes, `1` = diabetes). Class split is about 65 : 35, so always predicting the majority class already scores accuracy **0.651**.

Zeros are missing-value codes for glucose, blood pressure, skinfold, insulin, and BMI. **Pregnancies = 0 is a real value** (never pregnant) and is left unchanged. Details: [`data/README.md`](data/README.md).

## Method

Every model uses the same stratified folds and the same pipeline, fitted on the training split only:

1. Recode medically impossible zeros as missing (not pregnancies).
2. Median imputation.
3. Standard scaling.
4. Optional random oversampling of the minority class (training fold only; idle at prediction time).
5. Classifier.

**Proposed network** (`MLPClassifier` in [`src/diabetes_prediction/train.py`](src/diabetes_prediction/train.py)):

- Input layer: 8 biomarkers
- Hidden layer 1: 16 units, ReLU
- Hidden layer 2: 8 units, ReLU
- Output: 1 unit, logistic sigmoid (class-1 probability)
- Adam, L2 penalty `10⁻³`, batch size 32, early stopping

Baselines on the same folds: dummy (majority class), logistic regression, random forest.

The fitted pipeline used by the website is [`final_report/results/proposed_mlp_pipeline.joblib`](final_report/results/proposed_mlp_pipeline.joblib) (steps: `zero_as_missing → impute → scale → resample → clf`).

## Results

Repeated stratified cross-validation (**10 × 5 = 50 splits**), proposed MLP with training-fold oversampling:

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Dummy (majority) | 0.651 | 0.000 | 0.000 | 0.500 |
| Logistic regression | 0.754 | 0.623 | 0.718 | 0.836 |
| Random forest | 0.761 | 0.636 | 0.761 | 0.838 |
| **GlucoNet ANN + ROS** | **0.746** | **0.613** | **0.768** | **0.834** |

Hold-out (154 rows never used in fitting): **43 of 54** diabetic cases recovered, ROC-AUC **0.821**. Accuracy sits only a little above the 0.651 majority floor, so **ranking (ROC-AUC) and sensitivity** are the useful numbers.

Permutation importance (drop in hold-out ROC-AUC): glucose **0.120**, then age **0.028**, BMI **0.026**, pregnancies **0.024**. Glucose carries most of the signal.

## Contents of this repository

| Item | Location |
|------|----------|
| Source (load, preprocess, train, evaluate) | [`src/diabetes_prediction/`](src/diabetes_prediction/) |
| Trained ANN pipeline | [`final_report/results/proposed_mlp_pipeline.joblib`](final_report/results/proposed_mlp_pipeline.joblib) |
| Interactive web application | [`demo/app.py`](demo/app.py), [`demo/index.html`](demo/index.html) |
| Executed project notebook | [`notebooks/Diabetes_Prediction_ANN_Project_Notebook.ipynb`](notebooks/Diabetes_Prediction_ANN_Project_Notebook.ipynb) |
| Unit tests (zero handling, no leakage) | [`tests/`](tests/) |
| Project report | [`final_report/Diabetes_Prediction_ANN_Project_Report.pdf`](final_report/Diabetes_Prediction_ANN_Project_Report.pdf) |
| Presentation | [`final_report/Diabetes_Prediction_ANN_Project_Presentation.pptx`](final_report/Diabetes_Prediction_ANN_Project_Presentation.pptx) |
| Local dataset | [`data/pima_indians_diabetes.csv`](data/pima_indians_diabetes.csv) |

```text
GlucoNet-Diabetes-ANN/
├── data/                      Pima CSV
├── src/diabetes_prediction/   Python package
├── demo/                      GlucoNet web app (loads the .joblib)
├── notebooks/                 executed notebook
├── tests/                     pytest
├── reports/                   CV metrics and EDA figures
└── final_report/              report, slides, trained model
```

## How to run

Python 3.9 or newer. The CSV is already in `data/`; nothing is downloaded.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

**Web application.** Scores the eight biomarkers with the saved network (not a stand-in formula).

```bash
python demo/app.py
```

Open http://127.0.0.1:8765. Example records from the report: higher glucose (168 mg/dL, BMI 36.4, age 47) scores about **91%**; lower glucose (94 mg/dL, BMI 24.1, age 26) scores about **2%**.

**Command-line scoring** of held-out rows:

```bash
python final_report/predict_demo.py
```

**Retrain and tests:**

```bash
python run_experiment.py --seed 42 --splits 5 --select-k 5
PYTHONPATH=src python -m pytest -q
```

## Disclaimer

Educational screening illustration only. The sample is adult women from one community, examined decades ago. Do not apply the score to men, children, or other populations, and do not use it for treatment decisions. A physician and a diagnostic test (fasting glucose, OGTT, or HbA1c) remain the clinical standard.

MIT License. See [`LICENSE`](LICENSE).
