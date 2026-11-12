# GlucoNet — Diabetes Risk Screening with a Transparent ANN

Educational web app and research pipeline for **diabetes risk ranking** on the Pima Indians Diabetes table (768 adult women). A small feed-forward network (**8 → 16 → 8 → 1**, 289 weights) is trained inside a leakage-safe sklearn / imbalanced-learn pipeline and compared with logistic regression and a random forest on the same folds.

This is **not a diagnostic tool**. The output is a probability from one historical cohort. It does not replace fasting glucose, OGTT, HbA1c, or a physician’s judgement.

**Course:** 21CSE326T Artificial Neural Networks  
**Programme:** B.Tech. CSE (AIML), SRM Institute of Science and Technology, Tiruchirappalli  
**Authors:** Shanmugapriyan J (RA2411026050011) · Deepak V F (RA2411026050015)

## What to show in a viva

| What | Path |
|------|------|
| Trained model (open this file) | [`final_report/results/proposed_mlp_pipeline.joblib`](final_report/results/proposed_mlp_pipeline.joblib) |
| Network definition | [`src/diabetes_prediction/train.py`](src/diabetes_prediction/train.py) (`MLPClassifier`, hidden `(16, 8)`) |
| Preprocessing pipeline | [`src/diabetes_prediction/preprocess.py`](src/diabetes_prediction/preprocess.py) |
| Local demo server | [`demo/app.py`](demo/app.py) |
| GlucoNet UI | [`demo/index.html`](demo/index.html) |
| Classroom notebook | [`notebooks/Diabetes_Prediction_ANN_Project_Notebook.ipynb`](notebooks/Diabetes_Prediction_ANN_Project_Notebook.ipynb) |
| Report | [`final_report/Diabetes_Prediction_ANN_Project_Report.pdf`](final_report/Diabetes_Prediction_ANN_Project_Report.pdf) |
| Slides | [`final_report/Diabetes_Prediction_ANN_Project_Presentation.pptx`](final_report/Diabetes_Prediction_ANN_Project_Presentation.pptx) |

Headline result (10 × 5 repeated stratified CV, proposed MLP + training-fold oversampling): **ROC-AUC 0.834**, **sensitivity 0.768**. Hold-out: 43 of 54 diabetic cases, ROC-AUC **0.821**. Dummy majority accuracy is already **0.651**, so accuracy alone is not the story.

## Quick start

Python 3.9+ (3.11 used in development).

```bash
cd GlucoNet-Diabetes-ANN
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Run the live demo (GlucoNet)

```bash
python demo/app.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The page posts the eight biomarkers to `/predict`; the server loads the saved `.joblib` and returns `predict_proba`. No fake formula.

Worked examples (same records as the report):

- Higher glucose: 2 pregnancies, glucose 168, BMI 36.4, age 47 → about **91%**
- Lower glucose: 1 pregnancy, glucose 94, BMI 24.1, age 26 → about **2%**

### Score without a browser

```bash
python final_report/predict_demo.py
```

### Retrain / reproduce the comparison

The CSV is already in [`data/pima_indians_diabetes.csv`](data/pima_indians_diabetes.csv). The code never downloads data.

```bash
python run_experiment.py --seed 42 --splits 5 --select-k 5
PYTHONPATH=src python -m pytest -q
```

The saved pipeline used by the website was fitted in [`final_report/experiments.py`](final_report/experiments.py) (proposed MLP + ROS on the 614-row training split).

## Repository layout

```text
GlucoNet-Diabetes-ANN/
├── README.md
├── requirements.txt
├── pyproject.toml
├── run_experiment.py              # CLI: profile + model comparison
├── data/pima_indians_diabetes.csv # local Pima table (768 × 8 + Outcome)
├── src/diabetes_prediction/       # load, preprocess, train, evaluate
├── demo/                          # GlucoNet site + /predict backend
├── notebooks/                     # executed project notebook
├── tests/                         # leakage and zero-handling checks
├── reports/                       # metrics and EDA figures from a run
└── final_report/
    ├── results/proposed_mlp_pipeline.joblib
    ├── Diabetes_Prediction_ANN_Project_Report.pdf
    └── Diabetes_Prediction_ANN_Project_Presentation.pptx
```

## Model (what is inside the `.joblib`)

Pipeline steps, fitted on the **training split only**:

1. Recode medically impossible zeros as missing (glucose, blood pressure, skinfold, insulin, BMI). **Pregnancies = 0 is kept.**
2. Median imputation
3. Standard scaling
4. Random oversampling of the minority class (active at fit; idle at predict)
5. `MLPClassifier`: input 8, hidden 16 ReLU, hidden 8 ReLU, output 1 logistic sigmoid. Adam, L2 `10⁻³`, early stopping.

Baselines share the same folds: dummy (majority class), logistic regression, random forest.

Permutation importance on hold-out ROC-AUC is dominated by **glucose (0.120)**, then age, BMI, and pregnancies.

## Dataset

Pima Indians Diabetes Database (NIDDK; UCI extract). 768 examinations of adult women, age 21+. Target: WHO diabetes criteria from the original study (`Outcome` 0/1). See [`data/README.md`](data/README.md).

## Medical limits

- Educational / screening-style illustration only.
- 768 rows is small for a neural network.
- One community, adult women only — not men, not children, not a contemporary mixed clinic.
- Insulin and skinfold are heavily missing once zeros are recoded; median fill-in is a modelling convenience.

If a medical decision is needed, use a qualified clinician and an appropriate diagnostic test.

## Upload to GitHub

From this folder (Git is already initialised):

```bash
cd ~/Desktop/GlucoNet-Diabetes-ANN
gh repo create GlucoNet-Diabetes-ANN --public --source=. --remote=origin --push
```

Or create an empty repository on github.com, then:

```bash
cd ~/Desktop/GlucoNet-Diabetes-ANN
git remote add origin https://github.com/<your-username>/GlucoNet-Diabetes-ANN.git
git push -u origin main
```
