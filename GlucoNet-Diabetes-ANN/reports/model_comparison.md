# Model comparison

> Educational model only — not a medical diagnostic tool. The Pima Indians Diabetes dataset is small (768 rows) and limited to adult women (age 21+) of Pima heritage. Results do not generalize to other populations, men, children, or clinical decision-making.

Metrics are **mean ± standard deviation** over stratified folds.
Out-of-fold (OOF) ROC-AUC / PR-AUC use concatenated fold predictions.
Feature selection and resampling, when enabled, are fit on the training
fold only. Numbers below are computed results, not a performance claim
made before training.

## Summary table

| Experiment | Accuracy | Precision | Recall/Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| dummy_most_frequent | 0.651 ± 0.002 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.349 ± 0.002 |
| logistic_regression__all_features__no_resample | 0.747 ± 0.045 | 0.623 ± 0.070 | 0.727 ± 0.052 | 0.758 ± 0.074 | 0.669 ± 0.045 | 0.836 ± 0.023 | 0.725 ± 0.041 |
| logistic_regression__all_features__oversample | 0.742 ± 0.053 | 0.619 ± 0.077 | 0.709 ± 0.043 | 0.760 ± 0.076 | 0.659 ± 0.053 | 0.834 ± 0.023 | 0.719 ± 0.041 |
| logistic_regression__kbest5__no_resample | 0.737 ± 0.048 | 0.613 ± 0.068 | 0.690 ± 0.056 | 0.762 ± 0.069 | 0.648 ± 0.053 | 0.826 ± 0.035 | 0.711 ± 0.057 |
| logistic_regression__kbest5__oversample | 0.737 ± 0.057 | 0.615 ± 0.079 | 0.686 ± 0.066 | 0.764 ± 0.075 | 0.646 ± 0.065 | 0.827 ± 0.036 | 0.715 ± 0.055 |
| random_forest__all_features__no_resample | 0.766 ± 0.038 | 0.636 ± 0.050 | 0.780 ± 0.037 | 0.758 ± 0.056 | 0.700 ± 0.037 | 0.837 ± 0.023 | 0.712 ± 0.020 |
| random_forest__all_features__oversample | 0.759 ± 0.042 | 0.631 ± 0.057 | 0.769 ± 0.038 | 0.754 ± 0.066 | 0.691 ± 0.038 | 0.832 ± 0.022 | 0.702 ± 0.013 |
| random_forest__kbest5__no_resample | 0.766 ± 0.038 | 0.640 ± 0.056 | 0.769 ± 0.021 | 0.764 ± 0.063 | 0.697 ± 0.032 | 0.837 ± 0.029 | 0.717 ± 0.026 |
| random_forest__kbest5__oversample | 0.762 ± 0.040 | 0.633 ± 0.058 | 0.776 ± 0.019 | 0.754 ± 0.064 | 0.696 ± 0.035 | 0.840 ± 0.030 | 0.725 ± 0.030 |
| mlp_ann__all_features__no_resample | 0.762 ± 0.033 | 0.698 ± 0.078 | 0.571 ± 0.058 | 0.864 ± 0.047 | 0.625 ± 0.048 | 0.832 ± 0.033 | 0.699 ± 0.063 |
| mlp_ann__all_features__oversample | 0.746 ± 0.046 | 0.613 ± 0.067 | 0.769 ± 0.021 | 0.734 ± 0.069 | 0.680 ± 0.044 | 0.834 ± 0.024 | 0.696 ± 0.056 |
| mlp_ann__kbest5__no_resample | 0.741 ± 0.033 | 0.657 ± 0.074 | 0.556 ± 0.054 | 0.840 ± 0.051 | 0.600 ± 0.043 | 0.789 ± 0.065 | 0.671 ± 0.066 |
| mlp_ann__kbest5__oversample | 0.751 ± 0.043 | 0.622 ± 0.058 | 0.750 ± 0.059 | 0.752 ± 0.063 | 0.678 ± 0.045 | 0.842 ± 0.028 | 0.727 ± 0.039 |

## Out-of-fold confusion matrices

### dummy_most_frequent

```
              Pred 0    Pred 1
Actual 0        500        0
Actual 1        268        0
```

### logistic_regression__all_features__no_resample

```
              Pred 0    Pred 1
Actual 0        379      121
Actual 1         73      195
```

### logistic_regression__all_features__oversample

```
              Pred 0    Pred 1
Actual 0        380      120
Actual 1         78      190
```

### logistic_regression__kbest5__no_resample

```
              Pred 0    Pred 1
Actual 0        381      119
Actual 1         83      185
```

### logistic_regression__kbest5__oversample

```
              Pred 0    Pred 1
Actual 0        382      118
Actual 1         84      184
```

### random_forest__all_features__no_resample

```
              Pred 0    Pred 1
Actual 0        379      121
Actual 1         59      209
```

### random_forest__all_features__oversample

```
              Pred 0    Pred 1
Actual 0        377      123
Actual 1         62      206
```

### random_forest__kbest5__no_resample

```
              Pred 0    Pred 1
Actual 0        382      118
Actual 1         62      206
```

### random_forest__kbest5__oversample

```
              Pred 0    Pred 1
Actual 0        377      123
Actual 1         60      208
```

### mlp_ann__all_features__no_resample

```
              Pred 0    Pred 1
Actual 0        432       68
Actual 1        115      153
```

### mlp_ann__all_features__oversample

```
              Pred 0    Pred 1
Actual 0        367      133
Actual 1         62      206
```

### mlp_ann__kbest5__no_resample

```
              Pred 0    Pred 1
Actual 0        420       80
Actual 1        119      149
```

### mlp_ann__kbest5__oversample

```
              Pred 0    Pred 1
Actual 0        376      124
Actual 1         67      201
```


## Features selected within each training fold

### logistic_regression__kbest5__no_resample
- Fold 1: Pregnancies, Glucose, Insulin, BMI, Age
- Fold 2: Pregnancies, Glucose, SkinThickness, BMI, Age
- Fold 3: Pregnancies, Glucose, SkinThickness, BMI, Age
- Fold 4: Glucose, SkinThickness, Insulin, BMI, Age
- Fold 5: Glucose, SkinThickness, Insulin, BMI, Age
- Selection frequency: Glucose (5), BMI (5), Age (5), SkinThickness (4), Pregnancies (3), Insulin (3)

### logistic_regression__kbest5__oversample
- Fold 1: Pregnancies, Glucose, Insulin, BMI, Age
- Fold 2: Pregnancies, Glucose, SkinThickness, BMI, Age
- Fold 3: Pregnancies, Glucose, SkinThickness, BMI, Age
- Fold 4: Glucose, SkinThickness, Insulin, BMI, Age
- Fold 5: Glucose, SkinThickness, Insulin, BMI, Age
- Selection frequency: Glucose (5), BMI (5), Age (5), SkinThickness (4), Pregnancies (3), Insulin (3)

### random_forest__kbest5__no_resample
- Fold 1: Pregnancies, Glucose, Insulin, BMI, Age
- Fold 2: Pregnancies, Glucose, SkinThickness, BMI, Age
- Fold 3: Pregnancies, Glucose, SkinThickness, BMI, Age
- Fold 4: Glucose, SkinThickness, Insulin, BMI, Age
- Fold 5: Glucose, SkinThickness, Insulin, BMI, Age
- Selection frequency: Glucose (5), BMI (5), Age (5), SkinThickness (4), Pregnancies (3), Insulin (3)

### random_forest__kbest5__oversample
- Fold 1: Pregnancies, Glucose, Insulin, BMI, Age
- Fold 2: Pregnancies, Glucose, SkinThickness, BMI, Age
- Fold 3: Pregnancies, Glucose, SkinThickness, BMI, Age
- Fold 4: Glucose, SkinThickness, Insulin, BMI, Age
- Fold 5: Glucose, SkinThickness, Insulin, BMI, Age
- Selection frequency: Glucose (5), BMI (5), Age (5), SkinThickness (4), Pregnancies (3), Insulin (3)

### mlp_ann__kbest5__no_resample
- Fold 1: Pregnancies, Glucose, Insulin, BMI, Age
- Fold 2: Pregnancies, Glucose, SkinThickness, BMI, Age
- Fold 3: Pregnancies, Glucose, SkinThickness, BMI, Age
- Fold 4: Glucose, SkinThickness, Insulin, BMI, Age
- Fold 5: Glucose, SkinThickness, Insulin, BMI, Age
- Selection frequency: Glucose (5), BMI (5), Age (5), SkinThickness (4), Pregnancies (3), Insulin (3)

### mlp_ann__kbest5__oversample
- Fold 1: Pregnancies, Glucose, Insulin, BMI, Age
- Fold 2: Pregnancies, Glucose, SkinThickness, BMI, Age
- Fold 3: Pregnancies, Glucose, SkinThickness, BMI, Age
- Fold 4: Glucose, SkinThickness, Insulin, BMI, Age
- Fold 5: Glucose, SkinThickness, Insulin, BMI, Age
- Selection frequency: Glucose (5), BMI (5), Age (5), SkinThickness (4), Pregnancies (3), Insulin (3)

## How to read these metrics

- **Accuracy** can look high if the model mostly predicts the majority class.
- **Recall / sensitivity** is the share of true diabetes cases found.
- **Specificity** is the share of true non-diabetes cases correctly kept negative.
- **PR-AUC** is often more informative than ROC-AUC on imbalanced data.
- The dummy most-frequent model is a sanity floor, not a clinical baseline.
