# Dataset: Pima Indians Diabetes

The CSV used by this project is already in this folder:

```text
data/pima_indians_diabetes.csv
```

The code **does not download** data. If this file is missing, loaders stop and print this path.

## Expected layout

Header row plus 768 data rows (a headerless 9-column file is also accepted):

```text
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
```

`Outcome` is `0` (no diabetes) or `1` (diabetes). Weka labels `tested_negative` / `tested_positive` are mapped automatically.

## Source

- National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- UCI Machine Learning Repository extract (*Pima Indians Diabetes Database*)
- Smith et al., 1988, ADAP neural model on this cohort

## Who is in this table

- 768 rows
- Adult women, age 21+
- Pima heritage, near Phoenix, Arizona, USA
- Binary target: whether the person met WHO diabetes criteria in the original study

This is a small, historically specific sample. It is **not** a general-population medical dataset.

## Zeros that are not real measurements

Several columns use `0` as a missing-value code. The preprocessing pipeline recodes those zeros as missing **only** for:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

`Pregnancies = 0` is kept. It means the person has never been pregnant.
