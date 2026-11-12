# Data profile — Pima Indians Diabetes

> Educational model only — not a medical diagnostic tool. The Pima Indians Diabetes dataset is small (768 rows) and limited to adult women (age 21+) of Pima heritage. Results do not generalize to other populations, men, children, or clinical decision-making.

## Shape and duplicates

- Rows: 768
- Columns: 9
- Duplicate rows: 0

## Class balance

- Outcome=0 (no diabetes): 500
- Outcome=1 (diabetes): 268
- Positive rate: 0.349
- The minority class is Outcome=1 (diabetes). Accuracy alone can look optimistic because predicting 'no diabetes' for every row would already score about 65.1% on this table.

## Missing values (true NaN in the file)

- Pregnancies: 0
- Glucose: 0
- BloodPressure: 0
- SkinThickness: 0
- Insulin: 0
- BMI: 0
- DiabetesPedigreeFunction: 0
- Age: 0

## Zero counts (raw file)

- Pregnancies: 111
- Glucose: 5
- BloodPressure: 35
- SkinThickness: 227
- Insulin: 374
- BMI: 11
- DiabetesPedigreeFunction: 0
- Age: 0

## Zeros recoded as missing (justified)

These fields use 0 as a missing-value code. Recoding happens inside the
training pipeline so the rule is applied the same way on every fold.

- **Glucose**: 5 zeros. A recorded plasma glucose of 0 mg/dL is not a plausible OGTT result and is treated as a missing measurement in this dataset.
- **BloodPressure**: 35 zeros. A diastolic blood pressure of 0 mm Hg is not a valid clinical reading in living adults, so 0 is treated as missing.
- **SkinThickness**: 227 zeros. A triceps skinfold of 0 mm is not physiologically plausible, so 0 is treated as a missing measurement.
- **Insulin**: 374 zeros. A 2-hour serum insulin of 0 is the missing-value code used in this dataset; it is not interpreted as a true lab result of zero.
- **BMI**: 11 zeros. A body-mass index of 0 is not a valid measurement, so 0 is treated as missing.

## Zeros kept

Pregnancies=0 is kept. It means the person has never been pregnant, which is a valid value in this dataset of adult women.

- Pregnancies: 111 zeros kept
- DiabetesPedigreeFunction: 0 zeros kept
- Age: 0 zeros kept

## Missingness after zero recoding

- Pregnancies: 0 (0.0%)
- Glucose: 5 (0.7%)
- BloodPressure: 35 (4.6%)
- SkinThickness: 227 (29.6%)
- Insulin: 374 (48.7%)
- BMI: 11 (1.4%)
- DiabetesPedigreeFunction: 0 (0.0%)
- Age: 0 (0.0%)

Columns with at least 20% missing after recoding: SkinThickness, Insulin. Median imputation will fill these, but the imputed values
are a modeling convenience, not recovered lab measurements.

## IQR outlier counts (1.5×IQR on raw values)

These counts use the raw file, so coded-missing zeros (especially
BloodPressure and Insulin) can appear as outliers. That is expected.

- Pregnancies: 4
- Glucose: 5
- BloodPressure: 45
- SkinThickness: 1
- Insulin: 34
- BMI: 19
- DiabetesPedigreeFunction: 29
- Age: 9

## Summary statistics (raw values, zeros still present)

```
       Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin         BMI  DiabetesPedigreeFunction         Age
count   768.000000  768.000000     768.000000     768.000000  768.000000  768.000000                768.000000  768.000000
mean      3.845052  120.894531      69.105469      20.536458   79.799479   31.992578                  0.471876   33.240885
std       3.369578   31.972618      19.355807      15.952218  115.244002    7.884160                  0.331329   11.760232
min       0.000000    0.000000       0.000000       0.000000    0.000000    0.000000                  0.078000   21.000000
25%       1.000000   99.000000      62.000000       0.000000    0.000000   27.300000                  0.243750   24.000000
50%       3.000000  117.000000      72.000000      23.000000   30.500000   32.000000                  0.372500   29.000000
75%       6.000000  140.250000      80.000000      32.000000  127.250000   36.600000                  0.626250   41.000000
max      17.000000  199.000000     122.000000      99.000000  846.000000   67.100000                  2.420000   81.000000
```
