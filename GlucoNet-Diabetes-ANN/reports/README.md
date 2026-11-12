# Generated reports

This folder is filled when you run:

```bash
python run_experiment.py
```

Typical outputs:

| File | Contents |
|------|----------|
| `data_profile.md` / `.json` | Class balance, duplicates, missing values, zero counts |
| `model_comparison.md` | Stratified CV metrics and confusion matrices |
| `metrics.csv` / `metrics.json` | Machine-readable fold summaries |
| `figures/` | EDA and evaluation plots (including SelectKBest frequency) |

A completed run is already included here so the GitHub copy can be inspected without retraining.
