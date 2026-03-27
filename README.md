# Railway Delay Analysis

A data mining project for railway delay prediction using classical machine learning, gradient boosting, and deep learning approaches.

## What this repository contains

This repository is organized around two main analysis notebooks plus supporting artifacts:

- `notebooks/25MSA23234_DuongBinhAn_Fall25.ipynb`: main end-to-end experimentation notebook (EDA, feature engineering, training, comparison, export).
- `notebooks/regression_pipeline_rmse.ipynb`: regression-focused pipeline and evaluation flow.
- `src/utils/feature_helpers.py`: reusable helper functions for lag and rolling delay features.
- `reports/railway_delay_analysis_report.md`: written summary report.

## Clean project structure

```text
railway-delay/
|-- README.md
|-- .gitignore
|-- docs/
|   |-- README.md
|-- reports/
|   |-- railway_delay_analysis_report.md
|-- src/
|   |-- README.md
|   |-- utils/
|       |-- feature_helpers.py
|-- notebooks/
|   |-- 25MSA23234_DuongBinhAn_Fall25.ipynb
|   |-- regression_pipeline_clean.ipynb
|   |-- regression_pipeline_rmse.ipynb
|   |-- figures/
|   |-- models/
|   |-- catboost_info/
|-- models/
|-- .venv/ (local environment, not tracked)
|-- miniconda/ (local environment, not tracked)
```

## Environment setup

### 1. Python version

Recommended: Python 3.10+ (project also has local environments under `.venv` and `miniconda`).

### 2. Install dependencies

If you use an existing local environment:

```powershell
.\.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm catboost shap tensorflow joblib jupyter
```

If you prefer Conda:

```powershell
conda activate <your-env>
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm catboost shap tensorflow joblib jupyter
```

## How to run

### Run notebooks

```powershell
jupyter notebook
```

Then execute either:

1. `notebooks/25MSA23234_DuongBinhAn_Fall25.ipynb` for full training/benchmarking workflow.
2. `notebooks/regression_pipeline_rmse.ipynb` for regression RMSE-focused workflow.

### Notebook output locations

- Models: `notebooks/models/`
- Figures: `notebooks/figures/`
- Training logs (CatBoost): `notebooks/catboost_info/`

## Core workflow summary

1. Data loading and preprocessing.
2. Feature engineering (including lag/rolling features in `src/utils/feature_helpers.py`).
3. Train multiple model families.
4. Tune and compare model performance.
5. Export best models and supporting artifacts.
6. Generate diagnostic plots and metrics summaries.

## Important notes

- Large datasets are intentionally ignored from Git and should live under `data/` locally.
- Generated models/plots can be large; keep only final artifacts you need.
- `__pycache__`, temporary notebook backups, and one-off fix scripts were removed during cleanup.
- `.gitignore` has been tightened to prevent committing environment/cache noise.

## Reproducibility checklist

- Use a clean environment.
- Keep raw and processed data paths consistent.
- Execute notebook cells from top to bottom.
- Re-run full training if models in `notebooks/models/` are deleted.

## Documentation

- Main analysis report: `reports/railway_delay_analysis_report.md`
- Additional data notes: `docs/README.md`
- Source code guidance: `src/README.md`

## Author

MSE Data Mining Project (Academic)
