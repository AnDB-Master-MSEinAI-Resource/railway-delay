# Professional Folder Organization Script
# Railway Delay Analysis Project

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan
Write-Host "  ORGANIZING PROJECT FILES INTO PROFESSIONAL STRUCTURE" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan

# Move raw data
Write-Host "`n[1/5] Moving raw data files..." -ForegroundColor Green
Move-Item -Path "railway-delay-dataset.csv" -Destination "data\raw\" -Force -ErrorAction SilentlyContinue

# Move processed data
Write-Host "[2/5] Moving processed data files..." -ForegroundColor Green
Move-Item -Path "train_data.csv" -Destination "data\processed\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "test_data.csv" -Destination "data\processed\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "merged_train_data.csv" -Destination "data\processed\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "dirty_train_data.csv" -Destination "data\interim\" -Force -ErrorAction SilentlyContinue

# Move notebooks
Write-Host "[3/5] Moving notebook files..." -ForegroundColor Green
Move-Item -Path "railway_delay_analysis.ipynb" -Destination "notebooks\" -Force -ErrorAction SilentlyContinue

# Move figures
Write-Host "[4/5] Moving visualization files..." -ForegroundColor Green
Move-Item -Path "*.png" -Destination "results\figures\" -Force -ErrorAction SilentlyContinue

# Create README files
Write-Host "[5/5] Creating README files..." -ForegroundColor Green

# Main README
$mainReadme = @"
# Railway Delay Analysis - Data Mining Project

## 📋 Project Overview
Comprehensive data mining and machine learning analysis for predicting railway delays using advanced techniques including traditional ML, deep learning, hyperparameter tuning, and clustering.

## 🗂️ Project Structure
``````
railway-delay/
│
├── data/                      # Data directory
│   ├── raw/                   # Original, immutable data
│   ├── interim/               # Intermediate data (dirty, experimental)
│   └── processed/             # Final processed data ready for modeling
│
├── notebooks/                 # Jupyter/IPython notebooks
│   └── railway_delay_analysis.ipynb
│
├── src/                       # Source code for use in this project
│   ├── data/                  # Scripts to download or generate data
│   ├── features/              # Scripts for feature engineering
│   ├── models/                # Scripts to train models
│   └── visualization/         # Scripts to create visualizations
│
├── models/                    # Trained and serialized models
│
├── results/                   # Analysis results
│   ├── figures/               # Generated graphics and figures
│   └── metrics/               # Model performance metrics
│
├── docs/                      # Documentation
│
├── .gitignore                 # Git ignore file
└── README.md                  # This file
``````

## 🚀 Getting Started

### Prerequisites
``````bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
``````

### Running the Analysis
1. Open ``notebooks/railway_delay_analysis.ipynb``
2. Run cells sequentially from top to bottom
3. Results will be saved in ``results/`` directory

## 📊 Models Implemented
- **Traditional ML**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, Naive Bayes
- **Deep Learning**: Multi-layer Neural Network with Dropout and Batch Normalization
- **Clustering**: K-Means, DBSCAN with PCA visualization
- **Optimization**: Grid Search, Cross-Validation

## 📈 Key Results
- Best Model Performance: Check ``results/metrics/``
- Visualizations: Check ``results/figures/``
- Feature Importance: Analyzed in notebook

## 👥 Author
Data Mining Project - MSE Program

## 📄 License
Educational Project
"@

Set-Content -Path "README.md" -Value $mainReadme -Force

# Data README
$dataReadme = @"
# Data Directory

## Structure
- **raw/**: Original datasets (never modify)
- **interim/**: Intermediate data transformations
- **processed/**: Final clean datasets for modeling

## Files

### Raw Data
- ``railway-delay-dataset.csv``: Original railway delay dataset

### Processed Data
- ``train_data.csv``: Training dataset (80%)
- ``test_data.csv``: Test dataset (20%)
- ``merged_train_data.csv``: Combined training data with clean and dirty samples

### Interim Data
- ``dirty_train_data.csv``: Training data with intentionally injected errors for data cleaning exercises

## Data Schema
Refer to ``../docs/data_schema.md`` for detailed field descriptions.
"@

Set-Content -Path "data\README.md" -Value $dataReadme -Force

# Source code README
$srcReadme = @"
# Source Code Directory

## Structure
Create subdirectories as needed:
- ``data/``: Data processing scripts
- ``features/``: Feature engineering modules
- ``models/``: Model training and evaluation scripts
- ``visualization/``: Plotting and visualization utilities

## Usage
Place reusable Python modules here to keep notebooks clean and organized.
"@

Set-Content -Path "src\README.md" -Value $srcReadme -Force

# Models README
$modelsReadme = @"
# Models Directory

Store trained model files here:
- Serialized models (.pkl, .h5, .joblib)
- Model checkpoints
- Model configurations

## Naming Convention
``{model_name}_{date}_{performance}.ext``

Example: ``random_forest_20251130_f1_0.85.pkl``
"@

Set-Content -Path "models\README.md" -Value $modelsReadme -Force

# Results README
$resultsReadme = @"
# Results Directory

## Structure
- **figures/**: All generated visualizations (PNG, SVG)
- **metrics/**: Model performance metrics (CSV, JSON)

## Figures
All plots and charts generated during analysis.

## Metrics
Performance metrics for all trained models.
"@

Set-Content -Path "results\README.md" -Value $resultsReadme -Force

Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan
Write-Host "  ✅ PROJECT ORGANIZATION COMPLETE!" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan

Write-Host "`n📁 Professional folder structure created:" -ForegroundColor Yellow
Write-Host "   ├── data/" -ForegroundColor Cyan
Write-Host "   │   ├── raw/          (original data)" -ForegroundColor Gray
Write-Host "   │   ├── interim/      (intermediate data)" -ForegroundColor Gray
Write-Host "   │   └── processed/    (clean data)" -ForegroundColor Gray
Write-Host "   ├── notebooks/        (analysis notebooks)" -ForegroundColor Cyan
Write-Host "   ├── src/              (source code)" -ForegroundColor Cyan
Write-Host "   ├── models/           (trained models)" -ForegroundColor Cyan
Write-Host "   ├── results/" -ForegroundColor Cyan
Write-Host "   │   ├── figures/      (visualizations)" -ForegroundColor Gray
Write-Host "   │   └── metrics/      (performance data)" -ForegroundColor Gray
Write-Host "   └── docs/             (documentation)" -ForegroundColor Cyan

Write-Host "`n📄 README files created in each directory" -ForegroundColor Yellow
Write-Host "`n🎯 Next steps:" -ForegroundColor Yellow
Write-Host "   1. Review the main README.md" -ForegroundColor White
Write-Host "   2. Check that all files moved correctly" -ForegroundColor White
Write-Host "   3. Update notebook file paths if needed" -ForegroundColor White
Write-Host "   4. Add your source code to src/ directory" -ForegroundColor White

Write-Host "`n"
