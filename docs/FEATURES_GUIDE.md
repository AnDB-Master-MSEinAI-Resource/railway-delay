# Railway Delay Analysis - Features Guide

## 🎯 Overview

This guide explains all the advanced features added to the Railway Delay Analysis project, including model versioning, SHAP analysis, comprehensive reporting, and advanced visualizations.

---

## 📦 Model Versioning System

### Features
- **Automatic Version Control**: Each model is saved with a timestamp-based version (e.g., `v20251201_143025`)
- **Metadata Tracking**: Stores model type, metrics, parameters, file size, and notes
- **Version Log**: Central JSON file tracking all model versions
- **Easy Loading**: Load any previous model version by name and version

### Usage

```python
# Initialize version manager
version_manager = ModelVersionManager()

# Save a model with version control
version, path = version_manager.save_model(
    model=trained_model,
    model_name="RandomForest",
    metrics={'accuracy': 0.95, 'f1_score': 0.93},
    params=model.get_params(),
    notes="Production model"
)

# List all versions of a specific model
versions = version_manager.list_versions(model_name="RandomForest")

# Load a specific version
model = version_manager.load_model(version="v20251201_143025", model_name="RandomForest")
```

### Files Generated
- `models/{model_name}_{version}.pkl` - Serialized model
- `models/{model_name}_{version}_metadata.json` - Model metadata
- `models/version_log.json` - Complete version history

---

## 🔍 SHAP Analysis

### What is SHAP?
SHAP (SHapley Additive exPlanations) provides model interpretability by showing:
- Which features are most important
- How feature values impact predictions
- Feature interactions and dependencies

### Analyses Performed

#### 1. **Summary Bar Plot**
- Shows feature importance across all predictions
- Saved as: `results/figures/shap_summary_bar.png`

#### 2. **Beeswarm Plot**
- Detailed view of feature impacts with value distribution
- Color indicates feature value (high/low)
- Position shows SHAP value (impact on prediction)
- Saved as: `results/figures/shap_beeswarm.png`

#### 3. **Dependence Plots**
- Shows relationship between feature values and SHAP values
- Reveals non-linear relationships and interactions
- Generated for top 4 most important features
- Saved as: `results/figures/shap_dependence.png`

### Example Interpretation

```
If a SHAP plot shows:
- "delay_minutes" with high positive SHAP value → Strongly predicts delay
- "weather_condition=Rain" with negative SHAP value → Reduces delay probability
```

---

## 📊 Comprehensive Reporting

### 1. CSV Metrics Report
**File**: `results/metrics/model_performance.csv`

Contains detailed metrics for all models:
- Accuracy, Precision, Recall, F1-Score
- Balanced Accuracy
- Cohen's Kappa, Matthews Correlation Coefficient
- G-Mean, ROC-AUC

### 2. Best Model Info
**File**: `results/metrics/best_model_info.json`

JSON file with:
- Best model name and timestamp
- Complete metrics dictionary
- Training/test sample counts
- Feature list

### 3. HTML Report
**File**: `results/model_report.html`

Beautiful web-based report including:
- Dataset overview with key statistics
- Best model highlight with top metrics
- Complete performance comparison table
- Key insights and recommendations
- Links to all generated files

**To view**: Open `results/model_report.html` in any web browser

### 4. Analysis Summary
**File**: `results/analysis_summary.json`

JSON summary with:
- Analysis date and data sources
- Sample counts and feature count
- Best model performance
- Top 3 models ranking
- Average performance metrics

---

## 📈 Advanced Visualizations

### 1. Comprehensive Model Comparison
**File**: `results/figures/comprehensive_model_comparison.png`

4-panel chart showing:
- **Panel 1**: Overall metrics comparison (bar chart)
- **Panel 2**: Advanced metrics trends (line chart)
- **Panel 3**: F1-Score ranking (horizontal bar)
- **Panel 4**: Balanced Accuracy vs ROC-AUC scatter plot

### 2. Learning Curves
**File**: `results/figures/learning_curves.png`

Shows training and validation scores across different dataset sizes for top 2 models.

**Interpretation**:
- Converging curves → Model is learning well
- Gap between curves → Potential overfitting
- Both curves low → Underfitting

### 3. Confusion Matrix Grid
**File**: `results/figures/all_confusion_matrices.png`

Heatmaps for all models showing:
- True Positives, False Positives
- True Negatives, False Negatives
- F1-Score in title

### 4. Radar Chart
**File**: `results/figures/radar_chart_comparison.png`

Multi-metric comparison showing:
- All models on one chart
- 6 key metrics (accuracy, precision, recall, F1, balanced accuracy, Cohen's kappa)
- Easy visual comparison of model strengths/weaknesses

### 5. ROC Curves & Confusion Matrices
**File**: `results/figures/roc_confusion_matrices.png`

Combined visualization showing:
- ROC curves with AUC scores
- Confusion matrices for each model
- Classification reports

---

## 🚀 Running the Complete Analysis

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Navigate to Notebooks
```bash
cd notebooks
```

### Step 3: Open Jupyter
```bash
jupyter notebook railway_delay_analysis.ipynb
```

### Step 4: Run All Cells
- **Option A**: Menu → Cell → Run All
- **Option B**: Run cells sequentially for step-by-step progress

### Expected Runtime
- Full analysis: 20-40 minutes (depending on hardware)
- Quick test (with sampling): 5-10 minutes

---

## 📁 Output Directory Structure

```
results/
├── model_report.html              # Main HTML report
├── analysis_summary.json          # JSON summary
├── metrics/
│   ├── model_performance.csv      # All model metrics
│   └── best_model_info.json       # Best model details
└── figures/
    ├── comprehensive_model_comparison.png
    ├── learning_curves.png
    ├── all_confusion_matrices.png
    ├── radar_chart_comparison.png
    ├── shap_summary_bar.png
    ├── shap_beeswarm.png
    └── shap_dependence.png

models/
├── version_log.json               # All model versions
├── {model}_{version}.pkl          # Saved models
└── {model}_{version}_metadata.json # Model metadata
```

---

## 🔧 Configuration Options

### Data Sampling
Adjust sample size in the data loading cell:
```python
df = pd.read_csv(train_file_path, low_memory=False, nrows=500000)  # Change nrows
```

### SHAP Sample Size
Modify in SHAP analysis cell:
```python
shap_sample_size = min(1000, len(X_train_fast))  # Change 1000 to desired size
```

### Number of Models for SHAP
Change in SHAP initialization:
```python
top_models = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:3]  # Change 3
```

### Learning Curve Parameters
Modify in learning curve cell:
```python
train_sizes = np.linspace(0.1, 1.0, 10)  # 10 points from 10% to 100%
cv=3  # 3-fold cross-validation
```

---

## 💡 Best Practices

### 1. Model Versioning
- Always add descriptive notes when saving models
- Include key experiment parameters in metadata
- Regularly clean old/unused model versions

### 2. SHAP Analysis
- Use smaller samples (1000-5000) for faster computation
- TreeExplainer is faster than KernelExplainer
- Focus on top 3-5 models for detailed analysis

### 3. Reporting
- Generate reports after each major experiment
- Compare HTML reports across different runs
- Keep CSV metrics for programmatic analysis

### 4. Visualizations
- Save high-DPI figures (150+ dpi) for publications
- Use consistent color schemes across charts
- Include model metrics in chart titles

---

## 🐛 Troubleshooting

### SHAP Memory Error
```python
# Reduce sample size
shap_sample_size = min(500, len(X_train_fast))  # Smaller sample
```

### Slow Learning Curves
```python
# Reduce CV folds and training sizes
train_sizes = np.linspace(0.3, 1.0, 5)  # Fewer points
cv=2  # Less cross-validation
```

### Model Loading Error
```python
# Check version log
versions = version_manager.list_versions()
print(json.dumps(versions, indent=2))
```

### HTML Report Not Displaying
- Ensure UTF-8 encoding is supported
- Open in modern browser (Chrome, Firefox, Edge)
- Check file permissions

---

## 📚 Additional Resources

### SHAP Documentation
- Official Docs: https://shap.readthedocs.io/
- GitHub: https://github.com/slundberg/shap
- Paper: https://arxiv.org/abs/1705.07874

### Scikit-learn Metrics
- User Guide: https://scikit-learn.org/stable/modules/model_evaluation.html
- API Reference: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

### Model Persistence
- Joblib: https://joblib.readthedocs.io/
- Pickle: https://docs.python.org/3/library/pickle.html

---

## 🎓 Key Takeaways

1. **Model Versioning**: Ensures reproducibility and model tracking
2. **SHAP Analysis**: Provides explainability and trust in predictions
3. **Comprehensive Reporting**: Facilitates communication with stakeholders
4. **Advanced Visualizations**: Reveals insights not visible in raw metrics
5. **Automated Pipeline**: Saves time and reduces manual errors

---

**Last Updated**: December 2025

**Questions or Issues?**
- Check the main README.md
- Review code comments in the notebook
- Consult GETTING_STARTED.md for setup help
