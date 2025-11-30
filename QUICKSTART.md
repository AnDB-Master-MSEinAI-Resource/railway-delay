# 🎯 Quick Start Guide - Enhanced Railway Delay Analysis

## What's New? ✨

Your railway delay analysis notebook now includes:

### 1. 📦 **Model Versioning**
- Every trained model is automatically saved with:
  - Version number (timestamp-based)
  - Complete metadata (metrics, parameters, notes)
  - Central version log for tracking

### 2. 🔍 **SHAP Analysis**
- Understand **why** models make predictions
- See which features are most important
- Visualize feature interactions and dependencies
- 3 types of plots: summary bar, beeswarm, dependence

### 3. 📊 **Comprehensive Reports**
- Beautiful HTML report (open in browser)
- CSV metrics table
- JSON summaries for automation
- Best model recommendations

### 4. 📈 **Advanced Visualizations**
- 4-panel comparison chart
- Learning curves
- Confusion matrix grid
- Radar chart
- ROC curves

---

## 🚀 How to Run

### Option 1: Full Analysis
```bash
cd notebooks
jupyter notebook railway_delay_analysis.ipynb
# Run All Cells (Cell → Run All)
```

### Option 2: Step by Step
Run cells in order:
1. **Cells 1-10**: Setup and data loading ✅
2. **Cells 11-30**: EDA and preprocessing ✅
3. **Cells 31-47**: Model training ✅
4. **New Cells**: Model versioning 🆕
5. **New Cells**: SHAP analysis 🆕
6. **New Cells**: Reports & visualizations 🆕

---

## 📂 What Gets Generated

After running the notebook, you'll have:

### In `models/` folder:
- `{ModelName}_v{timestamp}.pkl` - Saved models
- `{ModelName}_v{timestamp}_metadata.json` - Model info
- `version_log.json` - All versions tracked

### In `results/` folder:
- `model_report.html` - **Open this first!** 🌟
- `analysis_summary.json` - Key findings
- `metrics/model_performance.csv` - All metrics
- `figures/*.png` - All visualizations

---

## 🎨 Visualizations You'll Get

1. **comprehensive_model_comparison.png** - 4-panel overview
2. **learning_curves.png** - Training progress
3. **all_confusion_matrices.png** - Prediction accuracy
4. **radar_chart_comparison.png** - Multi-metric view
5. **shap_summary_bar.png** - Feature importance
6. **shap_beeswarm.png** - Detailed feature impact
7. **shap_dependence.png** - Feature relationships

---

## 💡 Key Features

### Data Sources (Updated!)
- **Training**: `data/processed/merged_train_data.csv` (clean + dirty data)
- **Testing**: `data/raw/railway-delay-dataset.csv` (original dataset)

### Models Trained
1. Logistic Regression
2. Decision Tree
3. Random Forest ⭐
4. Gradient Boosting ⭐
5. K-Nearest Neighbors
6. Naive Bayes

### Metrics Evaluated
- Accuracy
- Precision, Recall, F1-Score
- Balanced Accuracy
- Cohen's Kappa
- Matthews Correlation Coefficient
- G-Mean
- ROC-AUC

---

## 🏆 Expected Results

After running, you should see:
- ✅ All 6 models trained successfully
- ✅ 6 models saved with version control
- ✅ SHAP analysis for top 3 models
- ✅ 10+ visualizations generated
- ✅ HTML report created
- ✅ Metrics CSV saved

**Best Model**: Typically Random Forest or Gradient Boosting
**Expected F1-Score**: 0.85 - 0.95 (depending on data quality)

---

## 🔍 How to Use SHAP Insights

### Reading SHAP Plots

**Summary Bar Chart**:
- Shows average feature importance
- Longer bar = more important feature

**Beeswarm Plot**:
- Red dots = high feature values
- Blue dots = low feature values
- Position (left/right) = impact on prediction

**Dependence Plots**:
- X-axis = feature value
- Y-axis = SHAP value (impact)
- Shows non-linear relationships

### Example Interpretation
```
If SHAP shows:
- "scheduled_time" has high importance
  → Departure time strongly affects delays

- "weather_condition=Rain" pushes predictions right
  → Rain increases delay probability

- "operator=CompanyA" pushes predictions left
  → CompanyA has fewer delays
```

---

## 📊 Viewing the HTML Report

1. Navigate to: `results/model_report.html`
2. Double-click to open in browser
3. You'll see:
   - Dataset overview
   - Best model highlighted
   - Complete metrics table
   - Key insights
   - File locations

---

## ⚙️ Customization

### Change Sample Size
In cell 8 (data loading):
```python
df = pd.read_csv(train_file_path, nrows=500000)  # Adjust this number
```

### Analyze More/Fewer Models with SHAP
In SHAP analysis cell:
```python
top_models = sorted(...)[:3]  # Change 3 to desired number
```

### Faster Learning Curves
```python
train_sizes = np.linspace(0.3, 1.0, 5)  # Fewer points
cv=2  # Less cross-validation
```

---

## 🐛 Common Issues

### Issue: "Package not found"
```bash
pip install shap joblib
```

### Issue: SHAP takes too long
Reduce sample size in SHAP cell:
```python
shap_sample_size = min(500, len(X_train_fast))
```

### Issue: Out of memory
Reduce data loading:
```python
df = pd.read_csv(train_file_path, nrows=100000)  # Smaller sample
```

---

## 📈 Performance Tips

### Fast Analysis (~10 minutes)
- Load 100K training samples
- Load 20K test samples
- SHAP sample: 500
- Top 2 models for SHAP

### Full Analysis (~30 minutes)
- Load 500K training samples
- Load 100K test samples
- SHAP sample: 1000
- Top 3 models for SHAP

### Maximum Analysis (~60 minutes)
- Load all data (no nrows limit)
- SHAP sample: 5000
- All models for SHAP
- More cross-validation folds

---

## 🎯 Next Steps

1. **Run the notebook** - See all features in action
2. **Open HTML report** - Beautiful summary
3. **Examine SHAP plots** - Understand predictions
4. **Check model versions** - Review saved models
5. **Analyze metrics CSV** - Deep dive into numbers

---

## 📞 Need Help?

Check these files:
- `README.md` - Project overview
- `docs/GETTING_STARTED.md` - Setup guide
- `docs/FEATURES_GUIDE.md` - Detailed feature docs
- Notebook comments - Inline explanations

---

## ✅ Checklist

Before running:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] In `notebooks/` directory
- [ ] Jupyter running

After running:
- [ ] All cells executed successfully
- [ ] Models saved in `models/`
- [ ] Figures saved in `results/figures/`
- [ ] HTML report generated
- [ ] Best model identified

---

**Happy Analyzing! 🚂📊**

Generated on: December 1, 2025
