# Results Directory

## 📁 Structure

- **figures/**: All generated visualizations and plots
- **metrics/**: Model performance metrics and comparison data

## 📊 Figures (`figures/`)

All visualization outputs from the analysis, including:

### Model Performance Visualizations
- `model_comparison.png`: Bar charts comparing all models
- `confusion_matrices.png`: Confusion matrices for top models
- `roc_curves.png`: ROC curves with AUC scores
- `feature_importance.png`: Feature importance plots

### Training Visualizations
- `training_history.png`: Loss and accuracy curves
- `learning_curves.png`: Training vs validation performance
- `deep_learning_evaluation.png`: Neural network performance

### Hyperparameter Tuning
- `hyperparameter_tuning_results.png`: Before/after tuning comparison
- `grid_search_heatmap.png`: Parameter grid results

### Cross-Validation
- `cross_validation_results.png`: CV scores with error bars
- `stability_analysis.png`: Model stability metrics

### Clustering Analysis
- `clustering_optimization.png`: Elbow method and silhouette scores
- `clustering_pca.png`: PCA visualization of clusters
- `cluster_profiles.png`: Cluster characteristics

### Exploratory Data Analysis
- `data_distribution.png`: Feature distributions
- `correlation_matrix.png`: Feature correlation heatmap
- `missing_values.png`: Missing value analysis
- `delay_patterns.png`: Delay pattern analysis

## 📈 Metrics (`metrics/`)

Structured data files containing model performance:

### Suggested Files

**`model_performance.csv`**
```csv
Model,Accuracy,Precision,Recall,F1-Score,ROC-AUC,Training_Time
Random Forest,0.8912,0.8745,0.8623,0.8684,0.9234,12.5
Gradient Boosting,0.8856,0.8698,0.8578,0.8637,0.9189,18.3
Neural Network,0.8923,0.8768,0.8645,0.8706,0.9256,45.2
```

**`cross_validation_results.json`**
```json
{
  "random_forest": {
    "accuracy_mean": 0.8912,
    "accuracy_std": 0.0145,
    "f1_mean": 0.8684,
    "f1_std": 0.0168
  }
}
```

**`hyperparameter_tuning.csv`**
- Best parameters for each model
- Grid search results
- Tuning time and improvements

**`feature_importance.csv`**
- Feature names and importance scores
- Ranked by importance

## 💾 File Formats

- **PNG**: High-quality plots (300 DPI for reports)
- **SVG**: Vector graphics (for presentations)
- **CSV**: Tabular metrics data
- **JSON**: Structured configuration and results

## 📝 Naming Conventions

### Figures
- Use descriptive, lowercase names with underscores
- Include version or date if multiple iterations
- Examples: `model_comparison_v2.png`, `clustering_20251130.png`

### Metrics
- Use clear, structured names
- Include aggregation level (e.g., `model_summary.csv`, `detailed_metrics.json`)

## 🔧 Automation

Create a utility script to automatically save results:

```python
# src/utils/save_results.py

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def save_figure(fig, name, dpi=150):
    """Save figure with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f'../results/figures/{name}_{timestamp}.png'
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {filename}")

def save_metrics(metrics_dict, name):
    """Save metrics as CSV."""
    df = pd.DataFrame(metrics_dict).T
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f'../results/metrics/{name}_{timestamp}.csv'
    df.to_csv(filename)
    print(f"Saved: {filename}")
```

## 📊 Analysis Reports

Consider creating summary reports combining figures and metrics:

1. **Executive Summary**: Top-level findings with key visualizations
2. **Technical Report**: Detailed analysis with all figures
3. **Model Card**: Production model documentation

## 🔗 Integration with Notebooks

In your notebooks, save results automatically:

```python
# At the end of analysis sections
plt.savefig('../results/figures/model_comparison.png', dpi=150, bbox_inches='tight')
results_df.to_csv('../results/metrics/model_performance.csv', index=False)
```

## ⚠️ Important Notes

- Keep results organized by date or version
- Document any manual analysis in a separate notes file
- Clean up old/duplicate results periodically
- Backup important results before major changes

## 📋 Checklist

After completing analysis, ensure you have:

- [ ] Model comparison charts
- [ ] Confusion matrices for best models
- [ ] ROC curves
- [ ] Feature importance plots
- [ ] Model performance metrics CSV
- [ ] Cross-validation results
- [ ] Training history (for deep learning)
- [ ] Clustering visualizations
- [ ] Summary report/presentation

---

**Last Updated**: November 2025
