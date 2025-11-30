# Project Setup & Getting Started Guide

## 📋 Overview

This guide will help you get started with the Railway Delay Analysis project. The project structure follows data science best practices for reproducibility and maintainability.

## 🗂️ Folder Structure

```
railway-delay/
│
├── data/                          # All data files
│   ├── raw/                       # Original, immutable data
│   │   └── railway-delay-dataset.csv
│   ├── interim/                   # Intermediate transformations
│   │   └── dirty_train_data.csv
│   └── processed/                 # Final clean datasets
│       ├── train_data.csv
│       ├── test_data.csv
│       └── merged_train_data.csv
│
├── notebooks/                     # Jupyter notebooks
│   └── railway_delay_analysis.ipynb
│
├── src/                          # Reusable source code
│   ├── data/                     # Data processing scripts
│   ├── features/                 # Feature engineering
│   ├── models/                   # Model training scripts
│   └── visualization/            # Plotting utilities
│
├── models/                       # Trained model files
│
├── results/                      # Analysis outputs
│   ├── figures/                  # Visualizations
│   └── metrics/                  # Performance metrics
│
├── docs/                         # Documentation
│
├── .gitignore                    # Git ignore file
├── README.md                     # Project overview
├── requirements.txt              # Python dependencies
└── organize_files.ps1           # Organization script
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda install --file requirements.txt
```

### 2. Open the Notebook

```bash
# Navigate to notebooks directory
cd notebooks

# Launch Jupyter
jupyter notebook railway_delay_analysis.ipynb
```

### 3. Run the Analysis

1. Open `railway_delay_analysis.ipynb`
2. Run cells sequentially (Cell → Run All)
3. View results in `results/` directory

## 📊 Data Files

### Raw Data (Never Modify)
- **Location**: `data/raw/`
- **File**: `railway-delay-dataset.csv` (5.8M records)

### Processed Data (Ready for Analysis)
- **Location**: `data/processed/`
- **Files**:
  - `train_data.csv` - Training set (80%)
  - `test_data.csv` - Test set (20%)
  - `merged_train_data.csv` - Combined dataset

### Interim Data (Experimental)
- **Location**: `data/interim/`
- **File**: `dirty_train_data.csv` - Data with quality issues

## 🔧 Configuration

### File Paths in Notebook

The notebook uses relative paths from the `notebooks/` directory:

```python
train_file_path = os.path.join('..', 'data', 'processed', 'merged_train_data.csv')
test_file_path = os.path.join('..', 'data', 'processed', 'test_data.csv')
```

### Adjusting Sample Size

To work with different data sizes, modify in the notebook:

```python
# Load full dataset
df = pd.read_csv(train_file_path, low_memory=False)

# Or load sample
df = pd.read_csv(train_file_path, low_memory=False, nrows=100000)
```

## 📈 Running the Analysis

### Complete Workflow

1. **Data Loading** (Cell 8)
2. **Exploratory Data Analysis** (Cells 10-26)
3. **Data Preprocessing** (Cells 28-38)
4. **Feature Engineering** (Cells 32-37)
5. **Model Training** (Cells 45-54)
6. **Hyperparameter Tuning** (Cells 102-103)
7. **Cross-Validation** (Cells 104-105)
8. **Deep Learning** (Cells 97-100)
9. **Clustering** (Cells 57-64)
10. **Results & Insights** (Cells 73-75, 108)

### Individual Components

Run specific sections independently:

- **EDA Only**: Run cells 1-26
- **Models Only**: Run cells 1-8, then 40-54
- **Deep Learning Only**: Run cells 1-8, 40-42, then 97-100
- **Clustering Only**: Run cells 1-8, 28-30, then 57-64

## 💾 Saving Results

### Automatic Saving

Results are automatically saved to `results/` directory:

- **Figures**: PNG files in `results/figures/`
- **Metrics**: CSV/JSON files in `results/metrics/`

### Manual Saving

```python
# Save figure
plt.savefig('../results/figures/my_plot.png', dpi=150, bbox_inches='tight')

# Save metrics
results_df.to_csv('../results/metrics/model_performance.csv', index=False)

# Save model
import joblib
joblib.dump(model, '../models/my_model.pkl')
```

## 🧪 Testing & Validation

### Quick Test Run

1. Load 10,000 records only
2. Run through complete pipeline
3. Verify outputs in `results/`

### Full Analysis

1. Load 500,000+ records
2. Complete pipeline with all models
3. Expected runtime: 30-60 minutes (CPU)

## ⚙️ Advanced Configuration

### GPU Acceleration (Optional)

```python
# Check GPU availability
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
```

### Parallel Processing

```python
# Use all CPU cores
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_jobs=-1)  # Use all cores
```

## 📝 Best Practices

1. **Never modify raw data** - Always work with copies
2. **Use version control** - Commit code, not large data files
3. **Document changes** - Update README when adding features
4. **Save important models** - Store trained models in `models/`
5. **Clean results periodically** - Remove old/duplicate outputs

## 🐛 Troubleshooting

### "File not found" Error

```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solution**: Check file paths are correct relative to notebook location

### Memory Error

```
MemoryError: Unable to allocate array
```

**Solution**: Reduce sample size in data loading cell

### Package Not Found

```
ModuleNotFoundError: No module named 'package'
```

**Solution**: Install missing package
```bash
pip install package-name
```

## 📚 Additional Resources

- **Main README**: `README.md` - Project overview
- **Data README**: `data/README.md` - Data descriptions
- **Source README**: `src/README.md` - Code organization
- **Models README**: `models/README.md` - Model management
- **Results README**: `results/README.md` - Output structure

## 🤝 Contributing

For questions or contributions:
1. Check existing documentation
2. Review code comments in notebook
3. Contact project maintainer

## 📄 License

Educational Project - Academic Use Only

---

**Last Updated**: November 2025

**Next Steps**:
1. ✅ Install dependencies
2. ✅ Review data in `data/` folder
3. ✅ Open and run notebook
4. ✅ Check results in `results/` folder
5. ✅ Read analysis conclusions
