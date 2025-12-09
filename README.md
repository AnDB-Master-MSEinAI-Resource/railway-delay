# Railway Delay Analysis - Data Mining Project

## 📋 Project Overview
Comprehensive data mining and machine learning analysis for predicting railway delays using advanced techniques including traditional ML, deep learning, hyperparameter tuning, and clustering.

## 🗂️ Project Structure
```
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
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow imbalanced-learn
```

### Running the Analysis
1. Open `notebooks/railway_delay_analysis.ipynb`
2. Update file paths in the notebook if needed
3. Run cells sequentially from top to bottom
4. Results will be saved in `results/` directory

## 📄 Analytical Report

- A consolidated analytical report summarizing data preparation, EDA, feature engineering, and model evaluation has been added to `reports/railway_delay_analysis_report.md`.


## 🧾 Daily Activity Reports

- Run the report generator to create a per-day summary of experiments and metrics saved under `models/metrics_log.csv`:

```powershell
python reports/daily_report_generator.py
```

- The generator creates `reports/daily_activity_report.csv` and `reports/daily_activity_report.md` files.

## 🧠 Memory-friendly Data Loading

- If your machine has limited memory and reading `merged_train_data.csv` causes MemoryError, set `DOWNSAMPLE=True` in the notebook or use the smart loader included in `notebooks/regression_pipeline_rmse.ipynb` (function `smart_read_csv`) which will downcast dtypes and fallback to chunked read when necessary.

## 📊 Models Implemented

### Traditional Machine Learning
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Naive Bayes

### Deep Learning
- Multi-layer Neural Network
- Dropout regularization
- Batch Normalization
- Early stopping

### Optimization Techniques
- Grid Search hyperparameter tuning
- Stratified K-Fold cross-validation
- Feature importance analysis

### Clustering
- K-Means clustering
- DBSCAN
- PCA dimensionality reduction

## 📈 Key Features

### Data Processing
- Missing value imputation
- Outlier detection and handling
- Feature engineering
- Data scaling and normalization
- Categorical encoding

### Evaluation Metrics
- **Standard**: Accuracy, Precision, Recall, F1-Score
- **Advanced**: Balanced Accuracy, Cohen's Kappa, MCC, G-Mean, ROC-AUC
- **Clustering**: Silhouette Score, Davies-Bouldin Score

### Visualizations
All visualizations are saved in `results/figures/`:
- Confusion matrices
- ROC curves
- Feature importance plots
- Training history curves
- Clustering visualizations
- Model comparison charts

## 📁 Data Files

### Raw Data (`data/raw/`)
- `railway-delay-dataset.csv`: Original dataset (5.8M records)

### Processed Data (`data/processed/`)
- `train_data.csv`: Training dataset (80% split)
- `test_data.csv`: Test dataset (20% split)
- `merged_train_data.csv`: Combined clean and dirty training data

### Interim Data (`data/interim/`)
- `dirty_train_data.csv`: Data with intentionally injected errors for data quality analysis

## 🎯 Results Summary

Results are stored in `results/`:
- **figures/**: All generated plots and visualizations
- **metrics/**: Model performance metrics (to be generated)

## 📚 Documentation

See `docs/` directory for:
- Data schema documentation
- Model architecture details
- Analysis methodology
- Project reports

## 🔬 Methodology

1. **Data Exploration**: Comprehensive EDA with statistical analysis
2. **Data Preprocessing**: Cleaning, transformation, feature engineering
3. **Model Training**: Multiple algorithms with proper validation
4. **Hyperparameter Tuning**: Systematic optimization
5. **Model Evaluation**: Comprehensive metrics and comparisons
6. **Clustering Analysis**: Pattern discovery
7. **Insights Generation**: Business-focused recommendations

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine learning algorithms
- **TensorFlow/Keras**: Deep learning
- **Matplotlib & Seaborn**: Visualization
- **Jupyter Notebook**: Interactive analysis

## 📝 Notes

- Large datasets are sampled for efficient training (adjustable in notebook)
- GPU acceleration supported for deep learning models
- All random seeds are set for reproducibility (random_state=42)

## 👥 Author
**Data Mining Project**  
MSE Program - Master of Software Engineering

## 📄 License
Educational Project - Academic Use

## 🤝 Contributing
This is an educational project. For suggestions or improvements, please contact the project author.

---

**Last Updated**: November 2025
