# Source Code Directory

## 📁 Structure

Organize your reusable Python code into these subdirectories:

```
src/
├── data/              # Data processing and loading
├── features/          # Feature engineering
├── models/            # Model training and evaluation
└── visualization/     # Plotting and visualization utilities
```

## 📝 Purpose

Keep your notebooks clean and maintainable by extracting reusable code into Python modules. This promotes:

- **Code reusability**: Use the same functions across multiple notebooks
- **Better testing**: Unit test your functions separately
- **Cleaner notebooks**: Focus on analysis and insights, not implementation details
- **Team collaboration**: Share code more easily

## 🔧 Suggested Modules

### `data/`
- `load_data.py`: Data loading utilities
- `preprocess.py`: Data cleaning and preprocessing functions
- `split_data.py`: Train/test/validation splitting logic

### `features/`
- `engineer.py`: Feature engineering functions
- `encoding.py`: Categorical encoding utilities
- `scaling.py`: Feature scaling and normalization

### `models/`
- `train.py`: Model training pipelines
- `evaluate.py`: Model evaluation functions
- `tune.py`: Hyperparameter tuning utilities
- `save_load.py`: Model persistence

### `visualization/`
- `plots.py`: Common plotting functions
- `metrics_viz.py`: Metrics visualization
- `exploratory.py`: EDA plotting utilities

## 💡 Usage Example

```python
# In your notebook
import sys
sys.path.append('../src')

from data.load_data import load_railway_data
from features.engineer import create_temporal_features
from models.train import train_classifier
from visualization.plots import plot_confusion_matrix

# Use the functions
df = load_railway_data('data/processed/train_data.csv')
df = create_temporal_features(df)
model = train_classifier(df, model_type='random_forest')
plot_confusion_matrix(y_true, y_pred)
```

## 📋 Best Practices

1. **Use docstrings**: Document all functions with clear descriptions, parameters, and return values
2. **Type hints**: Use Python type hints for better code clarity
3. **Error handling**: Include proper exception handling
4. **Logging**: Use logging instead of print statements
5. **Configuration**: Use config files for parameters and paths
6. **Testing**: Write unit tests for your functions

## 🔗 Example Module Structure

```python
# src/data/load_data.py

import pandas as pd
from typing import Optional

def load_railway_data(
    filepath: str,
    nrows: Optional[int] = None,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Load railway delay dataset from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    nrows : int, optional
        Number of rows to read
    sample_size : int, optional
        Random sample size
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    df = pd.read_csv(filepath, nrows=nrows, low_memory=False)
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    return df
```

## 📦 Dependencies

Create a `requirements.txt` in the project root with all dependencies:

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
tensorflow>=2.12.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

Install with: `pip install -r requirements.txt`

## 🚀 Getting Started

1. Create subdirectories as needed
2. Write your first utility function
3. Test it independently
4. Import and use in notebooks
5. Gradually refactor notebook code into modules

---

**Last Updated**: November 2025
