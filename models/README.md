# Trained Models Directory

## 📁 Purpose

Store all trained and serialized machine learning models here for:
- Model versioning
- Model deployment
- Model comparison
- Reproducibility

## 💾 Supported Formats

- **`.pkl`**: Pickle files (scikit-learn models)
- **`.joblib`**: Joblib files (efficient for large numpy arrays)
- **`.h5`** / **`.keras`**: TensorFlow/Keras models
- **`.pt`**: PyTorch models
- **`.json`**: Model configurations

## 📝 Naming Convention

Use descriptive names with metadata:

```
{model_type}_{date}_{metric}_{score}.{extension}
```

**Examples:**
- `random_forest_20251130_f1_0.8523.pkl`
- `neural_network_20251130_acc_0.8912.h5`
- `gradient_boosting_tuned_20251201_auc_0.9145.joblib`

## 🏗️ Suggested Structure

```
models/
├── baseline/              # Initial baseline models
├── tuned/                 # Hyperparameter-tuned models
├── production/            # Production-ready models
└── experimental/          # Experimental models
```

## 💡 Usage Example

### Saving a Model

```python
import joblib
from datetime import datetime

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
score = f1_score(y_test, model.predict(X_test))

# Save with metadata
date_str = datetime.now().strftime('%Y%m%d')
filename = f'random_forest_{date_str}_f1_{score:.4f}.joblib'
joblib.dump(model, f'../models/{filename}')

# Also save metadata
metadata = {
    'model_type': 'RandomForestClassifier',
    'training_date': date_str,
    'f1_score': score,
    'parameters': model.get_params()
}
import json
with open(f'../models/{filename}.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

### Loading a Model

```python
import joblib

# Load model
model = joblib.load('../models/random_forest_20251130_f1_0.8523.joblib')

# Make predictions
predictions = model.predict(X_new)
```

## 📊 Model Registry

Keep a `model_registry.csv` to track all models:

| Model Name | Type | Date | F1 Score | Accuracy | Notes |
|------------|------|------|----------|----------|-------|
| rf_v1.pkl | Random Forest | 2025-11-30 | 0.8523 | 0.8912 | Baseline |
| rf_tuned_v2.pkl | Random Forest | 2025-12-01 | 0.8745 | 0.9023 | Grid search tuned |

## 🔒 Best Practices

1. **Version control**: Never commit large model files to Git (add to .gitignore)
2. **Documentation**: Always save model metadata alongside the model
3. **Testing**: Test loaded models before deployment
4. **Backup**: Keep backups of production models
5. **Cleanup**: Periodically remove outdated experimental models

## ⚠️ Important Notes

- Models can be large files (100MB+), keep only necessary ones
- Add `*.pkl`, `*.h5`, `*.joblib` to `.gitignore`
- Use cloud storage (AWS S3, Azure Blob) for production models
- Document any preprocessing required before model inference

---

**Last Updated**: November 2025
