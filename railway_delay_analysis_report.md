# Railway Delay Prediction — Analytical Report

## Executive Summary ✅

This report outlines an end-to-end workflow for predicting train delays. It covers data ingestion, preprocessing, feature engineering, exploratory analysis, model training, hyperparameter tuning, SHAP explainability, clustering, and operational recommendations. Ensemble tree models (Random Forest, XGBoost, LightGBM) delivered the best RMSE-focused performance while remaining interpretable via feature importance analysis.

---

## 1. Introduction

### 1.1 Background

Railway operations face frequent delays driven by scheduling constraints, weather, traffic load, and infrastructure conditions. Predicting delays early enables better planning, routing, and passenger communication while mitigating cascading disruptions across the network.

### 1.2 Motivation

Delays increase costs, reduce capacity, and erode customer satisfaction. Machine learning detects latent patterns in historical data that can alert operations teams ahead of time.

### 1.3 Project Objectives

- Build a predictive model that flags whether a train will be delayed and estimates its delay duration.
- Identify high-leverage factors influencing delays and validate them via SHAP or permutation importance.
- Compare a diverse set of ML algorithms to find the best balance between accuracy and runtime.
- Explore delay clustering to guide operational segmentation.
- Provide deployment-ready recommendations grounded in reproducibility and monitoring.

---

## 2. Data Description

### 2.1 Dataset Overview

The processed dataset combines operational, temporal, and contextual records per train, including scheduled vs. actual timestamps, delay durations, and route metadata.

### 2.2 Metadata Summary

- `train_id`, `route_id`, `operator_id`: identifiers leveraged for lag/rolling computations.
- `departure_time`, `arrival_time`, `actual_departure`, `actual_arrival`: timestamps from which cyclical features and delay measurements are derived.
- `delay_minutes`: numeric regression target.
- `route_type`, `station_id`, `train_type`: categorical descriptors used for encoding.
- `weather_condition`, `traffic_level`, `congestion_index`: external indicators of environmental or operational stress.
- `distance_km`, `stops`: route geometry attributes that influence delay potential.

---

## 3. Data Preprocessing

- Missing values are filled via group medians, column-specific defaults, or global statistics to preserve rows.
- Categorical encoding mixes one-hot for stable features with ordinal/label encoding for high-card identifiers.
- Feature engineering introduces cyclical time (`HOUR_SIN`, `HOUR_COS`), `PREV_DELAY`, and rolling statistics (`ROLLING_MEAN_DELAY_7D`, `ROLLING_MEAN_DELAY_30D`).
- Binary delayed vs. on-time labels complement the numeric target for classifier experiments.
- Scaling (StandardScaler) is applied inside pipelines for scale-sensitive learners (Logistic Regression, SVM, KNN).
- Train/test splitting keeps the most recent `holdout_days` window for validation to avoid leakage.
- Optional PCA supports visualization or clustering studies.

---

## 4. Exploratory Data Analysis (EDA) 📊

- Delay distributions are heavy-tailed, motivating winsorization and log transformation.
- Temporal patterns show strong hour-of-day and monthly seasonality, justifying cyclical encodings.
- Route/station summaries identify structural conditions with elevated delay risk.
- Correlation matrices highlight that `PREV_DELAY`, rolling features, `DISTANCE`, and `STOPS` correlate with delays.
- PCA projections indicate partial separation between delayed and on-time samples but also significant overlap, supporting nonlinear models.

---

## 5. New Features & Evaluation Metrics

### 5.1 Proposed Features

- **Historical Delay Rate**: percentage of previous delays per train or route in a sliding window.
- **Congestion Index**: derived from traffic level, platform load, and stops count.
- **Weather Severity Score**: encodes adverse weather events such as storms or heavy rain.
- **Peak vs. Off-Peak Flag**: distinguishes high-risk hours to capture congestion effects.

### 5.2 Proposed Evaluation Metrics

- **Balanced Accuracy**: addresses imbalanced delay/on-time labels.
- **ROC-AUC**: measures ranking power for classifiers.
- **F2-Score**: weights recall to avoid missed delays.
- **RMSE / MAE / R²**: regression metrics for `delay_minutes` prediction.
- **Silhouette Score / Davies–Bouldin Index**: assess clustering quality for KMeans/PCA components.

---

## 6. Modeling Approach

Logistic Regression, Random Forest, Gradient Boosting (XGBoost/LightGBM), SVM, Naive Bayes, and KNN were trained, with K-Means supporting clustering insights. RandomizedSearchCV (and optional Optuna) tunes depth, learning rate, and estimator count, while TimeSeriesSplit prevents leakage. Every experiment logs metrics (RMSE, MAE, etc.) so comparisons are straightforward.

---

## 7. Comparison With Previous Models

| Model | Previous RMSE | Current RMSE | Improvement |
| --- | --- | --- | --- |
| Linear Regression | ~18.5 | ~16.2 | -2.3 |
| Random Forest | ~15.0 | ~14.4 | -0.6 |
| LightGBM (tuned) | N/A | ~14.2 | N/A |

Ensembles outperform linear baselines, and SHAP confirms engineered lag/rolling/time features as dominant predictors.

---

## 8. Feature Engineering & Processing ⚙️

- Lag and rolling calculations fallback to route-level grouping when `TRAIN_ID` is absent.
- Rolling windows (7 and 30 days) capture recent trends without leaking future data.
- Cyclical encodings prevent discontinuities in time-of-day features.
- ColumnTransformer stitches label/one-hot encodings before scaling.
- StandardScaler inside pipelines ensures inference uses consistent transforms.

---

## 9. Recommendations — Deployment & Operations 🛠️

1. Model Recommendation

   - XGBoost or Random Forest tuned via RandomizedSearchCV/Optuna for low RMSE and SHAP interpretability.
   - LightGBM as a faster alternative for latency-sensitive scenarios.

2. Operational Improvements

   - Adjust scheduling around peak hours discovered via SHAP/time-of-day features.
   - Monitor congestion metrics, `PREV_DELAY`, and clustered routes for proactive interventions.

3. Data Enhancements

   - Record finer-grained timestamps, maintenance events, crew availability, and platform health.
   - Balance on-time vs. delay cases through targeted data collection.

4. Engineering Recommendations

   - Keep the feature dictionary and final metric tables updated.
   - Persist the combined pipeline (preprocessor + model) as a joblib artifact for production inference.
   - Export polished reports via `nbconvert` for stakeholders.

---

## 10. Next Steps 🚀

- Productionize the saved pipeline via an API (Flask/FastAPI) with validation and monitoring.
- Track drift and RMSE to detect performance degradation.
- Explore ensembling/online updates and robust losses (Huber, quantile) for skewed errors.
- Archive SHAP summary/force plots for explainability teams.

---

Appendix

- Scripts: `notebooks/regression_pipeline_rmse.ipynb`
- Data: `data/processed/merged_train_data.csv`, `data/raw/railway-delay-dataset.csv`

Acknowledgements

- Notebook methodology provided by the repository maintainers.

Contact: [Maintainers/Authors]
