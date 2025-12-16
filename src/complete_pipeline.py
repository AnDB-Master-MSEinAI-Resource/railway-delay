"""
Complete Railway Delay Prediction Pipeline
==========================================
This script implements a comprehensive machine learning pipeline for railway delay prediction
including both regression and classification tasks.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Tuple, Dict, List, Any

# Scikit-learn imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, precision_score, recall_score, fbeta_score
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression

# Optional imports
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
warnings.filterwarnings('ignore')


class RailwayDelayPipeline:
    """Complete pipeline for railway delay prediction"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train_reg = None
        self.y_test_reg = None
        self.y_train_clf = None
        self.y_test_clf = None
        self.feature_names = None
        self.preprocessor = None
        self.models = {}
        self.results = {}
        
    # ========================================================================
    # 1. DATA LOADING
    # ========================================================================
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load railway delay dataset"""
        print(f"Loading data from: {filepath}")
        self.data = pd.read_csv(filepath)
        print(f"✓ Data loaded: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns")
        return self.data
    
    # ========================================================================
    # 2. DATA DESCRIPTION & QUALITY REPORT
    # ========================================================================
    
    def generate_data_dictionary(self) -> pd.DataFrame:
        """Generate comprehensive data dictionary"""
        if self.data is None:
            raise ValueError("Load data first using load_data()")
        
        data_dict = []
        
        for col in self.data.columns:
            col_info = {
                'Column Name': col,
                'Data Type': str(self.data[col].dtype),
                'Non-Null Count': self.data[col].notna().sum(),
                'Missing Count': self.data[col].isna().sum(),
                'Missing %': f"{self.data[col].isna().sum() / len(self.data) * 100:.2f}%",
                'Unique Values': self.data[col].nunique(),
                'Sample Value': str(self.data[col].dropna().iloc[0]) if self.data[col].notna().any() else 'N/A'
            }
            
            # Add statistical info for numeric columns
            if pd.api.types.is_numeric_dtype(self.data[col]):
                col_info.update({
                    'Mean': f"{self.data[col].mean():.2f}" if self.data[col].notna().any() else 'N/A',
                    'Std': f"{self.data[col].std():.2f}" if self.data[col].notna().any() else 'N/A',
                    'Min': f"{self.data[col].min():.2f}" if self.data[col].notna().any() else 'N/A',
                    'Max': f"{self.data[col].max():.2f}" if self.data[col].notna().any() else 'N/A',
                })
            
            data_dict.append(col_info)
        
        return pd.DataFrame(data_dict)
    
    def print_data_summary(self):
        """Print comprehensive data summary"""
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        
        print(f"\nShape: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Duplicate rows: {self.data.duplicated().sum():,}")
        
        print("\n" + "-"*80)
        print("Missing Values Summary:")
        print("-"*80)
        missing = self.data.isna().sum()
        missing_pct = (missing / len(self.data) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing': missing,
            'Percentage': missing_pct
        }).query('Missing > 0').sort_values('Missing', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df)
        else:
            print("No missing values found!")
        
        print("\n" + "-"*80)
        print("Data Types:")
        print("-"*80)
        print(self.data.dtypes.value_counts())
        
    # ========================================================================
    # 3. DATA PREPROCESSING
    # ========================================================================
    
    def preprocess_data(self, target_col='DELAY_MINUTES', delay_threshold=5):
        """
        Comprehensive data preprocessing
        
        Parameters:
        -----------
        target_col : str
            Name of the target column (delay in minutes)
        delay_threshold : int
            Threshold in minutes to define IS_DELAYED classification
        """
        if self.data is None:
            raise ValueError("Load data first using load_data()")
        
        df = self.data.copy()
        
        print("\n" + "="*80)
        print("PREPROCESSING PIPELINE")
        print("="*80)
        
        # 3.1 Handle datetime columns
        datetime_cols = df.select_dtypes(include=['object']).columns
        datetime_cols = [col for col in datetime_cols if 'date' in col.lower() or 'time' in col.lower()]
        
        if datetime_cols:
            print(f"\n✓ Converting {len(datetime_cols)} datetime columns...")
            for col in datetime_cols:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        # 3.2 Create target variables
        print(f"\n✓ Creating target variables...")
        print(f"   - Regression target: {target_col}")
        print(f"   - Classification target: IS_DELAYED (threshold={delay_threshold} minutes)")
        
        if target_col not in df.columns:
            # Try to infer or create target
            print(f"   Warning: '{target_col}' not found. Looking for alternatives...")
            possible_targets = [col for col in df.columns if 'delay' in col.lower()]
            if possible_targets:
                target_col = possible_targets[0]
                print(f"   Using: {target_col}")
            else:
                raise ValueError(f"Cannot find target column. Available columns: {list(df.columns)}")
        
        # Create classification target
        df['IS_DELAYED'] = (df[target_col] > delay_threshold).astype(int)
        
        # 3.3 Handle missing values
        print(f"\n✓ Handling missing values...")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from feature columns
        numeric_cols = [col for col in numeric_cols if col not in [target_col, 'IS_DELAYED']]
        
        # Impute missing values
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        for col in categorical_cols:
            if df[col].isna().any():
                df[col].fillna('Unknown', inplace=True)
        
        # 3.4 Feature engineering
        print(f"\n✓ Engineering features...")
        df = self._engineer_features(df)
        
        # 3.5 Handle outliers (winsorization)
        print(f"\n✓ Handling outliers (95th percentile clipping)...")
        for col in numeric_cols:
            if col in df.columns:
                upper_limit = df[col].quantile(0.95)
                lower_limit = df[col].quantile(0.05)
                df[col] = df[col].clip(lower_limit, upper_limit)
        
        print(f"\n✓ Preprocessing complete!")
        print(f"   Final shape: {df.shape}")
        
        self.data = df
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        
        # Find datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_cols) > 0:
            dt_col = datetime_cols[0]
            
            # Time-based features
            df['hour'] = df[dt_col].dt.hour
            df['day_of_week'] = df[dt_col].dt.dayofweek
            df['month'] = df[dt_col].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_peak_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                                   (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            print(f"   - Created time-based features from {dt_col}")
        
        # Interaction features (example)
        if 'is_peak_hour' in df.columns and 'is_weekend' in df.columns:
            df['peak_weekend'] = df['is_peak_hour'] * df['is_weekend']
        
        return df
    
    # ========================================================================
    # 4. TRAIN-TEST SPLIT (TIME-AWARE)
    # ========================================================================
    
    def time_aware_split(self, target_col='DELAY_MINUTES', 
                        test_size=0.2, date_col=None):
        """
        Time-aware train-test split (no shuffling)
        
        Parameters:
        -----------
        target_col : str
            Target column name
        test_size : float
            Fraction of data for test set
        date_col : str
            Column to use for chronological sorting (optional)
        """
        if self.data is None:
            raise ValueError("Preprocess data first")
        
        df = self.data.copy()
        
        print("\n" + "="*80)
        print("TRAIN-TEST SPLIT (TIME-AWARE)")
        print("="*80)
        
        # Sort by date if available
        if date_col and date_col in df.columns:
            df = df.sort_values(date_col).reset_index(drop=True)
            print(f"✓ Data sorted by {date_col}")
        
        # Define features and targets
        exclude_cols = [target_col, 'IS_DELAYED']
        
        # Remove datetime columns and object columns with too many unique values
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        exclude_cols.extend(datetime_cols)
        
        X = df.drop(columns=exclude_cols, errors='ignore')
        y_reg = df[target_col]
        y_clf = df['IS_DELAYED']
        
        # Remove remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Time-based split
        split_idx = int(len(df) * (1 - test_size))
        
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train_reg = y_reg.iloc[:split_idx]
        self.y_test_reg = y_reg.iloc[split_idx:]
        self.y_train_clf = y_clf.iloc[:split_idx]
        self.y_test_clf = y_clf.iloc[split_idx:]
        
        self.feature_names = X.columns.tolist()
        
        print(f"\n✓ Split complete:")
        print(f"   Train set: {len(self.X_train):,} samples")
        print(f"   Test set:  {len(self.X_test):,} samples")
        print(f"   Features:  {len(self.feature_names)}")
        print(f"\n   Regression target distribution:")
        print(f"   Train mean: {self.y_train_reg.mean():.2f} minutes")
        print(f"   Test mean:  {self.y_test_reg.mean():.2f} minutes")
        print(f"\n   Classification target distribution:")
        print(f"   Train delayed: {self.y_train_clf.sum():,} ({self.y_train_clf.mean()*100:.1f}%)")
        print(f"   Test delayed:  {self.y_test_clf.sum():,} ({self.y_test_clf.mean()*100:.1f}%)")
        
    # ========================================================================
    # 5. MODEL TRAINING - REGRESSION
    # ========================================================================
    
    def train_regression_models(self):
        """Train multiple regression models"""
        
        print("\n" + "="*80)
        print("TRAINING REGRESSION MODELS")
        print("="*80)
        
        # Baseline: Median predictor
        print("\n1. Baseline: Median Predictor")
        median_pred = np.full(len(self.y_test_reg), self.y_train_reg.median())
        baseline_rmse = np.sqrt(mean_squared_error(self.y_test_reg, median_pred))
        baseline_mae = mean_absolute_error(self.y_test_reg, median_pred)
        
        self.results['Baseline_Median'] = {
            'model_type': 'regression',
            'predictions': median_pred,
            'rmse': baseline_rmse,
            'mae': baseline_mae,
            'r2': r2_score(self.y_test_reg, median_pred)
        }
        print(f"   RMSE: {baseline_rmse:.4f} | MAE: {baseline_mae:.4f}")
        
        # Ridge Regression
        print("\n2. Ridge Regression")
        ridge = Ridge(random_state=self.random_state)
        ridge.fit(self.X_train, self.y_train_reg)
        ridge_pred = ridge.predict(self.X_test)
        
        self.models['Ridge'] = ridge
        self.results['Ridge'] = {
            'model_type': 'regression',
            'predictions': ridge_pred,
            'rmse': np.sqrt(mean_squared_error(self.y_test_reg, ridge_pred)),
            'mae': mean_absolute_error(self.y_test_reg, ridge_pred),
            'r2': r2_score(self.y_test_reg, ridge_pred)
        }
        print(f"   RMSE: {self.results['Ridge']['rmse']:.4f} | MAE: {self.results['Ridge']['mae']:.4f}")
        
        # Random Forest
        print("\n3. Random Forest Regressor")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train_reg)
        rf_pred = rf.predict(self.X_test)
        
        self.models['RandomForest_Reg'] = rf
        self.results['RandomForest_Reg'] = {
            'model_type': 'regression',
            'predictions': rf_pred,
            'rmse': np.sqrt(mean_squared_error(self.y_test_reg, rf_pred)),
            'mae': mean_absolute_error(self.y_test_reg, rf_pred),
            'r2': r2_score(self.y_test_reg, rf_pred)
        }
        print(f"   RMSE: {self.results['RandomForest_Reg']['rmse']:.4f} | MAE: {self.results['RandomForest_Reg']['mae']:.4f}")
        
        # XGBoost (if available)
        if HAS_XGB:
            print("\n4. XGBoost Regressor")
            xgb = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )
            xgb.fit(self.X_train, self.y_train_reg)
            xgb_pred = xgb.predict(self.X_test)
            
            self.models['XGBoost_Reg'] = xgb
            self.results['XGBoost_Reg'] = {
                'model_type': 'regression',
                'predictions': xgb_pred,
                'rmse': np.sqrt(mean_squared_error(self.y_test_reg, xgb_pred)),
                'mae': mean_absolute_error(self.y_test_reg, xgb_pred),
                'r2': r2_score(self.y_test_reg, xgb_pred)
            }
            print(f"   RMSE: {self.results['XGBoost_Reg']['rmse']:.4f} | MAE: {self.results['XGBoost_Reg']['mae']:.4f}")
        
        # LightGBM (if available)
        if HAS_LGBM:
            print("\n5. LightGBM Regressor")
            lgbm = LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
            lgbm.fit(self.X_train, self.y_train_reg)
            lgbm_pred = lgbm.predict(self.X_test)
            
            self.models['LightGBM_Reg'] = lgbm
            self.results['LightGBM_Reg'] = {
                'model_type': 'regression',
                'predictions': lgbm_pred,
                'rmse': np.sqrt(mean_squared_error(self.y_test_reg, lgbm_pred)),
                'mae': mean_absolute_error(self.y_test_reg, lgbm_pred),
                'r2': r2_score(self.y_test_reg, lgbm_pred)
            }
            print(f"   RMSE: {self.results['LightGBM_Reg']['rmse']:.4f} | MAE: {self.results['LightGBM_Reg']['mae']:.4f}")
        
        print("\n✓ Regression models trained successfully!")
        
    # ========================================================================
    # 6. MODEL TRAINING - CLASSIFICATION
    # ========================================================================
    
    def train_classification_models(self):
        """Train classification models for IS_DELAYED prediction"""
        
        print("\n" + "="*80)
        print("TRAINING CLASSIFICATION MODELS")
        print("="*80)
        
        # Baseline: Majority class
        print("\n1. Baseline: Majority Class Predictor")
        majority_class = self.y_train_clf.mode()[0]
        baseline_pred = np.full(len(self.y_test_clf), majority_class)
        baseline_proba = np.column_stack([1-baseline_pred, baseline_pred])
        
        precision, recall, _ = precision_recall_curve(self.y_test_clf, baseline_proba[:, 1])
        pr_auc = auc(recall, precision)
        
        self.results['Baseline_Majority'] = {
            'model_type': 'classification',
            'predictions': baseline_pred,
            'probabilities': baseline_proba,
            'pr_auc': pr_auc,
            'f2_score': fbeta_score(self.y_test_clf, baseline_pred, beta=2),
            'recall': recall_score(self.y_test_clf, baseline_pred),
            'precision': precision_score(self.y_test_clf, baseline_pred)
        }
        print(f"   PR-AUC: {pr_auc:.4f} | F2: {self.results['Baseline_Majority']['f2_score']:.4f}")
        
        # Logistic Regression
        print("\n2. Logistic Regression")
        logreg = LogisticRegression(
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000
        )
        logreg.fit(self.X_train, self.y_train_clf)
        logreg_pred = logreg.predict(self.X_test)
        logreg_proba = logreg.predict_proba(self.X_test)
        
        precision, recall, _ = precision_recall_curve(self.y_test_clf, logreg_proba[:, 1])
        pr_auc = auc(recall, precision)
        
        self.models['LogisticRegression'] = logreg
        self.results['LogisticRegression'] = {
            'model_type': 'classification',
            'predictions': logreg_pred,
            'probabilities': logreg_proba,
            'pr_auc': pr_auc,
            'f2_score': fbeta_score(self.y_test_clf, logreg_pred, beta=2),
            'recall': recall_score(self.y_test_clf, logreg_pred),
            'precision': precision_score(self.y_test_clf, logreg_pred)
        }
        print(f"   PR-AUC: {pr_auc:.4f} | F2: {self.results['LogisticRegression']['f2_score']:.4f}")
        
        # Random Forest
        print("\n3. Random Forest Classifier")
        rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_clf.fit(self.X_train, self.y_train_clf)
        rf_clf_pred = rf_clf.predict(self.X_test)
        rf_clf_proba = rf_clf.predict_proba(self.X_test)
        
        precision, recall, _ = precision_recall_curve(self.y_test_clf, rf_clf_proba[:, 1])
        pr_auc = auc(recall, precision)
        
        self.models['RandomForest_Clf'] = rf_clf
        self.results['RandomForest_Clf'] = {
            'model_type': 'classification',
            'predictions': rf_clf_pred,
            'probabilities': rf_clf_proba,
            'pr_auc': pr_auc,
            'f2_score': fbeta_score(self.y_test_clf, rf_clf_pred, beta=2),
            'recall': recall_score(self.y_test_clf, rf_clf_pred),
            'precision': precision_score(self.y_test_clf, rf_clf_pred)
        }
        print(f"   PR-AUC: {pr_auc:.4f} | F2: {self.results['RandomForest_Clf']['f2_score']:.4f}")
        
        # XGBoost (if available)
        if HAS_XGB:
            print("\n4. XGBoost Classifier")
            scale_pos_weight = (self.y_train_clf == 0).sum() / (self.y_train_clf == 1).sum()
            xgb_clf = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                n_jobs=-1
            )
            xgb_clf.fit(self.X_train, self.y_train_clf)
            xgb_clf_pred = xgb_clf.predict(self.X_test)
            xgb_clf_proba = xgb_clf.predict_proba(self.X_test)
            
            precision, recall, _ = precision_recall_curve(self.y_test_clf, xgb_clf_proba[:, 1])
            pr_auc = auc(recall, precision)
            
            self.models['XGBoost_Clf'] = xgb_clf
            self.results['XGBoost_Clf'] = {
                'model_type': 'classification',
                'predictions': xgb_clf_pred,
                'probabilities': xgb_clf_proba,
                'pr_auc': pr_auc,
                'f2_score': fbeta_score(self.y_test_clf, xgb_clf_pred, beta=2),
                'recall': recall_score(self.y_test_clf, xgb_clf_pred),
                'precision': precision_score(self.y_test_clf, xgb_clf_pred)
            }
            print(f"   PR-AUC: {pr_auc:.4f} | F2: {self.results['XGBoost_Clf']['f2_score']:.4f}")
        
        print("\n✓ Classification models trained successfully!")
    
    # ========================================================================
    # 7. MODEL COMPARISON
    # ========================================================================
    
    def compare_models(self) -> pd.DataFrame:
        """Generate comprehensive model comparison table"""
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            if results['model_type'] == 'regression':
                comparison_data.append({
                    'Model': model_name,
                    'Task': 'Regression',
                    'RMSE': f"{results['rmse']:.4f}",
                    'MAE': f"{results['mae']:.4f}",
                    'R²': f"{results['r2']:.4f}",
                    'PR-AUC': '-',
                    'F2': '-',
                    'Recall': '-'
                })
            else:  # classification
                comparison_data.append({
                    'Model': model_name,
                    'Task': 'Classification',
                    'RMSE': '-',
                    'MAE': '-',
                    'R²': '-',
                    'PR-AUC': f"{results['pr_auc']:.4f}",
                    'F2': f"{results['f2_score']:.4f}",
                    'Recall': f"{results['recall']:.4f}"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    # ========================================================================
    # 8. FEATURE IMPORTANCE
    # ========================================================================
    
    def plot_feature_importance(self, model_name='RandomForest_Reg', top_n=20):
        """Plot feature importance for tree-based models"""
        
        if model_name not in self.models:
            print(f"Model '{model_name}' not found!")
            return
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(10, 8))
            plt.title(f'Top {top_n} Feature Importances - {model_name}')
            plt.barh(range(top_n), importances[indices])
            plt.yticks(range(top_n), [self.feature_names[i] for i in indices])
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.gca().invert_yaxis()
            plt.show()
        else:
            print(f"Model {model_name} does not have feature_importances_ attribute")
    
    # ========================================================================
    # 9. SAVE/LOAD PIPELINE
    # ========================================================================
    
    def save_pipeline(self, filepath: str, model_name: str):
        """Save trained model and preprocessing pipeline"""
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found!")
        
        pipeline_data = {
            'model': self.models[model_name],
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        
        joblib.dump(pipeline_data, filepath)
        print(f"✓ Pipeline saved to: {filepath}")
    
    def load_pipeline(self, filepath: str):
        """Load saved pipeline"""
        
        pipeline_data = joblib.load(filepath)
        print(f"✓ Pipeline loaded from: {filepath}")
        return pipeline_data


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Initialize pipeline
    pipeline = RailwayDelayPipeline(random_state=42)
    
    # Load data
    data_path = "../docs/merged_train_data.csv"
    pipeline.load_data(data_path)
    
    # Generate data dictionary
    data_dict = pipeline.generate_data_dictionary()
    print("\n" + "="*80)
    print("DATA DICTIONARY")
    print("="*80)
    print(data_dict.to_string(index=False))
    
    # Print summary
    pipeline.print_data_summary()
    
    # Preprocess
    pipeline.preprocess_data(target_col='DELAY_MINUTES', delay_threshold=5)
    
    # Train-test split
    pipeline.time_aware_split(target_col='DELAY_MINUTES', test_size=0.2)
    
    # Train models
    pipeline.train_regression_models()
    pipeline.train_classification_models()
    
    # Compare models
    comparison = pipeline.compare_models()
    
    # Feature importance
    pipeline.plot_feature_importance('RandomForest_Reg', top_n=15)
    
    # Save best model
    # pipeline.save_pipeline('models/best_model.pkl', 'XGBoost_Reg')
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE!")
    print("="*80)
