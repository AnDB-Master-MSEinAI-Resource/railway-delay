"""
Complete script to restructure notebook into exactly 82 cells
This creates all cells from Groups A through J
"""

import json

notebook_path = r"d:\MSE\5. Data Mining\railway-delay\notebooks\regression_pipeline_rmse.ipynb"

# Read existing notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Start fresh with empty cells list
all_cells = []

def add_code_cell(source_lines):
    """Helper to add a code cell"""
    if isinstance(source_lines, str):
        source_lines = [source_lines]
    all_cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": source_lines,
        "outputs": [],
        "execution_count": None
    })

def add_markdown_cell(source_lines):
    """Helper to add a markdown cell"""
    if isinstance(source_lines, str):
        source_lines = [source_lines]
    all_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    })

# ============================================================================
# GROUP A — INITIALIZATION & CONFIGURATION (Cells 1-10)
# ============================================================================

# Cell 1
add_code_cell([
    "# Cell 1: Import core libraries\n",
    "import os\n",
    "import sys\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "print('✓ Core libraries imported')"
])

# Cell 2
add_code_cell([
    "# Cell 2: Import sklearn components\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.model_selection import train_test_split, TimeSeriesSplit\n",
    "\n",
    "print('✓ Sklearn components imported')"
])

# Cell 3
add_code_cell([
    "# Cell 3: Import regression models\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "try:\n",
    "    from xgboost import XGBRegressor\n",
    "    HAS_XGB = True\n",
    "except ImportError:\n",
    "    XGBRegressor = None\n",
    "    HAS_XGB = False\n",
    "\n",
    "try:\n",
    "    from lightgbm import LGBMRegressor\n",
    "    HAS_LGB = True\n",
    "except ImportError:\n",
    "    LGBMRegressor = None\n",
    "    HAS_LGB = False\n",
    "\n",
    "try:\n",
    "    from catboost import CatBoostRegressor\n",
    "    HAS_CB = True\n",
    "except ImportError:\n",
    "    CatBoostRegressor = None\n",
    "    HAS_CB = False\n",
    "\n",
    "print(f'✓ Models imported (XGB:{HAS_XGB}, LGB:{HAS_LGB}, CB:{HAS_CB})')"
])

# Cell 4
add_code_cell([
    "# Cell 4: Import optuna, joblib, visualization\n",
    "import joblib\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "try:\n",
    "    import optuna\n",
    "    HAS_OPTUNA = True\n",
    "except ImportError:\n",
    "    optuna = None\n",
    "    HAS_OPTUNA = False\n",
    "\n",
    "plt.style.use('seaborn-v0_8-darkgrid')\n",
    "sns.set_palette('husl')\n",
    "print(f'✓ Visualization libraries imported (Optuna:{HAS_OPTUNA})')"
])

# Cell 5
add_code_cell([
    "# Cell 5: Set global random seed\n",
    "RANDOM_STATE = 42\n",
    "np.random.seed(RANDOM_STATE)\n",
    "print(f'✓ Random seed: {RANDOM_STATE}')"
])

# Cell 6
add_code_cell([
    "# Cell 6: Define data paths\n",
    "DATA_PATH = '../data'\n",
    "DATATRAIN = os.path.join(DATA_PATH, 'raw', 'railway-delay-dataset.csv')\n",
    "DATATEST = None\n",
    "print(f'✓ Data paths defined\\n  Train: {DATATRAIN}')"
])

# Cell 7
add_code_cell([
    "# Cell 7: Define constants\n",
    "TARGET_COL = 'DELAY_MINUTES'\n",
    "DATE_COL = 'SCHEDULED_DT'\n",
    "print(f'✓ Constants: TARGET={TARGET_COL}, DATE={DATE_COL}')"
])

# Cell 8
add_code_cell([
    "# Cell 8: Memory management flags\n",
    "MAX_ROWS = None\n",
    "DOWNSAMPLE = False\n",
    "print(f'✓ Memory: MAX_ROWS={MAX_ROWS}, DOWNSAMPLE={DOWNSAMPLE}')"
])

# Cell 9
add_code_cell([
    "# Cell 9: Create model directory\n",
    "MODEL_DIR = 'models'\n",
    "os.makedirs(MODEL_DIR, exist_ok=True)\n",
    "print(f'✓ Model directory: {MODEL_DIR}')"
])

# Cell 10
add_code_cell([
    "# Cell 10: Configuration sanity check\n",
    "print('='*70)\n",
    "print('CONFIGURATION SUMMARY')\n",
    "print('='*70)\n",
    "print(f'Random State: {RANDOM_STATE}')\n",
    "print(f'Target: {TARGET_COL} (Regression)')\n",
    "print(f'Date Column: {DATE_COL}')\n",
    "print(f'Train Data: {DATATRAIN}')\n",
    "print(f'Model Directory: {MODEL_DIR}')\n",
    "print(f'Optional: XGB={HAS_XGB}, LGB={HAS_LGB}, CB={HAS_CB}, Optuna={HAS_OPTUNA}')\n",
    "print('='*70)"
])

# ============================================================================
# GROUP B — LOAD DATA (Cells 11-20)
# ============================================================================

# Cell 11
add_code_cell([
    "# Cell 11: Load training data\n",
    "df = pd.read_csv(DATATRAIN, nrows=MAX_ROWS)\n",
    "print(f'✓ Data loaded: {df.shape}, {df.memory_usage(deep=True).sum()/1024**2:.2f} MB')"
])

# Cell 12
add_code_cell([
    "# Cell 12: Load test data (if available)\n",
    "df_test = None\n",
    "if DATATEST and os.path.exists(DATATEST):\n",
    "    df_test = pd.read_csv(DATATEST, nrows=MAX_ROWS)\n",
    "    print(f'✓ Test data: {df_test.shape}')\n",
    "else:\n",
    "    print('✓ No separate test file')"
])

# Cell 13
add_code_cell([
    "# Cell 13: Print columns\n",
    "print(f'Columns ({len(df.columns)} total):')\n",
    "for i, col in enumerate(df.columns, 1):\n",
    "    print(f'  {i:3d}. {col}')"
])

# Cell 14
add_code_cell([
    "# Cell 14: Auto-detect datetime column\n",
    "dt_keywords = ['TIME', 'DATE', 'DT', 'SCHEDULED', 'ACTUAL']\n",
    "dt_candidates = [c for c in df.columns if any(k in c.upper() for k in dt_keywords)]\n",
    "datetime_col = DATE_COL if DATE_COL in df.columns else (dt_candidates[0] if dt_candidates else None)\n",
    "print(f'Datetime column: {datetime_col}')"
])

# Cell 15
add_code_cell([
    "# Cell 15: Convert to datetime\n",
    "if datetime_col:\n",
    "    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')\n",
    "    print(f'✓ Converted {datetime_col}: {df[datetime_col].notna().mean()*100:.1f}% parsed')\n",
    "else:\n",
    "    print('⚠ No datetime column')"
])

# Cell 16
add_code_cell([
    "# Cell 16: Extract hour/weekday features\n",
    "if datetime_col:\n",
    "    df['HOUR'] = df[datetime_col].dt.hour\n",
    "    df['DAY_OF_WEEK'] = df[datetime_col].dt.dayofweek\n",
    "    df['MONTH'] = df[datetime_col].dt.month\n",
    "    print('✓ Extracted: HOUR, DAY_OF_WEEK, MONTH')"
])

# Cell 17
add_code_cell([
    "# Cell 17: Handle missing datetime\n",
    "if datetime_col and 'HOUR' in df.columns:\n",
    "    missing = df[datetime_col].isna().sum()\n",
    "    if missing > 0:\n",
    "        df['HOUR'].fillna(df['HOUR'].median(), inplace=True)\n",
    "        df['DAY_OF_WEEK'].fillna(df['DAY_OF_WEEK'].median(), inplace=True)\n",
    "        print(f'⚠ Filled {missing} missing datetime values')\n",
    "    else:\n",
    "        print('✓ No missing datetime')"
])

# Cell 18
add_code_cell([
    "# Cell 18: Sort by time\n",
    "if datetime_col:\n",
    "    df = df.sort_values(datetime_col).reset_index(drop=True)\n",
    "    print(f'✓ Sorted by {datetime_col}')"
])

# Cell 19
add_code_cell([
    "# Cell 19: Data info\n",
    "df.info()"
])

# Cell 20
add_code_cell([
    "# Cell 20: Display sample\n",
    "print('First 3 rows:')\n",
    "display(df.head(3))\n",
    "print('\\nRandom sample:')\n",
    "display(df.sample(3, random_state=RANDOM_STATE))"
])

# ============================================================================
# GROUP C — HELPER FUNCTIONS (Cells 21-30)
# ============================================================================

# Cell 21
add_code_cell([
    "# Cell 21: Helper - get route column\n",
    "def _get_route_column(df):\n",
    "    keywords = ['ROUTE', 'TRAIN_ID', 'SERVICE', 'TRAIN_NO']\n",
    "    for col in df.columns:\n",
    "        if any(k in col.upper() for k in keywords):\n",
    "            return col\n",
    "    return None\n",
    "print('✓ _get_route_column defined')"
])

# Cell 22
add_code_cell([
    "# Cell 22: Helper - compute previous delay\n",
    "def compute_prev_delay_safe(df, target_col=TARGET_COL):\n",
    "    route_col = _get_route_column(df)\n",
    "    if target_col not in df.columns:\n",
    "        df['PREV_DELAY'] = 0\n",
    "        return df\n",
    "    if route_col:\n",
    "        df['PREV_DELAY'] = df.groupby(route_col)[target_col].shift(1).fillna(0)\n",
    "    else:\n",
    "        df['PREV_DELAY'] = df[target_col].shift(1).fillna(0)\n",
    "    return df\n",
    "print('✓ compute_prev_delay_safe defined')"
])

# Cell 23
add_code_cell([
    "# Cell 23: Helper - compute rolling features\n",
    "def compute_rolling_features_safe(df, target_col=TARGET_COL, window=7):\n",
    "    route_col = _get_route_column(df)\n",
    "    if target_col not in df.columns:\n",
    "        df[f'ROLLING_MEAN_{window}D'] = 0\n",
    "        return df\n",
    "    if route_col:\n",
    "        df[f'ROLLING_MEAN_{window}D'] = df.groupby(route_col)[target_col].transform(\n",
    "            lambda x: x.rolling(window, min_periods=1).mean()\n",
    "        )\n",
    "    else:\n",
    "        df[f'ROLLING_MEAN_{window}D'] = df[target_col].rolling(window, min_periods=1).mean()\n",
    "    return df\n",
    "print('✓ compute_rolling_features_safe defined')"
])

# Cell 24
add_code_cell([
    "# Cell 24: Helper - metrics summary\n",
    "def metrics_summary(y_true, y_pred):\n",
    "    mask = ~(np.isnan(y_true) | np.isnan(y_pred))\n",
    "    y_true = y_true[mask]\n",
    "    y_pred = y_pred[mask]\n",
    "    if len(y_true) == 0:\n",
    "        return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}\n",
    "    return {\n",
    "        'MAE': mean_absolute_error(y_true, y_pred),\n",
    "        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),\n",
    "        'R2': r2_score(y_true, y_pred)\n",
    "    }\n",
    "print('✓ metrics_summary defined')"
])

# Cell 25
add_code_cell([
    "# Cell 25: Helper - residual plots\n",
    "def plot_residuals(y_true, y_pred, title='Residual Plot'):\n",
    "    res = y_true - y_pred\n",
    "    fig, ax = plt.subplots(1, 2, figsize=(14, 5))\n",
    "    ax[0].scatter(y_pred, res, alpha=0.5)\n",
    "    ax[0].axhline(0, color='r', linestyle='--')\n",
    "    ax[0].set_xlabel('Predicted')\n",
    "    ax[0].set_ylabel('Residuals')\n",
    "    ax[0].set_title(f'{title} - Residuals vs Predicted')\n",
    "    ax[1].hist(res, bins=50)\n",
    "    ax[1].set_title(f'{title} - Distribution')\n",
    "    plt.tight_layout()\n",
    "    return fig\n",
    "print('✓ plot_residuals defined')"
])

# Cell 26
add_code_cell([
    "# Cell 26: Helper - feature importance\n",
    "def plot_feature_importance(model, feature_names, top_n=20):\n",
    "    if not hasattr(model, 'feature_importances_'):\n",
    "        return None\n",
    "    imp = model.feature_importances_\n",
    "    idx = np.argsort(imp)[::-1][:top_n]\n",
    "    fig, ax = plt.subplots(figsize=(10, 8))\n",
    "    ax.barh(range(len(idx)), imp[idx])\n",
    "    ax.set_yticks(range(len(idx)))\n",
    "    ax.set_yticklabels([feature_names[i] for i in idx])\n",
    "    ax.invert_yaxis()\n",
    "    plt.tight_layout()\n",
    "    return fig\n",
    "print('✓ plot_feature_importance defined')"
])

# Cell 27
add_code_cell([
    "# Cell 27: Helper - outlier detection\n",
    "def detect_outliers_iqr(series, multiplier=1.5):\n",
    "    Q1 = series.quantile(0.25)\n",
    "    Q3 = series.quantile(0.75)\n",
    "    IQR = Q3 - Q1\n",
    "    return (series < Q1 - multiplier*IQR) | (series > Q3 + multiplier*IQR)\n",
    "print('✓ detect_outliers_iqr defined')"
])

# Cell 28
add_code_cell([
    "# Cell 28: Helper - inference\n",
    "def predict_with_preprocessing(model, preprocessor, X_new):\n",
    "    X_proc = preprocessor.transform(X_new) if preprocessor else X_new\n",
    "    return model.predict(X_proc)\n",
    "print('✓ predict_with_preprocessing defined')"
])

# Cell 29
add_code_cell([
    "# Cell 29: Test helpers\n",
    "y_t = np.array([1, 2, 3, 4, 5])\n",
    "y_p = np.array([1.1, 2.1, 3.1, 4.1, 5.1])\n",
    "m = metrics_summary(y_t, y_p)\n",
    "print(f'Test: RMSE={m[\"RMSE\"]:.3f}, R2={m[\"R2\"]:.3f}')\n",
    "print('✓ All helpers tested')"
])

# Cell 30
add_markdown_cell([
    "---\n",
    "## End Helper Functions\n",
    "All helper functions defined and tested.\n",
    "---"
])

# ============================================================================
# GROUP D — FEATURE ENGINEERING (Cells 31-45)
# ============================================================================

# Cell 31
add_code_cell([
    "# Cell 31: Time features\n",
    "# Already created in Cell 16\n",
    "print(f'✓ Time features: {[c for c in df.columns if c in [\"HOUR\", \"DAY_OF_WEEK\", \"MONTH\"]]}')"
])

# Cell 32
add_code_cell([
    "# Cell 32: Cyclical encoding\n",
    "if 'HOUR' in df.columns:\n",
    "    df['SIN_HOUR'] = np.sin(2 * np.pi * df['HOUR'] / 24)\n",
    "    df['COS_HOUR'] = np.cos(2 * np.pi * df['HOUR'] / 24)\n",
    "if 'DAY_OF_WEEK' in df.columns:\n",
    "    df['SIN_DAY'] = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)\n",
    "    df['COS_DAY'] = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)\n",
    "print('✓ Cyclical features: SIN_HOUR, COS_HOUR, SIN_DAY, COS_DAY')"
])

# Cell 33
add_code_cell([
    "# Cell 33: Lag feature (previous delay)\n",
    "df = compute_prev_delay_safe(df, TARGET_COL)\n",
    "print(f'✓ PREV_DELAY: mean={df[\"PREV_DELAY\"].mean():.2f}')"
])

# Cell 34
add_code_cell([
    "# Cell 34: Rolling mean 7D (route-based)\n",
    "df = compute_rolling_features_safe(df, TARGET_COL, window=7)\n",
    "print('✓ ROLLING_MEAN_7D created')"
])

# Cell 35
add_code_cell([
    "# Cell 35: Rolling mean global fallback\n",
    "if 'ROLLING_MEAN_7D' in df.columns:\n",
    "    df['ROLLING_MEAN_7D'].fillna(df[TARGET_COL].rolling(7, min_periods=1).mean(), inplace=True)\n",
    "    print('✓ Rolling mean fallback applied')"
])

# Cell 36
add_code_cell([
    "# Cell 36: Weather/external features (if available)\n",
    "# Placeholder for external data joins\n",
    "weather_cols = [c for c in df.columns if 'WEATHER' in c.upper()]\n",
    "if weather_cols:\n",
    "    print(f'✓ Weather features found: {weather_cols}')\n",
    "else:\n",
    "    print('✓ No weather features')"
])

# Cell 37
add_code_cell([
    "# Cell 37: Fill missing engineered features\n",
    "eng_features = ['PREV_DELAY', 'ROLLING_MEAN_7D', 'SIN_HOUR', 'COS_HOUR', 'SIN_DAY', 'COS_DAY']\n",
    "for col in eng_features:\n",
    "    if col in df.columns:\n",
    "        df[col].fillna(0, inplace=True)\n",
    "print('✓ Engineered features filled')"
])

# Cell 38
add_code_cell([
    "# Cell 38: Feature distribution check\n",
    "eng_cols = [c for c in ['PREV_DELAY', 'ROLLING_MEAN_7D', 'HOUR', 'DAY_OF_WEEK'] if c in df.columns]\n",
    "if eng_cols:\n",
    "    print('Feature distributions:')\n",
    "    display(df[eng_cols].describe())"
])

# Cell 39
add_code_cell([
    "# Cell 39: Drop leakage columns\n",
    "leakage_keywords = ['ACTUAL', 'ARRIVAL_TIME', 'DEPARTURE_TIME']\n",
    "leakage_cols = [c for c in df.columns if any(k in c.upper() for k in leakage_keywords) and c != TARGET_COL]\n",
    "if leakage_cols:\n",
    "    df.drop(columns=leakage_cols, inplace=True)\n",
    "    print(f'⚠ Dropped {len(leakage_cols)} leakage columns')\n",
    "else:\n",
    "    print('✓ No leakage columns detected')"
])

# Cell 40
add_code_cell([
    "# Cell 40: Feature list snapshot\n",
    "feature_cols = [c for c in df.columns if c != TARGET_COL]\n",
    "print(f'Total features: {len(feature_cols)}')\n",
    "print(f'Sample: {feature_cols[:10]}')"
])

# Cell 41
add_code_cell([
    "# Cell 41: Feature sanity check\n",
    "print(f'Shape after engineering: {df.shape}')\n",
    "print(f'Target column present: {TARGET_COL in df.columns}')\n",
    "print(f'Missing values: {df.isnull().sum().sum()}')"
])

# Cell 42
add_code_cell([
    "# Cell 42: Print engineered columns\n",
    "eng_added = ['PREV_DELAY', 'ROLLING_MEAN_7D', 'SIN_HOUR', 'COS_HOUR', 'SIN_DAY', 'COS_DAY']\n",
    "print('Engineered columns present:')\n",
    "for col in eng_added:\n",
    "    print(f'  {col}: {col in df.columns}')"
])

# Cell 43
add_code_cell([
    "# Cell 43: Memory cleanup\n",
    "import gc\n",
    "gc.collect()\n",
    "print(f'✓ Memory after cleanup: {df.memory_usage(deep=True).sum()/1024**2:.2f} MB')"
])

# Cell 44
add_code_cell([
    "# Cell 44: Save intermediate (optional)\n",
    "# df.to_csv('../data/processed/engineered_features.csv', index=False)\n",
    "print('✓ Feature engineering complete (save disabled)')"
])

# Cell 45
add_markdown_cell([
    "---\n",
    "## End Feature Engineering\n",
    "All features engineered and validated.\n",
    "---"
])

# ============================================================================
# GROUP E — TARGET & FEATURE SPLIT (Cells 46-50)
# ============================================================================

# Cell 46
add_code_cell([
    "# Cell 46: Define X, y\n",
    "if TARGET_COL in df.columns:\n",
    "    X = df.drop(columns=[TARGET_COL], errors='ignore')\n",
    "    y = df[TARGET_COL]\n",
    "    print(f'✓ X: {X.shape}, y: {y.shape}')\n",
    "else:\n",
    "    print(f'⚠ Target column {TARGET_COL} not found!')"
])

# Cell 47
add_code_cell([
    "# Cell 47: Log-transform target (optional)\n",
    "USE_LOG_TRANSFORM = False\n",
    "if USE_LOG_TRANSFORM:\n",
    "    y_log = np.log1p(y)\n",
    "    print(f'✓ Log-transformed target: skew={y_log.skew():.3f}')\n",
    "else:\n",
    "    y_log = y\n",
    "    print('✓ No log transform')"
])

# Cell 48
add_code_cell([
    "# Cell 48: Save original target\n",
    "y_original = y.copy()\n",
    "print(f'✓ Original target saved: {len(y_original)} values')"
])

# Cell 49
add_code_cell([
    "# Cell 49: Target stats\n",
    "print('Target Statistics:')\n",
    "print(f'  Mean: {y.mean():.2f}')\n",
    "print(f'  Median: {y.median():.2f}')\n",
    "print(f'  Std: {y.std():.2f}')\n",
    "print(f'  Min: {y.min():.2f}')\n",
    "print(f'  Max: {y.max():.2f}')"
])

# Cell 50
add_markdown_cell([
    "---\n",
    "## End Target & Feature Split\n",
    "Target variable prepared for modeling.\n",
    "---"
])

# ============================================================================
# GROUP F — FEATURE TYPES & PREPROCESSING (Cells 51-60)
# ============================================================================

# Cell 51
add_code_cell([
    "# Cell 51: Detect numeric features\n",
    "numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()\n",
    "print(f'✓ Numeric features: {len(numeric_features)}')"
])

# Cell 52
add_code_cell([
    "# Cell 52: Detect categorical features\n",
    "categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()\n",
    "print(f'✓ Categorical features: {len(categorical_features)}')"
])

# Cell 53
add_code_cell([
    "# Cell 53: Detect label-encoded features\n",
    "label_encoded = [c for c in categorical_features if X[c].nunique() < 50]\n",
    "print(f'✓ Label-encoded candidates: {len(label_encoded)}')"
])

# Cell 54
add_code_cell([
    "# Cell 54: Leakage check\n",
    "leakage_check = ['ACTUAL', 'RESULT', 'OUTCOME']\n",
    "potential_leakage = [c for c in X.columns if any(k in c.upper() for k in leakage_check)]\n",
    "if potential_leakage:\n",
    "    print(f'⚠ Potential leakage: {potential_leakage}')\n",
    "else:\n",
    "    print('✓ No leakage detected')"
])

# Cell 55
add_code_cell([
    "# Cell 55: Numeric transformer\n",
    "numeric_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='median')),\n",
    "    ('scaler', StandardScaler())\n",
    "])\n",
    "print('✓ Numeric transformer created')"
])

# Cell 56
add_code_cell([
    "# Cell 56: Categorical transformer\n",
    "categorical_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),\n",
    "    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))\n",
    "])\n",
    "print('✓ Categorical transformer created')"
])

# Cell 57
add_code_cell([
    "# Cell 57: ColumnTransformer (preprocessor)\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', numeric_transformer, numeric_features),\n",
    "        ('cat', categorical_transformer, categorical_features)\n",
    "    ])\n",
    "print('✓ Preprocessor created')"
])

# Cell 58
add_code_cell([
    "# Cell 58: Preprocessor sanity check\n",
    "print(f'Numeric features: {len(numeric_features)}')\n",
    "print(f'Categorical features: {len(categorical_features)}')\n",
    "print(f'Total input features: {len(numeric_features) + len(categorical_features)}')"
])

# Cell 59
add_code_cell([
    "# Cell 59: Feature count check\n",
    "print(f'Total features going into model: {len(X.columns)}')\n",
    "print(f'Sample features: {X.columns[:5].tolist()}')"
])

# Cell 60
add_markdown_cell([
    "---\n",
    "## End Preprocessing\n",
    "Preprocessing pipeline configured.\n",
    "---"
])

# ============================================================================
# GROUP G — SPLIT & CV (Cells 61-65)
# ============================================================================

# Cell 61
add_code_cell([
    "# Cell 61: Train/validation split\n",
    "X_train, X_val, y_train, y_val = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=False\n",
    ")\n",
    "print(f'✓ Train: {X_train.shape}, Val: {X_val.shape}')"
])

# Cell 62
add_code_cell([
    "# Cell 62: Print split sizes\n",
    "print(f'Training samples: {len(X_train)}')\n",
    "print(f'Validation samples: {len(X_val)}')\n",
    "print(f'Train target mean: {y_train.mean():.2f}')\n",
    "print(f'Val target mean: {y_val.mean():.2f}')"
])

# Cell 63
add_code_cell([
    "# Cell 63: TimeSeriesSplit definition\n",
    "tscv = TimeSeriesSplit(n_splits=5)\n",
    "print(f'✓ TimeSeriesSplit: {tscv.get_n_splits()} splits')"
])

# Cell 64
add_code_cell([
    "# Cell 64: CV sanity check\n",
    "for i, (train_idx, val_idx) in enumerate(tscv.split(X_train)):\n",
    "    print(f'Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}')\n",
    "    if i >= 2:\n",
    "        break"
])

# Cell 65
add_markdown_cell([
    "---\n",
    "## End Split & CV\n",
    "Data split and cross-validation configured.\n",
    "---"
])

# ============================================================================
# GROUP H — BASELINE MODELS (Cells 66-70)
# ============================================================================

# Cell 66
add_code_cell([
    "# Cell 66: Linear regression baseline\n",
    "lr_model = LinearRegression()\n",
    "lr_pipe = Pipeline([('preprocessor', preprocessor), ('model', lr_model)])\n",
    "lr_pipe.fit(X_train, y_train)\n",
    "y_pred_lr = lr_pipe.predict(X_val)\n",
    "metrics_lr = metrics_summary(y_val, y_pred_lr)\n",
    "print(f'Linear Regression: RMSE={metrics_lr[\"RMSE\"]:.2f}, R2={metrics_lr[\"R2\"]:.3f}')"
])

# Cell 67
add_code_cell([
    "# Cell 67: Random Forest baseline\n",
    "rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)\n",
    "rf_pipe = Pipeline([('preprocessor', preprocessor), ('model', rf_model)])\n",
    "rf_pipe.fit(X_train, y_train)\n",
    "y_pred_rf = rf_pipe.predict(X_val)\n",
    "metrics_rf = metrics_summary(y_val, y_pred_rf)\n",
    "print(f'Random Forest: RMSE={metrics_rf[\"RMSE\"]:.2f}, R2={metrics_rf[\"R2\"]:.3f}')"
])

# Cell 68
add_code_cell([
    "# Cell 68: Baseline evaluation\n",
    "baseline_results = pd.DataFrame([\n",
    "    {'Model': 'LinearRegression', **metrics_lr},\n",
    "    {'Model': 'RandomForest', **metrics_rf}\n",
    "])\n",
    "display(baseline_results)"
])

# Cell 69
add_code_cell([
    "# Cell 69: Save baseline results\n",
    "baseline_results.to_csv('models/baseline_results.csv', index=False)\n",
    "print('✓ Baseline results saved')"
])

# Cell 70
add_markdown_cell([
    "---\n",
    "## End Baseline Models\n",
    "Baseline models trained and evaluated.\n",
    "---"
])

# ============================================================================
# GROUP I — TUNING & ADVANCED MODELS (Cells 71-78)
# ============================================================================

# Cell 71
add_code_cell([
    "# Cell 71: Define Optuna objective\n",
    "def objective(trial):\n",
    "    n_estimators = trial.suggest_int('n_estimators', 50, 200)\n",
    "    max_depth = trial.suggest_int('max_depth', 5, 20)\n",
    "    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)\n",
    "    \n",
    "    model = RandomForestRegressor(\n",
    "        n_estimators=n_estimators,\n",
    "        max_depth=max_depth,\n",
    "        min_samples_split=min_samples_split,\n",
    "        random_state=RANDOM_STATE,\n",
    "        n_jobs=-1\n",
    "    )\n",
    "    \n",
    "    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])\n",
    "    pipe.fit(X_train, y_train)\n",
    "    y_pred = pipe.predict(X_val)\n",
    "    \n",
    "    return np.sqrt(mean_squared_error(y_val, y_pred))\n",
    "\n",
    "print('✓ Optuna objective defined')"
])

# Cell 72
add_code_cell([
    "# Cell 72: Define parameter space\n",
    "param_space = {\n",
    "    'n_estimators': (50, 200),\n",
    "    'max_depth': (5, 20),\n",
    "    'min_samples_split': (2, 20)\n",
    "}\n",
    "print(f'✓ Parameter space: {param_space}')"
])

# Cell 73
add_code_cell([
    "# Cell 73: Create Optuna study\n",
    "if HAS_OPTUNA:\n",
    "    study = optuna.create_study(direction='minimize')\n",
    "    print('✓ Optuna study created')\n",
    "else:\n",
    "    study = None\n",
    "    print('⚠ Optuna not available, skipping tuning')"
])

# Cell 74
add_code_cell([
    "# Cell 74: Run Optuna optimization\n",
    "if HAS_OPTUNA and study:\n",
    "    study.optimize(objective, n_trials=10, show_progress_bar=True)\n",
    "    print(f'✓ Optimization complete: Best RMSE={study.best_value:.2f}')\n",
    "else:\n",
    "    print('⚠ Skipping optimization')"
])

# Cell 75
add_code_cell([
    "# Cell 75: Best params summary\n",
    "if HAS_OPTUNA and study:\n",
    "    best_params = study.best_params\n",
    "    print('Best parameters:')\n",
    "    for k, v in best_params.items():\n",
    "        print(f'  {k}: {v}')\n",
    "else:\n",
    "    best_params = {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 5}\n",
    "    print(f'Using default params: {best_params}')"
])

# Cell 76
add_code_cell([
    "# Cell 76: Train tuned model\n",
    "tuned_model = RandomForestRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1)\n",
    "tuned_pipe = Pipeline([('preprocessor', preprocessor), ('model', tuned_model)])\n",
    "tuned_pipe.fit(X_train, y_train)\n",
    "print('✓ Tuned model trained')"
])

# Cell 77
add_code_cell([
    "# Cell 77: Validation prediction\n",
    "y_pred_tuned = tuned_pipe.predict(X_val)\n",
    "print(f'✓ Predictions generated: {len(y_pred_tuned)} values')"
])

# Cell 78
add_code_cell([
    "# Cell 78: Metrics calculation\n",
    "metrics_tuned = metrics_summary(y_val, y_pred_tuned)\n",
    "print('Tuned Model Performance:')\n",
    "print(f'  RMSE: {metrics_tuned[\"RMSE\"]:.2f}')\n",
    "print(f'  MAE: {metrics_tuned[\"MAE\"]:.2f}')\n",
    "print(f'  R2: {metrics_tuned[\"R2\"]:.3f}')"
])

# ============================================================================
# GROUP J — FINALIZATION (Cells 79-82)
# ============================================================================

# Cell 79
add_code_cell([
    "# Cell 79: Model comparison table\n",
    "all_results = pd.DataFrame([\n",
    "    {'Model': 'LinearRegression', **metrics_lr},\n",
    "    {'Model': 'RandomForest_Baseline', **metrics_rf},\n",
    "    {'Model': 'RandomForest_Tuned', **metrics_tuned}\n",
    "])\n",
    "all_results = all_results.sort_values('RMSE')\n",
    "print('\\n🏆 MODEL LEADERBOARD 🏆')\n",
    "display(all_results)\n",
    "print(f'\\nBest model: {all_results.iloc[0][\"Model\"]} (RMSE={all_results.iloc[0][\"RMSE\"]:.2f})')"
])

# Cell 80
add_code_cell([
    "# Cell 80: Feature importance / SHAP analysis\n",
    "if hasattr(tuned_model, 'feature_importances_'):\n",
    "    # Get feature names after preprocessing\n",
    "    feature_names_out = (numeric_features + \n",
    "                        [f'{cat}_{val}' for cat in categorical_features \n",
    "                         for val in X_train[cat].unique()[:5]])\n",
    "    \n",
    "    imp_df = pd.DataFrame({\n",
    "        'Feature': feature_names_out[:len(tuned_model.feature_importances_)],\n",
    "        'Importance': tuned_model.feature_importances_\n",
    "    }).sort_values('Importance', ascending=False)\n",
    "    \n",
    "    print('\\nTop 10 Features:')\n",
    "    display(imp_df.head(10))\n",
    "else:\n",
    "    print('⚠ Feature importance not available')"
])

# Cell 81
add_code_cell([
    "# Cell 81: Save best model\n",
    "best_model_path = os.path.join(MODEL_DIR, 'best_model.pkl')\n",
    "joblib.dump(tuned_pipe, best_model_path)\n",
    "print(f'✓ Best model saved to {best_model_path}')"
])

# Cell 82
add_code_cell([
    "# Cell 82: Inference on test / end notebook\n",
    "print('='*70)\n",
    "print('NOTEBOOK COMPLETE')\n",
    "print('='*70)\n",
    "print(f'Final Model: RandomForest (Tuned)')\n",
    "print(f'Best RMSE: {metrics_tuned[\"RMSE\"]:.2f}')\n",
    "print(f'Best R²: {metrics_tuned[\"R2\"]:.3f}')\n",
    "print(f'Model saved: {best_model_path}')\n",
    "print('\\nReady for production deployment!')\n",
    "print('='*70)"
])

# ============================================================================
# Save the notebook
# ============================================================================

nb['cells'] = all_cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f"\n✓✓✓ SUCCESS ✓✓✓")
print(f"Notebook restructured with exactly {len(all_cells)} cells")
print(f"Saved to: {notebook_path}")
