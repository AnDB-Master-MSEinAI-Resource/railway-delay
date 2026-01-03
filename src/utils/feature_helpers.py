import pandas as pd
import numpy as np


def _get_route_column(df, candidates=None):
    if candidates is None:
        candidates = ['STATION_ID', 'ROUTE_ID', 'TRAIN_ID', 'TRAIN_NUMBER', 'STATION', 'ROUTE']
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_prev_delay_safe(df_in, target_col='TARGET', schedule_col='SCHEDULED_DT', candidates=None, default_fill=-1):
    df = df_in.copy()
    if candidates is None:
        candidates = ['TRAIN_ID','STATION_ID','ROUTE_ID','TRAIN_NUMBER','ROUTE','STATION']

    def _get_route_local(df):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    route_local = _get_route_local(df)

    if schedule_col not in df.columns:
        df['PREV_DELAY'] = default_fill
        return df

    df[schedule_col] = pd.to_datetime(df[schedule_col], errors='coerce')

    if route_local is not None and route_local in df.columns:
        df = df.sort_values([route_local, schedule_col])
        df['PREV_DELAY'] = df.groupby(route_local)[target_col].shift(1)
    else:
        df = df.sort_values(schedule_col)
        df['PREV_DELAY'] = df[target_col].shift(1)

    try:
        if route_local is not None and route_local in df.columns:
            df['PREV_DELAY'] = df['PREV_DELAY'].fillna(df.groupby(route_local)[target_col].transform('median'))
        else:
            df['PREV_DELAY'] = df['PREV_DELAY'].fillna(df[target_col].median() if target_col in df.columns else default_fill)
    except Exception:
        df['PREV_DELAY'] = df['PREV_DELAY'].fillna(default_fill)

    df['PREV_DELAY'] = pd.to_numeric(df['PREV_DELAY'], errors='coerce').fillna(default_fill)
    return df


def compute_rolling_features_safe(df_in, target_col='TARGET', schedule_col='SCHEDULED_DT', candidates=None, w7='7D', w30='30D'):
    df = df_in.copy()
    if candidates is None:
        candidates = ['STATION_ID', 'ROUTE_ID', 'TRAIN_ID', 'TRAIN_NUMBER', 'ROUTE', 'STATION']

    def _get_route_col_local(df):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    route_col_local = _get_route_col_local(df)

    if schedule_col not in df.columns:
        df['ROLLING_MEAN_DELAY_7D'] = df[target_col].median() if target_col in df.columns else np.nan
        df['ROLLING_MEAN_DELAY_30D'] = df[target_col].median() if target_col in df.columns else np.nan
        return df

    df[schedule_col] = pd.to_datetime(df[schedule_col], errors='coerce')

    # Drop rows with NaT in schedule_col for rolling computation
    valid_mask = df[schedule_col].notna()
    df_valid = df[valid_mask].copy()
    df_invalid = df[~valid_mask].copy()

    if len(df_valid) == 0:
        df['ROLLING_MEAN_DELAY_7D'] = df[target_col].median() if target_col in df.columns else np.nan
        df['ROLLING_MEAN_DELAY_30D'] = df[target_col].median() if target_col in df.columns else np.nan
        return df

    if route_col_local is not None and route_col_local in df_valid.columns:
        df_valid = df_valid.sort_values([route_col_local, schedule_col])
        tmp = df_valid.set_index(schedule_col)
        tmp['ROLLING_MEAN_DELAY_7D'] = tmp.groupby(route_col_local)[target_col].transform(lambda x: x.rolling(w7).mean())
        tmp['ROLLING_MEAN_DELAY_30D'] = tmp.groupby(route_col_local)[target_col].transform(lambda x: x.rolling(w30).mean())
        df_valid['ROLLING_MEAN_DELAY_7D'] = tmp['ROLLING_MEAN_DELAY_7D'].values
        df_valid['ROLLING_MEAN_DELAY_30D'] = tmp['ROLLING_MEAN_DELAY_30D'].values
    else:
        df_valid = df_valid.sort_values(schedule_col)
        idxed = df_valid.set_index(schedule_col)
        idxed['ROLLING_MEAN_DELAY_7D'] = idxed[target_col].rolling(w7).mean()
        idxed['ROLLING_MEAN_DELAY_30D'] = idxed[target_col].rolling(w30).mean()
        df_valid['ROLLING_MEAN_DELAY_7D'] = idxed['ROLLING_MEAN_DELAY_7D'].values
        df_valid['ROLLING_MEAN_DELAY_30D'] = idxed['ROLLING_MEAN_DELAY_30D'].values

    # Fill NaN values in valid data
    try:
        if route_col_local is not None and route_col_local in df_valid.columns:
            df_valid['ROLLING_MEAN_DELAY_7D'] = df_valid['ROLLING_MEAN_DELAY_7D'].fillna(df_valid.groupby(route_col_local)[target_col].transform('median'))
            df_valid['ROLLING_MEAN_DELAY_30D'] = df_valid['ROLLING_MEAN_DELAY_30D'].fillna(df_valid.groupby(route_col_local)[target_col].transform('median'))
        else:
            df_valid['ROLLING_MEAN_DELAY_7D'] = df_valid['ROLLING_MEAN_DELAY_7D'].fillna(df_valid[target_col].median() if target_col in df_valid.columns else np.nan)
            df_valid['ROLLING_MEAN_DELAY_30D'] = df_valid['ROLLING_MEAN_DELAY_30D'].fillna(df_valid[target_col].median() if target_col in df_valid.columns else np.nan)
    except Exception:
        df_valid['ROLLING_MEAN_DELAY_7D'] = df_valid['ROLLING_MEAN_DELAY_7D'].fillna(df_valid[target_col].median() if target_col in df_valid.columns else np.nan)
        df_valid['ROLLING_MEAN_DELAY_30D'] = df_valid['ROLLING_MEAN_DELAY_30D'].fillna(df_valid[target_col].median() if target_col in df_valid.columns else np.nan)

    # For invalid rows, fill with median
    median_7d = df_valid[target_col].median() if target_col in df_valid.columns else np.nan
    median_30d = df_valid[target_col].median() if target_col in df_valid.columns else np.nan
    df_invalid['ROLLING_MEAN_DELAY_7D'] = median_7d
    df_invalid['ROLLING_MEAN_DELAY_30D'] = median_30d

    # Combine back
    df_combined = pd.concat([df_valid, df_invalid], ignore_index=True)
    return df_combined
