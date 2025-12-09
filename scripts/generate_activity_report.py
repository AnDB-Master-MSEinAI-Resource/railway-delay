#!/usr/bin/env python3
"""
Generate a daily activity report from models/metrics_log.csv and optional manual daily log.
Produces CSV summary and plots saved in reports/daily_activity.
"""
import os
import json
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports', 'daily_activity')

METRICS_LOG = os.path.join(MODEL_DIR, 'metrics_log.csv')
MANUAL_LOG = os.path.join(REPORT_DIR, 'daily_log_template.csv')
OUT_CSV = os.path.join(REPORT_DIR, 'daily_report.csv')


def parse_metrics_log(path):
    """Read metrics_log.csv which is expected to have 'timestamp,entry' rows where 'entry' is a JSON.
    We'll parse them and return a DataFrame.
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    # read in chunks to avoid MemoryError
    it = pd.read_csv(path, chunksize=1000, dtype=str)
    for chunk in it:
        for _, r in chunk.iterrows():
            try:
                ts = r['timestamp'] if 'timestamp' in r else None
                entry = r['entry'] if 'entry' in r else r.get('metrics') or r.get('metrics')
                if ts is None and 'date' in r:
                    ts = r['date']
                if entry is None:
                    continue
                obj = json.loads(entry) if isinstance(entry, str) else entry
                row = {'raw_timestamp': ts}
                # expand top-level fields
                for k, v in obj.items():
                    if k == 'metrics' and isinstance(v, list):
                        # if metrics is a list of models, take each or aggregate
                        # we'll gather best metrics (lowest rmse) for that timestamp
                        try:
                            metrics_df = pd.DataFrame(v)
                            row['best_rmse'] = metrics_df['rmse'].min() if 'rmse' in metrics_df.columns else np.nan
                            row['best_model'] = metrics_df.loc[metrics_df['rmse'].idxmin()]['model'] if 'rmse' in metrics_df.columns else None
                            # sum tuning_time if present
                            row['tuning_time_s'] = metrics_df['tuning_time_s'].sum() if 'tuning_time_s' in metrics_df.columns else np.nan
                        except Exception:
                            pass
                    else:
                        row[k] = v
                rows.append(row)
            except Exception:
                continue
    if len(rows) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # parse timestamp
    if 'raw_timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['raw_timestamp'], errors='coerce')
    else:
        df['timestamp'] = pd.NaT
    df['date'] = pd.to_datetime(df['timestamp'].dt.date)
    return df


def compute_effort_accuracy(df):
    """Compute an example effort accuracy metric: RMSE improvement per hour.
    Requires `best_rmse` and `tuning_time_s` (or `hours_spent` from manual log).
    """
    if df.empty:
        return df
    # Sort by timestamp
    df = df.sort_values('timestamp')
    # Compute previous best_rmse
    df['prev_best_rmse'] = df['best_rmse'].shift(1)
    df['rmse_delta'] = df['prev_best_rmse'] - df['best_rmse']
    # hours spent fallback: use tuning_time_s if present, convert to hours
    df['hours_spent'] = df.get('tuning_time_s', np.nan) / 3600.0
    # for empty hours_spent, set to 0 to avoid div by zero; will fill NAs later
    df['effort_accuracy'] = df['rmse_delta'] / (df['hours_spent'].replace(0, np.nan))
    return df


def generate_report(output_csv=OUT_CSV):
    os.makedirs(REPORT_DIR, exist_ok=True)
    df_metrics = parse_metrics_log(METRICS_LOG)
    if df_metrics.empty:
        print('No metrics log found or empty metrics log. Nothing to summarize.')
        return None
    df_metrics = compute_effort_accuracy(df_metrics)
    # merge manual daily logs if present
    if os.path.exists(MANUAL_LOG):
        df_manual = pd.read_csv(MANUAL_LOG, parse_dates=['date'])
        # aggregate manual time per day
        df_manual_agg = df_manual.groupby(df_manual['date'].dt.date).agg({
            'hours_spent': 'sum', 'work_summary': lambda x: ' | '.join(x), 'notes': lambda x: ' | '.join(x)
        }).reset_index().rename(columns={'date':'date_manual'})
        # convert date to datetime
        df_manual_agg['date'] = pd.to_datetime(df_manual_agg['date_manual'])
        # merge on date
        df_out = pd.merge(df_metrics, df_manual_agg[['date','hours_spent','work_summary','notes']], on='date', how='left', suffixes=('','_manual'))
    else:
        df_out = df_metrics
    df_out.to_csv(output_csv, index=False)
    print('Saved daily report to', output_csv)
    return df_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate daily activity report')
    parser.add_argument('--out', default=OUT_CSV, help='Output CSV path')
    args = parser.parse_args()
    df = generate_report(args.out)
    if df is not None:
        print('Daily report generated with rows:', len(df))
