#!/usr/bin/env python3
"""
generate_daily_report.py
Generates daily activity reports based on `models/metrics_log.csv` and `models/metrics_summary.csv`.

Output:
- `reports/daily_report_YYYY-MM-DD.md` per date in log
- `reports/index.md` summarizing dates and top highlights

Usage:
    python reports/generate_daily_report.py
"""
import os
import json
from datetime import datetime
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, 'models')
REPORTS_DIR = os.path.join(ROOT, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

LOG_FILE = os.path.join(MODELS_DIR, 'metrics_log.csv')

def parse_metrics_log(path=LOG_FILE):
    if not os.path.exists(path):
        print(f'Metrics log not found: {path}')
        return pd.DataFrame()
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                timestamp, payload = line.split(',', 1)
                payload = json.loads(payload)
                metrics = payload.get('metrics', [])
                for m in metrics:
                    entry = dict(m)
                    entry['timestamp'] = pd.to_datetime(timestamp)
                    rows.append(entry)
            except Exception as e:
                print('Failed parse line:', e)
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
    return df

def make_daily_reports(df, out_dir=REPORTS_DIR):
    if df.empty:
        print('No metrics to report')
        return
    summaries = []
    for dt, g in df.groupby('date'):
        n = len(g)
        mean_rmse = g['rmse'].mean() if 'rmse' in g.columns else None
        best_idx = g['rmse'].idxmin() if 'rmse' in g.columns else None
        best_row = g.loc[best_idx].to_dict() if best_idx is not None else {}
        total_tuning_s = g['tuning_time_s'].sum() if 'tuning_time_s' in g.columns else 0
        baseline_rmse = None
        try:
            baseline_rmse = g.loc[g['model']=='Baseline Mean','rmse'].values[0]
        except Exception:
            baseline_rmse = None
        improved = 0
        if baseline_rmse is not None and 'rmse' in g.columns:
            improved = (g['rmse'] < baseline_rmse).sum()
            improvement_rate = improved / len(g)
        else:
            improvement_rate = None

        md_path = os.path.join(out_dir, f'daily_report_{dt}.md')
        with open(md_path, 'w', encoding='utf-8') as fh:
            fh.write(f'# Daily Report: {dt}\n\n')
            fh.write(f'- Experiments: {n}\n')
            fh.write(f'- Mean RMSE: {mean_rmse}\n')
            fh.write(f'- Best model: {best_row.get("model", "n/a")} (RMSE={best_row.get("rmse", "n/a")})\n')
            fh.write(f'- Total tuning time (hours): {total_tuning_s/3600:.2f}\n')
            if improvement_rate is not None:
                fh.write(f'- Improvement rate vs baseline: {improvement_rate:.2%}\n')
            fh.write('\n')
            fh.write('## Experiments summary\n')
            fh.write(g[['model','rmse','mae','r2','tuning_time_s']].sort_values('rmse').to_markdown(index=False))
        summaries.append({'date': dt, 'n': n, 'mean_rmse': mean_rmse, 'best_model': best_row.get('model', ''), 'total_tuning_hours': total_tuning_s/3600, 'report_path': md_path})
    # index
    index_path = os.path.join(out_dir, 'index.md')
    with open(index_path, 'w', encoding='utf-8') as fh:
        fh.write('# Daily Activity Reports\n\n')
        for s in sorted(summaries, key=lambda x: x['date']):
            fh.write(f"- {s['date']}: {s['n']} experiments, mean RMSE={s['mean_rmse']}, best={s['best_model']}, tuning hours={s['total_tuning_hours']:.2f} - [report]({os.path.relpath(s['report_path'], out_dir)})\n")
    print('Created reports in', out_dir)

def main():
    df = parse_metrics_log()
    if df.empty:
        print('No log entries parsed; nothing to generate')
        return
    make_daily_reports(df)

if __name__ == '__main__':
    main()
