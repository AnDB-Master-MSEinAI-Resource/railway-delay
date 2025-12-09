import os
import json
from datetime import datetime
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_LOG = os.path.join(ROOT, 'models', 'metrics_log.csv')
OUTPUT_CSV = os.path.join(ROOT, 'reports', 'daily_activity_report.csv')
OUTPUT_MD = os.path.join(ROOT, 'reports', 'daily_activity_report.md')

def parse_metrics_log(path=METRICS_LOG):
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
                # skip malformed lines
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
    return df

def build_daily_report(metrics_log_path=None, output_csv=OUTPUT_CSV, output_md=OUTPUT_MD):
    path = metrics_log_path or METRICS_LOG
    df = parse_metrics_log(path)
    if df.empty:
        print('No metrics found; skipping')
        return
    rows = []
    for dt, g in df.groupby('date'):
        n_runs = len(g)
        avg_rmse = g['rmse'].mean() if 'rmse' in g.columns else None
        best_rmse = g['rmse'].min() if 'rmse' in g.columns else None
        best_model = g.loc[g['rmse'].idxmin()]['model'] if 'rmse' in g.columns else ''
        tuning_hours = g['tuning_time_s'].sum() / 3600 if 'tuning_time_s' in g.columns else 0
        # Baseline improvement: check if baseline row exists per day
        baseline_rmse = None
        if 'model' in g.columns and 'Baseline Mean' in g['model'].values:
            baseline_rmse = g.loc[g['model']=='Baseline Mean', 'rmse'].values[0]
        improvement_rate = None
        if baseline_rmse is not None and 'rmse' in g.columns:
            improvement_rate = (g['rmse'] < baseline_rmse).sum() / len(g)
        rows.append({'date': dt, 'n_runs': n_runs, 'avg_rmse': avg_rmse, 'best_rmse': best_rmse, 'best_model': best_model, 'tuning_hours': tuning_hours, 'baseline_rmse': baseline_rmse, 'improvement': improvement_rate})

    out_df = pd.DataFrame(rows).sort_values('date')
    out_df.to_csv(output_csv, index=False)

    with open(output_md, 'w', encoding='utf-8') as fh:
        fh.write('# Daily Activity Report\n\n')
        for _, r in out_df.iterrows():
            fh.write(f"## {r['date']}\n")
            fh.write(f"- Experiments run: {r['n_runs']}\n")
            fh.write(f"- Avg RMSE: {r['avg_rmse']}\n")
            fh.write(f"- Best: {r['best_model']} (RMSE={r['best_rmse']})\n")
            fh.write(f"- Tuning hours: {r['tuning_hours']:.2f}\n")
            if pd.notna(r['improvement']):
                fh.write(f"- Improvement rate vs baseline: {r['improvement']:.2%}\n")
            fh.write('\n')
    print('Saved daily CSV to', output_csv)
    print('Saved MD report to', output_md)

if __name__ == '__main__':
    build_daily_report()
"""daily_report_generator.py

Generates a daily activity report using logs saved by the main notebook and git commit history.

This is an offline script to create a CSV/Markdown summary with per-day:
- number of experiments (metrics logged), best RMSE (and avg RMSE)
- number of commits and commit messages from git
- simple 'effort accuracy' metrics computed as day-over-day RMSE improvement and improvement/commit count

Usage: python reports/daily_report_generator.py
"""

import csv
import json
import os
import subprocess
import sys
from collections import defaultdict, OrderedDict
from datetime import datetime
from statistics import mean

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
TASKS_LOG = os.path.join(REPO_ROOT, 'reports', 'daily_task_log.csv')
METRICS_LOG = os.path.join(MODELS_DIR, 'metrics_log.csv')
OUTPUT_CSV = os.path.join(REPO_ROOT, 'reports', 'daily_activity_report.csv')
OUTPUT_MD = os.path.join(REPO_ROOT, 'reports', 'daily_activity_report.md')


def parse_metrics_log(path):
    """Load metrics_log.csv where each row = timestamp,entry (JSON) where entry contains metrics list"""
    if not os.path.exists(path):
        print(f"No metrics_log.csv found at {path}")
        return []
    entries = []
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        headers = next(reader, None)
        for row in reader:
            if not row:
                continue
            # expected format: timestamp,entry
            if len(row) < 2:
                continue
            try:
                ts = datetime.fromisoformat(row[0])
            except Exception:
                try:
                    ts = datetime.fromisoformat(row[0].strip())
                except Exception:
                    ts = None
            entry_str = row[1]
            try:
                entry = json.loads(entry_str)
            except Exception:
                try:
                    entry = json.loads(entry_str.replace("'", '"'))
                except Exception:
                    entry = entry_str
            entries.append({'timestamp': ts, 'entry': entry})
    return entries


def get_git_commits_by_day(since_days=None):
    """Get commit summary grouped by day. If since_days provided, limit to that many days."""
    cmd = ['git', 'log', '--pretty=format:%ci::%s']  # e.g., 2020-01-01 12:34:56 +0000::message
    if since_days is not None:
        cmd = ['git', 'log', f'--since={since_days}.days', '--pretty=format:%ci::%s']
    try:
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        out = res.stdout.strip()
    except Exception as e:
        print('git log failed:', e)
        return {}
    commits_by_day = defaultdict(list)
    if not out:
        return commits_by_day
    for line in out.split('\n'):
        if '::' in line:
            date_str, msg = line.split('::', 1)
            # date part is like: 2020-01-01 12:34:56 +0100
            try:
                date = datetime.fromisoformat(date_str.strip())
                day = date.date().isoformat()
                commits_by_day[day].append(msg.strip())
            except Exception:
                continue
    return commits_by_day


def summarize_by_day(metrics_entries):
    """Return dict date -> {'models': [...], 'rmses': [...], 'counts': int, 'best_rmse'}"""
    per_day = defaultdict(list)
    for rec in metrics_entries:
        ts = rec.get('timestamp')
        if ts is None:
            # try to extract from entry if included
            ts = datetime.now()
        day = ts.date().isoformat()
        per_day[day].append(rec['entry'])

    summary = OrderedDict()
    for day in sorted(per_day.keys()):
        entries = per_day[day]
        models = []
        rmses = []
        # entries may include a dict with 'metrics' key as list; otherwise inspect
        for e in entries:
            if isinstance(e, dict) and 'metrics' in e:
                # metrics is a list of metric dicts (model and rmse etc.) or single dict
                try:
                    if isinstance(e['metrics'], list):
                        for m in e['metrics']:
                            models.append(m.get('model', 'unknown'))
                            rm = m.get('rmse') or m.get('rmse_log', None) or m.get('rmse_original', None) or None
                            if rm is None:
                                # try known keys
                                rm = m.get('rmse', None)
                            if rm is not None:
                                rmses.append(float(rm))
                    elif isinstance(e['metrics'], dict):
                        m = e['metrics']
                        models.append(m.get('model', 'unknown'))
                        if 'rmse' in m:
                            rmses.append(float(m['rmse']))
                except Exception:
                    pass
            elif isinstance(e, dict) and 'rmse' in e:
                models.append(e.get('model', 'unknown'))
                rmses.append(float(e.get('rmse')))
        summary[day] = {
            'models_run': models,
            'n_runs': len(models),
            'rmses': rmses,
            'best_rmse': (min(rmses) if rmses else None),
            'avg_rmse': (mean(rmses) if rmses else None),
        }
    return summary


def build_daily_report(metrics_log_path, out_csv=OUTPUT_CSV, out_md=OUTPUT_MD):
    metrics_entries = parse_metrics_log(metrics_log_path)
    # read optional daily manual tasks log (user can fill this CSV with date, tasks, planned_hours, actual_hours)
    tasks_map = {}
    if os.path.exists(TASKS_LOG):
        try:
            with open(TASKS_LOG, newline='', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    d = row.get('date')
                    tasks_map[d] = row
        except Exception:
            tasks_map = {}
    commits_by_day = get_git_commits_by_day()
    summary = summarize_by_day(metrics_entries)

    # compute effort accuracy: day-over-day best_rmse improvements
    rows = []
    prev_best = None
    for day, val in summary.items():
        commits = commits_by_day.get(day, [])
        best_rmse = val['best_rmse']
        avg_rmse = val['avg_rmse']
        n_runs = val['n_runs']
        improvement = None
        improvement_pct = None
        per_commit_improvement = None
        if prev_best is not None and best_rmse is not None:
            improvement = max(prev_best - best_rmse, 0)
            if prev_best > 0:
                improvement_pct = improvement / prev_best * 100
            per_commit_improvement = improvement / max(1, len(commits))
        rows.append({
            'date': day,
            'commits_count': len(commits),
            'commits': ' | '.join(commits[:5]) if commits else '',
            'tasks': tasks_map.get(day, {}).get('tasks', '') if tasks_map else '',
            'planned_hours': tasks_map.get(day, {}).get('planned_hours', '') if tasks_map else '',
            'actual_hours': tasks_map.get(day, {}).get('actual_hours', '') if tasks_map else '',
            'n_runs': n_runs,
            'best_rmse': best_rmse,
            'avg_rmse': avg_rmse,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'per_commit_improvement': per_commit_improvement,
        })
        if best_rmse is not None:
            prev_best = best_rmse

    # write csv
    keys = ['date', 'commits_count', 'commits', 'tasks', 'planned_hours', 'actual_hours', 'n_runs', 'best_rmse', 'avg_rmse', 'improvement', 'improvement_pct', 'per_commit_improvement']
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # write markdown
    with open(out_md, 'w', encoding='utf-8') as fh:
        fh.write('# Daily Activity Report\n\n')
        fh.write(f'Generated: {datetime.now().isoformat()}\n\n')
        for r in rows:
            fh.write(f"## {r['date']}\n")
            fh.write(f"- Commits: {r['commits_count']}\n")
            if r['commits']:
                fh.write(f"  - messages: {r['commits']}\n")
            if r.get('tasks'):
                fh.write(f"- Tasks (manual log): {r.get('tasks')}\n")
            if r.get('planned_hours') or r.get('actual_hours'):
                fh.write(f"- Planned hours: {r.get('planned_hours')} - Actual hours: {r.get('actual_hours')}\n")
            fh.write(f"- Experiment runs: {r['n_runs']}\n")
            fh.write(f"- Best RMSE: {r['best_rmse']}\n")
            fh.write(f"- Avg RMSE: {r['avg_rmse']}\n")
            if r['improvement'] is not None:
                fh.write(f"- Improvement vs prev day: {r['improvement']} ({r['improvement_pct']:.2f}%){' per commit: ' + str(round(r['per_commit_improvement'], 6)) if r['per_commit_improvement'] is not None else ''}\n")
            fh.write('\n')
    print('Daily report written:', out_csv, out_md)


if __name__ == '__main__':
    build_daily_report(METRICS_LOG)
