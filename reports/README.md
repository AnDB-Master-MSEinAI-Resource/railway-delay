# Daily Activity Report Generator

This folder contains a small script and a notebook to create a daily activity report from existing experiment logs and git commit history.

Files included:
- `daily_report_generator.py`: script that reads `models/metrics_log.csv` and the repo git history and generates per-day reports (CSV and markdown).
- `daily_activity_report.csv`: generated daily report (if script run).
- A notebook `notebooks/daily_activity_report.ipynb` that provides an interactive display and exports the same CSV and markdown.

Usage (quick):
1. Ensure `models/metrics_log.csv` exists (created by running the main notebook saving logs to `models/metrics_log.csv`).
2. Run the generator script from the repository root:

```powershell
python scripts/generate_activity_report.py
```

Result: `reports/daily_activity_report.csv` and `reports/daily_activity_report.md` are generated.

Effort accuracy metric definition:
- "Best RMSE improvement" is used as a simple signal of progress.
- Day-over-day improvement and per-commit normalized improvement (improvement/commits) are shown as basic accuracy indicators of effort.

This is a simple tool meant to be a starting point; customize it to your processes and metrics.