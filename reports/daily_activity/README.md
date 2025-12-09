# Daily Activity Reports

This folder contains daily activity reports and templates generated from the notebooks and scripts. The reports are designed to summarize:

- The experiments added to `models/metrics_log.csv` (time-stamped entries), including RMSE and other metrics.
- A daily log (you can add your manual daily notes in `daily_log_template.csv`).
- An "effort accuracy" metric which indicates improvement in model quality (like change in RMSE) per unit time (hours) spent.

Files:
- `daily_report.csv`: generated daily summary CSV that consolidates metrics and daily logs.
- `daily_log_template.csv`: a CSV template to capture daily manual notes and the hours you worked.

Usage:
- Use the notebook `notebooks/daily_activity_report.ipynb` or the script `scripts/generate_activity_report.py` to run the report generation.

Notes:
- If your `models/metrics_log.csv` is very large, the notebook and scripts read it in chunks to avoid memory issues.
- The report and accuracy calculation are examples and can be adjusted to fit your working definitions of effort and accuracy.
