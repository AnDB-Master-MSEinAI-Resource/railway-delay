# Daily Report Template

Use the script `scripts/generate_activity_report.py` or the notebook `notebooks/daily_activity_report.ipynb` to generate reports daily. The output CSV file will be saved to `reports/daily_activity/daily_report.csv`.

Columns produced:

- date: date of the log
- raw_timestamp: time the log was added
- best_rmse: best RMSE value found for that timestamp
- best_model: model name with the best RMSE
- tuning_time_s: tuning time (seconds) aggregated for that timestamp (if any)
- hours_spent: derived from tuning_time_s or manual log
- rmse_delta: change in best_rmse from previous entry
- effort_accuracy: rmse_delta / hours_spent

Add a manual daily log to `reports/daily_activity/daily_log_template.csv` to provide more context (hours_spent, work summary).