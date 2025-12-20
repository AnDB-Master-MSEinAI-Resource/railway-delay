# Check if the replacement worked
with open('notebooks/regression_pipeline_rmse.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

old_line = "        'class_imbalance': f\"1:{(1-df['IS_DELAYED'].mean())/max(df['IS_DELAYED'].mean(), 0.001):.2f}\" if 'df' in globals() else 'N/A'\n"
print('Old line exists:', old_line in content)

new_line = "        'target_distribution': f\"Mean: {df['DELAY_ARRIVAL'].mean():.2f}, Std: {df['DELAY_ARRIVAL'].std():.2f}\" if 'df' in globals() and 'DELAY_ARRIVAL' in df.columns else 'N/A'\n"
print('New line exists:', new_line in content)

# Find the line with IS_DELAYED
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'IS_DELAYED' in line and 'class_imbalance' in line:
        print(f'Found at line {i+1}: {line}')