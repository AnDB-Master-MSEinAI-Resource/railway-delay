import re

# Read the file
with open('notebooks/regression_pipeline_rmse.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# The line is JSON-encoded, so we need to match the exact format
old_line = '    "        \'class_imbalance\': f\\"1:{(1-df[\'IS_DELAYED\'].mean())/max(df[\'IS_DELAYED\'].mean(), 0.001):.2f}\\" if \'df\' in globals() else \'N/A\'\\n",'
new_line = '    "        \'target_distribution\': f\\"Mean: {df[\'DELAY_ARRIVAL\'].mean():.2f}, Std: {df[\'DELAY_ARRIVAL\'].std():.2f}\\" if \'df\' in globals() and \'DELAY_ARRIVAL\' in df.columns else \'N/A\'\\n",'

content = content.replace(old_line, new_line)

# Write back
with open('notebooks/regression_pipeline_rmse.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fix applied successfully')