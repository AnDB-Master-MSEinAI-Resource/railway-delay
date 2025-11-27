import pandas as pd
import numpy as np
import random
import os

input_file = 'train_data.csv'
output_file = 'dirty_train_data.csv'

print(f"Loading {input_file}...")
df = pd.read_csv(input_file, low_memory=False)
print(f"Loaded {len(df)} rows.")

# Helper to get random indices
def get_random_indices(count):
    return np.random.choice(df.index, count, replace=False)

# 1. Missing Values
print("Injecting Missing Values...")
idxs = get_random_indices(1000)
df.loc[idxs, 'DELAY_DEPARTURE'] = np.nan
idxs = get_random_indices(1000)
df.loc[idxs, 'TRAIN_OPERATOR'] = ""
idxs = get_random_indices(1000)
df.loc[idxs, 'SOURCE_STATION'] = " "
idxs = get_random_indices(1000)
df.loc[idxs, 'DESTINATION_STATION'] = "N/A"

# 2. Wrong Data Types
print("Injecting Wrong Data Types...")
# Convert columns to object to allow mixed types without error/warning
df['DELAY_ARRIVAL'] = df['DELAY_ARRIVAL'].astype(object)
idxs = get_random_indices(500)
df.loc[idxs, 'DELAY_ARRIVAL'] = "abc"

df['TRAIN_OPERATOR'] = df['TRAIN_OPERATOR'].astype(object)
idxs = get_random_indices(500)
df.loc[idxs, 'TRAIN_OPERATOR'] = 12345

df['CANCELLED'] = df['CANCELLED'].astype(object)
idxs = get_random_indices(500)
df.loc[idxs, 'CANCELLED'] = "yes" 
idxs = get_random_indices(500)
df.loc[idxs, 'CANCELLED'] = "no"

# 3. Outliers
print("Injecting Outliers...")
idxs = get_random_indices(200)
df.loc[idxs, 'DELAY_DEPARTURE'] = 99999
idxs = get_random_indices(200)
df.loc[idxs, 'DELAY_DEPARTURE'] = -999

# 4. Impossible Negative Values
print("Injecting Impossible Negative Values...")
idxs = get_random_indices(200)
df.loc[idxs, 'DISTANCE_KM'] = -100
idxs = get_random_indices(200)
df.loc[idxs, 'RUN_TIME'] = -50

# 5. Out-of-Range Values
print("Injecting Out-of-Range Values...")
idxs = get_random_indices(200)
df.loc[idxs, 'MONTH'] = 13
idxs = get_random_indices(200)
df.loc[idxs, 'DAY'] = 32

# 7. Incorrect Timestamp Format
print("Injecting Incorrect Timestamp Formats...")
df['SCHEDULED_DEPARTURE'] = df['SCHEDULED_DEPARTURE'].astype(object)
idxs = get_random_indices(200)
df.loc[idxs, 'SCHEDULED_DEPARTURE'] = "25:00"
idxs = get_random_indices(200)
df.loc[idxs, 'SCHEDULED_DEPARTURE'] = "2025-15-01"

# 8. Unsorted Time Series
print("Unsorting Data...")
df = df.sample(frac=1).reset_index(drop=True)

# 9. Incorrect Decimal Formatting
print("Injecting Incorrect Decimal Formatting...")
df['DISTANCE_KM'] = df['DISTANCE_KM'].astype(object)
idxs = get_random_indices(500)
# Use a lambda to avoid applying to NaNs if any
df.loc[idxs, 'DISTANCE_KM'] = df.loc[idxs, 'DISTANCE_KM'].apply(lambda x: str(x).replace('.', ',') if pd.notnull(x) else x)

# 10. String Noise
print("Injecting String Noise...")
idxs = get_random_indices(500)
df.loc[idxs, 'TRAIN_OPERATOR'] = df.loc[idxs, 'TRAIN_OPERATOR'].astype(str) + " "
idxs = get_random_indices(500)
df.loc[idxs, 'TRAIN_OPERATOR'] = df.loc[idxs, 'TRAIN_OPERATOR'].astype(str) + "#"

# 11. Sensor Drift
print("Injecting Sensor Drift...")
start_idx = random.randint(0, len(df) - 1000)
drift = np.linspace(0, 50, 1000)
# Ensure we are working with numeric types for addition, might fail if we already injected strings
# So we'll just overwrite
df.iloc[start_idx:start_idx+1000, df.columns.get_loc('DELAY_DEPARTURE')] = drift

# 12. Sensor Dropout
print("Injecting Sensor Dropout...")
start_idx = random.randint(0, len(df) - 1000)
df.iloc[start_idx:start_idx+1000, df.columns.get_loc('DELAY_DEPARTURE')] = np.nan

# 13. Random Glitches
print("Injecting Random Glitches...")
idxs = get_random_indices(100)
df.loc[idxs, 'DELAY_DEPARTURE'] = np.random.randint(1000, 5000, size=100)

# 14. Impossible Biological/Physical Behavior
print("Injecting Impossible Behavior...")
idxs = get_random_indices(500)
df.loc[idxs, 'ACTUAL_ARRIVAL'] = 100
df.loc[idxs, 'ACTUAL_DEPARTURE'] = 2300

# 15. Contradictory Data
print("Injecting Contradictory Data...")
idxs = get_random_indices(500)
df.loc[idxs, 'CANCELLED'] = 1
df.loc[idxs, 'DELAY_ARRIVAL'] = 0

# 16. Incorrect Derived Values
print("Injecting Incorrect Derived Values...")
idxs = get_random_indices(500)
df.loc[idxs, 'DELAY_ARRIVAL'] = 10000

print(f"Saving main data to {output_file}...")
df.to_csv(output_file, index=False)

# 17. Duplicate Rows
print("Injecting Duplicate Rows...")
# Read back a small chunk or just use sample from memory if it's still there
duplicates = df.sample(n=1000)
duplicates.to_csv(output_file, mode='a', index=False, header=False)

# 18. Irrelevant or External Data
print("Injecting Irrelevant Data...")
irrelevant_row = pd.DataFrame([{col: 'IRRELEVANT' for col in df.columns}])
irrelevant_row.to_csv(output_file, mode='a', index=False, header=False)

print("Done!")
