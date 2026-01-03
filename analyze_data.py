import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('D:/MSE/5. Data Mining/railway-delay/data/processed/merged_train_data.csv', nrows=10000)

print("="*60)
print("DATA ANALYSIS REPORT")
print("="*60)

print(f"\nShape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for col in df.columns:
    print(f"  - {col}")

print("\nData Types:")
print(df.dtypes)

print("\nNull Counts:")
print(df.isnull().sum())

print("\nSample Statistics for Numeric Columns:")
print(df.describe())

print("\nTarget Variable (DELAY_ARRIVAL) Stats:")
if 'DELAY_ARRIVAL' in df.columns:
    print(f"  Mean: {df['DELAY_ARRIVAL'].mean():.2f}")
    print(f"  Std: {df['DELAY_ARRIVAL'].std():.2f}")
    print(f"  Min: {df['DELAY_ARRIVAL'].min():.2f}")
    print(f"  Max: {df['DELAY_ARRIVAL'].max():.2f}")
    print(f"  Median: {df['DELAY_ARRIVAL'].median():.2f}")

print("\nCategorical Columns Unique Values:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"  {col}: {df[col].nunique()} unique values")
