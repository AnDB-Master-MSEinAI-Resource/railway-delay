# Data Directory

## 📁 Structure

- **raw/**: Original datasets (never modify these files)
- **interim/**: Intermediate data transformations and experiments
- **processed/**: Final clean datasets ready for modeling

## 📊 Files Description

### Raw Data (`raw/`)

**`railway-delay-dataset.csv`**
- Original railway delay dataset
- **Size**: ~5.8 million records
- **Columns**: 31 features including temporal, operational, and delay information
- **Source**: Railway operations data
- **Note**: This file should NEVER be modified. Keep it as the single source of truth.

### Processed Data (`processed/`)

**`train_data.csv`**
- Training dataset (80% of original data)
- **Records**: ~4.66 million
- **Purpose**: Model training and validation
- **Preprocessing**: Basic cleaning applied

**`test_data.csv`**
- Test dataset (20% of original data)
- **Records**: ~1.16 million
- **Purpose**: Final model evaluation
- **Preprocessing**: Same transformations as training data

**`merged_train_data.csv`**
- Combined training data (clean + dirty samples)
- **Purpose**: Training models with real-world data quality issues
- **Note**: Contains both original clean data and intentionally corrupted data

### Interim Data (`interim/`)

**`dirty_train_data.csv`**
- Training data with intentionally injected data quality issues
- **Purpose**: Data cleaning practice and robustness testing
- **Issues included**:
  - Missing values (NULL, empty cells, whitespace)
  - Wrong data types
  - Outliers
  - Impossible negative values
  - Out-of-range values
  - Duplicate timestamps
  - Incorrect formats
  - Unsorted time series
  - Sensor drift and dropout
  - Contradictory data

## 🔄 Data Flow

```
railway-delay-dataset.csv (raw)
         ↓
    [Split 80/20]
         ↓
    ┌────┴────┐
    ↓         ↓
train_data   test_data
(processed)  (processed)
    ↓
[Add errors]
    ↓
dirty_train_data
  (interim)
    ↓
 [Merge]
    ↓
merged_train_data
  (processed)
```

## 📋 Data Schema

### Key Features

**Temporal Features:**
- `YEAR`, `MONTH`, `DAY`, `DAY_OF_WEEK`
- `SCHEDULED_DEPARTURE`, `ACTUAL_DEPARTURE`
- `SCHEDULED_ARRIVAL`, `ACTUAL_ARRIVAL`

**Operational Features:**
- `TRAIN_OPERATOR`: Airline/operator code
- `TRAIN_NUMBER`: Flight number
- `SOURCE_STATION`, `DESTINATION_STATION`
- `DISTANCE_KM`: Route distance
- `RUN_TIME`, `ELAPSED_TIME`

**Delay Features:**
- `DELAY_DEPARTURE`: Departure delay in minutes
- `DELAY_ARRIVAL`: Arrival delay in minutes
- `SYSTEM_DELAY`, `SECURITY_DELAY`
- `TRAIN_OPERATOR_DELAY`, `LATE_TRAIN_DELAY`
- `WEATHER_DELAY`

**Status Features:**
- `DIVERTED`: Route diverted (0/1)
- `CANCELLED`: Flight cancelled (0/1)
- `CANCELLATION_REASON`

## 🔒 Data Handling Best Practices

1. **Never modify raw data**: Always work with copies in processed/
2. **Document transformations**: Keep track of all preprocessing steps
3. **Version control**: Use Git for data pipeline scripts, not large CSV files
4. **Backup important files**: Keep backups of processed datasets
5. **Use consistent naming**: Follow the established naming conventions

## 📝 Notes

- Large files (>50MB) are not included in Git version control
- Add data files to `.gitignore` to prevent accidental commits
- Use relative paths in code for portability
- Document any manual data corrections

## 🔗 Related Documentation

- See `../docs/data_schema.md` for detailed field descriptions
- See `../notebooks/` for data exploration and analysis
- See `../src/data/` for data processing scripts

---

**Last Updated**: November 2025
