# Chapter 5 - Data Quality and Preprocessing

## Outcomes (what I can do after this)

- [ ] I can detect missing timestamps, duplicates, and ordering issues
- [ ] I can standardize time zones and keep a clean unique_id/ds/y contract
- [ ] I can repair gaps without leaking future information
- [ ] I can explain why integrity gates happen before modeling

## Prerequisite (read first)

- Chapter 0 for the time-series contract (unique_id, ds, y)
- Chapter 1 for ingestion and UTC normalization

## Concepts (plain English)

- **Missing timestamps**: Gaps in the expected frequency (hourly, daily, etc.)
- **Duplicates**: Multiple rows with the same (unique_id, ds)
- **Integrity gate**: A hard check that stops the pipeline when data is invalid
- **Resampling**: Creating a regular time index and aligning data to it
- **DST risk**: Local timestamps can repeat or skip hours

## Architecture (what we are building)

### Inputs
- Time series data with columns [unique_id, ds, y]
- Expected frequency (hourly for the EIA example)

### Outputs
- Validation report with duplicate and missing counts
- Cleaned data ready for forecasting

### Invariants (must always hold)
- unique_id and ds identify a single row
- ds is timezone-naive UTC
- ds is monotonically increasing

### Failure modes
- DST creates duplicates or missing hours
- API returns unsorted or partial data
- Local time zone slips into modeling steps

## Files touched

- `src/chapter1/validate.py` - validate_time_index
- `src/chapter1/prepare.py` - normalize time to UTC
- `src/chapter3/tasks.py` - compute_time_series_integrity

## Step-by-step walkthrough

### 1) Create a small series with issues
```python
import pandas as pd
import numpy as np

from src.chapter1.validate import validate_time_index

ds = pd.date_range("2024-01-01", periods=48, freq="H")
df = pd.DataFrame({"unique_id": "series_1", "ds": ds, "y": np.arange(len(ds))})

# Remove one hour and duplicate another
broken = df.drop(index=[10]).reset_index(drop=True)
broken = pd.concat([broken, broken.iloc[[20]]], ignore_index=True)

result = validate_time_index(broken)
print(result)
```
- Expect: duplicates > 0 and missing hours > 0

### 2) Repair duplicates and align to the expected index
```python
fixed = broken.drop_duplicates(subset=["unique_id", "ds"]).sort_values("ds")
full_index = pd.date_range(fixed["ds"].min(), fixed["ds"].max(), freq="H")
fixed = fixed.set_index("ds").reindex(full_index).rename_axis("ds").reset_index()
print(fixed.isna().sum())
```
- Expect: y has NaN where the gap was

### 3) Decide a missing-value policy
```python
# Example: forward fill for short gaps only
fixed["y"] = fixed["y"].ffill(limit=1)
print(fixed.isna().sum())
```
- Expect: short gaps filled, long gaps remain NaN

## Metrics & success criteria

- 0 duplicates on (unique_id, ds)
- Missing hours are known and explained
- ds is UTC and sorted

## Pitfalls (things that commonly break)

1. Filling gaps with future values (leakage)
2. Mixing local time and UTC in the same column
3. Ignoring duplicates created by DST

## Mini-checkpoint (prove you learned it)

1. Why must ds be timezone-naive UTC before modeling?
2. What is the difference between missing hours and duplicates?

**Answers:**
1. StatsForecast and backtesting assume a clean, monotonic UTC index.
2. Missing hours are gaps in the expected frequency; duplicates are repeated timestamps.
