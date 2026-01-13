# Chapter 0 - Time Series Objects & Contracts (Python)

## Outcomes (what I can do after this)

- [ ] I can explain the Python equivalents of R ts / tsibble / timetk
- [ ] I can create a forecasting-ready DataFrame with columns unique_id, ds, y
- [ ] I can normalize timestamps to UTC and reason about DST edge cases
- [ ] I can validate time-series integrity before modeling
- [ ] I can add basic time-based features without leakage

## Concepts (plain English)

- **Datetime vs Timestamp**: Python `datetime` is the standard type; pandas wraps it as `Timestamp`
- **Timezone-aware vs naive**: naive timestamps have no timezone; always normalize to UTC
- **ts (R)**: a single series with a time index -> `pd.Series` with a `DatetimeIndex`
- **tsibble (R)**: tidy time-series table -> DataFrame with `unique_id, ds, y`
- **timetk (R)**: time-based feature helpers -> pandas `.dt`, `shift`, `rolling`, `expanding`
- **Contract**: invariants that every downstream stage assumes (sorted, unique, regular)

## R to Python mapping (where it fits)

- **ts** -> `pd.Series` with `DatetimeIndex` (single series; good for quick checks)
- **tsibble** -> long DataFrame `unique_id, ds, y` (StatsForecast and pipeline contract)
- **timetk** -> pandas feature engineering helpers + optional `statsmodels` diagnostics
- **Validation layer** -> `validate_time_series_integrity()` / `validate_time_index()`

## Architecture (what we're building)

### Inputs
- Single series or tidy table from any source (API, CSV, Parquet)
- Minimum fields: timestamps + values; plus series id for multi-series

### Outputs
- Forecasting-ready DataFrame with columns `[unique_id, ds, y]` (UTC, hourly)

### Invariants (must always hold)
- `ds` is normalized to UTC (timezone-naive in StatsForecast pipeline)
- No duplicate `(unique_id, ds)` pairs
- Regular hourly frequency with no gaps (unless explicitly allowed)
- Data sorted by `unique_id, ds`
- No leakage: features are built from past data only

### Failure modes
- Naive local timestamps (no timezone) -> DST ambiguity
- DST fall-back duplicates (same hour twice) -> duplicate pairs
- Missing hours -> gaps break backtesting
- Mixed frequencies across series -> invalid model input

## Files touched

- **`src/chapter0/objects.py`** - Chapter 0 helpers (ts, tsibble, timetk-style)
- **`src/chapter1/prepare.py`** - timezone parsing and normalization
- **`src/chapter1/eia_data_simple.py`** - `prepare_for_forecasting()`, `validate_time_series_integrity()`
- **`src/chapter1/validate.py`** - `validate_time_index()` and report printer
- **`src/chapter3/tasks.py`** - `compute_time_series_integrity()` gate in pipeline

## Step-by-step walkthrough

### 1) Create a single-series "ts" object
```python
import pandas as pd

idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
y = pd.Series([100, 102, 98, 101, 103, 99], index=idx)
print(type(y), y.index.tz)
```
- **Expect**: pandas Series with UTC-aware index

### 2) Convert to a "tsibble" style table
```python
df = y.reset_index()
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_localize(None)
df["unique_id"] = "NG_US48"
df = df[["unique_id", "ds", "y"]]
print(df.head())
```
- **Expect**: columns `[unique_id, ds, y]` with timezone-naive UTC timestamps

### 3) Validate the time-series contract
```python
from src.chapter1.validate import validate_time_index, print_validation_report

report = validate_time_index(df)
print_validation_report(report)
```
- **Expect**: PASS, no duplicates, no missing hours

### 4) Add timetk-style features (safe, leakage-free)
```python
df_features = df.assign(
    hour=df["ds"].dt.hour,
    dayofweek=df["ds"].dt.dayofweek,
    y_lag1=df["y"].shift(1),
    y_roll24=df["y"].rolling(24, min_periods=1).mean()
)
print(df_features.head())
```
- **Expect**: new columns; note the first row has NaNs for lagged values

## Metrics & success criteria

- `validate_time_index(df).is_valid` is True
- `df["ds"].dt.tz` is None (timezone-naive UTC)
- No duplicate pairs or missing hours

## Pitfalls (things that commonly break)

1. **Naive timestamps**: missing timezone info hides DST issues
2. **DST transitions**: fall-back creates duplicates, spring-forward creates gaps
3. **Wide format**: multiple value columns; StatsForecast expects long format
4. **Sorting**: unsorted `ds` breaks backtesting and feature alignment
5. **Leakage**: rolling/lag features must use past values only

## Mini-checkpoint (prove you learned it)

Answer these questions:

1. **What is the Python equivalent of an R `ts` object?**
2. **Why does the pipeline require `unique_id, ds, y` even for a single series?**
3. **What two DST problems does the integrity check catch?**
4. **Which functions enforce the data contract in this repo?**

**Answers:**
1. A `pd.Series` with a `DatetimeIndex`.
2. StatsForecast is multi-series-first; the contract stays consistent across one or many series.
3. Fall-back duplicates and spring-forward missing hours.
4. `validate_time_index()` and `validate_time_series_integrity()` / `compute_time_series_integrity()`.

## Exercises (optional, but recommended)

### Easy
1. Build a 48-hour series and verify `validate_time_index()` passes.
2. Introduce one duplicate timestamp and confirm the report fails.

### Medium
1. Create a series in `US/Eastern`, convert to UTC, and confirm `ds` is UTC.
2. Introduce a 3-hour gap and compute how many missing hours are reported.

### Hard
1. Write a helper that enforces the `[unique_id, ds, y]` contract and call it before backtesting.
2. Add a safe holiday or calendar feature using only `ds` (no future leakage).
3. Extend the validation to allow missing hours within a known maintenance window.
