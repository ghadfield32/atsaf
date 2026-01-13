# Chapter 1 — Data Ingestion & Preparation

## Outcomes (what I can do after this)

- [ ] I can pull raw EIA electricity data via REST API with proper pagination and error handling
- [ ] I can validate time-series data for duplicates, missing hours, and DST edge cases
- [ ] I can transform raw data into forecasting-ready format (unique_id, ds, y)
- [ ] I can explain why UTC normalization and data sorting matter for backtesting

## Prerequisite (read first)

- Chapter 0 for the time-series contract (unique_id, ds, y) and UTC rules

## Concepts (plain English)

- **API Pagination**: Splitting large datasets into fixed-size chunks to avoid timeouts
- **Datetime Normalization**: Converting all timestamps to UTC and handling DST transitions
- **Time-series Integrity**: Detecting duplicates, missing hours, and repeated timestamps (DST fall-back)
- **Monotonicity**: Ensuring timestamps are sorted chronologically (required for backtesting)
- **Data Schema**: unique_id (series identifier), ds (datetime), y (numeric value); tsibble analog in Python

## Architecture (what we're building)

### Inputs
- EIA API credentials (via environment variable `EIA_API_KEY`)
- Data range (start_date, end_date as YYYY-MM-DD strings)
- Series identifier (respondent, fueltype)

### Outputs
- **raw.parquet**: Unmodified API response (columns: time, value, respondent, fueltype)
- **clean.parquet**: Normalized data (columns: unique_id, ds, y, with UTC timestamps)
- **metadata.json**: Data snapshot (row count, date range, integrity report)

### Invariants (must always hold)
- Timestamps must be monotonically increasing (no gaps, no duplicates, no reversals)
- All values must be numeric and non-negative (electricity generation)
- All timestamps must be in UTC (no local timezones)
- No data loss: raw.parquet row count ≥ clean.parquet row count

### Failure modes
- API unavailable → retries up to 3 times, then raises RuntimeError
- DST transition creates duplicate hour → integrity check catches it, raises ValueError
- Missing hours in sequence → detected and reported, task fails if threshold exceeded
- Non-numeric values → converted during prepare, logged as NaN if conversion fails

## Files touched

- **`src/chapter1/eia_data_simple.py`** (1,773 lines) — Main orchestrator (see Methods section)
  - **`EIADataFetcher`** class: Pull, prepare, validate, format for forecasting
  - **`ExperimentConfig`**: Defines horizon, windows, models for downstream chapters
- **`src/chapter1/ingest.py`** — Paginated API calls with stable sort
- **`src/chapter1/prepare.py`** — Datetime parsing and numeric conversion
- **`src/chapter1/validate.py`** — Time-series integrity checks (duplicates, missing hours, DST)
- **`src/chapter1/config.py`** — Settings (API key, date ranges, respondent/fuel types)

## Step-by-step walkthrough

### 1) Initialize the fetcher
```python
from src.chapter1.eia_data_simple import EIADataFetcher
import os

api_key = os.getenv("EIA_API_KEY")
fetcher = EIADataFetcher(api_key)
```
- **Expect**: No errors. Fetcher is ready to pull data.
- **If it fails**: Check that `.env` file exists and contains `EIA_API_KEY=<your_key>`

### 2) Pull raw data
```python
df_raw = fetcher.pull_data(
    start_date="2023-06-01",
    end_date="2023-06-30",
    respondent="NG_US48",
    fueltype="NG"
)
print(f"Raw rows: {len(df_raw)}, Columns: {df_raw.columns.tolist()}")
```
- **Expect**:
  - ~720 rows (30 days × 24 hours)
  - Columns: `['time', 'value', 'respondent', 'fueltype']`
  - time is string (e.g., "2023-06-01T00:00:00-04:00")
  - value is string (e.g., "1234.56")
- **If it fails**:
  - "API Error": Check API credentials and network connectivity
  - "Empty response": Check date range and respondent/fueltype exist in EIA

### 3) Prepare (normalize) data
```python
df_prepared = fetcher.prepare_data(df_raw)
print(df_prepared.head())
print(f"Data types: {df_prepared.dtypes.to_dict()}")
```
- **Expect**:
  - Columns: `['time', 'value', 'respondent', 'fueltype']`
  - `time` is `datetime64[ns, UTC]` (pandas Timestamp)
  - `value` is `float64`
  - No NaN values
- **If it fails**:
  - Conversion error: Check for non-numeric values in API response (may indicate API schema change)

### 4) Validate time-series integrity
```python
is_valid = fetcher.validate_data(df_prepared)
print(f"Basic validation: {is_valid}")

integrity = fetcher.validate_time_series_integrity(df_prepared, unique_id="respondent")
print(f"Integrity status: {integrity['status']}")
print(f"Duplicate pairs: {integrity.get('duplicate_count', 0)}")
print(f"Missing hours: {integrity.get('missing_hours_total', 0)}")
```
- **Expect**:
  - `is_valid` = True
  - `integrity['status']` = "valid"
  - Duplicate and missing hour counts = 0 (or acceptable based on thresholds)
- **If it fails**:
  - "Duplicates detected": DST fall-back (1 hour repeats). Check integrity['duplicate_pairs']
  - "Missing hours": Gap in data (e.g., server downtime). Check `integrity['missing_gaps_count']` and `longest_gap_hours`

### 5) Format for forecasting
```python
df_forecast = fetcher.prepare_for_forecasting(df_prepared, unique_id="respondent")
print(df_forecast.columns.tolist())
print(df_forecast.head())
```
- **Expect**:
  - Columns: `['unique_id', 'ds', 'y']`
  - `unique_id` = "NG_US48" (constant)
  - `ds` = datetime (UTC, hourly, no gaps)
  - `y` = numeric value (energy generation)
- **If it fails**: Time-series integrity failed (duplicates, gaps). Re-run step 4 to diagnose.

### 6) End-to-end test
```python
df_full = fetcher.full_pipeline(
    start_date="2023-06-01",
    end_date="2023-06-30",
    unique_id_col="respondent"
)
print(f"Output rows: {len(df_full)}, Columns: {df_full.columns.tolist()}")
```
- **Expect**: Same as step 5 (columns: unique_id, ds, y)
- **If it fails**: Check logs for which step failed (pull, prepare, validate, or format)

## Metrics & success criteria

### Primary metric
- **Integrity status**: All records pass validation (no duplicates, no missing hours)

### Secondary metrics
- **Data freshness**: Last record within 24 hours of run time
- **Row count consistency**: Raw ≥ Clean (loss should be minimal, <5%)
- **Timezone correctness**: All timestamps are UTC

### "Good enough" threshold
- No duplicates (count = 0)
- No missing hours within the date range
- All values numeric and ≥ 0

### What would make me re-ingest / re-validate
- API returns unexpected schema (missing columns, wrong data type)
- >5% row loss during prepare (indicates conversion failures)
- New duplicates appear (suggests API bug or DST change)

## Pitfalls (things that commonly break)

1. **Timezone confusion**:
   - API returns local time with offset (e.g., "-04:00"), but we convert to UTC
   - If you skip UTC normalization, backtesting will be incorrect

2. **DST transitions** (US Eastern Time, US Central Time):
   - Fall-back (Nov, 1-2 AM repeats) → creates duplicate rows → integrity check catches it
   - Spring-forward (Mar, 2-3 AM is skipped) → missing row → detected as gap
   - Always run integrity check; don't assume API data is DST-clean

3. **Empty or sparse data**:
   - Some respondent/fueltype combos may have gaps (e.g., solar only during day)
   - Prepare handles NaN → fill_value; validate reports gaps
   - Don't force forecasting on sparse series without understanding why

4. **API pagination edge case**:
   - If dataset is exactly a multiple of page size, last page may appear empty
   - Code has stable sort to avoid this, but monitor for off-by-one errors

5. **Numeric conversion failing silently**:
   - If a few rows have non-numeric values, `pd.to_numeric(..., errors='coerce')` sets them to NaN
   - Validate always checks for NaN; if any appear, investigate API response

## Mini-checkpoint (prove you learned it)

Answer these questions:

1. **Explain the purpose of UTC normalization**. Why can't we just use the API's local timezone?
2. **What is a DST edge case** and how does our code detect it?
3. **What are the three checks in `validate_time_series_integrity()`**? (duplicates, missing hours, …)
4. **Why do we create a separate `prepare_for_forecasting()` step** instead of just using the raw API columns?

**Answers:**
1. Backtesting assumes monotonic, non-overlapping timestamps. Local timezones have repeated/missing hours during DST transitions. UTC eliminates this ambiguity.
2. DST fall-back repeats 1 hour (e.g., 1:30 AM occurs twice). Our integrity check detects duplicate (unique_id, ds) pairs and reports them.
3. Duplicates (same timestamp twice), missing hours (gaps in sequence), DST repeated hours (detected by duplicate).
4. StatsForecast expects schema `(unique_id, ds, y)` with no extra columns. Transforming here makes the interface explicit and validates assumptions.

## Exercises (optional, but recommended)

### Easy
1. Pull data for a different respondent (e.g., "NG_CA1") and verify the row count matches your date range.
2. Change start_date to a DST transition date (e.g., "2023-11-05") and inspect the integrity report.

### Medium
1. Intentionally corrupt one row (e.g., set a timestamp to 12:30 instead of 12:00) and run integrity check. What gets reported?
2. Pull a 1-year dataset and plot y vs ds. Look for gaps. Explain what you see (summer maintenance? solar seasonal?).

### Hard
1. Modify `validate_time_series_integrity()` to allow *up to* 2 duplicate pairs and *up to* 3 missing hours. Test that the threshold works correctly.
2. Write a function that estimates "optimal fill strategy" for missing hours: forward-fill vs interpolation vs skip. Test on a 3-month dataset with known gaps.
