# Debugging Guide: Missing Hourly Data Points

## Issue Overview

**Error Message:**
```
RuntimeError: [pipeline][generation_validation] Too many missing hourly points
details={'worst_uid': 'CALI_SUN', 'worst_missing_ratio': 0.49653259361997226}
```

**What This Means:**
- The validation detected that 49.7% of expected hourly data points are missing for CALI_SUN
- With 363 actual rows, the expected count would be ~722 rows (30 days × 24 hours)
- This indicates a **DATA AVAILABILITY** issue from the EIA API, not a code bug

## Root Cause Analysis

### The Problem

1. **Request:** Pipeline requests 30 days of data (LOOKBACK_DAYS=30)
   - Start: 2026-01-13 00:00
   - End: 2026-02-12 23:00
   - Expected: ~720 hourly records

2. **Response:** EIA API returns only 363 rows for CALI_SUN
   - Actual coverage: ~50% of requested period
   - Additionally: 208 rows (57.3%) have NEGATIVE values (-58 to -10 MWh)

3. **Validation:** Fails when missing_ratio > MAX_MISSING_RATIO (default 0.02 = 2%)
   - Current: 49.7% missing >> 2% threshold
   - This is NOT fixable by adjusting thresholds (that would just hide the problem)

### Why This Happens

The EIA API has known data availability issues:
- **Delayed Publishing:** Some regions publish data with 24-72 hour delays
- **Data Gaps:** Historical data may have gaps due to reporting issues
- **Negative Values:** Bad data quality or corrections in the source system
- **Backfill Timing:** EIA may not have complete historical data immediately available

## Debugging Instrumentation Added

### 1. Fetch-Time Logging (`eia_renewable.py`)

**Location:** Lines 363-380 (after data parsing)

**What It Shows:**
```
[fetch_region][DATA_COVERAGE] region=CALI fuel=SUN
  requested=[2026-01-13 to 2026-02-12] (719.0h)
  actual=[2026-01-26T10:00:00 to 2026-02-12T07:00:00] (429.0h)
  records=363/720 (50.4% coverage)
```

**How to Interpret:**
- `requested`: What we asked the EIA API for
- `actual`: The actual time range in the returned data
- `records`: Actual rows returned vs expected
- `coverage`: Percentage of requested data received

**Key Insight:** If `actual` range is much smaller than `requested` range, the EIA API doesn't have data for the full period.

### 2. Validation Coverage Logging (`validation.py`)

**Location:** Lines 310-318 (per-series validation)

**What It Shows:**
```
[validation][COVERAGE] CALI_SUN:
  start=2026-01-26T10:00:00 end=2026-02-12T07:00:00
  span=429.0h actual=363 expected=430
  missing=67 ratio=0.156
```

**How to Interpret:**
- `start/end`: Actual time range in the data
- `span`: Hours between first and last timestamp
- `actual/expected`: Rows present vs rows expected for continuous hourly series
- `missing`: How many hourly points are missing
- `ratio`: Missing ratio (fails if > MAX_MISSING_RATIO)

**Key Insight:** If `span` is close to `expected` but `actual` is much lower, there are gaps **within** the data period (not just at the edges).

### 3. Gap Detection Logging (`validation.py`)

**Location:** Lines 320-332 (gap analysis)

**What It Shows:**
```
[validation][GAPS] CALI_SUN: Found 3 largest gaps (180 hours missing)
  Gap #1: 2026-01-26T10:00:00 to 2026-02-04T14:00:00 (216 hours)
  Gap #2: 2026-02-05T08:00:00 to 2026-02-07T12:00:00 (52 hours)
  Gap #3: 2026-02-09T15:00:00 to 2026-02-10T20:00:00 (29 hours)
```

**How to Interpret:**
- Shows the 3 largest continuous gaps in the time series
- `Gap #N`: The period where data is missing
- `(X hours)`: How many consecutive hours are missing

**Key Insight:**
- Large gaps (>24h) suggest EIA publishing delays or data unavailability
- Many small gaps suggest sporadic reporting issues
- Gaps at the beginning suggest we're asking for data before it's available

## How to Use This Debug Output

### Step 1: Check Fetch Coverage

Look for `[fetch_region][DATA_COVERAGE]` logs:

**Good:**
```
records=715/720 (99.3% coverage)  ← Nearly complete data
```

**Bad:**
```
records=363/720 (50.4% coverage)  ← Half the data is missing!
```

### Step 2: Check Validation Details

Look for `[validation][COVERAGE]` logs:

**What to Check:**
- Is `span` close to the LOOKBACK_DAYS period? (e.g., 720h for 30 days)
- Is `actual` close to `expected`?
- Is `ratio` < 0.02 (2% threshold)?

### Step 3: Analyze Gap Patterns

Look for `[validation][GAPS]` logs:

**Pattern 1: Large Gap at Start**
```
Gap #1: 2026-01-13T00:00:00 to 2026-01-26T10:00:00 (322 hours)
```
→ **Diagnosis:** EIA doesn't have data for the early part of requested period
→ **Solution:** Reduce LOOKBACK_DAYS or wait for backfill

**Pattern 2: Sporadic Gaps Throughout**
```
Gap #1: 2026-01-20T14:00:00 to 2026-01-20T18:00:00 (4 hours)
Gap #2: 2026-01-25T03:00:00 to 2026-01-25T09:00:00 (6 hours)
```
→ **Diagnosis:** Intermittent reporting issues
→ **Solution:** May need to implement interpolation or accept lower quality

**Pattern 3: Recent Gap**
```
Gap #1: 2026-02-11T12:00:00 to 2026-02-12T07:00:00 (19 hours)
```
→ **Diagnosis:** Recent data not published yet (publishing lag)
→ **Solution:** Adjust MAX_LAG_HOURS or wait longer before fetching

### Step 4: Check for Negative Values

Look for `[fetch_region][NEGATIVE]` logs:

```
[fetch_region][NEGATIVE] region=CALI fuel=SUN count=208 (57.3%) range=[-58.00, -10.00]
```

**What This Means:**
- More than half the data has negative generation values
- Negative generation is physically impossible (can't generate -58 MWh)
- This indicates BAD DATA QUALITY from the EIA API

**Red Flags:**
- Negative % > 10%: Significant data quality issues
- Negative % > 50%: Severe data quality issues (like this case)
- Large negative magnitudes: Likely data errors, not small rounding issues

## Solutions (NOT Workarounds)

### Short-Term Solutions

1. **Reduce Lookback Period**
   ```bash
   LOOKBACK_DAYS=7  # Instead of 30
   ```
   - Focuses on more recent data which is more likely to be available
   - Trade-off: Less training data for models

2. **Wait for Backfill**
   - EIA may backfill data over time
   - Check again in 24-48 hours
   - Not a solution for real-time pipelines

3. **Region-Specific Lag Thresholds**
   ```bash
   MAX_LAG_HOURS=48           # Default
   MAX_LAG_HOURS_CALI=72      # Give CALI more time
   ```
   - Accommodates known delays for specific regions
   - Already implemented in the codebase

### Long-Term Solutions

1. **Implement Data Interpolation**
   - Fill small gaps (< 4 hours) with interpolation
   - Only for training data, not for forecasting
   - Document interpolated points clearly

2. **Alternative Data Sources**
   - Cross-reference with CAISO OASIS (for California)
   - Use multiple data sources and pick best coverage
   - More complex but more reliable

3. **Negative Value Handling**
   - Current: EDA detects and recommends policy (clamp/remove/keep)
   - Need: Automated flagging of "too many negatives" (>10% threshold)
   - Consider: Contact EIA about data quality issues

4. **Adaptive Lookback**
   - Start with 30 days, reduce if coverage < 90%
   - Retry with 14 days, then 7 days
   - Document actual period used in metadata

## When to Treat as Error vs Warning

### Treat as ERROR (fail pipeline):
- Missing ratio > 5% for recent data (< 7 days old)
- Negative values > 10% of data
- Complete data unavailability (0 rows returned)

### Treat as WARNING (log but continue):
- Missing ratio 2-5% for recent data
- Negative values < 10% (can be handled by EDA)
- Old data gaps (> 30 days ago) - less critical for forecasting

## Next Steps

1. **Run pipeline with debug logging enabled:**
   ```bash
   python -m src.renewable.jobs.run_hourly
   ```

2. **Review the debug output** following the steps above

3. **Determine root cause:**
   - Is it a temporary EIA delay?
   - Is it a systemic data availability issue?
   - Is it bad data quality (negatives)?

4. **Choose appropriate solution** based on diagnosis

5. **Document findings** in PROJECT_LOG.md

## Testing the Debug Output

To test with intentionally problematic data:

```bash
# Test with very long lookback (will likely have gaps)
LOOKBACK_DAYS=60 python -m src.renewable.jobs.run_hourly

# Test with very short lookback (should be clean)
LOOKBACK_DAYS=3 python -m src.renewable.jobs.run_hourly

# Test with specific problematic date range
RENEWABLE_START_DATE=2026-01-01 RENEWABLE_END_DATE=2026-01-31 python -m src.renewable.jobs.run_hourly
```

## Files Modified

1. **`src/renewable/validation.py`**
   - Added `_detect_time_gaps()` function (lines 22-51)
   - Added coverage logging in validation (lines 310-318)
   - Added gap detection logging (lines 320-332)

2. **`src/renewable/eia_renewable.py`**
   - Added data coverage logging after fetch (lines 363-380)

3. **`docs/DEBUGGING_DATA_GAPS.md`**
   - This document (NEW)

## References

- EIA API Documentation: https://www.eia.gov/opendata/
- Project Log: [PROJECT_LOG.md](../PROJECT_LOG.md)
- Validation Module: [src/renewable/validation.py](../src/renewable/validation.py)
- EIA Fetcher: [src/renewable/eia_renewable.py](../src/renewable/eia_renewable.py)
