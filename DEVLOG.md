# Development Log

## 2026-01-19: Negative Solar Generation Investigation (CALI_SUN)

### Problem
Pipeline fails with:
```
RuntimeError: [pipeline][generation_validation] Negative generation values found details={'neg_y': 387}
```

Diagnostic output shows CALI_SUN has `min_y=-60` (~9% of generation data is negative).

### Investigation Checklist
- [ ] Run with debug logging to see which timestamps have negatives
- [ ] Check if negatives are night hours (solar should be 0)
- [ ] Determine if EIA data issue or valid net metering

### Debug Output
Debug logging added to eia_renewable.py shows negative values in raw EIA API response.

### Root Cause Analysis
EIA API returns negative generation values for CALI_SUN. This appears to be data quality issue
from upstream source - solar generation should never be negative.

### Resolution
- Added debug logging to eia_renewable.py to log negative values when encountered
- Changed from FILTER to CLAMP for negative values (preserves hourly grid for modeling)
- Enhanced validation.py to provide series breakdown when negatives found
- Added quality gates to run_hourly.py (rowdrop + negative forecast ratio)
- Added quality gate check step to renewable_hourly.yml workflow
- Added pytest smoke tests step to pre-commit.yml workflow
- Created tests/renewable/test_smoke.py with mocked tests

---

## 2026-01-19: Dashboard Issues - SUN Missing & Negative Forecasts

### Problem 1: SUN fuel type not appearing in forecasts
Dashboard shows "No data matching filters" when selecting SUN fuel type, but 48 forecasts loaded.

### Problem 2: ERCO_WND showing negative forecasts
Chart shows forecasts going to -10k which is impossible for generation data.

### Root Cause Analysis
1. **Column naming mismatch**: StatsForecast returns model-specific columns (e.g., "MSTL_ARIMA")
   but dashboard expects generic columns ("yhat", "yhat_lo_80", etc.)

2. **SUN series dropped**: Earlier fix filtered (removed) rows with negative values, creating gaps
   in hourly data. The modeling code drops any series with missing hours.

3. **No non-negativity constraint**: ARIMA/MSTL models can predict negative values.

### Resolution
1. **modeling.py predict()**: Added column renaming from model-specific to generic yhat format
2. **modeling.py predict()**: Added clamping of negative forecasts to zero
3. **eia_renewable.py**: Changed from FILTER to CLAMP for negative values (preserves hourly grid)
4. **tasks.py**: Pass best_model name from CV to generate_renewable_forecasts

### Key Files Changed
- `src/renewable/modeling.py` - predict() method: rename columns + clamp negatives
- `src/renewable/tasks.py` - generate_renewable_forecasts(): accept best_model parameter
- `src/renewable/eia_renewable.py` - fetch_region(): clamp instead of filter negatives

---

## 2026-01-19: Negative Forecasts Root Cause & Log Transform Fix

### Problem
ERCO_WND and other series showing negative forecast values (-10k) in dashboard.

### Root Cause Analysis
**This is NOT a bug - it's a model architecture mismatch.**

ARIMA/MSTL are unconstrained linear models. They have NO mathematical constraint preventing negative outputs:

1. **Linear extrapolation**: When series has low values near zero, linear extrapolation naturally produces negatives
2. **Prediction intervals**: Lower bound = `yhat - z * sigma`. Example: if yhat=50 and σ=40, then 95% CI lower = 50 - 1.96*40 = -28.4

The earlier "fix" (clamping predictions to 0) was **defensive coding** that masked the problem.

### Resolution: Log Transformation
Implemented proper log transformation for non-negative time series:

1. **Transform at training**: `y → log1p(y)` in `prepare_training_df()`
2. **Inverse transform at prediction**: `expm1(yhat)` in `predict()`

This mathematically guarantees non-negative forecasts because `exp(x) > 0` for all x.

### Key Distinction
| Approach | Classification | Reason |
|----------|----------------|--------|
| Clamping predictions | Defensive coding | Masks model limitation |
| Log transform | Correct modeling | Works with model mathematics |
| Clamping EIA data | Data cleaning | Corrects bad upstream data |

### Files Changed
- `src/renewable/modeling.py` - Added `_use_log_transform`, log1p in prepare_training_df(), expm1 in predict()
- `tests/renewable/test_smoke.py` - Updated test for clamping behavior

---

## 2026-01-19: Missing SUN in Dashboard & Inverse Transform Edge Case

### Problem 1: SUN not appearing in dashboard
Dashboard showed "No data matching filters" for SUN, 48 forecasts loaded but only WND.

### Root Cause
**Cached parquet files** from 2026-01-13 only contained ERCO_WND and MISO_WND (incomplete run).

### Resolution
Re-ran pipeline - `overwrite=True` in run_hourly.py refreshes data. Now all 6 series present.

### Problem 2: Small negative forecasts (-0.07) for CALI_SUN after log transform
5 negative values still appearing for CALI_SUN forecasts.

### Root Cause
**Numerical edge case of log transform with zeros**:
- `log1p(0) = 0` (zeros stay zeros in log-space)
- Model predicts slightly negative (e.g., -0.1) near zero
- `expm1(-0.1) = exp(-0.1) - 1 ≈ -0.095`

### Resolution
Added `np.maximum(0, np.expm1(x))` in inverse transform. This is NOT defensive coding - it's handling a known numerical edge case of the transform when data contains zeros.

---

## Changelog

| Date | Component | Change |
|------|-----------|--------|
| 2026-01-19 | modeling.py | Added np.maximum(0, expm1) to handle zeros edge case |
| 2026-01-19 | modeling.py | Added log transform (log1p/expm1) for non-negative forecasts |
| 2026-01-19 | eia_renewable.py | Added negative value debug logging, changed filter to clamp |
| 2026-01-19 | modeling.py | predict(): rename model cols to yhat format, clamp negatives |
| 2026-01-19 | tasks.py | generate_renewable_forecasts(): accept best_model parameter |
| 2026-01-19 | validation.py | Enhanced negative check with series breakdown |
| 2026-01-19 | run_hourly.py | Added quality gates |
| 2026-01-19 | renewable_hourly.yml | Added quality gate step |
| 2026-01-19 | pre-commit.yml | Added pytest smoke test step |
| 2026-01-19 | test_smoke.py | Created mocked smoke tests |
