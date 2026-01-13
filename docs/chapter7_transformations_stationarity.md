# Chapter 7 - Transformations and Stationarity

## Outcomes (what I can do after this)

- [ ] I can apply log or Box-Cox style transforms safely
- [ ] I can difference a series to remove trend
- [ ] I can run a stationarity test when needed

## Prerequisite (read first)

- Chapter 0 for basic time series handling
- Chapter 6 for backtesting context

## Concepts (plain English)

- **Transformations**: Log or Box-Cox to stabilize variance
- **Differencing**: Subtract lagged values to remove trend
- **Stationarity**: Stable mean and variance over time
- **Unit root tests**: ADF or KPSS as a quick check

## Files touched

- statsmodels (optional): stationarity tests

## Step-by-step walkthrough

### 1) Build a trending series
```python
import pandas as pd
import numpy as np

ds = pd.date_range("2024-01-01", periods=200, freq="H")
trend = 5 + 0.05 * np.arange(len(ds))
noise = np.random.normal(scale=0.5, size=len(ds))
series = pd.Series(trend + noise, index=ds)
```

### 2) Apply log and differencing
```python
log_series = np.log1p(series)
diff_series = series.diff().dropna()
print("Original mean:", series.mean())
print("Diff mean:", diff_series.mean())
```

### 3) Run an ADF test (if available)
```python
try:
    from statsmodels.tsa.stattools import adfuller
    p_value = adfuller(series.values)[1]
    p_value_diff = adfuller(diff_series.values)[1]
    print("ADF p-value (orig):", p_value)
    print("ADF p-value (diff):", p_value_diff)
except Exception as exc:
    print("statsmodels not available:", exc)
```

## Metrics & success criteria

- Differencing reduces trend-related drift
- Stationarity test p-value decreases after differencing

## Pitfalls (things that commonly break)

1. Applying log to negative values
2. Differencing too aggressively and losing signal
3. Treating stationarity tests as absolute truth

## Mini-checkpoint (prove you learned it)

1. When should you consider differencing a series?
2. Why do we avoid log(0) without an offset?

**Answers:**
1. When trend dominates and residuals show non-stationarity.
2. log(0) is undefined, so we use log1p or an offset.
