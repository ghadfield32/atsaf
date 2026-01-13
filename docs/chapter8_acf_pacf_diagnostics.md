# Chapter 8 - ACF, PACF, and Diagnostics

## Outcomes (what I can do after this)

- [ ] I can compute ACF and PACF values to inspect lag structure
- [ ] I can run a basic residual diagnostic test
- [ ] I can decide when ARIMA-style models are appropriate

## Prerequisite (read first)

- Chapter 7 for transformations and stationarity context

## Concepts (plain English)

- **ACF**: Autocorrelation at each lag
- **PACF**: Partial autocorrelation after removing shorter lags
- **Residual diagnostics**: Check for leftover structure

## Files touched

- statsmodels (optional): acf, pacf, ljungbox

## Step-by-step walkthrough

### 1) Create a seasonal series
```python
import pandas as pd
import numpy as np

ds = pd.date_range("2024-01-01", periods=200, freq="H")
series = pd.Series(
    np.sin(np.arange(len(ds)) / 6.0) + np.random.normal(scale=0.3, size=len(ds)),
    index=ds,
)
```

### 2) Compute ACF and PACF
```python
try:
    from statsmodels.tsa.stattools import acf, pacf
    acf_vals = acf(series.values, nlags=10)
    pacf_vals = pacf(series.values, nlags=10)
    print("ACF:", acf_vals)
    print("PACF:", pacf_vals)
except Exception as exc:
    print("statsmodels not available:", exc)
```

### 3) Run a residual test (optional)
```python
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(series.values, lags=[10], return_df=True)
    print(lb)
except Exception as exc:
    print("statsmodels not available:", exc)
```

## Metrics & success criteria

- ACF/PACF patterns are consistent with model choice
- Residual tests show weak autocorrelation

## Pitfalls (things that commonly break)

1. Reading ACF/PACF without checking stationarity
2. Interpreting noisy small samples as strong structure

## Mini-checkpoint (prove you learned it)

1. What does a slow ACF decay suggest?
2. Why do we test residuals after fitting a model?

**Answers:**
1. The series likely has trend or is non-stationary.
2. Residuals should look like noise if the model captured structure.
