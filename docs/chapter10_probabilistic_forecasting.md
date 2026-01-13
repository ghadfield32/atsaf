# Chapter 10 - Probabilistic Forecasting and Calibration

## Outcomes (what I can do after this)

- [ ] I can compute prediction interval coverage
- [ ] I can compare nominal and empirical coverage
- [ ] I can spot overconfident or underconfident intervals

## Prerequisite (read first)

- Chapter 6 for evaluation basics

## Concepts (plain English)

- **Prediction interval**: A range that should contain the true value
- **Coverage**: Percent of actuals inside the interval
- **Calibration**: Agreement between nominal and empirical coverage

## Files touched

- `src/chapter2/evaluation.py` - coverage metric

## Step-by-step walkthrough

### 1) Simulate predictions and intervals
```python
import numpy as np
from src.chapter2.evaluation import ForecastMetrics

rng = np.random.default_rng(7)
y_true = rng.normal(loc=100, scale=5, size=200)
yhat = y_true + rng.normal(scale=2, size=200)

interval = 4.0
lower = yhat - interval
upper = yhat + interval
```

### 2) Compute coverage
```python
coverage = ForecastMetrics.coverage(y_true, lower, upper)
print("Coverage:", coverage)
```

### 3) Compare to nominal
```python
nominal = 95
print("Nominal:", nominal)
print("Gap:", nominal - coverage)
```

## Metrics & success criteria

- Empirical coverage is close to nominal (+/- 5 is a common rule)

## Pitfalls (things that commonly break)

1. Intervals computed on the wrong scale
2. Ignoring missing values in coverage calculation

## Mini-checkpoint (prove you learned it)

1. What does coverage above nominal usually indicate?
2. Why should we check calibration after backtesting?

**Answers:**
1. Intervals are too wide or overly conservative.
2. It reveals whether uncertainty estimates are trustworthy.
