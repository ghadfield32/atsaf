# Chapter 6 - Backtesting and Evaluation

## Outcomes (what I can do after this)

- [ ] I can build rolling-origin splits without leakage
- [ ] I can compute RMSE, MAE, and MASE on holdout windows
- [ ] I can compare models fairly using identical splits

## Prerequisite (read first)

- Chapter 0 for the time-series contract
- Chapter 2 for model training utilities

## Concepts (plain English)

- **Rolling-origin CV**: Move the train/test cutoff forward through time
- **Horizon**: Number of steps to predict in each window
- **Leakage**: Using future data to predict the past
- **Leaderboard**: Ranked models with metrics across windows

## Architecture (what we are building)

### Inputs
- Forecasting-ready data [unique_id, ds, y]
- Backtesting parameters (horizon, step size, windows)

### Outputs
- Split definitions
- Metrics per model per split

### Invariants (must always hold)
- Train end < test start
- Same splits for all models

## Files touched

- `src/chapter2/backtesting.py` - rolling and expanding splits
- `src/chapter2/evaluation.py` - RMSE, MAE, MASE, coverage
- `src/chapter2/training.py` - orchestration pipeline

## Step-by-step walkthrough

### 1) Create a synthetic series
```python
import pandas as pd
import numpy as np

from src.chapter2.backtesting import RollingWindowBacktest
from src.chapter2.evaluation import ForecastMetrics

ds = pd.date_range("2024-01-01", periods=240, freq="H")
y = 10 + 0.1 * np.arange(len(ds)) + np.random.normal(scale=0.5, size=len(ds))

df = pd.DataFrame({"unique_id": "series_1", "ds": ds, "y": y})
```

### 2) Generate rolling splits
```python
backtest = RollingWindowBacktest(min_train_size=120, test_size=24, step_size=24)
splits = backtest.generate_splits(df, unique_id="series_1")
print(splits[0].info)
```
- Expect: train_end < test_start

### 3) Score a naive baseline
```python
split = splits[0]
train = df.iloc[split.train_indices]
test = df.iloc[split.test_indices]

yhat = np.repeat(train["y"].iloc[-1], len(test))
rmse = ForecastMetrics.rmse(test["y"].values, yhat)
print("RMSE:", rmse)
```

## Metrics & success criteria

- Primary: RMSE
- Secondary: MAE, MASE

## Pitfalls (things that commonly break)

1. Using shuffled data (breaks time order)
2. Using future features in train windows
3. Comparing models on different splits

## Mini-checkpoint (prove you learned it)

1. Why is rolling-origin CV preferred over random splits?
2. What does MASE tell you that RMSE does not?

**Answers:**
1. It preserves time order and avoids leakage.
2. MASE scales error relative to a seasonal naive baseline.
