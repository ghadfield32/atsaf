# Chapter 9 - Exogenous Regressors and Interventions

## Outcomes (what I can do after this)

- [ ] I can add weather, price, or event signals to a forecast
- [ ] I can avoid leakage by lagging or aligning features
- [ ] I can measure the lift from exogenous drivers

## Prerequisite (read first)

- Chapter 6 for backtesting context
- Chapter 7 for transformations

## Concepts (plain English)

- **Exogenous regressors**: External drivers such as weather or holidays
- **Intervention**: A one-off change or regime shift
- **Alignment**: Features must be known at prediction time

## Files touched

- `src/chapter2/feature_engineering.py` - lag and rolling features

## Step-by-step walkthrough

### 1) Create data with a weather driver and event
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.chapter2.evaluation import ForecastMetrics

ds = pd.date_range("2024-01-01", periods=240, freq="H")
temp = 20 + 5 * np.sin(np.arange(len(ds)) / 24.0)
event = (np.arange(len(ds)) % 72 == 0).astype(int)
y = 50 + 0.8 * temp + 10 * event + np.random.normal(scale=2.0, size=len(ds))

df = pd.DataFrame({"ds": ds, "temp": temp, "event": event, "y": y})
```

### 2) Train a simple regression model
```python
train = df.iloc[:-24]
test = df.iloc[-24:]

model = LinearRegression()
model.fit(train[["temp", "event"]], train["y"])
pred = model.predict(test[["temp", "event"]])
rmse = ForecastMetrics.rmse(test["y"].values, pred)
print("RMSE:", rmse)
```

### 3) Prevent leakage by shifting features
```python
shifted = df.copy()
shifted[["temp", "event"]] = shifted[["temp", "event"]].shift(1)
print(shifted.head(3))
```

## Metrics & success criteria

- RMSE improves vs a no-feature baseline
- Features are aligned to prediction time

## Pitfalls (things that commonly break)

1. Using future weather values for training
2. Encoding event flags after the fact

## Mini-checkpoint (prove you learned it)

1. Why do we shift features for multi-step forecasting?
2. What qualifies as an intervention feature?

**Answers:**
1. To avoid using information that would not be available at prediction time.
2. A binary or numeric signal representing a known event or change point.
