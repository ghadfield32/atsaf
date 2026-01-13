# Chapter 12 - Special Data Types

## Outcomes (what I can do after this)

- [ ] I can handle zero-heavy or count series safely
- [ ] I can choose metrics that do not explode on zeros
- [ ] I can explain when GLM-style models are needed

## Prerequisite (read first)

- Chapter 6 for evaluation metrics

## Concepts (plain English)

- **Intermittent demand**: Many zeros with occasional spikes
- **Count data**: Non-negative integers (Poisson, NB)
- **Zero inflation**: Extra zeros beyond Poisson expectations

## Step-by-step walkthrough

### 1) Simulate intermittent demand
```python
import numpy as np
from src.chapter2.evaluation import ForecastMetrics

rng = np.random.default_rng(42)
y_true = rng.poisson(lam=1.0, size=100)
y_true[rng.choice(len(y_true), size=40, replace=False)] = 0

yhat = np.maximum(0, y_true + rng.normal(scale=0.5, size=len(y_true)))
```

### 2) Compare metrics
```python
mae = ForecastMetrics.mae(y_true, yhat)
mape = ForecastMetrics.mape(y_true, yhat)
mase = ForecastMetrics.mase(y_true, yhat, y_true, season_length=1)
print("MAE:", mae)
print("MAPE:", mape)
print("MASE:", mase)
```

## Metrics & success criteria

- Prefer MAE or MASE when zeros are common
- MAPE should be treated with caution

## Pitfalls (things that commonly break)

1. Using MAPE on zeros
2. Predicting negative counts

## Mini-checkpoint (prove you learned it)

1. Why is MAPE unreliable for zero-heavy series?
2. Which metrics are safer alternatives?

**Answers:**
1. Division by values near zero makes MAPE explode or become undefined.
2. MAE and MASE are safer choices.
