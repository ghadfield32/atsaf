# Chapter 13 - Model Selection and Ensembling

## Outcomes (what I can do after this)

- [ ] I can rank models using consistent metrics
- [ ] I can select a champion model for deployment
- [ ] I can build a simple ensemble baseline

## Prerequisite (read first)

- Chapter 6 for backtesting and metrics

## Concepts (plain English)

- **Champion-challenger**: Track a best model and compare new candidates
- **Ensembling**: Combine multiple forecasts to reduce variance
- **Baseline discipline**: Always compare to a naive model

## Files touched

- `src/chapter2/evaluation.py` - metric helpers

## Step-by-step walkthrough

### 1) Simulate predictions from two models
```python
import numpy as np
from src.chapter2.evaluation import ForecastMetrics

rng = np.random.default_rng(123)
y_true = rng.normal(loc=50, scale=3, size=60)
model_a = y_true + rng.normal(scale=2, size=60)
model_b = y_true + rng.normal(scale=2.5, size=60)
ensemble = (model_a + model_b) / 2
```

### 2) Compare RMSE
```python
rmse_a = ForecastMetrics.rmse(y_true, model_a)
rmse_b = ForecastMetrics.rmse(y_true, model_b)
rmse_ens = ForecastMetrics.rmse(y_true, ensemble)
print("RMSE A:", rmse_a)
print("RMSE B:", rmse_b)
print("RMSE Ensemble:", rmse_ens)
```

## Metrics & success criteria

- Champion has the lowest primary metric
- Ensemble is competitive or more stable

## Pitfalls (things that commonly break)

1. Selecting models with different splits
2. Ignoring variance across windows

## Mini-checkpoint (prove you learned it)

1. Why keep a naive baseline in the leaderboard?
2. What is a simple ensemble that is often hard to beat?

**Answers:**
1. It prevents overfitting and keeps improvements honest.
2. A mean or median of strong models.
