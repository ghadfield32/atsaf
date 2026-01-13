# Chapter 11 - Hierarchical and Multi-series Forecasting

## Outcomes (what I can do after this)

- [ ] I can structure multiple series using unique_id
- [ ] I can compare local and global summaries
- [ ] I can explain why reconciliation matters at scale

## Prerequisite (read first)

- Chapter 0 for the time-series contract

## Concepts (plain English)

- **Local model**: One model per series
- **Global model**: One model across many series
- **Reconciliation**: Making forecasts consistent across levels

## Step-by-step walkthrough

### 1) Build a multi-series table
```python
import pandas as pd
import numpy as np

ds = pd.date_range("2024-01-01", periods=48, freq="H")
df = pd.DataFrame({
    "unique_id": ["series_a"] * len(ds) + ["series_b"] * len(ds),
    "ds": list(ds) + list(ds),
    "y": np.concatenate([
        10 + np.random.normal(scale=1.0, size=len(ds)),
        20 + np.random.normal(scale=1.5, size=len(ds)),
    ])
})
```

### 2) Compare local and global stats
```python
local_means = df.groupby("unique_id")["y"].mean()
global_mean = df["y"].mean()
print("Local means:")
print(local_means)
print("Global mean:", global_mean)
```

## Metrics & success criteria

- All series share the same schema
- Local and global summaries are consistent

## Pitfalls (things that commonly break)

1. Mixing series with different frequencies
2. Forgetting to include unique_id in joins

## Mini-checkpoint (prove you learned it)

1. Why use a global model at scale?
2. What does reconciliation prevent?

**Answers:**
1. It shares information across series and improves data efficiency.
2. It prevents total forecasts from conflicting with sub-series forecasts.
