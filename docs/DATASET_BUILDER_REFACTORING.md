# Dataset Builder & EDA Refactoring

## Summary

Simplified the renewable energy dataset builder and EDA modules to follow a **clean separation of concerns**:

- **Production path**: Load → Quick validation → Single canonical build → Ready for modeling
- **Diagnostic tool**: EDA (optional, separate) informs decisions but doesn't block the main path
- **Learning/demos**: Policy comparisons behind a `RUN_DEMOS` flag

---

## Changes Made

### 1. EDA Module (`src/renewable/eda.py`)

#### Removed:
- ❌ Seaborn dependency (`sns.set_style()`)
- ❌ Pandas deprecation warnings (fixed 3 deprecations):
  - `fillna(method='ffill')` → `ffill()`
  - `fillna(method='bfill')` → `bfill()`
  - `freq='H'` → `freq='h'`
  - `groupby().apply(lambda ...)` → explicit groupby operations

#### Why:
- Seaborn was only used for styling; matplotlib defaults are sufficient
- Pandas deprecations cause future compatibility issues
- Cleaner code with fewer dependencies

### 2. Dataset Builder (`src/renewable/dataset_builder.py`)

#### Main Changes:

**Before (Overdone):**
```
[2/5] Build with negative_policy='clamp'
[3/5] Demo negative_policy='fail_loud'
[4/5] Demo negative_policy='hybrid'
[5/5] Compare final datasets
→ Redundant demos every run, confuses which is "canonical"
```

**After (Clean & Simple):**
```
[1/3] Load & validate data (quick checks)
[2/3] Build modeling dataset (ONE canonical build, auto-selects policy)
[3/3] Inspect dataset (quality checks & sample)
→ Production path is clear & intentional
```

#### Smart Policy Selection:
The builder now **auto-selects** the best policy based on EDA findings:

```python
# No negatives found → fail_loud (detect if they appear upstream)
# <1% negatives → fail_loud (stay strict)
# >1% negatives → clamp (tolerate & log with diagnostics)
```

#### Demo Mode (Optional):
Run policy comparisons with an environment variable:
```bash
# Production path (default)
python -m src.renewable.dataset_builder

# With policy demos (educational)
RUN_DEMOS=1 python -m src.renewable.dataset_builder
```

---

## New Workflow

### Standard Production Pipeline

```python
# Load data
generation_df = pd.read_parquet("data/renewable/generation.parquet")
weather_df = pd.read_parquet("data/renewable/weather.parquet")

# Build modeling dataset (ONE call, canonical path)
from src.renewable.dataset_builder import build_modeling_dataset

modeling_df, report = build_modeling_dataset(
    generation_df,
    weather_df,
    negative_policy="fail_loud",  # Or auto-selected by CLI
    hourly_grid_policy="drop_incomplete_series",
    output_dir=Path("data/renewable/preprocessing/latest")
)

# Use for modeling (weather already merged)
from src.renewable.modeling import RenewableForecastModel

model = RenewableForecastModel(horizon=24)
cv_results, leaderboard = model.cross_validate(
    df=modeling_df,
    weather_df=None,  # Already merged by dataset_builder!
    n_windows=3
)
```

### When to Run EDA

EDA is now a **diagnostic tool**, not part of the main path:

```bash
# Run when:
# - You're adding new regions/fuel types
# - You suspect data quality drift
# - You want to understand the data before modeling

python scripts/run_eda_renewable.py

# Or with filters:
python scripts/run_eda_renewable.py --regions CALI,ERCO --fuels WND
```

**EDA outputs inform dataset builder policy choices**, but don't block the pipeline.

---

## Output Structure

### Canonical build saves to:
```
data/renewable/preprocessing/latest/
├── preprocessing_report.json       # Full diagnostics
├── negative_values_detail.json     # (if negatives found)
└── missing_hours_detail.json       # (if series dropped)
```

### EDA saves to:
```
reports/renewable/eda/YYYYMMDD_HHMMSS/
├── metadata.json
├── eda_summary.json                # Key findings
├── seasonality/
├── zero_inflation/
├── coverage/
├── negative_values/
└── weather_alignment/
```

---

## Comparison: Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| **Main build steps** | 2 or 3 policy demos every run | 1 canonical build + optional demos |
| **Policy selection** | Manual decision required | Auto-selected based on data |
| **EDA timing** | Encouraged to run before building | Optional diagnostic tool |
| **Dependencies** | Seaborn (unused after styling) | Removed seaborn |
| **Deprecations** | 3 pandas warnings | None |
| **Clear "true path"?** | Ambiguous (clamp? fail_loud? hybrid?) | Clear (auto-selected or specified) |

---

## Key Insights from Current Data

Your EDA showed:
- ✅ **No negatives** → Policy: `fail_loud` (stay strict, detect upstream changes)
- ✅ **No missing hours** → Policy: `drop_incomplete_series` (preserves all series)
- ✅ **Weather 100% merged** → Include all 7 weather variables
- ✅ **Series >98% complete** → No data quality issues

→ The builder **automatically selects** these optimal policies.

---

## Migration Guide

### If you have existing scripts:

**Old approach:**
```python
# You had to pick a policy manually
modeling_df, report = build_modeling_dataset(
    generation_df, weather_df,
    negative_policy="clamp",  # Which one?
    ...
)
```

**New approach:**
```python
# Auto-selects policy based on EDA findings
modeling_df, report = build_modeling_dataset(
    generation_df, weather_df,
    negative_policy="fail_loud",  # Or let CLI auto-select
    ...
)
```

No code changes needed — signature is identical.

---

## Next Steps

1. **Review the refactored code** (`src/renewable/eda.py` and `src/renewable/dataset_builder.py`)
2. **Run the canonical build**: `python -m src.renewable.dataset_builder`
3. **Check the output** in `data/renewable/preprocessing/latest/`
4. **Optional**: Run EDA diagnostics if you want to understand the data deeper

---

## Removed Complexity

### What you no longer need to decide:
- ❌ Which policy to demo
- ❌ Whether to run EDA before building
- ❌ Which redundant builds to save
- ❌ Seaborn dependencies
- ❌ Pandas deprecation warnings

### What's now automatic:
- ✅ Policy selection (based on data characteristics)
- ✅ Diagnostic output (always saved)
- ✅ Recommendation logic (built-in)
- ✅ Clean separation (production vs learning)
