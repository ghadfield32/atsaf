# Renewable Energy Forecasting Pipeline - Complete Architecture

**Version:** 2.0
**Last Updated:** 2026-01-20
**Status:** Production

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module Architecture](#2-module-architecture)
3. [Step-by-Step Pipeline Flow](#3-step-by-step-pipeline-flow)
4. [Future Todos & Improvements](#4-future-todos--improvements)
5. [Known Issues & Fixes](#5-known-issues--areas-requiring-fixes)
6. [Critical File Locations](#6-critical-file-locations)
7. [Quick Start Guide](#7-quick-start-guide)

---

## 1. System Overview

### 1.1 Purpose

Production-grade renewable energy forecasting system that generates 24-hour probabilistic forecasts for wind and solar generation across multiple US regions using EIA API data and weather regressors from Open-Meteo.

### 1.2 High-Level Data Flow

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  EIA API    │──┬──▶│ Data         │──┬──▶│ StatsForecast│
│ (WND/SUN)   │  │   │ Pipeline     │  │   │ Models       │
└─────────────┘  │   └──────────────┘  │   └─────────────┘
                 │                     │           │
┌─────────────┐  │   ┌──────────────┐  │   ┌─────▼──────┐
│ Open-Meteo  │──┘   │ Validation   │  │   │Probabilistic│
│ Weather API │      │ & Quality    │  │   │Forecasts    │
└─────────────┘      │ Gates        │  │   │(80%, 95% CI)│
                     └──────────────┘  │   └────────────┘
                                       │           │
                                       │   ┌───────▼─────┐
                                       └──▶│  Artifacts  │
                                           │  Commit &   │
                                           │  Dashboard  │
                                           └─────────────┘
```

### 1.3 Pipeline Execution Modes

| Mode | Trigger | Use Case |
|------|---------|----------|
| **Scheduled (Production)** | GitHub Actions cron (`17 * * * *`) | Hourly automated forecasts |
| **Manual** | CLI via `run_hourly.py` | Ad-hoc runs, debugging |
| **Development** | Jupyter notebook interactive | Feature development, exploration |
| **Airflow (Optional)** | DAG builder | Complex orchestration workflows |

### 1.4 Key Features

✅ **Multi-region forecasting** - 5 US regions (CALI, ERCO, MISO, PJM, SWPP)
✅ **Multi-fuel support** - Wind (WND) and Solar (SUN)
✅ **Probabilistic intervals** - 80% and 95% confidence bounds
✅ **Weather-enhanced models** - 7 weather variables as exogenous regressors
✅ **Drift monitoring** - Automatic detection of model degradation
✅ **Quality gates** - Rowdrop and negative forecast validation
✅ **Git-based artifacts** - Version-controlled data lineage
✅ **Streamlit dashboard** - Interactive visualization

---


## 2. Module Architecture

### 2.1 Core Modules (`src/renewable/`)

```
src/renewable/
├── eia_renewable.py         # EIA API client (generation data)
├── open_meteo.py            # Weather API client (7 variables)
├── regions.py               # Region/fuel type registry (11 regions)
├── validation.py            # 10-step data quality validation
├── modeling.py              # StatsForecast models + log transform
├── tasks.py                 # 5-task pipeline orchestration
├── db.py                    # SQLite persistence layer
├── dashboard.py             # Streamlit 4-tab dashboard
├── dag_builder.py           # Airflow integration (optional)
└── jobs/
    └── run_hourly.py        # Production entry point + quality gates
```

---

### 2.2 Module Details

#### **eia_renewable.py** - EIA API Integration

**Class:** `EIARenewableFetcher`

**Responsibilities:**
- Fetch hourly renewable generation data from EIA API
- Handle pagination (5000 records/request)
- Rate limiting (5 req/sec, 0.2s delay)
- Multi-region parallel fetching (ThreadPoolExecutor, max_workers=3)
- **Data cleaning:** Clamp negative values at ingestion

**Critical Decision: Why Clamp Negatives?**

> **Problem:** EIA API returns erroneous negative values for solar generation (e.g., CALI_SUN: -60 MWh).
> **Why Clamp?** Clamping at fetch preserves hourly grid structure required for modeling.
> **Why NOT Filter?** Filtering creates gaps in the time series, causing `_enforce_hourly_grid()` to fail.
> **Classification:** This is **data cleaning** (correcting bad upstream data), NOT defensive coding.

**Key Methods:**

| Method | Purpose | Returns |
|--------|---------|---------|
| `fetch_region(region, fuel, start, end)` | Single region with diagnostics | DataFrame |
| `fetch_all_regions(fuel, regions, ...)` | Parallel multi-region fetch | Combined DataFrame |
| `get_series_summary(df)` | Aggregated stats (min, max, zero%) | Dict |

**Output Format:**
```python
DataFrame: [unique_id, ds, y]
unique_id: "{region}_{fuel_type}"  # e.g., "CALI_WND", "ERCO_SUN"
ds: pandas Timestamp (naive UTC)
y: float (MWh, clamped >= 0)
```

**Environment Variables:**
- `EIA_API_KEY` (required) - API key from EIA dashboard
- Rate limits: 5 requests/second (built-in delay)

---

#### **open_meteo.py** - Weather Data Provider

**Class:** `OpenMeteoRenewable`

**Responsibilities:**
- Fetch historical weather (archive-api.open-meteo.com)
- Fetch forecast weather (api.open-meteo.com)
- Exponential backoff retry logic (socket timeouts)
- Strict validation (fail-loud on missing variables)

**Critical Decision: Why Two Endpoints?**

> **Historical Endpoint:** Training data (no leakage of future actuals)
> **Forecast Endpoint:** Prediction data (realistic - weather forecasts are available IRL)
>
> Combining both allows the model to use forecasted weather for predictions while training on historical weather only.

**Weather Variables (7):**
```python
["temperature_2m",        # °C
 "wind_speed_10m",        # km/h
 "wind_speed_100m",       # km/h (wind turbine height)
 "wind_direction_10m",    # degrees
 "direct_radiation",      # W/m²
 "diffuse_radiation",     # W/m²
 "cloud_cover"]           # %
```

**Recent Fix (2026-01-19):** Added socket timeout retry (`connect=3, read=3`) + increased timeout to 60s

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `fetch_historical(lat, lon, start, end)` | Archive API for training |
| `fetch_forecast(lat, lon, horizon_hours)` | Forecast API for predictions |
| `fetch_all_regions_historical(regions, dates)` | Parallel historical fetch |
| `fetch_all_regions_forecast(regions, horizon)` | Parallel forecast fetch |

**Output Format:**
```python
DataFrame: [ds, region, temperature_2m, wind_speed_10m, ...]
ds: pandas Timestamp (naive UTC)
region: str (CALI, ERCO, MISO, etc.)
```

---

#### **regions.py** - Region & Fuel Type Registry

**Purpose:** Central registry for region metadata and validation.

**Data Structure:**
```python
REGIONS: dict[str, RegionInfo]
# RegionInfo(name, lat, lon, timezone, eia_respondent)

# Fetch-enabled regions (5):
CALI, ERCO, MISO, PJM, SWPP  # Have EIA respondent mapping

# Weather-only regions (6):
NW, SW, SE, FLA, CAR, TEN  # No EIA mapping, weather analysis only

# Aggregate:
US48  # National aggregate

FUEL_TYPES: {"WND": "Wind", "SUN": "Solar"}
```

**Critical Decision: Why Separate Weather-Only Regions?**

> Some regions lack EIA respondent mapping but are useful for weather data analysis.
> This separation allows expansion without breaking fetch logic.
> Weather-only regions can be added for climate analysis without requiring EIA data.

**Key Functions:**
- `get_eia_respondent(region)` - Fails loud if region not mapped
- `validate_region(code)` - Boolean check
- `validate_fuel_type(code)` - Boolean check
- `get_region_coords(region)` - Returns (lat, lon)
- `list_regions()` - Returns all region codes

---

#### **validation.py** - Data Quality Gates

**Function:** `validate_generation_df(df, max_lag_hours, max_missing_ratio, expected_series)`

**10-Step Validation Checklist:**

```
1. ✓ Required columns present (unique_id, ds, y)
2. ✓ DataFrame not empty
3. ✓ DateTime parsing success (ds column)
4. ✓ Numeric parsing success (y column)
5. ⚠️ CRITICAL: No negative values (generation integrity)
6. ✓ No duplicate (unique_id, ds) pairs
7. ✓ Expected series all present
8. ⚠️ CRITICAL: Freshness check (lag_hours <= max_lag_hours)
9. ✓ Per-series freshness (no stale individual series)
10. ✓ Hourly grid completeness (missing_ratio <= max_missing_ratio)
```

**Critical Decision: Why 48h Lag Threshold?**

> **Problem:** EIA hourly data has typical publishing delay of 12-24h.
> **Initial Threshold:** 3h caused false failures (data "stale" but actually normal).
> **Fix:** Changed to 48h to match operational reality.
> **Trade-off:** 48h provides margin for worst-case delays without being too lenient.

**Returns:**
```python
@dataclass
class ValidationReport:
    ok: bool              # True if all checks pass
    message: str          # Human-readable status
    details: dict         # Diagnostic info (row_count, lag_hours, etc.)
```

**Example Failure Output:**
```
VALIDATION_FAILED: Negative generation values found
Details: {
  "neg_y": 387,
  "by_series": [
    {"unique_id": "CALI_SUN", "count": 387, "min_y": -60, "max_y": -1}
  ],
  "sample": [10 rows with timestamps and values]
}
```

---

#### **modeling.py** - Forecasting Engine

**Class:** `RenewableForecastModel`

**Responsibilities:**
- Feature engineering (time + weather)
- Multi-model training via StatsForecast
- Cross-validation with time-series splits
- Probabilistic forecasting (prediction intervals)
- **Log transformation for non-negativity**

**Models Trained (4):**

| Model | Configuration | Purpose |
|-------|--------------|---------|
| `AutoARIMA` | `season_length=24` | Automatic ARIMA with hourly seasonality |
| `SeasonalNaive` | `season_length=24` | Baseline (repeat last season) |
| `AutoETS` | `season_length=24` | Exponential smoothing |
| `MSTL` | `seasons=[24, 168]` | **Best performer** - multi-seasonal (day + week) |

**Critical Decision: Why Log Transform?**

> **Problem:** ARIMA/MSTL are unconstrained linear models that can predict negative values.
> **Bad Solution:** Clamp predictions to 0 → **defensive coding** that masks the limitation.
> **Correct Solution:** Log transform:
> ```python
> # At training:
> y_log = log1p(y)  # log(y + 1)
>
> # At prediction:
> yhat = expm1(yhat_log)  # exp(yhat) - 1
> ```
> **Why This Works:** Mathematically guarantees `yhat >= 0` since `exp(x) > 0` for all x.

**Exception: Numerical Edge Case**

Small negatives (~-0.07) can occur when data contains zeros:
- `log1p(0) = 0` (zeros stay zeros in log-space)
- Model predicts slightly negative (e.g., -0.1) near zero
- `expm1(-0.1) ≈ -0.095`

**Handled with:** `np.maximum(0, expm1(x))`
**Classification:** This is NOT defensive coding - it's handling a known numerical edge case of the transform.

**Feature Engineering:**

```python
# Time features (cyclic encoding for stationarity)
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
dow_sin = sin(2π * day_of_week / 7)
dow_cos = cos(2π * day_of_week / 7)

# Weather regressors (7 variables)
temperature_2m, wind_speed_10m, wind_speed_100m,
wind_direction_10m, direct_radiation, diffuse_radiation, cloud_cover
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `prepare_training_df(gen_df, weather_df)` | Feature engineering + validation |
| `fit(gen_df, weather_df)` | Train all 4 models via StatsForecast |
| `cross_validate(gen_df, weather_df, n_windows, step)` | Time-series CV with adaptive logic |
| `predict(future_weather, best_model)` | Generate forecasts + intervals |
| `build_future_X_df(future_weather)` | Align forecast weather to horizon |

**Supporting Functions:**

| Function | Purpose |
|----------|---------|
| `compute_leaderboard(cv_df, confidence_levels)` | Model comparison (RMSE, MAE, coverage) |
| `compute_baseline_metrics(cv_df, model, k=2.0)` | Drift detection thresholds (mean + k*std) |
| `_enforce_hourly_grid(df, policy)` | Gap detection and enforcement |
| `_add_time_features(df)` | Cyclic time encoding |
| `_missing_hour_blocks(ds)` | Identify contiguous missing periods |

---

#### **tasks.py** - Pipeline Orchestration

**Class:** `RenewablePipelineConfig`

**Configuration Defaults:**
```python
regions: ["CALI", "ERCO", "MISO", "PJM", "SWPP"]
fuel_types: ["WND", "SUN"]
lookback_days: 30                    # Training window
horizon: 24                          # Forecast hours ahead
confidence_levels: (80, 95)          # Prediction intervals
cv_windows: 5                        # Cross-validation folds
cv_step_size: 168                    # 1 week in hours
data_dir: "data/renewable"
overwrite: False
```

**Pipeline Tasks (Sequential):**

| # | Task | Function | Output | Purpose |
|---|------|----------|--------|---------|
| 1 | Fetch Generation | `fetch_renewable_data()` | generation.parquet | EIA API → parquet |
| 2 | Fetch Weather | `fetch_renewable_weather()` | weather.parquet | Open-Meteo → parquet |
| 3 | Train Models | `train_renewable_models()` | cv_results, leaderboard, baseline | Model selection via CV |
| 4 | Generate Forecasts | `generate_renewable_forecasts()` | forecasts.parquet | 24h predictions |
| 5 | Compute Drift | `compute_renewable_drift()` | drift status dict | Model degradation detection |

**Main Orchestrator:** `run_full_pipeline(config) -> dict`

**Critical Decision: Per-Region Publishing Lag Handling**

**Problem:** EIA publishes data with different lags per region.

Example from 2026-01-19 run:
```
MISO: 04:00 UTC (earliest)
CALI: 05:00 UTC
ERCO: 06:00 UTC (latest, 2h ahead of MISO)
```

**Old Approach (BROKEN):**
```python
last_gen_ds = generation_df["ds"].max()  # Returns 06:00 (ERCO's max)
future_weather = weather_df[weather_df["ds"] > last_gen_ds]  # Only 07:00+
# FAIL: MISO needs weather from 05:00 onwards
```

**Fixed Approach (2026-01-20):**
```python
per_series_max = generation_df.groupby("unique_id")["ds"].max()
min_of_max = per_series_max.min()  # Use earliest series' max (04:00)
future_weather = weather_df[weather_df["ds"] > min_of_max]  # Covers all series
```

**Why This Works:** Each series gets forecast starting from its own `max(ds) + 1h`, and weather starts from the earliest series' max, ensuring no gaps.

---

#### **jobs/run_hourly.py** - Production Entry Point

**Main Orchestrator:** `run_hourly_pipeline()`

**Execution Flow:**
```
1. Load config from environment variables
   └─> Parse RENEWABLE_REGIONS, RENEWABLE_FUELS, etc.

2. Execute run_full_pipeline(config, fetch_diagnostics)
   └─> Tasks 1-4 (fetch, train, forecast)

3. Generate generation coverage summary
   └─> Per-series row counts, min/max timestamps

4. Run validation gate
   └─> validate_generation_df() with configurable thresholds

5. Check quality gates
   ├─> Rowdrop: curr_rows >= prev_rows * (1 - max_rowdrop_pct)
   └─> Neg Forecast: negative_ratio <= max_neg_forecast_ratio

6. Write run_log.json with results + diagnostics
   └─> Includes validation, quality gates, coverage

7. Exit with error code on failure
   └─> Enables GitHub Actions to detect failures
```

**Quality Gates:**

| Gate | Formula | Threshold | Purpose |
|------|---------|-----------|---------|
| **Rowdrop** | `curr_rows >= prev * 0.7` | 30% | Detect EIA API outages |
| **Neg Forecast** | `negative_ratio <= 0.10` | 10% | Detect model issues |

**Environment Variables (Configurable):**

```bash
# Data fetching
RENEWABLE_REGIONS="CALI,ERCO,MISO"
RENEWABLE_FUELS="WND,SUN"
LOOKBACK_DAYS=30

# Forecasting
RENEWABLE_HORIZON=24
RENEWABLE_CV_WINDOWS=2
RENEWABLE_CV_STEP_SIZE=168

# Validation
MAX_LAG_HOURS=48                  # ⚠️ Critical: EIA publishing delay
MAX_MISSING_RATIO=0.02            # 2% missing hours allowed

# Quality gates
MAX_ROWDROP_PCT=0.30             # 30% row drop allowed
MAX_NEG_FORECAST_RATIO=0.10      # 10% negative forecasts allowed

# Output
RENEWABLE_DATA_DIR="data/renewable"
```

---

### 2.3 GitHub Actions Workflow

**File:** `.github/workflows/renewable_hourly.yml`

**Trigger:**
- **Scheduled:** `cron: "17 * * * *"` (17 minutes past each hour)
- **Manual:** `workflow_dispatch` (GitHub UI button)

**Concurrency:** Single active run with `cancel-in-progress: true`

**Timeout:** 25 minutes

**Steps:**

```
1. Checkout code (actions/checkout@v4)
   └─> Latest main branch

2. Setup Python 3.11 (actions/setup-python@v5)
   └─> Consistent runtime environment

3. Validate EIA_API_KEY secret exists
   └─> Fail early if missing

4. Install dependencies
   └─> pandas, statsforecast, requests, pyarrow

5. Run pipeline (python -m src.renewable.jobs.run_hourly)
   └─> Hourly forecast generation
   └─> Exit code 0 (success) or 1 (failure)

6. Quality gate check (parse run_log.json)
   └─> Verify validation.ok == True
   └─> Verify quality_gates.rowdrop.ok == True
   └─> Verify quality_gates.neg_forecast.ok == True
   └─> Exit with code 1 if any gate fails

7. Summarize run (GitHub step summary)
   └─> Display validation status, lag_hours, max_ds

8. Commit artifacts
   ├─> generation.parquet
   ├─> weather.parquet
   ├─> forecasts.parquet
   └─> run_log.json
   └─> Git commit with message "renewable: hourly data update (UTC)"
   └─> Git push to main branch
```

**Critical Decision: Why Commit Artifacts?**

> **Version Control:** Track data changes over time (know what changed when)
> **Reproducibility:** Exact datasets used for each forecast (audit trail)
> **Debugging:** Historical data for investigating issues (no external DB needed)
> **Dashboard Data Source:** Streamlit reads committed parquets directly (simple architecture)
>
> **Trade-off:** Repo size grows, but Git handles this well with compression.
> **Mitigation:** Archive old artifacts periodically (manual cleanup).

---

### 2.4 Database & Dashboard

#### **db.py** - Persistence Layer (Optional)

**SQLite Schema:**

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `renewable_forecasts` | unique_id, ds, yhat, yhat_lo_80, yhat_hi_80, ... | Point + interval predictions |
| `renewable_scores` | unique_id, model, rmse, mae, coverage_80, ... | Model evaluation metrics |
| `weather_features` | region, ds, temperature_2m, wind_speed_10m, ... | Historical + forecast weather |
| `drift_alerts` | alert_at, unique_id, current_rmse, threshold_rmse, ... | Drift detection events |
| `baseline_metrics` | unique_id, model, drift_threshold_rmse, ... | CV-derived thresholds (JSON) |

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `init_renewable_db(db_path)` | Create schema if not exists |
| `save_forecasts(db_path, df, run_id, model)` | Batch insert predictions |
| `save_weather(db_path, df, is_forecast)` | Weather data persistence |
| `save_drift_alert(db_path, uid, rmse, threshold)` | Alert logging |
| `get_recent_forecasts(db_path, hours=48)` | Query by recency |
| `get_drift_alerts(db_path, hours=24, severity)` | Alert queries |

**Note:** Database is optional - dashboard primarily reads from committed parquets.

---

#### **dashboard.py** - Streamlit Visualization

**UI Tabs:**

| Tab | Visualizations | Purpose |
|-----|----------------|---------|
| **Forecasts** | Line plot with 80%/95% prediction intervals | Interactive forecast viewer |
| **Drift Monitor** | Alert timeline + metric comparisons | Model degradation tracking |
| **Coverage** | Empirical vs nominal interval coverage | Calibration analysis |
| **Weather** | Regional weather variable heatmaps | Regressor inspection |

**Data Sources (Priority Order):**

1. **Pipeline parquets** (generation.parquet, weather.parquet, forecasts.parquet)
2. **SQLite database** (fallback if parquets missing)
3. **Demo data** (fallback for display)

**Launch:**
```bash
streamlit run src/renewable/dashboard.py
```

**Dashboard Features:**
- Region filter (multi-select)
- Fuel type filter (WND/SUN)
- Date range selector
- Downloadable forecast table (planned)
- Residual analysis plots (planned)

---

## 3. Step-by-Step Pipeline Flow with Decision Points

### Stage 1: Data Ingestion

```
┌─────────────────────────────────────────────────┐
│ 1a. Fetch EIA Generation Data                   │
│     (eia_renewable.py)                          │
├─────────────────────────────────────────────────┤
│ • Multi-region parallel fetch (ThreadPool)      │
│ • Pagination handling (5000 records/request)    │
│ • Rate limiting (5 req/sec)                     │
│ • CLAMP negatives to 0 (data cleaning)         │
│ • Track diagnostics per region                  │
└─────────────────────────────────────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │ Decision: Why Clamp vs Filter?     │
    ├────────────────────────────────────┤
    │ CLAMP: Preserves hourly grid       │
    │   └─> Modeling requires complete   │
    │       hourly sequence               │
    │ FILTER: Creates gaps                │
    │   └─> _enforce_hourly_grid() fails │
    │                                    │
    │ Choice: CLAMP (data cleaning)      │
    │                                    │
    │ Classification:                    │
    │ ✓ Data cleaning (bad upstream)    │
    │ ✗ NOT defensive coding             │
    └─────────────────┬──────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ 1b. Fetch Weather Data                          │
│     (open_meteo.py)                             │
├─────────────────────────────────────────────────┤
│ • Historical: archive-api.open-meteo.com        │
│   └─> Training data (no leakage)               │
│ • Forecast: api.open-meteo.com                  │
│   └─> Prediction data (realistic)              │
│ • Socket timeout retry (connect=3, read=3)     │
│ • Exponential backoff (1s, 2s, 4s)             │
│ • Combine & deduplicate (keep last)            │
└─────────────────────────────────────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │ Decision: Why Two Endpoints?       │
    ├────────────────────────────────────┤
    │ Historical:                        │
    │ • Training data only               │
    │ • No future information leakage    │
    │                                    │
    │ Forecast:                          │
    │ • Realistic prediction data        │
    │ • Weather forecasts available IRL  │
    │                                    │
    │ Choice: Combine both endpoints     │
    │                                    │
    │ Trade-off:                         │
    │ ✓ Prevents leakage                │
    │ ✓ Realistic predictions            │
    │ ✗ More API calls (2x)              │
    └─────────────────┬──────────────────┘
                      │
                      ▼
         ✅ Save generation.parquet
         ✅ Save weather.parquet
```

---

### Stage 2: Validation & Quality Gates

```
┌─────────────────────────────────────────────────┐
│ 2a. Generation Data Validation                   │
│     (validation.py: validate_generation_df)     │
├─────────────────────────────────────────────────┤
│ 1. ✓ Required columns (unique_id, ds, y)       │
│ 2. ✓ Non-empty DataFrame                       │
│ 3. ✓ DateTime parsing (pd.to_datetime)         │
│ 4. ✓ Numeric parsing (pd.to_numeric)           │
│ 5. ⚠️ CRITICAL: No negatives (generation)      │
│ 6. ✓ No duplicates (unique_id, ds) pairs       │
│ 7. ✓ Expected series all present               │
│ 8. ⚠️ CRITICAL: Freshness (lag <= 48h)        │
│ 9. ✓ Per-series freshness (no stale series)    │
│ 10. ✓ Hourly grid completeness (max 2% gaps)  │
└─────────────────────────────────────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │ Decision: Why 48h Lag Threshold?   │
    ├────────────────────────────────────┤
    │ EIA Reality:                       │
    │ • Typical delay: 12-24h            │
    │ • Worst-case: 36-48h               │
    │                                    │
    │ Initial (3h):                      │
    │ • False failures                   │
    │ • Pipeline unstable                │
    │                                    │
    │ Fixed (48h):                       │
    │ • Matches operational reality      │
    │ • Reduces false positives          │
    │                                    │
    │ Choice: 48h (operational reality)  │
    │                                    │
    │ Evidence:                          │
    │ • 2026-01-19: 22h lag (normal)     │
    │ • 2026-01-20: MISO 2h behind ERCO  │
    └─────────────────┬──────────────────┘
                      │
                      ▼ FAIL → Exit(1)
                      │     Write run_log.json
                      │     Show validation error
                      │
                      ▼ PASS
         ✅ Continue to Model Training
```

---

### Stage 3: Model Training & Forecasting

```
┌─────────────────────────────────────────────────┐
│ 3a. Feature Engineering                          │
│     (modeling.py: prepare_training_df)          │
├─────────────────────────────────────────────────┤
│ • Merge generation + weather (left join)        │
│   └─> Fail-loud if weather missing (no fill)   │
│ • Add time features (hour_sin, hour_cos, ...)  │
│ • LOG TRANSFORM: y → log1p(y)                   │
│   └─> Ensure y >= 0 (validation already done)  │
│ • Enforce hourly grid (detect gaps)             │
│ • Split into train/validation (if CV)           │
└─────────────────────────────────────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │ Decision: Why Log Transform?       │
    ├────────────────────────────────────┤
    │ Problem:                           │
    │ • ARIMA/MSTL predict negatives     │
    │ • Example: ERCO_WND → -10k MWh     │
    │                                    │
    │ Bad Solution: Clamp predictions    │
    │ • Classification: Defensive coding │
    │ • Masks model limitation           │
    │                                    │
    │ Correct Solution: Log transform    │
    │ • y_log = log1p(y)                │
    │ • yhat = expm1(yhat_log)          │
    │ • Guarantees: yhat >= 0           │
    │ • Works WITH model mathematics     │
    │                                    │
    │ Choice: log1p/expm1 transform      │
    │                                    │
    │ Edge Case: Zeros                   │
    │ • log1p(0) = 0                    │
    │ • Model predicts -0.1 in log space│
    │ • expm1(-0.1) ≈ -0.095            │
    │ • Handle: np.maximum(0, expm1(x)) │
    │ • NOT defensive (known edge case) │
    └─────────────────┬──────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ 3b. Model Training                               │
│     (modeling.py: fit)                          │
├─────────────────────────────────────────────────┤
│ • AutoARIMA (seasonal=24)                       │
│   └─> Automatic ARIMA with hourly seasonality  │
│ • SeasonalNaive (seasonal=24)                   │
│   └─> Baseline (repeat last 24h)               │
│ • AutoETS (seasonal=24)                         │
│   └─> Exponential smoothing                    │
│ • MSTL (seasons=[24, 168])  ← Best performer   │
│   └─> Multi-seasonal (daily + weekly)          │
│                                                 │
│ StatsForecast handles:                          │
│ • Model selection (auto hyperparameters)        │
│ • Multi-series training (vectorized)            │
│ • Prediction intervals (quantile regression)    │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ 3c. Cross-Validation                             │
│     (modeling.py: cross_validate)               │
├─────────────────────────────────────────────────┤
│ • Time-series CV (n_windows=2, step=168h)      │
│   └─> Adaptive logic based on min_series_len   │
│ • Compute leaderboard                           │
│   ├─> RMSE (point accuracy)                    │
│   ├─> MAE (robust to outliers)                 │
│   ├─> MAPE (NOT used for solar - div by 0)    │
│   ├─> Coverage_80 (empirical interval coverage)│
│   └─> Coverage_95 (empirical interval coverage)│
│ • Select best model (typically MSTL)           │
│ • Compute baseline thresholds for drift        │
│   └─> threshold = mean + 2.0 * std             │
└─────────────────────────────────────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │ Decision: Why 2 CV Windows?        │
    ├────────────────────────────────────┤
    │ Trade-off:                         │
    │ • More windows: Better reliability │
    │ • Fewer windows: Faster execution  │
    │                                    │
    │ Hourly Pipeline:                   │
    │ • Need fast execution (<25min)     │
    │ • 2 windows = ~5-10min             │
    │ • 5 windows = ~15-25min            │
    │                                    │
    │ Adaptive Logic:                    │
    │ n = max(2, min(cv_windows,        │
    │         avail / step_size))       │
    │                                    │
    │ Choice: 2 windows (speed priority) │
    │                                    │
    │ Note: Could increase to 3-5 for   │
    │       better reliability if needed │
    └─────────────────┬──────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ 3d. Generate Forecasts                           │
│     (modeling.py: predict)                      │
├─────────────────────────────────────────────────┤
│ • Build future X DataFrame                      │
│   ├─> Create ds grid: [max(ds)+1h, ..., +24h] │
│   ├─> Add time features (hour_sin, dow_cos)    │
│   └─> Merge forecast weather (left join)       │
│ • Per-region lag handling (min of max)         │
│   └─> Ensure MISO has weather from 05:00      │
│ • Predict with best model (MSTL)               │
│ • INVERSE TRANSFORM: expm1(yhat_log)            │
│ • Edge case handling: np.maximum(0, expm1(x))  │
│ • Rename columns: model-specific → yhat        │
└─────────────────────────────────────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │ Decision: Per-Region Lag Handling  │
    ├────────────────────────────────────┤
    │ Problem (2026-01-20):              │
    │ • MISO: 04:00 UTC (earliest)       │
    │ • CALI: 05:00 UTC                  │
    │ • ERCO: 06:00 UTC (latest)         │
    │                                    │
    │ Old (BROKEN):                      │
    │ last_gen_ds = df["ds"].max()      │
    │ # Returns 06:00 (ERCO)             │
    │ future_weather = weather[ds>06:00] │
    │ # Only 07:00+ available            │
    │ # FAIL: MISO needs 05:00           │
    │                                    │
    │ Fixed:                             │
    │ per_series_max = groupby.max()     │
    │ min_of_max = per_series_max.min()  │
    │ # Returns 04:00 (MISO)             │
    │ future_weather = weather[ds>04:00] │
    │ # Covers all series                │
    │                                    │
    │ Choice: min(per_series_max)        │
    │                                    │
    │ Why This Works:                    │
    │ • Each series forecasts from own max│
    │ • Weather covers earliest max      │
    │ • No gaps, no leakage              │
    └─────────────────┬──────────────────┘
                      │
                      ▼
         ✅ Save forecasts.parquet
            (yhat, yhat_lo_80, yhat_hi_80,
             yhat_lo_95, yhat_hi_95)
```

---

### Stage 4: Quality Gates & Artifacts

```
┌─────────────────────────────────────────────────┐
│ 4a. Quality Gates                                │
│     (run_hourly.py)                             │
├─────────────────────────────────────────────────┤
│ • Rowdrop Gate:                                 │
│   ├─> curr_rows >= prev_rows * 0.7             │
│   ├─> Purpose: Detect EIA API outages          │
│   └─> Threshold: 30% drop allowed              │
│                                                 │
│ • Negative Forecast Gate:                       │
│   ├─> negative_ratio <= 0.10                   │
│   ├─> Purpose: Detect model issues             │
│   └─> Threshold: 10% negatives allowed         │
│       (should be ~0% with log transform)       │
└─────────────────────────────────────────────────┘
                      │
                      ▼ FAIL → Exit(1)
                      │     Write run_log.json
                      │     Show gate failure
                      │
                      ▼ PASS
┌─────────────────────────────────────────────────┐
│ 4b. Write Run Log                                │
│     (run_hourly.py)                             │
├─────────────────────────────────────────────────┤
│ run_log.json:                                   │
│ {                                               │
│   "run_at_utc": "2026-01-20T17:15:00Z",       │
│   "config": {regions, fuels, horizon, ...},    │
│   "pipeline_results": {                        │
│     "generation_rows": 4320,                   │
│     "series_count": 6,                         │
│     "best_model": "MSTL",                      │
│     "best_rmse": 1234.5                        │
│   },                                           │
│   "validation": {                              │
│     "ok": true,                                │
│     "message": "OK",                           │
│     "details": {lag_hours: 22.0, ...}         │
│   },                                           │
│   "quality_gates": {                           │
│     "rowdrop": {ok: true, ...},                │
│     "neg_forecast": {ok: true, ...}            │
│   },                                           │
│   "diagnostics": {fetch: [...], coverage: ...} │
│ }                                              │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ 4c. Commit Artifacts                             │
│     (GitHub Actions)                            │
├─────────────────────────────────────────────────┤
│ git add:                                        │
│ • data/renewable/generation.parquet             │
│ • data/renewable/weather.parquet                │
│ • data/renewable/forecasts.parquet              │
│ • data/renewable/run_log.json                   │
│                                                 │
│ git commit -m "renewable: hourly data update"  │
│ git push origin main                            │
└─────────────────────────────────────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │ Decision: Why Commit Artifacts?    │
    ├────────────────────────────────────┤
    │ Benefits:                          │
    │ • Version control data changes     │
    │   └─> Know what changed when       │
    │ • Reproducibility                  │
    │   └─> Exact datasets per forecast  │
    │ • Debugging                        │
    │   └─> Historical data available    │
    │ • Dashboard data source            │
    │   └─> Simple architecture (no DB)  │
    │                                    │
    │ Trade-offs:                        │
    │ ✓ Simple (no external DB)          │
    │ ✓ Audit trail (Git history)       │
    │ ✗ Repo size growth                 │
    │   └─> Mitigate: Archive old data  │
    │                                    │
    │ Choice: Git-based artifact store   │
    │                                    │
    │ Alternative: Could use S3/GCS if   │
    │             repo size becomes issue│
    └─────────────────┬──────────────────┘
                      │
                      ▼
         ✅ Pipeline Complete
            Artifacts committed
            Dashboard can visualize
```

---

## 4. Future Todos & Improvements

### High Priority (Production Enhancements)

#### 4.1 Model Improvements

- [ ] **Ensemble forecasting**
  - Combine MSTL + AutoARIMA weighted average
  - Rationale: Reduce variance, improve robustness
  - Implementation: `forecast = 0.7*mstl + 0.3*arima`
  - Expected impact: 5-10% RMSE improvement

- [ ] **Adaptive horizon**
  - Extend to 48h/72h forecasts
  - Rationale: Energy traders need longer visibility
  - Challenge: Weather forecast accuracy degrades
  - Implementation: Train separate models per horizon

- [ ] **Region-specific models**
  - Train separate models per region instead of multi-series
  - Rationale: Capture region-specific patterns (MISO solar night issue)
  - Trade-off: More models = slower training
  - Implementation: Loop over regions, train individually

- [ ] **Hyperparameter tuning**
  - Grid search for MSTL/AutoARIMA parameters
  - Rationale: Current uses defaults, may be suboptimal
  - Parameters: `season_length`, `trend_forecaster`, `decomposition_type`
  - Implementation: Use `optuna` or `scikit-optimize`

**Overall Rationale:** Current single-model approach may underfit regional patterns and could benefit from ensemble methods.

---

#### 4.2 Feature Engineering

- [ ] **Additional weather variables**
  - Humidity, precipitation, pressure
  - Rationale: May improve wind/solar prediction
  - Data source: Open-Meteo already provides these
  - Implementation: Add to `WEATHER_VARS` list

- [ ] **Lagged features**
  - Previous day generation (t-24, t-168)
  - Rationale: Capture autoregressive patterns
  - Implementation: `df['y_lag24'] = df.groupby('unique_id')['y'].shift(24)`
  - Challenge: Handle missing lags at series start

- [ ] **Calendar features**
  - Holidays, weekend flags, month
  - Rationale: Energy usage patterns differ on holidays
  - Data source: `holidays` Python library
  - Implementation: Add boolean columns to training data

- [ ] **Interaction terms**
  - Temperature × wind_speed, cloud_cover × radiation
  - Rationale: Non-linear relationships exist
  - Implementation: `df['temp_wind'] = df['temp'] * df['wind']`
  - Challenge: Feature explosion (manage dimensionality)

**Overall Rationale:** More features could capture non-linear relationships and improve forecast accuracy.

---

#### 4.3 Data Quality

- [ ] **Automated outlier detection**
  - Z-score or IQR-based flagging
  - Rationale: Detect EIA data anomalies beyond negatives
  - Implementation: Flag values > 3 std deviations from mean
  - Action: Log outliers, optionally exclude from training

- [ ] **Missing data imputation**
  - Forward-fill vs interpolation analysis
  - Rationale: Better than dropping incomplete series
  - Comparison: Test RMSE with different imputation strategies
  - Current: Raises error on missing hours (strict)

- [ ] **EIA API backup**
  - Alternative data source for outages
  - Rationale: Single point of failure (EIA API)
  - Options: ERCOT/ISO direct APIs, paid data providers
  - Implementation: Fallback fetch logic with priority ordering

- [ ] **Weather API fallback**
  - Secondary weather provider
  - Rationale: Open-Meteo occasionally times out
  - Options: OpenWeatherMap, Visual Crossing, NOAA
  - Implementation: Try Open-Meteo first, fallback on error

**Overall Rationale:** Improve robustness to upstream failures and data quality issues.

---

#### 4.4 Monitoring & Alerting

- [ ] **Slack/email alerts**
  - Drift detection notifications
  - Rationale: Proactive issue awareness
  - Trigger: When `current_rmse > threshold_rmse`
  - Implementation: GitHub Actions → Slack webhook

- [ ] **Prometheus metrics**
  - Pipeline execution time, error rates
  - Rationale: Track operational health
  - Metrics: `pipeline_duration_seconds`, `fetch_errors_total`
  - Implementation: Export from run_hourly.py

- [ ] **Model performance tracking**
  - Time-series database (InfluxDB)
  - Rationale: Track RMSE/MAE over time
  - Implementation: Save metrics after each run
  - Visualization: Grafana dashboard

- [ ] **Forecast accuracy dashboard**
  - Retrospective analysis (actual vs forecast)
  - Rationale: Understand where models fail
  - Implementation: Join forecasts with next day's actuals
  - Visualization: Streamlit tab with error plots

**Overall Rationale:** Operational visibility for production system to detect issues early.

---

### Medium Priority (Usability & Extensibility)

#### 4.5 Dashboard Enhancements

- [ ] **Interactive region selection**
  - Multi-region comparison plots
  - Implementation: Plotly with checkbox filters
  - Use case: Compare CALI_SUN vs ERCO_SUN forecasts

- [ ] **Downloadable forecasts**
  - CSV export functionality
  - Implementation: Streamlit `st.download_button`
  - Format: Tidy format with all columns

- [ ] **Historical accuracy plots**
  - Actual vs forecast over time
  - Implementation: Rolling 7-day window
  - Visualization: Line plot with shaded intervals

- [ ] **Residual analysis**
  - ACF/PACF of forecast errors
  - Purpose: Diagnose model misspecification
  - Implementation: `statsmodels` autocorrelation functions

**Overall Rationale:** Better user experience for stakeholders and data scientists.

---

#### 4.6 Testing

- [ ] **Integration tests**
  - Full pipeline end-to-end
  - Implementation: `tests/renewable/test_integration.py`
  - Coverage: Fetch → Train → Forecast → Validate

- [ ] **Property-based tests**
  - Hypothesis for validation logic
  - Example: "Negative values always rejected"
  - Implementation: `@given(st.dataframes(...))`

- [ ] **Load tests**
  - Simulate multi-region scale-up
  - Purpose: Test with 50+ regions
  - Implementation: Locust or custom script

- [ ] **Mock API responses**
  - Comprehensive test fixtures
  - Current: Minimal mocking in test_smoke.py
  - Implementation: Save real API responses as fixtures

**Overall Rationale:** Current test coverage is minimal (~30%). Need >80% for production confidence.

---

#### 4.7 Configuration Management

- [ ] **YAML config files**
  - Replace env vars with structured config
  - Benefits: Type checking, validation, versioning
  - Implementation: `config/renewable.yaml` + `pydantic`

- [ ] **Multi-environment support**
  - Dev/staging/prod configs
  - Example: `config/renewable.dev.yaml`
  - Implementation: `ENV=prod python -m src.renewable.jobs.run_hourly`

- [ ] **Secret management**
  - Migrate to vault/secrets manager
  - Current: GitHub Secrets (acceptable)
  - Future: AWS Secrets Manager, HashiCorp Vault

- [ ] **Feature flags**
  - Toggle experimental features
  - Example: `USE_ENSEMBLE_FORECASTS=false`
  - Implementation: LaunchDarkly or simple config flags

**Overall Rationale:** Easier configuration management at scale and across environments.

---

### Low Priority (Research & Experimentation)

#### 4.8 Advanced Models

- [ ] **Deep learning** - LSTM/Transformer for long sequences
- [ ] **Causal inference** - Counterfactual weather scenarios
- [ ] **Probabilistic programming** - Bayesian hierarchical models (PyMC)
- [ ] **Online learning** - Incremental model updates without retraining

**Overall Rationale:** Explore state-of-art approaches for potential accuracy gains.

---

#### 4.9 Visualization

- [ ] **Geographic heatmaps** - Regional generation overlays on US map
- [ ] **Animated forecasts** - Time-lapse visualization of forecast evolution
- [ ] **Uncertainty quantification** - Fan charts for prediction intervals

**Overall Rationale:** Improve interpretability for non-technical stakeholders.

---

## 5. Known Issues & Areas Requiring Fixes

### Critical Issues (Immediate Attention Required)

#### 5.1 MISO_SUN Night Value Inconsistency

**Status:** ⚠️ Documented, not fixed

**Problem:**
EIA API reports small positive values (1-132 MWh) for MISO solar at night instead of zeros.
Training data has **0% zeros** at night (should be ~68% like CALI/ERCO).

**Impact:**
Model forecasts 9-15 MWh at night for MISO_SUN (learns from bad training data).
Chart appears flat because y-axis scales to 60k (9-15 MWh invisible).

**Root Cause:**
EIA data quality issue (upstream source problem).

**Evidence:**
```
CALI_SUN: 68.5% zeros at night (correct)
ERCO_SUN: 66.8% zeros at night (correct)
MISO_SUN: 0% zeros at night (EIA issue)
```

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| 1. Accept as-is | No code changes | Bad forecasts for MISO_SUN |
| 2. Contact EIA | Fixes upstream | Slow, may not respond |
| 3. Post-process MISO_SUN | Correct forecasts | Adds region-specific logic |
| 4. Separate validation | Flag for review | Manual intervention needed |

**Recommendation:** Option 3 (post-process) + Option 2 (report to EIA)

**Implementation:**
```python
# In eia_renewable.py:fetch_region()
if region == "MISO" and fuel_type == "SUN":
    # Zero out night hours in Central Time (UTC-6)
    # Night: 00:00-07:00 Central = 06:00-13:00 UTC
    night_mask = (df["ds"].dt.hour >= 6) & (df["ds"].dt.hour <= 13)
    zero_count = night_mask.sum()
    df.loc[night_mask, "y"] = 0
    print(f"[MISO_SUN] Zeroed {zero_count} night values (EIA data quality fix)")
```

**Classification:** This is **data cleaning** (correcting bad upstream data), NOT defensive coding.

---

#### 5.2 Weather API Timeout Handling

**Status:** 🔶 Partially fixed (2026-01-19)

**Current Implementation:**
- 3 retries with exponential backoff
- 60s timeout per request
- Socket timeout retry (`connect=3, read=3`)

**Remaining Issue:**
If **all** regions fail (Open-Meteo complete outage), pipeline crashes with generic error:
```
RuntimeError: [weather][ALIGN] Missing weather after merge rows=4320
```

**User-Facing Problem:**
Error doesn't tell user that Open-Meteo is down (appears as code bug).

**Fix:** Add better error message.

```python
# In open_meteo.py:fetch_all_regions_historical()
if not all_dfs:
    raise RuntimeError(
        "[fetch_all_regions_historical] Failed to fetch weather for ALL regions.\n"
        "This suggests Open-Meteo API is experiencing an outage.\n"
        "Recommended actions:\n"
        "1. Wait 5-10 minutes and retry\n"
        "2. Check Open-Meteo status: https://open-meteo.com/\n"
        "3. Use cached weather if available (set overwrite=False)\n"
        "4. Consider fallback weather provider (see FUTURE_TODOS.md)"
    )
```

---

#### 5.3 CV Adaptive Logic Can Be Too Aggressive

**Status:** ⚠️ Known issue

**Problem:**
When `min_series_len` is small (e.g., 200 rows), `n_windows` can drop to 2, making CV unreliable.

**Current Logic:**
```python
n_windows = max(2, min(config.cv_windows, available_for_cv // step_size))
```

**Example Failure Case:**
```
min_series_len = 200 rows
horizon = 24
available_for_cv = 200 - 24 = 176 rows
step_size = 168 hours (1 week)
n_windows = 176 / 168 = 1.04 → max(2, 1) = 2 windows
```

With only 2 windows, CV estimates are unreliable (high variance).

**Fix:** Increase minimum to 3 windows.

```python
# In tasks.py:train_renewable_models()
n_windows = max(2, min(config.cv_windows, available_for_cv // step_size))
# CHANGE TO:
n_windows = max(3, min(config.cv_windows, available_for_cv // step_size))
```

**Alternative:** Reduce `step_size` dynamically:
```python
step_size = min(config.cv_step_size, max(24, available_for_cv // 4))
n_windows = max(3, available_for_cv // step_size)
```

---

### Medium Issues (Plan to Fix)

#### 5.4 Dashboard Caching

**Problem:** Streamlit reloads parquets on every widget interaction (slow for large files).

**Current:** No caching, reads from disk every time.

**Fix:** Add Streamlit cache decorator.

```python
@st.cache_data
def load_generation_data():
    return pd.read_parquet("data/renewable/generation.parquet")

@st.cache_data
def load_forecasts():
    return pd.read_parquet("data/renewable/forecasts.parquet")
```

**Expected Impact:** 5-10x faster dashboard interactions.

---

#### 5.5 Run Log JSON Size Growth

**Problem:** `run_log.json` includes full diagnostics, can grow to 100KB+ over time.

**Current:** Overwrites previous run (only latest kept).

**Alternative Approaches:**

| Approach | Pros | Cons |
|----------|------|------|
| Keep only latest | Simple, current behavior | No history |
| Append-only log file | Full history | File grows indefinitely |
| Archive old logs | Balance size/history | More complex |
| Database storage | Queryable history | Requires DB setup |

**Recommendation:** Archive old logs monthly.

```bash
# In GitHub Actions, add archival step
mkdir -p data/renewable/archive
mv data/renewable/run_log.json \
   data/renewable/archive/run_log_$(date +%Y%m).json
```

---

#### 5.6 No Rollback on Pipeline Failure

**Problem:** If forecast generation fails after fetch, partial artifacts remain (generation.parquet exists, forecasts.parquet missing).

**Current:** No transactional behavior.

**Impact:** Dashboard may show stale data or error on missing file.

**Fix:** Implement transaction-like behavior.

```python
# Pseudocode for run_hourly.py
def run_hourly_pipeline():
    # Stage artifacts in temporary directory
    temp_dir = Path("data/renewable/.temp")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Run pipeline, write to temp
        ...

        # If successful, move temp -> prod (atomic)
        shutil.move(temp_dir / "generation.parquet",
                    data_dir / "generation.parquet")
        shutil.move(temp_dir / "forecasts.parquet",
                    data_dir / "forecasts.parquet")
    except Exception as e:
        # On error, clean up temp (no partial artifacts)
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
```

---

### Low Priority Issues (Document Only)

#### 5.7 FutureWarnings in Dependencies

**Status:** 🟢 Mostly fixed

**Fixed (2026-01-19):**
- pandas `floor("H")` → `floor("h")` in validation.py
- pandas `groupby.apply(...)` → `groupby.apply(..., include_groups=False)` in modeling.py

**Remaining:**
StatsForecast may emit warnings on newer pandas versions (pandas 2.x).

**Action:** Monitor StatsForecast releases and update when fix is available.

---

#### 5.8 Timezone Handling Complexity

**Status:** 🟡 Working but brittle

**Current Implementation:**
Everything converted to naive UTC timestamps (no timezone info attached).

**Rationale:**
- Simplifies date arithmetic (no DST issues)
- EIA data already in UTC
- Open-Meteo allows UTC specification

**Risk:**
If EIA changes timezone handling, logic may break silently.

**Example Failure Mode:**
```python
# If EIA starts returning local time instead of UTC
df["ds"] = pd.to_datetime(df["ds"])  # Assumes UTC but gets EST
# Merge with weather fails (off by 5-8 hours)
```

**Mitigation:**
Add comprehensive tests for timezone edge cases.

```python
def test_timezone_consistency():
    """Verify all timestamps are naive UTC."""
    gen_df = load_generation_data()
    weather_df = load_weather_data()

    assert gen_df["ds"].dt.tz is None, "Generation ds should be naive UTC"
    assert weather_df["ds"].dt.tz is None, "Weather ds should be naive UTC"

    # Verify timestamps align (no offset)
    merged = gen_df.merge(weather_df, on=["ds", "region"], how="inner")
    assert len(merged) > 0, "Merge should succeed if timezones aligned"
```

---

## 6. Critical File Locations

### 6.1 Core Pipeline

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/renewable/tasks.py` | Main orchestration | `run_full_pipeline()`, `fetch_renewable_data()`, `train_renewable_models()` |
| `src/renewable/modeling.py` | Forecasting models | `RenewableForecastModel.fit()`, `.predict()`, `.cross_validate()` |
| `src/renewable/jobs/run_hourly.py` | Production entry | `run_hourly_pipeline()`, quality gates |

### 6.2 Data Sources

| File | Purpose | API Endpoint |
|------|---------|--------------|
| `src/renewable/eia_renewable.py` | Generation data | EIA API v2 |
| `src/renewable/open_meteo.py` | Weather data | Open-Meteo archive + forecast APIs |
| `src/renewable/regions.py` | Region metadata | N/A (static registry) |

### 6.3 Quality & Validation

| File | Purpose | Validation Steps |
|------|---------|------------------|
| `src/renewable/validation.py` | Data quality checks | 10-step validation checklist |
| `src/renewable/jobs/run_hourly.py` | Quality gates | Rowdrop, neg forecast gates |

### 6.4 CI/CD & Automation

| File | Purpose | Trigger |
|------|---------|---------|
| `.github/workflows/renewable_hourly.yml` | Hourly automation | Cron + manual |
| `.github/workflows/pre-commit.yml` | Pre-commit tests | Git hook |

### 6.5 Documentation

| File | Purpose |
|------|---------|
| `RENEWABLE_PIPELINE_ARCHITECTURE.md` | This file - complete architecture docs |
| `chapters/renewable_energy_forecasting.ipynb` | Tutorial notebook |
| `DEVLOG.md` | Development history + investigation logs |
| `README.md` | Project overview (update with link to this file) |

### 6.6 Tests

| File | Purpose | Coverage |
|------|---------|----------|
| `tests/renewable/test_smoke.py` | Mocked smoke tests | Validation, EIA fetch |
| `tests/renewable/test_renewable.py` | Integration tests | End-to-end (planned) |

### 6.7 Data Artifacts (Git-Tracked)

| File | Purpose | Update Frequency |
|------|---------|------------------|
| `data/renewable/generation.parquet` | Training data | Hourly |
| `data/renewable/weather.parquet` | Historical + forecast weather | Hourly |
| `data/renewable/forecasts.parquet` | 24h predictions | Hourly |
| `data/renewable/run_log.json` | Pipeline diagnostics | Hourly |

---

## 7. Quick Start Guide

### 7.1 Development Setup

```bash
# Clone repo
git clone https://github.com/YOUR_ORG/atsaf.git
cd atsaf

# Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set EIA API key
export EIA_API_KEY="your_key_here"  # Windows: set EIA_API_KEY=...
```

### 7.2 Run Pipeline Locally

```bash
# Full pipeline (fetch + train + forecast)
python -m src.renewable.jobs.run_hourly

# Custom config via env vars
export RENEWABLE_REGIONS="CALI,ERCO"
export LOOKBACK_DAYS=14
python -m src.renewable.jobs.run_hourly
```

### 7.3 Launch Dashboard

```bash
streamlit run src/renewable/dashboard.py
# Opens at http://localhost:8501
```

### 7.4 Run Tests

```bash
# Smoke tests (fast, mocked)
pytest tests/renewable/test_smoke.py -v

# All tests
pytest tests/renewable/ -v

# With coverage
pytest tests/renewable/ --cov=src.renewable --cov-report=html
```

### 7.5 Jupyter Notebook Tutorial

```bash
jupyter notebook chapters/renewable_energy_forecasting.ipynb
```

### 7.6 GitHub Actions (Production)

Runs automatically every hour at :17 (17 * * * *)

Manual trigger:
```
1. Go to Actions tab
2. Select "renewable-hourly-update"
3. Click "Run workflow"
```

---

## Appendices

### A. Environment Variables Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `EIA_API_KEY` | **(required)** | EIA API authentication |
| `RENEWABLE_REGIONS` | `CALI,ERCO,MISO` | Comma-separated region codes |
| `RENEWABLE_FUELS` | `WND,SUN` | Comma-separated fuel types |
| `LOOKBACK_DAYS` | `30` | Training window |
| `RENEWABLE_HORIZON` | `24` | Forecast hours ahead |
| `RENEWABLE_CV_WINDOWS` | `2` | Cross-validation folds |
| `RENEWABLE_CV_STEP_SIZE` | `168` | CV step size (hours) |
| `MAX_LAG_HOURS` | `48` | Freshness threshold |
| `MAX_MISSING_RATIO` | `0.02` | Missing hours tolerance |
| `MAX_ROWDROP_PCT` | `0.30` | Rowdrop gate threshold |
| `MAX_NEG_FORECAST_RATIO` | `0.10` | Neg forecast gate threshold |
| `RENEWABLE_DATA_DIR` | `data/renewable` | Output directory |

### B. Data Format Specifications

**Generation Data:**
```
DataFrame columns: [unique_id, ds, y]
unique_id: str (format: "{region}_{fuel_type}")
ds: pandas Timestamp (naive UTC)
y: float (MWh, >= 0)
```

**Weather Data:**
```
DataFrame columns: [ds, region, temperature_2m, wind_speed_10m, ...]
ds: pandas Timestamp (naive UTC)
region: str (CALI, ERCO, etc.)
{weather_vars}: float
```

**Forecast Data:**
```
DataFrame columns: [unique_id, ds, yhat, yhat_lo_80, yhat_hi_80, yhat_lo_95, yhat_hi_95]
unique_id: str
ds: pandas Timestamp (naive UTC)
yhat: float (point forecast)
yhat_lo_*, yhat_hi_*: float (prediction intervals)
```

### C. Glossary

| Term | Definition |
|------|------------|
| **EIA** | U.S. Energy Information Administration |
| **CALI** | California (CAISO) |
| **ERCO** | ERCOT (Texas) |
| **MISO** | Midcontinent ISO |
| **PJM** | Pennsylvania-New Jersey-Maryland Interconnection |
| **SWPP** | Southwest Power Pool |
| **WND** | Wind generation |
| **SUN** | Solar generation |
| **ARIMA** | Autoregressive Integrated Moving Average |
| **MSTL** | Multiple Seasonal-Trend decomposition using Loess |
| **ETS** | Error-Trend-Seasonal exponential smoothing |
| **CV** | Cross-validation |
| **RMSE** | Root Mean Squared Error |
| **MAE** | Mean Absolute Error |
| **Drift** | Model degradation over time |
| **Exogenous** | External variables (weather) |
| **Naive UTC** | Timestamp without timezone info (assumed UTC) |
| **Log Transform** | `y_log = log1p(y)` to ensure non-negative forecasts |

---

**Document End**

*For questions or contributions, see `CONTRIBUTING.md` or open an issue on GitHub.*
