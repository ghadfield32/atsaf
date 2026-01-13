# Chapter 2 — Experimentation & Backtesting

## Outcomes (what I can do after this)

- [ ] I can set up and run rolling-origin cross-validation on time-series data
- [ ] I can build a leaderboard comparing multiple models and pick a champion
- [ ] I can explain why RMSE is primary and when metrics like MAPE become unreliable
- [ ] I can interpret prediction interval coverage and assess whether intervals are well-calibrated
- [ ] I can explain the difference between rolling-window and expanding-window backtesting strategies

## Prerequisite (read first)

- Chapter 0 for the time-series contract (unique_id, ds, y) and UTC rules

## Concepts (plain English)

- **Backtesting (rolling-origin CV)**: Simulating real-world forecasting by repeating train→test splits forward through time. Prevents temporal leakage.
- **Horizon (h)**: How many time steps ahead we forecast (e.g., 24 hours)
- **Step size**: How far forward we move the training/test boundary each iteration (e.g., 168 hours = 1 week)
- **Expanding window**: Training set grows over time; test set is always fixed size
- **Rolling window**: Both train and test windows slide forward together; train size stays constant
- **Leakage**: Using future data to train models that predict the past (always a bug in backtesting)
- **Coverage**: Percentage of actual values that fall inside the model's [lo, hi] prediction interval (should ≈ confidence level)
- **MASE**: Mean Absolute Scaled Error; normalized by seasonal naive baseline; doesn't blow up at zero like MAPE
- **Feature engineering**: Lagged values and time-based features (used by XGBoostModel)

## Where this fits in the learning map (R → Python)

- **Seasonal analysis** → seasonal naive baselines and MSTL models in `src/chapter1/eia_data_simple.py`
- **Correlation analysis** → optional ACF/PACF diagnostics (use statsmodels if needed; not in pipeline by default)
- **Smoothing methods** → `ExponentialSmoothingModel` in `src/chapter2/models.py` and Holt-Winters in the StatsForecast pipeline
- **Decomposition** → MSTL models in `src/chapter1/eia_data_simple.py`
- **Forecasting strategies** → rolling-origin CV in `src/chapter2/backtesting.py` and `EIADataFetcher.cross_validate()`

## Feature engineering (optional ML track)

- The StatsForecast pipeline only uses `unique_id`, `ds`, `y` (no X features).
- `src/chapter2/feature_engineering.py` is for EDA or ML models that consume features.
- For multi-step horizons, avoid leakage by using a forecasting-aware tool (e.g., MLForecast) or recursive feature generation.

## Architecture (what we're building)

### Inputs
- **Forecasting-ready DataFrame**: columns `[unique_id, ds, y]`, UTC timestamps, no gaps/duplicates
- **Backtest config**: horizon (h), n_windows, step_size, confidence level
- **Model registry**: list of model classes to train and evaluate

### Outputs
- **cv_results.parquet**: Wide format from StatsForecast.cross_validation()
  - Columns: unique_id, ds, [model names], [model-lo-95], [model-hi-95]
- **leaderboard.parquet**: Model rankings (see below)
- **Metrics for each model**: RMSE, MAE, MAPE, MASE, Coverage (%)

### Invariants (must always hold)
- No temporal leakage: training set ends strictly before test period begins
- No data shuffling: timestamps always sorted ascending
- Test periods do not overlap (no double-counting of actuals)
- All models trained on identical train/test splits (fair comparison)

### Failure modes
- Sparse data in train window → model.fit() returns all NaN → metrics become NaN → leaderboard shows NaN rank
- Sparse or zero values in test → MAPE undefined or infinite → use MASE/RMSE instead
- Horizon too large → forecasting beyond data → poor coverage and high error
- n_windows too large → leftover data at end not evaluated → incomplete picture

## Files touched

- **`src/chapter1/eia_data_simple.py`**
  - `cross_validate()` method: Runs rolling-origin CV with StatsForecast or traditional models
  - `evaluate_forecast()`: Computes RMSE/MAPE/MASE/coverage on holdout split
  - `register_best_model()`: Logs champion model to MLflow
  - `ExperimentConfig`: Dataclass defining horizon, n_windows, step_size, model list

- **`src/chapter2/backtesting.py`**
  - `RollingWindowBacktest` class: Fixed-size training window slides forward
  - `ExpandingWindowBacktest` class: Training window grows; test set fixed size
  - `BacktestingStrategy` interface: Routes to above based on config

- **`src/chapter2/models.py`**
  - `ForecastModel` (abstract base): fit(), predict(), get_name()
  - Concrete implementations:
    - `ExponentialSmoothingModel`: Simple trend-based smoothing (baseline)
    - `ARIMAModel`: StatModels ARIMA with automatic parameter tuning
    - `ProphetModel`: Facebook Prophet with seasonal decomposition
    - `XGBoostModel`: Lagged features → tree ensemble
  - `ModelFactory`: Create models by name string

- **`src/chapter2/feature_engineering.py`**
  - `build_timetk_features()`: Calendar + lag + rolling features (pandas-only)

- **`src/chapter2/training.py`**
  - `TrainingPipeline`: Orchestrates training across backtesting splits
  - For each split: fit all models, generate forecasts, compute metrics
  - Returns: results DataFrame with all metrics

- **`src/chapter2/evaluation.py`**
  - `ModelSelector` class: select_best_model(), generate_leaderboard()
  - Metrics computation (NaN-aware): RMSE, MAE, MAPE, MASE, Coverage
  - Ranking by primary metric (default: RMSE)

## Step-by-step walkthrough

### 1) Prepare forecasting-ready data
```python
from src.chapter1.eia_data_simple import EIADataFetcher
import os

api_key = os.getenv("EIA_API_KEY")
fetcher = EIADataFetcher(api_key)

df_clean = fetcher.pull_data(
    start_date="2023-01-01",
    end_date="2023-12-31"
).pipe(fetcher.prepare_data)

df_forecast = fetcher.prepare_for_forecasting(
    df_clean, unique_id="NG_US48"
)
print(f"Rows: {len(df_forecast)}, Columns: {df_forecast.columns.tolist()}")
```
- **Expect**: 8,760 rows (365 days × 24 hours), columns: [unique_id, ds, y]
- **If it fails**: Check Chapter 1 prerequisites (API key, date range, data integrity)

### 2) Define backtest configuration
```python
from src.chapter1.eia_data_simple import ExperimentConfig

config = ExperimentConfig(
    name="baseline_experiment",
    horizon=24,          # Forecast next 24 hours
    n_windows=5,         # 5 train/test splits
    step_size=168,       # Move forward 1 week each time
    confidence_level=95,
    models=["AutoARIMA", "SeasonalNaive", "HoltWinters"],
    metrics=["rmse", "mape", "mase", "coverage"]
)
print(f"Config: {config}")
```
- **Expect**: Config object with all parameters set
- **If it fails**: Check that model names exist in chapter2.models.ModelFactory

### 3) Run cross-validation
```python
cv_results, leaderboard = fetcher.cross_validate(
    df_forecast,
    horizon=config.horizon,
    n_windows=config.n_windows,
    step_size=config.step_size,
    level=[config.confidence_level],
    models_to_train=config.models
)

print("CV Results shape:", cv_results.shape)
print("\nLeaderboard (top 3):")
print(leaderboard.head(3))
```
- **Expect**:
  - `cv_results`: Wide DataFrame with columns like `[unique_id, ds, AutoARIMA, AutoARIMA-lo-95, AutoARIMA-hi-95, ...]`
  - `leaderboard`: Ranked by RMSE (ascending)
    - Columns: model, rmse_mean, rmse_std, mape_mean, mase_mean, coverage_pct, rank
    - Top row = best model (lowest RMSE)
- **If it fails**:
  - NaN in leaderboard: Insufficient data in train window or model failed to converge
  - Memory error: Too many windows or horizon; reduce n_windows or horizon

### 4) Interpret the leaderboard
```python
print(f"Champion model: {leaderboard.iloc[0]['model']}")
print(f"RMSE: {leaderboard.iloc[0]['rmse_mean']:.2f} ± {leaderboard.iloc[0]['rmse_std']:.2f}")
print(f"Coverage: {leaderboard.iloc[0]['coverage_pct']:.1f}%")

# Check if coverage is reasonable (should be near confidence_level)
expected_coverage = 95
actual_coverage = leaderboard.iloc[0]['coverage_pct']
if abs(actual_coverage - expected_coverage) > 5:
    print(f"⚠️  Coverage is {actual_coverage:.1f}%, expected ≈{expected_coverage}%")
```
- **Expect**:
  - Champion RMSE is lowest among all models
  - Coverage ≈ 95% (within ±5% is acceptable)
  - MASE < 1 means better than seasonal naive baseline
- **If it fails**:
  - Coverage >> 95%: Intervals too wide; model is overconfident
  - Coverage << 95%: Intervals too tight; model underestimates uncertainty
  - MASE > 1: Model worse than naive seasonal; may need more tuning or longer training window

### 5) Evaluate on holdout split
```python
# Train on first 80%, test on last 20%
n_test = int(len(df_forecast) * 0.2)
train = df_forecast.iloc[:-n_test]
test = df_forecast.iloc[-n_test:]

# Get champion from leaderboard
champion_model_name = leaderboard.iloc[0]['model']

# Forecast and evaluate
forecast = fetcher.evaluate_forecast(
    train, test,
    model=champion_model_name,
    horizon=config.horizon,
    confidence_level=config.confidence_level
)

print(f"Holdout RMSE: {forecast['rmse']:.2f}")
print(f"Holdout Coverage: {forecast['coverage_pct']:.1f}%")
```
- **Expect**: Metrics on holdout set consistent with CV results (±10% variance is normal)
- **If it fails**: Holdout metrics much worse than CV → possible distribution shift or model overfitting to CV splits

### 6) Register champion in MLflow
```python
model_uri = fetcher.register_best_model(
    leaderboard=leaderboard,
    config=config,
    df_train=df_forecast,
    run_name="experiment_v1"
)

print(f"Model registered: {model_uri}")
```
- **Expect**: MLflow run created with model artifact, metrics logged
- **If it fails**: MLflow server unavailable or config missing; check MLflow setup

## Metrics & success criteria

### Primary metric
- **RMSE** (Root Mean Squared Error): Lower is better. Penalizes large errors heavily. Use as main ranking criterion.

### Secondary metrics
- **MASE** (Mean Absolute Scaled Error): Normalized by seasonal naive; MASE < 1 = better than baseline
- **Coverage %**: Should be ≈ confidence_level (95%); ±5% acceptable
- **MAE**: Absolute error in original units; easier to interpret than RMSE

### "Good enough" threshold
- RMSE < [domain-specific baseline] (depends on magnitude of y; compare to naive)
- MASE < 1 (beating seasonal naive)
- Coverage 90% to 98% (if using 95% confidence level)

### What would make me retrain / change strategy
- RMSE increases >20% vs previous experiment → model degradation; check for data drift
- Coverage < 85% or > 99% → intervals miscalibrated; recalibrate or use different method
- MASE > 1.5 → worse than naive; horizon may be too long or train window too short
- Multiple models have RMSE = NaN → insufficient training data; increase train window or reduce horizon

## Pitfalls (things that commonly break)

1. **MAPE with zero or near-zero values**:
   - MAPE = sum(|y_true - y_pred| / |y_true|) → explodes when y_true ≈ 0
   - Example: Solar generation at night = 0 → MAPE = ∞
   - **Fix**: Always use RMSE/MAE/MASE as primary metrics; report MAPE with caution or only on non-zero subset

2. **Temporal leakage (common mistake)**:
   - If train and test overlap or test comes before train, metrics are meaningless
   - Our code validates no leakage, but if you modify it, this will silently break results
   - **Fix**: Always verify: `cutoff_date < test_start_date`

3. **Confidence level hard-coded**:
   - Currently we hard-code "95" in several places; if you want different confidence level, it may not work
   - **Fix**: Pass confidence_level through all functions consistently

4. **Too many windows with sparse data**:
   - If n_windows is large and data is sparse, training windows may be nearly empty
   - Model fails to fit → returns NaN → leaderboard shows NaN ranks
   - **Fix**: Check that min_train_size ≥ 100 and each window has data; reduce n_windows if needed

5. **Horizon too large**:
   - If h=168 (1 week ahead), but your model only sees 30-day history, uncertainty is huge
   - Coverage will be poor; consider shortening horizon or lengthening training window
   - **Fix**: Plot y vs ds to understand seasonality; set horizon ≤ 1 season

## Mini-checkpoint (prove you learned it)

Answer these:

1. **Explain rolling-origin vs expanding-window backtesting**. When would you use each?
2. **Why does MAPE fail when y=0**? What metric is more robust?
3. **What does "coverage" mean in the context of prediction intervals?**
4. **If leaderboard shows coverage=85% (below 95%), what does it mean and how would you fix it?**

**Answers:**
1. Rolling-origin: train window is fixed size, slides forward (mimics real forecast deployment). Expanding: train grows over time (useful to detect model degradation as data volume increases). Use rolling-origin to match production; use expanding to detect drift.
2. MAPE = |error| / |y|; when y≈0, ratio explodes. Use RMSE (absolute scale) or MASE (normalized by baseline) instead.
3. Coverage is the % of test actuals that fall within [lower, upper] prediction interval. Should ≈ confidence level (95% for 95% interval).
4. Coverage=85% means the interval is too tight (underestimating uncertainty). Model may be overconfident. Expand intervals or recalibrate using conformal prediction.

## Exercises (optional, but recommended)

### Easy
1. Run cross-validation with horizon=12 (instead of 24) and compare RMSE vs horizon=24. Explain the difference.
2. Extract leaderboard for one model and plot its RMSE across the 5 CV windows. Is it stable or does it spike?

### Medium
1. Run cross-validation with step_size=24 (daily) instead of 168 (weekly). How many windows are created? How do metrics change?
2. Identify the bottom-ranked model in leaderboard. Calculate MASE manually for one test window and verify it's > 1.

### Hard
1. Implement a custom metric (e.g., "% of forecasts within 10% of actual") and add it to leaderboard.
2. Run cross-validation for 3 different series (e.g., different respondents or fueltypes) and compare their leaderboards. Which series is easier/harder to forecast? Why?
3. Modify the confidence level to 80% (instead of 95%) and re-run CV. Compare coverage between 80% and 95% intervals. Verify they differ as expected.
