# Chapter 4 — Monitoring, Drift Detection & Alerts

## Outcomes (what I can do after this)

- [ ] I can persist forecasts and actuals into a queryable database
- [ ] I can compute rolling accuracy metrics and detect model drift
- [ ] I can set alert thresholds based on backtest performance
- [ ] I can run health checks (freshness, completeness, forecast staleness)
- [ ] I can interpret drift reports and decide when to retrain

## Concepts (plain English)

- **Forecast persistence**: Store predictions in a time-series database for later scoring
- **Actuals**: Real observed values that come in after the forecast horizon (e.g., 24 hours later)
- **Scoring**: Comparing predictions vs actuals using metrics (RMSE, MAPE, coverage)
- **Drift**: Model performance degrades over time (e.g., MAPE increases from 5% to 8%)
- **Restatement**: Re-forecasting recent periods as new actuals arrive (e.g., rescore last 7 days hourly)
- **Health check**: Monitoring data freshness, completeness, and forecast staleness
- **Alert threshold**: Metric value that triggers an alert (e.g., MAPE > 10% is critical)
- **Rolling window**: Computing metrics over a sliding time window (e.g., last 7 days)

## Architecture (what we're building)

### Inputs
- **Forecasts table** (from Chapter 3):
  - Columns: run_id, created_ts_utc, unique_id, ds, model, yhat, lo, hi
  - One row per (model, unique_id, hour)

- **Actuals** (append-only stream):
  - Same schema as forecasts, but marked as observed
  - Arrives continuously (e.g., hourly updates from Chapter 1 data fetch)

- **Backtest metrics** (from Chapter 2):
  - Historical RMSE, MAPE, coverage from cross-validation
  - Used to compute alert thresholds (mean ± k*std)

### Outputs
- **Metrics database** (SQLite):
  - `pipeline_runs`: Execution log (when pipeline ran, status, rows processed)
  - `forecasts`: Stored predictions (queryable by model, date range, unique_id)
  - `forecast_scores`: Rolling metrics (RMSE/MAPE/coverage per horizon)
  - `alerts`: Alert events (triggered when threshold breached)

- **Health report**:
  - Data freshness: Hours since last ingest
  - Data completeness: Missing hours in recent window
  - Forecast freshness: Hours since predictions generated

### Invariants (must always hold)
- Forecast stored once, scored multiple times (as actuals arrive)
- Alerts are immutable (never deleted, only logged)
- Health checks don't require actuals (can run in real-time, before scoring)
- No missing primary keys: (run_id, model, unique_id, ds) uniquely identifies a forecast

### Failure modes
- Forecast not yet scored: yhat exists but no metrics (actuals haven't arrived yet)
- Actuals missing entirely: Can't score forecasts → alert STALE_FORECAST
- Data drift (schema changed): New columns appear; scoring fails → alert PIPELINE_FAILURE
- Threshold too sensitive: Generates too many alerts → tune k (std multiplier)

## Files touched

- **`src/chapter4/db.py`** (279 lines)
  - `init_monitoring_db()`: Creates SQLite schema (pipeline_runs, forecasts, forecast_scores, alerts)
  - `MonitoringDB` class: Read/write interface to monitoring database

- **`src/chapter4/forecast_store.py`** (code present but not in line count)
  - `persist_forecasts()`: Unpivots wide forecast table → long table → writes to forecasts table

- **`src/chapter4/scoring.py`** (code present)
  - `score_forecasts()`: Join forecasts with actuals, compute RMSE/MAPE/coverage per horizon

- **`src/chapter4/drift.py`** (code present)
  - `compute_drift_threshold_from_backtest()`: Threshold = best_metric + k*std
  - `rolling_accuracy()`: Query historical metrics for a model/series/horizon
  - `detect_drift()`: Compare latest metric to threshold, return drift/ok/no_data
  - `write_alert()`: Persist alert to database

- **`src/chapter4/health.py`** (code present)
  - `check_freshness()`: Hours since last ingest
  - `check_completeness()`: Missing hours in recent data
  - `check_forecast_freshness()`: Hours since predictions generated
  - `full_health_check()`: Combined report

- **`src/chapter4/alerts.py`** (279 lines)
  - `AlertSeverity` enum: INFO, WARNING, CRITICAL
  - `AlertType` enum: STALE_DATA, MISSING_DATA, STALE_FORECAST, DATA_DRIFT, MODEL_DRIFT, PIPELINE_FAILURE
  - `AlertConfig`: Thresholds for data freshness, completeness, forecast age, drift severity
  - `check_alerts()`: Evaluate health + drift against thresholds
  - `send_alert()`: Route to log/email/slack (currently log-only)

- **`src/chapter4/run_log.py`** (code present)
  - `log_run()`: Log pipeline execution (start time, status, row counts, duration)

- **`src/chapter4/config.py`** (21 lines)
  - Configuration for alert thresholds and monitoring

## Step-by-step walkthrough

### 1) Initialize monitoring database
```python
from src.chapter4.db import init_monitoring_db

db_path = "monitoring.db"
init_monitoring_db(db_path)

# Verify schema created
import sqlite3
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"Tables: {[t[0] for t in tables]}")
```
- **Expect**: 4 tables created: `[pipeline_runs, forecasts, forecast_scores, alerts]`
- **If it fails**: Check that db_path is writable and SQLite is installed

### 2) Persist forecasts from Chapter 3
```python
from src.chapter4.forecast_store import persist_forecasts
import pandas as pd

# Load forecast from Chapter 3 (wide format)
forecasts_wide = pd.read_parquet("artifacts/predictions.parquet")
print(f"Wide shape: {forecasts_wide.shape}, Columns: {forecasts_wide.columns.tolist()}")

# Persist to monitoring DB (converts to long format)
persist_forecasts(
    db_path="monitoring.db",
    forecasts_wide=forecasts_wide,
    run_id="run_20240101_060000",
    model_name="AutoARIMA"
)

# Verify in DB
conn = sqlite3.connect("monitoring.db")
df_stored = pd.read_sql_query(
    "SELECT * FROM forecasts LIMIT 5",
    conn
)
print(f"\nStored forecasts:\n{df_stored}")
```
- **Expect**: 5 rows with columns [run_id, created_ts_utc, unique_id, ds, model, yhat, lo, hi]
- **If it fails**: forecasts_wide may have wrong schema; check expected wide columns (model name, model-lo-95, model-hi-95)

### 3) Check data freshness and completeness
```python
from src.chapter4.health import full_health_check

report = full_health_check(
    clean_data_path="data/clean.parquet",
    predictions_path="artifacts/predictions.parquet",
    now=pd.Timestamp.utcnow()
)

print(f"Health Report:")
print(f"  Data freshness: {report['data_freshness_hours']:.1f} hours old")
print(f"  Data completeness: {report['missing_hours']:.0f} missing hours")
print(f"  Forecast freshness: {report['forecast_freshness_hours']:.1f} hours old")
```
- **Expect**:
  - data_freshness_hours ≈ 0 (just ingested)
  - missing_hours = 0 (complete data)
  - forecast_freshness_hours ≈ 0 (just generated)
- **If it fails**: Check that clean_data_path and predictions_path exist

### 4) Compute drift threshold from backtest
```python
from src.chapter4.drift import compute_drift_threshold_from_backtest
import pandas as pd

# Load backtest leaderboard from Chapter 2
leaderboard = pd.read_parquet("artifacts/leaderboard.parquet")
champion_model = leaderboard.iloc[0]['model']
backtest_rmse_mean = leaderboard.iloc[0]['rmse_mean']
backtest_rmse_std = leaderboard.iloc[0]['rmse_std']

# Threshold = mean + 2*std (configurable k)
threshold = compute_drift_threshold_from_backtest(
    metric_mean=backtest_rmse_mean,
    metric_std=backtest_rmse_std,
    k=2.0,  # Standard deviations above mean
    metric_name="RMSE"
)

print(f"Backtest RMSE: {backtest_rmse_mean:.2f} ± {backtest_rmse_std:.2f}")
print(f"Drift threshold (RMSE > {threshold:.2f}): Alert triggered")
```
- **Expect**: threshold ≈ backtest_rmse_mean + 2*backtest_rmse_std
- **If it fails**: leaderboard may not have expected columns (rmse_mean, rmse_std)

### 5) Score forecasts vs actuals
```python
from src.chapter4.scoring import score_forecasts
import pandas as pd

# Load stored forecasts
conn = sqlite3.connect("monitoring.db")
forecasts_df = pd.read_sql_query(
    "SELECT * FROM forecasts WHERE run_id='run_20240101_060000'",
    conn
)

# Load actuals (same schema as forecasts, but y_actual instead of yhat)
# For now, use clean data as "actuals" (in real deployment, actuals come from live ingest)
actuals_df = pd.read_parquet("data/clean.parquet")
actuals_df.columns = ['unique_id', 'ds', 'y_actual']

# Score (compute metrics)
scores = score_forecasts(
    forecasts=forecasts_df,
    actuals=actuals_df,
    horizon_hours=24
)

print(f"Scores shape: {scores.shape}")
print(f"Columns: {scores.columns.tolist()}")
print(f"Sample:\n{scores.head()}")
```
- **Expect**: Columns [run_id, model, unique_id, horizon_hours, rmse, mape, coverage_pct, valid_rows]
- **If it fails**: actuals_df schema may not match; check column names

### 6) Detect drift
```python
from src.chapter4.drift import detect_drift

# Get rolling metrics for the model over last 7 days
rolling_metrics = rolling_accuracy(
    db_path="monitoring.db",
    model_name="AutoARIMA",
    unique_id="NG_US48",
    horizon_hours=24,
    window_days=7
)

# Latest metric (most recent day)
if len(rolling_metrics) > 0:
    latest_rmse = rolling_metrics.iloc[-1]['rmse']
    threshold = 12.5  # From step 4

    status = detect_drift(
        latest_metric=latest_rmse,
        threshold=threshold,
        metric_name="RMSE"
    )

    print(f"Latest RMSE: {latest_rmse:.2f}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Status: {status}")
else:
    print("No historical metrics; skipping drift check")
```
- **Expect**: status = "ok" (if latest_rmse < threshold) or "drift" (if latest_rmse > threshold)
- **If it fails**: rolling_metrics may be empty (no historical data yet); this is normal on first run

### 7) Run health checks and trigger alerts
```python
from src.chapter4.alerts import check_alerts, AlertConfig
from src.chapter4.db import MonitoringDB

# Set alert config
config = AlertConfig(
    data_freshness_warning_hours=4,
    data_freshness_critical_hours=6,
    missing_data_warning_hours=1,
    missing_data_critical_hours=3,
    forecast_freshness_warning_hours=12,
    forecast_freshness_critical_hours=24,
    drift_warning_threshold=0.1,      # 10% increase
    drift_critical_threshold=0.3,     # 30% increase
    rmse_increase_warning_pct=15,
    rmse_increase_critical_pct=25
)

# Run checks
health = full_health_check(...)
db = MonitoringDB("monitoring.db")

alerts = check_alerts(
    health_report=health,
    config=config,
    db=db,
    model_name="AutoARIMA"
)

print(f"Alerts triggered: {len(alerts)}")
for alert in alerts:
    print(f"  [{alert.severity}] {alert.alert_type}: {alert.message}")
```
- **Expect**: No alerts if system is healthy
- **If it fails**: Check health report values (freshness, completeness) against thresholds

### 8) Query historical metrics
```python
import pandas as pd
import sqlite3

conn = sqlite3.connect("monitoring.db")

# Leaderboard: average metrics per model (across all dates/unique_ids)
leaderboard = pd.read_sql_query("""
    SELECT
        model,
        ROUND(AVG(rmse), 2) as avg_rmse,
        ROUND(AVG(mape), 2) as avg_mape,
        ROUND(AVG(coverage_pct), 1) as avg_coverage,
        COUNT(*) as n_scores
    FROM forecast_scores
    GROUP BY model
    ORDER BY avg_rmse
""", conn)
print(f"Leaderboard:\n{leaderboard}")

# Time-series of RMSE for champion model
rmse_over_time = pd.read_sql_query("""
    SELECT
        DATE(scored_ts_utc) as date,
        ROUND(AVG(rmse), 2) as daily_rmse
    FROM forecast_scores
    WHERE model = 'AutoARIMA'
    GROUP BY DATE(scored_ts_utc)
    ORDER BY date DESC
    LIMIT 14
""", conn)
print(f"\nRMSE over last 14 days:\n{rmse_over_time}")

# Alerts log
alerts = pd.read_sql_query("""
    SELECT alert_ts_utc, alert_type, severity, message
    FROM alerts
    ORDER BY alert_ts_utc DESC
    LIMIT 10
""", conn)
print(f"\nRecent alerts:\n{alerts}")
```
- **Expect**:
  - Leaderboard sorted by avg_rmse (champion first)
  - RMSE values stable or gradually increasing (if drifting)
  - Alerts only if thresholds breached
- **If it fails**: No data in tables (need to run earlier steps first)

## Metrics & success criteria

### Primary metric
- **Drift detection latency**: Alert triggered within 1 hour of drift occurring

### Secondary metrics
- **False positive rate**: <5% of alerts are spurious (threshold too sensitive)
- **Coverage tracking**: Prediction intervals stay well-calibrated (coverage ≈ 95%)
- **Data freshness**: Max age of ingested data < 6 hours

### "Good enough" threshold
- Alert thresholds based on backtest performance (mean ± 2*std)
- Health checks run daily with no critical alerts
- Forecasts are scored within 24 hours of horizon completion

### What would make me retrain / change monitoring
- MAPE increases > 50% vs backtest → drift detected → retrain
- Coverage < 85% or > 99% → intervals miscalibrated → recalibrate or add conformal prediction
- Data freshness > 24 hours → data pipeline broken → investigate ingestion
- >10% of forecasts missing actuals → scoring broken → investigate data join

## Pitfalls (things that commonly break)

1. **No actuals data initially**:
   - Forecasts arrive immediately, but actuals take 24+ hours to arrive
   - Scoring fails if you try to score today's 24-hour forecast (actuals not yet observed)
   - **Fix**: Implement "future actuals" step: daily ingest fetches latest actuals and back-scores forecasts

2. **Threshold too aggressive**:
   - If k=1 (mean + 1*std), threshold is too low → constant false alarms
   - If k=3 (mean + 3*std), threshold too high → misses real drift
   - **Recommendation**: Start with k=2, tune based on false positive rate after 2 weeks

3. **Rolling window includes forecast horizon**:
   - If scoring yesterday's forecast with today's data, that's correct (forecast completed)
   - If trying to score today's forecast with today's data, that's wrong (horizon incomplete)
   - **Fix**: Only score forecasts where horizon_date ≤ today - 1 day

4. **Alerts not being sent**:
   - Current implementation logs alerts to database; email/Slack are stubs
   - If you expect email alerts, they won't come
   - **Fix**: Implement send_alert() with real email/Slack integration

5. **No baseline for drift detection**:
   - First few days of scores are too sparse to compare rolling window
   - Drift detection only works after backtest threshold is set
   - **Fix**: Seed threshold from backtest leaderboard (not from live scores)

6. **Duplicate actuals**:
   - If data ingest is rerun for the same date range, actuals may duplicate
   - Scoring will count same actual twice → metrics become meaningless
   - **Fix**: Use `INSERT OR REPLACE` (upsert) when writing actuals; key on (unique_id, ds)

## Mini-checkpoint (prove you learned it)

Answer these:

1. **Why can't we score a 24-hour forecast immediately after generating it?** When is it safe to score?
2. **Explain the drift threshold formula**: threshold = mean + k*std. What does k control?
3. **What's the difference between a forecast not being scored yet vs. a forecast drifting?** How does the system tell them apart?
4. **If the threshold is threshold = 12.5 RMSE and latest RMSE = 12.3, should an alert trigger?**

**Answers:**
1. We can't score it immediately because the actual value isn't observed yet (it's 24 hours in the future). Safe to score once the forecast period is complete and actuals are available (next day).
2. k=std multiplier. Higher k = wider band (fewer alerts, lower false positive rate). Lower k = tighter band (more alerts, higher false positive rate). k=2 is standard (95% of data under normal distribution).
3. Not yet scored: yhat exists, no metric row in forecast_scores (actuals haven't arrived). Drifting: metric row exists, but rmse > threshold. System checks both: if forecast too old without score → STALE_FORECAST alert; if metric exists and rmse > threshold → DRIFT alert.
4. No alert. Latest RMSE (12.3) < threshold (12.5), so status="ok". Alert only triggers if latest RMSE > 12.5.

## Exercises (optional, but recommended)

### Easy
1. Run full health check 3 times (at different times). How do freshness values change? When does data become "critical"?
2. Query the alerts table and count alerts by type and severity. Which type is most common?

### Medium
1. Manually modify one forecast's yhat value (e.g., multiply by 2) and re-score. How much does RMSE change? Does it trigger a drift alert?
2. Set drift threshold very low (k=0.5) and re-run drift detection. How many false alarms do you get?
3. Write a SQL query to find the longest period without any forecast scores (data gap). Investigate why.

### Hard
1. Implement restatement logic: for each day, re-score all forecasts from last 7 days as new actuals arrive. Verify RMSE stabilizes over the 7-day window.
2. Build a dashboard query that shows rolling 7-day RMSE for each model. Plot to CSV or JSON for visualization.
3. Implement automatic threshold tuning: adjust k based on false positive rate from last 30 days. If >5% alerts are false (later retracted), reduce k; if <2% alerts, increase k.
