# Chapter 3 — Orchestration & Pipeline DAG

## Outcomes (what I can do after this)

- [ ] I can run an end-to-end forecasting pipeline from the CLI
- [ ] I can understand how tasks decompose a workflow and why idempotency matters
- [ ] I can deploy the pipeline as an Airflow DAG (if Airflow is installed)
- [ ] I can visualize the pipeline dependency graph and explain task ordering
- [ ] I can modify task configurations without breaking downstream dependencies

## Concepts (plain English)

- **Task**: An atomic unit of work that can be re-run independently (pull data, validate, train, forecast)
- **DAG** (Directed Acyclic Graph): A visual representation of task dependencies (X → Y means X must finish before Y starts)
- **Idempotency**: A task can be re-run multiple times with the same inputs and produce the same outputs (no side effects like duplicate records)
- **Atomic writes**: File writes that either fully complete or fail (no partial files left behind)
- **Linear pipeline**: Tasks execute sequentially in a fixed order (no branching or conditional execution)
- **CLI**: Command-line interface to trigger the pipeline (vs. scheduled in Airflow)
- **Run ID**: A unique identifier for one pipeline execution (timestamps, UUIDs, etc.) used to group artifacts

## Architecture (what we're building)

### Inputs
- **CLI arguments**:
  - `--start-date` YYYY-MM-DD
  - `--end-date` YYYY-MM-DD
  - `--horizon` integer (hours ahead to forecast)
  - `--overwrite` boolean (re-run all tasks if True)
  - `--output-dir` path (default: "artifacts/")

- **Environment**: `.env` file with `EIA_API_KEY`

### Outputs
- **data/raw.parquet**: Raw API response
- **data/clean.parquet**: Normalized data (UTC, valid schema)
- **data/metadata.json**: Ingestion snapshot (date range, row count, integrity report)
- **artifacts/cv_results.parquet**: Cross-validation forecast table
- **artifacts/leaderboard.parquet**: Model rankings
- **artifacts/predictions.parquet**: Final forecast (trained on all clean data)
- **mlflow/**: MLflow experiment and model artifact

### Pipeline Flow
```
ingest_eia()
    ↓ (produces data/raw.parquet)
prepare_clean()
    ↓ (produces data/clean.parquet)
validate_clean()
    ↓ (validates, produces report)
train_backtest_select()
    ↓ (produces cv_results, leaderboard)
register_champion()
    ↓ (registers model in MLflow)
forecast_publish()
    ↓ (produces predictions)
(optional) Chapter 4 integration
```

### Invariants (must always hold)
- Each task's output directory exists before writing
- Outputs are written atomically (all or nothing)
- If `--overwrite=False`, skip task if output already exists (idempotent)
- Task ordering is strict: no task runs until all predecessors finish
- All intermediate files include run_id for traceability

### Failure modes
- API unavailable during ingest_eia() → task fails, pipeline stops, no partial files
- Validation fails (duplicates, missing hours) → pipeline stops before training (prevents bad model)
- Training runs out of memory → no model artifact written (MLflow stays clean)
- Forecast publish fails → no predictions written, but leaderboard is preserved (can retry)

## Files touched

- **`src/chapter3/tasks.py`** (433 lines)
  - `ingest_eia(config, run_id)` → raw.parquet
  - `prepare_clean(raw_path, config, run_id)` → clean.parquet, metadata.json
  - `validate_clean(clean_path, run_id)` → raises ValueError if not valid
  - `train_backtest_select(clean_path, config, run_id)` → cv_results.parquet, leaderboard.parquet
  - `register_champion(leaderboard, config, clean_path, run_id)` → registers in MLflow
  - `forecast_publish(clean_path, config, run_id)` → predictions.parquet

- **`src/chapter3/dag_builder.py`** (121 lines)
  - `build_daily_dag()`: Returns Airflow DAG (if airflow is installed)
  - `build_dag_dot()`: Returns DOT graph string for CLI visualization
  - DAG has schedule_interval = "0 6 * * *" (6 AM UTC daily)

- **`src/chapter3/cli.py`** (69 lines)
  - `run()`: Typer command to execute pipeline from CLI
  - Parses arguments, generates run_id, calls tasks in sequence

- **`src/chapter3/config.py`** (67 lines)
  - `PipelineConfig` dataclass: all configuration parameters

- **`src/chapter3/io_utils.py`** (33 lines)
  - Helpers for atomic parquet/JSON writes

## Step-by-step walkthrough

### 1) Verify setup and prerequisites
```bash
cd c:\docker_projects\atsaf
python -m pytest tests/ -v  # Run tests to verify environment
echo $EIA_API_KEY  # Verify API key is set in .env
```
- **Expect**: Tests pass; API key is set
- **If it fails**: Check .env file and install dependencies with `pip install -e .` or `uv sync`

### 2) View pipeline DAG (without running)
```bash
cd c:\docker_projects\atsaf
python -c "
from src.chapter3.dag_builder import build_dag_dot
dot_string = build_dag_dot()
print(dot_string)
"
# Copy output to https://dreampuf.github.io/GraphvizOnline/ to visualize
```
- **Expect**: DOT graph showing 6 tasks and dependencies: ingest → prepare → validate → train → register → forecast
- **If it fails**: Check that dag_builder.py is present and imports are correct

### 3) Run pipeline end-to-end (CLI)
```bash
cd c:\docker_projects\atsaf
python -m src.chapter3.cli run \
  --start-date 2023-06-01 \
  --end-date 2023-09-30 \
  --horizon 24 \
  --output-dir artifacts/
```
- **Expect**:
  - Logs: "Task: ingest_eia started", "Task: prepare_clean started", etc.
  - Files created: data/raw.parquet, data/clean.parquet, artifacts/cv_results.parquet, etc.
  - Run finishes in 2-10 minutes depending on data size and model training
- **If it fails**:
  - "API Error": Check EIA_API_KEY in .env
  - "Validation failed": Re-run Chapter 1 to diagnose integrity issues
  - "Out of memory": Reduce date range or horizon; reduce n_windows

### 4) Inspect task outputs
```python
import pandas as pd

# Check raw data
df_raw = pd.read_parquet("c:\docker_projects\atsaf\data\raw.parquet")
print(f"Raw shape: {df_raw.shape}, Columns: {df_raw.columns.tolist()}")

# Check clean data
df_clean = pd.read_parquet("c:\docker_projects\atsaf\data\clean.parquet")
print(f"Clean shape: {df_clean.shape}, Columns: {df_clean.columns.tolist()}")

# Check leaderboard
leaderboard = pd.read_parquet("c:\docker_projects\atsaf\artifacts\leaderboard.parquet")
print(f"\nLeaderboard:\n{leaderboard.head()}")

# Check metadata
import json
with open("c:\docker_projects\atsaf\data\metadata.json") as f:
    metadata = json.load(f)
    print(f"\nMetadata: {metadata}")
```
- **Expect**:
  - raw: columns [time, value, respondent, fueltype]
  - clean: columns [unique_id, ds, y] with UTC timestamps
  - leaderboard: columns [model, rmse_mean, rmse_std, rank]
  - metadata: status='valid', row_count, date_range
- **If it fails**: Check pipeline logs to see which task failed

### 5) Re-run with `--overwrite` flag
```bash
cd c:\docker_projects\atsaf
python -m src.chapter3.cli run \
  --start-date 2023-06-01 \
  --end-date 2023-09-30 \
  --horizon 24 \
  --overwrite  # Force re-run all tasks
```
- **Expect**: Same outputs as step 3 (deterministic)
- **If it fails**: Something is non-deterministic (e.g., random seed not set in model)

### 6) Re-run without `--overwrite` (test idempotency)
```bash
cd c:\docker_projects\atsaf
python -m src.chapter3.cli run \
  --start-date 2023-06-01 \
  --end-date 2023-09-30 \
  --horizon 24
  # No --overwrite flag
```
- **Expect**: Pipeline finishes immediately (skips all tasks because files exist)
- **If it fails**: Tasks are not checking for existing outputs; check `if output_path.exists(): return` in each task

### 7) Deploy to Airflow (optional, if Airflow is installed)
```bash
# Set Airflow home
export AIRFLOW_HOME=~/airflow_atsaf

# Initialize DB
airflow db init

# Create Airflow user
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com

# Copy DAG to Airflow DAGs folder
cp src/chapter3/dag_builder.py ~/airflow_atsaf/dags/atsaf_daily_pipeline.py

# Start scheduler and webserver
airflow scheduler &
airflow webserver --port 8080 &

# Visit http://localhost:8080 to see DAG
```
- **Expect**: DAG appears in Airflow UI; can manually trigger runs
- **If it fails**: Check Airflow logs at ~/airflow_atsaf/logs/

## Metrics & success criteria

### Primary metric
- **Pipeline success rate**: 100% of runs complete without errors

### Secondary metrics
- **Task execution time**: Each task < 5 minutes (ingest < 1min, train < 3min)
- **Idempotency**: Re-running with same inputs produces same outputs
- **File integrity**: All output files valid (non-empty, correct schema)

### "Good enough" threshold
- Pipeline completes in < 10 minutes for 12-month dataset
- All 6 tasks execute in order with no skipped tasks (on first run)
- Artifacts folder contains all expected files (raw, clean, cv_results, leaderboard, predictions)

### What would make me redesign
- Task execution time > 30 minutes → parallelize or reduce data
- Output files corrupt (can't read as parquet) → implement better error handling
- Idempotency broken (re-run produces different results) → add random seed or determinism

## Pitfalls (things that commonly break)

1. **File path confusion (Windows vs Linux)**:
   - Windows uses `\`, Linux uses `/`
   - Our code uses `pathlib.Path` which handles both, but command-line args may have issues
   - **Fix**: Always use forward slashes in CLI args or use raw strings (r"path\to\file")

2. **Task ordering assumptions**:
   - If you refactor and skip validate_clean(), bad data gets trained
   - Our linear order is strict: ingest → prepare → validate → train → register → forecast
   - **Fix**: Never skip or reorder tasks without understanding downstream impact

3. **Output directory doesn't exist**:
   - If `artifacts/` or `data/` directory doesn't exist, writes fail silently
   - **Fix**: Run `mkdir -p artifacts/ data/` first, or let tasks create them

4. **API rate limiting**:
   - EIA API limits to ~50 requests/second; if pulling large date ranges, may timeout
   - **Fix**: Add delay between requests or use batch API endpoint

5. **MLflow registration fails silently**:
   - If MLflow is not initialized or artifact store is misconfigured, registration may skip
   - **Fix**: Check `mlflow.get_tracking_uri()` and `mlflow.get_artifact_uri()` before deploying

6. **run_id collision**:
   - If two pipelines run simultaneously with same run_id, they may overwrite each other
   - **Fix**: Use UUID or timestamp + hostname in run_id

## Mini-checkpoint (prove you learned it)

Answer these:

1. **Draw the DAG by hand**: 6 boxes for tasks, arrows for dependencies. Explain why no arrows go backward.
2. **What is idempotency and why does prepare_clean() check if output exists before writing?**
3. **If validate_clean() is skipped, what could go wrong?** Give a concrete example.
4. **If horizon=24 is changed to horizon=48 in CLI, which tasks re-run and which are cached?**

**Answers:**
1. Boxes: [ingest] → [prepare] → [validate] → [train] → [register] → [forecast]. No backward arrows because time flows forward; we can't validate before ingesting.
2. Idempotency means re-running with same inputs yields same outputs (no hidden state). prepare_clean() checks if output exists; if so, skips work (saves time). This requires no side effects (pure function style).
3. Without validation, bad data (duplicates, missing hours) enters training. Model learns on leaky/corrupt series → poor forecasts. In production, invalid data gets a bad model deployed.
4. Only train_backtest_select() and forecast_publish() re-run (they depend on horizon). ingest, prepare, validate use same data regardless of horizon, so cached files are reused.

## Exercises (optional, but recommended)

### Easy
1. Run the pipeline with a 1-month date range and measure total execution time. Then run with 3-month range. Does time scale linearly with data size?
2. Manually delete `artifacts/leaderboard.parquet` and re-run with `--overwrite=false`. Does the pipeline detect missing file and re-train?

### Medium
1. Modify the CLI to accept a `--model-list` argument (e.g., `--model-list AutoARIMA,HoltWinters`) and pass it to train_backtest_select(). Test with different model combos.
2. Add a new task `save_metrics_csv()` that converts artifacts/leaderboard.parquet to CSV. Update the DAG to place it after forecast_publish().

### Hard
1. Implement `--check-only` flag that runs through ingest/prepare/validate but skips training and forecasting. Useful for data validation without model cost.
2. Modify the pipeline to run for 5 different respondents (e.g., NG_CA1, NG_TX, NG_US48, etc.) in parallel within one `run` call. Track run_id per respondent to avoid collisions.
3. Deploy to Airflow and set up a task retry policy: if a task fails, retry 3 times with 5-minute delays. Test by injecting a temporary API error and verifying retry behavior.
