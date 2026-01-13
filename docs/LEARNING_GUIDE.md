# Complete Learning Guide — ATSAF Course Materials

## Overview

This directory contains comprehensive learning documentation for the ATSAF (Automated Time Series Forecasting) project. All materials are aligned with the actual codebase implementation, verified against source files, and include runnable examples, checkpoints, and exercises.

## Structure

Each chapter markdown follows a consistent template:

1. **Outcomes** — What you can do after completing the chapter
2. **Concepts** — Plain English explanations of key terms
3. **Architecture** — What inputs/outputs/invariants define the system
4. **Files Touched** — Which source files to read and understand
5. **Step-by-Step Walkthrough** — Runnable examples with expected outputs
6. **Metrics & Success Criteria** — How to know you've succeeded
7. **Pitfalls** — Common mistakes and how to avoid them
8. **Mini-Checkpoint** — Self-test questions
9. **Exercises** — Optional hands-on practice (Easy/Medium/Hard)

---

## Chapters

### [Chapter 0: Time Series Objects & Contracts (Python)](chapter0_time_series_objects.md)

**Timeframe**: ~30-45 minutes to read and run

**What you'll learn:**
- How Python represents time-series objects (Series, DatetimeIndex)
- The `unique_id, ds, y` data contract and why StatsForecast depends on it
- How to normalize timestamps to UTC and detect DST edge cases
- How to validate time-series integrity before modeling

**Key files:**
- `src/chapter0/objects.py` (Chapter 0 helpers)
- `src/chapter1/validate.py` (validate_time_index)
- `src/chapter1/eia_data_simple.py` (prepare_for_forecasting, validate_time_series_integrity)
- `src/chapter3/tasks.py` (compute_time_series_integrity)

**Success criteria:**
- Can explain ts vs tsibble vs timetk in Python terms
- Can build and validate a forecasting-ready table
- Can spot timezone/DST risks before modeling

---

### [Chapter 1: Data Ingestion & Preparation](chapter1_ingestion.md)

**Timeframe**: ~2-4 hours to read, understand, and run

**What you'll learn:**
- How to fetch data from the EIA API with pagination and error handling
- How to normalize data to UTC and validate time-series integrity
- Why DST transitions break naive assumptions and how to detect them
- How to transform raw data into forecasting-ready format (unique_id, ds, y)

**Key files:**
- `src/chapter1/eia_data_simple.py` (main orchestrator)
- `src/chapter1/ingest.py`, `prepare.py`, `validate.py` (helper modules)

**Success criteria:**
- Can pull raw data, validate it, and format for forecasting
- Understand time-series integrity checks (duplicates, missing hours, DST)
- Can explain why UTC normalization is non-negotiable

---

### [Chapter 2: Experimentation & Backtesting](chapter2_experimentation.md)

**Timeframe**: ~3-5 hours to read, understand, and experiment

**What you'll learn:**
- How to design rolling-origin cross-validation for time-series
- How to build a leaderboard comparing multiple models
- When metrics like MAPE fail and what to use instead (RMSE, MASE)
- How to calibrate prediction intervals and interpret coverage

**Key files:**
- `src/chapter2/backtesting.py` (rolling/expanding window strategies)
- `src/chapter2/models.py` (model implementations: ARIMA, Prophet, XGBoost)
- `src/chapter2/training.py` (orchestrate CV across splits)
- `src/chapter2/evaluation.py` (compute metrics, rank models)

**Success criteria:**
- Can run CV and reproduce a leaderboard
- Understand temporal leakage and why it's prevented
- Can explain MAPE pitfalls and recommend alternatives

---

### [Chapter 3: Orchestration & Pipeline DAG](chapter3_orchestration.md)

**Timeframe**: ~2-3 hours to read, understand, and deploy

**What you'll learn:**
- How to decompose a forecasting workflow into independent, rerunnable tasks
- Why idempotency matters and how atomic writes prevent corruption
- How to build a DAG (Directed Acyclic Graph) of dependencies
- How to deploy to Airflow for automated scheduling

**Key files:**
- `src/chapter3/tasks.py` (6 tasks: ingest, prepare, validate, train, register, forecast)
- `src/chapter3/dag_builder.py` (Airflow DAG definition)
- `src/chapter3/cli.py` (Typer-based CLI for manual runs)

**Success criteria:**
- Can run pipeline end-to-end from CLI
- Understand why tasks are linear (no branching, yet)
- Can verify idempotency: re-running produces same outputs

---

### [Chapter 4: Monitoring, Drift Detection & Alerts](chapter4_monitoring.md)

**Timeframe**: ~4-6 hours to read, understand, and implement

**What you'll learn:**
- How to persist forecasts in a queryable database
- How to score forecasts against actuals (as they arrive)
- How to detect model drift using statistical thresholds
- How to monitor data freshness, completeness, and forecast staleness
- When and how to trigger alerts

**Key files:**
- `src/chapter4/db.py` (SQLite schema and CRUD)
- `src/chapter4/forecast_store.py` (persist wide → long format)
- `src/chapter4/scoring.py` (join forecasts with actuals)
- `src/chapter4/drift.py` (threshold-based drift detection)
- `src/chapter4/alerts.py` (alert configuration and checking)
- `src/chapter4/health.py` (data freshness/completeness checks)

**Success criteria:**
- Can persist forecasts and score them
- Understand drift thresholds (mean ± k*std from backtest)
- Can interpret alerts and decide when to retrain

---

### [Chapter 5: Data Quality and Preprocessing](chapter5_data_quality.md)

**Timeframe**: ~1-2 hours to read and run

**What you'll learn:**
- How to detect missing timestamps and duplicates
- How to repair gaps without leakage
- Why UTC normalization is non-negotiable

**Key files:**
- `src/chapter1/validate.py` (validate_time_index)
- `src/chapter1/prepare.py` (normalize_time)
- `src/chapter3/tasks.py` (compute_time_series_integrity)

**Success criteria:**
- Can produce a clean unique_id/ds/y table with no gaps

---

### [Chapter 6: Backtesting and Evaluation](chapter6_backtesting_evaluation.md)

**Timeframe**: ~2-3 hours to read and run

**What you'll learn:**
- How to create rolling-origin splits without leakage
- How to compute RMSE, MAE, and MASE on holdouts
- How to compare models on identical windows

**Key files:**
- `src/chapter2/backtesting.py` (rolling and expanding splits)
- `src/chapter2/evaluation.py` (metrics)
- `src/chapter1/eia_data_simple.py` (cross_validate)

**Success criteria:**
- Can build a small leaderboard from rolling splits

---

### [Chapter 7: Transformations and Stationarity](chapter7_transformations_stationarity.md)

**Timeframe**: ~1-2 hours to read and run

**What you'll learn:**
- When to use log transforms or differencing
- How to check stationarity with a quick test
- How transformations affect evaluation

**Key files:**
- statsmodels (ADF/KPSS tests)

**Success criteria:**
- Can justify a transform before modeling

---

### [Chapter 8: ACF, PACF, and Diagnostics](chapter8_acf_pacf_diagnostics.md)

**Timeframe**: ~1-2 hours to read and run

**What you'll learn:**
- How to compute ACF/PACF for model clues
- How to run a basic residual diagnostic
- How to interpret lag structure

**Key files:**
- statsmodels (acf, pacf, ljungbox)

**Success criteria:**
- Can explain a simple ACF/PACF pattern

---

### [Chapter 9: Exogenous Regressors and Interventions](chapter9_exogenous_interventions.md)

**Timeframe**: ~2-3 hours to read and run

**What you'll learn:**
- How to add weather, holiday, or event features
- How to align features to avoid leakage
- How to measure feature impact

**Key files:**
- `src/chapter2/feature_engineering.py` (lag/rolling features)

**Success criteria:**
- Can improve a baseline with safe exogenous features

---

### [Chapter 10: Probabilistic Forecasting and Calibration](chapter10_probabilistic_forecasting.md)

**Timeframe**: ~1-2 hours to read and run

**What you'll learn:**
- How to compute prediction interval coverage
- How to compare nominal vs empirical coverage
- How to spot miscalibrated intervals

**Key files:**
- `src/chapter2/evaluation.py` (coverage)
- `src/chapter1/eia_data_simple.py` (forecast intervals)

**Success criteria:**
- Can report coverage for a holdout window

---

### [Chapter 11: Hierarchical and Multi-series Forecasting](chapter11_hierarchical_multiseries.md)

**Timeframe**: ~1-2 hours to read and run

**What you'll learn:**
- How to structure multiple series with unique_id
- How to compare local vs global summaries
- Why reconciliation matters for totals

**Key files:**
- `src/chapter1/prepare.py` (forecasting format)
- `src/chapter1/validate.py` (integrity checks)

**Success criteria:**
- Can prepare and summarize multi-series data

---

### [Chapter 12: Special Data Types](chapter12_special_data_types.md)

**Timeframe**: ~1-2 hours to read and run

**What you'll learn:**
- How to handle zero-heavy or count series
- Which metrics fail on zeros and which do not

**Key files:**
- `src/chapter2/evaluation.py` (metric helpers)

**Success criteria:**
- Can choose MAE/MASE over MAPE when needed

---

### [Chapter 13: Model Selection and Ensembling](chapter13_model_selection_ensembling.md)

**Timeframe**: ~1-2 hours to read and run

**What you'll learn:**
- How to pick a champion from a leaderboard
- How to build a simple ensemble baseline
- Why a naive baseline should never be dropped

**Key files:**
- `src/chapter2/evaluation.py` (metrics)
- `src/chapter2/training.py` (training orchestration)

**Success criteria:**
- Can compare models and report a winner

---

## Quick Start

### 1) Run the full pipeline
```bash
cd c:\docker_projects\atsaf
python -m src.chapter3.cli run \
  --start-date 2023-06-01 \
  --end-date 2023-09-30 \
  --horizon 24
```

### 2) Read the relevant chapter
- New to the objects and contracts? → Read [Chapter 0](chapter0_time_series_objects.md)
- Just ingested data? → Read [Chapter 1](chapter1_ingestion.md)
- Training models? → Read [Chapter 2](chapter2_experimentation.md)
- Running pipelines? → Read [Chapter 3](chapter3_orchestration.md)
- Monitoring forecasts? → Read [Chapter 4](chapter4_monitoring.md)

### 3) Run the chapter's walkthrough
Each chapter has a "Step-by-Step Walkthrough" section with copy-paste examples.

### 4) Attempt the mini-checkpoint
Self-test at the end of each chapter to verify understanding.

### 5) (Optional) Attempt exercises
Exercises range from easy (data exploration) to hard (refactoring for new requirements).

---

## Alignment with Actual Code

### What Changed From Initial Plan

The plan initially suggested implementing MLForecast + conformal prediction intervals, but **the actual codebase uses StatsForecast with model-based intervals**. All learning materials have been rewritten to match the real implementation.

Similarly, Chapter 3 was described as having "branching logic" for no-new-data / validation failures, but the **actual pipeline is linear** (strict sequence). All examples reflect this reality.

### Code Fixes Applied

To ensure learning materials match code exactly, the following issues were corrected:

1. **validate_clean() key mismatch** (Chapter 3, tasks.py)
   - Error message was referencing non-existent keys (`duplicate_count`, `missing_hours_total`)
   - **Fixed**: Changed to use actual keys from `compute_time_series_integrity()` (`duplicate_pairs`, `missing_hours`)

2. **Hardcoded confidence level "95"** (Chapter 1, eia_data_simple.py)
   - `evaluate_forecast()` accepted `confidence_level` parameter but then hardcoded 95 when looking up `{model}-lo-95` / `{model}-hi-95` columns
   - **Fixed**: Parameterized to use `confidence_level` variable (lines 913-914, 933-934)

3. **MAPE with zeros warning** (Chapter 1, eia_data_simple.py)
   - MAPE metric can explode or become infinite when y ≈ 0
   - **Added**: Warning in docstring recommending RMSE/MAE/MASE instead
   - Also documented in Chapter 2 pitfalls section

---

## Design Principles Reflected in Materials

### 1) Verifiability
- Every claim about the code is traceable to actual source files
- Every example is runnable (not pseudo-code)
- Success criteria are observable (not vague)

### 2) Fail-Loud Pattern
- Code raises clear errors when assumptions are violated (e.g., validate_clean() raises on integrity failure)
- Learning materials emphasize *why* this matters (prevents bad data from reaching training)

### 3) Idempotency & Atomicity
- Tasks can be re-run and produce same outputs
- Writes are atomic: all-or-nothing (not partial files left behind)
- Learning materials emphasize *when* to exploit this and *how* to verify it works

### 4) Metrics Over Promises
- Primary metrics clearly identified (RMSE > MAPE, MASE for drift detection)
- Secondary metrics explained with context (why MAPE fails, why coverage matters)
- Thresholds are data-driven (from backtest, not magic numbers)

---

## Common Learning Paths

### Path 1: "I just want to run forecasts"
1. Read Chapter 0 (10 min) — understand objects + contract
2. Read Chapter 1 (15 min) — understand data
3. Skim Chapter 2 (10 min) — know the model exists
4. Read Chapter 3 (20 min) — run the pipeline
5. **Done!**

### Path 2: "I want to understand the full system"
1. Read Chapter 0 (30 min) + run walkthrough (15 min)
2. Read Chapter 1 (1 hour) + run walkthrough (30 min)
3. Read Chapter 2 (1.5 hours) + run walkthrough (1 hour) + attempt exercises
4. Read Chapter 3 (1 hour) + run walkthrough (30 min)
5. Read Chapter 4 (1.5 hours) + run walkthrough (1 hour)
6. **Total: ~9 hours**

### Path 3: "I want to modify the system"
1. Do Path 2 above
2. For each chapter you want to modify, read "Pitfalls" section carefully
3. Read the actual source code (not just examples)
4. Make changes incrementally, verify idempotency at each step
5. Update this learning guide if you change architecture

---

## FAQ

### Q: Why is Chapter X different from the course plan?
**A:** The course plan described one implementation (MLForecast + branching DAG + conformal intervals), but your actual codebase uses StatsForecast + linear DAG + model-based intervals. Learning materials reflect what you *actually* built, not the original plan.

### Q: How do I know if I've mastered a chapter?
**A:** Complete the "Mini-Checkpoint" questions without looking at answers. If you can explain all 4 questions clearly, you've got it.

### Q: Can I just read the code without the learning guides?
**A:** Yes, but you'll spend more time and miss important context. The guides are condensed versions of "what matters" and "why." The code has all the details; the guides have all the insight.

### Q: Why does the MAPE warning say it's undefined at zero?
**A:** MAPE = sum(|error| / |actual|). When actual ≈ 0, the denominator is tiny → ratio explodes. This happens with solar (night = 0), wind (calm = 0), or any fuel with downtime. RMSE/MAE don't have this problem.

### Q: What should I do if my forecasts drift?
**A:** (1) Re-run Chapter 4 "detect_drift()" to confirm threshold breached. (2) Re-ingest recent data (Chapter 1) to check for data quality issues. (3) Re-train (Chapter 2, 3) with recent data. (4) Check for external events (holidays, policy changes, weather patterns).

---

## Next Steps

### Immediate
- [ ] Run full pipeline from Chapter 3 CLI
- [ ] Read Chapter 0 (objects + data contract)
- [ ] Read Chapter 1 (data understanding)
- [ ] Read Chapter 3 (pipeline orchestration)

### Short-term (1-2 weeks)
- [ ] Complete Chapter 2 walkthrough + exercises
- [ ] Deploy to Airflow (Chapter 3, optional)
- [ ] Set up monitoring alerts (Chapter 4, optional)

### Medium-term (1-3 months)
- [ ] Modify pipeline for different respondent/fuel types
- [ ] Implement email/Slack alerts (Chapter 4)
- [ ] Build dashboard (query Chapter 4 database)
- [ ] Retrain on drift (orchestrate Chapter 3 → 4 feedback loop)

### Long-term
- [ ] Implement branching logic (no-new-data → skip training)
- [ ] Add conformal prediction intervals (Chapter 2 extension)
- [ ] Multi-model ensemble (Chapter 2 extension)
- [ ] Automated hyperparameter tuning (Chapter 2 extension)

---

## Feedback & Improvements

These learning materials are **living documentation**. If you find:
- An example that doesn't run
- A concept that's unclear
- A pitfall not mentioned
- A code issue not listed above

Please update the relevant chapter markdown. **Keep materials honest**: if it's not in the code, don't claim it in the guide.

---

**Last updated:** January 11, 2026
**Alignment verified against:** src/chapter[1-4]/, latest commits
