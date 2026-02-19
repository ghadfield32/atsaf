# DEVLOG — atsaf renewable pipeline

Compact record of what was done, what is in progress, and what is pending.
One-to-two liners per entry. Organised by section/topic.

---

## GitHub Actions / CI

### Timeout & Pipeline Performance

| Date | Status | Entry |
|------|--------|-------|
| 2026-02-17 | DONE | **Diagnosed** "Error: The operation was canceled." as GitHub Actions `timeout-minutes: 25` kill — pipeline ran 23m 11s, 15s over limit. |
| 2026-02-17 | DONE | **Root cause**: `train_interpretability_models()` used hardcoded `lags=168` (7-day LightGBM window) × 6 series + SHAP on 50% sample — the dominant runtime contributor. |
| 2026-02-17 | DONE | **Fix**: Made `lags` configurable via `RENEWABLE_INTERPRETABILITY_LAGS` env var (default `48`); set `48` in workflow. Reduces lag-feature matrix from 168→48 columns (71% smaller). |
| 2026-02-17 | DONE | **Observability**: `step_timings` existed in code but used `logger.info()` — invisible in Actions (only WARNING+ surfaces). Added `print()` alongside every step start/end + per-series timing in interpretability loop. |
| 2026-02-17 | DONE | Added `SKIP_EDA: "false"` to workflow so EDA can be toggled without code changes. |
| 2026-02-19 | DONE | Reconfirmed latest failure pattern: negative-value lines are warnings; job is still timing out (`run_hourly` ~23m+ plus setup exceeds 25m limit). |
| 2026-02-19 | DONE | Added step heartbeat instrumentation in `run_full_pipeline()` (`[pipeline][HEARTBEAT]`) to emit progress every N seconds during long steps (`fetch`, `eda`, `train_cv`, `forecast`, `interpretability`). |
| 2026-02-19 | DONE | Added runtime controls in `run_hourly`: `RENEWABLE_ENABLE_INTERPRETABILITY` and `PIPELINE_HEARTBEAT_SECONDS`; surfaced both in `run_log.json` config for traceability. |
| 2026-02-19 | DONE | Hourly workflow now sets `RENEWABLE_ENABLE_INTERPRETABILITY=false` and `PIPELINE_HEARTBEAT_SECONDS=60` so CI prioritizes forecast pipeline within time budget while keeping live progress logs. |

### Negative Values (CALI_SUN)

| Date | Status | Entry |
|------|--------|-------|
| 2026-02-17 | CONFIRMED-NORMAL | 376 negative CALI_SUN values (52.1%) are **expected nighttime EIA net-generation readings** ([-57, -9] MWh/hr). Correctly clamped to zero by dataset builder. Not a bug. |
| 2026-02-17 | WATCH | If CALI_SUN negative ratio ever exceeds 60% or range drops below -100 MWh, re-examine EIA data source for reporting change. |

---

## Data Pipeline

### EIA Fetch

| Date | Status | Entry |
|------|--------|-------|
| ongoing | OK | Fetcher uses `max_workers=3`, `timeout=90s`, `max_retries=3`; 6 series fetched cleanly in single pages. |

### Validation

| Date | Status | Entry |
|------|--------|-------|
| ongoing | OK | `validate_generation_df()` logs negatives as WARNING but does not fail — delegates to dataset builder per `negative_policy`. |

### Interpretability

| Date | Status | Entry |
|------|--------|-------|
| 2026-02-17 | DONE | Lag count now `RENEWABLE_INTERPRETABILITY_LAGS` (default 48). Full 168-lag runs can be triggered locally via env override. |
| PENDING | - | Monitor next CI run timing; if interpretability still >10 min at 48 lags, consider reducing `shap_max_samples` from 1000 or `n_estimators` from 100. |

### StatsForecast CV

| Date | Status | Entry |
|------|--------|-------|
| 2026-02-16 | OK | Best model: AutoTheta (RMSE 3745). CV uses 2 windows, step=168h. 5 models × 6 series fit within ~5-8 min typically. |
| PENDING | - | If CV becomes a bottleneck, consider reducing model set or setting `RENEWABLE_CV_WINDOWS: "1"` temporarily. |
| 2026-02-19 | WATCH | Local profiling showed `train_cv` can run for many minutes with high CPU and sparse logs; heartbeat now makes this visible during CI execution. |

---

## Pending / To Do

- [ ] Observe next scheduled CI run — confirm step timings appear in Actions log and total < 20 min.
- [ ] If interpretability at 48 lags still slow: tune `shap_max_samples` or `n_estimators`.
- [ ] Consider `SKIP_EDA: "true"` in workflow (EDA adds plot generation overhead; recommendations are stable).
- [ ] Investigate whether `concurrency: cancel-in-progress: true` could cancel a valid long run if a manual dispatch overlaps.
- [ ] After next successful run, verify `run_log.json` includes `config.enable_interpretability` and `config.heartbeat_seconds` for auditability.
