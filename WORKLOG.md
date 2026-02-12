# Work Log

## Renewable Pipeline (CALI 358-hour EIA data gap crashing pipeline)
- 2026-02-12 Done: Root-caused CI crash #538 to EIA/CAISO reporting outage: CALI missing data Jan 26 – Feb 10 (358h gap, 363 vs 721 rows). `validate_generation_df` triggers `max_missing_ratio=0.02` (CALI_SUN=49.65%) and kills entire pipeline — even though ERCO/MISO have complete data.
- 2026-02-12 Done: Added `compute_per_series_gap_ratios()` to `validation.py` — diagnostic function returns per-series missing ratios without pass/fail decisions.
- 2026-02-12 Done: Added Step 1.5 "GAP_FILTER" in `run_full_pipeline()` (`tasks.py`): drops series exceeding `max_missing_ratio`, saves filtered parquet, logs all decisions, fails only if remaining series < 2.
- 2026-02-12 Done: Updated `run_hourly.py` post-pipeline validation to exclude gap-filtered series from `expected_series`, using `results["gap_filter"]["series_dropped"]`.
- 2026-02-12 Done: Verified all 3 files compile and existing tests (25 pass) show no regressions; 13 pre-existing failures from prior refactoring unrelated to this change.
- 2026-02-12 Next: Push and re-run hourly workflow to confirm CALI is dropped cleanly and ERCO+MISO proceed to forecasts.

## Renewable Pipeline (validation UnboundLocalError on pd)
- 2026-02-12 Done: Root-caused CI crash to function-local `import pandas as pd` in `validate_generation_df`, which shadowed module `pd` and failed at first `pd.to_datetime(...)`.
- 2026-02-12 Done: Added `VALIDATION_DEBUG` plumbing (`run_hourly.py` -> `tasks.py` -> `validation.py`) plus staged snapshots (schema/nulls/dtypes/ds-range/y-stats) for stepwise diagnosis without mutating data.
- 2026-02-12 Doing: Added regression tests for stale-series branch execution and debug snapshot emission to prevent scoping regressions.
- 2026-02-12 Next: Re-run hourly workflow with `VALIDATION_DEBUG=true` to capture detailed state if any post-fix validation issue remains.

## Renewable Pipeline (EIA 504 fetch failures)
- 2026-02-02 Done: Added request-level diagnostics (status counts, last URL, retries) and surfaced EIA fetch settings from env.
- 2026-02-02 Done: Added sanitized URL logging for freshness probes to pinpoint failing endpoints without leaking API keys.
- 2026-02-02 Next: Re-run hourly job with EIA_DEBUG_REQUESTS=true and consider EIA_MAX_WORKERS=1 to confirm whether 504s persist.

## Renewable Pipeline (Freshness log crash on None lag)
- 2026-01-29 Done: Isolated TypeError to freshness logging formatting when probe times out and lag_hours is None.
- 2026-01-29 Doing: Add explicit lag="unknown" logging and surface probe error in the per-series log line.
- 2026-01-29 Next: Re-run hourly job to confirm timeout cases log cleanly without crashing.

## Renewable Forecast Diagnostics (ERCO_WND negatives)
- 2026-01-13 Done: Added series/forecast diagnostics to log min/max, negatives, and missing values in generation data and forecasts.
- 2026-01-14 Done: Added run_log diagnostics for negative forecast summaries and generation coverage to preserve samples after pipeline runs.
- 2026-01-14 Done: Added per-model forecast summaries + explicit model selection logging to isolate which model goes negative.
- 2026-01-13 Doing: Investigating why ERCO_WND forecasts go negative by tracing data and model outputs with the new diagnostics.
- 2026-01-14 Next: Re-run the pipeline to capture negative forecast samples and confirm whether negatives correlate with low/zero generation periods.

## Renewable Forecast Diagnostics (Missing CALI series)
- 2026-01-13 Done: Added extra EIA fetch diagnostics and generation coverage logging to surface why CALI_SUN/CALI_WND are empty.
- 2026-01-14 Done: Persisted per-region EIA fetch metadata into run_log.json for each hourly run.
- 2026-01-14 Done: Added cached-generation/weather coverage checks so missing SUN series are flagged even when loading existing parquet files.
- 2026-01-13 Doing: Compare EIA response metadata for CALI vs ERCO/MISO to confirm dataset availability and date range alignment.
- 2026-01-14 Next: Re-run the hourly pipeline and review run_log.json for CALI total/offset/period metadata.

## Renewable Dashboard (SUN filter empty)
- 2026-01-14 Done: Added forecast data debug panels to surface source, columns, and fuel_type/region counts before and after filtering.
- 2026-01-14 Doing: Verify forecasts.parquet and DB fuel_type values to confirm whether SUN series are missing or mislabeled.

## Renewable Pipeline (Missing _log_series_summary import)
- 2026-01-15 Done: Restored `_log_series_summary` diagnostics helper in modeling to unblock pipeline imports and log core series health stats.
