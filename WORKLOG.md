# Work Log

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
