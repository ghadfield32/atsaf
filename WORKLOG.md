# Work Log

## Renewable Forecast Diagnostics (ERCO_WND negatives)
- 2026-01-13 Done: Added series/forecast diagnostics to log min/max, negatives, and missing values in generation data and forecasts.
- 2026-01-14 Done: Added run_log diagnostics for negative forecast summaries and generation coverage to preserve samples after pipeline runs.
- 2026-01-13 Doing: Investigating why ERCO_WND forecasts go negative by tracing data and model outputs with the new diagnostics.
- 2026-01-14 Next: Re-run the pipeline to capture negative forecast samples and confirm whether negatives correlate with low/zero generation periods.

## Renewable Forecast Diagnostics (Missing CALI series)
- 2026-01-13 Done: Added extra EIA fetch diagnostics and generation coverage logging to surface why CALI_SUN/CALI_WND are empty.
- 2026-01-14 Done: Persisted per-region EIA fetch metadata into run_log.json for each hourly run.
- 2026-01-13 Doing: Compare EIA response metadata for CALI vs ERCO/MISO to confirm dataset availability and date range alignment.
- 2026-01-14 Next: Re-run the hourly pipeline and review run_log.json for CALI total/offset/period metadata.
