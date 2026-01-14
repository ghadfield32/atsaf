# Work Log

## Renewable Forecast Diagnostics (ERCO_WND negatives)
- 2026-01-13 Done: Added series/forecast diagnostics to log min/max, negatives, and missing values in generation data and forecasts.
- 2026-01-13 Doing: Investigating why ERCO_WND forecasts go negative by tracing data and model outputs with the new diagnostics.
- 2026-01-13 Next: Run the pipeline with diagnostics on real ERCO_WND data and capture any negative rows in training/forecast outputs.

## Renewable Forecast Diagnostics (Missing CALI series)
- 2026-01-13 Done: Added extra EIA fetch diagnostics and generation coverage logging to surface why CALI_SUN/CALI_WND are empty.
- 2026-01-13 Doing: Compare EIA response metadata for CALI vs ERCO/MISO to confirm dataset availability and date range alignment.
- 2026-01-13 Next: Re-run the hourly pipeline and capture fetch/coverage logs to confirm whether CALI is absent or mis-keyed.
