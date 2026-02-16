# file: src/renewable/jobs/run_hourly.py
"""Hourly renewable pipeline entry point with validation."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from src.renewable.tasks import RenewablePipelineConfig, run_full_pipeline
from src.renewable.validation import validate_generation_df
from src.renewable.data_freshness import check_all_series_freshness, FreshnessCheckResult

load_dotenv()

# Suppress DeprecationWarning from statsforecast library (invalid escape sequences)
# These are in third-party code we cannot fix directly
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='statsforecast')


def _env_list(name: str, default_csv: str) -> list[str]:
    raw = os.getenv(name, default_csv)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _expected_series(regions: list[str], fuel_types: list[str]) -> list[str]:
    return [f"{region}_{fuel}" for region in regions for fuel in fuel_types]


def _sanitize_for_json(obj):
    """
    Recursively sanitize data structure for JSON serialization.

    Replaces NaN/Infinity values with None to ensure valid JSON compliance.
    Python's json.dumps() by default writes NaN/Infinity as JavaScript literals,
    which are NOT valid JSON per RFC 8259.

    Args:
        obj: Any Python object (dict, list, primitive, etc.)

    Returns:
        Sanitized copy with NaN/Infinity replaced by None
    """
    import math

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif hasattr(obj, "item"):  # numpy scalar
        try:
            extracted = obj.item()
            if isinstance(extracted, float) and (math.isnan(extracted) or math.isinf(extracted)):
                return None
            return extracted
        except Exception:
            return obj
    else:
        return obj


def _json_default(value: object) -> str:
    """
    JSON serializer for objects not serializable by default json module.

    Handles:
    - pd.Timestamp/datetime → ISO format strings
    - Other types → string representation

    Note: NaN/Infinity are handled by _sanitize_for_json() before serialization.
    """
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def _summarize_generation_coverage(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"row_count": 0, "series_count": 0, "coverage": []}

    coverage = (
        df.groupby("unique_id")["ds"]
        .agg(min_ds="min", max_ds="max", rows="count")
        .reset_index()
        .sort_values("unique_id")
    )
    return {
        "row_count": int(len(df)),
        "series_count": int(df["unique_id"].nunique()),
        "coverage": coverage.to_dict(orient="records"),
    }


def _read_previous_run_summary(data_dir: str) -> dict | None:
    """Read previous run_log.json for rowcount comparison."""
    path = Path(data_dir) / "run_log.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _summarize_negative_forecasts(
    df: pd.DataFrame,
    sample_rows: int = 5,
) -> dict:
    if df.empty or "yhat" not in df.columns:
        return {
            "row_count": int(len(df)),
            "negative_rows": 0,
            "series": [],
            "sample": [],
        }

    neg = df[df["yhat"] < 0]
    if neg.empty:
        return {
            "row_count": int(len(df)),
            "negative_rows": 0,
            "series": [],
            "sample": [],
        }

    series_summary = (
        neg.groupby("unique_id")["yhat"]
        .agg(count="count", min_value="min", max_value="max", mean_value="mean")
        .reset_index()
        .sort_values("unique_id")
    )
    sample = neg[["unique_id", "ds", "yhat"]].head(sample_rows)
    return {
        "row_count": int(len(df)),
        "negative_rows": int(len(neg)),
        "series": series_summary.to_dict(orient="records"),
        "sample": sample.to_dict(orient="records"),
    }


def run_hourly_pipeline() -> dict:
    data_dir = os.getenv("RENEWABLE_DATA_DIR", "data/renewable")
    regions = _env_list("RENEWABLE_REGIONS", "CALI,ERCO,MISO")
    fuel_types = _env_list("RENEWABLE_FUELS", "WND,SUN")
    lookback_days = _env_int("LOOKBACK_DAYS", 30)

    # Horizon configuration: support both preset and direct override
    horizon_preset = os.getenv("RENEWABLE_HORIZON_PRESET", None)  # "24h" | "48h" | "72h"
    horizon_override = _env_int("RENEWABLE_HORIZON", 0)  # Legacy direct override

    # If direct override is set, use it; otherwise use preset (or None for default)
    if horizon_override > 0:
        horizon = horizon_override
        horizon_preset = None  # Ignore preset if direct override is set
    else:
        horizon = 24  # Default, may be overridden by preset

    cv_windows = _env_int("RENEWABLE_CV_WINDOWS", 2)
    cv_step_size = _env_int("RENEWABLE_CV_STEP_SIZE", 168)

    start_date = os.getenv("RENEWABLE_START_DATE", "")
    end_date = os.getenv("RENEWABLE_END_DATE", "")

    # Check if we should force run (e.g., manual dispatch from dashboard)
    force_run = os.getenv("FORCE_RUN", "false").lower() == "true"

    # DEBUG: Log force_run status
    print(f"[pipeline] FORCE_RUN={force_run}")

    # Validation thresholds (used in freshness check + validation)
    max_lag_hours = _env_int("MAX_LAG_HOURS", 48)
    max_missing_ratio = _env_float("MAX_MISSING_RATIO", 0.02)
    validation_debug = _env_bool("VALIDATION_DEBUG", False)
    if validation_debug:
        print("[config] Validation debug snapshots enabled (VALIDATION_DEBUG=true)")

    # Per-region lag threshold overrides (optional)
    region_lag_thresholds = {}
    for region in regions:
        region_threshold = _env_int(f"MAX_LAG_HOURS_{region}", None)
        if region_threshold is not None:
            region_lag_thresholds[region] = region_threshold
            print(
                f"[config] Region-specific threshold: "
                f"{region} max_lag_hours={region_threshold}h"
            )

    # Data freshness check - skip full pipeline if no new data
    if not force_run:
        api_key = os.getenv("EIA_API_KEY", "")
        if not api_key:
            print("WARNING: EIA_API_KEY not set, skipping freshness check")
        else:
            # Track timing
            freshness_check_start = datetime.now(timezone.utc)
            print(
                f"[pipeline][TIMING] Freshness check starting at "
                f"{freshness_check_start.isoformat()}"
            )

            run_log_path = Path(data_dir) / "run_log.json"
            freshness = check_all_series_freshness(
                regions=regions,
                fuel_types=fuel_types,
                run_log_path=run_log_path,
                api_key=api_key,
                max_lag_hours=max_lag_hours,
            )

            # Log freshness check completion
            freshness_check_end = datetime.now(timezone.utc)
            freshness_elapsed = (
                (freshness_check_end - freshness_check_start).total_seconds()
            )
            print(
                f"[pipeline][TIMING] Freshness check completed at "
                f"{freshness_check_end.isoformat()} "
                f"(elapsed: {freshness_elapsed:.1f}s)"
            )

            if not freshness.has_new_data:
                # No new data - return early with skip status
                print(f"SKIPPED: {freshness.summary}")
                print("[freshness][DETAILS] Per-series status:")
                for series_id, status in freshness.series_status.items():
                    prev = status.get("prev_max_ds", "N/A")
                    current = status.get("current_max_ds", "N/A")
                    is_new = status.get("is_new", False)
                    lag = status.get("lag_hours", "N/A")
                    stale_flag = status.get("is_stale", False)

                    status_str = "NEW" if is_new else "unchanged"
                    if stale_flag:
                        status_str += " (STALE)"

                    print(
                        f"  {series_id}: prev={prev} current={current} "
                        f"lag={lag}h ({status_str})"
                    )

                skip_log = {
                    "run_at_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "skipped",
                    "reason": "no_new_data",
                    "freshness_check": {
                        "checked_at_utc": freshness.checked_at_utc,
                        "summary": freshness.summary,
                        "series_status": freshness.series_status,
                    },
                    "config": {
                        "regions": regions,
                        "fuel_types": fuel_types,
                        "data_dir": data_dir,
                    },
                }

                # Write skip log (append to run_log.json)
                Path(data_dir).mkdir(parents=True, exist_ok=True)
                skip_log_path = Path(data_dir) / "skip_log.json"
                skip_log_path.write_text(
                    json.dumps(skip_log, indent=2, default=_json_default)
                )

                print(f"Skip log written to: {skip_log_path}")

                # Set output for GitHub Actions
                github_output = os.getenv("GITHUB_OUTPUT")
                if github_output:
                    with open(github_output, "a") as f:
                        f.write("status=skipped\n")

                return skip_log

            # Freshness check passed - log details
            print(f"Freshness check: {freshness.summary}")
            print("[freshness][DETAILS] Per-series status:")
            for series_id, status in freshness.series_status.items():
                prev = status.get("prev_max_ds", "N/A")
                current = status.get("current_max_ds", "N/A")
                is_new = status.get("is_new", False)
                lag = status.get("lag_hours", "N/A")
                stale_flag = status.get("is_stale", False)

                status_str = "NEW" if is_new else "unchanged"
                if stale_flag:
                    status_str += " (STALE)"

                print(
                    f"  {series_id}: prev={prev} current={current} "
                    f"lag={lag}h ({status_str})"
                )
    else:
        print(
            "[pipeline] FORCE_RUN=true - "
            "bypassing freshness check (manual run requested)"
        )
        print("[pipeline] Pipeline will run regardless of data freshness")

    # Pipeline execution timing
    pipeline_start_time = datetime.now(timezone.utc)
    if 'freshness_check_start' in locals():
        elapsed_since_freshness = (
            (pipeline_start_time - freshness_check_start).total_seconds() / 60.0
        )
        print(
            f"[pipeline][TIMING] Pipeline starting at "
            f"{pipeline_start_time.isoformat()}"
        )
        print(
            f"[pipeline][TIMING] Elapsed since freshness check: "
            f"{elapsed_since_freshness:.2f} minutes"
        )
    else:
        print(
            f"[pipeline][TIMING] Pipeline starting at "
            f"{pipeline_start_time.isoformat()} (FORCE_RUN, no freshness check)"
        )

    n_jobs = _env_int("RENEWABLE_N_JOBS", 1)

    cfg = RenewablePipelineConfig(
        regions=regions,
        fuel_types=fuel_types,
        lookback_days=lookback_days,
        horizon=horizon,
        horizon_preset=horizon_preset,  # Apply preset if specified
        cv_windows=cv_windows,
        cv_step_size=cv_step_size,
        n_jobs=n_jobs,
        data_dir=data_dir,
        overwrite=True,
        start_date=start_date,
        end_date=end_date,
    )
    print(f"[config] n_jobs={n_jobs} (RENEWABLE_N_JOBS)")

    # Add option to skip EDA for fast iteration
    skip_eda = os.getenv("SKIP_EDA", "false").lower() == "true"

    fetch_diagnostics: list[dict] = []
    results = run_full_pipeline(
        cfg,
        fetch_diagnostics=fetch_diagnostics,
        skip_eda=skip_eda,
        max_lag_hours=max_lag_hours,
        max_missing_ratio=max_missing_ratio,
        validation_debug=validation_debug,
    )

    # Validation timing
    validation_start_time = datetime.now(timezone.utc)
    elapsed_since_pipeline = (
        (validation_start_time - pipeline_start_time).total_seconds() / 60.0
    )
    print(
        f"[pipeline][TIMING] Validation starting at "
        f"{validation_start_time.isoformat()}"
    )
    print(
        f"[pipeline][TIMING] Elapsed since pipeline start: "
        f"{elapsed_since_pipeline:.2f} minutes"
    )

    gen_path = cfg.generation_path()
    gen_df = pd.read_parquet(gen_path)
    generation_coverage = _summarize_generation_coverage(gen_df)

    # Respect series dropped by the pipeline's gap filter so that the
    # post-pipeline validation uses the same expected-series list.
    dropped_series = (
        results.get("gap_filter", {}).get("series_dropped", [])
    )
    active_series = [
        s for s in _expected_series(regions, fuel_types)
        if s not in dropped_series
    ]
    if dropped_series:
        print(
            f"[validation] Excluding {len(dropped_series)} gap-filtered "
            f"series from post-pipeline check: {dropped_series}"
        )

    report = validate_generation_df(
        gen_df,
        max_lag_hours=max_lag_hours,
        max_missing_ratio=max_missing_ratio,
        expected_series=active_series,
        region_lag_thresholds=(
            region_lag_thresholds if region_lag_thresholds else None
        ),
        debug=validation_debug,
    )

    forecasts_df = pd.read_parquet(cfg.forecasts_path())
    negative_forecasts = _summarize_negative_forecasts(forecasts_df)

    # Quality gates
    max_rowdrop_pct = _env_float("MAX_ROWDROP_PCT", 0.30)
    max_neg_forecast_ratio = _env_float("MAX_NEG_FORECAST_RATIO", 0.10)

    prev_run = _read_previous_run_summary(data_dir)
    prev_gen_rows = 0
    if prev_run:
        prev_gen_rows = prev_run.get("pipeline_results", {}).get("generation_rows", 0)

    # If the gap filter intentionally dropped series, adjust the baseline
    # so the rowdrop gate compares like-for-like (same number of series).
    dropped_series = results.get("gap_filter", {}).get("series_dropped", [])
    if dropped_series and prev_gen_rows > 0:
        total_series = len(_expected_series(regions, fuel_types))
        active_count = total_series - len(dropped_series)
        if total_series > 0 and active_count > 0:
            adjusted_prev = int(prev_gen_rows * active_count / total_series)
            print(
                f"[quality_gate] Rowdrop baseline adjusted for "
                f"{len(dropped_series)} gap-filtered series: "
                f"{prev_gen_rows} → {adjusted_prev}"
            )
            prev_gen_rows = adjusted_prev

    curr_gen_rows = results.get("generation_rows", 0)
    rowdrop_ok = True
    if prev_gen_rows > 0:
        floor_ok = int(prev_gen_rows * (1.0 - max_rowdrop_pct))
        rowdrop_ok = curr_gen_rows >= floor_ok

    neg_forecast_ratio = 0.0
    if negative_forecasts["row_count"] > 0:
        neg_forecast_ratio = (
            negative_forecasts["negative_rows"] / negative_forecasts["row_count"]
        )
    neg_forecast_ok = neg_forecast_ratio <= max_neg_forecast_ratio

    quality_gates = {
        "rowdrop": {
            "ok": rowdrop_ok,
            "prev_rows": prev_gen_rows,
            "curr_rows": curr_gen_rows,
            "max_rowdrop_pct": max_rowdrop_pct,
        },
        "neg_forecast": {
            "ok": neg_forecast_ok,
            "ratio": neg_forecast_ratio,
            "max_ratio": max_neg_forecast_ratio,
        },
    }

    run_log = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "regions": regions,
            "fuel_types": fuel_types,
            "lookback_days": lookback_days,
            "horizon": horizon,
            "cv_windows": cv_windows,
            "cv_step_size": cv_step_size,
            "data_dir": data_dir,
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
        },
        "pipeline_results": results,
        "validation": {
            "ok": report.ok,
            "message": report.message,
            "details": report.details,
            "max_lag_hours": max_lag_hours,
            "max_missing_ratio": max_missing_ratio,
            "validation_debug": validation_debug,
        },
        "diagnostics": {
            "fetch": fetch_diagnostics,
            "generation_coverage": generation_coverage,
            "negative_forecasts": negative_forecasts,
        },
        "quality_gates": quality_gates,
    }

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    # Sanitize data to replace NaN/Infinity with None (valid JSON)
    # NaN/Infinity are not valid JSON per RFC 8259
    sanitized_log = _sanitize_for_json(run_log)
    # Use allow_nan=False to catch any NaN that slipped through
    (Path(data_dir) / "run_log.json").write_text(
        json.dumps(sanitized_log, indent=2, default=_json_default, allow_nan=False)
    )

    # Check validation
    if not report.ok:
        raise SystemExit(f"VALIDATION_FAILED: {report.message} | {report.details}")

    # Check quality gates
    if not rowdrop_ok:
        raise SystemExit(
            f"QUALITY_GATE_FAILED: rowdrop | "
            f"curr={curr_gen_rows} prev={prev_gen_rows} max_drop={max_rowdrop_pct:.0%}"
        )
    if not neg_forecast_ok:
        raise SystemExit(
            f"QUALITY_GATE_FAILED: neg_forecast | "
            f"ratio={neg_forecast_ratio:.1%} max={max_neg_forecast_ratio:.0%}"
        )

    # Set output for GitHub Actions (successful run)
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write("status=success\n")

    return run_log


def main() -> None:
    run_hourly_pipeline()


if __name__ == "__main__":
    main()
