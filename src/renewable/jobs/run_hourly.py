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

load_dotenv()


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


def _expected_series(regions: list[str], fuel_types: list[str]) -> list[str]:
    return [f"{region}_{fuel}" for region in regions for fuel in fuel_types]


def _json_default(value: object) -> str:
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
    horizon = _env_int("RENEWABLE_HORIZON", 24)
    cv_windows = _env_int("RENEWABLE_CV_WINDOWS", 2)
    cv_step_size = _env_int("RENEWABLE_CV_STEP_SIZE", 168)

    start_date = os.getenv("RENEWABLE_START_DATE", "")
    end_date = os.getenv("RENEWABLE_END_DATE", "")

    cfg = RenewablePipelineConfig(
        regions=regions,
        fuel_types=fuel_types,
        lookback_days=lookback_days,
        horizon=horizon,
        data_dir=data_dir,
        overwrite=True,
        start_date=start_date,
        end_date=end_date,
    )
    cfg.cv_windows = cv_windows
    cfg.cv_step_size = cv_step_size

    fetch_diagnostics: list[dict] = []
    results = run_full_pipeline(cfg, fetch_diagnostics=fetch_diagnostics)

    gen_path = cfg.generation_path()
    gen_df = pd.read_parquet(gen_path)
    generation_coverage = _summarize_generation_coverage(gen_df)

    max_lag_hours = _env_int("MAX_LAG_HOURS", 48)  # EIA publishes with 12-24h delay
    max_missing_ratio = _env_float("MAX_MISSING_RATIO", 0.02)
    report = validate_generation_df(
        gen_df,
        max_lag_hours=max_lag_hours,
        max_missing_ratio=max_missing_ratio,
        expected_series=_expected_series(regions, fuel_types),
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
        },
        "diagnostics": {
            "fetch": fetch_diagnostics,
            "generation_coverage": generation_coverage,
            "negative_forecasts": negative_forecasts,
        },
        "quality_gates": quality_gates,
    }

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    (Path(data_dir) / "run_log.json").write_text(
        json.dumps(run_log, indent=2, default=_json_default)
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

    return run_log


def main() -> None:
    run_hourly_pipeline()


if __name__ == "__main__":
    main()
