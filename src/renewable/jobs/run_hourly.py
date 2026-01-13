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

    results = run_full_pipeline(cfg)

    gen_path = cfg.generation_path()
    gen_df = pd.read_parquet(gen_path)

    max_lag_hours = _env_int("MAX_LAG_HOURS", 3)
    max_missing_ratio = _env_float("MAX_MISSING_RATIO", 0.02)
    report = validate_generation_df(
        gen_df,
        max_lag_hours=max_lag_hours,
        max_missing_ratio=max_missing_ratio,
        expected_series=_expected_series(regions, fuel_types),
    )

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
    }

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    (Path(data_dir) / "run_log.json").write_text(
        json.dumps(run_log, indent=2, default=_json_default)
    )

    if not report.ok:
        raise SystemExit(f"VALIDATION_FAILED: {report.message} | {report.details}")

    return run_log


def main() -> None:
    run_hourly_pipeline()


if __name__ == "__main__":
    main()
