from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from src.chapter1.validate import validate_time_index
from src.chapter3.io_utils import atomic_write_json, atomic_write_parquet, ensure_dir
from src.chapter4.forecast_store import persist_forecasts

from .airnow import load_airnow_api_key, prepare_airnow_for_forecasting, pull_airnow_historical
from .config import AQIPipelineConfig
from .modeling import (
    apply_residual_corrector,
    forecast_baseline,
    run_baseline_backtest,
    train_residual_corrector,
)
from .open_meteo import prepare_weather, pull_open_meteo

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ingest_airnow(config: AQIPipelineConfig, run_id: str = "") -> str:
    raw_path = config.raw_airnow_path()
    ensure_dir(raw_path.parent)

    if raw_path.exists() and not config.overwrite:
        logger.info("[airnow-ingest] raw exists, skipping: %s", raw_path)
        return str(raw_path)

    api_key = load_airnow_api_key()
    df_raw = pull_airnow_historical(config, api_key)
    atomic_write_parquet(df_raw, raw_path)
    logger.info("[airnow-ingest] wrote raw: %s (%s rows)", raw_path, len(df_raw))
    return str(raw_path)


def ingest_weather(config: AQIPipelineConfig, run_id: str = "") -> str:
    raw_path = config.raw_weather_path()
    ensure_dir(raw_path.parent)

    if raw_path.exists() and not config.overwrite:
        logger.info("[weather-ingest] raw exists, skipping: %s", raw_path)
        return str(raw_path)

    df_raw = pull_open_meteo(config)
    atomic_write_parquet(df_raw, raw_path)
    logger.info("[weather-ingest] wrote raw: %s (%s rows)", raw_path, len(df_raw))
    return str(raw_path)


def prepare_aqi(raw_path: str, config: AQIPipelineConfig, run_id: str = "") -> str:
    clean_path = config.clean_aqi_path()
    ensure_dir(clean_path.parent)

    if clean_path.exists() and not config.overwrite:
        logger.info("[aqi-prepare] clean exists, skipping: %s", clean_path)
        return str(clean_path)

    df_raw = pd.read_parquet(raw_path)
    df_aqi = prepare_airnow_for_forecasting(df_raw, config)

    metadata = {
        "pull_timestamp": _utc_iso(),
        "location_name": config.location_name,
        "latitude": config.latitude,
        "longitude": config.longitude,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "raw_rows": int(len(df_raw)),
        "clean_rows": int(len(df_aqi)),
    }

    atomic_write_parquet(df_aqi, clean_path)
    atomic_write_json(metadata, config.metadata_path())
    logger.info("[aqi-prepare] wrote clean: %s (%s rows)", clean_path, len(df_aqi))
    return str(clean_path)


def prepare_weather_data(raw_path: str, config: AQIPipelineConfig, run_id: str = "") -> str:
    clean_path = config.clean_weather_path()
    ensure_dir(clean_path.parent)

    if clean_path.exists() and not config.overwrite:
        logger.info("[weather-prepare] clean exists, skipping: %s", clean_path)
        return str(clean_path)

    df_raw = pd.read_parquet(raw_path)
    df_weather = prepare_weather(df_raw, config)
    atomic_write_parquet(df_weather, clean_path)
    logger.info("[weather-prepare] wrote clean: %s (%s rows)", clean_path, len(df_weather))
    return str(clean_path)


def merge_features(
    clean_aqi_path: str,
    clean_weather_path: str,
    config: AQIPipelineConfig,
    run_id: str = "",
) -> str:
    merged_path = config.merged_path()
    ensure_dir(merged_path.parent)

    if merged_path.exists() and not config.overwrite:
        logger.info("[merge] merged exists, skipping: %s", merged_path)
        return str(merged_path)

    df_aqi = pd.read_parquet(clean_aqi_path)
    df_weather = pd.read_parquet(clean_weather_path)

    df_merged = df_aqi.merge(df_weather, on="ds", how="left")
    atomic_write_parquet(df_merged, merged_path)
    logger.info("[merge] wrote merged: %s (%s rows)", merged_path, len(df_merged))
    return str(merged_path)


def validate_aqi(clean_path: str, config: AQIPipelineConfig, run_id: str = "") -> Dict:
    df_aqi = pd.read_parquet(clean_path)
    result = validate_time_index(df_aqi)

    if not result.is_valid:
        raise ValueError(
            f"Time-series integrity failed: {result.n_duplicates} duplicates; "
            f"{result.n_missing_hours} missing hours; monotonic={result.is_monotonic}"
        )

    return {
        "status": "valid",
        "rows": result.n_rows,
        "duplicates": result.n_duplicates,
        "missing_hours": result.n_missing_hours,
    }


def train_backtest_select(clean_path: str, config: AQIPipelineConfig, run_id: str = "") -> Dict:
    df_aqi = pd.read_parquet(clean_path)

    cv_df, leaderboard = run_baseline_backtest(df_aqi, config)

    atomic_write_parquet(cv_df, config.cv_results_path())
    atomic_write_parquet(leaderboard, config.leaderboard_path())

    logger.info(
        "[train] cv_results=%s leaderboard=%s",
        config.cv_results_path(),
        config.leaderboard_path(),
    )

    return {
        "cv_results_path": str(config.cv_results_path()),
        "leaderboard_path": str(config.leaderboard_path()),
    }


def train_residual_model(
    clean_weather_path: str,
    cv_results_path: str,
    leaderboard_path: str,
    config: AQIPipelineConfig,
    run_id: str = "",
) -> str:
    from joblib import dump

    df_weather = pd.read_parquet(clean_weather_path)
    cv_df = pd.read_parquet(cv_results_path)
    leaderboard = pd.read_parquet(leaderboard_path)

    payload = train_residual_corrector(cv_df, leaderboard, df_weather, config)
    ensure_dir(config.residual_model_path().parent)
    dump(payload, config.residual_model_path())
    logger.info("[residual] wrote model: %s", config.residual_model_path())
    return str(config.residual_model_path())


def forecast_publish(
    clean_aqi_path: str,
    clean_weather_path: str,
    config: AQIPipelineConfig,
    run_id: str = "",
) -> str:
    df_aqi = pd.read_parquet(clean_aqi_path)
    forecast_df = forecast_baseline(df_aqi, config)

    if config.use_residual and config.residual_model_path().exists():
        from joblib import load

        df_weather = pd.read_parquet(clean_weather_path)
        payload = load(config.residual_model_path())
        forecast_df = apply_residual_corrector(
            forecast_df=forecast_df,
            weather_df=df_weather,
            payload=payload,
            config=config,
        )

    atomic_write_parquet(forecast_df, config.predictions_path())
    logger.info("[forecast] wrote predictions: %s", config.predictions_path())

    try:
        from src.chapter4.config import MonitoringConfig

        persist_forecasts(
            MonitoringConfig().db_path,
            run_id or config.run_id(),
            forecast_df,
            confidence_level=config.confidence_level,
        )
    except Exception as exc:
        logger.debug("[forecast] persist_forecasts skipped: %s", exc)

    return str(config.predictions_path())


def run_full_pipeline(config: AQIPipelineConfig) -> Dict:
    run_id = config.run_id()

    raw_airnow = ingest_airnow(config, run_id=run_id)
    raw_weather = ingest_weather(config, run_id=run_id)
    clean_aqi = prepare_aqi(raw_airnow, config, run_id=run_id)
    clean_weather = prepare_weather_data(raw_weather, config, run_id=run_id)
    merged_path = merge_features(clean_aqi, clean_weather, config, run_id=run_id)
    validation = validate_aqi(clean_aqi, config, run_id=run_id)
    train_info = train_backtest_select(clean_aqi, config, run_id=run_id)

    residual_path = None
    if config.use_residual:
        try:
            residual_path = train_residual_model(
                clean_weather_path=clean_weather,
                cv_results_path=train_info["cv_results_path"],
                leaderboard_path=train_info["leaderboard_path"],
                config=config,
                run_id=run_id,
            )
        except Exception as exc:
            logger.warning("[residual] skipped: %s", exc)

    forecast_path = forecast_publish(
        clean_aqi_path=clean_aqi,
        clean_weather_path=clean_weather,
        config=config,
        run_id=run_id,
    )

    return {
        "run_id": run_id,
        "raw_airnow": raw_airnow,
        "raw_weather": raw_weather,
        "clean_aqi": clean_aqi,
        "clean_weather": clean_weather,
        "merged_features": merged_path,
        "validation": validation,
        "cv_results": train_info["cv_results_path"],
        "leaderboard": train_info["leaderboard_path"],
        "residual_model": residual_path,
        "forecast": forecast_path,
    }
