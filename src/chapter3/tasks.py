# file: src/chapter3/tasks.py
"""
Chapter 3: Idempotent Pipeline Tasks

These tasks are designed to be:
- deterministic for a given config (historical pulls)
- atomic on write
- safe to rerun (overwrite flag controls)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import pandas as pd

from src.chapter3.config import PipelineConfig
from src.chapter3.io_utils import atomic_write_json, atomic_write_parquet, ensure_dir

logger = logging.getLogger(__name__)


def _import_fetcher():
    """
    Supports either:
      - eia_data_simple.py at repo root
      - or src/chapter1/eia_data_simple.py (if you later move it)
    """
    try:
        from eia_data_simple import EIADataFetcher, ExperimentConfig  # type: ignore
        return EIADataFetcher, ExperimentConfig
    except Exception:
        from ..chapter1.eia_data_simple import EIADataFetcher, ExperimentConfig  # type: ignore
        return EIADataFetcher, ExperimentConfig


def _require_api_key() -> str:
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "EIA_API_KEY is missing. Add it to your environment or .env file."
        )
    return api_key


def compute_time_series_integrity(df_forecast: pd.DataFrame) -> Dict:
    """
    Computes the same core invariants as your fetcher’s integrity method,
    but returns a dict reliably (your current method prints but doesn’t return).
    Requires columns: unique_id, ds, y
    """
    if not {"unique_id", "ds", "y"}.issubset(df_forecast.columns):
        raise ValueError(f"Expected unique_id/ds/y, got {df_forecast.columns.tolist()}")

    df = df_forecast.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # duplicates on (unique_id, ds)
    dup_counts = df.groupby(["unique_id", "ds"]).size()
    duplicate_pairs = int((dup_counts > 1).sum())

    missing_hours = 0
    longest_gap_hours = 0.0
    gaps_detail = []

    for uid in df["unique_id"].unique():
        sub = df[df["unique_id"] == uid].sort_values("ds").reset_index(drop=True)
        diffs = sub["ds"].diff()
        expected = pd.Timedelta(hours=1)

        # gaps > 1 hour
        miss_mask = diffs > expected
        missing_hours += int(miss_mask.sum())

        if len(diffs) > 0 and diffs.notna().any():
            max_gap = diffs.max()
            if pd.notna(max_gap):
                gap_h = max_gap.total_seconds() / 3600
                longest_gap_hours = max(longest_gap_hours, gap_h)

        if miss_mask.any():
            idxs = sub.index[miss_mask].tolist()
            for idx in idxs[:10]:
                gaps_detail.append(
                    {
                        "unique_id": uid,
                        "before_ds": sub.loc[idx - 1, "ds"],
                        "after_ds": sub.loc[idx, "ds"],
                        "gap_hours": float((sub.loc[idx, "ds"] - sub.loc[idx - 1, "ds"]).total_seconds() / 3600),
                    }
                )

    status = "valid" if (duplicate_pairs == 0 and missing_hours == 0) else "invalid"
    return {
        "status": status,
        "duplicate_pairs": duplicate_pairs,
        "missing_hours": int(missing_hours),
        "longest_gap_hours": float(longest_gap_hours),
        "gaps_detail": gaps_detail,
        "n_rows": int(len(df)),
    }


def ingest_eia(config: PipelineConfig, run_id: str = "") -> str:
    """
    Task 1: Pull raw data and save data/raw.parquet
    """
    raw_path = config.raw_path()
    ensure_dir(raw_path.parent)

    if raw_path.exists() and not config.overwrite:
        logger.info(f"[ingest] raw exists, skipping: {raw_path}")
        return str(raw_path)

    EIADataFetcher, _ = _import_fetcher()
    api_key = _require_api_key()
    fetcher = EIADataFetcher(api_key)

    df_raw = fetcher.pull_data(
        start_date=config.start_date,
        end_date=config.end_date,
        respondent=config.respondent,
        fueltype=config.fueltype,
    )

    atomic_write_parquet(df_raw, raw_path)
    logger.info(f"[ingest] wrote raw: {raw_path} ({len(df_raw)} rows)")

    # Optional: log to Chapter 4 monitoring
    if run_id:
        try:
            from src.chapter4.run_log import log_run
            from src.chapter4.config import MonitoringConfig
            log_run(
                MonitoringConfig().db_path,
                run_id,
                "success",
                "ingest",
                raw_rows=len(df_raw)
            )
        except Exception as e:
            logger.debug(f"[ingest] Chapter 4 logging failed: {e}")

    return str(raw_path)


def prepare_clean(raw_path: str, config: PipelineConfig, run_id: str = "") -> str:
    """
    Task 2: Prepare clean dataset and save data/clean.parquet + data/metadata.json
    """
    clean_path = config.clean_path()
    ensure_dir(clean_path.parent)

    if clean_path.exists() and not config.overwrite:
        logger.info(f"[prepare] clean exists, skipping: {clean_path}")
        return str(clean_path)

    EIADataFetcher, _ = _import_fetcher()
    api_key = os.getenv("EIA_API_KEY", "dummy")  # no API call in this step
    fetcher = EIADataFetcher(api_key)

    df_raw = pd.read_parquet(raw_path)
    df_clean = fetcher.prepare_data(df_raw, timezone_policy="UTC")

    # Keep metadata for Chapter 4 health checks
    metadata = {
        "pull_timestamp": datetime.now(timezone.utc).isoformat(),
        "respondent": config.respondent,
        "fueltype": config.fueltype,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "raw_rows": int(len(df_raw)),
        "clean_rows": int(len(df_clean)),
    }

    atomic_write_parquet(df_clean, clean_path)
    atomic_write_json(metadata, config.metadata_path())

    logger.info(f"[prepare] wrote clean: {clean_path} ({len(df_clean)} rows)")

    # Optional: log to Chapter 4 monitoring
    if run_id:
        try:
            from src.chapter4.run_log import log_run
            from src.chapter4.config import MonitoringConfig
            log_run(
                MonitoringConfig().db_path,
                run_id,
                "success",
                "prepare",
                clean_rows=len(df_clean)
            )
        except Exception as e:
            logger.debug(f"[prepare] Chapter 4 logging failed: {e}")

    return str(clean_path)


def validate_clean(clean_path: str, run_id: str = "") -> Dict:
    """
    Task 3: Validate time series integrity (duplicates + missing hours).
    """
    EIADataFetcher, _ = _import_fetcher()
    api_key = os.getenv("EIA_API_KEY", "dummy")
    fetcher = EIADataFetcher(api_key)

    df_clean = pd.read_parquet(clean_path)
    df_forecast = fetcher.prepare_for_forecasting(df_clean, unique_id="NG_US48")

    report = compute_time_series_integrity(df_forecast)
    if report["status"] != "valid":
        raise ValueError(
            f"Time-series integrity failed: "
            f"{report['duplicate_pairs']} duplicate (unique_id, ds) pairs; "
            f"{report['missing_hours']} missing hours; "
            f"longest gap={report['longest_gap_hours']:.1f}h"
        )

    logger.info(f"[validate] OK: {report}")

    # Optional: log to Chapter 4 monitoring
    if run_id:
        try:
            from src.chapter4.run_log import log_run
            from src.chapter4.config import MonitoringConfig
            log_run(
                MonitoringConfig().db_path,
                run_id,
                "success",
                "validate"
            )
        except Exception as e:
            logger.debug(f"[validate] Chapter 4 logging failed: {e}")

    return report


def train_backtest_select(clean_path: str, config: PipelineConfig, run_id: str = "") -> pd.DataFrame:
    """
    Task 4: Cross-validate and write artifacts/leaderboard.parquet + artifacts/cv_results.parquet
    Returns leaderboard.
    """
    leaderboard_path = config.leaderboard_path()
    cv_path = config.cv_results_path()
    ensure_dir(leaderboard_path.parent)

    if leaderboard_path.exists() and not config.overwrite:
        logger.info(f"[train] leaderboard exists, skipping: {leaderboard_path}")
        return pd.read_parquet(leaderboard_path)

    EIADataFetcher, ExperimentConfig = _import_fetcher()
    api_key = os.getenv("EIA_API_KEY", "dummy")
    fetcher = EIADataFetcher(api_key)

    df_clean = pd.read_parquet(clean_path)
    df_forecast = fetcher.prepare_for_forecasting(df_clean, unique_id="NG_US48")

    exp = ExperimentConfig(
        name=config.experiment_name,
        horizon=config.horizon,
        n_windows=config.n_windows,
        step_size=config.step_size,
        confidence_level=config.confidence_level,
    )

    cv_results, leaderboard = fetcher.cross_validate(df_forecast, config=exp)

    atomic_write_parquet(cv_results, cv_path)
    atomic_write_parquet(leaderboard, leaderboard_path)

    logger.info(f"[train] wrote cv: {cv_path} ({len(cv_results)} rows)")
    logger.info(f"[train] wrote leaderboard: {leaderboard_path} ({len(leaderboard)} rows)")

    # Optional: log to Chapter 4 monitoring
    if run_id:
        try:
            from src.chapter4.run_log import log_run
            from src.chapter4.config import MonitoringConfig
            log_run(
                MonitoringConfig().db_path,
                run_id,
                "success",
                "train"
            )
        except Exception as e:
            logger.debug(f"[train] Chapter 4 logging failed: {e}")

    return leaderboard


def register_champion(leaderboard: pd.DataFrame, config: PipelineConfig, clean_path: str, run_id: str = "") -> Optional[str]:
    """
    Task 5: Register best model in MLflow (if available).
    """
    EIADataFetcher, _ = _import_fetcher()
    api_key = os.getenv("EIA_API_KEY", "dummy")
    fetcher = EIADataFetcher(api_key)

    df_clean = pd.read_parquet(clean_path)
    df_forecast = fetcher.prepare_for_forecasting(df_clean, unique_id="NG_US48")

    model_uri = fetcher.register_best_model(
        leaderboard=leaderboard,
        experiment_name=config.experiment_name,
        alias=config.model_alias,
        train_df=df_forecast,
        default_horizon=config.horizon,
        freq="h",
    )
    logger.info(f"[register] model_uri={model_uri}")

    # Optional: log to Chapter 4 monitoring
    if run_id:
        try:
            from src.chapter4.run_log import log_run
            from src.chapter4.config import MonitoringConfig
            log_run(
                MonitoringConfig().db_path,
                run_id,
                "success",
                "register"
            )
        except Exception as e:
            logger.debug(f"[register] Chapter 4 logging failed: {e}")

    return model_uri



def forecast_publish(clean_path: str, config: PipelineConfig, run_id: str = "") -> str:
    """
    Task 6: Fit on all clean data and publish artifacts/predictions.parquet
    """
    pred_path = config.predictions_path()
    ensure_dir(pred_path.parent)

    if pred_path.exists() and not config.overwrite:
        logger.info(f"[forecast] predictions exist, skipping: {pred_path}")
        return str(pred_path)

    EIADataFetcher, _ = _import_fetcher()
    api_key = os.getenv("EIA_API_KEY", "dummy")
    fetcher = EIADataFetcher(api_key)

    df_clean = pd.read_parquet(clean_path)
    df_train = fetcher.prepare_for_forecasting(df_clean, unique_id="NG_US48")

    forecast_df = fetcher.forecast(
        train_df=df_train,
        horizon=config.horizon,
        confidence_level=config.confidence_level,
    )

    atomic_write_parquet(forecast_df, pred_path)
    logger.info(f"[forecast] wrote predictions: {pred_path} ({len(forecast_df)} rows)")

    # Optional: persist to Chapter 4 monitoring store
    if run_id:
        try:
            from src.chapter4.forecast_store import persist_forecasts
            from src.chapter4.config import MonitoringConfig

            persist_forecasts(
                db_path=MonitoringConfig().db_path,
                run_id=run_id,
                forecast_df=forecast_df,
                confidence_level=config.confidence_level,
            )
            logger.info(f"[forecast] persisted to Chapter 4 store")
        except ImportError:
            logger.debug("[forecast] Chapter 4 not available, skipping persist")
        except Exception as e:
            logger.warning(f"[forecast] Chapter 4 persist failed: {e}")

    # Optional: log to Chapter 4 monitoring
    if run_id:
        try:
            from src.chapter4.run_log import log_run
            from src.chapter4.config import MonitoringConfig
            log_run(
                MonitoringConfig().db_path,
                run_id,
                "success",
                "forecast"
            )
        except Exception as e:
            logger.debug(f"[forecast] Chapter 4 logging failed: {e}")

    return str(pred_path)


def run_full_pipeline(config: PipelineConfig) -> Dict:
    """
    Runs tasks in order and returns a summary dict.
    """
    logger.info("=" * 60)
    logger.info("START PIPELINE")
    logger.info("=" * 60)

    # Compute run_id once and pass through all tasks (for Chapter 4 monitoring)
    run_id = config.run_id()
    logger.info(f"Pipeline run_id: {run_id}")

    raw = ingest_eia(config, run_id=run_id)
    clean = prepare_clean(raw, config, run_id=run_id)
    integrity = validate_clean(clean, run_id=run_id)
    leaderboard = train_backtest_select(clean, config, run_id=run_id)
    model_uri = register_champion(leaderboard, config, clean, run_id=run_id)
    predictions = forecast_publish(clean, config, run_id=run_id)

    best_model = leaderboard.iloc[0]["model"] if len(leaderboard) else None
    best_rmse = leaderboard.iloc[0]["rmse_mean"] if len(leaderboard) else None

    out = {
        "raw_path": raw,
        "clean_path": clean,
        "integrity": integrity,
        "leaderboard_path": str(config.leaderboard_path()),
        "cv_results_path": str(config.cv_results_path()),
        "predictions_path": predictions,
        "best_model": best_model,
        "best_rmse_mean": best_rmse,
        "model_uri": model_uri,
        "run_id": run_id,
    }

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    return out
