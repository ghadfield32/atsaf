# file: src/renewable/tasks.py
"""Renewable energy forecasting pipeline tasks.

Idempotent tasks for:
- Fetching EIA renewable generation data
- Fetching weather data from Open-Meteo
- Training probabilistic models
- Generating forecasts with intervals
- Computing drift metrics
"""

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Optional imports for interpretability (LightGBM + skforecast)
try:
    from lightgbm import LGBMRegressor
    from skforecast.recursive import ForecasterRecursive
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    INTERPRETABILITY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("lightgbm and/or skforecast not installed - interpretability features unavailable")

from src.renewable.eia_renewable import EIARenewableFetcher
from src.renewable.modeling import (
    RenewableForecastModel,
    compute_baseline_metrics,
    enforce_physical_constraints,
    WEATHER_VARS,
)
from src.renewable.model_interpretability import (
    InterpretabilityReport,
    generate_full_interpretability_report,
)
from src.renewable.open_meteo import OpenMeteoRenewable
from src.renewable.dataset_builder import _add_time_features, build_modeling_dataset

logger = logging.getLogger(__name__)


@dataclass
class RenewablePipelineConfig:
    """Configuration for renewable forecasting pipeline."""

    # Data parameters
    regions: list[str] = field(default_factory=lambda: ["CALI", "ERCO", "MISO", "PJM", "SWPP"])
    fuel_types: list[str] = field(default_factory=lambda: ["WND", "SUN"])
    start_date: str = ""  # Set dynamically
    end_date: str = ""  # Set dynamically
    lookback_days: int = 30

    # Forecast parameters
    horizon: int = 24
    confidence_levels: tuple[int, int] = (80, 95)
    horizon_preset: Optional[str] = None  # "24h" | "48h" | "72h"

    # CV parameters
    cv_windows: int = 5
    cv_step_size: int = 168  # 1 week

    # Model parameters
    enable_interpretability: bool = True  # LightGBM SHAP analysis (on by default)

    # Preprocessing parameters
    negative_policy: str = "clamp"  # "clamp" | "fail_loud" | "hybrid"
    hourly_grid_policy: str = "drop_incomplete_series"  # "drop_incomplete_series" | "fail_loud"

    # Output paths
    data_dir: str = "data/renewable"
    overwrite: bool = False

    # Horizon preset definitions (class-level constant)
    _PRESETS = {
        "24h": {"horizon": 24, "cv_windows": 2, "lookback_days": 15},
        "48h": {"horizon": 48, "cv_windows": 3, "lookback_days": 21},
        "72h": {"horizon": 72, "cv_windows": 3, "lookback_days": 28},
    }

    def __post_init__(self):
        # Apply horizon preset if specified
        if self.horizon_preset and self.horizon_preset in self._PRESETS:
            preset = self._PRESETS[self.horizon_preset]
            # Use object.__setattr__ since this is a dataclass
            object.__setattr__(self, "horizon", preset["horizon"])
            object.__setattr__(self, "cv_windows", preset["cv_windows"])
            object.__setattr__(self, "lookback_days", preset["lookback_days"])
            logger.info(f"[config] Applied preset '{self.horizon_preset}': horizon={preset['horizon']}h")

        # Set default dates if not provided
        if not self.end_date:
            self.end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not self.start_date:
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
            start = end - timedelta(days=self.lookback_days)
            self.start_date = start.strftime("%Y-%m-%d")

        # Validate configuration
        warnings = self._validate()
        for warning in warnings:
            logger.warning(f"[config] {warning}")

    def _validate(self) -> list[str]:
        """Validate configuration and return warnings."""
        warnings = []

        # Check minimum data requirement
        available_hours = self.lookback_days * 24
        required_hours = self.horizon + (self.cv_windows * self.cv_step_size)
        if available_hours < required_hours:
            warnings.append(
                f"Insufficient data: need {required_hours}h, have {available_hours}h. "
                f"Increase lookback_days to {(required_hours // 24) + 1} or reduce cv_windows."
            )

        # Warn about accuracy degradation
        if self.horizon > 72:
            warnings.append(
                f"Horizon {self.horizon}h exceeds recommended max (72h). "
                f"Weather forecast accuracy degrades significantly beyond 3 days."
            )

        return warnings

    def generation_path(self) -> Path:
        return Path(self.data_dir) / "generation.parquet"

    def weather_path(self) -> Path:
        return Path(self.data_dir) / "weather.parquet"

    def forecasts_path(self) -> Path:
        return Path(self.data_dir) / "forecasts.parquet"

    def baseline_path(self) -> Path:
        return Path(self.data_dir) / "baseline.json"

    def interpretability_dir(self) -> Path:
        return Path(self.data_dir) / "interpretability"

    def preprocessing_dir(self) -> Path:
        return Path(self.data_dir) / "preprocessing"


def fetch_renewable_data(
    config: RenewablePipelineConfig,
    fetch_diagnostics: Optional[list[dict]] = None,
) -> pd.DataFrame:
    """Task 1: Fetch EIA generation data for all regions and fuel types.

    Args:
        config: Pipeline configuration
        fetch_diagnostics: Optional list to capture per-region fetch metadata

    Returns:
        DataFrame with columns [unique_id, ds, y]
    """
    output_path = config.generation_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _log_generation_summary(df: pd.DataFrame, source: str) -> None:


        expected_series = {
            f"{region}_{fuel}" for region in config.regions for fuel in config.fuel_types
        }
        present_series = set(df["unique_id"]) if "unique_id" in df.columns else set()
        missing_series = sorted(expected_series - present_series)
        if missing_series:
            logger.warning(
                "[fetch_generation] Missing expected series (%s): %s",
                source,
                missing_series,
            )

        if df.empty:
            logger.warning("[fetch_generation] No generation data rows (%s).", source)
            return

        coverage = (
            df.groupby("unique_id")["ds"]
            .agg(min_ds="min", max_ds="max", rows="count")
            .reset_index()
            .sort_values("unique_id")
        )
        max_series_log = 25
        if len(coverage) > max_series_log:
            logger.info(
                "[fetch_generation] Coverage (%s, first %s series):\n%s",
                source,
                max_series_log,
                coverage.head(max_series_log).to_string(index=False),
            )
        else:
            logger.info("[fetch_generation] Coverage (%s):\n%s", source, coverage.to_string(index=False))

    if output_path.exists() and not config.overwrite:
        logger.info(f"[fetch_generation] exists, loading: {output_path}")
        cached = pd.read_parquet(output_path)
        # Log cached coverage to surface missing series without refetching.
        _log_generation_summary(cached, source="cache")
        return cached

    logger.info(f"[fetch_generation] Fetching {config.fuel_types} for {config.regions}")

    # Use longer timeout (90s) to handle slow EIA API responses
    fetcher = EIARenewableFetcher(timeout=90)
    all_dfs = []

    for fuel_type in config.fuel_types:
        df = fetcher.fetch_all_regions(
            fuel_type=fuel_type,
            start_date=config.start_date,
            end_date=config.end_date,
            regions=config.regions,
            diagnostics=fetch_diagnostics,
        )
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Log fresh coverage to highlight gaps or unexpected negatives.
    _log_generation_summary(combined, source="fresh")

    if fetch_diagnostics:
        empty_series = [
            entry
            for entry in fetch_diagnostics
            if entry.get("empty")
        ]
        for entry in empty_series:
            logger.warning(
                "[fetch_generation] Empty series detail: region=%s fuel=%s total=%s pages=%s",
                entry.get("region"),
                entry.get("fuel_type"),
                entry.get("total_records"),
                entry.get("pages"),
            )

    combined.to_parquet(output_path, index=False)
    logger.info(f"[fetch_generation] Saved: {output_path} ({len(combined)} rows)")

    return combined


def fetch_renewable_weather(
    config: RenewablePipelineConfig,
    include_forecast: bool = True,
) -> pd.DataFrame:
    """Task 2: Fetch weather data for all regions.

    Args:
        config: Pipeline configuration
        include_forecast: Include forecast weather for predictions

    Returns:
        DataFrame with columns [ds, region, weather_vars...]
    """
    output_path = config.weather_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _log_weather_summary(df: pd.DataFrame, source: str) -> None:
        if df.empty:
            logger.warning("[fetch_weather] No weather data rows (%s).", source)
            return

        coverage = (
            df.groupby("region")["ds"]
            .agg(min_ds="min", max_ds="max", rows="count")
            .reset_index()
            .sort_values("region")
        )
        max_region_log = 25
        if len(coverage) > max_region_log:
            logger.info(
                "[fetch_weather] Coverage (%s, first %s regions):\n%s",
                source,
                max_region_log,
                coverage.head(max_region_log).to_string(index=False),
            )
        else:
            logger.info("[fetch_weather] Coverage (%s):\n%s", source, coverage.to_string(index=False))

        missing_cols = [
            col for col in OpenMeteoRenewable.WEATHER_VARS if col not in df.columns
        ]
        if missing_cols:
            logger.warning(
                "[fetch_weather] Missing expected weather columns (%s): %s",
                source,
                missing_cols,
            )

        missing_values = {
            col: int(df[col].isna().sum())
            for col in OpenMeteoRenewable.WEATHER_VARS
            if col in df.columns and df[col].isna().any()
        }
        if missing_values:
            logger.warning(
                "[fetch_weather] Missing weather values (%s): %s",
                source,
                missing_values,
            )

    if output_path.exists() and not config.overwrite:
        logger.info(f"[fetch_weather] exists, loading: {output_path}")
        cached = pd.read_parquet(output_path)
        # Log cached weather coverage to surface missing regions/columns.
        _log_weather_summary(cached, source="cache")
        return cached

    logger.info(f"[fetch_weather] Fetching weather for {config.regions}")

    weather = OpenMeteoRenewable()

    # Historical weather
    hist_df = weather.fetch_all_regions_historical(
        regions=config.regions,
        start_date=config.start_date,
        end_date=config.end_date,
    )

    # Validate historical weather result
    if hist_df.empty:
        raise RuntimeError(
            "[fetch_weather] Historical weather returned empty DataFrame. "
            "fetch_all_regions_historical should raise an error on failure, "
            "but received empty result. Check fetch logic."
        )

    if not {"ds", "region"}.issubset(hist_df.columns):
        missing_cols = {"ds", "region"} - set(hist_df.columns)
        raise ValueError(
            f"[fetch_weather] Weather DataFrame missing required columns: {missing_cols}"
        )

    hist_regions = hist_df['region'].nunique()
    hist_rows = len(hist_df)
    logger.info(
        f"[fetch_weather] Historical: {hist_regions} regions, {hist_rows} rows"
    )

    # Forecast weather (for prediction, prevents leakage)
    if include_forecast:
        fcst_df = weather.fetch_all_regions_forecast(
            regions=config.regions,
            horizon_hours=config.horizon + 24,  # Buffer
        )

        # Validate forecast weather result
        if fcst_df.empty:
            logger.warning(
                "[fetch_weather] Forecast weather returned empty DataFrame. "
                "Using historical data only for model training and predictions."
            )
            combined = hist_df
        else:
            fcst_rows = len(fcst_df)
            logger.info(f"[fetch_weather] Forecast: {fcst_rows} rows")

            # Combine, preferring forecast for overlapping times
            combined = pd.concat([hist_df, fcst_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["ds", "region"], keep="last")
    else:
        combined = hist_df

    combined = combined.sort_values(["region", "ds"]).reset_index(drop=True)

    # Log fresh weather coverage and missing values before saving.
    _log_weather_summary(combined, source="fresh")

    combined.to_parquet(output_path, index=False)
    logger.info(f"[fetch_weather] Saved: {output_path} ({len(combined)} rows)")

    return combined


def train_renewable_models(
    config: RenewablePipelineConfig,
    modeling_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Task 3: Train models and compute baseline metrics via cross-validation.

    Args:
        config: Pipeline configuration
        modeling_df: Preprocessed modeling dataset from build_modeling_dataset()
                     (loads and builds from scratch if None)

    Returns:
        Tuple of (cv_results, leaderboard, baseline_metrics)
    """
    # Load and preprocess data if not provided
    if modeling_df is None:
        generation_df = pd.read_parquet(config.generation_path())
        weather_df = pd.read_parquet(config.weather_path())

        logger.info("[train_models] Building modeling dataset...")
        modeling_df, _ = build_modeling_dataset(
            generation_df,
            weather_df,
            negative_policy='clamp_to_zero',
            output_dir=config.preprocessing_dir()
        )

    logger.info(f"[train_models] Training on {len(modeling_df)} rows")

    model = RenewableForecastModel(
        horizon=config.horizon,
        confidence_levels=config.confidence_levels,
    )

    # Compute adaptive CV settings based on shortest series
    min_series_len = modeling_df.groupby("unique_id").size().min()

    # CV needs: horizon + (n_windows * step_size) rows minimum
    # Solve for n_windows: n_windows = (min_series_len - horizon) / step_size
    available_for_cv = min_series_len - config.horizon

    # Adjust step_size and n_windows to fit data
    step_size = min(config.cv_step_size, max(24, available_for_cv // 3))
    n_windows = min(config.cv_windows, max(2, available_for_cv // step_size))

    logger.info(
        f"[train_models] Adaptive CV: {n_windows} windows, "
        f"step={step_size}h (min_series={min_series_len} rows)"
    )

    # Cross-validation (modeling_df already has weather merged and time features added)
    cv_results, leaderboard = model.cross_validate(
        df=modeling_df,
        n_windows=n_windows,
        step_size=step_size,
    )

    best_model = leaderboard.iloc[0]["model"]
    baseline = compute_baseline_metrics(cv_results, model_name=best_model)

    logger.info(f"[train_models] Best model: {best_model}, RMSE: {baseline['rmse_mean']:.1f}")

    return cv_results, leaderboard, baseline


def train_interpretability_models(
    config: RenewablePipelineConfig,
    generation_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
) -> dict[str, InterpretabilityReport]:
    """Train LightGBM models and generate interpretability reports per series.

    This trains a separate LightGBM model for each series (region × fuel type)
    and generates SHAP, partial dependence, and feature importance artifacts.

    Note: LightGBM is used for interpretability only. The primary forecasts
    come from statistical models (MSTL/ARIMA) which provide better uncertainty
    quantification.

    Args:
        config: Pipeline configuration
        generation_df: Generation data (loads from file if None)
        weather_df: Weather data (loads from file if None)

    Returns:
        Dict mapping series_id -> InterpretabilityReport
    """
    # Load data if not provided
    if generation_df is None:
        generation_df = pd.read_parquet(config.generation_path())
    if weather_df is None:
        weather_df = pd.read_parquet(config.weather_path())

    logger.info(f"[train_interpretability] Training LightGBM for {generation_df['unique_id'].nunique()} series")

    # Ensure datetime types
    generation_df = generation_df.copy()
    generation_df["ds"] = pd.to_datetime(generation_df["ds"], errors="raise")
    weather_df = weather_df.copy()
    weather_df["ds"] = pd.to_datetime(weather_df["ds"], errors="raise")

    reports: dict[str, InterpretabilityReport] = {}
    output_dir = config.interpretability_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    for uid in sorted(generation_df["unique_id"].unique()):
        logger.info(f"[train_interpretability] Processing {uid}...")

        # Extract series data
        series_data = generation_df[generation_df["unique_id"] == uid].copy()
        series_data = series_data.sort_values("ds")

        # Prepare target series with proper frequency
        y = series_data.set_index("ds")["y"]
        y.index = pd.DatetimeIndex(y.index, freq="h")  # Set hourly frequency

        # Prepare exogenous features
        region = uid.split("_")[0]
        series_weather = weather_df[weather_df["region"] == region].copy()

        if series_weather.empty:
            logger.warning(f"[train_interpretability] No weather data for region {region}, skipping {uid}")
            continue

        # Merge weather to series timestamps
        series_data = series_data.merge(
            series_weather[["ds"] + [c for c in WEATHER_VARS if c in series_weather.columns]],
            on="ds",
            how="left",
        )

        # Add time features
        series_data = _add_time_features(series_data)

        # Build exog DataFrame aligned with y
        exog_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        exog_cols += [c for c in WEATHER_VARS if c in series_data.columns]
        exog = series_data.set_index("ds")[exog_cols]

        # Check for missing weather
        missing_weather = exog.isna().any(axis=1).sum()
        if missing_weather > 0:
            logger.warning(f"[train_interpretability] {uid}: {missing_weather} rows with missing weather, filling with ffill/bfill")
            exog = exog.ffill().bfill()

        # Fit LightGBM forecaster
        try:
            if not INTERPRETABILITY_AVAILABLE:
                logger.warning(f"[train_interpretability] {uid}: lightgbm/skforecast not available, skipping")
                continue

            # Create skforecast ForecasterRecursive with LightGBM estimator
            forecaster = ForecasterRecursive(
                estimator=LGBMRegressor(
                    random_state=42,
                    verbose=-1,
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=6,
                ),
                lags=168,  # 7 days of lags
            )
            forecaster.fit(y=y, exog=exog)

            # Create training matrices for SHAP analysis
            X_train, y_train = forecaster.create_train_X_y(y=y, exog=exog)

            # Generate interpretability report
            series_output_dir = output_dir / uid
            report = generate_full_interpretability_report(
                forecaster=forecaster,
                X_train=X_train,
                series_id=uid,
                output_dir=series_output_dir,
                top_n_features=5,
                shap_sample_frac=0.5,
                shap_max_samples=1000,
            )
            reports[uid] = report

            logger.info(
                f"[train_interpretability] {uid}: top_features={report.top_features[:3]}"
            )

        except Exception as e:
            logger.error(f"[train_interpretability] {uid}: Failed to train - {e}")
            continue

    logger.info(f"[train_interpretability] Generated {len(reports)} interpretability reports")
    return reports


def generate_renewable_forecasts(
    config: RenewablePipelineConfig,
    modeling_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
    best_model: str = "MSTL_ARIMA",
) -> pd.DataFrame:
    """Task 4: Generate forecasts with prediction intervals.

    Args:
        config: Pipeline configuration
        modeling_df: Preprocessed modeling dataset (if None, loads and builds)
        weather_df: Raw weather data with forecast (if None, loads from file)
        best_model: Model to use for forecasting

    Returns:
        Forecast DataFrame with physical constraints applied
    """
    output_path = config.forecasts_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data if not provided
    if modeling_df is None:
        generation_df = pd.read_parquet(config.generation_path())
        if weather_df is None:
            weather_df = pd.read_parquet(config.weather_path())

        logger.info("[generate_forecasts] Building modeling dataset...")
        modeling_df, _ = build_modeling_dataset(
            generation_df,
            weather_df,
            negative_policy='clamp_to_zero',
            output_dir=config.preprocessing_dir()
        )

    if weather_df is None:
        weather_df = pd.read_parquet(config.weather_path())

    logger.info(
        f"[generate_forecasts] Generating {config.horizon}h forecasts "
        f"using model={best_model}"
    )

    # Ensure datetime types
    modeling_df = modeling_df.copy()
    modeling_df["ds"] = pd.to_datetime(modeling_df["ds"], errors="raise")
    weather_df = weather_df.copy()
    weather_df["ds"] = pd.to_datetime(weather_df["ds"], errors="raise")

    model = RenewableForecastModel(
        horizon=config.horizon,
        confidence_levels=config.confidence_levels,
    )

    # Fit on preprocessed modeling data
    model.fit(modeling_df)

    # Prepare future exogenous features for forecasting
    # We need weather + time features for the forecast horizon
    per_series_max = modeling_df.groupby("unique_id")["ds"].max()
    logger.info(
        f"[generate_forecasts] Per-series max timestamps:\n"
        f"{per_series_max.to_dict()}"
    )

    min_of_max = per_series_max.min()
    global_max = modeling_df["ds"].max()

    logger.info(
        f"[generate_forecasts] Min of series maxes: {min_of_max}, "
        f"Global max: {global_max}, "
        f"Delta: {(global_max - min_of_max).total_seconds() / 3600:.1f}h"
    )

    # Get future weather beyond the last training timestamp
    future_weather = weather_df[weather_df["ds"] > min_of_max].copy()

    if future_weather.empty:
        raise RuntimeError(
            "[generate_forecasts] No future weather found after last "
            f"training timestamp. min_of_max={min_of_max}"
        )

    # Build future_exog by preparing timestamps and merging weather
    unique_ids = modeling_df["unique_id"].unique()
    future_timestamps = pd.date_range(
        start=min_of_max + pd.Timedelta(hours=1),
        periods=config.horizon,
        freq="h"
    )

    # Create future_exog with all series x timestamps combinations
    future_exog = pd.DataFrame([
        {"unique_id": uid, "ds": ts}
        for uid in unique_ids
        for ts in future_timestamps
    ])

    # Add region for weather merge
    future_exog["region"] = future_exog["unique_id"].str.split("_").str[0]

    # Merge weather
    available_weather_vars = [
        c for c in WEATHER_VARS if c in future_weather.columns
    ]
    future_exog = future_exog.merge(
        future_weather[["ds", "region"] + available_weather_vars],
        on=["ds", "region"],
        how="left"
    )

    # Check for missing weather
    missing_weather = future_exog[available_weather_vars].isna().any(axis=1)
    if missing_weather.any():
        missing_count = missing_weather.sum()
        logger.warning(
            f"[generate_forecasts] {missing_count} future rows missing "
            f"weather, dropping them"
        )
        future_exog = future_exog[~missing_weather].reset_index(drop=True)

    # Add time features (same as dataset_builder)
    future_exog["hour"] = future_exog["ds"].dt.hour
    future_exog["dow"] = future_exog["ds"].dt.dayofweek
    future_exog["hour_sin"] = np.sin(2 * np.pi * future_exog["hour"] / 24)
    future_exog["hour_cos"] = np.cos(2 * np.pi * future_exog["hour"] / 24)
    future_exog["dow_sin"] = np.sin(2 * np.pi * future_exog["dow"] / 7)
    future_exog["dow_cos"] = np.cos(2 * np.pi * future_exog["dow"] / 7)
    future_exog = future_exog.drop(columns=["hour", "dow", "region"])

    # Generate forecasts
    logger.info(
        f"[generate_forecasts] Generating predictions using "
        f"model: {best_model}"
    )
    forecasts = model.predict(future_exog=future_exog, best_model=best_model)

    # CRITICAL: Apply physical constraints (renewable generation cannot be negative)
    # This matches the constraint enforcement during cross-validation (modeling.py:301)
    forecasts = enforce_physical_constraints(forecasts, min_value=0.0)
    logger.info("[generate_forecasts] Applied physical constraints (clipped to [0, ∞))")

    logger.info(
        f"[generate_forecasts] Generated {len(forecasts)} forecast rows "
        f"for {forecasts['unique_id'].nunique()} series"
    )

    forecasts.to_parquet(output_path, index=False)
    logger.info(
        f"[generate_forecasts] Saved: {output_path} ({len(forecasts)} rows)"
    )

    return forecasts


def compute_renewable_drift(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    baseline_metrics: dict,
) -> dict:
    """Task 5: Detect drift by comparing current metrics to baseline.

    Drift is flagged when current RMSE > baseline_mean + 2*baseline_std

    Args:
        predictions: Forecast DataFrame with [unique_id, ds, yhat]
        actuals: Actual values DataFrame with [unique_id, ds, y]
        baseline_metrics: Baseline from cross-validation

    Returns:
        Dictionary with drift status and details
    """
    from src.chapter2.evaluation import ForecastMetrics

    # Merge predictions with actuals
    merged = predictions.merge(
        actuals[["unique_id", "ds", "y"]],
        on=["unique_id", "ds"],
        how="inner",
    )

    if len(merged) == 0:
        return {
            "status": "no_data",
            "message": "No overlapping data between predictions and actuals",
        }

    # Compute current metrics
    y_true = merged["y"].values
    y_pred = merged["yhat"].values

    current_rmse = ForecastMetrics.rmse(y_true, y_pred)
    current_mae = ForecastMetrics.mae(y_true, y_pred)

    # Check against threshold
    threshold = baseline_metrics.get("drift_threshold_rmse", float("inf"))
    is_drifting = current_rmse > threshold

    result = {
        "status": "drift_detected" if is_drifting else "stable",
        "current_rmse": float(current_rmse),
        "current_mae": float(current_mae),
        "baseline_rmse": float(baseline_metrics.get("rmse_mean", 0)),
        "drift_threshold": float(threshold),
        "threshold_exceeded_by": float(max(0, current_rmse - threshold)),
        "n_predictions": len(merged),
        "timestamp": datetime.utcnow().isoformat(),
    }

    if is_drifting:
        logger.warning(
            f"[drift] DRIFT DETECTED: RMSE={current_rmse:.1f} > threshold={threshold:.1f}"
        )
    else:
        logger.info(f"[drift] Stable: RMSE={current_rmse:.1f} <= threshold={threshold:.1f}")

    return result


def run_full_pipeline(
    config: RenewablePipelineConfig,
    fetch_diagnostics: Optional[list[dict]] = None,
    skip_eda: bool = False,
    *,
    max_lag_hours: int = 48,
    max_missing_ratio: float = 0.02,
) -> dict:
    """Run the complete renewable forecasting pipeline.

    Steps:
    1. Fetch generation data
    2. Fetch weather data
    3. Run EDA and get recommendations (NEW)
    4. Build datasets per fuel type using recommendations
    5. Train models (CV)
    6. Generate forecasts

    Args:
        config: Pipeline configuration
        fetch_diagnostics: Optional list to capture per-region fetch metadata

    Returns:
        Dictionary with pipeline results
    """
    logger.info(f"[pipeline] Starting: {config.start_date} to {config.end_date}")
    logger.info(f"[pipeline] Regions: {config.regions}")
    logger.info(f"[pipeline] Fuel types: {config.fuel_types}")

    results = {}

    # Step 1: Fetch generation
    generation_df = fetch_renewable_data(config, fetch_diagnostics=fetch_diagnostics)
    results["generation_rows"] = len(generation_df)
    results["series_count"] = generation_df["unique_id"].nunique()

    from src.renewable.validation import validate_generation_df

    expected_series = [f"{r}_{f}" for r in config.regions for f in config.fuel_types]
    rep = validate_generation_df(
        generation_df,
        expected_series=expected_series,
        max_missing_ratio=max_missing_ratio,
        max_lag_hours=max_lag_hours,  # choose a value consistent with EIA publishing lag
    )
    if not rep.ok:
        raise RuntimeError(f"[pipeline][generation_validation] {rep.message} details={rep.details}")

    # Step 2: Fetch weather
    weather_df = fetch_renewable_weather(config)
    results["weather_rows"] = len(weather_df)

    # Step 3: Run EDA (NEW)
    eda_recommendations = None

    if not skip_eda:
        logger.info("[pipeline] Running EDA to generate preprocessing recommendations")
        from src.renewable.eda import run_full_eda

        eda_output_dir = Path(config.data_dir) / "eda"
        eda_recommendations = run_full_eda(
            generation_df,
            weather_df,
            eda_output_dir,
        )

        results["eda"] = {
            "output_dir": str(eda_output_dir),
            "negative_policy": eda_recommendations.preprocessing.negative_policy,
            "confidence": eda_recommendations.preprocessing.negative_confidence,
        }
        logger.info(f"[pipeline] EDA complete. Recommended policy: {eda_recommendations.preprocessing.negative_policy}")
    else:
        logger.warning("[pipeline] Skipping EDA - using default preprocessing policies")

    # Step 4: Build datasets per fuel type (MODIFIED)
    logger.info("[pipeline] Building modeling datasets (fuel-type specific)")

    from src.renewable.dataset_builder import build_dataset_by_fuel_type

    fuel_datasets = {}
    last_prep_report = None

    for fuel_type in config.fuel_types:
        logger.info(f"[pipeline] Building {fuel_type} dataset...")

        fuel_output_dir = config.preprocessing_dir() / fuel_type.lower()

        modeling_df_fuel, prep_report = build_dataset_by_fuel_type(
            generation_df,
            weather_df,
            fuel_type=fuel_type,
            output_dir=fuel_output_dir,
            eda_recommendations=eda_recommendations.preprocessing if eda_recommendations else None,
        )

        fuel_datasets[fuel_type] = modeling_df_fuel
        last_prep_report = prep_report  # Keep last report for backward compatibility

        logger.info(f"[pipeline] {fuel_type}: {prep_report.input_rows:,} → {prep_report.output_rows:,} rows")

    # Combine all fuel datasets
    modeling_df = pd.concat(fuel_datasets.values(), ignore_index=True)

    # DEBUG: Log dataset combination
    logger.info(f"[pipeline] Combined {len(fuel_datasets)} fuel-type datasets into {len(modeling_df):,} rows")

    # Save combined modeling dataset for analysis and testing
    modeling_dataset_path = Path(config.data_dir) / "modeling_dataset.parquet"
    modeling_df.to_parquet(modeling_dataset_path, index=False)
    logger.info(f"[pipeline] Saved modeling dataset: {modeling_dataset_path} ({len(modeling_df):,} rows)")

    # Initialize preprocessing results
    results["preprocessing"] = {
        "rows_input": len(generation_df),
        "rows_output": len(modeling_df),
    }

    # Use last report for backward compatibility
    prep_report = last_prep_report

    # Extract time and weather features from output_features (from last fuel type)
    if prep_report:
        logger.debug("[pipeline] Extracting features from last fuel type's preprocessing report")

        time_features = [
            f for f in prep_report.output_features
            if f in ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        ]
        weather_features = [
            f for f in prep_report.output_features
            if f in prep_report.weather_vars_used
        ]

        logger.debug(f"[pipeline] Found {len(time_features)} time features, {len(weather_features)} weather features")

        results["preprocessing"].update({
            "series_dropped": len(prep_report.series_dropped_incomplete),
            "negative_action": prep_report.negative_report.action_taken,
            "time_features": time_features,
            "weather_features": weather_features,
        })
        logger.info(
            f"[pipeline] Preprocessing: {len(generation_df):,} → "
            f"{len(modeling_df):,} rows"
        )
    else:
        logger.warning("[pipeline] No preprocessing report available - skipping feature extraction")

    # Step 5: Train and validate (on preprocessed data)
    cv_results, leaderboard, baseline = train_renewable_models(
        config, modeling_df
    )
    best_model = leaderboard.iloc[0]["model"]
    results["best_model"] = best_model
    results["best_rmse"] = float(leaderboard.iloc[0]["rmse"])
    results["baseline"] = baseline
    # Save full leaderboard for dashboard display
    results["leaderboard"] = leaderboard.to_dict(orient="records")

    # Step 6: Generate forecasts (use the best model from CV)
    # Pass weather_df for future weather (forecast horizon)
    forecasts = generate_renewable_forecasts(
        config, modeling_df, weather_df, best_model=best_model
    )
    results["forecast_rows"] = len(forecasts)

    # Step 5: Train LightGBM models and generate interpretability reports (optional)
    # (LightGBM is for interpretability only - MSTL/ARIMA provide primary forecasts)
    if config.enable_interpretability:
        logger.info("[pipeline] Training interpretability models (LightGBM + SHAP)")
        try:
            interpretability_reports = train_interpretability_models(
                config, generation_df, weather_df
            )
            results["interpretability"] = {
                "series_count": len(interpretability_reports),
                "series": list(interpretability_reports.keys()),
                "output_dir": str(config.interpretability_dir()),
            }

            # Add top features summary per series
            for uid, report in interpretability_reports.items():
                results["interpretability"][f"{uid}_top_features"] = report.top_features[:3]

        except Exception as e:
            logger.warning(f"[pipeline] Interpretability training failed (non-fatal): {e}")
            results["interpretability"] = {"error": str(e)}
    else:
        logger.info("[pipeline] Interpretability disabled (enable_interpretability=False)")
        results["interpretability"] = {"enabled": False}

    if fetch_diagnostics is not None:
        results["fetch_diagnostics"] = fetch_diagnostics

    logger.info(f"[pipeline] Complete. Best model: {results['best_model']}")

    return results


def main():
    """CLI entry point for renewable pipeline."""
    parser = argparse.ArgumentParser(
        description="Renewable Energy Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Preset Examples:
  # Fast development (24h forecast, 2 CV windows, 15 days lookback)
  python -m src.renewable.tasks --preset 24h

  # Standard forecasting (48h forecast, 3 CV windows, 21 days lookback)
  python -m src.renewable.tasks --preset 48h

  # Extended planning (72h forecast, 3 CV windows, 28 days lookback)
  python -m src.renewable.tasks --preset 72h

Custom Examples:
  # 24h preset but only CALI region, skip interpretability
  python -m src.renewable.tasks --preset 24h --regions CALI --no-interpretability

  # Custom: 36h forecast with 4 CV windows
  python -m src.renewable.tasks --horizon 36 --cv-windows 4 --lookback-days 30
        """
    )

    # Preset system (NEW)
    parser.add_argument(
        "--preset",
        type=str,
        choices=["24h", "48h", "72h"],
        help="Quick preset: 24h (fast dev), 48h (standard), 72h (extended planning)",
    )

    # Flags (NEW)
    parser.add_argument(
        "--no-interpretability",
        action="store_true",
        help="Disable LightGBM interpretability analysis (speeds up pipeline)",
    )

    # Data parameters (existing)
    parser.add_argument(
        "--regions",
        type=str,
        help="Override regions (comma-separated, e.g., CALI,ERCO,MISO)",
    )
    parser.add_argument(
        "--fuel",
        type=str,
        help="Override fuel types (comma-separated, e.g., WND,SUN)",
    )

    # Forecast parameters (existing + new)
    parser.add_argument(
        "--horizon",
        type=int,
        help="Override forecast horizon in hours",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        help="Override lookback days",
    )
    parser.add_argument(
        "--cv-windows",
        type=int,
        help="Override CV windows count",
    )

    # Output parameters
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data files",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/renewable",
        help="Output directory (default: data/renewable)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Build config with preset support
    if args.preset:
        # Apply preset defaults
        logger.info(f"[CLI] Applying preset: {args.preset}")
        config = RenewablePipelineConfig(
            horizon_preset=args.preset,  # This triggers __post_init__ to apply preset
            regions=args.regions.split(",") if args.regions else ["CALI", "ERCO", "MISO"],
            fuel_types=args.fuel.split(",") if args.fuel else ["WND", "SUN"],
            enable_interpretability=not args.no_interpretability,
            overwrite=args.overwrite,
            data_dir=args.data_dir,
        )

        # Allow CLI overrides of preset values
        if args.horizon is not None:
            object.__setattr__(config, "horizon", args.horizon)
            logger.info(f"[CLI] Override: horizon={args.horizon}h")
        if args.lookback_days is not None:
            object.__setattr__(config, "lookback_days", args.lookback_days)
            logger.info(f"[CLI] Override: lookback_days={args.lookback_days}")
        if args.cv_windows is not None:
            object.__setattr__(config, "cv_windows", args.cv_windows)
            logger.info(f"[CLI] Override: cv_windows={args.cv_windows}")

    else:
        # No preset: use explicit values or defaults
        config = RenewablePipelineConfig(
            regions=args.regions.split(",") if args.regions else ["CALI", "ERCO", "MISO"],
            fuel_types=args.fuel.split(",") if args.fuel else ["WND", "SUN"],
            lookback_days=args.lookback_days if args.lookback_days else 30,
            horizon=args.horizon if args.horizon else 24,
            cv_windows=args.cv_windows if args.cv_windows else 5,
            enable_interpretability=not args.no_interpretability,
            overwrite=args.overwrite,
            data_dir=args.data_dir,
        )

    # Run pipeline
    results = run_full_pipeline(config)

    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    print(f"  Series count: {results['series_count']}")
    print(f"  Generation rows: {results['generation_rows']}")
    print(f"  Weather rows: {results['weather_rows']}")
    print(f"  Forecast rows: {results['forecast_rows']}")
    print(f"  Best model: {results['best_model']}")
    print(f"  Best RMSE: {results['best_rmse']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
