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
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.renewable.eia_renewable import EIARenewableFetcher
from src.renewable.modeling import (
    RenewableForecastModel,
    _log_series_summary,
    compute_baseline_metrics,
)
from src.renewable.open_meteo import OpenMeteoRenewable
from src.renewable.regions import REGIONS, list_regions

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

    # CV parameters
    cv_windows: int = 5
    cv_step_size: int = 168  # 1 week

    # Output paths
    data_dir: str = "data/renewable"
    overwrite: bool = False

    def __post_init__(self):
        # Set default dates if not provided
        if not self.end_date:
            self.end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not self.start_date:
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
            start = end - timedelta(days=self.lookback_days)
            self.start_date = start.strftime("%Y-%m-%d")

    def generation_path(self) -> Path:
        return Path(self.data_dir) / "generation.parquet"

    def weather_path(self) -> Path:
        return Path(self.data_dir) / "weather.parquet"

    def forecasts_path(self) -> Path:
        return Path(self.data_dir) / "forecasts.parquet"

    def baseline_path(self) -> Path:
        return Path(self.data_dir) / "baseline.json"


def fetch_renewable_data(
    config: RenewablePipelineConfig,
) -> pd.DataFrame:
    """Task 1: Fetch EIA generation data for all regions and fuel types.

    Args:
        config: Pipeline configuration

    Returns:
        DataFrame with columns [unique_id, ds, y]
    """
    output_path = config.generation_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not config.overwrite:
        logger.info(f"[fetch_generation] exists, loading: {output_path}")
        return pd.read_parquet(output_path)

    logger.info(f"[fetch_generation] Fetching {config.fuel_types} for {config.regions}")

    fetcher = EIARenewableFetcher()
    all_dfs = []

    for fuel_type in config.fuel_types:
        df = fetcher.fetch_all_regions(
            fuel_type=fuel_type,
            start_date=config.start_date,
            end_date=config.end_date,
            regions=config.regions,
        )
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    _log_series_summary(combined, value_col="y", label="generation_data")

    expected_series = {
        f"{region}_{fuel}" for region in config.regions for fuel in config.fuel_types
    }
    present_series = set(combined["unique_id"])
    missing_series = sorted(expected_series - present_series)
    if missing_series:
        logger.warning(
            f"[fetch_generation] Missing expected series after fetch: {missing_series}"
        )

    coverage = (
        combined.groupby("unique_id")["ds"]
        .agg(min_ds="min", max_ds="max", rows="count")
        .reset_index()
        .sort_values("unique_id")
    )
    max_series_log = 25
    if len(coverage) > max_series_log:
        logger.info(
            "[fetch_generation] Coverage (first %s series):\n%s",
            max_series_log,
            coverage.head(max_series_log).to_string(index=False),
        )
    else:
        logger.info("[fetch_generation] Coverage:\n%s", coverage.to_string(index=False))

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

    if output_path.exists() and not config.overwrite:
        logger.info(f"[fetch_weather] exists, loading: {output_path}")
        return pd.read_parquet(output_path)

    logger.info(f"[fetch_weather] Fetching weather for {config.regions}")

    weather = OpenMeteoRenewable()

    # Historical weather
    hist_df = weather.fetch_all_regions_historical(
        regions=config.regions,
        start_date=config.start_date,
        end_date=config.end_date,
    )

    # Forecast weather (for prediction, prevents leakage)
    if include_forecast:
        fcst_df = weather.fetch_all_regions_forecast(
            regions=config.regions,
            horizon_hours=config.horizon + 24,  # Buffer
        )

        # Combine, preferring forecast for overlapping times
        combined = pd.concat([hist_df, fcst_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ds", "region"], keep="last")
    else:
        combined = hist_df

    combined = combined.sort_values(["region", "ds"]).reset_index(drop=True)

    combined.to_parquet(output_path, index=False)
    logger.info(f"[fetch_weather] Saved: {output_path} ({len(combined)} rows)")

    return combined


def train_renewable_models(
    config: RenewablePipelineConfig,
    generation_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Task 3: Train models and compute baseline metrics via cross-validation.

    Args:
        config: Pipeline configuration
        generation_df: Generation data (loads from file if None)
        weather_df: Weather data (loads from file if None)

    Returns:
        Tuple of (cv_results, leaderboard, baseline_metrics)
    """
    # Load data if not provided
    if generation_df is None:
        generation_df = pd.read_parquet(config.generation_path())
    if weather_df is None:
        weather_df = pd.read_parquet(config.weather_path())

    logger.info(f"[train_models] Training on {len(generation_df)} rows")

    model = RenewableForecastModel(
        horizon=config.horizon,
        confidence_levels=config.confidence_levels,
    )

    # Compute adaptive CV settings based on shortest series
    min_series_len = generation_df.groupby("unique_id").size().min()

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

    # Cross-validation
    cv_results, leaderboard = model.cross_validate(
        df=generation_df,
        weather_df=weather_df,
        n_windows=n_windows,
        step_size=step_size,
    )

    # Compute baseline for drift detection
    best_model = leaderboard.iloc[0]["model"]
    baseline = compute_baseline_metrics(cv_results, model_name=best_model)

    logger.info(f"[train_models] Best model: {best_model}, RMSE: {baseline['rmse_mean']:.1f}")

    return cv_results, leaderboard, baseline


def generate_renewable_forecasts(
    config: RenewablePipelineConfig,
    generation_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Task 4: Generate forecasts with prediction intervals.

    Args:
        config: Pipeline configuration
        generation_df: Generation data (loads from file if None)
        weather_df: Weather data (loads from file if None)

    Returns:
        DataFrame with forecasts [unique_id, ds, yhat, intervals...]
    """
    output_path = config.forecasts_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data if not provided
    if generation_df is None:
        generation_df = pd.read_parquet(config.generation_path())
    if weather_df is None:
        weather_df = pd.read_parquet(config.weather_path())

    logger.info(f"[generate_forecasts] Generating {config.horizon}h forecasts")

    model = RenewableForecastModel(
        horizon=config.horizon,
        confidence_levels=config.confidence_levels,
    )

    # Fit on all data
    model.fit(generation_df, weather_df)

    # Predict
    forecasts = model.predict()

    forecasts.to_parquet(output_path, index=False)
    logger.info(f"[generate_forecasts] Saved: {output_path} ({len(forecasts)} rows)")

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


def run_full_pipeline(config: RenewablePipelineConfig) -> dict:
    """Run the complete renewable forecasting pipeline.

    Steps:
    1. Fetch generation data
    2. Fetch weather data
    3. Train models (CV)
    4. Generate forecasts

    Args:
        config: Pipeline configuration

    Returns:
        Dictionary with pipeline results
    """
    logger.info(f"[pipeline] Starting: {config.start_date} to {config.end_date}")
    logger.info(f"[pipeline] Regions: {config.regions}")
    logger.info(f"[pipeline] Fuel types: {config.fuel_types}")

    results = {}

    # Step 1: Fetch generation
    generation_df = fetch_renewable_data(config)
    results["generation_rows"] = len(generation_df)
    results["series_count"] = generation_df["unique_id"].nunique()

    # Step 2: Fetch weather
    weather_df = fetch_renewable_weather(config)
    results["weather_rows"] = len(weather_df)

    # Step 3: Train and validate
    cv_results, leaderboard, baseline = train_renewable_models(
        config, generation_df, weather_df
    )
    results["best_model"] = leaderboard.iloc[0]["model"]
    results["best_rmse"] = float(leaderboard.iloc[0]["rmse"])
    results["baseline"] = baseline

    # Step 4: Generate forecasts
    forecasts = generate_renewable_forecasts(config, generation_df, weather_df)
    results["forecast_rows"] = len(forecasts)

    logger.info(f"[pipeline] Complete. Best model: {results['best_model']}")

    return results


def main():
    """CLI entry point for renewable pipeline."""
    parser = argparse.ArgumentParser(description="Renewable Energy Forecasting Pipeline")

    parser.add_argument(
        "--regions",
        type=str,
        default="CALI,ERCO,MISO",
        help="Comma-separated region codes (default: CALI,ERCO,MISO)",
    )
    parser.add_argument(
        "--fuel",
        type=str,
        default="WND,SUN",
        help="Comma-separated fuel types (default: WND,SUN)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Lookback days (default: 30)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Forecast horizon in hours (default: 24)",
    )
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

    # Build config
    config = RenewablePipelineConfig(
        regions=args.regions.split(","),
        fuel_types=args.fuel.split(","),
        lookback_days=args.days,
        horizon=args.horizon,
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
