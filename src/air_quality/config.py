from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class AQIPipelineConfig:
    # Location + data range
    start_date: str = "2024-01-01"
    end_date: str = "2024-03-31"
    latitude: float = 34.0522
    longitude: float = -118.2437
    distance_miles: int = 25
    location_name: str = "los_angeles_ca"
    local_timezone: str = "America/Los_Angeles"

    # AirNow ingestion
    airnow_base_url: str = "https://www.airnowapi.org/aq/observation/latLong/historical/"
    airnow_date_format: str = "%Y-%m-%d"
    parameter_name: Optional[str] = None  # e.g., "PM2.5" to filter a single pollutant
    aqi_aggregation: str = "max"       # max/mean for multiple monitors or pollutants

    # Open-Meteo ingestion
    open_meteo_base_url: str = "https://archive-api.open-meteo.com/v1/archive"
    weather_variables: Tuple[str, ...] = (
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
        "pressure_msl",
    )

    # IO
    data_dir: str = "data/aqi"
    artifacts_dir: str = "artifacts/aqi"
    overwrite: bool = False

    # Forecasting / backtest
    horizon: int = 24
    n_windows: int = 5
    step_size: int = 168
    confidence_level: int = 95

    # Residual corrector
    use_residual: bool = True
    residual_lags: Tuple[int, ...] = (1, 2, 3, 24)
    residual_windows: Tuple[int, ...] = (24, 72)
    residual_model: str = "ridge"

    # Scheduling / metadata
    schedule: str = "0 6 * * *"
    experiment_name: str = "aqi_forecasting"
    model_alias: str = "champion"

    def run_id(self) -> str:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def series_id(self) -> str:
        return f"aqi_{self.location_name.lower()}"

    def data_path(self) -> Path:
        return Path(self.data_dir)

    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    def raw_airnow_path(self) -> Path:
        return self.data_path() / "raw_airnow.parquet"

    def raw_weather_path(self) -> Path:
        return self.data_path() / "raw_weather.parquet"

    def clean_aqi_path(self) -> Path:
        return self.data_path() / "aqi.parquet"

    def clean_weather_path(self) -> Path:
        return self.data_path() / "weather.parquet"

    def merged_path(self) -> Path:
        return self.data_path() / "aqi_weather.parquet"

    def metadata_path(self) -> Path:
        return self.data_path() / "metadata.json"

    def leaderboard_path(self) -> Path:
        return self.artifacts_path() / "leaderboard.parquet"

    def cv_results_path(self) -> Path:
        return self.artifacts_path() / "cv_results.parquet"

    def predictions_path(self) -> Path:
        return self.artifacts_path() / "predictions.parquet"

    def residual_model_path(self) -> Path:
        return self.artifacts_path() / "residual_model.joblib"
