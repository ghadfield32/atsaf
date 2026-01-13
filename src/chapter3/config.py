# file: src/chapter3/config.py
"""
Chapter 3: Pipeline Configuration
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    # Data parameters
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    respondent: str = "US48"
    fueltype: str = "NG"

    # IO
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    overwrite: bool = False

    # Forecasting / backtest
    horizon: int = 24
    n_windows: int = 5
    step_size: int = 168
    confidence_level: int = 95

    # Scheduling (Airflow)
    schedule: str = "0 6 * * *"
    retries: int = 3
    retry_delay_minutes: int = 5

    # MLflow
    experiment_name: str = "eia_forecasting"
    model_alias: str = "champion"

    def run_id(self) -> str:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def data_path(self) -> Path:
        return Path(self.data_dir)

    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    def raw_path(self) -> Path:
        return self.data_path() / "raw.parquet"

    def clean_path(self) -> Path:
        return self.data_path() / "clean.parquet"

    def metadata_path(self) -> Path:
        return self.data_path() / "metadata.json"

    def leaderboard_path(self) -> Path:
        return self.artifacts_path() / "leaderboard.parquet"

    def cv_results_path(self) -> Path:
        return self.artifacts_path() / "cv_results.parquet"

    def predictions_path(self) -> Path:
        return self.artifacts_path() / "predictions.parquet"
