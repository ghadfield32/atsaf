"""
Chapter 4: Pipeline Health Checks

Monitors pipeline health:
- Data freshness (hours since last ingest)
- Data completeness (missing hours)
- Forecast freshness (hours since last prediction)
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class HealthReport:
    """Results of pipeline health check"""
    freshness_hours: float          # Hours since last ingest
    missing_hour_count: int         # Missing hours in recent data
    forecast_age_hours: float       # Hours since last forecast
    last_ingest_time: Optional[str] # Timestamp of last ingest
    last_forecast_time: Optional[str]  # Timestamp of last forecast
    data_rows: int                  # Total rows in clean data
    is_healthy: bool                # Overall health status


def check_freshness(metadata_path: str = "data/metadata.json") -> float:
    """
    Calculate hours since last successful ingest.

    Args:
        metadata_path: Path to metadata.json file

    Returns:
        Hours since last ingest (float), or inf if no metadata
    """
    path = Path(metadata_path)
    if not path.exists():
        return float("inf")

    with open(path) as f:
        metadata = json.load(f)

    pull_timestamp = metadata.get("pull_timestamp")
    if not pull_timestamp:
        return float("inf")

    # Parse ISO timestamp
    last_pull = datetime.fromisoformat(pull_timestamp.replace("Z", "+00:00"))
    if last_pull.tzinfo is None:
        last_pull = last_pull.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    delta = now - last_pull
    return delta.total_seconds() / 3600


def check_completeness(
    clean_path: str = "data/clean.parquet",
    expected_hours: int = 24,
) -> int:
    """
    Count missing hours in most recent data window.

    Args:
        clean_path: Path to clean.parquet
        expected_hours: Number of hours to check (default: last 24)

    Returns:
        Count of missing hours
    """
    path = Path(clean_path)
    if not path.exists():
        return expected_hours  # All missing if file doesn't exist

    df = pd.read_parquet(clean_path)

    # Get most recent data
    if "period" in df.columns:
        df["period"] = pd.to_datetime(df["period"], utc=True)
        max_time = df["period"].max()
        min_expected = max_time - pd.Timedelta(hours=expected_hours)

        # Generate expected range
        expected_range = pd.date_range(
            start=min_expected,
            end=max_time,
            freq="h",
        )

        actual_hours = set(df["period"])
        missing = expected_range.difference(actual_hours)
        return len(missing)

    return 0


def check_forecast_freshness(
    predictions_path: str = "artifacts/predictions.parquet",
) -> float:
    """
    Calculate hours since last forecast was generated.

    Args:
        predictions_path: Path to predictions.parquet

    Returns:
        Hours since file modification (float), or inf if no file
    """
    path = Path(predictions_path)
    if not path.exists():
        return float("inf")

    # Use file modification time
    mtime = path.stat().st_mtime
    file_time = datetime.fromtimestamp(mtime, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = now - file_time
    return delta.total_seconds() / 3600


def get_last_ingest_time(metadata_path: str = "data/metadata.json") -> Optional[str]:
    """Get the timestamp of the last successful ingest"""
    path = Path(metadata_path)
    if not path.exists():
        return None

    with open(path) as f:
        metadata = json.load(f)
    return metadata.get("pull_timestamp")


def get_data_row_count(clean_path: str = "data/clean.parquet") -> int:
    """Get total row count in clean data"""
    path = Path(clean_path)
    if not path.exists():
        return 0
    df = pd.read_parquet(clean_path)
    return len(df)


def full_health_check(
    data_dir: str = "data",
    artifacts_dir: str = "artifacts",
    freshness_threshold_hours: float = 6.0,
    missing_hour_threshold: int = 0,
    forecast_age_threshold_hours: float = 24.0,
) -> HealthReport:
    """
    Run all health checks and return summary.

    Args:
        data_dir: Directory containing raw/clean data
        artifacts_dir: Directory containing predictions
        freshness_threshold_hours: Max acceptable hours since ingest
        missing_hour_threshold: Max acceptable missing hours
        forecast_age_threshold_hours: Max acceptable forecast age

    Returns:
        HealthReport with all check results
    """
    metadata_path = f"{data_dir}/metadata.json"
    clean_path = f"{data_dir}/clean.parquet"
    predictions_path = f"{artifacts_dir}/predictions.parquet"

    # Run checks
    freshness_hours = check_freshness(metadata_path)
    missing_hour_count = check_completeness(clean_path)
    forecast_age_hours = check_forecast_freshness(predictions_path)
    last_ingest_time = get_last_ingest_time(metadata_path)
    data_rows = get_data_row_count(clean_path)

    # Get last forecast time from file
    pred_path = Path(predictions_path)
    last_forecast_time = None
    if pred_path.exists():
        mtime = pred_path.stat().st_mtime
        last_forecast_time = datetime.fromtimestamp(
            mtime, tz=timezone.utc
        ).isoformat()

    # Determine overall health
    is_healthy = (
        freshness_hours <= freshness_threshold_hours
        and missing_hour_count <= missing_hour_threshold
        and forecast_age_hours <= forecast_age_threshold_hours
    )

    return HealthReport(
        freshness_hours=freshness_hours,
        missing_hour_count=missing_hour_count,
        forecast_age_hours=forecast_age_hours,
        last_ingest_time=last_ingest_time,
        last_forecast_time=last_forecast_time,
        data_rows=data_rows,
        is_healthy=is_healthy,
    )


def print_health_report(report: HealthReport) -> None:
    """Print a human-readable health report"""
    status = "HEALTHY" if report.is_healthy else "UNHEALTHY"
    print(f"\n=== Pipeline Health: {status} ===")
    print(f"Data freshness: {report.freshness_hours:.1f} hours")
    print(f"Missing hours: {report.missing_hour_count}")
    print(f"Forecast age: {report.forecast_age_hours:.1f} hours")
    print(f"Total rows: {report.data_rows}")
    if report.last_ingest_time:
        print(f"Last ingest: {report.last_ingest_time}")
    if report.last_forecast_time:
        print(f"Last forecast: {report.last_forecast_time}")
