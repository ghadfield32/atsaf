from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import AQIPipelineConfig

logger = logging.getLogger(__name__)

_TZ_MAP = {
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
    "MST": "America/Denver",
    "MDT": "America/Denver",
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
    "EST": "America/New_York",
    "EDT": "America/New_York",
    "HST": "Pacific/Honolulu",
    "AKST": "America/Anchorage",
    "AKDT": "America/Anchorage",
}


def load_airnow_api_key() -> str:
    api_key = os.getenv("AIRNOW_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "AIRNOW_API_KEY is missing. Add it to your environment or .env file."
        )
    return api_key


def _create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _resolve_timezone(tz_name: Optional[str], fallback: str) -> str:
    if not tz_name:
        return fallback
    if tz_name in _TZ_MAP:
        return _TZ_MAP[tz_name]
    return tz_name


def pull_airnow_historical(config: AQIPipelineConfig, api_key: str) -> pd.DataFrame:
    """
    Pull historical AQI observations from AirNow for a lat/long window.

    AirNow expects a per-date request, so we loop day-by-day.
    """
    session = _create_session()

    start = datetime.strptime(config.start_date, "%Y-%m-%d")
    end = datetime.strptime(config.end_date, "%Y-%m-%d")

    records: list[dict] = []
    date = start
    while date <= end:
        params = {
            "format": "application/json",
            "latitude": config.latitude,
            "longitude": config.longitude,
            "distance": config.distance_miles,
            "date": date.strftime(config.airnow_date_format),
            "API_KEY": api_key,
        }

        resp = session.get(config.airnow_base_url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        if isinstance(payload, list) and payload:
            records.extend(payload)
            logger.info(
                "[airnow] %s: %s records", date.date().isoformat(), len(payload)
            )
        else:
            logger.info("[airnow] %s: 0 records", date.date().isoformat())

        date += timedelta(days=1)

    df = pd.DataFrame.from_records(records)
    logger.info("[airnow] total records: %s", len(df))
    return df


def normalize_airnow_observations(
    df_raw: pd.DataFrame,
    config: AQIPipelineConfig,
) -> pd.DataFrame:
    """
    Normalize AirNow response to UTC timestamps and consistent columns.
    """
    if df_raw.empty:
        return df_raw.copy()

    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in ("DateObserved", "HourObserved", "LocalTimeZone", "AQI", "ParameterName"):
        if col not in df.columns:
            df[col] = None

    date_str = df["DateObserved"].astype(str)
    hour_str = df["HourObserved"].astype(str).str.zfill(2)
    df["local_dt"] = pd.to_datetime(
        date_str + " " + hour_str + ":00",
        errors="coerce",
    )

    tz_resolved = df["LocalTimeZone"].apply(
        lambda tz: _resolve_timezone(tz, config.local_timezone)
    )

    def _to_utc(row: pd.Series) -> Optional[pd.Timestamp]:
        ts = row["local_dt"]
        if pd.isna(ts):
            return pd.NaT
        tz_name = row["tz_resolved"]
        try:
            return ts.tz_localize(ZoneInfo(tz_name)).tz_convert("UTC")
        except Exception:
            return ts.tz_localize(ZoneInfo("UTC"))

    df["tz_resolved"] = tz_resolved
    df["ds_utc"] = df.apply(_to_utc, axis=1)
    df["aqi"] = pd.to_numeric(df["AQI"], errors="coerce")
    df["parameter_name"] = df["ParameterName"]

    return df


def prepare_airnow_for_forecasting(
    df_raw: pd.DataFrame,
    config: AQIPipelineConfig,
) -> pd.DataFrame:
    """
    Convert AirNow observations to StatsForecast-ready format.

    Returns columns: unique_id, ds (timezone-naive UTC), y
    """
    df = normalize_airnow_observations(df_raw, config)

    if df.empty:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])

    if config.parameter_name:
        df = df[df["parameter_name"] == config.parameter_name]

    df = df.dropna(subset=["ds_utc", "aqi"])

    agg_method = config.aqi_aggregation.lower()
    if agg_method not in ("max", "mean"):
        raise ValueError(f"aqi_aggregation must be max/mean, got {config.aqi_aggregation}")

    grouped = (
        df.groupby("ds_utc", as_index=False)["aqi"]
        .agg(agg_method)
        .rename(columns={"aqi": "y"})
    )

    grouped["unique_id"] = config.series_id()
    grouped["ds"] = pd.to_datetime(grouped["ds_utc"], utc=True).dt.tz_localize(None)
    grouped = grouped.drop(columns=["ds_utc"])

    grouped = grouped.sort_values("ds").reset_index(drop=True)
    return grouped[["unique_id", "ds", "y"]]
