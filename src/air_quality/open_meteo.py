from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import AQIPipelineConfig

logger = logging.getLogger(__name__)


def _create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def pull_open_meteo(config: AQIPipelineConfig) -> pd.DataFrame:
    """
    Pull historical hourly weather data from Open-Meteo archive API.
    """
    session = _create_session()
    end_dt = datetime.strptime(config.end_date, "%Y-%m-%d")
    end_dt = end_dt + timedelta(hours=config.horizon)
    end_date = end_dt.date().isoformat()

    params = {
        "latitude": config.latitude,
        "longitude": config.longitude,
        "start_date": config.start_date,
        "end_date": end_date,
        "hourly": ",".join(config.weather_variables),
        "timezone": "UTC",
    }

    resp = session.get(config.open_meteo_base_url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame(columns=["ds"])

    data = {
        "ds": pd.to_datetime(times, errors="coerce", utc=True).tz_localize(None)
    }

    for var in config.weather_variables:
        values = hourly.get(var)
        if values is None:
            logger.warning("[open-meteo] missing variable: %s", var)
            values = [None] * len(times)
        data[var] = values

    df = pd.DataFrame(data)
    return df


def prepare_weather(df_raw: pd.DataFrame, config: AQIPipelineConfig) -> pd.DataFrame:
    """
    Normalize weather data to numeric columns and sorted timestamps.
    """
    if df_raw.empty:
        return df_raw.copy()

    df = df_raw.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True).dt.tz_localize(None)

    for col in config.weather_variables:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("ds").reset_index(drop=True)
    return df
