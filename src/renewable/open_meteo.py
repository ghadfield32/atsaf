# src/renewable/open_meteo.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.renewable.regions import get_region_coords, validate_region


@dataclass(frozen=True)
class OpenMeteoEndpoints:
    historical_url: str = "https://archive-api.open-meteo.com/v1/archive"
    forecast_url: str = "https://api.open-meteo.com/v1/forecast"


class OpenMeteoRenewable:
    """
    Fetch weather features for renewable energy forecasting.

    Strict-by-default:
    - If Open-Meteo doesn't return a requested variable, we raise.
    - We do NOT fabricate values or silently "fill" missing columns.
    """

    WEATHER_VARS = [
        "temperature_2m",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_10m",
        "direct_radiation",
        "diffuse_radiation",
        "cloud_cover",
    ]

    def __init__(self, timeout: int = 60, *, strict: bool = True):
        self.timeout = timeout
        self.strict = strict
        self.endpoints = OpenMeteoEndpoints()
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1.0,  # 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
            connect=3,  # Retry on connection errors
            read=3,     # Retry on read timeouts
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def fetch_historical(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        variables: Optional[list[str]] = None,
        *,
        debug: bool = False,
    ) -> pd.DataFrame:
        if variables is None:
            variables = self.WEATHER_VARS

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": "UTC",
        }

        resp = self.session.get(self.endpoints.historical_url, params=params, timeout=self.timeout)
        if debug:
            print(f"[OPENMETEO][HIST] status={resp.status_code} url={resp.url}")
        resp.raise_for_status()

        try:
            data = resp.json()
        except requests.exceptions.JSONDecodeError as e:
            # Log actual response content for debugging
            content_preview = resp.text[:500] if resp.text else "(empty)"
            raise ValueError(
                f"[OPENMETEO][HIST] Invalid JSON response. "
                f"status={resp.status_code} content_preview={content_preview}"
            ) from e

        return self._parse_response(data, variables, debug=debug, request_url=resp.url)

    def fetch_forecast(
        self,
        lat: float,
        lon: float,
        horizon_hours: int = 48,
        variables: Optional[list[str]] = None,
        *,
        debug: bool = False,
    ) -> pd.DataFrame:
        if variables is None:
            variables = self.WEATHER_VARS

        forecast_days = min((horizon_hours // 24) + 1, 16)
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(variables),
            "timezone": "UTC",
            "forecast_days": forecast_days,
        }

        resp = self.session.get(self.endpoints.forecast_url, params=params, timeout=self.timeout)
        if debug:
            print(f"[OPENMETEO][FCST] status={resp.status_code} url={resp.url}")
        resp.raise_for_status()

        try:
            data = resp.json()
        except requests.exceptions.JSONDecodeError as e:
            content_preview = resp.text[:500] if resp.text else "(empty)"
            raise ValueError(
                f"[OPENMETEO][FCST] Invalid JSON response. "
                f"status={resp.status_code} content_preview={content_preview}"
            ) from e

        df = self._parse_response(data, variables, debug=debug, request_url=resp.url)

        # Trim to requested horizon (ds is naive UTC)
        if len(df) > 0:
            cutoff = datetime.utcnow() + timedelta(hours=horizon_hours)
            df = df[df["ds"] <= cutoff].reset_index(drop=True)

        return df

    def fetch_for_region(
        self,
        region_code: str,
        start_date: str,
        end_date: str,
        *,
        debug: bool = False,
    ) -> pd.DataFrame:
        if not validate_region(region_code):
            raise ValueError(f"Invalid region_code: {region_code}")

        lat, lon = get_region_coords(region_code)
        df = self.fetch_historical(lat, lon, start_date, end_date, debug=debug)
        df["region"] = region_code
        return df

    def fetch_all_regions_historical(
        self,
        regions: list[str],
        start_date: str,
        end_date: str,
        *,
        debug: bool = False,
    ) -> pd.DataFrame:
        all_dfs: list[pd.DataFrame] = []
        for region in regions:
            try:
                df = self.fetch_for_region(region, start_date, end_date, debug=debug)
                all_dfs.append(df)
                print(f"[OK] Weather for {region}: {len(df)} rows")
            except requests.exceptions.Timeout as e:
                print(f"[FAIL] Weather for {region}: TIMEOUT after {self.timeout}s - {type(e).__name__}: {e}")
            except requests.exceptions.ConnectionError as e:
                print(f"[FAIL] Weather for {region}: CONNECTION_ERROR - {type(e).__name__}: {e}")
            except requests.exceptions.JSONDecodeError as e:
                print(f"[FAIL] Weather for {region}: JSON_PARSE_ERROR - {type(e).__name__}: {e}")
            except Exception as e:
                print(f"[FAIL] Weather for {region}: {type(e).__name__}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        return (
            pd.concat(all_dfs, ignore_index=True)
            .sort_values(["region", "ds"])
            .reset_index(drop=True)
        )

    def _parse_response(
        self,
        data: dict,
        variables: list[str],
        *,
        debug: bool,
        request_url: str,
    ) -> pd.DataFrame:
        hourly = data.get("hourly")
        if not isinstance(hourly, dict):
            raise ValueError(f"Open-Meteo response missing/invalid 'hourly'. url={request_url}")

        times = hourly.get("time")
        if not isinstance(times, list) or len(times) == 0:
            raise ValueError(f"Open-Meteo response has no hourly time grid. url={request_url}")

        # Build ds (naive UTC)
        ds = pd.to_datetime(times, errors="coerce", utc=True).tz_localize(None)
        if ds.isna().any():
            bad = int(ds.isna().sum())
            raise ValueError(f"Open-Meteo returned unparsable times. bad={bad} url={request_url}")

        df_data = {"ds": ds}

        # Strict variable presence: raise if missing (no silent None padding)
        missing_vars = [v for v in variables if v not in hourly]
        if missing_vars and self.strict:
            raise ValueError(f"Open-Meteo missing requested vars={missing_vars}. url={request_url}")

        for var in variables:
            values = hourly.get(var)
            if values is None:
                # If not strict, keep as all-NA but be explicit (not hidden)
                df_data[var] = [None] * len(ds)
                continue

            if not isinstance(values, list):
                raise ValueError(f"Open-Meteo var '{var}' not a list. type={type(values)} url={request_url}")

            if len(values) != len(ds):
                raise ValueError(
                    f"Open-Meteo length mismatch for '{var}': "
                    f"len(values)={len(values)} len(time)={len(ds)} url={request_url}"
                )

            df_data[var] = pd.to_numeric(values, errors="coerce")

        df = pd.DataFrame(df_data).sort_values("ds").reset_index(drop=True)

        if debug:
            dup = int(df["ds"].duplicated().sum())
            na_counts = {v: int(df[v].isna().sum()) for v in variables if v in df.columns}
            print(f"[OPENMETEO][PARSE] rows={len(df)} dup_ds={dup} na_counts(sample)={dict(list(na_counts.items())[:3])}")

        return df

    def fetch_for_region_forecast(
        self,
        region_code: str,
        horizon_hours: int = 48,
        variables: Optional[list[str]] = None,
        *,
        debug: bool = False,
    ) -> pd.DataFrame:
        if not validate_region(region_code):
            raise ValueError(f"Invalid region_code: {region_code}")

        lat, lon = get_region_coords(region_code)
        df = self.fetch_forecast(lat, lon, horizon_hours=horizon_hours, variables=variables, debug=debug)
        df["region"] = region_code
        return df


    def fetch_all_regions_forecast(
        self,
        regions: list[str],
        horizon_hours: int = 48,
        variables: Optional[list[str]] = None,
        *,
        debug: bool = False,
    ) -> pd.DataFrame:
        all_dfs: list[pd.DataFrame] = []
        for region in regions:
            try:
                df = self.fetch_for_region_forecast(
                    region, horizon_hours=horizon_hours, variables=variables, debug=debug
                )
                all_dfs.append(df)
                print(f"[OK] Forecast weather for {region}: {len(df)} rows")
            except requests.exceptions.Timeout as e:
                print(f"[FAIL] Forecast weather for {region}: TIMEOUT after {self.timeout}s - {type(e).__name__}: {e}")
            except requests.exceptions.ConnectionError as e:
                print(f"[FAIL] Forecast weather for {region}: CONNECTION_ERROR - {type(e).__name__}: {e}")
            except requests.exceptions.JSONDecodeError as e:
                print(f"[FAIL] Forecast weather for {region}: JSON_PARSE_ERROR - {type(e).__name__}: {e}")
            except Exception as e:
                print(f"[FAIL] Forecast weather for {region}: {type(e).__name__}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        return (
            pd.concat(all_dfs, ignore_index=True)
            .sort_values(["region", "ds"])
            .reset_index(drop=True)
        )



if __name__ == "__main__":
    # Real API smoke test (no key needed)
    weather = OpenMeteoRenewable(strict=True)

    print("=== Testing Historical Weather (REAL API) ===")
    hist_df = weather.fetch_for_region("CALI", "2024-12-01", "2024-12-03", debug=True)
    print(f"Historical rows: {len(hist_df)}")
    print(hist_df.head())
