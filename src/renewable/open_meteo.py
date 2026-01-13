"""Open-Meteo weather data integration for renewable energy forecasting.

This module fetches weather features relevant to wind and solar generation:
- Wind: wind_speed_10m, wind_speed_100m, wind_direction_10m
- Solar: direct_radiation, diffuse_radiation, cloud_cover
- Common: temperature_2m

Provides both historical and forecast data to prevent leakage:
- Training: Use fetch_historical() with dates matching generation data
- Prediction: Use fetch_forecast() for future hours (up to 16 days)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.renewable.regions import REGIONS, get_region_coords

logger = logging.getLogger(__name__)


class OpenMeteoRenewable:
    """Fetch weather features for renewable energy forecasting.

    Uses Open-Meteo APIs (free, no API key required):
    - Historical: archive-api.open-meteo.com (past data)
    - Forecast: api.open-meteo.com (up to 16 days ahead)

    Example:
        >>> weather = OpenMeteoRenewable()
        >>> # Historical for training
        >>> hist = weather.fetch_historical(
        ...     lat=36.7, lon=-119.4,
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-07"
        ... )
        >>> # Forecast for prediction (no leakage!)
        >>> fcst = weather.fetch_forecast(lat=36.7, lon=-119.4, horizon_hours=24)
    """

    HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    # Weather variables relevant for renewable forecasting
    WEATHER_VARS = [
        "temperature_2m",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_10m",
        "direct_radiation",
        "diffuse_radiation",
        "cloud_cover",
    ]

    # Subset for wind forecasting
    WIND_VARS = [
        "temperature_2m",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_10m",
    ]

    # Subset for solar forecasting
    SOLAR_VARS = [
        "temperature_2m",
        "direct_radiation",
        "diffuse_radiation",
        "cloud_cover",
    ]

    def __init__(self, timeout: int = 30):
        """Initialize the weather fetcher.

        Args:
            timeout: Request timeout in seconds (default: 30)
        """
        self.timeout = timeout
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a session with retry logic."""
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
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
    ) -> pd.DataFrame:
        """Fetch historical hourly weather data.

        Use this for training data to align with historical generation.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: Weather variables to fetch (default: all)

        Returns:
            DataFrame with columns [ds, temperature_2m, wind_speed_10m, ...]
        """
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

        logger.info(f"Fetching historical weather: {start_date} to {end_date}")

        response = self.session.get(
            self.HISTORICAL_URL, params=params, timeout=self.timeout
        )
        response.raise_for_status()

        return self._parse_response(response.json(), variables)

    def fetch_forecast(
        self,
        lat: float,
        lon: float,
        horizon_hours: int = 48,
        variables: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Fetch weather forecast for future predictions.

        CRITICAL: Use this for prediction features to avoid leakage!
        Do NOT use historical weather for future forecasts.

        Args:
            lat: Latitude
            lon: Longitude
            horizon_hours: Number of hours to forecast (default: 48, max ~384)
            variables: Weather variables to fetch (default: all)

        Returns:
            DataFrame with columns [ds, temperature_2m, wind_speed_10m, ...]
        """
        if variables is None:
            variables = self.WEATHER_VARS

        # Calculate forecast days needed
        forecast_days = min((horizon_hours // 24) + 1, 16)

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(variables),
            "timezone": "UTC",
            "forecast_days": forecast_days,
        }

        logger.info(f"Fetching weather forecast: {horizon_hours}h ({forecast_days} days)")

        response = self.session.get(
            self.FORECAST_URL, params=params, timeout=self.timeout
        )
        response.raise_for_status()

        df = self._parse_response(response.json(), variables)

        # Trim to requested horizon
        if len(df) > 0:
            cutoff = datetime.utcnow() + timedelta(hours=horizon_hours)
            df = df[df["ds"] <= cutoff].reset_index(drop=True)

        return df

    def fetch_for_region(
        self,
        region_code: str,
        start_date: str,
        end_date: str,
        variables: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Fetch historical weather using region centroid coordinates.

        Args:
            region_code: EIA region code (e.g., 'CALI', 'ERCO')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: Weather variables to fetch

        Returns:
            DataFrame with weather data for the region
        """
        lat, lon = get_region_coords(region_code)
        df = self.fetch_historical(lat, lon, start_date, end_date, variables)
        df["region"] = region_code
        return df

    def fetch_forecast_for_region(
        self,
        region_code: str,
        horizon_hours: int = 48,
        variables: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Fetch weather forecast for a region.

        Args:
            region_code: EIA region code (e.g., 'CALI', 'ERCO')
            horizon_hours: Number of hours to forecast
            variables: Weather variables to fetch

        Returns:
            DataFrame with forecast data for the region
        """
        lat, lon = get_region_coords(region_code)
        df = self.fetch_forecast(lat, lon, horizon_hours, variables)
        df["region"] = region_code
        return df

    def fetch_all_regions_historical(
        self,
        regions: list[str],
        start_date: str,
        end_date: str,
        variables: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Fetch historical weather for multiple regions.

        Args:
            regions: List of region codes
            start_date: Start date
            end_date: End date
            variables: Weather variables to fetch

        Returns:
            DataFrame with columns [ds, region, temperature_2m, ...]
        """
        all_dfs = []

        for region in regions:
            try:
                df = self.fetch_for_region(region, start_date, end_date, variables)
                all_dfs.append(df)
                logger.info(f"[OK] Weather for {region}: {len(df)} rows")
            except Exception as e:
                logger.error(f"[FAIL] Weather for {region}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values(["region", "ds"]).reset_index(drop=True)

        return combined

    def fetch_all_regions_forecast(
        self,
        regions: list[str],
        horizon_hours: int = 48,
        variables: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Fetch weather forecast for multiple regions.

        Args:
            regions: List of region codes
            horizon_hours: Number of hours to forecast
            variables: Weather variables to fetch

        Returns:
            DataFrame with columns [ds, region, temperature_2m, ...]
        """
        all_dfs = []

        for region in regions:
            try:
                df = self.fetch_forecast_for_region(region, horizon_hours, variables)
                all_dfs.append(df)
                logger.info(f"[OK] Forecast for {region}: {len(df)} rows")
            except Exception as e:
                logger.error(f"[FAIL] Forecast for {region}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values(["region", "ds"]).reset_index(drop=True)

        return combined

    def _parse_response(
        self, data: dict, variables: list[str]
    ) -> pd.DataFrame:
        """Parse Open-Meteo API response to DataFrame.

        Args:
            data: JSON response from API
            variables: Expected variables

        Returns:
            DataFrame with parsed data
        """
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        if not times:
            logger.warning("No data returned from Open-Meteo API")
            return pd.DataFrame(columns=["ds"] + variables)

        # Parse timestamps to UTC-naive datetimes
        df_data = {
            "ds": pd.to_datetime(times, errors="coerce", utc=True).tz_localize(None)
        }

        for var in variables:
            values = hourly.get(var)
            if values is None:
                logger.warning(f"Missing variable in response: {var}")
                values = [None] * len(times)
            df_data[var] = values

        df = pd.DataFrame(df_data)

        # Convert to numeric
        for var in variables:
            if var in df.columns:
                df[var] = pd.to_numeric(df[var], errors="coerce")

        df = df.sort_values("ds").reset_index(drop=True)

        return df

    def merge_with_generation(
        self,
        generation_df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge generation data with weather features.

        Args:
            generation_df: DataFrame with [unique_id, ds, y]
            weather_df: DataFrame with [ds, region, weather_vars...]

        Returns:
            Merged DataFrame with weather features added
        """
        # Extract region from unique_id (format: "REGION_FUEL")
        gen = generation_df.copy()
        gen["region"] = gen["unique_id"].str.split("_").str[0]

        # Merge on region and timestamp
        merged = gen.merge(
            weather_df,
            on=["region", "ds"],
            how="left",
        )

        # Drop the temporary region column
        merged = merged.drop(columns=["region"])

        missing_count = merged[self.WEATHER_VARS[0]].isna().sum()
        if missing_count > 0:
            logger.warning(
                f"Weather merge: {missing_count}/{len(merged)} rows missing weather data"
            )

        return merged


if __name__ == "__main__":
    # Test the weather fetcher
    logging.basicConfig(level=logging.INFO)

    weather = OpenMeteoRenewable()

    # Test historical fetch
    print("\n=== Testing historical weather ===")
    hist_df = weather.fetch_for_region(
        region_code="CALI",
        start_date="2024-12-01",
        end_date="2024-12-07",
    )
    print(f"Historical: {len(hist_df)} rows")
    print(hist_df.head())

    # Test forecast fetch
    print("\n=== Testing weather forecast ===")
    fcst_df = weather.fetch_forecast_for_region(
        region_code="CALI",
        horizon_hours=24,
    )
    print(f"Forecast: {len(fcst_df)} rows")
    print(fcst_df.head())

    # Test multi-region
    print("\n=== Testing multi-region weather ===")
    multi_df = weather.fetch_all_regions_historical(
        regions=["CALI", "ERCO", "MISO"],
        start_date="2024-12-01",
        end_date="2024-12-03",
    )
    print(f"Multi-region: {len(multi_df)} rows")
    print(multi_df.groupby("region").size())
