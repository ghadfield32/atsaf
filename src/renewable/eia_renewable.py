"""EIA renewable energy data fetcher for multi-series forecasting.

This module fetches wind (WND) and solar (SUN) generation data from the EIA API
for multiple regions, outputting data in StatsForecast format [unique_id, ds, y].
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from src.renewable.regions import FUEL_TYPES, REGIONS, validate_fuel_type, validate_region

load_dotenv()

logger = logging.getLogger(__name__)


class EIARenewableFetcher:
    """Fetch wind/solar data for multiple regions from EIA API.

    Outputs data in StatsForecast multi-series format:
    - unique_id: "{region}_{fuel_type}" (e.g., "CALI_WND")
    - ds: timezone-naive UTC datetime
    - y: generation value (MWh)

    Example:
        >>> fetcher = EIARenewableFetcher(api_key="your_key")
        >>> df = fetcher.fetch_all_regions(
        ...     fuel_type="WND",
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-07"
        ... )
        >>> print(df["unique_id"].unique())
        ['CALI_WND', 'ERCO_WND', 'MISO_WND', ...]
    """

    BASE_URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
    MAX_RECORDS_PER_REQUEST = 5000
    RATE_LIMIT_DELAY = 0.2  # 5 requests/second max

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the fetcher with API credentials.

        Args:
            api_key: EIA API key. If None, reads from EIA_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("EIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "EIA API key required. Set EIA_API_KEY environment variable "
                "or pass api_key parameter."
            )
        logger.info(f"EIARenewableFetcher initialized (API key length: {len(self.api_key)})")

    def fetch_region(
        self,
        region: str,
        fuel_type: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch data for a single region and fuel type.

        Args:
            region: EIA region code (e.g., 'CALI', 'ERCO')
            fuel_type: 'WND' for wind or 'SUN' for solar
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns [ds, value, region, fuel_type]

        Raises:
            ValueError: If region or fuel_type is invalid
            requests.HTTPError: If API request fails
        """
        if not validate_region(region):
            raise ValueError(f"Invalid region: {region}. Valid: {list(REGIONS.keys())}")
        if not validate_fuel_type(fuel_type):
            raise ValueError(f"Invalid fuel type: {fuel_type}. Valid: {list(FUEL_TYPES.keys())}")

        all_records = []
        offset = 0
        last_response_meta: Optional[dict] = None

        while True:
            params = {
                "api_key": self.api_key,
                "data[]": "value",
                "facets[respondent][]": region,
                "facets[fueltype][]": fuel_type,
                "frequency": "hourly",
                "start": f"{start_date}T00",
                "end": f"{end_date}T23",
                "length": self.MAX_RECORDS_PER_REQUEST,
                "offset": offset,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
            }
            request_meta = {
                "respondent": region,
                "fuel_type": fuel_type,
                "start": start_date,
                "end": end_date,
                "offset": offset,
                "length": self.MAX_RECORDS_PER_REQUEST,
                "frequency": "hourly",
            }

            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not self._validate_response(data):
                logger.warning(
                    f"Invalid response for {region}/{fuel_type}. "
                    f"request={request_meta}"
                )
                break

            records = data["response"]["data"]
            response_info = data.get("response", {})
            last_response_meta = {
                "total": response_info.get("total"),
                "returned": len(records),
                "offset": offset,
                "start": start_date,
                "end": end_date,
            }
            if not records:
                logger.warning(
                    f"No records in response for {region}/{fuel_type} "
                    f"(total={last_response_meta['total']}, offset={offset}, "
                    f"start={start_date}, end={end_date})"
                )
                break

            all_records.extend(records)
            offset += self.MAX_RECORDS_PER_REQUEST

            # Rate limiting
            time.sleep(self.RATE_LIMIT_DELAY)

        if not all_records:
            logger.warning(f"No data returned for {region}/{fuel_type}")
            if last_response_meta is not None:
                logger.warning(
                    f"Response meta for {region}/{fuel_type}: {last_response_meta}"
                )
            return pd.DataFrame(columns=["ds", "value", "region", "fuel_type"])

        df = pd.DataFrame(all_records)

        # Parse datetime
        df["ds"] = pd.to_datetime(df["period"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["region"] = region
        df["fuel_type"] = fuel_type

        # Drop rows with parsing errors
        df = df.dropna(subset=["ds", "value"])

        # Convert to UTC naive (StatsForecast requirement)
        if df["ds"].dt.tz is not None:
            df["ds"] = df["ds"].dt.tz_convert("UTC").dt.tz_localize(None)

        # Sort by datetime
        df = df.sort_values("ds").reset_index(drop=True)

        logger.info(f"Fetched {len(df)} rows for {region}/{fuel_type}")

        return df[["ds", "value", "region", "fuel_type"]]

    def fetch_all_regions(
        self,
        fuel_type: str,
        start_date: str,
        end_date: str,
        regions: Optional[list[str]] = None,
        max_workers: int = 3,
    ) -> pd.DataFrame:
        """Fetch data for all regions, return in StatsForecast format.

        Args:
            fuel_type: 'WND' for wind or 'SUN' for solar
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            regions: List of region codes. If None, uses all regions except US48.
            max_workers: Number of parallel workers (default: 3 for rate limiting)

        Returns:
            DataFrame with columns [unique_id, ds, y] in StatsForecast format.
            unique_id format: "{region}_{fuel_type}" (e.g., "CALI_WND")
        """
        if not validate_fuel_type(fuel_type):
            raise ValueError(f"Invalid fuel type: {fuel_type}")

        # Default to all regions except US48 (aggregate)
        if regions is None:
            regions = [r for r in REGIONS.keys() if r != "US48"]

        logger.info(
            f"Fetching {fuel_type} for {len(regions)} regions: {regions} "
            f"(start={start_date}, end={end_date})"
        )

        all_dfs = []
        failed_regions = []
        empty_regions = []

        # Fetch regions with controlled parallelism
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_region = {
                executor.submit(
                    self.fetch_region, region, fuel_type, start_date, end_date
                ): region
                for region in regions
            }

            for future in as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    df = future.result()
                    if len(df) > 0:
                        all_dfs.append(df)
                        logger.info(f"[OK] {region}: {len(df)} rows")
                    else:
                        logger.warning(f"[EMPTY] {region}: no data")
                        empty_regions.append(region)
                except Exception as e:
                    logger.error(f"[FAIL] {region}: {e}")
                    failed_regions.append(region)

        if not all_dfs:
            logger.error("No data retrieved from any region")
            return pd.DataFrame(columns=["unique_id", "ds", "y"])

        # Combine all regions
        combined = pd.concat(all_dfs, ignore_index=True)

        # Create unique_id for StatsForecast format
        combined["unique_id"] = combined["region"] + "_" + combined["fuel_type"]

        # Rename to StatsForecast convention
        combined = combined.rename(columns={"value": "y"})

        # Select and order columns
        result = combined[["unique_id", "ds", "y"]].copy()

        # Sort by unique_id and ds
        result = result.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        logger.info(
            f"Combined dataset: {len(result)} rows, "
            f"{result['unique_id'].nunique()} series, "
            f"{len(failed_regions)} failed regions, "
            f"{len(empty_regions)} empty regions"
        )

        if failed_regions:
            logger.warning(f"Failed regions: {failed_regions}")
        if empty_regions:
            logger.warning(f"Empty regions: {empty_regions}")

        return result

    def fetch_both_fuel_types(
        self,
        start_date: str,
        end_date: str,
        regions: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Fetch both wind and solar data for all regions.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            regions: List of region codes. If None, uses all regions except US48.

        Returns:
            DataFrame with columns [unique_id, ds, y] containing both WND and SUN.
            unique_id examples: "CALI_WND", "CALI_SUN", "ERCO_WND", etc.
        """
        dfs = []

        for fuel_type in ["WND", "SUN"]:
            logger.info(f"Fetching {fuel_type} data...")
            df = self.fetch_all_regions(
                fuel_type=fuel_type,
                start_date=start_date,
                end_date=end_date,
                regions=regions,
            )
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        logger.info(
            f"Combined WND+SUN: {len(combined)} rows, "
            f"{combined['unique_id'].nunique()} series"
        )

        return combined

    def _validate_response(self, data: dict) -> bool:
        """Validate API response structure.

        Args:
            data: Parsed JSON response

        Returns:
            True if response is valid, False otherwise
        """
        if not isinstance(data, dict):
            return False

        if "response" not in data:
            return False

        if "data" not in data["response"]:
            return False

        return True

    def get_date_range(self, df: pd.DataFrame) -> tuple[str, str]:
        """Get the date range of a dataset.

        Args:
            df: DataFrame with 'ds' column

        Returns:
            Tuple of (min_date, max_date) as strings
        """
        min_date = df["ds"].min().strftime("%Y-%m-%d")
        max_date = df["ds"].max().strftime("%Y-%m-%d")
        return (min_date, max_date)

    def get_series_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics per series.

        Args:
            df: DataFrame in StatsForecast format [unique_id, ds, y]

        Returns:
            DataFrame with summary stats per unique_id
        """
        summary = df.groupby("unique_id").agg(
            count=("y", "count"),
            min_value=("y", "min"),
            max_value=("y", "max"),
            mean_value=("y", "mean"),
            std_value=("y", "std"),
            min_date=("ds", "min"),
            max_date=("ds", "max"),
            zero_count=("y", lambda x: (x == 0).sum()),
        ).reset_index()

        summary["zero_pct"] = (summary["zero_count"] / summary["count"] * 100).round(1)

        return summary


if __name__ == "__main__":
    # Test the fetcher
    logging.basicConfig(level=logging.INFO)

    fetcher = EIARenewableFetcher()

    # Test single region
    print("\n=== Testing single region fetch ===")
    df_single = fetcher.fetch_region(
        region="CALI",
        fuel_type="WND",
        start_date="2024-12-01",
        end_date="2024-12-07",
    )
    print(f"Single region: {len(df_single)} rows")
    print(df_single.head())

    # Test multi-region
    print("\n=== Testing multi-region fetch ===")
    df_multi = fetcher.fetch_all_regions(
        fuel_type="WND",
        start_date="2024-12-01",
        end_date="2024-12-07",
        regions=["CALI", "ERCO", "MISO"],
    )
    print(f"Multi-region: {len(df_multi)} rows")
    print(f"Series: {df_multi['unique_id'].unique()}")

    # Summary
    print("\n=== Series summary ===")
    summary = fetcher.get_series_summary(df_multi)
    print(summary)
