"""
Chapter 1 Step 3: Prepare / Normalize Time

Standardize datetime to UTC and create canonical columns:
- unique_id: series identifier
- ds: datetime (UTC)
- y: numeric value
"""

import pandas as pd


def normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw API data to clean time series format.

    Steps:
    1. Parse period to datetime (UTC)
    2. Convert value to numeric
    3. Sort by datetime
    4. Standardize column names

    Args:
        df: Raw DataFrame from pull_data_paged

    Returns:
        Cleaned DataFrame with period, value, respondent, fueltype
    """
    df = df.copy()

    # Parse period to datetime with UTC
    df["period"] = pd.to_datetime(df["period"], utc=True)

    # Convert value to numeric
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Sort by datetime (should already be sorted from API, but ensure)
    df = df.sort_values("period").reset_index(drop=True)

    # Keep source timezone info if debugging needed
    df["tz_source"] = "UTC"

    # Select and rename columns
    df = df[["period", "value", "respondent", "fueltype", "tz_source"]]

    print(f"Normalized: {len(df)} rows, {df['period'].min()} to {df['period'].max()}")

    return df


def prepare_for_forecasting(
    df: pd.DataFrame,
    unique_id: str = "NG_US48",
) -> pd.DataFrame:
    """
    Prepare data for statsforecast/mlforecast format.

    Statsforecast requires: unique_id, ds (timezone-naive UTC), y (values)

    Args:
        df: Cleaned DataFrame from normalize_time
        unique_id: Series identifier (default: fuel_region like "NG_US48")

    Returns:
        DataFrame with columns [unique_id, ds, y]
        Note: ds is timezone-naive UTC (required by StatsForecast)
    """
    # Convert ds to timezone-naive UTC (required by StatsForecast)
    ds_naive = pd.to_datetime(df["period"], errors="raise", utc=True).dt.tz_localize(None)

    df_forecast = pd.DataFrame({
        "unique_id": unique_id,
        "ds": ds_naive,
        "y": df["value"],
    })

    print(f"Forecast format: {len(df_forecast)} rows, unique_id={unique_id}")
    print(f"  ds: timezone-naive UTC")

    return df_forecast
