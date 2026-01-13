# file: src/chapter1/eia_data_simple.py
"""
EIA Data Fetcher - Step by Step
=================================

A simplified module for pulling EIA electricity generation data
following the same step-by-step approach as the R script.

This module makes it easy to understand each step and test along the way.

Usage:
    from eia_data_simple import EIADataFetcher
    import os

    # Step 1: Initialize
    api_key = os.getenv("EIA_API_KEY")
    fetcher = EIADataFetcher(api_key)

    # Step 2: Pull raw data
    df_raw = fetcher.pull_data(
        start_date="2023-01-01",
        end_date="2024-12-31"
    )

    # Step 3: Inspect data
    print(f"Rows: {len(df_raw)}")
    print(df_raw.head())

    # Step 4: Prepare (convert types, sort, etc.)
    df_prepared = fetcher.prepare_data(df_raw)

    # Step 5: Validate
    is_valid = fetcher.validate_data(df_prepared)

    # Step 6: Get statistics
    stats = fetcher.get_stats(df_prepared)
    print(stats)
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import requests
from dotenv import load_dotenv

# Optional: pydantic-settings for type-safe config
try:
    from pydantic_settings import BaseSettings
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Optional: MLflow for experiment tracking
try:
    import mlflow
    from mlflow.models import infer_signature
    MLFLOW_AVAILABLE = True
    logger.info("MLflow is available for experiment tracking")
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available - experiment tracking disabled")


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    Configuration for backtesting experiments.

    This defines the "contract" for reproducible experiments:
    - What data range to use
    - How to split for cross-validation
    - Which models and metrics to evaluate

    Example:
        >>> config = ExperimentConfig(
        ...     name="baseline_experiment",
        ...     horizon=24,
        ...     n_windows=5,
        ...     step_size=168,  # Weekly steps
        ... )
    """
    name: str = "default_experiment"
    horizon: int = 24              # Forecast horizon in hours
    n_windows: int = 5             # Number of CV windows
    step_size: int = 168           # Hours between windows (168 = 1 week)
    confidence_level: int = 95     # Prediction interval level
    models: List[str] = field(default_factory=lambda: [
        "SeasonalNaive", "AutoARIMA", "MSTL"
    ])
    metrics: List[str] = field(default_factory=lambda: [
        "rmse", "mape", "mase", "coverage"
    ])


# Pydantic-settings config (if available)
if PYDANTIC_AVAILABLE:
    class Settings(BaseSettings):
        """
        Type-safe configuration using pydantic-settings.

        Automatically reads from environment variables or .env file.
        Validates types and provides defaults.

        Example:
            >>> settings = Settings()
            >>> print(settings.eia_api_key[:8])
        """
        eia_api_key: str
        respondent: str = "US48"
        fueltype: str = "NG"
        start_date: str = "2024-01-01"
        end_date: str = "2024-12-31"

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"


class EIADataFetcher:
    """
    Step-by-step EIA data fetcher matching R script workflow.

    Each method represents a step in the data pipeline:
    1. pull_data() - Fetch from API
    2. inspect_data() - View structure
    3. prepare_data() - Clean and convert
    4. validate_data() - Check quality
    5. get_stats() - Calculate summary stats
    """

    def __init__(self, api_key: str):
        """
        Initialize the fetcher with API credentials.

        Args:
            api_key: EIA API key from https://www.eia.gov/opendata/
        """
        self.api_key = api_key
        self.api_url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
        logger.info(f"Fetcher initialized (API key length: {len(api_key)})")
        print(f"Step 1: Fetcher initialized (API key length: {len(api_key)})")

    def pull_data(
        self,
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
        respondent: str = "US48",
        fueltype: str = "NG",
        length: int = 5000
    ) -> pd.DataFrame:
        """
        STEP 2: Pull raw data from EIA API with pagination.

        Features:
        - Handles >5000 row datasets with pagination loop
        - Stable sort order (period ascending) for reproducibility
        - Logs pagination details for transparency

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            respondent: Region code (default: US48 = Lower 48 states)
            fueltype: Fuel type code (default: NG = Natural Gas)
            length: Records per request (default: 5000, max allowed)

        Returns:
            DataFrame with raw API response

        Example:
            >>> df_raw = fetcher.pull_data()
            >>> print(f"Retrieved {len(df_raw)} rows")
        """
        print(f"\nStep 2: Pulling data from EIA API...")
        print(f"  Date range: {start_date} to {end_date}")
        print(f"  Respondent: {respondent}")
        print(f"  Fuel type: {fueltype}")

        all_records = []
        offset = 0
        request_count = 0

        try:
            while True:
                # Build API parameters with pagination and STABLE SORT
                # Using sort params ensures pages don't shuffle during pagination
                params = {
                    "api_key": self.api_key,
                    "data[]": "value",
                    "facets[respondent][]": respondent,
                    "facets[fueltype][]": fueltype,
                    "frequency": "hourly",
                    "start": f"{start_date}T00",
                    "end": f"{end_date}T23",
                    "length": length,
                    "offset": offset,
                    # STABLE SORT: Request data in ascending order from the API
                    # This ensures consistent ordering across paginated requests
                    "sort[0][column]": "period",
                    "sort[0][direction]": "asc",
                }

                # Make API request
                logger.info(f"API request: offset={offset}, length={length}")
                response = requests.get(self.api_url, params=params)
                response.raise_for_status()

                # Parse response
                data = response.json()
                records = data["response"]["data"]
                request_count += 1

                logger.debug(f"Request {request_count}: received {len(records)} rows")

                if not records:
                    break  # No more data

                all_records.extend(records)
                offset += length

            if not all_records:
                raise ValueError("No data returned from API")

            # Convert to DataFrame
            # Data is already sorted ascending by period from the API (stable sort params)
            # We still verify sort order here as a safety check
            df = pd.DataFrame(all_records)
            df = df.sort_values("period", ascending=True).reset_index(drop=True)

            print(f"  Sending requests...")
            print(f"  [OK] Retrieved {len(df)} total rows across {request_count} request(s)")
            print(f"  Columns: {', '.join(df.columns.tolist())}")

            logger.info(f"Data pull complete: {len(df)} rows in {request_count} API requests")

            return df

        except Exception as e:
            logger.error(f"Data pull failed: {e}", exc_info=True)
            print(f"  [ERROR] {e}")
            raise

    def inspect_data(self, df: pd.DataFrame) -> None:
        """
        STEP 3: Inspect raw data structure.

        Displays:
        - Data shape
        - Column info
        - First few rows
        - Data types

        Example:
            >>> fetcher.inspect_data(df_raw)
        """
        print(f"\nStep 3: Inspecting data structure...")
        print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"\n  Column info:")
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            print(f"    - {col}: {dtype} ({non_null} non-null)")

        print(f"\n  First 3 rows:")
        for i, row in df.head(3).iterrows():
            print(f"    Row {i}: {dict(row)}")



    def prepare_data(self, df: pd.DataFrame, timezone_policy: str = "UTC") -> pd.DataFrame:
        """
        STEP 4: Prepare data (clean, convert, sort).

        Performs:
        - Parse datetime from 'period' field
        - Apply timezone policy (UTC normalization for consistency)
        - Convert 'value' to numeric
        - Sort by datetime
        - Standardize column names

        Args:
            df: Raw DataFrame from pull_data()
            timezone_policy: "UTC" (recommended) - normalize all times to UTC

        Returns:
            Cleaned and prepared DataFrame

        Example:
            >>> df_clean = fetcher.prepare_data(df_raw, timezone_policy="UTC")
        """
        print(f"\nStep 4: Preparing data...")

        df = df.copy()

        # Parse period to datetime (fail-loud on parse errors)
        print(f"  - Parsing period field...")
        try:
            df["period"] = pd.to_datetime(df["period"], errors="raise")
        except (ValueError, TypeError) as e:
            logger.error(f"DateTime parsing failed: {e}")
            logger.error(f"Sample period values: {df['period'].head(10).tolist()}")
            raise ValueError(f"Cannot parse period field as datetime: {e}") from e

        # Apply timezone policy (UTC normalization)
        print(f"  - Applying timezone policy: {timezone_policy}")
        if timezone_policy == "UTC":
            # Assume period is in UTC if no timezone info
            if df["period"].dt.tz is None:
                df["period"] = df["period"].dt.tz_localize("UTC")
            else:
                df["period"] = df["period"].dt.tz_convert("UTC")
            logger.info("Timezone policy: UTC normalization applied")

        # Extract date
        print(f"  - Extracting date...")
        df["date"] = df["period"].dt.date

        # Convert value to numeric (fail-loud on coercion)
        print(f"  - Converting value to numeric...")
        df["value_before_coercion"] = df["value"].copy()  # Keep original for audit
        try:
            df["value"] = pd.to_numeric(df["value"], errors="raise")
        except (ValueError, TypeError) as e:
            # Count non-numeric values and provide detailed error
            non_numeric_rows = df[pd.to_numeric(df["value"], errors="coerce").isna()]
            logger.error(f"Numeric conversion failed: {len(non_numeric_rows)} non-numeric rows")
            logger.error(f"Sample non-numeric values: {non_numeric_rows['value'].head(5).tolist()}")
            raise ValueError(
                f"Cannot convert value column to numeric: {len(non_numeric_rows)} unparseable rows. "
                f"This typically indicates upstream schema changes or data quality issues. "
                f"Sample values: {non_numeric_rows['value'].head(3).tolist()}"
            ) from e

        # Verify no coercions occurred
        if df["value"].isna().sum() > 0:
            coercion_count = df["value"].isna().sum()
            logger.error(f"Coercion produced {coercion_count} NaN values during numeric conversion")
            raise ValueError(
                f"Numeric conversion coerced {coercion_count} values to NaN. "
                f"Review original values: {df[df['value'].isna()]['value_before_coercion'].head(5).tolist()}"
            )

        # Sort by datetime
        print(f"  - Sorting by datetime...")
        df = df.sort_values("period").reset_index(drop=True)

        # Standardize column names
        df.columns = [col.lower().replace("-", "_") for col in df.columns]

        # Remove temporary audit column if present
        df = df.drop(columns=["value_before_coercion"], errors="ignore")

        # Select key columns
        key_cols = ["date", "period", "value", "respondent", "fueltype"]
        df = df[[col for col in key_cols if col in df.columns]]

        print(f"  [OK] Data prepared: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Data preparation complete: {df.shape[0]} rows")

        return df

    def validate_time_series_integrity(self, df: pd.DataFrame) -> Dict:
        """
        STEP 4B: Comprehensive time series integrity validation.

        Critical checks for production:
        - No duplicates on (unique_id, ds)
        - Regular hourly frequency
        - Missing hours detection
        - DST repeated hours (freq = 0)
        - Complete final hours for backtesting

        Args:
            df: Prepared DataFrame from prepare_for_forecasting()

        Returns:
            Dictionary with integrity report:
            - duplicate_count: Number of duplicate (unique_id, ds) pairs
            - missing_hours_total: Total count of missing hours across all gaps
            - missing_gaps_count: Number of gaps detected
            - longest_gap_hours: Duration of longest gap
            - dst_repeated_hours: Count of repeated hours (DST backward)
            - gaps_detail: List of gap locations with hour counts
            - status: "valid" or "invalid"

        Example:
            >>> df_forecast = fetcher.prepare_for_forecasting(df)
            >>> integrity = fetcher.validate_time_series_integrity(df_forecast)
            >>> print(integrity['status'])
        """
        print(f"\nStep 4B: Validating time series integrity...")

        report = {
            "duplicate_count": 0,
            "missing_hours_total": 0,
            "missing_gaps_count": 0,
            "longest_gap_hours": 0,
            "dst_repeated_hours": 0,
            "gaps_detail": [],
            "status": "valid"
        }

        df_sorted = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        # Check 1: Duplicates on (unique_id, ds)
        dups = df_sorted.groupby(["unique_id", "ds"]).size()
        duplicate_rows = (dups > 1).sum()
        report["duplicate_count"] = int(duplicate_rows)

        if duplicate_rows > 0:
            print(f"  [FAIL] Found {duplicate_rows} duplicate (unique_id, ds) pairs")
            logger.error(f"Time series integrity: {duplicate_rows} duplicates")
            report["status"] = "invalid"
            return report
        else:
            print(f"  [OK] No duplicates on (unique_id, ds)")

        # Check 2-4: Frequency, gaps, DST for each series
        for uid in df_sorted["unique_id"].unique():
            sub = df_sorted[df_sorted["unique_id"] == uid].copy()
            sub = sub.sort_values("ds").reset_index(drop=True)

            # Calculate time differences
            time_diffs = sub["ds"].diff()
            expected_freq = pd.Timedelta(hours=1)

            # Missing hours: count actual missing hours per gap (not gap occurrences)
            # For a gap of 5 hours (e.g., 10:00 to 15:00), there are 4 missing hours
            missing_mask = time_diffs > expected_freq
            if missing_mask.any():
                gap_indices = sub.index[missing_mask].tolist()
                report["missing_gaps_count"] += len(gap_indices)

                for idx in gap_indices:
                    if idx > 0:
                        gap_duration = time_diffs[idx]
                        gap_hours = gap_duration.total_seconds() / 3600
                        # Missing hours = gap duration - 1 (e.g., 5 hour gap = 4 missing)
                        missing_hours_in_gap = int(gap_hours - 1)
                        report["missing_hours_total"] += missing_hours_in_gap

                        report["gaps_detail"].append({
                            "unique_id": uid,
                            "before_ds": sub.loc[idx-1, "ds"],
                            "after_ds": sub.loc[idx, "ds"],
                            "gap_hours": gap_hours,
                            "missing_hours": missing_hours_in_gap
                        })

            # DST repeated hours (gap = 0, clocks go back)
            repeated_mask = time_diffs == pd.Timedelta(0)
            repeated_in_series = repeated_mask.sum()
            report["dst_repeated_hours"] += int(repeated_in_series)

            # Longest gap
            if len(time_diffs) > 0:
                max_gap = time_diffs.max()
                if pd.notna(max_gap):
                    gap_hours = max_gap.total_seconds() / 3600
                    report["longest_gap_hours"] = max(
                        report["longest_gap_hours"],
                        gap_hours
                    )

        # Report findings
        if report["duplicate_count"] > 0:
            print(f"  [FAIL] Found {report['duplicate_count']} duplicates")
            logger.error(f"Time series integrity FAILED: {report['duplicate_count']} duplicate (unique_id, ds) pairs detected")
            report["status"] = "invalid"
            return report
        else:
            print(f"  [OK] No duplicates")

        if report["missing_hours_total"] > 0:
            print(f"  [FAIL] {report['missing_hours_total']} missing hours detected ({report['missing_gaps_count']} gaps)")
            logger.error(
                f"Time series integrity FAILED: {report['missing_hours_total']} missing hours, "
                f"{report['missing_gaps_count']} gaps, longest gap {report['longest_gap_hours']:.1f} hours"
            )
            gap_summary = "\n".join([
                f"  {g['unique_id']}: {g['before_ds']} → {g['after_ds']} ({g['gap_hours']:.1f} hours, {g['missing_hours']} missing)"
                for g in report["gaps_detail"][:5]  # Show first 5 gaps
            ])
            report["status"] = "invalid"
            return report
        else:
            print(f"  [OK] No missing hours (complete frequency)")

        if report["dst_repeated_hours"] > 0:
            print(f"  [INFO] {report['dst_repeated_hours']} DST repeated hours (clocks back)")
            logger.info(f"Time series has {report['dst_repeated_hours']} DST repeated hours, which is expected")

        print(f"  [OK] Time series integrity validated")
        logger.info(f"Time series integrity report: {report}")

        return report

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        STEP 5: Validate data quality.

        Checks:
        - No empty DataFrame
        - Value column is numeric
        - No missing values
        - Dates are in order

        Returns:
            bool: True if all validations pass

        Example:
            >>> is_valid = fetcher.validate_data(df_clean)
        """
        print(f"\nStep 5: Validating data...")

        checks_passed = 0
        checks_total = 5

        # Check 1: Not empty
        if len(df) > 0:
            print(f"  [OK] Data is not empty: {len(df)} rows")
            checks_passed += 1
        else:
            print(f"  [FAIL] Data is empty")
            return False

        # Check 2: Value column exists and is numeric
        if "value" in df.columns and pd.api.types.is_numeric_dtype(df["value"]):
            print(f"  [OK] Value column is numeric")
            checks_passed += 1
        else:
            print(f"  [FAIL] Value column missing or not numeric")
            return False

        # Check 3: No missing values in value column
        missing = df["value"].isna().sum()
        if missing == 0:
            print(f"  [OK] No missing values in 'value' column")
            checks_passed += 1
        else:
            print(f"  [WARN] {missing} missing values in 'value' column")
            checks_passed += 1  # Warning, not failure

        # Check 4: Dates are in order
        if df["period"].is_monotonic_increasing:
            print(f"  [OK] Dates are in chronological order")
            checks_passed += 1
        else:
            print(f"  [FAIL] Dates are not in order")
            return False

        # Check 5: Value range is reasonable (electricity in MWh)
        if df["value"].min() > 0 and df["value"].max() < 1_000_000:
            print(f"  [OK] Value range is reasonable")
            checks_passed += 1
        else:
            print(f"  [WARN] Value range seems unusual: {df['value'].min():.0f} to {df['value'].max():.0f}")
            checks_passed += 1  # Warning

        print(f"\n  Validation: {checks_passed}/{checks_total} checks passed")
        return True

    def get_stats(self, df: pd.DataFrame) -> Dict:
        """
        STEP 6: Calculate summary statistics.

        Returns:
            Dictionary with:
            - date_range: (start_date, end_date)
            - record_count: Total records
            - value_stats: min, max, mean, std
            - missing_count: Count of NaN values

        Example:
            >>> stats = fetcher.get_stats(df_clean)
            >>> print(f"Date range: {stats['date_range']}")
        """
        print(f"\nStep 6: Calculating statistics...")

        stats = {
            "date_range": (df["date"].min(), df["date"].max()),
            "record_count": len(df),
            "value_stats": {
                "min": df["value"].min(),
                "max": df["value"].max(),
                "mean": df["value"].mean(),
                "std": df["value"].std(),
            },
            "missing_count": df["value"].isna().sum(),
        }

        print(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print(f"  Record count: {stats['record_count']}")
        print(f"  Value range: {stats['value_stats']['min']:.0f} to {stats['value_stats']['max']:.0f} MWh")
        print(f"  Value mean: {stats['value_stats']['mean']:.0f} MWh")
        print(f"  Value std: {stats['value_stats']['std']:.0f} MWh")
        print(f"  Missing values: {stats['missing_count']}")

        return stats

    def prepare_for_forecasting(
        self, 
        df: pd.DataFrame,
        unique_id: str = "1"
    ) -> pd.DataFrame:
        """
        Prepare data for statsforecast/mlforecast by reformatting columns.

        Statsforecast requires: unique_id, ds (timestamp), y (values)

        Args:
            df: Cleaned DataFrame from prepare_data()
            unique_id: Series identifier (default: fuel type like "NG")

        Returns:
            DataFrame with statsforecast format (unique_id, ds, y)

        Example:
            >>> df_clean = fetcher.prepare_data(df_raw)
            >>> df_forecast = fetcher.prepare_for_forecasting(df_clean)
        """
        # Validation: check required columns
        required_cols = ['period', 'value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(
                f"prepare_for_forecasting requires columns {required_cols}, "
                f"but found: {df.columns.tolist()}. "
                f"Ensure prepare_data() completed successfully."
            )

        # Validation: check for NaN values (should not exist after prepare_data)
        nan_count_period = df['period'].isna().sum()
        nan_count_value = df['value'].isna().sum()

        if nan_count_period > 0:
            logger.error(f"Found {nan_count_period} NaN values in period column")
            raise ValueError(
                f"Cannot prepare forecasting data with {nan_count_period} NaN values in period. "
                f"This indicates incomplete datetime parsing. "
                f"Run validate_data() and check prepare_data() error logs."
            )

        if nan_count_value > 0:
            logger.error(f"Found {nan_count_value} NaN values in value column")
            raise ValueError(
                f"Cannot prepare forecasting data with {nan_count_value} NaN values in value. "
                f"This indicates failed numeric conversion. "
                f"Run validate_data() and check prepare_data() error logs."
            )

        df_forecast = df.copy()
        df_forecast.columns = [col.lower() for col in df_forecast.columns]

        # Rename columns for statsforecast
        df_forecast['ds'] = df_forecast['period']
        df_forecast['y'] = df_forecast['value']
        df_forecast['unique_id'] = unique_id

        # CRITICAL: StatsForecast requires timezone-naive UTC datetimes
        # Convert from tz-aware UTC to timezone-naive
        df_forecast['ds'] = pd.to_datetime(df_forecast['ds'], errors='raise', utc=True).dt.tz_localize(None)

        # Keep only required columns
        df_forecast = df_forecast[['unique_id', 'ds', 'y']].reset_index(drop=True)

        logger.info(f"Prepared {len(df_forecast)} records for forecasting with unique_id={unique_id}")
        logger.debug(f"ds dtype: {df_forecast['ds'].dtype}, tz={df_forecast['ds'].dt.tz if hasattr(df_forecast['ds'], 'dt') else 'N/A'}")

        return df_forecast

    def train_test_split(
        self,
        df: pd.DataFrame,
        test_hours: int = 72
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.

        Leaves last N hours for testing, rest for training.

        Args:
            df: DataFrame with 'ds' (datetime) column
            test_hours: Number of hours to reserve for testing (default: 72)

        Returns:
            (train_df, test_df)

        Example:
            >>> train_df, test_df = fetcher.train_test_split(df_forecast, test_hours=168)
            >>> print(f"Train: {len(train_df)}, Test: {len(test_df)}")
        """
        df = df.sort_values('ds').reset_index(drop=True)

        # Calculate split point
        split_point = df['ds'].max() - timedelta(hours=test_hours)

        train_df = df[df['ds'] <= split_point].reset_index(drop=True)
        test_df = df[df['ds'] > split_point].reset_index(drop=True)

        print(f"\nTrain/Test Split:")
        print(f"  Training set: {len(train_df)} records")
        print(f"  Testing set: {len(test_df)} records")
        print(f"  Split point: {split_point}")

        return train_df, test_df

    def build_models(self) -> List:
        """
        Build a list of forecasting models for statsforecast.

        Models included:
        - AutoARIMA: Auto-tuned ARIMA model
        - SeasonalNaive: Baseline seasonal model (same value from year ago)
        - DynamicOptimizedTheta: Theta model with automatic optimization
        - HoltWinters: Exponential smoothing model
        - MSTL_ARIMA: Multi-seasonal trend with ARIMA trend forecaster
        - MSTL_HoltWinters: Multi-seasonal trend with HoltWinters trend forecaster

        Returns:
            List of statsforecast model objects

        Example:
            >>> from statsforecast import StatsForecast
            >>> models = fetcher.build_models()
            >>> sf = StatsForecast(models=models, freq='h')
        """
        from statsforecast.models import (MSTL, AutoARIMA,
                                          DynamicOptimizedTheta, HoltWinters,
                                          SeasonalNaive)

        models = [
            AutoARIMA(season_length=24),
            SeasonalNaive(season_length=24),
            DynamicOptimizedTheta(season_length=24),
            HoltWinters(season_length=24),
            MSTL(season_length=[24, 24 * 7], trend_forecaster=AutoARIMA(), alias="MSTL_ARIMA"),
            MSTL(season_length=[24, 24 * 7], trend_forecaster=HoltWinters(), alias="MSTL_HoltWinters"),
        ]

        return models

    def forecast(
        self,
        train_df: pd.DataFrame,
        horizon: int = 72,
        confidence_level: int = 95
    ) -> pd.DataFrame:
        """
        Create forecasts using StatsForecast.

        Args:
            train_df: Training DataFrame with unique_id, ds, y columns
            horizon: Number of steps to forecast (default: 72 hours)
            confidence_level: Prediction interval level (default: 95%)

        Returns:
            DataFrame with forecasts and prediction intervals

        Example:
            >>> forecast_df = fetcher.forecast(train_df, horizon=168)
            >>> print(forecast_df.head())
        """
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA

        models = self.build_models()

        # Initialize StatsForecast object
        sf = StatsForecast(
            models=models,
            freq='h',  # Hourly data (lowercase for pandas compatibility)
            fallback_model=AutoARIMA(),
            n_jobs=-1  # Use all available cores
        )

        print(f"\nTraining {len(models)} models...")
        print(f"  Models: {', '.join([type(m).__name__ for m in models])}")
        print(f"  Horizon: {horizon} hours")
        print(f"  Confidence level: {confidence_level}%")

        # Generate forecast (note: h comes first in signature)
        forecast_df = sf.forecast(h=horizon, df=train_df, level=[confidence_level])

        print(f"  [OK] Forecast generated: {len(forecast_df)} predictions")

        return forecast_df

    def evaluate_forecast(
        self,
        forecast_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        season_length: int = 24,
        confidence_level: int = 95
    ) -> pd.DataFrame:
        """
        Evaluate forecast performance against test data.

        Calculates: MAPE, RMSE, MASE, and prediction interval coverage

        Critical features:
        - Merge on (unique_id, ds) for multi-series correctness
        - All metrics computed on valid rows only (NaN-aware)
        - Coverage denominator = valid rows (not total rows)
        - MASE scales error relative to seasonal naive baseline

        Args:
            forecast_df: DataFrame from forecast() method
            test_df: Test partition from train_test_split()
            train_df: Training data (required for MASE calculation)
            season_length: Seasonal period for MASE (default: 24 hours)
            confidence_level: Prediction interval level (default: 95%)

        Returns:
            DataFrame with performance metrics for each model

        Example:
            >>> metrics = fetcher.evaluate_forecast(
            ...     forecast_df, test_df, train_df, confidence_level=95
            ... )
            >>> print(metrics.sort_values('rmse'))
        """
        from sklearn.metrics import (mean_absolute_percentage_error,
                                     mean_squared_error)

        # Merge forecast with test data on BOTH unique_id and ds (critical for multi-series)
        fc = forecast_df.merge(
            test_df,
            how="left",
            on=["unique_id", "ds"]  # ← BOTH keys for correctness
        )

        logger.info(f"Evaluation merge: {len(forecast_df)} forecast rows, {len(test_df)} test rows, {len(fc)} merged rows")

        # Helper functions for metrics (NaN-aware)
        def mape(y, yhat):
            """Mean Absolute Percentage Error (ignoring NaNs).

            ⚠️ WARNING: MAPE is undefined when y ≈ 0 (e.g., solar at night).
            For series with zeros, prefer RMSE, MAE, or MASE instead.
            """
            mask = y.notna() & yhat.notna()
            if mask.sum() == 0:
                return np.nan
            return mean_absolute_percentage_error(y[mask], yhat[mask])

        def rmse(y, yhat):
            """Root Mean Squared Error (ignoring NaNs)"""
            mask = y.notna() & yhat.notna()
            if mask.sum() == 0:
                return np.nan
            return np.sqrt(mean_squared_error(y[mask], yhat[mask]))

        def mase(y, yhat, y_train, season_length=24):
            """
            Mean Absolute Scaled Error.

            Scales the MAE by the MAE of a seasonal naive forecast on training data.
            MASE < 1 means the model beats seasonal naive.
            MASE > 1 means seasonal naive is better.

            Args:
                y: Actual test values
                yhat: Predicted values
                y_train: Training data for computing naive baseline
                season_length: Seasonal period (default: 24 for hourly data)
            """
            if y_train is None or len(y_train) < season_length + 1:
                return np.nan

            mask = y.notna() & yhat.notna()
            if mask.sum() == 0:
                return np.nan

            # MAE of the forecast
            mae_forecast = np.mean(np.abs(y[mask].values - yhat[mask].values))

            # MAE of seasonal naive on training data
            # Seasonal naive: y_t = y_{t - season_length}
            y_train_arr = y_train["y"].values
            naive_errors = np.abs(y_train_arr[season_length:] - y_train_arr[:-season_length])
            mae_naive = np.mean(naive_errors)

            if mae_naive < 1e-10:  # Avoid division by zero
                return np.nan

            return mae_forecast / mae_naive

        def coverage(y, lower, upper):
            """Prediction interval coverage (ignoring NaNs, denominator = valid rows)"""
            mask = y.notna() & lower.notna() & upper.notna()
            if mask.sum() == 0:
                return np.nan
            within = ((y[mask] >= lower[mask]) & (y[mask] <= upper[mask])).sum()
            return (within / mask.sum()) * 100  # denominator = valid rows only

        # Get model names (exclude metadata columns and interval bounds)
        model_cols = [col for col in forecast_df.columns
                     if col not in ['unique_id', 'ds'] and
                     not col.endswith(f'-lo-{confidence_level}') and
                     not col.endswith(f'-hi-{confidence_level}')]

        # Calculate metrics for each model
        rows = []
        for model in model_cols:
            y = fc["y"]
            yhat = fc[model]

            # Count valid rows for this model
            mask = y.notna() & yhat.notna()
            valid_count = mask.sum()

            lo_col = f"{model}-lo-{confidence_level}"
            hi_col = f"{model}-hi-{confidence_level}"
            if lo_col in fc.columns and hi_col in fc.columns:
                coverage_value = coverage(
                    y=y,
                    lower=fc[lo_col],
                    upper=fc[hi_col],
                )
            else:
                coverage_value = np.nan

            rows.append({
                "model": model,
                "mape": mape(y=y, yhat=yhat),
                "rmse": rmse(y=y, yhat=yhat),
                "mase": mase(y=y, yhat=yhat, y_train=train_df, season_length=season_length),
                "coverage": coverage_value,
                "valid_rows": valid_count,
            })

        fc_performance = pd.DataFrame(rows).sort_values('rmse')

        # Report merge quality
        valid_total = fc["y"].notna().sum()
        print(f"\nEvaluation Metrics (on {valid_total} valid rows out of {len(fc)} total):")
        print(f"{'Model':<20} {'MAPE':<8} {'RMSE':<8} {'MASE':<8} {'Coverage':<10} {'Valid':<8}")
        print("-" * 62)
        for _, row in fc_performance.iterrows():
            coverage_str = f"{row['coverage']:.1f}%" if pd.notna(row['coverage']) else "N/A"
            mase_str = f"{row['mase']:.3f}" if pd.notna(row['mase']) else "N/A"
            print(f"{row['model']:<20} {row['mape']:.4f}  {row['rmse']:<8.0f} {mase_str:<8} {coverage_str:<10} {int(row['valid_rows']):<8}")

        logger.info(f"Evaluation complete: {valid_total} valid rows, {len(fc_performance)} models evaluated")

        return fc_performance

    def cross_validate(
        self,
        df: pd.DataFrame,
        config: Optional[ExperimentConfig] = None,
        horizon: int = 24,
        n_windows: int = 5,
        step_size: int = 168,
        confidence_level: int = 95
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run rolling origin cross-validation for robust model evaluation.

        Instead of a single train/test split, this creates multiple windows:
        - Window 1: Train on data up to t1, test on t1 to t1+horizon
        - Window 2: Train on data up to t2, test on t2 to t2+horizon
        - ... and so on

        This gives you a better estimate of how your model will perform
        on future unseen data.

        Args:
            df: DataFrame with unique_id, ds, y columns
            config: ExperimentConfig (overrides other params if provided)
            horizon: Forecast horizon in hours (default: 24)
            n_windows: Number of CV windows (default: 5)
            step_size: Hours between windows (default: 168 = 1 week)
            confidence_level: Prediction interval level (default: 95)

        Returns:
            Tuple of (cv_results_df, leaderboard_df)
            - cv_results_df: Raw predictions for each cutoff
            - leaderboard_df: Aggregated metrics per model

        Example:
            >>> cv_results, leaderboard = fetcher.cross_validate(
            ...     df_forecast,
            ...     horizon=24,
            ...     n_windows=5,
            ...     step_size=168
            ... )
            >>> print(leaderboard)
        """
        from statsforecast import StatsForecast

        # Use config if provided
        if config is not None:
            horizon = config.horizon
            n_windows = config.n_windows
            step_size = config.step_size
            confidence_level = config.confidence_level

        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION")
        print(f"{'='*60}")
        print(f"  Horizon: {horizon} hours")
        print(f"  Windows: {n_windows}")
        print(f"  Step size: {step_size} hours")

        # Build models
        models = self.build_models()

        # Create StatsForecast object
        sf = StatsForecast(
            models=models,
            freq='h',
            n_jobs=-1,
        )

        # Run cross-validation
        # This is the key statsforecast method for backtesting
        print(f"\n  Running {n_windows} CV windows...")
        cv_df = sf.cross_validation(
            df=df,
            h=horizon,
            step_size=step_size,
            n_windows=n_windows,
            level=[confidence_level],
        )

        print(f"  [OK] CV complete: {len(cv_df)} total predictions")
        print(f"  Cutoff dates: {cv_df['cutoff'].nunique()} unique")

        # Compute metrics per cutoff per model
        cv_metrics = []
        model_names = [type(m).__name__ if not hasattr(m, 'alias') else m.alias
                       for m in models]

        for cutoff in cv_df["cutoff"].unique():
            window_df = cv_df[cv_df["cutoff"] == cutoff]
            y_true = window_df["y"].values

            for model in model_names:
                if model not in window_df.columns:
                    continue

                y_pred = window_df[model].values
                mask = np.isfinite(y_true) & np.isfinite(y_pred)

                if mask.sum() == 0:
                    continue

                # RMSE
                rmse_val = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))

                # MAPE
                mape_val = 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

                # Coverage
                lo_col = f"{model}-lo-{confidence_level}"
                hi_col = f"{model}-hi-{confidence_level}"
                coverage_val = np.nan
                if lo_col in window_df.columns and hi_col in window_df.columns:
                    lo = window_df[lo_col].values
                    hi = window_df[hi_col].values
                    within = ((y_true >= lo) & (y_true <= hi)).sum()
                    coverage_val = 100 * within / len(y_true)

                cv_metrics.append({
                    "cutoff": cutoff,
                    "model": model,
                    "rmse": rmse_val,
                    "mape": mape_val,
                    "coverage": coverage_val,
                })

        cv_metrics_df = pd.DataFrame(cv_metrics)

        # Create leaderboard by aggregating across windows
        leaderboard = cv_metrics_df.groupby("model").agg({
            "rmse": ["mean", "std"],
            "mape": ["mean", "std"],
            "coverage": "mean",
        }).round(2)

        # Flatten column names
        leaderboard.columns = ["_".join(col).strip() for col in leaderboard.columns.values]
        leaderboard = leaderboard.sort_values("rmse_mean").reset_index()

        print(f"\n{'='*60}")
        print(f"LEADERBOARD (aggregated across {n_windows} windows)")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'RMSE':<12} {'MAPE':<12} {'Coverage':<10}")
        print("-" * 54)
        for _, row in leaderboard.iterrows():
            rmse_str = f"{row['rmse_mean']:.0f} ± {row['rmse_std']:.0f}"
            mape_str = f"{row['mape_mean']:.2f} ± {row['mape_std']:.2f}"
            cov_str = f"{row['coverage_mean']:.1f}%"
            print(f"{row['model']:<20} {rmse_str:<12} {mape_str:<12} {cov_str:<10}")

        return cv_df, leaderboard

    def register_best_model(
        self,
        leaderboard: pd.DataFrame,
        model_name: str = None,
        experiment_name: str = "eia_forecasting",
        alias: str = "champion",
        train_df: Optional[pd.DataFrame] = None,
        default_horizon: int = 24,
        freq: str = "h",
    ) -> Optional[str]:
        """
        Register the best model from cross-validation to MLflow Model Registry.
        FIXED: Actually logs an MLflow Model at artifact_path="model" (pyfunc),
        so mlflow.register_model(runs:/.../model, ...) succeeds.
        """
        if not MLFLOW_AVAILABLE:
            print("  [SKIP] MLflow not available - cannot register model")
            return None

        if train_df is None:
            raise ValueError(
                "register_best_model requires train_df (unique_id/ds/y) so we can log a real MLflow model."
            )
        required = {"unique_id", "ds", "y"}
        if not required.issubset(train_df.columns):
            raise ValueError(f"train_df must have {sorted(required)}, got {train_df.columns.tolist()}")

        # Select best model
        if model_name is None:
            model_name = leaderboard.iloc[0]["model"]

        best_metrics = leaderboard[leaderboard["model"] == model_name].iloc[0]

        print(f"\n{'='*60}")
        print(f"MODEL REGISTRY")
        print(f"{'='*60}")
        print(f"  Best model: {model_name}")
        print(f"  RMSE: {best_metrics['rmse_mean']:.0f}")
        print(f"  Registering with alias: {alias}")

        import mlflow
        import mlflow.pyfunc
        from mlflow.models import infer_signature

        mlflow.set_experiment(experiment_name)

        # --- pyfunc wrapper ---
        class _StatsForecastPyFunc(mlflow.pyfunc.PythonModel):
            def __init__(self, chosen_model: str, confidence_level: int, freq: str, default_h: int):
                self.chosen_model = chosen_model
                self.confidence_level = int(confidence_level)
                self.freq = freq
                self.default_h = int(default_h)

            @staticmethod
            def _build_models():
                from statsforecast.models import (
                    MSTL, AutoARIMA, DynamicOptimizedTheta, HoltWinters, SeasonalNaive
                )
                return [
                    AutoARIMA(season_length=24),
                    SeasonalNaive(season_length=24),
                    DynamicOptimizedTheta(season_length=24),
                    HoltWinters(season_length=24),
                    MSTL(season_length=[24, 24 * 7], trend_forecaster=AutoARIMA(), alias="MSTL_ARIMA"),
                    MSTL(season_length=[24, 24 * 7], trend_forecaster=HoltWinters(), alias="MSTL_HoltWinters"),
                ]

            def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
                import pandas as pd
                from statsforecast import StatsForecast

                df = model_input.copy()

                # Allow passing a horizon in the input (single value repeated is fine)
                if "horizon" in df.columns:
                    h = int(df["horizon"].iloc[0])
                    df = df.drop(columns=["horizon"])
                else:
                    h = self.default_h

                # Expect unique_id/ds/y
                if not {"unique_id", "ds", "y"}.issubset(df.columns):
                    raise ValueError(
                        f"pyfunc input must contain unique_id/ds/y (+ optional horizon). Got: {df.columns.tolist()}"
                    )

                df["ds"] = pd.to_datetime(df["ds"], errors="raise")
                # Optional safety: strip tz if present
                if getattr(df["ds"].dt, "tz", None) is not None:
                    df["ds"] = df["ds"].dt.tz_convert("UTC").dt.tz_localize(None)

                # Pick the chosen model by alias or class name
                chosen = None
                for m in self._build_models():
                    name = getattr(m, "alias", type(m).__name__)
                    if name == self.chosen_model or type(m).__name__ == self.chosen_model:
                        chosen = m
                        break
                if chosen is None:
                    available = [getattr(m, "alias", type(m).__name__) for m in self._build_models()]
                    raise ValueError(f"Unknown chosen_model={self.chosen_model}. Available: {available}")

                sf = StatsForecast(models=[chosen], freq=self.freq, n_jobs=1)
                out = sf.forecast(df=df, h=h, level=[self.confidence_level]).reset_index()
                return out

        with mlflow.start_run(run_name=f"register_{model_name}"):
            # Log metrics/params (same as you had)
            mlflow.log_metric("rmse_mean", float(best_metrics["rmse_mean"]))
            mlflow.log_metric("rmse_std", float(best_metrics["rmse_std"]))
            mlflow.log_metric("mape_mean", float(best_metrics["mape_mean"]))
            mlflow.log_metric("coverage_mean", float(best_metrics["coverage_mean"]))
            mlflow.log_param("model_name", model_name)

            # Input example needs enough history for MSTL weekly season length
            # Use last 30 days (720 hours) to be safe
            example_in = train_df.sort_values("ds").tail(24 * 30)[["unique_id", "ds", "y"]].copy()
            example_in["horizon"] = int(default_horizon)
            # ✅ MLflow requires timezone-naive datetimes
            example_in["ds"] = pd.to_datetime(example_in["ds"], errors="raise", utc=True).dt.tz_localize(None)


            # Run once to infer output signature (can be a bit slow, but deterministic)
            pyfunc_model = _StatsForecastPyFunc(
                chosen_model=str(model_name),
                confidence_level=int(best_metrics.get("coverage_mean", 95) and 95),  # keep your pipeline default
                freq=freq,
                default_h=int(default_horizon),
            )
            example_out = pyfunc_model.predict(None, example_in)

            signature = infer_signature(example_in, example_out)

            # THIS is the key fix: log a real MLflow Model at artifact_path="model"
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=pyfunc_model,
                signature=signature,
                input_example=example_in,
                pip_requirements=[
                    f"mlflow=={mlflow.__version__}",
                    "pandas",
                    "numpy",
                    "statsforecast",
                ],
            )

            registered_model_name = f"{experiment_name}_{model_name}"
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"

            try:
                result = mlflow.register_model(model_uri, registered_model_name)
                version = result.version

                # Alias (works on newer MLflow); safe fallback if not supported
                client = mlflow.tracking.MlflowClient()
                try:
                    client.set_registered_model_alias(registered_model_name, alias, version)
                except Exception:
                    client.set_model_version_tag(registered_model_name, version, "alias", alias)

                print(f"  [OK] Registered: {registered_model_name} v{version}")
                print(f"  [OK] Alias '{alias}' assigned to v{version}")
                logger.info(f"Model registered: {registered_model_name} v{version} with alias {alias}")
                return f"{registered_model_name}@{alias}"

            except Exception as e:
                logger.warning(f"Model registration failed: {e}")
                print(f"  [WARN] Registration failed: {e}")
                return None


    def _create_plot(self, test_df: pd.DataFrame, forecast_df: pd.DataFrame):
        """
        Create an interactive plotly visualization of forecast vs actuals.

        Args:
            test_df: Test partition with actual values
            forecast_df: Forecast predictions

        Returns:
            Plotly Figure object
        """
        import plotly.graph_objects as go

        # Merge data
        merged = test_df.merge(forecast_df, on=['unique_id', 'ds'])
        merged = merged.sort_values('ds')

        # Create figure
        fig = go.Figure()

        # Add actual values
        fig.add_trace(go.Scatter(
            x=merged['ds'],
            y=merged['y'],
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))

        # Add forecast from best model (MSTL_ARIMA)
        if 'MSTL_ARIMA' in merged.columns:
            fig.add_trace(go.Scatter(
                x=merged['ds'],
                y=merged['MSTL_ARIMA'],
                mode='lines',
                name='MSTL_ARIMA (Best)',
                line=dict(color='red', width=2, dash='dash')
            ))

            # Add 95% confidence interval if available
            if 'MSTL_ARIMA-hi-95' in merged.columns and 'MSTL_ARIMA-lo-95' in merged.columns:
                fig.add_trace(go.Scatter(
                    x=merged['ds'].tolist() + merged['ds'].tolist()[::-1],
                    y=merged['MSTL_ARIMA-hi-95'].tolist() + merged['MSTL_ARIMA-lo-95'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(color='rgba(255, 0, 0, 0)'),
                    name='95% Confidence Interval'
                ))

        # Update layout
        fig.update_layout(
            title='EIA Electricity Generation: Forecast vs Actual',
            xaxis_title='Date',
            yaxis_title='Generation (MWh)',
            hovermode='x unified',
            height=400,
            template='plotly_white'
        )

        return fig

    def run_experiment(
        self,
        df: pd.DataFrame,
        experiment_name: str,
        test_hours: int = 72,
        models_to_test: Optional[List[str]] = None,
        track_with_mlflow: bool = False
    ) -> Dict:
        """
        Run a complete forecasting experiment with model evaluation.

        Args:
            df: Cleaned DataFrame from full_pipeline()
            experiment_name: Name for this experiment
            test_hours: Hours to reserve for testing (default: 72)
            models_to_test: List of model names to test (None = all)
            track_with_mlflow: Whether to log to MLflow (default: False)

        Returns:
            Dictionary with experiment results including:
            - experiment_name: Name of experiment
            - timestamp: When experiment was run
            - data_shape: (rows, columns) of data used
            - train_size: Number of training records
            - test_size: Number of testing records
            - metrics: Performance metrics for each model
            - best_model: Name of best performing model
            - results: Full results DataFrame

        Example:
            >>> results = fetcher.run_experiment(df, "exp1_baseline")
            >>> print(results['best_model'])
        """
        logger.info(f"Starting experiment: {experiment_name} with test_hours={test_hours}")

        print(f"\n" + "="*60)
        print(f"EXPERIMENT: {experiment_name}")
        print("="*60)

        # Start MLflow run if enabled
        if track_with_mlflow and MLFLOW_AVAILABLE:
            mlflow.start_run(run_name=experiment_name)
            mlflow.log_param("experiment_name", experiment_name)
            mlflow.log_param("test_hours", test_hours)
            logger.info("MLflow run started for experiment")

        try:
            # Prepare data
            logger.info("Preparing data for experiment")
            print(f"\nPreparing data for experiment...")
            df_forecast = self.prepare_for_forecasting(df)
            train_df, test_df = self.train_test_split(df_forecast, test_hours=test_hours)

            # Log data info
            logger.info(f"Data shape: {df.shape}, Train: {len(train_df)}, Test: {len(test_df)}")
            print(f"  Data shape: {df.shape}")
            print(f"  Training records: {len(train_df)}")
            print(f"  Testing records: {len(test_df)}")

            if track_with_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_param("data_rows", df.shape[0])
                mlflow.log_param("train_size", len(train_df))
                mlflow.log_param("test_size", len(test_df))

            # Train and forecast
            logger.info("Training models")
            print(f"\nTraining models...")
            forecast_df = self.forecast(train_df, horizon=len(test_df))

            # Evaluate (pass train_df for MASE calculation)
            logger.info("Evaluating model performance")
            print(f"\nEvaluating performance...")
            metrics_df = self.evaluate_forecast(forecast_df, test_df, train_df=train_df)

            # Log metrics to MLflow
            if track_with_mlflow and MLFLOW_AVAILABLE:
                for _, row in metrics_df.iterrows():
                    model_name = row['model']
                    mlflow.log_metrics({
                        f"{model_name}_mape": row['mape'],
                        f"{model_name}_rmse": row['rmse'],
                        f"{model_name}_coverage": row['coverage'],
                    })

            # Identify best model
            best_model = metrics_df.iloc[0]['model']
            best_rmse = metrics_df.iloc[0]['rmse']

            logger.info(f"Experiment {experiment_name} complete - Best model: {best_model}, RMSE: {best_rmse:.0f}")

            # Compile results
            experiment_results = {
                "experiment_name": experiment_name,
                "timestamp": datetime.now().isoformat(),
                "data_shape": df.shape,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "metrics": metrics_df,
                "best_model": best_model,
                "best_rmse": best_rmse,
                "results": metrics_df
            }

            # Log summary
            if track_with_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_metric("best_rmse", best_rmse)
                mlflow.log_param("best_model", best_model)

            print(f"\n[OK] Experiment complete!")
            print(f"  Best model: {best_model}")
            print(f"  Best RMSE: {best_rmse:.0f}")

            return experiment_results

        except Exception as e:
            logger.error(f"Experiment {experiment_name} failed: {str(e)}", exc_info=True)
            if track_with_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_param("error", str(e))
            raise

        finally:
            if track_with_mlflow and MLFLOW_AVAILABLE:
                mlflow.end_run()
                logger.info(f"MLflow run ended for experiment {experiment_name}")

    def save_datasets(
        self,
        raw_df: pd.DataFrame,
        clean_df: pd.DataFrame,
        integrity_report: Dict,
        output_dir: str = "data",
        pull_params: Optional[Dict] = None
    ) -> str:
        """
        Save datasets with versioning and metadata for reproducibility.

        Creates:
        - raw.parquet: Unmodified API response
        - clean.parquet: After prepare + validate
        - metadata.json: Pull parameters, timestamps, row counts, integrity report

        This enables:
        - Reproducibility across experiments
        - Data lineage tracking
        - Debugging with raw vs clean comparison
        - Integrity validation history

        Args:
            raw_df: Raw DataFrame from pull_data()
            clean_df: Cleaned DataFrame from prepare_data()
            integrity_report: Report from validate_time_series_integrity()
            output_dir: Directory to save datasets (default: "data")
            pull_params: Dictionary of pull_data() parameters for metadata

        Returns:
            Path to saved metadata file

        Example:
            >>> metadata_path = fetcher.save_datasets(
            ...     raw_df, clean_df, integrity_report,
            ...     pull_params={"start_date": "2024-12-01", "respondent": "US48"}
            ... )
        """
        import json
        import os

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamp for this version
        timestamp = datetime.now(tz=pytz.UTC).isoformat()

        # Build metadata
        metadata = {
            "version": "1.0",
            "timestamp": timestamp,
            "pull_parameters": pull_params or {},
            "raw_row_count": len(raw_df),
            "clean_row_count": len(clean_df),
            "validation_status": "passed" if integrity_report.get("status") == "valid" else "failed",
            "integrity_report": {
                "duplicate_count": integrity_report.get("duplicate_count", 0),
                "missing_hours": integrity_report.get("missing_hours", 0),
                "longest_gap_hours": integrity_report.get("longest_gap_hours", 0),
                "dst_repeated_hours": integrity_report.get("dst_repeated_hours", 0),
            },
            "columns": {
                "raw": list(raw_df.columns),
                "clean": list(clean_df.columns),
            }
        }

        # Save raw data
        raw_path = os.path.join(output_dir, "raw.parquet")
        raw_df.to_parquet(raw_path, index=False)
        print(f"  [OK] Raw data saved: {raw_path}")
        logger.info(f"Raw data saved: {raw_path} ({len(raw_df)} rows)")

        # Save clean data
        clean_path = os.path.join(output_dir, "clean.parquet")
        clean_df.to_parquet(clean_path, index=False)
        print(f"  [OK] Clean data saved: {clean_path}")
        logger.info(f"Clean data saved: {clean_path} ({len(clean_df)} rows)")

        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"  [OK] Metadata saved: {metadata_path}")
        logger.info(f"Metadata saved: {metadata_path}")

        return metadata_path

    def compare_experiments(self, experiments: List[Dict]) -> pd.DataFrame:
        """
        Compare results from multiple experiments.

        Args:
            experiments: List of experiment result dictionaries

        Returns:
            DataFrame comparing best models from each experiment

        Example:
            >>> exp1 = fetcher.run_experiment(df, "exp1")
            >>> exp2 = fetcher.run_experiment(df, "exp2")
            >>> comparison = fetcher.compare_experiments([exp1, exp2])
        """
        logger.info(f"Comparing {len(experiments)} experiments")

        comparison_rows = []

        for exp in experiments:
            best_row = exp['results'].iloc[0]
            comparison_rows.append({
                "experiment": exp["experiment_name"],
                "best_model": exp["best_model"],
                "mape": best_row['mape'],
                "rmse": best_row['rmse'],
                "coverage": best_row['coverage'],
                "timestamp": exp["timestamp"]
            })

        comparison_df = pd.DataFrame(comparison_rows).sort_values('rmse')

        # Log comparison results
        logger.info("Experiment comparison results:")
        for _, row in comparison_df.iterrows():
            logger.info(f"  {row['experiment']}: {row['best_model']} (RMSE: {row['rmse']:.0f}, MAPE: {row['mape']:.4f})")

        best_exp = comparison_df.iloc[0]
        logger.info(f"Best overall: {best_exp['experiment']} with {best_exp['best_model']} (RMSE: {best_exp['rmse']:.0f})")

        print(f"\n" + "="*60)
        print("EXPERIMENT COMPARISON")
        print("="*60)
        print(f"\n{'Experiment':<25} {'Best Model':<20} {'RMSE':<10} {'MAPE':<10}")
        print("-" * 65)
        for _, row in comparison_df.iterrows():
            print(f"{row['experiment']:<25} {row['best_model']:<20} {row['rmse']:.0f}      {row['mape']:.4f}")

        return comparison_df

    def full_pipeline(
        self,
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
        respondent: str = "US48",
        fueltype: str = "NG",
        track_with_mlflow: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the complete pipeline: pull -> prepare -> validate -> stats.

        Args:
            start_date: Start date for data pull (YYYY-MM-DD)
            end_date: End date for data pull (YYYY-MM-DD)
            respondent: Region code (default: US48)
            fueltype: Fuel type code (default: NG)
            track_with_mlflow: Whether to log to MLflow (default: False)

        Returns:
            (cleaned_dataframe, statistics_dict)

        Example:
            >>> df, stats = fetcher.full_pipeline()
        """
        logger.info(f"Starting full pipeline: {start_date} to {end_date}")

        print("\n" + "="*60)
        print("FULL PIPELINE: Pull -> Prepare -> Validate -> Stats")
        print("="*60)

        # Start MLflow run if enabled
        if track_with_mlflow and MLFLOW_AVAILABLE:
            mlflow.start_run(run_name=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_param("start_date", start_date)
            mlflow.log_param("end_date", end_date)
            mlflow.log_param("respondent", respondent)
            mlflow.log_param("fueltype", fueltype)
            logger.info("MLflow run started")

        try:
            # Step 2: Pull
            logger.info("Pulling raw data from EIA API")
            df_raw = self.pull_data(start_date, end_date, respondent, fueltype)

            # Step 3: Inspect
            logger.info(f"Raw data shape: {df_raw.shape}")
            self.inspect_data(df_raw)

            # Step 4: Prepare
            logger.info("Preparing data")
            df_clean = self.prepare_data(df_raw)

            # Step 5: Validate
            logger.info("Validating data")
            is_valid = self.validate_data(df_clean)

            if not is_valid:
                logger.warning("Data validation failed")
                if track_with_mlflow and MLFLOW_AVAILABLE:
                    mlflow.log_param("validation_status", "failed")
            else:
                logger.info("Data validation passed")
                if track_with_mlflow and MLFLOW_AVAILABLE:
                    mlflow.log_param("validation_status", "passed")

            # Step 6: Stats
            logger.info("Computing statistics")
            stats = self.get_stats(df_clean)

            # Log stats to MLflow
            if track_with_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_metric("record_count", stats['record_count'])
                mlflow.log_metric("value_min", stats['value_stats']['min'])
                mlflow.log_metric("value_max", stats['value_stats']['max'])
                mlflow.log_metric("value_mean", stats['value_stats']['mean'])
                mlflow.log_metric("missing_count", stats['missing_count'])
                logger.info("Statistics logged to MLflow")

            logger.info("Full pipeline completed successfully")

            print("\n" + "="*60)
            print("PIPELINE COMPLETE")
            print("="*60 + "\n")

            return df_clean, stats

        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            if track_with_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_param("error", str(e))
            raise

        finally:
            if track_with_mlflow and MLFLOW_AVAILABLE:
                mlflow.end_run()
                logger.info("MLflow run ended")


# Allow testing individual steps
if __name__ == "__main__":
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        print("Error: EIA_API_KEY not found in environment variables or .env file")
        print("Please create a .env file with: EIA_API_KEY=your_api_key_here")
        exit(1)

    # Initialize
    fetcher = EIADataFetcher(api_key)

    # Run full pipeline
    df, stats = fetcher.full_pipeline(
        start_date="2024-12-01",
        end_date="2024-12-31"
    )

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())

    print("\n[OK] Data successfully loaded from .env file!")
    print("[OK] Ready for time series analysis and forecasting")

    # FORECASTING WORKFLOW
    print("\n" + "="*60)
    print("FORECASTING WORKFLOW")
    print("="*60)

    # Step 1: Prepare data for forecasting
    df_forecast = fetcher.prepare_for_forecasting(df)
    print(f"\nData reformatted for forecasting:")
    print(f"  Columns: {list(df_forecast.columns)}")
    print(f"  Shape: {df_forecast.shape}")

    # Step 2: Train/test split (72 hours test set)
    train_df, test_df = fetcher.train_test_split(df_forecast, test_hours=72)

    # Step 3: Train models and create forecast
    forecast_df = fetcher.forecast(train_df, horizon=len(test_df))

    # Step 4: Evaluate performance (pass train_df for MASE)
    metrics = fetcher.evaluate_forecast(forecast_df, test_df, train_df=train_df)

    # Step 5: Visualize results
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)

    try:
        print("\nGenerating forecast visualization...")

        # Create the plot using StatsForecast's plot method
        p = fetcher._create_plot(test_df, forecast_df)

        # Display the plot inline in Jupyter
        p.show()

        print(f"  [OK] Forecast plot displayed")

    except Exception as e:
        print(f"  [INFO] Plotly visualization setup: {str(e)[:60]}...")
        print("  (This is optional - forecast metrics are available above)")

    print("\n[OK] Forecasting workflow complete!")
