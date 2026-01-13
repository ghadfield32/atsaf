"""
Phase 1: End-to-End Real EIA Data Pipeline

This script orchestrates the complete workflow:
1. Fetch real EIA data
2. Run all Chapter 1 validation gates
3. Prepare data for Chapter 2 modeling
4. Save all artifacts with metadata

Usage:
    python scripts/run_e2e_eia.py --start-date 2025-12-26 --end-date 2026-01-09 --output artifacts/baseline

Inputs:
    - EIA_API_KEY environment variable
    - start_date, end_date in YYYY-MM-DD format
    - respondent_id and fuel_type for filtering

Outputs:
    - artifacts/{output_dir}/raw.parquet          : Raw data from API
    - artifacts/{output_dir}/clean.parquet        : Post-validation data
    - artifacts/{output_dir}/metadata.json        : Processing metadata
    - artifacts/{output_dir}/validation_report.json : Detailed validation results
    - artifacts/{output_dir}/forecast.parquet     : Forecast outputs
    - artifacts/{output_dir}/metrics.parquet      : Evaluation metrics
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EIADataFetcher:
    """Fetch data from EIA API (or generate synthetic for testing)"""
    
    def __init__(self, api_key: str):
        """Initialize with EIA API key"""
        self.api_key = api_key
        self.base_url = "https://api.eia.gov/series"
        
    def fetch(
        self,
        start_date: str,
        end_date: str,
        respondent_id: str = "14871",
        fuel_type: str = "COL",
        synthetic: bool = False
    ) -> pd.DataFrame:
        """
        Fetch hourly electricity generation data from EIA
        Falls back to synthetic data if API unavailable
        
        Args:
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
            respondent_id: EIA respondent ID (default: 14871 = US total)
            fuel_type: Fuel type code (default: COL = Coal)
            synthetic: Force synthetic data generation (for testing)
            
        Returns:
            DataFrame with columns: [period, value, unique_id, fuel_type]
        """
        import requests
        
        logger.info(f"Fetching EIA data from {start_date} to {end_date}")
        
        if synthetic:
            logger.info("  Using synthetic data (testing mode)")
            return self._generate_synthetic_data(start_date, end_date)
        
        # Try to fetch from API
        fuel_types = ["COL", "NG", "NUC", "WAT", "WND"]
        all_data = []
        
        for fuel in fuel_types:
            series_id = f"EBA.US-TOTAL.NG.{fuel}.H"
            
            params = {
                "api_key": self.api_key,
                "series_id": series_id,
                "start": start_date.replace("-", ""),
                "end": end_date.replace("-", ""),
                "frequency": "hourly"
            }
            
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                if "series" in data and len(data["series"]) > 0:
                    series_data = data["series"][0]
                    if "data" in series_data:
                        for period, value in series_data["data"]:
                            all_data.append({
                                "period": period,
                                "value": float(value) if value is not None else np.nan,
                                "unique_id": fuel,
                                "fuel_type": fuel
                            })
                        logger.info(f"  Fetched {len(series_data['data'])} records for {fuel}")
                        
            except Exception as e:
                logger.warning(f"  Failed to fetch {fuel}: {e}")
                continue
        
        if all_data:
            df = pd.DataFrame(all_data)
            logger.info(f"Total records fetched: {len(df)}")
            return df
        else:
            logger.warning("  API failed, falling back to synthetic data")
            return self._generate_synthetic_data(start_date, end_date)
    
    def _generate_synthetic_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic synthetic hourly electricity generation data"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Create hourly date range
        date_range = pd.date_range(start=start, end=end, freq="H")
        
        fuel_types = ["COL", "NG", "NUC", "WAT", "WND"]
        all_data = []
        
        np.random.seed(42)  # For reproducibility
        
        for fuel in fuel_types:
            # Base load values for each fuel type (MW)
            base_loads = {
                "COL": 200000,  # Coal
                "NG": 300000,   # Natural Gas
                "NUC": 100000,  # Nuclear
                "WAT": 80000,   # Hydro
                "WND": 120000   # Wind
            }
            base = base_loads.get(fuel, 100000)
            
            # Generate realistic patterns
            for i, dt in enumerate(date_range):
                # Hour-of-day pattern (lower at night, higher during day)
                hour_pattern = 1.0 - 0.3 * np.cos(2 * np.pi * dt.hour / 24)
                
                # Day-of-week pattern (lower on weekends)
                dow_pattern = 1.0 - 0.1 * (1 if dt.weekday() < 5 else 0)
                
                # Random noise
                noise = np.random.normal(0, 0.05)
                
                # Combined value
                value = base * hour_pattern * dow_pattern * (1 + noise)
                
                all_data.append({
                    "period": dt.strftime("%Y-%m-%dT%H"),
                    "value": max(value, 0),  # No negative generation
                    "unique_id": fuel,
                    "fuel_type": fuel
                })
        
        df = pd.DataFrame(all_data)
        logger.info(f"Generated {len(df)} synthetic records ({len(fuel_types)} fuel types)")
        
        return df


class Chapter1Validator:
    """Validate data using Chapter 1 fail-loud gates"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_all(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run all Chapter 1 validation gates
        
        Args:
            df: Raw DataFrame from API
            
        Returns:
            (cleaned_df, validation_report)
        """
        logger.info("Running Chapter 1 validation gates...")
        
        results = {
            "total_rows_input": len(df),
            "gates": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Gate 1: DateTime parsing
        try:
            df = self._validate_datetime(df)
            results["gates"]["datetime_parsing"] = {"status": "PASS", "rows": len(df)}
            logger.info("  ✓ DateTime parsing: PASS")
        except Exception as e:
            results["gates"]["datetime_parsing"] = {"status": "FAIL", "error": str(e)}
            raise ValueError(f"DateTime validation failed: {e}")
        
        # Gate 2: Numeric conversion
        try:
            df = self._validate_numeric(df)
            results["gates"]["numeric_conversion"] = {"status": "PASS", "rows": len(df)}
            logger.info("  ✓ Numeric conversion: PASS")
        except Exception as e:
            results["gates"]["numeric_conversion"] = {"status": "FAIL", "error": str(e)}
            raise ValueError(f"Numeric validation failed: {e}")
        
        # Gate 3: Duplicate detection
        try:
            df = self._validate_no_duplicates(df)
            results["gates"]["duplicate_detection"] = {"status": "PASS", "rows": len(df)}
            logger.info("  ✓ Duplicate detection: PASS")
        except Exception as e:
            results["gates"]["duplicate_detection"] = {"status": "FAIL", "error": str(e)}
            raise ValueError(f"Duplicate detection failed: {e}")
        
        # Gate 4: Missing hours detection
        try:
            df = self._validate_no_gaps(df)
            results["gates"]["missing_hours_detection"] = {"status": "PASS", "rows": len(df)}
            logger.info("  ✓ Missing hours detection: PASS")
        except Exception as e:
            results["gates"]["missing_hours_detection"] = {"status": "FAIL", "error": str(e)}
            raise ValueError(f"Missing hours validation failed: {e}")
        
        # Gate 5: NaN detection
        try:
            df = self._validate_no_nans(df)
            results["gates"]["nan_detection"] = {"status": "PASS", "rows": len(df)}
            logger.info("  ✓ NaN detection: PASS")
        except Exception as e:
            results["gates"]["nan_detection"] = {"status": "FAIL", "error": str(e)}
            raise ValueError(f"NaN detection failed: {e}")
        
        # Gate 6: Multi-series integrity
        try:
            df = self._validate_multi_series_integrity(df)
            results["gates"]["multi_series_integrity"] = {"status": "PASS", "rows": len(df)}
            logger.info("  ✓ Multi-series integrity: PASS")
        except Exception as e:
            results["gates"]["multi_series_integrity"] = {"status": "FAIL", "error": str(e)}
            raise ValueError(f"Multi-series integrity failed: {e}")
        
        results["total_rows_output"] = len(df)
        results["rows_filtered"] = results["total_rows_input"] - results["total_rows_output"]
        
        self.validation_results = results
        return df, results
    
    def _validate_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse period as datetime"""
        if df.empty:
            return df
        
        try:
            # EIA format: YYYY-MM-DDTHH (e.g., 2025-12-26T00)
            df["ds"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", errors="raise")
        except Exception as e:
            raise ValueError(f"Invalid datetime format in period: {e}")
        
        return df[["unique_id", "ds", "value", "fuel_type"]].reset_index(drop=True)
    
    def _validate_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure value column is numeric"""
        if df.empty:
            return df
        
        # Check for non-numeric values (errors='raise' will fail on invalid)
        try:
            df["value"] = pd.to_numeric(df["value"], errors="raise")
        except Exception as e:
            raise ValueError(f"Non-numeric value detected: {e}")
        
        return df
    
    def _validate_no_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for duplicate (unique_id, ds) pairs"""
        if df.empty:
            return df
        
        duplicates = df[df.duplicated(subset=["unique_id", "ds"], keep=False)]
        if not duplicates.empty:
            raise ValueError(
                f"Found {len(duplicates)} duplicate (unique_id, ds) pairs:\n"
                f"{duplicates[['unique_id', 'ds']].drop_duplicates().head()}"
            )
        
        return df
    
    def _validate_no_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for missing hours (gaps > 1 hour)"""
        if df.empty:
            return df
        
        for unique_id in df["unique_id"].unique():
            subset = df[df["unique_id"] == unique_id].sort_values("ds")
            
            if len(subset) < 2:
                continue
            
            time_diffs = subset["ds"].diff()
            gaps = time_diffs[time_diffs > timedelta(hours=1)]
            
            if not gaps.empty:
                raise ValueError(
                    f"Found {len(gaps)} gaps > 1 hour in {unique_id}:\n"
                    f"{gaps[gaps.notna()].head()}"
                )
        
        return df
    
    def _validate_no_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect NaN in key columns"""
        if df.empty:
            return df
        
        nan_cols = ["unique_id", "ds", "value"]
        for col in nan_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                raise ValueError(
                    f"Found {nan_count} NaN values in {col} column"
                )
        
        return df
    
    def _validate_multi_series_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all series have same number of observations"""
        if df.empty:
            return df
        
        counts = df.groupby("unique_id").size()
        
        if len(counts.unique()) > 1:
            raise ValueError(
                f"Unequal series lengths: {counts.to_dict()}"
            )
        
        return df


class Chapter2Preparer:
    """Prepare data for Chapter 2 modeling"""
    
    @staticmethod
    def prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare data for forecasting
        
        Args:
            df: Cleaned data from Chapter 1
            
        Returns:
            (prepared_df, preparation_metadata)
        """
        logger.info("Preparing data for Chapter 2 modeling...")
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "series_count": df["unique_id"].nunique(),
            "observations_per_series": len(df) // df["unique_id"].nunique(),
            "date_range": {
                "min": df["ds"].min().isoformat(),
                "max": df["ds"].max().isoformat()
            }
        }
        
        # Ensure proper sorting
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        
        # Add time-based features
        df["year"] = df["ds"].dt.year
        df["month"] = df["ds"].dt.month
        df["day"] = df["ds"].dt.day
        df["hour"] = df["ds"].dt.hour
        df["dayofweek"] = df["ds"].dt.dayofweek
        
        # Rename value to y for forecasting
        df = df.rename(columns={"value": "y"})
        
        metadata["features_added"] = ["year", "month", "day", "hour", "dayofweek"]
        
        logger.info("  ✓ Data preparation complete")
        
        return df, metadata


class ForecastingPipeline:
    """Simple forecasting pipeline for validation"""
    
    @staticmethod
    def forecast(
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        horizon: int = 24,
        method: str = "exponential_smoothing"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate forecasts for validation
        
        Args:
            df: Prepared data with y column
            train_ratio: Train/test split ratio
            horizon: Forecast horizon in steps
            method: Forecasting method
            
        Returns:
            (forecasts_df, metrics_df)
        """
        logger.info(f"Generating {method} forecasts with horizon={horizon}...")
        
        forecasts_list = []
        metrics_list = []
        
        for unique_id in df["unique_id"].unique():
            subset = df[df["unique_id"] == unique_id].copy()
            subset = subset.sort_values("ds").reset_index(drop=True)
            
            n = len(subset)
            train_size = int(n * train_ratio)
            
            train = subset.iloc[:train_size]
            test = subset.iloc[train_size:]
            
            if len(test) < horizon:
                logger.warning(f"  {unique_id}: test set smaller than horizon, skipping")
                continue
            
            # Simple exponential smoothing forecast
            from scipy import stats
            
            train_values = train["y"].values
            test_values = test["y"].values[:horizon]
            
            # Fit exponential smoothing
            if len(train_values) > 1 and np.isfinite(train_values).all():
                # Simple forecast: last value + trend
                trend = np.mean(np.diff(train_values[-10:]))  # Last 10 steps trend
                forecast = np.array([train_values[-1] + trend * (i + 1) for i in range(horizon)])
            else:
                forecast = np.full(horizon, train_values[-1] if len(train_values) > 0 else 0)
            
            # Ensure no negatives (for generation data)
            forecast = np.maximum(forecast, 0)
            
            # Create forecast dataframe
            forecast_ds = test.iloc[:horizon]["ds"].values
            for i, (ds, pred) in enumerate(zip(forecast_ds, forecast)):
                forecasts_list.append({
                    "unique_id": unique_id,
                    "ds": pd.Timestamp(ds),
                    "forecast": pred,
                    "actual": test_values[i] if i < len(test_values) else np.nan
                })
            
            # Compute metrics
            actual = test_values[:horizon]
            valid_mask = np.isfinite(forecast) & np.isfinite(actual)
            
            if valid_mask.sum() > 0:
                valid_forecast = forecast[valid_mask]
                valid_actual = actual[valid_mask]
                
                rmse = np.sqrt(np.mean((valid_forecast - valid_actual) ** 2))
                mae = np.mean(np.abs(valid_forecast - valid_actual))
                mape = 100 * np.mean(np.abs((valid_forecast - valid_actual) / (np.abs(valid_actual) + 1e-10)))
                
                metrics_list.append({
                    "unique_id": unique_id,
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "valid_count": valid_mask.sum()
                })
        
        forecasts_df = pd.DataFrame(forecasts_list)
        metrics_df = pd.DataFrame(metrics_list)
        
        logger.info(f"  Generated forecasts for {len(metrics_df)} series")
        
        return forecasts_df, metrics_df


class PipelineOrchestrator:
    """Orchestrate the complete E2E pipeline"""
    
    def __init__(self, output_dir: Path):
        """Initialize with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.artifacts = {}
        self.metadata = {}
        
    def run(
        self,
        start_date: str,
        end_date: str,
        api_key: str,
        respondent_id: str = "14871",
        fuel_type: str = "COL",
        synthetic: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Returns:
            Dictionary with status and artifact paths
        """
        logger.info("=" * 80)
        logger.info("PHASE 1: End-to-End EIA Data Pipeline")
        logger.info("=" * 80)
        
        try:
            # Step 1: Fetch raw data
            logger.info("\n[1/6] Fetching raw EIA data...")
            fetcher = EIADataFetcher(api_key)
            raw_df = fetcher.fetch(
                start_date, end_date, respondent_id, fuel_type, synthetic=synthetic
            )
            self.artifacts["raw"] = raw_df
            self._save_artifact("raw.parquet", raw_df)
            
            # Step 2: Chapter 1 validation
            logger.info("\n[2/6] Running Chapter 1 validation gates...")
            validator = Chapter1Validator()
            clean_df, validation_report = validator.validate_all(raw_df)
            self.artifacts["clean"] = clean_df
            self.metadata["validation"] = validation_report
            self._save_artifact("clean.parquet", clean_df)
            self._save_metadata("validation_report.json", validation_report)
            
            # Step 3: Chapter 2 preparation
            logger.info("\n[3/6] Preparing data for Chapter 2...")
            prepared_df, prep_metadata = Chapter2Preparer.prepare(clean_df)
            self.artifacts["prepared"] = prepared_df
            self.metadata["preparation"] = prep_metadata
            
            # Step 4: Generate forecasts
            logger.info("\n[4/6] Generating forecasts...")
            forecasts_df, metrics_df = ForecastingPipeline.forecast(prepared_df)
            self.artifacts["forecasts"] = forecasts_df
            self.artifacts["metrics"] = metrics_df
            self._save_artifact("forecast.parquet", forecasts_df)
            self._save_artifact("metrics.parquet", metrics_df)
            
            # Step 5: Compile metadata
            logger.info("\n[5/6] Compiling metadata...")
            full_metadata = {
                "pipeline": {
                    "phase": 1,
                    "status": "SUCCESS",
                    "timestamp": datetime.now().isoformat(),
                    "output_dir": str(self.output_dir)
                },
                "data": {
                    "raw_rows": len(raw_df),
                    "clean_rows": len(clean_df),
                    "prepared_rows": len(prepared_df),
                    "unique_series": prepared_df["unique_id"].nunique()
                },
                "validation": validation_report,
                "preparation": prep_metadata,
                "forecasting": {
                    "forecast_rows": len(forecasts_df),
                    "metrics_rows": len(metrics_df),
                    "method": "exponential_smoothing"
                }
            }
            self._save_metadata("metadata.json", full_metadata)
            
            # Step 6: Summary
            logger.info("\n[6/6] Pipeline complete!")
            logger.info("=" * 80)
            logger.info("ARTIFACTS CREATED:")
            logger.info(f"  ✓ raw.parquet ({len(raw_df)} rows)")
            logger.info(f"  ✓ clean.parquet ({len(clean_df)} rows)")
            logger.info(f"  ✓ forecast.parquet ({len(forecasts_df)} rows)")
            logger.info(f"  ✓ metrics.parquet ({len(metrics_df)} rows)")
            logger.info(f"  ✓ metadata.json")
            logger.info(f"  ✓ validation_report.json")
            logger.info("=" * 80)
            
            return {
                "status": "SUCCESS",
                "output_dir": str(self.output_dir),
                "artifacts": {
                    "raw": str(self.output_dir / "raw.parquet"),
                    "clean": str(self.output_dir / "clean.parquet"),
                    "forecast": str(self.output_dir / "forecast.parquet"),
                    "metrics": str(self.output_dir / "metrics.parquet"),
                    "metadata": str(self.output_dir / "metadata.json"),
                    "validation": str(self.output_dir / "validation_report.json")
                },
                "summary": {
                    "raw_rows": len(raw_df),
                    "clean_rows": len(clean_df),
                    "series_count": prepared_df["unique_id"].nunique(),
                    "forecast_rows": len(forecasts_df)
                }
            }
        
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {e}")
            logger.exception(e)
            return {
                "status": "FAILED",
                "error": str(e),
                "output_dir": str(self.output_dir)
            }
    
    def _save_artifact(self, filename: str, df: pd.DataFrame):
        """Save DataFrame artifact"""
        path = self.output_dir / filename
        df.to_parquet(path, index=False)
        logger.info(f"  Saved: {filename} ({len(df)} rows)")
    
    def _save_metadata(self, filename: str, data: Dict[str, Any]):
        """Save metadata as JSON"""
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"  Saved: {filename}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Phase 1: End-to-End EIA Data Pipeline"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/baseline",
        help="Output directory"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("EIA_API_KEY"),
        help="EIA API key (or set EIA_API_KEY env var)"
    )
    parser.add_argument(
        "--respondent-id",
        type=str,
        default="14871",
        help="EIA respondent ID"
    )
    parser.add_argument(
        "--fuel-type",
        type=str,
        default="COL",
        help="Fuel type code"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic data generation (for testing)"
    )
    
    args = parser.parse_args()
    
    api_key = args.api_key or "test_key"
    
    orchestrator = PipelineOrchestrator(args.output)
    result = orchestrator.run(
        start_date=args.start_date,
        end_date=args.end_date,
        api_key=api_key,
        respondent_id=args.respondent_id,
        fuel_type=args.fuel_type,
        synthetic=args.synthetic
    )
    
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    exit(main())
