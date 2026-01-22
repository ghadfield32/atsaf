# file: src/renewable/dataset_builder.py
"""
Dataset Builder for Renewable Energy Forecasting

This module transforms raw EIA/weather data into modeling-ready datasets with:
1. Transparent preprocessing based on EDA findings
2. Physical constraint enforcement (non-negativity)
3. Comprehensive diagnostics

KEY PRINCIPLE:
Renewable energy generation CANNOT be negative. This is a physical law.
- Solar panels: 0 to max capacity
- Wind turbines: 0 to max capacity

Any negative values in raw data are data quality issues (metering, net generation
accounting, etc.) and should be handled transparently.

Preprocessing Policies:
- clamp_to_zero: Set negative values to 0 (recommended for most cases)
- investigate: Fail with detailed diagnostics (for initial data exploration)
- pass_through: No modification (only if you understand why negatives exist)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Weather variables from Open-Meteo
WEATHER_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "direct_radiation",
    "diffuse_radiation",
    "cloud_cover",
]

# Time features for modeling
TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]


@dataclass
class NegativeValueReport:
    """Report on negative values found and handled."""
    total_negative_count: int
    total_rows: int
    negative_ratio: float
    by_series: Dict[str, Dict[str, Any]]
    action_taken: str
    samples: List[Dict[str, Any]]


@dataclass
class PreprocessingReport:
    """Complete report of all preprocessing steps."""
    timestamp: str

    # Input stats
    input_rows: int
    input_series: int
    input_date_range: Dict[str, str]

    # Negative handling
    negative_report: NegativeValueReport

    # Missing data
    missing_hours_dropped: int
    series_dropped_incomplete: List[str]

    # Weather alignment
    weather_coverage: float
    weather_vars_used: List[str]

    # Output stats
    output_rows: int
    output_series: int
    output_features: List[str]

    # Configuration used
    config: Dict[str, Any]


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features (hour, day of week)."""
    out = df.copy()
    out["hour"] = out["ds"].dt.hour
    out["dow"] = out["ds"].dt.dayofweek

    # Cyclical encoding (sin/cos transform)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)

    return out.drop(columns=["hour", "dow"])


def _apply_negative_policy(
    df: pd.DataFrame,
    policy: str,
) -> Tuple[pd.DataFrame, NegativeValueReport]:
    """
    Apply negative value policy (NO INVESTIGATION - that's EDA's job).

    This function ONLY applies the policy. Investigation should be done
    in eda.py before calling dataset builder.

    Physical Reality:
    - Renewable energy generation CANNOT be negative
    - Negative values are ALWAYS data quality issues

    Policies:
    - clamp_to_zero: Set negatives to 0 (RECOMMENDED from EDA)
    - investigate: Fail with message to run EDA first
    - pass_through: No modification (if EDA recommends)

    Args:
        df: DataFrame with [unique_id, ds, y]
        policy: Policy to apply (from EDA recommendations)

    Returns:
        (processed_df, report)
    """
    neg_mask = df['y'] < 0
    neg_count = int(neg_mask.sum())
    total_rows = len(df)

    # DEBUG: Log what we're working with
    logger.debug(f"[_apply_negative_policy] Processing {total_rows} rows, found {neg_count} negatives")

    # Build report
    by_series = {}
    samples = []

    if neg_count > 0:
        for uid in df.loc[neg_mask, 'unique_id'].unique():
            series_mask = (df['unique_id'] == uid) & neg_mask
            series_neg = df.loc[series_mask]
            series_total = len(df[df['unique_id'] == uid])

            by_series[uid] = {
                'count': int(series_mask.sum()),
                'min_value': float(series_neg['y'].min()),
                'max_value': float(series_neg['y'].max()),
            }

            # Just a few samples for audit
            for _, row in series_neg.head(3).iterrows():
                samples.append({
                    'unique_id': row['unique_id'],
                    'ds': str(row['ds']),
                    'y': float(row['y']),
                })

    # Calculate negative ratio
    negative_ratio = float(neg_count / total_rows) if total_rows > 0 else 0.0

    # DEBUG: Log report creation
    logger.debug(f"[_apply_negative_policy] Creating report: {neg_count}/{total_rows} = {negative_ratio:.2%} negatives")

    report = NegativeValueReport(
        total_negative_count=neg_count,
        total_rows=total_rows,
        negative_ratio=negative_ratio,
        by_series=by_series,
        action_taken=policy,
        samples=samples,
    )

    # Apply policy
    if neg_count == 0:
        report.action_taken = 'none_needed'
        return df.copy(), report

    if policy == 'investigate':
        raise ValueError(
            f"NEGATIVE VALUES DETECTED: {neg_count} negatives found.\n"
            f"Run EDA first to investigate root cause:\n"
            f"  from src.renewable.eda import run_full_eda\n"
            f"  recommendations = run_full_eda(generation_df, weather_df, output_dir)\n"
            f"Then use recommended policy from EDA."
        )

    elif policy == 'clamp_to_zero':
        out = df.copy()
        out['y'] = out['y'].clip(lower=0)
        logger.info(
            f"[PREPROCESSING] Clamped {neg_count} negative values to 0 "
            f"({100*neg_count/total_rows:.2f}% of data)"
        )

        # Log per-series
        for uid, info in by_series.items():
            logger.info(
                f"  {uid}: {info['count']} negatives clamped "
                f"(range: [{info['min_value']:.1f}, {info['max_value']:.1f}])"
            )

        report.action_taken = 'clamped_to_zero'
        return out, report

    elif policy == 'pass_through':
        logger.warning(f"[PREPROCESSING] Passing through {neg_count} negative values")
        report.action_taken = 'passed_through'
        return df.copy(), report

    else:
        raise ValueError(f"Unknown negative_policy: {policy}")


def _enforce_hourly_grid(
    df: pd.DataFrame,
    max_missing_ratio: float = 0.02,
) -> Tuple[pd.DataFrame, List[str], int]:
    """
    Enforce complete hourly grid (no gaps).

    For time series forecasting, we need continuous hourly data.
    Series with too many gaps are dropped (no imputation - we don't fabricate data).

    Args:
        df: DataFrame with [unique_id, ds, y]
        max_missing_ratio: Maximum allowed ratio of missing hours

    Returns:
        (filtered_df, dropped_series, total_missing_hours)
    """
    dropped_series = []
    total_missing = 0

    keep_rows = []

    for uid, group in df.groupby('unique_id'):
        group = group.sort_values('ds')
        start = group['ds'].min()
        end = group['ds'].max()

        expected_hours = pd.date_range(start, end, freq='h')
        actual_hours = len(group)
        expected_count = len(expected_hours)

        missing_count = expected_count - actual_hours
        missing_ratio = missing_count / expected_count if expected_count > 0 else 0

        total_missing += missing_count

        if missing_ratio > max_missing_ratio:
            dropped_series.append(uid)
            logger.warning(
                f"[GRID] Dropping {uid}: missing {missing_count} hours "
                f"({missing_ratio:.1%} > {max_missing_ratio:.1%} threshold)"
            )
        else:
            keep_rows.append(group)

    if not keep_rows:
        raise RuntimeError(
            f"All series dropped due to missing hours. "
            f"Dropped: {dropped_series}. Consider increasing max_missing_ratio."
        )

    filtered = pd.concat(keep_rows, ignore_index=True)
    return filtered, dropped_series, total_missing


def _align_weather(
    df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, float, List[str]]:
    """
    Align weather data to generation timestamps.

    Args:
        df: Generation DataFrame (must have 'region' column or unique_id with region prefix)
        weather_df: Weather DataFrame with [ds, region, weather_vars...]

    Returns:
        (merged_df, coverage_ratio, weather_vars_used)
    """
    work = df.copy()

    # Extract region from unique_id if not present
    if 'region' not in work.columns:
        work['region'] = work['unique_id'].str.split('_').str[0]

    # Find available weather variables
    available_vars = [c for c in WEATHER_VARS if c in weather_df.columns]
    if not available_vars:
        raise ValueError(
            f"No weather variables found in weather_df. "
            f"Expected: {WEATHER_VARS}, Got: {weather_df.columns.tolist()}"
        )

    # Merge
    merged = work.merge(
        weather_df[['ds', 'region'] + available_vars],
        on=['ds', 'region'],
        how='left',
        validate='many_to_one',
    )

    # Check coverage
    missing_weather = merged[available_vars].isna().any(axis=1)
    coverage = 1 - (missing_weather.sum() / len(merged))

    if missing_weather.any():
        missing_count = int(missing_weather.sum())
        logger.warning(
            f"[WEATHER] {missing_count} rows ({1-coverage:.1%}) missing weather data"
        )

        # Drop rows with missing weather (no fabrication)
        merged = merged[~missing_weather].reset_index(drop=True)
        logger.warning(f"[WEATHER] Dropped {missing_count} rows with missing weather")

    # Drop region column (not needed for modeling)
    merged = merged.drop(columns=['region'])

    return merged, coverage, available_vars


def build_modeling_dataset(
    generation_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    *,
    negative_policy: str = 'clamp_to_zero',
    max_missing_ratio: float = 0.02,
    output_dir: Optional[Path] = None,
    eda_recommendations: Optional['PreprocessingRecommendation'] = None,
) -> Tuple[pd.DataFrame, PreprocessingReport]:
    """
    Build modeling-ready dataset from raw data.

    RECOMMENDED: Run EDA first and pass recommendations via eda_recommendations.

    Pipeline:
    1. Validate inputs
    2. Handle negative values (based on policy)
    3. Enforce hourly grid (drop incomplete series)
    4. Add time features
    5. Align weather data

    Args:
        generation_df: Raw generation data [unique_id, ds, y]
        weather_df: Raw weather data [ds, region, weather_vars...]
        negative_policy: How to handle negative values (override if no EDA)
            - 'clamp_to_zero': Set to 0 (RECOMMENDED)
            - 'investigate': Fail with diagnostics
            - 'pass_through': No modification
        max_missing_ratio: Max ratio of missing hours before dropping series
        output_dir: Optional directory for detailed reports
        eda_recommendations: Recommendations from run_full_eda() (PREFERRED)

    Returns:
        (modeling_df, preprocessing_report)
    """
    logger.info("=" * 60)
    logger.info("DATASET BUILDER - Building Modeling Dataset")
    logger.info("=" * 60)

    # Use EDA recommendations if provided
    if eda_recommendations is not None:
        negative_policy = eda_recommendations.negative_policy
        max_missing_ratio = eda_recommendations.max_missing_ratio
        logger.info("[DATASET_BUILDER] Using EDA recommendations")
        logger.info(f"  Policy: {negative_policy} (confidence: {eda_recommendations.negative_confidence})")
        logger.info(f"  Reason: {eda_recommendations.negative_reason}")
    else:
        logger.warning("[DATASET_BUILDER] No EDA recommendations - using defaults")
        logger.warning("  RECOMMENDED: Run EDA first for data-driven decisions")

    # Validate inputs
    required_gen = {'unique_id', 'ds', 'y'}
    if not required_gen.issubset(generation_df.columns):
        missing = required_gen - set(generation_df.columns)
        raise ValueError(f"generation_df missing columns: {missing}")

    if generation_df.empty:
        raise ValueError("generation_df is empty")

    required_weather = {'ds', 'region'}
    if not required_weather.issubset(weather_df.columns):
        missing = required_weather - set(weather_df.columns)
        raise ValueError(f"weather_df missing columns: {missing}")

    # Ensure datetime
    work = generation_df.copy()
    work['ds'] = pd.to_datetime(work['ds'])
    weather_df = weather_df.copy()
    weather_df['ds'] = pd.to_datetime(weather_df['ds'])

    input_rows = len(work)
    input_series = work['unique_id'].nunique()
    input_date_range = {
        'start': str(work['ds'].min()),
        'end': str(work['ds'].max()),
    }

    logger.info(f"Input: {input_rows:,} rows, {input_series} series")
    logger.info(f"Date range: {input_date_range['start']} to {input_date_range['end']}")

    # Step 1: Apply negative value policy
    logger.info(f"\n[1/4] Applying negative value policy (policy={negative_policy})...")
    work, neg_report = _apply_negative_policy(work, policy=negative_policy)

    # Step 2: Enforce hourly grid
    logger.info(f"\n[2/4] Enforcing hourly grid (max_missing={max_missing_ratio:.1%})...")
    work, dropped_series, missing_hours = _enforce_hourly_grid(
        work, max_missing_ratio=max_missing_ratio
    )

    # Step 3: Add time features
    logger.info("\n[3/4] Adding time features...")
    work = _add_time_features(work)
    logger.info(f"   Added: {TIME_FEATURES}")

    # Step 4: Align weather
    logger.info("\n[4/4] Aligning weather data...")
    work, weather_coverage, weather_vars = _align_weather(work, weather_df)
    logger.info(f"   Coverage: {weather_coverage:.1%}")
    logger.info(f"   Variables: {weather_vars}")

    # Sort and finalize
    work = work.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    output_rows = len(work)
    output_series = work['unique_id'].nunique()
    output_features = ['unique_id', 'ds', 'y'] + TIME_FEATURES + weather_vars

    # Build report
    report = PreprocessingReport(
        timestamp=datetime.now().isoformat(),
        input_rows=input_rows,
        input_series=input_series,
        input_date_range=input_date_range,
        negative_report=neg_report,
        missing_hours_dropped=missing_hours,
        series_dropped_incomplete=dropped_series,
        weather_coverage=weather_coverage,
        weather_vars_used=weather_vars,
        output_rows=output_rows,
        output_series=output_series,
        output_features=output_features,
        config={
            'negative_policy': negative_policy,
            'max_missing_ratio': max_missing_ratio,
        },
    )

    # Save report if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_dict = asdict(report)
        report_dict['negative_report'] = asdict(report.negative_report)

        report_file = output_dir / 'preprocessing_report.json'
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
            f.write('\n')  # POSIX standard: files should end with newline
        logger.info(f"\n[REPORT] Saved to: {report_file}")

    logger.info("\n" + "=" * 60)
    logger.info("DATASET BUILDER - Complete")
    logger.info(f"Output: {output_rows:,} rows, {output_series} series")
    logger.info(f"Dropped: {input_rows - output_rows:,} rows")
    logger.info("=" * 60)

    return work, report


# ============================================================================
# Dataset-Specific Builders
# ============================================================================

class RenewableDatasetBuilder:
    """Base class for fuel-type-specific dataset builders."""

    def __init__(
        self,
        fuel_type: str,
        eda_recommendations: Optional['PreprocessingRecommendation'] = None,
    ):
        self.fuel_type = fuel_type
        self.eda_recommendations = eda_recommendations
        self.config = self._get_default_config()

        if eda_recommendations:
            self.config['negative_policy'] = eda_recommendations.negative_policy
            self.config['max_missing_ratio'] = eda_recommendations.max_missing_ratio

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config for this fuel type."""
        return {
            'negative_policy': 'clamp_to_zero',
            'max_missing_ratio': 0.02,
        }

    def build(
        self,
        generation_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, PreprocessingReport]:
        """Build dataset using fuel-type-specific logic."""

        # Filter to this fuel type only
        fuel_series = [uid for uid in generation_df['unique_id'].unique() if self.fuel_type in uid]

        if not fuel_series:
            raise ValueError(f"No series found for fuel type: {self.fuel_type}")

        filtered_df = generation_df[generation_df['unique_id'].isin(fuel_series)].copy()

        logger.info(f"[{self.fuel_type}_BUILDER] Building dataset for {len(fuel_series)} series")

        return build_modeling_dataset(
            filtered_df,
            weather_df,
            negative_policy=self.config['negative_policy'],
            max_missing_ratio=self.config['max_missing_ratio'],
            output_dir=output_dir,
            eda_recommendations=self.eda_recommendations,
        )


class SolarDatasetBuilder(RenewableDatasetBuilder):
    """Solar-specific dataset builder."""

    def __init__(self, eda_recommendations: Optional['PreprocessingRecommendation'] = None):
        super().__init__('SUN', eda_recommendations)

    def _get_default_config(self) -> Dict[str, Any]:
        config = super()._get_default_config()
        # Solar: might tolerate slightly more missing data at night
        config['max_missing_ratio'] = 0.03
        return config


class WindDatasetBuilder(RenewableDatasetBuilder):
    """Wind-specific dataset builder."""

    def __init__(self, eda_recommendations: Optional['PreprocessingRecommendation'] = None):
        super().__init__('WND', eda_recommendations)


def build_dataset_by_fuel_type(
    generation_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fuel_type: str,
    output_dir: Optional[Path] = None,
    eda_recommendations: Optional['PreprocessingRecommendation'] = None,
) -> Tuple[pd.DataFrame, PreprocessingReport]:
    """
    Factory function to build dataset using appropriate fuel-specific builder.

    Args:
        generation_df: Raw generation data
        weather_df: Raw weather data
        fuel_type: 'SUN' or 'WND'
        output_dir: Optional output directory
        eda_recommendations: Optional EDA recommendations

    Returns:
        (modeling_df, report)
    """
    builders = {
        'SUN': SolarDatasetBuilder,
        'WND': WindDatasetBuilder,
    }

    if fuel_type not in builders:
        raise ValueError(f"Unknown fuel type: {fuel_type}")

    builder = builders[fuel_type](eda_recommendations)
    return builder.build(generation_df, weather_df, output_dir)


if __name__ == "__main__":
    """Test dataset builder with real data."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    generation_path = Path("data/renewable/generation.parquet")
    weather_path = Path("data/renewable/weather.parquet")

    if not generation_path.exists() or not weather_path.exists():
        print("Data files not found. Run pipeline first.")
        sys.exit(1)

    generation_df = pd.read_parquet(generation_path)
    weather_df = pd.read_parquet(weather_path)

    # First investigate negatives
    print("\n[TEST 1] Investigating negatives...")
    try:
        _, _ = build_modeling_dataset(
            generation_df, weather_df,
            negative_policy='investigate',
            output_dir=Path("data/renewable/test_investigate")
        )
    except ValueError as e:
        print(str(e))

    # Then build with clamp
    print("\n[TEST 2] Building with clamp_to_zero...")
    modeling_df, report = build_modeling_dataset(
        generation_df, weather_df,
        negative_policy='clamp_to_zero',
        output_dir=Path("data/renewable/preprocessing")
    )

    print(f"\nFinal dataset shape: {modeling_df.shape}")
    print(f"Columns: {modeling_df.columns.tolist()}")
    print(f"\nSample:")
    print(modeling_df.head())
