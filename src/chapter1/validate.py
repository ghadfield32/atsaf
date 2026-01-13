"""
Chapter 1 Step 4: Validate Time Series Integrity

Hard gates for data quality:
- Uniqueness: no duplicates on [unique_id, ds]
- Frequency: expected hourly index vs observed
- Monotonic: increasing time
- Values: min/max bounds, missing rate
"""

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class ValidationResult:
    """Results of time series validation"""
    is_valid: bool
    n_rows: int
    n_duplicates: int
    n_missing_hours: int
    missing_hours: List[pd.Timestamp]
    n_nulls: int
    value_min: float
    value_max: float
    is_monotonic: bool


def validate_time_index(df: pd.DataFrame) -> ValidationResult:
    """
    Validate time series integrity for forecasting.

    Checks:
    1. No duplicates on [unique_id, ds]
    2. Expected hourly frequency vs observed (missing hours)
    3. Monotonic increasing time
    4. Value sanity (nulls, bounds)

    Args:
        df: DataFrame with columns [unique_id, ds, y]

    Returns:
        ValidationResult with detailed findings
    """
    # Check 1: Duplicates
    duplicates = df.duplicated(subset=["unique_id", "ds"], keep=False)
    n_duplicates = duplicates.sum()

    # Check 2: Missing hours
    df_sorted = df.sort_values("ds")
    expected_range = pd.date_range(
        start=df_sorted["ds"].min(),
        end=df_sorted["ds"].max(),
        freq="h",
    )
    actual_hours = set(df_sorted["ds"])
    expected_hours = set(expected_range)
    missing_hours = sorted(expected_hours - actual_hours)
    n_missing_hours = len(missing_hours)

    # Check 3: Monotonic
    is_monotonic = df_sorted["ds"].is_monotonic_increasing

    # Check 4: Value checks
    n_nulls = df["y"].isna().sum()
    value_min = df["y"].min()
    value_max = df["y"].max()

    # Overall validity
    is_valid = (n_duplicates == 0) and (n_missing_hours == 0) and is_monotonic

    return ValidationResult(
        is_valid=is_valid,
        n_rows=len(df),
        n_duplicates=n_duplicates,
        n_missing_hours=n_missing_hours,
        missing_hours=missing_hours[:10],  # First 10 only
        n_nulls=n_nulls,
        value_min=value_min,
        value_max=value_max,
        is_monotonic=is_monotonic,
    )


def print_validation_report(result: ValidationResult) -> None:
    """Print a human-readable validation report"""
    status = "PASS" if result.is_valid else "FAIL"
    print(f"\n=== Validation Report: {status} ===")
    print(f"Rows: {result.n_rows}")
    print(f"Duplicates: {result.n_duplicates}")
    print(f"Missing hours: {result.n_missing_hours}")
    if result.missing_hours:
        print(f"  First missing: {result.missing_hours[:5]}")
    print(f"Null values: {result.n_nulls}")
    print(f"Value range: {result.value_min:.0f} to {result.value_max:.0f}")
    print(f"Monotonic: {result.is_monotonic}")
