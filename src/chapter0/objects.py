# file: src/chapter0/objects.py
"""
Chapter 0: Python equivalents for ts / tsibble / timetk concepts.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd

from src.chapter1.validate import validate_time_index


def normalize_to_utc(df: pd.DataFrame, ds_col: str = "ds") -> pd.DataFrame:
    """
    Normalize a datetime column to timezone-naive UTC.

    StatsForecast expects timezone-naive UTC timestamps.
    """
    if ds_col not in df.columns:
        raise ValueError(f"Missing required datetime column: {ds_col}")

    normalized = df.copy()
    normalized[ds_col] = (
        pd.to_datetime(normalized[ds_col], errors="raise", utc=True)
        .dt.tz_localize(None)
    )
    return normalized


def to_ts_series(
    df: pd.DataFrame,
    ds_col: str = "ds",
    y_col: str = "y",
) -> pd.Series:
    """
    Create a single-series ts object: pd.Series with a DatetimeIndex (UTC-aware).
    """
    if ds_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Expected columns: {ds_col}, {y_col}")

    ds = pd.to_datetime(df[ds_col], errors="raise", utc=True)
    series = pd.Series(df[y_col].to_numpy(), index=ds)
    series.index = series.index.tz_convert("UTC")
    return series


def to_tsibble(
    df: pd.DataFrame,
    unique_id_col: str = "unique_id",
    ds_col: str = "ds",
    y_col: str = "y",
) -> pd.DataFrame:
    """
    Create a tidy time-series table with columns [unique_id, ds, y].
    """
    missing = [col for col in (unique_id_col, ds_col, y_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    tidy = df[[unique_id_col, ds_col, y_col]].copy()
    tidy.columns = ["unique_id", "ds", "y"]
    tidy = normalize_to_utc(tidy, ds_col="ds")
    return tidy


def validate_tsibble(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate the tsibble contract using the Chapter 1 integrity check.
    """
    result = validate_time_index(df)
    if result.is_valid:
        return True, "valid"
    return False, "invalid"


def assert_tsibble_contract(df: pd.DataFrame) -> None:
    """
    Raise a ValueError if the tsibble contract is violated.
    """
    result = validate_time_index(df)
    if not result.is_valid:
        raise ValueError(
            f"Invalid tsibble: duplicates={result.n_duplicates}, "
            f"missing_hours={result.n_missing_hours}, "
            f"monotonic={result.is_monotonic}"
        )


def add_time_features(
    df: pd.DataFrame,
    ds_col: str = "ds",
    y_col: str = "y",
    lags: Iterable[int] = (1, 24, 168),
    rolling_windows: Iterable[int] = (24, 168),
) -> pd.DataFrame:
    """
    Basic timetk-style features using pandas (safe, past-only).
    """
    features = df.copy()
    features = normalize_to_utc(features, ds_col=ds_col)
    features = features.sort_values(ds_col).reset_index(drop=True)

    features["hour"] = features[ds_col].dt.hour
    features["dayofweek"] = features[ds_col].dt.dayofweek
    features["month"] = features[ds_col].dt.month

    for lag in lags:
        features[f"y_lag_{lag}"] = features[y_col].shift(lag)

    shifted = features[y_col].shift(1)
    for window in rolling_windows:
        features[f"y_roll_mean_{window}"] = shifted.rolling(window, min_periods=1).mean()

    return features
