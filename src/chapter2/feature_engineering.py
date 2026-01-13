# file: src/chapter2/feature_engineering.py
"""
Chapter 2: Simple timetk-style feature helpers (pandas only).

These helpers are not used by the default StatsForecast pipeline. Use them
for EDA or when you add an ML-based forecasting track that explicitly consumes
X features (e.g., MLForecast or sklearn models).
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def add_calendar_features(df: pd.DataFrame, ds_col: str = "ds") -> pd.DataFrame:
    """
    Add basic calendar features from a datetime column.
    """
    features = df.copy()
    if ds_col not in features.columns:
        raise ValueError(f"Missing datetime column: {ds_col}")

    ds = pd.to_datetime(features[ds_col], errors="raise")
    features["hour"] = ds.dt.hour
    features["dayofweek"] = ds.dt.dayofweek
    features["dayofyear"] = ds.dt.dayofyear
    features["month"] = ds.dt.month
    features["is_weekend"] = ds.dt.dayofweek >= 5
    return features


def add_lag_features(
    df: pd.DataFrame,
    y_col: str = "y",
    lags: Iterable[int] = (1, 24, 168),
) -> pd.DataFrame:
    """
    Add lag features using past values only.
    """
    features = df.copy()
    if y_col not in features.columns:
        raise ValueError(f"Missing value column: {y_col}")

    for lag in lags:
        features[f"{y_col}_lag_{lag}"] = features[y_col].shift(lag)
    return features


def add_lag_features_multi(
    df: pd.DataFrame,
    cols: Iterable[str],
    lags: Iterable[int] = (1, 24, 168),
) -> pd.DataFrame:
    """
    Add lag features for multiple columns using past values only.
    """
    features = df.copy()
    for col in cols:
        if col not in features.columns:
            raise ValueError(f"Missing value column: {col}")
        for lag in lags:
            features[f"{col}_lag_{lag}"] = features[col].shift(lag)
    return features


def add_rolling_features(
    df: pd.DataFrame,
    y_col: str = "y",
    windows: Iterable[int] = (24, 168),
) -> pd.DataFrame:
    """
    Add rolling-window mean features with a 1-step shift to avoid leakage.
    """
    features = df.copy()
    if y_col not in features.columns:
        raise ValueError(f"Missing value column: {y_col}")

    shifted = features[y_col].shift(1)
    for window in windows:
        features[f"{y_col}_roll_mean_{window}"] = shifted.rolling(window, min_periods=1).mean()
    return features


def add_rolling_features_multi(
    df: pd.DataFrame,
    cols: Iterable[str],
    windows: Iterable[int] = (24, 168),
) -> pd.DataFrame:
    """
    Add rolling-window mean features for multiple columns with a 1-step shift.
    """
    features = df.copy()
    for col in cols:
        if col not in features.columns:
            raise ValueError(f"Missing value column: {col}")
        shifted = features[col].shift(1)
        for window in windows:
            features[f"{col}_roll_mean_{window}"] = shifted.rolling(window, min_periods=1).mean()
    return features


def build_timetk_features(
    df: pd.DataFrame,
    ds_col: str = "ds",
    y_col: str = "y",
    lags: Iterable[int] = (1, 24, 168),
    windows: Iterable[int] = (24, 168),
) -> pd.DataFrame:
    """
    Convenience wrapper: calendar + lag + rolling features.
    """
    features = add_calendar_features(df, ds_col=ds_col)
    features = add_lag_features(features, y_col=y_col, lags=lags)
    features = add_rolling_features(features, y_col=y_col, windows=windows)
    return features


def build_timetk_features_multi(
    df: pd.DataFrame,
    ds_col: str = "ds",
    feature_cols: Optional[Iterable[str]] = None,
    lags: Iterable[int] = (1, 24, 168),
    windows: Iterable[int] = (24, 168),
) -> pd.DataFrame:
    """
    Calendar + lag + rolling features for multiple columns.
    """
    features = add_calendar_features(df, ds_col=ds_col)
    if feature_cols is None:
        feature_cols = [c for c in features.columns if c != ds_col]
    features = add_lag_features_multi(features, cols=feature_cols, lags=lags)
    features = add_rolling_features_multi(features, cols=feature_cols, windows=windows)
    return features
