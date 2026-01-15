# file: src/chapter2/evaluation.py
"""
Chapter 2: Model Evaluation Metrics

Computes forecasting metrics with explicit NaN handling (fail-loud principle).
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ForecastMetrics:
    """Compute and track forecasting evaluation metrics"""

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error

        Explicit NaN masking (fail-loud):
        - Returns NaN if no valid predictions
        - Masks NaN/inf values before computation
        """
        valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)

        if valid_mask.sum() == 0:
            return np.nan

        return np.sqrt(np.mean((y_pred[valid_mask] - y_true[valid_mask]) ** 2))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error

        Explicit NaN masking (fail-loud):
        - Returns NaN if no valid predictions
        - Masks NaN/inf values before computation
        """
        valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)

        if valid_mask.sum() == 0:
            return np.nan

        return np.mean(np.abs(y_pred[valid_mask] - y_true[valid_mask]))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error (%)

        Explicit NaN masking (fail-loud):
        - Returns NaN if no valid predictions
        - Masks NaN/inf values and zero y_true before computation
        """
        valid_mask = (
            np.isfinite(y_pred) &
            np.isfinite(y_true) &
            (np.abs(y_true) > 1e-10)
        )

        if valid_mask.sum() == 0:
            return np.nan

        ape = np.abs((y_pred[valid_mask] - y_true[valid_mask]) / np.abs(y_true[valid_mask]))
        return 100 * np.mean(ape)

    @staticmethod
    def mase(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        season_length: int = 24
    ) -> float:
        """
        Mean Absolute Scaled Error

        Scales error relative to naive seasonal forecasting.

        Explicit NaN masking (fail-loud):
        - Returns NaN if insufficient training data
        - Masks NaN/inf values before computation
        """
        # Check minimum training data
        if len(y_train) < season_length:
            return np.nan

        # Compute seasonal naive MAE
        try:
            mae_train = np.mean(np.abs(
                y_train[season_length:] - y_train[:-season_length]
            ))
        except:
            return np.nan

        if mae_train < 1e-10:
            return np.nan

        # Compute test MAE
        valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)

        if valid_mask.sum() == 0:
            return np.nan

        mae_test = np.mean(np.abs(y_pred[valid_mask] - y_true[valid_mask]))

        return mae_test / mae_train

    @staticmethod
    def coverage(
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> float:
        """
        Prediction Interval Coverage (%)

        Percentage of actual values within prediction interval.

        Explicit NaN masking (fail-loud):
        - Returns NaN if no valid predictions
        - Counts valid (non-NaN) rows in denominator
        """
        valid_mask = (
            np.isfinite(y_true) &
            np.isfinite(lower) &
            np.isfinite(upper)
        )

        if valid_mask.sum() == 0:
            return np.nan

        covered = (y_true[valid_mask] >= lower[valid_mask]) & \
                  (y_true[valid_mask] <= upper[valid_mask])

        return 100 * np.mean(covered)

    @staticmethod
    def compute_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics at once

        Args:
            y_true: Actual values
            y_pred: Predictions
            y_train: Training values (for MASE)

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "rmse": ForecastMetrics.rmse(y_true, y_pred),
            "mae": ForecastMetrics.mae(y_true, y_pred),
            "mape": ForecastMetrics.mape(y_true, y_pred),
        }

        if y_train is not None:
            metrics["mase"] = ForecastMetrics.mase(
                y_true, y_pred, y_train, season_length=24
            )

        return metrics


def compute_series_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    valid_threshold: int = 1
) -> Dict[str, float]:
    """
    Compute metrics with explicit validation

    Args:
        y_true: Actual values
        y_pred: Predictions
        y_train: Training values (for MASE)
        valid_threshold: Minimum valid predictions required

    Returns:
        Dictionary of metrics
    """
    # Count valid predictions
    valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)
    valid_count = valid_mask.sum()

    if valid_count < valid_threshold:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "mase": np.nan,
            "valid_count": valid_count,
            "error": f"Insufficient valid predictions: {valid_count} < {valid_threshold}"
        }

    metrics = ForecastMetrics.compute_all(y_true, y_pred, y_train)
    metrics["valid_count"] = valid_count

    return metrics


def aggregate_metrics(
    results: pd.DataFrame,
    by: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate metrics across splits and series

    Args:
        results: DataFrame with metric columns
        by: Groupby column ("model_name", "unique_id", etc.)

    Returns:
        Aggregated metrics DataFrame
    """
    metric_cols = ["rmse", "mae", "mape", "mase"]

    if by is None:
        # Overall aggregation
        agg = results[metric_cols].agg([
            ("mean", "mean"),
            ("std", "std"),
            ("min", "min"),
            ("max", "max")
        ])
        return agg
    else:
        # Grouped aggregation
        agg = results.groupby(by)[metric_cols].agg([
            ("mean", "mean"),
            ("std", "std"),
            ("count", "count")
        ])
        return agg
        return agg
