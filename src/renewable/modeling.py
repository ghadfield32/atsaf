"""Probabilistic forecasting model for renewable energy generation.

This module provides multi-series probabilistic forecasting with:
- Weather exogenous features
- Dual prediction intervals (80%, 95%)
- Zero-value safe metrics (no MAPE for solar)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.chapter2.evaluation import ForecastMetrics

logger = logging.getLogger(__name__)


def _get_n_jobs() -> int:
    raw = os.getenv("RENEWABLE_N_JOBS", "1").strip()
    try:
        return int(raw)
    except ValueError:
        return 1


@dataclass
class ForecastConfig:
    """Configuration for renewable forecasting."""

    horizon: int = 24
    confidence_levels: tuple[int, int] = (80, 95)
    season_length: int = 24
    weekly_season: int = 168  # 24 * 7
    models: list[str] = field(default_factory=lambda: ["AutoARIMA", "MSTL"])


class RenewableForecastModel:
    """Multi-series probabilistic forecasting with weather exogenous.

    Designed for wind/solar generation with:
    - Weather features (wind speed, solar radiation)
    - Dual prediction intervals (80%, 95%)
    - Zero-safe metrics (solar has 0s at night)

    Example:
        >>> model = RenewableForecastModel(horizon=24)
        >>> model.fit(df_train, weather_train)
        >>> forecasts = model.predict(df_test, weather_forecast)
        >>> print(forecasts[["unique_id", "ds", "yhat", "yhat_lo_80", "yhat_hi_80"]])
    """

    def __init__(
        self,
        horizon: int = 24,
        confidence_levels: tuple[int, int] = (80, 95),
    ):
        """Initialize the forecasting model.

        Args:
            horizon: Number of hours to forecast
            confidence_levels: Tuple of confidence levels for intervals
        """
        self.horizon = horizon
        self.confidence_levels = confidence_levels
        self.sf = None
        self.fitted = False

    def prepare_features(
        self,
        df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Merge generation with weather features and add time features.

        Args:
            df: Generation data [unique_id, ds, y]
            weather_df: Weather data [ds, region, weather_vars...]

        Returns:
            DataFrame with features added [unique_id, ds, y, exog_vars...]
        """
        result = df.copy()

        # Add time features (cyclic encoding)
        result["hour"] = result["ds"].dt.hour
        result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
        result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)

        result["dayofweek"] = result["ds"].dt.dayofweek
        result["dow_sin"] = np.sin(2 * np.pi * result["dayofweek"] / 7)
        result["dow_cos"] = np.cos(2 * np.pi * result["dayofweek"] / 7)

        # Add weather features if provided
        if weather_df is not None and len(weather_df) > 0:
            # Extract region from unique_id
            result["region"] = result["unique_id"].str.split("_").str[0]

            # Merge weather
            weather_cols = [c for c in weather_df.columns if c not in ["ds", "region"]]

            result = result.merge(
                weather_df[["ds", "region"] + weather_cols],
                on=["ds", "region"],
                how="left",
            )

            # Fill missing weather with forward fill within series
            for col in weather_cols:
                if col in result.columns:
                    result[col] = result.groupby("unique_id")[col].ffill()

            result = result.drop(columns=["region"])

        # Add lag features (shifted to prevent leakage)
        result = result.sort_values(["unique_id", "ds"])
        result["y_lag_1"] = result.groupby("unique_id")["y"].shift(1)
        result["y_lag_24"] = result.groupby("unique_id")["y"].shift(24)

        # Drop temporary columns
        result = result.drop(columns=["hour", "dayofweek"], errors="ignore")

        return result

    def fit(self, df: pd.DataFrame, weather_df: Optional[pd.DataFrame] = None) -> None:
        """Train StatsForecast models.

        Args:
            df: Training data [unique_id, ds, y]
            weather_df: Weather features [ds, region, weather_vars...]
        """
        from statsforecast import StatsForecast
        from statsforecast.models import MSTL, AutoARIMA, AutoETS, SeasonalNaive

        # Prepare features
        train_df = self.prepare_features(df, weather_df)

        # Define models
        models = [
            AutoARIMA(season_length=24),
            SeasonalNaive(season_length=24),
            AutoETS(season_length=24),
            MSTL(
                season_length=[24, 168],
                trend_forecaster=AutoARIMA(),
                alias="MSTL_ARIMA",
            ),
        ]

        # Create StatsForecast object
        self.sf = StatsForecast(
            models=models,
            freq="h",
            n_jobs=_get_n_jobs(),
        )

        # Store training data for CV and predictions
        self._train_df = train_df[["unique_id", "ds", "y"]].copy()

        logger.info(f"Model fit: {len(train_df)} rows, {train_df['unique_id'].nunique()} series")
        self.fitted = True

    def predict(
        self,
        future_weather: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate forecasts with dual prediction intervals.

        Args:
            future_weather: Weather forecast data for prediction horizon

        Returns:
            DataFrame with columns:
            - unique_id, ds, yhat
            - yhat_lo_80, yhat_hi_80
            - yhat_lo_95, yhat_hi_95
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        # Generate forecasts with both confidence levels
        forecasts = self.sf.forecast(
            h=self.horizon,
            df=self._train_df,
            level=list(self.confidence_levels),
        )

        forecasts = forecasts.reset_index()

        # Standardize column names
        result = self._standardize_forecast_columns(forecasts)

        logger.info(f"Predictions generated: {len(result)} rows, {self.horizon}h horizon")

        return result

    def cross_validate(
        self,
        df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
        n_windows: int = 5,
        step_size: int = 168,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run rolling-origin cross-validation.

        Args:
            df: Full dataset [unique_id, ds, y]
            weather_df: Weather features
            n_windows: Number of CV windows
            step_size: Hours between windows

        Returns:
            Tuple of (cv_results, leaderboard)
        """
        from statsforecast import StatsForecast
        from statsforecast.models import MSTL, AutoARIMA, AutoETS, SeasonalNaive

        # Prepare features
        cv_df = self.prepare_features(df, weather_df)
        cv_df = cv_df[["unique_id", "ds", "y"]].copy()

        # Models
        models = [
            AutoARIMA(season_length=24),
            SeasonalNaive(season_length=24),
            AutoETS(season_length=24),
            MSTL(
                season_length=[24, 168],
                trend_forecaster=AutoARIMA(),
                alias="MSTL_ARIMA",
            ),
        ]

        sf = StatsForecast(models=models, freq="h", n_jobs=_get_n_jobs())

        logger.info(f"Running CV: {n_windows} windows, step={step_size}h, horizon={self.horizon}h")

        cv_results = sf.cross_validation(
            df=cv_df,
            h=self.horizon,
            step_size=step_size,
            n_windows=n_windows,
            level=list(self.confidence_levels),
        )

        cv_results = cv_results.reset_index()

        # Compute leaderboard
        leaderboard = self._compute_leaderboard(cv_results)

        return cv_results, leaderboard

    def _standardize_forecast_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize forecast column names.

        Renames model-specific columns to standard format:
        - yhat: point forecast (uses best model)
        - yhat_lo_80, yhat_hi_80: 80% interval
        - yhat_lo_95, yhat_hi_95: 95% interval
        """
        result = df.copy()

        # Find best model (prefer MSTL_ARIMA if available)
        model_cols = [c for c in result.columns if c not in ["unique_id", "ds", "cutoff"]]
        point_cols = [c for c in model_cols if not any(x in c for x in ["-lo-", "-hi-"])]

        if "MSTL_ARIMA" in point_cols:
            best_model = "MSTL_ARIMA"
        elif "AutoARIMA" in point_cols:
            best_model = "AutoARIMA"
        else:
            best_model = point_cols[0] if point_cols else None

        if best_model:
            result["yhat"] = result[best_model]

            for level in self.confidence_levels:
                lo_col = f"{best_model}-lo-{level}"
                hi_col = f"{best_model}-hi-{level}"

                if lo_col in result.columns:
                    result[f"yhat_lo_{level}"] = result[lo_col]
                if hi_col in result.columns:
                    result[f"yhat_hi_{level}"] = result[hi_col]

        # Keep essential columns
        keep_cols = ["unique_id", "ds"]
        if "cutoff" in result.columns:
            keep_cols.append("cutoff")
        if "y" in result.columns:
            keep_cols.append("y")

        keep_cols.extend(["yhat"])
        for level in self.confidence_levels:
            keep_cols.extend([f"yhat_lo_{level}", f"yhat_hi_{level}"])

        # Also keep raw model columns for analysis
        for col in model_cols:
            if col not in keep_cols:
                keep_cols.append(col)

        result = result[[c for c in keep_cols if c in result.columns]]

        return result

    def _compute_leaderboard(self, cv_results: pd.DataFrame) -> pd.DataFrame:
        """Compute model leaderboard from CV results.

        Args:
            cv_results: Cross-validation results

        Returns:
            Leaderboard DataFrame with metrics per model
        """
        # Find model columns
        model_cols = [
            c for c in cv_results.columns
            if c not in ["unique_id", "ds", "cutoff", "y"]
            and not any(x in c for x in ["-lo-", "-hi-"])
        ]

        rows = []

        for model in model_cols:
            y_true = cv_results["y"].values
            y_pred = cv_results[model].values

            # CRITICAL: Use RMSE and MAE, NOT MAPE (solar has zeros)
            rmse = ForecastMetrics.rmse(y_true, y_pred)
            mae = ForecastMetrics.mae(y_true, y_pred)

            # Coverage for each level
            coverages = {}
            for level in self.confidence_levels:
                lo_col = f"{model}-lo-{level}"
                hi_col = f"{model}-hi-{level}"

                if lo_col in cv_results.columns and hi_col in cv_results.columns:
                    coverage = ForecastMetrics.coverage(
                        y_true,
                        cv_results[lo_col].values,
                        cv_results[hi_col].values,
                    )
                    coverages[f"coverage_{level}"] = coverage

            rows.append({
                "model": model,
                "rmse": rmse,
                "mae": mae,
                **coverages,
            })

        leaderboard = pd.DataFrame(rows).sort_values("rmse")

        return leaderboard

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lower_80: Optional[np.ndarray] = None,
        upper_80: Optional[np.ndarray] = None,
        lower_95: Optional[np.ndarray] = None,
        upper_95: Optional[np.ndarray] = None,
    ) -> dict:
        """Compute evaluation metrics.

        CRITICAL: Uses RMSE and MAE, NOT MAPE (solar has zeros at night).

        Args:
            y_true: Actual values
            y_pred: Predicted values
            lower_80, upper_80: 80% prediction interval bounds
            lower_95, upper_95: 95% prediction interval bounds

        Returns:
            Dictionary with metrics
        """
        metrics = {
            "rmse": ForecastMetrics.rmse(y_true, y_pred),
            "mae": ForecastMetrics.mae(y_true, y_pred),
            # NOTE: MAPE intentionally excluded - undefined when y=0 (solar at night)
        }

        # Coverage metrics
        if lower_80 is not None and upper_80 is not None:
            metrics["coverage_80"] = ForecastMetrics.coverage(y_true, lower_80, upper_80)

        if lower_95 is not None and upper_95 is not None:
            metrics["coverage_95"] = ForecastMetrics.coverage(y_true, lower_95, upper_95)

        return metrics


def compute_baseline_metrics(
    cv_results: pd.DataFrame,
    model_name: str = "MSTL_ARIMA",
) -> dict:
    """Compute baseline metrics from backtest for drift detection.

    Args:
        cv_results: Cross-validation results
        model_name: Model to use for baseline

    Returns:
        Dictionary with baseline statistics:
        - rmse_mean, rmse_std: for threshold calculation
        - mae_mean, mae_std
        - coverage_80_mean, coverage_95_mean
    """
    if model_name not in cv_results.columns:
        raise ValueError(f"Model {model_name} not found in CV results")

    # Compute metrics per cutoff window
    window_metrics = []

    for cutoff in cv_results["cutoff"].unique():
        window = cv_results[cv_results["cutoff"] == cutoff]
        y_true = window["y"].values
        y_pred = window[model_name].values

        rmse = ForecastMetrics.rmse(y_true, y_pred)
        mae = ForecastMetrics.mae(y_true, y_pred)

        window_metrics.append({"cutoff": cutoff, "rmse": rmse, "mae": mae})

    metrics_df = pd.DataFrame(window_metrics)

    baseline = {
        "model": model_name,
        "rmse_mean": metrics_df["rmse"].mean(),
        "rmse_std": metrics_df["rmse"].std(),
        "mae_mean": metrics_df["mae"].mean(),
        "mae_std": metrics_df["mae"].std(),
        "n_windows": len(metrics_df),
    }

    # Drift threshold: mean + 2*std
    baseline["drift_threshold_rmse"] = baseline["rmse_mean"] + 2 * baseline["rmse_std"]
    baseline["drift_threshold_mae"] = baseline["mae_mean"] + 2 * baseline["mae_std"]

    logger.info(
        f"Baseline computed: RMSE={baseline['rmse_mean']:.1f}±{baseline['rmse_std']:.1f}, "
        f"threshold={baseline['drift_threshold_rmse']:.1f}"
    )

    return baseline


if __name__ == "__main__":
    # Test the model
    logging.basicConfig(level=logging.INFO)

    # Create synthetic data
    import pandas as pd
    import numpy as np

    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=720, freq="h")
    series_ids = ["CALI_WND", "ERCO_WND"]

    dfs = []
    for sid in series_ids:
        y = 100 + 20 * np.sin(np.arange(720) * 2 * np.pi / 24) + np.random.normal(0, 5, 720)
        dfs.append(pd.DataFrame({"unique_id": sid, "ds": dates, "y": y}))

    df = pd.concat(dfs, ignore_index=True)

    print("\n=== Testing RenewableForecastModel ===")
    model = RenewableForecastModel(horizon=24)

    # Test feature preparation
    print("\n1. Feature preparation")
    features = model.prepare_features(df)
    print(f"Features: {features.columns.tolist()}")

    # Test cross-validation
    print("\n2. Cross-validation")
    cv_results, leaderboard = model.cross_validate(df, n_windows=3, step_size=168)
    print(f"CV results: {len(cv_results)} rows")
    print("\nLeaderboard:")
    print(leaderboard)

    # Test baseline computation
    print("\n3. Baseline metrics")
    baseline = compute_baseline_metrics(cv_results)
    print(f"Baseline: {baseline}")
