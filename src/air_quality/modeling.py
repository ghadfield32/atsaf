from __future__ import annotations

import logging
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from src.chapter2.evaluation import ForecastMetrics
from src.chapter2.feature_engineering import build_timetk_features_multi

from .config import AQIPipelineConfig

logger = logging.getLogger(__name__)


def build_baseline_models(season_length: int = 24) -> list:
    from statsforecast.models import AutoARIMA, DynamicOptimizedTheta, HoltWinters, SeasonalNaive

    return [
        AutoARIMA(season_length=season_length),
        SeasonalNaive(season_length=season_length),
        DynamicOptimizedTheta(season_length=season_length),
        HoltWinters(season_length=season_length),
    ]


def _model_names(models: Iterable) -> list[str]:
    names = []
    for model in models:
        if hasattr(model, "alias"):
            names.append(model.alias)
        else:
            names.append(type(model).__name__)
    return names


def run_baseline_backtest(
    df: pd.DataFrame,
    config: AQIPipelineConfig,
    season_length: int = 24,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling-origin CV with StatsForecast + RMSE/MAE/MASE leaderboard.
    """
    from statsforecast import StatsForecast

    if df.empty:
        return df.copy(), pd.DataFrame()

    models = build_baseline_models(season_length=season_length)
    sf = StatsForecast(models=models, freq="h", n_jobs=-1)

    cv_df = sf.cross_validation(
        df=df,
        h=config.horizon,
        step_size=config.step_size,
        n_windows=config.n_windows,
        level=[config.confidence_level],
    )

    model_names = _model_names(models)
    metrics_rows = []

    for cutoff in cv_df["cutoff"].unique():
        window_df = cv_df[cv_df["cutoff"] == cutoff]
        y_true = window_df["y"].values

        train_mask = df["ds"] <= cutoff
        y_train = df.loc[train_mask, "y"].values

        for model in model_names:
            if model not in window_df.columns:
                continue

            y_pred = window_df[model].values
            valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
            valid_rows = int(valid_mask.sum())

            rmse_val = ForecastMetrics.rmse(y_true, y_pred)
            mae_val = ForecastMetrics.mae(y_true, y_pred)
            mase_val = ForecastMetrics.mase(y_true, y_pred, y_train, season_length=season_length)

            coverage_val = np.nan
            lo_col = f"{model}-lo-{config.confidence_level}"
            hi_col = f"{model}-hi-{config.confidence_level}"
            if lo_col in window_df.columns and hi_col in window_df.columns:
                coverage_val = ForecastMetrics.coverage(
                    y_true,
                    window_df[lo_col].values,
                    window_df[hi_col].values,
                )

            metrics_rows.append(
                {
                    "cutoff": cutoff,
                    "model": model,
                    "rmse": rmse_val,
                    "mae": mae_val,
                    "mase": mase_val,
                    "coverage": coverage_val,
                    "valid_rows": valid_rows,
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        return cv_df, pd.DataFrame()

    leaderboard = (
        metrics_df.groupby("model")
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mase_mean=("mase", "mean"),
            mase_std=("mase", "std"),
            coverage_mean=("coverage", "mean"),
            valid_rows=("valid_rows", "sum"),
        )
        .reset_index()
    )

    leaderboard = leaderboard.sort_values("rmse_mean").reset_index(drop=True)
    leaderboard["rank"] = leaderboard.index + 1
    return cv_df, leaderboard


def forecast_baseline(
    df: pd.DataFrame,
    config: AQIPipelineConfig,
    season_length: int = 24,
) -> pd.DataFrame:
    from statsforecast import StatsForecast

    models = build_baseline_models(season_length=season_length)
    sf = StatsForecast(models=models, freq="h", n_jobs=-1)

    forecast_df = sf.forecast(
        df=df,
        h=config.horizon,
        level=[config.confidence_level],
    )

    return forecast_df


def _build_weather_features(
    weather_df: pd.DataFrame,
    config: AQIPipelineConfig,
) -> Tuple[pd.DataFrame, list[str]]:
    features = build_timetk_features_multi(
        weather_df,
        ds_col="ds",
        feature_cols=list(config.weather_variables),
        lags=config.residual_lags,
        windows=config.residual_windows,
    )

    feature_cols = [c for c in features.columns if c not in ("ds",)]
    return features[["ds"] + feature_cols], feature_cols


def train_residual_corrector(
    cv_df: pd.DataFrame,
    leaderboard: pd.DataFrame,
    weather_df: pd.DataFrame,
    config: AQIPipelineConfig,
):
    """
    Train a residual model on rolling-origin residuals + weather features.
    """
    if cv_df.empty or leaderboard.empty or weather_df.empty:
        raise ValueError("cv_df, leaderboard, and weather_df must be non-empty")

    champion = leaderboard.iloc[0]["model"]
    if champion not in cv_df.columns:
        raise ValueError(f"Champion model {champion} not found in cv_df columns")

    residuals = cv_df[["ds", "y", champion]].rename(columns={champion: "yhat"})
    residuals["residual"] = residuals["y"] - residuals["yhat"]
    residuals = residuals.dropna(subset=["residual"])

    features_df, feature_cols = _build_weather_features(weather_df, config)
    train_df = residuals.merge(features_df, on="ds", how="inner")
    train_df = train_df.dropna(subset=feature_cols + ["residual"])

    if train_df.empty:
        raise ValueError("No training rows after feature alignment. Check weather coverage.")

    X = train_df[feature_cols].to_numpy()
    y = train_df["residual"].to_numpy()

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    model.fit(X, y)

    payload = {
        "model": model,
        "feature_cols": feature_cols,
        "champion": champion,
    }

    return payload


def apply_residual_corrector(
    forecast_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    payload: dict,
    config: AQIPipelineConfig,
) -> pd.DataFrame:
    """
    Apply residual model to a baseline forecast, returning adjusted column.
    """
    if forecast_df.empty or weather_df.empty:
        return forecast_df

    champion = payload["champion"]
    if champion not in forecast_df.columns:
        raise ValueError(f"Champion model {champion} not in forecast columns")

    features_df, feature_cols = _build_weather_features(weather_df, config)
    features_df = features_df.set_index("ds")

    fc = forecast_df.copy()
    fc["ds"] = pd.to_datetime(fc["ds"], errors="coerce", utc=True).dt.tz_localize(None)

    feature_aligned = features_df.reindex(fc["ds"]).reset_index()
    X = feature_aligned[feature_cols].to_numpy()

    residual_pred = payload["model"].predict(X)
    adjusted_col = f"{champion}_residual_adjusted"

    fc[adjusted_col] = fc[champion].values + residual_pred
    return fc
