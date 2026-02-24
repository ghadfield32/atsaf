# file: src/renewable/modeling.py
"""
Renewable Energy Forecasting Models

This module provides probabilistic forecasting with PHYSICAL CONSTRAINTS.

KEY PRINCIPLE:
Statistical models (ARIMA, ETS, etc.) can produce negative forecasts and
prediction intervals because they assume Gaussian errors. However:

  RENEWABLE ENERGY GENERATION CANNOT BE NEGATIVE.

This module enforces this physical constraint by clipping ALL forecasts
and prediction intervals to [0, ∞). This is NOT "defensive coding" - it's
applying domain knowledge about physical reality.

Model Architecture:
1. StatsForecast for multi-series probabilistic forecasting
2. Post-processing to enforce [0, ∞) constraint
3. Calibration check for prediction intervals
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


WEATHER_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "direct_radiation",
    "diffuse_radiation",
    "cloud_cover",
]

TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]


@dataclass
class ForecastConfig:
    """Configuration for forecasting."""
    horizon: int = 24
    confidence_levels: Tuple[int, int] = (80, 95)

    # Physical constraints
    enforce_non_negative: bool = True  # ALWAYS True for renewable energy

    # CV settings
    cv_windows: int = 3
    cv_step_size: int = 168  # 1 week


def enforce_physical_constraints(
    df: pd.DataFrame,
    min_value: float = 0.0,
) -> pd.DataFrame:
    """
    Enforce physical constraints on forecasts.

    For renewable energy:
    - Generation cannot be negative
    - All forecast columns (point and intervals) are clipped to [0, ∞)

    This is applying physical reality:
    - A solar panel cannot generate negative power
    - A wind turbine cannot generate negative power

    Args:
        df: Forecast DataFrame with columns like yhat, yhat_lo_80, yhat_hi_80, etc.
        min_value: Minimum physical value (0 for generation)

    Returns:
        DataFrame with all forecasts clipped to [min_value, ∞)
    """
    # Identify forecast columns (exclude unique_id, ds, etc.)
    exclude_cols = {'unique_id', 'ds', 'cutoff', 'y', 'region', 'fuel_type'}
    forecast_cols = [c for c in df.columns if c not in exclude_cols]

    # Count values that will be clipped
    clip_counts = {}
    for col in forecast_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            below_min = (df[col] < min_value).sum()
            if below_min > 0:
                clip_counts[col] = int(below_min)

    if clip_counts:
        total_clipped = sum(clip_counts.values())
        total_values = len(df) * len(forecast_cols)
        logger.info(
            f"[PHYSICAL CONSTRAINT] Clipping {total_clipped} values to >= {min_value} "
            f"({total_clipped/total_values:.1%} of forecast values)"
        )
        for col, count in clip_counts.items():
            logger.debug(f"  {col}: {count} values clipped")

    # Apply constraint
    result = df.copy()
    for col in forecast_cols:
        if col in result.columns and pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].clip(lower=min_value)

    return result


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute forecast evaluation metrics.

    Uses RMSE and MAE (NOT MAPE because y=0 at night for solar).
    """
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'valid_rows': 0}

    errors = y_true - y_pred

    return {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mae': float(np.mean(np.abs(errors))),
        'valid_rows': int(len(y_true)),
    }


def compute_coverage(
    y_true: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
) -> float:
    """Compute prediction interval coverage."""
    valid = np.isfinite(y_true) & np.isfinite(y_lo) & np.isfinite(y_hi)
    if valid.sum() == 0:
        return np.nan

    in_interval = (y_true[valid] >= y_lo[valid]) & (y_true[valid] <= y_hi[valid])
    return float(in_interval.mean())


class RenewableForecastModel:
    """
    Probabilistic forecasting model with physical constraints.

    Uses StatsForecast for efficient multi-series forecasting,
    then enforces non-negativity on all outputs.
    """

    def __init__(
        self,
        horizon: int = 24,
        confidence_levels: Tuple[int, int] = (80, 95),
        n_jobs: int = 1,
    ):
        self.horizon = horizon
        self.confidence_levels = confidence_levels
        self.n_jobs = n_jobs
        self.sf = None
        self._train_df = None
        self._exog_cols: List[str] = []
        self.fitted = False
        self.last_cv_diagnostics: Dict[str, Any] = {}

    def _prepare_training_df(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare training DataFrame.

        Expects preprocessed data from dataset_builder (already has time features
        and weather aligned).
        """
        required = {'unique_id', 'ds', 'y'}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        if df.empty:
            raise ValueError("Empty DataFrame")

        # Check for required features
        time_features = [c for c in TIME_FEATURES if c in df.columns]
        weather_features = [c for c in WEATHER_VARS if c in df.columns]

        if not time_features:
            raise ValueError(
                "No time features found. Data should be preprocessed by dataset_builder."
            )

        self._exog_cols = time_features + weather_features

        work = df.copy()
        work['ds'] = pd.to_datetime(work['ds'])
        work = work.sort_values(['unique_id', 'ds']).reset_index(drop=True)

        # Validate no negatives in training data
        neg_count = (work['y'] < 0).sum()
        if neg_count > 0:
            raise ValueError(
                f"Training data contains {neg_count} negative values. "
                f"Data should be preprocessed by dataset_builder with negative_policy='clamp_to_zero'."
            )

        logger.info(
            f"[TRAIN] Prepared: {len(work):,} rows, {work['unique_id'].nunique()} series, "
            f"{len(self._exog_cols)} exog features"
        )

        return work

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit models on training data.

        Args:
            df: Preprocessed DataFrame from dataset_builder
        """
        from statsforecast import StatsForecast

        train_df = self._prepare_training_df(df)
        model_specs = self._build_model_specs()
        models = [model for _, model in model_specs]

        self.sf = StatsForecast(models=models, freq='h', n_jobs=self.n_jobs)
        self._train_df = train_df
        self.fitted = True

        logger.info(
            "[FIT] Fitted %d models (%s) on %d rows (n_jobs=%d)",
            len(models),
            [name for name, _ in model_specs],
            len(train_df),
            self.n_jobs,
        )

    def _build_model_specs(self) -> List[Tuple[str, Any]]:
        """Build the default model set used for training/CV."""
        from statsforecast.models import (MSTL, AutoARIMA, AutoETS,
                                          SeasonalNaive)

        model_specs: List[Tuple[str, Any]] = [
            (
                "MSTL_ARIMA",
                MSTL(
                    season_length=[24, 168],
                    trend_forecaster=AutoARIMA(),
                    alias="MSTL_ARIMA",
                ),
            ),
            ("AutoARIMA", AutoARIMA(season_length=24)),
            ("AutoETS", AutoETS(season_length=24)),
            ("SeasonalNaive", SeasonalNaive(season_length=24)),
        ]

        try:
            from statsforecast.models import AutoTheta

            model_specs.append(("AutoTheta", AutoTheta(season_length=24)))
        except ImportError:
            logger.info("[models] AutoTheta unavailable; continuing without it")

        return model_specs

    def _select_model_specs(
        self,
        model_specs: Sequence[Tuple[str, Any]],
        model_allowlist: Optional[List[str]],
    ) -> List[Tuple[str, Any]]:
        """Filter model specs by alias; fail loudly for unknown aliases."""
        if not model_allowlist:
            return list(model_specs)

        requested = [name.strip() for name in model_allowlist if name.strip()]
        if not requested:
            return list(model_specs)

        requested_map = {name.lower(): name for name in requested}
        selected = [
            (name, model)
            for name, model in model_specs
            if name.lower() in requested_map
        ]

        selected_names = {name.lower() for name, _ in selected}
        missing = [requested_map[key] for key in requested_map if key not in selected_names]
        if missing:
            available = [name for name, _ in model_specs]
            raise ValueError(
                f"Unknown model aliases in RENEWABLE_CV_MODEL_ALLOWLIST: {missing}. "
                f"Available models: {available}"
            )

        return selected

    def cross_validate(
        self,
        df: pd.DataFrame,
        n_windows: int = 3,
        step_size: int = 168,
        *,
        profile_models: bool = False,
        model_allowlist: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform cross-validation.

        Returns:
            (cv_results, leaderboard)
        """
        from statsforecast import StatsForecast

        train_df = self._prepare_training_df(df)
        model_specs = self._select_model_specs(
            self._build_model_specs(),
            model_allowlist=model_allowlist,
        )
        model_names = [name for name, _ in model_specs]

        series_lengths = train_df.groupby("unique_id").size()
        diagnostics: Dict[str, Any] = {
            "status": "running",
            "mode": "per_model" if profile_models else "batch",
            "n_windows": int(n_windows),
            "step_size_hours": int(step_size),
            "horizon_hours": int(self.horizon),
            "n_jobs": int(self.n_jobs),
            "train_rows": int(len(train_df)),
            "series_count": int(train_df["unique_id"].nunique()),
            "series_length_min": int(series_lengths.min()),
            "series_length_max": int(series_lengths.max()),
            "models": model_names,
            "model_count": int(len(model_names)),
            "per_model_seconds": {},
        }
        self.last_cv_diagnostics = diagnostics

        total_started = time.monotonic()
        current_model: Optional[str] = None

        logger.info(
            "[CV] Running: %d windows, step=%dh, horizon=%dh, n_jobs=%d, "
            "mode=%s, models=%s",
            n_windows,
            step_size,
            self.horizon,
            self.n_jobs,
            diagnostics["mode"],
            model_names,
        )

        try:
            if profile_models:
                cv_parts: List[pd.DataFrame] = []
                merge_keys = ["unique_id", "ds", "cutoff", "y"]

                for model_name, model_obj in model_specs:
                    current_model = model_name
                    model_started = time.monotonic()
                    logger.info("[CV][MODEL] %s starting", model_name)

                    sf = StatsForecast(models=[model_obj], freq='h', n_jobs=self.n_jobs)
                    model_cv = sf.cross_validation(
                        df=train_df,
                        h=self.horizon,
                        step_size=step_size,
                        n_windows=n_windows,
                        level=list(self.confidence_levels),
                    ).reset_index(drop=True)

                    model_elapsed = time.monotonic() - model_started
                    diagnostics["per_model_seconds"][model_name] = round(model_elapsed, 3)
                    logger.info(
                        "[CV][MODEL] %s done in %.1fs (rows=%d)",
                        model_name,
                        model_elapsed,
                        len(model_cv),
                    )
                    cv_parts.append(model_cv)

                cv = cv_parts[0]
                for part in cv_parts[1:]:
                    extra_cols = [c for c in part.columns if c not in merge_keys]
                    expected_rows = len(cv)
                    cv = cv.merge(
                        part[merge_keys + extra_cols],
                        on=merge_keys,
                        how="inner",
                        validate="one_to_one",
                    )
                    if len(cv) != expected_rows:
                        raise RuntimeError(
                            "Per-model CV merge changed row count: "
                            f"{expected_rows} -> {len(cv)}"
                        )
            else:
                sf = StatsForecast(
                    models=[model for _, model in model_specs],
                    freq='h',
                    n_jobs=self.n_jobs,
                )
                cv = sf.cross_validation(
                    df=train_df,
                    h=self.horizon,
                    step_size=step_size,
                    n_windows=n_windows,
                    level=list(self.confidence_levels),
                ).reset_index(drop=True)
        except Exception:
            diagnostics["status"] = "failed"
            diagnostics["failed_model"] = current_model
            diagnostics["total_seconds"] = round(time.monotonic() - total_started, 3)
            self.last_cv_diagnostics = diagnostics
            raise

        diagnostics["status"] = "ok"
        diagnostics["cv_rows"] = int(len(cv))
        diagnostics["total_seconds"] = round(time.monotonic() - total_started, 3)
        self.last_cv_diagnostics = diagnostics
        logger.info(
            "[CV][TIMING] total=%.1fs rows=%d mode=%s",
            diagnostics["total_seconds"],
            len(cv),
            diagnostics["mode"],
        )

        # CRITICAL: Apply physical constraints to CV results
        cv = enforce_physical_constraints(cv, min_value=0.0)

        # Build leaderboard
        leaderboard = self._build_leaderboard(cv)

        return cv, leaderboard

    def _build_leaderboard(self, cv_df: pd.DataFrame) -> pd.DataFrame:
        """Build model comparison leaderboard from CV results."""
        # Find model columns (not id/ds/cutoff/y, not interval columns)
        exclude = {'unique_id', 'ds', 'cutoff', 'y', 'index'}
        interval_pattern = re.compile(r'-(lo|hi)-\d+$')

        model_cols = [
            c for c in cv_df.columns
            if c not in exclude and not interval_pattern.search(c)
        ]

        rows = []
        y_true = cv_df['y'].values

        for model in model_cols:
            y_pred = cv_df[model].values
            metrics = compute_metrics(y_true, y_pred)

            row = {
                'model': model,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'valid_rows': metrics['valid_rows'],
            }

            # Add coverage for each confidence level
            for level in self.confidence_levels:
                lo_col = f"{model}-lo-{level}"
                hi_col = f"{model}-hi-{level}"
                if lo_col in cv_df.columns and hi_col in cv_df.columns:
                    coverage = compute_coverage(
                        y_true,
                        cv_df[lo_col].values,
                        cv_df[hi_col].values,
                    )
                    row[f'coverage_{level}'] = coverage

            rows.append(row)

        leaderboard = pd.DataFrame(rows)
        leaderboard = leaderboard.sort_values('rmse').reset_index(drop=True)

        return leaderboard

    def predict(
        self,
        future_exog: pd.DataFrame,
        best_model: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate forecasts.

        Args:
            future_exog: DataFrame with future exogenous features
                         Must have [unique_id, ds] + exog features
            best_model: If specified, only return this model's predictions

        Returns:
            Forecast DataFrame with physical constraints applied
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first")

        # Build future X_df
        X_df = self._build_future_X(future_exog)

        # Generate forecasts
        fcst = self.sf.forecast(
            h=self.horizon,
            df=self._train_df,
            X_df=X_df,
            level=list(self.confidence_levels),
        ).reset_index()

        # CRITICAL: Apply physical constraints
        fcst = enforce_physical_constraints(fcst, min_value=0.0)

        # If best_model specified, filter
        if best_model is not None:
            if best_model not in fcst.columns:
                available = [c for c in fcst.columns if c not in ['unique_id', 'ds']]
                raise ValueError(
                    f"Model '{best_model}' not found. Available: {available}"
                )

            keep_cols = ['unique_id', 'ds', best_model]
            rename_map = {best_model: 'yhat'}

            for level in self.confidence_levels:
                lo = f"{best_model}-lo-{level}"
                hi = f"{best_model}-hi-{level}"
                if lo in fcst.columns:
                    keep_cols.append(lo)
                    rename_map[lo] = f'yhat_lo_{level}'
                if hi in fcst.columns:
                    keep_cols.append(hi)
                    rename_map[hi] = f'yhat_hi_{level}'

            fcst = fcst[keep_cols].rename(columns=rename_map)

        return fcst

    def _build_future_X(self, future_exog: pd.DataFrame) -> pd.DataFrame:
        """Build future exogenous feature DataFrame."""
        required = {'unique_id', 'ds'}
        if not required.issubset(future_exog.columns):
            missing = required - set(future_exog.columns)
            raise ValueError(f"future_exog missing: {missing}")

        # Check exog columns
        missing_exog = [c for c in self._exog_cols if c not in future_exog.columns]
        if missing_exog:
            raise ValueError(
                f"future_exog missing required features: {missing_exog}. "
                f"Expected: {self._exog_cols}"
            )

        X = future_exog[['unique_id', 'ds'] + self._exog_cols].copy()
        X = X.sort_values(['unique_id', 'ds']).reset_index(drop=True)

        return X


def compute_baseline_metrics(
    cv_df: pd.DataFrame,
    model_name: str,
    threshold_k: float = 2.0,
) -> Dict[str, Any]:
    """
    Compute baseline metrics for drift detection.

    Args:
        cv_df: Cross-validation results
        model_name: Model to compute baseline for
        threshold_k: k for threshold = mean + k*std

    Returns:
        Baseline metrics dictionary
    """
    if model_name not in cv_df.columns:
        raise ValueError(f"Model '{model_name}' not in CV results")

    # Compute per-window metrics
    def window_rmse(g):
        metrics = compute_metrics(g['y'].values, g[model_name].values)
        return metrics['rmse']

    per_window = cv_df.groupby(['unique_id', 'cutoff']).apply(
        window_rmse, include_groups=False
    )

    rmse_mean = float(per_window.mean())
    rmse_std = float(per_window.std())

    baseline = {
        'model': model_name,
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std,
        'drift_threshold_rmse': rmse_mean + threshold_k * rmse_std,
        'n_windows': int(per_window.notna().sum()),
    }

    return baseline


if __name__ == "__main__":
    """
    Smoke test for renewable forecasting models.

    This test validates the complete modeling pipeline using real data:
    1. Loads modeling dataset created by the pipeline
    2. Runs cross-validation with multiple models
    3. Validates physical constraints (no negative forecasts)
    4. Displays performance leaderboard
    """
    import sys
    from pathlib import Path

    from src.renewable.dataset_builder import build_modeling_dataset
    from src.renewable.eia_renewable import EIARenewableFetcher
    from src.renewable.open_meteo import OpenMeteoRenewable

    logging.basicConfig(level=logging.INFO)
    fetcher = EIARenewableFetcher(debug_env=True)

    print("=== Testing Single Region Fetch ===")
    df_single = fetcher.fetch_region("CALI", "WND", "2024-12-01", "2024-12-03", debug=True)
    print(f"Single region: {len(df_single)} rows")
    print(df_single.head())

    print("\n=== Testing Multi-Region Fetch ===")
    df_multi = fetcher.fetch_all_regions("WND", "2024-12-01", "2024-12-03", regions=["CALI", "ERCO", "MISO"])
    print(f"\nMulti-region: {len(df_multi)} rows")
    print(f"Series: {df_multi['unique_id'].unique().tolist()}")

    print("\n=== Series Summary ===")
    print(fetcher.get_series_summary(df_multi))

    # sun checks:
    f = EIARenewableFetcher()
    df = f.fetch_region("CALI", "SUN", "2024-12-01", "2024-12-03", debug=True)
    print(df.head(), len(df))


    # Real API smoke test (no key needed)
    weather = OpenMeteoRenewable(strict=True)

    print("=== Testing Historical Weather (REAL API) ===")
    hist_df = weather.fetch_for_region("CALI", "2024-12-01", "2024-12-03", debug=True)
    print(f"Historical rows: {len(hist_df)}")
    print(hist_df.head())


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

    # CRITICAL: Save the modeling dataset for smoke test
    # This step was missing, causing FileNotFoundError below
    data_path = Path("data/renewable/modeling_dataset.parquet")
    print(f"\n[TEST 3] Saving modeling dataset to {data_path}...")
    modeling_df.to_parquet(data_path, index=False)
    print(f"✅ Saved {len(modeling_df):,} rows ({data_path.stat().st_size / 1024:.1f} KB)")

    # smoke test on these
    # Load preprocessed modeling dataset from pipeline (to verify save worked)
    print("\n" + "="*80)
    print("SMOKE TEST: Renewable Forecasting Models")
    print("="*80)
    print(f"Loading data from: {data_path}")

    df = pd.read_parquet(data_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Series: {df['unique_id'].unique().tolist()}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print()

    # Run cross-validation
    print("Running cross-validation (3 windows, 168h step)...")
    model = RenewableForecastModel(horizon=24, confidence_levels=(80, 95))
    cv, leaderboard = model.cross_validate(df, n_windows=3, step_size=168)

    print("\n" + "="*80)
    print("LEADERBOARD (sorted by RMSE)")
    print("="*80)
    print(leaderboard.to_string(index=False))

    print("\n" + "="*80)
    print("PHYSICAL CONSTRAINT VALIDATION")
    print("="*80)
    print(f"Min forecast (MSTL_ARIMA): {cv['MSTL_ARIMA'].min():.2f} MWh")
    print(f"Max forecast (MSTL_ARIMA): {cv['MSTL_ARIMA'].max():.2f} MWh")
    print(f"Any negative forecasts: {(cv['MSTL_ARIMA'] < 0).any()}")

    if (cv['MSTL_ARIMA'] < 0).any():
        print("\n⚠️ WARNING: Negative forecasts detected!")
        print("This violates physical constraints (renewable generation cannot be negative)")
        neg_count = (cv['MSTL_ARIMA'] < 0).sum()
        print(f"Count: {neg_count} out of {len(cv)} ({100*neg_count/len(cv):.2f}%)")
    else:
        print("\n✅ SUCCESS: All forecasts are non-negative (physical constraints satisfied)")

    print("="*80)

    print("="*80)
