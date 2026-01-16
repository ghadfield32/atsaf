# file: src/renewable/modeling.py

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Sequence
import re
from typing import Any

from src.chapter2.evaluation import ForecastMetrics


WEATHER_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "direct_radiation",
    "diffuse_radiation",
    "cloud_cover",
]


def _log_series_summary(df: pd.DataFrame, *, value_col: str = "y", label: str = "series") -> None:
    if df.empty:
        print(f"[{label}] EMPTY")
        return

    tmp = df.copy()
    tmp["ds"] = pd.to_datetime(tmp["ds"], errors="coerce")

    def _mode_delta_hours(g: pd.Series) -> float:
        d = g.sort_values().diff().dropna()
        if d.empty:
            return float("nan")
        return float(d.dt.total_seconds().div(3600).mode().iloc[0])

    g = tmp.groupby("unique_id").agg(
        rows=(value_col, "count"),
        na_y=(value_col, lambda s: int(s.isna().sum())),
        min_ds=("ds", "min"),
        max_ds=("ds", "max"),
        min_y=(value_col, "min"),
        max_y=(value_col, "max"),
        mean_y=(value_col, "mean"),
        zero_y=(value_col, lambda s: int((s == 0).sum())),
        mode_delta_hours=("ds", _mode_delta_hours),
    ).reset_index().sort_values("unique_id")

    print(f"[{label}] series={g['unique_id'].nunique()} rows={len(tmp)}")
    print(g.head(20).to_string(index=False))

def _missing_hour_blocks(ds: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """
    Return contiguous blocks of missing hourly timestamps.
    Each tuple: (block_start, block_end, n_hours)
    """
    ds = pd.to_datetime(ds, errors="raise").sort_values()
    start, end = ds.iloc[0], ds.iloc[-1]
    expected = pd.date_range(start, end, freq="h")
    missing = expected.difference(ds)

    if missing.empty:
        return []

    blocks = []
    block_start = missing[0]
    prev = missing[0]
    for t in missing[1:]:
        if t - prev == pd.Timedelta(hours=1):
            prev = t
        else:
            n = int((prev - block_start).total_seconds() / 3600) + 1
            blocks.append((block_start, prev, n))
            block_start = t
            prev = t
    n = int((prev - block_start).total_seconds() / 3600) + 1
    blocks.append((block_start, prev, n))
    return blocks


def _hourly_grid_report(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "unique_id",
        "start",
        "end",
        "expected_hours",
        "actual_hours",
        "missing_hours",
        "missing_ratio",
        "n_missing_blocks",
        "largest_missing_block_hours",
        "first_missing_block_start",
        "first_missing_block_end",
    ]

    if df.empty:
        # Return an empty report with a stable schema (so callers can fail-loud cleanly)
        return pd.DataFrame(columns=cols)

    rows = []
    for uid, g in df.groupby("unique_id"):
        g = g.sort_values("ds")
        start, end = g["ds"].iloc[0], g["ds"].iloc[-1]
        expected = pd.date_range(start, end, freq="h")
        missing = expected.difference(g["ds"])
        blocks = _missing_hour_blocks(g["ds"])

        rows.append(
            {
                "unique_id": uid,
                "start": start,
                "end": end,
                "expected_hours": int(len(expected)),
                "actual_hours": int(len(g)),
                "missing_hours": int(len(missing)),
                "missing_ratio": float(len(missing) / max(len(expected), 1)),
                "n_missing_blocks": int(len(blocks)),
                "largest_missing_block_hours": int(max([b[2] for b in blocks], default=0)),
                "first_missing_block_start": blocks[0][0] if blocks else pd.NaT,
                "first_missing_block_end": blocks[0][1] if blocks else pd.NaT,
            }
        )

    rep = pd.DataFrame(rows)
    return rep.sort_values(["missing_ratio", "missing_hours"], ascending=False)


def _enforce_hourly_grid(
    df: pd.DataFrame,
    *,
    label: str,
    policy: str = "raise",  # "raise" | "drop_incomplete_series"
) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError(
            f"[{label}][GRID] Cannot enforce hourly grid: input dataframe is empty. "
            "This is upstream (fetch) failure, not a grid issue."
        )

    rep = _hourly_grid_report(df)
    if rep.empty:
        raise RuntimeError(
            f"[{label}][GRID] No series found to report on (rep empty). "
            "This indicates upstream emptiness or missing 'unique_id' groups."
        )

    worst = rep.iloc[0].to_dict()

    if worst["missing_hours"] == 0:
        return df

    print(f"[{label}][GRID] report (top):\n{rep.head(10).to_string(index=False)}")

    if policy == "drop_incomplete_series":
        bad_uids = rep.loc[rep["missing_hours"] > 0, "unique_id"].tolist()
        kept = df.loc[~df["unique_id"].isin(bad_uids)].copy()
        print(f"[{label}][GRID] policy=drop_incomplete_series dropped={bad_uids} kept_series={kept['unique_id'].nunique()}")
        if kept.empty:
            raise RuntimeError(f"[{label}][GRID] all series dropped due to missing hours")
        return kept

    worst_uid = worst["unique_id"]
    g = df[df["unique_id"] == worst_uid].sort_values("ds")
    blocks = _missing_hour_blocks(g["ds"])
    raise RuntimeError(
        f"[{label}][GRID] Missing hours detected (no imputation). "
        f"worst_unique_id={worst_uid} missing_hours={worst['missing_hours']} "
        f"missing_ratio={worst['missing_ratio']:.3f} blocks(sample)={blocks[:3]}"
    )


def _validate_hourly_grid_fail_loud(
    df: pd.DataFrame,
    *,
    max_missing_ratio: float = 0.0,
    label: str = "generation",
) -> None:
    # Keep your original basic checks:
    if df.empty:
        raise RuntimeError(f"[{label}] empty dataframe")

    bad = df["ds"].isna().sum()
    if bad:
        raise RuntimeError(f"[{label}] ds has NaT values bad={int(bad)}")

    dup = df.duplicated(subset=["unique_id", "ds"]).sum()
    if dup:
        raise RuntimeError(f"[{label}] duplicate (unique_id, ds) rows dup={int(dup)}")

    rep = _hourly_grid_report(df)
    worst = rep.iloc[0].to_dict()
    if worst["missing_ratio"] > max_missing_ratio:
        print(f"[{label}][GRID] report (top):\n{rep.head(10).to_string(index=False)}")
        worst_uid = worst["unique_id"]
        g = df[df["unique_id"] == worst_uid].sort_values("ds")
        blocks = _missing_hour_blocks(g["ds"])
        raise RuntimeError(
            f"[{label}][GRID] Missing hours detected (no imputation allowed). "
            f"unique_id={worst_uid} missing_hours={worst['missing_hours']} "
            f"missing_ratio={worst['missing_ratio']:.3f} blocks(sample)={blocks[:3]}"
        )



def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["ds"].dt.hour
    out["dow"] = out["ds"].dt.dayofweek

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)

    return out.drop(columns=["hour", "dow"])

def _infer_model_columns(cv_df: pd.DataFrame) -> list[str]:
    """
    Infer StatsForecast model prediction columns from a cross_validation dataframe.

    We treat as "model columns" those that:
      - are not core columns (unique_id, ds, cutoff, y)
      - are not interval columns like '<model>-lo-80' or '<model>-hi-95'
    """
    core = {"unique_id", "ds", "cutoff", "y"}
    cols = [c for c in cv_df.columns if c not in core]

    model_cols: set[str] = set()
    interval_pat = re.compile(r"-(lo|hi)-\d+$")
    for c in cols:
        if interval_pat.search(c):
            continue
        model_cols.add(c)

    return sorted(model_cols)


def compute_leaderboard(
    cv_df: pd.DataFrame,
    *,
    confidence_levels: tuple[int, int] = (80, 95),
) -> pd.DataFrame:
    """
    Build an aggregated leaderboard from StatsForecast cross_validation output.

    Returns columns:
      - model, rmse, mae, mape, valid_rows
      - coverage_<level> if interval columns exist
    """
    required = {"y", "unique_id", "ds", "cutoff"}
    missing = required - set(cv_df.columns)
    if missing:
        raise ValueError(f"[leaderboard] cv_df missing required columns: {sorted(missing)}")

    model_cols = _infer_model_columns(cv_df)
    if not model_cols:
        raise RuntimeError(
            f"[leaderboard] Could not infer any model prediction columns. "
            f"cv_df columns={cv_df.columns.tolist()}"
        )

    rows: list[dict[str, Any]] = []
    y_true = cv_df["y"].to_numpy()

    for m in model_cols:
        if m not in cv_df.columns:
            continue

        y_pred = cv_df[m].to_numpy()
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        valid_rows = int(valid_mask.sum())

        metrics = {
            "model": m,
            "rmse": float(ForecastMetrics.rmse(y_true, y_pred)),
            "mae": float(ForecastMetrics.mae(y_true, y_pred)),
            "mape": float(ForecastMetrics.mape(y_true, y_pred)),
            "valid_rows": valid_rows,
        }

        # Coverage if interval columns exist
        for lvl in confidence_levels:
            lo_col = f"{m}-lo-{lvl}"
            hi_col = f"{m}-hi-{lvl}"
            if lo_col in cv_df.columns and hi_col in cv_df.columns:
                cov = ForecastMetrics.coverage(
                    y_true,
                    cv_df[lo_col].to_numpy(),
                    cv_df[hi_col].to_numpy(),
                )
                metrics[f"coverage_{lvl}"] = float(cov)

        rows.append(metrics)

    lb = pd.DataFrame(rows)
    if lb.empty:
        raise RuntimeError("[leaderboard] computed empty leaderboard (no usable model columns).")

    # Fail-loud sorting: rmse NaNs should sort last
    lb = lb.sort_values(["rmse"], ascending=True, na_position="last").reset_index(drop=True)
    return lb


def compute_baseline_metrics(
    cv_df: pd.DataFrame,
    *,
    model_name: str,
    threshold_k: float = 2.0,
) -> dict:
    """
    Compute baseline metrics for drift detection from CV output.

    We compute RMSE/MAE per (unique_id, cutoff) window, then aggregate:
      rmse_mean, rmse_std, drift_threshold_rmse = mean + k*std

    No imputation/filling: metrics are computed only from finite values.
    """
    required = {"unique_id", "cutoff", "y", model_name}
    missing = required - set(cv_df.columns)
    if missing:
        raise ValueError(
            f"[baseline] cv_df missing required columns for model '{model_name}': {sorted(missing)}"
        )

    # Compute per-window metrics (unique_id, cutoff)
    def _window_metrics(g: pd.DataFrame) -> pd.Series:
        yt = g["y"].to_numpy()
        yp = g[model_name].to_numpy()
        valid = np.isfinite(yt) & np.isfinite(yp)
        if valid.sum() == 0:
            return pd.Series({"rmse": np.nan, "mae": np.nan, "valid_rows": 0})
        return pd.Series({
            "rmse": ForecastMetrics.rmse(yt, yp),
            "mae": ForecastMetrics.mae(yt, yp),
            "valid_rows": int(valid.sum()),
        })

    per_window = (
        cv_df.groupby(["unique_id", "cutoff"], sort=False, dropna=False)
        .apply(_window_metrics)
        .reset_index()
    )

    # Fail loud if baseline is entirely NaN
    if per_window["rmse"].notna().sum() == 0:
        sample_cols = ["unique_id", "cutoff", "y", model_name]
        raise RuntimeError(
            "[baseline] All per-window RMSE are NaN. "
            "This usually means predictions or y are non-finite everywhere. "
            f"Sample:\n{cv_df[sample_cols].head(20).to_string(index=False)}"
        )

    rmse_mean = float(per_window["rmse"].mean(skipna=True))
    rmse_std = float(per_window["rmse"].std(skipna=True, ddof=0))
    mae_mean = float(per_window["mae"].mean(skipna=True))
    mae_std = float(per_window["mae"].std(skipna=True, ddof=0))

    baseline = {
        "model": model_name,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "drift_threshold_rmse": float(rmse_mean + threshold_k * rmse_std),
        "drift_threshold_mae": float(mae_mean + threshold_k * mae_std),
        "n_series": int(per_window["unique_id"].nunique()),
        "n_windows": int(per_window["cutoff"].nunique()),
        "per_window_rows": int(len(per_window)),
    }

    # Optional per-series baseline (useful later if you want drift per series)
    per_series = (
        per_window.groupby("unique_id")[["rmse", "mae"]]
        .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", lambda s: s.std(ddof=0)),
             mae_mean=("mae", "mean"), mae_std=("mae", lambda s: s.std(ddof=0)))
        .reset_index()
    )
    per_series["drift_threshold_rmse"] = per_series["rmse_mean"] + threshold_k * per_series["rmse_std"]
    per_series["drift_threshold_mae"] = per_series["mae_mean"] + threshold_k * per_series["mae_std"]
    baseline["per_series"] = per_series.to_dict(orient="records")

    return baseline



@dataclass
class ForecastConfig:
    horizon: int = 24
    confidence_levels: tuple[int, int] = (80, 95)


class RenewableForecastModel:
    def __init__(self, horizon: int = 24, confidence_levels: tuple[int, int] = (80, 95)):
        self.horizon = horizon
        self.confidence_levels = confidence_levels
        self.sf = None
        self._train_df = None  # contains y + exog columns
        self._exog_cols: list[str] = []
        self.fitted = False

    def prepare_training_df(self, df: pd.DataFrame, weather_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        req = {"unique_id", "ds", "y"}
        if not req.issubset(df.columns):
            raise ValueError(f"generation df missing cols={sorted(req - set(df.columns))}")

        if df.empty:
            raise RuntimeError(
                "[generation] Empty generation dataframe passed into modeling. "
                "This is upstream (EIA fetch/cache) failure — inspect fetch_diagnostics and fetch_generation logs."
            )

        work = df.copy()
        work["ds"] = pd.to_datetime(work["ds"], errors="raise")
        work = work.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        y_null = work["y"].isna()
        if y_null.any():
            sample = work.loc[y_null, ["unique_id", "ds", "y"]].head(25)
            raise RuntimeError(
                f"[generation][Y] Found null y values (no imputation). rows={int(y_null.sum())}. "
                f"Sample:\n{sample.to_string(index=False)}"
            )

        work = _enforce_hourly_grid(work, label="generation", policy="drop_incomplete_series")
        work = _add_time_features(work)

        if weather_df is not None and not weather_df.empty:
            if not {"ds", "region"}.issubset(weather_df.columns):
                raise ValueError("weather_df must have columns ['ds','region', ...]")

            work["region"] = work["unique_id"].str.split("_").str[0]

            wcols = [c for c in WEATHER_VARS if c in weather_df.columns]
            if not wcols:
                raise ValueError("weather_df has none of expected WEATHER_VARS")

            merged = work.merge(
                weather_df[["ds", "region"] + wcols],
                on=["ds", "region"],
                how="left",
                validate="many_to_one",
            )

            missing_any = merged[wcols].isna().any(axis=1)
            if missing_any.any():
                sample = merged.loc[missing_any, ["unique_id", "ds", "region"] + wcols].head(10)
                raise RuntimeError(
                    f"[weather][ALIGN] Missing weather after merge rows={int(missing_any.sum())}. "
                    f"Sample:\n{sample.to_string(index=False)}"
                )

            work = merged.drop(columns=["region"])
            self._exog_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"] + wcols
        else:
            self._exog_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

        return work



    def fit(self, df: pd.DataFrame, weather_df: Optional[pd.DataFrame] = None) -> None:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, SeasonalNaive, AutoETS, MSTL

        train_df = self.prepare_training_df(df, weather_df)

        models = [
            AutoARIMA(season_length=24),
            SeasonalNaive(season_length=24),
            AutoETS(season_length=24),
            MSTL(season_length=[24, 168], trend_forecaster=AutoARIMA(), alias="MSTL_ARIMA"),
        ]

        self.sf = StatsForecast(models=models, freq="h", n_jobs=-1)
        self._train_df = train_df
        self.fitted = True

        print(f"[fit] rows={len(train_df)} series={train_df['unique_id'].nunique()} exog_cols={self._exog_cols}")

    def build_future_X_df(self, future_weather: pd.DataFrame) -> pd.DataFrame:
        """
        Build future X_df for forecast horizon using forecast weather.
        Must include: unique_id, ds, and exactly the exog columns used in training.
        """
        if not self.fitted:
            raise RuntimeError("fit() first")

        if future_weather is None or future_weather.empty:
            raise RuntimeError("future_weather required to forecast with regressors (no fabrication).")

        if not {"ds", "region"}.issubset(future_weather.columns):
            raise ValueError("future_weather must have columns ['ds','region', ...]")

        # Create the future ds grid per series
        last_ds = self._train_df.groupby("unique_id")["ds"].max()
        frames = []
        for uid, end in last_ds.items():
            future_ds = pd.date_range(end + pd.Timedelta(hours=1), periods=self.horizon, freq="h")
            frames.append(pd.DataFrame({"unique_id": uid, "ds": future_ds}))
        X = pd.concat(frames, ignore_index=True)

        X = _add_time_features(X)
        X["region"] = X["unique_id"].str.split("_").str[0]

        wcols = [c for c in WEATHER_VARS if c in future_weather.columns]
        X = X.merge(
            future_weather[["ds", "region"] + wcols],
            on=["ds", "region"],
            how="left",
            validate="many_to_one",
        )

        # Fail loud on missing future regressors
        needed = [c for c in self._exog_cols if c not in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]]  # weather cols
        if needed:
            missing_any = X[needed].isna().any(axis=1)
            if missing_any.any():
                sample = X.loc[missing_any, ["unique_id", "ds", "region"] + needed].head(10)
                raise RuntimeError(
                    f"[future_weather][ALIGN] Missing future weather rows={int(missing_any.sum())}. "
                    f"Sample:\n{sample.to_string(index=False)}"
                )

        X = X.drop(columns=["region"])
        keep = ["unique_id", "ds"] + self._exog_cols
        return X[keep].sort_values(["unique_id", "ds"]).reset_index(drop=True)

    def predict(self, future_weather: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("fit() first")

        X_df = self.build_future_X_df(future_weather)

        # IMPORTANT: If you fit models using exogenous regressors, you must supply X_df at forecast time.
        fcst = self.sf.forecast(
            h=self.horizon,
            df=self._train_df,
            X_df=X_df,
            level=list(self.confidence_levels),
        ).reset_index()

        return fcst

    def cross_validate(
        self,
        df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
        n_windows: int = 3,
        step_size: int = 168,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, SeasonalNaive, AutoETS, MSTL

        train_df = self.prepare_training_df(df, weather_df)

        models = [
            AutoARIMA(season_length=24),
            SeasonalNaive(season_length=24),
            AutoETS(season_length=24),
            MSTL(season_length=[24, 168], trend_forecaster=AutoARIMA(), alias="MSTL_ARIMA"),
        ]
        sf = StatsForecast(models=models, freq="h", n_jobs=-1)

        print(
            f"[cv] windows={n_windows} step={step_size} h={self.horizon} "
            f"rows={len(train_df)} series={train_df['unique_id'].nunique()}"
        )

        cv = sf.cross_validation(
            df=train_df,
            h=self.horizon,
            step_size=step_size,
            n_windows=n_windows,
            level=list(self.confidence_levels),
        ).reset_index()

        leaderboard = compute_leaderboard(cv, confidence_levels=self.confidence_levels)
        return cv, leaderboard



if __name__ == "__main__":
    # REAL EXAMPLE: multi-series WND with strict gates and CV

    from src.renewable.eia_renewable import EIARenewableFetcher
    from src.renewable.open_meteo import OpenMeteoRenewable

    regions = ["CALI", "ERCO", "MISO"]
    fuel = "WND"
    start_date = "2024-11-01"
    end_date = "2024-12-15"

    fetcher = EIARenewableFetcher(debug_env=True)
    gen = fetcher.fetch_all_regions(fuel, start_date, end_date, regions=regions)
    _log_series_summary(gen, label="generation_raw")

    weather_api = OpenMeteoRenewable(strict=True)
    wx_hist = weather_api.fetch_all_regions_historical(regions, start_date, end_date, debug=True)

    model = RenewableForecastModel(horizon=24, confidence_levels=(80, 95))

    # CV (historical): regressors live in df, no filling allowed
    cv = model.cross_validate(gen, weather_df=wx_hist, n_windows=3, step_size=168)
    print(cv.head().to_string(index=False))

    # Optional: fit + forecast next 24h using forecast weather (no leakage)
    # wx_future = weather_api.fetch_all_regions_forecast(regions, horizon_hours=48, debug=True)
    # model.fit(gen, weather_df=wx_hist)
    # fcst = model.predict(future_weather=wx_future)
    # print(fcst.head().to_string(index=False))
