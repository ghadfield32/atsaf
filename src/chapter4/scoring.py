# file: src/chapter4/scoring.py
from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from src.chapter4.db import connect

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def score_forecasts(
    db_path: str,
    actuals_df: pd.DataFrame,
    max_horizon_hours: int = 72
) -> pd.DataFrame:
    """
    actuals_df must be statsforecast format: unique_id, ds, y (timezone-naive UTC preferred)

    Scores all forecasts in DB that match actuals_df on (unique_id, ds).
    Aggregates per (run_id, model, unique_id, horizon_hours).

    horizon_hours is computed as ds - forecast_created_ts (rounded to hours).
    """
    required = {"unique_id", "ds", "y"}
    if not required.issubset(actuals_df.columns):
        raise ValueError(f"actuals_df missing {required}, got {actuals_df.columns.tolist()}")

    act = actuals_df.copy()
    act["ds"] = pd.to_datetime(act["ds"], errors="raise", utc=True).dt.tz_localize(None)

    con = connect(db_path)
    fc = pd.read_sql_query("SELECT * FROM forecasts", con)
    con.close()

    if fc.empty:
        return pd.DataFrame()

    fc["ds"] = pd.to_datetime(fc["ds"], errors="raise")
    fc["created_ts_utc"] = pd.to_datetime(fc["created_ts_utc"], errors="raise", utc=True).dt.tz_localize(None)

    merged = fc.merge(act, on=["unique_id", "ds"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["horizon_hours"] = ((merged["ds"] - merged["created_ts_utc"]).dt.total_seconds() / 3600.0).round().astype(int)
    merged = merged[(merged["horizon_hours"] >= 1) & (merged["horizon_hours"] <= max_horizon_hours)]

    def rmse(y, yhat):
        m = np.isfinite(y) & np.isfinite(yhat)
        if m.sum() == 0:
            return np.nan, 0
        return float(np.sqrt(np.mean((y[m] - yhat[m]) ** 2))), int(m.sum())

    def mape(y, yhat):
        m = np.isfinite(y) & np.isfinite(yhat) & (np.abs(y) > 1e-12)
        if m.sum() == 0:
            return np.nan
        return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])))

    def coverage(y, lo, hi):
        m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        if m.sum() == 0:
            return np.nan
        return float(100.0 * np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))

    rows = []
    for (run_id, model, uid, h), g in merged.groupby(["run_id", "model", "unique_id", "horizon_hours"]):
        y = g["y"].to_numpy()
        yhat = g["yhat"].to_numpy()
        lo = g["lo"].to_numpy()
        hi = g["hi"].to_numpy()

        r, n = rmse(y, yhat)
        rows.append({
            "scored_ts_utc": _utc_iso(),
            "run_id": run_id,
            "model": model,
            "unique_id": uid,
            "horizon_hours": int(h),
            "rmse": r,
            "mape": mape(y, yhat),
            "coverage_pct": coverage(y, lo, hi),
            "valid_rows": n,
        })

    scores = pd.DataFrame(rows)

    con = connect(db_path)
    con.executemany("""
        INSERT OR REPLACE INTO forecast_scores
        (scored_ts_utc, run_id, model, unique_id, horizon_hours, rmse, mape, coverage_pct, valid_rows)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (
            r.scored_ts_utc, r.run_id, r.model, r.unique_id, r.horizon_hours,
            None if pd.isna(r.rmse) else float(r.rmse),
            None if pd.isna(r.mape) else float(r.mape),
            None if pd.isna(r.coverage_pct) else float(r.coverage_pct),
            int(r.valid_rows),
        )
        for r in scores.itertuples(index=False)
    ])
    con.commit()
    con.close()

    return scores
