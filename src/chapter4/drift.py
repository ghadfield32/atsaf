# file: src/chapter4/drift.py
from __future__ import annotations
import json
from datetime import datetime, timezone
import pandas as pd
from src.chapter4.db import connect

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def compute_drift_threshold_from_backtest(
    leaderboard_path: str,
    metric_col: str = "mape_mean",
    std_col: str = "mape_std",
    k: float = 2.0
) -> float:
    lb = pd.read_parquet(leaderboard_path)
    if lb.empty:
        raise ValueError("Leaderboard is empty; cannot compute drift threshold.")
    best = lb.iloc[0]
    if metric_col not in lb.columns or std_col not in lb.columns:
        raise ValueError(f"Expected {metric_col} and {std_col} in leaderboard columns: {lb.columns.tolist()}")
    return float(best[metric_col] + k * best[std_col])

def rolling_accuracy(
    db_path: str,
    model: str,
    unique_id: str,
    horizon_hours: int = 24
) -> pd.DataFrame:
    con = connect(db_path)
    df = pd.read_sql_query("""
        SELECT scored_ts_utc, run_id, model, unique_id, horizon_hours, rmse, mape, coverage_pct, valid_rows
        FROM forecast_scores
        WHERE model = ? AND unique_id = ? AND horizon_hours = ?
        ORDER BY scored_ts_utc ASC
    """, con, params=(model, unique_id, int(horizon_hours)))
    con.close()

    if df.empty:
        return df

    df["scored_ts_utc"] = pd.to_datetime(df["scored_ts_utc"], errors="raise", utc=True)
    return df

def detect_drift(
    db_path: str,
    leaderboard_path: str,
    model: str,
    unique_id: str,
    horizon_hours: int,
    k: float = 2.0,
) -> dict:
    thr = compute_drift_threshold_from_backtest(leaderboard_path, k=k)
    hist = rolling_accuracy(db_path, model=model, unique_id=unique_id, horizon_hours=horizon_hours)

    if hist.empty:
        return {"status": "no_data", "threshold": thr}

    latest = hist.iloc[-1]
    drifted = (pd.notna(latest["mape"]) and float(latest["mape"]) > thr)

    return {
        "status": "drift" if drifted else "ok",
        "threshold": float(thr),
        "latest_mape": None if pd.isna(latest["mape"]) else float(latest["mape"]),
        "latest_rmse": None if pd.isna(latest["rmse"]) else float(latest["rmse"]),
        "latest_scored_ts": str(latest["scored_ts_utc"]),
        "model": model,
        "unique_id": unique_id,
        "horizon_hours": int(horizon_hours),
    }

def write_alert(db_path: str, alert_type: str, severity: str, message: str, metadata: dict) -> None:
    con = connect(db_path)
    con.execute("""
        INSERT INTO alerts (alert_ts_utc, alert_type, severity, message, metadata_json)
        VALUES (?, ?, ?, ?, ?)
    """, (_utc_iso(), alert_type, severity, message, json.dumps(metadata)))
    con.commit()
    con.close()
