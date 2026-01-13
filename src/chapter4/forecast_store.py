# file: src/chapter4/forecast_store.py
from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd
from src.chapter4.db import connect

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def persist_forecasts(
    db_path: str,
    run_id: str,
    forecast_df: pd.DataFrame,
    confidence_level: int = 95
) -> None:
    """
    Expects StatsForecast output:
      columns include: unique_id, ds, <model1>, <model1>-lo-95, <model1>-hi-95, ...
    Writes long format into SQLite.
    """
    required = {"unique_id", "ds"}
    if not required.issubset(forecast_df.columns):
        raise ValueError(f"forecast_df missing {required}, got {forecast_df.columns.tolist()}")

    df = forecast_df.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="raise", utc=True).dt.tz_localize(None)

    # Model columns = non-metadata and not interval cols
    model_cols = [
        c for c in df.columns
        if c not in ("unique_id", "ds")
        and not c.endswith(f"-lo-{confidence_level}")
        and not c.endswith(f"-hi-{confidence_level}")
    ]

    rows = []
    created_ts = _utc_iso()

    for m in model_cols:
        lo_col = f"{m}-lo-{confidence_level}"
        hi_col = f"{m}-hi-{confidence_level}"
        has_int = (lo_col in df.columns) and (hi_col in df.columns)

        tmp = df[["unique_id", "ds", m]].rename(columns={m: "yhat"})
        tmp["model"] = m
        tmp["lo"] = df[lo_col] if has_int else None
        tmp["hi"] = df[hi_col] if has_int else None
        tmp["run_id"] = run_id
        tmp["created_ts_utc"] = created_ts

        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)

    con = connect(db_path)
    con.executemany("""
        INSERT OR REPLACE INTO forecasts
        (run_id, created_ts_utc, unique_id, ds, model, yhat, lo, hi)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (
            r.run_id,
            r.created_ts_utc,
            r.unique_id,
            str(r.ds),
            r.model,
            None if pd.isna(r.yhat) else float(r.yhat),
            None if pd.isna(r.lo) else float(r.lo),
            None if pd.isna(r.hi) else float(r.hi),
        )
        for r in out.itertuples(index=False)
    ])
    con.commit()
    con.close()
