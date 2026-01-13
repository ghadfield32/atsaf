# file: src/chapter4/db.py
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Tuple

def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_db(db_path: str) -> None:
    con = connect(db_path)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id TEXT PRIMARY KEY,
        ts_utc TEXT NOT NULL,
        status TEXT NOT NULL,
        step TEXT NOT NULL,
        message TEXT,
        raw_rows INTEGER,
        clean_rows INTEGER,
        duration_sec REAL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS forecasts (
        run_id TEXT NOT NULL,
        created_ts_utc TEXT NOT NULL,
        unique_id TEXT NOT NULL,
        ds TEXT NOT NULL,
        model TEXT NOT NULL,
        yhat REAL,
        lo REAL,
        hi REAL,
        PRIMARY KEY (run_id, model, unique_id, ds)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS forecast_scores (
        scored_ts_utc TEXT NOT NULL,
        run_id TEXT NOT NULL,
        model TEXT NOT NULL,
        unique_id TEXT NOT NULL,
        horizon_hours INTEGER NOT NULL,
        rmse REAL,
        mape REAL,
        coverage_pct REAL,
        valid_rows INTEGER,
        PRIMARY KEY (run_id, model, unique_id, horizon_hours)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        alert_ts_utc TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        message TEXT NOT NULL,
        metadata_json TEXT
    );
    """)

    con.commit()
    con.close()
