# file: src/chapter4/run_log.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from src.chapter4.db import connect

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log_run(
    db_path: str,
    run_id: str,
    status: str,
    step: str,
    message: Optional[str] = None,
    raw_rows: Optional[int] = None,
    clean_rows: Optional[int] = None,
    duration_sec: Optional[float] = None,
) -> None:
    con = connect(db_path)
    con.execute("""
        INSERT OR REPLACE INTO pipeline_runs
        (run_id, ts_utc, status, step, message, raw_rows, clean_rows, duration_sec)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (run_id, now_utc_iso(), status, step, message, raw_rows, clean_rows, duration_sec))
    con.commit()
    con.close()
