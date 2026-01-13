"""Database schema and operations for renewable forecasting.

Extends the Chapter 4 monitoring database with:
- Prediction intervals (80%, 95%)
- Weather features table
- Renewable-specific columns (fuel_type, region)
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def connect(db_path: str) -> sqlite3.Connection:
    """Connect to SQLite database with optimized settings."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def init_renewable_db(db_path: str) -> None:
    """Initialize renewable forecasting database schema.

    Creates tables:
    - renewable_forecasts: Forecasts with dual intervals
    - renewable_scores: Evaluation metrics with coverage
    - weather_features: Weather data by region
    - drift_alerts: Drift detection history
    - baseline_metrics: Backtest baselines for drift thresholds
    """
    con = connect(db_path)
    cur = con.cursor()

    # Forecasts with dual prediction intervals
    cur.execute("""
    CREATE TABLE IF NOT EXISTS renewable_forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        unique_id TEXT NOT NULL,
        region TEXT NOT NULL,
        fuel_type TEXT NOT NULL,
        ds TEXT NOT NULL,
        model TEXT NOT NULL,
        yhat REAL,
        yhat_lo_80 REAL,
        yhat_hi_80 REAL,
        yhat_lo_95 REAL,
        yhat_hi_95 REAL,
        UNIQUE (run_id, model, unique_id, ds)
    );
    """)

    # Index for efficient queries
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_forecasts_region_ds
    ON renewable_forecasts (region, ds);
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_forecasts_fuel_ds
    ON renewable_forecasts (fuel_type, ds);
    """)

    # Evaluation scores with dual coverage
    cur.execute("""
    CREATE TABLE IF NOT EXISTS renewable_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scored_at TEXT NOT NULL,
        run_id TEXT NOT NULL,
        unique_id TEXT NOT NULL,
        region TEXT NOT NULL,
        fuel_type TEXT NOT NULL,
        model TEXT NOT NULL,
        horizon_hours INTEGER NOT NULL,
        rmse REAL,
        mae REAL,
        coverage_80 REAL,
        coverage_95 REAL,
        valid_rows INTEGER,
        UNIQUE (run_id, model, unique_id, horizon_hours)
    );
    """)

    # Weather features by region
    cur.execute("""
    CREATE TABLE IF NOT EXISTS weather_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        region TEXT NOT NULL,
        ds TEXT NOT NULL,
        temperature_2m REAL,
        wind_speed_10m REAL,
        wind_speed_100m REAL,
        wind_direction_10m REAL,
        direct_radiation REAL,
        diffuse_radiation REAL,
        cloud_cover REAL,
        is_forecast INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (region, ds, is_forecast)
    );
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_weather_region_ds
    ON weather_features (region, ds);
    """)

    # Drift detection alerts
    cur.execute("""
    CREATE TABLE IF NOT EXISTS drift_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_at TEXT NOT NULL,
        run_id TEXT,
        unique_id TEXT,
        region TEXT,
        fuel_type TEXT,
        alert_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        current_rmse REAL,
        threshold_rmse REAL,
        message TEXT,
        metadata_json TEXT
    );
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_drift_alerts_time
    ON drift_alerts (alert_at);
    """)

    # Baseline metrics for drift detection
    cur.execute("""
    CREATE TABLE IF NOT EXISTS baseline_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        unique_id TEXT NOT NULL,
        model TEXT NOT NULL,
        rmse_mean REAL NOT NULL,
        rmse_std REAL NOT NULL,
        mae_mean REAL,
        mae_std REAL,
        drift_threshold_rmse REAL NOT NULL,
        drift_threshold_mae REAL,
        n_windows INTEGER,
        metadata_json TEXT,
        UNIQUE (unique_id, model)
    );
    """)

    con.commit()
    con.close()


def save_forecasts(
    db_path: str,
    forecasts_df: pd.DataFrame,
    run_id: str,
    model: str = "MSTL_ARIMA",
) -> int:
    """Save forecasts to database.

    Args:
        db_path: Path to SQLite database
        forecasts_df: DataFrame with [unique_id, ds, yhat, yhat_lo_80, ...]
        run_id: Pipeline run identifier
        model: Model name

    Returns:
        Number of rows inserted
    """
    con = connect(db_path)
    created_at = datetime.utcnow().isoformat()

    rows = []
    for _, row in forecasts_df.iterrows():
        unique_id = row["unique_id"]
        parts = unique_id.split("_")
        region = parts[0] if len(parts) > 0 else ""
        fuel_type = parts[1] if len(parts) > 1 else ""

        rows.append((
            run_id,
            created_at,
            unique_id,
            region,
            fuel_type,
            str(row["ds"]),
            model,
            row.get("yhat"),
            row.get("yhat_lo_80"),
            row.get("yhat_hi_80"),
            row.get("yhat_lo_95"),
            row.get("yhat_hi_95"),
        ))

    cur = con.cursor()
    cur.executemany("""
        INSERT OR REPLACE INTO renewable_forecasts
        (run_id, created_at, unique_id, region, fuel_type, ds, model,
         yhat, yhat_lo_80, yhat_hi_80, yhat_lo_95, yhat_hi_95)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    con.commit()
    con.close()

    return len(rows)


def save_weather(
    db_path: str,
    weather_df: pd.DataFrame,
    is_forecast: bool = False,
) -> int:
    """Save weather features to database.

    Args:
        db_path: Path to SQLite database
        weather_df: DataFrame with [ds, region, weather_vars...]
        is_forecast: True if this is forecast weather data

    Returns:
        Number of rows inserted
    """
    con = connect(db_path)

    weather_cols = [
        "temperature_2m", "wind_speed_10m", "wind_speed_100m",
        "wind_direction_10m", "direct_radiation", "diffuse_radiation", "cloud_cover"
    ]

    rows = []
    for _, row in weather_df.iterrows():
        values = [row.get(col) for col in weather_cols]
        rows.append((
            row["region"],
            str(row["ds"]),
            *values,
            1 if is_forecast else 0,
        ))

    cur = con.cursor()
    cur.executemany(f"""
        INSERT OR REPLACE INTO weather_features
        (region, ds, {', '.join(weather_cols)}, is_forecast)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    con.commit()
    con.close()

    return len(rows)


def save_drift_alert(
    db_path: str,
    run_id: str,
    unique_id: str,
    current_rmse: float,
    threshold_rmse: float,
    severity: str = "warning",
    metadata: Optional[dict] = None,
) -> None:
    """Save drift detection alert.

    Args:
        db_path: Path to SQLite database
        run_id: Pipeline run identifier
        unique_id: Series identifier
        current_rmse: Current RMSE value
        threshold_rmse: Drift threshold
        severity: Alert severity (info, warning, critical)
        metadata: Additional metadata
    """
    con = connect(db_path)

    parts = unique_id.split("_")
    region = parts[0] if len(parts) > 0 else ""
    fuel_type = parts[1] if len(parts) > 1 else ""

    alert_type = "drift_detected" if current_rmse > threshold_rmse else "drift_check"
    message = (
        f"RMSE {current_rmse:.1f} {'>' if current_rmse > threshold_rmse else '<='} "
        f"threshold {threshold_rmse:.1f}"
    )

    cur = con.cursor()
    cur.execute("""
        INSERT INTO drift_alerts
        (alert_at, run_id, unique_id, region, fuel_type, alert_type, severity,
         current_rmse, threshold_rmse, message, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        run_id,
        unique_id,
        region,
        fuel_type,
        alert_type,
        severity,
        current_rmse,
        threshold_rmse,
        message,
        json.dumps(metadata) if metadata else None,
    ))

    con.commit()
    con.close()


def save_baseline(
    db_path: str,
    unique_id: str,
    model: str,
    baseline: dict,
) -> None:
    """Save baseline metrics for drift detection.

    Args:
        db_path: Path to SQLite database
        unique_id: Series identifier
        model: Model name
        baseline: Baseline metrics dictionary
    """
    con = connect(db_path)
    cur = con.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO baseline_metrics
        (created_at, unique_id, model, rmse_mean, rmse_std, mae_mean, mae_std,
         drift_threshold_rmse, drift_threshold_mae, n_windows, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        unique_id,
        model,
        baseline.get("rmse_mean"),
        baseline.get("rmse_std"),
        baseline.get("mae_mean"),
        baseline.get("mae_std"),
        baseline.get("drift_threshold_rmse"),
        baseline.get("drift_threshold_mae"),
        baseline.get("n_windows"),
        json.dumps(baseline),
    ))

    con.commit()
    con.close()


def get_recent_forecasts(
    db_path: str,
    region: Optional[str] = None,
    fuel_type: Optional[str] = None,
    hours: int = 48,
) -> pd.DataFrame:
    """Get recent forecasts from database.

    Args:
        db_path: Path to SQLite database
        region: Filter by region (optional)
        fuel_type: Filter by fuel type (optional)
        hours: Hours of history to retrieve

    Returns:
        DataFrame with forecasts
    """
    con = connect(db_path)

    query = """
        SELECT *
        FROM renewable_forecasts
        WHERE datetime(created_at) > datetime('now', ?)
    """
    params = [f"-{hours} hours"]

    if region:
        query += " AND region = ?"
        params.append(region)

    if fuel_type:
        query += " AND fuel_type = ?"
        params.append(fuel_type)

    query += " ORDER BY ds DESC"

    df = pd.read_sql_query(query, con, params=params)
    con.close()

    return df


def get_drift_alerts(
    db_path: str,
    hours: int = 24,
    severity: Optional[str] = None,
) -> pd.DataFrame:
    """Get recent drift alerts.

    Args:
        db_path: Path to SQLite database
        hours: Hours of history
        severity: Filter by severity (optional)

    Returns:
        DataFrame with alerts
    """
    con = connect(db_path)

    query = """
        SELECT *
        FROM drift_alerts
        WHERE datetime(alert_at) > datetime('now', ?)
    """
    params = [f"-{hours} hours"]

    if severity:
        query += " AND severity = ?"
        params.append(severity)

    query += " ORDER BY alert_at DESC"

    df = pd.read_sql_query(query, con, params=params)
    con.close()

    return df


if __name__ == "__main__":
    # Test database initialization
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/test_renewable.db"

        print("Initializing database...")
        init_renewable_db(db_path)

        print("Database initialized successfully!")

        # Test connection
        con = connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cur.fetchall()
        print(f"Tables created: {[t[0] for t in tables]}")
        con.close()
