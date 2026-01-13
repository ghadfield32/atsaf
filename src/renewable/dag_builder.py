"""Renewable pipeline DAG builder for Airflow."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except Exception:
    AIRFLOW_AVAILABLE = False
    DAG = None
    PythonOperator = None


DEFAULT_ARGS = {
    "owner": "data-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def build_hourly_dag(
    dag_id: str = "renewable_hourly_pipeline",
    schedule: str = "17 * * * *",
    start_date: Optional[datetime] = None,
    default_args: Optional[Dict[str, Any]] = None,
) -> "DAG":
    if not AIRFLOW_AVAILABLE:
        raise ImportError("Airflow is not installed. Install apache-airflow to use build_hourly_dag().")

    from src.renewable.jobs.run_hourly import run_hourly_pipeline

    if start_date is None:
        start_date = datetime.utcnow() - timedelta(days=1)
    if default_args is None:
        default_args = DEFAULT_ARGS.copy()

    with DAG(
        dag_id=dag_id,
        default_args=default_args,
        description="Renewable hourly pipeline",
        schedule_interval=schedule,
        start_date=start_date,
        catchup=False,
        max_active_runs=1,
        tags=["renewable", "eia", "forecasting"],
    ) as dag:
        PythonOperator(
            task_id="run_hourly",
            python_callable=run_hourly_pipeline,
        )

    return dag


def build_dag_dot() -> str:
    return """digraph RENEWABLE_PIPELINE {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#e8f5e9"];

  run_hourly;
}"""
