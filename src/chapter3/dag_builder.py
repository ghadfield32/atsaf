# file: src/chapter3/dag_builder.py
"""
Chapter 3: DAG Builder

- If Airflow is installed: returns a real Airflow DAG
- If not: provides a DOT graph string for notebook visualization
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from src.chapter3.config import PipelineConfig

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
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


def build_daily_dag(
    dag_id: str = "eia_daily_pipeline",
    schedule: str = "0 6 * * *",
    start_date: Optional[datetime] = None,
    default_args: Optional[Dict[str, Any]] = None,
    config: Optional[PipelineConfig] = None,
) -> "DAG":
    if not AIRFLOW_AVAILABLE:
        raise ImportError("Airflow is not installed. Install apache-airflow to use build_daily_dag().")

    from src.chapter3.tasks import (
        ingest_eia,
        prepare_clean,
        validate_clean,
        train_backtest_select,
        forecast_publish,
        register_champion,
    )

    if start_date is None:
        start_date = datetime.utcnow() - timedelta(days=1)
    if default_args is None:
        default_args = DEFAULT_ARGS.copy()
    if config is None:
        config = PipelineConfig()

    with DAG(
        dag_id=dag_id,
        default_args=default_args,
        description="EIA Forecasting Pipeline",
        schedule_interval=schedule,
        start_date=start_date,
        catchup=False,
        tags=["eia", "forecasting"],
    ) as dag:

        t_ingest = PythonOperator(
            task_id="ingest",
            python_callable=lambda: ingest_eia(config),
        )

        t_prepare = PythonOperator(
            task_id="prepare",
            python_callable=lambda: prepare_clean(str(config.raw_path()), config),
        )

        t_validate = PythonOperator(
            task_id="validate",
            python_callable=lambda: validate_clean(str(config.clean_path())),
        )

        def _train():
            lb = train_backtest_select(str(config.clean_path()), config)
            # returning a DataFrame is not ideal for XCom; write path is the artifact
            return str(config.leaderboard_path())

        t_train = PythonOperator(task_id="train", python_callable=_train)

        def _register():
            import pandas as pd
            lb = pd.read_parquet(config.leaderboard_path())
            return register_champion(lb, config, str(config.clean_path()))


        t_register = PythonOperator(task_id="register", python_callable=_register)

        t_forecast = PythonOperator(
            task_id="forecast",
            python_callable=lambda: forecast_publish(str(config.clean_path()), config),
        )

        t_ingest >> t_prepare >> t_validate >> t_train >> t_register >> t_forecast

    return dag


def build_dag_dot() -> str:
    """
    DOT graph fallback (works without Airflow) for notebook visualization.
    """
    return """digraph EIA_PIPELINE {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#eef2ff"];

  ingest -> prepare -> validate -> train -> register -> forecast;
}"""
