from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from .config import AQIPipelineConfig

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
    dag_id: str = "aqi_daily_pipeline",
    schedule: str = "0 6 * * *",
    start_date: Optional[datetime] = None,
    default_args: Optional[Dict[str, Any]] = None,
    config: Optional[AQIPipelineConfig] = None,
) -> "DAG":
    if not AIRFLOW_AVAILABLE:
        raise ImportError("Airflow is not installed. Install apache-airflow to use build_daily_dag().")

    from .tasks import (
        ingest_airnow,
        ingest_weather,
        merge_features,
        prepare_aqi,
        prepare_weather_data,
        forecast_publish,
        train_backtest_select,
        validate_aqi,
    )

    if start_date is None:
        start_date = datetime.utcnow() - timedelta(days=1)
    if default_args is None:
        default_args = DEFAULT_ARGS.copy()
    if config is None:
        config = AQIPipelineConfig()

    with DAG(
        dag_id=dag_id,
        default_args=default_args,
        description="AQI Forecasting Pipeline",
        schedule_interval=schedule,
        start_date=start_date,
        catchup=False,
        tags=["aqi", "forecasting"],
    ) as dag:

        t_airnow = PythonOperator(
            task_id="ingest_airnow",
            python_callable=lambda: ingest_airnow(config),
        )

        t_weather = PythonOperator(
            task_id="ingest_weather",
            python_callable=lambda: ingest_weather(config),
        )

        t_prepare_aqi = PythonOperator(
            task_id="prepare_aqi",
            python_callable=lambda: prepare_aqi(str(config.raw_airnow_path()), config),
        )

        t_prepare_weather = PythonOperator(
            task_id="prepare_weather",
            python_callable=lambda: prepare_weather_data(str(config.raw_weather_path()), config),
        )

        t_merge = PythonOperator(
            task_id="merge_features",
            python_callable=lambda: merge_features(
                str(config.clean_aqi_path()),
                str(config.clean_weather_path()),
                config,
            ),
        )

        t_validate = PythonOperator(
            task_id="validate",
            python_callable=lambda: validate_aqi(str(config.clean_aqi_path()), config),
        )

        t_train = PythonOperator(
            task_id="train",
            python_callable=lambda: train_backtest_select(str(config.clean_aqi_path()), config),
        )

        t_forecast = PythonOperator(
            task_id="forecast",
            python_callable=lambda: forecast_publish(
                str(config.clean_aqi_path()),
                str(config.clean_weather_path()),
                config,
            ),
        )

        t_airnow >> t_prepare_aqi
        t_weather >> t_prepare_weather
        [t_prepare_aqi, t_prepare_weather] >> t_merge >> t_validate >> t_train >> t_forecast

    return dag


def build_dag_dot() -> str:
    return """digraph AQI_PIPELINE {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#e8f5e9"];

  ingest_airnow -> prepare_aqi -> merge_features -> validate -> train -> forecast;
  ingest_weather -> prepare_weather -> merge_features;
}"""
