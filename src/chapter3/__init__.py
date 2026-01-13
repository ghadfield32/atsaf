"""
Chapter 3: Automation (ETL + Orchestration)

This module provides:
1. Idempotent pipeline tasks (tasks.py)
2. Typer CLI for local execution (cli.py)
3. Airflow DAG builder for production (dag_builder.py)

Usage (CLI):
    python -m chapter3.cli run-pipeline --start-date 2024-01-01
    python -m chapter3.cli run-ingest --start-date 2024-06-01

Usage (Python):
    from chapter3.tasks import ingest_eia, train_backtest_select
    from chapter3.config import PipelineConfig
"""

from .config import PipelineConfig
from .tasks import (forecast_publish, ingest_eia, prepare_clean,
                    register_champion, train_backtest_select)

__all__ = [
    "PipelineConfig",
    "ingest_eia",
    "prepare_clean",
    "train_backtest_select",
    "register_champion",
    "forecast_publish",
]
