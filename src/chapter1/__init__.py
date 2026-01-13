"""
Chapter 1: EDA & Data Pipeline

Simple, step-by-step functions for learning:
1. config - Load API key and settings
2. ingest - Paginated API calls with stable sort
3. prepare - Normalize time to UTC
4. validate - Check time series integrity
"""

from .config import Settings, load_settings
from .ingest import pull_data_paged
from .prepare import normalize_time, prepare_for_forecasting
from .validate import validate_time_index, print_validation_report

__all__ = [
    "Settings",
    "load_settings",
    "pull_data_paged",
    "normalize_time",
    "prepare_for_forecasting",
    "validate_time_index",
    "print_validation_report",
]
