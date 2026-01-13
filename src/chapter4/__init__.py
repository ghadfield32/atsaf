"""
Chapter 4: Monitoring + Drift Detection

This module provides:
1. Pipeline health checks (health.py)
2. Evidently-based drift detection (drift.py)
3. Alert logic with thresholds (alerts.py)
4. Streamlit monitoring dashboard (dashboard.py)

Usage (Python):
    from chapter4.health import full_health_check
    from chapter4.drift import detect_data_drift, rolling_performance

Usage (Dashboard):
    streamlit run src/chapter4/dashboard.py
"""

from .health import HealthReport, full_health_check, check_freshness
from .drift import detect_data_drift, rolling_performance
from .alerts import AlertConfig, check_alerts, Alert

__all__ = [
    "HealthReport",
    "full_health_check",
    "check_freshness",
    "detect_data_drift",
    "rolling_performance",
    "AlertConfig",
    "check_alerts",
    "Alert",
]
