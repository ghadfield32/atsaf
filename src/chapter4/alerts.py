"""
Chapter 4: Alert Logic

Threshold-based alerting for pipeline health and drift detection.
Supports multiple notification channels (log, email, slack).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from .health import HealthReport
from .drift import DriftReport

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    STALE_DATA = "stale_data"
    MISSING_DATA = "missing_data"
    STALE_FORECAST = "stale_forecast"
    DATA_DRIFT = "data_drift"
    MODEL_DRIFT = "model_drift"
    PIPELINE_FAILURE = "pipeline_failure"


@dataclass
class Alert:
    """Represents a triggered alert"""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: str
    value: Optional[float] = None      # Current value that triggered alert
    threshold: Optional[float] = None  # Threshold that was exceeded
    details: Optional[str] = None      # Additional context


@dataclass
class AlertConfig:
    """Configuration for alert thresholds"""
    # Data freshness thresholds (hours)
    freshness_warning_hours: float = 4.0
    freshness_critical_hours: float = 6.0

    # Missing data thresholds
    missing_warning_hours: int = 1
    missing_critical_hours: int = 3

    # Forecast freshness thresholds (hours)
    forecast_warning_hours: float = 12.0
    forecast_critical_hours: float = 24.0

    # Drift thresholds
    drift_warning_score: float = 0.1
    drift_critical_score: float = 0.3

    # RMSE increase threshold (relative to baseline)
    rmse_increase_warning: float = 0.15
    rmse_increase_critical: float = 0.25


def check_alerts(
    health: HealthReport,
    drift: Optional[DriftReport] = None,
    config: Optional[AlertConfig] = None,
) -> List[Alert]:
    """
    Check all thresholds and return list of triggered alerts.

    Args:
        health: Current health report
        drift: Optional drift report
        config: Alert configuration (uses defaults if None)

    Returns:
        List of triggered Alert objects
    """
    if config is None:
        config = AlertConfig()

    alerts = []
    timestamp = datetime.now(timezone.utc).isoformat()

    # Check data freshness
    if health.freshness_hours >= config.freshness_critical_hours:
        alerts.append(Alert(
            alert_type=AlertType.STALE_DATA,
            severity=AlertSeverity.CRITICAL,
            message=f"Data is {health.freshness_hours:.1f} hours old (critical threshold: {config.freshness_critical_hours}h)",
            timestamp=timestamp,
            value=health.freshness_hours,
            threshold=config.freshness_critical_hours,
        ))
    elif health.freshness_hours >= config.freshness_warning_hours:
        alerts.append(Alert(
            alert_type=AlertType.STALE_DATA,
            severity=AlertSeverity.WARNING,
            message=f"Data is {health.freshness_hours:.1f} hours old (warning threshold: {config.freshness_warning_hours}h)",
            timestamp=timestamp,
            value=health.freshness_hours,
            threshold=config.freshness_warning_hours,
        ))

    # Check missing hours
    if health.missing_hour_count >= config.missing_critical_hours:
        alerts.append(Alert(
            alert_type=AlertType.MISSING_DATA,
            severity=AlertSeverity.CRITICAL,
            message=f"{health.missing_hour_count} hours missing (critical threshold: {config.missing_critical_hours})",
            timestamp=timestamp,
            value=health.missing_hour_count,
            threshold=config.missing_critical_hours,
        ))
    elif health.missing_hour_count >= config.missing_warning_hours:
        alerts.append(Alert(
            alert_type=AlertType.MISSING_DATA,
            severity=AlertSeverity.WARNING,
            message=f"{health.missing_hour_count} hours missing (warning threshold: {config.missing_warning_hours})",
            timestamp=timestamp,
            value=health.missing_hour_count,
            threshold=config.missing_warning_hours,
        ))

    # Check forecast freshness
    if health.forecast_age_hours >= config.forecast_critical_hours:
        alerts.append(Alert(
            alert_type=AlertType.STALE_FORECAST,
            severity=AlertSeverity.CRITICAL,
            message=f"Forecast is {health.forecast_age_hours:.1f} hours old (critical threshold: {config.forecast_critical_hours}h)",
            timestamp=timestamp,
            value=health.forecast_age_hours,
            threshold=config.forecast_critical_hours,
        ))
    elif health.forecast_age_hours >= config.forecast_warning_hours:
        alerts.append(Alert(
            alert_type=AlertType.STALE_FORECAST,
            severity=AlertSeverity.WARNING,
            message=f"Forecast is {health.forecast_age_hours:.1f} hours old (warning threshold: {config.forecast_warning_hours}h)",
            timestamp=timestamp,
            value=health.forecast_age_hours,
            threshold=config.forecast_warning_hours,
        ))

    # Check drift (if provided)
    if drift is not None:
        if drift.drift_score >= config.drift_critical_score:
            alerts.append(Alert(
                alert_type=AlertType.DATA_DRIFT,
                severity=AlertSeverity.CRITICAL,
                message=f"Data drift detected (score: {drift.drift_score:.2f}, threshold: {config.drift_critical_score})",
                timestamp=timestamp,
                value=drift.drift_score,
                threshold=config.drift_critical_score,
                details=f"Drifted columns: {drift.drifted_columns}",
            ))
        elif drift.drift_score >= config.drift_warning_score:
            alerts.append(Alert(
                alert_type=AlertType.DATA_DRIFT,
                severity=AlertSeverity.WARNING,
                message=f"Possible data drift (score: {drift.drift_score:.2f}, threshold: {config.drift_warning_score})",
                timestamp=timestamp,
                value=drift.drift_score,
                threshold=config.drift_warning_score,
                details=f"Drifted columns: {drift.drifted_columns}",
            ))

    return alerts


def send_alert(alert: Alert, channel: str = "log") -> bool:
    """
    Send alert via specified channel.

    Args:
        alert: Alert to send
        channel: Notification channel ("log", "email", "slack")

    Returns:
        True if sent successfully
    """
    if channel == "log":
        return _send_log_alert(alert)
    elif channel == "email":
        return _send_email_alert(alert)
    elif channel == "slack":
        return _send_slack_alert(alert)
    else:
        logger.warning(f"Unknown alert channel: {channel}")
        return False


def _send_log_alert(alert: Alert) -> bool:
    """Send alert to logging system"""
    level = {
        AlertSeverity.INFO: logging.INFO,
        AlertSeverity.WARNING: logging.WARNING,
        AlertSeverity.CRITICAL: logging.CRITICAL,
    }.get(alert.severity, logging.INFO)

    logger.log(level, f"[{alert.alert_type.value}] {alert.message}")
    if alert.details:
        logger.log(level, f"  Details: {alert.details}")

    return True


def _send_email_alert(alert: Alert) -> bool:
    """Send alert via email (placeholder)"""
    # In production, integrate with SMTP or email service
    logger.info(f"[EMAIL] Would send: {alert.message}")
    return True


def _send_slack_alert(alert: Alert) -> bool:
    """Send alert via Slack (placeholder)"""
    # In production, integrate with Slack webhook
    logger.info(f"[SLACK] Would send: {alert.message}")
    return True


def should_demote_model(
    drift_history: List[DriftReport],
    n_consecutive_failures: int = 3,
) -> bool:
    """
    Check if model should be demoted based on consecutive drift detections.

    Args:
        drift_history: List of recent drift reports
        n_consecutive_failures: Number of consecutive failures to trigger demotion

    Returns:
        True if model should be demoted
    """
    if len(drift_history) < n_consecutive_failures:
        return False

    recent = drift_history[-n_consecutive_failures:]
    return all(r.drift_detected for r in recent)


def format_alert_summary(alerts: List[Alert]) -> str:
    """Format alerts into a readable summary string"""
    if not alerts:
        return "No alerts triggered."

    lines = [f"=== {len(alerts)} Alert(s) Triggered ==="]

    critical = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
    warning = [a for a in alerts if a.severity == AlertSeverity.WARNING]
    info = [a for a in alerts if a.severity == AlertSeverity.INFO]

    if critical:
        lines.append(f"\nCRITICAL ({len(critical)}):")
        for a in critical:
            lines.append(f"  - {a.message}")

    if warning:
        lines.append(f"\nWARNING ({len(warning)}):")
        for a in warning:
            lines.append(f"  - {a.message}")

    if info:
        lines.append(f"\nINFO ({len(info)}):")
        for a in info:
            lines.append(f"  - {a.message}")

    return "\n".join(lines)
