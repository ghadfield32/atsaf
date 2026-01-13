# file: src/chapter4/config.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class MonitoringConfig:
    db_path: str = "artifacts/monitoring/monitoring.sqlite"

    # Restatement strategy (data can be revised)
    restatement_lookback_hours: int = 336  # 2 weeks (matches your notes)

    # Drift windows (rolling accuracy)
    roll_7d_hours: int = 24 * 7
    roll_14d_hours: int = 24 * 14

    # Drift threshold policy (data-driven, computed from backtests)
    # e.g. threshold = mean + k*std (k chosen)
    drift_std_k: float = 2.0

    # Alert routing (keep simple at first)
    alert_print_only: bool = True
