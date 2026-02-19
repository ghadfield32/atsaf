"""Tests for runtime diagnostics and execution controls."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.renewable.jobs import run_hourly
from src.renewable.tasks import _run_with_step_heartbeat


def test_run_with_step_heartbeat_returns_result_when_disabled() -> None:
    """Heartbeat wrapper should be transparent when interval is disabled."""

    result = _run_with_step_heartbeat(
        step_name="unit_test_step",
        fn=lambda: {"ok": True},
        heartbeat_seconds=0,
    )

    assert result == {"ok": True}


def test_run_with_step_heartbeat_emits_progress_logs(caplog) -> None:
    """Heartbeat wrapper should emit periodic progress for long-running work."""

    def _slow_step() -> str:
        time.sleep(1.1)
        return "done"

    with caplog.at_level(logging.INFO):
        result = _run_with_step_heartbeat(
            step_name="slow_step",
            fn=_slow_step,
            heartbeat_seconds=1,
        )

    assert result == "done"
    assert "[pipeline][HEARTBEAT] step=slow_step" in caplog.text


def test_run_hourly_pipeline_applies_runtime_controls(monkeypatch, tmp_path: Path) -> None:
    """run_hourly should pass env-based runtime controls into pipeline config/call."""

    monkeypatch.setenv("FORCE_RUN", "true")
    monkeypatch.setenv("RENEWABLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RENEWABLE_REGIONS", "CALI")
    monkeypatch.setenv("RENEWABLE_FUELS", "SUN")
    monkeypatch.setenv("RENEWABLE_ENABLE_INTERPRETABILITY", "false")
    monkeypatch.setenv("PIPELINE_HEARTBEAT_SECONDS", "17")
    monkeypatch.setenv("SKIP_EDA", "true")

    captured: dict[str, object] = {}

    def _fake_run_full_pipeline(cfg, **kwargs):
        captured["cfg"] = cfg
        captured["kwargs"] = kwargs
        return {
            "generation_rows": 1,
            "gap_filter": {"series_dropped": []},
            "best_model": "SeasonalNaive",
            "best_rmse": 1.0,
        }

    gen_df = pd.DataFrame(
        {
            "unique_id": ["CALI_SUN"],
            "ds": [pd.Timestamp("2026-02-19T10:00:00Z")],
            "y": [10.0],
        }
    )
    fcst_df = pd.DataFrame(
        {
            "unique_id": ["CALI_SUN"],
            "ds": [pd.Timestamp("2026-02-19T11:00:00Z")],
            "yhat": [11.0],
        }
    )

    def _fake_read_parquet(path):
        path_str = str(path)
        if path_str.endswith("generation.parquet"):
            return gen_df
        if path_str.endswith("forecasts.parquet"):
            return fcst_df
        raise AssertionError(f"Unexpected parquet read: {path_str}")

    monkeypatch.setattr(run_hourly, "run_full_pipeline", _fake_run_full_pipeline)
    monkeypatch.setattr(run_hourly.pd, "read_parquet", _fake_read_parquet)
    monkeypatch.setattr(
        run_hourly,
        "validate_generation_df",
        lambda *args, **kwargs: SimpleNamespace(ok=True, message="OK", details={}),
    )

    result = run_hourly.run_hourly_pipeline()

    cfg = captured["cfg"]
    kwargs = captured["kwargs"]
    assert cfg.enable_interpretability is False
    assert kwargs["heartbeat_seconds"] == 17

    assert result["config"]["enable_interpretability"] is False
    assert result["config"]["heartbeat_seconds"] == 17
    assert result["config"]["skip_eda"] is True

    run_log_path = tmp_path / "run_log.json"
    assert run_log_path.exists()
    run_log = json.loads(run_log_path.read_text())
    assert run_log["config"]["enable_interpretability"] is False
    assert run_log["config"]["heartbeat_seconds"] == 17
