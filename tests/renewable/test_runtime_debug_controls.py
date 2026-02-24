"""Tests for runtime diagnostics and execution controls."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.renewable.jobs import run_hourly
from src.renewable.modeling import RenewableForecastModel
from src.renewable import tasks as renewable_tasks
from src.renewable.tasks import (
    RenewablePipelineConfig,
    _run_with_step_heartbeat,
    train_renewable_models,
)


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
    monkeypatch.setenv("RENEWABLE_CV_PROFILE_MODELS", "true")
    monkeypatch.setenv("RENEWABLE_CV_MODEL_ALLOWLIST", "MSTL_ARIMA,SeasonalNaive")
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
    assert result["config"]["cv_profile_models"] is True
    assert result["config"]["cv_model_allowlist"] == ["MSTL_ARIMA", "SeasonalNaive"]
    assert result["config"]["skip_eda"] is True

    run_log_path = tmp_path / "run_log.json"
    assert run_log_path.exists()
    run_log = json.loads(run_log_path.read_text())
    assert run_log["config"]["enable_interpretability"] is False
    assert run_log["config"]["heartbeat_seconds"] == 17
    assert run_log["config"]["cv_profile_models"] is True
    assert run_log["config"]["cv_model_allowlist"] == ["MSTL_ARIMA", "SeasonalNaive"]


def test_train_renewable_models_passes_cv_debug_env(monkeypatch) -> None:
    """train_renewable_models should pass CV debug controls into cross_validate."""

    monkeypatch.setenv("RENEWABLE_CV_PROFILE_MODELS", "true")
    monkeypatch.setenv("RENEWABLE_CV_MODEL_ALLOWLIST", "MSTL_ARIMA,SeasonalNaive")

    captured: dict[str, object] = {}

    cv_results = pd.DataFrame(
        {
            "unique_id": ["CALI_SUN"],
            "ds": [pd.Timestamp("2026-02-20T00:00:00Z")],
            "cutoff": [pd.Timestamp("2026-02-19T00:00:00Z")],
            "y": [10.0],
            "MSTL_ARIMA": [9.5],
        }
    )
    leaderboard = pd.DataFrame(
        {
            "model": ["MSTL_ARIMA"],
            "rmse": [0.5],
            "mae": [0.4],
            "valid_rows": [1],
        }
    )

    def _fake_cross_validate(
        self,
        df: pd.DataFrame,
        n_windows: int = 3,
        step_size: int = 168,
        *,
        profile_models: bool = False,
        model_allowlist: list[str] | None = None,
    ):
        captured["n_windows"] = n_windows
        captured["step_size"] = step_size
        captured["profile_models"] = profile_models
        captured["model_allowlist"] = model_allowlist
        captured["rows"] = len(df)
        self.last_cv_diagnostics = {
            "status": "ok",
            "mode": "per_model",
            "models": model_allowlist,
            "total_seconds": 1.23,
            "cv_rows": 1,
            "per_model_seconds": {"MSTL_ARIMA": 0.75},
        }
        return cv_results, leaderboard

    monkeypatch.setattr(RenewableForecastModel, "cross_validate", _fake_cross_validate)
    monkeypatch.setattr(
        renewable_tasks,
        "compute_baseline_metrics",
        lambda _cv, model_name: {
            "model": model_name,
            "rmse_mean": 0.5,
            "rmse_std": 0.0,
            "drift_threshold_rmse": 0.5,
            "n_windows": 1,
        },
    )

    config = RenewablePipelineConfig(
        regions=["CALI"],
        fuel_types=["SUN"],
        lookback_days=30,
        horizon=24,
        cv_windows=2,
        cv_step_size=24,
        n_jobs=1,
        enable_interpretability=False,
    )
    modeling_df = pd.DataFrame(
        {
            "unique_id": ["CALI_SUN"] * 200,
            "ds": pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC"),
            "y": [10.0] * 200,
        }
    )

    _, _, baseline, cv_diag = train_renewable_models(
        config,
        modeling_df,
        return_diagnostics=True,
    )

    assert captured["profile_models"] is True
    assert captured["model_allowlist"] == ["MSTL_ARIMA", "SeasonalNaive"]
    assert baseline["model"] == "MSTL_ARIMA"
    assert cv_diag["mode"] == "per_model"
