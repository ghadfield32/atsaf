"""Targeted regression tests for validation debugging and pd scoping."""

from __future__ import annotations

import logging

import pandas as pd

from src.renewable.validation import validate_generation_df


def test_validate_generation_df_stale_series_branch_regression() -> None:
    """Ensure stale-series evaluation works (previously crashed on local pd shadowing)."""
    now = pd.Timestamp.now(tz="UTC").floor("h")
    df = pd.DataFrame(
        {
            "unique_id": ["CALI_SUN", "CALI_SUN", "ERCO_SUN", "ERCO_SUN"],
            "ds": [
                now - pd.Timedelta(hours=10),
                now - pd.Timedelta(hours=9),
                now - pd.Timedelta(hours=2),
                now - pd.Timedelta(hours=1),
            ],
            "y": [10.0, 12.0, 40.0, 42.0],
        }
    )

    report = validate_generation_df(df, max_lag_hours=3)

    assert report.ok is False
    assert report.message == "Stale series found"
    assert "CALI_SUN" in report.details["stale_series"]


def test_validate_generation_df_debug_snapshot_logging(caplog) -> None:
    """Debug mode should emit stage snapshots without mutating validation behavior."""
    now = pd.Timestamp.now(tz="UTC").floor("h")
    df = pd.DataFrame(
        {
            "unique_id": ["ERCO_WND", "ERCO_WND", "ERCO_WND"],
            "ds": [now - pd.Timedelta(hours=2), now - pd.Timedelta(hours=1), now],
            "y": [120.0, 110.0, 115.0],
        }
    )

    with caplog.at_level(logging.WARNING):
        report = validate_generation_df(df, max_lag_hours=3, debug=True)

    assert report.ok is True
    assert "[validation][DEBUG][raw_input]" in caplog.text
    assert "[validation][DEBUG][after_ds_parse]" in caplog.text
    assert "[validation][DEBUG][after_y_parse]" in caplog.text
