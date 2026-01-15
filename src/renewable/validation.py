# file: src/renewable/validation.py
"""Validation utilities for renewable generation data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    message: str
    details: dict


def validate_generation_df(
    df: pd.DataFrame,
    *,
    max_lag_hours: int = 3,
    max_missing_ratio: float = 0.02,
    expected_series: Optional[Iterable[str]] = None,
) -> ValidationReport:
    required = {"unique_id", "ds", "y"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        return ValidationReport(
            False,
            "Missing required columns",
            {"missing_cols": sorted(missing_cols)},
        )

    if df.empty:
        return ValidationReport(False, "Generation data is empty", {})

    work = df.copy()

    work["ds"] = pd.to_datetime(work["ds"], errors="coerce", utc=True)
    if work["ds"].isna().any():
        return ValidationReport(
            False,
            "Unparseable ds values found",
            {"bad_ds": int(work["ds"].isna().sum())},
        )

    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    if work["y"].isna().any():
        return ValidationReport(
            False,
            "Unparseable y values found",
            {"bad_y": int(work["y"].isna().sum())},
        )

    if (work["y"] < 0).any():
        return ValidationReport(
            False,
            "Negative generation values found",
            {"neg_y": int((work["y"] < 0).sum())},
        )

    dup = work.duplicated(subset=["unique_id", "ds"]).sum()
    if dup:
        return ValidationReport(
            False,
            "Duplicate (unique_id, ds) rows found",
            {"duplicates": int(dup)},
        )

    if expected_series:
        expected = sorted(set(expected_series))
        present = sorted(set(work["unique_id"]))
        missing_series = sorted(set(expected) - set(present))
        if missing_series:
            return ValidationReport(
                False,
                "Missing expected series",
                {"missing_series": missing_series, "present_series": present},
            )

    now_utc = pd.Timestamp.now(tz="UTC").floor("H")
    max_ds = work["ds"].max()
    lag_hours = (now_utc - max_ds).total_seconds() / 3600.0
    if lag_hours > max_lag_hours:
        return ValidationReport(
            False,
            "Data not fresh enough",
            {
                "now_utc": now_utc.isoformat(),
                "max_ds": max_ds.isoformat(),
                "lag_hours": lag_hours,
            },
        )

    series_max = work.groupby("unique_id")["ds"].max()
    series_lag = (now_utc - series_max).dt.total_seconds() / 3600.0
    stale = series_lag[series_lag > max_lag_hours].sort_values(ascending=False)
    if not stale.empty:
        return ValidationReport(
            False,
            "Stale series found",
            {
                "stale_series": stale.head(10).to_dict(),
                "max_lag_hours": max_lag_hours,
            },
        )

    missing_ratios = {}
    for uid, group in work.groupby("unique_id"):
        group = group.sort_values("ds")
        start = group["ds"].iloc[0]
        end = group["ds"].iloc[-1]
        expected = int(((end - start) / pd.Timedelta(hours=1)) + 1)
        actual = len(group)
        missing = max(expected - actual, 0)
        missing_ratios[uid] = missing / max(expected, 1)

    worst_uid = max(missing_ratios, key=missing_ratios.get)
    worst_ratio = missing_ratios[worst_uid]
    if worst_ratio > max_missing_ratio:
        return ValidationReport(
            False,
            "Too many missing hourly points",
            {"worst_uid": worst_uid, "worst_missing_ratio": worst_ratio},
        )

    return ValidationReport(
        True,
        "OK",
        {
            "row_count": len(work),
            "series_count": int(work["unique_id"].nunique()),
            "max_ds": max_ds.isoformat(),
            "lag_hours": lag_hours,
            "worst_missing_ratio": worst_ratio,
        },
    )
