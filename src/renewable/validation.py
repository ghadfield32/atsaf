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
    region_lag_thresholds: Optional[dict[str, int]] = None,
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

    # Check for negative values and log warning (but allow to pass)
    # Dataset builder will handle negatives per configured policy
    if (work["y"] < 0).any():
        import logging
        logger = logging.getLogger(__name__)

        neg_mask = work["y"] < 0
        neg_count = int(neg_mask.sum())
        by_series = (
            work[neg_mask]
            .groupby("unique_id")
            .agg(count=("y", "count"), min_y=("y", "min"), max_y=("y", "max"))
            .reset_index()
        )

        logger.warning(
            "[validation][NEGATIVE] Found %d negative values "
            "(%.1f%%) across %d series",
            neg_count,
            100 * neg_count / len(work),
            len(by_series)
        )

        for _, row in by_series.iterrows():
            logger.warning(
                "  Series %s: %d negative values, range=[%.1f, %.1f]",
                row["unique_id"], row["count"],
                row["min_y"], row["max_y"]
            )

        logger.info(
            "[validation][NEGATIVE] Negatives will be handled by dataset builder "
            "according to configured negative_policy"
        )

        # Continue validation instead of failing
        # (Dataset builder will handle negatives per policy)

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

    import logging
    logger = logging.getLogger(__name__)

    # Calculate timing with diagnostic logging
    now_raw = pd.Timestamp.now(tz="UTC")
    now_utc = now_raw.floor("h")
    logger.info(
        f"[validation][TIMING] now_utc={now_utc.isoformat()} "
        f"(floored to hour)"
    )
    logger.info(
        f"[validation][TIMING] now_raw={now_raw.isoformat()} "
        f"(before floor)"
    )

    max_ds = work["ds"].max()
    logger.info(
        f"[validation][TIMING] Overall max_ds={max_ds.isoformat()}"
    )

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

    # Per-series lag calculation with diagnostic logging
    series_max = work.groupby("unique_id")["ds"].max()
    series_lag = (now_utc - series_max).dt.total_seconds() / 3600.0

    # Check per-series with region-specific thresholds if provided
    stale_series_dict = {}
    for uid, max_ds_series in series_max.items():
        series_lag_val = series_lag[uid]

        # Extract region from unique_id (format: "REGION_FUEL")
        region = uid.split("_")[0] if "_" in uid else None

        # Use region-specific threshold if available, else default
        threshold = max_lag_hours
        if region_lag_thresholds and region in region_lag_thresholds:
            threshold = region_lag_thresholds[region]

        # Check if stale
        is_stale = series_lag_val > threshold
        status = "STALE" if is_stale else "OK"

        # Log diagnosis
        logger.info(
            f"[validation][TIMING] {uid}: "
            f"max_ds={max_ds_series.isoformat()} "
            f"lag={series_lag_val:.1f}h "
            f"(threshold={threshold}h) [{status}]"
        )

        if is_stale:
            stale_series_dict[uid] = series_lag_val

    # Convert to pandas Series for compatibility with existing code
    if stale_series_dict:
        import pandas as pd
        stale = pd.Series(stale_series_dict).sort_values(ascending=False)
    else:
        stale = pd.Series(dtype=float)
    if not stale.empty:
        # Include per-series timestamps for stale-series root cause.
        stale_series = stale.head(10)
        stale_max_ds = {
            uid: series_max.loc[uid].isoformat()
            for uid in stale_series.index
        }

        # Include per-series thresholds used
        stale_thresholds = {}
        for uid in stale_series.index:
            region = uid.split("_")[0] if "_" in uid else None
            if region_lag_thresholds and region in region_lag_thresholds:
                stale_thresholds[uid] = region_lag_thresholds[region]
            else:
                stale_thresholds[uid] = max_lag_hours

        return ValidationReport(
            False,
            "Stale series found",
            {
                "stale_series": stale_series.to_dict(),
                "stale_series_max_ds": stale_max_ds,
                "stale_series_thresholds": stale_thresholds,
                "now_utc": now_utc.isoformat(),
                "max_ds_overall": max_ds.isoformat(),
                "max_lag_hours": max_lag_hours,
                "region_lag_thresholds": region_lag_thresholds,
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
