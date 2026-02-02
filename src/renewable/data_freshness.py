# src/renewable/data_freshness.py
"""
Lightweight EIA data freshness checking.

This module provides functions to check if new data is available from the EIA API
before running the full pipeline. It compares the current max timestamps with
the previous run's max timestamps to determine if a full pipeline run is needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests

from src.renewable.regions import get_eia_respondent

logger = logging.getLogger(__name__)


def _sanitize_url(url: str) -> str:
    parts = urlsplit(url)
    q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.lower() != "api_key"]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q), parts.fragment))


@dataclass(frozen=True)
class FreshnessCheckResult:
    """Result of a data freshness check."""

    has_new_data: bool
    checked_at_utc: str
    series_status: dict[str, dict] = field(default_factory=dict)
    summary: str = ""


def load_previous_max_ds(run_log_path: Path) -> dict[str, str]:
    """
    Load per-series max_ds from previous run_log.json.

    Args:
        run_log_path: Path to run_log.json

    Returns:
        Dict mapping unique_id -> max_ds ISO string.
        Empty dict if file doesn't exist or is malformed.
    """
    if not run_log_path.exists():
        logger.info("[freshness] No previous run_log.json found - first run")
        return {}

    try:
        data = json.loads(run_log_path.read_text(encoding="utf-8"))

        # Navigate to diagnostics.generation_coverage.coverage
        coverage = (
            data.get("diagnostics", {})
            .get("generation_coverage", {})
            .get("coverage", [])
        )

        if not coverage:
            logger.warning("[freshness] run_log.json has no coverage data")
            return {}

        result = {}
        for item in coverage:
            uid = item.get("unique_id")
            max_ds = item.get("max_ds")
            if uid and max_ds:
                result[uid] = max_ds

        logger.info(f"[freshness] Loaded {len(result)} series from previous run_log")
        return result

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"[freshness] Failed to parse run_log.json: {e}")
        return {}


def probe_eia_latest(
    api_key: str,
    region: str,
    fuel_type: str,
    *,
    timeout: int = 15,
    max_lag_hours: Optional[int] = None,
) -> dict:
    """
    Fetch only the single most recent record from EIA API.

    This is a lightweight probe that uses:
    - length=1 (only fetch 1 record)
    - sort by period DESC (most recent first)

    NEW: Now calculates lag and includes in return value for diagnosis.

    Args:
        api_key: EIA API key
        region: Region code (CALI, ERCO, MISO, etc.)
        fuel_type: Fuel type (WND, SUN)
        timeout: Request timeout in seconds
        max_lag_hours: Optional threshold to flag stale data

    Returns:
        Dict with keys:
        - 'timestamp': ISO timestamp string or None
        - 'lag_hours': Float hours since now, or None
        - 'is_stale': Bool if lag > max_lag_hours (if threshold provided)
        - 'error': Error message if probe failed, or None
    """
    safe_url = None
    try:
        respondent = get_eia_respondent(region)

        params = {
            "api_key": api_key,
            "data[]": "value",
            "facets[respondent][]": respondent,
            "facets[fueltype][]": fuel_type,
            "frequency": "hourly",
            "length": 1,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
        }

        base_url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
        # Build a sanitized URL for logging without leaking the API key.
        prepared = requests.Request("GET", base_url, params=params).prepare()
        safe_url = _sanitize_url(prepared.url)
        resp = requests.get(base_url, params=params, timeout=timeout)
        resp.raise_for_status()

        payload = resp.json()
        response = payload.get("response", {})
        records = response.get("data", [])

        if not records:
            logger.warning(f"[probe] {region}_{fuel_type}: No records returned")
            return {
                "timestamp": None,
                "lag_hours": None,
                "is_stale": None,
                "error": "No records returned from EIA API"
            }

        period = records[0].get("period")
        if not period:
            logger.warning(f"[probe] {region}_{fuel_type}: Record missing 'period'")
            return {
                "timestamp": None,
                "lag_hours": None,
                "is_stale": None,
                "error": "Record missing 'period' field"
            }

        # Parse to consistent ISO format
        ts = pd.to_datetime(period, utc=True)
        timestamp_iso = ts.isoformat()

        # Calculate lag from now (floored to hour for consistency)
        now_utc = pd.Timestamp.now(tz="UTC").floor("h")
        lag_hours = (now_utc - ts).total_seconds() / 3600.0

        # Determine staleness if threshold provided
        is_stale = (
            (lag_hours > max_lag_hours)
            if max_lag_hours is not None
            else None
        )

        # Log with diagnostic info
        status_str = ""
        if is_stale is not None:
            if is_stale:
                status_str = f" (STALE > {max_lag_hours}h)"
            else:
                status_str = f" (OK < {max_lag_hours}h)"
        series_id = f"{region}_{fuel_type}"
        logger.info(
            f"[freshness][PROBE] {series_id}: latest_ds={timestamp_iso} "
            f"lag={lag_hours:.1f}h{status_str}"
        )

        return {
            "timestamp": timestamp_iso,
            "lag_hours": lag_hours,
            "is_stale": is_stale,
            "error": None
        }

    except requests.RequestException as e:
        url_msg = f" url={safe_url}" if safe_url else ""
        logger.warning(f"[probe] {region}_{fuel_type}: API error: {e}{url_msg}")
        return {
            "timestamp": None,
            "lag_hours": None,
            "is_stale": None,
            "error": f"API error: {e}"
        }
    except Exception as e:
        logger.warning(f"[probe] {region}_{fuel_type}: Unexpected error: {e}")
        return {
            "timestamp": None,
            "lag_hours": None,
            "is_stale": None,
            "error": f"Unexpected error: {e}"
        }


def _compare_timestamps(prev: Optional[str], current: Optional[str]) -> bool:
    """
    Return True if current is strictly newer than prev.

    Handles None values conservatively (assume new data if unknown).
    """
    if not prev or not current:
        return True  # Unknown = assume new data (conservative)

    try:
        prev_dt = pd.to_datetime(prev, utc=True)
        curr_dt = pd.to_datetime(current, utc=True)
        return curr_dt > prev_dt
    except Exception:
        return True  # Parse error = assume new data


def check_all_series_freshness(
    regions: list[str],
    fuel_types: list[str],
    run_log_path: Path,
    api_key: str,
    max_lag_hours: int = 48,
) -> FreshnessCheckResult:
    """
    Check all series for new data availability AND freshness.

    NEW: Now also validates that series are not stale beyond max_lag_hours.

    Args:
        regions: List of region codes (e.g., ["CALI", "ERCO", "MISO"])
        fuel_types: List of fuel types (e.g., ["WND", "SUN"])
        run_log_path: Path to previous run_log.json
        api_key: EIA API key
        max_lag_hours: Max allowed lag before considering data stale

    Returns:
        FreshnessCheckResult with has_new_data flag and per-series status.
        has_new_data=False if:
        1. No series has changed since last run, OR
        2. Some series have changed but at least one is stale
    """
    checked_at = datetime.now(timezone.utc).isoformat()

    # 1. Load previous max_ds values
    prev_max_ds = load_previous_max_ds(run_log_path)

    # 2. If no previous run_log, always run full pipeline (first run)
    if not prev_max_ds:
        return FreshnessCheckResult(
            has_new_data=True,
            checked_at_utc=checked_at,
            series_status={},
            summary=(
                "No previous run_log.json found - "
                "running full pipeline (first run)"
            ),
        )

    # 3. Probe each series
    series_status: dict[str, dict] = {}
    has_any_new = False
    new_series: list[str] = []
    error_series: list[str] = []
    stale_series: list[str] = []

    for region in regions:
        for fuel_type in fuel_types:
            series_id = f"{region}_{fuel_type}"
            prev = prev_max_ds.get(series_id)

            # Probe with lag calculation
            probe_result = probe_eia_latest(
                api_key, region, fuel_type, max_lag_hours=max_lag_hours
            )

            current = probe_result.get("timestamp")
            lag_hours = probe_result.get("lag_hours")
            is_stale = probe_result.get("is_stale")
            error = probe_result.get("error")

            # Determine if this series has new data
            if current is None:
                # API error - be conservative, assume new data
                is_new = True
                error_series.append(series_id)
                logger.warning(
                    f"[freshness] {series_id}: probe failed, "
                    f"assuming new data"
                )
            else:
                is_new = _compare_timestamps(prev, current)

            series_status[series_id] = {
                "prev_max_ds": prev,
                "current_max_ds": current,
                "lag_hours": lag_hours,
                "is_stale": is_stale,
                "is_new": is_new,
                "error": error,
            }

            if is_new:
                has_any_new = True
                if current is not None:
                    new_series.append(series_id)

            # Track stale series
            if is_stale:
                stale_series.append(series_id)

            # Log each series check
            status_str = "NEW" if is_new else "unchanged"
            if is_stale:
                status_str += " (STALE)"
            # Avoid formatting None lag_hours; surface missing lag explicitly.
            if lag_hours is None:
                lag_str = "unknown"
            else:
                lag_str = f"{lag_hours:.1f}h"
            error_str = f" error={error}" if error else ""
            logger.info(
                f"[freshness] {series_id}: prev={prev} current={current} "
                f"lag={lag_str} ({status_str}){error_str}"
            )

    # 4. Check for stale series (blocks pipeline run)
    if stale_series:
        summary = (
            f"Stale series detected (>{max_lag_hours}h): "
            f"{', '.join(stale_series)}"
        )
        logger.warning(f"[freshness] {summary}")
        return FreshnessCheckResult(
            has_new_data=False,  # BLOCK pipeline run
            checked_at_utc=checked_at,
            series_status=series_status,
            summary=summary,
        )

    # 5. Build summary for non-stale cases
    if error_series:
        summary = (
            f"Probe errors for {error_series}, "
            f"assuming new data available"
        )
    elif new_series:
        summary = f"New data found for: {', '.join(new_series)}"
    else:
        summary = "No new data found for any series"

    return FreshnessCheckResult(
        has_new_data=has_any_new,
        checked_at_utc=checked_at,
        series_status=series_status,
        summary=summary,
    )


if __name__ == "__main__":
    # Quick test
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        print("EIA_API_KEY not set")
        exit(1)

    run_log_path = Path("data/renewable/run_log.json")

    result = check_all_series_freshness(
        regions=["CALI", "ERCO", "MISO"],
        fuel_types=["WND", "SUN"],
        run_log_path=run_log_path,
        api_key=api_key,
    )

    print(f"\nFreshness Check Result:")
    print(f"  has_new_data: {result.has_new_data}")
    print(f"  checked_at: {result.checked_at_utc}")
    print(f"  summary: {result.summary}")
    print(f"\nPer-series status:")
    for series_id, status in result.series_status.items():
        print(f"  {series_id}: {status}")
