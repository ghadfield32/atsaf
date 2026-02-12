# src/renewable/eia_renewable.py
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import find_dotenv, load_dotenv

from src.renewable.regions import (REGIONS, get_eia_respondent,
                                   validate_fuel_type, validate_region)

logger = logging.getLogger(__name__)


def _sanitize_url(url: str) -> str:
    parts = urlsplit(url)
    q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.lower() != "api_key"]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q), parts.fragment))


def _load_env_once(*, debug: bool = False) -> Optional[str]:
    """
    Load .env if present.
    - Primary: find_dotenv(usecwd=True) (walk up from CWD)
    - Fallback: repo_root/.env based on this file location
    Returns the path loaded (or None).
    """
    # 1) Try from current working directory upward
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
        if debug:
            logger.info("Loaded .env via find_dotenv: %s", dotenv_path)
        return dotenv_path

    # 2) Fallback: assume src-layout -> repo root is ../../ from this file
    try:
        repo_root = Path(__file__).resolve().parents[2]
        fallback = repo_root / ".env"
        if fallback.exists():
            load_dotenv(fallback, override=False)
            if debug:
                logger.info("Loaded .env via fallback: %s", str(fallback))
            return str(fallback)
    except Exception:
        pass

    if debug:
        logger.info("No .env found to load.")
    return None


class EIARenewableFetcher:
    BASE_URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
    MAX_RECORDS_PER_REQUEST = 5000
    RATE_LIMIT_DELAY = 0.2  # 5 requests/second max

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        timeout: int = 60,
        debug_env: bool = False,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        retry_statuses: Optional[tuple[int, ...]] = None,
        debug_requests: bool = False,
    ):
        """
        Initialize API key and configuration.

        Args:
            api_key: EIA API key (or reads from EIA_API_KEY env var)
            timeout: Request timeout in seconds (default: 60)
            debug_env: Enable debug logging for environment loading
        """
        loaded_env = _load_env_once(debug=debug_env)

        self.api_key = api_key or os.getenv("EIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "EIA API key required but not found.\n"
                "- Ensure .env contains EIA_API_KEY=...\n"
                "- Ensure your process CWD is under the repo (so find_dotenv can locate it), OR\n"
                "- Pass api_key=... explicitly.\n"
                f"Loaded .env path: {loaded_env}"
            )

        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_statuses = retry_statuses or (429, 500, 502, 503, 504)
        self.debug_requests = debug_requests
        self.session = self._create_session()  # Add retry-enabled session

        # Debug without leaking the key
        if debug_env:
            masked = self.api_key[:4] + "..." + self.api_key[-4:] if len(self.api_key) >= 8 else "***"
            logger.info("EIA_API_KEY loaded (masked): %s", masked)
            logger.info("Request timeout: %d seconds", self.timeout)
        if debug_env or self.debug_requests:
            logger.info(
                "EIA retries configured: total=%s backoff=%s statuses=%s",
                self.max_retries,
                self.backoff_factor,
                self.retry_statuses,
            )

    def _create_session(self) -> requests.Session:
        """Create requests Session with retry logic for transient errors."""
        session = requests.Session()
        retries = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,  # 1s, 2s, 4s between retries
            status_forcelist=self.retry_statuses,  # Retry on server errors and rate limits
            allowed_methods=frozenset(["GET"]),
            connect=self.max_retries,  # Retry on connection errors
            read=self.max_retries,     # Retry on read timeouts
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    @staticmethod
    def _extract_eia_response(payload: dict, *, request_url: Optional[str] = None) -> tuple[list[dict], dict]:
        if not isinstance(payload, dict):
            raise TypeError(f"EIA payload is not a dict. type={type(payload)} url={request_url}")

        if "error" in payload and payload.get("response") is None:
            raise ValueError(f"EIA returned error payload. url={request_url} error={payload.get('error')}")

        if "response" not in payload:
            raise ValueError(
                f"EIA payload missing 'response'. url={request_url} keys={list(payload.keys())[:25]}"
            )

        response = payload.get("response") or {}
        if not isinstance(response, dict):
            raise TypeError(f"EIA payload['response'] is not a dict. type={type(response)} url={request_url}")

        if "data" not in response:
            raise ValueError(
                f"EIA response missing 'data'. url={request_url} response_keys={list(response.keys())[:25]}"
            )

        records = response.get("data") or []
        if not isinstance(records, list):
            raise TypeError(f"EIA response['data'] is not a list. type={type(records)} url={request_url}")

        total = response.get("total", None)
        offset = response.get("offset", None)

        meta_obj = response.get("metadata") or {}
        if isinstance(meta_obj, dict):
            if total is None and "total" in meta_obj:
                total = meta_obj.get("total")
            if offset is None and "offset" in meta_obj:
                offset = meta_obj.get("offset")

        try:
            total = int(total) if total is not None else None
        except Exception:
            pass
        try:
            offset = int(offset) if offset is not None else None
        except Exception:
            pass

        return records, {"total": total, "offset": offset}

    def fetch_region(
        self,
        region: str,
        fuel_type: str,
        start_date: str,
        end_date: str,
        *,
        debug: bool = False,
        diag: Optional[dict] = None,
    ) -> pd.DataFrame:
        if not validate_region(region):
            raise ValueError(f"Invalid region: {region}")
        if not validate_fuel_type(fuel_type):
            raise ValueError(f"Invalid fuel type: {fuel_type}")

        respondent = get_eia_respondent(region)

        all_records: list[dict] = []
        offset = 0

        # ✅ FIX: initialize loop diagnostics counters
        page_count = 0
        total_hint: Optional[int] = None
        request_attempts = 0
        request_failures = 0
        status_counts: dict[int, int] = {}
        last_error: Optional[str] = None
        last_url: Optional[str] = None

        if diag is not None:
            diag.update({
                "region": region,
                "fuel_type": fuel_type,
                "start_date": start_date,
                "end_date": end_date,
                "total_records": None,
                "pages": 0,
                "rows_parsed": 0,
                "empty": None,
                "request_attempts": 0,
                "request_failures": 0,
                "status_counts": {},
                "last_request_url": None,
                "last_error": None,
            })

        while True:
            params = {
                "api_key": self.api_key,
                "data[]": "value",
                "facets[respondent][]": respondent,
                "facets[fueltype][]": fuel_type,
                "frequency": "hourly",
                "start": f"{start_date}T00",
                "end": f"{end_date}T23",
                "length": self.MAX_RECORDS_PER_REQUEST,
                "offset": offset,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
            }

            # Prepare a sanitized URL for logging and diagnostics.
            req = requests.Request("GET", self.BASE_URL, params=params)
            prepared = self.session.prepare_request(req)
            safe_url = _sanitize_url(prepared.url)

            request_attempts += 1
            start_ts = time.monotonic()
            try:
                resp = self.session.send(prepared, timeout=self.timeout)
                elapsed = time.monotonic() - start_ts
                last_url = safe_url
                status_counts[resp.status_code] = status_counts.get(resp.status_code, 0) + 1

                if self.debug_requests or debug:
                    logger.info(
                        "[fetch_region][REQUEST_OK] region=%s fuel=%s status=%s elapsed=%.2fs offset=%s url=%s",
                        region, fuel_type, resp.status_code, elapsed, offset, safe_url
                    )

                resp.raise_for_status()
                payload = resp.json()
            except requests.RequestException as e:
                elapsed = time.monotonic() - start_ts
                request_failures += 1
                last_error = str(e)
                if self.debug_requests or debug:
                    logger.warning(
                        "[fetch_region][REQUEST_FAIL] region=%s fuel=%s offset=%s elapsed=%.2fs error=%s url=%s",
                        region, fuel_type, offset, elapsed, last_error, safe_url
                    )
                if diag is not None:
                    diag.update({
                        "request_attempts": request_attempts,
                        "request_failures": request_failures,
                        "status_counts": status_counts,
                        "last_request_url": safe_url,
                        "last_error": last_error,
                    })
                raise
            except Exception as e:
                elapsed = time.monotonic() - start_ts
                last_error = f"{type(e).__name__}: {e}"
                if self.debug_requests or debug:
                    logger.warning(
                        "[fetch_region][REQUEST_ERROR] region=%s fuel=%s offset=%s elapsed=%.2fs error=%s url=%s",
                        region, fuel_type, offset, elapsed, last_error, safe_url
                    )
                if diag is not None:
                    diag.update({
                        "request_attempts": request_attempts,
                        "request_failures": request_failures,
                        "status_counts": status_counts,
                        "last_request_url": safe_url,
                        "last_error": last_error,
                    })
                raise

            records, meta = self._extract_eia_response(payload, request_url=safe_url)

            page_count += 1
            if total_hint is None:
                total_hint = meta.get("total")

            returned = len(records)

            if debug:
                safe_url = _sanitize_url(resp.url)
                print(
                    f"[PAGE] region={region} fuel={fuel_type} returned={returned} "
                    f"offset={offset} total={meta.get('total')} url={safe_url}"
                )

            # Empty on first page: legitimate empty series for that window
            if returned == 0 and offset == 0:
                if diag is not None:
                    diag.update({
                        "region": region,
                        "fuel_type": fuel_type,
                        "start_date": start_date,
                        "end_date": end_date,
                        "total_records": total_hint,
                        "pages": page_count,
                        "rows_parsed": 0,
                        "empty": True,
                        "request_attempts": request_attempts,
                        "request_failures": request_failures,
                        "status_counts": status_counts,
                        "last_request_url": last_url,
                        "last_error": last_error,
                    })
                return pd.DataFrame(columns=["ds", "value", "region", "fuel_type"])

            if returned == 0:
                break

            all_records.extend(records)

            if returned < self.MAX_RECORDS_PER_REQUEST:
                break

            offset += self.MAX_RECORDS_PER_REQUEST
            time.sleep(self.RATE_LIMIT_DELAY)

        df = pd.DataFrame(all_records)

        missing_cols = [c for c in ["period", "value"] if c not in df.columns]
        if missing_cols:
            sample_keys = sorted(set().union(*(r.keys() for r in all_records[:5]))) if all_records else []
            raise ValueError(
                f"EIA records missing expected keys {missing_cols}. "
                f"columns={df.columns.tolist()} sample_record_keys={sample_keys}"
            )

        # EIA returns timestamps in UTC format WITHOUT timezone marker (e.g., "2026-01-21T00")
        # Simply parse and treat as UTC (no conversion needed)
        df["ds"] = pd.to_datetime(df["period"], utc=True, errors="coerce").dt.tz_localize(None)

        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df["region"] = region
        df["fuel_type"] = fuel_type

        df = df.dropna(subset=["ds", "value"]).sort_values("ds").reset_index(drop=True)

        # DEBUG: Log data coverage details
        if not df.empty:
            actual_start = df["ds"].min()
            actual_end = df["ds"].max()
            actual_span_hours = (actual_end - actual_start).total_seconds() / 3600.0
            requested_start_dt = pd.to_datetime(f"{start_date}T00", utc=True).tz_localize(None)
            requested_end_dt = pd.to_datetime(f"{end_date}T23", utc=True).tz_localize(None)
            requested_span_hours = (requested_end_dt - requested_start_dt).total_seconds() / 3600.0
            expected_records = int(requested_span_hours + 1)
            coverage_pct = 100 * len(df) / max(expected_records, 1)

            logger.info(
                "[fetch_region][DATA_COVERAGE] region=%s fuel=%s "
                "requested=[%s to %s] (%.1fh) "
                "actual=[%s to %s] (%.1fh) "
                "records=%d/%d (%.1f%% coverage)",
                region, fuel_type,
                start_date, end_date, requested_span_hours,
                actual_start.isoformat(), actual_end.isoformat(), actual_span_hours,
                len(df), expected_records, coverage_pct
            )

        # Log negative values for investigation (but don't clamp - let dataset builder handle)
        neg_mask = df["value"] < 0
        if neg_mask.any():
            neg_count = int(neg_mask.sum())
            neg_min = float(df.loc[neg_mask, "value"].min())
            neg_max = float(df.loc[neg_mask, "value"].max())
            neg_pct = 100 * neg_count / max(len(df), 1)
            logger.warning(
                "[fetch_region][NEGATIVE] region=%s fuel=%s count=%d (%.1f%%) range=[%.2f, %.2f]",
                region, fuel_type, neg_count, neg_pct, neg_min, neg_max,
            )
            # Log sample for debugging
            neg_sample = df.loc[neg_mask, ["ds", "value"]].head(5)
            for _, row in neg_sample.iterrows():
                logger.debug("  ds=%s value=%.2f", row["ds"], row["value"])

            # NOTE: Keeping negative values in raw data for transparency
            # Dataset builder will handle negatives according to configured policy

        if diag is not None:
            diag.update({
                "region": region,
                "fuel_type": fuel_type,
                "start_date": start_date,
                "end_date": end_date,
                "total_records": total_hint,
                "pages": page_count,
                "rows_parsed": int(len(df)),
                "empty": bool(len(df) == 0),
                "request_attempts": request_attempts,
                "request_failures": request_failures,
                "status_counts": status_counts,
                "last_request_url": last_url,
                "last_error": last_error,
            })

        return df[["ds", "value", "region", "fuel_type"]]

    def fetch_all_regions(
        self,
        fuel_type: str,
        start_date: str,
        end_date: str,
        regions: Optional[list[str]] = None,
        max_workers: int = 3,
        diagnostics: Optional[list[dict]] = None,
    ) -> pd.DataFrame:
        """Fetch generation data for all regions for a given fuel type.

        Args:
            fuel_type: Fuel type code (WND, SUN, etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            regions: List of region codes (defaults to all non-US48 regions)
            max_workers: Number of parallel workers
            diagnostics: Optional list to collect diagnostic info

        Returns:
            DataFrame with columns [unique_id, ds, y]

        Raises:
            RuntimeError: If no regions could be fetched (complete failure)
        """
        if regions is None:
            # Only include regions with configured EIA respondent (exclude US48 and None)
            regions = [
                code for code, info in REGIONS.items()
                if code != "US48" and info.eia_respondent is not None
            ]

        all_dfs: list[pd.DataFrame] = []
        failed_regions: list[tuple[str, str]] = []  # (region, error_msg)
        diag_map: Optional[dict[str, dict]] = None

        if diagnostics is not None:
            diag_map = {region: {} for region in regions}

        def _run_one(region: str) -> tuple[str, pd.DataFrame, Optional[dict]]:
            d = diag_map[region] if diag_map is not None else None
            df = self.fetch_region(region, fuel_type, start_date, end_date, diag=d)
            return region, df, d

        if self.debug_requests:
            logger.info(
                "[fetch_all_regions] fuel=%s regions=%s max_workers=%d",
                fuel_type, regions, max_workers,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one, region): region for region in regions}
            for future in as_completed(futures):
                region = futures[future]
                d = diag_map.get(region) if diag_map is not None else None
                try:
                    _, df, d = future.result()
                    if diagnostics is not None and d is not None:
                        diagnostics.append(d)

                    if len(df) > 0:
                        all_dfs.append(df)
                        print(f"[OK] {region}: {len(df)} rows")
                    else:
                        print(f"[EMPTY] {region}: 0 rows")
                        failed_regions.append((region, "Empty response (0 rows)"))
                except Exception as e:
                    failed_regions.append((region, str(e)))
                    if diagnostics is not None:
                        if d is None:
                            d = {
                                "region": region,
                                "fuel_type": fuel_type,
                                "start_date": start_date,
                                "end_date": end_date,
                            }
                        d.setdefault("error", str(e))
                        diagnostics.append(d)
                    print(f"[FAIL] {region}: {e}")

        # Explicit validation: require at least one successful region
        if not all_dfs:
            error_details = "; ".join([f"{r[0]}({r[1][:80]})" for r in failed_regions])
            raise RuntimeError(
                f"[EIA][FETCH] Failed to fetch {fuel_type} data for ALL regions. "
                f"Failures: {error_details}. "
                f"Check EIA API availability, API key validity, network connectivity, "
                f"and consider increasing timeout or reducing concurrency."
            )

        # Warn if partial failure (some regions succeeded, some failed)
        if failed_regions:
            failed_count = len(failed_regions)
            total_count = len(regions)
            print(f"[WARNING] Partial {fuel_type} fetch: {failed_count}/{total_count} regions failed")
            for region, error_msg in failed_regions:
                # Print first 100 chars of error
                print(f"  - {region}: {error_msg[:100]}")

        combined = pd.concat(all_dfs, ignore_index=True)
        combined["unique_id"] = combined["region"] + "_" + combined["fuel_type"]
        combined = combined.rename(columns={"value": "y"})

        result = combined[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)

        print(f"[SUMMARY] {fuel_type} data: {result['unique_id'].nunique()} series, {len(result)} total rows")

        return result

    def get_series_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("unique_id").agg(
            count=("y", "count"),
            min_value=("y", "min"),
            max_value=("y", "max"),
            mean_value=("y", "mean"),
            zero_count=("y", lambda x: (x == 0).sum()),
        ).reset_index()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    fetcher = EIARenewableFetcher(debug_env=True)

    print("=== Testing Single Region Fetch ===")
    df_single = fetcher.fetch_region("CALI", "WND", "2024-12-01", "2024-12-03", debug=True)
    print(f"Single region: {len(df_single)} rows")
    print(df_single.head())

    print("\n=== Testing Multi-Region Fetch ===")
    df_multi = fetcher.fetch_all_regions("WND", "2024-12-01", "2024-12-03", regions=["CALI", "ERCO", "MISO"])
    print(f"\nMulti-region: {len(df_multi)} rows")
    print(f"Series: {df_multi['unique_id'].unique().tolist()}")

    print("\n=== Series Summary ===")
    print(fetcher.get_series_summary(df_multi))

    # sun checks:
    f = EIARenewableFetcher()
    df = f.fetch_region("CALI", "SUN", "2024-12-01", "2024-12-03", debug=True)
    print(df.head(), len(df))
