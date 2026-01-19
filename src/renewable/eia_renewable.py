# src/renewable/eia_renewable.py
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from src.renewable.regions import REGIONS, get_eia_respondent, validate_fuel_type, validate_region

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

    def __init__(self, api_key: Optional[str] = None, *, debug_env: bool = False):
        """
        Initialize API key. Pulls from:
        1) explicit api_key argument
        2) environment variable EIA_API_KEY (optionally loaded from .env)
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

        # Debug without leaking the key
        if debug_env:
            masked = self.api_key[:4] + "..." + self.api_key[-4:] if len(self.api_key) >= 8 else "***"
            logger.info("EIA_API_KEY loaded (masked): %s", masked)

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

            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()

            records, meta = self._extract_eia_response(payload, request_url=resp.url)

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

        df["ds"] = pd.to_datetime(df["period"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df["region"] = region
        df["fuel_type"] = fuel_type

        df = df.dropna(subset=["ds", "value"]).sort_values("ds").reset_index(drop=True)

        # DEBUG: Log negative values for investigation
        neg_mask = df["value"] < 0
        if neg_mask.any():
            neg_count = int(neg_mask.sum())
            neg_min = float(df.loc[neg_mask, "value"].min())
            neg_max = float(df.loc[neg_mask, "value"].max())
            neg_sample = df.loc[neg_mask, ["ds", "value"]].head(10)
            logger.warning(
                "[fetch_region][NEGATIVE] region=%s fuel=%s count=%d min=%.2f max=%.2f",
                region, fuel_type, neg_count, neg_min, neg_max,
            )
            for _, row in neg_sample.iterrows():
                logger.warning("  ds=%s value=%.2f", row["ds"], row["value"])

            # Clamp negative values to zero (preserves hourly grid for modeling)
            # Note: Removing rows would create gaps that cause series to be dropped
            logger.warning(
                "[fetch_region][CLAMP] Clamping %d negative values to 0 for %s_%s (%.1f%%)",
                neg_count, region, fuel_type, 100 * neg_count / max(len(df), 1),
            )
            df["value"] = df["value"].clip(lower=0)

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
        if regions is None:
            regions = [r for r in REGIONS.keys() if r != "US48"]

        all_dfs: list[pd.DataFrame] = []

        def _run_one(region: str) -> tuple[str, pd.DataFrame, dict]:
            d: dict = {}
            df = self.fetch_region(region, fuel_type, start_date, end_date, diag=d)
            return region, df, d

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one, region): region for region in regions}
            for future in as_completed(futures):
                region = futures[future]
                try:
                    _, df, d = future.result()
                    if diagnostics is not None:
                        diagnostics.append(d)

                    if len(df) > 0:
                        all_dfs.append(df)
                        print(f"[OK] {region}: {len(df)} rows")
                    else:
                        print(f"[EMPTY] {region}: 0 rows")
                except Exception as e:
                    if diagnostics is not None:
                        diagnostics.append({
                            "region": region,
                            "fuel_type": fuel_type,
                            "start_date": start_date,
                            "end_date": end_date,
                            "error": str(e),
                        })
                    print(f"[FAIL] {region}: {e}")

        if not all_dfs:
            return pd.DataFrame(columns=["unique_id", "ds", "y"])

        combined = pd.concat(all_dfs, ignore_index=True)
        combined["unique_id"] = combined["region"] + "_" + combined["fuel_type"]
        combined = combined.rename(columns={"value": "y"})
        return combined[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)

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
