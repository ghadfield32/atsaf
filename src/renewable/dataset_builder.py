# file: src/renewable/dataset_builder.py
"""
Dataset Builder for Renewable Energy Forecasting

Consolidates all preprocessing transformations with transparent diagnostics:
1. Negative value handling (clamp, fail-loud, or hybrid)
2. Hourly grid enforcement (drop_incomplete or fail-loud)
3. Weather alignment (merge and validate)
4. Time feature engineering (hour/dow sin/cos)

Input:
  - Raw generation_df from EIA (unique_id, ds, y)
  - Raw weather_df from Open-Meteo (ds, region, weather_vars)

Output:
  - Modeling-ready DataFrame (unique_id, ds, y, weather_vars, time_features)
  - PreprocessingReport with comprehensive diagnostics

Guarantees:
  - Hourly grid enforced (no gaps or fail-loud)
  - Negative values handled per policy
  - Weather aligned to generation timestamps
  - Time features added
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Weather variables expected from Open-Meteo
WEATHER_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "direct_radiation",
    "diffuse_radiation",
    "cloud_cover",
]


@dataclass
class PreprocessingReport:
    """Educational diagnostics for what preprocessing occurred."""

    series_processed: int
    rows_input: int
    rows_output: int

    # Negative handling
    negative_values_found: Dict[str, Dict[str, Any]]  # uid -> {count, min, max, ratio, timestamps}
    negative_values_action: str  # "clamped" | "failed" | "passed"

    # Hourly grid
    series_dropped_incomplete: list[str]
    missing_hour_summary: Dict[str, Any]  # Summary of missing hour blocks

    # Weather alignment
    weather_coverage_by_region: Dict[str, float]
    weather_alignment_failures: list[Dict[str, Any]]

    # Features
    time_features_added: list[str]
    weather_features_added: list[str]

    timestamp: str


def _missing_hour_blocks(ds: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """
    Return contiguous blocks of missing hourly timestamps.
    Each tuple: (block_start, block_end, n_hours)
    """
    ds = pd.to_datetime(ds, errors="raise").sort_values()
    start, end = ds.iloc[0], ds.iloc[-1]
    expected = pd.date_range(start, end, freq="h")
    missing = expected.difference(ds)

    if missing.empty:
        return []

    blocks = []
    block_start = missing[0]
    prev = missing[0]
    for t in missing[1:]:
        if t - prev == pd.Timedelta(hours=1):
            prev = t
        else:
            n = int((prev - block_start).total_seconds() / 3600) + 1
            blocks.append((block_start, prev, n))
            block_start = t
            prev = t
    n = int((prev - block_start).total_seconds() / 3600) + 1
    blocks.append((block_start, prev, n))
    return blocks


def _hourly_grid_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate hourly grid coverage report per series."""
    cols = [
        "unique_id",
        "start",
        "end",
        "expected_hours",
        "actual_hours",
        "missing_hours",
        "missing_ratio",
        "n_missing_blocks",
        "largest_missing_block_hours",
    ]

    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for uid, g in df.groupby("unique_id"):
        g = g.sort_values("ds")
        start, end = g["ds"].iloc[0], g["ds"].iloc[-1]
        expected = pd.date_range(start, end, freq="h")
        missing = expected.difference(g["ds"])
        blocks = _missing_hour_blocks(g["ds"])

        rows.append(
            {
                "unique_id": uid,
                "start": start,
                "end": end,
                "expected_hours": int(len(expected)),
                "actual_hours": int(len(g)),
                "missing_hours": int(len(missing)),
                "missing_ratio": float(len(missing) / max(len(expected), 1)),
                "n_missing_blocks": int(len(blocks)),
                "largest_missing_block_hours": int(max([b[2] for b in blocks], default=0)),
            }
        )

    rep = pd.DataFrame(rows)
    return rep.sort_values(["missing_ratio", "missing_hours"], ascending=False)


def _handle_negative_values(
    df: pd.DataFrame,
    policy: str,
    label: str = "generation"
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle negative generation values according to policy.

    Args:
        df: DataFrame with columns [unique_id, ds, y]
        policy: "clamp" | "fail_loud" | "hybrid"
        label: Label for logging

    Returns:
        (processed_df, diagnostics_dict)
    """
    diagnostics = {
        'total_rows': len(df),
        'negative_count': int((df['y'] < 0).sum()),
        'negative_ratio': float((df['y'] < 0).sum() / len(df) if len(df) > 0 else 0),
        'series_with_negatives': {},
        'action_taken': policy
    }

    if diagnostics['negative_count'] == 0:
        logger.info(f"[{label}][NEGATIVES] No negative values found")
        diagnostics['action_taken'] = 'passed'
        return df.copy(), diagnostics

    # Analyze per-series
    for uid in df['unique_id'].unique():
        series_df = df[df['unique_id'] == uid]
        neg_mask = series_df['y'] < 0

        if neg_mask.any():
            neg_samples = series_df[neg_mask].head(10)

            diagnostics['series_with_negatives'][uid] = {
                'count': int(neg_mask.sum()),
                'ratio': float(neg_mask.sum() / len(series_df)),
                'min_value': float(series_df[neg_mask]['y'].min()),
                'max_value': float(series_df[neg_mask]['y'].max()),
                'mean_value': float(series_df[neg_mask]['y'].mean()),
                'sample_timestamps': neg_samples['ds'].astype(str).tolist()
            }

            logger.warning(
                f"[{label}][NEGATIVES] {uid}: count={neg_mask.sum()} "
                f"min={series_df[neg_mask]['y'].min():.2f} max={series_df[neg_mask]['y'].max():.2f}"
            )

    # Apply policy
    out = df.copy()

    if policy == "fail_loud":
        raise RuntimeError(
            f"[{label}][NEGATIVES] Found {diagnostics['negative_count']} negative values "
            f"({diagnostics['negative_ratio']:.2%}) across {len(diagnostics['series_with_negatives'])} series. "
            f"Policy=fail_loud prohibits negative values. Details: {diagnostics['series_with_negatives']}"
        )

    elif policy == "clamp":
        out['y'] = out['y'].clip(lower=0)
        logger.warning(
            f"[{label}][NEGATIVES] Clamped {diagnostics['negative_count']} negative values to 0 "
            f"({diagnostics['negative_ratio']:.2%})"
        )
        diagnostics['action_taken'] = 'clamped'

    elif policy == "hybrid":
        # Hybrid: clamp if <1% per series, fail if >=1%
        fail_series = []
        for uid, info in diagnostics['series_with_negatives'].items():
            if info['ratio'] >= 0.01:  # 1% threshold
                fail_series.append(uid)

        if fail_series:
            raise RuntimeError(
                f"[{label}][NEGATIVES] Policy=hybrid: Found series with >=1% negative values: {fail_series}. "
                f"This indicates data quality issues. Details: {diagnostics['series_with_negatives']}"
            )

        # Clamp small amounts
        out['y'] = out['y'].clip(lower=0)
        logger.warning(
            f"[{label}][NEGATIVES] Policy=hybrid: Clamped {diagnostics['negative_count']} negative values "
            f"(all series <1% ratio)"
        )
        diagnostics['action_taken'] = 'clamped (hybrid)'

    else:
        raise ValueError(f"Unknown negative handling policy: {policy}")

    return out, diagnostics


def _enforce_hourly_grid(
    df: pd.DataFrame,
    policy: str,
    label: str = "generation"
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enforce hourly grid according to policy.

    Args:
        df: DataFrame with columns [unique_id, ds, y]
        policy: "drop_incomplete_series" | "fail_loud"
        label: Label for logging

    Returns:
        (processed_df, diagnostics_dict)
    """
    if df.empty:
        raise RuntimeError(
            f"[{label}][GRID] Cannot enforce hourly grid: input dataframe is empty. "
            "This is upstream (fetch) failure, not a grid issue."
        )

    rep = _hourly_grid_report(df)
    if rep.empty:
        raise RuntimeError(
            f"[{label}][GRID] No series found to report on (rep empty). "
            "This indicates upstream emptiness or missing 'unique_id' groups."
        )

    diagnostics = {
        'series_count': len(rep),
        'series_with_missing': int((rep['missing_hours'] > 0).sum()),
        'total_missing_hours': int(rep['missing_hours'].sum()),
        'worst_missing_ratio': float(rep['missing_ratio'].max()),
        'policy': policy,
        'series_dropped': [],
        'missing_hour_blocks': rep.to_dict(orient='records')
    }

    worst = rep.iloc[0].to_dict()

    if worst["missing_hours"] == 0:
        logger.info(f"[{label}][GRID] No missing hours detected")
        return df.copy(), diagnostics

    logger.warning(f"[{label}][GRID] Missing hours detected:\n{rep.head(10).to_string(index=False)}")

    if policy == "drop_incomplete_series":
        bad_uids = rep.loc[rep["missing_hours"] > 0, "unique_id"].tolist()
        kept = df.loc[~df["unique_id"].isin(bad_uids)].copy()

        diagnostics['series_dropped'] = bad_uids

        logger.warning(
            f"[{label}][GRID] policy=drop_incomplete_series dropped={len(bad_uids)} "
            f"kept_series={kept['unique_id'].nunique()}"
        )

        if kept.empty:
            raise RuntimeError(f"[{label}][GRID] all series dropped due to missing hours")

        return kept, diagnostics

    elif policy == "fail_loud":
        worst_uid = worst["unique_id"]
        g = df[df["unique_id"] == worst_uid].sort_values("ds")
        blocks = _missing_hour_blocks(g["ds"])
        raise RuntimeError(
            f"[{label}][GRID] Missing hours detected (no imputation). "
            f"worst_unique_id={worst_uid} missing_hours={worst['missing_hours']} "
            f"missing_ratio={worst['missing_ratio']:.3f} blocks(sample)={blocks[:3]}"
        )

    else:
        raise ValueError(f"Unknown hourly grid policy: {policy}")


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features (hour, day of week)."""
    out = df.copy()
    out["hour"] = out["ds"].dt.hour
    out["dow"] = out["ds"].dt.dayofweek

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)

    return out.drop(columns=["hour", "dow"])


def _align_weather(
    df: pd.DataFrame,
    weather_df: pd.DataFrame
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Align weather data to generation timestamps.

    Args:
        df: Generation DataFrame with columns [unique_id, ds, y, ...]
        weather_df: Weather DataFrame with columns [ds, region, weather_vars]

    Returns:
        (merged_df, diagnostics_dict)
    """
    # Extract region from unique_id (e.g., "CALI_WND" -> "CALI")
    work = df.copy()
    work["region"] = work["unique_id"].str.split("_").str[0]

    # Check required columns
    if not {"ds", "region"}.issubset(weather_df.columns):
        raise ValueError("weather_df must have columns ['ds', 'region']")

    wcols = [c for c in WEATHER_VARS if c in weather_df.columns]
    if not wcols:
        raise ValueError("weather_df has none of expected WEATHER_VARS")

    # Merge
    merged = work.merge(
        weather_df[["ds", "region"] + wcols],
        on=["ds", "region"],
        how="left",
        validate="many_to_one",
    )

    # Check for missing weather after merge
    missing_any = merged[wcols].isna().any(axis=1)

    diagnostics = {
        'weather_vars': wcols,
        'merge_rows': len(merged),
        'missing_weather_rows': int(missing_any.sum()),
        'missing_weather_ratio': float(missing_any.sum() / len(merged) if len(merged) > 0 else 0),
        'coverage_by_region': {}
    }

    # Per-region coverage
    for region in merged['region'].unique():
        region_df = merged[merged['region'] == region]
        region_missing = region_df[wcols].isna().any(axis=1).sum()
        diagnostics['coverage_by_region'][region] = float(
            1 - (region_missing / len(region_df)) if len(region_df) > 0 else 0
        )

    if missing_any.any():
        sample = merged.loc[missing_any, ["unique_id", "ds", "region"] + wcols].head(10)
        logger.error(
            f"[WEATHER][ALIGN] Missing weather after merge rows={int(missing_any.sum())}. "
            f"Sample:\n{sample.to_string(index=False)}"
        )
        raise RuntimeError(
            f"[WEATHER][ALIGN] Missing weather after merge rows={int(missing_any.sum())}. "
            f"Check that weather_df covers the same date range and regions as generation_df."
        )

    # Drop region column (not needed for modeling)
    output = merged.drop(columns=["region"])

    logger.info(f"[WEATHER][ALIGN] Successfully merged {len(wcols)} weather variables")

    return output, diagnostics


def build_modeling_dataset(
    generation_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    *,
    negative_policy: str = "clamp",
    hourly_grid_policy: str = "drop_incomplete_series",
    output_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, PreprocessingReport]:
    """
    Build modeling-ready dataset with comprehensive diagnostics.

    Args:
        generation_df: Raw generation DataFrame (unique_id, ds, y)
        weather_df: Raw weather DataFrame (ds, region, weather_vars)
        negative_policy: "clamp" | "fail_loud" | "hybrid"
        hourly_grid_policy: "drop_incomplete_series" | "fail_loud"
        output_dir: Optional directory to save detailed diagnostics

    Returns:
        (modeling_df, preprocessing_report)

    Raises:
        RuntimeError: If data quality issues detected and policy is fail_loud
        ValueError: If invalid policy specified
    """
    from datetime import datetime

    logger.info("=" * 80)
    logger.info("DATASET BUILDER - Starting preprocessing")
    logger.info("=" * 80)

    # Validate inputs
    req = {"unique_id", "ds", "y"}
    if not req.issubset(generation_df.columns):
        raise ValueError(f"generation_df missing cols={sorted(req - set(generation_df.columns))}")

    if generation_df.empty:
        raise RuntimeError(
            "[GENERATION] Empty generation dataframe. "
            "This is upstream (EIA fetch/cache) failure."
        )

    rows_input = len(generation_df)
    series_input = generation_df['unique_id'].nunique()

    logger.info(f"Input: {rows_input:,} rows, {series_input} series")
    logger.info(f"Policies: negative={negative_policy}, hourly_grid={hourly_grid_policy}")

    # Step 1: Handle negative values
    work, neg_diagnostics = _handle_negative_values(
        generation_df,
        policy=negative_policy,
        label="generation"
    )

    # Step 2: Enforce hourly grid
    work, grid_diagnostics = _enforce_hourly_grid(
        work,
        policy=hourly_grid_policy,
        label="generation"
    )

    # Step 3: Add time features
    work = _add_time_features(work)
    time_features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    logger.info(f"[TIME_FEATURES] Added: {time_features}")

    # Step 4: Align weather (if provided)
    weather_features = []
    weather_diagnostics = {}

    if weather_df is not None and not weather_df.empty:
        work, weather_diagnostics = _align_weather(work, weather_df)
        weather_features = [c for c in WEATHER_VARS if c in work.columns]
    else:
        logger.warning("[WEATHER] No weather data provided, skipping alignment")

    # Final validation
    y_null = work["y"].isna()
    if y_null.any():
        sample = work.loc[y_null, ["unique_id", "ds", "y"]].head(25)
        raise RuntimeError(
            f"[GENERATION][Y] Found null y values after preprocessing (rows={int(y_null.sum())}). "
            f"Sample:\n{sample.to_string(index=False)}"
        )

    rows_output = len(work)
    series_output = work['unique_id'].nunique()

    # Create report
    report = PreprocessingReport(
        series_processed=series_output,
        rows_input=rows_input,
        rows_output=rows_output,
        negative_values_found=neg_diagnostics.get('series_with_negatives', {}),
        negative_values_action=neg_diagnostics.get('action_taken', 'unknown'),
        series_dropped_incomplete=grid_diagnostics.get('series_dropped', []),
        missing_hour_summary=grid_diagnostics,
        weather_coverage_by_region=weather_diagnostics.get('coverage_by_region', {}),
        weather_alignment_failures=[],
        time_features_added=time_features,
        weather_features_added=weather_features,
        timestamp=datetime.now().isoformat()
    )

    logger.info("=" * 80)
    logger.info("DATASET BUILDER - Complete")
    logger.info(f"Output: {rows_output:,} rows, {series_output} series")
    logger.info(f"Dropped: {rows_input - rows_output:,} rows ({(rows_input - rows_output)/rows_input*100:.1f}%)")
    logger.info(f"Series dropped: {len(report.series_dropped_incomplete)}")
    logger.info(f"Features added: {len(time_features)} time + {len(weather_features)} weather")
    logger.info("=" * 80)

    # Save detailed diagnostics if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_file = output_dir / "preprocessing_report.json"
        report_file.write_text(json.dumps(asdict(report), indent=2, default=str))
        logger.info(f"[REPORT] Saved to: {report_file}")

        # Save negative samples if any
        if neg_diagnostics.get('negative_count', 0) > 0:
            neg_detail = output_dir / "negative_values_detail.json"
            neg_detail.write_text(json.dumps(neg_diagnostics, indent=2, default=str))
            logger.info(f"[NEGATIVES] Details saved to: {neg_detail}")

        # Save grid report
        if grid_diagnostics.get('series_dropped'):
            grid_detail = output_dir / "missing_hours_detail.json"
            grid_detail.write_text(json.dumps(grid_diagnostics, indent=2, default=str))
            logger.info(f"[GRID] Details saved to: {grid_detail}")

    return work, report


if __name__ == "__main__":
    """
    Build production-ready modeling dataset from raw renewable energy data.

    MAIN PATH (default):
      - Loads raw generation and weather data
      - Runs ONE canonical preprocessing pass
      - Saves modeling-ready dataset
      - Reports diagnostics

    DEMO MODE (set RUN_DEMOS=True):
      - Shows how different policies compare
      - Demonstrates clamp, fail_loud, hybrid approaches
      - Educational tool only (not production path)

    Usage:
        python -m src.renewable.dataset_builder           # Production path
        RUN_DEMOS=1 python -m src.renewable.dataset_builder  # With demos
    """
    import sys
    import logging
    import os
    from pathlib import Path
    from dataclasses import asdict

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Configuration: set RUN_DEMOS=1 in environment to run policy comparisons
    RUN_DEMOS = os.environ.get('RUN_DEMOS', '0').lower() in ('1', 'true', 'yes')

    print("=" * 80)
    print("DATASET BUILDER - Production Pipeline")
    print("=" * 80)
    if RUN_DEMOS:
        print("(DEMO MODE: Running policy comparisons)")
    print("=" * 80)

    # Step 1: Load & validate raw data
    print("\n[1/3] Loading & validating raw data...")

    generation_path = Path("data/renewable/generation.parquet")
    weather_path = Path("data/renewable/weather.parquet")

    if not generation_path.exists():
        print(f"[ERROR] Generation data not found at {generation_path}")
        print("   Please run the pipeline first: python -m src.renewable.tasks --preset 24h")
        sys.exit(1)

    if not weather_path.exists():
        print(f"[ERROR] Weather data not found at {weather_path}")
        print("   Please run the pipeline first: python -m src.renewable.tasks --preset 24h")
        sys.exit(1)

    generation_df = pd.read_parquet(generation_path)
    weather_df = pd.read_parquet(weather_path)

    print(f"   [OK] Generation: {len(generation_df):,} rows, {generation_df['unique_id'].nunique()} series")
    print(f"   [OK] Weather: {len(weather_df):,} rows")
    print(f"   [OK] Date range: {generation_df['ds'].min().date()} to {generation_df['ds'].max().date()}")

    # Quick data sanity checks
    neg_count = (generation_df['y'] < 0).sum()
    neg_ratio = neg_count / len(generation_df) if len(generation_df) > 0 else 0

    has_duplicates = generation_df.duplicated(subset=['unique_id', 'ds']).sum()

    print(f"\n   [CHECKS]")
    print(f"      Negatives: {neg_count} ({neg_ratio:.4%})" + (" [WARNING]" if neg_count > 0 else " [CLEAN]"))
    print(f"      Duplicates: {has_duplicates}" + (" [WARNING]" if has_duplicates > 0 else " [CLEAN]"))

    # Determine best policy based on data characteristics
    print("\n   [POLICY SELECTION]")
    if neg_count == 0:
        chosen_negative_policy = "fail_loud"
        print(f"      Negatives: None → negative_policy='fail_loud' (fail fast on upstream changes)")
    elif neg_ratio < 0.01:
        chosen_negative_policy = "fail_loud"
        print(f"      Negatives: {neg_ratio:.4%} (rare) → negative_policy='fail_loud' (detect if they appear)")
    else:
        chosen_negative_policy = "clamp"
        print(f"      Negatives: {neg_ratio:.4%} (substantial) → negative_policy='clamp' (tolerate & log)")

    chosen_grid_policy = "drop_incomplete_series"
    print(f"      Grid enforcement: {chosen_grid_policy} (drop series with gaps)")

    # Step 2: Build dataset (canonical production path)
    print("\n[2/3] Building modeling dataset (ONE canonical build)...")

    output_dir = Path("data/renewable/preprocessing/latest")

    modeling_df, report = build_modeling_dataset(
        generation_df,
        weather_df,
        negative_policy=chosen_negative_policy,
        hourly_grid_policy=chosen_grid_policy,
        output_dir=output_dir
    )

    print(f"\n   [RESULT] Preprocessing Summary:")
    print(f"      Input:  {report.rows_input:,} rows → Output: {report.rows_output:,} rows")
    print(f"      Dropped: {report.rows_input - report.rows_output:,} rows ({(report.rows_input - report.rows_output)/report.rows_input*100:.2f}%)")
    print(f"      Series: {report.series_processed} (dropped {len(report.series_dropped_incomplete)})")
    print(f"      Negative action: {report.negative_values_action}")
    print(f"      Features: {len(report.time_features_added)} time + {len(report.weather_features_added)} weather")

    # Step 3: Dataset inspection
    print("\n[3/3] Dataset Inspection...")

    print(f"\n   [DATA] Modeling-Ready Dataset:")
    print(f"      Shape: {modeling_df.shape}")
    print(f"      Memory: {modeling_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print(f"\n   [FEATURES] Added:")
    print(f"      Time: {report.time_features_added}")
    print(f"      Weather: {report.weather_features_added[:3]} ... ({len(report.weather_features_added)} total)")

    print(f"\n   [QUALITY] Data Checks:")
    print(f"      Nulls in y: {modeling_df['y'].isna().sum()}")
    print(f"      Negatives in y: {(modeling_df['y'] < 0).sum()}")
    print(f"      Duplicates: {modeling_df.duplicated(subset=['unique_id', 'ds']).sum()}")
    weather_nulls = modeling_df[report.weather_features_added].isna().sum().sum() if report.weather_features_added else 0
    print(f"      Weather nulls: {weather_nulls}")

    print(f"\n   [SAMPLE] First 3 rows:")
    print(modeling_df.head(3)[['unique_id', 'ds', 'y', 'hour_sin', 'temperature_2m']].to_string(index=False))

    print("\n" + "=" * 80)
    print("[SUCCESS] PRODUCTION BUILD COMPLETE")
    print("=" * 80)
    print(f"\nSaved to: {output_dir}/")
    print(f"  - preprocessing_report.json   # Full diagnostics")
    if report.negative_values_found:
        print(f"  - negative_values_detail.json # Negative sample analysis")
    if report.series_dropped_incomplete:
        print(f"  - missing_hours_detail.json   # Dropped series details")
    print("=" * 80)

    # ============================================================================
    # DEMO MODE (optional, run with: RUN_DEMOS=1 python -m src.renewable.dataset_builder)
    # ============================================================================
    if RUN_DEMOS:
        print("\n" + "=" * 80)
        print("[DEMO MODE] Policy Comparison (Educational)")
        print("=" * 80)
        print("(This demonstrates different policies. For production, use the canonical build above.)\n")

        # Demo 1: Try fail_loud if negatives exist
        if neg_count > 0:
            print("[DEMO 1/2] Testing negative_policy='fail_loud'...")
            try:
                modeling_df_fail, _ = build_modeling_dataset(
                    generation_df, weather_df,
                    negative_policy="fail_loud",
                    hourly_grid_policy="drop_incomplete_series",
                    output_dir=None
                )
                print("      [OK] No negatives detected (would pass production)")
            except RuntimeError as e:
                print(f"      [RAISED] {str(e)[:150]}...")

        # Demo 2: Try hybrid
        print("\n[DEMO 2/2] Testing negative_policy='hybrid' (if <1% per series)...")
        try:
            modeling_df_hybrid, report_hybrid = build_modeling_dataset(
                generation_df, weather_df,
                negative_policy="hybrid",
                hourly_grid_policy="drop_incomplete_series",
                output_dir=None
            )
            print(f"      [OK] Policy='hybrid' succeeded ({report_hybrid.negative_values_action})")
        except RuntimeError as e:
            print(f"      [RAISED] {str(e)[:150]}...")

        print("\n[NOTE] Demos show policy behavior. Production uses: negative_policy='{}', grid='drop_incomplete'".format(chosen_negative_policy))

    print("\n[NEXT] Use modeling_df for forecasting (weather_df=None, already merged)")
    print("=" * 80)
