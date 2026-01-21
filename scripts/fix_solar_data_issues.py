"""
Solar Data Correction Script

Based on findings from investigate_solar_data_quality.py, this script:
1. Fixes timezone inconsistencies (converts all to UTC)
2. Validates generation-weather alignment (drops unmatched rows)
3. Removes physically impossible data (radiation=0, generation>0)

Philosophy:
- Only fix data quality issues, don't filter legitimate patterns
- Preserve as much data as possible
- Document all changes with before/after metrics

Usage:
    # Review investigation findings first!
    python scripts/investigate_solar_data_quality.py > report.txt

    # Then run corrections
    python scripts/fix_solar_data_issues.py

Output:
    - data/renewable/generation_corrected.parquet (corrected generation data)
    - Prints summary of changes
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def fix_timezone_issues(df: pd.DataFrame, target_tz="UTC") -> pd.DataFrame:
    """Ensure all timestamps are in UTC."""
    if df["ds"].dt.tz is None:
        print(f"  Adding timezone (assuming {target_tz})")
        df = df.copy()
        df["ds"] = df["ds"].dt.tz_localize(target_tz)
    elif str(df["ds"].dt.tz) != target_tz:
        print(f"  Converting from {df['ds'].dt.tz} to {target_tz}")
        df = df.copy()
        df["ds"] = df["ds"].dt.tz_convert(target_tz)
    else:
        print(f"  Already in {target_tz}, no changes needed")

    return df


def validate_generation_weather_alignment(gen_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure generation data has matching weather."""
    gen_df = gen_df.copy()
    gen_df["region"] = gen_df["unique_id"].str.split("_").str[0]

    merged = gen_df.merge(
        weather_df[["region", "ds"]], on=["region", "ds"], how="inner", indicator=True
    )

    missing_count = len(gen_df) - len(merged)
    if missing_count > 0:
        print(f"  ⚠️ Dropped {missing_count} generation rows with no matching weather")
        print(f"     ({missing_count / len(gen_df) * 100:.2f}% of data)")
    else:
        print(f"  ✓ All generation rows have matching weather")

    return merged.drop(columns=["_merge", "region"])


def remove_physically_impossible_data(gen_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Remove data points that violate physics (ONLY truly impossible cases)."""
    gen_df = gen_df.copy()
    gen_df["region"] = gen_df["unique_id"].str.split("_").str[0]

    # Merge with weather to get radiation values
    merged = gen_df.merge(weather_df, on=["region", "ds"], how="left")

    # Identify solar series
    solar_mask = merged["unique_id"].str.endswith("_SUN")

    # ONLY remove if solar radiation is literally 0 but generation is high
    # (not filtering by hour - let model learn natural patterns)
    if "direct_radiation" in merged.columns and "diffuse_radiation" in merged.columns:
        total_radiation = merged["direct_radiation"] + merged["diffuse_radiation"]
        impossible_mask = solar_mask & (total_radiation == 0) & (merged["y"] > 10)

        if impossible_mask.sum() > 0:
            print(
                f"  ⚠️ Removing {impossible_mask.sum()} rows: zero radiation + high generation (>10 MW)"
            )
            print(f"     ({impossible_mask.sum() / len(merged) * 100:.2f}% of data)")

            # Show sample of removed data
            removed_sample = merged[impossible_mask].head(3)
            print(f"\n  Sample removed data:")
            for _, row in removed_sample.iterrows():
                print(f"    {row['unique_id']} @ {row['ds']}: y={row['y']:.1f} MW, radiation=0 W/m²")

            merged = merged[~impossible_mask]
        else:
            print(f"  ✓ No physically impossible data found")
    else:
        print(f"  ⚠️ Warning: Radiation columns not found, skipping physical validation")

    # Keep only original generation columns
    return merged[["unique_id", "ds", "y"]]


def main():
    """Run data correction pipeline."""
    print("=" * 80)
    print("SOLAR DATA CORRECTION")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading data...")
    try:
        gen_df = pd.read_parquet("data/renewable/generation.parquet")
        weather_df = pd.read_parquet("data/renewable/weather.parquet")
        print(f"  Loaded {len(gen_df)} generation rows, {len(weather_df)} weather rows")
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Required data file not found: {e}")
        print("Please run the pipeline first to generate data files.")
        return 1

    original_row_count = len(gen_df)

    # Fix timezones
    print("\n[2/5] Fixing timezones...")
    gen_df = fix_timezone_issues(gen_df)
    weather_df = fix_timezone_issues(weather_df)

    # Validate alignment
    print("\n[3/5] Validating alignment...")
    gen_df = validate_generation_weather_alignment(gen_df, weather_df)

    # Remove physically impossible data
    print("\n[4/5] Removing physically impossible data...")
    cleaned_df = remove_physically_impossible_data(gen_df, weather_df)

    # Save corrected data
    print("\n[5/5] Saving corrected data...")
    output_path = Path("data/renewable/generation_corrected.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_parquet(output_path, index=False)
    print(f"  ✓ Saved to {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("CORRECTION SUMMARY")
    print("=" * 80)
    print(f"Original rows:  {original_row_count}")
    print(f"Corrected rows: {len(cleaned_df)}")
    print(f"Rows removed:   {original_row_count - len(cleaned_df)} ({(original_row_count - len(cleaned_df)) / original_row_count * 100:.2f}%)")
    print(f"\nNext step: Retrain models with corrected data")
    print(f"  export RENEWABLE_USE_CORRECTED_DATA=true")
    print(f"  python -m src.renewable.jobs.run_hourly")

    return 0


if __name__ == "__main__":
    sys.exit(main())
