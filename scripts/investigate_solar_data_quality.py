"""
Solar Data Quality Investigation Script

This script performs comprehensive analysis of solar generation data to identify:
1. Timezone consistency issues
2. Timestamp alignment problems
3. Night-time generation anomalies
4. Solar radiation data quality
5. Generation vs radiation correlation
6. Forecast quality issues

Usage:
    python scripts/investigate_solar_data_quality.py > solar_investigation_report.txt

Output:
    Detailed report with 6 checks for each solar series, identifying root causes
    of data quality issues that may affect model performance.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run comprehensive solar data quality investigation."""
    print("=" * 80)
    print("SOLAR DATA QUALITY INVESTIGATION")
    print("=" * 80)

    # Load all data
    try:
        gen_df = pd.read_parquet("data/renewable/generation.parquet")
        weather_df = pd.read_parquet("data/renewable/weather.parquet")
        forecasts_df = pd.read_parquet("data/renewable/forecasts.parquet")
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Required data file not found: {e}")
        print("Please run the pipeline first to generate data files.")
        return 1

    # Focus on all solar series
    solar_series = [s for s in gen_df["unique_id"].unique() if s.endswith("_SUN")]
    print(f"\nFound {len(solar_series)} solar series: {solar_series}")

    for series_id in solar_series:
        print(f"\n{'=' * 80}")
        print(f"SERIES: {series_id}")
        print(f"{'=' * 80}")

        region = series_id.split("_")[0]
        series_gen = gen_df[gen_df["unique_id"] == series_id].copy()
        series_weather = weather_df[weather_df["region"] == region].copy()
        series_forecast = forecasts_df[forecasts_df["unique_id"] == series_id].copy()

        # === CHECK 1: Timezone Consistency ===
        print("\n[CHECK 1] TIMEZONE CONSISTENCY")
        print(f"  Generation timezone: {series_gen['ds'].dt.tz}")
        print(f"  Weather timezone: {series_weather['ds'].dt.tz}")
        if "ds" in series_forecast.columns:
            print(f"  Forecast timezone: {series_forecast['ds'].dt.tz}")

        if series_gen["ds"].dt.tz != series_weather["ds"].dt.tz:
            print("  ⚠️ WARNING: Timezone mismatch detected!")

        # === CHECK 2: Timestamp Alignment ===
        print("\n[CHECK 2] TIMESTAMP ALIGNMENT")
        merged = series_gen.merge(
            series_weather,
            left_on="ds",
            right_on="ds",
            how="inner",
            suffixes=("_gen", "_weather"),
        )

        total_gen_rows = len(series_gen)
        matched_rows = len(merged)
        match_rate = (matched_rows / total_gen_rows * 100) if total_gen_rows > 0 else 0

        print(f"  Generation rows: {total_gen_rows}")
        print(f"  Weather rows: {len(series_weather)}")
        print(f"  Matched rows: {matched_rows} ({match_rate:.1f}%)")

        if match_rate < 99:
            print(f"  ⚠️ WARNING: {100 - match_rate:.1f}% of generation data has no matching weather!")

            # Find unmatched timestamps
            unmatched = series_gen[~series_gen["ds"].isin(merged["ds"])]
            print(f"  Sample unmatched generation timestamps:")
            print(unmatched["ds"].head(10).tolist())

        # === CHECK 3: Night-Time Generation in Training Data ===
        print("\n[CHECK 3] NIGHT-TIME GENERATION (Training Data)")
        series_gen["hour"] = series_gen["ds"].dt.hour
        night_mask = (series_gen["hour"] < 6) | (series_gen["hour"] > 18)
        night_gen = series_gen[night_mask]

        print(
            f"  Night hours (< 6 or > 18): {len(night_gen)} / {len(series_gen)} rows ({len(night_gen) / len(series_gen) * 100:.1f}%)"
        )
        print(f"  Mean night generation: {night_gen['y'].mean():.1f} MW")
        print(f"  Max night generation: {night_gen['y'].max():.1f} MW")
        print(f"  Median night generation: {night_gen['y'].median():.1f} MW")

        if night_gen["y"].max() > 100:
            print(f"  ⚠️ WARNING: High night-time generation detected (max: {night_gen['y'].max():.1f} MW)")
            print(f"  Sample high night generation:")
            high_night_samples = night_gen[night_gen["y"] > 100].sort_values("y", ascending=False).head(5)
            print(high_night_samples[["ds", "y", "hour"]])

        # === CHECK 4: Solar Radiation During Night Hours ===
        print("\n[CHECK 4] SOLAR RADIATION DURING NIGHT HOURS")
        night_merged = merged[(merged["ds"].dt.hour < 6) | (merged["ds"].dt.hour > 18)]

        if len(night_merged) > 0:
            print(f"  Mean direct radiation at night: {night_merged['direct_radiation'].mean():.1f} W/m²")
            print(f"  Mean diffuse radiation at night: {night_merged['diffuse_radiation'].mean():.1f} W/m²")
            print(f"  Max direct radiation at night: {night_merged['direct_radiation'].max():.1f} W/m²")

            if night_merged["direct_radiation"].max() > 10:
                print(f"  ⚠️ WARNING: Non-zero radiation at night detected!")
                print(f"  Sample:")
                print(
                    night_merged[night_merged["direct_radiation"] > 10][
                        ["ds", "hour", "direct_radiation", "diffuse_radiation"]
                    ].head()
                )

        # === CHECK 5: Generation vs Radiation Correlation ===
        print("\n[CHECK 5] GENERATION vs RADIATION CORRELATION")
        day_mask = (merged["ds"].dt.hour >= 6) & (merged["ds"].dt.hour <= 18)
        day_merged = merged[day_mask]

        if len(day_merged) > 0:
            total_radiation = day_merged["direct_radiation"] + day_merged["diffuse_radiation"]
            correlation = np.corrcoef(day_merged["y"], total_radiation)[0, 1]
            print(f"  Daytime correlation (generation vs total_radiation): {correlation:.3f}")

            if correlation < 0.5:
                print(f"  ⚠️ WARNING: Low correlation between generation and radiation!")

            # Check for high generation with zero radiation
            zero_rad_high_gen = day_merged[(total_radiation < 10) & (day_merged["y"] > 1000)]
            if len(zero_rad_high_gen) > 0:
                print(
                    f"  ⚠️ WARNING: {len(zero_rad_high_gen)} daytime rows with high generation but zero radiation!"
                )
                print(f"  Sample:")
                print(zero_rad_high_gen[["ds", "y", "direct_radiation", "diffuse_radiation"]].head())

        # === CHECK 6: Forecast Quality (Night-Time Predictions) ===
        if len(series_forecast) > 0:
            print("\n[CHECK 6] FORECAST NIGHT-TIME PREDICTIONS")
            series_forecast["hour"] = pd.to_datetime(series_forecast["ds"]).dt.hour
            night_fcst = series_forecast[(series_forecast["hour"] < 6) | (series_forecast["hour"] > 18)]

            print(f"  Night forecast rows: {len(night_fcst)} / {len(series_forecast)}")
            if "yhat" in night_fcst.columns:
                print(f"  Mean night forecast: {night_fcst['yhat'].mean():.1f} MW")
                print(f"  Max night forecast: {night_fcst['yhat'].max():.1f} MW")

                if night_fcst["yhat"].max() > 100:
                    print(f"  ⚠️ WARNING: High night-time forecasts detected!")
                    high_fcst = night_fcst[night_fcst["yhat"] > 100].sort_values("yhat", ascending=False).head(5)
                    print(high_fcst[["ds", "yhat", "hour"]])

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
    print("\nNext steps based on findings:")
    print("1. If timezone mismatch → Convert all timestamps to UTC")
    print("2. If timestamp alignment < 99% → Investigate missing weather data")
    print("3. If high night generation in training data → Check EIA data source/processing")
    print("4. If zero radiation but high generation → Weather-generation misalignment")
    print("5. If low correlation → Investigate feature engineering or data quality")
    print("\nRun 'python scripts/fix_solar_data_issues.py' to apply corrections")

    return 0


if __name__ == "__main__":
    sys.exit(main())
