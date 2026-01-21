#!/usr/bin/env python
# file: scripts/run_eda_renewable.py
"""
CLI entry point for renewable energy EDA analysis.

Usage:
    # Run on latest data
    python scripts/run_eda_renewable.py

    # Run on specific date range
    python scripts/run_eda_renewable.py --start 2025-01-01 --end 2025-12-31

    # Run on specific regions/fuels
    python scripts/run_eda_renewable.py --regions CALI,ERCO --fuels WND,SUN

    # Custom output directory
    python scripts/run_eda_renewable.py --output reports/custom_eda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.renewable.eda import generate_eda_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate EDA report for renewable energy forecasting data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all available data
  python scripts/run_eda_renewable.py

  # Filter by date range
  python scripts/run_eda_renewable.py --start 2025-01-01 --end 2025-12-31

  # Filter by regions and fuels
  python scripts/run_eda_renewable.py --regions CALI,ERCO --fuels WND

  # Custom output location
  python scripts/run_eda_renewable.py --output reports/my_eda
        """
    )

    parser.add_argument(
        '--generation',
        type=str,
        default='data/renewable/generation.parquet',
        help='Path to generation parquet file (default: data/renewable/generation.parquet)'
    )

    parser.add_argument(
        '--weather',
        type=str,
        default='data/renewable/weather.parquet',
        help='Path to weather parquet file (default: data/renewable/weather.parquet)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='reports/renewable/eda',
        help='Output directory for EDA reports (default: reports/renewable/eda)'
    )

    parser.add_argument(
        '--start',
        type=str,
        help='Filter start date (YYYY-MM-DD format)'
    )

    parser.add_argument(
        '--end',
        type=str,
        help='Filter end date (YYYY-MM-DD format)'
    )

    parser.add_argument(
        '--regions',
        type=str,
        help='Comma-separated region codes to include (e.g., CALI,ERCO,MISO)'
    )

    parser.add_argument(
        '--fuels',
        type=str,
        help='Comma-separated fuel types to include (e.g., WND,SUN)'
    )

    args = parser.parse_args()

    # Resolve paths
    generation_path = Path(args.generation)
    weather_path = Path(args.weather)
    output_dir = Path(args.output)

    # Check if files exist
    if not generation_path.exists():
        print(f"❌ Error: Generation file not found: {generation_path}")
        print(f"   Please run the pipeline first or provide a valid --generation path")
        sys.exit(1)

    if not weather_path.exists():
        print(f"❌ Error: Weather file not found: {weather_path}")
        print(f"   Please run the pipeline first or provide a valid --weather path")
        sys.exit(1)

    # Load data
    print(f"📂 Loading generation data from: {generation_path}")
    generation_df = pd.read_parquet(generation_path)
    print(f"   ✓ Loaded {len(generation_df):,} rows, {generation_df['unique_id'].nunique()} series")

    print(f"📂 Loading weather data from: {weather_path}")
    weather_df = pd.read_parquet(weather_path)
    print(f"   ✓ Loaded {len(weather_df):,} rows")

    # Apply filters
    original_gen_count = len(generation_df)

    # Date range filter
    if args.start or args.end:
        generation_df['ds'] = pd.to_datetime(generation_df['ds'])
        weather_df['ds'] = pd.to_datetime(weather_df['ds'])

        if args.start:
            start_date = pd.to_datetime(args.start)
            generation_df = generation_df[generation_df['ds'] >= start_date]
            weather_df = weather_df[weather_df['ds'] >= start_date]
            print(f"🔍 Filtered to dates >= {args.start}")

        if args.end:
            end_date = pd.to_datetime(args.end)
            generation_df = generation_df[generation_df['ds'] <= end_date]
            weather_df = weather_df[weather_df['ds'] <= end_date]
            print(f"🔍 Filtered to dates <= {args.end}")

    # Region filter
    if args.regions:
        regions = [r.strip().upper() for r in args.regions.split(',')]
        generation_df = generation_df[
            generation_df['unique_id'].str.split('_').str[0].isin(regions)
        ]
        weather_df['region'] = weather_df.get('region', weather_df['ds'].astype(str))  # Handle if region col missing
        if 'region' in weather_df.columns:
            weather_df = weather_df[weather_df['region'].isin(regions)]
        print(f"🔍 Filtered to regions: {', '.join(regions)}")

    # Fuel type filter
    if args.fuels:
        fuels = [f.strip().upper() for f in args.fuels.split(',')]
        generation_df = generation_df[
            generation_df['unique_id'].str.split('_').str[1].isin(fuels)
        ]
        print(f"🔍 Filtered to fuel types: {', '.join(fuels)}")

    filtered_gen_count = len(generation_df)
    print(f"   ✓ After filters: {filtered_gen_count:,} generation rows ({filtered_gen_count/original_gen_count*100:.1f}%)")

    # Check if data remains
    if generation_df.empty:
        print("❌ Error: No data remaining after applying filters")
        sys.exit(1)

    if weather_df.empty:
        print("⚠️  Warning: No weather data remaining after applying filters")
        print("   EDA will run but weather alignment analysis will be limited")

    # Generate report
    print("\n" + "=" * 80)
    print("GENERATING EDA REPORT")
    print("=" * 80)

    try:
        html_file = generate_eda_report(
            generation_df=generation_df,
            weather_df=weather_df,
            output_dir=output_dir
        )

        print("\n" + "=" * 80)
        print("✅ SUCCESS")
        print("=" * 80)
        print(f"📊 HTML Report: {html_file}")
        print(f"📁 All outputs: {html_file.parent}")
        print("\nOpen the HTML report in your browser to view results:")
        print(f"   {html_file.absolute()}")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR GENERATING EDA REPORT")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
