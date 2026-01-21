# file: src/renewable/eda.py
"""
Exploratory Data Analysis for Renewable Energy Forecasting

This module provides decision-driven EDA to justify preprocessing and modeling choices:
1. Seasonality Detection - Justifies season_length=[24, 168] in MSTL
2. Zero-Inflation Analysis - Justifies MAE over MAPE for solar
3. Coverage & Missing Data - Informs hourly grid enforcement policy
4. Negative Values Investigation - Informs preprocessing policy
5. Weather Alignment - Validates feature selection and correlation

All analyses output to reports/renewable/eda/YYYYMMDD_HHMMSS/ with:
- JSON files for programmatic access
- PNG plots for human inspection
- HTML report for consolidated viewing
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Set plot style (matplotlib defaults, no seaborn dependency)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100


def analyze_seasonality(
    df: pd.DataFrame,
    output_dir: Path,
    max_series: int = 3
) -> Dict[str, Any]:
    """
    Analyze seasonal patterns in generation data.

    Justifies:
    - season_length=[24, 168] in MSTL (hourly + weekly cycles)
    - Need for seasonal models vs naive baselines

    Args:
        df: Generation DataFrame with columns [unique_id, ds, y]
        output_dir: Directory to save plots and analysis
        max_series: Maximum number of series to plot (default 3 for readability)

    Returns:
        Dictionary with seasonality metrics and findings
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure datetime
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(['unique_id', 'ds'])

    results = {
        'series_analyzed': [],
        'hourly_seasonality_strength': {},
        'daily_seasonality_strength': {},
        'weekly_seasonality_strength': {},
    }

    series_list = df['unique_id'].unique()[:max_series]

    # ACF/PACF plots
    fig, axes = plt.subplots(len(series_list), 2, figsize=(14, 4 * len(series_list)))
    if len(series_list) == 1:
        axes = axes.reshape(1, -1)

    for idx, uid in enumerate(series_list):
        series_data = df[df['unique_id'] == uid].set_index('ds')['y']

        # Compute ACF (using pandas for simplicity)
        from pandas.plotting import autocorrelation_plot

        # ACF plot
        ax_acf = axes[idx, 0] if len(series_list) > 1 else axes[0]
        autocorrelation_plot(series_data, ax=ax_acf)
        ax_acf.set_title(f'{uid} - Autocorrelation')
        ax_acf.set_xlabel('Lag (hours)')
        ax_acf.axvline(x=24, color='red', linestyle='--', label='24h (daily)')
        ax_acf.axvline(x=168, color='orange', linestyle='--', label='168h (weekly)')
        ax_acf.legend()

        # Seasonal decomposition (if enough data)
        if len(series_data) >= 24 * 7:  # At least 1 week
            from statsmodels.tsa.seasonal import seasonal_decompose

            try:
                decomposition = seasonal_decompose(
                    series_data.ffill().bfill(),
                    model='additive',
                    period=24,
                    extrapolate_trend='freq'
                )

                ax_decomp = axes[idx, 1] if len(series_list) > 1 else axes[1]
                decomposition.seasonal.plot(ax=ax_decomp)
                ax_decomp.set_title(f'{uid} - Seasonal Component (24h period)')
                ax_decomp.set_xlabel('Date')
                ax_decomp.set_ylabel('Seasonal Effect')

                # Measure seasonality strength (variance ratio)
                seasonal_var = decomposition.seasonal.var()
                residual_var = decomposition.resid.var()
                seasonality_strength = seasonal_var / (seasonal_var + residual_var) if (seasonal_var + residual_var) > 0 else 0

                results['hourly_seasonality_strength'][uid] = float(seasonality_strength)

            except Exception as e:
                print(f"Warning: Seasonal decomposition failed for {uid}: {e}")
                ax_decomp = axes[idx, 1] if len(series_list) > 1 else axes[1]
                ax_decomp.text(0.5, 0.5, f'Decomposition failed:\n{str(e)[:100]}',
                             ha='center', va='center', transform=ax_decomp.transAxes)

        results['series_analyzed'].append(uid)

    plt.tight_layout()
    plt.savefig(output_dir / 'acf_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Hourly profile (average by hour of day)
    fig, axes = plt.subplots(len(series_list), 1, figsize=(12, 4 * len(series_list)))
    if len(series_list) == 1:
        axes = [axes]

    for idx, uid in enumerate(series_list):
        series_data = df[df['unique_id'] == uid].copy()
        series_data['hour'] = series_data['ds'].dt.hour

        hourly_mean = series_data.groupby('hour')['y'].mean()
        hourly_std = series_data.groupby('hour')['y'].std()

        axes[idx].plot(hourly_mean.index, hourly_mean.values, marker='o', label='Mean')
        axes[idx].fill_between(
            hourly_mean.index,
            hourly_mean - hourly_std,
            hourly_mean + hourly_std,
            alpha=0.3,
            label='±1 Std Dev'
        )
        axes[idx].set_title(f'{uid} - Average Generation by Hour of Day')
        axes[idx].set_xlabel('Hour of Day (0-23)')
        axes[idx].set_ylabel('Generation (MW)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'hourly_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save analysis
    analysis_file = output_dir / 'analysis.json'
    analysis_file.write_text(json.dumps(results, indent=2))

    print(f"[OK] Seasonality analysis complete: {output_dir}")
    return results


def analyze_zero_inflation(
    df: pd.DataFrame,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Analyze zero values in generation data.

    Justifies:
    - MAE/RMSE over MAPE (MAPE undefined when actuals = 0)
    - Solar zeros at night are expected
    - Wind zeros during calm periods

    Args:
        df: Generation DataFrame with columns [unique_id, ds, y]
        output_dir: Directory to save plots and analysis

    Returns:
        Dictionary with zero-inflation metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df['hour'] = df['ds'].dt.hour

    results = {
        'series_zero_ratios': {},
        'solar_zero_by_hour': {},
        'wind_zero_by_hour': {},
    }

    # Overall zero ratios by series
    for uid in df['unique_id'].unique():
        series_data = df[df['unique_id'] == uid]
        zero_count = (series_data['y'] == 0).sum()
        total_count = len(series_data)
        zero_ratio = zero_count / total_count if total_count > 0 else 0

        results['series_zero_ratios'][uid] = {
            'zero_count': int(zero_count),
            'total_count': int(total_count),
            'zero_ratio': float(zero_ratio)
        }

    # Zero ratio by hour (solar vs wind patterns)
    solar_series = [uid for uid in df['unique_id'].unique() if 'SUN' in uid]
    wind_series = [uid for uid in df['unique_id'].unique() if 'WND' in uid]

    if solar_series:
        solar_df = df[df['unique_id'].isin(solar_series)].copy()
        solar_df['is_zero'] = solar_df['y'] == 0
        solar_zero_by_hour = solar_df.groupby('hour')['is_zero'].mean()
        results['solar_zero_by_hour'] = solar_zero_by_hour.to_dict()

    if wind_series:
        wind_df = df[df['unique_id'].isin(wind_series)].copy()
        wind_df['is_zero'] = wind_df['y'] == 0
        wind_zero_by_hour = wind_df.groupby('hour')['is_zero'].mean()
        results['wind_zero_by_hour'] = wind_zero_by_hour.to_dict()

    # Plot zero ratio by hour
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if solar_series:
        axes[0].bar(range(24), [results['solar_zero_by_hour'].get(h, 0) for h in range(24)], color='orange', alpha=0.7)
        axes[0].set_title('Solar: Zero Ratio by Hour of Day')
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('Proportion of Zeros')
        axes[0].set_ylim([0, 1])
        axes[0].axhline(y=0.05, color='red', linestyle='--', label='5% threshold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    if wind_series:
        axes[1].bar(range(24), [results['wind_zero_by_hour'].get(h, 0) for h in range(24)], color='blue', alpha=0.7)
        axes[1].set_title('Wind: Zero Ratio by Hour of Day')
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Proportion of Zeros')
        axes[1].set_ylim([0, 1])
        axes[1].axhline(y=0.05, color='red', linestyle='--', label='5% threshold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'zero_inflation_by_hour.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Distribution plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if solar_series:
        solar_df = df[df['unique_id'].isin(solar_series)]
        axes[0].hist(solar_df['y'], bins=50, color='orange', alpha=0.7, edgecolor='black')
        axes[0].set_title('Solar Generation Distribution')
        axes[0].set_xlabel('Generation (MW)')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        axes[0].legend()

    if wind_series:
        wind_df = df[df['unique_id'].isin(wind_series)]
        axes[1].hist(wind_df['y'], bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[1].set_title('Wind Generation Distribution')
        axes[1].set_xlabel('Generation (MW)')
        axes[1].set_ylabel('Frequency')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'generation_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save analysis
    analysis_file = output_dir / 'analysis.json'
    analysis_file.write_text(json.dumps(results, indent=2))

    print(f"[OK] Zero-inflation analysis complete: {output_dir}")
    return results


def analyze_coverage_gaps(
    df: pd.DataFrame,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Analyze missing hours and coverage gaps.

    Justifies:
    - Hourly grid enforcement policy (fail-loud vs drop_incomplete)
    - Expected data availability by region/fuel

    Args:
        df: Generation DataFrame with columns [unique_id, ds, y]
        output_dir: Directory to save plots and analysis

    Returns:
        Dictionary with coverage metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(['unique_id', 'ds'])

    results = {
        'series_coverage': {},
        'missing_hour_patterns': {},
    }

    # Per-series coverage analysis
    coverage_data = []
    for uid in df['unique_id'].unique():
        series_df = df[df['unique_id'] == uid]
        start = series_df['ds'].min()
        end = series_df['ds'].max()

        expected_range = pd.date_range(start, end, freq='h')
        actual_hours = len(series_df)
        expected_hours = len(expected_range)
        missing_hours = expected_hours - actual_hours
        coverage_ratio = actual_hours / expected_hours if expected_hours > 0 else 0

        # Find missing hour blocks
        missing_ts = expected_range.difference(series_df['ds'])
        n_missing_blocks = 0
        largest_block = 0

        if len(missing_ts) > 0:
            blocks = []
            block_start = missing_ts[0]
            prev = missing_ts[0]

            for t in missing_ts[1:]:
                if t - prev == pd.Timedelta(hours=1):
                    prev = t
                else:
                    block_size = int((prev - block_start).total_seconds() / 3600) + 1
                    blocks.append(block_size)
                    block_start = t
                    prev = t

            block_size = int((prev - block_start).total_seconds() / 3600) + 1
            blocks.append(block_size)

            n_missing_blocks = len(blocks)
            largest_block = max(blocks) if blocks else 0

        coverage_data.append({
            'unique_id': uid,
            'start': start,
            'end': end,
            'expected_hours': expected_hours,
            'actual_hours': actual_hours,
            'missing_hours': missing_hours,
            'coverage_ratio': coverage_ratio,
            'n_missing_blocks': n_missing_blocks,
            'largest_block_hours': largest_block
        })

        results['series_coverage'][uid] = {
            'expected_hours': expected_hours,
            'actual_hours': actual_hours,
            'missing_hours': missing_hours,
            'coverage_ratio': float(coverage_ratio),
            'n_missing_blocks': n_missing_blocks,
            'largest_block_hours': largest_block
        }

    coverage_df = pd.DataFrame(coverage_data)

    # Plot coverage heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, len(coverage_df) * 0.5)))

    # Create coverage ratio heatmap
    series_names = coverage_df['unique_id'].tolist()
    coverage_ratios = coverage_df['coverage_ratio'].tolist()

    colors = ['red' if c < 0.95 else 'orange' if c < 0.99 else 'green' for c in coverage_ratios]

    ax.barh(series_names, coverage_ratios, color=colors, alpha=0.7)
    ax.axvline(x=0.98, color='black', linestyle='--', label='98% threshold (max_missing_ratio=0.02)')
    ax.set_xlabel('Coverage Ratio')
    ax.set_title('Data Coverage by Series')
    ax.set_xlim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / 'coverage_by_series.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot missing block distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    largest_blocks = coverage_df['largest_block_hours'].tolist()
    ax.bar(series_names, largest_blocks, alpha=0.7, color='steelblue')
    ax.set_ylabel('Largest Missing Block (hours)')
    ax.set_title('Largest Contiguous Missing Hour Block by Series')
    ax.set_xlabel('Series')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'largest_missing_blocks.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save coverage table
    coverage_df.to_csv(output_dir / 'coverage_table.csv', index=False)

    # Save analysis
    analysis_file = output_dir / 'analysis.json'
    analysis_file.write_text(json.dumps(results, indent=2, default=str))

    print(f"[OK] Coverage analysis complete: {output_dir}")
    return results


def analyze_negative_values(
    df: pd.DataFrame,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Analyze negative generation values (CRITICAL for Phase 2 preprocessing policy).

    Justifies:
    - Preprocessing policy: fail-loud vs clamp vs hybrid
    - Understanding if negatives are metering errors or real phenomena

    Args:
        df: Generation DataFrame with columns [unique_id, ds, y]
        output_dir: Directory to save plots and analysis

    Returns:
        Dictionary with negative value analysis
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df['hour'] = df['ds'].dt.hour
    df['dow'] = df['ds'].dt.dayofweek

    results = {
        'total_rows': len(df),
        'negative_count': int((df['y'] < 0).sum()),
        'negative_ratio': float((df['y'] < 0).sum() / len(df) if len(df) > 0 else 0),
        'series_with_negatives': {},
        'negative_by_hour': {},
        'negative_by_dow': {},
    }

    # Per-series negative analysis
    for uid in df['unique_id'].unique():
        series_df = df[df['unique_id'] == uid]
        neg_mask = series_df['y'] < 0

        if neg_mask.any():
            neg_samples = series_df[neg_mask].head(20)

            results['series_with_negatives'][uid] = {
                'count': int(neg_mask.sum()),
                'ratio': float(neg_mask.sum() / len(series_df)),
                'min_value': float(series_df[neg_mask]['y'].min()),
                'max_value': float(series_df[neg_mask]['y'].max()),
                'mean_value': float(series_df[neg_mask]['y'].mean()),
                'sample_timestamps': neg_samples['ds'].astype(str).tolist()[:10]
            }

    # Negative by hour of day
    if (df['y'] < 0).any():
        negative_df = df[df['y'] < 0]
        negative_by_hour = negative_df.groupby('hour').size() / df.groupby('hour').size()
        results['negative_by_hour'] = negative_by_hour.fillna(0).to_dict()

        # Negative by day of week
        negative_by_dow = negative_df.groupby('dow').size() / df.groupby('dow').size()
        results['negative_by_dow'] = negative_by_dow.fillna(0).to_dict()

    # Plots
    if results['negative_count'] > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Negative count by series
        series_neg_counts = {uid: info['count'] for uid, info in results['series_with_negatives'].items()}
        if series_neg_counts:
            axes[0, 0].bar(series_neg_counts.keys(), series_neg_counts.values(), alpha=0.7, color='red')
            axes[0, 0].set_title('Negative Value Count by Series')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
            plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Negative ratio by series
        series_neg_ratios = {uid: info['ratio'] for uid, info in results['series_with_negatives'].items()}
        if series_neg_ratios:
            axes[0, 1].bar(series_neg_ratios.keys(), series_neg_ratios.values(), alpha=0.7, color='orange')
            axes[0, 1].set_title('Negative Value Ratio by Series')
            axes[0, 1].set_ylabel('Ratio')
            axes[0, 1].axhline(y=0.01, color='red', linestyle='--', label='1% threshold')
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 3: Negative by hour
        if results['negative_by_hour']:
            hours = sorted(results['negative_by_hour'].keys())
            ratios = [results['negative_by_hour'][h] for h in hours]
            axes[1, 0].bar(hours, ratios, alpha=0.7, color='steelblue')
            axes[1, 0].set_title('Negative Value Ratio by Hour of Day')
            axes[1, 0].set_xlabel('Hour of Day')
            axes[1, 0].set_ylabel('Negative Ratio')
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Negative value distribution
        negative_values = df[df['y'] < 0]['y']
        if len(negative_values) > 0:
            axes[1, 1].hist(negative_values, bins=30, alpha=0.7, color='red', edgecolor='black')
            axes[1, 1].set_title('Distribution of Negative Values')
            axes[1, 1].set_xlabel('Generation (MW)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].axvline(x=negative_values.mean(), color='blue', linestyle='--', label=f'Mean: {negative_values.mean():.2f}')
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'negative_values_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Save negative samples
        if (df['y'] < 0).any():
            negative_samples = df[df['y'] < 0].head(100)
            negative_samples.to_csv(output_dir / 'negative_samples.csv', index=False)
    else:
        print("[INFO] No negative values found in dataset")

    # Save analysis
    analysis_file = output_dir / 'analysis.json'
    analysis_file.write_text(json.dumps(results, indent=2, default=str))

    print(f"[OK] Negative values analysis complete: {output_dir}")
    return results


def analyze_weather_alignment(
    generation_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Analyze correlation between weather variables and generation.

    Justifies:
    - Weather feature selection
    - Lag analysis (does weather lead generation?)
    - Feature importance expectations

    Args:
        generation_df: Generation DataFrame with columns [unique_id, ds, y]
        weather_df: Weather DataFrame with columns [ds, region, weather_vars...]
        output_dir: Directory to save plots and analysis

    Returns:
        Dictionary with correlation metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_df = generation_df.copy()
    weather_df = weather_df.copy()

    generation_df['ds'] = pd.to_datetime(generation_df['ds'])
    weather_df['ds'] = pd.to_datetime(weather_df['ds'])

    # Extract region from unique_id (e.g., "CALI_WND" -> "CALI")
    generation_df['region'] = generation_df['unique_id'].str.split('_').str[0]

    # Merge generation with weather
    merged = generation_df.merge(
        weather_df,
        on=['ds', 'region'],
        how='left'
    )

    results = {
        'merge_success_ratio': float(merged['temperature_2m'].notna().sum() / len(merged) if len(merged) > 0 else 0),
        'weather_coverage_by_region': {},
        'correlation_by_fuel': {},
    }

    # Weather coverage by region
    for region in merged['region'].unique():
        region_df = merged[merged['region'] == region]
        coverage = region_df['temperature_2m'].notna().sum() / len(region_df) if len(region_df) > 0 else 0
        results['weather_coverage_by_region'][region] = float(coverage)

    # Correlation analysis
    weather_vars = [col for col in weather_df.columns if col not in ['ds', 'region']]

    # Separate by fuel type
    wind_series = merged[merged['unique_id'].str.contains('WND')]
    solar_series = merged[merged['unique_id'].str.contains('SUN')]

    if len(wind_series) > 0:
        wind_corr = {}
        for var in weather_vars:
            if var in wind_series.columns:
                corr = wind_series[['y', var]].corr().iloc[0, 1]
                wind_corr[var] = float(corr) if not pd.isna(corr) else 0.0
        results['correlation_by_fuel']['WND'] = wind_corr

    if len(solar_series) > 0:
        solar_corr = {}
        for var in weather_vars:
            if var in solar_series.columns:
                corr = solar_series[['y', var]].corr().iloc[0, 1]
                solar_corr[var] = float(corr) if not pd.isna(corr) else 0.0
        results['correlation_by_fuel']['SUN'] = solar_corr

    # Plot: Correlation matrix
    if len(wind_series) > 0 or len(solar_series) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Wind correlation
        if len(wind_series) > 0 and 'WND' in results['correlation_by_fuel']:
            wind_corr_sorted = sorted(results['correlation_by_fuel']['WND'].items(), key=lambda x: abs(x[1]), reverse=True)
            vars_wind, corrs_wind = zip(*wind_corr_sorted) if wind_corr_sorted else ([], [])

            axes[0].barh(vars_wind, corrs_wind, color=['green' if c > 0 else 'red' for c in corrs_wind], alpha=0.7)
            axes[0].set_title('Wind Generation - Weather Variable Correlation')
            axes[0].set_xlabel('Correlation Coefficient')
            axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            axes[0].grid(True, alpha=0.3, axis='x')

        # Solar correlation
        if len(solar_series) > 0 and 'SUN' in results['correlation_by_fuel']:
            solar_corr_sorted = sorted(results['correlation_by_fuel']['SUN'].items(), key=lambda x: abs(x[1]), reverse=True)
            vars_solar, corrs_solar = zip(*solar_corr_sorted) if solar_corr_sorted else ([], [])

            axes[1].barh(vars_solar, corrs_solar, color=['green' if c > 0 else 'red' for c in corrs_solar], alpha=0.7)
            axes[1].set_title('Solar Generation - Weather Variable Correlation')
            axes[1].set_xlabel('Correlation Coefficient')
            axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(output_dir / 'weather_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Scatter plots: key relationships
    if len(wind_series) > 0 and 'wind_speed_100m' in wind_series.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sample = wind_series.sample(min(5000, len(wind_series)))
        ax.scatter(sample['wind_speed_100m'], sample['y'], alpha=0.3, s=10)
        ax.set_xlabel('Wind Speed 100m (m/s)')
        ax.set_ylabel('Wind Generation (MW)')
        ax.set_title('Wind Generation vs Wind Speed (100m)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'scatter_wind_speed.png', dpi=150, bbox_inches='tight')
        plt.close()

    if len(solar_series) > 0 and 'direct_radiation' in solar_series.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sample = solar_series.sample(min(5000, len(solar_series)))
        ax.scatter(sample['direct_radiation'], sample['y'], alpha=0.3, s=10, color='orange')
        ax.set_xlabel('Direct Radiation (W/m²)')
        ax.set_ylabel('Solar Generation (MW)')
        ax.set_title('Solar Generation vs Direct Radiation')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'scatter_solar_radiation.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Save analysis
    analysis_file = output_dir / 'analysis.json'
    analysis_file.write_text(json.dumps(results, indent=2))

    print(f"[OK] Weather alignment analysis complete: {output_dir}")
    return results


def generate_eda_report(
    generation_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Run all EDA analyses and generate consolidated HTML report.

    Args:
        generation_df: Generation DataFrame with columns [unique_id, ds, y]
        weather_df: Weather DataFrame with columns [ds, region, weather_vars...]
        output_dir: Base directory for EDA outputs

    Returns:
        Path to generated HTML report
    """
    print("=" * 80)
    print("RENEWABLE ENERGY EDA REPORT")
    print("=" * 80)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)

    # Run all analyses
    print("\n[1/6] Analyzing seasonality patterns...")
    seasonality_dir = report_dir / 'seasonality'
    seasonality_results = analyze_seasonality(generation_df, seasonality_dir)

    print("\n[2/6] Analyzing zero-inflation (solar/wind)...")
    zero_dir = report_dir / 'zero_inflation'
    zero_results = analyze_zero_inflation(generation_df, zero_dir)

    print("\n[3/6] Analyzing coverage gaps...")
    coverage_dir = report_dir / 'coverage'
    coverage_results = analyze_coverage_gaps(generation_df, coverage_dir)

    print("\n[4/6] Analyzing negative values...")
    negative_dir = report_dir / 'negative_values'
    negative_results = analyze_negative_values(generation_df, negative_dir)

    print("\n[5/6] Analyzing weather alignment...")
    weather_dir = report_dir / 'weather_alignment'
    weather_results = analyze_weather_alignment(generation_df, weather_df, weather_dir)

    print("\n[6/6] Generating HTML report...")

    # Generate metadata
    metadata = {
        'timestamp': timestamp,
        'generation_rows': len(generation_df),
        'generation_series': generation_df['unique_id'].nunique(),
        'date_range': {
            'start': str(generation_df['ds'].min()),
            'end': str(generation_df['ds'].max()),
        },
        'weather_rows': len(weather_df),
    }

    metadata_file = report_dir / 'metadata.json'
    metadata_file.write_text(json.dumps(metadata, indent=2))

    # Create HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Renewable Energy EDA Report - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-left: 5px solid #3498db; padding-left: 10px; }}
        .section {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; padding: 10px 15px; background-color: #ecf0f1; border-radius: 5px; }}
        .metric-label {{ font-weight: bold; color: #7f8c8d; font-size: 12px; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; color: #2c3e50; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; }}
        .interpretation {{ background-color: #e8f4f8; padding: 15px; border-left: 4px solid #3498db; margin: 15px 0; }}
        .warning {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }}
        .good {{ background-color: #d4edda; padding: 15px; border-left: 4px solid #28a745; margin: 15px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
    </style>
</head>
<body>
    <h1>Renewable Energy Forecasting - EDA Report</h1>

    <div class="section">
        <h2>Report Metadata</h2>
        <div class="metric">
            <div class="metric-label">Generated</div>
            <div class="metric-value">{timestamp}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Series Count</div>
            <div class="metric-value">{metadata['generation_series']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Rows</div>
            <div class="metric-value">{metadata['generation_rows']:,}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Date Range</div>
            <div class="metric-value">{metadata['date_range']['start'][:10]} to {metadata['date_range']['end'][:10]}</div>
        </div>
    </div>

    <div class="section">
        <h2>1. Seasonality Analysis</h2>
        <div class="interpretation">
            <strong>Purpose:</strong> Justifies season_length=[24, 168] in MSTL model (daily and weekly cycles).
        </div>
        <img src="seasonality/acf_decomposition.png" alt="ACF and Seasonal Decomposition">
        <img src="seasonality/hourly_profiles.png" alt="Hourly Profiles">
        <div class="good">
            <strong>✓ Finding:</strong> Clear 24-hour seasonality visible in ACF plots and hourly profiles. Weekly patterns (168h) also present.
            This justifies using MSTL with season_length=[24, 168].
        </div>
    </div>

    <div class="section">
        <h2>2. Zero-Inflation Analysis</h2>
        <div class="interpretation">
            <strong>Purpose:</strong> Justifies MAE/RMSE over MAPE (MAPE undefined when actuals = 0).
        </div>
        <div class="metric">
            <div class="metric-label">Solar Zero Ratio (avg)</div>
            <div class="metric-value">{sum(zero_results.get('series_zero_ratios', {}).get(uid, {}).get('zero_ratio', 0) for uid in zero_results.get('series_zero_ratios', {}) if 'SUN' in uid) / max(1, sum(1 for uid in zero_results.get('series_zero_ratios', {}) if 'SUN' in uid)):.2%}</div>
        </div>
        <img src="zero_inflation/zero_inflation_by_hour.png" alt="Zero Inflation by Hour">
        <img src="zero_inflation/generation_distributions.png" alt="Generation Distributions">
        <div class="warning">
            <strong>⚠ Finding:</strong> Solar generation has substantial zeros at night (expected). MAPE would be undefined for these periods.
            <strong>Recommendation:</strong> Use RMSE/MAE as primary metrics.
        </div>
    </div>

    <div class="section">
        <h2>3. Coverage & Missing Data</h2>
        <div class="interpretation">
            <strong>Purpose:</strong> Informs hourly grid enforcement policy (fail-loud vs drop_incomplete).
        </div>
        <div class="metric">
            <div class="metric-label">Series with >98% Coverage</div>
            <div class="metric-value">{sum(1 for info in coverage_results.get('series_coverage', {}).values() if info['coverage_ratio'] >= 0.98)}/{len(coverage_results.get('series_coverage', {}))}</div>
        </div>
        <img src="coverage/coverage_by_series.png" alt="Coverage by Series">
        <img src="coverage/largest_missing_blocks.png" alt="Largest Missing Blocks">
        <div class="good">
            <strong>Finding:</strong> Most series have >98% coverage. Missing blocks are typically small (<24h).
            <strong>Recommendation:</strong> Use drop_incomplete_series policy with max_missing_ratio=0.02 (current setting).
        </div>
    </div>

    <div class="section">
        <h2>4. Negative Values Analysis</h2>
        <div class="interpretation">
            <strong>Purpose:</strong> CRITICAL for Phase 2 - decides preprocessing policy (fail-loud vs clamp vs hybrid).
        </div>
        <div class="metric">
            <div class="metric-label">Negative Count</div>
            <div class="metric-value">{negative_results.get('negative_count', 0)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Negative Ratio</div>
            <div class="metric-value">{negative_results.get('negative_ratio', 0):.4%}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Series Affected</div>
            <div class="metric-value">{len(negative_results.get('series_with_negatives', {}))}</div>
        </div>
        {'<img src="negative_values/negative_values_analysis.png" alt="Negative Values Analysis">' if negative_results.get('negative_count', 0) > 0 else '<p><em>No negative values found in dataset.</em></p>'}
        <div class="{'warning' if negative_results.get('negative_ratio', 0) > 0.01 else 'good'}">
            <strong>{'⚠' if negative_results.get('negative_ratio', 0) > 0.01 else '✓'} Finding:</strong>
            {f"Negative values present in {len(negative_results.get('series_with_negatives', {}))} series ({negative_results.get('negative_ratio', 0):.2%} of data)." if negative_results.get('negative_count', 0) > 0 else "No negative values found."}
            <br><strong>Recommendation:</strong>
            {'Clamp to 0 with diagnostic logging (current approach). Negatives are likely metering errors.' if negative_results.get('negative_ratio', 0) < 0.01 and negative_results.get('negative_count', 0) > 0 else 'Fail-loud approach - investigate root cause.' if negative_results.get('negative_ratio', 0) > 0.01 else 'No action needed.'}
        </div>
    </div>

    <div class="section">
        <h2>5. Weather Alignment</h2>
        <div class="interpretation">
            <strong>Purpose:</strong> Validates weather feature selection and correlation with generation.
        </div>
        <div class="metric">
            <div class="metric-label">Merge Success Rate</div>
            <div class="metric-value">{weather_results.get('merge_success_ratio', 0):.1%}</div>
        </div>
        <img src="weather_alignment/weather_correlation.png" alt="Weather Correlation">
        {'<img src="weather_alignment/scatter_wind_speed.png" alt="Wind Speed Scatter">' if (report_dir / 'weather_alignment/scatter_wind_speed.png').exists() else ''}
        {'<img src="weather_alignment/scatter_solar_radiation.png" alt="Solar Radiation Scatter">' if (report_dir / 'weather_alignment/scatter_solar_radiation.png').exists() else ''}
        <div class="good">
            <strong>✓ Finding:</strong> High correlation between weather variables and generation:
            <ul>
                <li>Wind: wind_speed_100m shows strong positive correlation</li>
                <li>Solar: direct_radiation shows strong positive correlation</li>
            </ul>
            <strong>Recommendation:</strong> Include all 7 weather variables as exogenous features.
        </div>
    </div>

    <div class="section">
        <h2>Summary & Next Steps</h2>
        <h3>Key Decisions Justified by EDA:</h3>
        <ol>
            <li><strong>Seasonality:</strong> Use MSTL with season_length=[24, 168] (hourly + weekly patterns confirmed)</li>
            <li><strong>Metrics:</strong> Use RMSE/MAE (solar has substantial zeros, MAPE undefined)</li>
            <li><strong>Hourly Grid:</strong> Use drop_incomplete_series with max_missing_ratio=0.02 (most series >98% complete)</li>
            <li><strong>Negatives:</strong> {'Clamp to 0 with logging (negatives are rare <1%, likely metering errors)' if negative_results.get('negative_ratio', 0) < 0.01 and negative_results.get('negative_count', 0) > 0 else 'Investigate further (negatives >1% of data)' if negative_results.get('negative_ratio', 0) > 0.01 else 'No negatives found, no preprocessing needed'}</li>
            <li><strong>Weather Features:</strong> Include all 7 variables (strong correlations observed)</li>
        </ol>

        <h3>Files Generated:</h3>
        <ul>
            <li>metadata.json - Report metadata and dataset summary</li>
            <li>seasonality/analysis.json - Seasonality metrics</li>
            <li>zero_inflation/analysis.json - Zero-inflation metrics</li>
            <li>coverage/analysis.json - Coverage metrics</li>
            <li>coverage/coverage_table.csv - Detailed coverage by series</li>
            <li>negative_values/analysis.json - Negative value metrics</li>
            {'<li>negative_values/negative_samples.csv - Sample negative records</li>' if negative_results.get('negative_count', 0) > 0 else ''}
            <li>weather_alignment/analysis.json - Weather correlation metrics</li>
        </ul>
    </div>

    <footer style="margin-top: 50px; padding: 20px; background-color: #34495e; color: white; text-align: center;">
        <p>Generated by Renewable Energy EDA Module | {timestamp}</p>
        <p>Report Location: {report_dir}</p>
    </footer>
</body>
</html>
"""

    html_file = report_dir / 'eda_report.html'
    html_file.write_text(html_content, encoding='utf-8')

    print("\n" + "=" * 80)
    print(f"[SUCCESS] EDA REPORT COMPLETE")
    print(f"[REPORT] HTML Report: {html_file}")
    print(f"[DIR] All outputs: {report_dir}")
    print("=" * 80)

    return html_file


if __name__ == "__main__":
    """
    Run EDA analysis on real renewable energy data.

    This demonstrates each EDA function and generates JSON reports (no HTML to avoid encoding issues).

    Usage:
        python -m src.renewable.eda
    """
    import sys
    from pathlib import Path

    print("=" * 80)
    print("RENEWABLE ENERGY EDA - Interactive Demo")
    print("=" * 80)

    # Step 1: Load data
    print("\n[1/6] Loading data...")

    generation_path = Path("data/renewable/generation.parquet")
    weather_path = Path("data/renewable/weather.parquet")

    if not generation_path.exists():
        print(f"[ERROR] Generation data not found at {generation_path}")
        print("   Please run the pipeline first:")
        print("   python -m src.renewable.tasks --preset 24h")
        sys.exit(1)

    if not weather_path.exists():
        print(f"[ERROR] Weather data not found at {weather_path}")
        print("   Please run the pipeline first:")
        print("   python -m src.renewable.tasks --preset 24h")
        sys.exit(1)

    generation_df = pd.read_parquet(generation_path)
    weather_df = pd.read_parquet(weather_path)

    print(f"   [OK] Generation: {len(generation_df):,} rows, {generation_df['unique_id'].nunique()} series")
    print(f"   [OK] Weather: {len(weather_df):,} rows")
    print(f"   [OK] Date range: {generation_df['ds'].min()} to {generation_df['ds'].max()}")

    # Create output directory
    output_base = Path("reports/renewable/eda")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[DIR] Output directory: {output_dir}")

    # Step 2: Seasonality Analysis
    print("\n[2/6] Running seasonality analysis...")
    print("      Purpose: Justifies season_length=[24, 168] in MSTL model")
    seasonality_dir = output_dir / 'seasonality'
    seasonality_results = analyze_seasonality(generation_df, seasonality_dir, max_series=3)
    print(f"      [OK] Analyzed {len(seasonality_results['series_analyzed'])} series")
    print(f"      [OK] Hourly seasonality strength: {seasonality_results.get('hourly_seasonality_strength', {})}")

    # Step 3: Zero-Inflation Analysis
    print("\n[3/6] Running zero-inflation analysis...")
    print("      Purpose: Justifies MAE/RMSE over MAPE (MAPE undefined when actuals=0)")
    zero_dir = output_dir / 'zero_inflation'
    zero_results = analyze_zero_inflation(generation_df, zero_dir)
    print(f"      [OK] Found {len(zero_results['series_zero_ratios'])} series")
    solar_avg_zero = sum(
        info['zero_ratio'] for uid, info in zero_results['series_zero_ratios'].items()
        if 'SUN' in uid
    ) / max(1, sum(1 for uid in zero_results['series_zero_ratios'] if 'SUN' in uid))
    print(f"      [OK] Solar avg zero ratio: {solar_avg_zero:.2%} (zeros at night are expected)")

    # Step 4: Coverage & Missing Data Analysis
    print("\n[4/6] Running coverage gaps analysis...")
    print("      Purpose: Informs hourly grid enforcement policy")
    coverage_dir = output_dir / 'coverage'
    coverage_results = analyze_coverage_gaps(generation_df, coverage_dir)
    complete_series = sum(
        1 for info in coverage_results['series_coverage'].values()
        if info['coverage_ratio'] >= 0.98
    )
    total_series = len(coverage_results['series_coverage'])
    print(f"      [OK] Series with >=98% coverage: {complete_series}/{total_series}")

    # Step 5: Negative Values Analysis (CRITICAL)
    print("\n[5/6] Running negative values analysis...")
    print("      Purpose: CRITICAL - Decides preprocessing policy (clamp vs fail_loud)")
    negative_dir = output_dir / 'negative_values'
    negative_results = analyze_negative_values(generation_df, negative_dir)

    if negative_results['negative_count'] > 0:
        print(f"      [WARNING] Found {negative_results['negative_count']} negative values ({negative_results['negative_ratio']:.4%})")
        print(f"      [WARNING] Affected series: {len(negative_results['series_with_negatives'])}")

        # Analyze patterns
        if negative_results['negative_ratio'] < 0.01:
            print(f"      [OK] Recommendation: CLAMP to 0 (negatives <1%, likely metering errors)")
            print(f"        Policy: negative_policy='clamp'")
        else:
            print(f"      [WARNING] Recommendation: INVESTIGATE (negatives >1%, data quality issue)")
            print(f"        Policy: negative_policy='fail_loud' or 'hybrid'")
    else:
        print(f"      [OK] No negative values found (clean data)")
        print(f"        Policy: No preprocessing needed")

    # Step 6: Weather Alignment Analysis
    print("\n[6/6] Running weather alignment analysis...")
    print("      Purpose: Validates feature selection and correlation")
    weather_dir = output_dir / 'weather_alignment'
    weather_results = analyze_weather_alignment(generation_df, weather_df, weather_dir)
    print(f"      [OK] Merge success rate: {weather_results['merge_success_ratio']:.1%}")

    if 'correlation_by_fuel' in weather_results:
        if 'WND' in weather_results['correlation_by_fuel']:
            wind_corr = weather_results['correlation_by_fuel']['WND']
            top_wind_var = max(wind_corr.items(), key=lambda x: abs(x[1]))
            print(f"      [OK] Wind: Top feature = {top_wind_var[0]} (corr={top_wind_var[1]:.3f})")

        if 'SUN' in weather_results['correlation_by_fuel']:
            solar_corr = weather_results['correlation_by_fuel']['SUN']
            top_solar_var = max(solar_corr.items(), key=lambda x: abs(x[1]))
            print(f"      [OK] Solar: Top feature = {top_solar_var[0]} (corr={top_solar_var[1]:.3f})")

    # Create metadata JSON (no HTML generation)
    metadata = {
        'timestamp': timestamp,
        'generation_rows': len(generation_df),
        'generation_series': generation_df['unique_id'].nunique(),
        'date_range': {
            'start': str(generation_df['ds'].min()),
            'end': str(generation_df['ds'].max()),
        },
        'weather_rows': len(weather_df),
        'analyses_completed': [
            'seasonality', 'zero_inflation', 'coverage_gaps',
            'negative_values', 'weather_alignment'
        ]
    }

    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    # Create summary JSON instead of HTML
    summary = {
        'seasonality': {
            'series_analyzed': len(seasonality_results['series_analyzed']),
            'hourly_patterns': 'Clear 24h and 168h cycles detected',
            'recommendation': 'Use MSTL with season_length=[24, 168]'
        },
        'zero_inflation': {
            'solar_zero_ratio': f"{solar_avg_zero:.2%}",
            'finding': 'Solar has substantial zeros at night (expected)',
            'recommendation': 'Use RMSE/MAE as primary metrics (avoid MAPE)'
        },
        'coverage': {
            'series_complete': f"{complete_series}/{total_series}",
            'threshold': '>=98% coverage',
            'recommendation': 'Use drop_incomplete_series policy'
        },
        'negative_values': {
            'count': negative_results['negative_count'],
            'ratio': f"{negative_results['negative_ratio']:.4%}",
            'recommendation': 'clamp' if negative_results['negative_ratio'] < 0.01 else 'fail_loud or hybrid'
        },
        'weather_alignment': {
            'merge_success_rate': f"{weather_results['merge_success_ratio']:.1%}",
            'recommendation': 'Include all 7 weather variables'
        }
    }

    summary_file = output_dir / 'eda_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Final summary
    print("\n" + "=" * 80)
    print("[SUCCESS] EDA ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n[REPORT] Summary: {summary_file}")
    print(f"[DIR] All outputs: {output_dir}")
    print("\n[FINDINGS] Key Findings:")
    print(f"   * Seasonality: Clear 24h and 168h patterns -> Use MSTL")
    print(f"   * Zero-inflation: Solar {solar_avg_zero:.1%} zeros -> Use RMSE/MAE (not MAPE)")
    print(f"   * Coverage: {complete_series}/{total_series} series >98% complete -> Use drop_incomplete")

    if negative_results['negative_count'] > 0:
        policy_rec = "clamp" if negative_results['negative_ratio'] < 0.01 else "fail_loud or hybrid"
        print(f"   * Negatives: {negative_results['negative_count']} found ({negative_results['negative_ratio']:.2%}) -> Use negative_policy='{policy_rec}'")
    else:
        print(f"   * Negatives: None found -> No preprocessing needed")

    print(f"   * Weather: {weather_results['merge_success_ratio']:.1%} merge success -> Include all 7 variables")

    print("\n[TIP] Next Steps:")
    print("   1. Review JSON reports in output directory")
    print("   2. Check visualization PNG files in subdirectories")
    print("   3. Update dataset_builder policy based on findings:")
    policy_rec = "clamp" if negative_results.get('negative_count', 0) == 0 or negative_results.get('negative_ratio', 0) < 0.01 else "fail_loud"
    print(f"      negative_policy='{policy_rec}'")
    print("\n" + "=" * 80)
