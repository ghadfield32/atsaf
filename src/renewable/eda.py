# file: src/renewable/eda.py
"""
Enhanced Exploratory Data Analysis for Renewable Energy Forecasting

This module provides decision-driven EDA with emphasis on:
1. Understanding WHY negative values exist (not just detecting them)
2. Providing actionable recommendations based on findings
3. Validating physical constraints for renewable energy data

Key Principle: Renewable energy generation CANNOT be negative.
- Solar panels produce 0-X power, never negative
- Wind turbines produce 0-X power, never negative
- Negative values in data are ALWAYS data quality issues that need investigation
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100


class NegativeValueInvestigation:
    """
    Deep investigation into why negative values exist in renewable generation data.

    EIA Data Context:
    - EIA reports "net generation" which is gross generation minus station use
    - Station auxiliary loads (cooling, controls, etc.) can exceed generation during:
      * Low-wind periods for wind farms
      * Night/cloudy periods for solar (if inverters consume standby power)
      * Startup/shutdown events

    This is VALID data but represents net metering, not physical impossibility.
    However, for FORECASTING purposes, we typically want to predict gross generation
    or at minimum clamp to zero since negative "production" isn't meaningful for
    grid planning.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with columns [unique_id, ds, y] where y is generation
        """
        self.df = df.copy()
        self.df['ds'] = pd.to_datetime(self.df['ds'])
        self.df['hour'] = self.df['ds'].dt.hour
        self.df['dow'] = self.df['ds'].dt.dayofweek
        self.df['date'] = self.df['ds'].dt.date

    def get_negative_summary(self) -> Dict[str, Any]:
        """Get high-level summary of negative values."""
        neg_mask = self.df['y'] < 0

        summary = {
            'total_rows': len(self.df),
            'negative_count': int(neg_mask.sum()),
            'negative_ratio': float(neg_mask.sum() / len(self.df)) if len(self.df) > 0 else 0,
            'affected_series': self.df.loc[neg_mask, 'unique_id'].unique().tolist(),
            'min_value': float(self.df['y'].min()),
            'max_negative': float(self.df.loc[neg_mask, 'y'].max()) if neg_mask.any() else None,
        }
        return summary

    def analyze_negative_patterns(self) -> Dict[str, Any]:
        """
        Investigate WHEN and WHERE negatives occur to understand root cause.

        Key questions:
        1. Are negatives concentrated in specific hours? (auxiliary load pattern)
        2. Are negatives concentrated in specific series? (regional data issue)
        3. What's the magnitude? (small negatives = metering noise, large = real issue)
        """
        neg_df = self.df[self.df['y'] < 0].copy()

        if len(neg_df) == 0:
            return {'status': 'no_negatives_found'}

        analysis = {
            'by_series': {},
            'by_hour': {},
            'by_dow': {},
            'magnitude_analysis': {},
            'temporal_clustering': {},
        }

        # 1. Analyze by series
        for uid in neg_df['unique_id'].unique():
            series_neg = neg_df[neg_df['unique_id'] == uid]
            series_total = self.df[self.df['unique_id'] == uid]

            fuel_type = uid.split('_')[1] if '_' in uid else 'UNKNOWN'

            analysis['by_series'][uid] = {
                'count': int(len(series_neg)),
                'ratio': float(len(series_neg) / len(series_total)),
                'fuel_type': fuel_type,
                'min_value': float(series_neg['y'].min()),
                'max_value': float(series_neg['y'].max()),
                'mean_value': float(series_neg['y'].mean()),
                'std_value': float(series_neg['y'].std()),
            }

        # 2. Analyze by hour (Are negatives at night for solar? Low-wind hours for wind?)
        hour_counts = neg_df.groupby('hour').size()
        total_by_hour = self.df.groupby('hour').size()
        neg_ratio_by_hour = (hour_counts / total_by_hour).fillna(0)

        analysis['by_hour'] = {
            'counts': hour_counts.to_dict(),
            'ratios': neg_ratio_by_hour.to_dict(),
            'peak_negative_hour': int(neg_ratio_by_hour.idxmax()) if len(neg_ratio_by_hour) > 0 else None,
        }

        # 3. Magnitude analysis - categorize severity
        neg_values = neg_df['y'].values
        analysis['magnitude_analysis'] = {
            'tiny_negatives_count': int((neg_values > -10).sum()),  # Likely metering noise
            'small_negatives_count': int(((neg_values <= -10) & (neg_values > -100)).sum()),
            'medium_negatives_count': int(((neg_values <= -100) & (neg_values > -1000)).sum()),
            'large_negatives_count': int((neg_values <= -1000).sum()),  # Significant issue
            'percentiles': {
                'p5': float(np.percentile(neg_values, 5)),
                'p25': float(np.percentile(neg_values, 25)),
                'p50': float(np.percentile(neg_values, 50)),
                'p75': float(np.percentile(neg_values, 75)),
                'p95': float(np.percentile(neg_values, 95)),
            }
        }

        # 4. Check for temporal clustering (consecutive hours of negatives)
        for uid in neg_df['unique_id'].unique():
            series_df = self.df[self.df['unique_id'] == uid].sort_values('ds')
            series_df['is_negative'] = series_df['y'] < 0

            # Find consecutive negative runs
            series_df['neg_block'] = (series_df['is_negative'] != series_df['is_negative'].shift()).cumsum()
            neg_blocks = series_df[series_df['is_negative']].groupby('neg_block').agg(
                start=('ds', 'min'),
                end=('ds', 'max'),
                duration_hours=('ds', 'count'),
                min_value=('y', 'min'),
            ).reset_index(drop=True)

            if len(neg_blocks) > 0:
                analysis['temporal_clustering'][uid] = {
                    'num_blocks': len(neg_blocks),
                    'avg_block_duration_hours': float(neg_blocks['duration_hours'].mean()),
                    'max_block_duration_hours': int(neg_blocks['duration_hours'].max()),
                    'longest_block_start': str(neg_blocks.loc[neg_blocks['duration_hours'].idxmax(), 'start']),
                }

        return analysis

    def determine_root_cause(self) -> Dict[str, Any]:
        """
        Based on patterns, determine the likely root cause and recommend action.

        Possible causes:
        1. NET GENERATION DATA: EIA reports net = gross - auxiliary. This is valid.
        2. METERING NOISE: Tiny negatives (-1 to -10 MWh) are measurement error.
        3. DATA REPORTING ERROR: Large sporadic negatives are likely errors.
        4. SYSTEMATIC ISSUE: Negatives always at same time = station use pattern.
        """
        patterns = self.analyze_negative_patterns()

        if patterns.get('status') == 'no_negatives_found':
            return {
                'root_cause': 'NONE',
                'confidence': 'HIGH',
                'recommendation': 'No action needed - data is clean',
                'preprocessing_policy': 'pass_through',
            }

        magnitude = patterns.get('magnitude_analysis', {})
        total_neg = sum([
            magnitude.get('tiny_negatives_count', 0),
            magnitude.get('small_negatives_count', 0),
            magnitude.get('medium_negatives_count', 0),
            magnitude.get('large_negatives_count', 0),
        ])

        # Determine root cause based on patterns
        root_cause_analysis = {
            'factors': [],
            'root_cause': 'UNKNOWN',
            'confidence': 'LOW',
            'recommendation': '',
            'preprocessing_policy': 'clamp_to_zero',
        }

        # Check if mostly tiny negatives (metering noise)
        if magnitude.get('tiny_negatives_count', 0) / max(total_neg, 1) > 0.9:
            root_cause_analysis['factors'].append('90%+ negatives are tiny (<10 MWh)')
            root_cause_analysis['root_cause'] = 'METERING_NOISE'
            root_cause_analysis['confidence'] = 'HIGH'
            root_cause_analysis['recommendation'] = (
                'Tiny negatives are measurement noise. Safe to clamp to 0.'
            )
            root_cause_analysis['preprocessing_policy'] = 'clamp_to_zero'

        # Check if negatives are systematic (same hours)
        elif patterns.get('by_hour', {}).get('peak_negative_hour') is not None:
            hour_ratios = patterns.get('by_hour', {}).get('ratios', {})
            max_ratio = max(hour_ratios.values()) if hour_ratios else 0

            if max_ratio > 0.3:  # >30% of negatives in one hour
                root_cause_analysis['factors'].append(f'Negatives concentrated at specific hours')
                root_cause_analysis['root_cause'] = 'NET_GENERATION_AUXILIARY_LOAD'
                root_cause_analysis['confidence'] = 'MEDIUM'
                root_cause_analysis['recommendation'] = (
                    'Negatives likely represent station auxiliary loads exceeding generation. '
                    'This is valid net generation data. For forecasting, clamp to 0 since '
                    'we want to predict usable power output.'
                )
                root_cause_analysis['preprocessing_policy'] = 'clamp_to_zero'

        # Check for large sporadic negatives (data errors)
        if magnitude.get('large_negatives_count', 0) > 0:
            root_cause_analysis['factors'].append(f"{magnitude.get('large_negatives_count')} large negatives (<-1000 MWh)")

            # If ONLY large negatives and they're sporadic, likely errors
            if magnitude.get('tiny_negatives_count', 0) == 0 and magnitude.get('small_negatives_count', 0) == 0:
                root_cause_analysis['root_cause'] = 'DATA_REPORTING_ERROR'
                root_cause_analysis['confidence'] = 'MEDIUM'
                root_cause_analysis['recommendation'] = (
                    'Large sporadic negatives are likely data reporting errors. '
                    'Recommend clamping to 0 or investigating with EIA.'
                )

        return root_cause_analysis

    def generate_report(self, output_dir: Path) -> Dict[str, Any]:
        """Generate comprehensive negative value investigation report."""
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = self.get_negative_summary()
        patterns = self.analyze_negative_patterns()
        root_cause = self.determine_root_cause()

        report = {
            'summary': summary,
            'patterns': patterns,
            'root_cause_analysis': root_cause,
            'generated_at': datetime.now().isoformat(),
        }

        # Save JSON report
        report_file = output_dir / 'negative_investigation.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate visualizations if negatives exist
        if summary['negative_count'] > 0:
            self._plot_negative_analysis(patterns, output_dir)

        return report

    def _plot_negative_analysis(self, patterns: Dict, output_dir: Path) -> None:
        """Create diagnostic plots for negative value analysis."""
        neg_df = self.df[self.df['y'] < 0]

        if len(neg_df) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Distribution of negative values
        ax = axes[0, 0]
        neg_values = neg_df['y'].values
        ax.hist(neg_values, bins=50, color='red', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.median(neg_values), color='blue', linestyle='--',
                   label=f'Median: {np.median(neg_values):.1f}')
        ax.set_xlabel('Generation (MWh)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Negative Values')
        ax.legend()

        # 2. Negative ratio by hour
        ax = axes[0, 1]
        hour_ratios = patterns.get('by_hour', {}).get('ratios', {})
        if hour_ratios:
            hours = sorted(hour_ratios.keys())
            ratios = [hour_ratios[h] for h in hours]
            ax.bar(hours, ratios, color='orange', alpha=0.7)
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Negative Ratio')
            ax.set_title('When Do Negatives Occur? (by Hour)')
            ax.set_xticks(range(0, 24, 2))

        # 3. Negative values by series
        ax = axes[1, 0]
        series_data = patterns.get('by_series', {})
        if series_data:
            series_names = list(series_data.keys())
            series_counts = [series_data[s]['count'] for s in series_names]
            colors = ['red' if 'SUN' in s else 'blue' for s in series_names]
            ax.barh(series_names, series_counts, color=colors, alpha=0.7)
            ax.set_xlabel('Negative Count')
            ax.set_title('Negatives by Series (Blue=Wind, Red=Solar)')

        # 4. Time series with negatives highlighted
        ax = axes[1, 1]
        # Plot first affected series
        affected = patterns.get('by_series', {})
        if affected:
            first_series = list(affected.keys())[0]
            series_df = self.df[self.df['unique_id'] == first_series].sort_values('ds')
            ax.plot(series_df['ds'], series_df['y'], 'b-', alpha=0.5, label='Generation')
            neg_mask = series_df['y'] < 0
            ax.scatter(series_df.loc[neg_mask, 'ds'], series_df.loc[neg_mask, 'y'],
                      c='red', s=20, label='Negative values', zorder=5)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax.set_xlabel('Date')
            ax.set_ylabel('Generation (MWh)')
            ax.set_title(f'Time Series: {first_series}')
            ax.legend()
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(output_dir / 'negative_investigation.png', dpi=150, bbox_inches='tight')
        plt.close()


def run_full_eda(
    generation_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run comprehensive EDA with emphasis on understanding data quality issues.

    This function produces actionable insights, not just statistics.

    Args:
        generation_df: DataFrame with columns [unique_id, ds, y]
        weather_df: DataFrame with columns [ds, region, weather_vars...]
        output_dir: Directory to save all outputs

    Returns:
        Dictionary with all EDA results and recommendations
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RENEWABLE ENERGY EDA - Comprehensive Analysis")
    print("=" * 80)

    results = {
        'timestamp': timestamp,
        'output_dir': str(report_dir),
        'data_summary': {},
        'negative_investigation': {},
        'seasonality': {},
        'zero_inflation': {},
        'weather_alignment': {},
        'recommendations': {},
    }

    # 1. Data Summary
    print("\n[1/5] Data Summary...")
    results['data_summary'] = {
        'generation_rows': len(generation_df),
        'generation_series': generation_df['unique_id'].nunique(),
        'series_list': generation_df['unique_id'].unique().tolist(),
        'date_range': {
            'start': str(generation_df['ds'].min()),
            'end': str(generation_df['ds'].max()),
        },
        'weather_rows': len(weather_df),
        'weather_regions': weather_df['region'].nunique() if 'region' in weather_df.columns else 0,
    }
    print(f"   Generation: {results['data_summary']['generation_rows']:,} rows, "
          f"{results['data_summary']['generation_series']} series")

    # 2. CRITICAL: Negative Value Investigation
    print("\n[2/5] Negative Value Investigation (CRITICAL)...")
    neg_investigator = NegativeValueInvestigation(generation_df)
    results['negative_investigation'] = neg_investigator.generate_report(
        report_dir / 'negative_values'
    )

    neg_summary = results['negative_investigation']['summary']
    root_cause = results['negative_investigation']['root_cause_analysis']

    if neg_summary['negative_count'] > 0:
        print(f"   [WARNING] Found {neg_summary['negative_count']} negative values "
              f"({neg_summary['negative_ratio']:.2%})")
        print(f"   [WARNING] Affected series: {neg_summary['affected_series']}")
        print(f"   [ANALYSIS] Root cause: {root_cause['root_cause']} "
              f"(confidence: {root_cause['confidence']})")
        print(f"   [RECOMMENDATION] {root_cause['recommendation']}")
    else:
        print("   [OK] No negative values found")

    # 3. Seasonality Analysis
    print("\n[3/5] Seasonality Analysis...")
    seasonality_dir = report_dir / 'seasonality'
    seasonality_dir.mkdir(parents=True, exist_ok=True)

    seasonality_results = _analyze_seasonality(generation_df, seasonality_dir)
    results['seasonality'] = seasonality_results
    print(f"   [OK] Analyzed {len(seasonality_results.get('series_analyzed', []))} series")

    # 4. Zero-Inflation Analysis
    print("\n[4/5] Zero-Inflation Analysis...")
    zero_dir = report_dir / 'zero_inflation'
    zero_dir.mkdir(parents=True, exist_ok=True)

    zero_results = _analyze_zero_inflation(generation_df, zero_dir)
    results['zero_inflation'] = zero_results

    solar_series = [uid for uid in generation_df['unique_id'].unique() if 'SUN' in uid]
    if solar_series:
        avg_zero = sum(
            zero_results['series_zero_ratios'].get(uid, {}).get('zero_ratio', 0)
            for uid in solar_series
        ) / len(solar_series)
        print(f"   [OK] Solar zero ratio: {avg_zero:.1%} (zeros at night expected)")

    # 5. Weather Alignment
    print("\n[5/5] Weather Alignment...")
    weather_dir = report_dir / 'weather_alignment'
    weather_dir.mkdir(parents=True, exist_ok=True)

    weather_results = _analyze_weather_alignment(generation_df, weather_df, weather_dir)
    results['weather_alignment'] = weather_results
    print(f"   [OK] Merge success rate: {weather_results['merge_success_ratio']:.1%}")

    # Generate Final Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    recommendations = {
        'preprocessing': {},
        'modeling': {},
        'evaluation': {},
    }

    # Preprocessing recommendations based on negative analysis
    if neg_summary['negative_count'] > 0:
        policy = root_cause.get('preprocessing_policy', 'clamp_to_zero')
        recommendations['preprocessing']['negative_handling'] = {
            'policy': policy,
            'reason': root_cause['recommendation'],
            'affected_series': neg_summary['affected_series'],
        }
        print(f"\n[PREPROCESSING] Negative Handling: {policy}")
        print(f"   Reason: {root_cause['recommendation']}")
    else:
        recommendations['preprocessing']['negative_handling'] = {
            'policy': 'none_needed',
            'reason': 'No negative values in raw data',
        }

    # Modeling recommendations
    recommendations['modeling'] = {
        'seasonality': 'Use MSTL with season_length=[24, 168] (daily + weekly)',
        'forecast_constraints': 'ALWAYS clip forecasts and intervals to [0, ∞)',
        'reason': 'Physical constraint: renewable generation cannot be negative',
    }
    print(f"\n[MODELING] Forecast Constraints: Clip to [0, ∞)")
    print(f"   Reason: Physical constraint - renewable generation cannot be negative")

    # Evaluation recommendations
    recommendations['evaluation'] = {
        'metrics': ['RMSE', 'MAE'],
        'avoid': 'MAPE (undefined when y=0)',
        'reason': f"Solar has {avg_zero:.1%} zeros (nighttime)" if solar_series else "Standard metrics",
    }
    print(f"\n[EVALUATION] Use RMSE/MAE, avoid MAPE")

    results['recommendations'] = recommendations

    # Save full report
    report_file = report_dir / 'eda_report.json'
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] EDA complete. Report saved to: {report_dir}")
    print("=" * 80)

    return results


def _analyze_seasonality(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """Analyze seasonal patterns in generation data."""
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])

    results = {
        'series_analyzed': [],
        'hourly_patterns': {},
    }

    for uid in df['unique_id'].unique()[:3]:  # Analyze first 3 series
        series_data = df[df['unique_id'] == uid].copy()
        series_data['hour'] = series_data['ds'].dt.hour

        hourly_profile = series_data.groupby('hour')['y'].agg(['mean', 'std']).reset_index()
        results['series_analyzed'].append(uid)
        results['hourly_patterns'][uid] = hourly_profile.to_dict(orient='records')

    # Create visualization
    fig, axes = plt.subplots(1, len(results['series_analyzed']),
                            figsize=(5 * len(results['series_analyzed']), 4))
    if len(results['series_analyzed']) == 1:
        axes = [axes]

    for idx, uid in enumerate(results['series_analyzed']):
        series_data = df[df['unique_id'] == uid].copy()
        series_data['hour'] = series_data['ds'].dt.hour
        hourly_mean = series_data.groupby('hour')['y'].mean()
        hourly_std = series_data.groupby('hour')['y'].std()

        axes[idx].plot(hourly_mean.index, hourly_mean.values, marker='o')
        axes[idx].fill_between(hourly_mean.index,
                               hourly_mean - hourly_std,
                               hourly_mean + hourly_std, alpha=0.3)
        axes[idx].set_xlabel('Hour of Day')
        axes[idx].set_ylabel('Generation (MWh)')
        axes[idx].set_title(f'{uid} - Hourly Profile')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'hourly_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()

    return results


def _analyze_zero_inflation(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """Analyze zero values in generation data."""
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df['hour'] = df['ds'].dt.hour

    results = {
        'series_zero_ratios': {},
    }

    for uid in df['unique_id'].unique():
        series_data = df[df['unique_id'] == uid]
        zero_count = (series_data['y'] == 0).sum()
        total_count = len(series_data)

        results['series_zero_ratios'][uid] = {
            'zero_count': int(zero_count),
            'total_count': int(total_count),
            'zero_ratio': float(zero_count / total_count) if total_count > 0 else 0,
        }

    # Visualization
    solar_series = [uid for uid in df['unique_id'].unique() if 'SUN' in uid]
    wind_series = [uid for uid in df['unique_id'].unique() if 'WND' in uid]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Solar zeros by hour
    if solar_series:
        solar_df = df[df['unique_id'].isin(solar_series)]
        solar_zero_by_hour = solar_df.groupby('hour').apply(
            lambda x: (x['y'] == 0).mean()
        )
        axes[0].bar(solar_zero_by_hour.index, solar_zero_by_hour.values,
                   color='orange', alpha=0.7)
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('Zero Ratio')
        axes[0].set_title('Solar: Zero Ratio by Hour (Night = Expected)')
        axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    # Wind zeros by hour
    if wind_series:
        wind_df = df[df['unique_id'].isin(wind_series)]
        wind_zero_by_hour = wind_df.groupby('hour').apply(
            lambda x: (x['y'] == 0).mean()
        )
        axes[1].bar(wind_zero_by_hour.index, wind_zero_by_hour.values,
                   color='blue', alpha=0.7)
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Zero Ratio')
        axes[1].set_title('Wind: Zero Ratio by Hour')

    plt.tight_layout()
    plt.savefig(output_dir / 'zero_inflation.png', dpi=150, bbox_inches='tight')
    plt.close()

    return results


def _analyze_weather_alignment(
    generation_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    output_dir: Path
) -> Dict[str, Any]:
    """Analyze weather-generation correlation."""
    generation_df = generation_df.copy()
    weather_df = weather_df.copy()

    generation_df['ds'] = pd.to_datetime(generation_df['ds'])
    weather_df['ds'] = pd.to_datetime(weather_df['ds'])
    generation_df['region'] = generation_df['unique_id'].str.split('_').str[0]

    merged = generation_df.merge(weather_df, on=['ds', 'region'], how='left')

    weather_vars = [c for c in weather_df.columns
                   if c not in ['ds', 'region'] and c in merged.columns]

    results = {
        'merge_success_ratio': float(
            merged[weather_vars[0]].notna().mean() if weather_vars else 0
        ),
        'correlation_by_fuel': {},
    }

    # Calculate correlations
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

    # Save results
    with open(output_dir / 'weather_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    """Run EDA on renewable energy data."""
    import sys

    generation_path = Path("data/renewable/generation.parquet")
    weather_path = Path("data/renewable/weather.parquet")

    if not generation_path.exists() or not weather_path.exists():
        print("Data files not found. Run pipeline first.")
        sys.exit(1)

    generation_df = pd.read_parquet(generation_path)
    weather_df = pd.read_parquet(weather_path)

    output_dir = Path("reports/renewable/eda")

    results = run_full_eda(generation_df, weather_df, output_dir)
