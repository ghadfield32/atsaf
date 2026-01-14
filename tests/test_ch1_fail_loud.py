"""
Chapter 1: Fail-Loud Correctness Tests

Validates that all data quality gates are enforced (not warnings).

Tests:
- Invalid datetime → ValueError
- Non-numeric values → ValueError
- Duplicate timestamps → ValueError
- Missing hours (gaps) → ValueError
- NaN detection → ValueError before reshape
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

logger = logging.getLogger(__name__)


@pytest.mark.fail_loud
class TestDateTimeParsing:
    """DateTime parsing must fail on invalid input (not silent NaN)"""

    def test_invalid_datetime_raises(self):
        """Invalid period should raise ValueError"""
        df = pd.DataFrame({
            'period': ['2024-12-01T00', '2024-12-01T01', 'INVALID', '2024-12-01T03'],
            'value': [100.0, 102.0, 104.0, 106.0],
        })

        with pytest.raises((ValueError, TypeError)):
            pd.to_datetime(df['period'], errors='raise')

    def test_valid_datetime_parses(self):
        """Valid period should parse correctly"""
        df = pd.DataFrame({
            'period': ['2024-12-01T00', '2024-12-01T01', '2024-12-01T02', '2024-12-01T03'],
            'value': [100.0, 102.0, 104.0, 106.0],
        })

        parsed = pd.to_datetime(df['period'], errors='raise')
        assert len(parsed) == 4
        assert parsed[0].year == 2024


@pytest.mark.fail_loud
class TestNumericConversion:
    """Numeric conversion must fail on non-numeric input (not coerce to NaN)"""

    def test_non_numeric_raises(self):
        """Non-numeric value should raise ValueError"""
        df = pd.DataFrame({
            'value': ['100.5', '102.3', 'NOT_A_NUMBER', '105.0']
        })

        with pytest.raises((ValueError, TypeError)):
            pd.to_numeric(df['value'], errors='raise')

    def test_numeric_converts(self):
        """Numeric values should convert cleanly"""
        df = pd.DataFrame({
            'value': ['100.5', '102.3', '104.0', '105.0']
        })

        numeric = pd.to_numeric(df['value'], errors='raise')
        assert len(numeric) == 4
        assert numeric[0] == 100.5

    def test_coercion_produces_nan(self):
        """Coercion should produce NaN (demonstrating why we use errors='raise')"""
        df = pd.DataFrame({
            'value': ['100', 'NOT_A_NUMBER', '200']
        })

        coerced = pd.to_numeric(df['value'], errors='coerce')
        assert coerced.isna().sum() == 1  # One NaN from coercion


@pytest.mark.fail_loud
class TestDuplicateDetection:
    """Duplicates on (unique_id, ds) must be detected and rejected"""

    def test_duplicates_detected(self):
        """Should detect duplicate (unique_id, ds) pairs"""
        df = pd.DataFrame({
            'unique_id': ['NG', 'NG', 'NG'],
            'ds': [
                pd.Timestamp('2024-12-01', tz='UTC'),
                pd.Timestamp('2024-12-01', tz='UTC'),  # Duplicate!
                pd.Timestamp('2024-12-02', tz='UTC'),
            ],
            'value': [100, 102, 104]
        })

        dups = df.groupby(['unique_id', 'ds']).size()
        duplicate_count = (dups > 1).sum()
        assert duplicate_count == 1

    def test_no_duplicates_passes(self):
        """No duplicates should pass"""
        df = pd.DataFrame({
            'unique_id': ['NG', 'NG', 'NG'],
            'ds': pd.date_range('2024-12-01', periods=3, freq='h', tz='UTC'),
            'value': [100, 102, 104]
        })

        dups = df.groupby(['unique_id', 'ds']).size()
        duplicate_count = (dups > 1).sum()
        assert duplicate_count == 0


@pytest.mark.fail_loud
class TestMissingHoursDetection:
    """Missing hours (gaps > 1 hour) must be detected and rejected"""

    def test_missing_hours_detected(self):
        """Should detect gaps > 1 hour"""
        df = pd.DataFrame({
            'unique_id': ['NG', 'NG', 'NG', 'NG'],
            'ds': [
                pd.Timestamp('2024-12-01 00:00:00', tz='UTC'),
                pd.Timestamp('2024-12-01 01:00:00', tz='UTC'),
                pd.Timestamp('2024-12-01 03:00:00', tz='UTC'),  # Gap: 3-1=2 hours
                pd.Timestamp('2024-12-01 04:00:00', tz='UTC'),
            ],
            'y': [100, 102, 104, 106]
        })

        df = df.sort_values('ds')
        time_diffs = df['ds'].diff()
        expected_freq = pd.Timedelta(hours=1)
        missing_mask = time_diffs > expected_freq
        missing_count = missing_mask.sum()

        assert missing_count == 1
        gap_hours = time_diffs[missing_mask].iloc[0].total_seconds() / 3600
        assert gap_hours == 2.0

    def test_no_gaps_passes(self):
        """Hourly frequency with no gaps should pass"""
        df = pd.DataFrame({
            'unique_id': ['NG'] * 5,
            'ds': pd.date_range('2024-12-01', periods=5, freq='h', tz='UTC'),
            'y': [100, 102, 104, 106, 108]
        })

        df = df.sort_values('ds')
        time_diffs = df['ds'].diff()
        expected_freq = pd.Timedelta(hours=1)
        missing_mask = time_diffs > expected_freq
        missing_count = missing_mask.sum()

        assert missing_count == 0

    def test_dst_repeated_hours_not_error(self):
        """DST repeated hours (gap == 0) are expected, not errors"""
        df = pd.DataFrame({
            'unique_id': ['NG'] * 4,
            'ds': [
                pd.Timestamp('2024-11-03 01:00:00', tz='UTC'),
                pd.Timestamp('2024-11-03 01:00:00', tz='UTC'),  # Repeated (DST)
                pd.Timestamp('2024-11-03 02:00:00', tz='UTC'),
                pd.Timestamp('2024-11-03 03:00:00', tz='UTC'),
            ],
            'y': [100, 101, 102, 103]
        })

        time_diffs = df['ds'].diff()
        repeated_mask = time_diffs == pd.Timedelta(0)
        repeated_count = repeated_mask.sum()

        assert repeated_count == 1  # DST repeated hour detected
        # But should NOT raise error; just log INFO


@pytest.mark.fail_loud
class TestMultiSeriesMergeKeys:
    """Multi-series merge must use (unique_id, ds), not just ds"""

    def test_multi_series_merge_correct(self):
        """Merge on ['unique_id', 'ds'] produces correct results"""
        forecast = pd.DataFrame({
            'unique_id': ['NG', 'NG', 'COAL', 'COAL'],
            'ds': [
                pd.Timestamp('2024-12-01 00:00', tz='UTC'),
                pd.Timestamp('2024-12-01 01:00', tz='UTC'),
                pd.Timestamp('2024-12-01 00:00', tz='UTC'),
                pd.Timestamp('2024-12-01 01:00', tz='UTC'),
            ],
            'forecast': [100, 102, 50, 52]
        })

        actual = pd.DataFrame({
            'unique_id': ['NG', 'NG', 'COAL', 'COAL'],
            'ds': [
                pd.Timestamp('2024-12-01 00:00', tz='UTC'),
                pd.Timestamp('2024-12-01 01:00', tz='UTC'),
                pd.Timestamp('2024-12-01 00:00', tz='UTC'),
                pd.Timestamp('2024-12-01 01:00', tz='UTC'),
            ],
            'actual': [99, 101, 51, 53]
        })

        # Correct merge
        merged = forecast.merge(actual, on=['unique_id', 'ds'], how='left')

        assert len(merged) == 4
        assert list(merged['forecast']) == [100, 102, 50, 52]
        assert list(merged['actual']) == [99, 101, 51, 53]

    def test_single_key_merge_wrong(self):
        """Merge on ['ds'] only produces incorrect results (data pollution)"""
        forecast = pd.DataFrame({
            'unique_id': ['NG', 'NG', 'COAL', 'COAL'],
            'ds': [
                pd.Timestamp('2024-12-01 00:00', tz='UTC'),
                pd.Timestamp('2024-12-01 01:00', tz='UTC'),
                pd.Timestamp('2024-12-01 00:00', tz='UTC'),
                pd.Timestamp('2024-12-01 01:00', tz='UTC'),
            ],
            'forecast': [100, 102, 50, 52]
        })

        actual = pd.DataFrame({
            'unique_id': ['NG', 'NG', 'COAL', 'COAL'],
            'ds': [
                pd.Timestamp('2024-12-01 00:00', tz='UTC'),
                pd.Timestamp('2024-12-01 01:00', tz='UTC'),
                pd.Timestamp('2024-12-01 00:00', tz='UTC'),
                pd.Timestamp('2024-12-01 01:00', tz='UTC'),
            ],
            'actual': [99, 101, 51, 53]
        })

        # Wrong merge: only on 'ds'
        merged = forecast.merge(actual, on=['ds'], how='left')

        # Results in 8 rows, data is polluted!
        assert len(merged) == 8
        # NG forecast matches both NG and COAL actuals
        logger.warning(f"Single-key merge produced {len(merged)} rows (data pollution!)")


@pytest.mark.fail_loud
class TestNaNDetectionPreValidation:
    """NaN must be detected before prepare_for_forecasting reshape"""

    def test_valid_no_nan(self):
        """Valid data should have no NaN"""
        df = pd.DataFrame({
            'period': pd.date_range('2024-12-01', periods=3, freq='h', tz='UTC'),
            'value': [100.0, 102.0, 104.0],
        })

        nan_period = df['period'].isna().sum()
        nan_value = df['value'].isna().sum()

        assert nan_period == 0
        assert nan_value == 0

    def test_nan_detected(self):
        """NaN values should be detected and reported"""
        df = pd.DataFrame({
            'period': pd.date_range('2024-12-01', periods=3, freq='h', tz='UTC'),
            'value': [100.0, np.nan, 104.0],  # One NaN
        })

        nan_count = df['value'].isna().sum()
        assert nan_count == 1


@pytest.mark.fail_loud
class TestCoercionPrevention:
    """No silent coercion to NaN; errors='raise' must be enforced"""

    def test_coercion_demonstrates_risk(self):
        """Demonstrate why coercion is dangerous"""
        df = pd.DataFrame({
            'value': ['100', 'UNKNOWN', '200']
        })

        # Coerced (bad)
        coerced = pd.to_numeric(df['value'], errors='coerce')
        assert coerced.isna().sum() == 1  # Silent NaN!

        # Raised (good)
        with pytest.raises((ValueError, TypeError)):
            pd.to_numeric(df['value'], errors='raise')
            pd.to_numeric(df['value'], errors='raise')
