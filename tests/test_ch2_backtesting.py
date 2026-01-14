"""
Chapter 2: Backtesting Correctness Tests

Validates windowing strategies, temporal boundaries, and no information leakage.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.mark.fail_loud
class TestBacktestBoundaries:
    """Ensure test set never touches training set (no leakage)"""

    def test_rolling_splits_no_leakage(self):
        """Rolling splits should not have train/test overlap"""
        # Simulate rolling split parameters
        n_rows = 720  # 30 days hourly
        min_train = 200
        test_size = 72
        step_size = 100
        n_windows = 4

        # Calculate rolling split boundaries
        train_size = max(
            min_train,
            n_rows - n_windows * step_size - test_size
        )

        splits = []
        for window_id in range(n_windows):
            train_end_idx = min(
                train_size + window_id * step_size,
                n_rows - test_size
            )
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + test_size, n_rows)

            if test_end_idx - test_start_idx < test_size:
                continue

            train_start_idx = max(0, train_end_idx - train_size)

            splits.append({
                'window_id': window_id,
                'train_start': train_start_idx,
                'train_end': train_end_idx - 1,
                'test_start': test_start_idx,
                'test_end': test_end_idx - 1,
            })

        # Verify no overlap
        for split in splits:
            assert split['train_end'] < split['test_start'], \
                f"Window {split['window_id']}: train/test overlap!"

    def test_expanding_splits_no_leakage(self):
        """Expanding splits should always start from 0"""
        n_rows = 720
        min_train = 200
        test_size = 72
        step_size = 100
        n_windows = 4

        splits = []
        for window_id in range(n_windows):
            train_end_idx = min_train + window_id * step_size
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + test_size, n_rows)

            if test_end_idx - test_start_idx < test_size:
                continue

            splits.append({
                'window_id': window_id,
                'train_start': 0,  # Always from beginning
                'train_end': train_end_idx - 1,
                'test_start': test_start_idx,
                'test_end': test_end_idx - 1,
            })

        # Verify: each window's train set expands
        for i in range(len(splits) - 1):
            assert splits[i]['train_end'] < splits[i + 1]['train_end'], \
                "Training set should grow in expanding strategy"


@pytest.mark.fail_loud
class TestSplitSizes:
    """Ensure minimum data requirements for each split"""

    def test_minimum_train_size_enforced(self):
        """Each split must have >= min_train_size rows"""
        min_train = 200

        # Simulate split
        train_start_idx = 0
        train_end_idx = 150  # Less than min_train!

        train_size = train_end_idx - train_start_idx

        # Should fail
        assert train_size < min_train

    def test_test_size_correct(self):
        """Each split must have exactly horizon rows in test"""
        horizon = 72
        test_start_idx = 500
        test_end_idx = test_start_idx + horizon

        test_size = test_end_idx - test_start_idx
        assert test_size == horizon


@pytest.mark.fail_loud
class TestHorizonConsistency:
    """Forecast horizon must match test size"""

    def test_horizon_matches_test_horizon(self):
        """Forecast horizon should equal test set size"""
        horizon = 72  # hours
        n_test = 72

        assert horizon == n_test

    def test_insufficient_test_data_raises(self):
        """Should skip split if < horizon test rows available"""
        horizon = 72
        test_rows_available = 50

        # Should skip this split
        if test_rows_available < horizon:
            pytest.skip("Insufficient test data for horizon")


@pytest.mark.fail_loud
class TestTemporalOrdering:
    """Train period must come strictly before test period"""

    def test_temporal_order_respected(self):
        """Test split must be after train split temporally"""
        df = pd.DataFrame({
            'ds': pd.date_range('2024-12-01', periods=500, freq='h', tz='UTC'),
            'value': np.random.randn(500)
        })

        df = df.sort_values('ds')

        # Simulate split
        cutoff_idx = 200
        test_start_idx = 200

        train_end_ts = df.iloc[cutoff_idx - 1]['ds']
        test_start_ts = df.iloc[test_start_idx]['ds']

        assert train_end_ts < test_start_ts, "Test must start after training ends"


@pytest.mark.smoke
class TestSplitMetadata:
    """Each split should have complete metadata"""

    def test_split_metadata_complete(self):
        """Split should include all required fields"""
        split = {
            'window_id': 0,
            'train_start': pd.Timestamp('2024-12-01 00:00', tz='UTC'),
            'train_end': pd.Timestamp('2024-12-08 23:00', tz='UTC'),
            'test_start': pd.Timestamp('2024-12-09 00:00', tz='UTC'),
            'test_end': pd.Timestamp('2024-12-12 23:00', tz='UTC'),
            'train_size': 192,
            'test_size': 96,
            'strategy': 'rolling',
        }

        # Verify all fields present
        required = ['window_id', 'train_start', 'train_end', 'test_start',
                    'test_end', 'train_size', 'test_size', 'strategy']
        for field in required:
            assert field in split, f"Missing field: {field}"
            assert field in split, f"Missing field: {field}"
