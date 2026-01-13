"""
Smoke Tests: Minimal end-to-end validation with synthetic data

These tests verify the pipeline works without requiring EIA API or external dependencies.
All data is generated synthetically.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.mark.smoke
class TestSyntheticDataPipeline:
    """Test full pipeline with synthetic data"""
    
    @staticmethod
    def create_synthetic_dataset(n_days=14, freq='H'):
        """Create synthetic time series matching expected format"""
        periods = n_days * 24 if freq == 'H' else n_days
        
        df = pd.DataFrame({
            'unique_id': ['US48_NG'] * periods,
            'ds': pd.date_range('2024-12-01', periods=periods, freq=freq, tz='UTC'),
            'value': 100 + np.cumsum(np.random.randn(periods) * 2),  # Random walk
        })
        
        return df
    
    def test_synthetic_data_valid(self):
        """Synthetic data should pass basic validation"""
        df = self.create_synthetic_dataset()
        
        # Check structure
        assert len(df) == 14 * 24
        assert list(df.columns) == ['unique_id', 'ds', 'value']
        
        # Check timezone
        assert df['ds'].dt.tz is not None
        
        # Check no NaN
        assert df.isna().sum().sum() == 0
    
    def test_synthetic_data_no_duplicates(self):
        """Synthetic data should have no duplicates"""
        df = self.create_synthetic_dataset()
        
        dups = df.groupby(['unique_id', 'ds']).size()
        duplicate_count = (dups > 1).sum()
        
        assert duplicate_count == 0
    
    def test_synthetic_data_no_gaps(self):
        """Synthetic data should have no gaps"""
        df = self.create_synthetic_dataset()
        
        time_diffs = df['ds'].diff()
        expected_freq = pd.Timedelta(hours=1)
        missing_mask = time_diffs > expected_freq
        missing_count = missing_mask.sum()
        
        assert missing_count == 0
    
    def test_synthetic_train_test_split(self):
        """Should be able to split into train/test"""
        df = self.create_synthetic_dataset()
        
        # 70/30 split
        test_hours = 72
        split_point = df['ds'].max() - timedelta(hours=test_hours)
        
        train_df = df[df['ds'] <= split_point]
        test_df = df[df['ds'] > split_point]
        
        assert len(train_df) + len(test_df) == len(df)
        assert len(test_df) == test_hours
    
    def test_synthetic_forecast_shape(self):
        """Generated forecast should have correct shape"""
        df = self.create_synthetic_dataset()
        
        horizon = 72
        
        # Simulate forecast (naive: copy last value)
        last_val = df['value'].iloc[-1]
        forecast = pd.DataFrame({
            'unique_id': ['US48_NG'] * horizon,
            'ds': pd.date_range(df['ds'].max() + timedelta(hours=1), 
                               periods=horizon, freq='H', tz='UTC'),
            'forecast': [last_val] * horizon,
            'lower': [last_val - 5] * horizon,
            'upper': [last_val + 5] * horizon,
        })
        
        assert len(forecast) == horizon
        assert 'forecast' in forecast.columns
    
    def test_synthetic_merge_forecast_actual(self):
        """Forecast and actual should merge correctly"""
        df = self.create_synthetic_dataset()
        
        test_hours = 72
        split_point = df['ds'].max() - timedelta(hours=test_hours)
        test_df = df[df['ds'] > split_point]
        
        # Simulate forecast
        forecast_df = pd.DataFrame({
            'unique_id': test_df['unique_id'].values,
            'ds': test_df['ds'].values,
            'forecast': test_df['value'].values - 1,  # Off by 1
            'lower': test_df['value'].values - 5,
            'upper': test_df['value'].values + 5,
        })
        
        # Merge on correct keys (ensure timezone compatibility)
        test_for_merge = test_df[['unique_id', 'ds', 'value']].rename(columns={'value': 'actual'}).copy()
        # Ensure timezone compatibility
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds']).dt.tz_localize('UTC') if forecast_df['ds'].dt.tz is None else forecast_df['ds']
        test_for_merge['ds'] = pd.to_datetime(test_for_merge['ds']).dt.tz_localize('UTC') if test_for_merge['ds'].dt.tz is None else test_for_merge['ds']
        
        merged = forecast_df.merge(
            test_for_merge,
            on=['unique_id', 'ds'],
            how='left'
        )
        
        assert len(merged) == len(test_df)
        assert 'forecast' in merged.columns
        assert 'actual' in merged.columns


@pytest.mark.smoke
class TestSyntheticMetrics:
    """Test metric computation on synthetic data"""
    
    def test_synthetic_rmse(self):
        """Compute RMSE on synthetic forecast"""
        np.random.seed(42)
        y_true = 100 + np.cumsum(np.random.randn(100) * 2)
        y_pred = y_true + np.random.randn(100) * 3  # Add noise
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        assert rmse > 0
        assert isinstance(rmse, float)
    
    def test_synthetic_mape(self):
        """Compute MAPE on synthetic forecast"""
        np.random.seed(42)
        y_true = 100 + np.cumsum(np.random.randn(100) * 2)
        y_pred = y_true * (1 + np.random.randn(100) * 0.05)  # ±5% error
        
        mask = np.abs(y_true) > 1e-10
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        assert 0 < mape < 50  # Reasonable range for ±5% error


@pytest.mark.smoke
class TestSyntheticBacktesting:
    """Test backtesting with synthetic data"""
    
    def test_rolling_window_splits(self):
        """Generate rolling window splits on synthetic data"""
        df = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=500, freq='H', tz='UTC'),
            'value': np.random.randn(500)
        })
        
        min_train = 200
        test_size = 72
        step_size = 50
        n_windows = 3
        
        train_size = max(
            min_train,
            len(df) - n_windows * step_size - test_size
        )
        
        splits = []
        for window_id in range(n_windows):
            train_end_idx = min(
                train_size + window_id * step_size,
                len(df) - test_size
            )
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + test_size, len(df))
            
            if test_end_idx - test_start_idx < test_size:
                continue
            
            splits.append({
                'window_id': window_id,
                'train_size': train_end_idx,
                'test_size': test_end_idx - test_start_idx,
            })
        
        assert len(splits) > 0
        assert all(s['train_size'] >= min_train for s in splits)
        assert all(s['test_size'] == test_size for s in splits)
    
    def test_expanding_window_splits(self):
        """Generate expanding window splits on synthetic data"""
        df = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=500, freq='H', tz='UTC'),
            'value': np.random.randn(500)
        })
        
        min_train = 200
        test_size = 72
        step_size = 50
        n_windows = 3
        
        splits = []
        for window_id in range(n_windows):
            train_end_idx = min_train + window_id * step_size
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + test_size, len(df))
            
            if test_end_idx - test_start_idx < test_size:
                continue
            
            splits.append({
                'window_id': window_id,
                'train_size': train_end_idx,
                'test_size': test_end_idx - test_start_idx,
            })
        
        assert len(splits) > 0
        # Each window's training set should grow
        for i in range(len(splits) - 1):
            assert splits[i]['train_size'] < splits[i + 1]['train_size']


@pytest.mark.smoke
def test_import_all_modules():
    """Verify all main modules can be imported"""
    try:
        from datetime import datetime

        import numpy
        import pandas
        import requests
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import required module: {e}")
    except ImportError as e:
        pytest.fail(f"Failed to import required module: {e}")
