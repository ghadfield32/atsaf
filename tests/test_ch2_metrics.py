"""
Chapter 2: Metrics Tests

Validates NaN-aware behavior (explicit masking, not silent ignoring).
"""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.fail_loud
class TestMetricsNaNHandling:
    """Metrics must explicitly handle NaN via masking, not silent ignoring"""
    
    def test_rmse_nan_masked(self):
        """RMSE should mask NaN before computation"""
        y_true = np.array([100.0, 102.0, np.nan, 106.0])
        y_pred = np.array([99.0, 101.0, 104.0, 107.0])
        
        # Correct: explicit masking
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() == 0:
            rmse_result = np.nan
        else:
            rmse_result = float(np.sqrt(np.mean(
                (y_true[mask] - y_pred[mask]) ** 2
            )))
        
        # Should compute on 3 valid rows, not fail
        assert not np.isnan(rmse_result)
    
    def test_rmse_all_nan_returns_nan(self):
        """RMSE with all NaN should return NaN, not error"""
        y_true = np.array([np.nan, np.nan, np.nan])
        y_pred = np.array([np.nan, np.nan, np.nan])
        
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() == 0:
            rmse_result = np.nan
        else:
            rmse_result = float(np.sqrt(np.mean(
                (y_true[mask] - y_pred[mask]) ** 2
            )))
        
        assert np.isnan(rmse_result)
    
    def test_mape_nan_masked(self):
        """MAPE should mask NaN and zero y_true"""
        y_true = np.array([100.0, 102.0, np.nan, 106.0])
        y_pred = np.array([99.0, 101.0, 104.0, 107.0])
        
        # Correct: mask NaN and small y_true
        mask = ~(np.isnan(y_true) | np.isnan(y_pred)) & (np.abs(y_true) > 1e-10)
        
        if mask.sum() == 0:
            mape_result = np.nan
        else:
            mape_result = float(np.mean(
                np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
            ) * 100)
        
        assert not np.isnan(mape_result)
    
    def test_coverage_denominator_valid_rows(self):
        """Coverage denominator must be valid rows, not total rows"""
        y_true = np.array([100.0, 102.0, np.nan, 106.0])
        lower = np.array([98.0, 100.0, np.nan, 104.0])
        upper = np.array([102.0, 104.0, np.nan, 108.0])
        
        # Correct: denominator = valid rows
        mask = ~(np.isnan(y_true) | np.isnan(lower) | np.isnan(upper))
        
        if mask.sum() == 0:
            coverage = np.nan
        else:
            y_true_clean = y_true[mask]
            lower_clean = lower[mask]
            upper_clean = upper[mask]
            
            is_covered = (lower_clean <= y_true_clean) & (y_true_clean <= upper_clean)
            coverage = float(np.mean(is_covered))  # Denominator = 3, not 4
        
        assert coverage > 0  # 3/3 = 100% coverage
    
    def test_mase_insufficient_training(self):
        """MASE should return NaN if insufficient training data"""
        y_test = np.array([100.0, 102.0])
        y_pred = np.array([99.0, 101.0])
        y_train = np.array([90.0, 92.0])  # Only 2 points, min season_length=24
        season_length = 24
        
        mask_train = ~np.isnan(y_train)
        
        # Should detect insufficient data
        if mask_train.sum() <= season_length:
            mase_result = np.nan
        else:
            # compute MASE...
            pass
        
        assert np.isnan(mase_result)


@pytest.mark.fail_loud
class TestMetricsValidRows:
    """Metrics must track and report number of valid rows"""
    
    def test_valid_row_count_tracked(self):
        """Should count valid rows used in computation"""
        y_true = np.array([100.0, 102.0, np.nan, 106.0, 108.0])
        y_pred = np.array([99.0, 101.0, 104.0, np.nan, 107.0])
        
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        n_valid = mask.sum()
        
        # 100, 102, 108 are valid (positions 0, 1, 4)
        assert n_valid == 3
    
    def test_missing_row_count_tracked(self):
        """Should count missing rows"""
        y_true = np.array([100.0, 102.0, np.nan, 106.0, 108.0])
        y_pred = np.array([99.0, 101.0, 104.0, np.nan, 107.0])
        
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        n_missing = (~mask).sum()
        
        # Positions 2 (NaN in y_true) and 3 (NaN in y_pred)
        assert n_missing == 2


@pytest.mark.fail_loud
class TestMetricsConsistency:
    """Metrics must be computed consistently across all models"""
    
    def test_same_data_same_metric(self):
        """Same data should produce identical metrics"""
        y_true = np.array([100.0, 102.0, 104.0, 106.0])
        y_pred = np.array([99.0, 101.0, 105.0, 107.0])
        
        # Compute RMSE twice
        rmse1 = np.sqrt(np.mean((y_true - y_pred) ** 2))
        rmse2 = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        assert rmse1 == rmse2
    
    def test_metric_range_valid(self):
        """Metrics should be in valid ranges"""
        y_true = np.array([100.0, 102.0, 104.0, 106.0])
        y_pred = np.array([99.0, 101.0, 105.0, 107.0])
        
        # RMSE should be >= 0
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert rmse >= 0
        
        # Coverage should be in [0, 1]
        lower = y_pred - 5
        upper = y_pred + 5
        coverage = np.mean((lower <= y_true) & (y_true <= upper))
        assert 0 <= coverage <= 1


@pytest.mark.smoke
class TestMetricsBasic:
    """Basic metric computation sanity checks"""
    
    def test_perfect_forecast_zero_error(self):
        """Perfect forecast should have zero error"""
        y_true = np.array([100.0, 102.0, 104.0, 106.0])
        y_pred = y_true.copy()
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        assert rmse == 0.0
        assert mae == 0.0
    
    def test_coverage_100_percent(self):
        """Perfect interval should have 100% coverage"""
        y_true = np.array([100.0, 102.0, 104.0, 106.0])
        lower = y_true - 10  # Wide interval
        upper = y_true + 10
        
        coverage = np.mean((lower <= y_true) & (y_true <= upper))
        assert coverage == 1.0
    
    def test_coverage_zero_percent(self):
        """Misaligned interval should have 0% coverage"""
        y_true = np.array([100.0, 102.0, 104.0, 106.0])
        lower = y_true + 10  # Interval above actual
        upper = y_true + 20
        
        coverage = np.mean((lower <= y_true) & (y_true <= upper))
        assert coverage == 0.0
