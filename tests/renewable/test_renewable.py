"""Tests for renewable energy forecasting module.

Run with:
    pytest tests/renewable/ -v
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.renewable.regions import (
    FUEL_TYPES,
    REGIONS,
    get_region_coords,
    get_region_info,
    list_regions,
    validate_fuel_type,
    validate_region,
)
from src.renewable.modeling import (
    ForecastConfig,
    RenewableForecastModel,
    compute_baseline_metrics,
)
from src.renewable.db import (
    connect,
    init_renewable_db,
    save_forecasts,
    save_drift_alert,
    get_recent_forecasts,
    get_drift_alerts,
)


class TestRegions:
    """Tests for region definitions."""

    def test_regions_not_empty(self):
        """REGIONS dictionary should contain entries."""
        assert len(REGIONS) > 0

    def test_fuel_types_defined(self):
        """FUEL_TYPES should include WND and SUN."""
        assert "WND" in FUEL_TYPES
        assert "SUN" in FUEL_TYPES

    def test_get_region_coords_valid(self):
        """get_region_coords should return valid lat/lon."""
        lat, lon = get_region_coords("CALI")
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180

    def test_get_region_coords_invalid(self):
        """get_region_coords should raise KeyError for invalid region."""
        with pytest.raises(KeyError):
            get_region_coords("INVALID")

    def test_get_region_info(self):
        """get_region_info should return RegionInfo tuple."""
        info = get_region_info("ERCO")
        assert info.name == "ERCOT (Texas)"
        assert info.lat == 31.0
        assert info.lon == -100.0

    def test_list_regions(self):
        """list_regions should return sorted list."""
        regions = list_regions()
        assert isinstance(regions, list)
        assert len(regions) > 0
        assert regions == sorted(regions)

    def test_validate_region(self):
        """validate_region should work correctly."""
        assert validate_region("CALI") is True
        assert validate_region("INVALID") is False

    def test_validate_fuel_type(self):
        """validate_fuel_type should work correctly."""
        assert validate_fuel_type("WND") is True
        assert validate_fuel_type("SUN") is True
        assert validate_fuel_type("INVALID") is False


class TestForecastConfig:
    """Tests for ForecastConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = ForecastConfig()
        assert config.horizon == 24
        assert config.confidence_levels == (80, 95)
        assert config.season_length == 24

    def test_custom_config(self):
        """Custom config values should be set correctly."""
        config = ForecastConfig(
            horizon=48,
            confidence_levels=(70, 90),
        )
        assert config.horizon == 48
        assert config.confidence_levels == (70, 90)


class TestRenewableForecastModel:
    """Tests for RenewableForecastModel."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=720, freq="h")
        series_ids = ["CALI_WND", "ERCO_WND"]

        dfs = []
        for sid in series_ids:
            y = 100 + 20 * np.sin(np.arange(720) * 2 * np.pi / 24) + np.random.normal(0, 5, 720)
            dfs.append(pd.DataFrame({"unique_id": sid, "ds": dates, "y": y}))

        return pd.concat(dfs, ignore_index=True)

    def test_model_initialization(self):
        """Model should initialize with default parameters."""
        model = RenewableForecastModel()
        assert model.horizon == 24
        assert model.confidence_levels == (80, 95)
        assert model.fitted is False

    def test_prepare_features(self, sample_data):
        """prepare_features should add time features."""
        model = RenewableForecastModel()
        features = model.prepare_features(sample_data)

        assert "hour_sin" in features.columns
        assert "hour_cos" in features.columns
        assert "dow_sin" in features.columns
        assert "dow_cos" in features.columns

    def test_prepare_features_lag(self, sample_data):
        """prepare_features should add lag features."""
        model = RenewableForecastModel()
        features = model.prepare_features(sample_data)

        assert "y_lag_1" in features.columns
        assert "y_lag_24" in features.columns

    def test_predict_without_fit_raises(self):
        """predict() without fit() should raise RuntimeError."""
        model = RenewableForecastModel()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict()


class TestMetrics:
    """Tests for metrics computation."""

    def test_compute_metrics_no_mape(self):
        """compute_metrics should NOT include MAPE (solar has zeros)."""
        model = RenewableForecastModel()

        y_true = np.array([100, 0, 50, 0, 75])  # Zeros for solar at night
        y_pred = np.array([95, 5, 48, 2, 80])

        metrics = model.compute_metrics(y_true, y_pred)

        assert "rmse" in metrics
        assert "mae" in metrics
        # MAPE should NOT be in metrics (undefined for y=0)
        assert "mape" not in metrics


class TestDatabase:
    """Tests for database operations."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test.db"
            init_renewable_db(db_path)
            yield db_path

    def test_init_creates_tables(self, temp_db):
        """init_renewable_db should create all required tables."""
        con = connect(temp_db)
        cur = con.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}

        assert "renewable_forecasts" in tables
        assert "renewable_scores" in tables
        assert "weather_features" in tables
        assert "drift_alerts" in tables
        assert "baseline_metrics" in tables

        con.close()

    def test_save_and_get_forecasts(self, temp_db):
        """save_forecasts and get_recent_forecasts should work."""
        forecasts = pd.DataFrame({
            "unique_id": ["CALI_WND", "CALI_WND"],
            "ds": [datetime.utcnow(), datetime.utcnow() + timedelta(hours=1)],
            "yhat": [100.0, 105.0],
            "yhat_lo_80": [90.0, 95.0],
            "yhat_hi_80": [110.0, 115.0],
            "yhat_lo_95": [80.0, 85.0],
            "yhat_hi_95": [120.0, 125.0],
        })

        rows = save_forecasts(temp_db, forecasts, run_id="test_run")
        assert rows == 2

        # Get forecasts back
        result = get_recent_forecasts(temp_db, hours=24)
        assert len(result) == 2

    def test_save_drift_alert(self, temp_db):
        """save_drift_alert should store alert correctly."""
        save_drift_alert(
            temp_db,
            run_id="test_run",
            unique_id="CALI_WND",
            current_rmse=150.0,
            threshold_rmse=100.0,
            severity="warning",
        )

        alerts = get_drift_alerts(temp_db, hours=24)
        assert len(alerts) == 1
        assert alerts.iloc[0]["severity"] == "warning"

    def test_get_drift_alerts_empty(self, temp_db):
        """get_drift_alerts should return empty DataFrame when no alerts."""
        alerts = get_drift_alerts(temp_db, hours=24)
        assert len(alerts) == 0


class TestBaselineMetrics:
    """Tests for baseline metrics computation."""

    @pytest.fixture
    def sample_cv_results(self):
        """Create sample CV results for testing."""
        np.random.seed(42)

        data = []
        for cutoff_offset in range(5):
            cutoff = datetime(2024, 1, 1) + timedelta(days=cutoff_offset * 7)
            for h in range(24):
                ds = cutoff + timedelta(hours=h)
                y = 100 + np.random.normal(0, 10)
                yhat = y + np.random.normal(0, 5)

                data.append({
                    "unique_id": "CALI_WND",
                    "ds": ds,
                    "cutoff": cutoff,
                    "y": y,
                    "MSTL_ARIMA": yhat,
                })

        return pd.DataFrame(data)

    def test_compute_baseline_metrics(self, sample_cv_results):
        """compute_baseline_metrics should return expected keys."""
        baseline = compute_baseline_metrics(sample_cv_results, model_name="MSTL_ARIMA")

        assert "model" in baseline
        assert "rmse_mean" in baseline
        assert "rmse_std" in baseline
        assert "drift_threshold_rmse" in baseline

    def test_drift_threshold_formula(self, sample_cv_results):
        """Drift threshold should be mean + 2*std."""
        baseline = compute_baseline_metrics(sample_cv_results, model_name="MSTL_ARIMA")

        expected_threshold = baseline["rmse_mean"] + 2 * baseline["rmse_std"]
        assert abs(baseline["drift_threshold_rmse"] - expected_threshold) < 0.01

    def test_compute_baseline_invalid_model(self, sample_cv_results):
        """compute_baseline_metrics should raise for invalid model."""
        with pytest.raises(ValueError, match="not found"):
            compute_baseline_metrics(sample_cv_results, model_name="INVALID_MODEL")


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_workflow_with_synthetic_data(self):
        """Test full workflow from data to forecast."""
        np.random.seed(42)

        # Create synthetic data
        dates = pd.date_range("2024-01-01", periods=360, freq="h")
        df = pd.DataFrame({
            "unique_id": "TEST_WND",
            "ds": dates,
            "y": 100 + 20 * np.sin(np.arange(360) * 2 * np.pi / 24) + np.random.normal(0, 5, 360),
        })

        # Initialize model
        model = RenewableForecastModel(horizon=24, confidence_levels=(80, 95))

        # Fit model
        model.fit(df)
        assert model.fitted is True

        # Generate predictions
        predictions = model.predict()
        assert len(predictions) > 0
        assert "yhat" in predictions.columns
        assert "yhat_lo_80" in predictions.columns
        assert "yhat_hi_95" in predictions.columns

    def test_cv_produces_leaderboard(self):
        """Cross-validation should produce valid leaderboard."""
        np.random.seed(42)

        dates = pd.date_range("2024-01-01", periods=720, freq="h")
        df = pd.DataFrame({
            "unique_id": "TEST_WND",
            "ds": dates,
            "y": 100 + 20 * np.sin(np.arange(720) * 2 * np.pi / 24) + np.random.normal(0, 5, 720),
        })

        model = RenewableForecastModel(horizon=24)
        cv_results, leaderboard = model.cross_validate(df, n_windows=3, step_size=168)

        assert len(leaderboard) > 0
        assert "model" in leaderboard.columns
        assert "rmse" in leaderboard.columns
        assert leaderboard["rmse"].iloc[0] <= leaderboard["rmse"].iloc[-1]  # Sorted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
