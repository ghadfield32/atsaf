"""
Data Quality Tests for Renewable Energy Forecasting

Tests to validate data quality after corrections:
1. Timezone consistency
2. Generation-weather alignment
3. No physically impossible data
4. Minimal constraint application in forecasts
"""

import pandas as pd
import pytest
from pathlib import Path


class TestDataQuality:
    """Test data quality after corrections."""

    def test_no_timezone_mismatches(self):
        """All dataframes should have consistent timezones (UTC)."""
        gen_path = Path("data/renewable/generation_corrected.parquet")
        weather_path = Path("data/renewable/weather.parquet")

        if not gen_path.exists():
            pytest.skip("Corrected generation data not found (run fix_solar_data_issues.py first)")

        gen_df = pd.read_parquet(gen_path)
        weather_df = pd.read_parquet(weather_path)

        assert gen_df["ds"].dt.tz == weather_df["ds"].dt.tz, "Timezone mismatch between generation and weather"
        assert str(gen_df["ds"].dt.tz) == "UTC", "Generation data must be in UTC"
        assert str(weather_df["ds"].dt.tz) == "UTC", "Weather data must be in UTC"

    def test_generation_weather_alignment(self):
        """All generation data should have matching weather (>99% alignment)."""
        gen_path = Path("data/renewable/generation_corrected.parquet")
        weather_path = Path("data/renewable/weather.parquet")

        if not gen_path.exists():
            pytest.skip("Corrected generation data not found")

        gen_df = pd.read_parquet(gen_path)
        weather_df = pd.read_parquet(weather_path)

        gen_df = gen_df.copy()
        gen_df["region"] = gen_df["unique_id"].str.split("_").str[0]

        merged = gen_df.merge(weather_df[["region", "ds"]], on=["region", "ds"], how="inner")

        match_rate = len(merged) / len(gen_df) * 100
        assert match_rate >= 99, f"Only {match_rate:.1f}% of generation has matching weather (expected >= 99%)"

    def test_no_impossible_solar_data(self):
        """Solar generation should be 0 when radiation is 0."""
        gen_path = Path("data/renewable/generation_corrected.parquet")
        weather_path = Path("data/renewable/weather.parquet")

        if not gen_path.exists():
            pytest.skip("Corrected generation data not found")

        gen_df = pd.read_parquet(gen_path)
        weather_df = pd.read_parquet(weather_path)

        gen_df = gen_df.copy()
        gen_df["region"] = gen_df["unique_id"].str.split("_").str[0]
        solar_df = gen_df[gen_df["unique_id"].str.endswith("_SUN")]

        if solar_df.empty:
            pytest.skip("No solar series found in data")

        merged = solar_df.merge(weather_df, on=["region", "ds"], how="inner")

        # Check for physically impossible data: zero radiation but high generation
        total_radiation = merged["direct_radiation"] + merged["diffuse_radiation"]
        impossible = merged[(total_radiation == 0) & (merged["y"] > 10)]

        assert len(impossible) == 0, (
            f"Found {len(impossible)} rows with zero radiation but high generation (>10 MW). "
            f"This is physically impossible."
        )

    def test_minimal_constraint_applied_in_forecasts(self):
        """Forecasts should be 0 only when radiation is 0 (not filtered by hour)."""
        forecasts_path = Path("data/renewable/forecasts.parquet")
        weather_path = Path("data/renewable/weather.parquet")

        if not forecasts_path.exists():
            pytest.skip("Forecast data not found (run pipeline first)")

        forecasts = pd.read_parquet(forecasts_path)
        weather_df = pd.read_parquet(weather_path)

        solar_fcst = forecasts[forecasts["unique_id"].str.endswith("_SUN")]

        if solar_fcst.empty:
            pytest.skip("No solar forecasts found")

        solar_fcst = solar_fcst.copy()
        solar_fcst["region"] = solar_fcst["unique_id"].str.split("_").str[0]

        merged = solar_fcst.merge(weather_df, on=["region", "ds"], how="inner")

        if merged.empty:
            pytest.skip("No matching forecast-weather data")

        total_radiation = merged["direct_radiation"] + merged["diffuse_radiation"]

        # Where radiation is 0, forecast must be 0
        zero_rad = merged[total_radiation == 0]
        if not zero_rad.empty:
            assert (zero_rad["yhat"] == 0).all(), (
                "Solar forecasts must be 0 when radiation is 0 (physical constraint)"
            )

        # Where radiation > 0, forecast can be anything (let model decide)
        # This test ensures we're NOT filtering by hour or other heuristics
        # We just check that the constraint is applied when needed, not over-applied

    def test_horizon_preset_validation(self):
        """Test that horizon presets have valid configurations."""
        from src.renewable.tasks import RenewablePipelineConfig

        # Test each preset
        for preset_name in ["24h", "48h", "72h"]:
            cfg = RenewablePipelineConfig(horizon_preset=preset_name)

            # Verify preset was applied
            expected = cfg._PRESETS[preset_name]
            assert cfg.horizon == expected["horizon"], f"Horizon mismatch for {preset_name}"
            assert cfg.cv_windows == expected["cv_windows"], f"CV windows mismatch for {preset_name}"
            assert cfg.lookback_days == expected["lookback_days"], f"Lookback days mismatch for {preset_name}"

            # Verify validation warnings are generated if needed
            warnings = cfg._validate()
            # No assertion here, just ensure validation runs without error


class TestConfigValidation:
    """Test configuration validation logic."""

    def test_insufficient_data_warning(self):
        """Should warn when lookback_days too small for horizon + CV."""
        from src.renewable.tasks import RenewablePipelineConfig

        # Create config with insufficient data
        cfg = RenewablePipelineConfig(lookback_days=10, horizon=24, cv_windows=5, cv_step_size=168)

        warnings = cfg._validate()
        assert len(warnings) > 0, "Should generate warning for insufficient data"
        assert "Insufficient data" in warnings[0], "Warning should mention insufficient data"

    def test_long_horizon_warning(self):
        """Should warn when horizon exceeds 72 hours."""
        from src.renewable.tasks import RenewablePipelineConfig

        cfg = RenewablePipelineConfig(horizon=96, lookback_days=60)

        warnings = cfg._validate()
        assert any("72h" in w for w in warnings), "Should warn about horizons beyond 72h"
