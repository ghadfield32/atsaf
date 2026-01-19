"""Smoke tests for renewable pipeline with mocked API responses.

Run with:
    pytest tests/renewable/test_smoke.py -v
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.renewable.validation import ValidationReport, validate_generation_df


class TestValidationSmoke:
    """Smoke tests for validation module."""

    def test_validate_rejects_negatives(self):
        """Validation should fail on negative generation values."""
        df = pd.DataFrame({
            "unique_id": ["CALI_SUN"] * 5,
            "ds": pd.date_range("2024-01-01", periods=5, freq="h"),
            "y": [100, 50, -10, 80, 90],
        })

        report = validate_generation_df(df, max_lag_hours=100000)

        assert not report.ok
        assert "Negative" in report.message
        assert report.details["neg_y"] == 1
        assert "by_series" in report.details
        assert "sample" in report.details

    def test_validate_accepts_positive(self):
        """Validation should pass on positive generation values (ignoring freshness)."""
        df = pd.DataFrame({
            "unique_id": ["CALI_SUN"] * 5,
            "ds": pd.date_range("2024-01-01", periods=5, freq="h"),
            "y": [100, 50, 0, 80, 90],
        })

        report = validate_generation_df(df, max_lag_hours=100000)

        if not report.ok:
            assert "Negative" not in report.message

    def test_validate_accepts_zeros(self):
        """Validation should accept zero values (solar at night)."""
        df = pd.DataFrame({
            "unique_id": ["CALI_SUN"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10, freq="h"),
            "y": [0, 0, 0, 50, 100, 150, 100, 50, 0, 0],
        })

        report = validate_generation_df(df, max_lag_hours=100000)

        if not report.ok:
            assert "Negative" not in report.message

    def test_validate_returns_series_breakdown(self):
        """Validation should return per-series breakdown for negatives."""
        df = pd.DataFrame({
            "unique_id": ["CALI_SUN"] * 3 + ["ERCO_SUN"] * 3,
            "ds": list(pd.date_range("2024-01-01", periods=3, freq="h")) * 2,
            "y": [100, -20, 50, 80, -5, 90],
        })

        report = validate_generation_df(df, max_lag_hours=100000)

        assert not report.ok
        assert report.details["neg_y"] == 2
        by_series = report.details["by_series"]
        assert len(by_series) == 2


class TestEIAFetcherSmoke:
    """Smoke tests for EIA fetcher with mocked responses."""

    @patch("src.renewable.eia_renewable.requests.get")
    @patch.dict("os.environ", {"EIA_API_KEY": "test_key"})
    def test_fetch_region_handles_negative_values(self, mock_get):
        """Fetcher should log and clamp negative values to zero."""
        from src.renewable.eia_renewable import EIARenewableFetcher

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": {
                "data": [
                    {"period": "2024-01-01T10", "value": 100},
                    {"period": "2024-01-01T11", "value": -50},
                    {"period": "2024-01-01T12", "value": 80},
                ],
                "total": 3,
            }
        }
        mock_response.url = "https://api.eia.gov/test"
        mock_get.return_value = mock_response

        fetcher = EIARenewableFetcher()
        df = fetcher.fetch_region("CALI", "SUN", "2024-01-01", "2024-01-01")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Clamping preserves all rows (vs filtering)
        assert (df["value"] >= 0).all()  # Negatives clamped to 0

    @patch("src.renewable.eia_renewable.requests.get")
    @patch.dict("os.environ", {"EIA_API_KEY": "test_key"})
    def test_fetch_region_returns_dataframe(self, mock_get):
        """Fetcher should return valid DataFrame."""
        from src.renewable.eia_renewable import EIARenewableFetcher

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": {
                "data": [
                    {"period": "2024-01-01T10", "value": 100},
                    {"period": "2024-01-01T11", "value": 150},
                    {"period": "2024-01-01T12", "value": 80},
                ],
                "total": 3,
            }
        }
        mock_response.url = "https://api.eia.gov/test"
        mock_get.return_value = mock_response

        fetcher = EIARenewableFetcher()
        df = fetcher.fetch_region("CALI", "SUN", "2024-01-01", "2024-01-01")

        assert isinstance(df, pd.DataFrame)
        assert "ds" in df.columns
        assert "value" in df.columns
        assert "region" in df.columns
        assert "fuel_type" in df.columns
        assert len(df) == 3


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_report_ok_true(self):
        """OK report should have ok=True."""
        report = ValidationReport(True, "OK", {"row_count": 100})
        assert report.ok is True
        assert report.message == "OK"

    def test_report_ok_false(self):
        """Failed report should have ok=False."""
        report = ValidationReport(False, "Error", {"error": "details"})
        assert report.ok is False
        assert report.message == "Error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
