"""Renewable energy forecasting module.

This module provides tools for:
- Fetching wind/solar generation data from EIA API
- Integrating weather features from Open-Meteo
- Multi-series probabilistic forecasting
- Drift monitoring dashboard

Usage:
    from src.renewable import EIARenewableFetcher, OpenMeteoRenewable, RenewableForecastModel

    # Fetch generation data
    fetcher = EIARenewableFetcher()
    generation = fetcher.fetch_all_regions("WND", "2024-01-01", "2024-01-07")

    # Fetch weather
    weather = OpenMeteoRenewable()
    weather_data = weather.fetch_all_regions_historical(["CALI", "ERCO"], "2024-01-01", "2024-01-07")

    # Train and forecast
    model = RenewableForecastModel(horizon=24)
    model.fit(generation, weather_data)
    forecasts = model.predict()
"""

from src.renewable.regions import (
    FUEL_TYPES,
    REGIONS,
    get_region_coords,
    get_region_info,
    list_regions,
    validate_fuel_type,
    validate_region,
)
from src.renewable.eia_renewable import EIARenewableFetcher
from src.renewable.open_meteo import OpenMeteoRenewable
from src.renewable.modeling import (
    ForecastConfig,
    RenewableForecastModel,
    compute_baseline_metrics,
)
from src.renewable.tasks import (
    RenewablePipelineConfig,
    fetch_renewable_data,
    fetch_renewable_weather,
    generate_renewable_forecasts,
    run_full_pipeline,
)
from src.renewable.db import (
    init_renewable_db,
    save_forecasts,
    save_weather,
    save_drift_alert,
    get_recent_forecasts,
    get_drift_alerts,
)

__all__ = [
    # Regions
    "REGIONS",
    "FUEL_TYPES",
    "get_region_coords",
    "get_region_info",
    "list_regions",
    "validate_region",
    "validate_fuel_type",
    # Data fetching
    "EIARenewableFetcher",
    "OpenMeteoRenewable",
    # Modeling
    "ForecastConfig",
    "RenewableForecastModel",
    "compute_baseline_metrics",
    # Pipeline
    "RenewablePipelineConfig",
    "fetch_renewable_data",
    "fetch_renewable_weather",
    "generate_renewable_forecasts",
    "run_full_pipeline",
    # Database
    "init_renewable_db",
    "save_forecasts",
    "save_weather",
    "save_drift_alert",
    "get_recent_forecasts",
    "get_drift_alerts",
]
