"""
Air Quality project pipeline (AQI + weather covariates).

Modules:
- config: pipeline configuration and paths
- airnow: AirNow AQI ingestion + normalization
- open_meteo: weather ingestion + normalization
- modeling: baseline StatsForecast + residual corrector
- tasks: Chapter 3-style orchestration tasks
"""
