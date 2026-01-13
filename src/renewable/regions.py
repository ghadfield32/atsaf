"""EIA region definitions with geographic coordinates.

This module maps EIA balancing authority regions to their approximate
geographic centroids for weather data lookup.
"""

from typing import NamedTuple


class RegionInfo(NamedTuple):
    """Region metadata for EIA and weather lookups."""

    name: str
    lat: float
    lon: float
    timezone: str


# EIA Balancing Authority regions with centroid coordinates
# Coordinates are approximate centroids for weather API lookup
REGIONS: dict[str, RegionInfo] = {
    # Western Interconnection
    "CALI": RegionInfo(
        name="California ISO",
        lat=36.7,
        lon=-119.4,
        timezone="America/Los_Angeles",
    ),
    "NW": RegionInfo(
        name="Northwest",
        lat=45.5,
        lon=-122.0,
        timezone="America/Los_Angeles",
    ),
    "SW": RegionInfo(
        name="Southwest",
        lat=33.5,
        lon=-112.0,
        timezone="America/Phoenix",
    ),

    # Texas Interconnection
    "ERCO": RegionInfo(
        name="ERCOT (Texas)",
        lat=31.0,
        lon=-100.0,
        timezone="America/Chicago",
    ),

    # Eastern Interconnection - Northeast
    "NE": RegionInfo(
        name="New England ISO",
        lat=42.3,
        lon=-71.5,
        timezone="America/New_York",
    ),
    "NY": RegionInfo(
        name="New York ISO",
        lat=42.5,
        lon=-75.5,
        timezone="America/New_York",
    ),
    "PJM": RegionInfo(
        name="PJM Interconnection",
        lat=40.0,
        lon=-77.0,
        timezone="America/New_York",
    ),

    # Eastern Interconnection - Midwest
    "MISO": RegionInfo(
        name="Midcontinent ISO",
        lat=41.0,
        lon=-93.0,
        timezone="America/Chicago",
    ),
    "SWPP": RegionInfo(
        name="Southwest Power Pool",
        lat=36.0,
        lon=-97.0,
        timezone="America/Chicago",
    ),
    "CENT": RegionInfo(
        name="Central",
        lat=39.0,
        lon=-95.0,
        timezone="America/Chicago",
    ),

    # Eastern Interconnection - Southeast
    "SE": RegionInfo(
        name="Southeast",
        lat=33.0,
        lon=-84.0,
        timezone="America/New_York",
    ),
    "FLA": RegionInfo(
        name="Florida",
        lat=28.0,
        lon=-82.0,
        timezone="America/New_York",
    ),
    "CAR": RegionInfo(
        name="Carolinas",
        lat=35.5,
        lon=-80.0,
        timezone="America/New_York",
    ),
    "TEN": RegionInfo(
        name="Tennessee Valley",
        lat=35.5,
        lon=-86.0,
        timezone="America/Chicago",
    ),

    # Aggregates
    "US48": RegionInfo(
        name="Lower 48 States",
        lat=39.8,
        lon=-98.5,
        timezone="America/Chicago",
    ),
}

# Fuel type codes for renewable generation
FUEL_TYPES = {
    "WND": "Wind",
    "SUN": "Solar",
}


def get_region_coords(region_code: str) -> tuple[float, float]:
    """Return (lat, lon) for weather API lookup.

    Args:
        region_code: EIA region code (e.g., 'CALI', 'ERCO')

    Returns:
        Tuple of (latitude, longitude)

    Raises:
        KeyError: If region_code is not found
    """
    region = REGIONS[region_code]
    return (region.lat, region.lon)


def list_regions() -> list[str]:
    """Return all valid region codes.

    Returns:
        List of region codes sorted alphabetically
    """
    return sorted(REGIONS.keys())


def get_region_info(region_code: str) -> RegionInfo:
    """Get full region information.

    Args:
        region_code: EIA region code

    Returns:
        RegionInfo named tuple with name, lat, lon, timezone
    """
    return REGIONS[region_code]


def validate_region(region_code: str) -> bool:
    """Check if region code is valid.

    Args:
        region_code: Region code to validate

    Returns:
        True if valid, False otherwise
    """
    return region_code in REGIONS


def validate_fuel_type(fuel_type: str) -> bool:
    """Check if fuel type code is valid.

    Args:
        fuel_type: Fuel type code (WND, SUN)

    Returns:
        True if valid, False otherwise
    """
    return fuel_type in FUEL_TYPES
