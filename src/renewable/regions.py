# src/renewable/regions.py
from __future__ import annotations

from typing import NamedTuple, Optional


class RegionInfo(NamedTuple):
    """Region metadata for EIA and weather lookups."""
    name: str
    lat: float
    lon: float
    timezone: str
    # Some internal regions may not map cleanly to an EIA respondent.
    # We keep them in REGIONS for weather/features, but EIA fetch requires this.
    eia_respondent: Optional[str] = None


REGIONS: dict[str, RegionInfo] = {
    # Western Interconnection
    "CALI": RegionInfo(
        name="California ISO",
        lat=36.7,
        lon=-119.4,
        timezone="America/Los_Angeles",
        eia_respondent="CISO",
    ),
    "NW": RegionInfo(
        name="Northwest",
        lat=45.5,
        lon=-122.0,
        timezone="America/Los_Angeles",
        eia_respondent=None,  # intentionally unset until verified
    ),
    "SW": RegionInfo(
        name="Southwest",
        lat=33.5,
        lon=-112.0,
        timezone="America/Phoenix",
        eia_respondent=None,  # intentionally unset until verified
    ),

    # Texas Interconnection
    "ERCO": RegionInfo(
        name="ERCOT (Texas)",
        lat=31.0,
        lon=-100.0,
        timezone="America/Chicago",
        eia_respondent="ERCO",
    ),

    # Midwest
    "MISO": RegionInfo(
        name="Midcontinent ISO",
        lat=41.0,
        lon=-93.0,
        timezone="America/Chicago",
        eia_respondent="MISO",
    ),
    "PJM": RegionInfo(
        name="PJM Interconnection",
        lat=39.0,
        lon=-77.0,
        timezone="America/New_York",
        eia_respondent="PJM",
    ),
    "SWPP": RegionInfo(
        name="Southwest Power Pool",
        lat=37.0,
        lon=-97.0,
        timezone="America/Chicago",
        eia_respondent="SWPP",
    ),

    # Internal/aggregate regions kept for non-EIA use (weather/features/etc.)
    "SE": RegionInfo(name="Southeast", lat=33.0, lon=-84.0, timezone="America/New_York", eia_respondent=None),
    "FLA": RegionInfo(name="Florida", lat=28.0, lon=-82.0, timezone="America/New_York", eia_respondent=None),
    "CAR": RegionInfo(name="Carolinas", lat=35.5, lon=-80.0, timezone="America/New_York", eia_respondent=None),
    "TEN": RegionInfo(name="Tennessee Valley", lat=35.5, lon=-86.0, timezone="America/Chicago", eia_respondent=None),

    "US48": RegionInfo(name="Lower 48 States", lat=39.8, lon=-98.5, timezone="America/Chicago", eia_respondent=None),
}

FUEL_TYPES = {"WND": "Wind", "SUN": "Solar"}


def list_regions() -> list[str]:
    return sorted(REGIONS.keys())


def get_region_info(region_code: str) -> RegionInfo:
    return REGIONS[region_code]


def get_region_coords(region_code: str) -> tuple[float, float]:
    r = REGIONS[region_code]
    return (r.lat, r.lon)


def get_eia_respondent(region_code: str) -> str:
    """Return the code EIA expects for facets[respondent][]. Fail loudly if missing."""
    info = REGIONS[region_code]
    if not info.eia_respondent:
        raise ValueError(
            f"Region '{region_code}' has no configured eia_respondent. "
            f"Set REGIONS['{region_code}'].eia_respondent to a verified EIA respondent code "
            f"before using it for EIA fetches."
        )
    return info.eia_respondent


def validate_region(region_code: str) -> bool:
    return region_code in REGIONS


def validate_fuel_type(fuel_type: str) -> bool:
    return fuel_type in FUEL_TYPES



if __name__ == "__main__":
    # Example run - test region functions

    print("=== Available Regions ===")
    print(f"Total regions: {len(REGIONS)}")
    print(f"Region codes: {list_regions()}")

    print("\n=== Example: California ===")
    cali_info = get_region_info("CALI")
    print(f"Name: {cali_info.name}")
    print(f"Coordinates: ({cali_info.lat}, {cali_info.lon})")
    print(f"Timezone: {cali_info.timezone}")

    print("\n=== Weather API Coordinates ===")
    for region in ["CALI", "ERCO", "MISO"]:
        lat, lon = get_region_coords(region)
        print(f"{region}: lat={lat}, lon={lon}")

    print("\n=== Fuel Types ===")
    for code, name in FUEL_TYPES.items():
        print(f"{code}: {name}")

    print("\n=== Validation ===")
    print(f"validate_region('CALI'): {validate_region('CALI')}")
    print(f"validate_region('INVALID'): {validate_region('INVALID')}")
    print(f"validate_fuel_type('WND'): {validate_fuel_type('WND')}")
