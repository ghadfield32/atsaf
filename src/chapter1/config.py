"""
Chapter 1 Step 1: Configuration + Secrets

Keep EIA_API_KEY in env (prod) / .env (local).
Use a Settings object so every run logs the same config.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    """Configuration for EIA data pipeline"""
    api_key: str
    respondent: str = "US48"
    fueltype: str = "NG"
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    page_size: int = 5000


def load_settings(
    respondent: str = "US48",
    fueltype: str = "NG",
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
) -> Settings:
    """
    Load settings from environment.

    Reads EIA_API_KEY from .env file or environment variable.
    """
    load_dotenv()

    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise ValueError("EIA_API_KEY not found. Set it in .env or environment.")

    return Settings(
        api_key=api_key,
        respondent=respondent,
        fueltype=fueltype,
        start_date=start_date,
        end_date=end_date,
    )
