"""
Chapter 1 Step 2: Ingest from EIA API

Production-correct pagination with stable sort:
- Uses EIA API v2 offset/length for pagination
- Adds stable sorting so pages don't shuffle
- Uses requests.Session with retries
"""

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Settings


def create_session() -> requests.Session:
    """Create session with retry logic"""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def pull_data_paged(settings: Settings) -> pd.DataFrame:
    """
    Pull data from EIA API with pagination and stable sort.

    Args:
        settings: Configuration with API key and query parameters

    Returns:
        DataFrame with all records from the API

    Implementation pattern:
    1. offset=0
    2. GET with length, offset, and sort params
    3. Append records
    4. Stop when returned records < length
    """
    url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
    session = create_session()

    all_records = []
    offset = 0
    page_size = settings.page_size

    print(f"Pulling data: {settings.start_date} to {settings.end_date}")
    print(f"  Respondent: {settings.respondent}, Fuel: {settings.fueltype}")

    while True:
        params = {
            "api_key": settings.api_key,
            "data[]": "value",
            "facets[respondent][]": settings.respondent,
            "facets[fueltype][]": settings.fueltype,
            "start": f"{settings.start_date}T00",
            "end": f"{settings.end_date}T23",
            "length": page_size,
            "offset": offset,
            # Stable sort: ascending by period
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }

        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        records = data["response"]["data"]

        if not records:
            break

        all_records.extend(records)
        print(f"  Page {offset // page_size + 1}: {len(records)} records")

        if len(records) < page_size:
            break

        offset += page_size

    df = pd.DataFrame(all_records)
    print(f"  Total: {len(df)} records")

    return df
