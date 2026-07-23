"""EA Hydrology API client for on-demand historical backtests.

Mirrors ea_client.py's shape but hits the Hydrology API instead of the live
flood-monitoring API -- that's the one with deep history (used to build the
training set, see src/download_readings.py), so it's what backtesting a past
date needs. This does a single-station lookup per request (via the
?wiskiID= filter) rather than the bulk fetch src/download_readings.py does
for training -- fetching the whole station list on every backtest request
would be wasteful.
"""
import datetime as dt

import requests

from src.download_readings import instantaneous_measure_id

STATIONS_URL = "https://environment.data.gov.uk/hydrology/id/stations.json"
READINGS_URL = "https://environment.data.gov.uk/hydrology/id/measures/{measure_id}/readings"
TIMEOUT_SECONDS = 30


def get_measure_id(wiski_id: str, parameter: str) -> str:
    response = requests.get(STATIONS_URL, params={"wiskiID": wiski_id}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    items = response.json().get("items", [])
    if not items:
        raise ValueError(f"no Hydrology station found for wiskiID {wiski_id}")
    measure_id = instantaneous_measure_id(items[0].get("measures", []), parameter)
    if not measure_id:
        raise ValueError(f"no '{parameter}' measure found for wiskiID {wiski_id}")
    return measure_id


def get_readings(measure_id: str, start: dt.date, end: dt.date) -> list[dict]:
    """Readings for [start, end] inclusive -- callers should not request beyond
    `end` when backtesting, to avoid feeding future data into a past prediction."""
    url = READINGS_URL.format(measure_id=measure_id)
    response = requests.get(
        url,
        params={"mineq-date": start.isoformat(), "maxeq-date": end.isoformat(), "_limit": 1_000_000},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json().get("items", [])
