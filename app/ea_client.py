import requests

BASE_URL = "https://environment.data.gov.uk/flood-monitoring"
TIMEOUT_SECONDS = 10


def get_latest_readings(station_id: str) -> list[dict]:
    url = f"{BASE_URL}/id/stations/{station_id}/readings"
    response = requests.get(url, params={"_sorted": "", "_limit": 10}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("items", [])


def get_station(station_id: str) -> dict:
    url = f"{BASE_URL}/id/stations/{station_id}"
    response = requests.get(url, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("items", {})
