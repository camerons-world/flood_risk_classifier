"""UK postcode -> lat/long via postcodes.io -- free, keyless, no auth."""
import requests

BASE_URL = "https://api.postcodes.io/postcodes"
TIMEOUT_SECONDS = 10


def geocode(postcode: str) -> tuple[float, float]:
    """Returns (lat, long). Raises ValueError for an unrecognized postcode."""
    response = requests.get(f"{BASE_URL}/{postcode}", timeout=TIMEOUT_SECONDS)
    if response.status_code == 404:
        raise ValueError(f"postcode not found: {postcode}")
    response.raise_for_status()
    result = response.json()["result"]
    return result["latitude"], result["longitude"]
