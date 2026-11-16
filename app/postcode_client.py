"""UK postcode -> lat/long via postcodes.io -- free, keyless, no auth."""
import requests

BASE_URL = "https://api.postcodes.io/postcodes"
TIMEOUT_SECONDS = 10


class PostcodeLookupError(Exception):
    """postcodes.io is unreachable, timed out, or returned a server error."""


def geocode(postcode: str) -> tuple[float, float]:
    """Returns (lat, long). Raises ValueError for an unrecognized postcode,
    PostcodeLookupError if the postcode service itself fails."""
    try:
        response = requests.get(f"{BASE_URL}/{postcode}", timeout=TIMEOUT_SECONDS)
        if response.status_code == 404:
            raise ValueError(f"postcode not found: {postcode}")
        response.raise_for_status()
        result = response.json()["result"]
    except (requests.RequestException, KeyError) as exc:
        raise PostcodeLookupError("postcode lookup service unavailable") from exc
    return result["latitude"], result["longitude"]
