import datetime as dt

import requests

BASE_URL = "https://environment.data.gov.uk/flood-monitoring"
TIMEOUT_SECONDS = 15
HISTORY_HOURS = 180  # >=168h (7-day rainfall) window + buffer for gaps
CACHE_TTL_SECONDS = 600  # readings update ~every 15 min; avoids hammering the EA API on repeat requests

# (station_id, parameter) -> {"readings": [...], "fetched_at": datetime}
_cache: dict[tuple[str, str], dict] = {}


def _as_list(measures):
    return measures if isinstance(measures, list) else [measures]


def get_station(station_id: str) -> dict:
    url = f"{BASE_URL}/id/stations/{station_id}"
    response = requests.get(url, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("items", {})


def get_measure_id(station_id: str, parameter: str) -> str:
    station = get_station(station_id)
    measures = [m for m in _as_list(station.get("measures", [])) if m.get("parameter") == parameter]
    if not measures:
        raise ValueError(f"no '{parameter}' measure found for station {station_id}")
    preferred = next((m for m in measures if m.get("qualifier") == "Stage"), measures[0])
    return preferred["@id"].rsplit("/", 1)[-1]


def _fetch_recent_readings(station_id: str, parameter: str, hours: int) -> list[dict]:
    measure_id = get_measure_id(station_id, parameter)
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"{BASE_URL}/id/measures/{measure_id}/readings"
    response = requests.get(url, params={"since": since, "_limit": 100_000}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("items", [])


def get_recent_readings(station_id: str, parameter: str, hours: int = HISTORY_HOURS) -> tuple[list[dict], bool]:
    """Trailing `hours` of readings for a station's given parameter (level/flow/rainfall).

    Returns (readings, stale). A short-lived cache avoids re-fetching on every
    request; if a live fetch fails and a cached (possibly older) result
    exists, that's returned instead with stale=True rather than raising --
    degrading gracefully beats a hard error. Only raises if there's nothing
    to fall back to.
    """
    key = (station_id, parameter)
    now = dt.datetime.now(dt.timezone.utc)
    cached = _cache.get(key)

    if cached and (now - cached["fetched_at"]).total_seconds() < CACHE_TTL_SECONDS:
        return cached["readings"], False

    try:
        readings = _fetch_recent_readings(station_id, parameter, hours)
        _cache[key] = {"readings": readings, "fetched_at": now}
        return readings, False
    except Exception:
        if cached is not None:
            return cached["readings"], True
        raise
