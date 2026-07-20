"""Select the set of stations used for training and live inference.

Matches live flood-monitoring stations (used for real-time inference) against
the EA Hydrology API (used for historic training data) via wiskiID, then
attaches the nearest rainfall gauge to each for rainfall-accumulation features.
"""
import math

import pandas as pd
import requests

FLOOD_MONITORING_URL = "https://environment.data.gov.uk/flood-monitoring/id/stations"
HYDROLOGY_URL = "https://environment.data.gov.uk/hydrology/id/stations.json"
TIMEOUT_SECONDS = 60

# approx bounding box for Greater London
GREATER_LONDON_BBOX = {"lat_min": 51.286, "lat_max": 51.691, "long_min": -0.510, "long_max": 0.334}


def _first(value):
    return value[0] if isinstance(value, list) else value


def fetch_live_level_stations(bbox: dict | None = None) -> pd.DataFrame:
    r = requests.get(FLOOD_MONITORING_URL, params={"parameter": "level", "_limit": 10000}, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    items = r.json().get("items", [])
    rows = []
    for s in items:
        if not s.get("wiskiID") or s.get("lat") is None:
            continue
        lat, long_ = _first(s["lat"]), _first(s["long"])
        if bbox and not (bbox["lat_min"] <= lat <= bbox["lat_max"] and bbox["long_min"] <= long_ <= bbox["long_max"]):
            continue
        rows.append(
            {
                "station_id": s.get("notation"),
                "wiski_id": _first(s.get("wiskiID")),
                "label": s.get("label"),
                "river_name": s.get("riverName"),
                "catchment_name": s.get("catchmentName"),
                "lat": lat,
                "long": long_,
            }
        )
    return pd.DataFrame(rows)


def fetch_hydrology_stations() -> pd.DataFrame:
    r = requests.get(HYDROLOGY_URL, params={"_limit": 10000}, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    items = r.json().get("items", [])
    rows = []
    for s in items:
        measures = s.get("measures", [])
        params = {m.get("parameter") for m in measures}
        wiski_ids = s.get("wiskiID")
        wiski_ids = wiski_ids if isinstance(wiski_ids, list) else [wiski_ids]
        for wiski_id in wiski_ids:
            rows.append(
                {
                    "wiski_id": wiski_id,
                    "hydrology_notation": s.get("notation"),
                    "easting": s.get("easting"),
                    "northing": s.get("northing"),
                    "date_opened": s.get("dateOpened"),
                    "has_level_or_flow": bool({"level", "flow"} & params),
                    "has_rainfall": "rainfall" in params,
                }
            )
    return pd.DataFrame(rows)


def _distance(e1, n1, e2, n2):
    return math.hypot(e1 - e2, n1 - n2)


def attach_nearest_rainfall_station(level_stations: pd.DataFrame, rainfall_stations: pd.DataFrame) -> pd.DataFrame:
    rainfall_stations = rainfall_stations.dropna(subset=["easting", "northing"])
    nearest_ids = []
    nearest_dist_m = []
    for _, row in level_stations.iterrows():
        if pd.isna(row["easting"]) or pd.isna(row["northing"]):
            nearest_ids.append(None)
            nearest_dist_m.append(None)
            continue
        dists = rainfall_stations.apply(
            lambda r: _distance(row["easting"], row["northing"], r["easting"], r["northing"]), axis=1
        )
        idx = dists.idxmin()
        nearest_ids.append(rainfall_stations.loc[idx, "wiski_id"])
        nearest_dist_m.append(dists.loc[idx])
    level_stations = level_stations.copy()
    level_stations["nearest_rainfall_wiski_id"] = nearest_ids
    level_stations["nearest_rainfall_distance_m"] = nearest_dist_m
    return level_stations


def select_training_stations(bbox: dict | None = GREATER_LONDON_BBOX, max_stations: int | None = None) -> pd.DataFrame:
    live = fetch_live_level_stations(bbox=bbox)
    hydro = fetch_hydrology_stations()

    matched = live.merge(hydro[hydro["has_level_or_flow"]], on="wiski_id", how="inner")
    matched = matched.sort_values("catchment_name")
    if max_stations:
        matched = matched.head(max_stations)

    rainfall_stations = hydro[hydro["has_rainfall"]]
    selected = attach_nearest_rainfall_station(matched, rainfall_stations)
    return selected.reset_index(drop=True)


if __name__ == "__main__":
    stations = select_training_stations()
    stations.to_csv("data/stations.csv", index=False)
    print(f"selected {len(stations)} stations -> data/stations.csv")
