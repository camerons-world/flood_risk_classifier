"""Load the Historic Flood Warnings dataset and match each warning to the
nearest selected station by lat/long, so we can build training labels.

This keeps every normalised event type (Alert, Warning, Severe Warning, and
their Update variants -- Updates still mark the area as actively under
warning, useful for a "was this area at elevated risk" label). Deciding
which severities count as the positive training label happens downstream,
in train.py, using the WARNING_TYPES constant below.
"""
import math

import pandas as pd
import requests

WARNINGS_ODS = "data/raw_warnings/flood_warnings_raw.ods"
FLOOD_AREAS_URL = "https://environment.data.gov.uk/flood-monitoring/id/floodAreas"
TIMEOUT_SECONDS = 60
MAX_MATCH_DISTANCE_KM = 5

TYPE_NORMALISE = {
    "flood alert": "Flood Alert",
    "flood watch": "Flood Alert",  # pre-2010 name for the same product
    "flood warning": "Flood Warning",
    "severe flood warning": "Severe Flood Warning",
    "update flood alert": "Update Flood Alert",
    "update flood warning": "Update Flood Warning",
    "update severe flood warning": "Update Severe Flood Warning",
}
WARNING_TYPES = {"Flood Warning", "Severe Flood Warning"}


def load_historic_warnings() -> pd.DataFrame:
    df = pd.read_excel(WARNINGS_ODS, sheet_name="tblAllData", engine="odf")
    df["TYPE"] = df["TYPE"].str.strip().str.lower().map(TYPE_NORMALISE)
    df = df.dropna(subset=["TYPE"])
    df = df.rename(columns={"DATE": "date_time", "CODE": "flood_area_code", "TYPE": "warning_type"})
    return df[["date_time", "flood_area_code", "warning_type"]]


def fetch_flood_areas() -> pd.DataFrame:
    r = requests.get(FLOOD_AREAS_URL, params={"_limit": 10000}, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    items = r.json().get("items", [])
    return pd.DataFrame(
        [{"flood_area_code": i.get("notation"), "fa_lat": i.get("lat"), "fa_long": i.get("long")} for i in items]
    )


def _haversine_km(lat1, long1, lat2, long2):
    r = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(long2 - long1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _nearest_station_lookup(flood_areas: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """One row per unique flood_area_code -- nearest station is fixed per area, not per event."""
    nearest_station_id = []
    nearest_km = []
    for _, row in flood_areas.iterrows():
        dists = stations.apply(lambda s: _haversine_km(row["fa_lat"], row["fa_long"], s["lat"], s["long"]), axis=1)
        idx = dists.idxmin()
        nearest_station_id.append(stations.loc[idx, "station_id"])
        nearest_km.append(dists.loc[idx])
    flood_areas = flood_areas.copy()
    flood_areas["station_id"] = nearest_station_id
    flood_areas["distance_km"] = nearest_km
    return flood_areas


def match_warnings_to_stations(
    warnings: pd.DataFrame, stations: pd.DataFrame, max_distance_km: float = MAX_MATCH_DISTANCE_KM
) -> pd.DataFrame:
    flood_areas = fetch_flood_areas().dropna(subset=["fa_lat", "fa_long"])
    relevant_codes = flood_areas[flood_areas["flood_area_code"].isin(warnings["flood_area_code"])]
    lookup = _nearest_station_lookup(relevant_codes, stations)
    lookup = lookup[lookup["distance_km"] <= max_distance_km]

    matched = warnings.merge(lookup[["flood_area_code", "station_id", "distance_km"]], on="flood_area_code", how="inner")
    return matched.reset_index(drop=True)


if __name__ == "__main__":
    stations = pd.read_csv("data/stations.csv")
    warnings = load_historic_warnings()
    print(f"{len(warnings)} warning-type events loaded (post Flood Watch/Alert normalisation)")

    matched = match_warnings_to_stations(warnings, stations)
    matched.to_csv("data/warnings_matched.csv", index=False)
    print(f"{len(matched)} events matched to a selected station within {MAX_MATCH_DISTANCE_KM}km -> data/warnings_matched.csv")
    print(matched["warning_type"].value_counts())
