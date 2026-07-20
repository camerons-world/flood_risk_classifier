"""Download historic 15-minute readings for the selected stations from the
EA Hydrology API and cache them locally as Parquet (data/raw/, gitignored).

One station's full 2010-onwards history comes back in a single request if
_limit is set high enough -- no need to paginate by year.
"""
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

HYDROLOGY_URL = "https://environment.data.gov.uk/hydrology/id/stations.json"
READINGS_URL = "https://environment.data.gov.uk/hydrology/id/measures/{measure_id}/readings"
TIMEOUT_SECONDS = 120
START_DATE = "2010-01-01"
RAW_DIR = Path("data/raw")
MAX_WORKERS = 8


def _instantaneous_measure_id(measures: list[dict], parameter: str) -> str | None:
    # rainfall gauges report 900s totals ("-t-900-"), not instantaneous points
    infix = "-t-900-" if parameter == "rainfall" else "-i-900-"
    for m in measures:
        measure_id = m.get("@id", "")
        if f"-{parameter}{infix}" in measure_id and "qualified" in measure_id:
            return measure_id.rsplit("/", 1)[-1]
    return None


def build_measure_lookup() -> dict:
    """wiski_id -> {'level': measure_id, 'flow': measure_id, 'rainfall': measure_id}"""
    r = requests.get(HYDROLOGY_URL, params={"_limit": 10000}, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    lookup = {}
    for s in r.json().get("items", []):
        wiski_ids = s.get("wiskiID")
        wiski_ids = wiski_ids if isinstance(wiski_ids, list) else [wiski_ids]
        measures = s.get("measures", [])
        entry = {
            "level": _instantaneous_measure_id(measures, "level"),
            "flow": _instantaneous_measure_id(measures, "flow"),
            "rainfall": _instantaneous_measure_id(measures, "rainfall"),
        }
        for wiski_id in wiski_ids:
            if wiski_id:
                lookup[wiski_id] = entry
    return lookup


def fetch_measure_readings(measure_id: str) -> pd.DataFrame:
    url = READINGS_URL.format(measure_id=measure_id)
    end_date = dt.date.today().isoformat()
    r = requests.get(
        url, params={"mineq-date": START_DATE, "maxeq-date": end_date, "_limit": 1_000_000}, timeout=TIMEOUT_SECONDS
    )
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        return pd.DataFrame(columns=["dateTime", "value", "quality"])
    df = pd.DataFrame(items)[["dateTime", "value", "quality"]]
    df["dateTime"] = pd.to_datetime(df["dateTime"])
    return df


def download_all(stations_csv: str = "data/stations.csv") -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(stations_csv)
    lookup = build_measure_lookup()

    jobs = {}  # output_path -> measure_id
    for _, row in stations.iterrows():
        entry = lookup.get(row["wiski_id"], {})
        parameter = "level" if entry.get("level") else "flow"
        measure_id = entry.get(parameter)
        if measure_id:
            jobs[RAW_DIR / f"{row['station_id']}_{parameter}.parquet"] = measure_id

    unique_rainfall_ids = stations["nearest_rainfall_wiski_id"].dropna().unique()
    for rainfall_wiski_id in unique_rainfall_ids:
        entry = lookup.get(rainfall_wiski_id, {})
        measure_id = entry.get("rainfall")
        if measure_id:
            jobs[RAW_DIR / f"rain_{rainfall_wiski_id}.parquet"] = measure_id

    print(f"{len(jobs)} measures to download")

    def _run(path_and_measure):
        path, measure_id = path_and_measure
        if path.exists():
            return path, "skipped (cached)"
        for attempt in range(3):
            try:
                df = fetch_measure_readings(measure_id)
                df.to_parquet(path, index=False)
                return path, f"{len(df)} rows"
            except Exception as exc:
                if attempt == 2:
                    return path, f"FAILED: {exc}"
        return path, "FAILED: retries exhausted"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_run, item) for item in jobs.items()]
        for i, future in enumerate(as_completed(futures), 1):
            path, status = future.result()
            print(f"[{i}/{len(jobs)}] {path.name}: {status}")


if __name__ == "__main__":
    download_all()
