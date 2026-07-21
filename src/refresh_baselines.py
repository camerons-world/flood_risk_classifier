"""Snapshot each station's current 90-day rolling baseline to a small CSV.

Exists because the live flood-monitoring API only serves ~29 days of
history per station (confirmed by direct query) -- not enough to compute
the 90-day baseline live. This reuses the cached Hydrology API data and the
same features.rolling_baseline() used at training time, then ships the
result as a static, periodically-refreshed snapshot rather than a true live
recompute. Re-run this script to refresh it; the deployed app just reads
whatever's in data/station_baselines.csv.
"""
import datetime as dt

import pandas as pd

from src import features
from src.assemble_dataset import _find_readings_path, _load_series

OUT_PATH = "data/station_baselines.csv"


def compute_current_baseline(station_id: str) -> float | None:
    found = _find_readings_path(station_id)
    if found is None:
        return None
    path, _parameter = found
    series = _load_series(path)
    baseline_series = features.rolling_baseline(series).dropna()
    return float(baseline_series.iloc[-1]) if not baseline_series.empty else None


def refresh_all(stations_csv: str = "data/stations.csv") -> pd.DataFrame:
    stations = pd.read_csv(stations_csv)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = [
        {"station_id": row["station_id"], "baseline": compute_current_baseline(row["station_id"]), "computed_at": now}
        for _, row in stations.iterrows()
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = refresh_all()
    result.to_csv(OUT_PATH, index=False)
    print(f"{result['baseline'].notna().sum()}/{len(result)} stations have a baseline -> {OUT_PATH}")
