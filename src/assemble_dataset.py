"""Build the labeled training table: one row per station per hour, with
engineered features and a binary "Flood Warning within next 24h" label.

Rolling features (rate-of-rise, rainfall accumulation, baseline) are computed
at the native ~15-minute reading cadence so short-lived spikes aren't
smoothed away, then the result is downsampled to on-the-hour snapshots --
that keeps the final table a manageable size without losing the signal those
rolling windows were built to capture.
"""
from pathlib import Path

import pandas as pd

from src import features
from src.flood_warnings import WARNING_TYPES

RAW_DIR = Path("data/raw")
LABEL_WINDOW_HOURS = 24
MAX_GAP_FILL_PERIODS = 4  # forward-fill gaps up to 1 hour; longer gaps stay NaN


def _load_series(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    df = df.dropna(subset=["value"]).drop_duplicates(subset=["dateTime"])
    s = df.set_index("dateTime")["value"].sort_index()
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="15min")
    return s.reindex(full_index).ffill(limit=MAX_GAP_FILL_PERIODS)


def _find_readings_path(station_id: str) -> tuple[Path, str] | None:
    for parameter in ("level", "flow"):
        path = RAW_DIR / f"{station_id}_{parameter}.parquet"
        if path.exists():
            return path, parameter
    return None


def _labels_for_station(index: pd.DatetimeIndex, station_warnings: pd.DataFrame) -> pd.Series:
    events = pd.to_datetime(station_warnings["date_time"]).sort_values()
    if events.empty:
        return pd.Series(0, index=index, dtype="int8")

    readings = pd.DataFrame({"dateTime": index}).sort_values("dateTime")
    events_df = pd.DataFrame({"dateTime": events, "event_time": events})
    merged = pd.merge_asof(readings, events_df, on="dateTime", direction="forward")
    within_window = (merged["event_time"] - merged["dateTime"]) <= pd.Timedelta(hours=LABEL_WINDOW_HOURS)
    label = within_window.fillna(False).astype("int8")
    label.index = merged["dateTime"].values
    return label.reindex(index)


def build_station_table(row: pd.Series, warnings: pd.DataFrame) -> pd.DataFrame | None:
    found = _find_readings_path(row["station_id"])
    if found is None:
        return None
    level_path, parameter = found
    level = _load_series(level_path)

    rainfall_path = RAW_DIR / f"rain_{row['nearest_rainfall_wiski_id']}.parquet"
    rainfall = _load_series(rainfall_path) if rainfall_path.exists() else pd.Series(dtype=float)
    rainfall = rainfall.reindex(level.index).fillna(0.0)

    df = pd.DataFrame(index=level.index)
    df["value"] = level
    df["rate_of_rise_1h"] = features.rate_of_rise(level, hours=1)
    df["rate_of_rise_6h"] = features.rate_of_rise(level, hours=6)
    df["rainfall_6h"] = features.rolling_rainfall(rainfall, hours=6)
    df["rainfall_24h"] = features.rolling_rainfall(rainfall, hours=24)
    df["rainfall_72h"] = features.rolling_rainfall(rainfall, hours=72)
    df["baseline"] = features.rolling_baseline(level)
    df["level_minus_baseline"] = df["value"] - df["baseline"]
    df = df.join(features.seasonality(df.index))

    station_warnings = warnings[
        (warnings["station_id"] == row["station_id"]) & (warnings["warning_type"].isin(WARNING_TYPES))
    ]
    df["label"] = _labels_for_station(df.index, station_warnings)

    df["station_id"] = row["station_id"]
    df["parameter"] = parameter
    df = df[df.index.minute == 0]  # downsample to hourly snapshots
    df = df.dropna(subset=["baseline", "rate_of_rise_1h", "rate_of_rise_6h"])
    return df.reset_index(names="date_time")


def assemble_training_table(stations_csv: str = "data/stations.csv", warnings_csv: str = "data/warnings_matched.csv") -> pd.DataFrame:
    stations = pd.read_csv(stations_csv)
    warnings = pd.read_csv(warnings_csv)

    tables = []
    for _, row in stations.iterrows():
        table = build_station_table(row, warnings)
        if table is not None:
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


if __name__ == "__main__":
    out_path = Path("data/processed")
    out_path.mkdir(parents=True, exist_ok=True)
    table = assemble_training_table()
    table.to_parquet(out_path / "training_table.parquet", index=False)
    print(f"{len(table):,} rows across {table['station_id'].nunique()} stations")
    print(f"positive rate: {table['label'].mean():.4%}")
