"""Loads the trained artifact once and turns live readings into a prediction.

Feature computation reuses src/features.py -- the exact same functions used
to build the training table -- so live and training features can't silently
drift apart from each other.
"""
from pathlib import Path

import joblib
import pandas as pd

from app import ea_client
from src import features
from src.explain import build_explainer, top_factors

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
STATIONS_PATH = ROOT / "data" / "stations.csv"
BASELINES_PATH = ROOT / "data" / "station_baselines.csv"

_state = {}


def load_model() -> dict:
    if not _state:
        bundle = joblib.load(MODEL_PATH)
        _state["bundle"] = bundle
        _state["explainer"] = build_explainer(bundle["model"])
        _state["stations"] = pd.read_csv(STATIONS_PATH).set_index("station_id")
        _state["baselines"] = pd.read_csv(BASELINES_PATH).set_index("station_id")["baseline"]
    return _state


def _readings_to_series(readings: list[dict]) -> pd.Series:
    if not readings:
        return pd.Series(dtype=float)
    df = pd.DataFrame(readings)
    df["dateTime"] = pd.to_datetime(df["dateTime"]).dt.tz_localize(None)
    s = df.set_index("dateTime")["value"].sort_index()
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="15min")
    return s.reindex(full_index).ffill(limit=4)


def compute_features(station_id: str, level_readings: list[dict], rainfall_readings: list[dict]) -> pd.DataFrame:
    state = load_model()
    if station_id not in state["stations"].index:
        raise ValueError(f"unknown station: {station_id}")
    baseline = state["baselines"].get(station_id)
    if baseline is None or pd.isna(baseline):
        raise ValueError(f"no cached baseline for station {station_id}")

    level = _readings_to_series(level_readings)
    rainfall = _readings_to_series(rainfall_readings).reindex(level.index).fillna(0.0)
    if level.empty or level.isna().all():
        raise ValueError(f"no usable live readings for station {station_id}")

    latest_time = level.dropna().index[-1]
    row = pd.DataFrame(
        {
            "rate_of_rise_1h": [features.rate_of_rise(level, hours=1).loc[latest_time]],
            "rate_of_rise_6h": [features.rate_of_rise(level, hours=6).loc[latest_time]],
            "rainfall_6h": [features.rolling_rainfall(rainfall, hours=6).loc[latest_time]],
            "rainfall_24h": [features.rolling_rainfall(rainfall, hours=24).loc[latest_time]],
            "rainfall_72h": [features.rolling_rainfall(rainfall, hours=72).loc[latest_time]],
            "level_minus_baseline": [level.loc[latest_time] - baseline],
        }
    )
    row = row.join(features.seasonality(pd.DatetimeIndex([latest_time])).reset_index(drop=True))
    row["station_id"] = pd.Categorical([station_id], categories=state["bundle"]["station_categories"])
    return row[state["bundle"]["feature_columns"]], latest_time


def predict(station_id: str) -> dict:
    state = load_model()
    if station_id not in state["stations"].index:
        raise ValueError(f"unknown station: {station_id}")
    station_row = state["stations"].loc[station_id]

    level_readings = ea_client.get_recent_readings(station_id, station_row["parameter"])
    rainfall_readings = ea_client.get_recent_readings(station_row["live_rainfall_station_id"], "rainfall")

    row, latest_time = compute_features(station_id, level_readings, rainfall_readings)
    bundle = state["bundle"]

    proba = float(bundle["model"].predict_proba(row)[:, 1][0])
    flagged = proba >= bundle["threshold"]
    explanation = top_factors(state["explainer"], row, bundle["feature_columns"])

    return {
        "station_id": station_id,
        "risk_score": proba,
        "elevated_risk": bool(flagged),
        "threshold": bundle["threshold"],
        "as_of": latest_time.isoformat(),
        "explanation": explanation,
    }
