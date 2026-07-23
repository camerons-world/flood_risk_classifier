"""Loads the trained artifact once and turns live or historical readings
into a prediction.

Feature computation reuses src/features.py -- the exact same functions used
to build the training table -- so live/backtest and training features can't
silently drift apart from each other.
"""
import datetime as dt
from pathlib import Path

import joblib
import pandas as pd

from app import ea_client, hydrology_client, postcode_client
from src import features
from src.explain import build_explainer, top_factors
from src.geo import haversine_km

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
STATIONS_PATH = ROOT / "data" / "stations.csv"
BASELINES_PATH = ROOT / "data" / "station_baselines.csv"

MAX_STATION_DISTANCE_KM = 15.0
NEARBY_STATION_COUNT = 3  # primary + 2 for context
BACKTEST_WINDOW_DAYS = 97  # 90 for baseline + 7 for the rolling rainfall window
BACKTEST_MIN_DATE = dt.date(2010, 4, 1)  # first date with a real 90-day trailing window

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
    df["dateTime"] = pd.to_datetime(df["dateTime"])
    if df["dateTime"].dt.tz is not None:
        df["dateTime"] = df["dateTime"].dt.tz_localize(None)  # live EA readings are tz-aware; Hydrology's aren't
    s = df.set_index("dateTime")["value"].sort_index()
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="15min")
    return s.reindex(full_index).ffill(limit=4)


def compute_features(
    station_id: str, level_readings: list[dict], rainfall_readings: list[dict], baseline: float | None = None
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """If baseline isn't given, it's derived from the level series itself --
    only possible when enough trailing history was fetched (the backtest
    path always does this; the live "now" path can't, since the live API
    only serves ~29 days, so it passes the cached baseline instead)."""
    state = load_model()
    if station_id not in state["stations"].index:
        raise ValueError(f"unknown station: {station_id}")

    level = _readings_to_series(level_readings)
    rainfall = _readings_to_series(rainfall_readings).reindex(level.index).fillna(0.0)
    if level.empty or level.isna().all():
        raise ValueError(f"no usable readings for station {station_id}")

    latest_time = level.dropna().index[-1]

    if baseline is None:
        baseline = features.rolling_baseline(level).loc[latest_time]
    if baseline is None or pd.isna(baseline):
        raise ValueError(f"could not determine a baseline for station {station_id}")

    row = pd.DataFrame(
        {
            "rate_of_rise_1h": [features.rate_of_rise(level, hours=1).loc[latest_time]],
            "rate_of_rise_6h": [features.rate_of_rise(level, hours=6).loc[latest_time]],
            "rate_of_rise_12h": [features.rate_of_rise(level, hours=12).loc[latest_time]],
            "rainfall_6h": [features.rolling_rainfall(rainfall, hours=6).loc[latest_time]],
            "rainfall_24h": [features.rolling_rainfall(rainfall, hours=24).loc[latest_time]],
            "rainfall_72h": [features.rolling_rainfall(rainfall, hours=72).loc[latest_time]],
            "rainfall_7d": [features.rolling_rainfall(rainfall, hours=168).loc[latest_time]],
            "rainfall_peak_intensity": [features.peak_rainfall_intensity(rainfall).loc[latest_time]],
            "level_minus_baseline": [level.loc[latest_time] - baseline],
            "level_vs_24h_peak": [level.loc[latest_time] - features.rolling_max(level, hours=24).loc[latest_time]],
        }
    )
    row = row.join(features.seasonality(pd.DatetimeIndex([latest_time])).reset_index(drop=True))
    row["station_id"] = pd.Categorical([station_id], categories=state["bundle"]["station_categories"])
    return row[state["bundle"]["feature_columns"]], latest_time


def list_stations() -> list[dict]:
    state = load_model()
    stations = state["stations"].reset_index()[["station_id", "label", "catchment_name"]]
    return stations.to_dict(orient="records")


def nearest_stations(lat: float, long: float, k: int = NEARBY_STATION_COUNT, max_km: float = MAX_STATION_DISTANCE_KM) -> list[dict]:
    state = load_model()
    stations = state["stations"].reset_index()
    stations = stations.assign(distance_km=stations.apply(lambda s: haversine_km(lat, long, s["lat"], s["long"]), axis=1))
    nearby = stations[stations["distance_km"] <= max_km].nsmallest(k, "distance_km")
    return nearby[["station_id", "label", "distance_km"]].to_dict(orient="records")


def resolve_station(station_id: str | None, postcode: str | None) -> tuple[str, list[dict] | None]:
    """Returns (station_id, nearby_stations). nearby_stations is None when a
    station_id was given directly (no postcode resolution happened)."""
    if bool(station_id) == bool(postcode):
        raise ValueError("provide exactly one of station_id or postcode")

    if station_id:
        state = load_model()
        if station_id not in state["stations"].index:
            raise ValueError(f"unknown station: {station_id}")
        return station_id, None

    lat, long = postcode_client.geocode(postcode)
    nearby = nearest_stations(lat, long)
    if not nearby:
        raise ValueError(f"no monitored station within {MAX_STATION_DISTANCE_KM:.0f}km of postcode {postcode}")
    return nearby[0]["station_id"], nearby


def _predict_live(station_id: str) -> dict:
    state = load_model()
    station_row = state["stations"].loc[station_id]
    baseline = state["baselines"].get(station_id)
    if baseline is None or pd.isna(baseline):
        raise ValueError(f"no cached baseline for station {station_id}")

    level_readings, level_stale = ea_client.get_recent_readings(station_id, station_row["parameter"])
    rainfall_readings, rainfall_stale = ea_client.get_recent_readings(station_row["live_rainfall_station_id"], "rainfall")

    row, latest_time = compute_features(station_id, level_readings, rainfall_readings, baseline=baseline)
    return _score(row, latest_time, stale=level_stale or rainfall_stale)


def _predict_backtest(station_id: str, date: dt.date) -> dict:
    """Callers (app/main.py) are expected to validate the date range up front --
    this trusts it's already within [BACKTEST_MIN_DATE, yesterday]."""
    state = load_model()
    station_row = state["stations"].loc[station_id]
    start_date = date - dt.timedelta(days=BACKTEST_WINDOW_DAYS)

    level_measure = hydrology_client.get_measure_id(station_row["wiski_id"], station_row["parameter"])
    level_readings = hydrology_client.get_readings(level_measure, start_date, date)

    rainfall_measure = hydrology_client.get_measure_id(station_row["nearest_rainfall_wiski_id"], "rainfall")
    rainfall_readings = hydrology_client.get_readings(rainfall_measure, start_date, date)

    row, latest_time = compute_features(station_id, level_readings, rainfall_readings)
    result = _score(row, latest_time, stale=False)
    result["mode"] = "backtest"
    return result


def _score(row: pd.DataFrame, latest_time: pd.Timestamp, stale: bool) -> dict:
    state = load_model()
    bundle = state["bundle"]
    proba = float(bundle["model"].predict_proba(row)[:, 1][0])
    flagged = proba >= bundle["threshold"]
    explanation = top_factors(state["explainer"], row, bundle["feature_columns"])

    return {
        "risk_score": proba,
        "elevated_risk": bool(flagged),
        "threshold": bundle["threshold"],
        "as_of": latest_time.isoformat(),
        "stale": stale,
        "explanation": explanation,
    }


def predict(station_id: str | None = None, postcode: str | None = None, date: dt.date | None = None) -> dict:
    resolved_id, nearby = resolve_station(station_id, postcode)

    result = _predict_backtest(resolved_id, date) if date else _predict_live(resolved_id)
    result["station_id"] = resolved_id

    if nearby is not None:
        primary, others = nearby[0], nearby[1:]
        result["resolved_station"] = {"station_id": primary["station_id"], "label": primary["label"], "distance_km": primary["distance_km"]}
        result["nearby_stations"] = [_nearby_summary(s, date) for s in others]

    return result


def _nearby_summary(station: dict, date: dt.date | None) -> dict:
    try:
        prediction = _predict_backtest(station["station_id"], date) if date else _predict_live(station["station_id"])
        return {
            "station_id": station["station_id"],
            "label": station["label"],
            "distance_km": station["distance_km"],
            "risk_score": prediction["risk_score"],
            "elevated_risk": prediction["elevated_risk"],
        }
    except Exception:
        return {"station_id": station["station_id"], "label": station["label"], "distance_km": station["distance_km"], "risk_score": None, "elevated_risk": None}
