"""Cost-based operating threshold selection.

The label is per station-hour ("a warning will fire within the next 24h"),
so a single real flood event shows up as a run of consecutive positive rows
-- not one row. Costing a missed warning per row would charge the same
event's cost once for every hour in that run (up to ~24x overstatement).
Instead: a missed *event* (a contiguous run of label==1 for a station) costs
once, however many rows it spans; a false alarm costs once per row, since
each spurious flag really does waste someone's time separately.

Cost values (see README for the full writeup):
- COST_MISSED_WARNING: ABI's reported 2025 average UK domestic flood claim
  (~GBP 30,000, up 60% YoY) -- a real, sourced figure.
- COST_FALSE_ALARM: no authoritative figure exists for this (it's internal
  ops cost, never published) -- GBP 100 is a stated assumption, roughly
  1-2 hours of an underwriting analyst reviewing a flagged station.
"""
import numpy as np
import pandas as pd

COST_MISSED_WARNING = 30_000
COST_FALSE_ALARM = 100


def _episode_ids(df: pd.DataFrame) -> pd.Series:
    """Contiguous run of label==1 within a station -> one episode id; label==0 rows get NaN."""
    df = df.sort_values(["station_id", "date_time"])
    is_positive = df["label"] == 1
    same_station = df["station_id"] == df["station_id"].shift(1)
    starts_new_episode = is_positive & ~(is_positive.shift(1, fill_value=False) & same_station)
    episode_id = starts_new_episode.cumsum()
    return episode_id.where(is_positive)


def expected_cost(df: pd.DataFrame, proba_col: str, threshold: float) -> float:
    """df needs columns: station_id, date_time, label, and `proba_col`."""
    df = df.copy()
    df["episode_id"] = _episode_ids(df)
    flagged = df[proba_col] >= threshold

    false_alarm_count = ((df["label"] == 0) & flagged).sum()

    episode_flagged = df.loc[df["episode_id"].notna()].groupby("episode_id").apply(
        lambda g: flagged.loc[g.index].any(), include_groups=False
    )
    missed_event_count = (~episode_flagged).sum()

    return missed_event_count * COST_MISSED_WARNING + false_alarm_count * COST_FALSE_ALARM


def select_threshold(df: pd.DataFrame, proba_col: str, thresholds=None) -> dict:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    curve = pd.DataFrame(
        {"threshold": t, "expected_cost": expected_cost(df, proba_col, t)} for t in thresholds
    )
    best = curve.loc[curve["expected_cost"].idxmin()]
    return {"best_threshold": float(best["threshold"]), "expected_cost": float(best["expected_cost"]), "curve": curve}
