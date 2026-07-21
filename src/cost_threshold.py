"""Cost-based operating threshold selection.

The label is per station-hour ("a warning will fire within the next 24h"),
so both a real flood event and a sustained false alarm show up as a *run* of
consecutive rows, not a single row. Costing either per row would charge the
same event/false-signal cost once for every hour it spans -- overstating
both sides by however many hours the run happens to cover. Instead, both
sides are costed per contiguous episode:
- A missed event (a run of label==1 for a station with no flagged row in it)
  costs once, however many rows it spans. These are naturally bounded to
  <=24 rows by how the label itself is constructed (assemble_dataset.py), so
  no separate cap is needed here.
- A false alarm (a run of flagged==True for a station that never overlaps a
  real label==1 row) also costs once per FALSE_ALARM_MAX_HOURS block, not
  once per run -- a run has no natural bound the way a label run does. An
  uncapped version of this was tried and gamed by a model that, once given
  station_id as a feature, just stayed flagged for one station across nearly
  the entire test period (one run of ~18,000 hours) for a single GBP 100
  charge, which is not how alert fatigue actually works. Capping at 24h
  forces a persistently-wrong station to keep incurring cost, using the same
  natural time unit as the label's own lookback window.

Cost values (see README for the full writeup):
- COST_MISSED_WARNING: ABI's reported 2025 average UK domestic flood claim
  (~GBP 30,000, up 60% YoY) -- a real, sourced figure.
- COST_FALSE_ALARM: no authoritative figure exists for this (it's internal
  ops cost, never published) -- GBP 100 is a stated assumption, roughly
  1-2 hours of an underwriting analyst reviewing a flagged station.

Threshold selection: pure cost minimization picked a threshold that only
caught ~27% of held-out flood episodes -- mathematically optimal given the
cost values, but too conservative to be a useful underwriting signal in
practice. MIN_EPISODE_RECALL reframes this as a constrained optimization
(cheapest threshold that still catches at least half of all episodes), which
is a common real-world framing -- insurers often carry a minimum-recall SLA
alongside a cost target, not a pure cost objective.
"""
import numpy as np
import pandas as pd

COST_MISSED_WARNING = 30_000
COST_FALSE_ALARM = 100
FALSE_ALARM_MAX_HOURS = 24
MIN_EPISODE_RECALL = 0.5


def _contiguous_run_ids(df: pd.DataFrame, flag_col: str, max_run_length: int | None = None) -> pd.Series:
    """Contiguous True run of flag_col within a station -> one run id; False rows get NaN.
    With max_run_length set, a run longer than that gets split into successive sub-runs."""
    is_true = df[flag_col]
    same_station = df["station_id"] == df["station_id"].shift(1)
    starts_new_run = is_true & ~(is_true.shift(1, fill_value=False) & same_station)
    run_id = starts_new_run.cumsum()
    run_id = run_id.where(is_true)

    if max_run_length is not None:
        position_in_run = df.groupby(run_id)[flag_col].cumcount()
        run_id = run_id * 1_000_000 + (position_in_run // max_run_length)

    return run_id


def evaluate_threshold(df: pd.DataFrame, proba_col: str, threshold: float) -> dict:
    """df needs columns: station_id, date_time, label, and `proba_col`. Must already be
    sorted by (station_id, date_time) -- callers should sort once rather than per threshold."""
    df = df.copy()
    df["flagged"] = df[proba_col] >= threshold

    event_run_id = _contiguous_run_ids(df.assign(is_label=df["label"].astype(bool)), "is_label")
    event_flagged = df.loc[event_run_id.notna()].groupby(event_run_id[event_run_id.notna()])["flagged"].any()
    missed_event_count = (~event_flagged).sum()
    episode_recall = float(event_flagged.mean()) if len(event_flagged) else float("nan")

    alarm_run_id = _contiguous_run_ids(df, "flagged", max_run_length=FALSE_ALARM_MAX_HOURS)
    alarm_has_true_positive = df.loc[alarm_run_id.notna()].groupby(alarm_run_id[alarm_run_id.notna()])["label"].max()
    false_alarm_count = (alarm_has_true_positive == 0).sum()

    cost = missed_event_count * COST_MISSED_WARNING + false_alarm_count * COST_FALSE_ALARM
    return {"expected_cost": cost, "episode_recall": episode_recall, "episodes_caught": int(event_flagged.sum()), "episodes_total": len(event_flagged)}


def expected_cost(df: pd.DataFrame, proba_col: str, threshold: float) -> float:
    return evaluate_threshold(df, proba_col, threshold)["expected_cost"]


def select_threshold(df: pd.DataFrame, proba_col: str, thresholds=None, min_episode_recall: float | None = MIN_EPISODE_RECALL) -> dict:
    """Cheapest threshold with episode_recall >= min_episode_recall, if set;
    otherwise pure cost minimization (min_episode_recall=None)."""
    df = df.sort_values(["station_id", "date_time"]).reset_index(drop=True)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = [{"threshold": t, **evaluate_threshold(df, proba_col, t)} for t in thresholds]
    curve = pd.DataFrame(rows)

    candidates = curve[curve["episode_recall"] >= min_episode_recall] if min_episode_recall else curve
    if candidates.empty:
        candidates = curve  # constraint unreachable at any threshold in range -- fall back to pure cost min

    best = candidates.loc[candidates["expected_cost"].idxmin()]
    return {
        "best_threshold": float(best["threshold"]),
        "expected_cost": float(best["expected_cost"]),
        "episode_recall": float(best["episode_recall"]),
        "curve": curve,
    }
