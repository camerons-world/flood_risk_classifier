import pandas as pd
import pytest

from src.cost_threshold import (
    COST_FALSE_ALARM,
    COST_MISSED_WARNING,
    MIN_EPISODE_RECALL,
    evaluate_threshold,
    select_threshold,
)


def _frame(*stations):
    """Each station is (station_id, [(label, proba), ...]) -> hourly rows, sorted."""
    frames = []
    for station_id, rows in stations:
        frames.append(
            pd.DataFrame(
                {
                    "station_id": station_id,
                    "date_time": pd.date_range("2024-01-01", periods=len(rows), freq="h"),
                    "label": [label for label, _ in rows],
                    "proba": [proba for _, proba in rows],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_multi_row_missed_episode_costs_once():
    # 5-hour flood episode, never flagged: one missed-warning charge, not five.
    df = _frame(("A", [(0, 0.0)] * 3 + [(1, 0.0)] * 5 + [(0, 0.0)] * 3))

    result = evaluate_threshold(df, "proba", 0.5)

    assert result["expected_cost"] == COST_MISSED_WARNING
    assert result["episodes_total"] == 1
    assert result["episodes_caught"] == 0


def test_episode_caught_by_any_row_costs_nothing():
    df = _frame(("A", [(1, 0.0), (1, 0.9), (1, 0.0)]))

    result = evaluate_threshold(df, "proba", 0.5)

    assert result["expected_cost"] == 0
    assert result["episode_recall"] == 1.0


def test_48h_false_alarm_run_costs_two_false_alarms():
    df = _frame(("A", [(0, 0.9)] * 48))

    result = evaluate_threshold(df, "proba", 0.5)

    assert result["expected_cost"] == 2 * COST_FALSE_ALARM


def test_false_alarm_runs_do_not_join_across_stations():
    # Flagged rows end station A and start station B: two runs, not one.
    df = _frame(
        ("A", [(0, 0.0), (0, 0.9), (0, 0.9)]),
        ("B", [(0, 0.9), (0, 0.9), (0, 0.0)]),
    )

    result = evaluate_threshold(df, "proba", 0.5)

    assert result["expected_cost"] == 2 * COST_FALSE_ALARM


def test_event_runs_do_not_join_across_stations():
    # Label==1 rows end station A and start station B: two episodes, both missed.
    df = _frame(
        ("A", [(0, 0.0), (1, 0.0), (1, 0.0)]),
        ("B", [(1, 0.0), (1, 0.0), (0, 0.0)]),
    )

    result = evaluate_threshold(df, "proba", 0.5)

    assert result["episodes_total"] == 2
    assert result["expected_cost"] == 2 * COST_MISSED_WARNING


def test_select_threshold_falls_back_to_cost_min_when_recall_floor_unreachable():
    # Three episodes, only one ever scores above 0: best possible recall is 1/3.
    episode_hit = [(1, 0.9)] * 2
    episode_miss = [(1, 0.0)] * 2
    gap = [(0, 0.0)]
    df = _frame(("A", episode_hit + gap + episode_miss + gap + episode_miss))
    thresholds = [0.5, 0.95]

    result = select_threshold(df, "proba", thresholds=thresholds)

    assert result["episode_recall"] == pytest.approx(1 / 3)
    assert result["episode_recall"] < MIN_EPISODE_RECALL
    # Fallback picks the cheapest threshold (catches one episode), same as pure cost-min.
    assert result["best_threshold"] == 0.5
    assert result["expected_cost"] == 2 * COST_MISSED_WARNING
    pure = select_threshold(df, "proba", thresholds=thresholds, min_episode_recall=None)
    assert pure["best_threshold"] == result["best_threshold"]


def test_select_threshold_respects_recall_floor_when_reachable():
    # Catching the episode costs a false alarm, but the 50% recall floor forces it.
    df = _frame(("A", [(1, 0.6), (0, 0.0)] + [(0, 0.3)] + [(0, 0.0)] + [(1, 0.0)] * 2))
    thresholds = [0.2, 0.5, 0.9]

    result = select_threshold(df, "proba", thresholds=thresholds)

    assert result["episode_recall"] >= MIN_EPISODE_RECALL
