"""Per-prediction SHAP explanations for the main model.

Builds the TreeExplainer once (construction parses the whole tree ensemble,
not cheap to redo per request) and reuses it for every prediction -- callers
should build_explainer() once at process start, not per call.

Output is translated into plain language here (not just raw SHAP floats)
since this is what the frontend and API response show directly to a reader --
friendly labels, a human month instead of raw sin/cos, and a direction
instead of expecting readers to know SHAP values are signed log-odds.
"""
import math
from datetime import date, timedelta

import shap
import pandas as pd

TOP_N = 4

FEATURE_LABELS = {
    "rate_of_rise_1h": "River level rising (last 1h)",
    "rate_of_rise_6h": "River level rising (last 6h)",
    "rate_of_rise_12h": "River level rising (last 12h)",
    "rainfall_6h": "Rainfall (last 6h)",
    "rainfall_24h": "Rainfall (last 24h)",
    "rainfall_72h": "Rainfall (last 72h)",
    "rainfall_7d": "Rainfall (last 7 days)",
    "rainfall_peak_intensity": "Heaviest rainfall burst (last 6h)",
    "level_minus_baseline": "River level vs. normal",
    "level_vs_24h_peak": "River level vs. recent peak (24h)",
    "station_id": "This station's typical risk",
}
RAINFALL_FEATURES = {"rainfall_6h", "rainfall_24h", "rainfall_72h", "rainfall_7d", "rainfall_peak_intensity"}
SEASON_FEATURES = ("season_sin", "season_cos")


def build_explainer(model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def _to_native(value):
    return value.item() if hasattr(value, "item") else str(value)


def _month_name(sin_value: float, cos_value: float) -> str:
    angle = math.atan2(sin_value, cos_value)
    day_of_year = (angle / (2 * math.pi) * 365.25) % 365.25
    return (date(2001, 1, 1) + timedelta(days=day_of_year)).strftime("%B")


def _friendly_value(feature: str, value) -> str:
    if feature in RAINFALL_FEATURES:
        return f"{value:.1f}mm"
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def _factor(feature: str, label: str, value, shap_contribution: float) -> dict:
    return {
        "feature": feature,
        "label": label,
        "value": value,
        "shap_contribution": shap_contribution,
        "direction": "increases risk" if shap_contribution > 0 else "decreases risk",
    }


def top_factors(explainer: shap.TreeExplainer, row: pd.DataFrame, feature_columns: list[str], top_n: int = TOP_N) -> list[dict]:
    """row must have exactly one row, columns matching feature_columns (station_id included).

    season_sin/season_cos are combined into one "Time of year" factor (their
    SHAP contributions summed, their sin/cos pair converted to a month name)
    rather than shown as two separately-labeled trig components.
    """
    shap_values = explainer(row[feature_columns]).values[0]
    values = row[feature_columns].iloc[0]

    contributions = []
    season_shap = 0.0
    season_seen = False
    for i, feature in enumerate(feature_columns):
        if feature in SEASON_FEATURES:
            season_shap += float(shap_values[i])
            season_seen = True
            continue
        value = _friendly_value(feature, _to_native(values[feature]))
        contributions.append(_factor(feature, FEATURE_LABELS.get(feature, feature), value, float(shap_values[i])))

    if season_seen:
        month = _month_name(values["season_sin"], values["season_cos"])
        contributions.append(_factor("season", "Time of year", month, season_shap))

    contributions.sort(key=lambda c: abs(c["shap_contribution"]), reverse=True)
    return contributions[:top_n]
