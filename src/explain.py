"""Per-prediction SHAP explanations for the main model.

Builds the TreeExplainer once (construction parses the whole tree ensemble,
not cheap to redo per request) and reuses it for every prediction -- callers
should build_explainer() once at process start, not per call.
"""
import shap
import pandas as pd

TOP_N = 4


def build_explainer(model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def _to_native(value):
    return value.item() if hasattr(value, "item") else str(value)


def top_factors(explainer: shap.TreeExplainer, row: pd.DataFrame, feature_columns: list[str], top_n: int = TOP_N) -> list[dict]:
    """row must have exactly one row, columns matching feature_columns (station_id included)."""
    shap_values = explainer(row[feature_columns]).values[0]
    values = row[feature_columns].iloc[0]

    contributions = [
        {"feature": feature, "value": _to_native(values[feature]), "shap_contribution": float(shap_values[i])}
        for i, feature in enumerate(feature_columns)
    ]
    contributions.sort(key=lambda c: abs(c["shap_contribution"]), reverse=True)
    return contributions[:top_n]
