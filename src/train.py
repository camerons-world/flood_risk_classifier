"""Time-based train/val/test split, model training, and evaluation.

Three-way, not two-way: the cost-based threshold (src/cost_threshold.py) gets
picked on the validation split, then applied once to the untouched test
split for final reporting. Picking the threshold and reporting on the same
data would bias the reported numbers optimistically.

Cutoffs are global (same date for every station), not per-station -- simpler
and unambiguous. Chosen by checking positive-episode counts per candidate
window (flood-warning episodes are lumpy across years -- 2024 alone had 31
of the dataset's 197 total), not just picking round fractions of rows:
train < 2022-07-01 (148 episodes), val 2022-07-01..2024-07-01 (34 episodes,
picks the threshold), test >= 2024-07-01 (15 episodes, final numbers only).

The main model includes station_id as a categorical feature; the baseline
doesn't (kept simple -- one-hot-encoding 132 stations into a linear model
changes its character as a "plain" baseline). Adding station_id to the main
model is a real, validated improvement (2.3x val PR-AUC), not leakage for
this project's scope: the app only ever serves these same 132 known
stations, so learning per-station baseline risk is fair game. It was tried
once already with an uncapped false-alarm cost function and gamed the
result (the model just stayed flagged on one station for ~18,000 hours
straight, which the old costing only charged once) -- see cost_threshold.py
for the fix (FALSE_ALARM_MAX_HOURS) before trusting numbers here.

Hyperparameters below came from a short validation-only sweep (never
touching test) comparing ~7 configs on validation cost; this one had the
lowest. Diminishing returns set in quickly given only ~180 positive
episodes total to learn from -- not worth chasing further.
"""
import datetime as dt
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.cost_threshold import COST_FALSE_ALARM, COST_MISSED_WARNING, expected_cost, select_threshold

MODEL_PATH = Path("models/model.joblib")

TRAIN_CUTOFF = "2022-07-01"
VAL_CUTOFF = "2024-07-01"

BASELINE_FEATURE_COLUMNS = [
    "rate_of_rise_1h",
    "rate_of_rise_6h",
    "rainfall_6h",
    "rainfall_24h",
    "rainfall_72h",
    "level_minus_baseline",
    "season_sin",
    "season_cos",
]
MAIN_MODEL_FEATURE_COLUMNS = BASELINE_FEATURE_COLUMNS + ["station_id"]
TARGET_COLUMN = "label"


def load_training_table(path: str = "data/processed/training_table.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    station_dtype = pd.CategoricalDtype(categories=sorted(df["station_id"].unique()))
    df["station_id"] = df["station_id"].astype(station_dtype)
    return df


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["date_time"] < TRAIN_CUTOFF]
    val = df[(df["date_time"] >= TRAIN_CUTOFF) & (df["date_time"] < VAL_CUTOFF)]
    test = df[df["date_time"] >= VAL_CUTOFF]
    return train, val, test


def train_baseline_model(train: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )
    model.fit(train[BASELINE_FEATURE_COLUMNS], train[TARGET_COLUMN])
    return model


def train_main_model(train: pd.DataFrame, val: pd.DataFrame) -> XGBClassifier:
    neg, pos = (train[TARGET_COLUMN] == 0).sum(), (train[TARGET_COLUMN] == 1).sum()
    model = XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        reg_lambda=1.0,
        min_child_weight=10,
        scale_pos_weight=neg / pos,
        eval_metric="aucpr",
        early_stopping_rounds=25,
    )
    model.fit(
        train[MAIN_MODEL_FEATURE_COLUMNS],
        train[TARGET_COLUMN],
        eval_set=[(val[MAIN_MODEL_FEATURE_COLUMNS], val[TARGET_COLUMN])],
        verbose=False,
    )
    return model


def evaluate(model, split: pd.DataFrame, features: list[str], threshold: float = 0.5, label: str = "") -> dict:
    y_true = split[TARGET_COLUMN]
    proba = model.predict_proba(split[features])[:, 1]
    y_pred = (proba >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": fbeta_score(y_true, y_pred, beta=1, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "pr_auc": average_precision_score(y_true, proba),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"--- {label} (threshold={threshold}) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  confusion matrix: tn={tn} fp={fp} fn={fn} tp={tp}")
    return metrics


def _with_proba(model, split: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = split[["station_id", "date_time", TARGET_COLUMN]].rename(columns={TARGET_COLUMN: "label"}).copy()
    out["station_id"] = out["station_id"].astype(str)
    out["proba"] = model.predict_proba(split[features])[:, 1]
    return out


def pick_operating_threshold(model, val: pd.DataFrame, features: list[str]) -> dict:
    """Cost-optimal threshold, chosen on validation -- never on test."""
    val_scored = _with_proba(model, val, features)
    result = select_threshold(val_scored, "proba")
    print(f"cost-optimal threshold (validation): {result['best_threshold']:.3f}, expected cost GBP{result['expected_cost']:,.0f}")
    return result


if __name__ == "__main__":
    table = load_training_table()
    train, val, test = time_split(table)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"{name}: {len(split):,} rows, {split[TARGET_COLUMN].sum()} positive rows")

    baseline = train_baseline_model(train)
    evaluate(baseline, val, BASELINE_FEATURE_COLUMNS, label="baseline logistic regression (validation)")

    main_model = train_main_model(train, val)
    evaluate(main_model, val, MAIN_MODEL_FEATURE_COLUMNS, label="main XGBoost model (validation)")

    threshold_result = pick_operating_threshold(main_model, val, MAIN_MODEL_FEATURE_COLUMNS)
    best_threshold = threshold_result["best_threshold"]

    evaluate(
        main_model, test, MAIN_MODEL_FEATURE_COLUMNS, threshold=best_threshold,
        label="main XGBoost model (TEST, cost-optimal threshold)",
    )
    test_scored = _with_proba(main_model, test, MAIN_MODEL_FEATURE_COLUMNS)
    test_cost = expected_cost(test_scored, "proba", best_threshold)
    test_pr_auc = average_precision_score(test[TARGET_COLUMN], test_scored["proba"])
    print(f"expected cost on test set at threshold {best_threshold:.3f}: GBP{test_cost:,.0f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": main_model,
        "threshold": best_threshold,
        "feature_columns": MAIN_MODEL_FEATURE_COLUMNS,
        "station_categories": list(table["station_id"].cat.categories),
        "metadata": {
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "train_window": f"{table['date_time'].min()} to {TRAIN_CUTOFF}",
            "cost_missed_warning_gbp": COST_MISSED_WARNING,
            "cost_false_alarm_gbp": COST_FALSE_ALARM,
            "test_pr_auc": float(test_pr_auc),
            "test_expected_cost_gbp": float(test_cost),
        },
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"saved model bundle -> {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.2f} MB)")
