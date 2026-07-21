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
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_CUTOFF = "2022-07-01"
VAL_CUTOFF = "2024-07-01"

FEATURE_COLUMNS = [
    "rate_of_rise_1h",
    "rate_of_rise_6h",
    "rainfall_6h",
    "rainfall_24h",
    "rainfall_72h",
    "level_minus_baseline",
    "season_sin",
    "season_cos",
]
TARGET_COLUMN = "label"


def load_training_table(path: str = "data/processed/training_table.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


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
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    return model


def evaluate(model, split: pd.DataFrame, threshold: float = 0.5, label: str = "") -> dict:
    y_true = split[TARGET_COLUMN]
    proba = model.predict_proba(split[FEATURE_COLUMNS])[:, 1]
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


if __name__ == "__main__":
    table = load_training_table()
    train, val, test = time_split(table)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"{name}: {len(split):,} rows, {split[TARGET_COLUMN].sum()} positive rows")

    baseline = train_baseline_model(train)
    evaluate(baseline, val, label="baseline logistic regression (validation)")
