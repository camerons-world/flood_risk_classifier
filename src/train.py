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
from pathlib import Path

import pandas as pd

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


if __name__ == "__main__":
    table = load_training_table()
    train, val, test = time_split(table)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"{name}: {len(split):,} rows, {split[TARGET_COLUMN].sum()} positive rows")
