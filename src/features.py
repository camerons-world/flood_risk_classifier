import pandas as pd


def rate_of_rise(level_series: pd.Series, window: int = 4) -> pd.Series:
    return level_series.diff(window) / window


def rolling_rainfall(rainfall_series: pd.Series, hours: int) -> pd.Series:
    periods = hours * 4  # 15-minute readings
    return rainfall_series.rolling(periods).sum()


def station_baseline(level_series: pd.Series) -> float:
    return level_series.median()
