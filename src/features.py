import numpy as np
import pandas as pd

READINGS_PER_HOUR = 4  # 15-minute cadence


def rate_of_rise(level_series: pd.Series, hours: int = 1) -> pd.Series:
    periods = hours * READINGS_PER_HOUR
    return level_series.diff(periods) / hours


def rolling_rainfall(rainfall_series: pd.Series, hours: int) -> pd.Series:
    periods = hours * READINGS_PER_HOUR
    return rainfall_series.rolling(periods, min_periods=1).sum()


def rolling_baseline(level_series: pd.Series, days: int = 90) -> pd.Series:
    """A trailing median -- this station's 'typical' level over the recent past."""
    periods = days * 24 * READINGS_PER_HOUR
    return level_series.rolling(periods, min_periods=READINGS_PER_HOUR * 24).median()


def seasonality(datetime_index: pd.DatetimeIndex) -> pd.DataFrame:
    day_of_year = datetime_index.dayofyear
    angle = 2 * np.pi * day_of_year / 365.25
    return pd.DataFrame({"season_sin": np.sin(angle), "season_cos": np.cos(angle)}, index=datetime_index)
