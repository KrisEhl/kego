import datetime
import functools
import re

import numpy as np
import pandas as pd


def round_to_base(x: float, base: float):
    return base * round(x / base)


def round_offset(values: np.ndarray, decimal: int, offset: float) -> np.ndarray:
    values += offset
    values = np.round(values, decimals=decimal)
    values -= offset
    return values


def round_numpy_time_to_base_minutes(
    time: np.datetime64, base: int = 5
) -> np.datetime64:
    tm = datetime.datetime.utcfromtimestamp(time.tolist() / 1e9)
    tm += datetime.timedelta(minutes=base / 2)
    tm -= datetime.timedelta(
        minutes=tm.minute % base, seconds=tm.second, microseconds=tm.microsecond
    )
    return np.datetime64(tm)


def round_time_to_base_minutes(time: np.datetime64, base: int = 5):
    tm = pd.to_datetime(time).round(f"{base}min")
    return np.datetime64(tm)


def vetorize_time_to_base_minutes(times: np.ndarray, base: int = 5):
    function = functools.partial(round_numpy_time_to_base_minutes, base=base)
    rounded = np.array(list(map(function, times)))
    return rounded


def str_to_delta_time(string: str) -> tuple[float, str]:
    time, units, _ = re.split("([a-zA-Z]+)$", string)
    return float(time), units
