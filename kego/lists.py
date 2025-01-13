from typing import Iterable, overload

import numpy as np


def flatten_list(lis: list[list]) -> list:
    return [item for sublist in lis for item in sublist]


def to_nlength_tuple(x: object, n: int = 2) -> tuple:
    if not (isinstance(x, list) or isinstance(x, tuple) or isinstance(x, np.ndarray)):
        return tuple([x] * n)
    elif isinstance(x, list) or isinstance(x, np.ndarray):
        x = tuple(x)
    if len(x) != n:
        raise ValueError(f"{x} doesn't have expected length {n=}")
    return x


def is_in_list_and_remove(to_check, _list):
    is_in = False
    if to_check in _list:
        is_in = True
    if is_in:
        _list.remove(to_check)
    return is_in


def is_listlike(to_check: object):
    if isinstance(to_check, Iterable) and not isinstance(to_check, str):
        return True
    return False
