import ast
import collections
import datetime
import functools
import itertools
import logging
import multiprocessing
import pathlib
import re
import typing as t
import warnings
from collections.abc import Iterable
from functools import wraps
from time import time
from typing import Optional

import numpy as np
import pandas as pd


def timing(f: t.Callable):
    """
    Wrapper that returns execution time and arguments of function.
    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logging.info(f"func:{f.__name__} args:[{kw}] took: {te-ts} sec")
        return result

    return wrap


def get_processes(processes):
    if processes < 0:
        processes = multiprocessing.cpu_count() + processes + 1
    return processes


def parallelize(
    function: t.Callable,
    args_zipped: t.Iterable,
    processes: int = -1,
    single_arg: bool = False,
    kwargs_as_dict: t.Optional[dict] = None,
):
    """
    parallelize function with args provided in zipped format
    ----------
    function: function to parallelize
    args: args of function in zipped format

    Returns
    -------
    function applied to args
    """
    if processes == 1:
        results = []
        if kwargs_as_dict is None:
            kwargs_as_dict = {}
        for args in args_zipped:
            if isinstance(args, str):
                args = (args,)
            results.append(function(*args, **kwargs_as_dict))
        return results
    processes = get_processes(processes)
    print(f"n processes: {processes}")
    if single_arg and kwargs_as_dict is None:
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(function, args_zipped)
    else:
        if single_arg:
            args_zipped = ((arg,) for arg in args_zipped)
        if kwargs_as_dict is not None:
            kwargs_iter = itertools.repeat(kwargs_as_dict)
        else:
            kwargs_iter = itertools.repeat(dict())
        with multiprocessing.Pool(processes=processes) as pool:
            results = starmap_with_kwargs(pool, function, args_zipped, kwargs_iter)
    return results


def get_random_indices(n_sample: int, size_data: int) -> np.ndarray:
    """Draw `n_sample` indices from range(`size_data`)"""
    return np.random.choice(
        range(size_data),
        n_sample if n_sample < size_data else size_data,
        replace=False,
    )


def starmap_with_kwargs(
    pool, function: t.Callable, args_iter: t.Iterable, kwargs_iter: t.Iterable
):
    """Helper function to parallelize functions with args and kwargs"""
    if kwargs_iter is None:
        args_for_starmap = zip(itertools.repeat(function), args_iter)
    else:
        args_for_starmap = zip(itertools.repeat(function), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(fn: t.Callable, args, kwargs):
    """Helper function to parallelize functions with args and kwargs"""
    return fn(*args, **kwargs)


def evaluate_string(x: object):
    try:
        return ast.literal_eval(str(x))
    except ValueError:
        return x


def chunks(sequence: t.Sequence, n: int) -> t.Iterator:
    """Yield successive `n`-sized chunks from `sequence`."""
    if n == -1:
        n = len(sequence)
    for i in range(0, len(sequence), n):
        yield sequence[slice(i, i + n)]


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


def _assert_same_type(variable, type_from_variable):
    """Check if `variable` shares type with `type_from_variable`"""
    if isinstance(variable, collections.abc.Iterable):
        if not isinstance(type_from_variable, collections.abc.Iterable):
            raise TypeError(
                f"Variable {variable=} is iterable but type is not {type_from_variable=}"
            )
        for v, typ in zip(variable, type_from_variable):
            _assert_same_type(v, typ)
    else:
        if not isinstance(variable, type(type_from_variable)):
            raise TypeError(
                f"Variable {variable=} different type than expected {type_from_variable=}"
            )


def assert_same_type_as(
    variable: object, type_from_variable: object, alternative: Optional[object] = None
):
    """Check if `variable` is same type as `type_from_variable` unless `variable` is `alternative`"""
    if variable is alternative:
        return
    try:
        _assert_same_type(variable, type_from_variable)
    except TypeError:
        raise TypeError(
            f"Variable {variable} not of expected type {type_from_variable}"
        )


def all_same_type(variable_list: t.Iterable, type_: type):
    for var in variable_list:
        if not isinstance(var, type_):
            raise ValueError(f"{var} not of type {type_.__name__}!")


def assert_shape(
    variable, shape: t.Tuple, name: Optional[str] = None, ignore_none: bool = True
):
    if ignore_none:
        if variable is None:
            return
    if np.shape(variable) != shape:
        raise TypeError(
            f"{name+': ' if name is not None else ''}{variable=} doesn't have required {shape=}!"
        )


def flatten_list(lis: list[list]) -> list:
    return [item for sublist in lis for item in sublist]


def str_to_delta_time(string: str) -> t.Tuple[float, str]:
    time, units, _ = re.split("([a-zA-Z]+)$", string)
    return float(time), units


def to_nlength_tuple(x: object, n: int = 2) -> tuple:
    if not (isinstance(x, list) or isinstance(x, tuple) or isinstance(x, np.ndarray)):
        return tuple([x] * n)
    elif isinstance(x, list) or isinstance(x, np.ndarray):
        x = tuple(x)
    if len(x) != n:
        raise ValueError(f"{x} doesn't have expected length {n=}")
    return x


def assert_all_same_type(variable_list: Iterable, type_: type):
    for var in variable_list:
        if not isinstance(var, type_):
            raise ValueError(f"{var} not of type {type_.__name__}!")


def validate_array(array: np.ndarray, type_: str = "float", name: str | None = None):
    """Validate numpy based on `type`, e.g. no nan-values"""
    if type_ == "float":
        if np.sum(np.isnan(array.astype(float))) > 0:
            raise ValueError(
                f"Found nan-values {f'in {name}' if name is not None else ''}!"
            )
    else:
        raise ValueError(f"{type_=} unknown!")


def override_class_method(class_instance, method_name, target_class):
    class_method = getattr(target_class, method_name)

    def new_method(*args, **kwargs):
        return class_method(class_instance, *args, **kwargs)

    setattr(class_instance, method_name, new_method)


def is_in_list_and_remove(to_check, _list):
    is_in = False
    if to_check in _list:
        is_in = True
    if is_in:
        _list.remove(to_check)
    return is_in


def _import_torch(file):
    try:
        import torch as torch  # noqa: E501
    except ModuleNotFoundError:
        warnings.warn(
            f"Need to install `torch` to use all functionality in {pathlib.Path(file).parent}."
        )
    else:
        return torch
