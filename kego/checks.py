import ast
import collections
from collections.abc import Iterable
from typing import Iterator, Optional, Sequence

import numpy as np


def get_random_indices(n_sample: int, size_data: int) -> np.ndarray:
    """Draw `n_sample` indices from range(`size_data`)"""
    return np.random.choice(
        range(size_data),
        n_sample if n_sample < size_data else size_data,
        replace=False,
    )


def evaluate_string(x: object):
    try:
        return ast.literal_eval(str(x))
    except ValueError:
        return x


def chunks(sequence: Sequence, n: int) -> Iterator:
    """Yield successive `n`-sized chunks from `sequence`."""
    if n == -1:
        n = len(sequence)
    for i in range(0, len(sequence), n):
        yield sequence[slice(i, i + n)]


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


def all_same_type(variable_list: Iterable, type_: type):
    for var in variable_list:
        if not isinstance(var, type_):
            raise ValueError(f"{var} not of type {type_.__name__}!")


def assert_shape(
    variable, shape: tuple, name: Optional[str] = None, ignore_none: bool = True
):
    if ignore_none:
        if variable is None:
            return
    if np.shape(variable) != shape:
        raise TypeError(
            f"{name+': ' if name is not None else ''}{variable=} doesn't have required {shape=}!"
        )


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
