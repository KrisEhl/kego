import itertools
import multiprocessing
from typing import Callable, Iterable, Optional


def get_processes(processes):
    if processes < 0:
        processes = multiprocessing.cpu_count() + processes + 1
    return processes


def parallelize(
    function: Callable,
    args_zipped: Iterable,
    processes: int = -1,
    single_arg: bool = False,
    kwargs_as_dict: Optional[dict] = None,
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


def starmap_with_kwargs(
    pool, function: Callable, args_iter: Iterable, kwargs_iter: Iterable
):
    """Helper function to parallelize functions with args and kwargs"""
    if kwargs_iter is None:
        args_for_starmap = zip(itertools.repeat(function), args_iter)
    else:
        args_for_starmap = zip(itertools.repeat(function), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(fn: Callable, args, kwargs):
    """Helper function to parallelize functions with args and kwargs"""
    return fn(*args, **kwargs)
