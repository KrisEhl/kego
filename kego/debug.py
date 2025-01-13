import logging
from functools import wraps
from time import time
from typing import Callable


def timing(f: Callable):
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
