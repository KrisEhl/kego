import os

import numpy as np

from kego.constants import TYPE_FILEPATHS


def list_files(
    path,
    match_filename: str | None = None,
    level_max: int = np.inf,
    return_absolute_path: bool = False,
):
    path = path.__str__()
    c = path.count(os.sep)
    for root, dirs, files in os.walk(path):
        for name in files:
            result = os.path.join(root, name)
            if root.count(os.sep) - c - 1 <= level_max and (
                match_filename is None or name == match_filename
            ):
                if return_absolute_path:
                    result = os.path.abspath(result)
                yield result
