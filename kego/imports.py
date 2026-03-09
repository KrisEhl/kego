import pathlib
import warnings


def _import_torch(file):
    try:
        import torch as torch
    except ModuleNotFoundError:
        warnings.warn(
            f"Need to install `torch` to use all functionality in {pathlib.Path(file).parent}.",
            stacklevel=2,
        )
    else:
        return torch
