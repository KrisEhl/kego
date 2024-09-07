import pathlib
import warnings


def _import_torch(file):
    try:
        import torch as torch  # noqa: E501
    except ModuleNotFoundError:
        warnings.warn(
            f"Need to install `torch` to use all functionality in {pathlib.Path(file).parent}."
        )
    else:
        return torch
