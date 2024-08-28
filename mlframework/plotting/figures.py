import logging
import os.path
import pathlib

import mlframework.constants


def save_figure(
    fig: mlframework.constants.TYPE_MATPLOTLIB_FIGURES,
    filename: str | pathlib.Path | None = None,
    dpi: int = 450,
) -> None:
    """Save figure to filename"""
    if filename is not None:
        logging.info(f"... saving {filename}")
        folder = os.path.split(filename.__str__())[0]
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(filename, bbox_inches="tight", dpi=dpi)
