from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from ..lines import plot_line
from .constants_frame import (
    TYPE_DATA_INPUT,
    TYPE_DATAFRAME,
    TYPE_PLOT_INPUT,
    TYPE_PLOT_STYLE,
)
from .scaffold import Scaffold


def _extract_values_and_lables_from_data_input(
    data: TYPE_PLOT_INPUT, df: pl.DataFrame | None = None
) -> tuple[np.ndarray, str]:
    match type(data):
        case str:
            if df is None:
                raise ValueError(f"Need to provide df to use string inputs ({data})!")
            return df[data].to_numpy(), data  # type: ignore


class Frame:
    def __init__(
        self,
        nx: int | None = None,
        ny: int | None = None,
        figure_size: tuple[float, float] | None = None,
    ):
        self.nx = nx
        self.ny = ny
        self.figure_size = figure_size

    def plot(
        self,
        x: TYPE_DATA_INPUT = None,
        y: TYPE_DATA_INPUT = None,
        style: TYPE_PLOT_STYLE = None,
        df: TYPE_DATAFRAME | None = None,
    ):

        match style:
            case "line":
                axes = plot_line(x=x, y=y)
        return

    def line(self, x: TYPE_DATA_INPUT = None, y: TYPE_DATA_INPUT = None):
        return

    def draw(self):
        return
