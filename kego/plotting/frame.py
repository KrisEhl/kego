from dataclasses import dataclass
from typing import Literal, TypeAlias, Union

import numpy as np
import pandas as pd
import polars as pl

from .lines import plot_line

TYPE_DATA_INPUT = Union[
    str,
    pl.Series,
    np.ndarray,
    float,
    list[str],
    list[pl.Series],
    list[np.ndarray],
    list[float],
    list[list[float]],
    None,
]

TYPE_PLOT_INPUT = Union[
    str,
    pd.Series,
    np.ndarray,
    float,
    list[float],
    None,
]

TYPE_PLOT_STYLE: TypeAlias = (
    Literal["line", "histogram", "hist", "hist2d", "histogram_2d", "map", "scatter"]
    | None
)


def _extract_values_and_lables_from_data_input(
    data: TYPE_PLOT_INPUT, df: pl.DataFrame | None = None
):
    match type(data):
        case str:
            if df is None:
                raise ValueError(f"Need to provide df to use string inputs ({data})!")
            return df[data], data


@dataclass
class ConfigPlot:
    x: TYPE_PLOT_INPUT
    y: TYPE_PLOT_INPUT
    c: TYPE_PLOT_INPUT
    style: TYPE_PLOT_STYLE

    def _resolve_inputs(self): ...


class Frame:
    def __init__(self, nx: int, ny: int, figure_size: tuple[float, float]):
        self.nx = nx
        self.ny = ny
        self.figure_size = figure_size

    def plot(
        self,
        x: TYPE_DATA_INPUT = None,
        y: TYPE_DATA_INPUT = None,
        style: TYPE_PLOT_STYLE = None,
    ):
        match style:
            case "line":
                axes = plot_line(x=x, y=y)
        return

    def line(self, x: TYPE_DATA_INPUT = None, y: TYPE_DATA_INPUT = None):
        return

    def draw(self):
        return
