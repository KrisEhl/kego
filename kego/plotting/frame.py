import logging
from dataclasses import dataclass
from typing import Literal, TypeAlias, Union

import numpy as np
import pandas as pd
import polars as pl

from .lines import plot_line

LOGGER = logging.getLogger(__name__)
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

TYPE_COLOR_INPUT = Union[
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

TYPE_LABEL = str | None
TYPE_LIM: TypeAlias = tuple[float | None, float | None] | None

TYPE_PLOT_STYLE: TypeAlias = (
    Literal["line", "histogram", "hist", "hist2d", "histogram_2d", "map", "scatter"]
    | None
)


def _extract_values_and_lable_from_data_input(
    data: TYPE_PLOT_INPUT, label: str | None, df: pl.DataFrame | None = None
) -> tuple[float | np.ndarray | None, str | None]:
    match data:
        case str():
            if df is None:
                raise ValueError(f"Need to provide df to use string inputs ({data})!")
            if data not in df:
                raise ValueError(f"Column {data} not found in df {df.columns=}!")
            if label is None:
                label = data
            return np.array(df[data]), label
        case None:
            return None, label
        case _:
            return np.array(data), label


def _extract_colors_and_label_from_data_input(
    data: TYPE_PLOT_INPUT, label: str | None, df: pl.DataFrame | None = None
) -> tuple[float | np.ndarray | None | str, str | None]:
    match data:
        case str():
            if df is None:
                raise ValueError(f"Need to provide df to use string inputs ({data})!")
            if data not in df:
                LOGGER.debug(
                    f"Column {data} not found in {df.columns}, assuming color value!"
                )
                colors = data
            else:
                colors = np.array(df[data])
                if label is None:
                    label = data
            return colors, label
        case None:
            return None, label
        case _:
            return np.array(data), label


@dataclass
class ConfigPlot:
    x: TYPE_PLOT_INPUT
    y: TYPE_PLOT_INPUT

    limits_x: TYPE_LIM
    limits_y: TYPE_LIM

    label_x: TYPE_LABEL
    label_y: TYPE_LABEL
    label_colorbar: TYPE_LABEL

    style: TYPE_PLOT_STYLE

    def _resolve_inputs(self):
        self.x, self.label_x = _extract_values_and_lable_from_data_input(
            data=self.x, label=self.label_x
        )
        self.y, self.label_y = _extract_values_and_lable_from_data_input(
            data=self.y, label=self.label_y
        )

    def __post_init__(self):
        self._resolve_inputs()


@dataclass
class ConfigPlotHistogram2d(ConfigPlot):
    statistics: Literal["sum", "min", "max", "abs"]
    limits_colorbar: TYPE_LIM


@dataclass
class ConfigPlotLine(ConfigPlot):
    colors: TYPE_PLOT_INPUT

    def _resolve_inputs(self):
        self.colors, self.label_colorbar = _extract_colors_and_label_from_data_input(
            data=self.colors, label=self.label_colorbar
        )


@dataclass
class PlotCollection:
    plots: list[ConfigPlotHistogram2d | ConfigPlotLine]


@dataclass
class ConfigStyle:
    font_size: int | None


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
