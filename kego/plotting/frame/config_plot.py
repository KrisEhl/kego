import abc
import logging
from dataclasses import dataclass
from typing import Literal, Type

import numpy as np
import polars as pl

from ..histograms import plot_histogram
from .constants_frame import (
    TYPE_DATA_INPUT,
    TYPE_DATAFRAME,
    TYPE_LABEL,
    TYPE_LIM,
    TYPE_PLOT_INPUT,
    TYPE_PLOT_STYLE,
)

LOGGER = logging.getLogger(__name__)


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
class ConfigPlotStyle:
    fontsize: float | None


@dataclass
class ConfigPlot(metaclass=abc.ABCMeta):
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

    @abc.abstractmethod
    def draw(self):
        raise NotImplementedError("Required to implement 'draw' method!")


@dataclass
class ConfigPlotHistogram(ConfigPlot):
    def draw(self):
        axes = plot_histogram(
            key_or_values=self.x,
            xlim=self.limits_x,
            ylim=self.limits_y,
            label_x=self.label_x,
            label_y=self.label_y,
        )


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


@dataclass
class ConfigFrame:
    config_style: ConfigStyle
    config_plot: Type[ConfigPlot]
