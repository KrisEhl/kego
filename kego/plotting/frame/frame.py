import logging
from dataclasses import dataclass
from typing import Literal, Type, TypeAlias, Union

import numpy as np
import pandas as pd
import polars as pl

from ..histograms import plot_histogram
from ..lines import plot_line
from .config_plot import ConfigFrame, ConfigPlotHistogram, ConfigPlotStyle, ConfigStyle
from .constants_frame import (
    TYPE_DATA_INPUT,
    TYPE_DATAFRAME,
    TYPE_LABEL,
    TYPE_LIM,
    TYPE_PLOT_INPUT,
    TYPE_PLOT_STYLE,
)
from .scaffold import Scaffold

LOGGER = logging.getLogger(__name__)


def _extract_values_and_lables_from_data_input(
    data: TYPE_PLOT_INPUT, df: pl.DataFrame | None = None
) -> tuple[np.ndarray | None, str | None]:
    if isinstance(data, str):
        if df is None:
            raise ValueError(f"Need to provide df to use string inputs ({data})!")
        return df[data].to_numpy(), data
    return None, None


class Frame:
    def __init__(
        self,
        nx: int | None = 2,
        ny: int | None = 1,
        figure_size: tuple[float, float] | None = None,
        fontsize: int = 10,
    ):
        self.nx = nx
        self.ny = ny
        self.figure_size = figure_size
        self.scaffold = Scaffold.from_nx_ny(nx=nx, ny=ny)
        self.fontsize = fontsize

    def plot(
        self,
        x: TYPE_DATA_INPUT | Literal["all"] = None,
        y: TYPE_DATA_INPUT = None,
        style: TYPE_PLOT_STYLE = None,
        df: TYPE_DATAFRAME | None = None,
        fontsize: int | None = None,
    ):

        if style in ["line"]:
            axes = plot_line(x=x, y=y)
        elif style in ["histogram", "hist"]:
            if y is not None:
                raise ValueError(f"{y=} should be None when plotting '{style}'.")
            if x is "all":
                if df is None:
                    raise ValueError(f"Need to provide {df=} to use {x=}!")
                for column in df.columns:
                    self.scaffold.set(
                        confif_plot=ConfigFrame(
                            config_plot=ConfigPlotHistogram(
                                x=column, df=df, style="hist"
                            ),
                            config_style=ConfigStyle(
                                fontsize=(
                                    fontsize if fontsize is not None else self.fontsize
                                )
                            ),
                        )
                    )
        return

    def draw(self):
        for i in enumerate(self.scaffold.ntot):
            self.scaffold.entries[i].draw()
