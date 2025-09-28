from typing import Literal, TypeAlias, Union

import numpy as np
import pandas as pd
import polars as pl

TYPE_DATAFRAME: TypeAlias = pl.DataFrame

TYPE_PLOT_STYLE: TypeAlias = (
    Literal["line", "histogram", "hist", "hist2d", "histogram_2d", "map", "scatter"]
    | None
)

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
