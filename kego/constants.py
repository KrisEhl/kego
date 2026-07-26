import os
import pathlib
from pathlib import Path
from typing import Literal

import matplotlib.axes
import matplotlib.colorbar
import matplotlib.figure
import pandas as pd

_PATH_FILE = Path(__file__).parent

TIME_TYPE_PANDAS = pd.Timestamp
type TYPE_MATPLOTLIB_AXES = matplotlib.axes.Axes
type TYPE_MATPLOTLIB_FIGURES = matplotlib.figure.Figure
type TYPE_MATPLOTLIB_COLORBAR = matplotlib.colorbar.Colorbar
type TYPE_FILEPATHS = str | pathlib.Path
type AXES_STYLE_TYPE = Literal["axes_single", "axes_grid"]
DEFAULT_COLORMAP: str = "viridis"
DEFAULT_FONTSIZE_LARGE: float = 20
DEFAULT_FONTSIZE_SMALL: float = 12
DEFAULT_FIGURE_SIZE: tuple[float, float] = (10.0, 6.0)

PATH_DATA = Path(os.environ.get("KEGO_PATH_DATA", _PATH_FILE / "../data")).expanduser().absolute()
