import os
import pathlib
from pathlib import Path
from typing import Literal, TypeAlias

import matplotlib.axes
import matplotlib.colorbar
import matplotlib.figure
import pandas as pd

_PATH_FILE = Path(__file__).parent

TIME_TYPE_PANDAS = pd.Timestamp
TYPE_MATPLOTLIB_AXES: TypeAlias = matplotlib.axes.Axes
TYPE_MATPLOTLIB_FIGURES: TypeAlias = matplotlib.figure.Figure
TYPE_MATPLOTLIB_COLORBAR: TypeAlias = matplotlib.colorbar.Colorbar
TYPE_FILEPATHS: TypeAlias = str | pathlib.Path
AXES_STYLE_TYPE: TypeAlias = Literal["axes_single", "axes_grid"]
DEFAULT_COLORMAP: str = "viridis"
DEFAULT_FONTSIZE_LARGE: float = 20
DEFAULT_FONTSIZE_SMALL: float = 12
DEFAULT_FIGURE_SIZE: tuple[float, float] = (10.0, 6.0)

PATH_DATA = Path(os.environ.get("KEGO_PATH_DATA", _PATH_FILE / "../data")).absolute()
