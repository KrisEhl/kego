from dataclasses import dataclass

from .constants_frame import TYPE_DATAFRAME, TYPE_PLOT_INPUT, TYPE_PLOT_STYLE


@dataclass
class ConfigPlotHistogram:
    x: TYPE_PLOT_INPUT
    df: TYPE_DATAFRAME
    style: TYPE_PLOT_STYLE

    def _resolve_inputs(self): ...


@dataclass
class ConfigPlotStyle:
    fontsize: float | None
