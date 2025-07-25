from dataclasses import dataclass

from .constants_frame import TYPE_PLOT_INPUT, TYPE_PLOT_STYLE


@dataclass
class ConfigPlot:
    x: TYPE_PLOT_INPUT
    y: TYPE_PLOT_INPUT
    c: TYPE_PLOT_INPUT
    style: TYPE_PLOT_STYLE

    def _resolve_inputs(self): ...
