"""Plotting package — Qt-dependent plot rendering primitives.

Submodules:
  - grids: pure computation for Ia/Gm/Rp/mu/Pa grids
  - grouping: CurveData builders and clustering helpers
  - overlays: Qt overlay drawing functions (zone, limits, load line, model)
  - renderer: PlotRenderer class (Qt-dependent rendering)
"""

from app.plotting.renderer import PlotRenderer

__all__ = ["PlotRenderer"]
