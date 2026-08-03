"""Heatmap marker with crosshair for pyqtgraph PlotWidgets.

Shows interpolated Z value from a 2D grid as the mouse moves over
an ImageItem heatmap. Displays crosshair lines, a dot marker, and
a tooltip with bilinear-interpolated values.

The pure ``interp_bilinear`` helper lives in :mod:`lm19.heatmap_interp`.

Usage:
    marker = HeatmapMarker(plot_widget, "Ia", "mA", ".2f")
    marker.set_grid(z_grid, x_vals, y_vals)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QLabel

from lm19.heatmap_interp import interp_bilinear
from lm19.plot_style import (
    COLOR_QPOINT, COLOR_TOOLTIP_BG, COLOR_TOOLTIP_BORDER, COLOR_ZONE,
)


class HeatmapMarker:
    """Crosshair marker for 2D heatmap (ImageItem) plots.

    Tracks mouse position and shows bilinear-interpolated Z value
    from the underlying grid. No lock/cycle behaviour.

    Lifecycle:
    - Create once per plot widget.
    - plot.clear() is NOT called on heatmap plots, so reattach is not needed.
    - Call set_grid() after each render to update the data.
    """

    def __init__(self, plot_widget: pg.PlotWidget,
                 z_name: str = "Z", z_unit: str = "", z_fmt: str = ".2f") -> None:
        self._plot = plot_widget
        self._z_name = z_name
        self._z_unit = z_unit
        self._z_fmt = z_fmt

        self._z_grid: Optional[np.ndarray] = None
        self._x_vals: Optional[np.ndarray] = None
        self._y_vals: Optional[np.ndarray] = None
        self._aux: list = []  # [(name, unit, fmt, grid, x_vals, y_vals), ...]
        self._enabled: bool = True
        self._visible: bool = False

        # --- Visual items ---
        pen_cross = pg.mkPen(COLOR_ZONE, width=1,
                             style=Qt.PenStyle.DashLine)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pen_cross)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pen_cross)
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._vline.setZValue(1000)
        self._hline.setZValue(1000)

        self._scatter = pg.ScatterPlotItem(
            size=12, brush=pg.mkBrush(COLOR_QPOINT),
            pen=pg.mkPen("w", width=2),
        )
        self._scatter.setVisible(False)
        self._scatter.setZValue(1001)

        plot_widget.addItem(self._vline, ignoreBounds=True)
        plot_widget.addItem(self._hline, ignoreBounds=True)
        plot_widget.addItem(self._scatter, ignoreBounds=True)

        # Tooltip label on viewport
        self._tooltip = QLabel(plot_widget.viewport())
        self._tooltip.setStyleSheet(
            f"background-color: {COLOR_TOOLTIP_BG}; border: 1px solid {COLOR_TOOLTIP_BORDER};"
            "padding: 4px; font-size: 11px;"
        )
        self._tooltip.setTextFormat(Qt.TextFormat.RichText)
        self._tooltip.hide()

        # Mouse tracking
        self._proxy = pg.SignalProxy(
            plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_move,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_grid(self, z: np.ndarray, x_vals, y_vals) -> None:
        """Set new grid data. x_vals = columns (Ua), y_vals = rows (Ug1)."""
        self._z_grid = np.asarray(z, dtype=float)
        self._x_vals = np.asarray(x_vals, dtype=float)
        self._y_vals = np.asarray(y_vals, dtype=float)
        self.hide()

    def set_fields(self, z_name: str, z_unit: str, z_fmt: str) -> None:
        """Change Z-value description."""
        self._z_name = z_name
        self._z_unit = z_unit
        self._z_fmt = z_fmt

    def set_aux_grids(self, aux: list) -> None:
        """Set auxiliary grids shown in tooltip below the main Z value.

        Args:
            aux: list of (name, unit, fmt, grid, x_vals, y_vals) tuples.
                 Each grid is interpolated at the cursor position and
                 appended to the tooltip.  Pass [] to clear.
        """
        self._aux = list(aux)

    def hide(self) -> None:
        """Hide all visual elements."""
        self._visible = False
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._scatter.setVisible(False)
        self._tooltip.hide()

    def clear(self) -> None:
        """Remove grid data and hide marker."""
        self._z_grid = None
        self._x_vals = None
        self._y_vals = None
        self._aux = []
        self.hide()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable mouse tracking."""
        self._enabled = enabled
        if not enabled:
            self.hide()

    # ------------------------------------------------------------------
    # Internal: mouse handling
    # ------------------------------------------------------------------

    def _on_mouse_move(self, evt) -> None:
        if not self._enabled or self._z_grid is None:
            return

        pos = evt[0]
        if not self._plot.sceneBoundingRect().contains(pos):
            self.hide()
            return

        vb = self._plot.getPlotItem().vb
        mouse_point = vb.mapSceneToView(pos)
        mx = mouse_point.x()
        my = mouse_point.y()

        # Clamp to grid bounds so tooltip works at heatmap edges
        mx = float(np.clip(mx, self._x_vals[0], self._x_vals[-1]))
        my = float(np.clip(my, self._y_vals[0], self._y_vals[-1]))

        z_val = interp_bilinear(self._z_grid, self._x_vals, self._y_vals,
                                mx, my)
        if z_val is None or np.isnan(z_val):
            self.hide()
            return

        self._show_at(mx, my, z_val, vb)

    # ------------------------------------------------------------------
    # Internal: display
    # ------------------------------------------------------------------

    def _show_at(self, x: float, y: float, z_val: float, vb) -> None:
        """Position crosshair, marker and tooltip at (x, y)."""
        self._vline.setPos(x)
        self._hline.setPos(y)
        self._vline.setVisible(True)
        self._hline.setVisible(True)

        self._scatter.setData([x], [y])
        self._scatter.setVisible(True)

        # Build tooltip
        z_str = f"{self._z_name}: {z_val:{self._z_fmt}} {self._z_unit}"
        aux_parts: list = []
        for a_name, a_unit, a_fmt, a_grid, a_x, a_y in self._aux:
            a_val = interp_bilinear(a_grid, a_x, a_y, x, y)
            if a_val is not None and not np.isnan(a_val):
                aux_parts.append(f"{a_name}: {a_val:{a_fmt}} {a_unit}")
        x_str = f"Ua: {x:.1f} V"
        y_str = f"Ug1: {y:.1f} V"
        lines = [z_str]
        if aux_parts:
            lines.append(" &nbsp;×&nbsp; ".join(aux_parts))
        lines.append(f"{x_str} &nbsp; {y_str}")
        text = "<br>".join(lines)

        self._tooltip.setText(text)
        self._tooltip.adjustSize()

        # Position tooltip near the marker
        scene_pos = vb.mapViewToScene(QPointF(x, y))
        widget_pos = self._plot.mapFromScene(scene_pos)
        tip_x = widget_pos.x() + 18
        tip_y = widget_pos.y() - 10

        vp_rect = self._plot.viewport().rect()
        tw = self._tooltip.width()
        th = self._tooltip.height()
        if tip_x + tw > vp_rect.width():
            tip_x = widget_pos.x() - tw - 10
        if tip_x < 5:
            tip_x = 5
        if tip_y + th > vp_rect.height():
            tip_y = vp_rect.height() - th - 5
        if tip_y < 0:
            tip_y = 5

        self._tooltip.move(int(tip_x), int(tip_y))
        self._tooltip.raise_()
        self._tooltip.show()
        self._visible = True
