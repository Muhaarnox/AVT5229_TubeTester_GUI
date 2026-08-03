"""Snap-to-curve marker with crosshair for pyqtgraph PlotWidgets.

Provides a visual marker (dot) that snaps to the nearest curve on the plot,
crosshair lines, and a tooltip showing interpolated values. The marker
slides along curves as the mouse moves horizontally, with click to switch
curves and Ctrl+click to freeze/unfreeze.

Pure data structures (CurveData, FIELDS_*, build_compare_curves, …) live
in :mod:`lm19.curve_data`.

Usage:
    marker = CurveMarker(plot_widget, fields=FIELDS_2D)
    marker.set_curves(curves)  # list of CurveData
    # marker auto-tracks mouse; call marker.clear() before re-render
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QLabel

from lm19.constants import EPS, MW_PER_W
from lm19.curve_data import CurveData, FIELDS_2D
from lm19.label_formats import format_label
from lm19.plot_style import (
    COLOR_QPOINT, COLOR_TOOLTIP_BG, COLOR_TOOLTIP_BORDER, COLOR_ZONE,
)


class CurveMarker:
    """Snap-to-curve marker with crosshair and tooltip.

    Features:
    - Crosshair (dashed h+v lines) follows interpolated position
    - Bright dot marker snaps to nearest curve
    - Tooltip shows interpolated values
    - Free mode: mouse auto-snaps to nearest curve
    - Click: lock to current curve (marker slides along it, won't jump)
    - Click again: unlock (back to auto-snap)

    Lifecycle:
    - Create once per plot widget.
    - After plot.clear() call reattach() to re-add visual items.
    - Signal connections are made once and survive plot.clear().
    """

    def __init__(self, plot_widget: pg.PlotWidget, fields: Optional[Dict] = None,
                 lock_px: int = 15) -> None:
        self._plot = plot_widget
        self._fields = fields or FIELDS_2D
        self._lock_px = lock_px  # click-to-lock proximity in pixels
        self._curves: List[CurveData] = []
        self._current_curve_idx: int = -1
        self._locked: bool = False
        self._visible: bool = False
        self._enabled: bool = True  # False → suppress all mouse handling

        # --- Visual items (created once, survive reattach) ---
        pen_cross = pg.mkPen(COLOR_ZONE, width=1,
                             style=Qt.PenStyle.DashLine)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pen_cross)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pen_cross)
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        # High Z-values so crosshair/marker render on top of data curves
        self._vline.setZValue(1000)
        self._hline.setZValue(1000)

        self._scatter = pg.ScatterPlotItem(
            size=12, brush=pg.mkBrush(COLOR_QPOINT),
            pen=pg.mkPen("w", width=2),
        )
        self._scatter.setVisible(False)
        self._scatter.setZValue(1001)  # above crosshair

        # Add items to plot
        plot_widget.addItem(self._vline, ignoreBounds=True)
        plot_widget.addItem(self._hline, ignoreBounds=True)
        plot_widget.addItem(self._scatter, ignoreBounds=True)

        # Tooltip label — child of viewport so it renders ON TOP of the scene
        self._tooltip = QLabel(plot_widget.viewport())
        self._tooltip.setStyleSheet(
            f"background-color: {COLOR_TOOLTIP_BG};"
            f" border: 1px solid {COLOR_TOOLTIP_BORDER};"
            " padding: 4px; font-size: 11px;"
        )
        self._tooltip.setTextFormat(Qt.TextFormat.RichText)
        self._tooltip.hide()

        # --- Connect signals once ---
        self._proxy = pg.SignalProxy(
            plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_move,
        )
        plot_widget.scene().sigMouseClicked.connect(self._on_mouse_click)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reattach(self) -> None:
        """Re-add visual items after plot.clear() removed them.

        Call this instead of creating a new CurveMarker to avoid
        accumulating signal connections.
        """
        self._plot.addItem(self._vline, ignoreBounds=True)
        self._plot.addItem(self._hline, ignoreBounds=True)
        self._plot.addItem(self._scatter, ignoreBounds=True)
        self._curves = []
        self._current_curve_idx = -1
        self._locked = False
        self.hide()

    def set_curves(self, curves: List[CurveData]) -> None:
        """Set new curve data. Resets current curve selection."""
        self._curves = curves
        self._current_curve_idx = -1
        self._locked = False
        self.hide()

    def clear(self) -> None:
        """Remove all curves and hide marker."""
        self._curves = []
        self._current_curve_idx = -1
        self._locked = False
        self.hide()

    def hide(self) -> None:
        """Hide all visual elements."""
        self._visible = False
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._scatter.setVisible(False)
        self._tooltip.hide()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable mouse tracking.

        When disabled, the marker ignores all mouse events and hides.
        Use this to suppress the marker during live scanning so it
        doesn't snap to partially-built curves.
        """
        self._enabled = enabled
        if not enabled:
            self.hide()

    def set_fields(self, fields: Dict) -> None:
        """Change tooltip field configuration."""
        self._fields = fields

    # ------------------------------------------------------------------
    # Internal: mouse handling
    # ------------------------------------------------------------------

    def _on_mouse_move(self, evt) -> None:
        """Track mouse: locked → slide along curve, free → snap to nearest."""
        if not self._enabled or not self._curves:
            return

        pos = evt[0]
        if not self._plot.sceneBoundingRect().contains(pos):
            if not self._locked:
                self.hide()
            return

        vb = self._plot.getPlotItem().vb
        mouse_point = vb.mapSceneToView(pos)
        mx = mouse_point.x()
        my = mouse_point.y()

        if self._locked and self._current_curve_idx >= 0:
            # Locked: slide along the locked curve
            curve = self._curves[self._current_curve_idx]
            yi = self._interp_y(curve, mx)
            if yi is None:
                return  # outside X range — keep last position
            self._show_at(mx, yi, curve, vb)
        else:
            # Free: snap to nearest curve
            best_idx = self._find_nearest_curve(mx, my)
            if best_idx < 0:
                self.hide()
                return
            self._current_curve_idx = best_idx
            curve = self._curves[best_idx]
            yi = self._interp_y(curve, mx)
            if yi is None:
                self.hide()
                return
            self._show_at(mx, yi, curve, vb)

    def _on_mouse_click(self, evt) -> None:
        """Left click: lock to curve / cycle through overlapping curves.

        Lock only activates when the click is close to the nearest curve
        (within _lock_px pixels).
        """
        if not self._enabled or not self._curves:
            return
        if evt.button() != Qt.MouseButton.LeftButton:
            return

        pos = evt.scenePos()
        if not self._plot.sceneBoundingRect().contains(pos):
            return

        vb = self._plot.getPlotItem().vb
        mouse_point = vb.mapSceneToView(pos)
        mx = mouse_point.x()
        my = mouse_point.y()

        if self._locked:
            if self._is_near_curve(mx, my, self._current_curve_idx):
                # Click near locked curve → cycle to next curve at this X
                next_idx = self._find_next_curve(mx, my, self._current_curve_idx)
                if next_idx >= 0 and next_idx != self._current_curve_idx:
                    self._current_curve_idx = next_idx
                    curve = self._curves[next_idx]
                    yi = self._interp_y(curve, mx)
                    if yi is not None:
                        self._show_at(mx, yi, curve, vb)
                else:
                    # Only one curve here → unlock
                    self._locked = False
            else:
                # Click away from curve → unlock
                self._locked = False
        elif self._visible and self._current_curve_idx >= 0:
            # Free → lock only if click is close to the current curve
            if self._is_near_curve(mx, my, self._current_curve_idx):
                self._locked = True

    # ------------------------------------------------------------------
    # Internal: curve math
    # ------------------------------------------------------------------

    def _is_near_curve(self, mx: float, my: float, curve_idx: int) -> bool:
        """Check if (mx, my) is close to the curve (within _lock_px pixels)."""
        curve = self._curves[curve_idx]
        yi = self._interp_y(curve, mx)
        if yi is None:
            return False
        # Convert data Y-distance to pixel distance
        vb = self._plot.getPlotItem().vb
        p1 = vb.mapViewToScene(QPointF(mx, my))
        p2 = vb.mapViewToScene(QPointF(mx, yi))
        dy_px = abs(p1.y() - p2.y())
        return dy_px < self._lock_px

    @staticmethod
    def _interp_y(curve: CurveData, x_val: float) -> Optional[float]:
        """Interpolate Y value on curve at given X. Returns None if out of range."""
        xarr = curve.x
        yarr = curve.y
        if len(xarr) < 2:
            return None
        if x_val < xarr[0] or x_val > xarr[-1]:
            return None
        return float(np.interp(x_val, xarr, yarr))

    def _find_nearest_curve(self, mx: float, my: float) -> int:
        """Find curve index nearest to (mx, my) by normalized 2D distance."""
        vr = self._plot.getPlotItem().vb.viewRange()
        x_range = max(vr[0][1] - vr[0][0], EPS)
        y_range = max(vr[1][1] - vr[1][0], EPS)

        best_idx = -1
        best_dist = float('inf')

        for idx, curve in enumerate(self._curves):
            yi = self._interp_y(curve, mx)
            if yi is not None:
                dy = (yi - my) / y_range
                dist = dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            else:
                # Try nearest point on curve
                if len(curve.x) == 0:
                    continue
                dists = ((curve.x - mx) / x_range) ** 2 + ((curve.y - my) / y_range) ** 2
                min_d = float(np.min(dists))
                if min_d < best_dist:
                    best_dist = min_d
                    best_idx = idx

        return best_idx

    def _find_next_curve(self, mx: float, my: float, current_idx: int) -> int:
        """Find the next curve at mx, cycling through by Y distance.

        Returns the curve after current_idx in the Y-distance-sorted list.
        Wraps around to the nearest if at end of list.
        """
        # Build list of (distance, idx) for all curves covering mx
        candidates = []
        for idx, curve in enumerate(self._curves):
            yi = self._interp_y(curve, mx)
            if yi is not None:
                candidates.append((abs(yi - my), idx))
        if not candidates:
            return -1
        # Sort by distance
        candidates.sort(key=lambda t: t[0])
        # Find current_idx position in sorted list
        indices = [c[1] for c in candidates]
        try:
            pos = indices.index(current_idx)
            # Advance to next; wrap around
            next_pos = (pos + 1) % len(indices)
            return indices[next_pos]
        except ValueError:
            return indices[0]

    # ------------------------------------------------------------------
    # Internal: display
    # ------------------------------------------------------------------

    def _show_at(self, x: float, y: float, curve: CurveData, vb) -> None:
        """Position crosshair, marker and tooltip at (x, y)."""
        self._vline.setPos(x)
        self._hline.setPos(y)
        self._vline.setVisible(True)
        self._hline.setVisible(True)

        self._scatter.setData([x], [y])
        self._scatter.setVisible(True)

        # Build tooltip text
        f = self._fields
        x_str = f"{f['x_name']}: {x:{f['x_fmt']}} {f['x_unit']}"
        y_str = f"{f['y_name']}: {y:{f['y_fmt']}} {f['y_unit']}"

        lines = []
        # Line 1: source name (if any)
        if curve.label:
            lines.append(f"<b>{curve.label}</b>")
        # Line 2: primary values (Y, X)
        lines.append(f"{y_str} &nbsp; {x_str}")
        # Line 3+: extra interpolated fields (Ug1, Ug2, Ig2, etc.)
        extra_parts = []
        for ef in f.get("extra", []):
            arr = curve.extra.get(ef["key"])
            if arr is not None and len(arr) == len(curve.x):
                ev = self._interp_y(
                    CurveData(x=curve.x, y=arr), x
                )
                if ev is not None:
                    extra_parts.append(f"{ef['name']}: {ev:{ef['fmt']}} {ef['unit']}")
        if extra_parts:
            lines.append(" &nbsp; ".join(extra_parts))

        # Pa = Ua * Ia / MW_PER_W (for 2D/Compare where x=Ua, y=Ia)
        if f.get("show_pa"):
            pa = x * y / MW_PER_W
            pa_pg2 = format_label("pa", pa)
            # Pg2 = Ug2 * Ig2 / MW_PER_W (if both arrays present)
            ug2_arr = curve.extra.get("Ug2")
            ig2_arr = curve.extra.get("Ig2")
            if (ug2_arr is not None and ig2_arr is not None
                    and len(ug2_arr) == len(curve.x)
                    and len(ig2_arr) == len(curve.x)):
                ug2_v = self._interp_y(CurveData(x=curve.x, y=ug2_arr), x)
                ig2_v = self._interp_y(CurveData(x=curve.x, y=ig2_arr), x)
                if ug2_v is not None and ig2_v is not None and ug2_v > 0:
                    pg2 = ug2_v * ig2_v / MW_PER_W
                    pa_pg2 += " &nbsp; " + format_label("pg2", pg2)
            lines.append(pa_pg2)

        if self._locked:
            lines.append("<i>locked</i>")

        self._tooltip.setText("<br>".join(lines))
        self._tooltip.adjustSize()

        # Position tooltip near the marker point
        # Convert data coords → scene coords → viewport (widget) coords
        scene_pos = vb.mapViewToScene(QPointF(x, y))
        widget_pos = self._plot.mapFromScene(scene_pos)
        tip_x = widget_pos.x() + 18
        tip_y = widget_pos.y() - 10

        # Keep tooltip inside viewport bounds
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
