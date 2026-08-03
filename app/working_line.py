"""Live working-line layer on the 2D plot.

Source of truth is the amp panel: the controller listens to its
controls, recomputes a LIGHT bundle via
``engine.compute_working_line`` on a debounce (the same math and
routing as full Analyze — unity is guaranteed by the engine) and
INCREMENTALLY updates its own pyqtgraph items — no full plot
re-render per tick (~446 ms full render vs single-digit ms bundle).

Method visibility: every number in the info line carries the label of
the method actually used; fallbacks change the label, never go silent.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import pyqtgraph as pg
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QLabel

from i18n_setup import t
from lm19.amp_engine import (
    HD_METHOD_5POINT,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_DFT,
    AmplifierEngine,
    AmpParams,
    WorkingLineView,
)
from lm19.amplifier.constants import CIRCUIT_PP
from lm19.plot_style import (
    COLOR_LOAD_LINE,
    COLOR_PP_COMPOSITE,
    COLOR_SWING,
    WORKING_LINE_FAMILY_ALPHA,
)
from lm19.label_formats import format_label

log = logging.getLogger(__name__)

# ── module local constants ──
# Coalescing window for spin ticks (with the debounce a ~200 ms tick
# is acceptable: perceived latency = debounce + tick).
WORKING_LINE_DEBOUNCE_MS = 200
# Budget for the lm19 recompute of the light bundle per tick.
# Measured on a real 2827-point pentode scan: data paths <= 2.7 ms;
# the heaviest UL-PP path (wrapped intersections + model-Chebyshev +
# family) peaks at 24.8 ms -> x2 headroom.
WORKING_LINE_RECOMPUTE_BUDGET_MS = 60
# Budget for the FULL tick (recompute + incremental item update).
# Measured: SE 10.2 / PP-Cheb 16.1 / UL-PP worst 69.6 ms -> x2
# headroom; below the ~200 ms perception ceiling with the debounce.
# Exceeding it logs a WARNING (visible degradation).
WORKING_LINE_TICK_BUDGET_MS = 150
_ISECT_LABEL_FONT_PT = 8
# 2D swing markers (alphas/sizes mirror the retired
# overlays.draw_analysis_markers look).
_SWING_MARK_ALPHA = 200      # swing-end triangles
_SWING_LINE_ALPHA = 160      # bold swing segment
_SWING_HALF_ALPHA = 140      # half-point squares (+/- swing/2)
_SWING_SEGMENT_WIDTH = 4.0
_SWING_MARK_SIZE = 10
_SWING_HALF_SIZE = 7
_KINK_SIZE = 9               # partner-cutoff marker (A->AB) on 2D
_METHOD_KEYS = {
    HD_METHOD_5POINT: "plot.Wl_method_5point",
    HD_METHOD_CHEBYSHEV: "plot.Wl_method_chebyshev",
    HD_METHOD_DFT: "plot.Wl_method_dft",
}


class WorkingLineController(QObject):
    """Owns the line/Q/intersection items on the 2D plot.

    Wiring (main_window): amp-panel controls -> ``schedule()``;
    ``load_line_cb`` -> ``set_visible``; full Analyze ->
    ``apply_full_result``; scan-data change -> ``invalidate``;
    after a full 2D re-render -> ``reattach``.
    """

    def __init__(
        self,
        plot: pg.PlotWidget,
        engine: AmplifierEngine,
        get_params: Callable[[], AmpParams],
        info_label: Optional[QLabel] = None,
        renderer: Optional[object] = None,
    ) -> None:
        super().__init__()
        self._plot = plot
        self._engine = engine
        self._get_params = get_params
        self._info_label = info_label
        self._renderer = renderer

        self._visible = False
        self._points_epoch = 0
        self._cache_key: Optional[tuple] = None
        self._view: Optional[WorkingLineView] = None
        self.last_recompute_ms: float = 0.0
        self.last_tick_ms: float = 0.0

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(WORKING_LINE_DEBOUNCE_MS)
        self._timer.timeout.connect(self._recompute)

        pen_line = pg.mkPen(COLOR_LOAD_LINE, width=2.0,
                            style=pg.QtCore.Qt.PenStyle.DashDotLine)
        pen_dc = pg.mkPen(COLOR_LOAD_LINE, width=1.0,
                          style=pg.QtCore.Qt.PenStyle.DotLine)
        self._line_item = pg.PlotDataItem(pen=pen_line)
        self._dc_item = pg.PlotDataItem(pen=pen_dc)
        self._isect_item = pg.PlotDataItem(
            pen=None, symbol="d", symbolSize=10,
            symbolBrush=COLOR_LOAD_LINE,
            symbolPen=pg.mkPen("w", width=1))
        self._q_item = pg.PlotDataItem(
            pen=None, symbol="x", symbolSize=12,
            symbolBrush=COLOR_LOAD_LINE,
            symbolPen=pg.mkPen("w", width=2))
        sw_mark = pg.mkColor(COLOR_SWING)
        sw_mark.setAlpha(_SWING_MARK_ALPHA)
        sw_line = pg.mkColor(COLOR_SWING)
        sw_line.setAlpha(_SWING_LINE_ALPHA)
        sw_half = pg.mkColor(COLOR_SWING)
        sw_half.setAlpha(_SWING_HALF_ALPHA)
        self._swing_item = pg.PlotDataItem(
            pen=pg.mkPen(sw_line, width=_SWING_SEGMENT_WIDTH),
            symbol="t", symbolSize=_SWING_MARK_SIZE,
            symbolBrush=sw_mark, symbolPen=pg.mkPen("w", width=1))
        self._swing_half_item = pg.PlotDataItem(
            pen=None, symbol="s", symbolSize=_SWING_HALF_SIZE,
            symbolBrush=sw_half, symbolPen=pg.mkPen("w", width=1))
        # A->AB kink: partner-cutoff point on the joint trajectory.
        self._kink_item = pg.PlotDataItem(
            pen=None, symbol="o", symbolSize=_KINK_SIZE,
            symbolBrush=COLOR_PP_COMPOSITE,
            symbolPen=pg.mkPen("w", width=1))
        self._items = [self._line_item, self._dc_item,
                       self._isect_item, self._q_item,
                       self._swing_item, self._swing_half_item,
                       self._kink_item]
        # Pool of dashed UL-family curves (size changes rarely — with
        # the Ug1 count; items are reused).
        self._family_items: List[pg.PlotDataItem] = []
        self._labels: List[pg.TextItem] = []
        self._attached = False

    def _family_pen(self) -> pg.QtGui.QPen:
        c = pg.mkColor(COLOR_LOAD_LINE)
        c.setAlpha(WORKING_LINE_FAMILY_ALPHA)
        return pg.mkPen(c, width=1.0,
                        style=pg.QtCore.Qt.PenStyle.DashLine)

    def _render_family(self, view: WorkingLineView) -> None:
        fam = view.model_family
        while len(self._family_items) < len(fam):
            item = pg.PlotDataItem(pen=self._family_pen())
            self._family_items.append(item)
            if self._attached:
                self._plot.addItem(item)
        for item, (_g, ua, ia) in zip(self._family_items, fam):
            item.setData(ua, ia)
        for item in self._family_items[len(fam):]:
            item.setData([], [])

    # -- public slots / API ------------------------------------------

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        if not self._visible:
            self._clear_items()
            self._set_info("")
            self._feed_legacy(None)
            return
        # OFF->ON with unchanged parameters: _recompute would hit the
        # cache and NOT redraw the cleared items (line and Q used to
        # vanish until the first parameter change). The cached view is
        # valid — draw immediately, no debounce, no recompute.
        if (self._view is not None
                and self._params_key(self._get_params()) == self._cache_key):
            self._render_view(self._view)
            return
        self.schedule()

    def schedule(self, *_args) -> None:
        """Panel-control tick: coalesced by the debounce timer
        (last-write-wins — parameters are read when it fires)."""
        if not self._visible:
            return
        self._timer.start()

    def invalidate(self) -> None:
        """Scan data changed — the cache is invalid."""
        self._points_epoch += 1
        self._cache_key = None
        if self._visible:
            self._timer.start()

    def apply_full_result(self, view: Optional[WorkingLineView]) -> None:
        """Full Analyze already computed the view — show, no recompute."""
        if view is None:
            return
        self._cache_key = self._params_key(self._get_params())
        self._render_view(view)

    def reattach(self) -> None:
        """A full 2D re-render calls plot.clear() — re-add the items."""
        self._attached = False
        if self._visible and self._view is not None:
            self._attach()
            self._render_view(self._view)

    # -- internals ---------------------------------------------------

    def _params_key(self, p: AmpParams) -> tuple:
        return (self._points_epoch, p.circuit, p.ub, p.ra, p.ug1_bias,
                p.half_swing, p.ug2_filter, p.hd_method, p.ul_tap,
                p.ra_dc, p.cf_rk, p.cf_rl, p.pp_raa, p.pp_ra_dc,
                p.pp_matched, p.pp_tube_b_sid, p.series_id,
                tuple(p.sources))

    def _recompute(self) -> None:
        if not self._visible:
            return
        params = self._get_params()
        key = self._params_key(params)
        if key == self._cache_key and self._view is not None:
            return
        t0 = time.perf_counter()
        view = self._engine.compute_working_line(params)
        self.last_recompute_ms = (time.perf_counter() - t0) * 1000.0
        if self.last_recompute_ms > WORKING_LINE_RECOMPUTE_BUDGET_MS:
            log.warning(
                "working line recompute %.1f ms > budget %d ms "
                "(circuit=%s, method=%s)", self.last_recompute_ms,
                WORKING_LINE_RECOMPUTE_BUDGET_MS, params.circuit,
                params.hd_method)
        self._cache_key = key
        self._render_view(view)
        self.last_tick_ms = (time.perf_counter() - t0) * 1000.0
        if self.last_tick_ms > WORKING_LINE_TICK_BUDGET_MS:
            log.warning(
                "working line tick %.1f ms > budget %d ms",
                self.last_tick_ms, WORKING_LINE_TICK_BUDGET_MS)

    def _attach(self) -> None:
        if self._attached:
            return
        for it in self._items + self._family_items:
            self._plot.addItem(it)
        self._attached = True

    def _clear_items(self) -> None:
        for it in self._items + self._family_items:
            it.setData([], [])
        for lbl in self._labels:
            self._plot.removeItem(lbl)
        self._labels.clear()

    def _render_view(self, view: WorkingLineView) -> None:
        self._view = view
        if not self._visible:
            return
        self._attach()
        for lbl in self._labels:
            self._plot.removeItem(lbl)
        self._labels.clear()

        if view.error is not None or not view.polyline:
            for it in self._items + self._family_items:
                it.setData([], [])
            self._set_info(self._info_text(view))
            self._feed_legacy(view)
            return

        xs = [p[0] for p in view.polyline]
        ys = [p[1] for p in view.polyline]
        self._line_item.setData(xs, ys)
        if view.dc_polyline:
            self._dc_item.setData([p[0] for p in view.dc_polyline],
                                  [p[1] for p in view.dc_polyline])
        else:
            self._dc_item.setData([], [])

        isects = view.intersections
        self._isect_item.setData([p["ua"] for p in isects],
                                 [p["ia"] for p in isects])
        for p in isects:
            lbl = pg.TextItem(
                format_label("ug1_short", p["ug1"]),
                color=COLOR_LOAD_LINE, anchor=(0.5, 1.3))
            lbl.setFont(pg.QtGui.QFont("", _ISECT_LABEL_FONT_PT))
            lbl.setPos(p["ua"], p["ia"])
            self._plot.addItem(lbl)
            self._labels.append(lbl)

        q = self._q_for_marker(view)
        if q is not None:
            self._q_item.setData([q[0]], [q[1]])
        else:
            self._q_item.setData([], [])

        # Swing markers: swing ends + segment + half-points.
        geo = view.swing_geometry or {}
        pt_neg, pt_pos = geo.get("pt_neg"), geo.get("pt_pos")
        if pt_neg and pt_pos:
            self._swing_item.setData(
                [pt_neg["ua"], pt_pos["ua"]],
                [pt_neg["ia"], pt_pos["ia"]])
        else:
            self._swing_item.setData([], [])
        pt_lh, pt_hh = geo.get("pt_low_half"), geo.get("pt_high_half")
        if pt_lh and pt_hh:
            self._swing_half_item.setData(
                [pt_lh["ua"], pt_hh["ua"]],
                [pt_lh["ia"], pt_hh["ia"]])
        else:
            self._swing_half_item.setData([], [])

        # A->AB kink (PP+joint only; None = pure class A or display).
        kink = getattr(view, "pp_kink", None)
        if kink:
            self._kink_item.setData([kink["ua"]], [kink["ia"]])
            k_lbl = pg.TextItem(t("plot.Wl_kink"),
                                color=COLOR_PP_COMPOSITE, anchor=(0, 1.2))
            k_lbl.setFont(pg.QtGui.QFont("", _ISECT_LABEL_FONT_PT))
            k_lbl.setPos(kink["ua"], kink["ia"])
            self._plot.addItem(k_lbl)
            self._labels.append(k_lbl)
        else:
            self._kink_item.setData([], [])

        self._render_family(view)

        self._set_info(self._info_text(view))
        self._feed_legacy(view)

    @staticmethod
    def _q_for_marker(view: WorkingLineView) -> Optional[Tuple[float, float]]:
        if view.q_ua is not None and view.q_ia is not None:
            return view.q_ua, view.q_ia
        hd = view.hd or {}
        if hd.get("ua_0") is not None and hd.get("ia_0") is not None:
            return hd["ua_0"], hd["ia_0"]
        # Method-independent geometry: the Q cross stays alive with a
        # failed HD on the resistive circuit (xfmr/PP get it from DC-Q).
        geo = view.swing_geometry or {}
        if geo.get("ua_0") is not None and geo.get("ia_0") is not None:
            return geo["ua_0"], geo["ia_0"]
        return None

    def _info_text(self, view: WorkingLineView) -> str:
        """Info line: every number carries its method label."""
        if view.error == "no_data":
            return t("plot.Wl_no_data")
        if view.error == "needs_ug2":
            return t("plot.Wl_needs_ug2")
        ll = view.load_line
        parts: List[str] = []
        if ll is not None:
            parts.append(ll.label())
        # Line-source visibility: joint trajectory from the model vs
        # the straight Zaa/4 idealization — the user must see which of
        # the two is drawn.
        if view.circuit == CIRCUIT_PP:
            parts.append(t("plot.Wl_line_joint") if view.pp_trajectory
                         else t("plot.Wl_line_display"))
        # Iq/Ua_q directly in the info line (previously the number
        # existed only in the results panel after a full Analyze).
        q = self._q_for_marker(view)
        if q is not None:
            parts.append(t("plot.Wl_q", ua=f"{q[0]:.0f}", ia=f"{q[1]:.1f}"))
        method = t(_METHOD_KEYS.get(view.method_used,
                                    _METHOD_KEYS[HD_METHOD_5POINT]))
        hd = view.hd
        if hd is not None:
            parts.append(
                t("plot.Wl_hd_line",
                  thd=f"{hd.get('thd', 0.0):.2f}",
                  hd2=f"{hd.get('hd2', 0.0):.2f}",
                  hd3=f"{hd.get('hd3', 0.0):.2f}",
                  method=method))
            pout = hd.get("pout_mw")
            if pout:
                parts.append(t("plot.Wl_pout", pout=f"{pout / 1000.0:.2f}"))
        elif view.hd_error:
            parts.append(t("plot.Wl_hd_error",
                           code=view.hd_error, method=method))
        if view.note == "fixed_ug2":
            parts.append(t("plot.Wl_note_fixed_ug2"))
        return "  │  ".join(parts)

    def _set_info(self, text: str) -> None:
        if self._info_label is None:
            return
        self._info_label.setText(text)
        self._info_label.setVisible(bool(text))

    def _feed_legacy(self, view: Optional[WorkingLineView]) -> None:
        """Compatibility: the curves tab and other tabs Q markers
        read renderer._load_line_intersections/_analysis.

        analysis = hd + swing_geometry: THD/HD numbers stay
        method-specific (from hd), while the WORKING-POINT geometry
        (Q + swing points) comes from swing_geometry when present —
        geometry keys are display semantics (per-tube for PP), whereas
        e.g. the PP hd dict carries the COMPOSITE ia_0 (~0 for a
        matched pair), which printed "Q: Ia=0.0 mA" in the PDF report.
        The geometry is drawn ALWAYS (Chebyshev/DFT and hd=None alike).
        After the feed, Q/swing markers on heatmaps and Transfer are
        redrawn right here — without this call the markers would not
        appear on enable and would go stale on disable.
        """
        if self._renderer is None:
            return
        if view is None:
            self._renderer._load_line_intersections = []
            self._renderer._load_line_analysis = None
            self._renderer._pp_composite = []
            self._renderer._pp_bias = 0.0
            self._renderer.draw_qpoint_all()
            return
        self._renderer._load_line_intersections = list(view.intersections)
        # PP composite: an empty list for non-PP circuits resets the
        # curve (switching PP->SE must not leave a tail on Transfer).
        self._renderer._pp_composite = list(view.pp_composite)
        self._renderer._pp_bias = view.pp_bias
        geo = dict(view.swing_geometry) if view.swing_geometry else {}
        hd = dict(view.hd) if view.hd else {}
        analysis = {**hd, **geo} if (geo or hd) else None
        q = self._q_for_marker(view)
        if analysis is not None and q is not None:
            analysis.setdefault("ua_0", q[0])
            analysis.setdefault("ia_0", q[1])
        if analysis is not None:
            # Engine-resolved method — the same single source the info
            # line shows; the PDF distortion block prints it (method
            # visibility: the 5-point dict carries no "method" of its
            # own and auto-routing means the numbers are NOT 5-point).
            analysis["method"] = view.method_used
        self._renderer._load_line_analysis = analysis
        self._renderer.draw_qpoint_all()
