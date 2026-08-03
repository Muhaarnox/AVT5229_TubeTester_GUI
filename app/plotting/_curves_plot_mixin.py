"""Curves-plot rendering mixin for ``PlotRenderer``.

Contains the ``render_curves`` family of line plots
(Gm / Rp / Mu / Ia / Ig2 / Pa / Pig2).

The mixin assumes its host class provides (set by ``PlotRenderer.__init__``):
  - ``self.curves_plot``, ``self.transfer_plot`` — pyqtgraph widgets
  - ``self._marker_curves``, ``self._marker_transfer`` — CurveMarker
  - ``self._curves_labels``, ``self._transfer_labels`` — list of labels
  - ``self._ug2_curves_items``, ``self._ug2_transfer_items`` — Ug2→items
  - ``self._all_marker_curves_curves`` / ``_transfer`` — marker stores
  - ``self.palette`` — color cycle
  - ``self.is_triode``, ``self.ug2_cluster_thr``, ``self.ug1_cluster_thr``
    via ``RendererContext`` properties

It does NOT define ``__init__`` — all attributes come from the host class.
``self.SCAN_SYMBOL``, ``CURVES_SYMBOL_SIZE`` are class-level constants
inherited from ``PlotRenderer``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg

from i18n_setup import t
from lm19.constants import (
    IG2_NEAR_ZERO_MA as _IG2_NEAR_ZERO_MA,
    MW_PER_W as _MW_PER_W,
)
from lm19.curve_data import CurveData, build_curves_fields
from lm19.label_formats import format_label
from lm19.plot_style import (
    COLOR_CURVE_HALO as _COLOR_CURVE_HALO,
    COLOR_LIMIT as _COLOR_LIMIT,
    COLOR_LOAD_LINE as _COLOR_LOAD_LINE,
    COLOR_PP_COMPOSITE as _COLOR_PP_COMPOSITE,
    LOAD_LINE_CURVE_MIN_WIDTH as _LL_CURVE_MIN_W,
    LOAD_LINE_HALO_EXTRA_W as _LL_HALO_EXTRA_W,
    LOAD_LINE_MARKER_ALPHA as _LL_MARKER_ALPHA,
    LOAD_LINE_HALO_ALPHA as _LL_HALO_ALPHA,
    TRANSFER_DIM_ALPHA as _TRANSFER_DIM_ALPHA,
)


def _load_line_tint(alpha: int):
    """COLOR_LOAD_LINE with the given alpha (ML-019/020: no raw RGB)."""
    c = pg.mkColor(_COLOR_LOAD_LINE)
    c.setAlpha(alpha)
    return c
from lm19.plotting.grids import (
    build_ia_grid,
    cluster_nominal as _cluster_nominal_fn,
    compute_gm_rp_grids,
    filter_ug2_multi,
    nominal_key as _nominal_key_fn,
)


class _CurvesPlotMixin:
    """Render Gm/Rp/Mu/Ia/Ig2/Pa/Pig2 vs Ua/Ug1 as families of line curves.

    Host: ``PlotRenderer`` (which provides ``__init__`` setup of all
    referenced ``self.<attr>`` items + class-level constants like
    ``SCAN_SYMBOL`` / ``CURVES_SYMBOL_SIZE``).
    """

    _CURVES_Y_LABELS = {
        "Gm": 'plot.Gm_mA_V',
        "Rp": 'plot.Rp_kOhm',
        "mu": 'plot.mu',
        "Ia": 'plot.Ia_mA',
        "Ig2": 'plot.Ig2_mA',
        "Pa": 'plot.Pa_W',
        "Pig2": 'plot.Pig2_W',
        "Ia/Ig2": 'plot.Ia_Ig2_ratio',
    }

    def render_curves(self, points: List[Dict],
                      y_param: str = "Gm",
                      x_param: str = "Ua",
                      select_ug2_slice=None,
                      ug2_values: List[float] = None,
                      nominal_s: float = None,
                      line_width: float = 2.0,
                      series_labels: Optional[Dict[int, str]] = None,
                      track_sids: set = None,
                      target_plot=None,
                      ub_ref: Optional[float] = None) -> None:
        """Render Gm/Rp/mu/Ia/Ig2/Pa/Pig2 as families of line curves.

        When points contain multiple series_id values, each series is rendered
        with a distinct pen style while keeping the same colour palette for
        matching Ug1/Ua curves.

        Pass *target_plot* to render onto a different PlotWidget (e.g.
        ``self.transfer_plot``).  The appropriate item stores, labels list
        and CurveMarker are selected automatically.
        """
        plot = target_plot or self.curves_plot
        if plot is None:
            return
        if plot is self.transfer_plot:
            ug2_store = self._ug2_transfer_items
            labels = self._transfer_labels
            marker = self._marker_transfer
        else:
            ug2_store = self._ug2_curves_items
            labels = self._curves_labels
            marker = self._marker_curves
        self._rc_plot = plot
        self._rc_ug2_store = ug2_store
        self._rc_labels = labels
        self._rc_marker = marker
        self._rc_cd_accum = []
        # Ub reference for the "≈Ub" slice accent on the Transfer plot.
        self._rc_ub_ref = ub_ref

        plot.clear()
        ug2_store.clear()
        if plot is self.transfer_plot:
            self._ua_transfer_items.clear()
        for item in labels:
            plot.removeItem(item)
        labels.clear()
        if marker:
            marker.reattach()
            marker.set_fields(build_curves_fields(y_param, x_param))

        _x_labels = {"Ua": 'plot.Ua_V', "Ug1": 'plot.Ug1_V', "Ug2": 'plot.Ug2_V'}
        x_label_key = _x_labels.get(x_param, 'plot.Ua_V')
        y_label_key = self._CURVES_Y_LABELS.get(y_param, 'plot.Ia_mA')
        plot.setLabel("bottom", t(x_label_key))
        plot.setLabel("left", t(y_label_key))

        if not points:
            return

        by_sid: Dict[int, List[Dict]] = {}
        for p in points:
            by_sid.setdefault(p.get("series_id", 0), []).append(p)
        sids = sorted(by_sid.keys())
        multi_series = len(sids) > 1

        rc_plot_item = plot.getPlotItem()
        if multi_series:
            if getattr(rc_plot_item, "legend", None) is None:
                rc_plot_item.addLegend()

        if track_sids is None:
            track_sids = set()

        for sid_idx, sid in enumerate(sids):
            sid_pts = by_sid[sid]
            ps = self._overlay_pen_styles[sid_idx % len(self._overlay_pen_styles)]
            sid_triode = sid in track_sids

            if y_param in ("Ia", "Ig2", "Pa", "Pig2", "Ia/Ig2"):
                if x_param == "Ug2":
                    self._render_curves_raw(sid_pts, y_param, x_param,
                                            line_width, pen_style=ps)
                else:
                    self._render_curves_raw_multi_ug2(
                        sid_pts, y_param, x_param, line_width, ug2_values,
                        pen_style=ps, is_triode_eff=sid_triode)
            elif x_param != "Ug2":
                self._render_curves_grid(
                    sid_pts, y_param, x_param, line_width,
                    ug2_values, pen_style=ps, is_triode_eff=sid_triode)

            if multi_series:
                label = (series_labels or {}).get(
                    sid, "Current scan" if sid == 0 else f"Series {sid}")
                pen_leg = pg.mkPen(self.palette[0], width=line_width, style=ps)
                plot.plot([], [], pen=pen_leg, name=label)

        if nominal_s is not None and nominal_s > 0 and y_param == "Gm":
            pen_ref = pg.mkPen(_COLOR_LIMIT, width=1.5,
                               style=pg.QtCore.Qt.PenStyle.DashLine)
            plot.addItem(pg.InfiniteLine(
                pos=nominal_s, angle=0, pen=pen_ref,
                label=f"S nom={nominal_s:.1f}",
                labelOpts={"color": _COLOR_LIMIT},
            ))

        if y_param == "Ia" and x_param in ("Ug1", "Ua"):
            self._render_curves_load_line(plot, line_width, x_param=x_param)
        if y_param == "Ia" and x_param == "Ug1":
            # The composite is a function of Ug1: it does not exist on
            # Ia(Ua) axes.
            self._render_curves_pp_composite(plot, line_width)

        self._finalize_curves_marker()

    def _render_curves_grid(self, points: List[Dict],
                            y_param: str, x_param: str,
                            line_width: float,
                            ug2_values: Optional[List[float]] = None,
                            pen_style=None,
                            is_triode_eff: bool = False) -> None:
        """Render grid-based Gm/Rp/mu curves for a single series."""
        ug2_groups = filter_ug2_multi(
            points, is_triode_eff, ug2_values, self.ug2_cluster_thr)
        if not ug2_groups:
            return

        multi_ug2 = len(ug2_groups) > 1
        rc_pi = self._rc_plot.getPlotItem()
        if multi_ug2:
            if getattr(rc_pi, "legend", None) is None:
                rc_pi.addLegend()

        for ug2_idx, (ug2_nom, subset) in enumerate(sorted(ug2_groups.items())):
            g = build_ia_grid(subset, self.ug1_cluster_thr, self.ua_cluster_thr)
            if g is None:
                continue
            ua_vals, ug1_vals, ia_grid = g["ua_vals"], g["ug1_vals"], g["ia_grid"]
            gm_grid, rp_grid = compute_gm_rp_grids(ua_vals, ug1_vals, ia_grid)

            ug2_color = self.palette[ug2_idx % len(self.palette)] if multi_ug2 else None
            ug2_suffix = f" @ {ug2_nom:.0f}V" if multi_ug2 else ""

            self._curves_collect = []
            if x_param == "Ua":
                self._render_curves_vs_ua(
                    ua_vals, ug1_vals, ia_grid,
                    gm_grid, rp_grid, y_param, line_width,
                    color_override=ug2_color, label_suffix=ug2_suffix,
                    pen_style=pen_style, ug2_val=ug2_nom)
            else:
                self._render_curves_vs_ug1(
                    ua_vals, ug1_vals, ia_grid,
                    gm_grid, rp_grid, y_param, line_width,
                    color_override=ug2_color, label_suffix=ug2_suffix,
                    pen_style=pen_style, ug2_val=ug2_nom)

            if multi_ug2:
                pen_leg = pg.mkPen(ug2_color, width=line_width, style=pen_style or pg.QtCore.Qt.PenStyle.SolidLine)
                leg_item = self._rc_plot.plot([], [], pen=pen_leg,
                                     name=f"Ug2={ug2_nom:.0f}V")
                self._curves_collect.append(leg_item)

            if not is_triode_eff:
                self._rc_ug2_store.setdefault(ug2_nom, []).extend(
                    self._curves_collect)
            self._curves_collect = None

    def _finalize_curves_marker(self) -> None:
        """Pass accumulated CurveData to the current target marker."""
        if self._rc_marker:
            if self._rc_marker is self._marker_transfer:
                self._all_marker_curves_transfer = list(self._rc_cd_accum)
            elif self._rc_marker is self._marker_curves:
                self._all_marker_curves_curves = list(self._rc_cd_accum)
            self._rc_marker.set_curves(self._rc_cd_accum)

    def _render_curves_load_line(self, plot, line_width: float,
                                 x_param: str = "Ug1") -> None:
        """Draw load-line intersections and Q-point on Ia curves."""
        isects = getattr(self, "_load_line_intersections", [])
        if not isects:
            return
        x_key = "ug1" if x_param == "Ug1" else "ua"
        xs = [p[x_key] for p in isects]
        ys = [p["ia"] for p in isects]
        # Halo underlay: the dynamic transfer curve must stand out from
        # the family of static slices it crosses.
        ll_width = max(line_width, _LL_CURVE_MIN_W)
        plot.plot(xs, ys, pen=pg.mkPen(_COLOR_CURVE_HALO,
                                       width=ll_width + _LL_HALO_EXTRA_W))
        pen_ll = pg.mkPen(_COLOR_LOAD_LINE, width=ll_width,
                          style=pg.QtCore.Qt.PenStyle.SolidLine)
        plot.plot(
            xs, ys, pen=pen_ll,
            symbol="d", symbolSize=8,
            symbolBrush=_COLOR_LOAD_LINE, symbolPen=pg.mkPen("w", width=1),
        )
        if len(xs) >= 2:
            pen_ideal = pg.mkPen(_COLOR_LOAD_LINE, width=1,
                                 style=pg.QtCore.Qt.PenStyle.DotLine)
            plot.plot([xs[0], xs[-1]], [ys[0], ys[-1]], pen=pen_ideal)
        lbl = pg.TextItem(t('plot.Load_line_label'), color=_COLOR_LOAD_LINE,
                          anchor=(0, 1))
        lbl.setFont(pg.QtGui.QFont("", 9, pg.QtGui.QFont.Weight.Bold))
        mid = len(xs) // 2
        lbl.setPos(xs[mid], ys[mid])
        plot.addItem(lbl)
        self._rc_labels.append(lbl)

        # Draw Q-point and crosshair when load-line analysis is available.
        analysis = getattr(self, "_load_line_analysis", None)
        if analysis:
            x_q = analysis.get("ug1_0") if x_param == "Ug1" else analysis.get("ua_0")
            y_q = analysis.get("ia_0")
            if x_q is not None and y_q is not None:
                plot.plot(
                    [x_q], [y_q], pen=None,
                    symbol="x", symbolSize=12,
                    symbolBrush=pg.mkBrush(_load_line_tint(_LL_MARKER_ALPHA)),
                    symbolPen=pg.mkPen("w", width=2),
                )
                cross_pen = pg.mkPen(_load_line_tint(_LL_HALO_ALPHA), width=1)
                vline = pg.InfiniteLine(pos=x_q, angle=90, pen=cross_pen)
                hline = pg.InfiniteLine(pos=y_q, angle=0, pen=cross_pen)
                plot.addItem(vline)
                plot.addItem(hline)
                self._rc_labels.extend([vline, hline])

    def _render_curves_pp_composite(self, plot, line_width: float) -> None:
        """PP composite transfer curve, folded into the positive quadrant.

        Solid branch — ``ia_composite >= 0`` as computed; dashed branch —
        the negative half mirrored around the bias (``fold_pp_composite``).
        For a matched pair the branches coincide; a visible gap between
        them is the even-harmonic residue of the pair. Fed by
        ``WorkingLineController`` (``self._pp_composite`` — data-path
        composite at Ua=Ub, the same curve pp_distortion analyzes).
        """
        comp = getattr(self, "_pp_composite", None)
        if not comp:
            return
        from lm19.amplifier import fold_pp_composite
        direct, mirrored = fold_pp_composite(
            comp, getattr(self, "_pp_bias", 0.0))
        pen_w = max(line_width, _LL_CURVE_MIN_W)
        label_pos = None
        if len(direct) >= 2:
            xs = [p[0] for p in direct]
            ys = [p[1] for p in direct]
            plot.plot(xs, ys, pen=pg.mkPen(
                _COLOR_CURVE_HALO, width=pen_w + _LL_HALO_EXTRA_W))
            plot.plot(xs, ys,
                      pen=pg.mkPen(_COLOR_PP_COMPOSITE, width=pen_w))
            label_pos = (xs[-1], ys[-1])
        if len(mirrored) >= 2:
            xs = [p[0] for p in mirrored]
            ys = [p[1] for p in mirrored]
            plot.plot(xs, ys, pen=pg.mkPen(
                _COLOR_PP_COMPOSITE, width=pen_w,
                style=pg.QtCore.Qt.PenStyle.DashLine))
            if label_pos is None:
                label_pos = (xs[-1], ys[-1])
        if label_pos is None:
            return
        lbl = pg.TextItem(t('plot.Pp_composite_label'),
                          color=_COLOR_PP_COMPOSITE, anchor=(0, 1))
        lbl.setFont(pg.QtGui.QFont("", 9, pg.QtGui.QFont.Weight.Bold))
        lbl.setPos(*label_pos)
        plot.addItem(lbl)
        self._rc_labels.append(lbl)

    def _render_curves_vs_ua(self, ua_vals, ug1_vals, ia_grid,
                             gm_grid, rp_grid,
                             y_param: str, line_width: float,
                             color_override=None, label_suffix: str = "",
                             pen_style=None, ug2_val: float = None) -> None:
        """Plot parameter(Ua) curves, one per Ug1."""
        n_ug1, n_ua = len(ug1_vals), len(ua_vals)

        if y_param == "Gm":
            for i in range(n_ug1 - 1):
                ug1_mid = (ug1_vals[i] + ug1_vals[i + 1]) / 2.0
                xs, ys, ia_list = [], [], []
                for j in range(n_ua):
                    v = gm_grid[i, j]
                    if not np.isnan(v):
                        xs.append(ua_vals[j])
                        ys.append(v)
                        ia_avg = (ia_grid[i, j] + ia_grid[i + 1, j]) / 2.0 if not np.isnan(ia_grid[i, j]) and not np.isnan(ia_grid[i + 1, j]) else np.nan
                        ia_list.append(ia_avg)
                extra = self._grid_extra(xs, ug1_mid, ia_list, ug2_val=ug2_val)
                self._plot_curve_line(xs, ys, i, ug1_mid, line_width,
                                      color_override=color_override,
                                      label_suffix=label_suffix, extra=extra,
                                      pen_style=pen_style)

        elif y_param == "Rp":
            ua_mids = [(ua_vals[j] + ua_vals[j + 1]) / 2.0 for j in range(n_ua - 1)]
            for i in range(n_ug1):
                xs, ys, ia_list = [], [], []
                for j in range(n_ua - 1):
                    v = rp_grid[i, j]
                    if not np.isnan(v):
                        xs.append(ua_mids[j])
                        ys.append(v)
                        ia_avg = (ia_grid[i, j] + ia_grid[i, j + 1]) / 2.0 if not np.isnan(ia_grid[i, j]) and not np.isnan(ia_grid[i, j + 1]) else np.nan
                        ia_list.append(ia_avg)
                extra = self._grid_extra(xs, ug1_vals[i], ia_list, ug2_val=ug2_val)
                self._plot_curve_line(xs, ys, i, ug1_vals[i], line_width,
                                      color_override=color_override,
                                      label_suffix=label_suffix, extra=extra,
                                      pen_style=pen_style)

        elif y_param == "mu":
            ua_mids = [(ua_vals[j] + ua_vals[j + 1]) / 2.0 for j in range(n_ua - 1)]
            for i in range(n_ug1 - 1):
                ug1_mid = (ug1_vals[i] + ug1_vals[i + 1]) / 2.0
                xs, ys, ia_list = [], [], []
                for j in range(n_ua - 1):
                    gm_left = gm_grid[i, j] if j < gm_grid.shape[1] else np.nan
                    gm_right = gm_grid[i, j + 1] if j + 1 < gm_grid.shape[1] else np.nan
                    rp_val = rp_grid[i, j] if i < rp_grid.shape[0] else np.nan
                    if not np.isnan(gm_left) and not np.isnan(gm_right) and not np.isnan(rp_val):
                        gm_avg = (gm_left + gm_right) / 2.0
                        mu = gm_avg * rp_val
                        xs.append(ua_mids[j])
                        ys.append(mu)
                        ia_vals_4 = [ia_grid[k, l] for k in (i, i + 1) for l in (j, j + 1) if not np.isnan(ia_grid[k, l])]
                        ia_list.append(np.mean(ia_vals_4) if ia_vals_4 else np.nan)
                extra = self._grid_extra(xs, ug1_mid, ia_list, ug2_val=ug2_val)
                self._plot_curve_line(xs, ys, i, ug1_mid, line_width,
                                      color_override=color_override,
                                      label_suffix=label_suffix, extra=extra,
                                      pen_style=pen_style)

    def _render_curves_vs_ug1(self, ua_vals, ug1_vals, ia_grid,
                              gm_grid, rp_grid,
                              y_param: str, line_width: float,
                              color_override=None, label_suffix: str = "",
                              pen_style=None, ug2_val: float = None) -> None:
        """Plot parameter(Ug1) curves, one per Ua."""
        n_ug1, n_ua = len(ug1_vals), len(ua_vals)

        if y_param == "Gm":
            ug1_mids = [(ug1_vals[i] + ug1_vals[i + 1]) / 2.0 for i in range(n_ug1 - 1)]
            for j in range(n_ua):
                xs, ys, ia_list = [], [], []
                for i in range(n_ug1 - 1):
                    v = gm_grid[i, j]
                    if not np.isnan(v):
                        xs.append(ug1_mids[i])
                        ys.append(v)
                        ia_avg = (ia_grid[i, j] + ia_grid[i + 1, j]) / 2.0 if not np.isnan(ia_grid[i, j]) and not np.isnan(ia_grid[i + 1, j]) else np.nan
                        ia_list.append(ia_avg)
                extra = self._grid_extra_ua(xs, ua_vals[j], ia_list, ug2_val=ug2_val)
                self._plot_curve_line(xs, ys, j, ua_vals[j], line_width,
                                      label_fmt="ua", color_override=color_override,
                                      label_suffix=label_suffix, extra=extra,
                                      pen_style=pen_style)

        elif y_param == "Rp":
            for j in range(n_ua - 1):
                ua_mid = (ua_vals[j] + ua_vals[j + 1]) / 2.0
                xs, ys, ia_list = [], [], []
                for i in range(n_ug1):
                    v = rp_grid[i, j]
                    if not np.isnan(v):
                        xs.append(ug1_vals[i])
                        ys.append(v)
                        ia_avg = (ia_grid[i, j] + ia_grid[i, j + 1]) / 2.0 if not np.isnan(ia_grid[i, j]) and not np.isnan(ia_grid[i, j + 1]) else np.nan
                        ia_list.append(ia_avg)
                extra = self._grid_extra_ua(xs, ua_mid, ia_list, ug2_val=ug2_val)
                self._plot_curve_line(xs, ys, j, ua_mid, line_width,
                                      label_fmt="ua", color_override=color_override,
                                      label_suffix=label_suffix, extra=extra,
                                      pen_style=pen_style)

        elif y_param == "mu":
            ua_mids = [(ua_vals[j] + ua_vals[j + 1]) / 2.0 for j in range(n_ua - 1)]
            ug1_mids = [(ug1_vals[i] + ug1_vals[i + 1]) / 2.0 for i in range(n_ug1 - 1)]
            for j in range(n_ua - 1):
                xs, ys, ia_list = [], [], []
                for i in range(n_ug1 - 1):
                    gm_val = gm_grid[i, j]
                    gm_next = gm_grid[i, j + 1] if j + 1 < gm_grid.shape[1] else np.nan
                    rp_val = rp_grid[i, j] if i < rp_grid.shape[0] else np.nan
                    if not np.isnan(gm_val) and not np.isnan(gm_next) and not np.isnan(rp_val):
                        gm_avg = (gm_val + gm_next) / 2.0
                        mu = gm_avg * rp_val
                        xs.append(ug1_mids[i])
                        ys.append(mu)
                        ia_vals_4 = [ia_grid[k, l] for k in (i, i + 1) for l in (j, j + 1) if not np.isnan(ia_grid[k, l])]
                        ia_list.append(np.mean(ia_vals_4) if ia_vals_4 else np.nan)
                extra = self._grid_extra_ua(xs, ua_mids[j], ia_list, ug2_val=ug2_val)
                self._plot_curve_line(xs, ys, j, ua_mids[j], line_width,
                                      label_fmt="ua", color_override=color_override,
                                      label_suffix=label_suffix, extra=extra,
                                      pen_style=pen_style)

    @staticmethod
    def _grid_extra(xs: list, ug1_val: float, ia_list: list,
                    ug2_val: float = None) -> Dict[str, np.ndarray]:
        """Build extra arrays for grid-based curves (x=Ua, step=Ug1)."""
        n = len(xs)
        ua = np.array(xs, dtype=float)
        ia = np.array(ia_list, dtype=float)
        d = {
            "Ug1": np.full(n, ug1_val),
            "Ua": ua,
            "Ia": ia,
            "Pa": ua * ia / _MW_PER_W,
        }
        if ug2_val is not None:
            d["Ug2"] = np.full(n, ug2_val)
        return d

    @staticmethod
    def _grid_extra_ua(xs: list, ua_val: float, ia_list: list,
                       ug2_val: float = None) -> Dict[str, np.ndarray]:
        """Build extra arrays for grid-based curves (x=Ug1, step=Ua)."""
        n = len(xs)
        ia = np.array(ia_list, dtype=float)
        d = {
            "Ua": np.full(n, ua_val),
            "Ug1": np.array(xs, dtype=float),
            "Ia": ia,
            "Pa": np.full(n, ua_val) * ia / _MW_PER_W,
        }
        if ug2_val is not None:
            d["Ug2"] = np.full(n, ug2_val)
        return d

    @staticmethod
    def _raw_extra(pts: List[Dict]) -> Dict[str, np.ndarray]:
        """Build extra arrays for CurveMarker from raw measurement points."""
        ua = np.array([p.get("ua", 0.0) for p in pts], dtype=float)
        ia = np.array([p.get("ia", 0.0) for p in pts], dtype=float)
        return {
            "Ua":  ua,
            "Ug1": np.array([p.get("ug1", 0.0) for p in pts], dtype=float),
            "Ug2": np.array([p.get("ug2", 0.0) for p in pts], dtype=float),
            "Ia":  ia,
            "Ig2": np.array([p.get("ig2", 0.0) for p in pts], dtype=float),
            "Pa":  ua * ia / _MW_PER_W,
        }

    @staticmethod
    def _raw_y_values(points: List[Dict], y_param: str) -> List[float]:
        """Extract Y values for raw curve params."""
        if y_param == "Ia":
            return [p.get("ia", 0.0) for p in points]
        elif y_param == "Ig2":
            return [p.get("ig2", 0.0) for p in points]
        elif y_param == "Pa":
            return [p.get("ua", 0.0) * p.get("ia", 0.0) / _MW_PER_W for p in points]
        elif y_param == "Pig2":
            return [p.get("ug2", 0.0) * p.get("ig2", 0.0) / _MW_PER_W for p in points]
        elif y_param == "Ia/Ig2":
            return [
                p.get("ia", 0.0) / p.get("ig2", 0.0)
                if p.get("ig2", 0.0) > _IG2_NEAR_ZERO_MA else 0.0
                for p in points
            ]
        return [0.0] * len(points)

    def _render_curves_raw_multi_ug2(
        self, points: List[Dict], y_param: str, x_param: str,
        line_width: float, ug2_values=None, pen_style=None,
        is_triode_eff: bool = False,
    ) -> None:
        """Render raw Ia/Ig2/Pa/Pig2 with multi-Ug2 filtering and coloring."""
        ug2_groups = filter_ug2_multi(
            points, is_triode_eff, ug2_values, self.ug2_cluster_thr)
        if not ug2_groups:
            return

        multi = len(ug2_groups) > 1
        if multi:
            rc_pi = self._rc_plot.getPlotItem()
            if getattr(rc_pi, "legend", None) is None:
                rc_pi.addLegend()

        # Transfer accents: Ua-slice bookkeeping (Ua filter store), the
        # "≈Ub" label suffix, and dimming of slices outside the working
        # line's Ua swing. Only for Ia(Ug1) on the Transfer plot.
        is_transfer_ua = (self._rc_plot is self.transfer_plot
                          and x_param == "Ug1")
        swing_lo = swing_hi = None
        if is_transfer_ua:
            analysis = getattr(self, "_load_line_analysis", None) or {}
            pt_neg, pt_pos = analysis.get("pt_neg"), analysis.get("pt_pos")
            if pt_neg and pt_pos:
                swing_lo, swing_hi = sorted((pt_neg["ua"], pt_pos["ua"]))

        global_idx = 0
        for ug2_idx, (ug2_nom, subset) in enumerate(sorted(ug2_groups.items())):
            x_key = "ua" if x_param == "Ua" else "ug1"
            if x_param == "Ua":
                step_key, step_thr = "ug1", self.ug1_cluster_thr
                step_fmt = "ug1"
            else:
                step_key, step_thr = "ua", self.ua_cluster_thr
                step_fmt = "ua"

            step_raw = sorted({round(p.get(step_key, 0.0), 2) for p in subset})
            step_noms = _cluster_nominal_fn(step_raw, threshold=step_thr)
            groups: Dict[float, List[Dict]] = {}
            for p in subset:
                sv = _nominal_key_fn(round(p.get(step_key, 0.0), 2), step_noms)
                groups.setdefault(sv, []).append(p)

            ug2_color = self.palette[ug2_idx % len(self.palette)] if multi else None
            ug2_suffix = f" @ {ug2_nom:.0f}V" if multi else ""

            # Slice closest to Ub gets the "≈Ub" mark — but only when Ub
            # actually falls inside the measured Ua span (a mark on a far
            # slice would claim a reference the data does not reach).
            near_ub = None
            ub_ref = getattr(self, "_rc_ub_ref", None)
            if is_transfer_ua and ub_ref and step_noms:
                if (min(step_noms) - step_thr <= ub_ref
                        <= max(step_noms) + step_thr):
                    near_ub = min(step_noms, key=lambda v: abs(v - ub_ref))

            self._curves_collect = []
            for step_val in sorted(groups.keys()):
                pts = sorted(groups[step_val], key=lambda p: p.get(x_key, 0.0))
                xs = [p[x_key] for p in pts]
                ys = self._raw_y_values(pts, y_param)
                if len(xs) < 2:
                    continue
                extra = self._raw_extra(pts)
                suffix = ug2_suffix
                dim = False
                if is_transfer_ua:
                    if near_ub is not None and step_val == near_ub:
                        suffix = f"{ug2_suffix} {t('plot.Ua_near_ub')}"
                    # Boundary inclusive: a slice AT the swing edge is
                    # part of the working region — not dimmed.
                    if swing_lo is not None and (step_val < swing_lo
                                                 or step_val > swing_hi):
                        dim = True
                items = self._plot_curve_line(
                    xs, ys, global_idx, step_val, line_width,
                    label_fmt=step_fmt, color_override=ug2_color,
                    label_suffix=suffix, extra=extra,
                    pen_style=pen_style, dim=dim)
                if items and is_transfer_ua:
                    self._ua_transfer_items.setdefault(
                        step_val, []).extend(items)
                global_idx += 1

            if multi:
                pen_leg = pg.mkPen(ug2_color, width=line_width)
                leg_item = self._rc_plot.plot([], [], pen=pen_leg,
                                     name=f"Ug2={ug2_nom:.0f}V")
                self._curves_collect.append(leg_item)

            if not is_triode_eff:
                self._rc_ug2_store.setdefault(ug2_nom, []).extend(
                    self._curves_collect)
            self._curves_collect = None

    def _render_curves_raw(self, points: List[Dict], y_param: str,
                           x_param: str, line_width: float,
                           pen_style=None) -> None:
        """Render raw Ia/Ig2/Pa/Pig2 curves (no grid computation needed)."""
        if x_param == "Ug2":
            x_key = "ug2"
            step_key, step_thr = "ug1", self.ug1_cluster_thr
            step_fmt = "ug1"
        elif x_param == "Ua":
            x_key = "ua"
            step_key, step_thr = "ug1", self.ug1_cluster_thr
            step_fmt = "ug1"
        else:
            x_key = "ug1"
            step_key, step_thr = "ua", self.ua_cluster_thr
            step_fmt = "ua"

        step_raw = sorted({round(p.get(step_key, 0.0), 2) for p in points})
        step_noms = _cluster_nominal_fn(step_raw, threshold=step_thr)
        groups: Dict[float, List[Dict]] = {}
        for p in points:
            sv = _nominal_key_fn(round(p.get(step_key, 0.0), 2), step_noms)
            groups.setdefault(sv, []).append(p)

        for idx, step_val in enumerate(sorted(groups.keys())):
            pts = sorted(groups[step_val], key=lambda p: p.get(x_key, 0.0))
            xs = [p[x_key] for p in pts]
            ys = self._raw_y_values(pts, y_param)
            if len(xs) < 2:
                continue
            extra = self._raw_extra(pts)
            self._plot_curve_line(xs, ys, idx, step_val, line_width,
                                      label_fmt=step_fmt, extra=extra,
                                      pen_style=pen_style)

    def _plot_curve_line(self, xs, ys, color_idx: int, step_value: float,
                         line_width: float, label_fmt: str = "ug1",
                         color_override=None, label_suffix: str = "",
                         extra: Optional[Dict[str, np.ndarray]] = None,
                         pen_style=None, dim: bool = False) -> Optional[tuple]:
        """Draw one curve line with endpoint label on the current target plot.

        Returns ``(plot_item, label)`` so callers can register the items
        in per-slice visibility stores; ``None`` when nothing was drawn.
        ``dim=True`` renders the curve and label with reduced alpha
        (out-of-swing Transfer slices).
        """
        if len(xs) < 2:
            return None
        if pen_style is None:
            pen_style = pg.QtCore.Qt.PenStyle.SolidLine
        color = color_override if color_override else self.palette[color_idx % len(self.palette)]
        if dim:
            c = pg.mkColor(color)
            c.setAlpha(_TRANSFER_DIM_ALPHA)
            color = c
        plot_item = self._rc_plot.plot(
            xs, ys,
            pen=pg.mkPen(color, width=line_width, style=pen_style),
            symbol=self.SCAN_SYMBOL, symbolSize=self.CURVES_SYMBOL_SIZE,
            symbolBrush=color,
        )
        label_text = format_label(label_fmt, step_value) + label_suffix
        label = pg.TextItem(
            label_text, color=color, anchor=(0, 1),
        )
        label.setPos(xs[-1], ys[-1])
        self._rc_plot.addItem(label)
        self._rc_labels.append(label)
        if self._curves_collect is not None:
            self._curves_collect.append(plot_item)
            self._curves_collect.append(label)
        self._rc_cd_accum.append(CurveData(
            x=np.array(xs, dtype=float),
            y=np.array(ys, dtype=float),
            label=label_text,
            extra=extra or {},
        ))
        return (plot_item, label)
