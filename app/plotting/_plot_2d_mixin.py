"""2D-plot rendering mixin for ``PlotRenderer``.

The mixin contains:
  - ``render_plot_2d`` and its 14+ private helpers
    (cache build, compare overlay, current-scan modes, overlay items,
    line labels, Ug2 colorbar, visibility filters, incremental render)
  - Q-point family (``draw_qpoint_all`` + ``_clear_qpoint_items``) —
    bundled here because Q-point markers read ``self._load_line_analysis``
    set by ``render_plot_2d``.

Host class (``PlotRenderer``) is responsible for:
  - ``__init__`` setup of every ``self.<attr>`` referenced below
  - class-level constants (``SCAN_SYMBOL``, ``OVERLAY_LW_MIN``, …)
  - ``_cluster_nominal``/``_nominal_key`` static methods
  - ``self.ctx`` (RendererContext) — accessed via property descriptors
    (``self.is_triode``, ``self.track_sids``, ``self.ug1_cluster_thr``,
    ``self.ug2_cluster_thr``)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pyqtgraph as pg

from i18n_setup import t
from lm19.curve_data import CurveData
from lm19.label_formats import format_label
from lm19.plotting.grouping import build_curves_2d as _build_curves_2d_fn
from app.plotting import overlays as _ov


class _Plot2DMixin:
    """Render the main Ia(Ua) 2D plot, its overlays, and Q-points."""

    # Pen styles for compare overlay (index matches QComboBox order).
    # Defined on the mixin so render_compare_overlay can index it via
    # ``self._overlay_pen_styles`` regardless of whether the host class
    # also re-defines it.
    _overlay_pen_styles = [
        pg.QtCore.Qt.PenStyle.SolidLine,
        pg.QtCore.Qt.PenStyle.DashLine,
        pg.QtCore.Qt.PenStyle.DotLine,
        pg.QtCore.Qt.PenStyle.DashDotLine,
    ]

    # ------------------------------------------------------------------
    # Grouping cache
    # ------------------------------------------------------------------

    def _ensure_2d_cache(self, points, track_sids, series_labels):
        """Return cached 2D grouping or compute and cache it."""
        # Validate by object identity (strong ref), NOT id(): a freed list's
        # address can be reused by a new same-length list, so id()+len could
        # serve stale grouping for a different lamp. Holding the list keeps it
        # alive while cached, so `is` is exact.
        if (self._2d_cache is not None
                and self._2d_cache["_plen"] == len(points)
                and self._2d_cache["_points_ref"] is points):
            return self._2d_cache

        _cn = self._cluster_nominal
        _nk = self._nominal_key

        compare_pts = [p for p in points if p.get("series_id", 0) != 0]
        current_pts = [p for p in points if p.get("series_id") == 0]

        ug1_values = []
        ug2_values = []
        by_ug1 = {}
        by_ug1_ug2 = {}

        if current_pts:
            ug1_raw = sorted({round(p.get("ug1", 0.0), 2) for p in current_pts})
            ug1_values = _cn(ug1_raw, threshold=self.ug1_cluster_thr)
            ug2_raw = sorted({round(p.get("ug2", 0.0), 2) for p in current_pts})
            ug2_values = _cn(ug2_raw, threshold=self.ug2_cluster_thr)

            raw_ug1 = {}
            raw_ug1_ug2 = {}
            for p in current_pts:
                g1 = _nk(round(p.get("ug1", 0.0), 2), ug1_values)
                g2 = _nk(round(p.get("ug2", 0.0), 2), ug2_values)
                raw_ug1.setdefault(g1, []).append(p)
                raw_ug1_ug2.setdefault((g1, g2), []).append(p)

            for g1, pts in raw_ug1.items():
                by_ug1[g1] = sorted(pts, key=lambda p: p["ua"])
            for key, pts in raw_ug1_ug2.items():
                by_ug1_ug2[key] = sorted(pts, key=lambda p: p["ua"])

        compare_by_sid = {}
        compare_ug2_values = []
        compare_ug2_noms_by_sid = {}
        compare_ug1_noms_by_sid = {}
        compare_by_ug1_per_sid = {}
        compare_by_ug1_ug2_per_sid = {}

        if compare_pts:
            for p in compare_pts:
                compare_by_sid.setdefault(p.get("series_id", 0), []).append(p)
            for sid, pts in compare_by_sid.items():
                pts.sort(key=lambda p: p["ua"])
                ug1_raw = sorted({round(p.get("ug1", 0.0), 2) for p in pts})
                ug1_noms = _cn(ug1_raw, threshold=self.ug1_cluster_thr)
                compare_ug1_noms_by_sid[sid] = ug1_noms
                ug2_raw = sorted({round(p.get("ug2", 0.0), 2) for p in pts})
                ug2_noms = _cn(ug2_raw, threshold=self.ug2_cluster_thr)
                compare_ug2_noms_by_sid[sid] = ug2_noms
                sid_is_track = sid in track_sids
                sid_by_ug1 = {}
                sid_by_ug1_ug2 = {}
                for p in pts:
                    g1 = _nk(round(p.get("ug1", 0.0), 2), ug1_noms)
                    g2 = _nk(round(p.get("ug2", 0.0), 2), ug2_noms)
                    sid_by_ug1.setdefault(g1, []).append(p)
                    if not sid_is_track:
                        sid_by_ug1_ug2.setdefault((g1, g2), []).append(p)
                for g1, g1pts in sid_by_ug1.items():
                    g1pts.sort(key=lambda p: p["ua"])
                for key, kpts in sid_by_ug1_ug2.items():
                    kpts.sort(key=lambda p: p["ua"])
                compare_by_ug1_per_sid[sid] = sid_by_ug1
                compare_by_ug1_ug2_per_sid[sid] = sid_by_ug1_ug2
            non_track_ug2 = [p for p in compare_pts
                             if p.get("series_id", 0) not in track_sids]
            all_ug2 = sorted({round(p.get("ug2", 0.0), 2) for p in non_track_ug2})
            compare_ug2_values = _cn(all_ug2, threshold=self.ug2_cluster_thr)

        marker_curves = self._build_curves_2d(
            points, track_sids, series_labels)

        self._2d_cache = {
            "_points_ref": points, "_plen": len(points),
            "compare_pts": compare_pts,
            "current_pts": current_pts,
            "ug1_values": ug1_values,
            "ug2_values": ug2_values,
            "by_ug1": by_ug1,
            "by_ug1_ug2": by_ug1_ug2,
            "compare_by_sid": compare_by_sid,
            "compare_ug2_values": compare_ug2_values,
            "compare_ug2_noms_by_sid": compare_ug2_noms_by_sid,
            "compare_ug1_noms_by_sid": compare_ug1_noms_by_sid,
            "compare_by_ug1_per_sid": compare_by_ug1_per_sid,
            "compare_by_ug1_ug2_per_sid": compare_by_ug1_ug2_per_sid,
            "marker_curves": marker_curves,
        }
        return self._2d_cache

    # ------------------------------------------------------------------
    # Compare/overlay rendering
    # ------------------------------------------------------------------

    def _render_compare_overlay(self, cache: dict, series_labels: dict,
                                series_colors: Optional[dict],
                                ug2_mode_series: bool, line_width: float,
                                has_current: bool,
                                overlay_pen_style: int = 0,
                                track_sids: Optional[set] = None,
                                series_models: Optional[dict] = None,
                                series_grids: Optional[dict] = None) -> None:
        """Render compare/overlay series on the 2D plot.

        Each series_id now represents a whole lamp entry with multiple Ug1/Ug2
        curves inside.  We group by Ug1 (and Ug2 for pentodes) and draw
        separate curves per group, using the same color for all curves of the
        same series.
        """
        if track_sids is None:
            track_sids = set()
        compare_by_sid = cache["compare_by_sid"]
        series_ids = sorted(compare_by_sid.keys())
        compare_lw = max(self.OVERLAY_LW_MIN, line_width * self.OVERLAY_LW_FACTOR) if has_current else line_width
        compare_sym = "s" if has_current else self.SCAN_SYMBOL
        compare_sym_size = self.OVERLAY_SYMBOL_SIZE
        pen_style = self._overlay_pen_styles[overlay_pen_style % len(self._overlay_pen_styles)]

        compare_sids = cache.get("compare_by_sid", {}).keys()
        all_track = bool(
            track_sids is not None and compare_sids
            and all(sid in track_sids for sid in compare_sids))
        if not ug2_mode_series and not all_track:
            ug2_values = cache["compare_ug2_values"]
            min_v = min(ug2_values) if ug2_values else 0.0
            max_v = max(ug2_values) if ug2_values else 1.0
            if not has_current:
                self._show_ug2_colorbar(self.cmap, min_v, max_v)
        else:
            ug2_values = []
            min_v = max_v = 0.0

        legend_added_sids = set()
        for idx, series_id in enumerate(series_ids):
            if series_colors and series_id in series_colors:
                base_color = series_colors[series_id]
            else:
                base_color = self.palette[idx % len(self.palette)]

            series_name = series_labels.get(series_id, f"Series {series_id}")
            by_ug1 = cache["compare_by_ug1_per_sid"].get(series_id, {})
            by_ug1_ug2 = cache["compare_by_ug1_ug2_per_sid"].get(series_id, {})
            ug1_noms = cache["compare_ug1_noms_by_sid"].get(series_id, [])
            ug2_noms = cache["compare_ug2_noms_by_sid"].get(series_id, [])

            sid_is_track = series_id in track_sids
            if sid_is_track:
                groups = by_ug1
            else:
                groups = by_ug1_ug2

            is_model = (series_models is not None
                        and series_id in series_models)
            model = series_models.get(series_id) if series_models else None
            grid = (series_grids or {}).get(series_id)

            # Model series: dashed pen, no symbols, dense sampling
            if is_model:
                cur_pen_style = pg.QtCore.Qt.PenStyle.DashLine
                cur_sym = None
                cur_sym_size = 0
            else:
                cur_pen_style = pen_style
                cur_sym = compare_sym
                cur_sym_size = compare_sym_size

            first_curve = series_id not in legend_added_sids
            for key in sorted(groups.keys()):
                pts = groups[key]
                if not pts:
                    continue

                if is_model and model is not None:
                    # Dense rendering via model.ia()
                    ua_min = min(p["ua"] for p in pts)
                    ua_max_v = max(p["ua"] for p in pts)
                    dense_n = self.MODEL_DENSE_SAMPLES
                    xs = np.linspace(max(ua_min, 0.1), ua_max_v, dense_n).tolist()
                    ug1_val = key if not isinstance(key, tuple) else key[0]
                    if sid_is_track and grid:
                        ys = [model.ia(ua, ug1_val,
                                       ua + grid.ug2_offset) for ua in xs]
                    elif isinstance(key, tuple) and len(key) > 1:
                        ug2_v = key[1]
                        ys = [model.ia(ua, ug1_val, ug2_v) for ua in xs]
                    else:
                        ys = [model.ia(ua, ug1_val, 0.0) for ua in xs]
                else:
                    xs = [p["ua"] for p in pts]
                    ys = [p["ia"] for p in pts]

                if not ug2_mode_series and not sid_is_track and not has_current:
                    ug2 = key[1] if isinstance(key, tuple) and len(key) > 1 else 0.0
                    ug2 = self._nominal_key(ug2, ug2_values) if ug2_values else ug2
                    norm = 0.0 if max_v == min_v else (ug2 - min_v) / (max_v - min_v)
                    color = self.cmap.map(norm, mode="qcolor")
                else:
                    color = base_color

                plot_name = series_name if first_curve else None
                if first_curve:
                    legend_added_sids.add(series_id)
                    first_curve = False

                item = self.plot.plot(
                    xs, ys,
                    pen=pg.mkPen(color, width=compare_lw, style=cur_pen_style),
                    symbol=cur_sym, symbolSize=cur_sym_size,
                    symbolBrush=color, name=plot_name,
                )
                self._sid_plot_items.setdefault(series_id, []).append(item)
                if not sid_is_track:
                    ug2_key = key[1] if isinstance(key, tuple) and len(key) > 1 else 0.0
                    if ug2_values:
                        ug2_key = self._nominal_key(ug2_key, ug2_values)
                    self._ug2_plot_items.setdefault(ug2_key, []).append(item)

                if xs and ys:
                    if sid_is_track:
                        ug1_val = key if not isinstance(key, tuple) else key[0]
                        lbl_item = self._make_line_label(
                            xs[-1], ys[-1],
                            format_label("ug1_short", ug1_val), color)
                        self._sid_plot_items.setdefault(series_id, []).append(lbl_item)
                    elif isinstance(key, tuple) and len(key) == 2:
                        ug1_val, ug2_val = key
                        txt = (format_label("ug1_short", ug1_val) + "/"
                               + format_label("ug2_short", ug2_val))
                        lbl_item = self._make_line_label(
                            xs[-1], ys[-1], txt, color)
                        self._ug2_plot_items.setdefault(ug2_key, []).append(lbl_item)
                        self._sid_plot_items.setdefault(series_id, []).append(lbl_item)

    # ------------------------------------------------------------------
    # Public render entry point
    # ------------------------------------------------------------------

    def render_plot_2d(
        self,
        points: list,
        ug2_mode_series: bool,
        series_labels: dict,
        legend_hidden: bool,
        zone_rect: Optional[tuple] = None,
        track_sids: Optional[set] = None,
        line_width: float = 2.0,
        series_colors: Optional[dict] = None,
        pa_max: Optional[float] = None,
        ua_max_limit: Optional[float] = None,
        ia_max_limit: Optional[float] = None,
        overlay_pen_style: int = 0,
        series_models: Optional[dict] = None,
        series_grids: Optional[dict] = None,
        pg2_max: Optional[float] = None,
        pg2_points: Optional[list] = None,
    ) -> None:
        self.plot.clear()
        self._clear_line_labels()
        self._ug2_plot_items.clear()
        self._sid_plot_items.clear()
        self._marker_2d.reattach()

        cache = self._ensure_2d_cache(points, track_sids or set(),
                                      series_labels)
        compare_pts = cache["compare_pts"]
        current_pts = cache["current_pts"]
        has_both = bool(compare_pts) and bool(current_pts)

        plot_item = self.plot.getPlotItem()
        if compare_pts or ug2_mode_series:
            if getattr(plot_item, "legend", None) is None:
                plot_item.addLegend()

        if compare_pts:
            self._render_compare_overlay(
                cache, series_labels, series_colors,
                ug2_mode_series, line_width, has_both, overlay_pen_style,
                track_sids or set(),
                series_models=series_models or {},
                series_grids=series_grids or {},
            )

        if current_pts:
            self._render_current_scan(cache, track_sids or set(),
                                      ug2_mode_series, line_width)
        elif not compare_pts:
            self._hide_ug2_colorbar()

        self.plot.setLabel("bottom", t('plot.Ua_V'))

        legend = getattr(plot_item, "legend", None)
        if legend is not None:
            if legend_hidden:
                legend.hide()
            else:
                legend.show()

        self.plot.setLabel("left", t('plot.Ia_mA'))

        self._render_zone_rect(zone_rect, points)
        self._render_pa_hyperbola(pa_max)
        self._render_pg2_zone(pg2_max, pg2_points)
        self._render_ua_limit(ua_max_limit)
        self._render_ia_limit(ia_max_limit)

        self._all_marker_curves_2d = cache["marker_curves"]
        self._marker_2d.set_curves(list(self._all_marker_curves_2d))

    # ------------------------------------------------------------------
    # CurveData builders for snap-to-curve marker
    # ------------------------------------------------------------------

    def _build_curves_2d(self, points, track_sids=None,
                         series_labels=None, ug1_cluster_thr=None,
                         ug2_cluster_thr=None):
        if ug1_cluster_thr is None:
            ug1_cluster_thr = getattr(self, "ug1_cluster_thr", self.DEFAULT_UG1_CLUSTER_THR)
        if ug2_cluster_thr is None:
            ug2_cluster_thr = getattr(self, "ug2_cluster_thr", self.DEFAULT_UG2_CLUSTER_THR)
        return _build_curves_2d_fn(
            points, track_sids, series_labels,
            ug1_cluster_thr, ug2_cluster_thr)

    # ------------------------------------------------------------------
    # render_plot_2d sub-methods
    # ------------------------------------------------------------------

    def _render_current_scan(self, cache: dict,
                             track_sids: set, ug2_mode_series: bool,
                             line_width: float) -> None:
        """Render current scan data curves on the 2D plot.

        Grouping follows ``track_sids`` alone. The lamp selector
        (``is_triode``) says what is armed next, not how the plotted
        points were taken; the caller resolves that per series and puts
        the verdict in ``track_sids``.
        """
        if 0 in track_sids:
            self._render_current_track_mode(cache, line_width)
        else:
            ug2_values = cache["ug2_values"]
            min_v = min(ug2_values) if ug2_values else 0.0
            max_v = max(ug2_values) if ug2_values else 1.0
            if not ug2_mode_series:
                self._render_current_color_mode(cache, min_v, max_v, line_width)
            else:
                self._render_current_series_mode(cache, line_width)

    def _render_current_track_mode(self, cache: dict,
                                   line_width: float) -> None:
        """Ug2 tracks Ua: group only by Ug1, color by palette."""
        self._hide_ug2_colorbar()
        by_ug1 = cache["by_ug1"]
        ug1_values = cache["ug1_values"]
        for idx, ug1 in enumerate(ug1_values):
            pts = by_ug1.get(ug1, [])
            if not pts:
                continue
            xs = [p["ua"] for p in pts]
            ys = [p["ia"] for p in pts]
            color = self.palette[idx % len(self.palette)]
            item = self.plot.plot(
                xs, ys,
                pen=pg.mkPen(color, width=line_width),
                symbol=self.SCAN_SYMBOL, symbolSize=self.SCAN_SYMBOL_SIZE, symbolBrush=color,
            )
            self._sid_plot_items.setdefault(0, []).append(item)
            self._add_line_label(xs[-1], ys[-1], format_label("ug1", ug1), color)

    def _render_current_color_mode(self, cache: dict,
                                   min_v: float, max_v: float,
                                   line_width: float) -> None:
        """Ug2 as continuous color: Ia(Ua) for each Ug1, colored by Ug2."""
        self._show_ug2_colorbar(self.cmap, min_v, max_v)
        by_ug1_ug2 = cache["by_ug1_ug2"]
        ug1_values = cache["ug1_values"]
        ug2_values = cache["ug2_values"]
        for ug1 in ug1_values:
            label_point = None
            label_color = None
            for ug2 in ug2_values:
                pts = by_ug1_ug2.get((ug1, ug2), [])
                if not pts:
                    continue
                xs = [p["ua"] for p in pts]
                ys = [p["ia"] for p in pts]
                norm = 0.0 if max_v == min_v else (ug2 - min_v) / (max_v - min_v)
                color = self.cmap.map(norm, mode="qcolor")
                item = self.plot.plot(
                    xs, ys,
                    pen=pg.mkPen(color, width=line_width),
                    symbol=self.SCAN_SYMBOL, symbolSize=self.SCAN_SYMBOL_SIZE, symbolBrush=color,
                )
                self._sid_plot_items.setdefault(0, []).append(item)
                self._ug2_plot_items.setdefault(ug2, []).append(item)
                if label_point is None:
                    label_point = (xs[-1], ys[-1])
                    label_color = color
            if label_point:
                lbl = self._make_line_label(
                    label_point[0], label_point[1],
                    format_label("ug1", ug1), label_color,
                )
                for ug2 in ug2_values:
                    if by_ug1_ug2.get((ug1, ug2)):
                        self._ug2_plot_items.setdefault(ug2, []).append(lbl)
                        break

    def _render_current_series_mode(self, cache: dict,
                                    line_width: float) -> None:
        """Ug2 as discrete families: Ia(Ua), distinct color per Ug2."""
        self._hide_ug2_colorbar()
        by_ug1_ug2 = cache["by_ug1_ug2"]
        ug1_values = cache["ug1_values"]
        ug2_values = cache["ug2_values"]
        ug2_legend_added = set()
        for ug1 in ug1_values:
            label_point = None
            label_color = None
            for ug2_idx, ug2 in enumerate(ug2_values):
                pts = by_ug1_ug2.get((ug1, ug2), [])
                if not pts:
                    continue
                xs = [p["ua"] for p in pts]
                ys = [p["ia"] for p in pts]
                color = self.palette[ug2_idx % len(self.palette)]
                name = None
                if ug2 not in ug2_legend_added:
                    name = format_label("ug2", ug2)
                    ug2_legend_added.add(ug2)
                item = self.plot.plot(
                    xs, ys,
                    pen=pg.mkPen(color, width=line_width),
                    symbol=self.SCAN_SYMBOL, symbolSize=self.SCAN_SYMBOL_SIZE, symbolBrush=color,
                    name=name,
                )
                self._sid_plot_items.setdefault(0, []).append(item)
                self._ug2_plot_items.setdefault(ug2, []).append(item)
                if label_point is None:
                    label_point = (xs[-1], ys[-1])
                    label_color = color
            if label_point:
                lbl = self._make_line_label(
                    label_point[0], label_point[1],
                    format_label("ug1", ug1), label_color,
                )
                for ug2 in ug2_values:
                    if by_ug1_ug2.get((ug1, ug2)):
                        self._ug2_plot_items.setdefault(ug2, []).append(lbl)
                        break

    # ------------------------------------------------------------------
    # Overlay items (delegate to overlays.py)
    # ------------------------------------------------------------------

    def _render_zone_rect(self, zone_rect: Optional[tuple], points: list) -> None:
        _ov.draw_zone_rect(self.plot, zone_rect, points)

    def _render_pa_hyperbola(self, pa_max: Optional[float]) -> None:
        _ov.draw_pa_hyperbola(self.plot, pa_max)

    def _render_pg2_zone(self, pg2_max: Optional[float],
                         pg2_points: Optional[list]) -> None:
        _ov.draw_pg2_zone(self.plot, pg2_points or [], pg2_max)

    def _render_ua_limit(self, ua_max_limit: Optional[float]) -> None:
        _ov.draw_ua_limit(self.plot, ua_max_limit)

    def _render_ia_limit(self, ia_max_limit: Optional[float]) -> None:
        _ov.draw_ia_limit(self.plot, ia_max_limit)


    # ------------------------------------------------------------------
    # Incremental scan rendering (live)
    # ------------------------------------------------------------------

    def render_curve_incremental(self, current_curve_points, all_points, event, ug2_mode_series, labeled_ug1, line_width: float = 2.0):
        if not current_curve_points:
            return labeled_ug1
        ug1 = round(event.get("ug1", 0.0), 2)
        ug2 = round(event.get("ug2", 0.0), 2)

        triode_eff = 0 in self.track_sids
        if triode_eff:
            ug1_all = sorted({round(p.get("ug1", 0.0), 2) for p in all_points})
            ug1_noms = self._cluster_nominal(ug1_all, threshold=self.ug1_cluster_thr)
            ug1_key = self._nominal_key(ug1, ug1_noms)
            ug1_idx = ug1_noms.index(ug1_key) if ug1_key in ug1_noms else 0
            color = self.palette[ug1_idx % len(self.palette)]
        elif ug2_mode_series:
            # Discrete families: color by Ug2 index in palette
            ug2_values = sorted({round(p.get("ug2", 0.0), 2) for p in all_points})
            ug2_noms = self._cluster_nominal(ug2_values, threshold=self.ug2_cluster_thr)
            ug2_key = self._nominal_key(ug2, ug2_noms)
            ug2_idx = ug2_noms.index(ug2_key) if ug2_key in ug2_noms else 0
            color = self.palette[ug2_idx % len(self.palette)]
        else:
            ug2_values = sorted({round(p.get("ug2", 0.0), 2) for p in all_points})
            min_v = min(ug2_values) if ug2_values else 0.0
            max_v = max(ug2_values) if ug2_values else 1.0
            norm = 0.0 if max_v == min_v else (ug2 - min_v) / (max_v - min_v)
            color = self.cmap.map(norm, mode="qcolor")

        curve = sorted(current_curve_points, key=lambda p: p["ua"])
        xs = [p["ua"] for p in curve]
        ys = [p["ia"] for p in curve]
        self.plot.plot(
            xs,
            ys,
            pen=pg.mkPen(color, width=line_width),
            symbol=self.SCAN_SYMBOL,
            symbolSize=self.OVERLAY_SYMBOL_SIZE,
            symbolBrush=color,
        )
        if ug1 not in labeled_ug1 and xs and ys:
            self._add_line_label(xs[-1], ys[-1], format_label("ug1", ug1), color)
            labeled_ug1.add(ug1)
        if not triode_eff and not ug2_mode_series:
            self._show_ug2_colorbar(self.cmap, min_v, max_v)
        return labeled_ug1

    # ------------------------------------------------------------------
    # Visibility / removal
    # ------------------------------------------------------------------

    def set_ug2_visibility(self, checked_ug2: list, ug2_cluster_thr: float) -> None:
        """Show/hide 2D and Curves plot items by Ug2.

        Transfer items go through the combined Ug2 × Ua predicate
        (:meth:`_apply_transfer_visibility`) — a direct ``setVisible``
        here would clobber the Ua filter and vice versa.
        """
        for store in (self._ug2_plot_items, self._ug2_curves_items):
            if not store:
                continue
            for ug2_nom, items in store.items():
                visible = any(abs(ug2_nom - v) <= ug2_cluster_thr
                              for v in checked_ug2)
                for item in items:
                    item.setVisible(visible)
        self._transfer_ug2_filter = (list(checked_ug2), ug2_cluster_thr)
        self._apply_transfer_visibility()
        # Filter marker curves to match visible Ug2 values
        self._apply_ug2_marker_filter(checked_ug2, ug2_cluster_thr)

    def set_transfer_ua_visibility(self, checked_ua: list,
                                   ua_cluster_thr: float) -> None:
        """Show/hide Transfer plot items by Ua slice (Ua-filter combo)."""
        self._transfer_ua_filter = (list(checked_ua), ua_cluster_thr)
        self._apply_transfer_visibility()

    def _apply_transfer_visibility(self) -> None:
        """Recompute Transfer item visibility from BOTH filters (AND).

        An item may sit in the Ug2 store only (Ug2 legend stubs), the Ua
        store only (triode/track slices), or both — a missing filter side
        defaults to visible.

        Attribute reads are defensive (``getattr``): some tests build the
        renderer via ``object.__new__`` without ``__init__`` (same pattern
        as ``_get_ctx``).
        """
        allowed: Dict = {}
        for store, filt in (
            (getattr(self, "_ug2_transfer_items", None),
             getattr(self, "_transfer_ug2_filter", None)),
            (getattr(self, "_ua_transfer_items", None),
             getattr(self, "_transfer_ua_filter", None)),
        ):
            if not store:
                continue
            for nom, items in store.items():
                ok = (filt is None
                      or any(abs(nom - v) <= filt[1] for v in filt[0]))
                for item in items:
                    allowed[item] = allowed.get(item, True) and ok
        for item, ok in allowed.items():
            item.setVisible(ok)
        self._apply_transfer_marker_filter()

    def _apply_transfer_marker_filter(self) -> None:
        """Transfer snap-marker curves: keep only Ug2- AND Ua-visible."""
        curves = getattr(self, "_all_marker_curves_transfer", None)
        marker = getattr(self, "_marker_transfer", None)
        if not marker or not curves:
            return
        ug2 = getattr(self, "_transfer_ug2_filter", None)
        ua = getattr(self, "_transfer_ua_filter", None)
        filtered = [
            c for c in curves
            if (ug2 is None or self._curve_ug2_visible(c, ug2[0], ug2[1]))
            and (ua is None or self._curve_ua_visible(c, ua[0], ua[1]))
        ]
        marker.set_curves(filtered)

    def _apply_ug2_marker_filter(self, checked_ug2: list,
                                  ug2_cluster_thr: float) -> None:
        """Update 2D/Curves marker curves to exclude hidden Ug2 values.

        The Transfer marker is handled by
        :meth:`_apply_transfer_marker_filter` (needs the Ua filter too).
        """
        for attr_curves, attr_marker in (
            ("_all_marker_curves_2d", "_marker_2d"),
            ("_all_marker_curves_curves", "_marker_curves"),
        ):
            all_curves = getattr(self, attr_curves, None)
            marker = getattr(self, attr_marker, None)
            if not marker or not all_curves:
                continue
            filtered = [
                c for c in all_curves
                if self._curve_ug2_visible(c, checked_ug2, ug2_cluster_thr)
            ]
            marker.set_curves(filtered)

    def remove_series_items(self, sids: set) -> None:
        """Remove plot items for given series_ids from 2D plot."""
        for sid in sids:
            for item in self._sid_plot_items.pop(sid, []):
                self.plot.removeItem(item)
        # Update marker curves
        all_curves = getattr(self, "_all_marker_curves_2d", None)
        if all_curves and self._marker_2d:
            filtered = [c for c in all_curves
                        if c.extra.get("series_id", 0) not in sids]
            self._all_marker_curves_2d = filtered
            self._marker_2d.set_curves(filtered)

    def set_sid_visibility(self, visible_sids: set) -> None:
        """Show/hide 2D plot items by series_id."""
        for sid, items in self._sid_plot_items.items():
            vis = sid in visible_sids
            for item in items:
                item.setVisible(vis)
        # Update marker curves to match
        all_curves = getattr(self, "_all_marker_curves_2d", None)
        if all_curves and self._marker_2d:
            filtered = [
                c for c in all_curves
                if c.extra.get("series_id", 0) in visible_sids
            ]
            self._marker_2d.set_curves(filtered)

    @staticmethod
    def _curve_ug2_visible(curve: CurveData, checked_ug2: list,
                           thr: float) -> bool:
        """Check if a CurveData's Ug2 matches any checked value."""
        ug2_arr = curve.extra.get("Ug2")
        if ug2_arr is None or len(ug2_arr) == 0:
            return True  # no Ug2 info → always visible
        # Use the representative Ug2 value (first element, since grouped curves
        # have a consistent nominal Ug2)
        ug2_val = float(ug2_arr[0])
        return any(abs(ug2_val - v) <= thr for v in checked_ug2)

    @staticmethod
    def _curve_ua_visible(curve: CurveData, checked_ua: list,
                          thr: float) -> bool:
        """Check if a CurveData's Ua slice matches any checked value.

        Transfer curves are grouped by Ua nominal, so the first element
        is representative (within the cluster threshold).
        """
        ua_arr = curve.extra.get("Ua")
        if ua_arr is None or len(ua_arr) == 0:
            return True  # no Ua info → always visible
        ua_val = float(ua_arr[0])
        return any(abs(ua_val - v) <= thr for v in checked_ua)

    # ------------------------------------------------------------------
    # Line labels + Ug2 colorbar
    # ------------------------------------------------------------------

    def _clear_line_labels(self) -> None:
        for item in self.line_labels:
            self.plot.removeItem(item)
        self.line_labels = []

    def _add_line_label(self, x: float, y: float, text: str, color) -> None:
        self._make_line_label(x, y, text, color)

    def _make_line_label(self, x: float, y: float, text: str, color):
        """Create, add and return a TextItem label on the 2D plot."""
        label = pg.TextItem(text, color=color, anchor=(0, 1))
        label.setPos(x, y)
        self.plot.addItem(label)
        self.line_labels.append(label)
        return label

    def _show_ug2_colorbar(self, cmap, min_v: float, max_v: float) -> None:
        if self.ug2_colorbar is None:
            self._ug2_colorbar_image = pg.ImageItem()
            self._ug2_colorbar_image.setLevels((min_v, max_v))
            self.ug2_colorbar = pg.ColorBarItem(
                values=(min_v, max_v),
                colorMap=cmap,
                label=t('plot.Ug2_V'),
            )
            self.ug2_colorbar.setImageItem(
                self._ug2_colorbar_image,
                insert_in=self.plot.getPlotItem(),
            )
        else:
            self.ug2_colorbar.setLevels((min_v, max_v))
        self.ug2_colorbar.show()

    def _hide_ug2_colorbar(self) -> None:
        if self.ug2_colorbar:
            self.ug2_colorbar.hide()

    # ------------------------------------------------------------------
    # Q-point markers (read self._load_line_analysis from render_plot_2d)
    # ------------------------------------------------------------------

    def draw_qpoint_all(self) -> None:
        """Draw Q-point + swing markers on heatmaps and range on Transfer/Curves.

        Reads from self._load_line_analysis (set by render_plot_2d).
        Safe to call even if analysis is None (clears old markers).
        """
        self._clear_qpoint_items()
        analysis = getattr(self, "_load_line_analysis", None)
        if not analysis:
            return

        # Heatmaps: Q-point + swing endpoints (Ua × Ug1 axes)
        heatmap_plots = [
            ("contour", self.contour_plot),
            ("gm", self.gm_plot),
            ("rp", self.rp_plot),
            ("mu", self.mu_plot),
            ("pa_map", self.pa_map_plot),
        ]
        for key, plot in heatmap_plots:
            if plot is not None:
                items = _ov.draw_qpoint_on_heatmap(plot, analysis)
                if items:
                    self._qpoint_items[key] = items

        # Transfer: swing range (X = Ug1)
        if self.transfer_plot is not None:
            items = _ov.draw_swing_range_lines(
                self.transfer_plot, analysis, x_param="Ug1")
            if items:
                self._qpoint_items["transfer"] = items

    def _clear_qpoint_items(self) -> None:
        """Remove all Q-point/swing items from all plots."""
        plot_map = {
            "contour": self.contour_plot,
            "gm": self.gm_plot,
            "rp": self.rp_plot,
            "mu": self.mu_plot,
            "pa_map": self.pa_map_plot,
            "transfer": self.transfer_plot,
        }
        for key, items in self._qpoint_items.items():
            plot = plot_map.get(key)
            if plot is not None:
                for item in items:
                    try:
                        plot.removeItem(item)
                    except (RuntimeError, ReferenceError):
                        # Qt: item already removed / parent destroyed.
                        # AttributeError etc. (programming bugs) propagate.
                        pass
        self._qpoint_items.clear()
