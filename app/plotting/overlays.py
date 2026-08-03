"""Overlay drawing functions for the 2D Ia(Ua) plot.

Each function takes a ``pg.PlotWidget`` (or its PlotItem) as the first
argument and draws directly into it.  This keeps ``PlotRenderer`` slim
and gives a clear list of all overlay types.
"""
import logging
import pyqtgraph as pg
import numpy as np
from typing import List, Dict, Optional
from i18n_setup import t
from lm19.label_formats import format_label
from lm19.plot_style import (
    PLOT_PADDING, COLOR_LIMIT, COLOR_PG2, COLOR_LOAD_LINE, COLOR_QPOINT,
    DANGER_FILL_ALPHA,
    COLOR_SWING, COLOR_ZONE, QPOINT_SIZE,
)

log = logging.getLogger(__name__)


def _danger_brush():
    """Translucent over-limit fill DERIVED from COLOR_LIMIT (ML-022/023:
    was a raw pure-red RGBA literal duplicated in three zones)."""
    c = pg.mkColor(COLOR_LIMIT)
    c.setAlpha(DANGER_FILL_ALPHA)
    return pg.mkBrush(c)


def draw_zone_rect(plot: pg.PlotWidget, zone_rect: Optional[dict],
                   points: List[Dict]) -> None:
    """Draw zone rectangle vertical lines and adjust X range."""
    if not zone_rect or zone_rect.get("ua_min") is None or zone_rect.get("ua_max") is None:
        return
    ua_min = float(zone_rect["ua_min"])
    ua_max = float(zone_rect["ua_max"])
    pen = pg.mkPen(COLOR_ZONE, width=1, style=pg.QtCore.Qt.PenStyle.DashLine)
    plot.addItem(pg.InfiniteLine(pos=ua_min, angle=90, pen=pen))
    plot.addItem(pg.InfiniteLine(pos=ua_max, angle=90, pen=pen))
    if points:
        ua_vals = [p.get("ua") for p in points if "ua" in p]
        if ua_vals:
            x_lo = min(min(ua_vals), ua_min)
            x_hi = max(max(ua_vals), ua_max)
            plot.getPlotItem().getViewBox().setXRange(x_lo, x_hi, padding=PLOT_PADDING)


def draw_pa_hyperbola(plot: pg.PlotWidget, pa_max: Optional[float]) -> None:
    """Draw Pa_max hyperbola and filled danger zone."""
    if pa_max is None or pa_max <= 0:
        return
    vb = plot.getPlotItem().getViewBox()
    x_range = vb.viewRange()[0]
    y_range = vb.viewRange()[1]
    ua_lo = max(x_range[0], 1.0)
    ua_hi = max(x_range[1], ua_lo + 1.0)
    ia_top = y_range[1]
    ua_arr = np.linspace(ua_lo, ua_hi, 300)
    ia_limit = pa_max * 1000.0 / ua_arr
    ia_limit = np.clip(ia_limit, 0, ia_top)
    ia_ceil = np.full_like(ua_arr, ia_top)
    curve_limit = pg.PlotDataItem(ua_arr, ia_limit)
    curve_ceil = pg.PlotDataItem(ua_arr, ia_ceil)
    fill = pg.FillBetweenItem(curve_limit, curve_ceil,
                              brush=_danger_brush())
    plot.addItem(fill)
    pen_pa = pg.mkPen(COLOR_LIMIT, width=1.5,
                      style=pg.QtCore.Qt.PenStyle.DashLine)
    plot.plot(ua_arr, ia_limit, pen=pen_pa)


def draw_pg2_zone(plot: pg.PlotWidget, points: List[Dict],
                  pg2_max: Optional[float]) -> None:
    """Draw Pg2_max exceedance zone built from actual measurement data.

    For the selected Ug2 slice, computes Pg2 = Ug2 × Ig2 / 1000 at each
    point.  On each Ug1 curve finds the Ua boundary where Pg2 crosses
    *pg2_max* and fills the danger region (orange).
    """
    if pg2_max is None or pg2_max <= 0 or not points:
        return
    # Group points by Ug1 (rounded to 0.1 V)
    from lm19.plotting.grids import cluster_nominal, nominal_key
    from lm19.constants import UG1_CLUSTER_THR, UG1_ROUND

    ug1_raw = sorted({round(p.get("ug1", 0.0), UG1_ROUND) for p in points})
    ug1_noms = cluster_nominal(ug1_raw, threshold=UG1_CLUSTER_THR)
    if not ug1_noms:
        return

    curves: Dict[float, List[Dict]] = {}
    for p in points:
        ig2 = p.get("ig2", 0.0)
        ug2 = p.get("ug2", 0.0)
        pg2 = ug2 * ig2 / 1000.0
        ug1_nom = nominal_key(round(p.get("ug1", 0.0), UG1_ROUND), ug1_noms)
        curves.setdefault(ug1_nom, []).append({
            "ua": p["ua"], "ia": p.get("ia", 0.0), "pg2": pg2,
        })
    # Sort each curve by Ua
    for pts in curves.values():
        pts.sort(key=lambda x: x["ua"])

    # Find boundary points: for each Ug1 curve, interpolate where Pg2
    # crosses pg2_max.  In pentodes, Pg2 is typically highest at low Ua
    # (knee region) and decreases with Ua.
    boundary_pts: List[tuple] = []  # (ua, ia) on the boundary
    over_pts: List[tuple] = []      # all points exceeding limit

    for ug1_nom in sorted(curves.keys()):
        pts = curves[ug1_nom]
        if len(pts) < 2:
            continue
        curve_over = [p for p in pts if p["pg2"] > pg2_max]
        for p in curve_over:
            over_pts.append((p["ua"], p["ia"]))

        # Walk the curve to find crossing(s)
        has_crossing = False
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            over0 = p0["pg2"] > pg2_max
            over1 = p1["pg2"] > pg2_max
            if over0 != over1:
                d0 = p0["pg2"] - pg2_max
                d1 = p1["pg2"] - pg2_max
                if abs(d1 - d0) > 1e-12:
                    frac = d0 / (d0 - d1)
                    ua_cross = p0["ua"] + frac * (p1["ua"] - p0["ua"])
                    ia_cross = p0["ia"] + frac * (p1["ia"] - p0["ia"])
                    boundary_pts.append((ua_cross, ia_cross))
                    has_crossing = True

        if not has_crossing and curve_over:
            # Entire curve exceeds Pg2_max — use its rightmost point
            # so the filled zone extends to cover the whole curve.
            rightmost = max(curve_over, key=lambda p: p["ua"])
            boundary_pts.append((rightmost["ua"], rightmost["ia"]))

    if not over_pts:
        # Failure visibility: silent return when feature is enabled would
        # leave the user puzzled why nothing is drawn. Surface the reason.
        max_pg2 = max((p["pg2"] for pts in curves.values() for p in pts),
                      default=0.0)
        ug2_vals = sorted({round(p.get("ug2", 0.0), 1) for p in points})
        log.warning(
            "Pg2 zone enabled (Pg2_max=%.2f W) but no measured points "
            "exceed limit on current Ug2 slice "
            "(max Pg2 in data=%.3f W, slice Ug2=%s V)",
            pg2_max, max_pg2, ug2_vals,
        )
        return

    vb = plot.getPlotItem().getViewBox()
    x_range = vb.viewRange()[0]
    ua_lo = max(x_range[0], 0.0)

    # --- Filled polygon: boundary → left edge closure ---
    if len(boundary_pts) >= 2:
        boundary_pts.sort(key=lambda p: p[1])  # sort by Ia (bottom→top)
        # FillBetweenItem between boundary and left wall
        left_x = [ua_lo] * len(boundary_pts)
        left_y = [p[1] for p in boundary_pts]
        bnd_x = [p[0] for p in boundary_pts]
        bnd_y = [p[1] for p in boundary_pts]
        curve_bnd = pg.PlotDataItem(bnd_x, bnd_y)
        curve_left = pg.PlotDataItem(left_x, left_y)
        _pg2_rgb = pg.mkColor(COLOR_PG2).getRgb()[:3]
        fill = pg.FillBetweenItem(curve_left, curve_bnd,
                                  brush=pg.mkBrush(*_pg2_rgb, 30))
        plot.addItem(fill)
        # Dashed boundary line
        pen_pg2 = pg.mkPen(pg.mkColor(*_pg2_rgb, 200), width=1.5,
                           style=pg.QtCore.Qt.PenStyle.DashLine)
        plot.plot(bnd_x, bnd_y, pen=pen_pg2)

    # --- Orange markers on individual over-limit points ---
    _pg2_rgb = pg.mkColor(COLOR_PG2).getRgb()[:3]
    xs = [p[0] for p in over_pts]
    ys = [p[1] for p in over_pts]
    plot.plot(
        xs, ys, pen=None,
        symbol="o", symbolSize=6,
        symbolBrush=pg.mkBrush(*_pg2_rgb, 160),
        symbolPen=pg.mkPen(pg.mkColor(*_pg2_rgb, 200), width=1),
    )


def draw_ua_limit(plot: pg.PlotWidget, ua_max_limit: Optional[float]) -> None:
    """Draw Ua_max vertical danger zone."""
    if ua_max_limit is None or ua_max_limit <= 0:
        return
    vb = plot.getPlotItem().getViewBox()
    x_range = vb.viewRange()[0]
    y_range = vb.viewRange()[1]
    ua_hi = x_range[1]
    if ua_max_limit >= ua_hi:
        return
    ia_lo = y_range[0]
    ia_hi = y_range[1]
    ia_arr = np.array([ia_lo, ia_hi])
    ua_left = np.full_like(ia_arr, ua_max_limit)
    ua_right = np.full_like(ia_arr, ua_hi)
    curve_left = pg.PlotDataItem(ua_left, ia_arr)
    curve_right = pg.PlotDataItem(ua_right, ia_arr)
    fill_ua = pg.FillBetweenItem(curve_left, curve_right,
                                 brush=_danger_brush())
    plot.addItem(fill_ua)
    pen_ua = pg.mkPen(COLOR_LIMIT, width=1.5,
                      style=pg.QtCore.Qt.PenStyle.DashLine)
    plot.addItem(pg.InfiniteLine(pos=ua_max_limit, angle=90, pen=pen_ua))


def draw_ia_limit(plot: pg.PlotWidget, ia_max_limit: Optional[float]) -> None:
    """Draw Ia_max horizontal danger zone."""
    if ia_max_limit is None or ia_max_limit <= 0:
        return
    vb = plot.getPlotItem().getViewBox()
    x_range = vb.viewRange()[0]
    y_range = vb.viewRange()[1]
    ia_hi = y_range[1]
    if ia_max_limit >= ia_hi:
        return
    ua_lo = x_range[0]
    ua_hi = x_range[1]
    ua_arr = np.array([ua_lo, ua_hi])
    ia_bottom = np.full_like(ua_arr, ia_max_limit)
    ia_top = np.full_like(ua_arr, ia_hi)
    curve_bottom = pg.PlotDataItem(ua_arr, ia_bottom)
    curve_top = pg.PlotDataItem(ua_arr, ia_top)
    fill_ia = pg.FillBetweenItem(curve_bottom, curve_top,
                                 brush=_danger_brush())
    plot.addItem(fill_ia)
    pen_ia = pg.mkPen(COLOR_LIMIT, width=1.5,
                      style=pg.QtCore.Qt.PenStyle.DashLine)
    plot.addItem(pg.InfiniteLine(pos=ia_max_limit, angle=0, pen=pen_ia))


def draw_analysis_markers(plot: pg.PlotWidget, analysis: Dict,
                          line_labels: list) -> None:
    """Draw Q-point, swing range markers, and half-point markers."""
    q_ua, q_ia = analysis["ua_0"], analysis["ia_0"]
    plot.plot(
        [q_ua], [q_ia], pen=None,
        symbol="o", symbolSize=QPOINT_SIZE,
        symbolBrush=pg.mkBrush(pg.mkColor(COLOR_QPOINT).getRgb()[:3] + (180,)),
        symbolPen=pg.mkPen("w", width=2),
    )
    lbl_q = pg.TextItem(
        "Q", color=COLOR_QPOINT,
        anchor=(0.5, -0.5),
    )
    lbl_q.setFont(pg.QtGui.QFont("", 9, pg.QtGui.QFont.Weight.Bold))
    lbl_q.setPos(q_ua, q_ia)
    plot.addItem(lbl_q)
    line_labels.append(lbl_q)

    pt_neg = analysis["pt_neg"]
    pt_pos = analysis["pt_pos"]
    swing_xs = [pt_neg["ua"], pt_pos["ua"]]
    swing_ys = [pt_neg["ia"], pt_pos["ia"]]
    _sw_rgb = pg.mkColor(COLOR_SWING).getRgb()[:3]
    plot.plot(
        swing_xs, swing_ys, pen=None,
        symbol="t", symbolSize=10,
        symbolBrush=pg.mkBrush(*_sw_rgb, 200),
        symbolPen=pg.mkPen("w", width=1),
    )

    pen_sw = pg.mkPen(pg.mkColor(*_sw_rgb, 160), width=4)
    plot.plot(swing_xs, swing_ys, pen=pen_sw)

    pt_lh = analysis["pt_low_half"]
    pt_hh = analysis["pt_high_half"]
    half_xs = [pt_lh["ua"], pt_hh["ua"]]
    half_ys = [pt_lh["ia"], pt_hh["ia"]]
    plot.plot(
        half_xs, half_ys, pen=None,
        symbol="s", symbolSize=7,
        symbolBrush=pg.mkBrush(*_sw_rgb, 140),
        symbolPen=pg.mkPen("w", width=1),
    )


# ── Q-point + swing markers for heatmap / line plots ──────────────

_SWING_SIZE = 8
_SWING_LINE_ALPHA = 140


def draw_qpoint_on_heatmap(
    plot: pg.PlotWidget,
    analysis: Dict,
) -> list:
    """Draw Q-point and swing endpoints on a heatmap (axes Ua × Ug1).

    Returns list of items added (caller must store for later removal).
    """
    items: list = []
    ua_q = analysis.get("ua_0")
    ug1_q = analysis.get("ug1_0")
    if ua_q is None or ug1_q is None:
        return items

    # Swing endpoints
    pt_neg = analysis.get("pt_neg")
    pt_pos = analysis.get("pt_pos")
    if pt_neg and pt_pos:
        xs = [pt_neg["ua"], ua_q, pt_pos["ua"]]
        ys = [pt_neg["ug1"], ug1_q, pt_pos["ug1"]]
        # Connecting line
        _hm_sw_rgb = pg.mkColor(COLOR_SWING).getRgb()[:3]
        pen = pg.mkPen(pg.mkColor(*_hm_sw_rgb, _SWING_LINE_ALPHA), width=2.5)
        line = pg.PlotCurveItem(xs, ys, pen=pen)
        line.setZValue(899)
        plot.addItem(line)
        items.append(line)
        # Swing markers (triangles)
        swing_sc = pg.ScatterPlotItem(
            [pt_neg["ua"], pt_pos["ua"]],
            [pt_neg["ug1"], pt_pos["ug1"]],
            size=_SWING_SIZE, symbol="t",
            brush=pg.mkBrush(*_hm_sw_rgb, 200),
            pen=pg.mkPen("w", width=1),
        )
        swing_sc.setZValue(900)
        plot.addItem(swing_sc)
        items.append(swing_sc)

    # Q-point marker (red dot)
    q_sc = pg.ScatterPlotItem(
        [ua_q], [ug1_q], size=QPOINT_SIZE,
        brush=pg.mkBrush(COLOR_QPOINT),
        pen=pg.mkPen("w", width=2),
    )
    q_sc.setZValue(901)
    plot.addItem(q_sc)
    items.append(q_sc)

    # "Q" label
    lbl = pg.TextItem("Q", color=COLOR_QPOINT, anchor=(0.5, -0.5))
    lbl.setFont(pg.QtGui.QFont("", 9, pg.QtGui.QFont.Weight.Bold))
    lbl.setPos(ua_q, ug1_q)
    lbl.setZValue(902)
    plot.addItem(lbl)
    items.append(lbl)

    return items


def draw_swing_range_lines(
    plot: pg.PlotWidget,
    analysis: Dict,
    x_param: str = "Ug1",
) -> list:
    """Draw vertical swing-range lines on Transfer or Curves plots.

    x_param: "Ug1" or "Ua" — determines which analysis keys to use.
    Returns list of added items.
    """
    items: list = []
    pt_neg = analysis.get("pt_neg")
    pt_pos = analysis.get("pt_pos")
    if not pt_neg or not pt_pos:
        return items

    key = "ug1" if x_param == "Ug1" else "ua"
    x_lo = min(pt_neg[key], pt_pos[key])
    x_hi = max(pt_neg[key], pt_pos[key])

    # Shaded region between swing limits
    _sr_rgb = pg.mkColor(COLOR_SWING).getRgb()[:3]
    fill_pen = pg.mkPen(pg.mkColor(*_sr_rgb, 40))
    fill_brush = pg.mkBrush(pg.mkColor(*_sr_rgb, 25))
    region = pg.LinearRegionItem(
        values=(x_lo, x_hi), movable=False,
        pen=fill_pen, brush=fill_brush,
    )
    region.setZValue(50)
    plot.addItem(region)
    items.append(region)

    # Q-point vertical line
    q_val = analysis.get("ug1_0") if x_param == "Ug1" else analysis.get("ua_0")
    if q_val is not None:
        q_pen = pg.mkPen(COLOR_QPOINT, width=1.5,
                         style=pg.QtCore.Qt.PenStyle.DashLine)
        q_line = pg.InfiniteLine(pos=q_val, angle=90, pen=q_pen)
        q_line.setZValue(51)
        plot.addItem(q_line)
        items.append(q_line)

    return items
