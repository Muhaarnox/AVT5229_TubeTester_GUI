"""Grouping and CurveData builders for snap-to-curve markers.

Extracts clustering, grouping, and CurveData construction from PlotRenderer
so the logic is testable without Qt.

Re-exports ``cluster_nominal`` and ``nominal_key`` from :mod:`.grids` for
convenience.
"""


from __future__ import annotations

import numpy as np
from typing import List, Dict

from lm19.plotting.grids import cluster_nominal, nominal_key
from lm19.curve_data import CurveData
from lm19.label_formats import format_label
from lm19.constants import UG1_CLUSTER_THR, UG2_CLUSTER_THR, UG1_ROUND, UG2_ROUND


def build_curves_2d(
    points: List[Dict],
    track_sids: set = None,
    series_labels: Dict = None,
    ug1_cluster_thr: float = UG1_CLUSTER_THR,
    ug2_cluster_thr: float = UG2_CLUSTER_THR,
) -> List[CurveData]:
    """Build CurveData list from 2D plot points, grouped by (Ug1, Ug2, series).

    *track_sids* — set of series_id values where Ug2 tracks Ua (triode /
    triode-connected).  For those series Ug2 is collapsed to 0 in the
    grouping key so all points fall into one curve per Ug1.
    """
    if not points:
        return []
    if track_sids is None:
        track_sids = set()
    if series_labels is None:
        series_labels = {}
    curves: list = []

    _cn, _nk = cluster_nominal, nominal_key

    ug1_raw = sorted({round(p.get("ug1", 0.0), UG1_ROUND) for p in points})
    ug1_noms = _cn(ug1_raw, threshold=ug1_cluster_thr)
    ug2_raw = sorted({round(p.get("ug2", 0.0), UG2_ROUND) for p in points})
    ug2_noms = _cn(ug2_raw, threshold=ug2_cluster_thr)

    groups: Dict[tuple, List[Dict]] = {}
    for p in points:
        sid = p.get("series_id", 0)
        ug1 = _nk(round(p.get("ug1", 0.0), UG1_ROUND), ug1_noms)
        skip_ug2 = sid in track_sids
        ug2 = 0 if skip_ug2 else _nk(round(p.get("ug2", 0.0), UG2_ROUND), ug2_noms)
        key = (sid, ug1, ug2)
        groups.setdefault(key, []).append(p)

    for (sid, ug1, ug2), pts in groups.items():
        pts_sorted = sorted(pts, key=lambda p: p["ua"])
        xs = np.array([p["ua"] for p in pts_sorted], dtype=float)
        ys = np.array([p["ia"] for p in pts_sorted], dtype=float)
        if len(xs) < 2:
            continue
        label = ""
        if sid != "" and sid is not None:
            sname = series_labels.get(sid, "")
            if sname:
                base = sname.split(" Ug1=")[0]
                label = base
            else:
                lamp_type = pts_sorted[0].get("lamp_type", "")
                lamp_id = pts_sorted[0].get("lamp_id", "")
                label = f"{lamp_type}/{lamp_id}" if lamp_type else f"#{sid}"
        extra: dict = {"series_id": sid}
        n = len(pts_sorted)
        extra["Ug1"] = np.full(n, ug1, dtype=float)
        ug2_arr = np.array([p.get("ug2", 0.0) for p in pts_sorted], dtype=float)
        extra["Ug2"] = ug2_arr
        ig2_arr = np.array([p.get("ig2", 0.0) for p in pts_sorted], dtype=float)
        if np.any(ig2_arr != 0):
            extra["Ig2"] = ig2_arr
        curves.append(CurveData(x=xs, y=ys, label=label, extra=extra))
    return curves


def build_curves_transfer(
    points: List[Dict],
    is_triode: bool,
    ug2_cluster_thr: float = UG2_CLUSTER_THR,
) -> List[CurveData]:
    """Build CurveData for Transfer plot: Ia(Ug1) grouped by Ua."""
    if not points:
        return []
    _cn, _nk = cluster_nominal, nominal_key
    curves: list = []
    ug2_raw = sorted({round(p.get("ug2", 0.0), UG2_ROUND) for p in points})
    ug2_noms = _cn(ug2_raw, threshold=ug2_cluster_thr) if not is_triode else [0.0]
    groups: Dict[tuple, List[Dict]] = {}
    for p in points:
        ua = round(p.get("ua", 0.0), 0)
        ug2 = _nk(round(p.get("ug2", 0.0), UG2_ROUND), ug2_noms) if not is_triode else 0
        key = (ua, ug2)
        groups.setdefault(key, []).append(p)

    for (ua, ug2), pts in groups.items():
        pts_sorted = sorted(pts, key=lambda p: p.get("ug1", 0.0))
        xs = np.array([p.get("ug1", 0.0) for p in pts_sorted], dtype=float)
        ys = np.array([p["ia"] for p in pts_sorted], dtype=float)
        if len(xs) < 2:
            continue
        parts = [format_label("ua", ua)]
        if not is_triode and ug2 != 0:
            parts.append(format_label("ug2", ug2))
        curves.append(CurveData(x=xs, y=ys, label=" | ".join(parts)))
    return curves


def build_curves_ig2(
    points: List[Dict],
    ug1_cluster_thr: float = UG1_CLUSTER_THR,
) -> List[CurveData]:
    """Build CurveData for Ig2 plot: Ig2(Ua) grouped by Ug1."""
    if not points:
        return []
    _cn, _nk = cluster_nominal, nominal_key
    curves: list = []
    ug1_raw = sorted({round(p.get("ug1", 0.0), UG1_ROUND) for p in points})
    ug1_noms = _cn(ug1_raw, threshold=ug1_cluster_thr)
    groups: Dict[float, List[Dict]] = {}
    for p in points:
        ug1 = _nk(round(p.get("ug1", 0.0), UG1_ROUND), ug1_noms)
        groups.setdefault(ug1, []).append(p)

    for ug1, pts in groups.items():
        pts_sorted = sorted(pts, key=lambda p: p["ua"])
        xs = np.array([p["ua"] for p in pts_sorted], dtype=float)
        ys = np.array([p.get("ig2", 0.0) for p in pts_sorted], dtype=float)
        if len(xs) < 2:
            continue
        label = format_label("ug1", ug1)
        extra: dict = {}
        ia_arr = np.array([p.get("ia", 0.0) for p in pts_sorted], dtype=float)
        if np.any(ia_arr != 0):
            extra["Ia"] = ia_arr
        ug2_arr = np.array([p.get("ug2", 0.0) for p in pts_sorted], dtype=float)
        if np.any(ug2_arr != 0):
            extra["Ug2"] = ug2_arr
        curves.append(CurveData(x=xs, y=ys, label=label, extra=extra))
    return curves
