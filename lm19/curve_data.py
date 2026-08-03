"""Pure-data curve helpers used by CurveMarker (UI) and pure logic.

No Qt / pyqtgraph dependency — safe to import from anywhere in ``lm19/``.
The interactive ``CurveMarker`` widget that consumes these structures lives
in ``app/curve_marker.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from lm19.constants import UG1_CLUSTER_THR, UG2_CLUSTER_THR
# ML-135: canonical clustering helpers live in lm19/plotting/grids.py —
# aliased imports instead of a private copy (single-source ratchet in
# tests/test_dedup_guards.py).
from lm19.plotting.grids import cluster_nominal as _cluster_nominal
from lm19.plotting.grids import nominal_key as _nominal_key


@dataclass
class CurveData:
    """One curve for a marker to snap to.

    Attributes:
        x:     sorted array of X values (e.g. Ua or Ug1)
        y:     corresponding Y values (e.g. Ia or Ig2)
        label: curve identifier shown in tooltip (e.g. "Ug1 -2.0 V")
        extra: additional interpolatable arrays, e.g. {"Ig2": np.array}
    """
    x: np.ndarray
    y: np.ndarray
    label: str = ""
    extra: Dict[str, np.ndarray] = field(default_factory=dict)


# Predefined field configs for different plot types
FIELDS_2D = {
    "x_name": "Ua", "x_unit": "V", "x_fmt": ".1f",
    "y_name": "Ia", "y_unit": "mA", "y_fmt": ".2f",
    "show_pa": True,
    "extra": [
        {"key": "Ug1", "name": "Ug1", "unit": "V", "fmt": ".1f"},
        {"key": "Ug2", "name": "Ug2", "unit": "V", "fmt": ".0f"},
        {"key": "Ig2", "name": "Ig2", "unit": "mA", "fmt": ".2f"},
    ],
}
FIELDS_TRANSFER = {
    "x_name": "Ug1", "x_unit": "V", "x_fmt": ".2f",
    "y_name": "Ia", "y_unit": "mA", "y_fmt": ".2f",
    "show_pa": False,
}
FIELDS_IG2 = {
    "x_name": "Ua", "x_unit": "V", "x_fmt": ".1f",
    "y_name": "Ig2", "y_unit": "mA", "y_fmt": ".2f",
    "show_pa": False,
    "extra": [
        {"key": "Ia", "name": "Ia", "unit": "mA", "fmt": ".2f"},
        {"key": "Ug2", "name": "Ug2", "unit": "V", "fmt": ".0f"},
    ],
}
FIELDS_COMPARE = {
    "x_name": "Ua", "x_unit": "V", "x_fmt": ".1f",
    "y_name": "Ia", "y_unit": "mA", "y_fmt": ".2f",
    "show_pa": True,
    "extra": [
        {"key": "Ug1", "name": "Ug1", "unit": "V", "fmt": ".1f"},
        {"key": "Ug2", "name": "Ug2", "unit": "V", "fmt": ".0f"},
        {"key": "Ig2", "name": "Ig2", "unit": "mA", "fmt": ".2f"},
    ],
}


_CURVES_Y_META = {
    "Gm":    {"name": "Gm",    "unit": "mA/V", "fmt": ".2f"},
    "Rp":    {"name": "Rp",    "unit": "kΩ",   "fmt": ".2f"},
    "mu":    {"name": "μ",     "unit": "",      "fmt": ".1f"},
    "Ia":    {"name": "Ia",    "unit": "mA",    "fmt": ".2f"},
    "Ig2":   {"name": "Ig2",   "unit": "mA",    "fmt": ".2f"},
    "Pa":    {"name": "Pa",    "unit": "W",     "fmt": ".3f"},
    "Pig2":  {"name": "Pig2",  "unit": "W",     "fmt": ".3f"},
    "Ia/Ig2":{"name": "Ia/Ig2","unit": "",      "fmt": ".1f"},
}
_CURVES_X_META = {
    "Ua":  {"name": "Ua",  "unit": "V", "fmt": ".1f"},
    "Ug1": {"name": "Ug1", "unit": "V", "fmt": ".2f"},
    "Ug2": {"name": "Ug2", "unit": "V", "fmt": ".0f"},
}


_ALL_EXTRA = [
    {"key": "Ua",  "name": "Ua",  "unit": "V",  "fmt": ".1f"},
    {"key": "Ug1", "name": "Ug1", "unit": "V",  "fmt": ".2f"},
    {"key": "Ug2", "name": "Ug2", "unit": "V",  "fmt": ".0f"},
    {"key": "Ia",  "name": "Ia",  "unit": "mA", "fmt": ".2f"},
    {"key": "Ig2", "name": "Ig2", "unit": "mA", "fmt": ".2f"},
    {"key": "Pa",  "name": "Pa",  "unit": "W",  "fmt": ".3f"},
]


def build_curves_fields(y_param: str, x_param: str) -> Dict:
    """Build a CurveMarker fields dict for given Curves tab parameters.

    All extra fields are always included; CurveMarker silently skips
    any extras not populated in CurveData.
    """
    ym = _CURVES_Y_META.get(y_param, _CURVES_Y_META["Ia"])
    xm = _CURVES_X_META.get(x_param, _CURVES_X_META["Ua"])
    return {
        "x_name": xm["name"], "x_unit": xm["unit"], "x_fmt": xm["fmt"],
        "y_name": ym["name"], "y_unit": ym["unit"], "y_fmt": ym["fmt"],
        "show_pa": False,
        "extra": list(_ALL_EXTRA),
    }


# ------------------------------------------------------------------
# Standalone curve builders (testable without Qt widgets)
# ------------------------------------------------------------------

DEFAULT_UG1_CLUSTER_THR = UG1_CLUSTER_THR
DEFAULT_UG2_CLUSTER_THR = UG2_CLUSTER_THR


def build_compare_curves(points: List[Dict],
                         ug1_cluster_thr: float = DEFAULT_UG1_CLUSTER_THR,
                         ug2_cluster_thr: float = DEFAULT_UG2_CLUSTER_THR,
                         ) -> List[CurveData]:
    """Build CurveData from compare points grouped by (entry, lamp, ug1, ug2).

    Pure function — no Qt dependency. Used by CompareTab._build_compare_curves.
    Auto-detects triode-connected mode per entry (ug2 varies with ua) so mixed
    overlay data works correctly. Uses ``_entry_idx`` to distinguish measurements.
    """
    if not points:
        return []

    # Build cluster nominals once for consistent grouping
    ug1_raw = sorted({round(p.get("ug1", 0.0), 2) for p in points})
    ug1_noms = _cluster_nominal(ug1_raw, threshold=ug1_cluster_thr)
    ug2_raw = sorted({round(p.get("ug2", 0.0), 2) for p in points})
    ug2_noms = _cluster_nominal(ug2_raw, threshold=ug2_cluster_thr)

    # Per-entry auto-detection: which entries have ug2 tracking ua?
    pre_groups: Dict[tuple, List[Dict]] = {}
    for p in points:
        eidx = p.get("_entry_idx", 0)
        ug1 = _nominal_key(round(p.get("ug1", 0.0), 2), ug1_noms)
        pre_groups.setdefault((eidx, ug1), []).append(p)

    track_entries: set = set()  # entry indices where ug2 should be ignored
    entry_vary: Dict[int, list] = {}  # eidx -> [varies, total]
    for (eidx, ug1), pts in pre_groups.items():
        if eidx not in entry_vary:
            entry_vary[eidx] = [0, 0]
        if len(pts) < 2:
            continue
        entry_vary[eidx][1] += 1
        ug2_clustered = {_nominal_key(round(p.get("ug2", 0.0), 2), ug2_noms) for p in pts}
        if len(ug2_clustered) > len(pts) * 0.5:
            entry_vary[eidx][0] += 1
    for eidx, (varies, total) in entry_vary.items():
        if total > 0 and varies > total * 0.5:
            track_entries.add(eidx)

    groups: Dict[tuple, List[Dict]] = {}
    for p in points:
        eidx = p.get("_entry_idx", 0)
        lt = p.get("lamp_type", "")
        lid = p.get("lamp_id", "")
        ug1 = _nominal_key(round(p.get("ug1", 0.0), 2), ug1_noms)
        ug2 = 0 if eidx in track_entries else _nominal_key(round(p.get("ug2", 0.0), 2), ug2_noms)
        key = (eidx, lt, lid, ug1, ug2)
        groups.setdefault(key, []).append(p)

    curves = []
    for (eidx, lt, lid, ug1, ug2), pts in groups.items():
        pts_sorted = sorted(pts, key=lambda p: p["ua"])
        xs = np.array([p["ua"] for p in pts_sorted], dtype=float)
        ys = np.array([p["ia"] for p in pts_sorted], dtype=float)
        if len(xs) < 2:
            continue
        # Label = source name only (Ug1/Ug2 go into extra)
        entry_name = pts_sorted[0].get("_entry_name", "")
        label = entry_name or (f"{lt}/{lid}" if lt else lid)
        n = len(pts_sorted)
        extra = {}
        extra["Ug1"] = np.full(n, ug1, dtype=float)
        ug2_arr = np.array([p.get("ug2", 0.0) for p in pts_sorted], dtype=float)
        extra["Ug2"] = ug2_arr
        ig2_arr = np.array([p.get("ig2", 0.0) for p in pts_sorted], dtype=float)
        if np.any(ig2_arr != 0):
            extra["Ig2"] = ig2_arr
        curves.append(CurveData(x=xs, y=ys, label=label, extra=extra))
    return curves
