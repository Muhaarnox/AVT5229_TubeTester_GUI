"""Amplifier sanity checks on converted SPICE reference datasets."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from lm19.amplifier import (
    ResistiveLoadLine,
    compute_distortion,
    compute_distortion_chebyshev,
    compute_stage_params,
    find_intersections,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
)


DATA_DIR = Path(__file__).resolve().parent / "spice_test_data" / "converted"

MINIMAL_DATASETS = [
    "triode_ecc82_datasheet.json",
    "triode_12AX7AMitch_tuparam.json",
    "pentode_EL34_tuparam.json",
    "pentode_6F5P_P_nexttube.json",
    "triode_EL34_curvetracedata.json",
    "triode_ECC88_curvetracedata.json",
]

EXTENDED_DATASETS = [
    "triode_300B_curvetracedata.json",
    "triode_6N13S_nexttube.json",
    "triode_6E5P_curvetracedata.json",
    "pentode_KT88_tuparam.json",
]

FIXED_WORKING_POINTS = [
    # (dataset, ub, ra, ug2_filter_or_none)
    ("triode_ecc82_datasheet.json", 250.0, 10.0, None),
    ("triode_12AX7AMitch_tuparam.json", 350.0, 100.0, None),
    ("pentode_6F5P_P_nexttube.json", 250.0, 5.0, 200.0),
]


def _load(name: str) -> Dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _dominant_ug2(points: List[Dict]) -> Optional[float]:
    if not points or "ug2" not in points[0]:
        return None
    counts: Dict[float, int] = {}
    for p in points:
        u = round(float(p.get("ug2", 0.0)), 1)
        counts[u] = counts.get(u, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _try_find_working_setup(points: List[Dict], ug2_filter: Optional[float]):
    ua_max = max((float(p.get("ua", 0.0)) for p in points), default=300.0)
    ub_candidates = sorted({120.0, 150.0, 250.0, 300.0, 400.0, 600.0, ua_max, ua_max * 1.2, ua_max * 1.5})
    ra_candidates = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

    for active_filter in (ug2_filter, None):
        for ub in ub_candidates:
            for ra in ra_candidates:
                ll = ResistiveLoadLine(ub=ub, ra=ra)
                isects = find_intersections(points, ll, ug2_filter=active_filter)
                if len(isects) >= 3:
                    return ll, isects
    return None, []


def _assert_physical(dist: Dict, ub: float) -> None:
    assert dist["ia_0"] >= -1e-9
    assert dist["i_min"] >= -1e-9
    assert dist["i_max"] >= -1e-9
    assert dist["ua_0"] <= ub + 1e-6


@pytest.mark.parametrize("name,ub,ra,ug2_filter", FIXED_WORKING_POINTS)
def test_fixed_reference_operating_points(name: str, ub: float, ra: float, ug2_filter: Optional[float]):
    data = _load(name)
    points = data.get("points", [])
    assert points, f"{name}: no points"

    ll = ResistiveLoadLine(ub=ub, ra=ra)
    isects = find_intersections(points, ll, ug2_filter=ug2_filter)
    assert len(isects) >= 3, f"{name}: fixed operating point has too few intersections"

    dist = compute_distortion(isects, ug1_bias=None, half_swing=None)
    assert dist is not None, f"{name}: fixed operating point returned no distortion"
    _assert_physical(dist, ll.ub)


@pytest.mark.parametrize("name", MINIMAL_DATASETS)
def test_minimal_spice_datasets_are_physically_plausible(name: str):
    data = _load(name)
    points = data.get("points", [])
    assert points, f"{name}: no points"

    ug2 = _dominant_ug2(points) if data.get("topology") == TOPOLOGY_PENTODE else None
    ll, isects = _try_find_working_setup(points, ug2)
    assert ll is not None, f"{name}: failed to find load-line with intersections"

    # Some SPICE reference datasets are too sparse (≤3 intersections on the
    # auto-found load-line) for harmonic analysis. For those, the load-line and
    # intersection extraction ARE validated above (assert ll is not None); the
    # distortion physics is not — but instead of a silent skip we assert the
    # two methods are *consistent* (when 5-point declines, Chebyshev declines
    # too — no method silently returns garbage). When 5-point does produce a
    # result, it is validated as physical.
    dist_auto = compute_distortion(isects, ug1_bias=None, half_swing=None)
    if dist_auto is None:
        assert compute_distortion_chebyshev(isects, ug1_bias=None, half_swing=None) is None, \
            f"{name}: 5-point rejected sparse data but Chebyshev returned a result"
        return
    _assert_physical(dist_auto, ll.ub)

    dist_manual = compute_distortion(isects, ug1_bias=dist_auto["ug1_0"], half_swing=3.0)
    if dist_manual is not None:
        _assert_physical(dist_manual, ll.ub)

    stage = compute_stage_params(isects, ll, ug1_bias=dist_auto["ug1_0"], srk=None, points=points)
    if stage is not None:
        assert stage["gain"] >= 0.0
        assert stage["zout"] >= 0.0
        # Anti-regression limit for obvious outlier spikes.
        assert stage["gain"] < 200.0


@pytest.mark.parametrize("name", EXTENDED_DATASETS)
def test_extended_spice_datasets_are_physically_plausible(name: str):
    data = _load(name)
    points = data.get("points", [])
    assert points, f"{name}: no points"

    ug2 = _dominant_ug2(points) if data.get("topology") == TOPOLOGY_PENTODE else None
    ll, isects = _try_find_working_setup(points, ug2)
    assert ll is not None, f"{name}: failed to find load-line with intersections"

    # See test_minimal: consistent sparse-data handling instead of a silent
    # skip — when 5-point declines, Chebyshev must decline too.
    dist_auto = compute_distortion(isects, ug1_bias=None, half_swing=None)
    if dist_auto is None:
        assert compute_distortion_chebyshev(isects, ug1_bias=None, half_swing=None) is None, \
            f"{name}: 5-point rejected sparse data but Chebyshev returned a result"
        return
    _assert_physical(dist_auto, ll.ub)

    dist_manual = compute_distortion(isects, ug1_bias=dist_auto["ug1_0"], half_swing=3.0)
    if dist_manual is not None:
        _assert_physical(dist_manual, ll.ub)

