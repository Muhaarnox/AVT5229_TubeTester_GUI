"""Working line of the output stage — kinked PP geometry.

Geometry of `pp_working_line_ia` / `working_line_polyline` and the
kinked PP branches of `find_intersections` (data) and
`find_intersections_model` (model twin). THD paths (composite/
joint-solve) are NOT touched — this is the display/diagnostic layer.

Run:  py -m pytest tests/test_working_line.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lm19.amplifier import (
    CathodeFollowerLoadLine,
    PushPullLoadLine,
    ResistiveLoadLine,
    TransformerLoadLine,
    compute_headroom,
    find_intersections,
    find_intersections_model,
    pp_working_line_ia,
    working_line_polyline,
)

# ── module local constants ──
# Non-symmetric EL84-class case: Ub=300, Ra_aa=8k, Iq=36 mA ->
# Z2=4, Z4=2, kink (156, 72), cutoff 444 (probed values).
_UB = 300.0
_RAA = 8.0
_IQ = 36.0
_Z2 = _RAA / 2.0
_Z4 = _RAA / 4.0
_KINK_UA = _UB - _IQ * _Z2          # 156.0
_CUTOFF_UA = _UB + _IQ * _Z2        # 444.0


def _fn(ua: float) -> float:
    return float(pp_working_line_ia(ua, _UB, _IQ, _RAA))


# ═══════════════════════════════════════════════════════════════════
#  pp_working_line_ia geometry
# ═══════════════════════════════════════════════════════════════════

class TestPpWorkingLineGeometry:

    def test_value_at_q_and_cutoff(self):
        assert _fn(_UB) == pytest.approx(_IQ)
        assert _fn(_CUTOFF_UA) == pytest.approx(0.0, abs=1e-9)

    def test_kink_exactly_at_partner_cutoff(self):
        """Kink exactly at (Ua_q - Iq*Z2, 2*Iq) — not at Iq, not Z4."""
        assert _fn(_KINK_UA) == pytest.approx(2.0 * _IQ)

    def test_continuity_at_kink(self):
        eps = 1e-6
        assert _fn(_KINK_UA - eps) == pytest.approx(_fn(_KINK_UA + eps),
                                                    abs=1e-3)

    def test_segment_slopes_both_sides(self):
        """Both ends: slope -1/Z2 above the kink, -1/Z4 below."""
        d = 1.0
        slope_a = (_fn(_UB + d) - _fn(_UB - d)) / (2 * d)
        assert slope_a == pytest.approx(-1.0 / _Z2, rel=1e-9)
        ua_b = _KINK_UA / 2.0
        slope_b = (_fn(ua_b + d) - _fn(ua_b - d)) / (2 * d)
        assert slope_b == pytest.approx(-1.0 / _Z4, rel=1e-9)

    def test_straight_z4_line_disagrees_in_the_middle(self):
        """Discriminates the old straight -1/Z4 line through Q: in the
        middle of class A the difference is material (a symmetric
        point would coincide)."""
        ua_mid = (_KINK_UA + _UB) / 2.0            # inside class A
        old_straight = _IQ - (ua_mid - _UB) / _Z4
        assert abs(_fn(ua_mid) - old_straight) > 10.0

    def test_array_matches_scalar_loop(self):
        ua = np.linspace(0.0, 500.0, 101)
        vec = pp_working_line_ia(ua, _UB, _IQ, _RAA)
        scal = np.array([_fn(u) for u in ua])
        assert np.array_equal(np.asarray(vec, dtype=float), scal)

    def test_degenerate_ra_zero_returns_q_current(self):
        assert float(pp_working_line_ia(100.0, _UB, _IQ, 0.0)) == _IQ

    def test_beyond_cutoff_goes_negative_not_clamped(self):
        """Negative space: beyond cutoff the formula goes negative —
        the sign-change intersection search relies on this."""
        assert _fn(_CUTOFF_UA + 50.0) < 0.0


# ═══════════════════════════════════════════════════════════════════
#  working_line_polyline
# ═══════════════════════════════════════════════════════════════════

class TestWorkingLinePolyline:

    def test_pp_three_vertices_formula(self):
        poly = working_line_polyline(PushPullLoadLine(ub=_UB, ra_aa=_RAA),
                                     q_ua=_UB, q_ia=_IQ)
        assert len(poly) == 3
        assert poly[1] == (pytest.approx(_KINK_UA), pytest.approx(2 * _IQ))
        assert poly[2] == (pytest.approx(_CUTOFF_UA), pytest.approx(0.0))
        assert poly[0][0] == 0.0
        assert poly[0][1] == pytest.approx(2 * _IQ + _KINK_UA / _Z4)

    def test_pp_polyline_equivalent_to_fn_on_vertices_and_midpoints(self):
        """Polyline <-> function equivalence: vertices AND segment
        midpoints (segments match, not just the nodes)."""
        poly = working_line_polyline(PushPullLoadLine(ub=_UB, ra_aa=_RAA),
                                     q_ua=_UB, q_ia=_IQ)
        checks = list(poly)
        for (x0, y0), (x1, y1) in zip(poly, poly[1:]):
            checks.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
        for ua, ia in checks:
            assert _fn(ua) == pytest.approx(ia, abs=1e-9), ua

    def test_pp_degenerate_iq_zero_pure_class_b(self):
        poly = working_line_polyline(PushPullLoadLine(ub=_UB, ra_aa=_RAA),
                                     q_ua=_UB, q_ia=0.0)
        assert poly == [(0.0, pytest.approx(_UB / _Z4)),
                        (pytest.approx(_UB), 0.0)]

    def test_pp_kink_left_of_zero_truncated(self):
        """Huge Iq: the kink is left of ua=0 — class-A segment only."""
        iq = 100.0                                  # kink = 300−400 < 0
        poly = working_line_polyline(PushPullLoadLine(ub=_UB, ra_aa=_RAA),
                                     q_ua=_UB, q_ia=iq)
        assert len(poly) == 2
        assert poly[0] == (0.0, pytest.approx(iq + _UB / _Z2))
        assert poly[1] == (pytest.approx(_UB + iq * _Z2), 0.0)

    def test_pp_without_q_returns_empty(self):
        assert working_line_polyline(
            PushPullLoadLine(ub=_UB, ra_aa=_RAA)) == []

    def test_resistive_and_cf(self):
        assert working_line_polyline(ResistiveLoadLine(ub=250.0, ra=10.0)) \
            == [(0.0, 25.0), (250.0, 0.0)]
        assert working_line_polyline(
            CathodeFollowerLoadLine(ub=250.0, rk=10.0, rl=15.0)) \
            == [(0.0, 10.0), (250.0, 0.0)]

    def test_transformer_ac_through_q(self):
        ll = TransformerLoadLine(ub=250.0, ra_dc=0.2, ra_ac=5.0)
        poly = working_line_polyline(ll, q_ua=245.0, q_ia=48.0)
        assert poly == [(0.0, pytest.approx(48.0 + 245.0 / 5.0)),
                        (pytest.approx(245.0 + 48.0 * 5.0), 0.0)]
        assert working_line_polyline(ll) == []     # no Q — nothing to draw

    def test_degenerate_impedances_empty(self):
        assert working_line_polyline(ResistiveLoadLine(ub=250.0, ra=0.0)) == []
        assert working_line_polyline(
            PushPullLoadLine(ub=_UB, ra_aa=0.0), q_ua=_UB, q_ia=_IQ) == []


# ═══════════════════════════════════════════════════════════════════
#  Kinked intersections — data branch and the model twin
# ═══════════════════════════════════════════════════════════════════

# Flat curves (Ia = const per curve) -> analytic intersections:
#   c <= 2*Iq:  ua = q_ua + (Iq - c)*Z2      (class A)
#   c >  2*Iq:  ua = kink_ua - (c - 2*Iq)*Z4 (class B)
_BIAS = -5.0
_RA_DC = 0.1
_LEVELS = {-2.0: 100.0, _BIAS: _IQ, -8.0: 20.0}   # mA; -2 -> class B


def _flat_points() -> List[Dict]:
    pts = []
    for ug1, c in _LEVELS.items():
        for ua in range(0, 501, 2):
            pts.append({"ua": float(ua), "ug1": ug1, "ia": c, "ug2": 0.0})
    return pts


class _FlatModel:
    """Flat-curve model (twin of the data set); array-safe."""

    def ia(self, ua: float, ug1: float, ug2: float = 0.0) -> float:
        return _LEVELS.get(round(ug1, 1), 0.0)

    def ia_array(self, ua, ug1, ug2=0.0):
        ua_b, ug1_b = np.broadcast_arrays(np.asarray(ua, dtype=float),
                                          np.asarray(ug1, dtype=float))
        out = np.zeros_like(ug1_b, dtype=float)
        for lvl, c in _LEVELS.items():
            out = np.where(np.isclose(ug1_b, lvl), c, out)
        return out


def _expected_ua(c: float, q_ua: float, q_ia: float) -> float:
    if c <= 2.0 * q_ia:
        return q_ua + (q_ia - c) * _Z2
    return (q_ua - q_ia * _Z2) - (c - 2.0 * q_ia) * _Z4


class TestPpIntersectionsKinked:

    def _q(self):
        # DC line (Ub, ra_dc) x flat bias curve: Iq=36 ->
        # q_ua = Ub − 36·0.1 = 296.4
        return _UB - _IQ * _RA_DC, _IQ

    def test_data_branch_matches_analytic_kinked(self):
        ll = PushPullLoadLine(ub=_UB, ra_aa=_RAA, ra_dc=_RA_DC)
        isects = find_intersections(_flat_points(), ll, ug1_bias=_BIAS)
        got = {round(p["ug1"], 1): p["ua"] for p in isects}
        q_ua, q_ia = self._q()
        for ug1, c in _LEVELS.items():
            exp = _expected_ua(c, q_ua, q_ia)
            assert got[round(ug1, 1)] == pytest.approx(exp, abs=0.05), (
                f"ug1={ug1}: got {got.get(round(ug1, 1))}, want {exp}")

    def test_data_branch_discriminates_old_straight_line(self):
        """Class-B curve (-2 V, 100 mA): the old straight line would
        give q_ua + (Iq-c)*Z4 — tens of volts right of the true kink."""
        ll = PushPullLoadLine(ub=_UB, ra_aa=_RAA, ra_dc=_RA_DC)
        isects = find_intersections(_flat_points(), ll, ug1_bias=_BIAS)
        got = {round(p["ug1"], 1): p["ua"] for p in isects}
        q_ua, q_ia = self._q()
        old_straight = q_ua + (q_ia - 100.0) * _Z4
        assert abs(got[-2.0] - old_straight) > 20.0

    def test_model_twin_matches_analytic_kinked(self):
        ll = PushPullLoadLine(ub=_UB, ra_aa=_RAA, ra_dc=_RA_DC)
        isects = find_intersections_model(
            _FlatModel(), ll, list(_LEVELS.keys()), ug2=0.0,
            ug1_bias=_BIAS)
        got = {round(p["ug1"], 1): p["ua"] for p in isects}
        q_ua, q_ia = self._q()
        for ug1, c in _LEVELS.items():
            exp = _expected_ua(c, q_ua, q_ia)
            assert got[round(ug1, 1)] == pytest.approx(exp, abs=0.5), (
                f"ug1={ug1}: got {got.get(round(ug1, 1))}, want {exp}")

    def test_model_twin_reach_extends_by_z2(self):
        """Kink cutoff sits on Z2 (farther than the old line Z4 reach):
        a deep intersection beyond (q_ua+Iq*Z4)*margin and beyond the
        default 500 V ceiling must be found, not silently cut."""
        iq = 100.0
        levels = {_BIAS: iq, -9.0: 10.0}
        # analytic: ua_x = q_ua + (100-10)*4 ~ 660 — beyond 500 and
        # beyond the Z4 reach (~525), inside the Z2 reach (~735)

        class _M(_FlatModel):
            def ia(self, ua, ug1, ug2=0.0):
                return levels.get(round(ug1, 1), 0.0)

            def ia_array(self, ua, ug1, ug2=0.0):
                ua_b, ug1_b = np.broadcast_arrays(
                    np.asarray(ua, dtype=float),
                    np.asarray(ug1, dtype=float))
                out = np.zeros_like(ug1_b, dtype=float)
                for lvl, c in levels.items():
                    out = np.where(np.isclose(ug1_b, lvl), c, out)
                return out

        ll = PushPullLoadLine(ub=_UB, ra_aa=_RAA, ra_dc=_RA_DC)
        isects = find_intersections_model(
            _M(), ll, [-9.0], ug2=0.0, ug1_bias=_BIAS)
        assert isects, "deep-bias intersection cut off by reach"
        q_ua = _UB - iq * _RA_DC
        exp = _expected_ua(10.0, q_ua, iq)
        assert isects[0]["ua"] == pytest.approx(exp, abs=1.0)

    def test_headroom_consumer_sees_kinked_positions(self):
        """The consumer (compute_headroom) receives kinked isects —
        edge-curve currents are honest."""
        ll = PushPullLoadLine(ub=_UB, ra_aa=_RAA, ra_dc=_RA_DC)
        isects = find_intersections(_flat_points(), ll, ug1_bias=_BIAS)
        hr = compute_headroom(isects, _BIAS)
        assert hr is not None
        assert hr["ia_max"] == pytest.approx(100.0)
        assert hr["ia_min"] == pytest.approx(20.0)
