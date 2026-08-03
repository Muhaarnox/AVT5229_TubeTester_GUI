"""Joint-solve PP trajectory on the working line.

PP + model: line, intersection markers and swing geometry all come
from the joint pair solve REGARDLESS of the HD method; every plot
feeds from existing view fields (polyline/intersections/geometry).
No model / mismatched pair / diverged solve: display line + UI note.

The stub model does not depend on Ua (Ia = gm*(Ug1 - cutoff)):
trajectory slopes are analytically exact and discriminate a straight
Zaa/4 line: class A (both conduct) has dUa/dIa = -Zaa/2;
class B (partner cut off): dUa/dIa = -Zaa/4 through (Ua_q, 0), not Q.

Pins:
  1.  Slopes Zaa/2 / Zaa/4, kink at partner cutoff, pure class A
      without a kink, cutoff end ABOVE the supply, exact grid
      ordinates.
  2.  Discriminator: joint trajectory in the view even at
      hd_method=5POINT (data) — not only for dft.
  3.  polyline/intersections/swing_geometry substituted (markers ON
      the trajectory, DIFFERENT from the straight line).
  4-5. Fallbacks (display + note); controller kink item + info line.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.amplifier import pp_joint_trajectory
from lm19.amplifier.constants import CIRCUIT_PP, HD_METHOD_5POINT
from lm19.amplifier.loadlines import PushPullLoadLine
from lm19.amp_engine import AmplifierEngine, AmpParams, WorkingLineView
from lm19.constants import TOPOLOGY_PENTODE, TOPOLOGY_TRIODE

pytestmark = [pytest.mark.smoke_analysis]

# ── Module local helpers ──

_GM = 2.0        # mA/V
_CUTOFF = -20.0  # V
UB = 250.0
RAA = 8.0        # kΩ → ra_per_tube = 2, Zaa/2 = 4
BIAS = -10.0     # Iq = 20 mA; the partner cuts off at drive +10 V


class _LinearTube:
    """Ia = gm*(Ug1 - cutoff): depends on neither Ua nor Ug2."""

    topology = TOPOLOGY_TRIODE
    model_type = "stub"
    name = "linear stub"

    @staticmethod
    def ia(ua: float, ug1: float, ug2: float = 0.0) -> float:
        return max(0.0, _GM * (ug1 - _CUTOFF))


def _ll() -> PushPullLoadLine:
    return PushPullLoadLine(ub=UB, ra_aa=RAA, ra_dc=0.1)


def _traj(swing: float, n: int = 65):
    return pp_joint_trajectory(_LinearTube(), _ll(), BIAS, swing, 0.0,
                               n_points=n)


def _slope(p0: Dict, p1: Dict) -> float:
    return (p1["ua"] - p0["ua"]) / (p1["ia"] - p0["ia"])


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ======================================================================
# 1. Pure trajectory physics
# ======================================================================


class TestTrajectoryPhysics:

    def test_class_a_slope_is_zaa_over_2(self):
        """Near Q both tubes conduct: slope -Zaa/2 = -4 kOhm — twice
        as steep as the straight display line (-Zaa/4 = -2 kOhm)."""
        pts = _traj(swing=8.0)["points"]
        n = len(pts)
        s = _slope(pts[n // 2 - 2], pts[n // 2 + 2])
        assert s == pytest.approx(-RAA / 2.0, rel=1e-3)

    def test_no_kink_in_pure_class_a(self):
        """Swing 8 < 10 (partner cutoff): pure class A — no kink."""
        assert _traj(swing=8.0)["kink"] is None

    def test_kink_at_partner_cutoff_and_class_b_slope(self):
        """Swing 12 > 10: kink near ug1_a ~ 0 (partner cut off), then
        slope -Zaa/4."""
        out = _traj(swing=12.0)
        kink = out["kink"]
        assert kink is not None
        assert kink["ug1"] == pytest.approx(0.0, abs=0.5)
        pts = out["points"]
        after = [p for p in pts if p["ug1"] > kink["ug1"] + 0.5]
        assert len(after) >= 2
        s = _slope(after[0], after[-1])
        assert s == pytest.approx(-RAA / 4.0, rel=1e-3)

    def test_cutoff_side_anode_flies_above_supply(self):
        """Cutoff end: the partner pulls the anode ABOVE the supply
        via the transformer (a line through Q understates this)."""
        pts = _traj(swing=8.0)["points"]
        assert pts[0]["ug1"] == pytest.approx(BIAS - 8.0)
        assert pts[0]["ua"] > UB  # 248 + 64 = 312 V for the stub

    def test_grid_contains_exact_geometry_ordinates(self):
        """65-point grid: indices 0, n//4, n//2, 3n//4, n-1 are exactly
        drives -s, -s/2, 0, +s/2, +s (geometry with no interpolation)."""
        pts = _traj(swing=8.0)["points"]
        n = len(pts)
        assert pts[n // 4]["ug1"] == pytest.approx(BIAS - 4.0)
        assert pts[n // 2]["ug1"] == pytest.approx(BIAS)
        assert pts[3 * n // 4]["ug1"] == pytest.approx(BIAS + 4.0)
        assert [p["ug1"] for p in pts] == sorted(p["ug1"] for p in pts)

    def test_degenerate_inputs_return_none(self):
        assert _traj(swing=0.01) is None
        bad = PushPullLoadLine(ub=UB, ra_aa=0.0, ra_dc=0.1)
        assert pp_joint_trajectory(_LinearTube(), bad, BIAS, 8.0) is None


class _QuadraticTube:
    """Ia = c*(Ug1 - cutoff)^2, independent of Ua.

    Discriminates the joint solve against the ANALYTIC polyline
    pp_working_line_ia (display path): for a square-law tube the
    class-A trajectory is curved (the analytic draws a straight Zaa/2)
    and the partner cuts off at ia_a = 4*Iq, not the analytic 2*Iq.
    """

    topology = TOPOLOGY_TRIODE
    model_type = "stub_quad"
    name = "quadratic stub"
    C = 0.05

    @classmethod
    def ia(cls, ua: float, ug1: float, ug2: float = 0.0) -> float:
        d = max(0.0, ug1 - _CUTOFF)
        return cls.C * d * d


class TestJointVsAnalyticKink:
    """Joint must beat the analytic polyline on a nonlinear tube."""

    # Bias -10 for the square-law stub: Iq = 0.05*10^2 = 5 mA;
    # ia_a - ia_b = 0.05*40*d = 2d -> v = 4d -> ua = ua_q - 4d (class A).

    @staticmethod
    def _qtraj(swing: float):
        return pp_joint_trajectory(_QuadraticTube(), _ll(), BIAS, swing)

    def test_class_a_current_differs_from_analytic_line(self):
        """Drive +4 (index 3n//4 at swing 8): joint ia =
        0.05*14^2 = 9.8 mA at ua = ua_q - 16; the analytic Zaa/2 line
        gives Iq + 16/4 = 9.0 mA — class-A curvature is visible."""
        from lm19.amplifier import pp_working_line_ia
        pts = self._qtraj(swing=8.0)["points"]
        n = len(pts)
        ua_q = pts[n // 2]["ua"]
        iq = pts[n // 2]["ia"]
        assert iq == pytest.approx(5.0, rel=1e-2)
        p4 = pts[3 * n // 4]
        analytic = float(pp_working_line_ia(p4["ua"], ua_q, iq, RAA))
        assert p4["ia"] == pytest.approx(9.8, rel=0.02)
        assert p4["ia"] > analytic + 0.5, (
            f"joint {p4['ia']:.2f} vs analytic {analytic:.2f} — "
            "class-A curvature lost")

    def test_kink_at_true_partner_cutoff_not_at_2iq(self):
        """The square-law partner cuts off at drive +10, i.e.
        ia_a = 0.05*20^2 = 20 mA = 4*Iq. The analytic puts the kink at
        2*Iq = 10 mA — joint must find the real one."""
        out = self._qtraj(swing=12.0)
        kink = out["kink"]
        assert kink is not None
        # The square-law partner fades tangentially — detection at the
        # CUTOFF_IA_MA threshold fires slightly before the 4*Iq=20
        # asymptote (about 18 in practice). Discrimination from the
        # analytic kink (2*Iq=10) keeps a wide margin.
        assert 15.0 < kink["ia"] <= 20.5, kink["ia"]
        assert abs(kink["ia"] - 10.0) > 5.0                  # not 2*Iq


# ======================================================================
# 2-4. Engine: variant-3 routing
# ======================================================================


def _grid_points() -> List[Dict]:
    pts = []
    for ug1 in [-2.0, -6.0, -10.0, -14.0, -18.0]:
        for ua in range(50, 425, 25):
            ia = max(0.0, _GM * (ug1 - _CUTOFF))
            pts.append({"ua": float(ua), "ug1": ug1, "ia": ia,
                        "ug2": 0.0, "series_id": 0})
    return pts


def _engine(with_model: bool = True) -> AmplifierEngine:
    eng = AmplifierEngine()
    eng.set_data(_grid_points(), is_triode=True,
                 series_models={0: _LinearTube()} if with_model else None)
    return eng


def _pp_params(**kw) -> AmpParams:
    base = dict(circuit=CIRCUIT_PP, ub=UB, pp_raa=RAA, ug1_bias=BIAS,
                half_swing=8.0, hd_method=HD_METHOD_5POINT, series_id=0)
    base.update(kw)
    return AmpParams(**base)


class TestEngineVariant3:

    def test_joint_even_for_data_hd_method(self):
        """THE key pin: method is 5-point (data) but a model exists ->
        the trajectory is joint, not display."""
        view = _engine().compute_working_line(_pp_params())
        assert view.pp_trajectory, "joint trajectory lost for 5point"

    def test_polyline_is_the_trajectory(self):
        """The straight line is REPLACED by the trajectory (a mutant
        keeping the 2-vertex line fails here)."""
        view = _engine().compute_working_line(_pp_params())
        assert len(view.polyline) == len(view.pp_trajectory)
        assert view.polyline == [(p["ua"], p["ia"])
                                 for p in view.pp_trajectory]

    def test_intersections_on_trajectory_not_on_straight_line(self):
        """Markers moved onto the trajectory: in class A their Ua
        differs from the straight Zaa/4 line through Q."""
        view = _engine().compute_working_line(_pp_params())
        isects = view.intersections
        assert isects, "no joint intersections"
        # Markers only inside the swing
        for p in isects:
            assert BIAS - 8.0 <= p["ug1"] <= BIAS + 8.0
        # Point at ug1=-6 (drive +4, class A): joint ua = ua_q -
        # Zaa/2*dIa = 248 - 4*8 = 216; a straight line gives 248 - 16 = 232.
        p6 = next(p for p in isects if abs(p["ug1"] + 6.0) < 0.1)
        assert p6["ua"] == pytest.approx(216.0, abs=2.0)
        assert abs(p6["ua"] - 232.0) > 5.0  # not the display value

    def test_swing_geometry_from_trajectory(self):
        view = _engine().compute_working_line(_pp_params())
        geo = view.swing_geometry
        assert geo["pt_pos"]["ug1"] == pytest.approx(BIAS + 8.0)
        assert geo["pt_neg"]["ug1"] == pytest.approx(BIAS - 8.0)
        # EXACT joint value: ua_q + Zaa/2*dIa = 248 + 4*16 = 312.
        # Display interpolation along Zaa/4 would give 280 — '> UB'
        # does not discriminate; the exact value does.
        assert geo["pt_neg"]["ua"] == pytest.approx(312.0, abs=2.0)
        assert geo["pt_pos"]["ua"] == pytest.approx(184.0, abs=2.0)
        assert geo["pt_low_half"]["ug1"] == pytest.approx(BIAS - 4.0)

    def test_no_model_falls_back_to_display(self):
        view = _engine(with_model=False).compute_working_line(_pp_params())
        assert view.pp_trajectory == []
        assert view.pp_kink is None
        # Display polyline has 2-3 vertices (through Q), not 65 points.
        assert len(view.polyline) <= 3

    def test_mismatched_pair_falls_back_to_display(self):
        pts = _grid_points() + [
            {**p, "series_id": 2, "ia": p["ia"] * 0.5}
            for p in _grid_points()]
        eng = AmplifierEngine()
        eng.set_data(pts, is_triode=True,
                     series_models={0: _LinearTube()})
        view = eng.compute_working_line(
            _pp_params(pp_matched=False, pp_tube_b_sid=2))
        assert view.pp_trajectory == []

    def test_kink_flows_into_view(self):
        view = _engine().compute_working_line(_pp_params(half_swing=12.0))
        assert view.pp_kink is not None
        assert view.pp_kink["ug1"] == pytest.approx(0.0, abs=0.5)

    def test_intersections_joint_vs_analytic_on_nonlinear_tube(self):
        """Square-law tube: the joint marker at ug1=-6 (drive +4) sits
        at ua_q-16 (v = 4*drive); the analytic polyline would put the
        intersection at ua_q-19.2 (ia 9.8 = 5 + delta/4). On a linear
        stub both paths coincide — only a nonlinear one discriminates
        the "markers not moved" mutation."""
        pts = []
        for ug1 in [-2.0, -6.0, -10.0, -14.0, -18.0]:
            for ua in range(50, 425, 25):
                pts.append({"ua": float(ua), "ug1": ug1,
                            "ia": _QuadraticTube.ia(ua, ug1),
                            "ug2": 0.0, "series_id": 0})
        eng = AmplifierEngine()
        eng.set_data(pts, is_triode=True,
                     series_models={0: _QuadraticTube()})
        view = eng.compute_working_line(_pp_params())
        ua_q = view.swing_geometry["ua_0"]
        p6 = next(p for p in view.intersections
                  if abs(p["ug1"] + 6.0) < 0.1)
        assert p6["ua"] == pytest.approx(ua_q - 16.0, abs=1.0)   # joint
        assert abs(p6["ua"] - (ua_q - 19.2)) > 1.5               # not analytic

    def test_analyze_path_also_joint(self):
        """Call-site twin: full analyze() builds the view through the
        same internals."""
        result = _engine().analyze(_pp_params())
        assert result.working_line is not None
        assert result.working_line.pp_trajectory


class _Ug2SensitiveTube:
    """Pentode stub with Ug2-dependent Ia: Ia = 2*(Ug1+20) + 0.05*Ug2.

    Linear/square-law stubs ignore Ug2 — the UL wrapper is INVISIBLE
    on them and the mutation "joint gets the raw model instead of the
    wrapped one" would pass every other pin."""

    topology = TOPOLOGY_PENTODE
    model_type = "stub_p"
    name = "ug2 stub"

    @staticmethod
    def ia(ua: float, ug1: float, ug2: float = 0.0) -> float:
        return max(0.0, 2.0 * (ug1 + 20.0) + 0.05 * ug2)


class TestUlJointInteraction:
    """UL tap>0: the joint trajectory must be computed with the WRAPPED
    model (Ug2_eff = Ug2_nom*(1-tap) + Ua*tap), and the UL family must
    coexist with the joint substitution."""

    def _view(self, ul_tap):
        pts = []
        for ug1 in [-2.0, -6.0, -10.0, -14.0, -18.0]:
            for ua in range(50, 425, 25):
                pts.append({"ua": float(ua), "ug1": ug1,
                            "ia": _Ug2SensitiveTube.ia(ua, ug1, 250.0),
                            "ug2": 250.0, "series_id": 0})
        eng = AmplifierEngine()
        eng.set_data(pts, is_triode=False,
                     series_models={0: _Ug2SensitiveTube()})
        return eng.compute_working_line(_pp_params(
            ug2_filter=250.0, ul_tap=ul_tap))

    def test_trajectory_sees_ul_wrapper(self):
        """tap=0.4 vs tap=None: Ug2_eff depends on Ua, so trajectory
        currents differ. The "raw model in joint" mutant yields
        identical trajectories."""
        base = self._view(None).pp_trajectory
        ul = self._view(0.4).pp_trajectory
        assert base and ul
        d_ia = max(abs(a["ia"] - b["ia"]) for a, b in zip(base, ul))
        assert d_ia > 0.5, "UL wrapper did not reach the joint trajectory"

    def test_ul_family_coexists_with_joint(self):
        view = self._view(0.4)
        assert view.pp_trajectory, "joint lost with UL"
        assert view.model_family, "UL family lost with joint"
        # The family sits on the joint markers' Ug1 grid (one source).
        fam_ug1 = {g for g, _, _ in view.model_family}
        isect_ug1 = {round(p["ug1"], 2) for p in view.intersections}
        assert fam_ug1 == isect_ug1


# ======================================================================
# 5. Controller: kink item + line-source note
# ======================================================================


def _make_renderer():
    import pyqtgraph as pg

    from app.plotting import PlotRenderer

    r = PlotRenderer(
        plot=pg.PlotWidget(),
        contour_plot=pg.PlotWidget(),
        contour_image=pg.ImageItem(),
        transfer_plot=pg.PlotWidget(),
        curves_plot=pg.PlotWidget(),
    )
    r.is_triode = True
    r.track_sids = set()
    return r


def _controller(r):
    from app.working_line import WorkingLineController
    return WorkingLineController(
        plot=r.plot, engine=MagicMock(),
        get_params=lambda: None, renderer=r)


@pytest.mark.usefixtures("qapp")
class TestControllerJoint:

    @staticmethod
    def _joint_view(kink: bool = True) -> WorkingLineView:
        pts = _traj(swing=12.0 if kink else 8.0)["points"]
        view = WorkingLineView(
            circuit=CIRCUIT_PP,
            polyline=[(p["ua"], p["ia"]) for p in pts],
            intersections=[pts[len(pts) // 2]],
            pp_trajectory=pts,
        )
        if kink:
            view.pp_kink = _traj(swing=12.0)["kink"]
        return view

    def test_kink_item_set_and_cleared(self):
        r = _make_renderer()
        wl = _controller(r)
        wl._visible = True
        wl._render_view(self._joint_view(kink=True))
        xs, _ = wl._kink_item.getData()
        assert xs is not None and len(xs) == 1
        wl._render_view(self._joint_view(kink=False))
        xs, _ = wl._kink_item.getData()
        assert xs is None or len(xs) == 0

    def test_info_line_labels_line_source(self):
        r = _make_renderer()
        wl = _controller(r)
        joint = wl._info_text(self._joint_view())
        display = wl._info_text(WorkingLineView(circuit=CIRCUIT_PP))
        assert "joint" in joint
        assert "joint" not in display and "Za-a/2" in display

    def test_bent_polyline_reaches_controller_line_item(self):
        """The 65-vertex trajectory is drawn by the existing line item
        (architectural invariant: the controller needs no new code)."""
        r = _make_renderer()
        wl = _controller(r)
        wl._visible = True
        wl._render_view(self._joint_view())
        xs, _ = wl._line_item.getData()
        assert len(xs) == 65
