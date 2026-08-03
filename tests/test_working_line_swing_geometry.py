"""Working-line Q/swing marker regression tests.

Symptom: the line and swing segments used to draw always and on every
plot; after moving parameters to the amp panel they stopped. Causes:
only the 5-point dict carried the geometry (auto method = Chebyshev/
DFT have no pt_*), draw_qpoint_all was not called on tick, and the 2D
swing markers had no call site.

Pins:
  1.  Engine: swing_geometry present with CHEBYSHEV (discriminates
      "5-point only"), sides not swapped (pt_pos = bias + swing),
      half-points at +/- swing/2.
  2.  Geometry is method-independent: 5point == chebyshev bit-exact.
  3.  Auto swing (half_swing=None): geometry uses the actual
      hd["half_swing"] (kills a "params.half_swing only" mutant).
  4.  HD failed (hd=None) -> Q geometry still present.
  5.  Feed: analysis = geometry + hd (THD per method, pt_* always).
  6-7. Feed calls draw_qpoint_all: markers appear on tick and are
      REMOVED when switched off (stale markers were the main
      symptom).
  8.  2D swing items: filled from geometry, emptied without it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.amplifier.constants import (
    CIRCUIT_CF,
    CIRCUIT_PP,
    CIRCUIT_SE,
    CIRCUIT_SE_XFMR,
    HD_METHOD_5POINT,
    HD_METHOD_CHEBYSHEV,
)
from lm19.amp_engine import AmplifierEngine, AmpParams, WorkingLineView
from lm19.constants import TOPOLOGY_TRIODE

pytestmark = [pytest.mark.smoke_analysis]

BIAS = -10.0
SWING = 4.0


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ── Module local helpers ──

def _linear_points() -> List[Dict]:
    """Triode grid, linear transfer: Ia = (Ug1+20)·Ua/400."""
    pts = []
    for ug1 in [0.0, -2.0, -4.0, -6.0, -8.0, -10.0, -12.0, -14.0,
                -16.0, -18.0, -20.0]:
        for ua in range(50, 425, 25):
            ia = max(0.0, (ug1 + 20.0) * ua / 400.0)
            pts.append({"ua": float(ua), "ug1": ug1, "ia": ia,
                        "ug2": 0.0, "series_id": 0})
    return pts


def _se_params(**kw) -> AmpParams:
    base = dict(circuit=CIRCUIT_SE, ub=250.0, ra=5.0, ug1_bias=BIAS,
                half_swing=SWING, hd_method=HD_METHOD_CHEBYSHEV)
    base.update(kw)
    return AmpParams(**base)


def _engine() -> AmplifierEngine:
    eng = AmplifierEngine()
    eng.set_data(_linear_points(), is_triode=True)
    return eng


GEO_KEYS = ("ug1_0", "ua_0", "ia_0", "pt_neg", "pt_pos",
            "pt_low_half", "pt_high_half")


# ======================================================================
# 1-4. Engine: method-independent geometry
# ======================================================================


class TestEngineSwingGeometry:

    def test_chebyshev_view_carries_full_geometry(self):
        """THE regression: auto method = Chebyshev whose hd dict has
        no pt_* — the geometry must arrive as a separate field."""
        view = _engine().compute_working_line(_se_params())
        geo = view.swing_geometry
        for key in GEO_KEYS:
            assert key in geo, f"swing_geometry lost {key!r}"
        # Sides: pt_pos toward grid zero (bias + swing), pt_neg toward cutoff.
        assert geo["pt_pos"]["ug1"] == pytest.approx(BIAS + SWING)
        assert geo["pt_neg"]["ug1"] == pytest.approx(BIAS - SWING)
        assert geo["pt_high_half"]["ug1"] == pytest.approx(BIAS + SWING / 2)
        assert geo["pt_low_half"]["ug1"] == pytest.approx(BIAS - SWING / 2)
        assert geo["ug1_0"] == pytest.approx(BIAS)
        # ...and hd genuinely has no pt_* (else the pin is degenerate).
        assert view.hd is not None and "pt_neg" not in view.hd

    def test_geometry_is_method_independent(self):
        eng = _engine()
        geo_5p = eng.compute_working_line(
            _se_params(hd_method=HD_METHOD_5POINT)).swing_geometry
        geo_ch = eng.compute_working_line(
            _se_params(hd_method=HD_METHOD_CHEBYSHEV)).swing_geometry
        assert geo_5p == geo_ch

    def test_auto_swing_uses_resolved_hd_swing(self):
        """half_swing=None (auto): geometry is built from the actual
        hd swing — a "manual swing only" mutant yields empty pt_*."""
        view = _engine().compute_working_line(_se_params(half_swing=None))
        assert view.hd is not None and view.hd.get("half_swing")
        geo = view.swing_geometry
        assert "pt_neg" in geo and "pt_pos" in geo
        assert geo["pt_pos"]["ug1"] == pytest.approx(
            BIAS + view.hd["half_swing"])

    def test_hd_failure_keeps_q_geometry(self):
        """HD did not compute (bias outside the measured Ug1 range ->
        DIST_ERR_BIAS_OUTSIDE) — Q geometry on the plots must still
        live (interp_intersection extrapolates)."""
        view = _engine().compute_working_line(_se_params(ug1_bias=-25.0))
        assert view.hd is None and view.hd_error is not None
        geo = view.swing_geometry
        assert geo.get("ua_0") is not None
        assert geo.get("ug1_0") == pytest.approx(-25.0)


class TestSeXfmrView:
    """SE transformer at the view level (pins existed only at the dist
    level): AC polyline through Q up to Ua_q + Iq*Ra (~2x Ub), a
    separate DC line, cutoff end of the swing ABOVE the supply. The
    model is a linear ua-independent stub: values are analytic."""

    class _Lin:
        topology = TOPOLOGY_TRIODE
        model_type = "stub"
        name = "lin"

        @staticmethod
        def ia(ua: float, ug1: float, ug2: float = 0.0) -> float:
            return max(0.0, 2.0 * (ug1 + 20.0))

    def _view(self):
        # Data CONSISTENT with the model (Ia = 2*(Ug1+20), ua-
        # independent): the data-first DC Q uses the DATA — a mismatch
        # would shift Q and every analytic expectation below.
        pts = [{"ua": float(ua), "ug1": ug1,
                "ia": self._Lin.ia(float(ua), ug1),
                "ug2": 0.0, "series_id": 0}
               for ug1 in [-2.0, -6.0, -10.0, -14.0, -18.0]
               for ua in range(50, 425, 25)]
        eng = AmplifierEngine()
        eng.set_data(pts, is_triode=True,
                     series_models={0: self._Lin()})
        # ra=5 (Ra_ac), ra_dc=0.05 (AmpParams default) -> Ua_q = 250 - 1 = 249
        return eng.compute_working_line(AmpParams(
            circuit=CIRCUIT_SE_XFMR, ub=250.0, ra=5.0, ug1_bias=BIAS,
            half_swing=SWING, hd_method=HD_METHOD_5POINT, series_id=0))

    def test_polyline_is_ac_line_through_q(self):
        view = self._view()
        assert len(view.polyline) == 2
        # Cutoff end: Ua_q + Iq*Ra = 249 + 20*5 = 349 — well above Ub.
        end_ua = max(p[0] for p in view.polyline)
        assert end_ua == pytest.approx(349.0, abs=2.0)
        assert view.dc_polyline, "xfmr DC line disappeared"

    def test_swing_endpoint_above_supply(self):
        geo = self._view().swing_geometry
        # pt_neg (bias−4): ia=12 → ua = 249 + (20−12)·5 = 289 > Ub.
        assert geo["pt_neg"]["ua"] == pytest.approx(289.0, abs=2.0)
        assert geo["pt_neg"]["ua"] > 250.0
        # Q on the DC line: Ua_q ~ 249, Iq = 20.
        assert geo["ua_0"] == pytest.approx(249.0, abs=2.0)


class TestCfView:
    """Cathode follower at the view level — the last circuit without
    working-line pins (resistive/xfmr/PP are covered). CF line:
    Ia = (Ub − Ua)/(Rk + Rl); polyline endpoints and Q consistency are
    contract-level (feedback specifics live in the CF load line)."""

    def _view(self):
        eng = _engine()
        # cf_rk=1, cf_rl=4 → line from (0, 50 mA) to (250, 0).
        return eng.compute_working_line(AmpParams(
            circuit=CIRCUIT_CF, ub=250.0, cf_rk=1.0, cf_rl=4.0,
            ug1_bias=BIAS, half_swing=SWING,
            hd_method=HD_METHOD_5POINT))

    def test_polyline_is_cf_dc_line(self):
        view = self._view()
        assert len(view.polyline) == 2
        (ua0, ia0), (ua1, ia1) = sorted(view.polyline)
        assert (ua0, ia0) == (0.0, pytest.approx(50.0))
        assert (ua1, ia1) == (pytest.approx(250.0), 0.0)

    def test_q_lies_on_the_cf_line(self):
        geo = self._view().swing_geometry
        assert geo.get("ua_0") is not None
        # Q must satisfy the line equation Ia = (Ub − Ua)/(Rk+Rl).
        expected_ia = (250.0 - geo["ua_0"]) / 5.0
        assert geo["ia_0"] == pytest.approx(expected_ia, rel=0.05)
        assert 0.0 < geo["ua_0"] < 250.0

    def test_swing_geometry_present(self):
        geo = self._view().swing_geometry
        assert "pt_neg" in geo and "pt_pos" in geo
        assert geo["pt_pos"]["ug1"] == pytest.approx(BIAS + SWING)


class TestPpSwingGeometry:
    """PP twin: the pp_distortion hd dict has NO ua_0 and no pt_*
    (ia_0 is composite, ~0 for a matched pair) — geometry must come
    from the tube-A display intersections (the same ra_per_tube
    approximation as the 2D intersection markers)."""

    def test_pp_view_carries_geometry(self):
        view = _engine().compute_working_line(AmpParams(
            circuit=CIRCUIT_PP, ub=250.0, pp_raa=8.0, ug1_bias=BIAS,
            half_swing=SWING, hd_method=HD_METHOD_5POINT))
        geo = view.swing_geometry
        assert "pt_neg" in geo and "pt_pos" in geo, (
            "PP lost swing geometry")
        assert geo["pt_pos"]["ug1"] == pytest.approx(BIAS + SWING)
        assert geo["pt_neg"]["ug1"] == pytest.approx(BIAS - SWING)
        assert geo.get("ua_0") is not None
        # Non-degeneracy: hd really has no ua_0/pt_* (otherwise this
        # pin would pass on the old feed too).
        assert view.hd is not None
        assert "ua_0" not in view.hd and "pt_neg" not in view.hd

    def test_pp_feed_analysis_has_ua0_and_swing(self, qapp):
        """Q on heatmaps needs ua_0+ug1_0; for PP only the geometry
        provides ua_0. THD stays method-specific; ia_0 is the PER-TUBE
        working point (geometry), NOT the composite ~0 of the hd dict —
        the composite used to leak into the PDF Q-point line."""
        r = _make_renderer()
        wl = _controller(r)
        eng = _engine()
        view = eng.compute_working_line(AmpParams(
            circuit=CIRCUIT_PP, ub=250.0, pp_raa=8.0, ug1_bias=BIAS,
            half_swing=SWING, hd_method=HD_METHOD_5POINT))
        wl._feed_legacy(view)
        analysis = r._load_line_analysis
        assert analysis["ua_0"] is not None
        assert "pt_pos" in analysis
        assert analysis["thd"] == view.hd["thd"]      # method value from hd
        # Per-tube Iq (linear grid: Ia at bias −10 is well above zero);
        # the composite hd ia_0 for a matched pair is ~0.
        assert analysis["ia_0"] > 1.0
        assert abs(view.hd.get("ia_0", 0.0)) < 0.5    # anti-degeneracy
        assert "transfer" in r._qpoint_items          # swing region alive


# ======================================================================
# 5-7. Controller feed: merge + redraw
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


def _geo() -> Dict:
    return {
        "ug1_0": BIAS, "ua_0": 200.0, "ia_0": 10.0,
        "pt_neg": {"ua": 240.0, "ia": 5.0, "ug1": BIAS - SWING},
        "pt_pos": {"ua": 160.0, "ia": 15.0, "ug1": BIAS + SWING},
        "pt_low_half": {"ua": 220.0, "ia": 7.5, "ug1": BIAS - SWING / 2},
        "pt_high_half": {"ua": 180.0, "ia": 12.5, "ug1": BIAS + SWING / 2},
    }


@pytest.mark.usefixtures("qapp")
class TestFeedMergeAndRedraw:

    def test_analysis_merges_geometry_and_hd(self):
        r = _make_renderer()
        wl = _controller(r)
        view = WorkingLineView(
            circuit=CIRCUIT_SE, swing_geometry=_geo(),
            hd={"thd": 2.5, "hd2": 2.0, "hd3": 1.0, "half_swing": SWING,
                "ua_0": 201.0, "ia_0": 10.1, "ug1_0": BIAS})
        wl._feed_legacy(view)
        analysis = r._load_line_analysis
        assert analysis["thd"] == 2.5                # method value from hd
        assert analysis["pt_neg"]["ua"] == 240.0     # geometry always
        # Geometry keys carry DISPLAY semantics and win over hd: the PP
        # hd dict has a COMPOSITE ia_0 (~0 for a matched pair) which
        # used to print "Q: Ia=0.0 mA" in the PDF report.
        assert analysis["ua_0"] == 200.0             # geometry wins
        assert analysis["ia_0"] == 10.0

    def test_analysis_carries_engine_resolved_method(self):
        """The PDF distortion block prints analysis["method"]; the feed
        must stamp the ENGINE-resolved method (single source with the
        info line) — the 5-point dict has no "method" of its own, and a
        dict-supplied one must NOT win over the resolved value."""
        r = _make_renderer()
        wl = _controller(r)
        wl._feed_legacy(WorkingLineView(
            circuit=CIRCUIT_SE, swing_geometry=_geo(),
            hd={"thd": 1.0, "method": "stale-dict-code"},
            method_used=HD_METHOD_CHEBYSHEV))
        assert r._load_line_analysis["method"] == HD_METHOD_CHEBYSHEV

    def test_hd_none_analysis_falls_back_to_geometry(self):
        r = _make_renderer()
        wl = _controller(r)
        wl._feed_legacy(WorkingLineView(circuit=CIRCUIT_SE,
                                        swing_geometry=_geo(), hd=None))
        analysis = r._load_line_analysis
        assert analysis is not None
        assert analysis["ua_0"] == 200.0
        assert "pt_pos" in analysis

    def test_feed_redraws_qpoints_on_tick_and_clear(self):
        """Spy on redraw: markers must refresh on feed AND on switch-
        off (stale markers were the user-visible symptom)."""
        r = _make_renderer()
        wl = _controller(r)
        calls = []
        r.draw_qpoint_all = lambda: calls.append(1)
        wl._feed_legacy(WorkingLineView(circuit=CIRCUIT_SE,
                                        swing_geometry=_geo()))
        assert len(calls) == 1
        wl._feed_legacy(None)
        assert len(calls) == 2
        assert r._load_line_analysis is None

    def test_markers_appear_and_disappear_on_real_renderer(self):
        """Integration: feed draws the swing region on Transfer and Q
        on contour; feed(None) removes everything (nothing stale)."""
        r = _make_renderer()
        wl = _controller(r)
        wl._feed_legacy(WorkingLineView(circuit=CIRCUIT_SE,
                                        swing_geometry=_geo()))
        assert "transfer" in r._qpoint_items
        assert "contour" in r._qpoint_items
        wl._feed_legacy(None)
        assert r._qpoint_items == {}


@pytest.mark.usefixtures("qapp")
class TestInfoLineQ:
    """Iq is visible in the live info line, and the 2D Q cross lives
    with hd=None and for the resistive circuit (geometry fallback)."""

    def test_info_line_shows_q(self):
        r = _make_renderer()
        wl = _controller(r)
        view = WorkingLineView(circuit=CIRCUIT_SE,
                               swing_geometry=_geo(), hd=None)
        text = wl._info_text(view)
        assert "200" in text and "10.0" in text, text

    def test_q_marker_falls_back_to_geometry(self):
        from app.working_line import WorkingLineController
        view = WorkingLineView(circuit=CIRCUIT_SE,
                               swing_geometry=_geo(), hd=None)
        assert WorkingLineController._q_for_marker(view) == (200.0, 10.0)
        # hd on top of geometry (method Q is more precise)
        view.hd = {"ua_0": 201.0, "ia_0": 10.5}
        assert WorkingLineController._q_for_marker(view) == (201.0, 10.5)


@pytest.mark.usefixtures("qapp")
class TestVisibilityToggle:
    """OFF->ON of the Load line checkbox: line and Q used to vanish
    until the first parameter change. Cause: set_visible(False) clears
    the items but not the cache; the next tick hit the cache in
    _recompute and skipped the redraw."""

    def test_toggle_off_on_redraws_from_cache(self):
        from lm19.amp_engine import WorkingLineView
        r = _make_renderer()
        params = _se_params()
        view = WorkingLineView(
            circuit=CIRCUIT_SE,
            polyline=[(0.0, 50.0), (250.0, 0.0)],
            intersections=[{"ua": 200.0, "ia": 10.0, "ug1": BIAS}],
            swing_geometry=_geo(),
        )
        engine = MagicMock()
        engine.compute_working_line.return_value = view

        from app.working_line import WorkingLineController
        wl = WorkingLineController(plot=r.plot, engine=engine,
                                   get_params=lambda: params, renderer=r)
        wl.set_visible(True)
        wl._recompute()                      # first tick: computed and drew
        assert engine.compute_working_line.call_count == 1
        assert len(wl._line_item.getData()[0]) == 2

        wl.set_visible(False)                # switched off: all removed
        assert wl._line_item.getData()[0] is None or \
            len(wl._line_item.getData()[0]) == 0
        assert r._qpoint_items == {}

        wl.set_visible(True)                 # ON again WITHOUT any
        xs, _ = wl._line_item.getData()      # parameter change: cache draw
        assert xs is not None and len(xs) == 2, (
            "OFF->ON: line not redrawn (cache hit ate the render)")
        assert "transfer" in r._qpoint_items  # Q/swing markers back
        # ...and without a recompute (cache still valid).
        assert engine.compute_working_line.call_count == 1


# ======================================================================
# 8. 2D swing markers (controller items)
# ======================================================================


@pytest.mark.usefixtures("qapp")
class TestSwing2DMarkers:

    @staticmethod
    def _view_with_geo() -> WorkingLineView:
        return WorkingLineView(
            circuit=CIRCUIT_SE,
            polyline=[(0.0, 50.0), (400.0, 0.0)],
            intersections=[{"ua": 160.0, "ia": 15.0, "ug1": -6.0},
                           {"ua": 200.0, "ia": 10.0, "ug1": -10.0},
                           {"ua": 240.0, "ia": 5.0, "ug1": -14.0}],
            swing_geometry=_geo(),
        )

    def test_render_fills_swing_items(self):
        r = _make_renderer()
        wl = _controller(r)
        wl._visible = True
        wl._render_view(self._view_with_geo())
        xs, ys = wl._swing_item.getData()
        assert list(xs) == [240.0, 160.0] and list(ys) == [5.0, 15.0]
        hx, hy = wl._swing_half_item.getData()
        assert list(hx) == [220.0, 180.0] and list(hy) == [7.5, 12.5]

    def test_no_geometry_empties_swing_items(self):
        r = _make_renderer()
        wl = _controller(r)
        wl._visible = True
        wl._render_view(self._view_with_geo())
        view = self._view_with_geo()
        view.swing_geometry = {}
        wl._render_view(view)
        xs, _ = wl._swing_item.getData()
        assert xs is None or len(xs) == 0

    def test_set_visible_false_clears_swing_items(self):
        r = _make_renderer()
        wl = _controller(r)
        wl._visible = True
        wl._render_view(self._view_with_geo())
        wl.set_visible(False)
        xs, _ = wl._swing_item.getData()
        assert xs is None or len(xs) == 0
        hx, _ = wl._swing_half_item.getData()
        assert hx is None or len(hx) == 0
