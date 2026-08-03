"""PP composite on the Transfer tab.

Pins:
  1.  fold_pp_composite: folding around BIAS (not zero), with sign
      flip; a matched pair -> branches coincide, imbalance -> a gap;
      the ia==0 boundary goes to the direct branch.
  2.  Engine: both paths (analyze AND compute_working_line) put the
      composite into WorkingLineView + pp_bias; SE circuit -> empty.
  3.  Mismatched pair: points_b reaches composite_characteristic at
      BOTH call sites (a lost kwarg silently gives a matched composite).
  4.  Drawing: solid + dashed branches in COLOR_PP_COMPOSITE with an
      underlay and label; gates — only Ia(Ug1); Ua/Ug2 filters do not
      hide the curve (negative space).
  5.  Controller feed: view -> renderer, reset on None and on SE view.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.amplifier import composite_characteristic, fold_pp_composite
from lm19.amplifier.constants import (
    CIRCUIT_PP,
    CIRCUIT_SE,
    HD_METHOD_5POINT,
)
from lm19.amp_engine import AmplifierEngine, AmpParams, WorkingLineView
from lm19.plot_style import COLOR_PP_COMPOSITE

pytestmark = [pytest.mark.smoke_analysis]

BIAS = -10.0


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ── Module local helpers ──

def _linear_points(scale: float = 1.0, series_id: int = 0) -> List[Dict]:
    """Triode-ish grid, linear transfer at any Ua slice:
    Ia = scale·(Ug1+20)·Ua/400 → composite around bias −10 is odd."""
    pts = []
    for ug1 in [0.0, -2.0, -4.0, -6.0, -8.0, -10.0, -12.0, -14.0,
                -16.0, -18.0, -20.0]:
        for ua in range(50, 425, 25):
            ia = max(0.0, scale * (ug1 + 20.0) * ua / 400.0)
            pts.append({"ua": float(ua), "ug1": ug1, "ia": ia,
                        "ug2": 0.0, "series_id": series_id})
    return pts


def _pp_params(**kw) -> AmpParams:
    base = dict(circuit=CIRCUIT_PP, ub=250.0, pp_raa=8.0, ug1_bias=BIAS,
                half_swing=5.0, hd_method=HD_METHOD_5POINT)
    base.update(kw)
    return AmpParams(**base)


def _make_engine(points: List[Dict]) -> AmplifierEngine:
    eng = AmplifierEngine()
    eng.set_data(points, is_triode=True)
    return eng


# ======================================================================
# 1. fold_pp_composite (pure)
# ======================================================================


class TestFoldPpComposite:

    @staticmethod
    def _comp(asym: float = 0.0) -> List[Dict]:
        """Composite rows around BIAS; asym > 0 skews the negative half
        (mismatched pair) — non-degenerate, unequal spacing."""
        rows = []
        for v in (-7.0, -4.5, -2.0, 0.0, 2.0, 4.5, 7.0):
            ia = 1.25 * v
            if ia < 0:
                ia *= (1.0 + asym)
            rows.append({"ug1": BIAS + v, "ia_a": 0.0, "ia_b": 0.0,
                         "ia_composite": ia})
        return rows

    def test_matched_halves_coincide(self):
        direct, mirrored = fold_pp_composite(self._comp(), BIAS)
        # ia==0 row (boundary) belongs to direct
        assert len(direct) == 4 and len(mirrored) == 3
        d = {round(x, 6): y for x, y in direct}
        for x, y in mirrored:
            assert d[round(x, 6)] == pytest.approx(y), (x, y)

    def test_mirror_is_around_bias_not_zero(self):
        """Mutation 'fold around 0' would send points to +Ug1 territory."""
        _, mirrored = fold_pp_composite(self._comp(), BIAS)
        xs = [x for x, _ in mirrored]
        # negative half lives at ug1 < bias; its mirror must land at
        # ug1 > bias but still NEGATIVE grid volts (2·(−10) − ug1).
        assert xs == pytest.approx([-8.0, -5.5, -3.0])

    def test_mismatch_produces_gap(self):
        direct, mirrored = fold_pp_composite(self._comp(asym=0.3), BIAS)
        d = {round(x, 6): y for x, y in direct}
        gaps = [abs(d[round(x, 6)] - y) for x, y in mirrored]
        assert max(gaps) > 0.5  # visible even-harmonic residue

    def test_sign_flip_present(self):
        """Mirrored Y values must be positive (fold, not raw copy)."""
        _, mirrored = fold_pp_composite(self._comp(), BIAS)
        assert all(y > 0 for _, y in mirrored)

    def test_boundary_zero_goes_direct(self):
        direct, mirrored = fold_pp_composite(
            [{"ug1": BIAS, "ia_a": 0, "ia_b": 0, "ia_composite": 0.0}], BIAS)
        assert direct == [(BIAS, 0.0)] and mirrored == []


# ======================================================================
# 2-3. Engine: both view-builders carry the composite
# ======================================================================


class TestEngineCompositeInView:

    def test_working_line_pp_has_composite_and_bias(self):
        eng = _make_engine(_linear_points())
        view = eng.compute_working_line(_pp_params())
        assert view.pp_composite, "PP view lost the composite"
        assert view.pp_bias == BIAS
        # Odd symmetry of the linear grid: composite(bias) ≈ 0
        at_bias = min(view.pp_composite,
                      key=lambda p: abs(p["ug1"] - BIAS))
        assert at_bias["ia_composite"] == pytest.approx(0.0, abs=1e-6)

    def test_working_line_se_has_no_composite(self):
        eng = _make_engine(_linear_points())
        view = eng.compute_working_line(
            AmpParams(circuit=CIRCUIT_SE, ub=250.0, ra=5.0, ug1_bias=BIAS,
                      hd_method=HD_METHOD_5POINT))
        assert view.pp_composite == []

    def test_analyze_pp_view_has_composite(self):
        eng = _make_engine(_linear_points())
        result = eng.analyze(_pp_params())
        assert result.working_line is not None
        assert result.working_line.pp_composite

    @pytest.mark.parametrize("path", ["analyze", "working_line"])
    def test_mismatched_pair_reaches_composite(self, path):
        """Call-site spy (both builders): tube B's data must feed the
        composite — a dropped points_b kwarg silently yields matched."""
        pts = _linear_points() + _linear_points(scale=0.5, series_id=2)
        eng = _make_engine(pts)
        params = _pp_params(pp_matched=False, pp_tube_b_sid=2, series_id=0)
        if path == "analyze":
            view = eng.analyze(params).working_line
        else:
            view = eng.compute_working_line(params)
        # At ug1=-4 the mirror point is -16: A gives 0.625·4=2.5 mA,
        # tube B at half current gives 1.25 — matched would show 2.5.
        row = min(view.pp_composite, key=lambda p: abs(p["ug1"] + 4.0))
        assert row["ia_b"] == pytest.approx(1.25, rel=1e-3)
        assert row["ia_composite"] == pytest.approx(
            row["ia_a"] - 1.25, rel=1e-3)

    def test_composite_matches_pure_function(self):
        """The view carries EXACTLY composite_characteristic at Ua=Ub —
        not some other slice (ua_ref swap would shift values)."""
        pts = _linear_points()
        eng = _make_engine(pts)
        view = eng.compute_working_line(_pp_params())
        expected = composite_characteristic(
            pts, None, ug1_bias=BIAS, ug2_filter=None, ua_ref=250.0)
        assert view.pp_composite == expected


# ======================================================================
# 4. Rendering on the Transfer plot
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


def _composite_pens(plot):
    """(solid, dashed) data items drawn in the composite colour."""
    import pyqtgraph as pg
    solid, dashed = [], []
    want = pg.mkColor(COLOR_PP_COMPOSITE).getRgb()
    for it in plot.getPlotItem().listDataItems():
        pen = it.opts.get("pen")
        if pen is None:
            continue
        pen = pg.mkPen(pen)
        if pen.color().getRgb() != want:
            continue
        from PySide6.QtCore import Qt
        if pen.style() == Qt.PenStyle.DashLine:
            dashed.append(it)
        else:
            solid.append(it)
    return solid, dashed


def _feed_composite(r):
    r._pp_composite = composite_characteristic(
        _linear_points(scale=0.9), None, ug1_bias=BIAS, ua_ref=250.0)
    r._pp_bias = BIAS


@pytest.mark.usefixtures("qapp")
class TestCompositeRendering:

    def test_transfer_draws_solid_dashed_and_label(self):
        r = _make_renderer()
        _feed_composite(r)
        r.render_curves(_linear_points(), y_param="Ia", x_param="Ug1",
                        target_plot=r.transfer_plot)
        solid, dashed = _composite_pens(r.transfer_plot)
        assert solid and dashed, "composite branches missing"
        texts = [lbl.textItem.toPlainText() for lbl in r._transfer_labels
                 if hasattr(lbl, "textItem")]
        assert any("PP" in tx for tx in texts), texts

    def test_no_composite_without_data(self):
        r = _make_renderer()
        r.render_curves(_linear_points(), y_param="Ia", x_param="Ug1",
                        target_plot=r.transfer_plot)
        solid, dashed = _composite_pens(r.transfer_plot)
        assert not solid and not dashed

    def test_gate_only_ia_vs_ug1(self):
        """Composite is a Ug1-domain object: Ia(Ua) and Gm views must
        not draw it."""
        r = _make_renderer()
        _feed_composite(r)
        r.render_curves(_linear_points(), y_param="Ia", x_param="Ua")
        solid, dashed = _composite_pens(r.curves_plot)
        assert not solid and not dashed
        r.render_curves(_linear_points(), y_param="Gm", x_param="Ug1")
        solid, dashed = _composite_pens(r.curves_plot)
        assert not solid and not dashed

    def test_curves_tab_ia_vs_ug1_also_draws(self):
        """Twin target: the Curves tab shows the same overlay set as
        Transfer for Ia(Ug1) (consistency with the load-line overlay)."""
        r = _make_renderer()
        _feed_composite(r)
        r.render_curves(_linear_points(), y_param="Ia", x_param="Ug1")
        solid, dashed = _composite_pens(r.curves_plot)
        assert solid and dashed

    def test_ua_filter_does_not_hide_composite(self):
        """Negative space: the composite is not a Ua slice — filters
        must leave it alone."""
        r = _make_renderer()
        _feed_composite(r)
        r.render_curves(_linear_points(), y_param="Ia", x_param="Ug1",
                        target_plot=r.transfer_plot)
        r.set_transfer_ua_visibility([100.0], r.ua_cluster_thr)
        solid, dashed = _composite_pens(r.transfer_plot)
        assert all(it.isVisible() for it in solid + dashed)

    def test_folded_branches_positive_ia(self):
        """Everything drawn sits in the visible quadrant (Y ≥ 0)."""
        import numpy as np
        r = _make_renderer()
        _feed_composite(r)
        r.render_curves(_linear_points(), y_param="Ia", x_param="Ug1",
                        target_plot=r.transfer_plot)
        solid, dashed = _composite_pens(r.transfer_plot)
        for it in solid + dashed:
            _, ys = it.getData()
            assert np.nanmin(ys) >= -1e-9


# ======================================================================
# 5. Controller feed
# ======================================================================


@pytest.mark.usefixtures("qapp")
class TestControllerFeed:

    def _controller(self, r):
        from app.working_line import WorkingLineController
        return WorkingLineController(
            plot=r.plot, engine=MagicMock(),
            get_params=lambda: None, renderer=r)

    def test_feed_sets_and_none_resets(self):
        r = _make_renderer()
        wl = self._controller(r)
        comp = [{"ug1": BIAS, "ia_a": 1.0, "ia_b": 1.0,
                 "ia_composite": 0.0}]
        wl._feed_legacy(WorkingLineView(circuit=CIRCUIT_PP,
                                        pp_composite=comp, pp_bias=BIAS))
        assert r._pp_composite == comp and r._pp_bias == BIAS
        wl._feed_legacy(None)
        assert r._pp_composite == [] and r._pp_bias == 0.0

    def test_se_view_clears_previous_pp(self):
        """PP → SE switch must not leave a stale composite behind."""
        r = _make_renderer()
        wl = self._controller(r)
        wl._feed_legacy(WorkingLineView(
            circuit=CIRCUIT_PP,
            pp_composite=[{"ug1": BIAS, "ia_a": 1.0, "ia_b": 1.0,
                           "ia_composite": 0.5}],
            pp_bias=BIAS))
        wl._feed_legacy(WorkingLineView(circuit=CIRCUIT_SE))
        assert r._pp_composite == []
