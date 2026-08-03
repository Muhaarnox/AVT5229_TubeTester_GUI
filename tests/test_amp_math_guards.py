"""Math guards of the optimizer/amplifier analysis.

Pins (integrating earlier fixes — UL-Chebyshev, ML-118):
- ML-063: a SINGLE UL anchor (ug2_nom) for grid, swing sweep and
  refine — the median of measured Ug2 when no filter is set;
  previously grid used ub_values[0] and refine used best.ub, so the
  phases modeled different screen physics during a Ub sweep;
- ML-053: DFT without ug1_bias derives a mid-range bias from the
  intersections (previously 0 V: a fully open tube, garbage THD
  silently);
- ML-052: a manual PP half_swing beyond the data is clamped with the
  ``manual_swing_clamped`` flag (SE convention), the panel shows a
  notice;
- ML-051: a tiny manual swing yields a reachable
  ``DIST_ERR_MANUAL_SWING_SMALL`` (5-point AND chebyshev; DFT
  returned None); - ML-066: the reference x0 is clipped to bounds (triode and pentode) so scipy no longer fails "x0 is infeasible".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lm19.amplifier import PushPullLoadLine, ResistiveLoadLine
from lm19.amplifier.constants import DIST_ERR_MANUAL_SWING_SMALL, MIN_SWING_V
from lm19.amplifier.constants import (
    CIRCUIT_PP,
    HD_METHOD_5POINT,
    HD_METHOD_DFT,
)


@pytest.fixture(scope="module")
def el84():
    from lm19.tube_sim import quick_pentode
    return quick_pentode("EL84")


@pytest.fixture(scope="module")
def triode_12au7():
    from lm19.tube_sim import quick_triode
    return quick_triode("12AU7")


# -- ML-063: single UL anchor ----------------------------------------

class TestUlAnchorUnified:

    def test_resolver_priority(self):
        from lm19.optimizer import _resolve_ul_ug2_nom
        # NON-symmetric data: median=250, mean~263.3 — a median->mean
        # mutation is distinguishable.
        pts = [{"ug2": 240.0}, {"ug2": 250.0}, {"ug2": 300.0}]
        assert _resolve_ul_ug2_nom(300.0, pts, 999.0) == 300.0  # filter
        assert _resolve_ul_ug2_nom(None, pts, 999.0) == 250.0   # median
        assert _resolve_ul_ug2_nom(None, [{"ug2": 0.0}], 999.0) == 999.0
        assert _resolve_ul_ug2_nom(None, None, 999.0) == 999.0  # fallback

    def test_grid_and_refine_share_anchor(self, el84, monkeypatch):
        """With a Ub grid and NO ug2_filter, grid and refine must build
        the UL wrapper with ONE ug2_nom (data median = 250), not with
        ub_values[0]=290 and best.ub respectively."""
        import lm19.optimizer as opt
        model, pts = el84
        noms: List[float] = []
        orig = opt.UltralinearModelWrapper

        class Spy(orig):
            def __init__(self, m, ug2_nom, tap):
                noms.append(float(ug2_nom))
                super().__init__(m, ug2_nom=ug2_nom, tap=tap)

        monkeypatch.setattr(opt, "UltralinearModelWrapper", Spy)
        c = opt.OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0, pa_max_w=12.0, hd_method=HD_METHOD_DFT,
            ug1_range=(-12.0, -10.0), ra_range=(8.0, 8.0),
            ug1_steps=2, ra_steps=1, swing_steps=2,
            ub_range=(290.0, 310.0), ub_steps=2,
            ul_tap_mode="off", ul_tap_manual=0.43,
        )
        r = opt.optimize_pp(pts, ub=300.0, constraints=c, model=model)
        assert r.best is not None
        opt.refine_optimum(r.best, pts, model, c)
        assert noms, "UL wrapper must have been built"
        assert set(noms) == {250.0}, \
            f"grid/refine must share the median anchor, got {sorted(set(noms))}"


# -- ML-053: DFT bias from data, not 0 V -----------------------------

class TestDftBiasDerivation:

    def _isects(self):
        # intersection family around -8 V
        return [{"ug1": u, "ua": 200.0 + 5 * u, "ia": 30.0 + 2 * u}
                for u in (-12.0, -10.0, -8.0, -6.0, -4.0)]

    def test_none_bias_uses_mid_range(self, triode_12au7):
        from lm19.amplifier.sweeps import _compute_hd
        model, _ = triode_12au7
        ll = ResistiveLoadLine(250.0, 10.0)
        got = _compute_hd("dft", self._isects(), model, ll,
                          ug1_bias=None, half_swing=3.0)
        want = _compute_hd("dft", self._isects(), model, ll,
                           ug1_bias=-8.0, half_swing=3.0)
        assert got is not None and want is not None
        assert got["ug1_0"] == pytest.approx(-8.0)
        assert got["thd"] == pytest.approx(want["thd"])

    def test_none_bias_differs_from_zero_bias(self, triode_12au7):
        """The old behavior (bias 0 V — open tube) must produce
        DIFFERENT numbers — otherwise this pin is vacuous."""
        from lm19.amplifier.sweeps import _compute_hd
        model, _ = triode_12au7
        ll = ResistiveLoadLine(250.0, 10.0)
        derived = _compute_hd("dft", self._isects(), model, ll,
                              ug1_bias=None, half_swing=3.0)
        at_zero = _compute_hd("dft", self._isects(), model, ll,
                              ug1_bias=0.0, half_swing=3.0)
        assert derived is not None
        assert at_zero is None or \
            abs(derived["thd"] - at_zero["thd"]) > 1e-6

    def test_derivation_logs_warning(self, triode_12au7, caplog):
        """Failure visibility: the derived bias is a degradation."""
        import logging
        from lm19.amplifier.sweeps import _compute_hd
        model, _ = triode_12au7
        ll = ResistiveLoadLine(250.0, 10.0)
        with caplog.at_level(logging.WARNING,
                             logger="lm19.amplifier.sweeps"):
            _compute_hd("dft", self._isects(), model, ll,
                        ug1_bias=None, half_swing=3.0)
        assert any("derived" in r.message for r in caplog.records)

    def test_no_intersections_returns_none(self, triode_12au7):
        from lm19.amplifier.sweeps import _compute_hd
        model, _ = triode_12au7
        ll = ResistiveLoadLine(250.0, 10.0)
        assert _compute_hd("dft", [], model, ll,
                           ug1_bias=None, half_swing=3.0) is None


# -- ML-052: PP manual-swing clamp -----------------------------------

class TestPpManualSwingClamp:

    def test_oversized_swing_clamped_and_flagged(self, el84):
        from lm19.amplifier import pp_distortion
        _, pts = el84
        ll = PushPullLoadLine(300.0, 8.0)
        d = pp_distortion(pts, ll, ug1_bias=-11.0, half_swing=50.0,
                          ug2_filter=250.0)
        assert d is not None
        assert d["manual_swing_clamped"] is True
        assert d["requested_half_swing"] == pytest.approx(50.0)
        assert d["half_swing"] < 50.0

    def test_in_range_swing_not_flagged(self, el84):
        from lm19.amplifier import pp_distortion
        _, pts = el84
        ll = PushPullLoadLine(300.0, 8.0)
        auto = pp_distortion(pts, ll, ug1_bias=-11.0, half_swing=None,
                             ug2_filter=250.0)
        assert auto is not None
        inside = auto["half_swing"] * 0.5  # safely inside the data
        d = pp_distortion(pts, ll, ug1_bias=-11.0, half_swing=inside,
                          ug2_filter=250.0)
        assert d is not None
        assert d["manual_swing_clamped"] is False
        assert d["half_swing"] == pytest.approx(inside)

    def test_pp_panel_shows_clamp_notice(self, qapp, el84):
        from app.amplifier_tab import AmplifierTab
        from lm19.amp_engine import AmpParams, AmplifierEngine
        model, pts = el84
        eng = AmplifierEngine()
        eng.set_data(pts, is_triode=False)
        result = eng.analyze(AmpParams(
            ub=300.0, ra=8.0, ug1_bias=-11.0, half_swing=50.0,
            circuit=CIRCUIT_PP, pp_raa=8.0, pa_max=12.0, ug2_filter=250.0,
            hd_method=HD_METHOD_5POINT))
        assert result.pp_dist is not None
        assert result.pp_dist.get("manual_swing_clamped") is True
        tab = AmplifierTab()
        html = tab._format_pp_results(result)
        # notice = amp.swing_clamped with requested/used
        assert "50.0" in html


# -- ML-051: a tiny manual swing is a reachable error ----------------

class TestTinyManualSwingSurfaces:

    def _pts(self, el84):
        _, pts = el84
        return pts

    def test_5point_returns_none_and_diagnoses(self, el84):
        from lm19.amplifier import compute_distortion, find_intersections
        from lm19.amplifier.distortion import diagnose_distortion
        _, pts = el84
        ll = ResistiveLoadLine(300.0, 8.0)
        isects = find_intersections(pts, ll, ug2_filter=250.0,
                                    ug1_bias=-8.0)
        tiny = MIN_SWING_V / 2
        assert compute_distortion(isects, ug1_bias=-8.0,
                                  half_swing=tiny) is None
        assert diagnose_distortion(isects, ug1_bias=-8.0,
                                   half_swing=tiny) \
            == DIST_ERR_MANUAL_SWING_SMALL

    def test_chebyshev_returns_none_too(self, el84):
        from lm19.amplifier import (
            compute_distortion_chebyshev, find_intersections,
        )
        _, pts = el84
        ll = ResistiveLoadLine(300.0, 8.0)
        isects = find_intersections(pts, ll, ug2_filter=250.0,
                                    ug1_bias=-8.0)
        assert compute_distortion_chebyshev(
            isects, ug1_bias=-8.0, half_swing=MIN_SWING_V / 2) is None

    def test_auto_swing_none_still_works(self, el84):
        """half_swing=None (auto) is a legitimate path, untouched."""
        from lm19.amplifier import compute_distortion, find_intersections
        _, pts = el84
        ll = ResistiveLoadLine(300.0, 8.0)
        isects = find_intersections(pts, ll, ug2_filter=250.0,
                                    ug1_bias=-8.0)
        assert compute_distortion(isects, ug1_bias=-8.0,
                                  half_swing=None) is not None


class TestChebyshevClampFlags:
    """Both chebyshev variants used to clamp silently — the panel
    notice did not work for the chebyshev method; the PP variant
    also lacked the ML-051 tiny guard."""

    def test_se_chebyshev_flags_oversized_swing(self, el84):
        from lm19.amplifier import (
            compute_distortion_chebyshev, find_intersections,
        )
        _, pts = el84
        ll = ResistiveLoadLine(300.0, 8.0)
        isects = find_intersections(pts, ll, ug2_filter=250.0,
                                    ug1_bias=-8.0)
        d = compute_distortion_chebyshev(isects, ug1_bias=-8.0,
                                         half_swing=50.0)
        assert d is not None
        assert d["manual_swing_clamped"] is True
        assert d["requested_half_swing"] == pytest.approx(50.0)

    def test_pp_chebyshev_flags_oversized_swing(self, el84):
        from lm19.amplifier import compute_distortion_chebyshev_pp
        _, pts = el84
        ll = PushPullLoadLine(300.0, 8.0)
        d = compute_distortion_chebyshev_pp(
            pts, ll, ug1_bias=-11.0, half_swing=50.0, ug2_filter=250.0)
        assert d is not None
        assert d["manual_swing_clamped"] is True

    def test_pp_chebyshev_tiny_swing_is_error(self, el84):
        from lm19.amplifier import compute_distortion_chebyshev_pp
        _, pts = el84
        ll = PushPullLoadLine(300.0, 8.0)
        assert compute_distortion_chebyshev_pp(
            pts, ll, ug1_bias=-11.0, half_swing=MIN_SWING_V / 2,
            ug2_filter=250.0) is None

    def test_in_range_not_flagged(self, el84):
        from lm19.amplifier import (
            compute_distortion_chebyshev, find_intersections,
        )
        _, pts = el84
        ll = ResistiveLoadLine(300.0, 8.0)
        isects = find_intersections(pts, ll, ug2_filter=250.0,
                                    ug1_bias=-8.0)
        auto = compute_distortion_chebyshev(isects, ug1_bias=-8.0,
                                             half_swing=None)
        assert auto is not None
        inside = auto["half_swing"] * 0.8  # safely inside the data
        d = compute_distortion_chebyshev(isects, ug1_bias=-8.0,
                                         half_swing=inside)
        assert d is not None
        assert d["manual_swing_clamped"] is False


# -- ML-066: x0 clipped to bounds ------------------------------------

class TestX0ClippedToBounds:

    def _arrays(self, triode_12au7):
        _, pts = triode_12au7
        ua = np.array([p["ua"] for p in pts])
        ug1 = np.array([p["ug1"] for p in pts])
        ia = np.array([p["ia"] for p in pts]) / 1000.0
        return ua, ug1, ia

    def test_exotic_ref_does_not_crash_triode(self, triode_12au7):
        from lm19.spice_export import _fit_koren_scipy
        from lm19.tube_params import KorenParams
        ua, ug1, ia = self._arrays(triode_12au7)
        ref = KorenParams(mu=17.0, ex=1.35, kg1=60000.0,  # kg1 > hi=50000
                          kp=5000.0, kvb=300.0)           # kp > hi
        params, _cost, _conv = _fit_koren_scipy(ua, ug1, ia, ref_koren=ref)
        assert len(params) == 5 and np.all(np.isfinite(params))

    def test_low_side_ref_does_not_crash(self, triode_12au7):
        """Clipping only the upper bound would let a reference BELOW
        the bounds through (mu=1.0 < lo, kvb=1 < lo)."""
        from lm19.spice_export import _fit_koren_scipy
        from lm19.tube_params import KorenParams
        ua, ug1, ia = self._arrays(triode_12au7)
        ref = KorenParams(mu=1.0, ex=0.5, kg1=5.0, kp=1.0, kvb=1.0)
        params, _cost, _conv = _fit_koren_scipy(ua, ug1, ia, ref_koren=ref)
        assert len(params) == 5 and np.all(np.isfinite(params))

    def test_exotic_ref_does_not_crash_pentode(self, el84):
        from lm19.spice_export import _fit_pentode_scipy
        from lm19.tube_params import KorenParams
        _, pts = el84
        ua = np.array([p["ua"] for p in pts])
        ug1 = np.array([p["ug1"] for p in pts])
        ug2 = np.array([p.get("ug2", 250.0) for p in pts])
        ia = np.array([p["ia"] for p in pts]) / 1000.0
        ig2 = np.array([p.get("ig2", 0.0) for p in pts]) / 1000.0
        ref = KorenParams(mu=12.0, ex=1.35, kg1=600.0, kp=120.0,
                          kvb=12.0, kg2=50000.0)  # kg2 > hi=20000
        params, _cost, _conv = _fit_pentode_scipy(
            ua, ug1, ug2, ia, ig2, ref_koren=ref)
        assert len(params) == 6 and np.all(np.isfinite(params))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
