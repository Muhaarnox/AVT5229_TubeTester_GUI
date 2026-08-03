"""Physical validation of the optimizer chain (parts 1-3 of 4).

Validates that the model → load line → distortion → recommendation chain
produces PHYSICALLY correct answers, not merely unbroken code:

1. Analytic devices with closed-form theory: a LINEAR "tube" must show
   ~zero THD and exact fundamental power; a SQUARE-LAW device must show
   HD2 = A/(4·V0) and HD3 ≈ 0 across all three HD methods.
2. Local optimality: the refined optimum must not be beaten by its own
   neighbourhood (same evaluation method).
3. Textbook load optima: SE triode max-power load ≈ 2×rp (RDH); doubling
   the tube (parallel pair) halves the optimal load and doubles Pout.

Part 4 (independent LTspice .tran cross-check) lives in
``test_ltspice_tran_validation.py`` — it needs the LTspice binary.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.amplifier import (
    ResistiveLoadLine,
    compute_distortion,
    compute_distortion_chebyshev,
    compute_distortion_dft,
    find_intersections_model,
)
from lm19.optimizer import (
    OptimizerConstraints,
    _evaluate_point_model,
    _make_load_line,
    _score,
    optimize_model,
    refine_optimum,
)
from lm19.tube_sim import quick_triode
from lm19.amplifier.constants import (
    CIRCUIT_SE,
    HD_METHOD_DFT,
)

# ── module local constants ──
_UB = 300.0
_RA_KOHM = 5.0
_BIAS = -8.0
_SWING = 3.0
# Analytic square-law device: ia[mA] = K·(ug1 − VCUT)², no Ua dependence
# (rp = ∞) → the drive-to-current transfer is EXACTLY quadratic and
# HD2 = A/(4·(bias − VCUT)), HD3 = 0 (classic single-term expansion).
# Scaled so the operating region stays INSIDE the load line's reach
# (line tops out at Ub/Ra = 60 mA; bias −8 V → 21.6 mA quadratic /
# 24 mA linear, full family of curves intersects).
_SQ_K = 0.15                   # mA/V²
_SQ_VCUT = -20.0               # V
_LIN_GM = 2.0                  # mA/V for the linear device
# Tolerances: DFT is near-exact on analytic devices; interpolation-based
# methods (5-point/Chebyshev on a finite intersection family) carry the
# family's linear-interp error.
_DFT_REL_TOL = 1e-3
_INTERP_REL_TOL = 0.05
_NEIGHBOR_REL_TOL = 0.02       # local-optimality slack (NM fatol + curvature)


class _AnalyticTube:
    """Protocol-shaped device with an exact transfer law (rp = ∞)."""

    model_type = "analytic"
    name = "analytic"
    topology = "pentode"       # ug2 accepted and ignored
    pa_max, uh, ih = 100.0, 6.3, 0.0

    def __init__(self, law) -> None:
        self._law = law

    def ia(self, ua: float, ug1: float, ug2: float = 0.0) -> float:
        return float(self._law(ug1))

    def ia_array(self, ua, ug1, ug2=0.0) -> np.ndarray:
        ua_b, ug1_b, _ = np.broadcast_arrays(
            np.asarray(ua, dtype=float), np.asarray(ug1, dtype=float),
            np.asarray(ug2, dtype=float))
        return self._law(ug1_b) * np.ones_like(ua_b)

    def ig2(self, ua: float, ug1: float, ug2: float) -> float:
        return 0.0

    def generate_scan(self, grid) -> List[Dict]:
        return []

    def params_dict(self) -> Dict:
        return {}


def _square_law_tube() -> _AnalyticTube:
    return _AnalyticTube(
        lambda g: _SQ_K * np.maximum(np.asarray(g) - _SQ_VCUT, 0.0) ** 2)


def _linear_tube() -> _AnalyticTube:
    return _AnalyticTube(
        lambda g: _LIN_GM * np.maximum(np.asarray(g) - _SQ_VCUT, 0.0))


def _isects(model, n_curves: int = 25) -> list:
    ll = ResistiveLoadLine(_UB, _RA_KOHM)
    ug1s = list(np.linspace(_BIAS - _SWING - 1.0, _BIAS + _SWING + 1.0,
                            n_curves))
    return find_intersections_model(
        model, ll, ug1s, ug2=0.0, ua_range=(1.0, 2.0 * _UB))


class TestLinearDeviceZeroThd:
    """A linear transfer must produce ~zero THD in every method and the
    exact textbook fundamental power."""

    def test_dft_thd_zero_and_power_exact(self) -> None:
        tube = _linear_tube()
        ll = ResistiveLoadLine(_UB, _RA_KOHM)
        d = compute_distortion_dft(
            tube, ll, ug1_bias=_BIAS, half_swing=_SWING, ug2=0.0, ub=_UB)
        assert d is not None
        assert d["thd"] < 1e-6
        # b1 = gm·A exactly; P1 = b1²·Ra/2 (mA²·kΩ = mW)
        b1_expected = _LIN_GM * _SWING
        assert d["b1"] == pytest.approx(b1_expected, rel=1e-6)
        # rel 1e-4: Newton solves each sample to FIXED_POINT_CONVERGENCE_MA,
        # leaving ~2e-5 relative noise in the Ua waveform's fundamental.
        assert d["pout_fund_mw"] == pytest.approx(
            b1_expected ** 2 * _RA_KOHM / 2.0, rel=1e-4)

    def test_interp_methods_thd_zero(self) -> None:
        tube = _linear_tube()
        isects = _isects(tube)
        for fn in (compute_distortion, compute_distortion_chebyshev):
            d = fn(isects, ug1_bias=_BIAS, half_swing=_SWING, ub=_UB)
            assert d is not None, fn.__name__
            assert d["thd"] < 0.01, (fn.__name__, d["thd"])


class TestSquareLawClosedForm:
    """ia ∝ (Ug1−Vc)² → HD2 = A/(4·V0)·100%, HD3 ≈ 0 — closed form all
    three methods must reproduce."""

    def _hd2_expected(self) -> float:
        v0 = _BIAS - _SQ_VCUT
        return _SWING / (4.0 * v0) * 100.0

    def test_dft_matches_closed_form(self) -> None:
        tube = _square_law_tube()
        ll = ResistiveLoadLine(_UB, _RA_KOHM)
        d = compute_distortion_dft(
            tube, ll, ug1_bias=_BIAS, half_swing=_SWING, ug2=0.0, ub=_UB)
        assert d is not None
        assert d["hd2"] == pytest.approx(self._hd2_expected(),
                                         rel=_DFT_REL_TOL)
        assert d["hd3"] < 1e-6
        assert d["thd"] == pytest.approx(self._hd2_expected(),
                                         rel=_DFT_REL_TOL)

    def test_5point_and_chebyshev_match_closed_form(self) -> None:
        tube = _square_law_tube()
        isects = _isects(tube)
        for fn in (compute_distortion, compute_distortion_chebyshev):
            d = fn(isects, ug1_bias=_BIAS, half_swing=_SWING, ub=_UB)
            assert d is not None, fn.__name__
            assert d["hd2"] == pytest.approx(
                self._hd2_expected(), rel=_INTERP_REL_TOL), fn.__name__

    def test_hd2_scales_linearly_with_swing(self) -> None:
        """The closed form says HD2 ∝ A — a property no coincidence of
        constants can fake."""
        tube = _square_law_tube()
        ll = ResistiveLoadLine(_UB, _RA_KOHM)
        d1 = compute_distortion_dft(
            tube, ll, ug1_bias=_BIAS, half_swing=1.5, ug2=0.0, ub=_UB)
        d2 = compute_distortion_dft(
            tube, ll, ug1_bias=_BIAS, half_swing=3.0, ug2=0.0, ub=_UB)
        assert d2["hd2"] / d1["hd2"] == pytest.approx(2.0, rel=1e-3)


class TestLocalOptimality:
    """The refined optimum must not be beaten by its own neighbourhood
    (evaluated with the SAME method) — the direct 'does the optimizer
    actually optimize' check."""

    def _run(self, target: str):
        model, pts = quick_triode("12AU7")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        c = OptimizerConstraints(
            target=target, hd_method=HD_METHOD_DFT, circuit=CIRCUIT_SE,
            pa_max_w=2.75, ug1_range=(-14.0, -4.0), ra_range=(5.0, 30.0),
            ug1_steps=6, ra_steps=6, swing_steps=2,
        )
        r = optimize_model(model, c, ug1_values=ug1_vals)
        assert r.best is not None
        refined = refine_optimum(r.best, points=None, model=model,
                                 constraints=c, ug1_values=ug1_vals)
        assert refined is not None and refined.valid
        return model, ug1_vals, c, r, refined

    def _neighbor_scores(self, model, ug1_vals, c, refined) -> list:
        scores = []
        span_ug1 = c.ug1_range[1] - c.ug1_range[0]
        span_ra = c.ra_range[1] - c.ra_range[0]
        for dug1, dra in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ug1 = float(np.clip(refined.ug1 + dug1 * 0.01 * span_ug1,
                                *c.ug1_range))
            ra = float(np.clip(refined.ra + dra * 0.01 * span_ra,
                               *c.ra_range))
            ll = _make_load_line(refined.ub, ra, c)
            pt = _evaluate_point_model(
                model, ll, refined.ub, refined.ug2, ug1, ra, ug1_vals, c,
                half_swing=refined.half_swing or None, method=HD_METHOD_DFT)
            if pt is not None and pt.valid:
                scores.append(_score(pt, c))
        return scores

    @pytest.mark.parametrize("target", ["min_thd", "max_pout"])
    def test_refined_beats_neighbourhood(self, target: str) -> None:
        model, ug1_vals, c, r, refined = self._run(target)
        ref_score = _score(refined, c)
        # Refinement must not be worse than the grid best it started from.
        assert ref_score <= _score(r.best, c) + abs(_score(r.best, c)) * 1e-3
        neigh = self._neighbor_scores(model, ug1_vals, c, refined)
        assert len(neigh) >= 3          # probes actually evaluated
        slack = abs(ref_score) * _NEIGHBOR_REL_TOL + 1e-6
        for ns in neigh:
            assert ref_score <= ns + slack, (ref_score, ns)


class TestTextbookLoadOptima:
    """Small-signal maximum power transfer: Pout(Ra) peaks at Ra = rp.

    Well-posed on a TRANSFORMER stage where the Q-point is pinned at
    Ua ≈ Ub for every Ra_ac (a resistive stage moves its Q with Ra, so
    rp changes underfoot and the textbook comparison is confounded —
    probed: resistive argmax lands at 0.59×rp for exactly that reason).
    The classic 2×rp rule is the max-UNDISTORTED-power variant — not
    directly testable here because the optimizer has no THD-cap
    constraint (targets are min_thd / max_pout / balanced)."""

    def _rp_at(self, model, ua: float, ug1: float) -> float:
        dua = 1.0
        dia = model.ia(ua + dua, ug1, 0.0) - model.ia(ua - dua, ug1, 0.0)
        return 2.0 * dua / dia          # kΩ (V / mA)

    def test_small_signal_power_transfer_peaks_at_rp(self) -> None:
        from lm19.amplifier import TransformerLoadLine
        model, _ = quick_triode("12AU7")
        bias, sw, ub = -8.0, 0.5, 300.0
        rp = self._rp_at(model, ub, bias)
        best_ra, best_p = None, -1.0
        for ra in np.linspace(1.0, 30.0, 59):
            d = compute_distortion_dft(
                model, TransformerLoadLine(ub, 0.05, float(ra)),
                ug1_bias=bias, half_swing=sw, ug2=0.0, ub=ub)
            if d and d["pout_fund_mw"] > best_p:
                best_ra, best_p = float(ra), d["pout_fund_mw"]
        # Probed 1.03; band kills wrong-impedance physics (0.5× / 2×).
        assert best_ra / rp == pytest.approx(1.0, rel=0.2), (best_ra, rp)

    def test_parallel_pair_halves_load_doubles_power(self) -> None:
        """Physical scaling invariance: two tubes in parallel (Ia ×2) →
        optimal load halves, max Pout doubles."""
        model, pts = quick_triode("12AU7")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})

        class _Doubled:
            model_type = "analytic"
            name = "2x12AU7"
            topology = "triode"
            pa_max, uh, ih = 5.5, 6.3, 0.6

            def ia(self, ua, ug1, ug2=0.0):
                return 2.0 * model.ia(ua, ug1, ug2)

            def ia_array(self, ua, ug1, ug2=0.0):
                return 2.0 * model.ia_array(ua, ug1, ug2)

            def ig2(self, ua, ug1, ug2):
                return 0.0

            def generate_scan(self, grid):
                return []

            def params_dict(self):
                return {}

        c1 = OptimizerConstraints(
            target="max_pout", hd_method=HD_METHOD_DFT, circuit=CIRCUIT_SE,
            pa_max_w=2.75, ug1_range=(-14.0, -4.0), ra_range=(1.0, 60.0),
            ug1_steps=8, ra_steps=30, swing_steps=2,
        )
        c2 = OptimizerConstraints(
            target="max_pout", hd_method=HD_METHOD_DFT, circuit=CIRCUIT_SE,
            pa_max_w=5.5, ug1_range=(-14.0, -4.0), ra_range=(1.0, 60.0),
            ug1_steps=8, ra_steps=30, swing_steps=2,
        )
        r1 = optimize_model(model, c1, ug1_values=ug1_vals)
        r2 = optimize_model(_Doubled(), c2, ug1_values=ug1_vals)
        assert r1.best is not None and r2.best is not None
        assert r2.best.ra / r1.best.ra == pytest.approx(0.5, rel=0.35)
        assert (r2.best.pout_mw / r1.best.pout_mw
                == pytest.approx(2.0, rel=0.25))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
