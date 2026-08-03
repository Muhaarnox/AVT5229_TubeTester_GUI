"""Equivalence pins for the optimizer vectorization pack.

The optimizer hot paths were vectorized (ia_array kernels, one-call
intersection search, masked vector Newton for DFT, PP transfer-curve
cache, per-Ug1 pre-grouping). All of them promise BIT-IDENTICAL results
to the former scalar paths — these tests pin that promise:

  - ``ia_array`` == scalar ``ia()`` loop for Koren/Dempwolf/Reefman + UL
  - ``find_intersections(curves=...)`` == plain path
  - ``composite_characteristic(transfer=...)`` == plain path
  - ``pp_distortion(transfer=...)`` == plain path
  - vectorized DFT Newton == verbatim scalar reference loop
  - ``refine_optimum(cancelled=True)`` aborts, grid ``on_progress`` fires
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.amplifier import (
    PushPullLoadLine,
    ResistiveLoadLine,
    TransformerLoadLine,
    UltralinearModelWrapper,
    build_pp_transfer,
    composite_characteristic,
    compute_distortion_dft,
    find_intersections,
    find_intersections_model,
    pp_distortion,
)
from lm19.amplifier.constants import FIXED_POINT_CONVERGENCE_MA
from lm19.amplifier.distortion import group_curves_by_ug1
from lm19.optimizer import (
    OptimizerConstraints,
    optimize_measurements,
    optimize_model,
    refine_optimum,
)
from lm19.tube_model_base import model_ia_array
from lm19.tube_sim import quick_pentode, quick_triode
from lm19.amplifier.constants import (
    HD_METHOD_AUTO,
)

UG1_CURVES = [-2.0, -4.0, -6.0, -8.0, -10.0, -12.0]


@pytest.fixture(scope="module")
def el84():
    model, points = quick_pentode("EL84")
    return model, points


# ── ia_array == scalar ia() ──────────────────────────────────────────

class TestIaArrayEquivalence:
    UA = np.linspace(1.0, 400.0, 173)      # non-round count on purpose

    def _check(self, model, ug2: float) -> None:
        for ug1 in (-1.5, -7.0, -12.0):
            scalar = np.array([model.ia(float(u), ug1, ug2) for u in self.UA])
            vec = model.ia_array(self.UA, ug1, ug2)
            assert np.array_equal(scalar, vec)

    def test_koren_pentode(self, el84) -> None:
        self._check(el84[0], 250.0)

    def test_koren_triode(self) -> None:
        model, _ = quick_triode("12AX7")
        self._check(model, 0.0)

    def test_dempwolf_pentode(self, el84) -> None:
        from lm19.dempwolf import fit_dempwolf
        model = fit_dempwolf(el84[1], "pentode").model
        self._check(model, 250.0)

    def test_reefman_pentode(self, el84) -> None:
        from lm19.reefman import fit_reefman
        model = fit_reefman(el84[1], "pentode").model
        self._check(model, 250.0)

    def test_ul_wrapper_modulates_screen(self, el84) -> None:
        w = UltralinearModelWrapper(el84[0], ug2_nom=250.0, tap=0.43)
        scalar = np.array([w.ia(float(u), -7.0) for u in self.UA])
        assert np.array_equal(scalar, w.ia_array(self.UA, -7.0))

    def test_helper_fallback_matches_fast_path(self, el84) -> None:
        model = el84[0]

        class Bare:
            """Protocol-shaped model WITHOUT ia_array → scalar-loop fallback."""
            model_type = "koren"
            name = topology = "pentode"
            pa_max, uh, ih = 12.0, 6.3, 0.76

            def ia(self, ua, ug1, ug2=0.0):
                return model.ia(ua, ug1, ug2)

            def ig2(self, ua, ug1, ug2):
                return 0.0

            def generate_scan(self, grid):
                return []

            def params_dict(self):
                return {}

        fast = model_ia_array(model, self.UA, -7.0, 250.0)
        slow = model_ia_array(Bare(), self.UA, -7.0, 250.0)
        assert np.array_equal(fast, slow)


# ── intersection search paths ────────────────────────────────────────

class TestIntersectionEquivalence:
    def test_grouped_curves_match_plain_path(self, el84) -> None:
        _, points = el84
        for ug2_f in (250.0, None):
            curves = group_curves_by_ug1(points, ug2_filter=ug2_f)
            for ub, ra in ((300.0, 5.0), (250.0, 2.0), (350.0, 10.0)):
                ll = ResistiveLoadLine(ub, ra)
                plain = find_intersections(points, ll, ug2_filter=ug2_f)
                fast = find_intersections(points, ll, ug2_filter=ug2_f,
                                          curves=curves)
                assert plain == fast

    def test_model_intersections_physical(self, el84) -> None:
        model, _ = el84
        ll = ResistiveLoadLine(300.0, 5.0)
        isects = find_intersections_model(
            model, ll, UG1_CURVES, ug2=250.0, ua_range=(1.0, 400.0))
        assert len(isects) == len(UG1_CURVES)
        for p in isects:
            # every intersection lies ON the load line and conducts
            assert p["ia"] == pytest.approx(ll.ia_at_ua(p["ua"]), abs=1e-3)
            assert p["ia"] > 0.0

    def test_transformer_bias_path_still_works(self, el84) -> None:
        model, _ = el84
        ll = TransformerLoadLine(300.0, ra_dc=0.1, ra_ac=5.0)
        isects = find_intersections_model(
            model, ll, UG1_CURVES, ug2=250.0, ua_range=(1.0, 400.0),
            ug1_bias=-7.0)
        assert len(isects) >= 3
        # AC line lets Ua exceed Ub — the signature of the Q-point path
        assert max(p["ua"] for p in isects) > 300.0


# ── PP transfer cache ────────────────────────────────────────────────

class TestPPTransferEquivalence:
    def test_composite_with_transfer_identical(self, el84) -> None:
        _, points = el84
        transfer = build_pp_transfer(points, None, 250.0)
        for bias in (-6.0, -8.0):
            plain = composite_characteristic(points, ug1_bias=bias,
                                             ug2_filter=250.0)
            fast = composite_characteristic(points, ug1_bias=bias,
                                            ug2_filter=250.0,
                                            transfer=transfer)
            assert plain == fast

    def test_pp_distortion_with_transfer_identical(self, el84) -> None:
        _, points = el84
        ll = PushPullLoadLine(300.0, ra_aa=8.0)
        # transfer is Ub-dependent (Ia taken at Ua=Ub) — the provided pair
        # must be built at the load line's supply to match the plain path.
        transfer = build_pp_transfer(points, None, 250.0, ua_ref=ll.ub)
        plain = pp_distortion(points, ll, -7.0, ug2_filter=250.0)
        fast = pp_distortion(points, ll, -7.0, ug2_filter=250.0,
                             transfer=transfer)
        assert plain == fast
        assert 0.0 < plain["thd"] < 70.0        # physical sanity


# ── vectorized DFT Newton vs verbatim scalar reference ───────────────

class TestDftNewtonEquivalence:
    def test_se_matches_scalar_reference(self, el84) -> None:
        """The vector Newton must reproduce the former per-sample scalar
        loop bit-for-bit (same cold start, same update sequence)."""
        model, _ = el84
        ll = ResistiveLoadLine(300.0, 5.0)
        n = 128                                  # keep the reference fast
        ug2 = 250.0

        def reference_ia_t(bias: float, hs: float) -> np.ndarray:
            ll_fn = ll.ia_at_ua
            t = np.arange(n)
            ug1_t = bias + hs * np.cos(2.0 * np.pi * t / n)
            eps = ll.endpoints()
            ua_guess = (eps[0][0] + eps[1][0]) / 2.0
            ia_t = np.zeros(n)
            for i in range(n):
                g = float(ug1_t[i])
                ua = ua_guess
                for _ in range(30):
                    err = model.ia(ua, g, ug2) - ll_fn(ua)
                    if abs(err) < FIXED_POINT_CONVERGENCE_MA:
                        break
                    err2 = model.ia(ua + 0.5, g, ug2) - ll_fn(ua + 0.5)
                    deriv = (err2 - err) / 0.5
                    if abs(deriv) > 1e-6:
                        ua -= err / deriv
                    else:
                        ua += 1.0 if err > 0 else -1.0
                    ua = max(0.0, ua)
                ia_t[i] = max(0.0, model.ia(ua, g, ug2))
            return ia_t

        for bias, hs in ((-7.0, 4.0), (-9.0, 6.0)):   # incl. near-clipping
            ref = reference_ia_t(bias, hs)
            spec = np.fft.rfft(ref)
            mag = np.abs(spec) * 2.0 / n
            hd = {k: mag[k] / mag[1] * 100.0 for k in range(2, 10)}
            thd_ref = math.sqrt(sum(v * v for v in hd.values()))
            r = compute_distortion_dft(model, ll, ug1_bias=bias,
                                       half_swing=hs, ug2=ug2, ub=300.0,
                                       n_samples=n)
            assert r["thd"] == pytest.approx(thd_ref, abs=1e-9)

    def test_degenerate_derivative_uses_kick_step(self) -> None:
        """|deriv| ≤ 1e-6 → ±1 V kick instead of a Newton step (the branch
        never fires on smooth models, so pin it directly)."""
        from lm19.amplifier.distortion import (
            _NEWTON_KICK_STEP_V,
            _newton_solve_vec,
        )

        calls: list = []

        def flat_ia(ua_arr, _g):
            calls.append(np.array(ua_arr))
            return np.full_like(np.asarray(ua_arr, dtype=float), 5.0)

        ua, n_div, _ = _newton_solve_vec(
            flat_ia, lambda ua_arr: np.zeros_like(np.asarray(ua_arr)),
            np.array([-7.0]), 100.0, max_iter=3)
        # flat err=+5 → three kicks of +1 V from 100
        assert ua[0] == pytest.approx(100.0 + 3 * _NEWTON_KICK_STEP_V)
        assert n_div == 1                        # never converged

    def test_pp_dft_matched_pair_cancels_even_harmonics(self, el84) -> None:
        from lm19.amplifier import compute_distortion_dft_pp
        model, _ = el84
        ll = PushPullLoadLine(300.0, ra_aa=8.0)
        r = compute_distortion_dft_pp(model, ll, ug1_bias=-7.0,
                                      half_swing=4.0, ug2=250.0)
        assert r is not None
        assert r["hd2"] < 1e-4   # odd-symmetric composite (solver-tolerance noise ~1e-6..1e-5)
        assert 0.0 < r["thd"] < 70.0
        assert r["pout_mw"] > 0.0
        assert r["iq_per_tube"] > 0.0


# ── cancellation + progress plumbing ─────────────────────────────────

class TestRefineCancelAndProgress:
    def _constraints(self) -> OptimizerConstraints:
        return OptimizerConstraints(
            pa_max_w=12.0, ug1_range=(-12.0, -2.0), ra_range=(2.0, 10.0),
            ub_range=(250.0, 350.0), ug2_range=(200.0, 300.0),
            ub_steps=2, ug2_steps=2, ra_steps=4, ug1_steps=4,
            hd_method=HD_METHOD_AUTO,
        )

    def test_refine_cancelled_returns_none(self, el84) -> None:
        model, _ = el84
        c = self._constraints()
        res = optimize_model(model, c, ug1_values=UG1_CURVES)
        assert res.best is not None
        out = refine_optimum(res.best, points=None, model=model,
                             constraints=c, ug1_values=UG1_CURVES,
                             cancelled=lambda: True)
        assert out is None

    def test_grid_progress_monotonic_and_complete(self, el84) -> None:
        model, points = el84
        c = self._constraints()
        seen: list = []
        optimize_model(model, c, ug1_values=UG1_CURVES,
                       on_progress=lambda d, t: seen.append((d, t)))
        assert seen, "on_progress never fired"
        dones = [d for d, _ in seen]
        assert dones == sorted(dones)
        assert seen[-1][0] == seen[-1][1]       # finished the whole grid

        seen.clear()
        c2 = self._constraints()
        c2.ug2_range = None
        optimize_measurements(points, ub=300.0, constraints=c2,
                              ug2_values=[250.0],
                              on_progress=lambda d, t: seen.append((d, t)))
        assert seen and seen[-1][0] == seen[-1][1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
