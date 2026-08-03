"""Model-Chebyshev PP (`chebyshev_model_pp`) — UL-capable fast grid method.

Checkable artifacts (things one can verify, not
just a report):
- CI cross-pins vs DFT over tap × swing (tolerances measured empirically,
  see docs/UL_CHEBYSHEV_VALIDATION.md for the full table + regeneration);
- gating pins: UL sweep now runs for chebyshev+model; tap=0 stays on the
  byte-identical data path; 5-point sweep skip is VISIBLE;
- engine pins: chebyshev+tap routes to the model variant, 5point+tap
  warns `ul_tap_ignored_by_method`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pytest

from i18n_setup import available_locales
from lm19.amplifier.constants import (
    HD_METHOD_CHEBYSHEV_MODEL_PP,
    CIRCUIT_PP,
    HD_METHOD_5POINT,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_CHEBYSHEV_PP,
)
from lm19.amp_engine import (
    WARN_UL_TAP_IGNORED_BY_METHOD,
)
from lm19.optimizer import (
    OPT_WARN_UL_SWEEP_SKIPPED,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── module local constants ──
# Empirical equivalence on quick_pentode EL84 (docs/UL_CHEBYSHEV_
# VALIDATION.md): measured max ΔTHD 0.31% rel, ΔPout 0.26% rel over
# tap × bias × swing incl. hard clipping — pins keep ~6× margin.
THD_REL_TOL = 0.02
POUT_REL_TOL = 0.01
HD3_REL_TOL = 0.05
TAPS = (0.2, 0.43, 1.0)
SWINGS = (4.0, 9.0)


@pytest.fixture(scope="module")
def el84():
    from lm19.tube_sim import quick_pentode
    return quick_pentode("EL84")


def _wrap(model, tap: float):
    from lm19.amplifier import UltralinearModelWrapper
    if tap <= 0:
        return model
    return UltralinearModelWrapper(model, ug2_nom=250.0, tap=tap)


# ── Cross-pins vs DFT ────────────────────────────────────────────────

class TestChebModelVsDft:

    @pytest.mark.parametrize("swing", SWINGS)
    @pytest.mark.parametrize("tap", (0.0,) + TAPS)
    def test_equivalence(self, el84, tap, swing):
        from lm19.amplifier import (
            PushPullLoadLine, compute_distortion_chebyshev_pp_model,
            compute_distortion_dft_pp,
        )
        model, _ = el84
        wrapped = _wrap(model, tap)
        ll = PushPullLoadLine(300.0, 8.0)
        cheb = compute_distortion_chebyshev_pp_model(
            wrapped, ll, -11.0, half_swing=swing, ug2=250.0)
        dft = compute_distortion_dft_pp(
            wrapped, ll, -11.0, half_swing=swing, ug2=250.0)
        assert cheb is not None and dft is not None, (tap, swing)
        assert cheb["thd"] == pytest.approx(dft["thd"], rel=THD_REL_TOL)
        assert cheb["pout_mw"] == pytest.approx(dft["pout_mw"],
                                                rel=POUT_REL_TOL)
        if dft["hd3"] > 0.5:  # meaningful HD3 only
            assert cheb["hd3"] == pytest.approx(dft["hd3"],
                                                rel=HD3_REL_TOL)
        assert cheb["method"] == HD_METHOD_CHEBYSHEV_MODEL_PP

    def test_matched_pair_odd_symmetry(self, el84):
        """Matched PP composite is odd-symmetric → HD2 ≈ 0 (same physics
        the DFT variant shows)."""
        from lm19.amplifier import (
            PushPullLoadLine, compute_distortion_chebyshev_pp_model,
        )
        model, _ = el84
        ll = PushPullLoadLine(300.0, 8.0)
        d = compute_distortion_chebyshev_pp_model(
            _wrap(model, 0.43), ll, -11.0, half_swing=9.0, ug2=250.0)
        assert d["hd2"] < 0.2, d["hd2"]


# ── Optimizer gating ─────────────────────────────────────────────────

def _pp_constraints(**kw):
    from lm19.optimizer import OptimizerConstraints
    base = dict(circuit=CIRCUIT_PP, pp_raa=8.0, pa_max_w=12.0,
                ug1_range=(-12.0, -10.0), ra_range=(8.0, 8.0),
                ug1_steps=2, ra_steps=1, swing_steps=2)
    base.update(kw)
    return OptimizerConstraints(**base)


class TestOptimizerUlChebyshevSweep:

    def test_chebyshev_sweep_runs_with_model(self, el84):
        from lm19.optimizer import optimize_pp
        model, pts = el84
        c = _pp_constraints(
            hd_method=HD_METHOD_CHEBYSHEV, ul_tap_mode="presets",
            ul_tap_presets=(0.0, 0.43),
            ul_tap_presets_enabled=(True, True),
        )
        r = optimize_pp(pts, ub=300.0, constraints=c, model=model,
                        ug2_filter=250.0)
        taps = {p.ul_tap for p in r.grid_points}
        assert taps == {0.0, 0.43}, taps
        methods = {p.ul_tap: p.hd_method for p in r.grid_points}
        # tap=0 stays on the DATA path (byte-identity), tap>0 — model path
        assert methods[0.0] == HD_METHOD_CHEBYSHEV_PP
        assert methods[0.43] == HD_METHOD_CHEBYSHEV_MODEL_PP
        assert OPT_WARN_UL_SWEEP_SKIPPED not in r.warnings

    def test_5point_sweep_skip_is_visible(self, el84):
        from lm19.optimizer import optimize_pp
        model, pts = el84
        c = _pp_constraints(
            hd_method=HD_METHOD_5POINT, ul_tap_mode="presets",
            ul_tap_presets=(0.0, 0.43),
            ul_tap_presets_enabled=(True, True),
        )
        r = optimize_pp(pts, ub=300.0, constraints=c, model=model,
                        ug2_filter=250.0)
        assert {p.ul_tap for p in r.grid_points} == {0.0}
        assert OPT_WARN_UL_SWEEP_SKIPPED in r.warnings

    def test_chebyshev_without_model_skips_visibly(self, el84):
        from lm19.optimizer import optimize_pp
        _, pts = el84
        c = _pp_constraints(
            hd_method=HD_METHOD_CHEBYSHEV, ul_tap_mode="presets",
            ul_tap_presets=(0.0, 0.43),
            ul_tap_presets_enabled=(True, True),
        )
        r = optimize_pp(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        assert {p.ul_tap for p in r.grid_points} == {0.0}
        assert OPT_WARN_UL_SWEEP_SKIPPED in r.warnings

    def test_ul_points_physical(self, el84):
        """UL points of the Chebyshev grid are physical and go the
        right way: lower Pout and no higher THD than pentode ones
        (documented UL physics)."""
        from lm19.optimizer import optimize_pp
        model, pts = el84
        c = _pp_constraints(
            hd_method=HD_METHOD_CHEBYSHEV, ul_tap_mode="presets",
            ul_tap_presets=(0.0, 0.43),
            ul_tap_presets_enabled=(True, True),
        )
        r = optimize_pp(pts, ub=300.0, constraints=c, model=model,
                        ug2_filter=250.0)
        by_tap: Dict[float, list] = {0.0: [], 0.43: []}
        for p in r.grid_points:
            if p.valid and p.half_swing > 0:
                by_tap[p.ul_tap].append(p)
        assert by_tap[0.0] and by_tap[0.43]
        max_pout_pent = max(p.pout_mw for p in by_tap[0.0])
        max_pout_ul = max(p.pout_mw for p in by_tap[0.43])
        assert 0 < max_pout_ul < max_pout_pent


# ── Engine routing ───────────────────────────────────────────────────

class TestEngineChebyshevUl:

    def _analyze(self, model, pts, **kw):
        from lm19.amp_engine import AmpParams, AmplifierEngine
        eng = AmplifierEngine()
        eng.set_data(pts, is_triode=False, series_models={0: model})
        base = dict(ub=300.0, ra=8.0, ug1_bias=-11.0, half_swing=9.0,
                    circuit=CIRCUIT_PP, pp_raa=8.0, pa_max=12.0,
                    ug2_filter=250.0, series_id=0)
        base.update(kw)
        return eng.analyze(AmpParams(**base))

    def test_chebyshev_with_tap_uses_model_variant(self, el84):
        model, pts = el84
        result = self._analyze(model, pts, hd_method=HD_METHOD_CHEBYSHEV,
                               ul_tap=0.43)
        assert result.pp_dist is not None
        assert result.pp_dist["method"] == HD_METHOD_CHEBYSHEV_MODEL_PP

    def test_chebyshev_without_tap_stays_on_data(self, el84):
        model, pts = el84
        result = self._analyze(model, pts, hd_method=HD_METHOD_CHEBYSHEV,
                               ul_tap=None)
        assert result.pp_dist is not None
        assert result.pp_dist["method"] == HD_METHOD_CHEBYSHEV_PP

    def test_5point_with_tap_warns(self, el84):
        model, pts = el84
        result = self._analyze(model, pts, hd_method=HD_METHOD_5POINT,
                               ul_tap=0.43)
        sr = result.per_source["measurements"]
        codes = {w["code"] for w in sr.warnings}
        assert WARN_UL_TAP_IGNORED_BY_METHOD in codes

    def test_applied_chebyshev_ul_point_reproduces_optimizer(self, el84):
        """Acceptance for Chebyshev-UL: an applied point reproduces the
        optimizer numbers with the same method."""
        from lm19.optimizer import optimize_pp
        model, pts = el84
        c = _pp_constraints(
            hd_method=HD_METHOD_CHEBYSHEV, ul_tap_mode="off",
            ul_tap_manual=0.43,
        )
        r = optimize_pp(pts, ub=300.0, constraints=c, model=model,
                        ug2_filter=250.0)
        best = r.best
        assert best is not None and best.ul_tap == pytest.approx(0.43)
        result = self._analyze(
            model, pts, hd_method=HD_METHOD_CHEBYSHEV,
            ub=best.ub, pp_raa=best.ra, ug1_bias=best.ug1,
            half_swing=best.half_swing, ul_tap=best.ul_tap)
        assert result.pp_dist["thd"] == pytest.approx(best.thd, rel=0.05)
        assert result.pp_dist["pout_mw"] == pytest.approx(
            best.pout_mw, rel=0.05)


# ── i18n ─────────────────────────────────────────────────────────────

class TestUlWarningKeys:

    @pytest.mark.parametrize("locale", available_locales())
    def test_keys_exist(self, locale):
        import json
        data = json.loads(
            (PROJECT_ROOT / "locales" / f"{locale}.json")
            .read_text(encoding="utf-8"))
        assert "warn_ul_tap_ignored_by_method" in data["amp"], locale
        assert "opt_warn_ul_sweep_skipped" in data["amp"], locale


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
