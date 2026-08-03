"""THD-cap constraint (``OptimizerConstraints.thd_max_pct``).

Datasheet-style semantics: with target=max_pout the cap answers
"Pout at X% THD" — grid points over the cap keep swing-sweep
eligibility (``cap_only_fail``), phase 2 finds them a compliant
reduced swing, refine pushes Pout up to the cap boundary.

Run:  py -m pytest tests/test_optimizer_thd_cap.py -v
"""

import json
from pathlib import Path

import pytest

from i18n_setup import available_locales

from lm19.amplifier import ResistiveLoadLine
from lm19.optimizer import (
    OptimizerConstraints,
    _build_opt_point,
    _swing_sweep_candidates,
    optimize_measurements,
    optimize_pp,
    refine_optimum,
    TOP_N_FOR_SWING,
)
from tests._fixtures import make_triode_points
from lm19.amplifier.constants import (
    CIRCUIT_PP,
    HD_METHOD_5POINT,
)
from lm19.optimizer import (
    OPT_ERR_NO_POINTS_WITHIN_CONSTRAINTS,
    OPT_ERR_NO_POINTS_WITHIN_THD_CAP,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── module local constants ──
# Grid kept at ≤ TOP_N_FOR_SWING operating points so phase 2 sweeps
# EVERY candidate — makes the swing-rescue pin deterministic (no
# dependence on score ordering inside the top-N cut).
_UG1_STEPS = 4
_RA_STEPS = 5
_SWING_STEPS = 5


def _constraints(**kw) -> OptimizerConstraints:
    base = dict(
        target="max_pout",
        pa_max_w=100.0,
        ug1_range=(-9.0, -2.0),
        ra_range=(5.0, 20.0),
        ug1_steps=_UG1_STEPS,
        ra_steps=_RA_STEPS,
        swing_steps=_SWING_STEPS,
        hd_method=HD_METHOD_5POINT,
    )
    base.update(kw)
    return OptimizerConstraints(**base)


def _dist(thd: float, pout_mw: float = 500.0) -> dict:
    """Minimal distortion dict for direct _build_opt_point calls."""
    return {
        "pout_mw": pout_mw, "ua_0": 200.0, "ia_0": 20.0,
        "thd": thd, "hd2": 1.0, "hd3": 0.5,
        "i_min": 5.0, "half_swing": 2.0,
    }


def _build(thd: float, pout_mw: float = 500.0, **ckw):
    c = _constraints(**ckw)
    ll = ResistiveLoadLine(250.0, 10.0)
    return _build_opt_point(_dist(thd, pout_mw), 250.0, 0.0, -5.0, 10.0,
                            [], ll, c)


# ═══════════════════════════════════════════════════════════════════
#  Unit: cap predicate in the point builder
# ═══════════════════════════════════════════════════════════════════

class TestCapPredicate:

    def test_cap_off_high_thd_valid(self):
        pt = _build(thd=50.0, thd_max_pct=0.0)
        assert pt.valid is True
        assert pt.cap_only_fail is False

    def test_boundary_thd_equals_cap_passes(self):
        """≤ semantics: thd == cap is compliant."""
        pt = _build(thd=5.0, thd_max_pct=5.0)
        assert pt.valid is True
        assert pt.cap_only_fail is False

    def test_over_cap_invalid_but_swing_eligible(self):
        pt = _build(thd=5.01, thd_max_pct=5.0)
        assert pt.valid is False
        assert pt.cap_only_fail is True

    def test_hard_fail_plus_cap_not_swing_eligible(self):
        """cap_only_fail requires the HARD constraints to pass: a smaller
        swing cannot fix a Pa violation (Pa is a Q-point property), so
        such points must not enter the phase-2 candidate pool."""
        # dist: ua_0=200 V × ia_0=20 mA → Pa = 4 W > 1 W limit
        pt = _build(thd=5.01, thd_max_pct=5.0, pa_max_w=1.0)
        assert pt.valid is False
        assert pt.cap_only_fail is False

    def test_candidates_include_cap_only_fail(self):
        c = _constraints(thd_max_pct=5.0)
        good = _build(thd=2.0, thd_max_pct=5.0)
        capped = _build(thd=8.0, thd_max_pct=5.0)
        hard = _build(thd=8.0, thd_max_pct=5.0, pa_max_w=1.0)
        cands = _swing_sweep_candidates([good, capped, hard], c)
        assert good in cands
        assert capped in cands
        assert hard not in cands

    def test_candidates_respect_top_n(self):
        """Mutation-audit: input must be SHUFFLED — ascending
        input let a no-sort mutation survive (first-N happened to equal
        best-N)."""
        c = _constraints(target="min_thd", thd_max_pct=0.0)
        n = TOP_N_FOR_SWING + 10
        # deterministic permutation of 0..n-1 (stride 7, gcd(7, n) = 1)
        order = [(i * 7) % n for i in range(n)]
        pts = [_build(thd=float(v)) for v in order]
        cands = _swing_sweep_candidates(pts, c)
        assert len(cands) == TOP_N_FOR_SWING
        # min_thd target: exactly the N lowest-THD points selected
        selected = {p.thd for p in cands}
        excluded = {p.thd for p in pts} - selected
        assert max(selected) < min(excluded)

    def test_candidates_order_follows_target(self):
        """Mutation-audit: with target=max_pout the top-N cut
        must be by SCORE (highest Pout), not by THD — a thd-keyed sort
        survived the original pins (all test grids fit inside top-N)."""
        c = _constraints(target="max_pout", thd_max_pct=0.0)
        n = TOP_N_FOR_SWING + 10
        # equal THD, ascending pout: a thd-sort (stable) or no-sort keeps
        # input order and picks the LOWEST-pout points
        pts = [_build(thd=1.0, pout_mw=100.0 * (i + 1)) for i in range(n)]
        cands = _swing_sweep_candidates(pts, c)
        assert len(cands) == TOP_N_FOR_SWING
        selected = {p.pout_mw for p in cands}
        excluded = {p.pout_mw for p in pts} - selected
        assert min(selected) > max(excluded)


# ═══════════════════════════════════════════════════════════════════
#  Integration: SE (optimize_measurements)
# ═══════════════════════════════════════════════════════════════════

class TestCapFilterSE:

    @pytest.fixture(scope="class")
    def pts(self):
        return make_triode_points()

    def _run(self, pts, **ckw):
        return optimize_measurements(pts, ub=250.0,
                                     constraints=_constraints(**ckw))

    def test_cap_off_no_flags(self, pts):
        r = self._run(pts, thd_max_pct=0.0)
        assert r.error is None
        assert all(not p.cap_only_fail for p in r.grid_points)

    def test_cap_above_all_equivalent_to_off(self, pts):
        r_off = self._run(pts, thd_max_pct=0.0)
        r_high = self._run(pts, thd_max_pct=1000.0)
        assert r_high.error is None
        assert (r_high.best.ug1, r_high.best.ra, r_high.best.thd,
                r_high.best.pout_mw) == \
               (r_off.best.ug1, r_off.best.ra, r_off.best.thd,
                r_off.best.pout_mw)
        assert (sum(p.valid for p in r_high.grid_points)
                == sum(p.valid for p in r_off.grid_points))

    def test_cap_filters_valid_set_and_best(self, pts):
        r0 = self._run(pts, thd_max_pct=0.0)
        t0 = r0.best.thd
        assert t0 > 0.01, "precondition: max_pout best must have real THD"
        cap = t0 * 0.5
        r = self._run(pts, thd_max_pct=cap)
        assert r.error is None
        assert all(p.thd <= cap for p in r.grid_points if p.valid)
        assert r.best.thd <= cap
        # capping cannot increase the achievable Pout
        assert r.best.pout_mw <= r0.best.pout_mw + 1e-6

    def test_swing_sweep_rescues_capped_points(self, pts):
        """THE discriminating pin vs a naive filter: pick a cap below the
        min THD of the grid phase (swing_steps=1 run — grid points carry
        the auto/max swing) but above the min THD reachable via the
        phase-2 swing sweep. A naive filter (cap-failed points barred
        from top-N) skips phase 2 entirely → error; the two-flag design
        finds the compliant reduced swing."""
        r_grid = self._run(pts, thd_max_pct=0.0, swing_steps=1)
        r0 = self._run(pts, thd_max_pct=0.0)
        t_full = min(p.thd for p in r_grid.grid_points if p.valid)
        t_red = min(p.thd for p in r0.grid_points if p.valid)
        assert t_red < t_full, \
            "precondition: THD must fall with reduced swing on this data"
        cap = (t_red + t_full) / 2.0

        # phase 2 disabled → every (full-swing) point over cap → error
        r1 = self._run(pts, thd_max_pct=cap, swing_steps=1)
        assert r1.error == OPT_ERR_NO_POINTS_WITHIN_THD_CAP
        assert any(p.cap_only_fail for p in r1.grid_points)

        # phase 2 enabled → reduced-swing rescue
        r2 = self._run(pts, thd_max_pct=cap, swing_steps=_SWING_STEPS)
        assert r2.error is None
        assert r2.best is not None
        assert r2.best.thd <= cap

    def test_all_capped_specific_error(self, pts):
        r = self._run(pts, thd_max_pct=1e-6)
        assert r.error == OPT_ERR_NO_POINTS_WITHIN_THD_CAP
        assert r.best is None

    def test_hard_fail_keeps_generic_error(self, pts):
        """No cap_only_fail points → the generic constraints error (its
        advice about Pa/Pout is then the accurate one)."""
        r = self._run(pts, thd_max_pct=1e-6, pa_max_w=1e-9)
        assert r.error == OPT_ERR_NO_POINTS_WITHIN_CONSTRAINTS

    def test_mixed_failures_specific_error(self, pts):
        """Mutation-audit: ANY cap-only failure makes the
        specific error correct (raising the cap alone yields valid
        points) — an any→all mutation survived because the original
        all-capped pin had a homogeneous grid. Grid Pa spans
        0.39–2.82 W here, so pa_max=1.0 splits it."""
        r = self._run(pts, thd_max_pct=1e-6, pa_max_w=1.0)
        flags = [(p.valid, p.cap_only_fail) for p in r.grid_points]
        assert any(cof for _, cof in flags), "precondition: cap-only present"
        assert any(not v and not cof for v, cof in flags), \
            "precondition: pure hard-fail present"
        assert r.error == OPT_ERR_NO_POINTS_WITHIN_THD_CAP

    @pytest.mark.parametrize("target", ["min_thd", "max_pout", "balanced"])
    def test_all_targets_respect_cap(self, pts, target):
        r0 = self._run(pts, target="min_thd", thd_max_pct=0.0)
        cap = r0.best.thd * 1.5 + 0.01   # feasible by construction
        r = self._run(pts, target=target, thd_max_pct=cap)
        assert r.error is None
        assert r.best.thd <= cap


# ═══════════════════════════════════════════════════════════════════
#  Integration: refine honours the cap (penalty wall)
# ═══════════════════════════════════════════════════════════════════

class TestCapRefine:

    def test_refined_point_within_cap(self):
        pts = make_triode_points()
        c0 = _constraints(thd_max_pct=0.0)
        r0 = optimize_measurements(pts, ub=250.0, constraints=c0)
        cap = r0.best.thd * 0.5
        c = _constraints(thd_max_pct=cap)
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.best is not None
        refined = refine_optimum(r.best, points=pts, model=None,
                                 constraints=c)
        assert refined is not None
        assert refined.valid
        assert refined.thd <= cap


# ═══════════════════════════════════════════════════════════════════
#  Integration: PP (optimize_pp)
# ═══════════════════════════════════════════════════════════════════

class TestCapFilterPP:

    def test_pp_cap_filters(self):
        pts = make_triode_points()
        r0 = optimize_pp(pts, ub=250.0,
                         constraints=_constraints(circuit=CIRCUIT_PP,
                                                  thd_max_pct=0.0))
        assert r0.error is None
        thds = sorted(p.thd for p in r0.grid_points if p.valid and p.thd > 0)
        assert len(thds) >= 4, "precondition: PP grid must have THD spread"
        cap = thds[len(thds) // 2]   # median → some points above, some below
        r = optimize_pp(pts, ub=250.0,
                        constraints=_constraints(circuit=CIRCUIT_PP,
                                                 thd_max_pct=cap))
        assert r.error is None
        assert all(p.thd <= cap for p in r.grid_points if p.valid)
        assert r.best.thd <= cap
        if max(thds) > cap:
            assert any(p.cap_only_fail for p in r.grid_points)

    def test_pp_class_a_fail_not_cap_eligible(self):
        """Mutation-audit: the PP cap block must come AFTER
        the class-A filter — the class-A threshold is a Q-point property
        (P_A = Iq²·Ra_aa/8), a smaller swing cannot fix it, so such
        points are NOT cap_only_fail and the error stays generic.
        Moving the block above the filter survived the original pins."""
        pts = make_triode_points()
        r = optimize_pp(pts, ub=250.0,
                        constraints=_constraints(
                            circuit=CIRCUIT_PP, thd_max_pct=1e-6,
                            class_a_power_mode="absolute",
                            class_a_power_value=1e6))
        assert r.grid_points, "precondition: grid evaluated"
        assert not any(p.cap_only_fail for p in r.grid_points)
        assert r.error == OPT_ERR_NO_POINTS_WITHIN_CONSTRAINTS


# ═══════════════════════════════════════════════════════════════════
#  i18n: keys present in BOTH locales
# ═══════════════════════════════════════════════════════════════════

class TestCapI18n:

    @pytest.mark.parametrize("locale", available_locales())
    def test_keys_exist(self, locale):
        data = json.loads((PROJECT_ROOT / "locales" / f"{locale}.json")
                          .read_text(encoding="utf-8"))
        amp = data["amp"]
        for k in ("opt_thd_max", "opt_thd_max_tip",
                  "opt_err_no_points_within_thd_cap"):
            assert k in amp, (locale, k)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
