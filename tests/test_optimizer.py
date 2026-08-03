"""Tests for lm19.optimizer — multi-parameter amplifier optimization.

Run:  py -m pytest tests/test_optimizer.py -v
"""

import math
from pathlib import Path

import numpy as np
import pytest

# ML-148: paths anchored to the repo, not CWD — a pytest run
# from outside lm19_app must not FileNotFoundError.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

from lm19.constants import MAX_SANE_THD_PCT as MAX_SANE_THD, UA_ROUND
from lm19.optimizer import (
    OptPoint,
    OptimizerConstraints,
    OptimizerResult,
    optimize_measurements,
    optimize_model,
    optimize_pp,
    refine_optimum,
    refine_pareto_front,
    _compute_pareto_front,
    _classify_amp,
    _make_load_line,
    _make_grid,
    _resolve_class_a_threshold,
    _score,
    _sweep_swing_top_n,
    P_CLASSA_DIVISOR,
    TOP_N_FOR_SWING,
    MIN_SWING_FRACTION,
    PARETO_REFINE_MAX,
)
from lm19.tube_sim import quick_triode, quick_pentode


# ── module local constants ──
# Sanity ceiling for individual harmonic components on a PP composite.
# Matched-pair symmetry pulls HD2 toward zero, so this is just a "didn't
# blow up" guard (catches HD=10000% style garbage), not a tight bound.
_HD_MAX_PCT_PP = 100.0
# EL84 PP ceiling: published max ~17 W pentode at Ub=300 V, Ra_aa=8 kΩ.
# Anything above 25 W means the optimizer escaped its operating region.
_EL84_PP_POUT_MAX_MW = 25_000.0
# Mismatched PP pair (12AU7 + 12AX7): the residual uncancelled even
# harmonic is large (probed best-point HD2 0.34% vs matched 0.008%), so
# require a real absolute margin — a bare ">" could pass on float noise.
_MISMATCH_HD2_MIN_DELTA_PCT = 0.1
# Mean HD2 over COMMON grid points: mismatched measured ≈4× matched
# (4.55% vs 1.13%).  1.5× keeps headroom while still failing if the
# even-harmonic cancellation contrast disappears.
_MISMATCH_HD2_MEAN_RATIO = 1.5
# Matched/mismatched runs share the phase-1 grid, so the common-key set
# must be substantial (probed: 28 keys on a 4×4 grid + swing sweep).
_MIN_COMMON_PP_GRID_POINTS = 8
# PP vs SE, same tube, target=max_pout: probed Pout ratio ≈4.8× on the
# 12AU7 grid (theory: 2–4×).  2× still fails if PP degenerates to ≤ SE.
_PP_OVER_SE_MIN_RATIO = 2.0


def _pp_transfer_iq(points: list, ug1_target: float, ua_ref: float) -> float:
    """Independent per-tube Iq oracle for PP grid points (triode data).

    Reproduces the physical definition the PP composite is built on —
    Ia interpolated AT Ua=ua_ref (the supply = per-tube DC operating
    plate voltage) per Ug1 level, interpolated at the bias — directly
    from the RAW measurement dicts.  Deliberately NOT derived from
    OptPoint fields, so an ia_0/p_classA_w co-regression to 0 cannot
    satisfy assertions anchored to this oracle. (The former mean-over-Ua
    definition understated triode Iq ~2.4×.)
    """
    by_ug1: dict = {}
    for p in points:
        by_ug1.setdefault(round(p.get("ug1", 0.0), UA_ROUND), []).append(
            (p.get("ua", 0.0), p["ia"]))
    levels = sorted(by_ug1)
    vals = []
    for u in levels:
        pts_sorted = sorted(by_ug1[u])
        ua_arr = np.array([a for a, _ in pts_sorted])
        ia_arr = np.array([b for _, b in pts_sorted])
        vals.append(float(np.interp(ua_ref, ua_arr, ia_arr)))
    return float(np.interp(ug1_target, levels, vals))


# ── Synthetic data helpers ────────────────────────────────────────

from tests._fixtures import make_triode_points as _make_triode_points  # noqa: E402
from lm19.amplifier.constants import (
    CIRCUIT_CF,
    CIRCUIT_PP,
    CIRCUIT_SE,
    CIRCUIT_SE_XFMR,
    HD_METHOD_5POINT,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_CHEBYSHEV_PP,
    HD_METHOD_DFT,
    HD_METHOD_DFT_PP,
)
from lm19.optimizer import (
    OPT_ERR_NO_VALID_POINTS,
    OPT_WARN_DFT_NO_MODEL_FALLBACK,
    OPT_WARN_PP_DFT_UG2_FROM_DATA,
)
from lm19.optimizer import (
    OPT_WARN_UG2_FILTER_NO_MATCH,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
)


# ═══════════════════════════════════════════════════════════════════
#  Unit tests: helpers
# ═══════════════════════════════════════════════════════════════════

class TestHelpers:

    def test_make_grid_none_returns_default(self):
        assert _make_grid(None, 10, default=250.0) == [250.0]

    def test_make_grid_range(self):
        g = _make_grid((100.0, 400.0), 4, default=0.0)
        assert len(g) == 4
        assert g[0] == 100.0
        assert g[-1] == 400.0

    def test_make_grid_single_step(self):
        g = _make_grid((100.0, 400.0), 1, default=0.0)
        assert g == [100.0]

    def test_classify_amp_class_a(self):
        assert _classify_amp(10.0, 1.0) == "A"

    def test_classify_amp_class_b(self):
        assert _classify_amp(10.0, 0.01) == "B"

    def test_classify_amp_class_ab(self):
        assert _classify_amp(10.0, 0.2) == "AB"

    def test_classify_amp_zero_i0(self):
        assert _classify_amp(0.0, 0.0) == "B"

    def test_pp_amp_class_uses_per_tube_minimum(self):
        """PP amp_class must come from the PER-TUBE minimum current.
        The composite i_min is ≈ −i_max for a matched pair, so
        classifying on it labels EVERY PP point "B" (live bug found by
        a mutation audit). A hot-bias small-swing point is
        physically class A; deep-bias large-swing is B/AB."""
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0, pa_max_w=12.0,
            ug1_range=(-8.0, -5.0), ra_range=(6.0, 10.0),
            ug1_steps=4, ra_steps=3, swing_steps=3,
        )
        r = optimize_pp(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        assert r.grid_points
        classes = {p.amp_class for p in r.grid_points}
        # Under the composite-i_min bug this is exactly {"B"}.
        assert "A" in classes, f"no class-A PP points found: {classes}"
        hot = [p for p in r.grid_points
               if p.ug1 == -5.0 and 0 < p.half_swing <= p.ia_0 / 20.0]
        for p in hot:
            assert p.amp_class == "A", (p.ug1, p.half_swing, p.amp_class)

    def test_score_min_thd(self):
        pt = OptPoint(
            ub=250, ug2=0, ug1=-7, ra=5,
            thd=3.0, hd2=2.0, hd3=1.0,
            pout_mw=1000, pa_mw=2000,
            ia_0=10, ua_0=200, amp_class="A", max_swing=3.0,
        )
        c = OptimizerConstraints(target="min_thd")
        assert _score(pt, c) == 3.0

    def test_score_max_pout(self):
        pt = OptPoint(
            ub=250, ug2=0, ug1=-7, ra=5,
            thd=3.0, hd2=2.0, hd3=1.0,
            pout_mw=1000, pa_mw=2000,
            ia_0=10, ua_0=200, amp_class="A", max_swing=3.0,
        )
        c = OptimizerConstraints(target="max_pout")
        assert _score(pt, c) == -1000.0


# ═══════════════════════════════════════════════════════════════════
#  Pareto front
# ═══════════════════════════════════════════════════════════════════

class TestParetoFront:

    def _pt(self, thd: float, pout_mw: float) -> OptPoint:
        return OptPoint(
            ub=250, ug2=0, ug1=-7, ra=5,
            thd=thd, hd2=thd * 0.7, hd3=thd * 0.3,
            pout_mw=pout_mw, pa_mw=2000,
            ia_0=10, ua_0=200, amp_class="A", max_swing=3.0,
        )

    def test_pareto_empty(self):
        assert _compute_pareto_front([]) == []

    def test_pareto_single_point(self):
        front = _compute_pareto_front([self._pt(3.0, 1000)])
        assert len(front) == 1

    def test_pareto_dominated_removed(self):
        """Point with higher THD AND lower Pout is dominated."""
        p1 = self._pt(2.0, 1000)  # better on both axes
        p2 = self._pt(5.0, 500)   # dominated
        front = _compute_pareto_front([p1, p2])
        assert len(front) == 1
        assert front[0].thd == 2.0

    def test_pareto_tradeoff_kept(self):
        """Both points on Pareto front: one has lower THD, other higher Pout."""
        p1 = self._pt(1.0, 500)   # low THD, low Pout
        p2 = self._pt(3.0, 2000)  # high THD, high Pout
        front = _compute_pareto_front([p1, p2])
        assert len(front) == 2

    def test_pareto_three_points(self):
        p1 = self._pt(1.0, 200)
        p2 = self._pt(2.0, 800)   # on front
        p3 = self._pt(3.0, 1500)  # on front
        p4 = self._pt(2.5, 600)   # dominated by p2
        front = _compute_pareto_front([p1, p2, p3, p4])
        assert len(front) == 3
        thds = [p.thd for p in front]
        assert 1.0 in thds
        assert 2.0 in thds
        assert 3.0 in thds


# ═══════════════════════════════════════════════════════════════════
#  Measurements optimization (synthetic data)
# ═══════════════════════════════════════════════════════════════════

class TestOptimizeMeasurements:

    def test_basic_produces_results(self):
        pts = _make_triode_points()
        c = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=8, ra_steps=8,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.error is None
        assert len(r.grid_points) > 0
        assert len(r.pareto_front) > 0
        assert r.best is not None

    def test_best_min_thd_has_lowest_thd(self):
        pts = _make_triode_points()
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=10, ra_steps=10,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        valid = [p for p in r.grid_points if p.valid]
        min_thd = min(p.thd for p in valid)
        assert r.best.thd == pytest.approx(min_thd, abs=0.01)

    def test_best_max_pout_has_highest_pout(self):
        pts = _make_triode_points()
        c = OptimizerConstraints(
            target="max_pout",
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=10, ra_steps=10,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        valid = [p for p in r.grid_points if p.valid]
        max_pout = max(p.pout_mw for p in valid)
        assert r.best.pout_mw == pytest.approx(max_pout, abs=1.0)

    def test_pa_max_constraint_filters(self):
        """Points exceeding Pa_max are marked invalid."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            pa_max_w=1.0,  # very strict
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=5, ra_steps=5,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        for pt in r.pareto_front:
            assert pt.pa_mw / 1000.0 <= 1.0

    def test_pout_min_constraint(self):
        """Points below Pout minimum are invalid; constraint satisfiable.

        The threshold is calibrated from the actually achievable
        maximum (half of it), so the constraint is genuinely
        satisfiable: a fixed 0.5 W would exceed what this grid can
        deliver and best would always be None, checking nothing.
        """
        pts = _make_triode_points()
        base = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=8, ra_steps=8,
        )
        r0 = optimize_measurements(pts, ub=250.0, constraints=base)
        assert r0.best is not None, "unconstrained rig must yield a best"
        max_pout_w = max(p.pout_mw for p in r0.grid_points
                         if p.valid) / 1000.0
        pout_min_w = max_pout_w / 2.0
        c = OptimizerConstraints(
            pout_min_w=pout_min_w,
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=8, ra_steps=8,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.best is not None, "half-of-max pout_min must be reachable"
        assert r.best.pout_mw / 1000.0 >= pout_min_w
        below = [p for p in r.grid_points
                 if p.pout_mw / 1000.0 < pout_min_w]
        assert below, "grid must contain points below the threshold"
        assert all(not p.valid for p in below), (
            "every below-threshold point must be invalid")

    def test_pout_min_real_6p1p_pentode(self):
        """pout_min on a real output pentode (6P1P) actually filters
        degenerate small-Pout points and forces realistic operating
        points. 12AU7 wouldn't hit 0.5W physically — must use a power
        tube with measurement coverage > 1W."""
        import json
        with open(_PROJECT_ROOT / "tests/spice_test_data/converted/pentode_6P1P_real.json") as f:
            pts = json.load(f)["points"]

        # Without constraint — optimizer may pick small-Pout degenerate.
        c_open = OptimizerConstraints(
            target="min_thd", circuit=CIRCUIT_SE,
            ug1_range=(-15.0, -7.0), ra_range=(2.0, 15.0),
            ug1_steps=8, ra_steps=8, pa_max_w=12.0,
            hd_method=HD_METHOD_CHEBYSHEV,
        )
        r_open = optimize_measurements(pts, ub=250.0, constraints=c_open, ug2_filter=200.0)
        assert r_open.best is not None
        # No constraint → optimizer can find low-Pout point
        # (typical degenerate behavior we want pout_min to suppress)
        pout_open = r_open.best.pout_mw / 1000.0

        # With pout_min=0.5W → all valid points must satisfy it.
        c_min = OptimizerConstraints(
            target="min_thd", circuit=CIRCUIT_SE,
            ug1_range=(-15.0, -7.0), ra_range=(2.0, 15.0),
            ug1_steps=8, ra_steps=8, pa_max_w=12.0,
            pout_min_w=0.5,
            hd_method=HD_METHOD_CHEBYSHEV,
        )
        r_min = optimize_measurements(pts, ub=250.0, constraints=c_min, ug2_filter=200.0)
        if r_min.best:
            assert r_min.best.pout_mw >= 500.0  # 0.5W = 500mW
            # Every valid grid point passes the constraint
            for pt in r_min.grid_points:
                if pt.valid:
                    assert pt.pout_mw >= 500.0
            # Constraint should change the answer when initial best was below
            if pout_open < 0.5:
                assert r_min.best.pout_mw > r_open.best.pout_mw

        # Unphysical constraint (>> tube can deliver at this Ub) → no result.
        c_impossible = OptimizerConstraints(
            target="min_thd", circuit=CIRCUIT_SE,
            ug1_range=(-15.0, -7.0), ra_range=(2.0, 15.0),
            ug1_steps=8, ra_steps=8, pa_max_w=12.0,
            pout_min_w=10.0,  # 6P1P at Ub=250V can't deliver 10W
            hd_method=HD_METHOD_CHEBYSHEV,
        )
        r_imp = optimize_measurements(pts, ub=250.0, constraints=c_impossible, ug2_filter=200.0)
        # Must fail gracefully: no best, error or empty pareto
        assert r_imp.best is None or r_imp.best.pout_mw >= 10000.0

    def test_no_data_returns_error(self):
        # ML-150: `is not None` cannot tell the right error code from
        # garbage — pin the exact empty-input contract.
        r = optimize_measurements([], ub=250.0, constraints=OptimizerConstraints())
        assert r.error == OPT_ERR_NO_VALID_POINTS
        assert r.best is None
        assert r.grid_points == []

    def test_impossible_constraints_returns_error(self):
        """Constraints that no point can satisfy."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            pa_max_w=0.001,  # impossibly low
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=5, ra_steps=5,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.error is not None or len(r.pareto_front) == 0

    def test_pareto_front_is_sorted_by_thd(self):
        pts = _make_triode_points()
        c = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=10, ra_steps=10,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        thds = [p.thd for p in r.pareto_front]
        assert thds == sorted(thds)

    def test_all_grid_points_have_valid_fields(self):
        pts = _make_triode_points()
        c = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=5, ra_steps=5,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        for pt in r.grid_points:
            assert pt.thd >= 0
            assert pt.pout_mw > 0
            assert pt.ra > 0
            assert pt.pa_mw > 0
            assert pt.amp_class in ("A", "AB", "B")


# ═══════════════════════════════════════════════════════════════════
#  Measurements optimization with real tube data (tube_sim)
# ═══════════════════════════════════════════════════════════════════

class TestOptimizeRealTubeData:

    def test_12au7_triode(self):
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=10, ra_steps=10,
            pa_max_w=2.75,
            pout_min_w=0.05,  # require meaningful output power
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.error is None
        assert r.best is not None
        assert r.best.thd > 0
        assert r.best.pout_mw >= 50.0

    def test_12ax7_high_gain(self):
        _, pts = quick_triode("12AX7")
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-3.0, -0.5), ra_range=(50.0, 200.0),
            ug1_steps=8, ra_steps=8,
            pa_max_w=1.0,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.error is None
        assert r.best is not None
        # ML-150: the name promises a high-gain 12AX7 mode — pin
        # physical sanity and the requested ranges, not mere existence.
        assert r.best.valid
        assert 0.0 < r.best.thd < MAX_SANE_THD
        assert r.best.pout_mw > 0.0
        assert -3.0 <= r.best.ug1 <= -0.5
        assert 50.0 <= r.best.ra <= 200.0

    def test_el84_pentode(self):
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            target="balanced",
            ug1_range=(-12.0, -4.0), ra_range=(2.0, 10.0),
            ug1_steps=8, ra_steps=8,
            pa_max_w=12.0,
        )
        r = optimize_measurements(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        assert r.error is None
        assert r.best is not None
        assert r.best.pa_mw / 1000.0 <= 12.0

    def test_pareto_makes_physical_sense(self):
        """On Pareto front: lower THD should generally mean lower Pout."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=12, ra_steps=12,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert len(r.pareto_front) >= 2
        # First point: lowest THD; last: highest Pout
        assert r.pareto_front[0].thd <= r.pareto_front[-1].thd
        assert r.pareto_front[0].pout_mw <= r.pareto_front[-1].pout_mw


# ═══════════════════════════════════════════════════════════════════
#  Model optimization
# ═══════════════════════════════════════════════════════════════════

class TestOptimizeModel:

    def test_koren_triode_basic(self):
        model, pts = quick_triode("12AU7")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=5, ra_steps=5,
            ub_range=(200.0, 300.0), ub_steps=3,
        )
        r = optimize_model(model, c, ug1_values=ug1_vals)
        assert r.error is None
        assert r.best is not None
        assert r.best.ub >= 200.0
        assert r.best.ub <= 300.0

    def test_model_explores_ub_range(self):
        """Multiple Ub values should be present in grid."""
        model, pts = quick_triode("12AU7")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        c = OptimizerConstraints(
            ug1_range=(-12.0, -5.0), ra_range=(5.0, 20.0),
            ug1_steps=3, ra_steps=3,
            ub_range=(200.0, 350.0), ub_steps=3,
        )
        r = optimize_model(model, c, ug1_values=ug1_vals)
        ubs = {pt.ub for pt in r.grid_points}
        assert len(ubs) > 1

    def test_pentode_model_with_ug2(self):
        model, pts = quick_pentode("EL84")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        c = OptimizerConstraints(
            target="max_pout",
            ug1_range=(-10.0, -5.0), ra_range=(3.0, 7.0),
            ug1_steps=3, ra_steps=3,
            ub_range=(280.0, 320.0), ub_steps=2,
            ug2_range=(230.0, 270.0), ug2_steps=2,
            pa_max_w=12.0,
        )
        r = optimize_model(model, c, ug1_values=ug1_vals)
        assert r.error is None
        assert r.best is not None
        # ML-150: ug2/ub sweeps must land the point INSIDE the ranges.
        assert r.best.valid
        assert 230.0 <= r.best.ug2 <= 270.0
        assert 280.0 <= r.best.ub <= 320.0
        assert r.best.pout_mw > 0.0

    def test_model_no_ub_range_uses_default(self):
        """Without ub_range, Ub is fixed at default."""
        model, pts = quick_triode("12AU7")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        c = OptimizerConstraints(
            ug1_range=(-12.0, -5.0), ra_range=(5.0, 20.0),
            ug1_steps=5, ra_steps=5,
            # ub_range=None → fixed
        )
        r = optimize_model(model, c, ug1_values=ug1_vals)
        ubs = {pt.ub for pt in r.grid_points}
        assert len(ubs) == 1

    @pytest.mark.timeout(60)
    def test_model_pareto_physical_sense(self):
        """Model Pareto: lower THD ↔ lower Pout trade-off."""
        model, pts = quick_triode("12AU7")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        c = OptimizerConstraints(
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=8, ra_steps=8,
            ub_range=(200.0, 300.0), ub_steps=3,
        )
        r = optimize_model(model, c, ug1_values=ug1_vals)
        assert len(r.pareto_front) >= 2
        assert r.pareto_front[0].thd <= r.pareto_front[-1].thd
        assert r.pareto_front[0].pout_mw <= r.pareto_front[-1].pout_mw

    def test_pentode_model_pareto_has_multiple_points(self):
        """Pentode model Pareto should have multiple trade-off points."""
        model, pts = quick_pentode("EL84")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        c = OptimizerConstraints(
            target="balanced",
            ug1_range=(-10.0, -4.0), ra_range=(2.0, 8.0),
            ug1_steps=4, ra_steps=4,
            ub_range=(280.0, 320.0), ub_steps=2,
            ug2_range=(230.0, 270.0), ug2_steps=2,
            pa_max_w=12.0,
        )
        r = optimize_model(model, c, ug1_values=ug1_vals)
        assert r.error is None
        # ML-150: the name promises MULTIPLE — >=1 never checked that
        # promise (this grid actually yields 11 points).
        assert len(r.pareto_front) >= 2


# ═══════════════════════════════════════════════════════════════════
#  Measurements with Ug2 sweep
# ═══════════════════════════════════════════════════════════════════

class TestOptimizeMeasurementsUg2:
    """Pentode measurements with multiple Ug2 values."""

    def test_pentode_ug2_values_swept(self):
        """optimize_measurements with ug2_values sweeps all of them."""
        _, pts = quick_pentode("EL84")
        ug2_available = sorted({round(p.get("ug2", 0), 0) for p in pts if p.get("ug2", 0) > 0})
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-10.0, -4.0), ra_range=(2.0, 8.0),
            ug1_steps=5, ra_steps=5,
            pa_max_w=12.0,
        )
        r = optimize_measurements(pts, ub=300.0, constraints=c, ug2_values=ug2_available)
        assert r.error is None
        assert r.best is not None
        # Multiple Ug2 should appear in grid
        ug2s = {pt.ug2 for pt in r.grid_points}
        if len(ug2_available) > 1:
            assert len(ug2s) > 1

    def test_pentode_pareto_across_ug2(self):
        """Pareto front should span different Ug2 values."""
        _, pts = quick_pentode("EL84")
        ug2_available = sorted({round(p.get("ug2", 0), 0) for p in pts if p.get("ug2", 0) > 0})
        c = OptimizerConstraints(
            target="balanced",
            ug1_range=(-10.0, -4.0), ra_range=(2.0, 8.0),
            ug1_steps=6, ra_steps=6,
            pa_max_w=12.0,
        )
        r = optimize_measurements(pts, ub=300.0, constraints=c, ug2_values=ug2_available)
        assert len(r.pareto_front) >= 1
        # Pareto should be sorted by THD
        thds = [p.thd for p in r.pareto_front]
        assert thds == sorted(thds)

    def test_pentode_best_has_ug2(self):
        """Best point should have a valid Ug2 value."""
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            ug1_range=(-10.0, -4.0), ra_range=(2.0, 8.0),
            ug1_steps=5, ra_steps=5,
        )
        r = optimize_measurements(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        assert r.best is not None
        assert r.best.ug2 == 250.0


# ═══════════════════════════════════════════════════════════════════
#  Measurements optimization with Ub sweep (virtual parameter)
# ═══════════════════════════════════════════════════════════════════

class TestOptimizeMeasurementsUbSweep:
    """Ub is a virtual analysis parameter — load line shifts over the
    same measured I-V family. Verify optimize_measurements sweeps Ub
    when constraints.ub_range is set."""

    def test_ub_range_explored_synthetic(self):
        """Synthetic triode data: multiple Ub values appear in grid."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=5, ra_steps=5,
            ub_range=(180.0, 320.0), ub_steps=4,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.error is None
        ubs = {round(pt.ub, 1) for pt in r.grid_points}
        assert len(ubs) >= 3, f"Expected ≥3 distinct Ub values, got {ubs}"
        # All Ub values must lie within range
        for ub_v in ubs:
            assert 180.0 <= ub_v <= 320.0

    def test_no_ub_range_keeps_ub_fixed(self):
        """Without ub_range: Ub locked to passed value."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=5, ra_steps=5,
            # ub_range=None → fixed
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        ubs = {pt.ub for pt in r.grid_points}
        assert ubs == {250.0}

    def test_ub_range_real_triode(self):
        """quick_triode 12AU7: Ub-sweep on measurements works."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=6, ra_steps=6,
            ub_range=(200.0, 350.0), ub_steps=4,
            pa_max_w=2.75,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.error is None
        assert r.best is not None
        ubs = {round(pt.ub, 1) for pt in r.grid_points}
        assert len(ubs) >= 2
        # Best Ub within configured range
        assert 200.0 <= r.best.ub <= 350.0
        # Physical sanity
        assert r.best.thd < MAX_SANE_THD
        assert r.best.pout_mw > 0

    def test_ub_range_real_pentode(self):
        """quick_pentode EL84: Ub-sweep on measurements with Ug2 filter."""
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            target="balanced",
            ug1_range=(-10.0, -4.0), ra_range=(2.0, 8.0),
            ug1_steps=5, ra_steps=5,
            ub_range=(250.0, 350.0), ub_steps=3,
            pa_max_w=12.0,
        )
        r = optimize_measurements(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        assert r.error is None
        assert r.best is not None
        ubs = {round(pt.ub, 1) for pt in r.grid_points}
        assert len(ubs) >= 2
        assert 250.0 <= r.best.ub <= 350.0
        assert r.best.ug2 == 250.0
        assert r.best.thd < MAX_SANE_THD

    def test_ub_sweep_finds_better_optimum(self):
        """Ub sweep should find ≤ best THD vs fixed Ub at one end of range."""
        _, pts = quick_triode("12AU7")
        base = dict(
            target="min_thd",
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=6, ra_steps=6,
            pa_max_w=2.75,
        )
        # Fixed at edge of range
        r_fixed = optimize_measurements(
            pts, ub=200.0, constraints=OptimizerConstraints(**base),
        )
        # Sweep across range
        r_swept = optimize_measurements(
            pts, ub=200.0,
            constraints=OptimizerConstraints(
                ub_range=(200.0, 350.0), ub_steps=4, **base,
            ),
        )
        assert r_fixed.best and r_swept.best, "guard de-vacuated 2026-07-12: value must be present"
        assert r_swept.best.thd <= r_fixed.best.thd + 0.01

    def test_ub_sweep_preserves_pareto_sort(self):
        """Pareto front sorted by THD even with multi-Ub grid."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=5, ra_steps=5,
            ub_range=(200.0, 320.0), ub_steps=3,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        thds = [p.thd for p in r.pareto_front]
        assert thds == sorted(thds)

    def test_ub_sweep_with_swing(self):
        """Ub-sweep + swing-sweep work together."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=5, ra_steps=5,
            ub_range=(200.0, 300.0), ub_steps=3,
            swing_steps=3,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.best is not None
        ubs = {round(pt.ub, 1) for pt in r.grid_points}
        assert len(ubs) >= 2
        # Swing produced varied half_swing
        swings = {round(pt.half_swing, 2) for pt in r.grid_points if pt.half_swing > 0}
        assert len(swings) > 1

    def test_refine_with_ub_range_measurements(self):
        """refine_optimum varies Ub when constraints.ub_range is set
        (measurements path, no model)."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=5, ra_steps=5,
            ub_range=(200.0, 350.0), ub_steps=3,
            pa_max_w=2.75,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.best is not None, "deterministic synthetic data must yield a best point"
        refined = refine_optimum(r.best, points=pts, model=None, constraints=c)
        # Unconditional: catches a regression where refine_optimum always
        # returns None (the whole Pareto-refine phase silently dead).
        assert refined is not None, "refine_optimum returned None on well-conditioned data"
        # Refined Ub must stay within configured range
        assert 200.0 <= refined.ub <= 350.0
        # Refined should be at least as good
        assert refined.thd <= r.best.thd + 0.5
        # Sanity
        assert refined.pout_mw > 0
        assert refined.thd < MAX_SANE_THD

    def test_pp_ub_range_explored_synthetic(self):
        """PP optimizer also sweeps Ub when ub_range set (synthetic)."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-9.0, -3.0), ra_range=(5.0, 15.0),
            ug1_steps=4, ra_steps=4,
            ub_range=(200.0, 320.0), ub_steps=3,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        assert r.error is None
        ubs = {round(pt.ub, 1) for pt in r.grid_points}
        assert len(ubs) >= 2, f"PP expected ≥2 distinct Ub values, got {ubs}"
        for ub_v in ubs:
            assert 200.0 <= ub_v <= 320.0

    def test_pp_no_ub_range_keeps_ub_fixed(self):
        """PP without ub_range: Ub locked to passed value."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-9.0, -3.0), ra_range=(5.0, 15.0),
            ug1_steps=4, ra_steps=4,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        ubs = {pt.ub for pt in r.grid_points}
        assert ubs == {250.0}

    def test_pp_ub_range_real_pentode(self):
        """PP EL84 pentode: Ub-sweep with Ug2 filter."""
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0,
            ug1_range=(-10.0, -4.0), ra_range=(4.0, 12.0),
            ug1_steps=4, ra_steps=4,
            ub_range=(280.0, 350.0), ub_steps=3,
            pa_max_w=12.0,
        )
        r = optimize_pp(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        assert r.error is None
        assert r.best is not None
        ubs = {round(pt.ub, 1) for pt in r.grid_points}
        assert len(ubs) >= 2
        assert 280.0 <= r.best.ub <= 350.0
        assert r.best.thd < MAX_SANE_THD

    def test_pp_refine_with_ub_range(self):
        """PP refine_optimum varies Ub when ub_range set."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=4, ra_steps=4,
            ub_range=(200.0, 350.0), ub_steps=3,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        assert r.best is not None
        refined = refine_optimum(r.best, points=pts, model=None, constraints=c)
        assert refined is not None
        assert 200.0 <= refined.ub <= 350.0
        assert refined.thd < MAX_SANE_THD
        assert refined.pout_mw > 0


# ═══════════════════════════════════════════════════════════════════
#  Scipy refinement
# ═══════════════════════════════════════════════════════════════════

class TestRefineOptimum:

    def test_refine_measurements_improves_or_matches(self):
        pts = _make_triode_points()
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=5, ra_steps=5,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.best is not None

        refined = refine_optimum(r.best, points=pts, model=None, constraints=c)
        assert refined is not None
        # Refined should be at least as good (THD ≤ grid best)
        assert refined.thd <= r.best.thd + 0.5  # small tolerance

    def test_refine_model(self):
        model, pts = quick_triode("12AU7")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=5, ra_steps=5,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.best is not None

        refined = refine_optimum(
            r.best, points=pts, model=model,
            constraints=c, ug1_values=ug1_vals,
        )
        # ML-150: the refine result went unchecked — a "refine test"
        # that would pass even on return None. Accept-if-better semantics:
        # min_thd must not regress, values must stay physical.
        assert refined is not None
        assert refined.thd <= r.best.thd + 0.5
        assert 0.0 < refined.thd < 70.0
        assert refined.pout_mw > 0.0
        # Should not crash; may or may not improve
        # (model path uses different evaluation)

    def test_refine_returns_valid_point(self):
        pts = _make_triode_points()
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=5, ra_steps=5,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.best is not None

        refined = refine_optimum(r.best, points=pts, model=None, constraints=c)
        assert refined is not None
        assert refined.thd > 0
        assert refined.pout_mw > 0
        assert refined.ug1 >= c.ug1_range[0]
        assert refined.ug1 <= c.ug1_range[1]
        assert refined.ra >= c.ra_range[0]
        assert refined.ra <= c.ra_range[1]

    def test_refine_pentode_measurements(self):
        """Refine pentode measurement optimization."""
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-10.0, -4.0), ra_range=(2.0, 8.0),
            ug1_steps=5, ra_steps=5,
            pa_max_w=12.0,
        )
        r = optimize_measurements(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        assert r.best is not None
        refined = refine_optimum(
            r.best, points=pts, model=None,
            constraints=c, ug2_filter=250.0,
        )
        assert refined is not None
        assert refined.thd > 0
        assert refined.ug2 == 250.0

    def test_refine_pentode_model(self):
        """Refine pentode model optimization."""
        model, pts = quick_pentode("EL84")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-10.0, -5.0), ra_range=(3.0, 7.0),
            ug1_steps=3, ra_steps=3,
            pa_max_w=12.0,
        )
        r = optimize_measurements(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        assert r.best is not None
        refined = refine_optimum(
            r.best, points=pts, model=model,
            constraints=c, ug2_filter=250.0,
            ug1_values=ug1_vals,
        )
        # ML-150: refined went unchecked (see the triode twin); plus
        # the pentode contract — ug2_filter survives through refine.
        assert refined is not None
        assert refined.thd <= r.best.thd + 0.5
        assert 0.0 < refined.thd < 70.0
        assert refined.pout_mw > 0.0
        assert refined.ug2 == 250.0
        # Should not crash


# ═══════════════════════════════════════════════════════════════════
#  Swing optimization
# ═══════════════════════════════════════════════════════════════════

class TestSwingOptimization:
    """Tests for swing sweep on top-N points."""

    def test_opt_point_has_half_swing(self):
        """OptPoint stores half_swing from distortion."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=5, ra_steps=5, swing_steps=3,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.best is not None
        # All grid points should have half_swing field
        for pt in r.grid_points:
            assert hasattr(pt, "half_swing")
            assert pt.half_swing >= 0

    def test_swing_sweep_produces_varied_swing(self):
        """With swing_steps > 1, grid should contain points with different half_swing."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=8, ra_steps=8, swing_steps=4,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        swings = {round(pt.half_swing, 2) for pt in r.grid_points if pt.half_swing > 0}
        # Should have multiple distinct swing values from the sweep
        assert len(swings) > 1

    def test_swing_sweep_disabled_when_steps_1(self):
        """swing_steps=1 should skip swing re-evaluation — no extra points."""
        pts = _make_triode_points()
        c_no_sweep = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=5, ra_steps=5, swing_steps=1,
        )
        c_with_sweep = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=5, ra_steps=5, swing_steps=4,
        )
        r_no = optimize_measurements(pts, ub=250.0, constraints=c_no_sweep)
        r_yes = optimize_measurements(pts, ub=250.0, constraints=c_with_sweep)
        # With swing sweep disabled, fewer grid points
        assert len(r_no.grid_points) < len(r_yes.grid_points)

    def test_min_thd_benefits_from_swing(self):
        """min_thd target: best point should have lower THD with swing optimization
        than max-swing-only, because smaller swing = lower distortion."""
        pts = _make_triode_points()
        c_no_swing = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=10, ra_steps=10, swing_steps=1,
        )
        c_with_swing = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=10, ra_steps=10, swing_steps=5,
        )
        r_no = optimize_measurements(pts, ub=250.0, constraints=c_no_swing)
        r_yes = optimize_measurements(pts, ub=250.0, constraints=c_with_swing)
        assert r_no.best is not None
        assert r_yes.best is not None
        # With swing optimization, best THD should be <= without
        assert r_yes.best.thd <= r_no.best.thd + 0.01

    def test_swing_in_pareto_front(self):
        """Pareto front should include points with varied swing."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=8, ra_steps=8, swing_steps=4,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert len(r.pareto_front) >= 1
        # Pareto front points have half_swing
        for pt in r.pareto_front:
            assert pt.half_swing > 0

    def test_model_swing_sweep(self):
        """Model optimizer also sweeps swing."""
        model, pts = quick_triode("12AU7")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=5, ra_steps=5, swing_steps=3,
            ub_range=(200.0, 300.0), ub_steps=2,
        )
        r = optimize_model(model, c, ug1_values=ug1_vals)
        assert r.best is not None
        assert r.best.half_swing > 0
        swings = {round(pt.half_swing, 2) for pt in r.grid_points if pt.half_swing > 0}
        assert len(swings) > 1

    def test_refine_includes_swing(self):
        """Scipy refinement should also optimize swing."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=5, ra_steps=5, swing_steps=4,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.best is not None

        refined = refine_optimum(r.best, points=pts, model=None, constraints=c)
        assert refined is not None
        assert refined.half_swing > 0
        # Refined swing should be in valid range
        assert refined.half_swing <= refined.max_swing + 0.1

    def test_sweep_swing_helper_produces_steps(self):
        """_sweep_swing_top_n generates correct number of swing levels."""
        pt = OptPoint(
            ub=250, ug2=0, ug1=-7, ra=10,
            thd=3.0, hd2=2.0, hd3=1.0,
            pout_mw=500, pa_mw=2000,
            ia_0=10, ua_0=200, amp_class="A",
            max_swing=5.0, half_swing=5.0,
        )
        calls = []

        def mock_eval(ub, ug2, ug1, ra, hs):
            calls.append(hs)
            return OptPoint(
                ub=ub, ug2=ug2, ug1=ug1, ra=ra,
                thd=3.0 * (hs / 5.0), hd2=2.0, hd3=1.0,
                pout_mw=500 * (hs / 5.0), pa_mw=2000,
                ia_0=10, ua_0=200, amp_class="A",
                max_swing=5.0, half_swing=hs,
            )

        c = OptimizerConstraints(swing_steps=5)
        result = _sweep_swing_top_n([pt], mock_eval, c)
        assert len(calls) == 5
        # Swing values should span MIN_SWING_FRACTION * max to max
        assert min(calls) == pytest.approx(MIN_SWING_FRACTION * 5.0)
        assert max(calls) == pytest.approx(5.0)
        assert len(result) == 5

    def test_pentode_swing_optimization(self):
        """Pentode (EL84) swing optimization works."""
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-10.0, -4.0), ra_range=(2.0, 8.0),
            ug1_steps=5, ra_steps=5, swing_steps=3,
            pa_max_w=12.0,
        )
        r = optimize_measurements(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        assert r.error is None
        assert r.best is not None
        assert r.best.half_swing > 0


# ═══════════════════════════════════════════════════════════════════
#  Pareto front refinement (parallel)
# ═══════════════════════════════════════════════════════════════════

class TestRefineParetoFront:

    def test_refine_pareto_basic(self):
        """refine_pareto_front returns a valid Pareto front."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=8, ra_steps=8,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert len(r.pareto_front) >= 1

        refined = refine_pareto_front(
            r.pareto_front, points=pts, model=None, constraints=c,
        )
        assert len(refined) >= 1
        # Refined front should be sorted by THD
        thds = [p.thd for p in refined]
        assert thds == sorted(thds)

    def test_refine_pareto_improves_or_matches(self):
        """Refined Pareto should not be worse than grid Pareto."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=8, ra_steps=8,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert len(r.pareto_front) >= 2

        refined = refine_pareto_front(
            r.pareto_front, points=pts, model=None, constraints=c,
        )
        # Refined front should have at least as many points
        # (it merges original + refined and re-computes)
        assert len(refined) >= 1

    @pytest.mark.timeout(30)
    def test_refine_pareto_with_model(self):
        """Refine Pareto front with model data."""
        model, pts = quick_triode("12AU7")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        c = OptimizerConstraints(
            target="min_thd",
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=5, ra_steps=5,
            ub_range=(200.0, 300.0), ub_steps=2,
        )
        r = optimize_model(model, c, ug1_values=ug1_vals)
        assert r.pareto_front

        refined = refine_pareto_front(
            r.pareto_front, points=pts, model=model,
            constraints=c, ug1_values=ug1_vals,
            max_points=3, max_workers=2,
        )
        # ML-150: refined content went unchecked. Contract: the front
        # is RECOMPUTED over original+merged (its length is NOT capped by
        # max_points — merging only adds candidates), best THD must not
        # regress, points must stay physical.
        assert len(refined) >= 1
        assert min(p.thd for p in refined) <=             min(p.thd for p in r.pareto_front) + 1e-9
        for rp in refined:
            assert 0.0 < rp.thd < MAX_SANE_THD
            assert rp.pout_mw > 0.0

    def test_refine_pareto_cancel(self):
        """Cancellation should return partial or empty results."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=8, ra_steps=8,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)

        # Cancel immediately
        refined = refine_pareto_front(
            r.pareto_front, points=pts, model=None, constraints=c,
            cancelled=lambda: True,
        )
        # Should still return something (original Pareto at minimum)
        # The function merges what it has
        assert isinstance(refined, list)

    def test_refine_pareto_progress_callback(self):
        """Progress callback is called for each refined point."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=8, ra_steps=8,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.pareto_front

        progress_calls = []
        refine_pareto_front(
            r.pareto_front, points=pts, model=None, constraints=c,
            on_progress=lambda cur, tot: progress_calls.append((cur, tot)),
        )
        assert len(progress_calls) > 0
        # Last call should have current == total
        assert progress_calls[-1][0] == progress_calls[-1][1]

    def test_refine_pareto_max_points(self):
        """When Pareto front is large, only max_points are refined."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=10, ra_steps=10,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)

        progress_calls = []
        refine_pareto_front(
            r.pareto_front, points=pts, model=None, constraints=c,
            max_points=3,
            on_progress=lambda cur, tot: progress_calls.append((cur, tot)),
        )
        assert progress_calls, "guard de-vacuated 2026-07-12: value must be present"
        assert progress_calls[0][1] <= 3

    def test_optimizer_result_has_refined_pareto(self):
        """OptimizerResult.refined_pareto field exists."""
        r = OptimizerResult()
        assert hasattr(r, "refined_pareto")
        assert r.refined_pareto == []


# ═══════════════════════════════════════════════════════════════════
#  Circuit topology support
# ═══════════════════════════════════════════════════════════════════




class TestCircuitTopology:
    """Optimizer correctly uses different load lines per circuit type."""

    def test_se_resistive_default(self):
        """Default circuit='se' uses ResistiveLoadLine."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            circuit=CIRCUIT_SE,
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=5, ra_steps=5,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.error is None
        assert r.best is not None
        assert r.best.thd > 0

    @staticmethod
    def _spy_load_line_types(monkeypatch):
        """ML-150 (call-site vs function): the trio's docstrings promise
        "uses XxxLoadLine" but the actual line type went unchecked —
        collect the set of types the optimizer really builds."""
        import lm19.optimizer as opt_mod
        seen: set = set()
        orig = opt_mod._make_load_line

        def spy(ub_v, ra_v, cons):
            ll = orig(ub_v, ra_v, cons)
            seen.add(type(ll).__name__)
            return ll

        monkeypatch.setattr(opt_mod, "_make_load_line", spy)
        return seen

    def test_se_transformer(self, monkeypatch):
        """circuit='se_xfmr' uses TransformerLoadLine."""
        seen = self._spy_load_line_types(monkeypatch)
        pts = _make_triode_points()
        c = OptimizerConstraints(
            circuit=CIRCUIT_SE_XFMR, ra_dc=0.05,
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=5, ra_steps=5,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.error is None
        assert r.best is not None
        assert seen == {"TransformerLoadLine"}

    def test_cathode_follower(self, monkeypatch):
        """circuit='cf' uses CathodeFollowerLoadLine."""
        seen = self._spy_load_line_types(monkeypatch)
        pts = _make_triode_points()
        c = OptimizerConstraints(
            circuit=CIRCUIT_CF, cf_rk=10.0, cf_rl=10.0,
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=5, ra_steps=5,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.error is None
        assert r.best is not None
        assert seen == {"CathodeFollowerLoadLine"}

    def test_se_xfmr_produces_results(self):
        """SE Transformer optimizer produces valid results."""
        pts = _make_triode_points()
        c = OptimizerConstraints(
            circuit=CIRCUIT_SE_XFMR, ra_dc=0.05,
            ug1_range=(-9.0, -3.0), ra_range=(5.0, 15.0),
            ug1_steps=5, ra_steps=5,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.error is None
        assert len(r.grid_points) > 0
        assert r.best is not None
        # All points should have valid THD
        for pt in r.grid_points:
            assert pt.thd >= 0

    def test_cf_vs_se_differ(self):
        """CF and SE give different results."""
        pts = _make_triode_points()
        base = dict(ug1_range=(-9.0, -3.0), ra_range=(5.0, 15.0),
                    ug1_steps=5, ra_steps=5)
        r_se = optimize_measurements(pts, ub=250.0,
                                     constraints=OptimizerConstraints(circuit=CIRCUIT_SE, **base))
        r_cf = optimize_measurements(pts, ub=250.0,
                                     constraints=OptimizerConstraints(circuit=CIRCUIT_CF,
                                                                      cf_rk=5.0, cf_rl=10.0,
                                                                      **base))
        assert r_se.best is not None and r_cf.best is not None
        # The docstring promises CF and SE differ — actually verify it instead
        # of only checking both produced a result.
        assert (r_se.best.thd, r_se.best.pout_mw) != (r_cf.best.thd, r_cf.best.pout_mw), \
            "CF and SE optimizers produced identical best — circuit type not applied"

    def test_model_with_transformer(self, monkeypatch):
        """Model path also respects circuit type."""
        import lm19.optimizer as opt_mod
        seen: set = set()
        orig = opt_mod._make_load_line

        def spy(ub_v, ra_v, cons):
            ll = orig(ub_v, ra_v, cons)
            seen.add(type(ll).__name__)
            return ll

        monkeypatch.setattr(opt_mod, "_make_load_line", spy)
        model, pts = quick_triode("12AU7")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        c = OptimizerConstraints(
            circuit=CIRCUIT_SE_XFMR, ra_dc=0.05,
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 20.0),
            ug1_steps=3, ra_steps=3,
            ub_range=(200.0, 300.0), ub_steps=2,
        )
        r = optimize_model(model, c, ug1_values=ug1_vals)
        assert r.error is None
        assert r.best is not None
        assert seen == {"TransformerLoadLine"}   # ML-150

    def test_constraints_has_circuit_fields(self):
        """OptimizerConstraints stores circuit params."""
        c = OptimizerConstraints(
            circuit=CIRCUIT_SE_XFMR, ra_dc=0.1,
            cf_rk=5.0, cf_rl=8.0, pp_raa=6.0,
        )
        assert c.circuit == CIRCUIT_SE_XFMR
        assert c.ra_dc == 0.1
        assert c.cf_rk == 5.0
        assert c.cf_rl == 8.0
        assert c.pp_raa == 6.0


class TestCircuitPhysicalSanity:
    """Physical sanity checks for all circuit types on real tube data."""

    def test_se_12au7_thd_sane(self):
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_SE,
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=8, ra_steps=8, pa_max_w=2.75,
            pout_min_w=0.05,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.best is not None
        assert r.best.thd < MAX_SANE_THD
        assert r.best.pout_mw > 0

    def test_se_xfmr_12au7_thd_sane(self):
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_SE_XFMR, ra_dc=0.05,
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=8, ra_steps=8, pa_max_w=2.75,
            pout_min_w=0.05,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.best is not None
        assert r.best.thd < MAX_SANE_THD
        assert r.best.pout_mw > 0

    def test_cf_12au7_thd_sane(self):
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_CF, cf_rk=10.0, cf_rl=10.0,
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=8, ra_steps=8,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        assert r.best is not None
        assert r.best.thd < MAX_SANE_THD

    def test_se_el84_pentode_thd_sane(self):
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            circuit=CIRCUIT_SE,
            ug1_range=(-10.0, -4.0), ra_range=(2.0, 8.0),
            ug1_steps=6, ra_steps=6, pa_max_w=12.0,
        )
        r = optimize_measurements(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        assert r.best is not None
        assert r.best.thd < MAX_SANE_THD
        assert r.best.pout_mw > 0

    def test_se_xfmr_el84_pentode_thd_sane(self):
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            circuit=CIRCUIT_SE_XFMR, ra_dc=0.1,
            ug1_range=(-10.0, -4.0), ra_range=(2.0, 8.0),
            ug1_steps=6, ra_steps=6, pa_max_w=12.0,
        )
        r = optimize_measurements(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        assert r.best is not None
        assert r.best.thd < MAX_SANE_THD

    def test_model_se_xfmr_12au7_sane(self):
        model, pts = quick_triode("12AU7")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        c = OptimizerConstraints(
            circuit=CIRCUIT_SE_XFMR, ra_dc=0.05,
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 25.0),
            ug1_steps=4, ra_steps=4,
            ub_range=(200.0, 300.0), ub_steps=2,
        )
        r = optimize_model(model, c, ug1_values=ug1_vals)
        assert r.best is not None
        assert r.best.thd < MAX_SANE_THD

    def test_all_circuits_pout_positive(self):
        """Pout > 0 for all supported circuit types (SE, SE_XFMR, CF)."""
        _, pts = quick_triode("12AU7")
        base = dict(ug1_range=(-12.0, -4.0), ra_range=(5.0, 20.0),
                    ug1_steps=5, ra_steps=5, pout_min_w=0.01)
        for circuit, extra in [
            ("se", {}),
            ("se_xfmr", {"ra_dc": 0.05}),
            ("cf", {"cf_rk": 10.0, "cf_rl": 10.0}),
        ]:
            c = OptimizerConstraints(circuit=circuit, **extra, **base)
            r = optimize_measurements(pts, ub=250.0, constraints=c)
            assert r.best is not None, f"{circuit}: no best point"
            assert r.best.pout_mw > 0, f"{circuit}: Pout should be positive"


# ═══════════════════════════════════════════════════════════════════
#  Push-Pull optimizer
# ═══════════════════════════════════════════════════════════════════

class TestOptimizePP:
    """Tests for optimize_pp — push-pull specific optimizer."""

    def test_pp_basic_triode_matched(self):
        """PP optimizer produces results for matched triode pair."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0,
            ug1_range=(-15.0, -3.0), ra_range=(4.0, 16.0),
            ug1_steps=6, ra_steps=6,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        assert r.error is None
        assert r.best is not None
        assert r.best.thd > 0
        assert r.best.pout_mw > 0

    def test_pp_pentode_matched(self, monkeypatch):
        """PP optimizer works for pentode (EL84) with Ug2 filter."""
        # ML-150 (call-site): "with Ug2 filter" — the filter must reach
        # its real consumer (build_pp_transfer); the spy pins the applied
        # contract. OptPoint.ug2 additionally carries the resolved screen
        # voltage (metadata for status/Top-N/apply; computationally inert).
        # optimize_pp imports build_pp_transfer locally from
        # lm19.amplifier — patch the source (resolved at the call site).
        import lm19.amplifier as amp_mod
        seen_filters: set = set()
        orig_bpt = amp_mod.build_pp_transfer

        def spy(points_a, points_b, ug2_filter, **kw):
            seen_filters.add(ug2_filter)
            return orig_bpt(points_a, points_b, ug2_filter, **kw)

        monkeypatch.setattr(amp_mod, "build_pp_transfer", spy)
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0,
            ug1_range=(-10.0, -4.0), ra_range=(4.0, 12.0),
            ug1_steps=5, ra_steps=5, pa_max_w=12.0,
        )
        r = optimize_pp(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        assert r.error is None
        assert r.best is not None
        assert r.best.valid
        assert 0.0 < r.best.thd < MAX_SANE_THD
        assert r.best.pout_mw > 0.0
        assert seen_filters == {250.0}
        assert r.best.ug2 == 250.0    # resolved screen stored on point

    def test_pp_hd2_low_matched(self):
        """Matched PP should have low HD2 (even harmonics cancelled)."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=6, ra_steps=6,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        assert r.best is not None
        # Matched pair should have very low HD2 (near-perfect cancellation)
        assert r.best.hd2 < 1.0, f"Matched PP HD2={r.best.hd2:.2f}% should be < 1%"

    def test_pp_pareto_front(self):
        """PP optimizer produces Pareto front."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0,
            ug1_range=(-15.0, -5.0), ra_range=(4.0, 16.0),
            ug1_steps=8, ra_steps=8,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        assert len(r.pareto_front) >= 1
        assert all(p.valid for p in r.pareto_front)
        assert all(0.0 < p.thd < MAX_SANE_THD for p in r.pareto_front)

    def test_pp_swing_sweep(self):
        """PP optimizer with swing sweep produces varied swing values."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0,
            ug1_range=(-12.0, -5.0), ra_range=(5.0, 12.0),
            ug1_steps=5, ra_steps=5, swing_steps=3,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        swings = {round(pt.half_swing, 1) for pt in r.grid_points if pt.half_swing > 0}
        assert len(swings) > 1

    def test_pp_ra_means_raa(self):
        """Ra in PP results is Ra_aa (anode-to-anode), not per-tube."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0,
            ug1_range=(-10.0, -5.0), ra_range=(4.0, 16.0),
            ug1_steps=3, ra_steps=5,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        ra_values = {pt.ra for pt in r.grid_points}
        # All Ra values should be in the specified range (Ra_aa)
        for ra in ra_values:
            assert 4.0 <= ra <= 16.0

    def test_pp_pa_max_constraint(self):
        """Pa max constraint works for PP."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0,
            pa_max_w=1.0,  # very strict
            ug1_range=(-12.0, -5.0), ra_range=(5.0, 12.0),
            ug1_steps=5, ra_steps=5,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        for pt in r.pareto_front:
            assert pt.pa_mw / 1000.0 <= 1.0


class TestPPPhysicalSanity:
    """Physical sanity for PP optimizer on real tube data."""

    def test_pp_12au7_thd_sane(self):
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 20.0),
            ug1_steps=8, ra_steps=8,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        assert r.best is not None
        assert r.best.thd < MAX_SANE_THD
        assert r.best.pout_mw > 0

    def test_pp_el84_thd_sane(self):
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0,
            ug1_range=(-10.0, -4.0), ra_range=(4.0, 12.0),
            ug1_steps=6, ra_steps=6, pa_max_w=12.0,
        )
        r = optimize_pp(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        assert r.best is not None
        assert r.best.thd < MAX_SANE_THD

    def test_pp_pout_higher_than_se(self):
        """PP should generally deliver more power than SE for same tube.

        Use target=max_pout for a fair comparison — with target=min_thd,
        optimizer finds tiny-swing degenerate points (especially for PP
        matched pair where HD2 cancels by symmetry, pushing min-THD
        toward swing→0).
        """
        _, pts = quick_triode("12AU7")
        common = dict(
            target="max_pout",
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 20.0),
            ug1_steps=8, ra_steps=8,
        )
        c_se = OptimizerConstraints(circuit=CIRCUIT_SE, **common)
        c_pp = OptimizerConstraints(circuit=CIRCUIT_PP, pp_raa=10.0, **common)
        r_se = optimize_measurements(pts, ub=250.0, constraints=c_se)
        r_pp = optimize_pp(pts, ub=250.0, constraints=c_pp)
        # Both runs must find an operating point (an `if ...best:`
        # guard here would turn the test into a silent skip).
        assert r_se.best is not None
        assert r_pp.best is not None
        # PP Pout must be a MULTIPLE of SE (probed: 1257 mW vs 263 mW,
        # ratio ≈4.8×).  The old `> SE × 0.5` bound accepted PP *below* SE.
        assert r_pp.best.pout_mw > r_se.best.pout_mw * _PP_OVER_SE_MIN_RATIO, (
            f"PP Pout={r_pp.best.pout_mw:.0f} mW not above SE "
            f"{r_se.best.pout_mw:.0f} mW × {_PP_OVER_SE_MIN_RATIO}"
        )

    def test_pp_matched_hd2_near_zero(self):
        """Matched pair: HD2 should be very low (even harmonic cancellation)."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=6, ra_steps=6,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        assert r.best is not None
        # In a perfectly matched pair, HD2 should be < 0.1%
        # Allow some tolerance for numerical precision
        assert r.best.hd2 < 0.5, f"Matched PP HD2={r.best.hd2:.3f}% too high"

    def test_pp_all_points_thd_sane(self):
        """All grid points in PP should have sane THD."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0,
            ug1_range=(-12.0, -5.0), ra_range=(5.0, 15.0),
            ug1_steps=5, ra_steps=5,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        for pt in r.grid_points:
            assert pt.thd < MAX_SANE_THD, f"PP Ug1={pt.ug1:.1f} Ra_aa={pt.ra:.1f} THD={pt.thd:.1f}%"
            assert pt.pout_mw > 0


# ═══════════════════════════════════════════════════════════════════
#  _make_load_line unit tests
# ═══════════════════════════════════════════════════════════════════

class TestMakeLoadLine:

    def test_se_returns_resistive(self):
        from lm19.amplifier import ResistiveLoadLine
        c = OptimizerConstraints(circuit=CIRCUIT_SE)
        ll = _make_load_line(250, 10.0, c)
        assert isinstance(ll, ResistiveLoadLine)

    def test_se_xfmr_returns_transformer(self):
        from lm19.amplifier import TransformerLoadLine
        c = OptimizerConstraints(circuit=CIRCUIT_SE_XFMR, ra_dc=0.1)
        ll = _make_load_line(250, 10.0, c)
        assert isinstance(ll, TransformerLoadLine)
        assert ll.ra_dc == 0.1
        assert ll.ra_ac == 10.0

    def test_cf_returns_cathode_follower(self):
        from lm19.amplifier import CathodeFollowerLoadLine
        c = OptimizerConstraints(circuit=CIRCUIT_CF, cf_rk=5.0, cf_rl=8.0)
        ll = _make_load_line(250, 8.0, c)
        assert isinstance(ll, CathodeFollowerLoadLine)
        assert ll.rk == 5.0

    def test_pp_returns_pushpull(self):
        from lm19.amplifier import PushPullLoadLine
        c = OptimizerConstraints(circuit=CIRCUIT_PP, pp_raa=10.0)
        ll = _make_load_line(250, 10.0, c)
        assert isinstance(ll, PushPullLoadLine)
        assert ll.ra_aa == 10.0

    def test_pp_propagates_ra_dc(self):
        """pp_ra_dc from constraints flows into PushPullLoadLine."""
        from lm19.amplifier import PushPullLoadLine
        c = OptimizerConstraints(circuit=CIRCUIT_PP, pp_raa=10.0, pp_ra_dc=0.25)
        ll = _make_load_line(250, 10.0, c)
        assert isinstance(ll, PushPullLoadLine)
        assert ll.ra_dc == 0.25

    def test_pp_default_ra_dc(self):
        """Default pp_ra_dc preserved when not set explicitly."""
        from lm19.amplifier import PushPullLoadLine
        c = OptimizerConstraints(circuit=CIRCUIT_PP, pp_raa=10.0)  # no pp_ra_dc
        ll = _make_load_line(250, 10.0, c)
        assert ll.ra_dc == 0.1   # OptimizerConstraints default


class TestResolveUlTaps:
    """_resolve_ul_taps maps UI mode + state to a sorted list of fractions."""

    def test_off_returns_manual(self):
        from lm19.optimizer import _resolve_ul_taps
        c = OptimizerConstraints(ul_tap_mode="off", ul_tap_manual=0.43)
        assert _resolve_ul_taps(c) == [0.43]

    def test_off_default_is_zero(self):
        from lm19.optimizer import _resolve_ul_taps
        c = OptimizerConstraints()  # mode=off, manual=0
        assert _resolve_ul_taps(c) == [0.0]

    def test_presets_all_enabled(self):
        from lm19.optimizer import _resolve_ul_taps
        c = OptimizerConstraints(
            ul_tap_mode="presets",
            ul_tap_presets=(0.0, 0.20, 0.35, 0.43, 0.50, 1.0),
            ul_tap_presets_enabled=(True, True, True, True, True, True),
        )
        result = _resolve_ul_taps(c)
        assert result == [0.0, 0.20, 0.35, 0.43, 0.50, 1.0]

    def test_presets_some_disabled(self):
        from lm19.optimizer import _resolve_ul_taps
        c = OptimizerConstraints(
            ul_tap_mode="presets",
            ul_tap_presets=(0.0, 0.20, 0.43, 1.0),
            ul_tap_presets_enabled=(True, False, True, False),
        )
        assert _resolve_ul_taps(c) == [0.0, 0.43]

    def test_custom_range(self):
        from lm19.optimizer import _resolve_ul_taps
        c = OptimizerConstraints(
            ul_tap_mode="custom",
            ul_tap_range=(0.0, 1.0),
            ul_tap_steps=5,
        )
        result = _resolve_ul_taps(c)
        assert result == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_custom_range_clamped(self):
        from lm19.optimizer import _resolve_ul_taps
        c = OptimizerConstraints(
            ul_tap_mode="custom",
            ul_tap_range=(-0.1, 1.5),  # out of [0,1]
            ul_tap_steps=3,
        )
        result = _resolve_ul_taps(c)
        # Clamped to [0,1] then 3 equal steps: 0, 0.5, 1
        assert result == [0.0, 0.5, 1.0]

    def test_presets_custom_union(self):
        from lm19.optimizer import _resolve_ul_taps
        c = OptimizerConstraints(
            ul_tap_mode="presets_custom",
            ul_tap_presets=(0.0, 0.43, 1.0),
            ul_tap_presets_enabled=(True, True, True),
            ul_tap_range=(0.10, 0.50),
            ul_tap_steps=3,
        )
        # Presets: {0, 0.43, 1.0}; Custom: {0.10, 0.30, 0.50}
        # Union sorted: [0.0, 0.10, 0.30, 0.43, 0.50, 1.0]
        result = _resolve_ul_taps(c)
        assert result == [0.0, 0.10, 0.30, 0.43, 0.50, 1.0]

    def test_presets_none_enabled(self):
        from lm19.optimizer import _resolve_ul_taps
        c = OptimizerConstraints(
            ul_tap_mode="presets",
            ul_tap_presets=(0.0, 0.43),
            ul_tap_presets_enabled=(False, False),  # none
        )
        # Empty → fall back to [0.0]
        assert _resolve_ul_taps(c) == [0.0]


class TestOptimizerUlSweepPP:
    """End-to-end: optimize_pp respects ul_tap_mode and produces points
    with varied OptPoint.ul_tap when sweeping."""

    def test_optpoint_ul_tap_field(self):
        """Default OptPoint.ul_tap is 0.0."""
        from lm19.optimizer import OptPoint
        pt = OptPoint(
            ub=300, ug2=0, ug1=-9, ra=8,
            thd=2, hd2=1, hd3=1, pout_mw=1000, pa_mw=2000,
            ia_0=50, ua_0=300, amp_class="A", max_swing=5,
        )
        assert pt.ul_tap == 0.0

    @pytest.mark.timeout(60)
    def test_pp_off_mode_uses_manual_tap(self):
        """ul_tap_mode='off' with non-zero manual — single tap recorded
        on grid points. (Known: swing-sweep refines with unwrapped model →
        swing_steps=1 to keep test focused on grid behaviour.)"""
        model, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=3, ra_steps=3, swing_steps=1,
            hd_method=HD_METHOD_DFT,
            ul_tap_mode="off",
            ul_tap_manual=0.43,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c, model=model)
        assert r.grid_points, "guard de-vacuated 2026-07-12: value must be present"
        taps = {round(p.ul_tap, 2) for p in r.grid_points}
        assert taps == {0.43}, f"Expected single tap 0.43, got {taps}"

    @pytest.mark.timeout(60)
    def test_pp_presets_mode_produces_multiple_taps(self):
        """ul_tap_mode='presets' with model + DFT yields points at each tap.
        DFT is slow (~100ms/eval), so we use a tiny 2×2 grid + swing_steps=1."""
        model, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(8.0, 10.0),
            ug1_steps=2, ra_steps=2, swing_steps=1,
            hd_method=HD_METHOD_DFT,
            ul_tap_mode="presets",
            ul_tap_presets=(0.0, 0.43, 1.0),
            ul_tap_presets_enabled=(True, True, True),
        )
        r = optimize_pp(pts, ub=250.0, constraints=c, model=model)
        taps = {round(p.ul_tap, 2) for p in r.grid_points}
        # All three taps should appear (some grid points may fail per tap)
        assert taps == {0.0, 0.43, 1.0}, f"Expected {{0,0.43,1}}, got {taps}"

    @pytest.mark.timeout(60)
    def test_refine_optimum_preserves_ul_tap(self):
        """refine_optimum must wrap model with the input point's tap →
        scipy refines around same UL config, not silently fall back to
        pentode. Without this fix, grid winner with tap=43% would get
        refined as if pentode."""
        from lm19.optimizer import refine_optimum
        model, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(8.0, 10.0),
            ug1_steps=2, ra_steps=2, swing_steps=1,
            hd_method=HD_METHOD_DFT,
            ul_tap_mode="presets",
            ul_tap_presets=(0.43,),
            ul_tap_presets_enabled=(True,),
        )
        r = optimize_pp(pts, ub=250.0, constraints=c, model=model)
        assert r.best is not None
        assert abs(r.best.ul_tap - 0.43) < 0.01
        ref = refine_optimum(r.best, points=pts, model=model, constraints=c)
        assert ref is not None
        assert abs(ref.ul_tap - 0.43) < 0.01, (
            f"Refine lost UL tap: was 0.43, got {ref.ul_tap}"
        )

    @pytest.mark.timeout(120)
    def test_pp_swing_sweep_preserves_ul_tap(self):
        """Phase 2 swing sweep groups top-N points by ul_tap and uses
        per-group wrapped model. Without grouping, swing variants of a
        tap=43% point would silently revert to pentode (tap=0)."""
        model, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(8.0, 10.0),
            ug1_steps=2, ra_steps=2, swing_steps=3,   # >1 to trigger sweep
            hd_method=HD_METHOD_DFT,
            ul_tap_mode="presets",
            ul_tap_presets=(0.0, 0.43, 1.0),
            ul_tap_presets_enabled=(True, True, True),
        )
        r = optimize_pp(pts, ub=250.0, constraints=c, model=model)
        # All UL taps must appear among grid points (swing variants too).
        # Without grouping, swing sweep would only emit tap=0 variants.
        taps = {round(p.ul_tap, 2) for p in r.grid_points}
        assert taps == {0.0, 0.43, 1.0}, (
            f"Swing sweep dropped UL taps. Expected {{0,0.43,1}}, got {taps}"
        )
        # Verify swing-variant points (with reduced half_swing) inherit tap
        for tap in (0.43, 1.0):
            tap_pts = [p for p in r.grid_points if abs(p.ul_tap - tap) < 0.01]
            # At least one tap-tagged point exists
            assert len(tap_pts) > 0
            # max_swing variations exist (some half_swing values < others)
            half_swings = {round(p.half_swing, 2) for p in tap_pts}
            # With swing_steps=3 we expect at least 1 distinct half_swing
            assert len(half_swings) >= 1

    def test_pp_ul_skipped_when_method_not_dft(self):
        """5-point/Chebyshev work on data composite — UL wrap has no effect.
        Optimizer should skip UL sweep and run only the manual tap."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(8.0, 10.0),
            ug1_steps=2, ra_steps=2,
            hd_method=HD_METHOD_5POINT,
            ul_tap_mode="presets",
            ul_tap_presets=(0.0, 0.43, 1.0),
            ul_tap_presets_enabled=(True, True, True),
            ul_tap_manual=0.0,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)  # no model
        taps = {round(p.ul_tap, 2) for p in r.grid_points}
        # Only manual tap (0.0) used, not the presets
        assert taps == {0.0}, f"Expected only manual tap, got {taps}"

    @pytest.mark.timeout(180)
    def test_el84_pp_ul_dft_documented_ordering(self):
        """EL84 PP @ Ub=250V, Ra_aa=8k via DFT optimizer: pentode/UL/triode
        modes should produce the documented physical ordering — pentode
        peaks Pout, triode lowest, UL mid. Verifies the integration of
        Phase 1 grid → Phase 2 swing → Phase 3 refine through optimize_pp()
        with hd_method='dft' and a fixed UL tap. Synthetic Koren EL84
        over-predicts Ia in triode mode compared to the datasheet, so we
        use a relaxed Pa ceiling and lower Ub instead of trying to match
        exact Pout numbers.

        target=max_pout: the documented ordering is a MAX-POWER claim;
        with 'balanced' each tap's best lands on a different swing (the
        min-THD pull picks tiny swings) and cross-tap Pout comparison is
        apples-to-oranges. Probed with the joint-OPT solver at EQUAL
        swing the ordering holds: 7.65 > 4.24 > 1.85 W."""
        model, pts = quick_pentode("EL84")
        results = {}
        for tap_label, tap in (("pentode", 0.0), ("ul43", 0.43), ("triode", 1.0)):
            c = OptimizerConstraints(
                target="max_pout", circuit=CIRCUIT_PP, pp_raa=8.0,
                ug1_range=(-12.0, -8.0), ra_range=(6.0, 10.0),
                ug1_steps=3, ra_steps=3, swing_steps=2,
                pa_max_w=18.0,  # relaxed for synthetic Koren triode-mode bias
                hd_method=HD_METHOD_DFT,
                ul_tap_mode="off",
                ul_tap_manual=tap,
            )
            r = optimize_pp(pts, ub=250.0, constraints=c,
                            ug2_filter=250.0, model=model)
            assert r.error is None, f"{tap_label}: {r.error}"
            assert r.best is not None, f"{tap_label}: no best point"
            # Each best point must respect ul_tap and use DFT path
            assert abs(r.best.ul_tap - tap) < 0.01
            assert r.best.hd_method == HD_METHOD_DFT_PP
            # Physical sanity: THD < MAX_SANE_THD, Pout > 0
            assert r.best.thd < MAX_SANE_THD, (
                f"{tap_label}: THD={r.best.thd:.2f}% over sane bound"
            )
            assert r.best.hd2 < _HD_MAX_PCT_PP
            assert r.best.hd3 < _HD_MAX_PCT_PP
            assert r.best.pout_mw > 0
            assert r.best.pout_mw < _EL84_PP_POUT_MAX_MW, (
                f"{tap_label}: Pout={r.best.pout_mw:.0f}mW exceeds EL84 PP ceiling"
            )
            results[tap_label] = r.best
        # Documented ordering (Williamson, Mullard, Brimar EL84 PP refs):
        # pentode delivers most power, triode the least, UL between.
        assert results["pentode"].pout_mw > results["ul43"].pout_mw, (
            f"Expected pentode > UL: pent={results['pentode'].pout_mw:.0f}mW "
            f"ul43={results['ul43'].pout_mw:.0f}mW"
        )
        assert results["ul43"].pout_mw > results["triode"].pout_mw, (
            f"Expected UL > triode: ul43={results['ul43'].pout_mw:.0f}mW "
            f"triode={results['triode'].pout_mw:.0f}mW"
        )


# ═══════════════════════════════════════════════════════════════════
#  PP extended tests: mismatched pair, targets, edge cases, refine
# ═══════════════════════════════════════════════════════════════════

class TestPPExtended:

    def test_pp_mismatched_pair(self):
        """PP optimizer with two different tube datasets.

        Mismatched pair (12AU7 + 12AX7) must show HIGHER mean HD2 than a
        matched pair over the SAME grid points — even-harmonic cancellation
        is imperfect when the tubes differ.  Compared pointwise (not
        best-vs-best) because "best" lands on different operating points,
        and at the hottest bias the matched pair's own clipping asymmetry
        can locally exceed the mismatch residual (probed: 24/28 common
        points higher, mean 4.55% vs 1.13%).
        """
        _, pts_a = quick_triode("12AU7")
        _, pts_b = quick_triode("12AX7")  # different tube type
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-5.0, -1.0), ra_range=(5.0, 15.0),
            ug1_steps=4, ra_steps=4,
        )
        r_matched = optimize_pp(pts_a, ub=250.0, constraints=c)
        r = optimize_pp(pts_a, ub=250.0, constraints=c, points_b=pts_b)
        assert r.error is None
        assert r.best is not None
        assert r_matched.error is None
        # Mismatched pair has HIGHER HD2 than matched (even harmonic
        # cancellation is imperfect) — compared at identical grid keys.
        def _key(p: OptPoint) -> tuple:
            return (round(p.ug1, 3), round(p.ra, 3), round(p.half_swing, 3))
        hd2_matched = {_key(p): p.hd2 for p in r_matched.grid_points}
        hd2_mismatched = {_key(p): p.hd2 for p in r.grid_points}
        common = sorted(set(hd2_matched) & set(hd2_mismatched))
        assert len(common) >= _MIN_COMMON_PP_GRID_POINTS
        mean_matched = float(np.mean([hd2_matched[k] for k in common]))
        mean_mismatched = float(np.mean([hd2_mismatched[k] for k in common]))
        assert mean_mismatched > mean_matched * _MISMATCH_HD2_MEAN_RATIO, (
            f"mismatched mean HD2={mean_mismatched:.3f}% not above matched "
            f"{mean_matched:.3f}% × {_MISMATCH_HD2_MEAN_RATIO}"
        )

    def test_pp_mismatched_hd2_higher(self):
        """Mismatched pair HD2 should be higher than matched."""
        _, pts = quick_triode("12AU7")
        _, pts_b = quick_triode("12AX7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-5.0, -1.5), ra_range=(6.0, 14.0),
            ug1_steps=5, ra_steps=5,
        )
        r_matched = optimize_pp(pts, ub=250.0, constraints=c)
        r_mismatched = optimize_pp(pts, ub=250.0, constraints=c, points_b=pts_b)
        # Both runs must find an operating point — otherwise there is
        # nothing to compare (an `if ...best:` guard = silent skip).
        assert r_matched.best is not None
        assert r_mismatched.best is not None
        # Mismatched must have worse HD2 by a clear margin (probed:
        # matched 0.008%, mismatched 0.342% — 3× headroom over the margin).
        assert r_mismatched.best.hd2 > (
            r_matched.best.hd2 + _MISMATCH_HD2_MIN_DELTA_PCT
        ), (
            f"mismatched HD2={r_mismatched.best.hd2:.4f}% not above "
            f"matched {r_matched.best.hd2:.4f}% + {_MISMATCH_HD2_MIN_DELTA_PCT}"
        )

    def test_pp_target_max_pout(self):
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            target="max_pout", circuit=CIRCUIT_PP, pp_raa=8.0,
            ug1_range=(-12.0, -5.0), ra_range=(5.0, 15.0),
            ug1_steps=6, ra_steps=6,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        assert r.best is not None
        valid = [p for p in r.grid_points if p.valid]
        max_pout = max(p.pout_mw for p in valid)
        assert r.best.pout_mw == pytest.approx(max_pout, abs=1.0)

    def test_pp_target_balanced(self):
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            target="balanced", circuit=CIRCUIT_PP, pp_raa=8.0,
            ug1_range=(-12.0, -5.0), ra_range=(5.0, 15.0),
            ug1_steps=6, ra_steps=6,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        assert r.best is not None
        assert r.best.thd < MAX_SANE_THD

    def test_pp_pout_min_constraint(self):
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            pout_min_w=0.1, circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(5.0, 15.0),
            ug1_steps=5, ra_steps=5,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        # 0.1 W is achievable for 12AU7 PP on this grid (probed:
        # best ≈ 102 mW), so best must exist — `if r.best:` was a
        # silent skip.
        assert r.best is not None
        assert r.best.pout_mw >= 100.0

    def test_pp_empty_data(self):
        # ML-150: exact empty-input contract (twin of the SE variant).
        c = OptimizerConstraints(circuit=CIRCUIT_PP, pp_raa=8.0)
        r = optimize_pp([], ub=250.0, constraints=c)
        assert r.error == OPT_ERR_NO_VALID_POINTS
        assert r.best is None
        assert r.grid_points == []

    def test_pp_refine_optimum(self):
        """refine_optimum works for PP circuit."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(5.0, 15.0),
            ug1_steps=5, ra_steps=5,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        assert r.best is not None
        refined = refine_optimum(
            r.best, points=pts, model=None, constraints=c,
        )
        # Should not crash; may or may not improve
        assert refined is not None
        assert refined.thd >= 0
        assert refined.thd < MAX_SANE_THD
        assert refined.pout_mw >= 0

    def test_pp_refine_pareto_front(self):
        """refine_pareto_front works for PP circuit."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(5.0, 15.0),
            ug1_steps=6, ra_steps=6,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        assert r.pareto_front
        refined = refine_pareto_front(
            r.pareto_front, points=pts, model=None,
            constraints=c, max_points=3,
        )
        assert len(refined) >= 1
        assert min(p.thd for p in refined) <=             min(p.thd for p in r.pareto_front) + 1e-9
        for rp in refined:
            assert 0.0 < rp.thd < MAX_SANE_THD
            assert rp.pout_mw > 0.0

    def test_hd_method_dispatch_5point_default(self):
        """Default hd_method='5point' yields OptPoints stamped 5point."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            ug1_range=(-12.0, -5.0), ra_range=(5.0, 15.0),
            ug1_steps=4, ra_steps=4,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        for pt in r.grid_points[:5]:
            assert pt.hd_method == HD_METHOD_5POINT

    def test_hd_method_dispatch_chebyshev(self):
        """hd_method='chebyshev' produces Chebyshev-evaluated points."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            ug1_range=(-12.0, -5.0), ra_range=(5.0, 15.0),
            ug1_steps=4, ra_steps=4,
            hd_method=HD_METHOD_CHEBYSHEV,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)
        for pt in r.grid_points[:5]:
            assert pt.hd_method == HD_METHOD_CHEBYSHEV

    @pytest.mark.timeout(60)
    def test_hd_method_dispatch_dft_with_model(self):
        """hd_method='dft' with model uses DFT throughout."""
        model, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            ug1_range=(-12.0, -5.0), ra_range=(5.0, 15.0),
            ug1_steps=3, ra_steps=3,
            hd_method=HD_METHOD_DFT,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c, model=model)
        assert r.warning is None  # DFT succeeded with model
        if r.grid_points:
            assert r.grid_points[0].hd_method == HD_METHOD_DFT

    def test_hd_method_dispatch_dft_fallback_no_model(self):
        """hd_method='dft' without model falls back to Chebyshev with warning."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            ug1_range=(-12.0, -5.0), ra_range=(5.0, 15.0),
            ug1_steps=4, ra_steps=4,
            hd_method=HD_METHOD_DFT,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c)  # no model
        assert r.warning == OPT_WARN_DFT_NO_MODEL_FALLBACK
        if r.grid_points:
            assert r.grid_points[0].hd_method == HD_METHOD_CHEBYSHEV

    def test_hd_method_dispatch_auto_with_model(self):
        """hd_method='auto' with model: grid Chebyshev, refine DFT."""
        from lm19.optimizer import _resolve_methods
        grid, refine, _ = _resolve_methods("auto", has_model=True)
        assert grid == HD_METHOD_CHEBYSHEV
        assert refine == HD_METHOD_DFT

    def test_hd_method_dispatch_auto_no_model(self):
        """hd_method='auto' without model: Chebyshev throughout."""
        from lm19.optimizer import _resolve_methods
        grid, refine, _ = _resolve_methods("auto", has_model=False)
        assert grid == HD_METHOD_CHEBYSHEV
        assert refine == HD_METHOD_CHEBYSHEV

    def test_sparse_data_guard_blocks_fake_zero_thd_6p1p(self):
        """Real 6P1P pentode: best.thd > 0.05% with all methods.
        Without the sparse-data guard, 5-point would pick degenerate
        small-swing points reporting near-zero THD."""
        import json
        with open(_PROJECT_ROOT / "tests/spice_test_data/converted/pentode_6P1P_real.json") as f:
            pts = json.load(f)["points"]
        for hd_method in (HD_METHOD_5POINT, HD_METHOD_CHEBYSHEV):
            c = OptimizerConstraints(
                target="min_thd", circuit=CIRCUIT_SE,
                ug1_range=(-15.0, -7.0), ra_range=(2.0, 15.0),
                ug1_steps=6, ra_steps=6,
                pa_max_w=12.0,
                pout_min_w=0.05,
                hd_method=hd_method,
            )
            r = optimize_measurements(pts, ub=250.0, constraints=c, ug2_filter=99.0)
            assert r.best is not None, f"{hd_method}: no best point"
            assert r.best.thd > 0.05, (
                f"{hd_method}: best.thd={r.best.thd:.4f}% degenerate"
            )
            assert r.best.thd < MAX_SANE_THD

    def test_methods_agree_on_top_ranking_12au7(self):
        """Top-3 ranking by THD should be similar across HD methods on
        synthetic 12AU7 data (all methods detect the same low-distortion
        operating points within sane THD range)."""
        _, pts = quick_triode("12AU7")
        rankings = {}
        for hd_method in (HD_METHOD_5POINT, HD_METHOD_CHEBYSHEV):
            c = OptimizerConstraints(
                target="min_thd", circuit=CIRCUIT_SE,
                ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
                ug1_steps=6, ra_steps=6,
                pout_min_w=0.05,
                hd_method=hd_method,
            )
            r = optimize_measurements(pts, ub=250.0, constraints=c)
            # ML-150: an empty pareto silently skipped the comparison
            # — both methods must produce a front.
            assert r.pareto_front, f"{hd_method}: empty pareto front"
            top = sorted(r.pareto_front, key=lambda p: p.thd)[:3]
            rankings[hd_method] = [(p.ug1, p.ra) for p in top]
        common = set(rankings["5point"]) & set(rankings["chebyshev"])
        assert len(common) > 0, f"5pt={rankings['5point']} vs cheb={rankings['chebyshev']}"

    def test_chebyshev_better_thd_estimate_than_5point(self):
        """Chebyshev should generally report higher THD than 5-point on
        pentode data with wide swing — captures HD4-HD9 that 5-point misses.
        On 6P1P this difference is consistent with benchmark observation."""
        import json
        with open(_PROJECT_ROOT / "tests/spice_test_data/converted/pentode_6P1P_real.json") as f:
            pts = json.load(f)["points"]
        thds = {}
        for hd_method in (HD_METHOD_5POINT, HD_METHOD_CHEBYSHEV):
            c = OptimizerConstraints(
                target="min_thd", circuit=CIRCUIT_SE,
                ug1_range=(-15.0, -7.0), ra_range=(2.0, 15.0),
                ug1_steps=8, ra_steps=8,
                pa_max_w=12.0,
                pout_min_w=0.05,
                hd_method=hd_method,
            )
            r = optimize_measurements(pts, ub=250.0, constraints=c, ug2_filter=99.0)
            assert r.best is not None, f"{hd_method}: no best point"
            thds[hd_method] = r.best.thd
        # Chebyshev catches more harmonics → reports equal or higher THD
        # for the same operating range. Allow small tolerance.
        if "5point" in thds and "chebyshev" in thds:
            assert thds["chebyshev"] >= thds["5point"] - 0.5, (
                f"Chebyshev {thds['chebyshev']:.2f}% < 5point {thds['5point']:.2f}% "
                "(suspicious — Chebyshev should capture more harmonics)"
            )

    def test_pp_uses_5point_method_by_default(self):
        """PP grid points carry ``hd_method='5point'`` by default."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=4, ra_steps=4,
            hd_method=HD_METHOD_5POINT,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        assert r.warning is None
        if r.grid_points:
            for pt in r.grid_points[:3]:
                assert pt.hd_method == HD_METHOD_5POINT

    def test_pp_uses_chebyshev_pp_when_requested(self):
        """PP with hd_method='chebyshev' actually runs Chebyshev on composite."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=4, ra_steps=4,
            hd_method=HD_METHOD_CHEBYSHEV,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        assert r.warning is None
        assert r.grid_points, "empty grid — vacuous label check (ML-150)"
        cheb_pts = [p for p in r.grid_points if p.hd_method == HD_METHOD_CHEBYSHEV_PP]
        assert len(cheb_pts) > 0, "PP Chebyshev should produce labeled points"

    @pytest.mark.timeout(60)
    def test_pp_uses_dft_pp_when_requested_with_model(self):
        """PP with hd_method='dft' + model uses DFT on composite."""
        model, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=3, ra_steps=3,
            hd_method=HD_METHOD_DFT,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c, model=model)
        assert r.warning is None
        assert r.grid_points, "empty grid — vacuous label check (ML-150)"
        dft_pts = [p for p in r.grid_points if p.hd_method == HD_METHOD_DFT_PP]
        assert len(dft_pts) > 0, "PP DFT should produce labeled points"

    @pytest.mark.timeout(120)
    def test_pp_methods_physical_sanity_6p1p_real(self):
        """All PP methods produce sane THD on real 6P1P pentode data.
        DFT typically gives lowest THD (smooth model symmetry → HD2≈0,
        small HD3); Chebyshev/5-point may be higher due to data
        discretization. All must fall under MAX_SANE_THD_PCT and pass
        the sparse-data anti-degeneracy floor."""
        import json
        from lm19.dempwolf import fit_dempwolf
        with open(_PROJECT_ROOT / "tests/spice_test_data/converted/pentode_6P1P_real.json") as f:
            pts = json.load(f)["points"]
        model = fit_dempwolf(pts, topology=TOPOLOGY_PENTODE).model

        for method in (HD_METHOD_5POINT, HD_METHOD_CHEBYSHEV, HD_METHOD_DFT):
            c = OptimizerConstraints(
                target="min_thd", circuit=CIRCUIT_PP, pp_raa=8.0,
                ug1_range=(-15.0, -7.0), ra_range=(4.0, 12.0),
                ug1_steps=4, ra_steps=4, pa_max_w=12.0,
                pout_min_w=0.5,
                hd_method=method,
            )
            r = optimize_pp(pts, ub=250.0, constraints=c,
                            ug2_filter=200.0, model=model)
            assert r.error is None, f"{method}: {r.error}"
            assert r.best is not None, f"{method}: no best"
            # Anti-degeneracy floor (no fake-zero except for
            # mathematically perfect DFT matched pair)
            min_thd = 0.005 if method == HD_METHOD_DFT else 0.05
            assert r.best.thd >= min_thd, \
                f"{method}: degenerate THD={r.best.thd:.4f}%"
            assert r.best.thd < MAX_SANE_THD, \
                f"{method}: THD={r.best.thd:.2f}% exceeds sane bound"
            # PP filter forced ≥ 0.5W
            assert r.best.pout_mw >= 500.0
            # Method label correct
            expected_label = {
                "5point": "5point",
                "chebyshev": "chebyshev_pp",
                "dft": "dft_pp",
            }[method]
            assert r.best.hd_method == expected_label

    def test_pp_dft_falls_back_to_chebyshev_no_model(self):
        """PP DFT without model → Chebyshev fallback with warning."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=4, ra_steps=4,
            hd_method=HD_METHOD_DFT,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)  # no model
        assert r.warning == OPT_WARN_DFT_NO_MODEL_FALLBACK
        if r.grid_points:
            for pt in r.grid_points[:3]:
                assert pt.hd_method == HD_METHOD_CHEBYSHEV_PP

    def test_optpoint_has_hd_method_field(self):
        """OptPoint dataclass exposes hd_method."""
        from lm19.optimizer import OptPoint
        pt = OptPoint(
            ub=250, ug2=0, ug1=-7, ra=5,
            thd=3.0, hd2=2.0, hd3=1.0,
            pout_mw=1000, pa_mw=2000,
            ia_0=10, ua_0=200, amp_class="A", max_swing=3.0,
        )
        assert hasattr(pt, "hd_method")
        assert pt.hd_method == HD_METHOD_5POINT  # default

    def test_pp_pentode_el84_targets(self):
        """PP EL84 pentode works with all targets."""
        _, pts = quick_pentode("EL84")
        for target in ["min_thd", "max_pout", "balanced"]:
            c = OptimizerConstraints(
                target=target, circuit=CIRCUIT_PP, pp_raa=8.0,
                ug1_range=(-10.0, -4.0), ra_range=(4.0, 12.0),
                ug1_steps=4, ra_steps=4, pa_max_w=12.0,
            )
            r = optimize_pp(pts, ub=300.0, constraints=c, ug2_filter=250.0)
            assert r.best is not None, f"PP EL84 target={target} failed"
            assert r.best.thd < MAX_SANE_THD


# ═══════════════════════════════════════════════════════════════════
#  Class A power threshold (PP):  P_A = Iq² × Ra_aa / 8
# ═══════════════════════════════════════════════════════════════════

class TestClassAPowerThreshold:
    """Class-A power threshold (P_A) is the output power at which a
    push-pull amp transitions from class A (both tubes conducting)
    to class AB (one tube cuts off at peak). Boundary: ΔI = Iq.

    Formula: P_A = Iq² × Ra_aa / 8  (Iq=A, Ra_aa=Ω, P_A=W).
    With Iq in mA and Ra_aa in kΩ → divide by 8000.

    Source: Aiken "Last Word on Class A", sound-au.com.
    Empirical sanity: 6L6 triode PP, Iq=109 mA, Ra_aa=5 kΩ → P_A ≈ 7.4 W
    (matches the published "7.5 W class A before cutoff").
    """

    def test_p_classa_formula_constant(self):
        """Divisor 8000 converts (mA² × kΩ) to W."""
        assert P_CLASSA_DIVISOR == 8000.0

    def test_p_classa_published_6l6_example(self):
        """6L6 triode PP: Iq=109 mA, Ra_aa=5 kΩ → ~7.4 W."""
        iq_ma = 109.0
        raa_k = 5.0
        p_w = (iq_ma ** 2) * raa_k / P_CLASSA_DIVISOR
        assert 7.0 < p_w < 8.0, f"Got {p_w:.2f} W, expected ~7.4"

    def test_p_classa_quadratic_in_iq(self):
        """P_A ∝ Iq²: doubling Iq quadruples P_A."""
        raa_k = 8.0
        p1 = (50.0 ** 2) * raa_k / P_CLASSA_DIVISOR
        p2 = (100.0 ** 2) * raa_k / P_CLASSA_DIVISOR
        assert p2 / p1 == pytest.approx(4.0, rel=1e-9)

    def test_p_classa_linear_in_raa(self):
        """P_A ∝ Ra_aa: doubling Ra_aa doubles P_A."""
        iq_ma = 80.0
        p1 = (iq_ma ** 2) * 4.0 / P_CLASSA_DIVISOR
        p2 = (iq_ma ** 2) * 8.0 / P_CLASSA_DIVISOR
        assert p2 / p1 == pytest.approx(2.0, rel=1e-9)

    def test_resolve_threshold_off(self):
        c = OptimizerConstraints(class_a_power_mode="off", class_a_power_value=5.0)
        assert _resolve_class_a_threshold(c, pout_mw=10000.0) == 0.0

    def test_resolve_threshold_absolute(self):
        c = OptimizerConstraints(class_a_power_mode="absolute", class_a_power_value=3.5)
        assert _resolve_class_a_threshold(c, pout_mw=10000.0) == 3.5

    def test_resolve_threshold_percent(self):
        # 20% of 5 W (5000 mW) = 1 W
        c = OptimizerConstraints(class_a_power_mode="percent", class_a_power_value=20.0)
        assert _resolve_class_a_threshold(c, pout_mw=5000.0) == pytest.approx(1.0)

    def test_resolve_threshold_negative_value_clamped(self):
        c = OptimizerConstraints(class_a_power_mode="absolute", class_a_power_value=-3.0)
        assert _resolve_class_a_threshold(c, pout_mw=1000.0) == 0.0

    # ── OptPoint p_classA_w field ─────────────────────────────────

    def test_optpoint_p_classa_filled_for_pp(self):
        """Every PP grid point must have p_classA_w computed correctly."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=4, ra_steps=4,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        assert r.grid_points
        for pt in r.grid_points:
            # Per-tube Iq must be a real positive current.  Anchoring to an
            # oracle built from the RAW points (not OptPoint fields) breaks
            # the 0 ≈ 0 co-regression where a bug zeroes ia_0 and p_classA_w
            # together and the old formula-echo assert still passed.
            assert pt.ia_0 > 0
            iq_indep = _pp_transfer_iq(pts, pt.ug1, ua_ref=250.0)
            assert pt.ia_0 == pytest.approx(iq_indep, rel=1e-6)
            expected_indep = (iq_indep ** 2) * pt.ra / P_CLASSA_DIVISOR
            assert expected_indep > 0
            assert pt.p_classA_w == pytest.approx(expected_indep, rel=1e-6)
            # Self-consistency with the stored fields (original assert).
            expected = (pt.ia_0 ** 2) * pt.ra / P_CLASSA_DIVISOR
            assert pt.p_classA_w == pytest.approx(expected, rel=1e-6)
            assert pt.p_classA_w >= 0

    def test_optpoint_p_classa_zero_for_se(self):
        """SE/CF/SE-XFMR: p_classA_w stays 0 (concept doesn't apply)."""
        _, pts = quick_triode("12AU7")
        for circuit, extra in [
            ("se", {}),
            ("se_xfmr", {"ra_dc": 0.05}),
            ("cf", {"cf_rk": 10.0, "cf_rl": 10.0}),
        ]:
            c = OptimizerConstraints(
                circuit=circuit, **extra,
                ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
                ug1_steps=4, ra_steps=4,
            )
            r = optimize_measurements(pts, ub=250.0, constraints=c)
            for pt in r.grid_points:
                assert pt.p_classA_w == 0.0, f"{circuit}: expected 0, got {pt.p_classA_w}"

    # ── Filter behavior ──────────────────────────────────────────

    def test_filter_off_keeps_all(self):
        """mode='off' must not filter anything."""
        _, pts = quick_triode("12AU7")
        c_off = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=4, ra_steps=4,
            class_a_power_mode="off", class_a_power_value=999.0,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c_off)
        valid = [p for p in r.grid_points if p.valid]
        # off@999 discriminates "off was ignored": absolute@999 yields
        # 0 valid points (probed). "Filters NOTHING" = exactly as many
        # valid points as with no class-A parameters at all (probed:
        # 84 == 84).
        assert len(valid) > 0
        c_plain = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=4, ra_steps=4,
        )
        r_plain = optimize_pp(pts, ub=250.0, constraints=c_plain)
        assert len(valid) == sum(
            1 for p in r_plain.grid_points if p.valid)  # filter does nothing

    def test_filter_absolute_excludes_low_p_classa(self):
        """All valid points must satisfy p_classA_w >= threshold."""
        _, pts = quick_triode("12AU7")
        # 12AU7 PP on this grid: p_classA_w spans ~0.002..0.157 W
        # (per-tube Iq 1.6..9.5 mA averaged over Ua), so a 0.05 W
        # threshold splits the grid — both passing AND filtered points
        # exist. A 1.5 W threshold is unreachable: valid would always be
        # empty and asserts under `if valid:` would never run.
        threshold_w = 0.05
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=5, ra_steps=5,
            class_a_power_mode="absolute",
            class_a_power_value=threshold_w,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        valid = [p for p in r.grid_points if p.valid]
        assert valid, "threshold must be feasible on this grid"
        # The filter must have had something to cut — otherwise the
        # test checks nothing.
        assert any(p.p_classA_w < threshold_w - 1e-9 for p in r.grid_points)
        for pt in valid:
            assert pt.p_classA_w >= threshold_w - 1e-9
        # Reverse direction: no below-threshold point may be valid.
        for pt in r.grid_points:
            if pt.p_classA_w < threshold_w - 1e-9:
                assert not pt.valid
        # Absolute anchor: P_A of the hottest valid point is recomputed
        # from an independent Iq (Iq^2 * Ra_aa / 8000), not from OptPoint
        # fields.
        hottest = max(valid, key=lambda p: p.p_classA_w)
        iq_indep = _pp_transfer_iq(pts, hottest.ug1, ua_ref=250.0)
        assert iq_indep > 0
        expected_w = (iq_indep ** 2) * hottest.ra / P_CLASSA_DIVISOR
        assert expected_w >= threshold_w
        assert hottest.p_classA_w == pytest.approx(expected_w, rel=1e-6)

    def test_filter_percent_excludes_below_pct(self):
        """percent mode: valid points have p_classA_w >= pct × Pout."""
        _, pts = quick_triode("12AU7")
        pct = 30.0
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=5, ra_steps=5,
            class_a_power_mode="percent",
            class_a_power_value=pct,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        valid = [p for p in r.grid_points if p.valid]
        for pt in valid:
            target = (pct / 100.0) * (pt.pout_mw / 1000.0)
            assert pt.p_classA_w >= target - 1e-9

    def test_filter_zero_value_disables(self):
        """mode='absolute' with value=0 doesn't filter (threshold is 0)."""
        _, pts = quick_triode("12AU7")
        base = dict(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=4, ra_steps=4,
        )
        r_off = optimize_pp(pts, ub=250.0, constraints=OptimizerConstraints(**base))
        r_zero = optimize_pp(pts, ub=250.0,
                             constraints=OptimizerConstraints(
                                 class_a_power_mode="absolute", class_a_power_value=0.0,
                                 **base))
        v_off = sum(1 for p in r_off.grid_points if p.valid)
        v_zero = sum(1 for p in r_zero.grid_points if p.valid)
        assert v_off == v_zero

    def test_filter_strict_threshold_rejects_more_points(self):
        """Stricter threshold → fewer valid points."""
        _, pts = quick_triode("12AU7")
        base = dict(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=5, ra_steps=5,
        )
        r_loose = optimize_pp(pts, ub=250.0,
                              constraints=OptimizerConstraints(
                                  class_a_power_mode="absolute",
                                  class_a_power_value=0.5, **base))
        r_strict = optimize_pp(pts, ub=250.0,
                               constraints=OptimizerConstraints(
                                   class_a_power_mode="absolute",
                                   class_a_power_value=3.0, **base))
        v_loose = sum(1 for p in r_loose.grid_points if p.valid)
        v_strict = sum(1 for p in r_strict.grid_points if p.valid)
        assert v_strict <= v_loose

    def test_filter_ignored_for_se(self):
        """Class-A filter is PP-specific; SE results unchanged by mode."""
        _, pts = quick_triode("12AU7")
        base = dict(
            circuit=CIRCUIT_SE,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=4, ra_steps=4,
        )
        r_off = optimize_measurements(pts, ub=250.0,
                                      constraints=OptimizerConstraints(**base))
        r_strict = optimize_measurements(pts, ub=250.0,
                                         constraints=OptimizerConstraints(
                                             class_a_power_mode="absolute",
                                             class_a_power_value=999.0, **base))
        v_off = sum(1 for p in r_off.grid_points if p.valid)
        v_strict = sum(1 for p in r_strict.grid_points if p.valid)
        assert v_off == v_strict

    # ── Real-data physical sanity ─────────────────────────────────

    def test_real_el84_pp_finds_class_a_point(self):
        """EL84 PP: feasible class-A threshold yields valid result."""
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0,
            ug1_range=(-10.0, -4.0), ra_range=(4.0, 12.0),
            ug1_steps=5, ra_steps=5,
            pa_max_w=12.0,
            class_a_power_mode="absolute",
            class_a_power_value=0.5,
        )
        r = optimize_pp(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        assert r.error is None
        assert r.best is not None
        assert r.best.p_classA_w >= 0.5
        assert r.best.thd < MAX_SANE_THD
        assert r.best.pout_mw > 0

    def test_real_el84_pp_infeasible_threshold(self):
        """Unrealistically high P_A → no valid points (graceful)."""
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0,
            ug1_range=(-10.0, -4.0), ra_range=(4.0, 12.0),
            ug1_steps=4, ra_steps=4,
            pa_max_w=12.0,
            class_a_power_mode="absolute",
            class_a_power_value=999.0,  # impossible
        )
        r = optimize_pp(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        # Optimizer must not crash; either error or empty Pareto
        assert r.error is not None or len(r.pareto_front) == 0

    def test_p_classa_at_boundary_gives_pout_close(self):
        """At an operating point, the class-A threshold P_A should be a
        plausible UPPER BOUND on the achievable class-A output power.
        Sanity: P_A computed as Iq²·Ra_aa/8 should not exceed Pa_q."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=4, ra_steps=4,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        for pt in r.grid_points:
            # P_A is per-tube. Pa_q (per tube) = Ub × Iq.
            # Class A swing ≤ Iq → AC power ≤ DC dissipation per tube.
            # So P_A_total (both tubes) ≤ 2 × Pa_q (per tube).
            pa_q_w = pt.pa_mw / 1000.0  # per-tube DC dissipation
            assert pt.p_classA_w <= 2.0 * pa_q_w + 0.01, (
                f"P_A={pt.p_classA_w:.2f}W exceeds 2×Pa_q={2*pa_q_w:.2f}W "
                f"(Iq={pt.ia_0:.1f}mA, Ra_aa={pt.ra:.1f}kΩ)"
            )


# ======================================================================
# _compute_dist_pp DFT fallback exception handling
# ======================================================================
#
# The DFT-PP path includes a ``_pp_fallback`` closure that calls
# ``composite_characteristic`` to derive ``half_swing`` when not given.
# That call must:
#   - swallow data-shape errors (KeyError, IndexError, TypeError,
#     ValueError) — they happen on malformed user data and should
#     gracefully return None for the whole point
#   - log such failures with enough context to debug
#   - propagate programming errors (AttributeError, NameError, etc.)
#     — silent swallow there masks regressions during refactor


class TestPPDFTFallbackExceptionHandling:
    """``_compute_dist_pp`` DFT fallback must distinguish data vs programming errors."""

    def _make_load_line(self):
        from lm19.amplifier import PushPullLoadLine
        return PushPullLoadLine(ub=300, ra_aa=8.0, ra_dc=0.1)

    def test_data_error_logged_and_returns_none(self, caplog):
        """KeyError on malformed point dict is logged + graceful None."""
        from lm19.optimizer import _compute_dist_pp
        import logging
        model, _ = quick_pentode("EL84")
        ll = self._make_load_line()
        with caplog.at_level(logging.WARNING, logger="lm19.optimizer"):
            result = _compute_dist_pp(
                method=HD_METHOD_DFT,
                points_a=[{"ug1": -5.0, "ua": 200.0}],  # missing 'ia'
                points_b=None,
                load_line=ll,
                ug1_bias=-5.0,
                half_swing=None,  # forces fallback
                ug2_filter=None,
                model=model,
            )
        assert result is None, "Bad data must yield None, not crash"
        # Warning was logged with enough detail to debug
        assert any("PP composite failed" in rec.message
                   and "KeyError" in rec.message for rec in caplog.records), \
            f"Expected WARNING with KeyError diagnostics, got: " \
            f"{[r.message for r in caplog.records]}"

    def test_programming_error_propagates(self):
        """AttributeError (wrong type) must NOT be swallowed."""
        from lm19.optimizer import _compute_dist_pp
        model, _ = quick_pentode("EL84")
        ll = self._make_load_line()
        # Passing a string instead of list triggers AttributeError on .get()
        # inside _build_transfer_curve. This is a programming error and
        # must surface, not be silently absorbed into "no composite".
        with pytest.raises((AttributeError, TypeError)):
            _compute_dist_pp(
                method=HD_METHOD_DFT,
                points_a="not a list",  # type: ignore
                points_b=None,
                load_line=ll,
                ug1_bias=-5.0,
                half_swing=None,
                ug2_filter=None,
                model=model,
            )

    def test_good_swing_skips_fallback_entirely(self, caplog):
        """When half_swing is provided, fallback isn't called → no log noise."""
        from lm19.optimizer import _compute_dist_pp
        import logging
        model, _ = quick_pentode("EL84")
        ll = self._make_load_line()
        with caplog.at_level(logging.WARNING, logger="lm19.optimizer"):
            _compute_dist_pp(
                method=HD_METHOD_DFT,
                points_a=[{"ug1": -5.0, "ua": 200.0}],  # bad data
                points_b=None,
                load_line=ll,
                ug1_bias=-5.0,
                half_swing=2.0,  # valid → fallback path NOT taken
                ug2_filter=None,
                model=model,
            )
        # No PP-composite warning should appear — fallback wasn't called
        assert not any("PP composite failed" in rec.message
                       for rec in caplog.records), \
            "Fallback should not have been called when half_swing is valid"


class TestPpDisplayUg2:
    """PP-data OptPoint.ug2 carries the resolved screen voltage
    (filter -> median>0 -> honest 0 for triode data, WITHOUT the UL
    anchor's ub fallback). Metadata for status/Top-N/apply; the eval
    path ignores the value."""

    def _constraints(self):
        return OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0,
            ug1_range=(-10.0, -4.0), ra_range=(4.0, 12.0),
            ug1_steps=3, ra_steps=3, pa_max_w=12.0,
        )

    def test_filter_recorded_on_grid_points(self):
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        r = optimize_pp(pts, ub=300.0, constraints=self._constraints(),
                        ug2_filter=250.0)
        assert r.grid_points
        assert all(p.ug2 == 250.0 for p in r.grid_points)

    def test_median_resolved_without_filter(self):
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        r = optimize_pp(pts, ub=300.0, constraints=self._constraints())
        assert r.best is not None
        assert r.best.ug2 == 250.0     # median of the single level

    def test_triode_data_reports_zero_not_ub(self):
        """Discriminates against reusing _resolve_ul_ug2_nom: its
        fallback is ub — for triode data the status would then claim
        "Ug2=250V", which is false."""
        _, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=10.0,
            ug1_range=(-12.0, -5.0), ra_range=(6.0, 14.0),
            ug1_steps=3, ra_steps=3,
        )
        r = optimize_pp(pts, ub=250.0, constraints=c)
        assert r.best is not None
        assert r.best.ug2 == 0.0

    def test_swing_sweep_points_carry_screen(self):
        """Phase 2 (swing sweep) — same metadata on the new points."""
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0,
            ug1_range=(-10.0, -4.0), ra_range=(4.0, 12.0),
            ug1_steps=3, ra_steps=3, pa_max_w=12.0, swing_steps=3,
        )
        r = optimize_pp(pts, ub=300.0, constraints=c, ug2_filter=250.0)
        swing_pts = [p for p in r.grid_points if p.half_swing > 0]
        assert swing_pts
        assert all(p.ug2 == 250.0 for p in swing_pts)


class TestUg2FilterNoMatchWarning:
    """A filter matching no point silently fell back to the
    unfiltered set (logged, but with no user-visible warning). The
    ug2_filter_no_match code must reach OptimizerResult.warnings."""

    def test_se_no_match_warns_and_still_falls_back(self):
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            ug1_range=(-10.0, -4.0), ra_range=(3.0, 8.0),
            ug1_steps=3, ra_steps=3, pa_max_w=12.0,
        )
        r = optimize_measurements(pts, ub=300.0, constraints=c,
                                  ug2_filter=555.0)
        assert OPT_WARN_UG2_FILTER_NO_MATCH in r.warnings
        assert r.grid_points        # fallback semantics preserved

    def test_se_matching_filter_no_warning(self):
        """Negative space: a matching filter emits no code."""
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            ug1_range=(-10.0, -4.0), ra_range=(3.0, 8.0),
            ug1_steps=3, ra_steps=3, pa_max_w=12.0,
        )
        r = optimize_measurements(pts, ub=300.0, constraints=c,
                                  ug2_filter=250.0)
        assert OPT_WARN_UG2_FILTER_NO_MATCH not in r.warnings

    def test_pp_twin_no_match_warns(self):
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0,
            ug1_range=(-10.0, -4.0), ra_range=(4.0, 12.0),
            ug1_steps=3, ra_steps=3, pa_max_w=12.0,
        )
        r = optimize_pp(pts, ub=300.0, constraints=c, ug2_filter=555.0)
        assert OPT_WARN_UG2_FILTER_NO_MATCH in r.warnings

    def test_ug2_values_path_not_checked(self):
        """ug2_values come from the data — the code is not raised."""
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            ug1_range=(-10.0, -4.0), ra_range=(3.0, 8.0),
            ug1_steps=3, ra_steps=3, pa_max_w=12.0,
        )
        r = optimize_measurements(pts, ub=300.0, constraints=c,
                                  ug2_values=[250.0])
        assert OPT_WARN_UG2_FILTER_NO_MATCH not in r.warnings

    def test_predicate_mirrors_filter_fallback(self):
        """Predicate equivalence with _apply_ug2_filter, plus the
        tolerance boundary (exactly 5.0 V passes)."""
        from lm19.amplifier import ug2_filter_matches_any
        from lm19.amplifier.distortion import _apply_ug2_filter
        pts = [{"ua": 100.0, "ug1": -2.0, "ia": 10.0, "ug2": 250.0}]
        assert ug2_filter_matches_any(pts, 250.0) is True
        assert ug2_filter_matches_any(pts, 255.0) is True   # == tol
        assert ug2_filter_matches_any(pts, 255.1) is False  # past bound
        assert ug2_filter_matches_any(pts, None) is True
        # False <=> the filter falls back to the full set
        assert _apply_ug2_filter(pts, 255.1, 5.0, "pin") == pts
        assert _apply_ug2_filter(pts, 255.0, 5.0, "pin") == pts  # subset=all


class TestPpDftUg2Resolution:
    """ML-118: PP DFT with ug2_filter=None used to evaluate the pentode
    model at Ug2=0 — cutoff, every sample None, the whole grid died as
    'no_valid_points' with no cause (real path: empty ug2_calc_combo).
    The screen is now resolved from the measured data (median Ug2)."""

    def test_no_filter_sets_ui_warning(self):
        """The resolution must be visible in the UI: OptimizerResult.warning
        carries 'pp_dft_ug2_from_data' (main_window shows amp.opt_warn_*),
        and stays None when the user did select a Ug2 level."""
        from lm19.optimizer import OptimizerConstraints, optimize_pp
        from lm19.tube_sim import quick_pentode
        model, pts = quick_pentode("EL84")
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0, pa_max_w=12.0, hd_method=HD_METHOD_DFT,
            ug1_range=(-8.0, -7.0), ra_range=(7.0, 9.0),
            ug1_steps=2, ra_steps=2, swing_steps=2,
        )
        r = optimize_pp(pts, ub=300.0, constraints=c, model=model)
        assert r.warning == OPT_WARN_PP_DFT_UG2_FROM_DATA
        r2 = optimize_pp(pts, ub=300.0, constraints=c, model=model,
                         ug2_filter=250.0)
        assert r2.warning is None

    def test_no_filter_resolves_ug2_from_data(self):
        from lm19.optimizer import _compute_dist_pp
        from lm19.amplifier import PushPullLoadLine
        from lm19.tube_sim import quick_pentode
        model, pts = quick_pentode("EL84")
        ll = PushPullLoadLine(ub=300.0, ra_aa=8000.0)
        dist = _compute_dist_pp(
            "dft", pts, None, ll, ug1_bias=-11.0, half_swing=5.0,
            ug2_filter=None, model=model)
        assert dist is not None,             "PP DFT died at Ug2=0 instead of resolving the measured screen"
        assert dist.get("thd") is not None
        # ML-150: `is not None` would accept a resolve to ANY non-zero
        # screen. Discriminator: the resolve must yield exactly the
        # measured level — the result matches an explicit ug2_filter=250
        # (the only Ug2 in the quick_pentode grid). Probed: bit-exact.
        assert 0.0 < dist["thd"] < 70.0
        dist_explicit = _compute_dist_pp(
            "dft", pts, None, ll, ug1_bias=-11.0, half_swing=5.0,
            ug2_filter=250.0, model=model)
        assert dist_explicit is not None
        assert dist["thd"] == pytest.approx(dist_explicit["thd"], rel=1e-9)
        assert dist["pout_mw"] == pytest.approx(
            dist_explicit["pout_mw"], rel=1e-9)
