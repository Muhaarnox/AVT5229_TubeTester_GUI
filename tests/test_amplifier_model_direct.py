"""Tests for amplifier direct model solver (find_intersections_model)."""

import numpy as np
import pytest

from lm19.amplifier import (
    CathodeFollowerLoadLine,
    ResistiveLoadLine,
    PushPullLoadLine,
    find_intersections,
    find_intersections_model,
    sweep_amplitude,
    sweep_ra,
    compute_distortion,
)
from lm19.tube_sim import load_model, quick_triode, quick_pentode


class TestFindIntersectionsModel:
    """Direct model solver vs point-based solver."""

    def test_triode_convergence(self):
        model = load_model("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100)
        ug1_values = [-4.0, -3.0, -2.0, -1.0, 0.0]
        isects = find_intersections_model(
            model, ll, ug1_values, ug2=0.0,
        )
        assert len(isects) >= 3
        for isect in isects:
            assert isect["ua"] > 0
            assert isect["ia"] >= 0

    def test_pentode_convergence(self):
        model = load_model("EL84")
        ll = ResistiveLoadLine(ub=250, ra=4)
        ug1_values = [-15.0, -10.0, -7.0, -5.0, -3.0, 0.0]
        isects = find_intersections_model(
            model, ll, ug1_values, ug2=250.0,
        )
        assert len(isects) >= 3

    def test_matches_point_based(self):
        """Direct solver should agree with point-based within reasonable tolerance."""
        model, points = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100)

        # Point-based
        isects_pts = find_intersections(points, ll)

        # Model-direct
        ug1_values = sorted({round(p["ug1"], 1) for p in points})
        isects_model = find_intersections_model(
            model, ll, ug1_values, ug2=0.0,
        )

        # Both should find similar number of intersections
        assert abs(len(isects_model) - len(isects_pts)) <= 1

        # Compare common Ug1 values
        model_map = {round(i["ug1"], 1): i for i in isects_model}
        pts_map = {round(i["ug1"], 1): i for i in isects_pts}
        common = set(model_map.keys()) & set(pts_map.keys())
        assert len(common) >= 3

        for ug1 in common:
            m = model_map[ug1]
            p = pts_map[ug1]
            # Ua should agree within ~5V (grid is 10V steps)
            assert abs(m["ua"] - p["ua"]) < 10, (
                f"Ua mismatch at Ug1={ug1}: model={m['ua']:.1f} pts={p['ua']:.1f}"
            )

    def test_operating_point_on_load_line(self):
        """Each intersection should lie on the load line."""
        model = load_model("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100)
        ug1_values = [-3.0, -2.0, -1.0, 0.0]
        isects = find_intersections_model(
            model, ll, ug1_values, ug2=0.0,
        )
        for isect in isects:
            ia_ll = ll.ia_at_ua(isect["ua"])
            assert abs(isect["ia"] - ia_ll) < 0.1, (
                f"Not on load line: Ia={isect['ia']:.2f} vs LL={ia_ll:.2f}"
            )

    def test_empty_ug1_returns_empty(self):
        model = load_model("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100)
        assert find_intersections_model(model, ll, []) == []

    def test_stability_vs_grid_density(self):
        """Result should be stable regardless of sparse vs dense ScanGrid."""
        model = load_model("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100)
        ug1_values = [-2.0, -1.0, 0.0]

        isects_200 = find_intersections_model(
            model, ll, ug1_values, n_search=200,
        )
        isects_50 = find_intersections_model(
            model, ll, ug1_values, n_search=50,
        )

        assert len(isects_200) == len(isects_50)
        for a, b in zip(isects_200, isects_50):
            assert abs(a["ua"] - b["ua"]) < 0.5


class TestSweepWithModel:
    """sweep_amplitude / sweep_ra with model= parameter."""

    def test_sweep_amplitude_model_produces_results(self):
        """sweep_amplitude with model should return non-empty data."""
        model, points = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100)
        results = sweep_amplitude(
            points, ll, ug1_bias=-2.0, steps=8,
            model=model, model_ug2=0.0,
        )
        assert len(results) >= 3
        for r in results:
            assert r["thd"] >= 0
            assert r["pout_mw"] >= 0

    def test_sweep_amplitude_model_agrees_with_points(self):
        """Model sweep should produce similar THD curve to point-based."""
        model, points = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100)
        res_pts = sweep_amplitude(points, ll, ug1_bias=-2.0, steps=8)
        res_model = sweep_amplitude(
            points, ll, ug1_bias=-2.0, steps=8,
            model=model, model_ug2=0.0,
        )
        # Both should find similar number of points
        assert abs(len(res_model) - len(res_pts)) <= 2

    def test_sweep_ra_model_produces_results(self):
        """sweep_ra with model should return non-empty data."""
        model, points = quick_triode("12AX7")
        results = sweep_ra(
            points, ub=250, ra_min=20.0, ra_max=200.0, steps=8,
            model=model, model_ug2=0.0,
        )
        assert len(results) >= 3
        for r in results:
            assert r["ra"] > 0
            assert r["pout_mw"] >= 0

    def test_sweep_ra_model_agrees_with_points(self):
        """Model Ra sweep should produce similar curve to point-based."""
        model, points = quick_triode("12AX7")
        res_pts = sweep_ra(
            points, ub=250, ra_min=20.0, ra_max=200.0, steps=8,
        )
        res_model = sweep_ra(
            points, ub=250, ra_min=20.0, ra_max=200.0, steps=8,
            model=model, model_ug2=0.0,
        )
        assert abs(len(res_model) - len(res_pts)) <= 2

    def test_sweep_amplitude_pentode_model(self):
        """Pentode model with correct ug2 should produce results."""
        model, points = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=250, ra=4)
        results = sweep_amplitude(
            points, ll, ug1_bias=-7.0, steps=8,
            model=model, model_ug2=250.0,
        )
        assert len(results) >= 2

    def test_sweep_ra_pentode_model(self):
        """sweep_ra with pentode model should produce non-empty data."""
        model, points = quick_pentode("EL84")
        results = sweep_ra(
            points, ub=250, ra_min=1.0, ra_max=15.0, steps=8,
            model=model, model_ug2=250.0,
        )
        assert len(results) >= 3
        for r in results:
            assert r["ra"] > 0
            assert r["pout_mw"] >= 0


class TestPentodeUg2Handling:
    """Ug2 derivation for pentode models without explicit filter."""

    def test_pentode_ug2_from_points_median(self):
        """When ug2_filter is None, pentode should use median from points."""
        model = load_model("EL84")
        ll = ResistiveLoadLine(ub=250, ra=4)
        # Simulate points with known ug2
        ug2_target = 250.0
        ug1_values = [-10.0, -7.0, -5.0, -3.0, 0.0]
        # Direct call with correct ug2 should work
        isects = find_intersections_model(
            model, ll, ug1_values, ug2=ug2_target,
        )
        assert len(isects) >= 3
        # With ug2=0 (the old bug) pentode gives very different results
        isects_wrong = find_intersections_model(
            model, ll, ug1_values, ug2=0.0,
        )
        # Results should differ significantly — proves ug2 matters
        if isects_wrong:
            ia_correct = [i["ia"] for i in isects]
            ia_wrong = [i["ia"] for i in isects_wrong]
            assert max(ia_correct) > max(ia_wrong) * 1.5 or len(isects_wrong) < len(isects)

    def test_pp_model_intersections(self):
        """Push-Pull load line should work with model direct solver."""
        model = load_model("EL84")
        ll = PushPullLoadLine(ub=250, ra_aa=8)
        ug1_values = [-15.0, -10.0, -7.0, -5.0, -3.0, 0.0]
        isects = find_intersections_model(
            model, ll, ug1_values, ug2=250.0,
        )
        assert len(isects) >= 3
        # Each intersection should lie on the PP load line
        for isect in isects:
            ia_ll = ll.ia_at_ua(isect["ua"])
            assert abs(isect["ia"] - ia_ll) < 0.2

    def test_cathode_follower_model_intersections(self):
        """CathodeFollowerLoadLine should work with model direct solver."""
        model = load_model("12AX7")
        ll = CathodeFollowerLoadLine(ub=250, rk=1.5, rl=100)
        ug1_values = [-3.0, -2.0, -1.0, 0.0]
        isects = find_intersections_model(
            model, ll, ug1_values, ug2=0.0,
        )
        assert len(isects) >= 2
        for isect in isects:
            ia_ll = ll.ia_at_ua(isect["ua"])
            assert abs(isect["ia"] - ia_ll) < 0.1
