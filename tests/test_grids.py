"""Tests for lm19.plotting.grids — pure computation, no Qt required."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.plotting.grids import (
    cluster_nominal,
    nominal_key,
    filter_ug2_slice,
    filter_ug2_multi,
    build_ia_grid,
    build_ia_grid_averaged,
    compute_gm_rp_grids,
    compute_mu_grid,
    build_pa_grid,
)


def _make_triode_points(n_ug1=3, n_ua=5, ug2=0.0):
    """Simple triode-like grid: Ia linear in Ua and Ug1."""
    pts = []
    for ug1_idx in range(n_ug1):
        ug1 = -1.0 * ug1_idx
        for ua_idx in range(n_ua):
            ua = 50.0 + ua_idx * 50.0
            ia = max(0.0, 10.0 + ug1 * 2.0 + ua * 0.05)
            pts.append({"ua": ua, "ia": ia, "ug1": ug1, "ug2": ug2,
                        "ig2": 0.0, "uh": 6.3, "ih": 0.3})
    return pts


def _make_pentode_points(n_ug1=3, n_ua=5, ug2_values=None):
    """Pentode-like grid with multiple Ug2."""
    if ug2_values is None:
        ug2_values = [150.0, 250.0]
    pts = []
    for ug2 in ug2_values:
        for ug1_idx in range(n_ug1):
            ug1 = -1.0 * ug1_idx
            for ua_idx in range(n_ua):
                ua = 50.0 + ua_idx * 50.0
                ia = max(0.0, 10.0 + ug1 * 2.0 + ua * 0.03 + ug2 * 0.01)
                pts.append({"ua": ua, "ia": ia, "ug1": ug1, "ug2": ug2,
                            "ig2": 0.5, "uh": 6.3, "ih": 0.3})
    return pts


# =====================================================================
# filter_ug2_slice
# =====================================================================

class TestFilterUg2Slice(unittest.TestCase):

    def test_triode_returns_all(self):
        pts = _make_triode_points()
        result = filter_ug2_slice(pts, is_triode=True)
        self.assertEqual(len(result), len(pts))

    def test_pentode_filters_to_target(self):
        pts = _make_pentode_points(ug2_values=[150.0, 250.0])
        result = filter_ug2_slice(
            pts, is_triode=False,
            select_ug2_slice=lambda p: 250.0,
            ug2_cluster_thr=2.0,
        )
        ug2_vals = {round(p["ug2"]) for p in result}
        self.assertEqual(ug2_vals, {250})
        self.assertLess(len(result), len(pts))

    def test_no_slice_callable_returns_all(self):
        pts = _make_pentode_points()
        result = filter_ug2_slice(pts, is_triode=False, select_ug2_slice=None)
        self.assertEqual(len(result), len(pts))

    def test_empty_points(self):
        result = filter_ug2_slice([], is_triode=True)
        self.assertEqual(result, [])


# =====================================================================
# build_ia_grid
# =====================================================================

# =====================================================================
# filter_ug2_multi
# =====================================================================

class TestFilterUg2Multi(unittest.TestCase):

    def test_triode_single_group(self):
        pts = _make_triode_points()
        groups = filter_ug2_multi(pts, is_triode=True)
        self.assertEqual(list(groups.keys()), [0.0])
        self.assertEqual(len(groups[0.0]), len(pts))

    def test_pentode_all_groups(self):
        pts = _make_pentode_points(ug2_values=[150.0, 250.0])
        groups = filter_ug2_multi(pts, is_triode=False)
        self.assertEqual(len(groups), 2)
        self.assertIn(150.0, groups)
        self.assertIn(250.0, groups)

    def test_pentode_selected_subset(self):
        pts = _make_pentode_points(ug2_values=[100.0, 200.0, 300.0])
        groups = filter_ug2_multi(pts, is_triode=False, ug2_targets=[200.0])
        self.assertEqual(len(groups), 1)
        self.assertIn(200.0, groups)
        for p in groups[200.0]:
            self.assertAlmostEqual(p["ug2"], 200.0)

    def test_pentode_multiple_selected(self):
        pts = _make_pentode_points(ug2_values=[100.0, 200.0, 300.0])
        groups = filter_ug2_multi(pts, is_triode=False, ug2_targets=[100.0, 300.0])
        self.assertEqual(len(groups), 2)
        self.assertIn(100.0, groups)
        self.assertIn(300.0, groups)
        self.assertNotIn(200.0, groups)

    def test_empty_points(self):
        groups = filter_ug2_multi([], is_triode=False)
        self.assertEqual(groups, {})

    def test_none_targets_returns_all(self):
        pts = _make_pentode_points(ug2_values=[150.0, 250.0])
        groups = filter_ug2_multi(pts, is_triode=False, ug2_targets=None)
        self.assertEqual(len(groups), 2)


class TestBuildIaGrid(unittest.TestCase):

    def test_basic_grid(self):
        pts = _make_triode_points(n_ug1=3, n_ua=5)
        g = build_ia_grid(pts)
        self.assertIsNotNone(g)
        self.assertEqual(len(g["ua_vals"]), 5)
        self.assertEqual(len(g["ug1_vals"]), 3)
        self.assertEqual(g["ia_grid"].shape, (3, 5))
        self.assertFalse(np.any(np.isnan(g["ia_grid"])))

    def test_nan_for_missing_cells(self):
        pts = _make_triode_points(n_ug1=3, n_ua=5)
        pts = [p for p in pts if not (p["ug1"] == 0.0 and p["ua"] == 250.0)]
        g = build_ia_grid(pts)
        self.assertIsNotNone(g)
        self.assertTrue(np.isnan(g["ia_grid"][2, 4]))  # ug1=0 is index 2, ua=250 is index 4

    def test_too_few_points(self):
        pts = [{"ua": 100, "ug1": -1, "ia": 5}]
        self.assertIsNone(build_ia_grid(pts))

    def test_single_ua_returns_none(self):
        pts = [{"ua": 100, "ug1": -i, "ia": i * 5.0} for i in range(5)]
        self.assertIsNone(build_ia_grid(pts))

    def test_clustering_merges_close_ua(self):
        pts = []
        for ua in [99.8, 100.0, 100.2, 200.0]:
            for ug1 in [-2, -1, 0]:
                pts.append({"ua": ua, "ug1": ug1, "ia": 5.0})
        g = build_ia_grid(pts, ua_cluster_thr=1.0)
        self.assertIsNotNone(g)
        self.assertEqual(len(g["ua_vals"]), 2)


# =====================================================================
# build_ia_grid_averaged
# =====================================================================

class TestBuildIaGridAveraged(unittest.TestCase):

    def test_averages_duplicates(self):
        pts = [
            {"ua": 100, "ug1": -1, "ia": 4.0},
            {"ua": 100, "ug1": -1, "ia": 6.0},
            {"ua": 200, "ug1": -1, "ia": 8.0},
            {"ua": 100, "ug1": 0, "ia": 10.0},
            {"ua": 200, "ug1": 0, "ia": 12.0},
        ]
        g = build_ia_grid_averaged(pts)
        self.assertIsNotNone(g)
        # (ug1=-1, ua=100) should be average of 4.0 and 6.0 = 5.0
        ug1_idx = g["ug1_vals"].index(nominal_key(-1, g["ug1_vals"]))
        ua_idx = g["ua_vals"].index(nominal_key(100, g["ua_vals"]))
        self.assertAlmostEqual(g["ia_grid"][ug1_idx, ua_idx], 5.0)


# =====================================================================
# compute_gm_rp_grids
# =====================================================================

class TestComputeGmRpGrids(unittest.TestCase):

    def test_linear_ia_constant_gm(self):
        """Ia linear in Ug1 → constant Gm."""
        ua_vals = [100, 200, 300]
        ug1_vals = [-3, -2, -1, 0]
        # Ia = 5 + 2*ug1 + 0.05*ua (Gm = dIa/dUg1 = 2 mA/V)
        ia = np.array([
            [5 + 2 * ug1 + 0.05 * ua for ua in ua_vals]
            for ug1 in ug1_vals
        ], dtype=float)
        gm, rp = compute_gm_rp_grids(ua_vals, ug1_vals, ia)
        self.assertEqual(gm.shape, (3, 3))  # n_ug1-1 x n_ua
        # Gm should be 2.0 everywhere
        valid = gm[~np.isnan(gm)]
        np.testing.assert_allclose(valid, 2.0, atol=0.01)

    def test_linear_ia_constant_rp(self):
        """Ia linear in Ua → constant Rp."""
        ua_vals = [100, 200, 300]
        ug1_vals = [-2, -1, 0]
        # Ia = 5 + 0.05*ua → dIa/dUa = 0.05 → Rp = 1/0.05 = 20 kΩ
        ia = np.array([
            [5 + 0.05 * ua for ua in ua_vals]
            for ug1 in ug1_vals
        ], dtype=float)
        _, rp = compute_gm_rp_grids(ua_vals, ug1_vals, ia)
        self.assertEqual(rp.shape, (3, 2))  # n_ug1 x n_ua-1
        valid = rp[~np.isnan(rp)]
        np.testing.assert_allclose(valid, 20.0, atol=0.01)

    def test_constant_ia_rp_is_large(self):
        """Ia constant in Ua → Rp very large (slope near zero skipped)."""
        ua_vals = [100, 200, 300]
        ug1_vals = [-2, -1, 0]
        ia = np.full((3, 3), 10.0)
        _, rp = compute_gm_rp_grids(ua_vals, ug1_vals, ia)
        # slope = 0 → should be NaN (abs(slope) <= 1e-6)
        self.assertTrue(np.all(np.isnan(rp)))

    def test_nan_propagation(self):
        """NaN in Ia grid → NaN in Gm/Rp."""
        ua_vals = [100, 200]
        ug1_vals = [-1, 0]
        ia = np.array([[np.nan, 5.0], [3.0, 7.0]])
        gm, rp = compute_gm_rp_grids(ua_vals, ug1_vals, ia)
        self.assertTrue(np.isnan(gm[0, 0]))
        self.assertTrue(np.isnan(rp[0, 0]))

    def test_gm_units(self):
        """Gm in mA/V: Ia in mA, Ug1 in V."""
        ua_vals = [100, 200]
        ug1_vals = [-2, -1]  # 1V step
        # Ia at ug1=-2: [4, 6], at ug1=-1: [8, 10] → dIa = 4, dUg1 = 1 → Gm = 4
        ia = np.array([[4.0, 6.0], [8.0, 10.0]])
        gm, _ = compute_gm_rp_grids(ua_vals, ug1_vals, ia)
        np.testing.assert_allclose(gm[0, :], [4.0, 4.0])

    def test_rp_units(self):
        """Rp in kΩ: Ia in mA, Ua in V."""
        ua_vals = [100, 200]  # 100V step
        ug1_vals = [-1, 0]
        # Ia at ua=100: 5, at ua=200: 10 → dIa = 5 mA, dUa = 100V → Rp = 100/5 = 20 kΩ
        ia = np.array([[5.0, 10.0], [5.0, 10.0]])
        _, rp = compute_gm_rp_grids(ua_vals, ug1_vals, ia)
        np.testing.assert_allclose(rp[:, 0], [20.0, 20.0])


# =====================================================================
# compute_mu_grid
# =====================================================================

class TestComputeMuGrid(unittest.TestCase):

    def test_known_mu(self):
        """mu = Gm * Rp. Gm=2, Rp=50 → mu=100 (like a 12AX7)."""
        ua_vals = [100, 200, 300]
        ug1_vals = [-2, -1, 0]
        # Ia = 2*ug1 + 0.02*ua → Gm=2, slope_ua=0.02 → Rp=50
        ia = np.array([
            [2 * ug1 + 0.02 * ua for ua in ua_vals]
            for ug1 in ug1_vals
        ], dtype=float)
        gm, rp = compute_gm_rp_grids(ua_vals, ug1_vals, ia)
        mu = compute_mu_grid(gm, rp)
        self.assertEqual(mu.shape, (2, 2))
        valid = mu[~np.isnan(mu)]
        np.testing.assert_allclose(valid, 100.0, atol=1.0)

    def test_nan_when_gm_or_rp_nan(self):
        gm = np.array([[1.0, np.nan], [np.nan, 2.0]])
        rp = np.array([[np.nan, 3.0], [4.0, np.nan]])
        mu = compute_mu_grid(gm, rp)
        self.assertTrue(np.all(np.isnan(mu)))

    def test_shape(self):
        gm = np.full((4, 5), 2.0)
        rp = np.full((5, 4), 10.0)
        mu = compute_mu_grid(gm, rp)
        self.assertEqual(mu.shape, (4, 4))


# =====================================================================
# build_pa_grid
# =====================================================================

class TestBuildPaGrid(unittest.TestCase):

    def test_pa_values(self):
        pts = _make_triode_points(n_ug1=3, n_ua=5)
        g = build_pa_grid(pts)
        self.assertIsNotNone(g)
        # Check one point: ua=100, ug1=-1 → ia = 10 + (-1)*2 + 100*0.05 = 13
        # Pa = 100 * 13 / 1000 = 1.3 W
        ug1_idx = g["ug1_vals"].index(nominal_key(-1, g["ug1_vals"]))
        ua_idx = g["ua_vals"].index(nominal_key(100, g["ua_vals"]))
        expected_ia = 10.0 + (-1) * 2.0 + 100 * 0.05
        expected_pa = 100.0 * expected_ia / 1000.0
        self.assertAlmostEqual(g["pa_grid"][ug1_idx, ua_idx], expected_pa, places=2)

    def test_too_few_points(self):
        self.assertIsNone(build_pa_grid([{"ua": 100, "ug1": -1, "ia": 5}]))


if __name__ == "__main__":
    unittest.main()
