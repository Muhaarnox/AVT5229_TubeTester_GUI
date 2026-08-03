"""Unit tests for load line intersections, interpolation, 5-point distortion, and IMD.

Run:  py -m pytest tests/test_loadline.py -v
  or: py tests/test_loadline.py
"""

import os
import sys
import math
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.plotting import PlotRenderer


class TestFindLoadLineIntersections(unittest.TestCase):
    """Test PlotRenderer._find_load_line_intersections static method."""

    def test_basic_intersection(self):
        """Known linear Ia curve should intersect the load line."""
        # Ia = 10 * (1 - Ua/300) for Ug1=-1  (linear: Ia=10 at Ua=0, Ia=0 at Ua=300)
        # Load line: Ub=300, Ra=30kΩ  → Ia = (300 - Ua)/30
        # Intersection: 10*(1-Ua/300) = (300-Ua)/30 → solve
        # Both give Ia=0 at Ua=300, Ia=10 at Ua=0 → lines are identical? Let's pick something else.
        # Ia curve: constant Ia=5 mA for all Ua
        # Load line: Ub=400, Ra=40kΩ → Ia = (400-Ua)/40
        # Intersection: 5 = (400-Ua)/40 → Ua = 200
        points = [
            {"ua": 50, "ug1": -1.0, "ia": 5.0},
            {"ua": 100, "ug1": -1.0, "ia": 5.0},
            {"ua": 150, "ug1": -1.0, "ia": 5.0},
            {"ua": 200, "ug1": -1.0, "ia": 5.0},
            {"ua": 250, "ug1": -1.0, "ia": 5.0},
            {"ua": 300, "ug1": -1.0, "ia": 5.0},
        ]
        isects = PlotRenderer._find_load_line_intersections(points, ub=400, ra=40)
        self.assertEqual(len(isects), 1)
        self.assertAlmostEqual(isects[0]["ua"], 200.0, places=0)
        self.assertAlmostEqual(isects[0]["ia"], 5.0, places=1)

    def test_no_intersection(self):
        """Curve entirely above load line should give no intersections."""
        # Ia always much higher than load line
        points = [
            {"ua": 50, "ug1": -1.0, "ia": 100.0},
            {"ua": 100, "ug1": -1.0, "ia": 100.0},
            {"ua": 150, "ug1": -1.0, "ia": 100.0},
        ]
        # Load line: Ub=100, Ra=100kΩ → Ia = (100-Ua)/100, max 1 mA
        isects = PlotRenderer._find_load_line_intersections(points, ub=100, ra=100)
        self.assertEqual(len(isects), 0)

    def test_empty_points(self):
        self.assertEqual(PlotRenderer._find_load_line_intersections([], 300, 10), [])

    def test_invalid_params(self):
        points = [{"ua": 100, "ug1": -1, "ia": 5}]
        self.assertEqual(PlotRenderer._find_load_line_intersections(points, 0, 10), [])
        self.assertEqual(PlotRenderer._find_load_line_intersections(points, 300, 0), [])

    def test_multiple_ug1_curves(self):
        """Each Ug1 curve should produce at most one intersection."""
        # Two constant-Ia curves at different Ug1, covering wide Ua range
        points = [
            {"ua": 50, "ug1": -1.0, "ia": 5.0},
            {"ua": 150, "ug1": -1.0, "ia": 5.0},
            {"ua": 250, "ug1": -1.0, "ia": 5.0},
            {"ua": 350, "ug1": -1.0, "ia": 5.0},
            {"ua": 50, "ug1": -2.0, "ia": 3.0},
            {"ua": 150, "ug1": -2.0, "ia": 3.0},
            {"ua": 250, "ug1": -2.0, "ia": 3.0},
            {"ua": 350, "ug1": -2.0, "ia": 3.0},
        ]
        # Load line: Ub=400, Ra=40kΩ
        # → crosses Ia=5 at Ua=200, Ia=3 at Ua=280
        isects = PlotRenderer._find_load_line_intersections(points, ub=400, ra=40)
        self.assertEqual(len(isects), 2)
        # Sorted by ug1
        self.assertAlmostEqual(isects[0]["ug1"], -2.0)
        self.assertAlmostEqual(isects[1]["ug1"], -1.0)


class TestInterpIntersection(unittest.TestCase):
    """Test PlotRenderer._interp_intersection static method."""

    def test_empty(self):
        self.assertIsNone(PlotRenderer._interp_intersection([], -1.5))

    def test_single_point(self):
        pts = [{"ug1": -2.0, "ua": 200, "ia": 5.0}]
        result = PlotRenderer._interp_intersection(pts, -1.5)
        self.assertEqual(result["ia"], 5.0)

    def test_exact_match(self):
        pts = [
            {"ug1": -2.0, "ua": 200, "ia": 5.0},
            {"ug1": -1.0, "ua": 150, "ia": 8.0},
        ]
        result = PlotRenderer._interp_intersection(pts, -2.0)
        self.assertAlmostEqual(result["ia"], 5.0)

    def test_interpolation(self):
        """Midpoint interpolation between two known points."""
        pts = [
            {"ug1": -2.0, "ua": 200, "ia": 4.0},
            {"ug1": -1.0, "ua": 150, "ia": 8.0},
        ]
        result = PlotRenderer._interp_intersection(pts, -1.5)
        self.assertAlmostEqual(result["ia"], 6.0, places=1)
        self.assertAlmostEqual(result["ua"], 175.0, places=1)

    def test_extrapolation_below(self):
        """Extrapolation below the data range."""
        pts = [
            {"ug1": -2.0, "ua": 200, "ia": 4.0},
            {"ug1": -1.0, "ua": 150, "ia": 8.0},
        ]
        result = PlotRenderer._interp_intersection(pts, -3.0)
        # Linear extrapolation: ia = 4 + (-3 - (-2)) / (-1 - (-2)) * (8 - 4) = 4 - 4 = 0
        self.assertAlmostEqual(result["ia"], 0.0, places=1)


class TestCompute5PointDistortion(unittest.TestCase):
    """Test PlotRenderer._compute_5point_distortion."""

    def _make_linear_intersections(self):
        """Create perfectly linear intersections (zero distortion).

        Dense Ug1 grid (0.5V step) ensures 5-point sparse-data guard
        passes for typical bias/swing combos used in tests below.
        """
        # Ia = 10 + 2*Ug1  (linear)
        return [
            {"ug1": -3.0, "ua": 260, "ia": 4.0},
            {"ug1": -2.5, "ua": 250, "ia": 5.0},
            {"ug1": -2.0, "ua": 240, "ia": 6.0},
            {"ug1": -1.5, "ua": 230, "ia": 7.0},
            {"ug1": -1.0, "ua": 220, "ia": 8.0},
            {"ug1": -0.5, "ua": 210, "ia": 9.0},
            {"ug1": 0.0,  "ua": 200, "ia": 10.0},
        ]

    def test_not_enough_points(self):
        """Should return None with fewer than 3 intersections."""
        pts = [{"ug1": -2, "ua": 200, "ia": 5}]
        self.assertIsNone(PlotRenderer._compute_5point_distortion(pts))

    def test_linear_zero_distortion(self):
        """A perfectly linear transfer curve should give ~zero HD2/HD3."""
        isects = self._make_linear_intersections()
        result = PlotRenderer._compute_5point_distortion(isects)
        self.assertIsNotNone(result)
        self.assertLess(result["hd2"], 0.5)
        self.assertLess(result["hd3"], 0.5)

    def test_returns_swing_info(self):
        """Result should contain Q-point and swing info."""
        isects = self._make_linear_intersections()
        result = PlotRenderer._compute_5point_distortion(isects)
        self.assertIn("ug1_0", result)
        self.assertIn("ia_0", result)
        self.assertIn("half_swing", result)
        self.assertIn("pout_mw", result)
        self.assertIn("pt_neg", result)
        self.assertIn("pt_pos", result)

    def test_manual_half_swing(self):
        """Manual half_swing should trigger interpolation mode."""
        isects = self._make_linear_intersections()
        result = PlotRenderer._compute_5point_distortion(
            isects, ug1_bias=-1.5, half_swing=1.0,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result["interpolated"])

    def test_nonlinear_gives_hd2(self):
        """Quadratic nonlinearity should produce measurable HD2."""
        # Ia = 10 + 3*ug1 + 0.2*ug1^2  (asymmetric transfer curve → HD2)
        isects = []
        for ug1 in np.arange(-3.0, 0.5, 0.5):
            ia = 10.0 + 3.0 * ug1 + 0.2 * ug1 ** 2
            ua = 250.0 - 20.0 * ug1
            if ia > 0:
                isects.append({"ug1": float(ug1), "ua": float(ua), "ia": float(ia)})
        result = PlotRenderer._compute_5point_distortion(isects)
        self.assertIsNotNone(result)
        self.assertGreater(result["hd2"], 1.0)  # significant HD2


class TestComputeImd(unittest.TestCase):
    """Test PlotRenderer.compute_imd."""

    def test_not_enough_points(self):
        pts = [{"ug1": -2, "ua": 200, "ia": 5}]
        self.assertIsNone(PlotRenderer.compute_imd(pts))

    def test_linear_near_zero(self):
        """Perfectly linear data should give near-zero IMD."""
        isects = [
            {"ug1": -4.0, "ua": 260, "ia": 2.0},
            {"ug1": -3.0, "ua": 245, "ia": 4.0},
            {"ug1": -2.0, "ua": 230, "ia": 6.0},
            {"ug1": -1.0, "ua": 215, "ia": 8.0},
            {"ug1": 0.0, "ua": 200, "ia": 10.0},
        ]
        result = PlotRenderer.compute_imd(isects)
        self.assertIsNotNone(result)
        self.assertLess(result["imd2"], 1.0)
        self.assertLess(result["imd3"], 1.0)

    def test_nonlinear_gives_imd(self):
        """Quadratic nonlinearity should produce measurable IMD2."""
        isects = []
        for ug1 in np.arange(-4.0, 0.5, 0.5):
            ia = 5.0 + 2.0 * ug1 + 0.3 * ug1 ** 2
            ua = 250.0 - 10 * ug1
            if ia > 0:
                isects.append({"ug1": ug1, "ua": ua, "ia": ia})
        result = PlotRenderer.compute_imd(isects)
        self.assertIsNotNone(result)
        self.assertGreater(result["imd2"], 0.5)

    def test_returns_coefficients(self):
        """Result should contain polynomial coefficients."""
        isects = [
            {"ug1": -4.0, "ua": 260, "ia": 2.0},
            {"ug1": -3.0, "ua": 245, "ia": 4.0},
            {"ug1": -2.0, "ua": 230, "ia": 6.0},
            {"ug1": -1.0, "ua": 215, "ia": 8.0},
        ]
        result = PlotRenderer.compute_imd(isects)
        self.assertIsNotNone(result)
        self.assertIn("a0", result)
        self.assertIn("a1", result)
        self.assertIn("a2", result)
        self.assertIn("a3", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
