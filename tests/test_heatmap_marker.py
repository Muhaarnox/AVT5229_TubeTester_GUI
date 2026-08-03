"""Tests for HeatmapMarker bilinear interpolation (no Qt needed)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.heatmap_interp import interp_bilinear


class TestInterpBilinear(unittest.TestCase):
    """Test interp_bilinear pure function."""

    def _make_grid(self):
        """Simple 3x4 grid for testing.

        x_vals = [0, 10, 20, 30]  (4 columns)
        y_vals = [0, 5, 10]       (3 rows)

        z_grid[row, col]:
            row 0 (y=0):  [0, 10, 20, 30]
            row 1 (y=5):  [5, 15, 25, 35]
            row 2 (y=10): [10, 20, 30, 40]
        """
        x_vals = np.array([0.0, 10.0, 20.0, 30.0])
        y_vals = np.array([0.0, 5.0, 10.0])
        z_grid = np.array([
            [0, 10, 20, 30],
            [5, 15, 25, 35],
            [10, 20, 30, 40],
        ], dtype=float)
        return z_grid, x_vals, y_vals

    def test_exact_grid_point(self):
        """Interpolation at exact grid node returns exact value."""
        z, x, y = self._make_grid()
        self.assertAlmostEqual(interp_bilinear(z, x, y, 0, 0), 0.0)
        self.assertAlmostEqual(interp_bilinear(z, x, y, 10, 5), 15.0)
        self.assertAlmostEqual(interp_bilinear(z, x, y, 30, 10), 40.0)

    def test_center_of_cell(self):
        """Interpolation at center of a cell."""
        z, x, y = self._make_grid()
        # Center of cell (0,0)-(10,5): x=5, y=2.5
        # q00=0, q10=10, q01=5, q11=15
        # dx=0.5, dy=0.5
        # val = 0*0.25 + 10*0.25 + 5*0.25 + 15*0.25 = 7.5
        self.assertAlmostEqual(interp_bilinear(z, x, y, 5, 2.5), 7.5)

    def test_midpoint_x_axis(self):
        """Interpolation at midpoint along X, at grid Y."""
        z, x, y = self._make_grid()
        # (5, 0): between (0,0)=0 and (10,0)=10, dy=0
        self.assertAlmostEqual(interp_bilinear(z, x, y, 5, 0), 5.0)

    def test_midpoint_y_axis(self):
        """Interpolation at midpoint along Y, at grid X."""
        z, x, y = self._make_grid()
        # (0, 2.5): between (0,0)=0 and (0,5)=5, dx=0
        self.assertAlmostEqual(interp_bilinear(z, x, y, 0, 2.5), 2.5)

    def test_out_of_range_x_low(self):
        z, x, y = self._make_grid()
        self.assertIsNone(interp_bilinear(z, x, y, -0.1, 5))

    def test_out_of_range_x_high(self):
        z, x, y = self._make_grid()
        self.assertIsNone(interp_bilinear(z, x, y, 30.1, 5))

    def test_out_of_range_y_low(self):
        z, x, y = self._make_grid()
        self.assertIsNone(interp_bilinear(z, x, y, 10, -0.1))

    def test_out_of_range_y_high(self):
        z, x, y = self._make_grid()
        self.assertIsNone(interp_bilinear(z, x, y, 10, 10.1))

    def test_edge_max_x_max_y(self):
        """Exact max boundary should work."""
        z, x, y = self._make_grid()
        self.assertAlmostEqual(interp_bilinear(z, x, y, 30, 10), 40.0)

    def test_edge_min_x_min_y(self):
        """Exact min boundary should work."""
        z, x, y = self._make_grid()
        self.assertAlmostEqual(interp_bilinear(z, x, y, 0, 0), 0.0)

    def test_nan_one_corner(self):
        """One NaN corner: returns nearest valid corner.

        ML-141: the original pin froze the INVERTED distance formulas
        (its comment even noticed the anomaly — «dists formula uses
        (1-dx) not dx for q00») and asserted the FARTHEST corner q11=15.
        Correct geometry at (1, 1), dx=0.1, dy=0.2:
          q10 at (1,0): (1-0.1)² + 0.2²     = 0.85
          q01 at (0,1): 0.1² + (1-0.2)²     = 0.65  ← nearest valid
          q11 at (1,1): (1-0.1)² + (1-0.2)² = 1.45
        """
        z, x, y = self._make_grid()
        z[0, 0] = np.nan  # q00 = NaN (the truly nearest corner)
        result = interp_bilinear(z, x, y, 1, 1)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 5.0)   # q01

    def test_all_nan(self):
        """All corners NaN: returns None."""
        z = np.full((3, 4), np.nan)
        x = np.array([0.0, 10.0, 20.0, 30.0])
        y = np.array([0.0, 5.0, 10.0])
        self.assertIsNone(interp_bilinear(z, x, y, 5, 2.5))

    def test_single_row(self):
        """Grid with 1 row (ny < 2): returns None."""
        z = np.array([[1.0, 2.0, 3.0]])
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0])
        self.assertIsNone(interp_bilinear(z, x, y, 1, 0))

    def test_single_col(self):
        """Grid with 1 column (nx < 2): returns None."""
        z = np.array([[1.0], [2.0]])
        x = np.array([0.0])
        y = np.array([0.0, 1.0])
        self.assertIsNone(interp_bilinear(z, x, y, 0, 0.5))

    def test_none_inputs(self):
        self.assertIsNone(interp_bilinear(None, None, None, 0, 0))

    def test_shape_mismatch(self):
        """Grid shape doesn't match axis lengths: returns None."""
        z = np.array([[1, 2], [3, 4]], dtype=float)
        x = np.array([0.0, 1.0, 2.0])  # 3 vs 2 columns
        y = np.array([0.0, 1.0])
        self.assertIsNone(interp_bilinear(z, x, y, 0.5, 0.5))

    def test_uneven_spacing(self):
        """Non-uniform axis spacing (like ug1_mid)."""
        x = np.array([0.0, 10.0, 50.0])  # uneven
        y = np.array([0.0, 1.0, 5.0])    # uneven
        z = np.array([
            [0, 10, 50],
            [1, 11, 51],
            [5, 15, 55],
        ], dtype=float)
        # At (10, 1): exact grid point
        self.assertAlmostEqual(interp_bilinear(z, x, y, 10, 1), 11.0)
        # At (30, 3): cell x=[10,50], y=[1,5]
        # dx = (30-10)/(50-10) = 0.5, dy = (3-1)/(5-1) = 0.5
        # q00=11, q10=51, q01=15, q11=55
        # val = 11*0.25 + 51*0.25 + 15*0.25 + 55*0.25 = 33.0
        self.assertAlmostEqual(interp_bilinear(z, x, y, 30, 3), 33.0)

    def test_quarter_position(self):
        """Interpolation at 25% position within cell."""
        z, x, y = self._make_grid()
        # At (2.5, 1.25) in cell (0,0)-(10,5)
        # dx=0.25, dy=0.25
        # q00=0, q10=10, q01=5, q11=15
        # val = 0*0.75*0.75 + 10*0.25*0.75 + 5*0.75*0.25 + 15*0.25*0.25
        #     = 0 + 1.875 + 0.9375 + 0.9375 = 3.75
        self.assertAlmostEqual(interp_bilinear(z, x, y, 2.5, 1.25), 3.75)

    def test_nan_two_corners_nearest_valid(self):
        """Two NaN corners: returns nearest valid."""
        z, x, y = self._make_grid()
        z[0, 0] = np.nan  # q00
        z[0, 1] = np.nan  # q10
        # At (5, 2.5): q01=5, q11=15
        # dx=0.5, dy=0.5
        # q01 dist = (1-0.5)^2 + 0.5^2 = 0.5
        # q11 dist = 0.5^2 + 0.5^2 = 0.5
        # tie: first found wins (q01=5)
        result = interp_bilinear(z, x, y, 5, 2.5)
        self.assertIsNotNone(result)
        self.assertIn(result, [5.0, 15.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
