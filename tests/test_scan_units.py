"""Unit tests for scan helper functions: _frange, _snap_ug1_key, _exceeds_*."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from scan_test_helpers import (
    _frange, _snap_ug1_key, _exceeds_pa, _exceeds_pg2, _exceeds_ig2,
)


# ---------------------------------------------------------------------------
# _frange
# ---------------------------------------------------------------------------

class TestFrange(unittest.TestCase):

    def test_ascending(self):
        result = _frange(0, 30, 10)
        self.assertEqual(result, [0, 10, 20, 30])

    def test_ascending_non_exact(self):
        """Stop is not an exact multiple — last point should still be included."""
        result = _frange(0, 25, 10)
        self.assertEqual(result, [0, 10, 20])

    def test_descending(self):
        result = _frange(30, 0, 10)
        self.assertEqual(result, [30, 20, 10, 0])

    def test_step_zero(self):
        result = _frange(42, 100, 0)
        self.assertEqual(result, [42])

    def test_single_point(self):
        """start == stop → single point."""
        result = _frange(50, 50, 10)
        self.assertEqual(result, [50])

    def test_fractional_step(self):
        result = _frange(0, 1.0, 0.5)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.5)
        self.assertAlmostEqual(result[2], 1.0)
        self.assertEqual(len(result), 3)

    def test_negative_ascending(self):
        """Ug1 range: -20 to -10."""
        result = _frange(-20, -10, 2)
        self.assertEqual(result, [-20, -18, -16, -14, -12, -10])


# ---------------------------------------------------------------------------
# _snap_ug1_key
# ---------------------------------------------------------------------------

class TestSnapUg1Key(unittest.TestCase):

    def test_no_grid_returns_rounded(self):
        self.assertAlmostEqual(_snap_ug1_key(-4.567, None), -4.6)

    def test_empty_grid_returns_rounded(self):
        self.assertAlmostEqual(_snap_ug1_key(-4.567, []), -4.6)

    def test_exact_match(self):
        grid = [-10.0, -8.0, -6.0, -4.0, -2.0]
        self.assertAlmostEqual(_snap_ug1_key(-6.0, grid), -6.0)

    def test_snaps_to_nearest(self):
        grid = [-10.0, -8.0, -6.0, -4.0, -2.0]
        # -5.3 is closer to -6.0 than to -4.0
        self.assertAlmostEqual(_snap_ug1_key(-5.3, grid), -6.0)

    def test_snaps_to_nearest_other_side(self):
        grid = [-10.0, -8.0, -6.0, -4.0, -2.0]
        # -4.8 is closer to -4.0 than to -6.0
        # Wait: |-4.8 - (-4.0)| = 0.8, |-4.8 - (-6.0)| = 1.2 → snaps to -4.0
        self.assertAlmostEqual(_snap_ug1_key(-4.8, grid), -4.0)

    def test_small_device_drift(self):
        """Device reports -5.95 instead of -6.0 → snaps to -6.0."""
        grid = [-10.0, -8.0, -6.0, -4.0, -2.0]
        self.assertAlmostEqual(_snap_ug1_key(-5.95, grid), -6.0)

    def test_snaps_to_edge_start(self):
        grid = [-10.0, -8.0, -6.0]
        self.assertAlmostEqual(_snap_ug1_key(-11.0, grid), -10.0)

    def test_snaps_to_edge_end(self):
        grid = [-10.0, -8.0, -6.0]
        self.assertAlmostEqual(_snap_ug1_key(-5.0, grid), -6.0)


# ---------------------------------------------------------------------------
# _exceeds_pa / _exceeds_pg2 / _exceeds_ig2
# ---------------------------------------------------------------------------

class TestExceedsPa(unittest.TestCase):

    def test_zero_limit_disabled(self):
        point = {"ua": 300, "ia": 100}  # Pa = 30W
        self.assertFalse(_exceeds_pa(point, 0.0))

    def test_below_limit(self):
        # Pa = 100V × 10mA / 1000 = 1.0W
        point = {"ua": 100, "ia": 10}
        self.assertFalse(_exceeds_pa(point, 2.0))

    def test_above_limit(self):
        # Pa = 200V × 50mA / 1000 = 10W
        point = {"ua": 200, "ia": 50}
        self.assertTrue(_exceeds_pa(point, 5.0))

    def test_exact_limit(self):
        # Pa = 100V × 10mA / 1000 = 1.0W, limit = 1.0W
        # Not exceeded (equal, not greater)
        point = {"ua": 100, "ia": 10}
        self.assertFalse(_exceeds_pa(point, 1.0))


class TestExceedsPg2(unittest.TestCase):

    def test_zero_limit_disabled(self):
        point = {"ug2": 200, "ig2": 10}
        self.assertFalse(_exceeds_pg2(point, 0.0, 200))

    def test_zero_ug2_nominal_disabled(self):
        """Triode mode: ug2_nominal=0 → always False."""
        point = {"ug2": 200, "ig2": 10}
        self.assertFalse(_exceeds_pg2(point, 1.0, 0.0))

    def test_below_limit(self):
        # Pg2 = 100V × 5mA / 1000 = 0.5W
        point = {"ug2": 100, "ig2": 5}
        self.assertFalse(_exceeds_pg2(point, 1.0, 100))

    def test_above_limit(self):
        # Pg2 = 200V × 20mA / 1000 = 4W
        point = {"ug2": 200, "ig2": 20}
        self.assertTrue(_exceeds_pg2(point, 1.0, 200))


class TestExceedsIg2(unittest.TestCase):

    def test_zero_limit_disabled(self):
        point = {"ig2": 50}
        self.assertFalse(_exceeds_ig2(point, 0.0))

    def test_below_limit(self):
        point = {"ig2": 5}
        self.assertFalse(_exceeds_ig2(point, 10.0))

    def test_above_limit(self):
        point = {"ig2": 15}
        self.assertTrue(_exceeds_ig2(point, 10.0))

    def test_exact_limit_not_exceeded(self):
        point = {"ig2": 10}
        self.assertFalse(_exceeds_ig2(point, 10.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
