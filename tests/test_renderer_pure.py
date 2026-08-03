"""Tests for PlotRenderer pure-logic static/class methods.

Covers methods that contain real logic (not thin wrappers around amplifier):
  - _grid_extra: build extra arrays for grid curves (x=Ua, step=Ug1)
  - _grid_extra_ua: build extra arrays for grid curves (x=Ug1, step=Ua)
  - _raw_extra: build extra arrays from raw measurement points
  - _raw_y_values: extract Y values for all param branches
  - _curve_ug2_visible: Ug2 visibility filtering
  - _grid_from_points: grid building with triode/pentode logic
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.plotting import PlotRenderer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pt(ua=100.0, ug1=-2.0, ug2=200.0, ia=10.0, ig2=1.0):
    return {"ua": ua, "ug1": ug1, "ug2": ug2, "ia": ia,
            "ig2": ig2, "uh": 6.3, "ih": 0.3, "series_id": 0}


def _make_points(n_ug1=3, n_ua=5, ug2=250.0):
    pts = []
    for g in range(n_ug1):
        ug1 = -1.0 * g
        for a in range(n_ua):
            ua = 50.0 + a * 50.0
            ia = max(0.0, 10.0 + ug1 * 2 + ua * 0.05)
            pts.append(_pt(ua=ua, ug1=ug1, ug2=ug2, ia=ia, ig2=0.5))
    return pts


# ---------------------------------------------------------------------------
# _grid_extra
# ---------------------------------------------------------------------------

class TestGridExtra(unittest.TestCase):
    """PlotRenderer._grid_extra builds arrays for grid curves (x=Ua)."""

    def test_basic_shape(self):
        xs = [50.0, 100.0, 150.0]
        ia_list = [5.0, 10.0, 15.0]
        d = PlotRenderer._grid_extra(xs, ug1_val=-2.0, ia_list=ia_list)
        self.assertEqual(len(d["Ug1"]), 3)
        self.assertEqual(len(d["Ua"]), 3)
        self.assertEqual(len(d["Ia"]), 3)
        self.assertEqual(len(d["Pa"]), 3)
        self.assertNotIn("Ug2", d)

    def test_pa_calculation(self):
        xs = [100.0, 200.0]
        ia_list = [10.0, 20.0]
        d = PlotRenderer._grid_extra(xs, ug1_val=-1.0, ia_list=ia_list)
        # Pa = Ua * Ia / 1000
        np.testing.assert_allclose(d["Pa"], [1.0, 4.0])

    def test_ug1_filled(self):
        xs = [50.0, 100.0]
        d = PlotRenderer._grid_extra(xs, ug1_val=-3.5, ia_list=[1.0, 2.0])
        np.testing.assert_array_equal(d["Ug1"], [-3.5, -3.5])

    def test_with_ug2(self):
        xs = [100.0]
        d = PlotRenderer._grid_extra(xs, ug1_val=-1.0, ia_list=[5.0],
                                     ug2_val=250.0)
        self.assertIn("Ug2", d)
        np.testing.assert_array_equal(d["Ug2"], [250.0])

    def test_empty(self):
        d = PlotRenderer._grid_extra([], ug1_val=-1.0, ia_list=[])
        self.assertEqual(len(d["Ua"]), 0)
        self.assertEqual(len(d["Pa"]), 0)


# ---------------------------------------------------------------------------
# _grid_extra_ua
# ---------------------------------------------------------------------------

class TestGridExtraUa(unittest.TestCase):
    """PlotRenderer._grid_extra_ua builds arrays for grid curves (x=Ug1)."""

    def test_basic_shape(self):
        xs = [-1.0, -2.0, -3.0]
        ia_list = [15.0, 10.0, 5.0]
        d = PlotRenderer._grid_extra_ua(xs, ua_val=200.0, ia_list=ia_list)
        self.assertEqual(len(d["Ug1"]), 3)
        self.assertEqual(len(d["Ua"]), 3)
        np.testing.assert_array_equal(d["Ua"], [200.0, 200.0, 200.0])

    def test_pa_calculation(self):
        xs = [-1.0, -2.0]
        ia_list = [10.0, 5.0]
        d = PlotRenderer._grid_extra_ua(xs, ua_val=200.0, ia_list=ia_list)
        # Pa = ua_val * Ia / 1000
        np.testing.assert_allclose(d["Pa"], [2.0, 1.0])

    def test_with_ug2(self):
        xs = [-1.0]
        d = PlotRenderer._grid_extra_ua(xs, ua_val=150.0, ia_list=[8.0],
                                        ug2_val=300.0)
        self.assertIn("Ug2", d)
        np.testing.assert_array_equal(d["Ug2"], [300.0])

    def test_without_ug2(self):
        xs = [-1.0]
        d = PlotRenderer._grid_extra_ua(xs, ua_val=150.0, ia_list=[8.0])
        self.assertNotIn("Ug2", d)


# ---------------------------------------------------------------------------
# _raw_extra
# ---------------------------------------------------------------------------

class TestRawExtra(unittest.TestCase):
    """PlotRenderer._raw_extra builds extra arrays from raw points."""

    def test_all_fields_present(self):
        pts = [_pt(ua=100, ug1=-2, ug2=200, ia=10, ig2=1.0),
               _pt(ua=200, ug1=-3, ug2=250, ia=20, ig2=2.0)]
        d = PlotRenderer._raw_extra(pts)
        for key in ("Ua", "Ug1", "Ug2", "Ia", "Ig2", "Pa"):
            self.assertIn(key, d)
            self.assertEqual(len(d[key]), 2)

    def test_values(self):
        pts = [_pt(ua=100, ug1=-2, ug2=200, ia=10, ig2=1.5)]
        d = PlotRenderer._raw_extra(pts)
        self.assertAlmostEqual(d["Ua"][0], 100.0)
        self.assertAlmostEqual(d["Ug1"][0], -2.0)
        self.assertAlmostEqual(d["Ug2"][0], 200.0)
        self.assertAlmostEqual(d["Ia"][0], 10.0)
        self.assertAlmostEqual(d["Ig2"][0], 1.5)

    def test_pa_calculation(self):
        pts = [_pt(ua=200, ia=15)]
        d = PlotRenderer._raw_extra(pts)
        np.testing.assert_allclose(d["Pa"], [3.0])  # 200*15/1000

    def test_empty(self):
        d = PlotRenderer._raw_extra([])
        self.assertEqual(len(d["Ua"]), 0)

    def test_missing_fields_default_zero(self):
        pts = [{"uh": 6.3}]  # no ua, ia, etc.
        d = PlotRenderer._raw_extra(pts)
        self.assertAlmostEqual(d["Ua"][0], 0.0)
        self.assertAlmostEqual(d["Ia"][0], 0.0)
        self.assertAlmostEqual(d["Pa"][0], 0.0)


# ---------------------------------------------------------------------------
# _raw_y_values
# ---------------------------------------------------------------------------

class TestRawYValues(unittest.TestCase):
    """PlotRenderer._raw_y_values extracts Y for all param branches."""

    def setUp(self):
        self.pts = [
            _pt(ua=100, ia=10, ug2=200, ig2=2.0),
            _pt(ua=200, ia=20, ug2=250, ig2=5.0),
        ]

    def test_ia(self):
        ys = PlotRenderer._raw_y_values(self.pts, "Ia")
        self.assertEqual(ys, [10.0, 20.0])

    def test_ig2(self):
        ys = PlotRenderer._raw_y_values(self.pts, "Ig2")
        self.assertEqual(ys, [2.0, 5.0])

    def test_pa(self):
        ys = PlotRenderer._raw_y_values(self.pts, "Pa")
        # Pa = ua * ia / 1000
        self.assertAlmostEqual(ys[0], 1.0)    # 100*10/1000
        self.assertAlmostEqual(ys[1], 4.0)    # 200*20/1000

    def test_pig2(self):
        ys = PlotRenderer._raw_y_values(self.pts, "Pig2")
        # Pig2 = ug2 * ig2 / 1000
        self.assertAlmostEqual(ys[0], 0.4)    # 200*2/1000
        self.assertAlmostEqual(ys[1], 1.25)   # 250*5/1000

    def test_ia_ig2_ratio(self):
        ys = PlotRenderer._raw_y_values(self.pts, "Ia/Ig2")
        self.assertAlmostEqual(ys[0], 5.0)    # 10/2
        self.assertAlmostEqual(ys[1], 4.0)    # 20/5

    def test_ia_ig2_near_zero(self):
        """ig2 <= 0.01 should return 0.0 (avoid division by zero)."""
        pts = [_pt(ia=10, ig2=0.005)]
        ys = PlotRenderer._raw_y_values(pts, "Ia/Ig2")
        self.assertEqual(ys, [0.0])

    def test_ia_ig2_exactly_threshold(self):
        """ig2 == 0.01 should return 0.0 (boundary: not > 0.01)."""
        pts = [_pt(ia=10, ig2=0.01)]
        ys = PlotRenderer._raw_y_values(pts, "Ia/Ig2")
        self.assertEqual(ys, [0.0])

    def test_unknown_param(self):
        ys = PlotRenderer._raw_y_values(self.pts, "Unknown")
        self.assertEqual(ys, [0.0, 0.0])

    def test_empty_points(self):
        ys = PlotRenderer._raw_y_values([], "Ia")
        self.assertEqual(ys, [])


# ---------------------------------------------------------------------------
# _curve_ug2_visible
# ---------------------------------------------------------------------------

class TestCurveUg2Visible(unittest.TestCase):
    """PlotRenderer._curve_ug2_visible checks Ug2 filtering."""

    def _make_curve(self, ug2_val=None):
        """Build a minimal CurveData-like object with extra dict."""
        from lm19.curve_data import CurveData
        extra = {}
        if ug2_val is not None:
            extra["Ug2"] = np.array([ug2_val])
        return CurveData(
            x=np.array([1.0, 2.0]),
            y=np.array([3.0, 4.0]),
            extra=extra,
        )

    def test_no_ug2_always_visible(self):
        curve = self._make_curve(ug2_val=None)
        self.assertTrue(
            PlotRenderer._curve_ug2_visible(curve, [200.0], thr=2.0))

    def test_exact_match(self):
        curve = self._make_curve(ug2_val=200.0)
        self.assertTrue(
            PlotRenderer._curve_ug2_visible(curve, [200.0], thr=2.0))

    def test_within_threshold(self):
        curve = self._make_curve(ug2_val=201.5)
        self.assertTrue(
            PlotRenderer._curve_ug2_visible(curve, [200.0], thr=2.0))

    def test_outside_threshold(self):
        curve = self._make_curve(ug2_val=205.0)
        self.assertFalse(
            PlotRenderer._curve_ug2_visible(curve, [200.0], thr=2.0))

    def test_multiple_checked(self):
        curve = self._make_curve(ug2_val=300.0)
        self.assertTrue(
            PlotRenderer._curve_ug2_visible(curve, [200.0, 300.0], thr=2.0))

    def test_empty_checked_list(self):
        curve = self._make_curve(ug2_val=200.0)
        self.assertFalse(
            PlotRenderer._curve_ug2_visible(curve, [], thr=2.0))


# ---------------------------------------------------------------------------
# _grid_from_points
# ---------------------------------------------------------------------------

class TestGridFromPoints(unittest.TestCase):
    """PlotRenderer._grid_from_points builds ia grid."""

    def _make_renderer(self, is_triode=True):
        r = object.__new__(PlotRenderer)
        r.is_triode = is_triode
        r.track_sids = set()
        r.ua_cluster_thr = 2.0
        r.ug1_cluster_thr = 0.3
        r.ug2_cluster_thr = 2.0
        return r

    def test_triode_returns_grid(self):
        r = self._make_renderer(is_triode=True)
        pts = _make_points(n_ug1=3, n_ua=5, ug2=0.0)
        g = r._grid_from_points(pts, select_ug2_slice=lambda p: 0.0)
        self.assertIsNotNone(g)
        self.assertIn("ua", g)
        self.assertIn("ug1", g)
        self.assertIn("z", g)
        self.assertAlmostEqual(g["ug2"], 0.0)

    def test_empty_returns_none(self):
        r = self._make_renderer()
        self.assertIsNone(r._grid_from_points([], lambda p: 0.0))

    def test_pentode_with_ug2(self):
        r = self._make_renderer(is_triode=False)
        pts = _make_points(n_ug1=3, n_ua=5, ug2=250.0)
        g = r._grid_from_points(pts, select_ug2_slice=lambda p: 250.0)
        self.assertIsNotNone(g)
        self.assertAlmostEqual(g["ug2"], 250.0)

    def test_tracking_treated_as_triode(self):
        r = self._make_renderer(is_triode=False)
        r.track_sids = {0}
        pts = _make_points(n_ug1=3, n_ua=5, ug2=0.0)
        g = r._grid_from_points(pts, select_ug2_slice=lambda p: 0.0)
        self.assertIsNotNone(g)
        self.assertAlmostEqual(g["ug2"], 0.0)


if __name__ == "__main__":
    unittest.main()
