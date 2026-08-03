"""Tests for lamp & Ug2 selector logic in PlotManager."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.analysis import select_analysis_points, get_available_series


def _pts(n=5, series_id=0, ug2=150.0):
    """Generate simple test points with series_id (default 0 = scan)."""
    out = []
    for i in range(n):
        p = {"ua": 50.0 + i * 50, "ug1": -1.0, "ia": 5.0 + i, "ug2": ug2,
             "ig2": 0.1, "uh": 6.3, "ih": 0.3, "series_id": series_id}
        out.append(p)
    return out


class TestSelectAnalysisPoints(unittest.TestCase):

    def test_current_scan_preferred(self):
        pts = _pts(3, series_id=0) + _pts(2, series_id=1)
        result = select_analysis_points(pts)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(p.get("series_id") == 0 for p in result))

    def test_specific_series(self):
        pts = _pts(2, series_id=0) + _pts(3, series_id=1)
        result = select_analysis_points(pts, series_id=1)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(p["series_id"] == 1 for p in result))

    def test_fallback_all(self):
        pts = _pts(2, series_id=1) + _pts(3, series_id=2)
        result = select_analysis_points(pts)
        self.assertEqual(len(result), 5)

    def test_empty(self):
        self.assertEqual(select_analysis_points([]), [])


class TestGetAvailableSeries(unittest.TestCase):

    def test_current_and_series(self):
        pts = _pts(2, series_id=0) + _pts(3, series_id=1)
        labels = {1: "EL84"}
        sources = get_available_series(pts, labels)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["series_id"], 0)
        self.assertEqual(sources[0]["n_points"], 2)
        self.assertEqual(sources[1]["series_id"], 1)
        self.assertEqual(sources[1]["label"], "EL84")

    def test_multiple_series(self):
        pts = _pts(2, series_id=0) + _pts(3, series_id=1)
        sources = get_available_series(pts)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["series_id"], 0)
        self.assertEqual(sources[1]["series_id"], 1)

    def test_empty(self):
        self.assertEqual(get_available_series([]), [])


class TestPointInSeries(unittest.TestCase):
    """Test _point_in_series module-level function."""

    def test_scan_point_matches_zero(self):
        from app.plot_manager import _point_in_series
        p = {"ua": 100, "ug1": -1, "ia": 5, "series_id": 0}
        self.assertTrue(_point_in_series(p, [0, 1]))
        self.assertFalse(_point_in_series(p, [1, 2]))

    def test_series_id_matches(self):
        from app.plot_manager import _point_in_series
        p = {"ua": 100, "ug1": -1, "ia": 5, "series_id": 2}
        self.assertTrue(_point_in_series(p, [2, 3]))
        self.assertFalse(_point_in_series(p, [0, 1]))


class TestFilterBySeries(unittest.TestCase):
    """Test filter_by_series pure function."""

    def test_all_checked_returns_all(self):
        from app.plot_manager import filter_by_series
        pts = _pts(3, series_id=0) + _pts(2, series_id=1)
        result = filter_by_series(pts, [0, 1], total_count=2)
        self.assertEqual(len(result), 5)

    def test_partial_selection(self):
        from app.plot_manager import filter_by_series
        pts = _pts(3, series_id=0) + _pts(2, series_id=1) + _pts(4, series_id=2)
        result = filter_by_series(pts, [0, 2], total_count=3)
        self.assertEqual(len(result), 7)
        for p in result:
            self.assertIn(p["series_id"], (0, 2))

    def test_none_checked_returns_all(self):
        from app.plot_manager import filter_by_series
        pts = _pts(5)
        result = filter_by_series(pts, None, total_count=1)
        self.assertEqual(len(result), 5)

    def test_only_current_scan(self):
        from app.plot_manager import filter_by_series
        pts = _pts(3, series_id=0) + _pts(2, series_id=1)
        result = filter_by_series(pts, [0], total_count=2)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(p["series_id"] == 0 for p in result))

    def test_only_series(self):
        from app.plot_manager import filter_by_series
        pts = _pts(3, series_id=0) + _pts(2, series_id=1)
        result = filter_by_series(pts, [1], total_count=2)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(p.get("series_id") == 1 for p in result))

    def test_empty_points(self):
        from app.plot_manager import filter_by_series
        self.assertEqual(filter_by_series([], [0], total_count=1), [])


class TestFilterByUg2Display(unittest.TestCase):
    """Test filter_by_ug2_display pure function."""

    def test_triode_returns_all(self):
        from app.plot_manager import filter_by_ug2_display
        pts = _pts(5, ug2=0)
        result = filter_by_ug2_display(pts, [0.0], is_triode=True, ug2_cluster_thr=2.0)
        self.assertEqual(len(result), 5)

    def test_no_selection_returns_all(self):
        from app.plot_manager import filter_by_ug2_display
        pts = _pts(3, ug2=150) + _pts(3, ug2=200)
        result = filter_by_ug2_display(pts, [], is_triode=False, ug2_cluster_thr=2.0)
        self.assertEqual(len(result), 6)

    def test_none_selection_returns_all(self):
        from app.plot_manager import filter_by_ug2_display
        pts = _pts(3, ug2=150) + _pts(3, ug2=200)
        result = filter_by_ug2_display(pts, None, is_triode=False, ug2_cluster_thr=2.0)
        self.assertEqual(len(result), 6)

    def test_single_ug2_selected(self):
        from app.plot_manager import filter_by_ug2_display
        pts = _pts(3, ug2=150) + _pts(4, ug2=200)
        result = filter_by_ug2_display(pts, [150.0], is_triode=False, ug2_cluster_thr=2.0)
        self.assertEqual(len(result), 3)
        for p in result:
            self.assertAlmostEqual(p["ug2"], 150.0, delta=2.0)

    def test_multiple_ug2_selected(self):
        from app.plot_manager import filter_by_ug2_display
        pts = _pts(2, ug2=100) + _pts(3, ug2=200) + _pts(4, ug2=300)
        result = filter_by_ug2_display(pts, [100.0, 300.0], is_triode=False, ug2_cluster_thr=2.0)
        self.assertEqual(len(result), 6)

    def test_empty_points(self):
        from app.plot_manager import filter_by_ug2_display
        self.assertEqual(filter_by_ug2_display([], [150.0], is_triode=False, ug2_cluster_thr=2.0), [])


class TestFilterByCalcSeries(unittest.TestCase):
    """Test filter_by_calc_series pure function."""

    def test_none_returns_empty(self):
        from app.plot_manager import filter_by_calc_series
        pts = _pts(3, series_id=0) + _pts(2, series_id=1)
        result = filter_by_calc_series(pts, None)
        self.assertEqual(len(result), 0)

    def test_zero_returns_current_scan(self):
        from app.plot_manager import filter_by_calc_series
        pts = _pts(3, series_id=0) + _pts(2, series_id=1)
        result = filter_by_calc_series(pts, 0)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(p["series_id"] == 0 for p in result))

    def test_specific_series(self):
        from app.plot_manager import filter_by_calc_series
        pts = _pts(3, series_id=0) + _pts(4, series_id=1)
        result = filter_by_calc_series(pts, 1)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(p.get("series_id") == 1 for p in result))

    def test_nonexistent_series_fallback(self):
        from app.plot_manager import filter_by_calc_series
        pts = _pts(3, series_id=0)
        result = filter_by_calc_series(pts, 99)
        self.assertEqual(len(result), 3)

    def test_empty_points(self):
        from app.plot_manager import filter_by_calc_series
        self.assertEqual(filter_by_calc_series([], None), [])
        self.assertEqual(filter_by_calc_series([], 0), [])


class TestUg2Clustering(unittest.TestCase):
    """Test that refresh_ug2_combos produces clean values."""

    def test_cluster_averaging(self):
        from lm19.plotting.grids import cluster_nominal
        raw = [149.5, 149.8, 150.1, 150.3, 200.0, 200.2]
        thr = 2.0
        noms = cluster_nominal(raw, threshold=thr)
        values = []
        for nom in noms:
            members = [v for v in raw if abs(v - nom) <= thr]
            values.append(round(sum(members) / len(members), 1))
        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(values[0], 149.9, places=0)
        self.assertAlmostEqual(values[1], 200.1, places=0)


if __name__ == "__main__":
    unittest.main()
