"""Tests for PlotRenderer grouping cache.

Verifies that:
  - _ensure_2d_cache returns consistent groupings
  - invalidate_cache clears the cache
  - Repeated calls with same data reuse the cache
  - Cache is invalidated when data length changes
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.plotting import PlotRenderer


def _make_points(n_ug1=3, n_ua=5, ug2=250.0, series_id=0):
    """Generate simple triode-like points."""
    pts = []
    for ug1_idx in range(n_ug1):
        ug1 = -1.0 * ug1_idx
        for ua_idx in range(n_ua):
            ua = 50.0 + ua_idx * 50.0
            ia = max(0, 10.0 + ug1 * 2 + ua * 0.05)
            pts.append({"ua": ua, "ia": ia, "ug1": ug1, "ug2": ug2,
                         "ig2": 0.1, "uh": 6.3, "ih": 0.3,
                         "series_id": series_id})
    return pts


class TestEnsure2dCache(unittest.TestCase):
    """Test _ensure_2d_cache grouping and caching."""

    def _make_renderer(self):
        """Create a PlotRenderer with None plots (no Qt needed for cache)."""
        r = object.__new__(PlotRenderer)
        r.is_triode = True
        r.ua_cluster_thr = 2.0
        r.ug1_cluster_thr = 0.3
        r.ug2_cluster_thr = 2.0
        r._2d_cache = None
        r._ll_cache = {}
        return r

    def test_groups_by_ug1(self):
        r = self._make_renderer()
        pts = _make_points(n_ug1=3, n_ua=5)
        cache = r._ensure_2d_cache(pts, track_sids=set(),
                                   series_labels={})
        self.assertEqual(len(cache["ug1_values"]), 3)
        self.assertEqual(len(cache["by_ug1"]), 3)
        for ug1, grouped in cache["by_ug1"].items():
            self.assertEqual(len(grouped), 5)
            uas = [p["ua"] for p in grouped]
            self.assertEqual(uas, sorted(uas))

    def test_groups_by_ug1_ug2(self):
        r = self._make_renderer()
        pts = _make_points(n_ug1=2, n_ua=4, ug2=200)
        pts += [dict(p, ug2=300) for p in _make_points(n_ug1=2, n_ua=4)]
        cache = r._ensure_2d_cache(pts, track_sids=set(),
                                   series_labels={})
        self.assertEqual(len(cache["ug2_values"]), 2)
        self.assertGreater(len(cache["by_ug1_ug2"]), 0)

    def test_cache_reuse(self):
        r = self._make_renderer()
        pts = _make_points()
        c1 = r._ensure_2d_cache(pts, set(), {})
        c2 = r._ensure_2d_cache(pts, set(), {})
        self.assertIs(c1, c2)

    def test_invalidate_clears(self):
        r = self._make_renderer()
        pts = _make_points()
        r._ensure_2d_cache(pts, set(), {})
        self.assertIsNotNone(r._2d_cache)
        r.invalidate_cache()
        self.assertIsNone(r._2d_cache)

    def test_new_data_recomputes(self):
        r = self._make_renderer()
        pts = _make_points(n_ug1=2, n_ua=3)
        c1 = r._ensure_2d_cache(pts, set(), {})
        pts.append({"ua": 999, "ia": 1, "ug1": -5, "ug2": 250,
                     "ig2": 0, "uh": 6.3, "ih": 0.3})
        c2 = r._ensure_2d_cache(pts, set(), {})
        self.assertIsNot(c1, c2)

    def test_cache_keyed_by_identity_not_id(self):
        """Two distinct same-length lists must not share a cache even when id()
        collides (a freed list's address can be reused). The cache is keyed by
        object identity (`is`), not id()."""
        from unittest import mock
        r = self._make_renderer()
        a = _make_points(n_ug1=3, n_ua=5, ug2=250.0)
        b = _make_points(n_ug1=3, n_ua=5, ug2=999.0)  # same length, diff Ug2
        # Force the id() collision the bug relies on; the fixed code uses `is`
        # (no id() call), so this patch is a no-op for it but makes the old
        # id()+len path deterministically serve A's stale cache for B.
        with mock.patch("app.plotting._plot_2d_mixin.id",
                        return_value=12345, create=True):
            ca = r._ensure_2d_cache(a, set(), {})
            cb = r._ensure_2d_cache(b, set(), {})
        self.assertIsNot(ca, cb)
        self.assertIn(999.0, cb["ug2_values"])
        self.assertNotIn(250.0, cb["ug2_values"])

    def test_compare_pts_split(self):
        r = self._make_renderer()
        pts = _make_points(n_ug1=2, n_ua=3, series_id=0)
        compare = _make_points(n_ug1=1, n_ua=3, series_id=1)
        all_pts = pts + compare
        cache = r._ensure_2d_cache(all_pts, set(), {})
        self.assertEqual(len(cache["compare_pts"]), 3)
        self.assertEqual(len(cache["current_pts"]), 6)
        self.assertIn(1, cache["compare_by_sid"])


if __name__ == "__main__":
    unittest.main()
