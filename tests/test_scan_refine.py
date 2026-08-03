"""Tests for adaptive refine: down-sweep, grid helpers, interval detection, and integration."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from scan_test_helpers import (
    _make_mock_client, _get_param_default, _make_scan_settings,
    CalibrationData, ScanSettings, ScanRange,
    _build_down_sweep_ua, _closest_grid_idx,
    _find_refine_intervals, _find_refine_intervals_per_ug1,
    _build_refine_ua, _refine_curve_inline,
    run_scan,
)


class TestBuildDownSweepUa(unittest.TestCase):
    """Test _build_down_sweep_ua: soft bisection for coarse grids."""

    def _ua_list(self, result):
        """Extract Ua values from (ua, is_grid) tuples."""
        return [ua for ua, _ in result]

    def _grid_only(self, result):
        """Extract only grid-marked Ua values."""
        return [ua for ua, g in result if g]

    def _intermed_only(self, result):
        """Extract only intermediate (non-grid) Ua values."""
        return [ua for ua, g in result if not g]

    def test_no_bisection_small_step(self):
        """Step 10V < max_step 25V → no intermediate points, all grid."""
        grid_desc = [190, 180, 170, 160]
        result = _build_down_sweep_ua(grid_desc, 200.0, 25.0)
        self.assertEqual(self._ua_list(result), [190, 180, 170, 160])
        self.assertTrue(all(g for _, g in result))

    def test_bisection_50v_step(self):
        """Step 50V > max_step 25V → bisect; intermediates marked non-grid."""
        grid_desc = [150, 100, 50]
        result = _build_down_sweep_ua(grid_desc, 200.0, 25.0)
        ua_vals = self._ua_list(result)
        self.assertIn(175.0, ua_vals)
        self.assertIn(150.0, ua_vals)
        self.assertIn(125.0, ua_vals)
        self.assertIn(100.0, ua_vals)
        self.assertEqual(ua_vals[-1], 50)
        for i in range(1, len(ua_vals)):
            self.assertGreater(ua_vals[i - 1], ua_vals[i], "Must be descending")
        self.assertEqual(self._grid_only(result), [150, 100, 50])
        self.assertTrue(len(self._intermed_only(result)) > 0)

    def test_bisection_100v_step(self):
        """Step 100V → split into 4 parts; only last is grid."""
        grid_desc = [100]
        result = _build_down_sweep_ua(grid_desc, 200.0, 25.0)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0][0], 175.0, places=0)
        self.assertAlmostEqual(result[-1][0], 100.0, places=0)
        self.assertTrue(result[-1][1])
        self.assertFalse(result[0][1])

    def test_disabled_zero_max(self):
        """max_step=0 → no bisection, all grid."""
        grid_desc = [100, 0]
        result = _build_down_sweep_ua(grid_desc, 200.0, 0)
        self.assertEqual(self._ua_list(result), [100, 0])
        self.assertTrue(all(g for _, g in result))

    def test_exact_max_step_no_bisection(self):
        """Step exactly equals max_step → no bisection."""
        grid_desc = [175]
        result = _build_down_sweep_ua(grid_desc, 200.0, 25.0)
        self.assertEqual(self._ua_list(result), [175])
        self.assertTrue(result[0][1])

    def test_mixed_steps(self):
        """First step large (50V), rest normal (10V)."""
        grid_desc = [150, 140, 130]
        result = _build_down_sweep_ua(grid_desc, 200.0, 25.0)
        ua_vals = self._ua_list(result)
        self.assertIn(175.0, ua_vals)
        self.assertIn(150.0, ua_vals)
        self.assertIn(140.0, ua_vals)
        self.assertIn(130.0, ua_vals)
        self.assertEqual(ua_vals[-1], 130)
        self.assertEqual(self._grid_only(result), [150, 140, 130])

    def test_empty_grid(self):
        result = _build_down_sweep_ua([], 200.0, 25.0)
        self.assertEqual(result, [])


# ======================================================================
# Adaptive refine — analysis & grid construction
# ======================================================================

class TestClosestGridIdx(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(_closest_grid_idx(50, [0, 50, 100]), 1)

    def test_between(self):
        self.assertEqual(_closest_grid_idx(47, [0, 50, 100]), 1)

    def test_at_start(self):
        self.assertEqual(_closest_grid_idx(2, [0, 50, 100]), 0)

    def test_at_end(self):
        self.assertEqual(_closest_grid_idx(99, [0, 50, 100]), 2)


class TestFindRefineIntervals(unittest.TestCase):
    """Tests for _find_refine_intervals — coarse data analysis."""

    def _settings(self, **kw):
        defaults = dict(
            ua=ScanRange(0, 100, 10), ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(250, 250, 0), uh=6.3, ih=0.0,
            refine_enabled=True, refine_onset_ma=0.5,
            refine_curvature_thr=0.15, refine_gradient_ratio=3.0,
            refine_ig2_delta_min=0.5, refine_delta_ia_thr=0.25,
        )
        defaults.update(kw)
        return ScanSettings(**defaults)

    def test_flat_curve_no_refine(self):
        """All Ia identical → nothing triggers."""
        ua_vals = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        pts = [{"ua": ua, "ug1": 0.0, "ia": 5.0, "ig2": 0.5} for ua in ua_vals]
        intervals = _find_refine_intervals(pts, ua_vals, self._settings())
        self.assertEqual(intervals, set())

    def test_c1_onset_detected(self):
        """C1: current starts flowing → interval marked."""
        ua_vals = [0, 10, 20, 30, 40]
        pts = [
            {"ua": 0, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 10, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 20, "ug1": 0.0, "ia": 0.1, "ig2": 0.0},
            {"ua": 30, "ug1": 0.0, "ia": 2.0, "ig2": 0.0},
            {"ua": 40, "ug1": 0.0, "ia": 5.0, "ig2": 0.0},
        ]
        intervals = _find_refine_intervals(pts, ua_vals, self._settings())
        self.assertIn(2, intervals)

    def test_c2_curvature_detected(self):
        """C2: pentode knee → curvature triggers refinement."""
        ua_vals = [0, 10, 20, 30, 40, 50, 60]
        pts = [
            {"ua": 0, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 10, "ug1": 0.0, "ia": 0.5, "ig2": 0.0},
            {"ua": 20, "ug1": 0.0, "ia": 1.0, "ig2": 0.0},
            {"ua": 30, "ug1": 0.0, "ia": 8.0, "ig2": 0.0},  # knee
            {"ua": 40, "ug1": 0.0, "ia": 9.0, "ig2": 0.0},
            {"ua": 50, "ug1": 0.0, "ia": 9.5, "ig2": 0.0},
            {"ua": 60, "ug1": 0.0, "ia": 10.0, "ig2": 0.0},
        ]
        intervals = _find_refine_intervals(pts, ua_vals, self._settings())
        self.assertTrue(intervals & {1, 2, 3})

    def test_c3_gradient_ratio(self):
        """C3: slope changes drastically."""
        ua_vals = [0, 10, 20, 30, 40]
        pts = [
            {"ua": 0, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 10, "ug1": 0.0, "ia": 0.1, "ig2": 0.0},
            {"ua": 20, "ug1": 0.0, "ia": 0.2, "ig2": 0.0},
            {"ua": 30, "ug1": 0.0, "ia": 5.0, "ig2": 0.0},
            {"ua": 40, "ug1": 0.0, "ia": 6.0, "ig2": 0.0},
        ]
        intervals = _find_refine_intervals(
            pts, ua_vals,
            self._settings(refine_curvature_thr=100.0))  # disable C2
        self.assertTrue(intervals & {1, 2})

    def test_c4_ig2_kink_pentode(self):
        """C4: Ig2 non-monotonicity in pentode mode → kink detected."""
        ua_vals = [0, 10, 20, 30, 40]
        pts = [
            {"ua": 0, "ug1": 0.0, "ia": 5.0, "ig2": 10.0},
            {"ua": 10, "ug1": 0.0, "ia": 5.5, "ig2": 8.0},
            {"ua": 20, "ug1": 0.0, "ia": 4.0, "ig2": 12.0},  # kink
            {"ua": 30, "ug1": 0.0, "ia": 6.0, "ig2": 5.0},
            {"ua": 40, "ug1": 0.0, "ia": 6.5, "ig2": 3.0},
        ]
        s = self._settings(
            is_triode=False, refine_curvature_thr=100.0,
            refine_gradient_ratio=100.0, refine_delta_ia_thr=100.0)
        intervals = _find_refine_intervals(pts, ua_vals, s)
        self.assertTrue(intervals & {1, 2})

    def test_c4_ignored_for_triode(self):
        """C4: Ig2 criterion not applied in triode mode."""
        ua_vals = [0, 10, 20, 30, 40]
        pts = [
            {"ua": 0, "ug1": 0.0, "ia": 5.0, "ig2": 10.0},
            {"ua": 10, "ug1": 0.0, "ia": 5.1, "ig2": 8.0},
            {"ua": 20, "ug1": 0.0, "ia": 5.2, "ig2": 12.0},
            {"ua": 30, "ug1": 0.0, "ia": 5.3, "ig2": 5.0},
            {"ua": 40, "ug1": 0.0, "ia": 5.4, "ig2": 3.0},
        ]
        s = self._settings(
            is_triode=True, refine_curvature_thr=100.0,
            refine_gradient_ratio=100.0, refine_delta_ia_thr=100.0,
            refine_onset_ma=100.0)
        intervals = _find_refine_intervals(pts, ua_vals, s)
        self.assertEqual(intervals, set())

    def test_c5_large_ia_jump(self):
        """C5: big delta-Ia between two points."""
        ua_vals = [0, 10, 20, 30, 40]
        pts = [
            {"ua": 0, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 10, "ug1": 0.0, "ia": 1.0, "ig2": 0.0},
            {"ua": 20, "ug1": 0.0, "ia": 2.0, "ig2": 0.0},
            {"ua": 30, "ug1": 0.0, "ia": 10.0, "ig2": 0.0},
            {"ua": 40, "ug1": 0.0, "ia": 10.0, "ig2": 0.0},
        ]
        s = self._settings(
            refine_curvature_thr=100.0, refine_gradient_ratio=100.0,
            refine_onset_ma=100.0)
        intervals = _find_refine_intervals(pts, ua_vals, s)
        self.assertIn(2, intervals)

    def test_union_across_ug1(self):
        """Multiple Ug1 curves — union of triggered intervals."""
        ua_vals = [0, 10, 20, 30, 40]
        pts = [
            {"ua": 0, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 10, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 20, "ug1": 0.0, "ia": 5.0, "ig2": 0.0},
            {"ua": 30, "ug1": 0.0, "ia": 8.0, "ig2": 0.0},
            {"ua": 40, "ug1": 0.0, "ia": 10.0, "ig2": 0.0},
            {"ua": 0, "ug1": -2.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 10, "ug1": -2.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 20, "ug1": -2.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 30, "ug1": -2.0, "ia": 3.0, "ig2": 0.0},
            {"ua": 40, "ug1": -2.0, "ia": 7.0, "ig2": 0.0},
        ]
        s = self._settings(
            refine_curvature_thr=100.0, refine_gradient_ratio=100.0,
            refine_delta_ia_thr=100.0)
        intervals = _find_refine_intervals(pts, ua_vals, s)
        self.assertIn(1, intervals)
        self.assertIn(2, intervals)

    def test_empty_points(self):
        self.assertEqual(
            _find_refine_intervals([], [0, 10, 20], self._settings()), set())

    def test_too_few_ua_values(self):
        pts = [{"ua": 0, "ug1": 0.0, "ia": 5.0, "ig2": 0.0}]
        self.assertEqual(
            _find_refine_intervals(pts, [0, 10], self._settings()), set())


class TestFindRefineIntervalsPerUg1(unittest.TestCase):
    """Tests for _find_refine_intervals_per_ug1 — per-curve analysis."""

    def _settings(self, **kw):
        defaults = dict(
            ua=ScanRange(0, 100, 10), ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(250, 250, 0), uh=6.3, ih=0.0,
            refine_enabled=True, refine_onset_ma=0.5,
            refine_curvature_thr=0.15, refine_gradient_ratio=3.0,
            refine_ig2_delta_min=0.5, refine_delta_ia_thr=0.25,
        )
        defaults.update(kw)
        return ScanSettings(**defaults)

    def test_per_ug1_separates_intervals(self):
        """Different Ug1 curves get separate interval sets, not a union."""
        ua_vals = [0, 20, 40, 60, 80, 100]
        pts = [
            {"ua": 0, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 20, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 40, "ug1": 0.0, "ia": 5.0, "ig2": 0.0},
            {"ua": 60, "ug1": 0.0, "ia": 5.3, "ig2": 0.0},
            {"ua": 80, "ug1": 0.0, "ia": 5.6, "ig2": 0.0},
            {"ua": 100, "ug1": 0.0, "ia": 6.0, "ig2": 0.0},
            {"ua": 0, "ug1": -4.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 20, "ug1": -4.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 40, "ug1": -4.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 60, "ug1": -4.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 80, "ug1": -4.0, "ia": 3.0, "ig2": 0.0},
            {"ua": 100, "ug1": -4.0, "ia": 3.3, "ig2": 0.0},
        ]
        s = self._settings(
            refine_curvature_thr=100.0, refine_gradient_ratio=100.0,
            refine_delta_ia_thr=100.0, refine_onset_ma=0.5)
        per_ug1 = _find_refine_intervals_per_ug1(pts, ua_vals, s)
        self.assertIn(1, per_ug1.get(0.0, set()))
        self.assertNotIn(3, per_ug1.get(0.0, set()))
        self.assertIn(3, per_ug1.get(-4.0, set()))
        self.assertNotIn(1, per_ug1.get(-4.0, set()))

    def test_union_matches_old_behaviour(self):
        """Union of all per-Ug1 intervals equals _find_refine_intervals result."""
        ua_vals = [0, 10, 20, 30, 40]
        pts = [
            {"ua": 0, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 10, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 20, "ug1": 0.0, "ia": 5.0, "ig2": 0.0},
            {"ua": 30, "ug1": 0.0, "ia": 8.0, "ig2": 0.0},
            {"ua": 40, "ug1": 0.0, "ia": 10.0, "ig2": 0.0},
            {"ua": 0, "ug1": -2.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 10, "ug1": -2.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 20, "ug1": -2.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 30, "ug1": -2.0, "ia": 3.0, "ig2": 0.0},
            {"ua": 40, "ug1": -2.0, "ia": 7.0, "ig2": 0.0},
        ]
        s = self._settings(
            refine_curvature_thr=100.0, refine_gradient_ratio=100.0,
            refine_delta_ia_thr=100.0)
        per_ug1 = _find_refine_intervals_per_ug1(pts, ua_vals, s)
        union = set()
        for v in per_ug1.values():
            union |= v
        old_result = _find_refine_intervals(pts, ua_vals, s)
        self.assertEqual(union, old_result)

    def test_empty_inputs(self):
        self.assertEqual(
            _find_refine_intervals_per_ug1([], [0, 10, 20], self._settings()), {})
        pts = [{"ua": 0, "ug1": 0.0, "ia": 5.0, "ig2": 0.0}]
        self.assertEqual(
            _find_refine_intervals_per_ug1(pts, [0, 10], self._settings()), {})

    def test_flat_curve_excluded(self):
        """Ug1 curves with no triggers don't appear in result."""
        ua_vals = [0, 10, 20, 30, 40]
        pts = [{"ua": ua, "ug1": 0.0, "ia": 5.0, "ig2": 0.0} for ua in ua_vals]
        per_ug1 = _find_refine_intervals_per_ug1(pts, ua_vals, self._settings())
        self.assertEqual(per_ug1, {})


class TestBuildRefineUa(unittest.TestCase):
    """Tests for _build_refine_ua — recursive bisection."""

    def test_single_bisection(self):
        ua = [0, 10, 20, 30, 40]
        result = _build_refine_ua(ua, {2}, min_step=3.0, max_depth=1)
        self.assertEqual(result, [25])

    def test_double_bisection(self):
        ua = [0, 20, 40, 60]
        result = _build_refine_ua(ua, {1}, min_step=3.0, max_depth=2)
        self.assertEqual(result, [25, 30, 35])

    def test_min_step_stops_bisection(self):
        ua = [0, 10, 20]
        result = _build_refine_ua(ua, {1}, min_step=4.0, max_depth=3)
        self.assertEqual(result, [15])

    def test_min_step_prevents_any_bisection(self):
        ua = [0, 5, 10]
        result = _build_refine_ua(ua, {1}, min_step=3.0, max_depth=2)
        self.assertEqual(result, [])

    def test_rounding_prevents_duplicate(self):
        ua = [0, 1, 2]
        result = _build_refine_ua(ua, {0}, min_step=0.1, max_depth=1)
        self.assertEqual(result, [])

    def test_multiple_intervals(self):
        ua = [0, 10, 20, 30, 40, 50]
        result = _build_refine_ua(ua, {1, 3}, min_step=3.0, max_depth=1)
        self.assertEqual(result, [15, 35])

    def test_empty_intervals(self):
        self.assertEqual(_build_refine_ua([0, 10], set(), 3.0, 2), [])

    def test_max_depth_zero(self):
        self.assertEqual(_build_refine_ua([0, 10], {0}, 1.0, 0), [])


class TestRunScanRefine(unittest.TestCase):
    """Integration tests: refine pass in run_scan."""

    def _make_settings(self, **kwargs):
        defaults = dict(
            ua=ScanRange(0, 100, 20),   # [0, 20, 40, 60, 80, 100]
            ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(250, 250, 0),
            uh=6.3, ih=0.0,
            is_triode=False,
            ua_settle_per_volt_s=0.0, ua_settle_base_s=0.0,
            ua_tolerance=1.0, ua_retries=1,
            ug1_settle_per_volt_s=0.0, ug1_settle_base_s=0.0,
            ug1_tolerance=0.1, ug1_retries=1,
            ug2_settle_per_volt_s=0.0, ug2_settle_base_s=0.0,
            ug2_tolerance=1.0, ug2_retries=1,
            ia_samples=1, calibration=CalibrationData(),
        )
        defaults.update(kwargs)
        return ScanSettings(**defaults)

    @patch("time.sleep")
    def test_refine_disabled_no_extra(self, _):
        """refine_enabled=False → only coarse points."""
        client = _make_mock_client()
        client.get_param.side_effect = _get_param_default(100)
        settings = self._make_settings(refine_enabled=False)
        points = run_scan(client, settings)
        self.assertEqual(len(points), 6)

    @patch("time.sleep")
    def test_refine_pentode_adds_points(self, _):
        """Pentode scan with refine: knee triggers extra midpoints."""
        client = _make_mock_client()

        def get_param_side(name, real=False):
            last_ua = 0
            if client.set_param.called:
                for c in reversed(client.set_param.call_args_list):
                    if c[0][0] == "Ua":
                        last_ua = c[0][1]
                        break
            if name == "Ia" and real:
                if last_ua <= 20:
                    return 50
                elif last_ua <= 40:
                    return 5000
                else:
                    return 5500
            if name == "Ua" and real:
                return last_ua
            if name == "Ug2" and real:
                if client.set_param.called:
                    for c in reversed(client.set_param.call_args_list):
                        if c[0][0] == "Ug2":
                            return c[0][1]
                return 250
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        settings = self._make_settings(
            refine_enabled=True,
            refine_max_depth=1,
            refine_min_step_ua=3.0,
            refine_onset_ma=0.5,
            refine_curvature_thr=0.15,
            refine_gradient_ratio=3.0,
            refine_ig2_delta_min=0.5,
            refine_delta_ia_thr=0.25,
        )
        points = run_scan(client, settings)
        self.assertGreater(len(points), 6)
        ua_list = [p["ua"] for p in points]
        self.assertEqual(len(ua_list), len(set(ua_list)))

    @patch("time.sleep")
    def test_refine_triode_mode(self, _):
        """Triode scan with refine works (single Ug2 group)."""
        client = _make_mock_client()

        def get_param_side(name, real=False):
            last_ua = 0
            if client.set_param.called:
                for c in reversed(client.set_param.call_args_list):
                    if c[0][0] == "Ua":
                        last_ua = c[0][1]
                        break
            if name == "Uh" and real:
                return 630
            if name == "Ih" and real:
                return 30
            if name == "Ia" and real:
                return 0 if last_ua < 40 else 3000
            if name == "Ua" and real:
                return last_ua
            return 0

        client.get_param.side_effect = get_param_side
        settings = self._make_settings(
            is_triode=True,
            refine_enabled=True,
            refine_max_depth=1,
            refine_min_step_ua=3.0,
        )
        points = run_scan(client, settings)
        self.assertGreater(len(points), 6)

    @patch("time.sleep")
    def test_refine_track_per_ug1_not_full_range(self, _):
        """Track mode refine: per-Ug1 intervals, NOT full-range refine."""
        client = _make_mock_client()

        def get_param_side(name, real=False):
            last_ua = 0
            last_ug1_raw = 0
            if client.set_param.called:
                for c in reversed(client.set_param.call_args_list):
                    if c[0][0] == "Ua":
                        last_ua = c[0][1]
                        break
                for c in reversed(client.set_param.call_args_list):
                    if c[0][0] == "Ug1":
                        last_ug1_raw = c[0][1]
                        break
            ug1_v = -(last_ug1_raw / 10.0)
            if name == "Uh" and real:
                return 630
            if name == "Ih" and real:
                return 30
            if name == "Ia" and real:
                onset_ua = 20 + abs(ug1_v) * 20
                if last_ua < onset_ua:
                    return 0
                else:
                    return 5000
            if name == "Ua" and real:
                return last_ua
            if name == "Ug1" and real:
                return last_ug1_raw
            if name == "Ug2" and real:
                if client.set_param.called:
                    for c in reversed(client.set_param.call_args_list):
                        if c[0][0] == "Ug2":
                            return c[0][1]
                return 0
            return 0

        client.get_param.side_effect = get_param_side
        settings = self._make_settings(
            ua=ScanRange(0, 120, 20),
            ug1=ScanRange(0, -4, 2),
            ug2=ScanRange(0, 0, 0),
            ug2_track_ua=True,
            ug2_offset=0.0,
            refine_enabled=True,
            refine_max_depth=1,
            refine_min_step_ua=3.0,
        )
        coarse_count = 7 * 3
        points = run_scan(client, settings)
        refine_count = len(points) - coarse_count
        self.assertGreater(refine_count, 0, "Refine should add some points")
        self.assertLessEqual(refine_count, 10,
                             f"Per-Ug1 refine should be targeted, got {refine_count} "
                             f"refine points (expected ≤10, full-range would be ~18+)")

    @patch("time.sleep")
    def test_refine_flat_no_extra(self, _):
        """Completely flat curve → no refine intervals → no extra points."""
        client = _make_mock_client()
        client.get_param.side_effect = _get_param_default(500)
        settings = self._make_settings(refine_enabled=True, refine_max_depth=2)
        points = run_scan(client, settings)
        self.assertEqual(len(points), 6)

    @patch("time.sleep")
    def test_refine_sorted_output(self, _):
        """After refine, points are sorted by (ug2, ug1, ua)."""
        client = _make_mock_client()

        def get_param_side(name, real=False):
            last_ua = 0
            if client.set_param.called:
                for c in reversed(client.set_param.call_args_list):
                    if c[0][0] == "Ua":
                        last_ua = c[0][1]
                        break
            if name == "Ia" and real:
                return 0 if last_ua < 40 else 5000
            if name == "Ua" and real:
                return last_ua
            if name == "Ug2" and real:
                if client.set_param.called:
                    for c in reversed(client.set_param.call_args_list):
                        if c[0][0] == "Ug2":
                            return c[0][1]
                return 250
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        settings = self._make_settings(
            refine_enabled=True, refine_max_depth=1, refine_min_step_ua=3.0)
        points = run_scan(client, settings)
        self.assertGreater(len(points), 6)
        ua_list = [p["ua"] for p in points]
        self.assertEqual(ua_list, sorted(ua_list))

    @patch("time.sleep")
    def test_refine_respects_pg2_boundary(self, _):
        """Refine in independent Ug2 sweep must NOT add points below Pg2 safe boundary."""
        client = _make_mock_client()

        def get_param_side(name, real=False):
            last_ua = 0
            if client.set_param.called:
                for c in reversed(client.set_param.call_args_list):
                    if c[0][0] == "Ua":
                        last_ua = c[0][1]
                        break
            if name == "Ig2" and real:
                return 2000 if last_ua < 40 else 100
            if name == "Ia" and real:
                if last_ua < 40:
                    return 10
                elif last_ua <= 60:
                    return 5000
                else:
                    return 5500
            if name == "Ua" and real:
                return last_ua
            if name == "Ug2" and real:
                if client.set_param.called:
                    for c in reversed(client.set_param.call_args_list):
                        if c[0][0] == "Ug2":
                            return c[0][1]
                return 250
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        settings = self._make_settings(
            ua=ScanRange(0, 100, 20),
            ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(250, 250, 0),
            pig2_max_w=2.5,
            pig2_over_pct=20.0,
            refine_enabled=True,
            refine_max_depth=1,
            refine_min_step_ua=3.0,
            refine_onset_ma=0.5,
            refine_curvature_thr=0.15,
            refine_gradient_ratio=3.0,
            refine_ig2_delta_min=0.5,
            refine_delta_ia_thr=0.25,
        )
        points = run_scan(client, settings)
        for p in points:
            self.assertGreaterEqual(p["ua"], 40,
                                    f"Refine point at Ua={p['ua']} is below Pg2 safe boundary")

    @patch("time.sleep")
    def test_refine_track_mode_pg2_skip(self, _):
        """Track mode refine: Pg2-exceeding points skipped via continue."""
        client = _make_mock_client()

        def get_param_side(name, real=False):
            last_ua = 0
            if client.set_param.called:
                for c in reversed(client.set_param.call_args_list):
                    if c[0][0] == "Ua":
                        last_ua = c[0][1]
                        break
            if name == "Ia" and real:
                return 0 if last_ua < 30 else 3000
            if name == "Ig2" and real:
                if 25 <= last_ua <= 35:
                    return 1500
                return 50
            if name == "Ua" and real:
                return last_ua
            if name == "Ug2" and real:
                if client.set_param.called:
                    for c in reversed(client.set_param.call_args_list):
                        if c[0][0] == "Ug2":
                            return c[0][1]
                return 200
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        settings = self._make_settings(
            ua=ScanRange(0, 100, 20),
            ug2_track_ua=True,
            ug2_offset=200.0,
            pig2_max_w=2.5,
            pig2_over_pct=20.0,
            refine_enabled=True,
            refine_max_depth=1,
            refine_min_step_ua=3.0,
        )
        points = run_scan(client, settings)
        for p in points:
            pg2_w = p["ug2"] * p["ig2"] / 1000.0
            if pg2_w > 3.0:
                self.fail(f"Pg2-exceeding point was NOT skipped: Ua={p['ua']}, Pg2={pg2_w:.2f}W")

    @patch("time.sleep")
    def test_refine_independent_mode_pg2_skip(self, _):
        """ML-127: independent-mode refine must apply the Pg2 POWER check
        too — a refine midpoint can sit inside the Ig2 current limit yet
        over the Pg2 limit at high fixed Ug2. The old gate ran the Pg2
        check only when ug2_track_ua was set."""
        client = _make_mock_client()

        def get_param_side(name, real=False):
            last_ua = 0
            if client.set_param.called:
                for c in reversed(client.set_param.call_args_list):
                    if c[0][0] == "Ua":
                        last_ua = c[0][1]
                        break
            if name == "Ia" and real:
                return 10 if last_ua < 30 else 3000
            if name == "Ig2" and real:
                if 25 <= last_ua <= 35:
                    return 1500          # 15 mA @ Ug2=300 → Pg2 = 4.5 W
                return 50                # 0.5 mA → 0.15 W
            if name == "Ua" and real:
                return last_ua
            if name == "Ug2" and real:
                return 300
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        settings = self._make_settings(
            ua=ScanRange(0, 100, 20),
            ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(300, 300, 0),
            pig2_max_w=3.0,
            pig2_over_pct=0.0,           # exact 3.0 W limit
            refine_enabled=True,
            refine_max_depth=1,
            refine_min_step_ua=3.0,
        )
        points = run_scan(client, settings)
        self.assertTrue(points, "scan produced no points — test is vacuous")
        for p in points:
            pg2_w = p["ug2"] * p["ig2"] / 1000.0
            self.assertLessEqual(
                pg2_w, 3.0,
                f"Pg2-exceeding refine point not skipped: Ua={p['ua']}, "
                f"Pg2={pg2_w:.2f}W")

    @patch("time.sleep")
    def test_refine_pentode_multiple_ug2_respects_pg2(self, _):
        """Refine with multiple Ug2 levels: each respects its own Pg2 boundary."""
        client = _make_mock_client()

        def get_param_side(name, real=False):
            last_ua = 0
            last_ug2 = 200
            if client.set_param.called:
                for c in reversed(client.set_param.call_args_list):
                    if c[0][0] == "Ua":
                        last_ua = c[0][1]
                        break
                for c in reversed(client.set_param.call_args_list):
                    if c[0][0] == "Ug2":
                        last_ug2 = c[0][1]
                        break
            if name == "Ig2" and real:
                if last_ug2 >= 300:
                    return 2000 if last_ua < 60 else 100
                else:
                    return 2000 if last_ua < 20 else 100
            if name == "Ia" and real:
                return 10 if last_ua < 40 else 5000
            if name == "Ua" and real:
                return last_ua
            if name == "Ug2" and real:
                return last_ug2
            if name == "Ug1" and real:
                if client.set_param.called:
                    for c in reversed(client.set_param.call_args_list):
                        if c[0][0] == "Ug1":
                            return c[0][1]
                return 0
            return 0

        client.get_param.side_effect = get_param_side
        settings = self._make_settings(
            ua=ScanRange(0, 100, 20),
            ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(200, 300, 100),
            pig2_max_w=2.5,
            pig2_over_pct=20.0,
            refine_enabled=True,
            refine_max_depth=1,
            refine_min_step_ua=3.0,
        )
        points = run_scan(client, settings)
        pts_200 = [p for p in points if abs(p["ug2"] - 200) <= 2]
        for p in pts_200:
            self.assertGreaterEqual(p["ua"], 20,
                                    f"Ug2=200 refine below boundary: Ua={p['ua']}")
        pts_300 = [p for p in points if abs(p["ug2"] - 300) <= 2]
        for p in pts_300:
            self.assertGreaterEqual(p["ua"], 60,
                                    f"Ug2=300 refine below boundary: Ua={p['ua']}")

    @patch("time.sleep")
    def test_refine_pa_limit_breaks_refine_loop(self, _):
        """Pa limit should break the refine inner loop just like the coarse pass."""
        client = _make_mock_client()

        def get_param_side(name, real=False):
            last_ua = 0
            if client.set_param.called:
                for c in reversed(client.set_param.call_args_list):
                    if c[0][0] == "Ua":
                        last_ua = c[0][1]
                        break
            if name == "Ia" and real:
                if last_ua < 40:
                    return 0
                elif last_ua < 50:
                    return 3000
                else:
                    return 50000
            if name == "Ua" and real:
                return last_ua
            if name == "Ug2" and real:
                if client.set_param.called:
                    for c in reversed(client.set_param.call_args_list):
                        if c[0][0] == "Ug2":
                            return c[0][1]
                return 250
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        settings = self._make_settings(
            ua=ScanRange(0, 100, 20),
            ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(250, 250, 0),
            pa_max_w=3.0,
            pa_over_pct=20.0,
            refine_enabled=True,
            refine_max_depth=1,
            refine_min_step_ua=3.0,
        )
        points = run_scan(client, settings)
        for p in points:
            if p["ua"] >= 60:
                later_points = [pp for pp in points
                                if pp["ua"] > p["ua"] and abs(pp["ug2"] - p["ug2"]) < 2]
                self.assertEqual(len(later_points), 0,
                                 f"Points found after Pa break at Ua={p['ua']}")

    @patch("time.sleep")
    def test_refine_pg2_all_unsafe_no_crash(self, _):
        """When entire sweep is Pg2-unsafe, refine should not crash (IndexError)."""
        client = _make_mock_client()
        _state = {}

        def capture_set(name, value, delay=0.05):
            _state[name] = value

        client.set_param = MagicMock(side_effect=capture_set)

        def get_param_side(name, real=False):
            if name == "Ig2" and real:
                return 5000
            if name == "Ia" and real:
                return 0 if name == "Er" else 100
            if name == "Ug2" and real:
                return 250
            if name == "Ug1" and real:
                return 0
            return _state.get(name, 0)

        client.get_param.side_effect = get_param_side
        settings = self._make_settings(
            ua=ScanRange(0, 100, 20),
            ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(250, 250, 0),
            pig2_max_w=0.5,
            pig2_over_pct=20.0,
            refine_enabled=True,
            refine_max_depth=1,
            refine_min_step_ua=3.0,
        )
        points = run_scan(client, settings)
        self.assertEqual(len(points), 0)


class TestRefineCountEvent(unittest.TestCase):
    """Verify that inline refine emits refine_count events via progress."""

    def _make_settings(self, **kwargs):
        defaults = dict(
            ua=ScanRange(0, 100, 20),
            ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(250, 250, 0),
            uh=6.3, ih=0.0,
            is_triode=False,
            ua_settle_per_volt_s=0.0, ua_settle_base_s=0.0,
            ua_tolerance=1.0, ua_retries=1,
            ug1_settle_per_volt_s=0.0, ug1_settle_base_s=0.0,
            ug1_tolerance=0.1, ug1_retries=1,
            ug2_settle_per_volt_s=0.0, ug2_settle_base_s=0.0,
            ug2_tolerance=1.0, ug2_retries=1,
            ia_samples=1, calibration=CalibrationData(),
        )
        defaults.update(kwargs)
        return ScanSettings(**defaults)

    @patch("time.sleep")
    def test_refine_count_event_emitted(self, _):
        """When refine adds points, a refine_count event with positive count is emitted."""
        client = _make_mock_client()

        def get_param_side(name, real=False):
            last_ua = 0
            if client.set_param.called:
                for c in reversed(client.set_param.call_args_list):
                    if c[0][0] == "Ua":
                        last_ua = c[0][1]
                        break
            if name == "Ia" and real:
                return 0 if last_ua < 40 else 5000
            if name == "Ua" and real:
                return last_ua
            if name == "Ug2" and real:
                if client.set_param.called:
                    for c in reversed(client.set_param.call_args_list):
                        if c[0][0] == "Ug2":
                            return c[0][1]
                return 250
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        events = []
        settings = self._make_settings(
            refine_enabled=True, refine_max_depth=1, refine_min_step_ua=3.0)
        run_scan(client, settings, progress=lambda e: events.append(e))
        refine_events = [e for e in events if isinstance(e, dict)
                         and e.get("event") == "refine_count"]
        self.assertGreater(len(refine_events), 0,
                           "At least one refine_count event expected")
        for ev in refine_events:
            self.assertIn("count", ev)
            self.assertGreater(ev["count"], 0)

    @patch("time.sleep")
    def test_no_refine_count_when_disabled(self, _):
        """With refine_enabled=False, no refine_count events are emitted."""
        client = _make_mock_client()
        client.get_param.side_effect = _get_param_default(100)
        events = []
        settings = self._make_settings(refine_enabled=False)
        run_scan(client, settings, progress=lambda e: events.append(e))
        refine_events = [e for e in events if isinstance(e, dict)
                         and e.get("event") == "refine_count"]
        self.assertEqual(len(refine_events), 0)

    @patch("time.sleep")
    def test_no_old_refine_events(self, _):
        """refine_start and refine_done events must never appear."""
        client = _make_mock_client()

        def get_param_side(name, real=False):
            last_ua = 0
            if client.set_param.called:
                for c in reversed(client.set_param.call_args_list):
                    if c[0][0] == "Ua":
                        last_ua = c[0][1]
                        break
            if name == "Ia" and real:
                return 0 if last_ua < 40 else 5000
            if name == "Ua" and real:
                return last_ua
            if name == "Ug2" and real:
                if client.set_param.called:
                    for c in reversed(client.set_param.call_args_list):
                        if c[0][0] == "Ug2":
                            return c[0][1]
                return 250
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        events = []
        settings = self._make_settings(
            refine_enabled=True, refine_max_depth=1, refine_min_step_ua=3.0)
        run_scan(client, settings, progress=lambda e: events.append(e))
        old_events = [e for e in events if isinstance(e, dict)
                      and e.get("event") in ("refine_start", "refine_done")]
        self.assertEqual(len(old_events), 0,
                         f"Old refine events should not exist: {old_events}")

    @patch("time.sleep")
    def test_refine_count_sum_matches_extra_points(self, _):
        """Sum of all refine_count values should equal the number of extra points."""
        client = _make_mock_client()

        def get_param_side(name, real=False):
            last_ua = 0
            if client.set_param.called:
                for c in reversed(client.set_param.call_args_list):
                    if c[0][0] == "Ua":
                        last_ua = c[0][1]
                        break
            if name == "Ia" and real:
                return 0 if last_ua < 40 else 5000
            if name == "Ua" and real:
                return last_ua
            if name == "Ug2" and real:
                if client.set_param.called:
                    for c in reversed(client.set_param.call_args_list):
                        if c[0][0] == "Ug2":
                            return c[0][1]
                return 250
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        events = []
        settings = self._make_settings(
            refine_enabled=True, refine_max_depth=1, refine_min_step_ua=3.0)
        points = run_scan(client, settings, progress=lambda e: events.append(e))
        refine_total = sum(e["count"] for e in events
                          if isinstance(e, dict) and e.get("event") == "refine_count")
        coarse_count = 6
        extra_points = len(points) - coarse_count
        self.assertEqual(refine_total, extra_points,
                         f"refine_count sum ({refine_total}) != extra points ({extra_points})")


class TestRefineCurveInlineUnit(unittest.TestCase):
    """Unit tests for _refine_curve_inline helper."""

    def _settings(self, **kw):
        defaults = dict(
            ua=ScanRange(0, 100, 20), ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(250, 250, 0), uh=6.3, ih=0.0,
            is_triode=True,
            refine_enabled=True, refine_max_depth=1,
            refine_min_step_ua=3.0, refine_onset_ma=0.5,
            refine_curvature_thr=0.15, refine_gradient_ratio=3.0,
            refine_ig2_delta_min=0.5, refine_delta_ia_thr=0.25,
        )
        defaults.update(kw)
        return ScanSettings(**defaults)

    def _mock_read(self, ua_to_ia):
        """Return a read_point function that produces points based on Ua→Ia map."""
        self._last_ua = 0.0

        def settle_ua(ua):
            self._last_ua = ua
            return ua

        def read_point():
            ua = self._last_ua
            ia = ua_to_ia.get(ua, 0.0)
            return {"ua": ua, "ug1": 0.0, "ug2": 250.0, "ia": ia,
                    "ig2": 0.0, "uh": 6.3, "ih": 0.3}

        return settle_ua, read_point

    def test_refine_disabled_returns_empty(self):
        s = self._settings(refine_enabled=False)
        ua_vals = [0, 20, 40, 60, 80, 100]
        pts = [{"ua": u, "ug1": 0.0, "ia": 0 if u < 40 else 50.0, "ig2": 0.0}
               for u in ua_vals]
        settle_ua, read_point = self._mock_read({})
        result = _refine_curve_inline(pts, ua_vals, s, settle_ua, None,
                                      read_point, None, None, 0.0, 0.0)
        self.assertEqual(result, [])

    def test_too_few_points_returns_empty(self):
        s = self._settings()
        ua_vals = [0, 20, 40, 60, 80, 100]
        pts = [{"ua": 0, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
               {"ua": 20, "ug1": 0.0, "ia": 5.0, "ig2": 0.0}]
        settle_ua, read_point = self._mock_read({})
        result = _refine_curve_inline(pts, ua_vals, s, settle_ua, None,
                                      read_point, None, None, 0.0, 0.0)
        self.assertEqual(result, [])

    def test_flat_curve_returns_empty(self):
        s = self._settings()
        ua_vals = [0, 20, 40, 60, 80, 100]
        pts = [{"ua": u, "ug1": 0.0, "ia": 5.0, "ig2": 0.5} for u in ua_vals]
        settle_ua, read_point = self._mock_read({})
        result = _refine_curve_inline(pts, ua_vals, s, settle_ua, None,
                                      read_point, None, None, 0.0, 0.0)
        self.assertEqual(result, [])

    def test_onset_triggers_refine(self):
        """Curve with onset at Ua~40 should produce refine midpoints."""
        s = self._settings()
        ua_vals = [0, 20, 40, 60, 80, 100]
        pts = [
            {"ua": 0, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 20, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 40, "ug1": 0.0, "ia": 50.0, "ig2": 0.0},
            {"ua": 60, "ug1": 0.0, "ia": 55.0, "ig2": 0.0},
            {"ua": 80, "ug1": 0.0, "ia": 57.0, "ig2": 0.0},
            {"ua": 100, "ug1": 0.0, "ia": 58.0, "ig2": 0.0},
        ]
        ua_to_ia = {30: 25.0, 10: 0.0, 50: 52.0}
        settle_ua, read_point = self._mock_read(ua_to_ia)
        events = []
        result = _refine_curve_inline(pts, ua_vals, s, settle_ua, None,
                                      read_point, lambda e: events.append(e),
                                      None, 0.0, 0.0)
        self.assertGreater(len(result), 0)
        refine_ua_values = [p["ua"] for p in result]
        for ua in refine_ua_values:
            self.assertNotIn(ua, ua_vals, "Refine should not duplicate grid points")
        refine_events = [e for e in events if isinstance(e, dict)
                         and e.get("event") == "refine_count"]
        self.assertEqual(len(refine_events), 1)
        self.assertEqual(refine_events[0]["count"], len(result))

    def test_min_safe_ua_filters_refine_points(self):
        """Refine points below min_safe_ua should be excluded."""
        s = self._settings()
        ua_vals = [0, 20, 40, 60, 80, 100]
        pts = [
            {"ua": 0, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 20, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 40, "ug1": 0.0, "ia": 50.0, "ig2": 0.0},
            {"ua": 60, "ug1": 0.0, "ia": 55.0, "ig2": 0.0},
            {"ua": 80, "ug1": 0.0, "ia": 57.0, "ig2": 0.0},
            {"ua": 100, "ug1": 0.0, "ia": 58.0, "ig2": 0.0},
        ]
        ua_to_ia = {30: 25.0, 10: 0.0, 50: 52.0}
        settle_ua, read_point = self._mock_read(ua_to_ia)
        result = _refine_curve_inline(pts, ua_vals, s, settle_ua, None,
                                      read_point, None, None, 0.0, 0.0,
                                      min_safe_ua=35.0)
        for p in result:
            self.assertGreaterEqual(p["ua"], 35.0,
                                    f"Refine point Ua={p['ua']} below min_safe_ua=35")

    def test_stop_aborts_refine(self):
        """Stop callback should abort refine early."""
        s = self._settings()
        ua_vals = [0, 20, 40, 60, 80, 100]
        pts = [
            {"ua": 0, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 20, "ug1": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 40, "ug1": 0.0, "ia": 50.0, "ig2": 0.0},
            {"ua": 60, "ug1": 0.0, "ia": 55.0, "ig2": 0.0},
            {"ua": 80, "ug1": 0.0, "ia": 57.0, "ig2": 0.0},
            {"ua": 100, "ug1": 0.0, "ia": 58.0, "ig2": 0.0},
        ]
        ua_to_ia = {30: 25.0, 10: 0.0, 50: 52.0}
        settle_ua, read_point = self._mock_read(ua_to_ia)
        result = _refine_curve_inline(pts, ua_vals, s, settle_ua, None,
                                      read_point, None, lambda: True,
                                      0.0, 0.0)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Refine efficiency regression: point count stays within reasonable bounds
# ---------------------------------------------------------------------------

class TestRefineEfficiencyRegression(unittest.TestCase):
    """Weak regression guard for adaptive refinement.

    Uses synthetic pentode with sharp knee at Ua≈Ug2 to trigger refine.
    Asserts:
      - refine adds points vs plain grid (not broken / no-op)
      - refine count stays within 3× grid (not runaway)

    Thresholds intentionally loose — this is not a tuning test, just a
    regression guard against major refine breakage.  For tuning see
    ``tools/refine_efficiency.py``.
    """

    def _make_tube_client(self):
        """Mock client: synthetic pentode Ia(Ua, Ug1, Ug2) with soft knee."""
        import numpy as np
        client = MagicMock()
        state = {"Ua": 0.0, "Ug1": 0.0, "Ug2": 0.0}

        def cap_set(name, value, delay=0.05):
            state[name] = float(value)

        def get_s(name, real=False):
            if name == "Er":
                return 0
            if name in ("Ua", "Ug2") and real:
                return state.get(name, 0)
            if name == "Ug1" and real:
                return state.get("Ug1", 0)
            if name == "Ia" and real:
                ug1 = -state.get("Ug1", 0) / 100.0
                ug2 = state.get("Ug2", 0)
                ua = state.get("Ua", 0)
                if ug1 <= -10 or ug2 <= 0:
                    return 0
                ug1_f = (ug1 + 10) / 8.0
                ua_f = 1.0 - float(np.exp(-3.0 * ua / ug2))
                ia_ma = ug1_f * ua_f * 100.0
                return int(round(ia_ma / 0.01))
            if name == "Ig2" and real:
                return 0
            return 100

        client.set_param = MagicMock(side_effect=cap_set)
        client.get_param = MagicMock(side_effect=get_s)
        return client

    def _run(self, refine: bool):
        client = self._make_tube_client()
        settings = ScanSettings(
            ua=ScanRange(10, 300, 30),
            ug1=ScanRange(-8, -2, 2),
            ug2=ScanRange(200, 200, 0),
            uh=6.3, ih=0.0,
            is_triode=False, ug2_track_ua=False,
            ua_settle_per_volt_s=0.0, ua_settle_base_s=0.0,
            ua_tolerance=1.0, ua_retries=1,
            ug1_settle_per_volt_s=0.0, ug1_settle_base_s=0.0,
            ug1_tolerance=0.1, ug1_retries=1,
            ug2_settle_per_volt_s=0.0, ug2_settle_base_s=0.0,
            ug2_tolerance=1.0, ug2_retries=1,
            ia_samples=1,
            calibration=CalibrationData(),
            refine_enabled=refine,
            refine_max_depth=2,
            refine_min_step_ua=3.0,
            refine_onset_ma=0.5,
            refine_curvature_thr=0.1,
            refine_gradient_ratio=2.5,
            refine_ig2_delta_min=0.5,
            refine_delta_ia_thr=0.15,
        )
        with patch("time.sleep"):
            return run_scan(client, settings)

    def test_refine_adds_points_but_stays_bounded(self):
        """Refine: more points than grid, but less than 3× grid."""
        grid_pts = self._run(refine=False)
        refined_pts = self._run(refine=True)

        self.assertGreater(len(refined_pts), len(grid_pts),
            "Refine should add at least one point (synthetic curve has a knee)")
        self.assertLessEqual(len(refined_pts), 3 * len(grid_pts),
            f"Refine runaway: grid={len(grid_pts)}, refined={len(refined_pts)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
