"""Tests for CurveMarker snap-to-curve interpolation and CurveData builders."""

import os
import sys
import unittest

import numpy as np
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.curve_marker import CurveMarker
from lm19.curve_data import CurveData
from lm19.calibration import CalibrationData
from lm19.spice_export import _koren_ia, _koren_ia_pentode, SpiceFitResult
from lm19.constants import (
    TOPOLOGY_TRIODE,
)
from lm19.tube_model_base import (
    MODEL_TYPE_KOREN,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
)


class TestModelOverlayLogic(unittest.TestCase):
    """Test model overlay parameter flow without GUI."""

    def test_spice_fit_result_params_triode(self):
        """SpiceFitResult for triode should contain all required params."""
        result = SpiceFitResult(
            model_type=TOPOLOGY_TRIODE, algorithm=MODEL_TYPE_KOREN,
            params={"mu": 100, "ex": 1.4, "kg1": 1060, "kp": 600, "kvb": 300},
            rms_error=0.5, max_error=1.0, n_points=100, path="test.sub",
        )
        self.assertIn("mu", result.params)
        self.assertIn("ex", result.params)
        self.assertIn("kg1", result.params)
        self.assertIn("kp", result.params)
        self.assertIn("kvb", result.params)

    def test_spice_fit_result_params_pentode(self):
        """SpiceFitResult for pentode should also contain kg2."""
        result = SpiceFitResult(
            model_type=TOPOLOGY_PENTODE, algorithm=MODEL_TYPE_KOREN,
            params={"mu": 11, "ex": 1.35, "kg1": 650, "kp": 60,
                    "kvb": 24, "kg2": 4200},
            rms_error=1.0, max_error=3.0, n_points=200, path="test.sub",
        )
        self.assertIn("kg2", result.params)

    def test_overlay_triode_model_curve(self):
        """Triode overlay should produce valid Ia values."""
        params = {"mu": 100, "ex": 1.4, "kg1": 1060, "kp": 600, "kvb": 300}
        ua = np.linspace(10, 350, 50)
        ug1 = -2.0
        ia = _koren_ia(ua, np.full_like(ua, ug1),
                       params["mu"], params["ex"], params["kg1"],
                       params["kp"], params["kvb"]) * 1000.0
        self.assertTrue(np.all(ia >= 0))
        self.assertTrue(np.all(np.isfinite(ia)))

    def test_overlay_pentode_model_curve(self):
        """Pentode overlay should produce valid Ia values."""
        params = {"mu": 11, "ex": 1.35, "kg1": 650, "kp": 60, "kvb": 24}
        ua = np.linspace(10, 400, 50)
        ug1 = -10.0
        ug2 = 250.0
        ia = _koren_ia_pentode(ua, np.full_like(ua, ug1),
                               np.full_like(ua, ug2),
                               params["mu"], params["ex"], params["kg1"],
                               params["kp"], params["kvb"]) * 1000.0
        self.assertTrue(np.all(ia >= 0))
        self.assertTrue(np.all(np.isfinite(ia)))


class TestCurveMarkerInterp(unittest.TestCase):
    """Test CurveMarker._interp_y (static, no Qt needed)."""

    def _make_curve(self, xs, ys):
        return CurveData(x=np.array(xs, dtype=float),
                         y=np.array(ys, dtype=float))

    def test_exact_points(self):
        """Interpolation at exact data points returns exact values."""
        c = self._make_curve([0, 100, 200], [0, 10, 20])
        self.assertAlmostEqual(CurveMarker._interp_y(c, 0), 0.0)
        self.assertAlmostEqual(CurveMarker._interp_y(c, 100), 10.0)
        self.assertAlmostEqual(CurveMarker._interp_y(c, 200), 20.0)

    def test_midpoint_linear(self):
        """Linear interpolation at midpoint."""
        c = self._make_curve([0, 100], [0, 50])
        self.assertAlmostEqual(CurveMarker._interp_y(c, 50), 25.0)

    def test_quarter_interp(self):
        """Linear interpolation at 25% and 75%."""
        c = self._make_curve([0, 100, 200, 300], [0, 10, 20, 30])
        self.assertAlmostEqual(CurveMarker._interp_y(c, 150), 15.0)
        self.assertAlmostEqual(CurveMarker._interp_y(c, 75), 7.5)

    def test_out_of_range_returns_none(self):
        """X outside curve range returns None."""
        c = self._make_curve([50, 100, 200], [5, 10, 20])
        self.assertIsNone(CurveMarker._interp_y(c, 49.9))
        self.assertIsNone(CurveMarker._interp_y(c, 200.1))
        self.assertIsNone(CurveMarker._interp_y(c, -10))

    def test_single_point_returns_none(self):
        """Curve with < 2 points cannot be interpolated."""
        c = self._make_curve([100], [10])
        self.assertIsNone(CurveMarker._interp_y(c, 100))

    def test_empty_curve(self):
        c = self._make_curve([], [])
        self.assertIsNone(CurveMarker._interp_y(c, 0))

    def test_nonlinear_curve(self):
        """Interpolation on a curve with nonlinear Y values."""
        # Simulate Ia(Ua) curve: [0, 1, 4, 9] at Ua [0, 1, 2, 3]
        c = self._make_curve([0, 1, 2, 3], [0, 1, 4, 9])
        # At x=1.5, linear interp between (1,1) and (2,4) → 2.5
        self.assertAlmostEqual(CurveMarker._interp_y(c, 1.5), 2.5)


class TestCurveDataBuilders(unittest.TestCase):
    """Curve-grouping builders in ``lm19.plotting.grouping``."""

    def test_build_curves_2d_triode(self):
        """Build 2D curves for triode: group by Ug1 only."""
        from lm19.plotting.grouping import build_curves_2d
        points = [
            {"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 0},
            {"ua": 200, "ia": 10.0, "ug1": -1.0, "ug2": 0},
            {"ua": 100, "ia": 3.0, "ug1": -2.0, "ug2": 0},
            {"ua": 200, "ia": 6.0, "ug1": -2.0, "ug2": 0},
        ]
        curves = build_curves_2d(points, track_sids={0})
        self.assertEqual(len(curves), 2)
        for c in curves:
            self.assertEqual(len(c.x), 2)
            self.assertTrue(c.x[0] < c.x[1])
            self.assertIn("Ug1", c.extra)

    def test_build_curves_2d_pentode(self):
        """Build 2D curves for pentode: group by (Ug1, Ug2)."""
        from lm19.plotting.grouping import build_curves_2d
        points = [
            {"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 250},
            {"ua": 200, "ia": 10.0, "ug1": -1.0, "ug2": 250},
            {"ua": 100, "ia": 4.0, "ug1": -1.0, "ug2": 200},
            {"ua": 200, "ia": 8.0, "ug1": -1.0, "ug2": 200},
        ]
        curves = build_curves_2d(points)
        self.assertEqual(len(curves), 2)
        ug2_values = set()
        for c in curves:
            self.assertIn("Ug2", c.extra)
            ug2_values.add(float(c.extra["Ug2"][0]))
        self.assertEqual(ug2_values, {200.0, 250.0})

    def test_build_curves_2d_ug2_track(self):
        """With track_sids containing the series, Ug2 is ignored in grouping."""
        from lm19.plotting.grouping import build_curves_2d
        points = [
            {"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 100},
            {"ua": 200, "ia": 10.0, "ug1": -1.0, "ug2": 200},
        ]
        curves = build_curves_2d(points, track_sids={0})
        self.assertEqual(len(curves), 1)
        self.assertNotIn("Ug2", curves[0].label)

    def test_build_curves_2d_series(self):
        """Points with series_id are grouped separately."""
        from lm19.plotting.grouping import build_curves_2d
        points = [
            {"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 0},
            {"ua": 200, "ia": 10.0, "ug1": -1.0, "ug2": 0},
            {"ua": 100, "ia": 4.5, "ug1": -1.0, "ug2": 0,
             "series_id": "s1", "lamp_type": "ECC83", "lamp_id": "001"},
            {"ua": 200, "ia": 9.5, "ug1": -1.0, "ug2": 0,
             "series_id": "s1", "lamp_type": "ECC83", "lamp_id": "001"},
        ]
        curves = build_curves_2d(points, track_sids={0, "s1"})
        self.assertEqual(len(curves), 2)
        labels = " ".join(c.label for c in curves)
        self.assertIn("ECC83", labels)

    def test_build_curves_2d_empty(self):
        from lm19.plotting.grouping import build_curves_2d
        curves = build_curves_2d([])
        self.assertEqual(curves, [])

    def test_build_curves_2d_single_point_filtered(self):
        """Curves with < 2 points are filtered out."""
        from lm19.plotting.grouping import build_curves_2d
        points = [{"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 0}]
        curves = build_curves_2d(points, track_sids={0})
        self.assertEqual(len(curves), 0)

    def test_build_curves_transfer(self):
        """Build transfer curves: group by Ua."""
        from lm19.plotting.grouping import build_curves_transfer
        points = [
            {"ua": 250, "ia": 10.0, "ug1": -1.0, "ug2": 0},
            {"ua": 250, "ia": 5.0, "ug1": -2.0, "ug2": 0},
            {"ua": 250, "ia": 2.0, "ug1": -3.0, "ug2": 0},
            {"ua": 200, "ia": 8.0, "ug1": -1.0, "ug2": 0},
            {"ua": 200, "ia": 4.0, "ug1": -2.0, "ug2": 0},
        ]
        curves = build_curves_transfer(points, is_triode=True)
        self.assertEqual(len(curves), 2)
        for c in curves:
            self.assertIn("Ua", c.label)
            for i in range(len(c.x) - 1):
                self.assertLessEqual(c.x[i], c.x[i + 1])

    def test_build_curves_ig2(self):
        """Build Ig2 curves: group by Ug1."""
        from lm19.plotting.grouping import build_curves_ig2
        points = [
            {"ua": 100, "ia": 5.0, "ig2": 1.0, "ug1": -1.0},
            {"ua": 200, "ia": 10.0, "ig2": 2.0, "ug1": -1.0},
            {"ua": 100, "ia": 3.0, "ig2": 0.5, "ug1": -2.0},
            {"ua": 200, "ia": 6.0, "ig2": 1.5, "ug1": -2.0},
        ]
        curves = build_curves_ig2(points)
        self.assertEqual(len(curves), 2)
        for c in curves:
            self.assertIn("Ug1", c.label)
            self.assertTrue(all(y >= 0 for y in c.y))


class TestCurveMarkerInterpEdgeCases(unittest.TestCase):
    """Additional edge-case tests for _interp_y."""

    def _make_curve(self, xs, ys):
        return CurveData(x=np.array(xs, dtype=float),
                         y=np.array(ys, dtype=float))

    def test_at_first_boundary(self):
        """Interpolation exactly at first X returns first Y."""
        c = self._make_curve([50, 100, 200], [5, 10, 20])
        self.assertAlmostEqual(CurveMarker._interp_y(c, 50), 5.0)

    def test_at_last_boundary(self):
        """Interpolation exactly at last X returns last Y."""
        c = self._make_curve([50, 100, 200], [5, 10, 20])
        self.assertAlmostEqual(CurveMarker._interp_y(c, 200), 20.0)

    def test_two_points_only(self):
        """Minimum valid curve (2 points) interpolates correctly."""
        c = self._make_curve([0, 10], [0, 100])
        self.assertAlmostEqual(CurveMarker._interp_y(c, 5), 50.0)
        self.assertAlmostEqual(CurveMarker._interp_y(c, 0), 0.0)
        self.assertAlmostEqual(CurveMarker._interp_y(c, 10), 100.0)

    def test_negative_x_range(self):
        """Curves with negative X values (e.g. Ug1 transfer)."""
        c = self._make_curve([-4.0, -3.0, -2.0, -1.0, 0.0], [0, 2, 5, 10, 18])
        self.assertAlmostEqual(CurveMarker._interp_y(c, -2.5), 3.5)
        self.assertIsNone(CurveMarker._interp_y(c, -4.1))
        self.assertIsNone(CurveMarker._interp_y(c, 0.1))

    def test_large_curve(self):
        """Performance sanity: 1000-point curve interpolates correctly."""
        xs = np.linspace(0, 500, 1000)
        ys = xs ** 1.5 / 100.0  # typical Ia(Ua) shape
        c = CurveData(x=xs, y=ys)
        result = CurveMarker._interp_y(c, 250.0)
        self.assertIsNotNone(result)
        expected = 250.0 ** 1.5 / 100.0
        self.assertAlmostEqual(result, expected, places=1)

    def test_constant_y(self):
        """Flat curve (constant Y) interpolates to that constant."""
        c = self._make_curve([0, 100, 200], [5.0, 5.0, 5.0])
        self.assertAlmostEqual(CurveMarker._interp_y(c, 50), 5.0)
        self.assertAlmostEqual(CurveMarker._interp_y(c, 150), 5.0)


class TestCurveDataBuildersExtended(unittest.TestCase):
    """Extended tests for curve builders: pentode transfer, Ig2 edges, missing fields."""

    def test_build_curves_transfer_pentode(self):
        """Transfer curves for pentode: grouped by (Ua, Ug2)."""
        from lm19.plotting.grouping import build_curves_transfer
        points = [
            {"ua": 250, "ia": 10.0, "ug1": -1.0, "ug2": 250},
            {"ua": 250, "ia": 5.0, "ug1": -2.0, "ug2": 250},
            {"ua": 250, "ia": 8.0, "ug1": -1.0, "ug2": 200},
            {"ua": 250, "ia": 4.0, "ug1": -2.0, "ug2": 200},
        ]
        curves = build_curves_transfer(points, is_triode=False)
        self.assertEqual(len(curves), 2)
        for c in curves:
            self.assertIn("Ua", c.label)
            self.assertIn("Ug2", c.label)

    def test_build_curves_transfer_empty(self):
        from lm19.plotting.grouping import build_curves_transfer
        curves = build_curves_transfer([], is_triode=True)
        self.assertEqual(curves, [])

    def test_build_curves_transfer_single_point_filtered(self):
        """Transfer curve with single point per group is filtered out."""
        from lm19.plotting.grouping import build_curves_transfer
        points = [{"ua": 250, "ia": 10.0, "ug1": -1.0, "ug2": 0}]
        curves = build_curves_transfer(points, is_triode=True)
        self.assertEqual(len(curves), 0)

    def test_build_curves_ig2_empty(self):
        from lm19.plotting.grouping import build_curves_ig2
        curves = build_curves_ig2([])
        self.assertEqual(curves, [])

    def test_build_curves_ig2_single_point_filtered(self):
        from lm19.plotting.grouping import build_curves_ig2
        points = [{"ua": 100, "ia": 5.0, "ig2": 1.0, "ug1": -1.0}]
        curves = build_curves_ig2(points)
        self.assertEqual(len(curves), 0)

    def test_build_curves_ig2_missing_ig2_field(self):
        """Points without ig2 field default to 0."""
        from lm19.plotting.grouping import build_curves_ig2
        points = [
            {"ua": 100, "ia": 5.0, "ug1": -1.0},
            {"ua": 200, "ia": 10.0, "ug1": -1.0},
        ]
        curves = build_curves_ig2(points)
        self.assertEqual(len(curves), 1)
        self.assertTrue(all(y == 0.0 for y in curves[0].y))

    def test_build_curves_2d_missing_ug2_defaults_zero(self):
        """Points without ug2 field default to 0 in pentode mode."""
        from lm19.plotting.grouping import build_curves_2d
        points = [
            {"ua": 100, "ia": 5.0, "ug1": -1.0},
            {"ua": 200, "ia": 10.0, "ug1": -1.0},
        ]
        curves = build_curves_2d(points)
        self.assertEqual(len(curves), 1)

    def test_build_curves_2d_missing_ug1_defaults_zero(self):
        """Points without ug1 field default to 0."""
        from lm19.plotting.grouping import build_curves_2d
        points = [
            {"ua": 100, "ia": 5.0},
            {"ua": 200, "ia": 10.0},
        ]
        curves = build_curves_2d(points, track_sids={0})
        self.assertEqual(len(curves), 1)
        self.assertAlmostEqual(float(curves[0].extra["Ug1"][0]), 0.0)

    def test_build_curves_2d_sorted_output(self):
        """Output curves have X sorted ascending even if input is not."""
        from lm19.plotting.grouping import build_curves_2d
        points = [
            {"ua": 300, "ia": 15.0, "ug1": -1.0, "ug2": 0},
            {"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 0},
            {"ua": 200, "ia": 10.0, "ug1": -1.0, "ug2": 0},
        ]
        curves = build_curves_2d(points, track_sids={0})
        self.assertEqual(len(curves), 1)
        c = curves[0]
        self.assertTrue(all(c.x[i] <= c.x[i + 1] for i in range(len(c.x) - 1)))


class TestBuildCompareCurves(unittest.TestCase):
    """Test build_compare_curves standalone function."""

    def test_basic_grouping(self):
        """Group by (lamp_type, lamp_id, ug1, ug2)."""
        from lm19.curve_data import build_compare_curves
        points = [
            {"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 0,
             "lamp_type": "ECC83", "lamp_id": "001"},
            {"ua": 200, "ia": 10.0, "ug1": -1.0, "ug2": 0,
             "lamp_type": "ECC83", "lamp_id": "001"},
            {"ua": 100, "ia": 4.0, "ug1": -2.0, "ug2": 0,
             "lamp_type": "ECC83", "lamp_id": "001"},
            {"ua": 200, "ia": 8.0, "ug1": -2.0, "ug2": 0,
             "lamp_type": "ECC83", "lamp_id": "001"},
        ]
        curves = build_compare_curves(points)
        self.assertEqual(len(curves), 2)  # Two Ug1 values
        for c in curves:
            self.assertEqual(len(c.x), 2)
            self.assertIn("ECC83/001", c.label)

    def test_multiple_lamps(self):
        """Different lamps produce separate curves even at same Ug1."""
        from lm19.curve_data import build_compare_curves
        points = [
            {"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 0,
             "lamp_type": "ECC83", "lamp_id": "001"},
            {"ua": 200, "ia": 10.0, "ug1": -1.0, "ug2": 0,
             "lamp_type": "ECC83", "lamp_id": "001"},
            {"ua": 100, "ia": 4.5, "ug1": -1.0, "ug2": 0,
             "lamp_type": "ECC83", "lamp_id": "002"},
            {"ua": 200, "ia": 9.0, "ug1": -1.0, "ug2": 0,
             "lamp_type": "ECC83", "lamp_id": "002"},
        ]
        curves = build_compare_curves(points)
        self.assertEqual(len(curves), 2)  # Two lamps
        labels = " ".join(c.label for c in curves)
        self.assertIn("ECC83/001", labels)
        self.assertIn("ECC83/002", labels)

    def test_pentode_ug2_grouping(self):
        """Pentode: different Ug2 values create separate curves with Ug2 in extra."""
        from lm19.curve_data import build_compare_curves
        points = [
            {"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 250,
             "lamp_type": "EL34", "lamp_id": "001"},
            {"ua": 200, "ia": 10.0, "ug1": -1.0, "ug2": 250,
             "lamp_type": "EL34", "lamp_id": "001"},
            {"ua": 100, "ia": 4.0, "ug1": -1.0, "ug2": 200,
             "lamp_type": "EL34", "lamp_id": "001"},
            {"ua": 200, "ia": 8.0, "ug1": -1.0, "ug2": 200,
             "lamp_type": "EL34", "lamp_id": "001"},
        ]
        curves = build_compare_curves(points)
        self.assertEqual(len(curves), 2)
        # Ug2 now in extra arrays, not in label
        ug2_values = set()
        for c in curves:
            self.assertIn("Ug2", c.extra)
            ug2_values.add(float(c.extra["Ug2"][0]))
        self.assertEqual(ug2_values, {200.0, 250.0})

    def test_empty(self):
        from lm19.curve_data import build_compare_curves
        self.assertEqual(build_compare_curves([]), [])

    def test_single_point_filtered(self):
        from lm19.curve_data import build_compare_curves
        points = [
            {"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 0,
             "lamp_type": "ECC83", "lamp_id": "001"},
        ]
        self.assertEqual(len(build_compare_curves(points)), 0)

    def test_no_lamp_type(self):
        """Points without lamp_type still produce curves."""
        from lm19.curve_data import build_compare_curves
        points = [
            {"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 0, "lamp_id": "x"},
            {"ua": 200, "ia": 10.0, "ug1": -1.0, "ug2": 0, "lamp_id": "x"},
        ]
        curves = build_compare_curves(points)
        self.assertEqual(len(curves), 1)
        # Label should contain lamp_id when lamp_type is empty
        self.assertIn("x", curves[0].label)

    def test_sorted_by_ua(self):
        """Output X arrays are sorted even if input is unordered."""
        from lm19.curve_data import build_compare_curves
        points = [
            {"ua": 300, "ia": 15.0, "ug1": -1.0, "ug2": 0,
             "lamp_type": "T", "lamp_id": "1"},
            {"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 0,
             "lamp_type": "T", "lamp_id": "1"},
            {"ua": 200, "ia": 10.0, "ug1": -1.0, "ug2": 0,
             "lamp_type": "T", "lamp_id": "1"},
        ]
        curves = build_compare_curves(points)
        c = curves[0]
        self.assertTrue(all(c.x[i] <= c.x[i + 1] for i in range(len(c.x) - 1)))

    def test_same_lamp_id_different_entry(self):
        """Two physical lamps with same lamp_type+lamp_id but different _entry_idx."""
        from lm19.curve_data import build_compare_curves
        points = [
            {"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 0,
             "lamp_type": "ECC83", "lamp_id": "001",
             "_entry_idx": 0, "_entry_name": "Tube A"},
            {"ua": 200, "ia": 10.0, "ug1": -1.0, "ug2": 0,
             "lamp_type": "ECC83", "lamp_id": "001",
             "_entry_idx": 0, "_entry_name": "Tube A"},
            {"ua": 100, "ia": 4.5, "ug1": -1.0, "ug2": 0,
             "lamp_type": "ECC83", "lamp_id": "001",
             "_entry_idx": 1, "_entry_name": "Tube B"},
            {"ua": 200, "ia": 9.0, "ug1": -1.0, "ug2": 0,
             "lamp_type": "ECC83", "lamp_id": "001",
             "_entry_idx": 1, "_entry_name": "Tube B"},
        ]
        curves = build_compare_curves(points)
        self.assertEqual(len(curves), 2)
        labels = " ".join(c.label for c in curves)
        self.assertIn("Tube A", labels)
        self.assertIn("Tube B", labels)

    def test_mixed_track_and_sweep(self):
        """Mix of track-mode (ug2 varies) and sweep-mode entries.

        Entry 0: triode-connected (ug2 tracks ua) — one Ug1, varying Ug2.
        Entry 1: pentode sweep — one Ug1, fixed Ug2.
        Both must produce curves (per-entry detection).
        """
        from lm19.curve_data import build_compare_curves
        # Entry 0: ug2 tracks ua (triode-connected pentode)
        track_pts = [
            {"ua": ua, "ia": ua * 0.05, "ug1": -5.0, "ug2": float(ua),
             "lamp_type": "6P18P", "lamp_id": "L1",
             "_entry_idx": 0, "_entry_name": "Triode"}
            for ua in range(50, 250, 25)
        ]
        # Entry 1: fixed ug2=250
        sweep_pts = [
            {"ua": ua, "ia": ua * 0.04, "ug1": -5.0, "ug2": 250.0,
             "lamp_type": "6P18P", "lamp_id": "L1",
             "_entry_idx": 1, "_entry_name": "Pentode"}
            for ua in range(50, 250, 25)
        ]
        curves = build_compare_curves(track_pts + sweep_pts)
        # Must get at least 2 curves: one from each entry
        self.assertGreaterEqual(len(curves), 2)
        labels = " ".join(c.label for c in curves)
        self.assertIn("Triode", labels)
        self.assertIn("Pentode", labels)


class TestCurveDataIntegrity(unittest.TestCase):
    """Test CurveData dataclass."""

    def test_fields(self):
        c = CurveData(
            x=np.array([1, 2, 3], dtype=float),
            y=np.array([10, 20, 30], dtype=float),
            label="test",
            extra={"Ig2": np.array([0.1, 0.2, 0.3])},
        )
        self.assertEqual(c.label, "test")
        self.assertEqual(len(c.x), 3)
        self.assertEqual(len(c.extra["Ig2"]), 3)

    def test_default_extra_empty(self):
        c = CurveData(x=np.array([1.0]), y=np.array([2.0]))
        self.assertEqual(c.label, "")
        self.assertEqual(c.extra, {})

    def test_sorted_x(self):
        """Curves should have sorted X; interp depends on it."""
        c = CurveData(x=np.array([10, 20, 30], dtype=float),
                      y=np.array([1, 2, 3], dtype=float))
        self.assertTrue(all(c.x[i] <= c.x[i + 1] for i in range(len(c.x) - 1)))


def _make_mock_client(get_param_side_effect=None):
    """Create a mock LM19Serial client with optional get_param responses."""
    client = MagicMock()
    client.set_param = MagicMock()
    if get_param_side_effect is not None:
        client.get_param = MagicMock(side_effect=get_param_side_effect)
    else:
        client.get_param = MagicMock(return_value=0)
    return client


if __name__ == "__main__":
    unittest.main(verbosity=2)
