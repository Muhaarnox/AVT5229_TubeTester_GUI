"""Tests for heatmap improvement features:
- fill_nan_nearest
- suppress_sparse
- HeatmapMarker aux_grids and edge clamping
- draw_qpoint_on_heatmap / draw_swing_range_lines
"""
from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ── fill_nan_nearest ──────────────────────────────────────────────

class TestFillNanNearest(unittest.TestCase):

    def test_no_nan(self):
        from lm19.plotting.grids import fill_nan_nearest
        g = np.array([[1, 2], [3, 4]], dtype=float)
        filled = fill_nan_nearest(g)
        np.testing.assert_array_equal(filled, g)
        assert filled is not g  # returns copy

    def test_all_nan(self):
        from lm19.plotting.grids import fill_nan_nearest
        g = np.full((3, 3), np.nan)
        filled = fill_nan_nearest(g)
        assert np.all(np.isnan(filled))

    def test_single_nan_corner(self):
        from lm19.plotting.grids import fill_nan_nearest
        g = np.array([[np.nan, 2], [3, 4]], dtype=float)
        filled = fill_nan_nearest(g)
        assert not np.any(np.isnan(filled))
        # Corner should be filled from nearest neighbor (2 or 3)
        assert filled[0, 0] in (2.0, 3.0)

    def test_center_nan(self):
        from lm19.plotting.grids import fill_nan_nearest
        g = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]], dtype=float)
        filled = fill_nan_nearest(g)
        assert not np.any(np.isnan(filled))
        # Center filled from nearest — one of 2, 4, 6, 8
        assert filled[1, 1] in (2.0, 4.0, 6.0, 8.0)

    def test_row_of_nan(self):
        from lm19.plotting.grids import fill_nan_nearest
        g = np.array([[1, 2, 3], [np.nan, np.nan, np.nan], [7, 8, 9]], dtype=float)
        filled = fill_nan_nearest(g)
        assert not np.any(np.isnan(filled))

    def test_original_not_modified(self):
        from lm19.plotting.grids import fill_nan_nearest
        g = np.array([[1, np.nan], [3, 4]], dtype=float)
        fill_nan_nearest(g)
        assert np.isnan(g[0, 1])  # original untouched


# ── suppress_sparse ───────────────────────────────────────────────

class TestSuppressSparse(unittest.TestCase):

    def test_no_nan(self):
        from lm19.plotting.grids import suppress_sparse
        g = np.ones((3, 4))
        result = suppress_sparse(g, min_valid=2)
        np.testing.assert_array_equal(result, g)

    def test_all_nan_row_untouched(self):
        from lm19.plotting.grids import suppress_sparse
        g = np.array([[1, 2, 3], [np.nan, np.nan, np.nan], [4, 5, 6]], dtype=float)
        result = suppress_sparse(g, min_valid=1)
        # All-NaN row stays NaN with min_valid=1
        assert np.all(np.isnan(result[1]))

    def test_single_valid_suppressed_at_min2(self):
        from lm19.plotting.grids import suppress_sparse
        g = np.array([[1, 2, 3], [np.nan, 5, np.nan], [4, 5, 6]], dtype=float)
        result = suppress_sparse(g, min_valid=2)
        # Row 1 has only 1 valid cell → suppressed
        assert np.all(np.isnan(result[1]))

    def test_default_suppresses_isolated_point_row(self):
        """ML-065: the DEFAULT must implement the documented intent -
        MIN_SPARSE_VALID=1 made every production call a no-op."""
        from lm19.plotting.grids import suppress_sparse
        g = np.array([[1, 2, 3], [np.nan, 5, np.nan], [4, 5, 6]], dtype=float)
        result = suppress_sparse(g)
        assert np.all(np.isnan(result[1]))

    def test_default_suppresses_isolated_point_col(self):
        from lm19.plotting.grids import suppress_sparse
        g = np.array([[1, np.nan], [2, np.nan], [3, 7]], dtype=float)
        result = suppress_sparse(g)
        assert np.all(np.isnan(result[:, 1]))

    def test_default_keeps_two_valid_row(self):
        """Mutation-audit: a signature default decoupled from
        MIN_SPARSE_VALID (e.g. min_valid=3) survived - the constant pin
        does not see the signature. Two valid cells must survive the
        DEFAULT call."""
        from lm19.plotting.grids import suppress_sparse
        g = np.array([[1, 2, 3], [np.nan, 5, 6], [4, 5, 6]], dtype=float)
        result = suppress_sparse(g)
        assert not np.all(np.isnan(result[1]))

    def test_default_constant_is_two(self):
        from lm19.plotting.grids import MIN_SPARSE_VALID
        assert MIN_SPARSE_VALID == 2

    def test_single_valid_kept_at_min1(self):
        from lm19.plotting.grids import suppress_sparse
        g = np.array([[1, 2, 3], [np.nan, 5, np.nan], [4, 5, 6]], dtype=float)
        result = suppress_sparse(g, min_valid=1)
        # Row 1 has 1 valid cell → kept with min_valid=1
        assert result[1, 1] == 5.0

    def test_sparse_column_suppressed(self):
        from lm19.plotting.grids import suppress_sparse
        g = np.array([[1, np.nan, 3], [2, np.nan, 4], [3, 5, 5]], dtype=float)
        result = suppress_sparse(g, min_valid=2)
        # Column 1 has only 1 valid (row 2) → suppressed
        assert np.isnan(result[2, 1])

    def test_returns_copy(self):
        from lm19.plotting.grids import suppress_sparse
        g = np.array([[1, np.nan], [3, 4]], dtype=float)
        result = suppress_sparse(g, min_valid=2)
        assert result is not g


# ── HeatmapMarker aux_grids (Qt-dependent, use fixture) ──────────

def _ensure_qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class TestHeatmapMarkerAux(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._app = _ensure_qapp()

    def _make_marker(self):
        from app.heatmap_marker import HeatmapMarker
        import pyqtgraph as pg
        return HeatmapMarker(pg.PlotWidget(), "Z", "unit", ".1f")

    def test_set_aux_grids_clears(self):
        m = self._make_marker()
        m.set_aux_grids([("A", "u", ".1f", np.ones((3, 3)),
                          np.array([0, 1, 2.0]), np.array([0, 1, 2.0]))])
        assert len(m._aux) == 1
        m.set_aux_grids([])
        assert len(m._aux) == 0

    def test_clear_resets_aux(self):
        m = self._make_marker()
        m.set_aux_grids([("A", "u", ".1f", np.ones((3, 3)),
                          np.array([0, 1, 2.0]), np.array([0, 1, 2.0]))])
        m.clear()
        assert m._aux == []

    def test_set_grid_then_aux(self):
        m = self._make_marker()
        z = np.array([[1, 2], [3, 4]], dtype=float)
        x = np.array([0, 1.0])
        y = np.array([0, 1.0])
        m.set_grid(z, x, y)
        m.set_aux_grids([("Ia", "mA", ".2f", z, x, y)])
        assert m._z_grid is not None
        assert len(m._aux) == 1


# ── Q-point overlay ──────────────────────────────────────────────

class TestQpointOverlay(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._app = _ensure_qapp()

    def _make_plot(self):
        import pyqtgraph as pg
        return pg.PlotWidget()

    def test_qpoint_on_heatmap_returns_items(self):
        from app.plotting.overlays import draw_qpoint_on_heatmap
        analysis = {
            "ua_0": 200.0, "ug1_0": -8.0, "ia_0": 40.0,
            "pt_neg": {"ua": 260, "ug1": -12, "ia": 10},
            "pt_pos": {"ua": 140, "ug1": -4, "ia": 70},
        }
        items = draw_qpoint_on_heatmap(self._make_plot(), analysis)
        assert len(items) == 4

    def test_qpoint_on_heatmap_no_swing(self):
        from app.plotting.overlays import draw_qpoint_on_heatmap
        analysis = {"ua_0": 200.0, "ug1_0": -8.0, "ia_0": 40.0}
        items = draw_qpoint_on_heatmap(self._make_plot(), analysis)
        assert len(items) == 2

    def test_qpoint_on_heatmap_none_coords(self):
        from app.plotting.overlays import draw_qpoint_on_heatmap
        items = draw_qpoint_on_heatmap(
            self._make_plot(), {"ua_0": None, "ug1_0": -8.0})
        assert items == []

    def test_swing_range_lines_ug1(self):
        from app.plotting.overlays import draw_swing_range_lines
        analysis = {
            "ug1_0": -8.0,
            "pt_neg": {"ua": 260, "ug1": -12, "ia": 10},
            "pt_pos": {"ua": 140, "ug1": -4, "ia": 70},
        }
        items = draw_swing_range_lines(self._make_plot(), analysis, x_param="Ug1")
        assert len(items) == 2

    def test_swing_range_lines_ua(self):
        from app.plotting.overlays import draw_swing_range_lines
        analysis = {
            "ua_0": 200.0,
            "pt_neg": {"ua": 260, "ug1": -12, "ia": 10},
            "pt_pos": {"ua": 140, "ug1": -4, "ia": 70},
        }
        items = draw_swing_range_lines(self._make_plot(), analysis, x_param="Ua")
        assert len(items) == 2

    def test_swing_range_no_swing_pts(self):
        from app.plotting.overlays import draw_swing_range_lines
        items = draw_swing_range_lines(
            self._make_plot(), {"ug1_0": -8.0}, x_param="Ug1")
        assert items == []


# ── Marker edge clamping ─────────────────────────────────────────

class TestNanCornerFallbackNearest(unittest.TestCase):
    """ML-141: with a NaN corner the fallback must pick the NEAREST
    valid corner — the old distance formulas were inverted (q00 used the
    distance to q11 etc.) and picked the FARTHEST one."""

    def _grid(self, q00, q10, q01, q11):
        import numpy as np
        z = np.array([[q00, q10], [q01, q11]], dtype=float)
        return z, np.array([0.0, 10.0]), np.array([0.0, 10.0])

    def test_query_near_nan_corner_picks_nearest_valid(self):
        from lm19.heatmap_interp import interp_bilinear
        # q11 is NaN; query at (9, 8) is nearest to q10 at (10, 0)?? no:
        # corners: q00(0,0)=1, q10(10,0)=2, q01(0,10)=3, q11(10,10)=NaN.
        # query (9, 8): d²(q10) = 1 + 64 = 65; d²(q01) = 81 + 4 = 85;
        # d²(q00) = 81 + 64 = 145 → nearest valid is q10 = 2.0.
        # Inverted formulas picked q00 = 1.0.
        z, x, y = self._grid(1.0, 2.0, 3.0, float("nan"))
        val = interp_bilinear(z, x, y, 9.0, 8.0)
        self.assertEqual(val, 2.0)

    def test_query_near_valid_corner_with_far_nan(self):
        from lm19.heatmap_interp import interp_bilinear
        # q01 is NaN; query (1, 2) is nearest to q00 = 1.0.
        z, x, y = self._grid(1.0, 2.0, float("nan"), 4.0)
        val = interp_bilinear(z, x, y, 1.0, 2.0)
        self.assertEqual(val, 1.0)


class TestMarkerEdgeClamping(unittest.TestCase):

    def test_interp_at_exact_boundary(self):
        """interp_bilinear at exact grid boundary returns value."""
        from lm19.heatmap_interp import interp_bilinear
        z = np.array([[1, 2], [3, 4]], dtype=float)
        x = np.array([0, 10.0])
        y = np.array([0, 5.0])
        # Exact corners
        assert interp_bilinear(z, x, y, 0, 0) == 1.0
        assert interp_bilinear(z, x, y, 10, 5) == 4.0

    def test_interp_just_outside_returns_none(self):
        """interp_bilinear just outside boundary returns None."""
        from lm19.heatmap_interp import interp_bilinear
        z = np.array([[1, 2], [3, 4]], dtype=float)
        x = np.array([0, 10.0])
        y = np.array([0, 5.0])
        assert interp_bilinear(z, x, y, -0.1, 2.5) is None
        assert interp_bilinear(z, x, y, 10.1, 2.5) is None


if __name__ == "__main__":
    unittest.main()
