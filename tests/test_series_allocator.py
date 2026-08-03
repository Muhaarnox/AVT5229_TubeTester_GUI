"""Tests for PlotManager series_id allocation and removal."""

from unittest.mock import MagicMock

from app.plot_manager import PlotManager


def _make_pm():
    """Create a minimal PlotManager without Qt widgets."""
    pm = object.__new__(PlotManager)
    pm.points = []
    pm.series_labels = {}
    pm.series_colors = {}
    pm.series_ug2_track = {}
    pm.series_models = {}
    pm.series_grids = {}
    pm.is_triode = False
    pm.legend_hidden = False
    pm.current_curve_points = []
    pm.labeled_ug1 = set()
    pm.renderer = MagicMock()
    pm.renderer.ug2_cluster_thr = 5.0
    pm.w = {}
    return pm


class TestAllocateSeriesId:
    """allocate_series_id returns unique IDs."""

    def test_empty_returns_1(self):
        pm = _make_pm()
        assert pm.allocate_series_id() == 1

    def test_skips_existing_scan(self):
        pm = _make_pm()
        pm.points = [
            {"ua": 100, "ug1": -2, "ia": 5, "series_id": 0},
            {"ua": 100, "ug1": -2, "ia": 5, "series_id": 1},
        ]
        assert pm.allocate_series_id() == 2

    def test_skips_model_series(self):
        pm = _make_pm()
        pm.series_models[1] = MagicMock()
        assert pm.allocate_series_id() == 2

    def test_skips_both_points_and_models(self):
        pm = _make_pm()
        pm.points = [
            {"ua": 100, "ug1": -2, "ia": 5, "series_id": 1},
        ]
        pm.series_models[2] = MagicMock()
        assert pm.allocate_series_id() == 3

    def test_no_collision_after_remove(self):
        pm = _make_pm()
        pm.points = [
            {"ua": 100, "ug1": -2, "ia": 5, "series_id": 1},
        ]
        pm.series_labels[1] = "test"
        # After remove, id 1 becomes available again
        pm.remove_series(1)
        assert pm.allocate_series_id() == 1


class TestRemoveSeries:
    """remove_series cleans up all traces."""

    def test_removes_points(self):
        pm = _make_pm()
        pm.points = [
            {"ua": 100, "ug1": -2, "ia": 5, "series_id": 0},
            {"ua": 100, "ug1": -2, "ia": 5, "series_id": 1},
            {"ua": 200, "ug1": -4, "ia": 10, "series_id": 1},
        ]
        pm.series_labels[1] = "test"
        pm.series_colors[1] = "#ff0000"
        pm.series_ug2_track[1] = False
        pm.series_models[1] = MagicMock()
        pm.series_grids[1] = MagicMock()

        pm.remove_series(1)

        assert len(pm.points) == 1
        assert pm.points[0]["series_id"] == 0
        assert 1 not in pm.series_labels
        assert 1 not in pm.series_colors
        assert 1 not in pm.series_ug2_track
        assert 1 not in pm.series_models
        assert 1 not in pm.series_grids

    def test_remove_nonexistent_is_safe(self):
        pm = _make_pm()
        pm.remove_series(99)  # should not raise

    def test_preserves_other_series(self):
        pm = _make_pm()
        pm.points = [
            {"ua": 100, "ug1": -2, "ia": 5, "series_id": 1},
            {"ua": 100, "ug1": -2, "ia": 5, "series_id": 2},
        ]
        pm.series_labels[1] = "A"
        pm.series_labels[2] = "B"

        pm.remove_series(1)

        assert len(pm.points) == 1
        assert pm.points[0]["series_id"] == 2
        assert 2 in pm.series_labels
        assert pm.series_labels[2] == "B"


class TestClearAll:
    """clear_all clears model series too."""

    def test_clears_models_and_grids(self):
        pm = _make_pm()
        pm.series_models[1] = MagicMock()
        pm.series_grids[1] = MagicMock()
        pm.renderer.clear = MagicMock()

        pm.clear_all()

        assert pm.series_models == {}
        assert pm.series_grids == {}
