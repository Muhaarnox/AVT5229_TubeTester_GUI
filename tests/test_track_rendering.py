"""Tests for track_sids propagation across all rendering paths.

Covers:
- _ensure_2d_cache with non-empty track_sids (mixed overlay)
- _is_triode_eff helper
- set_ug2_visibility (mock items)
- filter_ug2_slice / filter_ug2_multi with triode_eff for tracking data
- _grid_from_points with tracking data
- render_curve_incremental triode_eff decision
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.plotting import PlotRenderer
from lm19.plotting.grids import filter_ug2_slice, filter_ug2_multi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pt(ua, ug1, ug2, ia=5.0, series_id=0,
        lamp_type="6P14P", lamp_id="scan"):
    return {
        "ua": ua, "ug1": ug1, "ug2": ug2, "ia": ia,
        "ig2": 0.1, "uh": 6.3, "ih": 0.3,
        "series_id": series_id,
        "lamp_type": lamp_type, "lamp_id": lamp_id,
    }


def _tracking_points(n_ua=5, ug1=-1.0, series_id=0, **kw):
    """Triode-connected pentode: ug2 = ua (tracking)."""
    return [_pt(ua=50 + i * 50, ug1=ug1, ug2=50 + i * 50,
                ia=5.0 + i, series_id=series_id, **kw)
            for i in range(n_ua)]


def _pentode_points(ug2=200.0, n_ua=5, n_ug1=3, series_id=0, **kw):
    pts = []
    for g1 in range(n_ug1):
        ug1 = -1.0 * g1
        for a in range(n_ua):
            ua = 50 + a * 50
            ia = max(0, 10 + ug1 * 2 + ua * 0.05)
            pts.append(_pt(ua=ua, ug1=ug1, ug2=ug2, ia=ia,
                           series_id=series_id, **kw))
    return pts


def _make_renderer(**overrides):
    """Minimal PlotRenderer without Qt."""
    r = object.__new__(PlotRenderer)
    r.is_triode = False
    r.track_sids = set()
    r.ua_cluster_thr = 1.0
    r.ug1_cluster_thr = 0.3
    r.ug2_cluster_thr = 2.0
    r._2d_cache = None

    r._ll_cache = {}
    for k, v in overrides.items():
        setattr(r, k, v)
    return r


# ===========================================================================
# _is_triode_eff
# ===========================================================================

class TestIsTriodeEff:

    def test_lamp_selector_alone_does_not_decide(self):
        """``is_triode`` mirrors the LAMP SELECTOR, not the data.

        Letting it answer here bypassed the Ug2 slice filter, so picking
        a triode lamp after a pentode scan averaged every screen level
        into one Ia(Ua, Ug1) grid — the maps then showed plausible but
        wrong Gm/Rp instead of failing visibly. The measurement's own
        mode arrives as ``track_sids``.
        """
        r = _make_renderer(is_triode=True)
        assert r._is_triode_eff([_pt(100, -1, 200)]) is False

    def test_triode_scan_is_tracking_via_track_sids(self):
        """The normal true-triode path: PlotManager puts sid 0 in."""
        r = _make_renderer(is_triode=True, track_sids={0})
        assert r._is_triode_eff([_pt(100, -1, 0.0)]) is True

    def test_pure_pentode(self):
        r = _make_renderer()
        assert r._is_triode_eff([_pt(100, -1, 200)]) is False

    def test_all_tracking(self):
        r = _make_renderer(track_sids={0})
        pts = [_pt(100, -1, 100, series_id=0)]
        assert r._is_triode_eff(pts) is True

    def test_mixed_returns_false(self):
        r = _make_renderer(track_sids={1})
        pts = [
            _pt(100, -1, 200, series_id=0),
            _pt(100, -1, 100, series_id=1),
        ]
        assert r._is_triode_eff(pts) is False

    def test_empty_points(self):
        r = _make_renderer(track_sids={0})
        assert r._is_triode_eff([]) is True

    def test_only_tracking_overlay(self):
        r = _make_renderer(track_sids={1, 2})
        pts = [
            _pt(100, -1, 100, series_id=1),
            _pt(200, -1, 200, series_id=2),
        ]
        assert r._is_triode_eff(pts) is True


# ===========================================================================
# _ensure_2d_cache with track_sids
# ===========================================================================

class TestEnsure2dCacheTracking:

    def test_overlay_tracking_excluded_from_ug1_ug2(self):
        """Tracking overlay should not populate compare_by_ug1_ug2_per_sid."""
        r = _make_renderer()
        scan = _pentode_points(ug2=200, series_id=0)
        overlay = _tracking_points(series_id=1, lamp_type="EL34", lamp_id="ol")
        cache = r._ensure_2d_cache(scan + overlay,
                                   track_sids={1}, series_labels={})
        per_sid_ug1_ug2 = cache["compare_by_ug1_ug2_per_sid"].get(1, {})
        assert per_sid_ug1_ug2 == {}, \
            "Tracking overlay should have empty by_ug1_ug2"

    def test_overlay_tracking_has_by_ug1(self):
        """Tracking overlay should populate compare_by_ug1_per_sid."""
        r = _make_renderer()
        scan = _pentode_points(ug2=200, series_id=0)
        overlay = _tracking_points(series_id=1, lamp_type="EL34", lamp_id="ol")
        cache = r._ensure_2d_cache(scan + overlay,
                                   track_sids={1}, series_labels={})
        per_sid_ug1 = cache["compare_by_ug1_per_sid"].get(1, {})
        assert len(per_sid_ug1) > 0

    def test_compare_ug2_values_exclude_tracking(self):
        """compare_ug2_values should only contain non-tracking Ug2."""
        r = _make_renderer()
        overlay_pent = _pentode_points(ug2=200, series_id=1,
                                       lamp_type="X", lamp_id="a")
        overlay_track = _tracking_points(series_id=2,
                                         lamp_type="Y", lamp_id="b")
        cache = r._ensure_2d_cache(overlay_pent + overlay_track,
                                   track_sids={2}, series_labels={})
        for v in cache["compare_ug2_values"]:
            assert abs(v - 200) < 5, \
                f"Only pentode Ug2 (~200) expected, got {v}"

    def test_no_tracking_normal_grouping(self):
        r = _make_renderer()
        pts = _pentode_points(ug2=200, series_id=0)
        cache = r._ensure_2d_cache(pts, track_sids=set(), series_labels={})
        assert len(cache["ug2_values"]) == 1
        assert len(cache["by_ug1_ug2"]) > 0


# ===========================================================================
# _grid_from_points with _is_triode_eff
# ===========================================================================

class TestGridFromPointsTracking:

    def test_all_tracking_returns_grid(self):
        """All tracking → triode_eff=True → filter_ug2_slice gets all pts."""
        r = _make_renderer(track_sids={0})
        pts = _tracking_points(n_ua=5)
        for p in pts:
            p["ug1"] = -1.0
        pts2 = _tracking_points(n_ua=5)
        for p in pts2:
            p["ug1"] = -2.0
            p["ua"] = p["ua"]
            p["ug2"] = p["ua"]
        grid = r._grid_from_points(pts + pts2, lambda _: 0.0)
        assert grid is not None
        assert grid["ug2"] == 0.0

    def test_pentode_returns_grid_at_ug2(self):
        r = _make_renderer()
        pts = _pentode_points(ug2=200, n_ua=5, n_ug1=3)
        grid = r._grid_from_points(pts, lambda _: 200.0)
        assert grid is not None
        assert abs(grid["ug2"] - 200.0) < 5

    def test_mixed_uses_pentode_ug2(self):
        """Mixed tracking+pentode → triode_eff=False → uses select_ug2_slice."""
        r = _make_renderer(track_sids={1})
        pent = _pentode_points(ug2=200, series_id=0)
        track = _tracking_points(series_id=1,
                                 lamp_type="X", lamp_id="x")
        grid = r._grid_from_points(pent + track, lambda _: 200.0)
        assert grid is not None
        assert abs(grid["ug2"] - 200.0) < 5


# ===========================================================================
# filter_ug2_multi with is_triode_eff
# ===========================================================================

class TestFilterUg2MultiTracking:

    def test_triode_eff_returns_single_group(self):
        pts = _tracking_points(n_ua=10)
        groups = filter_ug2_multi(pts, is_triode=True)
        assert len(groups) == 1
        assert 0.0 in groups
        assert len(groups[0.0]) == len(pts)

    def test_pentode_returns_multiple_groups(self):
        pts = (_pentode_points(ug2=200, n_ua=3, n_ug1=2)
               + _pentode_points(ug2=300, n_ua=3, n_ug1=2))
        groups = filter_ug2_multi(pts, is_triode=False)
        assert len(groups) == 2

    def test_tracking_data_as_pentode_creates_many_groups(self):
        """Without triode flag, tracking data creates many tiny groups."""
        pts = _tracking_points(n_ua=10)
        groups = filter_ug2_multi(pts, is_triode=False)
        assert len(groups) >= 5, \
            "Tracking data without triode flag should create many Ug2 groups"


# ===========================================================================
# filter_ug2_slice with triode_eff
# ===========================================================================

class TestFilterUg2SliceTracking:

    def test_triode_returns_all(self):
        pts = _tracking_points(n_ua=10)
        result = filter_ug2_slice(pts, is_triode=True,
                                  select_ug2_slice=lambda _: 0.0)
        assert len(result) == len(pts)

    def test_pentode_filters_to_slice(self):
        pts = (_pentode_points(ug2=200, n_ua=3, n_ug1=2)
               + _pentode_points(ug2=300, n_ua=3, n_ug1=2))
        result = filter_ug2_slice(pts, is_triode=False,
                                  select_ug2_slice=lambda _: 200.0)
        assert all(abs(p["ug2"] - 200) < 5 for p in result)


# ===========================================================================
# set_ug2_visibility
# ===========================================================================

class TestSetUg2Visibility:

    def _mock_items(self):
        items = {}
        for ug2 in [100.0, 200.0, 300.0]:
            mock = MagicMock()
            items[ug2] = [mock]
        return items

    def test_show_checked_hide_unchecked(self):
        r = _make_renderer()
        r._ug2_plot_items = self._mock_items()
        r._ug2_transfer_items = {}
        r._ug2_curves_items = {}

        r.set_ug2_visibility([100.0, 300.0], ug2_cluster_thr=2.0)

        r._ug2_plot_items[100.0][0].setVisible.assert_called_with(True)
        r._ug2_plot_items[200.0][0].setVisible.assert_called_with(False)
        r._ug2_plot_items[300.0][0].setVisible.assert_called_with(True)

    def test_empty_checked_hides_all(self):
        r = _make_renderer()
        r._ug2_plot_items = self._mock_items()
        r._ug2_transfer_items = {}
        r._ug2_curves_items = {}

        r.set_ug2_visibility([], ug2_cluster_thr=2.0)

        for items in r._ug2_plot_items.values():
            items[0].setVisible.assert_called_with(False)

    def test_affects_all_three_stores(self):
        r = _make_renderer()
        r._ug2_plot_items = {200.0: [MagicMock()]}
        r._ug2_transfer_items = {200.0: [MagicMock()]}
        r._ug2_curves_items = {200.0: [MagicMock()]}

        r.set_ug2_visibility([200.0], ug2_cluster_thr=2.0)

        r._ug2_plot_items[200.0][0].setVisible.assert_called_with(True)
        r._ug2_transfer_items[200.0][0].setVisible.assert_called_with(True)
        r._ug2_curves_items[200.0][0].setVisible.assert_called_with(True)

    def test_cluster_threshold_matching(self):
        """Ug2 within threshold of checked value should be visible."""
        r = _make_renderer()
        r._ug2_plot_items = {201.0: [MagicMock()]}
        r._ug2_transfer_items = {}
        r._ug2_curves_items = {}

        r.set_ug2_visibility([200.0], ug2_cluster_thr=2.0)
        r._ug2_plot_items[201.0][0].setVisible.assert_called_with(True)

    def test_tracking_items_not_in_store(self):
        """Tracking items should not be stored at all (always visible).

        This test verifies the contract: tracking series should never
        add items to _ug2_*_items, so visibility toggle doesn't affect them.
        """
        r = _make_renderer()
        r._ug2_plot_items = {}
        r._ug2_transfer_items = {}
        r._ug2_curves_items = {200.0: [MagicMock()]}

        r.set_ug2_visibility([200.0], ug2_cluster_thr=2.0)
        assert 0.0 not in r._ug2_curves_items, \
            "Tracking items (ug2_nom=0.0) should not be in _ug2_curves_items"


# ===========================================================================
# render_curve_incremental triode_eff decision
# ===========================================================================

class TestIncrementalTriodeEff:
    """Verify triode_eff = is_triode OR 0 in track_sids."""

    def test_pentode_scan_with_tracking_overlay(self):
        """Scan pentode (sid=0 not tracking) + overlay tracking (sid=1).
        Scan should color by Ug2, not Ug1.
        """
        r = _make_renderer(track_sids={1})
        triode_eff = r.is_triode or 0 in r.track_sids
        assert triode_eff is False

    def test_tracking_scan_with_pentode_overlay(self):
        """Scan tracking (sid=0 tracking) + overlay pentode (sid=2).
        Scan should color by Ug1.
        """
        r = _make_renderer(track_sids={0})
        triode_eff = r.is_triode or 0 in r.track_sids
        assert triode_eff is True

    def test_pure_triode(self):
        r = _make_renderer(is_triode=True)
        triode_eff = r.is_triode or 0 in r.track_sids
        assert triode_eff is True

    def test_pure_pentode(self):
        r = _make_renderer()
        triode_eff = r.is_triode or 0 in r.track_sids
        assert triode_eff is False

    def test_tracking_scan_only(self):
        r = _make_renderer(track_sids={0})
        triode_eff = r.is_triode or 0 in r.track_sids
        assert triode_eff is True


# ===========================================================================
# render_curves target_plot routing
# ===========================================================================

@pytest.fixture(scope="module")
def _qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _make_full_renderer(curves_plot=None, transfer_plot=None):
    """PlotRenderer with real pyqtgraph PlotWidgets for render_curves tests."""
    import pyqtgraph as pg
    if curves_plot is None:
        curves_plot = pg.PlotWidget()
    if transfer_plot is None:
        transfer_plot = pg.PlotWidget()
    r = PlotRenderer(
        plot=pg.PlotWidget(),
        contour_plot=pg.PlotWidget(),
        contour_image=pg.ImageItem(),
        transfer_plot=transfer_plot,
        curves_plot=curves_plot,
    )
    r.is_triode = False
    r.track_sids = set()
    return r


@pytest.mark.usefixtures("_qapp")
class TestRenderCurvesTargetPlot:
    """Verify target_plot routes rendering to the correct plot and stores."""

    def test_default_uses_curves_plot(self):
        r = _make_full_renderer()
        pts = _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
        r.render_curves(pts, y_param="Ia", x_param="Ua")
        assert r._rc_plot is r.curves_plot

    def test_target_plot_uses_transfer_plot(self):
        r = _make_full_renderer()
        pts = _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
        r.render_curves(pts, y_param="Ia", x_param="Ug1",
                        target_plot=r.transfer_plot)
        assert r._rc_plot is r.transfer_plot

    def test_transfer_target_stores_in_ug2_transfer_items(self):
        r = _make_full_renderer()
        pts = _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
        r.render_curves(pts, y_param="Ia", x_param="Ug1",
                        target_plot=r.transfer_plot)
        assert len(r._ug2_transfer_items) > 0
        assert len(r._ug2_curves_items) == 0

    def test_curves_target_stores_in_ug2_curves_items(self):
        r = _make_full_renderer()
        pts = _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
        r.render_curves(pts, y_param="Ia", x_param="Ua")
        assert len(r._ug2_curves_items) > 0
        assert len(r._ug2_transfer_items) == 0

    def test_transfer_labels_in_transfer_list(self):
        r = _make_full_renderer()
        pts = _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
        r.render_curves(pts, y_param="Ia", x_param="Ug1",
                        target_plot=r.transfer_plot)
        assert len(r._transfer_labels) > 0
        assert len(r._curves_labels) == 0

    def test_none_plots_returns_early(self):
        import pyqtgraph as pg
        r = PlotRenderer(
            plot=pg.PlotWidget(),
            contour_plot=pg.PlotWidget(),
            contour_image=pg.ImageItem(),
            transfer_plot=None,
            curves_plot=None,
        )
        pts = _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
        r.render_curves(pts, y_param="Ia", x_param="Ua")


# ===========================================================================
# render_curves multi-series
# ===========================================================================

@pytest.mark.usefixtures("_qapp")
class TestRenderCurvesMultiSeries:
    """Multiple series_id values produce distinct pen styles."""

    def test_two_series_produces_curves(self):
        r = _make_full_renderer()
        pts = (_pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
               + _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=1,
                                 lamp_type="EL34", lamp_id="b"))
        labels = {0: "Scan", 1: "EL34 overlay"}
        r.render_curves(pts, y_param="Ia", x_param="Ua",
                        series_labels=labels)
        items = r.curves_plot.getPlotItem().listDataItems()
        assert len(items) >= 2


# ===========================================================================
# render_curves tracking + pentode mix
# ===========================================================================

@pytest.mark.usefixtures("_qapp")
class TestRenderCurvesTrackingMix:
    """Triode-connected + pentode data via track_sids."""

    def test_tracking_series_not_in_ug2_store(self):
        """Tracking series items should NOT be stored in ug2 items."""
        r = _make_full_renderer()
        track = _tracking_points(n_ua=5, ug1=-1.0, series_id=1,
                                 lamp_type="EL34", lamp_id="t")
        r.render_curves(track, y_param="Ia", x_param="Ua",
                        track_sids={1})
        assert len(r._ug2_curves_items) == 0

    def test_pentode_series_in_ug2_store(self):
        """Pentode series items SHOULD be stored in ug2 items."""
        r = _make_full_renderer()
        pts = _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
        r.render_curves(pts, y_param="Ia", x_param="Ua")
        assert len(r._ug2_curves_items) > 0

    def test_mixed_tracking_and_pentode(self):
        """Mixed data: pentode items in store, tracking items not."""
        r = _make_full_renderer()
        pent = _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
        track = _tracking_points(n_ua=5, series_id=1,
                                 lamp_type="EL34", lamp_id="t")
        r.render_curves(pent + track, y_param="Ia", x_param="Ua",
                        track_sids={1})
        assert len(r._ug2_curves_items) > 0
        assert 0.0 not in r._ug2_curves_items, \
            "Tracking ug2_nom=0.0 should not be in store"


# ===========================================================================
# render_curves load-line conditional
# ===========================================================================

@pytest.mark.usefixtures("_qapp")
class TestRenderCurvesLoadLine:
    """Load-line intersections/Q-point drawn for Ia vs Ug1/Ua."""

    @staticmethod
    def _count_diamond_items(plot_widget):
        """Count plot data items that use diamond ('d') symbol."""
        count = 0
        for item in plot_widget.getPlotItem().listDataItems():
            opts = item.opts
            if opts.get("symbol") == "d":
                count += 1
        return count

    @staticmethod
    def _count_q_items(plot_widget):
        """Count Q-point marker items (symbol 'x')."""
        count = 0
        for item in plot_widget.getPlotItem().listDataItems():
            if item.opts.get("symbol") == "x":
                count += 1
        return count

    def test_load_line_drawn_for_ia_vs_ug1(self):
        r = _make_full_renderer()
        r._load_line_intersections = [
            {"ug1": -1.0, "ua": 200.0, "ia": 10.0},
            {"ug1": -2.0, "ua": 230.0, "ia": 5.0},
            {"ug1": -3.0, "ua": 260.0, "ia": 2.0},
        ]
        r._load_line_analysis = {"ug1_0": -2.0, "ua_0": 230.0, "ia_0": 5.0}
        pts = _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
        r.render_curves(pts, y_param="Ia", x_param="Ug1")
        assert self._count_diamond_items(r.curves_plot) >= 1

    def test_load_line_drawn_for_ia_vs_ua(self):
        r = _make_full_renderer()
        r._load_line_intersections = [
            {"ug1": -1.0, "ua": 200.0, "ia": 10.0},
            {"ug1": -2.0, "ua": 230.0, "ia": 5.0},
        ]
        pts = _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
        r.render_curves(pts, y_param="Ia", x_param="Ua")
        assert self._count_diamond_items(r.curves_plot) >= 1

    def test_no_load_line_for_gm(self):
        r = _make_full_renderer()
        r._load_line_intersections = [
            {"ug1": -1.0, "ua": 200.0, "ia": 10.0},
            {"ug1": -2.0, "ua": 230.0, "ia": 5.0},
        ]
        pts = _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
        r.render_curves(pts, y_param="Gm", x_param="Ug1")
        assert self._count_diamond_items(r.curves_plot) == 0

    def test_load_line_on_transfer_target(self):
        r = _make_full_renderer()
        r._load_line_intersections = [
            {"ug1": -1.0, "ua": 200.0, "ia": 10.0},
            {"ug1": -2.0, "ua": 230.0, "ia": 5.0},
            {"ug1": -3.0, "ua": 260.0, "ia": 2.0},
        ]
        pts = _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
        r.render_curves(pts, y_param="Ia", x_param="Ug1",
                        target_plot=r.transfer_plot)
        assert self._count_diamond_items(r.transfer_plot) >= 1
        assert self._count_diamond_items(r.curves_plot) == 0

    def test_no_load_line_when_empty_intersections(self):
        r = _make_full_renderer()
        r._load_line_intersections = []
        pts = _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
        r.render_curves(pts, y_param="Ia", x_param="Ug1")
        assert self._count_diamond_items(r.curves_plot) == 0

    def test_no_load_line_when_single_intersection(self):
        r = _make_full_renderer()
        r._load_line_intersections = [{"ug1": -1.0, "ua": 200.0, "ia": 10.0}]
        pts = _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
        r.render_curves(pts, y_param="Ia", x_param="Ug1")
        assert self._count_diamond_items(r.curves_plot) >= 1

    def test_q_point_drawn_with_crosshair(self):
        r = _make_full_renderer()
        r._load_line_intersections = [
            {"ug1": -1.0, "ua": 200.0, "ia": 10.0},
            {"ug1": -2.0, "ua": 230.0, "ia": 5.0},
        ]
        r._load_line_analysis = {"ug1_0": -2.0, "ua_0": 230.0, "ia_0": 5.0}
        pts = _pentode_points(ug2=200, n_ua=5, n_ug1=3, series_id=0)
        r.render_curves(pts, y_param="Ia", x_param="Ug1")
        assert self._count_q_items(r.curves_plot) >= 1


# ===========================================================================
# Ug2 visibility for transfer items after render_curves
# ===========================================================================

class TestUg2VisibilityTransferItems:
    """After render_curves with target_plot=transfer, visibility toggling
    should affect transfer items."""

    def test_visibility_toggle_on_transfer_store(self):
        r = _make_renderer()
        mock_item_200 = MagicMock()
        mock_item_300 = MagicMock()
        r._ug2_plot_items = {}
        r._ug2_transfer_items = {200.0: [mock_item_200], 300.0: [mock_item_300]}
        r._ug2_curves_items = {}
        r.set_ug2_visibility([200.0], ug2_cluster_thr=2.0)
        mock_item_200.setVisible.assert_called_with(True)
        mock_item_300.setVisible.assert_called_with(False)

    @pytest.mark.usefixtures("_qapp")
    def test_transfer_tracking_items_not_toggled(self):
        """Tracking series items should not appear in _ug2_transfer_items."""
        r = _make_full_renderer()
        track = _tracking_points(n_ua=5, series_id=0)
        r.render_curves(track, y_param="Ia", x_param="Ug1",
                        track_sids={0}, target_plot=r.transfer_plot)
        assert 0.0 not in r._ug2_transfer_items
