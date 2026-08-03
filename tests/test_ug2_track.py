"""Tests for Ug2-tracking flag propagation.

Covers:
- PlotManager._is_sid_ug2_track / _get_track_sids (mocked, no Qt)
- build_curves_2d with mixed track_sids
- compare_tab series_ug2_track construction logic
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.plotting.grouping import build_curves_2d
from lm19.measurements import get_ug2_mode
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)


# ---------------------------------------------------------------------------
# Helper: lightweight PlotManager mock (no Qt import needed)
# ---------------------------------------------------------------------------

def _make_plot_manager(is_triode=False, ug2_track_radio_checked=False,
                       series_ug2_track=None):
    """Build a minimal PlotManager-like object with _is_sid_ug2_track logic."""
    from app.plot_manager import PlotManager
    pm = object.__new__(PlotManager)
    pm.is_triode = is_triode
    pm.series_ug2_track = series_ug2_track or {}
    pm.renderer = MagicMock()

    radio = MagicMock()
    radio.isChecked.return_value = ug2_track_radio_checked
    pm.w = {"ug2_track_radio": radio}
    return pm


# ---------------------------------------------------------------------------
# Tests: _is_sid_ug2_track
# ---------------------------------------------------------------------------

class TestIsSidUg2Track:
    """PlotManager._is_sid_ug2_track: per-series track flag resolution."""

    def test_sid0_triode_returns_true(self):
        pm = _make_plot_manager(is_triode=True)
        assert pm._is_sid_ug2_track(0) is True

    def test_sid0_radio_checked_returns_true(self):
        pm = _make_plot_manager(ug2_track_radio_checked=True)
        assert pm._is_sid_ug2_track(0) is True

    def test_sid0_pentode_radio_unchecked_returns_false(self):
        pm = _make_plot_manager(is_triode=False, ug2_track_radio_checked=False)
        assert pm._is_sid_ug2_track(0) is False

    def test_sid0_no_radio_widget_returns_false(self):
        pm = _make_plot_manager()
        pm.w = {}
        assert pm._is_sid_ug2_track(0) is False

    def test_overlay_sid_tracked(self):
        pm = _make_plot_manager(series_ug2_track={1: True, 2: False})
        assert pm._is_sid_ug2_track(1) is True

    def test_overlay_sid_not_tracked(self):
        pm = _make_plot_manager(series_ug2_track={1: True, 2: False})
        assert pm._is_sid_ug2_track(2) is False

    def test_overlay_sid_unknown_defaults_false(self):
        pm = _make_plot_manager(series_ug2_track={1: True})
        assert pm._is_sid_ug2_track(99) is False


# ---------------------------------------------------------------------------
# Tests: the recorded scan flag outranks the scan-setup widgets
# ---------------------------------------------------------------------------

class TestScanTrackFlagRecorded:
    """sid 0 grouping follows the data on the plot, not the next run.

    The scan-setup radio and the lamp selector describe the run being
    ARMED; the plotted curves were measured in whatever mode was active
    when they were taken.  Re-arming the other mode must not regroup
    them (a triode-connected scan regrouped by (Ug1, Ug2) has one point
    per group and renders as loose symbols instead of curves).
    """

    def test_recorded_track_outranks_unchecked_radio(self):
        pm = _make_plot_manager(ug2_track_radio_checked=False,
                                series_ug2_track={0: True})
        assert pm._is_sid_ug2_track(0) is True

    def test_recorded_pentode_outranks_checked_radio(self):
        pm = _make_plot_manager(ug2_track_radio_checked=True,
                                series_ug2_track={0: False})
        assert pm._is_sid_ug2_track(0) is False

    def test_recorded_pentode_outranks_triode_lamp_selection(self):
        """Selecting a triode lamp after a pentode scan must not regroup it."""
        pm = _make_plot_manager(is_triode=True, series_ug2_track={0: False})
        assert pm._is_sid_ug2_track(0) is False

    def test_fallback_to_radio_before_any_scan(self):
        pm = _make_plot_manager(ug2_track_radio_checked=True)
        assert pm._is_sid_ug2_track(0) is True

    def test_fallback_to_triode_lamp_before_any_scan(self):
        pm = _make_plot_manager(is_triode=True)
        assert pm._is_sid_ug2_track(0) is True

    def test_get_track_sids_uses_recorded_flag(self):
        """The recorded flag reaches _get_track_sids, not just the predicate."""
        pm = _make_plot_manager(ug2_track_radio_checked=False,
                                series_ug2_track={0: True})
        pts = [{"series_id": 0, "ua": 100, "ug2": 100}]
        assert pm._get_track_sids(pts) == {0}

    def test_set_scan_ug2_track_records_and_invalidates(self):
        pm = _make_plot_manager(ug2_track_radio_checked=False)
        pm.set_scan_ug2_track(True)
        assert pm.series_ug2_track[0] is True
        assert pm._is_sid_ug2_track(0) is True
        pm.renderer.invalidate_cache.assert_called_once()

    def test_set_scan_ug2_track_stores_bool_not_truthy(self):
        """A falsy record must stay distinguishable from "no record"."""
        pm = _make_plot_manager(ug2_track_radio_checked=True)
        pm.set_scan_ug2_track(0)
        assert pm.series_ug2_track[0] is False
        assert pm._is_sid_ug2_track(0) is False

    def test_set_scan_ug2_track_leaves_overlays_alone(self):
        pm = _make_plot_manager(series_ug2_track={1: True, 2: False})
        pm.set_scan_ug2_track(False)
        assert pm.series_ug2_track == {0: False, 1: True, 2: False}


# ---------------------------------------------------------------------------
# Tests: _get_track_sids
# ---------------------------------------------------------------------------

class TestGetTrackSids:
    """PlotManager._get_track_sids: collect tracking sids from points."""

    def test_empty_points(self):
        pm = _make_plot_manager()
        assert pm._get_track_sids([]) == set()

    def test_scan_only_pentode(self):
        pm = _make_plot_manager(is_triode=False, ug2_track_radio_checked=False)
        pts = [{"series_id": 0, "ua": 100, "ug2": 200}]
        assert pm._get_track_sids(pts) == set()

    def test_scan_only_triode(self):
        pm = _make_plot_manager(is_triode=True)
        pts = [{"series_id": 0, "ua": 100}]
        assert pm._get_track_sids(pts) == {0}

    def test_scan_plus_overlay_mixed(self):
        pm = _make_plot_manager(
            is_triode=False,
            ug2_track_radio_checked=True,
            series_ug2_track={1: False, 2: True},
        )
        pts = [
            {"series_id": 0, "ua": 100},
            {"series_id": 1, "ua": 100},
            {"series_id": 2, "ua": 100},
        ]
        result = pm._get_track_sids(pts)
        assert result == {0, 2}

    def test_only_overlay_series(self):
        pm = _make_plot_manager(series_ug2_track={1: True, 2: False, 3: True})
        pts = [
            {"series_id": 1, "ua": 100},
            {"series_id": 2, "ua": 100},
            {"series_id": 3, "ua": 100},
        ]
        result = pm._get_track_sids(pts)
        assert result == {1, 3}

    def test_default_series_id_zero(self):
        """Points without explicit series_id default to 0."""
        pm = _make_plot_manager(is_triode=True)
        pts = [{"ua": 100, "ia": 5}]
        assert pm._get_track_sids(pts) == {0}


# ---------------------------------------------------------------------------
# Tests: build_curves_2d with mixed track_sids
# ---------------------------------------------------------------------------

class TestBuildCurves2dMixed:
    """build_curves_2d: mixed pentode + triode-connected series."""

    def test_mixed_pentode_and_track(self):
        """Pentode series groups by Ug2; track series collapses Ug2."""
        pentode_pts = [
            {"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 200, "series_id": 1},
            {"ua": 200, "ia": 10.0, "ug1": -1.0, "ug2": 200, "series_id": 1},
            {"ua": 100, "ia": 4.0, "ug1": -1.0, "ug2": 250, "series_id": 1},
            {"ua": 200, "ia": 8.0, "ug1": -1.0, "ug2": 250, "series_id": 1},
        ]
        track_pts = [
            {"ua": 100, "ia": 6.0, "ug1": -1.0, "ug2": 100, "series_id": 2},
            {"ua": 200, "ia": 11.0, "ug1": -1.0, "ug2": 200, "series_id": 2},
        ]
        all_pts = pentode_pts + track_pts
        curves = build_curves_2d(all_pts, track_sids={2})

        sid1_curves = [c for c in curves if any(
            p.get("series_id") == 1
            for p in pentode_pts
            if abs(p["ug1"] - float(c.extra["Ug1"][0])) < 0.01
        )]
        sid2_curves = [c for c in curves if len(c.x) == 2 and
                       abs(float(c.extra["Ug2"][0]) - 100) < 1 or
                       abs(float(c.extra["Ug2"][1]) - 200) < 1]

        pentode_count = sum(1 for c in curves
                            if len(c.x) == 2 and
                            abs(float(c.extra["Ug2"][0]) - float(c.extra["Ug2"][1])) < 1)
        track_count = sum(1 for c in curves
                          if len(c.x) == 2 and
                          abs(float(c.extra["Ug2"][0]) - float(c.extra["Ug2"][1])) > 1)
        assert pentode_count == 2, "Pentode series should have 2 curves (Ug2=200, Ug2=250)"
        assert track_count == 1, "Track series should have 1 curve (Ug2 collapsed)"

    def test_all_tracked(self):
        """When all series are tracked, every series collapses Ug2."""
        pts = [
            {"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 100, "series_id": 0},
            {"ua": 200, "ia": 10.0, "ug1": -1.0, "ug2": 200, "series_id": 0},
            {"ua": 100, "ia": 6.0, "ug1": -1.0, "ug2": 100, "series_id": 1},
            {"ua": 200, "ia": 11.0, "ug1": -1.0, "ug2": 200, "series_id": 1},
        ]
        curves = build_curves_2d(pts, track_sids={0, 1})
        assert len(curves) == 2, "Each series gets one curve (Ug1=-1, Ug2 collapsed)"

    def test_no_tracked(self):
        """Without tracking, all Ug2 values create separate groups."""
        pts = [
            {"ua": 100, "ia": 5.0, "ug1": -1.0, "ug2": 200, "series_id": 0},
            {"ua": 200, "ia": 10.0, "ug1": -1.0, "ug2": 200, "series_id": 0},
            {"ua": 100, "ia": 4.0, "ug1": -1.0, "ug2": 250, "series_id": 0},
            {"ua": 200, "ia": 8.0, "ug1": -1.0, "ug2": 250, "series_id": 0},
        ]
        curves = build_curves_2d(pts)
        assert len(curves) == 2


# ---------------------------------------------------------------------------
# Tests: compare_tab ug2_mode → series_ug2_track mapping
# ---------------------------------------------------------------------------

class TestCompareUg2TrackFlag:
    """Verify ug2_mode → is_track mapping used in compare_tab emit."""

    @pytest.mark.parametrize("ug2_mode,expected", [
        ("triode", True),
        ("triode_connected", True),
        ("pentode", False),
    ])
    def test_ug2_mode_to_track_flag(self, ug2_mode, expected):
        """The mapping matches compare_tab logic: triode/triode_connected → track."""
        is_track = ug2_mode in (TOPOLOGY_TRIODE, TOPOLOGY_TRIODE_CONNECTED)
        assert is_track is expected

    def test_get_ug2_mode_triode_entry(self):
        entry = {"data": {"topology": TOPOLOGY_TRIODE}}
        mode = get_ug2_mode(entry)
        assert mode in (TOPOLOGY_TRIODE, TOPOLOGY_TRIODE_CONNECTED)
        assert mode == TOPOLOGY_TRIODE

    def test_get_ug2_mode_triode_connected_entry(self):
        entry = {"data": {"scan": {"ug2_track_ua": True}}}
        mode = get_ug2_mode(entry)
        assert mode == TOPOLOGY_TRIODE_CONNECTED

    def test_get_ug2_mode_pentode_entry(self):
        entry = {"data": {"scan": {"ug2_mode": TOPOLOGY_PENTODE}}}
        mode = get_ug2_mode(entry)
        assert mode == TOPOLOGY_PENTODE

    def test_mixed_entries_produce_correct_dict(self):
        """Simulate compare_tab logic for multiple entries."""
        entries = [
            {"data": {"scan": {"ug2_mode": TOPOLOGY_PENTODE}}},
            {"data": {"scan": {"ug2_track_ua": True}}},
            {"data": {"topology": TOPOLOGY_TRIODE}},
        ]
        series_ug2_track = {}
        for sid, entry in enumerate(entries, start=1):
            ug2_mode = get_ug2_mode(entry)
            series_ug2_track[sid] = ug2_mode in (TOPOLOGY_TRIODE, TOPOLOGY_TRIODE_CONNECTED)

        assert series_ug2_track == {1: False, 2: True, 3: True}
