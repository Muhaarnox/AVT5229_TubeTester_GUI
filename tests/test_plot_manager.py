"""Tests for PlotManager state management and selector wiring.

Pure helpers (filter_by_series, filter_by_ug2_display, filter_by_calc_series)
are covered by ``test_selectors.py``.  This file targets the methods on the
PlotManager class itself: triode state propagation, ug2-tracking sid
resolution, display/calc combo refresh (with prev-selection restoration),
and render-dispatch logic.

PlotManager is constructed via ``object.__new__`` (skipping __init__) and
populated with MagicMock UI widgets — same approach as
``test_load_line_ug2_filter.py``.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.plot_manager import PlotManager


# ── Module local helpers ──

def _val(v):
    w = MagicMock()
    w.value.return_value = v
    return w


def _check(v):
    w = MagicMock()
    w.isChecked.return_value = v
    return w


def _pt(ua: float, ug1: float, ia: float, ug2: float = 0.0,
        series_id: int = 0) -> dict:
    return {
        "ua": ua, "ug1": ug1, "ia": ia, "ug2": ug2, "ig2": 0.1,
        "uh": 6.3, "ih": 0.3, "series_id": series_id,
    }


def _make_pm(*, is_triode: bool = False,
             ug2_track_checked: bool = False,
             series_ug2_track: dict = None) -> PlotManager:
    """Build a PlotManager with mocked renderer and minimal widget set."""
    pm = object.__new__(PlotManager)
    pm.points = []
    pm.is_triode = is_triode
    pm.legend_hidden = False
    pm.series_labels = {}
    pm.series_colors = {}
    pm.series_ug2_track = series_ug2_track or {}
    pm.series_models = {}
    pm.series_grids = {}
    pm.current_curve_points = []
    pm.labeled_ug1 = set()
    pm.working_line_reattach = None

    pm.renderer = MagicMock()
    pm.renderer.is_triode = is_triode
    pm.renderer.ug2_cluster_thr = 2.0

    pm.w = {
        "ug2_track_radio": _check(ug2_track_checked),
        "lamp_display_combo": None,
        "lamp_calc_combo": None,
        "ug2_display_combo": None,
        "ug2_calc_combo": None,
        "_source_label": None,
        "_calc_lamp_label": None,
        "_ug2_display_label": None,
        "_ug2_calc_label": None,
    }
    return pm


# ----------------------------------------------------------------------
# Triode state
# ----------------------------------------------------------------------


class TestSetTriode(unittest.TestCase):

    def test_propagates_to_renderer_and_invalidates_cache(self):
        pm = _make_pm(is_triode=False)
        pm.set_triode(True)
        self.assertTrue(pm.is_triode)
        self.assertTrue(pm.renderer.is_triode)
        pm.renderer.invalidate_cache.assert_called_once()

    def test_invalidate_cache_delegates(self):
        pm = _make_pm()
        pm.invalidate_cache()
        pm.renderer.invalidate_cache.assert_called_once()


# ----------------------------------------------------------------------
# Ug2 tracking series resolution
# ----------------------------------------------------------------------


class TestTrackSidResolution(unittest.TestCase):

    def test_main_scan_triode_is_tracking(self):
        pm = _make_pm(is_triode=True)
        self.assertTrue(pm._is_sid_ug2_track(0))

    def test_main_scan_pentode_with_radio_checked_is_tracking(self):
        pm = _make_pm(is_triode=False, ug2_track_checked=True)
        self.assertTrue(pm._is_sid_ug2_track(0))

    def test_main_scan_pentode_with_radio_unchecked_is_not_tracking(self):
        pm = _make_pm(is_triode=False, ug2_track_checked=False)
        self.assertFalse(pm._is_sid_ug2_track(0))

    def test_overlay_uses_series_ug2_track_dict(self):
        pm = _make_pm(series_ug2_track={1: True, 2: False})
        self.assertTrue(pm._is_sid_ug2_track(1))
        self.assertFalse(pm._is_sid_ug2_track(2))
        # Unknown series defaults to False
        self.assertFalse(pm._is_sid_ug2_track(99))

    def test_get_track_sids_returns_only_tracking_series(self):
        pm = _make_pm(is_triode=False, ug2_track_checked=False,
                      series_ug2_track={1: True, 2: False})
        points = [_pt(100, -1, 5, 250, sid) for sid in (0, 1, 2)]
        track = pm._get_track_sids(points)
        self.assertEqual(track, {1})

    def test_sync_track_sids_pushes_to_renderer(self):
        pm = _make_pm(series_ug2_track={1: True})
        pm.points = [_pt(100, -1, 5, 250, 1)]
        pm._sync_track_sids()
        self.assertEqual(pm.renderer.track_sids, {1})


# ----------------------------------------------------------------------
# Lamp combo refresh — bidirectional sync (display + calc)
# ----------------------------------------------------------------------


class _StubModel:
    """Minimal QStandardItemModel-shaped stub for CheckableComboBox."""

    def __init__(self) -> None:
        self.rows = []

    def clear(self) -> None:
        self.rows = []

    def appendRow(self, item) -> None:
        self.rows.append(item)

    def rowCount(self) -> int:
        return len(self.rows)


class _StubCheckable:
    """Minimal stand-in for the CheckableComboBox display widget."""

    def __init__(self) -> None:
        self._model = _StubModel()
        self._updating = False
        self._signals_blocked = False
        self._visible = False
        self._text_updates = 0

    def blockSignals(self, b: bool) -> None:
        self._signals_blocked = b

    def setVisible(self, v: bool) -> None:
        self._visible = v

    def isVisible(self) -> bool:
        return self._visible

    def _update_text(self) -> None:
        self._text_updates += 1


class _StubCalcCombo:
    """Minimal stand-in for the lamp_calc QComboBox."""

    def __init__(self) -> None:
        self._items = []  # (text, userData)
        self._current = -1
        self._visible = False
        self._signals_blocked = False

    def blockSignals(self, b: bool) -> None:
        self._signals_blocked = b

    def setVisible(self, v: bool) -> None:
        self._visible = v

    def clear(self) -> None:
        self._items = []
        self._current = -1

    def addItem(self, text: str, userData=None) -> None:
        self._items.append((text, userData))
        if self._current < 0:
            self._current = 0

    def count(self) -> int:
        return len(self._items)

    def currentText(self) -> str:
        if 0 <= self._current < len(self._items):
            return self._items[self._current][0]
        return ""

    def currentData(self):
        if 0 <= self._current < len(self._items):
            return self._items[self._current][1]
        return None

    def itemData(self, i: int):
        return self._items[i][1] if 0 <= i < len(self._items) else None

    def itemText(self, i: int) -> str:
        return self._items[i][0] if 0 <= i < len(self._items) else ""

    def findText(self, text: str) -> int:
        for i, (t, _) in enumerate(self._items):
            if t == text:
                return i
        return -1

    def setCurrentIndex(self, i: int) -> None:
        if 0 <= i < len(self._items):
            self._current = i


class TestRefreshLampCombos(unittest.TestCase):

    def _setup(self, *, with_overlay: bool = False):
        pm = _make_pm()
        pm.w["lamp_display_combo"] = _StubCheckable()
        pm.w["lamp_calc_combo"] = _StubCalcCombo()
        pm.w["_source_label"] = MagicMock()
        pm.w["_calc_lamp_label"] = MagicMock()
        pm.points = [_pt(100, -1, 5, 250, 0)]
        if with_overlay:
            pm.points.append(_pt(150, -2, 6, 250, 1))
            pm.series_labels[1] = "EL84-overlay"
        return pm

    def test_display_combo_populated_from_series(self):
        pm = self._setup(with_overlay=True)
        pm.refresh_lamp_combos()
        display = pm.w["lamp_display_combo"]
        self.assertEqual(display._model.rowCount(), 2)
        # Always visible (UI no-jump policy)
        self.assertTrue(display._visible)

    def test_calc_combo_populated_with_user_data(self):
        pm = self._setup(with_overlay=True)
        pm.refresh_lamp_combos()
        calc = pm.w["lamp_calc_combo"]
        self.assertEqual(calc.count(), 2)
        # Each item carries its series_id as userData
        sids = {calc.itemData(i) for i in range(calc.count())}
        self.assertEqual(sids, {0, 1})

    def test_calc_combo_restores_previous_selection(self):
        """Previous data selection survives refresh when still present."""
        pm = self._setup(with_overlay=True)
        pm.refresh_lamp_combos()
        calc = pm.w["lamp_calc_combo"]
        # Select overlay series (sid=1)
        for i in range(calc.count()):
            if calc.itemData(i) == 1:
                calc.setCurrentIndex(i)
                break
        # Refresh again — selection sid=1 must remain
        pm.refresh_lamp_combos()
        self.assertEqual(calc.currentData(), 1)

    def test_calc_combo_falls_back_to_first_when_prev_gone(self):
        """When the previously-selected series disappears, default to index 0."""
        pm = self._setup(with_overlay=True)
        pm.refresh_lamp_combos()
        calc = pm.w["lamp_calc_combo"]
        for i in range(calc.count()):
            if calc.itemData(i) == 1:
                calc.setCurrentIndex(i)
                break
        # Remove overlay; refresh must fall back to series 0
        pm.points = [_pt(100, -1, 5, 250, 0)]
        pm.series_labels.pop(1, None)
        pm.refresh_lamp_combos()
        self.assertEqual(calc.currentData(), 0)


# ----------------------------------------------------------------------
# Ug2 combo refresh — clustering + selection restore
# ----------------------------------------------------------------------


class _StubUg2Display:
    """Stand-in for the ug2_display CheckableComboBox."""

    def __init__(self) -> None:
        self._items = []
        self._visible = False

    def set_items(self, values: list) -> None:
        self._items = list(values)

    def setVisible(self, v: bool) -> None:
        self._visible = v


class TestRefreshUg2Combos(unittest.TestCase):

    def _setup(self):
        pm = _make_pm()
        pm.w["ug2_display_combo"] = _StubUg2Display()
        pm.w["ug2_calc_combo"] = _StubCalcCombo()
        pm.w["_ug2_display_label"] = MagicMock()
        pm.w["_ug2_calc_label"] = MagicMock()
        return pm

    def test_clusters_close_ug2_values(self):
        """Two readings 100.0/100.5 collapse to a single nominal."""
        pm = self._setup()
        pts = [
            _pt(100, -1, 5, 100.0),
            _pt(150, -1, 6, 100.5),
            _pt(100, -1, 5, 250.0),
        ]
        pm.refresh_ug2_combos(pts)
        display = pm.w["ug2_display_combo"]
        self.assertEqual(len(display._items), 2)

    def test_skips_tracking_series_from_ug2_options(self):
        """Series marked as ug2-tracking shouldn't contribute Ug2 setpoints."""
        pm = _make_pm(series_ug2_track={1: True})
        pm.w["ug2_display_combo"] = _StubUg2Display()
        pm.w["ug2_calc_combo"] = _StubCalcCombo()
        pm.w["_ug2_display_label"] = MagicMock()
        pm.w["_ug2_calc_label"] = MagicMock()
        pts = [
            _pt(100, -1, 5, 250.0, series_id=0),
            _pt(100, -1, 5, 88.0, series_id=1),  # tracking → ignored
        ]
        pm.refresh_ug2_combos(pts)
        display = pm.w["ug2_display_combo"]
        self.assertEqual(display._items, [250.0])

    def test_empty_points_still_makes_combos_visible(self):
        """No-data state must not hide combos (UI no-jump policy)."""
        pm = self._setup()
        pm.refresh_ug2_combos([])
        self.assertTrue(pm.w["ug2_display_combo"]._visible)
        self.assertTrue(pm.w["ug2_calc_combo"]._visible)

    def test_calc_combo_preserves_text_selection_across_refresh(self):
        pm = self._setup()
        pts1 = [_pt(100, -1, 5, 100.0), _pt(100, -1, 5, 250.0)]
        pm.refresh_ug2_combos(pts1)
        calc = pm.w["ug2_calc_combo"]
        # Select 250 V
        idx = calc.findText("250")
        self.assertGreaterEqual(idx, 0)
        calc.setCurrentIndex(idx)
        # Refresh with same set → selection survives
        pm.refresh_ug2_combos(pts1)
        self.assertEqual(calc.currentText(), "250")


# ----------------------------------------------------------------------
# select_ug2_calc — defensive parsing
# ----------------------------------------------------------------------


class TestSelectUg2Calc(unittest.TestCase):

    def test_returns_zero_when_combo_missing(self):
        pm = _make_pm()
        self.assertEqual(pm.select_ug2_calc([]), 0.0)

    def test_returns_zero_when_combo_empty(self):
        pm = _make_pm()
        pm.w["ug2_calc_combo"] = _StubCalcCombo()
        self.assertEqual(pm.select_ug2_calc([]), 0.0)

    def test_parses_current_text(self):
        pm = _make_pm()
        calc = _StubCalcCombo()
        calc.addItem("250")
        pm.w["ug2_calc_combo"] = calc
        self.assertEqual(pm.select_ug2_calc([]), 250.0)

    def test_returns_zero_when_text_unparseable(self):
        pm = _make_pm()
        calc = _StubCalcCombo()
        calc.addItem("not-a-number")
        pm.w["ug2_calc_combo"] = calc
        self.assertEqual(pm.select_ug2_calc([]), 0.0)


# ----------------------------------------------------------------------
# Render dispatchers — early-exit and delegation
# ----------------------------------------------------------------------


class TestRenderDispatchers(unittest.TestCase):

    def _setup(self, *, with_points: bool = True):
        pm = _make_pm()
        for name in ("_render_2d", "_render_transfer", "_render_contour",
                     "_render_gm_rp", "_render_pa_map", "_render_curves",
                     "_apply_ug2_visibility", "_update_amplifier_tab",
                     "_sync_track_sids"):
            setattr(pm, name, MagicMock())
        # A few zone widgets touched by render_all when no points
        for k in ("pa_max_cb", "ua_max_cb", "ia_max_limit_cb", "load_line_cb"):
            pm.w[k] = _check(False)
        if with_points:
            pm.points = [_pt(100, -1, 5, 250, 0)]
        return pm

    def test_render_all_skips_everything_when_no_points_and_no_overlays(self):
        pm = self._setup(with_points=False)
        pm.render_all()
        pm._render_2d.assert_not_called()
        pm._render_transfer.assert_not_called()
        # Amplifier tab still updated so it can show "no data"
        pm._update_amplifier_tab.assert_called_once()

    def test_render_all_renders_zones_only_when_overlay_checked(self):
        """Empty points but an overlay checkbox on → 2D zone is drawn.
        The line is no longer a renderer overlay (controller items) —
        the guard counts pa/ua/ia limit lines."""
        pm = self._setup(with_points=False)
        pm.w["pa_max_cb"] = _check(True)
        pm.render_all()
        pm._render_2d.assert_called_once_with([])

    def test_render_all_dispatches_to_all_renderers_when_points_present(self):
        pm = self._setup(with_points=True)
        pm.render_all()
        pm._sync_track_sids.assert_called_once()
        pm._render_2d.assert_called_once()
        pm._render_transfer.assert_called_once()
        pm._render_contour.assert_called_once()
        pm._render_gm_rp.assert_called_once()
        pm._render_pa_map.assert_called_once()
        pm._render_curves.assert_called_once()
        pm.renderer.draw_qpoint_all.assert_called_once()

    def test_render_curves_only_calls_only_curves(self):
        pm = self._setup(with_points=True)
        pm.render_curves_only()
        pm._render_curves.assert_called_once()
        pm._render_2d.assert_not_called()
        pm._render_transfer.assert_not_called()

    def test_render_2d_and_pa_calls_only_those(self):
        pm = self._setup(with_points=True)
        pm.render_2d_and_pa()
        pm._render_2d.assert_called_once()
        pm._render_pa_map.assert_called_once()
        pm._render_transfer.assert_not_called()
        pm._render_curves.assert_not_called()


class TestRender2dTrackSidsCallSite(unittest.TestCase):
    """``_render_2d`` must hand the renderer the RECORDED track sids.

    Pinning ``_is_sid_ug2_track`` alone does not prove the call site
    forwards its verdict: the scenario below is one where the recorded
    flag and the scan-setup radio DISAGREE, so a call site that re-read
    the radio (or passed nothing) produces a different argument.
    """

    def _setup(self, *, ug2_track_checked: bool):
        pm = _make_pm(ug2_track_checked=ug2_track_checked)
        for k in ("zone_ua_min", "zone_ua_max", "zone_ug1_min", "zone_ug1_max",
                  "pa_max_input", "pg2_max_input", "ua_max_input",
                  "ia_max_limit_input", "plot_line_width"):
            pm.w[k] = _val(1.0)
        for k in ("pa_max_cb", "pg2_max_cb", "ua_max_cb", "ia_max_limit_cb",
                  "ug2_mode_series"):
            pm.w[k] = _check(False)
        overlay = MagicMock()
        overlay.currentIndex.return_value = 0
        pm.w["overlay_pen_style"] = overlay
        pm.points = [_pt(100, -1, 5, 100, 0), _pt(200, -1, 9, 200, 0)]
        return pm

    def test_recorded_track_reaches_renderer_despite_unchecked_radio(self):
        pm = self._setup(ug2_track_checked=False)
        pm.set_scan_ug2_track(True)
        pm._render_2d(pm.points)
        kwargs = pm.renderer.render_plot_2d.call_args.kwargs
        self.assertEqual(kwargs["track_sids"], {0})

    def test_recorded_pentode_reaches_renderer_despite_checked_radio(self):
        pm = self._setup(ug2_track_checked=True)
        pm.set_scan_ug2_track(False)
        pm._render_2d(pm.points)
        kwargs = pm.renderer.render_plot_2d.call_args.kwargs
        self.assertEqual(kwargs["track_sids"], set())

    def test_sync_track_sids_uses_recorded_flag(self):
        """The renderer attribute (read by incremental/marker paths) too."""
        pm = self._setup(ug2_track_checked=False)
        pm.set_scan_ug2_track(True)
        pm._sync_track_sids()
        self.assertEqual(pm.renderer.track_sids, {0})


# ----------------------------------------------------------------------
# Overlay series swap ("Show on main plot" from Compare)
# ----------------------------------------------------------------------


class TestReplaceOverlaySeries(unittest.TestCase):
    """The swap replaces overlays only — the current scan keeps its own
    points AND its bookkeeping, and dropped model overlays take their
    model/grid entries with them."""

    def _make(self):
        pm = _make_pm(ug2_track_checked=False)
        pm.points = [_pt(100, -1, 5, 0, 0), _pt(200, -1, 9, 0, 0),
                     _pt(100, -2, 4, 250, 1)]
        pm.series_labels = {0: "scan", 1: "old overlay"}
        pm.series_colors = {0: "#111111", 1: "#222222"}
        pm.series_ug2_track = {0: True, 1: False}
        pm.series_models = {1: object()}
        pm.series_grids = {1: object()}
        return pm

    def test_scan_points_kept_and_overlay_points_replaced(self):
        pm = self._make()
        new_pts = [_pt(150, -3, 7, 300, 1), _pt(250, -3, 11, 300, 1)]
        pm.replace_overlay_series(new_pts, {1: "new"}, {1: "#333333"},
                                  {1: True})
        self.assertEqual([p["series_id"] for p in pm.points], [0, 0, 1, 1])
        self.assertEqual([p["ua"] for p in pm.points if p["series_id"] == 1],
                         [150, 250])

    def test_scan_track_flag_survives_the_swap(self):
        """Discriminator: the incoming dict has no sid 0, and the radio
        says "not tracking" — a wholesale replace would flip the scan."""
        pm = self._make()
        pm.replace_overlay_series([], {1: "new"}, {}, {1: True})
        self.assertIs(pm.series_ug2_track[0], True)
        self.assertTrue(pm._is_sid_ug2_track(0))

    def test_scan_label_and_color_survive_the_swap(self):
        pm = self._make()
        pm.replace_overlay_series([], {1: "new"}, {1: "#333333"}, {1: True})
        self.assertEqual(pm.series_labels[0], "scan")
        self.assertEqual(pm.series_colors[0], "#111111")

    def test_overlay_metadata_fully_replaced(self):
        pm = self._make()
        pm.replace_overlay_series([], {2: "new"}, {2: "#333333"}, {2: True})
        self.assertEqual(pm.series_labels.get(1), None)
        self.assertEqual(pm.series_labels[2], "new")
        self.assertEqual(pm.series_colors[2], "#333333")
        self.assertIs(pm.series_ug2_track[2], True)

    def test_model_bookkeeping_dropped_for_reused_sids(self):
        """A stale series_models entry would render an incoming compare
        series as a dashed model curve driven by the wrong model."""
        pm = self._make()
        pm.replace_overlay_series([_pt(150, -3, 7, 300, 1)], {1: "new"},
                                  {}, {1: False})
        self.assertEqual(pm.series_models, {})
        self.assertEqual(pm.series_grids, {})

    def test_model_on_an_untouched_sid_survives(self):
        """A fitted model is a result of its own — still an analysis
        source on the amp tab — so a swap that does not reuse its sid
        must not force a refit."""
        pm = self._make()
        model, grid = object(), object()
        pm.series_models = {5: model}
        pm.series_grids = {5: grid}
        pm.replace_overlay_series([_pt(150, -3, 7, 300, 1)], {1: "new"},
                                  {}, {1: False})
        self.assertIs(pm.series_models[5], model)
        self.assertIs(pm.series_grids[5], grid)

    def test_model_dropped_when_only_the_label_claims_the_sid(self):
        """The incoming set can claim a sid with no points yet (labels
        are the other half of the identity) — the model must still go."""
        pm = self._make()
        pm.series_models = {7: object()}
        pm.series_grids = {7: object()}
        pm.replace_overlay_series([], {7: "new"}, {}, {7: False})
        self.assertNotIn(7, pm.series_models)
        self.assertNotIn(7, pm.series_grids)

    def test_missing_track_dict_does_not_erase_scan_flag(self):
        pm = self._make()
        pm.replace_overlay_series([], {1: "new"}, {}, None)
        self.assertIs(pm.series_ug2_track[0], True)

    def test_no_scan_bookkeeping_leaves_no_sid0_entries(self):
        """Nothing is invented when the plot has no current scan."""
        pm = _make_pm()
        pm.points = [_pt(100, -2, 4, 250, 1)]
        pm.series_labels = {1: "old"}
        pm.series_colors = {}
        pm.series_ug2_track = {1: False}
        pm.replace_overlay_series([], {2: "new"}, {}, {2: True})
        self.assertNotIn(0, pm.series_labels)
        self.assertNotIn(0, pm.series_colors)
        self.assertNotIn(0, pm.series_ug2_track)


# ----------------------------------------------------------------------
# on_*_changed callbacks — behavioural smoke
# ----------------------------------------------------------------------


class TestSelectorCallbacks(unittest.TestCase):

    def test_on_display_filter_changed_propagates_visibility(self):
        pm = _make_pm()
        lamp = MagicMock()
        lamp.checked_values.return_value = [0, 2]
        pm.w["lamp_display_combo"] = lamp
        pm.on_display_filter_changed()
        pm.renderer.set_sid_visibility.assert_called_once_with({0, 2})

    def test_on_display_filter_changed_no_widget_is_noop(self):
        pm = _make_pm()
        pm.on_display_filter_changed()  # no widget → just returns
        pm.renderer.set_sid_visibility.assert_not_called()

    def test_on_ug2_calc_changed_skipped_when_no_data(self):
        pm = _make_pm()
        pm.render_all = MagicMock()
        pm.on_ug2_calc_changed()
        pm.render_all.assert_not_called()

    def test_on_ug2_calc_changed_triggers_full_render_with_data(self):
        pm = _make_pm()
        pm.points = [_pt(100, -1, 5, 250)]
        pm.render_all = MagicMock()
        pm.on_ug2_calc_changed()
        pm.render_all.assert_called_once()


if __name__ == "__main__":
    unittest.main()
