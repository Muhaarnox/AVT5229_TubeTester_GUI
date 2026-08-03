"""Acceptance pins: a plotted scan keeps its curves when the next run is armed.

These go through the real ``PlotRenderer`` (offscreen Qt) via
``PlotManager.render_2d_only`` — the whole chain recorded flag →
``_get_track_sids`` → ``render_plot_2d`` → grouping — and count what
actually reached the plot.

The discriminating case is a triode-connected scan (Ug2 = Ua + offset).
Grouped by (Ug1, Ug2) every point lands in its own group, so each curve
degenerates into a single symbol with no line: the failure is "the plot
falls apart into dots", not a subtle colour change.  Both Ug2 display
modes share that grouping, which is why the pins assert on both — the
mode switch is the usual trigger for the re-render, never the cause.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from app.plot_manager import PlotManager
from app.plotting.renderer import PlotRenderer

_qapp = QApplication.instance() or QApplication([])


# ── module local constants ──
# Space-charge stand-in for the synthetic tube: Ia = K*(Ug1 + Ua/MU)^1.5.
_MU = 20.0
_PERVEANCE_MA = 0.9
_UG2_OFFSET_V = 0.0     # triode-connected: screen strapped to the anode
_UA_START_V = 20.0
_UA_STOP_V = 300.0
_UA_STEP_V = 10.0       # > the 2 V Ug2 cluster threshold on purpose
_UG1_VALUES_V = (0.0, -2.0, -4.0, -6.0, -8.0, -10.0, -12.0, -14.0)
_UG2_LEVELS_V = (100.0, 150.0, 200.0, 250.0)


def _ia_ma(ua: float, ug1: float, ug2: float) -> float:
    drive = ug1 + ug2 / _MU
    return _PERVEANCE_MA * drive ** 1.5 if drive > 0 else 0.0


def _ua_grid() -> list:
    n = int(round((_UA_STOP_V - _UA_START_V) / _UA_STEP_V)) + 1
    return [_UA_START_V + i * _UA_STEP_V for i in range(n)]


def _sweep(ug1: float, ug2_of_ua) -> list:
    """One Ia(Ua) curve; points below cut-off are simply not measured."""
    out = []
    for ua in _ua_grid():
        ug2 = ug2_of_ua(ua)
        ia = _ia_ma(ua, ug1, ug2)
        if ia <= 0:
            continue
        out.append({"ua": ua, "ug1": ug1, "ug2": ug2, "ia": ia,
                    "ig2": 0.05 * ia, "series_id": 0})
    return out


def _curves_to_points(curves: list) -> tuple:
    """Flatten measurable curves; the count is the expected polyline count.

    Curves shorter than 2 points are dropped, so any single-point item on
    the plot is a grouping failure and never an artefact of the fixture.
    """
    kept = [c for c in curves if len(c) >= 2]
    return [p for c in kept for p in c], len(kept)


def _triode_connected_points() -> tuple:
    """Pentode wired as a triode: Ug2 follows Ua, so it is unique per point."""
    return _curves_to_points([
        _sweep(ug1, lambda ua: ua + _UG2_OFFSET_V) for ug1 in _UG1_VALUES_V
    ])


def _pentode_points() -> tuple:
    """True pentode: a handful of discrete screen setpoints."""
    return _curves_to_points([
        _sweep(ug1, lambda ua, ug2=ug2: ug2)
        for ug2 in _UG2_LEVELS_V for ug1 in _UG1_VALUES_V
    ])


def _val(v):
    w = MagicMock()
    w.value.return_value = v
    return w


def _check(v):
    w = MagicMock()
    w.isChecked.return_value = v
    return w


def _make_pm(*, ug2_track_checked: bool, ug2_mode_series: bool = False,
             is_triode: bool = False) -> PlotManager:
    """PlotManager driving a real renderer, with mocked control widgets."""
    renderer = PlotRenderer(pg.PlotWidget(), pg.PlotWidget(), pg.ImageItem(),
                            transfer_plot=pg.PlotWidget(),
                            curves_plot=pg.PlotWidget())
    widgets = {
        "ug2_track_radio": _check(ug2_track_checked),
        "ug2_mode_series": _check(ug2_mode_series),
        "overlay_pen_style": MagicMock(**{"currentIndex.return_value": 0}),
        "plot_line_width": _val(2.0),
    }
    for k in ("zone_ua_min", "zone_ua_max", "zone_ug1_min", "zone_ug1_max",
              "pa_max_input", "pg2_max_input", "ua_max_input",
              "ia_max_limit_input"):
        widgets[k] = _val(0.0)
    for k in ("pa_max_cb", "pg2_max_cb", "ua_max_cb", "ia_max_limit_cb"):
        widgets[k] = _check(False)
    for k in ("lamp_display_combo", "lamp_calc_combo", "ug2_display_combo",
              "ug2_calc_combo", "transfer_ua_combo"):
        widgets[k] = None
    pm = PlotManager(renderer, widgets)
    pm.set_triode(is_triode)
    return pm


def _line_and_dot_counts(pm: PlotManager) -> tuple:
    """(polylines with >= 2 points, isolated single-point items)."""
    lines = dots = 0
    for item in pm.renderer.plot.listDataItems():
        xs = item.getData()[0]
        n = 0 if xs is None else len(xs)
        if n >= 2:
            lines += 1
        elif n == 1:
            dots += 1
    return lines, dots


def _render_both_modes(pm: PlotManager) -> list:
    """Render in Ug2-as-color and Ug2-families order, as the radios do."""
    out = []
    for series_mode in (False, True):
        pm.w["ug2_mode_series"].isChecked.return_value = series_mode
        pm.render_2d_only()
        out.append(_line_and_dot_counts(pm))
    return out


class TestFixtureIsDiscriminating(unittest.TestCase):
    """The fixtures must be able to expose the failure at all."""

    def test_triode_connected_ug2_is_unique_per_point(self):
        """If Ug2 clustered into one nominal, (Ug1, Ug2) grouping would
        accidentally work and the pins below would prove nothing."""
        pts, n_curves = _triode_connected_points()
        per_curve = len(pts) / n_curves
        self.assertGreater(per_curve, 2)
        ug2_spread = max(p["ug2"] for p in pts) - min(p["ug2"] for p in pts)
        self.assertGreater(ug2_spread, _UA_STEP_V)

    def test_pentode_fixture_has_several_ug2_per_ug1(self):
        """Needed for the twin pin: merging Ug2 levels must be visible."""
        pts, n_curves = _pentode_points()
        by_ug1 = {}
        for p in pts:
            by_ug1.setdefault(p["ug1"], set()).add(p["ug2"])
        self.assertGreater(max(len(v) for v in by_ug1.values()), 1)
        self.assertGreater(n_curves, len(by_ug1))

    def test_pentode_ug2_levels_reach_different_ug1_depths(self):
        """Needed for the sliced-surface pins: if every level held the
        same Ug1 set, slicing and merging would yield the same grid
        shape and the pin could not tell them apart."""
        pts, _ = _pentode_points()
        by_ug2 = {}
        for p in pts:
            by_ug2.setdefault(p["ug2"], set()).add(p["ug1"])
        depths = {len(v) for v in by_ug2.values()}
        self.assertGreater(len(depths), 1)


class TestTriodeConnectedScanSurvivesRearming(unittest.TestCase):
    """A triode-connected scan stays curves after the next run is armed
    as a Ug2 sweep — the state that used to shatter it into symbols."""

    def test_curves_before_rearming(self):
        pm = _make_pm(ug2_track_checked=True)
        pm.points, expected = _triode_connected_points()
        pm.set_scan_ug2_track(True)
        for lines, dots in _render_both_modes(pm):
            self.assertEqual(lines, expected)
            self.assertEqual(dots, 0)

    def test_curves_survive_switching_the_scan_radio_to_sweep(self):
        pm = _make_pm(ug2_track_checked=True)
        pm.points, expected = _triode_connected_points()
        pm.set_scan_ug2_track(True)
        pm.render_2d_only()
        # User arms the next run as a pentode sweep of the same tube.
        pm.w["ug2_track_radio"].isChecked.return_value = False
        for lines, dots in _render_both_modes(pm):
            self.assertEqual(lines, expected)
            self.assertEqual(dots, 0)

    def test_curves_survive_selecting_a_pentode_lamp_afterwards(self):
        """The other widget feeding the old fallback: the lamp selector."""
        pm = _make_pm(ug2_track_checked=True, is_triode=True)
        pm.points, expected = _triode_connected_points()
        pm.set_scan_ug2_track(True)
        pm.set_triode(False)
        pm.w["ug2_track_radio"].isChecked.return_value = False
        for lines, dots in _render_both_modes(pm):
            self.assertEqual(lines, expected)
            self.assertEqual(dots, 0)

    def test_without_the_record_the_radio_still_decides(self):
        """Fallback pin: before any scan the widgets are all there is."""
        pm = _make_pm(ug2_track_checked=True)
        pm.points, expected = _triode_connected_points()
        lines, dots = _render_both_modes(pm)[0]
        self.assertEqual(lines, expected)
        self.assertEqual(dots, 0)


class TestPentodeScanSurvivesRearming(unittest.TestCase):
    """The twin case: a pentode scan must not collapse into Ug1-only
    curves when the next run is armed as triode-connected (Ug2 levels
    would merge into one zig-zag polyline per Ug1)."""

    def test_curves_before_rearming(self):
        pm = _make_pm(ug2_track_checked=False)
        pm.points, expected = _pentode_points()
        pm.set_scan_ug2_track(False)
        for lines, dots in _render_both_modes(pm):
            self.assertEqual(lines, expected)
            self.assertEqual(dots, 0)

    def test_curves_survive_switching_the_scan_radio_to_track(self):
        pm = _make_pm(ug2_track_checked=False)
        pm.points, expected = _pentode_points()
        pm.set_scan_ug2_track(False)
        pm.render_2d_only()
        pm.w["ug2_track_radio"].isChecked.return_value = True
        for lines, dots in _render_both_modes(pm):
            self.assertEqual(lines, expected)
            self.assertEqual(dots, 0)

    def test_curves_survive_selecting_a_triode_lamp_afterwards(self):
        pm = _make_pm(ug2_track_checked=False)
        pm.points, expected = _pentode_points()
        pm.set_scan_ug2_track(False)
        pm.set_triode(True)
        for lines, dots in _render_both_modes(pm):
            self.assertEqual(lines, expected)
            self.assertEqual(dots, 0)


class TestOverlaySeriesFollowsItsOwnMode(unittest.TestCase):
    """Twin surface: an overlay series is grouped by ITS recorded mode.

    Compare hands every overlay its own flag, so the lamp selector must
    not speak for them either — with a triode lamp picked, a pentode
    overlay used to collapse all its Ug2 levels into one zig-zag per Ug1.
    """

    def test_pentode_overlay_survives_a_triode_lamp_selection(self):
        pm = _make_pm(ug2_track_checked=True, is_triode=True)
        pts, expected = _pentode_points()
        pm.points = [dict(p, series_id=1) for p in pts]
        pm.series_labels = {1: "L1"}
        pm.series_ug2_track = {1: False}
        for lines, dots in _render_both_modes(pm):
            self.assertEqual(lines, expected)
            self.assertEqual(dots, 0)

    def test_pentode_overlay_keeps_its_ug2_colour_gradient(self):
        """Colour twin of the pin above: grouping alone cannot catch a
        lamp-selector leak into the overlay's ``all_track`` — that switch
        only feeds the colour normalisation and the Ug2 colorbar. Leaked,
        it empties ``ug2_values`` and every curve of a screen-swept
        overlay collapses to viridis(0): same polylines, one colour."""
        pm = _make_pm(ug2_track_checked=True, is_triode=True)
        pts, _ = _pentode_points()
        pm.points = [dict(p, series_id=1) for p in pts]
        pm.series_labels = {1: "L1"}
        pm.series_ug2_track = {1: False}
        pm.w["ug2_mode_series"].isChecked.return_value = False  # colour mode
        pm.render_2d_only()
        colours = self._curve_colours(pm)
        self.assertGreater(len(colours), 1,
                           "all overlay curves share one colour — Ug2 "
                           "normalisation lost its value range")

    def _curve_colours(self, pm) -> set:
        """Pen colours of items that carry actual curve data.

        The plot also holds service PlotDataItems — the snap marker (has
        a symbol but no data yet) and the zone rectangle — whose pens
        would let a single-colour collapse slip past a naive colour set.
        """
        out = set()
        for item in pm.renderer.plot.listDataItems():
            xs = item.getData()[0]
            if xs is not None and len(xs) >= 2:
                out.add(item.opts["pen"].color().name())
        return out

    def test_track_overlay_still_collapses_ug2(self):
        """The other end of the same switch: a recorded track overlay
        groups by Ug1 only, whatever the lamp selector says."""
        pm = _make_pm(ug2_track_checked=False, is_triode=False)
        pts, _ = _triode_connected_points()
        pm.points = [dict(p, series_id=1) for p in pts]
        pm.series_labels = {1: "L1"}
        pm.series_ug2_track = {1: True}
        lines, dots = _render_both_modes(pm)[0]
        self.assertEqual(lines, len(_UG1_VALUES_V))
        self.assertEqual(dots, 0)


class TestIncrementalRenderFollowsData(unittest.TestCase):
    """Live per-curve rendering during a scan reads the same verdict.

    It draws each finished curve as one polyline regardless of grouping,
    so the observable is the COLOUR: in Ug2-as-colour mode two curves at
    the same Ug1 but different Ug2 must get different viridis colours;
    falling back to the lamp selector paints both from the Ug1 palette
    slot and they come out identical.
    """

    def _last_pen_colour(self, pm) -> str:
        return pm.renderer.plot.listDataItems()[-1].opts["pen"].color().name()

    def test_pentode_curves_get_distinct_ug2_colours(self):
        pm = _make_pm(ug2_track_checked=True, is_triode=True)
        pts, _ = _pentode_points()
        pm.points = pts
        pm.set_scan_ug2_track(False)
        ug1 = _UG1_VALUES_V[0]
        colours = []
        for ug2 in (_UG2_LEVELS_V[0], _UG2_LEVELS_V[-1]):
            pm.current_curve_points = [p for p in pts
                                       if p["ug1"] == ug1 and p["ug2"] == ug2]
            self.assertTrue(pm.current_curve_points)
            pm.render_curve_incremental({"ug1": ug1, "ug2": ug2})
            colours.append(self._last_pen_colour(pm))
        self.assertNotEqual(colours[0], colours[1])

    def test_track_scan_curves_get_per_ug1_palette_colours(self):
        """Other end: a recorded track scan colours by Ug1, so two curves
        at different Ug1 differ even though Ug2 rises with Ua."""
        pm = _make_pm(ug2_track_checked=False, is_triode=False)
        pts, _ = _triode_connected_points()
        pm.points = pts
        pm.set_scan_ug2_track(True)
        colours = []
        for ug1 in (_UG1_VALUES_V[0], _UG1_VALUES_V[1]):
            curve = [p for p in pts if p["ug1"] == ug1]
            pm.current_curve_points = curve
            pm.render_curve_incremental({"ug1": ug1, "ug2": curve[0]["ug2"]})
            colours.append(self._last_pen_colour(pm))
        self.assertNotEqual(colours[0], colours[1])


class TestSlicedSurfacesFollowTheData(unittest.TestCase):
    """Heatmaps and the Curves tab keep their Ug2 slice/families after a
    triode lamp is picked.

    These surfaces show ONE screen level at a time (contour/Gm/Rp/Pa) or
    one family per level (Curves).  Taking "the data is triode" from the
    lamp selector bypassed the slice filter, so every level averaged into
    a single Ia(Ua, Ug1) grid — the maps then read plausible but wrong
    Gm/Rp instead of failing visibly, which is why these pins assert on
    grid SHAPE and family COUNT rather than on "something was drawn".
    """

    def _ug1_rows_of_contour(self, pm) -> int:
        grid = pm.renderer._marker_contour._z_grid
        return 0 if grid is None else grid.shape[0]

    def test_contour_stays_a_single_ug2_slice(self):
        pm = _make_pm(ug2_track_checked=False)
        pm.points, _ = _pentode_points()
        pm.set_scan_ug2_track(False)
        pm.render_slice_plots()
        sliced_rows = self._ug1_rows_of_contour(pm)
        # A slice cannot hold every Ug1 the whole scan has: the lower
        # screen levels cut off earlier (fixture is guarded below).
        all_ug1 = len({p["ug1"] for p in pm.points})
        self.assertGreater(sliced_rows, 0)
        self.assertLess(sliced_rows, all_ug1)

        pm.set_triode(True)          # user picks a triode lamp afterwards
        pm.render_slice_plots()
        self.assertEqual(self._ug1_rows_of_contour(pm), sliced_rows)

    def test_curves_tab_keeps_its_ug2_families(self):
        pm = _make_pm(ug2_track_checked=False)
        pm.points, _ = _pentode_points()
        pm.set_scan_ug2_track(False)
        pm.render_curves_only()
        families = sorted(pm.renderer._ug2_curves_items)
        self.assertEqual(len(families), len(_UG2_LEVELS_V))

        pm.set_triode(True)
        pm.render_curves_only()
        self.assertEqual(sorted(pm.renderer._ug2_curves_items), families)

    def test_true_triode_scan_still_uses_the_whole_set(self):
        """Other end: a recorded triode scan has no Ug2 axis to slice, so
        its contour keeps every Ug1 even with a pentode lamp selected."""
        pm = _make_pm(ug2_track_checked=True, is_triode=True)
        pm.points, _ = _triode_connected_points()
        pm.set_scan_ug2_track(True)
        pm.render_slice_plots()
        rows = self._ug1_rows_of_contour(pm)
        self.assertEqual(rows, len({p["ug1"] for p in pm.points}))

        pm.set_triode(False)
        pm.w["ug2_track_radio"].isChecked.return_value = False
        pm.render_slice_plots()
        self.assertEqual(self._ug1_rows_of_contour(pm), rows)


class TestModeToggleIsStable(unittest.TestCase):
    """Repeated Ug2 families ↔ colour switches must not degrade the plot:
    the mode is a drawing choice, the grouping is not part of it."""

    def test_repeated_toggles_keep_the_same_curves(self):
        pm = _make_pm(ug2_track_checked=False)
        pm.points, expected = _pentode_points()
        pm.set_scan_ug2_track(False)
        counts = []
        for _ in range(3):
            counts.extend(_render_both_modes(pm))
        self.assertEqual(len(set(counts)), 1, counts)
        self.assertEqual(counts[0], (expected, 0))


if __name__ == "__main__":
    unittest.main()
