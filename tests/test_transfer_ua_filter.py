"""Transfer tab Ua-slice filter + view presets.

Pins:
  1.  Preset helpers on non-degenerate (non-uniform, shuffled) Ua grids.
  2.  Transfer render fills the per-Ua item store; the Curves-tab twin
      render path must NOT (negative space).
  3.  set_transfer_ua_visibility toggles curve items AND labels, filters
      the transfer snap-marker, leaves 2D/Curves stores untouched.
  4.  Combined Ug2 × Ua predicate — either filter alone must not clobber
      the other, in both application orders.
  5.  Accents: exactly one "≈Ub" label (none when Ub is outside the data
      span); out-of-swing slices dimmed with boundary-inclusive edges;
      halo underlay beneath the dynamic load-line curve.
  6.  PlotManager: choke-point refresh (refresh_ug2_combos → Ua combo),
      preset application, manual edit → Custom, Ub from the amp panel.
  7.  MainWindow wiring: combo signals reach the handlers (dropped
      connect fails here), render_all re-applies the filter, and the
      view combo enumerates ALL registry modes (source-of-truth pin).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.plot_manager import (
    DATASHEET_UA_SLICES,
    PlotManager,
    TRANSFER_VIEW_ALL,
    TRANSFER_VIEW_CUSTOM,
    TRANSFER_VIEW_DATASHEET,
    TRANSFER_VIEW_LOADLINE,
    TRANSFER_VIEW_MODES,
    datasheet_ua_slices,
    loadline_ua_slices,
)

pytestmark = [pytest.mark.smoke_ui]


# ── Module local helpers ──

# Non-uniform Ua grid (uniform spacing would hide index-arithmetic
# mutations in the datasheet preset).
UA_GRID = [50.0, 90.0, 150.0, 220.0, 300.0]
UG1_GRID = [-1.0, -2.0, -4.0]


def _grid_points(ua_vals=None, ug1_vals=None, ug2_vals=(200.0,),
                 series_id=0):
    """Physically-plausible pentode-ish grid: Ia grows with Ua, drops
    with |Ug1|, scales with Ug2."""
    pts = []
    for ug2 in ug2_vals:
        for ug1 in (ug1_vals or UG1_GRID):
            for ua in (ua_vals or UA_GRID):
                ia = max(0.05, (ua / 40.0) * (5.0 + ug1) * (ug2 / 200.0))
                pts.append({"ua": ua, "ug1": ug1, "ia": ia, "ug2": ug2,
                            "ig2": ia * 0.1, "series_id": series_id})
    return pts


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _make_renderer():
    import pyqtgraph as pg

    from app.plotting import PlotRenderer

    r = PlotRenderer(
        plot=pg.PlotWidget(),
        contour_plot=pg.PlotWidget(),
        contour_image=pg.ImageItem(),
        transfer_plot=pg.PlotWidget(),
        curves_plot=pg.PlotWidget(),
    )
    r.is_triode = False
    r.track_sids = set()
    return r


def _render_transfer(r, pts, ub_ref=None, track_sids=None):
    r.render_curves(pts, y_param="Ia", x_param="Ug1",
                    target_plot=r.transfer_plot, ub_ref=ub_ref,
                    track_sids=track_sids)


def _label_texts(r):
    return [lbl.textItem.toPlainText() for lbl in r._transfer_labels
            if hasattr(lbl, "textItem")]


# ======================================================================
# 1. Preset helpers (pure logic)
# ======================================================================


class TestPresetHelpers:

    def test_datasheet_even_spacing_includes_endpoints(self):
        # Shuffled, 10 values → indices {0, 2, 4, 7, 9} of the sorted grid.
        vals = [70.0, 10.0, 100.0, 40.0, 90.0, 20.0, 60.0, 30.0, 80.0, 50.0]
        out = datasheet_ua_slices(vals, count=5)
        assert out == [10.0, 30.0, 50.0, 80.0, 100.0]
        # Discriminators: "first five" and "every 2nd" mutants differ.
        assert out != [10.0, 20.0, 30.0, 40.0, 50.0]
        assert out != [10.0, 30.0, 50.0, 70.0, 90.0]

    def test_datasheet_small_grid_returns_all_sorted(self):
        assert datasheet_ua_slices([300.0, 50.0, 150.0]) == [50.0, 150.0,
                                                             300.0]

    def test_datasheet_default_count_constant(self):
        vals = list(range(1, 21))
        assert len(datasheet_ua_slices([float(v) for v in vals])) == \
            DATASHEET_UA_SLICES

    def test_loadline_nearest_not_floor(self):
        noms = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
        # ub=180: nearest is 200 (floor mutant would pick 150).
        assert loadline_ua_slices(noms, 180.0) == [150.0, 200.0, 250.0]

    def test_loadline_clamps_at_both_ends(self):
        noms = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
        assert loadline_ua_slices(noms, 10.0) == [50.0, 100.0]
        assert loadline_ua_slices(noms, 1000.0) == [250.0, 300.0]

    def test_loadline_empty(self):
        assert loadline_ua_slices([], 250.0) == []


# ======================================================================
# 2-4. Renderer: store, visibility, combined predicate
# ======================================================================


@pytest.mark.usefixtures("qapp")
class TestUaStoreAndVisibility:

    def test_transfer_render_fills_ua_store(self):
        r = _make_renderer()
        _render_transfer(r, _grid_points())
        assert sorted(r._ua_transfer_items.keys()) == UA_GRID
        # Each slice registered its curve item AND its label.
        for items in r._ua_transfer_items.values():
            assert len(items) >= 2

    def test_curves_tab_twin_does_not_fill_ua_store(self):
        """render_curves(Ia, Ug1) on the CURVES plot is the same code
        path (step=Ua) — it must not feed the Transfer filter store."""
        r = _make_renderer()
        r.render_curves(_grid_points(), y_param="Ia", x_param="Ug1")
        assert r._ua_transfer_items == {}

    def test_visibility_toggle_hides_curves_and_labels(self):
        r = _make_renderer()
        _render_transfer(r, _grid_points())
        r.set_transfer_ua_visibility([150.0], r.ua_cluster_thr)
        for nom, items in r._ua_transfer_items.items():
            expect = nom == 150.0
            for item in items:
                assert item.isVisible() is expect, (nom, item)

    def test_visibility_filters_transfer_marker(self):
        r = _make_renderer()
        _render_transfer(r, _grid_points())
        n_all = len(r._marker_transfer._curves)
        r.set_transfer_ua_visibility([150.0, 300.0], r.ua_cluster_thr)
        n_left = len(r._marker_transfer._curves)
        assert 0 < n_left < n_all
        for c in r._marker_transfer._curves:
            assert float(c.extra["Ua"][0]) in (150.0, 300.0)

    def test_ua_filter_leaves_2d_and_curves_stores_alone(self):
        r = _make_renderer()
        r.render_curves(_grid_points(ug2_vals=(150.0, 250.0)),
                        y_param="Ia", x_param="Ua")  # curves tab, ug2 store
        _render_transfer(r, _grid_points())
        assert r._ug2_curves_items
        r.set_transfer_ua_visibility([90.0], r.ua_cluster_thr)
        for items in r._ug2_curves_items.values():
            for item in items:
                assert item.isVisible()

    @pytest.mark.parametrize("ug2_first", [True, False])
    def test_combined_ug2_and_ua_predicate(self, ug2_first):
        """Item visible iff BOTH filters pass — in either apply order.
        A direct setVisible in either setter would clobber the other."""
        r = _make_renderer()
        _render_transfer(r, _grid_points(ug2_vals=(150.0, 250.0)))
        assert sorted(r._ug2_transfer_items.keys()) == [150.0, 250.0]

        def apply_ug2():
            r.set_ug2_visibility([150.0], r.ug2_cluster_thr)

        def apply_ua():
            r.set_transfer_ua_visibility([90.0], r.ua_cluster_thr)

        if ug2_first:
            apply_ug2(); apply_ua()
        else:
            apply_ua(); apply_ug2()

        ug2_ok = {id(i) for i in r._ug2_transfer_items[150.0]}
        ug2_bad = {id(i) for i in r._ug2_transfer_items[250.0]}
        for nom, items in r._ua_transfer_items.items():
            for item in items:
                in_good_ug2 = id(item) in ug2_ok
                in_bad_ug2 = id(item) in ug2_bad
                expect = (nom == 90.0) and not in_bad_ug2 and (
                    in_good_ug2 or not (in_good_ug2 or in_bad_ug2))
                assert item.isVisible() is expect, (nom, in_good_ug2,
                                                    in_bad_ug2)
        # Marker sees the intersection only.
        for c in r._marker_transfer._curves:
            assert float(c.extra["Ua"][0]) == 90.0
            assert abs(float(c.extra["Ug2"][0]) - 150.0) <= r.ug2_cluster_thr

    def test_single_point_ua_group_still_dropped(self):
        """Refined lone-Ua points (1 point per slice) draw nothing and
        register nothing — regression guard."""
        r = _make_renderer()
        pts = _grid_points()
        pts.append({"ua": 137.0, "ug1": -2.0, "ia": 3.0, "ug2": 200.0,
                    "ig2": 0.3, "series_id": 0})
        _render_transfer(r, pts)
        assert 137.0 not in r._ua_transfer_items

    def test_rerender_clears_stale_store(self):
        r = _make_renderer()
        _render_transfer(r, _grid_points())
        _render_transfer(r, _grid_points(ua_vals=[60.0, 120.0]))
        assert sorted(r._ua_transfer_items.keys()) == [60.0, 120.0]

    def test_triode_path_fills_store_and_filters(self):
        """Triode/track twin: items live in the Ua store ONLY (no Ug2
        store entries) — the filter must still work there."""
        r = _make_renderer()
        r.is_triode = True
        pts = [{"ua": ua, "ug1": ug1,
                "ia": max(0.05, (ua / 40.0) * (5.0 + ug1)),
                "ug2": 0.0, "ig2": 0.0, "series_id": 0}
               for ug1 in UG1_GRID for ua in UA_GRID]
        # The measurement's own mode reaches the renderer as track_sids
        # (PlotManager._get_track_sids); the lamp selector does not speak
        # for the data.
        _render_transfer(r, pts, track_sids={0})
        assert r._ug2_transfer_items == {}          # triode: no Ug2 store
        assert sorted(r._ua_transfer_items.keys()) == UA_GRID
        r.set_transfer_ua_visibility([220.0], r.ua_cluster_thr)
        for nom, items in r._ua_transfer_items.items():
            expect = nom == 220.0
            for item in items:
                assert item.isVisible() is expect, nom
        for c in r._marker_transfer._curves:
            assert float(c.extra["Ua"][0]) == 220.0

    def test_track_mode_twin_fills_store_and_filters(self):
        """Triode-CONNECTED twin (track_sids, Ug2 = Ua + offset): reaches
        the same is_triode_eff branch through a different flag path."""
        r = _make_renderer()
        pts = [{"ua": ua, "ug1": ug1,
                "ia": max(0.05, (ua / 40.0) * (5.0 + ug1)),
                "ug2": ua + 20.0, "ig2": 0.1, "series_id": 0}
               for ug1 in UG1_GRID for ua in UA_GRID]
        r.render_curves(pts, y_param="Ia", x_param="Ug1",
                        target_plot=r.transfer_plot, track_sids={0})
        assert r._ug2_transfer_items == {}          # track: no Ug2 store
        assert sorted(r._ua_transfer_items.keys()) == UA_GRID
        r.set_transfer_ua_visibility([90.0, 300.0], r.ua_cluster_thr)
        for nom, items in r._ua_transfer_items.items():
            expect = nom in (90.0, 300.0)
            for item in items:
                assert item.isVisible() is expect, nom


# ======================================================================
# 5. Accents: ≈Ub label, swing dimming, halo
# ======================================================================


@pytest.mark.usefixtures("qapp")
class TestAccents:

    def test_near_ub_suffix_on_exactly_one_slice(self):
        r = _make_renderer()
        _render_transfer(r, _grid_points(), ub_ref=230.0)  # nearest: 220
        marked = [txt for txt in _label_texts(r) if "≈Ub" in txt]
        assert len(marked) == 1
        assert "220" in marked[0]

    def test_near_ub_suffix_absent_when_ub_outside_span(self):
        r = _make_renderer()
        _render_transfer(r, _grid_points(), ub_ref=800.0)
        assert not [txt for txt in _label_texts(r) if "≈Ub" in txt]

    def test_out_of_swing_slices_dimmed_boundary_inclusive(self):
        from lm19.plot_style import TRANSFER_DIM_ALPHA
        r = _make_renderer()
        # Swing covers Ua 90..220; boundary slices are NOT dimmed.
        r._load_line_analysis = {
            "pt_neg": {"ua": 220.0, "ia": 2.0, "ug1": -3.0},
            "pt_pos": {"ua": 90.0, "ia": 8.0, "ug1": -1.0},
        }
        _render_transfer(r, _grid_points())
        assert sorted(r._ua_transfer_items.keys()) == UA_GRID
        for nom, items in r._ua_transfer_items.items():
            pen = items[0].opts["pen"]
            alpha = pen.color().alpha()
            if nom in (50.0, 300.0):
                assert alpha == TRANSFER_DIM_ALPHA, nom
            else:  # 90 (boundary), 150, 220 (boundary)
                assert alpha == 255, nom

    def test_no_dimming_without_analysis(self):
        r = _make_renderer()
        _render_transfer(r, _grid_points())
        for items in r._ua_transfer_items.values():
            assert items[0].opts["pen"].color().alpha() == 255

    def test_load_line_curve_gets_halo_underlay(self):
        from lm19.plot_style import (
            COLOR_CURVE_HALO,
            LOAD_LINE_CURVE_MIN_WIDTH,
            LOAD_LINE_HALO_EXTRA_W,
        )
        import pyqtgraph as pg
        r = _make_renderer()
        r._load_line_intersections = [
            {"ua": 250.0, "ug1": -1.0, "ia": 9.0},
            {"ua": 200.0, "ug1": -2.0, "ia": 6.0},
            {"ua": 150.0, "ug1": -4.0, "ia": 3.0},
        ]
        _render_transfer(r, _grid_points())
        halo_rgb = pg.mkColor(COLOR_CURVE_HALO).getRgb()
        halos = [
            it for it in r.transfer_plot.getPlotItem().listDataItems()
            if it.opts.get("pen") is not None
            and pg.mkPen(it.opts["pen"]).color().getRgb() == halo_rgb
            and pg.mkPen(it.opts["pen"]).widthF() >= (
                LOAD_LINE_CURVE_MIN_WIDTH + LOAD_LINE_HALO_EXTRA_W)
        ]
        assert halos, "no halo underlay under the dynamic transfer curve"


# ======================================================================
# 5b. PDF WYSIWYG
# ======================================================================


@pytest.mark.usefixtures("qapp")
class TestPdfWysiwyg:
    """Transfer goes into the PDF as on screen — with the Ua filter on.

    The pin holds an external assumption: the export render
    (``render_plot_pixmap`` -> pyqtgraph ImageExporter) must NOT draw
    hidden items. A silent change of that behavior on a pyqtgraph
    upgrade would produce an 'all slices' report without a single
    change in our code.
    """

    def test_export_pixmap_respects_ua_filter(self):
        from app.export_manager import render_plot_pixmap
        r = _make_renderer()
        r.transfer_plot.resize(640, 480)
        _render_transfer(r, _grid_points())
        base = render_plot_pixmap(r.transfer_plot).toImage()
        again = render_plot_pixmap(r.transfer_plot).toImage()
        # Determinism first — without it the diff assert below would be
        # vacuously satisfiable by render noise.
        assert base == again
        r.set_transfer_ua_visibility([150.0], r.ua_cluster_thr)
        filtered = render_plot_pixmap(r.transfer_plot).toImage()
        assert filtered != base, (
            "hidden Ua slices leaked into the exported pixmap — "
            "PDF WYSIWYG assumption broken")


# ======================================================================
# 6. PlotManager: combo refresh, presets, custom switch
# ======================================================================


def _make_manager(qapp):
    from PySide6.QtWidgets import QComboBox

    from app.checkable_combo import CheckableComboBox

    r = _make_renderer()
    ua_combo = CheckableComboBox(placeholder="Ua")
    view_combo = QComboBox()
    for mode in (TRANSFER_VIEW_ALL, TRANSFER_VIEW_DATASHEET,
                 TRANSFER_VIEW_LOADLINE, TRANSFER_VIEW_CUSTOM):
        view_combo.addItem(mode, userData=mode)
    view_combo.setCurrentIndex(view_combo.findData(TRANSFER_VIEW_DATASHEET))
    # Mock keeps the real spinbox read interface (value() → float).
    panel = SimpleNamespace(ub_spin=SimpleNamespace(value=lambda: 220.0))
    w = {
        "transfer_ua_combo": ua_combo,
        "transfer_view_combo": view_combo,
        "amp_control_panel": panel,
    }
    return PlotManager(r, w), ua_combo, view_combo


@pytest.mark.usefixtures("qapp")
class TestPlotManagerPresets:

    def test_choke_point_refresh_fills_ua_combo(self, qapp):
        """refresh_ug2_combos is the single data-change choke point — it
        must refresh the Ua combo too (dropped call fails here)."""
        pm, ua_combo, _view = _make_manager(qapp)
        pm.refresh_ug2_combos(_grid_points())
        assert ua_combo.all_values() == UA_GRID

    def test_datasheet_default_checks_subset(self, qapp):
        pm, ua_combo, _view = _make_manager(qapp)
        pm.refresh_ua_combo(_grid_points(
            ua_vals=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
                     90.0, 100.0]))
        assert ua_combo.checked_values() == [10.0, 30.0, 50.0, 80.0, 100.0]

    def test_loadline_preset_uses_panel_ub(self, qapp):
        pm, ua_combo, view = _make_manager(qapp)
        view.setCurrentIndex(view.findData(TRANSFER_VIEW_LOADLINE))
        pm.refresh_ua_combo(_grid_points())  # ub=220 → [150, 220, 300]
        assert ua_combo.checked_values() == [150.0, 220.0, 300.0]

    def test_all_preset_checks_everything(self, qapp):
        pm, ua_combo, view = _make_manager(qapp)
        view.setCurrentIndex(view.findData(TRANSFER_VIEW_ALL))
        pm.refresh_ua_combo(_grid_points())
        assert ua_combo.checked_values() == UA_GRID

    def test_custom_preset_preserves_user_checks(self, qapp):
        pm, ua_combo, view = _make_manager(qapp)
        pm.refresh_ua_combo(_grid_points())
        ua_combo.set_checked_values([90.0])
        view.setCurrentIndex(view.findData(TRANSFER_VIEW_CUSTOM))
        pm.apply_transfer_view_preset()
        assert ua_combo.checked_values() == [90.0]

    def test_data_refresh_in_custom_mode_resets_to_all(self, qapp):
        """Documented semantics (twin of the Ug2 filter): a DATA refresh
        rebuilds the list all-checked even in Custom — stale manual
        checks are meaningless on a possibly different Ua grid. Preset
        re-application (previous test) is the path that preserves."""
        pm, ua_combo, view = _make_manager(qapp)
        pm.refresh_ua_combo(_grid_points())
        ua_combo.set_checked_values([90.0])
        view.setCurrentIndex(view.findData(TRANSFER_VIEW_CUSTOM))
        pm.refresh_ua_combo(_grid_points())  # new data arrives
        assert ua_combo.checked_values() == UA_GRID
        assert view.currentData() == TRANSFER_VIEW_CUSTOM  # label intact

    def test_manual_toggle_switches_view_to_custom_and_applies(self, qapp):
        pm, ua_combo, view = _make_manager(qapp)
        pm.refresh_ua_combo(_grid_points())
        applied = []
        pm.renderer.set_transfer_ua_visibility = (
            lambda checked, thr: applied.append(list(checked)))
        pm.on_ua_display_changed()
        assert view.currentData() == TRANSFER_VIEW_CUSTOM
        assert applied, "visibility not pushed to renderer"

    def test_view_change_applies_preset_and_visibility(self, qapp):
        pm, ua_combo, view = _make_manager(qapp)
        pm.refresh_ua_combo(_grid_points())
        applied = []
        pm.renderer.set_transfer_ua_visibility = (
            lambda checked, thr: applied.append(sorted(checked)))
        view.setCurrentIndex(view.findData(TRANSFER_VIEW_LOADLINE))
        pm.on_transfer_view_changed()
        assert applied and applied[-1] == [150.0, 220.0, 300.0]

    def test_registry_modes_all_apply_without_error(self, qapp):
        """Completeness from the source of truth: every registry mode
        applies; non-Custom modes give a non-empty value subset."""
        pm, ua_combo, view = _make_manager(qapp)
        pm.refresh_ua_combo(_grid_points())
        for mode in sorted(TRANSFER_VIEW_MODES):
            idx = view.findData(mode)
            assert idx >= 0, f"view combo is missing mode {mode!r}"
            view.setCurrentIndex(idx)
            pm.apply_transfer_view_preset()
            checked = ua_combo.checked_values()
            assert set(checked) <= set(UA_GRID)
            if mode != TRANSFER_VIEW_CUSTOM:
                assert checked, mode

    def test_visibility_uses_ua_threshold_not_ug2(self, qapp):
        """Argument-swap discriminator: at default thresholds ua==ug2
        (2.0 V) the swap is invisible, so pin with DIFFERENT thresholds.
        Combo values are cluster averages (offset from store nominals by
        1.5 V here) — only the Ua threshold keeps them matched."""
        import pyqtgraph as pg

        from PySide6.QtWidgets import QComboBox

        from app.checkable_combo import CheckableComboBox
        from app.plotting import PlotRenderer

        r = PlotRenderer(
            plot=pg.PlotWidget(),
            contour_plot=pg.PlotWidget(),
            contour_image=pg.ImageItem(),
            transfer_plot=pg.PlotWidget(),
            curves_plot=pg.PlotWidget(),
            ua_cluster_thr=5.0,
            ug2_cluster_thr=0.5,
        )
        r.is_triode = False
        r.track_sids = set()
        ua_combo = CheckableComboBox(placeholder="Ua")
        view_combo = QComboBox()
        view_combo.addItem(TRANSFER_VIEW_ALL, userData=TRANSFER_VIEW_ALL)
        pm = PlotManager(r, {"transfer_ua_combo": ua_combo,
                             "transfer_view_combo": view_combo})
        # Ua 100/103 cluster (thr 5) → store nominal 100.0, combo avg 101.5.
        pts = _grid_points(ua_vals=[100.0, 103.0, 250.0])
        _render_transfer(r, pts)
        pm.refresh_ua_combo(pts)
        pm._apply_ua_visibility()  # everything checked → all visible
        for nom, items in r._ua_transfer_items.items():
            for item in items:
                assert item.isVisible(), (
                    f"slice {nom} hidden — wrong cluster threshold "
                    f"passed to set_transfer_ua_visibility")

    def test_transfer_render_passes_panel_ub(self, qapp):
        """Call-site spy: _render_transfer must forward the panel Ub as
        ub_ref (unit behaviour of render_curves alone can't prove it)."""
        pm, _ua, _view = _make_manager(qapp)
        pm.w["plot_line_width"] = SimpleNamespace(value=lambda: 2.0)
        got = {}

        def spy(points, **kw):
            got.update(kw)

        pm.renderer.render_curves = spy
        pm._render_transfer(_grid_points())
        assert got.get("ub_ref") == 220.0


# ======================================================================
# 7. MainWindow wiring (dropped connect / dropped re-apply fail here)
# ======================================================================


class _Port:
    def __init__(self, device: str) -> None:
        self.device = device


@pytest.fixture
def window(qapp, monkeypatch):
    from app.main_window import MainWindow
    monkeypatch.setattr(
        "app.main_window.list_ports.comports",
        lambda: [_Port("COM1")],
    )
    w = MainWindow()
    yield w
    w.close()


class TestMainWindowWiring:

    def test_view_combo_enumerates_registry(self, window):
        modes = {window.transfer_view_combo.itemData(i)
                 for i in range(window.transfer_view_combo.count())}
        assert modes == set(TRANSFER_VIEW_MODES)
        assert window.transfer_view_combo.currentData() == \
            TRANSFER_VIEW_DATASHEET

    def test_ua_checkbox_toggle_reaches_renderer(self, window):
        """Dropped selectionChanged connect → this fails."""
        from PySide6.QtCore import Qt
        pm = window.plot_mgr
        pm.points = _grid_points()
        pm.refresh_ug2_combos(pm.points)
        applied = []
        pm.renderer.set_transfer_ua_visibility = (
            lambda checked, thr: applied.append(list(checked)))
        combo = window.transfer_ua_combo
        item = combo._model.item(0)
        new_state = (Qt.CheckState.Unchecked
                     if item.checkState() == Qt.CheckState.Checked
                     else Qt.CheckState.Checked)
        item.setCheckState(new_state)
        assert applied, "combo toggle did not reach the renderer"
        assert window.transfer_view_combo.currentData() == \
            TRANSFER_VIEW_CUSTOM

    def test_view_combo_change_reaches_preset(self, window):
        # >5 slices so Datasheet (default) ≠ All — with 5 the presets
        # coincide and a dropped connect would be invisible.
        wide = [20.0, 50.0, 90.0, 120.0, 150.0, 190.0, 220.0, 260.0,
                300.0, 340.0]
        pm = window.plot_mgr
        pm.points = _grid_points(ua_vals=wide)
        pm.refresh_ug2_combos(pm.points)
        assert len(window.transfer_ua_combo.checked_values()) == \
            DATASHEET_UA_SLICES
        window.transfer_view_combo.setCurrentIndex(
            window.transfer_view_combo.findData(TRANSFER_VIEW_ALL))
        assert window.transfer_ua_combo.checked_values() == wide

    def test_render_all_reapplies_ua_filter(self, window):
        """Dropped _apply_ua_visibility in render_all → stale visibility
        after a re-render → this fails."""
        pm = window.plot_mgr
        pm.points = _grid_points()
        pm.refresh_ug2_combos(pm.points)
        window.transfer_ua_combo.set_checked_values([150.0])
        pm.render_all()
        r = pm.renderer
        assert r._ua_transfer_items, "transfer render produced no slices"
        for nom, items in r._ua_transfer_items.items():
            expect = nom == 150.0
            for item in items:
                assert item.isVisible() is expect, nom

    def test_transfer_tab_controls_have_tooltips(self, window):
        tip = window.transfer_view_combo.toolTip()
        assert tip and not tip.startswith("tip.")
