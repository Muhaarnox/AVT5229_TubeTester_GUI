"""Heatmap colorbars + lock-scale + Compare legend semantics.

Pins:
  1.  Every heatmap render creates/updates its ColorBarItem and the bar
      levels EQUAL the image levels (the maps had no value scale at
      all — readings lived only in hover tooltips).
  2.  Lock scale: levels captured on lock survive a re-render with
      DIFFERENT data (autoLevels would re-stretch); unlock returns to
      auto on the next render. Non-degenerate: the two datasets have
      visibly different maxima.
  3.  Colormap change reaches the bars (update_heatmap_colorbars_cmap)
      AND the 2D Ug2 colorbar (its gradient used to stay on the scheme
      it was created with while the curves were already recolored).
  4.  Compare legend: one entry per LAMP with the lamp's color (solid),
      Ug2 levels as NEUTRAL-GRAY styled entries; no Ug2 entry painted
      with a lamp color (the old conflated legend). Duplicate
      measurement names get numbered suffixes instead of hiding the
      second color; triode entries are listed without Ug2 rows.
  5.  MainWindow wiring: the lock checkbox and the colormap combo
      actually reach the renderer (call-site != function).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.plot_style import COLOR_ZONE

pytestmark = [pytest.mark.smoke_ui]

UA_VALS = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
UG1_VALS = [-1.0, -2.0, -3.0, -4.0]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ── Module local helpers ──

def _make_renderer():
    import pyqtgraph as pg

    from app.plotting import PlotRenderer

    def _img_plot():
        plot = pg.PlotWidget()
        image = pg.ImageItem()
        plot.addItem(image)
        return plot, image

    contour_plot, contour_image = _img_plot()
    gm_plot, gm_image = _img_plot()
    rp_plot, rp_image = _img_plot()
    mu_plot, mu_image = _img_plot()
    pa_plot, pa_image = _img_plot()
    r = PlotRenderer(
        plot=pg.PlotWidget(),
        contour_plot=contour_plot,
        contour_image=contour_image,
        transfer_plot=pg.PlotWidget(),
        gm_plot=gm_plot, gm_image=gm_image,
        rp_plot=rp_plot, rp_image=rp_image,
        mu_plot=mu_plot, mu_image=mu_image,
        pa_map_plot=pa_plot, pa_map_image=pa_image,
        curves_plot=pg.PlotWidget(),
    )
    # The fixture points are triode data (Ug2 = 0). Say so the way the
    # renderer reads it — via track_sids, which is what PlotManager fills
    # from the measurement. ``is_triode`` mirrors the lamp SELECTOR and no
    # longer speaks for the data.
    r.is_triode = True
    r.track_sids = {0}
    return r


def _points(ia_scale: float = 1.0, ug1_gain: float = 1.0) -> List[Dict]:
    """Synthetic grid. ia_scale multiplies Ia uniformly (moves contour/
    Gm/Rp/Pa levels but leaves mu = Gm*Rp EXACTLY invariant); ug1_gain
    changes the Ug1 sensitivity (mu = gain*ua/(5+gain*ug1)) — required
    to discriminate mu-map level mutations."""
    pts = []
    for ug1 in UG1_VALS:
        for ua in UA_VALS:
            ia = max(0.05, ia_scale * (5.0 + ug1_gain * ug1) * ua / 40.0)
            pts.append({"ua": ua, "ug1": ug1, "ia": ia, "ug2": 0.0,
                        "ig2": 0.0, "series_id": 0})
    return pts


def _render_all_maps(r, pts):
    r.render_contour(pts, None)
    r.render_gm_rp(pts)
    r.render_pa_map(pts)


def _keys(r) -> tuple:
    """Heatmap keys derived from the SOURCE OF TRUTH enumeration — a
    hand-written shadow list would silently exclude a 6th map from the
    bar/lock/cmap pins."""
    return tuple(r._heatmap_images())


# ======================================================================
# 1. Bars exist and mirror image levels
# ======================================================================


@pytest.mark.usefixtures("qapp")
class TestColorbars:

    def test_every_map_gets_a_bar_with_image_levels(self):
        r = _make_renderer()
        _render_all_maps(r, _points())
        for key in _keys(r):
            bar = r._heatmap_bars.get(key)
            assert bar is not None, f"no colorbar for {key!r}"
            image = r._heatmap_images()[key]
            lo, hi = image.getLevels()
            blo, bhi = bar.levels()
            assert (blo, bhi) == pytest.approx((lo, hi)), key

    def test_first_render_levels_match_data_range(self):
        """Absolute pin: a values-clobber at bar creation (explicit
        values=(a,b) makes ColorBarItem force ITS levels onto the
        image) keeps bar==image equality true while every first render
        is visually saturated — only the absolute range catches it."""
        r = _make_renderer()
        pts = _points()
        _render_all_maps(r, pts)
        lo, hi = r._heatmap_images()["contour"].getLevels()
        ias = [p["ia"] for p in pts]
        assert lo == pytest.approx(min(ias))
        assert hi == pytest.approx(max(ias))

    def test_blank_render_hides_bars_data_render_reshows(self):
        """A value scale next to an emptied map would keep claiming
        the previous scan's range."""
        r = _make_renderer()
        _render_all_maps(r, _points())
        for key in _keys(r):
            assert r._heatmap_bars[key].isVisible(), key
        _render_all_maps(r, [])
        for key in _keys(r):
            assert not r._heatmap_bars[key].isVisible(), key
        _render_all_maps(r, _points())
        for key in _keys(r):
            assert r._heatmap_bars[key].isVisible(), key

    def test_rerender_updates_bar_levels(self):
        """autoLevels re-stretch must be REFLECTED by the bar — a stale
        bar would silently lie about the mapping."""
        r = _make_renderer()
        _render_all_maps(r, _points(ia_scale=1.0))
        lo1, hi1 = r._heatmap_bars["contour"].levels()
        _render_all_maps(r, _points(ia_scale=3.0))
        lo2, hi2 = r._heatmap_bars["contour"].levels()
        assert hi2 > hi1 * 2.0, (hi1, hi2)

    def test_bar_created_once_no_leak(self):
        r = _make_renderer()
        for _ in range(3):
            _render_all_maps(r, _points())
        assert len(r._heatmap_bars) == len(_keys(r))

    def test_cmap_change_reaches_bars(self):
        import pyqtgraph as pg
        r = _make_renderer()
        _render_all_maps(r, _points())
        new_cmap = pg.colormap.get("plasma")
        r.update_heatmap_colorbars_cmap(new_cmap)
        assert r.cmap is new_cmap
        # The bar's gradient follows (colorMap() getter on modern pg).
        bar = r._heatmap_bars["contour"]
        assert bar.colorMap() is new_cmap

    def test_cmap_change_reaches_ug2_colorbar_2d(self):
        """The 2D Ug2 colorbar is created once; without the explicit
        update its gradient kept the OLD scheme while curves were
        recolored — a lying legend."""
        import pyqtgraph as pg
        r = _make_renderer()
        r._show_ug2_colorbar(r.cmap, 100.0, 250.0)
        new_cmap = pg.colormap.get("plasma")
        assert r.ug2_colorbar.colorMap() is not new_cmap  # non-degenerate
        r.update_heatmap_colorbars_cmap(new_cmap)
        assert r.ug2_colorbar.colorMap() is new_cmap


# ======================================================================
# 2. Lock scale
# ======================================================================


@pytest.mark.usefixtures("qapp")
class TestLockScale:

    def test_lock_survives_rerender_with_different_data(self):
        r = _make_renderer()
        _render_all_maps(r, _points(ia_scale=1.0))
        locked_levels = {k: tuple(r._heatmap_images()[k].getLevels())
                        for k in _keys(r)}
        r.set_heatmap_scale_locked(True)
        # 3x hotter AND different Ug1 sensitivity: uniform scaling
        # alone leaves mu = Gm*Rp invariant, so a dropped level-apply
        # at the mu call-site would be invisible (degenerate data).
        _render_all_maps(r, _points(ia_scale=3.0, ug1_gain=0.5))
        for key in _keys(r):
            image = r._heatmap_images()[key]
            assert tuple(image.getLevels()) == pytest.approx(
                locked_levels[key]), f"{key}: lock did not hold"
            # ...and the bar shows the LOCKED mapping, not the new max.
            assert tuple(r._heatmap_bars[key].levels()) == pytest.approx(
                locked_levels[key]), key

    def test_cmap_change_preserves_levels_under_lock(self):
        """setColorMap re-asserts the bar's stored values onto its
        linked image (_update_items) — locked levels must survive a
        colormap swap on every map."""
        import pyqtgraph as pg
        r = _make_renderer()
        _render_all_maps(r, _points(ia_scale=1.0))
        r.set_heatmap_scale_locked(True)
        _render_all_maps(r, _points(ia_scale=3.0, ug1_gain=0.5))
        locked = {k: tuple(r._heatmap_images()[k].getLevels())
                  for k in _keys(r)}
        r.update_heatmap_colorbars_cmap(pg.colormap.get("plasma"))
        for key in _keys(r):
            assert tuple(r._heatmap_images()[key].getLevels()) == \
                pytest.approx(locked[key]), key

    def test_relock_after_unlock_captures_fresh_levels(self):
        """Unlock must CLEAR the capture: after lock->unlock->maps
        emptied->lock, the next render captures ITS OWN levels — a
        kept dict would resurrect the first lock's scale."""
        r = _make_renderer()
        _render_all_maps(r, _points(ia_scale=1.0))
        a_hi = r._heatmap_images()["contour"].getLevels()[1]
        r.set_heatmap_scale_locked(True)
        r.set_heatmap_scale_locked(False)
        _render_all_maps(r, [])                       # maps emptied
        r.set_heatmap_scale_locked(True)              # nothing to capture
        _render_all_maps(r, _points(ia_scale=3.0))    # capture-on-render
        fresh = tuple(r._heatmap_images()["contour"].getLevels())
        assert fresh[1] > a_hi * 2.0, "stale first-lock capture applied"
        _render_all_maps(r, _points(ia_scale=1.0))    # ...and it locks
        assert tuple(r._heatmap_images()["contour"].getLevels()) == \
            pytest.approx(fresh)

    def test_unlock_returns_to_autolevels(self):
        r = _make_renderer()
        _render_all_maps(r, _points(ia_scale=1.0))
        r.set_heatmap_scale_locked(True)
        _render_all_maps(r, _points(ia_scale=3.0))
        r.set_heatmap_scale_locked(False)
        _render_all_maps(r, _points(ia_scale=3.0))
        lo, hi = r._heatmap_images()["contour"].getLevels()
        # Auto again: the top level tracks the 3x data.
        assert hi > 20.0, hi

    def test_lock_with_empty_map_captures_on_first_render(self):
        """Enable the lock BEFORE any data: the first render captures
        its own levels instead of crashing or staying unlocked."""
        r = _make_renderer()
        r.set_heatmap_scale_locked(True)
        _render_all_maps(r, _points(ia_scale=1.0))
        first = tuple(r._heatmap_images()["contour"].getLevels())
        _render_all_maps(r, _points(ia_scale=3.0))
        assert tuple(
            r._heatmap_images()["contour"].getLevels()) == pytest.approx(
            first)


# ======================================================================
# 4. Compare legend semantics
# ======================================================================


def _entry(name: str, lamp_id: str, ug2_vals, triode=False,
           track=False) -> Dict:
    from lm19.constants import TOPOLOGY_PENTODE, TOPOLOGY_TRIODE
    pts = []
    if track:
        for ug1 in (-2.0, -4.0):
            for ua in (100.0, 200.0, 300.0):
                pts.append({"ua": ua, "ug1": ug1,
                            "ia": (5.0 + ug1) * ua / 50.0,
                            "ug2": ua + 30.0})
    else:
        for ug2 in ug2_vals:
            for ug1 in (-2.0, -4.0):
                for ua in (100.0, 200.0, 300.0):
                    pts.append({"ua": ua, "ug1": ug1,
                                "ia": (5.0 + ug1) * ua / 50.0,
                                "ug2": ug2})
    data: Dict = {"topology": TOPOLOGY_TRIODE if triode
                  else TOPOLOGY_PENTODE}
    if track:
        data["scan"] = {"ug2_track_ua": True}
    return {"name": name, "lamp_id": lamp_id, "lamp_type": "EL84",
            "timestamp": "2026-07-18T00:00:00",
            "points": pts,
            "data": data}


@pytest.mark.usefixtures("qapp")
class TestCompareLegend:

    def _tab_with_two_lamps(self):
        from PySide6.QtCore import Qt

        from app.compare_tab import CompareTab
        tab = CompareTab()
        tab.compare_entries = [
            _entry("Lamp A", "A1", (150.0, 250.0)),
            _entry("Lamp B", "B2", (150.0, 250.0)),
        ]
        tab._render_table(tab.compare_entries)
        for row in range(tab.table.rowCount()):
            tab.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
        tab._plot_selected()
        return tab

    @staticmethod
    def _legend_entries(tab):
        """[(label_text, pen), ...] from the pyqtgraph legend."""
        import pyqtgraph as pg
        out = []
        for sample, label in tab._compare_legend.items:
            item = sample.item
            pen = pg.mkPen(item.opts.get("pen"))
            out.append((label.text, pen))
        return out

    def test_lamps_listed_with_their_colors(self):
        import pyqtgraph as pg
        tab = self._tab_with_two_lamps()
        entries = self._legend_entries(tab)
        labels = [txt for txt, _ in entries]
        assert "Lamp A" in labels and "Lamp B" in labels
        # Lamp entries are solid and use two DIFFERENT colors.
        from PySide6.QtCore import Qt
        lamp_pens = [pen for txt, pen in entries
                     if txt in ("Lamp A", "Lamp B")]
        assert all(p.style() == Qt.PenStyle.SolidLine for p in lamp_pens)
        assert lamp_pens[0].color().name() != lamp_pens[1].color().name()
        # Each label carries THAT lamp's table color — "different
        # colors" alone would not catch a swapped/shifted pairing.
        from app.compare_tab import _COL_COLOR
        pens = {txt: pen for txt, pen in entries}
        for row, label in ((0, "Lamp A"), (1, "Lamp B")):
            table_color = pg.mkColor(
                tab.table.item(row, _COL_COLOR).text()).name()
            assert pens[label].color().name() == table_color, label

    def test_ug2_entries_neutral_gray_with_styles(self):
        import pyqtgraph as pg
        from PySide6.QtCore import Qt
        tab = self._tab_with_two_lamps()
        entries = self._legend_entries(tab)
        gray = pg.mkColor(COLOR_ZONE).name()
        ug2_entries = [(txt, pen) for txt, pen in entries
                       if "Ug2" in txt]
        assert len(ug2_entries) == 2                      # 150 and 250
        # Neutral gray — NOT any lamp color (the old conflated legend).
        assert all(pen.color().name() == gray for _, pen in ug2_entries)
        # Distinct line styles mirror the plot styling.
        styles = {pen.style() for _, pen in ug2_entries}
        assert len(styles) == 2
        assert Qt.PenStyle.SolidLine in styles            # first level

    def test_legend_stays_attached_across_replots(self):
        """Any second replot (checkbox toggle, Ug2 filter change) must
        keep the legend IN THE SCENE. pyqtgraph's addLegend() returns
        the existing (detached) legend object without re-parenting, so
        a remove/addLegend cycle makes the legend invisible in the live
        app while its .items list still looks populated to tests."""
        tab = self._tab_with_two_lamps()
        for _ in range(2):                       # 2nd and 3rd replots
            tab._plot_selected()
            legend = tab._compare_legend
            assert legend.scene() is not None
            assert legend.parentItem() is not None
            labels = [txt for txt, _ in self._legend_entries(tab)]
            assert "Lamp A" in labels and "Lamp B" in labels

    def test_duplicate_names_keep_both_colors(self):
        """Two runs sharing a measurement name: hiding the duplicate
        would leave the second plot color unnamed in the legend."""
        from PySide6.QtCore import Qt

        from app.compare_tab import CompareTab
        tab = CompareTab()
        tab.compare_entries = [
            _entry("Twin", "A1", (150.0,)),
            _entry("Twin", "A2", (150.0,)),
        ]
        tab._render_table(tab.compare_entries)
        for row in range(tab.table.rowCount()):
            tab.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
        tab._plot_selected()
        entries = self._legend_entries(tab)
        labels = [txt for txt, _ in entries]
        assert "Twin" in labels and "Twin (2)" in labels
        pens = {txt: pen for txt, pen in entries}
        assert pens["Twin"].color().name() != \
            pens["Twin (2)"].color().name()
        # Boundary twin of the `len(ug2_sorted) > 1` gate: with a
        # SINGLE Ug2 level there is nothing to disambiguate — no gray
        # Ug2 rows may appear (>= 1 mutation dies here).
        assert not any("Ug2" in txt for txt, _ in entries)

    def test_cluster_offset_lamp_keeps_curves_and_styles(self):
        """Lamp B measured 1.5 V above lamp A's Ug2 setpoints (inside
        the 2.0 V cluster threshold): B's per-entry nominals must map
        onto the GLOBAL cluster levels. The old per-entry membership
        test silently dropped every B curve (lying legend) and the
        style lookup fell back to solid, diverging from the gray
        Ug2 legend rows."""
        import pyqtgraph as pg
        from PySide6.QtCore import Qt

        from app.compare_tab import _COL_COLOR, CompareTab
        tab = CompareTab()
        tab.compare_entries = [
            _entry("Lamp A", "A1", (150.0, 250.0)),
            _entry("Lamp B", "B2", (151.5, 251.5)),
        ]
        tab._render_table(tab.compare_entries)
        for row in range(tab.table.rowCount()):
            tab.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
        tab._plot_selected()

        def curves_of(color_hex: str) -> list:
            want = pg.mkColor(color_hex).name()
            out = []
            for item in tab.plot.getPlotItem().listDataItems():
                xs = getattr(item, "xData", None)
                if xs is None or len(xs) == 0:
                    continue          # legend stubs / non-curve items
                pen = pg.mkPen(item.opts.get("pen"))
                if pen.color().name() == want:
                    out.append(item)
            return out

        a_curves = curves_of(tab.table.item(0, _COL_COLOR).text())
        b_curves = curves_of(tab.table.item(1, _COL_COLOR).text())
        assert len(a_curves) == 4                 # 2 Ug1 x 2 Ug2
        assert len(b_curves) == 4, "offset lamp's curves were dropped"
        # Styles come from the GLOBAL style map (Solid for 150-level,
        # Dash for 250-level) — not the solid-only fallback of an
        # unmapped per-entry nominal.
        b_styles = {pg.mkPen(i.opts.get("pen")).style() for i in b_curves}
        assert b_styles == {Qt.PenStyle.SolidLine, Qt.PenStyle.DashLine}

    def test_triode_entries_listed_without_ug2_rows(self):
        """Twin of the pentode path: triode lamps get legend entries
        too, and no Ug2 rows appear (nothing to disambiguate)."""
        from PySide6.QtCore import Qt

        from app.compare_tab import CompareTab
        tab = CompareTab()
        tab.compare_entries = [
            _entry("Tri A", "A1", (0.0,), triode=True),
            _entry("Tri B", "B2", (0.0,), triode=True),
        ]
        tab._render_table(tab.compare_entries)
        for row in range(tab.table.rowCount()):
            tab.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
        tab._plot_selected()
        entries = self._legend_entries(tab)
        labels = [txt for txt, _ in entries]
        assert "Tri A" in labels and "Tri B" in labels
        assert not any("Ug2" in txt for txt in labels)

    def test_track_entries_get_lamp_legend_rows(self):
        """Track-mode twin of the legend builder: ug2-track lamps are
        listed too, and their varying Ug2 spawns no Ug2 rows."""
        from PySide6.QtCore import Qt

        from app.compare_tab import CompareTab
        tab = CompareTab()
        tab.compare_entries = [
            _entry("Trk A", "A1", (), track=True),
            _entry("Trk B", "B2", (), track=True),
        ]
        tab._render_table(tab.compare_entries)
        for row in range(tab.table.rowCount()):
            tab.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
        tab._plot_selected()
        labels = [txt for txt, _ in self._legend_entries(tab)]
        assert "Trk A" in labels and "Trk B" in labels
        assert not any("Ug2" in txt for txt in labels)

    def test_unchecked_ug2_level_absent_from_legend(self):
        """The gray Ug2 rows honor the filter panel: a level whose
        curves are filtered out must not be advertised as a style."""
        tab = self._tab_with_two_lamps()        # levels 150/250 checked
        tab._ug2_checked[250.0] = False
        tab._plot_selected()
        entries = self._legend_entries(tab)
        ug2_rows = [txt for txt, _ in entries if "Ug2" in txt]
        assert len(ug2_rows) == 1
        assert "150" in ug2_rows[0] and "250" not in ug2_rows[0]


# ======================================================================
# 5. MainWindow wiring (dropped connect / dropped forward fail here)
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


class TestMainWindowHeatmapWiring:

    def test_lock_checkbox_drives_renderer_and_rerenders_on_unlock(
            self, window):
        r = window.plot_renderer
        assert r._heatmap_scale_locked is False
        window.heatmap_lock_cb.setChecked(True)
        assert r._heatmap_scale_locked is True
        calls: list = []
        window.plot_mgr.render_slice_plots = lambda: calls.append(1)
        window.heatmap_lock_cb.setChecked(False)
        assert r._heatmap_scale_locked is False
        # Unlock re-renders so the scale is honest immediately.
        assert calls == [1]

    def test_cmap_combo_reaches_renderer_and_bars(self, window):
        combo = window.heatmap_cmap_combo
        current = combo.currentText()
        target = next(combo.itemText(i) for i in range(combo.count())
                      if combo.itemText(i) != current)
        seen: list = []
        orig = window.plot_renderer.update_heatmap_colorbars_cmap
        window.plot_renderer.update_heatmap_colorbars_cmap = (
            lambda cmap: (seen.append(cmap), orig(cmap))[0])
        combo.setCurrentText(target)
        assert len(seen) == 1
        # The forwarded object must be an actual ColorMap — comparing
        # only `renderer.cmap is seen[0]` is self-referential and lets
        # a swapped argument (e.g. the LUT ndarray computed in the same
        # handler) pass the spy.
        import pyqtgraph as pg
        assert isinstance(seen[0], pg.ColorMap)
        # The forward actually ran: renderer.cmap switched.
        assert window.plot_renderer.cmap is seen[0]
