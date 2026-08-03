"""Graph honesty audit fixes.

Pins:
  1.  Pa map: a REAL Pa_max isoline (IsocurveItem, child of the
      ImageItem, level in watts 1:1 with the grid) — previously only
      a text label was drawn and the pen/array were dead code.
  2.  Isolines do not leak between re-renders and are removed by
      renderer.clear() — plot.removeItem() on an ImageItem child is a
      silent no-op, hence the separate cleanup (_remove_overlay_items).
  3.  Geometry: the isoline lies between known Ua indices
      (discriminates lost transposition / a foreign grid).
  4.  Gm map: the nominal_s parameter revived — an S_nom isoline.
  5.  Rp/mu hover markers receive UNclipped grids (clipping is display
      contrast only; the tooltip must show the computed value).
  6.  Amp tooltips/extremum markers: source label in multi-source
      (previously the first source was shown silently).
  7.  Dead _render_thd_pout/_render_hd_ra removed (they carried a
      units bug: raw mW on a W axis) — negative pin.
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

pytestmark = [pytest.mark.smoke_ui]


# ── Module local helpers ──

UA_VALS = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
UG1_VALS = [-1.0, -2.0, -3.0, -4.0]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


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
    r.is_triode = True   # single Ug2 slice → no ug2 filtering in grids
    r.track_sids = set()
    return r


def _flat_ia_points(ia_ma: float = 20.0) -> List[Dict]:
    """Ia identical everywhere → Pa = Ua·Ia/1000 depends on Ua ONLY.

    Pa runs 1.0…6.0 W over UA_VALS — a Pa_max level between two known
    Ua columns pins the iso-line's position/orientation.
    """
    return [{"ua": ua, "ug1": ug1, "ia": ia_ma, "ug2": 0.0, "series_id": 0}
            for ug1 in UG1_VALS for ua in UA_VALS]


def _gm_points() -> List[Dict]:
    """Physically plausible grid with ONE extreme-Rp cell.

    Ia grows with Ua (slope (5+ug1)/50 mA/V → Rp ≈ 12…50 kΩ) except on
    the ug1=-2 row between Ua 250→300, where the slope is 1e-4 mA/V —
    just above the EPS_COARSE=1e-6 validity guard — giving Rp = 10⁴ kΩ,
    far beyond the p95×1.5 display clip (single cell of 20 → outside
    the 95th percentile). NOT the last Ug1 row: µ uses Rp rows
    [0..n_ug1-2] only, so an outlier there must propagate into µ too.
    """
    pts = []
    for ug1 in UG1_VALS:
        for ua in UA_VALS:
            ia = (5.0 + ug1) * ua / 50.0 + 8.0
            if ug1 == -2.0 and ua == 300.0:
                ia = (5.0 + ug1) * 250.0 / 50.0 + 8.0 + 0.005
            pts.append({"ua": ua, "ug1": ug1, "ia": ia, "ug2": 0.0,
                        "series_id": 0})
    return pts


def _iso_items(overlay_items):
    import pyqtgraph as pg
    return [it for it in overlay_items if isinstance(it, pg.IsocurveItem)]


def _iso_children(image):
    import pyqtgraph as pg
    return [ch for ch in image.childItems()
            if isinstance(ch, pg.IsocurveItem)]


def _path_xy(iso):
    """Extract (x, y) vertices of the isocurve path (item coords)."""
    if iso.path is None:
        iso.generatePath()
    p = iso.path
    return [(p.elementAt(i).x, p.elementAt(i).y)
            for i in range(p.elementCount())]


# ======================================================================
# 1-3. Pa_max iso-line
# ======================================================================


@pytest.mark.usefixtures("qapp")
class TestPaMaxIsoLine:

    def test_iso_created_level_parent_and_label(self):
        import pyqtgraph as pg
        r = _make_renderer()
        r.render_pa_map(_flat_ia_points(), pa_max=2.5)
        isos = _iso_items(r._pa_overlay_items)
        assert len(isos) == 1
        assert isos[0].level == 2.5           # watts, 1:1 with pa_grid
        assert isos[0].parentItem() is r.pa_map_image
        labels = [it for it in r._pa_overlay_items
                  if isinstance(it, pg.TextItem)]
        assert labels and "Pa_max=2.5" in labels[0].textItem.toPlainText()

    def test_iso_geometry_between_known_ua_columns(self):
        """Pa depends on Ua only → the iso-line must be a straight cut
        between the Ua=100 (2.0 W) and Ua=150 (3.0 W) columns, spanning
        all Ug1 rows. A lost transpose / wrong grid flips the axes."""
        r = _make_renderer()
        r.render_pa_map(_flat_ia_points(ia_ma=20.0), pa_max=2.5)
        iso = _iso_items(r._pa_overlay_items)[0]
        pts = _path_xy(iso)
        assert pts, "iso-line produced an empty path"
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        # Image data is pa_grid.T → first axis (item x) is the Ua index.
        # Level 2.5 W lies exactly between Ua indices 1 and 2.
        assert 1.0 <= min(xs) and max(xs) <= 2.0, (min(xs), max(xs))
        # ...and the cut spans the Ug1 rows (4 rows → extent ≥ 2).
        assert max(ys) - min(ys) >= 2.0, (min(ys), max(ys))

    def test_no_iso_without_pa_max(self):
        r = _make_renderer()
        r.render_pa_map(_flat_ia_points(), pa_max=None)
        assert _iso_items(r._pa_overlay_items) == []
        assert _iso_children(r.pa_map_image) == []

    def test_iso_above_data_range_no_crash(self):
        r = _make_renderer()
        r.render_pa_map(_flat_ia_points(), pa_max=999.0)
        iso = _iso_items(r._pa_overlay_items)[0]
        assert _path_xy(iso) == []            # nothing to draw — no crash

    def test_rerender_does_not_leak_isolines(self):
        """plot.removeItem() no-ops on ImageItem children — the cleanup
        helper must detach them, else lines pile up render after render."""
        r = _make_renderer()
        for _ in range(3):
            r.render_pa_map(_flat_ia_points(), pa_max=2.5)
        assert len(_iso_children(r.pa_map_image)) == 1
        assert len(_iso_items(r._pa_overlay_items)) == 1

    def test_clear_detaches_isolines(self):
        r = _make_renderer()
        r.render_pa_map(_flat_ia_points(), pa_max=2.5)
        r.clear()
        assert _iso_children(r.pa_map_image) == []
        assert r._pa_overlay_items == []


# ======================================================================
# 4. S-nominal iso-line on the Gm map
# ======================================================================


@pytest.mark.usefixtures("qapp")
class TestGmNominalIso:

    def test_s_nom_iso_level_parent_and_label(self):
        import pyqtgraph as pg
        r = _make_renderer()
        r.render_gm_rp(_gm_points(), nominal_s=0.06)
        isos = _iso_items(r._gm_overlay_items)
        assert len(isos) == 1
        assert isos[0].level == 0.06          # mA/V, 1:1 with gm grid
        assert isos[0].parentItem() is r.gm_image
        labels = [it for it in r._gm_overlay_items
                  if isinstance(it, pg.TextItem)]
        assert labels and "S nom=" in labels[0].textItem.toPlainText()

    def test_no_iso_without_nominal_s(self):
        r = _make_renderer()
        r.render_gm_rp(_gm_points(), nominal_s=None)
        assert _iso_items(r._gm_overlay_items) == []

    def test_rerender_does_not_leak(self):
        r = _make_renderer()
        for _ in range(3):
            r.render_gm_rp(_gm_points(), nominal_s=0.06)
        assert len(_iso_children(r.gm_image)) == 1


# ======================================================================
# 5. Rp/µ hover markers get UNCLIPPED grids
# ======================================================================


@pytest.mark.usefixtures("qapp")
class TestUnclippedMarkerGrids:

    def test_rp_marker_reports_computed_value_not_clipped(self):
        """Display clips at p95×1.5 for contrast; the tooltip grid must
        keep the computed value (the exact regression being fixed)."""
        r = _make_renderer()
        r.render_gm_rp(_gm_points())
        marker_max = np.nanmax(r._marker_rp._z_grid)
        display_max = np.nanmax(r.rp_image.image)
        assert display_max < marker_max, (
            f"marker grid is clipped: display {display_max:.1f} vs "
            f"marker {marker_max:.1f}")
        # sanity: the extreme cell survived the slope-validity guard
        assert marker_max == pytest.approx(1e4, rel=0.01)

    def test_mu_marker_reports_computed_value_not_clipped(self):
        r = _make_renderer()
        r.render_gm_rp(_gm_points())
        marker_max = np.nanmax(r._marker_mu._z_grid)
        display_max = np.nanmax(r.mu_image.image)
        assert display_max < marker_max


# ======================================================================
# 6-7. Amplifier tab: source tags + dead code removed
# ======================================================================


def _sweep_amp_row(hs: float) -> Dict:
    return {"half_swing": hs, "hd2": 1.0, "hd3": 0.5, "thd": 1.2,
            "pout_mw": 1000.0 * hs}


def _sweep_ra_row(ra: float) -> Dict:
    return {"ra": ra, "hd2": 1.0, "hd3": 0.5, "thd": 1.2 + ra * 0.1,
            "pout_mw": 500.0 * ra}


def _result(source_names: List[str]):
    from lm19.amp_engine import AmpParams, AnalysisResult, SourceResult
    per = {
        name: SourceResult(
            sweep_amp=[_sweep_amp_row(hs) for hs in (2.0, 4.0, 6.0)],
            sweep_ra=[_sweep_ra_row(ra) for ra in (2.0, 5.0, 8.0)],
        )
        for name in source_names
    }
    return AnalysisResult(per_source=per, params=AmpParams())


@pytest.fixture()
def amp_tab(qapp):
    from app.amplifier_tab import AmplifierTab
    return AmplifierTab()


class TestAmpSourceTags:

    def test_single_source_no_tag(self, amp_tab):
        amp_tab._render_thd_pout_multi(_result(["measurements"]))
        assert amp_tab._amp_sweep_source is None
        text = amp_tab._amp_tooltip_text(3.0, _sweep_amp_row(3.0))
        assert "[" not in text

    def test_multi_source_tooltip_tagged_with_first_source(self, amp_tab):
        amp_tab._render_thd_pout_multi(_result(["measurements", "dempwolf"]))
        assert amp_tab._amp_sweep_source == "measurements"
        text = amp_tab._amp_tooltip_text(3.0, _sweep_amp_row(3.0))
        assert text.startswith("<b>[measurements]</b>")

    def test_multi_then_single_resets_tag(self, amp_tab):
        """Stale tag must not survive a re-render with one source."""
        amp_tab._render_thd_pout_multi(_result(["measurements", "dempwolf"]))
        amp_tab._render_thd_pout_multi(_result(["dempwolf"]))
        assert amp_tab._amp_sweep_source is None

    def test_ra_tooltip_and_markers_tagged(self, amp_tab):
        import pyqtgraph as pg
        amp_tab._render_hd_ra_multi(_result(["measurements", "koren"]))
        assert amp_tab._ra_sweep_source == "measurements"
        text = amp_tab._ra_tooltip_text(5.0, _sweep_ra_row(5.0))
        assert text.startswith("<b>[measurements]</b>")
        # min-THD / max-Pout extremum labels carry the tag too
        marker_texts = [
            it.textItem.toPlainText()
            for it in amp_tab.hd_ra_plot.getPlotItem().items
            if isinstance(it, pg.TextItem)
        ]
        tagged = [tx for tx in marker_texts if "[measurements]" in tx]
        assert len(tagged) == 2, marker_texts

    def test_ra_single_source_markers_untagged(self, amp_tab):
        import pyqtgraph as pg
        amp_tab._render_hd_ra_multi(_result(["measurements"]))
        assert amp_tab._ra_sweep_source is None
        marker_texts = [
            it.textItem.toPlainText()
            for it in amp_tab.hd_ra_plot.getPlotItem().items
            if isinstance(it, pg.TextItem)
        ]
        assert marker_texts and all("[" not in tx for tx in marker_texts)

    def test_clear_plots_resets_tags(self, amp_tab):
        amp_tab._render_thd_pout_multi(_result(["a", "b"]))
        amp_tab._render_hd_ra_multi(_result(["a", "b"]))
        amp_tab._clear_plots()
        assert amp_tab._amp_sweep_source is None
        assert amp_tab._ra_sweep_source is None

    def test_dead_single_source_renderers_removed(self, amp_tab):
        """Units-bugged dead code must not come back silently."""
        assert not hasattr(amp_tab, "_render_thd_pout")
        assert not hasattr(amp_tab, "_render_hd_ra")
