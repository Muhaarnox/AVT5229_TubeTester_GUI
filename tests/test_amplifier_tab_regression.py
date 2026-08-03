"""Regression tests for ``AmplifierTab``.

Covers the plot-only widget: ``AmplifierTab`` renders an
``AnalysisResult``; controls live in ``AmpControlPanel`` and
computation in ``AmplifierEngine``.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from app.amplifier_tab import AmplifierTab
from lm19.amp_engine import AmpParams, AnalysisResult, SourceResult
from lm19.amplifier import (
    ResistiveLoadLine,
    TransformerLoadLine,
    PushPullLoadLine,
    CathodeFollowerLoadLine,
)
from lm19.amplifier.constants import (
    CIRCUIT_PP,
    CIRCUIT_SE,
    HD_METHOD_5POINT,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_CHEBYSHEV_PP,
)

pytestmark = [pytest.mark.smoke_ui]


@pytest.fixture(autouse=True)
def _ensure_qapp():
    QApplication.instance() or QApplication([])


@pytest.fixture
def tab():
    t = AmplifierTab()
    yield t
    t.close()


# ── Fake results ──────────────────────────────────────────────────

_FAKE_DIST = {
    "hd2": 2.8, "hd3": 1.5, "thd": 3.2, "pout_mw": 1250.0,
    "ug1_0": -7.0, "ua_0": 200.0, "ia_0": 10.0,
    "i_max": 14.0, "i_min": 6.0,
    "ua_max": 220.0, "ua_min": 180.0,
    "half_swing": 3.0,
    "manual_swing_clamped": False,
    "requested_half_swing": 3.0,
    "insufficient_signal": False,
    "amp_class": "A",
    "pdc_mw": 2500.0, "eta_pct": 50.0,
}

_FAKE_STAGE = {"gain": 20.0, "gain_db": 26.0, "zout": 4.5, "df": 1.1, "method": "numerical"}
_FAKE_HEADROOM = {"max_swing": 4.5, "clip_neg": "cutoff", "clip_pos": "supply"}
_FAKE_IMD = {"imd2": 1.5, "imd3": 0.3, "imd_total": 1.53}

_FAKE_SWEEP_AMP = [
    {"half_swing": 1.0, "hd2": 1.0, "hd3": 0.5, "thd": 1.1, "pout_mw": 100},
    {"half_swing": 2.0, "hd2": 2.0, "hd3": 0.8, "thd": 2.2, "pout_mw": 400},
    {"half_swing": 3.0, "hd2": 2.8, "hd3": 1.5, "thd": 3.2, "pout_mw": 1250},
]

_FAKE_SWEEP_RA = [
    {"ra": 3.0, "hd2": 3.5, "hd3": 1.0, "thd": 3.6, "pout_mw": 2000},
    {"ra": 5.0, "hd2": 2.8, "hd3": 1.5, "thd": 3.2, "pout_mw": 1250},
    {"ra": 8.0, "hd2": 2.0, "hd3": 1.8, "thd": 2.7, "pout_mw": 800},
]


def _make_se_result(**overrides) -> AnalysisResult:
    """Build a standard SE AnalysisResult."""
    sr = SourceResult(
        dist=dict(_FAKE_DIST),
        imd=dict(_FAKE_IMD),
        headroom=dict(_FAKE_HEADROOM),
        stage=dict(_FAKE_STAGE),
        sweep_amp=list(_FAKE_SWEEP_AMP),
        sweep_ra=list(_FAKE_SWEEP_RA),
        method_used="chebyshev",
    )
    r = AnalysisResult(
        per_source={"measurements": sr},
        load_line=ResistiveLoadLine(250.0, 5.0),
        circuit=CIRCUIT_SE,
        params=AmpParams(ub=250, ra=5, ug1_bias=-7.0, half_swing=3.0),
    )
    for k, v in overrides.items():
        setattr(r, k, v)
    return r


def _make_pp_result() -> AnalysisResult:
    sr = SourceResult(
        headroom=dict(_FAKE_HEADROOM),
        sweep_amp=list(_FAKE_SWEEP_AMP),
        method_used="5point",
    )
    return AnalysisResult(
        per_source={"measurements": sr},
        pp_dist={
            "hd2": 0.5, "hd3": 0.3, "thd": 0.58, "pout_mw": 5000.0,
            "balance_error": 1.2,
        },
        load_line=PushPullLoadLine(250.0, 8.0),
        circuit=CIRCUIT_PP,
        params=AmpParams(ub=250, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_PP, pp_raa=8.0),
    )


# ═══════════════════════════════════════════════════════════════════
# 1. CREATION AND STRUCTURE
# ═══════════════════════════════════════════════════════════════════

class TestCreation:
    """Verify AmplifierTab creates expected plot widgets."""

    def test_creates_without_error(self):
        tab = AmplifierTab()
        assert tab is not None
        tab.close()

    def test_has_thd_pout_plot(self, tab):
        assert hasattr(tab, "thd_pout_plot")
        assert tab.thd_pout_plot is not None

    def test_has_hd_ra_plot(self, tab):
        assert hasattr(tab, "hd_ra_plot")
        assert tab.hd_ra_plot is not None

    def test_has_secondary_viewboxes(self, tab):
        assert hasattr(tab, "thd_pout_vb2")
        assert hasattr(tab, "hd_ra_vb2")

    def test_no_controls(self, tab):
        """Controls moved to AmpControlPanel."""
        assert not hasattr(tab, "circuit_combo")
        assert not hasattr(tab, "source_combo")
        assert not hasattr(tab, "pa_max_spin")
        assert not hasattr(tab, "nfb_check")
        assert not hasattr(tab, "results_label")


# ═══════════════════════════════════════════════════════════════════
# 2. RENDER SE RESULTS
# ═══════════════════════════════════════════════════════════════════

class TestRenderSE:
    """Test render() for SE circuits."""

    def test_render_returns_html(self, tab):
        result = _make_se_result()
        html = tab.render(result)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_render_contains_q_point(self, tab):
        html = tab.render(_make_se_result())
        assert "200" in html  # Ua_0
        assert "10" in html   # Ia_0

    def test_render_contains_thd(self, tab):
        html = tab.render(_make_se_result())
        assert "THD" in html
        assert "3.2" in html  # THD value

    def test_render_contains_headroom(self, tab):
        html = tab.render(_make_se_result())
        assert "4.5" in html  # headroom swing

    def test_render_contains_gain(self, tab):
        html = tab.render(_make_se_result())
        assert "20.0" in html  # gain
        assert "26.0" in html  # gain_db

    def test_render_contains_imd(self, tab):
        html = tab.render(_make_se_result())
        assert "1.5" in html  # imd2

    def test_render_shows_hd_method_line(self, tab):
        """Result HTML must include 'HD method: <name>' line so user knows
        which method (5point/chebyshev/dft) generated the THD value."""
        html = tab.render(_make_se_result())
        assert "HD method" in html
        # _make_se_result has dist without 'method' key, so falls back to
        # SourceResult.method_used="chebyshev"
        assert "chebyshev" in html

    def test_render_uses_dist_method_when_present(self, tab):
        """When dist['method'] is set (e.g. by compute_distortion_chebyshev),
        formatter prefers it over sr.method_used."""
        sr = SourceResult(
            dist={**_FAKE_DIST, "method": HD_METHOD_5POINT},
            stage=dict(_FAKE_STAGE),
            method_used="chebyshev",
        )
        result = AnalysisResult(
            per_source={"measurements": sr},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0),
        )
        html = tab.render(result)
        # dist['method']='5point' wins over method_used='chebyshev'
        assert "HD method: 5point" in html

    def test_render_with_nfb(self, tab):
        sr = SourceResult(
            dist=dict(_FAKE_DIST),
            stage=dict(_FAKE_STAGE),
            nfb={"nfb_db": 6.0, "gain_closed": 10.0, "gain_closed_db": 20.0,
                 "zout_closed": 1.5, "thd_closed": 1.6, "bw_factor": 2.0,
                 "desensitivity": 2.0},
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=list(_FAKE_SWEEP_RA),
            method_used="chebyshev",
        )
        result = AnalysisResult(
            per_source={"measurements": sr},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0, nfb_db=6.0),
        )
        html = tab.render(result)
        assert "NFB" in html

    def test_render_with_pg2(self, tab):
        sr = SourceResult(
            dist=dict(_FAKE_DIST),
            pg2_mw=500.0,
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=list(_FAKE_SWEEP_RA),
            method_used="chebyshev",
        )
        result = AnalysisResult(
            per_source={"measurements": sr},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0),
        )
        html = tab.render(result)
        assert "Pg2" in html


# ═══════════════════════════════════════════════════════════════════
# 3. RENDER PP RESULTS
# ═══════════════════════════════════════════════════════════════════

class TestRenderPP:
    """Test render() for Push-Pull circuits."""

    def test_pp_render_returns_html(self, tab):
        html = tab.render(_make_pp_result())
        assert isinstance(html, str)

    def test_pp_contains_balance_error(self, tab):
        html = tab.render(_make_pp_result())
        assert "1.2" in html  # balance_error

    def test_pp_contains_thd(self, tab):
        html = tab.render(_make_pp_result())
        assert "0.58" in html  # THD

    def test_pp_shows_hd_method_line(self, tab):
        """PP result must show 'HD method: <name>' line.
        _make_pp_result has pp_dist without 'method' key → no line shown
        because formatter only emits when pp_dist['method'] is truthy."""
        # Inject method key
        sr = SourceResult(headroom=dict(_FAKE_HEADROOM))
        result = AnalysisResult(
            per_source={"measurements": sr},
            pp_dist={
                "hd2": 0.5, "hd3": 0.3, "thd": 0.58, "pout_mw": 5000.0,
                "balance_error": 1.2,
                "method": HD_METHOD_CHEBYSHEV_PP,
            },
            load_line=PushPullLoadLine(250.0, 8.0),
            circuit=CIRCUIT_PP,
            params=AmpParams(ub=250, ra=8, ug1_bias=-7.0),
        )
        html = tab.render(result)
        assert "HD method: chebyshev_pp" in html


# ═══════════════════════════════════════════════════════════════════
# 4. ERROR STATES
# ═══════════════════════════════════════════════════════════════════

class TestErrors:
    """Test error display."""

    def test_no_data_error(self, tab):
        result = AnalysisResult(error="no_data")
        html = tab.render(result)
        assert html  # non-empty error message

    def test_enable_ll_error(self, tab):
        result = AnalysisResult(error="enable_ll")
        html = tab.render(result)
        assert html

    def test_not_enough_isects_error(self, tab):
        result = AnalysisResult(error="not_enough_isects")
        html = tab.render(result)
        assert html

    def test_pp_no_tube_b_error(self, tab):
        result = AnalysisResult(error="pp_no_tube_b")
        html = tab.render(result)
        assert html

    def test_unknown_error(self, tab):
        result = AnalysisResult(error="some_weird_error")
        html = tab.render(result)
        assert html


# ═══════════════════════════════════════════════════════════════════
# 5. CLEAR
# ═══════════════════════════════════════════════════════════════════

class TestClear:
    """Test clear method."""

    def test_clear_after_render(self, tab):
        tab.render(_make_se_result())
        tab.clear()
        # No crash


# ═══════════════════════════════════════════════════════════════════
# 6. PLOT DATA
# ═══════════════════════════════════════════════════════════════════

class TestPlotData:
    """Verify plots are populated by render()."""

    def test_thd_pout_has_items_after_render(self, tab):
        tab.render(_make_se_result())
        items = tab.thd_pout_plot.getPlotItem().listDataItems()
        assert len(items) > 0

    def test_hd_ra_has_items_after_render(self, tab):
        tab.render(_make_se_result())
        items = tab.hd_ra_plot.getPlotItem().listDataItems()
        assert len(items) > 0

    def test_empty_sweep_no_crash(self, tab):
        sr = SourceResult(
            dist=dict(_FAKE_DIST),
            sweep_amp=[],
            sweep_ra=[],
            method_used="5point",
        )
        result = AnalysisResult(
            per_source={"measurements": sr},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0),
        )
        html = tab.render(result)
        assert html  # Still produces results text

    def test_no_dist_shows_failed(self, tab):
        sr = SourceResult(method_used="5point")
        result = AnalysisResult(
            per_source={"measurements": sr},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0),
        )
        html = tab.render(result)
        assert html  # Shows "analysis_failed" text


# ═══════════════════════════════════════════════════════════════════
# 7. MULTI-SOURCE OVERLAY
# ═══════════════════════════════════════════════════════════════════

class TestMultiSource:
    """Test multi-source overlay rendering."""

    def _make_multi_result(self):
        sr1 = SourceResult(
            dist=dict(_FAKE_DIST),
            imd=dict(_FAKE_IMD),
            headroom=dict(_FAKE_HEADROOM),
            stage=dict(_FAKE_STAGE),
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=list(_FAKE_SWEEP_RA),
            method_used="chebyshev",
        )
        sr2 = SourceResult(
            dist={**_FAKE_DIST, "hd2": 3.5, "thd": 4.0, "pout_mw": 1100},
            stage={**_FAKE_STAGE, "gain": 18.0},
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=list(_FAKE_SWEEP_RA),
            method_used="dft",
        )
        return AnalysisResult(
            per_source={"measurements": sr1, "koren": sr2},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0, sources=["measurements", "koren"]),
        )

    def test_multi_source_returns_html(self, tab):
        html = tab.render(self._make_multi_result())
        assert isinstance(html, str)
        assert len(html) > 0

    def test_multi_source_has_table(self, tab):
        html = tab.render(self._make_multi_result())
        assert "<table" in html

    def test_multi_source_has_delta(self, tab):
        html = tab.render(self._make_multi_result())
        assert "Δ%" in html or "%" in html

    def test_multi_source_has_both_names(self, tab):
        html = tab.render(self._make_multi_result())
        assert "measurements" in html
        assert "koren" in html

    def test_multi_source_plots_populated(self, tab):
        tab.render(self._make_multi_result())
        thd_items = tab.thd_pout_plot.getPlotItem().listDataItems()
        assert len(thd_items) >= 2  # At least 2 THD curves


# ═══════════════════════════════════════════════════════════════════
# 8. FORMAT EDGE CASES
# ═══════════════════════════════════════════════════════════════════

class TestFormatEdgeCases:
    """Edge cases in results formatting."""

    def test_swing_clamped_message(self, tab):
        """Manual swing clamped shows warning."""
        sr = SourceResult(
            dist={
                **_FAKE_DIST,
                "manual_swing_clamped": True,
                "requested_half_swing": 5.0,
                "half_swing": 3.0,
            },
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=list(_FAKE_SWEEP_RA),
            method_used="chebyshev",
        )
        result = AnalysisResult(
            per_source={"measurements": sr},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0, half_swing=5.0),
        )
        html = tab.render(result)
        assert "5.0" in html  # requested
        assert "3.0" in html  # actual

    def test_insufficient_signal_message(self, tab):
        """Insufficient signal shows warning."""
        sr = SourceResult(
            dist={
                **_FAKE_DIST,
                "insufficient_signal": True,
            },
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=list(_FAKE_SWEEP_RA),
            method_used="chebyshev",
        )
        result = AnalysisResult(
            per_source={"measurements": sr},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0),
        )
        html = tab.render(result)
        assert html  # non-empty

    def test_multi_source_missing_dist_no_crash(self, tab):
        """One source has dist, another doesn't → no crash, delta shows —."""
        sr1 = SourceResult(
            dist=dict(_FAKE_DIST),
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=list(_FAKE_SWEEP_RA),
            method_used="chebyshev",
        )
        sr2 = SourceResult(method_used="dft")  # No dist
        result = AnalysisResult(
            per_source={"measurements": sr1, "koren": sr2},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0, sources=["measurements", "koren"]),
        )
        html = tab.render(result)
        assert "—" in html  # dash for missing values

    def test_multi_source_zero_ref_value_no_crash(self, tab):
        """ref_val=0 in delta calc → shows — instead of division error."""
        sr1 = SourceResult(
            dist={**_FAKE_DIST, "hd2": 0.0, "hd3": 0.0, "thd": 0.0},
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=list(_FAKE_SWEEP_RA),
            method_used="chebyshev",
        )
        sr2 = SourceResult(
            dist={**_FAKE_DIST, "hd2": 1.0, "hd3": 0.5, "thd": 1.1},
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=list(_FAKE_SWEEP_RA),
            method_used="dft",
        )
        result = AnalysisResult(
            per_source={"measurements": sr1, "koren": sr2},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0, sources=["measurements", "koren"]),
        )
        html = tab.render(result)
        assert html  # No division by zero crash

    def test_pp_no_pp_dist_shows_failed(self, tab):
        """PP result without pp_dist → analysis_failed."""
        sr = SourceResult(
            headroom=dict(_FAKE_HEADROOM),
            sweep_amp=list(_FAKE_SWEEP_AMP),
            method_used="5point",
        )
        result = AnalysisResult(
            per_source={"measurements": sr},
            pp_dist=None,  # Missing!
            load_line=PushPullLoadLine(250.0, 8.0),
            circuit=CIRCUIT_PP,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_PP, pp_raa=8.0),
        )
        html = tab.render(result)
        assert html  # Shows analysis_failed text

    def test_source_color_known(self, tab):
        """Known sources get their defined colors."""
        from app.ui_theme import SOURCE_COLORS
        color = tab._source_color("measurements")
        assert color == SOURCE_COLORS["measurements"]

    def test_source_color_unknown(self, tab):
        """Unknown source gets default color."""
        from app.ui_theme import SOURCE_COLOR_DEFAULT
        color = tab._source_color("unknown_source")
        assert color == SOURCE_COLOR_DEFAULT

    def test_clear_resets_both_plots(self, tab):
        tab.render(_make_se_result())
        tab.clear()
        items1 = tab.thd_pout_plot.getPlotItem().listDataItems()
        items2 = tab.hd_ra_plot.getPlotItem().listDataItems()
        assert len(items1) == 0
        assert len(items2) == 0


class TestHD45Display:
    """HD4/HD5 display in text results and plots."""

    def test_hd45_above_threshold_shown_in_text(self, tab):
        """HD4/HD5 >= 0.1% appear in results HTML."""
        dist = {**_FAKE_DIST, "hd4": 0.5, "hd5": 0.3}
        sr = SourceResult(
            dist=dist,
            headroom=dict(_FAKE_HEADROOM),
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=list(_FAKE_SWEEP_RA),
            method_used="chebyshev",
        )
        result = AnalysisResult(
            per_source={"measurements": sr},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0),
        )
        html = tab.render(result)
        assert "HD4=0.50%" in html
        assert "HD5=0.30%" in html

    def test_hd45_below_threshold_hidden_in_text(self, tab):
        """HD4/HD5 < 0.1% do not appear in results HTML."""
        dist = {**_FAKE_DIST, "hd4": 0.05, "hd5": 0.01}
        sr = SourceResult(
            dist=dist,
            headroom=dict(_FAKE_HEADROOM),
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=list(_FAKE_SWEEP_RA),
            method_used="chebyshev",
        )
        result = AnalysisResult(
            per_source={"measurements": sr},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0),
        )
        html = tab.render(result)
        assert "HD4" not in html
        assert "HD5" not in html

    def test_hd45_absent_no_crash(self, tab):
        """Missing hd4/hd5 in dist dict does not crash."""
        sr = SourceResult(
            dist=dict(_FAKE_DIST),  # no hd4/hd5 keys
            headroom=dict(_FAKE_HEADROOM),
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=list(_FAKE_SWEEP_RA),
            method_used="5point",
        )
        result = AnalysisResult(
            per_source={"measurements": sr},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0),
        )
        html = tab.render(result)
        assert html  # no crash, non-empty

    def test_interp_sweep_basic(self, tab):
        """Crosshair interpolation between two sweep points."""
        tab._ra_sweep_data = [
            {"ra": 5.0, "thd": 3.0, "hd2": 2.0, "pout_mw": 1000, "gain": 10.0, "zout": 4.0, "pa_mw": 2000},
            {"ra": 10.0, "thd": 2.0, "hd2": 1.0, "pout_mw": 500, "gain": 20.0, "zout": 8.0, "pa_mw": 1500},
        ]
        interp = tab._interp_sweep(7.5)
        assert interp is not None
        assert abs(interp["thd"] - 2.5) < 0.01
        assert abs(interp["gain"] - 15.0) < 0.01

    def test_interp_sweep_out_of_range(self, tab):
        """Out-of-range Ra returns None."""
        tab._ra_sweep_data = [
            {"ra": 5.0, "thd": 3.0},
            {"ra": 10.0, "thd": 2.0},
        ]
        assert tab._interp_sweep(1.0) is None
        assert tab._interp_sweep(15.0) is None

    def test_interp_sweep_empty(self, tab):
        tab._ra_sweep_data = []
        assert tab._interp_sweep(5.0) is None

    def test_ra_clicked_signal_exists(self, tab):
        """AmplifierTab has ra_clicked signal."""
        assert hasattr(tab, "ra_clicked")

    def test_ra_markers_rendered(self, tab):
        """Min THD and max Pout markers rendered without crash."""
        sr = SourceResult(
            dist=dict(_FAKE_DIST),
            headroom=dict(_FAKE_HEADROOM),
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=list(_FAKE_SWEEP_RA),
            method_used="chebyshev",
        )
        result = AnalysisResult(
            per_source={"measurements": sr},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0),
        )
        tab.render(result)
        # Check that plot has items (curves + markers + lines)
        items = tab.hd_ra_plot.getPlotItem().listDataItems()
        assert len(items) > 3  # HD2, HD3, THD + at least markers

    def test_ra_markers_empty_sweep_no_crash(self, tab):
        """Empty sweep_ra → no markers, no crash."""
        sr = SourceResult(
            dist=dict(_FAKE_DIST),
            headroom=dict(_FAKE_HEADROOM),
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=[],
            method_used="chebyshev",
        )
        result = AnalysisResult(
            per_source={"measurements": sr},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0),
        )
        tab.render(result)  # no crash

    def test_chebyshev_limited_notice_shown(self, tab):
        """When Chebyshev max_harmonic < 9, notice is shown."""
        dist = {**_FAKE_DIST, "method": HD_METHOD_CHEBYSHEV, "max_harmonic": 5}
        sr = SourceResult(
            dist=dist,
            headroom=dict(_FAKE_HEADROOM),
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=list(_FAKE_SWEEP_RA),
            method_used="chebyshev",
        )
        result = AnalysisResult(
            per_source={"measurements": sr},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0),
        )
        html = tab.render(result)
        assert "HD2" in html
        assert "HD5" in html  # "HD2–HD5" in the limited notice

    def test_chebyshev_full_no_notice(self, tab):
        """When Chebyshev max_harmonic == 9, no limitation notice."""
        dist = {**_FAKE_DIST, "method": HD_METHOD_CHEBYSHEV, "max_harmonic": 9}
        sr = SourceResult(
            dist=dist,
            headroom=dict(_FAKE_HEADROOM),
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=list(_FAKE_SWEEP_RA),
            method_used="chebyshev",
        )
        result = AnalysisResult(
            per_source={"measurements": sr},
            load_line=ResistiveLoadLine(250.0, 5.0),
            circuit=CIRCUIT_SE,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0),
        )
        html = tab.render(result)
        # No "chebyshev_limited" key content — "need more" should not appear
        assert "need more" not in html.lower()

    def test_hd45_in_pp_results(self, tab):
        """HD4/HD5 shown in PP results when above threshold."""
        pp_dist = {
            "hd2": 0.5, "hd3": 2.0, "hd4": 0.2, "hd5": 0.15,
            "thd": 2.1, "pout_mw": 2000,
            "balance_error": 1.0,
        }
        # PP render needs at least one source in per_source to pass the empty check
        sr = SourceResult(
            dist=dict(_FAKE_DIST),
            sweep_amp=list(_FAKE_SWEEP_AMP),
            sweep_ra=[],
        )
        result = AnalysisResult(
            per_source={"measurements": sr},
            pp_dist=pp_dist,
            load_line=PushPullLoadLine(250.0, 8.0),
            circuit=CIRCUIT_PP,
            params=AmpParams(ub=250, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_PP, pp_raa=8.0),
        )
        html = tab.render(result)
        assert "HD4=0.20%" in html
        assert "HD5=0.15%" in html


class TestParetoPlot:
    """Pareto rendering on left plot."""

    def _make_opt_result(self):
        from lm19.optimizer import OptPoint, OptimizerResult
        pts = [
            OptPoint(ub=250, ug2=0, ug1=-5, ra=5, thd=1.0, hd2=0.8, hd3=0.3,
                     pout_mw=500, pa_mw=2000, ia_0=10, ua_0=200, amp_class="A", max_swing=3.0),
            OptPoint(ub=250, ug2=0, ug1=-7, ra=10, thd=2.0, hd2=1.5, hd3=0.8,
                     pout_mw=1200, pa_mw=1800, ia_0=8, ua_0=225, amp_class="A", max_swing=4.0),
            OptPoint(ub=250, ug2=0, ug1=-9, ra=15, thd=3.5, hd2=2.5, hd3=1.5,
                     pout_mw=2000, pa_mw=1500, ia_0=6, ua_0=250, amp_class="AB", max_swing=5.0),
        ]
        return OptimizerResult(
            grid_points=pts,
            pareto_front=pts,
            best=pts[0],
            refined=pts[0],
        )

    def test_render_pareto_no_crash(self, tab):
        opt = self._make_opt_result()
        tab.render_pareto(opt)
        assert tab._pareto_mode is True

    def test_clear_pareto_restores_mode(self, tab):
        opt = self._make_opt_result()
        tab.render_pareto(opt)
        tab.clear_pareto()
        assert tab._pareto_mode is False

    def test_pareto_data_stored(self, tab):
        opt = self._make_opt_result()
        tab.render_pareto(opt)
        assert len(tab._pareto_data) == 3

    def test_pareto_empty_result_no_crash(self, tab):
        from lm19.optimizer import OptimizerResult
        opt = OptimizerResult(error="no_valid_points")
        tab.render_pareto(opt)

    def test_pareto_clicked_signal_exists(self, tab):
        assert hasattr(tab, "pareto_clicked")
