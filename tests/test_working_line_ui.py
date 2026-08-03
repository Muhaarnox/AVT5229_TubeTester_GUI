"""WorkingLineController — the live working-line layer.

Perf pins are designed WITHOUT wall-clock (no flakes): they
discriminate incrementality (items are reused, no full re-render),
debounce (N ticks -> 1 recompute), cache (same params -> 0
recomputes), last-write-wins, and method visibility (the label
equals the method actually used; a fallback changes the label).

Run:  py -m pytest tests/test_working_line_ui.py -v
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import i18n_setup
from i18n_setup import t
from lm19.amplifier.constants import (
    CIRCUIT_PP,
    CIRCUIT_SE_XFMR,
    HD_METHOD_5POINT,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_DFT,
)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    i18n_setup.setup("en")
    return QApplication.instance() or QApplication([])


def _triode_points():
    # Koren synthetic data (hand-made linear curves used to yield
    # hd=None — sparse/degenerate for 5-point — and the label pins
    # passed through the error branch, weaker than intended).
    from lm19.tube_sim import quick_triode
    _, pts = quick_triode("12AU7")
    return pts


@pytest.fixture()
def rig(qapp, qtbot):
    """Engine + plot + controller + controllable params getter."""
    import pyqtgraph as pg
    from PySide6.QtWidgets import QLabel
    from lm19.amp_engine import AmplifierEngine, AmpParams
    from app.working_line import WorkingLineController

    engine = AmplifierEngine()
    engine.set_data(_triode_points(), series_labels={}, srk=None,
                    is_triode=True)
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    info = QLabel()
    state = {"params": AmpParams(ub=300.0, ra=10.0, ug1_bias=-6.0,
                                 half_swing=3.0, hd_method=HD_METHOD_5POINT)}
    ctrl = WorkingLineController(
        plot, engine, lambda: state["params"], info_label=info)
    return ctrl, state, plot, info, engine


def _wait_idle(qtbot, ctrl):
    qtbot.waitUntil(lambda: not ctrl._timer.isActive(), timeout=2000)


class TestDebounceCacheLww:

    def test_n_ticks_one_recompute(self, rig, qtbot, monkeypatch):
        ctrl, state, *_ = rig
        calls = []
        orig = ctrl._engine.compute_working_line
        monkeypatch.setattr(ctrl._engine, "compute_working_line",
                            lambda p: calls.append(1) or orig(p))
        ctrl.set_visible(True)
        _wait_idle(qtbot, ctrl)
        calls.clear()
        # DIFFERENT params each tick: the cache cannot mask a missing
        # debounce (identical ticks would be eaten by the cache).
        for ra in (11.0, 12.0, 13.0, 14.0, 15.0):
            state["params"].ra = ra
            ctrl.schedule()
        _wait_idle(qtbot, ctrl)
        assert len(calls) == 1

    def test_cache_repeat_zero_recompute(self, rig, qtbot, monkeypatch):
        ctrl, state, *_ = rig
        calls = []
        orig = ctrl._engine.compute_working_line
        monkeypatch.setattr(ctrl._engine, "compute_working_line",
                            lambda p: calls.append(1) or orig(p))
        ctrl.set_visible(True)
        _wait_idle(qtbot, ctrl)
        assert len(calls) == 1
        ctrl.schedule()                     # same params
        _wait_idle(qtbot, ctrl)
        assert len(calls) == 1              # cache: 0 new recomputes
        state["params"].ra = 12.0           # change -> recompute
        ctrl.schedule()
        _wait_idle(qtbot, ctrl)
        assert len(calls) == 2

    def test_last_write_wins(self, rig, qtbot, monkeypatch):
        """Params are read when the timer fires — the LAST value is
        computed, intermediate ones are never computed at all."""
        ctrl, state, *_ = rig
        seen = []
        orig = ctrl._engine.compute_working_line
        monkeypatch.setattr(ctrl._engine, "compute_working_line",
                            lambda p: seen.append(p.ra) or orig(p))
        ctrl.set_visible(True)
        _wait_idle(qtbot, ctrl)
        seen.clear()
        for ra in (11.0, 13.0, 17.0):
            state["params"].ra = ra
            ctrl.schedule()
        _wait_idle(qtbot, ctrl)
        assert seen == [17.0]

    def test_invalidate_recomputes_same_params(self, rig, qtbot,
                                               monkeypatch):
        ctrl, state, *_ = rig
        calls = []
        orig = ctrl._engine.compute_working_line
        monkeypatch.setattr(ctrl._engine, "compute_working_line",
                            lambda p: calls.append(1) or orig(p))
        ctrl.set_visible(True)
        _wait_idle(qtbot, ctrl)
        ctrl.invalidate()                   # data changed
        _wait_idle(qtbot, ctrl)
        assert len(calls) == 2


class TestIncrementalItems:

    def test_items_reused_across_ticks(self, rig, qtbot):
        """Incrementality: the same PlotDataItems updated via setData,
        not recreated (the main anti-regression of the 446 ms tick)."""
        ctrl, state, plot, *_ = rig
        ctrl.set_visible(True)
        _wait_idle(qtbot, ctrl)
        line_id = id(ctrl._line_item)
        xd = ctrl._line_item.xData
        xs1 = list(xd if xd is not None else [])
        state["params"].ra = 20.0
        ctrl.schedule()
        _wait_idle(qtbot, ctrl)
        assert id(ctrl._line_item) == line_id
        xd2 = ctrl._line_item.xData
        xs2 = list(xd2 if xd2 is not None else [])
        assert xs1 and xs2
        assert ctrl._line_item in plot.listDataItems()

    def test_reattach_after_full_render_clear(self, rig, qtbot):
        ctrl, state, plot, *_ = rig
        ctrl.set_visible(True)
        _wait_idle(qtbot, ctrl)
        assert ctrl._line_item in plot.listDataItems()
        plot.clear()                        # full scene re-render
        assert ctrl._line_item not in plot.listDataItems()
        ctrl.reattach()
        assert ctrl._line_item in plot.listDataItems()
        xd3 = ctrl._line_item.xData
        assert xd3 is not None and len(xd3)   # data restored
        # Intersection labels are recreated IN THE SCENE (a
        # PlotDataItem remembers data even without re-adding —
        # the discriminator is the TextItems).
        assert ctrl._labels
        assert all(lbl.scene() is not None for lbl in ctrl._labels)

    def test_hidden_no_compute_and_empty_items(self, rig, qtbot,
                                               monkeypatch):
        ctrl, state, plot, info, _ = rig
        calls = []
        monkeypatch.setattr(ctrl._engine, "compute_working_line",
                            lambda p: calls.append(1))
        ctrl.schedule()                     # invisible — silence
        qtbot.wait(300)
        assert calls == []
        ctrl.set_visible(False)
        assert not info.isVisible() or info.text() == ""


class TestEngineViewAudit:
    """View contract holes found by the wiring audit."""

    def test_needs_ug2_returns_code_not_raise(self):
        """Live opens NO dialogs and does not kill the tick: a pentode
        model without a valid Ug2 -> view.error='needs_ug2'."""
        from lm19.amp_engine import AmplifierEngine, AmpParams
        from lm19.tube_sim import quick_pentode
        model, pts = quick_pentode("EL84")
        broken = [{**p, "ug2": 0.0} for p in pts]     # dead sensor
        eng = AmplifierEngine()
        eng.set_data(broken, series_labels={}, srk=None, is_triode=False,
                     series_models={0: model})
        p = AmpParams(ub=300.0, ug1_bias=-7.0, series_id=0)
        v = eng.compute_working_line(p)
        assert v.error == "needs_ug2"
        assert v.polyline == []

    def test_needs_ug2_info_label(self, rig, qtbot, monkeypatch):
        from lm19.amp_engine import WorkingLineView
        ctrl, state, plot, info, _ = rig
        monkeypatch.setattr(
            ctrl._engine, "compute_working_line",
            lambda p: WorkingLineView(circuit=p.circuit,
                                      error="needs_ug2"))
        ctrl.set_visible(True)
        qtbot.waitUntil(lambda: not ctrl._timer.isActive(), timeout=2000)
        assert info.text() == t("plot.Wl_needs_ug2")

    def test_dc_polyline_formulas_xfmr_and_pp(self):
        """DC dash: xfmr — endpoints_dc; PP — (Ub, pp_ra_dc), NOT
        ra_per_tube (mutant discriminator)."""
        from lm19.amp_engine import AmplifierEngine, AmpParams
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        eng = AmplifierEngine()
        eng.set_data(pts, series_labels={}, srk=None, is_triode=False)
        vx = eng.compute_working_line(AmpParams(
            ub=300.0, ra=5.0, ra_dc=0.2, ug1_bias=-7.0,
            circuit=CIRCUIT_SE_XFMR, ug2_filter=250.0))
        assert vx.dc_polyline == [(0.0, 300.0 / 0.2), (300.0, 0.0)]
        vp = eng.compute_working_line(AmpParams(
            ub=300.0, ug1_bias=-7.0, circuit=CIRCUIT_PP, pp_raa=8.0,
            pp_ra_dc=0.1, ug2_filter=250.0))
        assert vp.dc_polyline == [(0.0, 300.0 / 0.1), (300.0, 0.0)]

    def test_display_q_model_fallback(self, monkeypatch):
        """Q for the polyline: data-Q failed -> model fallback. Data-Q
        is self-sufficient on realistic data (it interpolates even
        across holes in the fixtures), so the failure is simulated at
        the unit level and the result is compared directly against
        _find_model_dc_q_point."""
        import lm19.amp_engine as ae
        from lm19.amp_engine import AmplifierEngine, AmpParams
        from lm19.amp_engine import _find_model_dc_q_point
        from lm19.tube_sim import quick_pentode
        model, pts = quick_pentode("EL84")
        eng = AmplifierEngine()
        eng.set_data(pts, series_labels={}, srk=None, is_triode=False,
                     series_models={0: model})
        monkeypatch.setattr(ae, "_find_dc_q_point",
                            lambda *a, **k: None)
        p = AmpParams(ub=300.0, ug1_bias=-7.0, circuit=CIRCUIT_PP, pp_raa=8.0,
                      ug2_filter=250.0, series_id=0)
        v = eng.compute_working_line(p)
        assert v.q_ua is not None and v.q_ia is not None
        assert v.polyline
        from lm19.constants import MODEL_UA_MIN_V, MODEL_UA_MAX_DEFAULT_V
        exp = _find_model_dc_q_point(
            eng._series_models[0], 300.0, 0.1, -7.0, 250.0,
            (MODEL_UA_MIN_V, MODEL_UA_MAX_DEFAULT_V))
        assert exp is not None
        assert v.q_ua != 0 and abs(v.q_ua - exp[0]) < 1e-6
        assert abs(v.q_ia - exp[1]) < 1e-6

    def test_se_model_source_live_parity(self):
        """SE model source (dense-Chebyshev branch): live == analyze."""
        from lm19.amp_engine import AmplifierEngine, AmpParams
        from lm19.tube_sim import quick_pentode
        model, pts = quick_pentode("EL84")
        eng = AmplifierEngine()
        eng.set_data(pts, series_labels={}, srk=None, is_triode=False,
                     series_models={0: model})
        src = model.model_type
        p = AmpParams(ub=300.0, ra=5.0, ug1_bias=-7.0, half_swing=3.0,
                      ug2_filter=250.0, hd_method=HD_METHOD_CHEBYSHEV,
                      sources=[src], series_id=0)
        live = eng.compute_working_line(p)
        full = eng.analyze(p).working_line
        assert live.hd is not None and full.hd is not None
        assert live.hd["thd"] == full.hd["thd"]


class TestFeedLegacyConsumers:
    """The curves tab and Q markers read renderer._load_line_* — the
    controller must feed them (mutant: dropped _feed_legacy)."""

    class _R:
        """Stub renderer mirrors the contract the controller consumes
        (mock rule). draw_qpoint_all: the feed itself redraws the
        Q/swing markers."""

        _load_line_intersections: list = []
        _load_line_analysis = None
        _pp_composite: list = []
        _pp_bias: float = 0.0

        def __init__(self) -> None:
            self.qpoint_redraws = 0

        def draw_qpoint_all(self) -> None:
            self.qpoint_redraws += 1

    def test_renderer_attrs_fed_and_cleared(self, qapp, qtbot):
        import pyqtgraph as pg
        from lm19.amp_engine import AmplifierEngine, AmpParams
        from app.working_line import WorkingLineController
        eng = AmplifierEngine()
        eng.set_data(_triode_points(), series_labels={}, srk=None,
                     is_triode=True)
        plot = pg.PlotWidget()
        qtbot.addWidget(plot)
        renderer = self._R()
        p = AmpParams(ub=300.0, ra=10.0, ug1_bias=-6.0, half_swing=3.0,
                      hd_method=HD_METHOD_5POINT)
        ctrl = WorkingLineController(plot, eng, lambda: p,
                                     renderer=renderer)
        ctrl.set_visible(True)
        qtbot.waitUntil(lambda: not ctrl._timer.isActive(), timeout=2000)
        assert renderer._load_line_intersections
        assert renderer._load_line_analysis is not None
        assert renderer._load_line_analysis.get("ua_0") is not None
        # Geometry arrives with non-5point methods too; the feed
        # itself redraws the markers (they used to go stale).
        assert renderer._load_line_analysis.get("pt_neg") is not None
        assert renderer.qpoint_redraws >= 1
        redraws_before_off = renderer.qpoint_redraws
        ctrl.set_visible(False)
        assert renderer._load_line_intersections == []
        assert renderer._load_line_analysis is None
        assert renderer.qpoint_redraws > redraws_before_off


class TestInvalidateWiringRatchet:
    """A scan-data change must invalidate the live cache: a source
    ratchet over app/ — every plot_mgr.points assignment carries
    working_line.invalidate() nearby (the audit found 3 sites)."""

    def test_invalidate_sites_present(self):
        import re
        total = 0
        for sub in ("app",):
            for py in (PROJECT_ROOT / sub).rglob("*.py"):
                total += py.read_text(encoding="utf-8").count(
                    "working_line.invalidate()")
        assert total >= 3, f"invalidate call sites: {total} < 3"


class TestUlModelFamily:
    """Dashed model UL family — the curves the tube actually follows
    at tap>0; computed by the SAME wrapped object as intersections."""

    def _engine_pp(self, tap):
        from lm19.amp_engine import AmplifierEngine, AmpParams
        from lm19.tube_sim import quick_pentode
        model, pts = quick_pentode("EL84")
        eng = AmplifierEngine()
        eng.set_data(pts, series_labels={}, srk=None, is_triode=False,
                     series_models={0: model})
        p = AmpParams(ub=300.0, ug1_bias=-7.0, half_swing=3.0,
                      circuit=CIRCUIT_PP, pp_raa=8.0, ug2_filter=250.0,
                      hd_method=HD_METHOD_CHEBYSHEV, ul_tap=tap, series_id=0)
        return eng, p, model

    def test_family_matches_wrapped_model_not_raw_ug2(self):
        """Discriminator: curves == the UL-wrapped model (Ug2_eff(Ua)),
        NOT the raw fixed-Ug2 one (the difference is material)."""
        import numpy as np
        from lm19.amplifier import UltralinearModelWrapper
        from lm19.tube_model_base import model_ia_array
        eng, p, model = self._engine_pp(0.35)
        v = eng.compute_working_line(p)
        assert v.model_family
        g, ua, ia = v.model_family[len(v.model_family) // 2]
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.35)
        exp = np.asarray(model_ia_array(
            wrapped, ua, np.full_like(ua, g), 250.0), dtype=float)
        assert np.allclose(ia, exp, rtol=1e-9, atol=1e-9)
        raw = np.asarray(model_ia_array(
            model, ua, np.full_like(ua, g), 250.0), dtype=float)
        assert float(np.max(np.abs(exp - raw))) > 0.5   # raw Ug2 != UL

    def test_family_curves_match_intersection_ug1s(self):
        eng, p, _ = self._engine_pp(0.35)
        v = eng.compute_working_line(p)
        fam_ug1 = [g for g, _ua, _ia in v.model_family]
        assert fam_ug1 == sorted({round(i["ug1"], 2)
                                  for i in v.intersections})

    def test_tap_zero_and_no_model_negative_space(self):
        eng, p, _ = self._engine_pp(0.0)
        assert eng.compute_working_line(p).model_family == []
        # no model: family empty + honest fixed_ug2 note
        from lm19.amp_engine import AmplifierEngine, AmpParams
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        eng2 = AmplifierEngine()
        eng2.set_data(pts, series_labels={}, srk=None, is_triode=False)
        p2 = AmpParams(ub=300.0, ug1_bias=-7.0, circuit=CIRCUIT_PP, pp_raa=8.0,
                       ug2_filter=250.0, ul_tap=0.35)
        v2 = eng2.compute_working_line(p2)
        assert v2.model_family == []
        assert v2.note == "fixed_ug2"

    def test_controller_draws_and_redraws_family(self, qapp, qtbot):
        """Live: family items in the scene; a tap change redraws (the
        cache key includes ul_tap); tap->0 clears."""
        import numpy as np
        import pyqtgraph as pg
        from app.working_line import WorkingLineController
        eng, p, _ = self._engine_pp(0.35)
        plot = pg.PlotWidget()
        qtbot.addWidget(plot)
        state = {"p": p}
        ctrl = WorkingLineController(plot, eng, lambda: state["p"])
        ctrl.set_visible(True)
        qtbot.waitUntil(lambda: not ctrl._timer.isActive(), timeout=3000)
        drawn = [it for it in ctrl._family_items
                 if it.xData is not None and len(it.xData)]
        assert drawn
        assert all(it.scene() is not None for it in drawn)
        first = np.array(drawn[0].yData)
        # tap change -> curve values change
        from lm19.amp_engine import AmpParams
        state["p"] = AmpParams(**{**p.__dict__, "ul_tap": 0.8})
        ctrl.schedule()
        qtbot.waitUntil(lambda: not ctrl._timer.isActive(), timeout=3000)
        second = np.array(ctrl._family_items[0].yData)
        assert not np.array_equal(first, second)
        # tap -> 0: family cleared
        state["p"] = AmpParams(**{**p.__dict__, "ul_tap": 0.0})
        ctrl.schedule()
        qtbot.waitUntil(lambda: not ctrl._timer.isActive(), timeout=3000)
        assert all(it.xData is None or not len(it.xData)
                   for it in ctrl._family_items)

    def test_reattach_restores_family(self, qapp, qtbot):
        import pyqtgraph as pg
        from app.working_line import WorkingLineController
        eng, p, _ = self._engine_pp(0.35)
        plot = pg.PlotWidget()
        qtbot.addWidget(plot)
        ctrl = WorkingLineController(plot, eng, lambda: p)
        ctrl.set_visible(True)
        qtbot.waitUntil(lambda: not ctrl._timer.isActive(), timeout=3000)
        assert any(it.scene() is not None for it in ctrl._family_items)
        plot.clear()
        ctrl.reattach()
        drawn = [it for it in ctrl._family_items
                 if it.xData is not None and len(it.xData)]
        assert drawn
        assert all(it.scene() is not None for it in drawn)


class TestMethodVisibilityRule:
    """Method label on every displayed number."""

    def test_info_carries_method_label(self, rig, qtbot):
        ctrl, state, plot, info, _ = rig
        state["params"].hd_method = HD_METHOD_CHEBYSHEV
        ctrl.set_visible(True)
        _wait_idle(qtbot, ctrl)
        assert t("plot.Wl_method_chebyshev") in info.text()

    def test_dft_without_model_label_shows_5point(self, rig, qtbot):
        """Fallback changes the LABEL: DFT without a model -> 5-point."""
        ctrl, state, plot, info, _ = rig
        state["params"].hd_method = HD_METHOD_DFT
        ctrl.set_visible(True)
        _wait_idle(qtbot, ctrl)
        assert t("plot.Wl_method_5point") in info.text()
        assert t("plot.Wl_method_dft") not in info.text()

    def test_live_hd_equals_analyze_hd_real_data(self, qapp):
        """Live/panel unity on REAL data: live-layer THD == full
        Analyze THD with the same method (single routing — a forced-
        method mutant in compute_working_line is caught here)."""
        from lm19.amp_engine import AmplifierEngine, AmpParams
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        eng = AmplifierEngine()
        eng.set_data(pts, series_labels={}, srk=None, is_triode=False)
        p = AmpParams(ub=300.0, ug1_bias=-7.0, half_swing=3.0,
                      circuit=CIRCUIT_PP, pp_raa=8.0,
                      ug2_filter=250.0, hd_method=HD_METHOD_CHEBYSHEV)
        live = eng.compute_working_line(p)
        full = eng.analyze(p).working_line
        assert live.hd is not None and full.hd is not None
        assert live.hd["thd"] == full.hd["thd"]
        assert live.method_used == full.method_used == HD_METHOD_CHEBYSHEV
        assert live.polyline == full.polyline    # same kink geometry

    def test_budget_overrun_logged(self, rig, qtbot, monkeypatch, caplog):
        """Degradation is visible: exceeding the budget logs WARNING."""
        import time as _time
        from lm19.amp_engine import WorkingLineView
        ctrl, state, *_ = rig

        def slow(p):
            _time.sleep(0.06)
            return WorkingLineView(circuit=p.circuit)

        monkeypatch.setattr(ctrl._engine, "compute_working_line", slow)
        ctrl.set_visible(True)
        with caplog.at_level(logging.WARNING, logger="app.working_line"):
            _wait_idle(qtbot, ctrl)
        assert any("recompute" in r.message and "budget" in r.message
                   for r in caplog.records)
