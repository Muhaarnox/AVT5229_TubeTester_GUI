"""Smoke tests for UI-adjacent logic paths."""

import json
import logging
import os
import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

# Keep Qt headless-friendly in CI/local terminals.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.compare_tab import CompareTab
from app.main import _setup_logging
from app.plot_manager import PlotManager
from lm19.constants import (
    TOPOLOGY_PENTODE,
)
# AmplifierTab is not imported here directly — controls live in
# AmpControlPanel and computation in AmplifierEngine.

pytestmark = [pytest.mark.smoke_ui]


class _BoolWidget:
    def __init__(self, value: bool):
        self._value = value

    def isChecked(self):
        return self._value


class _ValueWidget:
    def __init__(self, value: float):
        self._value = value

    def value(self):
        return self._value


class _TextCombo:
    def __init__(self, text: str):
        self._text = text

    def count(self):
        return 1

    def currentText(self):
        return self._text


class _AmpEngineMock:
    """Mock AmplifierEngine for PlotManager tests."""
    def __init__(self):
        self.set_data_calls = 0

    def set_data(self, points, **kwargs):
        self.set_data_calls += 1

    def available_models(self):
        return {}


class _AmpPanelMock:
    """Mock AmpControlPanel signal emitter."""
    def __init__(self):
        self.emit_calls = 0
        self._series_id = None

    class settings_changed:
        _calls = 0
        @classmethod
        def emit(cls):
            cls._calls += 1

    def selected_series_id(self):
        return self._series_id

    def set_series_items(self, labels, *, current_sid=None):
        pass

    def set_available_models(self, model_labels):
        pass

    def set_pp_tube_b_items(self, labels):
        pass

    def set_optimizer_model_mode(self, has_model):
        pass


@pytest.mark.smoke
def test_plot_manager_render_no_crash_smoke():
    engine = _AmpEngineMock()
    panel = _AmpPanelMock()
    widgets = {
        "pa_max_cb": _BoolWidget(False),
        "ua_max_cb": _BoolWidget(False),
        "ia_max_limit_cb": _BoolWidget(False),
        "load_line_cb": _BoolWidget(False),
        "amp_engine": engine,
        "amp_control_panel": panel,
        "srk_data": None,
    }
    manager = PlotManager(plot_renderer=object(), widgets=widgets)
    manager.points = []

    manager.render_all()

    assert engine.set_data_calls == 1


@pytest.mark.smoke
def test_compare_tab_overlay_basic_smoke():
    QApplication.instance() or QApplication([])
    tab = CompareTab(marker_lock_px=10)
    emitted = []
    tab.show_on_main_plot.connect(lambda points, labels, colors: emitted.append((points, labels, colors)))

    try:
        entry = {
            "lamp_type": "EL84",
            "lamp_id": "L1",
            "name": "run1",
            "timestamp": "2026-02-25T12:00:00",
            "points": [
                {"ua": 100.0, "ug1": -5.0, "ug2": 250.0, "ia": 8.0, "ig2": 0.2, "uh": 6.3, "ih": 0.7},
                {"ua": 150.0, "ug1": -5.0, "ug2": 250.0, "ia": 12.0, "ig2": 0.3, "uh": 6.3, "ih": 0.7},
            ],
            "data": {"topology": TOPOLOGY_PENTODE},
        }
        tab.compare_entries = [entry]
        tab._render_table(tab.compare_entries)
        tab.table.selectRow(0)

        tab._show_selected_on_main_plot()

        assert emitted
        points, labels, _colors = emitted[0]
        assert len(points) == 2
        assert labels
        assert "series_id" in points[0]
    finally:
        tab.close()


@pytest.mark.smoke
def test_main_logging_setup_smoke(monkeypatch, tmp_path):
    cfg_path = tmp_path / "app.json"
    log_rel = "logs/smoke.log"
    cfg_path.write_text(
        json.dumps(
            {
                "log_level": "INFO",
                "log_file": log_rel,
                "log_file_level": "WARNING",
            }
        ),
        encoding="utf-8",
    )

    import app.main as app_main

    root = logging.getLogger()
    old_handlers = list(root.handlers)
    for h in root.handlers:
        h.close()
    root.handlers = []

    monkeypatch.setattr(app_main, "_cfg_path", cfg_path)
    _setup_logging()

    try:
        assert root.handlers
        log_file = Path(app_main.__file__).resolve().parent.parent / log_rel
        assert log_file.exists()
    finally:
        for h in root.handlers:
            h.close()
        root.handlers = old_handlers


@pytest.mark.smoke
def test_amp_engine_uses_ug2_filter_for_stage_params(monkeypatch):
    """Verify AmplifierEngine passes Ug2-filtered points to compute_stage_params."""
    QApplication.instance() or QApplication([])
    from lm19.amp_engine import AmplifierEngine, AmpParams

    captured = {"ug2_values": []}

    def _fake_find_intersections(points, _ll, ug2_filter=None, ug1_bias=None):
        return [
            {"ug1": -8.0, "ua": 220.0, "ia": 8.0},
            {"ug1": -7.0, "ua": 200.0, "ia": 10.0},
            {"ug1": -6.0, "ua": 180.0, "ia": 12.0},
        ]

    def _fake_compute_stage_params(_isects, _ll, ug1_bias=None, srk=None, points=None, model=None, model_ug2=0.0):
        captured["ug2_values"] = [round(p.get("ug2", 0.0), 1) for p in (points or [])]
        return {"gain": 5.0, "gain_db": 14.0, "zout": 4.5, "df": 1.0, "method": "numerical"}

    # Monkeypatch in amp_engine module (direct imports)
    _dist_result = {
        "hd2": 1.0, "hd3": 1.0, "thd": 1.41, "pout_mw": 120.0,
        "ug1_0": -7.0, "ua_0": 200.0, "ia_0": 10.0,
        "i_max": 12.0, "i_min": 8.0, "ua_max": 220.0, "ua_min": 180.0,
        "half_swing": 3.0, "manual_swing_clamped": False,
        "insufficient_signal": False, "amp_class": "A",
    }
    monkeypatch.setattr("lm19.amp_engine.find_intersections", _fake_find_intersections)
    monkeypatch.setattr("lm19.amp_engine.compute_distortion", lambda *a, **kw: dict(_dist_result))
    monkeypatch.setattr("lm19.amp_engine.compute_imd", lambda *a, **kw: None)
    monkeypatch.setattr("lm19.amp_engine.compute_headroom", lambda *a, **kw: None)
    monkeypatch.setattr("lm19.amp_engine.compute_stage_params", _fake_compute_stage_params)
    monkeypatch.setattr("lm19.amp_engine.sweep_amplitude", lambda *a, **kw: [])
    monkeypatch.setattr("lm19.amp_engine.sweep_ra", lambda *a, **kw: [])

    points = [
        {"ua": 100.0, "ug1": -8.0, "ug2": 100.0, "ia": 4.0, "ig2": 0.2, "series_id": 0},
        {"ua": 150.0, "ug1": -7.0, "ug2": 100.0, "ia": 6.0, "ig2": 0.3, "series_id": 0},
        {"ua": 200.0, "ug1": -6.0, "ug2": 249.0, "ia": 10.0, "ig2": 1.0, "series_id": 0},
        {"ua": 250.0, "ug1": -5.0, "ug2": 250.0, "ia": 14.0, "ig2": 1.5, "series_id": 0},
    ]

    engine = AmplifierEngine()
    engine.set_data(points, is_triode=False)
    result = engine.analyze(AmpParams(
        ub=250.0, ra=5.0, ug1_bias=-7.0, half_swing=3.0,
        ug2_filter=250.0,
    ))
    assert captured["ug2_values"]
    assert all(abs(v - 250.0) <= 5.0 for v in captured["ug2_values"])
