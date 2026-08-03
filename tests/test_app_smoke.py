"""Smoke tests for app bootstrap and main window construction."""

import os
import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

# Keep Qt headless-friendly in CI/local terminals.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow

pytestmark = [pytest.mark.smoke_ui]


@pytest.mark.smoke
def test_app_boots_and_qapplication_created():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    assert app.applicationName() is not None


@pytest.mark.smoke
def test_main_window_constructs_and_core_tabs_present():
    QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.centralWidget() is not None
        assert hasattr(window, "manual_tab")
        assert hasattr(window, "compare_tab")
        assert hasattr(window, "amplifier_tab")
        assert window.tabs.count() >= 3
    finally:
        window.close()


@pytest.mark.smoke
def test_mainwindow_load_lamps_and_ports_smoke(monkeypatch):
    class _Port:
        def __init__(self, device: str):
            self.device = device

    monkeypatch.setattr("app.main_window.list_ports.comports", lambda: [_Port("COM7"), _Port("COM9")])

    QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.lamp_combo.count() > 0
        assert window.port_combo.count() == 2
        assert window.port_combo.itemText(0) == "COM7"
    finally:
        window.close()
