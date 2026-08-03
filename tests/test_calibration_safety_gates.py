"""Hardware-write safety gates for the Calibration tab + wizard (#14).

Calibration writes setpoints like every other subsystem, so it must honour
the emergency write-lock and the hw-busy arbiter. These tests pin that the
"Test set" buttons and the wizard launch refuse to write when the lock is
set or another subsystem owns the hardware, and that the per-page write
guard skips the in-wizard set_param.

All headless (QT_QPA_PLATFORM=offscreen) with a MagicMock client — no real
serial.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from PySide6.QtWidgets import QApplication

import app.calibration_tab as cal_tab_mod
from app.calibration_tab import CalibrationTab, _CHANNEL_ROWS
from app.calibration_wizard import _CurrentMeasurePage, _VoltageMeasurePage
from lm19.app_config import AppConfig
from lm19.calibration import CalibrationData

QApplication.instance() or QApplication([])

_UA_ROW = _CHANNEL_ROWS.index(("ua", "read"))


def _make_tab(locked, busy):
    """CalibrationTab with a mock open client and flag-cell gates.

    ``locked`` / ``busy`` are 1-element lists so a test can flip them
    after construction.
    """
    client = MagicMock()
    client.is_open.return_value = True
    tab = CalibrationTab(
        get_client=lambda: client,
        get_calibration=lambda: CalibrationData(),
        set_calibration=lambda c: None,
        app_config=AppConfig(),
        get_write_locked=lambda: locked[0],
        get_hw_busy=lambda: busy[0],
    )
    tab.coeff_table.setCurrentCell(_UA_ROW, 0)   # select a settable channel
    tab.test_spin.setValue(100.0)
    return tab, client


class TestTestSetGated(unittest.TestCase):
    """The "Test set" / "Test set raw" buttons honour both gates (#14)."""

    @patch.object(cal_tab_mod.QMessageBox, "warning")
    def test_emergency_lock_blocks_calibrated_set(self, _warn):
        locked, busy = [True], [None]
        tab, client = _make_tab(locked, busy)
        tab._set_test_value()
        client.set_param.assert_not_called()      # FAILS under revert (no gate)
        # Unlock → the same action now writes (proves the gate, not a dead button).
        locked[0] = False
        tab._set_test_value()
        client.set_param.assert_called_once()

    @patch.object(cal_tab_mod.QMessageBox, "warning")
    def test_emergency_lock_blocks_raw_set(self, _warn):
        # The raw "Test set raw" button shares the choke point — also gated.
        locked, busy = [True], [None]
        tab, client = _make_tab(locked, busy)
        tab._set_test_value_raw()
        client.set_param.assert_not_called()
        locked[0] = False
        tab._set_test_value_raw()
        client.set_param.assert_called_once()

    @patch.object(cal_tab_mod.QMessageBox, "warning")
    def test_hw_busy_blocks_set(self, _warn):
        locked, busy = [False], ["scan"]
        tab, client = _make_tab(locked, busy)
        tab._set_test_value()
        client.set_param.assert_not_called()
        busy[0] = None
        tab._set_test_value()
        client.set_param.assert_called_once()


class TestWizardLaunchGated(unittest.TestCase):
    """Launching the wizard is blocked under lock/busy — its first page
    commands a setpoint, so the launch itself must be gated (#14)."""

    @patch.object(cal_tab_mod.QMessageBox, "warning")
    @patch("app.calibration_wizard.CalibrationWizard")
    def test_emergency_lock_blocks_wizard_launch(self, mock_wizard, _warn):
        locked, busy = [True], [None]
        tab, _client = _make_tab(locked, busy)
        tab._start_wizard()
        mock_wizard.assert_not_called()           # never even constructed

    @patch.object(cal_tab_mod.QMessageBox, "warning")
    @patch("app.calibration_wizard.CalibrationWizard")
    def test_hw_busy_blocks_wizard_launch(self, mock_wizard, _warn):
        locked, busy = [False], ["health"]
        tab, _client = _make_tab(locked, busy)
        tab._start_wizard()
        mock_wizard.assert_not_called()

    @patch.object(cal_tab_mod.QMessageBox, "warning")
    @patch("app.calibration_wizard.CalibrationWizard")
    def test_free_hardware_launches_wizard(self, mock_wizard, _warn):
        # Guard against over-blocking: free + unlocked → the wizard launches.
        mock_wizard.return_value.exec.return_value = False
        locked, busy = [False], [None]
        tab, _client = _make_tab(locked, busy)
        tab._start_wizard()
        mock_wizard.assert_called_once()


class TestPerPageWriteGuard(unittest.TestCase):
    """Defense-in-depth: a wizard page skips its set_param when the guard
    rejects (lock/busy flipped while the modal wizard is open) (#14)."""

    @patch("time.sleep")
    def test_voltage_initialize_respects_guard(self, _sleep):
        client = MagicMock()
        page = _VoltageMeasurePage(
            client, "ua", 50.0, "Low", "V", 1, 0, write_guard=lambda: False)
        page.initializePage()
        page.cleanupPage()
        client.set_param.assert_not_called()      # FAILS under revert

    @patch("time.sleep")
    def test_voltage_initialize_writes_when_allowed(self, _sleep):
        client = MagicMock()
        page = _VoltageMeasurePage(
            client, "ua", 50.0, "Low", "V", 1, 0, write_guard=lambda: True)
        page.initializePage()
        page.cleanupPage()
        client.set_param.assert_called_once_with("Ua", 50)

    @patch("time.sleep")
    def test_current_apply_respects_guard(self, _sleep):
        client = MagicMock()
        page = _CurrentMeasurePage(
            client, "ia_low", "ua", 100.0, "Low", "mA", 1, 0,
            write_guard=lambda: False)
        page._apply_voltage()
        client.set_param.assert_not_called()

    @patch("time.sleep")
    def test_current_apply_writes_when_allowed(self, _sleep):
        client = MagicMock()
        page = _CurrentMeasurePage(
            client, "ia_low", "ua", 100.0, "Low", "mA", 1, 0,
            write_guard=lambda: True)
        page._apply_voltage()
        client.set_param.assert_called_once_with("Ua", 100)


if __name__ == "__main__":
    unittest.main()
