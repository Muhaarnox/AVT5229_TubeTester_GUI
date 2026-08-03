"""Calibration and reset: UI-path correctness.

Pins:
- ML-030: manual coefficient input outside GAIN/OFFSET_BOUNDS asks
  for confirmation (default No blocks the write, Yes is an expert
  override);
- ML-031: ``prompt_save_if_dirty`` returns False on a failed save
  (a failure != 'changes may be lost');
- ML-032: the wizard blocks an ia_low/ia_high calibration point whose
  current landed in the wrong hardware range (``IA_RANGE_THRESHOLD``
  boundary);
- ML-049: ResetWorker honestly performs ``ug1_settle_s`` after
  setting the cutoff Ug1 (used to be a dead parameter).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lm19.app_config import AppConfig
from lm19.calibration import IA_RANGE_THRESHOLD, CalibrationData


def _make_tab(qapp):
    from app.calibration_tab import CalibrationTab
    client = MagicMock()
    client.is_open.return_value = True
    cal = CalibrationData()
    tab = CalibrationTab(
        get_client=lambda: client,
        get_calibration=lambda: cal,
        set_calibration=lambda c: None,
        app_config=AppConfig(),
        get_write_locked=lambda: False,
        get_hw_busy=lambda: None,
    )
    return tab, cal


def _select_read_channel(tab, channel: str = "ua") -> None:
    """Select the given READ row in the coefficients table."""
    from app.calibration_tab import _CHANNEL_ROWS
    tab.coeff_table.setCurrentCell(_CHANNEL_ROWS.index((channel, "read")), 0)


# ── ML-030: manual bounds validation ─────────────────────────────────

class TestManualBoundsValidation:

    def test_out_of_bounds_declined_not_stored(self, qapp, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        tab, cal = _make_tab(qapp)
        _select_read_channel(tab, "ua")
        tab.gain_spin.setValue(1.45)  # far outside 0.8–1.2
        asked = {}
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: asked.setdefault(
                "v", QMessageBox.StandardButton.No)))
        tab._apply_manual()
        assert asked, "out-of-bounds apply must ask"
        assert cal.get_channel("ua", "read").is_default(), \
            "declined out-of-bounds coefficients must NOT be stored"

    def test_out_of_bounds_confirmed_stores(self, qapp, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        tab, cal = _make_tab(qapp)
        _select_read_channel(tab, "ua")
        tab.gain_spin.setValue(1.45)
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
        tab._apply_manual()
        assert cal.get_channel("ua", "read").gain == pytest.approx(1.45)

    def test_in_bounds_applies_without_dialog(self, qapp, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        tab, cal = _make_tab(qapp)
        _select_read_channel(tab, "ua")
        tab.gain_spin.setValue(1.05)
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: pytest.fail("no dialog expected")))
        tab._apply_manual()
        assert cal.get_channel("ua", "read").gain == pytest.approx(1.05)


# ── ML-031: honest prompt_save_if_dirty ──────────────────────────────

class TestPromptSaveHonesty:

    def test_failed_save_returns_false(self, qapp, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        tab, cal = _make_tab(qapp)
        # make it dirty
        _select_read_channel(tab, "ua")
        tab.gain_spin.setValue(1.05)
        tab._apply_manual()
        assert tab.has_unsaved_changes()
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Save))
        monkeypatch.setattr(
            QMessageBox, "warning", staticmethod(lambda *a, **k: None))
        monkeypatch.setattr(
            CalibrationData, "save",
            lambda self, path: (_ for _ in ()).throw(OSError("disk full")))
        assert tab.prompt_save_if_dirty() is False

    def test_successful_save_returns_true(self, qapp, monkeypatch, tmp_path):
        from PySide6.QtWidgets import QMessageBox
        import app.calibration_tab as ct
        tab, cal = _make_tab(qapp)
        _select_read_channel(tab, "ua")
        tab.gain_spin.setValue(1.05)
        tab._apply_manual()
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Save))
        monkeypatch.setattr(ct, "calibration_path",
                            lambda: tmp_path / "calibration.json")
        assert tab.prompt_save_if_dirty() is True
        assert (tmp_path / "calibration.json").exists()


# ── ML-032: wizard current-range check ───────────────────────────────

class TestWizardCurrentRange:

    def _page(self, qapp, channel: str):
        from app.calibration_wizard import _CurrentMeasurePage
        client = MagicMock()
        client.is_open.return_value = True
        page = _CurrentMeasurePage(
            client, channel, source_ch="ua", default_source_v=100.0,
            title="t", unit="mA", cal_samples=1, cal_interval_ms=1)
        # simulate a completed measurement
        page.dev_stats = {"mean": 1.0, "sigma": 0.0}
        page.meter_spin.setValue(1.0)
        return page

    @pytest.mark.parametrize("channel,reading,ok", [
        ("ia_low", IA_RANGE_THRESHOLD - 5.0, True),
        ("ia_low", IA_RANGE_THRESHOLD + 5.0, False),
        ("ia_high", IA_RANGE_THRESHOLD + 5.0, True),
        ("ia_high", IA_RANGE_THRESHOLD - 5.0, False),
    ])
    def test_range_gate(self, qapp, monkeypatch, channel, reading, ok):
        from PySide6.QtWidgets import QMessageBox
        warned = {}
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a, **k: warned.setdefault("y", True)))
        page = self._page(qapp, channel)
        page.dev_reading = reading
        assert page.validatePage() is ok
        assert bool(warned) is (not ok)

    def test_ig2_has_no_range_gate(self, qapp, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a, **k: pytest.fail("no dialog expected")))
        page = self._page(qapp, "ig2")
        page.dev_reading = 50.0
        assert page.validatePage() is True


# ── ML-049: ResetWorker settle ───────────────────────────────────────

class TestResetWorkerSettle:

    def _worker(self, settle: float):
        from app.workers import ResetWorker
        client = MagicMock()
        client.is_open.return_value = True
        w = ResetWorker(client, ug1_value=-40.0, ug1_settle_s=settle)
        return w, client

    def test_settle_executed_after_ug1(self, qapp):
        w, client = self._worker(0.25)
        slept = []
        w.msleep = slept.append  # instance shadow — no real sleep
        done = []
        w.finished.connect(lambda: done.append(True))
        w._execute()
        assert slept == [250], "ug1_settle_s must actually be slept"
        assert done

    def test_zero_settle_no_sleep(self, qapp):
        w, client = self._worker(0.0)
        slept = []
        w.msleep = slept.append
        w._execute()
        assert slept == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
