"""Worker lifecycle pins (ML-002 / ML-003 / ML-010).

- ML-010: stop() vs _handle_comm_error() race — clear() erased the
  stop() wakeup and wait() blocked forever (zombie worker).
- ML-002: HealthTab reassigned/dropped its worker without
  BaseWorker.cleanup() — stale queued signals + live-QThread GC abort.
- ML-003: opt_cancel_btn.clicked was connected to each new worker
  per run without disconnect — stacking connections kept old workers
  reachable forever.

Run:  py -m pytest tests/test_worker_lifecycle.py -v
"""

import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from i18n_setup import t

# ── module local constants ──
# Generous vs the global 10 s pytest timeout, tiny vs a real deadlock.
_JOIN_TIMEOUT_S = 3.0
# Long enough for the handler thread to enter wait(); still trivial.
_HANDLER_ENTER_DELAY_S = 0.1


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ═══════════════════════════════════════════════════════════════════
#  ML-010: ScanWorker stop() vs _handle_comm_error() race
# ═══════════════════════════════════════════════════════════════════

def _make_scan_worker():
    from app.workers import ScanWorker
    from lm19.scan.settings import ScanRange, ScanSettings
    settings = ScanSettings(
        ua=ScanRange(0.0, 10.0, 5.0),
        ug1=ScanRange(-5.0, 0.0, 1.0),
        ug2=ScanRange(0.0, 0.0, 1.0),
        uh=6.3, ih=0.3,
    )
    return ScanWorker(client=MagicMock(), settings=settings)


def _run_handler(worker, out):
    out["result"] = worker._handle_comm_error("comm lost", 1)


class TestCommErrorStopRace:

    def test_stop_before_handler_returns_abort(self, qapp):
        """The ML-010 window: stop() lands BEFORE the handler's clear() —
        the wakeup is erased and, without the re-check, wait() blocks
        forever."""
        w = _make_scan_worker()
        w.stop()
        out = {}
        th = threading.Thread(target=_run_handler, args=(w, out), daemon=True)
        th.start()
        th.join(_JOIN_TIMEOUT_S)
        assert not th.is_alive(), \
            "_handle_comm_error deadlocked after stop() (ML-010)"
        assert out["result"] == "abort"

    def test_stop_during_wait_aborts(self, qapp):
        w = _make_scan_worker()
        out = {}
        th = threading.Thread(target=_run_handler, args=(w, out), daemon=True)
        th.start()
        time.sleep(_HANDLER_ENTER_DELAY_S)
        w.stop()
        th.join(_JOIN_TIMEOUT_S)
        assert not th.is_alive()
        assert out["result"] == "abort"

    def test_respond_passes_decision(self, qapp):
        """Normal path unaffected by the race fix."""
        w = _make_scan_worker()
        out = {}
        th = threading.Thread(target=_run_handler, args=(w, out), daemon=True)
        th.start()
        time.sleep(_HANDLER_ENTER_DELAY_S)
        w.respond_comm_error("retry")
        th.join(_JOIN_TIMEOUT_S)
        assert not th.is_alive()
        assert out["result"] == "retry"


# ═══════════════════════════════════════════════════════════════════
#  ML-002: HealthTab worker reattach canon
# ═══════════════════════════════════════════════════════════════════

def _make_health_tab():
    QApplication.instance() or QApplication([])
    from app.app_context import AppContext
    from app.health_tab import HealthTab
    from lm19.app_config import AppConfig
    from lm19.calibration import CalibrationData
    cfg = AppConfig()
    cal = CalibrationData()
    ctx = AppContext(
        get_client=lambda: None,
        get_write_locked=lambda: False,
        get_app_config=lambda: cfg,
        get_calibration=lambda: cal,
        get_lamps=lambda: [],
        get_current_tube_type=lambda: "",
        get_current_lamp_id=lambda: "",
        set_poller_active=lambda _b: None,
    )
    return HealthTab(ctx)


def _fake_worker(*, running: bool = False, cleanup_ok: bool = True):
    w = MagicMock()
    w.isRunning.return_value = running
    w.cleanup.return_value = cleanup_ok
    return w


def _prime_start_path(tab, launched):
    """Stub everything _start_test needs up to the worker guard."""
    tab._validate_test_inputs = lambda: (
        MagicMock(warmup_s=1), "L1", "test", MagicMock())
    tab._check_heater_for_test = lambda _c, _l: True
    tab._resolve_reference = lambda _l: None
    tab._update_reference_info = lambda _r: None
    tab._launch_health_worker = lambda *a, **k: launched.append(a)


class TestHealthWorkerReattach:

    def test_start_cleans_up_finished_worker(self, qapp):
        tab = _make_health_tab()
        old = _fake_worker()
        tab.worker = old
        launched = []
        _prime_start_path(tab, launched)
        tab._start_test()
        old.cleanup.assert_called_once()
        assert tab.worker is None
        assert launched, "new test must start after successful cleanup"

    def test_start_zombie_blocks_restart_visibly(self, qapp, monkeypatch):
        """cleanup() False → keep the reference (live QThread freed by GC
        aborts the process), refuse the start, tell the user."""
        warnings = []
        monkeypatch.setattr(
            "app.health_tab.QMessageBox.warning",
            lambda *a, **k: warnings.append(a))
        tab = _make_health_tab()
        old = _fake_worker(cleanup_ok=False)
        tab.worker = old
        launched = []
        _prime_start_path(tab, launched)
        tab._start_test()
        assert not launched, "zombie thread must block a new start"
        assert tab.worker is old, "zombie reference must be retained"
        assert warnings, "refusal must reach the user (rule 2026-07-04)"

    def test_start_running_worker_shows_dialog(self, qapp, monkeypatch):
        warnings = []
        monkeypatch.setattr(
            "app.health_tab.QMessageBox.warning",
            lambda *a, **k: warnings.append(a))
        tab = _make_health_tab()
        old = _fake_worker(running=True)
        tab.worker = old
        launched = []
        _prime_start_path(tab, launched)
        tab._start_test()
        assert not launched
        old.cleanup.assert_not_called()
        assert warnings

    def test_cleanup_after_test_drops_worker(self, qapp):
        tab = _make_health_tab()
        old = _fake_worker()
        tab.worker = old
        tab._cleanup_after_test()
        old.cleanup.assert_called_once()
        assert tab.worker is None

    def test_cleanup_after_test_keeps_zombie_visible(self, qapp):
        tab = _make_health_tab()
        old = _fake_worker(cleanup_ok=False)
        tab.worker = old
        tab._cleanup_after_test()
        assert tab.worker is old, "zombie reference must be retained"
        assert tab.progress_label.text() == t("health.Worker_stuck")
        assert tab.live_state.text() == t("health.State_error")

    def test_cleanup_after_test_without_worker(self, qapp):
        tab = _make_health_tab()
        tab.worker = None
        tab._cleanup_after_test()   # no crash
        assert tab.worker is None


# ═══════════════════════════════════════════════════════════════════
#  ML-003: Cancel button wired once to a stable slot
# ═══════════════════════════════════════════════════════════════════

class _Port:
    def __init__(self, device: str) -> None:
        self.device = device


@pytest.mark.smoke_ui
class TestOptCancelWiring:

    @pytest.fixture
    def window(self, qapp, monkeypatch):
        monkeypatch.setattr(
            "app.main_window.list_ports.comports",
            lambda: [_Port("COM1")],
        )
        from app.main_window import MainWindow
        w = MainWindow()
        yield w
        w.close()

    def test_cancel_routes_to_current_worker_only(self, window):
        """Reassigning _opt_worker must reroute Cancel to the NEW worker
        without touching the old one — the per-run connect design fanned
        the click out to every worker ever started."""
        w1 = MagicMock()
        w2 = MagicMock()
        window._opt_worker = w1
        window._opt_worker = w2
        # clicked.emit() (not click()): the panel starts disabled without
        # scan data, and a disabled button swallows click(). The pin
        # targets the WIRING — the emit fires the real connection, and
        # without the builder-time connect it lands nowhere.
        window.amp_control_panel.opt_cancel_btn.clicked.emit()
        w2.cancel.assert_called_once()
        w1.cancel.assert_not_called()

    def test_cancel_without_worker_is_noop(self, window):
        window._opt_worker = None
        window.amp_control_panel.opt_cancel_btn.clicked.emit()   # no crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
