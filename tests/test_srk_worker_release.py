"""#26 — SrkController worker release follows the cleanup() canon.

The old code did ``worker.wait(500); worker = None`` ignoring the wait
result: a thread that had not stopped in 500 ms lost its last reference —
a live QThread freed by GC aborts the process — and its still-connected
signals could contaminate the next measurement. ``_release_worker`` now
uses ``BaseWorker.cleanup() -> bool`` and RETAINS the reference when the
thread won't stop; ``measure_after_scan`` falls back to scan-data SRK
instead of double-commanding the hardware.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PySide6.QtWidgets import QApplication

from app.srk_widget import SrkController

QApplication.instance() or QApplication([])


def _stub_worker(running: bool, cleanup_ok: bool) -> MagicMock:
    w = MagicMock()
    w.isRunning.return_value = running
    w.cleanup.return_value = cleanup_ok
    return w


class TestReleaseWorker:
    def test_no_worker_is_safe(self) -> None:
        c = SrkController()
        assert c._release_worker() is True

    def test_stopped_worker_cleaned_and_dropped(self) -> None:
        c = SrkController()
        w = _stub_worker(running=False, cleanup_ok=True)
        c.worker = w
        assert c._release_worker() is True
        w.cleanup.assert_called_once()
        assert c.worker is None

    def test_running_worker_refused_and_retained(self) -> None:
        c = SrkController()
        w = _stub_worker(running=True, cleanup_ok=True)
        c.worker = w
        assert c._release_worker() is False
        assert c.worker is w                 # reference kept

    def test_stuck_worker_reference_retained(self) -> None:
        """The #26 pin: cleanup() timing out must NOT drop the reference
        (the old wait-and-drop did — GC of a live QThread aborts)."""
        c = SrkController()
        w = _stub_worker(running=False, cleanup_ok=False)
        c.worker = w
        assert c._release_worker() is False
        assert c.worker is w                 # revert → dropped → fails


class TestMeasureAfterScanBusyFallback:
    def test_busy_worker_falls_back_to_scan_data(self) -> None:
        """A still-running previous SRK must not be raced: the scan flow
        completes via the scan-data fallback path instead."""
        c = SrkController()
        c.worker = _stub_worker(running=True, cleanup_ok=True)
        pts = [{"ua": 100.0, "ug1": -2.0, "ia": 5.0, "ug2": 0.0}]
        with patch.object(c, "_on_after_scan_failed") as fallback, \
                patch("app.srk_widget.SrkWorker") as worker_cls:
            c.measure_after_scan(
                MagicMock(), {"ua_min": 50, "ua_max": 150, "ug1_min": -3,
                              "ug1_max": -1, "ug2": 0},
                MagicMock(), MagicMock(), False, 0.0, 1,
                pending_points=pts, pending_meta={})
            fallback.assert_called_once()
            worker_cls.assert_not_called()   # no second worker started
        assert c._pending_points == pts      # payload staged for fallback

    def test_free_worker_starts_measurement(self) -> None:
        c = SrkController()
        with patch("app.srk_widget.SrkWorker") as worker_cls:
            c.measure_after_scan(
                MagicMock(), {"ua_min": 50, "ua_max": 150, "ug1_min": -3,
                              "ug1_max": -1, "ug2": 0},
                MagicMock(), MagicMock(), False, 0.0, 1,
                pending_points=[], pending_meta={})
            worker_cls.assert_called_once()
            worker_cls.return_value.start.assert_called_once()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
