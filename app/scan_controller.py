"""Scan orchestration — owns scan, preheat, and reset workers."""

import logging
import time
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Signal

from lm19.app_config import AppConfig
from lm19.calibration import CalibrationData
from lm19.protocol import LM19Serial
from lm19.scan import ScanSettings, scan_point_count
from app.workers import ScanWorker, PreheatWorker, ResetWorker

log = logging.getLogger(__name__)

# ── module local constants ──
# How long shutdown() waits for each worker to stop before giving up and
# keeping its reference (so a still-running QThread is not freed by GC).
_WORKER_SHUTDOWN_WAIT_MS = 1500


class ScanController(QObject):
    """Manages scan, preheat, and reset worker lifecycle.

    Emits signals for all scan events — does NOT touch UI widgets.
    """

    # --- scan signals ---
    scan_started = Signal()
    scan_progress = Signal(object)          # point dict (data or event)
    scan_finished = Signal(list)            # list of points
    scan_failed = Signal(str)               # error message
    scan_comm_error = Signal(str, int)      # message, attempt

    # --- preheat signals ---
    preheat_started = Signal()
    preheat_progress = Signal(float, float, int)  # uh, ih, remaining
    preheat_finished = Signal()
    preheat_failed = Signal(str)

    # --- reset signals ---
    reset_finished = Signal()
    reset_failed = Signal(str)

    def __init__(self, app_config: AppConfig, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._app_config = app_config
        self._scan_worker: Optional[ScanWorker] = None
        self._preheat_worker: Optional[PreheatWorker] = None
        self._reset_worker: Optional[ResetWorker] = None

        self.scan_in_progress: bool = False
        self.reset_on_finish: bool = False
        self.preheat_done: bool = False
        self.scan_total_points: int = 0
        self._scan_start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def scan_worker(self) -> Optional[ScanWorker]:
        return self._scan_worker

    @property
    def preheat_worker(self) -> Optional[PreheatWorker]:
        return self._preheat_worker

    @property
    def reset_worker(self) -> Optional[ResetWorker]:
        return self._reset_worker

    @property
    def is_scanning(self) -> bool:
        return self.scan_in_progress

    @property
    def scan_start_time(self) -> Optional[float]:
        return self._scan_start_time

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def start_scan(self, client: LM19Serial, settings: ScanSettings) -> None:
        """Start a scan with the given settings."""
        if self._scan_worker and self._scan_worker.isRunning():
            return
        # Drop any stale signal connections from a previous worker. Without
        # this, queued signals (progress/finished/comm_error) from the prior
        # worker's run() can fire on the new worker's slots after re-assign.
        if self._scan_worker:
            self._scan_worker.cleanup()
            self._scan_worker = None
        self.scan_total_points = scan_point_count(settings)
        self._scan_start_time = time.monotonic()
        self.scan_in_progress = True
        self.reset_on_finish = False

        self._scan_worker = ScanWorker(client, settings)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.failed.connect(self._on_scan_failed)
        self._scan_worker.comm_error.connect(self.scan_comm_error.emit)
        self._scan_worker.start()
        self.scan_started.emit()

    def stop_scan(self) -> None:
        """Request scan stop."""
        if self._scan_worker:
            self._scan_worker.stop()
            self.reset_on_finish = True

    def respond_comm_error(self, decision: str) -> None:
        """Forward UI decision to scan worker."""
        if self._scan_worker:
            self._scan_worker.respond_comm_error(decision)

    def _on_scan_progress(self, point: Dict) -> None:
        event = point.get("event")
        if event == "refine_count":
            self.scan_total_points += point.get("count", 0)
        self.scan_progress.emit(point)

    def _on_scan_finished(self, points) -> None:
        self.scan_in_progress = False
        self.scan_finished.emit(points)

    def _on_scan_failed(self, message: str) -> None:
        self.scan_in_progress = False
        self.scan_failed.emit(message)

    # ------------------------------------------------------------------
    # Preheat
    # ------------------------------------------------------------------

    def start_preheat(
        self,
        client: LM19Serial,
        target_uh: float,
        target_ih: float,
        warmup_s: int,
        calibration: Optional[CalibrationData] = None,
    ) -> None:
        """Start the heater preheat sequence.

        *calibration* is threaded into ``PreheatWorker`` (plan B): heater
        setpoints get SET feedforward, progress readings become physical.
        """
        if self._preheat_worker and self._preheat_worker.isRunning():
            return
        if self._preheat_worker:
            self._preheat_worker.cleanup()
            self._preheat_worker = None
        self.preheat_done = False
        if calibration is not None:
            self._preheat_worker = PreheatWorker(
                client, target_uh, target_ih, warmup_s,
                calibration=calibration)
        else:
            # Legacy 4-arg construction kept verbatim: callers without a
            # calibration get the exact pre-plan-B worker call, and
            # PreheatWorker(calibration=None) is identity anyway.
            self._preheat_worker = PreheatWorker(
                client, target_uh, target_ih, warmup_s)
        self._preheat_worker.progress.connect(self.preheat_progress.emit)
        self._preheat_worker.finished.connect(self._on_preheat_finished)
        self._preheat_worker.failed.connect(self.preheat_failed.emit)
        self._preheat_worker.start()
        self.preheat_started.emit()

    def stop_preheat(self) -> None:
        """Stop the preheat worker."""
        if self._preheat_worker:
            self._preheat_worker.stop()

    def _on_preheat_finished(self) -> None:
        self.preheat_done = True
        self.preheat_finished.emit()

    # ------------------------------------------------------------------
    # Reset outputs
    # ------------------------------------------------------------------

    def reset_outputs(
        self,
        client: LM19Serial,
        reset_heater: bool = True,
        reset_order: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Queue a reset worker to safely zero device outputs.

        Returns ``None`` when the reset was queued, or a reason code when
        the request was DROPPED — ``"port_closed"`` (no way to zero the
        outputs) / ``"reset_busy"`` (another reset already running). The
        caller must surface a dropped request to the operator (ML-090):
        a silently ignored reset leaves the tube energized while the UI
        implies a safe state.
        """
        if not client or not client.is_open():
            log.warning("reset_outputs dropped — port closed; outputs "
                        "keep their last setpoints")
            return "port_closed"
        # When this reset zeroes the heater, drain a still-running preheat
        # worker first: PreheatWorker writes Uh/Ih setpoints between ramp steps
        # and would re-assert the heater after ResetWorker zeroes it (the
        # firmware serializes single commands, not the two logical sequences).
        # stop_preheat() only flags the worker; here we wait for it to actually
        # exit. Keep the reference if it does not drain — a live QThread freed
        # by GC aborts the process.
        if (reset_heater and self._preheat_worker
                and self._preheat_worker.isRunning()):
            if self._preheat_worker.cleanup():
                self._preheat_worker = None
            else:
                log.warning(
                    "preheat worker did not drain before heater reset — "
                    "Uh/Ih re-assert risk")
        if self._reset_worker and self._reset_worker.isRunning():
            log.warning("reset_outputs dropped — a reset worker is already "
                        "running (requested reset_heater=%s ignored)",
                        reset_heater)
            return "reset_busy"
        if self._reset_worker:
            self._reset_worker.cleanup()
            self._reset_worker = None
        self._reset_worker = ResetWorker(
            client,
            self._app_config.ug1_after_stop,
            ug1_settle_s=self._app_config.ug1_settle_s,
            reset_heater=reset_heater,
            order=reset_order,
        )
        self._reset_worker.finished.connect(self.reset_finished.emit)
        self._reset_worker.failed.connect(self.reset_failed.emit)
        self._reset_worker.start()
        self.reset_on_finish = False
        return None

    # ------------------------------------------------------------------
    # Emergency stop
    # ------------------------------------------------------------------

    def stop_all(self) -> None:
        """Stop all active workers immediately."""
        workers = [self._scan_worker, self._preheat_worker, self._reset_worker]
        for w in workers:
            if w and w.isRunning():
                w.stop()
        for w in workers:
            if w and w.isRunning():
                w.wait(1200)
        self.scan_in_progress = False

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop all workers. Call from closeEvent.

        A worker that does not stop within the timeout keeps its reference so
        a still-running QThread is not freed by GC (which aborts the process);
        it is logged and left to terminate with the process.
        """
        attrs = ["_scan_worker", "_preheat_worker", "_reset_worker"]
        for attr in attrs:
            w = getattr(self, attr)
            if w and w.isRunning():
                w.stop()
        for attr in attrs:
            w = getattr(self, attr)
            if w and w.isRunning() and not w.wait(_WORKER_SHUTDOWN_WAIT_MS):
                log.warning("%s still running at shutdown — keeping reference",
                            type(w).__name__)
                continue
            setattr(self, attr, None)
