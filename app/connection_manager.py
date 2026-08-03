"""Connection lifecycle manager — owns LM19Serial client and ParamPoller."""

import logging
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Signal

from lm19.app_config import AppConfig
from lm19.protocol import LM19Serial
from app.workers import ParamPoller, CheckComWorker

log = logging.getLogger(__name__)


class ConnectionManager(QObject):
    """Manages serial connection, live-param polling, and COM check.

    Emits signals for state changes — does NOT touch UI widgets directly.
    """

    # --- state signals ---
    connected = Signal(str)         # port name
    disconnected = Signal()
    check_ok = Signal(int)          # Ua raw value
    check_fail = Signal(str)        # error message

    # --- live data signals ---
    live_params = Signal(object)    # dict of raw param values
    live_error = Signal(str)        # poller error

    # --- IO activity signals ---
    io_stats_updated = Signal(int, int, object)  # tx, rx, last_rx (float|None)

    def __init__(self, app_config: AppConfig, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._app_config = app_config
        self._client: Optional[LM19Serial] = None
        self._param_poller: Optional[ParamPoller] = None
        self._check_com_worker: Optional[CheckComWorker] = None
        # Workers that did not stop within their cleanup timeout — retained so
        # GC never frees a live QThread (which aborts the process). Never
        # dropped; they wind down on their own and die with the process.
        self._zombies: List = []
        self._last_tx_count: Optional[int] = None
        self._last_rx_count: Optional[int] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def client(self) -> Optional[LM19Serial]:
        return self._client

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_open()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect_port(self, port: str) -> None:
        """Open serial port and start live polling."""
        self._client = LM19Serial(
            port,
            timeout=self._app_config.serial_timeout_s,
            write_timeout=self._app_config.serial_write_timeout_s,
            read_param_timeout=self._app_config.read_param_timeout_s,
            read_lcd_timeout=self._app_config.read_lcd_timeout_s,
            set_param_delay_s=self._app_config.serial_set_param_delay_s,
        )
        self._client.open()
        self._last_tx_count = None
        self._last_rx_count = None
        self._start_poller()
        self.connected.emit(port)

    def disconnect(self) -> None:
        """Stop polling, close port, emit disconnected."""
        self._stop_poller()
        if self._client and self._client.is_open():
            self._client.close()
        self._client = None
        self._last_tx_count = None
        self._last_rx_count = None
        self.disconnected.emit()

    # ------------------------------------------------------------------
    # COM check
    # ------------------------------------------------------------------

    def check_com(self) -> None:
        """Start a background COM check (read Ua)."""
        if not self.is_connected:
            return
        if self._check_com_worker and self._check_com_worker.isRunning():
            return
        if self._check_com_worker:
            self._check_com_worker.cleanup()
            self._check_com_worker = None
        self._check_com_worker = CheckComWorker(self._client)
        self._check_com_worker.finished.connect(self._on_check_ok)
        self._check_com_worker.failed.connect(self._on_check_fail)
        self._check_com_worker.start()

    def _on_check_ok(self, value: int) -> None:
        self.check_ok.emit(value)

    def _on_check_fail(self, message: str) -> None:
        self.check_fail.emit(message)

    # ------------------------------------------------------------------
    # Poller control
    # ------------------------------------------------------------------

    def set_poller_active(self, active: bool) -> None:
        """Pause / resume live polling.

        With ``live_poll_during_test`` the pause request is ignored: the
        live panel keeps reading through scans and health tests. Frame
        integrity is not at stake — ``LM19Serial.get_param`` holds the
        port lock for the whole write+read transaction, and ``?Er;`` is a
        pure read that does not clear the firmware error flag. The cost is
        bus time: a poll cycle is ~140 ms at 9600 baud, so a measurement
        worker waits that much longer for the lock.
        """
        if not active and self._app_config.live_poll_during_test:
            return
        if self._param_poller:
            self._param_poller.set_active(active)

    def _start_poller(self) -> None:
        if self._param_poller:
            self._stop_poller()
            if self._param_poller is not None:
                # _stop_poller() could not drain it and deliberately retained
                # the reference. Move it aside BEFORE creating the new poller —
                # otherwise the assignment below would clobber the only
                # reference to a still-running, parentless QThread → GC abort.
                self._zombies.append(self._param_poller)
                self._param_poller = None
        self._param_poller = ParamPoller(
            self._client, interval_ms=self._app_config.live_poll_ms
        )
        self._param_poller.updated.connect(self._on_poller_data)
        self._param_poller.failed.connect(self.live_error.emit)
        self._param_poller.start()

    def _stop_poller(self) -> None:
        if self._param_poller:
            # cleanup() stops, waits, and disconnects stale signals. Drop the
            # reference only if it actually stopped — otherwise keep it so a
            # still-running QThread is not freed by GC (which aborts the
            # process).
            if self._param_poller.cleanup(timeout_ms=1000):
                self._param_poller = None

    def _on_poller_data(self, data: Dict) -> None:
        self.live_params.emit(data)
        self._emit_io_stats()

    # ------------------------------------------------------------------
    # IO stats
    # ------------------------------------------------------------------

    def update_io_stats(self) -> None:
        """Force an IO stats update (e.g. after manual send)."""
        self._emit_io_stats()

    def _emit_io_stats(self) -> None:
        if not self._client:
            return
        tx, rx, last_rx = self._client.stats()
        self._last_tx_count = tx
        self._last_rx_count = rx
        self.io_stats_updated.emit(tx, rx, last_rx)

    # ------------------------------------------------------------------
    # Trace
    # ------------------------------------------------------------------

    def apply_trace(self, enabled: bool, trace_path) -> None:
        """Enable/disable COM trace logging on the client."""
        if self._client:
            self._client.set_trace(enabled, trace_path)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop all workers, close port. Call from closeEvent."""
        for attr in ("_param_poller", "_check_com_worker"):
            w = getattr(self, attr)
            if w is None:
                continue
            # cleanup() returns False if the thread did not stop in time — then
            # signals are left connected and the reference must be retained
            # (a live QThread freed by GC aborts the process). Stash the zombie
            # instead of unconditionally nulling.
            if w.cleanup(timeout_ms=1500):
                setattr(self, attr, None)
            else:
                self._zombies.append(w)
                setattr(self, attr, None)
        if self._client and self._client.is_open():
            try:
                self._client.close()
            except (OSError, RuntimeError) as exc:
                # SerialException is an OSError subclass. A port that fails
                # to close at shutdown is a real deviation (stuck handle) —
                # WARNING, not debug (ML-149); programming errors propagate.
                log.warning("Error closing serial port during shutdown: %s",
                            exc)
        self._client = None
