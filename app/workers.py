import logging
import threading
from typing import List, Optional

from serial import SerialException

log = logging.getLogger(__name__)

from PySide6.QtCore import QMetaMethod, QThread, Signal

from lm19.protocol import (
    LM19Serial,
    decode_ih,
    decode_uh,
    encode_ih,
    encode_uh,
    encode_ug1,
)
from lm19.calibration import CalibrationData
from lm19.config import LampConfig
from lm19.scan import ScanSettings, run_scan
from lm19.scan.exceptions import HealthProtectionError
from lm19.health import run_health_test


class BaseWorker(QThread):
    """Base class for worker threads with common stop/fail pattern.

    ``client`` is optional: most workers talk to the LM19 over serial,
    but some (e.g. ``OptimizeWorker``) operate purely on in-memory data
    and don't need a hardware connection — they pass ``client=None``.
    """

    failed = Signal(str)

    def __init__(self, client: Optional[LM19Serial] = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            self._execute()
        except Exception as exc:
            log.exception("Worker %s failed", type(self).__name__)
            self.failed.emit(str(exc))

    def _execute(self) -> None:
        raise NotImplementedError

    def cleanup(self, timeout_ms: int = 1500) -> bool:
        """Stop worker, wait for completion, disconnect signals.

        Returns ``True`` if the thread is stopped (or was never running) and
        its signals were disconnected. Returns ``False`` if it did not stop
        within ``timeout_ms`` — in that case signals are LEFT connected and
        the caller MUST keep its reference: a still-running ``QThread`` freed
        by GC aborts the process. Safe to call even if not running.
        """
        if self.isRunning():
            self.stop()
            if not self.wait(timeout_ms):
                log.warning(
                    "Worker %s did not stop within %d ms — keeping signals "
                    "connected; caller must retain the reference",
                    type(self).__name__, timeout_ms,
                )
                return False
        # Disconnect signals defined on this class hierarchy (not QThread builtins).
        # ML-092: on PySide6 a no-receiver disconnect() does NOT raise — it
        # emits a RuntimeWarning (the old except branch was dead code and the
        # warnings spammed stderr, masking real ones). Check connectedness
        # via QMetaMethod first; nothing to catch afterwards.
        for cls in type(self).__mro__:
            if cls is QThread or cls is QThread.__base__:
                break
            for attr_name, attr in vars(cls).items():
                if isinstance(attr, Signal):
                    bound = getattr(self, attr_name)
                    meta = QMetaMethod.fromSignal(bound)
                    if self.isSignalConnected(meta):
                        bound.disconnect()
        return True


class ScanWorker(BaseWorker):
    progress = Signal(object)
    finished = Signal(object)
    comm_error = Signal(str, int)

    def __init__(self, client: LM19Serial, settings: ScanSettings) -> None:
        super().__init__(client)
        self.settings = settings
        self._comm_event = threading.Event()
        self._comm_response: Optional[str] = None

    def stop(self) -> None:
        self._stop_requested = True
        self._comm_event.set()

    def respond_comm_error(self, decision: str) -> None:
        """Called from UI thread to unblock the worker after comm_error."""
        self._comm_response = decision
        self._comm_event.set()

    def _handle_comm_error(self, message: str, attempt: int) -> str:
        """Called from worker thread — blocks until UI responds."""
        self._comm_event.clear()
        self._comm_response = None
        # ML-010: stop() sets _stop_requested BEFORE set(). If stop() landed
        # before the clear() above, its wakeup is erased and wait() would
        # block forever (zombie worker). Re-checking here closes every
        # interleaving: a stop() after this check finds the event cleared
        # and its set() wakes the wait below.
        if self._stop_requested:
            return "abort"
        self.comm_error.emit(message, attempt)
        self._comm_event.wait()
        if self._stop_requested:
            return "abort"
        return self._comm_response or "abort"

    def _execute(self) -> None:
        points = run_scan(
            self.client,
            self.settings,
            progress=self.progress.emit,
            stop=lambda: self._stop_requested,
            on_comm_error=self._handle_comm_error,
        )
        self.finished.emit(points)


class PreheatWorker(BaseWorker):
    progress = Signal(float, float, int)
    finished = Signal()

    def __init__(
        self,
        client: LM19Serial,
        target_uh: float,
        target_ih: float,
        warmup_s: int,
        calibration: Optional[CalibrationData] = None,
    ) -> None:
        super().__init__(client)
        self.target_uh = target_uh
        self.target_ih = target_ih
        self.warmup_s = warmup_s
        # Plan B (docs/CALIBRATION_PLAN.md §3.2): heater setpoints get SET
        # feedforward, progress readings are reported in the physical
        # domain. None → identity (legacy callers keep raw behaviour).
        self.calibration = calibration

    def _cal_set(self, channel: str, value: float) -> float:
        return (self.calibration.apply_set(channel, value)
                if self.calibration is not None else value)

    def _cal_read(self, channel: str, value: float) -> float:
        return (self.calibration.apply_read(channel, value)
                if self.calibration is not None else value)

    def _execute(self) -> None:
        use_uh = self.target_uh > 0.0
        target = self.target_uh if use_uh else self.target_ih
        if target <= 0.0:
            raise ValueError("Target heater value is zero")

        # Ramp starts from the current physical value so calibrated and
        # commanded domains agree on the step sizes.
        if use_uh:
            start_value = self._cal_read(
                "uh", decode_uh(self.client.get_param("Uh", real=True)))
        else:
            start_value = self._cal_read(
                "ih", decode_ih(self.client.get_param("Ih", real=True)))

        if start_value > target:
            # Pre-write stop gate: do not (re-)assert the heater if a stop was
            # flagged (e.g. an emergency zero just ran) — re-energizing Uh/Ih
            # after the synchronous zero is the residual the gate closes.
            if self._stop_requested:
                return
            self._set_heater(use_uh, target)
            self._emit_readings(0)
        else:
            ramp_s = max(3, min(12, int(self.warmup_s * 0.2)))
            steps = max(6, int(ramp_s / 0.5))
            for step in range(steps):
                if self._stop_requested:
                    return
                value = start_value + (step + 1) / steps * (target - start_value)
                # Pre-write stop gate (see ramp-down branch): never re-assert
                # the heater after a stop was flagged.
                if self._stop_requested:
                    return
                self._set_heater(use_uh, value)
                self._emit_readings(0)
                self.msleep(500)

        remaining = self.warmup_s
        while remaining > 0:
            if self._stop_requested:
                return
            self._emit_readings(remaining)
            self.msleep(1000)
            remaining -= 1

        self._emit_readings(0)
        self.finished.emit()

    def _set_heater(self, use_uh: bool, value: float) -> None:
        """Send heater setpoint (Uh or Ih) — working point, SET feedforward."""
        if use_uh:
            self.client.set_param("Uh", encode_uh(self._cal_set("uh", value)))
        else:
            self.client.set_param("Ih", encode_ih(self._cal_set("ih", value)))

    def _emit_readings(self, remaining: int) -> None:
        """Emit physical (calibrated) Uh/Ih readings for the UI."""
        if self.target_uh > 0:
            uh = self._cal_read(
                "uh", decode_uh(self.client.get_param("Uh", real=True)))
            ih = self._cal_read(
                "ih", decode_ih(self.client.get_param("Ih", real=True)))
            self.progress.emit(uh, ih, remaining)
        else:
            ih = self._cal_read(
                "ih", decode_ih(self.client.get_param("Ih", real=True)))
            self.progress.emit(0.0, ih, remaining)


class ParamPoller(BaseWorker):
    updated = Signal(object)

    def __init__(self, client: LM19Serial, interval_ms: int = 800) -> None:
        super().__init__(client)
        self.interval_ms = interval_ms
        self._active = True
        self._failing = False

    def set_active(self, active: bool) -> None:
        self._active = active

    def _execute(self) -> None:
        while not self._stop_requested:
            if not self._active or not self.client.is_open():
                self.msleep(self.interval_ms)
                continue
            try:
                data = {
                    "ua": self.client.get_param("Ua", real=True),
                    "ug1": self.client.get_param("Ug1", real=True),
                    "ug2": self.client.get_param("Ug2", real=True),
                    "uh": self.client.get_param("Uh", real=True),
                    "ih": self.client.get_param("Ih", real=True),
                    "ia": self.client.get_param("Ia", real=True),
                    "ig2": self.client.get_param("Ig2", real=True),
                    "an": self.client.get_param("An", real=False),
                    "er": self.client.get_param("Er", real=False),
                }
            except (SerialException, TimeoutError, OSError, ValueError,
                    RuntimeError) as exc:
                # ML-108: narrow except (programming errors propagate to
                # BaseWorker.run → log.exception + failed), log the comm
                # error, and emit failed once per failure STREAK — the old
                # code re-emitted every 800 ms, flashing the COM-error
                # status/LED for the whole outage.
                if not self._failing:
                    self._failing = True
                    log.warning("Live poll failed: %s", exc)
                    self.failed.emit(str(exc))
            else:
                if self._failing:
                    self._failing = False
                    log.info("Live poll recovered")
                self.updated.emit(data)
            self.msleep(self.interval_ms)


class CheckComWorker(BaseWorker):
    finished = Signal(int)

    def __init__(self, client: LM19Serial) -> None:
        super().__init__(client)

    def _execute(self) -> None:
        value = self.client.get_param("Ua", real=True)
        self.finished.emit(value)


class ResetWorker(BaseWorker):
    finished = Signal()

    def __init__(
        self,
        client: LM19Serial,
        ug1_value: float,
        ug1_settle_s: float = 0.3,
        reset_heater: bool = True,
        order: Optional[List[str]] = None,
    ) -> None:
        super().__init__(client)
        self.ug1_value = ug1_value
        self.ug1_settle_s = ug1_settle_s
        self.reset_heater = reset_heater
        self.order = order or ["Ug2", "Ug1", "Ua", "Uh", "Ih"]

    def _execute(self) -> None:
        failed_params: List[str] = []
        for param in self.order:
            try:
                if param == "Ug2":
                    self.client.set_param("Ug2", 0)
                elif param == "Ug1":
                    self.client.set_param("Ug1", encode_ug1(self.ug1_value))
                    # ML-049: honor the configured settle — the Ug1 charge
                    # pump is slow; let the cutoff bias establish before
                    # the remaining outputs are sequenced. Plain sleep:
                    # this is a safety path that must run to completion.
                    if self.ug1_settle_s > 0:
                        self.msleep(int(self.ug1_settle_s * 1000))
                elif param == "Ua":
                    self.client.set_param("Ua", 0)
                elif param == "Uh":
                    if self.reset_heater:
                        self.client.set_param("Uh", 0)
                elif param == "Ih":
                    if self.reset_heater:
                        self.client.set_param("Ih", 0)
            except (SerialException, RuntimeError, OSError) as exc:
                # Best-effort safety reset: never abort on the first broken
                # write — keep zeroing the remaining outputs (ML-124; the
                # closed-port preflight raises RuntimeError, which the old
                # SerialException-only catch let kill the loop with Ua/Uh
                # still un-reset). Programming errors propagate to
                # BaseWorker.run (failure-visibility principle 1).
                log.warning("Reset of %s failed: %s", param, exc)
                failed_params.append(param)
        if failed_params:
            # Failure visibility (ML-125): a reset that did not actually
            # zero every output must NOT report success — HV may still sit
            # on the tube while the UI shows a completed reset.
            self.failed.emit("reset incomplete, outputs not zeroed: "
                             + ", ".join(failed_params))
            return
        self.finished.emit()


class HealthWorker(BaseWorker):
    progress = Signal(object)
    finished = Signal(object)
    # Emitted instead of ``failed`` when Pa/Pg2 safety limit trips during
    # the OP-approach ramp. Payload is a ``HealthProtectionPayload``
    # dataclass with all diagnostics for the UI dialog.
    protection_triggered = Signal(object)

    def __init__(
        self,
        client: LM19Serial,
        lamp: LampConfig,
        app_config,
        calibration,
        lamp_id: str,
        name: str,
        reference_mode: str,
        reference: Optional[dict],
        emission_enabled: bool,
        measurement_plan: Optional[dict],
        warmup_s: int,
    ) -> None:
        super().__init__(client)
        self.lamp = lamp
        self.app_config = app_config
        self.calibration = calibration
        self.lamp_id = lamp_id
        self.name = name
        self.reference_mode = reference_mode
        self.reference = reference
        self.emission_enabled = emission_enabled
        self.measurement_plan = measurement_plan
        self.warmup_s = warmup_s

    def _execute(self) -> None:
        try:
            measurement = run_health_test(
                client=self.client,
                lamp=self.lamp,
                cfg=self.app_config,
                calibration=self.calibration,
                lamp_id=self.lamp_id,
                name=self.name,
                reference_mode=self.reference_mode,
                reference=self.reference,
                emission_enabled=self.emission_enabled,
                measurement_plan=self.measurement_plan,
                warmup_s=self.warmup_s,
                progress=self.progress.emit,
                stop=lambda: self._stop_requested,
            )
        except HealthProtectionError as exc:
            log.warning("Health protection tripped: %s", exc)
            self.protection_triggered.emit(exc.payload)
            return
        self.finished.emit(measurement)

    def run(self) -> None:
        """Suppress ``failed`` on a user stop. ``run_health_test`` raises
        ``RuntimeError('Health test stopped')`` when stopped mid-SRK (so no
        partial result is emitted as complete); that is not a user-visible
        error, so do not surface it as one."""
        try:
            self._execute()
        except Exception as exc:
            log.exception("HealthWorker failed")
            if not self._stop_requested:
                self.failed.emit(str(exc))


class AmpVerifyWorker(BaseWorker):
    """Run the LTspice amplifier verification off the UI thread.

    Pure-compute worker (``client=None``): fits the model, drives LTspice
    batch runs and emits the parsed ``VerifyResult``. ``stop()`` is polled
    between runs and while LTspice executes — a cancelled job still emits
    its partial result (with a ``"cancelled"`` warning) so completed runs
    stay visible.
    """

    progress = Signal(str)
    finished_result = Signal(object)  # VerifyResult

    def __init__(self, request, workdir: str,
                 ltspice_exe: Optional[str] = None, parent=None) -> None:
        super().__init__(client=None, parent=parent)
        self._request = request
        self._workdir = workdir
        self._ltspice_exe = ltspice_exe

    def _execute(self) -> None:
        from lm19.ltspice_verify import run_verification

        result = run_verification(
            self._request,
            workdir=self._workdir,
            ltspice_exe=self._ltspice_exe,
            stop=lambda: self._stop_requested,
            progress=self.progress.emit,
        )
        self.finished_result.emit(result)
