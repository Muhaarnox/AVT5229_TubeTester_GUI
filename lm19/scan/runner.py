"""Top-level ``run_scan`` orchestrator.

Coordinates settle wrappers, comm-error retry, hardware-protection recovery,
and the three sweep modes (triode / Ug2-track / independent-Ug2). Emits
structured progress events for UI.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

from serial import SerialException

log = logging.getLogger(__name__)

from lm19.protocol import (
    LM19Serial,
    decode_err,
    decode_ia,
    decode_ig2,
    decode_ih,
    decode_ug1,
    decode_uh,
    encode_ih,
    encode_uh,
)
from lm19.constants import TOPOLOGY_TRIODE
from lm19.scan.events import ScanProgress
from lm19.scan.exceptions import (
    HeaterLostError,
    ProtectionError,
    _BreakSweep,
    _SkipPoint,
)
from lm19.scan.io import (
    _interruptible_sleep,
    _read_measurement_point,
    _set_param_calibrated,
    _try_reopen,
)
from lm19.scan.protection import (
    _check_heater,
    _restore_heater_and_wait,
    _wait_for_err_clear,
)
from lm19.scan.settings import (
    COMM_AUTO_RETRY_DELAY_S,
    COMM_USER_RETRY_DELAY_S,
    ScanSettings,
    _frange,
    scan_point_count,
)
from lm19.scan.sweepers import (
    _SweepCtx,
    _sweep_triode,
    _sweep_ug2_independent,
    _sweep_ug2_track,
)


def run_scan(
    client: LM19Serial,
    settings: ScanSettings,
    progress: Optional[Callable[[ScanProgress], None]] = None,
    stop: Optional[Callable[[], bool]] = None,
    on_comm_error: Optional[Callable[[str, int], str]] = None,
) -> List[Dict]:
    """Run a full IV scan.

    *on_comm_error(message, attempt) -> "retry"|"skip"|"abort"*
    Called when silent auto-retries are exhausted.  Return value controls
    the scan flow: ``"retry"`` re-attempts the point, ``"skip"`` skips it,
    ``"abort"`` re-raises the exception (existing partial-save logic applies).
    """
    scan_start_t = time.monotonic()
    points: List[Dict] = []
    max_retries = max(settings.comm_retries, 0)
    # ML-108/109: settle/outlier degradation counters (filled by io helpers)
    # — reported in the scan_summary event so the UI dialog can show them.
    io_stats: Dict[str, int] = {}

    def _read_point() -> Optional[Dict]:
        """Wrapper around _read_measurement_point with comm-error retry,
        hardware-error check, and heater-alive check."""
        nonlocal heater_lost_msg, prev_ua, prev_ug1, prev_ug2
        attempt = 0
        while True:
            try:
                point = _read_measurement_point(
                    client, settings.calibration, settings.ia_samples,
                    ia_outlier_ratio=settings.ia_outlier_ratio,
                    ia_outlier_reread_samples=(
                        settings.ia_outlier_reread_samples),
                    stats=io_stats)
            except (ValueError, RuntimeError, SerialException) as exc:
                attempt += 1
                if attempt <= max_retries and not isinstance(exc, SerialException):
                    log.warning("Comm error reading point (auto-retry %d/%d): %s",
                                attempt, max_retries, exc)
                    client.flush_input()
                    if _interruptible_sleep(COMM_AUTO_RETRY_DELAY_S, stop):
                        return None
                    continue
                if on_comm_error:
                    decision = on_comm_error(str(exc), attempt)
                    if decision == "retry":
                        _try_reopen(client)
                        if _interruptible_sleep(COMM_USER_RETRY_DELAY_S, stop):
                            return None
                        attempt = 0
                        continue
                    if decision == "skip":
                        log.info("User skipped point after comm error: %s", exc)
                        return None
                raise
            else:
                # Hardware error check
                er = point.get("er", 0)
                if er:
                    errors = decode_err(er)
                    abbrs = ", ".join(a for a, _ in errors)
                    log.warning("Hardware error Er=0x%02X (%s)", er, abbrs)
                    if progress:
                        progress({"event": "hw_protection", "er": er,
                                  "errors": [(a, k) for a, k in errors]})
                    msg = f"Hardware protection: {abbrs}"
                    while True:
                        if not on_comm_error:
                            raise RuntimeError(msg)
                        if stop and stop():
                            raise RuntimeError(msg + " (cancelled)")
                        decision = on_comm_error(msg, 0)
                        if decision in ("abort", "skip"):
                            raise RuntimeError(msg)
                        # Pass stop so this poll exits promptly on cancel
                        # instead of blocking up to 15 s.
                        er_now = _wait_for_err_clear(client, stop=stop)
                        if stop and stop():
                            raise RuntimeError(msg + " (cancelled)")
                        if er_now == 0:
                            break
                        # Er not cleared — show dialog again
                    _restore_heater_and_wait(client, settings, progress, stop)
                    prev_ua = 0.0
                    prev_ug1 = 0.0
                    prev_ug2 = 0.0
                    if progress:
                        progress({"event": "hw_protection_cleared"})
                    raise _BreakSweep()

                if heater_lost_msg is None:
                    try:
                        _check_heater(point, settings)
                    except HeaterLostError as exc:
                        heater_lost_msg = str(exc)
                        log.warning("Heater lost: %s", exc)
                        return None
                return point

    # An selector stays raw (discrete channel). Initial heater setpoints
    # are working-point commands → SET feedforward (plan B);
    # settings.calibration may be None (bare test settings) → raw.
    client.set_param("An", settings.an)
    if settings.uh > 0:
        uh_cmd = (settings.calibration.apply_set("uh", settings.uh)
                  if settings.calibration is not None else settings.uh)
        client.set_param("Uh", encode_uh(uh_cmd))
    if settings.ih > 0:
        ih_cmd = (settings.calibration.apply_set("ih", settings.ih)
                  if settings.calibration is not None else settings.ih)
        client.set_param("Ih", encode_ih(ih_cmd))

    ua_values = _frange(settings.ua.start, settings.ua.stop, settings.ua.step)
    ug1_values = _frange(settings.ug1.start, settings.ug1.stop, settings.ug1.step)

    # Pa exceedance limit: pa_max_w * (1 + pa_over_pct / 100)
    pa_limit = 0.0
    if settings.pa_max_w > 0:
        pa_limit = settings.pa_max_w * (1.0 + settings.pa_over_pct / 100.0)

    # Pg2 exceedance limit: pig2_max_w * (1 + pig2_over_pct / 100)
    pg2_limit = 0.0
    if settings.pig2_max_w > 0:
        pg2_limit = settings.pig2_max_w * (1.0 + settings.pig2_over_pct / 100.0)

    # Log scan configuration at start
    if settings.is_triode:
        mode = TOPOLOGY_TRIODE
    elif settings.ug2_track_ua:
        mode = f"ug2_track (offset={settings.ug2_offset:+.0f})"
    else:
        mode = "pentode (independent Ug2)"
    log.info("Scan start: mode=%s, Ua=%.0f..%.0f/%.0f, Ug1=%.1f..%.1f/%.1f, "
             "Ug2=%.0f..%.0f/%.0f, Uh=%.1f, Ih=%.2f, An=%d, "
             "Pa_limit=%.2fW, Pg2_limit=%.2fW, Ig2_limit=%.2fmA, "
             "refine=%s, points=%d",
             mode,
             settings.ua.start, settings.ua.stop, settings.ua.step,
             settings.ug1.start, settings.ug1.stop, settings.ug1.step,
             settings.ug2.start, settings.ug2.stop, settings.ug2.step,
             settings.uh, settings.ih, settings.an,
             pa_limit, pg2_limit, settings.ig2_max_ma,
             settings.refine_enabled,
             scan_point_count(settings))

    # Track previous values for dynamic settle calculation.
    # Caller domain is physical (plan B): init from calibrated reads,
    # the adapter converts to the command domain internally.
    cal = settings.calibration

    def _phys(channel: str, value: float) -> float:
        return cal.apply_read(channel, value) if cal is not None else value

    prev_ua = _phys("ua", float(client.get_param("Ua", real=True)))
    prev_ug1 = _phys("ug1", decode_ug1(client.get_param("Ug1", real=True)))
    prev_ug2 = _phys("ug2", float(client.get_param("Ug2", real=True)))

    def _settle_ua(target_ua: float) -> float:
        nonlocal prev_ua
        actual = _set_param_calibrated(
            client, "Ua", "ua", target_ua, prev_ua, cal,
            settle_per_volt_s=settings.ua_settle_per_volt_s,
            settle_base_s=settings.ua_settle_base_s,
            tolerance=settings.ua_tolerance,
            max_retries=settings.ua_retries,
            stop=stop, stats=io_stats,
        )
        prev_ua = actual
        return actual

    def _settle_ug1(target_ug1: float) -> float:
        nonlocal prev_ug1
        from lm19.protocol import encode_ug1  # local import to avoid cycle in this scope
        actual = _set_param_calibrated(
            client, "Ug1", "ug1", target_ug1, prev_ug1, cal,
            settle_per_volt_s=settings.ug1_settle_per_volt_s,
            settle_base_s=settings.ug1_settle_base_s,
            tolerance=settings.ug1_tolerance,
            max_retries=settings.ug1_retries,
            encode_fn=encode_ug1, decode_fn=decode_ug1,
            stop=stop, stats=io_stats,
        )
        prev_ug1 = actual
        return actual

    def _settle_ug2(target_ug2: float) -> float:
        nonlocal prev_ug2
        actual = _set_param_calibrated(
            client, "Ug2", "ug2", target_ug2, prev_ug2, cal,
            settle_per_volt_s=settings.ug2_settle_per_volt_s,
            settle_base_s=settings.ug2_settle_base_s,
            tolerance=settings.ug2_tolerance,
            max_retries=settings.ug2_retries,
            stop=stop, stats=io_stats,
        )
        prev_ug2 = actual
        return actual

    def _emit_protection_event(param_label: str) -> None:
        """Read current real values and emit a 'protection' progress event."""
        if not progress:
            return
        try:
            cal = settings.calibration
            evt = {
                "event": "protection",
                "param": param_label,
                "ua": cal.apply_read("ua", float(client.get_param("Ua", real=True))),
                "ug1": cal.apply_read("ug1", decode_ug1(client.get_param("Ug1", real=True))),
                "ug2": cal.apply_read("ug2", float(client.get_param("Ug2", real=True))),
                "ia": cal.apply_read("ia", decode_ia(client.get_param("Ia", real=True))),
                "ig2": cal.apply_read("ig2", decode_ig2(client.get_param("Ig2", real=True))),
                "uh": cal.apply_read("uh", decode_uh(client.get_param("Uh", real=True))),
                "ih": cal.apply_read("ih", decode_ih(client.get_param("Ih", real=True))),
            }
            progress(evt)
        except (ValueError, RuntimeError, SerialException) as exc:
            # Best-effort telemetry: a transient read/decode failure must not
            # abort the scan. Narrow catch + WARNING (principle 1) so a
            # programming-error regression (AttributeError/TypeError) surfaces
            # instead of being swallowed at the default INFO log level.
            log.warning("Failed to emit protection event: %s", exc)

    def _wrap_settle(fn, label):
        """Add comm-error retry (auto + user dialog) to a settle function."""
        def wrapped(target):
            attempt = 0
            while True:
                try:
                    return fn(target)
                except ProtectionError as exc:
                    # Protection is persistent — auto-retries are pointless,
                    # go straight to user dialog.
                    log.warning("Protection detected in %s: %s", label, exc)
                    _emit_protection_event(label)
                    if on_comm_error:
                        decision = on_comm_error(str(exc), attempt)
                        if decision == "retry":
                            client.flush_input()
                            if _interruptible_sleep(COMM_USER_RETRY_DELAY_S, stop):
                                raise _SkipPoint()
                            attempt = 0
                            continue
                        if decision == "skip":
                            log.info("User skipped %s: %s", label, exc)
                            raise _SkipPoint() from exc
                    raise
                except (ValueError, RuntimeError, SerialException) as exc:
                    attempt += 1
                    if attempt <= max_retries and not isinstance(exc, SerialException):
                        log.warning("Comm error in %s (auto-retry %d/%d): %s",
                                    label, attempt, max_retries, exc)
                        client.flush_input()
                        if _interruptible_sleep(COMM_AUTO_RETRY_DELAY_S, stop):
                            raise _SkipPoint()
                        continue
                    if on_comm_error:
                        decision = on_comm_error(str(exc), attempt)
                        if decision == "retry":
                            _try_reopen(client)
                            if _interruptible_sleep(COMM_USER_RETRY_DELAY_S, stop):
                                raise _SkipPoint()
                            attempt = 0
                            continue
                        if decision == "skip":
                            log.info("User skipped %s: %s", label, exc)
                            raise _SkipPoint() from exc
                    raise
        return wrapped

    _settle_ua = _wrap_settle(_settle_ua, "Ua settle")
    _settle_ug1 = _wrap_settle(_settle_ug1, "Ug1 settle")
    _settle_ug2 = _wrap_settle(_settle_ug2, "Ug2 settle")

    # Ig2 hardware current limit (hard, no tolerance)
    ig2_limit = settings.ig2_max_ma

    heater_lost_msg: Optional[str] = None

    def _stopped() -> bool:
        return heater_lost_msg is not None or (stop is not None and stop())

    curves_summary: List[Dict] = []
    ctx = _SweepCtx(
        settle_ua=_settle_ua, settle_ug1=_settle_ug1,
        settle_ug2=_settle_ug2, read_point=_read_point,
        stopped=_stopped, progress=progress, stop=stop,
        settings=settings, pa_limit=pa_limit, pg2_limit=pg2_limit,
        ig2_limit=ig2_limit, ua_values=ua_values, ug1_values=ug1_values,
        curves_summary=curves_summary,
    )

    if settings.is_triode:
        client.set_param("Ug2", 0)
        prev_ug2 = 0.0
        points = _sweep_triode(ctx)
    elif settings.ug2_track_ua:
        points = _sweep_ug2_track(ctx)
    else:
        points = _sweep_ug2_independent(ctx, prev_ua)

    if heater_lost_msg is not None and progress:
        progress({"event": "heater_lost", "message": heater_lost_msg})

    # Sort for consistent ordering (refine points were appended inline)
    if settings.refine_enabled and points:
        points.sort(key=lambda p: (round(p["ug2"], 1), round(p["ug1"], 1),
                                   p["ua"]))

    duration_s = time.monotonic() - scan_start_t

    # Log final summary
    status_counts: Dict[str, int] = {}
    for c in curves_summary:
        status_counts[c["status"]] = status_counts.get(c["status"], 0) + 1
    status_str = ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
    dur_str = f"{int(duration_s // 60)}m{int(duration_s % 60):02d}s"
    log.info("Scan end: %d points in %s, %d curves (%s)%s",
             len(points), dur_str, len(curves_summary),
             status_str or "none",
             f", heater lost: {heater_lost_msg}" if heater_lost_msg else "")

    # Emit scan summary for UI dialog (per-curve outcomes + duration).
    if progress:
        progress({
            "event": "scan_summary",
            "duration_s": duration_s,
            "total_points": len(points),
            "curves": curves_summary,
            "heater_lost": heater_lost_msg,
            "settle_out_of_tolerance": io_stats.get(
                "settle_out_of_tolerance", 0),
            "ia_outlier_rereads": io_stats.get("ia_outlier_rereads", 0),
            "ia_unstable_points": io_stats.get("ia_unstable_points", 0),
        })

    return points
