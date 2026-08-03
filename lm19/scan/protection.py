"""Heater alive checks + Pa/Pg2/Ig2 limit predicates + recovery primitives.

These guard scan loops from over-power, lost-heater, and hardware errors.
``_wait_for_err_clear`` and ``_restore_heater_and_wait`` are used by
``run_scan`` after a hardware-protection event.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Optional, TYPE_CHECKING

log = logging.getLogger(__name__)

from lm19.constants import MW_PER_W as _MW_PER_W
from lm19.protocol import LM19Serial, decode_ih, decode_uh, encode_ih, encode_uh
from lm19.scan.events import ScanProgress
from lm19.scan.exceptions import HeaterLostError

if TYPE_CHECKING:
    from lm19.calibration import CalibrationData
    from lm19.scan.settings import ScanSettings


# ── Heater-loss thresholds ────────────────────────────────────────────
_UH_LOST_THRESHOLD = 0.5   # V — below this, heater is considered dead
_IH_LOST_THRESHOLD = 0.02  # A — below this, heater is considered dead

# ── Hardware error recovery constants ────────────────────────────────
_ERR_POLL_INTERVAL_S = 0.3
_ERR_POLL_TIMEOUT_S = 15.0
_HEATER_RESTORE_TIMEOUT_S = 30.0
_HEATER_TOLERANCE_V = 0.3
_HEATER_TOLERANCE_A = 0.02
_HEATER_POLL_INTERVAL_S = 0.5


def _wait_for_err_clear(
    client: LM19Serial,
    timeout: float = _ERR_POLL_TIMEOUT_S,
    stop: Optional[Callable[[], bool]] = None,
) -> int:
    """Poll Er until firmware clears it (user resets on device).

    Args:
        client: serial connection.
        timeout: maximum seconds to wait before giving up.
        stop: optional callable returning True when caller wants to abort.
            Checked between polls so the worker can exit promptly when
            the user clicks Cancel; without it the function would block
            for the full ``timeout`` (~15 s) ignoring the request.

    Returns:
        Last Er value. 0 = cleared successfully; non-zero = either
        timeout reached or stop callback fired (caller should re-check
        ``stop()`` to distinguish).
    """
    start = time.monotonic()
    er = client.get_param("Er", real=False)
    while er != 0 and time.monotonic() - start < timeout:
        if stop and stop():
            log.info("Er-clear poll cancelled by user (Er=0x%02X)", er)
            return er
        time.sleep(_ERR_POLL_INTERVAL_S)
        er = client.get_param("Er", real=False)
    if er != 0:
        log.warning("Er did not clear within %.0fs", timeout)
    return er


def _restore_heater_and_wait(
    client: LM19Serial,
    settings: "ScanSettings",
    progress: Optional[Callable[[ScanProgress], None]],
    stop: Optional[Callable[[], bool]],
    calibration: Optional["CalibrationData"] = None,
) -> None:
    """Re-send heater setpoints and wait for actual value to reach target.

    Hardware has a single heater control channel: either voltage (Uh) or
    current (Ih) regulation, never both simultaneously.  The checks below
    return as soon as the active channel settles.

    Plan B (docs/CALIBRATION_PLAN.md §2): the heater restore is a
    working-point command → SET feedforward applies; the wait-loop
    comparison runs in the physical domain (``apply_read``), otherwise
    a READ offset larger than the tolerance would spin the loop to the
    30 s timeout even though the heater is physically on target.
    *calibration* overrides ``settings.calibration`` for non-scan
    callers; both ``None`` (or default coefficients) → exact raw
    behaviour.
    """
    cal = calibration if calibration is not None else settings.calibration

    def _set_cmd(channel: str, target: float) -> float:
        return cal.apply_set(channel, target) if cal is not None else target

    def _read_phys(channel: str, decoded: float) -> float:
        return cal.apply_read(channel, decoded) if cal is not None else decoded

    if settings.uh > 0:
        client.set_param("Uh", encode_uh(_set_cmd("uh", settings.uh)))
    if settings.ih > 0:
        client.set_param("Ih", encode_ih(_set_cmd("ih", settings.ih)))

    if settings.uh <= 0 and settings.ih <= 0:
        return  # no heater configured — nothing to wait for

    start = time.monotonic()
    while time.monotonic() - start < _HEATER_RESTORE_TIMEOUT_S:
        if stop and stop():
            return
        uh_real = _read_phys("uh", decode_uh(client.get_param("Uh", real=True)))
        ih_real = _read_phys("ih", decode_ih(client.get_param("Ih", real=True)))
        if progress:
            progress({"event": "heater_restoring",
                       "uh": uh_real, "ih": ih_real})
        if settings.uh > 0 and abs(uh_real - settings.uh) <= _HEATER_TOLERANCE_V:
            return
        if settings.ih > 0 and abs(ih_real - settings.ih) <= _HEATER_TOLERANCE_A:
            return
        time.sleep(_HEATER_POLL_INTERVAL_S)
    log.warning("Heater did not reach target within %.0fs",
                _HEATER_RESTORE_TIMEOUT_S)


def _check_heater(point: Dict, settings: "ScanSettings") -> None:
    """Raise HeaterLostError if heater readings indicate loss of filament.

    Only checks channels that are configured (uh > 0 or ih > 0).
    """
    if settings.uh > 0 and point["uh"] < _UH_LOST_THRESHOLD:
        raise HeaterLostError(
            f"Heater voltage lost: Uh={point['uh']:.2f} V "
            f"(expected ~{settings.uh:.1f} V)"
        )
    if settings.ih > 0 and point["ih"] < _IH_LOST_THRESHOLD:
        raise HeaterLostError(
            f"Heater current lost: Ih={point['ih']:.3f} A "
            f"(expected ~{settings.ih:.2f} A)"
        )


# ---------------------------------------------------------------------------
# Limit-check helpers (eliminate repetition in sweep loops)
# ---------------------------------------------------------------------------

def _exceeds_pa(point: Dict, pa_limit: float) -> bool:
    """True if anode power exceeds pa_limit (W). Returns False if limit is 0."""
    if pa_limit <= 0:
        return False
    return point["ua"] * point["ia"] / _MW_PER_W > pa_limit


def _exceeds_pg2(point: Dict, pg2_limit: float, ug2_nominal: float = 0.0) -> bool:
    """True if screen-grid power exceeds pg2_limit (W)."""
    if pg2_limit <= 0 or ug2_nominal <= 0:
        return False
    return point["ug2"] * point["ig2"] / _MW_PER_W > pg2_limit


def _exceeds_ig2(point: Dict, ig2_limit: float) -> bool:
    """True if screen-grid current exceeds ig2_limit (mA)."""
    if ig2_limit <= 0:
        return False
    return point["ig2"] > ig2_limit
