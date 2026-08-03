"""SRK (S, R, K) measurement at zone corner points.

Reuses ``_set_param_calibrated`` and ``_read_measurement_point`` from
``lm19.scan.io`` so SRK shares the same feedforward/settle/verify
behaviour as the main scan (caller domain is physical, plan B).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

from lm19.analysis import Zone, compute_k, compute_sr_zone, estimate_srk_uncertainty
from lm19.calibration import CalibrationData
from lm19.protocol import LM19Serial, decode_ug1, encode_ug1
from lm19.protocol import decode_err
from lm19.scan.io import _read_measurement_point, _set_param_calibrated
from lm19.scan.protection import _check_heater
from lm19.scan.settings import (
    DEFAULT_SRK_SETTLE_BASE_S,
    DEFAULT_SRK_SETTLE_PER_VOLT_S,
    DEFAULT_SRK_UA_TOLERANCE,
    DEFAULT_SRK_UG2_TOLERANCE,
    DEFAULT_SRK_VERIFY_RETRIES,
)


class SrkVerifyError(Exception):
    """Raised when values fail to settle within tolerance."""
    pass


@dataclass
class SrkSettings:
    ua_min: float
    ua_max: float
    ug1_min: float
    ug1_max: float
    ug2: float
    samples: int = 5
    settle_s: float = 1.0
    calibration: CalibrationData = None  # type: ignore[assignment]
    # ML-136: ug1_settle_s removed — the field was never read by
    # measure_srk (Ug1 settle is driven by settle_per_volt_s/
    # settle_base_s from srk.json); app.json:ug1_settle_s only
    # affects the Ug1 reset (ResetWorker).
    ug1_verify_tolerance: float = 0.2
    verify_retries: int = DEFAULT_SRK_VERIFY_RETRIES
    ua_tolerance: float = DEFAULT_SRK_UA_TOLERANCE
    ug2_tolerance: float = DEFAULT_SRK_UG2_TOLERANCE
    settle_per_volt_s: float = DEFAULT_SRK_SETTLE_PER_VOLT_S
    settle_base_s: float = DEFAULT_SRK_SETTLE_BASE_S
    is_triode: bool = False
    ug2_track_ua: bool = False
    ug2_offset: float = 0.0
    ug1_step: float = 0.0
    # Expected heater setpoints for the loss check (0 = channel unused).
    # Same semantics as ScanSettings.uh/ih — _check_heater duck-types them.
    uh: float = 0.0
    ih: float = 0.0
    # ug1_step > 0 enables Ug1 sweep (mini-scan) for higher S/R/K precision.
    # Hardware limit: min reliable step = 0.04V (≥1 ADC LSB per step).
    # ATmega16 10-bit ADC, Ug1 charge pump comparator: ug1set = round(val*1024/Vref).
    # Steps < 0.04V may map to the same ug1set → same physical voltage.
    # Recommended: 0.04V (max points, good) or 0.08V (≥2 LSB, ironclad).


def _srk_ug1_values(settings: SrkSettings) -> List[float]:
    """Build the list of Ug1 values for SRK measurement.

    When ``ug1_step > 0``, generates values from *ug1_min* to *ug1_max*
    (inclusive) with the given step.  Otherwise falls back to the classic
    2-point list ``[ug1_min, ug1_max]``.

    Both *ug1_min* and *ug1_max* are negative (e.g. -4, -2), so stepping
    goes from more-negative to less-negative.

    Hardware context: each step must produce a distinct ``ug1set`` value
    in the firmware (10-bit ADC).  At 0.04V, the minimum ADC LSB diff is 1
    across the full 0-24V range.  At 0.08V, it's always ≥2 LSB.
    """
    if settings.ug1_step <= 0:
        return [settings.ug1_min, settings.ug1_max]

    step = settings.ug1_step
    vals: List[float] = []
    v = settings.ug1_min
    while v < settings.ug1_max - step * 0.01:  # tolerance for float rounding
        vals.append(round(v, 4))
        v += step
    # Always include ug1_max
    if not vals or abs(vals[-1] - settings.ug1_max) > step * 0.01:
        vals.append(round(settings.ug1_max, 4))
    return vals


def measure_srk(
    client: LM19Serial,
    settings: SrkSettings,
    progress: Optional[Callable[[int, int], None]] = None,
    stop: Optional[Callable[[], bool]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], List[Dict], Dict[str, Optional[float]]]:
    """Measure S, R, K parameters in the zone.

    When ``settings.ug1_step > 0``, Ug1 is swept from *ug1_min* to
    *ug1_max* with the given step (mini-scan), producing many more
    points and significantly higher precision for S (transconductance)
    via linear regression.

    When ``ug1_step == 0`` (default), the classic 4-corner-point mode
    is used: only ``ug1_min`` and ``ug1_max``.

    Uses the same ``_set_param_calibrated`` / ``_read_measurement_point``
    helpers as ``run_scan`` for consistent feedforward/settle/verify
    behaviour (caller domain is physical, plan B).  After each settle,
    the physical actual is checked against the SRK-specific tolerance
    and ``SrkVerifyError`` is raised on failure (stricter than scan's
    silent-continue policy).

    Returns (S, R, K, points).
    Points contain the full 7-field dict (ua, ug1, ug2, ia, ig2, uh, ih).
    """
    points: List[Dict] = []

    ua_values = [settings.ua_min, settings.ua_max]
    ug1_values = _srk_ug1_values(settings)

    total = len(ug1_values) * len(ua_values)
    done = 0

    # Track previous values for dynamic settle calculation.
    # Caller domain is physical (plan B): init from calibrated reads,
    # the adapter converts to the command domain internally.
    cal = settings.calibration

    def _phys(channel: str, value: float) -> float:
        return cal.apply_read(channel, value) if cal is not None else value

    prev_ua = _phys("ua", float(client.get_param("Ua", real=True)))
    prev_ug1 = _phys("ug1", decode_ug1(client.get_param("Ug1", real=True)))
    prev_ug2 = _phys("ug2", float(client.get_param("Ug2", real=True)))

    ug2_set = False
    actual_ug2 = prev_ug2

    # True triode: set Ug2=0 once, no verification needed
    if settings.is_triode:
        client.set_param("Ug2", 0)
        prev_ug2 = 0.0
        actual_ug2 = 0.0
        ug2_set = True

    for ug1 in ug1_values:
        if stop and stop():
            break

        # Track mode: drop Ug2 to the level matching the FIRST Ua before
        # touching Ug1/Ua for this curve (ML-129). Coming from the previous
        # curve the device sits at Ua=ua_max with Ug2=ua_max+offset; setting
        # Ua down first would leave the screen far above the anode for a
        # full settle (Ig2 spike). Mirrors _sweep_ug2_track's entry order.
        if settings.ug2_track_ua and not settings.is_triode:
            ug2_entry = max(0.0, ua_values[0] + settings.ug2_offset)
            actual_ug2 = _set_param_calibrated(
                client, "Ug2", "ug2", ug2_entry, prev_ug2, cal,
                settle_per_volt_s=0,
                settle_base_s=settings.settle_s,
                tolerance=settings.ug2_tolerance,
                max_retries=settings.verify_retries,
                stop=stop,
            )
            prev_ug2 = actual_ug2
            if abs(actual_ug2 - ug2_entry) > settings.ug2_tolerance:
                raise SrkVerifyError(
                    f"Ug2 failed to settle: target={ug2_entry:.1f}, "
                    f"actual={actual_ug2:.1f}")
            ug2_set = True

        # --- Set Ug1 with dynamic settle (SRK uses longer per-volt rate) ---
        actual_ug1 = _set_param_calibrated(
            client, "Ug1", "ug1", ug1, prev_ug1, cal,
            settle_per_volt_s=settings.settle_per_volt_s,
            settle_base_s=settings.settle_base_s,
            tolerance=settings.ug1_verify_tolerance,
            max_retries=settings.verify_retries,
            encode_fn=encode_ug1, decode_fn=decode_ug1,
            stop=stop,
        )
        prev_ug1 = actual_ug1
        if abs(actual_ug1 - ug1) > settings.ug1_verify_tolerance:
            raise SrkVerifyError(
                f"Ug1 failed to settle: target={ug1:.2f}, actual={actual_ug1:.2f}")

        for ua in ua_values:
            if stop and stop():
                break

            # Determine Ug2 target
            if settings.is_triode:
                target_ug2 = 0.0
            elif settings.ug2_track_ua:
                target_ug2 = max(0, ua + settings.ug2_offset)
            else:
                target_ug2 = settings.ug2

            # --- Set Ua (first — safer for screen grid in track mode) ---
            actual_ua = _set_param_calibrated(
                client, "Ua", "ua", ua, prev_ua, cal,
                settle_per_volt_s=0,
                settle_base_s=settings.settle_s,  # fixed settle per attempt
                tolerance=settings.ua_tolerance,
                max_retries=settings.verify_retries,
                stop=stop,
            )
            prev_ua = actual_ua
            if abs(actual_ua - ua) > settings.ua_tolerance:
                raise SrkVerifyError(
                    f"Ua failed to settle: target={ua:.1f}, actual={actual_ua:.1f}")

            # --- Set Ug2 (skip for triode; for fixed Ug2, set only once) ---
            if not settings.is_triode and (settings.ug2_track_ua or not ug2_set):
                actual_ug2 = _set_param_calibrated(
                    client, "Ug2", "ug2", target_ug2, prev_ug2, cal,
                    settle_per_volt_s=0,
                    settle_base_s=settings.settle_s,
                    tolerance=settings.ug2_tolerance,
                    max_retries=settings.verify_retries,
                    stop=stop,
                )
                prev_ug2 = actual_ug2
                if abs(actual_ug2 - target_ug2) > settings.ug2_tolerance:
                    raise SrkVerifyError(
                        f"Ug2 failed to settle: target={target_ug2:.1f}, "
                        f"actual={actual_ug2:.1f}")
                ug2_set = True

            # --- Verify Ug1 hasn't drifted during Ua/Ug2 settle ---
            actual_ug1_check = _phys(
                "ug1", decode_ug1(client.get_param("Ug1", real=True)))
            if abs(actual_ug1_check - ug1) > settings.ug1_verify_tolerance:
                raise SrkVerifyError(
                    f"Ug1 drifted: target={ug1:.2f}, actual={actual_ug1_check:.2f}")

            # --- Read full measurement (Ia+Ig2 averaged, all params) ---
            point = _read_measurement_point(
                client, settings.calibration, settings.samples)
            # ML-112: a protection trip or dying heater mid-SRK yields
            # garbage S/R/K — abort loudly (SrkVerifyError reaches the UI
            # via SrkWorker.failed), never average the corrupt corner in.
            er = int(point.get("er", 0) or 0)
            if er:
                abbrs = ", ".join(a for a, _ in decode_err(er))
                raise SrkVerifyError(
                    f"Hardware protection during SRK (Er=0x{er:02X}: "
                    f"{abbrs}) — aborting measurement")
            _check_heater(point, settings)  # raises HeaterLostError
            # Override with verified voltage values for accuracy
            point["ua"] = actual_ua
            point["ug1"] = actual_ug1_check
            point["ug2"] = actual_ug2

            points.append(point)
            done += 1
            if progress:
                progress(done, total)

    # Compute S, R, K
    zone = Zone(
        ua_min=settings.ua_min,
        ua_max=settings.ua_max,
        ug1_min=settings.ug1_min,
        ug1_max=settings.ug1_max,
        ug2=settings.ug2,
        is_triode=settings.is_triode,
        ug2_track_ua=settings.ug2_track_ua,
        ug2_offset=settings.ug2_offset,
    )
    s, r, _ = compute_sr_zone(points, zone)
    k = compute_k(s, r)

    delta_ua = (settings.ua_max - settings.ua_min) / 2.0
    delta_ug1 = (settings.ug1_max - settings.ug1_min) / 2.0
    uncertainty = estimate_srk_uncertainty(s, r, delta_ua, delta_ug1, settings.samples)

    return s, r, k, points, uncertainty
