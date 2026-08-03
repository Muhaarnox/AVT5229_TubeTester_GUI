from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, replace as dataclass_replace
from typing import Callable, Dict, List, Optional, Tuple, Protocol

import serial

from .analysis import (
    _linear_regression, compute_mu_g1g2, compute_sg2_direct,
    compute_srk_direct, estimate_sg2_uncertainty, estimate_srk_uncertainty,
)
from .calibration import CalibrationData, SET_LIMITS
from .config import LampConfig
from .config import DEFAULT_LIMITS
from .protocol import (
    LM19Serial, decode_ia, decode_ig2, decode_ih, decode_ug1, decode_uh,
    decode_err, encode_uh, encode_ih, encode_ug1,
    UA_RESOLUTION_V, UG1_RESOLUTION_V, UG2_RESOLUTION_V,
    IA_NOISE_200MA, DEFAULT_HEALTH_DELTA_PCT,
    DELTA_UA_MIN_V, DELTA_UA_MAX_V,
    DELTA_UG1_MIN_V, DELTA_UG1_MAX_V,
    DELTA_UG2_MIN_V, DELTA_UG2_MAX_V,
)
from .scan import (
    _read_measurement_point, _set_param_calibrated, _set_param_with_settle,
)
from .scan.exceptions import (
    HealthProtectionError, HealthProtectionPayload,
)
from .scan.protection import _exceeds_pa, _exceeds_pg2
from .health_events import HealthProgress
from .constants import (
    EPS,
    EPS_COARSE,
    MW_PER_W,
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)

log = logging.getLogger(__name__)


class HealthConfig(Protocol):
    """Config attributes consumed by lm19.health (ML-025).

    Structural typing: tests pass lightweight stand-ins. The CI pin
    in tests/test_conventions_guards.py keeps this in sync both ways
    (every direct ``cfg.<attr>`` use ⊆ Protocol ⊆ AppConfig) — the
    typo-catcher the old bare ``cfg`` parameter lacked. Optional
    attrs read via ``getattr(cfg, ..., default)`` stay out.
    """

    health_delta_ug2_pct: int
    health_bias_servo_max_iter: int
    health_bias_servo_max_shift_v: float
    health_bias_servo_pa_ceiling_pct: float
    health_bias_servo_shift_margin: float
    health_bias_servo_step_v: float
    health_bias_servo_tol_ma: float
    health_bias_servo_ug1_floor_v: float
    health_emission_enabled_default: bool
    health_emission_knee_drop_pct: float
    health_emission_min_ik_ratio: float
    health_emission_mode_default: str
    health_emission_ratio_good_min: float
    health_emission_ratio_nominal: float
    health_emission_ratio_weak_min: float
    health_emission_sweep_max_total_s: float
    health_emission_uh_sweep_abs_min_ratio: float
    health_emission_uh_sweep_min_ratio: float
    health_emission_uh_sweep_steps: int
    health_emission_sample_period_s: float
    health_emission_stable_max_s: int
    health_emission_stable_min_s: int
    health_emission_stable_slope_threshold_ma_per_s: float
    health_emission_stable_warmup_ratio: float
    health_emission_stable_window_points: int
    health_emission_uh_ratio: float
    health_ia_samples: int
    health_op_ramp_enabled: bool
    health_op_ug1_ramp_step_v: float
    health_pa_safety_pct: float
    health_pig2_safety_pct: float
    health_renormalize_weights_if_metric_missing: bool
    health_ua_retries: int
    health_ua_settle_base_s: float
    health_ua_settle_per_volt_s: float
    health_ua_tolerance_v: float
    health_ug1_retries: int
    health_ug1_settle_base_s: float
    health_ug1_settle_per_volt_s: float
    health_ug1_tolerance_v: float
    health_ug2_retries: int
    health_ug2_settle_base_s: float
    health_ug2_settle_per_volt_s: float
    health_ug2_tolerance_v: float
    health_verdict_good_min: float
    health_verdict_strong_min: float
    health_verdict_weak_min: float
    health_weight_emission: float
    health_weight_ia: float
    health_weight_rh: float
    health_weight_s: float
    health_weight_screen: float
    ug1_after_stop: float




# ── module local constants ──

# Empirical ETA heuristic for the Uh80 stabilization phase
# (``_run_stabilized_ia80``): when the Ia slope is above the stability
# threshold by some fraction f = (|slope| - slope_thr) / |slope|, the
# remaining time-to-stable is estimated as ``f × _UH80_ETA_MAX_S``
# seconds. f saturates at 1 (slope ≫ thr), so this constant sets the
# **upper bound** of any single ETA estimate; the caller still clamps
# to ``t_max_s - elapsed``. 10 s matches the typical observed
# stabilization tail across pentodes/triodes with cfg-default sample
# period of 0.5 s and window of 6 points.
_UH80_ETA_MAX_S = 10.0

# Heater settle/verify parameters for ``_set_uh_with_verify``. The heater
# is thermally slow compared to the electrode supplies, hence the larger
# base settle; tolerance matches the 0.1 V Uh resolution with headroom.
_UH_SETTLE_PER_VOLT_S = 0.1
_UH_SETTLE_BASE_S = 0.5
_UH_TOLERANCE_V = 0.2
_UH_MAX_RETRIES = 2
# Current-heater (Ih-driven) lamps verify against an amps tolerance — the 0.2 V
# Uh tolerance would be absurdly loose on a ~0.3 A heater current.
_IH_TOLERANCE_A = 0.05


# ── emission verdict vocabulary ──────────────────────────────────────
# Standalone cathode-reserve scale, deliberately independent of the
# weighted index: Ia and S carry 0.75 of the index weight, so a tube
# whose cathode reserve is already gone can still score "Good" overall.
# The reserve verdict is the one signal that says "the cathode itself
# is on the way out", so it must not be averaged away.
EMISSION_VERDICT_NORMAL = "normal"
EMISSION_VERDICT_WEAKENED = "weakened"
EMISSION_VERDICT_EXHAUSTED = "exhausted"
EMISSION_VERDICT_NA = "na"
# wear scale from "Emission Labs TB-08" and the Emission Labs bulletin
# "About Lifetime of our tubes", see SOURCES_INDEX.md
EMISSION_VERDICTS = frozenset({
    EMISSION_VERDICT_NORMAL,
    EMISSION_VERDICT_WEAKENED,
    EMISSION_VERDICT_EXHAUSTED,
    EMISSION_VERDICT_NA,
})

# ── composite verdict vocabulary ─────────────────────────────────────
# Contract codes, not display text: they are persisted in every saved
# measurement and compared as strings by the history filter, so they must
# survive a locale change. Translation happens at the UI edge.
HEALTH_VERDICT_STRONG = "Strong"
HEALTH_VERDICT_GOOD = "Good"
HEALTH_VERDICT_WEAK = "Weak"
HEALTH_VERDICT_REPLACE = "Replace"
HEALTH_VERDICT_NA = "N/A"
HEALTH_VERDICTS = frozenset({
    HEALTH_VERDICT_STRONG,
    HEALTH_VERDICT_GOOD,
    HEALTH_VERDICT_WEAK,
    HEALTH_VERDICT_REPLACE,
    HEALTH_VERDICT_NA,
})
# Ordered worst-to-best for anything that needs to rank them.
HEALTH_VERDICT_ORDER = (
    HEALTH_VERDICT_REPLACE,
    HEALTH_VERDICT_WEAK,
    HEALTH_VERDICT_GOOD,
    HEALTH_VERDICT_STRONG,
)

# ── bias-servo status vocabulary ─────────────────────────────────────
# Transconductance is a function of anode current, not of grid bias, so
# S measured at a fixed Ug1 on a tube whose Ia has drifted is compared
# against a reference taken at a different point of the same curve. The
# servo removes that confound by driving Ia to the reference current
# first. Every non-ok status must reach the user: an S silently measured
# at the wrong current looks exactly like a healthy S.
BIAS_SERVO_OK = "ok"
BIAS_SERVO_DISABLED = "disabled"
BIAS_SERVO_NO_REFERENCE = "no_reference"
BIAS_SERVO_UNREACHABLE = "unreachable"
BIAS_SERVO_STATUSES = frozenset({
    BIAS_SERVO_OK,
    BIAS_SERVO_DISABLED,
    BIAS_SERVO_NO_REFERENCE,
    BIAS_SERVO_UNREACHABLE,
})

# ── measurement-point step tags (bias servo) ────────────────────────
# Contract strings crossing the module boundary (written here, read by
# the steps table and tests). Three tags so the table can tell the
# ACCEPTED operating point from intermediate probes and from the
# restore-to-plan probe — with a single shared tag the measuring
# point cannot be identified in the saved point list.
STEP_BIAS_SERVO = "bias_servo"            # intermediate walk/bisect probe
STEP_BIAS_SERVO_OP = "bias_servo_op"      # the accepted measuring point
STEP_BIAS_SERVO_RESTORE = "bias_servo_restore"  # back-to-plan after failure
BIAS_SERVO_STEPS = frozenset({
    STEP_BIAS_SERVO, STEP_BIAS_SERVO_OP, STEP_BIAS_SERVO_RESTORE,
})

# ── emission mode vocabulary ─────────────────────────────────────────
# "single" probes the Ia(Uh) curve at one reduced heater point; "sweep"
# walks it and locates the knee where emission stops being space-charge
# limited. The knee voltage is what migrates upward as the cathode's
# emitting material is consumed, so it is the wear signal proper.
EMISSION_MODE_SINGLE = "single"
EMISSION_MODE_SWEEP = "sweep"
EMISSION_MODES = frozenset({EMISSION_MODE_SINGLE, EMISSION_MODE_SWEEP})


# ── Pure helpers ─────────────────────────────────────────────────────

def _safe_pct(value: Optional[float], ref: Optional[float]) -> Optional[float]:
    if value is None or ref is None or abs(ref) < EPS:
        return None
    return 100.0 * value / ref


def _clamp_score(v: Optional[float], max_pct: float = 130.0) -> Optional[float]:
    if v is None:
        return None
    return max(0.0, min(max_pct, v))


def _weighted_index(metrics: Dict[str, Optional[float]], weights: Dict[str, float], renormalize: bool) -> Optional[float]:
    pairs = [(k, metrics.get(k), float(weights.get(k, 0.0))) for k in weights]
    valid = [(v, w) for _, v, w in pairs if v is not None and w > 0]
    if not valid:
        return None
    if renormalize:
        sum_w = sum(w for _, w in valid) or 1.0
        return sum(v * (w / sum_w) for v, w in valid)
    return sum(v * w for v, w in valid)


def _verdict(index: Optional[float], cfg: HealthConfig) -> str:
    if index is None:
        return HEALTH_VERDICT_NA
    if index >= cfg.health_verdict_strong_min:
        return HEALTH_VERDICT_STRONG
    if index >= cfg.health_verdict_good_min:
        return HEALTH_VERDICT_GOOD
    if index >= cfg.health_verdict_weak_min:
        return HEALTH_VERDICT_WEAK
    return HEALTH_VERDICT_REPLACE


def _emission_verdict(emission_ratio: Optional[float], cfg: HealthConfig) -> str:
    """Absolute cathode-reserve scale for ``EmissionRatio = Ia80/Ia100``.

    Absolute on purpose: unlike ``emission_score`` it is not divided by a
    reference ratio, so it still answers "is this cathode near depletion"
    when the reference itself came from an already-worn tube.
    """
    if emission_ratio is None:
        return EMISSION_VERDICT_NA
    if emission_ratio >= cfg.health_emission_ratio_good_min:
        return EMISSION_VERDICT_NORMAL
    if emission_ratio >= cfg.health_emission_ratio_weak_min:
        return EMISSION_VERDICT_WEAKENED
    return EMISSION_VERDICT_EXHAUSTED


def _slope(x: List[float], y: List[float]) -> float:
    result = _linear_regression(x, y)
    return result if result is not None else 0.0


def _ug2_for_ua(lamp: LampConfig, ua: float,
                ug2_track: bool, ug2_offset: float, ug2_op: float) -> float:
    if lamp.is_triode:
        return 0.0
    if ug2_track:
        return max(0.0, ua + ug2_offset)
    return ug2_op


def _ug2_mode_for(lamp: LampConfig, targets: "_HealthTargets") -> str:
    """Scan-mode vocabulary for the lamp under the planned targets.

    One definition for both consumers: the stored ``conditions`` and the
    OP setup used to derive them must agree, and a matching pool keyed on
    ``ug2_mode`` empties silently when they drift apart.
    """
    if lamp.is_triode:
        return TOPOLOGY_TRIODE
    return (TOPOLOGY_TRIODE_CONNECTED if targets.ug2_track_ua
            else TOPOLOGY_PENTODE)


def _extract_refs(ref: Optional[Dict], lamp: LampConfig) -> Dict:
    ref = ref or {}
    rv = ref.get("reference", {}) if isinstance(ref.get("reference"), dict) else ref
    rh = rv.get("rh")
    if rh is None:
        rh = (lamp.uh / lamp.ih) if lamp.ih > 0 else None
    screen = rv.get("screen_ratio")
    if screen is None and lamp.ia > 0:
        screen = lamp.ig2 / lamp.ia
    emission_ratio = rv.get("emission_ratio")
    return {
        "ia": float(rv.get("ia", lamp.ia)) if rv.get("ia") is not None else None,
        "s": float(rv.get("s", lamp.s)) if rv.get("s") is not None else None,
        "r": float(rv.get("r", lamp.r)) if rv.get("r") is not None else None,
        "k": float(rv.get("k", lamp.k)) if rv.get("k") is not None else None,
        "rh": rh,
        "screen_ratio": screen,
        # A per-tube reserve baseline beats the global nominal: the
        # expected Ia80/Ia100 depends on how deep the operating point
        # sits in space-charge limitation, which is per tube type.
        "emission_ratio": float(emission_ratio) if isinstance(emission_ratio, (int, float)) else None,
    }


def _clamp_delta(raw_delta: float, min_v: float, max_v: float,
                 resolution: float) -> float:
    """Clamp a delta value to [min_v, max_v], rounded to resolution."""
    clamped = max(min_v, min(max_v, raw_delta))
    return round(clamped / resolution) * resolution


def clamp_delta_ua(raw_delta: float,
                   min_v: float = DELTA_UA_MIN_V,
                   max_v: float = DELTA_UA_MAX_V,
                   resolution: float = UA_RESOLUTION_V) -> float:
    return _clamp_delta(raw_delta, min_v, max_v, resolution)


def clamp_delta_ug1(raw_delta: float,
                    min_v: float = DELTA_UG1_MIN_V,
                    max_v: float = DELTA_UG1_MAX_V,
                    resolution: float = UG1_RESOLUTION_V) -> float:
    return _clamp_delta(raw_delta, min_v, max_v, resolution)


def clamp_delta_ug2(raw_delta: float,
                    min_v: float = DELTA_UG2_MIN_V,
                    max_v: float = DELTA_UG2_MAX_V,
                    resolution: float = UG2_RESOLUTION_V) -> float:
    return _clamp_delta(raw_delta, min_v, max_v, resolution)


def _compute_shifted_center(
    target: float, delta: float, limit: float,
) -> Tuple[float, str]:
    """Return (center, method) ensuring center ± delta fits within [0, limit].

    Returns ("central") when no shift is needed, ("shifted_op") otherwise.
    """
    if target + delta <= limit and target - delta >= 0:
        return target, "central"
    center = min(target, limit - delta)
    center = max(center, delta)
    return round(center), "shifted_op"


def compute_shifted_r_center(
    target_ua: float, delta_ua: float,
    ua_max: float = DEFAULT_LIMITS["ua_max"],
) -> Tuple[float, str]:
    """Return (r_phase_center, method) for the R-phase Ua sweep."""
    return _compute_shifted_center(target_ua, delta_ua, ua_max)


def compute_shifted_sg2_center(
    target_ug2: float, delta_ug2: float,
    ug2_max: float = DEFAULT_LIMITS["ug2_max"],
) -> Tuple[float, str]:
    """Return (sg2_phase_center, method) for the Sg2-phase Ug2 sweep."""
    return _compute_shifted_center(target_ug2, delta_ug2, ug2_max)


def _build_ug1_sweep(target_ug1: float, delta_ug1: float, n_ug1: int) -> List[float]:
    # Defense-in-depth (the UI also validates with abs(ug1) <= delta_ug1): a
    # sweep top at/above 0 V is the onset of grid conduction → grid current +
    # garbage S-measurement. Fail loud rather than silently produce such a sweep
    # (failure-visibility). `>= 0` matches the UI boundary exactly.
    if target_ug1 + delta_ug1 >= 0.0:
        raise ValueError(
            f"Ug1 sweep reaches/crosses zero: target_ug1={target_ug1} + "
            f"delta_ug1={delta_ug1} >= 0 (positive grid)")
    if n_ug1 <= 3:
        return [target_ug1 - delta_ug1, target_ug1 + delta_ug1]
    step_v = 2.0 * delta_ug1 / (n_ug1 - 1)
    result = []
    for i in range(n_ug1):
        v = target_ug1 - delta_ug1 + i * step_v
        if abs(v - target_ug1) > EPS_COARSE:
            result.append(round(v, 4))
    return result


def _ia_sample_delay_s(cfg: HealthConfig) -> float:
    """Sample delay between Ia readings (seconds)."""
    return getattr(cfg, "health_ia_sample_delay_ms", 50) / 1000.0


def _raise_on_hw_error(er_raw: int) -> None:
    """Raise RuntimeError if hardware error bitmask is non-zero."""
    if not er_raw:
        return
    errors = decode_err(er_raw)
    if errors:
        abbrs = ", ".join(abbr for abbr, _ in errors)
        raise RuntimeError(f"Hardware protection triggered: {abbrs}")
    raise RuntimeError(f"Hardware protection triggered: Er=0x{er_raw:02X}")


def _check_pa_pg2_at_step(
    point: Dict,
    lamp: LampConfig,
    cfg: HealthConfig,
    *,
    step_idx: int,
    total_steps: int,
    start_ug1: float,
    target_ug1: float,
    lamp_id: str,
    ug2_mode: str,
) -> None:
    """Raise ``HealthProtectionError`` if Pa or Pg2 exceed the safety limit.

    Pa is checked when ``lamp.pa_max`` is set. Pg2 is checked when
    ``lamp.pig2_max`` is set AND Ug2 > 0 (i.e. not a true triode). Both
    limits are scaled by their respective ``cfg.health_*_safety_pct``
    (e.g. 120 means trip at 120% of the datasheet maximum).

    DELIBERATELY called only on the OP ramp, NOT in the S/R/Sg2 sweep
    phases (ML-126, accepted by design): the phase excursions
    around the OP are small (S·δUg1, ±δUa) and last seconds — far inside
    the thermal time constants behind the continuous Pa/Pg2 ratings — while
    a soft check there would add false-trip aborts on a single noisy point.
    The hardware Er register remains the per-point backstop in every phase.
    """
    pa_safety_pct = float(cfg.health_pa_safety_pct)
    pig2_safety_pct = float(cfg.health_pig2_safety_pct)
    pa_limit = (lamp.pa_max or 0.0) * pa_safety_pct / 100.0
    pg2_limit = (lamp.pig2_max or 0.0) * pig2_safety_pct / 100.0

    if lamp.pa_max is not None and _exceeds_pa(point, pa_limit):
        pa_w = point["ua"] * point["ia"] / MW_PER_W
        payload = HealthProtectionPayload(
            kind="pa",
            ua=float(point["ua"]),
            ug1=float(point["ug1"]),
            ug2=float(point.get("ug2", 0.0)),
            ia_ma=float(point["ia"]),
            ig2_ma=float(point.get("ig2", 0.0)),
            measured_w=pa_w,
            limit_w=pa_limit,
            datasheet_max_w=lamp.pa_max,
            safety_pct=pa_safety_pct,
            step_idx=step_idx,
            total_steps=total_steps,
            start_ug1=start_ug1,
            target_ug1=target_ug1,
            tube_type=lamp.tube_type,
            lamp_id=lamp_id,
            topology=lamp.topology,
            ug2_mode=ug2_mode,
        )
        raise HealthProtectionError(
            f"Pa protection: Pa={pa_w:.2f} W > limit={pa_limit:.2f} W "
            f"(Ua={point['ua']:.1f} V, Ia={point['ia']:.2f} mA) "
            f"at ramp step {step_idx}/{total_steps}",
            payload,
        )

    if lamp.pig2_max is not None and _exceeds_pg2(point, pg2_limit,
                                                    ug2_nominal=point.get("ug2", 0.0)):
        pg2_w = point["ug2"] * point["ig2"] / MW_PER_W
        payload = HealthProtectionPayload(
            kind="pg2",
            ua=float(point["ua"]),
            ug1=float(point["ug1"]),
            ug2=float(point.get("ug2", 0.0)),
            ia_ma=float(point["ia"]),
            ig2_ma=float(point.get("ig2", 0.0)),
            measured_w=pg2_w,
            limit_w=pg2_limit,
            datasheet_max_w=lamp.pig2_max,
            safety_pct=pig2_safety_pct,
            step_idx=step_idx,
            total_steps=total_steps,
            start_ug1=start_ug1,
            target_ug1=target_ug1,
            tube_type=lamp.tube_type,
            lamp_id=lamp_id,
            topology=lamp.topology,
            ug2_mode=ug2_mode,
        )
        raise HealthProtectionError(
            f"Pg2 protection: Pg2={pg2_w:.2f} W > limit={pg2_limit:.2f} W "
            f"(Ug2={point['ug2']:.1f} V, Ig2={point['ig2']:.2f} mA) "
            f"at ramp step {step_idx}/{total_steps}",
            payload,
        )


def _restore_ug1_safe_lock(
    client: LM19Serial,
    cfg: HealthConfig,
    calibration: CalibrationData,
    hw: "_HwState",
    protection_exc: Optional[BaseException] = None,
) -> None:
    """Drive Ug1 back to the safe lock after a protection trip.

    Safe-lock restore stays RAW (plan B raw-zero rule): apply_set could
    shift the bias toward 0 V = maximum anode current, the dangerous
    direction. Only the verify target and bookkeeping are domain
    converted — without read_inverse a READ gain error >= tolerance
    would burn every retry here.

    Shared by every phase that drives Ug1 under protection, so a new
    phase cannot ship a subtly different restore.
    """
    start_ug1 = float(cfg.ug1_after_stop)
    try:
        raw_actual = _set_param_with_settle(
            client, "Ug1", start_ug1, hw.prev_ug1,
            cfg.health_ug1_settle_per_volt_s, cfg.health_ug1_settle_base_s,
            cfg.health_ug1_tolerance_v, cfg.health_ug1_retries,
            encode_fn=encode_ug1, decode_fn=decode_ug1,
            verify_target=calibration.read_inverse("ug1", start_ug1),
        )
        hw.actual_ug1 = calibration.apply_read("ug1", raw_actual)
        hw.prev_ug1 = hw.actual_ug1
    except Exception as restore_exc:
        log.warning("Failed to restore Ug1 to safe lock after protection trip: %s",
                    restore_exc)
        # The tube may NOT be cut off — flag it so the protection
        # dialog warns the operator (failure-visibility rule).
        payload = getattr(protection_exc, "payload", None)
        if payload is not None:
            payload.ug1_restore_failed = True


def _ramp_ug1_to_op(
    client: LM19Serial,
    cfg: HealthConfig,
    calibration: CalibrationData,
    lamp: LampConfig,
    hw: "_HwState",
    target_ug1: float,
    lamp_id: str,
    ug2_mode: str,
    progress: Optional[Callable[[HealthProgress], None]],
    stop: Optional[Callable[[], bool]],
) -> Dict:
    """Step Ug1 from safe lock to target with Pa/Pg2 protection per step.

    Pre-condition: caller has already set Ug1 to ``cfg.ug1_after_stop``
    (safe lock), Ua to target, and Ug2 to target. The tube is therefore
    biased fully closed before any anode/screen voltage was applied.

    Steps Ug1 toward ``target_ug1`` by ``cfg.health_op_ug1_ramp_step_v``
    (default 1 V). After each step, takes a single-shot measurement and
    invokes ``_check_pa_pg2_at_step``. On trip, the function restores
    Ug1 to the safe-lock value before re-raising, so the tube is left
    closed.

    Returns the last single-shot point at target Ug1. ``_setup_op``
    discards this and takes a fresh averaged read for the OP statistics;
    the return value is mostly for tests/diagnostics.
    """
    start_ug1 = float(cfg.ug1_after_stop)
    step_v = max(0.1, float(cfg.health_op_ug1_ramp_step_v))

    # Total steps = number of intermediate Ug1 values from start to target.
    distance = abs(target_ug1 - start_ug1)
    total_steps = max(1, int(distance / step_v))
    if total_steps * step_v < distance - EPS_COARSE:
        total_steps += 1
    direction = 1.0 if target_ug1 > start_ug1 else -1.0

    try:
        for i in range(1, total_steps + 1):
            if stop and stop():
                raise RuntimeError("Health test stopped")
            # Last step always lands exactly on target.
            if i == total_steps:
                step_ug1 = target_ug1
            else:
                step_ug1 = start_ug1 + direction * step_v * i
            hw.actual_ug1 = _set_param_calibrated(
                client, "Ug1", "ug1", step_ug1, hw.prev_ug1, calibration,
                settle_per_volt_s=cfg.health_ug1_settle_per_volt_s,
                settle_base_s=cfg.health_ug1_settle_base_s,
                tolerance=cfg.health_ug1_tolerance_v,
                max_retries=cfg.health_ug1_retries,
                encode_fn=encode_ug1, decode_fn=decode_ug1,
                stop=stop,
            )
            hw.prev_ug1 = hw.actual_ug1
            pt = _read_measurement_point(client, calibration, 1, 0.0)
            _raise_on_hw_error(pt.get("er", 0))
            _check_pa_pg2_at_step(
                pt, lamp, cfg,
                step_idx=i, total_steps=total_steps,
                start_ug1=start_ug1, target_ug1=target_ug1,
                lamp_id=lamp_id, ug2_mode=ug2_mode,
            )
            if progress:
                progress({
                    "event": "op_ramp",
                    "step_idx": i,
                    "total_steps": total_steps,
                    "ug1": float(pt["ug1"]),
                    "target_ug1": float(target_ug1),
                    "start_ug1": start_ug1,
                    "ua": float(pt["ua"]),
                    "ug2": float(pt.get("ug2", 0.0)),
                    # Heater travels with every ramp step: a live display fed
                    # a heater-less event cannot tell "not reported" from
                    # "heater off" and renders a false 0 V / 0 A.
                    "uh": float(pt.get("uh", 0.0)),
                    "ih": float(pt.get("ih", 0.0)),
                    "ia_ma": float(pt["ia"]),
                    "ig2_ma": float(pt.get("ig2", 0.0)) if not lamp.is_triode else None,
                    "pa_w": float(pt["ua"]) * float(pt["ia"]) / MW_PER_W,
                    "pg2_w": (float(pt.get("ug2", 0.0)) * float(pt.get("ig2", 0.0))
                              / MW_PER_W) if not lamp.is_triode else None,
                })
        return pt
    except HealthProtectionError as protection_exc:
        _restore_ug1_safe_lock(client, cfg, calibration, hw, protection_exc)
        raise


# ── Mutable hardware state ───────────────────────────────────────────

@dataclass
class _HwState:
    prev_ua: float
    prev_ug1: float
    prev_ug2: float
    prev_uh: float
    actual_ua: float = 0.0
    actual_ug1: float = 0.0
    actual_ug2: float = 0.0


# ── Hardware interaction helpers ─────────────────────────────────────

def _set_heater_with_verify(client: LM19Serial, use_uh: bool, target: float,
                            prev: float, calibration: CalibrationData) -> float:
    """Set the heater on the channel the lamp actually uses — Uh (voltage) when
    *use_uh* else Ih (current) — via the feedforward adapter; returns physical
    actual. Lamps with a current heater (PCC84/85: uh=0, ih>0) would otherwise
    be driven by Uh=0, switching the heater OFF mid-test."""
    name, channel = ("Uh", "uh") if use_uh else ("Ih", "ih")
    enc, dec = (encode_uh, decode_uh) if use_uh else (encode_ih, decode_ih)
    tolerance = _UH_TOLERANCE_V if use_uh else _IH_TOLERANCE_A
    return _set_param_calibrated(
        client, name, channel, target, prev, calibration,
        settle_per_volt_s=_UH_SETTLE_PER_VOLT_S,
        settle_base_s=_UH_SETTLE_BASE_S,
        tolerance=tolerance,
        max_retries=_UH_MAX_RETRIES,
        encode_fn=enc,
        decode_fn=dec,
    )


def _set_uh_with_verify(client: LM19Serial, target_uh: float, prev_uh: float,
                        calibration: CalibrationData) -> float:
    """Set heater voltage via the feedforward adapter; returns physical Uh."""
    return _set_heater_with_verify(client, True, target_uh, prev_uh, calibration)


def _set_an_with_readback(client: LM19Serial, requested_an: int) -> Dict:
    """Set anode selector and return confirmed value from device."""
    client.set_param("An", requested_an)
    try:
        confirmed_raw = client.get_param("An", real=True)
        confirmed_an = int(confirmed_raw)
        return {
            "confirmed": True,
            "requested_an": int(requested_an),
            "actual_an": confirmed_an,
        }
    except (ValueError, serial.SerialException, OSError) as exc:
        log.warning("An readback failed, keeping last confirmed LivePanel value: %s", exc)
        return {
            "confirmed": False,
            "requested_an": int(requested_an),
            "actual_an": None,
            "error": str(exc),
        }


def _run_stabilized_ia80(
    client: LM19Serial,
    cfg: HealthConfig,
    calibration: CalibrationData,
    warmup_s: int,
    progress: Optional[Callable[[HealthProgress], None]] = None,
    stop: Optional[Callable[[], bool]] = None,
) -> Dict:
    ratio = float(cfg.health_emission_stable_warmup_ratio)
    min_s = int(cfg.health_emission_stable_min_s)
    max_s = int(cfg.health_emission_stable_max_s)
    t_max_s = max(min_s, min(max_s, int(ratio * max(0, warmup_s))))
    sample_period_s = max(0.5, float(cfg.health_emission_sample_period_s))
    window = max(3, int(cfg.health_emission_stable_window_points))
    slope_thr = abs(float(cfg.health_emission_stable_slope_threshold_ma_per_s))

    started = time.time()
    t_hist: List[float] = []
    ia_hist: List[float] = []
    eta_s_last: Optional[int] = None

    while True:
        if stop and stop():
            raise RuntimeError("Health test stopped")
        now = time.time()
        elapsed = now - started
        # All displayed readings go through READ calibration — the progress
        # event feeds the UI directly, raw decodes would lie to the user.
        ia_ma = calibration.apply_read("ia", decode_ia(client.get_param("Ia", real=True)))
        ig2_ma = calibration.apply_read("ig2", decode_ig2(client.get_param("Ig2", real=True)))
        uh_val = calibration.apply_read("uh", decode_uh(client.get_param("Uh", real=True)))
        ih_val = calibration.apply_read("ih", decode_ih(client.get_param("Ih", real=True)))
        # ML-041: the UI shows Pa = Ua·Ia during stabilization — carry the
        # measured Ua in the event instead of making the UI parse its own
        # Ua label text back into a number.
        ua_val = calibration.apply_read("ua", float(client.get_param("Ua", real=True)))
        _raise_on_hw_error(client.get_param("Er", real=False))
        t_hist.append(elapsed)
        ia_hist.append(ia_ma)

        stable = False
        slope_val = 0.0
        if len(ia_hist) >= window:
            x = t_hist[-window:]
            y = ia_hist[-window:]
            slope_val = _slope(x, y)
            stable = abs(slope_val) <= slope_thr
            if not stable:
                denom = max(abs(slope_val), EPS_COARSE)
                eta_s_last = int(max(0.0, min(
                    t_max_s - elapsed,
                    (abs(slope_val) - slope_thr) / denom * _UH80_ETA_MAX_S,
                )))
            else:
                eta_s_last = 0

        if progress:
            progress({
                "event": "uh80_stabilizing",
                "ua": ua_val,
                "elapsed_s": int(elapsed),
                "t_max_s": int(t_max_s),
                "eta_s": eta_s_last if eta_s_last is not None else max(0, int(t_max_s - elapsed)),
                "ia_ma": ia_ma,
                "ig2_ma": ig2_ma,
                "uh": uh_val,
                "ih": ih_val,
                "slope_ma_per_s": slope_val,
                "stable": stable,
            })

        if stable:
            return {
                "ia80": ia_ma,
                "stable": True,
                "elapsed_s": int(elapsed),
                "t_max_s": int(t_max_s),
                "eta_s_last": eta_s_last,
            }
        if elapsed >= t_max_s:
            return {
                "ia80": ia_ma,
                "stable": False,
                "elapsed_s": int(elapsed),
                "t_max_s": int(t_max_s),
                "eta_s_last": eta_s_last,
            }
        time.sleep(sample_period_s)


# ── Health test phases ───────────────────────────────────────────────

def _setup_op(
    client: LM19Serial, cfg: HealthConfig, calibration: CalibrationData, hw: _HwState,
    lamp: LampConfig,
    target_ua: float, target_ug1: float, target_ug2: float,
    ug2_mode: str, lamp_id: str,
    progress: Optional[Callable[[HealthProgress], None]],
    stop: Optional[Callable[[], bool]],
) -> Dict:
    """Drive the tube to OP via a safe-locked Ug1 ramp.

    Sequence:
      1. Ug1 → safe lock (``cfg.ug1_after_stop``, typically −24 V) so the
         tube is fully closed before any anode/screen voltage is applied.
      2. Ua → target_ua (tube closed → Ia ≈ 0, no Pa concern).
      3. Ug2 → target_ug2 (skipped for true triodes; tube still closed).
      4. Ramp Ug1: safe_lock → target_ug1 in ``cfg.health_op_ug1_ramp_step_v``
         steps (default 1 V), with Pa/Pg2 protection after each step.
      5. Final precise OP read on ``cfg.health_ia_samples`` samples.

    Step 4 is what protects the tube: a dud / shorted / over-emissive
    tube trips before reaching the planned bias instead of running at it.

    When ``cfg.health_op_ramp_enabled`` is False, falls back to the legacy
    direct set order (Ug1 → Ua → Ug2). This is purely for emergency
    debugging — leave it enabled in production.
    """
    if progress:
        progress({"event": "step", "step": "op"})

    if not cfg.health_op_ramp_enabled:
        return _setup_op_legacy(
            client, cfg, calibration, hw,
            target_ua, target_ug1, target_ug2, progress,
        )

    # 1. Lock Ug1 to safe-stop value (lamp fully closed). The command is
    # RAW (plan B raw-zero rule): apply_set could shift the safe-lock bias
    # toward 0 V = maximum anode current. The verify target and the
    # bookkeeping are domain-converted — without read_inverse a READ
    # gain error >= tolerance would burn every retry here.
    safe_lock = float(cfg.ug1_after_stop)
    raw_lock_actual = _set_param_with_settle(
        client, "Ug1", safe_lock, hw.prev_ug1,
        cfg.health_ug1_settle_per_volt_s, cfg.health_ug1_settle_base_s,
        cfg.health_ug1_tolerance_v, cfg.health_ug1_retries,
        encode_fn=encode_ug1, decode_fn=decode_ug1,
        verify_target=calibration.read_inverse("ug1", safe_lock),
    )
    hw.actual_ug1 = calibration.apply_read("ug1", raw_lock_actual)
    hw.prev_ug1 = hw.actual_ug1

    # 2. Ua → target (tube closed, safe)
    hw.actual_ua = _set_param_calibrated(
        client, "Ua", "ua", target_ua, hw.prev_ua, calibration,
        settle_per_volt_s=cfg.health_ua_settle_per_volt_s,
        settle_base_s=cfg.health_ua_settle_base_s,
        tolerance=cfg.health_ua_tolerance_v,
        max_retries=cfg.health_ua_retries,
        stop=stop,
    )
    if abs(hw.actual_ua - target_ua) > cfg.health_ua_tolerance_v:
        raise RuntimeError(f"Ua verify failed: target={target_ua:.1f}, actual={hw.actual_ua:.1f}")
    hw.prev_ua = hw.actual_ua

    # 3. Ug2 → target (tube closed, safe; skip for true triodes)
    if not lamp.is_triode:
        hw.actual_ug2 = _set_param_calibrated(
            client, "Ug2", "ug2", target_ug2, hw.prev_ug2, calibration,
            settle_per_volt_s=cfg.health_ug2_settle_per_volt_s,
            settle_base_s=cfg.health_ug2_settle_base_s,
            tolerance=cfg.health_ug2_tolerance_v,
            max_retries=cfg.health_ug2_retries,
            stop=stop,
        )
        if abs(hw.actual_ug2 - target_ug2) > cfg.health_ug2_tolerance_v:
            raise RuntimeError(f"Ug2 verify failed: target={target_ug2:.1f}, actual={hw.actual_ug2:.1f}")
        hw.prev_ug2 = hw.actual_ug2

    # 4. Ramp Ug1 to target with per-step Pa/Pg2 protection
    _ramp_ug1_to_op(
        client, cfg, calibration, lamp, hw,
        target_ug1=target_ug1, lamp_id=lamp_id, ug2_mode=ug2_mode,
        progress=progress, stop=stop,
    )
    if abs(hw.actual_ug1 - target_ug1) > cfg.health_ug1_tolerance_v:
        raise RuntimeError(
            f"Ug1 verify failed after ramp: target={target_ug1:.2f}, "
            f"actual={hw.actual_ug1:.2f}"
        )

    # 5. Final OP read with full averaging
    _delay = _ia_sample_delay_s(cfg)
    point = _read_measurement_point(client, calibration, cfg.health_ia_samples, _delay)
    _raise_on_hw_error(point.get("er", 0))
    if progress:
        progress({"event": "live_point", "point": dict(point, step="op")})
    return point


def _setup_op_legacy(
    client: LM19Serial, cfg: HealthConfig, calibration: CalibrationData, hw: _HwState,
    target_ua: float, target_ug1: float, target_ug2: float,
    progress: Optional[Callable[[HealthProgress], None]],
) -> Dict:
    """Legacy direct-set OP sequence (Ug1 → Ua → Ug2, no ramp).

    Kept as an emergency fallback toggled via ``health_op_ramp_enabled``.
    Used by ``_setup_op`` when ramp is disabled.
    """
    hw.actual_ug1 = _set_param_calibrated(
        client, "Ug1", "ug1", target_ug1, hw.prev_ug1, calibration,
        settle_per_volt_s=cfg.health_ug1_settle_per_volt_s,
        settle_base_s=cfg.health_ug1_settle_base_s,
        tolerance=cfg.health_ug1_tolerance_v,
        max_retries=cfg.health_ug1_retries,
        encode_fn=encode_ug1, decode_fn=decode_ug1,
    )
    if abs(hw.actual_ug1 - target_ug1) > cfg.health_ug1_tolerance_v:
        raise RuntimeError(f"Ug1 verify failed: target={target_ug1:.2f}, actual={hw.actual_ug1:.2f}")
    hw.prev_ug1 = hw.actual_ug1

    hw.actual_ua = _set_param_calibrated(
        client, "Ua", "ua", target_ua, hw.prev_ua, calibration,
        settle_per_volt_s=cfg.health_ua_settle_per_volt_s,
        settle_base_s=cfg.health_ua_settle_base_s,
        tolerance=cfg.health_ua_tolerance_v,
        max_retries=cfg.health_ua_retries,
    )
    if abs(hw.actual_ua - target_ua) > cfg.health_ua_tolerance_v:
        raise RuntimeError(f"Ua verify failed: target={target_ua:.1f}, actual={hw.actual_ua:.1f}")
    hw.prev_ua = hw.actual_ua

    hw.actual_ug2 = _set_param_calibrated(
        client, "Ug2", "ug2", target_ug2, hw.prev_ug2, calibration,
        settle_per_volt_s=cfg.health_ug2_settle_per_volt_s,
        settle_base_s=cfg.health_ug2_settle_base_s,
        tolerance=cfg.health_ug2_tolerance_v,
        max_retries=cfg.health_ug2_retries,
    )
    if abs(hw.actual_ug2 - target_ug2) > cfg.health_ug2_tolerance_v:
        raise RuntimeError(f"Ug2 verify failed: target={target_ug2:.1f}, actual={hw.actual_ug2:.1f}")
    hw.prev_ug2 = hw.actual_ug2

    _delay = _ia_sample_delay_s(cfg)
    point = _read_measurement_point(client, calibration, cfg.health_ia_samples, _delay)
    _raise_on_hw_error(point.get("er", 0))
    if progress:
        progress({"event": "live_point", "point": dict(point, step="op")})
    return point


# SRK phases follow "µTracer User Manual", see SOURCES_INDEX.md
def _measure_s_phase(
    client: LM19Serial, cfg: HealthConfig, calibration: CalibrationData, hw: _HwState,
    ug1_sweep: List[float], n_repeats: int,
    ia_op: float, target_ug1_op: float,
    srk_idx: int, total_srk: int,
    measurement_points: List[Dict],
    progress: Optional[Callable[[HealthProgress], None]],
    stop: Optional[Callable[[], bool]],
) -> Tuple[List[float], List[float], int]:
    """Phase 1: Ug1 sweep at Ua = OP for S (gm). Returns (s_ug1, s_ia, srk_idx)."""
    s_ug1: List[float] = [target_ug1_op]
    s_ia: List[float] = [ia_op]
    for ug1_val in ug1_sweep:
        if stop and stop():
            break
        hw.actual_ug1 = _set_param_calibrated(
            client, "Ug1", "ug1", ug1_val, hw.prev_ug1, calibration,
            settle_per_volt_s=cfg.health_ug1_settle_per_volt_s,
            settle_base_s=cfg.health_ug1_settle_base_s,
            tolerance=cfg.health_ug1_tolerance_v,
            max_retries=cfg.health_ug1_retries,
            encode_fn=encode_ug1, decode_fn=decode_ug1,
            stop=stop,
        )
        hw.prev_ug1 = hw.actual_ug1
        _delay = _ia_sample_delay_s(cfg)
        pt = _read_measurement_point(client, calibration, n_repeats, _delay)
        _raise_on_hw_error(pt.get("er", 0))
        s_ug1.append(float(pt["ug1"]))
        s_ia.append(float(pt.get("ia", 0.0)))
        srk_idx += 1
        measurement_points.append(dict(pt, step=f"srk_s{srk_idx}"))
        if progress:
            progress({"event": "srk_progress", "done": srk_idx, "total": total_srk})
            progress({"event": "live_point", "point": dict(pt, step=f"srk_s{srk_idx}")})
    return s_ug1, s_ia, srk_idx


def _measure_r_phase(
    client: LM19Serial, lamp: LampConfig, cfg: HealthConfig, calibration: CalibrationData,
    hw: _HwState,
    target_ua_op: float, target_ug1_op: float, target_ug2_op: float,
    delta_ua: float, n_repeats: int,
    ug2_track_ua: bool, ug2_offset: float,
    srk_idx: int, total_srk: int,
    measurement_points: List[Dict],
    progress: Optional[Callable[[HealthProgress], None]],
    stop: Optional[Callable[[], bool]],
) -> Tuple[List[float], List[float], int]:
    """Phase 2: Ua ± δ at Ug1 = OP for R (Rp). Returns (r_ua, r_ia, srk_idx)."""
    hw.actual_ug1 = _set_param_calibrated(
        client, "Ug1", "ug1", target_ug1_op, hw.prev_ug1, calibration,
        settle_per_volt_s=cfg.health_ug1_settle_per_volt_s,
        settle_base_s=cfg.health_ug1_settle_base_s,
        tolerance=cfg.health_ug1_tolerance_v,
        max_retries=cfg.health_ug1_retries,
        encode_fn=encode_ug1, decode_fn=decode_ug1,
        stop=stop,
    )
    hw.prev_ug1 = hw.actual_ug1

    r_ua: List[float] = []
    r_ia: List[float] = []
    for ua_val in [target_ua_op - delta_ua, target_ua_op + delta_ua]:
        if stop and stop():
            break
        hw.actual_ua = _set_param_calibrated(
            client, "Ua", "ua", ua_val, hw.prev_ua, calibration,
            settle_per_volt_s=cfg.health_ua_settle_per_volt_s,
            settle_base_s=cfg.health_ua_settle_base_s,
            tolerance=cfg.health_ua_tolerance_v,
            max_retries=cfg.health_ua_retries,
            stop=stop,
        )
        hw.prev_ua = hw.actual_ua
        tg2 = _ug2_for_ua(lamp, hw.actual_ua, ug2_track_ua, ug2_offset, target_ug2_op)
        if not lamp.is_triode and (ug2_track_ua or abs(tg2 - hw.prev_ug2) > 0.5):
            hw.actual_ug2 = _set_param_calibrated(
                client, "Ug2", "ug2", tg2, hw.prev_ug2, calibration,
                settle_per_volt_s=cfg.health_ug2_settle_per_volt_s,
                settle_base_s=cfg.health_ug2_settle_base_s,
                tolerance=cfg.health_ug2_tolerance_v,
                max_retries=cfg.health_ug2_retries,
                stop=stop,
            )
            hw.prev_ug2 = hw.actual_ug2
        _delay = _ia_sample_delay_s(cfg)
        pt = _read_measurement_point(client, calibration, n_repeats, _delay)
        _raise_on_hw_error(pt.get("er", 0))
        r_ua.append(float(pt["ua"]))
        r_ia.append(float(pt.get("ia", 0.0)))
        srk_idx += 1
        measurement_points.append(dict(pt, step=f"srk_r{srk_idx}"))
        if progress:
            progress({"event": "srk_progress", "done": srk_idx, "total": total_srk})
            progress({"event": "live_point", "point": dict(pt, step=f"srk_r{srk_idx}")})
    return r_ua, r_ia, srk_idx


def _measure_sg2_phase(
    client: LM19Serial, cfg: HealthConfig, calibration: CalibrationData, hw: _HwState,
    target_ua_op: float, target_ug2_op: float, delta_ug2: float, n_repeats: int,
    srk_idx: int, total_srk: int,
    measurement_points: List[Dict],
    progress: Optional[Callable[[HealthProgress], None]],
    stop: Optional[Callable[[], bool]],
) -> Tuple[List[float], List[float], int]:
    """Phase 2b: Ug2 ± δ at OP Ua and Ug1 for Sg2 (screen transconductance).

    Returns (sg2_ug2, sg2_ia, srk_idx).
    """
    # ML-060: the R-phase leaves the anode at its LAST sweep point
    # (r_center + δ); Sg2 is defined at the operating point — restore Ua
    # first, otherwise Sg2 (and mu_g1g2 derived from it) carries a
    # systematic Ua bias.
    if not (stop and stop()):
        hw.actual_ua = _set_param_calibrated(
            client, "Ua", "ua", target_ua_op, hw.prev_ua, calibration,
            settle_per_volt_s=cfg.health_ua_settle_per_volt_s,
            settle_base_s=cfg.health_ua_settle_base_s,
            tolerance=cfg.health_ua_tolerance_v,
            max_retries=cfg.health_ua_retries,
            stop=stop,
        )
        hw.prev_ua = hw.actual_ua

    sg2_ug2: List[float] = []
    sg2_ia: List[float] = []
    for ug2_val in [target_ug2_op - delta_ug2, target_ug2_op + delta_ug2]:
        if stop and stop():
            break
        hw.actual_ug2 = _set_param_calibrated(
            client, "Ug2", "ug2", ug2_val, hw.prev_ug2, calibration,
            settle_per_volt_s=cfg.health_ug2_settle_per_volt_s,
            settle_base_s=cfg.health_ug2_settle_base_s,
            tolerance=cfg.health_ug2_tolerance_v,
            max_retries=cfg.health_ug2_retries,
            stop=stop,
        )
        hw.prev_ug2 = hw.actual_ug2
        _delay = _ia_sample_delay_s(cfg)
        pt = _read_measurement_point(client, calibration, n_repeats, _delay)
        _raise_on_hw_error(pt.get("er", 0))
        sg2_ug2.append(float(pt["ug2"]))
        sg2_ia.append(float(pt.get("ia", 0.0)))
        srk_idx += 1
        measurement_points.append(dict(pt, step=f"srk_sg2_{srk_idx}"))
        if progress:
            progress({"event": "srk_progress", "done": srk_idx, "total": total_srk})
            progress({"event": "live_point", "point": dict(pt, step=f"srk_sg2_{srk_idx}")})
    return sg2_ug2, sg2_ia, srk_idx


def _restore_op(
    client: LM19Serial, lamp: LampConfig, cfg: HealthConfig,
    calibration: CalibrationData, hw: _HwState,
    target_ua_op: float, ug2_track_ua: bool, ug2_offset: float, target_ug2_op: float,
) -> None:
    """Restore Ua and Ug2 to OP after SRK sweeps."""
    hw.actual_ua = _set_param_calibrated(
        client, "Ua", "ua", target_ua_op, hw.prev_ua, calibration,
        settle_per_volt_s=cfg.health_ua_settle_per_volt_s,
        settle_base_s=cfg.health_ua_settle_base_s,
        tolerance=cfg.health_ua_tolerance_v,
        max_retries=cfg.health_ua_retries,
    )
    hw.prev_ua = hw.actual_ua
    tg2 = _ug2_for_ua(lamp, target_ua_op, ug2_track_ua, ug2_offset, target_ug2_op)
    if not lamp.is_triode and abs(tg2 - hw.prev_ug2) > 0.5:
        hw.actual_ug2 = _set_param_calibrated(
            client, "Ug2", "ug2", tg2, hw.prev_ug2, calibration,
            settle_per_volt_s=cfg.health_ug2_settle_per_volt_s,
            settle_base_s=cfg.health_ug2_settle_base_s,
            tolerance=cfg.health_ug2_tolerance_v,
            max_retries=cfg.health_ug2_retries,
        )
        hw.prev_ug2 = hw.actual_ug2


def _servo_ug1_to_ref_ia(
    client: LM19Serial,
    cfg: HealthConfig,
    calibration: CalibrationData,
    lamp: LampConfig,
    hw: _HwState,
    *,
    ref_ia: Optional[float],
    target_ug1: float,
    ia_at_target: float,
    point_at_target: Dict,
    ug2_mode: str,
    lamp_id: str,
    enabled: bool,
    measurement_points: List[Dict],
    progress: Optional[Callable[[HealthProgress], None]],
    stop: Optional[Callable[[], bool]],
) -> Dict:
    """Bisect Ug1 until Ia reaches the reference current, then stay there.

    Ia is monotonic in Ug1 (less negative bias -> more current), so the
    residual ``Ia(Ug1) - ref_ia`` is monotonic and a bracket is enough.

    Every probe goes through the same Pa/Pg2 check as the OP ramp: the
    servo walks the bias toward higher current, which is exactly the
    direction that overheats a tube whose Ia is high for its bias.

    Returns a dict that always carries ``status`` from
    ``BIAS_SERVO_STATUSES`` plus the final point — the caller keeps
    measuring wherever the servo left the tube, so a silent failure
    would move the OP without telling anyone.
    """
    result: Dict = {
        "status": BIAS_SERVO_DISABLED,
        "ug1": float(target_ug1),
        "ia": float(ia_at_target),
        "bias_shift_v": None,
        "iterations": 0,
        "point": point_at_target,
        "ref_ia": ref_ia,
        "shift_limit_v": None,
    }
    if not enabled:
        return result
    if ref_ia is None or ref_ia <= EPS:
        result["status"] = BIAS_SERVO_NO_REFERENCE
        log.warning("Bias servo skipped: reference Ia is unusable (%r)", ref_ia)
        return result

    tol = max(EPS, float(cfg.health_bias_servo_tol_ma))
    max_shift = abs(float(cfg.health_bias_servo_max_shift_v))
    max_iter = max(1, int(cfg.health_bias_servo_max_iter))
    step_pre = max(0.05, float(cfg.health_bias_servo_step_v))

    # Per-tube excursion limit: the shift a
    # worn tube needs is ~deficit/S, so the allowed excursion scales
    # with the measured current deficit instead of one park-wide
    # constant — which is too small for a 6L6 (needs ~6 V) and ~7x too
    # wide for a 12AX7 (needs ~0.4 V from a -2 V bias). The margin
    # covers a tube whose real S sagged below the datasheet value;
    # ``bias_servo_max_shift_v`` remains the absolute safety ceiling
    # and at least one step is always allowed.
    margin = max(1.0, float(cfg.health_bias_servo_shift_margin))
    s_ref = float(lamp.s or 0.0)
    if s_ref > EPS:
        estimated_shift = abs(float(ref_ia) - float(ia_at_target)) / s_ref
        shift_limit = min(max_shift, max(step_pre, margin * estimated_shift))
    else:
        shift_limit = max_shift
        log.warning("Bias servo: lamp S unavailable — using the global "
                    "excursion ceiling %.2f V", max_shift)
    result["shift_limit_v"] = shift_limit

    ug1_lo_limit, ug1_hi_limit = SET_LIMITS["ug1"]
    lo = max(ug1_lo_limit, target_ug1 - shift_limit)
    hi = min(ug1_hi_limit, target_ug1 + shift_limit)

    def _probe(ug1_value: float, idx: int,
               step_tag: str = STEP_BIAS_SERVO) -> Dict:
        hw.actual_ug1 = _set_param_calibrated(
            client, "Ug1", "ug1", ug1_value, hw.prev_ug1, calibration,
            settle_per_volt_s=cfg.health_ug1_settle_per_volt_s,
            settle_base_s=cfg.health_ug1_settle_base_s,
            tolerance=cfg.health_ug1_tolerance_v,
            max_retries=cfg.health_ug1_retries,
            encode_fn=encode_ug1, decode_fn=decode_ug1,
            stop=stop,
        )
        hw.prev_ug1 = hw.actual_ug1
        pt = _read_measurement_point(client, calibration,
                                     cfg.health_ia_samples, _ia_sample_delay_s(cfg))
        _raise_on_hw_error(pt.get("er", 0))
        # The reference the servo is converging to rides in every probe
        # point (saved and live): the points table renders the per-probe
        # deviation from it, which makes the convergence visible row by
        # row without consulting the result block.
        pt["ref_ia"] = float(ref_ia)
        _check_pa_pg2_at_step(
            pt, lamp, cfg,
            step_idx=idx, total_steps=max_iter,
            start_ug1=target_ug1, target_ug1=ug1_value,
            lamp_id=lamp_id, ug2_mode=ug2_mode,
        )
        if progress:
            # live_point first: the live panel shows the probe's readings
            # the same way every other phase reports its measurements.
            # The step tag rides along so the live steps table can name
            # rows the plan did not pre-create (probe count is unknown).
            progress({"event": "live_point", "point": dict(pt, step=step_tag)})
            progress({
                "event": "bias_servo",
                "iteration": idx,
                "max_iterations": max_iter,
                "ug1": float(pt["ug1"]),
                "ia_ma": float(pt["ia"]),
                "ref_ia_ma": float(ref_ia),
                "target_ug1": float(target_ug1),
            })
        return pt

    step_v = max(0.05, float(cfg.health_bias_servo_step_v))
    ug1_floor = max(EPS, float(cfg.health_bias_servo_ug1_floor_v))
    # Proactive power ceiling: the servo must stop short of the Pa
    # protection trip, not ride into it — the trip is the backstop,
    # never part of the control loop. Applied only while the current
    # is walked UP; walking down reduces Pa by construction.
    walking_up = ia_at_target < ref_ia
    ceiling_w = 0.0
    if lamp.pa_max is not None and walking_up:
        ceiling_w = (float(lamp.pa_max) * float(cfg.health_pa_safety_pct) / 100.0
                     * float(cfg.health_bias_servo_pa_ceiling_pct) / 100.0)

    def _over_ceiling(pt: Dict) -> bool:
        return (ceiling_w > 0.0
                and float(pt["ua"]) * float(pt["ia"]) / MW_PER_W > ceiling_w)

    def _fail_restore(idx: int) -> Dict:
        # Every failure path leaves the tube at the PLAN bias, and the
        # restore is a real probe on a live tube — record it. No shift
        # was applied to the measurement, so bias_shift_v stays None.
        restore_pt = _probe(target_ug1, idx, STEP_BIAS_SERVO_RESTORE)
        measurement_points.append(dict(restore_pt, step=STEP_BIAS_SERVO_RESTORE))
        result.update(status=BIAS_SERVO_UNREACHABLE, iterations=idx,
                      ug1=float(restore_pt["ug1"]),
                      ia=float(restore_pt["ia"]), point=restore_pt)
        return result

    try:
        if abs(ia_at_target - ref_ia) <= tol:
            result.update(status=BIAS_SERVO_OK, bias_shift_v=0.0)
            return result

        direction = 1.0 if walking_up else -1.0
        limit_ug1 = hi if walking_up else lo
        if abs(limit_ug1 - target_ug1) <= EPS:
            result["status"] = BIAS_SERVO_UNREACHABLE
            log.warning("Bias servo: no Ug1 headroom from %.2f V (limits %.2f..%.2f)",
                        target_ug1, lo, hi)
            return result

        # Phase 1 — walk from the plan bias in small steps. Overshoot is
        # bounded by one step's worth of current (S·step), unlike the old
        # edge-first bracket that slammed the full excursion in one probe.
        idx = 0
        prev_ug1 = target_ug1
        crossed_pt: Optional[Dict] = None
        best_pt: Optional[Dict] = None

        def _consider(pt: Dict) -> None:
            # Track the closest probe seen ANYWHERE (walk + bisection):
            # with measurement noise or thermal drift the nearest point
            # is not necessarily the last one, and discarding it turns
            # a reachable reference into a false "unreachable".
            nonlocal best_pt
            if best_pt is None or (abs(float(pt["ia"]) - ref_ia)
                                   < abs(float(best_pt["ia"]) - ref_ia)):
                best_pt = pt

        while idx < max_iter:
            if stop and stop():
                raise RuntimeError("Health test stopped")
            idx += 1
            next_ug1 = min(hi, max(lo, prev_ug1 + direction * step_v))
            pt = _probe(next_ug1, idx)
            measurement_points.append(dict(pt, step=STEP_BIAS_SERVO))
            ia_pt = float(pt["ia"])
            _consider(pt)
            result.update(iterations=idx, ug1=float(pt["ug1"]), ia=ia_pt, point=pt)
            if abs(ia_pt - ref_ia) <= tol:
                measurement_points[-1]["step"] = STEP_BIAS_SERVO_OP
                measurement_points[-1]["bias_shift_v"] = (
                    float(pt["ug1"]) - target_ug1)
                if progress:
                    # The accepted probe's live_point went out with a
                    # plain probe tag (acceptance was not known yet) —
                    # tell the live view which row became the OP.
                    progress({
                        "event": "bias_servo_accept",
                        "ug1": float(pt["ug1"]),
                        "ia_ma": float(pt["ia"]),
                        "bias_shift_v": float(pt["ug1"]) - target_ug1,
                    })
                result.update(status=BIAS_SERVO_OK,
                              bias_shift_v=float(pt["ug1"]) - target_ug1)
                return result
            if (ia_pt > ref_ia) if walking_up else (ia_pt < ref_ia):
                crossed_pt = pt
                break
            if _over_ceiling(pt):
                log.warning(
                    "Bias servo stopped at the Pa ceiling: %.2f W > %.2f W "
                    "(%.0f%% of the trip limit) at Ug1=%.2f V, Ia=%.2f mA — "
                    "ref %.2f mA unreachable, restoring plan bias",
                    float(pt["ua"]) * ia_pt / MW_PER_W, ceiling_w,
                    float(cfg.health_bias_servo_pa_ceiling_pct),
                    float(pt["ug1"]), ia_pt, ref_ia)
                return _fail_restore(idx + 1)
            if abs(next_ug1 - limit_ug1) <= EPS:
                log.warning(
                    "Bias servo: ref Ia=%.2f mA unreachable within %.2f V of "
                    "Ug1=%.2f V (got %.2f mA at %.2f V) — restoring plan bias",
                    ref_ia, shift_limit, target_ug1, ia_pt, float(pt["ug1"]))
                return _fail_restore(idx + 1)
            prev_ug1 = float(pt["ug1"])

        if crossed_pt is None:
            log.warning("Bias servo walk exhausted %d probes without reaching "
                        "ref %.2f mA — restoring plan bias", max_iter, ref_ia)
            return _fail_restore(idx + 1)

        # Phase 2 — bisect INSIDE the last step only (bracket <= step_v).
        # Ia is monotonic in Ug1, so the lower-Ug1 end is always below ref.
        low_ug1, high_ug1 = sorted((prev_ug1, float(crossed_pt["ug1"])))
        floor_stop = False
        while idx < max_iter:
            # The device cannot honor sub-resolution setpoints — probing
            # below the Ug1 floor burns budget without moving the tube.
            if high_ug1 - low_ug1 <= ug1_floor:
                floor_stop = True
                break
            if stop and stop():
                raise RuntimeError("Health test stopped")
            idx += 1
            pt = _probe(0.5 * (low_ug1 + high_ug1), idx)
            measurement_points.append(dict(pt, step=STEP_BIAS_SERVO))
            ia_mid = float(pt["ia"])
            _consider(pt)
            result.update(iterations=idx, ug1=float(pt["ug1"]), ia=ia_mid, point=pt)
            if abs(ia_mid - ref_ia) <= tol:
                measurement_points[-1]["step"] = STEP_BIAS_SERVO_OP
                measurement_points[-1]["bias_shift_v"] = (
                    float(pt["ug1"]) - target_ug1)
                if progress:
                    # The accepted probe's live_point went out with a
                    # plain probe tag (acceptance was not known yet) —
                    # tell the live view which row became the OP.
                    progress({
                        "event": "bias_servo_accept",
                        "ug1": float(pt["ug1"]),
                        "ia_ma": float(pt["ia"]),
                        "bias_shift_v": float(pt["ug1"]) - target_ug1,
                    })
                result.update(status=BIAS_SERVO_OK,
                              bias_shift_v=float(pt["ug1"]) - target_ug1)
                return result
            if ia_mid < ref_ia:
                low_ug1 = float(pt["ug1"])
            else:
                high_ug1 = float(pt["ug1"])

        if floor_stop:
            # The bracket is at hardware resolution: the best probe IS the
            # closest achievable point, so it is accepted — rejecting it
            # would make the servo unable to converge on any tube where
            # S×floor exceeds the Ia tolerance (EL84: ~1.1 mA vs 0.5).
            # The residual is bounded by S×floor by construction.
            accept_pt = best_pt
            if abs(float(result["ug1"]) - float(accept_pt["ug1"])) > EPS_COARSE:
                # The tube sits at the LAST probe, not the best one —
                # move it there and report what it actually measures.
                idx += 1
                accept_pt = _probe(float(accept_pt["ug1"]), idx, STEP_BIAS_SERVO_OP)
                measurement_points.append(dict(accept_pt, step=STEP_BIAS_SERVO_OP))
            else:
                # The last probe IS the accepted one — retag it so the
                # steps table can tell the measuring point from the
                # intermediate probes.
                measurement_points[-1]["step"] = STEP_BIAS_SERVO_OP
            measurement_points[-1]["bias_shift_v"] = (
                float(accept_pt["ug1"]) - target_ug1)
            if progress:
                progress({
                    "event": "bias_servo_accept",
                    "ug1": float(accept_pt["ug1"]),
                    "ia_ma": float(accept_pt["ia"]),
                    "bias_shift_v": float(accept_pt["ug1"]) - target_ug1,
                })
            residual = abs(float(accept_pt["ia"]) - ref_ia)
            if residual > tol:
                log.warning(
                    "Bias servo accepted the closest achievable point: "
                    "Ia=%.2f mA vs ref %.2f mA (residual %.2f > tol %.2f; "
                    "Ug1 floor %.2f V limits resolution to ~S*floor mA)",
                    float(accept_pt["ia"]), ref_ia, residual, tol, ug1_floor)
            result.update(status=BIAS_SERVO_OK, iterations=idx,
                          ug1=float(accept_pt["ug1"]),
                          ia=float(accept_pt["ia"]), point=accept_pt,
                          bias_shift_v=float(accept_pt["ug1"]) - target_ug1)
            return result

        log.warning("Bias servo did not converge: Ia=%.2f mA vs ref %.2f mA "
                    "(tol %.2f, bracket %.3f V) — restoring plan bias",
                    float(best_pt["ia"]) if best_pt else float("nan"),
                    ref_ia, tol, high_ug1 - low_ug1)
        return _fail_restore(idx + 1)
    except HealthProtectionError as protection_exc:
        _restore_ug1_safe_lock(client, cfg, calibration, hw, protection_exc)
        raise


def _emission_sensitivity(
    point_op: Dict, lamp: LampConfig, cfg: HealthConfig,
) -> Dict:
    """How much the reduced-heater probe can actually see at this OP.

    Ia(Uh) is flat while the cathode is space-charge limited, so a probe
    run at a small fraction of the tube's current capability returns
    ~1.0 even for a badly depleted cathode. Reporting the ratio turns a
    false "reserve normal" into a qualified one.
    """
    ik_max = lamp.ia_max_limit
    if not ik_max or ik_max <= EPS:
        return {"ratio": None, "low": False, "ik_op": None, "ik_max": None}
    ik_op = float(point_op.get("ia", 0.0)) + float(point_op.get("ig2", 0.0))
    ratio = ik_op / float(ik_max)
    low = ratio < float(cfg.health_emission_min_ik_ratio)
    if low:
        log.warning("Emission test sensitivity is low: Ik=%.2f mA is %.0f%% of "
                    "Ik_max=%.2f mA (threshold %.0f%%)",
                    ik_op, 100.0 * ratio, float(ik_max),
                    100.0 * float(cfg.health_emission_min_ik_ratio))
    return {"ratio": ratio, "low": low, "ik_op": ik_op, "ik_max": float(ik_max)}


def _emission_grid_step(cfg: HealthConfig) -> float:
    """Ratio decrement between sweep grid points (also the adaptive
    descent step, so extension points continue the configured pace)."""
    steps = max(2, int(cfg.health_emission_uh_sweep_steps))
    min_ratio = min(0.99, max(0.05, float(cfg.health_emission_uh_sweep_min_ratio)))
    return (1.0 - min_ratio) / (steps - 1)


def _emission_ratio_grid(cfg: HealthConfig, uh_ratio: float) -> List[float]:
    """Descending heater ratios for the sweep, always hitting *uh_ratio*.

    Keeping the configured single-point ratio on the grid is what lets
    ``emission_ratio`` stay bit-comparable between the two modes (and so
    with every historical measurement and saved baseline).
    """
    steps = max(2, int(cfg.health_emission_uh_sweep_steps))
    min_ratio = min(0.99, max(0.05, float(cfg.health_emission_uh_sweep_min_ratio)))
    step = _emission_grid_step(cfg)
    grid = {1.0, uh_ratio, min_ratio}
    for i in range(steps):
        grid.add(round(1.0 - step * i, 4))
    return sorted((r for r in grid if 0.0 < r <= 1.0), reverse=True)


# ── Miram knee fit ───────────────────────────────────────────────────
# Knee-estimate confidence registry (stored in health JSON, compared
# stringly by the UI — constants, not literals).
KNEE_CONF_OK = "ok"
KNEE_CONF_LOW = "low"
KNEE_CONFIDENCES = frozenset({KNEE_CONF_OK, KNEE_CONF_LOW})

# A point belongs to the plateau while it stays within this fraction of
# the knee criterion (drop_pct) of the EXTRAPOLATED plateau line. The
# line may tilt (cathode temperature shifts the contact potential, and
# work-function nonuniformity softens the plateau) — which is exactly
# why the drop is measured from the line, not from Ia at nominal.
_PLATEAU_MEMBER_FRACTION = 0.5
# A fitted "plateau" steeper than this many multiples of
# plateau_ia/uh_nom is no plateau at all: the physical space-charge tilt
# runs ~0.5-1 in these units while the emission branch is an order of
# magnitude steeper — such a tube is emission-limited at nominal already.
_PLATEAU_MAX_REL_SLOPE = 2.0


def _fit_line(pts: List[Dict]) -> Tuple[float, float]:
    """Least-squares ``ia = m*uh + b`` over points (exact for two)."""
    n = len(pts)
    mx = sum(float(p["uh"]) for p in pts) / n
    my = sum(float(p["ia"]) for p in pts) / n
    sxx = sum((float(p["uh"]) - mx) ** 2 for p in pts)
    sxy = sum((float(p["uh"]) - mx) * (float(p["ia"]) - my) for p in pts)
    m = sxy / sxx if sxx > EPS else 0.0
    return m, my - m * mx


def _classify_emission_curve(
    curve: List[Dict], drop_pct: float,
) -> Tuple[List[Dict], List[Dict], Optional[Tuple[float, float]]]:
    """Split a descending sweep into plateau and steep-branch points.

    The first two (hottest) points seed the plateau line; each next point
    joins the plateau while it stays within half the knee criterion of
    the line's extrapolation (refitting as it grows). The first point
    that falls away — and everything below it, the emission branch is
    monotone — is steep.

    Returns ``(plateau_pts, steep_pts, (m, b))``, or ``(usable, [],
    None)`` when fewer than two usable points exist.
    """
    usable = [p for p in curve
              if p.get("ia") is not None and p.get("uh") is not None]
    usable.sort(key=lambda p: -float(p["uh"]))
    if len(usable) < 2:
        return usable, [], None
    ref = abs(float(usable[0]["ia"]))
    tol = ref * (drop_pct / 100.0) * _PLATEAU_MEMBER_FRACTION
    plateau = usable[:2]
    steep: List[Dict] = []
    m, b = _fit_line(plateau)
    for p in usable[2:]:
        if not steep:
            predicted = m * float(p["uh"]) + b
            if predicted - float(p["ia"]) <= tol:
                plateau.append(p)
                m, b = _fit_line(plateau)
                continue
        steep.append(p)
    return plateau, steep, (m, b)


def _sweep_needs_extension(
    curve: List[Dict], uh_nom: float, drop_pct: float,
) -> bool:
    """Adaptive Miram descent predicate: keep stepping the heater down
    while the steep branch holds fewer than two points AND the curve
    still has a plateau to fall from. A tube that is emission-limited at
    nominal (no plateau) or dead resolves without going deeper."""
    plateau, steep, line = _classify_emission_curve(curve, drop_pct)
    if line is None:
        return True
    plateau_ia = float(plateau[0]["ia"]) if plateau else 0.0
    if plateau_ia <= EPS or uh_nom <= EPS:
        return False
    if line[0] > _PLATEAU_MAX_REL_SLOPE * plateau_ia / uh_nom:
        return False
    return len(steep) < 2


def _find_emission_knee(
    curve: List[Dict], uh_nom: float, drop_pct: float,
) -> Dict:
    """Locate the space-charge/emission knee — classic Miram two-line fit.

    The knee is the INTERSECTION of two fitted lines — the (possibly
    tilted) space-charge plateau and the emission-limited branch — not a
    threshold crossing: a healthy tube whose plateau merely sags a few
    percent by 80% heater must not read as "knee found".

    Outcomes:
      * >=2 steep points — two-line intersection, confidence "ok"
        (clamped into the bracket between the last plateau and the first
        steep point; needing the clamp demotes confidence to "low");
      * exactly 1 steep point — bracket midpoint, confidence "low";
      * no steep points — ``below_range``: the knee, if any, is deeper
        than the sweep went, the reserve is at least the swept span;
      * no plateau (the "plateau" fit is itself steep) — the tube is
        emission-limited at nominal already: knee at the top point,
        reserve 0, confidence "low".
    """
    out: Dict = {"uh_knee": None, "reserve_pct": None, "below_range": False,
                 "plateau_ia": None, "knee_confidence": None,
                 "plateau_slope_ma_per_v": None}
    usable = [p for p in curve
              if p.get("ia") is not None and p.get("uh") is not None]
    if len(usable) < 2 or uh_nom <= EPS:
        return out
    usable.sort(key=lambda p: -float(p["uh"]))
    plateau_ia = float(usable[0]["ia"])
    out["plateau_ia"] = plateau_ia
    if plateau_ia <= EPS:
        return out
    plateau, steep, line = _classify_emission_curve(curve, drop_pct)
    m1, b1 = line
    out["plateau_slope_ma_per_v"] = m1

    def _accept(uh_knee: float, conf: str) -> Dict:
        out["uh_knee"] = uh_knee
        out["reserve_pct"] = 100.0 * (uh_nom - uh_knee) / uh_nom
        out["knee_confidence"] = conf
        return out

    if m1 > _PLATEAU_MAX_REL_SLOPE * plateau_ia / uh_nom:
        return _accept(float(usable[0]["uh"]), KNEE_CONF_LOW)
    if not steep:
        lowest_uh = float(usable[-1]["uh"])
        out["below_range"] = True
        out["reserve_pct"] = 100.0 * (uh_nom - lowest_uh) / uh_nom
        return out
    hi_uh = float(plateau[-1]["uh"])   # last (lowest) plateau point
    lo_uh = float(steep[0]["uh"])      # first (highest) steep point
    if len(steep) == 1:
        return _accept(0.5 * (hi_uh + lo_uh), KNEE_CONF_LOW)
    m2, b2 = _fit_line(steep)
    if m2 - m1 <= EPS:
        # Degenerate: the "steep" points run parallel to the plateau
        # line (offset shelf) — the intersection does not exist, but the
        # bracket does.
        return _accept(0.5 * (hi_uh + lo_uh), KNEE_CONF_LOW)
    uh_knee = (b1 - b2) / (m2 - m1)
    if lo_uh - EPS <= uh_knee <= hi_uh + EPS:
        return _accept(uh_knee, KNEE_CONF_OK)
    return _accept(min(max(uh_knee, lo_uh), hi_uh), KNEE_CONF_LOW)


def _run_emission(
    client: LM19Serial, cfg: HealthConfig, calibration: CalibrationData, hw: _HwState,
    lamp: LampConfig, op_plan: Dict, emission_plan: Dict,
    warmup_s: int,
    measurement_points: List[Dict],
    progress: Optional[Callable[[HealthProgress], None]],
    stop: Optional[Callable[[], bool]],
) -> Dict:
    """Run emission test (100% Uh → reduced Uh → stabilize → measure).

    In ``EMISSION_MODE_SWEEP`` the heater walks a descending grid instead
    of a single reduced point, so the knee of the Ia(Uh) characteristic
    can be located. The configured single-point ratio always stays on the
    grid, which keeps ``emission_ratio`` comparable across modes and with
    every stored baseline.

    Returns dict with ia100, ia80, uh80, stable_info, confidence, plus
    the sweep block (curve/knee/reserve) when sweeping.
    """
    if progress:
        progress({"event": "step", "step": "uh80"})
    # Drive the heater on the channel the lamp actually uses. Current-heater
    # lamps (PCC84/85: uh=0, ih>0) driven by voltage would get Uh=0 — switching
    # the heater OFF mid-test and making the emission ratio garbage.
    use_uh = float(op_plan.get("uh", lamp.uh)) > 0.0
    heat_nom = (float(op_plan.get("uh", lamp.uh)) if use_uh
                else float(op_plan.get("ih", lamp.ih)))
    prev_heat = hw.prev_uh if use_uh else calibration.apply_read(
        "ih", decode_ih(client.get_param("Ih", real=True)))
    prev_heat = _set_heater_with_verify(client, use_uh, heat_nom, prev_heat, calibration)
    if use_uh:
        hw.prev_uh = prev_heat

    _delay = _ia_sample_delay_s(cfg)
    point_100 = _read_measurement_point(client, calibration, cfg.health_ia_samples, _delay)
    _raise_on_hw_error(point_100.get("er", 0))
    ia100 = float(point_100.get("ia", 0.0))
    measurement_points.append(dict(point_100, step="emission_100"))
    if progress:
        progress({"event": "live_point", "point": dict(point_100, step="emission_100")})

    # reduced-heater life test from "Accurate Tube Testing Information
    # (tubetesting.yolasite.com)", see SOURCES_INDEX.md
    emission_ratio = float(emission_plan.get("uh_ratio", cfg.health_emission_uh_ratio))
    mode = str(emission_plan.get("mode", cfg.health_emission_mode_default))
    if mode not in EMISSION_MODES:
        log.warning("Unknown emission mode %r — falling back to %s",
                    mode, EMISSION_MODE_SINGLE)
        mode = EMISSION_MODE_SINGLE
    ratios = ([r for r in _emission_ratio_grid(cfg, emission_ratio) if r < 1.0]
              if mode == EMISSION_MODE_SWEEP else [emission_ratio])

    sweep_budget_s = max(0.0, float(cfg.health_emission_sweep_max_total_s))
    sweep_started = time.time()
    curve: List[Dict] = [{"uh": heat_nom, "ia": ia100, "ratio": 1.0, "stable": True}]
    ia80: Optional[float] = None
    if abs(emission_ratio - 1.0) <= EPS_COARSE:
        # A 1.0 single-point ratio means "no reduction": the plateau point
        # IS the reference point. The sweep grid excludes 1.0 (already
        # measured as ia100), so without this seed the ratio would come
        # back None in sweep mode but ~1.0 in single mode.
        ia80 = ia100
    actual_heat80 = heat_nom
    stable_info: Dict = {"ia80": ia100, "stable": True, "elapsed_s": 0,
                         "t_max_s": 0, "eta_s_last": 0}
    any_unstable = False
    budget_exhausted = False

    # Adaptive Miram descent: past the configured grid the sweep keeps
    # stepping down (same pace) until the steep branch holds two points,
    # bounded by the absolute floor — brief starved operation is standard
    # Miram practice, but the floor keeps the cathode out of prolonged
    # deep emission limitation — and by the time budget below.
    grid_step = _emission_grid_step(cfg)
    abs_floor = min(0.99, max(0.05, float(
        cfg.health_emission_uh_sweep_abs_min_ratio)))
    drop_pct = float(cfg.health_emission_knee_drop_pct)
    pending: List[float] = list(ratios)
    idx = 0
    try:
        while pending:
            ratio = pending.pop(0)
            idx += 1
            if stop and stop():
                raise RuntimeError("Health test stopped")
            if (mode == EMISSION_MODE_SWEEP and sweep_budget_s > 0.0
                    and idx > 1 and time.time() - sweep_started > sweep_budget_s):
                budget_exhausted = True
                log.warning("Emission sweep budget of %.0f s exhausted after %d "
                            "points — knee estimate uses the partial curve",
                            sweep_budget_s, idx - 1)
                break
            heat_step = heat_nom * ratio
            prev_heat = _set_heater_with_verify(client, use_uh, heat_step,
                                                prev_heat, calibration)
            if use_uh:
                hw.prev_uh = prev_heat
            step_stable = _run_stabilized_ia80(
                client=client, cfg=cfg, calibration=calibration,
                warmup_s=warmup_s, progress=progress, stop=stop,
            )
            point_step = _read_measurement_point(client, calibration,
                                                 cfg.health_ia_samples, _delay)
            _raise_on_hw_error(point_step.get("er", 0))
            ia_step = float(point_step.get("ia", step_stable["ia80"]))
            measurement_points.append(dict(point_step, step="emission_80"))
            curve.append({"uh": prev_heat, "ia": ia_step, "ratio": ratio,
                          "stable": bool(step_stable.get("stable"))})
            if not step_stable.get("stable"):
                any_unstable = True
            if progress:
                progress({"event": "live_point", "point": dict(point_step, step="emission_80")})
                if mode == EMISSION_MODE_SWEEP:
                    progress({
                        "event": "emission_sweep",
                        "step_idx": idx,
                        "total_steps": idx + len(pending),
                        "uh": float(prev_heat),
                        "ratio": float(ratio),
                        "ia_ma": ia_step,
                        "ia100_ma": float(ia100),
                    })
            # The configured single-point ratio is the one that defines
            # emission_ratio in every mode.
            if abs(ratio - emission_ratio) <= EPS_COARSE:
                ia80 = ia_step
                actual_heat80 = prev_heat
                stable_info = step_stable
            if (mode == EMISSION_MODE_SWEEP and not pending
                    and _sweep_needs_extension(curve, heat_nom, drop_pct)):
                # The last step clamps TO the floor rather than stopping
                # one pace above it — the swept span (and so the reported
                # "reserve at least ...") reaches the floor exactly.
                nxt = max(round(ratio - grid_step, 4), round(abs_floor, 4))
                if nxt <= ratio - EPS:
                    pending.append(nxt)
    finally:
        # Restore heater to nominal on the same channel. Runs even on stop
        # or protection: leaving the cathode under-heated under load is
        # exactly the condition that poisons an oxide cathode.
        prev_heat = _set_heater_with_verify(client, use_uh, heat_nom, prev_heat, calibration)
        if use_uh:
            hw.prev_uh = prev_heat

    knee = _find_emission_knee(curve, heat_nom, drop_pct)
    confidence = "low" if (not stable_info.get("stable") or any_unstable
                           or budget_exhausted) else "ok"
    sweep = mode == EMISSION_MODE_SWEEP
    return {
        "ia100": ia100,
        "ia80": ia80,
        "uh80": actual_heat80,
        "stable_info": stable_info,
        "confidence": confidence,
        "mode": mode,
        "curve": curve if sweep else None,
        "uh_knee": knee["uh_knee"] if sweep else None,
        "reserve_pct": knee["reserve_pct"] if sweep else None,
        "knee_below_range": bool(knee["below_range"]) if sweep else False,
        "knee_confidence": knee["knee_confidence"] if sweep else None,
        "plateau_slope_ma_per_v": (knee["plateau_slope_ma_per_v"]
                                   if sweep else None),
        "sweep_truncated": budget_exhausted,
    }


def _compute_scores(
    ia_op: float,
    s: Optional[float], r: Optional[float], k: Optional[float],
    point_op: Dict, lamp: LampConfig, cfg: HealthConfig,
    refs: Dict, emission_enabled: bool,
    ia80: Optional[float], ia100: Optional[float],
) -> Dict:
    """Compute all health scores, weighted index, and verdict."""
    ia_pct = _clamp_score(_safe_pct(ia_op, refs["ia"]))
    s_pct = _clamp_score(_safe_pct(s, refs["s"]))
    r_pct = _clamp_score(_safe_pct(r, refs["r"]))
    k_pct = _clamp_score(_safe_pct(k, refs["k"]))

    rh_measured = None
    ih_op = point_op.get("ih")
    uh_op = point_op.get("uh")
    if ih_op and ih_op > EPS:
        rh_measured = float(uh_op) / float(ih_op)
    rh_score = _clamp_score(_safe_pct(refs["rh"], rh_measured) if (refs["rh"] and rh_measured) else None)

    screen_score = None
    if not lamp.is_triode and point_op.get("ia", 0) > EPS and refs["screen_ratio"]:
        screen_ratio = point_op.get("ig2", 0.0) / point_op.get("ia", EPS)
        screen_score = _clamp_score(100.0 * (refs["screen_ratio"] / screen_ratio) if screen_ratio > EPS else None)

    emission_ratio = (ia80 / ia100) if (ia80 is not None and ia100 and ia100 > EPS) else None
    ref_emission_ratio = refs.get("emission_ratio")
    if not (isinstance(ref_emission_ratio, (int, float)) and ref_emission_ratio > EPS):
        ref_emission_ratio = float(cfg.health_emission_ratio_nominal)
    emission_score = None
    if emission_ratio is not None:
        emission_score = _clamp_score(
            100.0 * (emission_ratio / max(EPS, float(ref_emission_ratio)))
        )
    emission_verdict = (_emission_verdict(emission_ratio, cfg) if emission_enabled
                        else EMISSION_VERDICT_NA)
    sensitivity = _emission_sensitivity(point_op, lamp, cfg)
    low_sensitivity = bool(emission_enabled and emission_ratio is not None
                           and sensitivity["low"])

    score_map = {
        "ia": ia_pct,
        "s": s_pct,
        "rh": rh_score,
        "screen": screen_score if not lamp.is_triode else None,
        "emission": emission_score if emission_enabled else None,
    }
    weights = {
        "ia": cfg.health_weight_ia,
        "s": cfg.health_weight_s,
        "rh": cfg.health_weight_rh,
        "screen": cfg.health_weight_screen if not lamp.is_triode else 0.0,
        "emission": cfg.health_weight_emission if emission_enabled else 0.0,
    }
    index = _weighted_index(score_map, weights, bool(cfg.health_renormalize_weights_if_metric_missing))
    verdict_str = _verdict(index, cfg)

    return {
        "ia_pct": ia_pct, "s_pct": s_pct, "r_pct": r_pct, "k_pct": k_pct,
        "rh_score": rh_score, "screen_score": screen_score,
        "emission_ratio": emission_ratio, "emission_score": emission_score,
        "emission_ratio_ref": ref_emission_ratio,
        "emission_verdict": emission_verdict,
        "emission_sensitivity_ratio": sensitivity["ratio"],
        "emission_low_sensitivity": low_sensitivity,
        "index": index, "verdict": verdict_str,
    }


# ── Orchestrator helpers ─────────────────────────────────────────────


@dataclass
class _HealthTargets:
    """Parsed measurement plan + resolved target voltages."""
    op_plan: Dict
    srk_plan: Dict
    emission_plan: Dict
    an: int
    ug2_track_ua: bool
    ug2_offset: float
    target_ua: float
    target_ug1: float
    target_ug2: float
    is_pentode_mode: bool
    bias_servo_enabled: bool = False


def _parse_health_targets(measurement_plan: Optional[Dict], lamp: LampConfig,
                          cfg: Optional[HealthConfig] = None) -> _HealthTargets:
    """Parse measurement_plan dict into resolved targets + derived flags."""
    plan = measurement_plan or {}
    op_plan = plan.get("op") if isinstance(plan.get("op"), dict) else {}
    srk_plan = plan.get("srk") if isinstance(plan.get("srk"), dict) else {}
    emission_plan = plan.get("emission") if isinstance(plan.get("emission"), dict) else {}

    an = int(plan.get("an", 1))
    ug2_track_ua = bool(plan.get("ug2_track_ua", False)) and not lamp.is_triode
    ug2_offset = float(plan.get("ug2_offset", 0.0))

    target_ua = float(op_plan.get("ua", lamp.ua))
    target_ug1 = float(op_plan.get("ug1", lamp.ug1))
    if ug2_track_ua:
        target_ug2 = max(0.0, target_ua + ug2_offset)
    else:
        target_ug2 = 0.0 if lamp.is_triode else float(op_plan.get("ug2", lamp.ug2))

    is_pentode_mode = not lamp.is_triode and not ug2_track_ua

    bias_servo_plan = plan.get("bias_servo") if isinstance(plan.get("bias_servo"), dict) else {}
    # A plan that says nothing about the servo falls back to the config,
    # exactly like the emission mode does — otherwise a programmatic
    # caller silently ignores bias_servo_enabled_default.
    servo_default = bool(getattr(cfg, "health_bias_servo_enabled_default", False)) if cfg else False

    return _HealthTargets(
        op_plan=op_plan, srk_plan=srk_plan, emission_plan=emission_plan,
        an=an, ug2_track_ua=ug2_track_ua, ug2_offset=ug2_offset,
        target_ua=target_ua, target_ug1=target_ug1, target_ug2=target_ug2,
        is_pentode_mode=is_pentode_mode,
        bias_servo_enabled=bool(bias_servo_plan.get("enabled", servo_default)),
    )


def _run_srk_phase(
    client: LM19Serial,
    cfg: HealthConfig,
    calibration: CalibrationData,
    lamp: LampConfig,
    hw: "_HwState",
    targets: _HealthTargets,
    ia_op: float,
    measurement_points: List[Dict],
    progress: Optional[Callable[[HealthProgress], None]],
    stop: Optional[Callable[[], bool]],
) -> Dict:
    """Run S/R/Sg2 phases and aggregate SRK results + uncertainty.

    Returns dict with keys: s_measured, r_measured, k_measured, sg2_measured,
    mu_g1g2, srk_uncertainty, sg2_rel, delta_ua, delta_ug1, delta_ug2,
    n_points, n_repeats, r_center, r_method, sg2_center, sg2_method.
    """
    if progress:
        progress({"event": "step", "step": "srk"})

    srk_plan = targets.srk_plan
    pct = DEFAULT_HEALTH_DELTA_PCT / 100.0
    _ua_min = getattr(cfg, "health_delta_ua_min_v", DELTA_UA_MIN_V)
    _ua_max = getattr(cfg, "health_delta_ua_max_v", DELTA_UA_MAX_V)
    _ug1_min = getattr(cfg, "health_delta_ug1_min_v", DELTA_UG1_MIN_V)
    _ug1_max = getattr(cfg, "health_delta_ug1_max_v", DELTA_UG1_MAX_V)
    _ug2_min = getattr(cfg, "health_delta_ug2_min_v", DELTA_UG2_MIN_V)
    _ug2_max = getattr(cfg, "health_delta_ug2_max_v", DELTA_UG2_MAX_V)
    delta_ua = float(srk_plan.get(
        "delta_ua", clamp_delta_ua(targets.target_ua * pct, _ua_min, _ua_max)))
    delta_ug1 = float(srk_plan.get(
        "delta_ug1", clamp_delta_ug1(abs(targets.target_ug1) * pct, _ug1_min, _ug1_max)))
    n_points = int(srk_plan.get("points", 5))
    n_repeats = int(srk_plan.get("repeats", cfg.health_ia_samples))
    n_ug1 = n_points - 2

    pct_ug2 = cfg.health_delta_ug2_pct / 100.0
    delta_ug2 = float(srk_plan.get(
        "delta_ug2", clamp_delta_ug2(targets.target_ug2 * pct_ug2, _ug2_min, _ug2_max)
    )) if targets.is_pentode_mode else 0.0

    # Shifted OP for R-phase if Ua ± δ exceeds device limits
    r_center, r_method = compute_shifted_r_center(targets.target_ua, delta_ua)

    # Shifted center for Sg2-phase if Ug2 ± δ exceeds device limits
    sg2_center, sg2_method = compute_shifted_sg2_center(
        targets.target_ug2, delta_ug2,
    ) if targets.is_pentode_mode else (targets.target_ug2, "central")

    ug1_sweep = _build_ug1_sweep(targets.target_ug1, delta_ug1, n_ug1)
    total_srk = len(ug1_sweep) + 2 + (2 if targets.is_pentode_mode else 0)

    s_ug1, s_ia, srk_idx = _measure_s_phase(
        client, cfg, calibration, hw, ug1_sweep, n_repeats,
        ia_op, targets.target_ug1, srk_idx=0, total_srk=total_srk,
        measurement_points=measurement_points, progress=progress, stop=stop)

    r_ua, r_ia, srk_idx = _measure_r_phase(
        client, lamp, cfg, calibration, hw,
        r_center, targets.target_ug1, targets.target_ug2,
        delta_ua, n_repeats, targets.ug2_track_ua, targets.ug2_offset,
        srk_idx=srk_idx, total_srk=total_srk,
        measurement_points=measurement_points, progress=progress, stop=stop)

    sg2_ug2: List[float] = []
    sg2_ia: List[float] = []
    if targets.is_pentode_mode:
        sg2_ug2, sg2_ia, srk_idx = _measure_sg2_phase(
            client, cfg, calibration, hw,
            targets.target_ua, sg2_center, delta_ug2, n_repeats,
            srk_idx=srk_idx, total_srk=total_srk,
            measurement_points=measurement_points, progress=progress, stop=stop)

    s_measured, r_measured, k_measured = compute_srk_direct(s_ug1, s_ia, r_ua, r_ia)
    sg2_measured = compute_sg2_direct(sg2_ug2, sg2_ia) if targets.is_pentode_mode else None
    mu_g1g2 = compute_mu_g1g2(s_measured, sg2_measured)

    srk_uncertainty = estimate_srk_uncertainty(
        s_measured, r_measured, delta_ua, delta_ug1, n_repeats,
        sigma_ua=UA_RESOLUTION_V, sigma_ug1=UG1_RESOLUTION_V, sigma_ia=IA_NOISE_200MA)
    sg2_rel = estimate_sg2_uncertainty(
        sg2_measured, delta_ug2, n_repeats,
        sigma_ug2=UG2_RESOLUTION_V, sigma_ia=IA_NOISE_200MA,
    ) if targets.is_pentode_mode and delta_ug2 > 0 else None

    # Skip the OP restore when cancelling — there is no point driving Ua/Ug2
    # back to operating voltage just before the caller's safe-zero. (Without
    # this, _restore_op would re-apply full OP during a Cancel/emergency.)
    if not (stop and stop()):
        _restore_op(
            client, lamp, cfg, calibration, hw,
            targets.target_ua, targets.ug2_track_ua, targets.ug2_offset, targets.target_ug2,
        )

    return {
        "s_measured": s_measured,
        "r_measured": r_measured,
        "k_measured": k_measured,
        "sg2_measured": sg2_measured,
        "mu_g1g2": mu_g1g2,
        "srk_uncertainty": srk_uncertainty,
        "sg2_rel": sg2_rel,
        "delta_ua": delta_ua,
        "delta_ug1": delta_ug1,
        "delta_ug2": delta_ug2,
        "n_points": n_points,
        "n_repeats": n_repeats,
        "r_center": r_center,
        "r_method": r_method,
        "sg2_center": sg2_center,
        "sg2_method": sg2_method,
    }


def _build_health_result(
    *,
    lamp: LampConfig,
    cfg,
    lamp_id: str,
    name: str,
    reference: Optional[Dict],
    reference_mode: str,
    emission_enabled: bool,
    warmup_s: int,
    targets: _HealthTargets,
    srk: Dict,
    scores: Dict,
    point_op: Dict,
    measurement_points: List[Dict],
    ia_op: float,
    ia100: Optional[float],
    ia80: Optional[float],
    uh80: Optional[float],
    stable_info: Optional[Dict],
    confidence: str,
    servo: Optional[Dict] = None,
    emission_extra: Optional[Dict] = None,
    ia_plan_ma: Optional[float] = None,
) -> Dict:
    """Assemble the final ``run_health_test`` result dictionary.

    *ia_plan_ma* — the current measured at the PLAN bias before a
    successful servo moved the OP; None when the servo did not move it
    (then ``ia_pct`` itself is the plan-point figure).
    """
    op_plan = targets.op_plan
    uh_nom = float(op_plan.get("uh", lamp.uh))
    servo = servo or {"status": BIAS_SERVO_DISABLED, "bias_shift_v": None,
                      "ug1": targets.target_ug1, "ia": ia_op,
                      "iterations": 0, "ref_ia": None}
    emission_extra = emission_extra or {}

    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "tube_type": lamp.tube_type,
        "lamp_id": lamp_id,
        "name": name or "HealthCheck",
        "topology": lamp.topology,
        "source": "health",
        "srk": {
            "s": srk["s_measured"],
            "r": srk["r_measured"],
            "k": srk["k_measured"],
            "sg2": srk["sg2_measured"],
            "mu_g1g2": srk["mu_g1g2"],
            "uncertainty": {
                **(srk["srk_uncertainty"] or {}),
                "sg2_rel": srk["sg2_rel"],
            },
        },
        "health": {
            "version": 1,
            "reference_mode": reference_mode,
            "reference_label": (reference or {}).get("label", "datasheet") if isinstance(reference, dict) else "datasheet",
            "emission_enabled": bool(emission_enabled),
            "confidence": confidence,
            "metrics": {
                "ia_pct": scores["ia_pct"],
                "s_pct": scores["s_pct"],
                "r_pct": scores["r_pct"],
                "k_pct": scores["k_pct"],
                "rh_score": scores["rh_score"],
                "screen_score": scores["screen_score"],
                "emission_ratio": scores["emission_ratio"],
                "emission_score": scores["emission_score"],
                "emission_ratio_ref": scores["emission_ratio_ref"],
                "emission_verdict": scores["emission_verdict"],
                "emission_sensitivity_ratio": scores["emission_sensitivity_ratio"],
                "emission_low_sensitivity": scores["emission_low_sensitivity"],
                "bias_shift_v": servo["bias_shift_v"],
                # Plan-point deficit: with the servo on, ia_pct sits at
                # ~100 by construction, so THIS is the wear figure the
                # old fixed-bias ia_pct used to carry.
                "ia_plan_pct": (
                    100.0 * float(ia_plan_ma) / float(servo["ref_ia"])
                    if (ia_plan_ma is not None
                        and isinstance(servo.get("ref_ia"), (int, float))
                        and float(servo["ref_ia"]) > EPS)
                    else None
                ),
                "emission_reserve_pct": emission_extra.get("reserve_pct"),
                "emission_knee_confidence": emission_extra.get(
                    "knee_confidence"),
            },
            "bias_servo": {
                "status": servo["status"],
                "ug1": servo["ug1"],
                "ia": servo["ia"],
                "ref_ia": servo.get("ref_ia"),
                "iterations": servo["iterations"],
                "plan_ug1": targets.op_plan.get("ug1", lamp.ug1),
                # The per-tube excursion limit actually applied — the
                # first thing to look at when the status is unreachable.
                "shift_limit_v": servo.get("shift_limit_v"),
            },
            "emission_sweep": {
                "mode": emission_extra.get("mode", EMISSION_MODE_SINGLE),
                "curve": emission_extra.get("curve"),
                "uh_knee": emission_extra.get("uh_knee"),
                "reserve_pct": emission_extra.get("reserve_pct"),
                "knee_below_range": emission_extra.get("knee_below_range", False),
                "knee_confidence": emission_extra.get("knee_confidence"),
                "plateau_slope_ma_per_v": emission_extra.get(
                    "plateau_slope_ma_per_v"),
                "truncated": emission_extra.get("sweep_truncated", False),
            },
            "index": scores["index"],
            "verdict": scores["verdict"],
            "stabilization_config": {
                "warmup_ratio": cfg.health_emission_stable_warmup_ratio,
                "min_s": cfg.health_emission_stable_min_s,
                "max_s": cfg.health_emission_stable_max_s,
                "slope_threshold_ma_per_s": cfg.health_emission_stable_slope_threshold_ma_per_s,
            },
            "raw": {
                "ia_op": ia_op,
                "ia_plan_ma": ia_plan_ma,
                "ia100": ia100,
                "ia80": ia80,
                "uh100": lamp.uh,
                "uh80": uh80,
                "uh80_stabilization": {
                    "warmup_s": warmup_s,
                    **{k: (stable_info or {}).get(k)
                       for k in ("t_max_s", "elapsed_s", "eta_s_last", "stable")},
                },
            },
        },
        "conditions": {
            "ua": targets.target_ua,
            # The PLAN bias, not the servo outcome: conditions describe the
            # shared measurement protocol — that is what makes two lamps'
            # runs comparable for matching. The bias the servo actually
            # settled on is per-tube data and lives in health.bias_servo.ug1.
            "ug1": float(op_plan.get("ug1", lamp.ug1)),
            "ug2": targets.target_ug2,
            # Servo runs sit at the reference current, not at the plan-bias
            # current — they must never be matched against fixed-bias runs.
            "bias_servo": servo["status"] == BIAS_SERVO_OK,
            # uh/ih are the NOMINAL (target) heater, like ua/ug1/ug2. The heater
            # ACTUALLY applied is already in the OP measurement point
            # (measurement_points[0]["uh"]/["ih"], read back from the device);
            # a stuck-reduced heater is detectable as point_op.uh != uh, so it
            # is not duplicated here. (ih nominal was previously missing — only
            # uh was stored; added for symmetry.)
            "uh": uh_nom if emission_enabled else float(op_plan.get("uh", lamp.uh)),
            "ih": float(op_plan.get("ih", lamp.ih)),
            "an": targets.an,
            "ug2_track_ua": targets.ug2_track_ua,
            "ug2_offset": targets.ug2_offset,
            "ug2_mode": _ug2_mode_for(lamp, targets),
        },
        "measurement_plan": {
            "an": targets.an,
            "ug2_track_ua": targets.ug2_track_ua,
            "ug2_offset": targets.ug2_offset,
            "op": {
                "ua": targets.target_ua,
                # The plan, not the servo outcome — re-applying this plan
                # must reproduce the protocol, not freeze one tube's bias.
                "ug1": float(op_plan.get("ug1", lamp.ug1)),
                "ug2": targets.target_ug2,
                "uh": float(op_plan.get("uh", lamp.uh)),
                "ih": float(op_plan.get("ih", lamp.ih)),
            },
            "srk": {
                "delta_ua": srk["delta_ua"],
                "delta_ug1": srk["delta_ug1"],
                "delta_ug2": srk["delta_ug2"] if targets.is_pentode_mode else None,
                "points": srk["n_points"],
                "repeats": srk["n_repeats"],
                "r_center": srk["r_center"],
                "r_method": srk["r_method"],
                "sg2_center": srk["sg2_center"] if targets.is_pentode_mode else None,
                "sg2_method": srk["sg2_method"] if targets.is_pentode_mode else None,
            },
            "emission": {
                "enabled": bool(emission_enabled),
                "uh_ratio": float(targets.emission_plan.get(
                    "uh_ratio", cfg.health_emission_uh_ratio,
                )),
                # The requested mode (resolved like uh_ratio), not what ran:
                # with emission disabled nothing runs, but the plan choice
                # must still round-trip.
                "mode": str(targets.emission_plan.get(
                    "mode", cfg.health_emission_mode_default)),
            },
            "bias_servo": {
                "enabled": bool(targets.bias_servo_enabled),
            },
        },
        "measurement_points": measurement_points,
    }


# ── Orchestrator ─────────────────────────────────────────────────────

def run_health_test(
    client: LM19Serial,
    lamp: LampConfig,
    cfg: HealthConfig,
    calibration: CalibrationData,
    lamp_id: str,
    name: str,
    reference_mode: str,
    reference: Optional[Dict] = None,
    emission_enabled: Optional[bool] = None,
    measurement_plan: Optional[Dict] = None,
    warmup_s: Optional[int] = None,
    progress: Optional[Callable[[HealthProgress], None]] = None,
    stop: Optional[Callable[[], bool]] = None,
) -> Dict:
    if emission_enabled is None:
        emission_enabled = bool(cfg.health_emission_enabled_default)
    warmup_s = int(lamp.warmup_s if warmup_s is None else warmup_s)

    refs = _extract_refs(reference, lamp)
    targets = _parse_health_targets(measurement_plan, lamp, cfg)

    an_sync = _set_an_with_readback(client, targets.an)
    if progress:
        progress({"event": "anode_sync", **an_sync})

    # prev_* live in the PHYSICAL domain (plan B): the adapter converts to
    # the command domain internally, so the initial reads must be calibrated.
    hw = _HwState(
        prev_ua=calibration.apply_read(
            "ua", float(client.get_param("Ua", real=True))),
        prev_ug1=calibration.apply_read(
            "ug1", float(decode_ug1(client.get_param("Ug1", real=True)))),
        prev_ug2=calibration.apply_read(
            "ug2", float(client.get_param("Ug2", real=True))),
        prev_uh=calibration.apply_read(
            "uh", decode_uh(client.get_param("Uh", real=True))),
    )

    # Phase 1: Setup OP (with safe-lock + Ug1 ramp + Pa/Pg2 protection)
    ug2_mode = _ug2_mode_for(lamp, targets)
    point_op = _setup_op(
        client, cfg, calibration, hw, lamp,
        targets.target_ua, targets.target_ug1, targets.target_ug2,
        ug2_mode=ug2_mode, lamp_id=lamp_id,
        progress=progress, stop=stop,
    )
    measurement_points: List[Dict] = [dict(point_op, step="op")]
    ia_op = float(point_op.get("ia", 0.0))

    # Phase 1b: bias servo — bring Ia to the reference current so the SRK
    # phase measures S at the same point of the curve as the reference.
    # The plan-bias current is captured FIRST: once the OP moves, ia_pct
    # degenerates to ~100 by construction and this number becomes the
    # only record of the deficit the servo compensated.
    ia_at_plan = ia_op
    servo = _servo_ug1_to_ref_ia(
        client, cfg, calibration, lamp, hw,
        ref_ia=refs["ia"], target_ug1=targets.target_ug1,
        ia_at_target=ia_op, point_at_target=point_op,
        ug2_mode=ug2_mode, lamp_id=lamp_id,
        enabled=targets.bias_servo_enabled,
        measurement_points=measurement_points,
        progress=progress, stop=stop,
    )
    if servo["status"] == BIAS_SERVO_OK and servo["point"] is not point_op:
        # The OP moved: every downstream consumer (Ia%, SRK centre, Pa,
        # emission sensitivity) must see where the tube actually sits.
        point_op = servo["point"]
        ia_op = float(point_op.get("ia", ia_op))
        targets = dataclass_replace(targets, target_ug1=float(servo["ug1"]))

    # Phase 2: SRK measurement (S/R/Sg2 + uncertainty + restore_op)
    srk = _run_srk_phase(
        client, cfg, calibration, lamp, hw, targets,
        ia_op, measurement_points, progress, stop,
    )
    # The SRK phases break (not raise) on stop, leaving partial S/R/K. Abort
    # here so a stopped test never produces a "complete" result that could be
    # saved as a personal baseline (consistent with _run_stabilized_ia80's
    # raise; HealthWorker.run suppresses the failed signal on a user stop).
    if stop and stop():
        raise RuntimeError("Health test stopped")

    # Phase 3: Emission test (optional)
    ia100 = ia80 = uh80 = stable_info = None
    confidence = "ok"
    emission_extra: Optional[Dict] = None
    if emission_enabled:
        em = _run_emission(
            client, cfg, calibration, hw, lamp,
            targets.op_plan, targets.emission_plan,
            warmup_s, measurement_points, progress, stop,
        )
        ia100 = em["ia100"]
        ia80 = em["ia80"]
        uh80 = em["uh80"]
        stable_info = em["stable_info"]
        confidence = em["confidence"]
        emission_extra = em

    # Phase 4: Scoring
    scores = _compute_scores(
        ia_op, srk["s_measured"], srk["r_measured"], srk["k_measured"],
        point_op, lamp, cfg, refs, emission_enabled, ia80, ia100,
    )

    return _build_health_result(
        lamp=lamp, cfg=cfg, lamp_id=lamp_id, name=name,
        reference=reference, reference_mode=reference_mode,
        emission_enabled=emission_enabled, warmup_s=warmup_s,
        targets=targets, srk=srk, scores=scores,
        point_op=point_op, measurement_points=measurement_points,
        ia_op=ia_op, ia100=ia100, ia80=ia80, uh80=uh80,
        stable_info=stable_info, confidence=confidence,
        servo=servo, emission_extra=emission_extra,
        ia_plan_ma=ia_at_plan if servo["status"] == BIAS_SERVO_OK else None,
    )
