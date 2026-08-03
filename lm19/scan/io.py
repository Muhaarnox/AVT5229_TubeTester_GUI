"""Hardware I/O helpers for scan: settle/verify + measurement read.

These functions wrap the raw protocol layer with retry / verification /
calibration application. Used by both ``run_scan`` (in runner.py) and
``measure_srk`` (in srk.py).
"""

from __future__ import annotations

import logging
import time
from statistics import median
from typing import Callable, Dict, List, Optional, Tuple

from serial import SerialException

log = logging.getLogger(__name__)

from lm19.calibration import CalibrationData
from lm19.protocol import (
    LM19Serial,
    decode_ia,
    decode_ig2,
    decode_ih,
    decode_ug1,
    decode_uh,
)
from lm19.scan.exceptions import ProtectionError


_IA_OUTLIER_FLOOR = 0.5  # mA — outlier check needs one sample above this
_SETTLE_CHUNK_S = 0.05   # interruptible-sleep granularity for settle waits

# Sample-count thresholds of _robust_average
_TRIMMED_MEAN_MIN_N = 5  # drop min+max, average the rest
_MEDIAN_MIN_N = 3        # below this a plain mean is all the data allows

# MAD rejection of Ia samples, applied only after an outlier was detected
_MAD_REJECT_K = 3.0     # distance from the median, in MADs
_MAD_FLOOR_MA = 0.05    # ~2.5 LSB of the 20 mA range — tolerance floor for
                        # MAD = 0 (identical samples) and quantization noise
_MIN_KEPT_SAMPLES = 3   # never average fewer than this many survivors
_TOL_REL_EPS = 1e-9     # boundary padding against float round-off


def _interruptible_sleep(
    duration_s: float, stop: Optional[Callable[[], bool]]
) -> bool:
    """Sleep ``duration_s`` in small chunks, returning ``True`` as soon as
    ``stop()`` becomes true (without sleeping the remainder); ``False`` if the
    full duration elapsed. With ``stop=None`` it is a plain blocking sleep.

    Failure-visibility principle 3: worker-thread waits must be cancellable so Cancel
    is responsive even during a multi-second settle (SRK Ug1 can reach ~28 s).
    """
    if stop is None:
        if duration_s > 0:
            time.sleep(duration_s)
        return False
    if stop():
        return True
    slept = 0.0
    while slept < duration_s:
        chunk = min(_SETTLE_CHUNK_S, duration_s - slept)
        time.sleep(chunk)
        slept += chunk
        if stop():
            return True
    return False


def _set_param_with_settle(
    client: LM19Serial,
    name: str,
    target: float,
    prev_value: float,
    settle_per_volt_s: float,
    settle_base_s: float,
    tolerance: float,
    max_retries: int,
    encode_fn: Optional[Callable[[float], int]] = None,
    decode_fn: Optional[Callable[[int], float]] = None,
    *,
    verify_target: Optional[float] = None,
    stop: Optional[Callable[[], bool]] = None,
    stats: Optional[Dict[str, int]] = None,
) -> float:
    """Set a parameter with dynamic settle time and verification.

    Settle time = |target - prev_value| * settle_per_volt_s + settle_base_s.
    After settle, reads back the real value and retries if outside tolerance.
    Returns the actual (decoded) value read from the device.

    The pipe is raw — no calibration inside. *verify_target* is the
    device-domain reading expected for the commanded target (plan B:
    callers pass ``calibration.read_inverse(target_phys)``); ``None``
    compares the reading against *target* itself. The setpoint
    protection check always uses *target*: the setpoint never passes
    through the ADC, so the command domain is the correct one there.
    """
    expected = target if verify_target is None else verify_target
    if stop is not None and stop():
        # Already cancelled before the setpoint is issued — do NOT write a new
        # value. If an emergency zero just ran, re-issuing the setpoint here
        # would re-energize the output. Return the device-domain expected value
        # (the worker discards it at its next boundary; see the settle path).
        return expected
    raw = encode_fn(target) if encode_fn else int(round(target))
    client.set_param(name, raw)
    settle = abs(target - prev_value) * settle_per_volt_s + settle_base_s
    if _interruptible_sleep(settle, stop):
        # Cancelled — return the device-domain *expected* value (not the
        # command-domain target): the calibrated wrapper feeds it through
        # apply_read, which then round-trips to the physical target, so a
        # cancelled SRK set does not trip a spurious SrkVerifyError. The
        # worker exits at its next boundary and discards the value anyway.
        return expected
    # Verify with retries
    actual = target  # fallback if no retries
    for attempt in range(max(max_retries, 1)):
        raw_read = client.get_param(name, real=True)
        actual = decode_fn(raw_read) if decode_fn else float(raw_read)
        if abs(actual - expected) <= tolerance:
            return actual
        # Retry: re-send and wait base settle
        if attempt < max_retries - 1:
            if stop is not None and stop():
                # Cancelled — do NOT re-assert the setpoint. The verify-retry
                # re-send is the path that could re-energize an output after an
                # emergency zero (readback ~0 != target → resend target).
                return expected
            client.set_param(name, raw)
            if _interruptible_sleep(settle_base_s, stop):
                return expected  # cancelled mid-retry (device-domain, see above)

    # Settle failed — check if firmware zeroed the setpoint (protection).
    # OVERIA zeroes uaset/ug2set, OVERIH zeroes uhset/ihset.
    if abs(target) > tolerance:
        try:
            sp_raw = client.get_param(name, real=False)
            sp = decode_fn(sp_raw) if decode_fn else float(sp_raw)
        except (ValueError, RuntimeError, SerialException):
            sp = None
        if sp is not None and abs(sp) < tolerance:
            raise ProtectionError(
                f"Device protection active — {name}: "
                f"target={target:.1f}, setpoint={sp:.1f}, actual={actual:.1f}"
            )
    # ML-108: out-of-tolerance after exhausting retries (and no protection
    # trip) — the point is measured OFF the requested operating point.
    # Callers thread ``stats`` so the scan summary can show the count.
    log.warning("%s failed to settle after %d retries: target=%.2f, "
                "expected=%.2f, actual=%.2f (tolerance %.2f)",
                name, max_retries, target, expected, actual, tolerance)
    if stats is not None:
        stats["settle_out_of_tolerance"] = (
            stats.get("settle_out_of_tolerance", 0) + 1)
    return actual


def _set_param_calibrated(
    client: LM19Serial,
    name: str,
    channel: str,
    target_phys: float,
    prev_phys: float,
    calibration: Optional[CalibrationData],
    *,
    settle_per_volt_s: float,
    settle_base_s: float,
    tolerance: float,
    max_retries: int,
    encode_fn: Optional[Callable[[float], int]] = None,
    decode_fn: Optional[Callable[[int], float]] = None,
    stop: Optional[Callable[[], bool]] = None,
    stats: Optional[Dict[str, int]] = None,
) -> float:
    """Feedforward calibration adapter around ``_set_param_with_settle``.

    Plan B (docs/CALIBRATION_PLAN.md §2): the caller domain is physical,
    the pipe stays raw; all conversions happen here, once per call:

        cmd      = apply_set(channel, target_phys)    — pre-corrected command
        expected = read_inverse(channel, target_phys) — verify target
        returns    apply_read(channel, decoded actual) — physical actual

    ``calibration=None`` and default coefficients degrade to the exact
    raw behaviour. *tolerance* is given in physical units but compared
    in the device-reading domain — the distortion equals the READ gain
    (within a few percent), an accepted approximation.
    """
    if calibration is None:
        return _set_param_with_settle(
            client, name, target_phys, prev_phys,
            settle_per_volt_s, settle_base_s, tolerance, max_retries,
            encode_fn=encode_fn, decode_fn=decode_fn, stop=stop, stats=stats,
        )
    cmd = calibration.apply_set(channel, target_phys)
    # Settle-time delta only — unclamped linear transform, no warning noise
    prev_cmd = calibration.get_channel(channel, "set").apply(prev_phys)
    expected = calibration.read_inverse(channel, target_phys)
    actual_dev = _set_param_with_settle(
        client, name, cmd, prev_cmd,
        settle_per_volt_s, settle_base_s, tolerance, max_retries,
        encode_fn=encode_fn, decode_fn=decode_fn,
        verify_target=expected, stop=stop, stats=stats,
    )
    return calibration.apply_read(channel, actual_dev)


def _robust_average(values: List[float]) -> float:
    """Trimmed mean for >= 5 values, median for 3–4, mean for 1–2."""
    n = len(values)
    if n == 0:
        return 0.0
    if n < _MEDIAN_MIN_N:
        return sum(values) / n
    s = sorted(values)
    if n >= _TRIMMED_MEAN_MIN_N:
        trimmed = s[1:-1]
        return sum(trimmed) / len(trimmed)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _ia_spread(values: List[float]) -> float:
    """max/min of *values*, or 0.0 when the check is not armed.

    Armed by the LARGEST sample: a reading that collapses towards zero
    (intermittent contact) is the loudest instability signal there is, so
    gating on the smallest sample would silence exactly the severe case.
    Below the floor the whole point is near-zero noise, where max/min
    carries no information. A sample of exactly zero leaves the ratio
    undefined, which counts as an outlier outright.

    Shared by the detection pass and the post-rejection control pass, so
    the two can never judge the same samples by different rules.
    """
    if not values:
        return 0.0
    lo, hi = min(values), max(values)
    if hi <= _IA_OUTLIER_FLOOR:
        return 0.0
    return hi / lo if lo > 0 else float("inf")


def _reject_outlier_indices(
    values: List[float],
) -> Tuple[List[int], List[int]]:
    """Split sample indices into kept / rejected by MAD distance.

    Scale-adaptive by construction — the tolerance follows the scatter of
    the samples themselves, so one rule covers 0.5 mA and 150 mA alike,
    unlike a ratio (meaningless near zero) or an absolute band (blind at
    high currents). Identical samples give MAD = 0, where the floor keeps
    the rule from rejecting everything but the median.

    A majority always survives, by construction: at least half of the
    deviations are <= MAD (that is what a median is), and the tolerance
    is at least ``_MAD_REJECT_K`` MADs, so the rule can never strip the
    pool down to a couple of samples. Pools of ``_MIN_KEPT_SAMPLES`` or
    fewer are returned untouched — there the caller uses the median.
    """
    n = len(values)
    if n <= _MIN_KEPT_SAMPLES:
        return list(range(n)), []
    med = median(values)
    tol = max(_MAD_REJECT_K * median([abs(v - med) for v in values]),
              _MAD_FLOOR_MA)
    # Pad the boundary against float round-off: a sample sitting exactly
    # at k*MAD must not be kept or rejected by the last bit of |v - med|.
    limit = tol * (1.0 + _TOL_REL_EPS)
    kept = [i for i, v in enumerate(values) if abs(v - med) <= limit]
    kept_set = set(kept)
    return kept, [i for i in range(n) if i not in kept_set]


def _read_measurement_point(
    client: LM19Serial,
    calibration: CalibrationData,
    ia_samples: int = 1,
    sample_delay_s: float = 0.0,
    ia_outlier_ratio: float = 0.0,
    ia_outlier_reread_samples: int = 3,
    stats: Optional[Dict[str, int]] = None,
) -> Dict:
    """Read all real parameters from device and return a measurement point dict.

    If ia_samples > 1, Ia and Ig2 are read multiple times.
    Calibration corrections are applied to all channels.

    When *ia_outlier_ratio* > 0 and ia_samples >= 3, an outlier check is
    performed on Ia samples (see ``_ia_spread``). On a hit,
    *ia_outlier_reread_samples* extra samples are read, the pool is
    cleaned by ``_reject_outlier_indices``, and Ia/Ig2 become the mean of
    the survivors — Ig2 follows the sample positions Ia rejected, since a
    transient corrupts the whole sample, not one channel of it. With
    nothing to reject the median of the pool is used instead.

    A control pass then re-measures the spread of the survivors: a point
    that stays beyond the threshold is unstable rather than spiky, which
    no averaging rule can repair, so it is logged and counted in
    ``stats["ia_unstable_points"]``.

    ``ia_outlier_reread_samples=0`` keeps the warning and the counters but
    skips the extra batch — diagnostics without the extra read time.

    Points without a detected outlier keep the plain robust average, and
    other channels always use the arithmetic mean over all samples.
    """
    n = max(ia_samples, 1)
    ia_vals: List[float] = []
    ig2_vals: List[float] = []
    ua_sum = 0.0
    ug1_sum = 0.0
    ug2_sum = 0.0
    uh_sum = 0.0
    ih_sum = 0.0

    def _read_one(delay: bool = False) -> None:
        nonlocal ua_sum, ug1_sum, ug2_sum, uh_sum, ih_sum
        if delay and sample_delay_s > 0:
            time.sleep(sample_delay_s)
        ia_vals.append(decode_ia(client.get_param("Ia", real=True)))
        ig2_vals.append(decode_ig2(client.get_param("Ig2", real=True)))
        ua_sum += float(client.get_param("Ua", real=True))
        ug1_sum += decode_ug1(client.get_param("Ug1", real=True))
        ug2_sum += float(client.get_param("Ug2", real=True))
        uh_sum += decode_uh(client.get_param("Uh", real=True))
        ih_sum += decode_ih(client.get_param("Ih", real=True))

    for i in range(n):
        _read_one(delay=(i > 0))

    outlier_detected = False
    if ia_outlier_ratio > 0 and n >= _MEDIAN_MIN_N:
        ia_min = min(ia_vals)
        ia_max = max(ia_vals)
        spread = _ia_spread(ia_vals)
        if spread > ia_outlier_ratio:
            outlier_detected = True
            # ML-109: an unstable Ia reading is a data-quality signal
            # (bad contact, unstable tube) — never re-read silently.
            # The operating point comes from the same samples and stays in
            # the raw decoded domain the check itself ran in, so the line
            # and the decision describe the same numbers. Without it a
            # warning cannot be placed on the curve it came from.
            reread = max(ia_outlier_reread_samples, 0)
            log.warning("Ia outlier at Ua=%.1f V, Ug1=%.2f V, Ug2=%.1f V:"
                        " spread %.3f–%.3f mA (ratio %.2f > %.2f)"
                        " — re-reading %d extra samples, robust averaging",
                        ua_sum / n, ug1_sum / n, ug2_sum / n,
                        ia_min, ia_max, spread, ia_outlier_ratio, reread)
            if stats is not None:
                stats["ia_outlier_rereads"] = (
                    stats.get("ia_outlier_rereads", 0) + 1)
            for i in range(reread):
                _read_one(delay=True)

    total = len(ia_vals)
    if outlier_detected:
        kept, rejected = _reject_outlier_indices(ia_vals)
        if rejected:
            ia_avg = sum(ia_vals[i] for i in kept) / len(kept)
            ig2_avg = sum(ig2_vals[i] for i in kept) / len(kept)
            method = f"mean of {len(kept)} kept"
        else:
            # Nothing stood out from the rest: the spike is still in the
            # set (that is why we are here), so the mean would carry it —
            # the median is the only safe estimator left.
            ia_avg = median(ia_vals)
            ig2_avg = median(ig2_vals)
            method = f"median of {total}"
    elif n >= _MEDIAN_MIN_N:
        kept, rejected = list(range(total)), []
        ia_avg = _robust_average(ia_vals)
        ig2_avg = _robust_average(ig2_vals)
    else:
        kept, rejected = list(range(total)), []
        ia_avg = sum(ia_vals) / total
        ig2_avg = sum(ig2_vals) / total

    if outlier_detected:
        # Resolution of the warning above: the whole pool with rejected
        # samples in brackets, and the value that ends up in the point.
        rejected_set = set(rejected)
        pool = ", ".join(f"[{v:.3f}]" if i in rejected_set else f"{v:.3f}"
                         for i, v in enumerate(ia_vals))
        log.warning("Ia outlier resolved: samples [%s] mA (brackets rejected),"
                    " %s → Ia=%.3f mA, Ig2=%.3f mA",
                    pool, method, ia_avg, ig2_avg)
        # Second control pass: rejection removes single spikes, but a
        # point that is simply unstable keeps its spread afterwards, and
        # no averaging rule can fix that — say so instead of shipping a
        # confident-looking number.
        # Only meaningful when rejection had a chance at all: on a pool
        # too small to reject from, "still unstable" would just restate
        # the detection above.
        residual = _ia_spread([ia_vals[i] for i in kept])
        if total > _MIN_KEPT_SAMPLES and residual > ia_outlier_ratio:
            log.warning("Ia still unstable after re-read at Ua=%.1f V,"
                        " Ug1=%.2f V, Ug2=%.1f V: kept %.3f–%.3f mA"
                        " (ratio %.2f > %.2f) — rejection could not clean"
                        " it up", ua_sum / total, ug1_sum / total,
                        ug2_sum / total,
                        min(ia_vals[i] for i in kept),
                        max(ia_vals[i] for i in kept),
                        residual, ia_outlier_ratio)
            if stats is not None:
                stats["ia_unstable_points"] = (
                    stats.get("ia_unstable_points", 0) + 1)

    n_all = total if outlier_detected else n

    return {
        "ua": calibration.apply_read("ua", ua_sum / n_all),
        "ug1": calibration.apply_read("ug1", ug1_sum / n_all),
        "ug2": calibration.apply_read("ug2", ug2_sum / n_all),
        "ia": calibration.apply_read("ia", ia_avg),
        "ig2": calibration.apply_read("ig2", ig2_avg),
        "uh": calibration.apply_read("uh", uh_sum / n_all),
        "ih": calibration.apply_read("ih", ih_sum / n_all),
        "er": client.get_param("Er", real=False),
    }


def _try_reopen(client: LM19Serial) -> None:
    """Attempt to reopen the serial port after USB-Serial adapter reconnect."""
    try:
        client.reopen()
        log.info("Serial port reopened successfully")
    except Exception as exc:
        log.warning("Failed to reopen serial port: %s", exc)
