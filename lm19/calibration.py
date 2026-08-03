"""Software calibration for LM19 tube tester measurement channels.

Each channel has a linear correction: corrected = raw * gain + offset.
READ channels correct measurements from the device.
SET channels correct commands sent to the device.

Default coefficients (gain=1.0, offset=0.0) are transparent — no correction.

Ia has two hardware ranges (20mA / 200mA) with auto-switching at ~17 mA.
Each range is calibrated independently: ia_low_read / ia_high_read.
The external API (apply_read("ia", x)) selects the correct range automatically.
"""


from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from lm19.constants import IA_HW_SCALE
from lm19.io_utils import write_json

log = logging.getLogger(__name__)

# IA_HW_SCALE imported from lm19.constants

# Firmware auto-switches Ia range at ~17 mA (ADC 950/1023 on 20mA,
# hysteresis down at ADC 85/1023 on 200mA ≈ 16.6 mA).
IA_RANGE_THRESHOLD = 17.0  # mA — boundary for selecting calibration coefficients

# Denominator guard for two-point fits and gain inversion
_EPS_DENOM = 1e-12

# Calibration file schema version. The plan B ug1 sign-domain change
# landed before any release, so no migration ladder exists yet (project
# convention: create it with the first real migration, not speculatively).
CALIBRATION_FILE_VERSION = 2

ALL_READ_CHANNELS = ("ua", "ug1", "ug2", "uh", "ih", "ia", "ig2")
ALL_SET_CHANNELS = ("ua", "ug1", "ug2", "uh", "ih")

# Internal channel keys that map to calibration.json entries
IA_RANGE_CHANNELS = ("ia_low", "ia_high")

CHANNEL_UNITS = {
    "ua": "V", "ug1": "V", "ug2": "V", "uh": "V",
    "ih": "A", "ia": "mA", "ia_low": "mA", "ia_high": "mA", "ig2": "mA",
}

# Default meter accuracy (±% of reading) per channel
DEFAULT_METER_ACCURACY_PCT: Dict[str, float] = {
    "ua": 0.5, "ug1": 0.5, "ug2": 0.5, "uh": 0.5,
    "ih": 1.0, "ia_low": 1.0, "ia_high": 1.0, "ig2": 1.0,
}

# Hardware limits for SET clamping (physical units).
# Ug1 uses the canonical negative physical domain (plan B,
# docs/CALIBRATION_PLAN.md §4): clamping a negative bias toward 0 V
# would mean maximum anode current — the dangerous direction.
SET_LIMITS = {
    "ua": (0.0, 300.0),
    "ug1": (-24.0, 0.0),
    "ug2": (0.0, 300.0),
    "uh": (0.0, 15.0),
    "ih": (0.0, 2.5),
}

# Gain sanity bounds per channel (for UI validation)
GAIN_BOUNDS = {
    "default": (0.80, 1.20),
}

# Offset sanity bounds per channel (physical units, for UI validation)
OFFSET_BOUNDS = {
    "ua": (-10.0, 10.0),
    "ug1": (-1.0, 1.0),
    "ug2": (-10.0, 10.0),
    "uh": (-1.0, 1.0),
    "ih": (-0.1, 0.1),
    "ia": (-1.0, 1.0),
    "ig2": (-1.0, 1.0),
}


def fit_within_bounds(channel: str, gain: float, offset: float) -> bool:
    """Sanity-check a fitted (gain, offset) pair against GAIN/OFFSET_BOUNDS.

    An out-of-bounds fit almost always means a meter unit or sign error
    (mV vs V, ×10 range). With plan B feedforward a stored bad fit drives
    real commands — e.g. a ug1 SET gain of 0.1 would bias the grid toward
    0 V (maximum anode current) — so callers must refuse such fits loudly
    instead of storing them. Unknown channel → KeyError (programming
    error, surfaces per failure-visibility principle 1).
    """
    gain_lo, gain_hi = GAIN_BOUNDS["default"]
    base = channel.replace("_low", "").replace("_high", "")
    off_lo, off_hi = OFFSET_BOUNDS[base]
    return gain_lo <= gain <= gain_hi and off_lo <= offset <= off_hi


@dataclass
class ChannelCal:
    gain: float = 1.0
    offset: float = 0.0
    calibrated_at: Optional[str] = None
    quality: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def is_default(self) -> bool:
        return self.gain == 1.0 and self.offset == 0.0

    def apply(self, value: float) -> float:
        return value * self.gain + self.offset


def _channel_key(channel: str, direction: str) -> str:
    return f"{channel}_{direction}"


def _default_meter_accuracy() -> Dict[str, float]:
    return dict(DEFAULT_METER_ACCURACY_PCT)


def _default_channels() -> Dict[str, ChannelCal]:
    channels: Dict[str, ChannelCal] = {}
    for ch in ALL_READ_CHANNELS:
        if ch == "ia":
            for ia_ch in IA_RANGE_CHANNELS:
                channels[_channel_key(ia_ch, "read")] = ChannelCal()
        else:
            channels[_channel_key(ch, "read")] = ChannelCal()
    for ch in ALL_SET_CHANNELS:
        channels[_channel_key(ch, "set")] = ChannelCal()
    return channels


class CalibrationData:
    """Manages per-channel calibration coefficients with load/save/apply."""

    def __init__(
        self,
        channels: Optional[Dict[str, ChannelCal]] = None,
        meter_accuracy_pct: Optional[Dict[str, float]] = None,
    ):
        self.channels: Dict[str, ChannelCal] = channels or _default_channels()
        self.meter_accuracy_pct: Dict[str, float] = (
            meter_accuracy_pct or _default_meter_accuracy()
        )
        self._dirty = False

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_clean(self) -> None:
        self._dirty = False

    # ── Apply corrections ────────────────────────────────────────────

    def apply_read(self, channel: str, decoded_value: float) -> float:
        if channel == "ia":
            ia_ch = "ia_low" if decoded_value < IA_RANGE_THRESHOLD else "ia_high"
            key = _channel_key(ia_ch, "read")
        else:
            key = _channel_key(channel, "read")
        ch = self.channels.get(key)
        if ch is None:
            return decoded_value
        return ch.apply(decoded_value)

    def meter_error(self, channel: str, reading: float) -> float:
        """Absolute meter error for a given reading based on ±% accuracy."""
        pct = self.meter_accuracy_pct.get(channel, 1.0)
        return abs(reading) * pct / 100.0

    def read_inverse(self, channel: str, value: float) -> float:
        """Inverse of apply_read: device-domain reading for a physical value.

        Used as ``verify_target`` in the feedforward settle flow — the
        reading the device is expected to report once the physical value
        equals the target. Currents are never commanded and their range
        selection is ambiguous in the physical domain, so "ia" is rejected.
        """
        if channel == "ia":
            raise ValueError(
                "read_inverse is undefined for 'ia' — currents are not commanded")
        key = _channel_key(channel, "read")
        ch = self.channels.get(key)
        if ch is None:
            return value
        if abs(ch.gain) < _EPS_DENOM:
            raise ValueError(f"READ gain for '{channel}' is effectively zero")
        return (value - ch.offset) / ch.gain

    def derive_set_two_point(
        self, channel: str,
        commanded_low: float, dev_low: float,
        commanded_high: float, dev_high: float,
    ) -> Tuple[float, float]:
        """Derive SET coefficients from (commanded, device reading) pairs.

        Plan B: the multimeter is only needed for the READ calibration.
        The SET correction is the inverse of the DAC transfer observed
        through the calibrated READ channel:
        ``actual_phys = apply_read(device reading)``.
        Requires a current READ calibration for *channel* — a stale READ
        makes the derived SET wrong in exactly the same proportion.
        """
        actual_low = self.apply_read(channel, dev_low)
        actual_high = self.apply_read(channel, dev_high)
        return self.compute_set_two_point(
            commanded_low, actual_low, commanded_high, actual_high)

    def apply_set(self, channel: str, desired_value: float) -> float:
        key = _channel_key(channel, "set")
        ch = self.channels.get(key)
        if ch is None:
            return desired_value
        corrected = ch.apply(desired_value)
        limits = SET_LIMITS.get(channel)
        if limits:
            lo, hi = limits
            if corrected < lo or corrected > hi:
                log.warning(
                    "SET calibration for %s clamped: %.3f → [%.1f, %.1f]",
                    channel, corrected, lo, hi,
                )
                corrected = max(lo, min(hi, corrected))
        return corrected

    def get_channel(self, channel: str, direction: str) -> ChannelCal:
        key = _channel_key(channel, direction)
        ch = self.channels.get(key)
        if ch is None:
            ch = ChannelCal()
            self.channels[key] = ch
        return ch

    def set_channel(self, channel: str, direction: str,
                    gain: float, offset: float,
                    timestamp: Optional[str] = None,
                    quality: Optional[Dict[str, Any]] = None) -> None:
        key = _channel_key(channel, direction)
        ts = timestamp or datetime.now().isoformat(timespec="seconds")
        self.channels[key] = ChannelCal(
            gain=gain, offset=offset, calibrated_at=ts, quality=quality,
        )
        self._dirty = True

    def reset_channel(self, channel: str, direction: str) -> None:
        key = _channel_key(channel, direction)
        self.channels[key] = ChannelCal()
        self._dirty = True

    def reset_all(self) -> None:
        self.channels = _default_channels()
        self._dirty = True

    # ── Two-point calibration math ───────────────────────────────────

    @staticmethod
    def compute_two_point(
        ref_low: float, ref_high: float,
        dev_low: float, dev_high: float,
    ) -> Tuple[float, float]:
        """Compute gain+offset from two reference/device pairs.

        Model: ref = dev * gain + offset
        Returns (gain, offset).
        """
        if abs(dev_high - dev_low) < _EPS_DENOM:
            raise ValueError("Device readings at two points are identical")
        gain = (ref_high - ref_low) / (dev_high - dev_low)
        offset = ref_low - dev_low * gain
        return gain, offset

    @staticmethod
    def compute_current_two_point(
        resistor_ohm: float,
        voltage_low: float, voltage_high: float,
        dev_ia_low: float, dev_ia_high: float,
    ) -> Tuple[float, float]:
        """Compute Ia/Ig2 gain+offset using a known resistor.

        voltage_low/high — actual voltage across the resistor (from
        multimeter or calibrated Ua/Ug2 reading).
        dev_ia_low/high — device Ia/Ig2 readings (mA) at those voltages.

        Expected current (mA) = voltage (V) / resistance (Ω) * 1000.
        """
        ref_low = voltage_low / resistor_ohm * 1000.0
        ref_high = voltage_high / resistor_ohm * 1000.0
        return CalibrationData.compute_two_point(
            ref_low, ref_high, dev_ia_low, dev_ia_high,
        )

    @staticmethod
    def compute_zero_offset(zero_reading: float) -> float:
        """Compute offset from a zero reading (no signal).

        If device reads X when it should read 0, offset = -X.
        """
        return -zero_reading

    # ── SET calibration math ─────────────────────────────────────────

    @staticmethod
    def compute_set_two_point(
        commanded_low: float, actual_low: float,
        commanded_high: float, actual_high: float,
    ) -> Tuple[float, float]:
        """Compute SET gain+offset from two commanded/actual pairs.

        We want: corrected_command = desired * gain + offset
        such that the hardware produces 'desired' when we send 'corrected_command'.

        Given: actual = commanded * hw_gain + hw_offset  (hardware model)
        We need: commanded = (desired - hw_offset) / hw_gain = desired/hw_gain - hw_offset/hw_gain
        So: gain = 1/hw_gain, offset = -hw_offset/hw_gain
        """
        if abs(commanded_high - commanded_low) < _EPS_DENOM:
            raise ValueError("Commanded values at two points are identical")
        hw_gain = (actual_high - actual_low) / (commanded_high - commanded_low)
        hw_offset = actual_low - commanded_low * hw_gain
        if abs(hw_gain) < _EPS_DENOM:
            raise ValueError("Hardware gain is effectively zero")
        set_gain = 1.0 / hw_gain
        set_offset = -hw_offset / hw_gain
        return set_gain, set_offset

    # ── Snapshot for undo ────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "channels": deepcopy(self.channels),
            "meter_accuracy_pct": dict(self.meter_accuracy_pct),
        }

    def restore(self, snapshot: dict) -> None:
        if isinstance(snapshot, dict) and "channels" in snapshot:
            self.channels = deepcopy(snapshot["channels"])
            self.meter_accuracy_pct = dict(
                snapshot.get("meter_accuracy_pct", _default_meter_accuracy()))
        else:
            self.channels = deepcopy(snapshot)
        self._dirty = True

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        data: Dict[str, Any] = {
            "version": CALIBRATION_FILE_VERSION,
            "meter_accuracy_pct": self.meter_accuracy_pct,
            "channels": {},
        }
        for key, ch in self.channels.items():
            data["channels"][key] = asdict(ch)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace (ML-099): calibration.json is the single source of
        # every feedforward coefficient — a crash mid-write would truncate
        # it and silently reset the tester to uncalibrated defaults on the
        # next start. A plain write_json stays the convention for measurement
        # files (many, individually re-creatable); this file is unique and
        # expensive to redo.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        write_json(tmp_path, data)
        os.replace(tmp_path, path)
        self._dirty = False
        log.info("Calibration saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "CalibrationData":
        if not path.exists():
            log.warning("No calibration file at %s, using defaults", path)
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to parse calibration file %s: %s", path, exc)
            return cls()

        channels = _default_channels()
        raw_channels = raw.get("channels", {})

        # Migrate v1 ia_read → ia_low_read + ia_high_read
        if "ia_read" in raw_channels and "ia_low_read" not in raw_channels:
            raw_channels["ia_low_read"] = raw_channels.pop("ia_read")
            raw_channels.setdefault("ia_high_read", raw_channels["ia_low_read"])

        for key, vals in raw_channels.items():
            if key in channels and isinstance(vals, dict):
                try:
                    channels[key] = ChannelCal(
                        gain=float(vals.get("gain", 1.0)),
                        offset=float(vals.get("offset", 0.0)),
                        calibrated_at=vals.get("calibrated_at"),
                        quality=vals.get("quality"),
                    )
                except (TypeError, ValueError) as exc:
                    # Valid JSON with a non-numeric gain/offset (hand-edited
                    # or partially written file) used to crash the app at
                    # startup (ML-100). Degrade the single channel to the
                    # identity default — loudly.
                    log.warning(
                        "Calibration channel %s has a corrupt value (%s) — "
                        "using identity defaults for it", key, exc)

        # Validate every stored fit against sanity bounds. With plan B
        # feedforward a bad coefficient drives real commands, so an
        # implausible fit (corrupted or hand-edited file, meter unit
        # error) is reset loudly instead of silently obeyed.
        for key, ch in channels.items():
            if ch.is_default():
                continue
            channel = key.rsplit("_", 1)[0]
            if not fit_within_bounds(channel, ch.gain, ch.offset):
                log.warning(
                    "Calibration %s (gain=%.6f, offset=%.4f) is outside "
                    "sane bounds — reset to default; re-run the wizard",
                    key, ch.gain, ch.offset)
                channels[key] = ChannelCal()

        meter_acc = _default_meter_accuracy()
        raw_meter = raw.get("meter_accuracy_pct", {})
        for k, v in raw_meter.items():
            if k in meter_acc:
                meter_acc[k] = float(v)

        log.info("Calibration loaded from %s (%d channels)", path, len(raw_channels))

        # Plan B behaviour change notice: SET coefficients are now applied
        # as feedforward to every working-point command. A user whose file
        # holds non-default SET values must see why commands shifted.
        active_set = [key for key, ch in channels.items()
                      if key.endswith("_set") and not ch.is_default()]
        if active_set:
            log.info(
                "SET calibration is applied to all working-point commands "
                "(plan B feedforward): %s", ", ".join(sorted(active_set)))

        return cls(channels, meter_acc)

    # ── Helpers ──────────────────────────────────────────────────────

    def calibrated_channels_count(self) -> int:
        return sum(1 for ch in self.channels.values() if not ch.is_default())

    def summary(self) -> str:
        parts = []
        for key, ch in sorted(self.channels.items()):
            if not ch.is_default():
                parts.append(f"{key}: gain={ch.gain:.6f} offset={ch.offset:.4f}")
        return "; ".join(parts) if parts else "all defaults"
