
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)

log = logging.getLogger(__name__)


@dataclass
class LampRange:
    min: float
    max: float
    step: float


@dataclass
class LampConfig:
    tube_type: str
    socket: str
    anodes: int
    warmup_s: int
    topology: str  # "triode" or "pentode"
    uh: float
    ih: float
    ug1: float
    ua: float
    ia: float
    ug2: float
    ig2: float
    s: float
    r: float
    k: float
    ranges: Dict[str, LampRange]
    limits: Dict[str, float]
    pa_max: Optional[float] = None
    pig2_max: Optional[float] = None
    ua_max_limit: Optional[float] = None
    ia_max_limit: Optional[float] = None
    ra: Optional[float] = None
    anode_default: Optional[int] = None

    @property
    def is_triode(self) -> bool:
        """True triode — no screen grid, Ug2 not applicable."""
        return self.topology == TOPOLOGY_TRIODE


DEFAULT_LIMITS = {
    "ua_max": 300.0,
    "ug2_max": 300.0,
    "ug1_max": 24.0,
    "uh_max": 15.0,
    "ih_max": 2.5,
    "ia_max": 200.0,
    "ig2_max": 20.0,
}

# lamps.json stores per-lamp ``ih_max`` in mA holding the TDSL NOMINAL
# heater current (sync_tdsl_limits writes IhA*1000); the loader converts it
# to A and applies the same +10% headroom the sync tool already bakes into
# ``uh_max`` (ML-130).
IH_MAX_HEADROOM = 1.1
_MA_PER_A = 1000.0


def _default_range(center: float, max_val: float, step: float, span: float) -> LampRange:
    low = max(0.0, center - span)
    high = min(max_val, center + span)
    return LampRange(min=low, max=high, step=step)


def _resolve_paths() -> Path:
    return Path(__file__).resolve().parents[1]


def load_lamps() -> List[LampConfig]:
    root = _resolve_paths()
    config_path = root / "config" / "lamps.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    limits = load_device_limits()

    lamps: List[LampConfig] = []
    for item in data.get("lamps", []):
        lamp_limits = limits.copy()
        lamp_limits.update(item.get("limits", {}))
        # Per-lamp top-level limit overrides (ML-130/131): the sync/extract
        # tools store them on the lamp card; if the loader ignored them, the
        # per-lamp heater/screen protection would be dead. The device limit
        # stays the hard ceiling. Units: uh_max V
        # (stored WITH the sync tool's +10% headroom), ih_max mA (TDSL
        # nominal, converted here), ug2_max V.
        nominal_uh = float(item.get("uh", 0) or 0)
        nominal_ih = float(item.get("ih", 0) or 0)
        if item.get("uh_max") is not None:
            card_uh_max = float(item["uh_max"])
            if 0 < card_uh_max < nominal_uh:
                # Broken card (seen: E182CC/ECC99 — 12.6 V heaters carrying
                # uh_max=6.9): a cap below the nominal would silently clamp
                # the heater to under-voltage. Floor at nominal+headroom,
                # mirroring the sync tool's "never below nominal" policy.
                log.warning(
                    "%s: uh_max=%.1fV below nominal uh=%.1fV — flooring at "
                    "nominal+%d%%", item["type"], card_uh_max, nominal_uh,
                    round((IH_MAX_HEADROOM - 1) * 100))
                card_uh_max = nominal_uh * IH_MAX_HEADROOM
            lamp_limits["uh_max"] = min(limits["uh_max"], card_uh_max)
        if item.get("ih_max") is not None:
            card_ih_max = float(item["ih_max"]) / _MA_PER_A * IH_MAX_HEADROOM
            if 0 < card_ih_max < nominal_ih:
                log.warning(
                    "%s: ih_max=%.0fmA below nominal ih=%.2fA — flooring at "
                    "nominal+%d%%", item["type"], float(item["ih_max"]),
                    nominal_ih, round((IH_MAX_HEADROOM - 1) * 100))
                card_ih_max = nominal_ih * IH_MAX_HEADROOM
            lamp_limits["ih_max"] = min(limits["ih_max"], card_ih_max)
        if item.get("ug2_max") is not None:
            lamp_limits["ug2_max"] = min(
                limits["ug2_max"], float(item["ug2_max"]))

        ranges: Dict[str, LampRange] = {}
        range_data = item.get("ranges", {})
        if range_data:
            for key, val in range_data.items():
                rng_max = float(val.get("max", 0))
                if key == "ug2":
                    # An explicit card range must not exceed the per-lamp
                    # screen-voltage cap (ML-131).
                    rng_max = min(rng_max, lamp_limits["ug2_max"])
                ranges[key] = LampRange(
                    min=float(val.get("min", 0)),
                    max=rng_max,
                    step=float(val.get("step", 1)),
                )
        else:
            ranges["ua"] = _default_range(float(item["ua"]), lamp_limits["ua_max"], 10.0, 50.0)
            ranges["ug1"] = _default_range(abs(float(item["ug1"])), lamp_limits["ug1_max"], 1.0, 5.0)
            ranges["ug2"] = _default_range(float(item["ug2"]), lamp_limits["ug2_max"], 10.0, 50.0)

        pa_max_raw = item.get("Pa_max")
        pa_max = float(pa_max_raw) if pa_max_raw is not None else None
        pig2_max_raw = item.get("Pig2_max")
        pig2_max = float(pig2_max_raw) if pig2_max_raw is not None else None
        ua_max_raw = item.get("ua_max")
        ua_max_limit = float(ua_max_raw) if ua_max_raw is not None else None
        ia_max_raw = item.get("ia_max")
        ia_max_limit = float(ia_max_raw) if ia_max_raw is not None else None
        ra_raw = item.get("Ra")
        ra = float(ra_raw) if ra_raw is not None else None
        anodes = int(item.get("anodes", 1))
        anode_default_raw = item.get("anode_default")
        if anodes > 1 and anode_default_raw in (1, 2):
            anode_default = int(anode_default_raw)
        else:
            anode_default = None

        lamps.append(
            LampConfig(
                tube_type=item["type"],
                socket=item.get("socket", ""),
                anodes=anodes,
                warmup_s=int(item.get("warmup_s", 120)),
                topology=item.get("topology", TOPOLOGY_PENTODE),
                uh=float(item.get("uh", 0)),
                ih=float(item.get("ih", 0)),
                ug1=float(item.get("ug1", 0)),
                ua=float(item.get("ua", 0)),
                ia=float(item.get("ia", 0)),
                ug2=float(item.get("ug2", 0)),
                ig2=float(item.get("ig2", 0)),
                s=float(item.get("s", 0)),
                r=float(item.get("r", 0)),
                k=float(item.get("k", 0)),
                ranges=ranges,
                limits=lamp_limits,
                pa_max=pa_max,
                pig2_max=pig2_max,
                ua_max_limit=ua_max_limit,
                ia_max_limit=ia_max_limit,
                ra=ra,
                anode_default=anode_default,
            )
        )
    return lamps


def load_device_limits() -> Dict[str, float]:
    """Load device limits from config/device.json, merged with DEFAULT_LIMITS."""
    root = _resolve_paths()
    config_path = root / "config" / "device.json"
    limits = DEFAULT_LIMITS.copy()
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            limits.update(data)
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            log.warning("Failed to load device limits from %s: %s", config_path, exc)
    return limits


def find_lamp(lamps: List[LampConfig], tube_type: str) -> Optional[LampConfig]:
    for lamp in lamps:
        if lamp.tube_type == tube_type:
            return lamp
    return None
