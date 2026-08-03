"""Amplifier presets — pre-baked configurations for quick start."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from lm19.amplifier.constants import (
    CIRCUIT_CF,
    CIRCUIT_PP,
    CIRCUIT_SE,
)


@dataclass
class AmplifierPreset:
    """Predefined amplifier configuration for quick start."""
    name: str
    tube: str
    circuit: str       # "se_triode" | "se_pentode" | "se_triode_connected"
    ub: float          # V
    ra: float          # kOhm
    ug1_bias: float    # V
    pa_max: float      # W
    description: str = ""


AMPLIFIER_PRESETS: List[AmplifierPreset] = [
    AmplifierPreset(
        name="12AX7 Preamp",
        tube="12AX7",
        circuit=CIRCUIT_SE,
        ub=250, ra=100.0, ug1_bias=-2.0, pa_max=1.0,
        description="Classic high-gain preamp stage",
    ),
    AmplifierPreset(
        name="12AU7 Driver",
        tube="12AU7",
        circuit=CIRCUIT_SE,
        ub=250, ra=47.0, ug1_bias=-8.5, pa_max=2.75,
        description="Medium-mu driver / line stage",
    ),
    AmplifierPreset(
        name="6SN7 SE",
        tube="6SN7",
        circuit=CIRCUIT_SE,
        ub=300, ra=15.0, ug1_bias=-8.0, pa_max=5.0,
        description="Classic 6SN7 SE amplifier",
    ),
    AmplifierPreset(
        name="EL84 SE Pentode",
        tube="EL84",
        circuit=CIRCUIT_SE,
        ub=250, ra=5.0, ug1_bias=-7.3, pa_max=12.0,
        description="EL84 single-ended pentode, Ug2=250V",
    ),
    AmplifierPreset(
        name="EL34 SE Pentode",
        tube="EL34",
        circuit=CIRCUIT_SE,
        ub=350, ra=3.5, ug1_bias=-20.0, pa_max=25.0,
        description="EL34 single-ended pentode, Ug2=250V",
    ),
    AmplifierPreset(
        name="EL34 Triode Connected",
        tube="EL34",
        circuit=CIRCUIT_SE,
        ub=350, ra=3.5, ug1_bias=-25.0, pa_max=25.0,
        description="EL34 with screen tied to plate",
    ),
    AmplifierPreset(
        name="12AU7 CF",
        tube="12AU7",
        circuit=CIRCUIT_CF,
        ub=250, ra=10.0, ug1_bias=-8.5, pa_max=2.75,
        description="12AU7 cathode follower (Rk=10k)",
    ),
    AmplifierPreset(
        name="EL84 PP",
        tube="EL84",
        circuit=CIRCUIT_PP,
        ub=300, ra=8.0, ug1_bias=-7.3, pa_max=12.0,
        description="EL84 push-pull, Ra_aa=8kΩ",
    ),
    AmplifierPreset(
        name="EL34 PP",
        tube="EL34",
        circuit=CIRCUIT_PP,
        ub=400, ra=6.6, ug1_bias=-20.0, pa_max=25.0,
        description="EL34 push-pull, Ra_aa=6.6kΩ",
    ),
]
