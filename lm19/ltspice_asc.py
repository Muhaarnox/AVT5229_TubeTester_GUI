"""LTSpice schematic (.asc) generation from templates.

Generates .asc and .asy files alongside the .sub model file,
so the user can open the schematic in LTSpice and immediately
run a simulation to verify the fitted model.

Templates are in config/templates/:
  - triode.asy / pentode.asy — tube symbol (rectangle with labeled pins)
  - test_triode.asc / test_pentode.asc — bare tube DC sweep
  - amp_se_triode.asc / amp_se_pentode.asc — SE resistive amplifier
  - amp_cf_triode.asc — cathode follower
  - amp_pp_triode.asc — push-pull (two tubes)
"""


from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from lm19.amplifier.constants import (
    CIRCUIT_CF,
    CIRCUIT_PP,
    CIRCUIT_SE,
    CIRCUIT_SE_XFMR,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
)

log = logging.getLogger(__name__)

# Template directory relative to this file
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "config" / "templates"

# Fallback defaults when measurement data is empty or incomplete
_DEFAULT_UA_MAX = 400       # V — typical anode voltage range
_DEFAULT_UG1_START = -8     # V — typical grid bias start
_DEFAULT_UG1_STOP = 0       # V — grid bias stop
_DEFAULT_STEP = 1           # V — minimum sweep step
_MIN_UG2_THRESHOLD = 10     # V — ignore Ug2 values below this (noise/zero)
_DEFAULT_UG2 = 250          # V — typical screen grid voltage


def _safe_name(tube_type: str) -> str:
    """Convert tube type to safe filename (same logic as spice_export)."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in tube_type)


def _extract_sweep_params(
    points: List[Dict], topology: str,
) -> Dict[str, str]:
    """Extract DC sweep parameters from measurement points.

    Returns dict with keys: ua_max, ua_step, ug1_start, ug1_stop,
    ug1_step, and optionally ug2.
    """
    ua_values = sorted(set(round(p["ua"], 1) for p in points))
    ug1_values = sorted(set(round(p["ug1"], 2) for p in points))

    ua_max = ua_values[-1] if ua_values else _DEFAULT_UA_MAX
    ua_step = _DEFAULT_STEP
    if len(ua_values) >= 2:
        diffs = [ua_values[i + 1] - ua_values[i] for i in range(len(ua_values) - 1)]
        ua_step = max(_DEFAULT_STEP, round(min(diffs)))

    # Ug1: from most negative to least negative
    ug1_start = ug1_values[0] if ug1_values else _DEFAULT_UG1_START
    ug1_stop = ug1_values[-1] if ug1_values else _DEFAULT_UG1_STOP
    ug1_step = _DEFAULT_STEP
    if len(ug1_values) >= 2:
        diffs = [ug1_values[i + 1] - ug1_values[i] for i in range(len(ug1_values) - 1)]
        ug1_step = round(max(diffs), 2)
        if ug1_step <= 0:
            ug1_step = _DEFAULT_STEP

    params = {
        "ua_max": str(int(ua_max)),
        "ua_step": str(int(ua_step)),
        "ug1_start": str(ug1_start),
        "ug1_stop": str(ug1_stop),
        "ug1_step": str(ug1_step),
    }

    if topology == TOPOLOGY_PENTODE:
        ug2_values = sorted(set(round(p.get("ug2", 0), 1) for p in points))
        ug2_values = [v for v in ug2_values if v > _MIN_UG2_THRESHOLD]
        if ug2_values:
            params["ug2"] = str(int(ug2_values[len(ug2_values) // 2]))
        else:
            params["ug2"] = str(_DEFAULT_UG2)

    return params


def generate_test_schematic(
    sub_path: str,
    tube_type: str,
    points: List[Dict],
    topology: str,
) -> Optional[str]:
    """Generate LTSpice test schematic (.asc + .asy) next to the .sub file.

    Args:
        sub_path: path to the exported .sub file.
        tube_type: tube type name (e.g. "12AX7").
        points: measurement data points.
        topology: "triode" or "pentode".

    Returns:
        Path to the generated .asc file, or None on error.
    """
    sub_path = Path(sub_path)
    out_dir = sub_path.parent
    safe = _safe_name(tube_type)

    # Pick template
    is_pentode = topology == TOPOLOGY_PENTODE
    tpl_asc = "test_pentode.asc" if is_pentode else "test_triode.asc"
    tpl_asy = "pentode.asy" if is_pentode else "triode.asy"

    tpl_asc_path = _TEMPLATES_DIR / tpl_asc
    tpl_asy_path = _TEMPLATES_DIR / tpl_asy

    if not tpl_asc_path.exists():
        log.error("Template not found: %s", tpl_asc_path)
        return None
    if not tpl_asy_path.exists():
        log.error("Template not found: %s", tpl_asy_path)
        return None

    # Extract sweep parameters from data
    sweep = _extract_sweep_params(points, topology)

    # Substitution values
    sub_filename = sub_path.name
    values = {
        "tube_name": safe,
        "sub_file": sub_filename,
        **sweep,
    }

    # Read and substitute .asc template
    asc_content = tpl_asc_path.read_text(encoding="utf-8")
    for key, val in values.items():
        asc_content = asc_content.replace("{" + key + "}", val)

    # Write .asc
    asc_path = out_dir / f"{safe}_test.asc"
    asc_path.write_text(asc_content, encoding="utf-8")
    log.info("Generated test schematic: %s", asc_path)

    # Copy .asy symbol (renamed to tube name)
    asy_path = out_dir / f"{safe}.asy"
    shutil.copy2(tpl_asy_path, asy_path)
    log.info("Copied symbol file: %s", asy_path)

    return str(asc_path)


# ── Amplifier circuit templates ───────────────────────────────────

# Default component values for amplifier schematics
_DEFAULT_RA_OHM = "47k"
_DEFAULT_RK_OHM = "1.5k"
_DEFAULT_RA_AA_OHM = "10k"
_DEFAULT_RA_PER_TUBE_OHM = "2.5k"
_DEFAULT_UB = "250"


def generate_amp_schematic(
    sub_path: str,
    tube_type: str,
    points: List[Dict],
    topology: str,
    circuit: str = CIRCUIT_SE,
    ub: float = 250.0,
    ra_ohm: str = "",
    rk_ohm: str = "",
    ra_dc_ohm: str = "",
    ug2: float = 250.0,
    ra_aa_ohm: str = "",
    r_load: str = "8",
    f_low: float = 20.0,
    tube_name_b: str = "",
    sub_file_b: str = "",
) -> Optional[str]:
    """Generate LTSpice amplifier schematic (.asc + .asy).

    Args:
        sub_path: path to the exported .sub file.
        tube_type: tube type name (e.g. "12AX7").
        points: measurement data points.
        topology: "triode" or "pentode".
        circuit: "se", "se_xfmr", "cf", or "pp".
        ub: supply voltage (V).
        ra_ohm: anode resistor value (e.g. "47k", "5.6k").
        rk_ohm: cathode resistor value (e.g. "1.5k", "470").
        ra_dc_ohm: DC winding resistance for SE Transformer (e.g. "50").
        ug2: screen grid voltage for pentodes (V).
        ra_aa_ohm: anode-to-anode impedance for PP (e.g. "10k").
        r_load: speaker load impedance (e.g. "4", "8", "16").
        f_low: transformer low-frequency -3dB point (Hz).

    Returns:
        Path to the generated .asc file, or None on error.
    """
    sub_path = Path(sub_path)
    out_dir = sub_path.parent
    safe = _safe_name(tube_type)
    is_pentode = topology == TOPOLOGY_PENTODE

    # Select template
    if circuit == CIRCUIT_PP:
        tpl_name = "amp_pp_pentode.asc" if is_pentode else "amp_pp_triode.asc"
    elif circuit == CIRCUIT_CF:
        tpl_name = "amp_cf_pentode.asc" if is_pentode else "amp_cf_triode.asc"
    elif circuit == CIRCUIT_SE_XFMR:
        tpl_name = "amp_se_xfmr_pentode.asc" if is_pentode else "amp_se_xfmr_triode.asc"
    elif is_pentode:
        tpl_name = "amp_se_pentode.asc"
    else:
        tpl_name = "amp_se_triode.asc"

    tpl_key = tpl_name.replace(".asc", "").replace("amp_", "")
    tpl_path = _TEMPLATES_DIR / tpl_name
    tpl_asy = "pentode.asy" if is_pentode else "triode.asy"
    tpl_asy_path = _TEMPLATES_DIR / tpl_asy

    if not tpl_path.exists():
        log.error("Amp template not found: %s", tpl_path)
        return None

    # Sweep parameters from data
    sweep = _extract_sweep_params(points, topology)

    # Component defaults
    if not ra_ohm:
        ra_ohm = _DEFAULT_RA_OHM
    if not rk_ohm:
        rk_ohm = _DEFAULT_RK_OHM
    if not ra_dc_ohm:
        ra_dc_ohm = "50"  # 50 Ohm typical transformer winding resistance
    if not ra_aa_ohm:
        ra_aa_ohm = _DEFAULT_RA_AA_OHM

    # PP: compute per-tube Ra
    ra_per_tube_ohm = _DEFAULT_RA_PER_TUBE_OHM
    if circuit == CIRCUIT_PP and ra_aa_ohm:
        try:
            # Parse "10k" → 10000 / 4 → "2.5k"
            ra_aa_val = _parse_resistance(ra_aa_ohm)
            ra_per_tube_ohm = _format_resistance(ra_aa_val / 4.0)
        except ValueError:
            pass

    # Transformer parameters
    # L_primary = Ra_reflected / (2*pi*f_low) — sets -3dB low-frequency point
    import math
    ra_val = _parse_resistance(ra_ohm) if ra_ohm else 47000
    r_load_val = float(r_load) if r_load else 8.0
    l_primary_val = ra_val / (2.0 * math.pi * max(f_low, 1.0))
    l_primary = f"{l_primary_val:.1f}" if l_primary_val >= 1.0 else f"{l_primary_val * 1000:.1f}m"
    # Secondary: L2 = L1 / n², where n² = Ra / Rload (turns ratio squared)
    turns_ratio_sq = ra_val / max(r_load_val, 1.0)
    l_secondary_val = l_primary_val / max(turns_ratio_sq, 1.0)
    l_secondary = f"{l_secondary_val:.4f}" if l_secondary_val >= 0.001 else f"{l_secondary_val * 1000:.2f}m"

    # Input signal amplitude (quarter of Ug1 range)
    ug1_range = abs(float(sweep.get("ug1_stop", "0")) - float(sweep.get("ug1_start", "-10")))
    vin_ampl = f"{ug1_range / 4:.1f}"

    # PP tube B: defaults to same as A (matched pair)
    safe_b = _safe_name(tube_name_b) if tube_name_b else safe
    sub_file_b_name = sub_file_b if sub_file_b else sub_path.name

    values = {
        "tube_name": safe,
        "tube_name_a": safe,
        "tube_name_b": safe_b,
        "sub_file": sub_path.name,
        "sub_file_a": sub_path.name,
        "sub_file_b": sub_file_b_name,
        "ub": str(int(ub)),
        "ra_ohm": ra_ohm,
        "rk_ohm": rk_ohm,
        "ra_dc_ohm": ra_dc_ohm,
        "ug2": str(int(ug2)),
        "ra_aa_ohm": ra_aa_ohm,
        "ra_per_tube_ohm": ra_per_tube_ohm,
        "l_primary": l_primary,
        "l_secondary": l_secondary,
        "r_load": r_load if r_load else "8",
        "vin_ampl": vin_ampl,
        "ug1_step_neg": str(-abs(float(sweep.get("ug1_step", "1")))),
        **sweep,
    }

    # Read and substitute template
    content = tpl_path.read_text(encoding="utf-8")
    for key, val in values.items():
        content = content.replace("{" + key + "}", val)

    # Write .asc
    suffix = f"_{circuit}" if circuit != CIRCUIT_SE else "_amp"
    asc_path = out_dir / f"{safe}{suffix}.asc"
    asc_path.write_text(content, encoding="utf-8")
    log.info("Generated amp schematic: %s", asc_path)

    # Copy .asy symbol(s). ML-061: overwrite unconditionally, like the
    # .asc/.sub — a stale 3-pin triode symbol left by an earlier export
    # breaks a pentode schematic's pinout.
    asy_path = out_dir / f"{safe}.asy"
    shutil.copy2(tpl_asy_path, asy_path)
    # For PP mismatched: copy .asy for tube B too
    if safe_b != safe:
        asy_b_path = out_dir / f"{safe_b}.asy"
        shutil.copy2(tpl_asy_path, asy_b_path)

    return str(asc_path)


def _parse_resistance(s: str) -> float:
    """Parse resistance string like '47k', '1.5k', '1Meg', '470' → Ohms."""
    s = s.strip()
    sl = s.lower()
    if sl.endswith("meg"):
        return float(s[:-3]) * 1e6
    if sl.endswith("k"):
        return float(s[:-1]) * 1000
    if sl.endswith("m"):
        return float(s[:-1]) * 1e6
    return float(s)


def _format_resistance(ohms: float) -> str:
    """Format resistance as string: 2500 → '2.5k', 470 → '470'."""
    if ohms >= 1e6:
        return f"{ohms / 1e6:.1f}Meg"
    if ohms >= 1000:
        val = ohms / 1000
        return f"{val:.1f}k" if val != int(val) else f"{int(val)}k"
    return f"{int(ohms)}"
