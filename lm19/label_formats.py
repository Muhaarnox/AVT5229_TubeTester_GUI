"""Centralized label / value format strings for UI display.

Single source of truth — import ``format_label`` everywhere instead of
hard-coding f-string precision.  CSV export and diagnostic error messages
have their own domain-specific formats and are intentionally *not* here.
"""


from __future__ import annotations

from typing import Optional

LABEL_FORMATS: dict[str, str] = {
    # --- Ug1 ---------------------------------------------------------------
    "ug1":          "Ug1 {value:.1f} V",
    "ug1_short":    "{value:.1f}V",
    "ug1_value":    "{value:.2f}",
    "ug1_unit":     "{value:.2f} V",
    # --- Ug2 ---------------------------------------------------------------
    "ug2":          "Ug2 {value:.0f} V",
    "ug2_short":    "{value:.0f}V",
    "ug2_value":    "{value:.0f}",
    "ug2_unit":     "{value:.0f} V",
    # --- Ua ----------------------------------------------------------------
    "ua":           "Ua {value:.0f} V",
    "ua_short":     "{value:.0f}V",
    "ua_value":     "{value:.0f}",
    "ua_unit":      "{value:.0f} V",
    # --- Ia ----------------------------------------------------------------
    "ia":           "Ia {value:.2f} mA",
    "ia_short":     "{value:.2f}mA",
    "ia_value":     "{value:.2f}",
    "ia_unit":      "{value:.2f} mA",
    # --- Ig2 ---------------------------------------------------------------
    "ig2":          "Ig2 {value:.2f} mA",
    "ig2_value":    "{value:.2f}",
    "ig2_unit":     "{value:.2f} mA",
    # --- Uh / Ih -----------------------------------------------------------
    "uh":           "Uh {value:.2f} V",
    "uh_value":     "{value:.2f}",
    "uh_unit":      "{value:.2f} V",
    "ih":           "Ih {value:.3f} A",
    "ih_value":     "{value:.3f}",
    "ih_unit":      "{value:.3f} A",
    # --- Pa ----------------------------------------------------------------
    "pa":           "Pa {value:.2f} W",
    "pa_value":     "{value:.2f}",
    "pa_unit":      "{value:.1f} W",
    "pa_limit":     "{value:.1f}/{limit:.1f} W",
    # --- Pg2 ---------------------------------------------------------------
    "pg2":          "Pg2 {value:.2f} W",
    "pg2_value":    "{value:.2f}",
    "pg2_unit":     "{value:.1f} W",
    "pg2_limit":    "{value:.1f}/{limit:.1f} W",
    # --- Load line / amplifier ---------------------------------------------
    "ra":           "Ra={value:.1f}kΩ",
    "ra_value":     "{value:.1f}",
    "pout_mw":      "Pout={value:.0f}mW",
    "pout_w":       "Pout={value:.2f}W",
    "hd":           "HD2={hd2:.1f}%  HD3={hd3:.1f}%",
    "imd":          "IMD2={imd2:.2f}%  IMD3={imd3:.2f}%",
    "swing":        "Ia_pp={ia_pp:.1f}mA  Ua_pp={ua_pp:.0f}V  Swing={swing:.1f}V",
    "q_point":      "Q: Ug1={ug1:.1f}V  Ua={ua:.0f}V  Ia={ia:.1f}mA",
    "load_header":  "Load line: Ub={ub:.0f}V  Ra={ra:.1f}kΩ",
    # --- Health / misc -----------------------------------------------------
    "index_pct":    "Index: {value:.0f}%",
    "emission":     "Em: {value:.3f}",
    "mu":           "{value:.1f}",
    "pct":          "{value:.0f}%",
    "err_abs":      "{value:.2f}±{error:.2f}",
    "err_plain":    "{value:.2f}",
}


def format_label(kind: str, value: float = 0.0, **kwargs) -> str:
    """Format a value using the named format from ``LABEL_FORMATS``.

    >>> format_label("ug1", -1.5)
    'Ug1 -1.5 V'
    >>> format_label("hd", hd2=2.3, hd3=0.1)
    'HD2=2.3%  HD3=0.1%'
    """
    fmt = LABEL_FORMATS.get(kind, "{value}")
    return fmt.format(value=value, **kwargs)
