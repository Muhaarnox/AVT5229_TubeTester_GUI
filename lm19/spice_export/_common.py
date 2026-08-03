"""Shared scaffolding for the SPICE export package.

Exposes the optional scipy import flag and ``SpiceFitResult`` dataclass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

log = logging.getLogger(__name__)

try:
    from scipy.optimize import least_squares as _least_squares  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    log.warning(
        "scipy not available — SPICE export will use numpy-only fitter "
        "(slower, less accurate). Install scipy for better results: "
        "pip install scipy"
    )


@dataclass
class SpiceFitResult:
    """Result of SPICE model fitting, used for overlay on plots."""
    model_type: str          # "triode" or "pentode"
    algorithm: str           # "koren", "dempwolf", or "reefman"
    params: Dict             # {mu, ex, kg1, kp, kvb, [kg2]}
    rms_error: float         # mA
    max_error: float         # mA
    n_points: int
    path: str                # saved .sub file path
    # Degradation codes (msg.Spice_warn_<code> i18n keys) — the export
    # dialog appends them as ⚠ lines (failure-visibility rule: a silently
    # degraded .sub looks identical to a good one).
    warnings: List[str] = field(default_factory=list)
