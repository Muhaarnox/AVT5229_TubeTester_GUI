"""ScanRange / ScanSettings dataclasses + constants + scan_point_count.

Pure data layer — no hardware I/O. Imported by every other scan submodule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from lm19.calibration import CalibrationData
from lm19.constants import EPS

# ── Internal numeric constants ─────────────────────────────────────────
_UA_DELTA_MIN = 0.01            # V — minimum Ua delta for slope calculation
_FRANGE_PRECISION = 6           # decimal digits for _frange rounding
_IA_RANGE_MIN = 0.1             # mA — floor for Ia range in interval analysis
_DOWN_SWEEP_GAP_FACTOR = 1.01   # gap tolerance for inserting sub-steps
_GRID_INTERVAL_MARGIN = 0.5     # V — overlap margin for _mark_grid_intervals

_IG2_PREDICT_MARGIN = 1.0       # predict against limit × margin (conservative)

# Quadratic Ig2 extrapolation (lm19/scan/refine.py::_predict_ig2) is
# unstable when the 3-point baseline is narrow: noise in slope estimates
# is amplified by 1/dua_total, and the dt² term blows the prediction
# off the chart. _UA_DELTA_MIN above is only a "don't divide by zero"
# guard. The wider baseline below is what actually keeps the quadratic
# coefficient physically meaningful — at 0.5 V the noise amplification
# is ~50× lower than at 0.01 V. CI-pinned to be at least 10×
# _UA_DELTA_MIN (see test_predict_ig2.py).
_QUADRATIC_MIN_BASELINE_V = 0.5

# Sanity ceiling for predicted Ig2: anything above this multiple of the
# Ig2 limit is clearly an extrapolation blow-up (real screen-grid currents
# don't 5×-overshoot the protection limit before the actual hardware trip
# fires). Predictions above this are dropped (None returned), so they
# don't trigger the predictive break with garbage data.
_PREDICT_MAX_OVER_LIMIT = 5.0

# If Pa is exceeded on the very first Ua point for this many consecutive
# Ug1 curves, abort the Ug1 sweep — further curves will only be worse.
# (UI guarantees Ua ascending and Ug1 from closed to open, so Pa only grows.
#  For independent Ug2 sweep, abort requires both up- and down-sweep empty.)
_MAX_CONSECUTIVE_PA_BREAKS = 1

# ── Public defaults (used by ScanSettings + UI / SRK) ──────────────────
DEFAULT_SCAN_UA_SETTLE_PER_VOLT_S = 0.0025
DEFAULT_SCAN_UA_SETTLE_BASE_S = 0.15
DEFAULT_SCAN_UA_TOLERANCE = 1.0
DEFAULT_SCAN_UA_RETRIES = 2
DEFAULT_SCAN_UG1_SETTLE_PER_VOLT_S = 0.02
DEFAULT_SCAN_UG1_SETTLE_BASE_S = 0.15
DEFAULT_SCAN_UG1_TOLERANCE = 0.1
DEFAULT_SCAN_UG1_RETRIES = 2
DEFAULT_SCAN_UG2_SETTLE_PER_VOLT_S = 0.0025
DEFAULT_SCAN_UG2_SETTLE_BASE_S = 0.15
DEFAULT_SCAN_UG2_TOLERANCE = 1.0
DEFAULT_SCAN_UG2_RETRIES = 1
DEFAULT_SRK_VERIFY_RETRIES = 3
DEFAULT_SRK_UA_TOLERANCE = 2.0
DEFAULT_SRK_UG2_TOLERANCE = 2.0
DEFAULT_SRK_SETTLE_PER_VOLT_S = 1.0
DEFAULT_SRK_SETTLE_BASE_S = 1.0

# Comm-error retry delays: short pause for automatic retries,
# longer pause when user explicitly requests retry via dialog.
COMM_AUTO_RETRY_DELAY_S = 0.1
COMM_USER_RETRY_DELAY_S = 0.2


@dataclass
class ScanRange:
    start: float
    stop: float
    step: float


@dataclass
class ScanSettings:
    ua: ScanRange
    ug1: ScanRange
    ug2: ScanRange
    uh: float
    ih: float
    an: int = 1
    is_triode: bool = False
    ug2_track_ua: bool = False
    ug2_offset: float = 0.0
    calibration: CalibrationData = None  # type: ignore[assignment]
    pa_max_w: float = 0.0
    pa_over_pct: float = 0.0
    # Per-parameter settle/verify settings
    ua_settle_per_volt_s: float = DEFAULT_SCAN_UA_SETTLE_PER_VOLT_S
    ua_settle_base_s: float = DEFAULT_SCAN_UA_SETTLE_BASE_S
    ua_tolerance: float = DEFAULT_SCAN_UA_TOLERANCE
    ua_retries: int = DEFAULT_SCAN_UA_RETRIES
    ug1_settle_per_volt_s: float = DEFAULT_SCAN_UG1_SETTLE_PER_VOLT_S
    ug1_settle_base_s: float = DEFAULT_SCAN_UG1_SETTLE_BASE_S
    ug1_tolerance: float = DEFAULT_SCAN_UG1_TOLERANCE
    ug1_retries: int = DEFAULT_SCAN_UG1_RETRIES
    ug2_settle_per_volt_s: float = DEFAULT_SCAN_UG2_SETTLE_PER_VOLT_S
    ug2_settle_base_s: float = DEFAULT_SCAN_UG2_SETTLE_BASE_S
    ug2_tolerance: float = DEFAULT_SCAN_UG2_TOLERANCE
    ug2_retries: int = DEFAULT_SCAN_UG2_RETRIES
    ia_samples: int = 3
    ia_outlier_ratio: float = 2.0
    ia_outlier_reread_samples: int = 3
    comm_retries: int = 2
    pig2_max_w: float = 0.0
    pig2_over_pct: float = 0.0
    ig2_max_ma: float = 0.0
    # Adaptive refine (two-pass)
    refine_enabled: bool = False
    refine_max_depth: int = 2
    refine_min_step_ua: float = 3.0
    refine_onset_ma: float = 0.5
    refine_curvature_thr: float = 0.15
    refine_gradient_ratio: float = 3.0
    refine_ig2_delta_min: float = 0.5
    refine_delta_ia_thr: float = 0.25
    down_max_step_v: float = 25.0


def _frange(start: float, stop: float, step: float) -> List[float]:
    """Inclusive float-range list with rounding to ``_FRANGE_PRECISION`` digits.

    Supports both ascending (``start <= stop``) and descending sweeps.
    Returns ``[start]`` for ``step == 0``.
    """
    values: List[float] = []
    if step == 0:
        return [start]
    if start <= stop:
        v = start
        while v <= stop + EPS:
            values.append(round(v, _FRANGE_PRECISION))
            v += step
    else:
        v = start
        while v >= stop - EPS:
            values.append(round(v, _FRANGE_PRECISION))
            v -= abs(step)
    return values


def scan_point_count(settings: ScanSettings) -> int:
    """Total number of points the scan will produce (no refine)."""
    ua_n = len(_frange(settings.ua.start, settings.ua.stop, settings.ua.step))
    ug1_n = len(_frange(settings.ug1.start, settings.ug1.stop, settings.ug1.step))
    if settings.is_triode or settings.ug2_track_ua:
        return ua_n * ug1_n
    ug2_n = len(_frange(settings.ug2.start, settings.ug2.stop, settings.ug2.step))
    return ua_n * ug1_n * ug2_n
