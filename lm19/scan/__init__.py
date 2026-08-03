"""Scan package — IV-curve measurement orchestrator.

Public API:
  - ``run_scan`` — top-level scan orchestrator
  - ``ScanRange``, ``ScanSettings`` — settings dataclasses
  - ``scan_point_count`` — total points the scan will produce
  - ``ProtectionError``, ``HeaterLostError`` — public exceptions

Submodules:
  - ``settings``   — dataclasses + DEFAULT_* constants + ``_frange``
  - ``exceptions`` — ``_SkipPoint``, ``_BreakSweep``, ``ProtectionError``,
    ``HeaterLostError``
  - ``io``         — ``_set_param_with_settle``, ``_set_param_calibrated``,
    ``_read_measurement_point``, ``_robust_average``, ``_try_reopen``
  - ``protection`` — heater & limit checks, recovery primitives
  - ``refine``     — adaptive refinement helpers + ``_refine_curve_inline``
  - ``sweepers``   — ``_SweepCtx`` + 3 sweep functions
  - ``runner``     — ``run_scan``

SRK measurement (``measure_srk``, ``SrkSettings``, ``SrkVerifyError``,
``_srk_ug1_values``) lives in :mod:`lm19.srk` and is re-exported here
so call sites that already read it from ``lm19.scan`` keep working.

Private helpers are also re-exported so test helpers and other
internal consumers can reach them through the package root.
"""

from __future__ import annotations

# Public dataclasses + helpers
from lm19.scan.settings import (
    COMM_AUTO_RETRY_DELAY_S,
    COMM_USER_RETRY_DELAY_S,
    DEFAULT_SCAN_UA_RETRIES,
    DEFAULT_SCAN_UA_SETTLE_BASE_S,
    DEFAULT_SCAN_UA_SETTLE_PER_VOLT_S,
    DEFAULT_SCAN_UA_TOLERANCE,
    DEFAULT_SCAN_UG1_RETRIES,
    DEFAULT_SCAN_UG1_SETTLE_BASE_S,
    DEFAULT_SCAN_UG1_SETTLE_PER_VOLT_S,
    DEFAULT_SCAN_UG1_TOLERANCE,
    DEFAULT_SCAN_UG2_RETRIES,
    DEFAULT_SCAN_UG2_SETTLE_BASE_S,
    DEFAULT_SCAN_UG2_SETTLE_PER_VOLT_S,
    DEFAULT_SCAN_UG2_TOLERANCE,
    DEFAULT_SRK_SETTLE_BASE_S,
    DEFAULT_SRK_SETTLE_PER_VOLT_S,
    DEFAULT_SRK_UA_TOLERANCE,
    DEFAULT_SRK_UG2_TOLERANCE,
    DEFAULT_SRK_VERIFY_RETRIES,
    ScanRange,
    ScanSettings,
    _frange,
    _IG2_PREDICT_MARGIN,
    _MAX_CONSECUTIVE_PA_BREAKS,
    scan_point_count,
)

# Exceptions
from lm19.scan.exceptions import (
    HeaterLostError,
    ProtectionError,
    _BreakSweep,
    _SkipPoint,
)

# I/O
from lm19.scan.io import (
    _read_measurement_point,
    _robust_average,
    _set_param_calibrated,
    _set_param_with_settle,
    _try_reopen,
)

# Protection / limits
from lm19.scan.protection import (
    _check_heater,
    _exceeds_ig2,
    _exceeds_pa,
    _exceeds_pg2,
    _restore_heater_and_wait,
    _wait_for_err_clear,
)

# Refine
from lm19.scan.refine import (
    _analyse_curve_intervals,
    _build_down_sweep_ua,
    _build_refine_ua,
    _closest_grid_idx,
    _find_refine_intervals,
    _find_refine_intervals_per_ug1,
    _mark_grid_intervals,
    _predict_ig2,
    _refine_curve_inline,
    _snap_ug1_key,
)

# Sweepers
from lm19.scan.sweepers import (
    _SweepCtx,
    _sweep_triode,
    _sweep_ug2_independent,
    _sweep_ug2_track,
)

# Runner
from lm19.scan.runner import run_scan

# SRK — re-exported so callers can keep importing from lm19.scan.
# Prefer ``from lm19.srk import …`` for new code.
from lm19.srk import SrkSettings, SrkVerifyError, _srk_ug1_values, measure_srk


__all__ = [
    # Public API
    "ScanRange",
    "ScanSettings",
    "scan_point_count",
    "run_scan",
    "ProtectionError",
    "HeaterLostError",
    # SRK (re-exported from lm19.srk)
    "SrkSettings",
    "SrkVerifyError",
    "measure_srk",
]
