"""Shared helpers for scan test modules."""

import os
import sys
import unittest
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.scan import (
    _set_param_with_settle,
    _read_measurement_point,
    _robust_average,
    _frange,
    _snap_ug1_key,
    _exceeds_pa,
    _exceeds_pg2,
    _exceeds_ig2,
    _find_refine_intervals,
    _find_refine_intervals_per_ug1,
    _build_refine_ua,
    _build_down_sweep_ua,
    _predict_ig2,
    _IG2_PREDICT_MARGIN,
    _closest_grid_idx,
    _refine_curve_inline,
    ProtectionError,
    _check_heater,
    HeaterLostError,
    _wait_for_err_clear,
    _restore_heater_and_wait,
    ScanSettings,
    ScanRange,
    run_scan,
    scan_point_count,
)
from lm19.protocol import encode_ug1, decode_ug1, decode_ig2
from lm19.calibration import CalibrationData, IA_HW_SCALE


def _make_mock_client(get_param_side_effect=None):
    client = MagicMock()
    client.set_param = MagicMock()
    if get_param_side_effect is not None:
        client.get_param = MagicMock(side_effect=get_param_side_effect)
    else:
        client.get_param = MagicMock(return_value=0)
    return client


def _get_param_default(default=100):
    """get_param side_effect: returns 0 for Er, `default` for all others.

    Use instead of ``return_value=100`` in tests that call run_scan,
    because raw 100 (0x64) has OVERIG bit set and triggers hw-error path.
    """
    def fn(name, real=False):
        if name == "Er":
            return 0
        return default
    return fn


def _make_cal(channel="ua", set_gain=1.0, set_offset=0.0,
              read_gain=1.0, read_offset=0.0):
    """CalibrationData with non-default SET/READ coefficients on one channel.

    Plan B (docs/CALIBRATION_PLAN.md): shared factory for tests exercising
    the feedforward / calibrated-verify paths.
    """
    cal = CalibrationData()
    if (set_gain, set_offset) != (1.0, 0.0):
        cal.set_channel(channel, "set", set_gain, set_offset)
    if (read_gain, read_offset) != (1.0, 0.0):
        cal.set_channel(channel, "read", read_gain, read_offset)
    return cal


def _make_scan_settings(**kwargs):
    """Create ScanSettings with sensible defaults, overridden by kwargs."""
    defaults = dict(
        ua=ScanRange(0, 100, 10),
        ug1=ScanRange(0, 0, 0),
        ug2=ScanRange(0, 0, 0),
        uh=6.3, ih=0.0,
        is_triode=True,
        ua_settle_per_volt_s=0.0, ua_settle_base_s=0.0,
        ua_tolerance=1.0, ua_retries=1,
        ug1_settle_per_volt_s=0.0, ug1_settle_base_s=0.0,
        ug1_tolerance=0.1, ug1_retries=1,
        ug2_settle_per_volt_s=0.0, ug2_settle_base_s=0.0,
        ug2_tolerance=1.0, ug2_retries=1,
        ia_samples=1, calibration=CalibrationData(),
    )
    defaults.update(kwargs)
    return ScanSettings(**defaults)
