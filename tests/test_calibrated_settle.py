"""Tests for _set_param_calibrated — the calibration adapter around
_set_param_with_settle (plan B, docs/CALIBRATION_PLAN.md).

Contract:
- caller domain is physical; adapter converts once (feedforward):
    cmd      = cal.apply_set(channel, target_phys)
    expected = cal.read_inverse(channel, target_phys)
    return     cal.apply_read(channel, actual_dev)
- the pipe (_set_param_with_settle) stays raw; protection setpoint
  check is unaffected by calibration;
- cal=None and default CalibrationData() are exact no-ops.
"""

import os
import sys
import unittest
from typing import Callable, Optional
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from scan_test_helpers import (
    _make_mock_client, _make_cal, CalibrationData, ProtectionError,
)
from lm19.scan import _set_param_calibrated
from lm19.protocol import encode_ug1, decode_ug1


# ── hardware simulator ────────────────────────────────────────────────

class _FakeDevice:
    """One-channel device: DAC + ADC linear error model.

    DAC: physical = dac_gain * decoded_cmd + dac_offset
    ADC: decoded reading r such that apply_read(r) == physical, i.e.
         r = (physical - read_offset) / read_gain
    """

    def __init__(self, dac_gain: float = 1.0, dac_offset: float = 0.0,
                 read_gain: float = 1.0, read_offset: float = 0.0,
                 decode_cmd: Optional[Callable] = None,
                 encode_reading: Optional[Callable] = None):
        self.dac_gain = dac_gain
        self.dac_offset = dac_offset
        self.read_gain = read_gain
        self.read_offset = read_offset
        self.decode_cmd = decode_cmd or (lambda raw: float(raw))
        self.encode_reading = encode_reading or (lambda v: v)
        self.set_calls = []
        self.setpoint_zeroed = False

    def set_param(self, name, raw):
        self.set_calls.append((name, raw))

    def get_param(self, name, real=False):
        if not real:
            # Setpoint readback (protection check path)
            if self.setpoint_zeroed or not self.set_calls:
                return 0
            return self.set_calls[-1][1]
        decoded_cmd = self.decode_cmd(self.set_calls[-1][1])
        phys = self.dac_gain * decoded_cmd + self.dac_offset
        reading = (phys - self.read_offset) / self.read_gain
        return self.encode_reading(reading)

    def is_open(self):
        return True


def _call(client, cal, target: float, prev: float = 0.0, *,
          channel: str = "ua", name: str = "Ua",
          tolerance: float = 1.0, max_retries: int = 2,
          encode_fn=None, decode_fn=None) -> float:
    return _set_param_calibrated(
        client, name, channel, target, prev, cal,
        settle_per_volt_s=0.0, settle_base_s=0.0,
        tolerance=tolerance, max_retries=max_retries,
        encode_fn=encode_fn, decode_fn=decode_fn,
    )


# ── tests ─────────────────────────────────────────────────────────────

class TestNoOpBehaviour(unittest.TestCase):
    """Default / None calibration must be bit-identical to the raw pipe."""

    @patch("time.sleep")
    def test_default_cal_sends_raw_target(self, mock_sleep):
        client = _make_mock_client()
        client.get_param.return_value = 200
        actual = _call(client, CalibrationData(), 200.0)
        self.assertAlmostEqual(actual, 200.0)
        client.set_param.assert_called_once_with("Ua", 200)
        self.assertEqual(client.get_param.call_count, 1)

    @patch("time.sleep")
    def test_none_cal_identity(self, mock_sleep):
        client = _make_mock_client()
        client.get_param.return_value = 200
        actual = _call(client, None, 200.0)
        self.assertAlmostEqual(actual, 200.0)
        client.set_param.assert_called_once_with("Ua", 200)


class TestFeedforward(unittest.TestCase):
    @patch("time.sleep")
    def test_dac_error_compensated_first_shot(self, mock_sleep):
        """SET cal pre-corrects the command: physical == target, one shot."""
        dac_gain, dac_offset = 1.04, 2.0
        device = _FakeDevice(dac_gain=dac_gain, dac_offset=dac_offset)
        cal = _make_cal(set_gain=1.0 / dac_gain, set_offset=-dac_offset / dac_gain)
        actual = _call(device, cal, 200.0)
        self.assertAlmostEqual(actual, 200.0, places=0)
        self.assertEqual(len(device.set_calls), 1)
        # Command was pre-corrected, not the raw target
        self.assertNotEqual(device.set_calls[0][1], 200)

    @patch("time.sleep")
    def test_verify_against_expected_not_cmd(self, mock_sleep):
        """READ != SET: verify must use read_inverse(target), not cmd.

        DAC +4%/+2V, ADC: phys = 1.02*r - 1. Device reading for
        phys=200 is r=(200+1)/1.02=197.06 while cmd=190.4 — comparing
        the reading against cmd would always retry and fail.
        """
        dac_gain, dac_offset = 1.04, 2.0
        read_gain, read_offset = 1.02, -1.0
        device = _FakeDevice(dac_gain=dac_gain, dac_offset=dac_offset,
                             read_gain=read_gain, read_offset=read_offset)
        cal = _make_cal(set_gain=1.0 / dac_gain, set_offset=-dac_offset / dac_gain,
                        read_gain=read_gain, read_offset=read_offset)
        actual = _call(device, cal, 200.0)
        # Single shot (no spurious retry) and physical result
        self.assertEqual(len(device.set_calls), 1)
        self.assertAlmostEqual(actual, 200.0, places=0)

    @patch("time.sleep")
    def test_returns_physical_actual_on_residual(self, mock_sleep):
        """Uncompensated residual is reported honestly in physical units."""
        # DAC has +3V offset the calibration does not know about
        device = _FakeDevice(dac_offset=3.0,
                             read_gain=1.02, read_offset=-1.0)
        cal = _make_cal(read_gain=1.02, read_offset=-1.0)
        actual = _call(device, cal, 200.0, max_retries=1, tolerance=1.0)
        self.assertAlmostEqual(actual, 203.0, places=6)

    @patch("time.sleep")
    def test_retry_resends_same_corrected_raw(self, mock_sleep):
        """Feedforward is computed once; retry re-sends the same command."""
        client = _make_mock_client()
        # expected = read_inverse(200) = (200-0.5)/1.0 = 199.5
        # first read off, second read on target
        client.get_param.side_effect = [150.0, 199.5]
        cal = _make_cal(set_gain=0.99, read_offset=0.5)
        _call(client, cal, 200.0)
        raws = [c.args[1] for c in client.set_param.call_args_list]
        self.assertEqual(len(raws), 2)
        self.assertEqual(raws[0], raws[1])
        self.assertEqual(raws[0], 198)  # int(round(200*0.99))

    @patch("time.sleep")
    def test_prev_phys_converted_for_settle(self, mock_sleep):
        """Settle time uses command-domain delta of pre-corrected values."""
        client = _make_mock_client()
        client.get_param.return_value = 250.0
        cal = _make_cal(set_gain=0.5)
        _set_param_calibrated(
            client, "Ua", "ua", 250.0, 50.0, cal,
            settle_per_volt_s=0.01, settle_base_s=0.1,
            tolerance=500.0, max_retries=1,
        )
        # cmd=125, prev_cmd=25 → settle = 100*0.01 + 0.1 = 1.1
        settle_time = mock_sleep.call_args_list[0].args[0]
        self.assertAlmostEqual(settle_time, 1.1, places=3)


class TestProtectionUnaffected(unittest.TestCase):
    @patch("time.sleep")
    def test_zeroed_setpoint_detected_with_offset_cal(self, mock_sleep):
        """READ offset must not mask the 'firmware zeroed setpoint' detect.

        This is the bug of the rejected closure design: apply_read inside
        decode_fn shifts the zero (apply_read(0)=offset > tolerance) and
        the ProtectionError is silently skipped.
        """
        device = _FakeDevice(dac_gain=0.0, dac_offset=0.0,
                             read_gain=1.0, read_offset=0.5)
        device.setpoint_zeroed = True
        cal = _make_cal(read_offset=0.5)
        with self.assertRaises(ProtectionError):
            _call(device, cal, 200.0, tolerance=0.3)


class TestUg1Domain(unittest.TestCase):
    @patch("time.sleep")
    def test_ug1_sign_through_adapter(self, mock_sleep):
        """Negative physical target → negative cmd → abs-encoded raw."""
        dac_gain = 0.98
        device = _FakeDevice(
            dac_gain=dac_gain,
            decode_cmd=decode_ug1,
            encode_reading=lambda v: int(round(abs(v) * 100)),
        )
        cal = _make_cal(channel="ug1", set_gain=1.0 / dac_gain)
        actual = _call(device, cal, -6.0, prev=-2.0,
                       channel="ug1", name="Ug1", tolerance=0.1,
                       encode_fn=encode_ug1, decode_fn=decode_ug1)
        # cmd = -6/0.98 = -6.122 → encode_ug1 → 612
        self.assertEqual(device.set_calls[0][1], 612)
        self.assertAlmostEqual(actual, -6.0, places=1)


if __name__ == "__main__":
    unittest.main()
