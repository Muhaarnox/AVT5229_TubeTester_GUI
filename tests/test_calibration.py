"""Tests for lm19.calibration module."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from lm19.calibration import (
    CalibrationData,
    ChannelCal,
    IA_HW_SCALE,
    IA_RANGE_THRESHOLD,
    IA_RANGE_CHANNELS,
    SET_LIMITS,
)


class TestChannelCal(unittest.TestCase):
    def test_default_is_transparent(self):
        cc = ChannelCal()
        self.assertAlmostEqual(cc.apply(100.0), 100.0)
        self.assertTrue(cc.is_default())

    def test_gain_only(self):
        cc = ChannelCal(gain=1.05)
        self.assertAlmostEqual(cc.apply(100.0), 105.0)
        self.assertFalse(cc.is_default())

    def test_offset_only(self):
        cc = ChannelCal(offset=-2.5)
        self.assertAlmostEqual(cc.apply(100.0), 97.5)

    def test_gain_and_offset(self):
        cc = ChannelCal(gain=0.98, offset=1.5)
        self.assertAlmostEqual(cc.apply(200.0), 200.0 * 0.98 + 1.5)

    def test_quality_field(self):
        cc = ChannelCal(gain=1.01, quality={"point_spread": 10.0})
        self.assertEqual(cc.quality["point_spread"], 10.0)
        self.assertFalse(cc.is_default())

    def test_quality_default_none(self):
        cc = ChannelCal()
        self.assertIsNone(cc.quality)


class TestCalibrationData(unittest.TestCase):
    def test_default_transparent(self):
        cal = CalibrationData()
        self.assertAlmostEqual(cal.apply_read("ua", 200.0), 200.0)
        self.assertAlmostEqual(cal.apply_set("ua", 200.0), 200.0)

    def test_apply_read_with_correction(self):
        cal = CalibrationData()
        cal.set_channel("ua", "read", 1.02, -3.0)
        self.assertAlmostEqual(cal.apply_read("ua", 200.0), 200.0 * 1.02 - 3.0)

    def test_apply_set_with_clamping(self):
        cal = CalibrationData()
        cal.set_channel("ua", "set", 1.0, 350.0)
        result = cal.apply_set("ua", 200.0)
        self.assertAlmostEqual(result, 300.0)

    def test_apply_set_negative_clamp(self):
        cal = CalibrationData()
        cal.set_channel("ua", "set", 1.0, -250.0)
        result = cal.apply_set("ua", 50.0)
        self.assertAlmostEqual(result, 0.0)

    def test_reset_channel(self):
        cal = CalibrationData()
        cal.set_channel("ua", "read", 1.05, -2.0)
        self.assertFalse(cal.get_channel("ua", "read").is_default())
        cal.reset_channel("ua", "read")
        self.assertTrue(cal.get_channel("ua", "read").is_default())

    def test_reset_all(self):
        cal = CalibrationData()
        cal.set_channel("ua", "read", 1.05, -2.0)
        cal.set_channel("ia_low", "read", 0.99, -0.08)
        self.assertEqual(cal.calibrated_channels_count(), 2)
        cal.reset_all()
        self.assertEqual(cal.calibrated_channels_count(), 0)

    def test_dirty_flag(self):
        cal = CalibrationData()
        self.assertFalse(cal.dirty)
        cal.set_channel("ua", "read", 1.0, 0.1)
        self.assertTrue(cal.dirty)
        cal.mark_clean()
        self.assertFalse(cal.dirty)

    def test_snapshot_restore(self):
        cal = CalibrationData()
        cal.set_channel("ua", "read", 1.05, -2.0)
        snap = cal.snapshot()
        cal.set_channel("ua", "read", 1.10, -5.0)
        self.assertAlmostEqual(cal.get_channel("ua", "read").gain, 1.10)
        cal.restore(snap)
        self.assertAlmostEqual(cal.get_channel("ua", "read").gain, 1.05)

    def test_snapshot_includes_meter_accuracy(self):
        cal = CalibrationData()
        cal.meter_accuracy_pct["ua"] = 0.3
        snap = cal.snapshot()
        cal.meter_accuracy_pct["ua"] = 2.0
        cal.restore(snap)
        self.assertAlmostEqual(cal.meter_accuracy_pct["ua"], 0.3)


class TestIaRangeSplit(unittest.TestCase):
    """Ia has two hardware ranges: ia_low (<17mA) and ia_high (>=17mA)."""

    def test_default_channels_have_ia_low_and_high(self):
        cal = CalibrationData()
        self.assertIn("ia_low_read", cal.channels)
        self.assertIn("ia_high_read", cal.channels)
        self.assertNotIn("ia_read", cal.channels)

    def test_apply_read_selects_low_range(self):
        cal = CalibrationData()
        cal.set_channel("ia_low", "read", 1.05, -0.1)
        cal.set_channel("ia_high", "read", 0.98, 0.5)
        result = cal.apply_read("ia", 10.0)
        self.assertAlmostEqual(result, 10.0 * 1.05 - 0.1)

    def test_apply_read_selects_high_range(self):
        cal = CalibrationData()
        cal.set_channel("ia_low", "read", 1.05, -0.1)
        cal.set_channel("ia_high", "read", 0.98, 0.5)
        result = cal.apply_read("ia", 50.0)
        self.assertAlmostEqual(result, 50.0 * 0.98 + 0.5)

    def test_threshold_boundary(self):
        cal = CalibrationData()
        cal.set_channel("ia_low", "read", 2.0, 0.0)
        cal.set_channel("ia_high", "read", 3.0, 0.0)
        below = cal.apply_read("ia", IA_RANGE_THRESHOLD - 0.01)
        at_threshold = cal.apply_read("ia", IA_RANGE_THRESHOLD)
        self.assertAlmostEqual(below, (IA_RANGE_THRESHOLD - 0.01) * 2.0)
        self.assertAlmostEqual(at_threshold, IA_RANGE_THRESHOLD * 3.0)

    def test_default_ia_transparent(self):
        cal = CalibrationData()
        self.assertAlmostEqual(cal.apply_read("ia", 5.0), 5.0)
        self.assertAlmostEqual(cal.apply_read("ia", 100.0), 100.0)


class TestMeterAccuracy(unittest.TestCase):
    def test_default_values(self):
        cal = CalibrationData()
        self.assertAlmostEqual(cal.meter_accuracy_pct["ua"], 0.5)
        self.assertAlmostEqual(cal.meter_accuracy_pct["ia_low"], 1.0)

    def test_meter_error(self):
        cal = CalibrationData()
        cal.meter_accuracy_pct["ua"] = 0.5
        err = cal.meter_error("ua", 200.0)
        self.assertAlmostEqual(err, 1.0)

    def test_meter_error_negative_reading(self):
        cal = CalibrationData()
        cal.meter_accuracy_pct["ug1"] = 0.5
        err = cal.meter_error("ug1", -10.0)
        self.assertAlmostEqual(err, 0.05)


class TestTwoPointCalibration(unittest.TestCase):
    def test_identity(self):
        gain, offset = CalibrationData.compute_two_point(50.0, 250.0, 50.0, 250.0)
        self.assertAlmostEqual(gain, 1.0)
        self.assertAlmostEqual(offset, 0.0)

    def test_gain_error(self):
        gain, offset = CalibrationData.compute_two_point(48.7, 251.3, 51.0, 249.0)
        self.assertAlmostEqual(gain, (251.3 - 48.7) / (249.0 - 51.0))
        expected_offset = 48.7 - 51.0 * gain
        self.assertAlmostEqual(offset, expected_offset)

    def test_roundtrip(self):
        gain, offset = CalibrationData.compute_two_point(48.7, 251.3, 51.0, 249.0)
        self.assertAlmostEqual(51.0 * gain + offset, 48.7, places=5)
        self.assertAlmostEqual(249.0 * gain + offset, 251.3, places=5)

    def test_identical_readings_raises(self):
        with self.assertRaises(ValueError):
            CalibrationData.compute_two_point(100, 200, 150, 150)


class TestCurrentTwoPoint(unittest.TestCase):
    def test_10k_resistor(self):
        r = 10_000
        v_low, v_high = 99.5, 199.2
        exp_low = v_low / r * 1000
        exp_high = v_high / r * 1000
        dev_low = exp_low * 1.02
        dev_high = exp_high * 1.02

        gain, offset = CalibrationData.compute_current_two_point(
            r, v_low, v_high, dev_low, dev_high)
        corrected_low = dev_low * gain + offset
        corrected_high = dev_high * gain + offset
        self.assertAlmostEqual(corrected_low, exp_low, places=3)
        self.assertAlmostEqual(corrected_high, exp_high, places=3)


class TestSetTwoPoint(unittest.TestCase):
    def test_identity(self):
        gain, offset = CalibrationData.compute_set_two_point(50, 50, 250, 250)
        self.assertAlmostEqual(gain, 1.0)
        self.assertAlmostEqual(offset, 0.0)

    def test_correction(self):
        gain, offset = CalibrationData.compute_set_two_point(50, 48.7, 250, 251.3)
        corrected_200 = 200 * gain + offset
        hw_gain = (251.3 - 48.7) / (250 - 50)
        hw_offset = 48.7 - 50 * hw_gain
        actual = corrected_200 * hw_gain + hw_offset
        self.assertAlmostEqual(actual, 200.0, places=3)


class TestZeroOffset(unittest.TestCase):
    def test_zero(self):
        offset = CalibrationData.compute_zero_offset(0.08)
        self.assertAlmostEqual(offset, -0.08)


class TestPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        cal = CalibrationData()
        cal.set_channel("ua", "read", 1.05, -2.0)
        cal.set_channel("ig2", "read", 0.99, -0.08)
        cal.set_channel("ia_low", "read", 1.01, -0.05,
                        quality={"point_spread": 8.5})
        cal.meter_accuracy_pct["ua"] = 0.3

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = Path(f.name)

        cal.save(path)
        loaded = CalibrationData.load(path)

        ua_cc = loaded.get_channel("ua", "read")
        self.assertAlmostEqual(ua_cc.gain, 1.05)
        self.assertAlmostEqual(ua_cc.offset, -2.0)
        self.assertIsNotNone(ua_cc.calibrated_at)

        ig2_cc = loaded.get_channel("ig2", "read")
        self.assertAlmostEqual(ig2_cc.gain, 0.99)
        self.assertAlmostEqual(ig2_cc.offset, -0.08)

        ia_low_cc = loaded.get_channel("ia_low", "read")
        self.assertAlmostEqual(ia_low_cc.gain, 1.01)
        self.assertIsNotNone(ia_low_cc.quality)
        self.assertAlmostEqual(ia_low_cc.quality["point_spread"], 8.5)

        self.assertTrue(loaded.get_channel("ug1", "read").is_default())

        self.assertAlmostEqual(loaded.meter_accuracy_pct["ua"], 0.3)
        self.assertAlmostEqual(loaded.meter_accuracy_pct["ig2"], 1.0)

        path.unlink()

    def test_load_missing_file(self):
        cal = CalibrationData.load(Path("/nonexistent/calibration.json"))
        self.assertEqual(cal.calibrated_channels_count(), 0)

    def test_load_corrupt_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("{not valid json")
            path = Path(f.name)
        cal = CalibrationData.load(path)
        self.assertEqual(cal.calibrated_channels_count(), 0)
        path.unlink()

    def test_load_corrupt_value_degrades_channel_not_crash(self):
        """ML-100: valid JSON with a non-numeric gain (hand-edited or
        partially written) must degrade that channel to identity WITH a
        warning — it used to raise ValueError out of load() and crash the
        app at startup (load_calibration in MainWindow.__init__)."""
        data = {
            "version": 2,
            "channels": {
                "ua_read": {"gain": "oops", "offset": 0.0},
                "ug2_read": {"gain": 1.05, "offset": -0.3},
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False,
                                         mode="w") as f:
            json.dump(data, f)
            path = Path(f.name)
        cal = CalibrationData.load(path)   # must not raise
        # corrupt channel -> identity default
        self.assertTrue(cal.channels["ua_read"].is_default())
        # healthy sibling channel still loads
        self.assertAlmostEqual(cal.channels["ug2_read"].gain, 1.05)
        path.unlink()

    def test_save_is_atomic_replace(self):
        """ML-099: save() must go through tmp + os.replace — a crash
        mid-write must never leave a truncated calibration.json (the single
        source of every feedforward coefficient)."""
        from unittest.mock import patch
        cal = CalibrationData()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "calibration.json"
            with patch("lm19.calibration.os.replace",
                       side_effect=os.replace) as repl:
                cal.save(path)
            repl.assert_called_once()
            src_arg, dst_arg = repl.call_args[0]
            self.assertEqual(Path(dst_arg), path)
            self.assertNotEqual(Path(src_arg), path)  # wrote to tmp first
            self.assertTrue(path.exists())
            # no leftover tmp file
            self.assertEqual(list(Path(d).glob("*.tmp")), [])

    def test_migrate_v1_ia_read(self):
        """Loading a v1 file with ia_read should migrate to ia_low_read + ia_high_read."""
        v1_data = {
            "version": 1,
            "channels": {
                "ua_read": {"gain": 1.0, "offset": 0.0, "calibrated_at": None},
                "ia_read": {"gain": 1.03, "offset": -0.1, "calibrated_at": "2025-01-01T00:00:00"},
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(v1_data, f)
            path = Path(f.name)

        cal = CalibrationData.load(path)
        ia_low = cal.get_channel("ia_low", "read")
        ia_high = cal.get_channel("ia_high", "read")
        self.assertAlmostEqual(ia_low.gain, 1.03)
        self.assertAlmostEqual(ia_low.offset, -0.1)
        self.assertAlmostEqual(ia_high.gain, 1.03)
        self.assertAlmostEqual(ia_high.offset, -0.1)

        path.unlink()

    def test_save_version_2(self):
        cal = CalibrationData()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = Path(f.name)
        cal.save(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(raw["version"], 2)
        self.assertIn("meter_accuracy_pct", raw)
        self.assertIn("ia_low_read", raw["channels"])
        self.assertIn("ia_high_read", raw["channels"])
        self.assertNotIn("ia_read", raw["channels"])
        path.unlink()


# ── Plan B (docs/CALIBRATION_PLAN.md): SET derived from READ + feedforward ──


class TestReadInverse(unittest.TestCase):
    """read_inverse(channel, value) — inverse of apply_read.

    Used as verify_target in the feedforward settle flow: the device
    reading expected when the physical value equals the target.
    """

    def test_default_identity(self):
        cal = CalibrationData()
        self.assertAlmostEqual(cal.read_inverse("ua", 200.0), 200.0)

    def test_roundtrip_with_gain_offset(self):
        cal = CalibrationData()
        cal.set_channel("ua", "read", 1.02, -3.0)
        x = 200.0
        self.assertAlmostEqual(
            cal.apply_read("ua", cal.read_inverse("ua", x)), x, places=9)
        self.assertAlmostEqual(
            cal.read_inverse("ua", cal.apply_read("ua", x)), x, places=9)

    def test_ug1_negative_domain(self):
        cal = CalibrationData()
        cal.set_channel("ug1", "read", 0.99, 0.02)
        v = -6.0
        inv = cal.read_inverse("ug1", v)
        self.assertLess(inv, 0.0)
        self.assertAlmostEqual(cal.apply_read("ug1", inv), v, places=9)

    def test_ia_raises(self):
        # Currents are never commanded; range selection is ambiguous in
        # the physical domain — read_inverse("ia") must refuse loudly.
        cal = CalibrationData()
        with self.assertRaises(ValueError):
            cal.read_inverse("ia", 10.0)

    def test_zero_gain_raises(self):
        cal = CalibrationData()
        cal.set_channel("ua", "read", 0.0, 0.0)
        with self.assertRaises(ValueError):
            cal.read_inverse("ua", 100.0)


class TestDeriveSetFromRead(unittest.TestCase):
    """derive_set_two_point — SET coefficients from (commanded, device
    reading) pairs through the READ calibration; no multimeter for SET.

    Hardware model used by the simulator below:
        DAC: physical = dac_gain * cmd + dac_offset
        ADC: reading r such that apply_read(r) == physical
    """

    @staticmethod
    def _device_reading(cmd, dac_gain, dac_offset, read_gain, read_offset):
        phys = dac_gain * cmd + dac_offset
        return (phys - read_offset) / read_gain

    def _derive(self, cal, channel, c_lo, c_hi, dac_gain, dac_offset):
        ch = cal.get_channel(channel, "read")
        r_lo = self._device_reading(c_lo, dac_gain, dac_offset, ch.gain, ch.offset)
        r_hi = self._device_reading(c_hi, dac_gain, dac_offset, ch.gain, ch.offset)
        return cal.derive_set_two_point(channel, c_lo, r_lo, c_hi, r_hi)

    def test_ideal_hardware_identity(self):
        cal = CalibrationData()
        gain, offset = self._derive(cal, "ua", 50.0, 250.0, 1.0, 0.0)
        self.assertAlmostEqual(gain, 1.0)
        self.assertAlmostEqual(offset, 0.0)

    def test_pure_dac_gain(self):
        cal = CalibrationData()
        gain, offset = self._derive(cal, "ua", 50.0, 250.0, 1.05, 0.0)
        self.assertAlmostEqual(gain, 1.0 / 1.05)
        self.assertAlmostEqual(offset, 0.0)

    def test_pure_dac_offset(self):
        cal = CalibrationData()
        gain, offset = self._derive(cal, "ua", 50.0, 250.0, 1.0, 3.0)
        self.assertAlmostEqual(gain, 1.0)
        self.assertAlmostEqual(offset, -3.0)

    def test_adc_error_does_not_leak_into_set(self):
        # READ calibration must fully cancel the ADC error: derived SET
        # depends only on the DAC parameters.
        cal = CalibrationData()
        cal.set_channel("ua", "read", 1.02, -1.0)
        gain, offset = self._derive(cal, "ua", 50.0, 250.0, 1.04, 2.0)
        self.assertAlmostEqual(gain, 1.0 / 1.04)
        self.assertAlmostEqual(offset, -2.0 / 1.04)

    def test_round_trip_compensates_dac(self):
        # apply_set with derived coefficients → physical equals desired.
        cal = CalibrationData()
        cal.set_channel("ua", "read", 0.98, 1.5)
        dac_gain, dac_offset = 1.03, -2.0
        gain, offset = self._derive(cal, "ua", 50.0, 250.0, dac_gain, dac_offset)
        cal.set_channel("ua", "set", gain, offset)
        desired = 200.0
        cmd = cal.apply_set("ua", desired)
        self.assertAlmostEqual(dac_gain * cmd + dac_offset, desired, places=6)

    def test_degenerate_commanded_raises(self):
        cal = CalibrationData()
        with self.assertRaises(ValueError):
            cal.derive_set_two_point("ua", 100.0, 99.0, 100.0, 201.0)

    def test_ug1_negative_domain(self):
        cal = CalibrationData()
        dac_gain, dac_offset = 0.98, -0.05
        gain, offset = self._derive(cal, "ug1", -2.0, -20.0, dac_gain, dac_offset)
        cal.set_channel("ug1", "set", gain, offset)
        cmd = cal.apply_set("ug1", -6.0)
        self.assertLess(cmd, 0.0)
        self.assertAlmostEqual(dac_gain * cmd + dac_offset, -6.0, places=6)


class TestApplySetUg1Domain(unittest.TestCase):
    """Canonical Ug1 domain in lm19/ is negative physical volts.

    SET_LIMITS must never clamp a negative bias toward 0 V — that is
    maximum anode current, the dangerous direction.
    """

    def test_limits_are_negative_domain(self):
        self.assertEqual(SET_LIMITS["ug1"], (-24.0, 0.0))

    def test_negative_target_passes_unclamped(self):
        cal = CalibrationData()
        self.assertAlmostEqual(cal.apply_set("ug1", -6.0), -6.0)

    def test_clamp_at_max_bias(self):
        cal = CalibrationData()
        self.assertAlmostEqual(cal.apply_set("ug1", -30.0), -24.0)

    def test_positive_corrected_clamps_to_zero(self):
        # A correction that pushes the command above 0 V must clamp back
        # to 0 V, never command positive grid bias.
        cal = CalibrationData()
        cal.set_channel("ug1", "set", 1.0, 7.0)
        self.assertAlmostEqual(cal.apply_set("ug1", -6.0), 0.0)


class TestLoadValidation(unittest.TestCase):
    """Stored fits outside sanity bounds are reset on load — with plan B
    feedforward a bad coefficient (corrupted or hand-edited file) drives
    real commands, so it must not be silently obeyed."""

    @staticmethod
    def _load(channels):
        data = {"version": 2, "channels": channels}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            return CalibrationData.load(path)
        finally:
            path.unlink()

    def test_out_of_bounds_read_reset(self):
        cal = self._load({
            "ua_read": {"gain": 10.0, "offset": 0.0,
                        "calibrated_at": "2026-06-12T00:00:00"},
        })
        self.assertTrue(cal.get_channel("ua", "read").is_default())

    def test_negative_ug1_set_gain_reset(self):
        # A negative ug1 SET gain would make apply_set("ug1", -6) clamp
        # to 0 V (max anode current) on every command.
        cal = self._load({
            "ug1_set": {"gain": -0.98, "offset": 0.1,
                        "calibrated_at": "2026-01-01T00:00:00"},
        })
        self.assertTrue(cal.get_channel("ug1", "set").is_default())

    def test_sane_fits_survive(self):
        cal = self._load({
            "ua_read": {"gain": 1.02, "offset": -3.0,
                        "calibrated_at": "2026-06-12T00:00:00"},
            "ug1_set": {"gain": 1.02, "offset": -0.05,
                        "calibrated_at": "2026-06-12T00:00:00"},
            "ia_low_read": {"gain": 0.99, "offset": -0.08,
                            "calibrated_at": "2026-06-12T00:00:00"},
        })
        self.assertAlmostEqual(cal.get_channel("ua", "read").gain, 1.02)
        self.assertAlmostEqual(cal.get_channel("ug1", "set").gain, 1.02)
        self.assertAlmostEqual(cal.get_channel("ia_low", "read").gain, 0.99)


class TestSetFeedforwardNotice(unittest.TestCase):
    """Loading a file with non-default SET coefficients logs the plan B
    behaviour-change notice (feedforward now applies them everywhere)."""

    def _load_with_set(self, set_gain):
        import logging
        data = {
            "version": 2,
            "channels": {
                "ua_set": {"gain": set_gain, "offset": 0.0,
                           "calibrated_at": "2026-01-01T00:00:00"},
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            with self.assertLogs("lm19.calibration", level=logging.INFO) as cm:
                CalibrationData.load(path)
            return "\n".join(cm.output)
        finally:
            path.unlink()

    def test_nondefault_set_logs_notice(self):
        out = self._load_with_set(1.02)
        self.assertIn("working-point commands", out)
        self.assertIn("ua_set", out)

    def test_default_set_no_notice(self):
        out = self._load_with_set(1.0)
        self.assertNotIn("working-point commands", out)


class TestIaHwScale(unittest.TestCase):
    def test_constant_value(self):
        self.assertAlmostEqual(IA_HW_SCALE, 0.01)


class TestIaRangeThreshold(unittest.TestCase):
    def test_threshold_value(self):
        self.assertAlmostEqual(IA_RANGE_THRESHOLD, 17.0)


if __name__ == "__main__":
    unittest.main()
