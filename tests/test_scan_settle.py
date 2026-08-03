"""Tests for scan settle, verify, protection detection, and settle comm error retry."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from scan_test_helpers import (
    _make_mock_client, _get_param_default, _make_scan_settings, _make_cal,
    CalibrationData, ScanSettings, ScanRange,
    _set_param_with_settle, run_scan, ProtectionError,
    encode_ug1, decode_ug1,
)


class TestInterruptibleSettle(unittest.TestCase):
    """Settle waits must abort promptly when stop() is true (principle 3)."""

    def test_interruptible_sleep_none_stop_single_sleep(self):
        from lm19.scan.io import _interruptible_sleep
        with patch("time.sleep") as mock_sleep:
            self.assertFalse(_interruptible_sleep(0.25, None))
            mock_sleep.assert_called_once_with(0.25)

    def test_interruptible_sleep_aborts_immediately_when_stop_true(self):
        from lm19.scan.io import _interruptible_sleep
        with patch("time.sleep") as mock_sleep:
            self.assertTrue(_interruptible_sleep(5.0, lambda: True))
            mock_sleep.assert_not_called()  # never slept the 5 s

    def test_interruptible_sleep_full_duration_when_stop_false(self):
        from lm19.scan.io import _interruptible_sleep
        with patch("time.sleep") as mock_sleep:
            self.assertFalse(_interruptible_sleep(0.2, lambda: False))
            self.assertEqual(mock_sleep.call_count, 4)  # 0.2 / 0.05 chunks

    @patch("time.sleep")
    def test_set_param_skips_write_when_stop_true_at_entry(self, mock_sleep):
        """If stop is already true on entry, the setpoint is NOT written: a
        worker flagged to stop must never re-assert an output (e.g. right after
        an emergency zero — readback ~0 != target would otherwise re-send the
        target). Returns the device-domain expected value; the worker discards
        it at its next boundary."""
        client = _make_mock_client()
        actual = _set_param_with_settle(
            client, "Ua", 250.0, 0.0,
            settle_per_volt_s=0.1, settle_base_s=1.0,
            tolerance=1.0, max_retries=3,
            stop=lambda: True,
        )
        self.assertAlmostEqual(actual, 250.0)
        self.assertEqual(client.set_param.call_count, 0)  # no write at all
        client.get_param.assert_not_called()  # verify reads skipped

    @patch("time.sleep")
    def test_set_param_aborts_during_settle_after_write(self, mock_sleep):
        """stop false on entry but true during the settle → the initial set
        HAPPENED (call_count == 1), but the settle wait and verify reads are
        skipped. (Entry-gate passes, in-flight settle catches the cancel.)"""
        client = _make_mock_client()
        calls = {"n": 0}

        def stop() -> bool:
            calls["n"] += 1
            return calls["n"] > 1  # False at entry-gate, True inside settle

        actual = _set_param_with_settle(
            client, "Ua", 250.0, 0.0,
            settle_per_volt_s=0.1, settle_base_s=1.0,
            tolerance=1.0, max_retries=3,
            stop=stop,
        )
        self.assertAlmostEqual(actual, 250.0)
        self.assertEqual(client.set_param.call_count, 1)  # initial set only
        client.get_param.assert_not_called()  # verify reads skipped

    @patch("time.sleep")
    def test_set_param_with_settle_cancel_returns_verify_target(self, mock_sleep):
        """On cancel the function must return the device-domain *verify_target*
        (so the calibrated wrapper's apply_read round-trips to the physical
        target), NOT the command-domain target — otherwise a cancelled SRK set
        with real calibration trips a spurious SrkVerifyError."""
        client = _make_mock_client()
        actual = _set_param_with_settle(
            client, "Ug1", 100.0, 0.0,            # command-domain target
            settle_per_volt_s=0.1, settle_base_s=1.0,
            tolerance=1.0, max_retries=2,
            verify_target=42.0,                   # device-domain (differs)
            stop=lambda: True,
        )
        self.assertAlmostEqual(actual, 42.0)      # expected, not target (100)
        client.get_param.assert_not_called()

    @patch("time.sleep")
    def test_set_param_with_settle_stop_false_verifies(self, mock_sleep):
        """With stop present but false, behaviour is unchanged (settle+verify)."""
        client = _make_mock_client()
        client.get_param.return_value = 250
        actual = _set_param_with_settle(
            client, "Ua", 250.0, 200.0,
            settle_per_volt_s=0.002, settle_base_s=0.15,
            tolerance=1.0, max_retries=2,
            stop=lambda: False,
        )
        self.assertAlmostEqual(actual, 250.0)
        client.get_param.assert_called()  # verification happened


class TestSetParamWithSettle(unittest.TestCase):
    """Tests for the universal _set_param_with_settle function."""

    @patch("time.sleep")
    def test_ua_settle_first_try_ok(self, mock_sleep):
        """Ua reaches target within tolerance on first read — no retry."""
        client = _make_mock_client()
        client.get_param.return_value = 250  # exact match
        actual = _set_param_with_settle(
            client, "Ua", 250.0, 200.0,
            settle_per_volt_s=0.002, settle_base_s=0.15,
            tolerance=1.0, max_retries=2,
        )
        self.assertAlmostEqual(actual, 250.0)
        # Dynamic settle: |250-200|*0.002 + 0.15 = 0.25
        mock_sleep.assert_called_once()
        settle_time = mock_sleep.call_args[0][0]
        self.assertAlmostEqual(settle_time, 0.25, places=3)
        # set_param called once (no retry needed)
        self.assertEqual(client.set_param.call_count, 1)
        # get_param called once for verification
        self.assertEqual(client.get_param.call_count, 1)

    @patch("time.sleep")
    def test_ua_settle_with_retry(self, mock_sleep):
        """Ua misses tolerance on first read, succeeds on retry."""
        client = _make_mock_client()
        # First read: 245 (off by 5, >1.0 tolerance), second read: 250 (ok)
        client.get_param.side_effect = [245, 250]
        actual = _set_param_with_settle(
            client, "Ua", 250.0, 200.0,
            settle_per_volt_s=0.002, settle_base_s=0.15,
            tolerance=1.0, max_retries=2,
        )
        self.assertAlmostEqual(actual, 250.0)
        # set_param: initial + 1 retry = 2
        self.assertEqual(client.set_param.call_count, 2)
        # sleep: initial settle + base settle for retry = 2
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("time.sleep")
    def test_ua_settle_all_retries_exhausted(self, mock_sleep):
        """Ua never reaches tolerance — returns last actual value."""
        client = _make_mock_client()
        # 2 real reads (retries) + 1 setpoint check (protection detection)
        client.get_param.side_effect = [245, 247, 250]
        actual = _set_param_with_settle(
            client, "Ua", 250.0, 200.0,
            settle_per_volt_s=0.002, settle_base_s=0.15,
            tolerance=1.0, max_retries=2,
        )
        # Returns last read value even if outside tolerance
        self.assertAlmostEqual(actual, 247.0)

    @patch("time.sleep")
    def test_ug1_with_encode_decode(self, mock_sleep):
        """Ug1 uses encode_ug1/decode_ug1 — negative physical values."""
        client = _make_mock_client()
        # encode_ug1(-6.0) = 600; real read returns raw=600 → decode_ug1(600) = -6.0
        client.get_param.return_value = 600
        actual = _set_param_with_settle(
            client, "Ug1", -6.0, -2.0,
            settle_per_volt_s=0.02, settle_base_s=0.15,
            tolerance=0.1, max_retries=2,
            encode_fn=encode_ug1, decode_fn=decode_ug1,
        )
        self.assertAlmostEqual(actual, -6.0)
        # set_param should receive raw=600
        client.set_param.assert_called_with("Ug1", 600)
        # Dynamic settle: |-6-(-2)|*0.02 + 0.15 = 0.23
        settle_time = mock_sleep.call_args[0][0]
        self.assertAlmostEqual(settle_time, 0.23, places=3)

    @patch("time.sleep")
    def test_ug2_single_retry(self, mock_sleep):
        """Ug2 with retries=1 — only one read attempt."""
        client = _make_mock_client()
        client.get_param.return_value = 245  # off by 5, >1.0 tolerance
        actual = _set_param_with_settle(
            client, "Ug2", 250.0, 250.0,
            settle_per_volt_s=0.002, settle_base_s=0.15,
            tolerance=1.0, max_retries=1,
        )
        # Returns 245 (no retry with max_retries=1)
        self.assertAlmostEqual(actual, 245.0)
        self.assertEqual(client.set_param.call_count, 1)
        # 1 real read + 1 setpoint check (protection detection)
        self.assertEqual(client.get_param.call_count, 2)

    @patch("time.sleep")
    def test_zero_delta_uses_base_settle(self, mock_sleep):
        """When prev == target, settle = base_s only."""
        client = _make_mock_client()
        client.get_param.side_effect = _get_param_default(100)
        _set_param_with_settle(
            client, "Ua", 100.0, 100.0,
            settle_per_volt_s=0.002, settle_base_s=0.15,
            tolerance=1.0, max_retries=1,
        )
        settle_time = mock_sleep.call_args[0][0]
        self.assertAlmostEqual(settle_time, 0.15, places=3)

    @patch("time.sleep")
    def test_large_delta_settle(self, mock_sleep):
        """Large voltage jump 0→300 yields settle ~0.75s."""
        client = _make_mock_client()
        client.get_param.return_value = 300
        _set_param_with_settle(
            client, "Ua", 300.0, 0.0,
            settle_per_volt_s=0.002, settle_base_s=0.15,
            tolerance=1.0, max_retries=1,
        )
        settle_time = mock_sleep.call_args[0][0]
        # 300 * 0.002 + 0.15 = 0.75
        self.assertAlmostEqual(settle_time, 0.75, places=3)

    @patch("time.sleep")
    def test_tolerance_boundary_exact(self, mock_sleep):
        """Value exactly at tolerance boundary should pass (<=)."""
        client = _make_mock_client()
        # target=250, tolerance=1.0, actual=251 → |251-250|=1.0 <= 1.0 → OK
        client.get_param.return_value = 251
        actual = _set_param_with_settle(
            client, "Ua", 250.0, 250.0,
            settle_per_volt_s=0.002, settle_base_s=0.15,
            tolerance=1.0, max_retries=2,
        )
        self.assertAlmostEqual(actual, 251.0)
        # Should pass on first read — no retry
        self.assertEqual(client.set_param.call_count, 1)
        self.assertEqual(client.get_param.call_count, 1)

    @patch("time.sleep")
    def test_tolerance_boundary_just_over(self, mock_sleep):
        """Value just over tolerance should trigger retry."""
        client = _make_mock_client()
        # target=250, tolerance=1.0, first read=252 (off by 2 > 1.0), retry → 250
        client.get_param.side_effect = [252, 250]
        actual = _set_param_with_settle(
            client, "Ua", 250.0, 250.0,
            settle_per_volt_s=0.002, settle_base_s=0.15,
            tolerance=1.0, max_retries=2,
        )
        self.assertAlmostEqual(actual, 250.0)
        # set_param: initial + retry = 2
        self.assertEqual(client.set_param.call_count, 2)

    @patch("time.sleep")
    def test_ug1_large_jump_settle(self, mock_sleep):
        """Ug1 large jump (-1 → -20) with settle_per_volt_s=0.02."""
        client = _make_mock_client()
        # decode_ug1(1900) = -19.0 (close enough for tolerance 0.1)
        client.get_param.return_value = 2000  # decode_ug1(2000) = -20.0
        actual = _set_param_with_settle(
            client, "Ug1", -20.0, -1.0,
            settle_per_volt_s=0.02, settle_base_s=0.15,
            tolerance=0.1, max_retries=2,
            encode_fn=encode_ug1, decode_fn=decode_ug1,
        )
        self.assertAlmostEqual(actual, -20.0)
        # Settle: |-20-(-1)|*0.02 + 0.15 = 19*0.02 + 0.15 = 0.53
        settle_time = mock_sleep.call_args[0][0]
        self.assertAlmostEqual(settle_time, 0.53, places=3)


class TestVerifyTarget(unittest.TestCase):
    """verify_target — expected device-space reading for the settle verify.

    Plan B (docs/CALIBRATION_PLAN.md): the caller passes
    read_inverse(target_phys); the pipe stays raw and just compares the
    decoded reading against verify_target instead of target.
    """

    @patch("time.sleep")
    def test_verify_target_success_first_shot(self, mock_sleep):
        """Reading matches verify_target (not target) → no retry."""
        client = _make_mock_client()
        client.get_param.return_value = 245  # == verify_target, != target
        actual = _set_param_with_settle(
            client, "Ua", 250.0, 200.0,
            settle_per_volt_s=0.002, settle_base_s=0.15,
            tolerance=1.0, max_retries=2,
            verify_target=245.0,
        )
        # Returns raw decoded reading, no transformation
        self.assertAlmostEqual(actual, 245.0)
        self.assertEqual(client.set_param.call_count, 1)
        self.assertEqual(client.get_param.call_count, 1)

    @patch("time.sleep")
    def test_verify_target_miss_retries(self, mock_sleep):
        """Reading equals target but not verify_target → retry happens."""
        client = _make_mock_client()
        # First read 250 (== target, but 5 off verify_target) → retry → 245
        client.get_param.side_effect = [250, 245]
        actual = _set_param_with_settle(
            client, "Ua", 250.0, 200.0,
            settle_per_volt_s=0.002, settle_base_s=0.15,
            tolerance=1.0, max_retries=2,
            verify_target=245.0,
        )
        self.assertAlmostEqual(actual, 245.0)
        self.assertEqual(client.set_param.call_count, 2)

    @patch("time.sleep")
    def test_none_behaves_as_target(self, mock_sleep):
        """verify_target=None → identical to comparing against target."""
        client = _make_mock_client()
        client.get_param.return_value = 250
        actual = _set_param_with_settle(
            client, "Ua", 250.0, 200.0,
            settle_per_volt_s=0.002, settle_base_s=0.15,
            tolerance=1.0, max_retries=2,
            verify_target=None,
        )
        self.assertAlmostEqual(actual, 250.0)
        self.assertEqual(client.set_param.call_count, 1)

    @patch("time.sleep")
    def test_retry_resends_same_raw(self, mock_sleep):
        """Feedforward contract: retry re-sends the same encoded command."""
        client = _make_mock_client()
        client.get_param.side_effect = [240, 245]
        _set_param_with_settle(
            client, "Ua", 250.0, 200.0,
            settle_per_volt_s=0.002, settle_base_s=0.15,
            tolerance=1.0, max_retries=2,
            verify_target=245.0,
        )
        raws = [c.args[1] for c in client.set_param.call_args_list]
        self.assertEqual(raws[0], raws[1])

    @patch("time.sleep")
    def test_protection_check_independent_of_verify_target(self, mock_sleep):
        """Setpoint protection check stays in raw command space.

        Firmware zeroed the setpoint (OVERIA) → ProtectionError must be
        raised regardless of verify_target.
        """
        client = _make_mock_client()
        # 2 verify reads (both off verify_target) + setpoint check → 0
        client.get_param.side_effect = [100, 100, 0]
        with self.assertRaises(ProtectionError):
            _set_param_with_settle(
                client, "Ua", 250.0, 200.0,
                settle_per_volt_s=0.002, settle_base_s=0.15,
                tolerance=1.0, max_retries=2,
                verify_target=245.0,
            )


class TestRunScanSettle(unittest.TestCase):
    """Integration tests for run_scan with settle/verify."""

    def _make_settings(self, **kwargs):
        """Create minimal ScanSettings for testing."""
        defaults = dict(
            ua=ScanRange(100, 200, 100),  # 2 points: 100, 200
            ug1=ScanRange(-2.0, -1.0, 1.0),  # 2 points: -2.0, -1.0
            ug2=ScanRange(250, 250, 0),  # single value
            uh=6.3, ih=0.0,
            is_triode=True,
            ua_settle_per_volt_s=0.002,
            ua_settle_base_s=0.01,  # tiny for fast test
            ua_tolerance=1.0,
            ua_retries=1,
            ug1_settle_per_volt_s=0.02,
            ug1_settle_base_s=0.01,
            ug1_tolerance=0.1,
            ug1_retries=1,
            ug2_settle_per_volt_s=0.002,
            ug2_settle_base_s=0.01,
            ug2_tolerance=1.0,
            ug2_retries=1,
            ia_samples=1,
            calibration=CalibrationData(),
        )
        defaults.update(kwargs)
        return ScanSettings(**defaults)

    @patch("time.sleep")
    def test_triode_scan_uses_settle(self, mock_sleep):
        """Triode scan calls _set_param_with_settle for Ua and Ug1."""
        client = _make_mock_client()
        # For the initial reads of Ua, Ug1, Ug2 (prev values)
        # then per-point: Ug1 settle verify, Ua settle verify, 7 measurement reads
        # 2 ug1 values × (1 verify + 2 ua × (1 verify + 7 reads))
        # = 2 × (1 + 2 × 8) = 2 × 17 = 34, plus 3 initial = 37
        client.get_param.side_effect = _get_param_default(100)  # all reads return 100 (0 for Er)
        settings = self._make_settings()
        points = run_scan(client, settings)
        # 2 ug1 × 2 ua = 4 points
        self.assertEqual(len(points), 4)
        # Verify sleep was called (dynamic settle)
        self.assertTrue(mock_sleep.call_count > 0)

    @patch("time.sleep")
    def test_pentode_scan_settles_ug2(self, mock_sleep):
        """Pentode scan calls settle for Ug2 as well."""
        # State-tracking mock: Ua reads what was set. Needed so that
        # safe_entry_idx mechanism in independent sweep sees a real
        # bidirectional coverage (not safe_entry=0 collapse).
        client = _make_mock_client()
        state = {"Ua": 0.0, "Ug1": 0.0, "Ug2": 0.0}

        def cap_set(name, value):
            state[name] = float(value)

        def get_s(name, real=False):
            if name == "Er":
                return 0
            if name in ("Ua", "Ug2") and real:
                return state.get(name, 0)
            if name == "Ug1" and real:
                return state.get("Ug1", 0)
            return 100   # Ia, Ig2, Uh, Ih etc.

        client.set_param = MagicMock(side_effect=cap_set)
        client.get_param = MagicMock(side_effect=get_s)
        settings = self._make_settings(
            is_triode=False,
            ug2=ScanRange(200, 250, 50),  # 2 ug2 values
            down_max_step_v=0,  # disable bisection for this test
        )
        points = run_scan(client, settings)
        # 2 ug2 × 2 ug1 × 2 ua = 8 points
        self.assertEqual(len(points), 8)

    @patch("time.sleep")
    def test_track_mode_settles_ug2_per_point(self, mock_sleep):
        """Track mode: Ug2 settled for each Ua point."""
        client = _make_mock_client()
        client.get_param.side_effect = _get_param_default(100)
        settings = self._make_settings(
            is_triode=False,
            ug2_track_ua=True,
            ug2_offset=0.0,
        )
        points = run_scan(client, settings)
        # 2 ug1 × 2 ua = 4 points
        self.assertEqual(len(points), 4)

    @patch("time.sleep")
    def test_ia_samples_in_scan(self, mock_sleep):
        """run_scan passes ia_samples to _read_measurement_point."""
        client = _make_mock_client()
        client.get_param.side_effect = _get_param_default(100)
        settings = self._make_settings(
            ua=ScanRange(100, 100, 0),  # single point
            ug1=ScanRange(-2.0, -2.0, 0),
            ia_samples=3,
        )
        points = run_scan(client, settings)
        self.assertEqual(len(points), 1)
        # Ia should be read 3 times: 3 Ia + 6 other = 9 reads per point
        # Plus: 3 initial reads + 1 Ug1 verify + 1 Ua verify = 5
        # Plus: Ug2 set_param(0) for triode
        # Total get_param calls = 3 + 1 + 1 + 3 + 6 = 14
        # Just verify we got a valid point
        self.assertIn("ia", points[0])

    @patch("time.sleep")
    def test_pa_limit_still_works(self, mock_sleep):
        """Pa exceedance limit should still break Ua loop."""
        client = _make_mock_client()
        client.get_param.side_effect = _get_param_default(100)

        def get_param_side_effect(name, real=False):
            if name == "Ia" and real:
                return 50000  # 50000*0.01=500mA → Pa=50W
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side_effect
        settings = self._make_settings(
            ua=ScanRange(100, 300, 100),  # 3 points
            ug1=ScanRange(-2.0, -2.0, 0),
            pa_max_w=5.0,
            pa_over_pct=10.0,  # limit = 5.5W
        )
        points = run_scan(client, settings)
        # Pa = 100 * 500 / 1000 = 50W >> 5.5W → should break after first point
        self.assertEqual(len(points), 1)

    @patch("time.sleep")
    def test_stop_callback_aborts_scan(self, mock_sleep):
        """Stop callback should abort scan early."""
        client = _make_mock_client()
        client.get_param.side_effect = _get_param_default(100)
        call_count = [0]

        def stop_after_2():
            call_count[0] += 1
            return call_count[0] > 2  # stop after 2 checks

        settings = self._make_settings(
            ua=ScanRange(100, 300, 100),  # 3 ua points
            ug1=ScanRange(-4.0, -1.0, 1.0),  # 4 ug1 points → 12 total
        )
        points = run_scan(client, settings, stop=stop_after_2)
        # Should have fewer than all 12 points
        self.assertLess(len(points), 12)

    @patch("time.sleep")
    def test_progress_callback_called(self, mock_sleep):
        """Progress callback receives point dicts and curve_done events."""
        client = _make_mock_client()
        client.get_param.side_effect = _get_param_default(100)
        events = []
        settings = self._make_settings(
            ua=ScanRange(100, 200, 100),  # 2 ua
            ug1=ScanRange(-2.0, -1.0, 1.0),  # 2 ug1
        )
        run_scan(client, settings, progress=lambda e: events.append(e))
        # 4 measurement points + 2 curve_done events = 6
        point_events = [e for e in events if "ua" in e]
        curve_done_events = [e for e in events if e.get("event") == "curve_done"]
        self.assertEqual(len(point_events), 4)
        self.assertEqual(len(curve_done_events), 2)
        # curve_done should carry ug1 value
        self.assertIn("ug1", curve_done_events[0])

    @patch("time.sleep")
    def test_track_mode_ug2_equals_ua_plus_offset(self, mock_sleep):
        """Track mode: Ug2 target = max(0, Ua + offset)."""
        client = _make_mock_client()
        set_calls = []
        original_set = client.set_param

        def capture_set(name, value, delay=0.05):
            set_calls.append((name, value))

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param.side_effect = _get_param_default(100)

        settings = self._make_settings(
            is_triode=False,
            ug2_track_ua=True,
            ug2_offset=-10.0,
            ua=ScanRange(100, 200, 100),  # Ua: 100, 200
            ug1=ScanRange(-2.0, -2.0, 0),
        )
        run_scan(client, settings)
        # Find Ug2 set_param calls (exclude initial An, Uh setup)
        ug2_sets = [v for n, v in set_calls if n == "Ug2"]
        # For Ua=100: Ug2 = max(0, 100 + (-10)) = 90
        # For Ua=200: Ug2 = max(0, 200 + (-10)) = 190
        self.assertIn(90, ug2_sets)
        self.assertIn(190, ug2_sets)

    @patch("time.sleep")
    def test_track_mode_ug2_clamps_to_zero(self, mock_sleep):
        """Track mode with large negative offset: Ug2 = max(0, ...)."""
        client = _make_mock_client()
        set_calls = []
        _state = {}

        def capture_set(name, value, delay=0.05):
            set_calls.append((name, value))
            _state[name] = value

        def capture_get(name, real=False):
            return _state.get(name, 0)

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=capture_get)

        settings = self._make_settings(
            is_triode=False,
            ug2_track_ua=True,
            ug2_offset=-200.0,  # Ua=50 → Ug2 = max(0, 50-200) = 0
            ua=ScanRange(50, 50, 0),
            ug1=ScanRange(-2.0, -2.0, 0),
        )
        run_scan(client, settings)
        ug2_sets = [v for n, v in set_calls if n == "Ug2"]
        # All Ug2 should be 0 (clamped)
        for v in ug2_sets:
            self.assertEqual(v, 0)


class TestSettleCommErrorRetry(unittest.TestCase):
    """Tests for comm error retry in settle functions (_wrap_settle)."""

    def _make_settings(self, **kwargs):
        defaults = dict(
            ua=ScanRange(100, 200, 100),   # 2 Ua points: 100, 200
            ug1=ScanRange(-2.0, -2.0, 0),  # single Ug1
            ug2=ScanRange(0, 0, 0),
            uh=6.3, ih=0.0,
            is_triode=True,
            ua_settle_per_volt_s=0.0, ua_settle_base_s=0.0,
            ua_tolerance=999.0, ua_retries=1,
            ug1_settle_per_volt_s=0.0, ug1_settle_base_s=0.0,
            ug1_tolerance=999.0, ug1_retries=1,
            ug2_settle_per_volt_s=0.0, ug2_settle_base_s=0.0,
            ug2_tolerance=999.0, ug2_retries=1,
            ia_samples=1, calibration=CalibrationData(),
            comm_retries=2,
        )
        defaults.update(kwargs)
        return ScanSettings(**defaults)

    @patch("time.sleep")
    def test_settle_auto_retry_succeeds(self, _):
        """Comm error during Ua settle, auto-retry succeeds — on_comm_error NOT called."""
        client = _make_mock_client()
        client.flush_input = MagicMock()
        ua_real_calls = [0]
        fail_on_call = 4  # first 3 = initial reads, 4th = first settle

        def get_param_side(name, real=False):
            if name == "Ua" and real:
                ua_real_calls[0] += 1
                if ua_real_calls[0] == fail_on_call:
                    raise ValueError("Unexpected param 0Ua, expected Ua")
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        on_error = MagicMock()
        settings = self._make_settings(comm_retries=2)
        points = run_scan(client, settings, on_comm_error=on_error)
        self.assertEqual(len(points), 2)
        on_error.assert_not_called()
        client.flush_input.assert_called()

    @patch("time.sleep")
    def test_settle_skip_drops_point(self, _):
        """Auto-retries exhausted in settle, user skips — point dropped, scan continues."""
        client = _make_mock_client()
        client.flush_input = MagicMock()
        ua_real_calls = [0]
        skip_done = [False]
        # fail on call 5: init(1) + init_ug1(skip) + init_ug2(skip) + curve_setup_ua(4) + settle_ug1(skip) + per-point_ua(5)
        # Actually: init reads = Ua(1), Ug1(skip), Ug2(skip); then curve setup _settle_ua(2), _settle_ug1(skip);
        # then per-point _settle_ua for Ua=100 → set_param_with_settle reads Ua(3)
        # Let's fail on call 4 (first per-point settle verify read) and let skip clear it
        fail_on_call = 5

        def get_param_side(name, real=False):
            if name == "Ua" and real:
                ua_real_calls[0] += 1
                if ua_real_calls[0] >= fail_on_call and not skip_done[0]:
                    raise ValueError("Unexpected param 0Ua, expected Ua")
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side

        def on_error(message, attempt):
            skip_done[0] = True
            return "skip"

        settings = self._make_settings(comm_retries=0)
        points = run_scan(client, settings, on_comm_error=on_error)
        # One point skipped due to settle error, rest measured
        self.assertGreaterEqual(len(points), 1)
        self.assertLess(len(points), 2)

    @patch("time.sleep")
    def test_settle_abort_raises(self, _):
        """Auto-retries exhausted in settle, user aborts — exception propagated."""
        client = _make_mock_client()
        client.flush_input = MagicMock()
        call_count = [0]

        def get_param_side(name, real=False):
            call_count[0] += 1
            # Let initial reads succeed (first 3 calls), then fail on settle
            if call_count[0] > 3 and name == "Ua" and real:
                raise ValueError("Unexpected param 0Ua, expected Ua")
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        on_error = MagicMock(return_value="abort")
        settings = self._make_settings(comm_retries=0)
        with self.assertRaises(ValueError):
            run_scan(client, settings, on_comm_error=on_error)
        on_error.assert_called_once()


class TestProtectionDetection(unittest.TestCase):
    """Tests for device protection detection in _set_param_with_settle."""

    @patch("time.sleep")
    def test_protection_detected_when_setpoint_zeroed(self, _):
        """Firmware zeros setpoint (OVERIA) → ProtectionError raised."""
        client = _make_mock_client()

        def get_param_side(name, real=False):
            if name == "Ua" and real:
                return 0  # real value: protection forced PWM to 0
            if name == "Ua" and not real:
                return 0  # setpoint: firmware zeroed uaset
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        with self.assertRaises(ProtectionError) as ctx:
            _set_param_with_settle(
                client, "Ua", 250.0, 0.0,
                settle_per_volt_s=0.0, settle_base_s=0.0,
                tolerance=1.0, max_retries=2,
            )
        self.assertIn("Ua", str(ctx.exception))
        self.assertIn("target=250.0", str(ctx.exception))

    @patch("time.sleep")
    def test_no_protection_when_setpoint_matches(self, _):
        """Setpoint preserved (normal settle failure) → returns actual, no exception."""
        client = _make_mock_client()

        def get_param_side(name, real=False):
            if name == "Ua" and real:
                return 240  # real value: just outside tolerance
            if name == "Ua" and not real:
                return 250  # setpoint intact — not protection
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        actual = _set_param_with_settle(
            client, "Ua", 250.0, 0.0,
            settle_per_volt_s=0.0, settle_base_s=0.0,
            tolerance=1.0, max_retries=2,
        )
        self.assertEqual(actual, 240.0)

    @patch("time.sleep")
    def test_no_protection_check_when_target_zero(self, _):
        """Target=0 → no protection check (would false-positive)."""
        client = _make_mock_client()
        client.get_param.return_value = 0
        actual = _set_param_with_settle(
            client, "Ua", 0.0, 250.0,
            settle_per_volt_s=0.0, settle_base_s=0.0,
            tolerance=1.0, max_retries=1,
        )
        self.assertEqual(actual, 0.0)

    @patch("time.sleep")
    def test_protection_in_scan_goes_to_user_dialog(self, _):
        """ProtectionError in settle skips auto-retries, goes to on_comm_error."""
        client = _make_mock_client()
        client.flush_input = MagicMock()
        call_count = [0]

        def get_param_side(name, real=False):
            if name == "Ua" and real:
                call_count[0] += 1
                # Let initial reads pass (first 3), then protection
                if call_count[0] <= 3:
                    return 0
                return 0  # protection: Ua always reads 0
            if name == "Ua" and not real:
                return 0  # setpoint zeroed by firmware
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        on_error = MagicMock(return_value="abort")

        settings = ScanSettings(
            ua=ScanRange(100, 100, 0), ug1=ScanRange(-2.0, -2.0, 0),
            ug2=ScanRange(0, 0, 0), uh=6.3, ih=0.0, is_triode=True,
            ua_settle_per_volt_s=0.0, ua_settle_base_s=0.0,
            ua_tolerance=1.0, ua_retries=1,
            ug1_settle_per_volt_s=0.0, ug1_settle_base_s=0.0,
            ug1_tolerance=999.0, ug1_retries=1,
            ug2_settle_per_volt_s=0.0, ug2_settle_base_s=0.0,
            ug2_tolerance=999.0, ug2_retries=1,
            ia_samples=1, calibration=CalibrationData(), comm_retries=2,
        )
        with self.assertRaises(ProtectionError):
            run_scan(client, settings, on_comm_error=on_error)
        on_error.assert_called_once()
        self.assertIn("protection", on_error.call_args[0][0].lower())


# ── Plan B (docs/CALIBRATION_PLAN.md §5.4): calibrated scan round-trip ──


class _DacErrorScanClient:
    """Stateful run_scan device: Ua DAC + ADC linear error models.

    DAC: physical Ua = dac_gain * commanded + dac_offset.
    ADC: reading r such that apply_read(r) == physical, i.e.
    r = (physical - read_offset) / read_gain (honest by default).
    Other channels ideal.
    """

    def __init__(self, dac_gain=1.04, dac_offset=2.0,
                 read_gain=1.0, read_offset=0.0):
        self.dac_gain = dac_gain
        self.dac_offset = dac_offset
        self.read_gain = read_gain
        self.read_offset = read_offset
        self.state = {"Ua": 0, "Ug1": 0, "Ug2": 0, "Uh": 0, "Ih": 0, "An": 1}
        self.set_calls = []

    def set_param(self, name, value):
        self.set_calls.append((name, value))
        self.state[name] = value

    def get_param(self, name, real=False):
        if name == "Er":
            return 0
        if name == "Ia":
            return 500   # 5.0 mA — constant plate current, value irrelevant
        if name == "Ig2":
            return 0
        if name == "Uh":
            return 63    # 6.3 V — heater present (passes _check_heater)
        if name == "Ih":
            return 30
        if not real:
            return self.state.get(name, 0)
        if name == "Ua":
            phys = self.dac_gain * float(self.state["Ua"]) + self.dac_offset
            return (phys - self.read_offset) / self.read_gain
        return self.state.get(name, 0)

    def is_open(self):
        return True


class TestScanCalibratedRoundTrip(unittest.TestCase):
    """Saved scan points must land on the requested grid in physical volts.

    Today the runner sends the raw target to a DAC with a systematic
    error, so the (honestly measured) points land at the shifted voltage.
    With feedforward (plan B) the command is pre-corrected and the
    points return to the requested grid.
    """

    @patch("time.sleep")
    def test_points_land_on_requested_grid(self, _sleep):
        dac_gain, dac_offset = 1.04, 2.0
        client = _DacErrorScanClient(dac_gain, dac_offset)
        settings = _make_scan_settings(
            ua=ScanRange(100, 200, 100),       # grid: 100, 200
            ug1=ScanRange(-2.0, -2.0, 0),
            calibration=_make_cal(
                channel="ua",
                set_gain=1.0 / dac_gain, set_offset=-dac_offset / dac_gain,
            ),
        )
        points = run_scan(client, settings)
        self.assertEqual(len(points), 2)
        # Without feedforward: 100 → phys 106, 200 → phys 210 (red).
        self.assertAlmostEqual(points[0]["ua"], 100.0, delta=1.5)
        self.assertAlmostEqual(points[1]["ua"], 200.0, delta=1.5)

    @patch("time.sleep")
    def test_points_land_on_grid_with_dac_and_adc_errors(self, _sleep):
        """Composed DAC + ADC errors inside a full scan (plan §5.4):
        feedforward corrects the DAC, READ corrects the ADC, the saved
        points land on the requested grid in physical volts."""
        dac_gain, dac_offset = 1.04, 2.0
        read_gain, read_offset = 1.02, -1.0
        client = _DacErrorScanClient(dac_gain, dac_offset,
                                     read_gain, read_offset)
        settings = _make_scan_settings(
            ua=ScanRange(100, 200, 100),
            ug1=ScanRange(-2.0, -2.0, 0),
            calibration=_make_cal(
                channel="ua",
                set_gain=1.0 / dac_gain, set_offset=-dac_offset / dac_gain,
                read_gain=read_gain, read_offset=read_offset,
            ),
        )
        points = run_scan(client, settings)
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0]["ua"], 100.0, delta=1.5)
        self.assertAlmostEqual(points[1]["ua"], 200.0, delta=1.5)

    @patch("time.sleep")
    def test_prev_init_via_apply_read(self, _sleep):
        """Pin (plan §5.4): the runner initializes prev_* from calibrated
        reads. Observable via the first Ua settle time: prev must be the
        physical value, not the raw decoded reading.

        Device at physical 0 V with READ offset 50 reads raw −50; a raw
        prev would inflate the settle delta from 100 V to 150 V.
        """
        client = _DacErrorScanClient(1.0, 0.0, read_gain=1.0,
                                     read_offset=50.0)
        settings = _make_scan_settings(
            ua=ScanRange(100, 100, 0),
            ug1=ScanRange(-2.0, -2.0, 0),
            ua_settle_per_volt_s=0.01, ua_settle_base_s=0.1,
            calibration=_make_cal(channel="ua", read_offset=50.0),
        )
        run_scan(client, settings)
        settle_times = [c.args[0] for c in _sleep.call_args_list
                        if c.args and c.args[0] > 0.05]
        # |cmd 100 − prev_cmd 0| * 0.01 + 0.1 = 1.1 (raw prev → 1.6)
        self.assertAlmostEqual(settle_times[0], 1.1, places=3)

    @patch("time.sleep")
    def test_triode_ug2_zeroing_stays_raw(self, _sleep):
        """Pin: triode-mode Ug2=0 must remain a raw literal command.

        apply_set(0) = offset could command a non-zero screen voltage —
        zeroing paths bypass the SET calibration by design.
        """
        client = _DacErrorScanClient(dac_gain=1.0, dac_offset=0.0)
        settings = _make_scan_settings(
            ua=ScanRange(100, 100, 0),
            ug1=ScanRange(-2.0, -2.0, 0),
            calibration=_make_cal(channel="ug2", set_gain=1.1, set_offset=5.0),
        )
        run_scan(client, settings)
        self.assertIn(("Ug2", 0), client.set_calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
