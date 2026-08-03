"""Tests for comm error retry, hw error handling, heater recovery, and protection detection."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from scan_test_helpers import (
    _make_mock_client, _get_param_default, _make_scan_settings,
    CalibrationData, ScanSettings, ScanRange,
    _wait_for_err_clear, _restore_heater_and_wait,
    run_scan,
)


class TestCommErrorRetry(unittest.TestCase):
    """Tests for comm error retry logic in run_scan (_read_point wrapper)."""

    def _make_settings(self, **kwargs):
        defaults = dict(
            ua=ScanRange(100, 200, 100),
            ug1=ScanRange(-2.0, -2.0, 0),
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
            comm_retries=2,
        )
        defaults.update(kwargs)
        return ScanSettings(**defaults)

    @patch("time.sleep")
    def test_auto_retry_succeeds_silently(self, _):
        """Comm error on first Ia read, auto-retry succeeds — on_comm_error NOT called."""
        client = _make_mock_client()
        client.flush_input = MagicMock()
        fail_on_ia = [1]

        def get_param_side(name, real=False):
            if name == "Ia" and real and fail_on_ia[0] > 0:
                fail_on_ia[0] -= 1
                raise ValueError("Invalid response: 'I'")
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        on_error = MagicMock()
        settings = self._make_settings(comm_retries=2)
        points = run_scan(client, settings, on_comm_error=on_error)
        self.assertEqual(len(points), 2)
        on_error.assert_not_called()
        client.flush_input.assert_called()

    @patch("time.sleep")
    def test_user_skip_drops_point(self, _):
        """Auto-retries exhausted, user skips — point dropped, scan continues."""
        client = _make_mock_client()
        client.flush_input = MagicMock()
        first_point_done = [False]

        def get_param_side(name, real=False):
            if name == "Ia" and real and not first_point_done[0]:
                raise ValueError("Invalid response: 'I'")
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side

        def on_error(message, attempt):
            first_point_done[0] = True
            return "skip"

        settings = self._make_settings(comm_retries=1)
        points = run_scan(client, settings, on_comm_error=on_error)
        self.assertEqual(len(points), 1)

    @patch("time.sleep")
    def test_user_retry_then_succeeds(self, _):
        """Auto-retries exhausted, user retries — counter resets, next attempt OK."""
        client = _make_mock_client()
        client.flush_input = MagicMock()
        fail_count = [3]

        def get_param_side(name, real=False):
            if name == "Ia" and real and fail_count[0] > 0:
                fail_count[0] -= 1
                raise ValueError("Invalid response: 'I'")
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        on_error_calls = []

        def on_error(message, attempt):
            on_error_calls.append(attempt)
            return "retry"

        settings = self._make_settings(comm_retries=1)
        points = run_scan(client, settings, on_comm_error=on_error)
        self.assertEqual(len(points), 2)
        self.assertEqual(len(on_error_calls), 1)

    @patch("time.sleep")
    def test_user_abort_raises(self, _):
        """Auto-retries exhausted, user aborts — ValueError re-raised."""
        client = _make_mock_client()
        client.flush_input = MagicMock()

        def get_param_side(name, real=False):
            if name == "Ia" and real:
                raise ValueError("Invalid response: 'I'")
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        on_error = MagicMock(return_value="abort")
        settings = self._make_settings(comm_retries=1)
        with self.assertRaises(ValueError):
            run_scan(client, settings, on_comm_error=on_error)
        on_error.assert_called_once()


class TestWaitForErrClear(unittest.TestCase):
    """Tests for _wait_for_err_clear helper."""

    @patch("time.sleep")
    @patch("time.monotonic")
    def test_clears_immediately(self, mock_time, _sleep):
        """Er=0 on first poll → returns 0."""
        mock_time.side_effect = [0.0, 0.1]
        client = _make_mock_client()
        client.get_param.return_value = 0
        result = _wait_for_err_clear(client)
        self.assertEqual(result, 0)

    @patch("time.sleep")
    @patch("time.monotonic")
    def test_clears_after_several_polls(self, mock_time, _sleep):
        """Er non-zero for 3 polls, then 0 → returns 0."""
        mock_time.side_effect = [0.0, 0.3, 0.6, 0.9, 1.2]
        client = _make_mock_client()
        client.get_param.side_effect = [2, 2, 2, 0]
        result = _wait_for_err_clear(client)
        self.assertEqual(result, 0)

    @patch("time.sleep")
    @patch("time.monotonic")
    def test_timeout_returns_last_er(self, mock_time, _sleep):
        """If Er never clears within timeout → returns last non-zero Er."""
        mock_time.side_effect = [0.0, 0.1, 100.0]
        client = _make_mock_client()
        client.get_param.return_value = 4  # OVERIG
        result = _wait_for_err_clear(client)
        self.assertEqual(result, 4)

    # ── Stop-callback: cancel must exit promptly ──────────────────────────
    #
    # ``run_scan`` is invoked from a QThread worker. Without a ``stop``
    # parameter the loop here would block for the full 15 s timeout,
    # ignoring the user's Cancel. With ``stop``, the next loop iteration
    # sees the flag and returns within one
    # ``_ERR_POLL_INTERVAL_S = 0.3 s`` tick.

    @patch("time.sleep")
    @patch("time.monotonic")
    def test_stop_callback_exits_early(self, mock_time, _sleep):
        """When stop() returns True between polls → returns last er
        without waiting full timeout."""
        mock_time.side_effect = [0.0, 0.1, 0.2]
        client = _make_mock_client()
        client.get_param.return_value = 2  # never clears

        # stop() returns False initially, then True
        stop_seq = [False, True]
        def stop():
            return stop_seq.pop(0) if stop_seq else True

        result = _wait_for_err_clear(client, timeout=15.0, stop=stop)
        # Should return non-zero er (cancellation, not success)
        self.assertEqual(result, 2)

    @patch("time.sleep")
    @patch("time.monotonic")
    def test_stop_none_default(self, mock_time, _sleep):
        """stop=None (the default) → wait runs without a cancel check."""
        mock_time.side_effect = [0.0, 0.1, 100.0]
        client = _make_mock_client()
        client.get_param.return_value = 1
        result = _wait_for_err_clear(client, stop=None)
        self.assertEqual(result, 1)

    @patch("time.sleep")
    @patch("time.monotonic")
    def test_stop_never_fires_runs_normally(self, mock_time, _sleep):
        """stop() always False → runs until er clears."""
        mock_time.side_effect = [0.0, 0.3, 0.6, 0.9]
        client = _make_mock_client()
        client.get_param.side_effect = [2, 2, 0]
        result = _wait_for_err_clear(client, stop=lambda: False)
        self.assertEqual(result, 0)

    def test_signature_has_stop_parameter(self):
        """Pin: stop must be in the signature.

        If a refactor removes ``stop`` parameter, the worker thread
        silently reverts to blocking on Cancel for up to 15 s.
        """
        import inspect
        sig = inspect.signature(_wait_for_err_clear)
        self.assertIn("stop", sig.parameters,
                      "_wait_for_err_clear must accept a stop callback; "
                      "without it the worker thread blocks UI on Cancel "
                      "for up to 15 s")
        self.assertIs(sig.parameters["stop"].default, None,
                      "stop must default to None so call sites without "
                      "a cancel callback keep working")


class TestRestoreHeaterAndWait(unittest.TestCase):
    """Tests for _restore_heater_and_wait helper."""

    @patch("time.sleep")
    @patch("time.monotonic")
    def test_heater_reaches_target(self, mock_time, _sleep):
        """Heater reaches Uh target within tolerance → returns."""
        mock_time.side_effect = [0.0, 0.5, 1.0, 1.5]
        client = _make_mock_client()
        client.get_param.side_effect = [
            10,    # Uh raw → 1.0V (far from 6.3)
            0,     # Ih raw → 0.0A
            63,    # Uh raw → 6.3V (at target)
            0,     # Ih raw → 0.0A
        ]
        settings = ScanSettings(
            ua=ScanRange(0, 100, 10), ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(0, 0, 0), uh=6.3, ih=0.0, is_triode=True,
            ua_settle_per_volt_s=0.0, ua_settle_base_s=0.0,
            ua_tolerance=1.0, ua_retries=1,
            ug1_settle_per_volt_s=0.0, ug1_settle_base_s=0.0,
            ug1_tolerance=0.1, ug1_retries=1,
            ug2_settle_per_volt_s=0.0, ug2_settle_base_s=0.0,
            ug2_tolerance=1.0, ug2_retries=1,
            ia_samples=1, calibration=CalibrationData(),
        )
        events = []
        _restore_heater_and_wait(client, settings,
                                 progress=lambda e: events.append(e),
                                 stop=None)
        client.set_param.assert_any_call("Uh", 63)
        restoring = [e for e in events if e.get("event") == "heater_restoring"]
        self.assertGreater(len(restoring), 0)

    @patch("time.sleep")
    @patch("time.monotonic")
    def test_heater_ih_mode(self, mock_time, _sleep):
        """Ih-regulated heater reaches target → returns."""
        mock_time.side_effect = [0.0, 0.5, 1.0, 1.5]
        client = _make_mock_client()
        client.get_param.side_effect = [
            0,     # Uh
            10,    # Ih=0.10A (far)
            0,     # Uh
            30,    # Ih=0.30A (at target)
        ]
        settings = ScanSettings(
            ua=ScanRange(0, 100, 10), ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(0, 0, 0), uh=0.0, ih=0.3, is_triode=True,
            ua_settle_per_volt_s=0.0, ua_settle_base_s=0.0,
            ua_tolerance=1.0, ua_retries=1,
            ug1_settle_per_volt_s=0.0, ug1_settle_base_s=0.0,
            ug1_tolerance=0.1, ug1_retries=1,
            ug2_settle_per_volt_s=0.0, ug2_settle_base_s=0.0,
            ug2_tolerance=1.0, ug2_retries=1,
            ia_samples=1, calibration=CalibrationData(),
        )
        _restore_heater_and_wait(client, settings, progress=None, stop=None)
        client.set_param.assert_any_call("Ih", 30)


class TestRestoreHeaterEdgeCases(unittest.TestCase):
    """Edge cases for _restore_heater_and_wait."""

    def _make_settings(self, **kwargs):
        defaults = dict(
            ua=ScanRange(0, 100, 10), ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(0, 0, 0), uh=0.0, ih=0.0, is_triode=True,
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

    @patch("time.sleep")
    def test_no_heater_returns_immediately(self, mock_sleep):
        """uh=0, ih=0 → no set_param, no polling, returns instantly."""
        client = _make_mock_client()
        settings = self._make_settings(uh=0.0, ih=0.0)
        _restore_heater_and_wait(client, settings, progress=None, stop=None)
        for c in client.set_param.call_args_list:
            self.assertNotIn(c[0][0], ("Uh", "Ih"))
        self.assertEqual(client.get_param.call_count, 0)
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    @patch("time.monotonic")
    def test_timeout_logs_warning(self, mock_time, _sleep):
        """Heater never reaches target → function returns after timeout."""
        mock_time.side_effect = [0.0, 0.5, 100.0]
        client = _make_mock_client()
        client.get_param.side_effect = [10, 0, 10, 0]
        settings = self._make_settings(uh=6.3)
        _restore_heater_and_wait(client, settings, progress=None, stop=None)

    @patch("time.sleep")
    @patch("time.monotonic")
    def test_stop_callback_aborts_wait(self, mock_time, _sleep):
        """Stop callback → exits polling early."""
        mock_time.side_effect = [0.0, 0.5]
        client = _make_mock_client()
        client.get_param.return_value = 0
        settings = self._make_settings(uh=6.3)
        _restore_heater_and_wait(client, settings, progress=None,
                                 stop=lambda: True)
        self.assertEqual(client.get_param.call_count, 0)


class TestHwErrorScan(unittest.TestCase):
    """Integration: hardware error mid-scan triggers recovery and _BreakSweep."""

    def _make_settings(self, **kwargs):
        defaults = dict(
            ua=ScanRange(0, 100, 50),
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

    @patch("time.sleep")
    def test_triode_hw_error_breaks_sweep_preserves_points(self, _):
        """OVERIA at Ua=100 breaks sweep; points at Ua=0 and 50 are preserved."""
        client = _make_mock_client()
        _state = {}
        er_trigger_ua = 100
        err_cleared = [False]

        def capture_set(name, value, delay=0.05):
            _state[name] = value

        def get_side(name, real=False):
            if name == "Er" and not real:
                ua_set = _state.get("Ua", 0)
                if ua_set >= er_trigger_ua and not err_cleared[0]:
                    return 2
                return 0
            if name == "Uh" and real:
                return 63
            if name == "Ih" and real:
                return 30
            if name == "Ua" and real:
                return _state.get("Ua", 0)
            return _state.get(name, 0)

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=get_side)

        events = []

        def on_err(msg, attempt):
            err_cleared[0] = True
            return "retry"

        settings = self._make_settings()
        points = run_scan(client, settings,
                          progress=lambda e: events.append(e),
                          on_comm_error=on_err)

        self.assertEqual(len(points), 2)
        point_uas = sorted(p["ua"] for p in points)
        self.assertEqual(point_uas, [0.0, 50.0])

        hw_events = [e for e in events if isinstance(e, dict)
                     and e.get("event") == "hw_protection"]
        self.assertEqual(len(hw_events), 1)
        cleared_events = [e for e in events if isinstance(e, dict)
                          and e.get("event") == "hw_protection_cleared"]
        self.assertEqual(len(cleared_events), 1)

    @patch("time.sleep")
    def test_hw_error_abort_raises(self, _):
        """User choosing abort on hw error raises RuntimeError."""
        client = _make_mock_client()
        _state = {}

        def capture_set(name, value, delay=0.05):
            _state[name] = value

        def get_side(name, real=False):
            if name == "Er" and not real:
                return 2
            if name == "Uh" and real:
                return 63
            if name == "Ih" and real:
                return 30
            return _state.get(name, 0)

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=get_side)

        settings = self._make_settings()
        with self.assertRaises(RuntimeError):
            run_scan(client, settings,
                     on_comm_error=lambda msg, att: "abort")

    @patch("time.sleep")
    def test_hw_error_no_callback_raises(self, _):
        """Without on_comm_error callback, hw error raises RuntimeError."""
        client = _make_mock_client()
        _state = {}

        def capture_set(name, value, delay=0.05):
            _state[name] = value

        def get_side(name, real=False):
            if name == "Er" and not real:
                return 2
            if name == "Uh" and real:
                return 63
            if name == "Ih" and real:
                return 30
            return _state.get(name, 0)

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=get_side)

        settings = self._make_settings()
        with self.assertRaises(RuntimeError):
            run_scan(client, settings, on_comm_error=None)

    @patch("time.sleep")
    def test_ug2_track_hw_error_preserves_partial(self, _):
        """OVERIA in ug2_track mode preserves points collected before error."""
        client = _make_mock_client()
        _state = {}
        err_cleared = [False]

        def capture_set(name, value, delay=0.05):
            _state[name] = value

        def get_side(name, real=False):
            if name == "Er" and not real:
                ua_set = _state.get("Ua", 0)
                if ua_set >= 100 and not err_cleared[0]:
                    return 2
                return 0
            if name == "Uh" and real:
                return 63
            if name == "Ih" and real:
                return 30
            if name == "Ua" and real:
                return _state.get("Ua", 0)
            if name == "Ug2" and real:
                return _state.get("Ug2", 0)
            return _state.get(name, 0)

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=get_side)

        def on_err(msg, attempt):
            err_cleared[0] = True
            return "retry"

        settings = self._make_settings(
            is_triode=False,
            ug2_track_ua=True,
            ug2_offset=0,
        )
        points = run_scan(client, settings, on_comm_error=on_err)
        self.assertGreater(len(points), 0)
        self.assertTrue(all(p["ua"] < 100 for p in points))


class TestHwErrorSkipIsAbort(unittest.TestCase):
    """on_comm_error returning 'skip' for hw protection should abort scan."""

    @patch("time.sleep")
    def test_skip_on_hw_error_raises(self, _):
        """'skip' decision on hardware protection raises RuntimeError."""
        client = _make_mock_client()
        _state = {}

        def capture_set(name, value, delay=0.05):
            _state[name] = value

        def get_side(name, real=False):
            if name == "Er" and not real:
                return 2
            if name == "Uh" and real:
                return 63
            if name == "Ih" and real:
                return 30
            return _state.get(name, 0)

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=get_side)

        settings = ScanSettings(
            ua=ScanRange(0, 100, 50), ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(0, 0, 0), uh=6.3, ih=0.0, is_triode=True,
            ua_settle_per_volt_s=0.0, ua_settle_base_s=0.0,
            ua_tolerance=1.0, ua_retries=1,
            ug1_settle_per_volt_s=0.0, ug1_settle_base_s=0.0,
            ug1_tolerance=0.1, ug1_retries=1,
            ug2_settle_per_volt_s=0.0, ug2_settle_base_s=0.0,
            ug2_tolerance=1.0, ug2_retries=1,
            ia_samples=1, calibration=CalibrationData(),
        )
        with self.assertRaises(RuntimeError) as ctx:
            run_scan(client, settings,
                     on_comm_error=lambda msg, att: "skip")
        self.assertIn("Hardware protection", str(ctx.exception))


class TestHwErrorIndependentUg2(unittest.TestCase):
    """Hardware error in independent Ug2 bidirectional mode."""

    @patch("time.sleep")
    def test_independent_ug2_hw_error_preserves_partial(self, _):
        """Error during up-sweep in independent Ug2 mode preserves points."""
        client = _make_mock_client()
        _state = {}
        err_cleared = [False]

        def capture_set(name, value, delay=0.05):
            _state[name] = value

        def get_side(name, real=False):
            if name == "Er" and not real:
                ua_set = _state.get("Ua", 0)
                if ua_set >= 200 and not err_cleared[0]:
                    return 2
                return 0
            if name == "Uh" and real:
                return 63
            if name == "Ih" and real:
                return 30
            if name == "Ua" and real:
                return _state.get("Ua", 0)
            if name == "Ug2" and real:
                return _state.get("Ug2", 0)
            return _state.get(name, 0)

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=get_side)

        def on_err(msg, attempt):
            err_cleared[0] = True
            return "retry"

        settings = ScanSettings(
            ua=ScanRange(0, 300, 100),
            ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(150, 150, 0),
            uh=6.3, ih=0.0,
            is_triode=False,
            ua_settle_per_volt_s=0.0, ua_settle_base_s=0.0,
            ua_tolerance=1.0, ua_retries=1,
            ug1_settle_per_volt_s=0.0, ug1_settle_base_s=0.0,
            ug1_tolerance=0.1, ug1_retries=1,
            ug2_settle_per_volt_s=0.0, ug2_settle_base_s=0.0,
            ug2_tolerance=1.0, ug2_retries=1,
            ia_samples=1, calibration=CalibrationData(),
        )
        events = []
        points = run_scan(client, settings,
                          progress=lambda e: events.append(e),
                          on_comm_error=on_err)
        self.assertGreater(len(points), 0)
        hw_ev = [e for e in events if isinstance(e, dict)
                 and e.get("event") == "hw_protection"]
        self.assertEqual(len(hw_ev), 1)
        cd_ev = [e for e in events if isinstance(e, dict)
                 and e.get("event") == "curve_done"]
        self.assertGreater(len(cd_ev), 0)


class TestHwErrorScanContinuation(unittest.TestCase):
    """After hw error recovery, scan continues with remaining curves."""

    @patch("time.sleep")
    def test_triode_continues_after_recovery(self, _):
        """Error on first Ug1 curve; second Ug1 curve measured fully."""
        client = _make_mock_client()
        _state = {}
        err_cleared = [False]
        error_count = [0]

        def capture_set(name, value, delay=0.05):
            _state[name] = value

        def get_side(name, real=False):
            if name == "Er" and not real:
                ua_set = _state.get("Ua", 0)
                ug1_set = _state.get("Ug1", 0)
                if ug1_set == 200 and ua_set >= 100 and error_count[0] == 0:
                    if not err_cleared[0]:
                        return 2
                return 0
            if name == "Uh" and real:
                return 63
            if name == "Ih" and real:
                return 30
            if name == "Ua" and real:
                return _state.get("Ua", 0)
            return _state.get(name, 0)

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=get_side)

        def on_err(msg, attempt):
            err_cleared[0] = True
            error_count[0] += 1
            return "retry"

        settings = ScanSettings(
            ua=ScanRange(0, 100, 50),
            ug1=ScanRange(-2.0, 0.0, 2),
            ug2=ScanRange(0, 0, 0),
            uh=6.3, ih=0.0, is_triode=True,
            ua_settle_per_volt_s=0.0, ua_settle_base_s=0.0,
            ua_tolerance=1.0, ua_retries=1,
            ug1_settle_per_volt_s=0.0, ug1_settle_base_s=0.0,
            ug1_tolerance=0.1, ug1_retries=1,
            ug2_settle_per_volt_s=0.0, ug2_settle_base_s=0.0,
            ug2_tolerance=1.0, ug2_retries=1,
            ia_samples=1, calibration=CalibrationData(),
        )
        events = []
        points = run_scan(client, settings,
                          progress=lambda e: events.append(e),
                          on_comm_error=on_err)
        curve1 = [p for p in points if abs(p["ug1"] - (-2.0)) < 0.5]
        curve2 = [p for p in points if abs(p["ug1"] - 0.0) < 0.5]
        self.assertEqual(len(curve1), 2, f"Curve1 should have 2 points: {curve1}")
        self.assertEqual(len(curve2), 3, f"Curve2 should have all 3 points: {curve2}")
        self.assertEqual(error_count[0], 1)
        cd = [e for e in events if isinstance(e, dict)
              and e.get("event") == "curve_done"]
        self.assertEqual(len(cd), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
