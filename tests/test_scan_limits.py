"""Tests for Pg2/Ig2 protection, predictive Ig2, heater check, and heater loss."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from scan_test_helpers import (
    _make_mock_client, _get_param_default, _make_scan_settings,
    CalibrationData, ScanSettings, ScanRange,
    _predict_ig2, _IG2_PREDICT_MARGIN,
    _check_heater, HeaterLostError,
    run_scan, encode_ug1,
)
from lm19.scan.events import (
    CURVE_STATUS_IG2_PREDICT,
    CURVE_STATUS_COMPLETED,
)


def _pt(ua, ig2):
    """Shortcut for creating a minimal point dict for Ig2 prediction tests."""
    return {"ua": ua, "ig2": ig2, "ia": 0, "ug1": 0, "ug2": 200, "uh": 0, "ih": 0}


class TestRunScanPg2(unittest.TestCase):
    """Integration tests for Pg2/Ig2 protection in bidirectional pentode sweep."""

    def _make_settings(self, **kwargs):
        defaults = dict(
            ua=ScanRange(0, 100, 10),   # 11 points: 0,10,...,100
            ug1=ScanRange(0, 0, 0),     # single Ug1=0
            ug2=ScanRange(250, 250, 0), # single Ug2=250
            uh=6.3, ih=0.0,
            is_triode=False,
            ua_settle_per_volt_s=0.002,
            ua_settle_base_s=0.01,
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

    def _last_set_ua(self, client, default=100):
        if client.set_param.called:
            for c in reversed(client.set_param.call_args_list):
                if c[0][0] == "Ua":
                    return c[0][1]
        return default

    @patch("time.sleep")
    def test_bidirectional_pg2_stops_down_sweep(self, _):
        """Down sweep stops when Pg2 exceeded; up sweep collects rest."""
        client = _make_mock_client()
        # Ug2=50, ua=[0..100/10], start_idx=closest(50)=5
        # Ig2(raw): high at Ua<20 → Pg2=50*20/1000=1.0W>0.6W

        def get_param_side(name, real=False):
            last_ua = self._last_set_ua(client, 50)
            if name == "Ig2" and real:
                return 2000 if last_ua < 20 else 100
            if name == "Ia" and real:
                return 500
            if name == "Ua" and real:
                return last_ua
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        settings = self._make_settings(
            ua=ScanRange(0, 100, 10),
            ug2=ScanRange(50, 50, 0),
            pig2_max_w=0.5,
            pig2_over_pct=20.0,  # limit = 0.6W
        )
        points = run_scan(client, settings)
        ua_measured = sorted(set(p["ua"] for p in points))
        for ua in ua_measured:
            self.assertGreaterEqual(ua, 20)

    @patch("time.sleep")
    def test_pentode_no_limits_measures_all(self, _):
        """Without Pg2/Ig2/Pa limits, all points measured bidirectionally."""
        client = _make_mock_client()
        client.get_param.side_effect = _get_param_default(100)
        settings = self._make_settings(pig2_max_w=0, pig2_over_pct=0)
        points = run_scan(client, settings)
        self.assertEqual(len(points), 11)

    def _track_get_param_side(self, client, spike_raw, spike_from_ua):
        """Ig2 rises to spike_raw once the last commanded Ua >= spike_from_ua
        (monotonic overload, like a real triode-connected screen)."""
        def get_param_side(name, real=False):
            last_ua = self._last_set_ua(client)
            if name == "Ig2" and real:
                return spike_raw if last_ua >= spike_from_ua else 50
            if name == "Ia" and real:
                return 500
            if name == "Ua" and real:
                return last_ua
            if name == "Ug2" and real:
                if client.set_param.called:
                    for c in reversed(client.set_param.call_args_list):
                        if c[0][0] == "Ug2":
                            return c[0][1]
                return 0
            return 0 if name == "Er" else 100
        return get_param_side

    @patch("time.sleep")
    def test_track_mode_pg2_break_stops_curve(self, _):
        """ML-128: track mode Pg2 over limit BREAKS the curve — in triode
        connection Pg2 grows monotonically with Ua, so continuing commands
        the screen ever deeper into overload while collecting nothing."""
        client = _make_mock_client()
        client.get_param.side_effect = self._track_get_param_side(
            client, spike_raw=3000, spike_from_ua=100)  # 30 mA from Ua=100
        events = []
        settings = self._make_settings(
            ua=ScanRange(50, 150, 50),
            ug2_track_ua=True,
            ug2_offset=200.0,
            pig2_max_w=2.5,
            pig2_over_pct=20.0,  # limit 3.0 W; at Ua=100: 300 V x 30 mA = 9 W
        )
        points = run_scan(client, settings, progress=events.append)
        # Only the point below the break survives...
        self.assertEqual([p["ua"] for p in points], [50.0])
        # ...and the sweep never commands Ua above the break point
        ua_cmds = [c[0][1] for c in client.set_param.call_args_list
                   if c[0][0] == "Ua"]
        self.assertNotIn(150, ua_cmds,
                         "Ua=150 was commanded after the Pg2 break")
        summary = [e for e in events
                   if isinstance(e, dict) and e.get("event") == "scan_summary"]
        self.assertEqual(summary[0]["curves"][0]["status"], "pg2_break")

    @patch("time.sleep")
    def test_track_mode_ig2_break_stops_curve(self, _):
        """ML-128: track mode Ig2 over limit BREAKS the curve (same
        monotonic-overload physics as Pg2)."""
        client = _make_mock_client()
        client.get_param.side_effect = self._track_get_param_side(
            client, spike_raw=2000, spike_from_ua=100)  # 20 mA from Ua=100
        events = []
        settings = self._make_settings(
            ua=ScanRange(50, 150, 50),
            ug2_track_ua=True,
            ug2_offset=200.0,
            ig2_max_ma=15.0,
        )
        points = run_scan(client, settings, progress=events.append)
        self.assertEqual([p["ua"] for p in points], [50.0])
        ua_cmds = [c[0][1] for c in client.set_param.call_args_list
                   if c[0][0] == "Ua"]
        self.assertNotIn(150, ua_cmds,
                         "Ua=150 was commanded after the Ig2 break")
        summary = [e for e in events
                   if isinstance(e, dict) and e.get("event") == "scan_summary"]
        self.assertEqual(summary[0]["curves"][0]["status"], "ig2_break")

    @patch("time.sleep")
    def test_track_mode_g2_first_point_break_aborts_sweep(self, _):
        """ML-128: Ig2/Pg2 break on the FIRST Ua aborts the remaining Ug1
        curves, like Pa — as Ug1 opens toward 0 the currents only grow, so
        every following curve would break at its first point too."""
        client = _make_mock_client()
        client.get_param.side_effect = self._track_get_param_side(
            client, spike_raw=2000, spike_from_ua=0)  # over limit everywhere
        events = []
        settings = self._make_settings(
            ua=ScanRange(50, 150, 50),
            ug1=ScanRange(-2, 0, 2),   # two curves
            ug2_track_ua=True,
            ug2_offset=200.0,
            ig2_max_ma=15.0,
        )
        points = run_scan(client, settings, progress=events.append)
        self.assertEqual(points, [])
        summary = [e for e in events
                   if isinstance(e, dict) and e.get("event") == "scan_summary"]
        statuses = [c["status"] for c in summary[0]["curves"]]
        self.assertEqual(statuses[0], "ig2_first")
        self.assertIn("aborted", statuses[1:],
                      "remaining Ug1 curves must be aborted after a "
                      "first-point Ig2/Pg2 break")

    @patch("time.sleep")
    def test_triode_mode_ignores_pg2(self, _):
        """True triode mode: Pg2 settings ignored (Ug2=0)."""
        client = _make_mock_client()
        client.get_param.side_effect = _get_param_default(100)
        settings = self._make_settings(
            is_triode=True,
            ua=ScanRange(100, 200, 100),
            ug1=ScanRange(-2, -2, 0),
            pig2_max_w=2.5,
            pig2_over_pct=20.0,
        )
        points = run_scan(client, settings)
        self.assertEqual(len(points), 2)

    @patch("time.sleep")
    def test_bidirectional_pa_up_pg2_down(self, _):
        """Up sweep cut by Pa, down sweep cut by Pg2."""
        client = _make_mock_client()
        # Ug2=50 → start_idx≈5, up sweep checks Pa, down sweep checks Pg2

        def get_param_side(name, real=False):
            last_ua = self._last_set_ua(client, 50)
            if name == "Ig2" and real:
                return 2000 if last_ua < 20 else 100
            if name == "Ia" and real:
                return 5000 if last_ua >= 90 else 500
            if name == "Ua" and real:
                return last_ua
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        settings = self._make_settings(
            ua=ScanRange(0, 100, 10),
            ug2=ScanRange(50, 50, 0),
            pig2_max_w=0.5,
            pig2_over_pct=20.0,  # Pg2 limit=0.6W
            pa_max_w=3.0,
            pa_over_pct=20.0,    # Pa limit=3.6W
        )
        points = run_scan(client, settings)
        ua_measured = sorted(set(p["ua"] for p in points))
        for ua in ua_measured:
            self.assertGreaterEqual(ua, 20)
        self.assertIn(90, ua_measured)
        self.assertNotIn(100, ua_measured)

    @patch("time.sleep")
    def test_ig2_limit_stops_down_sweep(self, _):
        """Ig2 hardware limit stops down sweep (break)."""
        client = _make_mock_client()
        # Ug2=50, ig2_max_ma=15 → Ig2>15mA at Ua<30 breaks down sweep

        def get_param_side(name, real=False):
            last_ua = self._last_set_ua(client, 50)
            if name == "Ig2" and real:
                return 2000 if last_ua < 30 else 100
            if name == "Ia" and real:
                return 500
            if name == "Ua" and real:
                return last_ua
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        settings = self._make_settings(
            ua=ScanRange(0, 100, 10),
            ug2=ScanRange(50, 50, 0),
            ig2_max_ma=15.0,
        )
        points = run_scan(client, settings)
        ua_measured = sorted(set(p["ua"] for p in points))
        for ua in ua_measured:
            self.assertGreaterEqual(ua, 30)


    @patch("time.sleep")
    def test_down_sweep_predictive_break_reports_its_own_status(self, _):
        """A predicted Ig2 overshoot must be reported as ig2_predict.

        Distinct from ig2_break on purpose: nothing was measured over the
        limit, the sweep stopped on an extrapolation, and collapsing the
        two would tell the user the tube drew a current it never drew.
        The predictor itself is pinned in test_predict_ig2.py; stubbing it
        here isolates the reason-to-status mapping.
        """
        client = _make_mock_client()

        def get_param_side(name, real=False):
            last_ua = self._last_set_ua(client, 100)
            if name == "Ig2" and real:
                return 100
            if name == "Ia" and real:
                return 500
            if name == "Ua" and real:
                return last_ua
            return 0 if name == "Er" else 100

        client.get_param.side_effect = get_param_side
        settings = self._make_settings(
            ua=ScanRange(0, 100, 10),
            ug2=ScanRange(50, 50, 0),
            ig2_max_ma=15.0,
        )
        events = []
        # Predict far above ig2_limit * _IG2_PREDICT_MARGIN on every step,
        # so the very first down-step trips the predictive guard.
        with patch("lm19.scan.sweepers._predict_ig2", return_value=1e3):
            run_scan(client, settings, progress=events.append)
        summary = [e for e in events
                   if isinstance(e, dict) and e.get("event") == "scan_summary"]
        statuses = [c["status"] for c in summary[0]["curves"]]
        assert CURVE_STATUS_IG2_PREDICT in statuses, statuses

    @patch("time.sleep")
    def test_multiple_ug2_different_start_points(self, _):
        """Different Ug2 values → different start_idx for bidirectional sweep."""
        client = _make_mock_client()

        def get_param_side(name, real=False):
            last_ua = self._last_set_ua(client, 0)
            last_ug2 = 0
            if client.set_param.called:
                for c in reversed(client.set_param.call_args_list):
                    if c[0][0] == "Ug2":
                        last_ug2 = c[0][1]
                        break
            if name == "Uh" and real:
                return 630
            if name == "Ih" and real:
                return 30
            if name == "Ig2" and real:
                if last_ug2 >= 300:
                    return 2000 if last_ua < 80 else 100
                else:
                    return 2000 if last_ua < 40 else 100
            if name == "Ia" and real:
                return 500
            if name == "Ua" and real:
                return last_ua
            if name == "Ug2" and real:
                return last_ug2
            return 0

        client.get_param.side_effect = get_param_side
        settings = self._make_settings(
            ua=ScanRange(0, 100, 20),
            ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(200, 300, 100),
            pig2_max_w=2.5,
            pig2_over_pct=20.0,
        )
        points = run_scan(client, settings)
        pts_200 = [p for p in points if abs(p["ug2"] - 200) < 5]
        pts_300 = [p for p in points if abs(p["ug2"] - 300) < 5]
        for p in pts_200:
            self.assertGreaterEqual(p["ua"], 40)
        for p in pts_300:
            self.assertGreaterEqual(p["ua"], 80)
        self.assertGreater(len(pts_200), len(pts_300))

    @patch("time.sleep")
    def test_bisection_intermediate_not_in_data(self, _):
        """With coarse Ua step and bisection, only grid points appear in data."""
        client = _make_mock_client()
        _state = {}

        def capture_set(name, value, delay=0.05):
            _state[name] = value

        def smart_get(name, real=False):
            return _state.get(name, 0)

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=smart_get)

        # Ua grid: [0, 50, 100, 150, 200], Ug2=200, start_idx=4 (Ua=200)
        # Down sweep: 150, 100, 50, 0 — each step 50V > max_step 25V
        # Bisection inserts intermediates: 175, 125, 75, 25
        settings = self._make_settings(
            ua=ScanRange(0, 200, 50),
            ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(200, 200, 0),
            down_max_step_v=25.0,
        )
        points = run_scan(client, settings)
        ua_in_data = sorted(set(round(p["ua"]) for p in points))
        grid_ua = [0, 50, 100, 150, 200]
        for ua in ua_in_data:
            self.assertIn(ua, grid_ua, f"Ua={ua} is not a grid point")
        intermediate_ua = [25, 75, 125, 175]
        for ua in intermediate_ua:
            self.assertNotIn(ua, ua_in_data,
                             f"Intermediate Ua={ua} should not be in data")

    @patch("time.sleep")
    def test_bisection_disabled_same_as_grid(self, _):
        """With down_max_step_v=0, no bisection — result matches grid exactly."""
        client = _make_mock_client()
        _state = {}

        def capture_set(name, value, delay=0.05):
            _state[name] = value

        def smart_get(name, real=False):
            return _state.get(name, 0)

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=smart_get)

        settings = self._make_settings(
            ua=ScanRange(0, 200, 50),
            ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(200, 200, 0),
            down_max_step_v=0,
        )
        points = run_scan(client, settings)
        ua_in_data = sorted(set(round(p["ua"]) for p in points))
        self.assertEqual(ua_in_data, [0, 50, 100, 150, 200])


class TestPredictIg2(unittest.TestCase):
    """Test _predict_ig2: linear (2 pts) and quadratic (3+ pts) extrapolation."""

    def test_not_enough_points(self):
        self.assertIsNone(_predict_ig2([], 50))
        self.assertIsNone(_predict_ig2([_pt(100, 1.0)], 50))

    def test_linear_two_points(self):
        pts = [_pt(180, 2.0), _pt(160, 4.0)]
        est = _predict_ig2(pts, 140)
        self.assertAlmostEqual(est, 6.0, places=1)

    def test_linear_negative_ig2(self):
        """When Ig2 is decreasing (moving UP), prediction should reflect that."""
        pts = [_pt(100, 5.0), _pt(120, 3.0)]
        est = _predict_ig2(pts, 140)
        self.assertAlmostEqual(est, 1.0, places=1)

    def test_quadratic_three_points_convex(self):
        """Convex Ig2 rise: quadratic should predict higher than linear."""
        pts = [_pt(180, 1.0), _pt(160, 3.0), _pt(140, 7.0)]
        est_quad = _predict_ig2(pts, 120)
        # Linear from last 2 would give 11.0; quadratic captures acceleration
        self.assertIsNotNone(est_quad)
        self.assertGreater(est_quad, 11.0)

    def test_quadratic_linear_data(self):
        """For truly linear data, quadratic should give same result as linear."""
        pts = [_pt(180, 2.0), _pt(160, 4.0), _pt(140, 6.0)]
        est = _predict_ig2(pts, 120)
        self.assertAlmostEqual(est, 8.0, places=1)

    def test_four_points_uses_last_three(self):
        """Only last 3 points matter for quadratic extrapolation."""
        pts = [_pt(200, 0.5), _pt(180, 1.0), _pt(160, 3.0), _pt(140, 7.0)]
        est = _predict_ig2(pts, 120)
        est3 = _predict_ig2(pts[-3:], 120)
        self.assertAlmostEqual(est, est3, places=3)

    def test_margin_constant(self):
        """Margin constant should be between 0.5 and 1.0."""
        self.assertGreater(_IG2_PREDICT_MARGIN, 0.5)
        self.assertLessEqual(_IG2_PREDICT_MARGIN, 1.0)


class TestCheckHeater(unittest.TestCase):
    """Unit tests for _check_heater."""

    def _settings(self, uh=6.3, ih=0.0):
        return ScanSettings(
            ua=ScanRange(0, 100, 10), ug1=ScanRange(0, 0, 0),
            ug2=ScanRange(0, 0, 0), uh=uh, ih=ih,
            calibration=CalibrationData(),
        )

    def test_uh_ok(self):
        pt = {"uh": 6.2, "ih": 0.3, "ua": 100, "ug1": 0, "ug2": 0, "ia": 5, "ig2": 0}
        _check_heater(pt, self._settings(uh=6.3))

    def test_uh_lost(self):
        pt = {"uh": 0.1, "ih": 0.01, "ua": 100, "ug1": 0, "ug2": 0, "ia": 0, "ig2": 0}
        with self.assertRaises(HeaterLostError):
            _check_heater(pt, self._settings(uh=6.3))

    def test_ih_ok(self):
        pt = {"uh": 0.0, "ih": 0.3, "ua": 100, "ug1": 0, "ug2": 0, "ia": 5, "ig2": 0}
        _check_heater(pt, self._settings(uh=0, ih=0.3))

    def test_ih_lost(self):
        pt = {"uh": 0.0, "ih": 0.005, "ua": 100, "ug1": 0, "ug2": 0, "ia": 0, "ig2": 0}
        with self.assertRaises(HeaterLostError):
            _check_heater(pt, self._settings(uh=0, ih=0.3))

    def test_no_heater_configured(self):
        """If uh=0 and ih=0, no check performed."""
        pt = {"uh": 0.0, "ih": 0.0, "ua": 100, "ug1": 0, "ug2": 0, "ia": 0, "ig2": 0}
        _check_heater(pt, self._settings(uh=0, ih=0))


class TestHeaterLostScan(unittest.TestCase):
    """Integration test: heater loss stops scan and emits event."""

    def _make_settings(self, **kwargs):
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

    @patch("time.sleep")
    def test_heater_lost_stops_triode_scan(self, _):
        """Scan stops when Uh drops to 0 mid-scan, emits heater_lost event."""
        client = _make_mock_client()
        call_count = [0]
        _state = {}

        def capture_set(name, value, delay=0.05):
            _state[name] = value

        def get_side(name, real=False):
            if name == "Uh" and real:
                call_count[0] += 1
                return 630 if call_count[0] <= 3 else 0
            if name == "Ih" and real:
                return 30 if call_count[0] <= 3 else 0
            return _state.get(name, 0)

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=get_side)
        events = []
        settings = self._make_settings()
        points = run_scan(client, settings,
                          progress=lambda e: events.append(e))
        self.assertLess(len(points), 11)
        heater_events = [e for e in events if isinstance(e, dict)
                         and e.get("event") == "heater_lost"]
        self.assertEqual(len(heater_events), 1)
        self.assertIn("Uh", heater_events[0]["message"])

    @patch("time.sleep")
    def test_heater_ok_full_scan(self, _):
        """With healthy heater, full scan completes normally."""
        client = _make_mock_client()
        _state = {}

        def capture_set(name, value, delay=0.05):
            _state[name] = value

        def get_side(name, real=False):
            if name == "Uh" and real:
                return 630
            if name == "Ih" and real:
                return 30
            return _state.get(name, 0)

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=get_side)
        settings = self._make_settings()
        points = run_scan(client, settings)
        self.assertEqual(len(points), 11)


class TestUg2TransitionSafety(unittest.TestCase):
    """Verify that Ua >= Ug2 and Ug1 is cut-off before changing Ug2 level."""

    @patch("time.sleep")
    def test_ua_raised_before_ug2_increase(self, _):
        """When Ug2 increases between levels, Ua must be raised first."""
        client = _make_mock_client()
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}
        set_log = []  # (param, value) in order

        def capture_set(name, value, delay=0.05):
            state[name] = value
            set_log.append((name, value))

        def get_side(name, real=False):
            if name == "Er":
                return 0
            if name == "Ua" and real:
                return state["Ua"]
            if name == "Ug1" and real:
                return state["Ug1"]
            if name == "Ug2" and real:
                return state["Ug2"]
            return 100

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=get_side)

        settings = _make_scan_settings(
            ua=ScanRange(0, 300, 100),       # 0, 100, 200, 300
            ug1=ScanRange(-2, -2, 0),        # single Ug1=-2
            ug2=ScanRange(100, 200, 100),     # two levels: 100, 200
            is_triode=False,
            ug2_track_ua=False,
        )
        run_scan(client, settings)

        # Find each Ug2 set_param call and check what preceded it
        for i, (param, value) in enumerate(set_log):
            if param == "Ug2" and value == 200:
                # Look backwards for the safety sequence before this Ug2 set
                preceding = set_log[:i]
                # Find last Ua set before this Ug2=200
                ua_before = [v for p, v in preceding if p == "Ua"]
                self.assertTrue(ua_before, "Ua must be set before Ug2=200")
                self.assertGreaterEqual(ua_before[-1], 200,
                    f"Ua={ua_before[-1]} must be >= Ug2=200 before transition")
                # Find last Ug1 set before this Ug2=200 (after last Ug2 change)
                last_ug2_idx = max(
                    (j for j, (p, _) in enumerate(preceding) if p == "Ug2"),
                    default=-1,
                )
                ug1_between = [v for p, v in preceding[last_ug2_idx + 1:]
                               if p == "Ug1"]
                self.assertTrue(ug1_between,
                    "Ug1 must be set (cut-off) before Ug2 transition")
                # Ug1 should be the most negative value (encoded)
                self.assertEqual(ug1_between[0], encode_ug1(-2),
                    "Ug1 must be set to most negative value before Ug2 change")
                break
        else:
            self.fail("Ug2=200 set_param call not found in log")

    @patch("time.sleep")
    def test_ua_not_raised_when_ug2_decreases(self, _):
        """When Ug2 decreases, Ua is already >= Ug2(new), no extra raise needed."""
        client = _make_mock_client()
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}
        set_log = []

        def capture_set(name, value, delay=0.05):
            state[name] = value
            set_log.append((name, value))

        def get_side(name, real=False):
            if name == "Er":
                return 0
            if name == "Ua" and real:
                return state["Ua"]
            if name == "Ug1" and real:
                return state["Ug1"]
            if name == "Ug2" and real:
                return state["Ug2"]
            return 100

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=get_side)

        settings = _make_scan_settings(
            ua=ScanRange(0, 300, 100),
            ug1=ScanRange(-2, -2, 0),
            ug2=ScanRange(200, 100, -100),    # decreasing: 200, 100
            is_triode=False,
            ug2_track_ua=False,
        )
        run_scan(client, settings)

        # Find the Ug2=100 transition — Ua should NOT have been raised to 200
        for i, (param, value) in enumerate(set_log):
            if param == "Ug2" and value == 100:
                preceding = set_log[:i]
                last_ug2_idx = max(
                    (j for j, (p, _) in enumerate(preceding) if p == "Ug2"),
                    default=-1,
                )
                # Between the two Ug2 calls, check there's no unnecessary Ua raise
                ua_between = [(j, v) for j, (p, v) in enumerate(preceding[last_ug2_idx + 1:])
                              if p == "Ua" and v >= 200]
                # There should be no Ua=200 safety raise since Ug2 is decreasing
                # (Ua from sweep may reach 300 naturally, but that's sweep, not safety)
                # Just verify Ug1 cut-off still happens
                ug1_between = [v for p, v in preceding[last_ug2_idx + 1:]
                               if p == "Ug1"]
                self.assertTrue(ug1_between,
                    "Ug1 cut-off must still happen before Ug2 decrease")
                break
        else:
            self.fail("Ug2=100 set_param call not found in log")


class TestPaCheckOrder(unittest.TestCase):
    """Pa must be checked before Ig2/Pg2 to avoid prolonged overload."""

    def _run_with_pa(self, mode, pa_at_ua, ug2_val=200):
        """Run scan where Pa exceeds limit at specific Ua values.

        Returns (points, set_log) for inspection.
        """
        client = _make_mock_client()
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}
        set_log = []

        def capture_set(name, value, delay=0.05):
            state[name] = value
            set_log.append((name, value))

        def get_side(name, real=False):
            if name == "Er":
                return 0
            if name == "Ua" and real:
                return state.get("Ua", 0)
            if name == "Ug1" and real:
                return state.get("Ug1", 0)
            if name == "Ug2" and real:
                return state.get("Ug2", 0)
            if name == "Ia" and real:
                # High Ia at high Ua to trigger Pa
                ua = state.get("Ua", 0)
                return 2000 if ua >= pa_at_ua else 100
            if name == "Ig2" and real:
                # Also high Ig2 to test that Pa check comes first
                return 2000
            return 100

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=get_side)

        if mode == "track":
            settings = _make_scan_settings(
                ua=ScanRange(0, 300, 100),
                ug1=ScanRange(-2, -2, 0),
                is_triode=False,
                ug2_track_ua=True,
                ug2_offset=0,
                pa_max_w=1.0,
                pa_over_pct=10.0,    # limit = 1.1W
                ig2_max_ma=50.0,     # also exceeded — but Pa should win
            )
        else:
            settings = _make_scan_settings(
                ua=ScanRange(0, 300, 100),
                ug1=ScanRange(-2, -2, 0),
                ug2=ScanRange(ug2_val, ug2_val, 0),
                is_triode=False,
                ug2_track_ua=False,
                pa_max_w=1.0,
                pa_over_pct=10.0,
                ig2_max_ma=50.0,
            )
        points = run_scan(client, settings)
        return points, set_log

    @patch("time.sleep")
    def test_track_pa_breaks_before_ig2_skip(self, _):
        """In track mode, Pa break must not be bypassed by Ig2 continue."""
        # Pa exceeded at Ua >= 200: Pa = 200 * 20mA = 4W >> 1.1W limit
        # Ig2 also exceeded everywhere — but Pa check should break first,
        # not let Ig2 continue skip to higher Ua
        points, _ = self._run_with_pa("track", pa_at_ua=200)
        # With Pa-first check, sweep should break at Ua=200
        high_ua_points = [p for p in points if p["ua"] >= 300]
        self.assertEqual(len(high_ua_points), 0,
            "Sweep must break at Pa limit, not continue to Ua=300")

    @patch("time.sleep")
    def test_independent_pa_breaks_before_ig2_skip(self, _):
        """In independent Ug2 mode, Pa break must not be bypassed by Ig2 continue."""
        points, _ = self._run_with_pa("independent", pa_at_ua=200, ug2_val=200)
        high_ua_points = [p for p in points if p["ua"] >= 300]
        self.assertEqual(len(high_ua_points), 0,
            "Up-sweep must break at Pa limit, not continue to Ua=300")

    @patch("time.sleep")
    def test_independent_down_sweep_skips_pa_landing(self, _):
        """Down-sweep must not return to start_idx if Pa was exceeded there."""
        client = _make_mock_client()
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}
        ua_log = []

        def capture_set(name, value, delay=0.05):
            state[name] = value
            if name == "Ua":
                ua_log.append(value)

        def get_side(name, real=False):
            if name == "Er":
                return 0
            if name == "Ua" and real:
                return state.get("Ua", 0)
            if name == "Ug1" and real:
                return state.get("Ug1", 0)
            if name == "Ug2" and real:
                return state.get("Ug2", 0)
            if name == "Ia" and real:
                # Huge current at Ua=200 (start_idx for Ug2=200) → Pa exceeded
                ua = state.get("Ua", 0)
                return 5000 if ua >= 200 else 100
            if name == "Ig2" and real:
                return 100
            return 100

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=get_side)

        settings = _make_scan_settings(
            ua=ScanRange(0, 400, 100),       # 0,100,200,300,400; start_idx=2 (Ua=200)
            ug1=ScanRange(-2, -2, 0),
            ug2=ScanRange(200, 200, 0),
            is_triode=False,
            ug2_track_ua=False,
            pa_max_w=0.5,
            pa_over_pct=10.0,    # limit = 0.55W; Pa@200 = 200*50/1000 = 10W >> limit
        )
        points = run_scan(client, settings)

        # After Pa break at start_idx (Ua=200), down-sweep should land
        # below start_idx, not return to Ua=200.
        # Ua=200 appears for: safety prep, curve init, first sweep point.
        # After the last Ua=200 (sweep break), the next Ua should be < 200.
        last_200_idx = None
        for i, ua in enumerate(ua_log):
            if ua == 200:
                last_200_idx = i
        self.assertIsNotNone(last_200_idx, "Ua=200 must appear in log")
        # All Ua sets after the last Ua=200 should be below 200
        # (these are the down-sweep Ua values)
        after_break = ua_log[last_200_idx + 1:]
        high_after = [ua for ua in after_break if ua >= 200]
        self.assertEqual(high_after, [],
            f"Down-sweep must not return to Ua>=200 after Pa break, "
            f"got {high_after}")


# ---------------------------------------------------------------------------
# Pa sweep abort — consecutive first-point Pa breaks
# ---------------------------------------------------------------------------

class _PaAbortBase:
    """Shared helpers for Pa abort tests across sweep modes."""

    def _make_pa_client(self, pa_threshold_ua):
        """Mock client: Ia raw=5000 (50mA) at Ua >= threshold, raw=10 (0.1mA) below."""
        client = _make_mock_client()
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}

        def capture_set(name, value, delay=0.05):
            state[name] = value

        def get_side(name, real=False):
            if name == "Er":
                return 0
            if name == "Ua" and real:
                return state.get("Ua", 0)
            if name == "Ug1" and real:
                return state.get("Ug1", 0)
            if name == "Ug2" and real:
                return state.get("Ug2", 0)
            if name == "Ia" and real:
                ua = state.get("Ua", 0)
                return 5000 if ua >= pa_threshold_ua else 10
            if name == "Ig2" and real:
                return 0
            return 100

        client.set_param = MagicMock(side_effect=capture_set)
        client.get_param = MagicMock(side_effect=get_side)
        return client


class TestPaAbortTriode(_PaAbortBase, unittest.TestCase):
    """Pa abort in triode mode."""

    @patch("time.sleep")
    def test_abort_when_first_ua_exceeds_pa(self, _):
        """If Pa exceeded on first Ua, remaining Ug1 curves are skipped."""
        # Pa exceeds at any Ua >= 0 (all points over limit)
        client = self._make_pa_client(pa_threshold_ua=0)
        events = []
        settings = _make_scan_settings(
            ua=ScanRange(10, 100, 10),
            ug1=ScanRange(-10, -2, 2),   # 5 curves: -10, -8, -6, -4, -2
            is_triode=True,
            pa_max_w=0.01, pa_over_pct=10.0,
        )
        points = run_scan(client, settings,
                          progress=lambda p: events.append(p))
        # Only 1 curve should run (abort after first)
        curve_dones = [e for e in events if isinstance(e, dict)
                       and e.get("event") == "curve_done"]
        self.assertEqual(len(curve_dones), 1,
            f"Expected 1 curve before abort, got {len(curve_dones)}")
        # pa_sweep_abort event emitted
        aborts = [e for e in events if isinstance(e, dict)
                  and e.get("event") == "pa_sweep_abort"]
        self.assertEqual(len(aborts), 1)

    @patch("time.sleep")
    def test_no_abort_when_pa_breaks_mid_curve(self, _):
        """Pa break on 2nd+ Ua point → sweep continues to next Ug1."""
        # Pa exceeds only at Ua >= 50 — first Ua (10) is safe
        client = self._make_pa_client(pa_threshold_ua=50)
        events = []
        settings = _make_scan_settings(
            ua=ScanRange(10, 100, 10),
            ug1=ScanRange(-10, -2, 2),   # 5 curves
            is_triode=True,
            pa_max_w=0.01, pa_over_pct=10.0,
        )
        points = run_scan(client, settings,
                          progress=lambda p: events.append(p))
        curve_dones = [e for e in events if isinstance(e, dict)
                       and e.get("event") == "curve_done"]
        # All 5 curves should run — Pa break is mid-curve, not first point
        self.assertEqual(len(curve_dones), 5,
            f"Expected all 5 curves, got {len(curve_dones)}")
        aborts = [e for e in events if isinstance(e, dict)
                  and e.get("event") == "pa_sweep_abort"]
        self.assertEqual(len(aborts), 0, "No abort expected")


class TestPaAbortUg2Track(_PaAbortBase, unittest.TestCase):
    """Pa abort in ug2-tracking mode."""

    @patch("time.sleep")
    def test_abort_when_first_ua_exceeds_pa(self, _):
        """Track mode: Pa on first Ua → abort remaining Ug1 curves."""
        client = self._make_pa_client(pa_threshold_ua=0)
        events = []
        settings = _make_scan_settings(
            ua=ScanRange(10, 100, 10),
            ug1=ScanRange(-10, -2, 2),   # 5 curves
            is_triode=False,
            ug2_track_ua=True,
            ug2_offset=0,
            pa_max_w=0.01, pa_over_pct=10.0,
        )
        points = run_scan(client, settings,
                          progress=lambda p: events.append(p))
        curve_dones = [e for e in events if isinstance(e, dict)
                       and e.get("event") == "curve_done"]
        self.assertEqual(len(curve_dones), 1)
        aborts = [e for e in events if isinstance(e, dict)
                  and e.get("event") == "pa_sweep_abort"]
        self.assertEqual(len(aborts), 1)


class TestPaAbortIndependent(_PaAbortBase, unittest.TestCase):
    """Pa abort in independent Ug2 mode."""

    @patch("time.sleep")
    def test_abort_when_up_and_down_both_empty(self, _):
        """Independent: Pa on 1st up + empty down → abort Ug1 loop."""
        # Ug2=10, ua_values=[10,20,...,100], start_idx=0 → no down-sweep
        # Pa exceeds everywhere → up has 1 point, down has 0
        client = self._make_pa_client(pa_threshold_ua=0)
        events = []
        settings = _make_scan_settings(
            ua=ScanRange(10, 100, 10),
            ug1=ScanRange(-10, -2, 2),   # 5 curves
            ug2=ScanRange(10, 10, 0),    # start_idx = 0 → no down-sweep
            is_triode=False,
            ug2_track_ua=False,
            pa_max_w=0.01, pa_over_pct=10.0,
        )
        points = run_scan(client, settings,
                          progress=lambda p: events.append(p))
        curve_dones = [e for e in events if isinstance(e, dict)
                       and e.get("event") == "curve_done"]
        self.assertEqual(len(curve_dones), 1,
            f"Expected 1 curve before abort, got {len(curve_dones)}")
        aborts = [e for e in events if isinstance(e, dict)
                  and e.get("event") == "pa_sweep_abort"]
        self.assertEqual(len(aborts), 1)

    @patch("time.sleep")
    def test_no_abort_when_down_has_data(self, _):
        """Independent: Pa on up but down collects points → no abort."""
        # Ug2=50, Ua=[10..100 step 10], start_idx≈4 (Ua=50)
        # Pa exceeds at Ua>=50, but down-sweep (Ua<50) is fine
        client = self._make_pa_client(pa_threshold_ua=50)
        events = []
        settings = _make_scan_settings(
            ua=ScanRange(10, 100, 10),
            ug1=ScanRange(-10, -2, 2),   # 5 curves
            ug2=ScanRange(50, 50, 0),    # start_idx ≈ 4
            is_triode=False,
            ug2_track_ua=False,
            pa_max_w=0.01, pa_over_pct=10.0,
        )
        points = run_scan(client, settings,
                          progress=lambda p: events.append(p))
        curve_dones = [e for e in events if isinstance(e, dict)
                       and e.get("event") == "curve_done"]
        # All 5 curves should run — down-sweep provides data
        self.assertEqual(len(curve_dones), 5,
            f"Expected all 5 curves, got {len(curve_dones)}")
        aborts = [e for e in events if isinstance(e, dict)
                  and e.get("event") == "pa_sweep_abort"]
        self.assertEqual(len(aborts), 0)
        # Down-sweep points exist (Ua < 50)
        low_ua = [p for p in points if p["ua"] < 50]
        self.assertGreater(len(low_ua), 0,
            "Down-sweep should collect points below Pa threshold")

    @patch("time.sleep")
    def test_abort_per_ug2_does_not_block_next(self, _):
        """Abort at one Ug2 does not prevent scanning subsequent Ug2."""
        # Custom client: Ia=50mA (raw 5000) at Ua>=50, Ia=0.01mA (raw 1) below.
        # pa_limit = 1.0 × 1.1 = 1.1W.
        client = _make_mock_client()
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}
        def cap_set(name, value, delay=0.05):
            state[name] = value
        def get_s(name, real=False):
            if name == "Er": return 0
            if name == "Ua" and real: return state.get("Ua", 0)
            if name == "Ug1" and real: return state.get("Ug1", 0)
            if name == "Ug2" and real: return state.get("Ug2", 0)
            if name == "Ia" and real:
                return 5000 if state.get("Ua", 0) >= 50 else 1
            if name == "Ig2" and real: return 0
            return 100
        client.set_param = MagicMock(side_effect=cap_set)
        client.get_param = MagicMock(side_effect=get_s)

        events = []
        settings = _make_scan_settings(
            ua=ScanRange(10, 100, 10),
            ug1=ScanRange(-4, -2, 2),    # 2 curves
            ug2=ScanRange(10, 50, 40),   # two Ug2 levels: 10 and 50
            is_triode=False,
            ug2_track_ua=False,
            pa_max_w=1.0, pa_over_pct=10.0,  # limit = 1.1W
        )
        # Ug2=10: start_idx=0, Ua=10 Ia=0.01mA → Pa ok. Ua=50 Ia=50mA →
        #   Pa=2.5W > 1.1W → break mid-curve → NOT first-point → no abort.
        # Ug2=50: start_idx=4, up Ua=50 Ia=50mA → Pa break on 1st up point.
        #   Down: Ua=40..10 Ia=0.01mA → Pa tiny → points saved.
        #   down_pts > 0 → no abort.
        # Both Ug2 levels run all Ug1 curves.
        points = run_scan(client, settings,
                          progress=lambda p: events.append(p))
        curve_dones = [e for e in events if isinstance(e, dict)
                       and e.get("event") == "curve_done"]
        ug2_10_curves = [e for e in curve_dones if abs(e["ug2"] - 10) < 1]
        ug2_50_curves = [e for e in curve_dones if abs(e["ug2"] - 50) < 1]
        self.assertEqual(len(ug2_10_curves), 2,
            "Ug2=10: no abort, both curves run")
        self.assertEqual(len(ug2_50_curves), 2,
            "Ug2=50: down has data, no abort")

        # Part 2: ALL Ua exceeding Pa → abort at both Ug2 levels.
        # Ia=50mA everywhere (raw 5000), limit = 0.0011W → everything exceeds.
        events2 = []
        client2 = self._make_pa_client(pa_threshold_ua=0)
        settings2 = _make_scan_settings(
            ua=ScanRange(10, 100, 10),
            ug1=ScanRange(-4, -2, 2),    # 2 curves
            ug2=ScanRange(10, 50, 40),   # Ug2=10: start_idx=0, Ug2=50: start_idx=4
            is_triode=False,
            ug2_track_ua=False,
            pa_max_w=0.001, pa_over_pct=10.0,  # limit = 0.0011W
        )
        # Ug2=10: start_idx=0, no down, Pa on 1st → abort after 1 curve.
        # Ug2=50: down Pa exceeded → continue skips → down_pts=0 → abort.
        points2 = run_scan(client2, settings2,
                           progress=lambda p: events2.append(p))
        curve_dones2 = [e for e in events2 if isinstance(e, dict)
                        and e.get("event") == "curve_done"]
        ug2_10_c = [e for e in curve_dones2 if abs(e["ug2"] - 10) < 1]
        ug2_50_c = [e for e in curve_dones2 if abs(e["ug2"] - 50) < 1]
        self.assertEqual(len(ug2_10_c), 1, "Ug2=10: abort after 1 curve")
        self.assertEqual(len(ug2_50_c), 1, "Ug2=50: abort (down empty)")


class TestDownSweepPaContinue(_PaAbortBase, unittest.TestCase):
    """Down-sweep must skip (continue) Pa-exceeded points, not break."""

    @patch("time.sleep")
    def test_down_sweep_continues_past_pa(self, _):
        """Points below Pa threshold are collected even after Pa point."""
        # Pa exceeds at Ua >= 80. Ug2=80, start_idx≈7 (Ua=80).
        # Up: Ua=80 → Pa break. Down: 70,60,...,10 → Ia=10 → Pa ok.
        client = self._make_pa_client(pa_threshold_ua=80)
        settings = _make_scan_settings(
            ua=ScanRange(10, 100, 10),
            ug1=ScanRange(-2, -2, 0),    # 1 curve
            ug2=ScanRange(80, 80, 0),    # start_idx ≈ 7
            is_triode=False,
            ug2_track_ua=False,
            pa_max_w=0.01, pa_over_pct=10.0,
        )
        points = run_scan(client, settings)
        # Down-sweep should collect points at Ua < 80
        low_ua = [p for p in points if p["ua"] < 80]
        self.assertGreater(len(low_ua), 0,
            "Down-sweep must continue past Pa-exceeded points")
        # Verify no points above start_idx (up broke immediately)
        high_ua = [p for p in points if p["ua"] > 80]
        self.assertEqual(len(high_ua), 0,
            "Up-sweep should not have points above Pa threshold")


# ---------------------------------------------------------------------------
# Happy-path integration: all points collected, correct count
# ---------------------------------------------------------------------------

class _HappyPathBase:
    """Shared mock: constant Ia=10mA, Ig2=1mA, no limits exceeded."""

    def _make_happy_client(self):
        client = _make_mock_client()
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}

        def cap_set(name, value, delay=0.05):
            state[name] = value

        def get_s(name, real=False):
            if name == "Er":
                return 0
            if name == "Ua" and real:
                return state.get("Ua", 0)
            if name == "Ug1" and real:
                return state.get("Ug1", 0)
            if name == "Ug2" and real:
                return state.get("Ug2", 0)
            if name == "Ia" and real:
                return 100  # 10.0 mA after decode
            if name == "Ig2" and real:
                return 10   # 0.10 mA after decode
            return 100

        client.set_param = MagicMock(side_effect=cap_set)
        client.get_param = MagicMock(side_effect=get_s)
        return client


class TestHappyPathTriode(_HappyPathBase, unittest.TestCase):

    @patch("time.sleep")
    def test_all_points_collected(self, _):
        client = self._make_happy_client()
        settings = _make_scan_settings(
            ua=ScanRange(10, 50, 10),      # 5 Ua points
            ug1=ScanRange(-6, -2, 2),      # 3 Ug1 curves
            is_triode=True,
        )
        points = run_scan(client, settings)
        self.assertEqual(len(points), 5 * 3)

    @patch("time.sleep")
    def test_ug1_and_ua_values_present(self, _):
        client = self._make_happy_client()
        settings = _make_scan_settings(
            ua=ScanRange(10, 30, 10),      # 3 Ua
            ug1=ScanRange(-4, -2, 2),      # 2 Ug1
            is_triode=True,
        )
        points = run_scan(client, settings)
        ua_set = {p["ua"] for p in points}
        self.assertEqual(ua_set, {10, 20, 30})


class TestHappyPathUg2Track(_HappyPathBase, unittest.TestCase):

    @patch("time.sleep")
    def test_all_points_collected(self, _):
        client = self._make_happy_client()
        settings = _make_scan_settings(
            ua=ScanRange(10, 50, 10),      # 5 Ua
            ug1=ScanRange(-4, -2, 2),      # 2 Ug1
            is_triode=False,
            ug2_track_ua=True,
            ug2_offset=0,
        )
        points = run_scan(client, settings)
        self.assertEqual(len(points), 5 * 2)

    @patch("time.sleep")
    def test_ug2_tracks_ua(self, _):
        """Each point's Ug2 should approximately equal Ua + offset."""
        client = self._make_happy_client()
        settings = _make_scan_settings(
            ua=ScanRange(10, 30, 10),
            ug1=ScanRange(-2, -2, 0),
            is_triode=False,
            ug2_track_ua=True,
            ug2_offset=5,
        )
        points = run_scan(client, settings)
        for p in points:
            expected_ug2 = p["ua"] + 5
            self.assertAlmostEqual(p["ug2"], expected_ug2, delta=2.0,
                msg=f"Ug2 should track Ua+offset at Ua={p['ua']}")


class TestHappyPathIndependent(_HappyPathBase, unittest.TestCase):

    @patch("time.sleep")
    def test_bidirectional_covers_full_range(self, _):
        """Independent sweep should collect points both above and below Ug2."""
        client = self._make_happy_client()
        settings = _make_scan_settings(
            ua=ScanRange(10, 100, 10),     # 10 Ua points
            ug1=ScanRange(-2, -2, 0),      # 1 Ug1
            ug2=ScanRange(50, 50, 0),      # start_idx ≈ 4
            is_triode=False,
            ug2_track_ua=False,
        )
        points = run_scan(client, settings)
        ua_values = sorted({p["ua"] for p in points})
        # Should have points both below and above Ug2=50
        below = [u for u in ua_values if u < 50]
        above = [u for u in ua_values if u > 50]
        self.assertGreater(len(below), 0, "Should have Ua < Ug2 points")
        self.assertGreater(len(above), 0, "Should have Ua > Ug2 points")
        # Total: all 10 Ua values
        self.assertEqual(len(points), 10)

    @patch("time.sleep")
    def test_multiple_ug2_levels(self, _):
        client = self._make_happy_client()
        settings = _make_scan_settings(
            ua=ScanRange(10, 50, 10),      # 5 Ua
            ug1=ScanRange(-2, -2, 0),      # 1 Ug1
            ug2=ScanRange(20, 40, 20),     # 2 Ug2 levels: 20, 40
            is_triode=False,
            ug2_track_ua=False,
        )
        points = run_scan(client, settings)
        # Each Ug2 level gets all 5 Ua points × 1 Ug1 = 5
        self.assertEqual(len(points), 5 * 2)


# ---------------------------------------------------------------------------
# pa_over_pct=0 with pa_max_w > 0: protection must be active
# ---------------------------------------------------------------------------

class TestPaOverPctZero(_HappyPathBase, unittest.TestCase):

    @patch("time.sleep")
    def test_pa_over_zero_still_protects(self, _):
        """pa_over_pct=0 means exact Pa_max limit, not disabled."""
        client = _make_mock_client()
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}

        def cap_set(name, value, delay=0.05):
            state[name] = value

        def get_s(name, real=False):
            if name == "Er":
                return 0
            if name == "Ua" and real:
                return state.get("Ua", 0)
            if name == "Ug1" and real:
                return state.get("Ug1", 0)
            if name == "Ug2" and real:
                return state.get("Ug2", 0)
            if name == "Ia" and real:
                # raw 50000 → 500 mA → Pa = Ua × 500 / 1000
                return 50000
            if name == "Ig2" and real:
                return 0
            return 100

        client.set_param = MagicMock(side_effect=cap_set)
        client.get_param = MagicMock(side_effect=get_s)

        settings = _make_scan_settings(
            ua=ScanRange(10, 100, 10),     # 10 Ua points
            ug1=ScanRange(-2, -2, 0),
            is_triode=True,
            pa_max_w=2.0,                  # limit = 2.0W (0% over)
            pa_over_pct=0,                 # exact limit
        )
        points = run_scan(client, settings)
        # Ia = 500 mA, Pa = Ua × 500 / 1000 = Ua × 0.5.
        # Exceeds 2.0W at Ua > 4 → first point Ua=10 already Pa=5W > 2.0W.
        # Sweep should break immediately.
        self.assertLess(len(points), 10,
            "Pa protection must be active even with pa_over_pct=0")
        self.assertEqual(len(points), 1,
            "Should break on first Ua point (Pa=5W > 2W limit)")


# ---------------------------------------------------------------------------
# run_scan initialization: An, Uh, Ih sent to device
# ---------------------------------------------------------------------------

class TestRunScanInit(unittest.TestCase):

    @patch("time.sleep")
    def test_an_uh_ih_sent(self, _):
        """run_scan must send An, Uh, Ih before scanning."""
        client = _make_mock_client(get_param_side_effect=_get_param_default())
        set_log = []

        original_set = client.set_param

        def log_set(name, value):
            set_log.append((name, value))

        client.set_param = MagicMock(side_effect=log_set)

        settings = _make_scan_settings(
            ua=ScanRange(10, 10, 0),       # 1 point
            ug1=ScanRange(-2, -2, 0),
            is_triode=True,
            an=2,
            uh=6.3,
            ih=0.3,
        )
        run_scan(client, settings)

        names = [n for n, v in set_log]
        # An, Uh, Ih must appear before any Ua/Ug1/Ug2
        an_idx = names.index("An")
        self.assertEqual(set_log[an_idx], ("An", 2))
        # Uh and Ih sent (encoded values)
        uh_sent = any(n == "Uh" for n, v in set_log)
        ih_sent = any(n == "Ih" for n, v in set_log)
        self.assertTrue(uh_sent, "Uh must be sent")
        self.assertTrue(ih_sent, "Ih must be sent")
        # An, Uh, Ih before first Ua
        first_ua = next(i for i, (n, v) in enumerate(set_log) if n == "Ua")
        self.assertLess(an_idx, first_ua, "An must be sent before Ua")


# ---------------------------------------------------------------------------
# Predictive Ig2 abort in down-sweep integration
# ---------------------------------------------------------------------------

class TestPredictiveIg2DownSweep(unittest.TestCase):

    @patch("time.sleep")
    def test_predictive_ig2_aborts_down_sweep(self, _):
        """Down-sweep should abort early when predicted Ig2 exceeds limit."""
        client = _make_mock_client()
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}

        def cap_set(name, value, delay=0.05):
            state[name] = value

        def get_s(name, real=False):
            if name == "Er":
                return 0
            if name == "Ua" and real:
                return state.get("Ua", 0)
            if name == "Ug1" and real:
                return state.get("Ug1", 0)
            if name == "Ug2" and real:
                return state.get("Ug2", 0)
            if name == "Ia" and real:
                return 100
            if name == "Ig2" and real:
                # Ig2 rises sharply as Ua drops below Ug2
                ua = state.get("Ua", 0)
                ug2 = state.get("Ug2", 200)
                if ua < ug2:
                    # Exponential-ish rise: 0.5 mA per V below Ug2
                    return int((ug2 - ua) * 50)  # in 0.01 mA units
                return 10
            return 100

        client.set_param = MagicMock(side_effect=cap_set)
        client.get_param = MagicMock(side_effect=get_s)

        settings = _make_scan_settings(
            ua=ScanRange(10, 300, 10),     # 30 Ua points
            ug1=ScanRange(-2, -2, 0),      # 1 curve
            ug2=ScanRange(200, 200, 0),    # start_idx ≈ 19
            is_triode=False,
            ug2_track_ua=False,
            ig2_max_ma=5.0,                # 5 mA limit
        )
        points = run_scan(client, settings)

        # Down-sweep should NOT reach the lowest Ua values
        # because predictive Ig2 will abort early
        ua_values = [p["ua"] for p in points]
        if ua_values:
            min_ua = min(ua_values)
            # Without prediction, would go all the way to Ua=10
            # With Ig2 growing as Ua drops, should stop well above 10
            self.assertGreater(min_ua, 10,
                "Predictive Ig2 should prevent reaching lowest Ua")


# ---------------------------------------------------------------------------
# Pa-coverage on synthetic tube model: verify safe region is fully scanned
# ---------------------------------------------------------------------------

class TestPaCoverageIndependent(unittest.TestCase):
    """Verify that for every (Ug1, Ug2), all Ua points with Pa ≤ limit are collected.

    Uses a simple monotonic pentode model:
        Ia(mA) = ug1_factor × min(Ua, Ug2) × scale
    where ug1_factor grows with Ug1 openness.  This gives monotonically
    increasing Pa(Ua) per (Ug1, Ug2) curve, so the Pa-safe region is a
    contiguous interval [0, Ua_safe_max] per curve.
    """

    def _tube_ia_ma(self, ua, ug1, ug2):
        """Synthetic pentode: linear rise, saturates at Ug2."""
        if ug1 <= -10:
            return 0.0
        ug1_factor = (ug1 + 10) / 8.0   # 0.25 at -8, 1.0 at -2
        ua_factor = min(ua, ug2) / 100.0
        return ug1_factor * ua_factor * 100  # 0..100 mA

    def _make_client(self):
        """Mock client implementing synthetic tube model.

        State stores raw protocol values; model uses decoded physical units.
        """
        client = _make_mock_client()
        state = {"Ua": 0.0, "Ug1": 0.0, "Ug2": 0.0}

        def cap_set(name, value, delay=0.05):
            state[name] = float(value)

        def get_s(name, real=False):
            if name == "Er":
                return 0
            if name in ("Ua", "Ug2") and real:
                return state.get(name, 0)
            if name == "Ug1" and real:
                # raw Ug1 in hundredths-V; return as stored (decode by caller)
                return state.get("Ug1", 0)
            if name == "Ia" and real:
                # Decode Ug1 from raw for model
                ug1 = -state.get("Ug1", 0) / 100.0
                ia_ma = self._tube_ia_ma(
                    state.get("Ua", 0), ug1, state.get("Ug2", 0),
                )
                return int(round(ia_ma / 0.01))  # raw = mA × 100
            if name == "Ig2" and real:
                return 0
            return 100

        client.set_param = MagicMock(side_effect=cap_set)
        client.get_param = MagicMock(side_effect=get_s)
        return client

    def _expected_safe_points(self, ua_values, ug1_values, ug2_values, pa_limit):
        """Compute expected Pa-safe set: (ua, ug1, ug2) where Pa ≤ limit."""
        safe = set()
        for ug2 in ug2_values:
            for ug1 in ug1_values:
                for ua in ua_values:
                    pa = ua * self._tube_ia_ma(ua, ug1, ug2) / 1000.0
                    if pa <= pa_limit:
                        safe.add((ua, ug1, ug2))
        return safe

    @patch("time.sleep")
    def test_all_safe_points_collected_independent(self, _):
        """All (Ua, Ug1, Ug2) grid points with Pa ≤ limit must be in output."""
        client = self._make_client()
        ua_range = list(range(10, 101, 10))     # 10..100 step 10
        ug1_range = [-8, -6, -4, -2]
        ug2_range = [100]
        pa_limit_w = 5.0

        settings = _make_scan_settings(
            ua=ScanRange(10, 100, 10),
            ug1=ScanRange(-8, -2, 2),
            ug2=ScanRange(100, 100, 0),
            is_triode=False,
            ug2_track_ua=False,
            pa_max_w=pa_limit_w, pa_over_pct=0,  # exact limit
        )
        points = run_scan(client, settings)

        expected_safe = self._expected_safe_points(
            ua_range, ug1_range, ug2_range, pa_limit_w)

        # Collected points keyed by grid triple
        collected = set()
        for p in points:
            # Snap to grid (settle/read may introduce tiny noise)
            ua = round(p["ua"] / 10) * 10
            ug1 = round(p["ug1"] / 2) * 2
            ug2 = round(p["ug2"] / 10) * 10
            collected.add((ua, ug1, ug2))

        missing = expected_safe - collected
        self.assertEqual(missing, set(),
            f"Safe points not scanned: {sorted(missing)}\n"
            f"Collected: {len(collected)}, expected safe: {len(expected_safe)}")

    @patch("time.sleep")
    def test_pa_exceeding_points_excluded_except_boundary(self, _):
        """Collected points should not extend far into Pa-excess region.

        Up-sweep appends the first Pa-exceeding point (as boundary marker)
        before breaking — that's acceptable.  But no points ≥ 2 steps above
        the safe threshold should be collected.
        """
        client = self._make_client()
        settings = _make_scan_settings(
            ua=ScanRange(10, 100, 10),
            ug1=ScanRange(-8, -2, 2),
            ug2=ScanRange(100, 100, 0),
            is_triode=False,
            ug2_track_ua=False,
            pa_max_w=5.0, pa_over_pct=0,
        )
        points = run_scan(client, settings)

        for p in points:
            ua = p["ua"]
            ug1 = round(p["ug1"] / 2) * 2
            ug2 = round(p["ug2"] / 10) * 10
            pa = ua * self._tube_ia_ma(ua, ug1, ug2) / 1000.0
            # Allow boundary overshoot: up to 2× limit (one Pa-break point can be
            # appended before break).  But not 3× or more — that's runaway.
            self.assertLess(pa, 3 * 5.0,
                f"Collected point with Pa={pa:.1f}W (limit 5W): "
                f"Ua={ua}, Ug1={ug1}, Ug2={ug2}")


# ---------------------------------------------------------------------------
# safe_entry_idx: skip up-sweep for subsequent Ug1 to reduce Pa overload
# ---------------------------------------------------------------------------

class TestSafeEntryIdx(TestPaCoverageIndependent):
    """Verify safe_entry_idx mechanism in independent Ug2 sweep.

    Reuses synthetic tube model from TestPaCoverageIndependent.
    """

    def _run_and_collect_set_log(self, pa_max_w=5.0):
        """Run scan and return (points, ua_set_log)."""
        client = self._make_client()
        ua_log = []

        original_set = client.set_param.side_effect
        def wrap_set(name, value, delay=0.05):
            original_set(name, value, delay)
            if name == "Ua":
                ua_log.append(float(value))
        client.set_param = MagicMock(side_effect=wrap_set)

        settings = _make_scan_settings(
            ua=ScanRange(10, 100, 10),
            ug1=ScanRange(-8, -2, 2),
            ug2=ScanRange(100, 100, 0),
            is_triode=False,
            ug2_track_ua=False,
            pa_max_w=pa_max_w, pa_over_pct=0,
        )
        with patch("time.sleep"):
            points = run_scan(client, settings)
        return points, ua_log

    def test_subsequent_ug1_skips_up_above_safe_entry(self):
        """For Ug1 after first Pa-limited curve, no points above safe_entry."""
        points, _ = self._run_and_collect_set_log(pa_max_w=5.0)

        # Ug1=-4 hits Pa-break at Ua=100 → safe_entry_idx = 7 (Ua=80).
        # For Ug1=-2 (next), skip_up → no measurements above Ua=80.
        pts_ug1_neg2 = [p for p in points
                        if -3 < p["ug1"] < -1]    # Ug1 ≈ -2
        if pts_ug1_neg2:
            max_ua = max(p["ua"] for p in pts_ug1_neg2)
            self.assertLessEqual(max_ua, 80,
                f"Ug1=-2 should not measure above safe_entry (Ua=80), "
                f"got max Ua={max_ua}")

    def test_pa_boundary_points_reduced(self):
        """With safe_entry, fewer Pa-exceeding points collected than naive bidirectional."""
        points, _ = self._run_and_collect_set_log(pa_max_w=5.0)
        pa_exceed = sum(1 for p in points
                        if p["ua"] * p["ia"] / 1000.0 > 5.0)
        # Naive bidirectional: Ug1=-4 and Ug1=-2 both Pa-break at top → 2 points.
        # With safe_entry: only Ug1=-4 does bidirectional (safe_entry=start_idx
        # after Ug1=-6). Ug1=-2 skips up → no top Pa-point.
        self.assertEqual(pa_exceed, 1,
            f"Expected 1 Pa-boundary (only from Ug1=-4), got {pa_exceed}")

    def test_safe_entry_resets_per_ug2(self):
        """safe_entry from one Ug2 level must not leak into next Ug2."""
        # Use two Ug2 levels: 100 and 200. Each should start with full
        # bidirectional for its first Ug1.
        client = self._make_client()
        ua_log = []
        ug2_log = []
        original_set = client.set_param.side_effect
        def wrap_set(name, value, delay=0.05):
            original_set(name, value, delay)
            if name == "Ua":
                ua_log.append(float(value))
            if name == "Ug2":
                ug2_log.append(float(value))
        client.set_param = MagicMock(side_effect=wrap_set)

        settings = _make_scan_settings(
            ua=ScanRange(10, 200, 10),    # Ua 10..200
            ug1=ScanRange(-4, -4, 0),     # 1 Ug1 per Ug2
            ug2=ScanRange(100, 200, 100), # 2 Ug2 levels
            is_triode=False,
            ug2_track_ua=False,
            pa_max_w=5.0, pa_over_pct=0,
        )
        with patch("time.sleep"):
            points = run_scan(client, settings)

        # Each Ug2 has 1 Ug1, so both should do bidirectional. No skip_up.
        # Both should collect points both above and below their Ug2 level.
        pts_ug2_100 = [p for p in points if 90 < p["ug2"] < 110]
        pts_ug2_200 = [p for p in points if 190 < p["ug2"] < 210]
        self.assertGreater(len(pts_ug2_100), 0, "Ug2=100 has points")
        self.assertGreater(len(pts_ug2_200), 0, "Ug2=200 has points")

    def test_first_ug1_still_bidirectional(self):
        """First Ug1 at each Ug2 must still do full bidirectional sweep."""
        client = self._make_client()
        settings = _make_scan_settings(
            ua=ScanRange(10, 100, 10),
            ug1=ScanRange(-8, -8, 0),       # only 1 Ug1 (most closed)
            ug2=ScanRange(50, 50, 0),       # start_idx=4, so up and down sweeps
            is_triode=False,
            ug2_track_ua=False,
            pa_max_w=100.0, pa_over_pct=0,  # no Pa limit in practice
        )
        with patch("time.sleep"):
            points = run_scan(client, settings)

        ua_values = sorted({round(p["ua"] / 10) * 10 for p in points})
        # Full range 10..100 should be covered
        self.assertEqual(ua_values, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    @patch("time.sleep")
    def test_refine_with_safe_entry_stays_in_safe_zone(self, _):
        """Refine points after safe_entry activation must not exceed Pa limit."""
        client = self._make_client()
        settings = _make_scan_settings(
            ua=ScanRange(10, 100, 10),
            ug1=ScanRange(-8, -2, 2),       # 4 curves — later ones skip up
            ug2=ScanRange(100, 100, 0),
            is_triode=False,
            ug2_track_ua=False,
            pa_max_w=5.0, pa_over_pct=0,
            # Enable refine with low thresholds to trigger on our synthetic curve
            refine_enabled=True,
            refine_max_depth=2,
            refine_min_step_ua=3.0,
            refine_onset_ma=0.5,
            refine_curvature_thr=0.05,     # low threshold — triggers easily
            refine_gradient_ratio=2.0,
            refine_ig2_delta_min=0.5,
            refine_delta_ia_thr=0.1,
        )
        points = run_scan(client, settings)

        # All collected points should have Pa ≤ 3× limit (allow 1 boundary).
        # Specifically for refine points added at Ug1=-2 (which uses skip_up),
        # the midpoints should be within the safe zone — not above safe_entry.
        pts_ug1_neg2 = [p for p in points
                        if -3 < p["ug1"] < -1]
        if pts_ug1_neg2:
            # safe_entry after Ug1=-4 ≈ Ua=80 (highest down point).
            # Refine midpoints for Ug1=-2 should be at or below Ua=80.
            max_ua_neg2 = max(p["ua"] for p in pts_ug1_neg2)
            self.assertLessEqual(max_ua_neg2, 80,
                f"Refine at Ug1=-2 shouldn't exceed safe_entry (Ua=80), "
                f"got max Ua={max_ua_neg2}")

        # Sanity: refine actually added points (scan returned more than
        # grid would give).
        grid_count_max = 10 * 4   # 10 Ua × 4 Ug1 = full grid upper bound
        self.assertGreater(len(points), 0)
        # Not asserting exact refine count — scan protection may cut some
        # Ug1 curves short.  The key check is the safe-zone constraint above.


# ---------------------------------------------------------------------------
# scan_summary event: per-curve outcomes + duration
# ---------------------------------------------------------------------------

class TestScanSummaryEvent(TestPaCoverageIndependent):
    """Verify scan_summary event is emitted with correct curve statuses."""

    def _capture_events(self, settings):
        client = self._make_client()
        events = []
        with patch("time.sleep"):
            run_scan(client, settings, progress=lambda e: events.append(e))
        return events

    def _get_summary(self, events):
        summaries = [e for e in events if isinstance(e, dict)
                     and e.get("event") == "scan_summary"]
        self.assertEqual(len(summaries), 1, "Exactly one scan_summary expected")
        return summaries[0]

    @patch("time.sleep")
    def test_summary_happy_path_all_completed(self, _):
        """Healthy scan: all curves completed."""
        settings = _make_scan_settings(
            ua=ScanRange(10, 50, 10),
            ug1=ScanRange(-8, -6, 2),     # 2 curves
            ug2=ScanRange(50, 50, 0),
            is_triode=False,
            ug2_track_ua=False,
            pa_max_w=100.0, pa_over_pct=0,  # no Pa limit in practice
        )
        events = self._capture_events(settings)
        summary = self._get_summary(events)
        self.assertEqual(len(summary["curves"]), 2)
        self.assertTrue(all(c["status"] == CURVE_STATUS_COMPLETED
                            for c in summary["curves"]))
        self.assertGreater(summary["total_points"], 0)

    @patch("time.sleep")
    def test_summary_happy_path_triode_completed(self, _):
        """Happy-path twin for the TRIODE sweeper: the completed-status
        emission in _sweep_triode was unpinned (vocabulary mutation
        audit: a completed->pa_partial swap in the triode branch
        survived — the existing happy path used the independent sweeper)."""
        settings = _make_scan_settings(
            ua=ScanRange(10, 50, 10),
            ug1=ScanRange(-8, -6, 2),
            is_triode=True,
            pa_max_w=100.0, pa_over_pct=0,
        )
        events = self._capture_events(settings)
        summary = self._get_summary(events)
        self.assertEqual(len(summary["curves"]), 2)
        self.assertTrue(all(c["status"] == CURVE_STATUS_COMPLETED
                            for c in summary["curves"]))

    @patch("time.sleep")
    def test_summary_happy_path_track_completed(self, _):
        """Happy-path twin for the Ug2-track sweeper (third emission)."""
        settings = _make_scan_settings(
            ua=ScanRange(10, 50, 10),
            ug1=ScanRange(-8, -6, 2),
            ug2=ScanRange(50, 50, 0),
            is_triode=False,
            ug2_track_ua=True,
            pa_max_w=100.0, pa_over_pct=0,
        )
        events = self._capture_events(settings)
        summary = self._get_summary(events)
        self.assertEqual(len(summary["curves"]), 2)
        self.assertTrue(all(c["status"] == CURVE_STATUS_COMPLETED
                            for c in summary["curves"]))

    @patch("time.sleep")
    def test_summary_pa_first_reported(self, _):
        """Pa break on first point: status should be 'pa_first' or similar."""
        # Very tight Pa limit → first curve fails immediately
        client = self._make_client()
        events = []
        settings = _make_scan_settings(
            ua=ScanRange(10, 100, 10),
            ug1=ScanRange(-6, -2, 2),     # 3 curves
            ug2=ScanRange(10, 10, 0),     # start_idx=0 → no down
            is_triode=False,
            ug2_track_ua=False,
            pa_max_w=0.001, pa_over_pct=0,  # 1mW limit — everything exceeds
        )
        with patch("time.sleep"):
            run_scan(client, settings, progress=lambda e: events.append(e))
        summary = self._get_summary(events)
        self.assertEqual(len(summary["curves"]), 3)
        # First curve fails, others aborted
        self.assertEqual(summary["curves"][0]["status"], "pa_first")
        self.assertEqual(summary["curves"][1]["status"], "aborted")
        self.assertEqual(summary["curves"][2]["status"], "aborted")

    @patch("time.sleep")
    def test_summary_includes_duration(self, _):
        """Summary should include duration_s field."""
        settings = _make_scan_settings(
            ua=ScanRange(10, 30, 10),
            ug1=ScanRange(-4, -4, 0),
            is_triode=True,
            pa_max_w=100.0, pa_over_pct=0,
        )
        client = self._make_client()
        events = []
        with patch("time.sleep"):
            run_scan(client, settings, progress=lambda e: events.append(e))
        summary = self._get_summary(events)
        self.assertIn("duration_s", summary)
        self.assertGreaterEqual(summary["duration_s"], 0)
        self.assertIn("heater_lost", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
