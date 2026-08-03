"""Tests for SRK (transconductance / plate-resistance / amplification-factor)
measurement logic: measure_srk, _srk_ug1_values helpers, Ug1-sweep mode,
and real-tube Koren-model validation.
"""

import os
import sys
import math
import unittest

import numpy as np
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.scan import SrkSettings, SrkVerifyError, measure_srk, _srk_ug1_values
from lm19.protocol import encode_ug1, decode_ug1
from lm19.calibration import CalibrationData, IA_HW_SCALE
from lm19.spice_export import _koren_ia, _koren_ia_pentode
from lm19.analysis import Zone, compute_s, compute_r, compute_k


class TestMeasureSrk(unittest.TestCase):
    """Tests for ``measure_srk`` (S/R/K corner-method measurement)."""

    @staticmethod
    def _make_srk_mock(ia_map=None, ia_const=10, ig2_val=0,
                       uh_val=63, ih_val=30):
        """Stateful mock: tracks set_param, dispatches get_param by name."""
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}

        def _set(name, value):
            state[name] = value

        def _get(name, real=False):
            if name == "Ia":
                if ia_map:
                    return ia_map.get((state["Ua"], state["Ug1"]), ia_const)
                return ia_const
            if name == "Ig2":
                return ig2_val
            if name == "Uh":
                return uh_val
            if name == "Ih":
                return ih_val
            return state.get(name, 0)

        client = MagicMock()
        client.set_param = MagicMock(side_effect=_set)
        client.get_param = MagicMock(side_effect=_get)
        return client, state

    # -- basic triode ---------------------------------------------------

    @patch("time.sleep")
    def test_triode_basic_srk(self, _sleep):
        """Triode: 4 corner points, correct S/R/K, full 7-field point dict."""
        # Ia(ua, ug1_raw): encode_ug1(-2)=200, encode_ug1(-4)=400
        ia_map = {
            (100, 200): 500,   # ua=100, ug1=-2 → decode_ia → 5 mA
            (200, 200): 1000,  # ua=200, ug1=-2 → decode_ia → 10 mA
            (100, 400): 200,   # ua=100, ug1=-4 → decode_ia → 2 mA
            (200, 400): 700,   # ua=200, ug1=-4 → decode_ia → 7 mA
        }
        client, _ = self._make_srk_mock(ia_map=ia_map, uh_val=63, ih_val=30)
        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,  # numerical order for Zone._in_zone
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=True,
        )
        s, r, k, points, _unc = measure_srk(client, settings)

        self.assertEqual(len(points), 4)
        # S = 1.5 mA/V  (linear regression ia vs ug1)
        self.assertIsNotNone(s)
        self.assertAlmostEqual(s, 1.5, places=2)
        # R = 20.0 kΩ  (1 / slope of ia vs ua)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 20.0, places=1)
        # K = S × R = 30
        self.assertIsNotNone(k)
        self.assertAlmostEqual(k, 30.0, places=1)
        # Points have all 7 fields (new: ig2, uh, ih)
        for p in points:
            for key in ("ua", "ug1", "ug2", "ia", "ig2", "uh", "ih"):
                self.assertIn(key, p)
        # Uh/Ih decoded correctly (63/10=6.3 V, 30/100=0.3 A)
        self.assertAlmostEqual(points[0]["uh"], 6.3, places=1)
        self.assertAlmostEqual(points[0]["ih"], 0.3, places=2)

    # -- pentode with fixed Ug2 ----------------------------------------

    @patch("time.sleep")
    def test_pentode_fixed_ug2(self, _sleep):
        """Pentode with fixed Ug2: set once, all points carry correct Ug2."""
        ia_map = {
            (100, 200): 800,
            (200, 200): 1500,
            (100, 400): 300,
            (200, 400): 900,
        }
        client, _ = self._make_srk_mock(ia_map=ia_map, ig2_val=50)
        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=150, samples=1, calibration=CalibrationData(),
            is_triode=False, ug2_track_ua=False,
        )
        s, r, k, points, _unc = measure_srk(client, settings)

        self.assertEqual(len(points), 4)
        for p in points:
            self.assertAlmostEqual(p["ug2"], 150.0)
        # Ig2 decoded: 50/100 = 0.5 mA
        for p in points:
            self.assertAlmostEqual(p["ig2"], 0.5, places=2)
        self.assertIsNotNone(s)
        self.assertIsNotNone(r)
        self.assertIsNotNone(k)

    # -- pentode track mode ---------------------------------------------

    @patch("time.sleep")
    def test_pentode_track_ug2(self, _sleep):
        """Track mode: Ug2 = Ua + offset for each point."""
        ia_map = {
            (100, 200): 600,
            (200, 200): 1200,
            (100, 400): 300,
            (200, 400): 800,
        }
        client, _ = self._make_srk_mock(ia_map=ia_map, ig2_val=20)
        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=False, ug2_track_ua=True, ug2_offset=10,
        )
        s, r, k, points, _unc = measure_srk(client, settings)

        self.assertEqual(len(points), 4)
        # Point 0: ua=100 → ug2=110
        self.assertAlmostEqual(points[0]["ug2"], 110.0)
        # Point 1: ua=200 → ug2=210
        self.assertAlmostEqual(points[1]["ug2"], 210.0)
        # Point 2: ua=100 → ug2=110
        self.assertAlmostEqual(points[2]["ug2"], 110.0)
        # Point 3: ua=200 → ug2=210
        self.assertAlmostEqual(points[3]["ug2"], 210.0)

    # -- track mode: screen never left above a descending anode ---------

    @patch("time.sleep")
    def test_track_mode_ug2_dropped_before_ua_descends(self, _sleep):
        """ML-129: between Ug1 curves the device sits at Ua=ua_max with
        Ug2=ua_max+offset; Ug2 must be dropped to the first-Ua level BEFORE
        Ua descends, otherwise the screen stays far above the anode for a
        full settle (Ig2 spike). Mirrors _sweep_ug2_track's entry order."""
        ia_map = {
            (100, 200): 600,
            (200, 200): 1200,
            (100, 400): 300,
            (200, 400): 800,
        }
        client, state = self._make_srk_mock(ia_map=ia_map, ig2_val=20)
        commands = []

        def _set(name, value):
            commands.append((name, value))
            state[name] = value

        client.set_param = MagicMock(side_effect=_set)
        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=False, ug2_track_ua=True, ug2_offset=10,
        )
        measure_srk(client, settings)

        # Replay the command stream: whenever Ua is commanded DOWN, the Ug2
        # setpoint at that moment must already match the new (lower) level.
        ua_state, ug2_state = 0.0, 0.0
        for name, value in commands:
            if name == "Ua":
                if value < ua_state:
                    self.assertLessEqual(
                        ug2_state, value + settings.ug2_offset + 1.0,
                        f"Ua commanded down to {value} V while Ug2 still at "
                        f"{ug2_state} V — screen left above the anode")
                ua_state = value
            elif name == "Ug2":
                ug2_state = value

    # -- Ua verify failure ----------------------------------------------

    @patch("time.sleep")
    def test_ua_verify_failure_raises(self, _sleep):
        """Ua fails to settle within tolerance → SrkVerifyError."""
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}

        def _set(name, value):
            state[name] = value

        def _get(name, real=False):
            if name == "Ua":
                return state["Ua"] + 10   # always 10 V off
            return state.get(name, 0)

        client = MagicMock()
        client.set_param = MagicMock(side_effect=_set)
        client.get_param = MagicMock(side_effect=_get)

        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=0, samples=1, is_triode=True,
            ua_tolerance=2.0, verify_retries=3,
        )
        with self.assertRaises(SrkVerifyError) as ctx:
            measure_srk(client, settings)
        self.assertIn("Ua failed to settle", str(ctx.exception))

    # -- Ug1 drift during Ua/Ug2 settle --------------------------------

    @patch("time.sleep")
    def test_ug1_drift_raises(self, _sleep):
        """Ug1 drifts during Ua settle → SrkVerifyError."""
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}
        drift = {"active": False}

        def _set(name, value):
            state[name] = value
            if name == "Ua":
                drift["active"] = True     # Ug1 drifts after Ua set

        def _get(name, real=False):
            if name == "Ug1" and drift["active"]:
                return state["Ug1"] + 50   # 5 V raw drift (≫0.2 V tolerance)
            return state.get(name, 0)

        client = MagicMock()
        client.set_param = MagicMock(side_effect=_set)
        client.get_param = MagicMock(side_effect=_get)

        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=0, samples=1, is_triode=True,
            ug1_verify_tolerance=0.2,
        )
        with self.assertRaises(SrkVerifyError) as ctx:
            measure_srk(client, settings)
        self.assertIn("Ug1 drifted", str(ctx.exception))

    # -- Ug2 verify failure ---------------------------------------------

    @patch("time.sleep")
    def test_ug2_verify_failure_raises(self, _sleep):
        """Pentode: Ug2 never reaches target → SrkVerifyError."""
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}

        def _set(name, value):
            state[name] = value

        def _get(name, real=False):
            if name == "Ug2":
                return state.get("Ug2", 0) + 10   # always 10 V off
            return state.get(name, 0)

        client = MagicMock()
        client.set_param = MagicMock(side_effect=_set)
        client.get_param = MagicMock(side_effect=_get)

        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=150, samples=1,
            is_triode=False, ug2_track_ua=False,
            ug2_tolerance=2.0, verify_retries=3,
        )
        with self.assertRaises(SrkVerifyError) as ctx:
            measure_srk(client, settings)
        self.assertIn("Ug2 failed to settle", str(ctx.exception))

    # -- stop callback --------------------------------------------------

    @patch("time.sleep")
    def test_stop_callback(self, _sleep):
        """Stop after 2 points — only first Ug1 curve measured."""
        client, _ = self._make_srk_mock(ia_const=10)
        call_count = [0]

        def progress(done, total):
            call_count[0] = done

        def stop():
            return call_count[0] >= 2

        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=True,
        )
        s, r, k, points, _unc = measure_srk(
            client, settings, progress=progress, stop=stop)
        self.assertEqual(len(points), 2)

    # -- safety: Ua set before Ug2 in track mode ------------------------

    @patch("time.sleep")
    def test_ua_set_before_ug2_in_track_mode(self, _sleep):
        """Track-mode command order, two-tier policy:

        - WITHIN a curve (ascending Ua) each per-point Ug2 follows its Ua —
          the tube briefly sees higher Ua with the old lower Ug2 (lower
          transient Pg2);
        - at CURVE ENTRY a lone Ug2 pre-drop right before the Ug1 command is
          deliberate (ML-129): it lowers the screen before Ua descends from
          the previous curve's maximum.
        """
        client, _ = self._make_srk_mock(ia_const=10)
        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=False, ug2_track_ua=True, ug2_offset=0,
        )
        measure_srk(client, settings)

        set_names = [c[0][0] for c in client.set_param.call_args_list]
        ug2_indices = [i for i, n in enumerate(set_names) if n == "Ug2"]

        for ug2_i in ug2_indices:
            # Entry pre-drop: the next Ua/Ug1 command ahead is the curve's
            # Ug1 — this Ug2 belongs to the ML-129 entry sequence, skip.
            is_entry_drop = False
            for j in range(ug2_i + 1, len(set_names)):
                if set_names[j] == "Ug1":
                    is_entry_drop = True
                    break
                if set_names[j] == "Ua":
                    break
            if is_entry_drop:
                continue
            # Per-point order: walk backwards, must find "Ua" before "Ug1"
            found_ua = False
            for j in range(ug2_i - 1, -1, -1):
                if set_names[j] == "Ua":
                    found_ua = True
                    break
                if set_names[j] == "Ug1":
                    break
            self.assertTrue(
                found_ua,
                f"Ug2 at pos {ug2_i} not preceded by Ua within same point")

    # -- samples averaging (Ia / Ig2) ----------------------------------

    @patch("time.sleep")
    def test_samples_averaging(self, _sleep):
        """With samples=3, Ia and Ig2 are averaged via _read_measurement_point."""
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}
        ia_reads = [1000, 1200, 1100]   # raw; avg decode_ia → 11.0 mA per point
        ia_idx = [0]

        def _set(name, value):
            state[name] = value

        def _get(name, real=False):
            if name == "Ia":
                val = ia_reads[ia_idx[0] % len(ia_reads)]
                ia_idx[0] += 1
                return val
            if name == "Ig2":
                return 60    # decode_ig2 → 0.6 mA
            if name == "Uh":
                return 63
            if name == "Ih":
                return 30
            return state.get(name, 0)

        client = MagicMock()
        client.set_param = MagicMock(side_effect=_set)
        client.get_param = MagicMock(side_effect=_get)

        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=0, samples=3, calibration=CalibrationData(),
            is_triode=True,
        )
        s, r, k, points, _unc = measure_srk(client, settings)

        # First point: (10+12+11)/3 = 11.0 mA
        self.assertAlmostEqual(points[0]["ia"], 11.0, places=1)
        # Ig2 avg: 3×(60/100) / 3 = 0.6 mA
        self.assertAlmostEqual(points[0]["ig2"], 0.6, places=2)

    # -- progress callback fires ----------------------------------------

    @patch("time.sleep")
    def test_progress_callback(self, _sleep):
        """Progress callback receives (done, 4) for each measured point."""
        client, _ = self._make_srk_mock(ia_const=10)
        progress_calls = []

        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=True,
        )
        measure_srk(client, settings,
                     progress=lambda d, t: progress_calls.append((d, t)))

        self.assertEqual(progress_calls,
                         [(1, 4), (2, 4), (3, 4), (4, 4)])


class TestSrkUg1Values(unittest.TestCase):
    """Tests for _srk_ug1_values helper (Ug1 step generation)."""

    def test_no_step_returns_two_points(self):
        """ug1_step=0 → classic [ug1_min, ug1_max]."""
        s = SrkSettings(ua_min=100, ua_max=200, ug1_min=-4, ug1_max=-2, ug2=0)
        vals = _srk_ug1_values(s)
        self.assertEqual(vals, [-4, -2])

    def test_negative_step_returns_two_points(self):
        """ug1_step<0 treated as disabled."""
        s = SrkSettings(ua_min=100, ua_max=200, ug1_min=-4, ug1_max=-2, ug2=0,
                        ug1_step=-0.1)
        vals = _srk_ug1_values(s)
        self.assertEqual(vals, [-4, -2])

    def test_step_0_04_in_0_48_zone(self):
        """Step 0.04V in zone -2.48 to -2.0 → 13 values."""
        s = SrkSettings(ua_min=100, ua_max=200, ug1_min=-2.48, ug1_max=-2.0,
                        ug2=0, ug1_step=0.04)
        vals = _srk_ug1_values(s)
        self.assertEqual(len(vals), 13)
        self.assertAlmostEqual(vals[0], -2.48, places=4)
        self.assertAlmostEqual(vals[-1], -2.0, places=4)

    def test_step_0_08_in_0_48_zone(self):
        """Step 0.08V in zone -2.48 to -2.0 → 7 values."""
        s = SrkSettings(ua_min=100, ua_max=200, ug1_min=-2.48, ug1_max=-2.0,
                        ug2=0, ug1_step=0.08)
        vals = _srk_ug1_values(s)
        self.assertEqual(len(vals), 7)
        self.assertAlmostEqual(vals[0], -2.48, places=4)
        self.assertAlmostEqual(vals[-1], -2.0, places=4)

    def test_step_larger_than_range(self):
        """Step > range → still includes both endpoints."""
        s = SrkSettings(ua_min=100, ua_max=200, ug1_min=-4, ug1_max=-2,
                        ug2=0, ug1_step=5.0)
        vals = _srk_ug1_values(s)
        self.assertEqual(len(vals), 2)
        self.assertAlmostEqual(vals[0], -4.0)
        self.assertAlmostEqual(vals[-1], -2.0)

    def test_step_exact_division(self):
        """Step exactly divides range → no duplicate endpoint."""
        s = SrkSettings(ua_min=100, ua_max=200, ug1_min=-4, ug1_max=-2,
                        ug2=0, ug1_step=0.5)
        vals = _srk_ug1_values(s)
        self.assertEqual(len(vals), 5)  # -4, -3.5, -3.0, -2.5, -2.0
        self.assertAlmostEqual(vals[0], -4.0)
        self.assertAlmostEqual(vals[-1], -2.0)

    def test_monotonically_increasing(self):
        """All values are strictly increasing (Ug1 from more-negative to less-negative)."""
        s = SrkSettings(ua_min=100, ua_max=200, ug1_min=-4, ug1_max=-2,
                        ug2=0, ug1_step=0.06)
        vals = _srk_ug1_values(s)
        for i in range(len(vals) - 1):
            self.assertLess(vals[i], vals[i + 1])


class TestMeasureSrkSweep(unittest.TestCase):
    """Tests for measure_srk with ug1_step > 0 (Ug1 sweep / mini-scan)."""

    @staticmethod
    def _make_srk_mock(ia_fn=None, ia_const=10, ig2_val=0,
                       uh_val=63, ih_val=30):
        """Stateful mock with optional Ia function: ia_fn(ua_raw, ug1_raw) -> raw Ia."""
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}

        def _set(name, value):
            state[name] = value

        def _get(name, real=False):
            if name == "Ia":
                if ia_fn:
                    return ia_fn(state["Ua"], state["Ug1"])
                return ia_const
            if name == "Ig2":
                return ig2_val
            if name == "Uh":
                return uh_val
            if name == "Ih":
                return ih_val
            return state.get(name, 0)

        client = MagicMock()
        client.set_param = MagicMock(side_effect=_set)
        client.get_param = MagicMock(side_effect=_get)
        return client, state

    @patch("time.sleep")
    def test_sweep_produces_correct_point_count(self, _sleep):
        """Sweep 0.04V in 0.48V zone → 13 Ug1 × 2 Ua = 26 points."""
        client, _ = self._make_srk_mock(ia_const=10)
        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-2.48, ug1_max=-2.0,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=True, ug1_step=0.04,
        )
        s, r, k, points, _unc = measure_srk(client, settings)
        self.assertEqual(len(points), 26)

    @patch("time.sleep")
    def test_sweep_improves_s_precision(self, _sleep):
        """With a linear Ia(ug1), sweep S should match the slope exactly."""
        # Ia = 2.0 * abs(ug1) + 0.01 * ua  (linear, S = -2.0 mA/V)
        # encode_ug1(-2) = 200, Ia_raw → decode_ia → mA
        def ia_fn(ua_raw, ug1_raw):
            # ug1_raw = abs(V)*100, decode: -(ug1_raw/100)
            ug1 = -(ug1_raw / 100.0)
            # Ia(ug1) = S * ug1 + offset + small ua dependence
            # S = -2.0 mA/V (negative because ug1 is negative)
            ia = 2.0 * (-ug1) + 0.01 * ua_raw  # always positive
            return ia / IA_HW_SCALE  # raw protocol value → decode_ia gives mA

        client, _ = self._make_srk_mock(ia_fn=ia_fn)
        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=True, ug1_step=0.04,
        )
        s, r, k, points, _unc = measure_srk(client, settings)

        # S = dIa/dUg1 for linear function: slope = -2.0 (less negative ug1 → lower Ia)
        self.assertIsNotNone(s)
        self.assertAlmostEqual(s, -2.0, places=2)
        # Many points → good regression
        self.assertGreater(len(points), 20)

    @patch("time.sleep")
    def test_sweep_all_points_in_zone(self, _sleep):
        """All swept points should be within the SRK zone for regression."""
        client, _ = self._make_srk_mock(ia_const=10)
        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=True, ug1_step=0.1,
        )
        s, r, k, points, _unc = measure_srk(client, settings)

        for p in points:
            self.assertGreaterEqual(p["ua"], 100)
            self.assertLessEqual(p["ua"], 200)
            self.assertGreaterEqual(p["ug1"], -4)
            self.assertLessEqual(p["ug1"], -2)

    @patch("time.sleep")
    def test_sweep_pentode_track_mode(self, _sleep):
        """Sweep mode works correctly with pentode track-Ua mode."""
        client, _ = self._make_srk_mock(ia_const=10, ig2_val=20)
        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=False, ug2_track_ua=True, ug2_offset=10,
            ug1_step=0.5,
        )
        s, r, k, points, _unc = measure_srk(client, settings)

        # 5 ug1 values × 2 ua = 10 points
        self.assertEqual(len(points), 10)
        # Check Ug2 tracking
        for p in points:
            expected_ug2 = p["ua"] + 10
            self.assertAlmostEqual(p["ug2"], expected_ug2)

    @patch("time.sleep")
    def test_sweep_progress_callback(self, _sleep):
        """Progress callback receives correct (done, total) for sweep."""
        client, _ = self._make_srk_mock(ia_const=10)
        progress_calls = []
        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=True, ug1_step=1.0,
        )
        measure_srk(client, settings,
                     progress=lambda d, t: progress_calls.append((d, t)))

        # 3 ug1 values (-4, -3, -2) × 2 ua = 6 points
        self.assertEqual(len(progress_calls), 6)
        self.assertEqual(progress_calls[-1], (6, 6))

    @patch("time.sleep")
    def test_sweep_stop_mid_measurement(self, _sleep):
        """Stop callback halts sweep partway through."""
        client, _ = self._make_srk_mock(ia_const=10)
        call_count = [0]

        def progress(done, total):
            call_count[0] = done

        def stop():
            return call_count[0] >= 4

        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=True, ug1_step=0.5,
        )
        s, r, k, points, _unc = measure_srk(
            client, settings, progress=progress, stop=stop)
        self.assertEqual(len(points), 4)


class TestSrkRealTubes(unittest.TestCase):
    """S/R/K tests using Koren models of real tubes from tube_params.json.

    Instead of synthetic constants, the mock client computes Ia from the
    Koren triode/pentode equation with reference parameters.  This verifies:
      - measure_srk produces S/R/K matching direct Koren analysis (no noise)
      - S, R, K fall in physically reasonable ranges for each tube type
      - K ≈ mu for triodes (zone-averaged, within ~50%)
      - Pentodes have very high R (plate resistance)

    Reference: Norman Koren tube.lib parameters via config/tube_params.json.
    """

    # -- Mock factories ------------------------------------------------

    @staticmethod
    def _triode_mock(mu, ex, kg1, kp, kvb, uh_raw=63, ih_raw=30):
        """Mock LM19Serial using Koren triode model for Ia."""
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}

        def _set(name, value):
            state[name] = value

        def _get(name, real=False):
            if name == "Ia":
                ua = float(max(state["Ua"], 0.01))
                ug1 = -(state["Ug1"] / 100.0)  # decode_ug1
                ia_a = _koren_ia(
                    np.array([ua]), np.array([ug1]),
                    mu, ex, kg1, kp, kvb,
                )[0]
                return ia_a * 1000.0 / IA_HW_SCALE  # raw protocol value
            if name == "Ig2":
                return 0
            if name == "Uh":
                return uh_raw
            if name == "Ih":
                return ih_raw
            return state.get(name, 0)

        client = MagicMock()
        client.set_param = MagicMock(side_effect=_set)
        client.get_param = MagicMock(side_effect=_get)
        return client

    @staticmethod
    def _pentode_mock(mu, ex, kg1, kp, kvb, ig2_raw=50,
                      uh_raw=63, ih_raw=30):
        """Mock LM19Serial using Koren pentode model for Ia."""
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}

        def _set(name, value):
            state[name] = value

        def _get(name, real=False):
            if name == "Ia":
                ua = float(max(state["Ua"], 0.01))
                ug1 = -(state["Ug1"] / 100.0)
                ug2 = float(max(state["Ug2"], 0.01))
                ia_a = _koren_ia_pentode(
                    np.array([ua]), np.array([ug1]), np.array([ug2]),
                    mu, ex, kg1, kp, kvb,
                )[0]
                return ia_a * 1000.0 / IA_HW_SCALE  # raw protocol value
            if name == "Ig2":
                return ig2_raw
            if name == "Uh":
                return uh_raw
            if name == "Ih":
                return ih_raw
            return state.get(name, 0)

        client = MagicMock()
        client.set_param = MagicMock(side_effect=_set)
        client.get_param = MagicMock(side_effect=_get)
        return client

    # -- Expected-value helpers ----------------------------------------

    @staticmethod
    def _expected_triode(ua_min, ua_max, ug1_min, ug1_max,
                         mu, ex, kg1, kp, kvb, ug1_step=0.0):
        """Direct S/R/K from Koren triode model (no encode/decode)."""
        s_set = SrkSettings(
            ua_min=ua_min, ua_max=ua_max,
            ug1_min=ug1_min, ug1_max=ug1_max,
            ug2=0, is_triode=True, ug1_step=ug1_step,
        )
        ug1_vals = _srk_ug1_values(s_set)
        points = []
        for ug1 in ug1_vals:
            for ua in [ua_min, ua_max]:
                ia_a = _koren_ia(
                    np.array([float(ua)]), np.array([float(ug1)]),
                    mu, ex, kg1, kp, kvb,
                )[0]
                points.append({"ua": float(ua), "ug1": float(ug1),
                               "ug2": 0.0, "ia": float(ia_a * 1000.0)})
        zone = Zone(ua_min, ua_max, ug1_min, ug1_max, is_triode=True)
        return (compute_s(points, zone),
                compute_r(points, zone),
                compute_k(compute_s(points, zone), compute_r(points, zone)),
                len(points))

    @staticmethod
    def _expected_pentode(ua_min, ua_max, ug1_min, ug1_max, ug2,
                          mu, ex, kg1, kp, kvb, ug1_step=0.0):
        """Direct S/R/K from Koren pentode model (no encode/decode)."""
        s_set = SrkSettings(
            ua_min=ua_min, ua_max=ua_max,
            ug1_min=ug1_min, ug1_max=ug1_max,
            ug2=ug2, is_triode=False, ug1_step=ug1_step,
        )
        ug1_vals = _srk_ug1_values(s_set)
        points = []
        for ug1 in ug1_vals:
            for ua in [ua_min, ua_max]:
                ia_a = _koren_ia_pentode(
                    np.array([float(ua)]), np.array([float(ug1)]),
                    np.array([float(ug2)]),
                    mu, ex, kg1, kp, kvb,
                )[0]
                points.append({"ua": float(ua), "ug1": float(ug1),
                               "ug2": float(ug2), "ia": float(ia_a * 1000.0)})
        zone = Zone(ua_min, ua_max, ug1_min, ug1_max, ug2=ug2,
                    is_triode=False)
        return (compute_s(points, zone),
                compute_r(points, zone),
                compute_k(compute_s(points, zone), compute_r(points, zone)),
                len(points))

    # ================================================================
    # Triode tests — individual tube types
    # ================================================================

    @patch("time.sleep")
    def test_12ax7_classic_4point(self, _sleep):
        """12AX7/ECC83: classic 4-corner SRK, K ≈ mu ≈ 100."""
        mu, ex, kg1, kp, kvb = 100.0, 1.4, 1060.0, 600.0, 300.0
        client = self._triode_mock(mu, ex, kg1, kp, kvb)

        settings = SrkSettings(
            ua_min=200, ua_max=300,
            ug1_min=-2.0, ug1_max=-1.0,
            ug2=0, samples=1, calibration=CalibrationData(), is_triode=True,
        )
        s, r, k, points, _unc = measure_srk(client, settings)
        exp_s, exp_r, exp_k, exp_n = self._expected_triode(
            200, 300, -2.0, -1.0, mu, ex, kg1, kp, kvb)

        self.assertEqual(len(points), 4)
        self.assertAlmostEqual(s, exp_s, places=4)
        self.assertAlmostEqual(r, exp_r, places=1)
        # Physical: 12AX7 gm ≈ 1.0-2.5 mA/V, rp ≈ 35-80 kΩ
        self.assertGreater(s, 0.5)
        self.assertLess(s, 4.0)
        self.assertGreater(r, 20)
        self.assertLess(r, 120)

    @patch("time.sleep")
    def test_12ax7_sweep_matches_model(self, _sleep):
        """12AX7/ECC83: sweep 0.04V matches direct Koren computation."""
        mu, ex, kg1, kp, kvb = 100.0, 1.4, 1060.0, 600.0, 300.0
        client = self._triode_mock(mu, ex, kg1, kp, kvb)

        settings = SrkSettings(
            ua_min=200, ua_max=300,
            ug1_min=-2.0, ug1_max=-1.0,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=True, ug1_step=0.04,
        )
        s, r, k, points, _unc = measure_srk(client, settings)
        exp_s, exp_r, exp_k, exp_n = self._expected_triode(
            200, 300, -2.0, -1.0, mu, ex, kg1, kp, kvb, ug1_step=0.04)

        self.assertEqual(len(points), exp_n)
        self.assertGreater(len(points), 20)  # many sweep points
        self.assertAlmostEqual(s, exp_s, places=4)
        self.assertAlmostEqual(r, exp_r, places=1)
        self.assertAlmostEqual(k, exp_k, places=1)

    @patch("time.sleep")
    def test_12au7_sweep(self, _sleep):
        """12AU7/ECC82: low-mu triode, K ≈ 17-21."""
        mu, ex, kg1, kp, kvb = 21.5, 1.3, 1180.0, 84.0, 300.0
        client = self._triode_mock(mu, ex, kg1, kp, kvb)

        settings = SrkSettings(
            ua_min=150, ua_max=250,
            ug1_min=-10.0, ug1_max=-6.0,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=True, ug1_step=0.08,
        )
        s, r, k, points, _unc = measure_srk(client, settings)
        exp_s, exp_r, exp_k, _ = self._expected_triode(
            150, 250, -10.0, -6.0, mu, ex, kg1, kp, kvb, ug1_step=0.08)

        self.assertAlmostEqual(s, exp_s, places=4)
        self.assertAlmostEqual(r, exp_r, places=1)
        # 12AU7: S ≈ 2-3 mA/V, R ≈ 5-15 kΩ
        self.assertGreater(s, 1.0)
        self.assertLess(s, 5.0)
        self.assertGreater(r, 3)
        self.assertLess(r, 25)
        # K should be in ballpark of mu=21.5
        self.assertGreater(k, 8)
        self.assertLess(k, 40)

    @patch("time.sleep")
    def test_6sn7_sweep(self, _sleep):
        """6SN7: medium-mu triode, K ≈ 20."""
        mu, ex, kg1, kp, kvb = 20.0, 1.3, 1350.0, 130.0, 300.0
        client = self._triode_mock(mu, ex, kg1, kp, kvb)

        settings = SrkSettings(
            ua_min=150, ua_max=250,
            ug1_min=-10.0, ug1_max=-6.0,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=True, ug1_step=0.08,
        )
        s, r, k, points, _unc = measure_srk(client, settings)
        exp_s, exp_r, exp_k, _ = self._expected_triode(
            150, 250, -10.0, -6.0, mu, ex, kg1, kp, kvb, ug1_step=0.08)

        self.assertAlmostEqual(s, exp_s, places=4)
        self.assertAlmostEqual(r, exp_r, places=1)
        # 6SN7: S ≈ 2-3 mA/V, R ≈ 5-12 kΩ
        self.assertGreater(s, 1.0)
        self.assertLess(s, 5.0)
        self.assertGreater(r, 3)
        self.assertLess(r, 20)

    @patch("time.sleep")
    def test_6dj8_sweep(self, _sleep):
        """6DJ8/ECC88: high-gm triode, S > 5 mA/V."""
        mu, ex, kg1, kp, kvb = 33.0, 1.3, 330.0, 320.0, 300.0
        client = self._triode_mock(mu, ex, kg1, kp, kvb)

        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-2.0, ug1_max=-1.0,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=True, ug1_step=0.04,
        )
        s, r, k, points, _unc = measure_srk(client, settings)
        exp_s, exp_r, exp_k, _ = self._expected_triode(
            100, 200, -2.0, -1.0, mu, ex, kg1, kp, kvb, ug1_step=0.04)

        self.assertAlmostEqual(s, exp_s, places=4)
        self.assertAlmostEqual(r, exp_r, places=1)
        # 6DJ8: very high gm, low rp
        self.assertGreater(s, 3.0)
        self.assertLess(s, 25.0)
        self.assertGreater(r, 0.5)
        self.assertLess(r, 15)

    @patch("time.sleep")
    def test_6sl7_sweep(self, _sleep):
        """6SL7: high-mu triode (mu=70), K should be large."""
        mu, ex, kg1, kp, kvb = 70.0, 1.35, 1600.0, 525.0, 300.0
        client = self._triode_mock(mu, ex, kg1, kp, kvb)

        settings = SrkSettings(
            ua_min=150, ua_max=250,
            ug1_min=-3.0, ug1_max=-1.0,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=True, ug1_step=0.04,
        )
        s, r, k, points, _unc = measure_srk(client, settings)
        exp_s, exp_r, exp_k, _ = self._expected_triode(
            150, 250, -3.0, -1.0, mu, ex, kg1, kp, kvb, ug1_step=0.04)

        self.assertAlmostEqual(s, exp_s, places=4)
        self.assertAlmostEqual(r, exp_r, places=1)
        # 6SL7: high mu → high K
        self.assertGreater(k, 25)
        self.assertLess(k, 120)

    # ================================================================
    # Pentode tests
    # ================================================================

    @patch("time.sleep")
    def test_el84_pentode_sweep(self, _sleep):
        """EL84 pentode: high gm, very high rp."""
        mu, ex, kg1, kp, kvb = 19.0, 1.35, 650.0, 300.0, 48.0
        client = self._pentode_mock(mu, ex, kg1, kp, kvb)

        settings = SrkSettings(
            ua_min=200, ua_max=300,
            ug1_min=-8.0, ug1_max=-6.0,
            ug2=250, samples=1, calibration=CalibrationData(),
            is_triode=False, ug1_step=0.08,
        )
        s, r, k, points, _unc = measure_srk(client, settings)
        exp_s, exp_r, exp_k, exp_n = self._expected_pentode(
            200, 300, -8.0, -6.0, 250,
            mu, ex, kg1, kp, kvb, ug1_step=0.08)

        self.assertEqual(len(points), exp_n)
        self.assertAlmostEqual(s, exp_s, places=3)
        # Pentode: very high plate resistance → less precision match
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, exp_r, delta=max(abs(exp_r) * 0.01, 1))
        # EL84: S ≈ 5-15 mA/V; R ≫ 20 kΩ (pentode characteristic)
        self.assertGreater(s, 3.0)
        self.assertLess(s, 25.0)
        self.assertGreater(r, 10)

    @patch("time.sleep")
    def test_el34_pentode_sweep(self, _sleep):
        """EL34 pentode: power tube, physically reasonable S/R/K."""
        mu, ex, kg1, kp, kvb = 11.0, 1.35, 650.0, 60.0, 24.0
        client = self._pentode_mock(mu, ex, kg1, kp, kvb)

        settings = SrkSettings(
            ua_min=200, ua_max=300,
            ug1_min=-20.0, ug1_max=-16.0,
            ug2=250, samples=1, calibration=CalibrationData(),
            is_triode=False, ug1_step=0.08,
        )
        s, r, k, points, _unc = measure_srk(client, settings)
        exp_s, exp_r, exp_k, _ = self._expected_pentode(
            200, 300, -20.0, -16.0, 250,
            mu, ex, kg1, kp, kvb, ug1_step=0.08)

        self.assertAlmostEqual(s, exp_s, places=3)
        self.assertIsNotNone(r)
        # EL34: S ≈ 5-15 mA/V for power pentode
        self.assertGreater(s, 3.0)
        self.assertLess(s, 25.0)
        self.assertGreater(r, 5)

    @patch("time.sleep")
    def test_ef86_pentode_sweep(self, _sleep):
        """EF86 small-signal pentode: moderate S, very high R."""
        mu, ex, kg1, kp, kvb = 38.0, 1.4, 2000.0, 200.0, 40.0
        client = self._pentode_mock(mu, ex, kg1, kp, kvb)

        settings = SrkSettings(
            ua_min=150, ua_max=250,
            ug1_min=-3.0, ug1_max=-1.0,
            ug2=140, samples=1, calibration=CalibrationData(),
            is_triode=False, ug1_step=0.04,
        )
        s, r, k, points, _unc = measure_srk(client, settings)
        exp_s, exp_r, exp_k, _ = self._expected_pentode(
            150, 250, -3.0, -1.0, 140,
            mu, ex, kg1, kp, kvb, ug1_step=0.04)

        self.assertAlmostEqual(s, exp_s, places=3)
        self.assertIsNotNone(r)
        # EF86: S ≈ 1-3 mA/V, R very high (pentode)
        self.assertGreater(s, 0.5)
        self.assertLess(s, 8.0)
        self.assertGreater(r, 20)

    # ================================================================
    # Cross-tube parametric checks
    # ================================================================

    @patch("time.sleep")
    def test_triode_k_approximates_mu(self, _sleep):
        """For all triodes, zone-averaged K should be within 0.3-1.5 × mu."""
        triodes = [
            # (name, mu, ex, kg1, kp, kvb, ua_min, ua_max, ug1_min, ug1_max)
            ("12AX7",  100.0, 1.4,  1060.0, 600.0, 300.0, 200, 300, -2.0,  -1.0),
            ("12AU7",   21.5, 1.3,  1180.0,  84.0, 300.0, 150, 250, -10.0, -6.0),
            ("12AT7",   60.0, 1.35,  460.0, 300.0, 300.0, 150, 250, -3.0,  -1.0),
            ("6SN7",    20.0, 1.3,  1350.0, 130.0, 300.0, 150, 250, -10.0, -6.0),
            ("6DJ8",    33.0, 1.3,   330.0, 320.0, 300.0, 100, 200, -2.0,  -1.0),
            ("6SL7",    70.0, 1.35, 1600.0, 525.0, 300.0, 150, 250, -3.0,  -1.0),
        ]
        for name, mu, ex, kg1, kp, kvb, ua0, ua1, ug0, ug1 in triodes:
            with self.subTest(tube=name, mu=mu):
                client = self._triode_mock(mu, ex, kg1, kp, kvb)
                settings = SrkSettings(
                    ua_min=ua0, ua_max=ua1,
                    ug1_min=ug0, ug1_max=ug1,
                    ug2=0, samples=1, calibration=CalibrationData(),
                    is_triode=True, ug1_step=0.04,
                )
                s, r, k, _, _unc = measure_srk(client, settings)
                self.assertIsNotNone(k, f"{name}: K is None")
                ratio = k / mu
                self.assertGreater(
                    ratio, 0.3,
                    f"{name}: K={k:.1f} too low vs mu={mu} (ratio={ratio:.2f})")
                self.assertLess(
                    ratio, 1.5,
                    f"{name}: K={k:.1f} too high vs mu={mu} (ratio={ratio:.2f})")

    @patch("time.sleep")
    def test_pentode_r_much_larger_than_triode(self, _sleep):
        """Pentode rp should be much larger than triode rp at similar S."""
        # EL84 pentode
        client_p = self._pentode_mock(19.0, 1.35, 650.0, 300.0, 48.0)
        settings_p = SrkSettings(
            ua_min=200, ua_max=300,
            ug1_min=-8.0, ug1_max=-6.0,
            ug2=250, samples=1, calibration=CalibrationData(),
            is_triode=False, ug1_step=0.08,
        )
        _, r_pent, _, _, _unc = measure_srk(client_p, settings_p)

        # 12AU7 triode (similar S range)
        client_t = self._triode_mock(21.5, 1.3, 1180.0, 84.0, 300.0)
        settings_t = SrkSettings(
            ua_min=150, ua_max=250,
            ug1_min=-10.0, ug1_max=-6.0,
            ug2=0, samples=1, calibration=CalibrationData(),
            is_triode=True, ug1_step=0.08,
        )
        _, r_tri, _, _, _unc = measure_srk(client_t, settings_t)

        # Pentode R should be at least 3× higher than triode R
        self.assertGreater(r_pent, r_tri * 3,
                           f"Pentode R={r_pent:.0f} should be >> triode R={r_tri:.0f}")

    @patch("time.sleep")
    def test_sweep_vs_classic_same_zone(self, _sleep):
        """Sweep and classic modes should give similar S/R/K (deterministic model)."""
        mu, ex, kg1, kp, kvb = 100.0, 1.4, 1060.0, 600.0, 300.0
        base = dict(ua_min=200, ua_max=300, ug1_min=-2.0, ug1_max=-1.0,
                    ug2=0, samples=1, calibration=CalibrationData(), is_triode=True)

        # Classic: 4 points
        client_c = self._triode_mock(mu, ex, kg1, kp, kvb)
        s_c, r_c, k_c, pts_c, _unc_c = measure_srk(client_c, SrkSettings(**base))

        # Sweep: many points
        client_s = self._triode_mock(mu, ex, kg1, kp, kvb)
        s_s, r_s, k_s, pts_s, _unc_s = measure_srk(
            client_s, SrkSettings(**base, ug1_step=0.04))

        self.assertEqual(len(pts_c), 4)
        self.assertGreater(len(pts_s), 20)
        # With a smooth model (no noise), both should give very close results
        # (slight difference from zone-averaged non-linearity over more points)
        self.assertAlmostEqual(s_c, s_s, delta=abs(s_c) * 0.15)
        self.assertAlmostEqual(r_c, r_s, delta=abs(r_c) * 0.15)


# ── Plan B (docs/CALIBRATION_PLAN.md §5.4): SRK actuals in physical domain ──


class TestSrkCalibratedActuals(unittest.TestCase):
    """SRK overwrites point ua/ug1/ug2 with settle actuals (srk.py:202-204).

    Those actuals are raw decoded device readings today, while
    _read_measurement_point had already written calibrated (physical)
    values — the overwrite silently degrades the point. After plan B the
    adapter returns apply_read(decoded) and the overwrite is consistent.

    SET cal stays default here so the commanded raws (ia_map keys) are
    identical before and after the refactor — the red comes purely from
    the READ-domain overwrite.
    """

    READ_GAIN = 1.005

    @classmethod
    def _make_read_cal_mock(cls, ia_map, ig2_val=0):
        """Stateful mock: device holds commanded value physically; its ADC
        reading r satisfies apply_read(r) == physical (r = phys / gain)."""
        state = {"Ua": 0, "Ug1": 0, "Ug2": 0}
        g = cls.READ_GAIN

        def _set(name, value):
            state[name] = value

        def _get(name, real=False):
            if name == "Ia":
                return ia_map.get((state["Ua"], state["Ug1"]), 10)
            if name == "Ig2":
                return ig2_val
            if name == "Uh":
                return 63
            if name == "Ih":
                return 30
            if not real:
                return state.get(name, 0)
            if name in ("Ua", "Ug2"):
                return float(state[name]) / g
            if name == "Ug1":
                phys = decode_ug1(state["Ug1"])
                return int(round(abs(phys / g) * 100))
            return state.get(name, 0)

        client = MagicMock()
        client.set_param = MagicMock(side_effect=_set)
        client.get_param = MagicMock(side_effect=_get)
        return client

    @staticmethod
    def _make_read_cal(gain):
        cal = CalibrationData()
        for ch in ("ua", "ug1", "ug2"):
            cal.set_channel(ch, "read", gain, 0.0)
        return cal

    @patch("time.sleep")
    def test_point_voltages_are_physical(self, _sleep):
        ia_map = {
            (100, 200): 800, (200, 200): 1500,
            (100, 400): 300, (200, 400): 900,
        }
        client = self._make_read_cal_mock(ia_map, ig2_val=50)
        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=150, samples=1,
            calibration=self._make_read_cal(self.READ_GAIN),
            is_triode=False, ug2_track_ua=False,
        )
        s, r, k, points, _unc = measure_srk(client, settings)

        self.assertEqual(len(points), 4)
        # Raw decoded today: ua 99.50/199.00, ug2 149.25, ug1 -3.98/-1.99.
        for p in points:
            self.assertAlmostEqual(
                p["ua"], 100.0 if p["ua"] < 150 else 200.0, delta=0.3)
            self.assertAlmostEqual(p["ug2"], 150.0, delta=0.3)
            self.assertAlmostEqual(
                p["ug1"], -4.0 if p["ug1"] < -3.0 else -2.0, delta=0.01)

    @patch("time.sleep")
    def test_triode_ug2_zeroing_stays_raw(self, _sleep):
        """Pin: triode-mode Ug2=0 (srk.py:134) must stay a raw literal."""
        ia_map = {
            (100, 200): 500, (200, 200): 1000,
            (100, 400): 200, (200, 400): 700,
        }
        client = self._make_read_cal_mock(ia_map)
        cal = self._make_read_cal(self.READ_GAIN)
        cal.set_channel("ug2", "set", 1.1, 5.0)
        settings = SrkSettings(
            ua_min=100, ua_max=200,
            ug1_min=-4, ug1_max=-2,
            ug2=0, samples=1, calibration=cal,
            is_triode=True,
        )
        measure_srk(client, settings)
        client.set_param.assert_any_call("Ug2", 0)


class TestNoDeadUg1SettleKnob(unittest.TestCase):
    """ML-136: the dead SrkSettings.ug1_settle_s field is removed —
    the app.json:ug1_settle_s knob never acted on SRK (Ug1 settle is
    driven by srk.json: settle_per_volt_s/settle_base_s)."""

    def test_ctor_rejects_removed_field(self):
        with self.assertRaises(TypeError):
            SrkSettings(ua_min=100, ua_max=200, ug1_min=-4, ug1_max=-2,
                        ug2=0, ug1_settle_s=0.5)

    def test_build_settings_no_longer_passes_it(self):
        """Call site: SrkController.build_settings constructs without
        TypeError, Ug1 settle comes from the srk config knobs."""
        from lm19.app_config import AppConfig
        from app.srk_widget import SrkController
        cfg = AppConfig(srk_settle_per_volt_s=2.5, srk_settle_base_s=3.5)
        zone = {"ua_min": 100.0, "ua_max": 200.0, "ug1_min": -4.0,
                "ug1_max": -2.0, "ug2": 150.0}
        s = SrkController.build_settings(zone, cfg, None, ug2_track=False,
                                         ug2_offset=0.0)
        self.assertEqual(s.settle_per_volt_s, 2.5)
        self.assertEqual(s.settle_base_s, 3.5)
        self.assertFalse(hasattr(s, "ug1_settle_s"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
