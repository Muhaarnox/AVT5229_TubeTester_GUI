"""Periphery calibration tests — plan B stage 5.7 (docs/CALIBRATION_PLAN.md).

Covers the periphery call sites that must become calibration-aware:

- ``lm19/scan/protection.py::_restore_heater_and_wait`` — heater-restore
  comparison must run in the physical domain (``apply_read``); a READ
  offset larger than ``_HEATER_TOLERANCE_V`` otherwise makes the wait
  loop spin to the 30 s timeout even though the physical heater value
  is exactly on target.
- ``app/lamp_panel.py::check_heater_level`` — gains a ``calibration``
  parameter; the now-value must be the calibrated (physical) reading.
- ``app/workers.py::PreheatWorker`` — gains ``calibration`` in the
  constructor; ``_emit_readings`` reports calibrated values.
- ``app/workers.py::ResetWorker`` — pin: shutdown zeroing stays RAW.
  ``apply_set(0) == offset`` could command a non-zero voltage, so the
  raw-shutdown rule must never be "fixed" into the calibrated path.
"""

import itertools
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Headless Qt: must be set before any app.* (PySide6) import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from scan_test_helpers import (
    _make_cal,
    _make_mock_client,
    _make_scan_settings,
    _restore_heater_and_wait,
)
from lm19.config import LampConfig
from app.lamp_panel import check_heater_level
from app.workers import PreheatWorker, ResetWorker
from lm19.constants import (
    TOPOLOGY_PENTODE,
)


def _monotonic_steps(step: float = 0.5):
    """Endless fake clock advancing `step` seconds per call.

    A plain list side_effect would StopIteration when the buggy raw
    comparison spins to the 30 s timeout — the generator lets the loop
    run out naturally so the test fails on poll count, not on fixture.
    """
    return (step * i for i in itertools.count())


def _heater_get_param(uh_raw: int):
    """get_param side_effect: fixed raw Uh reading, 0 for everything else."""
    def fn(name, real=False):
        if name == "Uh":
            return uh_raw
        return 0
    return fn


def _make_lamp(**kwargs):
    """Minimal LampConfig with physically plausible EL84-like defaults."""
    defaults = dict(
        tube_type="EL84", socket="B9A", anodes=1, warmup_s=60,
        topology=TOPOLOGY_PENTODE, uh=6.3, ih=0.0, ug1=-7.0, ua=250.0,
        ia=48.0, ug2=250.0, ig2=5.5, s=11.3, r=38.0, k=20.0,
        ranges={}, limits={},
    )
    defaults.update(kwargs)
    return LampConfig(**defaults)


# ── _restore_heater_and_wait ──────────────────────────────────────────

class TestRestoreHeaterCalibrated(unittest.TestCase):
    """Heater-restore wait must compare in the physical domain."""

    @staticmethod
    def _uh_poll_count(client) -> int:
        return sum(1 for c in client.get_param.call_args_list
                   if c.args and c.args[0] == "Uh")

    @patch("time.monotonic")
    @patch("time.sleep")
    def test_read_offset_converges_in_physical_domain(self, _sleep, mock_mono):
        """READ cal uh offset=+0.5: device raw 5.8 V is physically 6.3 V.

        Raw comparison |5.8 − 6.3| = 0.5 > _HEATER_TOLERANCE_V = 0.3
        never converges and spins to the 30 s timeout; the calibrated
        comparison (apply_read) lands on the very first poll.
        """
        mock_mono.side_effect = _monotonic_steps()
        cal = _make_cal(channel="uh", read_offset=0.5)
        # Physical Uh == settings.uh == 6.3 → raw reading (6.3−0.5)/1.0
        # = 5.8 V → device units 58.
        client = _make_mock_client(_heater_get_param(58))
        settings = _make_scan_settings(uh=6.3, calibration=cal)
        _restore_heater_and_wait(client, settings, progress=None, stop=None)
        self.assertLessEqual(
            self._uh_poll_count(client), 2,
            "heater-restore compared the raw reading against the physical "
            "target and spun to timeout instead of converging via apply_read",
        )

    @patch("time.monotonic")
    @patch("time.sleep")
    def test_accepts_calibration_kwarg(self, _sleep, mock_mono):
        """Contract: explicit ``calibration`` parameter for non-scan callers.

        cal is passed both via settings and via the kwarg so the test is
        green regardless of which channel the implementation reads; today
        the unknown kwarg raises TypeError (missing API).
        """
        mock_mono.side_effect = _monotonic_steps()
        cal = _make_cal(channel="uh", read_offset=0.5)
        client = _make_mock_client(_heater_get_param(58))
        settings = _make_scan_settings(uh=6.3, calibration=cal)
        _restore_heater_and_wait(client, settings, progress=None, stop=None,
                                 calibration=cal)
        self.assertLessEqual(self._uh_poll_count(client), 2)

    @patch("time.monotonic")
    @patch("time.sleep")
    def test_default_cal_raw_comparison_preserved(self, _sleep, mock_mono):
        """Pin: default calibration is a no-op.

        Raw 6.3 V converges immediately and the heater re-send command
        stays encode_uh(settings.uh) — identical to pre-refactor raw
        behavior.
        """
        mock_mono.side_effect = _monotonic_steps()
        client = _make_mock_client(_heater_get_param(63))
        settings = _make_scan_settings(uh=6.3)  # default CalibrationData()
        _restore_heater_and_wait(client, settings, progress=None, stop=None)
        self.assertLessEqual(self._uh_poll_count(client), 2)
        client.set_param.assert_any_call("Uh", 63)


# ── check_heater_level ────────────────────────────────────────────────

class TestCheckHeaterLevelCalibrated(unittest.TestCase):
    """check_heater_level must report the calibrated now-value."""

    def test_uh_now_value_calibrated(self):
        """Voltage-heated lamp: now = apply_read(decode_uh(raw))."""
        cal = _make_cal(channel="uh", read_offset=0.5)
        client = _make_mock_client(_heater_get_param(58))
        now, required, unit = check_heater_level(
            client, _make_lamp(), 0.8, calibration=cal)
        self.assertAlmostEqual(now, 6.3)  # 5.8 raw + 0.5 READ offset
        self.assertAlmostEqual(required, 6.3 * 0.8)
        self.assertEqual(unit, "V")

    def test_ih_now_value_calibrated(self):
        """Current-heated lamp: now = apply_read(decode_ih(raw))."""
        cal = _make_cal(channel="ih", read_gain=1.05)
        client = _make_mock_client(
            lambda name, real=False: 80 if name == "Ih" else 0)
        lamp = _make_lamp(uh=0.0, ih=0.76)
        now, required, unit = check_heater_level(
            client, lamp, 0.9, calibration=cal)
        self.assertAlmostEqual(now, 0.80 * 1.05)
        self.assertAlmostEqual(required, 0.76 * 0.9)
        self.assertEqual(unit, "A")

    def test_legacy_call_without_calibration(self):
        """Pin: 3-arg call keeps working (calibration optional → identity)."""
        client = _make_mock_client(_heater_get_param(58))
        now, required, unit = check_heater_level(client, _make_lamp(), 0.8)
        self.assertAlmostEqual(now, 5.8)
        self.assertAlmostEqual(required, 6.3 * 0.8)
        self.assertEqual(unit, "V")


# ── PreheatWorker ─────────────────────────────────────────────────────

class TestPreheatWorkerCalibration(unittest.TestCase):
    """PreheatWorker gets calibration via the constructor (plan §3.2)."""

    def test_ctor_accepts_calibration(self):
        cal = _make_cal(channel="uh", read_offset=0.5)
        w = PreheatWorker(MagicMock(), 6.3, 0.0, 1, calibration=cal)
        # Existing fields must survive the signature change.
        self.assertAlmostEqual(w.target_uh, 6.3)
        self.assertAlmostEqual(w.target_ih, 0.0)
        self.assertEqual(w.warmup_s, 1)

    def test_emit_readings_calibrated(self):
        """progress payload carries physical (apply_read) values.

        No .start(): _emit_readings is called directly on the worker —
        direct-connection signals work without a running thread.
        """
        cal = _make_cal(channel="uh", read_offset=0.5)
        client = _make_mock_client(
            lambda name, real=False: {"Uh": 58, "Ih": 84}.get(name, 0))
        w = PreheatWorker(client, 6.3, 0.0, 1, calibration=cal)
        received = []
        w.progress.connect(lambda uh, ih, rem: received.append((uh, ih, rem)))
        w._emit_readings(5)
        self.assertEqual(len(received), 1)
        uh, ih, rem = received[0]
        self.assertAlmostEqual(uh, 6.3)   # calibrated, not raw 5.8
        self.assertAlmostEqual(ih, 0.84)  # ih READ cal is default → identity
        self.assertEqual(rem, 5)


# ── ResetWorker — raw shutdown pin ────────────────────────────────────

class TestResetWorkerRawShutdown(unittest.TestCase):
    """Pin: shutdown zeroing must stay RAW (plan §2/§3.3).

    apply_set(channel, 0) == offset — with a non-zero SET offset the
    "calibrated zero" would command a non-zero voltage during shutdown.
    """

    def test_hv_zeroing_uses_raw_literals(self):
        client = MagicMock()
        w = ResetWorker(client, ug1_value=-8.0)
        w._execute()
        client.set_param.assert_any_call("Ua", 0)
        client.set_param.assert_any_call("Ug2", 0)
        client.set_param.assert_any_call("Uh", 0)

    def test_safe_lock_ug1_raw_encoded(self):
        """Safe-lock Ug1 is a raw command: encode_ug1(−8.0) == 800.

        Literal pinned (not recomputed via encode_ug1) so a calibrated
        rewrite of the command path fails this test loudly.
        """
        client = MagicMock()
        w = ResetWorker(client, ug1_value=-8.0)
        w._execute()
        client.set_param.assert_any_call("Ug1", 800)


class TestManualTabShutdownRaw(unittest.TestCase):
    """Pin (plan §5.7): manual tab HV/heater shutdown paths stay raw.

    A future "fix" routing them through apply_set would command offset
    volts during a safety drop. Methods are called unbound on a stub —
    they only touch the client, app-config and guard callbacks.
    """

    @staticmethod
    def _stub(client):
        from types import SimpleNamespace
        return SimpleNamespace(
            _can_write_or_warn=lambda: True,
            _get_client_or_warn=lambda: client,
            get_app_config=lambda: SimpleNamespace(ug1_after_stop=-8.0),
            # Non-default cal available: would distort commands if the
            # shutdown paths ever consulted it.
            get_calibration=lambda: _make_cal(
                channel="ug2", set_gain=1.1, set_offset=5.0),
        )

    def test_reset_hv_raw_literals(self):
        from app.manual_tab import ManualTab
        client = MagicMock()
        ManualTab._reset_hv(self._stub(client))
        client.set_param.assert_any_call("Ug2", 0)
        client.set_param.assert_any_call("Ug1", 800)  # encode_ug1(-8.0)
        client.set_param.assert_any_call("Ua", 0)

    def test_reset_all_raw_literals(self):
        from app.manual_tab import ManualTab
        client = MagicMock()
        ManualTab._reset_all(self._stub(client))
        client.set_param.assert_any_call("Ug2", 0)
        client.set_param.assert_any_call("Ug1", 800)
        client.set_param.assert_any_call("Ua", 0)
        client.set_param.assert_any_call("Uh", 0)
        # Ih=0 is the safety-critical line: in current-mode heaters the heater
        # is driven via Ih, so _reset_all must zero it. Pin it (was unguarded —
        # deleting the line left this test green).
        client.set_param.assert_any_call("Ih", 0)


class TestMainWindowHeaterZeroRaw(unittest.TestCase):
    """Live heater spinbox handlers: zero means "heater off" → raw literal.

    Review finding: _on_uh/_on_ih_changed fire on every valueChanged
    (including programmatic setValue during lamp switch / settings load);
    routing 0 through apply_set would command offset volts instead of off.
    Methods are called unbound on a stub — MainWindow itself is too heavy
    to instantiate here, and the handlers only touch client/calibration.
    """

    @staticmethod
    def _stub():
        from types import SimpleNamespace
        return SimpleNamespace(
            client=MagicMock(),
            calibration=_make_cal(channel="uh", set_gain=1.02,
                                  set_offset=0.3),
            _can_send_heater=lambda: True,
        )

    def test_uh_zero_sends_raw_zero(self):
        from app.main_window import MainWindow
        stub = self._stub()
        MainWindow._on_uh_changed(stub, 0.0)
        stub.client.set_param.assert_called_once_with("Uh", 0)

    def test_ih_zero_sends_raw_zero(self):
        from app.main_window import MainWindow
        stub = self._stub()
        stub.calibration = _make_cal(channel="ih", set_gain=1.05,
                                     set_offset=0.05)
        MainWindow._on_ih_changed(stub, 0.0)
        stub.client.set_param.assert_called_once_with("Ih", 0)

    def test_uh_nonzero_is_feedforward(self):
        from app.main_window import MainWindow
        from lm19.protocol import encode_uh
        stub = self._stub()
        MainWindow._on_uh_changed(stub, 6.3)
        expected = encode_uh(stub.calibration.apply_set("uh", 6.3))
        stub.client.set_param.assert_called_once_with("Uh", expected)


if __name__ == "__main__":
    unittest.main()
