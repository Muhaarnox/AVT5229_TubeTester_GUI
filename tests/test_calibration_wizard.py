"""Tests for the calibration wizard result/measure pages
(plan B, docs/CALIBRATION_PLAN.md §5.6).

Contract for _VoltageResultPage:
- SET coefficients are derived from the freshly computed READ via
  CalibrationData.derive_set_two_point (commanded + device readings;
  the multimeter feeds READ only);
- the SET section is marked "auto-derived" (i18n key cal.Wiz_set_auto);
- unchecking "apply READ" disables "apply SET" (SET is derived from
  READ — no basis, no apply);
- ug1 READ fit normalizes signs: canonical lm19/ domain is negative
  physical volts, a negative gain is impossible.

Pinned behavior (must stay green through stage 5):
- _VoltageMeasurePage commands encoder(target) RAW even when a SET
  calibration is available — the wizard characterizes the
  *uncalibrated* transfer (feedforward would compound the correction);
- derived SET numerically equals the legacy meter-based fit when READ
  is fresh (the two-point READ fit passes exactly through both meter
  points), guarding against deriving from a stale READ.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Headless Qt before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from serial import SerialException

from PySide6.QtWidgets import QApplication, QLabel, QWizard

from scan_test_helpers import _make_cal, CalibrationData
import app.calibration_wizard as wizard_mod
from app.calibration_wizard import (
    CalibrationWizard,
    _CurrentMeasurePage,
    _VoltageMeasurePage,
    _VoltageResultPage,
)
from i18n_setup import t

QApplication.instance() or QApplication([])


# ── fixtures ──────────────────────────────────────────────────────────

class _StubMeasurePage:
    """Plain stand-in for _VoltageMeasurePage with exactly the attrs the
    result page consumes (commanded, dev_reading, dev_stats,
    meter_reading) — same interface as the real page."""

    def __init__(self, commanded: float, dev_reading: float,
                 meter_reading: float, sigma: float = 0.01, n: int = 10):
        self.commanded = float(commanded)
        self.dev_reading = float(dev_reading)
        self.dev_sigma = float(sigma)
        self.dev_stats = {
            "mean": float(dev_reading), "sigma": float(sigma),
            "min": float(dev_reading), "max": float(dev_reading), "n": n,
        }
        self.meter_reading = float(meter_reading)


def _make_result_page(channel="ua", cal=None, lo=None, hi=None,
                      has_set=True):
    """Result page with distinct commanded / dev / meter values so a
    test can tell exactly which numbers fed which fit."""
    cal = cal if cal is not None else CalibrationData()
    lo = lo or _StubMeasurePage(commanded=50.0, dev_reading=49.1,
                                meter_reading=50.4)
    hi = hi or _StubMeasurePage(commanded=250.0, dev_reading=247.3,
                                meter_reading=251.2)
    page = _VoltageResultPage(cal, channel, lo, hi, "V", has_set)
    return page, cal, lo, hi


# ── tests ─────────────────────────────────────────────────────────────

class TestSetDerivedFromRead(unittest.TestCase):
    """SET must come from CalibrationData.derive_set_two_point.

    Numerically the derived SET equals the legacy
    compute_set_two_point(commanded, meter) when READ is freshly
    fitted, so the contract is pinned structurally: the result page
    must call the derive API.  patch.object(create=False) doubles as
    the red trigger — AttributeError until derive_set_two_point exists.
    """

    DERIVED = (0.93, -1.7)

    def test_set_derived_from_read(self):
        page, cal, lo, hi = _make_result_page()
        with patch.object(CalibrationData, "derive_set_two_point",
                          return_value=self.DERIVED) as derive:
            page.initializePage()

        self.assertEqual(derive.call_count, 1)
        passed = (list(derive.call_args.args)
                  + list(derive.call_args.kwargs.values()))
        # SET derivation consumes (commanded, device reading) pairs...
        for v in (lo.commanded, lo.dev_reading, hi.commanded, hi.dev_reading):
            self.assertIn(v, passed)
        # ...and never the multimeter entries — those feed READ only.
        for v in (lo.meter_reading, hi.meter_reading):
            self.assertNotIn(v, passed)
        # The derive result is what the page exposes for validatePage.
        self.assertAlmostEqual(page.set_gain, self.DERIVED[0])
        self.assertAlmostEqual(page.set_offset, self.DERIVED[1])

    def test_validate_page_writes_derived_set_with_quality(self):
        page, cal, lo, hi = _make_result_page()
        with patch.object(CalibrationData, "derive_set_two_point",
                          return_value=self.DERIVED):
            page.initializePage()
            page.validatePage()

        # READ written from the (meter, dev) fit, with quality metadata.
        exp_gain, exp_offset = CalibrationData.compute_two_point(
            lo.meter_reading, hi.meter_reading,
            lo.dev_reading, hi.dev_reading)
        read_ch = cal.get_channel("ua", "read")
        self.assertAlmostEqual(read_ch.gain, exp_gain)
        self.assertAlmostEqual(read_ch.offset, exp_offset)
        self.assertIsNotNone(read_ch.quality)

        # SET written from the derive result, also with quality.
        set_ch = cal.get_channel("ua", "set")
        self.assertAlmostEqual(set_ch.gain, self.DERIVED[0])
        self.assertAlmostEqual(set_ch.offset, self.DERIVED[1])
        self.assertIsNotNone(set_ch.quality)

    def test_set_values_equal_meter_based_fit_pin(self):
        """PIN — fresh READ passes through both meter points, so derived
        SET must numerically equal the legacy meter-based fit.

        A pre-existing (stale) READ calibration on the channel makes the
        trap real: deriving from self.calibration's old READ instead of
        the freshly computed one would shift the result.
        """
        stale = _make_cal(channel="ua", read_gain=1.10, read_offset=4.0)
        page, cal, lo, hi = _make_result_page(cal=stale)
        page.initializePage()
        exp_gain, exp_offset = CalibrationData.compute_set_two_point(
            lo.commanded, lo.meter_reading,
            hi.commanded, hi.meter_reading)
        self.assertAlmostEqual(page.set_gain, exp_gain, places=9)
        self.assertAlmostEqual(page.set_offset, exp_offset, places=9)


class TestSetSectionMarkedAuto(unittest.TestCase):
    def test_set_section_marked_auto(self):
        """SET section must be labelled auto-derived via cal.Wiz_set_auto.

        t() falls back to the key itself for missing keys, so the
        visibility assertion stays valid before and after the locale
        entry lands (stage 7); the key-usage assertion is the red core.
        """
        used_keys = []
        real_t = wizard_mod.t

        def recording_t(key, **kwargs):
            used_keys.append(key)
            return real_t(key, **kwargs)

        # Page construction AND initializePage both inside the recorder:
        # the implementation may set the marker in either place.
        with patch.object(wizard_mod, "t", new=recording_t):
            page, _, _, _ = _make_result_page()
            page.initializePage()

        self.assertIn("cal.Wiz_set_auto", used_keys)

        marker = t("cal.Wiz_set_auto")
        texts = [w.text() for w in page.findChildren(QLabel)]
        texts.append(page.apply_set_cb.text())
        self.assertTrue(
            any(marker in txt for txt in texts),
            f"auto-derived marker {marker!r} not visible on result page",
        )


class TestSetCheckboxCoupledToRead(unittest.TestCase):
    def test_set_checkbox_disabled_without_read(self):
        """SET is derived from READ: refusing READ removes the basis for
        SET, so unchecking apply_read must disable (and uncheck)
        apply_set — a disabled-but-checked box would still be written
        by validatePage."""
        page, _, _, _ = _make_result_page()
        page.initializePage()
        # Sanity precondition: both sections computed fine.
        self.assertTrue(page.apply_read_cb.isChecked())
        self.assertTrue(page.apply_set_cb.isEnabled())

        page.apply_read_cb.setChecked(False)

        self.assertFalse(page.apply_set_cb.isEnabled())
        self.assertFalse(page.apply_set_cb.isChecked())


class TestMeasurePageCommandsRaw(unittest.TestCase):
    """PIN — must stay green through stage 5.

    The wizard characterizes the *uncalibrated* transfer: measurement
    pages command encoder(target) raw even when a SET calibration is
    available to the wizard. Feedforward here would compound the
    correction into the new fit.
    """

    @patch("time.sleep")
    def test_measure_page_sets_commanded_raw(self, mock_sleep):
        client = MagicMock()
        client.get_param = MagicMock(return_value=0)
        cal = _make_cal(channel="ua", set_gain=0.9, set_offset=2.0)

        wizard = CalibrationWizard(client, cal, "ua",
                                   cal_samples=1, cal_interval_ms=0)
        low_page = wizard.page(wizard.pageIds()[1])
        self.assertIsInstance(low_page, _VoltageMeasurePage)

        target = low_page.target
        raw = int(round(target))
        # Guard that the pin is meaningful: apply_set would change the
        # command if it leaked into the measure page.
        self.assertNotEqual(int(round(cal.apply_set("ua", target))), raw)

        low_page.initializePage()
        low_page.cleanupPage()  # stop the live-poll timer

        client.set_param.assert_called_once_with("Ua", raw)
        # Bookkeeping stays in the commanded (raw) domain too.
        self.assertEqual(low_page.commanded, float(target))


class TestHeaterComplementZeroed(unittest.TestCase):
    """ML-121: the firmware's UART path does not zero the complementary
    heater setpoint (`!Uh=` leaves ihset intact, unlike the internal
    lamp-template paths), and with both setpoints non-zero the firmware
    stabilization loop runs both `if(uhset>0)`/`if(ihset>0)` blocks against
    the same PWM variable — they fight for the heater. Calibrating uh/ih
    must raw-zero the other channel first and safe-reset both on close."""

    @patch("time.sleep")
    def test_uh_measure_page_zeros_ih_first(self, _):
        client = MagicMock()
        client.get_param = MagicMock(return_value=0)
        page = _VoltageMeasurePage(client, "uh", 3.0, "Low", "V", 1, 0)
        page.initializePage()
        page.cleanupPage()
        names = [c[0][0] for c in client.set_param.call_args_list]
        self.assertIn("Ih", names)
        self.assertIn("Uh", names)
        self.assertLess(names.index("Ih"), names.index("Uh"),
                        "complementary Ih must be zeroed BEFORE Uh is set")
        ih_call = client.set_param.call_args_list[names.index("Ih")]
        self.assertEqual(ih_call[0][1], 0)   # raw zero, not encoded

    @patch("time.sleep")
    def test_ih_measure_page_zeros_uh_first(self, _):
        client = MagicMock()
        client.get_param = MagicMock(return_value=0)
        page = _VoltageMeasurePage(client, "ih", 0.5, "Low", "A", 1, 0)
        page.initializePage()
        page.cleanupPage()
        names = [c[0][0] for c in client.set_param.call_args_list]
        self.assertLess(names.index("Uh"), names.index("Ih"),
                        "complementary Uh must be zeroed BEFORE Ih is set")

    @patch("time.sleep")
    def test_non_heater_channel_has_no_complement_write(self, _):
        client = MagicMock()
        client.get_param = MagicMock(return_value=0)
        page = _VoltageMeasurePage(client, "ua", 50.0, "Low", "V", 1, 0)
        page.initializePage()
        page.cleanupPage()
        names = [c[0][0] for c in client.set_param.call_args_list]
        self.assertEqual(names, ["Ua"])

    @patch("time.sleep")
    def test_uh_wizard_safe_resets_both_heater_channels(self, _):
        client = MagicMock()
        client.get_param = MagicMock(return_value=0)
        wizard = CalibrationWizard(client, CalibrationData(), "uh",
                                   cal_samples=1, cal_interval_ms=0)
        self.assertEqual(wizard._touched_channels, ["uh", "ih"])
        client.set_param.reset_mock()
        wizard.done(0)
        names = [c[0][0] for c in client.set_param.call_args_list]
        self.assertIn("Uh", names)
        self.assertIn("Ih", names)


class TestFitBoundsRefused(unittest.TestCase):
    """Out-of-bounds fits are refused before they become live feedforward.

    A consistent meter scale error (wrong DMM range) produces an
    internally consistent fit with near-zero residuals — the quality
    display cannot catch it, only sanity bounds can. With plan B a
    stored bad fit drives every actuator command.
    """

    def _bad_scale_page(self):
        # Meter on the wrong range: ×0.1 readings for the 50/250 V points
        # → internally consistent fit with gain ≈ 0.1.
        lo = _StubMeasurePage(commanded=50.0, dev_reading=49.1,
                              meter_reading=5.04)
        hi = _StubMeasurePage(commanded=250.0, dev_reading=247.3,
                              meter_reading=25.12)
        return _VoltageResultPage(
            CalibrationData(), "ua", lo, hi, "V", has_set=True)

    def test_scale_error_refused(self):
        page = self._bad_scale_page()
        page.initializePage()
        self.assertFalse(page._read_ok)
        self.assertFalse(page.apply_read_cb.isEnabled())
        self.assertFalse(page.apply_read_cb.isChecked())
        self.assertFalse(page.apply_set_cb.isChecked())
        page.validatePage()
        self.assertTrue(page.calibration.get_channel("ua", "read").is_default())
        self.assertTrue(page.calibration.get_channel("ua", "set").is_default())

    def test_zero_meter_refused(self):
        # Meter spinboxes left at the default 0.0 at both points → READ
        # gain 0, which would make read_inverse raise on every command.
        lo = _StubMeasurePage(commanded=50.0, dev_reading=49.1,
                              meter_reading=0.0)
        hi = _StubMeasurePage(commanded=250.0, dev_reading=247.3,
                              meter_reading=0.0)
        page = _VoltageResultPage(
            CalibrationData(), "ua", lo, hi, "V", has_set=True)
        page.initializePage()
        self.assertFalse(page._read_ok)
        page.validatePage()
        self.assertTrue(page.calibration.get_channel("ua", "read").is_default())


class TestStaleStoredSetReset(unittest.TestCase):
    def test_read_rewrite_without_set_resets_stored_set(self):
        """READ replaced while SET not rewritten (user opt-out): the
        previously stored SET was derived from the old READ — stale
        feedforward must be reset, not silently kept."""
        stored = _make_cal(channel="ua", set_gain=0.95, set_offset=1.0)
        page, cal, lo, hi = _make_result_page(cal=stored)
        page.initializePage()
        self.assertTrue(page._set_ok)
        page.apply_set_cb.setChecked(False)  # user keeps READ only

        page.validatePage()

        self.assertFalse(cal.get_channel("ua", "read").is_default())
        self.assertTrue(cal.get_channel("ua", "set").is_default())


class TestResultPageReentry(unittest.TestCase):
    def test_reentry_reevaluates(self):
        """Back → re-measure → Next must re-evaluate from scratch: a
        second visit with degenerate data must refuse what the first
        visit accepted (stale _read_ok/_set_ok must not be written)."""
        page, cal, lo, hi = _make_result_page()
        page.initializePage()
        self.assertTrue(page._read_ok)
        self.assertTrue(page._set_ok)

        # Re-measured points came back degenerate (identical readings).
        hi.dev_reading = lo.dev_reading
        page.initializePage()
        self.assertFalse(page._read_ok)
        self.assertFalse(page._set_ok)

        page.validatePage()
        self.assertTrue(cal.get_channel("ua", "read").is_default())
        self.assertTrue(cal.get_channel("ua", "set").is_default())


class TestUg1SignNormalization(unittest.TestCase):
    """Canonical ug1 domain in lm19/ is negative physical volts.

    DMMs are commonly hooked up to show the bias magnitude (+), while
    decode_ug1 is always negative. The READ fit must normalize signs —
    a negative gain corrupts every downstream apply_read/read_inverse.
    """

    def _make_ug1_page(self, meter_lo: float, meter_hi: float):
        # Real wizard low/high targets for ug1 are +2 / +20 (commanded
        # domain); device readings come back negative via decode_ug1.
        lo = _StubMeasurePage(commanded=2.0, dev_reading=-2.0,
                              meter_reading=meter_lo)
        hi = _StubMeasurePage(commanded=20.0, dev_reading=-20.0,
                              meter_reading=meter_hi)
        page = _VoltageResultPage(CalibrationData(), "ug1", lo, hi, "V",
                                  has_set=False)
        return page, lo, hi

    def test_positive_meter_entry_never_yields_negative_gain(self):
        page, lo, _ = self._make_ug1_page(meter_lo=2.05, meter_hi=20.3)
        page.initializePage()
        self.assertGreater(page.read_gain, 0.0)
        # The normalized fit maps device readings into the negative
        # physical domain: apply_read(dev_lo) == -|meter_lo|.
        self.assertAlmostEqual(
            page.read_gain * lo.dev_reading + page.read_offset,
            -abs(lo.meter_reading), places=6)

    def test_negative_meter_entry_unchanged_pin(self):
        """PIN — a correctly signed meter entry must not be flipped:
        normalization is idempotent."""
        page, lo, _ = self._make_ug1_page(meter_lo=-2.05, meter_hi=-20.3)
        page.initializePage()
        self.assertGreater(page.read_gain, 0.0)
        self.assertAlmostEqual(
            page.read_gain * lo.dev_reading + page.read_offset,
            -2.05, places=6)


def _voltage_page(client, cal_samples=3):
    return _VoltageMeasurePage(
        client, "ua", 50.0, "Low", "V", cal_samples, 0)


def _current_page(client, cal_samples=3):
    return _CurrentMeasurePage(
        client, "ia_low", "ua", 100.0, "Low", "mA", cal_samples, 0)


class TestMeasureReentrancyGuard(unittest.TestCase):
    """#15 — processEvents() in the measure loop lets a queued Next re-enter
    _do_measure mid-flight. The _measuring guard makes the nested call a
    no-op (no nested second measurement, no double serial commanding)."""

    def _reentrant_client(self, page_box):
        client = MagicMock()
        calls = {"n": 0}

        def side_effect(*_a, **_k):
            calls["n"] += 1
            if calls["n"] == 1:
                # Simulate the Next/validatePage slot that processEvents
                # would dispatch mid-measure: re-enter _do_measure.
                page_box[0]._do_measure()
            return 100

        client.get_param.side_effect = side_effect
        return client

    @patch("time.sleep")
    def test_voltage_reentrancy_is_noop(self, _sleep):
        box = [None]
        client = self._reentrant_client(box)
        page = _voltage_page(client, cal_samples=3)
        box[0] = page
        page._do_measure()
        # Guarded: exactly 3 reads. Under revert the nested loop runs too → 6.
        # (call_count is the load-bearing check; dev_stats["n"] would be 3
        # either way — the last loop to finish always writes n=3 — so it is
        # NOT a discriminating assertion and is intentionally omitted.)
        self.assertEqual(client.get_param.call_count, 3)
        self.assertTrue(page.measure_btn.isEnabled())
        self.assertFalse(page._measuring)       # flag released, not stuck

    @patch("time.sleep")
    def test_current_reentrancy_is_noop(self, _sleep):
        box = [None]
        client = self._reentrant_client(box)
        page = _current_page(client, cal_samples=3)
        box[0] = page
        page._do_measure()
        self.assertEqual(client.get_param.call_count, 3)
        self.assertFalse(page._measuring)       # flag released, not stuck

    @patch("time.sleep")
    def test_nav_buttons_greyed_during_measure(self, _sleep):
        """#15 defense-in-depth: nav buttons are disabled mid-measure and
        restored after. Only fires when the page is attached to a real
        QWizard (standalone pages → wizard() is None → _nav_buttons() []),
        so build a real wizard here."""
        client = MagicMock()
        wizard = CalibrationWizard(client, CalibrationData(), "ua",
                                   cal_samples=2, cal_interval_ms=0)
        page = wizard.page(wizard.pageIds()[1])     # first measure page
        self.assertTrue(page._nav_buttons())        # attached → non-empty
        mid = []

        def side_effect(*_a, **_k):
            mid.append([b.isEnabled() for b in page._nav_buttons()])
            return 100

        client.get_param.side_effect = side_effect
        page._do_measure()
        # Disabled during every read (revert with no grey-out → all True).
        self.assertTrue(mid)
        self.assertTrue(all(not e for snap in mid for e in snap))
        # Restored afterwards by the finally.
        self.assertTrue(all(b.isEnabled() for b in page._nav_buttons()))
        page.cleanupPage()

    @patch("time.sleep")
    def test_validatepage_while_measuring_does_not_advance(self, _sleep):
        """A Next arriving while a measurement is in flight must not advance
        even if a PRIOR measurement populated dev_stats."""
        client = MagicMock()
        page = _voltage_page(client)
        page.dev_stats = {"mean": 1.0, "sigma": 0.0, "min": 1.0,
                          "max": 1.0, "n": 1}     # a prior measure exists
        page._measuring = True                     # a new one is in flight
        client.get_param.reset_mock()
        self.assertFalse(page.validatePage())      # must NOT advance (revert→True)
        client.get_param.assert_not_called()


class TestWizardDoneSafeResetVisibility(unittest.TestCase):
    """#16 — a failed safe-reset on close must be surfaced (the operator must
    not believe the tester is de-energized) and only expected comm/data
    errors may be caught; programming errors propagate."""

    @patch.object(wizard_mod.QMessageBox, "warning")
    def test_comm_error_warns_and_surfaces(self, mwarn):
        # Every member of the caught tuple must surface, not just one — a
        # typo dropping one type would otherwise go unnoticed.
        for exc in (OSError("io"), ValueError("bad"),
                    RuntimeError("Serial port is not open"),
                    SerialException("link")):
            with self.subTest(exc=type(exc).__name__):
                mwarn.reset_mock()
                client = MagicMock()
                client.set_param.side_effect = exc
                wizard = CalibrationWizard(client, CalibrationData(), "ua",
                                           cal_samples=1, cal_interval_ms=0)
                with self.assertLogs("app.calibration_wizard",
                                     level="WARNING") as logctx:
                    wizard.done(1)
                mwarn.assert_called_once()       # user-visible signal fired
                joined = " ".join(logctx.output).lower()
                self.assertIn("ua", joined)      # channel logged

    @patch.object(wizard_mod.QMessageBox, "warning")
    def test_programming_error_propagates(self, _mwarn):
        # Programming errors must NOT be swallowed (bare-except revert would).
        for exc in (AttributeError("boom"), TypeError("nope"), KeyError("k")):
            with self.subTest(exc=type(exc).__name__):
                client = MagicMock()
                client.set_param.side_effect = exc
                wizard = CalibrationWizard(client, CalibrationData(), "ua",
                                           cal_samples=1, cal_interval_ms=0)
                with self.assertRaises(type(exc)):
                    wizard.done(1)

    @patch.object(wizard_mod.QMessageBox, "warning")
    def test_happy_path_no_popup(self, mwarn):
        client = MagicMock()           # set_param returns a Mock, no raise
        cal = CalibrationData()
        wizard = CalibrationWizard(client, cal, "ua",
                                   cal_samples=1, cal_interval_ms=0)
        # Pin that the dialog actually closes (super().done) — a refactor
        # dropping the close while keeping the reset must fail here.
        with patch.object(wizard_mod.QWizard, "done") as super_done:
            wizard.done(1)
            super_done.assert_called_once_with(1)
        mwarn.assert_not_called()
        client.set_param.assert_any_call("Ua", 0)   # safe-reset attempted


class TestMeasurePageBlocksOnFailure(unittest.TestCase):
    """#17 — validatePage must refuse to advance past a missing/failed
    measurement instead of carrying a fabricated 0.0 reading forward."""

    @patch("time.sleep")
    def test_voltage_failed_measure_blocks(self, _sleep):
        client = MagicMock()
        client.get_param.side_effect = TimeoutError("link down")
        page = _voltage_page(client, cal_samples=1)
        self.assertEqual(page.dev_stats, {})
        ok = page.validatePage()          # auto-retry also fails
        self.assertFalse(ok)              # revert returns True
        self.assertEqual(page.dev_stats, {})
        self.assertTrue(page.dev_label.text())   # user sees a reason
        page.cleanupPage()

    @patch("time.sleep")
    def test_current_failed_measure_blocks(self, _sleep):
        client = MagicMock()
        client.get_param.side_effect = TimeoutError("link down")
        page = _current_page(client, cal_samples=1)
        ok = page.validatePage()
        self.assertFalse(ok)
        self.assertEqual(page.dev_stats, {})
        page.cleanupPage()

    @patch("time.sleep")
    def test_never_measured_shows_required_message(self, _sleep):
        client = MagicMock()
        page = _voltage_page(client)
        page.dev_stats = {}
        with patch.object(page, "_do_measure"):   # auto-retry does nothing
            ok = page.validatePage()
        self.assertFalse(ok)
        self.assertEqual(page.dev_label.text(), t("cal.Wiz_measure_required"))
        page.cleanupPage()

    @patch("time.sleep")
    def test_voltage_success_advances(self, _sleep):
        client = MagicMock()
        client.get_param.return_value = 100       # decodes fine
        page = _voltage_page(client, cal_samples=1)
        page._do_measure()
        self.assertTrue(page.dev_stats)
        self.assertTrue(page.validatePage())      # N=1 success not over-blocked
        page.cleanupPage()

    @patch("time.sleep")
    def test_success_stops_poll_timer_on_forward_nav(self, _sleep):
        """P1: forward-nav after a successful auto-measure must NOT leave the
        poll timer running (QWizard skips cleanupPage on Next, so a hidden
        page would keep hitting get_param)."""
        client = MagicMock()
        client.get_param.return_value = 100
        page = _voltage_page(client, cal_samples=1)
        page._poll_timer.start()                  # as initializePage does
        self.assertTrue(page.validatePage())      # empty stats → auto-measure ok
        self.assertFalse(page._poll_timer.isActive())   # revert → still active
        page.cleanupPage()

    @patch("time.sleep")
    def test_current_success_advances(self, _sleep):
        client = MagicMock()
        client.get_param.return_value = 100
        page = _current_page(client, cal_samples=1)
        page._do_measure()
        self.assertTrue(page.dev_stats)
        self.assertTrue(page.validatePage())
        page.cleanupPage()


if __name__ == "__main__":
    unittest.main()
