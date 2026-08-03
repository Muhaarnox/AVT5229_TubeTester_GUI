"""Step-by-step calibration wizard dialog for a single channel.

Voltage channels: one wizard collects commanded, device reading and
multimeter at two points. The multimeter feeds the READ fit only; the
SET correction is auto-derived from the fresh READ + the observed DAC
transfer (``derive_set_two_point``, plan B — docs/CALIBRATION_PLAN.md).
Fits outside sanity bounds (``fit_within_bounds``) are refused.

Current channels (Ia low/high, Ig2): ammeter method — user connects ammeter
in series, device sets voltage to produce current through a load, user reads
ammeter. Two points → READ correction. Same structure as voltage.

All measure pages feature:
- Live polling (QTimer, ~1s) showing instantaneous device reading
- N-point averaged "Measure" button with sigma/stats
- Quality metadata stored per-channel
"""

import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional

from serial import SerialException

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from app.widget_factory import make_double_spinbox, make_int_spinbox

from lm19.calibration import (
    ALL_SET_CHANNELS,
    CHANNEL_UNITS,
    DEFAULT_METER_ACCURACY_PCT,
    IA_RANGE_CHANNELS,
    IA_RANGE_THRESHOLD,
    CalibrationData,
    fit_within_bounds,
)
from lm19.config import DEFAULT_LIMITS
from lm19.protocol import (
    LM19Serial,
    decode_ia,
    decode_ig2,
    decode_ih,
    decode_ug1,
    decode_uh,
    encode_ih,
    encode_ug1,
    encode_uh,
)
from app.ui_theme import COLOR_MID_GRAY, STYLE_BOLD, STYLE_SECONDARY_SMALL
from i18n_setup import t

log = logging.getLogger(__name__)

_DECODE = {
    "ua": ("Ua", lambda raw: float(raw)),
    "ug1": ("Ug1", decode_ug1),
    "ug2": ("Ug2", lambda raw: float(raw)),
    "uh": ("Uh", decode_uh),
    "ih": ("Ih", decode_ih),
    "ia": ("Ia", decode_ia),
    "ig2": ("Ig2", decode_ig2),
}

_ENCODE = {
    "ua":  lambda v: int(round(v)),
    "ug1": encode_ug1,
    "ug2": lambda v: int(round(v)),
    "uh":  encode_uh,
    "ih":  encode_ih,
}

_L = DEFAULT_LIMITS

_SAFE_RESET = {
    "ua": 0, "ug2": 0, "ug1": _L["ug1_max"], "uh": 0, "ih": 0,
}

# Heater setpoints are NOT mutually exclusive on the firmware's UART path:
# unlike the internal lamp-template paths, `!Uh=` does not zero `ihset`
# (and vice versa), and the stabilization loop then runs BOTH `if(uhset>0)`
# and `if(ihset>0)` blocks against the same PWM variable — they fight for
# the heater. Calibrating one heater channel must therefore raw-zero the
# other first, and safe-reset both on close (ML-121).
_HEATER_COMPLEMENT = {"uh": "ih", "ih": "uh"}

_VOLTAGE_LOW = {
    "ua":  round(_L["ua_max"] / 6, -1),       # 50
    "ug2": round(_L["ug2_max"] / 6, -1),      # 50
    "ug1": round(_L["ug1_max"] / 12),          # 2
    "uh":  round(_L["uh_max"] / 5),            # 3
    "ih":  round(_L["ih_max"] / 5, 1),         # 0.5
}
_VOLTAGE_HIGH = {
    "ua":  round(_L["ua_max"] * 5 / 6, -1),   # 250
    "ug2": round(_L["ug2_max"] * 5 / 6, -1),  # 250
    "ug1": round(_L["ug1_max"] * 5 / 6),      # 20
    "uh":  round(_L["uh_max"] * 4 / 5),       # 12
    "ih":  round(_L["ih_max"] * 4 / 5, 1),    # 2.0
}

_CURRENT_SRC_DEFAULTS = {
    "ia_low":  {"source": "ua", "low": round(_L["ua_max"] / 6, -1),
                "high": round(_L["ua_max"] / 2, -1)},
    "ia_high": {"source": "ua", "low": round(_L["ua_max"] / 3, -1),
                "high": round(_L["ua_max"] * 2 / 3, -1)},
    "ig2":     {"source": "ug2", "low": round(_L["ug2_max"] / 3, -1),
                "high": round(_L["ug2_max"] * 2 / 3, -1)},
}

_LOAD_RECOMMENDATIONS = {
    "ia_low": "cal.Wiz_load_rec_ia_low",
    "ia_high": "cal.Wiz_load_rec_ia_high",
    "ig2": "cal.Wiz_load_rec_ig2",
}


class CalibrationWizard(QWizard):
    """Two-point calibration wizard for a single channel.

    For voltage channels: collects commanded, device reading, and
    multimeter at two points — fits READ from the multimeter and
    auto-derives SET from that fresh READ (plan B).

    For current channels (ia_low, ia_high, ig2): ammeter method — sets source
    voltage to produce two different currents, compares device vs ammeter.
    """

    def __init__(
        self,
        client: LM19Serial,
        calibration: CalibrationData,
        channel: str,
        cal_samples: int = 10,
        cal_interval_ms: int = 200,
        parent=None,
        write_guard: Callable[[], bool] = lambda: True,
    ):
        super().__init__(parent)
        self.client = client
        self.calibration = calibration
        self.channel = channel
        self.cal_samples = cal_samples
        self.cal_interval_ms = cal_interval_ms
        # Defense-in-depth write gate (plan: #14). The launch path
        # (_start_wizard) already refuses to open under emergency-lock /
        # hw-busy; this guards per-page set_param if state changes while
        # the wizard is open. Default no-op keeps direct constructions
        # (tests) writing unconditionally.
        self._write_guard = write_guard

        base_ch = channel.replace("_low", "").replace("_high", "")
        self.unit = CHANNEL_UNITS.get(channel, CHANNEL_UNITS.get(base_ch, ""))

        self.setWindowTitle(t("cal.Wizard_title", channel=channel.upper()))
        self.setMinimumWidth(560)
        self._touched_channels: List[str] = []

        if channel in IA_RANGE_CHANNELS or channel == "ig2":
            self._build_current_wizard()
        else:
            self._build_voltage_wizard()

    def _build_voltage_wizard(self) -> None:
        ch = self.channel
        low_v = _VOLTAGE_LOW.get(ch, round(_L["ua_max"] / 6, -1))
        high_v = _VOLTAGE_HIGH.get(ch, round(_L["ua_max"] * 5 / 6, -1))
        has_set = ch in ALL_SET_CHANNELS
        self._touched_channels = [ch]
        if ch in _HEATER_COMPLEMENT:
            # done() must safe-reset BOTH heater setpoints: the measure
            # pages raw-zero the complement before commanding the
            # calibrated one (see _HEATER_COMPLEMENT).
            self._touched_channels.append(_HEATER_COMPLEMENT[ch])

        page1 = QWizardPage()
        page1.setTitle(t("cal.Wiz_prep_title"))
        lay1 = QVBoxLayout(page1)
        lay1.addWidget(QLabel(t("cal.Wiz_connect_multimeter_combined",
                                channel=ch.upper())))
        self.addPage(page1)

        page2 = _VoltageMeasurePage(
            self.client, ch, low_v, t("cal.Wiz_low_point"), self.unit,
            self.cal_samples, self.cal_interval_ms, self._write_guard)
        self.addPage(page2)

        page3 = _VoltageMeasurePage(
            self.client, ch, high_v, t("cal.Wiz_high_point"), self.unit,
            self.cal_samples, self.cal_interval_ms, self._write_guard)
        self.addPage(page3)

        page4 = _VoltageResultPage(
            self.calibration, ch, page2, page3, self.unit, has_set)
        self.addPage(page4)

    def _build_current_wizard(self) -> None:
        ch = self.channel
        cfg = _CURRENT_SRC_DEFAULTS.get(ch, _CURRENT_SRC_DEFAULTS["ig2"])
        source_ch = cfg["source"]
        self._touched_channels = [source_ch]

        page1 = _CurrentPrepPage(ch)
        self.addPage(page1)

        page2 = _CurrentMeasurePage(
            self.client, ch, source_ch, cfg["low"],
            t("cal.Wiz_low_point"), self.unit,
            self.cal_samples, self.cal_interval_ms, self._write_guard)
        self.addPage(page2)

        page3 = _CurrentMeasurePage(
            self.client, ch, source_ch, cfg["high"],
            t("cal.Wiz_high_point"), self.unit,
            self.cal_samples, self.cal_interval_ms, self._write_guard)
        self.addPage(page3)

        page4 = _ReadResultPage(self.calibration, ch, page2, page3, self.unit)
        self.addPage(page4)

    def done(self, result: int) -> None:
        # Safe-reset every touched channel on close. NOT write-guarded:
        # driving outputs to the safe state must always be attempted, even
        # under emergency-lock / busy (like _emergency_zero_outputs).
        failed: List[str] = []
        for ch in self._touched_channels:
            if ch not in _SAFE_RESET or ch not in _ENCODE:
                continue
            param_name = _DECODE[ch][0]
            try:
                self.client.set_param(param_name, _ENCODE[ch](_SAFE_RESET[ch]))
            except (OSError, ValueError, RuntimeError, SerialException) as exc:
                # Reset failed → outputs may still be live. Catch only the
                # expected comm/data errors and surface them; programming
                # errors (Attribute/Type/KeyError from a refactor) must
                # propagate so the regression is visible, not masked
                # (failure-visibility pr. 1).
                failed.append(ch)
                log.warning(
                    "Safe-reset of %s after wizard failed: %s", ch, exc)
        if failed:
            # The operator must not believe the tester is de-energized when
            # it is not — show it before the dialog tears down.
            QMessageBox.warning(
                self, t("msg.COM"),
                t("cal.Wiz_reset_failed",
                  channels=", ".join(c.upper() for c in failed)))
        super().done(result)


# ── Shared measurement logic ─────────────────────────────────────────

def _build_quality_dict(
    ref_low: float, ref_high: float,
    stats_low: Dict, stats_high: Dict,
    gain: float, offset: float,
    meter_accuracy_pct: float,
) -> Dict[str, Any]:
    cal_dev_low = stats_low["mean"] * gain + offset
    cal_dev_high = stats_high["mean"] * gain + offset
    return {
        "low_point": {
            "ref": ref_low,
            "dev_mean": stats_low["mean"],
            "dev_sigma": stats_low["sigma"],
            "n": stats_low["n"],
        },
        "high_point": {
            "ref": ref_high,
            "dev_mean": stats_high["mean"],
            "dev_sigma": stats_high["sigma"],
            "n": stats_high["n"],
        },
        "point_spread": abs(ref_high - ref_low),
        "residual_low": abs(cal_dev_low - ref_low),
        "residual_high": abs(cal_dev_high - ref_high),
        "meter_accuracy_pct": meter_accuracy_pct,
    }


def _stats_through_read(
    stats: Dict[str, float], read_gain: float, read_offset: float,
) -> Dict[str, float]:
    """Map device-reading statistics into the physical domain via the READ fit.

    The derived SET quality compares commanded values against physical
    actuals, but measurement statistics are collected in the device-reading
    domain — the affine READ fit moves mean/min/max, |gain| scales sigma.
    """
    lo_v = stats["min"] * read_gain + read_offset
    hi_v = stats["max"] * read_gain + read_offset
    return {
        "mean": stats["mean"] * read_gain + read_offset,
        "sigma": stats["sigma"] * abs(read_gain),
        "min": min(lo_v, hi_v),
        "max": max(lo_v, hi_v),
        "n": stats["n"],
    }


# ── Wizard pages ─────────────────────────────────────────────────────

class _MeasurePageBase(QWizardPage):
    """Shared measurement logic for the voltage and current measure pages.

    Carries the N-point averaged read with a reentrancy guard, the live-poll
    teardown, and a ``validatePage`` that refuses to advance past a missing or
    failed measurement. The two pages were byte-for-byte duplicates; a single
    base keeps the safety guards from drifting between them.

    Subclasses build their own UI and MUST set, before any measurement:
      - ``self.client``        — LM19Serial
      - ``self._read_ch``      — protocol channel key into ``_DECODE``
      - ``self.unit``          — display unit
      - widgets ``measure_btn``, ``progress``, ``dev_label``, ``live_label``,
        ``n_spin``, ``meter_spin`` and the ``_poll_timer``
      - result attrs ``dev_reading``, ``dev_sigma``, ``dev_stats``,
        ``meter_reading`` and timing ``cal_interval_ms``
    """

    def __init__(self, write_guard: Callable[[], bool] = lambda: True) -> None:
        super().__init__()
        self._write_guard = write_guard
        # processEvents() in the measure loop dispatches queued Next/Back/
        # Cancel clicks; without this guard validatePage re-enters
        # _do_measure mid-flight → nested second measurement, double serial
        # commanding, corrupted stats. Mirrors compare_tab._match_running.
        self._measuring = False

    def _nav_buttons(self) -> List[Any]:
        """Wizard nav buttons to grey out while measuring (defense-in-depth
        on top of the _measuring flag — a disabled button cannot even queue
        a reentrant click)."""
        w = self.wizard()
        if w is None:
            return []
        wanted = (
            QWizard.WizardButton.NextButton,
            QWizard.WizardButton.BackButton,
            QWizard.WizardButton.CancelButton,
            QWizard.WizardButton.FinishButton,
        )
        return [b for b in (w.button(x) for x in wanted) if b is not None]

    def cleanupPage(self) -> None:
        self._poll_timer.stop()

    def _poll_live(self) -> None:
        # Periodic live-value polling — best-effort. Narrow except to
        # transient comm errors (serial timeout, malformed response,
        # OS-level I/O); programming errors propagate.
        try:
            param_name, decode_fn = _DECODE[self._read_ch]
            raw = self.client.get_param(param_name, real=True)
            val = decode_fn(raw)
            self.live_label.setText(
                t("cal.Wiz_live_value", value=f"{val:.3f}", unit=self.unit))
        except (ValueError, TimeoutError, OSError):
            pass

    def _do_measure(self) -> None:
        if self._measuring:
            return
        self._measuring = True
        self._poll_timer.stop()
        nav = self._nav_buttons()
        for btn in nav:
            btn.setEnabled(False)
        n = self.n_spin.value()
        self.progress.setRange(0, n)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.measure_btn.setEnabled(False)

        param_name, decode_fn = _DECODE[self._read_ch]
        from PySide6.QtWidgets import QApplication
        readings: List[float] = []
        try:
            for i in range(n):
                if i > 0:
                    time.sleep(self.cal_interval_ms / 1000.0)
                raw = self.client.get_param(param_name, real=True)
                readings.append(decode_fn(raw))
                self.progress.setValue(i + 1)
                QApplication.processEvents()
            mean = sum(readings) / len(readings)
            if len(readings) > 1:
                variance = sum((x - mean) ** 2 for x in readings) / (len(readings) - 1)
                sigma = math.sqrt(variance)
            else:
                sigma = 0.0
            self.dev_reading = mean
            self.dev_sigma = sigma
            self.dev_stats = {
                "mean": mean, "sigma": sigma,
                "min": min(readings), "max": max(readings), "n": len(readings),
            }
            self.dev_label.setText(
                t("cal.Wiz_measure_result",
                  value=f"{mean:.3f}", sigma=f"{sigma:.4f}",
                  n=n, unit=self.unit))
        except (ValueError, TimeoutError, OSError) as exc:
            # Comm/data error during the averaged read: leave dev_stats
            # empty so validatePage refuses to advance. Programming errors
            # propagate (failure-visibility pr. 1).
            self.dev_label.setText(t("cal.Wiz_error_generic", error=str(exc)))
        finally:
            self.progress.setVisible(False)
            self.measure_btn.setEnabled(True)
            self._poll_timer.start()
            self._measuring = False
            for btn in nav:
                btn.setEnabled(True)

    def validatePage(self) -> bool:
        self._poll_timer.stop()
        self.meter_reading = self.meter_spin.value()
        if self._measuring:
            # A Next dispatched mid-measure (via processEvents) must not
            # advance or kick off a nested measurement.
            return False
        if not self.dev_stats:
            self._do_measure()       # one auto-attempt, as before
        if not self.dev_stats:
            # Still nothing valid (never measured, or the read failed):
            # block Next and say why instead of advancing with a fabricated
            # 0.0 reading (failure-visibility). Keep an existing comm-error
            # message if _do_measure just set one. The poll timer is already
            # running (restarted by _do_measure's finally) — the page stays
            # open and keeps polling.
            if not self.dev_label.text():
                self.dev_label.setText(t("cal.Wiz_measure_required"))
            return False
        # Advancing forward: QWizard does NOT call cleanupPage on Next, so
        # stop polling here or this hidden page keeps hitting get_param.
        self._poll_timer.stop()
        return True


class _VoltageMeasurePage(_MeasurePageBase):
    """Unified voltage measurement page with live polling and N-point averaging."""

    def __init__(self, client, channel, target_value, title, unit,
                 cal_samples, cal_interval_ms, write_guard=lambda: True):
        super().__init__(write_guard)
        self.client = client
        self.channel = channel
        self._read_ch = channel
        self.target = target_value
        self.unit = unit
        self.cal_samples = cal_samples
        self.cal_interval_ms = cal_interval_ms
        self.setTitle(title)

        self.commanded = float(target_value)
        self.dev_reading = 0.0
        self.dev_sigma = 0.0
        self.dev_stats: Dict[str, float] = {}
        self.meter_reading = 0.0

        lay = QVBoxLayout(self)
        self.info_label = QLabel("")
        lay.addWidget(self.info_label)

        self.live_label = QLabel("")
        self.live_label.setStyleSheet(f"color: {COLOR_MID_GRAY};")
        lay.addWidget(self.live_label)

        self.dev_label = QLabel("")
        self.dev_label.setStyleSheet(STYLE_BOLD)
        lay.addWidget(self.dev_label)

        measure_row = QHBoxLayout()
        self.measure_btn = QPushButton(t("cal.Wiz_measure"))
        self.measure_btn.clicked.connect(self._do_measure)
        measure_row.addWidget(self.measure_btn)
        measure_row.addWidget(QLabel(t("common.Label_colon", label=t("common.N"))))
        self.n_spin = make_int_spinbox(
            min_val=1, max_val=100, value=cal_samples,
        )
        measure_row.addWidget(self.n_spin)
        lay.addLayout(measure_row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        meter_row = QHBoxLayout()
        meter_row.addWidget(QLabel(t("cal.Wiz_multimeter_reading")))
        self.meter_spin = make_double_spinbox(
            min_val=-500, max_val=500, value=0.0,
            decimals=3, step=0.1,
        )
        meter_row.addWidget(self.meter_spin)
        meter_row.addWidget(QLabel(unit))
        lay.addLayout(meter_row)

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_live)
        self._poll_timer.setInterval(1000)

    def initializePage(self) -> None:
        self.info_label.setText(
            t("cal.Wiz_setting_channel",
              channel=self.channel.upper(), value=self.target, unit=self.unit))
        self.commanded = float(self.target)
        if self._write_guard():
            try:
                param_name = _DECODE[self.channel][0]
                encoder = _ENCODE[self.channel]
                complement = _HEATER_COMPLEMENT.get(self.channel)
                if complement is not None:
                    # Raw zero (project rule: shutdowns/zeroing bypass
                    # calibration) — see _HEATER_COMPLEMENT for the
                    # firmware PWM-fight rationale (ML-121).
                    self.client.set_param(_DECODE[complement][0], 0)
                self.client.set_param(param_name, encoder(self.target))
                time.sleep(0.5)
            except (OSError, ValueError, RuntimeError, SerialException) as exc:
                # Expected comm/data error → show it; programming errors
                # propagate (failure-visibility pr. 1).
                self.info_label.setText(
                    t("cal.Wiz_error_setting",
                      channel=self.channel, error=str(exc)))
        else:
            # Emergency-lock / hw-busy changed while the wizard is open —
            # do not command the channel (defense-in-depth, #14).
            self.info_label.setText(t("msg.Hw_busy"))
        self._poll_timer.start()


class _CurrentPrepPage(QWizardPage):
    """Current calibration prep: load recommendations."""

    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.setTitle(t("cal.Wiz_prep_title"))
        lay = QVBoxLayout(self)

        lay.addWidget(QLabel(t("cal.Wiz_connect_ammeter",
                                channel=channel.upper())))
        lay.addSpacing(10)

        rec_key = _LOAD_RECOMMENDATIONS.get(channel, "cal.Wiz_load_rec_ig2")
        lay.addWidget(QLabel(t(rec_key)))


class _CurrentMeasurePage(_MeasurePageBase):
    """Current channel measurement page with editable source voltage,
    live polling, and N-point averaging."""

    def __init__(self, client, channel, source_ch, default_source_v, title, unit,
                 cal_samples, cal_interval_ms, write_guard=lambda: True):
        super().__init__(write_guard)
        self.client = client
        self.channel = channel
        self.source_ch = source_ch
        self.default_source_v = default_source_v
        self.unit = unit
        self.cal_samples = cal_samples
        self.cal_interval_ms = cal_interval_ms
        self.setTitle(title)

        self.dev_reading = 0.0
        self.dev_sigma = 0.0
        self.dev_stats: Dict[str, float] = {}
        self.meter_reading = 0.0

        # The actual protocol channel for reading current (base reads via
        # self._read_ch).
        self._current_ch = channel.replace("_low", "").replace("_high", "")
        self._read_ch = self._current_ch

        lay = QVBoxLayout(self)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel(t("cal.Wiz_source_voltage")))
        src_max = _L.get(f"{source_ch}_max", _L["ua_max"])
        self.src_spin = make_double_spinbox(
            min_val=10, max_val=src_max, value=default_source_v,
            decimals=1, step=10,
        )
        src_row.addWidget(self.src_spin)
        src_row.addWidget(QLabel(t("common.V")))
        self.apply_v_btn = QPushButton(t("cal.Wiz_apply_voltage"))
        self.apply_v_btn.clicked.connect(self._apply_voltage)
        src_row.addWidget(self.apply_v_btn)
        lay.addLayout(src_row)

        self.info_label = QLabel("")
        lay.addWidget(self.info_label)

        self.live_label = QLabel("")
        self.live_label.setStyleSheet(f"color: {COLOR_MID_GRAY};")
        lay.addWidget(self.live_label)

        self.dev_label = QLabel("")
        self.dev_label.setStyleSheet(STYLE_BOLD)
        lay.addWidget(self.dev_label)

        measure_row = QHBoxLayout()
        self.measure_btn = QPushButton(t("cal.Wiz_measure"))
        self.measure_btn.clicked.connect(self._do_measure)
        measure_row.addWidget(self.measure_btn)
        measure_row.addWidget(QLabel(t("common.Label_colon", label=t("common.N"))))
        self.n_spin = make_int_spinbox(
            min_val=1, max_val=100, value=cal_samples,
        )
        measure_row.addWidget(self.n_spin)
        lay.addLayout(measure_row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        meter_row = QHBoxLayout()
        meter_row.addWidget(QLabel(t("cal.Wiz_ammeter_reading")))
        self.meter_spin = make_double_spinbox(
            min_val=-500, max_val=500, value=0.0,
            decimals=3, step=0.1,
        )
        meter_row.addWidget(self.meter_spin)
        meter_row.addWidget(QLabel(unit))
        lay.addLayout(meter_row)

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_live)
        self._poll_timer.setInterval(1000)

    def initializePage(self) -> None:
        self._apply_voltage()
        self._poll_timer.start()

    def _apply_voltage(self) -> None:
        source_param = _DECODE[self.source_ch][0]
        v = self.src_spin.value()
        self.info_label.setText(
            t("cal.Wiz_setting_source_voltage",
              source=self.source_ch.upper(), value=int(v)))
        if not self._write_guard():
            # Emergency-lock / hw-busy changed while the wizard is open —
            # do not command the source (defense-in-depth, #14).
            self.info_label.setText(t("msg.Hw_busy"))
            return
        try:
            encoder = _ENCODE[self.source_ch]
            self.client.set_param(source_param, encoder(v))
            time.sleep(0.5)
        except (OSError, ValueError, RuntimeError, SerialException) as exc:
            # Expected comm/data error → show it; programming errors
            # propagate (failure-visibility pr. 1).
            self.info_label.setText(t("cal.Wiz_error_generic", error=str(exc)))


    def validatePage(self) -> bool:
        if not super().validatePage():
            return False
        # ML-032: apply_read picks ia_low/ia_high coefficients by the
        # DECODED current vs IA_RANGE_THRESHOLD at runtime — a calibration
        # point measured in the WRONG range would be stored for a range
        # the firmware never uses it in (wrong load resistor / source V).
        if self.channel in IA_RANGE_CHANNELS:
            in_low = self.dev_reading < IA_RANGE_THRESHOLD
            expected_low = self.channel == "ia_low"
            if in_low != expected_low:
                QMessageBox.warning(
                    self, t("cal.Wizard_title"),
                    t("cal.Wiz_current_wrong_range",
                      channel=self.channel,
                      measured=f"{self.dev_reading:.2f}",
                      threshold=f"{IA_RANGE_THRESHOLD:.0f}"))
                self._poll_timer.start()  # keep the live label running
                return False
        return True


class _VoltageResultPage(QWizardPage):
    """Result page: fits READ from the multimeter, derives SET from it.

    Plan B (docs/CALIBRATION_PLAN.md §5.6): the multimeter feeds the READ
    fit only; SET is the inverse of the DAC transfer observed through the
    freshly fitted READ — (commanded, device reading) pairs via
    ``CalibrationData.derive_set_two_point``.
    """

    def __init__(
        self,
        calibration: CalibrationData,
        channel: str,
        low_page: "_VoltageMeasurePage",
        high_page: "_VoltageMeasurePage",
        unit: str,
        has_set: bool,
    ) -> None:
        super().__init__()
        self.calibration = calibration
        self.channel = channel
        self.low_page = low_page
        self.high_page = high_page
        self.unit = unit
        self.has_set = has_set
        self.setTitle(t("cal.Wiz_results"))

        self.read_gain = 1.0
        self.read_offset = 0.0
        self.set_gain = 1.0
        self.set_offset = 0.0
        self._read_ok = False
        self._set_ok = False
        self._read_quality: Optional[Dict] = None
        self._set_quality: Optional[Dict] = None

        lay = QVBoxLayout(self)

        self.read_label = QLabel("")
        self.read_label.setWordWrap(True)
        lay.addWidget(self.read_label)
        self.read_quality_label = QLabel("")
        self.read_quality_label.setWordWrap(True)
        self.read_quality_label.setStyleSheet(STYLE_SECONDARY_SMALL)
        lay.addWidget(self.read_quality_label)
        self.apply_read_cb = QCheckBox(t("cal.Wiz_apply_read"))
        self.apply_read_cb.setChecked(True)
        self.apply_read_cb.setToolTip(t("cal.Wiz_apply_read_tip"))
        lay.addWidget(self.apply_read_cb)

        if has_set:
            lay.addSpacing(10)
            # SET is no longer fitted from the meter — surface the
            # derivation so the user does not look for a SET procedure.
            self.set_auto_label = QLabel(t("cal.Wiz_set_auto"))
            self.set_auto_label.setWordWrap(True)
            self.set_auto_label.setStyleSheet(STYLE_SECONDARY_SMALL)
            lay.addWidget(self.set_auto_label)
            self.set_label = QLabel("")
            self.set_label.setWordWrap(True)
            lay.addWidget(self.set_label)
            self.set_quality_label = QLabel("")
            self.set_quality_label.setWordWrap(True)
            self.set_quality_label.setStyleSheet(STYLE_SECONDARY_SMALL)
            lay.addWidget(self.set_quality_label)
            self.apply_set_cb = QCheckBox(t("cal.Wiz_apply_set"))
            self.apply_set_cb.setChecked(True)
            self.apply_set_cb.setToolTip(t("cal.Wiz_apply_set_tip"))
            lay.addWidget(self.apply_set_cb)
            # SET has no basis without READ — couple the checkboxes.
            self.apply_read_cb.toggled.connect(self._on_read_toggled)

    def _on_read_toggled(self, checked: bool) -> None:
        """SET is derived from READ: refusing READ removes the basis for
        SET, so the SET checkbox must follow (a disabled-but-checked box
        would still be written by validatePage)."""
        if checked:
            self.apply_set_cb.setEnabled(self._set_ok)
            self.apply_set_cb.setChecked(self._set_ok)
        else:
            self.apply_set_cb.setChecked(False)
            self.apply_set_cb.setEnabled(False)

    def initializePage(self) -> None:
        lo = self.low_page
        hi = self.high_page
        meter_pct = self.calibration.meter_accuracy_pct.get(
            self.channel, DEFAULT_METER_ACCURACY_PCT.get(self.channel, 1.0))

        # Re-entry (Back → re-measure → Next) must re-evaluate from
        # scratch — stale _set_ok/_read_ok from a previous visit would
        # otherwise let validatePage write outdated coefficients.
        self._read_ok = False
        self._set_ok = False
        self._read_quality = None
        self._set_quality = None
        self.read_quality_label.setText("")
        self.apply_read_cb.setEnabled(True)
        self.apply_read_cb.setChecked(True)
        if self.has_set:
            self.set_label.setText("")
            self.set_quality_label.setText("")

        # Sign normalization (plan B §4): the canonical ug1 domain in
        # lm19/ is negative physical volts, while the wizard commands a
        # positive magnitude and DMMs are commonly hooked up to show the
        # bias magnitude (+). Align meter and commanded values with the
        # (always negative) device reading so both fits stay in the
        # canonical domain — a positive meter entry must never produce a
        # negative READ gain. -abs() is idempotent for correctly signed
        # (already negative) entries.
        ref_lo, ref_hi = lo.meter_reading, hi.meter_reading
        cmd_lo, cmd_hi = lo.commanded, hi.commanded
        if self.channel == "ug1":
            ref_lo, ref_hi = -abs(ref_lo), -abs(ref_hi)
            cmd_lo, cmd_hi = -abs(cmd_lo), -abs(cmd_hi)

        # READ: multimeter (ref) vs device reading (dev)
        read_error: Optional[str] = None
        try:
            self.read_gain, self.read_offset = CalibrationData.compute_two_point(
                ref_lo, ref_hi,
                lo.dev_reading, hi.dev_reading,
            )
            # Refuse implausible fits before they go live: with plan B
            # feedforward a stored bad fit (meter unit/sign error) drives
            # every actuator command — and an internally consistent ×10
            # scale error has near-zero residuals, so quality display
            # alone cannot catch it.
            if not fit_within_bounds(
                    self.channel, self.read_gain, self.read_offset):
                raise ValueError(t(
                    "cal.Wiz_fit_out_of_bounds",
                    gain=f"{self.read_gain:.4f}",
                    offset=f"{self.read_offset:+.3f}", unit=self.unit))
            self._read_ok = True
            self.read_label.setText(
                t("cal.Wiz_read_result",
                  gain=f"{self.read_gain:.6f}",
                  offset=f"{self.read_offset:+.4f}",
                  unit=self.unit))

            if lo.dev_stats and hi.dev_stats:
                self._read_quality = _build_quality_dict(
                    ref_lo, ref_hi,
                    lo.dev_stats, hi.dev_stats,
                    self.read_gain, self.read_offset, meter_pct)
                self.read_quality_label.setText(
                    _format_quality(self._read_quality, self.unit))
        except ValueError as exc:
            # Expected data error (degenerate points, out-of-bounds fit);
            # programming errors propagate (failure-visibility pr. 1).
            read_error = str(exc)
            log.warning("READ fit for %s refused: %s", self.channel, exc)
            self.read_label.setText(t("cal.Wiz_read_error", error=read_error))
            self.apply_read_cb.setEnabled(False)
            self.apply_read_cb.setChecked(False)

        if not self.has_set:
            return

        # SET: derived from the FRESHLY fitted READ (plan B §5.6) — the
        # multimeter feeds READ only. Deriving through self.calibration
        # would silently use a stale stored READ and shift the result by
        # exactly the stale/fresh mismatch, so build a throwaway
        # CalibrationData carrying the fresh fit.
        if read_error is not None:
            # No READ fit — no basis for the derivation.
            self.set_label.setText(t("cal.Wiz_set_error", error=read_error))
            self.apply_set_cb.setEnabled(False)
            self.apply_set_cb.setChecked(False)
            return
        try:
            fresh = CalibrationData()
            fresh.set_channel(
                self.channel, "read", self.read_gain, self.read_offset)
            self.set_gain, self.set_offset = fresh.derive_set_two_point(
                self.channel,
                cmd_lo, lo.dev_reading,
                cmd_hi, hi.dev_reading,
            )
            if not fit_within_bounds(
                    self.channel, self.set_gain, self.set_offset):
                raise ValueError(t(
                    "cal.Wiz_fit_out_of_bounds",
                    gain=f"{self.set_gain:.4f}",
                    offset=f"{self.set_offset:+.3f}", unit=self.unit))
            self._set_ok = True
            # Re-entry reset leaves the SET checkbox following _set_ok
            # (via _on_read_toggled) — re-enable it on a fresh success.
            self.apply_set_cb.setEnabled(True)
            self.apply_set_cb.setChecked(True)
            self.set_label.setText(
                t("cal.Wiz_set_result",
                  gain=f"{self.set_gain:.6f}",
                  offset=f"{self.set_offset:+.4f}",
                  unit=self.unit))

            if lo.dev_stats and hi.dev_stats:
                # Quality in the SET model domain: ref = commanded,
                # "device" = physical actual through the fresh READ fit.
                self._set_quality = _build_quality_dict(
                    cmd_lo, cmd_hi,
                    _stats_through_read(
                        lo.dev_stats, self.read_gain, self.read_offset),
                    _stats_through_read(
                        hi.dev_stats, self.read_gain, self.read_offset),
                    self.set_gain, self.set_offset, meter_pct)
                self.set_quality_label.setText(
                    _format_quality(self._set_quality, self.unit))
        except ValueError as exc:
            log.warning("SET derivation for %s refused: %s", self.channel, exc)
            self.set_label.setText(t("cal.Wiz_set_error", error=str(exc)))
            self.apply_set_cb.setEnabled(False)
            self.apply_set_cb.setChecked(False)

    def validatePage(self) -> bool:
        read_applied = self._read_ok and self.apply_read_cb.isChecked()
        if read_applied:
            self.calibration.set_channel(
                self.channel, "read", self.read_gain, self.read_offset,
                quality=self._read_quality)
        # SET is only meaningful together with the READ it was derived
        # from — never write it when READ was not applied.
        set_written = (self.has_set and self._set_ok and read_applied
                       and self.apply_set_cb.isChecked())
        if set_written:
            self.calibration.set_channel(
                self.channel, "set", self.set_gain, self.set_offset,
                quality=self._set_quality)
        elif read_applied and self.has_set:
            # READ replaced but SET not rewritten (derivation refused or
            # user opt-out): a previously stored SET was derived from the
            # just-replaced READ — stale feedforward, reset it loudly.
            stored = self.calibration.channels.get(f"{self.channel}_set")
            if stored is not None and not stored.is_default():
                log.warning(
                    "READ for %s rewritten but SET was not — stored SET "
                    "is stale, reset to default; re-run the wizard to "
                    "derive a fresh SET", self.channel)
                self.calibration.reset_channel(self.channel, "set")
        return True


class _ReadResultPage(QWizardPage):
    """Result page for READ-only calibration (current channels) with quality."""

    def __init__(self, calibration, channel, low_page, high_page, unit):
        super().__init__()
        self.calibration = calibration
        self.channel = channel
        self.low_page = low_page
        self.high_page = high_page
        self.unit = unit
        self.setTitle(t("cal.Wiz_results"))
        self.gain = 1.0
        self.offset = 0.0
        self._ok = False
        self._quality: Optional[Dict] = None

        lay = QVBoxLayout(self)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        lay.addWidget(self.result_label)
        self.quality_label = QLabel("")
        self.quality_label.setWordWrap(True)
        self.quality_label.setStyleSheet(STYLE_SECONDARY_SMALL)
        lay.addWidget(self.quality_label)
        self.apply_cb = QCheckBox(t("cal.Wiz_apply_read"))
        self.apply_cb.setChecked(True)
        self.apply_cb.setToolTip(t("cal.Wiz_apply_read_tip"))
        lay.addWidget(self.apply_cb)

    def initializePage(self) -> None:
        lo = self.low_page
        hi = self.high_page
        meter_pct = self.calibration.meter_accuracy_pct.get(
            self.channel, DEFAULT_METER_ACCURACY_PCT.get(self.channel, 1.0))

        # Re-entry must re-evaluate from scratch (see _VoltageResultPage).
        self._ok = False
        self._quality = None
        self.quality_label.setText("")
        self.apply_cb.setEnabled(True)
        self.apply_cb.setChecked(True)

        try:
            self.gain, self.offset = CalibrationData.compute_two_point(
                lo.meter_reading, hi.meter_reading,
                lo.dev_reading, hi.dev_reading,
            )
            if not fit_within_bounds(self.channel, self.gain, self.offset):
                raise ValueError(t(
                    "cal.Wiz_fit_out_of_bounds",
                    gain=f"{self.gain:.4f}",
                    offset=f"{self.offset:+.3f}", unit=self.unit))
            self._ok = True
            self.result_label.setText(
                t("cal.Wiz_read_result",
                  gain=f"{self.gain:.6f}",
                  offset=f"{self.offset:+.4f}",
                  unit=self.unit))

            if lo.dev_stats and hi.dev_stats:
                self._quality = _build_quality_dict(
                    lo.meter_reading, hi.meter_reading,
                    lo.dev_stats, hi.dev_stats,
                    self.gain, self.offset, meter_pct)
                self.quality_label.setText(
                    _format_quality(self._quality, self.unit))
        except ValueError as exc:
            # Expected data error (degenerate points, out-of-bounds fit);
            # programming errors propagate (failure-visibility pr. 1).
            log.warning("READ fit for %s refused: %s", self.channel, exc)
            self.result_label.setText(t("cal.Wiz_read_error", error=str(exc)))
            self.apply_cb.setEnabled(False)
            self.apply_cb.setChecked(False)

    def validatePage(self) -> bool:
        if self._ok and self.apply_cb.isChecked():
            self.calibration.set_channel(
                self.channel, "read", self.gain, self.offset,
                quality=self._quality)
        return True


def _format_quality(q: Dict[str, Any], unit: str) -> str:
    """Format quality dict as human-readable text for the result page."""
    lines = []
    lo = q.get("low_point", {})
    hi = q.get("high_point", {})
    if lo:
        lines.append(t("cal.Wiz_quality_point",
                        label=t("cal.Wiz_low_point"),
                        sigma=f"{lo.get('dev_sigma', 0):.4f}",
                        n=lo.get("n", 0), unit=unit))
    if hi:
        lines.append(t("cal.Wiz_quality_point",
                        label=t("cal.Wiz_high_point"),
                        sigma=f"{hi.get('dev_sigma', 0):.4f}",
                        n=hi.get("n", 0), unit=unit))
    spread = q.get("point_spread", 0)
    lines.append(t("cal.Wiz_quality_spread",
                    spread=f"{spread:.3f}", unit=unit))
    res_lo = q.get("residual_low", 0)
    res_hi = q.get("residual_high", 0)
    lines.append(t("cal.Wiz_quality_residual",
                    low=f"{res_lo:.4f}", high=f"{res_hi:.4f}", unit=unit))
    pct = q.get("meter_accuracy_pct", 0)
    lines.append(t("cal.Wiz_quality_meter", pct=f"{pct:.1f}"))
    return "\n".join(lines)
