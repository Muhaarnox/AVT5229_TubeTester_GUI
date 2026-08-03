"""Calibration tab — per-channel gain/offset table, live readings, manual edit, wizard."""

import logging
import statistics
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ui_theme import STYLE_BOLD, STYLE_BOLD_LARGE, STYLE_ITALIC_PREVIEW
from app.widget_factory import make_double_spinbox, make_int_spinbox
from lm19.calibration import (
    ALL_READ_CHANNELS,
    ALL_SET_CHANNELS,
    CHANNEL_UNITS,
    GAIN_BOUNDS,
    IA_RANGE_CHANNELS,
    OFFSET_BOUNDS,
    fit_within_bounds,
    CalibrationData,
    ChannelCal,
)
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
from lm19.app_config import AppConfig, calibration_path
from serial import SerialException

from i18n_setup import t
from lm19.constants import EPS

log = logging.getLogger(__name__)

_COLOR_CAL_OK = QColor(220, 245, 220)
_COLOR_CAL_MANUAL = QColor(255, 245, 210)
_COLOR_DEFAULT = QColor(255, 255, 255)

_DECODE_READ = {
    "ua": lambda raw: float(raw),
    "ug1": lambda raw: decode_ug1(raw),
    "ug2": lambda raw: float(raw),
    "uh": lambda raw: decode_uh(raw),
    "ih": lambda raw: decode_ih(raw),
    "ia": lambda raw: decode_ia(raw),
    "ig2": lambda raw: decode_ig2(raw),
}

# Build channel rows for the coefficients table.
# Ia is split into ia_low/ia_high, each READ-only.
_CHANNEL_ROWS: List[tuple] = []
for _ch in ALL_READ_CHANNELS:
    if _ch == "ia":
        for _ia_ch in IA_RANGE_CHANNELS:
            _CHANNEL_ROWS.append((_ia_ch, "read"))
    else:
        _CHANNEL_ROWS.append((_ch, "read"))
        if _ch in ALL_SET_CHANNELS:
            _CHANNEL_ROWS.append((_ch, "set"))

_PREVIEW_BUFFER_SIZE = 5

# Encode functions and limits for setting test values on the device
_SET_ENCODE = {
    "ua":  ("Ua",  lambda v: int(round(v)),        0, 300),
    "ug1": ("Ug1", encode_ug1,                     0, 24),
    "ug2": ("Ug2", lambda v: int(round(v)),        0, 300),
    "uh":  ("Uh",  encode_uh,                      0, 15),
    "ih":  ("Ih",  encode_ih,                      0, 2.5),
}

# Base channel name for ia_low/ia_high → "ia" for protocol access
def _base_channel(ch: str) -> str:
    return ch.replace("_low", "").replace("_high", "")


class CalibrationTab(QWidget):
    def __init__(
        self,
        get_client: Callable[[], Optional[LM19Serial]],
        get_calibration: Callable[[], CalibrationData],
        set_calibration: Callable[[CalibrationData], None],
        app_config: AppConfig,
        parent=None,
        get_write_locked: Callable[[], bool] = lambda: False,
        get_hw_busy: Callable[[], Optional[str]] = lambda: None,
    ):
        super().__init__(parent)
        self.get_client = get_client
        self.get_calibration = get_calibration
        self.set_calibration = set_calibration
        self.app_config = app_config
        # Hardware-write gates — calibration writes setpoints like every
        # other subsystem, so it must honour the emergency write-lock and
        # the hw-busy arbiter (#14). Safe no-op defaults keep direct
        # constructions (tests) working unchanged.
        self.get_write_locked = get_write_locked
        self.get_hw_busy = get_hw_busy
        self._snapshot: Optional[dict] = None
        self._raw_buffers: Dict[str, Deque[float]] = {
            ch: deque(maxlen=_PREVIEW_BUFFER_SIZE) for ch in ALL_READ_CHANNELS
        }
        self._build_ui()
        self._refresh_table()

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_live)
        self._poll_timer.setInterval(1000)

    # ── UI Build ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Horizontal splitter: Live readings (left) | Coefficients (right)
        table_splitter = QSplitter(Qt.Orientation.Horizontal)
        table_splitter.addWidget(self._build_live_table())
        table_splitter.addWidget(self._build_coefficients_table())
        table_splitter.setStretchFactor(0, 2)
        table_splitter.setStretchFactor(1, 3)
        root.addWidget(table_splitter, 1)

        # Bottom: actions + manual edit
        bottom = QHBoxLayout()
        bottom.addWidget(self._build_actions_group())
        bottom.addWidget(self._build_manual_edit_group())
        bottom.addStretch()
        root.addLayout(bottom)

    def _build_live_table(self) -> QGroupBox:
        self.live_group = QGroupBox(t("cal.Live_readings"))
        live_layout = QVBoxLayout(self.live_group)
        self.live_table = QTableWidget(len(ALL_READ_CHANNELS), 4)
        self.live_table.setHorizontalHeaderLabels([
            t("cal.Channel"), t("cal.Raw"), t("cal.Calibrated"), t("cal.Delta"),
        ])
        self.live_table.verticalHeader().setVisible(False)
        self.live_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.live_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        hdr = self.live_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setStretchLastSection(True)
        for row, ch in enumerate(ALL_READ_CHANNELS):
            unit = CHANNEL_UNITS.get(ch, "")
            self.live_table.setItem(row, 0, QTableWidgetItem(f"{ch.upper()} ({unit})"))
            for col in range(1, 4):
                self.live_table.setItem(row, col, QTableWidgetItem("—"))
        live_layout.addWidget(self.live_table)
        return self.live_group

    def _build_coefficients_table(self) -> QGroupBox:
        coeff_group = QGroupBox(t("cal.Coefficients"))
        coeff_layout = QVBoxLayout(coeff_group)
        self.coeff_table = QTableWidget(len(_CHANNEL_ROWS), 5)
        self.coeff_table.setHorizontalHeaderLabels([
            t("cal.Channel"), t("cal.Type"), t("cal.Gain"),
            t("cal.Offset"), t("cal.Calibrated_at"),
        ])
        self.coeff_table.verticalHeader().setVisible(False)
        self.coeff_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.coeff_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.coeff_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        hdr2 = self.coeff_table.horizontalHeader()
        hdr2.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.coeff_table.currentCellChanged.connect(self._on_coeff_selection)
        coeff_layout.addWidget(self.coeff_table)
        return coeff_group

    def _build_actions_group(self) -> QGroupBox:
        actions_group = QGroupBox(t("cal.Actions"))
        actions_group.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        actions_layout = QVBoxLayout(actions_group)
        self.wizard_btn = QPushButton(t("cal.Wizard"))
        self.wizard_btn.setToolTip(t("tip.Cal_wizard"))
        self.wizard_btn.clicked.connect(self._start_wizard)
        self.reset_sel_btn = QPushButton(t("cal.Reset_selected"))
        self.reset_sel_btn.setToolTip(t("tip.Cal_reset_selected"))
        self.reset_sel_btn.clicked.connect(self._reset_selected)
        self.reset_all_btn = QPushButton(t("cal.Reset_all"))
        self.reset_all_btn.setToolTip(t("tip.Cal_reset_all"))
        self.reset_all_btn.clicked.connect(self._reset_all)
        actions_layout.addWidget(self.wizard_btn)
        actions_layout.addWidget(self.reset_sel_btn)
        actions_layout.addWidget(self.reset_all_btn)
        actions_layout.addStretch()
        self.save_btn = QPushButton(t("cal.Save"))
        self.save_btn.setToolTip(t("tip.Cal_save"))
        self.save_btn.setStyleSheet(STYLE_BOLD)
        self.save_btn.clicked.connect(self._save)
        self.discard_btn = QPushButton(t("cal.Discard"))
        self.discard_btn.setToolTip(t("tip.Cal_discard"))
        self.discard_btn.clicked.connect(self._discard)
        self.status_label = QLabel("")
        actions_layout.addWidget(self.save_btn)
        actions_layout.addWidget(self.discard_btn)
        actions_layout.addWidget(self.status_label)
        return actions_group

    def _build_manual_edit_group(self) -> QGroupBox:
        edit_group = QGroupBox(t("cal.Manual_edit"))
        edit_outer = QVBoxLayout(edit_group)
        edit_outer.setSpacing(4)

        self.edit_channel_label = QLabel("—")
        self.edit_channel_label.setStyleSheet(STYLE_BOLD_LARGE)
        edit_outer.addWidget(self.edit_channel_label)

        columns = QHBoxLayout()
        columns.setSpacing(6)

        # Left column: coefficients
        coeff_box = QGroupBox(t("cal.Coefficients"))
        coeff_form = QFormLayout(coeff_box)
        coeff_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)

        self.gain_spin = make_double_spinbox(
            min_val=0.5, max_val=1.5, value=1.0,
            step=0.001, decimals=6,
            tooltip_key="tip.Cal_gain",
        )
        coeff_form.addRow(t("cal.Gain") + ":", self.gain_spin)

        self.offset_spin = make_double_spinbox(
            min_val=-50.0, max_val=50.0, value=0.0,
            step=0.01, decimals=4,
            tooltip_key="tip.Cal_offset",
        )
        self.offset_unit_label = QLabel("")
        offset_row = QHBoxLayout()
        offset_row.addWidget(self.offset_spin)
        offset_row.addWidget(self.offset_unit_label)
        coeff_form.addRow(t("cal.Offset") + ":", offset_row)

        self.meter_spin = make_double_spinbox(
            min_val=0.01, max_val=10.0, value=1.0,
            step=0.1, decimals=2, suffix=" %",
            tooltip_key="tip.Cal_meter_pct",
        )
        coeff_form.addRow(t("cal.Meter_pct") + ":", self.meter_spin)

        columns.addWidget(coeff_box)

        # Right column: test + preview
        test_box = QGroupBox(t("cal.Test"))
        test_layout = QVBoxLayout(test_box)

        test_row = QHBoxLayout()
        test_row.addWidget(QLabel(t("cal.Test_value") + ":"))
        self.test_spin = make_double_spinbox(
            min_val=0, max_val=300, value=0,
            step=1.0, decimals=2,
            tooltip_key="tip.Cal_test_value",
        )
        test_row.addWidget(self.test_spin)
        self.test_unit_label = QLabel("")
        test_row.addWidget(self.test_unit_label)
        self.test_set_btn = QPushButton(t("cal.Test_set"))
        self.test_set_btn.setToolTip(t("cal.Test_set_tip"))
        self.test_set_btn.clicked.connect(self._set_test_value)
        test_row.addWidget(self.test_set_btn)
        self.test_raw_btn = QPushButton(t("cal.Test_set_raw"))
        self.test_raw_btn.setToolTip(t("cal.Test_set_raw_tip"))
        self.test_raw_btn.clicked.connect(self._set_test_value_raw)
        test_row.addWidget(self.test_raw_btn)
        test_layout.addLayout(test_row)

        preview_hdr = QHBoxLayout()
        preview_hdr.addWidget(QLabel(t("cal.Preview_label") + ":"))
        preview_hdr.addStretch()
        preview_hdr.addWidget(QLabel(t("common.Label_colon", label=t("common.N"))))
        self.preview_n_spin = make_int_spinbox(
            min_val=1, max_val=20, value=_PREVIEW_BUFFER_SIZE,
            tooltip_key="cal.Preview_n_tip",
            on_change=self._on_preview_n_changed,
        )
        preview_hdr.addWidget(self.preview_n_spin)
        test_layout.addLayout(preview_hdr)

        self.preview_label = QLabel("—")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet(STYLE_ITALIC_PREVIEW)
        test_layout.addWidget(self.preview_label)
        test_layout.addStretch()

        columns.addWidget(test_box)

        edit_outer.addLayout(columns)

        self.gain_spin.valueChanged.connect(self._update_preview)
        self.offset_spin.valueChanged.connect(self._update_preview)

        self.both_ia_cb = QCheckBox(t("cal.Apply_both_ia"))
        self.both_ia_cb.setToolTip(t("tip.Cal_apply_both_ia"))
        self.both_ia_cb.setVisible(False)
        edit_outer.addWidget(self.both_ia_cb)

        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton(t("cal.Apply"))
        self.apply_btn.setToolTip(t("tip.Cal_apply_manual"))
        self.apply_btn.clicked.connect(self._apply_manual)
        self.reset_one_btn = QPushButton(t("cal.Reset"))
        self.reset_one_btn.setToolTip(t("tip.Cal_reset_one"))
        self.reset_one_btn.clicked.connect(self._reset_one)
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.reset_one_btn)
        btn_row.addStretch()
        edit_outer.addLayout(btn_row)
        edit_outer.addStretch()

        return edit_group

    # ── Table refresh ────────────────────────────────────────────────

    def _refresh_table(self) -> None:
        cal = self.get_calibration()
        for row, (ch, direction) in enumerate(_CHANNEL_ROWS):
            cc = cal.get_channel(ch, direction)
            unit = CHANNEL_UNITS.get(ch, "")

            name_item = QTableWidgetItem(ch.upper())
            type_item = QTableWidgetItem(direction.upper())
            gain_item = QTableWidgetItem(f"{cc.gain:.6f}")
            offset_item = QTableWidgetItem(f"{cc.offset:.4f} {unit}")
            cal_at = cc.calibrated_at or "—"
            date_item = QTableWidgetItem(cal_at)

            if cc.is_default():
                bg = _COLOR_DEFAULT
            elif cc.calibrated_at:
                bg = _COLOR_CAL_OK
            else:
                bg = _COLOR_CAL_MANUAL

            for item in (name_item, type_item, gain_item, offset_item, date_item):
                item.setBackground(bg)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            self.coeff_table.setItem(row, 0, name_item)
            self.coeff_table.setItem(row, 1, type_item)
            self.coeff_table.setItem(row, 2, gain_item)
            self.coeff_table.setItem(row, 3, offset_item)
            self.coeff_table.setItem(row, 4, date_item)

        self._update_status()

    def _update_status(self) -> None:
        cal = self.get_calibration()
        n = cal.calibrated_channels_count()
        dirty = cal.dirty
        parts = []
        if n > 0:
            parts.append(t("cal.Status_calibrated", count=n))
        if dirty:
            parts.append(t("cal.Status_modified"))
        self.status_label.setText(" | ".join(parts) if parts else t("cal.Status_defaults"))

    # ── Selection ────────────────────────────────────────────────────

    def _selected_channel(self) -> Optional[tuple]:
        row = self.coeff_table.currentRow()
        if row < 0 or row >= len(_CHANNEL_ROWS):
            return None
        return _CHANNEL_ROWS[row]

    def _on_coeff_selection(self, row: int, col: int, prev_row: int, prev_col: int) -> None:
        sel = self._selected_channel()
        if not sel:
            self.edit_channel_label.setText("—")
            self.both_ia_cb.setVisible(False)
            self.preview_label.setText("")
            self._set_test_box_enabled(False)
            self.meter_spin.setEnabled(False)
            return
        ch, direction = sel
        base = _base_channel(ch)
        unit = CHANNEL_UNITS.get(ch, "") or CHANNEL_UNITS.get(base, "")
        cal = self.get_calibration()
        cc = cal.get_channel(ch, direction)
        self.edit_channel_label.setText(f"{ch.upper()} {direction.upper()}")
        self.gain_spin.setValue(cc.gain)
        self.offset_spin.setValue(cc.offset)
        self.offset_unit_label.setText(unit)

        # Meter accuracy: editable for READ, disabled for SET
        is_read = direction == "read"
        self.meter_spin.setEnabled(is_read)
        pct = cal.meter_accuracy_pct.get(ch, 1.0)
        self.meter_spin.setValue(pct)

        # SET rows are read-only (plan B, docs/CALIBRATION_PLAN.md):
        # the wizard derives SET from READ — a manual edit would
        # silently desync the pair.
        set_tip = t("cal.Set_auto_tooltip")
        self.gain_spin.setEnabled(is_read)
        self.gain_spin.setToolTip(t("tip.Cal_gain") if is_read else set_tip)
        self.offset_spin.setEnabled(is_read)
        self.offset_spin.setToolTip(t("tip.Cal_offset") if is_read else set_tip)
        self.apply_btn.setEnabled(is_read)
        self.apply_btn.setToolTip(
            t("tip.Cal_apply_manual") if is_read else set_tip)

        # Test value — visible only for settable channels
        has_set = base in _SET_ENCODE
        self._set_test_box_enabled(has_set)
        if has_set:
            _, _, lo, hi = _SET_ENCODE[base]
            self.test_spin.setRange(lo, hi)
            self.test_unit_label.setText(unit)

        # Clear rolling buffer for the base channel
        buf = self._raw_buffers.get(base)
        if buf is not None:
            buf.clear()
        self.preview_label.setText("")

        is_ia_range = ch in IA_RANGE_CHANNELS
        self.both_ia_cb.setVisible(is_ia_range)
        if is_ia_range:
            self.both_ia_cb.setChecked(False)

    # ── Manual edit ──────────────────────────────────────────────────

    def _warn_stale_set(self, ch: str, cal: CalibrationData) -> None:
        """SET is wizard-derived from READ: changing READ behind its back
        leaves the stored SET pre-correcting commands with a stale model.
        Surface it (failure-visibility) — the fix is to re-run the wizard.
        """
        set_ch = cal.channels.get(f"{ch}_set")
        if set_ch is None or set_ch.is_default():
            return
        log.warning(
            "READ for '%s' changed manually while a derived SET is active — "
            "SET is now stale, re-run the wizard", ch)
        QMessageBox.information(
            self, t("cal.Set_stale_title"),
            t("cal.Set_stale_warning", channel=ch.upper()))

    def _apply_manual(self) -> None:
        sel = self._selected_channel()
        if not sel:
            return
        ch, direction = sel
        if direction == "set":
            # SET coefficients are wizard-derived from READ (plan B) —
            # guard against programmatic calls; the UI is disabled too.
            return
        cal = self.get_calibration()
        if self._snapshot is None:
            self._snapshot = cal.snapshot()

        gain = self.gain_spin.value()
        offset = self.offset_spin.value()
        if not fit_within_bounds(ch, gain, offset):
            # ML-030: out-of-bounds coefficients almost always mean a unit
            # or sign error, and with plan B feedforward they drive REAL
            # commands. Confirm explicitly (expert override stays possible).
            g_lo, g_hi = GAIN_BOUNDS["default"]
            base = ch.replace("_low", "").replace("_high", "")
            o_lo, o_hi = OFFSET_BOUNDS[base]
            reply = QMessageBox.question(
                self, t("cal.Manual_edit"),
                t("cal.Manual_out_of_bounds",
                  gain=f"{gain:g}", offset=f"{offset:g}",
                  g_lo=f"{g_lo:g}", g_hi=f"{g_hi:g}",
                  o_lo=f"{o_lo:g}", o_hi=f"{o_hi:g}"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            log.warning("Manual calibration outside sanity bounds applied "
                        "by user: %s %s gain=%g offset=%g",
                        ch, direction, gain, offset)
        cal.set_channel(ch, direction, gain, offset)

        if direction == "read":
            cal.meter_accuracy_pct[ch] = self.meter_spin.value()
            self._warn_stale_set(ch, cal)

        if ch in IA_RANGE_CHANNELS and self.both_ia_cb.isChecked():
            other = "ia_high" if ch == "ia_low" else "ia_low"
            cal.set_channel(other, direction, gain, offset)
            if direction == "read":
                cal.meter_accuracy_pct[other] = self.meter_spin.value()

        self._refresh_table()

    def _reset_one(self) -> None:
        sel = self._selected_channel()
        if not sel:
            return
        ch, direction = sel
        cal = self.get_calibration()
        if self._snapshot is None:
            self._snapshot = cal.snapshot()
        cal.reset_channel(ch, direction)
        if direction == "read":
            self._warn_stale_set(ch, cal)

        if ch in IA_RANGE_CHANNELS and self.both_ia_cb.isChecked():
            other = "ia_high" if ch == "ia_low" else "ia_low"
            cal.reset_channel(other, direction)

        self._refresh_table()
        self._on_coeff_selection(self.coeff_table.currentRow(), 0, -1, -1)

    # ── Test value / Preview ────────────────────────────────────────

    def _on_preview_n_changed(self, n: int) -> None:
        self._raw_buffers = {
            ch: deque(self._raw_buffers.get(ch, []), maxlen=n)
            for ch in ALL_READ_CHANNELS
        }
        self._update_preview()

    def _set_test_box_enabled(self, enabled: bool) -> None:
        self.test_spin.setEnabled(enabled)
        self.test_set_btn.setEnabled(enabled)
        self.test_raw_btn.setEnabled(enabled)

    def _can_write_or_warn(self) -> bool:
        """Gate every hardware write through the emergency-lock and the
        hw-busy arbiter — mirrors ManualTab._can_write_or_warn (#14)."""
        if self.get_write_locked():
            QMessageBox.warning(
                self, t('msg.Reset_all_title'), t('msg.Emergency_write_block'))
            return False
        busy = self.get_hw_busy()
        if busy:
            log.warning("Calibration write blocked — hardware busy: %s", busy)
            QMessageBox.warning(
                self, t('msg.Reset_all_title'), t('msg.Hw_busy'))
            return False
        return True

    def _send_test_value(self, use_calibration: bool) -> None:
        sel = self._selected_channel()
        if not sel:
            return
        ch, _ = sel
        base = _base_channel(ch)
        enc_info = _SET_ENCODE.get(base)
        if not enc_info:
            return
        # Single choke point for both "Test set" and "Test set raw" buttons.
        if not self._can_write_or_warn():
            return
        client = self.get_client()
        if not client or not client.is_open():
            QMessageBox.warning(self, t("msg.COM"), t("msg.Connect_first"))
            return
        param_name, encode_fn, _, _ = enc_info
        value = self.test_spin.value()
        if base == "ug1":
            # The test spinbox holds the bias magnitude (0..24); the
            # canonical lm19/ domain — and the SET calibration with its
            # (-24, 0) clamp — is negative physical volts, so convert at
            # the UI boundary. encode_ug1 takes abs() for the wire.
            value = -value
        if use_calibration:
            cal = self.get_calibration()
            value = cal.apply_set(base, value)
        raw_int = encode_fn(value)
        try:
            client.set_param(param_name, raw_int)
        except (OSError, ValueError, RuntimeError, SerialException) as exc:
            # ML-077: the operator pressed Test-set and nothing happened —
            # a silent log line reads as "command accepted". Narrow except
            # (principle 1): programming errors propagate.
            log.warning("Failed to set test value: %s", exc)
            QMessageBox.warning(
                self, t("msg.Error"),
                t("cal.Test_set_failed", param=param_name, error=str(exc)))

    def _set_test_value(self) -> None:
        self._send_test_value(use_calibration=True)

    def _set_test_value_raw(self) -> None:
        self._send_test_value(use_calibration=False)

    def _update_preview(self) -> None:
        sel = self._selected_channel()
        if not sel:
            self.preview_label.setText("")
            return
        ch, direction = sel
        base = _base_channel(ch)
        buf = self._raw_buffers.get(base)
        if not buf:
            self.preview_label.setText("")
            return
        mean_raw = statistics.mean(buf)
        gain = self.gain_spin.value()
        offset = self.offset_spin.value()
        preview_val = mean_raw * gain + offset
        unit = CHANNEL_UNITS.get(ch, "") or CHANNEL_UNITS.get(base, "")
        fmt = ".3f" if base == "ih" else ".2f"
        if base in ("ua", "ug2"):
            fmt = ".1f"
        self.preview_label.setText(
            t("cal.Preview_result",
              raw=f"{mean_raw:{fmt}}",
              result=f"{preview_val:{fmt}}",
              unit=unit,
              n=len(buf))
        )

    # ── Actions ──────────────────────────────────────────────────────

    def _reset_selected(self) -> None:
        sel = self._selected_channel()
        if not sel:
            return
        self._reset_one()

    def _reset_all(self) -> None:
        reply = QMessageBox.question(
            self, t("cal.Reset_all_title"),
            t("cal.Reset_all_confirm"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        cal = self.get_calibration()
        if self._snapshot is None:
            self._snapshot = cal.snapshot()
        cal.reset_all()
        self._refresh_table()

    def _start_wizard(self) -> None:
        sel = self._selected_channel()
        if not sel:
            QMessageBox.information(self, t("cal.Wizard"), t("cal.Select_channel_first"))
            return
        ch, _direction = sel
        # The wizard commands setpoints from its first page — block the
        # launch under emergency-lock / hw-busy (#14).
        if not self._can_write_or_warn():
            return
        from app.calibration_wizard import CalibrationWizard
        client = self.get_client()
        if not client or not client.is_open():
            QMessageBox.warning(self, t("msg.COM"), t("msg.Connect_first"))
            return
        cal = self.get_calibration()
        if self._snapshot is None:
            self._snapshot = cal.snapshot()
        wizard = CalibrationWizard(
            client=client,
            calibration=cal,
            channel=ch,
            cal_samples=self.app_config.cal_measure_samples,
            cal_interval_ms=self.app_config.cal_measure_interval_ms,
            parent=self,
            # Defense-in-depth: re-check before each per-page set_param in
            # case lock/busy changes while the (modal) wizard is open.
            write_guard=lambda: (not self.get_write_locked()
                                 and self.get_hw_busy() is None),
        )
        if wizard.exec():
            self._refresh_table()

    # ── Save / Discard ───────────────────────────────────────────────

    def _save(self) -> bool:
        """Save calibration; returns True on success (ML-031 callers
        must not treat a failed save as 'changes are safe')."""
        cal = self.get_calibration()
        try:
            cal.save(calibration_path())
            self._snapshot = None
            self._update_status()
            log.info("Calibration saved: %s", cal.summary())
            return True
        except (OSError, ValueError, TypeError) as exc:
            log.exception("Calibration save failed")
            QMessageBox.warning(self, t("msg.Error"), str(exc))
            return False

    def _discard(self) -> None:
        if self._snapshot is None:
            return
        cal = self.get_calibration()
        cal.restore(self._snapshot)
        self._snapshot = None
        self._refresh_table()
        sel = self._selected_channel()
        if sel:
            self._on_coeff_selection(self.coeff_table.currentRow(), 0, -1, -1)

    # ── Live polling ─────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._poll_timer.start()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._poll_timer.stop()

    def _poll_live(self) -> None:
        client = self.get_client()
        if not client or not client.is_open():
            return
        cal = self.get_calibration()
        try:
            raw_data = {}
            for ch in ALL_READ_CHANNELS:
                param = ch.capitalize()
                if param == "Ig2":
                    param = "Ig2"
                elif param == "Ia":
                    param = "Ia"
                raw_int = client.get_param(param, real=True)
                decode_fn = _DECODE_READ[ch]
                decoded = decode_fn(raw_int)
                calibrated = cal.apply_read(ch, decoded)
                raw_data[ch] = (decoded, calibrated)
                self._raw_buffers[ch].append(decoded)
        except (ValueError, KeyError, OSError) as exc:
            # Transient data error during live polling tick (protocol
            # mismatch, comm glitch, OS-level serial). Skip this tick;
            # next timer fire will retry. Programming errors
            # (AttributeError, TypeError, NameError) propagate.
            # DEBUG so persistent errors don't spam the log.
            log.debug("calibration live poll failed: %s: %s",
                      type(exc).__name__, exc)
            return

        for row, ch in enumerate(ALL_READ_CHANNELS):
            decoded, calibrated = raw_data[ch]
            unit = CHANNEL_UNITS.get(ch, "")
            fmt = ".2f" if ch not in ("ih",) else ".3f"
            if ch in ("ua", "ug2"):
                fmt = ".1f"

            self.live_table.item(row, 1).setText(f"{decoded:{fmt}} {unit}")
            self.live_table.item(row, 2).setText(f"{calibrated:{fmt}} {unit}")

            delta = calibrated - decoded
            if abs(decoded) > EPS:
                pct = delta / decoded * 100
                self.live_table.item(row, 3).setText(f"{delta:+{fmt}} ({pct:+.1f}%)")
            else:
                self.live_table.item(row, 3).setText(f"{delta:+{fmt}}")

        self._update_preview()

    # ── Unsaved changes guard ────────────────────────────────────────

    def has_unsaved_changes(self) -> bool:
        return self._snapshot is not None and self.get_calibration().dirty

    def prompt_save_if_dirty(self) -> bool:
        """Prompt user to save unsaved changes. Returns True if OK to proceed."""
        if not self.has_unsaved_changes():
            return True
        reply = QMessageBox.question(
            self, t("cal.Unsaved_title"),
            t("cal.Unsaved_msg"),
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            # ML-031: a failed save must keep the user here — returning
            # True would let the caller discard the unsaved calibration.
            return self._save()
        if reply == QMessageBox.StandardButton.Discard:
            self._discard()
            return True
        return False
