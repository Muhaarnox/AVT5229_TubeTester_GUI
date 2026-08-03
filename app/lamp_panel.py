from typing import Dict, List, Optional, Tuple

from lm19.calibration import CalibrationData
from lm19.protocol import LM19Serial, decode_ih, decode_uh

from PySide6.QtCore import QDate, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QRadioButton,
    QVBoxLayout,
)

from app.ui_theme import COLOR_LIGHT_BLUE, MARGIN, SPACING_NORMAL
from i18n_setup import t
from lm19.config import LampConfig

DEFAULT_LAMP_ID = "L1"

# ── module local constants ──
_MFG_DATE_FORMAT = "MM.yyyy"
_MFG_DATE_MIN_YEAR = 1900
_MFG_JSON_FORMAT = "yyyy-MM"  # ISO-like, lexicographically sortable


class LampPanel(QGroupBox):
    """Reusable lamp-selection panel used by Measure and Health tabs."""

    tube_changed = Signal(str)
    lamp_id_changed = Signal(str)
    anode_changed = Signal(int)
    mfg_date_changed = Signal(str)

    def __init__(
        self,
        *,
        name_label: str = "",
        show_name: bool = True,
        parent=None,
    ) -> None:
        super().__init__(t("lamp.Lamp"), parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        outer.setSpacing(SPACING_NORMAL)
        # The operating-point line spans the full width UNDER this row, so
        # it starts at the socket letter instead of being indented into the
        # form column — the widest single string in the panel gets the
        # widest slot available to it.
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(SPACING_NORMAL)

        self.socket_label = QLabel("")
        font = QFont()
        font.setPointSize(28)
        font.setBold(True)
        self.socket_label.setFont(font)
        self.socket_label.setFixedWidth(44)
        self.socket_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.socket_label.setStyleSheet(f"QLabel {{ color: {COLOR_LIGHT_BLUE}; }}")
        self.socket_label.setToolTip(t("tip.Socket_type"))
        top_row.addWidget(self.socket_label)

        form = QFormLayout()

        self.tube_combo = QComboBox()
        self.tube_combo.setToolTip(t("tip.Lamp_type"))
        self.lamp_id_edit = QLineEdit(DEFAULT_LAMP_ID)
        self.lamp_id_edit.setToolTip(t("tip.Lamp_ID"))

        type_id_row = QHBoxLayout()
        type_id_row.addWidget(QLabel(t("lamp.Type")))
        type_id_row.addWidget(self.tube_combo)
        type_id_row.addWidget(QLabel(t("lamp.Lamp_ID")))
        type_id_row.addWidget(self.lamp_id_edit, 1)
        form.addRow(type_id_row)

        self.anode_group = QButtonGroup(self)
        self.anode_1 = QRadioButton("1")
        self.anode_2 = QRadioButton("2")
        self.anode_1.setChecked(True)
        self.anode_group.addButton(self.anode_1, 1)
        self.anode_group.addButton(self.anode_2, 2)
        self.anode_1.setToolTip(t("tip.Anode_select"))
        self.anode_2.setToolTip(t("tip.Anode_select"))
        anode_row = QHBoxLayout()
        anode_row.addWidget(self.anode_1)
        anode_row.addWidget(self.anode_2)
        anode_row.addStretch(1)
        self._anode_label_text = t("lamp.Anode")
        self._anode_row_widget = QHBoxLayout()
        form.addRow(self._anode_label_text, anode_row)
        self._anode_form_row_idx = form.rowCount() - 1

        self.name_edit = QLineEdit("")
        self.name_edit.setToolTip(t("tip.Measurement_name"))
        if show_name:
            form.addRow(name_label or t("lamp.Measurement_name"), self.name_edit)

        self.mfg_date_cb = QCheckBox(t("lamp.Mfg_date"))
        self.mfg_date_cb.setToolTip(t("tip.Mfg_date"))
        self.mfg_date_edit = QDateEdit()
        self.mfg_date_edit.setDisplayFormat(_MFG_DATE_FORMAT)
        self.mfg_date_edit.setCalendarPopup(True)
        self.mfg_date_edit.setMinimumDate(QDate(_MFG_DATE_MIN_YEAR, 1, 1))
        self.mfg_date_edit.setMaximumDate(QDate.currentDate())
        self.mfg_date_edit.setDate(QDate.currentDate())
        self.mfg_date_edit.setEnabled(False)
        self.mfg_date_edit.setToolTip(t("tip.Mfg_date"))
        mfg_row = QHBoxLayout()
        mfg_row.addWidget(self.mfg_date_cb)
        mfg_row.addWidget(self.mfg_date_edit, 1)
        form.addRow(mfg_row)

        # Registry of last-entered mfg_date per lamp_id (in-memory, session only)
        self._mfg_registry: Dict[str, str] = {}

        top_row.addLayout(form, 1)
        outer.addLayout(top_row)

        self.mode_label = QLabel(t("lamp.Mode"))
        outer.addWidget(self.mode_label)

        self.tube_combo.currentTextChanged.connect(self.tube_changed.emit)
        self.lamp_id_edit.textChanged.connect(self._on_lamp_id_changed)
        self.anode_group.idClicked.connect(self.anode_changed.emit)
        self.mfg_date_cb.toggled.connect(self._on_mfg_toggled)
        self.mfg_date_edit.dateChanged.connect(self._on_mfg_date_changed)

    # --- Public API ---

    def set_lamps(self, lamps: List[LampConfig], current: str = "") -> None:
        self.tube_combo.blockSignals(True)
        self.tube_combo.clear()
        for lamp in lamps:
            self.tube_combo.addItem(lamp.tube_type)
        idx = self.tube_combo.findText(current)
        self.tube_combo.setCurrentIndex(max(0, idx))
        self.tube_combo.blockSignals(False)

    def apply_lamp(self, lamp: LampConfig) -> None:
        self.socket_label.setText(lamp.socket if lamp.socket else "—")

        has_two = lamp.anodes >= 2
        self.anode_2.setEnabled(has_two)
        self._set_anode_visible(has_two)
        default_an = lamp.anode_default if has_two and lamp.anode_default in (1, 2) else 1
        (self.anode_2 if default_an == 2 else self.anode_1).setChecked(True)

        pa = (lamp.ua * lamp.ia) / 1000.0
        # ML-017: user-facing string — via t(), not hardcoded English.
        if lamp.is_triode:
            self.mode_label.setText(
                t("lamp.Mode_line_triode",
                  uh=f"{lamp.uh:.1f}", ih=f"{lamp.ih:.1f}",
                  ua=f"{lamp.ua:.0f}", ia=f"{lamp.ia:.0f}",
                  ug1=f"{lamp.ug1:.1f}", pa=f"{pa:.2f}")
            )
        else:
            pg2 = (lamp.ug2 * lamp.ig2) / 1000.0
            self.mode_label.setText(
                t("lamp.Mode_line_pentode",
                  uh=f"{lamp.uh:.1f}", ih=f"{lamp.ih:.1f}",
                  ua=f"{lamp.ua:.0f}", ia=f"{lamp.ia:.0f}",
                  ug1=f"{lamp.ug1:.1f}", ug2=f"{lamp.ug2:.0f}",
                  pa=f"{pa:.2f}", pg2=f"{pg2:.2f}")
            )

    def tube_type(self) -> str:
        return self.tube_combo.currentText().strip()

    def lamp_id(self) -> str:
        return self.lamp_id_edit.text().strip() or DEFAULT_LAMP_ID

    def name(self) -> str:
        return self.name_edit.text().strip()

    def anode(self) -> int:
        return self.anode_group.checkedId()

    def mfg_date(self) -> str:
        """Return manufacturing date as ``YYYY-MM`` string, or ``""`` if not set."""
        if not self.mfg_date_cb.isChecked():
            return ""
        return self.mfg_date_edit.date().toString(_MFG_JSON_FORMAT)

    def set_mfg_date(self, value: str) -> None:
        """Set manufacturing date from ``YYYY-MM`` string. Empty string clears.

        Programmatic setter: signals are blocked to avoid spurious change
        emissions, but the registry is updated explicitly so ``L1`` set via
        this path is remembered the same as ``L1`` set via the UI.
        """
        if not value:
            self.mfg_date_cb.blockSignals(True)
            self.mfg_date_cb.setChecked(False)
            self.mfg_date_edit.setEnabled(False)
            self.mfg_date_cb.blockSignals(False)
            return
        d = QDate.fromString(value, _MFG_JSON_FORMAT)
        if not d.isValid():
            return
        self.mfg_date_cb.blockSignals(True)
        self.mfg_date_edit.blockSignals(True)
        self.mfg_date_cb.setChecked(True)
        self.mfg_date_edit.setEnabled(True)
        self.mfg_date_edit.setDate(d)
        self.mfg_date_edit.blockSignals(False)
        self.mfg_date_cb.blockSignals(False)
        self._remember_current_mfg()

    def _on_lamp_id_changed(self, text: str) -> None:
        self.lamp_id_changed.emit(text)
        # Auto-fill mfg_date from registry if previously entered for this exact
        # lamp_id; otherwise snap the checkbox OFF so a date auto-filled during
        # an intermediate keystroke (e.g. 'L1' while typing 'L10') does not carry
        # over to an unrelated id. Use the raw id (no DEFAULT_LAMP_ID fallback) so
        # clearing the field does not auto-fill the default lamp's date.
        lid = text.strip()
        remembered = self._mfg_registry.get(lid) if lid else None
        if remembered:
            self.set_mfg_date(remembered)
        else:
            self.set_mfg_date("")

    def _on_mfg_toggled(self, checked: bool) -> None:
        self.mfg_date_edit.setEnabled(checked)
        if checked:
            self._remember_current_mfg()
        self.mfg_date_changed.emit(self.mfg_date())

    def _on_mfg_date_changed(self, _date) -> None:
        if self.mfg_date_cb.isChecked():
            self._remember_current_mfg()
            self.mfg_date_changed.emit(self.mfg_date())

    def _remember_current_mfg(self) -> None:
        value = self.mfg_date()
        if not value:
            return
        # Key off the RAW stripped id (no DEFAULT_LAMP_ID fallback) so write and
        # the lookup in _on_lamp_id_changed agree: an empty field must not stash
        # the date under 'L1' (where the lookup would never re-propose it).
        lid = self.lamp_id_edit.text().strip()
        if not lid:
            return
        self._mfg_registry[lid] = value

    def _set_anode_visible(self, visible: bool) -> None:
        layout = self.layout()
        if not layout:
            return
        form: Optional[QFormLayout] = None
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and isinstance(item.layout(), QFormLayout):
                form = item.layout()
                break
        if form is None:
            return
        for row in range(form.rowCount()):
            label_item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            field_item = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
            if label_item and label_item.widget():
                if label_item.widget().text() == self._anode_label_text:
                    label_item.widget().setVisible(visible)
                    if field_item and field_item.layout():
                        for j in range(field_item.layout().count()):
                            w = field_item.layout().itemAt(j).widget()
                            if w:
                                w.setVisible(visible)
                    break


def read_heater_level(
    client: LM19Serial,
    use_uh: bool,
    calibration: Optional[CalibrationData] = None,
) -> float:
    """Read the live heater value on one channel — V when *use_uh*, else A.

    *calibration* converts the device reading into the physical domain
    (plan B, docs/CALIBRATION_PLAN.md); ``None`` keeps the raw decode.
    """
    if use_uh:
        now = decode_uh(client.get_param("Uh", real=True))
        return calibration.apply_read("uh", now) if calibration else now
    now = decode_ih(client.get_param("Ih", real=True))
    return calibration.apply_read("ih", now) if calibration else now


def check_heater_level(
    client: LM19Serial,
    lamp: LampConfig,
    ratio: float,
    calibration: Optional[CalibrationData] = None,
) -> Tuple[float, float, str]:
    """Read current heater level and return (now_value, required_value, unit).

    Uses Uh (V) for voltage-heated lamps, Ih (A) for current-heated. The
    required value is the lamp nominal scaled by *ratio*.
    """
    use_uh = lamp.uh > 0
    now = read_heater_level(client, use_uh, calibration)
    nominal = lamp.uh if use_uh else lamp.ih
    return now, nominal * ratio, "V" if use_uh else "A"
