"""Import dialogs for external tube tester data.

ImportMetaDialog  -- fill missing metadata (tube type, Vs, Vh, lamp ID).
CsvImportDialog   -- CSV preview, separator, column mapping.
"""

from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from app.widget_factory import make_double_spinbox
from lm19.csv_import import VALID_KEYS, detect_columns, detect_separator, skip_comment_lines
from i18n_setup import t
from lm19.constants import (
    TOPOLOGY_PENTODE,
)
from lm19.constants import (
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)


# ======================================================================
# ImportMetaDialog -- shared dialog for missing parameters
# ======================================================================

class ImportMetaDialog(QDialog):
    """Dialog to fill metadata not present in the imported file.

    Shows fields for tube_type, lamp_id, name, description, topology mode,
    Ug2 (Vs), and Uh (Vh).
    Fields are pre-filled from *defaults* (e.g. guessed from filename).
    """

    def __init__(
        self,
        parent=None,
        *,
        defaults: Optional[Dict] = None,
        point_count: int = 0,
    ):
        super().__init__(parent)
        self.setWindowTitle(t("import.Meta_title"))
        self.setMinimumWidth(320)
        defaults = defaults or {}

        layout = QVBoxLayout(self)

        info = QLabel(t("msg.Import_success", count=point_count))
        layout.addWidget(info)

        form = QFormLayout()

        self.tube_type_edit = QLineEdit(str(defaults.get("tube_type", "")))
        self.tube_type_edit.setToolTip(t("tip.Import_tube_type"))
        form.addRow(t("import.Tube_type"), self.tube_type_edit)

        self.lamp_id_edit = QLineEdit(str(defaults.get("lamp_id", "")))
        self.lamp_id_edit.setToolTip(t("tip.Import_lamp_id"))
        form.addRow(t("import.Lamp_id"), self.lamp_id_edit)

        self.name_edit = QLineEdit(str(defaults.get("name", "")))
        self.name_edit.setToolTip(t("tip.Import_name"))
        form.addRow(t("import.Name"), self.name_edit)

        self.description_edit = QTextEdit(str(defaults.get("description", "")))
        self.description_edit.setMinimumHeight(90)
        self.description_edit.setToolTip(t("tip.Import_description"))
        form.addRow(t("import.Description"), self.description_edit)

        self.topology_combo = QComboBox()
        self.topology_combo.addItem(t("import.Mode_triode"), TOPOLOGY_TRIODE)
        self.topology_combo.addItem(t("import.Mode_triode_connected"), TOPOLOGY_TRIODE_CONNECTED)
        self.topology_combo.addItem(t("import.Mode_pentode"), TOPOLOGY_PENTODE)
        default_mode = str(defaults.get("ug2_mode", TOPOLOGY_PENTODE))
        idx = max(0, self.topology_combo.findData(default_mode))
        self.topology_combo.setCurrentIndex(idx)
        self.topology_combo.setToolTip(t("tip.Import_topology"))
        form.addRow(t("import.Topology"), self.topology_combo)

        self.ug2_spin = make_double_spinbox(
            min_val=0, max_val=400,
            value=float(defaults.get("vs", 0.0)),
            decimals=1, suffix=" V",
            tooltip_key="tip.Import_ug2",
        )
        form.addRow(t("import.Ug2"), self.ug2_spin)

        self.uh_spin = make_double_spinbox(
            min_val=0, max_val=20,
            value=float(defaults.get("vh", 0.0)),
            decimals=1, suffix=" V",
            tooltip_key="tip.Import_uh",
        )
        form.addRow(t("import.Uh"), self.uh_spin)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def tube_type(self) -> str:
        return self.tube_type_edit.text().strip() or t("import.Default_tube_type")

    @property
    def lamp_id(self) -> str:
        return self.lamp_id_edit.text().strip() or t("import.Default_lamp_id")

    @property
    def name(self) -> str:
        return self.name_edit.text().strip() or t("import.Default_name")

    @property
    def description(self) -> str:
        return self.description_edit.toPlainText().strip()

    @property
    def ug2_mode(self) -> str:
        return str(self.topology_combo.currentData())

    @property
    def ug2(self) -> float:
        return self.ug2_spin.value()

    @property
    def uh(self) -> float:
        return self.uh_spin.value()


# ======================================================================
# CsvImportDialog -- CSV preview + column mapping
# ======================================================================

class CsvImportDialog(QDialog):
    """Dialog for CSV import: preview, separator, column mapping."""

    _SKIP_TOKEN = "__skip__"
    _MAPPING_OPTIONS = [_SKIP_TOKEN] + list(VALID_KEYS)

    def __init__(self, parent=None, *, text: str = ""):
        super().__init__(parent)
        self.setWindowTitle(t("import.Csv_title"))
        self.setMinimumWidth(600)
        self.setMinimumHeight(450)
        self._text = text

        layout = QVBoxLayout(self)

        # --- Separator ---
        sep_group = QGroupBox(t("import.Separator"))
        sep_layout = QHBoxLayout(sep_group)
        self.rb_auto = QRadioButton(t("import.Auto"))
        self.rb_semi = QRadioButton(";")
        self.rb_comma = QRadioButton(",")
        self.rb_tab = QRadioButton(t("csv.Tab"))
        self.rb_auto.setChecked(True)
        for rb in (self.rb_auto, self.rb_semi, self.rb_comma, self.rb_tab):
            sep_layout.addWidget(rb)
            rb.toggled.connect(self._on_separator_changed)
        layout.addWidget(sep_group)

        # --- Preview ---
        preview_label = QLabel(t("import.Preview"))
        layout.addWidget(preview_label)
        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.preview_table.verticalHeader().setVisible(False)
        layout.addWidget(self.preview_table)

        # --- Column mapping ---
        mapping_label = QLabel(t("import.Column_mapping"))
        layout.addWidget(mapping_label)
        self.mapping_layout = QHBoxLayout()
        layout.addLayout(self.mapping_layout)
        self._mapping_combos: List[QComboBox] = []

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Initial render
        self._refresh()

    # ------------------------------------------------------------------

    @property
    def separator(self) -> str:
        if self.rb_semi.isChecked():
            return ";"
        if self.rb_comma.isChecked():
            return ","
        if self.rb_tab.isChecked():
            return "\t"
        # auto
        return detect_separator(self._text)

    @property
    def column_mapping(self) -> Dict[int, str]:
        """Return the current column mapping (col_index -> LM19 key)."""
        mapping: Dict[int, str] = {}
        for idx, combo in enumerate(self._mapping_combos):
            val = combo.currentData()
            if val != self._SKIP_TOKEN:
                mapping[idx] = str(val)
        return mapping

    # ------------------------------------------------------------------

    def _on_separator_changed(self, checked: bool = False) -> None:
        if checked:
            self._refresh()

    def _refresh(self) -> None:
        """Re-parse and refresh preview + mapping combos."""
        sep = self.separator
        lines = skip_comment_lines(self._text.splitlines())
        if not lines:
            return

        # Parse header and first N data rows
        header_cols = lines[0].split(sep)
        n_cols = len(header_cols)
        preview_rows = lines[1:11]

        # Update preview table
        self.preview_table.setColumnCount(n_cols)
        self.preview_table.setRowCount(len(preview_rows) + 1)
        self.preview_table.setHorizontalHeaderLabels(
            [t("import.Col_n", index=i) for i in range(n_cols)]
        )

        # Header row (row 0)
        for c, val in enumerate(header_cols):
            item = QTableWidgetItem(val.strip())
            item.setBackground(Qt.GlobalColor.lightGray)
            self.preview_table.setItem(0, c, item)

        # Data rows
        for r, line in enumerate(preview_rows):
            cols = line.split(sep)
            for c in range(n_cols):
                text = cols[c].strip() if c < len(cols) else ""
                self.preview_table.setItem(r + 1, c, QTableWidgetItem(text))

        self.preview_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        # Auto-detect mapping
        auto_map = detect_columns(header_cols)

        # Rebuild mapping combos
        for combo in self._mapping_combos:
            combo.setParent(None)
            combo.deleteLater()
        self._mapping_combos.clear()

        for i in range(n_cols):
            combo = QComboBox()
            combo.addItem(t("import.Skip"), self._SKIP_TOKEN)
            for key in VALID_KEYS:
                combo.addItem(key, key)
            # Set detected value
            if i in auto_map:
                idx = combo.findData(auto_map[i])
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            self._mapping_combos.append(combo)
            self.mapping_layout.addWidget(combo)
