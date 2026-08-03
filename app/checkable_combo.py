"""QComboBox with checkable items for multi-select."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QComboBox
from i18n_setup import t


class CheckableComboBox(QComboBox):
    """Drop-down with checkboxes.  Emits *selectionChanged* when any
    item is toggled.  The collapsed text shows selected items or 'All'.
    """

    selectionChanged = Signal()

    def __init__(self, parent=None, *, placeholder: str = "Ug2"):
        super().__init__(parent)
        self._placeholder = placeholder
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        self._model.dataChanged.connect(self._on_data_changed)
        self._updating = False

    # ---- public API ----

    def set_items(self, values: list[float]) -> None:
        """Replace all items with *values*, all checked by default."""
        self._updating = True
        self._model.clear()
        for v in values:
            item = QStandardItem(f"{v:.1f}")
            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            item.setData(Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
            item.setData(v, Qt.ItemDataRole.UserRole)
            self._model.appendRow(item)
        self._updating = False
        self._update_text()

    def checked_values(self) -> list[float]:
        """Return list of checked Ug2 values."""
        out = []
        for row in range(self._model.rowCount()):
            item = self._model.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out

    def set_string_items(
        self,
        items: list[tuple[str, str]],
        *,
        checked_keys: list[str] | None = None,
    ) -> None:
        """Replace all items with (key, label) pairs.

        Args:
            items: list of (key, display_label) tuples.
            checked_keys: which keys to check. None = all checked.
        """
        self._updating = True
        self._model.clear()
        check_all = checked_keys is None
        for key, label in items:
            item = QStandardItem(label)
            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            checked = check_all or key in (checked_keys or [])
            state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            item.setData(state, Qt.ItemDataRole.CheckStateRole)
            item.setData(key, Qt.ItemDataRole.UserRole)
            self._model.appendRow(item)
        self._updating = False
        self._update_text()

    def checked_keys(self) -> list[str]:
        """Return list of checked string keys (UserRole data)."""
        out = []
        for row in range(self._model.rowCount()):
            item = self._model.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out

    def all_values(self) -> list:
        """Return all item values (UserRole), checked or not."""
        return [self._model.item(row).data(Qt.ItemDataRole.UserRole)
                for row in range(self._model.rowCount())]

    def set_checked_values(self, values: list, *, tol: float = 0.0) -> None:
        """Programmatically check only *values* (numeric match ± tol).

        Does NOT emit ``selectionChanged`` — preset application must not
        be confused with a user edit (the caller applies visibility
        explicitly).
        """
        self._updating = True
        for row in range(self._model.rowCount()):
            item = self._model.item(row)
            v = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(v, (int, float)):
                checked = any(abs(v - float(target)) <= tol
                              for target in values)
            else:
                checked = v in values
            item.setCheckState(Qt.CheckState.Checked if checked
                               else Qt.CheckState.Unchecked)
        self._updating = False
        self._update_text()

    def set_all_checked(self, checked: bool = True) -> None:
        self._updating = True
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in range(self._model.rowCount()):
            self._model.item(row).setCheckState(state)
        self._updating = False
        self._update_text()

    # ---- internals ----

    def _on_data_changed(self, *_args):
        self._update_text()
        if not self._updating:
            self.selectionChanged.emit()

    def _summary_text(self) -> str:
        total = self._model.rowCount()
        if total == 0:
            return self._placeholder
        checked_rows = [
            row for row in range(total)
            if self._model.item(row).checkState() == Qt.CheckState.Checked
        ]
        if len(checked_rows) == total:
            return f"{self._placeholder}: {t('plot.All')}"
        if not checked_rows:
            return f"{self._placeholder}: —"
        parts = [self._model.item(row).text() for row in checked_rows]
        return f"{self._placeholder}: {', '.join(parts)}"

    def _update_text(self):
        txt = self._summary_text()
        self.setCurrentText(txt)
        self.setToolTip(txt)

    def paintEvent(self, event):
        """Override to show our summary text instead of current-item text."""
        from PySide6.QtWidgets import QStyle, QStylePainter, QStyleOptionComboBox
        painter = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        opt.currentText = self._summary_text()
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt)
