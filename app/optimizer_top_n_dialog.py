"""Top-N candidates picker dialog for the amplifier optimizer.

Modal dialog showing a sortable table of optimizer Pareto points with
parameters (Ub, Ug2, Ug1, Ra, swing), metrics (THD, Pout, P_A), and
the HD method used. Double-click a row or press Apply to commit the
selected operating point — `dlg.selected_point` then holds the chosen
OptPoint and `dlg.exec()` returns Accepted.

Used from main_window when the user clicks "Top candidates..." in the
optimizer status panel.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from app.widget_factory import FormattedNumericItem
from i18n_setup import t

if TYPE_CHECKING:
    from lm19.optimizer import OptPoint


# ── Module local constants ──
_PA_COL_INDEX = 8                     # P_A column is hidden when not PP
_RANK_COL_BG = QColor("#e8f4f8")      # subtle blue tint for rank column
_UL_COL_INDEX = 10                    # UL tap column hidden if all = 0
_TABLE_MIN_WIDTH = 760
_TABLE_MIN_HEIGHT = 320
_THD_PRECISION = 2
_POUT_PRECISION_W = 3
_PA_PRECISION_W = 2


class OptimizerTopNDialog(QDialog):
    """Modal dialog for selecting an OptPoint from a Pareto/top-N list."""

    def __init__(
        self,
        candidates: List["OptPoint"],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._candidates: List["OptPoint"] = list(candidates)
        self.selected_point: Optional["OptPoint"] = None

        self.setWindowTitle(t("amp.opt_top_n_title"))
        self.setMinimumSize(_TABLE_MIN_WIDTH, _TABLE_MIN_HEIGHT)

        layout = QVBoxLayout(self)

        self.table = self._build_table()
        layout.addWidget(self.table)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            t("amp.opt_apply_selected")
        )
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ── UI construction ──

    def _build_table(self) -> QTableWidget:
        headers = [
            "#",
            t("amp.opt_col_ub"),
            t("amp.opt_col_ug2"),
            t("amp.opt_col_ug1"),
            t("amp.opt_col_ra"),
            t("amp.opt_col_swing"),
            t("amp.opt_col_thd"),
            t("amp.opt_col_pout"),
            t("amp.opt_col_pa"),
            t("amp.opt_col_method"),
            t("amp.opt_col_ul_tap"),
        ]
        table = QTableWidget(len(self._candidates), len(headers), self)
        table.setHorizontalHeaderLabels(headers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        # Sorting must be enabled AFTER population — otherwise each setItem()
        # triggers a re-sort and rows shuffle on insertion.
        table.setSortingEnabled(False)
        table.doubleClicked.connect(lambda _idx: self._accept_selection())

        # P_A column meaningful only when at least one row reports > 0 (PP).
        any_pp = any(getattr(p, "p_classA_w", 0.0) > 0.0 for p in self._candidates)
        if not any_pp:
            table.setColumnHidden(_PA_COL_INDEX, True)

        # UL tap column meaningful only when at least one row has UL > 0
        any_ul = any(getattr(p, "ul_tap", 0.0) > 0.0 for p in self._candidates)
        if not any_ul:
            table.setColumnHidden(_UL_COL_INDEX, True)

        for row, pt in enumerate(self._candidates):
            self._populate_row(table, row, pt)

        # Enable sorting AFTER population is complete, then force ascending
        # rank order so display matches input (otherwise Qt may apply a
        # leftover descending sort indicator from the header).
        table.setSortingEnabled(True)
        table.sortByColumn(0, Qt.SortOrder.AscendingOrder)

        # Auto-resize columns to content; rank column tight
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        if len(headers) > 1:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
            table.setColumnWidth(0, 40)

        # Pre-select the first row (typically lowest THD)
        if self._candidates:
            table.selectRow(0)

        return table

    def _populate_row(
        self, table: QTableWidget, row: int, pt: "OptPoint",
    ) -> None:
        """Fill one row with values from an OptPoint."""
        cells: List[QTableWidgetItem] = []

        # Rank: numeric cell so the # column sorts 1,2,..,10 not "1","10","2".
        # UserRole still holds the original candidate index (int) for selection.
        rank_item = FormattedNumericItem(float(row + 1), str(row + 1))
        rank_item.setData(Qt.ItemDataRole.UserRole, row)
        cells.append(rank_item)

        cells.append(self._numeric_item(pt.ub, "{:.0f} V"))
        cells.append(self._numeric_item(pt.ug2, "{:.0f} V") if pt.ug2 > 0
                     else self._dash_item())
        cells.append(self._numeric_item(pt.ug1, "{:.1f} V"))
        cells.append(self._numeric_item(pt.ra, "{:.2f} kΩ"))
        cells.append(self._numeric_item(pt.half_swing, "{:.2f} V")
                     if pt.half_swing > 0 else self._dash_item())

        thd_item = self._numeric_item(pt.thd, f"{{:.{_THD_PRECISION}f}} %")
        cells.append(thd_item)

        cells.append(self._numeric_item(
            pt.pout_mw / 1000.0, f"{{:.{_POUT_PRECISION_W}f}} W",
        ))

        p_a = getattr(pt, "p_classA_w", 0.0) or 0.0
        if p_a > 0:
            cells.append(self._numeric_item(p_a, f"{{:.{_PA_PRECISION_W}f}} W"))
        else:
            cells.append(self._dash_item())

        cells.append(QTableWidgetItem(str(getattr(pt, "hd_method", "?"))))

        ul = getattr(pt, "ul_tap", 0.0) or 0.0
        if ul > 0.0:
            cells.append(self._numeric_item(ul * 100.0, "{:.0f} %"))
        else:
            cells.append(self._dash_item())

        # Color rank column subtly so the table isn't visually flat
        rank_item.setBackground(_RANK_COL_BG)

        for col, item in enumerate(cells):
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
            )
            table.setItem(row, col, item)

    @staticmethod
    def _numeric_item(value: float, fmt: str) -> QTableWidgetItem:
        """Create a NUMERICALLY-sortable cell holding a formatted number.

        Plain QTableWidgetItem sorts by the DisplayRole string (lexicographic:
        "10 kΩ" < "2 kΩ"); FormattedNumericItem compares the float in __lt__.
        """
        return FormattedNumericItem(float(value), fmt.format(value))

    @staticmethod
    def _dash_item() -> QTableWidgetItem:
        """A '—' cell that still sorts NUMERICALLY (value 0). A plain
        QTableWidgetItem here would mix with FormattedNumericItem in the column
        and break sorting (the C++ vs Python __lt__ paths disagree)."""
        return FormattedNumericItem(0.0, "—")

    # ── Actions ──

    def _accept_selection(self) -> None:
        """Resolve the currently selected row to an OptPoint and accept."""
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        # The rank in UserRole maps row → original candidate index.
        rank_item = self.table.item(rows[0].row(), 0)
        if rank_item is None:
            return
        original_idx = rank_item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(original_idx, int):
            return
        if 0 <= original_idx < len(self._candidates):
            self.selected_point = self._candidates[original_idx]
            self.accept()
