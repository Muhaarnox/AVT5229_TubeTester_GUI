"""Reusable Qt widget factories.

Replaces ~50 boilerplate ``QDoubleSpinBox`` / ``QSpinBox`` blocks across
``amp_control_panel`` / ``main_window`` / ``model_dialog`` / ``manual_tab``
/ calibration panels with two named factories that apply the project's
standard property combo (range, value, step, decimals, suffix, tooltip
via ``t()``, ``valueChanged`` signal hookup, optional fixed width).

Usage:

    self.ub_spin = make_double_spinbox(
        min_val=50.0, max_val=600.0, value=250.0,
        step=10.0, suffix=" V",
        tooltip_key="amp.ub_tip",
        on_change=self._on_setting_changed,
    )

The factories deliberately mirror the existing call ordering
(range → value → step → decimals → suffix → tooltip → signal) so
visible spinbox behavior (clamp on out-of-range value, signal firing
on initial setValue, etc.) does not change.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox, QGroupBox, QPushButton, QSpinBox, QStyle,
    QStyleOptionGroupBox, QTableWidgetItem, QWidget,
)

from i18n_setup import t

# ── module local constants ──
# Gap between a group-box title text and the button placed after it.
TITLE_ROW_BUTTON_GAP_PX = 8


class TitleRowButtonGroupBox(QGroupBox):
    """``QGroupBox`` with a small button in the title row, right after
    the title text.

    Qt has no title-row widget API for ``QGroupBox`` (the built-in
    checkbox via ``setCheckable`` is the only native occupant), so the
    button is a plain child repositioned on every resize. It uses
    otherwise-empty header space: zero extra rows or columns in the
    content layout, so the box loses a whole button row of height
    compared to placing the button in the grid.

    The title rectangle comes from the style (``SC_GroupBoxLabel``),
    not from font arithmetic — indents differ between styles.
    """

    def __init__(self, title: str, button: QPushButton,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        self._title_btn = button
        button.setParent(self)
        button.resize(button.sizeHint())

    def title_label_rect(self) -> QRect:
        opt = QStyleOptionGroupBox()
        self.initStyleOption(opt)
        return self.style().subControlRect(
            QStyle.ComplexControl.CC_GroupBox, opt,
            QStyle.SubControl.SC_GroupBoxLabel, self)

    def _place_title_button(self) -> None:
        rect = self.title_label_rect()
        btn = self._title_btn
        btn.resize(btn.sizeHint())
        y = rect.y() + (rect.height() - btn.height()) // 2
        btn.move(rect.right() + TITLE_ROW_BUTTON_GAP_PX, max(0, y))

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Covers the hidden-construction flow too: Qt delivers the
        # deferred resize on first show, so no showEvent override is
        # needed (verified — a show-only pin passes without one).
        super().resizeEvent(event)
        self._place_title_button()


class FormattedTextItem(QTableWidgetItem):
    """Table item with a frozen display string and a SEPARATE sort string.

    Plain ``QTableWidgetItem`` aliases DisplayRole and EditRole to one
    storage slot — ``setText(short)`` after ``setData(EditRole, full)``
    silently clobbers the sort key (ML-040: Mfg "1955-06" displayed as
    "55-06" sorted AFTER "26-07"). This twin of ``FormattedNumericItem``
    keeps the short display and sorts ``__lt__`` on the full string.
    """

    def __init__(self, full: str, display: str) -> None:
        super().__init__()
        self._full = full
        self._display = display
        super().setData(Qt.ItemDataRole.DisplayRole, display)

    def data(self, role: int):  # type: ignore[override]
        if role == Qt.ItemDataRole.EditRole:
            return self._full
        return super().data(role)

    def text(self) -> str:  # noqa: D102 — Qt override
        return self._display

    def __lt__(self, other: "QTableWidgetItem") -> bool:
        other_key = other.data(Qt.ItemDataRole.EditRole)
        return self._full < str(other_key if other_key is not None else "")


class FormattedNumericItem(QTableWidgetItem):
    """Numeric table cell with a frozen display string AND numeric sorting.

    Qt's default ``QTableWidgetItem`` coerces ``text()`` from the numeric
    ``EditRole`` (stripping trailing zeros: ``-7.0`` → ``-7``) and, worse,
    sorts by the DisplayRole STRING — so numeric columns sort lexicographically
    (``"10" < "2"``). This subclass keeps the formatted string for display and
    compares the original float in ``__lt__`` so sorting is numeric. ``EditRole``
    stays a float so CSV export / inline edit see a number, not a string.
    """

    def __init__(self, value: float, display: str) -> None:
        super().__init__()
        self._value: float = float(value)
        self._display: str = display
        self.setData(Qt.ItemDataRole.EditRole, float(value))

    def data(self, role: int):
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display
        return super().data(role)

    def __lt__(self, other) -> bool:
        if isinstance(other, FormattedNumericItem):
            return self._value < other._value
        # other is a plain QTableWidgetItem (e.g. a "—" dash cell). Do NOT call
        # super().__lt__() — on PySide6 that virtual call re-dispatches into this
        # Python override → infinite recursion. Compare against its numeric
        # EditRole if present, else fall back to a plain string compare.
        try:
            return self._value < float(other.data(Qt.ItemDataRole.EditRole))
        except (TypeError, ValueError):
            return self._display < (other.text() or "")

    def text(self) -> str:  # type: ignore[override]
        # Keep ``item.text()`` aligned with ``data(DisplayRole)`` so callers
        # (CSV export, copy-to-clipboard) see the formatted form.
        return self._display


def make_double_spinbox(
    *,
    min_val: float,
    max_val: float,
    value: float,
    step: Optional[float] = None,
    decimals: Optional[int] = None,
    suffix: str = "",
    prefix: str = "",
    tooltip_key: Optional[str] = None,
    on_change: Optional[Callable[[float], None]] = None,
    fixed_width: Optional[int] = None,
    keyboard_tracking: bool = True,
) -> QDoubleSpinBox:
    """Build a configured ``QDoubleSpinBox``.

    Args:
        min_val, max_val: spinbox range (inclusive).
        value: initial value (Qt clamps to range automatically).
        step: ``setSingleStep`` — omitted when ``None``.
        decimals: ``setDecimals`` — omitted when ``None``. Applied BEFORE
            ``setRange`` (Qt rounds bounds to the current decimals).
        suffix: ``setSuffix`` (e.g. ``" V"`` / ``" kΩ"``).
        prefix: ``setPrefix`` (e.g. ``"Ub "`` / ``"Ra "``).
        tooltip_key: i18n key passed through ``t()``.
        on_change: optional ``valueChanged`` slot (``float``).
        fixed_width: pixel width applied via ``setFixedWidth``.
        keyboard_tracking: pass ``False`` for spinboxes whose ``on_change``
            commands hardware — with Qt's default tracking, typing "12.6"
            fires valueChanged for the intermediate 1 and 12 too (ML-122).
    """
    sb = QDoubleSpinBox()
    # ML-048: setDecimals FIRST. Qt rounds bounds to the CURRENT decimals
    # at setRange time; Qt 6.10 happens to RESTORE the full-precision
    # bound on a later setDecimals, so the old order worked by accident —
    # this order does not depend on that undocumented restore.
    if decimals is not None:
        sb.setDecimals(decimals)
    sb.setRange(min_val, max_val)
    if step is not None:
        sb.setSingleStep(step)
    sb.setValue(value)
    if not keyboard_tracking:
        sb.setKeyboardTracking(False)
    if suffix:
        sb.setSuffix(suffix)
    if prefix:
        sb.setPrefix(prefix)
    if tooltip_key:
        sb.setToolTip(t(tooltip_key))
    if fixed_width is not None:
        sb.setFixedWidth(fixed_width)
    if on_change is not None:
        sb.valueChanged.connect(on_change)
    return sb


def make_int_spinbox(
    *,
    min_val: int,
    max_val: int,
    value: int,
    step: int = 1,
    suffix: str = "",
    tooltip_key: Optional[str] = None,
    on_change: Optional[Callable[[int], None]] = None,
    fixed_width: Optional[int] = None,
) -> QSpinBox:
    """Build a configured ``QSpinBox``.

    Same parameter shape as ``make_double_spinbox`` minus ``decimals``.
    """
    sb = QSpinBox()
    sb.setRange(min_val, max_val)
    sb.setSingleStep(step)
    sb.setValue(value)
    if suffix:
        sb.setSuffix(suffix)
    if tooltip_key:
        sb.setToolTip(t(tooltip_key))
    if fixed_width is not None:
        sb.setFixedWidth(fixed_width)
    if on_change is not None:
        sb.valueChanged.connect(on_change)
    return sb
