"""Smoke tests for ``app/widget_factory.py``.

Verifies that ``make_double_spinbox`` and ``make_int_spinbox`` apply
range/value/step/decimals/suffix/tooltip/signal correctly. These tests
fix the factory contract before migrating ~50 boilerplate call sites
across `amp_control_panel.py` / `main_window.py` / `model_dialog.py` /
`manual_tab.py` / calibration panels.
"""

from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PySide6.QtWidgets import QApplication

# Ensure a QApplication exists for spinbox construction
_qapp = QApplication.instance() or QApplication([])

from app.widget_factory import (
    make_double_spinbox, make_int_spinbox, FormattedNumericItem,
)


class TestFormattedNumericItem(unittest.TestCase):
    """Numeric-sorting table cell with a frozen display string."""

    def test_display_string_frozen(self):
        from PySide6.QtCore import Qt
        item = FormattedNumericItem(-7.0, "-7.0 V")
        self.assertEqual(item.text(), "-7.0 V")          # not float-coerced "-7"
        self.assertEqual(item.data(Qt.ItemDataRole.DisplayRole), "-7.0 V")
        self.assertEqual(item.data(Qt.ItemDataRole.EditRole), -7.0)  # float for CSV/sort

    def test_lt_numeric_not_lexicographic(self):
        # 2 < 9 < 10 numerically, not "10" < "2" < "9" lexicographically.
        a, b, c = (FormattedNumericItem(2.0, "2"), FormattedNumericItem(9.0, "9"),
                   FormattedNumericItem(10.0, "10"))
        self.assertTrue(a < b < c)
        self.assertFalse(c < a)

    def test_lt_does_not_recurse_in_mixed_table_sort(self):
        """Sorting a QTableWidget column that mixes FormattedNumericItem with a
        plain item must NOT RecursionError — super().__lt__ re-dispatches into
        this Python override on PySide6. Numeric rows keep numeric order."""
        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
        from PySide6.QtCore import Qt
        t = QTableWidget(3, 1)
        t.setItem(0, 0, FormattedNumericItem(250.0, "250"))
        t.setItem(1, 0, QTableWidgetItem("—"))   # plain (the recursion trigger)
        t.setItem(2, 0, FormattedNumericItem(150.0, "150"))
        t.sortItems(0, Qt.SortOrder.AscendingOrder)  # RecursionError on buggy super()
        nums = [t.item(r, 0).text() for r in range(3) if t.item(r, 0).text() != "—"]
        self.assertEqual(nums, ["150", "250"])


class TestMakeDoubleSpinbox(unittest.TestCase):
    """Coverage for ``make_double_spinbox`` factory."""

    def test_range_correct(self) -> None:
        sb = make_double_spinbox(min_val=0.0, max_val=100.0, value=50.0)
        self.assertAlmostEqual(sb.minimum(), 0.0)
        self.assertAlmostEqual(sb.maximum(), 100.0)

    def test_default_value(self) -> None:
        sb = make_double_spinbox(min_val=0.0, max_val=10.0, value=3.5)
        self.assertAlmostEqual(sb.value(), 3.5)

    def test_step_applied(self) -> None:
        sb = make_double_spinbox(
            min_val=0.0, max_val=10.0, value=5.0, step=0.25,
        )
        self.assertAlmostEqual(sb.singleStep(), 0.25)

    def test_decimals_applied(self) -> None:
        sb = make_double_spinbox(
            min_val=0.0, max_val=10.0, value=5.0, decimals=3,
        )
        self.assertEqual(sb.decimals(), 3)

    def test_suffix_applied(self) -> None:
        sb = make_double_spinbox(
            min_val=0.0, max_val=10.0, value=5.0, suffix=" kΩ",
        )
        self.assertEqual(sb.suffix(), " kΩ")

    def test_tooltip_via_t(self) -> None:
        # Use a known nonexistent key so t() returns the key itself; just
        # verify setToolTip was called with a non-empty string.
        sb = make_double_spinbox(
            min_val=0.0, max_val=10.0, value=5.0,
            tooltip_key="amp.ra_dc_tip",
        )
        self.assertTrue(len(sb.toolTip()) > 0)

    def test_signal_connected(self) -> None:
        captured = []

        def on_change(v: float) -> None:
            captured.append(v)

        sb = make_double_spinbox(
            min_val=0.0, max_val=10.0, value=5.0,
            on_change=on_change,
        )
        sb.setValue(7.5)
        self.assertEqual(captured, [7.5])

    def test_fixed_width_applied(self) -> None:
        sb = make_double_spinbox(
            min_val=0.0, max_val=10.0, value=5.0, fixed_width=80,
        )
        self.assertEqual(sb.minimumWidth(), 80)
        self.assertEqual(sb.maximumWidth(), 80)

    def test_no_optional_args_works(self) -> None:
        sb = make_double_spinbox(min_val=-5.0, max_val=5.0, value=0.0)
        self.assertAlmostEqual(sb.value(), 0.0)
        self.assertEqual(sb.suffix(), "")
        self.assertEqual(sb.toolTip(), "")

    def test_value_clamped_to_range(self) -> None:
        sb = make_double_spinbox(min_val=0.0, max_val=10.0, value=999.0)
        # Qt clamps value to range automatically — verify the factory
        # passes through correctly (range first, then value).
        self.assertAlmostEqual(sb.value(), 10.0)

    def test_prefix_applied(self) -> None:
        sb = make_double_spinbox(
            min_val=0.0, max_val=1000.0, value=250.0, prefix="Ub ",
        )
        self.assertEqual(sb.prefix(), "Ub ")


class TestMakeIntSpinbox(unittest.TestCase):
    """Coverage for ``make_int_spinbox`` factory."""

    def test_range_correct(self) -> None:
        sb = make_int_spinbox(min_val=1, max_val=100, value=50)
        self.assertEqual(sb.minimum(), 1)
        self.assertEqual(sb.maximum(), 100)

    def test_default_value(self) -> None:
        sb = make_int_spinbox(min_val=0, max_val=20, value=11)
        self.assertEqual(sb.value(), 11)

    def test_step_applied(self) -> None:
        sb = make_int_spinbox(min_val=0, max_val=100, value=50, step=5)
        self.assertEqual(sb.singleStep(), 5)

    def test_suffix_applied(self) -> None:
        sb = make_int_spinbox(
            min_val=0, max_val=100, value=50, suffix=" %",
        )
        self.assertEqual(sb.suffix(), " %")

    def test_signal_connected(self) -> None:
        captured = []

        def on_change(v: int) -> None:
            captured.append(v)

        sb = make_int_spinbox(
            min_val=0, max_val=100, value=10, on_change=on_change,
        )
        sb.setValue(42)
        self.assertEqual(captured, [42])

    def test_tooltip_via_t(self) -> None:
        sb = make_int_spinbox(
            min_val=0, max_val=100, value=10,
            tooltip_key="amp.ul_steps_tip",
        )
        self.assertTrue(len(sb.toolTip()) > 0)


if __name__ == "__main__":
    unittest.main()
