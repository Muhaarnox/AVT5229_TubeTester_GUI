"""Tests for CheckableComboBox widget."""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.checkable_combo import CheckableComboBox

pytestmark = [pytest.mark.smoke_ui]


@pytest.fixture(autouse=True)
def _ensure_qapp():
    QApplication.instance() or QApplication([])


@pytest.fixture
def combo():
    c = CheckableComboBox(placeholder="Test")
    yield c
    c.close()


# ═══════════════════════════════════════════════════════════════════
# 1. CREATION
# ══════════════════════════════════════════════════════���════════════

class TestCreation:
    def test_creates_without_error(self):
        c = CheckableComboBox()
        assert c is not None
        c.close()

    def test_custom_placeholder(self):
        c = CheckableComboBox(placeholder="Ug2")
        assert c._placeholder == "Ug2"
        c.close()

    def test_empty_initially(self, combo):
        assert combo.checked_values() == []
        assert combo.checked_keys() == []


# ══════════════════════════���════════════════════════════════���═══════
# 2. set_items (float values)
# ══���════════════════════════════════════════════════════════════════

class TestSetItems:
    def test_set_items_populates(self, combo):
        combo.set_items([100.0, 200.0, 300.0])
        assert combo._model.rowCount() == 3

    def test_all_checked_by_default(self, combo):
        combo.set_items([100.0, 200.0])
        assert combo.checked_values() == [100.0, 200.0]

    def test_checked_values_returns_floats(self, combo):
        combo.set_items([150.0])
        vals = combo.checked_values()
        assert vals == [150.0]
        assert isinstance(vals[0], float)

    def test_empty_list(self, combo):
        combo.set_items([])
        assert combo.checked_values() == []
        assert combo._model.rowCount() == 0

    def test_replace_items(self, combo):
        combo.set_items([100.0, 200.0])
        combo.set_items([300.0])
        assert combo._model.rowCount() == 1
        assert combo.checked_values() == [300.0]


# ═════════���═════════════════════════════════════════════════���═══════
# 3. set_string_items (key/label pairs)
# ═══════════════════��════════════════════════���══════════════════════

class TestSetStringItems:
    def test_populates_with_pairs(self, combo):
        combo.set_string_items([("a", "Alpha"), ("b", "Beta")])
        assert combo._model.rowCount() == 2

    def test_all_checked_when_no_filter(self, combo):
        combo.set_string_items([("a", "Alpha"), ("b", "Beta")])
        assert combo.checked_keys() == ["a", "b"]

    def test_checked_keys_filter(self, combo):
        combo.set_string_items(
            [("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")],
            checked_keys=["a", "c"],
        )
        assert combo.checked_keys() == ["a", "c"]

    def test_empty_checked_keys(self, combo):
        combo.set_string_items(
            [("a", "Alpha"), ("b", "Beta")],
            checked_keys=[],
        )
        assert combo.checked_keys() == []

    def test_none_checked_keys_means_all(self, combo):
        combo.set_string_items(
            [("x", "X"), ("y", "Y")],
            checked_keys=None,
        )
        assert combo.checked_keys() == ["x", "y"]

    def test_empty_items_list(self, combo):
        combo.set_string_items([])
        assert combo.checked_keys() == []


# ═══════════���════════════���═══════════════════════════════��══════════
# 4. set_all_checked
# ═════════════════════════════════���═════════════════════════════════

class TestSetAllChecked:
    def test_uncheck_all(self, combo):
        combo.set_items([1.0, 2.0, 3.0])
        combo.set_all_checked(False)
        assert combo.checked_values() == []

    def test_recheck_all(self, combo):
        combo.set_items([1.0, 2.0])
        combo.set_all_checked(False)
        combo.set_all_checked(True)
        assert combo.checked_values() == [1.0, 2.0]


# ══════════════════════════════════════════════════════════════════���
# 5. selectionChanged signal
# ═��════════════════════���════════════════════════════════════════════

class TestSignals:
    def test_signal_emitted_on_toggle(self, combo):
        combo.set_items([10.0, 20.0])
        received = []
        combo.selectionChanged.connect(lambda: received.append(True))
        # Manually uncheck first item
        item = combo._model.item(0)
        item.setCheckState(Qt.CheckState.Unchecked)
        assert len(received) > 0

    def test_no_signal_during_set_items(self, combo):
        received = []
        combo.selectionChanged.connect(lambda: received.append(True))
        combo.set_items([10.0, 20.0, 30.0])
        assert len(received) == 0

    def test_no_signal_during_set_string_items(self, combo):
        received = []
        combo.selectionChanged.connect(lambda: received.append(True))
        combo.set_string_items([("a", "A"), ("b", "B")])
        assert len(received) == 0


# ═══════════════════════════════════════════════════════════════════
# 6. Summary text
# ══���═════════════════════════════════════════════��══════════════════

class TestSummaryText:
    def test_empty_shows_placeholder(self, combo):
        assert combo._summary_text() == "Test"

    def test_all_checked_shows_all(self, combo):
        combo.set_items([1.0, 2.0])
        text = combo._summary_text()
        assert "All" in text or "all" in text.lower()

    def test_none_checked_shows_dash(self, combo):
        combo.set_items([1.0, 2.0])
        combo.set_all_checked(False)
        text = combo._summary_text()
        assert "—" in text

    def test_partial_shows_values(self, combo):
        combo.set_items([10.0, 20.0, 30.0])
        combo._model.item(1).setCheckState(Qt.CheckState.Unchecked)
        text = combo._summary_text()
        assert "10.0" in text
        assert "30.0" in text
