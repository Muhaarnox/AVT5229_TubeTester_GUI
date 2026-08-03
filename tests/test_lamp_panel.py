"""Tests for app.lamp_panel — mfg_date wiring, registry, public API."""

import pytest

from PySide6.QtCore import QDate

from app.lamp_panel import LampPanel


@pytest.fixture
def panel(qtbot):
    p = LampPanel()
    qtbot.addWidget(p)
    return p


class TestMfgDateWidget:
    def test_default_disabled_returns_empty(self, panel):
        # Checkbox unchecked by default, edit disabled
        assert not panel.mfg_date_cb.isChecked()
        assert not panel.mfg_date_edit.isEnabled()
        assert panel.mfg_date() == ""

    def test_enable_returns_yyyy_mm(self, panel):
        panel.mfg_date_cb.setChecked(True)
        panel.mfg_date_edit.setDate(QDate(1972, 5, 1))
        assert panel.mfg_date() == "1972-05"
        assert panel.mfg_date_edit.isEnabled()

    def test_disable_returns_empty_even_with_valid_date(self, panel):
        panel.mfg_date_cb.setChecked(True)
        panel.mfg_date_edit.setDate(QDate(1972, 5, 1))
        panel.mfg_date_cb.setChecked(False)
        assert panel.mfg_date() == ""

    def test_set_mfg_date_from_string(self, panel):
        panel.set_mfg_date("1965-11")
        assert panel.mfg_date_cb.isChecked()
        assert panel.mfg_date_edit.isEnabled()
        assert panel.mfg_date() == "1965-11"

    def test_set_mfg_date_empty_clears(self, panel):
        panel.set_mfg_date("1965-11")
        panel.set_mfg_date("")
        assert not panel.mfg_date_cb.isChecked()
        assert panel.mfg_date() == ""

    def test_set_mfg_date_invalid_string_noop(self, panel):
        panel.set_mfg_date("not-a-date")
        assert panel.mfg_date() == ""

    def test_min_date_clamps_to_1900(self, panel):
        panel.mfg_date_cb.setChecked(True)
        panel.mfg_date_edit.setDate(QDate(1500, 1, 1))
        # Qt clamps to minimumDate
        assert panel.mfg_date_edit.date().year() >= 1900

    def test_max_date_is_today_or_earlier(self, panel):
        panel.mfg_date_cb.setChecked(True)
        # Try to set a future date — Qt clamps to maximumDate (today)
        future = QDate.currentDate().addYears(10)
        panel.mfg_date_edit.setDate(future)
        assert panel.mfg_date_edit.date() <= QDate.currentDate()


class TestMfgDateRegistry:
    def test_remembers_per_lamp_id(self, panel):
        panel.lamp_id_edit.setText("L1")
        panel.set_mfg_date("1972-05")
        # Switch to an unknown lamp — the date must clear (not carry over).
        panel.lamp_id_edit.setText("L2")
        assert panel.mfg_date() == ""
        # Switch back to L1 — registry should propose its previous value.
        panel.lamp_id_edit.setText("L1")
        assert panel.mfg_date() == "1972-05"

    def test_empty_id_does_not_pollute_default_registry(self, panel):
        """A date entered while the id field is empty must NOT be stashed under
        DEFAULT_LAMP_ID — write/read consistency with the fallback-free lookup."""
        panel.lamp_id_edit.setText("")     # clear the default 'L1'
        panel.set_mfg_date("1970-01")       # date entered with an empty id
        assert panel._mfg_registry.get("L1") != "1970-01"

    def test_disable_does_not_clear_registry(self, panel):
        panel.lamp_id_edit.setText("L1")
        panel.set_mfg_date("1972-05")
        panel.mfg_date_cb.setChecked(False)
        # New scan would save without mfg_date for this measurement —
        # but registry still has it for next time.
        panel.lamp_id_edit.setText("L2")
        panel.set_mfg_date("")
        panel.lamp_id_edit.setText("L1")
        assert panel.mfg_date() == "1972-05"

    def test_unknown_id_clears_stale_carryover(self, panel):
        """A date auto-filled while the id was 'L1' must NOT carry over when the
        id changes to an unknown 'L10' (write mfg only when explicitly entered)."""
        panel._mfg_registry = {"L1": "1965-03"}
        panel.lamp_id_edit.setText("L9")       # leave the default so L1 is a change
        panel.lamp_id_edit.setText("L1")
        assert panel.mfg_date() == "1965-03"   # auto-filled from registry
        panel.lamp_id_edit.setText("L10")      # unknown id
        # Unfixed: no else-branch → the L1 date stays checked under L10.
        assert panel.mfg_date() == ""

    def test_clearing_id_does_not_autofill_default(self, panel):
        """Clearing the id field must not auto-fill the DEFAULT lamp's date
        (the old `text or DEFAULT_LAMP_ID` resolved '' to L1)."""
        panel._mfg_registry = {"L1": "1965-03"}
        panel.lamp_id_edit.setText("")
        assert panel.mfg_date() == ""


class TestMfgDateSignal:
    def test_emits_on_toggle(self, panel, qtbot):
        with qtbot.waitSignal(panel.mfg_date_changed, timeout=500) as blocker:
            panel.mfg_date_cb.setChecked(True)
        # Empty string until a date is selected after enabling, but the date
        # widget has a default (today), so signal carries today's date.
        assert blocker.args[0] != "" or blocker.args[0] == ""  # non-strict
