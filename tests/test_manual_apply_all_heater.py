"""Apply All must not put HV on a tube whose heater is not at a sane point.

Apply All commands the heater and Ua/Ug1/Ug2 in one shot, so on a cold or
starved tube the anode voltage lands on an unemissive cathode. The gate
checks TWO independent things, each reported on its own line:

* the SETPOINT is a normal operating point for the selected lamp — a 1 V
  setpoint held perfectly steady still starves a 6.3 V cathode, and an
  over-driven setpoint cooks it;
* the live reading is AT that setpoint — a deliberate reduced-heater
  experiment is legitimate, but only once the tube has settled there.

Both comparisons are two-sided and use ``app.json:
manual_heater_tolerance_pct``. Declining applies NOTHING.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.app_config import AppConfig  # noqa: E402
from lm19.calibration import CalibrationData  # noqa: E402
from lm19.constants import HEATER_NEAR_ZERO_A, HEATER_NEAR_ZERO_V  # noqa: E402
from lm19.protocol import encode_ih, encode_uh  # noqa: E402

# ── module local constants ──
_UH_SET = 6.3
_IH_SET = 0.8
_UA_SET = 250.0
_UG1_SET = -7.3
_UG2_SET = 200.0
_TOL = AppConfig().manual_heater_tolerance_pct


class _Spin:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def value(self) -> float:
        return self._value


class _Radio:
    def __init__(self, checked: bool) -> None:
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


def _lamp(uh: float = _UH_SET, ih: float = 0.0, tube_type: str = "EL84"):
    """Lamp card stand-in — the gate reads only these three fields."""
    return SimpleNamespace(tube_type=tube_type, uh=uh, ih=ih)


def _client(uh: float = _UH_SET, ih: float = _IH_SET):
    """Client whose live heater reads back the given physical values."""
    client = MagicMock()
    client.get_param.side_effect = lambda name, real=False: {
        "Uh": encode_uh(uh), "Ih": encode_ih(ih),
    }.get(name, 0)
    return client


def _stub(client, *, use_ih: bool = False, uh_set: float = _UH_SET,
          ih_set: float = _IH_SET, calibration=None, lamp=_lamp(),
          tolerance_pct: float = _TOL):
    """ManualTab stand-in: the methods under test only touch these attrs."""
    from app.manual_tab import ManualTab

    cfg = SimpleNamespace(manual_heater_tolerance_pct=tolerance_pct)
    stub = SimpleNamespace(
        heater_ih_radio=_Radio(use_ih),
        uh_spin=_Spin(uh_set),
        ih_spin=_Spin(ih_set),
        ua_spin=_Spin(_UA_SET),
        ug1_spin=_Spin(_UG1_SET),
        ug2_spin=_Spin(_UG2_SET),
        anode_group=SimpleNamespace(checkedId=lambda: 1),
        get_calibration=lambda: calibration or CalibrationData(),
        get_app_config=lambda: cfg,
        get_current_lamp=lambda: lamp,
        _can_write_or_warn=lambda: True,
        _get_client_or_warn=lambda: client,
        _off_by_more_than_tolerance=ManualTab._off_by_more_than_tolerance,
    )
    stub._heater_setpoint_reasons = (
        lambda *a: ManualTab._heater_setpoint_reasons(stub, *a))
    stub._heater_ready_or_confirmed = (
        lambda c: ManualTab._heater_ready_or_confirmed(stub, c))
    return stub


def _apply_all(stub):
    from app.manual_tab import ManualTab
    ManualTab._apply_all(stub)


def _hv_calls(client):
    """Every command that puts working-point voltage on the tube."""
    return [c for c in client.set_param.call_args_list
            if c.args and c.args[0] in ("Ua", "Ug1", "Ug2")]


def _ask(client, **stub_kw):
    """Run Apply All declining any dialog; return the reason keys shown."""
    with patch("app.manual_tab.QMessageBox") as mb, \
            patch("app.manual_tab.t", side_effect=lambda k, **kw: k):
        mb.question.return_value = mb.No
        _apply_all(_stub(client, **stub_kw))
    if not mb.question.called:
        return []
    return mb.question.call_args.args[2].split("\n\n")


class TestHeaterReady:
    """Setpoint matches the lamp AND the tube sits at it → no dialog."""

    def test_nominal_setpoint_reached_applies_without_dialog(self):
        client = _client(uh=_UH_SET)
        with patch("app.manual_tab.QMessageBox") as mb:
            _apply_all(_stub(client))
        mb.question.assert_not_called()
        mb.warning.assert_not_called()
        names = [c.args[0] for c in client.set_param.call_args_list]
        assert names == ["An", "Uh", "Ih", "Ug2", "Ua", "Ug1", "Ua", "Ug2"]

    @pytest.mark.parametrize("sign", (1.0, -1.0))
    def test_reading_inside_tolerance_passes(self, sign):
        """Both ends of the band: a deviation within it must not nag."""
        now = _UH_SET * (1.0 + sign * _TOL / 2 / 100.0)
        assert _ask(_client(uh=now)) == []

    @pytest.mark.parametrize("sign", (1.0, -1.0))
    def test_setpoint_inside_tolerance_passes(self, sign):
        """Same band on the setpoint-vs-lamp comparison."""
        target = _UH_SET * (1.0 + sign * _TOL / 2 / 100.0)
        assert _ask(_client(uh=target), uh_set=target) == []

    def test_current_heater_channel_uses_ih(self):
        """Twin channel: a current-heated lamp is judged on Ih, not Uh."""
        client = _client(uh=0.0, ih=_IH_SET)
        assert _ask(client, use_ih=True,
                    lamp=_lamp(uh=0.0, ih=_IH_SET)) == []


class TestSetpointVersusLamp:
    """The user's case: a starved setpoint held perfectly is still wrong."""

    def test_starved_setpoint_warns_even_when_reading_matches(self):
        """1 V setpoint, heater exactly at 1 V, lamp rated 6.3 V.

        Comparing the reading against the setpoint alone sees a perfect
        match — only the setpoint-vs-lamp check catches this.
        """
        reasons = _ask(_client(uh=1.0), uh_set=1.0)
        assert reasons == ["manual.Heater_reason_setpoint_off_lamp",
                           "manual.Heater_apply_anyway"]

    def test_over_driven_setpoint_warns(self):
        """The other end: 12.6 V into a 6.3 V heater cooks the cathode."""
        reasons = _ask(_client(uh=12.6), uh_set=12.6)
        assert "manual.Heater_reason_setpoint_off_lamp" in reasons

    def test_starved_setpoint_declined_commands_nothing(self):
        client = _client(uh=1.0)
        with patch("app.manual_tab.QMessageBox") as mb:
            mb.question.return_value = mb.No
            _apply_all(_stub(client, uh_set=1.0))
        assert client.set_param.call_args_list == []

    def test_current_channel_setpoint_off_lamp(self):
        """Twin: the Ih path compares against the lamp's rated Ih."""
        lamp = _lamp(uh=0.0, ih=_IH_SET)
        reasons = _ask(_client(ih=0.1), use_ih=True, ih_set=0.1, lamp=lamp)
        assert "manual.Heater_reason_setpoint_off_lamp" in reasons

    def test_no_lamp_selected_skips_the_lamp_check(self):
        """Without a card there is no nominal — only the reading is judged."""
        assert _ask(_client(uh=1.0), uh_set=1.0, lamp=None) == []

    def test_channel_mismatch_is_named(self):
        """Voltage-driving a current-heated lamp: no Uh rating to compare."""
        lamp = _lamp(uh=0.0, ih=_IH_SET)
        reasons = _ask(_client(uh=_UH_SET), lamp=lamp)
        assert "manual.Heater_reason_channel_mismatch" in reasons

    def test_both_levels_report_together(self):
        """Starved setpoint AND not settled → one dialog, both lines."""
        reasons = _ask(_client(uh=0.2), uh_set=1.0)
        assert reasons == ["manual.Heater_reason_setpoint_off_lamp",
                           "manual.Heater_reason_off",
                           "manual.Heater_apply_anyway"]


class TestReadingVersusSetpoint:
    """Second level: the tube must actually be at the setpoint."""

    def test_cold_heater_declined_commands_nothing(self):
        client = _client(uh=_UH_SET * 0.5)
        with patch("app.manual_tab.QMessageBox") as mb:
            mb.question.return_value = mb.No
            _apply_all(_stub(client))
        mb.question.assert_called_once()
        assert client.set_param.call_args_list == [], \
            "declined Apply All must not touch the device at all"

    def test_cold_heater_confirmed_applies_everything(self):
        client = _client(uh=_UH_SET * 0.5)
        with patch("app.manual_tab.QMessageBox") as mb:
            mb.question.return_value = mb.Yes
            _apply_all(_stub(client))
        names = [c.args[0] for c in client.set_param.call_args_list]
        assert names == ["An", "Uh", "Ih", "Ug2", "Ua", "Ug1", "Ua", "Ug2"]

    def test_over_driven_reading_also_asks(self):
        """Heater running above its own setpoint is not "ready" either."""
        reasons = _ask(_client(uh=_UH_SET * 1.3))
        assert reasons == ["manual.Heater_reason_off_setpoint",
                           "manual.Heater_apply_anyway"]

    def test_heater_off_message(self):
        client = _client(uh=HEATER_NEAR_ZERO_V / 2)
        assert "manual.Heater_reason_off" in _ask(client)

    def test_heater_off_message_current_channel(self):
        lamp = _lamp(uh=0.0, ih=_IH_SET)
        client = _client(ih=HEATER_NEAR_ZERO_A / 2)
        assert "manual.Heater_reason_off" in _ask(client, use_ih=True,
                                                  lamp=lamp)

    def test_below_setpoint_message(self):
        assert "manual.Heater_reason_off_setpoint" in _ask(
            _client(uh=_UH_SET * 0.5))

    def test_current_channel_deviation_asks(self):
        """Twin: the Ih path gates too (Uh sits at its setpoint here)."""
        lamp = _lamp(uh=0.0, ih=_IH_SET)
        client = _client(uh=_UH_SET, ih=_IH_SET * 0.4)
        assert "manual.Heater_reason_off_setpoint" in _ask(
            client, use_ih=True, lamp=lamp)


class TestZeroSetpoint:
    def test_zero_setpoint_message(self):
        """Setpoint 0 with a live heater: Apply All would switch it off."""
        reasons = _ask(_client(uh=_UH_SET), uh_set=0.0)
        assert "manual.Heater_reason_setpoint_zero" in reasons

    def test_zero_setpoint_never_divides_by_zero(self):
        client = _client(uh=_UH_SET)
        with patch("app.manual_tab.QMessageBox") as mb:
            mb.question.return_value = mb.No
            _apply_all(_stub(client, uh_set=0.0))  # must not raise
        assert not _hv_calls(client)

    def test_zero_setpoint_does_not_claim_lamp_mismatch_twice(self):
        """A zero setpoint is its own reason — but the lamp check still
        fires, since 0 V is not the rated heater either."""
        reasons = _ask(_client(uh=_UH_SET), uh_set=0.0)
        assert reasons.count("manual.Heater_reason_setpoint_zero") == 1


class TestToleranceFromConfig:
    """The band is configurable (app.json: manual_heater_tolerance_pct)."""

    def test_wider_tolerance_silences_a_deviation(self):
        """30% off the setpoint passes when the configured band is 50%."""
        assert _ask(_client(uh=_UH_SET * 0.7), tolerance_pct=50.0) == []

    def test_narrower_tolerance_catches_a_small_deviation(self):
        """The same 3% deviation asks once the band is 1%."""
        client = _client(uh=_UH_SET * 1.03)
        assert _ask(client, tolerance_pct=1.0) == [
            "manual.Heater_reason_off_setpoint", "manual.Heater_apply_anyway"]

    def test_shipped_default_is_documented_value(self):
        assert AppConfig().manual_heater_tolerance_pct == 10.0


class TestHeaterReadFailure:
    """Cannot confirm the heater → apply nothing (failure visibility)."""

    @pytest.mark.parametrize("exc", (OSError("port gone"),
                                     ValueError("bad frame"),
                                     TimeoutError("no answer")))
    def test_read_error_blocks_and_warns(self, exc):
        client = MagicMock()
        client.get_param.side_effect = exc
        with patch("app.manual_tab.QMessageBox") as mb:
            _apply_all(_stub(client))
        mb.warning.assert_called_once()
        mb.question.assert_not_called()
        assert client.set_param.call_args_list == []


class TestGateCallSite:
    """The gate is wired into Apply All and fed the real inputs."""

    def test_apply_all_consults_the_gate(self):
        """Spy: a future refactor dropping the call must fail here."""
        from app.manual_tab import ManualTab
        client = _client()
        stub = _stub(client)
        calls = []
        stub._heater_ready_or_confirmed = lambda c: (calls.append(c), False)[1]
        ManualTab._apply_all(stub)
        assert calls == [client]
        assert client.set_param.call_args_list == []

    def test_gate_reads_through_calibration(self):
        """Call-site pin: raw decode and calibrated value differ here, so a
        gate skipping calibration would judge the wrong number. Device holds
        6.3 V raw; a 0.5x READ gain makes it 3.15 V physical — far below the
        6.3 V setpoint, so the dialog must fire."""
        cal = CalibrationData()
        cal.set_channel("uh", "read", 0.5, 0.0)
        assert "manual.Heater_reason_off_setpoint" in _ask(
            _client(uh=_UH_SET), calibration=cal)


class TestOtherPathsUngated:
    """Deliberate scope pin: only Apply All asks."""

    def test_single_set_paths_do_not_consult_the_gate(self):
        import inspect
        from app.manual_tab import ManualTab
        for name in ("_set_ua", "_set_ug1", "_set_ug2"):
            src = inspect.getsource(getattr(ManualTab, name))
            assert "_heater_ready_or_confirmed" not in src, (
                f"{name} gained the Apply All gate — auto-apply would pop "
                "dialogs on every spinbox edit")


class TestInlineSetpointMarker:
    """5 % marker next to the heater fields (advisory, non-blocking).

    Deliberately tighter than the Apply All gate: the marker appears while
    a value is merely questionable, the dialog only when it is far enough
    out to interrupt for. Built on the real widgets — a stub cannot prove
    the label is wired to the spinbox signal.
    """

    @staticmethod
    def _tab(lamp=None):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from app.manual_tab import ManualTab
        return ManualTab(
            get_client=lambda: None,
            get_app_config=lambda: AppConfig(),
            get_calibration=lambda: CalibrationData(),
            get_write_locked=lambda: False,
            on_add_to_main_plot=lambda pts: None,
            get_current_lamp=lambda: lamp,
        )

    def test_no_marker_at_lamp_nominal(self):
        tab = self._tab(_lamp(uh=_UH_SET))
        tab.uh_spin.setValue(_UH_SET)
        assert tab.uh_warn_lbl.isHidden()

    def test_marker_appears_below_nominal(self):
        tab = self._tab(_lamp(uh=_UH_SET))
        tab.uh_spin.setValue(1.0)
        assert not tab.uh_warn_lbl.isHidden()
        assert tab.uh_warn_lbl.toolTip()

    def test_marker_appears_above_nominal(self):
        """Two-sided: an over-driven setpoint is flagged as well."""
        tab = self._tab(_lamp(uh=_UH_SET))
        tab.uh_spin.setValue(_UH_SET * 1.3)
        assert not tab.uh_warn_lbl.isHidden()

    def test_marker_clears_when_value_returns(self):
        tab = self._tab(_lamp(uh=_UH_SET))
        tab.uh_spin.setValue(1.0)
        tab.uh_spin.setValue(_UH_SET)
        assert tab.uh_warn_lbl.isHidden()
        assert tab.uh_warn_lbl.toolTip() == ""

    def test_marker_band_is_five_percent(self):
        """The advisory band is the live-panel badge band, not the gate's.

        6 % off must show the marker while the shipped 10 % gate stays
        quiet — this is what makes the two levels distinguishable.
        """
        from app.live_panel import HEATER_NOMINAL_TOLERANCE_PCT
        assert HEATER_NOMINAL_TOLERANCE_PCT < AppConfig().manual_heater_tolerance_pct
        tab = self._tab(_lamp(uh=_UH_SET))
        tab.uh_spin.setValue(_UH_SET * 1.06)
        assert not tab.uh_warn_lbl.isHidden()
        assert _ask(_client(uh=_UH_SET * 1.06), uh_set=_UH_SET * 1.06) == []

    def test_current_channel_marker(self):
        """Twin field: Ih is judged against the lamp's rated Ih."""
        tab = self._tab(_lamp(uh=0.0, ih=_IH_SET))
        tab.ih_spin.setValue(_IH_SET)
        assert tab.ih_warn_lbl.isHidden()
        tab.ih_spin.setValue(_IH_SET * 0.5)
        assert not tab.ih_warn_lbl.isHidden()

    def test_unused_channel_flagged_only_when_dialled_in(self):
        """A voltage-heated lamp: Ih=0 is normal, Ih>0 is the wrong channel."""
        tab = self._tab(_lamp(uh=_UH_SET, ih=0.0))
        tab.ih_spin.setValue(0.0)
        assert tab.ih_warn_lbl.isHidden()
        tab.ih_spin.setValue(0.5)
        assert not tab.ih_warn_lbl.isHidden()

    def test_no_lamp_no_marker(self):
        tab = self._tab(None)
        tab.uh_spin.setValue(1.0)
        assert tab.uh_warn_lbl.isHidden()

    def test_lamp_change_rejudges_markers(self):
        """Call-site pin: the setpoint did not move, the rating did."""
        lamp_box = {"lamp": _lamp(uh=_UH_SET)}
        tab = self._tab()
        tab.get_current_lamp = lambda: lamp_box["lamp"]
        tab.uh_spin.setValue(_UH_SET)
        assert tab.uh_warn_lbl.isHidden()
        lamp_box["lamp"] = _lamp(uh=12.6, tube_type="ECC83")
        tab.refresh_heater_setpoint_warnings()
        assert not tab.uh_warn_lbl.isHidden()

    def test_main_window_refreshes_markers_on_lamp_change(self):
        """The lamp-apply path must call the refresh — a spinbox signal
        never fires when only the selected lamp changes."""
        import inspect
        from app.main_window_settings import MainWindowSettings
        src = inspect.getsource(MainWindowSettings._apply_lamp)
        assert "refresh_heater_setpoint_warnings" in src


class TestMarkerLayoutStability:
    """The marker must not re-flow its row when it appears or clears.

    A plain hidden QLabel takes no space, so toggling it shifts every
    control to its right (and pushes the Set button out of the column the
    other rows keep).
    """

    @staticmethod
    def _tab():
        return TestInlineSetpointMarker._tab(_lamp(uh=_UH_SET))

    def test_heater_markers_retain_size_when_hidden(self):
        tab = self._tab()
        for label in (tab.uh_warn_lbl, tab.ih_warn_lbl):
            assert label.sizePolicy().retainSizeWhenHidden(), \
                "hidden marker collapses its row"

    def test_row_geometry_is_stable_across_toggle(self):
        """The Set button keeps its position when the marker turns on."""
        tab = self._tab()
        tab.show()
        tab.uh_spin.setValue(_UH_SET)
        assert tab.uh_warn_lbl.isHidden()
        before = tab.uh_btn.pos().x()
        tab.uh_spin.setValue(1.0)
        assert not tab.uh_warn_lbl.isHidden()
        assert tab.uh_btn.pos().x() == before, \
            "controls shifted when the marker appeared"
        tab.hide()

    def test_every_setpoint_row_reserves_the_same_slot(self):
        """All Set-values rows carry an equal-width slot, so their Set
        buttons line up whether or not the row can show a marker."""
        from app.ui_theme import MANUAL_WARN_COL_WIDTH
        tab = self._tab()
        slots = [tab.ua_warn_slot, tab.ug1_warn_slot, tab.ug2_warn_slot,
                 tab.uh_warn_lbl, tab.ih_warn_lbl]
        assert [s.width() for s in slots] == \
            [MANUAL_WARN_COL_WIDTH] * len(slots)

    def test_non_heater_slots_stay_empty(self):
        """The HV rows reserve space only — they never show a glyph."""
        tab = self._tab()
        for slot in (tab.ua_warn_slot, tab.ug1_warn_slot, tab.ug2_warn_slot):
            assert slot.text() == ""
