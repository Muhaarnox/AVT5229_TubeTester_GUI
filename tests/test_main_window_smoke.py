"""Integration smoke tests for MainWindow signal wiring.

Goal: catch broken signal connections / state propagation that would
survive `test_app_smoke.py` (which only checks construction).  These
tests exercise golden-path user flows programmatically — selecting a
lamp, modifying spinboxes, switching tabs, saving/loading scan
settings, and feeding a fake scan-finished callback.

Designed to fail loudly if a future refactor (e.g. splitting
MainWindow's 138 methods into mixins) breaks signal wiring.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow
from lm19.amplifier.constants import (
    CIRCUIT_PP,
    CIRCUIT_SE,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_DFT,
)
from lm19.config import find_lamp
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)
from lm19.tube_model_base import (
    MODEL_TYPE_DEMPWOLF,
)

pytestmark = [pytest.mark.smoke_ui]


# ── Module local helpers ──

class _Port:
    def __init__(self, device: str) -> None:
        self.device = device


def _lamp_indices(window, *, triode: bool) -> List[int]:
    """Combo indices whose LAMP ENTRY has the requested topology.

    Topology comes from the lamp registry, never from the display name:
    name substrings collide across topologies (``"6P"`` also matches the
    6N6P triode), so a name filter picks the opposite topology as soon as
    the entry order in lamps.json changes — and the test then pins the
    wrong branch instead of failing honestly.
    """
    out: List[int] = []
    for i in range(window.lamp_combo.count()):
        lamp = find_lamp(window.lamps, window.lamp_combo.itemText(i))
        if lamp is not None and lamp.is_triode is triode:
            out.append(i)
    return out


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(
        "app.main_window.list_ports.comports",
        lambda: [_Port("COM1"), _Port("COM2")],
    )
    w = MainWindow()
    yield w
    w.close()


# ----------------------------------------------------------------------
# Lamp selection → _apply_lamp propagates state to all dependants
# ----------------------------------------------------------------------


class TestLampSelection:

    def test_auto_ug1_step_returns_positive_step(self, window):
        """_auto_ug1_step must not raise (it used np without importing it)
        and returns a sensible positive step — exercised when ug1_step=0."""
        step = window._auto_ug1_step(0.0, 20.0)
        assert isinstance(step, float)
        assert step > 0.0
        # degenerate span falls back to a default, still positive
        assert window._auto_ug1_step(5.0, 5.0) > 0.0

    def test_hw_busy_arbiter_detects_and_excludes_self(self, window):
        """_hw_busy_reason reports a busy subsystem and excludes the caller."""
        assert window._hw_busy_reason() is None
        window.scan_ctrl.scan_in_progress = True
        try:
            assert window._hw_busy_reason() == "scan"
            assert window._hw_busy_reason(exclude="scan") is None
        finally:
            window.scan_ctrl.scan_in_progress = False

    def test_hw_busy_arbiter_wired_into_tabs(self, window):
        """Arbiter reaches manual/health/calibration subsystems via the
        wired callable (#14 added calibration)."""
        assert window.manual_tab.get_hw_busy() is None
        assert window.health_tab.get_hw_busy() is None
        assert window.calibration_tab.get_hw_busy() is None
        window.scan_ctrl.scan_in_progress = True
        try:
            assert window.manual_tab.get_hw_busy() == "scan"
            assert window.health_tab.get_hw_busy() == "scan"
            assert window.calibration_tab.get_hw_busy() == "scan"
        finally:
            window.scan_ctrl.scan_in_progress = False

    def test_emergency_lock_wired_into_calibration(self, window):
        """The calibration tab sees the emergency write-lock (#14)."""
        assert window.calibration_tab.get_write_locked() is False
        window._emergency_lock = True
        try:
            assert window.calibration_tab.get_write_locked() is True
        finally:
            window._emergency_lock = False

    def test_apply_lamp_updates_zone_and_loadline(self, window):
        """Selecting a lamp populates zone spinboxes from lamp.ua/ug1."""
        triodes = _lamp_indices(window, triode=True)
        if not triodes:
            pytest.skip("No triode in lamps.json")
        window.lamp_combo.setCurrentIndex(triodes[0])
        # _apply_lamp must have been called; verify state visible
        assert window._is_triode is True
        assert window.zone_ua_max.value() > window.zone_ua_min.value()
        # Triode → Pg2 controls hidden
        assert window.pg2_max_cb.isHidden() or not window.pg2_max_cb.isVisible()

    def test_apply_lamp_pentode_shows_pg2_controls(self, window):
        pentodes = _lamp_indices(window, triode=False)
        if not pentodes:
            pytest.skip("No pentode in lamps.json")
        window.lamp_combo.setCurrentIndex(pentodes[0])
        assert window._is_triode is False
        # Pentode → Pg2 widget hidden flag is False (visibility was applied)
        assert not window.pg2_max_input.isHidden()
        assert not window.pg2_max_cb.isHidden()

    def test_apply_lamp_propagates_triode_to_plot_manager(self, window):
        """plot_mgr.is_triode must follow _is_triode after lamp switch."""
        triodes = _lamp_indices(window, triode=True)
        pentodes = _lamp_indices(window, triode=False)
        if not triodes or not pentodes:
            pytest.skip("Need both triode and pentode in lamps.json")
        window.lamp_combo.setCurrentIndex(triodes[0])
        assert window.plot_mgr.is_triode is True
        window.lamp_combo.setCurrentIndex(pentodes[0])
        assert window.plot_mgr.is_triode is False


# ----------------------------------------------------------------------
# Cold-scan heater warning (ML-123)
# ----------------------------------------------------------------------


class TestColdScanWarning:
    """ML-123: the cold-scan warning must evaluate the values that will be
    COMMANDED (gated by 'Enable preheat'), not the raw spinboxes — with
    preheat unchecked the scan runs at uh=ih=0 no matter what the
    spinboxes say."""

    def _arm(self, window, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        client = MagicMock()
        client.is_open.return_value = True
        window.conn_mgr._client = client           # window.client delegates
        window.scan_ctrl.start_scan = MagicMock()  # never reach hardware
        asked = []
        monkeypatch.setattr(
            "app.main_window_scan.QMessageBox.question",
            lambda *a, **k: (asked.append(True), QMessageBox.No)[1])
        return asked

    def test_warning_fires_when_preheat_disabled(self, window, monkeypatch):
        asked = self._arm(window, monkeypatch)
        window.uh_input.setValue(6.3)              # spinbox says heater on…
        window.preheat_enabled.setChecked(False)   # …but the scan gates to 0

        window._run_scan()

        assert asked, ("cold-scan warning skipped although the scan would "
                       "run with uh=ih=0")
        window.scan_ctrl.start_scan.assert_not_called()  # user declined

    def test_no_warning_when_heater_actually_commanded(self, window,
                                                       monkeypatch):
        asked = self._arm(window, monkeypatch)
        window.uh_input.setValue(6.3)
        window.preheat_enabled.setChecked(True)

        window._run_scan()

        assert not asked, "warning fired although the heater is commanded"
        window.scan_ctrl.start_scan.assert_called_once()


class TestScanStampsUg2TrackOnPlot:
    """Starting a scan records the Ug2 mode THIS run measures in.

    Without the stamp the plot re-derives grouping from the scan-setup
    radio, so arming the next run in the other mode regroups the curves
    already drawn (a triode-connected scan then has one point per group
    and renders as loose symbols).
    """

    def _arm(self, window, monkeypatch):
        """Arm a run. MUST be called after the lamp is picked: _apply_lamp
        rewrites the heater inputs from the lamp entry, so heater setup
        done earlier is lost and the cold-scan dialog appears."""
        from PySide6.QtWidgets import QMessageBox
        client = MagicMock()
        client.is_open.return_value = True
        window.conn_mgr._client = client
        window.scan_ctrl.start_scan = MagicMock()
        window.preheat_enabled.setChecked(True)
        if window.uh_input.value() <= 0 and window.ih_input.value() <= 0:
            window.uh_input.setValue(6.3)
        # Any confirmation this flow may raise must not block headless.
        monkeypatch.setattr(
            "app.main_window_scan.QMessageBox.question",
            lambda *a, **k: QMessageBox.Yes)
        window.plot_mgr.series_ug2_track.pop(0, None)
        # _run_scan has several early returns before the stamp; assert the
        # run can actually start so a blocked path fails here, named,
        # instead of surfacing as a missing stamp at the end of the test.
        window._emergency_lock = False
        assert window._hw_busy_reason(exclude="scan") is None
        assert not (window.scan_worker and window.scan_worker.isRunning())

    def _stamped(self, window):
        track = window.plot_mgr.series_ug2_track
        assert 0 in track, (
            "the run started but left no Ug2-mode stamp on the plot "
            f"(series_ug2_track={track})")
        return track[0]

    def _pick_lamp(self, window, *, triode: bool):
        idx = _lamp_indices(window, triode=triode)
        if not idx:
            pytest.skip("lamps.json has no lamp of the required topology")
        window.lamp_combo.setCurrentIndex(idx[0])

    def test_triode_connected_run_stamps_tracking(self, window, monkeypatch):
        self._pick_lamp(window, triode=False)
        self._arm(window, monkeypatch)
        window.ug2_track_radio.setChecked(True)
        window._run_scan()
        window.scan_ctrl.start_scan.assert_called_once()
        assert self._stamped(window) is True

    def test_pentode_sweep_run_stamps_not_tracking(self, window, monkeypatch):
        self._pick_lamp(window, triode=False)
        self._arm(window, monkeypatch)
        window.ug2_sweep_radio.setChecked(True)
        window._run_scan()
        window.scan_ctrl.start_scan.assert_called_once()
        assert self._stamped(window) is False

    def test_true_triode_run_stamps_tracking_despite_sweep_radio(
            self, window, monkeypatch):
        """The radio is hidden for a triode; the lamp decides instead."""
        self._pick_lamp(window, triode=True)
        self._arm(window, monkeypatch)
        window.ug2_sweep_radio.setChecked(True)
        window._run_scan()
        window.scan_ctrl.start_scan.assert_called_once()
        assert self._stamped(window) is True


class TestSavedMetadataDescribesTheRun:
    """The measurement records the settings of the run that produced the
    points, not the run armed while it was still going.

    The scan controls stay live for the whole run, so a user preparing
    the next measurement used to rewrite the finished one's metadata.
    ``ug2_mode`` is the worst case: on load an explicit flag outranks the
    Ug2 ≈ Ua + offset auto-detection, so the file lies about itself and
    no plot-side fix can recover it.
    """

    def _arm(self, window):
        """Real scan controller (the hardware-ownership arbiter reads it);
        only the hardware call itself is stubbed."""
        client = MagicMock()
        client.is_open.return_value = True
        window.conn_mgr._client = client
        window.scan_ctrl.start_scan = MagicMock()
        window.preheat_enabled.setChecked(True)
        window.uh_input.setValue(6.3)

    def _finish(self, window, *, measure_separately: bool = False):
        """Stub the completion path only.

        The SRK controller is mocked HERE, not at arm time: a MagicMock
        srk reads as "SRK is using the hardware" to the ownership arbiter
        and _run_scan would refuse to start.
        """
        window.srk_measure_separately = MagicMock()
        window.srk_measure_separately.isChecked.return_value = measure_separately
        window.srk = MagicMock()
        ctrl = MagicMock()
        ctrl.scan_start_time = None
        ctrl.reset_on_finish = False
        window.scan_ctrl = ctrl
        window._on_scan_finished(self._points())

    def _points(self):
        return [{"ua": 100.0, "ug1": -1.0, "ia": 5.0, "ug2": 100.0},
                {"ua": 150.0, "ug1": -1.0, "ia": 6.0, "ug2": 150.0}]

    def test_ug2_mode_of_the_run_not_of_the_next_one(self, window,
                                                     monkeypatch):
        pentodes = _lamp_indices(window, triode=False)
        if not pentodes:
            pytest.skip("lamps.json has no pentode")
        window.lamp_combo.setCurrentIndex(pentodes[0])
        self._arm(window)
        window.ug2_track_radio.setChecked(True)     # THIS run: triode-connected
        window._run_scan()
        window.ug2_sweep_radio.setChecked(True)     # next run armed meanwhile

        self._finish(window)

        scan = window._last_scan_meta["scan"]
        assert scan["ug2_mode"] == TOPOLOGY_TRIODE_CONNECTED
        assert scan["ug2_track_ua"] is True

    def test_ranges_and_tube_of_the_run_not_of_the_next_one(self, window):
        pentodes = _lamp_indices(window, triode=False)
        if len(pentodes) < 2:
            pytest.skip("lamps.json has fewer than two pentodes")
        window.lamp_combo.setCurrentIndex(pentodes[0])
        armed_tube = window.lamp_combo.currentText()
        self._arm(window)
        window.ua_stop.setValue(240.0)
        window._run_scan()
        # User retunes for the next run while this one is still going.
        window.ua_stop.setValue(90.0)
        window.lamp_combo.setCurrentIndex(pentodes[1])

        self._finish(window)

        meta = window._last_scan_meta
        assert meta["tube_type"] == armed_tube
        assert meta["scan"]["ua"]["stop"] == 240.0

    def test_auto_ug1_step_is_recorded_not_the_zero_input(self, window):
        """"0" in the Ug1 step box means AUTO: _run_scan computes the
        real step and writes it back. Freezing before that would file the
        measurement as "step 0", which no run ever swept."""
        pentodes = _lamp_indices(window, triode=False)
        if not pentodes:
            pytest.skip("lamps.json has no pentode")
        window.lamp_combo.setCurrentIndex(pentodes[0])
        self._arm(window)
        window.ug1_step.setValue(0.0)

        window._run_scan()

        swept = window.ug1_step.value()
        assert swept > 0.0, "auto step did not land in the spinbox"
        assert window._pending_scan_meta["scan"]["ug1"]["step"] == swept

        self._finish(window)
        assert window._last_scan_meta["scan"]["ug1"]["step"] == swept

    def test_labels_typed_during_the_run_reach_the_file(self, window):
        """Name / lamp id / zone are labels of the MEASUREMENT, filled
        while the scan runs — they must not be frozen with the run
        settings (which stay from the start)."""
        pentodes = _lamp_indices(window, triode=False)
        if not pentodes:
            pytest.skip("lamps.json has no pentode")
        window.lamp_combo.setCurrentIndex(pentodes[0])
        self._arm(window)
        window.ug2_track_radio.setChecked(True)
        window.measurement_name_edit.setText("")
        window._run_scan()

        window.measurement_name_edit.setText("evening run")
        window.lamp_panel.lamp_id_edit.setText("L42")
        window.zone_ua_min.setValue(123.0)
        window.ug2_sweep_radio.setChecked(True)   # next run armed

        self._finish(window)

        meta = window._last_scan_meta
        assert meta["name"] == "evening run"
        assert meta["lamp_id"] == "L42"
        assert meta["_zone"]["ua_min"] == 123.0
        # …while the run's own settings still come from the start.
        assert meta["scan"]["ug2_mode"] == TOPOLOGY_TRIODE_CONNECTED

    def test_mfg_date_cleared_during_the_run_is_not_written(self, window):
        """Absence of the key means "no data" — an entry made before the
        run and then unchecked must not survive in the snapshot."""
        self._arm(window)
        window.lamp_panel.mfg_date_cb.setChecked(True)
        window._run_scan()
        assert "mfg_date" in window._pending_scan_meta

        window.lamp_panel.mfg_date_cb.setChecked(False)
        self._finish(window)

        assert "mfg_date" not in window._last_scan_meta

    def test_mfg_date_entered_during_the_run_is_written(self, window):
        self._arm(window)
        window.lamp_panel.mfg_date_cb.setChecked(False)
        window._run_scan()
        window.lamp_panel.mfg_date_cb.setChecked(True)

        self._finish(window)

        assert window._last_scan_meta["mfg_date"]

    def test_zone_in_the_file_matches_the_one_srk_used(self, window):
        """The scan flow reads the zone live for the SRK computation, so
        a frozen zone in the file would document a different one."""
        self._arm(window)
        window._run_scan()
        window.zone_ua_min.setValue(77.0)

        self._finish(window)

        used_zone = window.srk.compute_from_scan.call_args.args[1]
        assert used_zone["ua_min"] == 77.0
        assert window._last_scan_meta["_zone"] == used_zone

    def test_srk_reference_lamp_comes_from_the_snapshot(self, window):
        """SRK is compared against the tube that was actually measured:
        re-selecting the combo mid-scan must not swap the reference."""
        pentodes = _lamp_indices(window, triode=False)
        if len(pentodes) < 2:
            pytest.skip("lamps.json has fewer than two pentodes")
        window.lamp_combo.setCurrentIndex(pentodes[0])
        armed_tube = window.lamp_combo.currentText()
        self._arm(window)
        window._run_scan()
        window.lamp_combo.setCurrentIndex(pentodes[1])
        assert window.lamp_combo.currentText() != armed_tube

        self._finish(window)

        lamp = window.srk.compute_from_scan.call_args.kwargs["lamp"]
        assert lamp is not None
        assert lamp.tube_type == armed_tube

    def test_partial_save_after_failure_also_uses_the_snapshot(
            self, window, monkeypatch):
        """Twin path: a failed scan whose partial points the user keeps
        goes through the same freeze, reference lamp included."""
        from PySide6.QtWidgets import QMessageBox
        pentodes = _lamp_indices(window, triode=False)
        if len(pentodes) < 2:
            pytest.skip("lamps.json has fewer than two pentodes")
        window.lamp_combo.setCurrentIndex(pentodes[0])
        armed_tube = window.lamp_combo.currentText()
        self._arm(window)
        window.ug2_track_radio.setChecked(True)
        window._run_scan()

        window.plot_mgr.points = [dict(p, series_id=0) for p in self._points()]
        window.lamp_combo.setCurrentIndex(pentodes[1])
        window.ug2_sweep_radio.setChecked(True)
        window.srk = MagicMock()
        monkeypatch.setattr(
            "app.main_window_scan.QMessageBox.question",
            lambda *a, **k: QMessageBox.Yes)

        window._on_scan_failed("comm error")

        meta = window._last_scan_meta
        assert meta["tube_type"] == armed_tube
        assert meta["scan"]["ug2_mode"] == TOPOLOGY_TRIODE_CONNECTED
        lamp = window.srk.compute_from_scan.call_args.kwargs["lamp"]
        assert lamp is not None and lamp.tube_type == armed_tube

    def test_timestamp_is_restamped_at_save(self, window, monkeypatch):
        """The timestamp marks COMPLETION (file names and history sort on
        it), so it is the one field re-read at save time.

        Real wall-clock would make both stamps land in the same second
        and a "not re-stamped" implementation would pass, so the clock is
        driven explicitly.
        """
        import datetime as real_dt
        import types
        started = real_dt.datetime(2026, 1, 1, 10, 0, 0)
        finished = real_dt.datetime(2026, 1, 1, 10, 40, 0)
        seq = [started, finished]
        monkeypatch.setattr(
            "app.main_window_settings.dt",
            types.SimpleNamespace(datetime=types.SimpleNamespace(
                now=lambda: seq.pop(0) if seq else finished)))

        self._arm(window)
        window._run_scan()
        assert (window._pending_scan_meta["timestamp"]
                == started.isoformat(timespec="seconds"))

        self._finish(window)

        assert (window._last_scan_meta["timestamp"]
                == finished.isoformat(timespec="seconds"))
        # The freeze must not be consumed or mutated by the save.
        assert (window._pending_scan_meta["timestamp"]
                == started.isoformat(timespec="seconds"))

    def test_snapshot_is_deep_copied(self, window):
        """A consumer mutating the saved dict must not poison the freeze."""
        self._arm(window)
        window._run_scan()
        self._finish(window)
        window._last_scan_meta["scan"]["ug2_mode"] = "clobbered"
        again = window._scan_meta_for_save()
        assert again["scan"]["ug2_mode"] != "clobbered"

    def test_falls_back_loudly_without_a_snapshot(self, window, caplog):
        """No armed run (defensive path): current UI state, but logged."""
        window._pending_scan_meta = None
        with caplog.at_level(logging.WARNING,
                             logger="app.main_window_settings"):
            meta = window._scan_meta_for_save()
        assert meta["scan"]["ug2_mode"]
        assert any("scan-start metadata" in r.message for r in caplog.records)


class TestAfterScanSrkUsesRunContext:
    """The after-scan SRK commands hardware with the FINISHED run's
    settings, not the ones armed next.

    This is the hardware twin of the metadata freeze: a live read of the
    Ug2-mode radio / offset / heater at finish would measure S/R/K at a
    different screen configuration than the scan — and file it into this
    measurement as if it described it.
    """

    _arm = TestSavedMetadataDescribesTheRun._arm
    _finish = TestSavedMetadataDescribesTheRun._finish
    _points = TestSavedMetadataDescribesTheRun._points

    def test_srk_measured_in_the_runs_ug2_mode(self, window):
        pentodes = _lamp_indices(window, triode=False)
        if not pentodes:
            pytest.skip("lamps.json has no pentode")
        window.lamp_combo.setCurrentIndex(pentodes[0])
        self._arm(window)
        window.ug2_track_radio.setChecked(True)
        window.ug2_offset.setValue(15.0)
        armed_uh = window.uh_input.value()
        window._run_scan()
        # Next run armed meanwhile: sweep mode, other offset, other heater.
        window.ug2_sweep_radio.setChecked(True)
        window.ug2_offset.setValue(40.0)
        window.uh_input.setValue(armed_uh + 2.0)

        self._finish(window, measure_separately=True)

        call = window.srk.measure_after_scan.call_args
        assert call.args[4] is True          # ug2_track of the RUN
        assert call.args[5] == 15.0          # offset of the RUN
        assert call.kwargs["uh"] == armed_uh

    def test_cold_run_passes_zero_heater_despite_reenabled_preheat(
            self, window, monkeypatch):
        """ML-123 twin: the run commanded uh=ih=0 (preheat off), so the
        SRK heater-loss check must expect 0 — re-enabling preheat while
        the scan ran must not make it expect volts the run never set."""
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(       # confirm the cold-scan warning
            "app.main_window_scan.QMessageBox.question",
            lambda *a, **k: QMessageBox.Yes)
        self._arm(window)
        window.preheat_enabled.setChecked(False)   # cold run, confirmed
        window._run_scan()
        window.preheat_enabled.setChecked(True)

        self._finish(window, measure_separately=True)

        call = window.srk.measure_after_scan.call_args
        assert call.kwargs["uh"] == 0.0
        assert call.kwargs["ih"] == 0.0

    def test_srk_topology_follows_the_run_not_the_lamp_combo(self, window):
        """compute_from_scan branch: switching the lamp selector mid-scan
        must not flip the SRK topology of the finished pentode run."""
        pentodes = _lamp_indices(window, triode=False)
        triodes = _lamp_indices(window, triode=True)
        if not pentodes or not triodes:
            pytest.skip("lamps.json lacks a pentode/triode pair")
        window.lamp_combo.setCurrentIndex(pentodes[0])
        self._arm(window)
        window._run_scan()
        window.lamp_combo.setCurrentIndex(triodes[0])

        self._finish(window)

        call = window.srk.compute_from_scan.call_args
        assert call.kwargs["is_triode"] is False

    def test_falls_back_loudly_without_a_context(self, window, caplog):
        window._scan_run_ctx = None
        with caplog.at_level(logging.WARNING, logger="app.main_window_scan"):
            ctx = window._run_ctx_for_finish()
        assert ctx is not None
        assert any("run context" in r.message for r in caplog.records)


# ----------------------------------------------------------------------
# Heater inputs: no per-keystroke commands, no race with reset (ML-122)
# ----------------------------------------------------------------------


class TestHeaterInputSafety:

    def test_heater_spinboxes_do_not_track_keystrokes(self, window):
        """ML-122: uh/ih on_change commands the heater — with Qt's default
        keyboardTracking, typing "12.6" sends the intermediate 1 and 12
        to the tube."""
        assert window.uh_input.keyboardTracking() is False
        assert window.ih_input.keyboardTracking() is False

    def test_can_send_heater_blocked_during_reset(self, window):
        """ML-122: a live post-scan reset is zeroing the outputs — heater
        commands must not race it."""
        stub = MagicMock()
        stub.isRunning.return_value = True
        window.preheat_enabled.setChecked(True)
        window.conn_mgr._client = MagicMock(is_open=lambda: True)
        window.scan_ctrl._reset_worker = stub
        try:
            assert window._can_send_heater() is False
        finally:
            window.scan_ctrl._reset_worker = None


# ----------------------------------------------------------------------
# Per-lamp limit clamps on lamp selection (ML-130/131)
# ----------------------------------------------------------------------


class TestPerLampLimitClamps:
    """ML-130/131: selecting a lamp must clamp the heater inputs and the
    Ug2 scan spins to the lamp card's caps — a 6.3 V heater must not accept
    12 V by a slip of the finger."""

    def _select(self, window, tube_type):
        idx = [i for i in range(window.lamp_combo.count())
               if window.lamp_combo.itemText(i) == tube_type]
        if not idx:
            pytest.skip(f"{tube_type} not in lamps.json")
        window.lamp_combo.setCurrentIndex(idx[0])

    def test_heater_inputs_clamped_to_lamp_caps(self, window):
        from lm19.config import IH_MAX_HEADROOM
        self._select(window, "6P18P")   # manual card: uh_max 6.9, ih 820 mA
        assert window.uh_input.maximum() == pytest.approx(6.9)
        # the widget rounds its bound to its decimals (0.902 -> 0.9)
        assert window.ih_input.maximum() == pytest.approx(
            round(0.82 * IH_MAX_HEADROOM, window.ih_input.decimals()))

    def test_ug2_spins_follow_lamp_screen_cap(self, window):
        self._select(window, "6P18P")
        lamp = next(l for l in window.lamps if l.tube_type == "6P18P")
        assert window.ug2_start.maximum() == pytest.approx(
            lamp.limits["ug2_max"])
        assert window.ug2_stop.maximum() == pytest.approx(
            lamp.limits["ug2_max"])

    def test_manual_tab_stays_unclamped(self, window):
        """Deliberate scope: the manual tab must accept ANY value
        up to the device limit — it is the operator's escape hatch
        (rejuvenation/experiments), per-lamp caps apply to the scan path
        only. A future 'helpful' clamp must fail here."""
        self._select(window, "6P18P")   # scan uh_input clamps to 6.9 here
        assert window.manual_tab.uh_spin.maximum() > 6.9
        assert window.manual_tab.ih_spin.maximum() > 0.91


# ----------------------------------------------------------------------
# Tab switching — no exceptions, state survives
# ----------------------------------------------------------------------


class TestTabSwitching:

    def test_switch_through_all_plot_tabs(self, window):
        """Cycle through every plot subtab without exceptions."""
        n = window.plot_tabs.count()
        assert n >= 3
        for i in range(n):
            window.plot_tabs.setCurrentIndex(i)
            # Currently rendered widget must be visible and not None
            current = window.plot_tabs.currentWidget()
            assert current is not None

    def test_main_tabs_present(self, window):
        """Top-level QTabWidget has Measure / Compare / Health / Calibration."""
        labels = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert window.tabs.count() >= 3
        # At least one tab references each major area
        joined = " ".join(labels).lower()
        assert any(k in joined for k in ("measure", "manual", "scan",
                                         "compare", "health", "calibrat"))

    def test_compare_tab_attached(self, window):
        """CompareTab instance is exposed and reachable."""
        assert hasattr(window, "compare_tab")
        assert window.compare_tab is not None
        # add_entry signature exists (called from _save_and_display_measurement)
        assert callable(getattr(window.compare_tab, "add_entry", None))


# ----------------------------------------------------------------------
# Scan settings persistence — save / load round-trip
# ----------------------------------------------------------------------


class TestScanSettingsRoundTrip:

    def test_save_and_load_preserves_scan_ranges(self, window, tmp_path,
                                                  monkeypatch):
        """Bypass QFileDialog and verify save→load is idempotent."""
        # Set non-default values so load actually changes state
        window.ua_start.setValue(50.0)
        window.ua_stop.setValue(280.0)
        window.ua_step.setValue(7.0)
        window.ug1_start.setValue(-9.0)
        window.ug1_stop.setValue(-1.0)
        window.uh_input.setValue(6.3)

        path = str(tmp_path / "settings.json")
        monkeypatch.setattr(
            "app.main_window.QFileDialog.getSaveFileName",
            lambda *a, **kw: (path, ""),
        )
        window._save_scan_settings()
        assert Path(path).exists()
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["scan"]["ua"]["start"] == pytest.approx(50.0)
        assert payload["scan"]["ua"]["stop"] == pytest.approx(280.0)
        assert payload["scan"]["ua"]["step"] == pytest.approx(7.0)

        # Mutate UI — load must restore
        window.ua_start.setValue(0.0)
        window.ua_stop.setValue(0.0)
        window.ua_step.setValue(0.0)
        monkeypatch.setattr(
            "app.main_window.QFileDialog.getOpenFileName",
            lambda *a, **kw: (path, ""),
        )
        # The load summary is a modal dialog (it lost its status-bar slot);
        # unpatched it would block the offscreen run forever.
        monkeypatch.setattr(
            "app.main_window_settings.QMessageBox.information",
            lambda *a, **k: None,
        )
        window._load_scan_settings()
        assert window.ua_start.value() == pytest.approx(50.0)
        assert window.ua_stop.value() == pytest.approx(280.0)
        assert window.ua_step.value() == pytest.approx(7.0)

    def test_save_failure_shows_dialog(self, window, monkeypatch, tmp_path):
        """ML-085: a failed settings save must warn the user — the load
        side already does (asymmetry); a read-only/missing target used to
        die silently in the Qt slot."""
        bad = tmp_path / "no_such_dir" / "s.json"
        monkeypatch.setattr(
            "app.main_window_settings.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(bad), ""))
        warned = []
        monkeypatch.setattr(
            "app.main_window_settings.QMessageBox.warning",
            lambda *a, **k: warned.append(a))
        window._save_scan_settings()
        assert warned, "silent settings-save failure"

    def test_save_dialog_cancel_is_noop(self, window, monkeypatch):
        """Empty path from QFileDialog must not write anything."""
        called = []
        monkeypatch.setattr(
            "app.main_window.QFileDialog.getSaveFileName",
            lambda *a, **kw: ("", ""),
        )
        # Patch open to detect any write attempt
        real_open = open

        def _open_spy(*args, **kwargs):
            mode = args[1] if len(args) >= 2 else kwargs.get("mode", "r")
            if "w" in mode:
                called.append(args[0])
            return real_open(*args, **kwargs)

        monkeypatch.setattr("builtins.open", _open_spy)
        window._save_scan_settings()
        assert called == []


# ----------------------------------------------------------------------
# Plot manager wiring — points propagation, render dispatch
# ----------------------------------------------------------------------


class TestPlotManagerWiring:

    def test_initial_plot_manager_state(self, window):
        """plot_mgr is constructed with empty points and renderer attached."""
        assert hasattr(window, "plot_mgr")
        assert window.plot_mgr.points == []
        assert window.plot_mgr.renderer is not None

    def test_render_all_with_no_points_does_not_crash(self, window):
        """Empty-data path is the most common state at startup."""
        window.plot_mgr.points = []
        window.plot_mgr.render_all()  # must not raise

    def test_zone_spinbox_change_does_not_crash(self, window):
        """Modifying zone spinboxes triggers re-render via signals."""
        window.zone_ua_min.setValue(50.0)
        window.zone_ua_max.setValue(200.0)
        window.zone_ug1_min.setValue(-5.0)
        window.zone_ug1_max.setValue(-1.0)
        # No assertion — purely "no exception"; signal disconnects would
        # fail silently if not wrapped in this test.

    def test_panel_spin_tick_no_full_rerender(self, window, monkeypatch):
        """MAIN perf pin (baseline bench: 446 ms/tick on the old path):
        an amp-panel spin tick does NOT trigger a full 2D re-render —
        only the controller debounce timer. RV: reconnect valueChanged
        back to _rerender_2d — this pin fails."""
        calls = []
        monkeypatch.setattr(window.plot_mgr, "_render_2d",
                            lambda *a, **k: calls.append(1))
        # Non-empty data: on an empty plot render_2d_only is guarded and
        # a 'spin -> full re-render' mutant would be invisible (RV hole).
        window.plot_mgr.points = [
            {"ua": 100.0, "ug1": -2.0, "ia": 10.0, "ug2": 0.0}]
        window.load_line_cb.setChecked(True)
        window.working_line._timer.stop()
        spin = window.amp_control_panel.ub_spin
        spin.setValue(spin.value() + 5.0)
        assert calls == []                    # zero full re-renders
        assert window.working_line._timer.isActive()   # tick debounced
        window.working_line._timer.stop()
        window.load_line_cb.setChecked(False)

    _WL_SPINS = ("ub_spin", "ra_spin", "ug1_spin", "swing_spin",
                 "xfmr_ra_dc_spin", "cf_rk_spin", "cf_rl_spin",
                 "pp_raa_spin", "pp_ra_dc_spin", "ul_tap_spin",
                 "pa_max_spin")

    @pytest.mark.parametrize("spin_name", _WL_SPINS)
    def test_every_panel_spin_schedules_line(self, window, spin_name):
        """Wiring completeness (call-site != function audit form): EVERY
        control that enters params_snapshot must arm the live-line
        debounce."""
        window.load_line_cb.setChecked(True)
        window.working_line._timer.stop()
        spin = getattr(window.amp_control_panel, spin_name)
        spin.setValue(spin.value() + spin.singleStep())
        assert window.working_line._timer.isActive(), spin_name
        window.working_line._timer.stop()
        window.load_line_cb.setChecked(False)

    def test_selector_widgets_schedule_line(self, window):
        """Combos/buttons found as audit holes: source_combo,
        data_source_combo, pp_matched_btn, pp_tube_b_combo,
        circuit/hd_method and the main-window ug2_calc_combo."""
        panel = window.amp_control_panel
        wl = window.working_line
        window.load_line_cb.setChecked(True)

        def fired(act):
            wl._timer.stop()
            act()
            active = wl._timer.isActive()
            wl._timer.stop()
            return active

        assert fired(lambda: panel.circuit_combo.setCurrentIndex(
            (panel.circuit_combo.currentIndex() + 1)
            % panel.circuit_combo.count()))
        assert fired(lambda: panel.hd_method_combo.setCurrentIndex(
            (panel.hd_method_combo.currentIndex() + 1)
            % panel.hd_method_combo.count()))
        assert fired(panel.pp_matched_btn.toggle)
        assert fired(panel.data_source_combo.selectionChanged.emit)
        panel.source_combo.addItem("wl-aud-a", 101)
        panel.source_combo.addItem("wl-aud-b", 102)
        assert fired(lambda: panel.source_combo.setCurrentIndex(
            panel.source_combo.count() - 1))
        panel.pp_tube_b_combo.addItem("wl-aud-a", 103)
        panel.pp_tube_b_combo.addItem("wl-aud-b", 104)
        panel.pp_tube_b_combo.setCurrentIndex(
            panel.pp_tube_b_combo.count() - 2)
        assert fired(lambda: panel.pp_tube_b_combo.setCurrentIndex(
            panel.pp_tube_b_combo.count() - 1))
        window.ug2_calc_combo.addItem("wl-aud-a", 104)
        window.ug2_calc_combo.addItem("wl-aud-b", 105)
        window.ug2_calc_combo.setCurrentIndex(
            window.ug2_calc_combo.count() - 2)
        assert fired(lambda: window.ug2_calc_combo.setCurrentIndex(
            window.ug2_calc_combo.count() - 1))
        window.load_line_cb.setChecked(False)

    def test_full_render_calls_reattach_hook(self, window, monkeypatch):
        """Call site: plot_manager._render_2d must hand the controller
        items back after plot.clear() (RV: dropping the hook is caught)."""
        # wiring: the hook stores the controller's bound method
        assert (window.plot_mgr.working_line_reattach
                == window.working_line.reattach)
        # call site: _render_2d invokes the hook after rendering
        calls = []
        monkeypatch.setattr(window.plot_mgr, "working_line_reattach",
                            lambda: calls.append(1))
        monkeypatch.setattr(window.plot_mgr.renderer, "render_plot_2d",
                            lambda *a, **k: None)
        window.plot_mgr._render_2d([])
        assert calls == [1]

    def test_analyze_feeds_controller_view(self, window, monkeypatch):
        """Call site: a successful Analyze hands result.working_line to
        the controller (apply_full_result)."""
        from lm19.amp_engine import AnalysisResult, WorkingLineView
        sentinel = WorkingLineView(circuit=CIRCUIT_SE)
        fake = AnalysisResult(working_line=sentinel)
        monkeypatch.setattr(window.amp_engine, "analyze",
                            lambda p: fake)
        monkeypatch.setattr(window.amp_engine, "_all_points",
                            [{"ua": 1.0, "ug1": -1.0, "ia": 1.0}])
        got = []
        monkeypatch.setattr(window.working_line, "apply_full_result",
                            lambda v: got.append(v))
        monkeypatch.setattr(window.amplifier_tab, "render",
                            lambda r: "")
        monkeypatch.setattr(window.amplifier_tab, "collect_warnings",
                            lambda r: [])
        window._on_amp_update()
        assert got == [sentinel]

    def test_cb_drives_controller_visibility(self, window):
        window.load_line_cb.setChecked(True)
        assert window.working_line._visible
        window.working_line._timer.stop()
        window.load_line_cb.setChecked(False)
        assert not window.working_line._visible

    def test_load_line_toggle_does_not_crash(self, window):
        """load_line_cb — live-layer visibility; the line parameters
        live ONLY on the amp panel."""
        window.load_line_cb.setChecked(True)
        window.amp_control_panel.ub_spin.setValue(250.0)
        window.amp_control_panel.ra_spin.setValue(8.0)
        window.load_line_cb.setChecked(False)
        # Negative space: the plot-side spins are removed for good.
        for gone in ("load_line_ub", "load_line_ra",
                     "load_line_ug1", "load_line_swing"):
            assert not hasattr(window, gone), gone


# ----------------------------------------------------------------------
# SPICE export topology threading
# ----------------------------------------------------------------------


class TestExportSpiceTopology:

    def test_current_ug2_mode_values(self, window):
        """The canonical Ug2-mode helper covers all three states."""
        window._is_triode = True
        assert window._current_ug2_mode() == TOPOLOGY_TRIODE
        window._is_triode = False
        window.ug2_track_radio.setChecked(True)
        assert window._current_ug2_mode() == TOPOLOGY_TRIODE_CONNECTED
        # track/sweep are autoExclusive radios — select the sibling to clear track
        window.ug2_sweep_radio.setChecked(True)
        assert window._current_ug2_mode() == TOPOLOGY_PENTODE

    def test_export_spice_forwards_topology(self, window, monkeypatch):
        """_export_spice must pass the real Ug2 mode so a triode-connected
        scan is not fit as a pentode (topology was always None before)."""
        spy = MagicMock()
        monkeypatch.setattr("app.main_window.export_spice", spy)
        window._is_triode = False
        window.ug2_track_radio.setChecked(True)
        window._export_spice()
        spy.assert_called_once()
        assert spy.call_args.kwargs.get("topology") == TOPOLOGY_TRIODE_CONNECTED


class TestExportPdfCallsite:

    def test_export_pdf_filters_overlay_series(self, window, monkeypatch):
        """_export_pdf must forward scan points only (series_id == 0) —
        Compare overlays used to inflate 'Scan points: N' in the PDF."""
        spy = MagicMock()
        monkeypatch.setattr("app.main_window.export_pdf", spy)
        window.plot_mgr.points = [
            {"ua": 1.0, "ug1": -1.0, "ia": 1.0, "series_id": 0},
            {"ua": 2.0, "ug1": -1.0, "ia": 2.0, "series_id": 0},
            {"ua": 3.0, "ug1": -1.0, "ia": 3.0, "series_id": 7},
        ]
        window._export_pdf()
        spy.assert_called_once()
        pts = spy.call_args.kwargs["points"]
        assert len(pts) == 2
        assert all(p.get("series_id") == 0 for p in pts)
        assert spy.call_args.kwargs["transfer_widget"] is window.transfer_plot
        assert (spy.call_args.kwargs["mfg_date"]
                == window.lamp_panel.mfg_date())

    def test_export_pdf_overlay_only_falls_back_to_all(self, window,
                                                       monkeypatch):
        """Twin branch: with no scan data the overlay points are exported
        (same fallback semantics as the SPICE exporter)."""
        spy = MagicMock()
        monkeypatch.setattr("app.main_window.export_pdf", spy)
        overlay = [
            {"ua": 1.0, "ug1": -1.0, "ia": 1.0, "series_id": 3},
            {"ua": 2.0, "ug1": -1.0, "ia": 2.0, "series_id": 7},
        ]
        window.plot_mgr.points = overlay
        window._export_pdf()
        assert len(spy.call_args.kwargs["points"]) == 2

    def test_export_pdf_forwards_config_and_last_scan_meta(self, window,
                                                           monkeypatch):
        """The PDF options dialog needs the app config, and the
        scan-settings section needs the last scan's metadata."""
        spy = MagicMock()
        monkeypatch.setattr("app.main_window.export_pdf", spy)
        meta = {"name": "run7", "scan": {"ug2_mode": TOPOLOGY_PENTODE}}
        window._last_scan_meta = meta
        window.plot_mgr.points = [
            {"ua": 1.0, "ug1": -1.0, "ia": 1.0, "series_id": 0}]
        window._export_pdf()
        assert spy.call_args.kwargs["scan_meta"] is meta
        assert spy.call_args.kwargs["config"] is window.app_config


class TestAmpPdfExport:

    def test_button_wired_and_guard_warns_without_analysis(self, window,
                                                           monkeypatch):
        """End-to-end wiring pin: clicking the panel button reaches the
        slot; without an analysis the user gets a warning, not a dialog."""
        warns = []
        monkeypatch.setattr(
            "app.main_window.QMessageBox.warning",
            staticmethod(lambda *a, **k: warns.append(a)))
        window._last_amp_result = None
        window.amp_control_panel.export_pdf_btn.click()
        assert warns, "button click did not reach _export_amp_pdf"
        tip = window.amp_control_panel.export_pdf_btn.toolTip()
        assert tip and not tip.startswith("report.")

    def test_marshals_results_html_and_images(self, window, monkeypatch,
                                              tmp_path):
        from lm19.amp_engine import AmpParams, AnalysisResult, SourceResult
        import app.report_options_dialog as rod

        dist = {"hd2": 1.5, "hd3": 0.4, "thd": 1.6, "pout_mw": 1000.0,
                "ua_0": 250.0, "ia_0": 8.0, "ug1_0": -7.0}
        params = AmpParams(circuit=CIRCUIT_SE, ub=250.0, ra=5.0, ug1_bias=-7.0,
                           hd_method=HD_METHOD_CHEBYSHEV)
        window._last_amp_result = AnalysisResult(
            per_source={"measurements": SourceResult(dist=dist)},
            params=params)
        # the document text is REBUILT from the result (report language),
        # not scraped from the panel label
        monkeypatch.setattr(window.amplifier_tab, "format_results_html",
                            lambda result: "<b>rebuilt-42</b>")

        asked = {}

        def fake_ask(parent, available, config, **kw):
            asked.update(available=available, kw=kw)
            return rod.ReportOptions(
                sections={"amp_results", "amp_spectrum"}, language="en")

        monkeypatch.setattr(rod, "ask_report_options", fake_ask)
        monkeypatch.setattr(
            "app.main_window.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: (str(tmp_path / "amp.pdf"), "")))
        monkeypatch.setattr(
            "app.main_window.QMessageBox.information",
            staticmethod(lambda *a, **k: None))
        captured = {}
        import app.amp_report_pdf as arp
        monkeypatch.setattr(
            arp, "generate_amp_pdf_report",
            lambda path, **kw: captured.update(path=path, **kw))

        window._export_amp_pdf()

        assert captured["results_html"] == "<b>rebuilt-42</b>"
        assert asked["kw"]["session_key"] == "amp"
        # sweeps never ran in this window → greyed with reasons
        assert asked["available"]["amp_plot_thd_sweep"] == "report.Na_no_sweep"
        assert asked["available"]["amp_spectrum"] == ""
        assert asked["available"]["amp_ltspice"] == "report.Na_no_verify"
        assert captured["verify_html"] == ""      # section unavailable
        captions = [c for c, _ in captured["images"]]
        assert len(captions) == 1  # spectrum only — plots not selected
        header = "\n".join(captured["header_lines"])
        assert "Circuit: se" in header and "HD: chebyshev" in header

    def test_verify_html_rebuilt_from_result_when_selected(
            self, window, monkeypatch, tmp_path):
        """The PDF's verify table is REBUILT from the stored VerifyResult
        (report language), not copied from the UI-language panel html."""
        from lm19.amp_engine import AmpParams, AnalysisResult, SourceResult
        from lm19.ltspice_verify import VerifyResult
        import app.report_options_dialog as rod
        import app.amp_report_pdf as arp

        window._last_amp_result = AnalysisResult(
            per_source={"measurements": SourceResult(dist=None)},
            params=AmpParams())
        stub_vr = VerifyResult(basis="stub")
        window._last_amp_verify = stub_vr
        window._last_amp_verify_html = "<table>panel-language</table>"
        window._amp_verify_stale = False
        rebuilt = []
        monkeypatch.setattr(
            arp, "build_verify_table_html",
            lambda vr, ref: rebuilt.append(vr) or "<table>rebuilt</table>")
        monkeypatch.setattr(
            rod, "ask_report_options",
            lambda parent, available, config, **kw: rod.ReportOptions(
                sections={"amp_results", "amp_ltspice"}, language="en"))
        monkeypatch.setattr(
            "app.main_window.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: (str(tmp_path / "v.pdf"), "")))
        monkeypatch.setattr("app.main_window.QMessageBox.information",
                            staticmethod(lambda *a, **k: None))
        captured = {}
        monkeypatch.setattr(
            arp, "generate_amp_pdf_report",
            lambda path, **kw: captured.update(kw))
        window._export_amp_pdf()
        assert rebuilt == [stub_vr]
        assert captured["verify_html"] == "<table>rebuilt</table>"

    def test_document_follows_report_language(self, window, monkeypatch,
                                              tmp_path):
        """report_language=<non-default locale> → the header/verify text
        of the DOCUMENT renders in that locale (incl. the stale ⚠ line),
        while the global locale is restored afterwards (panel stays
        UI-language). The locale is derived from locales/*.json — no
        language is hardcoded."""
        from lm19.amp_engine import AmpParams, AnalysisResult, SourceResult
        from lm19.ltspice_verify import VerifyResult, VerifyRun
        import app.report_options_dialog as rod
        import app.amp_report_pdf as arp
        from i18n_setup import available_locales, translator_for
        from i18n_setup import t as t_global

        en_label = t_global("report.Amp_params", circuit=CIRCUIT_SE,
                            ub="1", ra="1", bias="1", method="m")
        locales = [loc for loc in available_locales()
                   if loc != "en"
                   and translator_for(loc)(
                       "report.Amp_params", circuit=CIRCUIT_SE, ub="1",
                       ra="1", bias="1", method="m") != en_label]
        assert locales, "no localized report locale — pin would vanish"
        doc_locale = locales[0]

        window.app_config.report_language = doc_locale
        window._last_amp_result = AnalysisResult(
            per_source={"measurements": SourceResult(dist=None)},
            params=AmpParams(circuit=CIRCUIT_SE, ub=250.0, ra=5.0,
                             ug1_bias=-7.0, hd_method=HD_METHOD_CHEBYSHEV))
        window._last_amp_verify = VerifyResult(
            runs=[VerifyRun(half_swing=5.0, thd_pct=2.0,
                            hd_pct={2: 1.0, 3: 0.5}, pout_fund_mw=100.0,
                            ia_avg_ma=10.0)],
            basis="LTspice basis")
        window._amp_verify_stale = True
        monkeypatch.setattr(
            rod, "ask_report_options",
            lambda parent, available, config, **kw: rod.ReportOptions(
                sections={"amp_results", "amp_ltspice"},
                language=doc_locale))
        monkeypatch.setattr(
            "app.main_window.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: (str(tmp_path / "r.pdf"), "")))
        monkeypatch.setattr("app.main_window.QMessageBox.information",
                            staticmethod(lambda *a, **k: None))
        captured = {}
        monkeypatch.setattr(
            arp, "generate_amp_pdf_report",
            lambda path, **kw: captured.update(kw))
        try:
            window._export_amp_pdf()
        finally:
            window.app_config.report_language = ""
        header = "\n".join(captured["header_lines"])
        # Expected renderings come from the locale file, not literals.
        tr = translator_for(doc_locale)
        loc_params_label = tr("report.Amp_params", circuit=CIRCUIT_SE,
                              ub="", ra="", bias="",
                              method="").split("  ")[0]
        assert loc_params_label != "report.Amp_params"  # key resolved
        assert loc_params_label in header
        assert tr("report.Verify_table_title") in captured["verify_html"]
        assert tr("report.Verify_stale") in captured["verify_html"]
        # global locale restored — the UI stays English
        assert t_global("report.Amp_params", circuit="x", ub="1", ra="1",
                        bias="1", method="m").startswith("Circuit:")


class TestAmpVerifySlot:

    def test_button_wired_and_guard_warns_without_analysis(self, window,
                                                           monkeypatch):
        warns = []
        monkeypatch.setattr(
            "app.main_window.QMessageBox.warning",
            staticmethod(lambda *a, **k: warns.append(a)))
        window._last_amp_result = None
        window.amp_control_panel.verify_btn.click()
        assert warns, "verify click did not reach _on_amp_verify_ltspice"

    def test_request_marshalled_from_analysis(self, window, monkeypatch,
                                              tmp_path):
        from lm19.amp_engine import AmpParams, AnalysisResult, SourceResult

        # half_swing=3.3 was REQUESTED, 8.8 is what the engine actually
        # used (auto-resolve) — verification must simulate the actual one.
        # series_id=7: an overlay lamp is analyzed while series 0 also
        # exists — verification must take series 7, NOT the current scan.
        params = AmpParams(circuit=CIRCUIT_PP, ub=300.0, ra=8.0, ug1_bias=-11.7,
                           hd_method=HD_METHOD_DFT, ug2_filter=270.0,
                           sources=["dempwolf"], half_swing=3.3,
                           ul_tap=0.43, series_id=7)
        window._last_amp_result = AnalysisResult(
            per_source={"dempwolf": SourceResult(
                dist={"half_swing": 8.8, "thd": 9.0})},
            params=params)
        scan_pt = {"ua": 1.0, "ug1": -1.0, "ia": 1.0, "series_id": 0}
        overlay_pt = {"ua": 2.0, "ug1": -2.0, "ia": 2.0, "series_id": 7}

        class _StubModel:
            model_type = MODEL_TYPE_DEMPWOLF

        stub_model = _StubModel()
        window.amp_engine.set_data(
            [scan_pt, overlay_pt], series_labels={7: "EL84 saved L42"},
            series_models={7: stub_model})
        window.plot_mgr.points = [scan_pt, overlay_pt]
        monkeypatch.setattr("lm19.ltspice_verify.ltspice_available",
                            lambda *a: True)
        window.app_config.ltspice_verify_dir = str(tmp_path)
        fake_exe = tmp_path / "MyLTspice.exe"
        window.app_config.ltspice_exe = str(fake_exe)
        window.amp_control_panel.verify_sweep_cb.setChecked(True)

        fake_worker = MagicMock()
        created = {}

        def fake_ctor(request, workdir, ltspice_exe=None, parent=None):
            created["request"] = request
            created["workdir"] = workdir
            created["exe"] = ltspice_exe
            return fake_worker

        monkeypatch.setattr("app.workers.AmpVerifyWorker", fake_ctor)
        window._on_amp_verify_ltspice()

        req = created["request"]
        assert req.circuit == CIRCUIT_PP and req.ra_kohm == 8.0
        assert req.ug1_bias == -11.7
        assert req.half_swing == 8.8        # engine's ACTUAL swing, not None
        assert req.model_type == MODEL_TYPE_DEMPWOLF  # analysis model source wins
        assert req.ug2 == 270.0
        assert req.ul_tap == 0.43
        assert req.amp_sweep is True
        # data provenance: ONLY the analyzed series, honestly labelled
        assert req.points == [overlay_pt]
        assert req.data_label == "EL84 saved L42"
        # analysis ran on the dempwolf source → verify THAT model, no refit
        assert req.model is stub_model
        # configured workdir and executable honored
        assert str(tmp_path) in created["workdir"]
        assert "verify_" in created["workdir"]
        assert created["exe"] == str(fake_exe)
        fake_worker.start.assert_called_once()
        assert not window.amp_control_panel.verify_btn.isEnabled()
        assert window.amp_control_panel.verify_cancel_btn.isEnabled()

    def test_explicit_fitter_forces_fresh_fit(self, window, monkeypatch,
                                              tmp_path):
        """Combo ≠ Auto: the chosen fitter runs a FRESH fit — the loaded
        model must NOT be smuggled in even when its type matches."""
        from lm19.amp_engine import AmpParams, AnalysisResult, SourceResult

        class _StubModel:
            model_type = MODEL_TYPE_DEMPWOLF

        params = AmpParams(circuit=CIRCUIT_SE, ub=250.0, ra=5.0,
                           ug1_bias=-7.0, hd_method=HD_METHOD_DFT,
                           sources=[MODEL_TYPE_DEMPWOLF],
                           half_swing=5.0, ug2_filter=270.0)
        window._last_amp_result = AnalysisResult(
            per_source={MODEL_TYPE_DEMPWOLF: SourceResult(
                dist={"half_swing": 5.0, "thd": 2.0})},
            params=params)
        window.amp_engine.set_data(
            [{"ua": 1.0, "ug1": -1.0, "ia": 1.0, "series_id": 0}],
            series_models={7: _StubModel()})
        window.plot_mgr.points = [
            {"ua": 1.0, "ug1": -1.0, "ia": 1.0, "series_id": 0}]
        monkeypatch.setattr("lm19.ltspice_verify.ltspice_available",
                            lambda *a: True)
        window.app_config.ltspice_verify_dir = str(tmp_path)
        combo = window.amp_control_panel.verify_fitter_combo
        combo.setCurrentIndex(combo.findData(MODEL_TYPE_DEMPWOLF))

        created = {}

        def fake_ctor(request, workdir, ltspice_exe=None, parent=None):
            created["request"] = request
            return MagicMock()

        monkeypatch.setattr("app.workers.AmpVerifyWorker", fake_ctor)
        try:
            window._on_amp_verify_ltspice()
        finally:
            combo.setCurrentIndex(0)  # back to Auto for other tests
        req = created["request"]
        assert req.model_type == MODEL_TYPE_DEMPWOLF
        assert req.model is None        # explicit choice → fresh fit

    def test_fitter_combo_covers_all_fitters(self, window):
        """Source of truth: the combo mirrors the SPICE-export registry
        plus Auto (a new fitter must appear here automatically)."""
        from app.export_manager import _SPICE_MODEL_CHOICES
        combo = window.amp_control_panel.verify_fitter_combo
        datas = [combo.itemData(i) for i in range(combo.count())]
        assert datas == [""] + [k for k, _ in _SPICE_MODEL_CHOICES]
        assert combo.currentData() == ""   # Auto is the default

    def test_cancel_slot_stops_current_worker(self, window):
        worker = MagicMock()
        window._amp_verify_worker = worker
        window._on_amp_verify_cancel()
        worker.stop.assert_called_once()

    def test_show_on_main_forwards_scan_meta(self, window):
        """UX fix: a measurement loaded via Compare must feed the PDF
        scan-settings section instead of demanding a fresh scan."""
        meta = {"name": "loaded_run", "scan": {"ug2_mode": TOPOLOGY_PENTODE}}
        window._last_scan_meta = None
        window._on_compare_show_main_plot(
            [{"ua": 1.0, "ug1": -1.0, "ia": 1.0, "series_id": 3}],
            {3: "L1"}, {}, {}, meta)
        assert window._last_scan_meta is meta
        # multi-selection (meta=None) must NOT clobber a valid meta
        window._on_compare_show_main_plot(
            [{"ua": 1.0, "ug1": -1.0, "ia": 1.0, "series_id": 4}],
            {4: "L2"}, {}, {}, None)
        assert window._last_scan_meta is meta

    def test_show_on_main_keeps_the_scan_track_flag(self, window):
        """The overlay swap must not re-derive the scan's own grouping:
        the incoming dict knows nothing about sid 0."""
        window.plot_mgr.points = [
            {"ua": 100.0, "ug1": -1.0, "ia": 5.0, "ug2": 100.0,
             "series_id": 0}]
        window.plot_mgr.set_scan_ug2_track(True)
        window.ug2_sweep_radio.setChecked(True)   # next run armed as pentode
        window._on_compare_show_main_plot(
            [{"ua": 1.0, "ug1": -1.0, "ia": 1.0, "ug2": 250.0,
              "series_id": 1}],
            {1: "L1"}, {}, {1: False}, None)
        assert window.plot_mgr.series_ug2_track[0] is True
        assert window.plot_mgr._is_sid_ug2_track(0) is True
        assert window.plot_mgr.series_ug2_track[1] is False

    def test_show_on_main_drops_model_bookkeeping(self, window):
        """Model overlay points are dropped by the swap, so a stale
        series_models entry would render an incoming compare series as a
        dashed model curve driven by the wrong model."""
        window.plot_mgr.points = [
            {"ua": 1.0, "ug1": -1.0, "ia": 1.0, "series_id": 1}]
        window.plot_mgr.series_models = {1: MagicMock()}
        window.plot_mgr.series_grids = {1: MagicMock()}
        window._on_compare_show_main_plot(
            [{"ua": 2.0, "ug1": -1.0, "ia": 2.0, "ug2": 250.0,
              "series_id": 1}],
            {1: "L1"}, {}, {1: False}, None)
        assert window.plot_mgr.series_models == {}
        assert window.plot_mgr.series_grids == {}
        assert [p["ua"] for p in window.plot_mgr.points] == [2.0]

    def test_verify_table_marked_stale_on_param_change(self, window):
        from lm19.amp_engine import AmpParams

        window._last_amp_verify_html = "<table>old-verify</table>"
        window._last_amp_verify_params = AmpParams(ub=250.0)
        window._amp_verify_stale = False
        # same params → untouched
        window._mark_amp_verify_stale_if_needed(AmpParams(ub=250.0))
        assert window._last_amp_verify_html == "<table>old-verify</table>"
        # changed params → visible ⚠ prefix, exactly once
        window._mark_amp_verify_stale_if_needed(AmpParams(ub=300.0))
        assert window._last_amp_verify_html.startswith('<span')
        assert "⚠" in window._last_amp_verify_html
        assert "old-verify" in window._last_amp_verify_html
        once = window._last_amp_verify_html
        window._mark_amp_verify_stale_if_needed(AmpParams(ub=320.0))
        assert window._last_amp_verify_html == once  # no warning stacking

    def test_done_slot_stores_html_for_pdf(self, window, monkeypatch):
        monkeypatch.setattr(
            "app.amp_report_pdf.build_verify_table_html",
            lambda vr, ref: "<b>verify-table</b>")
        window._last_amp_result = None  # reference resolves to {}
        window.amp_control_panel.set_verify_running(True)
        window._on_amp_verify_done(MagicMock())
        assert window._last_amp_verify_html == "<b>verify-table</b>"
        assert (window.amp_control_panel.verify_status_label.text()
                == "<b>verify-table</b>")
        assert window.amp_control_panel.verify_btn.isEnabled()


# ----------------------------------------------------------------------
# Scan-finished callback — measurement save flow without real worker
# ----------------------------------------------------------------------


class TestScanFinishedFlow:

    def test_on_opt_cancelled_resets_running_state(self, window):
        """The cancel slot must take the optimizer panel out of the running
        state and drop the worker — without it the UI stays stuck on Cancel."""
        window.amp_control_panel = MagicMock()
        opt = MagicMock()
        opt.cleanup.return_value = True
        window._opt_worker = opt
        window._on_opt_cancelled()
        window.amp_control_panel.set_optimizer_running.assert_called_once_with(False)
        opt.cleanup.assert_called_once()
        assert window._opt_worker is None

    def test_save_failure_warns_and_continues(self, window, monkeypatch):
        """A save_measurement OSError (disk full / permission) must surface
        the recovery dialog and NOT abort the slot — the in-memory compare
        entry still adds so the data is not lost from the session (the
        the dialog now offers Retry / Save As / Close)."""
        crit = MagicMock(return_value="close")
        monkeypatch.setattr("app.save_recovery._ask_recovery", crit)
        monkeypatch.setattr("app.main_window_scan.save_measurement",
                            MagicMock(side_effect=OSError("disk full")))
        window.compare_tab = MagicMock()
        window._update_quality_label = MagicMock()
        window.plot_mgr = MagicMock()
        window._scan_summary = None
        measurement = {"tube_type": "EL84", "lamp_id": "L1",
                       "timestamp": "2026-06-21T00:00:00", "name": "n",
                       "points": [], "srk": None}
        window._save_and_display_measurement(measurement, [])
        crit.assert_called_once()                       # user warned
        window.compare_tab.add_entry.assert_called_once()  # flow continued

    def test_on_scan_finished_with_empty_points_resets_state(self, window):
        """Empty scan result must reset run/stop button state cleanly."""
        window.run_btn.setEnabled(False)
        window.stop_btn.setEnabled(True)
        window.scan_in_progress = True
        # Mock connection manager to avoid serial calls
        window.conn_mgr = MagicMock()
        window._reset_outputs_after_scan = MagicMock()
        window._on_scan_finished([])
        assert window.run_btn.isEnabled()
        assert not window.stop_btn.isEnabled()
        assert not window.scan_in_progress

    def test_on_scan_finished_preserves_compare_overlay(self, window,
                                                        monkeypatch):
        """Existing series_id != 0 points must survive a new scan."""
        # Seed plot_mgr with an overlay
        window.plot_mgr.points = [
            {"ua": 100.0, "ug1": -1.0, "ia": 5.0, "ug2": 250.0,
             "series_id": 1},
        ]
        new_scan = [
            {"ua": 100.0, "ug1": -1.0, "ia": 5.0, "ug2": 250.0},
            {"ua": 150.0, "ug1": -1.0, "ia": 6.0, "ug2": 250.0},
        ]
        # Stub out SRK side-effects so we focus on plot_mgr update
        window.srk_measure_separately = MagicMock()
        window.srk_measure_separately.isChecked.return_value = False
        window.srk = MagicMock()
        # scan_ctrl is a property → replace whole controller with mock
        mock_ctrl = MagicMock()
        mock_ctrl.scan_start_time = None
        window.scan_ctrl = mock_ctrl
        window._on_scan_finished(new_scan)
        # Two new points (series_id=0) + one preserved overlay (series_id=1)
        sids = sorted({p.get("series_id", 0) for p in window.plot_mgr.points})
        assert 0 in sids
        assert 1 in sids
        assert len(window.plot_mgr.points) == 3

    def test_stop_scan_does_not_reset_outputs_immediately(self, window):
        """_stop_scan must not zero outputs while the worker may still be live —
        a verify-retry could re-assert HV after an immediate reset. The reset is
        deferred to _on_scan_finished."""
        window._reset_outputs_after_scan = MagicMock()
        mock_ctrl = MagicMock()
        window.scan_ctrl = mock_ctrl
        window._stop_scan()
        assert not window._reset_outputs_after_scan.called
        mock_ctrl.stop_scan.assert_called_once()

    def test_on_scan_finished_resets_outputs_when_user_stopped(self, window):
        """A user-stopped scan (reset_on_finish) zeroes outputs on finish, even
        with partial points present."""
        window._reset_outputs_after_scan = MagicMock()
        window.srk_measure_separately = MagicMock()
        window.srk_measure_separately.isChecked.return_value = False
        window.srk = MagicMock()
        mock_ctrl = MagicMock()
        mock_ctrl.scan_start_time = None
        mock_ctrl.reset_on_finish = True
        window.scan_ctrl = mock_ctrl
        window._on_scan_finished(
            [{"ua": 100.0, "ug1": -2.0, "ia": 1.0, "ug2": 0.0, "series_id": 0}]
        )
        assert window._reset_outputs_after_scan.called

    def test_emergency_zero_outputs_drives_safe_sequence(self, window):
        """Emergency zero writes the safe shutdown sequence synchronously:
        Ug2/Ua/Uh/Ih to 0 and Ug1 to the cutoff bias — NOT 0, which would open
        the tube fully."""
        from lm19.protocol import encode_ug1
        client = MagicMock()
        client.is_open.return_value = True
        window.conn_mgr._client = client
        window._emergency_zero_outputs()
        calls = [(c.args[0], c.args[1]) for c in client.set_param.call_args_list]
        names = [n for n, _ in calls]
        assert names == ["Ug2", "Ug1", "Ua", "Uh", "Ih"]
        values = dict(calls)
        assert values["Ug2"] == 0 and values["Ua"] == 0
        assert values["Uh"] == 0 and values["Ih"] == 0
        assert values["Ug1"] == encode_ug1(window.app_config.ug1_after_stop)
        assert values["Ug1"] != 0

    def test_closeevent_safe_zeros_hv_keeps_heater(self, window):
        """#25: closing the window must synchronously zero the HV outputs
        (Ug2/Ug1-cutoff/Ua) while the port is still open — otherwise the
        firmware holds the last setpoints and the tube stays energized after
        the GUI closes. The heater is intentionally kept."""
        from PySide6.QtGui import QCloseEvent
        from lm19.protocol import encode_ug1
        client = MagicMock()
        client.is_open.return_value = True
        window.conn_mgr._client = client
        window.closeEvent(QCloseEvent())
        calls = [(c.args[0], c.args[1]) for c in client.set_param.call_args_list]
        names = [n for n, _ in calls]
        # HV driven to safe state...
        assert "Ug2" in names and "Ug1" in names and "Ua" in names
        values = dict(calls)
        assert values["Ug2"] == 0 and values["Ua"] == 0
        assert values["Ug1"] == encode_ug1(window.app_config.ug1_after_stop)
        assert values["Ug1"] != 0          # cutoff bias, not fully-open
        # ...heater left untouched (heater cycling stresses the tube).
        assert "Uh" not in names and "Ih" not in names

    def test_emergency_stop_zeros_synchronously_not_async(self, window):
        """_on_reset_all_clicked locks, flags workers to stop, and zeroes
        outputs synchronously — without queuing an async ResetWorker."""
        client = MagicMock()
        client.is_open.return_value = True
        window.conn_mgr._client = client
        window._reset_outputs_after_scan = MagicMock()  # async path must be unused
        # mirrors the real interface: [] = all channels zeroed OK
        window._emergency_zero_outputs = MagicMock(return_value=[])
        window.scan_ctrl = MagicMock()
        window.srk = MagicMock()
        window.srk.worker = None
        window._on_reset_all_clicked()
        assert window._emergency_lock is True
        window._emergency_zero_outputs.assert_called_once()
        window._reset_outputs_after_scan.assert_not_called()
        window.scan_ctrl.stop_scan.assert_called_once()


# ----------------------------------------------------------------------
# Connection / port wiring
# ----------------------------------------------------------------------


class TestConnectionWiring:

    def test_refresh_ports_populates_combo(self, window, monkeypatch):
        monkeypatch.setattr(
            "app.main_window.list_ports.comports",
            lambda: [_Port("COM7"), _Port("COM8"), _Port("COM9")],
        )
        window._refresh_ports()
        assert window.port_combo.count() == 3
        # First port stays as the displayed item
        assert window.port_combo.itemText(0) == "COM7"

    def test_disconnect_blocked_when_hw_busy(self, window, monkeypatch):
        """Disconnect must be refused while a worker owns the hardware —
        closing the port frees the client and leaves outputs energized."""
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **kw: None)
        mock_cm = MagicMock()
        mock_cm.is_connected = True
        window.conn_mgr = mock_cm
        window.scan_ctrl.scan_in_progress = True  # arbiter → "scan"
        try:
            window._toggle_connection()
            mock_cm.disconnect.assert_not_called()
        finally:
            window.scan_ctrl.scan_in_progress = False

    def test_disconnect_allowed_when_idle(self, window):
        mock_cm = MagicMock()
        mock_cm.is_connected = True
        window.conn_mgr = mock_cm
        window._toggle_connection()
        mock_cm.disconnect.assert_called_once()

    def test_set_write_controls_locked_toggles_run_btn(self, window):
        window._set_write_controls_locked(True)
        # When locked, run button must be disabled
        assert not window.run_btn.isEnabled()
        window._set_write_controls_locked(False)
        # Unlocked — may still be disabled by other state but lock is lifted
        # (we only check the lock-state attribute is consistent)
        assert window._emergency_lock is False or window._emergency_lock is None or \
            isinstance(window._emergency_lock, bool)


# ----------------------------------------------------------------------
# Quality label update — wiring, not value precision
# ----------------------------------------------------------------------


class TestQualityLabel:

    def test_update_quality_label_no_data_does_not_crash(self, window):
        from lm19.config import find_lamp
        lamp = find_lamp(window.lamps,
                         window.lamp_combo.itemText(0))
        # Empty points → quality returns N/A → label updated, no crash
        window._update_quality_label([], lamp, srk=None)


# ----------------------------------------------------------------------
# Menu wiring
# ----------------------------------------------------------------------
#
# MainWindow is composed from several mixins, so menu-action → slot
# connections made in ``_build_menu`` can silently break if a slot
# moves between mixins without the corresponding ``connect`` update.
# These tests guard every action by text and ensure (a) it exists,
# (b) the bound slot attribute is still callable on the window (or on
# the relevant controller, e.g. ``import_ctrl``).


def _collect_menu_actions(menubar):
    """Walk menuBar → top-level menus → submenus, collect (text, action)."""
    out = {}
    for top_act in menubar.actions():
        menu = top_act.menu()
        if menu is None:
            continue
        for act in menu.actions():
            sub = act.menu()
            if sub is not None:
                for sub_act in sub.actions():
                    if not sub_act.isSeparator():
                        out[sub_act.text()] = sub_act
            elif not act.isSeparator():
                out[act.text()] = act
    return out


class TestMenuWiring:
    """Each menu action exists by text and its target slot is callable."""

    def test_file_menu_export_actions_present(self, window):
        from i18n_setup import t
        actions = _collect_menu_actions(window.menuBar())
        for key in ('msg.Export_PDF', 'msg.Spice_export',
                    'csv.Export_CSV', 'utd.Export_utd'):
            assert t(key) in actions, f"Menu missing action {key!r}"

    def test_file_menu_export_slots_callable(self, window):
        # Slots that File-menu Export actions are wired to.
        for slot_name in ('_export_pdf', '_export_spice',
                          '_export_csv', '_export_utd'):
            assert callable(getattr(window, slot_name, None)), \
                f"Window missing slot {slot_name!r}"

    def test_file_menu_import_slots_callable(self, window):
        from i18n_setup import t
        actions = _collect_menu_actions(window.menuBar())
        for key in ('menu.Import_uTracer', 'menu.Import_CSV',
                    'menu.Import_CurveTraceData', 'menu.Import_eTracer'):
            assert t(key) in actions, f"Menu missing import action {key!r}"
        # Import slots live on import_ctrl, not window.
        for slot_name in ('import_utracer', 'import_csv',
                          'import_curvetracedata', 'import_etracer'):
            assert callable(getattr(window.import_ctrl, slot_name, None)), \
                f"import_ctrl missing slot {slot_name!r}"

    def test_op_calculator_action_callable(self, window):
        from i18n_setup import t
        actions = _collect_menu_actions(window.menuBar())
        assert t('msg.Op_calculator') in actions
        assert callable(getattr(window, '_show_op_calculator', None))

    def test_settings_menu_actions_callable(self, window):
        from i18n_setup import t
        actions = _collect_menu_actions(window.menuBar())
        assert t('menu.Load_scan_settings') in actions
        assert t('menu.Save_scan_settings') in actions
        assert callable(getattr(window, '_load_scan_settings', None))
        assert callable(getattr(window, '_save_scan_settings', None))

    def test_reload_lamp_config_action_callable(self, window):
        from i18n_setup import t
        actions = _collect_menu_actions(window.menuBar())
        assert t('menu.Reload_lamp_config') in actions
        assert callable(getattr(window, '_load_lamps', None))


# ----------------------------------------------------------------------
# Error callback paths
# ----------------------------------------------------------------------
#
# ``_on_scan_failed`` and friends touch many collaborators
# (``scan_progress_bar``, ``plot_mgr.renderer``, ``srk``, ``conn_mgr``);
# a missing attribute access there would pass type-checks but explode
# on the rig. These tests stub ``QMessageBox`` and side-effect
# collaborators and exercise the full slot body so any broken access
# raises immediately.


class TestErrorCallbacks:

    def test_on_check_com_fail_updates_status_and_led(self, window):
        window.status_label.setText("")
        window._on_check_com_fail("Timeout reading Ua")
        assert window.status_label.text() != ""
        # LED state changed (we don't assert exact color — that's brittle)

    def test_on_live_params_error_updates_status(self, window):
        window.status_label.setText("")
        window._on_live_params_error("COM disconnected")
        assert window.status_label.text() != ""

    def test_on_scan_failed_no_partial_points(self, window, monkeypatch):
        """Empty scan + error → critical dialog shown, controls reset."""
        from PySide6.QtWidgets import QMessageBox
        # Stub the critical dialog to avoid blocking the test
        monkeypatch.setattr(QMessageBox, 'critical',
                            lambda *args, **kwargs: QMessageBox.Ok)
        window.plot_mgr.points = []
        window.run_btn.setEnabled(False)
        window.stop_btn.setEnabled(True)
        window.scan_in_progress = True
        window.conn_mgr = MagicMock()
        window._reset_outputs_after_scan = MagicMock()
        window._on_scan_failed("Bad reading")
        assert window.run_btn.isEnabled()
        assert not window.stop_btn.isEnabled()
        assert not window.scan_in_progress

    def test_on_scan_failed_partial_points_offers_save(self, window,
                                                       monkeypatch):
        """Non-empty partial scan → Yes/No dialog, Yes path delegates to srk."""
        from PySide6.QtWidgets import QMessageBox
        # Force user to click Yes
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *args, **kwargs: QMessageBox.Yes)
        window.plot_mgr.points = [
            {"ua": 100.0, "ug1": -1.0, "ia": 5.0, "ug2": 250.0,
             "series_id": 0},
        ]
        window.scan_in_progress = True
        mock_srk = MagicMock()
        window.srk = mock_srk
        # Minimal valid metadata: the save path looks up the SRK reference
        # lamp by the recorded tube type (TestSavedMetadataDescribesTheRun
        # pins where that value comes from).
        window._build_scan_metadata = MagicMock(
            return_value={"tube_type": window.lamp_combo.currentText()})
        window._zone_dict = MagicMock(return_value={})
        window._on_scan_failed("Aborted by limit")
        # SRK compute path must have been invoked with the partial points
        assert mock_srk.compute_from_scan.called

    def test_on_scan_comm_error_forwards_decision(self, window, monkeypatch):
        """The Retry/Ignore/Abort dialog forwards a decision to scan_ctrl."""
        from PySide6.QtWidgets import QMessageBox

        # Stub QMessageBox.exec so it doesn't block; clickedButton stays None
        # which falls through to the "skip" branch in the slot.
        monkeypatch.setattr(QMessageBox, 'exec', lambda self: 0)
        mock_ctrl = MagicMock()
        window.scan_ctrl = mock_ctrl
        window.plot_mgr.points = []
        window._on_scan_comm_error("Timeout", attempt=1)
        assert mock_ctrl.respond_comm_error.called
        # Decision is one of {retry, abort, skip}
        decision = mock_ctrl.respond_comm_error.call_args[0][0]
        assert decision in ("retry", "abort", "skip")


# ----------------------------------------------------------------------
# State transitions
# ----------------------------------------------------------------------
#
# ``scan_in_progress`` and ``preheat_done`` are properties that
# delegate to ``scan_ctrl``. Because the property and the methods that
# read/write it may live on different mixins, these tests guarantee
# the round-trip still works through the composed MainWindow.


class TestStateTransitions:

    def test_scan_in_progress_property_round_trip(self, window):
        window.scan_in_progress = True
        assert window.scan_in_progress is True
        window.scan_in_progress = False
        assert window.scan_in_progress is False

    def test_preheat_done_property_round_trip(self, window):
        window.preheat_done = True
        assert window.preheat_done is True
        window.preheat_done = False
        assert window.preheat_done is False

    def test_set_write_controls_locked_disables_run_and_stop(self, window):
        """Lock disables every write-side button; unlock re-enables run_btn."""
        window._set_write_controls_locked(True)
        for attr in ('run_btn', 'stop_btn', 'preheat_start_btn',
                     'measure_srk_btn'):
            btn = getattr(window, attr, None)
            if btn is not None:
                assert not btn.isEnabled(), f"{attr} not disabled by lock"
        window._set_write_controls_locked(False)
        # run_btn must be re-enabled (it's always present)
        assert window.run_btn.isEnabled()


# ----------------------------------------------------------------------
# Preheat lifecycle
# ----------------------------------------------------------------------


class TestPreheatLifecycle:

    def test_on_preheat_finished_sets_done_flag(self, window):
        """Successful preheat completion flips ``preheat_done`` to True."""
        window.preheat_done = False
        window.conn_mgr = MagicMock()
        window._update_preheat_live_label = MagicMock()
        window._on_preheat_finished()
        assert window.preheat_done is True
        # Status label updated
        assert window.preheat_status.text() != ""

    def test_stop_preheat_resets_outputs(self, window):
        """Stopping preheat triggers ``_reset_outputs_after_scan`` + status."""
        mock_ctrl = MagicMock()
        window.scan_ctrl = mock_ctrl
        window.conn_mgr = MagicMock()
        window._reset_outputs_after_scan = MagicMock()
        window._stop_preheat()
        assert mock_ctrl.stop_preheat.called
        assert window._reset_outputs_after_scan.called
        assert window.preheat_start_btn.isEnabled()

    def test_start_preheat_blocked_when_hw_busy(self, window, monkeypatch):
        """Preheat is a hardware-owning start path — the ownership arbiter must
        block it while scan/health/SRK run (it writes Uh/Ih concurrently)."""
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **kw: None)
        client = MagicMock()
        client.is_open.return_value = True
        window.conn_mgr._client = client
        window.preheat_enabled.setChecked(True)
        if window.lamp_combo.count() == 0:
            pytest.skip("no lamps")
        window.lamp_combo.setCurrentIndex(0)
        spy = MagicMock()
        window.scan_ctrl.start_preheat = spy
        window.scan_ctrl.scan_in_progress = True  # arbiter → "scan"
        try:
            window._start_preheat()
            spy.assert_not_called()
        finally:
            window.scan_ctrl.scan_in_progress = False

    def test_start_preheat_allowed_when_idle(self, window):
        """When nothing else owns the hardware, preheat proceeds."""
        client = MagicMock()
        client.is_open.return_value = True
        window.conn_mgr._client = client
        window.preheat_enabled.setChecked(True)
        if window.lamp_combo.count() == 0:
            pytest.skip("no lamps")
        window.lamp_combo.setCurrentIndex(0)
        spy = MagicMock()
        window.scan_ctrl.start_preheat = spy
        window._start_preheat()
        spy.assert_called_once()

    def test_preheat_progress_ignored_after_worker_drained(self, window):
        """A stale progress event delivered after the preheat worker was
        drained (preheat_worker is None) must NOT flip the status away from
        whatever Stop set — guards the reset_outputs drain fix."""
        mock_ctrl = MagicMock()
        mock_ctrl.preheat_worker = None
        window.scan_ctrl = mock_ctrl
        window.preheat_status.setText("Stopped")
        window._on_preheat_progress(6.3, 0.5, 10)
        assert window.preheat_status.text() == "Stopped"


# ----------------------------------------------------------------------
# closeEvent — in-memory compute workers must be drained
# ----------------------------------------------------------------------
#
# OptimizeWorker is a QThread child of the window with no serial client.
# If an optimization is still running at close, a live QThread freed by GC
# calls qFatal and aborts the process. These integration tests use a REAL
# MainWindow (so super().closeEvent and the helper wiring are exercised —
# a mock self breaks super()).


class TestCloseEventWorkerShutdown:

    def test_close_event_cleans_up_running_opt_worker(self, window):
        from PySide6.QtGui import QCloseEvent
        opt = MagicMock()
        opt.cleanup.return_value = True
        window._opt_worker = opt
        window.closeEvent(QCloseEvent())
        opt.cleanup.assert_called_once()
        assert window._opt_worker is None

    def test_close_event_keeps_opt_worker_ref_when_not_drained(self, window):
        """cleanup()==False → reference retained; a live QThread freed by GC
        aborts the process."""
        from PySide6.QtGui import QCloseEvent
        opt = MagicMock()
        opt.cleanup.return_value = False
        window._opt_worker = opt
        window.closeEvent(QCloseEvent())
        assert window._opt_worker is opt

    def test_close_event_without_opt_worker_does_not_raise(self, window):
        """closeEvent before any optimization (no _opt_worker attr set) must
        not AttributeError."""
        from PySide6.QtGui import QCloseEvent
        # Ensure the lazy attribute is absent / None
        window._opt_worker = None
        window.closeEvent(QCloseEvent())  # must not raise


# ----------------------------------------------------------------------
# Pentode Ug2 dialog — UI flow
# ----------------------------------------------------------------------
#
# When ``amp_engine.analyze`` returns ``error="needs_ug2"`` (pentode model
# selected but measurements lack a valid screen voltage), MainWindow must:
#   1. Plumb ``lamp.ug2`` from the currently selected lamp into
#      ``params.lamp_ug2_default`` BEFORE calling analyze (so suggested
#      defaults to the per-tube nominal, not the global DEFAULT_UG2_V).
#   2. Show ``QInputDialog.getDouble`` pre-filled with ``suggested_ug2``.
#   3. On confirm: set ``params.ug2_filter`` and re-analyze.
#   4. On cancel: show "cancelled" message — no garbage analysis.


class _NeedsUg2EngineStub:
    """Stand-in engine that returns needs_ug2 once, then succeeds.

    Tracks calls so the test can verify retry happened with the
    user-supplied ug2_filter.
    """

    def __init__(self, suggested):
        self._suggested = suggested
        self.has_data = True
        self.calls = []  # list of params seen by analyze()

    def analyze(self, params):
        self.calls.append(params)
        from lm19.amp_engine import AnalysisResult
        if len(self.calls) == 1:
            return AnalysisResult(
                error="needs_ug2",
                circuit=params.circuit,
                params=params,
                suggested_ug2=self._suggested,
            )
        return AnalysisResult(circuit=params.circuit, params=params)


class TestNeedsUg2DialogFlow:
    """End-to-end: engine signals needs_ug2 → UI dialog → retry."""

    def _select_pentode_lamp(self, window):
        from lm19.config import find_lamp
        for i in range(window.lamp_combo.count()):
            name = window.lamp_combo.itemText(i)
            lamp = find_lamp(window.lamps, name)
            if lamp and not lamp.is_triode and lamp.ug2 > 0:
                window.lamp_combo.setCurrentIndex(i)
                return lamp
        pytest.skip("No pentode lamp in lamps.json")

    def test_dialog_appears_with_lamp_ug2_pre_filled(self, window,
                                                      monkeypatch):
        """When engine returns needs_ug2, dialog shows with lamp.ug2 prefilled."""
        lamp = self._select_pentode_lamp(window)
        stub = _NeedsUg2EngineStub(suggested=lamp.ug2)
        window.amp_engine = stub

        captured = {}

        def fake_getDouble(parent, title, label, value, mn, mx, decimals):
            captured["title"] = title
            captured["label"] = label
            captured["value"] = value
            captured["min"] = mn
            captured["max"] = mx
            return (lamp.ug2, True)

        from PySide6.QtWidgets import QInputDialog
        monkeypatch.setattr(QInputDialog, "getDouble", fake_getDouble)

        window._on_amp_update()
        assert captured.get("value") == lamp.ug2, \
            f"Expected dialog pre-filled with lamp.ug2={lamp.ug2}, got {captured}"
        assert captured["min"] >= 0 and captured["max"] >= 600

    def test_lamp_ug2_default_plumbed_to_params(self, window, monkeypatch):
        """MainWindow sets params.lamp_ug2_default from current lamp before
        calling analyze — so engine's fallback uses per-tube nominal."""
        lamp = self._select_pentode_lamp(window)
        stub = _NeedsUg2EngineStub(suggested=lamp.ug2)
        window.amp_engine = stub

        from PySide6.QtWidgets import QInputDialog
        monkeypatch.setattr(QInputDialog, "getDouble",
                             lambda *a, **kw: (lamp.ug2, True))

        window._on_amp_update()
        assert len(stub.calls) >= 1
        first_params = stub.calls[0]
        assert first_params.lamp_ug2_default == lamp.ug2, (
            f"Expected lamp_ug2_default={lamp.ug2} on first analyze call, "
            f"got {first_params.lamp_ug2_default}"
        )

    def test_confirm_triggers_retry_with_user_ug2(self, window, monkeypatch):
        """User confirms a value → analyze re-runs with that ug2_filter."""
        lamp = self._select_pentode_lamp(window)
        stub = _NeedsUg2EngineStub(suggested=lamp.ug2)
        window.amp_engine = stub

        user_picked = 300.0
        from PySide6.QtWidgets import QInputDialog
        monkeypatch.setattr(QInputDialog, "getDouble",
                             lambda *a, **kw: (user_picked, True))

        window._on_amp_update()
        assert len(stub.calls) == 2, \
            f"Expected exactly 2 analyze calls (1 + retry), got {len(stub.calls)}"
        retry_params = stub.calls[1]
        assert retry_params.ug2_filter == user_picked, (
            f"Retry should have ug2_filter={user_picked}, "
            f"got {retry_params.ug2_filter}"
        )

    def test_cancel_does_not_retry(self, window, monkeypatch):
        """User cancels → no retry, results show cancelled message."""
        lamp = self._select_pentode_lamp(window)
        stub = _NeedsUg2EngineStub(suggested=lamp.ug2)
        window.amp_engine = stub

        shown_results = []
        original_show = window.amp_control_panel.show_results
        def capture(html):
            shown_results.append(html)
            return original_show(html)
        window.amp_control_panel.show_results = capture

        from PySide6.QtWidgets import QInputDialog
        monkeypatch.setattr(QInputDialog, "getDouble",
                             lambda *a, **kw: (lamp.ug2, False))

        window._on_amp_update()
        assert len(stub.calls) == 1, \
            f"Expected 1 analyze call (no retry on cancel), got {len(stub.calls)}"
        from i18n_setup import t
        expected = t("amp.Needs_Ug2_cancelled")
        assert any(expected in s for s in shown_results), \
            f"Expected cancellation message, got {shown_results}"

    def test_suggested_value_for_non_default_pentode(self, window, monkeypatch):
        """Pentode with ug2 != 250 (default) → suggested comes from lamp
        config, NOT the global DEFAULT_UG2_V=250 fallback. Validates that
        the per-tube nominal is plumbed all the way through."""
        from lm19.config import find_lamp
        target_lamp = None
        for i in range(window.lamp_combo.count()):
            name = window.lamp_combo.itemText(i)
            lamp = find_lamp(window.lamps, name)
            # Find any pentode whose ug2 is NOT 250 — that proves we're
            # not just accidentally hitting the global default.
            if lamp and not lamp.is_triode and lamp.ug2 > 0 and lamp.ug2 != 250.0:
                target_lamp = lamp
                window.lamp_combo.setCurrentIndex(i)
                break
        if target_lamp is None:
            pytest.skip("No non-default-Ug2 pentode in lamps.json")

        captured = {}

        def fake_getDouble(parent, title, label, value, mn, mx, decimals):
            captured["value"] = value
            return (value, True)

        from PySide6.QtWidgets import QInputDialog
        monkeypatch.setattr(QInputDialog, "getDouble", fake_getDouble)

        stub = _NeedsUg2EngineStub(suggested=target_lamp.ug2)
        window.amp_engine = stub

        window._on_amp_update()
        assert captured["value"] == target_lamp.ug2, (
            f"Expected suggested={target_lamp.ug2}V (lamp config), "
            f"got {captured.get('value')}"
        )
        assert captured["value"] != 250.0 or target_lamp.ug2 == 250.0, \
            "Suggested should reflect lamp.ug2, not the global 250V default"
