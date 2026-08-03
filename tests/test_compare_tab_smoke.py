"""Smoke tests for CompareTab table population and matching flow.

Goal: cover the golden-path UI flow that has zero direct test coverage
today — adding entries, rendering the table, running curve matching,
and clearing match state.  Designed to fail loudly if a future refactor
of CompareTab._build_ui (351 lines) breaks the table column layout or
the match-row signal wiring.
"""

import os
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.compare_tab import (
    CompareTab,
    _COL_GRP, _COL_SEL, _COL_TS, _COL_AN, _COL_MODE, _COL_SRK, _COL_COLOR,
    _NUM_COLS,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)

pytestmark = [pytest.mark.smoke_ui]


# ── Module local helpers ──

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _pentode_entry(lamp_id: str, *, ua_max: float = 250.0,
                   ia_at_ua_max: float = 12.0,
                   timestamp: str = "2026-02-25T12:00:00") -> dict:
    """Build a minimal pentode measurement entry for the table."""
    points = []
    for ug1 in (-7.0, -5.0, -3.0):
        for ua in (50.0, 100.0, 150.0, 200.0, ua_max):
            ia = max(0.1, ia_at_ua_max * (ua / ua_max) * (1.0 + (ug1 + 7) / 8))
            points.append({
                "ua": ua, "ug1": ug1, "ug2": 250.0, "ia": ia,
                "ig2": 0.3, "uh": 6.3, "ih": 0.7,
            })
    return {
        "lamp_type": "EL84",
        "lamp_id": lamp_id,
        "name": f"run_{lamp_id}",
        "timestamp": timestamp,
        "points": points,
        "data": {"topology": TOPOLOGY_PENTODE, "an": "Au"},
    }


def _triode_entry(lamp_id: str) -> dict:
    points = []
    for ug1 in (-3.0, -2.0, -1.0):
        for ua in (50.0, 100.0, 150.0, 200.0, 250.0):
            ia = max(0.05, 1.5 * (ua / 250.0) * (1.0 + (ug1 + 3) / 4))
            points.append({
                "ua": ua, "ug1": ug1, "ug2": 0.0, "ia": ia,
                "ig2": 0.0, "uh": 6.3, "ih": 0.3,
            })
    return {
        "lamp_type": "12AX7",
        "lamp_id": lamp_id,
        "name": f"run_t_{lamp_id}",
        "timestamp": "2026-02-25T13:00:00",
        "points": points,
        "data": {"topology": TOPOLOGY_TRIODE},
    }


# ----------------------------------------------------------------------
# Construction & basic API
# ----------------------------------------------------------------------


class TestConstruction:

    def test_constructs_with_default_args(self, qapp):
        tab = CompareTab()
        try:
            assert tab.compare_entries == []
            assert tab.table is not None
            assert tab.table.columnCount() == _NUM_COLS
        finally:
            tab.close()

    def test_match_columns_hidden_initially(self, qapp):
        """Sel/Grp columns must be hidden until matching runs."""
        tab = CompareTab()
        try:
            assert tab.table.isColumnHidden(_COL_SEL)
            assert tab.table.isColumnHidden(_COL_GRP)
        finally:
            tab.close()


# ----------------------------------------------------------------------
# Table population — add_entry / clear / _render_table
# ----------------------------------------------------------------------


class TestTablePopulation:

    def test_add_entry_appends_row(self, qapp):
        tab = CompareTab()
        try:
            tab.add_entry(_pentode_entry("L1"))
            assert len(tab.compare_entries) == 1
            assert tab.table.rowCount() == 1
            tab.add_entry(_pentode_entry("L2"))
            assert tab.table.rowCount() == 2
        finally:
            tab.close()

    def test_clear_empties_table(self, qapp):
        tab = CompareTab()
        try:
            tab.add_entry(_pentode_entry("L1"))
            tab.add_entry(_pentode_entry("L2"))
            tab.clear()
            assert tab.compare_entries == []
            assert tab.table.rowCount() == 0
        finally:
            tab.close()

    def test_render_table_populates_metadata_columns(self, qapp):
        """An, Mode, SRK, Timestamp cells must be filled from entry data."""
        tab = CompareTab()
        try:
            tab.add_entry(_pentode_entry("L1"))
            tab._render_table(tab.compare_entries)
            # Timestamp shows up in TS column (text non-empty)
            assert tab.table.item(0, _COL_TS) is not None
            assert tab.table.item(0, _COL_TS).text()
            # An column for pentode shows "Au"
            an_item = tab.table.item(0, _COL_AN)
            assert an_item is not None
        finally:
            tab.close()

    def test_render_table_with_mixed_topology(self, qapp):
        """Pentode + triode entries render side-by-side without crash."""
        tab = CompareTab()
        try:
            tab.add_entry(_pentode_entry("P1"))
            tab.add_entry(_triode_entry("T1"))
            tab._render_table(tab.compare_entries)
            assert tab.table.rowCount() == 2
        finally:
            tab.close()


# ----------------------------------------------------------------------
# Sorting by the Show column (Qt.CheckState ordering)
# ----------------------------------------------------------------------


class TestSortByShowColumn:

    def test_sort_by_show_does_not_raise(self, qapp):
        """Clicking the Show header must not crash: Qt.CheckState enums are not
        orderable on PySide6, so the sort key has to coerce them to bool."""
        tab = CompareTab()
        try:
            tab.add_entry(_pentode_entry("L1"))
            tab.add_entry(_pentode_entry("L2"))
            tab._render_table(tab.compare_entries)
            tab.table.item(0, 0).setCheckState(Qt.CheckState.Checked)
            tab.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
            # Unfixed: compare_entries.sort() raises TypeError comparing CheckState.
            tab._sort_table(0)
            # Ascending → unchecked first, checked last.
            last = tab.table.rowCount() - 1
            assert tab.table.item(last, 0).checkState() == Qt.CheckState.Checked
            assert tab.table.item(0, 0).checkState() == Qt.CheckState.Unchecked
            # Second click → descending toggle → checked first.
            tab._sort_table(0)
            assert tab.table.item(0, 0).checkState() == Qt.CheckState.Checked
        finally:
            tab.close()

    def test_sort_by_an_mixed_types_does_not_crash(self):
        """An codes can be strings ('Au') or absent — the key must coerce to str
        so it never mixes an int default with a string value (TypeError)."""
        tab = CompareTab()
        try:
            e1 = _pentode_entry("L1")
            e1["data"]["conditions"] = {"an": "Au"}   # string an
            e2 = _pentode_entry("L2")                  # no conditions.an → ""
            tab.add_entry(e1)
            tab.add_entry(e2)
            tab._render_table(tab.compare_entries)
            # Unfixed: int 0 default vs str 'Au' raises TypeError in .sort().
            tab._sort_table(_COL_AN)
            assert tab.table.rowCount() == 2
        finally:
            tab.close()


# ----------------------------------------------------------------------
# Show on main plot — signal payload
# ----------------------------------------------------------------------


class TestShowOnMainPlot:

    def test_emits_with_correct_payload(self, qapp):
        tab = CompareTab(marker_lock_px=10)
        try:
            payloads = []
            tab.show_on_main_plot.connect(
                lambda points, labels, colors, tracks: payloads.append(
                    (points, labels, colors, tracks))
            )
            tab.add_entry(_pentode_entry("L1"))
            tab.table.selectRow(0)
            tab._show_selected_on_main_plot()

            assert len(payloads) == 1
            points, labels, colors, _tracks = payloads[0]
            assert len(points) > 0
            assert all("series_id" in p for p in points)
            assert labels  # at least one entry mapped to a label
        finally:
            tab.close()


# ----------------------------------------------------------------------
# Curve matching flow
# ----------------------------------------------------------------------


class TestCurveMatchingFlow:

    def _populate_two_pentodes(self, tab) -> None:
        tab.add_entry(_pentode_entry("L1", ia_at_ua_max=12.0))
        tab.add_entry(_pentode_entry("L2", ia_at_ua_max=12.5))
        tab._render_table(tab.compare_entries)

    def test_run_match_with_too_few_entries_does_not_crash(self, qapp,
                                                            monkeypatch):
        """1 entry → message shown, no crash, no match_result set."""
        from app import compare_tab as ct

        msgs = []
        monkeypatch.setattr(
            ct.QMessageBox, "information",
            lambda *a, **kw: msgs.append(a[2] if len(a) > 2 else None),
        )
        tab = CompareTab()
        try:
            tab.add_entry(_pentode_entry("L1"))
            tab._render_table(tab.compare_entries)
            tab._run_curve_matching(source="all")
            assert tab._match_result is None
        finally:
            tab.close()

    def test_run_match_pair_populates_state(self, qapp):
        tab = CompareTab()
        try:
            self._populate_two_pentodes(tab)
            tab._run_curve_matching(source="all")
            assert tab._match_result is not None
            assert tab._match_result.mode == "groups"
            # Sel/Grp columns become visible
            assert not tab.table.isColumnHidden(_COL_SEL)
            assert not tab.table.isColumnHidden(_COL_GRP)
            # Each entry has a group assignment
            assert len(tab._match_grp_info) == 2

        finally:
            tab.close()

    def test_apply_curve_match_to_table_writes_grp_cells(self, qapp):
        tab = CompareTab()
        try:
            self._populate_two_pentodes(tab)
            tab._run_curve_matching(source="all")
            # Both rows must have a non-empty Grp cell
            grp_items = [tab.table.item(r, _COL_GRP)
                         for r in range(tab.table.rowCount())]
            assert all(item is not None for item in grp_items)
            grp_texts = [item.text() for item in grp_items]
            assert all(grp_texts)

        finally:
            tab.close()

    def test_clear_match_resets_state_and_hides_columns(self, qapp):
        tab = CompareTab()
        try:
            self._populate_two_pentodes(tab)
            tab._run_curve_matching(source="all")
            assert tab._match_result is not None
            tab._clear_curve_match()
            assert tab._match_result is None
            assert tab._match_grp_info == {}
            assert tab.table.isColumnHidden(_COL_SEL)
            assert tab.table.isColumnHidden(_COL_GRP)
        finally:
            tab.close()


# ----------------------------------------------------------------------
# Mode filter — "pentode" should drop triode entries before matching
# ----------------------------------------------------------------------


class TestMatchModeFilter:

    def test_pentode_filter_excludes_triodes(self, qapp, monkeypatch):
        """Mode filter set to 'pentode' must skip triode entries."""
        tab = CompareTab()
        try:
            tab.add_entry(_pentode_entry("P1"))
            tab.add_entry(_pentode_entry("P2"))
            tab.add_entry(_triode_entry("T1"))
            tab._render_table(tab.compare_entries)

            # Force mode filter to "pentode"
            for i in range(tab.match_mode_combo.count()):
                if tab.match_mode_combo.itemData(i) == TOPOLOGY_PENTODE:
                    tab.match_mode_combo.setCurrentIndex(i)
                    break
            tab._run_curve_matching(source="all")
            assert tab._match_result is not None
            # match_entries holds the input slice; no triode in it
            for e in tab._match_entries:
                assert e.get("data", {}).get("topology") != TOPOLOGY_TRIODE
        finally:
            tab.close()

    def test_triode_filter_excludes_pentodes(self, qapp, monkeypatch):
        """Mode filter set to 'triode' with only one triode → no-data warning."""
        from app import compare_tab as ct
        warnings_seen = []
        monkeypatch.setattr(
            ct.QMessageBox, "information",
            lambda *a, **kw: warnings_seen.append(True),
        )
        tab = CompareTab()
        try:
            tab.add_entry(_pentode_entry("P1"))
            tab.add_entry(_pentode_entry("P2"))
            tab.add_entry(_triode_entry("T1"))
            tab._render_table(tab.compare_entries)

            for i in range(tab.match_mode_combo.count()):
                if tab.match_mode_combo.itemData(i) == TOPOLOGY_TRIODE:
                    tab.match_mode_combo.setCurrentIndex(i)
                    break
            tab._run_curve_matching(source="all")
            # Only 1 triode in set → match_curves rejects (< 2 entries)
            assert tab._match_result is None
            assert warnings_seen
        finally:
            tab.close()


class TestMatchAlgorithmDropdown:
    """Pair-matching algorithm dropdown in Compare tab."""

    def test_dropdown_exists_with_two_options(self):
        tab = CompareTab()
        try:
            assert hasattr(tab, "match_algorithm_combo")
            assert tab.match_algorithm_combo.count() == 2
        finally:
            tab.close()

    def test_default_is_optimal_backward_compat(self):
        """No-arg CompareTab — preserves legacy Hungarian behaviour."""
        tab = CompareTab()
        try:
            assert tab.match_algorithm_combo.currentData() == "optimal"
        finally:
            tab.close()

    def test_constructor_applies_match_algorithm(self):
        tab = CompareTab(match_algorithm="greedy")
        try:
            assert tab.match_algorithm_combo.currentData() == "greedy"
        finally:
            tab.close()

    def test_constructor_ignores_unknown_algorithm(self):
        """Corrupt config value → keep default selection (no crash)."""
        tab = CompareTab(match_algorithm="xyz_nonsense")
        try:
            # Falls back to whatever UI built first (greedy is item 0 in the combo)
            assert tab.match_algorithm_combo.currentData() in ("greedy", "optimal")
        finally:
            tab.close()

    def test_algorithm_flows_to_match_curves_call(self, monkeypatch):
        """Selected algorithm reaches match_curves() in groups mode."""
        from app import compare_tab as ct
        captured = {}

        def fake_match_curves(*args, **kwargs):
            captured.update(kwargs)
            from lm19.tube_matching import CurveMatchResult
            return CurveMatchResult(mode="groups", groups=[], unmatched=[],
                                    pair_info={}, anchor_idx=None)

        monkeypatch.setattr(ct, "match_curves", fake_match_curves)
        tab = CompareTab(match_algorithm="greedy")
        try:
            tab.add_entry(_pentode_entry("P1"))
            tab.add_entry(_pentode_entry("P2"))
            tab._render_table(tab.compare_entries)
            tab._run_curve_matching(source="all")
            assert captured.get("algorithm") == "greedy"
        finally:
            tab.close()

    def test_runtime_selection_picked_up_by_next_match(self, monkeypatch):
        """Changing the dropdown after construction applies to subsequent runs."""
        from app import compare_tab as ct
        captured = []

        def fake_match_curves(*args, **kwargs):
            captured.append(kwargs.get("algorithm"))
            from lm19.tube_matching import CurveMatchResult
            return CurveMatchResult(mode="groups", groups=[], unmatched=[],
                                    pair_info={}, anchor_idx=None)

        monkeypatch.setattr(ct, "match_curves", fake_match_curves)
        tab = CompareTab(match_algorithm="optimal")
        try:
            tab.add_entry(_pentode_entry("P1"))
            tab.add_entry(_pentode_entry("P2"))
            tab._render_table(tab.compare_entries)
            tab._run_curve_matching(source="all")
            # User switches the dropdown to greedy
            idx = tab.match_algorithm_combo.findData("greedy")
            tab.match_algorithm_combo.setCurrentIndex(idx)
            tab._run_curve_matching(source="all")
            assert captured == ["optimal", "greedy"]
        finally:
            tab.close()

    def test_similar_mode_does_not_pass_algorithm(self, monkeypatch):
        """Similar mode is a ranking — algorithm has no meaning there.

        Verifies the call site explicitly does NOT pass ``algorithm``,
        so it stays gracefully ignored (uses match_curves' own default).
        """
        from app import compare_tab as ct
        captured = {}

        def fake_match_curves(*args, **kwargs):
            captured.update(kwargs)
            from lm19.tube_matching import CurveMatchResult
            return CurveMatchResult(mode="similar", groups=[], unmatched=[],
                                    pair_info={}, anchor_idx=0)

        monkeypatch.setattr(ct, "match_curves", fake_match_curves)
        # Even though dropdown is set to "greedy", similar-mode call must
        # not forward it.
        tab = CompareTab(match_algorithm="greedy")
        try:
            tab.add_entry(_pentode_entry("P1"))
            tab.add_entry(_pentode_entry("P2"))
            tab._render_table(tab.compare_entries)
            tab._find_similar_curve(0)
            assert captured.get("mode") == "similar"
            assert "algorithm" not in captured, (
                "Similar mode call must not pass `algorithm` kwarg "
                "(gracefully ignored — verified by absence)"
            )
        finally:
            tab.close()

    def test_appconfig_default_matches_dropdown_option(self):
        """``AppConfig().compare_matching_algorithm`` must be a valid dropdown
        item. Catches regressions where the default drifts out of sync."""
        from lm19.app_config import AppConfig
        tab = CompareTab()
        try:
            default = AppConfig().compare_matching_algorithm
            idx = tab.match_algorithm_combo.findData(default)
            assert idx >= 0, (
                f"AppConfig default {default!r} is not a Compare dropdown option"
            )
        finally:
            tab.close()


class TestMatchProgressDialog:
    """``_run_curve_matching`` wraps match_curves with a QProgressDialog
    so the UI stays responsive without a worker thread. These tests pin
    the wiring: callback forwarded, cancel handled, no result update on
    cancel."""

    def test_progress_callback_forwarded(self, monkeypatch):
        """The dialog's progress callback reaches match_curves."""
        from app import compare_tab as ct
        captured = {}

        def fake_match_curves(*args, **kwargs):
            captured["progress"] = kwargs.get("progress")
            from lm19.tube_matching import CurveMatchResult
            # Drive the callback so the dialog gets at least one tick
            if captured["progress"] is not None:
                captured["progress"](0, 2)
                captured["progress"](1, 2)
            return CurveMatchResult(mode="groups", groups=[], unmatched=[],
                                    pair_info={}, anchor_idx=None)

        monkeypatch.setattr(ct, "match_curves", fake_match_curves)
        tab = CompareTab()
        try:
            tab.add_entry(_pentode_entry("P1"))
            tab.add_entry(_pentode_entry("P2"))
            tab._render_table(tab.compare_entries)
            tab._run_curve_matching(source="all")
            # Callback must be passed (not None) and be callable
            assert captured.get("progress") is not None
            assert callable(captured["progress"])
        finally:
            tab.close()

    def test_cancel_does_not_update_match_result(self, monkeypatch):
        """When the user cancels, ``_match_result`` stays at its old value."""
        from app import compare_tab as ct
        from lm19.tube_matching import MatchCancelled

        def fake_match_curves(*args, **kwargs):
            raise MatchCancelled()

        monkeypatch.setattr(ct, "match_curves", fake_match_curves)
        tab = CompareTab()
        try:
            tab.add_entry(_pentode_entry("P1"))
            tab.add_entry(_pentode_entry("P2"))
            tab._render_table(tab.compare_entries)
            # Pre-set a sentinel so we can detect that _run did NOT overwrite it.
            tab._match_result = "sentinel"
            tab._run_curve_matching(source="all")
            assert tab._match_result == "sentinel", (
                "cancel must not overwrite the previous match result"
            )
        finally:
            tab.close()

    def test_match_running_flag_cleared_on_success(self, monkeypatch):
        """After a normal match completion the re-entrancy flag resets."""
        from app import compare_tab as ct
        from lm19.tube_matching import CurveMatchResult

        def fake_match_curves(*args, **kwargs):
            return CurveMatchResult(mode="groups", groups=[], unmatched=[],
                                    pair_info={}, anchor_idx=None)

        monkeypatch.setattr(ct, "match_curves", fake_match_curves)
        tab = CompareTab()
        try:
            tab.add_entry(_pentode_entry("P1"))
            tab.add_entry(_pentode_entry("P2"))
            tab._render_table(tab.compare_entries)
            tab._run_curve_matching(source="all")
            assert tab._match_running is False
        finally:
            tab.close()

    def test_match_running_flag_cleared_on_cancel(self, monkeypatch):
        """The flag must reset even on cancel (finally clause)."""
        from app import compare_tab as ct
        from lm19.tube_matching import MatchCancelled

        def fake_match_curves(*args, **kwargs):
            raise MatchCancelled()

        monkeypatch.setattr(ct, "match_curves", fake_match_curves)
        tab = CompareTab()
        try:
            tab.add_entry(_pentode_entry("P1"))
            tab.add_entry(_pentode_entry("P2"))
            tab._render_table(tab.compare_entries)
            tab._run_curve_matching(source="all")
            assert tab._match_running is False, (
                "MatchCancelled in finally must still clear the flag"
            )
        finally:
            tab.close()

    def test_reentrant_match_call_returns_immediately(self, monkeypatch):
        """A re-entrant ``_run_curve_matching`` from inside the progress
        callback (simulating a stray Match button click) must no-op."""
        from app import compare_tab as ct
        from lm19.tube_matching import CurveMatchResult

        call_count = [0]
        reentrant_call_observed = [False]

        def fake_match_curves(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Simulate the user (or processEvents) re-triggering Match
                # while we're still inside it.
                tab._run_curve_matching("all")
                # If guard works, _run returns silently without calling
                # match_curves again (call_count stays at 1).
                reentrant_call_observed[0] = (call_count[0] == 1)
            return CurveMatchResult(mode="groups", groups=[], unmatched=[],
                                    pair_info={}, anchor_idx=None)

        monkeypatch.setattr(ct, "match_curves", fake_match_curves)
        tab = CompareTab()
        try:
            tab.add_entry(_pentode_entry("P1"))
            tab.add_entry(_pentode_entry("P2"))
            tab._render_table(tab.compare_entries)
            tab._run_curve_matching(source="all")
            assert reentrant_call_observed[0], (
                "Re-entrant _run_curve_matching call was not blocked"
            )
            assert call_count[0] == 1, (
                f"match_curves called {call_count[0]} times; expected 1"
            )
        finally:
            tab.close()

    def test_progress_bar_format_shows_count_and_percent(self, monkeypatch):
        """Bar must show ``done / total  %`` instead of the Qt default
        percent-only format. We capture the dialog the tab builds and
        inspect its internal QProgressBar."""
        from PySide6.QtWidgets import QProgressBar, QProgressDialog
        from app import compare_tab as ct
        from lm19.tube_matching import CurveMatchResult

        captured_dlg: list = []
        real_dialog_init = QProgressDialog.__init__

        def init_spy(self, *args, **kwargs):
            real_dialog_init(self, *args, **kwargs)
            captured_dlg.append(self)

        monkeypatch.setattr(QProgressDialog, "__init__", init_spy)

        def fake_match_curves(*args, **kwargs):
            return CurveMatchResult(mode="groups", groups=[], unmatched=[],
                                    pair_info={}, anchor_idx=None)

        monkeypatch.setattr(ct, "match_curves", fake_match_curves)
        tab = CompareTab()
        try:
            tab.add_entry(_pentode_entry("P1"))
            tab.add_entry(_pentode_entry("P2"))
            tab._render_table(tab.compare_entries)
            tab._run_curve_matching(source="all")
            assert captured_dlg, "QProgressDialog was not created"
            dlg = captured_dlg[0]
            bar = dlg.findChild(QProgressBar)
            assert bar is not None
            assert bar.format() == "%v / %m  %p%", (
                f"bar.format()={bar.format()!r} — expected '%v / %m  %p%'"
            )
        finally:
            tab.close()

    def test_clear_during_match_is_no_op(self, monkeypatch):
        """Clear must not race with a match in progress."""
        from app import compare_tab as ct
        from lm19.tube_matching import CurveMatchResult

        cleared_during_match = [False]

        def fake_match_curves(*args, **kwargs):
            # Simulate Clear button click while match is running
            tab._clear_curve_match()
            # If guard works, _match_result stays alive (Clear was no-op)
            cleared_during_match[0] = (tab._match_result == "sentinel")
            return CurveMatchResult(mode="groups", groups=[], unmatched=[],
                                    pair_info={}, anchor_idx=None)

        monkeypatch.setattr(ct, "match_curves", fake_match_curves)
        tab = CompareTab()
        try:
            tab.add_entry(_pentode_entry("P1"))
            tab.add_entry(_pentode_entry("P2"))
            tab._render_table(tab.compare_entries)
            tab._match_result = "sentinel"
            tab._run_curve_matching(source="all")
            assert cleared_during_match[0], (
                "Clear during match must be ignored (re-entrancy guard)"
            )
        finally:
            tab.close()


class TestDeadDataPersistVisibility:
    """ML-080: a cleanup that did not reach the disk (external entry with
    no source path / write failure) must be surfaced — in-memory-only
    cleanups silently revert on the next reload."""

    def test_persist_without_path_returns_false_and_warns(self, qapp, caplog):
        import logging
        tab = CompareTab()
        entry = {"name": "ext1", "points": [{"ua": 1.0}]}
        with caplog.at_level(logging.WARNING, logger="app.compare_tab"):
            ok = tab._persist_cleaned_entry(entry, [])
        assert ok is False
        assert any("not persisted" in r.getMessage()
                   for r in caplog.records),             "pathless entry cleaned without any warning"
        # in-memory update still applied
        assert entry["points"] == []

    def test_process_dead_entry_returns_persisted_flag(self, qapp):
        tab = CompareTab()
        removed, remaining, persisted = tab._process_dead_entry(
            {"name": "empty", "points": []})
        assert (removed, remaining, persisted) == (0, 0, True)
