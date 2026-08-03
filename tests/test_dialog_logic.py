"""Behavior pins for dialog option logic (ML-146).

Six dialog classes (SpiceExportDialog / CsvOptionsDialog /
UtdExportDialog / ImportMetaDialog / CsvImportDialog / ClearSeriesDialog)
and health_plan_builder were never constructed by any test — their
option semantics (circuit switching, separators, column mapping,
series selection) reached export files and import mappings without
a single pin. All checks are observable contracts (accessors /
visibility via isHidden), non-degenerate data, twins covered
one by one.

Run:  py -m pytest tests/test_dialog_logic.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import i18n_setup
from i18n_setup import t
from lm19.amplifier.constants import (
    CIRCUIT_SE,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)
from lm19.tube_model_base import (
    MODEL_TYPE_KOREN,
)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    i18n_setup.setup("en")
    return QApplication.instance() or QApplication([])


# ═══════════════════════════════════════════════════════════════════
#  ClearSeriesDialog — series selection for removal
# ═══════════════════════════════════════════════════════════════════

def _series_info() -> List[Dict]:
    # Unsorted sids (7, 3, 12): the checkbox->sid mapping must not
    # depend on ordering.
    return [
        {"series_id": 7, "label": "Scan A", "n_points": 120},
        {"series_id": 3, "label": "Overlay B", "n_points": 45},
        {"series_id": 12, "label": "Import C", "n_points": 88},
    ]


class TestClearSeriesDialog:

    def test_defaults_nothing_selected_button_disabled(self, qapp):
        from app.clear_dialog import ClearSeriesDialog
        dlg = ClearSeriesDialog(_series_info())
        assert dlg.selected_series_ids == []
        assert dlg.is_remove_all is False
        assert not dlg._remove_sel_btn.isEnabled()

    def test_button_enables_and_disables_with_selection(self, qapp):
        """Both ends: enabled on first check, disabled on uncheck."""
        from app.clear_dialog import ClearSeriesDialog
        dlg = ClearSeriesDialog(_series_info())
        cb0 = dlg._checks[0][0]
        cb0.setChecked(True)
        assert dlg._remove_sel_btn.isEnabled()
        cb0.setChecked(False)
        assert not dlg._remove_sel_btn.isEnabled()

    def test_selected_ids_follow_checked_boxes(self, qapp):
        from app.clear_dialog import ClearSeriesDialog
        dlg = ClearSeriesDialog(_series_info())
        dlg._checks[0][0].setChecked(True)   # sid 7
        dlg._checks[2][0].setChecked(True)   # sid 12
        assert dlg.selected_series_ids == [7, 12]

    def test_remove_selected_accepts_without_remove_all(self, qapp):
        from app.clear_dialog import ClearSeriesDialog
        dlg = ClearSeriesDialog(_series_info())
        dlg._checks[1][0].setChecked(True)
        dlg._remove_sel_btn.click()
        assert dlg.result() == dlg.DialogCode.Accepted
        assert dlg.is_remove_all is False
        assert dlg.selected_series_ids == [3]

    def test_remove_all_needs_no_selection(self, qapp):
        """Negative space: Remove All works with zero checks."""
        from app.clear_dialog import ClearSeriesDialog
        dlg = ClearSeriesDialog(_series_info())
        dlg._on_remove_all()
        assert dlg.result() == dlg.DialogCode.Accepted
        assert dlg.is_remove_all is True
        assert dlg.selected_series_ids == []

    def test_empty_series_info(self, qapp):
        from app.clear_dialog import ClearSeriesDialog
        dlg = ClearSeriesDialog([])
        assert dlg.selected_series_ids == []
        assert not dlg._remove_sel_btn.isEnabled()


# ═══════════════════════════════════════════════════════════════════
#  SpiceExportDialog — model/circuit selection, circuit switching
# ═══════════════════════════════════════════════════════════════════

_TWO_SERIES = {1: "Series #1", 2: "Series #2"}


class TestSpiceExportDialog:

    def _dlg(self, series=None):
        from app.export_manager import SpiceExportDialog
        return SpiceExportDialog(series_labels=series or _TWO_SERIES)

    def test_defaults(self, qapp):
        dlg = self._dlg()
        assert dlg.model_type == MODEL_TYPE_KOREN
        assert dlg.generate_test_schematic is True
        assert dlg.generate_amp_schematic is False
        assert dlg.amp_circuit == CIRCUIT_SE
        assert dlg.r_load == "8"
        assert dlg.f_low == 20.0

    @pytest.mark.parametrize("idx,expected", [
        (1, "dempwolf"), (2, "reefman"), (0, "koren"),
    ])
    def test_model_type_follows_radio(self, qapp, idx, expected):
        dlg = self._dlg()
        dlg._model_radios[idx].setChecked(True)
        assert dlg.model_type == expected

    @pytest.mark.parametrize("circuit,xfmr_hidden,pp_hidden", [
        ("se", True, True),
        ("se_xfmr", False, True),
        ("cf", True, True),
        ("pp", False, False),
    ])
    def test_circuit_visibility_matrix(self, qapp, circuit,
                                       xfmr_hidden, pp_hidden):
        """All 4 circuits (twins): xfmr fields visible for se_xfmr/pp,
        Tube B — only for pp."""
        dlg = self._dlg()
        idx = dlg.amp_circuit_combo.findData(circuit)
        assert idx >= 0
        dlg.amp_circuit_combo.setCurrentIndex(idx)
        assert dlg.amp_circuit == circuit
        assert all(w.isHidden() == xfmr_hidden
                   for w in dlg._xfmr_widgets), circuit
        assert all(w.isHidden() == pp_hidden
                   for w in dlg._pp_widgets), circuit

    def test_source_label_says_tube_a_only_for_pp(self, qapp):
        dlg = self._dlg()
        dlg.amp_circuit_combo.setCurrentIndex(
            dlg.amp_circuit_combo.findData("pp"))
        assert dlg._source_label.text() == "Tube A:"
        dlg.amp_circuit_combo.setCurrentIndex(
            dlg.amp_circuit_combo.findData("se"))
        assert dlg._source_label.text() != "Tube A:"

    @pytest.mark.parametrize("idx,expected", [(0, "4"), (2, "16")])
    def test_r_load_follows_combo(self, qapp, idx, expected):
        dlg = self._dlg()
        dlg.rload_combo.setCurrentIndex(idx)
        assert dlg.r_load == expected

    def test_pp_tube_b_defaults_to_matched_none(self, qapp):
        """Negative space: Tube B default = matched (None)."""
        dlg = self._dlg()
        assert dlg.pp_tube_b_sid is None
        dlg.pp_tube_b_combo.setCurrentIndex(1)   # first real sid
        assert dlg.pp_tube_b_sid == 1
        dlg.pp_tube_b_combo.setCurrentIndex(2)   # second sid, not first
        assert dlg.pp_tube_b_sid == 2
        assert dlg.pp_tube_a_sid == 1            # first sid by default

    def test_single_series_hides_source_row(self, qapp):
        """has_series boundary: >1 -> visible, 1 -> hidden."""
        dlg_multi = self._dlg(_TWO_SERIES)
        assert all(not w.isHidden() for w in dlg_multi._source_widgets)
        dlg_single = self._dlg({1: "only"})
        assert all(w.isHidden() for w in dlg_single._source_widgets)

    def test_model_type_falls_back_when_nothing_checked(self, qapp):
        """The defensive fallback branch, unreachable through the UI.

        Radio auto-exclusion keeps one button checked, so this return is
        never taken in normal use -- and an edit that changed it to a
        different model type would ship silently.
        """
        dlg = self._dlg()
        for rb in dlg._model_radios:
            rb.setAutoExclusive(False)
            rb.setChecked(False)
        assert dlg.model_type == MODEL_TYPE_KOREN


# ===================================================================
#  ModelDialog._get_topology -- fit topology, all three branches
# ===================================================================

class _StopFit(Exception):
    """Sentinel: abort _on_accept once the topology has been captured."""


class TestModelDialogTopology:
    """Both halves of the dedup: the resolver AND its call site.

    ``triode_connected`` must reach the fitter as itself (it routes to the
    triode fitter, unlike a true pentode), so each branch is pinned; the
    spy then proves the fit path consumes the resolver's answer instead of
    deciding for itself again -- the exact shape this dedup removed.
    """

    _POINTS = [{"ua": 100.0, "ug1": -2.0, "ia": 5.0, "ug2": 100.0, "ig2": 0.5},
               {"ua": 200.0, "ug1": -2.0, "ia": 12.0, "ug2": 200.0, "ig2": 1.1}]

    def _dlg(self, *, is_triode: bool, track: bool):
        from app.model_dialog import ModelDialog
        return ModelDialog(points=list(self._POINTS), is_triode=is_triode,
                           scan_settings={"ug2_track_ua": track})

    @pytest.mark.parametrize("is_triode,track,expected", [
        (True, False, TOPOLOGY_TRIODE),
        (True, True, TOPOLOGY_TRIODE),        # true triode outranks track
        (False, True, TOPOLOGY_TRIODE_CONNECTED),
        (False, False, TOPOLOGY_PENTODE),
    ])
    def test_resolver_branches(self, qapp, is_triode, track, expected):
        dlg = self._dlg(is_triode=is_triode, track=track)
        assert dlg._get_topology() == expected

    @pytest.mark.parametrize("is_triode,track,expected", [
        (False, True, TOPOLOGY_TRIODE_CONNECTED),
        (False, False, TOPOLOGY_PENTODE),
        (True, False, TOPOLOGY_TRIODE),
    ])
    def test_fit_path_passes_resolved_topology(self, qapp, monkeypatch,
                                               is_triode, track, expected):
        """Call-site spy on the first consumer of the resolved topology.

        A unit pin on the resolver does not prove the caller uses it: the
        scenarios differ (a pentode and a triode-connected scan resolve to
        different values), so a hardcoded argument here fails.
        """
        import app.model_dialog as md
        dlg = self._dlg(is_triode=is_triode, track=track)
        seen = {}

        def spy(points, **kwargs):
            seen["topology"] = kwargs.get("topology")
            raise _StopFit

        monkeypatch.setattr(md, "detect_dead_data", spy)
        with pytest.raises(_StopFit):
            dlg._on_accept()
        assert seen["topology"] == expected



# ═══════════════════════════════════════════════════════════════════
#  CsvOptionsDialog — format/separator/computed/multi
# ═══════════════════════════════════════════════════════════════════

class TestCsvOptionsDialog:

    def _dlg(self, multi=False):
        from app.export_manager import CsvOptionsDialog
        return CsvOptionsDialog(multi=multi)

    def test_defaults(self, qapp):
        dlg = self._dlg()
        assert dlg.is_matrix is False
        assert dlg.separator == ";"
        assert dlg.include_computed is True
        assert dlg.separate_files is False
        assert dlg.matrix_param_combo.isHidden()
        assert dlg.cb_computed.isEnabled()

    @pytest.mark.parametrize("radio,expected", [
        ("rb_comma", ","), ("rb_tab", "\t"), ("rb_semi", ";"),
    ])
    def test_separator_twins(self, qapp, radio, expected):
        dlg = self._dlg()
        getattr(dlg, radio).setChecked(True)
        assert dlg.separator == expected

    def test_matrix_toggle_both_directions(self, qapp):
        """Matrix: parameter widgets visible, computed disabled; and
        restored on the way back (both ends)."""
        dlg = self._dlg()
        dlg.rb_matrix.setChecked(True)
        assert dlg.is_matrix is True
        assert not dlg.matrix_param_combo.isHidden()
        assert not dlg.cb_computed.isEnabled()
        dlg.rb_flat.setChecked(True)
        assert dlg.is_matrix is False
        assert dlg.matrix_param_combo.isHidden()
        assert dlg.cb_computed.isEnabled()

    def test_matrix_parameter_follows_combo(self, qapp):
        dlg = self._dlg()
        dlg.matrix_param_combo.setCurrentIndex(1)
        assert dlg.matrix_parameter == "Ig2"
        dlg.matrix_param_combo.setCurrentIndex(2)
        assert dlg.matrix_parameter == "Pa"

    def test_multi_mode_separate_files(self, qapp):
        dlg = self._dlg(multi=True)
        assert dlg.separate_files is False       # single file by default
        dlg.rb_separate.setChecked(True)
        assert dlg.separate_files is True

    def test_single_mode_has_no_multi_radios(self, qapp):
        """Negative space: without multi no radios and no crash."""
        dlg = self._dlg(multi=False)
        assert dlg.rb_separate is None
        assert dlg.separate_files is False


# ═══════════════════════════════════════════════════════════════════
#  UtdExportDialog — matrix orientation
# ═══════════════════════════════════════════════════════════════════

class TestUtdExportDialog:

    def _dlg(self):
        from app.export_manager import UtdExportDialog
        # 13 != 5 — a row/column swap is distinguishable.
        return UtdExportDialog(n_ua=13, n_ug1=5)

    def test_default_output_matrix_info(self, qapp):
        dlg = self._dlg()
        assert dlg.fmt == "output"
        info = dlg.matrix_info.text()
        assert "13" in info and "5" in info
        assert "Va" in info
        assert info.index("13") < info.index("5")   # rows=Ua first

    def test_transfer_swaps_rows_cols(self, qapp):
        dlg = self._dlg()
        dlg.rb_transfer.setChecked(True)
        assert dlg.fmt == "transfer"
        info = dlg.matrix_info.text()
        assert "Vg" in info
        assert info.index("5") < info.index("13")   # rows=Ug1 first


# ═══════════════════════════════════════════════════════════════════
#  ImportMetaDialog — import metadata
# ═══════════════════════════════════════════════════════════════════

class TestImportMetaDialog:

    def _dlg(self, defaults=None, count=42):
        from app.import_dialog import ImportMetaDialog
        return ImportMetaDialog(defaults=defaults, point_count=count)

    def test_defaults_flow_into_fields(self, qapp):
        dlg = self._dlg(defaults={
            "tube_type": "6P1P", "lamp_id": "L-07", "name": "run3",
            "description": "  padded  ", "ug2_mode": TOPOLOGY_TRIODE_CONNECTED,
            "vs": 215.0, "vh": 6.3,
        })
        assert dlg.tube_type == "6P1P"
        assert dlg.lamp_id == "L-07"
        assert dlg.name == "run3"
        assert dlg.description == "padded"        # strip
        assert dlg.ug2_mode == TOPOLOGY_TRIODE_CONNECTED
        assert dlg.ug2 == 215.0
        assert dlg.uh == 6.3

    def test_blank_fields_fall_back_to_i18n_defaults(self, qapp):
        dlg = self._dlg(defaults={})
        assert dlg.tube_type == t("import.Default_tube_type")
        assert dlg.lamp_id == t("import.Default_lamp_id")
        assert dlg.name == t("import.Default_name")

    @pytest.mark.parametrize("mode", ["triode", "triode_connected",
                                      "pentode"])
    def test_all_modes_selectable(self, qapp, mode):
        dlg = self._dlg(defaults={"ug2_mode": mode})
        assert dlg.ug2_mode == mode

    def test_mode_fallbacks(self, qapp):
        """Missing key -> pentode; garbage -> index 0 (triode) — pins
        the max(0, findData) guard."""
        assert self._dlg(defaults={}).ug2_mode == TOPOLOGY_PENTODE
        assert self._dlg(defaults={"ug2_mode": "bogus"}).ug2_mode == TOPOLOGY_TRIODE


# ═══════════════════════════════════════════════════════════════════
#  CsvImportDialog — separator + column mapping
# ═══════════════════════════════════════════════════════════════════

_CSV_TEXT = "Ua;Ug1;Ia;banana\n100;-2;10.5;x\n150;-2;15.1;y\n"


class TestCsvImportDialog:

    def _dlg(self, text=_CSV_TEXT):
        from app.import_dialog import CsvImportDialog
        return CsvImportDialog(text=text)

    def test_auto_separator_detects_semicolon(self, qapp):
        dlg = self._dlg()
        assert dlg.rb_auto.isChecked()
        assert dlg.separator == ";"

    @pytest.mark.parametrize("radio,expected", [
        ("rb_semi", ";"), ("rb_comma", ","), ("rb_tab", "\t"),
    ])
    def test_manual_separator_twins(self, qapp, radio, expected):
        dlg = self._dlg()
        getattr(dlg, radio).setChecked(True)
        assert dlg.separator == expected

    def test_auto_mapping_detects_known_and_skips_unknown(self, qapp):
        """A banana column (negative space) stays out of the mapping;
        known headers are detected by position."""
        dlg = self._dlg()
        mapping = dlg.column_mapping
        assert mapping[0] == "ua"
        assert mapping[1] == "ug1"
        assert mapping[2] == "ia"
        assert 3 not in mapping

    def test_manual_remap_overrides(self, qapp):
        dlg = self._dlg()
        combo = dlg._mapping_combos[3]
        idx = combo.findData("ug2")
        assert idx >= 0
        combo.setCurrentIndex(idx)
        assert dlg.column_mapping[3] == "ug2"

    def test_separator_switch_reparses_preview(self, qapp):
        """Switch call site: changing the radio must re-parse (the
        column count changes)."""
        text = "a,b,c\n1,2,3\n"
        dlg = self._dlg(text=text)
        dlg.rb_semi.setChecked(True)
        assert dlg.preview_table.columnCount() == 1   # 'a,b,c' whole
        dlg.rb_comma.setChecked(True)
        assert dlg.preview_table.columnCount() == 3
        assert len(dlg._mapping_combos) == 3

    def test_comment_lines_skipped_in_preview(self, qapp):
        text = "# comment line\nUa;Ia\n100;10\n"
        dlg = self._dlg(text=text)
        assert dlg.preview_table.item(0, 0).text() == "Ua"
        assert dlg.preview_table.item(0, 1).text() == "Ia"


# ═══════════════════════════════════════════════════════════════════
#  health_plan_builder — dataclass completeness + callback wiring
# ═══════════════════════════════════════════════════════════════════

class TestHealthPlanBuilder:

    def test_build_returns_complete_planwidgets(self, qapp):
        import dataclasses
        from PySide6.QtWidgets import QGroupBox, QWidget
        from app.health_plan_builder import PlanWidgets, build_plan_box
        host = QWidget()
        box, w = build_plan_box(host, on_ug2_mode_toggled=lambda _c: None)
        assert isinstance(box, QGroupBox)
        for f in dataclasses.fields(PlanWidgets):
            assert getattr(w, f.name) is not None, f.name

    def test_ug2_mode_callback_wired(self, qapp):
        """Call site vs function: toggling the track radio must fire
        the HealthTab callback (planned-steps preview)."""
        from PySide6.QtWidgets import QWidget
        from app.health_plan_builder import build_plan_box
        host = QWidget()
        fired: list = []
        _box, w = build_plan_box(host, on_ug2_mode_toggled=fired.append)
        w.ug2_track_radio.setChecked(True)
        assert fired and fired[-1] is True
        w.ug2_independent_radio.setChecked(True)
        assert fired[-1] is False
