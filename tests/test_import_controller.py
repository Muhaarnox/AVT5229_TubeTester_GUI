"""Tests for ImportController."""

from unittest.mock import MagicMock

import pytest

from app.import_controller import ImportController
from lm19.config import LampConfig
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)


def _make_lamp(name="6L6", topology=TOPOLOGY_PENTODE, uh=6.3):
    lamp = MagicMock(spec=LampConfig)
    lamp.tube_type = name
    lamp.topology = topology
    lamp.uh = uh
    return lamp


class TestImportDefaults:
    def test_defaults_from_stem_basic(self):
        ctrl = ImportController(
            parent_widget=MagicMock(),
            compare_tab=MagicMock(),
            tabs=MagicMock(),
            get_lamps=lambda: [],
        )
        defaults = ctrl._import_defaults_from_stem("6L6_test")
        assert defaults["tube_type"] == "6L6_test"
        assert defaults["lamp_id"] == "6L6_test"
        assert defaults["ug2_mode"] == TOPOLOGY_PENTODE
        assert defaults["vh"] == 0.0

    def test_defaults_with_guessed_type(self):
        lamp = _make_lamp("EL34", "pentode", 6.3)
        ctrl = ImportController(
            parent_widget=MagicMock(),
            compare_tab=MagicMock(),
            tabs=MagicMock(),
            get_lamps=lambda: [lamp],
        )
        defaults = ctrl._import_defaults_from_stem(
            "test", guessed_type="EL34", guessed_vs=250.0
        )
        assert defaults["tube_type"] == "EL34"
        assert defaults["vs"] == 250.0
        assert defaults["vh"] == 6.3

    def test_defaults_triode_lamp(self):
        lamp = _make_lamp("12AX7", "triode", 12.6)
        ctrl = ImportController(
            parent_widget=MagicMock(),
            compare_tab=MagicMock(),
            tabs=MagicMock(),
            get_lamps=lambda: [lamp],
        )
        defaults = ctrl._import_defaults_from_stem("12AX7")
        assert defaults["ug2_mode"] == TOPOLOGY_TRIODE
        assert defaults["vh"] == 12.6


class TestFinalizeImport:
    def test_finalize_saves_and_adds_entry(self):
        compare_tab = MagicMock()
        tabs = MagicMock()
        ctrl = ImportController(
            parent_widget=MagicMock(),
            compare_tab=compare_tab,
            tabs=tabs,
            get_lamps=lambda: [],
        )
        dlg = MagicMock()
        dlg.tube_type = "6L6"
        dlg.lamp_id = "L1"
        dlg.name = "test"
        dlg.description = "desc"
        dlg.ug2_mode = TOPOLOGY_PENTODE

        points = [{"ua": 100, "ia": 5, "ug1": -2, "ug2": 250, "uh": 6.3}]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "app.import_controller.save_imported_measurement",
                lambda *a, **kw: None,
            )
            ctrl._finalize_import(dlg, points, "test", "test", "/tmp/test.utd", "test")

        compare_tab.add_imported_entries.assert_called_once()
        entry = compare_tab.add_imported_entries.call_args[0][0][0]
        assert entry["lamp_type"] == "6L6"
        assert entry["lamp_id"] == "L1"
        assert len(entry["points"]) == 1
        tabs.setCurrentWidget.assert_called_once_with(compare_tab)


    def test_finalize_save_error_still_adds_entry(self):
        """If save_imported_measurement raises, entry should still be added to compare_tab."""
        compare_tab = MagicMock()
        tabs = MagicMock()
        parent = MagicMock()
        ctrl = ImportController(
            parent_widget=parent,
            compare_tab=compare_tab,
            tabs=tabs,
            get_lamps=lambda: [],
        )
        dlg = MagicMock()
        dlg.tube_type = "EL34"
        dlg.lamp_id = "L2"
        dlg.name = "broken"
        dlg.description = "desc"
        dlg.ug2_mode = TOPOLOGY_PENTODE

        points = [{"ua": 200, "ia": 10, "ug1": -5, "ug2": 250, "uh": 6.3}]

        def _raise(*a, **kw):
            raise OSError("disk full")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.import_controller.save_imported_measurement", _raise)
            mp.setattr("app.import_controller.QMessageBox", MagicMock())
            ctrl._finalize_import(dlg, points, "test", "test", "/tmp/x.utd", "x")

        # Entry must still be added despite save failure
        compare_tab.add_imported_entries.assert_called_once()
        entry = compare_tab.add_imported_entries.call_args[0][0][0]
        assert entry["lamp_type"] == "EL34"

    def test_finalize_builds_correct_topology(self):
        """Verify topology payload is included in the measurement."""
        compare_tab = MagicMock()
        ctrl = ImportController(
            parent_widget=MagicMock(),
            compare_tab=compare_tab,
            tabs=MagicMock(),
            get_lamps=lambda: [],
        )
        dlg = MagicMock()
        dlg.tube_type = "12AX7"
        dlg.lamp_id = "L1"
        dlg.name = "test"
        dlg.description = ""
        dlg.ug2_mode = TOPOLOGY_TRIODE

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.import_controller.save_imported_measurement", lambda *a, **kw: None)
            ctrl._finalize_import(dlg, [{"ua": 100, "ia": 1}], "test", "test", "/f", "s")

        entry = compare_tab.add_imported_entries.call_args[0][0][0]
        data = entry["data"]
        assert data.get("topology") == TOPOLOGY_TRIODE


class TestImportControllerConstruction:
    def test_construct(self):
        ctrl = ImportController(
            parent_widget=MagicMock(),
            compare_tab=MagicMock(),
            tabs=MagicMock(),
            get_lamps=lambda: [],
        )
        assert ctrl is not None
        assert callable(ctrl.import_utracer)
        assert callable(ctrl.import_csv)
        assert callable(ctrl.import_curvetracedata)
