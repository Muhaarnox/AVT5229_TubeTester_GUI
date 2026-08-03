"""Import-parser guards (ML-055 / ML-059 / ML-070 / ML-071).

- ML-055: parse_csv fills the ``stats`` out-dict so the UI can surface
  skipped-row counts (row-drop itself was fixed earlier).
- ML-059: eTracer _strip_nan removes only TRAILING padding; an interior
  NaN drops just that point, not the tail of valid data.
- ML-070: lookup_tube tries the full name before suffix-stripping —
  aliases like "RCA 2A3" must not be mangled to "RCA 2".
- ML-071: parse_utd skips truncated rows instead of fabricating Ia=0.

Run:  py -m pytest tests/test_import_guards.py -v
"""

import math
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.csv_import import parse_csv
from lm19.etracer_import import _strip_nan, parse_etracer_csv
from lm19.utracer_import import parse_utd

_NAN = float("nan")


# ═══════════════════════════════════════════════════════════════════
#  ML-059: eTracer — trailing padding vs interior corruption
# ═══════════════════════════════════════════════════════════════════

class TestStripNan:

    def test_trailing_nan_stripped(self):
        assert _strip_nan([1.0, 2.0, _NAN, _NAN]) == [1.0, 2.0]

    def test_interior_nan_preserved(self):
        """The old implementation broke at the FIRST NaN and returned
        [1.0], losing the valid tail."""
        out = _strip_nan([1.0, _NAN, 2.0, _NAN])
        assert len(out) == 3
        assert out[0] == 1.0 and out[2] == 2.0
        assert math.isnan(out[1])

    def test_all_nan_empty(self):
        assert _strip_nan([_NAN, _NAN]) == []

    def test_no_nan_unchanged(self):
        assert _strip_nan([1.0, 2.0]) == [1.0, 2.0]


def _write_etracer(tmp_path, rows):
    path = tmp_path / "sample.csv"
    lines = ["# ETRACER_CSV_FORMAT_VERSION: 2.0"] + rows
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


class TestEtracerInteriorNan:

    def test_interior_nan_drops_point_not_tail(self, tmp_path):
        """One corrupt hv1_i cell (index 1 of 4) must cost ONE point —
        the old break-at-first-NaN kept only index 0."""
        path = _write_etracer(tmp_path, [
            "1,10,20,30,40",     # HV1_V
            "1,1.0,xx,3.0,4.0",  # HV1_I — corrupt cell at idx 1
            "1,0,0,0,0",         # HV2_V
            "1,0,0,0,0",         # HV2_I
            "1,-1,-1,-1,-1",     # NEGV
            "1,0,0,0,0",         # SWEEP
        ])
        stats = {}
        parsed = parse_etracer_csv(path, stats=stats)
        curve = parsed["curves"][0]
        assert curve["hv1_v"] == [10.0, 30.0, 40.0]
        assert curve["hv1_i"] == [1.0, 3.0, 4.0]
        assert stats["nan_points"] == 1

    def test_trailing_padding_not_counted_as_loss(self, tmp_path):
        path = _write_etracer(tmp_path, [
            "1,10,20,nan,nan",
            "1,1.0,2.0,nan,nan",
            "1,0,0,nan,nan",
            "1,0,0,nan,nan",
            "1,-1,-1,nan,nan",
            "1,0,0,nan,nan",
        ])
        stats = {}
        parsed = parse_etracer_csv(path, stats=stats)
        curve = parsed["curves"][0]
        assert curve["hv1_v"] == [10.0, 20.0]
        assert stats["nan_points"] == 0

    def test_nan_first_sweep_cell_does_not_poison_sweep_src(self, tmp_path):
        path = _write_etracer(tmp_path, [
            "1,10,20,30,40",
            "1,1.0,2.0,3.0,4.0",
            "1,0,0,0,0",
            "1,0,0,0,0",
            "1,-1,-1,-1,-1",
            "1,nan,1,1,1",       # first sweep cell corrupt
        ])
        parsed = parse_etracer_csv(path)
        assert parsed["curves"][0]["sweep_src"] == 1.0


# ═══════════════════════════════════════════════════════════════════
#  ML-071: uTracer — truncated rows are skipped, not zero-filled
# ═══════════════════════════════════════════════════════════════════

def _write_utd(tmp_path, lines):
    path = tmp_path / "sample.utd"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


class TestUtdShortRows:

    def test_short_row_skipped_not_fabricated(self, tmp_path):
        path = _write_utd(tmp_path, [
            "Va Ia",
            "Vg = -1 V   Vg = -2 V",
            "10 1.0 2.0",
            "20 1.5",          # truncated: 1 of 2 columns
            "30 3.0 4.0",
        ])
        stats = {}
        parsed = parse_utd(path, stats=stats)
        assert parsed["x_values"] == [10.0, 30.0]
        assert parsed["ia_matrix"] == [[1.0, 2.0], [3.0, 4.0]]
        assert stats["short_rows"] == 1
        # the old zero-fill fabricated a dead-tube Ia=0.0 in row 20
        assert all(0.0 not in row for row in parsed["ia_matrix"])

    def test_short_row_skipped_pentode_ia_is(self, tmp_path):
        path = _write_utd(tmp_path, [
            "Va Ia Is",
            "Vg = -1 V  Vg = -1 V  Vg = -2 V  Vg = -2 V",
            "10 1.0 0.1 2.0 0.2",
            "20 1.5 0.1",      # truncated: 3 of 4 columns
            "30 3.0 0.3 4.0 0.4",
        ])
        stats = {}
        parsed = parse_utd(path, stats=stats)
        assert parsed["x_values"] == [10.0, 30.0]
        assert parsed["ia_matrix"] == [[1.0, 2.0], [3.0, 4.0]]
        assert parsed["is_matrix"] == [[0.1, 0.2], [0.3, 0.4]]
        assert stats["short_rows"] == 1

    def test_all_rows_short_raises(self, tmp_path):
        path = _write_utd(tmp_path, [
            "Va Ia",
            "Vg = -1 V   Vg = -2 V",
            "10 1.0",
            "20 1.5",
        ])
        with pytest.raises(ValueError):
            parse_utd(path)

    def test_full_file_no_loss(self, tmp_path):
        path = _write_utd(tmp_path, [
            "Va Ia",
            "Vg = -1 V   Vg = -2 V",
            "10 1.0 2.0",
            "20 1.5 2.5",
        ])
        stats = {}
        parsed = parse_utd(path, stats=stats)
        assert stats["short_rows"] == 0
        assert parsed["x_values"] == [10.0, 20.0]


# ═══════════════════════════════════════════════════════════════════
#  ML-055: CSV stats out-dict
# ═══════════════════════════════════════════════════════════════════

class TestCsvStats:

    _MAPPING = {0: "ua", 1: "ug1", 2: "ia"}

    def test_stats_filled_on_bad_row(self):
        text = "ua;ug1;ia\n100;-2;5.0\n200;xx;6.0\n300;-4;7.0\n"
        stats = {}
        points = parse_csv(text, self._MAPPING, ";", stats=stats)
        assert len(points) == 2
        assert stats["skipped_rows"] == 1
        assert stats["bad_cells"] == 1

    def test_stats_zero_on_clean_file(self):
        text = "ua;ug1;ia\n100;-2;5.0\n"
        stats = {}
        points = parse_csv(text, self._MAPPING, ";", stats=stats)
        assert len(points) == 1
        assert stats["skipped_rows"] == 0
        assert stats["bad_cells"] == 0


# ═══════════════════════════════════════════════════════════════════
#  ML-070: full-name lookup before suffix stripping
# ═══════════════════════════════════════════════════════════════════

class TestLookupFullNameFirst:

    def test_alias_ending_in_panel_pattern_resolves(self, monkeypatch):
        """'RCA 2A3' matches the panel-suffix regex (base 'RCA 2') — the
        as-is lookup must win before any stripping."""
        import lm19.tube_params as tp
        tp._ensure_loaded()
        canonical = tp._lookup.get("2A3")
        assert canonical, "precondition: 2A3 present in tube DB"
        monkeypatch.setitem(tp._lookup, "RCA 2A3", canonical)
        ref = tp.lookup_tube("RCA 2A3")
        assert ref is not None
        assert ref is tp.lookup_tube("2A3")

    def test_suffixed_lm19_name_still_resolves(self):
        """Regression guard: real LM19 panel-suffixed names keep working
        through the stripping path."""
        import lm19.tube_params as tp
        ref = tp.lookup_tube("EL84__B02")
        assert ref is not None
        assert ref is tp.lookup_tube("EL84")


# ═══════════════════════════════════════════════════════════════════
#  UI channel: ImportController._warn_skipped
# ═══════════════════════════════════════════════════════════════════

class TestWarnSkipped:

    def _controller(self):
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from app.import_controller import ImportController
        return ImportController(
            parent_widget=None, compare_tab=None, tabs=None,
            get_lamps=lambda: [])

    def test_zero_is_silent(self, monkeypatch):
        shown = []
        monkeypatch.setattr(
            "app.import_controller.QMessageBox.warning",
            lambda *a, **k: shown.append(a))
        self._controller()._warn_skipped(0)
        assert not shown

    def test_positive_count_shows_dialog(self, monkeypatch):
        shown = []
        monkeypatch.setattr(
            "app.import_controller.QMessageBox.warning",
            lambda *a, **k: shown.append(a))
        self._controller()._warn_skipped(3)
        assert len(shown) == 1
        assert "3" in shown[0][2]


class TestFlowSurfacesLoss:
    """Call-site pins: each import flow must route its loss counter into
    _warn_skipped (the helper alone doesn't prove the wiring)."""

    def _controller(self, recorded):
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from app.import_controller import ImportController
        ctrl = ImportController(
            parent_widget=None, compare_tab=None, tabs=None,
            get_lamps=lambda: [])
        ctrl._warn_skipped = lambda count: recorded.append(count)
        return ctrl

    @staticmethod
    def _rejected_meta_dialog(monkeypatch):
        """Stop each flow right after the warn point."""
        from unittest.mock import MagicMock
        meta_cls = MagicMock()
        meta_cls.DialogCode.Accepted = 1
        meta_cls.return_value.exec.return_value = 0   # Rejected
        monkeypatch.setattr("app.import_controller.ImportMetaDialog", meta_cls)

    def test_import_utracer_surfaces_short_rows(self, monkeypatch, tmp_path):
        path = _write_utd(tmp_path, [
            "Va Ia",
            "Vg = -1 V   Vg = -2 V",
            "10 1.0 2.0",
            "20 1.5",
            "30 3.0 4.0",
        ])
        monkeypatch.setattr(
            "app.import_controller.QFileDialog.getOpenFileName",
            lambda *a, **k: (path, ""))
        self._rejected_meta_dialog(monkeypatch)
        recorded = []
        self._controller(recorded).import_utracer()
        assert recorded == [1]

    def test_import_csv_surfaces_skipped_rows(self, monkeypatch, tmp_path):
        from unittest.mock import MagicMock
        f = tmp_path / "d.csv"
        f.write_text("ua;ug1;ia\n100;-2;5.0\n200;xx;6.0\n", encoding="utf-8")
        monkeypatch.setattr(
            "app.import_controller.QFileDialog.getOpenFileName",
            lambda *a, **k: (str(f), ""))
        csv_cls = MagicMock()
        csv_cls.DialogCode.Accepted = 1
        csv_dlg = csv_cls.return_value
        csv_dlg.exec.return_value = 1
        csv_dlg.column_mapping = {0: "ua", 1: "ug1", 2: "ia"}
        csv_dlg.separator = ";"
        monkeypatch.setattr("app.import_controller.CsvImportDialog", csv_cls)
        self._rejected_meta_dialog(monkeypatch)
        recorded = []
        self._controller(recorded).import_csv()
        assert recorded == [1]

    def test_import_etracer_surfaces_nan_points(self, monkeypatch, tmp_path):
        path = _write_etracer(tmp_path, [
            "1,10,20,30,40",
            "1,1.0,xx,3.0,4.0",
            "1,0,0,0,0",
            "1,0,0,0,0",
            "1,-1,-1,-1,-1",
            "1,0,0,0,0",
        ])
        monkeypatch.setattr(
            "app.import_controller.QFileDialog.getOpenFileName",
            lambda *a, **k: (path, ""))
        self._rejected_meta_dialog(monkeypatch)
        recorded = []
        self._controller(recorded).import_etracer()
        assert recorded == [1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
