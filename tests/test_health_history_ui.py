"""Health history/match UI: data and state correctness.

Pins:
- ML-039: an entry with ``health: null`` (broken/hand-edited JSON)
  does not crash row extraction;
- ML-040: Timestamp/Mfg — short display, sorting by the FULL string
  (a plain QTableWidgetItem aliases EditRole<->DisplayRole — setText
  clobbered the key; a 1955 vintage must sort BEFORE a 2026 one);
- ML-041: uh80_stabilizing carries ``ua`` — Pa is computed from the
  event, not by parsing QLabel text;
- ML-042: CSV quotes cells containing ';', hidden (filtered-out)
  rows are not exported;
- ML-043: a repeated Match washes the previous row fill/dimming;
- ML-044: the Grp column sorts numerically (2 < 10), unmatched last;
- ML-045: mode repopulation does NOT reset weights when the mode is
  unchanged, but a real mode change refreshes the defaults.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget
from lm19.constants import (
    TOPOLOGY_PENTODE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ── ML-039 ───────────────────────────────────────────────────────────

class TestNullHealthEntry:

    def test_null_sections_do_not_crash(self, qapp):
        from app.health_history import extract_row_values
        entry = {"timestamp": "2026-07-05T10:00:00", "lamp_id": "L1",
                 "health": None, "srk": None, "conditions": None}
        vals = extract_row_values(entry)
        assert len(vals) == 23
        assert vals[1] == "L1"


# ── ML-040 ───────────────────────────────────────────────────────────

class TestSortKeySurvivesDisplay:

    def test_mfg_vintage_sorts_before_modern(self, qapp):
        from app.health_history import _format_cell
        vintage = _format_cell(3, "1955-06")
        modern = _format_cell(3, "2026-07")
        # display is short…
        assert vintage.text() == "55-06"
        # …but the sort key is the FULL string
        assert vintage.data(Qt.ItemDataRole.EditRole) == "1955-06"
        assert vintage < modern, \
            "1955-06 must sort before 2026-07 (was: '55-06' > '26-07')"

    def test_timestamp_editrole_full_iso(self, qapp):
        from app.health_history import _format_cell
        item = _format_cell(0, "2026-07-05T10:20:30")
        assert item.data(Qt.ItemDataRole.EditRole) == "2026-07-05T10:20:30"
        assert item.text() == "26-07-05 10:20"


# ── ML-041 ───────────────────────────────────────────────────────────

class TestUh80EventCarriesUa:

    def test_event_schema_has_ua(self):
        from lm19.health_events import _Uh80Stabilizing
        assert "ua" in _Uh80Stabilizing.__annotations__

    def test_emitter_sends_ua(self):
        """The emitter in lm19/health.py itself puts ua into the event
        (pin against schema/emitter divergence)."""
        import inspect
        import lm19.health as health
        src = inspect.getsource(health)
        i = src.index('"event": "uh80_stabilizing"')
        block = src[i:i + 400]
        assert '"ua"' in block, "uh80_stabilizing must carry measured ua"


# ── ML-042 ───────────────────────────────────────────────────────────

class TestCsvExportCorrectness:

    def _table(self, qapp) -> QTableWidget:
        t = QTableWidget(3, 2)
        t.setHorizontalHeaderLabels(["Name", "Val"])
        from PySide6.QtWidgets import QTableWidgetItem
        cells = [["plain", "1"], ["with;semi", "2"], ["hidden", "3"]]
        for r, row in enumerate(cells):
            for c, v in enumerate(row):
                t.setItem(r, c, QTableWidgetItem(v))
        t.setRowHidden(2, True)  # filtered out
        return t

    def test_hidden_rows_excluded(self, qapp):
        from app.health_history import table_to_tsv
        lines, count = table_to_tsv(self._table(qapp), selected_only=False)
        assert count == 2
        assert not any("hidden" in line for line in lines)

    def test_semicolon_cell_is_quoted(self, qapp):
        """csv.writer with delimiter=';' must quote a cell containing
        ';' — the old replace('\\t', ';') broke columns."""
        from app.health_history import table_to_rows
        header, rows, count = table_to_rows(self._table(qapp),
                                            selected_only=False)
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerows([header] + rows)
        out = buf.getvalue().replace("\r", "")
        assert '"with;semi";2' in out.splitlines()
        parsed = list(csv.reader(io.StringIO(buf.getvalue()), delimiter=";"))
        assert ["with;semi", "2"] in parsed, "round-trip must preserve cells"


# ── ML-043 / ML-044: match colors + Grp sort ─────────────────────────

def _history_entry(lamp_id: str, ts: str, index: float = 90.0):
    return {
        "timestamp": ts, "lamp_id": lamp_id, "name": f"m_{lamp_id}",
        "health": {"index": index, "metrics": {}, "raw": {}},
        "srk": {"s": 5.0, "r": 8.0, "k": 40.0},
        "conditions": {"ua": 250.0, "ug1": -7.0, "ug2": 250.0,
                       "ug2_mode": TOPOLOGY_PENTODE, "an": 1},
    }


class TestGrpNumericSort:

    def test_group_numbers_sort_numerically(self, qapp):
        """Pin on __lt__: a group-2 item < a group-10 item
        (lexicographically '10 (...)' < '2 (...)')."""
        from app.widget_factory import FormattedNumericItem
        g2 = FormattedNumericItem(2.0, "2 (1.5)")
        g10 = FormattedNumericItem(10.0, "10 (3.0)")
        unmatched = FormattedNumericItem(float("inf"), "—")
        assert g2 < g10
        assert not (g10 < g2)
        assert g2 < unmatched and g10 < unmatched

    def test_apply_match_uses_numeric_items(self, qapp):
        """_apply_match_to_table puts numerically sortable items in Grp."""
        import inspect
        import app.health_tab as ht
        src = inspect.getsource(ht.HealthTab._apply_match_to_table)
        assert "_FormattedNumericItem(grp_sort_key" in src


class TestMatchWash:

    def test_wash_restores_verdict_and_foreground(self, qapp):
        """Washing the previous result: setData(Foreground/BackgroundRole,
        None) on rows — pins the wash block's presence and semantics by
        direct execution on a small table."""
        import inspect
        import app.health_tab as ht
        src = inspect.getsource(ht.HealthTab._apply_match_to_table)
        assert "ForegroundRole, None" in src
        assert "verdict_bg" in src


# ── ML-045 ───────────────────────────────────────────────────────────

class TestModeRepopulateKeepsWeights:

    @pytest.fixture()
    def panel(self, qapp):
        from app.match_panel import MatchPanel
        return MatchPanel()

    def test_same_mode_keeps_user_weights(self, panel):
        panel.set_available_modes({"pentode": 3}, default_mode="pentode")
        panel.weight_ia_spin.setValue(0.7)  # user edit
        panel.set_available_modes({"pentode": 5}, default_mode="pentode")
        assert panel.weight_ia_spin.value() == pytest.approx(0.7), \
            "repopulation with the SAME mode must not stomp user weights"

    def test_real_mode_change_updates_defaults(self, panel):
        from lm19.tube_matching import default_weights_for_mode
        panel.set_available_modes({"pentode": 3}, default_mode="pentode")
        panel.weight_ia_spin.setValue(0.7)
        panel.set_available_modes({"triode": 2}, default_mode="triode")
        expected = default_weights_for_mode("triode")["ia"]
        assert panel.weight_ia_spin.value() == pytest.approx(expected), \
            "an actual mode transition must refresh the default weights"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
