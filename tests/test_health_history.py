"""Tests for app.health_history module — includes pytest-qt widget tests."""

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
from PySide6.QtGui import QColor

from app.health_history import (
    entry_matches_filter,
    verdict_color,
    table_to_tsv,
    extract_row_values,
    populate_history_table,
    build_match_entry_info,
    build_match_active,
    build_matching_conditions,
    format_match_groups_text,
    format_match_csv_rows,
    _format_timestamp_short,
    _format_mfg_short,
    COL_SEL,
    COL_GRP,
)
from lm19.tube_matching import MatchResult, MatchGroup, TubeRecord
from app.ui_theme import (
    HEALTH_VERDICT_STRONG_BG,
    HEALTH_VERDICT_GOOD_BG,
    HEALTH_VERDICT_WEAK_BG,
    HEALTH_VERDICT_REPLACE_BG,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)


# ---------------------------------------------------------------------------
# verdict_color
# ---------------------------------------------------------------------------

class TestVerdictColor:
    def test_strong(self):
        c = verdict_color(90, strong_min=85, good_min=65, weak_min=40)
        assert c == HEALTH_VERDICT_STRONG_BG

    def test_good(self):
        c = verdict_color(70, strong_min=85, good_min=65, weak_min=40)
        assert c == HEALTH_VERDICT_GOOD_BG

    def test_weak(self):
        c = verdict_color(50, strong_min=85, good_min=65, weak_min=40)
        assert c == HEALTH_VERDICT_WEAK_BG

    def test_replace(self):
        c = verdict_color(20, strong_min=85, good_min=65, weak_min=40)
        assert c == HEALTH_VERDICT_REPLACE_BG

    def test_none_returns_none(self):
        assert verdict_color(None, 85, 65, 40) is None

    def test_string_returns_none(self):
        assert verdict_color("N/A", 85, 65, 40) is None

    def test_boundary_strong(self):
        c = verdict_color(85, strong_min=85, good_min=65, weak_min=40)
        assert c == HEALTH_VERDICT_STRONG_BG

    def test_boundary_good(self):
        c = verdict_color(65, strong_min=85, good_min=65, weak_min=40)
        assert c == HEALTH_VERDICT_GOOD_BG


# ---------------------------------------------------------------------------
# table_to_tsv (requires qtbot for QTableWidget)
# ---------------------------------------------------------------------------

class TestTableToTsv:
    def _make_table(self, qtbot, rows, cols, headers=None, data=None):
        table = QTableWidget(rows, cols)
        qtbot.addWidget(table)
        if headers:
            table.setHorizontalHeaderLabels(headers)
        if data:
            for r, row_data in enumerate(data):
                for c, val in enumerate(row_data):
                    table.setItem(r, c, QTableWidgetItem(str(val)))
        return table

    def test_empty_table(self, qtbot):
        table = self._make_table(qtbot, 0, 3, ["A", "B", "C"])
        lines, count = table_to_tsv(table)
        assert lines == []
        assert count == 0

    def test_basic(self, qtbot):
        table = self._make_table(
            qtbot, 2, 3,
            headers=["Type", "ID", "Index"],
            data=[
                ["6L6", "L1", "85"],
                ["EL34", "L2", "72"],
            ],
        )
        lines, count = table_to_tsv(table)
        assert count == 2
        assert len(lines) == 3  # header + 2 data rows
        assert lines[0] == "Type\tID\tIndex"
        assert lines[1] == "6L6\tL1\t85"
        assert lines[2] == "EL34\tL2\t72"

    def test_selected_only_no_selection(self, qtbot):
        table = self._make_table(
            qtbot, 2, 2,
            headers=["A", "B"],
            data=[["1", "2"], ["3", "4"]],
        )
        lines, count = table_to_tsv(table, selected_only=True)
        assert lines == []
        assert count == 0

    def test_missing_items(self, qtbot):
        """Cells without QTableWidgetItem should produce empty string."""
        table = QTableWidget(1, 2)
        qtbot.addWidget(table)
        table.setHorizontalHeaderLabels(["A", "B"])
        table.setItem(0, 0, QTableWidgetItem("val"))
        # col 1 has no item
        lines, count = table_to_tsv(table)
        assert count == 1
        assert lines[1] == "val\t"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# extract_row_values (pure logic, no Qt needed)
# ---------------------------------------------------------------------------

class TestExtractRowValues:
    def test_full_entry(self):
        entry = {
            "timestamp": "2024-01-15T10:00:00",
            "mfg_date": "1972-05",
            "lamp_id": "L1",
            "name": "test",
            "conditions": {"an": 1, "ug2_mode": TOPOLOGY_PENTODE,
                            "ua": 250.0, "ug1": -7.3, "ug2": 250.0},
            "health": {
                "index": 82,
                "metrics": {"ia_pct": 95, "s_pct": 88, "r_pct": 75, "k_pct": None,
                             "emission_ratio": 0.95},
                "raw": {"ia_op": 48.5},
                "reference_mode": "type",
            },
            "srk": {"s": 8.2, "r": 22.5, "k": None, "mu_g1g2": 4.1},
        }
        vals = extract_row_values(entry)
        # Column order: Timestamp, Lamp ID, Name, Mfg, An, Mode,
        #               Ua, Ug1, Ug2, Δbias, Index, Ia%, S%, R%, K%, Ia, S, R, K, µg2, Em, Ref
        assert len(vals) == 23
        assert vals[0] == "2024-01-15T10:00:00"
        assert vals[1] == "L1"
        assert vals[2] == "test"
        assert vals[3] == "1972-05"  # mfg_date
        assert vals[4] == 1  # an
        assert vals[6] == 250.0  # Ua
        assert vals[7] == -7.3   # Ug1
        assert vals[8] == 250.0  # Ug2
        assert vals[9] is None  # no servo -> empty Δbias
        assert vals[10] == 82  # index
        assert vals[16] == 8.2  # s
        assert vals[22] == "type"  # reference_mode

    def test_empty_entry(self):
        vals = extract_row_values({})
        assert len(vals) == 23
        assert vals[0] == ""  # timestamp
        assert vals[3] == ""  # mfg_date
        assert vals[6] is None  # Ua absent
        assert vals[7] is None  # Ug1 absent
        assert vals[8] is None  # Ug2 absent
        assert vals[10] is None  # index
        assert vals[22] == ""  # reference_mode

    def test_missing_optional_fields(self):
        entry = {
            "timestamp": "2024-01-01",
            "lamp_id": "L2",
            "name": "",
            "conditions": {"ug2_mode": TOPOLOGY_TRIODE},
            "health": {},
            "srk": {},
        }
        vals = extract_row_values(entry)
        assert vals[3] == ""  # mfg_date absent -> ""
        assert vals[4] == ""  # an is None -> ""
        assert vals[6] is None  # Ua absent -> None
        assert vals[10] is None  # no index

    def test_mfg_date_absent_returns_empty(self):
        entry = {"timestamp": "2024-01-01", "lamp_id": "L3"}
        vals = extract_row_values(entry)
        assert vals[3] == ""


# ---------------------------------------------------------------------------
# populate_history_table (requires qtbot)
# ---------------------------------------------------------------------------

class TestPopulateHistoryTable:
    def test_populates_rows(self, qtbot):
        table = QTableWidget(0, 23)
        qtbot.addWidget(table)
        entries = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "lamp_id": "L1",
                "name": "scan",
                "conditions": {"an": 1, "ug2_mode": TOPOLOGY_PENTODE},
                "health": {"index": 85, "metrics": {}, "raw": {}, "reference_mode": ""},
                "srk": {},
            },
            {
                "timestamp": "2024-01-02T10:00:00",
                "lamp_id": "L2",
                "name": "test",
                "conditions": {"an": 2, "ug2_mode": TOPOLOGY_TRIODE},
                "health": {"index": 50, "metrics": {}, "raw": {}, "reference_mode": ""},
                "srk": {},
            },
        ]
        populate_history_table(table, entries)
        assert table.rowCount() == 2
        # First column stores entry as UserRole
        item0 = table.item(0, 0)
        assert item0 is not None
        stored = item0.data(Qt.ItemDataRole.UserRole)
        assert isinstance(stored, dict)

    def test_empty_entries(self, qtbot):
        table = QTableWidget(0, 23)
        qtbot.addWidget(table)
        populate_history_table(table, [])
        assert table.rowCount() == 0

    def test_color_function_applied(self, qtbot):
        table = QTableWidget(0, 23)
        qtbot.addWidget(table)
        green = QColor(0, 255, 0)
        entries = [{
            "timestamp": "t1", "lamp_id": "L1", "name": "",
            "conditions": {}, "health": {"index": 90, "metrics": {}, "raw": {}},
            "srk": {},
        }]
        populate_history_table(table, entries, color_fn=lambda _: green)
        item = table.item(0, 1)
        assert item.background().color() == green

    def test_sel_grp_columns_empty(self, qtbot):
        table = QTableWidget(0, 25)
        qtbot.addWidget(table)
        entries = [{
            "timestamp": "t1", "lamp_id": "L1", "name": "",
            "conditions": {}, "health": {"metrics": {}, "raw": {}},
            "srk": {},
        }]
        populate_history_table(table, entries)
        sel_item = table.item(0, COL_SEL)
        grp_item = table.item(0, COL_GRP)
        assert sel_item is not None
        assert sel_item.text() == ""
        assert grp_item is not None


# ---------------------------------------------------------------------------
# format_match_groups_text / format_match_csv_rows (pure logic)
# ---------------------------------------------------------------------------

def _rec(lamp_id, ia=50.0, s=8.0, r=20.0, ts="t1"):
    return TubeRecord(lamp_id=lamp_id, timestamp=ts, an=1, ia=ia, s=s, r=r)


# ---------------------------------------------------------------------------
# build_match_entry_info
# ---------------------------------------------------------------------------

class TestBuildMatchEntryInfo:
    def test_empty(self):
        result = MatchResult(mode="groups", groups=[], unmatched=[])
        assert build_match_entry_info(result) == {}

    def test_groups_and_unmatched(self):
        g = MatchGroup(number=1, records=[_rec("L1", ts="t1"), _rec("L2", ts="t2")], delta=3.0)
        result = MatchResult(mode="groups", groups=[g], unmatched=[_rec("L3", ts="t3")])
        info = build_match_entry_info(result)
        assert info[("L1", "t1")] == (1, 3.0)
        assert info[("L2", "t2")] == (1, 3.0)
        assert info[("L3", "t3")] == (0, 0.0)  # unmatched


# ---------------------------------------------------------------------------
# build_match_active
# ---------------------------------------------------------------------------

class TestBuildMatchActive:
    def test_empty(self):
        result = MatchResult(mode="groups", groups=[], unmatched=[])
        assert build_match_active(result) == set()

    def test_collects_all(self):
        g = MatchGroup(number=1, records=[_rec("L1", ts="t1"), _rec("L2", ts="t2")], delta=2.0)
        result = MatchResult(mode="groups", groups=[g], unmatched=[_rec("L3", ts="t3")])
        active = build_match_active(result)
        assert active == {("L1", "t1"), ("L2", "t2"), ("L3", "t3")}


# ---------------------------------------------------------------------------
# build_matching_conditions
# ---------------------------------------------------------------------------

class TestBuildMatchingConditions:
    def _entry(self, ua=250.0, ug1=-8.0, ug2=250.0, mode="pentode"):
        return {"conditions": {"ua": ua, "ug1": ug1, "ug2": ug2, "ug2_mode": mode}}

    def test_basic(self):
        entries = [self._entry(ua=250, ug1=-8, ug2=250)]
        result = build_matching_conditions(entries, "pentode")
        assert result == (250.0, -8.0, 250.0, "pentode", False)

    def test_no_matching_mode(self):
        entries = [self._entry(mode="triode")]
        assert build_matching_conditions(entries, "pentode") is None

    def test_rounds_values(self):
        entries = [self._entry(ua=250.123, ug1=-8.456, ug2=249.789)]
        result = build_matching_conditions(entries, "pentode")
        assert result == (250.1, -8.5, 249.8, "pentode", False)


class TestFormatMatchGroupsText:
    def test_empty_result(self):
        result = MatchResult(mode="groups", groups=[], unmatched=[])
        lines = format_match_groups_text(result)
        assert lines == []

    def test_one_group(self):
        g = MatchGroup(number=1, records=[_rec("L1"), _rec("L2")], delta=3.5)
        result = MatchResult(mode="groups", groups=[g], unmatched=[])
        lines = format_match_groups_text(result, group_label="Grp")
        assert len(lines) == 1
        assert "Grp 1" in lines[0]
        assert "Δ=3.5%" in lines[0]
        assert "L1" in lines[0]
        assert "L2" in lines[0]

    def test_groups_and_unmatched(self):
        g = MatchGroup(number=1, records=[_rec("L1"), _rec("L2")], delta=2.0)
        result = MatchResult(mode="groups", groups=[g], unmatched=[_rec("L3")])
        lines = format_match_groups_text(result, unmatched_label="Unm")
        assert len(lines) == 2
        assert "Unm: L3" in lines[1]

    def test_multiple_groups(self):
        g1 = MatchGroup(number=1, records=[_rec("L1"), _rec("L2")], delta=1.0)
        g2 = MatchGroup(number=2, records=[_rec("L3"), _rec("L4")], delta=4.5)
        result = MatchResult(mode="groups", groups=[g1, g2], unmatched=[])
        lines = format_match_groups_text(result)
        assert len(lines) == 2

    def test_shared_group_carries_iq_imbalance(self):
        g = MatchGroup(number=1, records=[_rec("L1"), _rec("L2")], delta=2.0,
                       iq_imbalance_ma=4.5)
        result = MatchResult(mode="groups", groups=[g], unmatched=[])
        lines = format_match_groups_text(result)
        assert "δIq≈4.5 mA" in lines[0]

    def test_strict_group_has_no_iq_text(self):
        g = MatchGroup(number=1, records=[_rec("L1"), _rec("L2")], delta=2.0)
        result = MatchResult(mode="groups", groups=[g], unmatched=[])
        assert "δIq" not in format_match_groups_text(result)[0]


class TestFormatMatchCsvRows:
    def test_empty_result(self):
        result = MatchResult(mode="groups", groups=[], unmatched=[])
        rows = format_match_csv_rows(result)
        assert rows == []

    def test_one_group(self):
        g = MatchGroup(number=1, records=[_rec("L1", ia=50.12, s=8.123, r=20.5)], delta=3.5)
        result = MatchResult(mode="groups", groups=[g], unmatched=[])
        rows = format_match_csv_rows(result)
        # Columns: Group, Lamp ID, Mfg, Ia, S, R, Δ
        assert len(rows) == 1
        assert rows[0][0] == "1"  # group number
        assert rows[0][1] == "L1"
        assert rows[0][2] == ""    # mfg empty by default
        assert rows[0][3] == "50.12"  # ia
        assert rows[0][4] == "8.123"  # s
        assert rows[0][6] == "3.5"  # delta

    def test_unmatched_uses_dash(self):
        result = MatchResult(mode="groups", groups=[], unmatched=[_rec("L5")])
        rows = format_match_csv_rows(result)
        assert len(rows) == 1
        assert rows[0][0] == "—"
        assert rows[0][6] == "—"

    def test_mfg_pulled_from_entry(self):
        rec = _rec("L1")
        rec.entry = {"mfg_date": "1972-05"}
        g = MatchGroup(number=1, records=[rec], delta=1.0)
        result = MatchResult(mode="groups", groups=[g], unmatched=[])
        rows = format_match_csv_rows(result)
        assert rows[0][2] == "1972-05"

    def test_mixed(self):
        g = MatchGroup(number=1, records=[_rec("L1"), _rec("L2")], delta=2.0)
        result = MatchResult(mode="groups", groups=[g], unmatched=[_rec("L3")])
        rows = format_match_csv_rows(result)
        assert len(rows) == 3  # 2 in group + 1 unmatched

    def test_iq_column_stable_schema(self):
        # The δIq column exists in EVERY row: a value for shared-protocol
        # groups, an em-dash for strict groups and unmatched tubes.
        g_shared = MatchGroup(number=1, records=[_rec("L1")], delta=2.0,
                              iq_imbalance_ma=3.2)
        g_strict = MatchGroup(number=2, records=[_rec("L2")], delta=2.0)
        result = MatchResult(mode="groups", groups=[g_shared, g_strict],
                             unmatched=[_rec("L3")])
        rows = format_match_csv_rows(result)
        assert [len(r) for r in rows] == [8, 8, 8]
        assert rows[0][7] == "3.2"
        assert rows[1][7] == "—"
        assert rows[2][7] == "—"

    def test_row_width_matches_export_header(self):
        # The header lives at the call site (_export_match_csv) — a column
        # added to only one side silently misaligns the CSV.
        import ast
        import inspect
        import textwrap
        from app.health_tab import HealthTab
        src = textwrap.dedent(inspect.getsource(HealthTab._export_match_csv))
        header_len = None
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "writerow"):
                header_len = len(node.args[0].elts)
        assert header_len is not None
        g = MatchGroup(number=1, records=[_rec("L1")], delta=2.0)
        result = MatchResult(mode="groups", groups=[g], unmatched=[])
        assert len(format_match_csv_rows(result)[0]) == header_len


class TestNumericFormatting:
    """Pin: trailing zeros must survive ``setData(EditRole, float)``.

    Qt's ``QTableWidgetItem.text()`` coerces DisplayRole from EditRole
    when EditRole is numeric, stripping ``"-7.0"`` to ``"-7"`` and
    ``"0.950"`` to ``"0.95"``. ``_format_cell`` works around this by
    calling ``setText`` BEFORE ``setData(EditRole, …)`` — these tests
    catch any regression in that ordering.
    """

    def test_ua_integer_format(self, qtbot):
        from app.health_history import _format_cell
        # col 6 = Ua, integer V
        assert _format_cell(6, 250.0).text() == "250"
        assert _format_cell(6, 285.0).text() == "285"

    def test_ug1_tenths_with_trailing_zero(self, qtbot):
        from app.health_history import _format_cell
        # col 7 = Ug1, one decimal V — trailing zero must show
        assert _format_cell(7, -7.0).text() == "-7.0"
        assert _format_cell(7, -7.3).text() == "-7.3"
        assert _format_cell(7, -2.2).text() == "-2.2"

    def test_ug2_integer_format(self, qtbot):
        from app.health_history import _format_cell
        # col 8 = Ug2, integer V
        assert _format_cell(8, 250.0).text() == "250"
        assert _format_cell(8, 175.0).text() == "175"

    def test_emission_three_decimals(self, qtbot):
        from app.health_history import _format_cell
        # col 20 = emission_ratio, 3 decimals — trailing zero must survive
        assert _format_cell(20, 0.95).text() == "0.950"
        assert _format_cell(20, 1.0).text() == "1.000"

    def test_edit_role_remains_float_for_sorting(self, qtbot):
        from app.health_history import _format_cell
        # Sort key (EditRole) must be the original float — not the
        # display string — so external consumers (CSV, copy-paste,
        # inline edit) see a number.
        item = _format_cell(7, -7.0)
        assert item.data(Qt.ItemDataRole.EditRole) == -7.0

    def test_numeric_sort_more_negative_first(self, qtbot):
        """Sort must be numeric, not lexical: −10.0 < −7.3 < −7.0."""
        from app.health_history import _format_cell
        a = _format_cell(7, -10.0)
        b = _format_cell(7, -7.3)
        c = _format_cell(7, -7.0)
        assert a < b  # lex would give False (-10 vs -7)
        assert b < c
        assert a < c

    def test_numeric_sort_mixed_width(self, qtbot):
        """Ua sort: 99 < 100 (numeric), not lex '100' < '99'."""
        from app.health_history import _format_cell
        a = _format_cell(6, 99.0)
        b = _format_cell(6, 100.0)
        assert a < b  # numeric: yes; lex: '99' > '100' (no)


class TestHighlightedColumns:
    """Index (col 10) and Ia (col 15) are styled as the headline result
    cells: bold font + distinct background. The verdict-row tint must
    NOT override their background (see ``populate_history_table``)."""

    def test_index_cell_is_bold(self, qtbot):
        from app.health_history import _format_cell
        item = _format_cell(10, 82.0)
        assert item.font().bold(), "Index column must use bold font"

    def test_ia_cell_is_bold(self, qtbot):
        from app.health_history import _format_cell
        item = _format_cell(15, 48.5)
        assert item.font().bold(), "Ia column must use bold font"

    def test_index_cell_has_highlight_background(self, qtbot):
        from app.health_history import _format_cell
        from app.ui_theme import HEALTH_HISTORY_HIGHLIGHT_BG
        item = _format_cell(10, 82.0)
        assert item.background().color() == HEALTH_HISTORY_HIGHLIGHT_BG

    def test_non_highlight_cell_is_not_bold(self, qtbot):
        from app.health_history import _format_cell
        # Ug1 (col 7) — not in HEALTH_HISTORY_HIGHLIGHT_COLS
        item = _format_cell(7, -7.0)
        assert not item.font().bold()

    def test_populate_restores_resize_to_contents(self, qtbot):
        """Pin: populate must leave auto-size columns in ResizeToContents.

        ``populate_history_table`` temporarily switches them to Interactive
        for the bulk insert (avoids O(N²) column-width recalc). If the
        restore in the ``finally`` block ever breaks, columns would stay
        Interactive and stop auto-fitting their content.
        """
        from PySide6.QtWidgets import QHeaderView
        from app.health_history import populate_history_table
        table = QTableWidget(0, 23)
        qtbot.addWidget(table)
        hh = table.horizontalHeader()
        # Match the production setup: most cols ResizeToContents, a few Interactive.
        for c in range(23):
            hh.setSectionResizeMode(
                c,
                QHeaderView.ResizeMode.Interactive if c in (1, 2, 20)
                else QHeaderView.ResizeMode.ResizeToContents,
            )
        populate_history_table(table, [{
            "timestamp": "2026-05-13T10:00:00", "lamp_id": "L1", "name": "",
            "conditions": {}, "health": {"metrics": {}, "raw": {}}, "srk": {},
        }])
        # All non-text columns must be back to ResizeToContents.
        for c in range(23):
            expected = (QHeaderView.ResizeMode.Interactive if c in (1, 2, 20)
                        else QHeaderView.ResizeMode.ResizeToContents)
            assert hh.sectionResizeMode(c) == expected, (
                f"col {c} resize mode {hh.sectionResizeMode(c)} != {expected}"
            )

    def test_verdict_row_color_skips_highlight_cells(self, qtbot):
        """Verdict tint colours the whole row except Index/Ia."""
        from app.health_history import populate_history_table
        from app.ui_theme import (
            HEALTH_VERDICT_GOOD_BG, HEALTH_HISTORY_HIGHLIGHT_BG,
        )
        from PySide6.QtGui import QColor
        table = QTableWidget(0, 23)
        qtbot.addWidget(table)
        entries = [{
            "timestamp": "2026-05-13T10:00:00", "lamp_id": "L1", "name": "",
            "conditions": {"an": 1, "ug2_mode": TOPOLOGY_PENTODE,
                            "ua": 250.0, "ug1": -7.0, "ug2": 250.0},
            "health": {
                "index": 82,
                "metrics": {"ia_pct": 90, "s_pct": 85, "r_pct": 80, "k_pct": 80},
                "raw": {"ia_op": 48.0},
            },
            "srk": {"s": 8.0, "r": 22.0, "k": 18.0},
        }]
        populate_history_table(table, entries, color_fn=lambda _: HEALTH_VERDICT_GOOD_BG)
        # A "normal" result column gets the verdict tint
        assert table.item(0, 11).background().color() == HEALTH_VERDICT_GOOD_BG  # Ia%
        # Highlight columns keep their own background
        assert table.item(0, 10).background().color() == HEALTH_HISTORY_HIGHLIGHT_BG   # Index
        assert table.item(0, 15).background().color() == HEALTH_HISTORY_HIGHLIGHT_BG  # Ia


class TestDateFormatters:
    """Display-side formatters for Timestamp / Mfg in history table."""

    def test_timestamp_iso_full(self):
        assert _format_timestamp_short("2026-05-13T12:34:56") == "26-05-13 12:34"

    def test_timestamp_iso_with_space(self):
        # Legacy format some entries may carry
        assert _format_timestamp_short("2026-05-13 12:34:56") == "26-05-13 12:34"

    def test_timestamp_date_only(self):
        # No time component → returned as-is sans century
        assert _format_timestamp_short("2024-01-15") == "24-01-15"

    def test_timestamp_no_seconds(self):
        # Already minute-precision → drop century only, keep length
        assert _format_timestamp_short("2026-05-13T12:34") == "26-05-13 12:34"

    def test_timestamp_unexpected_format_passthrough(self):
        # Not 4-digit year prefix → returned unchanged (graceful degradation)
        assert _format_timestamp_short("not-a-date") == "not-a-date"
        assert _format_timestamp_short("") == ""

    def test_mfg_full_year(self):
        assert _format_mfg_short("2026-05") == "26-05"
        assert _format_mfg_short("1972-08") == "72-08"  # vintage tube

    def test_mfg_passthrough_when_no_year(self):
        assert _format_mfg_short("") == ""
        assert _format_mfg_short("XX-05") == "XX-05"


class TestConstants:
    def test_col_sel(self):
        # Base columns 0..22: Timestamp, LampID, Name, Mfg, An, Mode,
        # Ua, Ug1, Ug2, Δbias, Index, Ia%, S%, R%, K%, Ia, S, R, K, µg2, Em, Reserve, Ref.
        assert COL_SEL == 23

    def test_col_grp(self):
        assert COL_GRP == 24


class TestDbiasColumn:
    """Col 9: the bias-servo shift. The Ug1 condition column shows the
    PLAN bias (the matching protocol), so this column is the only
    table-level marker that the row was measured at a shifted point."""

    def _servo_entry(self):
        return {
            "timestamp": "2026-08-02T10:00:00", "lamp_id": "L1", "name": "s",
            "conditions": {"ua": 250.0, "ug1": -7.3, "ug2": 250.0,
                           "bias_servo": True},
            "health": {"index": 90.0,
                       "metrics": {"bias_shift_v": 1.6},
                       "raw": {"ia_op": 46.9},
                       "bias_servo": {"status": "ok", "ug1": -5.7}},
            "srk": {},
        }

    def test_extract_places_shift_at_col_9(self):
        vals = extract_row_values(self._servo_entry())
        assert vals[9] == 1.6

    def test_plain_run_has_empty_dbias(self):
        vals = extract_row_values({
            "timestamp": "t", "lamp_id": "L", "name": "",
            "conditions": {}, "health": {"metrics": {}}, "srk": {},
        })
        assert vals[9] is None

    def test_format_carries_explicit_sign(self, qtbot):
        from app.health_history import _format_cell, COL_DBIAS
        assert _format_cell(COL_DBIAS, 1.6).text() == "+1.60"
        assert _format_cell(COL_DBIAS, -0.85).text() == "-0.85"

    def test_populated_cell_has_actual_ug1_tooltip(self, qtbot):
        from app.health_history import COL_DBIAS
        table = QTableWidget(0, 25)
        qtbot.addWidget(table)
        populate_history_table(table, [self._servo_entry()])
        cell = table.item(0, COL_DBIAS)
        assert cell.text() == "+1.60"
        assert "-5.70" in cell.toolTip(), "hover must reveal the actual Ug1"

    def test_plain_cell_has_no_tooltip(self, qtbot):
        from app.health_history import COL_DBIAS
        table = QTableWidget(0, 25)
        qtbot.addWidget(table)
        populate_history_table(table, [{
            "timestamp": "t", "lamp_id": "L", "name": "",
            "conditions": {}, "health": {"metrics": {}}, "srk": {},
        }])
        cell = table.item(0, COL_DBIAS)
        assert cell.text() == "—"
        assert cell.toolTip() == ""


class TestReserveColumn:
    """Col 21: cathode reserve from the emission sweep. A bare
    percentage does not say whether it is good or bad, so the cell
    must carry the interpretation tooltip."""

    def _sweep_entry(self, reserve=15.0, below_range=False):
        return {
            "timestamp": "2026-08-02T12:00:00", "lamp_id": "L1", "name": "s",
            "conditions": {"ua": 250.0, "ug1": -7.3, "ug2": 250.0},
            "health": {"index": 85.0,
                       "metrics": {"emission_ratio": 0.7,
                                   "emission_reserve_pct": reserve},
                       "raw": {"ia_op": 46.9},
                       "emission_sweep": {"knee_below_range": below_range}},
            "srk": {},
        }

    def test_extract_places_reserve_at_col_21(self):
        vals = extract_row_values(self._sweep_entry(reserve=15.0))
        assert vals[21] == 15.0

    def test_single_point_run_has_empty_reserve(self):
        vals = extract_row_values({
            "timestamp": "t", "lamp_id": "L", "name": "",
            "conditions": {}, "health": {"metrics": {}}, "srk": {},
        })
        assert vals[21] is None

    def test_cell_carries_the_interpretation_tooltip(self, qtbot):
        from app.health_history import COL_RESERVE
        table = QTableWidget(0, 25)
        qtbot.addWidget(table)
        populate_history_table(table, [self._sweep_entry()])
        cell = table.item(0, COL_RESERVE)
        assert cell.text() == "15"
        tip = cell.toolTip()
        assert tip, "reserve without interpretation is a bare number"
        # Miram-method bands: floor / healthy / watch boundaries.
        assert "50" in tip and "30" in tip and "15" in tip, \
            "bands must be in the hint"

    def test_below_range_shows_a_lower_bound(self, qtbot):
        from app.health_history import COL_RESERVE
        table = QTableWidget(0, 25)
        qtbot.addWidget(table)
        populate_history_table(table, [self._sweep_entry(reserve=30.0,
                                                         below_range=True)])
        cell = table.item(0, COL_RESERVE)
        assert cell.text() == "≥30"
        # Numeric sort key survives the decorated display.
        assert cell.data(Qt.ItemDataRole.EditRole) == 30.0
