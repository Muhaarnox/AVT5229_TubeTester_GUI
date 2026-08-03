"""Health history table widget — owns the QTableWidget and filter controls."""

import re
from typing import Callable, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lm19.health import (
    HEALTH_VERDICT_GOOD, HEALTH_VERDICT_REPLACE, HEALTH_VERDICT_STRONG,
    HEALTH_VERDICT_WEAK,
)
from app.ui_theme import (
    HEALTH_MIN_SECTION_SIZE,
    HEALTH_VERDICT_STRONG_BG,
    HEALTH_VERDICT_GOOD_BG,
    HEALTH_VERDICT_WEAK_BG,
    HEALTH_VERDICT_REPLACE_BG,
    HEALTH_HISTORY_HIGHLIGHT_BG,
    HEALTH_HISTORY_HIGHLIGHT_COLS,
)
from app.widget_factory import FormattedNumericItem as _FormattedNumericItem
from app.widget_factory import FormattedTextItem as _FormattedTextItem
from i18n_setup import t
from lm19.health_measurements import list_health_entries, delete_health_measurement
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)

# Column indices for match-related columns (appended after the 21 base columns).
# Base columns 0..20: Timestamp, Lamp ID, Name, Mfg, An, Mode, Ua, Ug1, Ug2,
# Index, Ia%, S%, R%, K%, Ia, S, R, K, µg2, Emission, Ref.
COL_DBIAS = 9  # bias-servo shift; empty for plan-bias runs
COL_RESERVE = 21  # cathode reserve from the emission sweep; empty otherwise
COL_SEL = 23  # ● selected measurement marker
COL_GRP = 24  # group number or Δ


# ---------------------------------------------------------------------------
# Pure-logic helpers (testable without Qt)
# ---------------------------------------------------------------------------

# Sentinel stored as combo data for "no filtering on this axis". Shared
# so the predicate and the combo populators cannot drift apart.
FILTER_ALL = "all"


def entry_matches_filter(
    entry: Dict,
    *,
    regex: Optional[re.Pattern] = None,
    mode_filter: Optional[str] = None,
    verdict_filter: Optional[str] = None,
    verdict_thresholds: Optional[Dict[str, float]] = None,
    group_filter: Optional[str] = None,
    match_result=None,
    match_active: Optional[Set[Tuple[str, str]]] = None,
    hide_inactive: bool = False,
) -> bool:
    """Return True if *entry* should be visible given the active filters."""
    visible = True

    if regex:
        lamp_id = str(entry.get("lamp_id", ""))
        name = str(entry.get("name", ""))
        if not (regex.search(lamp_id) or regex.search(name)):
            visible = False

    if visible and mode_filter and mode_filter != "all":
        ug2_mode = (entry.get("conditions") or {}).get("ug2_mode", "")
        if ug2_mode != mode_filter:
            visible = False

    if visible and verdict_filter and verdict_filter != FILTER_ALL:
        h = entry.get("health") or {}
        index_val = h.get("index")
        thr = verdict_thresholds or {}
        if isinstance(index_val, (int, float)):
            if index_val >= thr.get("strong", 85):
                verdict = HEALTH_VERDICT_STRONG
            elif index_val >= thr.get("good", 65):
                verdict = HEALTH_VERDICT_GOOD
            elif index_val >= thr.get("weak", 40):
                verdict = HEALTH_VERDICT_WEAK
            else:
                verdict = HEALTH_VERDICT_REPLACE
        else:
            verdict = ""
        if verdict != verdict_filter:
            visible = False

    if visible and group_filter and group_filter != "all" and match_result:
        lamp_id = str(entry.get("lamp_id", ""))
        timestamp = str(entry.get("timestamp", ""))
        if group_filter == "unmatched":
            is_unmatched = any(
                r.lamp_id == lamp_id and r.timestamp == timestamp
                for r in match_result.unmatched)
            if not is_unmatched:
                visible = False
        elif group_filter.startswith("g"):
            grp_num = int(group_filter[1:])
            in_group = any(
                r.lamp_id == lamp_id and r.timestamp == timestamp
                for g in match_result.groups if g.number == grp_num
                for r in g.records)
            if not in_group:
                visible = False

    if visible and hide_inactive and match_active:
        lamp_id = str(entry.get("lamp_id", ""))
        timestamp = str(entry.get("timestamp", ""))
        # Row identity, not lamp identity: twin-anode records of one lamp
        # participate independently and must not hide each other.
        if (lamp_id, timestamp) not in match_active:
            visible = False

    return visible


def verdict_color(
    index_val,
    strong_min: float,
    good_min: float,
    weak_min: float,
) -> Optional[QColor]:
    """Return background QColor for a health index value."""
    if not isinstance(index_val, (int, float)):
        return None
    if index_val >= strong_min:
        return HEALTH_VERDICT_STRONG_BG
    if index_val >= good_min:
        return HEALTH_VERDICT_GOOD_BG
    if index_val >= weak_min:
        return HEALTH_VERDICT_WEAK_BG
    return HEALTH_VERDICT_REPLACE_BG


def extract_row_values(entry: Dict) -> List:
    """Extract display values from a health entry dict.

    Returns a list of 23 values matching the history table columns 0..22.
    Column order: Timestamp, Lamp ID, Name, Mfg, An, Mode,
                  Ua, Ug1, Ug2, Δbias, Index, Ia%, S%, R%, K%,
                  Ia, S, R, K, µg2, Emission, Reserve, Ref.
    Ug1 is the PLAN bias (the matching protocol); Δbias is the servo
    shift actually applied — plan + Δbias = the bias measured at.
    Mfg is stored YYYY-MM and sorts lexicographically == chronologically.
    """
    # ML-039: `.get(k, {})` does not protect against an explicit null in
    # a hand-edited/corrupt JSON — `or {}` does.
    h = entry.get("health") or {}
    metrics = h.get("metrics") or {}
    srk = entry.get("srk") or {}
    cond = entry.get("conditions") or {}
    an_val = cond.get("an")
    _mode_short = {
        TOPOLOGY_TRIODE: t("health.Mode_short_triode"),
        TOPOLOGY_PENTODE: t("health.Mode_short_pentode"),
        TOPOLOGY_TRIODE_CONNECTED: t("health.Mode_short_triode_connected"),
    }
    mode_val = _mode_short.get(cond.get("ug2_mode", ""), cond.get("ug2_mode", ""))
    raw = h.get("raw") or {}
    return [
        entry.get("timestamp", ""),
        entry.get("lamp_id", ""),
        entry.get("name", ""),
        entry.get("mfg_date", ""),
        an_val if an_val is not None else "",
        mode_val,
        cond.get("ua"),
        cond.get("ug1"),
        cond.get("ug2"),
        metrics.get("bias_shift_v"),
        h.get("index"),
        metrics.get("ia_pct"),
        metrics.get("s_pct"),
        metrics.get("r_pct"),
        metrics.get("k_pct"),
        raw.get("ia_op"),
        srk.get("s"),
        srk.get("r"),
        srk.get("k"),
        srk.get("mu_g1g2"),
        metrics.get("emission_ratio"),
        metrics.get("emission_reserve_pct"),
        h.get("reference_mode", ""),
    ]


def _format_timestamp_short(raw: str) -> str:
    """Render stored ISO timestamp as ``YY-MM-DD HH:MM`` for the history table.

    Input formats accepted (all produced by ``run_health_test``):
      - ``YYYY-MM-DDTHH:MM:SS`` (ISO, current)
      - ``YYYY-MM-DD HH:MM:SS`` (legacy with space)
      - ``YYYY-MM-DD``         (date-only legacy)

    Returns the original string if it does not start with a 4-digit year —
    we want graceful degradation, not a crash on unexpected input.
    """
    if len(raw) >= 4 and raw[:4].isdigit():
        body = raw[2:].replace("T", " ")  # drop century, normalize separator
        # "YY-MM-DD HH:MM:SS" (len 17) -> "YY-MM-DD HH:MM" (len 14).
        # Shorter inputs (date-only "YY-MM-DD", no-seconds "YY-MM-DD HH:MM")
        # pass through unchanged.
        if len(body) >= 16:
            body = body[:14]
        return body
    return raw


def _format_mfg_short(raw: str) -> str:
    """Render stored Mfg date ``YYYY-MM`` as ``YY-MM`` for compactness.

    Tooltip in ``_format_cell`` carries the full ``YYYY-MM`` so vintage
    tubes (year 1955 vs 2055) can be disambiguated on hover. Returns
    the input unchanged if it does not start with a 4-digit year.
    """
    if len(raw) >= 4 and raw[:4].isdigit():
        return raw[2:]
    return raw


def _format_cell(col: int, val) -> QTableWidgetItem:
    """Create a formatted QTableWidgetItem for the given column and value.

    Column layout (0..22):
      0 Timestamp, 1 Lamp ID, 2 Name, 3 Mfg, 4 An, 5 Mode,
      6 Ua, 7 Ug1, 8 Ug2, 9 Δbias,
      10 Index, 11 Ia%, 12 S%, 13 R%, 14 K%,
      15 Ia, 16 S, 17 R, 18 K, 19 µg2, 20 Emission, 21 Reserve, 22 Ref
    """
    if isinstance(val, (int, float)):
        # Numeric column: see ``_FormattedNumericItem`` for the Qt-quirk
        # workaround that keeps trailing zeros visible (``-7.0`` stays
        # ``-7.0``, not stripped to ``-7``) while numeric sort still
        # works.
        if col == 4:  # An — integer, no trailing-zero issue
            item = QTableWidgetItem()
            item.setText(str(int(val)))
            item.setData(Qt.ItemDataRole.EditRole, int(val))
            return item
        if col == 6 or col == 8:  # Ua, Ug2 — integer V (resolution 1 V)
            text = f"{float(val):.0f}"
        elif col == 7:  # Ug1 — tenths V (UI step 0.1 V)
            text = f"{float(val):.1f}"
        elif col == COL_DBIAS:
            # Explicit sign: the direction IS the diagnosis (a positive
            # shift = the tube needed opening = lost current).
            text = f"{float(val):+.2f}"
        elif 10 <= col <= 14:  # Index, Ia%, S%, R%, K%
            text = f"{float(val):.0f}"
        elif col == 19:  # µg2
            text = f"{float(val):.1f}"
        elif col == 20:  # emission_ratio
            text = f"{float(val):.3f}"
        elif col == COL_RESERVE:  # cathode reserve, % of nominal heater
            text = f"{float(val):.0f}"
        else:
            text = f"{float(val):.2f}"
        item = _FormattedNumericItem(float(val), text)
        if col in HEALTH_HISTORY_HIGHLIGHT_COLS:
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            item.setBackground(HEALTH_HISTORY_HIGHLIGHT_BG)
        return item
    else:
        item = QTableWidgetItem()
        # String columns: empty -> em-dash, else text.
        # Timestamp (col 0): display YY-MM-DD HH:MM, sort on full string.
        # Mfg (col 3): display YY-MM, tooltip + sort on full YYYY-MM.
        text_val = str(val) if val not in (None, "") else "—"
        if col == 0 and val:
            # ML-040: plain items alias EditRole↔DisplayRole — setText()
            # after setData(EditRole, full) clobbered the sort key.
            # FormattedTextItem keeps them separate (__lt__ on the full
            # string, display stays short).
            return _FormattedTextItem(str(val), _format_timestamp_short(str(val)))
        if col == 3 and val:
            full = str(val)
            item = _FormattedTextItem(full, _format_mfg_short(full))
            item.setToolTip(full)  # full YYYY-MM for vintage tube disambiguation
            return item
        item.setText(text_val)
    return item


def populate_history_table(
    table: QTableWidget,
    entries: List[Dict],
    color_fn: Optional[Callable] = None,
) -> None:
    """Fill *table* rows from *entries*.

    *color_fn*: ``(index_value) -> Optional[QColor]`` for verdict background.

    Performance: any column in ``QHeaderView.ResizeMode.ResizeToContents``
    triggers a full-column width recalculation on every ``setItem`` call,
    which scans all rows in that column. With many auto-size columns
    that turns populate into O(N²) in row count and showed up as visible
    lag on "Clear" in the Match workflow. We temporarily switch those
    columns to ``Interactive`` for the bulk insert and restore
    ``ResizeToContents`` after — Qt then does one final column-width
    pass instead of N redundant ones.
    """
    hh = table.horizontalHeader()
    _RTC = QHeaderView.ResizeMode.ResizeToContents
    _INTER = QHeaderView.ResizeMode.Interactive
    auto_cols = [c for c in range(table.columnCount())
                 if hh.sectionResizeMode(c) == _RTC]
    for c in auto_cols:
        hh.setSectionResizeMode(c, _INTER)
    try:
        table.setSortingEnabled(False)
        table.setRowCount(len(entries))
        for row_idx, e in enumerate(entries):
            vals = extract_row_values(e)
            index_val = (e.get("health") or {}).get("index")
            row_color = color_fn(index_val) if color_fn else None
            for col, val in enumerate(vals):
                item = _format_cell(col, val)
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, e)
                if col == COL_RESERVE and isinstance(val, (int, float)):
                    sweep = (e.get("health") or {}).get("emission_sweep") or {}
                    if sweep.get("knee_below_range"):
                        # The knee lies below the swept range: the value
                        # is a LOWER BOUND on the reserve, not the knee.
                        item = _FormattedNumericItem(
                            float(val), f"≥{float(val):.0f}")
                    # Interpretation hint: bands are engineering guides
                    # derived from the Miram-curve sources, not factory
                    # thresholds — the tooltip says so.
                    item.setToolTip(t("health.Reserve_tooltip"))
                if col == COL_DBIAS and isinstance(val, (int, float)):
                    # The Ug1 column shows the PLAN bias (the matching
                    # protocol); hover reveals where the servo actually
                    # measured and what the tube gave at the plan point.
                    h_blk = e.get("health") or {}
                    servo = h_blk.get("bias_servo") or {}
                    actual = servo.get("ug1")
                    if isinstance(actual, (int, float)):
                        tip = t("health.Dbias_tooltip", ug1=f"{actual:.2f}")
                        plan_ma = (h_blk.get("raw") or {}).get("ia_plan_ma")
                        plan_pct = (h_blk.get("metrics") or {}).get("ia_plan_pct")
                        if (isinstance(plan_ma, (int, float))
                                and isinstance(plan_pct, (int, float))):
                            tip += "\n" + t("health.Dbias_tooltip_plan",
                                            ia=f"{plan_ma:.1f}",
                                            pct=f"{plan_pct:.0f}")
                        item.setToolTip(tip)
                # Apply verdict tint to the whole row EXCEPT the headline
                # highlight columns — those keep their own bg so Index/Ia
                # stay visually distinct on every verdict variant.
                if row_color is not None and col not in HEALTH_HISTORY_HIGHLIGHT_COLS:
                    item.setBackground(row_color)
                table.setItem(row_idx, col, item)
            for extra_col in (COL_SEL, COL_GRP):
                table.setItem(row_idx, extra_col, QTableWidgetItem())
        table.setSortingEnabled(True)
        table.sortItems(0, Qt.SortOrder.DescendingOrder)
    finally:
        for c in auto_cols:
            hh.setSectionResizeMode(c, _RTC)


def build_match_entry_info(result) -> Dict[tuple, tuple]:
    """Build a lookup from (lamp_id, timestamp) -> (group_number, delta).

    Group number 0 means unmatched.
    """
    info: Dict[tuple, tuple] = {}
    for g in result.groups:
        for rec in g.records:
            info[(rec.lamp_id, rec.timestamp)] = (g.number, g.delta)
    for rec in result.unmatched:
        info[(rec.lamp_id, rec.timestamp)] = (0, 0.0)
    return info


def build_match_active(result) -> Set[Tuple[str, str]]:
    """(lamp_id, timestamp) of every row participating in the match:
    group members, unmatched pool members and the similar-mode anchor.

    Row identity, not one-slot-per-lamp: a lamp_id-keyed dict made
    twin-anode records of one lamp (and the anchor vs its own candidate
    twin) evict each other, dimming rows that DID participate.
    """
    active: Set[Tuple[str, str]] = set()
    for g in result.groups:
        for rec in g.records:
            active.add((rec.lamp_id, rec.timestamp))
    for rec in result.unmatched:
        active.add((rec.lamp_id, rec.timestamp))
    anchor = getattr(result, "anchor", None)
    if anchor is not None:
        active.add((anchor.lamp_id, anchor.timestamp))
    return active


def build_matching_conditions(
    entries: List[Dict],
    tube_mode: str,
) -> Optional[tuple]:
    """Build conditions tuple for match_tubes from entries.

    Returns ``(ua, ug1, ug2, tube_mode, bias_servo)`` or None — the same
    shape as ``lm19.tube_matching._conditions_key``, which compares
    against this tuple with ``==``. A shape drift between the two
    silently empties every match pool.
    """
    cond_entries = [
        e for e in entries
        if (e.get("conditions") or {}).get("ug2_mode",
                                            TOPOLOGY_PENTODE) == tube_mode
    ]
    if not cond_entries:
        return None
    c = cond_entries[0].get("conditions") or {}
    return (
        round(float(c.get("ua", 0)), 1),
        round(float(c.get("ug1", 0)), 1),
        round(float(c.get("ug2", 0)), 1),
        tube_mode,
        bool(c.get("bias_servo", False)),
    )


def format_match_groups_text(result, group_label: str = "Group", unmatched_label: str = "Unmatched") -> List[str]:
    """Build human-readable lines from a MatchResult.

    Returns list of formatted strings, one per group + unmatched section.
    """
    lines = []
    for g in result.groups:
        ids = ", ".join(r.lamp_id for r in g.records)
        # shared_bias protocol: the predicted quiescent-current imbalance
        # rides along - it shaped the selection, so the copied text
        # carries it too (method visibility).
        diq = getattr(g, "iq_imbalance_ma", None)
        diq_part = f", \u03b4Iq\u2248{diq:.1f} mA" if diq is not None else ""
        lines.append(
            f"{group_label} {g.number} (\u0394={g.delta:.1f}%{diq_part}): {ids}")
    if result.unmatched:
        ids = ", ".join(r.lamp_id for r in result.unmatched)
        lines.append(f"{unmatched_label}: {ids}")
    return lines


def format_match_csv_rows(result, group_label: str = "Group") -> List[List[str]]:
    """Build CSV rows from a MatchResult.

    Returns list of rows (each a list of strings), WITHOUT header.
    Columns: Group, Lamp ID, Mfg, Ia, S, R, Δ, δIq. The δIq column is
    always present for a stable schema; it carries a value only for
    shared_bias-protocol groups and "—" otherwise.
    """
    def _mfg(rec):
        e = rec.entry if rec.entry else {}
        return str(e.get("mfg_date") or "") if isinstance(e, dict) else ""

    rows = []
    for g in result.groups:
        diq = getattr(g, "iq_imbalance_ma", None)
        diq_str = f"{diq:.1f}" if diq is not None else "—"
        for rec in g.records:
            rows.append([
                str(g.number), rec.lamp_id, _mfg(rec),
                f"{rec.ia:.2f}", f"{rec.s:.3f}", f"{rec.r:.1f}",
                f"{g.delta:.1f}", diq_str,
            ])
    for rec in result.unmatched:
        rows.append([
            "—", rec.lamp_id, _mfg(rec),
            f"{rec.ia:.2f}", f"{rec.s:.3f}", f"{rec.r:.1f}", "—", "—",
        ])
    return rows


def table_to_tsv(
    table: QTableWidget,
    selected_only: bool = False,
) -> tuple:
    """Extract rows from a QTableWidget as TSV lines.

    Returns ``(lines_with_header, data_row_count)``.
    """
    row_count = table.rowCount()
    col_count = table.columnCount()
    if row_count <= 0 or col_count <= 0:
        return [], 0

    if selected_only:
        rows = sorted({idx.row() for idx in table.selectionModel().selectedRows()})
        if not rows:
            return [], 0
    else:
        rows = list(range(row_count))
    # ML-042: filters hide rows — an export of "the table" must match what
    # the user sees, not resurrect filtered-out measurements.
    rows = [r for r in rows if not table.isRowHidden(r)]
    if not rows:
        return [], 0

    headers = [
        table.horizontalHeaderItem(c).text()
        if table.horizontalHeaderItem(c) else ""
        for c in range(col_count)
    ]
    lines = ["\t".join(headers)]
    for r in rows:
        vals = []
        for c in range(col_count):
            item = table.item(r, c)
            vals.append(item.text() if item else "")
        lines.append("\t".join(vals))
    return lines, len(rows)


def table_to_rows(
    table: QTableWidget,
    selected_only: bool = False,
) -> tuple:
    """Like :func:`table_to_tsv` but returns ``(header_row, data_rows,
    count)`` as lists of cell strings — for writers that must quote
    cell values (ML-042: CSV via the ``csv`` module, not string replace).
    """
    lines, count = table_to_tsv(table, selected_only)
    if not count:
        return [], [], 0
    split = [line.split("\t") for line in lines]
    return split[0], split[1:], count
