"""Universal CSV/TSV importer for tube measurement data.

Handles CSV files from LM19 own export, uTracer (via Excel), and other
tube testers. Provides auto-detection of separator and column mapping.

No Qt dependency -- pure business logic, fully testable.
"""


from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# Known column name aliases -> LM19 canonical key
_COLUMN_ALIASES: Dict[str, str] = {
    # LM19 native names
    "ua": "ua", "ug1": "ug1", "ug2": "ug2",
    "ia": "ia", "ig2": "ig2", "uh": "uh", "ih": "ih",
    # uTracer names
    "va": "ua", "vg": "ug1", "vs": "ug2",
    "is": "ig2", "vh": "uh",
    # Common variants
    "v_a": "ua", "v_g": "ug1", "v_s": "ug2",
    "i_a": "ia", "i_s": "ig2", "v_h": "uh", "i_h": "ih",
    "vanode": "ua", "vgrid": "ug1", "vscreen": "ug2",
    "ianode": "ia", "iscreen": "ig2", "vheater": "uh",
    "anode_voltage": "ua", "grid_voltage": "ug1",
    "screen_voltage": "ug2", "anode_current": "ia",
    "screen_current": "ig2", "heater_voltage": "uh",
    "heater_current": "ih",
    # With units in parens (stripped during matching)
}

# All valid LM19 keys for column mapping
VALID_KEYS = ("ua", "ug1", "ug2", "ia", "ig2", "uh", "ih")


def detect_separator(text: str) -> str:
    """Auto-detect CSV separator from the first few lines.

    Checks tab, semicolon, comma in priority order.

    Args:
        text: file content (or first ~10 lines).

    Returns:
        Detected separator character: '\\t', ';', or ','.
    """
    # Take first 10 non-comment, non-empty lines
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
        if len(lines) >= 10:
            break

    if not lines:
        return ","

    # Count separators across all sample lines
    tab_count = sum(ln.count("\t") for ln in lines)
    semi_count = sum(ln.count(";") for ln in lines)
    comma_count = sum(ln.count(",") for ln in lines)

    # Prefer tab if present in most lines
    if tab_count >= len(lines):
        return "\t"
    if semi_count >= len(lines):
        return ";"
    if comma_count >= len(lines):
        return ","

    # Fallback: highest total count
    counts = {"\t": tab_count, ";": semi_count, ",": comma_count}
    return max(counts, key=counts.get)  # type: ignore[arg-type]


def _normalize_header(name: str) -> str:
    """Normalize a column header for matching.

    Strips units in parentheses, whitespace, and lowercases.
    """
    # Remove content in parentheses: "Ia (mA)" -> "Ia"
    name = re.sub(r"\s*\(.*?\)", "", name)
    return name.strip().lower().replace(" ", "_")


def detect_columns(headers: List[str]) -> Dict[int, str]:
    """Auto-detect column mapping from header names.

    Args:
        headers: list of raw column header strings.

    Returns:
        dict mapping column_index -> LM19 key (ua, ug1, ...).
        Columns that cannot be mapped are omitted.
    """
    mapping: Dict[int, str] = {}
    used_keys: set = set()

    for idx, raw in enumerate(headers):
        norm = _normalize_header(raw)
        if norm in _COLUMN_ALIASES:
            key = _COLUMN_ALIASES[norm]
            if key not in used_keys:
                mapping[idx] = key
                used_keys.add(key)

    return mapping


def skip_comment_lines(lines: List[str]) -> List[str]:
    """Filter out comment lines (starting with ``#``) and empty lines."""
    return [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def parse_csv(
    text: str,
    mapping: Dict[int, str],
    separator: str = ",",
    stats: Optional[Dict] = None,
) -> List[Dict]:
    """Parse CSV text into LM19 measurement points.

    Args:
        text: full file content.
        mapping: column_index -> LM19 key (ua, ug1, ug2, ia, ig2, uh, ih).
        separator: column separator.
        stats: optional out-dict — filled with ``skipped_rows``/``bad_cells``
            counters so the UI can surface the loss (failure-visibility rule: the
            WARNING alone is not a user-facing channel).

    Returns:
        list of dicts {ua, ug1, ug2, ia, ig2, uh, ih}.

    Failure-visibility trade-off: a row with ANY mapped, non-empty cell that
    will not parse is dropped whole (no silent 0.0) and counted in a WARNING.
    This means one corrupt optional field (e.g. a garbage ig2) discards the
    row's otherwise-good ua/ia — preferred over importing contaminated data,
    and the WARNING makes the loss visible. Grouped-thousands cells like
    "1,234.5" become "1.234.5" via the comma-decimal replace and are treated
    as corrupt.
    """
    lines = skip_comment_lines(text.splitlines())
    if not lines:
        return []

    # Skip header line (first non-comment line)
    data_lines = lines[1:]

    points: List[Dict] = []
    bad_cells = 0
    skipped_rows = 0
    for line in data_lines:
        cols = line.split(separator)
        point: Dict = {
            "ua": 0.0, "ug1": 0.0, "ug2": 0.0,
            "ia": 0.0, "ig2": 0.0, "uh": 0.0, "ih": 0.0,
        }
        valid = False
        row_bad = False
        for col_idx, key in mapping.items():
            if col_idx >= len(cols):
                continue
            # Accept both '.' and ',' as decimal separator. European
            # locales (de_DE, fr_FR, ru_RU, …) export CSVs with comma
            # decimals (``12,5``); without this replace, ``float('12,5')``
            # raises ValueError → field silently skipped → mostly-empty
            # point dicts. Safe for English '.'-decimal files: replace
            # is a no-op when no comma is present.
            raw = cols[col_idx].strip().replace(",", ".")
            if not raw:
                continue
            try:
                point[key] = float(raw)
                valid = True
            except ValueError:
                # A mapped, non-empty cell that won't parse is corruption, not
                # an absent column — drop the whole row instead of silently
                # leaving 0.0 in that slot (failure-visibility, no silent
                # contaminated data).
                bad_cells += 1
                row_bad = True
        if valid and not row_bad:
            points.append(point)
        elif row_bad:
            skipped_rows += 1

    if bad_cells or skipped_rows:
        log.warning(
            "CSV import: skipped %d row(s) with %d unparseable cell(s) "
            "out of %d data row(s)", skipped_rows, bad_cells, len(data_lines))
    if stats is not None:
        stats["skipped_rows"] = skipped_rows
        stats["bad_cells"] = bad_cells

    return points
