"""uTracer .utd file parser and converter to LM19 format.

Parses measurement data exported by the uTracer GUI (Ronald Dekker)
via "Save Measurement Matrix". Supports output curves I(Va, Vg) and
transfer curves I(Vg, Va), with optional screen current Is.

No Qt dependency -- pure business logic, fully testable.
"""


from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


def parse_utd(filepath: str, stats: Optional[Dict] = None) -> Dict:
    """Parse a uTracer .utd file.

    Args:
        filepath: path to the .utd file.

    Returns:
        dict with keys:
          format      -- 'output' (Va on X) or 'transfer' (Vg on X)
          has_is      -- True if screen current (Is) data present
          x_name      -- 'Va' or 'Vg'
          step_name   -- 'Vg' or 'Va'
          step_values -- list of stepping-variable values
          x_values    -- list of running-variable values
          ia_matrix   -- ia_matrix[x_idx][step_idx]
          is_matrix   -- is_matrix[x_idx][step_idx] or None

    Raises:
        ValueError: if the file cannot be parsed.
    """
    # utf-8-sig (not plain utf-8): Excel-exported CSV / Windows tools
    # routinely add a UTF-8 BOM. Plain utf-8 keeps it as ``﻿`` in
    # the first character of line 0, breaking ``"Va" in header.split()``
    # checks below — silent wrong-path parsing. utf-8-sig skips the BOM
    # if present and behaves identically otherwise.
    text = Path(filepath).read_text(encoding="utf-8-sig", errors="replace")
    lines = [ln.rstrip() for ln in text.splitlines()]

    if len(lines) < 3:
        raise ValueError("File too short: need at least 3 lines (header, steps, data)")

    # --- Row 0: header --------------------------------------------------
    header_line = lines[0]
    is_output = "Va" in header_line.split()[:3]
    has_is = "Is" in header_line

    # --- Row 1: stepping values -----------------------------------------
    step_line = lines[1]
    raw_step_values = [float(v) for v in re.findall(r"V\w*\s*=\s*(-?\S+)\s*V", step_line)]
    if not raw_step_values:
        raise ValueError("No stepping values found in line 2")

    # For pentode files, each Vg appears twice (Ia + Is columns).
    # Deduplicate by taking every other value when has_is is True.
    if has_is and len(raw_step_values) >= 2:
        step_values = raw_step_values[::2]
    else:
        step_values = raw_step_values

    # --- Rows 2+: data matrix -------------------------------------------
    x_values: List[float] = []
    raw_data: List[List[float]] = []

    for line in lines[2:]:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        try:
            nums = [float(t) for t in tokens]
        except ValueError:
            continue  # skip non-numeric lines
        if not nums:
            continue
        x_values.append(nums[0])
        raw_data.append(nums[1:])

    if not x_values:
        raise ValueError("No data rows found")

    # --- Separate Ia and Is columns -------------------------------------
    # ML-071: a row shorter than the expected column count is truncation —
    # zero-filling fabricated Ia=0 points (a dead-tube look) silently.
    # Skip such rows (with their x value) and surface the count.
    n_steps = len(step_values)
    short_rows = 0
    kept_x: List[float] = []
    if has_is:
        expected_cols = n_steps * 2
        ia_matrix: List[List[float]] = []
        is_matrix: List[List[float]] = []
        for x, row_data in zip(x_values, raw_data):
            if len(row_data) < expected_cols:
                short_rows += 1
                continue
            kept_x.append(x)
            ia_matrix.append([row_data[i * 2] for i in range(n_steps)])
            is_matrix.append([row_data[i * 2 + 1] for i in range(n_steps)])
    else:
        expected_cols = n_steps
        ia_matrix = []
        is_matrix = None  # type: ignore[assignment]
        for x, row_data in zip(x_values, raw_data):
            if len(row_data) < expected_cols:
                short_rows += 1
                continue
            kept_x.append(x)
            ia_matrix.append(row_data[:n_steps])
    x_values = kept_x

    if short_rows:
        log.warning(
            "uTracer import: skipped %d truncated data row(s) "
            "(expected %d columns)", short_rows, expected_cols)
    if stats is not None:
        stats["short_rows"] = short_rows
    if not x_values:
        raise ValueError(
            "All data rows are shorter than the expected column count "
            f"({expected_cols}) — file truncated or header mismatch")

    return {
        "format": "output" if is_output else "transfer",
        "has_is": has_is,
        "x_name": "Va" if is_output else "Vg",
        "step_name": "Vg" if is_output else "Va",
        "step_values": step_values,
        "x_values": x_values,
        "ia_matrix": ia_matrix,
        "is_matrix": is_matrix if has_is else None,
    }


def utd_to_lm19_points(
    parsed: Dict,
    vs: float = 0.0,
    vh: float = 0.0,
) -> List[Dict]:
    """Convert parsed .utd data to LM19 measurement points.

    Args:
        parsed: result of parse_utd().
        vs: screen grid voltage (Ug2/Vs) -- not stored in .utd file.
        vh: heater voltage (Uh/Vh) -- not stored in .utd file.

    Returns:
        list of dicts {ua, ug1, ug2, ia, ig2, uh, ih}.
    """
    points: List[Dict] = []
    is_output = parsed["format"] == "output"
    has_is = parsed["is_matrix"] is not None

    for x_idx, x_val in enumerate(parsed["x_values"]):
        for s_idx, s_val in enumerate(parsed["step_values"]):
            if is_output:
                ua = x_val
                ug1 = s_val
            else:
                ua = s_val
                ug1 = x_val

            ia = parsed["ia_matrix"][x_idx][s_idx]
            ig2 = parsed["is_matrix"][x_idx][s_idx] if has_is else 0.0

            points.append({
                "ua": ua,
                "ug1": ug1,
                "ug2": vs,
                "ia": ia,
                "ig2": ig2,
                "uh": vh,
                "ih": 0.0,
            })

    return points


def guess_meta_from_filename(filename: str) -> Dict:
    """Try to extract tube type, Vs, and lamp ID from the file name.

    Heuristics:
      - A trailing ``_<number>`` is interpreted as Vs (e.g. ``EL84_250.utd`` -> Vs=250).
      - The base name (before ``_<number>``) is the tube type.
      - The full stem is the lamp ID.

    Args:
        filename: file name (with or without directory/extension).

    Returns:
        dict with optional keys: tube_type, vs, lamp_id.
    """
    stem = Path(filename).stem
    result: Dict = {"lamp_id": stem}

    # Try to split off trailing _<digits> as Vs
    m = re.match(r"^(.+?)_(\d+)$", stem)
    if m:
        result["tube_type"] = m.group(1)
        result["vs"] = float(m.group(2))
    else:
        result["tube_type"] = stem

    return result
