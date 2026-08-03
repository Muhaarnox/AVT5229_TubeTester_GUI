"""CurveTraceData (.dat) parser and converter to LM19 points.

Upstream format from "pypsucurvetrace / curvetracedata",
see SOURCES_INDEX.md.

Parses raw `.dat` files from pypsucurvetrace/curvetracedata and converts them
to LM19 point dictionaries:
`{ua, ug1, ug2, ia, ig2, uh, ih}`.

No Qt dependency -- pure business logic, fully testable.
"""


from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List


def _guess_type_from_stem(stem: str) -> str:
    """Best-effort tube type extraction from specimen/file stem."""
    # Common specimen names: EL34_12, 801_VT62_3, EL34_1_VENDOR, ...
    m = re.match(r"^(.+?)_(\d+)(?:[_-].*)?$", stem)
    if m:
        return m.group(1)
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def parse_curvetracedata_dat(filepath: str) -> Dict:
    """Parse a CurveTraceData `.dat` file.

    Returns:
        dict with keys:
          sample_name: str
          date_str: str
          points: list[dict] with ua, ug1, ia_A
    """
    sample_name = ""
    date_str = ""
    points: List[Dict] = []
    seen = set()

    # utf-8-sig: skip the UTF-8 BOM if present (Excel-exported files
    # often have one). Plain utf-8 would leave it as ``﻿`` in
    # the first character — silent parse failure on the first field.
    text = Path(filepath).read_text(encoding="utf-8-sig", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("%"):
            m = re.search(r"Sample:\s*(.+)", line)
            if m:
                sample_name = m.group(1).strip()
            m = re.search(r"Date / time:\s*(.+)", line)
            if m:
                date_str = m.group(1).strip()
            continue

        parts = line.split()
        if len(parts) < 10:
            continue

        try:
            ua_meas = float(parts[2])  # PSU1 measured voltage (anode)
            ia_meas = float(parts[3])  # PSU1 measured current (A)
            limiter = int(parts[4])    # PSU1 limiter flag
            ug1_meas = float(parts[7])  # PSU2 measured voltage (grid)
        except (ValueError, IndexError):
            continue

        # Skip points where PSU1 current limiter engaged.
        if limiter == 1:
            continue

        ua = round(ua_meas, 1)
        ug1 = round(ug1_meas, 3)
        key = (ua, ug1)
        if key in seen:
            continue
        seen.add(key)

        points.append({
            "ua": ua,
            "ug1": ug1,
            "ia_A": ia_meas,
        })

    if not points:
        raise ValueError("No data points found in CurveTraceData file")

    return {
        "sample_name": sample_name,
        "date_str": date_str,
        "points": points,
    }


def dat_to_lm19_points(parsed: Dict, *, vs: float = 0.0, vh: float = 0.0) -> List[Dict]:
    """Convert parsed `.dat` payload to LM19 points.

    Notes:
      - CurveTraceData stores Ia in amperes; LM19 uses milliamperes.
      - Ig2 is not present in this format and is set to 0.0.
      - Uh / Ug2 are supplied by user metadata dialog.
    """
    points: List[Dict] = []
    for p in parsed.get("points", []):
        ia_ma = float(p.get("ia_A", 0.0)) * 1000.0
        points.append({
            "ua": float(p.get("ua", 0.0)),
            "ug1": float(p.get("ug1", 0.0)),
            "ug2": float(vs),
            "ia": ia_ma,
            "ig2": 0.0,
            "uh": float(vh),
            "ih": 0.0,
        })
    return points


def guess_meta_from_dat_filename(filename: str, sample_name: str = "") -> Dict:
    """Guess tube metadata from a `.dat` file path and sample name."""
    path = Path(filename)
    stem = path.stem

    # Common raw layout: .../<tube_type>/data/<file>.dat
    guessed_type = ""
    if path.parent.name.lower() == "data" and path.parent.parent.name:
        guessed_type = path.parent.parent.name
    else:
        guessed_type = _guess_type_from_stem(sample_name.strip() or stem)

    lamp_id = sample_name.strip() or stem
    return {
        "tube_type": guessed_type,
        "lamp_id": lamp_id,
        "name": lamp_id,
    }
