"""CSV export for tube measurement data.

Provides flat-table and matrix formats for measured points.
Pure functions — no Qt dependency.
"""


from __future__ import annotations

from typing import Dict, List, Optional
import io

from lm19.constants import UA_ROUND, UG1_ROUND


def format_csv(
    points: List[Dict],
    *,
    tube_type: str = "",
    lamp_id: str = "",
    name: str = "",
    timestamp: str = "",
    mfg_date: str = "",
    srk: Optional[Dict] = None,
    scan_info: str = "",
    separator: str = ";",
    include_computed: bool = True,
    is_triode: bool = False,
) -> str:
    """Format measurement points as a flat CSV table.

    Args:
        points: list of dicts with ua, ug1, ug2, ia, ig2, uh, ih.
        tube_type: tube type name for header comment.
        lamp_id: lamp identifier for header comment.
        name: measurement name for header comment.
        timestamp: measurement timestamp for header comment.
        srk: optional {s, r, k} dict for header comment.
        scan_info: optional scan range description.
        separator: column separator (";" or "," or "\\t").
        include_computed: add Pa, Pg2, Ik computed columns.

    Returns:
        Complete CSV content as string.
    """
    buf = io.StringIO()
    sep = separator

    # Header comments
    buf.write("# LM19 Tube Tester Export\n")
    if tube_type:
        line = f"# Tube: {tube_type}"
        if lamp_id:
            line += f"  ID: {lamp_id}"
        if name:
            line += f"  Name: {name}"
        buf.write(line + "\n")
    if timestamp:
        buf.write(f"# Date: {timestamp}\n")
    if mfg_date:
        buf.write(f"# Manufactured: {mfg_date}\n")
    if srk:
        s = srk.get("s")
        r = srk.get("r")
        k = srk.get("k")
        parts = []
        if s is not None:
            parts.append(f"S: {s:.2f} mA/V")
        if r is not None:
            parts.append(f"R: {r:.2f} kOhm")
        if k is not None:
            parts.append(f"K: {k:.3f}")
        if parts:
            buf.write(f"# {'  '.join(parts)}\n")
    if scan_info:
        buf.write(f"# Scan: {scan_info}\n")
    buf.write(f"# Points: {len(points)}\n")
    buf.write("#\n")

    # Column header
    if is_triode:
        cols = ["Ua", "Ug1", "Ia", "Uh", "Ih"]
        if include_computed:
            cols += ["Pa"]
    else:
        cols = ["Ua", "Ug1", "Ug2", "Ia", "Ig2", "Uh", "Ih"]
        if include_computed:
            cols += ["Pa", "Pg2", "Ik"]
    buf.write(sep.join(cols) + "\n")

    # Data rows
    for p in points:
        ua = p.get("ua", 0.0)
        ug1 = p.get("ug1", 0.0)
        ia = p.get("ia", 0.0)
        uh = p.get("uh", 0.0)
        ih = p.get("ih", 0.0)

        if is_triode:
            vals = [
                f"{ua:.1f}",
                f"{ug1:.2f}",
                f"{ia:.3f}",
                f"{uh:.2f}",
                f"{ih:.3f}",
            ]
            if include_computed:
                pa = ua * ia / 1000.0
                vals += [f"{pa:.3f}"]
        else:
            ug2 = p.get("ug2", 0.0)
            ig2 = p.get("ig2", 0.0)
            vals = [
                f"{ua:.1f}",
                f"{ug1:.2f}",
                f"{ug2:.1f}",
                f"{ia:.3f}",
                f"{ig2:.3f}",
                f"{uh:.2f}",
                f"{ih:.3f}",
            ]
            if include_computed:
                pa = ua * ia / 1000.0
                pg2 = ug2 * ig2 / 1000.0
                ik = ia + ig2
                vals += [f"{pa:.3f}", f"{pg2:.3f}", f"{ik:.3f}"]
        buf.write(sep.join(vals) + "\n")

    return buf.getvalue()


def format_matrix(
    points: List[Dict],
    *,
    tube_type: str = "",
    lamp_id: str = "",
    timestamp: str = "",
    mfg_date: str = "",
    separator: str = ";",
    parameter: str = "Ia",
    is_triode: bool = False,
) -> str:
    """Format measurement points as a matrix (Ug1 rows x Ua columns).

    Creates one matrix per unique Ug2 value.
    The matrix value is selected by ``parameter``: Ia, Ig2, Pa.

    Args:
        points: measurement data.
        tube_type: for header comment.
        lamp_id: for header comment.
        timestamp: for header comment.
        separator: column separator.
        parameter: which value to put in the matrix cells.

    Returns:
        CSV content with matrix blocks.
    """
    if not points:
        return ""

    buf = io.StringIO()
    sep = separator

    buf.write("# LM19 Tube Tester Export (matrix format)\n")
    if tube_type:
        line = f"# Tube: {tube_type}"
        if lamp_id:
            line += f"  ID: {lamp_id}"
        buf.write(line + "\n")
    if timestamp:
        buf.write(f"# Date: {timestamp}\n")
    if mfg_date:
        buf.write(f"# Manufactured: {mfg_date}\n")
    buf.write("#\n")

    # Group by Ug2 (for triodes: single group, no Ug2 header)
    if is_triode:
        ug2_set = [0.0]
    else:
        ug2_set = sorted(set(round(p.get("ug2", 0.0), 1) for p in points))
    ua_all = sorted(set(round(p.get("ua", 0.0), 1) for p in points))
    ug1_all = sorted(set(round(p.get("ug1", 0.0), 2) for p in points))

    for ug2 in ug2_set:
        if is_triode:
            subset = list(points)
        else:
            subset = [p for p in points if abs(p.get("ug2", 0.0) - ug2) < 0.5]
        if not subset:
            continue

        # Determine actual Ua/Ug1 for this Ug2
        ua_vals = sorted(set(round(p["ua"], UA_ROUND) for p in subset))
        ug1_vals = sorted(set(round(p["ug1"], UG1_ROUND) for p in subset))

        units = {"Ia": "mA", "Ig2": "mA", "Pa": "W"}.get(parameter, "")
        if is_triode:
            buf.write(f"# {parameter} ({units})\n")
        else:
            buf.write(f"# {parameter} ({units}) at Ug2={ug2:.1f}V\n")

        # Header row: Ua values
        buf.write(f"Ua:{sep}" + sep.join(f"{ua:.1f}" for ua in ua_vals) + "\n")

        # Build lookup
        lookup = {}
        for p in subset:
            key = (round(p["ug1"], UG1_ROUND), round(p["ua"], UA_ROUND))
            if parameter == "Ia":
                lookup[key] = p.get("ia", 0.0)
            elif parameter == "Ig2":
                lookup[key] = p.get("ig2", 0.0)
            elif parameter == "Pa":
                lookup[key] = p.get("ua", 0.0) * p.get("ia", 0.0) / 1000.0

        # Data rows
        for ug1 in ug1_vals:
            row = [f"Ug1={ug1:.2f}"]
            for ua in ua_vals:
                val = lookup.get((ug1, ua))
                if val is not None:
                    row.append(f"{val:.3f}")
                else:
                    row.append("")
            buf.write(sep.join(row) + "\n")

        buf.write("\n")

    return buf.getvalue()


def format_multi_csv(
    entries: List[Dict],
    *,
    separator: str = ";",
    include_computed: bool = True,
    is_triode: bool = False,
) -> str:
    """Format multiple measurements into a single CSV with series column.

    Args:
        entries: list of dicts with keys: lamp_type, lamp_id, name,
                 timestamp, points, and optionally data (full measurement).
        separator: column separator.
        include_computed: add Pa, Pg2, Ik columns.

    Returns:
        CSV content with Series column identifying each measurement.
    """
    buf = io.StringIO()
    sep = separator

    buf.write("# LM19 Tube Tester Export (multi-measurement)\n")
    buf.write(f"# Measurements: {len(entries)}\n")
    buf.write("#\n")

    if is_triode:
        cols = ["Series", "Lamp_Type", "Lamp_ID", "Mfg", "Ua", "Ug1",
                "Ia", "Uh", "Ih"]
        if include_computed:
            cols += ["Pa"]
    else:
        cols = ["Series", "Lamp_Type", "Lamp_ID", "Mfg", "Ua", "Ug1", "Ug2",
                "Ia", "Ig2", "Uh", "Ih"]
        if include_computed:
            cols += ["Pa", "Pg2", "Ik"]
    buf.write(sep.join(cols) + "\n")

    for idx, entry in enumerate(entries):
        lamp_type = entry.get("lamp_type", "")
        lamp_id = entry.get("lamp_id", "")
        name = entry.get("name", "")
        # mfg_date may live at entry root (from list_measurement_entries) or
        # inside entry["data"] (compare flow when added from scan).
        mfg = str(
            entry.get("mfg_date")
            or (entry.get("data") or {}).get("mfg_date")
            or ""
        )
        series_name = name if name else f"#{idx + 1}"
        pts = entry.get("points", [])

        for p in pts:
            ua = p.get("ua", 0.0)
            ug1 = p.get("ug1", 0.0)
            ia = p.get("ia", 0.0)
            uh = p.get("uh", 0.0)
            ih = p.get("ih", 0.0)

            if is_triode:
                vals = [
                    series_name, lamp_type, lamp_id, mfg,
                    f"{ua:.1f}", f"{ug1:.2f}",
                    f"{ia:.3f}", f"{uh:.2f}", f"{ih:.3f}",
                ]
                if include_computed:
                    pa = ua * ia / 1000.0
                    vals += [f"{pa:.3f}"]
            else:
                ug2 = p.get("ug2", 0.0)
                ig2 = p.get("ig2", 0.0)
                vals = [
                    series_name, lamp_type, lamp_id, mfg,
                    f"{ua:.1f}", f"{ug1:.2f}", f"{ug2:.1f}",
                    f"{ia:.3f}", f"{ig2:.3f}", f"{uh:.2f}", f"{ih:.3f}",
                ]
                if include_computed:
                    pa = ua * ia / 1000.0
                    pg2 = ug2 * ig2 / 1000.0
                    ik = ia + ig2
                    vals += [f"{pa:.3f}", f"{pg2:.3f}", f"{ik:.3f}"]
            buf.write(sep.join(vals) + "\n")

    return buf.getvalue()


def write_csv(path: str, content: str) -> None:
    """Write CSV content to file."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
