"""eTracer CSV v2.0 parser and converter to LM19 points.

Parses CSV files exported by eTracer software (Essues Technologies).
Format: groups of 6 rows per curve-set (HV1_V, HV1_I, HV2_V, HV2_I,
NEGV, SWEEP_SOURCE).

No Qt dependency -- pure business logic, fully testable.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)

log = logging.getLogger(__name__)

# ── module local constants ──
_ROWS_PER_CURVE = 6
_SWEEP_NEGV = 1.0
_SWEEP_HV2 = 2.0


def parse_etracer_csv(filepath: str, stats: Optional[Dict] = None) -> Dict:
    """Parse an eTracer CSV v2.0 file.

    Returns:
        dict with keys:
          version: str (format version, e.g. '2.0')
          etd_file: str (original .etd path from header)
          hv2_on: bool
          hv2_link: bool
          negv_setting: tuple(start, stop, step) or None
          hv2_setting: tuple(start, stop, step) or None
          curves: list of dicts, each with:
            curve_idx: int
            hv1_v: list[float]  (anode voltages)
            hv1_i: list[float]  (anode currents, mA)
            hv2_v: list[float]  (screen/unit2 voltages)
            hv2_i: list[float]  (screen/unit2 currents, mA)
            negv: list[float]   (grid voltages)
            sweep_src: float    (0, 1, or 2)

    Raises:
        ValueError: if the file cannot be parsed.
    """
    # utf-8-sig: strip BOM if present (Windows exports often add one).
    # With plain utf-8 the BOM would land in lines[0], breaking
    # header parsing silently.
    text = Path(filepath).read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()

    # --- Parse header comments ---
    version = ""
    etd_file = ""
    hv2_on = False
    hv2_link = False
    negv_setting: Optional[tuple] = None
    hv2_setting: Optional[tuple] = None

    data_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comment = stripped.lstrip("#").strip()
            m = re.search(r"ETRACER_CSV_FORMAT_VERSION:\s*(\S+)", comment)
            if m:
                version = m.group(1)
            m = re.search(r"ETD_FILE:\s*(.+)", comment)
            if m:
                etd_file = m.group(1).strip()
            m = re.search(r"HV2:(\w+)\s+HV2_LINK:(\w+)", comment)
            if m:
                hv2_on = m.group(1).upper() == "ON"
                hv2_link = m.group(2).upper() == "ON"
            m = re.search(r"NEGV_SETTING:\s*\[([^\]]+)\]", comment)
            if m:
                negv_setting = _parse_setting(m.group(1))
            m = re.search(r"HV2_SETTING:\s*\[([^\]]+)\]", comment)
            if m:
                hv2_setting = _parse_setting(m.group(1))
        else:
            data_lines.append(stripped)

    if not data_lines:
        raise ValueError("No data rows found in eTracer CSV file")

    # --- Parse data rows into curve groups ---
    rows_by_idx: Dict[int, List[List[float]]] = {}
    for line in data_lines:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            idx = int(float(parts[0]))
        except (ValueError, IndexError):
            continue
        values = []
        for v in parts[1:]:
            v = v.strip()
            if not v or v.lower() == "nan":
                values.append(float("nan"))
            else:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(float("nan"))
        if idx not in rows_by_idx:
            rows_by_idx[idx] = []
        rows_by_idx[idx].append(values)

    # --- Build curve list ---
    curves: List[Dict] = []
    nan_points = 0
    for idx in sorted(rows_by_idx.keys()):
        group = rows_by_idx[idx]
        if len(group) < _ROWS_PER_CURVE:
            continue  # incomplete curve-set, skip

        hv1_v = _strip_nan(group[0])
        hv1_i = _strip_nan(group[1])
        hv2_v = _strip_nan(group[2])
        hv2_i = _strip_nan(group[3])
        negv = _strip_nan(group[4])
        sweep_raw = _strip_nan(group[5])

        # All sweep_src values in a curve should be the same; take the
        # first FINITE one (a corrupt first cell must not become 'nan').
        sweep_src = next((v for v in sweep_raw if not math.isnan(v)), 0.0)

        # Trim all lists to the shortest length (misaligned nan padding).
        min_len = min(len(hv1_v), len(hv1_i), len(hv2_v), len(hv2_i),
                      len(negv), len(sweep_raw))
        if min_len == 0:
            continue

        # ML-059: _strip_nan removes only TRAILING padding now — the old
        # break-at-first-NaN lost the whole tail of valid data after one
        # corrupt cell. Drop just the corrupt point indices instead.
        data_arrays = (hv1_v, hv1_i, hv2_v, hv2_i, negv)
        keep = [i for i in range(min_len)
                if not any(math.isnan(a[i]) for a in data_arrays)]
        nan_points += min_len - len(keep)
        if not keep:
            continue

        curves.append({
            "curve_idx": idx,
            "hv1_v": [hv1_v[i] for i in keep],
            "hv1_i": [hv1_i[i] for i in keep],
            "hv2_v": [hv2_v[i] for i in keep],
            "hv2_i": [hv2_i[i] for i in keep],
            "negv": [negv[i] for i in keep],
            "sweep_src": sweep_src,
        })

    if nan_points:
        log.warning(
            "eTracer import: dropped %d point(s) with NaN cells inside "
            "curve data", nan_points)
    if stats is not None:
        stats["nan_points"] = nan_points

    if not curves:
        raise ValueError("No valid curve-sets found in eTracer CSV file")

    return {
        "version": version,
        "etd_file": etd_file,
        "hv2_on": hv2_on,
        "hv2_link": hv2_link,
        "negv_setting": negv_setting,
        "hv2_setting": hv2_setting,
        "curves": curves,
    }


def detect_topology(parsed: Dict) -> str:
    """Detect tube topology from parsed eTracer data.

    Returns:
        'triode' — pure triode (HV2 off)
        'triode_connected' — pentode in triode wiring (HV2 linked to HV1)
        'pentode' — true pentode mode (HV2 swept independently)
    """
    if not parsed.get("hv2_on"):
        return TOPOLOGY_TRIODE
    if parsed.get("hv2_link"):
        return TOPOLOGY_TRIODE_CONNECTED
    # Check sweep source: if any curve uses HV2 as sweep, it's pentode.
    for curve in parsed.get("curves", []):
        if curve.get("sweep_src") == _SWEEP_HV2:
            return TOPOLOGY_PENTODE
    # HV2 on but not linked and not sweeping — fixed screen voltage
    return TOPOLOGY_PENTODE


def etracer_to_lm19_points(
    parsed: Dict,
    *,
    vh: float = 0.0,
) -> List[Dict]:
    """Convert parsed eTracer data to LM19 measurement points.

    Args:
        parsed: result of parse_etracer_csv().
        vh: heater voltage (not stored in eTracer CSV).

    Returns:
        list of dicts {ua, ug1, ug2, ia, ig2, uh, ih}.
    """
    topology = detect_topology(parsed)
    points: List[Dict] = []

    for curve in parsed.get("curves", []):
        sweep_src = curve.get("sweep_src", 0.0)
        n = len(curve["hv1_v"])

        for i in range(n):
            ua = curve["hv1_v"][i]
            ia = curve["hv1_i"][i]

            if sweep_src == _SWEEP_HV2:
                # Pentode: NEGV row has the grid voltage, HV2 is the screen
                ug1 = curve["negv"][i]
                ug2 = curve["hv2_v"][i]
                ig2 = curve["hv2_i"][i]
            else:
                # NEGV is the grid sweep. Any non-triode topology has a driven
                # screen (triode-connected OR fixed-screen pentode) — keep its
                # recorded Ug2/Ig2; only a pure triode has no screen.
                ug1 = curve["negv"][i]
                if topology != TOPOLOGY_TRIODE:
                    ug2 = curve["hv2_v"][i]
                    ig2 = curve["hv2_i"][i]
                else:
                    ug2 = 0.0
                    ig2 = 0.0

            # eTracer stores grid voltage as positive; LM19 expects negative.
            if ug1 > 0:
                ug1 = -ug1

            points.append({
                "ua": round(ua, 2),
                "ug1": round(ug1, 2),
                "ug2": round(ug2, 2),
                "ia": round(ia, 3),
                "ig2": round(ig2, 3),
                "uh": vh,
                "ih": 0.0,
            })

    return points


def guess_meta_from_etracer(filepath: str, parsed: Dict) -> Dict:
    """Guess tube metadata from eTracer CSV file path and header.

    Priority: CSV filename stem (user-visible) > ETD_FILE header (fallback
    when CSV stem is unusable, e.g. contains spaces).

    Returns:
        dict with keys: tube_type, lamp_id, name, topology.
    """
    path = Path(filepath)
    stem = path.stem

    # Primary: CSV filename stem
    tube_type = _clean_tube_name(stem)

    # If CSV stem is problematic (spaces, empty), fall back to ETD header
    if not tube_type or " " in tube_type:
        etd_file = parsed.get("etd_file", "")
        if etd_file:
            etd_clean = _clean_tube_name(Path(etd_file).stem)
            if etd_clean and " " not in etd_clean:
                tube_type = etd_clean

    # Last resort: raw stem
    if not tube_type:
        tube_type = stem

    topology = detect_topology(parsed)

    return {
        "tube_type": tube_type,
        "lamp_id": stem,
        "name": stem,
        "topology": topology,
    }


# ── internal helpers ──

_TUBE_NAME_SUFFIXES = re.compile(
    r"[_-](triode|pentode|Quintet|DRU?|DR|VT\d+)$", re.IGNORECASE,
)


def _clean_tube_name(name: str) -> str:
    """Strip common suffixes from an eTracer tube name.

    Repeatedly removes trailing _triode, _pentode, _Quintet, _DR, _DRU
    to get the base tube type.
    """
    clean = name.strip()
    prev = ""
    while clean != prev:
        prev = clean
        clean = _TUBE_NAME_SUFFIXES.sub("", clean)
    return clean


def extract_heater_from_etd(csv_path: str, etd_filename: str) -> Optional[float]:
    """Try to find a companion .etd file and extract HEATER_V.

    Looks for the .etd file in the same directory as the CSV, using
    the filename from the ETD_FILE header.  Returns None if not found.
    """
    if not etd_filename:
        return None
    etd_name = Path(etd_filename).name
    candidate = Path(csv_path).parent / etd_name
    if not candidate.is_file():
        return None
    try:
        text = candidate.read_text(encoding="utf-8-sig", errors="replace")
        m = re.search(r"HEATER_V\s*=\s*([\d.]+)", text)
        if m:
            return float(m.group(1))
    except (OSError, ValueError):
        pass
    return None


def _parse_setting(s: str) -> Optional[tuple]:
    """Parse a setting string like '0.0:70.0:10.0' into (start, stop, step)."""
    parts = s.strip().split(":")
    if len(parts) == 3:
        try:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            pass
    return None


def _strip_nan(values: List[float]) -> List[float]:
    """Return values with trailing NaN (alignment padding) removed.

    ML-059: only TRAILING NaN is padding; an interior NaN is a corrupt
    cell and is handled per-point by the caller. The old implementation
    broke at the FIRST NaN and lost the tail of valid data.
    """
    end = len(values)
    while end > 0 and isinstance(values[end - 1], float) and math.isnan(values[end - 1]):
        end -= 1
    return values[:end]
