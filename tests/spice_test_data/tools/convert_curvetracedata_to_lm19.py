#!/usr/bin/env python3
"""
Convert pypsucurvetrace .dat files to unified LM19 JSON format.

Reads .dat files from mbrennwa/curvetracedata GitHub repo, parses
Ua/Ia/Ug1 measurement columns, and writes structured JSON files.

For tube types with multiple specimens, picks the first available
specimen for SPICE fitting data and records total specimen count.

Usage:
    python convert_curvetracedata_to_lm19.py
"""

import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR.parent / "raw" / "curvetracedata"
OUT_DIR = SCRIPT_DIR.parent / "converted"

TUBE_DEFS = {
    "ECC88": {
        "topology": "triode",
        "aliases": ["6DJ8", "E88CC", "6922", "6N23P"],
        "data_dir": "ECC88",
    },
    "EL34": {
        "topology": "triode",
        "aliases": [],
        "_note_topology": "Pentode tested in triode mode (G2=Anode)",
        "data_dir": "EL34",
    },
    "300B": {
        "topology": "triode",
        "aliases": [],
        "data_dir": "300B",
    },
    "KT66": {
        "topology": "triode",
        "aliases": [],
        "_note_topology": "Beam tetrode tested in triode mode (G2=Anode)",
        "data_dir": "KT66",
    },
    "6C33C": {
        "topology": "triode",
        "aliases": ["6S33S"],
        "data_dir": "6C33C",
    },
    "6N23P": {
        "topology": "triode",
        "aliases": ["ECC88", "6DJ8"],
        "data_dir": "6N23P_6H23pi",
    },
    "6N30P": {
        "topology": "triode",
        "aliases": ["6H30Pi"],
        "data_dir": "6N30P_6H30pi",
    },
    "ECC81": {
        "topology": "triode",
        "aliases": ["12AT7"],
        "data_dir": "ECC81_801",
    },
    "PCC88": {
        "topology": "triode",
        "aliases": ["7DJ8"],
        "data_dir": "PCC88",
    },
    "6E5P": {
        "topology": "triode",
        "aliases": [],
        "_note_topology": "Pentode tested in triode mode",
        "data_dir": "6E5P",
    },
    "D3A": {
        "topology": "triode",
        "aliases": [],
        "_note_topology": "Pentode tested in triode mode",
        "data_dir": "D3A",
    },
    "10_VT25": {
        "topology": "triode",
        "aliases": ["VT-25", "10"],
        "data_dir": "10_VT25",
    },
    "801_VT62": {
        "topology": "triode",
        "aliases": ["VT-62", "801"],
        "data_dir": "801_VT62",
    },
    "841_VT51": {
        "topology": "triode",
        "aliases": ["VT-51", "841"],
        "data_dir": "841_VT51",
    },
    "THF51": {
        "topology": "triode",
        "aliases": [],
        "data_dir": "THF51",
    },
    "807": {
        "topology": "triode",
        "aliases": [],
        "_note_topology": "Pentode tested in triode mode",
        "data_dir": "807",
    },
    "DL92": {
        "topology": "triode",
        "aliases": ["3S4"],
        "_note_topology": "Output pentode tested in triode mode",
        "data_dir": "DL92_3S4",
    },
    "DL94": {
        "topology": "triode",
        "aliases": ["3V4"],
        "_note_topology": "Output pentode tested in triode mode",
        "data_dir": "DL94",
    },
    "DL96": {
        "topology": "triode",
        "aliases": ["3C4"],
        "_note_topology": "Output pentode tested in triode mode",
        "data_dir": "DL96",
    },
    "DL98": {
        "topology": "triode",
        "aliases": ["3B4"],
        "_note_topology": "Output pentode tested in triode mode",
        "data_dir": "DL98_3B4",
    },
    "20B": {
        "topology": "triode",
        "aliases": [],
        "data_dir": "20B",
    },
    "32B": {
        "topology": "triode",
        "aliases": [],
        "data_dir": "32B",
    },
}

KOREN_PUBLISHED = {
    "ECC88": {"mu": 28, "ex": 1.3, "kg1": 330, "kp": 320, "kvb": 300},
    "EL34":  {"mu": 11.0, "ex": 1.35, "kg1": 650, "kp": 60, "kvb": 24,
              "_note": "Pentode params; triode-mode mu ~11-13"},
    "300B":  {"mu": 3.95, "ex": 1.4, "kg1": 1550, "kp": 65, "kvb": 300},
    "6C33C": {"mu": 3.1, "ex": 1.4, "kg1": 163, "kp": 15, "kvb": 300},
}

MIN_IA_THRESHOLD = 0.0001  # 0.1 mA minimum to avoid noise-floor data


def parse_dat_file(filepath):
    """Parse a pypsucurvetrace .dat file."""
    sample_name = None
    date_str = None
    idle_point = None
    points = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("%"):
                m = re.search(r"Sample:\s*(.+)", line)
                if m:
                    sample_name = m.group(1).strip()
                m = re.search(r"Date / time:\s*(.+)", line)
                if m:
                    date_str = m.group(1).strip()
                m = re.search(
                    r"OPERATING POINT.*U1\s*=\s*([\d.]+)\s*V\s+"
                    r"I1\s*=\s*([\d.]+)\s*A\s+"
                    r"U2\s*=\s*(-?[\d.]+)\s*V",
                    line,
                )
                if m:
                    idle_point = {
                        "ua": float(m.group(1)),
                        "ia_A": float(m.group(2)),
                        "ug1": float(m.group(3)),
                    }
                continue

            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                ua_meas = float(parts[2])
                ia_meas = float(parts[3])
                limiter = int(parts[4])
                ug1_meas = float(parts[7])
            except (ValueError, IndexError):
                continue

            if limiter == 1:
                continue

            points.append({
                "ua": round(ua_meas, 1),
                "ug1": round(ug1_meas, 3),
                "ia_A": ia_meas,
            })

    return sample_name, date_str, idle_point, points


def group_by_curves(points):
    """Group points into curves by Ug1 value."""
    curves = {}
    for pt in points:
        ug1_key = round(pt["ug1"], 1)
        curves.setdefault(ug1_key, []).append(pt)
    return curves


def filter_useful_points(points):
    """Filter out zero-current and noise-floor points."""
    filtered = []
    seen = set()
    for pt in points:
        if pt["ia_A"] < 0:
            continue
        key = (pt["ua"], round(pt["ug1"], 1))
        if key in seen:
            continue
        seen.add(key)
        filtered.append(pt)
    return filtered


def pick_representative_specimen(data_dir):
    """Pick first non-dead, non-reserved specimen."""
    dat_files = sorted(data_dir.glob("*.dat"))
    skip_patterns = ["DEAD", "SHORT", "reserved", "DEFECT", "BROKEN"]
    good_files = []
    for f in dat_files:
        name_upper = f.stem.upper()
        if any(p.upper() in name_upper for p in skip_patterns):
            continue
        good_files.append(f)
    return good_files[0] if good_files else (dat_files[0] if dat_files else None)


def count_specimens(data_dir):
    """Count total and usable specimens."""
    dat_files = list(data_dir.glob("*.dat"))
    skip_patterns = ["DEAD", "SHORT", "reserved", "DEFECT", "BROKEN"]
    good = [f for f in dat_files
            if not any(p.upper() in f.stem.upper() for p in skip_patterns)]
    return len(dat_files), len(good)


def build_json(tube_key, tube_def, sample_name, date_str, idle_point,
               points, total_specimens, usable_specimens, all_specimen_names):
    """Build LM19 JSON output."""
    topology = tube_def["topology"]
    aliases = tube_def.get("aliases", [])
    note_topo = tube_def.get("_note_topology", "")

    comment_parts = [f"{tube_key} {topology}"]
    if note_topo:
        comment_parts.append(f"({note_topo})")
    comment_parts.append(f"— pypsucurvetrace specimen {sample_name}")
    comment = " ".join(comment_parts)

    result = {
        "_comment": comment,
        "tube_type": tube_key,
        "topology": topology,
        "source": "pypsucurvetrace / Matthias Brennwald",
        "url": "https://github.com/mbrennwa/curvetracedata",
    }

    if aliases:
        result["aliases"] = aliases

    if tube_key in KOREN_PUBLISHED:
        result["published_params"] = {
            **KOREN_PUBLISHED[tube_key],
            "_note": "Koren tube.lib published values"
        }

    if idle_point:
        result["idle_point"] = {
            "ua": idle_point["ua"],
            "ug1": idle_point["ug1"],
            "ia": round(idle_point["ia_A"] * 1000, 4),
            "_note": "Operating point used for pre-heat/idle"
        }

    result["specimen"] = sample_name
    if date_str:
        result["measurement_date"] = date_str[:10]
    result["specimens_total"] = total_specimens
    result["specimens_usable"] = usable_specimens
    if len(all_specimen_names) <= 100:
        result["all_specimens"] = all_specimen_names

    json_points = []
    for pt in points:
        ia_mA = round(pt["ia_A"] * 1000, 4)
        json_points.append({
            "ua": pt["ua"],
            "ug1": round(pt["ug1"], 2),
            "ia": ia_mA,
        })

    result["points"] = json_points

    result["_units"] = {
        "ua": "V",
        "ug1": "V (negative)",
        "ia": "mA"
    }

    notes = []
    if note_topo:
        notes.append(note_topo)
    notes.append(f"{usable_specimens} of {total_specimens} specimens available for matching.")
    result["_notes"] = notes

    return result


def validate_json(data, filename):
    warnings = []
    points = data.get("points", [])
    if not points:
        warnings.append("Empty points array")
        return warnings
    for i, pt in enumerate(points):
        if pt.get("ia", 0) < 0:
            warnings.append(f"Point {i}: negative ia={pt['ia']}")
    return warnings


def get_specimen_names(data_dir):
    """Get list of specimen names from .dat files."""
    skip_patterns = ["DEAD", "SHORT", "reserved", "DEFECT", "BROKEN"]
    names = []
    for f in sorted(data_dir.glob("*.dat")):
        if any(p.upper() in f.stem.upper() for p in skip_patterns):
            continue
        names.append(f.stem)
    return names


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Raw data directory: {RAW_DIR}\n")

    total_files = 0
    total_points = 0
    converted = []

    for tube_key, tube_def in sorted(TUBE_DEFS.items()):
        data_dir = RAW_DIR / tube_def["data_dir"] / "data"
        if not data_dir.exists():
            print(f"SKIP {tube_key}: {data_dir} not found")
            continue

        total_spec, usable_spec = count_specimens(data_dir)
        if total_spec == 0:
            print(f"SKIP {tube_key}: no .dat files")
            continue

        specimen_file = pick_representative_specimen(data_dir)
        if specimen_file is None:
            print(f"SKIP {tube_key}: no usable specimen")
            continue

        sample_name, date_str, idle_point, raw_points = parse_dat_file(specimen_file)
        if not raw_points:
            print(f"SKIP {tube_key}: no data points in {specimen_file.name}")
            continue

        points = filter_useful_points(raw_points)
        if not points:
            print(f"SKIP {tube_key}: all points filtered out")
            continue

        all_names = get_specimen_names(data_dir)
        curves = group_by_curves(points)
        n_curves = len(curves)

        json_data = build_json(
            tube_key, tube_def, sample_name, date_str, idle_point,
            points, total_spec, usable_spec, all_names
        )

        warnings = validate_json(json_data, tube_key)

        out_name = f"triode_{tube_key}_curvetracedata.json"
        out_path = OUT_DIR / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        total_files += 1
        total_points += len(points)
        print(f"  {tube_key:12s} -> {out_name} "
              f"({len(points)} pts, {n_curves} curves, "
              f"{usable_spec}/{total_spec} specimens)")
        if warnings:
            for w in warnings:
                print(f"     WARN: {w}")

        converted.append({
            "file": out_name,
            "key": tube_key,
            "points": len(points),
            "curves": n_curves,
            "specimens": usable_spec,
            "warnings": len(warnings),
        })

    print(f"\n{'='*60}")
    print(f"Converted: {total_files} tube types, {total_points} total points")
    print(f"  Total specimens across all types: "
          f"{sum(c['specimens'] for c in converted)}")
    print(f"  Warnings: {sum(c['warnings'] for c in converted)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
