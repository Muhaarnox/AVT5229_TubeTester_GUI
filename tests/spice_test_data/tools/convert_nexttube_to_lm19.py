#!/usr/bin/env python3
"""
Convert Next-Tube.com CSV data files to unified LM19 JSON format.

Reads CSV files (converted from XLS via LibreOffice), parses Up/Ug/Us/Ip
data blocks, and writes structured JSON files with metadata and validation.

Usage:
    python convert_nexttube_to_lm19.py
"""

import csv
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CSV_DIR = SCRIPT_DIR.parent / "raw" / "next_tube" / "csv"
OUT_DIR = SCRIPT_DIR.parent / "converted"

TUBE_META = {
    "6N8C":    {"aliases": ["6SN7GT", "ECC32"],    "author": "Eugene V. Karpov", "date": "2002-01-08"},
    "6N6P":    {"aliases": [],                      "author": "Eugene V. Karpov", "date": "2002-01-08"},
    "6N1P":    {"aliases": ["ECC85", "6AQ8"],       "author": "Eugene V. Karpov", "date": "2002-01-09"},
    "6N5P":    {"aliases": [],                      "author": "Eugene V. Karpov", "date": "2002-01-09"},
    "GU50":    {"aliases": ["LS50", "P50-2"],       "author": "Eugene V. Karpov", "date": "2002-01-09"},
    "6C41C":   {"aliases": [],                      "author": "Eugene V. Karpov", "date": "2002-01-09"},
    "6N23P":   {"aliases": ["ECC88", "6DJ8"],       "author": "Eugene V. Karpov", "date": "2002-01-09"},
    "6P3C":    {"aliases": ["6L6G"],                "author": "Eugene V. Karpov", "date": "2002-01-10"},
    "6P14P":   {"aliases": ["EL84", "N329"],        "author": "Eugene V. Karpov", "date": "2002-03-27"},
    "6P45S":   {"aliases": [],                      "author": "Eugene V. Karpov", "date": "2002-03-27"},
    "6N13S":   {"aliases": ["6AS7G"],               "author": "Eugene V. Karpov", "date": "2002-09-02"},
    "6F5P":    {"aliases": ["6GV8", "ECL85"],       "author": "Eugene V. Karpov", "date": "2002-10-21"},
    "6N3P":    {"aliases": ["2C51", "6CC42"],       "author": "Eugene V. Karpov", "date": "2003-04-07"},
    "6S6B":    {"aliases": [],                      "author": "Eugene V. Karpov", "date": "2004-03-01"},
    "6N27P":   {"aliases": ["ECC86", "6GM8"],       "author": "Eugene V. Karpov", "date": "2004-12-06"},
    "6N2P":    {"aliases": ["12AX7"],               "author": "Eugene V. Karpov", "date": "2004-12-28"},
    "6S19P":   {"aliases": [],                      "author": "Eugene V. Karpov", "date": "2005-04-22"},
    "6S4S":    {"aliases": ["6A3", "6A5"],          "author": "Shevchenko",       "date": "2006-07-29"},
}

KOREN_PARAMS = {
    "6N2P":      {"mu": 106.00, "ex": 1.398, "kg1": 1326.3, "kp": 415.26, "kvb": 4680.6, "vct": 0.30},
    "6S19P":     {"mu": 3.71,   "ex": 1.000, "kg1": 297.0,  "kp": 12.17,  "kvb": 400.0},
    "6N8C":      {"mu": 20.06,  "ex": 1.306, "kg1": 978.5,  "kp": 108.75, "kvb": 300.0},
    "6N1P":      {"mu": 34.51,  "ex": 1.26,  "kg1": 2106.9, "kp": 238.29, "kvb": 300.0},
    "6N5P":      {"mu": 40.73,  "ex": 1.396, "kg1": 1362.2, "kp": 245.44, "kvb": 300.0},
    "6N23P":     {"mu": 36.21,  "ex": 1.316, "kg1": 1131.7, "kp": 171.10, "kvb": 300.0},
    "6N13S":     {"mu": 2.35,   "ex": 1.247, "kg1": 637.1,  "kp": 12.40,  "kvb": 300.0},
    "6N3P":      {"mu": 34.82,  "ex": 1.909, "kg1": 1445.3, "kp": 171.13, "kvb": 300.0},
    "6S6B":      {"mu": 25.50,  "ex": 1.361, "kg1": 846.1,  "kp": 203.20, "kvb": 300.0},
    "6N27P":     {"mu": 16.3,   "ex": 1.3,   "kg1": 350.1,  "kp": 76.28,  "kvb": 150.0},
    "6S4S":      {"mu": 4.32,   "ex": 1.078, "kg1": 523.3,  "kp": 43.32,  "kvb": 300.0},
    "6P14P_T":   {"mu": 21.49,  "ex": 1.428, "kg1": 514.4,  "kp": 136.46, "kvb": 300.0},
    "6P45S_T":   {"mu": 3.96,   "ex": 1.505, "kg1": 522.0,  "kp": 44.51,  "kvb": 300.0},
    "6F5P_T":    {"mu": 73.16,  "ex": 1.668, "kg1": 522.9,  "kp": 353.07, "kvb": 519.0, "vct": 0.60},
    "6F5P_PT":   {"mu": 7.49,   "ex": 1.622, "kg1": 1336.3, "kp": 59.91,  "kvb": 120.0},
    "6F5P_P":    {"mu": 11.5,   "ex": 1.204, "kg1": 303.0,  "kg2": 4500,  "kp": 34.87, "kvb": 20.1},
}

# Maps CSV filename -> list of sections to extract
# Each section: (key, topology, label, triode_group_range)
# triode_group_range: None = all, (start, end) = slice of triode groups
CSV_SECTIONS = {
    "6N8C.csv":    [("6N8C",    "triode", "triode", None)],
    "6N6P.csv":    [("6N6P",    "triode", "triode", None)],
    "6N1P.csv":    [("6N1P",    "triode", "triode", None)],
    "6N5P.csv":    [("6N5P",    "triode", "triode", None)],
    "6N23P.csv":   [("6N23P",   "triode", "triode", None)],
    "6N13S.csv":   [("6N13S",   "triode", "triode", None)],
    "6N3P.csv":    [("6N3P",    "triode", "triode", None)],
    "6S6B.csv":    [("6S6B",    "triode", "triode", None)],
    "6N27P.csv":   [("6N27P",   "triode", "triode", None)],
    "6N2P.csv":    [("6N2P",    "triode", "triode", None)],
    "6S19P.csv":   [("6S19P",   "triode", "triode", None)],
    "6C41C.csv":   [("6C41C",   "triode", "triode", None)],
    "6S4S.csv":    [("6S4S",    "triode", "triode", None)],
    "6P14P-T.csv": [("6P14P_T", "triode", "pentode in triode mode", None)],
    "6P45S-T.csv": [("6P45S_T", "triode", "pentode in triode mode", None)],
    "GU50.csv":    [
        ("GU50_T", "triode",  "triode mode",  None),
        ("GU50_P", "pentode", "pentode mode",  None),
    ],
    "6P3C.csv":    [
        ("6P3C_T", "triode",  "triode-connected mode", None),
        ("6P3C_P", "pentode", "pentode mode",  None),
    ],
    "6F5P.csv":    [
        ("6F5P_T",  "triode",  "triode section",                (0, 9)),
        ("6F5P_PT", "triode",  "pentode section in triode mode", (9, 22)),
        ("6F5P_P",  "pentode", "pentode section",                None),
    ],
}


def parse_csv_groups(filepath):
    """Parse Next-Tube CSV into data groups (triode/pentode)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)

    triode_groups = []
    pentode_groups = []
    i = 0
    while i < len(rows):
        row = rows[i]
        if not row:
            i += 1
            continue
        cell0 = row[0].strip()
        if cell0.startswith("Up") and "(V)" in cell0:
            ua_vals = _extract_nums(row[1:])
            if i + 1 >= len(rows):
                i += 1
                continue
            ug_row = rows[i + 1]
            ug_label = ug_row[0].strip() if ug_row else ""
            if not (ug_label.startswith("Ug") and "(V)" in ug_label):
                i += 1
                continue
            ug_vals = _extract_nums(ug_row[1:])
            if i + 2 >= len(rows):
                i += 2
                continue
            next_row = rows[i + 2]
            next_label = next_row[0].strip() if next_row else ""
            if next_label.startswith("Us") and "(V)" in next_label:
                us_vals = _extract_nums(next_row[1:])
                ip_row = rows[i + 3] if i + 3 < len(rows) else []
                ip_vals = _extract_nums(ip_row[1:]) if len(ip_row) > 1 else []
                n = min(len(ua_vals), len(ug_vals), len(us_vals), len(ip_vals))
                pentode_groups.append((ua_vals[:n], ug_vals[:n], us_vals[:n], ip_vals[:n]))
                i += 4
                continue
            elif next_label.startswith("Ip"):
                ip_vals = _extract_nums(next_row[1:])
                n = min(len(ua_vals), len(ug_vals), len(ip_vals))
                triode_groups.append((ua_vals[:n], ug_vals[:n], ip_vals[:n]))
                i += 3
                continue
        i += 1
    return triode_groups, pentode_groups


def _extract_nums(cells):
    vals = []
    for cell in cells:
        s = cell.strip().strip('"').replace(",", ".")
        if not s:
            continue
        try:
            vals.append(float(s))
        except ValueError:
            pass
    return vals


def build_triode_json(key, groups, group_range=None):
    """Build JSON from triode data groups."""
    if group_range is not None:
        start, end = group_range
        groups = groups[start:end]

    points = []
    for ua, ug, ip in groups:
        for j in range(len(ua)):
            ia_ma = round(ip[j] * 1000, 4)
            if ia_ma < 0:
                continue
            points.append({
                "ua": round(ua[j], 1),
                "ug1": round(ug[j], 2),
                "ia": ia_ma,
            })
    return points


def build_pentode_json(key, groups):
    """Build JSON from pentode data groups."""
    points = []
    for ua, ug, us, ip in groups:
        for j in range(len(ua)):
            points.append({
                "ua": round(ua[j], 1),
                "ug1": round(ug[j], 2),
                "ug2": round(us[j], 1),
                "ig2": None,
                "ia": round(ip[j] * 1000, 4),
            })
    return points


def make_output(key, topology, label, points, csv_name):
    """Build the full JSON output dict."""
    base_tube = key.split("_")[0]
    if base_tube.startswith("6P") or base_tube == "GU50":
        base_tube_for_meta = base_tube.replace("-T", "")
    meta_key = base_tube
    for mk in TUBE_META:
        if base_tube.startswith(mk) or mk.startswith(base_tube):
            meta_key = mk
            break

    meta = TUBE_META.get(meta_key, {})
    author = meta.get("author", "Eugene V. Karpov")
    date = meta.get("date", "")
    aliases = meta.get("aliases", [])

    result = {
        "_comment": f"{key} {label} — Next-Tube.com empirical data ({author}, {date})",
        "tube_type": key,
        "topology": topology,
        "source": f"Next-Tube.com / {author}",
        "url": "https://next-tube.com/data.php",
    }

    if aliases:
        result["aliases"] = aliases

    params = KOREN_PARAMS.get(key)
    if params:
        result["expected_params"] = {
            **params,
            "_note": f"Koren fit by {author}"
        }

    result["points"] = points

    units = {"ua": "V", "ug1": "V (negative)", "ia": "mA"}
    if topology == "pentode":
        units["ug2"] = "V"
        units["ig2"] = "mA (null = not measured)"
    result["_units"] = units

    return result


def validate_json(data, filename):
    warnings = []
    points = data.get("points", [])
    if not points:
        warnings.append("Empty points array")
        return warnings

    topology = data.get("topology", "triode")
    for i, pt in enumerate(points):
        if pt.get("ia", 0) < 0:
            warnings.append(f"Point {i}: negative ia={pt['ia']}")
        if topology == "pentode" and "ug2" not in pt:
            warnings.append(f"Point {i}: pentode missing 'ug2'")
    return warnings


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(CSV_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {CSV_DIR}\n")

    total_files = 0
    total_points = 0
    converted = []

    for csv_name, sections in sorted(CSV_SECTIONS.items()):
        csv_path = CSV_DIR / csv_name
        if not csv_path.exists():
            print(f"  SKIP {csv_name}: file not found")
            continue

        triode_groups, pentode_groups = parse_csv_groups(csv_path)
        print(f"--- {csv_name} --- ({len(triode_groups)} triode groups, {len(pentode_groups)} pentode groups)")

        for key, topology, label, group_range in sections:
            if topology == "triode":
                points = build_triode_json(key, triode_groups, group_range)
            else:
                points = build_pentode_json(key, pentode_groups)

            if not points:
                print(f"  SKIP {key}: no data points found")
                continue

            out_data = make_output(key, topology, label, points, csv_name)
            warnings = validate_json(out_data, key)

            out_name = f"{topology}_{key}_nexttube.json"
            out_path = OUT_DIR / out_name
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2, ensure_ascii=False)

            total_files += 1
            total_points += len(points)
            print(f"  -> {out_name} ({len(points)} points)")
            if warnings:
                for w in warnings:
                    print(f"     WARN: {w}")

            converted.append({
                "file": out_name,
                "key": key,
                "topology": topology,
                "points": len(points),
                "warnings": len(warnings),
            })

    print(f"\n{'='*60}")
    print(f"Converted: {total_files} files, {total_points} total points")
    triodes = [c for c in converted if c["topology"] == "triode"]
    pentodes = [c for c in converted if c["topology"] == "pentode"]
    print(f"  Triodes:  {len(triodes)} files, {sum(c['points'] for c in triodes)} points")
    print(f"  Pentodes: {len(pentodes)} files, {sum(c['points'] for c in pentodes)} points")
    print(f"  Warnings: {sum(c['warnings'] for c in converted)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
