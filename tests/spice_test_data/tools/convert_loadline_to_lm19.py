"""
Convert valve data from loadline_plotter CSV format to LM19 test JSON format.
Source: valves_data.csv, valves_specs.csv
Output: triode_{tube}_datasheet.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "raw" / "loadline_plotter"
OUT = BASE / "converted"

DATA_CSV = RAW / "valves_data.csv"
SPECS_CSV = RAW / "valves_specs.csv"

TUBE_ALIASES = {
    "E88CC": ["6DJ8", "6922"],
    "ECC85": ["6AQ8"],
    "ECC82": ["12AU7"],
    "ECC83": ["12AX7"],
    "ECC81": ["12AT7"],
    "12AY7": [],
}


def load_specs() -> dict[str, dict]:
    """Load valve specs from valves_specs.csv. Returns dict: tube_type -> specs."""
    specs = {}
    with open(SPECS_CSV, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip #VALVE header
        next(reader)  # skip valve, Pmax, VaMax...
        for row in reader:
            if len(row) < 5:
                continue
            valve = row[0].strip()
            try:
                pmax = float(row[1])
                va_max = float(row[2])
                mu = float(row[3])
                ra = float(row[4])
            except (ValueError, IndexError):
                continue
            specs[valve] = {
                "pa_max": pmax,
                "ua_max": va_max,
                "mu": mu,
                "ra": ra,
            }
    return specs


def load_data() -> dict[str, list[dict]]:
    """Load valve curves from valves_data.csv. Returns dict: tube_type -> list of points."""
    tubes: dict[str, list[dict]] = {}
    with open(DATA_CSV, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            valve = row[0].strip()
            gridcurve = row[1].strip()
            if gridcurve.upper() == "PMAX":
                continue
            try:
                ug1 = float(gridcurve)
            except ValueError:
                continue
            # Parse alternating X (ua), Y (ia) pairs from columns 2+
            values = [float(x.strip()) for x in row[2:] if x.strip()]
            points = []
            for i in range(0, len(values) - 1, 2):
                ua = values[i]
                ia = values[i + 1]
                points.append({"ua": ua, "ug1": ug1, "ia": ia})
            if valve not in tubes:
                tubes[valve] = []
            tubes[valve].extend(points)
    return tubes


def main() -> None:
    specs = load_specs()
    data = load_data()
    OUT.mkdir(parents=True, exist_ok=True)

    for tube_type, points in data.items():
        if not points:
            continue
        # Sort by ug1 then ua for consistent output
        points.sort(key=lambda p: (p["ug1"], p["ua"]))

        aliases = TUBE_ALIASES.get(tube_type, [])
        tube_specs = specs.get(tube_type, {})

        obj = {
            "_comment": f"Datasheet anode characteristics for {tube_type} triode section",
            "tube_type": tube_type,
            "topology": "triode",
            "aliases": aliases,
            "source": "loadline_plotter / Datasheet curves (GitHub: andmarti1424)",
            "url": "https://github.com/andmarti1424/loadline_plotter",
            "points": points,
            "datasheet_specs": tube_specs if tube_specs else None,
            "_units": {"ua": "V", "ug1": "V (negative)", "ia": "mA"},
        }
        if obj["datasheet_specs"] is None:
            del obj["datasheet_specs"]

        filename = f"triode_{tube_type.lower().replace(' ', '_')}_datasheet.json"
        out_path = OUT / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
