#!/usr/bin/env python3
"""
Convert eTracer CSV v2.0 files to unified LM19 JSON format.

Reads CSV files from tests/spice_test_data/raw/etracer_samples/, parses
measurement data via lm19.etracer_import, and writes structured JSON
files for SPICE model fitting tests.

Handles duplicate tube types (picks file with more data points).
Overlap tubes (already in converted/) are skipped by default; use
--include-overlap to re-convert them with eTracer source tag.

Usage:
    python convert_etracer_to_lm19.py [--include-overlap] [--dry-run]
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
SAMPLES_DIR = SCRIPT_DIR.parent / "raw" / "etracer_samples"
OUT_DIR = SCRIPT_DIR.parent / "converted"

# Add lm19_app to path for imports
sys.path.insert(0, str(SCRIPT_DIR.parent.parent.parent))

from lm19.etracer_import import (
    parse_etracer_csv,
    detect_topology,
    etracer_to_lm19_points,
    guess_meta_from_etracer,
)

# ── Heater voltages (from datasheets / .etd files) ──
# Cross-checked against .etd files and datasheets
HEATER_V: dict[str, float] = {
    "10Y": 7.5, "1626": 12.6, "1G4GT": 1.4, "26": 1.5, "30": 2.0,
    "3a-167m": 6.3, "404A_5847": 6.3, "4P1L": 4.0, "50": 7.5,
    "5842Q": 6.3, "6E5P": 6.3, "6E6P": 6.3, "6S19P": 6.3,
    "6S41S": 6.3, "6S45P": 6.3, "6S4P": 6.3, "6Z11P": 6.3,
    "6Z49P": 6.3, "6Z9P": 6.3, "71": 5.0, "801A": 7.5,
    "D3a": 6.3, "E180F": 6.3, "E810F": 6.3, "EC8010": 6.3,
    "EL34": 6.3, "EL84": 6.3, "KC3": 20.0, "KT66": 6.3,
    "RS241": 4.0, "TJ205D": 5.0, "TJ45n": 2.5, "VT52": 6.3,
    "VT67": 2.0,
}

# ── Known Koren published params (for cross-reference) ──
KOREN_PUBLISHED: dict[str, dict] = {
    "EL34": {"mu": 11.0, "ex": 1.35, "kg1": 650, "kp": 60, "kvb": 24,
             "_note": "Pentode params; triode-mode fit differs"},
    "EL84": {"mu": 19.5, "ex": 1.318, "kg1": 1460.8, "kp": 59.1, "kvb": 23.4,
             "_note": "Pentode params from Tube.lib"},
    "KT66": {"mu": 8.8, "ex": 1.35, "kg1": 1225, "kp": 38, "kvb": 16.3,
             "_note": "Pentode params from Tube.lib"},
}


def build_json(
    tube_type: str,
    topology: str,
    points: list[dict],
    csv_file: str,
    etd_file: str,
    vh: float,
) -> dict:
    """Build LM19 JSON output for one tube."""
    topo_label = topology.replace("_", " ")
    result: dict = {
        "_comment": f"{tube_type} {topo_label} — eTracer measurement from vt52.com",
        "tube_type": tube_type,
        "topology": "triode" if topology in ("triode", "triode_connected") else "pentode",
        "ug2_mode": topology,
        "source": "eTracer / vt52.com",
        "url": "http://www.vt52.com/etracer-files",
    }

    if tube_type in KOREN_PUBLISHED:
        result["published_params"] = {
            **KOREN_PUBLISHED[tube_type],
            "_note": KOREN_PUBLISHED[tube_type].get("_note", "Koren published values"),
        }

    # Compute stats
    ug1_vals = sorted(set(round(p["ug1"], 1) for p in points))
    ua_range = (min(p["ua"] for p in points), max(p["ua"] for p in points))

    result["csv_file"] = csv_file
    if etd_file:
        result["etd_file"] = Path(etd_file).name
    result["heater_v"] = vh

    json_points = []
    for p in points:
        pt: dict = {
            "ua": p["ua"],
            "ug1": p["ug1"],
            "ia": p["ia"],
        }
        if topology in ("triode_connected", "pentode"):
            pt["ug2"] = p["ug2"]
            if p.get("ig2", 0) > 0.01:
                pt["ig2"] = p["ig2"]
        json_points.append(pt)

    result["points"] = json_points

    result["_units"] = {"ua": "V", "ug1": "V (negative)", "ia": "mA"}
    if topology in ("triode_connected", "pentode"):
        result["_units"]["ug2"] = "V"

    result["_stats"] = {
        "n_points": len(json_points),
        "n_curves": len(ug1_vals),
        "ua_range": list(ua_range),
        "ug1_values": ug1_vals,
    }

    return result


def main() -> int:
    include_overlap = "--include-overlap" in sys.argv
    dry_run = "--dry-run" in sys.argv

    if not SAMPLES_DIR.is_dir():
        print(f"ERROR: samples directory not found: {SAMPLES_DIR}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect existing tube types in converted/
    existing_types: set[str] = set()
    for f in OUT_DIR.iterdir():
        if f.suffix == ".json":
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                existing_types.add(d.get("tube_type", "").upper())
            except (json.JSONDecodeError, KeyError):
                pass

    # Parse all CSV files, group by tube_type (pick best)
    candidates: dict[str, dict] = {}
    for csv_path in sorted(SAMPLES_DIR.glob("*.csv")):
        parsed = parse_etracer_csv(str(csv_path))
        meta = guess_meta_from_etracer(str(csv_path), parsed)
        tube_type = meta["tube_type"]
        topology = detect_topology(parsed)
        vh = HEATER_V.get(tube_type, 0.0)
        points = etracer_to_lm19_points(parsed, vh=vh)
        n_pts = len(points)

        key = tube_type.upper()
        if key in candidates and candidates[key]["n_pts"] >= n_pts:
            continue  # keep the one with more points

        candidates[key] = {
            "tube_type": tube_type,
            "topology": topology,
            "points": points,
            "n_pts": n_pts,
            "csv_file": csv_path.name,
            "etd_file": parsed.get("etd_file", ""),
            "vh": vh,
        }

    total_files = 0
    total_points = 0
    skipped_overlap = 0

    for key in sorted(candidates.keys()):
        c = candidates[key]
        tube_type = c["tube_type"]

        if tube_type.upper() in existing_types and not include_overlap:
            skipped_overlap += 1
            print(f"  SKIP {tube_type:15s} (already in converted/)")
            continue

        topo_prefix = "triode" if c["topology"] in ("triode", "triode_connected") else "pentode"
        out_name = f"{topo_prefix}_{tube_type}_etracer.json"
        out_path = OUT_DIR / out_name

        json_data = build_json(
            tube_type=tube_type,
            topology=c["topology"],
            points=c["points"],
            csv_file=c["csv_file"],
            etd_file=c["etd_file"],
            vh=c["vh"],
        )

        # Validate
        warnings = []
        for i, pt in enumerate(json_data["points"]):
            if pt.get("ia", 0) < -0.1:
                warnings.append(f"Point {i}: negative ia={pt['ia']}")

        if dry_run:
            print(f"  DRY  {tube_type:15s} -> {out_name} "
                  f"({c['n_pts']} pts, {c['topology']})")
        else:
            out_path.write_text(
                json.dumps(json_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            total_files += 1
            total_points += c["n_pts"]
            print(f"  OK   {tube_type:15s} -> {out_name} "
                  f"({c['n_pts']} pts, {c['topology']})")

        for w in warnings:
            print(f"       WARN: {w}")

    print(f"\n{'=' * 60}")
    if dry_run:
        print(f"Dry run complete. Would convert {len(candidates) - skipped_overlap} tubes.")
    else:
        print(f"Converted: {total_files} tube types, {total_points} total points")
    print(f"Skipped (overlap): {skipped_overlap}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
