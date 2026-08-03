#!/usr/bin/env python3
"""
Validate ALL converted JSON test data files.

Checks:
- JSON syntax
- Required fields present
- Data types correct
- Value ranges reasonable
- Consistency (arrays same length, etc.)
- Cross-reference with raw data where possible

Usage:
    python validate_all.py
"""

import json
import sys
from pathlib import Path

CONVERTED_DIR = Path(__file__).parent.parent / "converted"


def validate_file(filepath):
    """Validate a single JSON test data file. Returns (ok, errors, warnings)."""
    errors = []
    warnings = []

    # 1. JSON syntax
    try:
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"], []

    # 2. Required top-level fields
    required = ["tube_type", "topology", "source", "points", "_units"]
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return False, errors, warnings

    # 3. Topology validation
    topology = data["topology"]
    if topology not in ("triode", "pentode"):
        errors.append(f"Invalid topology: '{topology}'")

    # 4. Points validation
    points = data["points"]
    if not isinstance(points, list):
        errors.append("'points' is not a list")
        return False, errors, warnings

    if len(points) == 0:
        errors.append("Empty points array")
        return False, errors, warnings

    # 5. Per-point validation
    for i, pt in enumerate(points):
        if not isinstance(pt, dict):
            errors.append(f"Point {i}: not a dict")
            continue

        # Required fields
        if "ua" not in pt:
            errors.append(f"Point {i}: missing 'ua'")
        if "ug1" not in pt:
            errors.append(f"Point {i}: missing 'ug1'")
        if "ia" not in pt:
            errors.append(f"Point {i}: missing 'ia'")

        # Type checks
        for key in ("ua", "ug1", "ia"):
            if key in pt and not isinstance(pt[key], (int, float)):
                errors.append(f"Point {i}: '{key}' is not numeric: {type(pt[key])}")

        # Value range checks
        if "ua" in pt:
            if pt["ua"] < 0:
                errors.append(f"Point {i}: negative ua={pt['ua']}")
            if pt["ua"] > 2000:
                warnings.append(f"Point {i}: very high ua={pt['ua']}V")

        if "ia" in pt:
            if pt["ia"] < 0:
                errors.append(f"Point {i}: negative ia={pt['ia']}")
            if pt["ia"] > 1500:
                warnings.append(f"Point {i}: very high ia={pt['ia']}mA")

        if "ug1" in pt:
            if pt["ug1"] > 10:
                warnings.append(f"Point {i}: positive ug1={pt['ug1']}V (unusual)")

        # Pentode-specific
        if topology == "pentode":
            if "ug2" not in pt:
                errors.append(f"Point {i}: pentode missing 'ug2'")
            elif not isinstance(pt["ug2"], (int, float)):
                errors.append(f"Point {i}: 'ug2' not numeric")
            elif pt["ug2"] < 0:
                errors.append(f"Point {i}: negative ug2={pt['ug2']}")

            if "ig2" not in pt:
                warnings.append(f"Point {i}: pentode missing 'ig2' field")
            elif pt["ig2"] is not None and not isinstance(pt["ig2"], (int, float)):
                errors.append(f"Point {i}: 'ig2' not numeric or null")

    # 6. Check units consistency
    units = data["_units"]
    if "ia" in units and "mA" not in units["ia"]:
        warnings.append(f"ia units don't mention mA: '{units['ia']}'")

    # 7. Check expected_params if present
    if "expected_params" in data:
        ep = data["expected_params"]
        expected_keys = {"mu", "ex", "kg1", "kp", "kvb"}
        if topology == "pentode":
            expected_keys.add("kg2")
        for key in expected_keys:
            if key not in ep:
                warnings.append(f"expected_params missing '{key}'")
            elif key != "_note" and not isinstance(ep[key], (int, float)):
                errors.append(f"expected_params['{key}'] not numeric")

    # 8. Check published_params if present
    if "published_params" in data:
        pp = data["published_params"]
        for key in ("mu", "ex", "kg1", "kp", "kvb"):
            if key not in pp:
                warnings.append(f"published_params missing '{key}'")

    # 9. Duplicate point check
    seen = set()
    for i, pt in enumerate(points):
        key = (pt.get("ua"), pt.get("ug1"), pt.get("ug2"))
        if key in seen:
            warnings.append(f"Point {i}: duplicate (ua={pt.get('ua')}, ug1={pt.get('ug1')}, ug2={pt.get('ug2')})")
        seen.add(key)

    ok = len(errors) == 0
    return ok, errors, warnings


def main():
    json_files = sorted(CONVERTED_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {CONVERTED_DIR}")
        return 1

    print(f"Validating {len(json_files)} files in {CONVERTED_DIR}\n")

    total_ok = 0
    total_err = 0
    total_warn = 0
    all_valid = True
    summary = []

    for jf in json_files:
        ok, errors, warnings = validate_file(jf)
        n_err = len(errors)
        n_warn = len(warnings)

        # Count points
        try:
            with open(jf, encoding='utf-8') as f:
                data = json.load(f)
            n_pts = len(data.get("points", []))
            topology = data.get("topology", "?")
            tube = data.get("tube_type", "?")
        except Exception:
            n_pts = 0
            topology = "?"
            tube = "?"

        status = "OK" if ok else "FAIL"
        warn_str = f" ({n_warn} warnings)" if n_warn else ""

        if ok and n_warn == 0:
            symbol = "[OK]"
        elif ok:
            symbol = "[~~]"
        else:
            symbol = "[!!]"

        print(f"  {symbol} {jf.name:<45} {topology:>8} {tube:>10} {n_pts:>3}pts  {status}{warn_str}")

        if errors:
            for e in errors:
                print(f"      ERROR: {e}")
        if warnings:
            for w in warnings:
                print(f"      WARN:  {w}")

        if ok:
            total_ok += 1
        else:
            all_valid = False

        total_err += n_err
        total_warn += n_warn

        summary.append({
            "file": jf.name,
            "ok": ok,
            "errors": n_err,
            "warnings": n_warn,
            "points": n_pts,
            "topology": topology,
            "tube": tube,
        })

    print(f"\n{'='*70}")
    print(f"Results: {total_ok}/{len(json_files)} valid, {total_err} errors, {total_warn} warnings")

    # Group by topology
    triodes = [s for s in summary if s["topology"] == "triode"]
    pentodes = [s for s in summary if s["topology"] == "pentode"]
    print(f"\nTriodes:  {len(triodes)} files, {sum(s['points'] for s in triodes)} total points")
    print(f"Pentodes: {len(pentodes)} files, {sum(s['points'] for s in pentodes)} total points")
    print(f"Grand total: {sum(s['points'] for s in summary)} points")

    if not all_valid:
        print("\nSOME FILES HAVE ERRORS — fix before using in tests!")
        return 1

    print("\nAll files valid!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
