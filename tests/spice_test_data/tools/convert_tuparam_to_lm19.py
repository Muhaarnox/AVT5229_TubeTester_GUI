#!/usr/bin/env python3
"""
Convert all Koren Tuparam .m data files to unified LM19 JSON format.

Reads raw Matlab-style .m files, parses Vp/Vg/Vs/Idata arrays,
and writes structured JSON files with metadata and validation.

Usage:
    python convert_tuparam_to_lm19.py
"""

import json
import re
import sys
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR.parent / "raw" / "libs" / "koren" / "Tuparam"
OUT_DIR = SCRIPT_DIR.parent / "converted"

# Published Koren tube.lib reference parameters (from Tube.lib / Tube97.lib)
PUBLISHED_PARAMS = {
    "12AX7": {"mu": 100, "ex": 1.4, "kg1": 1060, "kp": 600, "kvb": 300},
    "12AU7": {"mu": 21.5, "ex": 1.3, "kg1": 1180, "kp": 84, "kvb": 300},
    "12AT7": {"mu": 60, "ex": 1.35, "kg1": 460, "kp": 300, "kvb": 300},
    "6DJ8":  {"mu": 28, "ex": 1.3, "kg1": 330, "kp": 320, "kvb": 300},
    "6SN7":  {"mu": 21.17, "ex": 1.33, "kg1": 232.6, "kp": 156.08, "kvb": 7.3},
    "6550":  {"mu": 7.9, "ex": 1.35, "kg1": 890, "kg2": 4200, "kp": 60, "kvb": 24},
    "6L6GC": {"mu": 8.7, "ex": 1.35, "kg1": 1460, "kg2": 4500, "kp": 48, "kvb": 12},
    "KT88":  {"mu": 8.8, "ex": 1.35, "kg1": 730, "kg2": 4800, "kp": 32, "kvb": 16},
    "EL34":  {"mu": 11.0, "ex": 1.35, "kg1": 650, "kg2": 4200, "kp": 60, "kvb": 24},
}

# Tube type aliases for matching published params
ALIASES = {
    "12AX7A": "12AX7",
    "12AU7A": "12AU7",
    "6550A": "6550",
    "6550C": "6550",
    "6L6GB": "6L6GC",  # same family
    "7025": "12AX7",
}

# Expected fit results from Tuparam (where known from comments in .m files)
EXPECTED_FIT = {
    "12AX7AMitch": {
        "mu": 101.24, "ex": 1.267, "kg1": 1002.9,
        "kp": 699.73, "kvb": 300.0, "vct": 0.0,
        "_note": "Tuparam fit result"
    },
    "7025": {
        "mu": 103.44, "ex": 1.245, "kg1": 1515.4,
        "kp": 903.23, "kvb": 99.2, "vct": 0.50,
        "_note": "Tuparam fit result — has contact potential VCT=0.5V"
    },
    "6550": {
        "mu": 8.45, "ex": 1.247, "kg1": 642.5,
        "kp": 48.92, "kg2": 4500, "kvb": 20.9, "vct": 0.0,
        "_note": "Tuparam fit result. KG2 was NOT optimized — kept at 4500 (user-provided)."
    },
}


def parse_matlab_array(text):
    """Parse Matlab array: [1 2 3] or [1, 2, 3] or [.001 .002]."""
    # Remove brackets
    text = text.strip()
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]
    # Split on whitespace or commas
    parts = re.split(r'[,\s]+', text.strip())
    values = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Handle Matlab notation: .123 → 0.123, -.123 → -0.123
        if p.startswith('.'):
            p = '0' + p
        elif p.startswith('-.'):
            p = '-0' + p[1:]
        values.append(float(p))
    return values


def parse_m_file(filepath):
    """Parse a Tuparam .m data file and extract tube data."""
    text = filepath.read_text(encoding='utf-8', errors='replace')
    lines = text.split('\n')

    data = {
        'TubeName': None,
        'Source': None,
        'Vp': None,
        'Vg': None,
        'Vs': None,
        'Idata': None,
        'VCT': 0.0,
        'MU': None,
        'KVB': None,
        'KG2': None,
        'KP': None,
        'Vpmax': None,
        'VGP': None,
        'VGTRI': None,
        'NOPT': None,
    }

    # Join continuation lines (lines ending with ...)
    joined = []
    for line in lines:
        line = line.rstrip()
        # Strip comments (% to end of line), but keep strings
        comment_pos = line.find('%')
        if comment_pos >= 0:
            line = line[:comment_pos].rstrip()
        if not line:
            continue
        if joined and joined[-1].endswith('...'):
            joined[-1] = joined[-1][:-3] + ' ' + line
        else:
            joined.append(line)

    # Split multi-statement lines on ';' (Matlab allows a; b; c on one line)
    statements = []
    for line in joined:
        parts = line.split(';')
        for p in parts:
            p = p.strip()
            if p:
                statements.append(p)

    for line in statements:

        # TubeName = '6550';
        m = re.match(r"TubeName\s*=\s*'([^']+)'", line)
        if m:
            data['TubeName'] = m.group(1)
            continue

        # Source = 'Tom Mitchell';
        m = re.match(r"Source\s*=\s*'([^']*(?:''[^']*)*)'", line)
        if m:
            data['Source'] = m.group(1).replace("''", "'")
            continue

        # VCT = .5;  or VCT = 0.5
        m = re.match(r"VCT\s*=\s*([.\d\-]+)", line)
        if m:
            val = m.group(1)
            if val.startswith('.'):
                val = '0' + val
            data['VCT'] = float(val)
            continue

        # MU = 10;
        m = re.match(r"MU\s*=\s*([\d.\-]+)", line)
        if m:
            data['MU'] = float(m.group(1))
            continue

        # KVB = 20;
        m = re.match(r"KVB\s*=\s*([\d.\-]+)", line)
        if m:
            data['KVB'] = float(m.group(1))
            continue

        # KG2 = 4500;
        m = re.match(r"KG2\s*=\s*([\d.\-]+)", line)
        if m:
            data['KG2'] = float(m.group(1))
            continue

        # KP = 100;
        m = re.match(r"KP\s*=\s*([\d.\-]+)", line)
        if m:
            data['KP'] = float(m.group(1))
            continue

        # NOPT = 4;
        m = re.match(r"NOPT\s*=\s*([\d]+)", line)
        if m:
            data['NOPT'] = int(m.group(1))
            continue

        # Vpmax = 500;
        m = re.match(r"Vpmax\s*=\s*([\d.\-]+)", line)
        if m:
            data['Vpmax'] = float(m.group(1))
            continue

        # Vp = [40 100 300 ...];
        m = re.match(r"Vp\s*=\s*(\[.+\])", line)
        if m:
            data['Vp'] = parse_matlab_array(m.group(1))
            continue

        # Vg = [0 0 0 -10 ...];
        m = re.match(r"Vg\s*=\s*(\[.+\])", line)
        if m:
            data['Vg'] = parse_matlab_array(m.group(1))
            continue

        # Vs = [300 300 300 ...]; or Vs = 250;
        m = re.match(r"Vs\s*=\s*(\[.+\])", line)
        if m:
            data['Vs'] = parse_matlab_array(m.group(1))
            continue
        m = re.match(r"Vs\s*=\s*([\d.\-]+)", line)
        if m:
            data['Vs'] = float(m.group(1))  # scalar
            continue

        # Idata = [.307 .405 ...];
        m = re.match(r"Idata\s*=\s*(\[.+\])", line)
        if m:
            data['Idata'] = parse_matlab_array(m.group(1))
            continue

    return data


def validate_parsed(data, filename):
    """Validate parsed data, return list of errors."""
    errors = []
    if data['TubeName'] is None:
        errors.append("Missing TubeName")
    if data['Vp'] is None:
        errors.append("Missing Vp array")
    if data['Vg'] is None:
        errors.append("Missing Vg array")
    if data['Idata'] is None:
        errors.append("Missing Idata array")

    if data['Vp'] and data['Vg']:
        if len(data['Vp']) != len(data['Vg']):
            errors.append(f"Vp ({len(data['Vp'])}) and Vg ({len(data['Vg'])}) length mismatch")
    if data['Vp'] and data['Idata']:
        if len(data['Vp']) != len(data['Idata']):
            errors.append(f"Vp ({len(data['Vp'])}) and Idata ({len(data['Idata'])}) length mismatch")

    if data['Vs'] is not None and isinstance(data['Vs'], list):
        if data['Vp'] and len(data['Vs']) != len(data['Vp']):
            errors.append(f"Vs ({len(data['Vs'])}) and Vp ({len(data['Vp'])}) length mismatch")

    # Check for non-negative plate voltages
    if data['Vp']:
        for v in data['Vp']:
            if v < 0:
                errors.append(f"Negative Vp value: {v}")

    # Check for non-negative Idata (currents in Amps)
    if data['Idata']:
        for i_val in data['Idata']:
            if i_val < 0:
                errors.append(f"Negative Idata value: {i_val}")

    return errors


def determine_topology(data):
    """Determine if tube is triode or pentode based on Vs presence."""
    return "pentode" if data['Vs'] is not None else "triode"


def expand_vs(vs, n):
    """Expand scalar Vs to array of length n."""
    if vs is None:
        return None
    if isinstance(vs, (int, float)):
        return [vs] * n
    return vs


def build_json(data, m_filename):
    """Build the output JSON structure."""
    topology = determine_topology(data)
    tube_name = data['TubeName']
    source = data['Source'] or "Unknown"
    vct = data['VCT']
    n = len(data['Vp'])

    # Expand Vs if scalar
    vs_arr = expand_vs(data['Vs'], n)

    # Build output file name
    stem = m_filename.stem  # e.g., "6550A", "12AX7AMitch"
    out_name = f"{topology}_{tube_name}_{stem}_tuparam.json"
    # Simplify: if stem starts with tube name, avoid duplication
    if stem.lower().startswith(tube_name.lower()):
        out_name = f"{topology}_{stem}_tuparam.json"

    # Build comment
    comment = f"{tube_name} {topology} test data from Norman Koren's Tuparam ({source})"

    result = {
        "_comment": comment,
        "tube_type": tube_name,
        "topology": topology,
        "source": f"Koren Tuparam / {source}",
        "url": "https://www.normankoren.com/Audio/Tube_params.html",
    }

    # Add aliases
    canonical = ALIASES.get(tube_name, tube_name)
    if canonical != tube_name:
        result["aliases"] = [canonical]

    # Add expected params if known
    if stem in EXPECTED_FIT:
        result["expected_params"] = EXPECTED_FIT[stem]

    # Add published params
    if canonical in PUBLISHED_PARAMS:
        result["published_params"] = {
            **PUBLISHED_PARAMS[canonical],
            "_note": "Koren tube.lib published values"
        }

    # Add initial guess params from .m file (useful for fitter seed values)
    initial_guess = {}
    if data['MU'] is not None:
        initial_guess["mu"] = data['MU']
    if data['KVB'] is not None:
        initial_guess["kvb"] = data['KVB']
    if data['KG2'] is not None:
        initial_guess["kg2"] = data['KG2']
    if data['KP'] is not None:
        initial_guess["kp"] = data['KP']
    if vct != 0:
        initial_guess["vct"] = vct
    if data['NOPT'] is not None:
        initial_guess["nopt"] = data['NOPT']
    if initial_guess:
        result["initial_guess"] = {
            **initial_guess,
            "_note": "Initial values from .m file (used as seed for optimizer)"
        }

    # Build points
    points = []
    for i in range(n):
        pt = {
            "ua": data['Vp'][i],
            "ug1": data['Vg'][i],
        }

        if topology == "pentode" and vs_arr:
            pt["ug2"] = vs_arr[i]
            pt["ig2"] = None  # Not measured in Tuparam
            # Mark triode-connected points using Koren's criterion:
            # triode-connected = Vs varies from primary Vs AND Vs == Vp
            primary_vs = vs_arr[0]
            is_triode = (abs(vs_arr[i] - data['Vp'][i]) < 0.1 and
                         abs(vs_arr[i] - primary_vs) > 0.1)
            pt["_triode_connected"] = is_triode

        # Convert Idata from Amps to mA
        pt["ia"] = round(data['Idata'][i] * 1000, 4)

        points.append(pt)

    result["points"] = points

    # Units
    units = {
        "ua": "V",
        "ug1": "V (negative)",
        "ia": "mA"
    }
    if topology == "pentode":
        units["ug2"] = "V"
        units["ig2"] = "mA (null = not measured)"

    result["_units"] = units

    # Notes
    notes = []
    if vct != 0:
        notes.append(f"Contact potential VCT = {vct}V applied to Vg in model equations.")
    if topology == "pentode" and vs_arr:
        primary_vs = vs_arr[0]
        tri_indices = [i for i in range(n)
                       if abs(vs_arr[i] - data['Vp'][i]) < 0.1
                       and abs(vs_arr[i] - primary_vs) > 0.1]
        diff_vs_indices = [i for i in range(n)
                           if abs(vs_arr[i] - primary_vs) > 0.1
                           and abs(vs_arr[i] - data['Vp'][i]) >= 0.1]
        if tri_indices:
            notes.append(f"Points at indices {tri_indices} are triode-connected (Ug2 = Ua).")
        if diff_vs_indices:
            notes.append(f"Points at indices {diff_vs_indices} have non-primary Ug2 (different screen voltage).")
        notes.append("Ig2 was not measured — Kg2 uses user-provided initial value.")
    if notes:
        result["_notes"] = notes

    return out_name, result


def validate_json(data, filename):
    """Validate a converted JSON structure. Returns list of warnings."""
    warnings = []

    # Check required fields
    for field in ("tube_type", "topology", "source", "points", "_units"):
        if field not in data:
            warnings.append(f"Missing required field: {field}")

    if "points" not in data:
        return warnings

    points = data["points"]
    if not points:
        warnings.append("Empty points array")
        return warnings

    topology = data.get("topology", "triode")

    for i, pt in enumerate(points):
        # Check required point fields
        if "ua" not in pt:
            warnings.append(f"Point {i}: missing 'ua'")
        if "ug1" not in pt:
            warnings.append(f"Point {i}: missing 'ug1'")
        if "ia" not in pt:
            warnings.append(f"Point {i}: missing 'ia'")

        # Value range checks
        if "ua" in pt and pt["ua"] < 0:
            warnings.append(f"Point {i}: negative ua={pt['ua']}")
        if "ia" in pt and pt["ia"] < 0:
            warnings.append(f"Point {i}: negative ia={pt['ia']}")
        if "ug1" in pt and pt["ug1"] > 5:
            warnings.append(f"Point {i}: suspiciously positive ug1={pt['ug1']}")

        if topology == "pentode":
            if "ug2" not in pt:
                warnings.append(f"Point {i}: pentode missing 'ug2'")
            elif pt["ug2"] < 0:
                warnings.append(f"Point {i}: negative ug2={pt['ug2']}")

        # Ia sanity check (mA): typical range 0.01 to 500+ mA
        if "ia" in pt and pt["ia"] > 1000:
            warnings.append(f"Point {i}: very high ia={pt['ia']} mA")

    return warnings


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m_files = sorted(RAW_DIR.glob("*.m"))
    # Exclude non-data files (TuParam.m, TuCalc.m, Optube.m)
    skip = {"TuParam.m", "TuCalc.m", "Optube.m"}
    m_files = [f for f in m_files if f.name not in skip]

    print(f"Found {len(m_files)} data files in {RAW_DIR}\n")

    all_ok = True
    converted = []

    for mf in m_files:
        print(f"--- {mf.name} ---")

        # Parse
        data = parse_m_file(mf)
        parse_errors = validate_parsed(data, mf.name)
        if parse_errors:
            print(f"  PARSE ERRORS:")
            for e in parse_errors:
                print(f"    - {e}")
            all_ok = False
            continue

        topology = determine_topology(data)
        n = len(data['Vp'])
        vct = data['VCT']
        print(f"  Tube: {data['TubeName']}  Topology: {topology}  Points: {n}  VCT: {vct}")
        print(f"  Source: {data['Source']}")

        # Build JSON
        out_name, json_data = build_json(data, mf)

        # Validate
        warnings = validate_json(json_data, out_name)
        if warnings:
            print(f"  WARNINGS:")
            for w in warnings:
                print(f"    - {w}")

        # Write
        out_path = OUT_DIR / out_name
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"  -> {out_name} ({n} points)")

        converted.append({
            "file": out_name,
            "tube": data['TubeName'],
            "topology": topology,
            "points": n,
            "vct": vct,
            "warnings": len(warnings),
        })

    print(f"\n{'='*60}")
    print(f"Converted: {len(converted)} / {len(m_files)} files")
    print(f"\nSummary:")
    print(f"  Triodes:  {sum(1 for c in converted if c['topology'] == 'triode')}")
    print(f"  Pentodes: {sum(1 for c in converted if c['topology'] == 'pentode')}")
    print(f"  With VCT: {sum(1 for c in converted if c['vct'] != 0)}")
    print(f"  Warnings: {sum(c['warnings'] for c in converted)}")

    if not all_ok:
        print("\nSOME FILES HAD ERRORS — check output above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
