"""Parse TubeLib.inc and add Reefman params + capacitances to tube_params.json.

One-shot script:
    cd d:/work/AVR/AVT5229/lm19_app
    py tools/parse_tubelib.py              # dry-run (shows what would change)
    py tools/parse_tubelib.py --apply      # writes tube_params.json

Parses .SUBCKT blocks from TubeLib.inc, extracts model type and parameters,
and merges them into the existing tube_params.json entries.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lm19.io_utils import write_json  # noqa: E402  (needs ROOT on sys.path)

TUBELIB = ROOT / "external_sources" / "data" / "TubeLib.inc"
TUBE_PARAMS = ROOT / "config" / "tube_params.json"

# Model types we care about (skip HepthodeD, Penthodeg3DE, DiodeK)
SUPPORTED_TYPES = {
    "BTetrodeD", "BTetrodeDE",
    "PenthodeD", "PenthodeDE",
    "PenthodeVD", "PenthodeVDE",
    "TriodeK",
}

# Types with secondary emission params
SEC_EMISSION_TYPES = {"BTetrodeD", "BTetrodeDE"}

# Types with variable-mu params
VARIABLE_MU_TYPES = {"PenthodeVD", "PenthodeVDE"}


def parse_tubelib(filepath: Path) -> list:
    """Parse TubeLib.inc, return list of (subckt_name, model_type, params_dict)."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    results = []

    # Split into .SUBCKT ... .ENDS blocks
    blocks = re.split(r'\.ENDS\b', text)

    for block in blocks:
        m = re.search(r'\.SUBCKT\s+(\S+)\s+(.+?)$', block, re.MULTILINE)
        if not m:
            continue
        subckt_name = m.group(1)

        # Find X1 line and all continuation lines (starting with +)
        x1_match = re.search(r'^(X1\s+.+)$', block, re.MULTILINE)
        if not x1_match:
            continue

        # Collect X1 line + continuation lines
        x1_start = x1_match.start()
        lines_after = block[x1_start:].split('\n')
        combined = lines_after[0]
        for line in lines_after[1:]:
            stripped = line.strip()
            if stripped.startswith('+'):
                combined += ' ' + stripped[1:]  # remove leading +
            elif stripped.startswith('*') or stripped == '':
                continue
            else:
                break

        # Determine model type from X1 line (check longer names first)
        model_type = None
        for mt in sorted(SUPPORTED_TYPES, key=len, reverse=True):
            # Word boundary: "BTetrodeDE" should not match "BTetrodeD"
            if re.search(r'\b' + re.escape(mt) + r'\b', combined):
                model_type = mt
                break
        if model_type is None:
            continue  # skip HepthodeD, Penthodeg3DE, DiodeK, etc.

        # Parse all key=value pairs
        params = _parse_kv(combined)
        results.append((subckt_name, model_type, params))

    return results


def _parse_kv(text: str) -> dict:
    """Extract key=value pairs from a SPICE parameter string."""
    params = {}
    # Match patterns like: MU= 23.36, kG1= 117.4, Sc=.38E-01, CCG1=10.8P
    for m in re.finditer(r'(\w+)\s*=\s*([^,\s;]+)', text):
        key = m.group(1)
        val_str = m.group(2).rstrip('pPfF')  # strip pF suffix from capacitances
        try:
            val = float(val_str)
            params[key] = val
        except ValueError:
            pass
    return params


def extract_reefman_params(model_type: str, params: dict) -> dict:
    """Extract reefman model params from parsed key-value pairs."""
    result = {"type": model_type}

    # Core params (all types)
    key_map = {
        "MU": "mu", "EX": "ex",
        "kG1": "kg1", "KG1": "kg1",
        "kG2": "kg2", "KG2": "kg2",
        "KP": "kp",
        "kVB": "kvb", "KVB": "kvb",
        "als": "als", "be": "be",
    }
    for src, dst in key_map.items():
        if src in params and dst not in result:
            result[dst] = params[src]

    # Recover A from Aokg1: A = Aokg1 * kg1
    if "Aokg1" in params and "kg1" in result:
        result["A"] = params["Aokg1"] * result["kg1"]

    # Secondary emission (BTetrode only)
    if model_type in SEC_EMISSION_TYPES:
        for key in ("Sc", "ap", "w", "nu", "lam"):
            if key in params:
                result[key] = params[key]

    # Variable-mu (V-variants)
    if model_type in VARIABLE_MU_TYPES:
        if "MUb" in params:
            result["mu_b"] = params["MUb"]
        if "EXb" in params:
            result["ex_b"] = params["EXb"]
        if "svar" in params:
            result["svar"] = params["svar"]

    return result


def extract_caps_pentode(params: dict) -> dict:
    """Extract pentode capacitances (5 values)."""
    caps = {}
    cap_map = {
        "CCG1": "ccg1", "CCG2": "ccg2",
        "CPG1": "cpg1", "CG1G2": "cg1g2",
        "CCP": "ccp",
    }
    for src, dst in cap_map.items():
        if src in params:
            caps[dst] = params[src]
    return caps


def extract_caps_triode(params: dict) -> dict:
    """Extract triode capacitances (3 values): ccg, cgp, ccp."""
    caps = {}
    cap_map = {"CCG": "ccg", "CGP": "cgp", "CCP": "ccp"}
    for src, dst in cap_map.items():
        if src in params:
            caps[dst] = params[src]
    return caps


# Combo tube suffix mapping: TubeLib suffix -> tube_params.json suffix
COMBO_SUFFIXES = {
    "_L": "_pentode",   # pentode section (European convention)
    "_F": "_pentode",   # pentode section (American convention)
    "_C": "_triode",    # triode section
}


def build_name_index(entries: list) -> dict:
    """Map normalized tube names to TubeLib entries.

    Returns dict: base_name -> (subckt_name, model_type, params)
    """
    index = {}
    for subckt_name, model_type, params in entries:
        is_triode = model_type == "TriodeK"

        base = subckt_name
        if base.endswith("_triode"):
            base = base[:-7]

        # Handle combo tube suffixes: ECL82_L -> ECL82_pentode
        combo_mapped = None
        for suffix, replacement in COMBO_SUFFIXES.items():
            if base.endswith(suffix):
                combo_mapped = base[:-len(suffix)] + replacement
                break

        key = (base, "triode" if is_triode else "pentode")
        index[key] = (subckt_name, model_type, params)

        # Also index under combo-mapped name
        if combo_mapped:
            key2 = (combo_mapped, "triode" if is_triode else "pentode")
            index[key2] = (subckt_name, model_type, params)

    return index


def match_to_tube_params(index: dict, tube_params: dict) -> dict:
    """Match TubeLib entries to existing tube_params.json entries.

    Returns dict: tube_name -> {reefman: ..., caps_pF: ...}
    """
    tubes = tube_params.get("tubes", {})
    updates = {}

    # Build alias -> primary_name mapping
    alias_map = {}
    for name, entry in tubes.items():
        if name.startswith("_"):
            continue
        alias_map[name.upper()] = name
        for alias in entry.get("aliases", []):
            alias_map[alias.upper()] = name

    for (base_name, variant), (subckt_name, model_type, params) in index.items():
        # Try to find matching tube in tube_params.json
        primary = alias_map.get(base_name.upper())
        if primary is None:
            # Try without trailing letters (e.g. "6L6GC" -> "6L6")
            stripped = re.sub(r'[A-Z]+$', '', base_name)
            if stripped != base_name:
                primary = alias_map.get(stripped.upper())
        if primary is None:
            continue

        entry = tubes[primary]
        topology = entry.get("topology", "triode")

        if primary not in updates:
            updates[primary] = {}

        if variant == "pentode" and topology == "pentode":
            # Pentode model -> reefman params
            updates[primary]["reefman"] = extract_reefman_params(model_type, params)
            caps = extract_caps_pentode(params)
            if caps:
                updates[primary]["caps_pF_pentode"] = caps
        elif variant == "triode":
            # Triode or triode-strapped -> caps only (Koren triode params already exist)
            if topology == "triode":
                caps = extract_caps_triode(params)
                if caps:
                    updates[primary]["caps_pF_triode"] = caps
            else:
                # Triode-strapped pentode: update caps if we don't have pentode caps
                caps = extract_caps_triode(params)
                if caps and "caps_pF_pentode" not in updates.get(primary, {}):
                    updates[primary]["caps_pF_triode_strapped"] = caps

    return updates


def apply_updates(tube_params: dict, updates: dict, dry_run: bool = True) -> None:
    """Apply updates to tube_params dict."""
    tubes = tube_params.get("tubes", {})

    for name, upd in sorted(updates.items()):
        entry = tubes[name]

        # Add reefman params
        if "reefman" in upd:
            reefman = upd["reefman"]
            reefman["_source"] = "TubeLib.inc (ExtractModel, Reefman 2016)"
            if dry_run:
                print(f"  {name}: + reefman ({reefman['type']}, "
                      f"mu={reefman.get('mu', '?')}, kg1={reefman.get('kg1', '?')})")
            else:
                entry["reefman"] = reefman

        # Update capacitances
        for caps_key in ("caps_pF_pentode", "caps_pF_triode", "caps_pF_triode_strapped"):
            if caps_key in upd:
                caps = upd[caps_key]
                # Only update if at least one non-zero value
                if any(v > 0 for v in caps.values()):
                    if dry_run:
                        print(f"  {name}: + caps_pF ({caps_key.split('_', 2)[-1]}): {caps}")
                    else:
                        # Merge: only add keys not already present
                        existing = entry.get("caps_pF", {})
                        for ck, cv in caps.items():
                            if ck not in existing:
                                existing[ck] = cv
                        entry["caps_pF"] = existing

        # Add RGI if not present
        if "reefman" in upd and "RGI" not in entry:
            rgi = upd["reefman"].get("RGI", 2000)
            # RGI is in the raw params, not in extracted reefman
            pass  # already in tube_params as "rgi"


def main():
    dry_run = "--apply" not in sys.argv

    if not TUBELIB.exists():
        print(f"TubeLib.inc not found: {TUBELIB}")
        return 1

    if not TUBE_PARAMS.exists():
        print(f"tube_params.json not found: {TUBE_PARAMS}")
        return 1

    # Parse TubeLib.inc
    entries = parse_tubelib(TUBELIB)
    print(f"Parsed {len(entries)} subcircuits from TubeLib.inc")

    # Count by type
    type_counts = {}
    for _, mt, _ in entries:
        type_counts[mt] = type_counts.get(mt, 0) + 1
    for mt, count in sorted(type_counts.items()):
        print(f"  {mt}: {count}")

    # Load tube_params.json
    with open(TUBE_PARAMS, "r", encoding="utf-8") as f:
        tube_params = json.load(f)

    # Build index and match
    index = build_name_index(entries)
    updates = match_to_tube_params(index, tube_params)

    print(f"\nMatched {len(updates)} tubes to tube_params.json:")
    if not updates:
        print("  (none)")
        return 0

    # Show unmatched pentode/tetrode entries
    matched_bases = set()
    tubes = tube_params.get("tubes", {})
    alias_map = {}
    for name, entry in tubes.items():
        if name.startswith("_"):
            continue
        alias_map[name.upper()] = name
        for alias in entry.get("aliases", []):
            alias_map[alias.upper()] = name

    for (base_name, variant), _ in index.items():
        if alias_map.get(base_name.upper()):
            matched_bases.add(base_name)

    unmatched = []
    for (base_name, variant), (subckt_name, model_type, _) in index.items():
        if base_name not in matched_bases and variant == "pentode":
            unmatched.append((base_name, model_type))

    if unmatched:
        print(f"\nUnmatched TubeLib pentodes (not in tube_params.json):")
        for name, mt in sorted(set(unmatched)):
            print(f"  {name} ({mt})")

    # Apply or show
    if dry_run:
        print(f"\nDry run — changes that would be applied:")
        apply_updates(tube_params, updates, dry_run=True)
        print(f"\nRun with --apply to write changes.")
    else:
        apply_updates(tube_params, updates, dry_run=False)
        write_json(TUBE_PARAMS, tube_params)
        print(f"\nWrote {TUBE_PARAMS}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
