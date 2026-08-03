"""Extract lamp definitions from LM19 firmware into config/lamps.json.

BOOTSTRAP-ONLY: this tool is meant for the INITIAL export. After it,
config/lamps.json is curated by hand (manually added lamps such as
GU-29, edited fields) — a re-run clobbers that curation. Therefore,
with an existing lamps.json the run is FORBIDDEN without an explicit
--force-overwrite; on a forced run, manual lamps (types absent from
the firmware) are carried over, but manual edits to firmware-derived
lamps are LOST.
"""

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lm19.io_utils import write_json  # noqa: E402  (needs ROOT on sys.path)

FIRMWARE = ROOT /".." /"LM19 firmware" / "TTesterLCD.c"
OUT = ROOT / "config" / "lamps.json"
LIMITS = ROOT / "config" / "lamp_limits.json"
NAME_RE = re.compile(r"^(?P<raw_type>.+?)(?P<socket>[A-Z])(?P<anode>[0-9])(?P<warmup>[0-9])$")
# ML-072: firmware service placeholder records are not lamps. PwrSupply
# used to be filtered only when a manual 6P18P existed, ForFutUse —
# never.
_PLACEHOLDER_TYPES = ("PwrSupply", "ForFutUse")


def _parse_name_meta(name: str) -> tuple[str, str, int, int]:
    """
    Parse firmware tube name tail: ...<socket><anode><warmup_min>.

    Examples:
      ECC83_G11 -> type=ECC83, socket=G, anode_sel=1, warmup_s=60
      EL84__B02 -> type=EL84, socket=B, anode_sel=0, warmup_s=120
      PCL86TJ12 -> type=PCL86T, socket=J, anode_sel=1, warmup_s=120
    """
    m = NAME_RE.match(name)
    if not m:
        return name.rstrip("_"), "", 0, 120

    raw_type = m.group("raw_type").rstrip("_")
    socket = m.group("socket")
    anode_sel = int(m.group("anode"))
    warmup_min = int(m.group("warmup"))
    warmup_s = warmup_min * 60
    return raw_type, socket, anode_sel, warmup_s


def parse_lamprom() -> list[dict]:
    text = FIRMWARE.read_text(encoding="utf-8", errors="ignore")
    start = text.find("lamprom[FLAMP] =")
    if start < 0:
        raise RuntimeError("lamprom array not found")
    block = text[start:]
    end = block.find("};")
    if end < 0:
        raise RuntimeError("lamprom block end not found")
    block = block[:end]

    by_type: dict[str, list[dict]] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        if "{" not in line or "}" not in line:
            continue
        content = line[line.find("{") + 1 : line.rfind("}")]
        tokens = [t.strip() for t in content.split(",") if t.strip()]
        if len(tokens) < 19:
            continue
        name_tokens = tokens[:9]
        name = "".join(t.strip().strip("'") for t in name_tokens)
        values = [int(t) for t in tokens[9:19]]
        (
            uhdef,
            ihdef,
            ug1def,
            uadef,
            iadef,
            ug2def,
            ig2def,
            sdef,
            rdef,
            kdef,
        ) = values
        base_type, socket, anode_sel, warmup_s = _parse_name_meta(name)
        entry = {
            "type": base_type,
            "socket": socket,
            "anodes": 2 if anode_sel in (1, 2) else 1,
            "warmup_s": warmup_s,
            "_anode_sel": anode_sel,
            "topology": "triode" if ug2def == 0 else "pentode",
            "uh": uhdef / 10.0,
            "ih": ihdef / 100.0,
            "ug1": -ug1def / 10.0,
            "ua": float(uadef),
            # Firmware stores Ia in deci-mA.
            "ia": iadef / 10.0,
            "ug2": float(ug2def),
            # Firmware stores Ig2 in centi-mA.
            "ig2": ig2def / 100.0,
            # SRK values are stored with one decimal place.
            "s": sdef / 10.0,
            "r": rdef / 10.0,
            "k": kdef / 10.0,
        }
        by_type.setdefault(base_type, []).append(entry)

    merged: list[dict] = []
    numeric_keys = ("uh", "ih", "ug1", "ua", "ia", "ug2", "ig2", "s", "r", "k")
    for base_type, items in by_type.items():
        if len(items) == 1:
            one = dict(items[0])
            if int(one.get("anodes", 1)) > 1:
                sel = int(one.get("_anode_sel", 1))
                one["anode_default"] = sel if sel in (1, 2) else 1
            one.pop("_anode_sel", None)
            merged.append(one)
            continue

        canonical = dict(items[0])
        same_values = all(
            (i.get("topology") == canonical.get("topology"))
            and all(float(i.get(k, 0.0)) == float(canonical.get(k, 0.0)) for k in numeric_keys)
            for i in items[1:]
        )
        if same_values:
            canonical["anodes"] = 2 if any(i.get("_anode_sel") in (1, 2) for i in items) else 1
            if canonical["anodes"] > 1:
                canonical["anode_default"] = 1
            canonical.pop("_anode_sel", None)
            merged.append(canonical)
            continue

        for i in sorted(items, key=lambda x: int(x.get("_anode_sel", 0))):
            out = dict(i)
            anode_sel = int(out.get("_anode_sel", 0))
            out["type"] = f"{base_type}_{anode_sel}" if anode_sel in (1, 2) else base_type
            out["anodes"] = 1
            out.pop("anode_default", None)
            out.pop("_anode_sel", None)
            merged.append(out)
    return merged


def _load_limits() -> dict[str, dict]:
    if not LIMITS.exists():
        return {}
    try:
        data = json.loads(LIMITS.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    raw = data.get("limits", {})
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _apply_limits(lamps: list[dict], limits: dict[str, dict]) -> None:
    allowed = {"Pa_max", "Pig2_max", "Ra", "ua_max", "ia_max", "uh_max",
               "ih_max", "ug2_max"}
    by_type = {x.get("type"): x for x in lamps}
    for tube_type, vals in limits.items():
        lamp = by_type.get(tube_type)
        if not lamp:
            continue
        for key, value in vals.items():
            if key in allowed and value is not None:
                lamp[key] = value


def _load_existing() -> list[dict]:
    if not OUT.exists():
        return []
    try:
        data = json.loads(OUT.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [x for x in data.get("lamps", []) if isinstance(x, dict)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BOOTSTRAP-ONLY: extract lamps from the LM19 firmware "
                    "into config/lamps.json (initial export).")
    parser.add_argument(
        "--force-overwrite", action="store_true",
        help="required when config/lamps.json already exists: the file is "
             "manually curated after the initial export; a re-run carries "
             "over manually ADDED lamps but LOSES manual edits to "
             "firmware-derived lamps")
    args = parser.parse_args(argv)

    existing = _load_existing()
    if OUT.exists() and not args.force_overwrite:
        print(
            f"REFUSED: {OUT} already exists.\n"
            "extract_lamps.py is a bootstrap-only tool for the INITIAL "
            "export. lamps.json is manually curated after that (manual "
            "lamps, edited fields); a re-run overwrites this curation.\n"
            "Re-run with --force-overwrite if you really mean it: manually "
            "added lamps are carried over, manual edits to firmware lamps "
            "are LOST.",
            file=sys.stderr,
        )
        return 2

    lamps = parse_lamprom()
    # ML-072: unconditional (was gated on the manual 6P18P entry existing).
    lamps = [x for x in lamps if x.get("type") not in _PLACEHOLDER_TYPES]
    limits = _load_limits()
    if limits:
        _apply_limits(lamps, limits)

    manual_6p18p = None
    for item in existing:
        if item.get("type") == "6P18P":
            manual_6p18p = item
            break

    if manual_6p18p is not None:
        manual_6p18p = dict(manual_6p18p)
        manual_6p18p.setdefault("anodes", 1)
        manual_6p18p.setdefault("warmup_s", 120)
        if int(manual_6p18p.get("anodes", 1)) > 1 and "anode_default" not in manual_6p18p:
            manual_6p18p["anode_default"] = 1
        lamps = [x for x in lamps if x.get("type") != "6P18P"]
        lamps.insert(0, manual_6p18p)

    # ML-072 variant B: carry over manual lamps — types from the curated
    # file absent from the firmware (e.g. GU-29); otherwise a forced run
    # would erase them.
    known_types = {x.get("type") for x in lamps}
    carried = [dict(x) for x in existing
               if x.get("type") not in known_types
               and x.get("type") not in _PLACEHOLDER_TYPES]
    for item in carried:
        print(f"Carried over manual lamp: {item.get('type')}")
    lamps.extend(carried)

    data = {"lamps": lamps}
    write_json(OUT, data)
    print(f"Wrote {len(lamps)} lamps to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
