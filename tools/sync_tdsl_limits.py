import json
import re
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lm19.io_utils import write_json
from lm19.tube_params import lookup_tube

LAMPS_PATH = ROOT / "config" / "lamps.json"
LIMITS_PATH = ROOT / "config" / "lamp_limits.json"
TDSL_BASE = "https://tdsl.duncanamps.com/show.php?des="

# Prefer direct type, then closest equivalents.
QUERY_MAP = {
    "6V6S": "6V6GT",
    "6F6S": "6F6G",
    "6P1P": "6AQ5",
    "EL90": "6AQ5",
    "EL95": "6DL5",
    "6N5S": "6AS7G",
    "6P18P": "EL84",
    "PCL86T": "PCL86",
    "PCL86P": "PCL86",
    "ECL86T": "ECL86",
    "ECL86P": "ECL86",
    "ECL82T": "ECL82",
    "ECL82P": "ECL82",
    "ECC832_1": "ECC832",
    "ECC832_2": "ECC832",
    "ForFutUse": "",
}


def _strip_tags(text: str) -> str:
    return re.sub(r"<.*?>", "", text).strip()


def _to_float(text: str):
    t = text.replace(",", ".").strip()
    # ranges like "20-75" are application data; ratings row should be scalar
    if re.fullmatch(r"-?\d+(?:\.\d+)?", t):
        return float(t)
    return None


def _fetch(url: str) -> str:
    with urlopen(url, timeout=25) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _parse_ratings(html: str):
    i = html.find("Vh<br>V")
    if i < 0:
        return None
    table_start = html.rfind("<table", 0, i)
    table_end = html.find("</table>", i)
    if table_start < 0 or table_end < 0:
        return None
    frag = html[table_start : table_end]

    headers = re.findall(r"<td class=\"bluer\">(.*?)</td>", frag, re.S)
    headers = [_strip_tags(h).replace("\xa0", " ").strip() for h in headers]
    if not headers:
        return None
    cells = re.findall(r"<td class=\"lightgreyr\">(.*?)</td>", frag, re.S)
    vals = [_strip_tags(c) for c in cells]
    if len(vals) < len(headers):
        return None
    row = vals[: len(headers)]
    by_name = {headers[i]: row[i] for i in range(len(headers))}

    return {
        "uh_max": _to_float(by_name.get("VhV", "")),
        "ih_a": _to_float(by_name.get("IhA", "")),
        "ua_max": _to_float(by_name.get("VaMaxV", "")),
        "ug2_max": _to_float(by_name.get("Vg2MaxV", "")),
        "Pa_max": _to_float(by_name.get("PaMaxW", "")),
        "Pig2_max": _to_float(by_name.get("Pg2MaxW", "")),
        "ik_max_ma": _to_float(by_name.get("IkMaxmA", "")),
    }


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    lamps = _load_json(LAMPS_PATH)["lamps"]
    limits_data = _load_json(LIMITS_PATH)
    limits = limits_data.get("limits", {})

    updated = 0
    resolved = 0
    unresolved = []

    for lamp in lamps:
        tube_type = lamp["type"]
        if tube_type == "6P18P":
            # Keep user-maintained manual card values for 6P18P.
            continue
        primary = QUERY_MAP.get(tube_type, tube_type)
        if not primary:
            unresolved.append((tube_type, "empty query mapping"))
            continue
        candidates = [primary, tube_type]
        ref = lookup_tube(tube_type)
        if ref:
            candidates.append(ref.name)
            candidates.extend(ref.aliases)
        # de-dup, keep order
        seen = set()
        candidates = [x for x in candidates if x and not (x in seen or seen.add(x))]

        ratings = None
        url = ""
        last_error = "ratings row not found"
        for q in candidates:
            test_url = f"{TDSL_BASE}{quote(q)}"
            try:
                html = _fetch(test_url)
            except Exception as exc:
                last_error = f"fetch failed: {exc}"
                continue
            parsed = _parse_ratings(html)
            if parsed:
                ratings = parsed
                url = test_url
                break
        if not ratings:
            unresolved.append((tube_type, last_error))
            continue

        entry = limits.setdefault(tube_type, {})
        entry["source"] = f"TDSL {url}"
        changed = False

        if ratings["Pa_max"] is not None:
            entry["Pa_max"] = ratings["Pa_max"]
            changed = True
        if ratings["ua_max"] is not None:
            entry["ua_max"] = ratings["ua_max"]
            changed = True
        if ratings["uh_max"] is not None:
            # keep +10% policy; never below nominal configured heater voltage
            nominal_uh = float(lamp.get("uh", 0.0))
            tdsl_uh_max = ratings["uh_max"] * 1.1
            base = max(nominal_uh * 1.1 if nominal_uh > 0 else 0.0, tdsl_uh_max)
            entry["uh_max"] = round(min(15.0, base), 1)
            changed = True
        if ratings["ih_a"] is not None:
            entry["ih_max"] = float(int(round(ratings["ih_a"] * 1000.0)))
            changed = True
        if ratings["ik_max_ma"] is not None:
            entry["ia_max"] = ratings["ik_max_ma"]
            changed = True
        if lamp.get("topology") == "pentode":
            if ratings["Pig2_max"] is not None:
                entry["Pig2_max"] = ratings["Pig2_max"]
                changed = True
            if ratings["ug2_max"] is not None:
                # Vg2MaxV used to be parsed and silently dropped
                # (ML-131) — the per-lamp screen-voltage cap never reached
                # the loader/UI chain.
                entry["ug2_max"] = ratings["ug2_max"]
                changed = True

        if changed:
            updated += 1
        resolved += 1

    limits_data["limits"] = dict(sorted(limits.items()))
    write_json(LIMITS_PATH, limits_data)

    print(f"Resolved via TDSL: {resolved}")
    print(f"Updated entries: {updated}")
    print(f"Unresolved: {len(unresolved)}")
    for tube_type, reason in unresolved[:40]:
        print(f" - {tube_type}: {reason}")


if __name__ == "__main__":
    main()
