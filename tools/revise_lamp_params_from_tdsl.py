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
REPORT_PATH = ROOT / "docs" / "LAMP_REVISION_REPORT.md"
TDSL_BASE = "https://tdsl.duncanamps.com/show.php?des="

QUERY_MAP = {
    "6V6S": "6V6GT",
    "6F6S": "6F6G",
    "6P1P": "6AQ5",
    "EL90": "6AQ5",
    "EL95": "6DL5",
    "6N5S": "6AS7G",
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

UNRESOLVED_LIMIT_SOURCES = {
    "6N6P": "Frank fallback https://frank.pocnet.net/sheets/124/6/6N6P.pdf",
    "6S19P": "Frank fallback https://frank.pocnet.net/sheets/113/6/6S19P.pdf",
    "6S2S": "Frank fallback https://frank.pocnet.net/sheets/129/6/6S2S.pdf",
    "ECC40": "Frank fallback https://frank.pocnet.net/sheets/030/e/ECC40.pdf",
    "ECC99": "JJ fallback https://www.jj-electronic.com/images/stories/productinfo/ECC99.pdf",
    "E182CC": "Frank fallback https://frank.pocnet.net/sheets/129/e/E182CC.pdf",
    "ForFutUse": "placeholder",
}


def _fetch(url: str) -> str:
    with urlopen(url, timeout=25) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _strip_tags(text: str) -> str:
    return re.sub(r"<.*?>", "", text).strip()


def _to_float(text: str):
    t = text.replace(",", ".").strip()
    if not t:
        return None
    # ML-074: parse ranges via the regex GROUPS — the old split("-", 1)
    # cut at the SIGN of a negative first number ("-2--4" → float("")).
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", t)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    if re.fullmatch(r"-?\d+(?:\.\d+)?", t):
        return float(t)
    return None


def _iter_candidate_queries(tube_type: str):
    first = QUERY_MAP.get(tube_type, tube_type)
    candidates = [first, tube_type]
    ref = lookup_tube(tube_type)
    if ref:
        candidates.append(ref.name)
        candidates.extend(ref.aliases)
    seen = set()
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            yield c


def _parse_table_by_header(html: str, header_token: str):
    i = html.find(header_token)
    if i < 0:
        return None
    s = html.find("<table", i)
    e = html.find("</table>", s)
    if s < 0 or e < 0:
        return None
    frag = html[s:e]
    headers = re.findall(r"<td class=\"(?:blue|bluer)\">(.*?)</td>", frag, re.S)
    headers = [_strip_tags(h).replace("\xa0", " ").strip() for h in headers]
    rows = []
    for row in re.findall(r"<tr>\s*((?:<td class=\"lightgrey[r]?\">.*?</td>\s*)+)</tr>", frag, re.S):
        cells = re.findall(r"<td class=\"lightgrey[r]?\">(.*?)</td>", row, re.S)
        vals = [_strip_tags(c) for c in cells]
        if len(vals) < len(headers):
            vals += [""] * (len(headers) - len(vals))
        rows.append({headers[idx]: vals[idx] for idx in range(len(headers))})
    return rows


def _pick_nominal_row(rows, topology: str, current: dict):
    if not rows:
        return None
    current_ua = float(current.get("ua", 0.0))
    current_ug2 = float(current.get("ug2", 0.0))
    current_ia = float(current.get("ia", 0.0))
    current_ug1 = float(current.get("ug1", 0.0))

    def score(row):
        va = _to_float(row.get("VaV", "")) or 0.0
        vg1 = _to_float(row.get("Vg1V", "")) or 0.0
        vg2 = _to_float(row.get("Vg2V", "")) or 0.0
        ia = _to_float(row.get("IamA", "")) or 0.0
        s = 0.0
        s += abs(va - current_ua)
        s += abs(ia - current_ia) * 2.0
        s += abs(vg1 - current_ug1) * 4.0
        if topology == "pentode":
            s += abs(vg2 - current_ug2)
        return s

    if topology == "triode":
        tri = []
        for r in rows:
            cls = r.get("Class", "")
            if "Triode" in cls and "P/P" not in cls:
                tri.append(r)
        if tri:
            return min(tri, key=score)
    else:
        pent = []
        for r in rows:
            cls = r.get("Class", "")
            if "A S/E" in cls and "triode" not in cls.lower():
                pent.append(r)
        if pent:
            return min(pent, key=score)
    return min(rows, key=score)


def _extract_nominal(row, topology: str):
    ua = _to_float(row.get("VaV", ""))
    ug1 = _to_float(row.get("Vg1V", ""))
    ia = _to_float(row.get("IamA", ""))
    s = _to_float(row.get("SmA/V", ""))
    ra_ohm = _to_float(row.get("RaΩ", ""))
    ug2 = _to_float(row.get("Vg2V", "")) if topology == "pentode" else 0.0
    ig2 = _to_float(row.get("Ig2mA", "")) if topology == "pentode" else 0.0
    if ua is None or ug1 is None or ia is None:
        return None
    r = (ra_ohm / 1000.0) if ra_ohm is not None else None
    k = (s * r) if (s is not None and r is not None) else None
    return {
        "ua": round(ua, 1),
        "ug1": round(ug1, 2),
        "ia": round(ia, 2),
        # ML-075: keep None — the merge loop skips None values, so a TDSL
        # row without Vg2/Ig2 must NOT zero the existing pentode nominal
        # ("or 0.0" bypassed the `if v is not None` filter). The triode
        # branch assigns the literal 0.0 (correct) and is preserved.
        "ug2": round(ug2, 1) if ug2 is not None else None,
        "ig2": round(ig2, 2) if ig2 is not None else None,
        "s": round(s, 2) if s is not None else None,
        "r": round(r, 2) if r is not None else None,
        "k": round(k, 2) if k is not None else None,
    }


def main():
    lamps_data = json.loads(LAMPS_PATH.read_text(encoding="utf-8"))
    limits_data = json.loads(LIMITS_PATH.read_text(encoding="utf-8"))
    limits = limits_data.get("limits", {})

    revised = 0
    unresolved_nom = []
    diffs = []

    for lamp in lamps_data["lamps"]:
        t = lamp["type"]
        if t in ("ForFutUse", "6P18P"):
            continue

        html = None
        source_url = None
        for q in _iter_candidate_queries(t):
            u = f"{TDSL_BASE}{quote(q)}"
            try:
                txt = _fetch(u)
            except Exception:
                continue
            rows = _parse_table_by_header(txt, "Application Data")
            if rows:
                html = txt
                source_url = u
                break
        if html is None:
            unresolved_nom.append(t)
            continue

        rows = _parse_table_by_header(html, "Application Data")
        row = _pick_nominal_row(rows, lamp["topology"], lamp)
        nom = _extract_nominal(row, lamp["topology"]) if row else None
        if not nom:
            unresolved_nom.append(t)
            continue

        before = {k: lamp.get(k) for k in ("ua", "ug1", "ia", "ug2", "ig2", "s", "r", "k")}
        for k, v in nom.items():
            if v is not None:
                lamp[k] = v
        after = {k: lamp.get(k) for k in ("ua", "ug1", "ia", "ug2", "ig2", "s", "r", "k")}
        if before != after:
            revised += 1
            diffs.append((t, before, after, source_url))

        lim = limits.setdefault(t, {})
        if str(lim.get("source", "")).startswith("TDSL "):
            lim["source"] = f"TDSL {source_url}"

    for t, src in UNRESOLVED_LIMIT_SOURCES.items():
        if t in limits and not str(limits[t].get("source", "")).startswith("TDSL "):
            limits[t]["source"] = src

    lamps_data["lamps"] = sorted(lamps_data["lamps"], key=lambda x: x["type"])
    write_json(LAMPS_PATH, lamps_data)
    limits_data["limits"] = dict(sorted(limits.items()))
    write_json(LIMITS_PATH, limits_data)

    lines = [
        "# Lamp Revision Report",
        "",
        f"- Revised nominal rows from TDSL: **{revised}**",
        f"- Unresolved nominal rows: **{len(unresolved_nom)}**",
        f"- Limits source TDSL coverage: **{sum(1 for v in limits.values() if str(v.get('source','')).startswith('TDSL '))}/{len(limits)}**",
        "",
        "## Unresolved Nominal",
    ]
    for t in unresolved_nom:
        lines.append(f"- {t}")
    lines.append("")
    lines.append("## Top Differences (before -> after)")
    for t, before, after, src in diffs[:40]:
        lines.append(f"- `{t}` ({src})")
        lines.append(
            f"  - ua {before['ua']} -> {after['ua']}, ug1 {before['ug1']} -> {after['ug1']}, ia {before['ia']} -> {after['ia']}"
        )
        lines.append(
            f"  - ug2 {before['ug2']} -> {after['ug2']}, ig2 {before['ig2']} -> {after['ig2']}, s {before['s']} -> {after['s']}, r {before['r']} -> {after['r']}, k {before['k']} -> {after['k']}"
        )
    # newline="" keeps the LF that "\n".join produced: text mode would
    # translate it to CRLF on Windows and flip the whole report file.
    with open(REPORT_PATH, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Revised nominals: {revised}")
    print(f"Unresolved nominals: {len(unresolved_nom)}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
