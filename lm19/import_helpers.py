"""Pure helpers for measurement import — Qt-free, unit-testable."""


from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
    UG2_MODES,
)


def import_topology_payload(ug2_mode: str) -> Dict:
    """Build topology/scan dict fragment from a ug2_mode string."""
    # Validated against the registry, not a local copy of it: a mode added
    # to UG2_MODES would otherwise be silently rewritten to pentode here.
    mode = ug2_mode if ug2_mode in UG2_MODES else TOPOLOGY_PENTODE
    topology = TOPOLOGY_TRIODE if mode == TOPOLOGY_TRIODE else TOPOLOGY_PENTODE
    return {
        "topology": topology,
        "scan": {
            "ug2_mode": mode,
            "ug2_track_ua": mode == TOPOLOGY_TRIODE_CONNECTED,
        },
    }


def first_non_comment_csv_header(text: str) -> str:
    """Return the first non-comment, non-empty line from CSV text."""
    for ln in text.splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            return s
    return ""


def csv_comment_lines(text: str, max_lines: int = 5) -> List[str]:
    """Extract leading comment lines (stripped of '#') from CSV text."""
    out: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            out.append(s.lstrip("#").strip())
            if len(out) >= max_lines:
                break
        elif out:
            break
    return out


def build_utd_description(path: str, parsed: Dict, guessed_vs: float) -> str:
    """Build a human-readable description for an imported uTracer file."""
    lines = [
        "Source: uTracer (.utd)",
        f"File: {Path(path).name}",
        f"Format: {parsed.get('format', 'unknown')}",
        f"Has Is/Ig2: {'yes' if parsed.get('has_is') else 'no'}",
        f"Rows: {len(parsed.get('x_values', []))}",
        f"Steps: {len(parsed.get('step_values', []))}",
    ]
    if guessed_vs > 0:
        lines.append(f"Guessed Vs from filename: {guessed_vs:g} V")
    return "\n".join(lines)


def build_csv_description(path: str, text: str) -> str:
    """Build a human-readable description for an imported CSV file."""
    header = first_non_comment_csv_header(text)
    comments = csv_comment_lines(text)
    lines = [
        "Source: CSV/TSV",
        f"File: {Path(path).name}",
    ]
    if header:
        lines.append(f"Header: {header}")
    if comments:
        lines.append("Comments:")
        lines.extend(f"- {c}" for c in comments)
    return "\n".join(lines)


def build_etracer_description(path: str, parsed: Dict) -> str:
    """Build a human-readable description for an imported eTracer file."""
    curves = parsed.get("curves", [])
    total_pts = sum(len(c.get("hv1_v", [])) for c in curves)
    version = parsed.get("version", "unknown")
    etd = parsed.get("etd_file", "")
    lines = [
        "Source: eTracer CSV",
        f"File: {Path(path).name}",
        f"Format version: {version}",
        f"Curves: {len(curves)}",
        f"Total points: {total_pts}",
        f"HV2: {'ON' if parsed.get('hv2_on') else 'OFF'}",
    ]
    if parsed.get("hv2_link"):
        lines.append("HV2 linked to HV1 (triode connection)")
    if etd:
        lines.append(f"ETD config: {Path(etd).name}")
    negv = parsed.get("negv_setting")
    if negv:
        lines.append(f"NEGV sweep: {negv[0]}..{negv[1]} step {negv[2]} V")
    return "\n".join(lines)


def build_ctd_description(path: str, parsed: Dict) -> str:
    """Build a human-readable description for an imported CurveTraceData file."""
    sample = str(parsed.get("sample_name", "")).strip()
    date_str = str(parsed.get("date_str", "")).strip()
    points = parsed.get("points", [])
    lines = [
        "Source: CurveTraceData (.dat)",
        f"File: {Path(path).name}",
        f"Points parsed: {len(points)}",
    ]
    if sample:
        lines.append(f"Sample: {sample}")
    if date_str:
        lines.append(f"Date/time: {date_str}")
    lines.append("Note: Ia imported from A to mA.")
    return "\n".join(lines)
