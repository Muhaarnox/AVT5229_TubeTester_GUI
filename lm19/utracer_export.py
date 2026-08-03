"""Export LM19 measurement data to uTracer .utd format.

Generates .utd files compatible with uTracer3 GUI, ExtractModel,
Load Line Tool, and utMax.  Supports output curves I(Va, Vg) and
transfer curves I(Vg, Va), with optional screen current Ig2.

Vs (screen voltage) is encoded in the filename by convention
(e.g. EL84_250.utd → Vs=250V).  Vh is not stored in .utd.

No Qt dependency — pure business logic, fully testable.
"""


from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from lm19.constants import EPS, UG1_ROUND, UA_ROUND

log = logging.getLogger(__name__)


def detect_best_format(points: List[Dict]) -> str:
    """Auto-detect whether output or transfer format is better.

    Heuristic: if there are more unique Ua values than unique Ug1
    values, output curves (Va on X) are the natural choice.
    Otherwise transfer curves (Vg on X).

    Returns:
        "output" or "transfer".
    """
    if not points:
        return "output"
    ua_set = set()
    ug1_set = set()
    for p in points:
        ua_set.add(round(p.get("ua", 0.0), 1))
        ug1_set.add(round(p.get("ug1", 0.0), UG1_ROUND))
    return "output" if len(ua_set) >= len(ug1_set) else "transfer"


def _build_matrix(
    points: List[Dict],
    fmt: str,
) -> Tuple[List[float], List[float], List[List[float]],
           Optional[List[List[float]]], int]:
    """Group points into x_values, step_values, ia_matrix, is_matrix.

    For output:  x = Va (rows), step = Vg (columns).
    For transfer: x = Vg (rows), step = Va (columns).

    Returns:
        (x_values, step_values, ia_matrix, is_matrix_or_None, holes) —
        *holes* counts grid cells with no measured point (written 0.0).
    """
    is_output = fmt == "output"

    x_key = "ua" if is_output else "ug1"
    step_key = "ug1" if is_output else "ua"
    # Precision must match _fmt_voltage output (2 decimal places)
    x_round = 1 if is_output else 2
    step_round = 2 if is_output else 1

    x_set: set = set()
    step_set: set = set()
    for p in points:
        x_set.add(round(p[x_key], x_round))
        step_set.add(round(p[step_key], step_round))

    x_values = sorted(x_set)
    step_values = sorted(step_set)

    lookup_ia: Dict[Tuple[float, float], float] = {}
    lookup_ig2: Dict[Tuple[float, float], float] = {}
    has_ig2 = False
    for p in points:
        xv = round(p[x_key], x_round)
        sv = round(p[step_key], step_round)
        key = (xv, sv)
        lookup_ia[key] = p.get("ia", 0.0)
        ig2 = p.get("ig2", 0.0)
        lookup_ig2[key] = ig2
        if abs(ig2) > 0.001:
            has_ig2 = True

    ia_matrix: List[List[float]] = []
    is_matrix: List[List[float]] = [] if has_ig2 else None  # type: ignore[assignment]

    holes = 0
    for xv in x_values:
        ia_row: List[float] = []
        is_row: List[float] = []
        for sv in step_values:
            key = (xv, sv)
            if key not in lookup_ia:
                holes += 1
            ia_row.append(lookup_ia.get(key, 0.0))
            if has_ig2:
                is_row.append(lookup_ig2.get(key, 0.0))
        ia_matrix.append(ia_row)
        if has_ig2:
            is_matrix.append(is_row)  # type: ignore[union-attr]

    if holes:
        # ML-117: a protection break leaves the rectangular grid ragged —
        # the missing cells are written as Ia=0.0, which external tools
        # read as real measurements. Count and surface, never silent.
        log.warning(".utd matrix has %d hole(s) out of %d cells — filled "
                    "with Ia=0.0 (curves cut by protection?)",
                    holes, len(x_values) * len(step_values))

    return x_values, step_values, ia_matrix, is_matrix, holes


def format_utd(
    points: List[Dict],
    *,
    fmt: Optional[str] = None,
    stats: Optional[Dict[str, int]] = None,
) -> str:
    """Format LM19 measurement points as uTracer .utd content.

    Args:
        points: list of dicts with ua, ug1, ug2, ia, ig2, uh, ih.
        fmt: "output" (Va on X) or "transfer" (Vg on X).
             None = auto-detect.

    Returns:
        Complete .utd file content as string.

    Raises:
        ValueError: if points is empty or has no grid.
    """
    if not points:
        raise ValueError("No points to export")

    if fmt is None:
        fmt = detect_best_format(points)

    x_values, step_values, ia_matrix, is_matrix, holes = _build_matrix(
        points, fmt)
    if stats is not None:
        stats["utd_matrix_holes"] = holes

    if not x_values or not step_values:
        raise ValueError("Cannot build matrix: no x or step values")

    is_output = fmt == "output"
    has_is = is_matrix is not None

    lines: List[str] = []

    # --- Line 1: header ---
    if has_is:
        x_label = "Va" if is_output else "Vg"
        lines.append(f"{x_label} (V) Ia (mA) Is (mA) ")
    else:
        x_label = "Va" if is_output else "Vg"
        lines.append(f"{x_label} (V) Ia (mA) ")

    # --- Line 2: stepping values ---
    step_label = "Vg" if is_output else "Va"
    if has_is:
        # Pentode: each step value appears twice (Ia + Is columns)
        parts = []
        for sv in step_values:
            tag = f" {step_label} = {_fmt_voltage(sv)} V"
            parts.append(tag)
            parts.append(tag)
        lines.append(" ".join(parts))
    else:
        parts = [f" {step_label} = {_fmt_voltage(sv)} V" for sv in step_values]
        lines.append("  ".join(parts))

    # --- Lines 3+: data rows ---
    for x_idx, xv in enumerate(x_values):
        tokens = [_fmt_voltage(xv)]
        if has_is:
            for s_idx in range(len(step_values)):
                tokens.append(_fmt_current(ia_matrix[x_idx][s_idx]))
                tokens.append(_fmt_current(is_matrix[x_idx][s_idx]))  # type: ignore[index]
        else:
            for s_idx in range(len(step_values)):
                tokens.append(_fmt_current(ia_matrix[x_idx][s_idx]))
        lines.append(" ".join(tokens))

    # Trailing newline
    lines.append("")
    return "\n".join(lines)


def suggest_filename(
    tube_type: str,
    points: List[Dict],
) -> str:
    """Suggest a .utd filename based on tube type and Ug2.

    Convention: <tube_type>_<Vs>.utd for pentodes, <tube_type>.utd for triodes.
    """
    ug2_values = set(round(p.get("ug2", 0.0), 0) for p in points)
    ug2_values.discard(0.0)
    if ug2_values:
        vs = int(max(ug2_values))
        return f"{tube_type}_{vs}.utd"
    return f"{tube_type}.utd"


def _fmt_voltage(v: float) -> str:
    """Format voltage with enough precision to preserve the value.

    Uses integer format for whole numbers, 1 decimal for .X values,
    2 decimals for .XX values (e.g. Vg steps of 0.25V).
    """
    if v == int(v):
        return str(int(v))
    # Check if 1 decimal is enough
    rounded_1 = round(v, 1)
    if abs(v - rounded_1) < EPS:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _fmt_current(ma: float) -> str:
    """Format current in mA with 3 decimal places (uTracer convention)."""
    if abs(ma) < 0.0005:
        return "0.000"
    return f"{ma:.3f}"
