"""Load line classes for the amplifier package.

Includes a shared linear ``Ia = (Ub - Ua) / R`` helper used by
``ResistiveLoadLine``, ``TransformerLoadLine``, ``CathodeFollowerLoadLine``,
and ``PushPullLoadLine``. Plus the ``LoadLine`` runtime-checkable Protocol
that all stage builders accept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


# ─── Shared linear load-line endpoints + Ia formula ────────────────────

def _linear_endpoints(ub: float, r: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Endpoints of `Ia = (Ub - Ua) / R` line: (0, Ub/R) → (Ub, 0)."""
    ia_max = ub / r if r > 0 else 0.0
    return ((0.0, ia_max), (ub, 0.0))


def _linear_ia_at_ua(ub: float, ua: float, r: float) -> float:
    """Compute Ia (mA) on `Ia = (Ub - Ua) / R`. Returns 0 for r ≤ 0."""
    if r <= 0:
        return 0.0
    return (ub - ua) / r


# ─── Public protocol + concrete load-line classes ─────────────────────

@runtime_checkable
class LoadLine(Protocol):
    """Protocol for load line calculation."""

    def ia_at_ua(self, ua: float) -> float:
        """Return Ia (mA) at given Ua (V)."""
        ...

    def endpoints(self, ua_max: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return ((ua_start, ia_start), (ua_end, ia_end)) for drawing."""
        ...

    def label(self) -> str:
        """Human-readable description."""
        ...


@dataclass
class ResistiveLoadLine:
    """Classic resistive load: Ia = (Ub - Ua) / Ra.

    Attributes:
        ub: Supply voltage, V.
        ra: Anode load resistance, kOhm.
    """
    ub: float
    ra: float

    def ia_at_ua(self, ua: float) -> float:
        return _linear_ia_at_ua(self.ub, ua, self.ra)

    def endpoints(self, ua_max: float = 0.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return _linear_endpoints(self.ub, self.ra)

    def label(self) -> str:
        return f"Ub={self.ub:.0f}V Ra={self.ra:.1f}kΩ"


@dataclass
class TransformerLoadLine:
    """Transformer-coupled load line with separate DC and AC paths.

    DC load line: Ra_dc = winding resistance (low, tens-hundreds Ohm).
    AC load line: Ra_ac = reflected load (typically kOhm).
    Q-point at intersection of DC load line and tube curve.
    Signal swing follows AC load line through Q-point.

    Attributes:
        ub: Supply voltage, V.
        ra_dc: DC resistance of primary winding, kOhm (typically 0.05-0.5).
        ra_ac: Reflected impedance of load, kOhm (typically 2-10).
    """
    ub: float
    ra_dc: float
    ra_ac: float

    def ia_at_ua(self, ua: float) -> float:
        return _linear_ia_at_ua(self.ub, ua, self.ra_ac)

    def ia_at_ua_dc(self, ua: float) -> float:
        return _linear_ia_at_ua(self.ub, ua, self.ra_dc)

    def ia_at_ua_ac(self, ua: float, q_ua: float, q_ia: float) -> float:
        """AC load line through Q-point: Ia = Ia_q - (Ua - Ua_q) / Ra_ac."""
        if self.ra_ac <= 0:
            return q_ia
        return q_ia - (ua - q_ua) / self.ra_ac

    def endpoints(self, ua_max: float = 0.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return _linear_endpoints(self.ub, self.ra_ac)

    def endpoints_dc(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return _linear_endpoints(self.ub, self.ra_dc)

    def label(self) -> str:
        return f"Ub={self.ub:.0f}V Ra_dc={self.ra_dc*1000:.0f}Ω Ra_ac={self.ra_ac:.1f}kΩ"


@dataclass
class CathodeFollowerLoadLine:
    """Cathode follower load line.

    In a CF the load is in the cathode. The DC load line equation:
    Ia = (Ub - Ua) / (Rk + Rl).

    Due to feedback, effective Ug1 = Ug1_input - Ia*Rk, so the
    operating point is determined by a system of equations. However
    for load-line / intersection analysis, the DC equation above
    suffices (intersections are identical to resistive with R = Rk+Rl).

    Attributes:
        ub: Supply voltage, V.
        rk: Cathode resistor, kOhm.
        rl: Load resistor, kOhm (AC load).
    """
    ub: float
    rk: float
    rl: float

    def ia_at_ua(self, ua: float) -> float:
        return _linear_ia_at_ua(self.ub, ua, self.rk + self.rl)

    def endpoints(self, ua_max: float = 0.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return _linear_endpoints(self.ub, self.rk + self.rl)

    def label(self) -> str:
        return f"CF Ub={self.ub:.0f}V Rk={self.rk:.1f}kΩ Rl={self.rl:.1f}kΩ"


@dataclass
class PushPullLoadLine:
    """Push-pull load line for transformer-coupled output stage.

    Each tube sees Ra_aa/4 (for matched transformer with center-tapped primary).
    Reflected impedance from "VTADiy — Push-Pull Loadline in Class AB"
    and "Vacuum-tube.eu — Push-Pull Transformer Impedance",
    see SOURCES_INDEX.md.
    The DC load line uses winding resistance; the AC line uses reflected load.

    Attributes:
        ub: Supply voltage, V (B+).
        ra_aa: Anode-to-anode load impedance, kOhm.
        ra_dc: DC winding resistance per half-primary, kOhm (default 0.1).
    """
    ub: float
    ra_aa: float
    ra_dc: float = 0.1

    @property
    def ra_per_tube(self) -> float:
        """Ra_aa / 4 — the COMPOSITE load-line slope (exact) and the
        per-tube load in class B (half the primary → (N/2)² reflection).

        In class A each tube individually sees Ra_aa/2 (the partner's
        antiphase current doubles the flux), kinking to Ra_aa/4 when the
        partner cuts off. ``compute_distortion_dft_pp`` models that kink
        exactly via the joint ideal-OPT solve (``_pp_joint_solve_vec``);
        composite-characteristic methods use this value as the composite
        slope (exact). Remaining fixed-value use: the per-tube Q-point
        AC lines in find_intersections*.
        """
        return self.ra_aa / 4.0

    @property
    def ra_class_a(self) -> float:
        """Ra_aa / 2 — per-tube load while the PARTNER still conducts
        (class-A region): the partner's antiphase current doubles the
        flux, so each tube sees half the anode-to-anode impedance. At
        the partner's cutoff (i = 2·Iq) the per-tube load kinks to
        ``ra_per_tube`` (Ra_aa/4). See ``pp_working_line_ia``."""
        return self.ra_aa / 2.0

    def ia_at_ua(self, ua: float) -> float:
        return _linear_ia_at_ua(self.ub, ua, self.ra_per_tube)

    def ia_at_ua_dc(self, ua: float) -> float:
        """DC load line: Ia = (Ub - Ua) / Ra_dc."""
        return _linear_ia_at_ua(self.ub, ua, self.ra_dc)

    def ia_at_ua_ac(self, ua: float, q_ua: float, q_ia: float) -> float:
        """AC load line through Q-point: Ia = Ia_q - (Ua - Ua_q) / Ra_per_tube."""
        ra = self.ra_per_tube
        if ra <= 0:
            return q_ia
        return q_ia - (ua - q_ua) / ra

    def endpoints(self, ua_max: float = 0.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return _linear_endpoints(self.ub, self.ra_per_tube)

    def label(self) -> str:
        return f"PP Ub={self.ub:.0f}V Ra_aa={self.ra_aa:.1f}kΩ (per tube {self.ra_per_tube:.2f}kΩ)"


# ─── Working-line geometry (single source for branches and drawing) ───

def pp_working_line_ia(ua, q_ua: float, q_ia: float, ra_aa: float):
    """Per-tube PP working-line current Ia(Ua), mA — KINKED trajectory.

    Physics (see PushPullLoadLine.ra_per_tube and
    external_sources/theory/vactube_push_pull_impedance.html): while the
    partner conducts (class A, this tube's current below 2·Iq) the tube
    sees ``Z2 = Ra_aa/2``; at the partner's cutoff — kink point
    ``K = (Ua_q − Iq·Z2, 2·Iq)`` — the load flattens to ``Z4 = Ra_aa/4``.
    The straight −1/Z4 line through Q (the naive display path)
    matches this trajectory only at K and asymptotically in class B; its
    middle understates current above Q and overstates it below.

    Array-safe (np.where): the model-intersection path evaluates it
    vectorized; scalar callers get a numpy scalar back (cast if the
    plain-float type is load-bearing). Like the straight-line closures,
    the formula is NOT clamped at cutoff — beyond ``Ua_q + Iq·Z2`` it
    goes negative, which the sign-change intersection search relies on.

    Degenerate ``ra_aa <= 0`` → constant ``q_ia`` (mirrors the old
    closures' guard).
    """
    if ra_aa <= 0:
        return q_ia + 0.0 * np.asarray(ua, dtype=float)
    z2 = ra_aa / 2.0
    z4 = ra_aa / 4.0
    kink_ua = q_ua - q_ia * z2
    ua_arr = np.asarray(ua, dtype=float)
    class_a = q_ia - (ua_arr - q_ua) / z2
    class_b = 2.0 * q_ia + (kink_ua - ua_arr) / z4
    return np.where(ua_arr >= kink_ua, class_a, class_b)


def working_line_polyline(
    load_line: "LoadLine",
    q_ua: Optional[float] = None,
    q_ia: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Vertices (ua, ia) of the circuit's working line for drawing.

    Single source of the line SHAPE:
      - Resistive / CF: the DC line, 2 vertices (Q not needed);
      - Transformer (SE): the AC line through Q with slope −1/Ra_ac,
        2 vertices (requires Q — [] without it);
      - Push-Pull: the kinked per-tube trajectory, 3 vertices
        [(0, ia_top), (kink_ua, 2·Iq), (cutoff_ua, 0)] — same geometry
        as ``pp_working_line_ia`` (equivalence is pinned). Degenerate
        Iq <= 0 → plain −1/Z4 line through (Ua_q, 0). If the kink lies
        left of ua=0 (huge Iq·Z2), the class-A segment is truncated at
        ua=0 and only 2 vertices are returned.

    Returns [] when the line is undefined (missing Q for Q-dependent
    circuits, non-positive impedances) — callers hide the overlay.
    """
    if isinstance(load_line, PushPullLoadLine):
        if q_ua is None or q_ia is None or load_line.ra_aa <= 0:
            return []
        z2 = load_line.ra_class_a
        z4 = load_line.ra_per_tube
        if q_ia <= 0:
            # Both tubes biased at cutoff: pure class-B line through Q.
            return [(0.0, q_ua / z4), (q_ua, 0.0)]
        kink_ua = q_ua - q_ia * z2
        cutoff_ua = q_ua + q_ia * z2
        if kink_ua <= 0.0:
            # Kink beyond the plot: class-A segment alone spans ua >= 0.
            return [(0.0, q_ia + q_ua / z2), (cutoff_ua, 0.0)]
        ia_top = 2.0 * q_ia + kink_ua / z4
        return [(0.0, ia_top), (kink_ua, 2.0 * q_ia), (cutoff_ua, 0.0)]
    if isinstance(load_line, TransformerLoadLine):
        if q_ua is None or q_ia is None or load_line.ra_ac <= 0:
            return []
        ra_ac = load_line.ra_ac
        return [(0.0, q_ia + q_ua / ra_ac), (q_ua + q_ia * ra_ac, 0.0)]
    if isinstance(load_line, CathodeFollowerLoadLine):
        r = load_line.rk + load_line.rl
        if r <= 0 or load_line.ub <= 0:
            return []
        return [(0.0, load_line.ub / r), (load_line.ub, 0.0)]
    # Resistive (and duck-typed lines with .ub/.ra)
    ub = getattr(load_line, "ub", 0.0)
    ra = getattr(load_line, "ra", 0.0)
    if ra <= 0 or ub <= 0:
        return []
    return [(0.0, ub / ra), (ub, 0.0)]
