"""Tube quality scoring, matching, aging analysis, and dead-data detection."""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from lm19.curve_data import _cluster_nominal, _nominal_key
from lm19.constants import UG1_CLUSTER_THR, UG1_ROUND, UG2_CLUSTER_THR
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)

# ── Amp class weighting for curve matching ──────────────────────────
# Controls how much weight the operating region vs knee/cutoff gets.
AMP_CLASS_A = "class_a"        # weight = Ia (operating point dominates)
AMP_CLASS_AB = "class_ab"      # weight = sqrt(Ia) (balanced)
AMP_CLASS_B = "class_b"        # weight = 1 (entire curve equal)

IA_WEIGHT_FLOOR = 0.01         # mA — avoid zero weights


@dataclass
class QualityReport:
    """Tube health assessment relative to datasheet."""
    ia_pct: Optional[float]  # measured Ia / nominal Ia at operating point (%)
    s_pct: Optional[float]   # measured S / nominal S (%)
    r_pct: Optional[float]   # measured R / nominal R (%)
    verdict: str             # "Strong", "Good", "Weak", "Replace", "N/A"


@dataclass
class MatchResult:
    """Quantitative matching between two tube measurements."""
    mean_delta: float   # mean |delta Ia| (mA)
    max_delta: float    # max |delta Ia| (mA)
    rms_delta: float    # RMS delta Ia (mA)
    match_pct: float    # 100 = identical, 0 = completely different
    n_points: int       # number of compared points


@dataclass
class AgingPoint:
    """Single aging data point."""
    timestamp: str
    ia_at_op: Optional[float]  # Ia at operating point (mA)
    s: Optional[float]
    r: Optional[float]
    k: Optional[float]
    name: str = ""


# ------------------------------------------------------------------
# Shared helpers (used by multiple functions in this module)
# ------------------------------------------------------------------

def _group_by_ug1(points: List[Dict], ug1_noms: list = None,
                   ug1_cluster_thr: float = UG1_CLUSTER_THR) -> Dict[float, List[Dict]]:
    """Group measurement points by clustered Ug1 nominal.

    If *ug1_noms* is provided, it is used as the shared cluster list
    (useful when comparing two measurements with a common grid).
    """
    if ug1_noms is None:
        raw = sorted({round(p["ug1"], UG1_ROUND) for p in points})
        ug1_noms = _cluster_nominal(raw, threshold=ug1_cluster_thr)
    groups: Dict[float, List[Dict]] = {}
    for p in points:
        ug1 = _nominal_key(round(p["ug1"], UG1_ROUND), ug1_noms)
        groups.setdefault(ug1, []).append(p)
    return groups


def _group_by_ug1_ug2(
    points: List[Dict],
    ug1_noms: list,
    ug2_noms: list,
) -> Dict[Tuple[float, float], List[Dict]]:
    """Group measurement points by (Ug1, Ug2) nominal pair.

    For multi-Ug2 pentode scans each (Ug1, Ug2) combination is a
    separate curve that must be compared independently.
    """
    groups: Dict[Tuple[float, float], List[Dict]] = {}
    for p in points:
        ug1 = _nominal_key(round(p["ug1"], UG1_ROUND), ug1_noms)
        ug2_raw = round(p.get("ug2", 0.0), 1)
        ug2 = _nominal_key(ug2_raw, ug2_noms) if ug2_noms else 0.0
        groups.setdefault((ug1, ug2), []).append(p)
    return groups


def _find_nearest_ia(
    points: List[Dict],
    nom_ua: float,
    nom_ug1: float,
    nom_ug2: float = 0.0,
    ug2_weight: float = 0.01,
    is_triode: bool = False,
) -> Optional[float]:
    """Find Ia of the point closest to the nominal operating point.

    Distance metric: (Ua - nom_ua)^2 + (Ug1 - nom_ug1)^2 + weight*(Ug2 - nom_ug2)^2.
    For triodes, Ug2 is excluded from the distance calculation.
    Returns Ia value or None if points is empty.
    """
    best_dist = float("inf")
    best_ia = None
    for p in points:
        d = (p["ua"] - nom_ua) ** 2 + (p["ug1"] - nom_ug1) ** 2
        if not is_triode and nom_ug2 > 0:
            d += (p.get("ug2", 0) - nom_ug2) ** 2 * ug2_weight
        if d < best_dist:
            best_dist = d
            best_ia = p["ia"]
    return best_ia


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def compute_quality(points: List[Dict], lamp_config, srk: Optional[Dict] = None) -> QualityReport:
    """Compute tube quality score by comparing measured data to datasheet.

    Args:
        points: scan data points
        lamp_config: LampConfig with nominal ia, s, r, k, ua, ug1
        srk: dict with measured s, r, k (optional)

    Returns:
        QualityReport with percentage scores and verdict.
    """
    if lamp_config is None:
        return QualityReport(None, None, None, "N/A")

    nom_ia = lamp_config.ia
    nom_s = lamp_config.s
    nom_r = lamp_config.r
    nom_ua = lamp_config.ua
    nom_ug1 = lamp_config.ug1
    nom_ug2 = lamp_config.ug2

    # Find Ia at operating point from scan data
    is_triode = getattr(lamp_config, 'is_triode', False)
    ia_pct = None
    if nom_ia > 0 and points:
        best_ia = _find_nearest_ia(points, nom_ua, nom_ug1, nom_ug2,
                                    is_triode=is_triode)
        if best_ia is not None:
            ia_pct = (best_ia / nom_ia) * 100.0

    # S/R/K percentages
    s_pct = None
    r_pct = None
    if srk:
        s_val = srk.get("s")
        r_val = srk.get("r")
        if s_val is not None and nom_s > 0:
            s_pct = (s_val / nom_s) * 100.0
        if r_val is not None and nom_r > 0:
            r_pct = (r_val / nom_r) * 100.0

    # Determine verdict from the most relevant metric
    scores = [x for x in [ia_pct, s_pct] if x is not None]
    if not scores:
        return QualityReport(ia_pct, s_pct, r_pct, "N/A")

    avg_score = sum(scores) / len(scores)
    if avg_score >= 110:
        verdict = "Strong"
    elif avg_score >= 80:
        verdict = "Good"
    elif avg_score >= 50:
        verdict = "Weak"
    else:
        verdict = "Replace"

    return QualityReport(ia_pct, s_pct, r_pct, verdict)


def _build_shared_nominals(
    points_a: List[Dict], points_b: List[Dict],
    ug1_cluster_thr: float = UG1_CLUSTER_THR,
) -> Tuple[list, list]:
    """Build shared Ug1 and Ug2 nominal lists from two point sets."""
    all_ug1_raw = sorted({round(p["ug1"], UG1_ROUND) for p in points_a}
                         | {round(p["ug1"], UG1_ROUND) for p in points_b})
    ug1_noms = _cluster_nominal(all_ug1_raw, threshold=ug1_cluster_thr)

    all_ug2_raw = sorted({round(p.get("ug2", 0.0), 1) for p in points_a}
                         | {round(p.get("ug2", 0.0), 1) for p in points_b})
    ug2_noms = _cluster_nominal(all_ug2_raw, threshold=UG2_CLUSTER_THR)

    return ug1_noms, ug2_noms


def _compare_curve_pair(
    curve_a: List[Dict], curve_b: List[Dict],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Interpolate two single-Ug2 curves to common Ua grid.

    Returns (deltas, ia_avg) arrays, or None if insufficient overlap.
    ia_avg is the mean of interpolated Ia from both curves (for weighting).
    """
    if not curve_a or not curve_b:
        return None
    curve_a = sorted(curve_a, key=lambda p: p["ua"])
    curve_b = sorted(curve_b, key=lambda p: p["ua"])

    ua_min = max(curve_a[0]["ua"], curve_b[0]["ua"])
    ua_max = min(curve_a[-1]["ua"], curve_b[-1]["ua"])
    if ua_max <= ua_min:
        return None

    ua_a = np.array([p["ua"] for p in curve_a])
    ia_a = np.array([p["ia"] for p in curve_a])
    ua_b = np.array([p["ua"] for p in curve_b])
    ia_b = np.array([p["ia"] for p in curve_b])

    ua_grid = np.unique(np.concatenate([
        ua_a[(ua_a >= ua_min) & (ua_a <= ua_max)],
        ua_b[(ua_b >= ua_min) & (ua_b <= ua_max)],
    ]))
    if len(ua_grid) < 2:
        return None

    ia_a_interp = np.interp(ua_grid, ua_a, ia_a)
    ia_b_interp = np.interp(ua_grid, ua_b, ia_b)
    deltas = ia_a_interp - ia_b_interp
    ia_avg = (np.abs(ia_a_interp) + np.abs(ia_b_interp)) / 2.0
    return deltas, ia_avg


def compute_matching(points_a: List[Dict], points_b: List[Dict],
                     ug2_tolerance: float = 5.0,
                     ug1_cluster_thr: float = UG1_CLUSTER_THR,
                     amp_class: str = AMP_CLASS_AB) -> Optional[MatchResult]:
    """Compute quantitative matching between two tube measurements.

    Groups points by (Ug1, Ug2) to correctly handle multi-Ug2 pentode
    scans.  Interpolates both datasets to a common Ua grid per curve
    and computes weighted delta Ia statistics.

    Weighting by amp_class:
    - class_a:  weight = Ia (operating point dominates, knee irrelevant)
    - class_ab: weight = sqrt(Ia) (balanced — knee and cutoff contribute)
    - class_b:  weight = 1 (entire curve equally important)

    Args:
        points_a: first measurement points
        points_b: second measurement points
        ug2_tolerance: tolerance for Ug2 matching
        ug1_cluster_thr: Ug1 clustering threshold (V)
        amp_class: weighting mode (AMP_CLASS_A, AMP_CLASS_AB, AMP_CLASS_B)

    Returns:
        MatchResult or None if insufficient data.
    """
    if not points_a or not points_b:
        return None

    ug1_noms, ug2_noms = _build_shared_nominals(
        points_a, points_b, ug1_cluster_thr)
    groups_a = _group_by_ug1_ug2(points_a, ug1_noms, ug2_noms)
    groups_b = _group_by_ug1_ug2(points_b, ug1_noms, ug2_noms)

    common_keys = set(groups_a.keys()) & set(groups_b.keys())
    if not common_keys:
        return None

    all_deltas = []
    all_ia_avg = []
    for key in sorted(common_keys):
        result = _compare_curve_pair(groups_a[key], groups_b[key])
        if result is not None:
            deltas, ia_avg = result
            all_deltas.extend(deltas.tolist())
            all_ia_avg.extend(ia_avg.tolist())

    if not all_deltas:
        return None

    deltas_arr = np.array(all_deltas)
    ia_avg_arr = np.maximum(np.array(all_ia_avg), IA_WEIGHT_FLOOR)

    # Compute weights based on amp class
    if amp_class == AMP_CLASS_A:
        weights = ia_avg_arr                  # linear: Ia dominates
    elif amp_class == AMP_CLASS_B:
        weights = np.ones_like(ia_avg_arr)    # uniform: all equal
    else:  # AMP_CLASS_AB (default)
        weights = np.sqrt(ia_avg_arr)         # sqrt: balanced

    # Unweighted statistics (for display)
    abs_deltas = np.abs(deltas_arr)
    mean_delta = float(np.mean(abs_deltas))
    max_delta = float(np.max(abs_deltas))

    # Weighted RMS and weighted mean Ia → match_pct
    w_sum = np.sum(weights)
    if w_sum < 1e-12:
        w_sum = 1.0
    rms_delta = float(np.sqrt(np.sum(weights * deltas_arr ** 2) / w_sum))
    mean_ia = float(np.sum(weights * ia_avg_arr) / w_sum)
    if mean_ia < 0.1:
        mean_ia = 0.1
    match_pct = max(0.0, 100.0 * (1.0 - rms_delta / mean_ia))

    return MatchResult(
        mean_delta=mean_delta,
        max_delta=max_delta,
        rms_delta=rms_delta,
        match_pct=match_pct,
        n_points=len(all_deltas),
    )


def compute_matching_curves(points_a: List[Dict], points_b: List[Dict],
                            ug1_cluster_thr: float = UG1_CLUSTER_THR) -> List[Dict]:
    """Compute delta Ia curves for matching visualization.

    Groups by (Ug1, Ug2) to correctly handle multi-Ug2 pentode scans.
    Returns list of dicts: {ug1, ug2, ua_values, delta_ia, delta_pct}
    for each common (Ug1, Ug2) pair.
    """
    ug1_noms, ug2_noms = _build_shared_nominals(
        points_a, points_b, ug1_cluster_thr)
    groups_a = _group_by_ug1_ug2(points_a, ug1_noms, ug2_noms)
    groups_b = _group_by_ug1_ug2(points_b, ug1_noms, ug2_noms)
    common_keys = sorted(set(groups_a.keys()) & set(groups_b.keys()))

    curves = []
    for key in common_keys:
        ug1, ug2 = key
        curve_a = sorted(groups_a[key], key=lambda p: p["ua"])
        curve_b = sorted(groups_b[key], key=lambda p: p["ua"])

        ua_min = max(curve_a[0]["ua"], curve_b[0]["ua"])
        ua_max = min(curve_a[-1]["ua"], curve_b[-1]["ua"])
        if ua_max <= ua_min:
            continue

        ua_a = np.array([p["ua"] for p in curve_a])
        ia_a = np.array([p["ia"] for p in curve_a])
        ua_b = np.array([p["ua"] for p in curve_b])
        ia_b = np.array([p["ia"] for p in curve_b])

        ua_grid = np.linspace(ua_min, ua_max, max(20, int((ua_max - ua_min) / 5)))
        ia_a_interp = np.interp(ua_grid, ua_a, ia_a)
        ia_b_interp = np.interp(ua_grid, ua_b, ia_b)
        delta = ia_a_interp - ia_b_interp

        avg_ia = (ia_a_interp + ia_b_interp) / 2.0
        avg_ia[avg_ia < 0.1] = 0.1
        delta_pct = delta / avg_ia * 100.0

        curves.append({
            "ug1": ug1,
            "ug2": ug2,
            "ua_values": ua_grid.tolist(),
            "delta_ia": delta.tolist(),
            "delta_pct": delta_pct.tolist(),
        })

    return curves


def compute_aging_trend(measurements: List[Dict], lamp_config=None
                        ) -> List[AgingPoint]:
    """Extract aging trend from a list of measurements for the same tube.

    Each measurement should have: timestamp, srk{s,r,k}, points[].
    If lamp_config provided, finds Ia at the nominal operating point.

    Returns list of AgingPoint sorted by timestamp.
    """
    trend = []
    for m in measurements:
        ts = m.get("timestamp", "")
        srk = m.get("srk", {}) or {}
        s = srk.get("s")
        r = srk.get("r")
        k = srk.get("k")
        name = m.get("name", "")

        ia_at_op = None
        points = m.get("points", [])
        if points and lamp_config is not None:
            ia_at_op = _find_nearest_ia(
                points, lamp_config.ua, lamp_config.ug1, lamp_config.ug2,
                is_triode=getattr(lamp_config, 'is_triode', False),
            )

        trend.append(AgingPoint(
            timestamp=ts,
            ia_at_op=ia_at_op,
            s=s, r=r, k=k,
            name=name,
        ))

    trend.sort(key=lambda a: a.timestamp)
    return trend


# ------------------------------------------------------------------
# Dead-data detection & cleanup (hardware/software protection artefacts)
# ------------------------------------------------------------------

# Ia threshold (mA): points below this are considered "dead" (zero current
# from hardware protection).  Normal cutoff also produces Ia≈0 but at the
# most-negative Ug1 values, while protection zeros current at the
# least-negative Ug1 — the heuristic uses scan iteration order to
# distinguish the two cases.
IA_DEAD_THR = 0.30  # mA


@dataclass
class DeadDataReport:
    """Result of dead-data analysis on a set of measurement points."""

    total_points: int
    dead_points: int
    dead_ug2_levels: List[float] = field(default_factory=list)
    partial_ug2_levels: List[Tuple[float, float]] = field(default_factory=list)
    """(ug2, ug1_transition) — Ug1 value at which data goes dead within level."""

    @property
    def live_points(self) -> int:
        return self.total_points - self.dead_points

    @property
    def dead_pct(self) -> float:
        if self.total_points == 0:
            return 0.0
        return self.dead_points / self.total_points * 100.0

    @property
    def has_dead_data(self) -> bool:
        return bool(self.dead_ug2_levels or self.partial_ug2_levels)


def _group_by_ug2(
    points: List[Dict],
    cluster_thr: float = UG2_CLUSTER_THR,
) -> Dict[float, List[Dict]]:
    """Group points by rounded Ug2 nominal value."""
    raw = sorted({round(p.get("ug2", 0), 0) for p in points})
    noms = _cluster_nominal(raw, threshold=cluster_thr)
    groups: Dict[float, List[Dict]] = {}
    for p in points:
        ug2 = _nominal_key(round(p.get("ug2", 0), 0), noms)
        groups.setdefault(ug2, []).append(p)
    return groups


def _is_dead_point(p: Dict, thr: float = IA_DEAD_THR) -> bool:
    return abs(p.get("ia", 0)) <= thr


def detect_dead_data(
    points: List[Dict],
    ia_thr: float = IA_DEAD_THR,
    topology: str = TOPOLOGY_PENTODE,
) -> DeadDataReport:
    """Analyse measurement points for hardware/software protection artefacts.

    Detection strategy (pentode scan order: Ug2 → Ug1(-8→-1) → Ua):

    1. **Dead Ug2 level**: max(Ia) across entire level <= ia_thr.
       Hardware protection fired during _settle_ug2() — all points dead.

    2. **Partial Ig2 protection within a level**: Ug1 iterates from most
       negative to least negative.  Protection fires at some Ug1=X, killing
       all subsequent curves (Ug1 ≥ X).  Find the transition point where
       live data switches to dead.  Normal cutoff (Ia≈0 at very negative
       Ug1) is the opposite pattern and is not flagged.

    For triodes (and triode_connected pentodes) the Ug2 dimension is not
    independent; only per-curve dead-point counting is done (can't
    distinguish cutoff from protection without Ug2 structure).

    Args:
        topology: "triode", "triode_connected", or "pentode".

    Returns:
        DeadDataReport with dead levels, partial levels, and counts.
    """
    if not points:
        return DeadDataReport(total_points=0, dead_points=0)

    _has_independent_ug2 = topology not in (TOPOLOGY_TRIODE, TOPOLOGY_TRIODE_CONNECTED)

    if not _has_independent_ug2:
        # For triodes: just count obviously dead points but don't remove
        # (can't distinguish cutoff from protection without Ug2 structure)
        dead_count = sum(1 for p in points if _is_dead_point(p, ia_thr))
        return DeadDataReport(total_points=len(points), dead_points=dead_count)

    # --- Pentode / tetrode analysis ---
    ug2_groups = _group_by_ug2(points)
    dead_levels: List[float] = []
    partial_levels: List[Tuple[float, float]] = []
    total_dead = 0

    for ug2_val in sorted(ug2_groups.keys()):
        level_pts = ug2_groups[ug2_val]
        max_ia = max(abs(p.get("ia", 0)) for p in level_pts)

        # Case 1: entire level is dead
        if max_ia <= ia_thr:
            dead_levels.append(ug2_val)
            total_dead += len(level_pts)
            continue

        # Case 2: partial protection within level
        # Group by Ug1 within this Ug2 level
        ug1_raw = sorted({round(p["ug1"], UG1_ROUND) for p in level_pts})
        ug1_noms = _cluster_nominal(ug1_raw, threshold=UG1_CLUSTER_THR)
        ug1_groups: Dict[float, List[Dict]] = {}
        for p in level_pts:
            ug1 = _nominal_key(round(p["ug1"], UG1_ROUND), ug1_noms)
            ug1_groups.setdefault(ug1, []).append(p)

        # Scan order: Ug1 ascending (most negative → least negative).
        # Protection fires at some Ug1 and kills all subsequent.
        # So we look for: live curves (max Ia > thr) followed by dead
        # curves (max Ia < thr) as Ug1 increases toward 0.
        sorted_ug1 = sorted(ug1_groups.keys())  # ascending
        if len(sorted_ug1) < 2:
            continue

        # Find transition: last live Ug1 followed by dead Ug1
        # A dead Ug1-curve: all points have Ia < thr
        curve_live = []
        for ug1 in sorted_ug1:
            curve_pts = ug1_groups[ug1]
            curve_max_ia = max(abs(p.get("ia", 0)) for p in curve_pts)
            curve_live.append(curve_max_ia > ia_thr)

        # Find transition from live to dead (scanning from negative to 0)
        # Normal cutoff: dead at beginning (negative Ug1), live later — NOT flagged
        # Protection: live at beginning, dead later — flagged
        transition_ug1 = None
        for i in range(len(sorted_ug1) - 1):
            if curve_live[i] and not curve_live[i + 1]:
                # Check that ALL subsequent Ug1 are dead (protection pattern)
                if all(not curve_live[j] for j in range(i + 1, len(sorted_ug1))):
                    transition_ug1 = sorted_ug1[i + 1]
                    break

        if transition_ug1 is not None:
            partial_levels.append((ug2_val, transition_ug1))
            # Count dead points in this level (at Ug1 >= transition)
            for ug1 in sorted_ug1:
                if ug1 >= transition_ug1:
                    total_dead += len(ug1_groups[ug1])

    return DeadDataReport(
        total_points=len(points),
        dead_points=total_dead,
        dead_ug2_levels=dead_levels,
        partial_ug2_levels=partial_levels,
    )


def clean_dead_points(
    points: List[Dict],
    report: Optional[DeadDataReport] = None,
    ia_thr: float = IA_DEAD_THR,
    topology: str = TOPOLOGY_PENTODE,
) -> List[Dict]:
    """Remove dead points identified by detect_dead_data().

    Dead Ug2 levels are removed entirely.
    Partial levels: all points at Ug1 >= transition Ug1 are removed.
    Returns a new list (input is not modified).
    """
    if report is None:
        report = detect_dead_data(points, ia_thr=ia_thr, topology=topology)

    if not report.has_dead_data:
        return list(points)

    dead_ug2_set = set(report.dead_ug2_levels)
    # Build (ug2 → transition_ug1) map for partial levels
    partial_map: Dict[float, float] = {}
    for ug2_val, ug1_trans in report.partial_ug2_levels:
        partial_map[ug2_val] = ug1_trans

    ug2_groups = _group_by_ug2(points)
    result: List[Dict] = []

    for ug2_val in sorted(ug2_groups.keys()):
        if ug2_val in dead_ug2_set:
            continue  # skip entire level
        level_pts = ug2_groups[ug2_val]
        if ug2_val in partial_map:
            ug1_trans = partial_map[ug2_val]
            # Keep only points with Ug1 < transition
            for p in level_pts:
                if round(p["ug1"], UG1_ROUND) < ug1_trans - UG1_CLUSTER_THR:
                    result.append(p)
        else:
            result.extend(level_pts)

    return result
