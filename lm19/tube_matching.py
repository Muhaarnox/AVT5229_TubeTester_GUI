"""Tube matching: find similar tubes and group into matched pairs/quads.

Distance metric uses absolute measured values (Ia mA, S mS, R kOhm),
normalised by sample mean, with configurable weights.
"""


from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)

log = logging.getLogger(__name__)


class MatchCancelled(Exception):
    """Raised when a user cancels a long-running match operation via the
    ``progress`` callback. The caller (UI) is expected to swallow this
    and leave the previous match result intact."""

# ── Default weights ──────────────────────────────────────────────────

DEFAULT_WEIGHTS_PENTODE: Dict[str, float] = {"ia": 0.5, "s": 0.5, "r": 0.0}
DEFAULT_WEIGHTS_TRIODE: Dict[str, float] = {"ia": 0.4, "s": 0.3, "r": 0.3}

METRIC_KEYS = ("ia", "s", "r")

MEAN_FLOOR = 1e-9        # avoid division by zero in normalisation
DISTANCE_SCALE = 100.0   # scale factor to express distance in %-like units

# ── Matching protocols ───────────────────────────────────────────────
# What "matched" means depends on how the buyer's amplifier biases its
# tubes (theory sources: external_sources/theory/, aiken/apex/tubesound):
#   strict          — conservative default: servo and fixed-bias runs
#                     never mix (exact conditions-key equality).
#   shared_bias     — ONE bias adjustment feeds both tubes: they must
#                     draw the same current at the SAME grid voltage, so
#                     the Ia term uses the plan-point current and pairs
#                     whose predicted quiescent-current imbalance exceeds
#                     the gate are incomparable (inf BEFORE selection —
#                     the algorithm then picks the next-best candidates).
#   individual_bias — each tube has its own bias adjustment: the operator
#                     dials DC balance regardless of bias spread, so Ia
#                     carries no weight and matching ranks S (and R) at
#                     the reference current; a tube whose bias shift
#                     exceeds the amp's adjustment range cannot be biased
#                     there at all and is incomparable.
MATCHING_PROTOCOL_STRICT = "strict"
MATCHING_PROTOCOL_SHARED = "shared_bias"
MATCHING_PROTOCOL_INDIVIDUAL = "individual_bias"
# protocol semantics follow "Aiken — The Last Word On Biasing",
# "Apex Tube Matching" and "TubeSound — Tube Matching with a Tube
# Tester", see SOURCES_INDEX.md
MATCHING_PROTOCOLS = (
    MATCHING_PROTOCOL_STRICT,
    MATCHING_PROTOCOL_SHARED,
    MATCHING_PROTOCOL_INDIVIDUAL,
)
DEFAULT_MATCHING_PROTOCOL = MATCHING_PROTOCOL_STRICT

# ── Pair-matching algorithm choices ──────────────────────────────────
# ``greedy`` is the default: pick the globally tightest unused pair
# first, repeat. Ideal for "pick K best pairs from a box of tubes,
# return the rest" — guarantees that no closer pair is left
# unpaired. ``optimal`` is Hungarian (scipy ``linear_sum_assignment``)
# which minimises the sum of distances across all pairs — best when
# every tube must end up in some pair within ``max_delta``.
PAIR_ALGORITHM_GREEDY = "greedy"
PAIR_ALGORITHM_OPTIMAL = "optimal"
PAIR_ALGORITHMS = (PAIR_ALGORITHM_GREEDY, PAIR_ALGORITHM_OPTIMAL)
DEFAULT_PAIR_ALGORITHM = PAIR_ALGORITHM_GREEDY

# ── Anchor errors (similar mode) ─────────────────────────────────────
# Contract codes crossing the module boundary (the Health tab maps them
# to user-facing messages): a requested "Find similar" anchor that
# cannot rank must say WHY instead of silently ranking around another
# lamp.
ANCHOR_ERR_NOT_FOUND = "anchor_not_found"        # lamp has no usable record
ANCHOR_ERR_INCOMPATIBLE = "anchor_incompatible"  # fails the protocol pool rule
MATCH_ANCHOR_ERRORS = frozenset({
    ANCHOR_ERR_NOT_FOUND, ANCHOR_ERR_INCOMPATIBLE,
})

# Delta quality thresholds (%) — for Health tab single-point matching
DELTA_EXCELLENT = 2.0    # ≤2% — hi-fi grade
DELTA_GOOD = 5.0         # ≤5% — industry standard matched pair
DELTA_FAIR = 10.0        # ≤10% — acceptable for guitar amps

# Curve delta thresholds by amp class — for Compare tab curve matching
# (excellent, good, fair); above fair → poor
CURVE_DELTA_THRESHOLDS: Dict[str, Tuple[float, float, float]] = {
    "class_a":  (2.0, 5.0, 10.0),   # tight: knee not reached
    "class_ab": (5.0, 10.0, 20.0),  # balanced: knee contributes
    "class_b":  (8.0, 15.0, 30.0),  # relaxed: full curve matters
}

# Curve overlap thresholds for Compare tab matching
MIN_OVERLAP_POINTS = 10   # below this → incomparable (distance = inf)
WARN_OVERLAP_POINTS = 30  # below this → result valid but flagged ⚠


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class TubeRecord:
    """One tube measurement selected for matching."""
    lamp_id: str
    timestamp: str
    an: int               # anode number
    ia: float             # mA  (absolute, from srk or health raw)
    s: float              # mS  (transconductance)
    r: float              # kOhm (plate resistance)
    index: Optional[float] = None  # health index
    # Protocol-aware fields: current at the PLAN bias (equals ``ia`` for
    # a fixed-bias run; ``raw.ia_plan_ma`` for a servo run, None when a
    # legacy servo entry predates that field) and the servo bias shift
    # (0.0 for fixed-bias runs — they sit at the plan bias by definition).
    ia_plan: Optional[float] = None
    bias_shift: Optional[float] = None
    servo: bool = False
    entry: Optional[Dict] = field(default=None, repr=False)


@dataclass
class MatchGroup:
    """A group of matched tubes."""
    number: int
    records: List[TubeRecord]
    delta: float          # max pairwise distance within group
    # shared_bias protocol only: predicted quiescent-current imbalance of
    # the group in a common-bias amplifier (max pairwise |Δ ia_plan|, mA).
    iq_imbalance_ma: Optional[float] = None


@dataclass
class MatchResult:
    """Result of matching operation."""
    mode: str             # "similar" or "groups"
    groups: List[MatchGroup]
    unmatched: List[TubeRecord]
    anchor: Optional[TubeRecord] = None  # for "similar" mode
    # Conditions tuple the pool was ACTUALLY filtered by — in similar
    # mode the anchor's own operating point wins over the caller's bulk
    # tuple, and the UI conditions label must show the truth.
    conditions_used: Optional[Tuple] = None
    # similar mode: MATCH_ANCHOR_ERRORS code when the requested anchor
    # cannot rank (no usable record / fails the protocol pool rule).
    anchor_error: Optional[str] = None


def default_weights_for_mode(ug2_mode: str) -> Dict[str, float]:
    """Return default matching weights appropriate for the tube mode."""
    if ug2_mode in (TOPOLOGY_TRIODE, TOPOLOGY_TRIODE_CONNECTED):
        return dict(DEFAULT_WEIGHTS_TRIODE)
    return dict(DEFAULT_WEIGHTS_PENTODE)


def delta_quality(delta: float, amp_class: Optional[str] = None) -> str:
    """Classify delta into quality tier: excellent/good/fair/poor.

    If amp_class is provided, uses curve thresholds (Compare tab).
    Otherwise uses fixed thresholds (Health tab single-point).
    """
    if amp_class and amp_class in CURVE_DELTA_THRESHOLDS:
        excellent, good, fair = CURVE_DELTA_THRESHOLDS[amp_class]
    else:
        excellent, good, fair = DELTA_EXCELLENT, DELTA_GOOD, DELTA_FAIR
    if delta <= excellent:
        return "excellent"
    if delta <= good:
        return "good"
    if delta <= fair:
        return "fair"
    return "poor"


# ── Extracting TubeRecord from measurement dict ─────────────────────

def _extract_record(entry: Dict) -> Optional[TubeRecord]:
    """Extract TubeRecord from a health measurement dict.

    Returns None if required fields are missing.
    """
    lamp_id = entry.get("lamp_id")
    if not lamp_id:
        return None

    conditions = entry.get("conditions") or {}
    an = int(conditions.get("an", 1))
    timestamp = str(entry.get("timestamp", ""))

    # Absolute values: prefer health.raw.ia_op, then srk values
    health = entry.get("health") or {}
    raw = health.get("raw") or {}
    srk = entry.get("srk") or {}

    ia = raw.get("ia_op") if raw.get("ia_op") is not None else srk.get("ia_op")
    s = srk.get("s")
    r = srk.get("r")

    if ia is None or s is None or r is None:
        return None
    if not (ia >= 0 and s >= 0 and r >= 0):
        return None

    index_val = health.get("index")

    # Protocol-aware fields. A fixed-bias run IS the plan point, so its
    # plan current is the op current and its shift is exactly 0; a servo
    # run carries both in the health blocks (None for legacy entries).
    servo = bool(conditions.get("bias_servo", False))
    if servo:
        metrics = health.get("metrics") or {}
        ia_plan = raw.get("ia_plan_ma")
        bias_shift = metrics.get("bias_shift_v")
    else:
        ia_plan = float(ia)
        bias_shift = 0.0

    return TubeRecord(
        lamp_id=str(lamp_id),
        timestamp=timestamp,
        an=an,
        ia=float(ia),
        s=float(s),
        r=float(r),
        index=float(index_val) if index_val is not None else None,
        ia_plan=float(ia_plan) if ia_plan is not None else None,
        bias_shift=float(bias_shift) if bias_shift is not None else None,
        servo=servo,
        entry=entry,
    )


def _conditions_key(entry: Dict) -> Tuple:
    """Return a hashable key for matching conditions
    (Ua, Ug1, Ug2, ug2_mode, bias_servo).

    Includes ug2_mode because S and R are computed differently in pentode
    vs triode-connected mode (different regression models), so measurements
    in different modes are not comparable.

    Includes the bias-servo flag because a servo run sits at the reference
    current (Ug1 adjusted per tube) while a fixed-bias run sits at the plan
    bias — under the default strict protocol the two are not comparable.
    Whether the flag actually separates pools is decided per matching
    protocol by ``_conditions_compatible`` (shared_bias ignores it,
    individual_bias requires it).
    ``conditions.ug1`` always stores the PLAN bias, so servo runs of
    different tubes share a key despite different actual biases. Entries
    saved before the flag existed read as False (fixed-bias), which is
    exactly what they were.
    """
    c = entry.get("conditions") or {}
    return (
        round(float(c.get("ua", 0)), 1),
        round(float(c.get("ug1", 0)), 1),
        round(float(c.get("ug2", 0)), 1),
        c.get("ug2_mode", TOPOLOGY_PENTODE),
        bool(c.get("bias_servo", False)),
    )


# ── Select one measurement per lamp ─────────────────────────────────

def _conditions_compatible(key: Tuple, conditions: Tuple, protocol: str) -> bool:
    """Protocol-aware pool membership test.

    strict: exact 5-tuple equality — servo runs only match servo runs.
    shared_bias: the operating point (ua, plan ug1, ug2, ug2_mode) must
        match; the bias-servo flag is ignored — plan-point currents make
        the two run kinds comparable.
    individual_bias: the operating point must match AND the entry must be
        a servo run (only servo runs sit at the reference current where S
        values are comparable).
    """
    if protocol == MATCHING_PROTOCOL_STRICT:
        return key == conditions
    if key[:4] != conditions[:4]:
        return False
    if protocol == MATCHING_PROTOCOL_INDIVIDUAL:
        return bool(key[4])
    return True


def select_measurements(
    entries: List[Dict],
    use: str = "latest",
    anode: str = "each",
    conditions: Optional[Tuple] = None,
    protocol: str = DEFAULT_MATCHING_PROTOCOL,
) -> List[TubeRecord]:
    """Select one measurement per (lamp_id, anode) for matching.

    Args:
        entries: raw health measurement dicts
        use: "latest" (most recent) or "best" (highest index)
        anode: "each" (per anode) or "combined" (average An1+An2)
        conditions: if set, only include entries matching (Ua, Ug1, Ug2)
        protocol: matching protocol — controls how the bias-servo flag of
            the conditions key participates in pool membership (see
            ``_conditions_compatible``)

    Returns:
        List of TubeRecord, one per lamp (or per lamp+anode if anode="each").
    """
    # Filter by conditions
    if conditions is not None:
        entries = [e for e in entries
                   if _conditions_compatible(_conditions_key(e), conditions,
                                             protocol)]

    # Extract records
    records: List[TubeRecord] = []
    for e in entries:
        rec = _extract_record(e)
        if rec is not None:
            records.append(rec)

    if not records:
        return []

    # Group by (lamp_id, an)
    groups: Dict[Tuple[str, int], List[TubeRecord]] = {}
    for rec in records:
        key = (rec.lamp_id, rec.an)
        groups.setdefault(key, []).append(rec)

    # Pick one per group
    selected: Dict[Tuple[str, int], TubeRecord] = {}
    for key, recs in groups.items():
        if use == "best":
            recs_with_index = [r for r in recs if r.index is not None]
            if recs_with_index:
                selected[key] = max(recs_with_index, key=lambda r: r.index)
            else:
                selected[key] = max(recs, key=lambda r: r.timestamp)
        else:  # latest
            selected[key] = max(recs, key=lambda r: r.timestamp)

    if anode != "combined":
        return list(selected.values())

    # Combined: average An1 + An2 per lamp_id
    by_lamp: Dict[str, List[TubeRecord]] = {}
    for (lid, _an), rec in selected.items():
        by_lamp.setdefault(lid, []).append(rec)

    combined: List[TubeRecord] = []
    for lid, recs in by_lamp.items():
        if len(recs) == 1:
            combined.append(recs[0])
        else:
            # Average metrics across anodes
            avg_ia = sum(r.ia for r in recs) / len(recs)
            avg_s = sum(r.s for r in recs) / len(recs)
            avg_r = sum(r.r for r in recs) / len(recs)
            indices = [r.index for r in recs if r.index is not None]
            avg_index = sum(indices) / len(indices) if indices else None
            # Protocol fields: plan current averages like ia; the bias
            # shift keeps the worst (largest-|·|) anode — the adjustment
            # range gate must hold for BOTH systems of a combined tube.
            plans = [r.ia_plan for r in recs]
            avg_plan = (sum(plans) / len(plans)
                        if all(p is not None for p in plans) else None)
            shifts = [r.bias_shift for r in recs]
            worst_shift = (max(shifts, key=abs)
                           if all(sh is not None for sh in shifts) else None)
            # Use the latest record as base
            base = max(recs, key=lambda r: r.timestamp)
            combined.append(TubeRecord(
                lamp_id=lid,
                timestamp=base.timestamp,
                an=0,  # 0 = combined
                ia=avg_ia,
                s=avg_s,
                r=avg_r,
                index=avg_index,
                ia_plan=avg_plan,
                bias_shift=worst_shift,
                servo=all(r.servo for r in recs),
                entry=base.entry,
            ))
    return combined


# ── Distance computation ─────────────────────────────────────────────

def compute_sample_means(records: List[TubeRecord]) -> Dict[str, float]:
    """Compute mean of each metric across records (for normalisation)."""
    n = len(records)
    if n == 0:
        return {k: 1.0 for k in METRIC_KEYS}
    means = {}
    for key in METRIC_KEYS:
        total = sum(getattr(r, key) for r in records)
        means[key] = total / n if n > 0 else 1.0
    # Avoid division by zero
    for key in means:
        if means[key] < MEAN_FLOOR:
            means[key] = 1.0
    return means


def compute_distance(
    a: TubeRecord,
    b: TubeRecord,
    weights: Dict[str, float],
    means: Dict[str, float],
) -> float:
    """Normalised weighted Euclidean distance between two tubes.

    Each metric is normalised by its sample mean before weighting.
    Result is in percent-like units.
    """
    total = 0.0
    for key in METRIC_KEYS:
        w = weights.get(key, 0.0)
        if w <= 0:
            continue
        va = getattr(a, key)
        vb = getattr(b, key)
        mean = means.get(key, 1.0)
        diff = (va - vb) / mean
        total += w * diff * diff
    return DISTANCE_SCALE * math.sqrt(total)


# ── Protocol machinery ──────────────────────────────────────────────

def predicted_iq_imbalance_ma(a: TubeRecord, b: TubeRecord) -> Optional[float]:
    """Predicted quiescent-current imbalance in a shared-bias amplifier.

    With one common grid voltage the two tubes settle at their plan-point
    currents, so the imbalance is directly |ia_plan(a) − ia_plan(b)| —
    the same quantity as |Δbias_a − Δbias_b|·S̄ but observed, not
    estimated. None when either plan current is unknown (legacy servo
    entry without ``raw.ia_plan_ma``).
    """
    if a.ia_plan is None or b.ia_plan is None:
        return None
    return abs(a.ia_plan - b.ia_plan)


def _weights_for_protocol(weights: Dict[str, float],
                          protocol: str) -> Dict[str, float]:
    """individual_bias: Ia carries no information (the operator dials DC
    balance per tube) — zero its weight and renormalise S/R so distances
    stay in the same %-like units as the other protocols."""
    if protocol != MATCHING_PROTOCOL_INDIVIDUAL:
        return weights
    w = dict(weights)
    w["ia"] = 0.0
    total = w.get("s", 0.0) + w.get("r", 0.0)
    if total <= 0:
        w["s"] = 1.0
    else:
        w["s"] = w.get("s", 0.0) / total
        w["r"] = w.get("r", 0.0) / total
    return w


def _shared_ia_records(records: List[TubeRecord],
                       protocol: str) -> List[TubeRecord]:
    """shared_bias: the matched current is the PLAN-point one — a servo
    run's op Ia sits at the reference by construction and would fake a
    perfect Ia match. Records keep their identity; only the metric value
    changes. Legacy servo entries without a plan current keep op Ia but
    are excluded pairwise by the protocol gate (never matched silently).
    """
    if protocol != MATCHING_PROTOCOL_SHARED:
        return records
    out = []
    for r in records:
        if r.ia_plan is not None and r.ia_plan != r.ia:
            out.append(replace(r, ia=r.ia_plan))
        else:
            out.append(r)
    return out


def _protocol_pair_allowed(a: TubeRecord, b: TubeRecord, protocol: str,
                           max_iq_imbalance_pct: float,
                           bias_adjust_limit_v: float) -> bool:
    """Pairwise protocol gate, applied BEFORE selection (a blocked pair
    turns to inf in the distance matrix, so the grouping algorithm walks
    on to the next-best candidates instead of post-filtering).

    shared_bias: both plan currents must be known; predicted δIq above
        ``max_iq_imbalance_pct`` (percent of the pair's mean plan
        current) is incomparable. 0 disables the δIq gate.
    individual_bias: both bias shifts must be known and each must fit the
        amplifier's bias-adjustment authority ``bias_adjust_limit_v``
        (absolute volts, pre-computed from the percent config by the
        caller; 0 disables the range gate).
    """
    if protocol == MATCHING_PROTOCOL_SHARED:
        diq = predicted_iq_imbalance_ma(a, b)
        if diq is None:
            return False
        if max_iq_imbalance_pct > 0:
            mean_ia = 0.5 * (a.ia_plan + b.ia_plan)
            if mean_ia > 0 and diq > max_iq_imbalance_pct / 100.0 * mean_ia:
                return False
        return True
    if protocol == MATCHING_PROTOCOL_INDIVIDUAL:
        for r in (a, b):
            if r.bias_shift is None:
                return False
            if bias_adjust_limit_v > 0 and abs(r.bias_shift) > bias_adjust_limit_v:
                return False
        return True
    return True


# ── Find similar ─────────────────────────────────────────────────────

def find_similar(
    anchor: TubeRecord,
    records: List[TubeRecord],
    weights: Optional[Dict[str, float]] = None,
    max_delta: float = 0.0,
    allowed: Optional[Callable[[TubeRecord, TubeRecord], bool]] = None,
) -> MatchResult:
    """Rank all records by distance from anchor.

    Args:
        anchor: reference tube
        records: all candidates (anchor is excluded automatically)
        weights: metric weights (default: DEFAULT_WEIGHTS_PENTODE)
        max_delta: exclude tubes with delta > this (0 = no limit)
        allowed: optional pairwise protocol gate — candidates failing it
            against the anchor are excluded from the ranking (same
            semantics as inf in the groups-mode distance matrix)

    Returns:
        MatchResult with mode="similar", groups as single-element groups
        sorted by delta ascending. Candidates cut by the protocol gate or
        by ``max_delta`` land in ``unmatched`` (visible, never silently
        dropped — groups mode surfaces the same records the same way).
    """
    weights = weights or dict(DEFAULT_WEIGHTS_PENTODE)
    all_recs = [anchor] + [r for r in records if r is not anchor]
    means = compute_sample_means(all_recs)

    others: List[TubeRecord] = []
    for r in records:
        if r is anchor:
            continue
        # Same lamp: exclude the same anode system, and ALWAYS exclude a
        # combined (an == 0) twin — it aggregates the anchor lamp itself,
        # so ranking it would find the tube "similar" to itself. Another
        # real anode system of the same envelope stays a candidate.
        if r.lamp_id == anchor.lamp_id and (
                r.an == anchor.an or 0 in (r.an, anchor.an)):
            continue
        others.append(r)

    ranked: List[Tuple[TubeRecord, float]] = []
    excluded: List[TubeRecord] = []
    for rec in others:
        if allowed is not None and not allowed(anchor, rec):
            excluded.append(rec)
            continue
        d = compute_distance(anchor, rec, weights, means)
        ranked.append((rec, d))
    ranked.sort(key=lambda x: x[1])

    if max_delta > 0:
        excluded.extend(r for r, d in ranked if d > max_delta)
        ranked = [(r, d) for r, d in ranked if d <= max_delta]

    groups = []
    for i, (rec, d) in enumerate(ranked):
        groups.append(MatchGroup(number=i + 1, records=[rec], delta=d))

    return MatchResult(
        mode="similar",
        groups=groups,
        unmatched=excluded,
        anchor=anchor,
    )


# ── Group into pairs/quads ──────────────────────────────────────────

def _build_distance_matrix(
    records: List[TubeRecord],
    weights: Dict[str, float],
    means: Dict[str, float],
    allowed: Optional[Callable[[TubeRecord, TubeRecord], bool]] = None,
) -> List[List[float]]:
    """Build NxN distance matrix.

    Pairs rejected by the ``allowed`` protocol gate become inf
    ("incomparable", ML-069 semantics) BEFORE any grouping runs — the
    selection algorithms then walk on to the next-best candidates.
    """
    n = len(records)
    matrix = [[0.0] * n for _ in range(n)]
    inf = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            if allowed is not None and not allowed(records[i], records[j]):
                d = inf
            else:
                d = compute_distance(records[i], records[j], weights, means)
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def _hungarian_assign(cost: List[List[float]]) -> List[Tuple[int, int]]:
    """Optimal min-weight matching for pairs using scipy or fallback.

    Returns list of (row, col) assignments.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        return list(zip(row_ind.tolist(), col_ind.tolist()))
    except ImportError:
        log.info("scipy not available, using greedy pair matching")
        return _greedy_pair_assign(cost)


def _greedy_pair_assign(cost: List[List[float]]) -> List[Tuple[int, int]]:
    """Greedy fallback: pick closest unmatched pair repeatedly."""
    n = len(cost)
    edges: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((cost[i][j], i, j))
    edges.sort()

    used = set()
    pairs: List[Tuple[int, int]] = []
    for d, i, j in edges:
        if i not in used and j not in used:
            pairs.append((i, j))
            used.add(i)
            used.add(j)
    return pairs


def group_by_distance_matrix(
    dist: List[List[float]],
    labels: List[str],
    group_size: int = 2,
    max_delta: float = 0.0,
    algorithm: str = DEFAULT_PAIR_ALGORITHM,
) -> Tuple[List[Tuple[List[int], float]], List[int]]:
    """Group items by pre-computed distance matrix.

    This is the shared core for both Health (Ia/S/R metric) and
    Compare (curve match_pct metric) tab matching.

    Args:
        dist: NxN distance matrix (0 = identical, inf = incomparable)
        labels: N labels (for logging only)
        group_size: 2 (pairs), 4 (quads), etc.
        max_delta: max allowed distance (0 = no limit)
        algorithm: pair selection strategy for ``group_size == 2`` —
            ``"greedy"`` (default, tightest-first) or ``"optimal"``
            (Hungarian, minimum sum of distances). Ignored for
            ``group_size > 2`` which always uses the avg-distance
            chunking strategy.

    Returns:
        (groups, unmatched) where:
        - groups: list of (indices, delta) — each group is a list of matrix
          indices and the max pairwise distance within the group
        - unmatched: list of indices not in any group
    """
    n = len(dist)
    if n < group_size:
        return [], list(range(n))

    if group_size == 2:
        return _group_pairs_from_matrix(dist, n, max_delta, algorithm)
    return _group_n_from_matrix(dist, n, group_size, max_delta)


def _group_pairs_from_matrix(
    dist: List[List[float]],
    n: int,
    max_delta: float,
    algorithm: str = DEFAULT_PAIR_ALGORITHM,
) -> Tuple[List[Tuple[List[int], float]], List[int]]:
    """Pair grouping on distance matrix.

    ``algorithm``:
      - ``"greedy"`` (default): pick the globally tightest unused pair
        first, repeat. Best for "pick K closest pairs from a larger
        pool, return the rest" — guarantees that no closer pair than
        the ones picked is left unpaired.
      - ``"optimal"``: Hungarian (scipy ``linear_sum_assignment``),
        falls back to greedy when scipy is unavailable. Minimises the
        **sum** of pair distances across all matched pairs — best
        when every tube must end up in some pair within ``max_delta``.
    """
    if algorithm == PAIR_ALGORITHM_GREEDY:
        return _group_pairs_greedy(dist, n, max_delta)
    if algorithm == PAIR_ALGORITHM_OPTIMAL:
        return _group_pairs_optimal(dist, n, max_delta)
    raise ValueError(
        f"Unknown pair algorithm {algorithm!r}; "
        f"expected one of {PAIR_ALGORITHMS}"
    )


def _group_pairs_greedy(
    dist: List[List[float]],
    n: int,
    max_delta: float,
) -> Tuple[List[Tuple[List[int], float]], List[int]]:
    """Tightest-first greedy matching.

    Enumerates all C(n, 2) pairs, sorts by distance ascending, and
    forms pairs in order while both endpoints are unused and the
    distance is within ``max_delta``. Once a pair exceeds the limit
    every subsequent pair would too (already sorted), so we stop
    early. Unmatched tubes return to the unmatched list.

    Compared to Hungarian: this never sacrifices a tight pair for a
    smaller sum, so the picked pairs are individually as close as
    possible. Total sum may be worse and more tubes may end up
    unmatched.
    """
    edges: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = dist[i][j]
            if math.isinf(d):
                continue
            edges.append((d, i, j))
    edges.sort()

    groups: List[Tuple[List[int], float]] = []
    matched: set = set()
    for d, i, j in edges:
        if max_delta > 0 and d > max_delta:
            break  # remaining edges are all worse
        if i in matched or j in matched:
            continue
        groups.append(([i, j], d))
        matched.add(i)
        matched.add(j)
    unmatched = [i for i in range(n) if i not in matched]
    return groups, unmatched


def _group_pairs_optimal(
    dist: List[List[float]],
    n: int,
    max_delta: float,
) -> Tuple[List[Tuple[List[int], float]], List[int]]:
    """Sum-optimal pair matching via Hungarian assignment."""
    INF = 1e9
    # Pad to even size: dummy node with zero cost
    size = n + (n % 2)
    cost = [[0.0] * size for _ in range(size)]
    for i in range(n):
        for j in range(n):
            d = dist[i][j]
            cost[i][j] = INF if i == j else (INF if math.isinf(d) else d)
        for j in range(n, size):
            cost[i][j] = 0.0
    for i in range(n, size):
        for j in range(size):
            cost[i][j] = 0.0

    assignments = _hungarian_assign(cost)

    seen = set()
    raw_pairs: List[Tuple[int, int, float]] = []
    for i, j in assignments:
        if i >= n or j >= n:
            continue
        key = (min(i, j), max(i, j))
        if key not in seen and i != j:
            seen.add(key)
            d = dist[key[0]][key[1]]
            if not math.isinf(d):
                raw_pairs.append((key[0], key[1], d))
    raw_pairs.sort(key=lambda x: x[2])

    groups: List[Tuple[List[int], float]] = []
    matched = set()
    for i, j, d in raw_pairs:
        if max_delta > 0 and d > max_delta:
            continue
        groups.append(([i, j], d))
        matched.add(i)
        matched.add(j)

    unmatched = [i for i in range(n) if i not in matched]
    return groups, unmatched


def _group_n_from_matrix(
    dist: List[List[float]],
    n: int,
    group_size: int,
    max_delta: float,
) -> Tuple[List[Tuple[List[int], float]], List[int]]:
    """Group N items using greedy sort on distance matrix.

    Sort by average distance to all others, then form groups from adjacent.
    """
    # Sort by average distance (ignoring inf)
    avg_dist: List[Tuple[float, int]] = []
    for i in range(n):
        finite = [dist[i][j] for j in range(n) if j != i and not math.isinf(dist[i][j])]
        avg = sum(finite) / len(finite) if finite else float("inf")
        avg_dist.append((avg, i))
    avg_dist.sort()
    sorted_indices = [idx for _, idx in avg_dist]

    groups: List[Tuple[List[int], float]] = []
    unmatched: List[int] = []
    pos = 0
    while pos + group_size <= len(sorted_indices):
        chunk = sorted_indices[pos:pos + group_size]
        max_d = 0.0
        for a in range(len(chunk)):
            for b in range(a + 1, len(chunk)):
                d = dist[chunk[a]][chunk[b]]
                max_d = max(max_d, d) if not math.isinf(d) else float("inf")
        # ML-069: inf means "incomparable" (different conditions / same
        # lamp), never "large distance" — such chunks must not group even
        # when the max_delta threshold filter is disabled (== 0).
        if math.isinf(max_d) or (max_delta > 0 and max_d > max_delta):
            unmatched.extend(chunk)
        else:
            groups.append((chunk, max_d))
        pos += group_size

    while pos < len(sorted_indices):
        unmatched.append(sorted_indices[pos])
        pos += 1

    return groups, unmatched


def group_pairs(
    records: List[TubeRecord],
    weights: Optional[Dict[str, float]] = None,
    max_delta: float = 0.0,
    algorithm: str = DEFAULT_PAIR_ALGORITHM,
    allowed: Optional[Callable[[TubeRecord, TubeRecord], bool]] = None,
) -> MatchResult:
    """Partition records into pairs.

    Args:
        records: tube records to pair.
        weights: per-metric weights for the distance function.
        max_delta: drop pairs above this distance (0 disables the filter).
        algorithm: ``"greedy"`` (default, tightest-first) or
            ``"optimal"`` (Hungarian min-sum). See
            ``_group_pairs_from_matrix`` for the tradeoff.
        allowed: optional pairwise protocol gate → inf in the matrix.
    """
    weights = weights or dict(DEFAULT_WEIGHTS_PENTODE)
    if len(records) < 2:
        return MatchResult(
            mode="groups",
            groups=[],
            unmatched=list(records),
        )

    means = compute_sample_means(records)
    dist = _build_distance_matrix(records, weights, means, allowed=allowed)
    labels = [r.lamp_id for r in records]

    raw_groups, raw_unmatched = group_by_distance_matrix(
        dist, labels, group_size=2, max_delta=max_delta, algorithm=algorithm)

    groups = [
        MatchGroup(number=i + 1,
                   records=[records[idx] for idx in indices],
                   delta=delta)
        for i, (indices, delta) in enumerate(raw_groups)
    ]
    unmatched = [records[idx] for idx in raw_unmatched]
    return MatchResult(mode="groups", groups=groups, unmatched=unmatched)


def group_n(
    records: List[TubeRecord],
    group_size: int = 4,
    weights: Optional[Dict[str, float]] = None,
    max_delta: float = 0.0,
    allowed: Optional[Callable[[TubeRecord, TubeRecord], bool]] = None,
) -> MatchResult:
    """Partition records into groups of group_size using greedy sort.

    Delegates to group_by_distance_matrix() after building the matrix.
    """
    weights = weights or dict(DEFAULT_WEIGHTS_PENTODE)
    if len(records) < group_size:
        return MatchResult(
            mode="groups",
            groups=[],
            unmatched=list(records),
        )

    means = compute_sample_means(records)
    dist = _build_distance_matrix(records, weights, means, allowed=allowed)
    labels = [r.lamp_id for r in records]

    raw_groups, raw_unmatched = group_by_distance_matrix(
        dist, labels, group_size=group_size, max_delta=max_delta)

    groups = [
        MatchGroup(number=i + 1,
                   records=[records[idx] for idx in indices],
                   delta=delta)
        for i, (indices, delta) in enumerate(raw_groups)
    ]
    unmatched = [records[idx] for idx in raw_unmatched]
    return MatchResult(mode="groups", groups=groups, unmatched=unmatched)


# ── High-level entry point ───────────────────────────────────────────

def _fill_iq_imbalance(result: MatchResult, protocol: str) -> None:
    """shared_bias only: annotate each group with its predicted δIq.

    Groups mode: max pairwise |Δ ia_plan| within the group. Similar mode:
    each single-record group vs the anchor.
    """
    if protocol != MATCHING_PROTOCOL_SHARED:
        return
    for g in result.groups:
        if result.mode == "similar" and result.anchor is not None and g.records:
            g.iq_imbalance_ma = predicted_iq_imbalance_ma(
                result.anchor, g.records[0])
            continue
        worst: Optional[float] = None
        for i in range(len(g.records)):
            for j in range(i + 1, len(g.records)):
                diq = predicted_iq_imbalance_ma(g.records[i], g.records[j])
                if diq is not None and (worst is None or diq > worst):
                    worst = diq
        g.iq_imbalance_ma = worst


def _warn_shared_legacy(records: List[TubeRecord], protocol: str) -> None:
    """Failure visibility: shared_bias records without a plan-point
    current fail every pairwise gate and can only surface as unmatched —
    say WHY in the log."""
    if protocol != MATCHING_PROTOCOL_SHARED:
        return
    legacy = sorted(r.lamp_id for r in records if r.ia_plan is None)
    if legacy:
        log.warning(
            "shared_bias matching: %d record(s) lack the plan-point "
            "current (servo runs saved before raw.ia_plan_ma "
            "existed) and can only end up unmatched: %s",
            len(legacy), ", ".join(legacy))


def _pick_anchor_entry(
    entries: List[Dict],
    lamp_id: str,
    an: Optional[int],
    use: str,
    ug2_mode: Optional[str],
    protocol: str,
) -> Optional[Dict]:
    """The anchor lamp's own latest/best entry for default "Find similar".

    The pool must be defined by the ANCHOR lamp's operating point — the
    newest entry of another lamp deciding the pool silently anchored the
    ranking on a different lamp whenever the clicked lamp was absent
    from that pool (likely with the bias-servo flag in the key: a mixed
    servo/fixed history at one OP splits into two pools).

    Filters: same lamp; same anode when *an* is given (falls back to any
    anode when the lamp has none recorded for it); same ug2_mode when
    given; protocol-eligible (the entry's conditions key passes the pool
    rule against itself — individual_bias admits only servo runs, so the
    lamp's latest SERVO run anchors even when a fixed run is newer).
    Returns None when nothing survives — the caller reports
    ANCHOR_ERR_NOT_FOUND instead of anchoring elsewhere.
    """
    cand = [e for e in entries if str(e.get("lamp_id", "")) == lamp_id]
    if an is not None:
        with_an = [e for e in cand
                   if int((e.get("conditions") or {}).get("an", 1)) == an]
        cand = with_an or cand
    if ug2_mode is not None:
        cand = [e for e in cand if _conditions_key(e)[3] == ug2_mode]
    cand = [e for e in cand
            if _conditions_compatible(_conditions_key(e), _conditions_key(e),
                                      protocol)]
    if not cand:
        return None
    if use == "best":
        with_index = [e for e in cand
                      if isinstance((e.get("health") or {}).get("index"),
                                    (int, float))]
        if with_index:
            return max(with_index,
                       key=lambda e: float(e["health"]["index"]))
    return max(cand, key=lambda e: str(e.get("timestamp", "")))


def match_tubes(
    entries: List[Dict],
    mode: str = "groups",
    group_size: int = 2,
    use: str = "latest",
    anode: str = "each",
    weights: Optional[Dict[str, float]] = None,
    max_delta: float = 0.0,
    conditions: Optional[Tuple] = None,
    anchor_lamp_id: Optional[str] = None,
    anchor_an: Optional[int] = None,
    anchor_timestamp: Optional[str] = None,
    algorithm: str = DEFAULT_PAIR_ALGORITHM,
    protocol: str = DEFAULT_MATCHING_PROTOCOL,
    max_iq_imbalance_pct: float = 0.0,
    bias_adjust_range_pct: float = 0.0,
) -> MatchResult:
    """Main entry point for tube matching.

    Args:
        entries: raw health measurement dicts
        mode: "groups" or "similar"
        group_size: 2 (pairs), 4 (quads), etc.
        use: "latest" or "best"
        anode: "each" or "combined"
        weights: metric weights
        max_delta: max allowed distance (0 = no limit)
        conditions: (Ua, Ug1, Ug2) filter, None = auto-detect from latest
        anchor_lamp_id: for "similar" mode — reference lamp. The pool and
            the gate are built from the ANCHOR's own conditions key (its
            latest/best eligible entry, or the anchor_timestamp one); a
            requested anchor that cannot rank yields an empty result with
            ``anchor_error`` set (MATCH_ANCHOR_ERRORS) — never a silent
            ranking around another lamp
        anchor_an: for "similar" mode — reference anode
        anchor_timestamp: for "similar" mode — anchor on this SPECIFIC
            measurement of the lamp (by timestamp) instead of its latest/best;
            None = use the lamp's latest/best (the default "Find similar")
        algorithm: pair-matching algorithm — ``"greedy"`` (default,
            tightest-first) or ``"optimal"`` (Hungarian min-sum).
            Only affects ``mode="groups"`` with ``group_size=2``.
        protocol: matching protocol (``MATCHING_PROTOCOLS``) — how the
            buyer's amplifier biases its tubes decides the pool, the
            metric and the pairwise gates; see the registry comment.
        max_iq_imbalance_pct: shared_bias gate — max predicted δIq as a
            percent of the pair's mean plan current (0 = off).
        bias_adjust_range_pct: individual_bias gate — the amplifier's
            bias-adjustment authority as a percent of the PLAN bias
            voltage (0 = off). Percent, not volts: a fixed volt span
            would be huge for a −2 V high-mu triode and negligible for
            a −60 V transmitting tube.

    Returns:
        MatchResult
    """
    if protocol not in MATCHING_PROTOCOLS:
        raise ValueError(
            f"Unknown matching protocol {protocol!r}; "
            f"expected one of {MATCHING_PROTOCOLS}"
        )
    weights = weights or dict(DEFAULT_WEIGHTS_PENTODE)
    weights = _weights_for_protocol(weights, protocol)

    # Auto-detect conditions from latest entry.
    # Assumes entries are sorted by timestamp descending (as returned by
    # list_health_entries / load_health_measurements).
    if conditions is None and entries:
        conditions = _conditions_key(entries[0])

    def _gate_for(
        cond: Optional[Tuple],
    ) -> Optional[Callable[[TubeRecord, TubeRecord], bool]]:
        if protocol == MATCHING_PROTOCOL_STRICT:
            return None
        # The individual-bias range gate scales from the PLAN bias of the
        # pool (cond[1]) — shared across all candidates by key
        # construction, so one absolute limit serves the whole pool.
        lim_v = 0.0
        if bias_adjust_range_pct > 0 and cond is not None:
            lim_v = bias_adjust_range_pct / 100.0 * abs(cond[1])

        def _gate(a: TubeRecord, b: TubeRecord, _lim: float = lim_v) -> bool:
            return _protocol_pair_allowed(
                a, b, protocol, max_iq_imbalance_pct, _lim)
        return _gate

    allowed = _gate_for(conditions)

    def _prepare(cond: Optional[Tuple]) -> List[TubeRecord]:
        recs = select_measurements(entries, use=use, anode=anode,
                                   conditions=cond, protocol=protocol)
        _warn_shared_legacy(recs, protocol)
        return _shared_ia_records(recs, protocol)

    if mode == "similar":
        anchor: Optional[TubeRecord] = None
        anchor_cond = conditions
        anchor_entry: Optional[Dict] = None

        def _fail(code: str) -> MatchResult:
            log.warning("similar-mode anchor %r unusable: %s",
                        anchor_lamp_id, code)
            return MatchResult(mode="similar", groups=[], unmatched=[],
                               anchor=None, conditions_used=anchor_cond,
                               anchor_error=code)

        # Anchor on a SPECIFIC clicked measurement (Find similar / this
        # measurement); an absent timestamp falls through to the lamp's
        # latest/best below.
        if anchor_lamp_id is not None and anchor_timestamp is not None:
            for e in entries:
                if (str(e.get("lamp_id", "")) == anchor_lamp_id
                        and str(e.get("timestamp", "")) == anchor_timestamp):
                    anchor_entry = e
                    break
        specific = anchor_entry is not None
        if anchor_entry is None and anchor_lamp_id is not None:
            # Default "Find similar": the anchor lamp's own latest/best
            # eligible entry defines the pool (see _pick_anchor_entry).
            anchor_entry = _pick_anchor_entry(
                entries, anchor_lamp_id, anchor_an, use,
                ug2_mode=conditions[3] if conditions else None,
                protocol=protocol)
            if anchor_entry is None:
                return _fail(ANCHOR_ERR_NOT_FOUND)

        if anchor_entry is not None:
            # Rank candidates at the ANCHOR's operating point, not the
            # auto-detected bulk conditions — otherwise the anchor (at
            # its own Ua/Ug1/Ug2) is distance-skewed against candidates
            # measured at a different point. The gate limit follows the
            # anchor's plan bias too.
            anchor_cond = _conditions_key(anchor_entry)
            # The anchor must satisfy the pool rule it imposes on the
            # candidates (individual_bias: only servo runs carry an S
            # comparable at the reference current).
            if not _conditions_compatible(anchor_cond, anchor_cond,
                                          protocol):
                return _fail(ANCHOR_ERR_INCOMPATIBLE)
            allowed = _gate_for(anchor_cond)
            records = _prepare(anchor_cond)
            if specific:
                rec = _extract_record(anchor_entry)
                if rec is None:
                    return _fail(ANCHOR_ERR_NOT_FOUND)
                anchor = _shared_ia_records([rec], protocol)[0]
            else:
                # The pool's own record for the lamp — keeps combined
                # anode semantics (an == 0 aggregates the requested one).
                for r in records:
                    if r.lamp_id == anchor_lamp_id and (
                            anchor_an is None or r.an in (anchor_an, 0)):
                        anchor = r
                        break
                if anchor is None:
                    return _fail(ANCHOR_ERR_NOT_FOUND)
            if (protocol == MATCHING_PROTOCOL_SHARED
                    and anchor.ia_plan is None):
                log.warning(
                    "shared_bias matching: anchor %s lacks the plan-point "
                    "current — every candidate fails the pairwise gate",
                    anchor.lamp_id)
        else:
            # Programmatic similar without an anchor request: rank around
            # the pool's first record, as before.
            records = _prepare(conditions)
            if records:
                anchor = records[0]
        if anchor is None:
            return MatchResult(mode="similar", groups=[], unmatched=[],
                               conditions_used=anchor_cond)
        result = find_similar(anchor, records, weights, max_delta,
                              allowed=allowed)
        result.conditions_used = anchor_cond
        _fill_iq_imbalance(result, protocol)
        return result

    # Groups mode
    records = _prepare(conditions)
    if group_size == 2:
        result = group_pairs(records, weights, max_delta,
                             algorithm=algorithm, allowed=allowed)
    else:
        result = group_n(records, group_size, weights, max_delta,
                         allowed=allowed)
    result.conditions_used = conditions
    _fill_iq_imbalance(result, protocol)
    return result


# ── Curve-based matching (Compare tab) ─────────────────────────────

@dataclass
class CurveDistanceInfo:
    """Distance and metadata for one pair in curve-based matching."""
    distance: float       # 100 - match_pct (0 = identical, 100 = different)
    n_points: int         # number of interpolation points compared
    low_overlap: bool     # True if n_points < WARN_OVERLAP_POINTS


@dataclass
class CurveMatchResult:
    """Result of curve-based matching on Compare tab."""
    mode: str                        # "similar" or "groups"
    groups: List[MatchGroup]
    unmatched: List[int]             # indices into the entries list
    pair_info: Dict[Tuple[int, int], CurveDistanceInfo]  # (i,j) → info
    anchor_idx: Optional[int] = None  # for "similar" mode


def build_curve_distance_matrix(
    entries: List[Dict],
    min_overlap: int = MIN_OVERLAP_POINTS,
    amp_class: str = "class_ab",
    progress: Optional[Callable[[int, int], bool]] = None,
) -> Tuple[List[List[float]], Dict[Tuple[int, int], CurveDistanceInfo]]:
    """Build NxN distance matrix from curve matching (Compare tab).

    Uses compute_matching() from quality.py for each pair.
    Distance = 100 - match_pct.

    Args:
        entries: list of compare entries, each must have "points" key
        min_overlap: minimum common points to consider a pair comparable
        amp_class: weighting mode ("class_a", "class_ab", "class_b")
        progress: optional callback ``(done, total) -> keep_going``. Called
            once per outer iteration (``done`` rows of the matrix
            processed out of ``total = n``). Return ``False`` to abort —
            the function then raises :class:`MatchCancelled`.

    Returns:
        (dist_matrix, pair_info) where:
        - dist_matrix: NxN, inf = incomparable
        - pair_info: (i,j) → CurveDistanceInfo for each computed pair
    """
    from .quality import compute_matching as _compute_matching

    n = len(entries)
    INF = float("inf")
    dist = [[0.0 if i == j else INF for j in range(n)] for i in range(n)]
    pair_info: Dict[Tuple[int, int], CurveDistanceInfo] = {}

    for i in range(n):
        if progress is not None and not progress(i, n):
            raise MatchCancelled()
        pts_i = entries[i].get("points", [])
        if not pts_i:
            continue
        lid_i = entries[i].get("lamp_id", "")
        for j in range(i + 1, n):
            pts_j = entries[j].get("points", [])
            if not pts_j:
                continue

            # Same lamp_id → incomparable (don't match tube with itself)
            lid_j = entries[j].get("lamp_id", "")
            if lid_i and lid_j and lid_i == lid_j:
                info = CurveDistanceInfo(
                    distance=INF, n_points=0, low_overlap=False)
                pair_info[(i, j)] = info
                pair_info[(j, i)] = info
                continue

            result = _compute_matching(pts_i, pts_j, amp_class=amp_class)
            if result is None:
                info = CurveDistanceInfo(
                    distance=INF, n_points=0, low_overlap=True)
            elif result.n_points < min_overlap:
                info = CurveDistanceInfo(
                    distance=INF, n_points=result.n_points, low_overlap=True)
            else:
                d = max(0.0, 100.0 - result.match_pct)
                info = CurveDistanceInfo(
                    distance=d,
                    n_points=result.n_points,
                    low_overlap=result.n_points < WARN_OVERLAP_POINTS,
                )
                dist[i][j] = d
                dist[j][i] = d

            pair_info[(i, j)] = info
            pair_info[(j, i)] = info

    return dist, pair_info


def build_curve_distance_row(
    entries: List[Dict],
    anchor_idx: int,
    min_overlap: int = MIN_OVERLAP_POINTS,
    amp_class: str = "class_ab",
    progress: Optional[Callable[[int, int], bool]] = None,
) -> Tuple[List[float], Dict[Tuple[int, int], CurveDistanceInfo]]:
    """Distances from one anchor to every other entry — O(N) (ML-145).

    "Find similar" needs only the anchor's row, but used to pay for the
    full O(N²) :func:`build_curve_distance_matrix` and throw the rest
    away. Same per-pair math as the matrix; the returned ``row`` is
    self-distance 0 at the anchor and ``inf`` for incomparable pairs.

    Args:
        entries: compare entries with a "points" key.
        anchor_idx: index whose row is computed.
        min_overlap: minimum common points for a comparable pair.
        amp_class: weighting mode.
        progress: ``(done, total) -> keep_going``; aborts with
            :class:`MatchCancelled` on ``False``.

    Returns:
        (row, pair_info) — ``row[j]`` distance anchor→j (inf if
        incomparable), ``pair_info`` only for the anchor's pairs.
    """
    from .quality import compute_matching as _compute_matching

    n = len(entries)
    INF = float("inf")
    row = [INF] * n
    pair_info: Dict[Tuple[int, int], CurveDistanceInfo] = {}
    if not (0 <= anchor_idx < n):
        return row, pair_info
    row[anchor_idx] = 0.0

    pts_a = entries[anchor_idx].get("points", [])
    lid_a = entries[anchor_idx].get("lamp_id", "")
    for j in range(n):
        if progress is not None and not progress(j, n):
            raise MatchCancelled()
        if j == anchor_idx or not pts_a:
            continue
        pts_j = entries[j].get("points", [])
        if not pts_j:
            continue

        lid_j = entries[j].get("lamp_id", "")
        if lid_a and lid_j and lid_a == lid_j:
            info = CurveDistanceInfo(distance=INF, n_points=0, low_overlap=False)
            pair_info[(anchor_idx, j)] = info
            pair_info[(j, anchor_idx)] = info
            continue

        result = _compute_matching(pts_a, pts_j, amp_class=amp_class)
        if result is None:
            info = CurveDistanceInfo(distance=INF, n_points=0, low_overlap=True)
        elif result.n_points < min_overlap:
            info = CurveDistanceInfo(
                distance=INF, n_points=result.n_points, low_overlap=True)
        else:
            d = max(0.0, 100.0 - result.match_pct)
            info = CurveDistanceInfo(
                distance=d,
                n_points=result.n_points,
                low_overlap=result.n_points < WARN_OVERLAP_POINTS,
            )
            row[j] = d

        pair_info[(anchor_idx, j)] = info
        pair_info[(j, anchor_idx)] = info

    return row, pair_info


def match_curves(
    entries: List[Dict],
    labels: List[str],
    mode: str = "groups",
    group_size: int = 2,
    max_delta: float = 0.0,
    min_overlap: int = MIN_OVERLAP_POINTS,
    anchor_idx: Optional[int] = None,
    amp_class: str = "class_ab",
    algorithm: str = PAIR_ALGORITHM_OPTIMAL,
    progress: Optional[Callable[[int, int], bool]] = None,
) -> CurveMatchResult:
    """Main entry point for curve-based matching (Compare tab).

    Args:
        entries: compare entries with "points" key
        labels: display labels (lamp_id or name) per entry
        mode: "groups" or "similar"
        group_size: for groups mode
        max_delta: max allowed distance (0 = no limit)
        min_overlap: min common points for valid comparison
        anchor_idx: index of anchor entry for "similar" mode
        algorithm: pair-matching strategy for ``group_size=2``.
            Defaults to ``"optimal"`` (Hungarian) — preserves the
            original Compare-tab behaviour. Health tab passes
            ``"greedy"`` via the MatchPanel dropdown.
        progress: optional ``(done, total) -> keep_going`` callback,
            forwarded to :func:`build_curve_distance_matrix` (the
            O(N²) loop that does the heavy lifting). Returning
            ``False`` aborts with :class:`MatchCancelled`.

    Returns:
        CurveMatchResult
    """
    n = len(entries)
    if n < 2:
        return CurveMatchResult(
            mode=mode, groups=[], unmatched=list(range(n)),
            pair_info={}, anchor_idx=anchor_idx)

    if mode == "similar":
        if anchor_idx is None or anchor_idx >= n:
            anchor_idx = 0
        # ML-145: only the anchor row is needed — O(N), not O(N²).
        row, pair_info = build_curve_distance_row(
            entries, anchor_idx, min_overlap, amp_class=amp_class,
            progress=progress)
        # Rank all others by distance from anchor
        ranked: List[Tuple[int, float]] = []
        for j in range(n):
            if j == anchor_idx:
                continue
            d = row[j]
            if not math.isinf(d):
                ranked.append((j, d))
        ranked.sort(key=lambda x: x[1])

        if max_delta > 0:
            ranked = [(j, d) for j, d in ranked if d <= max_delta]

        groups = [
            MatchGroup(number=i + 1, records=[], delta=d)
            for i, (j, d) in enumerate(ranked)
        ]
        # Store index in records field via a lightweight TubeRecord
        for grp, (j, _d) in zip(groups, ranked):
            grp.records = [TubeRecord(
                lamp_id=labels[j] if j < len(labels) else str(j),
                timestamp="", an=0, ia=0, s=0, r=0,
            )]
            grp.records[0].entry = {"_index": j}

        unmatched_set = {j for j, _ in ranked} | {anchor_idx}
        unmatched = [i for i in range(n) if i not in unmatched_set]

        return CurveMatchResult(
            mode="similar", groups=groups, unmatched=unmatched,
            pair_info=pair_info, anchor_idx=anchor_idx)

    # Groups mode — needs the full O(N²) matrix.
    dist, pair_info = build_curve_distance_matrix(entries, min_overlap,
                                                   amp_class=amp_class,
                                                   progress=progress)
    raw_groups, raw_unmatched = group_by_distance_matrix(
        dist, labels, group_size=group_size, max_delta=max_delta,
        algorithm=algorithm)

    groups = [
        MatchGroup(number=i + 1,
                   records=[TubeRecord(
                       lamp_id=labels[idx] if idx < len(labels) else str(idx),
                       timestamp="", an=0, ia=0, s=0, r=0,
                       entry={"_index": idx},
                   ) for idx in indices],
                   delta=delta)
        for i, (indices, delta) in enumerate(raw_groups)
    ]

    return CurveMatchResult(
        mode=mode, groups=groups, unmatched=raw_unmatched,
        pair_info=pair_info, anchor_idx=anchor_idx)
