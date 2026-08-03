"""Distortion analysis: intersections, 5-point/Chebyshev/DFT, and PP variants.

Single-ended (SE):
  - ``find_intersections``         — load-line × measured Ia(Ua) curves
  - ``find_intersections_model``   — load-line × tube-model curves
  - ``interp_intersection``        — linear interp between intersection pts
  - ``compute_distortion``         — 5-point Fourier (HD2, HD3)
  - ``compute_distortion_chebyshev`` — Chebyshev polynomial fit (HD2..HDn)
  - ``compute_distortion_dft``     — model-based DFT (HD2..HDn)
  - ``compute_imd``                — IMD2/IMD3 from polynomial Ia(Ug1)
  - ``diagnose_distortion``        — DIST_ERR_* code or "" if compute will succeed

Push-pull (PP):
  - ``composite_characteristic``   — mirror Ia_b around bias and subtract
  - ``pp_distortion``              — 5-point on composite
  - ``compute_distortion_chebyshev_pp`` — Chebyshev on composite
  - ``compute_distortion_dft_pp``  — DFT with self-consistent per-tube Newton
  - ``diagnose_pp_distortion``     — PP-specific failure codes

Internal helpers ``_find_dc_q_point``, ``_find_model_dc_q_point``,
``_build_transfer_curve``, ``_interp_transfer`` support transformer AC
load lines and the composite construction.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np

from lm19.amplifier.constants import (
    HD_METHOD_CHEBYSHEV_MODEL_PP,
    B_EXTRAP_WARN_EDGE_FRACTION,
    BALANCE_SWING_NEAR_ZERO,
    BISECT_CONVERGENCE_V,
    CHEBYSHEV_BOUNDARY,
    CHEBYSHEV_OVERFIT_RATIO,
    CLASS_A_RATIO,
    CLASS_B_RATIO,
    CUTOFF_IA_MA,
    DIST_ERR_BIAS_AT_EDGE,
    DIST_ERR_BIAS_OUTSIDE,
    DIST_ERR_FEW_INTERSECTIONS,
    DIST_ERR_MANUAL_SWING_SMALL,
    DIST_ERR_NO_SIGNAL,
    DIST_ERR_PP_NO_COMPOSITE,
    DIST_ERR_PP_RA_INVALID,
    DIST_ERR_SPARSE_DATA,
    DIST_ERR_UNKNOWN,
    EDGE_TOL_ABS,
    EDGE_TOL_REL,
    FIXED_POINT_CONVERGENCE_MA,
    BIAS_MATCH_TOLERANCE_V,
    MIN_B1_MA,
    MIN_CHEBYSHEV_HARMONIC,
    MIN_CURVES_IN_SWING,
    MIN_IA_SWING_MA,
    MIN_POUT_MW,
    MIN_SWING_V,
    MIN_UA_SWING_V,
    RA_DC_NEAR_ZERO_KOHM,
    UG1_MATCH_TOLERANCE_V,
)
from lm19.amplifier.loadlines import (
    pp_working_line_ia,
    LoadLine,
    PushPullLoadLine,
    TransformerLoadLine,
)
from lm19.constants import (
    BISECT_MAX_ITER,
    BISECT_TOLERANCE_V,
    DEFAULT_UB_V,
    MODEL_SEARCH_POINTS,
    MODEL_UA_MAX_DEFAULT_V,
    MODEL_UA_MIN_V,
    UA_ROUND,
)
from lm19.tube_model_base import model_ia_array
from lm19.amplifier.constants import (
    HD_METHOD_5POINT,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_CHEBYSHEV_PP,
    HD_METHOD_DFT,
    HD_METHOD_DFT_PP,
)

if TYPE_CHECKING:
    from lm19.tube_model_base import TubeModelProtocol

log = logging.getLogger(__name__)


def ug2_filter_matches_any(
    points: List[Dict],
    ug2_filter: Optional[float],
    ug2_tolerance: float = 5.0,
) -> bool:
    """True when the Ug2 filter keeps at least one point.

    Mirrors ``_apply_ug2_filter``'s predicate exactly (equivalence is
    pinned in tests). ``False`` means the filter is about to fall back
    LOUDLY to the unfiltered set — callers with a user-warning channel
    (the optimizer) surface it as ``ug2_filter_no_match``.
    """
    if ug2_filter is None:
        return True
    return any(abs(p.get("ug2", 0.0) - ug2_filter) <= ug2_tolerance
               for p in points)


def _apply_ug2_filter(
    points: List[Dict],
    ug2_filter: Optional[float],
    ug2_tolerance: float,
    caller: str,
) -> List[Dict]:
    """Filter points by screen voltage, falling back LOUDLY when empty.

    ML-095: an empty filter result used to fall back to the full set with
    log.debug (or silently) — for a multi-Ug2 pentode scan that mixes
    screen levels and makes intersections / composite / Q-point / THD
    physically meaningless with no signal at the default INFO level.
    """
    if ug2_filter is None:
        return points
    f = [p for p in points
         if abs(p.get("ug2", 0.0) - ug2_filter) <= ug2_tolerance]
    if f:
        return f
    log.warning(
        "%s: Ug2 filter %.1f V removed all points — falling back to the "
        "UNFILTERED set (Ug2 levels will mix; results are suspect)",
        caller, ug2_filter)
    return points


# ─── DC Q-point helpers (data-based and model-based) ──────────────────

def _find_dc_q_point(
    points: List[Dict],
    ub: float,
    ra_dc: float,
    ug1_bias: float,
    ug2_filter: Optional[float] = None,
    ug2_tolerance: float = 5.0,
) -> Optional[Tuple[float, float]]:
    """Find DC Q-point for transformer-coupled stage.

    The DC load line has very low slope (Ra_dc is winding resistance).
    The Q-point is where the tube curve at ug1_bias intersects this line.
    For very low Ra_dc, Ua_q ≈ Ub.

    Returns (ua_q, ia_q) or None.
    """
    filtered = _apply_ug2_filter(points, ug2_filter, ug2_tolerance,
                                 "_find_dc_q_point")

    ug1_levels = sorted({round(p["ug1"], UA_ROUND) for p in filtered})
    if not ug1_levels:
        return None
    ug1_near = min(ug1_levels, key=lambda u: abs(u - ug1_bias))
    bias_pts = sorted(
        [p for p in filtered if abs(round(p["ug1"], UA_ROUND) - ug1_near) < BIAS_MATCH_TOLERANCE_V],
        key=lambda p: p["ua"],
    )
    if len(bias_pts) < 2:
        return None

    if ra_dc < RA_DC_NEAR_ZERO_KOHM:
        for i in range(len(bias_pts) - 1):
            ua0, ia0 = bias_pts[i]["ua"], bias_pts[i]["ia"]
            ua1, ia1 = bias_pts[i + 1]["ua"], bias_pts[i + 1]["ia"]
            if ua0 <= ub <= ua1:
                t = (ub - ua0) / (ua1 - ua0) if ua1 != ua0 else 0.5
                ia_q = ia0 + t * (ia1 - ia0)
                return (ub, max(ia_q, 0.0))
        if ub <= bias_pts[0]["ua"]:
            return (ub, max(bias_pts[0]["ia"], 0.0))
        return (ub, max(bias_pts[-1]["ia"], 0.0))

    for i in range(len(bias_pts) - 1):
        ua0, ia0 = bias_pts[i]["ua"], bias_pts[i]["ia"]
        ua1, ia1 = bias_pts[i + 1]["ua"], bias_pts[i + 1]["ia"]
        ll0 = (ub - ua0) / ra_dc
        ll1 = (ub - ua1) / ra_dc
        d0 = ia0 - ll0
        d1 = ia1 - ll1
        if d0 * d1 <= 0 and (d0 != 0 or d1 != 0):
            t = d0 / (d0 - d1) if d0 != d1 else 0.5
            ua_q = ua0 + t * (ua1 - ua0)
            ia_q = ia0 + t * (ia1 - ia0)
            return (ua_q, max(ia_q, 0.0))

    return None


def _find_model_dc_q_point(
    model: "TubeModelProtocol",
    ub: float,
    ra_dc: float,
    ug1_bias: float,
    ug2: float,
    ua_range: Tuple[float, float],
    n_search: int = 200,
) -> Optional[Tuple[float, float]]:
    """Find DC Q-point using tube model (not measurement data).

    For low Ra_dc, Ua_q ≈ Ub; for higher Ra_dc, bisects to find crossing.
    """
    if ra_dc < RA_DC_NEAR_ZERO_KOHM:
        ua_q = min(ub, ua_range[1])
        ia_q = model.ia(ua_q, ug1_bias, ug2)
        return (ua_q, max(ia_q, 0.0))

    ua_arr = np.linspace(ua_range[0], min(ub, ua_range[1]), n_search)
    for i in range(len(ua_arr) - 1):
        ua0, ua1 = float(ua_arr[i]), float(ua_arr[i + 1])
        d0 = model.ia(ua0, ug1_bias, ug2) - (ub - ua0) / ra_dc
        d1 = model.ia(ua1, ug1_bias, ug2) - (ub - ua1) / ra_dc
        if d0 * d1 <= 0 and (d0 != 0 or d1 != 0):
            lo, hi = ua0, ua1
            for _ in range(BISECT_MAX_ITER):
                mid = (lo + hi) / 2
                d_mid = model.ia(mid, ug1_bias, ug2) - (ub - mid) / ra_dc
                d_lo = model.ia(lo, ug1_bias, ug2) - (ub - lo) / ra_dc
                if d_lo * d_mid <= 0:
                    hi = mid
                else:
                    lo = mid
                if hi - lo < BISECT_CONVERGENCE_V:
                    break
            ua_q = (lo + hi) / 2
            ia_q = model.ia(ua_q, ug1_bias, ug2)
            return (ua_q, max(ia_q, 0.0))

    return None


# ─── Intersection helpers ─────────────────────────────────────────────

# Transformer/PP AC-line reach: the zero-current end of the AC line lies at
# Ua_q + Iq·Ra_ac. The model intersection scan extends its upper Ua bound to
# that point × this margin so the last (deep-bias, low-current) curves of
# the family are not truncated by a fixed caller-side ceiling.
_AC_REACH_MARGIN = 1.05

def group_curves_by_ug1(
    points: List[Dict],
    ug2_filter: Optional[float] = None,
    ug2_tolerance: float = 5.0,
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """Pre-group measurement points into per-Ug1 sorted (ua, ia) curves.

    Mirrors :func:`find_intersections` grouping exactly (same Ug2 filter
    incl. the removed-all fallback, same 0.1 V Ug1 rounding, same per-curve
    Ua sort, <2-point groups dropped). The result depends only on
    (points, ug2_filter) — the optimizer grid re-derived it for every
    (ub, ra) combination; pass it back via ``find_intersections(...,
    curves=...)`` to skip the per-call regroup.
    """
    filtered = _apply_ug2_filter(points, ug2_filter, ug2_tolerance,
                                 "group_curves_by_ug1")

    ug1_map: Dict[float, List[Dict]] = {}
    for p in filtered:
        ug1 = round(p.get("ug1", 0.0), UA_ROUND)
        ug1_map.setdefault(ug1, []).append(p)

    curves: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for ug1, pts in ug1_map.items():
        if len(pts) < 2:
            continue
        pts_sorted = sorted(pts, key=lambda p: p["ua"])
        ua_arr = np.array([p["ua"] for p in pts_sorted], dtype=float)
        ia_arr = np.array([p["ia"] for p in pts_sorted], dtype=float)
        curves.append((ug1, ua_arr, ia_arr))
    return curves


def _intersections_from_curves(
    curves: List[Tuple[float, np.ndarray, np.ndarray]],
    ll_fn: Callable,
) -> List[Dict]:
    """Sign-change scan over pre-grouped curves (same math as the
    per-point loop in :func:`find_intersections`, vectorized per curve)."""
    intersections: List[Dict] = []
    for ug1, ua_arr, ia_arr in curves:
        ll_vals = np.broadcast_to(
            np.asarray(ll_fn(ua_arr), dtype=float), ua_arr.shape)
        d = ia_arr - ll_vals
        prod = d[:-1] * d[1:]
        change = (prod <= 0) & ((d[:-1] != 0.0) | (d[1:] != 0.0))
        if not change.any():
            continue
        i = int(np.argmax(change))
        d0, d1 = float(d[i]), float(d[i + 1])
        t = 0.5 if d0 == d1 else d0 / (d0 - d1)
        ua0, ua1 = float(ua_arr[i]), float(ua_arr[i + 1])
        ia0, ia1 = float(ia_arr[i]), float(ia_arr[i + 1])
        intersections.append({
            "ug1": ug1,
            "ua": ua0 + t * (ua1 - ua0),
            "ia": ia0 + t * (ia1 - ia0),
        })
    intersections.sort(key=lambda p: p["ug1"])
    return intersections


def find_intersections(
    points: List[Dict],
    load_line: LoadLine,
    ug2_filter: Optional[float] = None,
    ug2_tolerance: float = 5.0,
    ug1_bias: Optional[float] = None,
    curves: Optional[List[Tuple[float, np.ndarray, np.ndarray]]] = None,
) -> List[Dict]:
    """Find intersections of load_line with Ia(Ua) curves.

    Groups points by Ug1 (rounded to 0.1V), sorts each group by Ua,
    finds sign-change of (Ia_measured - Ia_loadline), interpolates.

    For TransformerLoadLine / PushPullLoadLine with ``ug1_bias`` provided,
    uses two-phase algorithm:
    1. Find DC Q-point (Ua ≈ Ub for low winding resistance).
    2. Build AC load line through Q-point with slope -1/Ra_ac.
    3. Find intersections with the AC line (Ua can exceed Ub).

    ``curves`` accepts a pre-built :func:`group_curves_by_ug1` result
    (matching the same points/ug2_filter) so hot callers skip the
    per-call filter+group+sort. Only used on the plain load-line path —
    the ``ug1_bias`` transformer branch needs the raw points.
    """
    if curves is not None and ug1_bias is None:
        return _intersections_from_curves(curves, load_line.ia_at_ua)

    if not points:
        return []

    filtered = _apply_ug2_filter(points, ug2_filter, ug2_tolerance,
                                 "find_intersections")

    ac_line_fn: Optional[Callable[[float], float]] = None
    if ug1_bias is not None:
        if isinstance(load_line, TransformerLoadLine):
            q = _find_dc_q_point(
                filtered, load_line.ub, load_line.ra_dc, ug1_bias,
                ug2_filter, ug2_tolerance,
            )
            if q is not None:
                q_ua, q_ia = q
                ra_ac = load_line.ra_ac

                def ac_line_fn(ua: float, _q_ua=q_ua, _q_ia=q_ia, _ra=ra_ac) -> float:
                    if _ra <= 0:
                        return _q_ia
                    return _q_ia - (ua - _q_ua) / _ra
        elif isinstance(load_line, PushPullLoadLine):
            q = _find_dc_q_point(
                filtered, load_line.ub, load_line.ra_dc, ug1_bias,
                ug2_filter, ug2_tolerance,
            )
            if q is not None:
                q_ua, q_ia = q
                ra_aa = load_line.ra_aa

                # Kinked per-tube trajectory (Z2 until partner
                # cutoff, Z4 beyond) instead of a straight -1/Z4 line
                # through Q — the straight line lied in the middle
                # (see pp_working_line_ia).
                def ac_line_fn(ua: float, _q_ua=q_ua, _q_ia=q_ia,
                               _ra_aa=ra_aa) -> float:
                    return float(pp_working_line_ia(ua, _q_ua, _q_ia,
                                                    _ra_aa))

    ll_fn = ac_line_fn if ac_line_fn is not None else load_line.ia_at_ua

    ug1_map: Dict[float, List[Dict]] = {}
    for p in filtered:
        ug1 = round(p.get("ug1", 0.0), UA_ROUND)
        ug1_map.setdefault(ug1, []).append(p)

    intersections: List[Dict] = []
    for ug1, pts in ug1_map.items():
        pts_sorted = sorted(pts, key=lambda p: p["ua"])
        if len(pts_sorted) < 2:
            continue
        for i in range(len(pts_sorted) - 1):
            ua0, ia0 = pts_sorted[i]["ua"], pts_sorted[i]["ia"]
            ua1, ia1 = pts_sorted[i + 1]["ua"], pts_sorted[i + 1]["ia"]
            ll0 = ll_fn(ua0)
            ll1 = ll_fn(ua1)
            d0 = ia0 - ll0
            d1 = ia1 - ll1
            if d0 * d1 <= 0 and (d0 != 0 or d1 != 0):
                if d0 == d1:
                    t = 0.5
                else:
                    t = d0 / (d0 - d1)
                ua_x = ua0 + t * (ua1 - ua0)
                ia_x = ia0 + t * (ia1 - ia0)
                intersections.append({"ug1": ug1, "ua": ua_x, "ia": ia_x})
                break
    intersections.sort(key=lambda p: p["ug1"])
    return intersections


def find_intersections_model(
    model: "TubeModelProtocol",
    load_line: LoadLine,
    ug1_values: List[float],
    ug2: float = 0.0,
    ua_range: Tuple[float, float] = (MODEL_UA_MIN_V, MODEL_UA_MAX_DEFAULT_V),
    n_search: int = MODEL_SEARCH_POINTS,
    ug1_bias: Optional[float] = None,
) -> List[Dict]:
    """Find load-line intersections using model.ia() directly.

    Uses dense evaluation + bisection for each Ug1 value.
    Much more accurate than point-based interpolation.
    """
    ll_fn = load_line.ia_at_ua
    ua_hi = ua_range[1]
    if ug1_bias is not None:
        if isinstance(load_line, TransformerLoadLine):
            ra_dc = load_line.ra_dc
            ra_ac = load_line.ra_ac
            ub = load_line.ub
            q = _find_model_dc_q_point(model, ub, ra_dc, ug1_bias, ug2, ua_range)
            if q is not None:
                q_ua, q_ia = q
                # The AC line reaches zero current at Ua = Ua_q + Iq·Ra_ac —
                # beyond any fixed caller heuristic (2×Ub) for hot Iq × big
                # Ra. Extend the scan so deep-bias intersections are not
                # silently truncated.
                if ra_ac > 0:
                    ua_hi = max(ua_hi,
                                (q_ua + q_ia * ra_ac) * _AC_REACH_MARGIN)

                def ll_fn(ua: float, _q_ua=q_ua, _q_ia=q_ia, _ra=ra_ac) -> float:
                    if _ra <= 0:
                        return _q_ia
                    return _q_ia - (ua - _q_ua) / _ra
        elif isinstance(load_line, PushPullLoadLine):
            ra_dc = load_line.ra_dc
            ra_aa = load_line.ra_aa
            ub = load_line.ub
            q = _find_model_dc_q_point(model, ub, ra_dc, ug1_bias, ug2, ua_range)
            if q is not None:
                q_ua, q_ia = q
                if ra_aa > 0:
                    # The kinked-trajectory cutoff lies on the
                    # class-A slope: Ua_q + Iq*Z2 (farther than the
                    # straight Z4 line) — reach extends along Z2,
                    # otherwise deep-bias intersections are cut silently.
                    ua_hi = max(ua_hi,
                                (q_ua + q_ia * load_line.ra_class_a)
                                * _AC_REACH_MARGIN)

                # Twin of the data branch — kinked trajectory;
                # array-safe (vectorized call in ll_row below).
                def ll_fn(ua, _q_ua=q_ua, _q_ia=q_ia, _ra_aa=ra_aa):
                    return pp_working_line_ia(ua, _q_ua, _q_ia, _ra_aa)

    ua_arr = np.linspace(ua_range[0], ua_hi, n_search)
    ug1_col = np.asarray(list(ug1_values), dtype=float)
    if ug1_col.size == 0:
        return []

    # Load-line values: all variants (linear classes + AC closures) are
    # pure arithmetic in ua, so one array call replaces the scalar loop.
    ll_row = np.broadcast_to(
        np.asarray(ll_fn(ua_arr), dtype=float), ua_arr.shape)

    # Dense scan for the whole Ug1 family in ONE vectorized model call —
    # the scalar per-point ia() loop here used to dominate the optimizer
    # grid (~30 ms/call → ~1 ms).
    ia_model = model_ia_array(model, ua_arr[None, :], ug1_col[:, None], ug2)
    diff = ia_model - ll_row[None, :]

    # First sign change per curve: diff[i]*diff[i+1] <= 0 with not-both-zero.
    prod = diff[:, :-1] * diff[:, 1:]
    change = (prod <= 0) & ((diff[:, :-1] != 0.0) | (diff[:, 1:] != 0.0))
    has_x = change.any(axis=1)
    if not has_x.any():
        return []
    first_i = np.argmax(change, axis=1)

    rows = np.nonzero(has_x)[0]
    idx = first_i[rows]
    lo = ua_arr[idx].astype(float)
    hi = ua_arr[idx + 1].astype(float)
    ug1_act = ug1_col[rows]
    # Carry f(lo) from the dense scan instead of re-evaluating it every
    # bisection iteration (halves the model calls; value is identical).
    d_lo = diff[rows, idx]

    active = np.ones(rows.shape[0], dtype=bool)
    for _ in range(BISECT_MAX_ITER):
        if not active.any():
            break
        mid = (lo[active] + hi[active]) / 2.0
        ia_mid = model_ia_array(model, mid, ug1_act[active], ug2)
        ll_mid = np.broadcast_to(
            np.asarray(ll_fn(mid), dtype=float), mid.shape)
        d_mid = ia_mid - ll_mid
        keep_left = d_lo[active] * d_mid <= 0
        hi_a = hi[active]
        lo_a = lo[active]
        hi[active] = np.where(keep_left, mid, hi_a)
        new_lo = np.where(keep_left, lo_a, mid)
        lo[active] = new_lo
        d_lo_a = d_lo[active]
        d_lo[active] = np.where(keep_left, d_lo_a, d_mid)
        conv = (hi[active] - lo[active]) < BISECT_TOLERANCE_V
        act_idx = np.nonzero(active)[0]
        active[act_idx[conv]] = False

    ua_x = (lo + hi) / 2.0
    ia_x = model_ia_array(model, ua_x, ug1_act, ug2)

    intersections: List[Dict] = [
        {"ug1": float(ug1_act[k]), "ua": float(ua_x[k]), "ia": float(ia_x[k])}
        for k in range(rows.shape[0])
    ]
    intersections.sort(key=lambda p: p["ug1"])
    return intersections


def interp_intersection(
    pts: List[Dict], target_ug1: float,
) -> Optional[Dict]:
    """Interpolate/extrapolate Ia and Ua at target_ug1 from intersection list.

    Linear interpolation when inside data range, linear extrapolation
    from two nearest edge points when outside.
    """
    if not pts:
        return None
    if len(pts) == 1:
        return dict(pts[0])
    for p in pts:
        if abs(p["ug1"] - target_ug1) < UG1_MATCH_TOLERANCE_V:
            return dict(p)
    sp = sorted(pts, key=lambda p: p["ug1"])
    below = None
    above = None
    for p in sp:
        if p["ug1"] <= target_ug1:
            below = p
        if p["ug1"] >= target_ug1 and above is None:
            above = p
    if below is not None and above is not None and below is not above:
        p0, p1 = below, above
    elif below is None:
        p0, p1 = sp[0], sp[1]
    else:
        p0, p1 = sp[-2], sp[-1]
    span = p1["ug1"] - p0["ug1"]
    if span == 0:
        return dict(p0)
    t = (target_ug1 - p0["ug1"]) / span
    return {
        "ug1": target_ug1,
        "ua": p0["ua"] + t * (p1["ua"] - p0["ua"]),
        "ia": p0["ia"] + t * (p1["ia"] - p0["ia"]),
    }


# ─── 5-point distortion (SE) ──────────────────────────────────────────

def diagnose_distortion(
    intersections: List[Dict],
    ug1_bias: Optional[float] = None,
    half_swing: Optional[float] = None,
) -> str:
    """Diagnose why compute_distortion would return None.

    Returns a DIST_ERR_* code based on upfront input checks. Returns
    DIST_ERR_UNKNOWN if all input-level checks pass — the failure must
    be in inner numerical conditions (i_max≈i_min, b1<MIN_B1_MA, manual
    swing pushed into Ia<0). Returns "" (empty) if compute_distortion
    should succeed.
    """
    if len(intersections) < 3:
        return DIST_ERR_FEW_INTERSECTIONS

    pts = sorted(intersections, key=lambda p: p["ug1"])
    ug1_min = pts[0]["ug1"]
    ug1_max = pts[-1]["ug1"]

    target = ug1_bias if ug1_bias is not None else (ug1_min + ug1_max) / 2.0

    range_span = max(ug1_max - ug1_min, 1e-3)
    if target < ug1_min - 0.5 * range_span or target > ug1_max + 0.5 * range_span:
        return DIST_ERR_BIAS_OUTSIDE

    # ML-051: a tiny manual swing must be classified BEFORE the
    # manual/auto split — `manual_swing` is False for it, so the old
    # in-else check was unreachable and diagnose returned 'unknown'.
    if half_swing is not None and 0 < half_swing < MIN_SWING_V:
        return DIST_ERR_MANUAL_SWING_SMALL
    manual_swing = half_swing is not None and half_swing > MIN_SWING_V
    if not manual_swing:
        target_clamped = max(ug1_min, min(ug1_max, target))
        auto_swing = min(target_clamped - ug1_min, ug1_max - target_clamped)
        if auto_swing < MIN_SWING_V:
            return DIST_ERR_BIAS_AT_EDGE
        effective_half_swing = auto_swing
    else:
        target_clamped = max(ug1_min, min(ug1_max, target))
        max_swing = min(target_clamped - ug1_min, ug1_max - target_clamped)
        if max_swing < MIN_SWING_V:
            return DIST_ERR_BIAS_AT_EDGE
        effective_half_swing = min(half_swing, max_swing) if half_swing else max_swing

    ug1_neg = target_clamped - effective_half_swing
    ug1_pos = target_clamped + effective_half_swing
    edge_tol = max((ug1_pos - ug1_neg) * EDGE_TOL_REL, EDGE_TOL_ABS)
    ug1_in_swing = [
        p["ug1"] for p in pts
        if ug1_neg - edge_tol <= p["ug1"] <= ug1_pos + edge_tol
    ]
    if len(ug1_in_swing) < MIN_CURVES_IN_SWING:
        return DIST_ERR_SPARSE_DATA
    n_strictly_inside = sum(
        1 for u in ug1_in_swing
        if ug1_neg + edge_tol < u < ug1_pos - edge_tol
    )
    if n_strictly_inside < 1:
        return DIST_ERR_SPARSE_DATA

    # A flat Ia window (crossings at nearly equal currents) kills the
    # fundamental inside compute_distortion — name it instead of the
    # catch-all "unknown" (reported symptom: "the line DOES cross
    # the curves at the right points", yet analysis failed).
    probe = window_b1_probe(intersections, ug1_bias=ug1_bias,
                            half_swing=half_swing)
    if probe is not None and probe["b1"] <= MIN_B1_MA:
        return DIST_ERR_NO_SIGNAL

    return DIST_ERR_UNKNOWN


def window_b1_probe(
    intersections: List[Dict],
    ug1_bias: Optional[float] = None,
    half_swing: Optional[float] = None,
) -> Optional[Dict]:
    """Ia window and 5-point fundamental — DIAGNOSTICS ONLY.

    Mirrors ``compute_distortion``'s sampling exactly: nearest-crossing
    snapping for auto swing, interpolation for a manual one — so a
    near-zero ``b1`` here means compute failed on the same inputs. The
    engine attaches the numbers to ``dist_error_params`` so the panel
    can SHOW the flat window instead of guessing.

    Returns ``{"i_min", "i_max", "b1"}`` (mA) or None when no window can
    be built (those cases carry their own DIST_ERR codes upstream).
    """
    if len(intersections) < 3:
        return None
    pts = sorted(intersections, key=lambda p: p["ug1"])
    ug1_all_min = pts[0]["ug1"]
    ug1_all_max = pts[-1]["ug1"]
    target = (ug1_bias if ug1_bias is not None
              else (ug1_all_min + ug1_all_max) / 2.0)
    manual = half_swing is not None and half_swing > MIN_SWING_V

    if manual:
        ug1_0_t = max(ug1_all_min, min(ug1_all_max, target))
        center = interp_intersection(pts, ug1_0_t)
        if center is None:
            return None
        ug1_0 = center["ug1"]
        max_swing = min(ug1_0 - ug1_all_min, ug1_all_max - ug1_0)
        if max_swing < MIN_SWING_V:
            return None
        eff = min(half_swing, max_swing)

        def _sample(u: float) -> Optional[Dict]:
            return interp_intersection(pts, u)
    else:
        center = min(pts, key=lambda p: abs(p["ug1"] - target))
        ug1_0 = center["ug1"]
        eff = min(abs(ug1_0 - ug1_all_min), abs(ug1_all_max - ug1_0))
        if eff < MIN_SWING_V:
            return None

        def _sample(u: float) -> Optional[Dict]:
            return min(pts, key=lambda p: abs(p["ug1"] - u))

    pt_neg = _sample(ug1_0 - eff)
    pt_pos = _sample(ug1_0 + eff)
    pt_low_half = _sample(ug1_0 - eff / 2.0)
    pt_high_half = _sample(ug1_0 + eff / 2.0)
    if None in (pt_neg, pt_pos, pt_low_half, pt_high_half):
        return None
    i_min = pt_neg["ia"]
    i_max = pt_pos["ia"]
    b1 = ((i_max - i_min)
          + (pt_high_half["ia"] - pt_low_half["ia"])) / 3.0
    return {"i_min": i_min, "i_max": i_max, "b1": b1}


def compute_distortion(
    intersections: List[Dict],
    ug1_bias: Optional[float] = None,
    half_swing: Optional[float] = None,
    ub: Optional[float] = None,
) -> Optional[Dict]:
    """5-point harmonic distortion analysis.

    Selected-ordinate method from "Radiotron Designer's Handbook,
    4th ed.", see SOURCES_INDEX.md.

    Returns:
        Dict with hd2, hd3, thd, pout_mw, ug1_0, ua_0, ia_0,
        i_max, i_min, ua_max, ua_min, half_swing, interpolated,
        pt_neg, pt_low_half, pt_center, pt_high_half, pt_pos,
        amp_class ("A"/"AB"/"B"), pdc_mw, eta_pct, pa_signal_mw (latter 3 need ub).
        Or None if insufficient data.
    """
    if len(intersections) < 3:
        return None

    pts = sorted(intersections, key=lambda p: p["ug1"])
    ug1_all_min = pts[0]["ug1"]
    ug1_all_max = pts[-1]["ug1"]

    if ug1_bias is not None:
        ug1_0_target = ug1_bias
    else:
        ug1_0_target = (ug1_all_min + ug1_all_max) / 2.0

    interpolated = False
    if half_swing is not None and 0 < half_swing < MIN_SWING_V:
        # ML-051: a tiny MANUAL swing used to silently fall through to the
        # full AUTO swing; returning None lets diagnose_distortion report
        # the (previously unreachable) DIST_ERR_MANUAL_SWING_SMALL.
        # Exactly 0.0 stays the legacy "auto" convention (unset spin).
        return None
    manual_swing_requested = half_swing is not None and half_swing > MIN_SWING_V
    requested_half_swing = float(half_swing) if manual_swing_requested else None
    manual_swing_clamped = False

    if manual_swing_requested:
        # Keep manual bias/swing strictly inside measured Ug1 range.
        ug1_0_target = max(ug1_all_min, min(ug1_all_max, ug1_0_target))
        center = interp_intersection(pts, ug1_0_target)
        if center is None:
            return None
        i_0 = center["ia"]
        ug1_0 = center["ug1"]
        ua_0 = center["ua"]

        max_swing = min(ug1_0 - ug1_all_min, ug1_all_max - ug1_0)
        if max_swing < MIN_SWING_V:
            return None
        if half_swing is None:
            return None
        effective_swing = min(half_swing, max_swing)
        if abs(effective_swing - half_swing) > 1e-9:
            manual_swing_clamped = True
        half_swing = effective_swing

        ug1_neg = ug1_0 - effective_swing
        ug1_pos = ug1_0 + effective_swing
        ug1_low_half = (ug1_neg + ug1_0) / 2.0
        ug1_high_half = (ug1_0 + ug1_pos) / 2.0

        pt_neg = interp_intersection(pts, ug1_neg)
        pt_pos = interp_intersection(pts, ug1_pos)
        pt_low_half = interp_intersection(pts, ug1_low_half)
        pt_high_half = interp_intersection(pts, ug1_high_half)
        interpolated = True
    else:
        center = min(pts, key=lambda p: abs(p["ug1"] - ug1_0_target))
        i_0 = center["ia"]
        ug1_0 = center["ug1"]
        ua_0 = center["ua"]

        dist_neg = abs(ug1_0 - ug1_all_min)
        dist_pos = abs(ug1_all_max - ug1_0)
        half_swing = min(dist_neg, dist_pos)
        if half_swing < MIN_SWING_V:
            return None

        ug1_neg = ug1_0 - half_swing
        ug1_pos = ug1_0 + half_swing

        pt_neg = min(pts, key=lambda p: abs(p["ug1"] - ug1_neg))
        pt_pos = min(pts, key=lambda p: abs(p["ug1"] - ug1_pos))
        ug1_low_half = (ug1_neg + ug1_0) / 2.0
        ug1_high_half = (ug1_0 + ug1_pos) / 2.0
        pt_low_half = min(pts, key=lambda p: abs(p["ug1"] - ug1_low_half))
        pt_high_half = min(pts, key=lambda p: abs(p["ug1"] - ug1_high_half))

    if pt_neg is None or pt_pos is None or pt_low_half is None or pt_high_half is None:
        return None

    # Sparse-data degeneracy guard.
    edge_tol = max((ug1_pos - ug1_neg) * EDGE_TOL_REL, EDGE_TOL_ABS)
    ug1_in_swing = [
        p["ug1"] for p in pts
        if ug1_neg - edge_tol <= p["ug1"] <= ug1_pos + edge_tol
    ]
    if len(ug1_in_swing) < MIN_CURVES_IN_SWING:
        return None
    n_strictly_inside = sum(
        1 for u in ug1_in_swing
        if ug1_neg + edge_tol < u < ug1_pos - edge_tol
    )
    if n_strictly_inside < 1:
        return None

    i_min = pt_neg["ia"]
    i_max = pt_pos["ia"]
    i_low_half = pt_low_half["ia"]
    i_high_half = pt_high_half["ia"]

    # 5-point Fourier coefficients (Radiotron Designer's Handbook method).
    swing = i_max - i_min
    half_diff = i_high_half - i_low_half
    b1 = (swing + half_diff) / 3.0
    if b1 <= MIN_B1_MA:
        return None
    b2 = (i_max + i_min - 2.0 * i_0) / 4.0
    b3 = (swing - 2.0 * half_diff) / 6.0

    hd2 = abs(b2 / b1) * 100.0
    hd3 = abs(b3 / b1) * 100.0
    thd = math.sqrt(hd2 ** 2 + hd3 ** 2)

    ua_swing = abs(pt_neg["ua"] - pt_pos["ua"])
    pout_mw = (swing * ua_swing) / 8.0

    if manual_swing_clamped and (i_0 < -0.01 or i_min < -0.01 or i_max < -0.01):
        return None

    insufficient_signal = (
        abs(i_max - i_min) < MIN_IA_SWING_MA
        or ua_swing < MIN_UA_SWING_V
        or pout_mw < MIN_POUT_MW
    )

    pdc_mw: Optional[float] = None
    eta_pct: Optional[float] = None
    pa_signal_mw: Optional[float] = None
    if ub is not None and ub > 0 and i_0 > 0:
        pdc_mw = ub * i_0
        eta_pct = pout_mw / pdc_mw * 100.0
        pa_signal_mw = pdc_mw - pout_mw

    if i_0 > 0:
        ratio = i_min / i_0
        amp_class = "A" if ratio > CLASS_A_RATIO else ("B" if ratio < CLASS_B_RATIO else "AB")
    else:
        amp_class = "B"

    return {
        "hd2": hd2, "hd3": hd3, "thd": thd, "pout_mw": pout_mw,
        "ug1_0": ug1_0, "ua_0": ua_0, "ia_0": i_0,
        "i_max": i_max, "i_min": i_min,
        "ua_max": pt_neg["ua"], "ua_min": pt_pos["ua"],
        "half_swing": half_swing,
        "interpolated": interpolated,
        "requested_half_swing": requested_half_swing,
        "manual_swing_clamped": manual_swing_clamped,
        "insufficient_signal": insufficient_signal,
        "pt_neg": pt_neg, "pt_low_half": pt_low_half,
        "pt_center": {"ug1": ug1_0, "ua": ua_0, "ia": i_0},
        "pt_high_half": pt_high_half, "pt_pos": pt_pos,
        "amp_class": amp_class,
        "pdc_mw": pdc_mw, "eta_pct": eta_pct, "pa_signal_mw": pa_signal_mw,
    }


# ─── Shared DFT Newton solver ────────────────────────────────────────

# Pre-existing inline values from the scalar Newton loops, named here when
# the loops were vectorized: probe step for the numeric derivative, the
# derivative magnitude below which a fixed kick is used instead of a
# Newton step, and the per-path iteration budgets.
_NEWTON_PROBE_STEP_V = 0.5
_NEWTON_DERIV_MIN = 1e-6
_NEWTON_KICK_STEP_V = 1.0
_DFT_NEWTON_MAX_ITER_SE = 30
_DFT_NEWTON_MAX_ITER_PP = 20


def _newton_solve_vec(
    ia_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ll_fn: Callable[[np.ndarray], np.ndarray],
    ug1_arr: np.ndarray,
    ua_init,
    max_iter: int,
) -> Tuple[np.ndarray, int, float]:
    """Solve Ia_model(ua, ug1) = Ia_ll(ua) for every sample at once.

    Element-wise mirror of the former per-sample scalar Newton loop: every
    sample cold-starts at ``ua_init``, follows the same update sequence
    (numeric derivative over ``_NEWTON_PROBE_STEP_V``, ±kick when the
    derivative vanishes, clamp to ua ≥ 0) and freezes once
    ``|err| < FIXED_POINT_CONVERGENCE_MA`` — so per-sample results match
    the scalar path bit-for-bit while the model is evaluated in two array
    calls per iteration instead of two Python calls per sample.

    Returns ``(ua, n_not_converged, max_residual_mA)``.
    """
    n = ug1_arr.shape[0]
    ua = np.full(n, float(ua_init)) if np.isscalar(ua_init) \
        else np.asarray(ua_init, dtype=float).copy()
    active = np.ones(n, dtype=bool)
    last_err = np.zeros(n, dtype=float)

    for _ in range(max_iter):
        if not active.any():
            break
        act_idx = np.nonzero(active)[0]
        ua_a = ua[act_idx]
        g_a = ug1_arr[act_idx]
        err = ia_fn(ua_a, g_a) - np.broadcast_to(
            np.asarray(ll_fn(ua_a), dtype=float), ua_a.shape)
        last_err[act_idx] = np.abs(err)
        conv = np.abs(err) < FIXED_POINT_CONVERGENCE_MA
        if conv.any():
            active[act_idx[conv]] = False
            still = ~conv
            if not still.any():
                continue
            act_idx = act_idx[still]
            ua_a = ua_a[still]
            g_a = g_a[still]
            err = err[still]
        ua_probe = ua_a + _NEWTON_PROBE_STEP_V
        err2 = ia_fn(ua_probe, g_a) - np.broadcast_to(
            np.asarray(ll_fn(ua_probe), dtype=float), ua_probe.shape)
        deriv = (err2 - err) / _NEWTON_PROBE_STEP_V
        use_newton = np.abs(deriv) > _NEWTON_DERIV_MIN
        deriv_safe = np.where(use_newton, deriv, 1.0)
        kick = np.where(err > 0, _NEWTON_KICK_STEP_V, -_NEWTON_KICK_STEP_V)
        new_ua = np.where(use_newton, ua_a - err / deriv_safe, ua_a + kick)
        ua[act_idx] = np.maximum(0.0, new_ua)

    n_diverged = int(active.sum())
    max_resid = float(last_err[active].max()) if n_diverged else 0.0
    return ua, n_diverged, max_resid


def _pp_joint_solve_vec(
    ia_a_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ia_b_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ug1_a: np.ndarray,
    ug1_b: np.ndarray,
    ua_q_a: float,
    ua_q_b: float,
    ra_per_tube: float,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Coupled push-pull solve for an ideal center-tapped OPT.

    One unknown per sample — the half-primary AC voltage ``v`` (both
    anodes deviate from their Q-points by ±v, enforced by transformer
    flux):

        ua_a = max(0, Ua_qa − v),   ua_b = max(0, Ua_qb + v)
        residual(v) = Ia_a(ua_a) − Ia_b(ua_b) − v / Ra_per_tube   [mA]

    This replaces the former two INDEPENDENT per-tube Newtons against a
    fixed −1/(Ra_aa/4) line. The class-AB kink emerges from the circuit
    equation itself: while both tubes conduct each one sees Ra_aa/2
    (∂v/∂Ia_a = 2·Ra_per_tube — the partner's antiphase current doubles
    the flux); once the partner cuts off (Ia_b ≡ 0) the active tube sees
    Ra_aa/4. No per-region heuristics.

    d(residual)/dv = −(gm_a + gm_b + 1/Ra) < 0 strictly → the root is
    UNIQUE and always bracketed by [−Ua_qb, +Ua_qa] (at the ends one
    anode hits 0 V and the sign of the residual is fixed). The solver is
    a safeguarded Newton: bracket maintained every iteration, Newton
    step accepted only inside the bracket, bisection otherwise — plain
    Newton oscillates on the steep pentode knee (Ia collapsing as
    ua → 0), which showed up as ~30% divergence on hot-swing points.

    Returns ``(ia_a, ia_b, ua_a, ua_b, n_not_converged, max_residual_mA)``
    with the per-tube current waveforms clamped to ≥ 0 like the former
    solver; the anode-voltage waveforms expose the per-tube load-line
    behaviour (class-A Zaa/2 vs class-B Zaa/4) for tests/diagnostics.
    """
    n = ug1_a.shape[0]
    lo = np.full(n, -float(ua_q_b))
    hi = np.full(n, float(ua_q_a))
    v = np.zeros(n, dtype=float)          # 0 is inside the bracket
    active = np.ones(n, dtype=bool)
    last_err = np.zeros(n, dtype=float)

    def _resid(v_act: np.ndarray, idx: np.ndarray) -> np.ndarray:
        ua_a = np.maximum(0.0, ua_q_a - v_act)
        ua_b = np.maximum(0.0, ua_q_b + v_act)
        return (ia_a_fn(ua_a, ug1_a[idx]) - ia_b_fn(ua_b, ug1_b[idx])
                - v_act / ra_per_tube)

    for _ in range(max_iter):
        if not active.any():
            break
        act_idx = np.nonzero(active)[0]
        v_a = v[act_idx]
        err = _resid(v_a, act_idx)
        last_err[act_idx] = np.abs(err)
        conv = np.abs(err) < FIXED_POINT_CONVERGENCE_MA
        if conv.any():
            active[act_idx[conv]] = False
            still = ~conv
            if not still.any():
                continue
            act_idx = act_idx[still]
            v_a = v_a[still]
            err = err[still]
        # Monotone residual → err > 0 means the root lies above v.
        pos = err > 0
        lo[act_idx] = np.where(pos, v_a, lo[act_idx])
        hi[act_idx] = np.where(pos, hi[act_idx], v_a)
        err2 = _resid(v_a + _NEWTON_PROBE_STEP_V, act_idx)
        deriv = (err2 - err) / _NEWTON_PROBE_STEP_V
        use_newton = np.abs(deriv) > _NEWTON_DERIV_MIN
        deriv_safe = np.where(use_newton, deriv, 1.0)
        v_newton = v_a - err / deriv_safe
        inside = (use_newton & (v_newton > lo[act_idx])
                  & (v_newton < hi[act_idx]))
        v[act_idx] = np.where(
            inside, v_newton, (lo[act_idx] + hi[act_idx]) / 2.0)

    ua_a = np.maximum(0.0, ua_q_a - v)
    ua_b = np.maximum(0.0, ua_q_b + v)
    ia_a = np.maximum(0.0, ia_a_fn(ua_a, ug1_a))
    ia_b = np.maximum(0.0, ia_b_fn(ua_b, ug1_b))
    n_diverged = int(active.sum())
    max_resid = float(last_err[active].max()) if n_diverged else 0.0
    return ia_a, ia_b, ua_a, ua_b, n_diverged, max_resid


# ─── Chebyshev / DFT distortion (SE) ─────────────────────────────────

def _chebyshev_decompose(
    u_fit, ia_fit, max_harmonic: int,
) -> Optional[Tuple[object, float, Dict[str, float], float]]:
    """Shared Chebyshev fit + harmonics extraction (used by SE and PP paths).

    Runs ``numpy.polynomial.chebyshev.chebfit`` and pulls out the
    fundamental + HD2..HDn coefficients.

    Returns ``(coeffs, c1, harmonics, thd)`` on success — caller uses
    ``coeffs`` for further chebval() evaluations and ``c1`` to detect
    near-zero fundamental. Returns ``None`` when:
      - ``chebfit`` raises ``LinAlgError`` / ``ValueError`` (degenerate fit)
      - ``|c1| < 1e-6`` (fundamental amplitude too small to derive HDs)

    ``harmonics`` is a dict ``{"hd2": pct, "hd3": pct, ...}`` with
    ``HDn = |c_n / c_1| * 100%``. ``thd`` is ``sqrt(sum(HDi^2))``.
    """
    from numpy.polynomial import chebyshev

    try:
        coeffs = chebyshev.chebfit(u_fit, ia_fit, deg=max_harmonic)
    except (np.linalg.LinAlgError, ValueError):
        return None

    c1 = coeffs[1] if len(coeffs) > 1 else 0.0
    if abs(c1) < 1e-6:
        return None

    harmonics: Dict[str, float] = {}
    for n in range(2, max_harmonic + 1):
        cn = coeffs[n] if n < len(coeffs) else 0.0
        harmonics[f"hd{n}"] = abs(cn / c1) * 100.0

    thd = math.sqrt(sum(h ** 2 for h in harmonics.values()))
    return coeffs, float(c1), harmonics, thd


def compute_distortion_chebyshev(
    intersections: List[Dict],
    ug1_bias: Optional[float] = None,
    half_swing: Optional[float] = None,
    ub: Optional[float] = None,
    max_harmonic: int = 9,
) -> Optional[Dict]:
    """Chebyshev polynomial harmonic distortion analysis — HD2 through HDn.

    Chebyshev-harmonic identity from "Kenny Peng — Chebyshev Polynomials
    and Harmonics" and "Clark — Nord Modular book, Ch. 12",
    see SOURCES_INDEX.md.

    Decomposes Ia(Ug1) along the load line into Chebyshev series.
    The key identity Tn(cos θ) = cos(nθ) means coefficients directly
    give harmonic amplitudes for a sinusoidal input.

    Best for measured data (10-30 points): no sampling artifacts,
    fits directly on measured intersections.
    """
    from numpy.polynomial import chebyshev

    n_pts = len(intersections)
    if n_pts < MIN_CHEBYSHEV_HARMONIC + 1:
        return None
    safe_max = max(2 * n_pts // CHEBYSHEV_OVERFIT_RATIO, MIN_CHEBYSHEV_HARMONIC)
    if max_harmonic > safe_max:
        max_harmonic = safe_max
        log.debug("Chebyshev: reduced max_harmonic to %d (have %d points)", max_harmonic, n_pts)

    pts = sorted(intersections, key=lambda p: p["ug1"])
    ug1_arr = np.array([p["ug1"] for p in pts])
    ia_arr = np.array([p["ia"] for p in pts])

    ug1_min, ug1_max = ug1_arr[0], ug1_arr[-1]

    if ug1_bias is not None:
        bias = max(ug1_min, min(ug1_max, ug1_bias))
    else:
        bias = (ug1_min + ug1_max) / 2.0

    if half_swing is not None and 0 < half_swing < MIN_SWING_V:
        # ML-051 (chebyshev twin): a tiny MANUAL swing must surface as
        # DIST_ERR_MANUAL_SWING_SMALL via diagnose, not silently become
        # the full auto swing. Exactly 0.0 = legacy "auto".
        return None
    manual_swing_clamped = False
    requested_half_swing = None
    if half_swing is not None and half_swing > MIN_SWING_V:
        requested_half_swing = float(half_swing)
        A = min(half_swing, bias - ug1_min, ug1_max - bias)
        if A < half_swing - 1e-9:
            # Mutation-audit: chebyshev clamped silently while
            # 5-point set the SE-convention flag — the panel notice never
            # fired for the chebyshev method.
            manual_swing_clamped = True
    else:
        A = min(bias - ug1_min, ug1_max - bias)

    if A < MIN_SWING_V:
        return None

    u = (ug1_arr - bias) / A
    mask = (u >= -CHEBYSHEV_BOUNDARY) & (u <= CHEBYSHEV_BOUNDARY)
    u_fit = u[mask]
    ia_fit = ia_arr[mask]

    if len(u_fit) < MIN_CHEBYSHEV_HARMONIC + 1:
        return None
    safe_max_fit = max(2 * len(u_fit) // CHEBYSHEV_OVERFIT_RATIO, MIN_CHEBYSHEV_HARMONIC)
    if max_harmonic > safe_max_fit:
        max_harmonic = safe_max_fit
        log.debug("Chebyshev: reduced max_harmonic to %d after range filter (%d points)",
                  max_harmonic, len(u_fit))

    u_fit = np.clip(u_fit, -1.0, 1.0)

    decomp = _chebyshev_decompose(u_fit, ia_fit, max_harmonic)
    if decomp is None:
        return None
    coeffs, c1, harmonics, thd = decomp

    ia_0 = float(chebyshev.chebval(0.0, coeffs))
    center = interp_intersection(pts, bias)
    ua_0 = center["ua"] if center else 0.0

    ia_at_pos = float(chebyshev.chebval(1.0, coeffs))
    ia_at_neg = float(chebyshev.chebval(-1.0, coeffs))
    pt_pos = interp_intersection(pts, bias + A)
    pt_neg = interp_intersection(pts, bias - A)
    ua_at_pos = pt_pos["ua"] if pt_pos else ua_0
    ua_at_neg = pt_neg["ua"] if pt_neg else ua_0
    i_swing = ia_at_pos - ia_at_neg
    ua_swing = abs(ua_at_neg - ua_at_pos)
    pout_mw = abs(i_swing * ua_swing) / 8.0

    pdc_mw: Optional[float] = None
    eta_pct: Optional[float] = None
    if ub is not None and ub > 0 and ia_0 > 0:
        pdc_mw = ub * ia_0
        eta_pct = pout_mw / pdc_mw * 100.0

    return {
        **harmonics,
        "thd": thd,
        "b1": c1,
        "pout_mw": pout_mw,
        "ug1_0": bias, "ua_0": ua_0, "ia_0": ia_0,
        "i_max": ia_at_pos, "i_min": ia_at_neg,
        "ua_max": ua_at_neg, "ua_min": ua_at_pos,
        "half_swing": A,
        "method": HD_METHOD_CHEBYSHEV,
        "manual_swing_clamped": manual_swing_clamped,
        "requested_half_swing": requested_half_swing,
        "max_harmonic": max_harmonic,
        "pdc_mw": pdc_mw, "eta_pct": eta_pct,
    }


def compute_distortion_dft(
    model: "TubeModelProtocol",
    load_line: "LoadLine",
    ug1_bias: float,
    half_swing: float,
    ug2: float = 0.0,
    ub: Optional[float] = None,
    n_samples: int = 1024,
    max_harmonic: int = 9,
    ug1_bias_for_ll: Optional[float] = None,
) -> Optional[Dict]:
    """DFT-based harmonic distortion analysis — HD2 through HDn.

    Generates a synthetic sinusoidal drive, computes Ia(t) using the
    tube model + load line, then applies FFT to extract harmonics.
    """
    if half_swing < MIN_SWING_V:
        return None

    if ug1_bias_for_ll is None:
        ug1_bias_for_ll = ug1_bias

    ll_fn = load_line.ia_at_ua
    if isinstance(load_line, TransformerLoadLine):
        ra_dc = load_line.ra_dc
        ra_ac = load_line.ra_ac
        ub_ll = load_line.ub
        q = _find_model_dc_q_point(model, ub_ll, ra_dc, ug1_bias_for_ll, ug2,
                                    (MODEL_UA_MIN_V, MODEL_UA_MAX_DEFAULT_V))
        if q is not None:
            q_ua, q_ia = q

            def ll_fn(ua: float, _q_ua=q_ua, _q_ia=q_ia, _ra=ra_ac) -> float:
                if _ra <= 0:
                    return _q_ia
                return _q_ia - (ua - _q_ua) / _ra
    elif isinstance(load_line, PushPullLoadLine):
        ra_dc = load_line.ra_dc
        ra_pt = load_line.ra_per_tube
        ub_ll = load_line.ub
        q = _find_model_dc_q_point(model, ub_ll, ra_dc, ug1_bias_for_ll, ug2,
                                    (MODEL_UA_MIN_V, MODEL_UA_MAX_DEFAULT_V))
        if q is not None:
            q_ua, q_ia = q

            def ll_fn(ua: float, _q_ua=q_ua, _q_ia=q_ia, _ra=ra_pt) -> float:
                if _ra <= 0:
                    return _q_ia
                return _q_ia - (ua - _q_ua) / _ra

    t = np.arange(n_samples)
    ug1_t = ug1_bias + half_swing * np.cos(2.0 * np.pi * t / n_samples)

    ua_guess = load_line.ub if hasattr(load_line, 'ub') else DEFAULT_UB_V
    eps = load_line.endpoints()
    if eps:
        ua_guess = (eps[0][0] + eps[1][0]) / 2.0

    # Track Newton convergence per call. In normal operation (verified
    # across 800 realistic ub/ra/bias/swing configs) all samples converge
    # within ~5 iterations. Divergence triggers only on extreme overdrive
    # (swing pushing into positive-grid / saturation-knee where Newton
    # bounces against the ``max(0, ua)`` clamp). When that happens, the
    # ``ia_t`` value used in FFT is taken at the clamped ``ua`` instead
    # of the physical load-line intersection — biasing the resulting THD.
    # The WARNING below makes that visible. All samples are independent
    # (each cold-starts at ua_guess), so the solve runs vectorized.
    ua_t, not_converged, max_residual_ma = _newton_solve_vec(
        lambda ua_arr, g_arr: model_ia_array(model, ua_arr, g_arr, ug2),
        ll_fn, ug1_t, ua_guess, _DFT_NEWTON_MAX_ITER_SE,
    )
    ia_t = np.maximum(0.0, model_ia_array(model, ua_t, ug1_t, ug2))

    # ML-138: NaN/Inf from model.ia (invalid fit region) sails through
    # every downstream sanity check — NaN comparisons are all False — and
    # lands in the UI/optimizer as THD=nan. Fail loudly instead.
    if not (np.all(np.isfinite(ia_t)) and np.all(np.isfinite(ua_t))):
        bad = int(np.count_nonzero(~np.isfinite(ia_t))
                  + np.count_nonzero(~np.isfinite(ua_t)))
        log.warning(
            "DFT: %d non-finite model sample(s) (ug1_bias=%.2f, "
            "half_swing=%.2f) — model invalid in the evaluated region",
            bad, ug1_bias, half_swing)
        return None

    if not_converged > 0:
        log.warning(
            "DFT Newton: %d/%d samples did not converge in %d iters "
            "(max residual %.3g mA, ug1_bias=%.2f, half_swing=%.2f) — "
            "THD may be biased near clipping",
            not_converged, n_samples, _DFT_NEWTON_MAX_ITER_SE,
            max_residual_ma, ug1_bias, half_swing,
        )

    spectrum = np.fft.rfft(ia_t)
    magnitudes = np.abs(spectrum) * 2.0 / n_samples

    if len(magnitudes) < max_harmonic + 1:
        return None

    b1 = magnitudes[1]
    if b1 < 1e-6:
        return None

    harmonics = {}
    for n in range(2, max_harmonic + 1):
        harmonics[f"hd{n}"] = float(magnitudes[n] / b1) * 100.0

    thd = math.sqrt(sum(h ** 2 for h in harmonics.values()))

    ia_0 = float(np.mean(ia_t))
    ua_0 = float(np.mean(ua_t))

    i_max = float(np.max(ia_t))
    i_min = float(np.min(ia_t))
    ua_max_val = float(np.max(ua_t))
    ua_min_val = float(np.min(ua_t))
    i_swing = i_max - i_min
    ua_swing = ua_max_val - ua_min_val
    pout_mw = i_swing * ua_swing / 8.0
    # Fundamental-only output power: P1 = Ia1·Ua1/2 from the two FFTs
    # (mA·V/2 = mW). Equals pout_mw for a pure sine; under distortion the
    # peak-based estimate above includes harmonic content — external
    # references (LTspice .four, analyzers) report the fundamental.
    ua_mag1 = float(np.abs(np.fft.rfft(ua_t))[1]) * 2.0 / n_samples
    pout_fund_mw = float(b1) * ua_mag1 / 2.0

    pa_avg_mw = float(np.mean(ua_t * ia_t))

    pdc_mw: Optional[float] = None
    eta_pct: Optional[float] = None
    pa_signal_mw: Optional[float] = None
    if ub is not None and ub > 0 and ia_0 > 0:
        pdc_mw = ub * ia_0
        eta_pct = pout_mw / pdc_mw * 100.0
        pa_signal_mw = pa_avg_mw

    return {
        **harmonics,
        "thd": thd,
        "b1": float(b1),
        "pout_mw": pout_mw,
        "ug1_0": ug1_bias, "ua_0": ua_0, "ia_0": ia_0,
        "i_max": i_max, "i_min": i_min,
        "ua_max": ua_max_val, "ua_min": ua_min_val,
        "half_swing": half_swing,
        "pout_fund_mw": pout_fund_mw,
        "method": HD_METHOD_DFT,
        "n_samples": n_samples,
        "max_harmonic": max_harmonic,
        "pdc_mw": pdc_mw, "eta_pct": eta_pct,
        "pa_signal_mw": pa_signal_mw,
        "pa_avg_mw": pa_avg_mw,
        # Newton solve health — THD is biased near clipping when samples
        # fail to converge; the engine surfaces this in the UI warnings.
        "n_not_converged": int(not_converged),
    }


# ─── Push-pull composite + distortion variants ───────────────────────

def _build_transfer_curve(
    points: List[Dict],
    ug2_filter: Optional[float] = None,
    ug2_tolerance: float = 5.0,
    ua_ref: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the Ia(Ug1) transfer curve.

    ``ua_ref`` given (the PP supply voltage — the per-tube DC operating
    plate voltage): Ia is INTERPOLATED at Ua=ua_ref per Ug1 curve — the
    standard transfer-characteristic quick method. ``np.interp`` clamps
    outside the measured Ua span (pentode-plateau assumption).

    ``ua_ref=None`` keeps the legacy mean over the whole Ua sweep. That
    average is acceptable for pentode plateaus but badly biased for
    triodes (probed 12AU7: Iq comes out ~0.41× of Ia(Ub, bias)), which
    skewed iq_per_tube → Pa checks / p_classA_w / amp_class on the PP
    data path. All load-line-aware callers now pass ``ua_ref=ll.ub``.
    """
    filtered = _apply_ug2_filter(points, ug2_filter, ug2_tolerance,
                                 "_build_transfer_curve")

    ug1_pts: Dict[float, List[Tuple[float, float]]] = {}
    for p in filtered:
        ug1 = round(p.get("ug1", 0.0), UA_ROUND)
        ug1_pts.setdefault(ug1, []).append((p.get("ua", 0.0), p["ia"]))

    if not ug1_pts:
        return np.array([]), np.array([])

    ug1_sorted = sorted(ug1_pts.keys())
    ug1_arr = np.array(ug1_sorted)
    ia_vals: List[float] = []
    for u in ug1_sorted:
        pts = ug1_pts[u]
        if ua_ref is None or len(pts) < 2:
            # legacy mean; single-point curves have nothing to interpolate
            ia_vals.append(float(np.mean([ia for _, ia in pts])))
            continue
        pts_sorted = sorted(pts)
        ua_c = np.array([ua for ua, _ in pts_sorted])
        ia_c = np.array([ia for _, ia in pts_sorted])
        ia_vals.append(float(np.interp(ua_ref, ua_c, ia_c)))
    return ug1_arr, np.array(ia_vals)


def _interp_transfer(ug1_arr, ia_arr, target: float) -> float:
    """Interpolate Ia at target Ug1 from transfer curve.

    Accepts either ``np.ndarray`` or list inputs (used for ia_a list +
    ug1_arr ndarray in PP composite Q-point lookup).
    """
    if len(ug1_arr) == 0:
        return 0.0
    return float(np.interp(target, ug1_arr, ia_arr))


def _interp_transfer_cutoff(
    targets: np.ndarray, ug1_arr: np.ndarray, ia_arr: np.ndarray,
) -> np.ndarray:
    """Interpolate Ia(Ug1) with cutoff-side extrapolation (ML-139).

    Inside the measured range — plain ``np.interp``. BELOW the most
    negative measured Ug1 the tube is driven toward cutoff: extrapolate
    with the space-charge 3/2 law ``Ia = k·(Ug1 − Vc)^1.5`` (k, Vc fitted
    from the two edge points) — the plain clamp held the edge current as
    a phantom constant across the whole opposite half-swing of the PP
    composite, and a linear tail misses the convex approach to cutoff
    (probed vs Koren truth: clamp 10.6 → linear 0.47 → 3/2 0.22 mA mean
    error on EL84). ABOVE the most positive measured Ug1 the clamp is
    kept: the current keeps rising and grid current sets in, so upward
    extrapolation is speculative. Degenerate data degrades safely:
    Ia==0 at the edge → 0 beyond it; non-increasing edge segment (noise)
    → the old clamp.
    """
    ia = np.interp(targets, ug1_arr, ia_arr)
    if len(ug1_arr) >= 2:
        below = targets < ug1_arr[0]
        if np.any(below):
            u0 = float(ug1_arr[0])
            u1 = float(ug1_arr[1])
            i0 = float(ia_arr[0])
            i1 = float(ia_arr[1])
            if i0 <= 0.0:
                ia[below] = 0.0
            elif u1 <= u0 or i1 <= i0:
                ia[below] = i0
            else:
                r = (i0 / i1) ** (2.0 / 3.0)
                vc = (u0 - r * u1) / (1.0 - r)
                k = i0 / (u0 - vc) ** 1.5
                dv = np.maximum(targets[below] - vc, 0.0)
                ia[below] = k * dv ** 1.5
    return ia


def _b_extrap_result_keys(
    comp_stats: Dict, ug1_bias: float, half_swing_used: float,
    i_max: float, i_min: float,
) -> Dict:
    """Window-aware ML-139 flags for the PP result dicts.

    ``span`` = how deep the analyzed window's lower edge dips below tube
    B's measured Ug1 edge (0 when every sampled point is data-backed —
    matched pairs with auto/clamped swing never extrapolate).
    ``significant`` = the uncertain tail can shape the result: B's edge
    current relative to the analyzed FUNDAMENTAL AMPLITUDE
    (b1 ≈ (i_max − i_min)/2) exceeds the threshold. Normalising by the
    signal — not by the grid max — keeps the criterion invariant to how
    far the positive Ug1 side was scanned.
    """
    edge_b = comp_stats.get("b_edge_ug1")
    ia_edge = float(comp_stats.get("b_edge_ia_ma", 0.0))
    span = (max(0.0, float(edge_b) - (ug1_bias - half_swing_used))
            if edge_b is not None else 0.0)
    b1_est = max((float(i_max) - float(i_min)) / 2.0, 1e-9)
    frac = ia_edge / b1_est
    return {
        "b_extrapolation_span_v": span,
        "b_edge_ia_fraction": frac,
        "b_extrapolation_significant": bool(
            span > 0 and frac > B_EXTRAP_WARN_EDGE_FRACTION),
    }


# Type of a pre-built PP transfer pair: (ug1_a, ia_a, ug1_b, ia_b).
PPTransfer = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def build_pp_transfer(
    points_a: List[Dict],
    points_b: Optional[List[Dict]] = None,
    ug2_filter: Optional[float] = None,
    ua_ref: Optional[float] = None,
) -> PPTransfer:
    """Pre-build both PP transfer curves once per dataset.

    The curves depend only on (points, ug2_filter) — the optimizer's grid
    rebuilt them for every (ub, ra, ug1) combination, which was ~90% of a
    PP 5-point/Chebyshev evaluation. Pass the result into
    ``composite_characteristic`` / ``pp_distortion`` / the Chebyshev PP
    variant via their ``transfer`` parameter.
    """
    ug1_a, ia_a = _build_transfer_curve(points_a, ug2_filter, ua_ref=ua_ref)
    if points_b is not None:
        ug1_b, ia_b = _build_transfer_curve(points_b, ug2_filter, ua_ref=ua_ref)
    else:
        ug1_b, ia_b = ug1_a, ia_a
    return ug1_a, ia_a, ug1_b, ia_b


def composite_characteristic(
    points_a: List[Dict],
    points_b: Optional[List[Dict]] = None,
    ug1_bias: float = -10.0,
    ug2_filter: Optional[float] = None,
    transfer: Optional[PPTransfer] = None,
    ua_ref: Optional[float] = None,
    stats: Optional[Dict] = None,
) -> List[Dict]:
    """Build push-pull composite transfer characteristic.

    For Class AB: Ia_composite(Ug1) = Ia_A(Ug1) - Ia_B(2*bias - Ug1)
    The mirror is around the bias point, not around 0V.

    If points_b is None, assumes matched pair (Ia_B = Ia_A).
    In a perfectly matched pair, HD2 is completely cancelled because
    the composite function is odd-symmetric around the bias point.

    ``transfer`` accepts a pre-built :func:`build_pp_transfer` pair so hot
    callers (optimizer grid) skip re-deriving the curves per call.
    """
    if transfer is not None:
        ug1_a, ia_a, ug1_b, ia_b = transfer
    else:
        ug1_a, ia_a, ug1_b, ia_b = build_pp_transfer(
            points_a, points_b, ug2_filter, ua_ref=ua_ref)
    if len(ug1_a) < 2 or len(ug1_b) < 2:
        return []

    # Mirror all points in one interp call (same math as the former
    # per-point _interp_transfer loop — np.interp is element-wise).
    # ML-139: cutoff-side extrapolation — mirror points beyond tube B's
    # most negative measured Ug1 decay to 0, not to a phantom clamp.
    ug1_f = np.asarray(ug1_a, dtype=float)
    mirror = 2.0 * ug1_bias - ug1_f
    ug1_b_arr = np.asarray(ug1_b, dtype=float)
    ia_b_arr = np.asarray(ia_b, dtype=float)
    ia_b_vals = _interp_transfer_cutoff(mirror, ug1_b_arr, ia_b_arr)

    # ML-139 visibility (failure-visibility rule): report B's data-edge facts so
    # the analysis variants can decide WINDOW-AWARE whether extrapolated
    # values enter the sampled swing (the composite is built over A's
    # whole grid, but the analyzed window mirrors onto itself — only a
    # window edge below B's data edge consumes extrapolated points).
    # The edge current is ABSOLUTE: the variants normalise it by their
    # analyzed signal amplitude (a grid-max denominator would depend on
    # how far the positive side was scanned — not physical).
    if stats is not None and len(ug1_b_arr):
        stats["b_edge_ug1"] = float(ug1_b_arr[0])
        stats["b_edge_ia_ma"] = float(ia_b_arr[0])

    result: List[Dict] = [
        {
            "ug1": float(ug1_f[i]),
            "ia_a": float(ia_a[i]),
            "ia_b": float(ia_b_vals[i]),
            "ia_composite": float(ia_a[i]) - float(ia_b_vals[i]),
        }
        for i in range(ug1_f.shape[0])
    ]
    result.sort(key=lambda p: p["ug1"])
    return result


def fold_pp_composite(
    composite: List[Dict], ug1_bias: float,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Fold the composite characteristic into the positive-Ia quadrant.

    Display helper for the Transfer tab (its Y axis starts at 0, so the
    negative half of the composite would be invisible). Returns
    ``(direct, mirrored)``:

    * ``direct`` — points with ``ia_composite >= 0`` as-is;
    * ``mirrored`` — points with ``ia_composite < 0`` reflected around
      the bias (``ug1 → 2·bias − ug1``) AND sign-flipped
      (``ia → −ia``).

    For a perfectly matched pair the composite is odd-symmetric around
    the bias, so the two branches coincide — any visible gap between
    them is the even-harmonic residue (pair imbalance). Both branches
    are sorted by ug1.
    """
    direct = [(p["ug1"], p["ia_composite"]) for p in composite
              if p["ia_composite"] >= 0.0]
    mirrored = [(2.0 * ug1_bias - p["ug1"], -p["ia_composite"])
                for p in composite if p["ia_composite"] < 0.0]
    direct.sort()
    mirrored.sort()
    return direct, mirrored


# Joint-solve trajectory point count for display. Odd with
# (n-1) % 4 == 0: the drive grid then contains EXACTLY 0, +-swing/2
# and +-swing — Q/swing/half-point geometry reads off by direct
# indexing, no interpolation.
PP_TRAJECTORY_POINTS = 65


def pp_joint_trajectory(
    model: "TubeModelProtocol",
    load_line: PushPullLoadLine,
    ug1_bias: float,
    half_swing: float,
    ug2: float = 0.0,
    n_points: int = PP_TRAJECTORY_POINTS,
) -> Optional[Dict]:
    """Tube-A anode trajectory from the pair joint-solve (AB kink).

    Static twin of :func:`compute_distortion_dft_pp`: the same DC-Q
    and the same safeguarded-Newton joint-solve
    (`_pp_joint_solve_vec`), but over a MONOTONIC drive grid
    -half_swing..+half_swing — a static curve needs each drive once,
    not as a time wave. Near Q the tube sees Zaa/2, beyond partner
    cutoff Zaa/4: the kink emerges from the circuit equation, no
    heuristics.

    Returns::

        {
          "points": [{"ug1","ua","ia"}, ...],  # tube A, ug1 ascending
          "kink":   {"ug1","ua","ia"} | None,  # partner cutoff (A->AB);
        }                                      # None = pure class A

    ``None`` on a degenerate swing/load or non-finite model samples
    (failure visibility: the caller falls back to the straight display
    line WITH a label, not silently).
    """
    if half_swing < MIN_SWING_V or n_points < 5:
        return None
    ra_per_tube = load_line.ra_per_tube
    if ra_per_tube <= 0:
        return None
    ub = load_line.ub
    ra_dc = max(getattr(load_line, "ra_dc", 0.1), 1e-3)

    def _ia_fn(ua_arr: np.ndarray, g_arr: np.ndarray) -> np.ndarray:
        return model_ia_array(model, ua_arr, g_arr, ug2)

    # DC-Q as in dft_pp: bias-curve intersection with the DC line.
    ua_q_sol, _n_div, _resid = _newton_solve_vec(
        _ia_fn, lambda ua: (ub - ua) / ra_dc,
        np.array([ug1_bias]), ub, _DFT_NEWTON_MAX_ITER_PP)
    ua_q = float(ua_q_sol[0])

    drive = np.linspace(-half_swing, half_swing, n_points)
    ug1_a = ug1_bias + drive
    ug1_b = ug1_bias - drive
    ia_a, ia_b, ua_a, _ua_b, n_div, resid = _pp_joint_solve_vec(
        _ia_fn, _ia_fn, ug1_a, ug1_b, ua_q, ua_q,
        ra_per_tube, _DFT_NEWTON_MAX_ITER_PP)
    if not (np.all(np.isfinite(ia_a)) and np.all(np.isfinite(ua_a))):
        log.warning(
            "pp_joint_trajectory: non-finite model samples "
            "(bias=%.2f, swing=%.2f) — falling back to display line",
            ug1_bias, half_swing)
        return None
    if n_div:
        log.warning(
            "pp_joint_trajectory: %d/%d samples not converged "
            "(max residual %.3g mA)", n_div, n_points, resid)

    points = [
        {"ug1": float(g), "ua": float(u), "ia": float(i)}
        for g, u, i in zip(ug1_a, ua_a, ia_a)
    ]

    # A->AB kink: the first drive-up sample where the partner is cut
    # off (both conduct before it — Zaa/2, after — Zaa/4). Search only
    # the positive half: on the negative half tube A itself cuts off.
    kink = None
    pos = drive > 0
    off = np.nonzero(pos & (ia_b <= CUTOFF_IA_MA))[0]
    if off.size and off[0] > 0 and ia_b[off[0] - 1] > CUTOFF_IA_MA:
        kink = points[int(off[0])]

    return {"points": points, "kink": kink}


def diagnose_pp_distortion(
    points_a: List[Dict],
    load_line: PushPullLoadLine,
    ug1_bias: float,
    points_b: Optional[List[Dict]] = None,
    half_swing: Optional[float] = None,
    ug2_filter: Optional[float] = None,
) -> str:
    """Diagnose why pp_distortion would return None.

    Returns DIST_ERR_PP_* code; "" if compute should succeed.
    """
    comp = composite_characteristic(
        points_a, points_b, ug1_bias=ug1_bias, ug2_filter=ug2_filter,
        ua_ref=load_line.ub,
    )
    if len(comp) < 5:
        return DIST_ERR_PP_NO_COMPOSITE

    if load_line.ra_per_tube <= 0:
        return DIST_ERR_PP_RA_INVALID

    isects_proxy = [{"ug1": c["ug1"], "ua": 0.0, "ia": 0.0} for c in comp]
    code = diagnose_distortion(isects_proxy, ug1_bias=ug1_bias, half_swing=half_swing)
    if code in (DIST_ERR_BIAS_OUTSIDE, DIST_ERR_BIAS_AT_EDGE,
                DIST_ERR_MANUAL_SWING_SMALL, DIST_ERR_FEW_INTERSECTIONS,
                DIST_ERR_SPARSE_DATA):
        return code

    return DIST_ERR_UNKNOWN


def pp_distortion(
    points_a: List[Dict],
    load_line: PushPullLoadLine,
    ug1_bias: float,
    points_b: Optional[List[Dict]] = None,
    half_swing: Optional[float] = None,
    ug2_filter: Optional[float] = None,
    transfer: Optional[PPTransfer] = None,
) -> Optional[Dict]:
    """Compute push-pull specific distortion.

    Even harmonics are cancelled in PP. HD3 dominates.
    SE vs PP behaviour from "Rod Elliott (ESP) — Valves: Distortion and
    Intermodulation", see SOURCES_INDEX.md.

    Builds composite characteristic, then applies 5-point method
    on the composite Ia (which already includes PP cancellation).
    ``transfer`` forwards a pre-built :func:`build_pp_transfer` pair.
    """
    comp_stats: Dict = {}
    comp = composite_characteristic(
        points_a, points_b, ug1_bias=ug1_bias, ug2_filter=ug2_filter,
        transfer=transfer, ua_ref=load_line.ub, stats=comp_stats,
    )
    if len(comp) < 5:
        return None

    ra_per_tube = load_line.ra_per_tube
    if ra_per_tube <= 0:
        return None

    comp_sorted = sorted(comp, key=lambda p: p["ug1"])
    ug1_arr = [c["ug1"] for c in comp_sorted]
    ia_arr = [c["ia_composite"] for c in comp_sorted]
    ia_a_arr = [c["ia_a"] for c in comp_sorted]

    ug1_min = ug1_arr[0]
    ug1_max = ug1_arr[-1]

    manual_swing_clamped = False
    requested_half_swing = half_swing
    max_available = min(abs(ug1_bias - ug1_min), abs(ug1_max - ug1_bias))
    if half_swing is None:
        half_swing = max_available
    elif half_swing > max_available:
        # ML-052: a manual swing beyond the composite range used to sample
        # the flat interpolation endpoints — plausible numbers, silently
        # wrong. Clamp with the SE-path flag (the panel shows the notice).
        half_swing = max_available
        manual_swing_clamped = True

    if half_swing < MIN_SWING_V:
        return None

    # Sparse-data degeneracy guard (mirror of the SE path): without enough real
    # composite Ug1 curves inside the swing window every 5-point sample is a
    # linear interpolation between the same two lines → b2,b3 ≈ 0 → fake-low THD.
    ug1_neg = ug1_bias - half_swing
    ug1_pos = ug1_bias + half_swing
    edge_tol = max((ug1_pos - ug1_neg) * EDGE_TOL_REL, EDGE_TOL_ABS)
    ug1_in_swing = [u for u in ug1_arr
                    if ug1_neg - edge_tol <= u <= ug1_pos + edge_tol]
    if len(ug1_in_swing) < MIN_CURVES_IN_SWING:
        return None
    n_strictly_inside = sum(1 for u in ug1_in_swing
                            if ug1_neg + edge_tol < u < ug1_pos - edge_tol)
    if n_strictly_inside < 1:
        return None

    def _interp_comp(target_ug1: float) -> float:
        if target_ug1 <= ug1_arr[0]:
            return ia_arr[0]
        if target_ug1 >= ug1_arr[-1]:
            return ia_arr[-1]
        for i in range(len(ug1_arr) - 1):
            if ug1_arr[i] <= target_ug1 <= ug1_arr[i + 1]:
                span = ug1_arr[i + 1] - ug1_arr[i]
                if span == 0:
                    return ia_arr[i]
                t = (target_ug1 - ug1_arr[i]) / span
                return ia_arr[i] + t * (ia_arr[i + 1] - ia_arr[i])
        return ia_arr[-1]

    i_0 = _interp_comp(ug1_bias)
    i_max = _interp_comp(ug1_bias + half_swing)
    i_min = _interp_comp(ug1_bias - half_swing)
    i_high_half = _interp_comp(ug1_bias + half_swing / 2)
    i_low_half = _interp_comp(ug1_bias - half_swing / 2)

    iq_per_tube = _interp_transfer(ug1_arr, ia_a_arr, ug1_bias)
    # Per-tube minimum (tube A at its most negative grid excursion). The
    # COMPOSITE i_min is ≈ −i_max for a matched pair — classifying amp
    # class on it made every PP point come out "B".
    i_min_per_tube = _interp_transfer(
        ug1_arr, ia_a_arr, ug1_bias - half_swing)

    swing = i_max - i_min
    if abs(swing) < BALANCE_SWING_NEAR_ZERO:
        return None

    half_diff = i_high_half - i_low_half
    b1 = (swing + half_diff) / 3.0
    if b1 <= 0:
        return None
    b2 = (i_max + i_min - 2.0 * i_0) / 4.0
    b3 = (swing - 2.0 * half_diff) / 6.0

    hd2 = abs(b2 / b1) * 100.0
    hd3 = abs(b3 / b1) * 100.0
    thd = math.sqrt(hd2 ** 2 + hd3 ** 2)

    # PP output power: P = Ipp_composite × Vpp_composite / 8.
    ua_swing = swing * ra_per_tube
    pout_mw = (swing * ua_swing) / 8.0
    balance_error = hd2

    return {
        "hd2": hd2,
        "hd3": hd3,
        "thd": thd,
        "pout_mw": pout_mw,
        "balance_error": balance_error,
        "ug1_0": ug1_bias,
        "ia_0": i_0,
        "iq_per_tube": iq_per_tube,
        "i_min_per_tube": i_min_per_tube,
        # ML-052: SE convention — the panel shows a clamp notice
        "manual_swing_clamped": manual_swing_clamped,
        "requested_half_swing": requested_half_swing,
        "half_swing": half_swing,
        "i_max": i_max,
        "i_min": i_min,
        "method": HD_METHOD_5POINT,
        **_b_extrap_result_keys(comp_stats, ug1_bias, half_swing,
                                i_max, i_min),
    }


def compute_distortion_chebyshev_pp(
    points_a: List[Dict],
    load_line: PushPullLoadLine,
    ug1_bias: float,
    points_b: Optional[List[Dict]] = None,
    half_swing: Optional[float] = None,
    ug2_filter: Optional[float] = None,
    max_harmonic: int = 9,
    transfer: Optional[PPTransfer] = None,
) -> Optional[Dict]:
    """Chebyshev polynomial harmonic distortion for push-pull composite.

    Same idea as compute_distortion_chebyshev (SE) but operates on the
    composite Ia(Ug1) characteristic produced by composite_characteristic.

    For matched pairs the composite is odd-symmetric → even Chebyshev
    coefficients (c0, c2, c4, ...) ≈ 0 → HD2, HD4 ≈ 0. Odd coefficients
    (c1, c3, c5, c7, c9) carry the actual harmonic content.
    ``transfer`` forwards a pre-built :func:`build_pp_transfer` pair.
    """
    from numpy.polynomial import chebyshev

    comp_stats: Dict = {}
    comp = composite_characteristic(
        points_a, points_b, ug1_bias=ug1_bias, ug2_filter=ug2_filter,
        transfer=transfer, ua_ref=load_line.ub, stats=comp_stats,
    )
    if len(comp) < MIN_CHEBYSHEV_HARMONIC + 1:
        return None

    ra_per_tube = load_line.ra_per_tube
    if ra_per_tube <= 0:
        return None

    comp_sorted = sorted(comp, key=lambda p: p["ug1"])
    ug1_arr = np.array([c["ug1"] for c in comp_sorted])
    ia_arr = np.array([c["ia_composite"] for c in comp_sorted])
    ia_a_arr = [c["ia_a"] for c in comp_sorted]

    ug1_min, ug1_max = ug1_arr[0], ug1_arr[-1]
    bias = max(ug1_min, min(ug1_max, ug1_bias))

    if half_swing is not None and 0 < half_swing < MIN_SWING_V:
        # ML-051 (PP chebyshev twin, mutation-audit): tiny manual swing is
        # a user error, not a silent full-auto substitute. 0.0 = "auto".
        return None
    manual_swing_clamped = False
    requested_half_swing = None
    if half_swing is not None and half_swing > MIN_SWING_V:
        requested_half_swing = float(half_swing)
        A = min(half_swing, bias - ug1_min, ug1_max - bias)
        if A < half_swing - 1e-9:
            # Mutation-audit: chebyshev clamped silently while
            # 5-point set the SE-convention flag — the panel notice never
            # fired for the chebyshev method.
            manual_swing_clamped = True
    else:
        A = min(bias - ug1_min, ug1_max - bias)
    if A < MIN_SWING_V:
        return None

    u = (ug1_arr - bias) / A
    mask = (u >= -CHEBYSHEV_BOUNDARY) & (u <= CHEBYSHEV_BOUNDARY)
    u_fit = np.clip(u[mask], -1.0, 1.0)
    ia_fit = ia_arr[mask]
    if len(u_fit) < MIN_CHEBYSHEV_HARMONIC + 1:
        return None

    safe_max = max(2 * len(u_fit) // CHEBYSHEV_OVERFIT_RATIO, MIN_CHEBYSHEV_HARMONIC)
    if max_harmonic > safe_max:
        max_harmonic = safe_max

    decomp = _chebyshev_decompose(u_fit, ia_fit, max_harmonic)
    if decomp is None:
        return None
    coeffs, c1, harmonics, thd = decomp

    ia_0 = float(chebyshev.chebval(0.0, coeffs))
    iq_per_tube = _interp_transfer(list(ug1_arr), ia_a_arr, bias)
    # Per-tube minimum for amp-class detection — see pp_distortion.
    i_min_per_tube = _interp_transfer(list(ug1_arr), ia_a_arr, bias - A)

    ia_pos = float(chebyshev.chebval(1.0, coeffs))
    ia_neg = float(chebyshev.chebval(-1.0, coeffs))
    swing = ia_pos - ia_neg
    if abs(swing) < BALANCE_SWING_NEAR_ZERO:
        return None
    ua_swing = swing * ra_per_tube
    pout_mw = (swing * ua_swing) / 8.0

    return {
        **harmonics,
        "thd": thd,
        "pout_mw": pout_mw,
        "balance_error": harmonics.get("hd2", 0.0),
        "ug1_0": bias,
        "ia_0": ia_0,
        "iq_per_tube": iq_per_tube,
        "i_min_per_tube": i_min_per_tube,
        "half_swing": A,
        "i_max": ia_pos,
        "i_min": ia_neg,
        "method": HD_METHOD_CHEBYSHEV_PP,
        **_b_extrap_result_keys(comp_stats, bias, A, ia_pos, ia_neg),
        "manual_swing_clamped": manual_swing_clamped,
        "requested_half_swing": requested_half_swing,
        "max_harmonic": max_harmonic,
    }


# Chebyshev–Gauss node count for the model-based PP Chebyshev method.
# Nodes u_k = cos(θ_k) with UNIFORM θ_k — node averages are time averages.
# 33 ≥ 2·max_harmonic(9)+1 with margin — ~30× fewer Newton solves than
# the 1024-sample DFT; measured wall speedup ≈ ×1.8 (vectorization
# amortizes samples; the Newton iterations dominate). See
# docs/UL_CHEBYSHEV_VALIDATION.md.
_CHEB_MODEL_PP_NODES = 33


def compute_distortion_chebyshev_pp_model(
    model: "TubeModelProtocol",
    load_line: PushPullLoadLine,
    ug1_bias: float,
    half_swing: float,
    ug2: float = 0.0,
    model_b: Optional["TubeModelProtocol"] = None,
    max_harmonic: int = 9,
    n_nodes: int = _CHEB_MODEL_PP_NODES,
) -> Optional[Dict]:
    """Chebyshev harmonics for the PP composite computed FROM THE MODEL.

    Purpose: a fast grid method for the UL tap sweep. The data-based
    :func:`compute_distortion_chebyshev_pp` cannot see an UL tap (the
    measured points were taken at fixed real Ug2); this variant solves
    the same coupled ideal-OPT equation as :func:`compute_distortion_dft_pp`
    (same ``_pp_joint_solve_vec`` — class-AB kink included) but at
    Chebyshev–Gauss nodes of the drive instead of 1024 sine samples
    (~×1.8 wall-clock on the grid).

    Mathematical equivalence to DFT on a memoryless composite: the drive
    is u = cos θ, the solved i_comp(u) is a static map, and
    T_n(cos θ) = cos nθ — so the Chebyshev coefficients ARE the harmonic
    amplitudes. Differences from DFT are pure polynomial truncation
    (harmonics above ``max_harmonic`` fold into the residual instead of
    being measured) — pinned against DFT in
    ``tests/test_ul_chebyshev_model.py`` and quantified in
    ``docs/UL_CHEBYSHEV_VALIDATION.md``.
    """
    if half_swing is None or half_swing <= 0:
        return None
    if model_b is None:
        model_b = model

    ra_per_tube = load_line.ra_per_tube
    if ra_per_tube <= 0:
        return None
    ub = load_line.ub
    ra_dc = max(getattr(load_line, "ra_dc", 0.1), 1e-3)

    _newton_stats = {"diverged": 0, "max_resid": 0.0}

    def _ia_a_fn(ua_arr: np.ndarray, g_arr: np.ndarray) -> np.ndarray:
        return model_ia_array(model, ua_arr, g_arr, ug2)

    def _ia_b_fn(ua_arr: np.ndarray, g_arr: np.ndarray) -> np.ndarray:
        return model_ia_array(model_b, ua_arr, g_arr, ug2)

    def _solve_dc(ia_fn, ug1_val: float) -> Tuple[float, float]:
        ua_sol, n_div, resid = _newton_solve_vec(
            ia_fn, lambda ua: (ub - ua) / ra_dc,
            np.array([ug1_val]), ub, _DFT_NEWTON_MAX_ITER_PP)
        _newton_stats["diverged"] += n_div
        if resid > _newton_stats["max_resid"]:
            _newton_stats["max_resid"] = resid
        ia_sol = np.maximum(0.0, ia_fn(ua_sol, np.array([ug1_val])))
        return float(ia_sol[0]), float(ua_sol[0])

    iq_a, ua_q_a = _solve_dc(_ia_a_fn, ug1_bias)
    _iq_b, ua_q_b = _solve_dc(_ia_b_fn, ug1_bias)

    # Chebyshev–Gauss nodes: θ uniform over (0, π) → u = cos θ ∈ (−1, 1).
    theta = np.pi * (np.arange(n_nodes) + 0.5) / n_nodes
    u = np.cos(theta)
    ug1_a = ug1_bias + half_swing * u
    ug1_b = ug1_bias - half_swing * u

    ia_a, ia_b, ua_a, _ua_b, n_div, resid = _pp_joint_solve_vec(
        _ia_a_fn, _ia_b_fn, ug1_a, ug1_b, ua_q_a, ua_q_b,
        ra_per_tube, _DFT_NEWTON_MAX_ITER_PP,
    )
    _newton_stats["diverged"] += n_div
    if resid > _newton_stats["max_resid"]:
        _newton_stats["max_resid"] = resid
    ia_comp = ia_a - ia_b

    # ML-138 (PP model-Chebyshev twin): non-finite model samples must not
    # reach the Chebyshev fit — NaN passes every downstream comparison.
    if not np.all(np.isfinite(ia_comp)):
        log.warning(
            "PP model-Chebyshev: %d non-finite model sample(s) "
            "(ug1_bias=%.2f, half_swing=%.2f) — model invalid in the "
            "evaluated region",
            int(np.count_nonzero(~np.isfinite(ia_comp))),
            ug1_bias, half_swing)
        return None

    if _newton_stats["diverged"] > 0:
        log.warning(
            "PP model-Chebyshev Newton: %d/%d solves did not converge "
            "(max residual %.3g mA, ug1_bias=%.2f, half_swing=%.2f) — "
            "THD may be biased near clipping",
            _newton_stats["diverged"], 2 + n_nodes,
            _newton_stats["max_resid"], ug1_bias, half_swing,
        )

    decomp = _chebyshev_decompose(u, ia_comp, max_harmonic)
    if decomp is None:
        return None
    coeffs, c1, harmonics, thd = decomp
    if abs(c1) < MIN_B1_MA:
        return None

    from numpy.polynomial import chebyshev
    ia_pos = float(chebyshev.chebval(1.0, coeffs))
    ia_neg = float(chebyshev.chebval(-1.0, coeffs))
    swing = ia_pos - ia_neg
    if abs(swing) < BALANCE_SWING_NEAR_ZERO:
        return None
    ua_swing = swing * ra_per_tube
    pout_mw = (swing * ua_swing) / 8.0

    # θ is uniform → node means ARE time averages (DFT-comparable).
    ia_0 = float(coeffs[0])
    pa_avg_mw = float(np.mean(ua_a * ia_a))

    return {
        **harmonics,
        "thd": thd,
        "pout_mw": pout_mw,
        "balance_error": harmonics.get("hd2", 0.0),
        "ug1_0": ug1_bias,
        "ia_0": ia_0,
        "iq_per_tube": iq_a,
        "i_min_per_tube": float(np.min(ia_a)),
        "half_swing": half_swing,
        "pout_fund_mw": abs(c1) * abs(c1) * ra_per_tube / 2.0,
        "pa_avg_mw": pa_avg_mw,
        "i_max": float(np.max(ia_comp)),
        "i_min": float(np.min(ia_comp)),
        "method": HD_METHOD_CHEBYSHEV_MODEL_PP,
        "max_harmonic": max_harmonic,
        "n_samples": n_nodes,
        "n_not_converged": int(_newton_stats["diverged"]),
    }


def compute_distortion_dft_pp(
    model: "TubeModelProtocol",
    load_line: PushPullLoadLine,
    ug1_bias: float,
    half_swing: float,
    ug2: float = 0.0,
    model_b: Optional["TubeModelProtocol"] = None,
    n_samples: int = 1024,
    max_harmonic: int = 9,
) -> Optional[Dict]:
    """DFT-based harmonic distortion for push-pull composite via model.

    Generates a sinusoidal Ug1 around bias, solves BOTH tubes jointly at
    every time sample against the ideal center-tapped OPT constraint
    (see :func:`_pp_joint_solve_vec`), then builds composite
    Ia(t) = ia_a(t) − ia_b(t) and applies FFT to extract HD2..HD9.

    The joint solve models the class-AB per-tube impedance kink exactly:
    Ra_aa/2 per tube while both conduct → Ra_aa/4 once the partner cuts
    off (formerly a fixed Ra_aa/4 line per tube — the documented <15%
    Pout approximation). UL/Triode wrappers still see the true varying
    per-tube Ua → modulated Ug2 → proper screen linearization.
    """
    if half_swing < MIN_SWING_V:
        return None
    if model_b is None:
        model_b = model

    ra_per_tube = load_line.ra_per_tube
    if ra_per_tube <= 0:
        return None
    ub = load_line.ub
    ra_dc = max(getattr(load_line, "ra_dc", 0.1), 1e-3)

    # Shared per-call Newton convergence stats (DC q-point + AC sweeps for
    # both tubes). See compute_distortion_dft for the same diagnostic
    # rationale: divergence indicates Newton bouncing against the
    # ``max(0, ua)`` clamp under overdrive — biases harmonic content.
    # Every solve (DC points and each AC sample) cold-starts independently,
    # so they all run through the vectorized solver.
    _newton_stats = {"diverged": 0, "max_resid": 0.0}

    def _ia_a_fn(ua_arr: np.ndarray, g_arr: np.ndarray) -> np.ndarray:
        return model_ia_array(model, ua_arr, g_arr, ug2)

    def _ia_b_fn(ua_arr: np.ndarray, g_arr: np.ndarray) -> np.ndarray:
        return model_ia_array(model_b, ua_arr, g_arr, ug2)

    def _solve(ia_fn, ll_fn_, ug1_arr, ua_init) -> Tuple[np.ndarray, np.ndarray]:
        """Vector Newton + the scalar path's final Ia re-read at the root."""
        ua_sol, n_div, resid = _newton_solve_vec(
            ia_fn, ll_fn_, ug1_arr, ua_init, _DFT_NEWTON_MAX_ITER_PP)
        _newton_stats["diverged"] += n_div
        if resid > _newton_stats["max_resid"]:
            _newton_stats["max_resid"] = resid
        ia_sol = np.maximum(0.0, ia_fn(ua_sol, ug1_arr))
        return ia_sol, ua_sol

    def _ll_dc(ua):
        return (ub - ua) / ra_dc

    bias_arr = np.array([ug1_bias])
    iq_a_v, ua_q_a_v = _solve(_ia_a_fn, _ll_dc, bias_arr, ub)
    iq_b_v, ua_q_b_v = _solve(_ia_b_fn, _ll_dc, bias_arr, ub)
    iq_a, ua_q_a = float(iq_a_v[0]), float(ua_q_a_v[0])
    iq_b, ua_q_b = float(iq_b_v[0]), float(ua_q_b_v[0])

    t = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    ug1_a = ug1_bias + half_swing * np.sin(t)
    ug1_b = ug1_bias - half_swing * np.sin(t)

    # Joint ideal-OPT solve — one coupled unknown per sample (see
    # _pp_joint_solve_vec; models the class-AB impedance kink exactly).
    ia_a, ia_b, _ua_a, _ua_b, n_div, resid = _pp_joint_solve_vec(
        _ia_a_fn, _ia_b_fn, ug1_a, ug1_b, ua_q_a, ua_q_b,
        ra_per_tube, _DFT_NEWTON_MAX_ITER_PP,
    )
    _newton_stats["diverged"] += n_div
    if resid > _newton_stats["max_resid"]:
        _newton_stats["max_resid"] = resid
    ia_comp = ia_a - ia_b

    # ML-138 (PP DFT twin): non-finite model samples must not reach the
    # FFT — NaN passes every downstream comparison.
    if not np.all(np.isfinite(ia_comp)):
        log.warning(
            "PP DFT: %d non-finite model sample(s) (ug1_bias=%.2f, "
            "half_swing=%.2f) — model invalid in the evaluated region",
            int(np.count_nonzero(~np.isfinite(ia_comp))),
            ug1_bias, half_swing)
        return None

    if _newton_stats["diverged"] > 0:
        # Total Newton solves: 2 (DC q-points) + n_samples (joint AC)
        total_calls = 2 + n_samples
        log.warning(
            "PP DFT Newton: %d/%d invocations did not converge in 20 iters "
            "(max residual %.3g mA, ug1_bias=%.2f, half_swing=%.2f) — "
            "THD may be biased near clipping",
            _newton_stats["diverged"], total_calls, _newton_stats["max_resid"],
            ug1_bias, half_swing,
        )

    spectrum = np.fft.rfft(ia_comp)
    magnitudes = (2.0 / n_samples) * np.abs(spectrum)
    if magnitudes[1] < MIN_B1_MA:
        return None

    b1 = float(magnitudes[1])
    harmonics: Dict[str, float] = {}
    for n in range(2, max_harmonic + 1):
        if n < len(magnitudes):
            harmonics[f"hd{n}"] = float(magnitudes[n] / b1 * 100.0)
        else:
            harmonics[f"hd{n}"] = 0.0
    thd = math.sqrt(sum(h ** 2 for h in harmonics.values()))

    iq_per_tube = iq_a
    ia_0 = float(np.mean(ia_comp))
    swing = float(np.max(ia_comp) - np.min(ia_comp))
    ua_swing = swing * ra_per_tube
    pout_mw = (swing * ua_swing) / 8.0
    # Fundamental-only power: P1 = I1²·Ra_per_tube/2 (composite current
    # fundamental b1 in mA, v = Ra_pt·i_comp) — like-for-like with
    # external fundamental-power references.
    pout_fund_mw = b1 * b1 * ra_per_tube / 2.0
    # Per-tube average anode dissipation UNDER SIGNAL (mW) — informational
    # field: the optimizer's Pa constraint still checks the Q-point only.
    pa_avg_mw = float(np.mean(_ua_a * ia_a))

    return {
        **harmonics,
        "thd": thd,
        "pout_mw": pout_mw,
        "balance_error": harmonics.get("hd2", 0.0),
        "ug1_0": ug1_bias,
        "ia_0": ia_0,
        "iq_per_tube": iq_per_tube,
        # Per-tube waveform minimum — exact here (see pp_distortion).
        "i_min_per_tube": float(np.min(ia_a)),
        "half_swing": half_swing,
        "pout_fund_mw": pout_fund_mw,
        "pa_avg_mw": pa_avg_mw,
        "i_max": float(np.max(ia_comp)),
        "i_min": float(np.min(ia_comp)),
        "method": HD_METHOD_DFT_PP,
        "max_harmonic": max_harmonic,
        "n_samples": n_samples,
        "n_not_converged": int(_newton_stats["diverged"]),
    }


# ─── IMD (SE) ────────────────────────────────────────────────────────

def compute_imd(
    intersections: List[Dict],
    ug1_bias: Optional[float] = None,
    half_swing: Optional[float] = None,
) -> Optional[Dict]:
    """Intermodulation distortion from polynomial Ia(Ug1).

    Taylor-series IMD from "Terman — Electronic and Radio Engineering,
    4th ed. (1955)", see SOURCES_INDEX.md.

    When ``half_swing`` is provided, only intersection points within
    [ug1_bias - half_swing, ug1_bias + half_swing] are used for fitting.
    This gives IMD coefficients relevant to the actual signal swing.
    """
    from lm19.constants import EPS

    if len(intersections) < 4:
        return None

    pts = sorted(intersections, key=lambda p: p["ug1"])
    ug1_arr = np.array([p["ug1"] for p in pts])
    ia_arr = np.array([p["ia"] for p in pts])

    if ug1_bias is not None:
        ug1_0 = ug1_bias
    else:
        ug1_0 = (ug1_arr[0] + ug1_arr[-1]) / 2.0

    if half_swing is not None and half_swing > 0:
        mask = np.abs(ug1_arr - ug1_0) <= half_swing * 1.05
        ug1_arr = ug1_arr[mask]
        ia_arr = ia_arr[mask]
        if len(ug1_arr) < 4:
            return None

    x = ug1_arr - ug1_0
    deg = 3 if len(x) >= 4 else min(2, len(x) - 1)

    try:
        coeffs = np.polyfit(x, ia_arr, deg)
    except (np.linalg.LinAlgError, ValueError):
        return None

    coeffs = coeffs[::-1]
    a0 = coeffs[0] if len(coeffs) > 0 else 0
    a1 = coeffs[1] if len(coeffs) > 1 else 0
    a2 = coeffs[2] if len(coeffs) > 2 else 0
    a3 = coeffs[3] if len(coeffs) > 3 else 0

    if abs(a1) < EPS:
        return None

    imd2 = abs(a2 / a1) * 100.0
    imd3 = abs(a3 / a1) * 100.0 if a3 != 0 else 0.0
    imd_total = math.sqrt(imd2 ** 2 + imd3 ** 2)

    return {
        "imd2": imd2, "imd3": imd3, "imd_total": imd_total,
        "a0": a0, "a1": a1, "a2": a2, "a3": a3,
    }
