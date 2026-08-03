"""Adaptive scan refinement: interval analysis + bisection + inline refine.

Two flavors of interval analysis:
- ``_find_refine_intervals`` — union across all Ug1 (used by independent Ug2
  sweep where each refine pass iterates all Ug1×Ua anyway).
- ``_find_refine_intervals_per_ug1`` — per-Ug1 dict (used by triode and
  ug2_track modes where Ug1's "interesting" Ua region varies).

``_refine_curve_inline`` is called immediately after each coarse Ua curve
(while Ug1/Ug2 are still settled) to add bisected Ua points without
re-traversing the whole grid.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from lm19.constants import EPS_COARSE, UA_ROUND
from lm19.scan.events import ScanProgress
from lm19.scan.exceptions import _SkipPoint
from lm19.scan.protection import _exceeds_ig2, _exceeds_pa, _exceeds_pg2
from lm19.scan.settings import (
    _DOWN_SWEEP_GAP_FACTOR,
    _GRID_INTERVAL_MARGIN,
    _IA_RANGE_MIN,
    _PREDICT_MAX_OVER_LIMIT,
    _QUADRATIC_MIN_BASELINE_V,
    _UA_DELTA_MIN,
    ScanSettings,
)


# ---------------------------------------------------------------------------
# Predictive Ig2 check for down-sweep
# ---------------------------------------------------------------------------

def _predict_ig2(
    down_pts: List[Dict],
    next_ua: float,
    ig2_limit: float = 0.0,
) -> Optional[float]:
    """Predict Ig2 at *next_ua* from already measured down-sweep points.

    Uses linear extrapolation for 2 points and quadratic for 3+, with
    physical sanity guards so a noise-amplified extrapolation cannot
    fire a bogus predictive break.

    Args:
        down_pts: down-sweep measurement points (descending Ua).
        next_ua: target Ua for prediction.
        ig2_limit: Ig2 protection limit in mA. When >0, predictions
            above ``_PREDICT_MAX_OVER_LIMIT × ig2_limit`` are clipped
            to ``None`` (caller treats as "no prediction available")
            because such values are extrapolation blow-up, not reality.

    Returns:
        Predicted Ig2 (mA), or None when prediction is unreliable
        (insufficient points, degenerate Ua spacing, unphysical result).

    Stability notes:
        - ``_UA_DELTA_MIN`` (0.01 V) is the divide-by-zero guard.
        - ``_QUADRATIC_MIN_BASELINE_V`` (0.5 V) is the noise-control
          baseline: quadratic fit needs a wide-enough span across the
          three points or the curvature coefficient ``a`` becomes
          dominated by measurement noise (1/dua_total amplification).
          When the baseline is too narrow, fall back to the linear
          extrapolation between the last two points.
        - Negative predictions are unphysical (Ig2 ≥ 0); clamp to the
          linear fallback (also clamped to 0).
    """
    n = len(down_pts)
    if n < 2:
        return None

    ua2, ig2_2 = down_pts[-1]["ua"], down_pts[-1]["ig2"]
    ua1, ig2_1 = down_pts[-2]["ua"], down_pts[-2]["ig2"]
    dua = ua2 - ua1
    if abs(dua) < _UA_DELTA_MIN:
        return None

    # Linear extrapolation — used as primary path (n=2) and as fallback
    # whenever the quadratic path is rejected.
    linear_slope = (ig2_2 - ig2_1) / dua
    linear_pred = ig2_2 + linear_slope * (next_ua - ua2)

    if n >= 3:
        ua0, ig2_0 = down_pts[-3]["ua"], down_pts[-3]["ig2"]
        dua0 = ua1 - ua0
        dua_total = ua2 - ua0
        # Two guards before trusting the quadratic fit:
        #   1. dua0 ≥ _UA_DELTA_MIN  — slope s1 not divide-by-zero
        #   2. |dua_total| ≥ _QUADRATIC_MIN_BASELINE_V — coefficient
        #      ``a = (s2 - s1)/dua_total`` is noise-stable (see settings.py).
        if (abs(dua0) >= _UA_DELTA_MIN
                and abs(dua_total) >= _QUADRATIC_MIN_BASELINE_V):
            s1 = (ig2_1 - ig2_0) / dua0
            s2 = (ig2_2 - ig2_1) / dua
            a = (s2 - s1) / dua_total
            dt = next_ua - ua2
            quad_pred = ig2_2 + s2 * dt + a * dt * dt

            # Negative Ig2 is unphysical → trust linear instead.
            if quad_pred < 0:
                return max(linear_pred, 0.0)
            # Astronomical predictions (5×+ over limit) are noise-blow-up,
            # not real curve behaviour — return None so the caller does
            # NOT fire a predictive break on garbage.
            if (ig2_limit > 0
                    and quad_pred > _PREDICT_MAX_OVER_LIMIT * ig2_limit):
                return None
            return quad_pred
        # Quadratic rejected → linear fallback (still clamp negatives).

    return max(linear_pred, 0.0)


# ---------------------------------------------------------------------------
# Down-sweep step bisection
# ---------------------------------------------------------------------------

def _build_down_sweep_ua(
    grid_ua_desc: List[float], prev_ua: float, max_step: float,
) -> List[Tuple[float, bool]]:
    """Build down-sweep Ua values, bisecting steps that exceed *max_step*.

    *grid_ua_desc* — grid Ua values in descending order (reversed slice).
    *prev_ua* — Ua after the soft-landing settle (typically ua_values[start_idx]).
    Returns a list of ``(ua, is_grid)`` tuples in descending order.
    Intermediate (non-grid) points are marked ``is_grid=False`` — they are
    used for safe movement and Ig2 checks but should NOT be saved as data.
    """
    if max_step <= 0:
        return [(ua, True) for ua in grid_ua_desc]
    result: List[Tuple[float, bool]] = []
    for ua in grid_ua_desc:
        gap = prev_ua - ua
        if gap > max_step * _DOWN_SWEEP_GAP_FACTOR:
            n_parts = max(2, round(gap / max_step))
            sub = gap / n_parts
            for i in range(1, n_parts):
                result.append((round(prev_ua - i * sub, UA_ROUND), False))
        result.append((ua, True))
        prev_ua = ua
    return result


def _closest_grid_idx(value: float, grid: List[float]) -> int:
    """Return index of the closest value in a sorted grid."""
    best = 0
    best_d = abs(value - grid[0])
    for i in range(1, len(grid)):
        d = abs(value - grid[i])
        if d < best_d:
            best_d = d
            best = i
    return best


def _mark_grid_intervals(
    ua_lo: float, ua_hi: float, ua_values: List[float], out: set,
) -> None:
    """Add all ua_values interval indices that overlap [ua_lo, ua_hi] to *out*."""
    for i in range(len(ua_values) - 1):
        if ua_values[i + 1] >= ua_lo - _GRID_INTERVAL_MARGIN and ua_values[i] <= ua_hi + _GRID_INTERVAL_MARGIN:
            out.add(i)


def _analyse_curve_intervals(
    curve: List[Dict],
    ua_values: List[float],
    onset: float,
    curv_thr: float,
    grad_thr: float,
    ig2_min: float,
    dia_thr: float,
    is_pentode_sweep: bool,
) -> set:
    """Analyse a single Ug1 curve and return interval indices needing refinement.

    Applies five criteria (C1–C5):
    C1 — onset (Ia crosses onset threshold),
    C2 — normalised curvature,
    C3 — gradient ratio,
    C4 — Ig2 non-monotonicity / kink (pentode independent sweep only),
    C5 — large absolute Ia jump.
    """
    curve.sort(key=lambda p: p["ua"])
    n = len(curve)
    if n < 2:
        return set()

    refine: set = set()
    ia = [p["ia"] for p in curve]
    ig2 = [p.get("ig2", 0.0) for p in curve]
    ua = [p["ua"] for p in curve]
    ia_range = max(max(ia) - min(ia), _IA_RANGE_MIN)

    for j in range(n):
        # ---- C1: Onset (0 → current) ----
        if j < n - 1:
            if ia[j] < onset and ia[j + 1] > onset:
                _mark_grid_intervals(ua[j], ua[j + 1], ua_values, refine)

        # ---- C5: Large absolute Ia jump ----
        if j < n - 1:
            if abs(ia[j + 1] - ia[j]) / ia_range > dia_thr:
                _mark_grid_intervals(ua[j], ua[j + 1], ua_values, refine)

        # Criteria requiring 3 consecutive points (j-1, j, j+1)
        if j == 0 or j == n - 1:
            continue

        # ---- C2: Normalised curvature ----
        d2 = abs(ia[j + 1] - 2.0 * ia[j] + ia[j - 1])
        if d2 / ia_range > curv_thr:
            _mark_grid_intervals(ua[j - 1], ua[j + 1], ua_values, refine)

        # ---- C3: Gradient ratio ----
        step_prev = ua[j] - ua[j - 1]
        step_next = ua[j + 1] - ua[j]
        if step_prev > 0 and step_next > 0:
            g_prev = abs((ia[j] - ia[j - 1]) / step_prev)
            g_next = abs((ia[j + 1] - ia[j]) / step_next)
            g_min = min(g_prev, g_next)
            g_max = max(g_prev, g_next)
            if g_min > EPS_COARSE and g_max / g_min > grad_thr:
                _mark_grid_intervals(ua[j - 1], ua[j + 1], ua_values, refine)

        # ---- C4: Ig2 non-monotonicity / kink (pentode only) ----
        if is_pentode_sweep:
            d_prev = ig2[j] - ig2[j - 1]
            d_next = ig2[j + 1] - ig2[j]
            if d_prev * d_next < 0 and abs(d_next - d_prev) > ig2_min:
                _mark_grid_intervals(ua[j - 1], ua[j + 1], ua_values, refine)

    # Clamp to valid range
    max_idx = len(ua_values) - 2
    return {i for i in refine if 0 <= i <= max_idx}


def _find_refine_intervals(
    curve_points: List[Dict],
    ua_values: List[float],
    settings: ScanSettings,
) -> set:
    """Analyze coarse curves and find Ua grid intervals needing refinement.

    Returns the **union** of all triggered interval indices across all
    Ug1 curves.  Used by the independent Ug2 sweep branch where the
    refine pass iterates all Ug1×refine_ua combinations anyway.
    """
    if len(ua_values) < 3 or not curve_points:
        return set()

    is_pentode_sweep = not settings.is_triode and not settings.ug2_track_ua
    onset = settings.refine_onset_ma
    curv_thr = settings.refine_curvature_thr
    grad_thr = settings.refine_gradient_ratio
    ig2_min = settings.refine_ig2_delta_min
    dia_thr = settings.refine_delta_ia_thr

    # Group by Ug1
    ug1_map: Dict[float, List[Dict]] = {}
    for p in curve_points:
        key = round(p["ug1"], 1)
        ug1_map.setdefault(key, []).append(p)

    refine: set = set()
    for _ug1_key, curve in ug1_map.items():
        refine |= _analyse_curve_intervals(
            curve, ua_values, onset, curv_thr, grad_thr,
            ig2_min, dia_thr, is_pentode_sweep,
        )
    return refine


def _snap_ug1_key(measured: float, ug1_grid: Optional[List[float]]) -> float:
    """Map a measured Ug1 to the nearest settings grid value (rounded).

    The device may report a slightly different Ug1 than what was set.
    Snapping ensures the per-Ug1 refine dict keys match the settings-
    based loop keys used during the refine measurement pass.
    """
    m = round(measured, 1)
    if not ug1_grid:
        return m
    best = ug1_grid[0]
    best_d = abs(round(best, 1) - m)
    for v in ug1_grid[1:]:
        d = abs(round(v, 1) - m)
        if d < best_d:
            best_d = d
            best = v
    return round(best, 1)


def _find_refine_intervals_per_ug1(
    curve_points: List[Dict],
    ua_values: List[float],
    settings: ScanSettings,
    ug1_values: Optional[List[float]] = None,
) -> Dict[float, set]:
    """Like _find_refine_intervals but returns per-Ug1 results.

    Returns ``{ug1_key: set_of_interval_indices}``.  Used by triode and
    ug2_track_ua modes where each Ug1 curve's "interesting" region
    (onset, knee) is at a different Ua — taking the union would refine
    nearly the entire range, defeating the purpose of adaptive refinement.

    When *ug1_values* is provided, measured Ug1 values are snapped to the
    nearest settings grid value so that the dict keys match the values
    used in the refine measurement loop.
    """
    if len(ua_values) < 3 or not curve_points:
        return {}

    is_pentode_sweep = not settings.is_triode and not settings.ug2_track_ua
    onset = settings.refine_onset_ma
    curv_thr = settings.refine_curvature_thr
    grad_thr = settings.refine_gradient_ratio
    ig2_min = settings.refine_ig2_delta_min
    dia_thr = settings.refine_delta_ia_thr

    # Group by Ug1 — snap to settings grid if provided
    ug1_map: Dict[float, List[Dict]] = {}
    for p in curve_points:
        key = _snap_ug1_key(p["ug1"], ug1_values)
        ug1_map.setdefault(key, []).append(p)

    result: Dict[float, set] = {}
    for ug1_key, curve in ug1_map.items():
        intervals = _analyse_curve_intervals(
            curve, ua_values, onset, curv_thr, grad_thr,
            ig2_min, dia_thr, is_pentode_sweep,
        )
        if intervals:
            result[ug1_key] = intervals
    return result


def _build_refine_ua(
    ua_values: List[float],
    refine_indices: set,
    min_step: float,
    max_depth: int,
) -> List[float]:
    """Generate new Ua midpoints by recursively bisecting marked intervals.

    Each interval ``[ua_values[i], ua_values[i+1]]`` where *i* is in
    *refine_indices* is bisected up to *max_depth* times.  Bisection stops
    when the sub-interval would be narrower than *min_step* or when
    ``round()`` produces a duplicate.

    Returns a **sorted list** of new Ua values (excluding original grid).
    """
    if not refine_indices or max_depth < 1:
        return []

    original = set(ua_values)
    new_values: set = set()

    def _bisect(lo: float, hi: float, depth: int) -> None:
        if depth <= 0:
            return
        if (hi - lo) < min_step * 2:
            return  # sub-intervals would be < min_step
        mid = round((lo + hi) / 2)
        if mid <= lo or mid >= hi:
            return  # duplicate after rounding
        if mid in original or mid in new_values:
            return
        new_values.add(float(mid))
        _bisect(lo, mid, depth - 1)
        _bisect(mid, hi, depth - 1)

    for i in sorted(refine_indices):
        if 0 <= i < len(ua_values) - 1:
            _bisect(ua_values[i], ua_values[i + 1], max_depth)

    return sorted(new_values)


def _refine_curve_inline(
    curve_points: List[Dict],
    ua_values: List[float],
    settings: ScanSettings,
    settle_ua: Callable,
    settle_ug2: Optional[Callable],
    read_point: Callable,
    progress: Optional[Callable[[ScanProgress], None]],
    stop: Optional[Callable[[], bool]],
    pa_limit: float,
    pg2_limit: float,
    min_safe_ua: float = 0.0,
    ig2_limit: float = 0.0,
) -> List[Dict]:
    """Analyse a just-measured Ua curve and immediately refine if needed.

    Called right after the coarse Ua sweep for a single (Ug1[, Ug2]) curve,
    while Ug1 (and Ug2) are still settled.  Returns the list of new refine
    points (already appended to the caller's points list via progress).
    """
    if not settings.refine_enabled or len(curve_points) < 3:
        return []
    is_pentode_sweep = not settings.is_triode and not settings.ug2_track_ua
    intervals = _analyse_curve_intervals(
        curve_points, ua_values,
        settings.refine_onset_ma, settings.refine_curvature_thr,
        settings.refine_gradient_ratio, settings.refine_ig2_delta_min,
        settings.refine_delta_ia_thr, is_pentode_sweep,
    )
    if not intervals:
        return []
    refine_ua = _build_refine_ua(
        ua_values, intervals,
        settings.refine_min_step_ua, settings.refine_max_depth,
    )
    if min_safe_ua > 0:
        refine_ua = [u for u in refine_ua if u >= min_safe_ua]
    if not refine_ua:
        return []

    if progress:
        progress({"event": "refine_count", "count": len(refine_ua)})

    new_points: List[Dict] = []
    # In Ug2-track mode the screen follows Ua; the device currently sits at the
    # last coarse point's Ua (the coarse sweep is ascending → that is the
    # highest Ua of the curve). Order each refine step by direction so the tube
    # never sees a low Ua under a still-high Ug2 — that transient is the
    # dangerous Ig2 spike: dropping Ua → lower Ug2 FIRST; raising Ua → Ua first.
    track = bool(settings.ug2_track_ua and settle_ug2)
    cur_ua = curve_points[-1]["ua"]
    for ua in refine_ua:
        if stop and stop():
            break
        try:
            if track:
                ug2_r = max(0, ua + settings.ug2_offset)
                if ua < cur_ua:
                    settle_ug2(ug2_r)
                    settle_ua(ua)
                else:
                    settle_ua(ua)
                    settle_ug2(ug2_r)
            else:
                settle_ua(ua)
        except _SkipPoint:
            continue
        cur_ua = ua
        point = read_point()
        if point is None:
            continue
        if _exceeds_ig2(point, ig2_limit):
            continue
        # Pg2 power check in BOTH modes (ML-127): point["ug2"] is the
        # measured screen voltage, valid for track AND independent sweeps.
        # The old track-only gate left independent refine midpoints
        # unchecked — inside the Ig2 current limit but over the Pg2 power
        # limit at high Ug2. (True triode: ug2=0 → never exceeds.)
        if _exceeds_pg2(point, pg2_limit, point["ug2"]):
            continue
        new_points.append(point)
        if progress:
            progress(point)
        if _exceeds_pa(point, pa_limit):
            break
    return new_points
