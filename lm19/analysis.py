from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from lm19.constants import (
    EPS, UA_ROUND, UG1_CLUSTER_THR, UG2_ZONE_TOLERANCE, UG2_ZONE_DEFAULT,
    EXPAND_MARGIN_UG1, EXPAND_MARGIN_UA, EXPAND_GAP,
    MIN_UG1_SPREAD, MIN_UA_SPREAD,
)

log = logging.getLogger(__name__)

# Minimum |dIa/dUa| slope treated as a real anode characteristic; below
# this the regression is degenerate. Shared by every R computation path.
_R_SLOPE_EPS = 1e-12


def _r_from_slope(slope: Optional[float], context: str) -> Optional[float]:
    """R (kΩ) from a dIa/dUa slope with a UNIFIED sign policy (ML-098).

    Physical Rp is positive; a negative slope is measurement noise. The
    four R paths used four different policies — one returned a negative R
    silently. Degenerate/negative slopes now yield None, negatives loudly.
    """
    if slope is None or abs(slope) <= _R_SLOPE_EPS:
        return None
    if slope < 0:
        log.warning("%s: negative dIa/dUa slope (%.3g) — noisy data, "
                    "R not computed", context, slope)
        return None
    return 1.0 / slope


@dataclass
class Zone:
    ua_min: float
    ua_max: float
    ug1_min: float
    ug1_max: float
    ug2: float = UG2_ZONE_DEFAULT
    ug2_tolerance: float = UG2_ZONE_TOLERANCE
    is_triode: bool = False
    ug2_track_ua: bool = False
    ug2_offset: float = 0.0


def _linear_regression(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def _in_zone(p: dict, zone: Zone) -> bool:
    if not (zone.ua_min <= p["ua"] <= zone.ua_max
            and zone.ug1_min <= p["ug1"] <= zone.ug1_max):
        return False
    if zone.is_triode:
        # True triode: no screen grid, skip Ug2 check entirely
        return True
    ug2 = p.get("ug2", 0.0)
    if zone.ug2_track_ua:
        # Triode-connected pentode: Ug2 should be ≈ Ua + offset
        return abs(ug2 - (p["ua"] + zone.ug2_offset)) <= zone.ug2_tolerance
    return abs(ug2 - zone.ug2) <= zone.ug2_tolerance


def count_in_zone(points: Iterable[dict], zone: Zone) -> int:
    return sum(1 for p in points if _in_zone(p, zone))


def compute_s(points: Iterable[dict], zone: Zone) -> Optional[float]:
    xs: List[float] = []
    ys: List[float] = []
    for p in points:
        if _in_zone(p, zone):
            xs.append(p["ug1"])
            ys.append(p["ia"])
    return _linear_regression(xs, ys)


def compute_r(points: Iterable[dict], zone: Zone) -> Optional[float]:
    xs: List[float] = []
    ys: List[float] = []
    for p in points:
        if _in_zone(p, zone):
            xs.append(p["ua"])
            ys.append(p["ia"])
    slope = _linear_regression(xs, ys)  # dIa/dUa = 1/R
    return _r_from_slope(slope, "compute_r")


def _in_zone_relaxed(p: dict, zone: Zone, *, skip_ug1: bool = False, skip_ua: bool = False) -> bool:
    """Like _in_zone but can skip Ug1 or Ua range checks."""
    if not skip_ua and not (zone.ua_min <= p["ua"] <= zone.ua_max):
        return False
    if not skip_ug1 and not (zone.ug1_min <= p["ug1"] <= zone.ug1_max):
        return False
    if zone.is_triode:
        return True
    ug2 = p.get("ug2", 0.0)
    if zone.ug2_track_ua:
        return abs(ug2 - (p["ua"] + zone.ug2_offset)) <= zone.ug2_tolerance
    return abs(ug2 - zone.ug2) <= zone.ug2_tolerance


def _try_expand_axis(
    points_list: List[dict], zone: Zone, axis: str,
    current_spread: float, min_spread: float,
) -> Optional[Zone]:
    """Expand zone along *axis* ('ug1' or 'ua') to include neighboring levels.

    Returns expanded Zone only if the original zone lies strictly between
    existing data levels (interpolation, not extrapolation).
    """
    if current_spread >= min_spread:
        return None

    skip_ug1 = (axis == "ug1")
    skip_ua = (axis == "ua")
    lo = zone.ug1_min if axis == "ug1" else zone.ua_min
    hi = zone.ug1_max if axis == "ug1" else zone.ua_max
    margin = EXPAND_MARGIN_UG1 if axis == "ug1" else EXPAND_MARGIN_UA

    candidates: set = set()
    for p in points_list:
        if _in_zone_relaxed(p, zone, skip_ug1=skip_ug1, skip_ua=skip_ua):
            candidates.add(round(p[axis], UA_ROUND))

    below = sorted(v for v in candidates if v < lo - EXPAND_GAP)
    above = sorted(v for v in candidates if v > hi + EXPAND_GAP)

    if not below or not above:
        return None

    new_lo = below[-1] - margin
    new_hi = above[0] + margin

    if axis == "ug1":
        return Zone(ua_min=zone.ua_min, ua_max=zone.ua_max,
                    ug1_min=new_lo, ug1_max=new_hi,
                    ug2=zone.ug2, ug2_tolerance=zone.ug2_tolerance,
                    is_triode=zone.is_triode,
                    ug2_track_ua=zone.ug2_track_ua, ug2_offset=zone.ug2_offset)
    return Zone(ua_min=new_lo, ua_max=new_hi,
                ug1_min=zone.ug1_min, ug1_max=zone.ug1_max,
                ug2=zone.ug2, ug2_tolerance=zone.ug2_tolerance,
                is_triode=zone.is_triode,
                ug2_track_ua=zone.ug2_track_ua, ug2_offset=zone.ug2_offset)


def _cluster_values(values: List[float], threshold: float) -> List[float]:
    """Group nearby values into clusters, return sorted cluster centers."""
    if not values:
        return []
    vals = sorted(values)
    clusters: List[List[float]] = [[vals[0]]]
    for v in vals[1:]:
        if v - clusters[-1][-1] <= threshold:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


_R_MIN_PTS = 3          # minimum Ua points per Ug1 level for R regression
_KNEE_SLOPE_FACTOR = 3  # local slope > factor × zone slope → knee region


def _expand_for_r_level(
    level_pts: List[dict],
    ua_min: float, ua_max: float,
    min_pts: int = _R_MIN_PTS,
) -> List[dict]:
    """Return points for R regression at one Ug1 level.

    Start with points inside [ua_min, ua_max].  If fewer than *min_pts*,
    expand — first upward (safe), then downward (stop before the knee).
    Knee is detected by a sharp increase in local dIa/dUa slope.
    """
    pts = sorted(level_pts, key=lambda p: p["ua"])
    in_zone = [p for p in pts if ua_min <= p["ua"] <= ua_max]
    if len(in_zone) >= min_pts:
        return in_zone

    above = [p for p in pts if p["ua"] > ua_max]       # ascending Ua
    below = [p for p in pts if p["ua"] < ua_min][::-1]  # descending Ua

    result = list(in_zone)

    # ── expand upward (safe — above zone, away from knee) ────────────
    for p in above:
        if len(result) >= min_pts:
            break
        result.append(p)

    if len(result) >= min_pts:
        return result

    # ── expand downward (careful — detect knee) ──────────────────────
    # Reference slope from points collected so far
    ref_slope: Optional[float] = None
    if len(result) >= 2:
        rs = sorted(result, key=lambda p: p["ua"])
        dua = rs[-1]["ua"] - rs[0]["ua"]
        if dua > 0:
            ref_slope = (rs[-1]["ia"] - rs[0]["ia"]) / dua

    for p in below:
        if len(result) >= min_pts:
            break
        # Check local slope against the nearest point already in result
        nearest = min((r for r in result if r["ua"] > p["ua"]),
                      key=lambda r: r["ua"], default=None)
        if nearest is not None and ref_slope is not None:
            gap = nearest["ua"] - p["ua"]
            if gap > 0:
                local_slope = (nearest["ia"] - p["ia"]) / gap
                if abs(local_slope) > _KNEE_SLOPE_FACTOR * abs(ref_slope):
                    break  # entering knee region
        result.append(p)

    return result


def _compute_r_per_ug1_level(
    all_points: List[dict], zone: Zone,
    ug1_cluster_thr: float = UG1_CLUSTER_THR,
) -> Optional[float]:
    """Compute R (kOhm) as median of per-Ug1-level univariate slopes.

    For pentodes, Ia depends on Ug1 ~100x more than on Ua.  Mixing Ug1
    levels in a single regression introduces confounding that dominates
    the tiny Ua->Ia effect.  Computing R at each Ug1 level independently
    (like Tube Health does) avoids this.

    Uses zone [ua_min, ua_max] as primary range.  If a level has fewer
    than ``_R_MIN_PTS`` points in the zone, the range is expanded —
    first upward, then downward (with knee detection to avoid the steep
    low-Ua region).
    """
    # All points matching zone Ug1 and Ug2 (no Ua filter yet)
    all_matching: List[dict] = [
        p for p in all_points
        if _in_zone_relaxed(p, zone, skip_ua=True)
    ]
    if not all_matching:
        return None

    # Cluster by Ug1 level
    ug1_vals = [p["ug1"] for p in all_matching]
    ug1_nominals = _cluster_values(ug1_vals, ug1_cluster_thr)

    r_values: List[float] = []
    for nom in ug1_nominals:
        level_pts = [p for p in all_matching
                     if abs(p["ug1"] - nom) <= ug1_cluster_thr]
        selected = _expand_for_r_level(
            level_pts, zone.ua_min, zone.ua_max)
        if len(selected) < 2:
            continue
        slope = _linear_regression(
            [p["ua"] for p in selected], [p["ia"] for p in selected])
        if slope is not None and slope > 1e-12:
            r_values.append(1.0 / slope)

    if not r_values:
        return None

    r_values.sort()
    mid = len(r_values) // 2
    if len(r_values) % 2 == 0:
        return (r_values[mid - 1] + r_values[mid]) / 2
    return r_values[mid]


def compute_sr_zone(
    points: Iterable[dict], zone: Zone,
    min_ug1_spread: float = MIN_UG1_SPREAD,
    min_ua_spread: float = MIN_UA_SPREAD,
    auto_expand: bool = True,
) -> Tuple[Optional[float], Optional[float], bool]:
    """Compute S (mA/V) and R (kOhm) from scan data in the given zone.

    For triodes and triode-connected pentodes, uses multivariate regression
    (Ia = a + S*Ug1 + (1/R)*Ua) which works well because the Ug1 and Ua
    effects are comparable in magnitude.

    For independent-Ug2 pentodes, S and R are computed separately:
      - S via univariate regression Ia vs Ug1 (all zone points)
      - R via per-Ug1-level univariate regression Ia vs Ua, then median
    This avoids the confounding problem where the Ug1 effect (~100x larger
    than Ua) corrupts the multivariate R estimate on unbalanced grids.

    When *auto_expand* is True and the zone contains insufficient Ug1
    (or Ua) variation, neighboring data levels are included — but only
    when the zone lies strictly between existing levels (interpolation,
    not extrapolation).

    Returns (S, R, expanded).  S or R may be None when insufficient data.
    *expanded* is True when the zone was auto-widened.
    """
    points_list = list(points)

    ug1_list: List[float] = []
    ua_list: List[float] = []
    ia_list: List[float] = []
    for p in points_list:
        if _in_zone(p, zone):
            ug1_list.append(p["ug1"])
            ua_list.append(p["ua"])
            ia_list.append(p["ia"])

    n = len(ug1_list)
    is_pentode = not zone.is_triode and not zone.ug2_track_ua

    if n < 2 and not is_pentode:
        return None, None, False

    ug1_spread = max(ug1_list) - min(ug1_list) if n >= 2 else 0.0
    ua_spread = max(ua_list) - min(ua_list) if n >= 2 else 0.0
    has_ug1 = ug1_spread >= min_ug1_spread
    has_ua = ua_spread >= min_ua_spread

    expanded = False
    if auto_expand and (not has_ug1 or (not has_ua and not is_pentode)):
        exp_zone = zone
        if not has_ug1:
            z = _try_expand_axis(points_list, exp_zone, "ug1",
                                 ug1_spread, min_ug1_spread)
            if z is not None:
                exp_zone = z
        if not is_pentode and not has_ua:
            z = _try_expand_axis(points_list, exp_zone, "ua",
                                 ua_spread, min_ua_spread)
            if z is not None:
                exp_zone = z

        if exp_zone is not zone:
            ug1_list, ua_list, ia_list = [], [], []
            for p in points_list:
                if _in_zone(p, exp_zone):
                    ug1_list.append(p["ug1"])
                    ua_list.append(p["ua"])
                    ia_list.append(p["ia"])
            n = len(ug1_list)
            if n >= 2:
                ug1_spread = max(ug1_list) - min(ug1_list)
                ua_spread = max(ua_list) - min(ua_list)
                has_ug1 = ug1_spread >= min_ug1_spread
                has_ua = ua_spread >= min_ua_spread
                expanded = True

    # ── Pentode per-level path ────────────────────────────────────────
    # R is computed from per-Ug1-level regression with zone-aware
    # expansion (_expand_for_r_level), so it does not need zone points.
    if is_pentode:
        s: Optional[float] = None
        r: Optional[float] = None
        if has_ug1:
            s = _linear_regression(ug1_list, ia_list)
        r = _compute_r_per_ug1_level(points_list, zone)
        return s, r, expanded

    if n < 2:
        return None, None, expanded

    # ── Triode / triode-connected: multivariate regression ────────────
    ug1_mean = sum(ug1_list) / n
    ua_mean = sum(ua_list) / n
    ia_mean = sum(ia_list) / n
    ug1_c = [v - ug1_mean for v in ug1_list]
    ua_c = [v - ua_mean for v in ua_list]
    ia_c = [v - ia_mean for v in ia_list]

    var_ug1 = sum(u * u for u in ug1_c)
    var_ua = sum(u * u for u in ua_c)
    cov_ug1_ua = sum(a * b for a, b in zip(ug1_c, ua_c))
    cov_ug1_ia = sum(u * y for u, y in zip(ug1_c, ia_c))
    cov_ua_ia = sum(u * y for u, y in zip(ua_c, ia_c))

    s = None
    r = None

    det = var_ug1 * var_ua - cov_ug1_ua ** 2
    if has_ug1 and has_ua and n >= 3 and abs(det) > 1e-12:
        b_ug1 = (var_ua * cov_ug1_ia - cov_ug1_ua * cov_ua_ia) / det
        b_ua = (var_ug1 * cov_ua_ia - cov_ug1_ua * cov_ug1_ia) / det
        s = b_ug1
        r = _r_from_slope(b_ua, "compute_sr_zone")

    if s is None and has_ug1 and var_ug1 > 0:
        s = cov_ug1_ia / var_ug1
    if r is None and has_ua and var_ua > 0:
        slope = cov_ua_ia / var_ua
        r = _r_from_slope(slope, "compute_sr_zone fallback")

    return s, r, expanded


def compute_k(s: Optional[float], r: Optional[float]) -> Optional[float]:
    if s is None or r is None:
        return None
    return r * s


# ── Direct SRK from value lists (used by health check & anywhere) ────

def compute_srk_direct(
    s_ug1: List[float],
    s_ia: List[float],
    r_ua: List[float],
    r_ia: List[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute S, R, K from raw voltage/current lists.

    S (gm, mA/V) via linear regression of Ia vs Ug1.
    R (Rp, kΩ) via central difference ΔUa/ΔIa (assumes 2+ points).
    K = S × R.

    Central-difference quick test from "µTracer User Manual",
    see SOURCES_INDEX.md.
    """
    s = _linear_regression(s_ug1, s_ia)
    r = None
    if len(r_ua) >= 2 and len(r_ia) >= 2:
        d_ia = r_ia[-1] - r_ia[0]
        d_ua = r_ua[-1] - r_ua[0]
        if abs(d_ia) > EPS and abs(d_ua) > EPS:
            # identical to the old (r_ua span)/(d_ia) for consistent signs;
            # a negative slope now yields None + WARNING (ML-098)
            r = _r_from_slope(d_ia / d_ua, "compute_srk_direct")
    k = compute_k(s, r)
    return s, r, k


def compute_sg2_direct(
    ug2_vals: List[float],
    ia_vals: List[float],
) -> Optional[float]:
    """Screen-grid transconductance Sg2 = ΔIa / ΔUg2 (mA/V) via central difference."""
    if len(ug2_vals) < 2 or len(ia_vals) < 2:
        return None
    d_ug2 = ug2_vals[-1] - ug2_vals[0]
    if abs(d_ug2) < EPS:
        return None
    return (ia_vals[-1] - ia_vals[0]) / d_ug2


def compute_mu_g1g2(
    s: Optional[float],
    sg2: Optional[float],
) -> Optional[float]:
    """µ(g1→g2) = S / Sg2  —  screen-to-control-grid amplification factor."""
    if s is None or sg2 is None or abs(sg2) < EPS:
        return None
    return abs(s / sg2)


def estimate_sg2_uncertainty(
    sg2: Optional[float],
    delta_ug2: float,
    n_repeats: int = 5,
    sigma_ug2: float = 1.0,
    sigma_ia: float = 0.05,
) -> Optional[float]:
    """Relative uncertainty for Sg2 (central difference). Same formula as for S."""
    if sg2 is None or abs(sg2) < EPS:
        return None
    sigma_ia_eff = sigma_ia / math.sqrt(max(n_repeats, 1))
    sigma_delta_ia = math.sqrt(2) * sigma_ia_eff
    delta_ia_sg2 = abs(sg2 * 2 * delta_ug2)
    err_ia = sigma_delta_ia / delta_ia_sg2 if delta_ia_sg2 > EPS else 1.0
    err_ug2 = sigma_ug2 / delta_ug2
    return math.sqrt(err_ia ** 2 + err_ug2 ** 2)


# ═══════════════════════════════════════════════════════════════════════════
#  Data Source Selection
# ═══════════════════════════════════════════════════════════════════════════

def select_analysis_points(
    all_points: List[Dict],
    series_id: Optional[int] = None,
) -> List[Dict]:
    """Select points for analysis.

    Priority:
      1. Current scan (series_id == 0) -- if available and no specific series requested.
      2. Specific series (by series_id) -- if requested.
      3. All points -- fallback.
    """
    if series_id is not None:
        specific = [p for p in all_points if p.get("series_id") == series_id]
        if specific:
            return specific
    current = [p for p in all_points if p.get("series_id") == 0]
    if current:
        return current
    return list(all_points)


def get_available_series(
    all_points: List[Dict],
    series_labels: Optional[Dict[int, str]] = None,
) -> List[Dict]:
    """List available data sources for analysis.

    Returns list of {series_id, label, n_points, has_ug2_sweep}.
    First entry (series_id=None) is current scan if present.
    """
    if series_labels is None:
        series_labels = {}

    sources: List[Dict] = []
    seen: set = set()
    for p in all_points:
        sid = p.get("series_id", 0)
        if sid not in seen:
            seen.add(sid)
            sp = [x for x in all_points if x.get("series_id", 0) == sid]
            label = series_labels.get(sid, "Current scan" if sid == 0 else f"Series {sid + 1}")
            sources.append({
                "series_id": sid,
                "label": label,
                "n_points": len(sp),
                "has_ug2_sweep": len({round(x.get("ug2", 0), 1) for x in sp}) > 1,
            })
    sources.sort(key=lambda s: s["series_id"])
    return sources


# ═══════════════════════════════════════════════════════════════════════════
#  SRK Uncertainty
# ═══════════════════════════════════════════════════════════════════════════

def estimate_srk_uncertainty(
    s: Optional[float],
    r: Optional[float],
    delta_ua: float,
    delta_ug1: float,
    n_repeats: int = 5,
    sigma_ua: float = 1.0,
    sigma_ug1: float = 0.04,
    sigma_ia: float = 0.05,
) -> Dict[str, Optional[float]]:
    """Estimate relative uncertainty (0..1) for S, R, K.

    Accuracy bands cross-checked against "Gamma Electronics —
    Transconductance Tube Tester Test Standard", see SOURCES_INDEX.md.

    Uses error propagation on central-difference formulas:
      S = ΔIa / (2·δVg)   →  σ(S)/S = √((σ_ΔIa/ΔIa)² + (σ_Vg/δVg)²)
      R = 2·δVa / ΔIa     →  σ(R)/R = √((σ_ΔIa/ΔIa)² + (σ_Va/δVa)²)
      K = S·R              →  σ(K)/K = √((σ(S)/S)² + (σ(R)/R)²)

    Args:
        sigma_ua:  single-reading Ua resolution (V), default UA_RESOLUTION_V
        sigma_ug1: single-reading Ug1 resolution (V), default UG1_RESOLUTION_V
        sigma_ia:  single-reading Ia noise (mA), default IA_NOISE_200MA
    """
    sigma_ia_eff = sigma_ia / math.sqrt(max(n_repeats, 1))
    sigma_delta_ia = math.sqrt(2) * sigma_ia_eff

    s_rel = None
    if s is not None and abs(s) > EPS:
        delta_ia_s = abs(s * 2 * delta_ug1)
        err_ia = sigma_delta_ia / delta_ia_s if delta_ia_s > EPS else 1.0
        err_vg = sigma_ug1 / delta_ug1
        s_rel = math.sqrt(err_ia ** 2 + err_vg ** 2)

    r_rel = None
    if r is not None and abs(r) > EPS:
        delta_ia_r = abs(2 * delta_ua / r)
        err_ia = sigma_delta_ia / delta_ia_r if delta_ia_r > EPS else 1.0
        err_va = sigma_ua / delta_ua
        r_rel = math.sqrt(err_ia ** 2 + err_va ** 2)

    k_rel = None
    if s_rel is not None and r_rel is not None:
        k_rel = math.sqrt(s_rel ** 2 + r_rel ** 2)

    return {"s_rel": s_rel, "r_rel": r_rel, "k_rel": k_rel}
