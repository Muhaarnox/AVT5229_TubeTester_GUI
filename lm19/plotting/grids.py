"""Pure computation for measurement grids: Ug2 filtering, Ia/Gm/Rp/mu/Pa grids.

All functions are free of Qt/pyqtgraph dependencies and fully unit-testable.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import List, Dict, Optional, Tuple, Callable

from lm19.constants import (
    UG1_CLUSTER_THR, UG2_CLUSTER_THR, UA_CLUSTER_THR,
    UA_ROUND, UG1_ROUND, UG2_ROUND, EPS_COARSE,
)


def fill_nan_nearest(grid: np.ndarray) -> np.ndarray:
    """Replace NaN cells with the value of the nearest non-NaN neighbour.

    Uses Euclidean distance transform to find the closest valid cell.
    Returns a copy; the original is not modified.
    If the grid has no NaN or is all NaN, returns as-is (copy).
    """
    mask = np.isnan(grid)
    if not mask.any() or mask.all():
        return grid.copy()
    ind = distance_transform_edt(mask, return_distances=False,
                                 return_indices=True)
    filled = grid.copy()
    filled[mask] = grid[tuple(ind)][:][mask]
    return filled


# ── Sparse suppression ─────────────────────────────────────────────
# ML-065: with the old value 1 ("fewer than 1 valid" = already all-NaN)
# every production call was a no-op. 2 implements the documented intent:
# suppress rows/cols whose single isolated point would smear into a band.
MIN_SPARSE_VALID = 2  # rows/cols with fewer valid cells are suppressed


def suppress_sparse(grid: np.ndarray,
                    min_valid: int = MIN_SPARSE_VALID) -> np.ndarray:
    """Set rows and columns with fewer than *min_valid* non-NaN cells to NaN.

    Prevents isolated single-point rows/columns from creating misleading
    visual artefacts when pyqtgraph scales a small heatmap image.
    Returns a copy.
    """
    out = grid.copy()
    # Suppress sparse rows
    for i in range(out.shape[0]):
        if np.count_nonzero(~np.isnan(out[i])) < min_valid:
            out[i, :] = np.nan
    # Suppress sparse columns
    for j in range(out.shape[1]):
        if np.count_nonzero(~np.isnan(out[:, j])) < min_valid:
            out[:, j] = np.nan
    return out


def cluster_nominal(values: list, threshold: float = 0.02) -> list:
    """Merge nearby values so one nominal per setpoint."""
    if not values:
        return []
    vals = sorted(values)
    out = [vals[0]]
    for v in vals[1:]:
        if v - out[-1] > threshold:
            out.append(v)
    return out


def nominal_key(val: float, nominals: list) -> float:
    """Return the nominal value nearest to *val*."""
    if not nominals:
        return val
    return min(nominals, key=lambda n: abs(val - n))


def filter_ug2_slice(
    points: List[Dict],
    is_triode: bool,
    select_ug2_slice: Optional[Callable] = None,
    ug2_cluster_thr: float = UG2_CLUSTER_THR,
) -> List[Dict]:
    """Filter points to a single Ug2 slice.

    For triodes returns all points unchanged.
    For pentodes uses *select_ug2_slice* to pick the target Ug2.
    """
    if not points:
        return []
    if is_triode:
        return list(points)
    if select_ug2_slice is None:
        return list(points)
    ug2_target = select_ug2_slice(points)
    ug2_raw = sorted({round(p.get("ug2", 0.0), UG2_ROUND) for p in points})
    ug2_noms = cluster_nominal(ug2_raw, threshold=ug2_cluster_thr)
    target_nom = nominal_key(round(ug2_target, UG2_ROUND), ug2_noms)
    return [p for p in points
            if nominal_key(round(p.get("ug2", 0.0), UG2_ROUND), ug2_noms) == target_nom]


def filter_ug2_multi(
    points: List[Dict],
    is_triode: bool,
    ug2_targets: Optional[List[float]] = None,
    ug2_cluster_thr: float = UG2_CLUSTER_THR,
) -> Dict[float, List[Dict]]:
    """Split points into subsets per Ug2 value.

    Returns ``{ug2_nominal: [points...]}``.
    For triodes returns ``{0.0: all_points}``.
    If *ug2_targets* is ``None`` or empty, returns all available Ug2 groups.
    """
    if not points:
        return {}
    if is_triode:
        return {0.0: list(points)}

    ug2_raw = sorted({round(p.get("ug2", 0.0), UG2_ROUND) for p in points})
    ug2_noms = cluster_nominal(ug2_raw, threshold=ug2_cluster_thr)
    if not ug2_noms:
        return {0.0: list(points)}

    if ug2_targets:
        wanted = {nominal_key(round(t, UG2_ROUND), ug2_noms) for t in ug2_targets}
    else:
        wanted = set(ug2_noms)

    result: Dict[float, List[Dict]] = {}
    for p in points:
        nom = nominal_key(round(p.get("ug2", 0.0), UG2_ROUND), ug2_noms)
        if nom in wanted:
            result.setdefault(nom, []).append(p)
    return result


def build_ia_grid(
    subset: List[Dict],
    ug1_cluster_thr: float = UG1_CLUSTER_THR,
    ua_cluster_thr: float = UA_CLUSTER_THR,
) -> Optional[Dict]:
    """Build an Ia grid from a Ug2-filtered point subset.

    Returns ``{"ua_vals", "ug1_vals", "ia_grid"}`` or ``None``
    if insufficient data.  *ia_grid* shape is ``(n_ug1, n_ua)``.
    """
    if len(subset) < 4:
        return None

    ua_raw = sorted({round(p["ua"], UA_ROUND) for p in subset})
    ug1_raw = sorted({round(p["ug1"], UG1_ROUND) for p in subset})
    ua_vals = cluster_nominal(ua_raw, threshold=ua_cluster_thr)
    ug1_vals = cluster_nominal(ug1_raw, threshold=ug1_cluster_thr)

    if len(ua_vals) < 2 or len(ug1_vals) < 2:
        return None

    ua_idx = {v: i for i, v in enumerate(ua_vals)}
    ug1_idx = {v: i for i, v in enumerate(ug1_vals)}
    ia_grid = np.full((len(ug1_vals), len(ua_vals)), np.nan)
    for p in subset:
        i = ug1_idx[nominal_key(round(p["ug1"], UG1_ROUND), ug1_vals)]
        j = ua_idx[nominal_key(round(p["ua"], UA_ROUND), ua_vals)]
        ia_grid[i, j] = p["ia"]

    return {"ua_vals": ua_vals, "ug1_vals": ug1_vals, "ia_grid": ia_grid}


def build_ia_grid_averaged(
    subset: List[Dict],
    ug1_cluster_thr: float = UG1_CLUSTER_THR,
    ua_cluster_thr: float = UA_CLUSTER_THR,
) -> Optional[Dict]:
    """Like :func:`build_ia_grid` but averages duplicate entries.

    Used by contour rendering where multiple points may map to the same cell.
    Returns ``{"ua_vals", "ug1_vals", "ia_grid", "ug2"}`` or ``None``.
    """
    if not subset:
        return None
    ua_raw = sorted({p["ua"] for p in subset})
    ug1_raw = sorted({p["ug1"] for p in subset})
    ua_vals = cluster_nominal(ua_raw, threshold=ua_cluster_thr)
    ug1_vals = cluster_nominal(ug1_raw, threshold=ug1_cluster_thr)
    if len(ua_vals) < 2 or len(ug1_vals) < 2:
        return None
    ua_idx = {v: i for i, v in enumerate(ua_vals)}
    ug1_idx = {v: i for i, v in enumerate(ug1_vals)}
    z = np.full((len(ug1_vals), len(ua_vals)), np.nan, dtype=float)
    counts = np.zeros_like(z)
    for p in subset:
        i = ug1_idx[nominal_key(p["ug1"], ug1_vals)]
        j = ua_idx[nominal_key(p["ua"], ua_vals)]
        if np.isnan(z[i, j]):
            z[i, j] = p["ia"]
        else:
            z[i, j] += p["ia"]
        counts[i, j] += 1
    counts[counts == 0] = 1
    z = z / counts
    return {"ua_vals": ua_vals, "ug1_vals": ug1_vals, "ia_grid": z}


def compute_gm_rp_grids(
    ua_vals: list,
    ug1_vals: list,
    ia_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Gm and Rp grids from an Ia grid via finite differences.

    Returns ``(gm_grid, rp_grid)`` where:

    * ``gm_grid`` shape ``(n_ug1-1, n_ua)`` — Gm = |dIa/dUg1| in mA/V
    * ``rp_grid`` shape ``(n_ug1, n_ua-1)`` — Rp = |1/(dIa/dUa)| in kΩ
    """
    n_ug1, n_ua = len(ug1_vals), len(ua_vals)

    gm_grid = np.full((n_ug1 - 1, n_ua), np.nan)
    for j in range(n_ua):
        for i in range(n_ug1 - 1):
            ia0, ia1 = ia_grid[i, j], ia_grid[i + 1, j]
            dug1 = ug1_vals[i + 1] - ug1_vals[i]
            if not np.isnan(ia0) and not np.isnan(ia1) and abs(dug1) > 0.01:
                gm_grid[i, j] = abs((ia1 - ia0) / dug1)

    rp_grid = np.full((n_ug1, n_ua - 1), np.nan)
    for i in range(n_ug1):
        for j in range(n_ua - 1):
            ia0, ia1 = ia_grid[i, j], ia_grid[i, j + 1]
            dua = ua_vals[j + 1] - ua_vals[j]
            if not np.isnan(ia0) and not np.isnan(ia1) and abs(dua) > 0.01:
                slope = (ia1 - ia0) / dua
                if abs(slope) > EPS_COARSE:
                    rp_grid[i, j] = abs(1.0 / slope)

    return gm_grid, rp_grid


def compute_mu_grid(gm_grid: np.ndarray, rp_grid: np.ndarray) -> np.ndarray:
    """Compute mu = Gm * Rp on the intersection of Gm and Rp grids.

    Gm shape ``(n_ug1-1, n_ua)``, Rp shape ``(n_ug1, n_ua-1)``.
    Returns mu shape ``(n_ug1-1, n_ua-1)`` — averaged Gm at adjacent Ua
    columns multiplied by Rp at corresponding rows.
    """
    n_ug1_m1 = gm_grid.shape[0]
    n_ua_m1 = rp_grid.shape[1]
    mu_grid = np.full((n_ug1_m1, n_ua_m1), np.nan)
    for i in range(n_ug1_m1):
        for j in range(n_ua_m1):
            gm_left = gm_grid[i, j]
            gm_right = gm_grid[i, j + 1] if j + 1 < gm_grid.shape[1] else np.nan
            rp_val = rp_grid[i, j] if i < rp_grid.shape[0] else np.nan
            if not np.isnan(gm_left) and not np.isnan(gm_right) and not np.isnan(rp_val):
                mu_grid[i, j] = (gm_left + gm_right) / 2.0 * rp_val
    return mu_grid


def build_pa_grid(
    subset: List[Dict],
    ug1_cluster_thr: float = UG1_CLUSTER_THR,
    ua_cluster_thr: float = UA_CLUSTER_THR,
) -> Optional[Dict]:
    """Build a Pa = Ua*Ia/1000 grid (watts).

    Returns ``{"ua_vals", "ug1_vals", "pa_grid"}`` or ``None``.
    """
    if len(subset) < 4:
        return None

    ua_raw = sorted({round(p["ua"], UA_ROUND) for p in subset})
    ug1_raw = sorted({round(p["ug1"], UG1_ROUND) for p in subset})
    ua_vals = cluster_nominal(ua_raw, threshold=ua_cluster_thr)
    ug1_vals = cluster_nominal(ug1_raw, threshold=ug1_cluster_thr)

    if len(ua_vals) < 2 or len(ug1_vals) < 2:
        return None

    ua_idx = {v: i for i, v in enumerate(ua_vals)}
    ug1_idx = {v: i for i, v in enumerate(ug1_vals)}
    pa_grid = np.full((len(ug1_vals), len(ua_vals)), np.nan)

    for p in subset:
        i = ug1_idx[nominal_key(round(p["ug1"], UG1_ROUND), ug1_vals)]
        j = ua_idx[nominal_key(round(p["ua"], UA_ROUND), ua_vals)]
        pa_grid[i, j] = p["ua"] * p["ia"] / 1000.0

    return {"ua_vals": ua_vals, "ug1_vals": ug1_vals, "pa_grid": pa_grid}
