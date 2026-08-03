"""Bilinear interpolation for heatmap grids — pure logic, no Qt.

The interactive ``HeatmapMarker`` widget that consumes this function
lives in ``app/heatmap_marker.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def interp_bilinear(z_grid: np.ndarray, x_vals: np.ndarray,
                    y_vals: np.ndarray, mx: float, my: float
                    ) -> Optional[float]:
    """Bilinear interpolation on a 2D grid.

    Args:
        z_grid: 2D array, shape (len(y_vals), len(x_vals)).
                z_grid[row, col] where row corresponds to y_vals,
                col corresponds to x_vals.
        x_vals: sorted 1D array of X-axis values (columns).
        y_vals: sorted 1D array of Y-axis values (rows).
        mx: query X coordinate.
        my: query Y coordinate.

    Returns:
        Interpolated value, or None if out of range or insufficient data.
    """
    if x_vals is None or y_vals is None or z_grid is None:
        return None
    nx = len(x_vals)
    ny = len(y_vals)
    if nx < 2 or ny < 2:
        return None
    if z_grid.shape != (ny, nx):
        return None

    # Check bounds
    if mx < x_vals[0] or mx > x_vals[-1]:
        return None
    if my < y_vals[0] or my > y_vals[-1]:
        return None

    # Find cell indices
    # ix: index such that x_vals[ix] <= mx <= x_vals[ix+1]
    ix = int(np.searchsorted(x_vals, mx, side='right')) - 1
    iy = int(np.searchsorted(y_vals, my, side='right')) - 1

    # Clamp to valid range (handle mx == x_vals[-1] edge)
    ix = min(ix, nx - 2)
    iy = min(iy, ny - 2)

    x0, x1 = x_vals[ix], x_vals[ix + 1]
    y0, y1 = y_vals[iy], y_vals[iy + 1]

    # Four corner values
    q00 = z_grid[iy, ix]
    q10 = z_grid[iy, ix + 1]
    q01 = z_grid[iy + 1, ix]
    q11 = z_grid[iy + 1, ix + 1]

    # If any corner is NaN, try nearest non-NaN
    corners = np.array([q00, q10, q01, q11])
    if np.any(np.isnan(corners)):
        valid = corners[~np.isnan(corners)]
        if len(valid) == 0:
            return None
        # Fallback: nearest valid corner
        dx = (mx - x0) / max(x1 - x0, 1e-12)
        dy = (my - y0) / max(y1 - y0, 1e-12)
        # ML-141: distance from the query point (dx, dy) to each corner —
        # the old formulas were inverted (q00 used the distance to q11
        # etc.), so the fallback picked the FARTHEST valid corner.
        # q00 sits at (0, 0), q10 at (1, 0), q01 at (0, 1), q11 at (1, 1).
        dists = [
            dx ** 2 + dy ** 2,                # q00: bottom-left  (0, 0)
            (1 - dx) ** 2 + dy ** 2,          # q10: bottom-right (1, 0)
            dx ** 2 + (1 - dy) ** 2,          # q01: top-left     (0, 1)
            (1 - dx) ** 2 + (1 - dy) ** 2,    # q11: top-right    (1, 1)
        ]
        best_val = None
        best_dist = float('inf')
        for i, (c, d) in enumerate(zip(corners, dists)):
            if not np.isnan(c) and d < best_dist:
                best_dist = d
                best_val = float(c)
        return best_val

    # Bilinear interpolation
    dx = (mx - x0) / (x1 - x0)
    dy = (my - y0) / (y1 - y0)

    val = (q00 * (1 - dx) * (1 - dy) +
           q10 * dx * (1 - dy) +
           q01 * (1 - dx) * dy +
           q11 * dx * dy)

    return float(val)
