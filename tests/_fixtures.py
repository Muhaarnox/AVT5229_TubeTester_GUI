"""Shared synthetic point generators for tests.

These wrappers produce realistic Ia(Ua, Ug1) data for tube test scenarios
without needing actual hardware or fitted models. Used by tests that
exercise amplifier/optimizer pipelines.

For tests that need a specific shape (single Ug2 level / dead Ug2 levels /
sparse grids), keep their own local fixtures — these helpers cover only
the most common case (dense Ua×Ug1 grid with optional Ug2).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def make_triode_points(
    n_ug1: int = 21,
    n_ua: int = 20,
    ub: float = 250.0,  # noqa: ARG001 — accepted for caller symmetry; not used by the Ia formula
    series_id: int = 1,
) -> List[Dict]:
    """Triode-like Ia(Ua, Ug1) synthetic scan.

    21 Ug1 levels by default ensures Chebyshev gets enough intersection
    points within typical swing ranges. Ia formula:
    ``Ia = max(0, 1.5 * (Ug1 + 10.5) * (1 + Ua/500))``
    """
    pts: List[Dict] = []
    ug1_values = np.linspace(-10.0, -1.0, n_ug1)
    ua_values = np.linspace(10.0, 400.0, n_ua)
    for ug1 in ug1_values:
        x = ug1 + 10.5
        for ua in ua_values:
            ia = max(0.0, 1.5 * x * (1.0 + ua / 500.0))
            pts.append({
                "ug1": round(float(ug1), 1),
                "ua": round(float(ua), 1),
                "ia": round(ia, 4),
                "series_id": series_id,
            })
    return pts


def make_pentode_points(
    n_ug1: int = 11,
    n_ua: int = 20,
    ub: float = 250.0,
    ug2: float = 200.0,
    series_id: int = 1,
) -> List[Dict]:
    """Pentode-like points: triode shape + fixed Ug2 attached."""
    pts = make_triode_points(n_ug1, n_ua, ub, series_id)
    for p in pts:
        p["ug2"] = ug2
    return pts
