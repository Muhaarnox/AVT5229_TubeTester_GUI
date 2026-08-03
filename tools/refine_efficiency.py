#!/usr/bin/env python
"""Benchmark adaptive refinement efficiency on synthetic pentode data.

Runs scans at several Ua step sizes with and without refine, measures:
  - total points collected
  - approximation error against a fine-step reference (step=1V)
  - refine ratio (refine_pts / grid_pts)

Usage:
    cd d:\\work\\AVR\\AVT5229\\lm19_app && py tools/refine_efficiency.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from scan_test_helpers import (  # noqa: E402
    _make_mock_client, _make_scan_settings, ScanRange, run_scan,
)


# ── Synthetic pentode model ──────────────────────────────────────────────

def tube_ia_ma(ua: float, ug1: float, ug2: float) -> float:
    """Simple monotonic pentode: linear rise, knee near Ua=Ug2, saturation above.

    Uses softer knee than a hard min() to give a curve with interesting
    curvature near Ua ≈ Ug2 — that's where refine should concentrate points.
    """
    if ug1 <= -10:
        return 0.0
    ug1_factor = (ug1 + 10) / 8.0                 # 0..1 as Ug1 opens
    # Soft knee: 1 - exp(-3 Ua/Ug2)
    ua_factor = 1.0 - np.exp(-3.0 * ua / ug2) if ug2 > 0 else 0.0
    return float(ug1_factor * ua_factor * 100)     # 0..100 mA


def make_tube_client():
    """Mock client implementing the synthetic tube model."""
    client = _make_mock_client()
    state = {"Ua": 0.0, "Ug1": 0.0, "Ug2": 0.0}

    def cap_set(name, value, delay=0.05):
        state[name] = float(value)

    def get_s(name, real=False):
        if name == "Er":
            return 0
        if name in ("Ua", "Ug2") and real:
            return state.get(name, 0)
        if name == "Ug1" and real:
            return state.get("Ug1", 0)
        if name == "Ia" and real:
            ug1 = -state.get("Ug1", 0) / 100.0
            ia = tube_ia_ma(state.get("Ua", 0), ug1, state.get("Ug2", 0))
            return int(round(ia / 0.01))
        if name == "Ig2" and real:
            return 0
        return 100

    client.set_param = MagicMock(side_effect=cap_set)
    client.get_param = MagicMock(side_effect=get_s)
    return client


# ── Benchmark ────────────────────────────────────────────────────────────

def run_scan_benchmark(step: float, refine: bool, ua_min=10, ua_max=300,
                       ug1_min=-8, ug1_max=-2, ug2=200):
    """Run a scan with given step and refine setting. Returns points list."""
    client = make_tube_client()
    settings = _make_scan_settings(
        ua=ScanRange(ua_min, ua_max, step),
        ug1=ScanRange(ug1_min, ug1_max, 2),
        ug2=ScanRange(ug2, ug2, 0),
        is_triode=False,
        ug2_track_ua=False,
        pa_max_w=0, pa_over_pct=0,         # no Pa limit
        refine_enabled=refine,
        refine_max_depth=2,
        refine_min_step_ua=max(1.0, step / 4.0),
        refine_onset_ma=0.5,
        refine_curvature_thr=0.1,
        refine_gradient_ratio=2.5,
        refine_ig2_delta_min=0.5,
        refine_delta_ia_thr=0.15,
    )
    # Patch the real time module: every `import time; time.sleep(...)` in
    # lm19/scan/{io,protection,runner}.py resolves to this mock at call
    # time, so all settle/retry delays are skipped. (The old
    # "lm19.scan.time.sleep" target crashed — the lm19.scan package never
    # imports time.)
    with patch("time.sleep"):
        points = run_scan(client, settings)
    return points


def approx_error(points, ua_grid, ug1, ug2):
    """Interpolate Ia from points at a dense ua_grid and compare to truth.

    Returns RMS error in mA.
    """
    curve = [p for p in points if abs(p["ug1"] - ug1) < 0.5
                              and abs(p["ug2"] - ug2) < 1.0]
    if len(curve) < 2:
        return float("inf")
    curve.sort(key=lambda p: p["ua"])
    xs = np.array([p["ua"] for p in curve])
    ys = np.array([p["ia"] for p in curve])

    errors = []
    for ua in ua_grid:
        if ua < xs[0] or ua > xs[-1]:
            continue  # don't extrapolate
        pred = np.interp(ua, xs, ys)
        truth = tube_ia_ma(ua, ug1, ug2)
        errors.append((pred - truth) ** 2)
    if not errors:
        return float("inf")
    return float(np.sqrt(np.mean(errors)))


def main():
    print("Adaptive refinement efficiency benchmark")
    print("=" * 78)
    print("Synthetic pentode: Ia = ug1_factor * (1 - exp(-3 Ua/Ug2)) * 100 mA")
    print("Ua 10..300, Ug1 -8..-2 step 2, Ug2=200")
    print()

    ug1_values = [-8, -6, -4, -2]
    ug2 = 200
    ua_dense = np.arange(10, 301, 1.0)   # reference grid

    header = f"{'step':>6} | {'refine':>7} | {'points':>7} | {'RMS err (mA)':>14} | {'ratio':>6}"
    print(header)
    print("-" * len(header))

    for step in [50, 25, 20, 10, 5]:
        for refine in [False, True]:
            points = run_scan_benchmark(step, refine, ug2=ug2)
            errs = [approx_error(points, ua_dense, ug1, ug2) for ug1 in ug1_values]
            valid = [e for e in errs if e != float("inf")]
            rms = float(np.mean(valid)) if valid else float("inf")

            # Baseline: grid only (approximate)
            if refine:
                grid = run_scan_benchmark(step, False, ug2=ug2)
                ratio = len(points) / max(len(grid), 1)
                ratio_str = f"{ratio:.2f}x"
            else:
                ratio_str = "-"

            flag = "refine" if refine else "grid"
            print(f"{step:>6.0f} | {flag:>7} | {len(points):>7d} | {rms:>14.3f} | {ratio_str:>6}")

    print()
    print("Interpretation:")
    print("  - RMS err: interpolation error against the ideal curve (mA)")
    print("  - ratio:   (refine pts) / (plain grid pts) at same step")
    print("  - refine wins when RMS err drops faster than point count grows")


if __name__ == "__main__":
    main()
