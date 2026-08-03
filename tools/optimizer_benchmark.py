#!/usr/bin/env python
"""Benchmark optimizer hot paths (before/after performance work).

Measures, on a synthetic EL84 Koren pentode:
  - single-eval costs: find_intersections_model, DFT SE, DFT PP, chebyshev
  - optimize_model grid at reduced steps (+ extrapolation to UI defaults)
  - optimize_measurements grid (no model)
  - one refine_optimum run at the auto-refine method (DFT when model given)
  - optimize_pp DFT grid at reduced steps (+ extrapolation)

Usage:
    cd d:\\work\\AVR\\AVT5229\\lm19_app && py tools/optimizer_benchmark.py [--json out.json]

UI-default grid: ub 8 x ug2 4 x ra 20 x ug1 20 = 12800 points. The
benchmark runs a reduced grid and reports the extrapolation factor so
before/after comparisons stay honest.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from lm19.amplifier import (
    PushPullLoadLine,
    ResistiveLoadLine,
    compute_distortion_chebyshev,
    compute_distortion_dft,
    compute_distortion_dft_pp,
    find_intersections_model,
)
from lm19.optimizer import (
    OptimizerConstraints,
    optimize_measurements,
    optimize_model,
    optimize_pp,
    refine_optimum,
)
from lm19.io_utils import write_json
from lm19.tube_sim import quick_pentode

# ── module local constants ──
# Reduced grid for the timed runs; extrapolation factor maps to UI defaults.
_RED_UB_STEPS = 2
_RED_UG2_STEPS = 2
_RED_RA_STEPS = 8
_RED_UG1_STEPS = 8
_DEFAULT_GRID = 8 * 4 * 20 * 20
_REDUCED_GRID = _RED_UB_STEPS * _RED_UG2_STEPS * _RED_RA_STEPS * _RED_UG1_STEPS
_EXTRAP = _DEFAULT_GRID / _REDUCED_GRID
_MICRO_REPS = 20
_UG1_CURVES = [-2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0]


def _timeit(fn, reps: int = _MICRO_REPS) -> float:
    """Median wall time of fn() over reps runs, seconds."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _constraints(**over) -> OptimizerConstraints:
    base = dict(
        target="min_thd",
        pa_max_w=12.0,
        ug1_range=(-12.0, -2.0),
        ra_range=(2.0, 10.0),
        ub_range=(200.0, 350.0),
        ug2_range=(150.0, 300.0),
        ub_steps=_RED_UB_STEPS,
        ug2_steps=_RED_UG2_STEPS,
        ra_steps=_RED_RA_STEPS,
        ug1_steps=_RED_UG1_STEPS,
    )
    base.update(over)
    return OptimizerConstraints(**base)


def main() -> None:
    out: dict = {}
    model, points = quick_pentode("EL84")
    print(f"EL84 Koren pentode, {len(points)} synthetic points, "
          f"K={len(_UG1_CURVES)} ug1 curves")
    print(f"Reduced grid {_REDUCED_GRID} pts, extrapolation x{_EXTRAP:.0f} "
          f"to UI default {_DEFAULT_GRID}")
    print("=" * 72)

    # ── micro: single evals ──────────────────────────────────────────
    ll = ResistiveLoadLine(300.0, 5.0)
    t = _timeit(lambda: find_intersections_model(
        model, ll, _UG1_CURVES, ug2=250.0, ua_range=(1.0, 400.0)))
    out["fim_ms"] = t * 1e3
    print(f"find_intersections_model (K=11):      {t*1e3:8.2f} ms")

    t = _timeit(lambda: compute_distortion_dft(
        model, ll, ug1_bias=-7.0, half_swing=4.0, ug2=250.0, ub=300.0))
    out["dft_se_ms"] = t * 1e3
    print(f"compute_distortion_dft (SE, n=1024):  {t*1e3:8.2f} ms")

    ll_pp = PushPullLoadLine(300.0, ra_aa=8.0)
    t = _timeit(lambda: compute_distortion_dft_pp(
        model, ll_pp, ug1_bias=-7.0, half_swing=4.0, ug2=250.0))
    out["dft_pp_ms"] = t * 1e3
    print(f"compute_distortion_dft_pp (n=1024):   {t*1e3:8.2f} ms")

    isects = find_intersections_model(
        model, ll, _UG1_CURVES, ug2=250.0, ua_range=(1.0, 400.0))
    t = _timeit(lambda: compute_distortion_chebyshev(
        isects, ug1_bias=-7.0, ub=300.0), reps=200)
    out["cheb_ms"] = t * 1e3
    print(f"compute_distortion_chebyshev (K=11):  {t*1e3:8.3f} ms")

    # ── optimize_model grid (auto -> chebyshev grid) ──────────────────
    c = _constraints(hd_method="auto")
    t0 = time.perf_counter()
    res = optimize_model(model, c, ug1_values=list(_UG1_CURVES))
    t_grid = time.perf_counter() - t0
    n_pts = len(res.grid_points)
    assert res.best is not None, "optimize_model returned no best point"
    out["model_grid_s"] = t_grid
    out["model_grid_extrap_s"] = t_grid * _EXTRAP
    print(f"optimize_model auto grid ({_REDUCED_GRID}p->{n_pts}): "
          f"{t_grid:7.2f} s  -> defaults ~{t_grid*_EXTRAP/60:5.1f} min")

    # ── refine (auto -> DFT with model) ───────────────────────────────
    t0 = time.perf_counter()
    refined = refine_optimum(res.best, points=None, model=model,
                             constraints=c, ug1_values=list(_UG1_CURVES))
    t_ref = time.perf_counter() - t0
    out["refine_one_s"] = t_ref
    out["refine_8_extrap_s"] = t_ref * 8
    print(f"refine_optimum x1 (auto->DFT):         {t_ref:7.2f} s  "
          f"-> 8 pareto pts ~{t_ref*8:6.1f} s   (refined={'ok' if refined else 'None'})")

    # ── optimize_measurements (no model, chebyshev) ──────────────────
    c_m = _constraints(hd_method="chebyshev", ug2_range=None)
    t0 = time.perf_counter()
    res_m = optimize_measurements(points, ub=300.0, constraints=c_m,
                                  ug2_values=[250.0])
    t_meas = time.perf_counter() - t0
    assert res_m.best is not None
    out["meas_grid_s"] = t_meas
    print(f"optimize_measurements chebyshev:      {t_meas:7.2f} s  "
          f"({len(res_m.grid_points)} pts)")

    # ── optimize_pp DFT + 2 UL taps ──────────────────────────────────
    c_pp = _constraints(
        hd_method="dft", circuit="pp", ug2_range=None,
        ra_range=(4.0, 12.0),
        ul_tap_mode="presets",
        ul_tap_presets=(0.0, 0.43),
        ul_tap_presets_enabled=(True, True),
    )
    pp_grid = _RED_UB_STEPS * _RED_RA_STEPS * _RED_UG1_STEPS * 2
    pp_default = 8 * 20 * 20 * 2
    t0 = time.perf_counter()
    res_pp = optimize_pp(points, ub=300.0, constraints=c_pp,
                         ug2_filter=250.0, model=model)
    t_pp = time.perf_counter() - t0
    out["pp_dft_grid_s"] = t_pp
    out["pp_dft_extrap_s"] = t_pp * pp_default / pp_grid
    print(f"optimize_pp DFT 2 taps ({pp_grid}p):      {t_pp:7.2f} s  "
          f"-> defaults ~{t_pp*pp_default/pp_grid/60:5.1f} min "
          f"({len(res_pp.grid_points)} pts, err={res_pp.error})")

    print("=" * 72)
    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        write_json(Path(path), out)
        print(f"saved -> {path}")


if __name__ == "__main__":
    main()
