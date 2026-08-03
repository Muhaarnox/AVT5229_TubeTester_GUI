"""Benchmark: speed vs physics of different HD methods on optimizer results.

Runs the optimizer with each hd_method (5point/chebyshev/auto/dft) on real
6P1P pentode data. Reports:

  • wall time per full optimizer run
  • best point THD from each method (sanity-bounded by the sparse-data guard)
  • cross-method agreement on top-3 grid candidates
  • verifies the sparse-data guard successfully blocks degenerate
    small-swing wins

Run:
    py tools/bench_optimizer_hd_methods.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

from lm19.amplifier import (
    PushPullLoadLine,
    ResistiveLoadLine,
    compute_distortion,
    compute_distortion_chebyshev,
    compute_distortion_dft,
    find_intersections,
    find_intersections_model,
)
from lm19.dempwolf import fit_dempwolf
from lm19.optimizer import (
    OptimizerConstraints,
    optimize_measurements,
)
from lm19.constants import MODEL_UA_MIN_V, MODEL_UA_MAX_DEFAULT_V


DATA_PATH = Path(__file__).parent.parent / "tests" / "spice_test_data" / "converted" / "pentode_6P1P_real.json"
TOP_N = 10
DFT_N_SAMPLES = 1024


def _load_points() -> List[Dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        d = json.load(f)
    pts = d["points"]
    for p in pts:
        if "ia" not in p or "ug1" not in p or "ua" not in p:
            continue
    return pts


def _time_call(fn, *args, **kwargs):
    """Run fn N times and return (avg_time_ms, last_result)."""
    n = 30
    t0 = time.perf_counter()
    result = None
    for _ in range(n):
        result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) / n * 1000.0
    return elapsed, result


def main() -> None:
    print(f"Loading 6P1P pentode data: {DATA_PATH.name}")
    pts = _load_points()
    print(f"  {len(pts)} points, {len(set(round(p['ug1'], 1) for p in pts))} Ug1 levels")

    # Filter to most-populated Ug2 (typical user case: pick one Ug2)
    ug2_counts = {}
    for p in pts:
        u = round(p.get("ug2", 0.0), 0)
        ug2_counts[u] = ug2_counts.get(u, 0) + 1
    main_ug2 = max(ug2_counts, key=ug2_counts.get)
    print(f"  Using Ug2 = {main_ug2} V ({ug2_counts[main_ug2]} pts)")

    # ── Fit Dempwolf model (Dempwolf is the best fitter for pentodes;
    #    benchmark on 86 datasets confirms ~80% wins) ──
    print("\nFitting Dempwolf model (for DFT method)...")
    t0 = time.perf_counter()
    model_result = fit_dempwolf(pts, topology="pentode")
    print(f"  fit time: {(time.perf_counter() - t0):.2f}s, RMS = {model_result.rms_error:.2f} mA")
    model = model_result.model

    # ── Run optimizer for each hd_method ──
    print("\nRunning optimizer for each hd_method:")
    print(f"{'method':<10}  {'time':>7}  {'best.thd':>9}  {'best.pout':>9}  {'method_used':<15}  {'warning'}")
    print("-" * 80)
    results_by_method = {}
    for hd_method in ("5point", "chebyshev", "auto", "dft"):
        c = OptimizerConstraints(
            target="min_thd", circuit="se",
            ug1_range=(-15.0, -7.0), ra_range=(2.0, 15.0),
            ug1_steps=8, ra_steps=8,
            pa_max_w=12.0,
            pout_min_w=0.05,
            swing_steps=3,
            hd_method=hd_method,
        )
        t0 = time.perf_counter()
        r = optimize_measurements(pts, ub=250.0, constraints=c,
                                  ug2_filter=main_ug2, model=model)
        dt = time.perf_counter() - t0
        results_by_method[hd_method] = r
        if r.best:
            print(f"{hd_method:<10}  {dt:6.2f}s  {r.best.thd:8.3f}%  "
                  f"{r.best.pout_mw / 1000:8.3f}W  {r.best.hd_method:<15}  "
                  f"{r.warning or '—'}")
        else:
            print(f"{hd_method:<10}  {dt:6.2f}s  no best - {r.error}")

    # ── Sparse-data guard verification: no fake-zero degeneracy ──
    print("\nSparse-data guard check (no fake-zero THD on degenerate small-swing points):")
    for hd_method, r in results_by_method.items():
        if r.best:
            assert r.best.thd > 0.05, (
                f"{hd_method}: best.thd={r.best.thd:.4f}% suspiciously low — "
                f"sparse-data guard may have failed"
            )
            print(f"  {hd_method}: best.thd = {r.best.thd:.3f}% (>= 0.05% sanity floor) OK")

    # ── Run optimizer (5-point) ──
    # Use Pout_min to force meaningful swing; 5-point degenerates at tiny swing
    # (5 sample points land on near-straight curve → fake THD ≈ 0).
    print("\nRunning optimizer (always 5-point in current code)...")
    constraints = OptimizerConstraints(
        target="min_thd", circuit="se",
        ug1_range=(-15.0, -7.0), ra_range=(2.0, 15.0),
        ug1_steps=10, ra_steps=10,
        pa_max_w=12.0,
        pout_min_w=0.2,            # force ≥ 0.2 W to avoid swing=0.4V degeneracy
        swing_steps=3,
    )
    t0 = time.perf_counter()
    result = optimize_measurements(
        pts, ub=250.0, constraints=constraints, ug2_filter=main_ug2,
    )
    opt_time = time.perf_counter() - t0
    print(f"  total time: {opt_time:.2f}s")
    print(f"  grid points: {len(result.grid_points)}, valid: {sum(1 for p in result.grid_points if p.valid)}")

    if not result.pareto_front:
        print("  No Pareto points - aborting.")
        return

    # Use Pareto front (diverse THD/Pout trade-offs) instead of grid top-N.
    # Pareto excludes degenerate small-swing points that grid top-by-THD favors.
    pareto = result.pareto_front
    top_n = pareto[:TOP_N] if len(pareto) >= TOP_N else pareto + sorted(
        [p for p in result.grid_points if p.valid and p not in pareto],
        key=lambda p: p.thd,
    )[: TOP_N - len(pareto)]
    print(f"\nTop {len(top_n)} by 5-point THD:")
    for i, pt in enumerate(top_n):
        print(f"  #{i+1}  Ug1={pt.ug1:6.2f}V  Ra={pt.ra:5.1f}kOhm  swing={pt.half_swing:5.2f}V  "
              f"THD_5pt={pt.thd:6.3f}%  Pout={pt.pout_mw / 1000:.3f}W")

    # ── Re-evaluate each top point with all 3 methods ──
    print("\nRe-evaluating top-N with Chebyshev and DFT:")
    print(f"{'#':>3}  {'5-point':>10}  {'Chebyshev':>10}  {'DFT':>10}  {'min/max':>8}")
    print("-" * 56)

    times_5pt, times_cheb, times_dft = [], [], []
    cross_check = []  # list of (thd_5pt, thd_cheb, thd_dft) per point

    for i, pt in enumerate(top_n):
        ll = ResistiveLoadLine(ub=250.0, ra=pt.ra)
        isects = find_intersections(pts, ll, ug2_filter=main_ug2)

        # 5-point
        t_5pt, r_5pt = _time_call(
            compute_distortion, isects, ug1_bias=pt.ug1,
            half_swing=pt.half_swing, ub=250.0,
        )
        times_5pt.append(t_5pt)
        thd_5pt = r_5pt["thd"] if r_5pt else float("nan")

        # Chebyshev
        t_cheb, r_cheb = _time_call(
            compute_distortion_chebyshev, isects, ug1_bias=pt.ug1,
            half_swing=pt.half_swing, ub=250.0,
        )
        times_cheb.append(t_cheb)
        thd_cheb = r_cheb["thd"] if r_cheb else float("nan")

        # DFT (via model)
        t_dft, r_dft = _time_call(
            compute_distortion_dft, model, ll,
            ug1_bias=pt.ug1, half_swing=pt.half_swing,
            ug2=main_ug2, ub=250.0, n_samples=DFT_N_SAMPLES,
        )
        times_dft.append(t_dft)
        thd_dft = r_dft["thd"] if r_dft else float("nan")

        valid_thds = [v for v in (thd_5pt, thd_cheb, thd_dft) if v == v and v > 0]
        ratio = (max(valid_thds) / min(valid_thds)) if len(valid_thds) >= 2 else float("nan")
        cross_check.append((thd_5pt, thd_cheb, thd_dft))

        print(f"{i+1:>3}  {thd_5pt:>9.3f}%  {thd_cheb:>9.3f}%  {thd_dft:>9.3f}%  {ratio:>7.2f}x")

    # ── Speed summary ──
    print("\nWall time per single THD evaluation (avg of 30 calls):")
    print(f"  5-point:    {sum(times_5pt) / len(times_5pt):6.3f} ms")
    print(f"  Chebyshev:  {sum(times_cheb) / len(times_cheb):6.3f} ms")
    print(f"  DFT (n={DFT_N_SAMPLES:>4}): {sum(times_dft) / len(times_dft):6.3f} ms")

    # Estimate full-grid optimization cost projection
    grid_size = constraints.ug1_steps * constraints.ra_steps * constraints.swing_steps
    print(f"\nProjected optimizer cost on this grid (~{grid_size} evaluations):")
    print(f"  5-point:    {sum(times_5pt) / len(times_5pt) * grid_size / 1000:6.2f} s")
    print(f"  Chebyshev:  {sum(times_cheb) / len(times_cheb) * grid_size / 1000:6.2f} s")
    print(f"  DFT:        {sum(times_dft) / len(times_dft) * grid_size / 1000:6.2f} s")

    # ── Ranking agreement ──
    print("\nRanking by THD (top-3 in each method):")
    for label, key_idx in [("5-point", 0), ("Chebyshev", 1), ("DFT", 2)]:
        ranked = sorted(
            range(len(top_n)),
            key=lambda i: cross_check[i][key_idx] if cross_check[i][key_idx] == cross_check[i][key_idx] else 1e9,
        )[:3]
        print(f"  {label:>10}: top-3 grid indices = {[i+1 for i in ranked]}")

    # ── Cross-method agreement summary ──
    print("\nCross-method THD ratio per point (max/min):")
    ratios = []
    for thds in cross_check:
        valid_thds = [v for v in thds if v == v and v > 0]
        if len(valid_thds) >= 2:
            ratios.append(max(valid_thds) / min(valid_thds))
    if ratios:
        print(f"  worst:   {max(ratios):.2f}x  (one method gives THD up to N times another)")
        print(f"  median:  {sorted(ratios)[len(ratios)//2]:.2f}x")
        print(f"  Target: cross-method THD agreement within ~3x")
        agree = all(r < 3.0 for r in ratios)
        print(f"  Within 3x for ALL points: {'YES' if agree else 'NO'}")


if __name__ == "__main__":
    main()
