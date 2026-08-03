"""Model-Chebyshev PP vs DFT PP — equivalence & speed comparison.

Regenerates the table in ``docs/UL_CHEBYSHEV_VALIDATION.md``:

    py tools/ul_cheb_vs_dft.py

Sweeps UL tap × bias × swing on the quick_pentode EL84 model, computes
THD/Pout with both methods, reports relative deltas and timings. Exit
code 1 when any delta exceeds the CI pin tolerances (so the tool itself
is a checkable artifact, not just a printout).
"""

from __future__ import annotations

import io
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.amplifier import (
    PushPullLoadLine,
    UltralinearModelWrapper,
    compute_distortion_chebyshev_pp_model,
    compute_distortion_dft_pp,
)
from lm19.tube_sim import quick_pentode

# Mirrors tests/test_ul_chebyshev_model.py pins
THD_REL_TOL = 0.02
POUT_REL_TOL = 0.01

TAPS = (0.0, 0.2, 0.43, 1.0)
BIASES = (-9.0, -11.0, -13.0)
SWINGS = (4.0, 9.0)
N_TIMING_REPEATS = 20


def main() -> int:
    # Windows console defaults to cp1251 — the table uses Δ/×.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    model, _ = quick_pentode("EL84")
    ll = PushPullLoadLine(300.0, 8.0)

    rows: List[str] = []
    failures = 0
    t_cheb_total = 0.0
    t_dft_total = 0.0
    n_evals = 0

    for tap in TAPS:
        m = (UltralinearModelWrapper(model, ug2_nom=250.0, tap=tap)
             if tap > 0 else model)
        for bias in BIASES:
            for swing in SWINGS:
                t0 = time.perf_counter()
                for _ in range(N_TIMING_REPEATS):
                    cheb = compute_distortion_chebyshev_pp_model(
                        m, ll, bias, half_swing=swing, ug2=250.0)
                t_cheb = (time.perf_counter() - t0) / N_TIMING_REPEATS
                t0 = time.perf_counter()
                for _ in range(N_TIMING_REPEATS):
                    dft = compute_distortion_dft_pp(
                        m, ll, bias, half_swing=swing, ug2=250.0)
                t_dft = (time.perf_counter() - t0) / N_TIMING_REPEATS
                if cheb is None or dft is None:
                    rows.append(
                        f"| {tap:.2f} | {bias:.0f} | {swing:.0f} | — | — |"
                        f" — | — | — | n/a |")
                    continue
                d_thd = abs(cheb["thd"] - dft["thd"]) / max(dft["thd"], 1e-9)
                d_pout = (abs(cheb["pout_mw"] - dft["pout_mw"])
                          / max(dft["pout_mw"], 1e-9))
                ok = d_thd <= THD_REL_TOL and d_pout <= POUT_REL_TOL
                failures += 0 if ok else 1
                t_cheb_total += t_cheb
                t_dft_total += t_dft
                n_evals += 1
                rows.append(
                    f"| {tap:.2f} | {bias:.0f} | {swing:.0f} "
                    f"| {dft['thd']:.3f} | {cheb['thd']:.3f} "
                    f"| {d_thd * 100:.2f}% | {d_pout * 100:.2f}% "
                    f"| ×{t_dft / max(t_cheb, 1e-9):.1f} "
                    f"| {'OK' if ok else 'FAIL'} |")

    speedup = t_dft_total / max(t_cheb_total, 1e-9)
    print("| tap | bias, V | swing, V | THD DFT, % | THD Cheb, % "
          "| ΔTHD rel | ΔPout rel | speedup | verdict |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(r)
    print(f"\nMean per-eval time: DFT {t_dft_total / n_evals * 1e3:.2f} ms, "
          f"Chebyshev {t_cheb_total / n_evals * 1e3:.2f} ms "
          f"(speedup ×{speedup:.1f}, {n_evals} points)")
    print(f"Tolerances: THD rel ≤ {THD_REL_TOL:.0%}, "
          f"Pout rel ≤ {POUT_REL_TOL:.0%}; failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
