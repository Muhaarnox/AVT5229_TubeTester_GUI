#!/usr/bin/env python
"""Benchmark all tube model fitters against test data.

Usage:
    py tools/fit_benchmark.py                    # all test data
    py tools/fit_benchmark.py EL84               # only files matching 'EL84'
    py tools/fit_benchmark.py pentode             # only pentode files
    py tools/fit_benchmark.py 6S19P 6C33C        # specific tubes
    py tools/fit_benchmark.py --real              # only *_real.json files
    py tools/fit_benchmark.py --no-ref            # disable reference params

Reference params (initial-guess seeding from tube_params.json) are only
consumed by the Koren fitter; fit_dempwolf/fit_reefman take measurement
data only, so ``--no-ref`` affects the Koren column alone.

Run from lm19_app/:
    cd d:\\work\\AVR\\AVT5229\\lm19_app && py tools/fit_benchmark.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from lm19.dempwolf import fit_dempwolf
from lm19.spice_export import fit_and_export_spice

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "spice_test_data" / "converted"

# Suffix guaranteed not to match any tube_params.json name/alias — passing
# "<tube>__no_ref__" to fit_and_export_spice makes lookup_tube() return None,
# which disables reference seeding without monkeypatching (a module-attribute
# patch never reached the from-import in lm19.spice_export anyway).
_NO_REF_SUFFIX = "__no_ref__"

FitOutcome = Tuple[float, float, Dict, float]  # (rms_mA, max_mA, params, sec)


def _fit_safe(
    points: List[Dict],
    topology: str,
    model_type: str,
    tube_name: str,
    use_ref: bool = True,
) -> Optional[FitOutcome]:
    """Fit model, return (rms, max_err, params, time_sec) or None on failure.

    Only expected fit/data errors are converted to None (and printed);
    programming errors propagate so lm19 API drift fails loudly instead of
    reading as "all models lost" (failure-visibility principle 1).
    """
    lookup_name = tube_name if use_ref else f"{tube_name}{_NO_REF_SUFFIX}"
    t0 = time.time()
    try:
        if model_type == "dempwolf":
            r = fit_dempwolf(points, topology)
            return r.rms_error, r.max_error, r.params, time.time() - t0
        elif model_type == "reefman":
            from lm19.reefman import fit_reefman
            r = fit_reefman(points, topology)
            return r.rms_error, r.max_error, r.params, time.time() - t0
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                r = fit_and_export_spice(
                    str(Path(tmpdir) / "bench.sub"), lookup_name, points,
                    topology=topology, model_type=model_type,
                )
                return r.rms_error, r.max_error, r.params, time.time() - t0
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
        print(f"    [{tube_name}] {model_type} failed: "
              f"{type(e).__name__}: {e}")
        return None


def run_benchmark(
    filters: Optional[Sequence[str]] = None,
    real_only: bool = False,
    use_ref: bool = True,
) -> None:
    """Run benchmark and print results table."""
    files = sorted(DATA_DIR.glob("*.json"))
    if not files:
        print(f"No test data found in {DATA_DIR}")
        return

    # Apply filters
    if filters:
        filtered = []
        for f in files:
            name_lower = f.stem.lower()
            if any(flt.lower() in name_lower for flt in filters):
                filtered.append(f)
        files = filtered

    if real_only:
        files = [f for f in files if "_real" in f.stem]

    if not files:
        print("No matching files found.")
        return

    # Header
    ref_tag = "" if use_ref else " (no ref)"
    print(f"\n{'Tube':>20} {'Topo':>7} {'Pts':>5} | "
          f"{'Koren'+ref_tag:>14} {'Dempwolf':>10} {'Reefman':>10} | "
          f"{'Winner':>10} {'D/K':>5}")
    print("-" * 95)

    summary = {"koren": 0, "dempwolf": 0, "reefman": 0, "total": 0}

    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"    skipping {path.name}: {type(e).__name__}: {e}")
            continue

        pts = data.get("points", [])
        if len(pts) < 15:
            continue

        tube = data.get("tube_type", path.stem)
        topo = data.get("topology", "triode" if "triode" in path.stem else "pentode")

        # Koren (the only fitter consuming reference params — via
        # lookup_tube(tube) inside fit_and_export_spice)
        rk = _fit_safe(pts, topo, "koren", tube, use_ref)
        rms_k = rk[0] if rk else float("inf")

        # Dempwolf (fitter takes no reference params)
        rd = _fit_safe(pts, topo, "dempwolf", tube)
        rms_d = rd[0] if rd else float("inf")

        # Reefman (pentode only; fitter takes no reference params)
        if topo == "pentode":
            rr = _fit_safe(pts, topo, "reefman", tube)
            rms_r = rr[0] if rr else float("inf")
        else:
            rr = None
            rms_r = float("inf")

        # Winner
        best = min(rms_k, rms_d, rms_r)
        if best == float("inf"):
            winner = "FAIL"
        elif best == rms_k:
            winner = "Koren"
            summary["koren"] += 1
        elif best == rms_d:
            winner = "Dempwolf"
            summary["dempwolf"] += 1
        else:
            winner = "Reefman"
            summary["reefman"] += 1
        summary["total"] += 1

        # Format
        def fmt(r: Optional[FitOutcome]) -> str:
            if r is None:
                return "    FAIL"
            return f"{r[0]:8.2f}mA"

        ratio = f"{rms_d / max(rms_k, 0.01):.1f}x" if rms_k < 1e6 and rms_d < 1e6 else "  —"

        print(f"{tube:>20} {topo:>7} {len(pts):5d} | "
              f"{fmt(rk):>14} {fmt(rd):>10} {fmt(rr) if topo == 'pentode' else '       N/A':>10} | "
              f"{winner:>10} {ratio:>5}")

    # Summary
    print("-" * 95)
    print(f"\nSummary: {summary['total']} datasets — "
          f"Koren wins {summary['koren']}, "
          f"Dempwolf wins {summary['dempwolf']}, "
          f"Reefman wins {summary['reefman']}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    real_only = "--real" in sys.argv
    no_ref = "--no-ref" in sys.argv

    run_benchmark(
        filters=args if args else None,
        real_only=real_only,
        use_ref=not no_ref,
    )
