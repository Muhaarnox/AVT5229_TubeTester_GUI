"""Compare tube fitting models on real measurement data.

Runs the PRODUCTION fitters — Koren, Dempwolf v2 and Reefman
(lm19.reefman.fit_reefman, which tries both Derk and DerkE splitting and
returns the better variant) — on the same measurement data and compares
RMS/max errors. ML-137: the former local Derk/DerkE copy had drifted from
the production fitter; the table now shows exactly what the app ships.

Usage:
    cd d:/work/AVR/AVT5229/lm19_app
    py tools/compare_models.py                 # list available tube types and exit
    py tools/compare_models.py --all           # explicitly run on all tube types
    py tools/compare_models.py EL84            # only EL84 subfolder
    py tools/compare_models.py EL84 6S19P      # multiple tube types
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_measurement(filepath: str) -> Tuple[str, List[Dict]]:
    """Load measurement file, return (topology, points)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    topology = data.get("topology", "triode")
    points = data.get("points", [])
    return topology, points


# ---------------------------------------------------------------------------
# Reefman row — production fitter (single source, ML-137)
# ---------------------------------------------------------------------------

def fit_reefman_row(points: List[Dict]) -> Dict:
    """Run the production Reefman fitter and shape a comparison row.

    ``fit_reefman`` tries both Derk (D) and DerkE (DE) splitting and
    returns the better fit; the winning variant is shown in the model
    name. ``rms_ig2`` comes from the fitter itself (``ModelFitResult``).
    """
    from lm19.reefman import fit_reefman

    r = fit_reefman(points, "pentode")
    variant = str(r.model.params_dict().get("type", ""))
    variant = variant.replace("BTetrode", "") or "?"
    return {
        "model": f"Reefman ({variant})",
        "rms_ia": r.rms_error,
        "max_ia": r.max_error,
        "rms_ig2": r.rms_ig2 if r.rms_ig2 is not None else 0.0,
        "n_points": r.n_points,
    }


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

def run_comparison(filepath: str):
    """Run all 4 models on one measurement file."""
    topology, points = load_measurement(filepath)
    fname = Path(filepath).name
    print(f"\n{'='*72}")
    print(f"File: {fname}")
    print(f"Topology: {topology}, Points: {len(points)}")
    print(f"{'='*72}")

    if topology == "triode":
        print("  (triode — Derk/DerkE not applicable, skipping)")
        # Only Koren and Dempwolf
        from lm19.tube_sim import fit_koren
        from lm19.dempwolf import fit_dempwolf

        results = []
        try:
            kr = fit_koren(points, topology)
            results.append(("Koren", kr.rms_error, kr.max_error, 0.0, kr.n_points))
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            print(f"  Koren FAILED: {e}")

        try:
            dr = fit_dempwolf(points, topology)
            results.append(("Dempwolf v2", dr.rms_error, dr.max_error, 0.0, dr.n_points))
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            print(f"  Dempwolf FAILED: {e}")

        _print_results(results, has_ig2=False)
        return

    # Pentode or triode_connected
    from lm19.tube_sim import fit_koren
    from lm19.dempwolf import fit_dempwolf

    results = []

    # 1. Koren
    try:
        kr = fit_koren(points, topology)
        results.append(("Koren", kr.rms_error, kr.max_error, 0.0, kr.n_points))
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
        print(f"  Koren FAILED: {e}")

    # 2. Dempwolf v2
    try:
        dr = fit_dempwolf(points, topology)
        results.append(("Dempwolf v2", dr.rms_error, dr.max_error, 0.0, dr.n_points))
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
        print(f"  Dempwolf FAILED: {e}")

    # Reefman needs true pentode data (independent Ug2)
    if topology == "pentode":
        # 3. Reefman (production fitter; picks the better of D/DE)
        try:
            rf = fit_reefman_row(points)
            results.append((rf["model"], rf["rms_ia"], rf["max_ia"],
                            rf["rms_ig2"], rf["n_points"]))
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            print(f"  Reefman FAILED: {e}")
    else:
        print("  (triode_connected — Reefman requires independent Ug2, skipping)")

    has_ig2 = topology == "pentode"
    _print_results(results, has_ig2)


def _print_results(results, has_ig2=False):
    if not results:
        print("  No results to display.")
        return

    # Sort by RMS Ia
    results.sort(key=lambda r: r[1])

    print()
    if has_ig2:
        print(f"  {'Model':<14} {'RMS Ia':>8} {'Max Ia':>8} {'RMS Ig2':>8} {'Points':>7}")
        print(f"  {'':─<14} {'(mA)':─>8} {'(mA)':─>8} {'(mA)':─>8} {'':─>7}")
        for name, rms, mx, rms_ig2, n in results:
            print(f"  {name:<14} {rms:8.3f} {mx:8.3f} {rms_ig2:8.3f} {n:7d}")
    else:
        print(f"  {'Model':<14} {'RMS Ia':>8} {'Max Ia':>8} {'Points':>7}")
        print(f"  {'':─<14} {'(mA)':─>8} {'(mA)':─>8} {'':─>7}")
        for name, rms, mx, _, n in results:
            print(f"  {name:<14} {rms:8.3f} {mx:8.3f} {n:7d}")

    best = results[0][0]
    print(f"\n  Winner (Ia RMS): {best}")


ALL_FLAG = "--all"


def _select_tube_dirs(root: Path, filters: list[str]) -> list[Path]:
    """Return subfolders of ``root`` filtered by name.

    Filters are case-insensitive substring matches against the folder name;
    a directory is included if it matches any filter. Empty filter list →
    empty result (caller must explicitly request all via a separate path).
    """
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not filters:
        return []
    needles = [f.lower() for f in filters]
    return [d for d in subdirs if any(n in d.name.lower() for n in needles)]


def _list_available_tube_types(root: Path) -> list[str]:
    """Return sorted names of immediate subdirectories under root."""
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _print_usage_with_types(root: Path) -> None:
    print("Usage:")
    print("  py tools/compare_models.py --all              # process every tube type")
    print("  py tools/compare_models.py <type> [<type>...] # filter by tube type name")
    print()
    types = _list_available_tube_types(root)
    if not types:
        print(f"(no tube subfolders under {root})")
        return
    print(f"Available tube types in {root} ({len(types)}):")
    for name in types:
        print(f"  {name}")


def main():
    import io
    import sys as _sys
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")

    from lm19.data_paths import measurements_root
    anchor = Path(__file__).resolve().parents[1]
    root = measurements_root(anchor)
    if not root.exists():
        print(f"Measurements root not found: {root}")
        return

    args = _sys.argv[1:]
    if not args:
        _print_usage_with_types(root)
        return

    if ALL_FLAG in args:
        extra = [a for a in args if a != ALL_FLAG]
        if extra:
            print(f"Refusing to combine {ALL_FLAG} with other filters: {' '.join(extra)}")
            return
        tube_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    else:
        tube_dirs = _select_tube_dirs(root, args)

    if not tube_dirs:
        print(f"No tube subfolders in {root} match filter: {' '.join(args)}")
        return

    total_files = 0
    for meas_dir in tube_dirs:
        files = sorted(meas_dir.glob("*.json"))
        if not files:
            continue
        print(f"\n### Tube type: {meas_dir.name} ({len(files)} files)")
        total_files += len(files)
        for f in files:
            try:
                run_comparison(str(f))
            except Exception as e:
                print(f"\nERROR processing {f.name}: {e}")

    print("\n" + "=" * 72)
    print(f"Done. Processed {total_files} files across {len(tube_dirs)} tube type(s).")


if __name__ == "__main__":
    main()
