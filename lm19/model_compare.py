"""Compare all registered tube models on the same measured data.

Pure logic module — no Qt dependency.  Used by ModelDialog's Compare All
feature and by tests.

Usage::

    from lm19.model_compare import compare_all_models
    rows = compare_all_models(points, topology=TOPOLOGY_PENTODE)
    for r in rows:
        print(r.label, r.rms_ia, r.status)
"""


from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from lm19.data_paths import measurements_root
from lm19.tube_model_base import (
    MODEL_REGISTRY,
    ModelFitResult,
    TubeModelProtocol,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
)
from lm19.tube_model_base import (
    MODEL_TYPES,
)

log = logging.getLogger(__name__)

# Models that support SPICE export (spice_export.py)
SPICE_MODELS = set(MODEL_TYPES)


# ---------------------------------------------------------------------------
# CompareRow
# ---------------------------------------------------------------------------

@dataclass
class CompareRow:
    """One row in the comparison table."""

    model_type: str
    label: str
    n_params: int
    rms_ia: Optional[float]       # mA
    max_ia: Optional[float]       # mA
    rms_ig2: Optional[float]      # mA  (None for triodes)
    max_ig2: Optional[float]      # mA  (None for triodes)
    rms_gm: Optional[float]       # mA/V
    spice_support: bool
    status: str                   # "OK", "Failed", "N/A"
    fit_result: Optional[ModelFitResult] = None


# ---------------------------------------------------------------------------
# gm computation
# ---------------------------------------------------------------------------

def compute_gm_from_data(points: List[Dict]) -> np.ndarray:
    """Numerical gm (mA/V) from measured data via central differences.

    Groups by (ug2, ua) within tolerance, differentiates Ia by Ug1.
    Returns array aligned with *points*; NaN where gm can't be computed.
    """
    n = len(points)
    gm = np.full(n, np.nan)

    # Build arrays
    ua = np.array([p["ua"] for p in points], dtype=float)
    ug1 = np.array([p["ug1"] for p in points], dtype=float)
    ug2 = np.array([p.get("ug2", 0.0) for p in points], dtype=float)
    ia = np.array([p["ia"] for p in points], dtype=float)

    # Group by (ug2, ua) with tolerance
    tol = 0.5  # V
    assigned = np.zeros(n, dtype=bool)

    for i in range(n):
        if assigned[i]:
            continue
        # Find all points with same (ug2, ua)
        mask = (~assigned
                & (np.abs(ug2 - ug2[i]) < tol)
                & (np.abs(ua - ua[i]) < tol))
        idx = np.where(mask)[0]
        if len(idx) < 2:
            assigned[idx] = True
            continue

        # Sort by ug1
        order = np.argsort(ug1[idx])
        idx_sorted = idx[order]
        ug1_s = ug1[idx_sorted]
        ia_s = ia[idx_sorted]

        m = len(idx_sorted)
        for j in range(m):
            if j == 0:
                # Forward difference
                dug = ug1_s[1] - ug1_s[0]
                if abs(dug) > 1e-6:
                    gm[idx_sorted[j]] = (ia_s[1] - ia_s[0]) / dug
            elif j == m - 1:
                # Backward difference
                dug = ug1_s[m - 1] - ug1_s[m - 2]
                if abs(dug) > 1e-6:
                    gm[idx_sorted[j]] = (ia_s[m - 1] - ia_s[m - 2]) / dug
            else:
                # Central difference
                dug = ug1_s[j + 1] - ug1_s[j - 1]
                if abs(dug) > 1e-6:
                    gm[idx_sorted[j]] = (ia_s[j + 1] - ia_s[j - 1]) / dug

        assigned[idx] = True

    return gm


def compute_gm_from_model(
    model: TubeModelProtocol,
    points: List[Dict],
    delta: float = 0.05,
) -> np.ndarray:
    """Model gm (mA/V) via symmetric finite difference."""
    n = len(points)
    gm = np.empty(n)
    for i, p in enumerate(points):
        ua = p["ua"]
        ug1 = p["ug1"]
        ug2 = p.get("ug2", 0.0)
        ia_plus = model.ia(ua, ug1 + delta, ug2)
        ia_minus = model.ia(ua, ug1 - delta, ug2)
        gm[i] = (ia_plus - ia_minus) / (2 * delta)
    return gm


def compute_rms_gm(
    model: TubeModelProtocol,
    points: List[Dict],
    data_gm: np.ndarray,
) -> Optional[float]:
    """RMS of (model_gm - data_gm) in mA/V.

    Only uses points where data_gm is valid and |data_gm| > 0.1 mA/V
    (filters out cutoff region).  Returns None if too few valid points.
    """
    model_gm = compute_gm_from_model(model, points)
    valid = ~np.isnan(data_gm) & (np.abs(data_gm) > 0.1)
    if np.sum(valid) < 3:
        return None
    diff = model_gm[valid] - data_gm[valid]
    return float(np.sqrt(np.mean(diff ** 2)))


# ---------------------------------------------------------------------------
# Model compatibility
# ---------------------------------------------------------------------------

# Models that only work with specific topologies
_PENTODE_ONLY = {"reefman"}


def _is_compatible(model_type: str, topology: str) -> bool:
    """Check if a model type can handle the given topology."""
    if model_type in _PENTODE_ONLY:
        return topology == TOPOLOGY_PENTODE
    return True


# ---------------------------------------------------------------------------
# compare_all_models
# ---------------------------------------------------------------------------

def compare_all_models(
    points: List[Dict],
    topology: str,
    cancelled: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[CompareRow]:
    """Run all registered fitters and return comparison rows.

    Args:
        points: measured data (list of dicts with ua, ug1, ug2, ia, ig2)
        topology: "triode", "triode_connected", or "pentode"
        cancelled: optional callable returning True to abort early
        on_progress: optional callback(current, total, label) for progress

    Returns:
        list of CompareRow sorted by rms_ia (best first), N/A at end.
    """
    # Ensure models are registered
    import lm19.tube_sim       # noqa: F401
    import lm19.dempwolf       # noqa: F401
    import lm19.reefman        # noqa: F401

    entries = list(MODEL_REGISTRY.items())
    total = len(entries)
    rows: List[CompareRow] = []

    # Pre-compute data gm once
    data_gm = compute_gm_from_data(points)

    for i, (model_type, entry) in enumerate(entries):
        if cancelled and cancelled():
            break

        if on_progress:
            on_progress(i, total, entry.label)

        if not _is_compatible(model_type, topology):
            rows.append(CompareRow(
                model_type=model_type,
                label=entry.label,
                n_params=0,
                rms_ia=None, max_ia=None,
                rms_ig2=None, max_ig2=None,
                rms_gm=None,
                spice_support=model_type in SPICE_MODELS,
                status="N/A",
            ))
            continue

        try:
            fit_result = entry.fitter(points, topology)
        except (ValueError, RuntimeError, KeyError,
                np.linalg.LinAlgError) as exc:
            # ML-103: narrow except — a refactor regression (Attribute/
            # TypeError) must crash visibly, not read as "model didn't fit".
            log.warning("Fitter %s failed: %s", model_type, exc)
            rows.append(CompareRow(
                model_type=model_type,
                label=entry.label,
                n_params=0,
                rms_ia=None, max_ia=None,
                rms_ig2=None, max_ig2=None,
                rms_gm=None,
                spice_support=model_type in SPICE_MODELS,
                status="Failed",
            ))
            continue

        # Compute gm error
        rms_gm = compute_rms_gm(fit_result.model, points, data_gm)

        rows.append(CompareRow(
            model_type=model_type,
            label=entry.label,
            n_params=len(fit_result.params),
            rms_ia=fit_result.rms_error,
            max_ia=fit_result.max_error,
            rms_ig2=fit_result.rms_ig2,
            max_ig2=fit_result.max_ig2,
            rms_gm=rms_gm,
            spice_support=model_type in SPICE_MODELS,
            status="OK",
            fit_result=fit_result,
        ))

    if on_progress:
        on_progress(total, total, "")

    # Sort: OK rows by rms_ia ascending, then Failed, then N/A
    def sort_key(r: CompareRow):
        status_order = {"OK": 0, "Failed": 1, "N/A": 2}
        return (status_order.get(r.status, 3), r.rms_ia or float("inf"))

    rows.sort(key=sort_key)
    return rows


# ---------------------------------------------------------------------------
# FileCompareRow / compare_all_files
# ---------------------------------------------------------------------------

@dataclass
class FileCompareRow:
    """Comparison results for one measurement file."""

    file_path: str
    file_name: str
    n_points: int
    topology: str
    models: List[CompareRow] = field(default_factory=list)


def compare_all_files(
    measurements_dir: Optional[str] = None,
) -> List[FileCompareRow]:
    """Scan all JSON files in measurements/ and compare all models on each.

    Args:
        measurements_dir: path to measurements directory.
            Defaults to ``<project>/measurements``.

    Returns:
        list of FileCompareRow, one per measurement file, sorted by filename.
    """
    if measurements_dir is None:
        anchor = Path(__file__).resolve().parents[1]
        measurements_dir = str(measurements_root(anchor))

    results: List[FileCompareRow] = []

    for root, _dirs, files in os.walk(measurements_dir):
        for fname in sorted(files):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Skipping %s: %s", fpath, exc)
                continue

            points = data.get("points", [])
            topology = data.get("topology", TOPOLOGY_PENTODE)
            if len(points) < 10:
                continue

            rows = compare_all_models(points, topology)
            results.append(FileCompareRow(
                file_path=fpath,
                file_name=fname,
                n_points=len(points),
                topology=topology,
                models=rows,
            ))

    results.sort(key=lambda r: r.file_name)
    return results


def print_comparison(results: Optional[List[FileCompareRow]] = None) -> None:
    """Pretty-print comparison table to stdout."""
    if results is None:
        results = compare_all_files()

    for fr in results:
        short = fr.file_name[:55]
        print(f"\n{'=' * 70}")
        print(f"{short}  ({fr.n_points} pts, {fr.topology})")
        print(f"{'=' * 70}")
        print(f"  {'Model':<22s} {'Ia RMS':>8s} {'Ia Max':>8s}"
              f" {'Ig2 RMS':>8s} {'gm RMS':>8s}  Status")
        print(f"  {'-' * 66}")
        for r in fr.models:
            ia_rms = f"{r.rms_ia:.3f}" if r.rms_ia is not None else "—"
            ia_max = f"{r.max_ia:.3f}" if r.max_ia is not None else "—"
            ig2 = f"{r.rms_ig2:.3f}" if r.rms_ig2 is not None else "—"
            gm = f"{r.rms_gm:.3f}" if r.rms_gm is not None else "—"
            print(f"  {r.label:<22s} {ia_rms:>8s} {ia_max:>8s}"
                  f" {ig2:>8s} {gm:>8s}  {r.status}")


if __name__ == "__main__":
    print_comparison()
