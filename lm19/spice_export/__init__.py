"""SPICE model export — package facade.

Fits Koren / Dempwolf / Reefman tube models to measured data and exports
SPICE subcircuit (.sub) files compatible with LTspice / PSpice.

Submodules:
  - ``_common``  — ``SpiceFitResult`` dataclass + ``_HAS_SCIPY`` flag
  - ``koren``    — Koren math (`_koren_ia`, ``_koren_ia_pentode``,
    ``_koren_ig2_pentode``), fitters (`_fit_koren_*`, ``_fit_pentode_*``),
    subcircuit generators (`_generate_triode/pentode_subcircuit`),
    fit-and-export pipelines (`fit_and_export_triode/pentode`)
  - ``dempwolf`` — Dempwolf v2 subcircuit generators + pipeline
  - ``reefman``  — Reefman (Derk/DerkE) subcircuit + pipeline

Subcircuit text is built with Python f-strings inside each submodule
rather than from external template files: the generators have heavy
conditional branching (model sub-variants, optional caps, BT/Vmu/DE
flags) which would force one template per branch combination.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from lm19.spice_export._common import SpiceFitResult, _HAS_SCIPY
from lm19.spice_export.dempwolf import (
    _generate_dempwolf_pentode_subcircuit,
    _generate_dempwolf_triode_subcircuit,
    fit_and_export_dempwolf,
)
from lm19.spice_export.koren import (
    _fit_koren_numpy,
    _fit_koren_scipy,
    _fit_pentode_numpy,
    _fit_pentode_scipy,
    _generate_pentode_subcircuit,
    _generate_triode_subcircuit,
    _koren_ia,
    _koren_ia_pentode,
    _koren_ig2_pentode,
    fit_and_export_pentode,
    fit_and_export_triode,
)
from lm19.spice_export.reefman import (
    _generate_reefman_subcircuit,
    fit_and_export_reefman,
)
from lm19.tube_params import lookup_tube
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)
from lm19.tube_model_base import (
    MODEL_TYPE_DEMPWOLF,
    MODEL_TYPE_KOREN,
    MODEL_TYPE_REEFMAN,
)

log = logging.getLogger(__name__)


# ── module local constants ──
# Heuristic to warn when an unknown tube's data is an independent-screen pentode
# but is about to be fit with the triode model. Keyed on Ug2 departing from Ua:
# true-triode data has Ug2=0 or Ug2=Ua and triode-connected has Ug2≈Ua, so
# neither is flagged — only a fixed screen swept against the anode trips this.
_UG2_PRESENT_MIN_V = 1.0              # Ug2 at/below this means no screen (true triode)
_UG2_INDEPENDENT_MIN_DELTA_V = 10.0  # |Ug2-Ua| above which a point looks independent-screen
_PENTODE_DETECT_FRACTION = 0.5       # fraction of points that must look independent-screen


def _looks_like_independent_pentode(points: List[Dict]) -> bool:
    """True if the data looks like an independent-screen pentode scan.

    Flags points whose screen voltage departs from the anode voltage — true
    triode (Ug2=0 or Ug2=Ua) and triode-connected (Ug2≈Ua) data are NOT
    flagged, so only a fixed screen swept against the anode trips this.
    """
    if not points:
        return False
    n_independent = 0
    for p in points:
        ug2 = p.get("ug2", 0.0)
        if (ug2 > _UG2_PRESENT_MIN_V
                and abs(ug2 - p.get("ua", 0.0)) > _UG2_INDEPENDENT_MIN_DELTA_V):
            n_independent += 1
    return n_independent >= _PENTODE_DETECT_FRACTION * len(points)


def export_spice_from_model(
    path: str,
    model,
    tube_type: Optional[str] = None,
    mfg_date: str = "",
    fit_info: Optional[Dict] = None,
) -> SpiceFitResult:
    """Write a ``.sub`` from an ALREADY-FITTED model object — no refit.

    The subcircuit text comes from the very same generators the fit path
    uses, so a model carrying a fit's own parameters exports
    byte-identically to that fit (equivalence pins in
    ``tests/test_spice_from_model.py``). Used by the SPICE-export dialog
    ("Export loaded model") and by the LTspice verification when the
    analysis ran on a model source.

    Args:
        path: output file path (.sub).
        model: TubeModelProtocol carrying its parameter struct
            (``TubeModel.koren`` / ``DempwolfModel.dempwolf`` /
            ``ReefmanModel.reefman``) plus ``model_type``/``topology``.
        tube_type: subcircuit/header name; default ``model.name``.
        mfg_date: manufacturing date for the header ("" = omitted).
        fit_info: optional ``{"rms_error", "max_error", "n_points",
            "ig2_rms", "backend"}`` for the header. Absent stats are
            omitted from the header — model objects carry no fit
            statistics and inventing them would be dishonest.

    Raises:
        RuntimeError: unknown model type, or a Koren pentode model
            without ``kg2`` (its screen current was never fitted).
    """
    info = fit_info or {}
    tube = tube_type or getattr(model, "name", "tube")
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in tube)
    ref = lookup_tube(tube)
    topology = getattr(model, "topology", TOPOLOGY_TRIODE)
    model_type = getattr(model, "model_type", None)
    rms = info.get("rms_error")
    max_err = info.get("max_error")
    n_points = info.get("n_points")
    ig2_rms = info.get("ig2_rms")
    backend = info.get("backend", "loaded model (no refit)")

    if model_type == MODEL_TYPE_KOREN:
        k = model.koren
        if topology == TOPOLOGY_PENTODE:
            if k.kg2 is None:
                raise RuntimeError(
                    "Koren pentode model has no kg2 — screen current was "
                    "never fitted, refusing to export a fabricated one")
            content = _generate_pentode_subcircuit(
                safe_name, tube, k.mu, k.ex, k.kg1, k.kp, k.kvb, k.kg2,
                rms, max_err, n_points, backend, ig2_rms, ref,
                mfg_date=mfg_date)
            kind = TOPOLOGY_PENTODE
        else:
            content = _generate_triode_subcircuit(
                safe_name, tube, k.mu, k.ex, k.kg1, k.kp, k.kvb,
                rms, max_err, n_points, backend, ref, mfg_date=mfg_date)
            kind = TOPOLOGY_TRIODE
        params = {"mu": k.mu, "ex": k.ex, "kg1": k.kg1, "kp": k.kp,
                  "kvb": k.kvb, **({"kg2": k.kg2} if k.kg2 is not None
                                   else {})}
    elif model_type == MODEL_TYPE_DEMPWOLF:
        dp = model.dempwolf
        if topology == TOPOLOGY_PENTODE:
            content = _generate_dempwolf_pentode_subcircuit(
                safe_name, tube, dp, rms, max_err, n_points, ig2_rms,
                ref, mfg_date=mfg_date)
            kind = TOPOLOGY_PENTODE
        else:
            content = _generate_dempwolf_triode_subcircuit(
                safe_name, tube, dp, rms, max_err, n_points,
                ref, mfg_date=mfg_date)
            kind = TOPOLOGY_TRIODE
        from dataclasses import asdict
        params = asdict(dp)
    elif model_type == MODEL_TYPE_REEFMAN:
        rp = model.reefman
        content = _generate_reefman_subcircuit(
            safe_name, tube, rp, rms, max_err, n_points, ig2_rms,
            ref, mfg_date=mfg_date)
        kind = TOPOLOGY_PENTODE
        from dataclasses import asdict
        params = asdict(rp)
    else:
        raise RuntimeError(
            f"model type '{model_type}' cannot be exported to SPICE")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    log.info("Exported loaded %s model to %s (no refit)", model_type, path)
    return SpiceFitResult(
        model_type=kind,
        algorithm=model_type,
        params=params,
        rms_error=rms,
        max_error=max_err,
        n_points=n_points or 0,
        path=path,
    )


def fit_and_export_spice(
    path: str,
    tube_type: str,
    points: List[Dict],
    topology: Optional[str] = None,
    model_type: str = MODEL_TYPE_KOREN,
    mfg_date: str = "",
) -> SpiceFitResult:
    """Fit tube model to measured data and export SPICE subcircuit.

    Auto-detects triode/pentode topology from tube_params.json or
    measurement data. Returns SpiceFitResult for model overlay on plots.

    Args:
        path: output file path (.sub)
        tube_type: tube type name for the subcircuit
        points: measurement dicts with ua, ug1, ug2, ia, ig2 keys
        topology: override — "triode", "pentode", "triode_connected", or None
        model_type: "koren", "dempwolf", or "reefman"

    Returns:
        SpiceFitResult with fitted params and fit quality.

    Raises:
        RuntimeError: if not enough valid data points.
    """
    ref = lookup_tube(tube_type)
    ref_koren = ref.koren if ref else None

    if topology is None:
        if ref:
            topology = ref.topology
            log.info("Auto-detected topology '%s' for '%s' → %s",
                     topology, tube_type, ref.name)
        else:
            topology = TOPOLOGY_TRIODE
            if _looks_like_independent_pentode(points):
                log.warning(
                    "No reference for '%s' and the data looks like an "
                    "independent-screen pentode (Ug2 swept against Ua) — "
                    "fitting as TRIODE is likely wrong; pass an explicit "
                    "topology or add the tube to tube_params.json", tube_type)
            else:
                log.info("No reference for '%s', defaulting to triode",
                         tube_type)

    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in tube_type)
    backend = "scipy" if _HAS_SCIPY else "numpy"

    if model_type == MODEL_TYPE_REEFMAN:
        if topology in (TOPOLOGY_TRIODE, TOPOLOGY_TRIODE_CONNECTED):
            raise RuntimeError("Reefman models require pentode data "
                               "(independent Ug2)")
        return fit_and_export_reefman(
            path, tube_type, safe_name, points, ref, backend,
            mfg_date=mfg_date)

    if model_type == MODEL_TYPE_DEMPWOLF:
        return fit_and_export_dempwolf(
            path, tube_type, safe_name, points, ref, topology, backend,
            mfg_date=mfg_date)

    # Default: Koren
    if topology == TOPOLOGY_TRIODE_CONNECTED:
        log.info("Pentode in triode connection → using triode model")
        topology = TOPOLOGY_TRIODE

    if topology == TOPOLOGY_PENTODE:
        return fit_and_export_pentode(
            path, tube_type, safe_name, points, ref, ref_koren, backend,
            mfg_date=mfg_date)
    return fit_and_export_triode(
        path, tube_type, safe_name, points, ref, ref_koren, backend,
        mfg_date=mfg_date)


__all__ = [
    # Public API
    "SpiceFitResult",
    "fit_and_export_spice",
    "export_spice_from_model",
    # Internal helpers re-exported for tube_sim, tests, and tooling
    "_HAS_SCIPY",
    "_koren_ia",
    "_koren_ia_pentode",
    "_koren_ig2_pentode",
    "_fit_koren_scipy",
    "_fit_koren_numpy",
    "_fit_pentode_scipy",
    "_fit_pentode_numpy",
    "_generate_triode_subcircuit",
    "_generate_pentode_subcircuit",
    "_generate_dempwolf_pentode_subcircuit",
    "_generate_dempwolf_triode_subcircuit",
    "_generate_reefman_subcircuit",
]
