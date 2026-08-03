"""Tube model protocol, registry, and shared fitting helpers.

Defines a formal contract (Protocol) for tube models, a registry for
dispatching model operations by model_type string, and helper functions
that consolidate boilerplate shared across all fitters
(``extract_arrays``, ``compute_fit_errors``, ``build_fit_result``).

Currently registered models:
  - "koren": Koren triode/pentode (registered in tube_sim.py on import)
  - "dempwolf": Dempwolf v2 triode/pentode
  - "reefman": Reefman BTetrodeD/DE pentode

Future models register themselves on import.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from lm19.constants import (
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)


# scipy.optimize.least_squares status codes (from scipy docs).
# success=False most commonly means status=0 (max_nfev hit without
# convergence — model gets noise instead of fit).
_LS_STATUS_DESCR: Dict[int, str] = {
    -1: "improper input",
    0: "max_nfev exceeded (no convergence)",
    1: "gtol satisfied",
    2: "ftol satisfied",
    3: "xtol satisfied",
    4: "ftol+xtol both satisfied",
}


class ConvergenceTracker:
    """Aggregate scipy convergence across multi-phase fits.

    Each ``check`` call wraps :func:`check_fit_convergence` and additionally
    flips ``all_converged`` to ``False`` if any phase didn't satisfy a
    termination criterion. The fitter passes the final ``all_converged``
    flag to ``build_fit_result``, which surfaces it in the UI as a
    "partial fit" badge.
    """

    def __init__(self) -> None:
        self.all_converged: bool = True

    def check(self, result: Any, phase_name: str,
              logger: logging.Logger, **context: Any) -> Any:
        if not getattr(result, "success", True):
            self.all_converged = False
        return check_fit_convergence(result, phase_name, logger, **context)


def check_fit_convergence(
    result: Any,
    phase_name: str,
    logger: logging.Logger,
    **context: Any,
) -> Any:
    """Log a WARNING if scipy least_squares didn't converge.

    Returns ``result`` unchanged so callers can chain
    ``params = check_fit_convergence(least_squares(...), 'phase1', log).x``.

    Caller still decides whether to use the un-converged result — the RMS
    error reported in ``ModelFitResult`` is the ground truth, but a
    silent un-converged fit means ``params`` may be at the bound or near
    the initial guess. This helper makes that visible in logs without
    failing the fit (sometimes a partial fit is still useful).

    Args:
        result: scipy ``OptimizeResult`` with ``success`` / ``status`` /
            ``cost`` / ``nfev`` attrs.
        phase_name: short name for the fit phase (e.g. "dempwolf phase1").
        logger: caller's module logger.
        **context: extra key=value pairs printed for debugging
            (e.g. ``tube='EL84'``, ``n_points=42``).
    """
    if getattr(result, "success", True):
        return result
    status = getattr(result, "status", None)
    descr = _LS_STATUS_DESCR.get(status, f"status={status}")
    ctx = " ".join(f"{k}={v}" for k, v in context.items())
    logger.warning(
        "fit %s did not converge: %s (cost=%.4g nfev=%d) %s",
        phase_name, descr,
        getattr(result, "cost", float("nan")),
        getattr(result, "nfev", -1),
        ctx,
    )
    return result


@runtime_checkable
class TubeModelProtocol(Protocol):
    """Formal contract for tube models.

    Any tube model (Koren, Dempwolf, etc.) must satisfy this interface.
    Used by PlotManager, Renderer, and Amplifier for model-agnostic access.

    **Thread-safety contract** — implementations MUST be **stateless after
    construction**: ``ia()`` / ``ig2()`` must read only constructor inputs
    (parameter dataclasses), never mutate ``self``, never use module-level
    caches keyed on per-call inputs, and never lazy-initialize on first
    call. ``lm19.optimizer.refine_pareto_front`` calls ``ia/ig2`` from
    multiple threads concurrently (``ThreadPoolExecutor``); a stateful
    model would race silently and corrupt optimization results.

    The three current implementations (``TubeModel`` for Koren,
    ``DempwolfModel``, ``ReefmanModel``) all satisfy this — verified via
    grep for ``self.X = …`` outside ``__init__``. If a future model adds
    ``@lru_cache`` on a bound method or ``self._memo`` for a slow
    computation, wrap the cache-bearing object per-thread (e.g. via
    ``copy.copy(model)`` in the parallel caller), or accept a Lock cost.
    """

    model_type: str
    name: str
    topology: str
    pa_max: float
    uh: float
    ih: float

    def ia(self, ua: float, ug1: float, ug2: float = 0.0) -> float:
        """Compute anode current Ia (mA) at a single operating point."""
        ...

    def ig2(self, ua: float, ug1: float, ug2: float) -> float:
        """Compute screen grid current Ig2 (mA).

        ua is required because Dempwolf ig2 depends on Va (current splitting).
        Koren adapter ignores ua.
        """
        ...

    def generate_scan(self, grid) -> List[Dict]:
        """Generate measurement points on the given ScanGrid."""
        ...

    def params_dict(self) -> Dict:
        """Return model parameters as a plain dict (for display/serialization)."""
        ...


def model_ia_array(
    model: "TubeModelProtocol",
    ua: "np.ndarray | float",
    ug1: "np.ndarray | float",
    ug2: "np.ndarray | float" = 0.0,
) -> np.ndarray:
    """Vectorized Ia (mA) over broadcastable (ua, ug1, ug2) arrays.

    Uses the model's ``ia_array`` fast path when it exists (all three
    built-in models and ``UltralinearModelWrapper`` provide one — same
    math as scalar ``ia()``, evaluated element-wise in numpy). Falls back
    to a scalar ``ia()`` loop for models/mocks without it, so the method
    is an optional protocol extension, not a breaking requirement.

    This is the hot path of the optimizer (intersection search + DFT
    Newton): a scalar ``ia()`` call costs ~15 µs of Python overhead vs
    ~0.1 µs/point through the vectorized kernels.
    """
    fast = getattr(model, "ia_array", None)
    if fast is not None:
        return np.asarray(fast(ua, ug1, ug2), dtype=float)
    ua_b, ug1_b, ug2_b = np.broadcast_arrays(
        np.asarray(ua, dtype=float),
        np.asarray(ug1, dtype=float),
        np.asarray(ug2, dtype=float),
    )
    out = np.empty(ua_b.shape, dtype=float)
    flat = out.ravel()
    for i, (a, g1, g2) in enumerate(
            zip(ua_b.ravel(), ug1_b.ravel(), ug2_b.ravel())):
        flat[i] = model.ia(float(a), float(g1), float(g2))
    return out


# Fit quality thresholds (RMS error as % of mean measured Ia)
# Tuned against Dempwolf benchmark: typical converged fits land at 1-3%
# for triodes / 3-7% for pentodes. >10% almost always means scipy gave
# noise or measurement is dominated by outliers.
FIT_QUALITY_GREEN_PCT = 2.0
FIT_QUALITY_YELLOW_PCT = 10.0


def compute_fit_quality(rms_error_mA: float,
                         mean_ia_mA: float) -> Tuple[float, str]:
    """Compute fit-quality verdict from RMS error relative to mean current.

    Returns ``(rms_pct, verdict)``:
      * ``rms_pct`` — RMS as percentage of mean Ia (0..∞)
      * ``verdict`` — ``"good"`` / ``"fair"`` / ``"poor"`` / ``"unknown"``
        (the last when ``mean_ia_mA <= 0``).

    Used by ``build_fit_result`` and surfaced in the model dialog as the
    main user-facing signal of whether a fit is trustworthy.
    """
    if mean_ia_mA <= 0:
        return 0.0, "unknown"
    rms_pct = abs(rms_error_mA) / mean_ia_mA * 100.0
    if rms_pct < FIT_QUALITY_GREEN_PCT:
        verdict = "good"
    elif rms_pct < FIT_QUALITY_YELLOW_PCT:
        verdict = "fair"
    else:
        verdict = "poor"
    return rms_pct, verdict



# -- Model contract vocabularies -------------------------------------
from lm19.constants import (  # noqa: F401  (re-export, canonical path)
    MODEL_TYPE_DEMPWOLF as MODEL_TYPE_DEMPWOLF,
    MODEL_TYPE_KOREN as MODEL_TYPE_KOREN,
    MODEL_TYPE_REEFMAN as MODEL_TYPE_REEFMAN,
    MODEL_TYPES as MODEL_TYPES,
)
# ModelFitResult.warnings codes — UI maps to model.warn_<code>;
# registry <-> locales tied by a bijection pin
# (test_conventions_guards).
MODEL_WARN_REEFMAN_FEW_UG2 = "reefman_few_ug2"
MODEL_WARN_DEMPWOLF_PHASE2_TRIODE_FORM = "dempwolf_phase2_triode_form"
MODEL_WARN_DEMPWOLF_NO_IG2_MASK = "dempwolf_no_ig2_mask"
MODEL_WARN_DEMPWOLF_NO_IG2_FULL = "dempwolf_no_ig2_full"
MODEL_WARN_DEMPWOLF_GRID_DEFAULTS = "dempwolf_grid_defaults"
MODEL_WARN_KOREN_KG2_UNFITTED = "koren_kg2_unfitted"
MODEL_WARNING_CODES = frozenset({
    MODEL_WARN_REEFMAN_FEW_UG2, MODEL_WARN_DEMPWOLF_PHASE2_TRIODE_FORM,
    MODEL_WARN_DEMPWOLF_NO_IG2_MASK, MODEL_WARN_DEMPWOLF_NO_IG2_FULL,
    MODEL_WARN_DEMPWOLF_GRID_DEFAULTS, MODEL_WARN_KOREN_KG2_UNFITTED,
})


@dataclass
class ModelFitResult:
    """Result of fitting a model to measured data.

    ``converged`` and ``quality`` are surfaced in the UI: a fit can be
    *valid but un-converged* (scipy hit max_nfev — ``params`` are the
    final iterate, not a true minimum) or *converged but poor quality*
    (RMS too high relative to mean Ia — suggests data issues or model
    mismatch). The user dialog colors the RMS line accordingly.
    """

    model_type: str
    topology: str
    params: Dict
    rms_error: float
    max_error: float
    n_points: int
    model: TubeModelProtocol
    # Pentode Ig2 errors (None for triodes)
    rms_ig2: Optional[float] = None
    max_ig2: Optional[float] = None
    # Quality / confidence signals — defaulted so fitters that don't
    # compute them yet (or callers that build the result by hand) still
    # get a valid object.
    converged: bool = True
    rms_pct: float = 0.0  # RMS as % of mean Ia
    quality: str = "unknown"  # "good" | "fair" | "poor" | "unknown"
    # Fitter degradation signals: {"code": str, **params} dicts, rendered
    # by the model dialog via model.warn_<code> i18n keys (failure-visibility rule:
    # a log-only warning never reaches the user).
    warnings: List[Dict] = field(default_factory=list)


@dataclass
class ModelRegistryEntry:
    """Registry entry for a model type."""

    label: str
    loader: Callable[[str], Optional[TubeModelProtocol]]
    fitter: Callable[[List[Dict], str], ModelFitResult]
    list_tubes: Callable[[], List[str]]


MODEL_REGISTRY: Dict[str, ModelRegistryEntry] = {}


def register_model(
    model_type: str,
    label: str,
    loader: Callable[[str], Optional[TubeModelProtocol]],
    fitter: Callable[[List[Dict], str], ModelFitResult],
    list_tubes: Callable[[], List[str]],
) -> None:
    """Register a model type in the global registry."""
    MODEL_REGISTRY[model_type] = ModelRegistryEntry(
        label=label,
        loader=loader,
        fitter=fitter,
        list_tubes=list_tubes,
    )


def list_all_tubes(model_type: str) -> List[str]:
    """List all available tubes for a given model type.

    Raises KeyError if model_type is not registered.
    """
    return MODEL_REGISTRY[model_type].list_tubes()


# ---------------------------------------------------------------------------
# Shared fitter helpers
# ---------------------------------------------------------------------------

# ── module local constants ──
_MA_PER_A = 1000.0          # conversion: A → mA
_DEFAULT_IG2_EPS_A = 1e-5   # treat Ig2 < 0.01 mA as "no Ig2 data"


@dataclass
class ExtractedFitData:
    """Filtered measurement arrays ready for tube-model fitting.

    All current values are in **Amperes** (after `mA / 1000.0` conversion).
    Voltages stay in Volts.

    Attributes:
        ua, ug1, ug2: filtered voltage arrays. ``ug2`` is a zeros array
            for ``topology=TOPOLOGY_TRIODE`` (caller can ignore).
        ia: filtered anode current, A.
        ig2: filtered screen current, A — or ``None`` if no Ig2 data was
            provided in the source points (all zeros / below threshold).
        n_points: number of points after filtering.
        has_ig2: whether the source data contains usable Ig2 measurements.
    """

    ua: np.ndarray
    ug1: np.ndarray
    ug2: np.ndarray
    ia: np.ndarray
    ig2: Optional[np.ndarray]
    n_points: int
    has_ig2: bool


def extract_arrays(
    points: List[Dict],
    *,
    topology: str,
    ia_thr_mA: float = 0.1,
    min_count: int = 10,
    ug2_min_v: float = 10.0,
    ig2_eps_A: float = _DEFAULT_IG2_EPS_A,
) -> ExtractedFitData:
    """Extract and filter Ua/Ug1/Ug2/Ia/Ig2 arrays for fitting.

    Filters:
      - ``ia_mA > ia_thr_mA`` (always)
      - ``ug2 > ug2_min_v`` (pentode only)

    Returned ``ia`` and ``ig2`` are in **Amperes**, not mA. Use
    ``compute_fit_errors`` to convert prediction-vs-measured residual
    back to mA for human-readable RMS/max metrics.

    Args:
        points: scan points (each dict has ``ua``, ``ug1``, ``ia`` always;
            ``ug2``, ``ig2`` optional).
        topology: ``"triode"`` (no Ug2 filter, ig2=None),
            ``"pentode"`` (filter ug2 > ug2_min_v),
            ``"triode_connected"`` (Ug2 defaults to Ua when missing).
        ia_thr_mA: minimum Ia in mA to keep a point. Koren default 0.1;
            Dempwolf uses 0.05 for tighter cutoff.
        min_count: raise RuntimeError if fewer points survive the filter.
        ug2_min_v: minimum Ug2 to keep (pentode only).
        ig2_eps_A: Ig2 below this threshold is treated as "no data";
            ``has_ig2`` = False and ``ig2`` returned as None.

    Raises:
        RuntimeError: if filtered count is less than ``min_count``,
            or if ``points`` is empty.
    """
    if not points:
        raise RuntimeError("No points to fit")

    ua_raw = np.array([p["ua"] for p in points], dtype=float)
    ug1_raw = np.array([p["ug1"] for p in points], dtype=float)
    ia_mA_raw = np.array([p["ia"] for p in points], dtype=float)

    if topology == TOPOLOGY_TRIODE:
        mask = ia_mA_raw > ia_thr_mA
        n_kept = int(mask.sum())
        if n_kept < min_count:
            raise RuntimeError(
                f"Not enough valid data points "
                f"(Ia > {ia_thr_mA} mA, need {min_count}, got {n_kept})"
            )
        ua = ua_raw[mask]
        ug1 = ug1_raw[mask]
        ia = ia_mA_raw[mask] / _MA_PER_A
        return ExtractedFitData(
            ua=ua, ug1=ug1, ug2=np.zeros_like(ua), ia=ia,
            ig2=None, n_points=n_kept, has_ig2=False,
        )

    # Pentode / triode_connected: include Ug2 + Ig2
    if topology == TOPOLOGY_TRIODE_CONNECTED:
        # Ug2 may be missing — default to Ua (Dempwolf convention)
        ug2_raw = np.array(
            [p.get("ug2", p["ua"]) for p in points], dtype=float)
    else:
        ug2_raw = np.array(
            [p.get("ug2", 0.0) for p in points], dtype=float)
    ig2_mA_raw = np.array(
        [p.get("ig2", 0.0) for p in points], dtype=float)

    mask = (ia_mA_raw > ia_thr_mA) & (ug2_raw > ug2_min_v)
    n_kept = int(mask.sum())
    if n_kept < min_count:
        raise RuntimeError(
            f"Not enough valid pentode data points "
            f"(Ia > {ia_thr_mA} mA, Ug2 > {ug2_min_v} V, "
            f"need {min_count}, got {n_kept})"
        )

    ua = ua_raw[mask]
    ug1 = ug1_raw[mask]
    ug2 = ug2_raw[mask]
    ia = ia_mA_raw[mask] / _MA_PER_A
    ig2_A = ig2_mA_raw[mask] / _MA_PER_A

    has_ig2 = bool(np.any(ig2_A > ig2_eps_A))
    ig2_out = ig2_A if has_ig2 else None

    return ExtractedFitData(
        ua=ua, ug1=ug1, ug2=ug2, ia=ia,
        ig2=ig2_out, n_points=n_kept, has_ig2=has_ig2,
    )


def compute_fit_errors(
    pred_A: np.ndarray, meas_A: np.ndarray,
) -> Tuple[float, float]:
    """Return ``(rms_mA, max_mA)`` from prediction-vs-measured arrays.

    Both inputs in Amperes; output in mA (multiplied by 1000).
    """
    diff_mA = (pred_A - meas_A) * _MA_PER_A
    rms = float(np.sqrt(np.mean(diff_mA ** 2)))
    max_err = float(np.max(np.abs(diff_mA)))
    return rms, max_err


def build_fit_result(
    *,
    model_type: str,
    topology: str,
    model: TubeModelProtocol,
    ia_pred_A: np.ndarray,
    ia_meas_A: np.ndarray,
    n_points: int,
    ig2_pred_A: Optional[np.ndarray] = None,
    ig2_meas_A: Optional[np.ndarray] = None,
    converged: bool = True,
    warnings: Optional[List[Dict]] = None,
) -> ModelFitResult:
    """Compute Ia (and optional Ig2) errors and pack into ``ModelFitResult``.

    Pass ``ig2_pred_A`` and ``ig2_meas_A`` together to record Ig2 fit quality.
    Either both ``None`` or both provided.

    ``converged=False`` means at least one phase of the fit hit scipy's
    ``max_nfev`` without satisfying the convergence tolerance — the
    returned ``params`` may be far from the true optimum. The flag is
    surfaced in the UI as a "partial fit" badge.
    """
    rms_error, max_error = compute_fit_errors(ia_pred_A, ia_meas_A)

    rms_ig2: Optional[float] = None
    max_ig2: Optional[float] = None
    if ig2_pred_A is not None and ig2_meas_A is not None:
        rms_ig2, max_ig2 = compute_fit_errors(ig2_pred_A, ig2_meas_A)

    # ia_meas_A is in Amperes; mean_ia for quality check should be in mA
    # (matching rms_error which is also in mA per compute_fit_errors).
    mean_ia_mA = float(np.mean(np.abs(ia_meas_A))) * _MA_PER_A
    rms_pct, quality = compute_fit_quality(rms_error, mean_ia_mA)

    return ModelFitResult(
        model_type=model_type,
        topology=topology,
        params=model.params_dict(),
        rms_error=rms_error,
        max_error=max_error,
        n_points=n_points,
        model=model,
        rms_ig2=rms_ig2,
        max_ig2=max_ig2,
        converged=converged,
        rms_pct=rms_pct,
        quality=quality,
        warnings=list(warnings) if warnings else [],
    )
