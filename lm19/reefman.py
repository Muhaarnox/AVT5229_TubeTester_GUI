"""Reefman (Derk/DerkE) tube models — eval, fit, registry.

Implements 6 model variants from Reefman Theory.pdf (2016) and TubeLib.inc:
  - PenthodeD / PenthodeDE        — true pentodes (Derk / DerkE splitting)
  - BTetrodeD / BTetrodeDE         — beam tetrodes (+ secondary emission)
  - PenthodeVD / PenthodeVDE       — variable-mu pentodes (+ blended cathode)

Equations verified against TubeLib.inc SPICE subcircuits.

Usage:
    from lm19.reefman import fit_reefman, load_reefman_model

No Qt dependencies.
"""


from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

import numpy as np
from scipy.optimize import least_squares

from lm19.tube_model_base import (
    TubeModelProtocol, ModelFitResult, register_model,
    extract_arrays, build_fit_result, check_fit_convergence,
)
from lm19.tube_params import ReefmanParams, lookup_tube, list_tubes
from lm19.tube_sim import ScanGrid
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)
from lm19.tube_model_base import (
    MODEL_TYPE_REEFMAN,
)
from lm19.tube_model_base import (
    MODEL_WARN_REEFMAN_FEW_UG2,
)


# Clip argument to exp() to prevent float64 overflow (exp overflows at ~709;
# 700 keeps headroom). A tight upper cap (e.g. +80)
# would saturate the softplus in-range for kp/mu > 80 fits (same bug class as
# the old ±50 clips in dempwolf.py / spice_export/koren.py; no benchmark
# dataset was observed in the zone thanks to the sqrt(Kvb+Vg2^2) denominator,
# fixed for consistency). The SPICE E1/E11 lines carry a mirroring
# MIN(...,700). Pinned in tests/test_model_exp_clip_pins.py.
_EXP_CLIP = 700.0


# ---------------------------------------------------------------------------
# Vectorized eval functions (numpy)
# ---------------------------------------------------------------------------

def _koren_cathode(vg2: np.ndarray, vg1: np.ndarray,
                   mu: float, ex: float, kp: float,
                   kvb: float) -> np.ndarray:
    """Koren cathode current Ip (Reefman variant).

    E1 = (Vg2/Kp) * ln(1 + exp(Kp * (1/mu + Vg1 / sqrt(Kvb + Vg2^2))))
    Ip = E1^Ex                   [for E1 ≥ 0; clamped to 0 below]

    The canonical Koren form is ``Ip = (E1^Ex + |E1|^Ex) / 2`` (a
    half-wave rectifier: equals E1^Ex when E1 > 0, zero when E1 < 0).
    The LTspice subcircuit (lm19/spice_export/reefman.py) renders that
    literally as ``0.5 * (PWR(E1, Ex) + PWRS(E1, Ex))`` because SPICE
    has no clamp primitive. Python clamps E1 to ≥0 first, so the two
    terms collapse to the same value and the half-wave averaging is
    unnecessary. Single ``np.power`` is identical numerically and
    halves the per-evaluation work in the inner fit loop.

    Note: uses sqrt(Kvb + Vg2^2) instead of Koren's Vg2 in denominator.
    Returns Ip in arbitrary units (needs /Kg1 or splitting for actual current).
    """
    denom = np.sqrt(kvb + vg2 ** 2)
    arg = np.clip(kp * (1.0 / mu + vg1 / denom), -_EXP_CLIP, _EXP_CLIP)
    e1 = (vg2 / kp) * np.log1p(np.exp(arg))
    e1_pos = np.maximum(e1, 0.0)
    return np.power(e1_pos, ex)


def _koren_cathode_vmu(vg2: np.ndarray, vg1: np.ndarray,
                       mu: float, ex: float, kp: float, kvb: float,
                       mu_b: float, ex_b: float,
                       svar: float) -> np.ndarray:
    """Variable-mu cathode current: blend of two Koren sections.

    Ip = (1 - svar) * Ip(mu, ex) + svar * Ip(mu_b, ex_b)
    """
    ip1 = _koren_cathode(vg2, vg1, mu, ex, kp, kvb)
    ip2 = _koren_cathode(vg2, vg1, mu_b, ex_b, kp, kvb)
    return (1.0 - svar) * ip1 + svar * ip2


def _derk_ia_ig2(va: np.ndarray, vg1: np.ndarray, vg2: np.ndarray,
                 p: ReefmanParams) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Ia and Ig2 for any Reefman model variant.

    Dispatches to the correct splitting and cathode functions based on p.type.
    Returns (Ia, Ig2) in Amperes.
    """
    # Cathode current
    is_vmu = p.type in ("PenthodeVD", "PenthodeVDE")
    if is_vmu and p.mu_b is not None and p.ex_b is not None:
        ip = _koren_cathode_vmu(vg2, vg1, p.mu, p.ex, p.kp, p.kvb,
                                p.mu_b, p.ex_b, p.svar)
    else:
        ip = _koren_cathode(vg2, vg1, p.mu, p.ex, p.kp, p.kvb)

    va_safe = np.maximum(va, 0.001)

    # Derived alpha
    alpha = 1.0 - (p.kg1 / p.kg2) * (1.0 + p.als)

    # Precomputed SPICE terms
    ookg1_m_ookg2 = 1.0 / p.kg1 - 1.0 / p.kg2
    aokg1 = p.A / p.kg1
    alkg1_p_alskg2 = alpha / p.kg1 + p.als / p.kg2

    # Splitting function
    is_de = p.type in ("BTetrodeDE", "PenthodeDE", "PenthodeVDE")
    if is_de:
        # DerkE: exp(-(β·Va)^1.5)
        f_va = np.exp(-np.power(p.be * va_safe, 1.5))
    else:
        # Derk: 1 / (1 + β·Va)
        f_va = 1.0 / (1.0 + p.be * va_safe)

    # Anode current splitting factor
    ia_split = ookg1_m_ookg2 + aokg1 * va_safe - alkg1_p_alskg2 * f_va

    # Screen current
    ig2 = ip / p.kg2 * (1.0 + p.als * f_va)

    # Anode current
    ia = ip * ia_split

    # Secondary emission (BTetrode types only)
    is_bt = p.type in ("BTetrodeD", "BTetrodeDE")
    if is_bt and p.Sc > 0:
        # Psec = Sc/Kg2 * Va * (1 + tanh(-ap*(Va - Vg2/lam + w + nu*Vg1)))
        # Theory.pdf eq. (43)-(46): the secondary-emission current LEAVES
        # the anode and ARRIVES at the screen — Psec enters Ig2 with "+",
        # so Ia + Ig2 stays independent of Sc (constant space current).
        # NB Reefman's own TubeLib.inc G2 sources omit the term (his
        # library diverges from his paper); LM19 follows the paper
        # — only loaded Sc>0 reference models are affected,
        # fit_reefman never fits Sc. Pinned in test_reefman_paper_pins.py.
        lam_safe = max(p.lam, 0.001)
        psec_arg = -p.ap * (va_safe - vg2 / lam_safe + p.w + p.nu * vg1)
        psec = (p.Sc / p.kg2) * va_safe * (1.0 + np.tanh(psec_arg))
        ia = ia - ip * psec
        ig2 = ig2 + ip * psec

    # Ia is NOT clamped to >= 0: in the dynatron region (Va well below the
    # crossover, secondary yield > 1) the net anode current physically
    # reverses — the paper (eq. 44/46), Reefman's own TubeLib.inc G1
    # sources and our SPICE export all permit Ia < 0, as does
    # DempwolfModel in its kink (the old np.maximum(ia, 0)
    # silently diverged from all three exactly there). Note the splitting
    # algebra: alpha/kg1 + als/kg2 == 1/kg1 - 1/kg2, so for fitted models
    # (Sc = 0) ia = ip*((1/kg1 - 1/kg2)*(1 - f) + A*va/kg1) — negative
    # only for nonphysical kg1 > kg2 trials. Ig2 stays clamped (every
    # term is non-negative anyway).
    return ia, np.maximum(ig2, 0.0)


# ---------------------------------------------------------------------------
# ReefmanModel — satisfies TubeModelProtocol
# ---------------------------------------------------------------------------

@dataclass
class ReefmanModel:
    """Tube model using Reefman (Derk/DerkE) equations.

    Satisfies TubeModelProtocol. Drop-in replacement for TubeModel (Koren)
    and DempwolfModel.
    """
    name: str
    topology: str
    reefman: ReefmanParams
    uh: float = 6.3
    ih: float = 0.3
    pa_max: float = 12.5
    model_type: str = MODEL_TYPE_REEFMAN

    def ia(self, ua: float, ug1: float, ug2: float = 0.0) -> float:
        """Compute Ia (mA) at a single operating point."""
        va = np.array([ua], dtype=float)
        vg1 = np.array([ug1], dtype=float)
        vg2_arr = np.array([ug2], dtype=float)
        ia_a, _ = _derk_ia_ig2(va, vg1, vg2_arr, self.reefman)
        return float(ia_a[0]) * 1000.0

    def ia_array(self, ua, ug1, ug2=0.0) -> np.ndarray:
        """Vectorized Ia (mA) over broadcastable arrays — the Derk/DerkE
        kernel is already array-native; scalar ``ia()`` just wraps it in
        1-element arrays (optimizer hot path)."""
        va, vg1, vg2 = np.broadcast_arrays(
            np.asarray(ua, dtype=float),
            np.asarray(ug1, dtype=float),
            np.asarray(ug2, dtype=float),
        )
        ia_a, _ = _derk_ia_ig2(va, vg1, vg2, self.reefman)
        return np.asarray(ia_a, dtype=float) * 1000.0

    def ig2(self, ua: float, ug1: float, ug2: float) -> float:
        """Compute Ig2 (mA). Returns 0 for triodes."""
        if self.topology == TOPOLOGY_TRIODE:
            return 0.0
        va = np.array([ua], dtype=float)
        vg1 = np.array([ug1], dtype=float)
        vg2_arr = np.array([ug2], dtype=float)
        _, ig2_a = _derk_ia_ig2(va, vg1, vg2_arr, self.reefman)
        return float(ig2_a[0]) * 1000.0

    def params_dict(self) -> Dict:
        """Return Reefman parameters as a plain dict."""
        d = {
            "type": self.reefman.type,
            "mu": self.reefman.mu,
            "ex": self.reefman.ex,
            "kg1": self.reefman.kg1,
            "kg2": self.reefman.kg2,
            "kp": self.reefman.kp,
            "kvb": self.reefman.kvb,
            "als": self.reefman.als,
            "be": self.reefman.be,
            "A": self.reefman.A,
        }
        if self.reefman.type in ("BTetrodeD", "BTetrodeDE"):
            d.update(Sc=self.reefman.Sc, ap=self.reefman.ap,
                     w=self.reefman.w, nu=self.reefman.nu,
                     lam=self.reefman.lam)
        if self.reefman.mu_b is not None:
            d.update(mu_b=self.reefman.mu_b, ex_b=self.reefman.ex_b,
                     svar=self.reefman.svar)
        return d

    def generate_scan(self, grid: ScanGrid) -> List[Dict]:
        """Generate measurement points matching LM19 scan format."""
        points: List[Dict] = []
        ua_values = np.arange(
            grid.ua[0], grid.ua[1] + grid.ua[2] * 0.5, grid.ua[2])
        ug1_values = np.arange(
            grid.ug1[0], grid.ug1[1] + grid.ug1[2] * 0.5, grid.ug1[2])

        if grid.ug2_track_ua:
            ug2_values = [0.0]  # placeholder, actual = ua + offset
        elif grid.ug2 is not None:
            ug2_values = np.arange(
                grid.ug2[0], grid.ug2[1] + grid.ug2[2] * 0.5, grid.ug2[2])
        else:
            ug2_values = [250.0]

        for ug2_nom in ug2_values:
            for ug1 in ug1_values:
                for ua in ua_values:
                    if grid.ug2_track_ua:
                        ug2_actual = float(ua) + grid.ug2_offset
                    else:
                        ug2_actual = float(ug2_nom)

                    ia_val = self.ia(float(ua), float(ug1), ug2_actual)
                    ig2_val = self.ig2(float(ua), float(ug1), ug2_actual)
                    points.append({
                        "ua": float(ua), "ug1": float(ug1),
                        "ug2": ug2_actual,
                        "ia": ia_val, "ig2": ig2_val,
                        "uh": grid.uh, "ih": grid.ih,
                    })
        return points


# ---------------------------------------------------------------------------
# Loader (reference mode)
# ---------------------------------------------------------------------------

def load_reefman_model(tube_name: str) -> Optional[ReefmanModel]:
    """Load a Reefman model from tube_params.json by name or alias.

    Returns ReefmanModel or None if tube has no Reefman parameters.
    """
    ref = lookup_tube(tube_name)
    if ref is None or ref.reefman is None:
        return None
    return ReefmanModel(
        name=ref.name,
        topology=ref.topology,
        reefman=ref.reefman,
    )


def _list_reefman_tubes() -> List[str]:
    """List tubes that have Reefman parameters."""
    all_tubes = list_tubes()
    result = []
    for name in all_tubes:
        ref = lookup_tube(name)
        if ref and ref.reefman is not None:
            result.append(name)
    return result


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

def fit_reefman(points: List[Dict], topology: str) -> ModelFitResult:
    """Fit Reefman (Derk/DerkE) model to measured data.

    Tries both Derk (D) and DerkE (DE) splitting, returns the better fit.
    Uses 2-phase fitting: cathode current first, then splitting.

    Args:
        points: measurement dicts with ua, ug1, ug2, ia, ig2 keys.
        topology: "triode", "pentode", or "triode_connected".

    Returns:
        ModelFitResult with fitted ReefmanModel.

    Raises:
        RuntimeError: if not enough valid data points.
    """
    if topology in (TOPOLOGY_TRIODE, TOPOLOGY_TRIODE_CONNECTED):
        raise RuntimeError("Reefman models require pentode data "
                           "(independent Ug2)")

    data = extract_arrays(
        points, topology=TOPOLOGY_PENTODE, ia_thr_mA=0.1, min_count=10,
    )
    va, vg1, vg2 = data.ua, data.ug1, data.ug2
    # Reefman's inner functions take mA arrays for ia_meas / ig2_meas;
    # `_fit_variant` and the final error calculation below both expect
    # this convention, so reconstruct mA arrays from the A-unit `data`.
    ia_meas = data.ia * 1000.0
    ig2_A_full = data.ig2 if data.ig2 is not None else np.zeros_like(data.ia)
    ig2_meas = ig2_A_full * 1000.0
    n_pts = data.n_points

    # Reefman current splitting works best with many Ug2 levels.
    # With < 3 Ug2 levels, splitting parameters may not converge reliably.
    fit_warnings: list = []
    ug2_levels = len(set(np.round(vg2, 0)))
    if ug2_levels < 3:
        log.warning(
            "Reefman: only %d Ug2 levels (recommend >= 3). "
            "Current splitting may be inaccurate.", ug2_levels,
        )
        fit_warnings.append({"code": MODEL_WARN_REEFMAN_FEW_UG2, "n": ug2_levels})

    # Reefman's `has_ig2` threshold (0.1 mA) is stricter than the generic
    # `extract_arrays` default (1e-5 A = 0.01 mA). Re-evaluate.
    has_ig2 = bool(np.any(ig2_meas > 0.1))

    # Try both D and DE, pick best
    best_result = None
    best_rms = np.inf

    for variant in ("D", "DE"):
        try:
            params, rms_ia, converged = _fit_variant(
                va, vg1, vg2, ia_meas, ig2_meas, has_ig2, variant)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            # _fit_variant raises RuntimeError if all retry starts in
            # phase1/2 failed (no scipy convergence on this variant).
            # Other ValueError/LinAlgError come from numpy on bad data.
            # Programming errors (AttributeError, TypeError, NameError)
            # propagate so a refactor regression isn't masked into
            # "Reefman fitting failed for both D and DE variants".
            log.warning(
                "reefman variant=%s fit failed: %s: %s",
                variant, type(e).__name__, e,
            )
            continue
        if rms_ia < best_rms:
            best_rms = rms_ia
            best_result = (params, rms_ia, variant, converged)

    if best_result is None:
        raise RuntimeError("Reefman fitting failed for both D and DE variants")

    params, rms_ia, variant, converged = best_result

    # Compute final errors via shared helper. _derk_ia_ig2 returns A-unit
    # arrays; convert ia_meas / ig2_meas back to A for build_fit_result.
    rp = _params_to_reefman(params, "BTetrode" + variant)
    ia_pred, ig2_pred = _derk_ia_ig2(va, vg1, vg2, rp)
    model = ReefmanModel(name="fit", topology=TOPOLOGY_PENTODE, reefman=rp)

    return build_fit_result(
        warnings=fit_warnings,
        model_type=MODEL_TYPE_REEFMAN,
        topology=TOPOLOGY_PENTODE,
        model=model,
        ia_pred_A=ia_pred,
        ia_meas_A=ia_meas / 1000.0,
        n_points=n_pts,
        ig2_pred_A=ig2_pred if has_ig2 else None,
        ig2_meas_A=(ig2_meas / 1000.0) if has_ig2 else None,
        converged=converged,
    )


def _fit_variant(va, vg1, vg2, ia_meas, ig2_meas, has_ig2,
                 variant: str) -> Tuple[np.ndarray, float, bool]:
    """Fit one variant (D or DE).

    Returns ``(params_array, rms_ia_mA, all_converged)``. ``all_converged``
    is False iff the final phase-3 combined fit hit max_nfev. Phase 1/2
    retry-loops are aggregated separately (their per-start divergence is
    expected — the loop picks the lowest-cost start), so they do NOT
    affect this flag.
    """

    # Work in Amperes
    ia_a = ia_meas / 1000.0
    ig2_a = ig2_meas / 1000.0

    # --- Phase 1: cathode current ---
    # Total current ≈ Ip * (1/Kg1)  (rough: ignoring splitting details)
    # Parameters: [mu, ex, kg1, kp, kvb]
    i_total = ia_a + ig2_a if has_ig2 else ia_a

    def residual_cathode(x):
        mu, ex, kg1, kp, kvb = x
        ip = _koren_cathode(vg2, vg1, mu, ex, kp, kvb)
        pred = ip / kg1
        return (pred - i_total) * 1000.0  # residual in mA

    # Initial guess: try from TubeLib-like values
    x0_cathode = [20.0, 1.3, 200.0, 100.0, 1000.0]
    lo_cathode = [3.0, 0.8, 10.0, 5.0, 0.0]
    hi_cathode = [500.0, 2.5, 100000.0, 2000.0, 50000.0]

    best_phase1 = None
    best_cost1 = np.inf

    for kp_init in [30.0, 80.0, 150.0, 300.0, 600.0]:
        x0_try = list(x0_cathode)
        x0_try[3] = kp_init
        try:
            res = least_squares(residual_cathode, x0_try,
                                bounds=(lo_cathode, hi_cathode),
                                method="trf", max_nfev=5000)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            log.warning(
                "reefman phase1 (cathode) start kp=%.0f failed: %s: %s",
                kp_init, type(e).__name__, e,
            )
            continue
        if not getattr(res, "success", True):
            log.debug(
                "reefman phase1 (cathode) start kp=%.0f did not converge "
                "(status=%s, cost=%.4g)",
                kp_init, getattr(res, "status", None), res.cost,
            )
        if res.cost < best_cost1:
            best_cost1 = res.cost
            best_phase1 = res.x

    if best_phase1 is None:
        raise RuntimeError("Phase 1 (cathode) failed")

    mu, ex, kg1, kp, kvb = best_phase1

    # --- Phase 2: splitting parameters ---
    # Parameters: [kg2, A, als, be]
    def residual_split(x):
        kg2, A_val, als, be = x
        p = ReefmanParams(
            type="BTetrode" + variant,
            mu=mu, ex=ex, kg1=kg1, kg2=kg2,
            kp=kp, kvb=kvb, als=als, be=be, A=A_val)
        ia_pred, ig2_pred = _derk_ia_ig2(va, vg1, vg2, p)
        r_ia = (ia_pred * 1000.0 - ia_meas)
        if has_ig2:
            r_ig2 = (ig2_pred * 1000.0 - ig2_meas)
            return np.concatenate([r_ia, 0.3 * r_ig2])
        return r_ia

    x0_split = [2000.0, 0.001, 3.0, 0.1]
    lo_split = [10.0, 0.0, 0.001, 0.0001]
    hi_split = [100000.0, 0.5, 30.0, 5.0]

    best_phase2 = None
    best_cost2 = np.inf

    for als_init in [1.0, 3.0, 7.0, 15.0]:
        x0_try = list(x0_split)
        x0_try[2] = als_init
        try:
            res = least_squares(residual_split, x0_try,
                                bounds=(lo_split, hi_split),
                                method="trf", max_nfev=5000)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            log.warning(
                "reefman phase2 (split) start als=%.1f failed: %s: %s",
                als_init, type(e).__name__, e,
            )
            continue
        if not getattr(res, "success", True):
            log.debug(
                "reefman phase2 (split) start als=%.1f did not converge "
                "(status=%s, cost=%.4g)",
                als_init, getattr(res, "status", None), res.cost,
            )
        if res.cost < best_cost2:
            best_cost2 = res.cost
            best_phase2 = res.x

    if best_phase2 is None:
        raise RuntimeError("Phase 2 (splitting) failed")

    kg2, A_val, als, be = best_phase2

    # --- Phase 3: combined optimization (all 9 params) ---
    x0_combined = [mu, ex, kg1, kg2, kp, kvb, A_val, als, be]
    lo_combined = [3.0, 0.8, 10.0, 10.0, 5.0, 0.0, 0.0, 0.001, 0.0001]
    hi_combined = [500.0, 2.5, 100000.0, 100000.0, 2000.0, 50000.0,
                   0.5, 30.0, 5.0]

    def residual_combined(x):
        mu_, ex_, kg1_, kg2_, kp_, kvb_, A_, als_, be_ = x
        p = ReefmanParams(
            type="BTetrode" + variant,
            mu=mu_, ex=ex_, kg1=kg1_, kg2=kg2_,
            kp=kp_, kvb=kvb_, als=als_, be=be_, A=A_)
        ia_pred, ig2_pred = _derk_ia_ig2(va, vg1, vg2, p)
        r_ia = (ia_pred * 1000.0 - ia_meas)
        if has_ig2:
            r_ig2 = (ig2_pred * 1000.0 - ig2_meas)
            return np.concatenate([r_ia, 0.3 * r_ig2])
        return r_ia

    res = least_squares(residual_combined, x0_combined,
                        bounds=(lo_combined, hi_combined),
                        method="trf", max_nfev=10000)
    check_fit_convergence(res, "reefman phase3 (combined)", log,
                          n_points=len(va), variant=variant)
    converged = bool(getattr(res, "success", True))

    params = res.x
    # Compute RMS Ia
    p_final = _params_to_reefman(params, "BTetrode" + variant)
    ia_pred, _ = _derk_ia_ig2(va, vg1, vg2, p_final)
    diff = ia_pred * 1000.0 - ia_meas
    rms_ia = float(np.sqrt(np.mean(diff ** 2)))

    return params, rms_ia, converged


def _params_to_reefman(x: np.ndarray, model_type: str) -> ReefmanParams:
    """Convert 9-element parameter array to ReefmanParams."""
    mu, ex, kg1, kg2, kp, kvb, A_val, als, be = x
    return ReefmanParams(
        type=model_type,
        mu=float(mu), ex=float(ex),
        kg1=float(kg1), kg2=float(kg2),
        kp=float(kp), kvb=float(kvb),
        als=float(als), be=float(be), A=float(A_val),
    )


# ---------------------------------------------------------------------------
# Register in MODEL_REGISTRY
# ---------------------------------------------------------------------------

register_model(
    model_type=MODEL_TYPE_REEFMAN,
    label="Reefman (Derk/DerkE)",
    loader=load_reefman_model,
    fitter=fit_reefman,
    list_tubes=_list_reefman_tubes,
)
