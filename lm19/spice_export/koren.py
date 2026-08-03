"""Koren tube model — math, fitters, subcircuit generation, fit pipeline.

Equations and parameter tables from "Norman Koren — Improved Vacuum Tube
Models for SPICE Simulations", see SOURCES_INDEX.md.

Canonical Koren triode model (Norman Koren, 1996):
    E1 = (Ua / Kp) * log(1 + exp(Kp * (1/mu + Ug1 / sqrt(Kvb + Ua^2))))
    Ia = 2 * E1^Ex / Kg1

Canonical Koren pentode model (Norman Koren, 1996):
    E1 = (Ug2 / Kp) * log(1 + exp(Kp * (1/mu + Ug1 / Ug2)))
    Ia = 2 * E1^Ex / Kg1 * arctan(Ua / Kvb)
    Ig2 = (Ug1 + Ug2/mu)^Ex / Kg2

Fitters:
    - scipy (preferred): Trust Region Reflective least_squares
    - numpy fallback: coordinate descent with golden-section line search
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from lm19.spice_export._common import SpiceFitResult, _HAS_SCIPY

if TYPE_CHECKING:
    from lm19.tube_params import KorenParams
from lm19.tube_model_base import (
    check_fit_convergence, compute_fit_errors, extract_arrays,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)

if _HAS_SCIPY:
    from scipy.optimize import least_squares as _least_squares

log = logging.getLogger(__name__)


# ── module local constants ──
# RGI fallback used only when a pentode ref explicitly disables rgi (rgi=0) but
# caps are present — keeps the {RGI} grid-stopper parameter declared. Pentode
# historically used 1000 Ω (triode declares ref.rgi directly, no fallback).
_DEFAULT_PENTODE_RGI_OHM = 1000

# Clip argument to exp() to prevent float64 overflow (exp overflows at ~709;
# 700 keeps headroom). A tight clip (e.g. ±50) SATURATES the
# Kp-normalized softplus in-range: arg = Kp*(1/mu + Vg/...) exceeds 50 near
# Vg = 0 for high-Kp/low-mu fits — 3 real benchmark datasets were affected
# (10_VT25 max_arg=77, 801_VT62 62, 6N5P 234 at fitted kp=1000), gm flattened
# to ~0 there and the (unclipped) SPICE export diverged from Python. Same fix
# and rationale as lm19/dempwolf.py _EXP_CLIP; the SPICE E1 lines carry a
# mirroring MIN(...,700). Pinned in tests/test_model_exp_clip_pins.py.
_EXP_CLIP = 700.0


# ─── TRIODE MATH ──────────────────────────────────────────────────────

def _koren_ia(ua: np.ndarray, ug1: np.ndarray, mu: float, ex: float,
              kg1: float, kp: float, kvb: float) -> np.ndarray:
    """Compute Ia (amps) using canonical Koren triode model (vectorized).

    E1 = (Ua/Kp) * log(1 + exp(Kp * (1/mu + Ug1 / sqrt(Kvb + Ua^2))))
    Ia = 2 * E1^Ex / Kg1  (for E1 > 0, else 0)
    """
    ua = np.maximum(ua, 0.01)
    arg = kp * (1.0 / mu + ug1 / np.sqrt(kvb + ua * ua))
    arg = np.clip(arg, -_EXP_CLIP, _EXP_CLIP)
    e1 = (ua / kp) * np.log1p(np.exp(arg))
    abs_e1 = np.maximum(np.abs(e1), 0.0)
    pwr = np.power(abs_e1, ex)
    pwrs = np.sign(e1) * pwr
    ia = (pwr + pwrs) / kg1
    return np.maximum(ia, 0.0)


# ─── TRIODE FITTERS ───────────────────────────────────────────────────

def _make_initial_guess(ua: np.ndarray, ug1: np.ndarray,
                        ref_koren: Optional["KorenParams"] = None) -> List[float]:
    """Build triode initial guess [mu, ex, kg1, kp, kvb]."""
    if ref_koren is not None:
        log.info(
            "Using reference Koren params as initial guess: "
            "mu=%.1f Ex=%.2f Kg1=%.1f Kp=%.1f Kvb=%.1f",
            ref_koren.mu, ref_koren.ex, ref_koren.kg1,
            ref_koren.kp, ref_koren.kvb,
        )
        return [ref_koren.mu, ref_koren.ex, ref_koren.kg1,
                ref_koren.kp, ref_koren.kvb]
    ug1_range = float(np.max(ug1) - np.min(ug1))
    ua_mean = float(np.mean(ua))
    mu0 = max(2.0, min(200.0, abs(ua_mean / max(abs(ug1_range), 1.0))))
    if mu0 < 8.0:
        return [mu0, 1.3, 200.0, 20.0, 300.0]
    return [mu0, 1.4, 1060.0, 600.0, 300.0]


def _fit_koren_scipy(ua: np.ndarray, ug1: np.ndarray, ia_meas: np.ndarray,
                     ref_koren: Optional["KorenParams"] = None,
                     ) -> Tuple[np.ndarray, float, bool]:
    """Fit Koren triode model using scipy least_squares (TRF)."""
    x0 = _make_initial_guess(ua, ug1, ref_koren)

    mu_est = x0[0]
    if mu_est < 8.0:
        bounds_lo = [1.5, 0.8, 10.0, 5.0, 5.0]
        bounds_hi = [15.0, 3.5, 50000.0, 200.0, 3000.0]
    elif mu_est < 30.0:
        bounds_lo = [2.0, 0.8, 10.0, 10.0, 5.0]
        bounds_hi = [100.0, 3.5, 50000.0, 1000.0, 3000.0]
    else:
        bounds_lo = [2.0, 0.8, 10.0, 20.0, 5.0]
        bounds_hi = [500.0, 3.5, 50000.0, 3000.0, 3000.0]

    # ML-066: reference params may lie outside the adaptive bounds
    # (e.g. an exotic kg1) — scipy raises "x0 is infeasible". Clip.
    x0 = list(np.clip(x0, bounds_lo, bounds_hi))

    def residual_vec(p):
        ia_pred = _koren_ia(ua, ug1, p[0], p[1], p[2], p[3], p[4])
        return ia_pred - ia_meas

    result = _least_squares(
        residual_vec, x0,
        bounds=(bounds_lo, bounds_hi),
        method="trf",
        max_nfev=5000,
    )
    check_fit_convergence(result, "koren scipy triode", log,
                          n_points=len(ua))
    params = np.array(result.x)
    cost = float(result.cost)
    return params, cost * 2.0, bool(result.success)


def _residuals(ua: np.ndarray, ug1: np.ndarray, ia_meas: np.ndarray,
               params: List[float]) -> float:
    """Compute triode sum of squared residuals (A^2)."""
    mu, ex, kg1, kp, kvb = params
    ia_pred = _koren_ia(ua, ug1, mu, ex, kg1, kp, kvb)
    return float(np.sum((ia_pred - ia_meas) ** 2))


def _fit_koren_numpy(ua: np.ndarray, ug1: np.ndarray, ia_meas: np.ndarray,
                     ref_koren: Optional["KorenParams"] = None,
                     ) -> Tuple[List[float], float, bool]:
    """Fit Koren triode model using coordinate descent (numpy only)."""
    x0 = _make_initial_guess(ua, ug1, ref_koren)
    params = np.array(x0)
    mu_est = x0[0]
    if mu_est < 8.0:
        bounds_lo = np.array([1.5, 0.8, 10.0, 5.0, 5.0])
        bounds_hi = np.array([15.0, 3.5, 50000.0, 200.0, 3000.0])
    elif mu_est < 30.0:
        bounds_lo = np.array([2.0, 0.8, 10.0, 10.0, 5.0])
        bounds_hi = np.array([100.0, 3.5, 50000.0, 1000.0, 3000.0])
    else:
        bounds_lo = np.array([2.0, 0.8, 10.0, 20.0, 5.0])
        bounds_hi = np.array([500.0, 3.5, 50000.0, 3000.0, 3000.0])
    best_cost = _residuals(ua, ug1, ia_meas, params)
    best_params = params.copy()
    gr = (np.sqrt(5) + 1) / 2

    for _ in range(30):
        improved = False
        for dim in range(5):
            lo, hi = bounds_lo[dim], bounds_hi[dim]
            a, b = lo, hi
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            for _ in range(40):
                trial_c = params.copy(); trial_c[dim] = c
                trial_d = params.copy(); trial_d[dim] = d
                fc = _residuals(ua, ug1, ia_meas, trial_c)
                fd = _residuals(ua, ug1, ia_meas, trial_d)
                if fc < fd:
                    b = d
                else:
                    a = c
                c = b - (b - a) / gr
                d = a + (b - a) / gr
                if abs(b - a) < 1e-6 * (bounds_hi[dim] - bounds_lo[dim]):
                    break
            opt_val = (a + b) / 2.0
            trial = params.copy(); trial[dim] = opt_val
            cost = _residuals(ua, ug1, ia_meas, trial)
            if cost < best_cost:
                best_cost = cost
                params[dim] = opt_val
                best_params = params.copy()
                improved = True
        if not improved:
            break
    return best_params, best_cost, True


# ─── PENTODE MATH ─────────────────────────────────────────────────────

def _koren_ia_pentode(ua: np.ndarray, ug1: np.ndarray, ug2: np.ndarray,
                      mu: float, ex: float, kg1: float, kp: float,
                      kvb: float) -> np.ndarray:
    """Compute Ia (amps) using Koren pentode model (vectorized).

    E1 = (Ug2/Kp) * log(1 + exp(Kp * (1/mu + Ug1/Ug2)))
    Ia = 2 * E1^Ex / Kg1 * arctan(Ua/Kvb)
    """
    ua = np.maximum(ua, 0.01)
    ug2 = np.maximum(ug2, 0.01)
    arg = kp * (1.0 / mu + ug1 / ug2)
    arg = np.clip(arg, -_EXP_CLIP, _EXP_CLIP)
    e1 = (ug2 / kp) * np.log1p(np.exp(arg))
    abs_e1 = np.maximum(np.abs(e1), 0.0)
    pwr = np.power(abs_e1, ex)
    pwrs = np.sign(e1) * pwr
    ia = (pwr + pwrs) / kg1 * np.arctan(ua / kvb)
    return np.maximum(ia, 0.0)


def _koren_ig2_pentode(ug1: np.ndarray, ug2: np.ndarray, mu: float,
                       ex: float, kg2: float) -> np.ndarray:
    """Compute Ig2 (amps) using Koren pentode screen current model.

    Ig2 = (Ug1 + Ug2/mu)^Ex / Kg2  for (Ug1 + Ug2/mu) > 0
    """
    v = ug1 + ug2 / mu
    v_pos = np.maximum(v, 0.0)
    return np.power(v_pos, ex) / kg2


# ─── PENTODE FITTERS ──────────────────────────────────────────────────

def _make_pentode_initial_guess(ua: np.ndarray, ug1: np.ndarray,
                                ug2: np.ndarray,
                                ref_koren: Optional["KorenParams"] = None,
                                ) -> List[float]:
    """Build pentode initial guess [mu, ex, kg1, kp, kvb, kg2]."""
    if ref_koren is not None:
        kg2 = ref_koren.kg2 if ref_koren.kg2 else 4500.0
        log.info(
            "Using reference Koren params as initial guess: "
            "mu=%.1f Ex=%.2f Kg1=%.1f Kp=%.1f Kvb=%.1f Kg2=%.1f",
            ref_koren.mu, ref_koren.ex, ref_koren.kg1,
            ref_koren.kp, ref_koren.kvb, kg2,
        )
        return [ref_koren.mu, ref_koren.ex, ref_koren.kg1,
                ref_koren.kp, ref_koren.kvb, kg2]
    ug1_range = float(np.max(ug1) - np.min(ug1))
    ug2_mean = float(np.mean(ug2))
    mu0 = max(3.0, min(50.0, abs(ug2_mean / max(abs(ug1_range), 1.0))))
    return [mu0, 1.35, 890.0, 60.0, 24.0, 4500.0]


def _pentode_residuals(ua: np.ndarray, ug1: np.ndarray, ug2: np.ndarray,
                       ia_meas: np.ndarray, ig2_meas: Optional[np.ndarray],
                       params: List[float], w_ig2: float = 0.3) -> float:
    """Compute pentode combined Ia + Ig2 sum of squared residuals."""
    mu, ex, kg1, kp, kvb, kg2 = params
    ia_pred = _koren_ia_pentode(ua, ug1, ug2, mu, ex, kg1, kp, kvb)
    cost = float(np.sum((ia_pred - ia_meas) ** 2))
    if ig2_meas is not None and kg2 > 0:
        ig2_pred = _koren_ig2_pentode(ug1, ug2, mu, ex, kg2)
        cost += w_ig2 * float(np.sum((ig2_pred - ig2_meas) ** 2))
    return cost


def _fit_pentode_scipy(ua: np.ndarray, ug1: np.ndarray, ug2: np.ndarray,
                       ia_meas: np.ndarray,
                       ig2_meas: Optional[np.ndarray] = None,
                       ref_koren: Optional["KorenParams"] = None,
                       ) -> Tuple[np.ndarray, float, bool]:
    """Fit Koren pentode model using scipy least_squares (TRF).

    Fits 6 parameters: mu, ex, kg1, kp, kvb, kg2.
    If ig2_meas is provided, combines Ia and Ig2 residuals.
    """
    x0 = _make_pentode_initial_guess(ua, ug1, ug2, ref_koren)
    bounds_lo = [2.0, 0.8, 10.0, 10.0, 1.0, 500.0]
    bounds_hi = [500.0, 3.5, 50000.0, 3000.0, 500.0, 20000.0]
    # ML-066: same clip as the triode fitter — reference-seeded x0 must
    # start inside the bounds.
    x0 = list(np.clip(x0, bounds_lo, bounds_hi))
    w_ig2 = 0.3

    def residual_vec(p):
        mu, ex, kg1, kp, kvb, kg2 = p
        ia_pred = _koren_ia_pentode(ua, ug1, ug2, mu, ex, kg1, kp, kvb)
        r_ia = ia_pred - ia_meas
        if ig2_meas is not None:
            ig2_pred = _koren_ig2_pentode(ug1, ug2, mu, ex, kg2)
            r_ig2 = (ig2_pred - ig2_meas) * np.sqrt(w_ig2)
            return np.concatenate([r_ia, r_ig2])
        return r_ia

    result = _least_squares(
        residual_vec, x0,
        bounds=(bounds_lo, bounds_hi),
        method="trf",
        max_nfev=8000,
    )
    check_fit_convergence(result, "koren scipy pentode", log,
                          n_points=len(ua), has_ig2=ig2_meas is not None)
    params = np.array(result.x)
    cost = float(result.cost)
    return params, cost * 2.0, bool(result.success)


def _fit_pentode_numpy(ua: np.ndarray, ug1: np.ndarray, ug2: np.ndarray,
                       ia_meas: np.ndarray,
                       ig2_meas: Optional[np.ndarray] = None,
                       ref_koren: Optional["KorenParams"] = None,
                       ) -> Tuple[List[float], float, bool]:
    """Fit Koren pentode model using coordinate descent (numpy only)."""
    x0 = _make_pentode_initial_guess(ua, ug1, ug2, ref_koren)
    params = np.array(x0)
    bounds_lo = np.array([2.0, 0.8, 10.0, 10.0, 1.0, 500.0])
    bounds_hi = np.array([500.0, 3.5, 50000.0, 3000.0, 500.0, 20000.0])
    best_cost = _pentode_residuals(ua, ug1, ug2, ia_meas, ig2_meas, params)
    best_params = params.copy()
    gr = (np.sqrt(5) + 1) / 2

    for _ in range(30):
        improved = False
        for dim in range(6):
            lo, hi = bounds_lo[dim], bounds_hi[dim]
            a, b = lo, hi
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            for _ in range(40):
                trial_c = params.copy(); trial_c[dim] = c
                trial_d = params.copy(); trial_d[dim] = d
                fc = _pentode_residuals(ua, ug1, ug2, ia_meas, ig2_meas, trial_c)
                fd = _pentode_residuals(ua, ug1, ug2, ia_meas, ig2_meas, trial_d)
                if fc < fd:
                    b = d
                else:
                    a = c
                c = b - (b - a) / gr
                d = a + (b - a) / gr
                if abs(b - a) < 1e-6 * (bounds_hi[dim] - bounds_lo[dim]):
                    break
            opt_val = (a + b) / 2.0
            trial = params.copy(); trial[dim] = opt_val
            cost = _pentode_residuals(ua, ug1, ug2, ia_meas, ig2_meas, trial)
            if cost < best_cost:
                best_cost = cost
                params[dim] = opt_val
                best_params = params.copy()
                improved = True
        if not improved:
            break
    return best_params, best_cost, True


# ─── SUBCIRCUIT GENERATORS ────────────────────────────────────────────

def _generate_triode_subcircuit(
    safe_name, tube_type,
    mu, ex, kg1, kp, kvb,
    rms_error, max_error, n_points, backend,
    ref=None,
    mfg_date: str = "",
):
    """Generate triode SPICE subcircuit text (3-pin: A, G, K)."""
    lines = []
    lines.append(f"; SPICE model for {tube_type}")
    if mfg_date:
        lines.append(f"; Manufactured: {mfg_date}")
    lines.append("; Generated by LM19 Tube Tester")
    lines.append("; Koren triode model (Norman Koren, 1996)")
    if ref and ref.source:
        lines.append(f"; Reference data source: {ref.source}")
    lines.append(f"; Fit backend: {backend}")
    if rms_error is not None:  # absent for exported-from-model .subs
        lines.append(f"; Fit quality: RMS error = {rms_error:.2f} mA, "
                     f"Max error = {max_error:.2f} mA")
        lines.append(f"; Data points used: {n_points}")
    lines.append(";")
    lines.append(f";   mu    = {mu:.4f}    ; amplification factor")
    lines.append(f";   Ex    = {ex:.4f}    ; Ia exponent")
    lines.append(f";   Kg1   = {kg1:.4f}   ; cathode current scale")
    lines.append(f";   Kp    = {kp:.4f}   ; pinch-off parameter")
    lines.append(f";   Kvb   = {kvb:.4f}   ; knee voltage factor (V^2)")

    if ref and ref.caps:
        lines.append(";")
        lines.append(f";   Ccg   = {ref.caps.ccg:.1f} pF  ; cathode-grid")
        lines.append(f";   Cgp   = {ref.caps.cgp:.2f} pF  ; grid-plate (Miller)")
        lines.append(f";   Cpk   = {ref.caps.ccp:.1f} pF  ; cathode-plate")
        lines.append(f";   Rgi   = {ref.rgi}      ; grid current resistor")

    lines.append(";")
    lines.append("; E1 = (Ua/Kp) * ln(1 + exp(Kp * (1/mu + Ug1/sqrt(Kvb + Ua^2))))")
    lines.append("; Ia = 2 * E1^Ex / Kg1  [A]")
    lines.append("; Pin order: Anode, Grid, Cathode")
    lines.append(";")

    vct = ref.vct if ref else 0.0
    lines.append(f".SUBCKT {safe_name} A G K")
    params_line = f"+ PARAMS: MU={mu:.4f} EX={ex:.4f} KG1={kg1:.4f} KP={kp:.4f} KVB={kvb:.4f}"
    if vct != 0.0:
        params_line += f" VCT={vct:.2f}"
    lines.append(params_line)

    # RGI must be declared whenever the {RGI} grid-stopper below is emitted —
    # otherwise a ref with rgi but no caps yields an undefined-parameter netlist.
    if ref and ref.caps:
        lines.append(f"+ CCG={ref.caps.ccg}P CGP={ref.caps.cgp}P "
                      f"CCP={ref.caps.ccp}P RGI={ref.rgi}")
    elif ref and ref.rgi:
        lines.append(f"+ CCG=0.0P CGP=0.0P CCP=0.0P RGI={ref.rgi}")

    lines.append("*")
    # MIN(...,700) mirrors Python _EXP_CLIP (exp overflow guard only).
    lines.append("E1 7 0 VALUE={V(A,K)/KP*LOG(1+EXP(MIN(KP*(1/MU+"
                 "V(G,K)/SQRT(KVB+V(A,K)*V(A,K))),700)))}")
    lines.append("RE1 7 0 1G")
    lines.append("G1 A K VALUE={(PWR(V(7),EX)+PWRS(V(7),EX))/KG1}")
    lines.append("RCP A K 1G")

    if ref and ref.caps:
        lines.append("*")
        lines.append(f"C1 G K {{CCG}}    ; cathode-grid  ({ref.caps.ccg} pF)")
        lines.append(f"C2 G A {{CGP}}    ; grid-plate    ({ref.caps.cgp} pF)")
        lines.append(f"C3 A K {{CCP}}    ; cathode-plate ({ref.caps.ccp} pF)")

    if ref and ref.rgi:
        lines.append("*")
        lines.append(f"R1 G 5 {{RGI}}    ; grid stopper ({ref.rgi} ohm)")
        lines.append("D3 5 K DX         ; grid-cathode diode")
        lines.append(".MODEL DX D(IS=1N RS=1 CJO=10PF TT=1N)")

    lines.append("*")
    lines.append(f".ENDS {safe_name}")
    lines.append("")
    return "\n".join(lines)


def _generate_pentode_subcircuit(
    safe_name, tube_type,
    mu, ex, kg1, kp, kvb, kg2,
    rms_error, max_error, n_points, backend,
    ig2_rms=None,
    ref=None,
    mfg_date: str = "",
):
    """Generate pentode SPICE subcircuit text (4-pin: A, G, K, G2)."""
    lines = []
    lines.append(f"; SPICE model for {tube_type}")
    if mfg_date:
        lines.append(f"; Manufactured: {mfg_date}")
    lines.append("; Generated by LM19 Tube Tester")
    lines.append("; Koren pentode model (Norman Koren, 1996)")
    if ref and ref.source:
        lines.append(f"; Reference data source: {ref.source}")
    lines.append(f"; Fit backend: {backend}")
    if rms_error is not None:  # absent for exported-from-model .subs
        lines.append(f"; Fit quality (Ia): RMS = {rms_error:.2f} mA, "
                     f"Max = {max_error:.2f} mA")
        if ig2_rms is not None:
            lines.append(f"; Fit quality (Ig2): RMS = {ig2_rms:.2f} mA")
        lines.append(f"; Data points used: {n_points}")
    lines.append(";")
    lines.append(f";   mu    = {mu:.4f}    ; amplification factor")
    lines.append(f";   Ex    = {ex:.4f}    ; Ia exponent")
    lines.append(f";   Kg1   = {kg1:.4f}   ; plate current scale")
    lines.append(f";   Kg2   = {kg2:.4f}   ; screen current scale")
    lines.append(f";   Kp    = {kp:.4f}   ; pinch-off parameter")
    lines.append(f";   Kvb   = {kvb:.4f}   ; knee voltage (V)")

    if ref and ref.caps:
        c = ref.caps
        lines.append(";")
        lines.append(f";   Ccg1  = {c.ccg1:.1f} pF  ; cathode-grid1")
        lines.append(f";   Ccg2  = {c.ccg2:.1f} pF  ; cathode-grid2")
        lines.append(f";   Cpg1  = {c.cpg1:.2f} pF  ; grid1-plate")
        lines.append(f";   Cg1g2 = {c.cg1g2:.2f} pF  ; grid1-grid2")
        lines.append(f";   Cpk   = {c.ccp:.1f} pF  ; cathode-plate")

    lines.append(";")
    lines.append("; E1 = (Ug2/Kp) * ln(1 + exp(Kp * (1/mu + Ug1/Ug2)))")
    lines.append("; Ia = 2 * E1^Ex / Kg1 * arctan(Ua/Kvb)  [A]")
    lines.append("; Ig2 = (Ug1 + Ug2/mu)^Ex / Kg2  [A]")
    lines.append("; Pin order: Anode, Grid1, Cathode, Grid2")
    lines.append(";")

    lines.append(f".SUBCKT {safe_name} A G K G2")
    lines.append(f"+ PARAMS: MU={mu:.4f} EX={ex:.4f} KG1={kg1:.4f} "
                 f"KG2={kg2:.4f} KP={kp:.4f} KVB={kvb:.4f}")

    # Pentode caps live in the pentode fields (ccg1/ccg2/cpg1/cg1g2/ccp), not
    # the triode fields. RGI declared in both branches so the {RGI} grid-stopper
    # below always has its parameter (undefined-parameter netlist otherwise).
    caps = ref.caps if ref and ref.caps else None
    rgi = ref.rgi if ref and ref.rgi else _DEFAULT_PENTODE_RGI_OHM
    if caps:
        lines.append(f"+ CCG1={caps.ccg1}P CCG2={caps.ccg2}P "
                      f"CPG1={caps.cpg1}P CG1G2={caps.cg1g2}P "
                      f"CCP={caps.ccp}P RGI={rgi}")
    elif ref and ref.rgi:
        lines.append(f"+ CCG1=0.0P CCG2=0.0P CPG1=0.0P CG1G2=0.0P "
                      f"CCP=0.0P RGI={rgi}")

    lines.append("*")
    lines.append("* E1: screen-grid controlled intermediate voltage")
    # MIN(...,700) mirrors Python _EXP_CLIP (exp overflow guard only).
    lines.append("E1 7 0 VALUE={V(G2,K)/KP*LOG(1+EXP(MIN("
                 "(1/MU+V(G,K)/V(G2,K))*KP,700)))}")
    lines.append("RE1 7 0 1G")
    lines.append("*")
    lines.append("* G1: anode current with arctan knee")
    lines.append("G1 A K VALUE={(PWR(V(7),EX)+PWRS(V(7),EX))/KG1*ATAN(V(A,K)/KVB)}")
    lines.append("RCP A K 1G")
    lines.append("*")
    lines.append("* G2: screen current source")
    lines.append("G2 G2 K VALUE={(EXP(EX*(LOG((V(G2,K)/MU)+V(G,K)))))/KG2}")
    lines.append("R2 G2 K 1G")

    if caps:
        lines.append("*")
        lines.append(f"C1 G K {{CCG1}}     ; cathode-grid1  ({caps.ccg1} pF)")
        lines.append(f"C4 G2 K {{CCG2}}    ; cathode-grid2  ({caps.ccg2} pF)")
        lines.append(f"C5 G2 G {{CG1G2}}   ; grid1-grid2    ({caps.cg1g2} pF)")
        lines.append(f"C2 A G {{CPG1}}     ; grid1-plate    ({caps.cpg1} pF)")
        lines.append(f"C3 A K {{CCP}}      ; cathode-plate  ({caps.ccp} pF)")

    if ref and ref.rgi:
        lines.append("*")
        lines.append(f"R1 G 5 {{RGI}}     ; grid stopper ({ref.rgi} ohm)")
        lines.append("D3 5 K DX          ; grid-cathode diode")
        lines.append(".MODEL DX D(IS=1N RS=1 CJO=10PF TT=1N)")

    lines.append("*")
    lines.append(f".ENDS {safe_name}")
    lines.append("")
    return "\n".join(lines)


# ─── FIT-AND-EXPORT PIPELINE ──────────────────────────────────────────

def fit_and_export_triode(
    path: str,
    tube_type: str,
    safe_name: str,
    points: List[Dict],
    ref,
    ref_koren,
    backend: str,
    mfg_date: str = "",
) -> SpiceFitResult:
    """Fit Koren triode model and write .sub file."""
    data = extract_arrays(points, topology=TOPOLOGY_TRIODE)

    if _HAS_SCIPY:
        log.info("Fitting Koren triode with scipy")
        params, _, _converged = _fit_koren_scipy(
            data.ua, data.ug1, data.ia, ref_koren)
    else:
        log.info("Fitting Koren triode with numpy")
        params, _, _converged = _fit_koren_numpy(
            data.ua, data.ug1, data.ia, ref_koren)

    mu, ex, kg1, kp, kvb = params

    ia_pred = _koren_ia(data.ua, data.ug1, mu, ex, kg1, kp, kvb)
    rms_error, max_error = compute_fit_errors(ia_pred, data.ia)

    log.info("Triode fit (%s): mu=%.2f Ex=%.3f Kg1=%.1f Kp=%.1f Kvb=%.1f  "
             "RMS=%.2f mA  Max=%.2f mA",
             backend, mu, ex, kg1, kp, kvb, rms_error, max_error)

    content = _generate_triode_subcircuit(
        safe_name, tube_type, mu, ex, kg1, kp, kvb,
        rms_error, max_error, data.n_points, backend, ref,
        mfg_date=mfg_date)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return SpiceFitResult(
        model_type=TOPOLOGY_TRIODE,
        algorithm="koren",
        params={"mu": mu, "ex": ex, "kg1": kg1, "kp": kp, "kvb": kvb},
        rms_error=rms_error,
        max_error=max_error,
        n_points=data.n_points,
        path=path,
    )


def fit_and_export_pentode(
    path: str,
    tube_type: str,
    safe_name: str,
    points: List[Dict],
    ref,
    ref_koren,
    backend: str,
    mfg_date: str = "",
) -> SpiceFitResult:
    """Fit Koren pentode model and write .sub file."""
    data = extract_arrays(points, topology=TOPOLOGY_PENTODE)

    if _HAS_SCIPY:
        log.info("Fitting Koren pentode with scipy (Ig2 data: %s)", data.has_ig2)
        params, _, _converged = _fit_pentode_scipy(
            data.ua, data.ug1, data.ug2, data.ia, data.ig2, ref_koren)
    else:
        log.info("Fitting Koren pentode with numpy (Ig2 data: %s)", data.has_ig2)
        params, _, _converged = _fit_pentode_numpy(
            data.ua, data.ug1, data.ug2, data.ia, data.ig2, ref_koren)

    mu, ex, kg1, kp, kvb, kg2 = params

    ia_pred = _koren_ia_pentode(
        data.ua, data.ug1, data.ug2, mu, ex, kg1, kp, kvb)
    rms_error, max_error = compute_fit_errors(ia_pred, data.ia)

    ig2_rms = None
    export_warnings: list = []
    if data.has_ig2:
        ig2_pred = _koren_ig2_pentode(data.ug1, data.ug2, mu, ex, kg2)
        ig2_rms, _ = compute_fit_errors(ig2_pred, data.ig2)
    else:
        # ML-111: without Ig2 data kg2 never entered the residual — the
        # .sub carries an initial-guess screen current that LTspice will
        # present as a fitted one.
        log.warning("Koren pentode SPICE export without Ig2 data — Kg2 is "
                    "NOT fitted (initial guess exported)")
        export_warnings.append("kg2_unfitted")

    log.info("Pentode fit (%s): mu=%.2f Ex=%.3f Kg1=%.1f Kp=%.1f Kvb=%.1f "
             "Kg2=%.1f  RMS=%.2f mA  Max=%.2f mA  Ig2_RMS=%s",
             backend, mu, ex, kg1, kp, kvb, kg2, rms_error, max_error,
             f"{ig2_rms:.2f} mA" if ig2_rms else "N/A")

    content = _generate_pentode_subcircuit(
        safe_name, tube_type, mu, ex, kg1, kp, kvb, kg2,
        rms_error, max_error, data.n_points, backend, ig2_rms, ref,
        mfg_date=mfg_date)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return SpiceFitResult(
        model_type=TOPOLOGY_PENTODE,
        algorithm="koren",
        params={"mu": mu, "ex": ex, "kg1": kg1, "kp": kp,
                "kvb": kvb, "kg2": kg2},
        rms_error=rms_error,
        max_error=max_error,
        n_points=data.n_points,
        path=path,
        warnings=export_warnings,
    )
