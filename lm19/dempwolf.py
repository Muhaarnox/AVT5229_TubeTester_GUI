"""Dempwolf Extended v2 vacuum tube model.

Unified model covering triodes, pentodes, beam tetrodes, and variable-mu
pentodes.  Based on Dempwolf & Zoelzer (DAFx-11, 2011) with pentode/beam
tetrode extensions and v2 improvements (fg2, Durchgriff, Vco).

See docs/DEMPWOLF_EXTENDED_MODEL.md for full derivation and equations.

No Qt dependencies.  Used by tests, amplifier analysis, and (optionally)
SPICE export.
"""


from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from lm19.tube_params import DempwolfParams, lookup_tube, list_tubes
from lm19.tube_model_base import (
    TubeModelProtocol, ModelFitResult, register_model,
    extract_arrays, build_fit_result, check_fit_convergence,
    ConvergenceTracker,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)
from lm19.tube_model_base import (
    MODEL_TYPE_DEMPWOLF,
)
from lm19.tube_model_base import (
    MODEL_WARN_DEMPWOLF_GRID_DEFAULTS,
    MODEL_WARN_DEMPWOLF_NO_IG2_FULL,
    MODEL_WARN_DEMPWOLF_NO_IG2_MASK,
    MODEL_WARN_DEMPWOLF_PHASE2_TRIODE_FORM,
)


def _check(tracker, result, phase_name, **ctx):
    """Dispatch to tracker.check() or check_fit_convergence() if no tracker."""
    if tracker is not None:
        tracker.check(result, phase_name, log, **ctx)
    else:
        check_fit_convergence(result, phase_name, log, **ctx)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------
# Clip argument to exp() to prevent float64 overflow (exp overflows at ~709;
# 700 keeps headroom). A tight clip (e.g. 50) SATURATES the
# Kp-normalized softplus in-range for hand-curated high-C parameter sets
# (C=750, mu=11, Vg2=250: arg = C*(1/mu + Vg/Vg2) > 50 for Vg > -6 V, so
# gm collapsed to ~0 near Vg=0 and the SPICE export — which never clipped —
# diverged from the Python model there). Real fits give C <= ~120 and
# arg <= ~5, so 700 is bit-identical for every fitted model; pinned in
# tests/test_dempwolf_paper_pins.py::TestNoInRangeSaturation.
_EXP_CLIP = 700.0
_V_MIN = 0.01          # minimum voltage to prevent division by zero
_KVB_MIN = 0.1         # minimum Kvb_eff to prevent arctan(inf)
_TWO_OVER_PI = 2.0 / np.pi

# Default Kvb_t (variable-mu knee softness, V²) used when grid-current
# fitting does NOT include this parameter (``fit_kvb_t=False``) and as
# initial guess when it does. Phase 2 bounds are 10–5000 V²; 300 V²
# (~17 V knee) is the typical mid-range value across small-signal and
# power tubes. Used in Phase 1 triode residual (where Kvb_t is held
# constant — Phase 1 only seeds mu/G/gamma/C) and as the placeholder
# returned when no grid-current data is available.
_KVB_T_DEFAULT_V2 = 300.0

# ── kink (negative-resistance) detection ──
# Points are bucketed per curve by BOTH Ug1 (0.1 V) and Ug2 (1 V): the old
# Ug1-only grouping interleaved the Ug2 levels of a multi-Ug2 scan and read
# the level steps as dips — every real LM19 pentode scan was classified as a
# beam tetrode and phase 5 fitted fictional secondary emission (true-pentode
# EL84 scans got σ ≈ 0.13–0.23). (Trade-off: a curve whose setpoint sits on a
# bucket edge can split — fine for clean LM19 setpoints.) The kink metric is
# the cumulative DRAWDOWN (running peak minus current Ia along ascending Ua),
# not the per-step diff: a dynatron valley descends monotonically across many
# Ua steps (its depth is split between steps at fine scan resolution), while
# noise jitters point-to-point and recovers immediately. Measured populations
# across the benchmark datasets: true-pentode scan/trace artifacts reach 2.6%
# of curve max (EL84 ER_L2), real dynatron valleys are ≥16% (6P1P real scan,
# 6L6-class synthetics ~24%); 5% separates them with wide margin on both
# sides. The absolute floor keeps tiny-current curves (fractions of a mA at
# deep bias) from tripping on µA-level noise.
_KINK_UG1_GROUP_ROUND = 1        # decimals for grouping points by Ug1 level
_KINK_UG2_GROUP_ROUND = 0        # decimals for grouping points by Ug2 level
_KINK_DRAWDOWN_REL = 0.05        # drawdown as a fraction of the curve max Ia
_KINK_DRAWDOWN_MIN_A = 0.1e-3    # absolute drawdown floor (A)

# Phases 1-2 on Ia-only data keep points above ~this fraction of the median
# screen voltage: below the knee Ia = Ik·α with α < 1, which poisons a
# cathode fit whose target is Ia alone (with Ig2 data the target Ik = Ia+Ig2
# is knee-independent and no masking is needed).
_KNEE_SAFE_FACTOR = 0.9

# Decimals for counting distinct Ug1 levels in the phase-2 grid region — kept
# separate from the kink grouping so retuning one policy never silently shifts
# the other. NOTE: np.round buckets to 0.1 V, so two genuine levels <0.05 V
# apart can collapse to one (→ triode-form fallback); fine for real setpoints.
_PHASE2_UG1_GROUP_ROUND = 1

# ── triode joint-refine bounds ──
# Union of the phase-1 adaptive bounds (all mu branches) and the phase-2
# grid/Kvb_t bounds: the joint pass must be able to hold any seed the earlier
# phases can produce. Order: [mu, G, gamma, C, Gg, xi, Cg, Kvb_t].
_TRIODE_REFINE_LO = [1.5, 1e-5, 0.8, 0.05, 1e-6, 0.8, 1.0, 10.0]
_TRIODE_REFINE_HI = [500.0, 0.5, 2.5, 50.0, 1e-2, 2.5, 50.0, 5000.0]
# Log-spread Kvb_t multi-starts (V²) spanning the Region-A bound range —
# C and Kvb_t are correlated in the low-current region, and a single start
# can settle in a local minimum (observed: true 1200 → fitted 16).
_TRIODE_REFINE_KVBT_STARTS = (30.0, 300.0, 3000.0)

# ── pentode / beam-tetrode fit bounds (docs §10.6 / §14.7) ──
# Single source for phase 4 AND the beam joint refine: the two previously
# carried divergent literals (joint refine allowed fg2 up to 0.5 and A up to
# 0.01 — A·Va at 500 V would scale emission 6×, far past the perturbative
# Durchgriff regime documented in §14.2).
# Order: [mu, G, gamma, C, Gg, xi, Cg, Kvb, Kvb1, fg2, A, Kn]
_PENTODE_BOUNDS_LO = [2.0, 1e-5, 0.8, 1.0, 1e-6, 0.8, 1.0,
                      0.5, 0.0, 0.0, 0.0, 0.7]
_PENTODE_BOUNDS_HI = [500.0, 1e-1, 2.5, 2000.0, 1e-1, 2.5, 50.0,
                      200.0, 5.0, 0.30, 0.001, 2.0]
# Secondary emission bounds (docs §10.6 σ/Ks + §14.7 λ/ν/w), shared by
# phase 5 and the beam joint refine. Order: [sigma, Ks, lam, nu, w]
# ν upper bound is 12, not the §14.7 "typical 1–4" ceiling of 6: the real
# 6P1P scan fits ν = 7.8 (kink shift vs Ug1), and clamping ν collapses the
# whole secondary-emission fit to σ ≈ 0 (kink lost, rms 0.69 → 1.05 mA).
_SEC_EMISSION_BOUNDS_LO = [0.0, 0.1, 0.5, 0.5, -50.0]
_SEC_EMISSION_BOUNDS_HI = [10.0, 20.0, 25.0, 12.0, 50.0]
# σ floor for the beam joint refine — it only runs after phase 5 classified
# the tube as a beam tetrode (σ > 0.01), and must not drift back to σ ≈ 0.
_JOINT_SIGMA_MIN = 0.01
# Phase-5 multi-starts: λ per §14.6 (≈1 beam tetrode, ≈15 pentode-like
# screening), ν spread across the space-charge range (real 6P1P fits ν≈8).
_PHASE5_LAM_STARTS = (1.0, 15.0)
_PHASE5_NU_STARTS = (1.0, 4.0, 8.0)

# Grid-current defaults [Gg (A), ξ, Cg] — phase-2 x0 AND the fallback when
# the data has no grid region (docs §10.2); the triode joint refine freezes
# the grid params at exactly these values in that case.
_GRID_CURRENT_DEFAULTS = (6e-4, 1.3, 10.0)


# ---------------------------------------------------------------------------
# Core model function — §14.5
# ---------------------------------------------------------------------------

def dempwolf_v2(
    vpk: float,
    vgk: float,
    vg2k: Optional[float] = None,
    *,
    p: DempwolfParams,
) -> Tuple[float, Optional[float], float]:
    """Compute currents using Dempwolf Extended v2 model.

    Args:
        vpk:  plate-to-cathode voltage (V).
        vgk:  grid-to-cathode voltage (V, negative for normal operation).
        vg2k: screen-to-cathode voltage (V).  None → triode mode.
        p:    model parameters (DempwolfParams).

    Returns:
        (ipk, ig2k, igk) — currents in **amperes**.
        ig2k is None for triodes.
    """
    is_triode = vg2k is None

    # --- voltage safety clamps ---
    vpk_safe = max(vpk, _V_MIN)
    vg2k_safe = max(vg2k, _V_MIN) if not is_triode else 0.0

    # --- grid effective voltage (Region A for triodes) ---
    if is_triode:
        v_grid_eff = vgk * vpk_safe / np.sqrt(p.Kvb_t + vpk_safe * vpk_safe)
        v_accel = vpk_safe
    else:
        v_grid_eff = vgk
        v_accel = vg2k_safe

    # --- cathode emission ---
    use_normalized = not is_triode  # Koren-style Vg2-normalization for pentodes
    if p.mu_b is not None and p.gamma_b is not None and p.svar > 0:
        # variable-mu: two sections
        ik = _cathode_current_varmu(
            v_accel, v_grid_eff, p.mu, p.G, p.gamma, p.C,
            p.mu_b, p.gamma_b, p.svar,
            normalized=use_normalized,
        )
    else:
        ik = _cathode_current(
            v_accel, v_grid_eff, p.mu, p.G, p.gamma, p.C,
            normalized=use_normalized,
        )

    # --- Durchgriff (v2, pentodes only) ---
    if not is_triode and p.A > 0:
        ik *= (1.0 + p.A * vpk_safe)

    # --- grid current ---
    igk = _grid_current(vgk, p.Gg, p.xi, p.Cg)

    # --- triode: simple subtraction ---
    if is_triode:
        ipk = max(ik - igk, 0.0)
        return ipk, None, igk

    # --- pentode current splitting ---
    i_through = max(ik - igk, 0.0)

    v_eff = max(v_accel / p.mu + vgk, 0.0)
    kvb_eff = max(p.Kvb + p.Kvb1 * v_eff, _KVB_MIN)

    alpha = (1.0 - p.fg2) * _TWO_OVER_PI * np.arctan(
        (vpk_safe / kvb_eff) ** p.Kn
    )

    ipk_primary = i_through * alpha
    ig2k_base = i_through * (1.0 - alpha)

    # --- secondary emission (beam tetrodes) ---
    if p.sigma > 0 and vg2k_safe > _V_MIN:
        vco = vg2k_safe / p.lam - p.nu * vgk - p.w
        vco_safe = max(vco, _V_MIN)
        x = max(1.0 - vpk_safe / vco_safe, 0.0)
        i_sec = (
            p.sigma * i_through * (vpk_safe / vg2k_safe)
            * x * np.exp(-p.Ks * x)
        )
        ipk = ipk_primary - i_sec
        ig2k = ig2k_base + i_sec
    else:
        ipk = ipk_primary
        ig2k = ig2k_base

    return ipk, ig2k, igk


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _softplus(arg: float, scale: float) -> float:
    """Compute ln(1 + exp(arg)) / scale with overflow protection."""
    arg_c = np.clip(arg, -_EXP_CLIP, _EXP_CLIP)
    return float(np.log1p(np.exp(arg_c))) / scale


def _cathode_current(
    v_accel: float, v_grid_eff: float,
    mu: float, G: float, gamma: float, C: float,
    *, normalized: bool = False,
) -> float:
    """Single-section cathode emission current (A).

    If normalized=True (pentode mode), use Koren-style Vg2-normalized form:
      sp = (v_accel / C) · log(1 + exp(C · (1/µ + v_grid_eff / v_accel)))
    Otherwise (triode mode), use original Dempwolf form:
      sp = log(1 + exp(C · (v_accel/µ + v_grid_eff))) / C
    """
    if normalized and v_accel > _V_MIN:
        arg_k = C * (1.0 / mu + v_grid_eff / v_accel)
        arg_c = float(np.clip(arg_k, -_EXP_CLIP, _EXP_CLIP))
        sp = (v_accel / C) * float(np.log1p(np.exp(arg_c)))
    else:
        arg_k = C * (v_accel / mu + v_grid_eff)
        sp = _softplus(arg_k, C)
    return G * max(sp, 0.0) ** gamma


def _cathode_current_varmu(
    v_accel: float, v_grid_eff: float,
    mu_a: float, G: float, gamma_a: float, C: float,
    mu_b: float, gamma_b: float, svar: float,
    *, normalized: bool = False,
) -> float:
    """Variable-mu cathode emission: two sections blended."""
    if normalized and v_accel > _V_MIN:
        arg_a = float(np.clip(C * (1.0 / mu_a + v_grid_eff / v_accel),
                              -_EXP_CLIP, _EXP_CLIP))
        sp_a = (v_accel / C) * float(np.log1p(np.exp(arg_a)))
        arg_b = float(np.clip(C * (1.0 / mu_b + v_grid_eff / v_accel),
                              -_EXP_CLIP, _EXP_CLIP))
        sp_b = (v_accel / C) * float(np.log1p(np.exp(arg_b)))
    else:
        arg_a = C * (v_accel / mu_a + v_grid_eff)
        sp_a = _softplus(arg_a, C)
        arg_b = C * (v_accel / mu_b + v_grid_eff)
        sp_b = _softplus(arg_b, C)
    ik_a = G * max(sp_a, 0.0) ** gamma_a
    ik_b = G * max(sp_b, 0.0) ** gamma_b

    return (1.0 - svar) * ik_a + svar * ik_b


def _grid_current(
    vgk: float, Gg: float, xi: float, Cg: float,
) -> float:
    """Grid current (A) — smooth softplus model."""
    arg_g = Cg * vgk
    sp = _softplus(arg_g, Cg)
    return Gg * max(sp, 0.0) ** xi


def _softplus_vec(arg: np.ndarray, scale: float) -> np.ndarray:
    """Element-wise ln(1 + exp(arg)) / scale with overflow protection."""
    arg_c = np.clip(arg, -_EXP_CLIP, _EXP_CLIP)
    return np.log1p(np.exp(arg_c)) / scale


def dempwolf_v2_ia_vec(
    vpk: np.ndarray,
    vgk: np.ndarray,
    vg2k: Optional[np.ndarray],
    *,
    p: DempwolfParams,
) -> np.ndarray:
    """Vectorized Ia (A) — element-wise mirror of :func:`dempwolf_v2`.

    Every branch of the scalar path (normalized vs plain cathode form,
    variable-mu blend, Durchgriff, secondary emission) is reproduced with
    the same formulas via element-wise masks, so results match the scalar
    call per element. Used by the optimizer hot path (``ia_array``); the
    fitting-only kernels ``_eval_*_vec`` intentionally differ (no grid
    current in triode phase 1, no varmu) and must not be used for it.
    """
    is_triode = vg2k is None
    vpk_safe = np.maximum(vpk, _V_MIN)

    if is_triode:
        v_grid_eff = vgk * vpk_safe / np.sqrt(p.Kvb_t + vpk_safe * vpk_safe)
        v_accel = vpk_safe
        norm_mask = None                       # triode: plain form always
    else:
        vg2k_safe = np.maximum(vg2k, _V_MIN)
        v_grid_eff = vgk
        v_accel = vg2k_safe
        # scalar: `if normalized and v_accel > _V_MIN` — element-wise here
        norm_mask = v_accel > _V_MIN

    def _cathode_vec(mu: float, gamma: float) -> np.ndarray:
        sp_plain = _softplus_vec(p.C * (v_accel / mu + v_grid_eff), p.C)
        if norm_mask is None:
            sp = sp_plain
        else:
            arg_n = np.clip(p.C * (1.0 / mu + v_grid_eff / v_accel),
                            -_EXP_CLIP, _EXP_CLIP)
            sp_norm = (v_accel / p.C) * np.log1p(np.exp(arg_n))
            sp = np.where(norm_mask, sp_norm, sp_plain)
        return p.G * np.maximum(sp, 0.0) ** gamma

    if p.mu_b is not None and p.gamma_b is not None and p.svar > 0:
        ik = ((1.0 - p.svar) * _cathode_vec(p.mu, p.gamma)
              + p.svar * _cathode_vec(p.mu_b, p.gamma_b))
    else:
        ik = _cathode_vec(p.mu, p.gamma)

    if not is_triode and p.A > 0:
        ik = ik * (1.0 + p.A * vpk_safe)

    igk = p.Gg * np.maximum(_softplus_vec(p.Cg * vgk, p.Cg), 0.0) ** p.xi

    if is_triode:
        return np.maximum(ik - igk, 0.0)

    i_through = np.maximum(ik - igk, 0.0)
    v_eff = np.maximum(v_accel / p.mu + vgk, 0.0)
    kvb_eff = np.maximum(p.Kvb + p.Kvb1 * v_eff, _KVB_MIN)
    alpha = (1.0 - p.fg2) * _TWO_OVER_PI * np.arctan(
        (vpk_safe / kvb_eff) ** p.Kn
    )
    ipk = i_through * alpha

    if p.sigma > 0:
        sec_mask = vg2k_safe > _V_MIN
        vco = vg2k_safe / p.lam - p.nu * vgk - p.w
        vco_safe = np.maximum(vco, _V_MIN)
        x = np.maximum(1.0 - vpk_safe / vco_safe, 0.0)
        i_sec = (p.sigma * i_through * (vpk_safe / vg2k_safe)
                 * x * np.exp(-p.Ks * x))
        ipk = ipk - np.where(sec_mask, i_sec, 0.0)

    return ipk


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def dempwolf_triode(
    vpk: float, vgk: float, *, p: DempwolfParams,
) -> Tuple[float, float]:
    """Triode mode.  Returns (ipk, igk) in amperes."""
    ipk, _, igk = dempwolf_v2(vpk, vgk, vg2k=None, p=p)
    return ipk, igk


def dempwolf_pentode(
    vpk: float, vgk: float, vg2k: float, *, p: DempwolfParams,
) -> Tuple[float, float, float]:
    """Pentode mode (σ=0).  Returns (ipk, ig2k, igk) in amperes."""
    ipk, ig2k, igk = dempwolf_v2(vpk, vgk, vg2k, p=p)
    return ipk, ig2k or 0.0, igk


def dempwolf_beam_tetrode(
    vpk: float, vgk: float, vg2k: float, *, p: DempwolfParams,
) -> Tuple[float, float, float]:
    """Beam tetrode mode (σ>0).  Returns (ipk, ig2k, igk) in amperes."""
    ipk, ig2k, igk = dempwolf_v2(vpk, vgk, vg2k, p=p)
    return ipk, ig2k or 0.0, igk


# ---------------------------------------------------------------------------
# DempwolfModel — satisfies TubeModelProtocol
# ---------------------------------------------------------------------------

# Import ScanGrid here to avoid circular dependency at module level.
# tube_sim.py imports from us, we import ScanGrid from tube_sim.
# Solved by lazy import inside generate_scan().

@dataclass
class DempwolfModel:
    """Simulated tube with Dempwolf Extended v2 parameters.

    Satisfies TubeModelProtocol.  Drop-in replacement for TubeModel (Koren).
    """
    name: str
    topology: str
    dempwolf: DempwolfParams
    uh: float = 6.3
    ih: float = 0.3
    pa_max: float = 12.5
    model_type: str = MODEL_TYPE_DEMPWOLF

    def ia(self, ua: float, ug1: float, ug2: float = 0.0) -> float:
        """Compute Ia (mA) at a single operating point."""
        vg2k = None if self.topology == TOPOLOGY_TRIODE else ug2
        ipk, _, _ = dempwolf_v2(ua, ug1, vg2k, p=self.dempwolf)
        return ipk * 1000.0

    def ia_array(self, ua, ug1, ug2=0.0) -> np.ndarray:
        """Vectorized Ia (mA) over broadcastable arrays — element-wise
        mirror of scalar ``ia()`` (optimizer hot path)."""
        ua_a, ug1_a, ug2_a = np.broadcast_arrays(
            np.asarray(ua, dtype=float),
            np.asarray(ug1, dtype=float),
            np.asarray(ug2, dtype=float),
        )
        vg2k = None if self.topology == TOPOLOGY_TRIODE else ug2_a
        return dempwolf_v2_ia_vec(ua_a, ug1_a, vg2k, p=self.dempwolf) * 1000.0

    def ig2(self, ua: float, ug1: float, ug2: float) -> float:
        """Compute Ig2 (mA).  Returns 0 for triodes."""
        if self.topology == TOPOLOGY_TRIODE:
            return 0.0
        _, ig2k, _ = dempwolf_v2(ua, ug1, ug2, p=self.dempwolf)
        return (ig2k or 0.0) * 1000.0

    def params_dict(self) -> Dict:
        """Return Dempwolf parameters as a plain dict."""
        d: Dict = {
            "mu": self.dempwolf.mu,
            "G": self.dempwolf.G,
            "gamma": self.dempwolf.gamma,
            "C": self.dempwolf.C,
            "Gg": self.dempwolf.Gg,
            "xi": self.dempwolf.xi,
            "Cg": self.dempwolf.Cg,
        }
        if self.topology == TOPOLOGY_TRIODE:
            d["Kvb_t"] = self.dempwolf.Kvb_t
        else:
            d["Kvb"] = self.dempwolf.Kvb
            d["Kvb1"] = self.dempwolf.Kvb1
            d["Kn"] = self.dempwolf.Kn
            d["fg2"] = self.dempwolf.fg2
            d["A"] = self.dempwolf.A
            if self.dempwolf.sigma > 0:
                d["sigma"] = self.dempwolf.sigma
                d["Ks"] = self.dempwolf.Ks
                d["lam"] = self.dempwolf.lam
                d["nu"] = self.dempwolf.nu
                d["w"] = self.dempwolf.w
        if self.dempwolf.mu_b is not None:
            d["mu_b"] = self.dempwolf.mu_b
            d["gamma_b"] = self.dempwolf.gamma_b
            d["svar"] = self.dempwolf.svar
        return d

    def generate_scan(self, grid) -> List[Dict]:
        """Generate measurement points matching LM19 scan format."""
        points: List[Dict] = []
        ua_values = np.arange(
            grid.ua[0], grid.ua[1] + grid.ua[2] * 0.5, grid.ua[2],
        )
        ug1_values = np.arange(
            grid.ug1[0], grid.ug1[1] + grid.ug1[2] * 0.5, grid.ug1[2],
        )

        if self.topology == TOPOLOGY_TRIODE or grid.ug2_track_ua:
            ug2_values = [0.0]
        elif grid.ug2 is not None:
            ug2_values = np.arange(
                grid.ug2[0], grid.ug2[1] + grid.ug2[2] * 0.5, grid.ug2[2],
            )
        else:
            ug2_values = [250.0]

        for ug2_nom in ug2_values:
            for ug1 in ug1_values:
                for ua in ua_values:
                    if self.topology == TOPOLOGY_TRIODE:
                        ug2_actual = 0.0
                    elif grid.ug2_track_ua:
                        ug2_actual = float(ua) + grid.ug2_offset
                    else:
                        ug2_actual = float(ug2_nom)

                    ia_val = self.ia(float(ua), float(ug1), ug2_actual)
                    ig2_val = self.ig2(float(ua), float(ug1), ug2_actual)

                    points.append({
                        "ua": float(ua),
                        "ug1": float(ug1),
                        "ug2": ug2_actual,
                        "ia": ia_val,
                        "ig2": ig2_val,
                        "uh": grid.uh,
                        "ih": grid.ih,
                    })
        return points


# ---------------------------------------------------------------------------
# Loader and registry
# ---------------------------------------------------------------------------

def load_dempwolf_model(tube_name: str) -> Optional[DempwolfModel]:
    """Load a Dempwolf model from tube_params.json by name or alias.

    Returns DempwolfModel or None if tube has no Dempwolf parameters.
    """
    ref = lookup_tube(tube_name)
    if ref is None or ref.dempwolf is None:
        return None
    return DempwolfModel(
        name=ref.name,
        topology=ref.topology,
        dempwolf=ref.dempwolf,
    )


def _list_dempwolf_tubes() -> List[str]:
    """List tubes that have Dempwolf parameters."""
    all_tubes = list_tubes()
    result = []
    for name in all_tubes:
        ref = lookup_tube(name)
        if ref and ref.dempwolf is not None:
            result.append(name)
    return result


def fit_dempwolf(points: List[Dict], topology: str) -> ModelFitResult:
    """Fit Dempwolf Extended v2 model to measured data.

    Uses phased fitting strategy (§10 + §14.6):
      Phase 1: cathode emission (µ, G, γ, C)
      Phase 2: grid current (Gg, ξ, Cg)  [if data available]
      Phase 3: pentode knee (Kvb, Kvb1, fg2, A)  [pentode only]
      Phase 4: full refinement (all params simultaneously)
      Phase 5: secondary emission (σ, Ks, λ, ν, w)  [beam tetrodes]

    Args:
        points: measurement dicts with ua, ug1, ug2, ia, ig2 keys.
                Ia/Ig2 in mA.
        topology: "triode", "pentode", or "triode_connected".

    Returns:
        ModelFitResult with fitted DempwolfModel.
    """
    from scipy.optimize import least_squares

    if topology == TOPOLOGY_TRIODE:
        return _fit_triode(points, least_squares)
    else:
        # pentode AND triode_connected use pentode path
        # (Kp-normalized softplus adapts to variable Vg2)
        return _fit_pentode(points, least_squares)


# ---------------------------------------------------------------------------
# Vectorized model evaluation for fitting
# ---------------------------------------------------------------------------

def _eval_pentode_vec(
    ua: np.ndarray, ug1: np.ndarray, ug2: np.ndarray,
    p: DempwolfParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized pentode Ia, Ig2 (A) for fitting."""
    vpk = np.maximum(ua, _V_MIN)
    vg2k = np.maximum(ug2, _V_MIN)

    # cathode emission (Koren-style Vg2-normalized softplus)
    arg_k = np.clip(p.C * (1.0 / p.mu + ug1 / vg2k), -_EXP_CLIP, _EXP_CLIP)
    sp = (vg2k / p.C) * np.log1p(np.exp(arg_k))
    ik = p.G * np.maximum(sp, 0.0) ** p.gamma

    # Durchgriff
    if p.A > 0:
        ik = ik * (1.0 + p.A * vpk)

    # grid current
    arg_g = np.clip(p.Cg * ug1, -_EXP_CLIP, _EXP_CLIP)
    sp_g = np.log1p(np.exp(arg_g)) / p.Cg
    igk = p.Gg * np.maximum(sp_g, 0.0) ** p.xi

    i_through = np.maximum(ik - igk, 0.0)

    # current splitting
    v_eff = np.maximum(vg2k / p.mu + ug1, 0.0)
    kvb_eff = np.maximum(p.Kvb + p.Kvb1 * v_eff, _KVB_MIN)
    alpha = (1.0 - p.fg2) * _TWO_OVER_PI * np.arctan(
        (vpk / kvb_eff) ** p.Kn
    )

    ipk = i_through * alpha
    ig2k = i_through * (1.0 - alpha)

    # secondary emission
    if p.sigma > 0:
        vco = vg2k / p.lam - p.nu * ug1 - p.w
        vco_safe = np.maximum(vco, _V_MIN)
        x = np.maximum(1.0 - vpk / vco_safe, 0.0)
        i_sec = p.sigma * i_through * (vpk / vg2k) * x * np.exp(-p.Ks * x)
        ipk = ipk - i_sec
        ig2k = ig2k + i_sec

    return ipk, ig2k


# ---------------------------------------------------------------------------
# Quick mu estimation (Koren-style, no full fit)
# ---------------------------------------------------------------------------

def _estimate_mu_from_data(ua: np.ndarray, ug1: np.ndarray, ia: np.ndarray) -> float:
    """Estimate amplification factor mu from measurement data.

    Uses the ratio of Ua to Ug1 at the onset of conduction:
    mu ≈ Ua_mid / |Ug1_cutoff|, where Ug1_cutoff is where Ia first
    becomes significant at median Ua.

    Falls back to simple heuristic if data is insufficient.
    """
    ua_med = float(np.median(ua))
    # Find Ug1 cutoff: the most negative Ug1 where Ia > 5% of max
    ia_threshold = float(np.max(ia)) * 0.05
    conducting = ug1[ia > ia_threshold]
    if len(conducting) < 2:
        # Fallback: ratio of ranges
        ug1_range = float(np.max(ug1) - np.min(ug1))
        return max(2.0, ua_med / max(ug1_range, 1.0))
    ug1_cutoff = float(np.min(conducting))
    if abs(ug1_cutoff) < 0.5:
        return 50.0  # very high mu
    mu_est = ua_med / abs(ug1_cutoff)
    return float(np.clip(mu_est, 2.0, 200.0))


# ---------------------------------------------------------------------------
# Phase 1: Triode core — µ, G, γ, C
# ---------------------------------------------------------------------------

def _fit_phase1(ua, ug1, ia, least_squares, ref=None, ug2=None, ig2=None,
                tracker=None):
    """Fit cathode emission params (µ, G, γ, C).

    For triode: IK = G · softplus(C · (Vpk/µ + Vgrid_eff))^γ
    For pentode: IK = G · softplus(C · (Vg2k/µ + Vgk))^γ

    Pentode mode fits to total cathode current Ik = Ia + Ig2 (no splitting
    approximation needed), following Reefman's approach.
    Uses only points with VGK < -0.5V where grid current ≈ 0.
    """
    is_pentode = ug2 is not None

    mask = ug1 < -0.5
    if np.sum(mask) < 5:
        mask = np.ones(len(ug1), dtype=bool)

    ua_f, ug1_f, ia_f = ua[mask], ug1[mask], ia[mask]
    ug2_f = ug2[mask] if is_pentode else None

    # For pentodes with Ig2 data, fit to total cathode current Ik = Ia + Ig2
    # This eliminates the need for approximate current splitting in Phase 1
    if is_pentode and ig2 is not None:
        ig2_f = ig2[mask]
        ik_f = ia_f + ig2_f  # total cathode current
    else:
        ik_f = ia_f  # triode or no Ig2 data: Ik ≈ Ia

    if is_pentode:
        def residual(x):
            mu, G, gamma, C = x
            vg2k = np.maximum(ug2_f, _V_MIN)
            arg = np.clip(C * (1.0 / mu + ug1_f / vg2k), -_EXP_CLIP, _EXP_CLIP)
            sp = (vg2k / C) * np.log1p(np.exp(arg))
            ik_pred = G * np.maximum(sp, 0.0) ** gamma
            return ik_pred - ik_f
    else:
        def residual(x):
            mu, G, gamma, C = x
            vpk = np.maximum(ua_f, _V_MIN)
            v_grid_eff = ug1_f * vpk / np.sqrt(_KVB_T_DEFAULT_V2 + vpk * vpk)
            arg = np.clip(C * (vpk / mu + v_grid_eff), -_EXP_CLIP, _EXP_CLIP)
            sp = np.log1p(np.exp(arg)) / C
            ia_pred = G * np.maximum(sp, 0.0) ** gamma
            return ia_pred - ik_f

    # --- Koren-seeded initial guess + adaptive bounds ---
    # Step 1: quick Koren fit to estimate mu (always converges)
    koren_mu = _estimate_mu_from_data(ua_f, ug1_f, ik_f)

    # Step 2: adaptive bounds based on estimated mu
    if is_pentode:
        bounds_lo = [2.0, 1e-5, 0.8, 1.0]
        bounds_hi = [500.0, 1e-1, 2.5, 500.0]
    elif koren_mu < 8.0:
        # Low-mu power triodes (6S19P, 6C33C): mu 2-8, high G, low C
        bounds_lo = [1.5, 1e-5, 0.8, 0.05]
        bounds_hi = [15.0, 0.5, 2.5, 10.0]
    elif koren_mu < 30.0:
        # Medium-mu triodes (12AU7, 6SN7): mu 15-25
        bounds_lo = [2.0, 1e-5, 0.8, 0.1]
        bounds_hi = [100.0, 0.1, 2.5, 20.0]
    else:
        # High-mu triodes (12AX7): mu 50-100
        bounds_lo = [2.0, 1e-5, 0.8, 0.5]
        bounds_hi = [500.0, 0.1, 2.5, 50.0]

    # Step 3: initial guess from Koren mu + data-adaptive G
    if ref:
        x0 = [ref.mu, ref.G, ref.gamma, ref.C]
    elif is_pentode:
        x0 = [12.0, 3e-3, 1.35, 48.0]
    else:
        ia_max = float(np.max(ik_f))
        ua_mid = float(np.median(ua_f))
        g_est = ia_max / max(ua_mid / max(koren_mu, 1.0), 1.0) ** 1.35
        g_est = float(np.clip(g_est, bounds_lo[1], bounds_hi[1]))
        c_est = max(bounds_lo[3], min(3.0, bounds_hi[3]))
        x0 = [koren_mu, g_est, 1.35, c_est]

    result = least_squares(
        residual, x0, bounds=(bounds_lo, bounds_hi),
        method="trf", max_nfev=5000,
    )
    _check(tracker, result, "dempwolf phase1 (cathode)",
           n_points=len(ua_f))
    return result.x[:4]  # [mu, G, gamma, C]


# ---------------------------------------------------------------------------
# Phase 2: Grid current — Gg, ξ, Cg  (+ optionally Kvb_t for triodes)
# ---------------------------------------------------------------------------

def _fit_phase2(ua, ug1, ia, phase1, least_squares, fit_kvb_t=False,
                ug2=None, ig2=None, tracker=None, warnings_out=None):
    """Fit grid current params using points near/above VGK = 0.

    Mirrors phase 1's cathode-current form: a triode uses the Vpk-normalized
    softplus, a pentode (``ug2`` given) uses the Vg2k-normalized softplus so the
    large pentode C does not push the argument past ``_EXP_CLIP`` and degenerate
    Ik to a constant. For a pentode the residual targets the total cathode-side
    current Ik − Igk = Ia + Ig2 (matching how phase 1 fits Ik = Ia + Ig2);
    without Ig2 data it falls back to Ia.

    If no grid current data, returns defaults.
    """
    mu, G, gamma, C = phase1
    is_pentode = ug2 is not None

    mask_grid = ug1 > -1.0
    if np.sum(mask_grid) < 3:
        # No grid current data — use defaults
        return list(_GRID_CURRENT_DEFAULTS), _KVB_T_DEFAULT_V2

    ua_f, ug1_f, ia_f = ua[mask_grid], ug1[mask_grid], ia[mask_grid]
    # The pentode cathode form only helps when the grid params are determinable
    # (≥2 distinct Ug1 levels in the grid region). With a single level the grid
    # fit is underdetermined and the pentode seed can destabilize phase 4 — keep
    # the triode form there (original, stable behaviour).
    n_grid_levels = len(np.unique(np.round(ug1_f, _PHASE2_UG1_GROUP_ROUND)))
    use_pentode = is_pentode and n_grid_levels >= 2
    if is_pentode and not use_pentode:
        log.warning("dempwolf pentode: only %d distinct Ug1 level(s) in the grid "
                    "region (Ug1>-1V) — phase 2 falls back to the triode-form "
                    "cathode (grid fit underdetermined)", n_grid_levels)
        if warnings_out is not None:
            warnings_out.append({"code": MODEL_WARN_DEMPWOLF_PHASE2_TRIODE_FORM,
                                 "n": n_grid_levels})
    ug2_f = ug2[mask_grid] if use_pentode else None
    if use_pentode and ig2 is not None:
        # Pentode: Ik − Igk = Ia + Ig2 (anode + screen current).
        ik_target = ia_f + ig2[mask_grid]
    else:
        ik_target = ia_f

    x0 = list(_GRID_CURRENT_DEFAULTS)
    bounds_lo = [1e-6, 0.8, 1.0]
    bounds_hi = [1e-2, 2.5, 50.0]

    if fit_kvb_t:
        x0.append(_KVB_T_DEFAULT_V2)
        bounds_lo.append(10.0)
        bounds_hi.append(5000.0)

    def residual(x):
        Gg, xi, Cg = x[0], x[1], x[2]
        Kvb_t = x[3] if fit_kvb_t else _KVB_T_DEFAULT_V2
        if use_pentode:
            vg2k = np.maximum(ug2_f, _V_MIN)
            arg_k = np.clip(C * (1.0 / mu + ug1_f / vg2k),
                            -_EXP_CLIP, _EXP_CLIP)
            sp_k = (vg2k / C) * np.log1p(np.exp(arg_k))
        else:
            vpk = np.maximum(ua_f, _V_MIN)
            v_grid_eff = ug1_f * vpk / np.sqrt(Kvb_t + vpk * vpk)
            arg_k = np.clip(C * (vpk / mu + v_grid_eff), -_EXP_CLIP, _EXP_CLIP)
            sp_k = np.log1p(np.exp(arg_k)) / C
        ik = G * np.maximum(sp_k, 0.0) ** gamma

        arg_g = np.clip(Cg * ug1_f, -_EXP_CLIP, _EXP_CLIP)
        sp_g = np.log1p(np.exp(arg_g)) / Cg
        igk = Gg * np.maximum(sp_g, 0.0) ** xi

        ia_pred = np.maximum(ik - igk, 0.0)
        return ia_pred - ik_target

    result = least_squares(
        residual, x0, bounds=(bounds_lo, bounds_hi),
        method="trf", max_nfev=3000,
    )
    _check(tracker, result, "dempwolf phase2 (grid current)",
           n_points=int(np.sum(mask_grid)),
           fit_kvb_t=fit_kvb_t)
    grid_params = list(result.x[:3])
    kvb_t = float(result.x[3]) if fit_kvb_t else _KVB_T_DEFAULT_V2
    return grid_params, kvb_t


# ---------------------------------------------------------------------------
# Triode joint refinement — all 8 parameters on the full dataset
# ---------------------------------------------------------------------------

def _fit_triode_refine(
    ua: np.ndarray, ug1: np.ndarray, ia: np.ndarray,
    seed: List[float], least_squares,
    fit_grid: bool = True,
    tracker: Optional[ConvergenceTracker] = None,
) -> np.ndarray:
    """Jointly refine the triode parameters on ALL points.

    Phases 1–2 alone leave the triode under-fit: phase 1 freezes Kvb_t at the
    default while fitting the cathode params, and phase 2 fits Kvb_t only on
    the grid-region subset (Ug1 > −1 V) — the wrong region for a Region-A
    parameter that acts at LOW Va (v_grid_eff departs from Vgk when
    Vpk² ≲ Kvb_t). The joint pass refits against the shipped-model formula
    (igk subtracted) on the full dataset and is kept only if it improves the
    Ia MSE, so it can never regress a dataset.

    ``fit_grid=False`` (no grid-region points in the data) freezes Gg/ξ/Cg at
    the seed values: with igk ≈ 0 everywhere they are unidentifiable and would
    wander to the bounds, exporting fictional grid current into SPICE models.

    Kvb_t is multi-started over a log grid: C (softplus width) and Kvb_t both
    shape the low-current region, and a single start from the phase-2 value
    can settle in a correlated local minimum.
    """
    seed_arr = np.clip(np.asarray(seed, dtype=float),
                       _TRIODE_REFINE_LO, _TRIODE_REFINE_HI)
    gg_fixed, xi_fixed, cg_fixed = seed_arr[4:7]

    def _model(x: np.ndarray) -> np.ndarray:
        if fit_grid:
            mu, G, gamma, C, Gg, xi, Cg, kvb_t = x
        else:
            mu, G, gamma, C, kvb_t = x
            Gg, xi, Cg = gg_fixed, xi_fixed, cg_fixed
        vpk = np.maximum(ua, _V_MIN)
        v_grid_eff = ug1 * vpk / np.sqrt(kvb_t + vpk * vpk)
        sp = _softplus_vec(C * (vpk / mu + v_grid_eff), C)
        ik = G * np.maximum(sp, 0.0) ** gamma
        igk = Gg * np.maximum(_softplus_vec(Cg * ug1, Cg), 0.0) ** xi
        return np.maximum(ik - igk, 0.0)

    def residual(x: np.ndarray) -> np.ndarray:
        return _model(x) - ia

    if fit_grid:
        x0_full = seed_arr
        lo, hi = _TRIODE_REFINE_LO, _TRIODE_REFINE_HI
    else:
        x0_full = np.concatenate([seed_arr[:4], seed_arr[7:8]])
        lo = _TRIODE_REFINE_LO[:4] + _TRIODE_REFINE_LO[7:8]
        hi = _TRIODE_REFINE_HI[:4] + _TRIODE_REFINE_HI[7:8]

    best_x = x0_full
    best_mse = float(np.mean((_model(x0_full) - ia) ** 2))
    seed_mse = best_mse

    kvbt_starts = {float(x0_full[-1])} | set(_TRIODE_REFINE_KVBT_STARTS)
    for kvbt0 in sorted(kvbt_starts):
        x0 = x0_full.copy()
        x0[-1] = kvbt0
        result = least_squares(
            residual, x0, bounds=(lo, hi), method="trf", max_nfev=20000,
        )
        _check(tracker, result, "dempwolf triode joint refine",
               n_points=len(ua), kvbt_start=kvbt0, fit_grid=fit_grid)
        mse = float(np.mean((_model(result.x) - ia) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_x = np.asarray(result.x, dtype=float)

    if best_mse >= seed_mse:
        log.warning("dempwolf triode joint refine did not improve MSE "
                    "(%.4g A^2) — keeping the phase 1-2 seed", seed_mse)

    if fit_grid:
        return best_x
    return np.concatenate([best_x[:4], [gg_fixed, xi_fixed, cg_fixed],
                           best_x[4:5]])


# ---------------------------------------------------------------------------
# Phase 3: Pentode knee — Kvb, Kvb1, fg2, A
# ---------------------------------------------------------------------------

def _fit_phase3(ua, ug1, ug2, ia, ig2, phase1, phase2, least_squares,
                tracker=None):
    """Fit knee parameters from pentode data."""
    mu, G, gamma, C = phase1
    Gg, xi, Cg = phase2

    # fg2 initial estimate from high-Va data
    has_ig2 = ig2 is not None and np.any(ig2 > 1e-5)
    fg2_est = 0.05
    if has_ig2:
        vg2_med = np.median(ug2)
        hi_va = ua > 2.0 * vg2_med
        if np.sum(hi_va) > 3:
            ratio = ig2[hi_va] / np.maximum(ia[hi_va] + ig2[hi_va], 1e-9)
            fg2_est = float(np.clip(np.median(ratio), 0.01, 0.25))

    x0 = [24.0, 0.3, fg2_est, 0.0002, 1.0]
    bounds_lo = [1.0, 0.0, 0.0, 0.0, 0.7]
    bounds_hi = [200.0, 5.0, 0.30, 0.001, 2.0]

    w_ig2 = 0.3

    def residual(x):
        Kvb, Kvb1, fg2, A, Kn = x
        p = DempwolfParams(
            mu=mu, G=G, gamma=gamma, C=C,
            Gg=Gg, xi=xi, Cg=Cg,
            Kvb=Kvb, Kvb1=Kvb1, Kn=Kn, fg2=fg2, A=A,
        )
        ia_pred, ig2_pred = _eval_pentode_vec(ua, ug1, ug2, p)
        r_ia = ia_pred - ia
        if has_ig2:
            r_ig2 = (ig2_pred - ig2) * np.sqrt(w_ig2)
            return np.concatenate([r_ia, r_ig2])
        return r_ia

    result = least_squares(
        residual, x0, bounds=(bounds_lo, bounds_hi),
        method="trf", max_nfev=5000,
    )
    _check(tracker, result, "dempwolf phase3 (knee)",
           n_points=len(ua), has_ig2=has_ig2)
    return result.x  # [Kvb, Kvb1, fg2, A, Kn]


# ---------------------------------------------------------------------------
# Phase 4: Full pentode refinement
# ---------------------------------------------------------------------------

def _fit_phase4(ua, ug1, ug2, ia, ig2, phase1, phase2, phase3,
                least_squares, tracker=None):
    """Refine all pentode parameters simultaneously."""
    mu, G, gamma, C = phase1
    Gg, xi, Cg = phase2
    Kvb, Kvb1, fg2, A, Kn = phase3

    has_ig2 = ig2 is not None and np.any(ig2 > 1e-5)
    w_ig2 = 0.3

    # 12 parameters: mu, G, gamma, C, Gg, xi, Cg, Kvb, Kvb1, fg2, A, Kn
    x0 = list(np.clip([mu, G, gamma, C, Gg, xi, Cg, Kvb, Kvb1, fg2, A, Kn],
                      _PENTODE_BOUNDS_LO, _PENTODE_BOUNDS_HI))
    bounds_lo = _PENTODE_BOUNDS_LO
    bounds_hi = _PENTODE_BOUNDS_HI

    def _make_params(x):
        return DempwolfParams(
            mu=x[0], G=x[1], gamma=x[2], C=x[3],
            Gg=x[4], xi=x[5], Cg=x[6],
            Kvb=x[7], Kvb1=x[8], Kn=x[11], fg2=x[9], A=x[10],
        )

    def residual(x):
        p = _make_params(x)
        ia_pred, ig2_pred = _eval_pentode_vec(ua, ug1, ug2, p)
        r_ia = ia_pred - ia
        if has_ig2:
            r_ig2 = (ig2_pred - ig2) * np.sqrt(w_ig2)
            return np.concatenate([r_ia, r_ig2])
        return r_ia

    def _ia_mse(x):
        p = _make_params(x)
        ia_pred, _ = _eval_pentode_vec(ua, ug1, ug2, p)
        return float(np.mean((ia_pred - ia) ** 2))

    # Start 0: standard L2 from phases 1-3 result
    result0 = least_squares(
        residual, x0, bounds=(bounds_lo, bounds_hi),
        method="trf", max_nfev=20000,
    )
    _check(tracker, result0, "dempwolf phase4 start=0 (full L2)",
           n_points=len(ua))
    best_x, best_cost = result0.x, _ia_mse(result0.x)

    # Starts 1-5: perturbed with robust loss to escape local minima.
    # These are EXPECTED to occasionally not converge (perturbed init may
    # land in a bad spot); we report at DEBUG so they don't spam logs.
    rng = np.random.default_rng(42)
    for start_i in range(5):
        x0_i = []
        for j, v in enumerate(x0):
            lo, hi = bounds_lo[j], bounds_hi[j]
            perturbed = v * (1.0 + rng.uniform(-0.15, 0.15))
            x0_i.append(np.clip(perturbed, lo, hi))
        result_i = least_squares(
            residual, x0_i, bounds=(bounds_lo, bounds_hi),
            method="trf", loss="soft_l1", f_scale=5e-3, max_nfev=20000,
        )
        if not getattr(result_i, "success", True):
            log.debug("dempwolf phase4 start=%d (perturbed) did not converge "
                      "(status=%s, cost=%.4g) — discarding if worse than best",
                      start_i + 1, getattr(result_i, "status", None),
                      getattr(result_i, "cost", float("nan")))
        cost_i = _ia_mse(result_i.x)
        if cost_i < best_cost:
            best_cost = cost_i
            best_x = result_i.x

    return best_x  # [mu, G, gamma, C, Gg, xi, Cg, Kvb, Kvb1, fg2, A, Kn]


# ---------------------------------------------------------------------------
# Phase 5: Secondary emission — σ, Ks, λ, ν, w
# ---------------------------------------------------------------------------

def _has_kink(
    ua: np.ndarray, ug1: np.ndarray, ug2: np.ndarray, ia: np.ndarray,
) -> bool:
    """Detect real negative-resistance (dynatron) kinks in the data.

    Checks per-(Ug1, Ug2) curves in the knee region (Va < Vg2 of the curve).
    Each point belongs to exactly one 0.1 V Ug1 × 1 V Ug2 bucket (np.round,
    no overlapping window), so neither a ±0.1 V neighbour Ug1 curve nor a
    different Ug2 level can interleave a spurious dip. The metric is the
    cumulative drawdown (running Ia peak minus Ia along ascending Ua); it
    counts only when deeper than max(_KINK_DRAWDOWN_REL · curve max,
    _KINK_DRAWDOWN_MIN_A) — see the constants block for the measured
    artifact-vs-dynatron populations behind the thresholds.
    """
    ug1_keys = np.round(ug1, _KINK_UG1_GROUP_ROUND)
    ug2_keys = np.round(ug2, _KINK_UG2_GROUP_ROUND)
    for vg in np.unique(ug1_keys):
        mask_g1 = ug1_keys == vg
        for vg2 in np.unique(ug2_keys[mask_g1]):
            curve_mask = mask_g1 & (ug2_keys == vg2) & (ua < vg2)
            if np.sum(curve_mask) < 4:
                continue
            sorted_idx = np.argsort(ua[curve_mask])
            ia_sorted = ia[curve_mask][sorted_idx]
            drawdown = float(np.max(np.maximum.accumulate(ia_sorted)
                                    - ia_sorted))
            floor = max(_KINK_DRAWDOWN_REL * float(np.max(ia_sorted)),
                        _KINK_DRAWDOWN_MIN_A)
            if drawdown > floor:
                return True
    return False


def _fit_phase5(ua, ug1, ug2, ia, ig2, phase4_params, least_squares,
                tracker=None):
    """Fit secondary emission params (beam tetrodes only)."""
    mu, G, gamma, C, Gg, xi, Cg, Kvb, Kvb1, fg2, A, Kn = phase4_params

    has_ig2 = ig2 is not None and np.any(ig2 > 1e-5)
    w_ig2 = 0.3

    x0 = [1.0, 2.0, 1.0, 2.0, 0.0]
    bounds_lo = _SEC_EMISSION_BOUNDS_LO
    bounds_hi = _SEC_EMISSION_BOUNDS_HI

    def residual(x):
        sigma, Ks, lam, nu, w = x
        p = DempwolfParams(
            mu=mu, G=G, gamma=gamma, C=C,
            Gg=Gg, xi=xi, Cg=Cg,
            Kvb=Kvb, Kvb1=Kvb1, Kn=Kn, fg2=fg2, A=A,
            sigma=sigma, Ks=Ks, lam=lam, nu=nu, w=w,
        )
        ia_pred, ig2_pred = _eval_pentode_vec(ua, ug1, ug2, p)
        r_ia = ia_pred - ia
        if has_ig2:
            r_ig2 = (ig2_pred - ig2) * np.sqrt(w_ig2)
            return np.concatenate([r_ia, r_ig2])
        return r_ia

    # Multi-start over the documented λ regimes (§14.6: λ ≈ 1 for beam
    # tetrodes, ≈ 15 for pentode-like screening) × a ν spread: the 5-param
    # landscape is multimodal and a single start from (λ=1, ν=2) can settle
    # in the σ ≈ 0 minimum even when a deep kink is present (observed on the
    # real 6P1P scan). Best start wins by residual cost.
    best_x = None
    best_cost = np.inf
    for lam0 in _PHASE5_LAM_STARTS:
        for nu0 in _PHASE5_NU_STARTS:
            x0_i = [x0[0], x0[1], lam0, nu0, x0[4]]
            result = least_squares(
                residual, x0_i, bounds=(bounds_lo, bounds_hi),
                method="trf", max_nfev=5000,
            )
            _check(tracker, result, "dempwolf phase5 (secondary emission)",
                   n_points=len(ua), lam_start=lam0, nu_start=nu0)
            if result.cost < best_cost:
                best_cost = result.cost
                best_x = result.x
    return best_x  # [sigma, Ks, lam, nu, w]


# ---------------------------------------------------------------------------
# Top-level fit orchestrators
# ---------------------------------------------------------------------------

def _fit_triode(points: List[Dict], least_squares) -> ModelFitResult:
    """Fit Dempwolf triode model (Phases 1 + 2)."""
    data = extract_arrays(
        points, topology=TOPOLOGY_TRIODE, ia_thr_mA=0.05, min_count=5)

    tracker = ConvergenceTracker()

    # Phase 1: cathode emission
    phase1 = _fit_phase1(data.ua, data.ug1, data.ia, least_squares,
                         tracker=tracker)
    mu, G, gamma, C = phase1

    # Phase 2: grid current + Kvb_t
    grid_params, kvb_t = _fit_phase2(
        data.ua, data.ug1, data.ia, phase1, least_squares, fit_kvb_t=True,
        tracker=tracker,
    )
    Gg, xi, Cg = grid_params

    # Joint refine on the full dataset (kept only if it improves MSE — see
    # _fit_triode_refine docstring). Same grid-data criterion as phase 2:
    # without grid-region points Gg/xi/Cg are unidentifiable and stay frozen.
    has_grid_data = int(np.sum(data.ug1 > -1.0)) >= 3
    fit_warnings: list = []
    if not has_grid_data:
        # ML-101: Gg/xi/Cg stay FROZEN at defaults (unidentifiable without
        # grid-region points) — the SPICE export gets a default grid
        # current, not a fitted one. Every real LM19 scan (no positive
        # grid) hits this; the model dialog must say so.
        log.warning("dempwolf triode: no grid-region points (Ug1 > -1 V) — "
                    "Gg/xi/Cg frozen at defaults; exported grid current is "
                    "not fitted")
        fit_warnings.append({"code": MODEL_WARN_DEMPWOLF_GRID_DEFAULTS})
    refined = _fit_triode_refine(
        data.ua, data.ug1, data.ia,
        [mu, G, gamma, C, Gg, xi, Cg, kvb_t],
        least_squares, fit_grid=has_grid_data, tracker=tracker,
    )
    mu, G, gamma, C, Gg, xi, Cg, kvb_t = refined

    params = DempwolfParams(
        mu=float(mu), G=float(G), gamma=float(gamma), C=float(C),
        Gg=float(Gg), xi=float(xi), Cg=float(Cg), Kvb_t=float(kvb_t),
    )

    # Quality metric must mirror the shipped model (igk subtracted): the old
    # fitting-only kernel omitted grid current and overstated rms up to 7x on
    # tubes with grid-region data (6N5P: reported 18.2 mA vs actual 2.5 mA).
    ia_pred = dempwolf_v2_ia_vec(data.ua, data.ug1, None, p=params)
    model = DempwolfModel(name="fit", topology=TOPOLOGY_TRIODE, dempwolf=params)

    return build_fit_result(
        model_type=MODEL_TYPE_DEMPWOLF,
        topology=TOPOLOGY_TRIODE,
        model=model,
        ia_pred_A=ia_pred,
        ia_meas_A=data.ia,
        n_points=data.n_points,
        converged=tracker.all_converged,
        warnings=fit_warnings,
    )


def _fit_pentode(points: List[Dict], least_squares) -> ModelFitResult:
    """Fit Dempwolf pentode / beam tetrode model (Phases 1–4, optionally 5)."""
    # triode_connected topology defaults missing ug2 → ua
    data = extract_arrays(
        points, topology=TOPOLOGY_TRIODE_CONNECTED,
        ia_thr_mA=0.05, min_count=10,
    )
    ua, ug1, ug2 = data.ua, data.ug1, data.ug2
    ia = data.ia
    has_ig2 = data.has_ig2
    ig2 = data.ig2
    # ig2_raw kept for ig2 error metrics — same as data.ig2 if has_ig2 else
    # zeros (we keep raw zeros to allow has_ig2=False path to skip entirely).
    ig2_raw = ig2 if has_ig2 else np.zeros_like(ia)

    tracker = ConvergenceTracker()
    fit_warnings: list = []

    is_triode_connected = bool(np.median(np.abs(ug2 - ua)) < 5.0)

    # Beam-tetrode detection (gates phase 5 only): real per-curve dynatron
    # dips. Decoupled from the phase-1/2 masking below — the old code masked
    # phases 1-4 whenever kink fired, so the knee parameters (Kvb/Kn/fg2)
    # were fit WITHOUT the knee data and phase 5's σ then acted as a
    # knee-shaping crutch on true pentodes.
    kink_detected = _has_kink(ua, ug1, ug2, ia)

    # Phases 1-2 (cathode/grid) masking criterion: Ig2 data availability.
    # With Ig2 the target Ik = Ia + Ig2 is knee-independent (exact current
    # conservation) — the full dataset is usable. Without Ig2 the Ia-only
    # target equals Ik·α, and below-knee points (α < 1) poison the cathode
    # fit — mask them. Phases 3-4 ALWAYS see the full dataset: the knee is
    # what they fit.
    use_mask = (not has_ig2) and (not is_triode_connected)
    if use_mask:
        safe_mask = ua > np.median(ug2) * _KNEE_SAFE_FACTOR
        n_masked = int(np.sum(~safe_mask))
        if np.sum(safe_mask) < 10:
            # Too few above-knee points to fit — keep the full dataset rather
            # than fitting <10 points.
            log.warning("dempwolf pentode: no Ig2 data and only %d/%d points "
                        "above the knee — phases 1-2 use the full dataset "
                        "(cathode fit may be biased)",
                        int(np.sum(safe_mask)), len(ua))
            fit_warnings.append({"code": MODEL_WARN_DEMPWOLF_NO_IG2_FULL,
                                 "n": int(np.sum(safe_mask)),
                                 "total": len(ua)})
            safe_mask = np.ones(len(ua), dtype=bool)
        else:
            log.warning("dempwolf pentode: no Ig2 data — masking %d/%d "
                        "below-knee points (Va < ~%.1f·Vg2) for phases 1-2",
                        n_masked, len(ua), _KNEE_SAFE_FACTOR)
            fit_warnings.append({"code": MODEL_WARN_DEMPWOLF_NO_IG2_MASK,
                                 "n": n_masked, "total": len(ua)})
        ua_s, ug1_s, ug2_s = ua[safe_mask], ug1[safe_mask], ug2[safe_mask]
        ia_s = ia[safe_mask]
        ig2_s = ig2[safe_mask] if ig2 is not None else None
    else:
        ua_s, ug1_s, ug2_s, ia_s, ig2_s = ua, ug1, ug2, ia, ig2

    # Phase 1: cathode emission (pentode formula: Vg2k/µ + Vgk)
    # For triode_connected (Ug2≈Ua), don't pass ig2 — fit to Ia only
    ig2_for_phase1 = None if is_triode_connected else ig2_s
    phase1 = _fit_phase1(ua_s, ug1_s, ia_s, least_squares, ug2=ug2_s,
                         ig2=ig2_for_phase1, tracker=tracker)

    # Phase 2: grid current. TRUE pentodes use the Vg2-aware cathode form with
    # the Ik=Ia+Ig2 target; triode_connected (Vg2≈Va) is physically a triode, so
    # it keeps the triode-form phase 2 (ug2=None) — the pentode form parameterizes
    # the grid region worse there and regresses the fit.
    ug2_for_phase2 = None if is_triode_connected else ug2_s
    grid_params, _ = _fit_phase2(ua_s, ug1_s, ia_s, phase1, least_squares,
                                 warnings_out=fit_warnings,
                                 ug2=ug2_for_phase2, ig2=ig2_for_phase1,
                                  tracker=tracker)

    # Phase 3: knee — always on the FULL dataset (below-knee points carry
    # the very shape Kvb/Kvb1/Kn/fg2 parameterize)
    phase3 = _fit_phase3(ua, ug1, ug2, ia, ig2, phase1,
                         grid_params, least_squares, tracker=tracker)

    # Phase 4: full refinement — always on the FULL dataset
    phase4 = _fit_phase4(ua, ug1, ug2, ia, ig2, phase1,
                         grid_params, phase3, least_squares, tracker=tracker)
    mu, G, gamma, C, Gg, xi, Cg, Kvb, Kvb1, fg2, A, Kn = phase4

    # Phase 5: secondary emission (only if kink detected)
    sigma = Ks = lam = nu = w = 0.0
    beam = False
    if kink_detected:
        se_params = _fit_phase5(ua, ug1, ug2, ia, ig2, phase4, least_squares,
                                tracker=tracker)
        sigma, Ks, lam, nu, w = [float(v) for v in se_params]
        beam = sigma > 0.01

        if beam:
            # Joint refinement: refit all params together on full data.
            # Bounds are the phase-4/phase-5 ones (docs §10.6/§14.7) with a
            # σ floor keeping the fit in the beam regime it was gated into.
            all_x0 = [mu, G, gamma, C, Gg, xi, Cg, Kvb, Kvb1, fg2, A, Kn,
                       sigma, Ks, lam, nu, w]
            bounds_lo = (_PENTODE_BOUNDS_LO
                         + [_JOINT_SIGMA_MIN] + _SEC_EMISSION_BOUNDS_LO[1:])
            bounds_hi = _PENTODE_BOUNDS_HI + _SEC_EMISSION_BOUNDS_HI
            all_x0 = list(np.clip(all_x0, bounds_lo, bounds_hi))

            w_ig2 = 0.3

            def joint_residual(x):
                p = DempwolfParams(
                    mu=x[0], G=x[1], gamma=x[2], C=x[3],
                    Gg=x[4], xi=x[5], Cg=x[6],
                    Kvb=x[7], Kvb1=x[8], Kn=x[11], fg2=x[9], A=x[10],
                    sigma=x[12], Ks=x[13], lam=x[14], nu=x[15], w=x[16],
                )
                ia_pred, ig2_pred = _eval_pentode_vec(ua, ug1, ug2, p)
                r_ia = ia_pred - ia
                if has_ig2:
                    r_ig2 = (ig2_pred - ig2) * np.sqrt(w_ig2)
                    return np.concatenate([r_ia, r_ig2])
                return r_ia

            result = least_squares(
                joint_residual, all_x0,
                bounds=(bounds_lo, bounds_hi),
                method="trf", max_nfev=10000,
            )
            _check(tracker, result, "dempwolf joint refine (with sec.emission)",
                   n_points=len(ua))
            (mu, G, gamma, C, Gg, xi, Cg, Kvb, Kvb1, fg2, A, Kn,
             sigma, Ks, lam, nu, w) = [float(v) for v in result.x]

    params = DempwolfParams(
        mu=float(mu), G=float(G), gamma=float(gamma), C=float(C),
        Gg=float(Gg), xi=float(xi), Cg=float(Cg),
        Kvb=float(Kvb), Kvb1=float(Kvb1), Kn=float(Kn),
        fg2=float(fg2), A=float(A),
        sigma=float(sigma), Ks=float(Ks),
        lam=float(lam), nu=float(nu), w=float(w),
    )

    # Compute fit quality
    ia_pred, ig2_pred = _eval_pentode_vec(ua, ug1, ug2, params)

    topo_out = TOPOLOGY_PENTODE  # beam tetrode is a pentode in configs
    model = DempwolfModel(name="fit", topology=topo_out, dempwolf=params)

    ig2_pred_for_metric = ig2_pred if (has_ig2 and ig2_pred is not None) else None
    ig2_meas_for_metric = ig2_raw if (has_ig2 and ig2_pred is not None) else None

    return build_fit_result(
        warnings=fit_warnings,
        model_type=MODEL_TYPE_DEMPWOLF,
        topology=topo_out,
        model=model,
        ia_pred_A=ia_pred,
        ia_meas_A=ia,
        n_points=data.n_points,
        ig2_pred_A=ig2_pred_for_metric,
        ig2_meas_A=ig2_meas_for_metric,
        converged=tracker.all_converged,
    )


# ---------------------------------------------------------------------------
# Register in model registry
# ---------------------------------------------------------------------------
register_model(
    model_type=MODEL_TYPE_DEMPWOLF,
    label="Dempwolf v2",
    loader=load_dempwolf_model,
    fitter=fit_dempwolf,
    list_tubes=_list_dempwolf_tubes,
)
