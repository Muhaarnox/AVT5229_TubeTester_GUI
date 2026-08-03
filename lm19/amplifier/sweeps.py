"""Parameter sweeps + headroom/Pa/IMD/Ig2 helpers.

Public functions:
  - ``compute_headroom``     — max symmetric swing limited by cutoff/grid/Pa
  - ``compute_pa_avg``       — true Pa over signal cycle (numerical)
  - ``compute_pg2``          — Pg2 = Ug2 × Ig2 (mW)
  - ``estimate_ig2_at_q``    — Ig2 at Q-point from nearby points
  - ``sweep_amplitude``      — THD vs Pout at varying half-swing
  - ``sweep_ra``             — THD/Pout vs Ra (resistive or transformer)
  - ``sweep_bias``           — THD vs Ug1 bias
  - ``sweep_pp_amplitude``   — push-pull amplitude sweep on composite
  - ``optimize_bias``        — pick best Ug1 for chosen target

Internal ``_compute_hd`` dispatches to 5-point / Chebyshev / DFT methods.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from lm19.amplifier.constants import (
    CUTOFF_IA_MA,
    IK_NEAR_ZERO_MA,
    MIN_SWEEP_SWING_V,
    MIN_SWING_V,
    N_PA_SAMPLES,
    Q_WINDOW_UA_V,
    Q_WINDOW_UG1_V,
    Q_WINDOW_UG2_V,
    SWEEP_BIAS_MARGIN,
)
from lm19.amplifier.distortion import (
    _DFT_NEWTON_MAX_ITER_SE,
    _find_model_dc_q_point,
    _newton_solve_vec,
    composite_characteristic,
    compute_distortion,
    compute_distortion_chebyshev,
    compute_distortion_dft,
    compute_imd,
    find_intersections,
    find_intersections_model,
    interp_intersection,
    pp_distortion,
)
from lm19.amplifier.loadlines import (
    LoadLine,
    PushPullLoadLine,
    ResistiveLoadLine,
    TransformerLoadLine,
)
from lm19.amplifier.stage_params import compute_stage_params
from lm19.tube_model_base import model_ia_array
from lm19.constants import (
    DEFAULT_UB_V,
    MODEL_UA_MAX_DEFAULT_V,
    MODEL_UA_MIN_V,
    MW_PER_W,
    UA_ROUND,
)
from lm19.amplifier.constants import (
    HD_METHOD_5POINT,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_DFT,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lm19.tube_model_base import TubeModelProtocol


# ── module local constants ──
# Clip argument to exp() in the inline grid-current softplus — float64
# overflow guard only (exp overflows at ~709). A tight clip (e.g. 50) is
# the same saturation bug class as the model kernels' old clips
# (lm19/dempwolf.py / spice_export/koren.py): the informational Ig1 was
# understated once Cg*Vgk_at_pos > 50 — reachable only with a positive
# ug1_bias (A2 analysis), where Vgk_at_pos = 2*bias. Pinned in
# tests/test_model_exp_clip_pins.py.
_EXP_CLIP = 700.0


# ─── Headroom (max swing) ────────────────────────────────────────────

def compute_headroom(
    intersections: List[Dict],
    ug1_bias: float,
    pa_max: Optional[float] = None,
    load_line: Optional[LoadLine] = None,
    grid_current_params: Optional[Dict] = None,
) -> Optional[Dict]:
    """Compute maximum symmetric swing before clipping.

    Determines maximum signal amplitude limited by:
    - cutoff: Ia -> 0 (negative swing limit)
    - grid_current: Ug1 -> 0 (positive swing limit)
    - pa_max: Pa = Ua * Ia > Pa_max (thermal limit)
    - data_limit: ran out of measurement data

    Args:
        grid_current_params: optional Dempwolf grid current parameters
            {"Gg": float(A), "xi": float, "Cg": float}. When provided,
            computes actual grid current at max positive swing instead
            of using the hard -0.1V threshold.
    """
    if len(intersections) < 3:
        return None

    pts = sorted(intersections, key=lambda p: p["ug1"])
    ug1_min = pts[0]["ug1"]
    ug1_max = pts[-1]["ug1"]

    q = interp_intersection(pts, ug1_bias)
    if q is None:
        return None

    pa_at_q = 0.0
    if load_line is not None:
        pa_at_q = q["ua"] * q["ia"] / MW_PER_W

    swing_neg = abs(ug1_bias - ug1_min)
    clip_neg = "data_limit"

    for p in pts:
        if p["ug1"] < ug1_bias and p["ia"] < CUTOFF_IA_MA:
            swing_neg = abs(ug1_bias - p["ug1"])
            clip_neg = "cutoff"
            break

    swing_pos = abs(ug1_max - ug1_bias)
    clip_pos = "data_limit"
    ig1_ma: Optional[float] = None

    if ug1_max >= -0.1:
        swing_pos = abs(ug1_bias)
        clip_pos = "grid_current"

    if grid_current_params is not None:
        Gg = grid_current_params.get("Gg", 0.0)
        xi = grid_current_params.get("xi", 1.3)
        Cg = grid_current_params.get("Cg", 10.0)
        if Gg > 0:
            ug1_at_pos = ug1_bias + swing_pos
            arg_g = Cg * ug1_at_pos
            sp = math.log(1.0 + math.exp(min(arg_g, _EXP_CLIP))) / Cg
            ig1_a = Gg * max(sp, 0.0) ** xi
            ig1_ma = ig1_a * 1000.0

    if pa_max is not None and load_line is not None:
        for p in pts:
            pa_point = p["ua"] * p["ia"] / MW_PER_W
            if pa_point > pa_max:
                swing_to_p = abs(p["ug1"] - ug1_bias)
                if p["ug1"] < ug1_bias and swing_to_p < swing_neg:
                    swing_neg = swing_to_p
                    clip_neg = "pa_max"
                elif p["ug1"] > ug1_bias and swing_to_p < swing_pos:
                    swing_pos = swing_to_p
                    clip_pos = "pa_max"

    max_swing = min(swing_neg, swing_pos)

    pt_at_neg = interp_intersection(pts, ug1_bias - max_swing)
    pt_at_pos = interp_intersection(pts, ug1_bias + max_swing)

    result = {
        "max_swing": max_swing,
        "clip_neg": clip_neg,
        "clip_pos": clip_pos,
        "swing_neg": swing_neg,
        "swing_pos": swing_pos,
        "ia_max": pt_at_pos["ia"] if pt_at_pos else 0.0,
        "ia_min": pt_at_neg["ia"] if pt_at_neg else 0.0,
        "pa_at_q": pa_at_q,
    }
    if ig1_ma is not None:
        result["ig1_ma"] = ig1_ma
    return result


# ─── Pa / Pg2 / Ig2 helpers ──────────────────────────────────────────

def compute_pa_avg(
    model: "TubeModelProtocol",
    load_line: "LoadLine",
    ug1_bias: float,
    half_swing: float,
    ug2: float = 0.0,
    ub: Optional[float] = None,
) -> Optional[Dict]:
    """Compute average plate dissipation over one signal cycle.

    Numerically integrates Pa = Ua × Ia over a sinusoidal Ug1 sweep
    using the tube model + load line. Accurate for all operating classes.

    For Class A:  Pa_avg ≈ Pdc − Pout  (DC current constant).
    For Class AB/B: Pa_avg differs from Pdc − Pout because average
    DC current increases with signal level.
    """
    if half_swing < MIN_SWING_V:
        return None

    ll_fn = load_line.ia_at_ua
    if isinstance(load_line, (TransformerLoadLine, PushPullLoadLine)):
        if isinstance(load_line, TransformerLoadLine):
            ra_dc, ra_ac = load_line.ra_dc, load_line.ra_ac
        else:
            ra_dc, ra_ac = load_line.ra_dc, load_line.ra_per_tube
        q = _find_model_dc_q_point(
            model, load_line.ub, ra_dc, ug1_bias, ug2,
            (MODEL_UA_MIN_V, MODEL_UA_MAX_DEFAULT_V),
        )
        if q is not None:
            q_ua, q_ia = q

            def ll_fn(ua: float, _q_ua=q_ua, _q_ia=q_ia, _ra=ra_ac) -> float:
                if _ra <= 0:
                    return _q_ia
                return _q_ia - (ua - _q_ua) / _ra

    eps = load_line.endpoints()
    ua_guess = (eps[0][0] + eps[1][0]) / 2.0 if eps else (ub or DEFAULT_UB_V)

    # ML-132: this loop held the last inline copy of the DFT Newton solver
    # (magic 0.5 / 1e-6 / ±1.0 / 30) — replaced by the shared vectorized
    # _newton_solve_vec. The drive stays math.sin and the accumulation stays
    # sequential: np.sin and pairwise np.sum differ from the scalar path by
    # ~1 ulp, which would break bit-identity with historical results.
    ug1_t = np.array([
        ug1_bias + half_swing * math.sin(2.0 * math.pi * i / N_PA_SAMPLES)
        for i in range(N_PA_SAMPLES)
    ])
    ua_t, not_converged, max_residual_ma = _newton_solve_vec(
        lambda ua_arr, g_arr: model_ia_array(model, ua_arr, g_arr, ug2),
        ll_fn, ug1_t, ua_guess, _DFT_NEWTON_MAX_ITER_SE,
    )
    ia_t = np.maximum(0.0, model_ia_array(model, ua_t, ug1_t, ug2))

    pa_sum = 0.0
    ia_sum = 0.0
    pa_peak = 0.0
    ia_peak = 0.0
    for ua_val, ia_val in zip(ua_t.tolist(), ia_t.tolist()):
        pa_val = ua_val * ia_val
        pa_sum += pa_val
        ia_sum += ia_val
        if pa_val > pa_peak:
            pa_peak = pa_val
        if ia_val > ia_peak:
            ia_peak = ia_val

    pa_avg_mw = pa_sum / N_PA_SAMPLES
    ia_avg = ia_sum / N_PA_SAMPLES

    if not_converged:
        # ML-096: the DFT twin in distortion.py reports its Newton failures;
        # Pa_avg is checked against Pa_max when picking the operating point,
        # so a value from non-converged samples must carry a signal too.
        log.warning("compute_pa_avg: %d/%d samples did not converge "
                    "(max residual %.3g mA) — Pa_avg may be inaccurate",
                    not_converged, N_PA_SAMPLES, max_residual_ma)

    result: Dict = {
        "pa_avg_mw": pa_avg_mw,
        "ia_avg": ia_avg,
        "pa_peak_mw": pa_peak,
        "ia_peak": ia_peak,
        "n_not_converged": not_converged,
    }

    if ub is not None and ub > 0 and ia_avg > 0:
        result["pdc_avg_mw"] = ub * ia_avg

    return result


def estimate_ig2_at_q(
    points: List[Dict],
    ug1_q: float,
    ua_q: float,
    ug2_filter: Optional[float] = None,
) -> float:
    """Estimate Ig2 at the Q-point from nearby measurement points.

    Returns Ig2 in mA (0.0 if no ig2 data available).
    Uses points within ±15V of Ua_q and ±1V of Ug1_q, optionally
    filtered by Ug2 level (±5V tolerance).
    """
    candidates = []
    for p in points:
        if "ig2" not in p or p["ig2"] is None:
            continue
        if ug2_filter is not None:
            ug2 = p.get("ug2", 0.0) or 0.0
            if abs(ug2 - ug2_filter) > Q_WINDOW_UG2_V:
                continue
        if abs(p["ua"] - ua_q) <= Q_WINDOW_UA_V and abs(p["ug1"] - ug1_q) <= Q_WINDOW_UG1_V:
            candidates.append(p["ig2"])
    if not candidates:
        return 0.0
    return float(np.median(candidates))


def compute_pg2(ug2: float, ig2_ma: float) -> float:
    """Screen grid dissipation: Pg2 = Ug2 × Ig2 (mW)."""
    return ug2 * ig2_ma


# ─── HD method dispatcher ────────────────────────────────────────────

def _compute_hd(
    method: str,
    isects: List[Dict],
    model: Optional["TubeModelProtocol"],
    load_line: LoadLine,
    ug1_bias: Optional[float],
    half_swing: Optional[float] = None,
    ug2: float = 0.0,
    ub: Optional[float] = None,
) -> Optional[Dict]:
    """Dispatch distortion calculation to the selected method.

    Args:
        method: "5point", "chebyshev", or "dft".
    """
    if method not in _HD_METHODS:
        # ML-097: a typo ('cheby') used to silently become 5-point.
        raise ValueError(
            f"unknown HD method {method!r}; expected one of {_HD_METHODS}")
    if method == HD_METHOD_DFT and model is None:
        # ML-097: the optimizer path surfaces this via
        # OptimizerResult.warning='dft_no_model_fallback'; the engine path
        # fell back in silence.
        log.warning("_compute_hd: 'dft' requested without a model — "
                    "falling back to 5-point")
    if method == HD_METHOD_DFT and model is not None:
        if ug1_bias is None:
            # ML-053: `ug1_bias or 0.0` evaluated the model at 0 V grid —
            # a fully open tube — and returned that THD as if it were the
            # operating point's. Derive the bias the way the interp
            # methods do (mid-range of the intersection family).
            ug1s = [p["ug1"] for p in isects if "ug1" in p]
            if not ug1s:
                return None
            ug1_bias = (min(ug1s) + max(ug1s)) / 2.0
            log.warning("_compute_hd: dft without ug1_bias — derived "
                        "mid-range bias %.2f V from intersections",
                        ug1_bias)
        hs = half_swing
        if hs is None or hs < MIN_SWING_V:
            hr = compute_headroom(isects, ug1_bias)
            hs = hr["max_swing"] if hr else None
        if hs is None or hs < MIN_SWING_V:
            return None
        return compute_distortion_dft(
            model, load_line, ug1_bias=ug1_bias,
            half_swing=hs, ug2=ug2, ub=ub,
        )
    if method == HD_METHOD_CHEBYSHEV:
        return compute_distortion_chebyshev(
            isects, ug1_bias=ug1_bias, half_swing=half_swing, ub=ub,
        )
    return compute_distortion(isects, ug1_bias=ug1_bias, half_swing=half_swing, ub=ub)


# Data methods (no AUTO — it resolves earlier); from the owner registry.
_HD_METHODS = (HD_METHOD_5POINT, HD_METHOD_CHEBYSHEV, HD_METHOD_DFT)


# ─── Sweeps ───────────────────────────────────────────────────────────

def sweep_amplitude(
    points: List[Dict],
    load_line: LoadLine,
    ug1_bias: float,
    ug2_filter: Optional[float] = None,
    steps: int = 40,
    model: Optional["TubeModelProtocol"] = None,
    model_ug2: float = 0.0,
    hd_method: str = HD_METHOD_5POINT,
) -> List[Dict]:
    """THD vs Pout — sweep signal amplitude.

    Sweep half_swing from MIN_SWEEP_SWING_V to max headroom.
    At each step computes HD2, HD3, THD, Pout, IMD.
    """
    if model is not None:
        ug1_values = sorted({round(p["ug1"], 1) for p in points})
        ua_max = max((p["ua"] for p in points), default=MODEL_UA_MAX_DEFAULT_V)
        isects = find_intersections_model(
            model, load_line, ug1_values, ug2=model_ug2,
            ua_range=(MODEL_UA_MIN_V, ua_max),
            ug1_bias=ug1_bias,
        )
    else:
        isects = find_intersections(points, load_line, ug2_filter=ug2_filter,
                                    ug1_bias=ug1_bias)
    if len(isects) < 3:
        return []

    headroom = compute_headroom(isects, ug1_bias)
    if headroom is None or headroom["max_swing"] < MIN_SWEEP_SWING_V:
        return []

    max_sw = headroom["max_swing"]
    results: List[Dict] = []

    for i in range(steps):
        hs = MIN_SWEEP_SWING_V + (max_sw - MIN_SWEEP_SWING_V) * i / max(steps - 1, 1)
        dist = _compute_hd(
            hd_method, isects, model, load_line, ug1_bias,
            half_swing=hs, ug2=model_ug2, ub=None,
        )
        if dist is None:
            continue
        imd = compute_imd(isects, ug1_bias=ug1_bias, half_swing=hs)
        results.append({
            "half_swing": hs,
            "hd2": dist["hd2"],
            "hd3": dist["hd3"],
            "hd4": dist.get("hd4", 0.0),
            "hd5": dist.get("hd5", 0.0),
            "thd": dist["thd"],
            "pout_mw": dist["pout_mw"],
            "imd2": imd["imd2"] if imd else 0.0,
            "imd3": imd["imd3"] if imd else 0.0,
        })

    return results


def sweep_ra(
    points: List[Dict],
    ub: float,
    ra_min: float = 0.5,
    ra_max: float = 50.0,
    ug1_bias: Optional[float] = None,
    half_swing: Optional[float] = None,
    ug2_filter: Optional[float] = None,
    steps: int = 60,
    model: Optional["TubeModelProtocol"] = None,
    model_ug2: float = 0.0,
    transformer: bool = False,
    ra_dc: float = 0.05,
    hd_method: str = HD_METHOD_5POINT,
) -> List[Dict]:
    """HD/Pout vs Ra — sweep load resistance.

    Args:
        transformer: if True, sweep with TransformerLoadLine (AC through
            Q-point) instead of ResistiveLoadLine. Requires ``ug1_bias``.
        ra_dc: DC winding resistance for transformer mode (kOhm).
        hd_method: "5point", "chebyshev", or "dft".
    """
    results: List[Dict] = []
    if model is not None:
        ug1_values = sorted({round(p["ug1"], 1) for p in points})
        ua_max = max((p["ua"] for p in points), default=MODEL_UA_MAX_DEFAULT_V)
    for i in range(steps):
        ra = ra_min + (ra_max - ra_min) * i / max(steps - 1, 1)
        if transformer:
            ll: LoadLine = TransformerLoadLine(ub, ra_dc=ra_dc, ra_ac=ra)
        else:
            ll = ResistiveLoadLine(ub, ra)
        if model is not None:
            isects = find_intersections_model(
                model, ll, ug1_values, ug2=model_ug2,
                ua_range=(MODEL_UA_MIN_V, ua_max),
                ug1_bias=ug1_bias,
            )
        else:
            isects = find_intersections(points, ll, ug2_filter=ug2_filter,
                                        ug1_bias=ug1_bias)
        analysis = _compute_hd(
            hd_method, isects, model, ll, ug1_bias,
            half_swing=half_swing, ug2=model_ug2, ub=ub,
        )
        if analysis:
            entry: Dict = {
                "ra": ra,
                "hd2": analysis["hd2"],
                "hd3": analysis["hd3"],
                "hd4": analysis.get("hd4", 0.0),
                "hd5": analysis.get("hd5", 0.0),
                "thd": analysis["thd"],
                "pout_mw": analysis["pout_mw"],
            }
            stage = compute_stage_params(
                isects, ll, ug1_bias=ug1_bias, points=points,
                model=model, model_ug2=model_ug2,
            )
            entry["gain"] = stage["gain"] if stage else 0.0
            entry["zout"] = stage["zout"] if stage else 0.0
            entry["pa_mw"] = analysis["ua_0"] * analysis["ia_0"]
            results.append(entry)
    return results


def sweep_ra_pp(
    points: List[Dict],
    ub: float,
    raa_min: float,
    raa_max: float,
    ug1_bias: float,
    half_swing: Optional[float] = None,
    ug2_filter: Optional[float] = None,
    steps: int = 60,
    points_b: Optional[List[Dict]] = None,
) -> List[Dict]:
    """THD/Pout vs Ra_aa for PP (5-point on the measured composite).

    The Ra-sweep dialog (circuit-aware input) — the plot-side button
    used to sweep the resistive line regardless of the circuit. The
    method is fixed (5-point composite) — the label must accompany
    the numbers (method-visibility rule).
    """
    out: List[Dict] = []
    for i in range(steps):
        raa = raa_min + (raa_max - raa_min) * i / max(steps - 1, 1)
        ll = PushPullLoadLine(ub, ra_aa=raa)
        d = pp_distortion(points, ll, ug1_bias,
                          points_b=points_b, half_swing=half_swing,
                          ug2_filter=ug2_filter)
        if d is None:
            continue
        out.append({"ra": raa, "hd2": d["hd2"], "hd3": d["hd3"],
                    "thd": d["thd"], "pout_mw": d["pout_mw"]})
    return out


def sweep_bias(
    points: List[Dict],
    load_line: LoadLine,
    half_swing: Optional[float] = None,
    ug2_filter: Optional[float] = None,
    steps: int = 30,
    ug1_bias: Optional[float] = None,
    model: Optional["TubeModelProtocol"] = None,
    model_ug2: float = 0.0,
) -> List[Dict]:
    """THD vs Ug1 bias — sweep operating point.

    Args:
        ug1_bias: if provided, used for transformer AC load line Q-point.
            When None, midpoint of Ug1 range is used for AC line.
        model: tube model for model-based intersections (more accurate).
        model_ug2: Ug2 value for model evaluation.
    """
    _ac_bias = ug1_bias
    if _ac_bias is None and isinstance(load_line, (TransformerLoadLine, PushPullLoadLine)):
        ug1_vals = sorted({round(p["ug1"], UA_ROUND) for p in points})
        if len(ug1_vals) >= 2:
            _ac_bias = (ug1_vals[0] + ug1_vals[-1]) / 2.0

    if model is not None:
        ug1_values = sorted({round(p["ug1"], 1) for p in points})
        ua_max = max((p["ua"] for p in points), default=MODEL_UA_MAX_DEFAULT_V)
        isects = find_intersections_model(
            model, load_line, ug1_values, ug2=model_ug2,
            ua_range=(MODEL_UA_MIN_V, ua_max),
            ug1_bias=_ac_bias,
        )
    else:
        isects = find_intersections(points, load_line, ug2_filter=ug2_filter,
                                    ug1_bias=_ac_bias)
    if len(isects) < 3:
        return []

    pts = sorted(isects, key=lambda p: p["ug1"])
    ug1_min = pts[0]["ug1"]
    ug1_max = pts[-1]["ug1"]

    margin = (ug1_max - ug1_min) * SWEEP_BIAS_MARGIN
    ug1_start = ug1_min + margin
    ug1_end = ug1_max - margin

    results: List[Dict] = []
    for i in range(steps):
        ug1 = ug1_start + (ug1_end - ug1_start) * i / max(steps - 1, 1)
        dist = compute_distortion(isects, ug1_bias=ug1, half_swing=half_swing)
        if dist is None:
            continue
        results.append({
            "ug1": ug1,
            "hd2": dist["hd2"],
            "hd3": dist["hd3"],
            "hd4": dist.get("hd4", 0.0),
            "hd5": dist.get("hd5", 0.0),
            "thd": dist["thd"],
            "pout_mw": dist["pout_mw"],
            "ia_0": dist["ia_0"],
            "ua_0": dist["ua_0"],
        })
    return results


def optimize_bias(
    points: List[Dict],
    load_line: LoadLine,
    half_swing: Optional[float] = None,
    ug2_filter: Optional[float] = None,
    target: str = "min_thd",
    model: Optional["TubeModelProtocol"] = None,
    model_ug2: float = 0.0,
) -> Optional[Dict]:
    """Find optimal Ug1 bias for given target.

    Targets:
        'min_thd'  — minimum THD
        'max_pout' — maximum Pout
        'balanced' — minimum THD where Pout >= 50% of max
    """
    bias_data = sweep_bias(
        points, load_line, half_swing=half_swing,
        ug2_filter=ug2_filter, steps=50,
        ug1_bias=None,
        model=model, model_ug2=model_ug2,
    )
    if not bias_data:
        return None

    if target == "max_pout":
        best = max(bias_data, key=lambda d: d["pout_mw"])
    elif target == "balanced":
        max_pout = max(d["pout_mw"] for d in bias_data)
        threshold = max_pout * 0.5
        candidates = [d for d in bias_data if d["pout_mw"] >= threshold]
        if not candidates:
            candidates = bias_data
        best = min(candidates, key=lambda d: d["thd"])
    else:
        best = min(bias_data, key=lambda d: d["thd"])

    isects = find_intersections(points, load_line, ug2_filter=ug2_filter,
                                ug1_bias=best["ug1"])
    headroom = compute_headroom(isects, best["ug1"])

    # Cathode bias resistor: Rk = |Ug1| / Ik.
    # For pentodes Ik = Ia + Ig2 (total cathode current).
    ik = best["ia_0"]
    ig2_q = estimate_ig2_at_q(points, best["ug1"], best["ua_0"], ug2_filter)
    if ig2_q > 0:
        ik += ig2_q
    rk = 0.0
    if ik > IK_NEAR_ZERO_MA:
        rk = abs(best["ug1"]) / ik * 1000.0

    return {
        "ug1_0": best["ug1"],
        "ua_0": best["ua_0"],
        "ia_0": best["ia_0"],
        "hd2": best["hd2"],
        "hd3": best["hd3"],
        "thd": best["thd"],
        "pout_mw": best["pout_mw"],
        "headroom": headroom["max_swing"] if headroom else 0.0,
        "rk_auto_bias": rk,
    }


def sweep_pp_amplitude(
    points_a: List[Dict],
    load_line: PushPullLoadLine,
    ug1_bias: float,
    points_b: Optional[List[Dict]] = None,
    ug2_filter: Optional[float] = None,
    steps: int = 40,
) -> List[Dict]:
    """THD vs Pout sweep for push-pull stage.

    Returns list of {half_swing, hd2, hd3, thd, pout_mw, balance_error}.
    """
    comp = composite_characteristic(
        points_a, points_b, ug1_bias=ug1_bias, ug2_filter=ug2_filter,
        ua_ref=load_line.ub,
    )
    if len(comp) < 5:
        return []

    comp_sorted = sorted(comp, key=lambda p: p["ug1"])
    ug1_min = comp_sorted[0]["ug1"]
    ug1_max = comp_sorted[-1]["ug1"]
    max_swing = min(abs(ug1_bias - ug1_min), abs(ug1_max - ug1_bias))

    if max_swing < MIN_SWEEP_SWING_V:
        return []

    results: List[Dict] = []
    for i in range(steps):
        hs = MIN_SWEEP_SWING_V + (max_swing - MIN_SWEEP_SWING_V) * i / max(steps - 1, 1)
        dist = pp_distortion(
            points_a, load_line, ug1_bias,
            points_b=points_b, half_swing=hs, ug2_filter=ug2_filter,
        )
        if dist is None:
            continue
        results.append({
            "half_swing": hs,
            "hd2": dist["hd2"],
            "hd3": dist["hd3"],
            "hd4": dist.get("hd4", 0.0),
            "hd5": dist.get("hd5", 0.0),
            "thd": dist["thd"],
            "pout_mw": dist["pout_mw"],
            "balance_error": dist["balance_error"],
        })
    return results
