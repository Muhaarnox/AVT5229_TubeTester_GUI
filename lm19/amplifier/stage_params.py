"""Stage parameters: gain, Zout, gm/ra estimation + UL wrapper.

Public functions:
  - ``model_gm_ra``         — finite-difference gm/ra from a tube model
  - ``compute_stage_params`` — gain/Zout/DF for resistive/transformer/PP
  - ``compute_cf_stage_params`` — cathode-follower variant
  - ``compute_nfb_effect``  — closed-loop params for given NFB (dB)
  - ``ul_screen_voltage``   — VG2K = Ug2_nom·(1−tap) + Ua·tap
  - ``UltralinearModelWrapper`` — wraps ``TubeModelProtocol`` with UL tap

Internal helpers ``_compare_srk``, ``_extract_ra``, ``_numerical_gm_ra``
are re-exported through ``lm19.amplifier`` as part of the package API
so callers can ``from lm19.amplifier import …`` without reaching into
this submodule.

Priority for gm/ra inside ``compute_stage_params``:
  1. model (``model_gm_ra``) — most accurate, works with UL wrapper
  2. numerical (``_numerical_gm_ra``) — from raw measurement points
  3. SRK — fallback from manual measurements (cross-checked via
     ``_compare_srk``)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from lm19.amplifier.constants import (
    DB_MULTIPLIER,
    MODEL_GM_DELTA_V,
    MODEL_RA_DELTA_V,
    Q_WINDOW_UA_V,
    RA_WINDOW_UA_V,
    SLOPE_NEAR_ZERO,
    SRK_DIVERGENCE_THRESHOLD_PCT,
    OUTLIER_MAD_FACTOR,
)
from lm19.amplifier.distortion import interp_intersection
from lm19.tube_model_base import model_ia_array
from lm19.amplifier.loadlines import (
    CathodeFollowerLoadLine,
    LoadLine,
    PushPullLoadLine,
    ResistiveLoadLine,
    TransformerLoadLine,
)
from lm19.constants import (
    EPS,
    EPS_COARSE,
    MAX_RA_KOHM,
    UA_ROUND,
)

if TYPE_CHECKING:
    from lm19.tube_model_base import TubeModelProtocol

log = logging.getLogger(__name__)


# ─── gm / ra estimators ──────────────────────────────────────────────

def model_gm_ra(
    model: "TubeModelProtocol",
    ua_q: float,
    ug1_q: float,
    ug2: float = 0.0,
) -> Optional[Dict]:
    """Compute gm, ra, mu from a tube model at a specific operating point.

    Uses central finite differences on model.ia():
        gm = ΔIa/ΔUg1  at constant Ua, Ug2   (mA/V)
        ra = ΔUa/ΔIa   at constant Ug1, Ug2  (kΩ)
        mu = gm × ra

    For UltralinearModelWrapper, ug2 argument is ignored by the wrapper
    and the dynamic UL screen voltage is used automatically.
    """
    dg = MODEL_GM_DELTA_V
    da = MODEL_RA_DELTA_V

    ia_gp = model.ia(ua_q, ug1_q + dg, ug2)
    ia_gm = model.ia(ua_q, ug1_q - dg, ug2)
    d_ia_g = ia_gp - ia_gm
    if abs(d_ia_g) < EPS:
        return None
    gm = d_ia_g / (2.0 * dg)
    if gm <= 0:
        return None

    ia_ap = model.ia(ua_q + da, ug1_q, ug2)
    ia_am = model.ia(ua_q - da, ug1_q, ug2)
    d_ia_a = ia_ap - ia_am
    if abs(d_ia_a) < EPS:
        ra = MAX_RA_KOHM
    else:
        ra = (2.0 * da) / d_ia_a
        if ra <= 0:
            return None
        ra = min(ra, MAX_RA_KOHM)

    mu = gm * ra
    return {"gm": gm, "ra": ra, "mu": mu, "method": "model"}


def _compare_srk(
    srk: Optional[Dict],
    gm_numerical: float,
    ra_numerical: float,
) -> Tuple[Optional[str], Optional[float]]:
    """Compare SRK data with numerical gm/ra estimates.

    Returns:
        (srk_check, srk_divergence_pct):
        - (None, None) if SRK absent
        - ("ok", max_pct) if within threshold
        - ("divergence", max_pct) if exceeds threshold
    """
    if not srk or not srk.get("s") or not srk.get("r"):
        return None, None

    gm_srk = srk["s"]
    ra_srk = srk["r"]

    pct_gm = abs(gm_srk - gm_numerical) / max(abs(gm_srk), abs(gm_numerical), EPS) * 100.0
    pct_ra = abs(ra_srk - ra_numerical) / max(abs(ra_srk), abs(ra_numerical), EPS) * 100.0
    max_pct = round(max(pct_gm, pct_ra), 1)

    if max_pct > SRK_DIVERGENCE_THRESHOLD_PCT:
        log.warning(
            "SRK divergence: gm %.1f%%, ra %.1f%% (threshold %.0f%%)",
            pct_gm, pct_ra, SRK_DIVERGENCE_THRESHOLD_PCT,
        )
        return "divergence", max_pct

    return "ok", max_pct


def _extract_ra(load_line: LoadLine) -> Optional[float]:
    """Extract effective Ra (kOhm) from a LoadLine."""
    if isinstance(load_line, ResistiveLoadLine):
        return load_line.ra
    if isinstance(load_line, TransformerLoadLine):
        return load_line.ra_ac
    if isinstance(load_line, CathodeFollowerLoadLine):
        return load_line.rk + load_line.rl
    if isinstance(load_line, PushPullLoadLine):
        return load_line.ra_per_tube
    if hasattr(load_line, "ra"):
        return load_line.ra
    return None


def _numerical_gm_ra(
    points: Optional[List[Dict]],
    intersections: List[Dict],
    ug1_bias: float,
    ua_tolerance: float = Q_WINDOW_UA_V,
) -> Optional[Dict]:
    """Estimate gm and ra from raw measurement data near Q-point.

    Uses raw scan points (not intersections) to compute derivatives
    at constant Ua (for gm) and constant Ug1 (for ra).
    """
    if not points or len(points) < 2:
        return None

    q = interp_intersection(intersections, ug1_bias)
    if q is None:
        return None
    ua_q = q["ua"]

    def _robust_slope(xs: List[float], ys: List[float]) -> Optional[float]:
        """Return linear slope with one-step robust outlier rejection."""
        if len(xs) < 2:
            return None
        x = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float)
        if np.ptp(x) < 1e-9:
            return None
        k, b = np.polyfit(x, y, 1)
        resid = y - (k * x + b)
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        if mad > 1e-9 and len(x) >= 4:
            mask = np.abs(resid - med) <= OUTLIER_MAD_FACTOR * mad
            if int(mask.sum()) >= 2:
                x2 = x[mask]
                y2 = y[mask]
                if np.ptp(x2) >= 1e-9:
                    k, _ = np.polyfit(x2, y2, 1)
        return float(k)

    near_ua = [p for p in points if abs(p["ua"] - ua_q) <= ua_tolerance]
    if len(near_ua) < 3:
        return None

    ug1_to_ia: Dict[float, List[float]] = {}
    for p in near_ua:
        key = round(p["ug1"], UA_ROUND)
        ug1_to_ia.setdefault(key, []).append(float(p["ia"]))

    ug1_levels = sorted(ug1_to_ia.keys(), key=lambda u: abs(u - ug1_bias))
    if len(ug1_levels) < 2:
        return None
    ug1_sel = sorted(ug1_levels[:7])
    x_gm = ug1_sel
    y_gm = [float(np.mean(ug1_to_ia[u])) for u in ug1_sel]
    gm_slope = _robust_slope(x_gm, y_gm)
    if gm_slope is None:
        return None
    gm = abs(gm_slope)
    if gm < EPS_COARSE:
        return None

    all_ug1 = sorted({round(p["ug1"], UA_ROUND) for p in points}, key=lambda u: abs(u - ug1_bias))
    if not all_ug1:
        return None
    # ML-140: fit the Ia(Ua) slope PER Ug1 curve and take the median —
    # ra is defined at CONSTANT Ug1. The old pooled fit ran one line
    # through the points of the 3 nearest curves; their inter-curve
    # offsets (gm·ΔUg1) contaminated the slope. And ra is LOCAL at Q:
    # fit inside a Ua window around ua_q (widening only for sparse
    # scans) — a whole-curve fit mixes the knee and the plateau.
    by_curve: Dict[float, List[Dict]] = {}
    for p in points:
        by_curve.setdefault(round(p["ug1"], UA_ROUND), []).append(p)
    slopes: List[float] = []
    for u in all_ug1[:3]:
        pts_c = by_curve.get(u, [])
        s_c: Optional[float] = None
        for win in (RA_WINDOW_UA_V, 2 * RA_WINDOW_UA_V, float("inf")):
            local = [p for p in pts_c
                     if abs(float(p["ua"]) - ua_q) <= win]
            if len(local) < 3:
                continue
            s_c = _robust_slope([float(p["ua"]) for p in local],
                                [float(p["ia"]) for p in local])
            if s_c is not None:
                break
        if s_c is not None:
            slopes.append(s_c)
    if not slopes:
        return None
    slope_ia_ua = float(np.median(slopes))
    if abs(slope_ia_ua) < SLOPE_NEAR_ZERO:
        ra = MAX_RA_KOHM
    else:
        ra = min(MAX_RA_KOHM, abs(1.0 / slope_ia_ua))

    mu = gm * ra
    return {"gm": gm, "ra": ra, "mu": mu, "method": "numerical"}


# ─── Stage parameter calculators ──────────────────────────────────────

def _resolve_tube_params(
    intersections: List[Dict],
    ug1_bias: Optional[float],
    srk: Optional[Dict],
    points: Optional[List[Dict]],
    model: Optional["TubeModelProtocol"],
    model_ug2: float,
) -> Optional[Dict]:
    """Resolve gm/ra/mu via the model → numerical → SRK priority chain.

    Shared by ``compute_stage_params`` and ``compute_cf_stage_params``
    (ML-133: the two inline blocks were byte-for-byte duplicates and the
    CF twin could drift silently). Returns ``{gm, ra, mu, method,
    srk_check, srk_divergence_pct}`` or ``None`` when no source is
    available.
    """
    if ug1_bias is None:
        pts = sorted(intersections, key=lambda p: p["ug1"])
        ug1_bias = (pts[0]["ug1"] + pts[-1]["ug1"]) / 2.0

    q = interp_intersection(intersections, ug1_bias)
    ua_q = q["ua"] if q else None

    tube_params = None
    if model is not None and ua_q is not None:
        tube_params = model_gm_ra(model, ua_q, ug1_bias, model_ug2)

    if tube_params is not None:
        method = "model"
    else:
        tube_params = _numerical_gm_ra(points, intersections, ug1_bias)
        if tube_params is not None:
            method = "numerical"
        elif srk and srk.get("s") and srk.get("r"):
            gm = srk["s"]
            ra = srk["r"]
            # SRK is itself the source here — no cross-check against it.
            return {"gm": gm, "ra": ra, "mu": srk.get("k", gm * ra),
                    "method": "srk", "srk_check": None,
                    "srk_divergence_pct": None}
        else:
            return None

    gm = tube_params["gm"]
    ra = tube_params["ra"]
    mu = tube_params["mu"]
    srk_check, srk_divergence_pct = _compare_srk(srk, gm, ra)
    return {"gm": gm, "ra": ra, "mu": mu, "method": method,
            "srk_check": srk_check,
            "srk_divergence_pct": srk_divergence_pct}


def compute_stage_params(
    intersections: List[Dict],
    load_line: LoadLine,
    ug1_bias: Optional[float] = None,
    srk: Optional[Dict] = None,
    points: Optional[List[Dict]] = None,
    model: Optional["TubeModelProtocol"] = None,
    model_ug2: float = 0.0,
) -> Optional[Dict]:
    """Compute voltage gain, output impedance and other stage parameters.

    Gain/Zout follow "Valley & Wallman — Vacuum Tube Amplifiers",
    see SOURCES_INDEX.md.

    Priority for gm/ra estimation:
      1. model (model_gm_ra) — most accurate, works with UL wrapper
      2. numerical (_numerical_gm_ra) — from raw measurement points
      3. SRK — fallback from manual measurements

    If SRK data is available, cross-checks against the primary result.
    """
    if isinstance(load_line, CathodeFollowerLoadLine):
        return compute_cf_stage_params(
            intersections, load_line, ug1_bias, srk, points,
            model=model, model_ug2=model_ug2,
        )

    if len(intersections) < 2:
        return None

    ra_load = _extract_ra(load_line)
    if ra_load is None or ra_load <= 0:
        return None

    resolved = _resolve_tube_params(
        intersections, ug1_bias, srk, points, model, model_ug2)
    if resolved is None:
        return None
    gm = resolved["gm"]
    ra = resolved["ra"]
    mu = resolved["mu"]

    gain = mu * ra_load / (ra + ra_load)
    gain_db = DB_MULTIPLIER * math.log10(max(gain, EPS_COARSE))

    zout = (ra * ra_load) / (ra + ra_load) if (ra + ra_load) > 0 else 0.0

    df = ra_load / zout if zout > 0 else None

    return {
        "gain": gain,
        "gain_db": gain_db,
        "zout": zout,
        "df": df,
        "gm": gm,
        "ra": ra,
        "mu": mu,
        "method": resolved["method"],
        "srk_check": resolved["srk_check"],
        "srk_divergence_pct": resolved["srk_divergence_pct"],
    }


def compute_cf_stage_params(
    intersections: List[Dict],
    load_line: CathodeFollowerLoadLine,
    ug1_bias: Optional[float] = None,
    srk: Optional[Dict] = None,
    points: Optional[List[Dict]] = None,
    model: Optional["TubeModelProtocol"] = None,
    model_ug2: float = 0.0,
) -> Optional[Dict]:
    """Compute cathode follower stage parameters.

    CF formulas follow "TubeCad — Cathode Follower Output Stage",
    see SOURCES_INDEX.md.

    Gain = mu * Rk / (ra + (mu + 1) * Rk)  ≈ 1 for Rk >> ra/(mu+1)
    Zout = ra / (mu + 1) ≈ 1/gm

    Priority: model → numerical → SRK (same as compute_stage_params).
    """
    if len(intersections) < 2:
        return None

    resolved = _resolve_tube_params(
        intersections, ug1_bias, srk, points, model, model_ug2)
    if resolved is None:
        return None
    gm = resolved["gm"]
    ra = resolved["ra"]
    mu = resolved["mu"]

    rk = load_line.rk

    gain = mu * rk / (ra + (mu + 1) * rk) if (ra + (mu + 1) * rk) > 0 else 0.0
    gain_db = DB_MULTIPLIER * math.log10(max(gain, EPS_COARSE))
    zout = ra / (mu + 1) if mu > 0 else ra

    rl = load_line.rl if hasattr(load_line, "rl") else 0.0
    df = rl / zout if zout > 0 and rl > 0 else None

    return {
        "gain": gain,
        "gain_db": gain_db,
        "zout": zout,
        "df": df,
        "gm": gm,
        "ra": ra,
        "mu": mu,
        "method": resolved["method"],
        "srk_check": resolved["srk_check"],
        "srk_divergence_pct": resolved["srk_divergence_pct"],
    }


def compute_nfb_effect(
    gain_open: float,
    zout_open: float,
    thd_open: float,
    nfb_db: float,
) -> Optional[Dict]:
    """Apply global negative feedback to open-loop amplifier parameters.

    Uses the classical linear feedback model:
        D = 10^(nfb_db/20)          — desensitivity factor
        β = (D − 1) / A             — feedback fraction
        X_closed = X_open / D       — for gain, Zout, THD

    Bandwidth is extended by factor D (returned as bw_factor).

    Accurate for moderate feedback (6–20 dB) typical of tube amplifiers.
    At very high feedback (>20 dB) THD reduction is optimistic because
    higher-order products become significant.

    Sources: RDH 4th ed. Ch.12, VTADiy §4.4, SB-LAB (2025).
    """
    if nfb_db < 0 or gain_open <= 0:
        return None

    D = 10.0 ** (nfb_db / 20.0)
    beta = (D - 1.0) / gain_open

    gain_closed = gain_open / D
    gain_closed_db = 20.0 * math.log10(max(gain_closed, EPS_COARSE))
    zout_closed = zout_open / D
    thd_closed = thd_open / D
    bw_factor = D

    return {
        "gain_closed": gain_closed,
        "gain_closed_db": gain_closed_db,
        "zout_closed": zout_closed,
        "thd_closed": thd_closed,
        "bw_factor": bw_factor,
        "desensitivity": D,
        "beta": beta,
        "nfb_db": nfb_db,
        "gain_open": gain_open,
    }


# ─── Ultralinear screen voltage + wrapper ────────────────────────────

def ul_screen_voltage(ua: float, ug2_nom: float, tap: float) -> float:
    """Compute effective screen voltage in ultralinear mode.

    VG2K = Ug2_nom * (1 - tap) + Ua * tap

    At tap=0 → pure pentode (VG2K = Ug2_nom, screen fixed).
    At tap=1 → triode-connected (VG2K = Ua, screen tied to plate).

    Sources: Hafler & Keroes 1951, Dempwolf model §5.1.
    """
    return ug2_nom * (1.0 - tap) + ua * tap


class UltralinearModelWrapper:
    """Wraps a TubeModelProtocol to apply ultralinear screen tap.

    Intercepts ia() and ig2() calls, replacing the fixed ug2 with
    a dynamic value computed from plate voltage and tap fraction:
        ug2_eff = ug2_nom * (1 - tap) + ua * tap

    All other attributes are forwarded to the underlying model.
    The wrapper satisfies TubeModelProtocol (duck-typing).
    """

    def __init__(
        self,
        model: "TubeModelProtocol",
        ug2_nom: float,
        tap: float,
    ) -> None:
        self._model = model
        self._ug2_nom = ug2_nom
        self._tap = tap

    @property
    def model_type(self) -> str:
        return self._model.model_type

    @property
    def name(self) -> str:
        return self._model.name

    @property
    def topology(self) -> str:
        return self._model.topology

    @property
    def pa_max(self) -> float:
        return self._model.pa_max

    @property
    def uh(self) -> float:
        return self._model.uh

    @property
    def ih(self) -> float:
        return self._model.ih

    def ia(self, ua: float, ug1: float, ug2: float = 0.0) -> float:
        """Anode current with UL screen voltage (ug2 arg ignored)."""
        ug2_eff = ul_screen_voltage(ua, self._ug2_nom, self._tap)
        return self._model.ia(ua, ug1, ug2_eff)

    def ia_array(self, ua, ug1, ug2=0.0) -> np.ndarray:
        """Vectorized Ia with UL screen voltage (ug2 arg ignored).

        ``ug2_eff`` follows ``ua`` element-wise — exactly the per-sample
        screen modulation the scalar path applies.
        """
        ua_arr = np.asarray(ua, dtype=float)
        ug2_eff = ul_screen_voltage(ua_arr, self._ug2_nom, self._tap)
        return model_ia_array(self._model, ua_arr, ug1, ug2_eff)

    def ig2(self, ua: float, ug1: float, ug2: float) -> float:
        """Screen current with UL screen voltage (ug2 arg ignored)."""
        ug2_eff = ul_screen_voltage(ua, self._ug2_nom, self._tap)
        return self._model.ig2(ua, ug1, ug2_eff)

    def generate_scan(self, grid) -> List[Dict]:
        """Forward to underlying model (no UL transform for scan)."""
        return self._model.generate_scan(grid)

    def params_dict(self) -> Dict:
        """Model params + UL-specific fields."""
        d = self._model.params_dict()
        d["ul_tap"] = self._tap
        d["ul_ug2_nom"] = self._ug2_nom
        return d

    @property
    def ug2_nom(self) -> float:
        return self._ug2_nom

    @property
    def tap(self) -> float:
        return self._tap

    def __getattr__(self, name: str) -> object:
        """Forward unknown attributes to the underlying model."""
        return getattr(self._model, name)
