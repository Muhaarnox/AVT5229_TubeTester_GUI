"""Multi-parameter amplifier optimizer.

Pure calculation module — no Qt dependencies.

Finds optimal (Ug1, Ra) or (Ub, Ug2, Ug1, Ra) combinations
for given performance targets and constraints.

Two-phase approach:
  1. Grid sweep — coarse search, produces Pareto front (THD vs Pout)
  2. scipy refine — fine-tune the best grid point

For measurements: Ub/Ug2 fixed by data → 2D sweep (Ug1 × Ra).
For models: all parameters free → up to 4D sweep.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from lm19.tube_model_base import TubeModelProtocol

from lm19.amplifier import (
    CathodeFollowerLoadLine,
    LoadLine,
    PushPullLoadLine,
    ResistiveLoadLine,
    TransformerLoadLine,
    UltralinearModelWrapper,
    compute_distortion,
    compute_distortion_chebyshev,
    compute_distortion_chebyshev_pp,
    compute_distortion_chebyshev_pp_model,
    compute_distortion_dft,
    compute_distortion_dft_pp,
    compute_headroom,
    find_intersections,
    ug2_filter_matches_any,
    find_intersections_model,
    pp_distortion,
)
from lm19.constants import (
    DEFAULT_UB_V,
    MODEL_UA_MAX_DEFAULT_V,
    MODEL_UA_MIN_V,
    MW_PER_W,
)
from lm19.amplifier.constants import (
    CIRCUIT_CF,
    CIRCUIT_PP,
    CIRCUIT_SE,
    CIRCUIT_SE_XFMR,
    HD_METHOD_5POINT,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_DFT,
)

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────
MIN_ISECTS_FOR_ANALYSIS = 3
MIN_POUT_MW = 1.0              # skip points with negligible power
CLASS_A_RATIO = 0.05           # i_min / i_0 above this → class A
CLASS_B_RATIO = 0.005          # i_min / i_0 below this → class B
PARETO_DOMINANCE_EPS = 1e-9    # tolerance for Pareto dominance check
DEFAULT_UG1_STEPS = 20
DEFAULT_RA_STEPS = 20
DEFAULT_UB_STEPS = 8
DEFAULT_UG2_STEPS = 4
REFINE_MAX_ITER = 200          # scipy max iterations
TOP_N_FOR_SWING = 20           # re-evaluate top-N grid points with swing sweep
DEFAULT_SWING_STEPS = 5        # swing levels between min and max
MIN_SWING_FRACTION = 0.3       # minimum swing as fraction of max_swing
PARETO_REFINE_MAX = 8          # max Pareto points to refine
PARETO_REFINE_WORKERS = 4      # ThreadPoolExecutor max_workers
# DEFAULT_UB_V imported from lm19.constants (single source of truth)
DEFAULT_UG1_GRID_SIZE = 30     # model Ug1 grid when ug1_values not provided
UA_MAX_FACTOR = 1.2            # Ua upper bound = Ub * factor
# se_xfmr: the AC load line passes through the DC Q-point at Ua ≈ Ub, so
# the anode swings ABOVE the supply — up to ~2×Ub for full class-A swing.
# The 1.2 factor truncated the intersection family at the negative-bias
# end (low currents intersect the AC line at high Ua).
XFMR_UA_MAX_FACTOR = 2.0
PENALTY_SCORE = 1e6            # infeasible-point objective value
REFINE_XATOL = 0.01           # Nelder-Mead position tolerance
REFINE_FATOL = 0.001          # Nelder-Mead function tolerance

# PP class-A power threshold conversion: Iq² × Ra_aa / 8
# class-A definition from "Aiken Amps — The Last Word on Class A" and
# "Rod Elliott (ESP) — Class-A Amplifiers Explained", see SOURCES_INDEX.md
# Iq in mA, Ra_aa in kΩ → P in W: ÷ 8000
# Derivation: P_A = Iq²(A) × Ra_aa(Ω) / 8 = (Iq_mA/1000)² × (Ra_kΩ × 1000) / 8
#           = Iq_mA² × Ra_kΩ / 8000
P_CLASSA_DIVISOR = 8000.0


def _resolve_ul_taps(constraints: "OptimizerConstraints") -> List[float]:
    """Resolve the list of UL tap fractions to sweep based on UI mode.

    Returns a sorted, deduplicated list of fractions (0.0..1.0).
    For "off" mode returns just [ul_tap_manual] so the optimizer still
    sweeps Ub/Ug2/Ug1/Ra at the user-fixed tap (default = pentode 0).
    """
    if constraints.ul_tap_mode == "off":
        return [max(0.0, min(1.0, constraints.ul_tap_manual))]

    taps: set = set()
    if constraints.ul_tap_mode in ("presets", "presets_custom"):
        for v, en in zip(
            constraints.ul_tap_presets, constraints.ul_tap_presets_enabled,
        ):
            if en:
                taps.add(round(float(v), 3))
    if constraints.ul_tap_mode in ("custom", "presets_custom"):
        lo, hi = constraints.ul_tap_range
        lo = max(0.0, min(1.0, lo))
        hi = max(0.0, min(1.0, hi))
        steps = max(1, constraints.ul_tap_steps)
        if steps == 1 or hi <= lo:
            taps.add(round(lo, 3))
        else:
            for i in range(steps):
                t = lo + (hi - lo) * i / (steps - 1)
                taps.add(round(t, 3))

    return sorted(taps) if taps else [0.0]



# -- Contract code vocabularies ---------------------------------------
# OptimizerResult.warnings / .warning — UI maps to amp.opt_warn_<code>;
# OptimizerResult.error — to amp.opt_err_<code>. Registry <-> locales
# are tied by a bijection pin (test_conventions_guards).
OPT_WARN_DFT_NO_MODEL_FALLBACK = "dft_no_model_fallback"
OPT_WARN_SE_XFMR_NO_MODEL = "se_xfmr_no_model"
OPT_WARN_UG2_FILTER_NO_MATCH = "ug2_filter_no_match"
OPT_WARN_DFT_MISMATCHED_PAIR = "dft_mismatched_pair"
OPT_WARN_UL_SWEEP_SKIPPED = "ul_sweep_skipped"
OPT_WARN_PP_DFT_UG2_FROM_DATA = "pp_dft_ug2_from_data"
OPT_WARN_REFINE_UNAVAILABLE = "refine_unavailable"
OPT_WARN_REFINE_FAILED = "refine_failed"
OPT_WARN_NM_NOT_CONVERGED = "nm_not_converged"
OPT_WARNING_CODES = frozenset({
    OPT_WARN_DFT_NO_MODEL_FALLBACK, OPT_WARN_SE_XFMR_NO_MODEL,
    OPT_WARN_UG2_FILTER_NO_MATCH, OPT_WARN_DFT_MISMATCHED_PAIR,
    OPT_WARN_UL_SWEEP_SKIPPED, OPT_WARN_PP_DFT_UG2_FROM_DATA,
    OPT_WARN_REFINE_UNAVAILABLE, OPT_WARN_REFINE_FAILED,
    OPT_WARN_NM_NOT_CONVERGED,
})
OPT_ERR_NO_VALID_POINTS = "no_valid_points"
OPT_ERR_NO_POINTS_WITHIN_CONSTRAINTS = "no_points_within_constraints"
OPT_ERR_NO_POINTS_WITHIN_THD_CAP = "no_points_within_thd_cap"
OPT_ERROR_CODES = frozenset({
    OPT_ERR_NO_VALID_POINTS, OPT_ERR_NO_POINTS_WITHIN_CONSTRAINTS,
    OPT_ERR_NO_POINTS_WITHIN_THD_CAP,
})

def _resolve_methods(
    hd_method: str, has_model: bool,
) -> Tuple[str, str, Optional[str]]:
    """Return (grid_method, refine_method, warning) per UI hd_method choice.

    Mapping:
      "5point"    → grid 5point, refine 5point
      "chebyshev" → grid chebyshev, refine chebyshev
      "dft"       → grid dft, refine dft (requires model; warns + falls
                    back to chebyshev if missing)
      "auto"      → grid chebyshev, refine dft if model else chebyshev
    """
    if hd_method == HD_METHOD_5POINT:
        return HD_METHOD_5POINT, HD_METHOD_5POINT, None
    if hd_method == HD_METHOD_CHEBYSHEV:
        return HD_METHOD_CHEBYSHEV, HD_METHOD_CHEBYSHEV, None
    if hd_method == HD_METHOD_DFT:
        if has_model:
            return HD_METHOD_DFT, HD_METHOD_DFT, None
        return HD_METHOD_CHEBYSHEV, HD_METHOD_CHEBYSHEV, OPT_WARN_DFT_NO_MODEL_FALLBACK
    # "auto" (default fallback)
    if has_model:
        return HD_METHOD_CHEBYSHEV, HD_METHOD_DFT, None
    return HD_METHOD_CHEBYSHEV, HD_METHOD_CHEBYSHEV, None


# DFT path requires an explicit signal swing. When the caller did not
# supply one, fall back to the widest symmetric swing implied by the data
# (headroom for SE, composite range for PP). MIN_SWING_V threshold avoids
# DFT'ing on degenerate sub-volt swings.
_MIN_DFT_SWING_V = 0.1


def _resolve_dft_swing(
    half_swing: Optional[float],
    fallback: Callable[[], Optional[float]],
) -> Optional[float]:
    """Resolve half_swing for DFT path; fall back when missing or too small."""
    if half_swing is None or half_swing < _MIN_DFT_SWING_V:
        half_swing = fallback()
    if half_swing is None or half_swing < _MIN_DFT_SWING_V:
        return None
    return half_swing


def _compute_dist(
    method: str,
    isects: List[Dict],
    model: Optional["TubeModelProtocol"],
    load_line: LoadLine,
    ug1_bias: float,
    half_swing: Optional[float],
    ug2: float,
    ub: float,
) -> Optional[Dict]:
    """Dispatch distortion calculation based on requested HD method.

    The "dft" branch only fires when model is provided AND swing is set
    (or auto-resolvable from headroom). DFT path returns None when swing
    cannot be resolved. Higher-level _resolve_methods() handles the
    "dft requested but no model" case by remapping method to "chebyshev"
    upstream, so this dispatcher only sees the resolved method.
    """
    if method == HD_METHOD_DFT and model is not None:
        def _se_fallback() -> Optional[float]:
            hr = compute_headroom(isects, ug1_bias)
            return hr["max_swing"] if hr else None

        hs = _resolve_dft_swing(half_swing, _se_fallback)
        if hs is None:
            return None
        return compute_distortion_dft(
            model, load_line, ug1_bias=ug1_bias,
            half_swing=hs, ug2=ug2, ub=ub,
        )
    if method == HD_METHOD_CHEBYSHEV:
        return compute_distortion_chebyshev(
            isects, ug1_bias=ug1_bias, half_swing=half_swing, ub=ub,
        )
    # "5point" (default)
    return compute_distortion(
        isects, ug1_bias=ug1_bias, half_swing=half_swing, ub=ub,
    )


def _compute_dist_pp(
    method: str,
    points_a: List[Dict],
    points_b: Optional[List[Dict]],
    load_line: "PushPullLoadLine",
    ug1_bias: float,
    half_swing: Optional[float],
    ug2_filter: Optional[float],
    model: Optional["TubeModelProtocol"],
    transfer: Optional[Tuple] = None,
) -> Optional[Dict]:
    """Dispatch PP distortion calculation based on requested HD method.

    DFT path requires a model and explicit half_swing. Chebyshev and
    5-point operate on the composite characteristic from measurements.
    ``transfer`` forwards a pre-built ``build_pp_transfer`` pair so the
    grid does not rebuild the transfer curves on every evaluation.
    """
    if model is not None and points_b is None and (
            method == HD_METHOD_DFT
            or (method == HD_METHOD_CHEBYSHEV
                and isinstance(model, UltralinearModelWrapper))):
        # points_b is None: the model paths solve a MATCHED pair from tube
        # A's model — a mismatched pair evaluated there would silently get
        # matched physics. Mismatched runs stay on the data composite
        # (optimize_pp warns once: dft_mismatched_pair).
        # Model-evaluated paths: DFT always; Chebyshev ONLY when the model
        # is UL-wrapped (tap > 0) — the data composite cannot see a tap,
        # and the unwrapped tap=0 case must stay on the byte-identical
        # data path (vectorized-equivalence pins).
        def _pp_fallback() -> Optional[float]:
            # Narrow except: data-shape errors (missing 'ia' key, malformed
            # numeric values, out-of-range index) are expected on user-
            # supplied measurements — log and treat as "no composite".
            # Programming errors (AttributeError, NameError, etc.) propagate
            # so a refactor regression isn't masked into "silently dropped
            # optimization points".
            from lm19.amplifier import composite_characteristic as _cc
            try:
                comp = _cc(points_a, points_b, ug1_bias=ug1_bias,
                           ug2_filter=ug2_filter, transfer=transfer,
                           ua_ref=load_line.ub)
            except (KeyError, IndexError, TypeError, ValueError) as e:
                log.warning(
                    "PP composite failed (treating as empty): %s: %s "
                    "(points_a=%d, points_b=%s, ug1_bias=%.2f, ug2_filter=%s)",
                    type(e).__name__, e,
                    len(points_a) if points_a else 0,
                    len(points_b) if points_b else None,
                    ug1_bias, ug2_filter,
                )
                comp = []
            if not comp:
                return None
            ug1s = [c["ug1"] for c in comp]
            return min(ug1_bias - min(ug1s), max(ug1s) - ug1_bias)

        hs = _resolve_dft_swing(half_swing, _pp_fallback)
        if hs is None:
            return None
        if ug2_filter is not None:
            ug2_for_dft = ug2_filter
        else:
            # ML-118: evaluating an unwrapped pentode model at Ug2=0 cuts
            # the tube off — every DFT sample dies and the whole grid ends
            # as 'no_valid_points' with no cause (real path: empty
            # ug2_calc_combo → ug2_filter=None). Resolve the screen from
            # the measured data instead; optimize_pp warns once.
            ug2s = [p.get("ug2", 0.0) for p in points_a] if points_a else []
            ug2_for_dft = float(np.median(ug2s)) if ug2s else 0.0
        if method == HD_METHOD_DFT:
            return compute_distortion_dft_pp(
                model, load_line, ug1_bias=ug1_bias,
                half_swing=hs, ug2=ug2_for_dft,
            )
        return compute_distortion_chebyshev_pp_model(
            model, load_line, ug1_bias=ug1_bias,
            half_swing=hs, ug2=ug2_for_dft,
        )
    if method == HD_METHOD_CHEBYSHEV:
        return compute_distortion_chebyshev_pp(
            points_a, load_line, ug1_bias=ug1_bias,
            points_b=points_b, half_swing=half_swing, ug2_filter=ug2_filter,
            transfer=transfer,
        )
    # "5point" (default)
    return pp_distortion(
        points_a, load_line, ug1_bias,
        points_b=points_b, half_swing=half_swing, ug2_filter=ug2_filter,
        transfer=transfer,
    )


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class OptimizerConstraints:
    """User-defined constraints for optimization."""

    target: str = "min_thd"        # "min_thd" | "max_pout" | "balanced"
    pa_max_w: float = 12.5         # max anode dissipation (W)
    pout_min_w: float = 0.0        # min output power (W), 0 = no constraint
    thd_max_pct: float = 0.0       # max THD (%), 0 = no constraint
    balanced_weight: float = 0.5   # weight for Pout in balanced mode (0..1)

    # HD method (matches UI hd_method_combo). "auto" = Chebyshev grid +
    # DFT refine if model present, else Chebyshev throughout.
    # "dft" = DFT throughout (slow, requires model; falls back to
    # Chebyshev with warning if model missing).
    hd_method: str = HD_METHOD_5POINT      # "5point" | "chebyshev" | "dft" | "auto"

    # PP class A power threshold (P_A = Iq² × Ra_aa / 8)
    # Off → no filter. Absolute → value in W. Percent → value as % of max Pout.
    class_a_power_mode: str = "off"     # "off" | "absolute" | "percent"
    class_a_power_value: float = 0.0    # W (absolute) or % (percent)

    # Circuit topology
    circuit: str = CIRCUIT_SE            # "se" | "se_xfmr" | "cf" | "pp"
    ra_dc: float = 0.05           # SE Transformer: DC winding resistance (kΩ)
    cf_rk: float = 10.0           # CF: cathode resistor (kΩ)
    cf_rl: float = 10.0           # CF: load resistor (kΩ)
    pp_raa: float = 8.0           # PP: anode-to-anode impedance (kΩ)
    pp_ra_dc: float = 0.1         # PP: half-primary DC winding resistance (kΩ)

    # Ultralinear tap sweep (PP only — ignored for SE/CF).
    # mode: "off" → no sweep, single tap = ul_tap_manual.
    # mode: "presets" → use historical preset values where enabled.
    # mode: "custom" → equally-spaced values across ul_tap_range.
    # mode: "presets_custom" → union of both.
    ul_tap_mode: str = "off"
    ul_tap_manual: float = 0.0       # tap when mode=off (0=pentode, 1=triode)
    # Historical UL taps with default-enabled flags (parallel tuples).
    # Values are fractions (0=pentode … 1=triode).
    ul_tap_presets: Tuple[float, ...] = (0.0, 0.20, 0.35, 0.43, 0.50, 1.0)
    ul_tap_presets_enabled: Tuple[bool, ...] = (True, True, True, True, True, True)
    ul_tap_range: Tuple[float, float] = (0.0, 1.0)
    ul_tap_steps: int = 11

    # Parameter ranges
    ug1_range: Tuple[float, float] = (-20.0, -1.0)
    ra_range: Tuple[float, float] = (1.0, 50.0)
    ub_range: Optional[Tuple[float, float]] = None   # None = fixed (measurements)
    ug2_range: Optional[Tuple[float, float]] = None   # None = fixed or triode

    # Grid resolution
    ug1_steps: int = DEFAULT_UG1_STEPS
    ra_steps: int = DEFAULT_RA_STEPS
    ub_steps: int = DEFAULT_UB_STEPS
    ug2_steps: int = DEFAULT_UG2_STEPS
    swing_steps: int = DEFAULT_SWING_STEPS


@dataclass
class OptPoint:
    """Single evaluated operating point."""

    ub: float
    ug2: float
    ug1: float
    ra: float
    thd: float
    hd2: float
    hd3: float
    pout_mw: float
    pa_mw: float          # Ua_q * Ia_q
    ia_0: float
    ua_0: float
    amp_class: str
    max_swing: float
    half_swing: float = 0.0   # optimized signal amplitude (V), 0 = max
    p_classA_w: float = 0.0   # PP only: class A power threshold (W); 0 for SE/CF
    hd_method: str = HD_METHOD_5POINT # method actually used to evaluate this point
    ul_tap: float = 0.0       # UL screen tap fraction (0=pentode, 1=triode); 0 if off
    valid: bool = True     # passes all constraints
    # Passes the hard constraints (Pa/Pout/class-A) but fails ONLY the
    # THD cap. THD falls with reduced swing, so such points stay eligible
    # for the phase-2 swing sweep — see _swing_sweep_candidates. Mutually
    # exclusive with valid=True.
    cap_only_fail: bool = False


@dataclass
class OptimizerResult:
    """Complete optimization result."""

    grid_points: List[OptPoint] = field(default_factory=list)
    pareto_front: List[OptPoint] = field(default_factory=list)
    best: Optional[OptPoint] = None
    refined: Optional[OptPoint] = None
    refined_pareto: List[OptPoint] = field(default_factory=list)
    error: Optional[str] = None
    warning: Optional[str] = None  # Non-fatal note (e.g., "dft requested but no model — used chebyshev")
    # Additional non-fatal degradation codes (refine phase etc.); the UI
    # renders both ``warning`` and this list via amp.opt_warn_* keys.
    warnings: List[str] = field(default_factory=list)


# ── Grid sweep for measurements ──────────────────────────────────

def optimize_measurements(
    points: List[Dict],
    ub: float,
    constraints: OptimizerConstraints,
    ug2_filter: Optional[float] = None,
    ug2_values: Optional[List[float]] = None,
    model: Optional["TubeModelProtocol"] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> OptimizerResult:
    """Grid sweep (Ub × Ug2 × Ug1 × Ra) for measurement data.

    Ub is virtual (analysis parameter): it shifts the load line
    `Ia = (Ub − Ua)/Ra` over the same measured I-V family. When
    `constraints.ub_range` is set, Ub is swept; otherwise fixed at `ub`.

    Ug2 iterates over available values from data (discrete, not
    continuous). If ug2_values is None and ug2_filter is set, only
    that single Ug2 is used.

    Args:
        points: raw measurement dicts [{ua, ug1, ia, ...}].
        ub: supply voltage (V) — fallback when ub_range is None.
        constraints: optimization constraints and ranges.
        ug2_filter: single Ug2 filter (used when ug2_values is None).
        ug2_values: list of available Ug2 values from data (pentode).
            When provided, sweeps all of them.

    Returns:
        OptimizerResult with grid points, Pareto front, and best point.
    """
    ug1_min, ug1_max = constraints.ug1_range
    ra_min, ra_max = constraints.ra_range

    # Ub sweep (virtual parameter — load line shifts over fixed I-V data)
    ub_values = _make_grid(constraints.ub_range, constraints.ub_steps, default=ub)

    # Determine which Ug2 values to sweep
    if ug2_values is not None and len(ug2_values) > 0:
        ug2_list = ug2_values
    elif ug2_filter is not None:
        ug2_list = [ug2_filter]
    else:
        ug2_list = [0.0]

    # Resolve grid/refine HD methods from constraints.hd_method choice
    grid_method, _refine_method, warning = _resolve_methods(
        constraints.hd_method, has_model=(model is not None),
    )

    # se_xfmr (#10): the physically correct AC load line passes through the
    # DC Q-point at Ua ≈ Ub and swings the anode ABOVE the supply — where
    # scan data does not exist. With a fitted model, intersections come
    # from the model (extrapolates past measured Ua, honours the Q-point);
    # without one, the naive straight line from Ub is kept and the
    # inaccuracy is surfaced to the user.
    xfmr_model_isects = constraints.circuit == CIRCUIT_SE_XFMR and model is not None
    if (constraints.circuit == CIRCUIT_SE_XFMR and model is None
            and warning is None):
        warning = OPT_WARN_SE_XFMR_NO_MODEL

    all_points: List[OptPoint] = []

    # Cache intersections for swing re-evaluation. Key is (ub, ug2, ra) —
    # plus ug1 in xfmr-model mode, where the AC line depends on the bias.
    isects_cache: Dict[Tuple, List[Dict]] = {}

    # Pre-group points per Ug2 once — grouping+sorting is invariant across
    # the whole (ub, ra) grid and used to be redone on every combination.
    from lm19.amplifier.distortion import group_curves_by_ug1
    curves_by_ug2: Dict[float, List] = {
        ug2: group_curves_by_ug1(
            points, ug2_filter=(ug2 if ug2 > 0 else None))
        for ug2 in ug2_list
    }

    # Cancellation: polled in every loop level (inner check is a cheap
    # callable call) → Cancel responds within a few evaluations.
    total_rows = len(ub_values) * len(ug2_list) * constraints.ra_steps
    rows_done = 0
    for ub_v in ub_values:
        if cancelled and cancelled():
            break
        for ug2 in ug2_list:
            ug2_filt = ug2 if ug2 > 0 else None
            # Data curve family — the model intersections in xfmr mode are
            # evaluated at the same Ug1 levels the scan actually measured.
            ug1_levels = [c[0] for c in curves_by_ug2[ug2]]
            for i_ra in range(constraints.ra_steps):
                if cancelled and cancelled():
                    break
                ra = ra_min + (ra_max - ra_min) * i_ra / max(constraints.ra_steps - 1, 1)
                ll = _make_load_line(ub_v, ra, constraints)
                rows_done += 1
                if on_progress:
                    on_progress(rows_done, total_rows)

                if xfmr_model_isects:
                    # Q-point AC line depends on ug1 → per-point isects.
                    if len(ug1_levels) < MIN_ISECTS_FOR_ANALYSIS:
                        continue
                    ua_range = (MODEL_UA_MIN_V,
                                _model_ua_max(ub_v, constraints.circuit))
                    for i_ug1 in range(constraints.ug1_steps):
                        if cancelled and cancelled():
                            break
                        ug1 = ug1_min + (ug1_max - ug1_min) * i_ug1 / max(constraints.ug1_steps - 1, 1)
                        isects = find_intersections_model(
                            model, ll, ug1_levels, ug2=ug2,
                            ua_range=ua_range, ug1_bias=ug1,
                        )
                        if len(isects) < MIN_ISECTS_FOR_ANALYSIS:
                            continue
                        isects_cache[(ub_v, ug2, ra, round(ug1, 6))] = isects
                        pt = _evaluate_point_measurements(
                            isects, ll, ub_v, ug2, ug1, ra, constraints,
                            method=grid_method, model=model,
                        )
                        if pt is not None:
                            all_points.append(pt)
                    continue

                isects = find_intersections(points, ll, ug2_filter=ug2_filt,
                                            curves=curves_by_ug2[ug2])
                if len(isects) < MIN_ISECTS_FOR_ANALYSIS:
                    continue
                isects_cache[(ub_v, ug2, ra)] = isects

                for i_ug1 in range(constraints.ug1_steps):
                    if cancelled and cancelled():
                        break
                    ug1 = ug1_min + (ug1_max - ug1_min) * i_ug1 / max(constraints.ug1_steps - 1, 1)
                    pt = _evaluate_point_measurements(
                        isects, ll, ub_v, ug2, ug1, ra, constraints,
                        method=grid_method, model=model,
                    )
                    if pt is not None:
                        all_points.append(pt)

    # Phase 2: swing sweep on top-N points (use cached intersections)
    if all_points and constraints.swing_steps > 1 and not (cancelled and cancelled()):
        top_n = _swing_sweep_candidates(all_points, constraints)
        if top_n:

            def eval_meas(ub_v: float, ug2_v: float, ug1_v: float,
                          ra_v: float, hs: float) -> Optional[OptPoint]:
                if xfmr_model_isects:
                    cache_key = (ub_v, ug2_v, ra_v, round(ug1_v, 6))
                else:
                    cache_key = (ub_v, ug2_v, ra_v)
                isects = isects_cache.get(cache_key)
                if isects is None or len(isects) < MIN_ISECTS_FOR_ANALYSIS:
                    return None
                ll = _make_load_line(ub_v, ra_v, constraints)
                dist = _compute_dist(
                    grid_method, isects, model, ll,
                    ug1_bias=ug1_v, half_swing=hs, ug2=ug2_v, ub=ub_v,
                )
                if dist is None:
                    return None
                return _build_opt_point(
                    dist, ub_v, ug2_v, ug1_v, ra_v, isects, ll, constraints,
                    method=grid_method,
                )

            swing_pts = _sweep_swing_top_n(top_n, eval_meas, constraints,
                                           cancelled=cancelled)
            all_points.extend(swing_pts)

    result = _build_result(all_points, constraints)
    if warning is not None:
        result.warning = warning
    # obs-2: a filter with zero matches -> fall back to the unfiltered
    # set (screen levels get mixed) — the code goes to the warning
    # channel. The ug2_values path is not checked: values come from the
    # data itself.
    if (ug2_values is None and ug2_filter is not None and points
            and not ug2_filter_matches_any(points, ug2_filter)):
        result.warnings.append(OPT_WARN_UG2_FILTER_NO_MATCH)
    return result


# ── Grid sweep for models ────────────────────────────────────────

def optimize_model(
    model: "TubeModelProtocol",
    constraints: OptimizerConstraints,
    ug1_values: Optional[List[float]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> OptimizerResult:
    """Up to 4D grid sweep (Ub × Ug2 × Ug1 × Ra) for tube models.

    When ub_range / ug2_range are None, those parameters are fixed.

    Args:
        model: TubeModelProtocol with ia(ua, ug1, ug2) method.
        constraints: optimization constraints and ranges.
        ug1_values: Ug1 values from data (for model intersection grid).
            If None, generated from ug1_range.

    Returns:
        OptimizerResult with grid points, Pareto front, and best point.
    """
    ug1_min, ug1_max = constraints.ug1_range
    ra_min, ra_max = constraints.ra_range

    # Build parameter grids
    ub_values = _make_grid(constraints.ub_range, constraints.ub_steps, default=DEFAULT_UB_V)
    ug2_values = _make_grid(constraints.ug2_range, constraints.ug2_steps, default=0.0)

    if ug1_values is None:
        ug1_values = list(np.linspace(ug1_min, ug1_max, DEFAULT_UG1_GRID_SIZE))

    grid_method, _refine_method, warning = _resolve_methods(
        constraints.hd_method, has_model=True,
    )

    all_points: List[OptPoint] = []

    # Resistive/CF load lines ignore ug1_bias → the intersection family is
    # invariant across the inner ug1 loop; compute it ONCE per (ub,ug2,ra).
    # Transformer (se_xfmr) shifts its AC line through a bias-dependent
    # Q-point, so it keeps the per-point search.
    ug1_invariant_ll = constraints.circuit not in (CIRCUIT_SE_XFMR, CIRCUIT_PP)

    total_rows = len(ub_values) * len(ug2_values) * constraints.ra_steps
    rows_done = 0
    for ub in ub_values:
        if cancelled and cancelled():
            break
        for ug2 in ug2_values:
            for i_ra in range(constraints.ra_steps):
                if cancelled and cancelled():
                    break
                ra = ra_min + (ra_max - ra_min) * i_ra / max(constraints.ra_steps - 1, 1)
                ll = _make_load_line(ub, ra, constraints)
                rows_done += 1
                if on_progress:
                    on_progress(rows_done, total_rows)

                isects_shared: Optional[List[Dict]] = None
                if ug1_invariant_ll:
                    ua_max = _model_ua_max(ub, constraints.circuit)
                    isects_shared = find_intersections_model(
                        model, ll, ug1_values, ug2=ug2,
                        ua_range=(MODEL_UA_MIN_V, ua_max),
                    )
                    if len(isects_shared) < MIN_ISECTS_FOR_ANALYSIS:
                        continue

                for i_ug1 in range(constraints.ug1_steps):
                    if cancelled and cancelled():
                        break
                    ug1 = ug1_min + (ug1_max - ug1_min) * i_ug1 / max(constraints.ug1_steps - 1, 1)
                    pt = _evaluate_point_model(
                        model, ll, ub, ug2, ug1, ra, ug1_values, constraints,
                        method=grid_method, isects=isects_shared,
                    )
                    if pt is not None:
                        all_points.append(pt)

    # Phase 2: swing sweep on top-N points (cache intersections)
    if all_points and constraints.swing_steps > 1 and not (cancelled and cancelled()):
        top_n = _swing_sweep_candidates(all_points, constraints)
        if top_n:

            # Pre-compute intersections for each unique (ub, ug2, ra)
            model_isects_cache: Dict[Tuple[float, float, float], List[Dict]] = {}
            for pt in top_n:
                key = (pt.ub, pt.ug2, pt.ra)
                if key not in model_isects_cache:
                    ll = _make_load_line(pt.ub, pt.ra, constraints)
                    ua_max = _model_ua_max(pt.ub, constraints.circuit)
                    isects = find_intersections_model(
                        model, ll, ug1_values, ug2=pt.ug2,
                        ua_range=(MODEL_UA_MIN_V, ua_max),
                        ug1_bias=pt.ug1,
                    )
                    model_isects_cache[key] = isects

            def eval_mdl(ub_v: float, ug2_v: float, ug1_v: float,
                         ra_v: float, hs: float) -> Optional[OptPoint]:
                isects = model_isects_cache.get((ub_v, ug2_v, ra_v), [])
                if len(isects) < MIN_ISECTS_FOR_ANALYSIS:
                    return None
                ll = _make_load_line(ub_v, ra_v, constraints)
                dist = _compute_dist(
                    grid_method, isects, model, ll,
                    ug1_bias=ug1_v, half_swing=hs, ug2=ug2_v, ub=ub_v,
                )
                if dist is None:
                    return None
                return _build_opt_point(
                    dist, ub_v, ug2_v, ug1_v, ra_v, isects, ll, constraints,
                    method=grid_method,
                )

            swing_pts = _sweep_swing_top_n(top_n, eval_mdl, constraints,
                                           cancelled=cancelled)
            all_points.extend(swing_pts)

    result = _build_result(all_points, constraints)
    if warning is not None:
        result.warning = warning
    return result


# ── Grid sweep for push-pull ────────────────────────────────────

def optimize_pp(
    points_a: List[Dict],
    ub: float,
    constraints: OptimizerConstraints,
    points_b: Optional[List[Dict]] = None,
    ug2_filter: Optional[float] = None,
    model: Optional["TubeModelProtocol"] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> OptimizerResult:
    """Grid sweep (Ub × Ug1 × Ra_aa) for push-pull stage.

    Uses _compute_dist_pp() dispatcher: 5-point / Chebyshev / DFT
    variants on the composite characteristic per constraints.hd_method.
    Ra range is interpreted as Ra_aa (anode-to-anode impedance).
    Ub is virtual: varied when constraints.ub_range is set, otherwise
    fixed at `ub`.

    Args:
        points_a: tube A measurement data.
        ub: supply voltage (V) — fallback when ub_range is None.
        constraints: optimization constraints. circuit must be "pp".
        points_b: tube B data (None = matched pair).
        ug2_filter: Ug2 filter for pentode data.
        model: optional tube model (enables DFT method).

    Returns:
        OptimizerResult with grid points, Pareto front, and best point.
    """
    ug1_min, ug1_max = constraints.ug1_range
    ra_min, ra_max = constraints.ra_range

    ub_values = _make_grid(constraints.ub_range, constraints.ub_steps, default=ub)

    # Resolve grid/refine HD methods for PP (refine method not used here —
    # refine_pareto_front handles refinement separately).
    grid_method, _refine_method, warning = _resolve_methods(
        constraints.hd_method, has_model=(model is not None),
    )
    result_warnings: List[str] = []

    # UL tap sweep: needs a model-evaluated method. DFT always qualifies;
    # Chebyshev qualifies through the model variant (tap>0 points route
    # to compute_distortion_chebyshev_pp_model). 5-point stays data-only.
    if (model is not None and points_b is not None
            and grid_method in (HD_METHOD_DFT, HD_METHOD_CHEBYSHEV)):
        # ③: model paths (DFT / UL model-Chebyshev) assume a MATCHED pair.
        log.warning("PP optimizer: mismatched pair (tube B data supplied) "
                    "— model-based %s falls back to the data composite; "
                    "UL sweep unavailable", grid_method)
        result_warnings.append(OPT_WARN_DFT_MISMATCHED_PAIR)
    if (model is not None and points_b is None
            and grid_method in (HD_METHOD_DFT, HD_METHOD_CHEBYSHEV)):
        # Chebyshev is UL-capable through the MODEL variant
        # (compute_distortion_chebyshev_pp_model — same joint solve as
        # DFT at ~30× fewer Newton calls); tap=0 points stay on the
        # data path. 5-point remains data-only.
        ul_taps = _resolve_ul_taps(constraints)
    else:
        ul_taps = [max(0.0, min(1.0, constraints.ul_tap_manual))]
        if constraints.ul_tap_mode != "off":
            # The user asked for a UL sweep it cannot run (no model /
            # 5-point method) — never skip silently (failure-visibility rule).
            log.warning("UL tap sweep skipped: requires a fitted model "
                        "and hd_method dft/chebyshev/auto (got method=%s, "
                        "model=%s)", grid_method,
                        "yes" if model is not None else "no")
            if OPT_WARN_UL_SWEEP_SKIPPED not in result_warnings:
                result_warnings.append(OPT_WARN_UL_SWEEP_SKIPPED)

    _model_path_used = grid_method == HD_METHOD_DFT or (
        grid_method == HD_METHOD_CHEBYSHEV and any(t > 0.0 for t in ul_taps))
    if _model_path_used and model is not None and ug2_filter is None:
        # ML-118: per-point resolution happens in _compute_dist_pp (median
        # measured Ug2); warn ONCE here instead of per grid evaluation, and
        # surface it in the UI via the standard OptimizerResult.warning
        # channel (amp.opt_warn_* — same as dft_no_model_fallback). Only
        # when the data actually carries a screen voltage: triode scans
        # resolve to Ug2=0, which is exactly right — no false alarm.
        _ug2s = [p.get("ug2", 0.0) for p in points_a] if points_a else []
        if _ug2s and float(np.median(_ug2s)) > 0.0:
            log.warning("PP DFT without a Ug2 filter — the model screen "
                        "voltage is resolved from the measured data "
                        "(median Ug2)")
            if warning is None:
                warning = OPT_WARN_PP_DFT_UG2_FROM_DATA

    all_points: List[OptPoint] = []

    # ug2_for_ul: nominal Ug2 for UL wrapper (ML-063: grid anchored on
    # ub_values[0] while refine anchored on best.ub — with a Ub sweep and
    # no ug2_filter the two phases evaluated DIFFERENT screen physics).
    # One resolution everywhere: filter → median of measured Ug2 (the
    # ML-118 convention; the wrapper ignores the evaluation ug2, only
    # ug2_nom matters) → the supplied ub as a last resort.
    ug2_nom = _resolve_ul_ug2_nom(ug2_filter, points_a, ub)
    # obs-1: the resolved screen is written into OptPoint.ug2 — status/
    # tooltip/Top-N/apply show the actual analysis screen (PP-data
    # points used to carry ug2=0 and the UI hid it behind a >0 guard).
    ug2_display = _resolve_display_ug2(ug2_filter, points_a)
    # obs-2: a filter matching no points falls back to the unfiltered
    # set (see _apply_ug2_filter, already logged) — surfaced to the UI
    # warning channel via a warning code.
    if (ug2_filter is not None and points_a
            and not ug2_filter_matches_any(points_a, ug2_filter)):
        result_warnings.append(OPT_WARN_UG2_FILTER_NO_MATCH)

    # Transfer curves depend on (points, ug2_filter, Ub) — Ia is taken AT
    # the supply voltage (per-tube DC operating point). Build once per Ub
    # instead of per grid point (was ~90% of a 5pt/cheb eval).
    from lm19.amplifier import build_pp_transfer
    pp_transfer_by_ub = {
        ub_v: build_pp_transfer(points_a, points_b, ug2_filter, ua_ref=ub_v)
        for ub_v in ub_values
    }

    total_rows = len(ul_taps) * len(ub_values) * constraints.ra_steps
    rows_done = 0
    for ul_tap in ul_taps:
        if cancelled and cancelled():
            break
        if ul_tap > 0.0 and model is not None:
            wrapped_model = UltralinearModelWrapper(
                model, ug2_nom=float(ug2_nom), tap=float(ul_tap),
            )
        else:
            wrapped_model = model

        for ub_v in ub_values:
            if cancelled and cancelled():
                break
            for i_ra in range(constraints.ra_steps):
                if cancelled and cancelled():
                    break
                ra_aa = ra_min + (ra_max - ra_min) * i_ra / max(constraints.ra_steps - 1, 1)
                ll = _make_load_line(ub_v, ra_aa, constraints)
                rows_done += 1
                if on_progress:
                    on_progress(rows_done, total_rows)

                for i_ug1 in range(constraints.ug1_steps):
                    if cancelled and cancelled():
                        break
                    ug1 = ug1_min + (ug1_max - ug1_min) * i_ug1 / max(constraints.ug1_steps - 1, 1)
                    pt = _evaluate_point_pp(
                        points_a, ll, ub_v, ug1, ra_aa, constraints,
                        points_b=points_b, ug2_filter=ug2_filter,
                        method=grid_method, model=wrapped_model,
                        ul_tap=ul_tap, transfer=pp_transfer_by_ub[ub_v],
                        ug2_display=ug2_display,
                    )
                    if pt is not None:
                        all_points.append(pt)

    # Phase 2: swing sweep on top-N points. Group by ul_tap so each
    # group reuses the correctly-wrapped model — without grouping the
    # closure couldn't know which tap to apply for each pt.
    if all_points and constraints.swing_steps > 1 and not (cancelled and cancelled()):
        top_n = _swing_sweep_candidates(all_points, constraints)
        if top_n:

            # Bucket top-N points by their tap value
            taps_in_top: Dict[float, List[OptPoint]] = {}
            for pt in top_n:
                taps_in_top.setdefault(round(pt.ul_tap, 3), []).append(pt)

            for tap, pts_at_tap in taps_in_top.items():
                if tap > 0.0 and model is not None:
                    sw_model = UltralinearModelWrapper(
                        model, ug2_nom=float(ug2_nom), tap=float(tap),
                    )
                else:
                    sw_model = model

                def _make_eval(_tap: float, _model):
                    def eval_pp(ub_v: float, ug2_v: float, ug1_v: float,
                                ra_v: float, hs: float) -> Optional[OptPoint]:
                        ll_sw = _make_load_line(ub_v, ra_v, constraints)
                        return _evaluate_point_pp(
                            points_a, ll_sw, ub_v, ug1_v, ra_v, constraints,
                            points_b=points_b, ug2_filter=ug2_filter,
                            half_swing=hs, method=grid_method, model=_model,
                            ul_tap=_tap, transfer=pp_transfer_by_ub.get(ub_v),
                            ug2_display=ug2_display,
                        )
                    return eval_pp

                swing_pts = _sweep_swing_top_n(
                    pts_at_tap, _make_eval(tap, sw_model), constraints,
                    cancelled=cancelled,
                )
                all_points.extend(swing_pts)

    result = _build_result(all_points, constraints)
    if warning is not None:
        result.warning = warning
    for code in result_warnings:
        if code != result.warning and code not in result.warnings:
            result.warnings.append(code)
    return result


def _resolve_ul_ug2_nom(
    ug2_filter: Optional[float],
    points: Optional[List[Dict]],
    fallback: float,
) -> float:
    """Nominal screen voltage for the UL wrapper — ONE rule for grid,
    swing sweep and refine (ML-063). Median of the measured Ug2 mirrors
    the ML-118 evaluation convention; ``fallback`` only fires for data
    with no screen voltage at all (triode-ish, where UL never engages).
    """
    if ug2_filter is not None:
        return float(ug2_filter)
    ug2s = [p.get("ug2", 0.0) for p in points] if points else []
    med = float(np.median(ug2s)) if ug2s else 0.0
    return med if med > 0.0 else float(fallback)


def _resolve_display_ug2(
    ug2_filter: Optional[float],
    points: Optional[List[Dict]],
) -> float:
    """Screen voltage recorded on PP-data OptPoints — display/apply
    metadata only (the PP data path evaluates the screen from the
    filtered points themselves; ``eval_pp`` ignores the value).
    Unlike ``_resolve_ul_ug2_nom`` there is NO ub-fallback: triode data
    honestly reports 0 (the UI hides Ug2 at 0 — a ub-valued screen on
    a triode would be a lie in the status line / Top-N / apply)."""
    if ug2_filter is not None:
        return float(ug2_filter)
    ug2s = [p.get("ug2", 0.0) for p in points] if points else []
    med = float(np.median(ug2s)) if ug2s else 0.0
    return med if med > 0.0 else 0.0


def _evaluate_point_pp(
    points_a: List[Dict],
    ll: PushPullLoadLine,
    ub: float,
    ug1: float,
    ra_aa: float,
    constraints: OptimizerConstraints,
    points_b: Optional[List[Dict]] = None,
    ug2_filter: Optional[float] = None,
    half_swing: Optional[float] = None,
    method: str = HD_METHOD_5POINT,
    model: Optional["TubeModelProtocol"] = None,
    ul_tap: float = 0.0,
    transfer: Optional[Tuple] = None,
    ug2_display: Optional[float] = None,
) -> Optional[OptPoint]:
    """Evaluate a single PP operating point via the chosen HD method.

    ul_tap is informational here — caller should already have wrapped
    `model` in UltralinearModelWrapper if tap > 0; we record the value
    on OptPoint so Top-N picker can show it. ``transfer`` forwards the
    pre-built PP transfer pair. ``ug2_display`` is the resolved screen
    recorded on the point (metadata for status/Top-N/apply; None →
    self-compute — grid callers pass the precomputed value to keep the
    hot path free of per-point medians).
    """
    if ug2_display is None:
        ug2_display = _resolve_display_ug2(ug2_filter, points_a)
    dist = _compute_dist_pp(
        method, points_a, points_b, ll,
        ug1_bias=ug1, half_swing=half_swing, ug2_filter=ug2_filter,
        model=model, transfer=transfer,
    )
    if dist is None:
        return None

    pout_mw = dist["pout_mw"]
    if pout_mw < MIN_POUT_MW:
        return None

    ia_0 = dist["ia_0"]                               # composite at center (≈0 for matched)
    iq_per_tube = dist.get("iq_per_tube", ia_0)        # actual DC plate current per tube

    # PP: each tube dissipates Pa = Ub × Iq_per_tube (NOT the composite ia_0,
    # which is ≈ 0 for a matched pair).
    pa_mw = ub * iq_per_tube  # V × mA = mW
    pa_w = pa_mw / MW_PER_W

    # Classify on the PER-TUBE minimum: the composite i_min is ≈ −i_max
    # for a matched pair (strongly negative), which made every PP point
    # come out class "B" regardless of the actual operating point.
    amp_class = _classify_amp(
        iq_per_tube, dist.get("i_min_per_tube", 0.0))
    max_swing = dist.get("half_swing", 0.0)

    # PP class-A power threshold: P_A = Iq² × Ra_aa / 8 (boundary where
    # the off-going tube reaches cutoff at signal amplitude ΔI = Iq).
    # Uses per-tube quiescent current, not composite.
    p_classA_w = (iq_per_tube * iq_per_tube) * ra_aa / P_CLASSA_DIVISOR

    valid = True
    if pa_w > constraints.pa_max_w:
        valid = False
    if constraints.pout_min_w > 0 and pout_mw / MW_PER_W < constraints.pout_min_w:
        valid = False

    # Class-A power threshold filter (PP-specific)
    threshold_w = _resolve_class_a_threshold(constraints, pout_mw)
    if threshold_w > 0 and p_classA_w < threshold_w:
        valid = False

    # THD-cap (0 = off, boundary thd == cap passes). A point that fails
    # ONLY the cap keeps swing-sweep eligibility via cap_only_fail.
    thd_capped = (constraints.thd_max_pct > 0
                  and dist["thd"] > constraints.thd_max_pct)
    cap_only_fail = valid and thd_capped
    if thd_capped:
        valid = False

    return OptPoint(
        ub=ub, ug2=ug2_display, ug1=ug1, ra=ra_aa,
        thd=dist["thd"], hd2=dist["hd2"], hd3=dist["hd3"],
        pout_mw=pout_mw, pa_mw=pa_mw,
        ia_0=iq_per_tube, ua_0=ub,  # PP Q-point: per-tube DC at Ua ≈ Ub
        amp_class=amp_class,
        max_swing=max_swing,
        half_swing=dist.get("half_swing", 0.0),
        p_classA_w=p_classA_w,
        hd_method=dist.get("method", HD_METHOD_5POINT),   # actual PP method used
        ul_tap=ul_tap,
        valid=valid,
        cap_only_fail=cap_only_fail,
    )


def _resolve_class_a_threshold(
    constraints: OptimizerConstraints, pout_mw: float,
) -> float:
    """Resolve class-A power threshold (W) from constraints.

    "off"      → 0 (no filter)
    "absolute" → constraints.class_a_power_value (W)
    "percent"  → fraction of current point's pout (W)
    """
    mode = constraints.class_a_power_mode
    val = constraints.class_a_power_value
    if mode == "absolute":
        return max(0.0, val)
    if mode == "percent":
        return max(0.0, val) / 100.0 * (pout_mw / MW_PER_W)
    return 0.0


# ── scipy refinement ─────────────────────────────────────────────

def _setup_refine_xbounds(
    best: OptPoint,
    constraints: OptimizerConstraints,
    *,
    vary_ub: bool,
    vary_ug2: bool,
    has_swing: bool,
) -> Tuple[List[float], List[Tuple[float, float]]]:
    """Build (x0, bounds) for scipy refinement.

    Vector layout: [ub?, ug2?, ug1, ra, half_swing?]
    Initial values are clamped into bounds (scipy warns otherwise).
    """
    x0: List[float] = []
    bounds: List[Tuple[float, float]] = []
    if vary_ub:
        x0.append(best.ub)
        bounds.append(constraints.ub_range)
    if vary_ug2:
        x0.append(best.ug2)
        bounds.append(constraints.ug2_range)
    x0.extend([best.ug1, best.ra])
    bounds.extend([constraints.ug1_range, constraints.ra_range])
    if has_swing:
        swing_bounds = (best.max_swing * MIN_SWING_FRACTION, best.max_swing)
        x0.append(best.half_swing)
        bounds.append(swing_bounds)

    for i, (lo, hi) in enumerate(bounds):
        x0[i] = max(lo, min(hi, x0[i]))
    return x0, bounds


def _evaluate_refined_point(
    *,
    ub_v: float,
    ug2_v: float,
    ug1_v: float,
    ra_v: float,
    hs: Optional[float],
    is_pp: bool,
    use_model: bool,
    points: Optional[List[Dict]],
    points_b: Optional[List[Dict]],
    model: Optional["TubeModelProtocol"],
    wrapped_model: Optional["TubeModelProtocol"],
    constraints: OptimizerConstraints,
    ug2_filter: Optional[float],
    ug1_values: Optional[List[float]],
    refine_method: str,
    ul_tap: float,
    pp_transfer: Optional[Tuple] = None,
    meas_curves: Optional[List] = None,
) -> Optional[OptPoint]:
    """Evaluate one (ub, ug2, ug1, ra, hs) tuple using the appropriate path.

    Used both by the scipy objective closure (which then scores) and
    by the post-minimize re-evaluation that returns the final OptPoint.
    Returns None for the measurements path when intersections are too sparse.
    ``pp_transfer`` / ``meas_curves`` carry per-refine-run precomputed data
    (transfer curves, per-Ug1 point groups) so the NM objective does not
    rebuild them on every evaluation.
    """
    ll = _make_load_line(ub_v, ra_v, constraints)
    if is_pp:
        return _evaluate_point_pp(
            points, ll, ub_v, ug1_v, ra_v, constraints,
            points_b=points_b, ug2_filter=ug2_filter,
            half_swing=hs, method=refine_method, model=wrapped_model,
            ul_tap=ul_tap, transfer=pp_transfer,
        )
    if use_model:
        ug1_grid = ug1_values or list(np.linspace(
            constraints.ug1_range[0], constraints.ug1_range[1], DEFAULT_UG1_GRID_SIZE,
        ))
        return _evaluate_point_model(
            model, ll, ub_v, ug2_v, ug1_v, ra_v, ug1_grid, constraints,
            half_swing=hs, method=refine_method,
        )
    isects = find_intersections(points, ll, ug2_filter=ug2_filter,
                                curves=meas_curves)
    if len(isects) < MIN_ISECTS_FOR_ANALYSIS:
        return None
    return _evaluate_point_measurements(
        isects, ll, ub_v, ug2_v, ug1_v, ra_v, constraints,
        half_swing=hs, method=refine_method, model=model,
    )


class _RefineX(NamedTuple):
    """Decoded scipy-optimizer vector for ``refine_optimum``.

    The vector layout depends on which axes are varied (Ub, Ug2, swing).
    ``_parse_x`` extracts named values; the type alias makes the
    5-element shape self-documenting (vs ``Tuple[float, float, float,
    float, Optional[float]]`` which has 4 indistinguishable floats).

    NamedTuple is a ``tuple`` subclass, so call sites can still unpack
    as ``ub, ug2, ug1, ra, hs = _parse_x(x)``.
    """
    ub: float
    ug2: float
    ug1: float
    ra: float
    hs: Optional[float]


class _RefineCancelled(Exception):
    """Raised inside the NM objective to abort a cancelled refinement."""


def refine_optimum(
    best: OptPoint,
    points: Optional[List[Dict]],
    model: Optional["TubeModelProtocol"],
    constraints: OptimizerConstraints,
    ug2_filter: Optional[float] = None,
    ug1_values: Optional[List[float]] = None,
    points_b: Optional[List[Dict]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    warnings_out: Optional[List[str]] = None,
) -> Optional[OptPoint]:
    """Refine the best grid point using scipy.optimize.

    Uses Nelder-Mead (no gradient needed) starting from the best grid point.
    ``cancelled`` is polled inside the objective so a user Cancel aborts
    mid-minimize instead of waiting out a full NM run.
    """
    def _add_warning(code: str) -> None:
        # list.append is atomic under the GIL — safe from pool threads.
        if warnings_out is not None and code not in warnings_out:
            warnings_out.append(code)

    try:
        from scipy.optimize import minimize
    except ImportError:
        log.warning("scipy not available, skipping refinement")
        _add_warning(OPT_WARN_REFINE_UNAVAILABLE)
        return None

    use_model = model is not None
    is_pp = constraints.circuit == CIRCUIT_PP

    # Per-run precomputed inputs for the objective (invariant across NM
    # evaluations): PP transfer curves / per-Ug1 measurement groups.
    pp_transfer = None
    meas_curves = None
    if is_pp and points and constraints.ub_range is None:
        # Transfer is taken at Ua=Ub — precompute only when NM does not
        # vary Ub; otherwise pp_distortion rebuilds per eval at ll.ub.
        from lm19.amplifier import build_pp_transfer
        pp_transfer = build_pp_transfer(points, points_b, ug2_filter,
                                        ua_ref=best.ub)
    elif not use_model and points:
        from lm19.amplifier.distortion import group_curves_by_ug1
        meas_curves = group_curves_by_ug1(points, ug2_filter=ug2_filter)

    # Preserve UL tap from the input point — scipy refinement should
    # search around the same UL configuration that grid found, not
    # silently fall back to pentode (tap=0).
    if is_pp and use_model and best.ul_tap > 0.0:
        # ML-063: same anchor rule as the grid (median of measured Ug2) —
        # best.ub stays only as the no-screen-data fallback.
        ug2_for_ul = _resolve_ul_ug2_nom(ug2_filter, points, best.ub)
        wrapped_model = UltralinearModelWrapper(
            model, ug2_nom=float(ug2_for_ul), tap=float(best.ul_tap),
        )
    else:
        wrapped_model = model

    # Resolve refine method (auto/dft → DFT if model exists; else Chebyshev)
    _grid_method, refine_method, _warn = _resolve_methods(
        constraints.hd_method, has_model=use_model,
    )

    has_swing = best.half_swing > 0 and best.max_swing > 0
    # Ub is virtual — varies for measurements/model/PP whenever ub_range set.
    vary_ub = constraints.ub_range is not None
    vary_ug2 = use_model and constraints.ug2_range is not None

    x0, bounds = _setup_refine_xbounds(
        best, constraints,
        vary_ub=vary_ub, vary_ug2=vary_ug2, has_swing=has_swing,
    )

    def _parse_x(x: np.ndarray) -> _RefineX:
        """Extract (ub, ug2, ug1, ra, half_swing) from optimizer vector."""
        x_list = [float(v) for v in x]
        hs = x_list.pop() if has_swing else None
        ub_v = x_list.pop(0) if vary_ub else best.ub
        ug2_v = x_list.pop(0) if vary_ug2 else best.ug2
        return _RefineX(ub_v, ug2_v, x_list[0], x_list[1], hs)

    def objective(x: np.ndarray) -> float:
        if cancelled is not None and cancelled():
            # Abort the whole minimize run — caught below. Without this a
            # user Cancel waits out up to REFINE_MAX_ITER NM iterations.
            raise _RefineCancelled()
        ub_v, ug2_v, ug1_v, ra_v, hs = _parse_x(x)
        pt = _evaluate_refined_point(
            ub_v=ub_v, ug2_v=ug2_v, ug1_v=ug1_v, ra_v=ra_v, hs=hs,
            is_pp=is_pp, use_model=use_model,
            points=points, points_b=points_b,
            model=model, wrapped_model=wrapped_model,
            constraints=constraints, ug2_filter=ug2_filter,
            ug1_values=ug1_values, refine_method=refine_method,
            ul_tap=best.ul_tap,
            pp_transfer=pp_transfer, meas_curves=meas_curves,
        )
        if pt is None or not pt.valid:
            return PENALTY_SCORE
        return _score(pt, constraints)

    try:
        result = minimize(
            objective, x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": REFINE_MAX_ITER, "xatol": REFINE_XATOL, "fatol": REFINE_FATOL},
        )
    except _RefineCancelled:
        return None
    except (ValueError, RuntimeError, FloatingPointError,
            np.linalg.LinAlgError):
        # ML-091: narrow except — an AttributeError/TypeError regression in
        # the objective must propagate, not silently fall back to the
        # unrefined grid point.
        log.exception("scipy refinement failed")
        _add_warning(OPT_WARN_REFINE_FAILED)
        return None

    if not result.success and result.fun >= PENALTY_SCORE:
        return None
    if not result.success:
        # ML: a Nelder-Mead that ran out of iterations but found a finite
        # score is kept — but per failure-visibility principle 2 the
        # "inexact win" must carry a signal.
        log.warning("refine_optimum: Nelder-Mead did not converge "
                    "(%d iterations) — keeping best-found point anyway",
                    getattr(result, "nit", -1))
        _add_warning(OPT_WARN_NM_NOT_CONVERGED)

    ub_v, ug2_v, ug1_v, ra_v, hs = _parse_x(result.x)
    return _evaluate_refined_point(
        ub_v=float(ub_v), ug2_v=float(ug2_v),
        ug1_v=float(ug1_v), ra_v=float(ra_v), hs=hs,
        is_pp=is_pp, use_model=use_model,
        points=points, points_b=points_b,
        model=model, wrapped_model=wrapped_model,
        constraints=constraints, ug2_filter=ug2_filter,
        ug1_values=ug1_values, refine_method=refine_method,
        ul_tap=best.ul_tap,
        pp_transfer=pp_transfer, meas_curves=meas_curves,
    )


# ── Pareto front refinement (parallel) ───────────────────────────

def refine_pareto_front(
    pareto: List[OptPoint],
    points: Optional[List[Dict]],
    model: Optional["TubeModelProtocol"],
    constraints: OptimizerConstraints,
    ug2_filter: Optional[float] = None,
    ug1_values: Optional[List[float]] = None,
    max_points: int = PARETO_REFINE_MAX,
    max_workers: int = PARETO_REFINE_WORKERS,
    cancelled: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    points_b: Optional[List[Dict]] = None,
    warnings_out: Optional[List[str]] = None,
) -> List[OptPoint]:
    """Refine multiple Pareto front points in parallel.

    Selects up to max_points evenly spaced along the front,
    refines each with scipy Nelder-Mead using ThreadPoolExecutor.

    **Thread-safety**: ``model`` is shared across worker threads. It MUST
    satisfy ``TubeModelProtocol``'s stateless-after-construction contract
    (see docstring there) — ``ia()`` / ``ig2()`` must not mutate ``self``
    nor depend on module-level mutable caches. The objective closure
    inside ``refine_optimum`` is read-only over ``points`` / ``points_b``
    / ``constraints``. ``UltralinearModelWrapper`` is constructed
    per-task. ``scipy.optimize.minimize(method='Nelder-Mead')`` is pure
    Python with state local to each call. Thread-safety verified
    empirically (35+ parallel refines, all byte-identical results).

    Args:
        pareto: Pareto front points (sorted by THD).
        points: measurement data (for measurements path).
        model: tube model (for model path). Must be stateless — see above.
        constraints: optimization constraints.
        ug2_filter: Ug2 filter for pentode data.
        ug1_values: Ug1 grid for model intersections.
        max_points: max number of Pareto points to refine.
        max_workers: thread pool size.
        cancelled: callable returning True to abort.
        on_progress: callable(current, total) for progress reporting.

    Returns:
        List of refined OptPoints (new Pareto front).
    """
    if not pareto:
        return []

    # Select evenly spaced points along the front
    if len(pareto) <= max_points:
        selected = list(pareto)
    else:
        indices = np.linspace(0, len(pareto) - 1, max_points, dtype=int)
        selected = [pareto[i] for i in indices]

    total = len(selected)
    refined: List[Optional[OptPoint]] = [None] * total

    def _refine_one(idx: int, pt: OptPoint) -> Tuple[int, Optional[OptPoint]]:
        if cancelled and cancelled():
            return idx, None
        return idx, refine_optimum(
            pt, points=points, model=model,
            constraints=constraints, ug2_filter=ug2_filter,
            ug1_values=ug1_values, points_b=points_b,
            cancelled=cancelled, warnings_out=warnings_out,
        )

    done_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_refine_one, i, pt): i
            for i, pt in enumerate(selected)
        }
        for future in as_completed(futures):
            if cancelled and cancelled():
                pool.shutdown(wait=False, cancel_futures=True)
                break
            idx, result = future.result()
            refined[idx] = result
            done_count += 1
            if on_progress:
                on_progress(done_count, total)

    # Merge: use refined point if better, otherwise keep original
    merged: List[OptPoint] = []
    for i, pt in enumerate(selected):
        ref = refined[i]
        if ref is not None and ref.valid:
            merged.append(ref)
        else:
            merged.append(pt)

    # Re-compute Pareto front from merged + original
    all_candidates = list(pareto) + merged
    return _compute_pareto_front(all_candidates)


# ── Internal helpers ─────────────────────────────────────────────

def _make_grid(
    value_range: Optional[Tuple[float, float]],
    steps: int,
    default: float,
) -> List[float]:
    """Build parameter grid. Returns [default] if range is None."""
    if value_range is None:
        return [default]
    lo, hi = value_range
    if steps <= 1 or lo >= hi:
        return [lo]
    return [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]


def _model_ua_max(ub: float, circuit: str) -> float:
    """Upper Ua bound for model intersection search.

    se_xfmr needs ~2×Ub (anode swings above the supply on the AC line
    through the Q-point); other circuits stay within Ub.
    """
    factor = XFMR_UA_MAX_FACTOR if circuit == CIRCUIT_SE_XFMR else UA_MAX_FACTOR
    return max(ub * factor, MODEL_UA_MAX_DEFAULT_V)


def _make_load_line(ub: float, ra: float, constraints: OptimizerConstraints) -> LoadLine:
    """Create the appropriate load line for the circuit topology."""
    if constraints.circuit == CIRCUIT_SE_XFMR:
        return TransformerLoadLine(ub, ra_dc=constraints.ra_dc, ra_ac=ra)
    if constraints.circuit == CIRCUIT_CF:
        return CathodeFollowerLoadLine(ub, rk=constraints.cf_rk, rl=ra)
    if constraints.circuit == CIRCUIT_PP:
        return PushPullLoadLine(ub, ra_aa=ra, ra_dc=constraints.pp_ra_dc)
    return ResistiveLoadLine(ub, ra)


def _classify_amp(i_0: float, i_min: float) -> str:
    """Determine amplifier class from quiescent and minimum current."""
    if i_0 <= 0:
        return "B"
    ratio = i_min / i_0
    if ratio > CLASS_A_RATIO:
        return "A"
    if ratio < CLASS_B_RATIO:
        return "B"
    return "AB"


def _evaluate_point_measurements(
    isects: List[Dict],
    ll: LoadLine,
    ub: float,
    ug2: float,
    ug1: float,
    ra: float,
    constraints: OptimizerConstraints,
    half_swing: Optional[float] = None,
    method: str = HD_METHOD_5POINT,
    model: Optional["TubeModelProtocol"] = None,
) -> Optional[OptPoint]:
    """Evaluate a single operating point from measurement intersections."""
    dist = _compute_dist(
        method, isects, model, ll, ug1_bias=ug1,
        half_swing=half_swing, ug2=ug2, ub=ub,
    )
    if dist is None:
        return None
    return _build_opt_point(dist, ub, ug2, ug1, ra, isects, ll, constraints, method)


def _evaluate_point_model(
    model: "TubeModelProtocol",
    ll: LoadLine,
    ub: float,
    ug2: float,
    ug1: float,
    ra: float,
    ug1_values: List[float],
    constraints: OptimizerConstraints,
    half_swing: Optional[float] = None,
    method: str = HD_METHOD_5POINT,
    isects: Optional[List[Dict]] = None,
) -> Optional[OptPoint]:
    """Evaluate a single operating point using a tube model.

    ``isects`` lets the grid loop pass pre-computed intersections when they
    are invariant across ug1 (Resistive/CF load lines ignore ``ug1_bias``) —
    without it the intersection search ran per grid point and dominated the
    whole phase-1 sweep.
    """
    if isects is None:
        ua_max = _model_ua_max(ub, constraints.circuit)
        isects = find_intersections_model(
            model, ll, ug1_values, ug2=ug2,
            ua_range=(MODEL_UA_MIN_V, ua_max),
            ug1_bias=ug1,
        )
    if len(isects) < MIN_ISECTS_FOR_ANALYSIS:
        return None

    dist = _compute_dist(
        method, isects, model, ll, ug1_bias=ug1,
        half_swing=half_swing, ug2=ug2, ub=ub,
    )
    if dist is None:
        return None
    return _build_opt_point(dist, ub, ug2, ug1, ra, isects, ll, constraints, method)


def _build_opt_point(
    dist: Dict,
    ub: float,
    ug2: float,
    ug1: float,
    ra: float,
    isects: List[Dict],
    ll: LoadLine,
    constraints: OptimizerConstraints,
    method: str = HD_METHOD_5POINT,
) -> Optional[OptPoint]:
    """Build OptPoint from distortion result and check constraints."""
    pout_mw = dist["pout_mw"]
    if pout_mw < MIN_POUT_MW:
        return None

    ua_0 = dist["ua_0"]
    ia_0 = dist["ia_0"]
    pa_mw = ua_0 * ia_0  # mW
    pa_w = pa_mw / MW_PER_W

    amp_class = _classify_amp(ia_0, dist.get("i_min", 0.0))

    headroom = compute_headroom(isects, ug1, pa_max=constraints.pa_max_w, load_line=ll)
    max_swing = headroom["max_swing"] if headroom else 0.0

    # Check constraints
    valid = True
    if pa_w > constraints.pa_max_w:
        valid = False
    if constraints.pout_min_w > 0 and pout_mw / MW_PER_W < constraints.pout_min_w:
        valid = False

    # THD-cap (0 = off, boundary thd == cap passes). A point that fails
    # ONLY the cap keeps swing-sweep eligibility via cap_only_fail.
    thd_capped = (constraints.thd_max_pct > 0
                  and dist["thd"] > constraints.thd_max_pct)
    cap_only_fail = valid and thd_capped
    if thd_capped:
        valid = False

    return OptPoint(
        ub=ub, ug2=ug2, ug1=ug1, ra=ra,
        thd=dist["thd"], hd2=dist["hd2"], hd3=dist["hd3"],
        pout_mw=pout_mw, pa_mw=pa_mw,
        ia_0=ia_0, ua_0=ua_0,
        amp_class=amp_class,
        max_swing=max_swing,
        half_swing=dist.get("half_swing", 0.0),
        hd_method=method,
        valid=valid,
        cap_only_fail=cap_only_fail,
    )


def _score(pt: OptPoint, constraints: OptimizerConstraints) -> float:
    """Compute scalar score for optimization (lower = better)."""
    if constraints.target == "min_thd":
        return pt.thd
    if constraints.target == "max_pout":
        return -pt.pout_mw
    # balanced: weighted combination
    # Normalize: THD typically 0.5-10%, Pout 100-10000 mW
    # Score = THD - weight * log10(Pout)
    w = constraints.balanced_weight
    pout_log = math.log10(max(pt.pout_mw, MIN_POUT_MW))
    return pt.thd - w * pout_log


def _swing_sweep_candidates(
    all_points: List[OptPoint],
    constraints: OptimizerConstraints,
) -> List[OptPoint]:
    """Top-N candidates for the phase-2 swing sweep.

    Includes cap_only_fail points: THD falls with reduced swing, so a
    point over the THD cap at max swing may satisfy it at a smaller one —
    dropping it here would lose the true Pout@THDcap optimum (and skip
    phase 2 entirely when the cap rejects the whole grid).
    """
    cands = [p for p in all_points if p.valid or p.cap_only_fail]
    return sorted(cands, key=lambda p: _score(p, constraints))[:TOP_N_FOR_SWING]


def _build_result(
    all_points: List[OptPoint],
    constraints: OptimizerConstraints,
) -> OptimizerResult:
    """Filter valid points, compute Pareto front, find best."""
    if not all_points:
        return OptimizerResult(error=OPT_ERR_NO_VALID_POINTS)

    valid = [p for p in all_points if p.valid]
    if not valid:
        # Distinguish "only the THD cap rejects everything": raising the
        # cap is then the actionable advice (failure-visibility) — any
        # cap_only_fail point becomes valid with a higher thd_max_pct.
        cap_only = any(p.cap_only_fail for p in all_points)
        return OptimizerResult(
            grid_points=all_points,
            error=(OPT_ERR_NO_POINTS_WITHIN_THD_CAP if cap_only
                   else OPT_ERR_NO_POINTS_WITHIN_CONSTRAINTS),
        )

    pareto = _compute_pareto_front(valid)
    best = min(valid, key=lambda p: _score(p, constraints))

    return OptimizerResult(
        grid_points=all_points,
        pareto_front=pareto,
        best=best,
    )


def _sweep_swing_top_n(
    top_points: List[OptPoint],
    eval_fn: Callable[[float, float, float, float, float], Optional[OptPoint]],
    constraints: OptimizerConstraints,
    cancelled: Optional[Callable[[], bool]] = None,
) -> List[OptPoint]:
    """Re-evaluate top-N points at multiple swing levels.

    For each point, sweeps half_swing from MIN_SWING_FRACTION * max_swing
    to max_swing in swing_steps levels, keeping the best per (ug1, ra, ub, ug2).

    Args:
        top_points: best points from grid sweep (evaluated at max swing).
        eval_fn: callable(ub, ug2, ug1, ra, half_swing) -> Optional[OptPoint].
        constraints: for swing_steps.
        cancelled: optional callable returning True to abort. Checked
            between top-N points (per-pt block of swing_steps may take
            seconds on DFT path, so granularity is one swing-cycle).

    Returns:
        Extended list of OptPoints including all swing variations.
    """
    swing_pts: List[OptPoint] = []
    for pt in top_points:
        if cancelled and cancelled():
            break
        if pt.max_swing <= 0:
            swing_pts.append(pt)
            continue
        sw_min = pt.max_swing * MIN_SWING_FRACTION
        steps = max(constraints.swing_steps, 2)
        for i_sw in range(steps):
            frac = sw_min + (pt.max_swing - sw_min) * i_sw / (steps - 1)
            new_pt = eval_fn(pt.ub, pt.ug2, pt.ug1, pt.ra, frac)
            if new_pt is not None:
                swing_pts.append(new_pt)
    return swing_pts


def _compute_pareto_front(points: List[OptPoint]) -> List[OptPoint]:
    """Extract Pareto front: min THD vs max Pout.

    A point is Pareto-optimal if no other point has both
    lower THD and higher Pout.
    """
    if not points:
        return []

    # Sort by THD ascending
    sorted_pts = sorted(points, key=lambda p: p.thd)
    front: List[OptPoint] = []
    max_pout_seen = -1.0

    for pt in sorted_pts:
        if pt.pout_mw > max_pout_seen + PARETO_DOMINANCE_EPS:
            front.append(pt)
            max_pout_seen = pt.pout_mw

    return front
