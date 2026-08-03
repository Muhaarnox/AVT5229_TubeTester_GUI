"""Amplifier analysis engine — pure computation, no Qt dependencies.

Extracts all analysis logic from AmplifierTab._do_update_analysis()
into a testable, UI-independent class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from lm19 import amplifier
from lm19.amplifier.distortion import (
    _find_dc_q_point,
    _find_model_dc_q_point,
)
from lm19.amplifier.loadlines import _linear_endpoints
from lm19.amplifier import (
    working_line_polyline,
    CathodeFollowerLoadLine,
    LoadLine,
    PushPullLoadLine,
    ResistiveLoadLine,
    TransformerLoadLine,
    composite_characteristic,
    compute_distortion,
    compute_distortion_chebyshev,
    compute_distortion_chebyshev_pp,
    compute_distortion_chebyshev_pp_model,
    compute_distortion_dft,
    compute_distortion_dft_pp,
    compute_headroom,
    compute_imd,
    compute_nfb_effect,
    compute_pa_avg,
    compute_pg2,
    compute_stage_params,
    diagnose_distortion,
    diagnose_pp_distortion,
    estimate_ig2_at_q,
    find_intersections,
    find_intersections_model,
    interp_intersection,
    pp_distortion,
    pp_joint_trajectory,
    window_b1_probe,
    UltralinearModelWrapper,
    select_analysis_points,
    sweep_amplitude,
    sweep_pp_amplitude,
    sweep_ra,
)
from lm19.constants import (
    DEFAULT_UG2_V,
    MODEL_UA_MAX_DEFAULT_V,
    MODEL_UA_MIN_V,
    UG2_ZONE_TOLERANCE,
)

if TYPE_CHECKING:
    from lm19.tube_model_base import TubeModelProtocol
from lm19.tube_model_base import model_ia_array
from lm19.amplifier.constants import (
    CIRCUIT_CF,
    CIRCUIT_PP,
    CIRCUIT_SE,
    CIRCUIT_SE_XFMR,
    DIST_ERR_BIAS_AT_EDGE,
    DIST_ERR_BIAS_OUTSIDE,
    DIST_ERR_NO_SIGNAL,
    HD_METHOD_5POINT,
    HD_METHOD_AUTO,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_DFT,
)
from lm19.constants import (
    TOPOLOGY_TRIODE,
)

log = logging.getLogger(__name__)

# ── amp_engine local constants ────────────────────────────────────────
MIN_HALF_SWING_V = 0.1         # below this, treat half_swing as "auto"
DEFAULT_HALF_SWING_V = 3.0     # fallback half_swing for DFT method
DEFAULT_RA_DC_KOHM = 0.05      # ra_dc for non-transformer circuits
UG1_ROUND_NDIGITS = 1          # rounding precision for ug1 grouping
from lm19.constants import (  # noqa: F401  (re-export, canonical path)
    SOURCE_MEASUREMENTS as SOURCE_MEASUREMENTS,
)
# -- Engine warning-code registries (contract vocabularies) ----------
# SourceResult.warnings entries are {"code": WARN_*}; the UI maps them
# to amp.warn_<code>. Registry <-> locales guarded by a bijection pin.
WARN_MODEL_FALLBACK = "model_fallback"
WARN_UG2_FILTER_EMPTY = "ug2_filter_empty"
WARN_DFT_NO_MODEL = "dft_no_model"
WARN_DFT_NEWTON_NOT_CONVERGED = "dft_newton_not_converged"
WARN_UL_TAP_IGNORED_BY_METHOD = "ul_tap_ignored_by_method"
WARN_PP_B_EDGE_EXTRAPOLATED = "pp_b_edge_extrapolated"
WARN_PA_AVG_NOT_CONVERGED = "pa_avg_not_converged"
ENGINE_WARNING_CODES = frozenset({
    WARN_MODEL_FALLBACK, WARN_UG2_FILTER_EMPTY, WARN_DFT_NO_MODEL,
    WARN_DFT_NEWTON_NOT_CONVERGED, WARN_UL_TAP_IGNORED_BY_METHOD,
    WARN_PP_B_EDGE_EXTRAPOLATED, WARN_PA_AVG_NOT_CONVERGED,
})
# WorkingLineView.error / .note and AnalysisResult.error codes.
VIEW_ERR_NO_DATA = "no_data"
VIEW_ERR_NEEDS_UG2 = "needs_ug2"
VIEW_NOTE_FIXED_UG2 = "fixed_ug2"
# HD_METHOD_* / CIRCUIT_* live in lm19/amplifier/constants.py (the
# single vocabulary registry); imported at the top of this module and
# still reachable as lm19.amp_engine.HD_METHOD_* for existing
# importers (working_line, tests).
# Ua points per model UL-family curve; one model_ia_array call renders
# the whole family, so the cost is milliseconds.
MODEL_FAMILY_POINTS = 100
CHEBYSHEV_MODEL_POINTS = 30    # dense Ug1 grid for Chebyshev with model
MIN_VALID_UG2_V = 5.0          # below this Ug2 is treated as invalid for pentode


class _NeedsUg2(Exception):
    """Engine signals UI that pentode analysis needs an explicit Ug2.

    Raised by ``_resolve_intersections`` when ``model.topology != TOPOLOGY_TRIODE``
    and neither ``params.ug2_filter`` nor the measurement points provide a
    physically-valid screen voltage (median > MIN_VALID_UG2_V). The caller
    catches and surfaces via ``AnalysisResult.error = "needs_ug2"`` so the
    UI can prompt the user with a suggested default.
    """

    def __init__(self, suggested: float) -> None:
        super().__init__("Pentode model needs explicit Ug2")
        self.suggested = suggested


# ═══════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AmpParams:
    """All parameters needed for a single analysis run."""

    ub: float = 250.0
    ra: float = 5.0
    ug1_bias: float = -7.0
    half_swing: Optional[float] = None
    circuit: str = CIRCUIT_SE  # "se" | "se_xfmr" | "cf" | "pp"
    pa_max: float = 12.5

    # Data source selection
    sources: List[str] = field(default_factory=lambda: [SOURCE_MEASUREMENTS])
    hd_method: str = HD_METHOD_AUTO
    series_id: Optional[int] = None

    # Ug2 filter
    ug2_filter: Optional[float] = None
    # Lamp's nominal Ug2 from config (LampConfig.ug2). Used as the
    # "suggested" value when the engine asks the user to specify Ug2 for
    # a pentode model (see _resolve_intersections validation). UI sets
    # this from the currently selected lamp before calling analyze().
    lamp_ug2_default: Optional[float] = None

    # NFB
    nfb_db: Optional[float] = None  # None = disabled

    # Ultralinear (PP pentode only)
    ul_tap: Optional[float] = None  # None = disabled; 0.0–1.0 tap fraction

    # Circuit-specific
    ra_dc: float = 0.05  # SE Transformer: DC winding resistance (kΩ)
    cf_rk: float = 10.0  # CF: cathode resistance (kΩ)
    cf_rl: float = 10.0  # CF: load resistance (kΩ)
    pp_raa: float = 8.0  # PP: Ra anode-to-anode (kΩ)
    pp_ra_dc: float = 0.1  # PP: half-primary DC winding resistance (kΩ)
    pp_matched: bool = True
    pp_tube_b_sid: Optional[int] = None  # PP: series_id for second tube

    # Display options
    show_hd45: bool = False  # show HD4/HD5 on plots
    show_gzp: bool = False   # show Gain/Zout/Pa on Ra plot

    # Sweep config
    amp_steps: int = 40
    ra_steps: int = 60
    ra_min_factor: float = 0.2
    ra_max_factor: float = 5.0
    ra_min_abs: float = 0.5
    ra_max_abs: float = 100.0


@dataclass
class SourceResult:
    """Analysis results for a single data source."""

    dist: Optional[Dict] = None
    dist_error: Optional[str] = None  # DIST_ERR_* code if dist is None
    # Diagnostic payload for dist_error (e.g. the measured Ug1 span for
    # the bias_outside/at_edge codes) — the panel message must tell the
    # user WHAT range is valid, not just that theirs is not.
    dist_error_params: Optional[Dict] = None
    imd: Optional[Dict] = None
    headroom: Optional[Dict] = None
    stage: Optional[Dict] = None
    sweep_amp: List[Dict] = field(default_factory=list)
    sweep_ra: List[Dict] = field(default_factory=list)
    pg2_mw: Optional[float] = None
    pa_avg: Optional[Dict] = None  # model-based average Pa over signal cycle
    # ML-094: True when a MODEL source was requested but the model was not
    # found (refit/removed) and the analysis silently ran on raw
    # measurements instead — the UI must not label such results as model-based.
    model_fallback: bool = False
    # Degradation signals for the UI: list of {"code": str, **params} dicts.
    # The report renders them as a ⚠ block (amp.warn_<code> i18n keys) and
    # the main-window status-bar indicator aggregates them per run.
    warnings: List[Dict] = field(default_factory=list)
    nfb: Optional[Dict] = None
    method_used: str = HD_METHOD_5POINT


@dataclass
class WorkingLineView:
    """Light-weight working-line bundle for the 2D plot.

    Built by the SAME engine internals (_make_load_line /
    _get_intersections / _dist_*_routed) both in full ``analyze()``
    (``AnalysisResult.working_line``) and in the live tick
    (``compute_working_line``): method routing and fallbacks cannot
    diverge between the plot and the panel.

    ``method_used`` is the HD method actually applied (every displayed
    number must carry its method label). ``note`` — e.g. ``fixed_ug2``:
    UL tap set but no model, so intersections use the fixed measured
    Ug2. ``error`` — ``no_data`` / ``needs_ug2`` (the live path opens
    no dialogs — the controller renders a marker instead).
    """

    circuit: str = CIRCUIT_SE
    load_line: Optional[LoadLine] = None
    # Line vertices (working_line_polyline); empty = draw nothing.
    polyline: List[Tuple[float, float]] = field(default_factory=list)
    # DC line (xfmr/PP) — dashed segment.
    dc_polyline: List[Tuple[float, float]] = field(default_factory=list)
    q_ua: Optional[float] = None
    q_ia: Optional[float] = None
    intersections: List[Dict] = field(default_factory=list)
    hd: Optional[Dict] = None
    hd_error: Optional[str] = None
    method_used: str = HD_METHOD_5POINT
    note: str = ""
    error: Optional[str] = None
    # Model UL family (ug1, ua[], ia[]): the curves the tube actually
    # follows when tap>0 (Ug2_eff = f(Ua)); drawn dashed. Empty when
    # tap=0 or without a model (then note='fixed_ug2').
    model_family: List[Tuple[float, np.ndarray, np.ndarray]] = \
        field(default_factory=list)
    # PP composite (data-path composite_characteristic at Ua=Ub — the
    # same curve pp_distortion/Chebyshev analyse): points
    # {ug1, ia_a, ia_b, ia_composite}, mirrored around pp_bias. Drawn
    # folded on Transfer (fold_pp_composite). Empty for non-PP circuits.
    pp_composite: List[Dict] = field(default_factory=list)
    pp_bias: float = 0.0
    # Method-independent Q/swing geometry on the working line.
    # (ug1_0/ua_0/ia_0 + pt_neg/pt_pos/pt_low_half/pt_high_half).
    # Only the 5-point dict used to carry this geometry; Chebyshev/DFT
    # silently lost the swing markers on every plot — yet this is pure
    # intersection interpolation, so the HD method must not affect it.
    swing_geometry: Dict = field(default_factory=dict)
    # PP + model: tube A anode trajectory from the joint solve (kink
    # Zaa/2 -> Zaa/4). Non-empty means polyline/intersections/
    # swing_geometry are ALREADY joint-based (single source for every
    # plot); empty means the Zaa/4 display line is in use (no model /
    # mismatched pair / diverged solve) with a note in the UI.
    pp_trajectory: List[Dict] = field(default_factory=list)
    pp_kink: Optional[Dict] = None


@dataclass
class AnalysisResult:
    """Complete analysis result across all sources."""

    per_source: Dict[str, SourceResult] = field(default_factory=dict)
    pp_dist: Optional[Dict] = None  # PP-specific composite result
    pp_dist_error: Optional[str] = None  # DIST_ERR_PP_* code if pp_dist is None
    load_line: Optional[LoadLine] = None
    error: Optional[str] = None
    circuit: str = CIRCUIT_SE
    params: Optional[AmpParams] = None
    # Set when ``error == "needs_ug2"``: a value the UI dialog should
    # pre-fill (lamp.ug2 from config if available, else DEFAULT_UG2_V).
    # The user is then prompted to confirm/edit; the engine never silently
    # picks a Ug2 for a pentode model when measurements lack one.
    suggested_ug2: Optional[float] = None
    # Working-line bundle for 2D (same math as the panel).
    working_line: Optional[WorkingLineView] = None


def resolve_hd_method(method: str, source_name: str) -> str:
    """Resolve 'auto' to a concrete method. SINGLE routing point:
    engine.analyze, compute_working_line (live layer) and any future
    consumer must call this — divergent rules are forbidden."""
    if method == HD_METHOD_AUTO:
        return (HD_METHOD_DFT if source_name != SOURCE_MEASUREMENTS
                else HD_METHOD_CHEBYSHEV)
    return method


# ═══════════════════════════════════════════════════════════════════════════
#  Engine
# ═══════════════════════════════════════════════════════════════════════════


class AmplifierEngine:
    """Pure computation engine for amplifier analysis.

    No Qt dependencies. Receives data via set_data(), computes via analyze().
    """

    def __init__(self) -> None:
        self._all_points: List[Dict] = []
        # ML-094: source names whose model was requested but not found
        # (analysis silently ran on measurements) — per-run, see analyze().
        self._model_fallback_sources: set = set()
        self._series_labels: Dict[int, str] = {}
        self._srk: Optional[Dict] = None
        self._is_triode: bool = True
        self._series_models: Dict[int, "TubeModelProtocol"] = {}

    # ── Data management ───────────────────────────────────────────

    def set_data(
        self,
        points: List[Dict],
        series_labels: Optional[Dict[int, str]] = None,
        srk: Optional[Dict] = None,
        is_triode: bool = True,
        series_models: Optional[Dict] = None,
    ) -> None:
        """Update measurement data and fitted models."""
        self._all_points = points
        self._series_labels = series_labels or {}
        self._srk = srk
        self._is_triode = is_triode
        self._series_models = series_models or {}

    @property
    def has_data(self) -> bool:
        return len(self._all_points) > 0

    def available_models(self) -> Dict[str, str]:
        """Return {source_key: display_label} for fitted models."""
        result = {}
        for sid, model in self._series_models.items():
            key = model.model_type if hasattr(model, "model_type") else f"model_{sid}"
            label = model.name if hasattr(model, "name") else key
            result[key] = label
        return result

    # ── Main analysis entry point ─────────────────────────────────

    def analyze(self, params: AmpParams) -> AnalysisResult:
        """Run complete analysis. Pure computation, no UI."""
        self._model_fallback_sources.clear()
        if not self._all_points:
            return AnalysisResult(error=VIEW_ERR_NO_DATA, circuit=params.circuit, params=params)

        points = select_analysis_points(
            self._all_points, series_id=params.series_id,
        )
        if not points:
            return AnalysisResult(error=VIEW_ERR_NO_DATA, circuit=params.circuit, params=params)

        # PP has its own pipeline
        try:
            if params.circuit == CIRCUIT_PP:
                return self._analyze_pp(params, points)
            return self._analyze_se_cf(params, points)
        except _NeedsUg2 as e:
            # Pentode model + missing/invalid Ug2 in measurements. Don't
            # silently fall back — surface to UI so user can confirm a
            # value (suggested from lamp config, else DEFAULT_UG2_V).
            return AnalysisResult(
                error=VIEW_ERR_NEEDS_UG2,
                circuit=params.circuit,
                params=params,
                suggested_ug2=e.suggested,
            )

    # ── SE / SE Transformer / CF pipeline ─────────────────────────

    def _analyze_se_cf(self, params: AmpParams, points: List[Dict]) -> AnalysisResult:
        """Analysis for SE, SE Transformer, and Cathode Follower."""
        ll = self._make_load_line(params)
        swing = params.half_swing if params.half_swing and params.half_swing > MIN_HALF_SWING_V else None

        result = AnalysisResult(
            load_line=ll,
            circuit=params.circuit,
            params=params,
        )

        first: Optional[Tuple] = None
        for source_name in params.sources:
            sr, isects, model, ug2_val = self._analyze_single_source(
                source_name, params, points, ll, swing,
            )
            result.per_source[source_name] = sr
            if first is None:
                first = (isects, model, ug2_val, sr)

        if first is not None:
            isects, model, ug2_val, sr = first
            result.working_line = self._build_working_line_view(
                params, points, ll, isects, model, ug2_val,
                sr.dist, sr.dist_error, sr.method_used,
            )
        return result

    def _analyze_single_source(
        self,
        source_name: str,
        params: AmpParams,
        points: List[Dict],
        ll: LoadLine,
        swing: Optional[float],
    ) -> Tuple[SourceResult, List[Dict], Optional["TubeModelProtocol"], float]:
        """Compute all metrics for one data source.

        Returns ``(sr, isects, model, ug2_val)`` — the caller reuses the
        triple to build ``AnalysisResult.working_line`` from the SAME
        data the panel numbers came from.
        """
        sr = SourceResult()

        # Get intersections
        isects, model, ug2_val = self._get_intersections(
            source_name, params, points, ll,
        )
        sr.model_fallback = source_name in self._model_fallback_sources
        if sr.model_fallback:
            sr.warnings.append({"code": WARN_MODEL_FALLBACK,
                                "source": source_name})

        if len(isects) < 3:
            return sr, isects, model, ug2_val

        # Distortion — single routing (shared with the live layer).
        sr.dist, sr.dist_error, method, sr.method_used = \
            self._dist_se_routed(
                source_name, params, points, ll, isects, model, ug2_val,
                swing, sr.warnings,
            )
        if (sr.dist_error in (DIST_ERR_BIAS_OUTSIDE, DIST_ERR_BIAS_AT_EDGE)
                and isects):
            # same span diagnose_distortion measured — the intersections
            # of the CURRENT Ug2 slice, not the whole plot
            ug1s = [p["ug1"] for p in isects]
            sr.dist_error_params = {"lo": min(ug1s), "hi": max(ug1s),
                                    "bias": params.ug1_bias}
        elif sr.dist_error == DIST_ERR_NO_SIGNAL and isects:
            probe = window_b1_probe(isects, ug1_bias=params.ug1_bias,
                                    half_swing=swing)
            if probe is not None:
                sr.dist_error_params = {
                    "imin": probe["i_min"], "imax": probe["i_max"],
                    # pre-formatted: a 0.003 mA fundamental would show
                    # as "0.0" through the generic .1f formatting
                    "b1": f"{probe['b1']:.3f}",
                }

        # IMD
        sr.imd = compute_imd(isects, ug1_bias=params.ug1_bias, half_swing=swing)

        # Headroom (with grid current quantification when Dempwolf params available)
        gc_params = self._get_grid_current_params(model)
        sr.headroom = compute_headroom(
            isects, params.ug1_bias, pa_max=params.pa_max, load_line=ll,
            grid_current_params=gc_params,
        )

        # Stage params (model has highest priority for gm/ra)
        filtered_pts = self._filter_ug2(points, params.ug2_filter)
        if params.ug2_filter is not None and points and not filtered_pts:
            # Same condition distortion._apply_ug2_filter falls back on:
            # every downstream consumer now mixes Ug2 levels.
            sr.warnings.append({"code": WARN_UG2_FILTER_EMPTY,
                                "ug2": params.ug2_filter})
        sr.stage = compute_stage_params(
            isects, ll, ug1_bias=params.ug1_bias,
            srk=self._srk, points=filtered_pts,
            model=model, model_ug2=ug2_val,
        )

        # Pg2
        if not self._is_triode and params.ug2_filter is not None and sr.dist is not None:
            ig2_q = estimate_ig2_at_q(
                filtered_pts, sr.dist["ug1_0"], sr.dist["ua_0"], params.ug2_filter,
            )
            if ig2_q > 0:
                sr.pg2_mw = compute_pg2(params.ug2_filter, ig2_q)

        # Average Pa under signal (model-based, accurate for all classes)
        if model is not None and sr.dist and swing:
            if not sr.dist.get("insufficient_signal"):
                sr.pa_avg = compute_pa_avg(
                    model, ll,
                    ug1_bias=params.ug1_bias, half_swing=swing,
                    ug2=ug2_val, ub=params.ub,
                )
                self._append_pa_avg_warning(sr)

        # NFB
        if params.nfb_db is not None and params.nfb_db > 0:
            if sr.dist and sr.stage and sr.stage.get("gain", 0) > 0:
                if not sr.dist.get("insufficient_signal"):
                    sr.nfb = compute_nfb_effect(
                        gain_open=sr.stage["gain"],
                        zout_open=sr.stage["zout"],
                        thd_open=sr.dist["thd"],
                        nfb_db=params.nfb_db,
                    )

        # Sweeps
        model_ug2 = ug2_val if model is not None else 0.0
        sr.sweep_amp = sweep_amplitude(
            points, ll, ug1_bias=params.ug1_bias,
            ug2_filter=params.ug2_filter, steps=params.amp_steps,
            model=model, model_ug2=model_ug2,
            hd_method=method,
        )

        if params.circuit == CIRCUIT_CF:
            sr.sweep_ra = []
        else:
            ra = params.ra
            sr.sweep_ra = sweep_ra(
                points, params.ub,
                ra_min=max(params.ra_min_abs, ra * params.ra_min_factor),
                ra_max=min(params.ra_max_abs, ra * params.ra_max_factor),
                ug1_bias=params.ug1_bias,
                half_swing=swing,
                ug2_filter=params.ug2_filter,
                steps=params.ra_steps,
                model=model, model_ug2=model_ug2,
                transformer=(params.circuit == CIRCUIT_SE_XFMR),
                ra_dc=params.ra_dc if params.circuit == CIRCUIT_SE_XFMR else DEFAULT_RA_DC_KOHM,
                hd_method=method,
            )

        return sr, isects, model, ug2_val

    def _dist_se_routed(
        self,
        source_name: str,
        params: AmpParams,
        points: List[Dict],
        ll: LoadLine,
        isects: List[Dict],
        model: Optional["TubeModelProtocol"],
        ug2_val: float,
        swing: Optional[float],
        warnings_out: List[Dict],
    ) -> Tuple[Optional[Dict], Optional[str], str, str]:
        """SE/CF/xfmr HD routing — SINGLE for analyze() and the live
        layer (divergent rules are forbidden).

        Returns ``(dist, dist_error, method_resolved, method_used)``:
        ``method_resolved`` feeds the sweeps (DFT stays DFT there — the
        sweep has its own fallback); ``method_used`` labels the result.
        """
        method = self._resolve_hd_method(params.hd_method, source_name)
        method_used = method
        if method == HD_METHOD_DFT and model is None:
            # sweeps._compute_hd falls back to 5-point below; make the
            # substitution visible next to the numbers it affects.
            warnings_out.append({"code": WARN_DFT_NO_MODEL})

        if method == HD_METHOD_CHEBYSHEV:
            # For model sources, generate dense intersections for Chebyshev
            cheb_isects = isects
            if model is not None and len(isects) >= 2:
                ug1_min = min(p["ug1"] for p in isects)
                ug1_max = max(p["ug1"] for p in isects)
                dense_ug1 = [
                    ug1_min + (ug1_max - ug1_min) * i / CHEBYSHEV_MODEL_POINTS
                    for i in range(CHEBYSHEV_MODEL_POINTS + 1)
                ]
                ua_max = max((p["ua"] for p in points), default=MODEL_UA_MAX_DEFAULT_V)
                cheb_isects = find_intersections_model(
                    model, ll, dense_ug1, ug2=ug2_val,
                    ua_range=(MODEL_UA_MIN_V, ua_max),
                    ug1_bias=params.ug1_bias,
                )
            dist = compute_distortion_chebyshev(
                cheb_isects, ug1_bias=params.ug1_bias, half_swing=swing, ub=params.ub,
            )
        elif method == HD_METHOD_DFT and model is not None:
            dist = compute_distortion_dft(
                model, ll,
                ug1_bias=params.ug1_bias, half_swing=swing or DEFAULT_HALF_SWING_V,
                ug2=ug2_val, ub=params.ub,
            )
        else:
            # 5-point (default / fallback)
            dist = compute_distortion(
                isects, ug1_bias=params.ug1_bias, half_swing=swing, ub=params.ub,
            )
            method_used = HD_METHOD_5POINT

        dist_error: Optional[str] = None
        if dist is None:
            dist_error = diagnose_distortion(
                isects, ug1_bias=params.ug1_bias, half_swing=swing,
            )
        else:
            n_nc = dist.get("n_not_converged", 0)
            if n_nc:
                warnings_out.append({
                    "code": WARN_DFT_NEWTON_NOT_CONVERGED, "n": n_nc,
                    "n_samples": dist.get("n_samples", 0),
                })
        return dist, dist_error, method, method_used

    # ── PP pipeline ───────────────────────────────────────────────

    def _analyze_pp(self, params: AmpParams, points: List[Dict]) -> AnalysisResult:
        """Push-Pull analysis."""
        ll = PushPullLoadLine(params.ub, params.pp_raa, ra_dc=params.pp_ra_dc)
        swing = params.half_swing if params.half_swing and params.half_swing > MIN_HALF_SWING_V else None

        result = AnalysisResult(
            load_line=ll,
            circuit=CIRCUIT_PP,
            params=params,
        )

        # Second tube data
        points_b = None
        if not params.pp_matched and params.pp_tube_b_sid is not None:
            points_b = select_analysis_points(
                self._all_points, series_id=params.pp_tube_b_sid,
            )
            if not points_b:
                result.error = "pp_no_tube_b"
                return result

        # Intersections first — the DFT route below needs the (possibly
        # UL-wrapped) model and ug2_val from the same resolution the SE
        # path uses.
        isects, model, ug2_val = self._get_intersections(
            SOURCE_MEASUREMENTS, params, points, ll,
        )

        # PP composite distortion — single routing (analyze + live).
        pp_warnings: List[Dict] = []
        result.pp_dist, result.pp_dist_error, pp_method_used = \
            self._dist_pp_routed(
                params, points, points_b, ll, model, ug2_val,
                swing, pp_warnings,
            )

        # Headroom (with grid current quantification)
        gc_params = self._get_grid_current_params(model)
        headroom = compute_headroom(
            isects, params.ug1_bias, pa_max=params.pa_max, load_line=ll,
            grid_current_params=gc_params,
        )

        # Stage params (with model priority for gm/ra)
        filtered_pts = self._filter_ug2(points, params.ug2_filter)
        stage = compute_stage_params(
            isects, ll, ug1_bias=params.ug1_bias,
            srk=self._srk, points=filtered_pts,
            model=model, model_ug2=ug2_val,
        )

        # Average Pa under signal (model-based)
        pa_avg = None
        if model is not None and swing:
            dist = result.pp_dist
            if dist and not dist.get("insufficient_signal"):
                pa_avg = compute_pa_avg(
                    model, ll,
                    ug1_bias=params.ug1_bias, half_swing=swing,
                    ug2=ug2_val, ub=params.ub,
                )

        # Sweep amplitude
        sweep_amp = sweep_pp_amplitude(
            points, ll, params.ug1_bias,
            points_b=points_b, ug2_filter=params.ug2_filter,
            steps=params.amp_steps,
        )

        # Store as single source result
        sr = SourceResult(
            headroom=headroom,
            stage=stage,
            pa_avg=pa_avg,
            sweep_amp=sweep_amp,
            method_used=pp_method_used,
        )
        sr.warnings.extend(pp_warnings)
        if params.ug2_filter is not None and points and not filtered_pts:
            sr.warnings.append({"code": WARN_UG2_FILTER_EMPTY,
                                "ug2": params.ug2_filter})
        if result.pp_dist is not None:
            n_nc = result.pp_dist.get("n_not_converged", 0)
            if n_nc:
                sr.warnings.append({
                    "code": WARN_DFT_NEWTON_NOT_CONVERGED, "n": n_nc,
                    "n_samples": result.pp_dist.get("n_samples", 0),
                })
        self._append_pa_avg_warning(sr)
        result.per_source[SOURCE_MEASUREMENTS] = sr

        result.working_line = self._build_working_line_view(
            params, points, ll, isects, model, ug2_val,
            result.pp_dist, result.pp_dist_error, pp_method_used,
            points_b=points_b,
        )
        return result

    # ── Helpers ────────────────────────────────────────────────────

    def _make_load_line(self, params: AmpParams) -> LoadLine:
        """Create LoadLine from params."""
        if params.circuit == CIRCUIT_CF:
            return CathodeFollowerLoadLine(params.ub, params.cf_rk, params.cf_rl)
        if params.circuit == CIRCUIT_SE_XFMR:
            return TransformerLoadLine(params.ub, ra_dc=params.ra_dc, ra_ac=params.ra)
        if params.circuit == CIRCUIT_PP:
            return PushPullLoadLine(params.ub, params.pp_raa, ra_dc=params.pp_ra_dc)
        return ResistiveLoadLine(params.ub, params.ra)

    def _get_intersections(
        self,
        source_name: str,
        params: AmpParams,
        points: List[Dict],
        ll: LoadLine,
    ) -> tuple[List[Dict], Optional["TubeModelProtocol"], float]:
        """Get intersections for a source. Returns (isects, model_or_none, ug2_val)."""
        model: Optional["TubeModelProtocol"] = None
        ug2_val = 0.0

        if source_name != SOURCE_MEASUREMENTS:
            # Find model by model_type
            for sid, m in self._series_models.items():
                mtype = m.model_type if hasattr(m, "model_type") else ""
                if mtype == source_name:
                    model = m
                    break
            if model is None:
                # ML-094: the requested model source is gone (refit under a
                # different type / removed) — the code below silently falls
                # through to the raw-measurements path while the result
                # keeps the model's name. Warn + flag for the UI.
                log.warning("Model source %r not found — falling back to "
                            "raw measurements", source_name)
                self._model_fallback_sources.add(source_name)

        if model is None and source_name == SOURCE_MEASUREMENTS:
            # Per-series model overrides the measurement-points path.
            if params.series_id is not None:
                model = self._series_models.get(params.series_id)

        if model is not None:
            ug1_values = sorted({round(p["ug1"], UG1_ROUND_NDIGITS) for p in points})
            if params.ug2_filter is not None:
                ug2_val = params.ug2_filter
            elif model.topology == TOPOLOGY_TRIODE:
                ug2_val = 0.0
            else:
                # Pentode/triode_connected: filter zeros (sensor failure or
                # missing data) before the median. Naive np.median including
                # zeros gives ug2=0 when every reading is zero → that
                # propagates to model.ia() and produces silent garbage
                # (pentode at screen=0 V → fully cut off, or UL wrapper
                # anchored at 0 → ug2_eff = tap×Ua nonsense).
                valid_ug2 = [p["ug2"] for p in points
                             if p.get("ug2", 0.0) > MIN_VALID_UG2_V]
                if valid_ug2:
                    ug2_val = float(np.median(valid_ug2))
                else:
                    # No valid Ug2 in measurements — bail out and let the
                    # UI ask the user. Suggested value: lamp.ug2 from
                    # config (caller sets params.lamp_ug2_default), else
                    # the global DEFAULT_UG2_V.
                    suggested = params.lamp_ug2_default
                    if suggested is None or suggested <= MIN_VALID_UG2_V:
                        suggested = DEFAULT_UG2_V
                    raise _NeedsUg2(suggested=suggested)

            # Validate ug2_filter when caller supplied one but it's zero
            # (e.g., UI populated combo from a corrupt scan).
            if (model.topology != TOPOLOGY_TRIODE
                    and ug2_val <= MIN_VALID_UG2_V):
                suggested = params.lamp_ug2_default
                if suggested is None or suggested <= MIN_VALID_UG2_V:
                    suggested = DEFAULT_UG2_V
                raise _NeedsUg2(suggested=suggested)

            # Ultralinear: wrap model so ia()/ig2() use dynamic Ug2(Ua).
            if (
                params.ul_tap is not None
                and params.ul_tap > 0.0
                and model.topology != TOPOLOGY_TRIODE
            ):
                model = UltralinearModelWrapper(model, ug2_nom=ug2_val, tap=params.ul_tap)

            ua_max = max((p["ua"] for p in points), default=MODEL_UA_MAX_DEFAULT_V)
            isects = find_intersections_model(
                model, ll, ug1_values, ug2=ug2_val,
                ua_range=(MODEL_UA_MIN_V, ua_max),
                ug1_bias=params.ug1_bias,
            )
        else:
            isects = find_intersections(
                points, ll, ug2_filter=params.ug2_filter,
                ug1_bias=params.ug1_bias,
            )

        return isects, model, ug2_val

    def _resolve_hd_method(self, method: str, source_name: str) -> str:
        """Resolve 'auto' to a concrete method (delegates to the
        module-level ``resolve_hd_method`` shared with the live layer)."""
        return resolve_hd_method(method, source_name)

    def _dist_pp_routed(
        self,
        params: AmpParams,
        points: List[Dict],
        points_b: Optional[List[Dict]],
        ll: "PushPullLoadLine",
        model: Optional["TubeModelProtocol"],
        ug2_val: float,
        swing: Optional[float],
        warnings_out: List[Dict],
    ) -> Tuple[Optional[Dict], Optional[str], str]:
        """PP HD routing (ML-050) — SINGLE for analyze() and live.

        Returns ``(pp_dist, pp_dist_error, method_used)``.
        """
        method = self._resolve_hd_method(params.hd_method, SOURCE_MEASUREMENTS)
        method_used = HD_METHOD_5POINT
        if method == HD_METHOD_DFT and (model is None or points_b is not None):
            # DFT needs a fitted model and (engine-side) a matched pair —
            # mirror the SE path: fall back visibly, never silently.
            warnings_out.append({"code": WARN_DFT_NO_MODEL})
            method = HD_METHOD_5POINT
        if method == HD_METHOD_DFT:
            pp_dist = compute_distortion_dft_pp(
                model, ll, params.ug1_bias,
                half_swing=swing or DEFAULT_HALF_SWING_V,
                ug2=ug2_val,
            )
            method_used = HD_METHOD_DFT
        elif method == HD_METHOD_CHEBYSHEV:
            if isinstance(model, UltralinearModelWrapper):
                # UL tap set: the data composite cannot see it — route to
                # the model variant (same joint solve as DFT, Chebyshev
                # extraction) so the panel numbers include the tap.
                pp_dist = compute_distortion_chebyshev_pp_model(
                    model, ll, params.ug1_bias,
                    half_swing=swing or DEFAULT_HALF_SWING_V,
                    ug2=ug2_val,
                )
            else:
                pp_dist = compute_distortion_chebyshev_pp(
                    points, ll, params.ug1_bias,
                    points_b=points_b, half_swing=swing,
                    ug2_filter=params.ug2_filter,
                )
            method_used = HD_METHOD_CHEBYSHEV
        else:
            pp_dist = pp_distortion(
                points, ll, params.ug1_bias,
                points_b=points_b, half_swing=swing,
                ug2_filter=params.ug2_filter,
            )
        if (params.ul_tap and params.ul_tap > 0.0
                and method_used == HD_METHOD_5POINT):
            # 5-point runs on the measured composite — a non-zero tap is
            # silently ignored by the THD math; say so (failure-visibility rule).
            warnings_out.append({"code": WARN_UL_TAP_IGNORED_BY_METHOD})
        if pp_dist and pp_dist.get("b_extrapolation_significant"):
            # ML-139 warn tier: B's data edge is far from cutoff — the
            # extrapolated tail materially shapes the composite. The
            # honest remedy is a deeper Ug1 scan of tube B.
            frac = float(pp_dist.get("b_edge_ia_fraction", 0.0))
            log.warning(
                "PP composite: tube B data edge is far from cutoff "
                "(edge Ia = %.0f%% of the analyzed signal amplitude) — "
                "extrapolated tail is significant; rescan tube B to "
                "deeper Ug1", 100.0 * frac)
            warnings_out.append({
                "code": WARN_PP_B_EDGE_EXTRAPOLATED,
                "edge_pct": f"{100.0 * frac:.0f}",
            })
        pp_dist_error: Optional[str] = None
        if pp_dist is None:
            pp_dist_error = diagnose_pp_distortion(
                points, ll, params.ug1_bias,
                points_b=points_b, half_swing=swing,
                ug2_filter=params.ug2_filter,
            )
        return pp_dist, pp_dist_error, method_used

    def _resolve_display_q(
        self,
        params: AmpParams,
        points: List[Dict],
        ll: LoadLine,
        model: Optional["TubeModelProtocol"],
        ug2_val: float,
    ) -> Optional[Tuple[float, float]]:
        """Q point for the polyline (xfmr/PP): data first, model as
        fallback. Resistive/CF lines need no Q (marker comes from hd)."""
        if isinstance(ll, TransformerLoadLine):
            ra_dc = ll.ra_dc
        elif isinstance(ll, PushPullLoadLine):
            ra_dc = ll.ra_dc
        else:
            return None
        q = _find_dc_q_point(
            points, ll.ub, ra_dc, params.ug1_bias,
            ug2_filter=params.ug2_filter,
        )
        if q is None and model is not None:
            q = _find_model_dc_q_point(
                model, ll.ub, ra_dc, params.ug1_bias, ug2_val,
                (MODEL_UA_MIN_V, MODEL_UA_MAX_DEFAULT_V),
            )
        return q

    @staticmethod
    def _swing_geometry(
        isects: List[Dict], ug1_bias: float,
        half_swing: Optional[float],
    ) -> Dict:
        """Method-independent Q/swing geometry on the working line.

        Pure intersection interpolation: Q at the bias, swing ends at
        bias +/- swing (pt_pos is the side toward grid zero), half-
        points at bias +/- swing/2. Consumers: 2D swing markers, heatmap
        Q/triangles, the Transfer swing region, slice dim-accent.
        """
        if len(isects) < 2:
            return {}
        geo: Dict = {}
        q = interp_intersection(isects, ug1_bias)
        if q is not None:
            geo.update(ug1_0=ug1_bias, ua_0=q["ua"], ia_0=q["ia"])
        if half_swing and half_swing > 0:
            pt_pos = interp_intersection(isects, ug1_bias + half_swing)
            pt_neg = interp_intersection(isects, ug1_bias - half_swing)
            if pt_pos and pt_neg:
                geo.update(pt_pos=pt_pos, pt_neg=pt_neg)
            pt_hh = interp_intersection(
                isects, ug1_bias + half_swing / 2.0)
            pt_lh = interp_intersection(
                isects, ug1_bias - half_swing / 2.0)
            if pt_hh and pt_lh:
                geo.update(pt_high_half=pt_hh, pt_low_half=pt_lh)
        return geo

    @staticmethod
    def _trajectory_at_ug1_grid(
        traj: List[Dict], isects: List[Dict],
    ) -> List[Dict]:
        """Intersection markers for joint mode: trajectory samples at
        MEASURED Ug1 values (display-intersection grid) inside the swing.

        Points outside the swing are dropped honestly: the tube never
        goes there — a marker on a non-existent stretch would lie.
        """
        if not traj:
            return []
        gs = np.array([p["ug1"] for p in traj])
        uas = np.array([p["ua"] for p in traj])
        ias = np.array([p["ia"] for p in traj])
        out: List[Dict] = []
        for g1 in sorted({round(p["ug1"], 2) for p in isects}):
            if gs[0] <= g1 <= gs[-1]:
                out.append({
                    "ug1": float(g1),
                    "ua": float(np.interp(g1, gs, uas)),
                    "ia": float(np.interp(g1, gs, ias)),
                })
        return out

    @staticmethod
    def _trajectory_geometry(traj: List[Dict], ug1_bias: float) -> Dict:
        """swing_geometry from exact trajectory ordinates.

        The pp_joint_trajectory grid is built so that indices 0, n//4,
        n//2, 3n//4, n-1 are exactly the drives -s, -s/2, 0, +s/2, +s
        (PP_TRAJECTORY_POINTS is odd and (n-1) % 4 == 0).
        """
        n = len(traj)
        q = traj[n // 2]
        return {
            "ug1_0": ug1_bias, "ua_0": q["ua"], "ia_0": q["ia"],
            "pt_neg": traj[0], "pt_pos": traj[-1],
            "pt_low_half": traj[n // 4],
            "pt_high_half": traj[3 * n // 4],
        }

    def _build_working_line_view(
        self,
        params: AmpParams,
        points: List[Dict],
        ll: LoadLine,
        isects: List[Dict],
        model: Optional["TubeModelProtocol"],
        ug2_val: float,
        hd: Optional[Dict],
        hd_error: Optional[str],
        method_used: str,
        points_b: Optional[List[Dict]] = None,
    ) -> WorkingLineView:
        """Assemble WorkingLineView from ALREADY computed analyze data.

        ``points_b`` is tube-B data for a PP mismatched pair: the
        composite must use the same pair as pp_dist (both call sites
        pass it; a lost argument silently yields a matched composite).
        """
        view = WorkingLineView(
            circuit=params.circuit,
            load_line=ll,
            intersections=isects,
            hd=hd,
            hd_error=hd_error,
            method_used=method_used,
        )
        if params.circuit == CIRCUIT_PP and points:
            # Data-path composite at Ua=Ub — exactly the curve that
            # pp_distortion/Chebyshev analyse (ua_ref=ll.ub, mirrored
            # around the bias); for the DFT method it stays the measured
            # reference (the model joint solve builds no such curve).
            view.pp_composite = composite_characteristic(
                points, points_b, ug1_bias=params.ug1_bias,
                ug2_filter=params.ug2_filter, ua_ref=ll.ub,
            )
            view.pp_bias = params.ug1_bias
        # Q/swing geometry — independent of the HD method (and of
        # whether HD computed at all): plot markers must survive an HD
        # failure. Manual panel swing wins; auto uses the actual hd value.
        swing_eff = (params.half_swing
                     if params.half_swing
                     and params.half_swing > MIN_HALF_SWING_V
                     else (hd or {}).get("half_swing"))
        view.swing_geometry = self._swing_geometry(
            isects, params.ug1_bias, swing_eff)
        q = self._resolve_display_q(params, points, ll, model, ug2_val)
        if q is not None:
            view.q_ua, view.q_ia = q
        view.polyline = working_line_polyline(ll, view.q_ua, view.q_ia)
        if isinstance(ll, TransformerLoadLine):
            p0, p1 = ll.endpoints_dc()
            view.dc_polyline = [p0, p1]
        elif isinstance(ll, PushPullLoadLine) and ll.ra_dc > 0:
            p0, p1 = _linear_endpoints(ll.ub, ll.ra_dc)
            view.dc_polyline = [p0, p1]
        # PP + model (either the model source, or a per-series fit of
        # measurements) -> line/markers/geometry come from the joint
        # pair solve REGARDLESS of the HD method. Single source for
        # every plot: polyline = the trajectory itself (kink Zaa/2 ->
        # Zaa/4), intersections = its samples at measured Ug1,
        # swing_geometry = its exact ordinates. A mismatched pair or a
        # diverged solve falls back to the display line (noted in UI).
        if (params.circuit == CIRCUIT_PP and model is not None
                and points_b is None and swing_eff
                and isinstance(ll, PushPullLoadLine)):
            traj = pp_joint_trajectory(
                model, ll, params.ug1_bias, swing_eff, ug2_val)
            if traj and len(traj["points"]) >= 5:
                traj_pts = traj["points"]
                view.pp_trajectory = traj_pts
                view.pp_kink = traj.get("kink")
                view.polyline = [(p["ua"], p["ia"]) for p in traj_pts]
                view.intersections = self._trajectory_at_ug1_grid(
                    traj_pts, isects)
                view.swing_geometry = self._trajectory_geometry(
                    traj_pts, params.ug1_bias)
        if (params.ul_tap and params.ul_tap > 0.0 and model is None
                and params.circuit != CIRCUIT_CF):
            # UL set but no model: intersections/Q use the fixed
            # measured Ug2 (method visibility — the UI shows a note).
            view.note = VIEW_NOTE_FIXED_UG2
        if (model is not None and params.ul_tap
                and params.ul_tap > 0.0 and view.intersections):
            # tap>0: the tube follows the UL family
            # (Ug2_eff = Ug2_nom*(1-tap) + Ua*tap) — computed with the
            # SAME wrapped object that produced the intersections (a pin
            # discriminates against using the raw fixed-Ug2 model).
            ug1s = sorted({round(p["ug1"], 2)
                           for p in view.intersections})
            ua_max = max((p["ua"] for p in points),
                         default=MODEL_UA_MAX_DEFAULT_V)
            ua_grid = np.linspace(MODEL_UA_MIN_V, ua_max,
                                  MODEL_FAMILY_POINTS)
            ia_grid = model_ia_array(
                model, ua_grid[None, :],
                np.asarray(ug1s, dtype=float)[:, None], ug2_val)
            view.model_family = [
                (g, ua_grid, np.asarray(ia_grid[i], dtype=float))
                for i, g in enumerate(ug1s)
            ]
        return view

    def compute_working_line(self, params: AmpParams) -> WorkingLineView:
        """Working-line bundle for the live tick: line + Q +
        intersections + HD by the selected method — using the SAME
        internals as full analyze() (single routing), but without
        sweeps/stage/headroom/Pa_avg. Opens no dialogs: needs_ug2 /
        no_data are returned as codes in ``view.error``."""
        view = WorkingLineView(circuit=params.circuit)
        if not self._all_points:
            view.error = VIEW_ERR_NO_DATA
            return view
        points = select_analysis_points(
            self._all_points, series_id=params.series_id,
        )
        if not points:
            view.error = VIEW_ERR_NO_DATA
            return view
        ll = self._make_load_line(params)
        source = params.sources[0] if params.sources else SOURCE_MEASUREMENTS
        try:
            isects, model, ug2_val = self._get_intersections(
                source, params, points, ll,
            )
        except _NeedsUg2:
            view.error = VIEW_ERR_NEEDS_UG2
            view.load_line = ll
            return view
        swing = (params.half_swing
                 if params.half_swing and params.half_swing > MIN_HALF_SWING_V
                 else None)
        scratch: List[Dict] = []
        points_b: Optional[List[Dict]] = None
        if params.circuit == CIRCUIT_PP:
            if not params.pp_matched and params.pp_tube_b_sid is not None:
                points_b = select_analysis_points(
                    self._all_points, series_id=params.pp_tube_b_sid,
                ) or None
            hd, hd_error, method_used = self._dist_pp_routed(
                params, points, points_b, ll, model, ug2_val,
                swing, scratch,
            )
        else:
            hd, hd_error, _method, method_used = self._dist_se_routed(
                source, params, points, ll, isects, model, ug2_val,
                swing, scratch,
            )
        return self._build_working_line_view(
            params, points, ll, isects, model, ug2_val,
            hd, hd_error, method_used, points_b=points_b,
        )

    @staticmethod
    def _append_pa_avg_warning(sr: SourceResult) -> None:
        """Surface non-converged Newton samples in Pa_avg (checked against
        Pa_max when picking the operating point)."""
        if sr.pa_avg is None:
            return
        n_nc = sr.pa_avg.get("n_not_converged", 0)
        if n_nc:
            sr.warnings.append({"code": WARN_PA_AVG_NOT_CONVERGED, "n": n_nc})

    def _filter_ug2(self, points: List[Dict], ug2_filter: Optional[float]) -> List[Dict]:
        """Filter points by Ug2 value."""
        if ug2_filter is None:
            return points
        return [
            p for p in points
            if abs(p.get("ug2", 0.0) - ug2_filter) <= UG2_ZONE_TOLERANCE
        ]

    @staticmethod
    def _get_grid_current_params(
        model: Optional["TubeModelProtocol"],
    ) -> Optional[Dict]:
        """Extract Dempwolf grid current params (Gg, xi, Cg) from model.

        Returns dict usable by compute_headroom(grid_current_params=...),
        or None if model doesn't have Dempwolf grid current parameters.
        """
        if model is None:
            return None
        # Dempwolf model stores params in .params attribute
        params = None
        if hasattr(model, "params") and hasattr(model.params, "Gg"):
            params = model.params
        elif hasattr(model, "_model"):
            # UltralinearModelWrapper — unwrap
            inner = model._model
            if hasattr(inner, "params") and hasattr(inner.params, "Gg"):
                params = inner.params
        if params is None:
            return None
        Gg = getattr(params, "Gg", 0.0)
        if Gg <= 0:
            return None
        return {
            "Gg": Gg,
            "xi": getattr(params, "xi", 1.3),
            "Cg": getattr(params, "Cg", 10.0),
        }
