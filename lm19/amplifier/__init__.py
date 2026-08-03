"""Amplifier design analysis engine — package facade.

Pure calculation module — no Qt dependencies.

Core formulas follow "Radiotron Designer's Handbook, 4th ed.",
see SOURCES_INDEX.md.

Provides load line models, distortion analysis (5-point, Chebyshev,
DFT, IMD), parameter sweeps, bias optimization, headroom estimation,
stage parameters (gain, Zout), and data source management.

Submodules:
  - ``constants``     — DIST_ERR_*, MIN_*, Q_WINDOW_*, Chebyshev/Pa
  - ``loadlines``     — LoadLine Protocol + 4 concrete classes
  - ``distortion``    — intersections, 5-point/Chebyshev/DFT, PP variants
  - ``stage_params``  — gm/ra estimators, gain/Zout/DF, UL wrapper, NFB
  - ``sweeps``        — sweep_*, optimize_bias, headroom, Pa, Pg2, Ig2
  - ``presets``       — AMPLIFIER_PRESETS

This ``__init__`` re-exports the public API so existing
``from lm19.amplifier import …`` imports keep working without changes.
"""

from __future__ import annotations

# Constants — error codes and tunable thresholds
from lm19.amplifier.constants import (
    BALANCE_SWING_NEAR_ZERO,
    BIAS_MATCH_TOLERANCE_V,
    BISECT_CONVERGENCE_V,
    CHEBYSHEV_BOUNDARY,
    CHEBYSHEV_OVERFIT_RATIO,
    CIRCUIT_CF,
    CIRCUIT_PP,
    CIRCUIT_SE,
    CIRCUIT_SE_XFMR,
    CIRCUITS,
    CLASS_A_RATIO,
    CLASS_B_RATIO,
    CUTOFF_IA_MA,
    DB_MULTIPLIER,
    DIST_ERR_BIAS_AT_EDGE,
    DIST_ERR_BIAS_OUTSIDE,
    DIST_ERR_FEW_INTERSECTIONS,
    DIST_ERR_MANUAL_SWING_CLIPPED,
    DIST_ERR_MANUAL_SWING_SMALL,
    DIST_ERR_NO_SIGNAL,
    DIST_ERR_PP_NO_COMPOSITE,
    DIST_ERR_PP_NO_SWING,
    DIST_ERR_PP_RA_INVALID,
    DIST_ERR_SPARSE_DATA,
    DIST_ERR_UNKNOWN,
    FIXED_POINT_CONVERGENCE_MA,
    HD_METHOD_5POINT,
    HD_METHOD_AUTO,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_CHEBYSHEV_MODEL_PP,
    HD_METHOD_CHEBYSHEV_PP,
    HD_METHOD_DFT,
    HD_METHOD_DFT_PP,
    HD_METHODS,
    IK_NEAR_ZERO_MA,
    MIN_B1_MA,
    MIN_CHEBYSHEV_HARMONIC,
    MIN_CURVES_IN_SWING,
    MIN_IA_SWING_MA,
    MIN_POUT_MW,
    MIN_SWEEP_SWING_V,
    MIN_SWING_V,
    MIN_UA_SWING_V,
    MODEL_GM_DELTA_V,
    MODEL_RA_DELTA_V,
    N_PA_SAMPLES,
    OUTLIER_MAD_FACTOR,
    Q_WINDOW_UA_V,
    Q_WINDOW_UG1_V,
    Q_WINDOW_UG2_V,
    RA_DC_NEAR_ZERO_KOHM,
    SLOPE_NEAR_ZERO,
    SRK_DIVERGENCE_THRESHOLD_PCT,
    SWEEP_BIAS_MARGIN,
    UG1_MATCH_TOLERANCE_V,
)

# Load lines
from lm19.amplifier.loadlines import (
    pp_working_line_ia,
    working_line_polyline,
    CathodeFollowerLoadLine,
    LoadLine,
    PushPullLoadLine,
    ResistiveLoadLine,
    TransformerLoadLine,
    _linear_endpoints,
    _linear_ia_at_ua,
)

# Distortion / intersections
from lm19.amplifier.distortion import (
    _build_transfer_curve,
    _find_dc_q_point,
    _find_model_dc_q_point,
    _interp_transfer,
    build_pp_transfer,
    composite_characteristic,
    compute_distortion,
    compute_distortion_chebyshev,
    compute_distortion_chebyshev_pp,
    compute_distortion_chebyshev_pp_model,
    compute_distortion_dft,
    compute_distortion_dft_pp,
    compute_imd,
    diagnose_distortion,
    window_b1_probe,
    diagnose_pp_distortion,
    find_intersections,
    fold_pp_composite,
    ug2_filter_matches_any,
    find_intersections_model,
    interp_intersection,
    pp_distortion,
    pp_joint_trajectory,
)

# Stage params + UL wrapper
from lm19.amplifier.stage_params import (
    UltralinearModelWrapper,
    _compare_srk,
    _extract_ra,
    _numerical_gm_ra,
    compute_cf_stage_params,
    compute_nfb_effect,
    compute_stage_params,
    model_gm_ra,
    ul_screen_voltage,
)

# Sweeps + headroom + Pa/Pg2/Ig2
from lm19.amplifier.sweeps import (
    _compute_hd,
    compute_headroom,
    compute_pa_avg,
    compute_pg2,
    estimate_ig2_at_q,
    optimize_bias,
    sweep_amplitude,
    sweep_bias,
    sweep_pp_amplitude,
    sweep_ra,
    sweep_ra_pp,
)

# Presets
from lm19.amplifier.presets import (
    AMPLIFIER_PRESETS,
    AmplifierPreset,
)

# Data Source Selection (re-exported from analysis.py)
from lm19.analysis import (
    get_available_series,
    select_analysis_points,
)


__all__ = [
    # Distortion error codes
    "DIST_ERR_BIAS_AT_EDGE",
    "DIST_ERR_BIAS_OUTSIDE",
    "DIST_ERR_FEW_INTERSECTIONS",
    "DIST_ERR_MANUAL_SWING_CLIPPED",
    "DIST_ERR_MANUAL_SWING_SMALL",
    "DIST_ERR_NO_SIGNAL",
    "DIST_ERR_PP_NO_COMPOSITE",
    "DIST_ERR_PP_NO_SWING",
    "DIST_ERR_PP_RA_INVALID",
    "DIST_ERR_SPARSE_DATA",
    "DIST_ERR_UNKNOWN",
    # Load lines
    "CathodeFollowerLoadLine",
    "LoadLine",
    "PushPullLoadLine",
    "ResistiveLoadLine",
    "TransformerLoadLine",
    # Distortion
    "build_pp_transfer",
    "composite_characteristic",
    "compute_distortion",
    "compute_distortion_chebyshev",
    "compute_distortion_chebyshev_pp",
    "compute_distortion_chebyshev_pp_model",
    "compute_distortion_dft",
    "compute_distortion_dft_pp",
    "compute_imd",
    "diagnose_distortion",
    "window_b1_probe",
    "diagnose_pp_distortion",
    "find_intersections",
    "find_intersections_model",
    "fold_pp_composite",
    "interp_intersection",
    "pp_distortion",
    "pp_joint_trajectory",
    # Stage params
    "UltralinearModelWrapper",
    "compute_cf_stage_params",
    "compute_nfb_effect",
    "compute_stage_params",
    "model_gm_ra",
    "ul_screen_voltage",
    # Sweeps + helpers
    "compute_headroom",
    "compute_pa_avg",
    "compute_pg2",
    "estimate_ig2_at_q",
    "optimize_bias",
    "sweep_amplitude",
    "sweep_bias",
    "sweep_pp_amplitude",
    "sweep_ra",
    # Presets
    "AMPLIFIER_PRESETS",
    "AmplifierPreset",
    # Data source selection (re-export)
    "get_available_series",
    "select_analysis_points",
]
