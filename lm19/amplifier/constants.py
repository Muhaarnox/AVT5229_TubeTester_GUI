"""Amplifier-package local constants.

Distortion failure codes, signal/swing thresholds, Q-point windows,
Chebyshev fit settings, and related numeric tolerances. Consumed by
``distortion``, ``sweeps``, ``stage_params``.

Public constants (e.g. ``DIST_ERR_*``) are re-exported by
``lm19.amplifier`` so callers can import them from the package root.
"""

from __future__ import annotations


# ── Distortion-failure diagnostic codes ──────────────────────────────
# Returned by diagnose_distortion() / diagnose_pp_distortion() to explain
# why compute_distortion / pp_distortion returned None. Translatable via
# ``amp.dist_err_<code>`` keys.
DIST_ERR_FEW_INTERSECTIONS = "few_intersections"   # load line crossed < 3 curves
DIST_ERR_BIAS_OUTSIDE = "bias_outside_data"        # Ug1 bias far outside measured Ug1 range
DIST_ERR_BIAS_AT_EDGE = "bias_at_data_edge"        # bias too close to data edge for any swing
DIST_ERR_MANUAL_SWING_SMALL = "manual_swing_small" # user-set swing < MIN_SWING_V
DIST_ERR_MANUAL_SWING_CLIPPED = "manual_swing_clipped"  # swing clamped into Ia<0 region
DIST_ERR_NO_SIGNAL = "no_signal"                   # i_max ≈ i_min (curve flat over swing)
DIST_ERR_PP_NO_COMPOSITE = "pp_no_composite"       # composite_characteristic gave < 5 pts
DIST_ERR_PP_RA_INVALID = "pp_ra_invalid"           # ra_per_tube ≤ 0
DIST_ERR_PP_NO_SWING = "pp_no_swing"               # composite swing near zero
DIST_ERR_SPARSE_DATA = "sparse_data"               # < MIN_CURVES_IN_SWING actual Ug1 curves
                                                   # cover swing → 5-point unreliable (interp only)
DIST_ERR_UNKNOWN = "unknown"                       # all upfront checks pass but compute fails


# -- Contract vocabularies (no-magic-strings rule) --------------------
# Single source of truth for values crossing module boundaries
# (UI combos -> engine/optimizer -> export/reports). Production and
# tests must import the constants, not literals — AST ratchet in
# tests/test_conventions_guards.py::TestContractVocabularies.

# hd_method: harmonic calculation method. Maps 1:1 to the UI
# hd_method_combo; resolve_hd_method (amp_engine) expands AUTO.
HD_METHOD_AUTO = "auto"
HD_METHOD_DFT = "dft"
HD_METHOD_CHEBYSHEV = "chebyshev"
HD_METHOD_5POINT = "5point"
HD_METHODS = frozenset({
    HD_METHOD_AUTO, HD_METHOD_DFT, HD_METHOD_CHEBYSHEV, HD_METHOD_5POINT,
})
# PP variant labels in result["method"] / OptPoint.hd_method — an
# OUTPUT vocabulary (not inputs): which method ACTUALLY computed the
# composite.
HD_METHOD_CHEBYSHEV_PP = "chebyshev_pp"
HD_METHOD_CHEBYSHEV_MODEL_PP = "chebyshev_model_pp"  # UL model-Chebyshev
HD_METHOD_DFT_PP = "dft_pp"

# circuit: stage type (amplifier panel, optimizer, SPICE export,
# LTspice verification).
CIRCUIT_SE = "se"
CIRCUIT_SE_XFMR = "se_xfmr"
CIRCUIT_CF = "cf"
CIRCUIT_PP = "pp"
CIRCUITS = frozenset({CIRCUIT_SE, CIRCUIT_SE_XFMR, CIRCUIT_CF, CIRCUIT_PP})


# ── Signal / swing thresholds ────────────────────────────────────────
MIN_SWING_V = 0.1              # minimum half-swing to consider non-trivial
# 5-point HD analysis needs at least 3 actual measurement Ug1 curves to
# fall within the swing window [ug1_bias-half_swing, ug1_bias+half_swing].
# Fewer → all 5 sample points are linear interpolations between only 2
# data lines → b2,b3 ≈ 0 algebraically → fake-near-zero THD.
# Additionally at least one curve must be strictly inside the window
# (not at the edges) so that 5-point's intermediate samples (low_half,
# high_half) sit on real curvature rather than further interpolation.
MIN_CURVES_IN_SWING = 3
# Tolerance for counting a Ug1 curve as inside the swing window — relative to
# the window width with an absolute floor, so floating-point edges are robust.
EDGE_TOL_REL = 1e-6            # fraction of the swing window
EDGE_TOL_ABS = 1e-9           # absolute floor (V)
MIN_B1_MA = 0.01               # minimum fundamental amplitude (mA) for valid distortion
CUTOFF_IA_MA = 0.05            # Ia threshold below which a point is treated as cutoff
MIN_IA_SWING_MA = 0.1          # minimum Ia peak-to-peak for "sufficient signal"
MIN_UA_SWING_V = 1.0           # minimum Ua swing for "sufficient signal"
MIN_POUT_MW = 1.0              # minimum Pout to consider meaningful
RA_DC_NEAR_ZERO_KOHM = 0.01    # below this Ra_dc ≈ 0, Ua_q ≈ Ub
BISECT_CONVERGENCE_V = 0.01    # bisection convergence for Q-point search
UG1_MATCH_TOLERANCE_V = 0.01   # Ug1 matching tolerance for interp_intersection
BIAS_MATCH_TOLERANCE_V = 0.05  # Ug1 tolerance for bias point matching
SLOPE_NEAR_ZERO = 0.001        # slope threshold for ra / Zout division guard
FIXED_POINT_CONVERGENCE_MA = 0.01  # CF iteration convergence (mA)
IK_NEAR_ZERO_MA = 0.01         # cathode current threshold for Rk calculation
BALANCE_SWING_NEAR_ZERO = 0.001  # PP balance error: guard against zero swing
DB_MULTIPLIER = 20.0           # dB = 20 * log10(ratio)

# ── Class detection (A / AB / B) ─────────────────────────────────────
CLASS_A_RATIO = 0.05           # i_min/i_0 above this → class A
CLASS_B_RATIO = 0.005          # i_min/i_0 below this → class B

# ── Chebyshev / Q-point / sweep ──────────────────────────────────────
CHEBYSHEV_BOUNDARY = 1.05      # normalized boundary margin for Chebyshev fit
Q_WINDOW_UA_V = 15.0           # Ua proximity window for Q-point quality check
# ML-140: ra is a LOCAL derivative at Q — the per-curve Ia(Ua) slope is
# fitted inside this window around ua_q (widened ×2 → whole curve when the
# scan is too sparse for ≥3 points). Probed on Koren EL84 @ Q=101 V:
# ±50 V lands within ~9% of the model derivative, the whole-curve fit is
# 20%+ off (knee + plateau mixed into one line).
RA_WINDOW_UA_V = 50.0
Q_WINDOW_UG1_V = 1.0           # Ug1 proximity window for Q-point quality check
Q_WINDOW_UG2_V = 5.0           # Ug2 proximity window for Ig2 lookup at Q-point
SWEEP_BIAS_MARGIN = 0.1        # fraction of Ug1 range as bias sweep margin
MIN_SWEEP_SWING_V = 0.5        # minimum swing for amplitude/Ra sweep
OUTLIER_MAD_FACTOR = 2.5       # MAD-based outlier rejection threshold for robust slope
# ML-139 visibility: the composite extrapolates tube B beyond its data
# edge toward cutoff (normal for class AB — the passport EL84 PP point
# mirrors 10 V past a scan that stops at the bias). The tail is UNCERTAIN
# when the current at B's data edge is still large RELATIVE TO THE
# ANALYZED SIGNAL AMPLITUDE (ia_edge / b1, b1 ≈ (i_max−i_min)/2) — a
# grid-max denominator would depend on how far the positive Ug1 side was
# scanned, which is not physical. Warn tier: recommend rescanning B to
# deeper Ug1. Probed EL84@-11V: edge Ia 12 mA; tail model = space-charge
# 3/2 law (clamp 10.6 → linear 0.47 → 3/2 0.22 mA mean error vs truth).
B_EXTRAP_WARN_EDGE_FRACTION = 0.15
SRK_DIVERGENCE_THRESHOLD_PCT = 20.0  # SRK vs numerical divergence warning threshold (%)
MIN_CHEBYSHEV_HARMONIC = 3     # need at least HD2 + HD3
CHEBYSHEV_OVERFIT_RATIO = 3    # need ≥3 points per 2 harmonics to avoid overfitting

# ── Pa averaging ─────────────────────────────────────────────────────
N_PA_SAMPLES = 64  # samples per cycle for Pa averaging (enough for < 0.1% error)

# ── Stage parameter finite differences (model gm/ra) ─────────────────
MODEL_GM_DELTA_V = 0.05   # Ug1 step for gm finite difference (V)
MODEL_RA_DELTA_V = 1.0    # Ua step for ra finite difference (V)
