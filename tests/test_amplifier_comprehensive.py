"""Comprehensive amplifier tests — analytical PP, external references, coverage gaps.

Addresses identified test coverage gaps:
- Analytical PP distortion (mirrors SE TestAnalyticalHD quality)
- PP Pout verified via independent per-tube derivation
- External reference validation against Koren model ground truth
- Pentode Rk with Ig2
- Pentode gain verification
- Composite characteristic full-curve symmetry
- PP sweep with mismatched pair
- Ug2 filter fallback behaviour

Run:  py -m pytest tests/test_amplifier_comprehensive.py -v
"""

import math
from typing import List, Dict

import numpy as np
import pytest

from lm19.amplifier import (
    ResistiveLoadLine,
    TransformerLoadLine,
    CathodeFollowerLoadLine,
    PushPullLoadLine,
    find_intersections,
    compute_distortion,
    compute_imd,
    compute_headroom,
    sweep_amplitude,
    sweep_ra,
    optimize_bias,
    compute_stage_params,
    compute_cf_stage_params,
    composite_characteristic,
    pp_distortion,
    sweep_pp_amplitude,
    estimate_ig2_at_q,
)
from lm19.tube_sim import (
    TubeModel,
    load_model,
    quick_triode,
    quick_pentode,
    ScanGrid,
)


# ── module local constants ──
# EL84 + 5k output transformer at Ub=200 V, ±1.5 V swing → probed ≈390 mW.
# A 100 mW floor discriminates real power transfer from a degenerate result.
MIN_PENTODE_XFMR_POUT_MW = 100.0
# EL84 PP (8k a-a) vs resistive SE (5k) at Ub=250 V: probed ratio ≈ 9×
# (resistive SE burns headroom in Ra; PP composite swings both ways).
# Require at least 2× so a regression to "PP ≈ SE" fails loudly.
PP_VS_SE_MIN_POUT_RATIO = 2.0
# Matched-pair composite at an ON-GRID bias must be ≈0 (probed exactly 0.0);
# 1 mA vs Iq ≈ 40 mA for EL84 is a discriminating bound.
COMPOSITE_CENTER_TOL_MA = 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers for analytical PP tests
# ═══════════════════════════════════════════════════════════════════════════

def _make_pp_composite_from_polynomial(
    a0: float, a1: float, a2: float, a3: float,
    bias: float, half_swing: float, n_points: int = 21,
) -> List[Dict]:
    """Build synthetic measurement points for a tube with Ia = a0 + a1*x + a2*x² + a3*x³.

    The composite Ia_A(ug1) - Ia_B(2*bias - ug1) for matched pair gives:
      - HD2 cancelled (even part cancels)
      - HD3 remains (odd part)

    Returns points suitable for composite_characteristic / pp_distortion.
    Ra=10 kOhm used for Ua = 200 - (ug1 - bias) * 10.
    """
    ra = 10.0
    ua_q = 200.0
    pts: List[Dict] = []
    ug1_values = np.linspace(bias - half_swing * 1.5, bias + half_swing * 1.5, n_points)
    for ug1 in ug1_values:
        x = ug1 - bias
        ia = a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3
        ua = ua_q - x * ra
        pts.append({"ug1": float(ug1), "ua": float(ua), "ia": float(max(ia, 0))})
    return pts


def _analytical_pp_fourier(a1: float, a3: float, half_swing: float):
    """Exact Fourier coefficients for the PP composite of matched pair.

    For Ia(x) = a0 + a1*x + a2*x² + a3*x³, the composite is:
      Ic(x) = Ia(bias+x) - Ia(bias-x) = 2*a1*x + 2*a3*x³  (only odd terms)

    With sinusoidal drive x = V*cos(θ):
      Ic = 2*a1*V*cos + 2*a3*V³*cos³
         = 2*a1*V*cos + 2*a3*V³*(3/4*cos + 1/4*cos3)
         = (2*a1*V + 1.5*a3*V³)*cos + 0.5*a3*V³*cos3

    So: B1 = 2*a1*V + 1.5*a3*V³, B2 = 0 (perfect cancellation), B3 = 0.5*a3*V³

    The 5-point method samples at x = ±V (θ=0,180), ±V/2 (θ=60,120), 0 (θ=90).
    """
    v = half_swing
    b1 = 2.0 * a1 * v + 1.5 * a3 * v ** 3
    b3 = 0.5 * a3 * v ** 3
    hd2 = 0.0  # perfect cancellation for matched pair
    hd3 = abs(b3 / b1) * 100.0 if abs(b1) > 1e-12 else 0.0
    return b1, hd2, hd3


# ═══════════════════════════════════════════════════════════════════════════
#  Analytical PP tests (mirror SE TestAnalyticalHD quality)
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalyticalPPFourier:
    """Verify PP distortion coefficients against exact Fourier analysis."""

    def test_pure_linear_zero_distortion(self):
        """Perfectly linear matched pair → HD2 = HD3 = 0."""
        pts = _make_pp_composite_from_polynomial(10.0, 2.0, 0.0, 0.0, bias=-8.0, half_swing=4.0)
        ll = PushPullLoadLine(ub=300, ra_aa=40.0)  # Ra_aa doesn't affect HD
        dist = pp_distortion(pts, ll, ug1_bias=-8.0, half_swing=4.0)
        assert dist is not None
        assert dist["hd2"] < 0.1, f"HD2={dist['hd2']:.3f}% should be ~0"
        assert dist["hd3"] < 0.1, f"HD3={dist['hd3']:.3f}% should be ~0"

    def test_quadratic_cancelled_in_pp(self):
        """Quadratic nonlinearity (a2) produces HD2 in SE but cancels in PP."""
        a0, a1, a2, a3 = 10.0, 2.0, 0.3, 0.0
        pts = _make_pp_composite_from_polynomial(a0, a1, a2, a3, bias=-8.0, half_swing=4.0)
        ll = PushPullLoadLine(ub=300, ra_aa=40.0)
        dist = pp_distortion(pts, ll, ug1_bias=-8.0, half_swing=4.0)
        assert dist is not None
        assert dist["hd2"] < 0.5, \
            f"PP HD2 should be ~0 for matched pair with quadratic only, got {dist['hd2']:.3f}%"
        assert dist["hd3"] < 0.1

    def test_cubic_hd3_matches_fourier(self):
        """Cubic nonlinearity: HD3 should match analytical prediction."""
        a0, a1, a2, a3 = 10.0, 2.0, 0.1, 0.02
        hs = 3.0
        pts = _make_pp_composite_from_polynomial(a0, a1, a2, a3, bias=-8.0, half_swing=hs)
        ll = PushPullLoadLine(ub=300, ra_aa=40.0)
        dist = pp_distortion(pts, ll, ug1_bias=-8.0, half_swing=hs)
        assert dist is not None

        _, hd2_exact, hd3_exact = _analytical_pp_fourier(a1, a3, hs)
        assert dist["hd2"] < 1.0, f"PP HD2 should be near 0, got {dist['hd2']:.3f}%"
        assert abs(dist["hd3"] - hd3_exact) < 0.5, \
            f"PP HD3: code={dist['hd3']:.3f}% exact={hd3_exact:.3f}%"

    def test_various_amplitudes(self):
        """HD3 should scale correctly with amplitude in PP."""
        a0, a1, a2, a3 = 10.0, 3.0, 0.2, 0.03
        for hs in [1.0, 2.0, 3.0, 4.0]:
            pts = _make_pp_composite_from_polynomial(a0, a1, a2, a3, bias=-8.0, half_swing=hs)
            ll = PushPullLoadLine(ub=300, ra_aa=40.0)
            dist = pp_distortion(pts, ll, ug1_bias=-8.0, half_swing=hs)
            if dist is None:
                continue
            _, _, hd3_exact = _analytical_pp_fourier(a1, a3, hs)
            # Tolerance grows with amplitude (composite averaging smooths extremes)
            tol = 0.5 + 0.3 * hs
            assert abs(dist["hd3"] - hd3_exact) < tol, \
                f"hs={hs}: PP HD3 code={dist['hd3']:.3f}% exact={hd3_exact:.3f}%"

    def test_high_cubic_distortion(self):
        """Higher a3 → higher HD3, verified analytically."""
        a0, a1, a2, a3 = 8.0, 2.0, 0.0, 0.05
        hs = 3.0
        pts = _make_pp_composite_from_polynomial(a0, a1, a2, a3, bias=-8.0, half_swing=hs)
        ll = PushPullLoadLine(ub=300, ra_aa=40.0)
        dist = pp_distortion(pts, ll, ug1_bias=-8.0, half_swing=hs)
        assert dist is not None
        _, _, hd3_exact = _analytical_pp_fourier(a1, a3, hs)
        assert hd3_exact > 2.0, "Test case should have significant HD3"
        assert abs(dist["hd3"] - hd3_exact) < 1.0

    def test_negative_a2_still_cancelled(self):
        """Negative a2 (asymmetric SE) still cancels in PP."""
        pts = _make_pp_composite_from_polynomial(10.0, 2.0, -0.2, 0.0, bias=-8.0, half_swing=4.0)
        ll = PushPullLoadLine(ub=300, ra_aa=40.0)
        dist = pp_distortion(pts, ll, ug1_bias=-8.0, half_swing=4.0)
        assert dist is not None
        assert dist["hd2"] < 0.5, f"Negative a2 should still cancel in PP, HD2={dist['hd2']:.3f}%"

    def test_thd_is_rss(self):
        """THD should equal sqrt(HD2² + HD3²) for PP."""
        a0, a1, a2, a3 = 10.0, 2.0, 0.15, 0.02
        pts = _make_pp_composite_from_polynomial(a0, a1, a2, a3, bias=-8.0, half_swing=3.0)
        ll = PushPullLoadLine(ub=300, ra_aa=40.0)
        dist = pp_distortion(pts, ll, ug1_bias=-8.0, half_swing=3.0)
        assert dist is not None
        expected = math.sqrt(dist["hd2"] ** 2 + dist["hd3"] ** 2)
        assert abs(dist["thd"] - expected) < 0.01


class TestAnalyticalPPPout:
    """Verify PP output power against independent derivations."""

    def test_pp_pout_matches_per_tube_derivation(self):
        """PP Pout = 2 × P_per_tube, via transformer coupling derivation.

        For center-tapped PP, both primary halves contribute constructively.
        composite swing × Ra_per_tube gives V_composite, and
        P = swing × V_composite / 8 (standard sinusoidal formula).
        """
        _, pts = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        dist = pp_distortion(pts, ll, ug1_bias=-10.0)
        assert dist is not None

        swing = dist["i_max"] - dist["i_min"]  # composite Ipp
        ra_pt = ll.ra_per_tube  # Ra_aa / 4
        # V_composite = swing * Ra_per_tube (both halves contribute)
        v_comp = swing * ra_pt
        expected_pout = swing * v_comp / 8.0
        assert abs(dist["pout_mw"] - expected_pout) < 0.01

    def test_pp_pout_from_secondary_side(self):
        """Verify PP Pout via secondary side: P = I_sec² × R_load / 2.

        For a 2N:n transformer, I_sec_peak = 2 × ΔI_tube × N1/N2.
        Total amp-turns: 2 × ΔI × N1, secondary current: 2×ΔI×N1/N2.
        P = (I_sec_peak)² × R_load / 2 = 2 × ΔI² × (N1/N2)² × R_load.
        With (N1/N2)² × R_load = Ra_per_tube:
        P = 2 × ΔI² × Ra_per_tube.
        ΔI = composite_amplitude / 2 = (i_max + |i_min|) / 4 for matched.
        """
        _, pts = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=10.0)
        dist = pp_distortion(pts, ll, ug1_bias=-10.0, half_swing=5.0)
        assert dist is not None

        # composite amplitude (peak from zero)
        i_amp = max(abs(dist["i_max"]), abs(dist["i_min"]))
        delta_i = i_amp / 2.0  # per-tube current amplitude
        # Power from transformer coupling: 2 × ΔI² × Ra_per_tube
        p_secondary = 2.0 * delta_i ** 2 * ll.ra_per_tube

        # The 5-point formula (swing*V/8) should give the same
        # within the tolerance of matched pair approximation
        assert abs(dist["pout_mw"] - p_secondary) / max(p_secondary, 0.01) < 0.15, \
            f"PP Pout: code={dist['pout_mw']:.2f} secondary={p_secondary:.2f}"

    def test_pp_linear_pout_exact(self):
        """For purely linear transfer, PP Pout can be computed analytically."""
        a0, a1 = 10.0, 2.0
        hs = 3.0
        bias = -8.0
        pts = _make_pp_composite_from_polynomial(a0, a1, 0.0, 0.0, bias=bias, half_swing=hs)
        ll = PushPullLoadLine(ub=300, ra_aa=40.0)
        dist = pp_distortion(pts, ll, ug1_bias=bias, half_swing=hs)
        assert dist is not None

        # For linear: composite = 2*a1*x, swing = 2*a1*hs - (-2*a1*hs) = 4*a1*hs
        expected_swing = 4.0 * a1 * hs
        swing = dist["i_max"] - dist["i_min"]
        assert abs(swing - expected_swing) < 0.5, \
            f"swing={swing:.2f} expected={expected_swing:.2f}"

        ua_swing = swing * ll.ra_per_tube
        expected_pout = swing * ua_swing / 8.0
        assert abs(dist["pout_mw"] - expected_pout) < 0.1


class TestPPFourierCoefficients:
    """Verify PP 5-point Fourier coefficient formulas against exact values."""

    def test_b1_matches_fundamental(self):
        """PP B1 = (swing + half_diff) / 3 should match exact B1 for composite."""
        a0, a1, a2, a3 = 10.0, 3.0, 0.2, 0.02
        hs = 3.0
        bias = -8.0

        # Exact composite values at 5 sampling points
        def comp(x):
            """Composite: Ia(bias+x) - Ia(bias-x) = 2*a1*x + 2*a3*x³"""
            return 2.0 * a1 * x + 2.0 * a3 * x ** 3

        i_max = comp(hs)
        i_min = comp(-hs)
        i_0 = comp(0)
        i_high = comp(hs / 2)
        i_low = comp(-hs / 2)

        swing = i_max - i_min
        half_diff = i_high - i_low

        # Code formula
        b1_code = (swing + half_diff) / 3.0
        b3_code = (swing - 2.0 * half_diff) / 6.0

        # Exact from Fourier
        b1_exact, _, _ = _analytical_pp_fourier(a1, a3, hs)
        b3_exact = 0.5 * a3 * hs ** 3

        assert abs(b1_code - b1_exact) < 1e-6, \
            f"PP B1: formula={b1_code:.6f} exact={b1_exact:.6f}"
        assert abs(b3_code - b3_exact) < 1e-6, \
            f"PP B3: formula={b3_code:.6f} exact={b3_exact:.6f}"

    def test_b2_zero_for_matched(self):
        """PP B2 should be zero for matched pair (even harmonics cancel)."""
        a0, a1, a2, a3 = 10.0, 3.0, 0.5, 0.02
        hs = 3.0

        def comp(x):
            return 2.0 * a1 * x + 2.0 * a3 * x ** 3

        i_max = comp(hs)
        i_min = comp(-hs)
        i_0 = comp(0)

        b2 = (i_max + i_min - 2.0 * i_0) / 4.0
        assert abs(b2) < 1e-10, f"PP B2 should be 0 for matched pair, got {b2}"


# ═══════════════════════════════════════════════════════════════════════════
#  External reference tests — Koren model ground truth
# ═══════════════════════════════════════════════════════════════════════════

class TestExternalReferenceTriode:
    """Validate amplifier calculations against known tube specifications.

    Uses Koren model (tube_sim) as ground truth — parameters from tube_params.json
    are fitted to datasheet curves, so they represent real tube behaviour.
    """

    def test_12ax7_preamp_gain(self):
        """12AX7: high-mu triode, gain ~ 40-60 at Ub=250V Ra=100kΩ.

        Datasheet: mu=100, ra~63kΩ, gm~1.6mA/V.
        Expected gain = mu*Ra/(ra+Ra) ≈ 100*100/(63+100) ≈ 61.
        """
        _, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3

        result = compute_stage_params(isects, ll, ug1_bias=-2.0, points=pts)
        assert result is not None
        assert 20.0 < result["gain"] < 100.0, \
            f"12AX7 gain={result['gain']:.1f} should be 20-100"
        assert result["gain_db"] > 20.0, f"12AX7 gain_db={result['gain_db']:.1f}"
        # gm should be ~1.0-2.5 mA/V
        assert 0.5 < result["gm"] < 5.0, f"12AX7 gm={result['gm']:.2f}"

    def test_12au7_medium_mu_gain(self):
        """12AU7: medium-mu triode, gain ~ 10-20 at Ub=250V Ra=10kΩ.

        Datasheet: mu=20, ra~7.7kΩ, gm~2.2mA/V.
        Expected gain = 20*10/(7.7+10) ≈ 11.3.
        """
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3

        result = compute_stage_params(isects, ll, ug1_bias=-8.0, points=pts)
        assert result is not None
        assert 5.0 < result["gain"] < 30.0, \
            f"12AU7 gain={result['gain']:.1f} should be 5-30"

    def test_12au7_cf_gain_near_unity(self):
        """12AU7 cathode follower: gain < 1, Zout << Ra."""
        _, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3

        result = compute_cf_stage_params(isects, ll, ug1_bias=-8.0, points=pts)
        assert result is not None
        assert 0.4 < result["gain"] < 1.0, \
            f"CF gain={result['gain']:.3f} should be 0.4-1.0"
        # Zout = ra/(mu+1) should be ~0.3-0.5 kΩ for 12AU7
        assert result["zout"] < 2.0, f"CF Zout={result['zout']:.2f}kΩ should be low"

    def test_12au7_distortion_at_moderate_swing(self):
        """12AU7 SE: THD at moderate swing should be 1-15%."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)

        dist = compute_distortion(isects, ug1_bias=-8.0, half_swing=3.0)
        assert dist is not None
        assert 0.5 < dist["thd"] < 20.0, f"12AU7 THD={dist['thd']:.1f}%"
        assert dist["pout_mw"] > 10.0, f"Should produce meaningful power"

    def test_el84_pentode_output_stage(self):
        """EL84 pentode: output power 1-5W range at Ub=250V Ra=5kΩ.

        Mullard spec: Class A, Ub=250V, Ra=5.2kΩ → ~4.3W.
        """
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=250, ra=5.0)
        isects = find_intersections(pts, ll, ug2_filter=250.0)
        if len(isects) < 3:
            # Try without ug2 filter
            isects = find_intersections(pts, ll)
        assert len(isects) >= 3

        dist = compute_distortion(isects, ug1_bias=-7.0)
        assert dist is not None
        assert dist["pout_mw"] > 100.0, \
            f"EL84 should produce meaningful output, got {dist['pout_mw']:.1f}mW"

    def test_el84_pp_output_power(self):
        """EL84 PP: much more power than resistive SE (probed ratio ≈ 9×).

        The resistive SE line wastes DC headroom in Ra while the PP composite
        swings the full supply both ways, so the ratio well exceeds the
        classic 2-4× transformer-SE figure. Require ≥ 2× as the floor.
        """
        _, pts = quick_pentode("EL84")
        ll_se = ResistiveLoadLine(ub=250, ra=5.0)
        ll_pp = PushPullLoadLine(ub=250, ra_aa=8.0)

        dist_se = compute_distortion(
            find_intersections(pts, ll_se), ug1_bias=-7.0,
        )
        dist_pp = pp_distortion(pts, ll_pp, ug1_bias=-7.0)

        assert dist_se is not None, "SE distortion must be computable"
        assert dist_pp is not None, "PP distortion must be computable"
        assert dist_pp["pout_mw"] > dist_se["pout_mw"] * PP_VS_SE_MIN_POUT_RATIO, \
            f"PP={dist_pp['pout_mw']:.0f}mW should be >{PP_VS_SE_MIN_POUT_RATIO}x " \
            f"SE={dist_se['pout_mw']:.0f}mW"

    def test_el84_pentode_gain_is_gm_times_ra(self):
        """For pentode (ra >> Ra_load), gain ≈ gm × Ra_load."""
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=250, ra=5.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3

        result = compute_stage_params(isects, ll, ug1_bias=-7.0, points=pts)
        assert result is not None
        # For pentode: gain ~ gm * Ra (since ra >> Ra)
        expected_gain = result["gm"] * 5.0  # gm in mA/V, Ra in kΩ → gain dimensionless
        if result["ra"] > 20.0:  # Only check if ra is indeed >> Ra
            assert abs(result["gain"] - expected_gain) / max(expected_gain, 1) < 0.3, \
                f"Pentode gain={result['gain']:.1f} expected≈gm*Ra={expected_gain:.1f}"


class TestExternalReferencePentode:
    """Additional pentode-specific validation."""

    def test_pentode_zout_near_ra_load(self):
        """For pentode, Zout = ra || Ra ≈ Ra when ra >> Ra."""
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=250, ra=5.0)
        isects = find_intersections(pts, ll)
        result = compute_stage_params(isects, ll, ug1_bias=-7.0, points=pts)
        # zout = ra||Ra is strictly below ra for ANY ra > 0, and never
        # exceeds Ra, by the parallel-resistance formula — no gating
        # on the fitted ra value is needed (an 'ra > 20' gate would
        # silently skip the whole check on low-ra fits).
        assert result is not None, "stage params must compute"
        assert result["zout"] < result["ra"], \
            f"Pentode Zout={result['zout']:.1f} should be < ra={result['ra']:.1f}"
        assert 0 < result["zout"] <= 5.0 + 1e-6, (
            f"Zout={result['zout']:.2f} exceeds Ra=5.0 (parallel)")

    def test_headroom_limited_by_cutoff_or_grid(self):
        """EL84 headroom should identify realistic clip limits."""
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=250, ra=5.0)
        isects = find_intersections(pts, ll)
        hr = compute_headroom(isects, ug1_bias=-7.0, pa_max=12.0)
        assert hr is not None
        assert hr["max_swing"] > 0.5, f"Should have some headroom"
        assert hr["clip_neg"] in ("cutoff", "data_limit", "pa_max")
        assert hr["clip_pos"] in ("grid_current", "data_limit", "pa_max")


# ═══════════════════════════════════════════════════════════════════════════
#  Coverage gaps
# ═══════════════════════════════════════════════════════════════════════════

class TestRkPentodeWithIg2:
    """Verify cathode bias resistor accounts for Ig2 in pentodes."""

    def test_rk_pentode_includes_ig2(self):
        """For pentode, Rk = |Ug1| / (Ia + Ig2) × 1000, not just Ia."""
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=250, ra=5.0)
        opt = optimize_bias(pts, ll, target="min_thd")
        assert opt is not None

        # Get Ig2 near Q-point
        ig2_q = estimate_ig2_at_q(pts, opt["ug1_0"], opt["ua_0"])
        if ig2_q > 0.1:  # Only test if we have significant Ig2
            # Expected Rk with Ig2
            ik = opt["ia_0"] + ig2_q
            rk_correct = abs(opt["ug1_0"]) / ik * 1000.0
            # The reported Rk should match (includes Ig2)
            assert abs(opt["rk_auto_bias"] - rk_correct) < rk_correct * 0.15, \
                f"Rk={opt['rk_auto_bias']:.0f}Ω expected≈{rk_correct:.0f}Ω (with Ig2={ig2_q:.1f}mA)"
            # And should be LESS than Rk without Ig2
            rk_without_ig2 = abs(opt["ug1_0"]) / opt["ia_0"] * 1000.0
            assert opt["rk_auto_bias"] < rk_without_ig2, \
                f"Rk with Ig2 ({opt['rk_auto_bias']:.0f}) should be < Rk without ({rk_without_ig2:.0f})"

    def test_rk_triode_unchanged(self):
        """For triode, Rk should still be |Ug1| / Ia × 1000 (no Ig2)."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        opt = optimize_bias(pts, ll, target="min_thd")
        assert opt is not None

        rk_expected = abs(opt["ug1_0"]) / opt["ia_0"] * 1000.0
        assert abs(opt["rk_auto_bias"] - rk_expected) < 1.0, \
            f"Triode Rk={opt['rk_auto_bias']:.0f} expected={rk_expected:.0f}"

    def test_rk_value_is_reasonable(self):
        """Rk should be in reasonable range for common tubes."""
        for tube_name, gen_func, bias in [
            ("12AU7", quick_triode, -8.0),
            ("12AX7", quick_triode, -2.0),
        ]:
            _, pts = gen_func(tube_name)
            ll = ResistiveLoadLine(ub=250, ra=10.0 if tube_name == "12AU7" else 100.0)
            opt = optimize_bias(pts, ll, target="min_thd")
            # optimize_bias must yield a result with positive Rk on
            # healthy triode rigs.
            assert opt is not None, f"{tube_name}: optimize_bias failed"
            assert opt["rk_auto_bias"] > 0, f"{tube_name}: Rk not positive"
            # Typical cathode resistors: 100Ω - 50kΩ
            assert 50 < opt["rk_auto_bias"] < 100_000, \
                f"{tube_name}: Rk={opt['rk_auto_bias']:.0f}Ω out of range"


class TestEstimateIg2:
    """Test the estimate_ig2_at_q helper function."""

    def test_returns_zero_for_triode(self):
        """Triode data has no ig2 → returns 0."""
        _, pts = quick_triode("12AU7")
        ig2 = estimate_ig2_at_q(pts, ug1_q=-8.0, ua_q=200.0)
        assert ig2 == 0.0

    def test_returns_positive_for_pentode(self):
        """Pentode data should have positive Ig2."""
        _, pts = quick_pentode("EL84")
        # Find a reasonable Q-point
        ig2 = estimate_ig2_at_q(pts, ug1_q=-7.0, ua_q=200.0, ug2_filter=250.0)
        # May or may not find points depending on scan grid, so check if found
        assert ig2 > 0, "guard de-vacuated 2026-07-12: value must be present"
        assert ig2 < 50.0, f"Ig2={ig2:.1f}mA seems too high"

    def test_ug2_filter_narrows_results(self):
        """Ug2 filter should select only matching points."""
        _, pts = quick_pentode("EL84")
        ig2_all = estimate_ig2_at_q(pts, ug1_q=-7.0, ua_q=200.0)
        ig2_250 = estimate_ig2_at_q(pts, ug1_q=-7.0, ua_q=200.0, ug2_filter=250.0)
        # Both should be non-negative
        assert ig2_all >= 0
        assert ig2_250 >= 0


class TestCompositeSymmetry:
    """Verify composite characteristic symmetry for matched pairs."""

    def test_odd_symmetry_full_curve(self):
        """Composite of matched pair should be odd-symmetric around bias."""
        _, pts = quick_triode("12AU7")
        bias = -10.0
        comp = composite_characteristic(pts, ug1_bias=bias)
        assert len(comp) >= 5

        # For each point, find its mirror and check antisymmetry
        sym_errors = []
        for c in comp:
            ug1 = c["ug1"]
            dx = ug1 - bias
            mirror_ug1 = bias - dx
            # Find closest point to mirror
            closest = min(comp, key=lambda p: abs(p["ug1"] - mirror_ug1))
            if abs(closest["ug1"] - mirror_ug1) < 0.5:
                # For matched pair: comp(bias+dx) ≈ -comp(bias-dx)
                expected = -closest["ia_composite"]
                if abs(c["ia_composite"]) > 0.1:
                    error = abs(c["ia_composite"] - expected) / abs(c["ia_composite"])
                    sym_errors.append(error)

        # Guard must hold — otherwise the symmetry was never actually checked
        # (probed: 10 mirror pairs found for the 12AU7 quick_triode grid)
        assert sym_errors, "no mirror pairs found — symmetry check did not run"
        avg_error = sum(sym_errors) / len(sym_errors)
        assert avg_error < 0.15, \
            f"Average symmetry error {avg_error:.3f} too high (should be <0.15)"

    def test_composite_zero_at_bias(self):
        """Composite Ia should be near 0 at the bias point."""
        _, pts = quick_triode("12AU7")
        comp = composite_characteristic(pts, ug1_bias=-10.0)
        center = min(comp, key=lambda c: abs(c["ug1"] - (-10.0)))
        assert abs(center["ia_composite"]) < 1.0, \
            f"Composite at bias = {center['ia_composite']:.3f}, should be ~0"

    def test_pentode_composite_symmetry(self):
        """Pentode composite should also be approximately odd-symmetric.

        For pentode, the composite is computed from the averaged Ia(Ug1)
        transfer curve across all Ua values. Bias -7.5 V lands exactly on the
        quick_pentode Ug1 grid (step 1.5 V) — an off-grid bias like -7.0 V
        snaps the center 0.5 V away and reads ≈ gm·2·dx ≈ 8.8 mA, which is a
        grid artifact, not asymmetry. On-grid the composite must be ≈ 0.
        """
        _, pts = quick_pentode("EL84")
        bias = -7.5  # on-grid; also a realistic EL84 operating bias
        comp = composite_characteristic(pts, ug1_bias=bias, ug2_filter=250.0)
        if len(comp) < 5:
            comp = composite_characteristic(pts, ug1_bias=bias)
        assert len(comp) >= 5, \
            f"composite must be computable from quick_pentode data, got {len(comp)} pts"
        center = min(comp, key=lambda c: abs(c["ug1"] - bias))
        assert abs(center["ia_composite"]) < COMPOSITE_CENTER_TOL_MA, \
            f"Pentode composite at bias = {center['ia_composite']:.1f}, should be near 0"


class TestPPSweepWithMismatch:
    """Test PP amplitude sweep with mismatched tube pairs."""

    def _make_mismatched_pair(self, tube_name="12AU7", mismatch_pct=10):
        """Create two tube datasets: one normal, one with scaled gm."""
        _, pts_a = quick_triode(tube_name)
        # Create mismatched tube B: scale Ia by (1 + mismatch/100)
        scale = 1.0 + mismatch_pct / 100.0
        pts_b = []
        for p in pts_a:
            pb = dict(p)
            pb["ia"] = p["ia"] * scale
            pts_b.append(pb)
        return pts_a, pts_b

    def test_mismatch_sweep_has_results(self):
        """PP sweep with mismatched pair should produce results."""
        pts_a, pts_b = self._make_mismatched_pair()
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        results = sweep_pp_amplitude(pts_a, ll, ug1_bias=-10.0, points_b=pts_b)
        assert len(results) > 3, "Mismatch sweep should produce results"

    def test_mismatch_increases_hd2_in_sweep(self):
        """Mismatched pair should show higher HD2 across sweep."""
        pts_a, pts_b = self._make_mismatched_pair(mismatch_pct=15)
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)

        sweep_matched = sweep_pp_amplitude(pts_a, ll, ug1_bias=-10.0)
        sweep_mismatch = sweep_pp_amplitude(pts_a, ll, ug1_bias=-10.0, points_b=pts_b)

        # Both sweeps must actually produce data (probed: 33 points each)
        assert len(sweep_matched) > 3, f"matched sweep empty: {len(sweep_matched)}"
        assert len(sweep_mismatch) > 3, f"mismatch sweep empty: {len(sweep_mismatch)}"
        avg_hd2_matched = sum(r["hd2"] for r in sweep_matched) / len(sweep_matched)
        avg_hd2_mismatch = sum(r["hd2"] for r in sweep_mismatch) / len(sweep_mismatch)
        assert avg_hd2_mismatch > avg_hd2_matched, \
            f"Mismatched HD2={avg_hd2_mismatch:.2f} should > matched={avg_hd2_matched:.2f}"

    def test_mismatch_balance_error_increases(self):
        """Balance error should increase with mismatch level."""
        pts_a_10, pts_b_10 = self._make_mismatched_pair(mismatch_pct=10)
        pts_a_20, pts_b_20 = self._make_mismatched_pair(mismatch_pct=20)
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)

        dist_10 = pp_distortion(pts_a_10, ll, -10.0, points_b=pts_b_10, half_swing=4.0)
        dist_20 = pp_distortion(pts_a_20, ll, -10.0, points_b=pts_b_20, half_swing=4.0)

        # Both must compute (probed: balance 0.99 vs 1.89) — a None here
        # would silently skip the property under test
        assert dist_10 is not None, "10% mismatch pp_distortion returned None"
        assert dist_20 is not None, "20% mismatch pp_distortion returned None"
        assert dist_20["balance_error"] > dist_10["balance_error"], \
            f"20% mismatch balance={dist_20['balance_error']:.2f} " \
            f"should > 10%={dist_10['balance_error']:.2f}"

    def test_pentode_mismatch_sweep(self):
        """Pentode PP with mismatch should also work."""
        _, pts_a = quick_pentode("EL84")
        pts_b = [dict(p, ia=p["ia"] * 1.1) for p in pts_a]
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)
        results = sweep_pp_amplitude(pts_a, ll, ug1_bias=-7.0, points_b=pts_b)
        assert len(results) >= 1, "Pentode mismatch sweep should work"


class TestUg2FilterFallback:
    """Test behaviour when Ug2 filter doesn't match any data."""

    def test_find_intersections_still_works(self):
        """Non-matching ug2_filter should fall back to all points."""
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects_normal = find_intersections(pts, ll)
        isects_bad_filter = find_intersections(pts, ll, ug2_filter=999.0)
        # Both should return results (fallback to unfiltered)
        assert len(isects_normal) >= 3
        assert len(isects_bad_filter) >= 3

    def test_composite_with_bad_ug2_filter(self):
        """Composite characteristic should work even with non-matching ug2."""
        _, pts = quick_pentode("EL84")
        comp = composite_characteristic(pts, ug1_bias=-7.0, ug2_filter=999.0)
        assert len(comp) >= 3, "Should fall back to all data"


class TestTransformerLoadLinePipeline:
    """End-to-end tests for transformer-coupled stages."""

    def test_transformer_se_distortion(self):
        """Transformer-coupled SE should produce valid distortion results."""
        _, pts = quick_triode("12AU7")
        ll = TransformerLoadLine(ub=300, ra_dc=0.1, ra_ac=5.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3

        dist = compute_distortion(isects, ug1_bias=-8.0, half_swing=3.0)
        assert dist is not None
        assert dist["pout_mw"] > 0
        assert dist["thd"] > 0

    def test_transformer_stage_params(self):
        """Stage params should use ra_ac for gain calculation."""
        _, pts = quick_triode("12AU7")
        ll = TransformerLoadLine(ub=300, ra_dc=0.1, ra_ac=5.0)
        isects = find_intersections(pts, ll)
        result = compute_stage_params(isects, ll, ug1_bias=-8.0, points=pts)
        assert result is not None
        # Gain should be positive and reasonable for transformer-coupled triode
        assert result["gain"] > 1.0, f"Transformer gain={result['gain']:.1f}"


class TestKorenModelConsistency:
    """Cross-validate amplifier calculations using Koren model directly.

    Generate synthetic measurement data from Koren model, then verify
    amplifier calculations against model-derived parameters.
    """

    def test_model_gm_matches_numerical(self):
        """Numerical gm from amplifier module should match model-derived gm."""
        model = load_model("12AU7")
        assert model is not None
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)

        result = compute_stage_params(isects, ll, ug1_bias=-8.0, points=pts)
        assert result is not None

        # Model-derived gm = dIa/dUg1 at Ua_q
        delta = 0.1
        ua_q = 170.0  # approximate
        gm_model = (model.ia(ua_q, -8.0 + delta) - model.ia(ua_q, -8.0 - delta)) / (2 * delta)

        if gm_model > 0.1:
            ratio = result["gm"] / gm_model
            assert 0.3 < ratio < 3.0, \
                f"gm: numerical={result['gm']:.2f} model={gm_model:.2f}"

    def test_model_mu_consistent(self):
        """mu = gm × ra should hold for computed parameters."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        result = compute_stage_params(isects, ll, ug1_bias=-8.0, points=pts)
        assert result is not None

        expected_mu = result["gm"] * result["ra"]
        assert abs(result["mu"] - expected_mu) / max(expected_mu, 0.1) < 0.01, \
            f"mu={result['mu']:.1f} should ≈ gm*ra={expected_mu:.1f}"


class TestSweepAmplitudeReferenceValues:
    """Verify sweep amplitude produces physically meaningful results."""

    def test_thd_at_zero_swing_near_zero(self):
        """At very small swing, THD should be very low."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-8.0, steps=40)
        assert len(results) > 5
        # First point (smallest swing) should have low THD
        assert results[0]["thd"] < 5.0, \
            f"THD at small swing = {results[0]['thd']:.1f}% should be low"

    def test_pout_proportional_to_swing_squared(self):
        """For small swings, Pout ∝ swing² (linear region)."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-8.0, steps=40)
        assert len(results) >= 4, "guard de-vacuated 2026-07-12: value must be present"
        r1, r2 = results[0], results[1]
        if r1["half_swing"] > 0.1 and r1["pout_mw"] > 0.01:
            swing_ratio = (r2["half_swing"] / r1["half_swing"]) ** 2
            pout_ratio = r2["pout_mw"] / r1["pout_mw"]
            # Should be approximately equal in linear region
            assert 0.3 < pout_ratio / swing_ratio < 3.0, \
                f"Pout ratio={pout_ratio:.2f} vs swing²_ratio={swing_ratio:.2f}"


class TestIMDHalfSwing:
    """Verify compute_imd half_swing parameter filters data."""

    def test_imd_changes_with_half_swing(self):
        """Narrower half_swing uses fewer points → different IMD coefficients."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)

        imd_wide = compute_imd(isects, ug1_bias=-8.0, half_swing=None)
        imd_narrow = compute_imd(isects, ug1_bias=-8.0, half_swing=2.0)

        assert imd_wide is not None
        # Narrow swing may return None if < 4 points in window, or different values
        if imd_narrow is not None:
            # With fewer points the polynomial fit changes
            # (may or may not differ significantly — just verify it runs)
            assert imd_narrow["imd2"] >= 0
            assert imd_narrow["imd3"] >= 0

    def test_imd_none_too_narrow_returns_none(self):
        """Very narrow half_swing with few points inside should return None."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)

        # Very narrow window — likely < 4 points inside
        result = compute_imd(isects, ug1_bias=-8.0, half_swing=0.1)
        # Contract disjunction: None (<4 points in window) OR valid imd.
        if result is not None:  # vacuity-ok: None is legal for a narrow window
            assert result["imd2"] >= 0

    def test_imd_full_swing_equals_no_swing(self):
        """Very wide half_swing should give same result as None."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)

        imd_none = compute_imd(isects, ug1_bias=-8.0, half_swing=None)
        imd_huge = compute_imd(isects, ug1_bias=-8.0, half_swing=100.0)

        assert imd_none is not None and imd_huge is not None
        # All points inside the window → same result
        assert abs(imd_none["imd2"] - imd_huge["imd2"]) < 0.01
        assert abs(imd_none["imd3"] - imd_huge["imd3"]) < 0.01


class TestPPSignedB1B3Consistency:
    """Verify PP B1/B3 use signed values consistently with SE."""

    def test_pp_and_se_b1_formula_consistent(self):
        """For the same polynomial, PP and SE B1 formulas should use same signs."""
        a0, a1, a2, a3 = 10.0, 2.0, 0.1, 0.01
        hs = 3.0
        bias = -8.0

        # SE: create intersections and compute
        se_isects = []
        ra = 10.0
        ua_q = 200.0
        for x in np.linspace(-hs * 1.5, hs * 1.5, 21):
            ia = a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3
            ua = ua_q - x * ra
            se_isects.append({"ug1": float(bias + x), "ua": float(ua), "ia": float(max(ia, 0))})

        se_dist = compute_distortion(se_isects, ug1_bias=bias, half_swing=hs)
        assert se_dist is not None

        # PP: same data, matched pair
        pp_pts = se_isects
        ll = PushPullLoadLine(ub=300, ra_aa=40.0)
        pp_dist = pp_distortion(pp_pts, ll, ug1_bias=bias, half_swing=hs)
        assert pp_dist is not None

        # SE should have HD2 from a2, PP should cancel it
        assert se_dist["hd2"] > 0.5, f"SE should have significant HD2"
        assert pp_dist["hd2"] < se_dist["hd2"], \
            f"PP HD2={pp_dist['hd2']:.2f} should be < SE HD2={se_dist['hd2']:.2f}"


# ═══════════════════════════════════════════════════════════════════════════
#  Transformer AC load line tests
# ═══════════════════════════════════════════════════════════════════════════

from lm19.amplifier import _find_dc_q_point
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)


class TestFindDcQPoint:
    """Test the DC Q-point finder for transformer coupling."""

    def test_q_point_near_ub_for_low_ra_dc(self):
        """With very low Ra_dc, Q-point Ua should be ≈ Ub."""
        _, pts = quick_triode("12AU7")
        q = _find_dc_q_point(pts, ub=250.0, ra_dc=0.001, ug1_bias=-8.0)
        assert q is not None
        ua_q, ia_q = q
        assert abs(ua_q - 250.0) < 1.0, f"Ua_q={ua_q:.1f} should be ≈250V"
        assert ia_q > 0, f"Ia_q={ia_q:.2f} should be positive"

    def test_q_point_with_real_ra_dc(self):
        """With typical Ra_dc (0.1 kΩ = 100Ω), Q-point shifts slightly."""
        _, pts = quick_triode("12AU7")
        q = _find_dc_q_point(pts, ub=250.0, ra_dc=0.1, ug1_bias=-8.0)
        assert q is not None
        ua_q, ia_q = q
        # With 100Ω winding, Ua_q = Ub - Ia_q * 0.1
        # For Ia_q ≈ 10mA: Ua_q ≈ 250 - 1 = 249V
        assert 200 < ua_q < 260, f"Ua_q={ua_q:.1f} out of range"
        assert ia_q > 0

    def test_pentode_q_point(self):
        """Pentode Q-point should also work."""
        _, pts = quick_pentode("EL84")
        q = _find_dc_q_point(pts, ub=250.0, ra_dc=0.001, ug1_bias=-7.0)
        assert q is not None
        ua_q, ia_q = q
        assert abs(ua_q - 250.0) < 1.0
        # EL84 at Ug1=-7V should have reasonable plate current
        assert ia_q > 5.0, f"EL84 Ia_q={ia_q:.1f}mA at Ug1=-7V seems too low"

    def test_returns_none_for_empty(self):
        q = _find_dc_q_point([], ub=250.0, ra_dc=0.1, ug1_bias=-8.0)
        assert q is None


class TestTransformerACLoadLine:
    """Test transformer AC load line intersection finding."""

    def test_ac_mode_q_point_is_realistic(self):
        """With AC mode, intersection at bias Ug1 should be near Ub.

        The AC line through Q-point (Ua≈Ub) gives intersections in the
        real operating region, not at the far-left (low Ua) end.
        """
        _, pts = quick_pentode("EL84")
        ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=5.0)

        isects_ac = find_intersections(pts, ll, ug1_bias=-7.0)
        assert len(isects_ac) >= 3

        # Find intersection near bias
        near_bias = [p for p in isects_ac if abs(p["ug1"] - (-7.0)) < 1.0]
        if near_bias:
            # Q-point Ua should be near Ub for transformer coupling
            assert near_bias[0]["ua"] > 200, \
                f"Ua at bias={near_bias[0]['ua']:.0f}V should be near Ub=250V"

    def test_ac_intersections_extend_beyond_ub(self):
        """AC load line intersections can have Ua > Ub."""
        _, pts = quick_triode("12AU7")
        ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=5.0)
        isects = find_intersections(pts, ll, ug1_bias=-8.0)
        assert len(isects) >= 3

        ua_values = [p["ua"] for p in isects]
        # At least some intersections should be below Ub (positive swing)
        assert min(ua_values) < 250, "Should have intersections below Ub"
        # With AC line through Q-point at Ua=250V, cutoff curves
        # intersect at Ua > 250V (tube Ia drops to zero at high Ua)

    def test_triode_pout_higher_with_transformer(self):
        """Transformer coupling should give more Pout than resistive for same Ra.

        The AC line through Q≈Ub swings above Ub, so the tube sees the full
        supply as headroom instead of losing DC drop across Ra
        (probed: xfmr ≈ 345 mW vs resistive ≈ 247 mW, ratio ≈ 1.4×).
        """
        _, pts = quick_triode("12AU7")

        ll_res = ResistiveLoadLine(ub=250, ra=5.0)
        ll_xfmr = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=5.0)

        isects_res = find_intersections(pts, ll_res)
        isects_xfmr = find_intersections(pts, ll_xfmr, ug1_bias=-8.0)

        dist_res = compute_distortion(isects_res, ug1_bias=-8.0)
        dist_xfmr = compute_distortion(isects_xfmr, ug1_bias=-8.0)

        assert dist_res is not None, "resistive distortion must be computable"
        assert dist_xfmr is not None, "transformer distortion must be computable"
        assert dist_xfmr["pout_mw"] > dist_res["pout_mw"], \
            f"Xfmr Pout={dist_xfmr['pout_mw']:.0f} should exceed " \
            f"Res={dist_res['pout_mw']:.0f}"

    def test_backward_compatible_without_ug1_bias(self):
        """Without ug1_bias, transformer behaves as before (no AC line).

        Comparing the no-arg call against ug1_bias=None alone is vacuous
        (None IS the default) — the claim only means something if passing
        a real bias actually switches to the AC line and changes the result.
        """
        _, pts = quick_triode("12AU7")
        ll = TransformerLoadLine(ub=300, ra_dc=0.1, ra_ac=5.0)
        isects_old = find_intersections(pts, ll)
        isects_none = find_intersections(pts, ll, ug1_bias=None)
        assert len(isects_old) >= 3, "DC-line path must find intersections"
        assert len(isects_old) == len(isects_none)

        # AC mode must actually differ from the legacy DC path
        # (probed: 11 DC intersections down to Ua≈172 V vs 5 AC ones ≥220 V)
        isects_ac = find_intersections(pts, ll, ug1_bias=-8.0)
        dc_set = [(round(p["ug1"], 3), round(p["ua"], 3)) for p in isects_old]
        ac_set = [(round(p["ug1"], 3), round(p["ua"], 3)) for p in isects_ac]
        assert dc_set != ac_set, \
            "ug1_bias had no effect — AC load line path did not engage"

    def test_pentode_transformer_meaningful_pout(self):
        """EL84 pentode with transformer should produce meaningful power.

        Ub=200 V (not 250) because the quick_pentode grid tops out at
        Ua=300 V: with Ub=250 the AC line's negative-going swing (Ua > Ub)
        leaves the grid, the bias curve becomes the lowest intersection and
        half_swing degenerates to 0 → compute_distortion returns None.
        """
        _, pts = quick_pentode("EL84")
        ll = TransformerLoadLine(ub=200, ra_dc=0.05, ra_ac=5.0)
        isects = find_intersections(pts, ll, ug1_bias=-7.0)

        assert len(isects) >= 3, f"expected ≥3 intersections, got {len(isects)}"
        dist = compute_distortion(isects, ug1_bias=-7.0)
        assert dist is not None, "distortion must be computable (probed ≈390 mW)"
        assert dist["pout_mw"] > MIN_PENTODE_XFMR_POUT_MW, \
            f"EL84 pentode transformer Pout={dist['pout_mw']:.0f}mW too low"

    def test_sweep_amplitude_with_transformer(self):
        """sweep_amplitude should work with transformer AC load line.

        Ub=250 V / bias=-8 V (not 300/-10): the quick_triode grid tops out at
        Ua=300 V, so an AC line through Q at Ua≈300 has no negative-swing
        intersections and the sweep comes back empty.
        """
        _, pts = quick_triode("12AU7")
        ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=10.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-8.0)
        assert len(results) > 3, \
            f"transformer AC sweep must produce results, got {len(results)}"
        assert all(r["pout_mw"] > 0 for r in results)
        # Pout grows monotonically with drive over the swept (unclipped) range
        pouts = [r["pout_mw"] for r in results]
        assert all(a < b for a, b in zip(pouts, pouts[1:])), \
            "Pout should increase monotonically with swing"

    def test_optimize_bias_with_transformer(self):
        """optimize_bias should work with transformer load line."""
        _, pts = quick_triode("12AU7")
        ll = TransformerLoadLine(ub=300, ra_dc=0.05, ra_ac=5.0)
        opt = optimize_bias(pts, ll, target="min_thd")
        assert opt is not None
        assert opt["pout_mw"] > 0
        assert opt["rk_auto_bias"] > 0


class TestPushPullACLoadLine:
    """Test PP AC load line methods and integration."""

    def test_pp_ia_at_ua_ac(self):
        """PP AC load line through Q-point."""
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        ia = ll.ia_at_ua_ac(200, q_ua=250, q_ia=48.0)
        # Ia = 48 - (200 - 250) / 2.0 = 48 + 25 = 73
        assert abs(ia - 73.0) < 0.01

    def test_pp_ia_at_ua_dc(self):
        """PP DC load line."""
        ll = PushPullLoadLine(ub=250, ra_aa=8.0, ra_dc=0.1)
        ia = ll.ia_at_ua_dc(200)
        assert abs(ia - (250 - 200) / 0.1) < 0.01

    def test_pp_with_ac_mode(self):
        """PP with ug1_bias should use AC load line with realistic Q-point."""
        _, pts = quick_pentode("EL84")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0, ra_dc=0.05)
        isects_ac = find_intersections(pts, ll, ug1_bias=-7.0)
        assert len(isects_ac) >= 3, f"PP AC mode should find intersections, got {len(isects_ac)}"
        # Q-point should be near Ub
        near_bias = [p for p in isects_ac if abs(p["ug1"] - (-7.0)) < 1.5]
        if near_bias:
            assert near_bias[0]["ua"] > 200, "PP Q-point Ua should be near Ub"


class TestRealDataTransformer:
    """Test transformer AC load line with real measurement data (if available)."""

    def _load_real_pentode(self):
        """Try to load real EL84 pentode fixture data."""
        import json
        from tests._real_data import EL84_PENTODE_FILES, converted_path
        for name in EL84_PENTODE_FILES:
            path = converted_path(name)
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("topology") == TOPOLOGY_PENTODE:
                    return data, data.get("points", [])
            except Exception:
                continue
        return None, None

    def test_real_el84_pentode_transformer_pout(self):
        """Real EL84 pentode data with transformer should give meaningful Pout."""
        data, pts = self._load_real_pentode()
        assert not (not pts), "silent skip de-vacuated"

        # Find dominant Ug2
        ug2_vals = [p.get("ug2", 0) for p in pts if p.get("ug2")]
        assert not (not ug2_vals), "silent skip de-vacuated"
        ug2_main = max(set(ug2_vals), key=ug2_vals.count)

        ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=5.0)

        # Compare old vs AC mode
        isects_old = find_intersections(pts, ll, ug2_filter=ug2_main)
        isects_ac = find_intersections(pts, ll, ug2_filter=ug2_main, ug1_bias=-7.0)

        n_old = len(isects_old)
        n_ac = len(isects_ac)

        # AC mode should find significantly more intersections for pentode
        assert n_ac >= n_old, \
            f"AC found {n_ac} vs old {n_old} intersections"

        assert n_ac >= 3, "guard de-vacuated 2026-07-12: value must be present"
        dist = compute_distortion(isects_ac, ug1_bias=-7.0)
        if dist:
            # Real EL84 pentode with 5kΩ transformer should give > 100mW
            # (limited by scan range, but much better than resistive line)
            assert dist["pout_mw"] > 10, \
                f"Real EL84 pentode Pout={dist['pout_mw']:.0f}mW"

    def test_real_el84_triode_connected(self):
        """Real EL84 triode-connected data should give valid results.

        Every fixture present on disk must pass the full pipeline
        (probed: both files → 13/15 intersections, Pout ≈ 260 mW).
        Asserts live OUTSIDE any try/except so failures surface;
        only file reading is guarded (narrowly).
        """
        import json
        from tests._real_data import EL84_TRIODE_FILES, converted_path

        checked = 0
        for name in EL84_TRIODE_FILES:
            path = converted_path(name)
            if not path.exists():
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                continue  # unreadable file — treat like missing
            # A corrupt fixture is a broken checkout, not "no data" — fail loudly
            data = json.loads(raw)
            assert data.get("topology") == TOPOLOGY_TRIODE, \
                f"{name}: fixture is documented as triode-connected"
            pts = data.get("points", [])
            assert len(pts) >= 20, f"{name}: only {len(pts)} points"

            ll = TransformerLoadLine(ub=300, ra_dc=0.05, ra_ac=5.0)
            isects = find_intersections(pts, ll, ug1_bias=-7.0)
            assert len(isects) >= 3, f"{name}: only {len(isects)} intersections"
            dist = compute_distortion(isects, ug1_bias=-7.0)
            assert dist is not None, f"{name}: compute_distortion returned None"
            assert dist["pout_mw"] > 10, \
                f"{name}: Triode-connected Pout={dist['pout_mw']:.0f}mW"
            checked += 1

        if checked == 0:
            pytest.skip("no real EL84 triode fixtures on disk")
