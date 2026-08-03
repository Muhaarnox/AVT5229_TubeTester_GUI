"""Tests for compute_distortion_chebyshev() — Chebyshev polynomial HD analysis."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.amplifier import (
    compute_distortion_chebyshev,
    compute_distortion,
    find_intersections,
    ResistiveLoadLine,
)
from lm19.tube_sim import quick_triode, quick_pentode

# Physical sanity bounds
from lm19.constants import MAX_SANE_THD_PCT, MAX_SANE_HD_PCT as MAX_SANE_HARMONIC_PCT
from lm19.amplifier.constants import (
    HD_METHOD_CHEBYSHEV,
)


def _make_pts_from_func(func, ug1_range=(-10.0, -1.0), n=20, ub=250.0, ra=5.0):
    """Generate intersection points from Ia = func(Ug1) on resistive load line."""
    pts = []
    for i in range(n):
        ug1 = ug1_range[0] + (ug1_range[1] - ug1_range[0]) * i / (n - 1)
        ia = max(0.0, func(ug1))
        ua = ub - ia * ra
        pts.append({"ug1": ug1, "ua": ua, "ia": ia})
    return pts


# ═══════════════════════════════════════════════════════════════════
# Analytical test cases
# ═══════════════════════════════════════════════════════════════════

class TestAnalytical:
    """Known nonlinearities should produce known harmonics."""

    def test_linear_gives_zero_thd(self):
        """Linear Ia(Ug1) → no distortion."""
        pts = _make_pts_from_func(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0), n=20)
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert r["thd"] < 0.5  # near-zero

    def test_quadratic_gives_hd2_only(self):
        """Quadratic Ia(Ug1) → HD2 dominant, HD3+ ≈ 0."""
        pts = _make_pts_from_func(
            lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2,
            n=20,
        )
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert r["hd2"] > 5.0  # significant HD2
        assert r["hd3"] < 1.0  # negligible HD3
        assert r["hd4"] < 1.0

    def test_cubic_gives_hd3_only(self):
        """Cubic Ia(Ug1) → HD3 dominant, HD2/HD4 ≈ 0."""
        pts = _make_pts_from_func(
            lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.1 * (ug1 + 7.0) ** 3,
            n=20,
        )
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert r["hd3"] > 1.0
        assert r["hd2"] < 1.0  # even harmonics small for odd nonlinearity
        assert r["hd4"] < 1.0

    def test_quadratic_hd2_value(self):
        """Check HD2 value for known quadratic: Ia = a0 + a1*x + a2*x².
        For Chebyshev: HD2 = |c2/c1| where c2 = a2*A/2, c1 = a1.
        So HD2 ≈ a2*A / (2*a1) * 100%.
        """
        a1, a2, A = 2.0, 0.3, 3.0
        expected_hd2 = a2 * A / (2.0 * a1) * 100.0  # = 22.5%
        pts = _make_pts_from_func(
            lambda ug1: 10.0 + a1 * (ug1 + 7.0) + a2 * (ug1 + 7.0) ** 2,
            n=20,
        )
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=A)
        assert r is not None
        assert abs(r["hd2"] - expected_hd2) < 2.0  # within 2%

    def test_returns_harmonics_up_to_max(self):
        """Result contains hd2 through hd9 with enough points."""
        pts = _make_pts_from_func(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0), n=30)
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        for n in range(2, 10):
            assert f"hd{n}" in r


# ═══════════════════════════════════════════════════════════════════
# Boundary cases
# ═══════════════════════════════════════════════════════════════════

class TestBoundary:
    """Edge cases and insufficient data."""

    def test_too_few_points_returns_none(self):
        pts = [{"ug1": -10, "ua": 200, "ia": 5}, {"ug1": -5, "ua": 180, "ia": 10}]
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=2.0)
        assert r is None

    def test_zero_swing_uses_max_symmetric(self):
        """half_swing=0 falls through to auto max symmetric swing."""
        pts = _make_pts_from_func(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0), n=20)
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=0.0)
        assert r is not None
        assert r["half_swing"] > 0.1

    def test_tiny_swing_is_an_error_not_auto(self):
        """ML-051: a positive manual swing below MIN_SWING_V is a user
        error, not a silent fall-back to the auto maximum. Exactly 0.0
        remains the "auto" convention — see
        test_zero_swing_uses_max_symmetric."""
        pts = _make_pts_from_func(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0), n=20)
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=0.05)
        assert r is None

    def test_flat_characteristic_near_zero_thd(self):
        """Constant Ia → b1 ≈ 0 → None (no signal)."""
        pts = [{"ug1": -10.0 + i, "ua": 200.0, "ia": 10.0} for i in range(15)]
        r = compute_distortion_chebyshev(pts, ug1_bias=-5.5, half_swing=3.0)
        assert r is None  # b1 ≈ 0

    def test_no_bias_uses_midpoint(self):
        pts = _make_pts_from_func(lambda ug1: 10.0 + 2.0 * (ug1 + 5.5), n=20)
        r = compute_distortion_chebyshev(pts, ug1_bias=None, half_swing=3.0)
        assert r is not None

    def test_no_swing_uses_max_symmetric(self):
        pts = _make_pts_from_func(lambda ug1: 10.0 + 2.0 * (ug1 + 5.5), n=20)
        r = compute_distortion_chebyshev(pts, ug1_bias=-5.5)
        assert r is not None
        assert r["half_swing"] > 0.1


# ═══════════════════════════════════════════════════════════════════
# Stability with different point counts
# ═══════════════════════════════════════════════════════════════════

class TestStability:
    """Results should be stable across different point densities."""

    @pytest.mark.parametrize("n_pts", [15, 20, 30, 50])
    def test_stable_hd2_with_varying_points(self, n_pts):
        func = lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2
        pts = _make_pts_from_func(func, n=n_pts)
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert abs(r["hd2"] - 22.5) < 3.0  # stable within ±3%

    def test_11_points_with_lower_max_harmonic(self):
        """11 points can still work with max_harmonic=5."""
        func = lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2
        pts = _make_pts_from_func(func, n=11)
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0, max_harmonic=5)
        assert r is not None
        assert abs(r["hd2"] - 22.5) < 5.0

    def test_custom_max_harmonic(self):
        pts = _make_pts_from_func(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0), n=20)
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0, max_harmonic=5)
        assert r is not None
        assert "hd5" in r
        assert "hd6" not in r
        assert r["max_harmonic"] == 5


# ═══════════════════════════════════════════════════════════════════
# Metadata
# ═══════════════════════════════════════════════════════════════════

class TestMetadata:
    """Result dict should have correct metadata."""

    def test_method_is_chebyshev(self):
        pts = _make_pts_from_func(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0), n=20)
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r["method"] == HD_METHOD_CHEBYSHEV

    def test_has_pout(self):
        pts = _make_pts_from_func(
            lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2, n=20
        )
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r["pout_mw"] > 0

    def test_has_pdc_eta_when_ub_given(self):
        pts = _make_pts_from_func(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0), n=20)
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0, ub=250.0)
        assert r["pdc_mw"] is not None
        assert r["eta_pct"] is not None

    def test_no_pdc_when_no_ub(self):
        pts = _make_pts_from_func(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0), n=20)
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r["pdc_mw"] is None

    def test_q_point_values(self):
        pts = _make_pts_from_func(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0), n=20)
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert abs(r["ug1_0"] - (-7.0)) < 0.1
        assert r["ia_0"] > 0
        assert r["ua_0"] > 0


# ═══════════════════════════════════════════════════════════════════
# Physical sanity — synthetic data
# ═══════════════════════════════════════════════════════════════════

class TestPhysicalSanitySynthetic:
    """Chebyshev results must obey physical constraints on synthetic data."""

    def test_linear_thd_near_zero(self):
        """Linear characteristic → THD ≈ 0%."""
        pts = _make_pts_from_func(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0), n=25)
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert r["thd"] < 1.0  # essentially zero, but allow numerical noise

    def test_quadratic_thd_under_limit(self):
        """Quadratic (moderate) → THD well below 100%."""
        pts = _make_pts_from_func(
            lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2, n=25,
        )
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert r["thd"] < MAX_SANE_THD_PCT
        assert r["hd2"] < MAX_SANE_HARMONIC_PCT

    def test_cubic_thd_under_limit(self):
        """Cubic nonlinearity → THD under limit."""
        pts = _make_pts_from_func(
            lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.05 * (ug1 + 7.0) ** 3, n=25,
        )
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert r["thd"] < MAX_SANE_THD_PCT

    def test_all_harmonics_below_fundamental(self):
        """No individual harmonic should exceed 100% (of fundamental)."""
        pts = _make_pts_from_func(
            lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2, n=25,
        )
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        for n in range(2, r["max_harmonic"] + 1):
            assert r[f"hd{n}"] < 100.0, f"hd{n}={r[f'hd{n}']:.1f}% exceeds 100%"

    def test_thd_is_rss_of_harmonics(self):
        """THD = sqrt(sum(HDn²)) — verify formula consistency."""
        pts = _make_pts_from_func(
            lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2, n=25,
        )
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        rss = math.sqrt(sum(r[f"hd{n}"] ** 2 for n in range(2, r["max_harmonic"] + 1)))
        assert r["thd"] == pytest.approx(rss, rel=1e-6)

    def test_pout_positive_and_finite(self):
        """Pout must be positive and finite."""
        pts = _make_pts_from_func(
            lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.2 * (ug1 + 7.0) ** 2, n=25,
        )
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert 0 < r["pout_mw"] < 1e6  # finite and positive

    def test_ia0_positive(self):
        """Q-point current must be positive."""
        pts = _make_pts_from_func(
            lambda ug1: 10.0 + 2.0 * (ug1 + 7.0), n=25,
        )
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert r["ia_0"] > 0

    @pytest.mark.parametrize("n_pts", [8, 10, 12, 15, 20, 30])
    def test_thd_stable_across_point_counts(self, n_pts):
        """THD should be under physical limit regardless of point count."""
        pts = _make_pts_from_func(
            lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2,
            n=n_pts,
        )
        r = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        assert r is not None, f"chebyshev must compute on clean synthetic n={n_pts}"
        assert r["thd"] < MAX_SANE_THD_PCT, (
            f"n_pts={n_pts}: THD={r['thd']:.1f}% exceeds {MAX_SANE_THD_PCT}%"
        )

    def test_smaller_swing_lower_thd(self):
        """Reducing swing should reduce or maintain THD (less nonlinearity)."""
        pts = _make_pts_from_func(
            lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2, n=25,
        )
        r_full = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        r_half = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=1.5)
        assert r_full is not None and r_half is not None
        assert r_half["thd"] <= r_full["thd"] + 0.5  # small tolerance


# ═══════════════════════════════════════════════════════════════════
# Physical sanity — real tube models (quick_triode / quick_pentode)
# ═══════════════════════════════════════════════════════════════════

class TestPhysicalSanityRealTubes:
    """Chebyshev on real tube model data must produce physical results."""

    def _run_chebyshev(self, tube_name, ub, ra, ug1_bias, half_swing,
                       pentode=False, ug2=None):
        """Helper: generate intersections and run Chebyshev."""
        if pentode:
            _, pts = quick_pentode(tube_name)
        else:
            _, pts = quick_triode(tube_name)
        ll = ResistiveLoadLine(ub, ra)
        ug2_filter = ug2 if ug2 and ug2 > 0 else None
        isects = find_intersections(pts, ll, ug2_filter=ug2_filter)
        if len(isects) < 4:
            return None
        return compute_distortion_chebyshev(
            isects, ug1_bias=ug1_bias, half_swing=half_swing, ub=ub,
        )

    def test_12au7_thd_under_limit(self):
        """12AU7 at max symmetric swing — THD must be physical."""
        r = self._run_chebyshev("12AU7", ub=250, ra=10.0, ug1_bias=-10.0, half_swing=None)
        assert r is not None
        assert r["thd"] < MAX_SANE_THD_PCT
        assert r["hd2"] < MAX_SANE_HARMONIC_PCT

    def test_12ax7_thd_under_limit(self):
        """12AX7 at max symmetric swing — THD must be physical."""
        r = self._run_chebyshev("12AX7", ub=250, ra=100.0, ug1_bias=-1.0, half_swing=None)
        assert r is not None
        assert r["thd"] < MAX_SANE_THD_PCT

    def test_el84_pentode_thd_under_limit(self):
        r = self._run_chebyshev("EL84", ub=250, ra=5.0, ug1_bias=-7.0,
                                half_swing=None, pentode=True, ug2=250)
        assert r is not None
        assert r["thd"] < MAX_SANE_THD_PCT

    def test_12au7_chebyshev_vs_5point_agree(self):
        """Chebyshev HD2 should be in the same ballpark as 5-point HD2."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(250, 10.0)
        isects = find_intersections(pts, ll)
        # Use max symmetric swing (None) — both methods use same strategy
        cheb = compute_distortion_chebyshev(isects, ug1_bias=-10.0)
        five = compute_distortion(isects, ug1_bias=-10.0)
        assert cheb is not None and five is not None
        # Both should report same order of magnitude for HD2
        assert abs(cheb["hd2"] - five["hd2"]) < max(cheb["hd2"], five["hd2"]) * 0.5 + 1.0

    def test_12au7_all_harmonics_under_100(self):
        """No harmonic should exceed fundamental on real tube data."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(250, 10.0)
        isects = find_intersections(pts, ll)
        r = compute_distortion_chebyshev(isects, ug1_bias=-10.0)
        assert r is not None
        for n in range(2, r["max_harmonic"] + 1):
            assert r[f"hd{n}"] < 100.0, f"12AU7: hd{n}={r[f'hd{n}']:.1f}%"

    def test_el84_all_harmonics_under_100(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(250, 5.0)
        isects = find_intersections(pts, ll, ug2_filter=250)
        r = compute_distortion_chebyshev(isects, ug1_bias=-7.0)
        assert r is not None
        for n in range(2, r["max_harmonic"] + 1):
            assert r[f"hd{n}"] < 100.0, f"EL84: hd{n}={r[f'hd{n}']:.1f}%"

    @pytest.mark.parametrize("tube,ub,ra,ug1", [
        ("12AU7", 250, 10.0, -10.0),
        ("12AU7", 300, 15.0, -10.0),
        ("12AX7", 250, 100.0, -1.0),
    ])
    def test_triode_pout_order_of_magnitude(self, tube, ub, ra, ug1):
        """Pout should be in milliwatts range for preamp triodes."""
        r = self._run_chebyshev(tube, ub=ub, ra=ra, ug1_bias=ug1, half_swing=None)
        assert r is not None, f"chebyshev must compute for {tube} at documented op"
        assert 0 < r["pout_mw"] < 5000, f"Pout={r['pout_mw']:.1f}mW out of range"

    def test_el84_pout_order_of_magnitude(self):
        """EL84 Pout should be in watts range."""
        r = self._run_chebyshev("EL84", ub=250, ra=5.0, ug1_bias=-7.0,
                                half_swing=None, pentode=True, ug2=250)
        assert r is not None, "chebyshev must compute for EL84 at documented op"
        assert r["pout_mw"] > 100, f"EL84 Pout={r['pout_mw']:.1f}mW too low"
        assert r["pout_mw"] < 20000, f"EL84 Pout={r['pout_mw']:.1f}mW too high"
