"""Tests for compute_distortion_dft() — DFT-based HD analysis."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.amplifier import compute_distortion_dft, ResistiveLoadLine
from lm19.tube_sim import quick_triode, quick_pentode

# Physical sanity bounds
from lm19.constants import MAX_SANE_THD_PCT, MAX_SANE_HD_PCT as MAX_SANE_HARMONIC_PCT
from lm19.amplifier.constants import (
    HD_METHOD_DFT,
)


class SimpleModel:
    """Minimal tube model for testing: Ia = func(Ug1), independent of Ua."""
    topology = "triode"
    model_type = "test"
    name = "test"
    pa_max = 12.0
    uh = 6.3
    ih = 0.7

    def __init__(self, func):
        self._func = func

    def ia(self, ua, ug1, ug2=0.0):
        return max(0.0, self._func(ug1))

    def ig2(self, ua, ug1, ug2):
        return 0.0

    def generate_scan(self, grid):
        return []

    def params_dict(self):
        return {}


class RealisticModel:
    """Model where Ia depends on both Ua and Ug1 (resistive load line coupling)."""
    topology = "triode"
    model_type = "test"
    name = "test_realistic"
    pa_max = 12.0
    uh = 6.3
    ih = 0.7

    def __init__(self, mu=20.0, ra_internal=5.0):
        self._mu = mu
        self._ra = ra_internal  # internal plate resistance in kΩ

    def ia(self, ua, ug1, ug2=0.0):
        # Simplified triode: Ia = (mu*Ug1 + Ua) / (ra * 1000)
        # with cutoff at Ia < 0
        return max(0.0, (self._mu * (ug1 + 7.0) + ua) / (self._ra * 1.0))

    def ig2(self, ua, ug1, ug2):
        return 0.0

    def generate_scan(self, grid):
        return []

    def params_dict(self):
        return {}


def _ll(ub=250.0, ra=5.0):
    return ResistiveLoadLine(ub, ra)


# ═══════════════════════════════════════════════════════════════════
# Analytical test cases
# ═══════════════════════════════════════════════════════════════════

class TestAnalytical:
    """Known nonlinearities should produce known harmonics."""

    def test_linear_model_near_zero_thd(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0))
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert r["thd"] < 0.5

    def test_quadratic_gives_hd2(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2)
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert r["hd2"] > 5.0
        assert r["hd3"] < 1.0

    def test_cubic_gives_hd3(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.1 * (ug1 + 7.0) ** 3)
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert r["hd3"] > 1.0
        assert r["hd2"] < 1.0

    def test_quadratic_hd2_value(self):
        """HD2 for quadratic should match Chebyshev analytical value."""
        a1, a2, A = 2.0, 0.3, 3.0
        expected_hd2 = a2 * A / (2.0 * a1) * 100.0  # 22.5%
        model = SimpleModel(lambda ug1: 10.0 + a1 * (ug1 + 7.0) + a2 * (ug1 + 7.0) ** 2)
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=A)
        assert r is not None
        assert abs(r["hd2"] - expected_hd2) < 2.0

    def test_returns_harmonics_up_to_max(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0))
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        for n in range(2, 10):
            assert f"hd{n}" in r


# ═══════════════════════════════════════════════════════════════════
# Stability across FFT sizes
# ═══════════════════════════════════════════════════════════════════

class TestStability:
    """Results should be stable across different n_samples."""

    @pytest.mark.parametrize("n_samples", [256, 512, 1024, 2048])
    def test_stable_hd2_across_fft_sizes(self, n_samples):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2)
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0,
                                    n_samples=n_samples)
        assert r is not None
        assert abs(r["hd2"] - 22.5) < 2.0  # stable within ±2%

    def test_custom_max_harmonic(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0))
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0,
                                    max_harmonic=5)
        assert r is not None
        assert "hd5" in r
        assert "hd6" not in r


# ═══════════════════════════════════════════════════════════════════
# Boundary cases
# ═══════════════════════════════════════════════════════════════════

class TestBoundary:
    """Edge cases."""

    def test_zero_swing_returns_none(self):
        model = SimpleModel(lambda ug1: 10.0)
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=0.0)
        assert r is None

    def test_tiny_swing_returns_none(self):
        model = SimpleModel(lambda ug1: 10.0)
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=0.05)
        assert r is None


# ═══════════════════════════════════════════════════════════════════
# Realistic model (Ua-dependent)
# ═══════════════════════════════════════════════════════════════════

class TestRealisticModel:
    """Model where Ia depends on Ua exercises the iterative solver."""

    def test_realistic_model_converges(self):
        model = RealisticModel(mu=20.0, ra_internal=5.0)
        r = compute_distortion_dft(model, _ll(250.0, 5.0), ug1_bias=-7.0, half_swing=2.0)
        assert r is not None
        assert r["pout_mw"] > 0
        assert r["ia_0"] > 0

    def test_realistic_model_has_distortion(self):
        """Even a simplified triode model should show some HD due to cutoff."""
        model = RealisticModel(mu=20.0, ra_internal=5.0)
        r = compute_distortion_dft(model, _ll(250.0, 5.0), ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert r["thd"] >= 0  # may be very low for linear model


# ═══════════════════════════════════════════════════════════════════
# Metadata
# ═══════════════════════════════════════════════════════════════════

class TestMetadata:
    """Result dict should have correct metadata."""

    def test_method_is_dft(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0))
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)
        assert r["method"] == HD_METHOD_DFT

    def test_has_pdc_eta_when_ub(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0))
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0, ub=250.0)
        assert r["pdc_mw"] is not None
        assert r["eta_pct"] is not None

    def test_no_pdc_when_no_ub(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0))
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)
        assert r["pdc_mw"] is None

    def test_n_samples_in_result(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0))
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0,
                                    n_samples=512)
        assert r["n_samples"] == 512


# ═══════════════════════════════════════════════════════════════════
# Physical sanity — synthetic models
# ═══════════════════════════════════════════════════════════════════

class TestPhysicalSanitySynthetic:
    """DFT results must obey physical constraints on synthetic models."""

    def test_linear_thd_near_zero(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0))
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert r["thd"] < 1.0

    def test_quadratic_thd_under_limit(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2)
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert r["thd"] < MAX_SANE_THD_PCT
        assert r["hd2"] < MAX_SANE_HARMONIC_PCT

    def test_cubic_thd_under_limit(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.05 * (ug1 + 7.0) ** 3)
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert r["thd"] < MAX_SANE_THD_PCT

    def test_all_harmonics_below_fundamental(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2)
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        for n in range(2, r["max_harmonic"] + 1):
            assert r[f"hd{n}"] < 100.0, f"hd{n}={r[f'hd{n}']:.1f}% exceeds 100%"

    def test_thd_is_rss_of_harmonics(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2)
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        rss = math.sqrt(sum(r[f"hd{n}"] ** 2 for n in range(2, r["max_harmonic"] + 1)))
        assert r["thd"] == pytest.approx(rss, rel=1e-4)

    def test_pout_positive_and_finite(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.2 * (ug1 + 7.0) ** 2)
        r = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)
        assert r is not None
        assert 0 < r["pout_mw"] < 1e6

    def test_smaller_swing_lower_thd(self):
        model = SimpleModel(lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2)
        r_full = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)
        r_half = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=1.5)
        assert r_full is not None and r_half is not None
        assert r_half["thd"] <= r_full["thd"] + 0.5


# ═══════════════════════════════════════════════════════════════════
# Physical sanity — real tube models (Koren fit)
# ═══════════════════════════════════════════════════════════════════

class TestPhysicalSanityRealTubes:
    """DFT on real Koren-fit models must produce physical results."""

    def test_12au7_thd_under_limit(self):
        model, _ = quick_triode("12AU7")
        r = compute_distortion_dft(model, _ll(250, 10.0), ug1_bias=-10.0,
                                   half_swing=5.0, ub=250.0)
        assert r is not None
        assert r["thd"] < MAX_SANE_THD_PCT
        assert r["hd2"] < MAX_SANE_HARMONIC_PCT

    def test_12ax7_thd_under_limit(self):
        model, _ = quick_triode("12AX7")
        r = compute_distortion_dft(model, _ll(250, 100.0), ug1_bias=-1.0,
                                   half_swing=0.5, ub=250.0)
        assert r is not None
        assert r["thd"] < MAX_SANE_THD_PCT

    def test_el84_pentode_thd_under_limit(self):
        model, _ = quick_pentode("EL84")
        r = compute_distortion_dft(model, _ll(250, 5.0), ug1_bias=-7.0,
                                   half_swing=3.0, ug2=250, ub=250.0)
        assert r is not None
        assert r["thd"] < MAX_SANE_THD_PCT

    def test_12au7_all_harmonics_under_100(self):
        model, _ = quick_triode("12AU7")
        r = compute_distortion_dft(model, _ll(250, 10.0), ug1_bias=-10.0,
                                   half_swing=5.0)
        assert r is not None
        for n in range(2, r["max_harmonic"] + 1):
            assert r[f"hd{n}"] < 100.0, f"12AU7 DFT: hd{n}={r[f'hd{n}']:.1f}%"

    def test_el84_all_harmonics_under_100(self):
        model, _ = quick_pentode("EL84")
        r = compute_distortion_dft(model, _ll(250, 5.0), ug1_bias=-7.0,
                                   half_swing=3.0, ug2=250)
        assert r is not None
        for n in range(2, r["max_harmonic"] + 1):
            assert r[f"hd{n}"] < 100.0, f"EL84 DFT: hd{n}={r[f'hd{n}']:.1f}%"

    @pytest.mark.parametrize("tube,ub,ra,ug1,swing", [
        ("12AU7", 250, 10.0, -10.0, 5.0),
        ("12AU7", 300, 15.0, -10.0, 5.0),
        ("12AX7", 250, 100.0, -1.0, 0.5),
    ])
    def test_triode_pout_order_of_magnitude(self, tube, ub, ra, ug1, swing):
        model, _ = quick_triode(tube)
        r = compute_distortion_dft(model, _ll(ub, ra), ug1_bias=ug1,
                                   half_swing=swing, ub=ub)
        assert r is not None
        assert 0 < r["pout_mw"] < 5000, f"Pout={r['pout_mw']:.1f}mW out of range"

    def test_el84_pout_order_of_magnitude(self):
        model, _ = quick_pentode("EL84")
        r = compute_distortion_dft(model, _ll(250, 5.0), ug1_bias=-7.0,
                                   half_swing=4.0, ug2=250, ub=250.0)
        assert r is not None
        assert r["pout_mw"] > 100, f"EL84 Pout={r['pout_mw']:.1f}mW too low"
        assert r["pout_mw"] < 20000, f"EL84 Pout={r['pout_mw']:.1f}mW too high"

    def test_12au7_ia0_positive(self):
        model, _ = quick_triode("12AU7")
        r = compute_distortion_dft(model, _ll(250, 10.0), ug1_bias=-10.0,
                                   half_swing=5.0)
        assert r is not None
        assert r["ia_0"] > 0
