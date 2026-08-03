"""Tests for tube simulator (tube_sim.py).

Run:  py -m pytest tests/test_tube_sim.py -v
"""

import numpy as np
import pytest

from lm19.tube_sim import (
    TubeModel, ScanGrid, load_model, fit_koren,
    quick_triode, quick_pentode,
    TRIODE_PRESETS, PENTODE_PRESETS,
)
from lm19.tube_params import KorenParams
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)
from lm19.tube_model_base import (
    MODEL_TYPE_KOREN,
)


class TestTubeModel:
    """Tests for TubeModel core methods."""

    def test_triode_ia_positive(self):
        model = load_model("12AX7")
        ia = model.ia(ua=250, ug1=-2.0)
        assert ia > 0
        assert ia < 10

    def test_triode_ia_cutoff(self):
        model = load_model("12AX7")
        ia = model.ia(ua=100, ug1=-10.0)
        assert ia < 0.01

    def test_triode_ia_increases_with_ua(self):
        model = load_model("12AU7")
        ia_100 = model.ia(ua=100, ug1=-5.0)
        ia_200 = model.ia(ua=200, ug1=-5.0)
        assert ia_200 > ia_100

    def test_triode_ia_increases_with_ug1(self):
        model = load_model("12AU7")
        ia_neg8 = model.ia(ua=200, ug1=-8.0)
        ia_neg2 = model.ia(ua=200, ug1=-2.0)
        assert ia_neg2 > ia_neg8

    def test_pentode_ia_positive(self):
        model = load_model("EL84")
        ia = model.ia(ua=250, ug1=-7.0, ug2=250)
        assert ia > 0
        assert ia < 100

    def test_pentode_ig2_positive(self):
        model = load_model("EL34")
        ig2 = model.ig2(ua=200.0, ug1=-10.0, ug2=250)
        assert ig2 > 0

    def test_triode_ig2_zero(self):
        model = load_model("12AX7")
        assert model.ig2(ua=200.0, ug1=-2.0, ug2=0) == 0.0

    def test_topology_correct(self):
        assert load_model("12AX7").topology == TOPOLOGY_TRIODE
        assert load_model("EL34").topology == TOPOLOGY_PENTODE


class TestScanGeneration:
    """Tests for generate_scan."""

    def test_triode_scan_point_count(self):
        model = load_model("12AX7")
        grid = ScanGrid(ua=(0, 300, 100), ug1=(-4, 0, 2))
        pts = model.generate_scan(grid)
        assert len(pts) == 4 * 3

    def test_pentode_scan_point_count(self):
        model = load_model("EL34")
        grid = ScanGrid(ua=(0, 200, 100), ug1=(-10, 0, 5), ug2=(200, 300, 100))
        pts = model.generate_scan(grid)
        assert len(pts) == 3 * 3 * 2

    def test_scan_format(self):
        _model, pts = quick_triode("12AX7")
        required_keys = {"ua", "ug1", "ug2", "ia", "ig2", "uh", "ih"}
        for p in pts:
            assert required_keys.issubset(p.keys())

    def test_triode_connected_pentode(self):
        model = load_model("EL34")
        grid = ScanGrid(
            ua=(50, 200, 50), ug1=(-20, -10, 5),
            ug2_track_ua=True, ug2_offset=0,
        )
        pts = model.generate_scan(grid)
        for p in pts:
            assert abs(p["ug2"] - p["ua"]) < 0.01

    def test_pentode_multi_ug2(self):
        model = load_model("EL84")
        grid = PENTODE_PRESETS["EL84_multi_ug2"]
        pts = model.generate_scan(grid)
        ug2_values = sorted(set(round(p["ug2"]) for p in pts))
        assert len(ug2_values) >= 3

    def test_all_ia_non_negative(self):
        _model, pts = quick_triode("12AU7")
        assert all(p["ia"] >= 0 for p in pts)

    def test_all_ig2_non_negative(self):
        _model, pts = quick_pentode("EL34")
        assert all(p["ig2"] >= 0 for p in pts)


class TestNoise:
    """Tests for add_noise."""

    def test_noise_preserves_count(self):
        model, pts = quick_triode("12AX7")
        noisy = model.add_noise(pts, sigma_pct=1.0, seed=42)
        assert len(noisy) == len(pts)

    def test_noise_changes_values(self):
        model, pts = quick_triode("12AX7")
        noisy = model.add_noise(pts, sigma_pct=2.0, seed=42)
        diffs = [abs(n["ia"] - o["ia"]) for n, o in zip(noisy, pts) if o["ia"] > 0.1]
        assert max(diffs) > 0

    def test_noise_reproducible(self):
        model, pts = quick_triode("12AX7")
        n1 = model.add_noise(pts, sigma_pct=1.0, seed=123)
        n2 = model.add_noise(pts, sigma_pct=1.0, seed=123)
        assert all(a["ia"] == b["ia"] for a, b in zip(n1, n2))

    def test_noise_non_negative(self):
        model, pts = quick_triode("12AX7")
        noisy = model.add_noise(pts, sigma_pct=5.0, seed=42)
        assert all(p["ia"] >= 0 for p in noisy)


class TestLoadModel:
    """Tests for load_model with aliases."""

    def test_load_by_primary_name(self):
        model = load_model("12AX7")
        assert model is not None
        assert model.name == "12AX7"

    def test_load_by_alias(self):
        model = load_model("ECC83")
        assert model is not None
        assert model.name == "12AX7"

    def test_load_pentode(self):
        model = load_model("6P14P")
        assert model is not None
        assert model.topology == TOPOLOGY_PENTODE

    def test_load_unknown_returns_none(self):
        assert load_model("NONEXISTENT_TUBE") is None

    def test_all_triode_presets_loadable(self):
        for name in TRIODE_PRESETS:
            model = load_model(name)
            assert model is not None, f"Cannot load {name}"

    def test_all_pentode_presets_loadable(self):
        for name in PENTODE_PRESETS:
            if name.endswith("_triode_connected") or name.endswith("_multi_ug2"):
                base = name.split("_")[0]
            else:
                base = name
            model = load_model(base)
            assert model is not None, f"Cannot load {base}"


class TestRoundTrip:
    """Validate that sim -> fit -> compare gives consistent parameters."""

    def test_triode_roundtrip_12ax7(self):
        """Generate 12AX7 curves from Koren params, fit back, check match."""
        try:
            from lm19.spice_export import _fit_koren_scipy
        except ImportError:
            return

        try:
            from scipy.optimize import least_squares  # noqa: F401
        except ImportError:
            return

        model, pts = quick_triode("12AX7")
        ua = np.array([p["ua"] for p in pts])
        ug1 = np.array([p["ug1"] for p in pts])
        ia = np.array([p["ia"] for p in pts]) / 1000.0

        params, cost, _ = _fit_koren_scipy(ua, ug1, ia, ref_koren=None)
        mu_fit = params[0]

        k = model.koren
        assert abs(mu_fit - k.mu) / k.mu < 0.3

        rms = np.sqrt(cost / len(ua)) * 1000
        assert rms < 0.5


class TestQuickHelpers:
    """Tests for quick_triode / quick_pentode convenience functions."""

    def test_quick_triode_default(self):
        model, pts = quick_triode()
        assert model.name == "12AX7"
        assert len(pts) > 0

    def test_quick_pentode_default(self):
        model, pts = quick_pentode()
        assert model.name == "EL84"
        assert len(pts) > 0

    def test_quick_triode_6sn7(self):
        model, pts = quick_triode("6SN7")
        assert model.topology == TOPOLOGY_TRIODE
        assert any(p["ia"] > 1 for p in pts)

    def test_quick_pentode_kt88(self):
        model, pts = quick_pentode("KT88")
        assert model.topology == TOPOLOGY_PENTODE
        assert len(pts) > 50

    def test_quick_triode_unknown_raises(self):
        try:
            quick_triode("NOPE")
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_quick_pentode_unknown_raises(self):
        try:
            quick_pentode("NOPE")
            assert False, "Should have raised"
        except ValueError:
            pass


# ===================================================================
# Koren pentode round-trip fit
# ===================================================================

class TestKorenPentodeRoundTrip:
    """Fit Koren pentode model on synthetic EL84 data."""

    @pytest.fixture(scope="class")
    def pentode_fit(self):
        _, pts = quick_pentode("EL84")
        return fit_koren(pts, "pentode")

    def test_fit_returns_result(self, pentode_fit):
        assert pentode_fit is not None
        assert pentode_fit.model_type == MODEL_TYPE_KOREN
        assert pentode_fit.topology == TOPOLOGY_PENTODE

    def test_rms_error_low(self, pentode_fit):
        """RMS on synthetic data should be very low."""
        assert pentode_fit.rms_error < 2.0, \
            f"RMS = {pentode_fit.rms_error:.2f} mA"

    def test_max_error_bounded(self, pentode_fit):
        assert pentode_fit.max_error < 10.0, \
            f"Max = {pentode_fit.max_error:.2f} mA"

    def test_mu_recovery(self, pentode_fit):
        """Fitted µ should be close to EL84 reference (19.5)."""
        mu = pentode_fit.params["mu"]
        assert abs(mu - 19.5) / 19.5 < 0.5, f"µ = {mu:.1f}"

    def test_n_points(self, pentode_fit):
        assert pentode_fit.n_points > 10

    def test_model_usable(self, pentode_fit):
        """Fitted model should produce reasonable Ia."""
        ia = pentode_fit.model.ia(250, -7, 250)
        assert 5.0 < ia < 100.0


# ===================================================================
# Koren Ig2 error fields
# ===================================================================

class TestKorenIg2Errors:
    """Verify rms_ig2 / max_ig2 are populated for pentode fits."""

    @pytest.fixture(scope="class")
    def pentode_fit(self):
        _, pts = quick_pentode("EL84")
        return fit_koren(pts, "pentode")

    def test_ig2_errors_present(self, pentode_fit):
        """Pentode fit should have Ig2 errors."""
        assert pentode_fit.rms_ig2 is not None
        assert pentode_fit.max_ig2 is not None

    def test_ig2_rms_positive(self, pentode_fit):
        assert pentode_fit.rms_ig2 >= 0

    def test_ig2_max_ge_rms(self, pentode_fit):
        assert pentode_fit.max_ig2 >= pentode_fit.rms_ig2

    def test_triode_no_ig2_errors(self):
        """Triode fit should NOT have Ig2 errors."""
        _, pts = quick_triode("12AX7")
        result = fit_koren(pts, "triode")
        assert result.rms_ig2 is None
        assert result.max_ig2 is None


# ===================================================================
# Koren noisy data
# ===================================================================

class TestKorenNoisyData:
    """Koren fitter robustness on noisy data."""

    @pytest.fixture(scope="class")
    def noisy_triode_fit(self):
        rng = np.random.RandomState(42)
        _, pts = quick_triode("12AX7")
        for p in pts:
            noise = 1.0 + rng.normal(0, 0.02)  # ±2%
            p["ia"] = max(p["ia"] * noise, 0.0)
        return fit_koren(pts, "triode")

    def test_rms_bounded(self, noisy_triode_fit):
        assert noisy_triode_fit.rms_error < 0.5, \
            f"RMS = {noisy_triode_fit.rms_error:.2f} mA"

    def test_mu_recovery(self, noisy_triode_fit):
        mu = noisy_triode_fit.params["mu"]
        assert abs(mu - 100.0) / 100.0 < 0.3, f"µ = {mu:.1f}"

    @pytest.fixture(scope="class")
    def noisy_pentode_fit(self):
        rng = np.random.RandomState(123)
        _, pts = quick_pentode("EL84")
        for p in pts:
            p["ia"] = max(p["ia"] * (1.0 + rng.normal(0, 0.03)), 0.0)
            p["ig2"] = max(p.get("ig2", 0) * (1.0 + rng.normal(0, 0.03)), 0.0)
        return fit_koren(pts, "pentode")

    def test_pentode_noisy_rms(self, noisy_pentode_fit):
        assert noisy_pentode_fit.rms_error < 3.0, \
            f"RMS = {noisy_pentode_fit.rms_error:.2f} mA"

    def test_pentode_noisy_mu(self, noisy_pentode_fit):
        mu = noisy_pentode_fit.params["mu"]
        assert 5.0 < mu < 50.0, f"µ = {mu:.1f}"


# ===================================================================
# Koren numerical safety
# ===================================================================

class TestKorenNumericalSafety:
    """Edge cases: Ua=0, extreme Ug1, extreme Ua."""

    def test_ua_zero_triode(self):
        """Ua=0 should not crash, Ia should be ~0."""
        model = load_model("12AX7")
        ia = model.ia(ua=0.0, ug1=-2.0)
        assert np.isfinite(ia)
        assert ia >= 0

    def test_ua_zero_pentode(self):
        """Ua=0 pentode should not crash."""
        model = load_model("EL84")
        ia = model.ia(ua=0.0, ug1=-7.0, ug2=250.0)
        assert np.isfinite(ia)
        assert ia >= 0

    def test_ug2_zero_pentode(self):
        """Ug2=0 should not crash."""
        model = load_model("EL84")
        ia = model.ia(ua=250.0, ug1=-7.0, ug2=0.0)
        assert np.isfinite(ia)

    def test_large_negative_ug1(self):
        """Very negative Ug1 should give ~0 current, not NaN."""
        model = load_model("12AX7")
        ia = model.ia(ua=300.0, ug1=-100.0)
        assert np.isfinite(ia)
        assert ia >= 0
        assert ia < 0.01

    def test_positive_ug1(self):
        """Positive Ug1 should not overflow."""
        model = load_model("12AX7")
        ia = model.ia(ua=300.0, ug1=5.0)
        assert np.isfinite(ia)
        assert ia > 0

    def test_very_large_ua(self):
        """Ua=1000V should not overflow."""
        model = load_model("EL34")
        ia = model.ia(ua=1000.0, ug1=-10.0, ug2=300.0)
        assert np.isfinite(ia)
        assert ia > 0


# ===================================================================
# Koren current conservation (pentode)
# ===================================================================

class TestKorenCurrentConservation:
    """Ia and Ig2 should both be non-negative for pentodes."""

    def test_ia_non_negative(self):
        model = load_model("EL84")
        for ug1 in [-15, -10, -5, -2, 0]:
            ia = model.ia(ua=250.0, ug1=float(ug1), ug2=250.0)
            assert ia >= 0, f"Negative Ia at Ug1={ug1}"

    def test_ig2_non_negative(self):
        model = load_model("EL34")
        for ug1 in [-30, -20, -10, -5, 0]:
            ig2 = model.ig2(ua=300.0, ug1=float(ug1), ug2=300.0)
            assert ig2 >= 0, f"Negative Ig2 at Ug1={ug1}"

    def test_ig2_decreases_with_negative_ug1(self):
        """More negative Ug1 → less Ig2."""
        model = load_model("EL34")
        ig2_0 = model.ig2(ua=300.0, ug1=0.0, ug2=300.0)
        ig2_20 = model.ig2(ua=300.0, ug1=-20.0, ug2=300.0)
        assert ig2_0 > ig2_20


# ===================================================================
# Koren fit edge cases
# ===================================================================

class TestKorenFitEdgeCases:
    """Fit edge cases: triode_connected, no-Ig2, too few points."""

    def test_triode_connected(self):
        """triode_connected topology should use triode fitter."""
        _, pts = quick_triode("12AX7")
        result = fit_koren(pts, "triode_connected")
        assert result is not None
        assert result.topology == TOPOLOGY_TRIODE
        assert result.rms_error < 1.0

    def test_pentode_no_ig2(self):
        """Pentode fit should work with ig2=0 (unmeasured)."""
        _, pts = quick_pentode("EL84")
        for p in pts:
            p["ig2"] = 0.0
        result = fit_koren(pts, "pentode")
        assert result is not None
        assert result.rms_ig2 is None  # no Ig2 data → no Ig2 errors

    def test_too_few_points_triode(self):
        """Should raise with <10 valid points."""
        pts = [{"ua": 200, "ug1": -2, "ia": 1.0}] * 5
        with pytest.raises(RuntimeError):
            fit_koren(pts, "triode")

    def test_too_few_points_pentode(self):
        pts = [{"ua": 200, "ug1": -5, "ug2": 250,
                "ia": 10.0, "ig2": 2.0}] * 5
        with pytest.raises(RuntimeError):
            fit_koren(pts, "pentode")


# ===================================================================
# Koren fit parameter recovery
# ===================================================================

class TestKorenParamRecovery:
    """Verify fitted parameters are close to reference values."""

    @pytest.fixture(scope="class")
    def triode_fit(self):
        _, pts = quick_triode("12AX7")
        return fit_koren(pts, "triode")

    @pytest.fixture(scope="class")
    def pentode_fit(self):
        _, pts = quick_pentode("EL84")
        return fit_koren(pts, "pentode")

    def test_triode_mu(self, triode_fit):
        mu = triode_fit.params["mu"]
        assert abs(mu - 100.0) / 100.0 < 0.3, f"µ = {mu:.1f}"

    def test_triode_ex(self, triode_fit):
        ex = triode_fit.params["ex"]
        assert 1.0 < ex < 2.0, f"ex = {ex:.2f}"

    def test_triode_kp(self, triode_fit):
        kp = triode_fit.params["kp"]
        assert 100 < kp < 2000, f"kp = {kp:.1f}"

    def test_pentode_mu(self, pentode_fit):
        mu = pentode_fit.params["mu"]
        assert 5.0 < mu < 50.0, f"µ = {mu:.1f}"

    def test_pentode_kvb(self, pentode_fit):
        kvb = pentode_fit.params["kvb"]
        assert kvb > 0, f"kvb = {kvb:.1f}"

    def test_pentode_kg2(self, pentode_fit):
        kg2 = pentode_fit.params.get("kg2")
        assert kg2 is not None
        assert kg2 > 0, f"kg2 = {kg2:.1f}"
