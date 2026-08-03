"""Tests for Reefman (Derk/DerkE) model — eval, loader, fitter, registry."""

import numpy as np
import pytest

from lm19.tube_params import ReefmanParams, lookup_tube
from lm19.reefman import (
    ReefmanModel,
    _koren_cathode,
    _derk_ia_ig2,
    load_reefman_model,
    fit_reefman,
    _list_reefman_tubes,
)
from lm19.tube_model_base import MODEL_REGISTRY
from lm19.constants import (
    TOPOLOGY_PENTODE,
)
from lm19.tube_model_base import (
    MODEL_TYPE_REEFMAN,
)


# ---------------------------------------------------------------------------
# Reference params from TubeLib.inc
# ---------------------------------------------------------------------------

EL84_PARAMS = ReefmanParams(
    type="BTetrodeD",
    mu=23.36, ex=1.138, kg1=117.4, kg2=1275.0,
    kp=152.4, kvb=4015.8,
    als=7.66, be=0.148, A=0.000434,
    Sc=0.038, ap=0.054, w=58.0, nu=1.56, lam=4.13,
)

EL34_PARAMS = ReefmanParams(
    type="BTetrodeD",
    mu=12.50, ex=1.363, kg1=217.7, kg2=1950.2,
    kp=50.5, kvb=1282.7,
    als=6.09, be=0.105, A=0.000348,
    Sc=0.022, ap=0.033, w=64.0, nu=2.91, lam=4.23,
)

SIX_L6_PARAMS = ReefmanParams(
    type="BTetrodeDE",
    mu=9.41, ex=1.306, kg1=446.6, kg2=6672.5,
    kp=45.2, kvb=3205.1,
    als=8.10, be=0.069, A=0.000491,
    Sc=0.031, ap=0.017, w=-5.0, nu=3.08, lam=14.66,
)


class TestKorenCathode:
    """Test Koren cathode current function."""

    def test_positive_current(self):
        """Cathode current should be positive for normal operating point."""
        vg2 = np.array([250.0])
        vg1 = np.array([-7.0])
        ip = _koren_cathode(vg2, vg1, mu=23.36, ex=1.138, kp=152.4, kvb=4015.8)
        assert ip[0] > 0

    def test_zero_at_cutoff(self):
        """Cathode current should be ~0 for very negative Vg1."""
        vg2 = np.array([250.0])
        vg1 = np.array([-50.0])
        ip = _koren_cathode(vg2, vg1, mu=23.36, ex=1.138, kp=152.4, kvb=4015.8)
        assert ip[0] < 1e-6

    def test_vectorized(self):
        """Should handle array inputs."""
        vg2 = np.array([250.0, 250.0, 250.0])
        vg1 = np.array([-5.0, -10.0, -15.0])
        ip = _koren_cathode(vg2, vg1, mu=23.36, ex=1.138, kp=152.4, kvb=4015.8)
        assert ip.shape == (3,)
        assert ip[0] > ip[1] > ip[2]  # more negative Vg1 → less current


class TestDerkIaIg2:
    """Test Derk/DerkE Ia and Ig2 computation."""

    @pytest.mark.parametrize("params,label", [
        (EL84_PARAMS, "EL84 BTetrodeD"),
        (EL34_PARAMS, "EL34 BTetrodeD"),
        (SIX_L6_PARAMS, "6L6GC BTetrodeDE"),
    ])
    def test_positive_currents(self, params, label):
        """Ia and Ig2 should be positive at typical operating point."""
        va = np.array([250.0])
        vg1 = np.array([-7.0])
        vg2 = np.array([250.0])
        ia, ig2 = _derk_ia_ig2(va, vg1, vg2, params)
        assert ia[0] > 0, f"{label}: Ia should be positive"
        assert ig2[0] > 0, f"{label}: Ig2 should be positive"

    def test_ia_increases_with_va(self):
        """Ia should generally increase with anode voltage."""
        va = np.array([50.0, 150.0, 250.0, 350.0])
        vg1 = np.full(4, -7.0)
        vg2 = np.full(4, 250.0)
        ia, _ = _derk_ia_ig2(va, vg1, vg2, EL84_PARAMS)
        # At least monotonically for Va > knee
        assert ia[-1] > ia[0]

    def test_cutoff(self):
        """Very negative Vg1 should give ~0 current."""
        va = np.array([250.0])
        vg1 = np.array([-30.0])
        vg2 = np.array([250.0])
        ia, ig2 = _derk_ia_ig2(va, vg1, vg2, EL84_PARAMS)
        assert ia[0] < 0.001  # < 1 mA in Amperes
        assert ig2[0] < 0.001

    def test_derke_splitting(self):
        """DerkE (exp) should differ from Derk (rational) in knee region."""
        va = np.linspace(10, 100, 10)
        vg1 = np.full(10, -5.0)
        vg2 = np.full(10, 250.0)

        # Make Derk version of 6L6 params
        p_derk = ReefmanParams(
            type="BTetrodeD",
            mu=9.41, ex=1.306, kg1=446.6, kg2=6672.5,
            kp=45.2, kvb=3205.1,
            als=8.10, be=0.069, A=0.000491,
        )
        ia_d, _ = _derk_ia_ig2(va, vg1, vg2, p_derk)
        ia_de, _ = _derk_ia_ig2(va, vg1, vg2, SIX_L6_PARAMS)

        # Should be different (different splitting functions)
        assert not np.allclose(ia_d, ia_de, rtol=0.01)

    def test_secondary_emission_effect(self):
        """Secondary emission should reduce Ia at low Va."""
        va = np.array([20.0])
        vg1 = np.array([-3.0])
        vg2 = np.array([250.0])

        p_no_sec = ReefmanParams(
            type="BTetrodeD",
            mu=23.36, ex=1.138, kg1=117.4, kg2=1275.0,
            kp=152.4, kvb=4015.8,
            als=7.66, be=0.148, A=0.000434,
            Sc=0.0,  # no secondary emission
        )
        ia_no_sec, _ = _derk_ia_ig2(va, vg1, vg2, p_no_sec)
        ia_with_sec, _ = _derk_ia_ig2(va, vg1, vg2, EL84_PARAMS)

        # Secondary emission should reduce Ia at low Va
        assert ia_with_sec[0] <= ia_no_sec[0]


class TestReefmanModel:
    """Test ReefmanModel class (TubeModelProtocol)."""

    def test_protocol_compliance(self):
        """ReefmanModel should have required attributes."""
        model = ReefmanModel(name="test", topology=TOPOLOGY_PENTODE,
                             reefman=EL84_PARAMS)
        assert hasattr(model, "model_type")
        assert hasattr(model, "ia")
        assert hasattr(model, "ig2")
        assert hasattr(model, "params_dict")
        assert hasattr(model, "generate_scan")
        assert model.model_type == MODEL_TYPE_REEFMAN

    def test_ia_returns_mA(self):
        """ia() should return milliamperes."""
        model = ReefmanModel(name="test", topology=TOPOLOGY_PENTODE,
                             reefman=EL84_PARAMS)
        ia = model.ia(250, -7, 250)
        # Typical EL84 at this point: 30-50 mA
        assert 10.0 < ia < 100.0

    def test_ig2_returns_mA(self):
        """ig2() should return milliamperes."""
        model = ReefmanModel(name="test", topology=TOPOLOGY_PENTODE,
                             reefman=EL84_PARAMS)
        ig2 = model.ig2(250, -7, 250)
        assert 0.5 < ig2 < 30.0

    def test_params_dict(self):
        """params_dict() should return expected keys."""
        model = ReefmanModel(name="test", topology=TOPOLOGY_PENTODE,
                             reefman=EL84_PARAMS)
        d = model.params_dict()
        assert "mu" in d
        assert "type" in d
        assert "Sc" in d  # BTetrodeD should have sec emission params
        assert d["type"] == "BTetrodeD"

    def test_generate_scan(self):
        """generate_scan() should produce points."""
        from lm19.tube_sim import ScanGrid
        model = ReefmanModel(name="test", topology=TOPOLOGY_PENTODE,
                             reefman=EL84_PARAMS)
        grid = ScanGrid(
            ua=(0, 300, 50), ug1=(-10, 0, 5),
            ug2=(250, 250, 1))
        pts = model.generate_scan(grid)
        assert len(pts) > 0
        assert "ia" in pts[0]
        assert "ig2" in pts[0]


class TestLoader:
    """Test load_reefman_model and _list_reefman_tubes."""

    def test_load_el84(self):
        """Should load EL84 model from tube_params.json."""
        model = load_reefman_model("EL84")
        assert model is not None
        assert model.reefman.type == "BTetrodeD"
        assert model.reefman.mu == pytest.approx(23.36)

    def test_load_6l6(self):
        """Should load 6L6 (BTetrodeDE) model."""
        model = load_reefman_model("6L6")
        assert model is not None
        assert model.reefman.type == "BTetrodeDE"

    def test_load_nonexistent(self):
        """Should return None for unknown tube."""
        assert load_reefman_model("NONEXIST_TUBE") is None

    def test_load_triode_returns_none(self):
        """Triodes have no Reefman params."""
        assert load_reefman_model("12AX7") is None

    def test_list_tubes(self):
        """Should list tubes with reefman params."""
        tubes = _list_reefman_tubes()
        assert len(tubes) > 0
        assert "EL84" in tubes
        assert "12AX7" not in tubes


class TestRegistry:
    """Test MODEL_REGISTRY integration."""

    def test_registered(self):
        """reefman should be in MODEL_REGISTRY."""
        assert "reefman" in MODEL_REGISTRY

    def test_label(self):
        assert MODEL_REGISTRY["reefman"].label == "Reefman (Derk/DerkE)"


class TestFitter:
    """Test fit_reefman on synthetic data."""

    def test_round_trip(self):
        """Fit should recover parameters from synthetic data."""
        from lm19.tube_sim import ScanGrid

        model = ReefmanModel(name="test", topology=TOPOLOGY_PENTODE,
                             reefman=EL84_PARAMS)
        grid = ScanGrid(
            ua=(0, 300, 10), ug1=(-15, 0, 1.5),
            ug2=(250, 250, 1))
        points = model.generate_scan(grid)

        result = fit_reefman(points, "pentode")
        assert result.rms_error < 1.0  # < 1 mA on synthetic data
        assert result.model_type == MODEL_TYPE_REEFMAN
        # Fitted mu should be close to original
        assert result.params["mu"] == pytest.approx(EL84_PARAMS.mu, rel=0.3)

    def test_rejects_triode(self):
        """Should raise for triode topology."""
        with pytest.raises(RuntimeError, match="pentode data"):
            fit_reefman([], "triode")

    def test_rejects_insufficient_data(self):
        """Should raise with too few points."""
        points = [{"ua": 250, "ug1": -7, "ug2": 250,
                    "ia": 48, "ig2": 5}] * 5
        with pytest.raises(RuntimeError, match="Not enough"):
            fit_reefman(points, "pentode")


class TestFitterDE:
    """Test Reefman fitter on BTetrodeDE (6L6GC) synthetic data."""

    @pytest.fixture(scope="class")
    def de_fit(self):
        from lm19.tube_sim import ScanGrid
        model = ReefmanModel(name="test", topology=TOPOLOGY_PENTODE,
                             reefman=SIX_L6_PARAMS)
        grid = ScanGrid(
            ua=(0, 400, 10), ug1=(-20, 0, 2),
            ug2=(250, 250, 1))
        points = model.generate_scan(grid)
        return fit_reefman(points, "pentode")

    def test_fit_returns_result(self, de_fit):
        assert de_fit is not None
        assert de_fit.model_type == MODEL_TYPE_REEFMAN

    def test_rms_low(self, de_fit):
        """RMS on synthetic DE data should be low."""
        assert de_fit.rms_error < 2.0, \
            f"RMS = {de_fit.rms_error:.2f} mA"

    def test_mu_recovery(self, de_fit):
        """Fitted µ should be close to 6L6GC reference (9.41)."""
        mu = de_fit.params["mu"]
        assert abs(mu - SIX_L6_PARAMS.mu) / SIX_L6_PARAMS.mu < 0.4, \
            f"µ = {mu:.1f}, expected ~{SIX_L6_PARAMS.mu}"


class TestFitterIg2Errors:
    """Verify rms_ig2 / max_ig2 are populated by Reefman fitter."""

    @pytest.fixture(scope="class")
    def fit_with_ig2(self):
        from lm19.tube_sim import ScanGrid
        model = ReefmanModel(name="test", topology=TOPOLOGY_PENTODE,
                             reefman=EL84_PARAMS)
        grid = ScanGrid(
            ua=(0, 300, 10), ug1=(-15, 0, 1.5),
            ug2=(250, 250, 1))
        points = model.generate_scan(grid)
        return fit_reefman(points, "pentode")

    def test_ig2_errors_present(self, fit_with_ig2):
        assert fit_with_ig2.rms_ig2 is not None
        assert fit_with_ig2.max_ig2 is not None

    def test_ig2_rms_positive(self, fit_with_ig2):
        assert fit_with_ig2.rms_ig2 >= 0

    def test_ig2_max_ge_rms(self, fit_with_ig2):
        assert fit_with_ig2.max_ig2 >= fit_with_ig2.rms_ig2

    def test_ig2_rms_low_on_synthetic(self, fit_with_ig2):
        """On synthetic data, Ig2 error should be small."""
        assert fit_with_ig2.rms_ig2 < 2.0, \
            f"RMS Ig2 = {fit_with_ig2.rms_ig2:.2f} mA"


class TestFitterNoisyData:
    """Reefman fitter robustness on noisy data."""

    @pytest.fixture(scope="class")
    def noisy_fit(self):
        from lm19.tube_sim import ScanGrid
        rng = np.random.RandomState(42)
        model = ReefmanModel(name="test", topology=TOPOLOGY_PENTODE,
                             reefman=EL84_PARAMS)
        grid = ScanGrid(
            ua=(0, 300, 10), ug1=(-15, 0, 1.5),
            ug2=(250, 250, 1))
        points = model.generate_scan(grid)
        for p in points:
            p["ia"] = max(p["ia"] * (1.0 + rng.normal(0, 0.03)), 0.0)
            p["ig2"] = max(p["ig2"] * (1.0 + rng.normal(0, 0.03)), 0.0)
        return fit_reefman(points, "pentode")

    def test_rms_bounded(self, noisy_fit):
        """RMS should remain bounded despite ±3% noise."""
        assert noisy_fit.rms_error < 3.0, \
            f"RMS = {noisy_fit.rms_error:.2f} mA"

    def test_mu_recovery(self, noisy_fit):
        mu = noisy_fit.params["mu"]
        assert abs(mu - EL84_PARAMS.mu) / EL84_PARAMS.mu < 0.4, \
            f"µ = {mu:.1f}, expected ~{EL84_PARAMS.mu}"


# ===================================================================
# Numerical safety
# ===================================================================

class TestReefmanNumericalSafety:
    """Edge cases: Va=0, extreme Vg1, Vg2=0."""

    def test_va_zero(self):
        """Va=0 should not crash."""
        va = np.array([0.0])
        vg1 = np.array([-7.0])
        vg2 = np.array([250.0])
        ia, ig2 = _derk_ia_ig2(va, vg1, vg2, EL84_PARAMS)
        assert np.isfinite(ia[0])
        assert np.isfinite(ig2[0])

    def test_vg2_zero(self):
        """Vg2=0 should not crash."""
        va = np.array([250.0])
        vg1 = np.array([-7.0])
        vg2 = np.array([0.0])
        ia, ig2 = _derk_ia_ig2(va, vg1, vg2, EL84_PARAMS)
        assert np.isfinite(ia[0])
        assert np.isfinite(ig2[0])

    def test_large_negative_vg1(self):
        """Very negative Vg1 → ~0 current, not NaN."""
        va = np.array([250.0])
        vg1 = np.array([-100.0])
        vg2 = np.array([250.0])
        ia, ig2 = _derk_ia_ig2(va, vg1, vg2, EL84_PARAMS)
        assert np.isfinite(ia[0])
        assert ia[0] >= 0
        assert ia[0] < 1e-3  # essentially zero

    def test_positive_vg1(self):
        """Positive Vg1 should not overflow."""
        va = np.array([250.0])
        vg1 = np.array([5.0])
        vg2 = np.array([250.0])
        ia, ig2 = _derk_ia_ig2(va, vg1, vg2, EL84_PARAMS)
        assert np.isfinite(ia[0])
        assert np.isfinite(ig2[0])

    def test_very_large_va(self):
        """Va=1000V should not overflow."""
        va = np.array([1000.0])
        vg1 = np.array([-7.0])
        vg2 = np.array([250.0])
        ia, ig2 = _derk_ia_ig2(va, vg1, vg2, EL84_PARAMS)
        assert np.isfinite(ia[0])
        assert ia[0] > 0

    def test_de_variant_safety(self):
        """BTetrodeDE should also handle edge cases."""
        va = np.array([0.0, 250.0, 1000.0])
        vg1 = np.array([-100.0, -7.0, 5.0])
        vg2 = np.array([250.0, 250.0, 250.0])
        ia, ig2 = _derk_ia_ig2(va, vg1, vg2, SIX_L6_PARAMS)
        assert np.all(np.isfinite(ia))
        assert np.all(np.isfinite(ig2))


# ===================================================================
# Current conservation
# ===================================================================

class TestReefmanCurrentConservation:
    """Ia and Ig2 should both be non-negative at normal operating points."""

    @pytest.mark.parametrize("params,label", [
        (EL84_PARAMS, "EL84"),
        (EL34_PARAMS, "EL34"),
        (SIX_L6_PARAMS, "6L6GC"),
    ])
    def test_ia_non_negative(self, params, label):
        va = np.array([50.0, 150.0, 250.0, 350.0])
        vg1 = np.full(4, -7.0)
        vg2 = np.full(4, 250.0)
        ia, _ = _derk_ia_ig2(va, vg1, vg2, params)
        assert np.all(ia >= 0), f"{label}: negative Ia detected"

    @pytest.mark.parametrize("params,label", [
        (EL84_PARAMS, "EL84"),
        (EL34_PARAMS, "EL34"),
        (SIX_L6_PARAMS, "6L6GC"),
    ])
    def test_ig2_non_negative(self, params, label):
        va = np.array([50.0, 150.0, 250.0, 350.0])
        vg1 = np.full(4, -7.0)
        vg2 = np.full(4, 250.0)
        _, ig2 = _derk_ia_ig2(va, vg1, vg2, params)
        assert np.all(ig2 >= 0), f"{label}: negative Ig2 detected"

    def test_ia_plus_ig2_le_cathode(self):
        """Ia + Ig2 should not exceed cathode current (Ik).

        With secondary emission, Ia + Ig2 can be less than Ik
        (some current returns to screen grid).
        """
        va = np.array([250.0])
        vg1 = np.array([-7.0])
        vg2 = np.array([250.0])
        ia, ig2 = _derk_ia_ig2(va, vg1, vg2, EL84_PARAMS)
        ik = _koren_cathode(vg2, vg1,
                            mu=EL84_PARAMS.mu, ex=EL84_PARAMS.ex,
                            kp=EL84_PARAMS.kp, kvb=EL84_PARAMS.kvb)
        total = ia[0] + ig2[0]
        # Ia + Ig2 should be in the same order of magnitude as Ik
        assert total > 0
        assert total <= ik[0] * 1.05, \
            f"Ia+Ig2={total:.6f} > Ik={ik[0]:.6f}"


# ===================================================================
# Fit: EL34 round-trip (BTetrodeD, different params)
# ===================================================================

class TestFitterEL34:
    """Round-trip on EL34 synthetic data (BTetrodeD)."""

    @pytest.fixture(scope="class")
    def el34_fit(self):
        from lm19.tube_sim import ScanGrid
        model = ReefmanModel(name="test", topology=TOPOLOGY_PENTODE,
                             reefman=EL34_PARAMS)
        grid = ScanGrid(
            ua=(0, 400, 10), ug1=(-20, 0, 2),
            ug2=(300, 300, 1))
        points = model.generate_scan(grid)
        return fit_reefman(points, "pentode")

    def test_fit_ok(self, el34_fit):
        assert el34_fit.model_type == MODEL_TYPE_REEFMAN
        # EL34 has secondary emission — fitter doesn't fit Sc/ap/w/nu/lam,
        # so RMS is higher than on EL84 synthetic data
        assert el34_fit.rms_error < 5.0, \
            f"RMS = {el34_fit.rms_error:.2f} mA"

    def test_mu_recovery(self, el34_fit):
        mu = el34_fit.params["mu"]
        assert abs(mu - EL34_PARAMS.mu) / EL34_PARAMS.mu < 0.4, \
            f"µ = {mu:.1f}, expected ~{EL34_PARAMS.mu}"


# ===================================================================
# Fit: no-Ig2 data
# ===================================================================

class TestFitterNoIg2:
    """Fitter should work when Ig2 data is missing."""

    def test_fit_without_ig2(self):
        from lm19.tube_sim import ScanGrid
        model = ReefmanModel(name="test", topology=TOPOLOGY_PENTODE,
                             reefman=EL84_PARAMS)
        grid = ScanGrid(
            ua=(0, 300, 10), ug1=(-15, 0, 1.5),
            ug2=(250, 250, 1))
        points = model.generate_scan(grid)
        for p in points:
            p["ig2"] = 0.0
        result = fit_reefman(points, "pentode")
        assert result is not None
        assert result.rms_error < 3.0
        assert result.rms_ig2 is None  # no Ig2 data


# ===================================================================
# Fit: multi-Ug2 round-trip
# ===================================================================

class TestFitterMultiUg2:
    """Round-trip with multiple Ug2 levels."""

    @pytest.fixture(scope="class")
    def multi_ug2_fit(self):
        from lm19.tube_sim import ScanGrid
        model = ReefmanModel(name="test", topology=TOPOLOGY_PENTODE,
                             reefman=EL84_PARAMS)
        grid = ScanGrid(
            ua=(0, 300, 15), ug1=(-12, 0, 2),
            ug2=(200, 300, 50))  # 3 Ug2 levels
        points = model.generate_scan(grid)
        return fit_reefman(points, "pentode")

    def test_fit_ok(self, multi_ug2_fit):
        assert multi_ug2_fit.model_type == MODEL_TYPE_REEFMAN

    def test_rms_low(self, multi_ug2_fit):
        assert multi_ug2_fit.rms_error < 2.0, \
            f"RMS = {multi_ug2_fit.rms_error:.2f} mA"

    def test_more_points(self, multi_ug2_fit):
        """Multi-Ug2 should use more data points."""
        assert multi_ug2_fit.n_points > 30
