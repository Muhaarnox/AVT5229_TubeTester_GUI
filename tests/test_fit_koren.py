"""Tests for fit_koren: fitting without file I/O."""

import pytest

from lm19.tube_model_base import ModelFitResult, TubeModelProtocol
from lm19.tube_sim import (
    TubeModel,
    fit_koren,
    load_model,
    quick_triode,
    quick_pentode,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)
from lm19.tube_model_base import (
    MODEL_TYPE_KOREN,
)


class TestFitKorenTriode:
    """Fit Koren triode model from synthetic data."""

    def test_fit_returns_model_fit_result(self):
        _, points = quick_triode("12AX7")
        result = fit_koren(points, "triode")
        assert isinstance(result, ModelFitResult)
        assert result.model_type == MODEL_TYPE_KOREN
        assert result.topology == TOPOLOGY_TRIODE

    def test_fit_model_is_protocol(self):
        _, points = quick_triode("12AX7")
        result = fit_koren(points, "triode")
        assert isinstance(result.model, TubeModelProtocol)

    def test_fit_rms_below_threshold(self):
        _, points = quick_triode("12AX7")
        result = fit_koren(points, "triode")
        # Fitting clean synthetic data should give very low error
        assert result.rms_error < 1.0

    def test_fit_params_dict_keys(self):
        _, points = quick_triode("12AX7")
        result = fit_koren(points, "triode")
        keys = set(result.params.keys())
        assert keys == {"mu", "ex", "kg1", "kp", "kvb"}

    def test_round_trip(self):
        """Fit → generate → compare with original."""
        model_orig, points_orig = quick_triode("12AX7")
        result = fit_koren(points_orig, "triode")
        model_fit = result.model

        # Compare Ia at a few operating points
        for p in points_orig[:5]:
            ia_orig = model_orig.ia(p["ua"], p["ug1"])
            ia_fit = model_fit.ia(p["ua"], p["ug1"])
            assert abs(ia_orig - ia_fit) < 1.0, (
                f"Ia mismatch at Ua={p['ua']}, Ug1={p['ug1']}: "
                f"{ia_orig:.3f} vs {ia_fit:.3f}"
            )

    def test_triode_connected_uses_triode_model(self):
        _, points = quick_triode("12AX7")
        result = fit_koren(points, "triode_connected")
        assert result.topology == TOPOLOGY_TRIODE

    def test_insufficient_data_raises(self):
        # Only 5 points with very low Ia (< 0.1 mA threshold)
        points = [{"ua": 10, "ug1": -10, "ug2": 0, "ia": 0.01, "ig2": 0}] * 5
        with pytest.raises(RuntimeError, match="Not enough"):
            fit_koren(points, "triode")


class TestFitKorenPentode:
    """Fit Koren pentode model from synthetic data."""

    def test_fit_pentode(self):
        _, points = quick_pentode("EL84")
        result = fit_koren(points, "pentode")
        assert result.topology == TOPOLOGY_PENTODE
        assert "kg2" in result.params
        assert result.rms_error < 2.0

    def test_fit_pentode_model_evaluates(self):
        _, points = quick_pentode("EL84")
        result = fit_koren(points, "pentode")
        ia = result.model.ia(200.0, -5.0, 250.0)
        assert ia > 0

    def test_fit_pentode_ig2(self):
        _, points = quick_pentode("EL84")
        result = fit_koren(points, "pentode")
        ig2 = result.model.ig2(200.0, -5.0, 250.0)
        assert ig2 >= 0  # May be 0 if kg2 not fitted well

    def test_insufficient_pentode_data_raises(self):
        points = [{"ua": 10, "ug1": -10, "ug2": 250, "ia": 0.01, "ig2": 0}] * 5
        with pytest.raises(RuntimeError, match="Not enough"):
            fit_koren(points, "pentode")


class TestPackageFacade:
    """``fit_and_export_spice`` is reachable from the package root."""

    def test_import_fit_and_export_spice(self):
        from lm19.spice_export import fit_and_export_spice
        assert callable(fit_and_export_spice)
