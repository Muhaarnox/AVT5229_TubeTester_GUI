"""Tests for tube_model_base: Protocol conformance, Registry."""

from lm19.tube_model_base import (
    MODEL_REGISTRY,
    ModelFitResult,
    ModelRegistryEntry,
    TubeModelProtocol,
    list_all_tubes,
    register_model,
)
from lm19.tube_sim import TubeModel, load_model

import pytest
from lm19.tube_model_base import (
    MODEL_TYPE_KOREN,
)


class TestProtocolConformance:
    """TubeModel must satisfy TubeModelProtocol at runtime."""

    def test_isinstance_triode(self):
        m = load_model("12AX7")
        assert isinstance(m, TubeModelProtocol)

    def test_isinstance_pentode(self):
        m = load_model("EL34")
        assert isinstance(m, TubeModelProtocol)

    def test_model_type_field(self):
        m = load_model("12AX7")
        assert m.model_type == MODEL_TYPE_KOREN

    def test_params_dict_triode(self):
        m = load_model("12AX7")
        d = m.params_dict()
        assert set(d.keys()) == {"mu", "ex", "kg1", "kp", "kvb"}
        assert all(isinstance(v, float) for v in d.values())

    def test_params_dict_pentode(self):
        m = load_model("EL34")
        d = m.params_dict()
        assert "kg2" in d
        assert set(d.keys()) == {"mu", "ex", "kg1", "kp", "kvb", "kg2"}

    def test_ig2_signature_accepts_ua(self):
        m = load_model("EL34")
        ig2 = m.ig2(ua=200.0, ug1=-10.0, ug2=250.0)
        assert ig2 > 0

    def test_triode_ig2_zero(self):
        m = load_model("12AX7")
        assert m.ig2(ua=100.0, ug1=-1.0, ug2=0.0) == 0.0

    def test_required_attributes(self):
        m = load_model("12AX7")
        for attr in ("model_type", "name", "topology", "pa_max", "uh", "ih"):
            assert hasattr(m, attr), f"Missing attribute: {attr}"


class TestRegistry:
    """MODEL_REGISTRY registration and lookup."""

    def test_koren_registered(self):
        assert "koren" in MODEL_REGISTRY

    def test_koren_entry_has_methods(self):
        entry = MODEL_REGISTRY["koren"]
        assert callable(entry.loader)
        assert callable(entry.fitter)
        assert callable(entry.list_tubes)

    def test_list_all_tubes_koren(self):
        tubes = list_all_tubes("koren")
        assert len(tubes) > 0
        assert "12AX7" in tubes

    def test_list_all_tubes_unknown_raises(self):
        with pytest.raises(KeyError):
            list_all_tubes("nonexistent_model_type")

    def test_loader_known_tube(self):
        entry = MODEL_REGISTRY["koren"]
        m = entry.loader("12AX7")
        assert m is not None
        assert isinstance(m, TubeModelProtocol)

    def test_loader_unknown_tube_returns_none(self):
        entry = MODEL_REGISTRY["koren"]
        assert entry.loader("NONEXISTENT_TUBE") is None

    def test_register_custom_model(self):
        """Register and then clean up a custom model type."""
        register_model(
            "test_dummy",
            label="Dummy",
            loader=lambda name: None,
            fitter=lambda pts, topo: None,
            list_tubes=lambda: ["DUMMY1"],
        )
        assert "test_dummy" in MODEL_REGISTRY
        assert list_all_tubes("test_dummy") == ["DUMMY1"]
        # Cleanup
        del MODEL_REGISTRY["test_dummy"]
