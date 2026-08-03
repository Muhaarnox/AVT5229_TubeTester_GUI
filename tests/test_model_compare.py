"""Tests for model comparison logic (lm19.model_compare)."""

import numpy as np
import pytest

from lm19.model_compare import (
    CompareRow,
    SPICE_MODELS,
    compare_all_models,
    compute_gm_from_data,
    compute_gm_from_model,
    compute_rms_gm,
    _is_compatible,
)
from lm19.tube_model_base import MODEL_REGISTRY
from lm19.tube_model_base import (
    MODEL_TYPE_KOREN,
    MODEL_TYPE_REEFMAN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_triode_points():
    """Generate synthetic triode data from 12AX7 reference model."""
    from lm19.tube_sim import quick_triode
    _, pts = quick_triode("12AX7")
    return pts


def _make_pentode_points():
    """Generate synthetic pentode data from EL84 reference model."""
    from lm19.tube_sim import quick_pentode
    _, pts = quick_pentode("EL84")
    return pts


# ---------------------------------------------------------------------------
# gm from data
# ---------------------------------------------------------------------------

class TestGmFromData:

    def test_linear_slope(self):
        """Known linear Ia(Ug1) should give constant gm."""
        # Ia = 2.0 * Ug1 + 10  (gm = 2.0 mA/V)
        points = []
        for ug1 in [-5.0, -4.0, -3.0, -2.0, -1.0]:
            ia = 2.0 * ug1 + 10.0
            points.append({"ua": 200.0, "ug1": ug1, "ug2": 0.0, "ia": ia})
        gm = compute_gm_from_data(points)
        # Central points should be exactly 2.0
        assert gm[1] == pytest.approx(2.0)
        assert gm[2] == pytest.approx(2.0)
        assert gm[3] == pytest.approx(2.0)

    def test_single_point_gives_nan(self):
        """Single point in group -> gm = NaN."""
        points = [{"ua": 200.0, "ug1": -5.0, "ug2": 0.0, "ia": 1.0}]
        gm = compute_gm_from_data(points)
        assert np.isnan(gm[0])

    def test_multiple_ua_groups(self):
        """Points at different Ua should form separate groups."""
        points = []
        for ua in [100.0, 200.0]:
            for ug1 in [-3.0, -2.0, -1.0]:
                # Different slope for different Ua
                ia = (1.0 + ua / 100.0) * ug1 + 10.0
                points.append({"ua": ua, "ug1": ug1, "ug2": 0.0, "ia": ia})
        gm = compute_gm_from_data(points)
        # Group at Ua=100: slope=2.0, Group at Ua=200: slope=3.0
        valid = ~np.isnan(gm)
        assert np.sum(valid) >= 4


# ---------------------------------------------------------------------------
# gm from model
# ---------------------------------------------------------------------------

class TestGmFromModel:

    def test_model_gm_positive(self):
        """Model gm should be positive (Ia increases with less negative Ug1)."""
        from lm19.tube_sim import quick_triode
        model, _ = quick_triode("12AX7")
        points = [
            {"ua": 200.0, "ug1": -2.0, "ug2": 0.0, "ia": 0.0},
        ]
        gm = compute_gm_from_model(model, points)
        assert gm[0] > 0  # gm should be positive

    def test_gm_shape(self):
        """Should return one gm per point."""
        from lm19.tube_sim import quick_triode
        model, pts = quick_triode("12AX7")
        gm = compute_gm_from_model(model, pts[:5])
        assert gm.shape == (5,)


# ---------------------------------------------------------------------------
# RMS gm
# ---------------------------------------------------------------------------

class TestRmsGm:

    def test_zero_error_on_self(self):
        """Model gm compared to itself should give ~0 RMS."""
        from lm19.tube_sim import quick_triode
        model, pts = quick_triode("12AX7")
        # Use model to generate "data" gm
        data_gm = compute_gm_from_model(model, pts)
        rms = compute_rms_gm(model, pts, data_gm)
        assert rms is not None
        assert rms < 0.01  # near zero

    def test_none_for_few_points(self):
        """Should return None if not enough valid gm points."""
        from lm19.tube_sim import quick_triode
        model, _ = quick_triode("12AX7")
        # All-NaN data_gm
        data_gm = np.full(3, np.nan)
        pts = [{"ua": 200.0, "ug1": -2.0, "ug2": 0.0}] * 3
        rms = compute_rms_gm(model, pts, data_gm)
        assert rms is None


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------

class TestCompatibility:

    def test_koren_all_topologies(self):
        assert _is_compatible("koren", "triode")
        assert _is_compatible("koren", "pentode")
        assert _is_compatible("koren", "triode_connected")

    def test_reefman_pentode_only(self):
        assert _is_compatible("reefman", "pentode")
        assert not _is_compatible("reefman", "triode")
        assert not _is_compatible("reefman", "triode_connected")

    def test_dempwolf_all_topologies(self):
        assert _is_compatible("dempwolf", "triode")
        assert _is_compatible("dempwolf", "pentode")


# ---------------------------------------------------------------------------
# SPICE support
# ---------------------------------------------------------------------------

class TestSpiceSupport:

    def test_all_models_supported(self):
        assert "koren" in SPICE_MODELS
        assert "dempwolf" in SPICE_MODELS
        assert "reefman" in SPICE_MODELS


# ---------------------------------------------------------------------------
# compare_all_models — triode
# ---------------------------------------------------------------------------

class TestCompareTriode:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.points = _make_triode_points()

    def test_returns_rows(self):
        rows = compare_all_models(self.points, "triode")
        assert len(rows) > 0

    def test_koren_ok(self):
        rows = compare_all_models(self.points, "triode")
        koren_rows = [r for r in rows if r.model_type == MODEL_TYPE_KOREN]
        assert len(koren_rows) == 1
        assert koren_rows[0].status == "OK"
        assert koren_rows[0].rms_ia is not None
        assert koren_rows[0].rms_ia >= 0

    def test_reefman_na_for_triode(self):
        rows = compare_all_models(self.points, "triode")
        reef_rows = [r for r in rows if r.model_type == MODEL_TYPE_REEFMAN]
        assert len(reef_rows) == 1
        assert reef_rows[0].status == "N/A"

    def test_no_ig2_for_triode(self):
        rows = compare_all_models(self.points, "triode")
        ok = [r for r in rows if r.status == "OK"]
        assert ok, "at least one model must fit OK (de-vacuated)"
        for r in ok:
            assert r.rms_ig2 is None

    def test_sorted_by_rms(self):
        rows = compare_all_models(self.points, "triode")
        ok_rows = [r for r in rows if r.status == "OK"]
        for i in range(len(ok_rows) - 1):
            assert ok_rows[i].rms_ia <= ok_rows[i + 1].rms_ia

    def test_na_rows_at_end(self):
        rows = compare_all_models(self.points, "triode")
        found_na = False
        for r in rows:
            if r.status == "N/A":
                found_na = True
            elif found_na:
                pytest.fail("N/A rows should be at end")

    def test_spice_column(self):
        rows = compare_all_models(self.points, "triode")
        for r in rows:
            assert r.spice_support is True

    def test_n_params(self):
        rows = compare_all_models(self.points, "triode")
        ok = [r for r in rows if r.status == "OK"]
        assert ok, "at least one model must fit OK (de-vacuated)"
        for r in ok:
            assert r.n_params > 0

    def test_gm_computed(self):
        rows = compare_all_models(self.points, "triode")
        ok_rows = [r for r in rows if r.status == "OK"]
        # At least one OK row should have gm computed
        assert any(r.rms_gm is not None for r in ok_rows)


# ---------------------------------------------------------------------------
# compare_all_models — pentode
# ---------------------------------------------------------------------------

class TestComparePentode:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.points = _make_pentode_points()

    def test_returns_rows(self):
        rows = compare_all_models(self.points, "pentode")
        assert len(rows) >= 2  # at least koren + reefman

    def test_koren_ok(self):
        rows = compare_all_models(self.points, "pentode")
        koren_rows = [r for r in rows if r.model_type == MODEL_TYPE_KOREN]
        assert len(koren_rows) == 1
        assert koren_rows[0].status == "OK"

    def test_reefman_ok_for_pentode(self):
        rows = compare_all_models(self.points, "pentode")
        reef_rows = [r for r in rows if r.model_type == MODEL_TYPE_REEFMAN]
        assert len(reef_rows) == 1
        assert reef_rows[0].status == "OK"

    def test_ig2_for_pentode(self):
        """Pentode fitters should report Ig2 errors."""
        rows = compare_all_models(self.points, "pentode")
        ok_rows = [r for r in rows if r.status == "OK"]
        # At least one should have ig2 errors
        assert any(r.rms_ig2 is not None for r in ok_rows)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

class TestCancellation:

    def test_cancel_stops_early(self):
        points = _make_triode_points()
        call_count = 0

        def cancel_after_one():
            nonlocal call_count
            call_count += 1
            return call_count > 1

        rows = compare_all_models(points, "triode", cancelled=cancel_after_one)
        # Should have at most 1 real result + early exit
        assert len(rows) <= len(MODEL_REGISTRY)


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class TestProgress:

    def test_progress_called(self):
        points = _make_triode_points()
        calls = []

        def on_progress(current, total, label):
            calls.append((current, total, label))

        compare_all_models(points, "triode", on_progress=on_progress)
        # Should call progress for each model + final
        assert len(calls) >= 2
        # Last call should have current == total
        assert calls[-1][0] == calls[-1][1]


class TestNarrowFitterExcept:
    """ML-103: a refactor regression (AttributeError/TypeError) in a fitter
    must crash visibly — the old broad except showed it as "model didn't
    fit", indistinguishable from honest non-convergence."""

    def _with_broken_fitter(self, exc_type, monkeypatch):
        from lm19 import tube_model_base as tmb
        from lm19.model_compare import compare_all_models
        from lm19.tube_sim import quick_triode
        _, pts = quick_triode("ECC83")
        entry = next(iter(tmb.MODEL_REGISTRY.values()))
        monkeypatch.setattr(
            entry, "fitter",
            lambda points, topology: (_ for _ in ()).throw(exc_type("x")))
        return compare_all_models, pts

    def test_programming_error_propagates(self, monkeypatch):
        import pytest
        compare_all_models, pts = self._with_broken_fitter(
            AttributeError, monkeypatch)
        with pytest.raises(AttributeError):
            compare_all_models(pts, "triode")

    def test_data_error_becomes_failed_row(self, monkeypatch):
        compare_all_models, pts = self._with_broken_fitter(
            ValueError, monkeypatch)
        rows = compare_all_models(pts, "triode")   # must not raise
        assert any(r.rms_ia is None for r in rows)
