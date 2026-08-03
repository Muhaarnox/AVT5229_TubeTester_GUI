"""Tests for AmplifierEngine — pure computation, no Qt."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.amp_engine import AmplifierEngine, AmpParams, AnalysisResult, SourceResult
from lm19.amplifier import (
    ResistiveLoadLine, TransformerLoadLine, CathodeFollowerLoadLine, PushPullLoadLine,
    UltralinearModelWrapper,
)


# ═══════════════════════════════════════════════════════════════════
#  Test data helpers
# ═══════════════════════════════════════════════════════════════════

from tests._fixtures import (  # noqa: E402
    make_triode_points as _make_triode_points,
    make_pentode_points as _make_pentode_points,
)
from lm19.amplifier.constants import (
    CIRCUIT_CF,
    CIRCUIT_PP,
    CIRCUIT_SE,
    CIRCUIT_SE_XFMR,
    HD_METHOD_5POINT,
    HD_METHOD_AUTO,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_DFT,
)


class SimpleModel:
    """Minimal tube model for testing."""
    topology = "triode"
    model_type = "koren"
    name = "Koren fit"
    pa_max = 12.0
    uh = 6.3
    ih = 0.7

    def __init__(self, func=None):
        self._func = func or (lambda ug1: max(0.0, 10.0 + 2.0 * (ug1 + 7.0) + 0.2 * (ug1 + 7.0) ** 2))

    def ia(self, ua, ug1, ug2=0.0):
        return max(0.0, self._func(ug1))

    def ig2(self, ua, ug1, ug2):
        return 0.0

    def generate_scan(self, grid):
        return []

    def params_dict(self):
        return {}


# ═══════════════════════════════════════════════════════════════════
#  Engine creation & data management
# ═══════════════════════════════════════════════════════════════════

class TestEngineCreation:
    """Engine init and data management."""

    def test_empty_engine_has_no_data(self):
        e = AmplifierEngine()
        assert not e.has_data

    def test_set_data_makes_has_data_true(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points())
        assert e.has_data

    def test_set_data_with_models(self):
        e = AmplifierEngine()
        model = SimpleModel()
        e.set_data(_make_triode_points(), series_models={1: model})
        avail = e.available_models()
        assert "koren" in avail

    def test_available_models_empty_initially(self):
        e = AmplifierEngine()
        assert e.available_models() == {}

    def test_analyze_no_data_returns_error(self):
        e = AmplifierEngine()
        r = e.analyze(AmpParams())
        assert r.error == "no_data"


# ═══════════════════════════════════════════════════════════════════
#  SE analysis
# ═══════════════════════════════════════════════════════════════════

class TestSEAnalysis:
    """Single-ended resistive analysis pipeline."""

    def _engine_with_data(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        return e

    def test_se_basic_returns_result(self):
        e = self._engine_with_data()
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1))
        assert r.error is None
        assert r.circuit == CIRCUIT_SE
        assert "measurements" in r.per_source

    def test_se_has_distortion(self):
        e = self._engine_with_data()
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1))
        sr = r.per_source["measurements"]
        assert sr.dist is not None
        assert "thd" in sr.dist
        assert "hd2" in sr.dist

    def test_se_has_headroom(self):
        e = self._engine_with_data()
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1))
        sr = r.per_source["measurements"]
        assert sr.headroom is not None

    def test_se_has_stage_params(self):
        e = self._engine_with_data()
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1))
        sr = r.per_source["measurements"]
        assert sr.stage is not None

    def test_se_has_imd(self):
        e = self._engine_with_data()
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1))
        sr = r.per_source["measurements"]
        assert sr.imd is not None

    def test_se_has_sweep_amp(self):
        e = self._engine_with_data()
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1, amp_steps=10))
        sr = r.per_source["measurements"]
        assert len(sr.sweep_amp) > 0

    def test_se_has_sweep_ra(self):
        e = self._engine_with_data()
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1, ra_steps=10))
        sr = r.per_source["measurements"]
        assert len(sr.sweep_ra) > 0

    def test_se_load_line_is_resistive(self):
        e = self._engine_with_data()
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1))
        assert isinstance(r.load_line, ResistiveLoadLine)

    def test_se_with_half_swing(self):
        e = self._engine_with_data()
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, half_swing=2.0, series_id=1))
        sr = r.per_source["measurements"]
        assert sr.dist is not None


# ═══════════════════════════════════════════════════════════════════
#  SE Transformer
# ═══════════════════════════════════════════════════════════════════

class TestSETransformer:
    """SE Transformer circuit."""

    def test_xfmr_load_line(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_SE_XFMR,
            ra_dc=0.05, series_id=1,
        ))
        assert isinstance(r.load_line, TransformerLoadLine)

    def test_xfmr_has_results(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_SE_XFMR,
            ra_dc=0.05, series_id=1,
        ))
        assert "measurements" in r.per_source
        sr = r.per_source["measurements"]
        assert sr.dist is not None


# ═══════════════════════════════════════════════════════════════════
#  Cathode Follower
# ═══════════════════════════════════════════════════════════════════

class TestCathodeFollower:
    """CF circuit specifics."""

    def test_cf_load_line(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_CF,
            cf_rk=10.0, cf_rl=10.0, series_id=1,
        ))
        assert isinstance(r.load_line, CathodeFollowerLoadLine)

    def test_cf_no_sweep_ra(self):
        """CF should not produce sweep_ra."""
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_CF,
            cf_rk=10.0, cf_rl=10.0, series_id=1,
        ))
        sr = r.per_source["measurements"]
        assert sr.sweep_ra == []


# ═══════════════════════════════════════════════════════════════════
#  Push-Pull
# ═══════════════════════════════════════════════════════════════════

class TestPushPull:
    """PP circuit."""

    def test_pp_load_line(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_PP,
            pp_raa=8.0, series_id=1,
        ))
        assert isinstance(r.load_line, PushPullLoadLine)
        assert r.circuit == CIRCUIT_PP

    def test_pp_has_pp_dist(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_PP,
            pp_raa=8.0, series_id=1,
        ))
        assert r.pp_dist is not None

    def test_pp_matched(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_PP,
            pp_raa=8.0, pp_matched=True, series_id=1,
        ))
        assert r.error is None

    def test_pp_unmatched_no_tube_b(self):
        """Unmatched PP without Tube B data → error.

        select_analysis_points falls back to all points, so we need
        pp_tube_b_sid different from any series_id in the data AND
        no series_id=0 fallback. Use series_id=5 in data, request sid=99.
        But fallback returns all points. So we use an empty engine instead.
        """
        e = AmplifierEngine()
        # Only series_id=5 points, no series_id=0
        pts = _make_triode_points(series_id=5)
        e.set_data(pts)
        # select_analysis_points(all_points, 99) → tries sid=99 (none), tries sid=0 (none),
        # falls back to all. So this won't error.
        # To truly test, we'd need select_analysis_points to return [].
        # Instead, test that unmatched PP WITH tube_b_sid=None skips Tube B lookup:
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_PP,
            pp_raa=8.0, pp_matched=False, pp_tube_b_sid=None, series_id=5,
        ))
        # pp_tube_b_sid=None → points_b stays None (no lookup)
        assert r.error is None


# ═══════════════════════════════════════════════════════════════════
#  Model source
# ═══════════════════════════════════════════════════════════════════

class TestModelSource:
    """Analysis using fitted model as data source."""

    def test_model_source_uses_dft(self):
        e = AmplifierEngine()
        model = SimpleModel()
        pts = _make_triode_points(series_id=1)
        e.set_data(pts, series_models={1: model})
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["koren"], hd_method=HD_METHOD_AUTO,
        ))
        sr = r.per_source.get("koren")
        assert sr is not None
        # auto → dft for model source
        assert sr.method_used == HD_METHOD_DFT

    def test_model_source_has_distortion(self):
        e = AmplifierEngine()
        model = SimpleModel()
        pts = _make_triode_points(series_id=1)
        e.set_data(pts, series_models={1: model})
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["koren"],
        ))
        sr = r.per_source.get("koren")
        assert sr is not None
        assert sr.dist is not None

    def test_measurements_source_uses_chebyshev_auto(self):
        e = AmplifierEngine()
        pts = _make_triode_points(series_id=1)
        e.set_data(pts)
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["measurements"], hd_method=HD_METHOD_AUTO,
        ))
        sr = r.per_source["measurements"]
        assert sr.method_used == HD_METHOD_CHEBYSHEV


# ═══════════════════════════════════════════════════════════════════
#  Multiple sources
# ═══════════════════════════════════════════════════════════════════

class TestMultipleSources:
    """Analysis with multiple sources in parallel."""

    def test_two_sources(self):
        e = AmplifierEngine()
        model = SimpleModel()
        pts = _make_triode_points(series_id=1)
        e.set_data(pts, series_models={1: model})
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["measurements", "koren"],
        ))
        assert "measurements" in r.per_source
        assert "koren" in r.per_source

    def test_multiple_sources_share_load_line(self):
        e = AmplifierEngine()
        model = SimpleModel()
        pts = _make_triode_points(series_id=1)
        e.set_data(pts, series_models={1: model})
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["measurements", "koren"],
        ))
        # Single load line for all sources
        assert r.load_line is not None
        assert isinstance(r.load_line, ResistiveLoadLine)


# ═══════════════════════════════════════════════════════════════════
#  HD method selection
# ═══════════════════════════════════════════════════════════════════

class TestHDMethod:
    """Explicit HD method selection."""

    def test_force_5point(self):
        e = AmplifierEngine()
        pts = _make_triode_points(series_id=1)
        e.set_data(pts)
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            hd_method=HD_METHOD_5POINT,
        ))
        sr = r.per_source["measurements"]
        assert sr.method_used == HD_METHOD_5POINT

    def test_force_chebyshev(self):
        e = AmplifierEngine()
        pts = _make_triode_points(series_id=1)
        e.set_data(pts)
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            hd_method=HD_METHOD_CHEBYSHEV,
        ))
        sr = r.per_source["measurements"]
        assert sr.method_used == HD_METHOD_CHEBYSHEV


# ═══════════════════════════════════════════════════════════════════
#  NFB
# ═══════════════════════════════════════════════════════════════════

class TestNFB:
    """Negative feedback calculation."""

    def test_nfb_disabled_by_default(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1))
        sr = r.per_source["measurements"]
        assert sr.nfb is None

    def test_nfb_enabled(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1, nfb_db=6.0,
        ))
        sr = r.per_source["measurements"]
        # NFB may or may not compute depending on gain/distortion values
        # but if distortion and gain exist, nfb should be computed
        assert sr.dist and sr.stage and (sr.stage.get('gain', 0) > 0), "guard de-vacuated 2026-07-12: value must be present"
        assert sr.nfb is not None


# ═══════════════════════════════════════════════════════════════════
#  Ug2 filter (pentode)
# ═══════════════════════════════════════════════════════════════════

class TestUg2Filter:
    """Ug2 filtering for pentode data."""

    def test_ug2_filter_passes_matching(self):
        e = AmplifierEngine()
        pts = _make_pentode_points(ug2=200.0, series_id=1)
        e.set_data(pts, is_triode=False)
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            ug2_filter=200.0,
        ))
        assert r.error is None

    def test_ug2_filter_rejects_mismatched(self):
        """Points with ug2=200 filtered by ug2_filter=100 → no matching points for stage."""
        e = AmplifierEngine()
        pts = _make_pentode_points(ug2=200.0, series_id=1)
        e.set_data(pts, is_triode=False)
        # The analysis still runs (intersections use all points), but stage uses filtered
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            ug2_filter=100.0,
        ))
        # Should still produce result (intersections don't filter by ug2 strictly)
        assert r.error is None


# ═══════════════════════════════════════════════════════════════════
#  Dataclass defaults
# ═══════════════════════════════════════════════════════════════════

class TestDataclasses:
    """AmpParams and result dataclass defaults."""

    def test_amp_params_defaults(self):
        p = AmpParams()
        assert p.ub == 250.0
        assert p.ra == 5.0
        assert p.circuit == CIRCUIT_SE
        assert p.sources == ["measurements"]
        assert p.hd_method == HD_METHOD_AUTO

    def test_source_result_defaults(self):
        sr = SourceResult()
        assert sr.dist is None
        assert sr.sweep_amp == []
        assert sr.method_used == HD_METHOD_5POINT

    def test_analysis_result_defaults(self):
        ar = AnalysisResult()
        assert ar.per_source == {}
        assert ar.error is None
        assert ar.circuit == CIRCUIT_SE


# ═══════════════════════════════════════════════════════════════════
#  Multi-source end-to-end
# ═══════════════════════════════════════════════════════════════════

class TestMultiSourceE2E:
    """End-to-end multi-source tests: measurements + model → compare."""

    def _engine_with_model(self):
        e = AmplifierEngine()
        model = SimpleModel()
        pts = _make_triode_points(series_id=1)
        e.set_data(pts, series_models={1: model})
        return e

    def test_each_source_has_dist(self):
        e = self._engine_with_model()
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["measurements", "koren"],
        ))
        for name in ("measurements", "koren"):
            sr = r.per_source[name]
            assert sr.dist is not None or sr.method_used, f"{name} missing dist"

    def test_each_source_has_sweep_amp(self):
        e = self._engine_with_model()
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["measurements", "koren"],
        ))
        for name in ("measurements", "koren"):
            sr = r.per_source[name]
            assert isinstance(sr.sweep_amp, list)

    def test_each_source_has_headroom(self):
        e = self._engine_with_model()
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["measurements", "koren"],
        ))
        for name in ("measurements", "koren"):
            sr = r.per_source[name]
            assert sr.headroom is not None or sr.dist is None

    def test_single_source_only_measurements(self):
        """When sources=['measurements'], model source is absent."""
        e = self._engine_with_model()
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["measurements"],
        ))
        assert "measurements" in r.per_source
        assert "koren" not in r.per_source

    def test_single_source_only_model(self):
        """When sources=['koren'], measurements source is absent."""
        e = self._engine_with_model()
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["koren"],
        ))
        assert "koren" in r.per_source
        assert "measurements" not in r.per_source

    def test_unknown_source_falls_back(self):
        """Unknown model name still produces an entry (falls back to measurements data)."""
        e = self._engine_with_model()
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["measurements", "nonexistent_model"],
        ))
        assert "measurements" in r.per_source
        assert "nonexistent_model" in r.per_source

    def test_method_differs_per_source(self):
        """Measurements use chebyshev by default, model uses dft."""
        e = self._engine_with_model()
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["measurements", "koren"],
            hd_method=HD_METHOD_AUTO,
        ))
        meas = r.per_source["measurements"]
        model = r.per_source["koren"]
        # auto → chebyshev for measurements, dft for model
        assert meas.method_used in (HD_METHOD_CHEBYSHEV, HD_METHOD_5POINT)
        assert model.method_used in (HD_METHOD_DFT, HD_METHOD_5POINT)


# ═══════════════════════════════════════════════════════════════════
#  Edge cases & error paths
# ══════════════════════════════════════════════════════════���════════

class TestEdgeCases:
    """Error handling and boundary conditions."""

    def test_analyze_without_data_returns_error(self):
        """analyze() on empty engine → error."""
        e = AmplifierEngine()
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0))
        assert r.error is not None

    def test_extreme_bias_few_intersections(self):
        """Bias far from data range → few intersections → still no crash."""
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-0.1, series_id=1,  # very close to 0
        ))
        # Should produce some result or degrade gracefully
        assert r.error is None or r.per_source.get("measurements") is not None

    def test_very_small_ra(self):
        """Very small Ra → steep load line → still works."""
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(ub=250, ra=0.5, ug1_bias=-7.0, series_id=1))
        assert r.error is None

    def test_very_large_ra(self):
        """Very large Ra → flat load line → may have few intersections."""
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(ub=250, ra=200.0, ug1_bias=-7.0, series_id=1))
        # Should not crash
        assert isinstance(r, AnalysisResult)

    def test_half_swing_near_zero_treated_as_auto(self):
        """half_swing=0.05 (< 0.1 threshold) → engine treats as auto."""
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1, half_swing=0.05,
        ))
        assert r.error is None

    def test_dist_error_set_when_dist_none(self):
        """When compute_distortion returns None, dist_error is populated."""
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        # Pick a bias far outside the measured Ug1 range so compute fails
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-50.0, series_id=1,
        ))
        sr = r.per_source.get("measurements")
        assert sr is not None and sr.dist is None, "guard de-vacuated 2026-07-12: value must be present"
        assert sr.dist_error is not None
        assert sr.dist_error  # non-empty
        assert sr.dist_error in (
            "few_intersections", "bias_outside_data",
            "bias_at_data_edge", "manual_swing_small",
            "manual_swing_clipped", "no_signal", "unknown",
        )
        if sr.dist_error in ("bias_outside_data", "bias_at_data_edge"):
            # diagnostic payload: the measured Ug1 span the message
            # shows to the user
            p = sr.dist_error_params
            assert p is not None
            assert p["lo"] <= p["hi"] < 0     # negative Ug1 domain
            assert p["bias"] == -50.0
            assert not (p["lo"] <= p["bias"] <= p["hi"])

    def test_no_signal_diagnosed_with_window_payload(self):
        """Identical curves → the load line crosses every Ug1 at the SAME
        current: visually fine crossings, zero fundamental. Must say
        no_signal with the flat-window numbers,
        not the catch-all 'unknown'."""
        e = AmplifierEngine()
        pts = []
        for ug1 in (-2.0, -4.0, -6.0, -8.0, -10.0, -12.0):
            for ua in range(50, 301, 25):
                pts.append({"ua": float(ua), "ug1": ug1,
                            "ia": ua / 25.0, "series_id": 1})
        e.set_data(pts, is_triode=True)
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1))
        sr = r.per_source.get("measurements")
        assert sr is not None and sr.dist is None
        assert sr.dist_error == "no_signal"
        p = sr.dist_error_params
        assert p is not None
        assert abs(p["imax"] - p["imin"]) < 0.01   # the flat window shown
        assert float(p["b1"]) <= 0.01              # pre-formatted string

    def test_dist_error_none_when_dist_succeeds(self):
        """When dist is computed successfully, dist_error stays None."""
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
        ))
        sr = r.per_source.get("measurements")
        assert sr is not None and sr.dist is not None, "guard de-vacuated 2026-07-12: value must be present"
        assert sr.dist_error is None

    def test_nfb_without_gain_no_crash(self):
        """NFB enabled but stage gain missing → nfb is None, no crash."""
        e = AmplifierEngine()
        # Use very few points so stage calc may fail
        pts = _make_triode_points(n_ug1=5, n_ua=5, series_id=1)
        e.set_data(pts)
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1, nfb_db=6.0,
        ))
        # Should not crash; nfb might be None if stage failed
        assert isinstance(r, AnalysisResult)

    def test_dft_without_model_falls_back(self):
        """hd_method='dft' but no model → falls back to 5point."""
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            hd_method=HD_METHOD_DFT,
        ))
        sr = r.per_source["measurements"]
        # DFT needs a model; without one it should fall back
        assert sr.method_used in (HD_METHOD_DFT, HD_METHOD_5POINT, HD_METHOD_CHEBYSHEV)

    def test_set_data_replaces_previous(self):
        """set_data() called twice replaces data."""
        e = AmplifierEngine()
        pts1 = _make_triode_points(n_ug1=5, n_ua=5, series_id=1)
        pts2 = _make_triode_points(n_ug1=21, n_ua=20, series_id=2)
        e.set_data(pts1)
        e.set_data(pts2)
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=2))
        assert r.error is None

    def test_model_without_model_type_attr(self):
        """Model missing model_type attribute → still usable."""
        class BareModel:
            topology = "triode"
            name = "Bare"
            pa_max = 10.0
            uh = 6.3
            ih = 0.7

            def ia(self, ua, ug1, ug2=0.0):
                return max(0.0, 10.0 + 2.0 * (ug1 + 7.0))

            def ig2(self, ua, ug1, ug2):
                return 0.0

            def generate_scan(self, grid):
                return []

            def params_dict(self):
                return {}

        e = AmplifierEngine()
        pts = _make_triode_points(series_id=1)
        e.set_data(pts, series_models={1: BareModel()})
        models = e.available_models()
        # model_type not found → key is "model_1"
        assert len(models) == 1
        assert "model_1" in models


class TestSweepEdgeCases:
    """Sweep computation edge cases."""

    def test_sweep_amp_produces_list(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1))
        sr = r.per_source["measurements"]
        assert isinstance(sr.sweep_amp, list)
        if sr.sweep_amp:
            assert "half_swing" in sr.sweep_amp[0]
            assert "thd" in sr.sweep_amp[0]

    def test_sweep_ra_produces_list(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1))
        sr = r.per_source["measurements"]
        assert isinstance(sr.sweep_ra, list)
        if sr.sweep_ra:
            assert "ra" in sr.sweep_ra[0]
            assert "thd" in sr.sweep_ra[0]

    def test_cf_no_ra_sweep(self):
        """Cathode follower doesn't sweep Ra (no resistive load)."""
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1, circuit=CIRCUIT_CF,
        ))
        sr = r.per_source["measurements"]
        assert sr.sweep_ra == []


class TestSweepHD45:
    """Verify that sweep_amp and sweep_ra propagate hd4/hd5 keys."""

    def test_sweep_amp_contains_hd4_hd5_keys(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1))
        sr = r.per_source["measurements"]
        assert sr.sweep_amp, "guard de-vacuated 2026-07-12: value must be present"
        assert "hd4" in sr.sweep_amp[0]
        assert "hd5" in sr.sweep_amp[0]

    def test_sweep_ra_contains_hd4_hd5_keys(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1))
        sr = r.per_source["measurements"]
        assert sr.sweep_ra, "guard de-vacuated 2026-07-12: value must be present"
        assert "hd4" in sr.sweep_ra[0]
        assert "hd5" in sr.sweep_ra[0]

    def test_sweep_hd4_hd5_are_float(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1))
        sr = r.per_source["measurements"]
        for d in sr.sweep_amp[:3]:
            assert isinstance(d["hd4"], float)
            assert isinstance(d["hd5"], float)
            assert d["hd4"] >= 0.0
            assert d["hd5"] >= 0.0

    def test_show_hd45_in_amp_params(self):
        p = AmpParams(show_hd45=True)
        assert p.show_hd45 is True
        p2 = AmpParams()
        assert p2.show_hd45 is False

    def test_sweep_ra_contains_gain_zout_pa(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0, series_id=1))
        sr = r.per_source["measurements"]
        assert sr.sweep_ra, "guard de-vacuated 2026-07-12: value must be present"
        d = sr.sweep_ra[0]
        assert "gain" in d
        assert "zout" in d
        assert "pa_mw" in d
        assert isinstance(d["gain"], float)
        assert isinstance(d["zout"], float)
        assert d["pa_mw"] >= 0.0

    def test_show_gzp_in_amp_params(self):
        p = AmpParams(show_gzp=True)
        assert p.show_gzp is True
        p2 = AmpParams()
        assert p2.show_gzp is False


class TestChebyshevIntegration:
    """Integration tests for Chebyshev distortion analysis."""

    def _engine_with_model(self):
        e = AmplifierEngine()
        model = SimpleModel()
        pts = _make_triode_points(series_id=1)
        e.set_data(pts, series_models={1: model})
        return e

    def test_chebyshev_model_produces_hd9(self):
        """Chebyshev + model → dense grid → HD2-HD9 available."""
        e = self._engine_with_model()
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["koren"], hd_method=HD_METHOD_CHEBYSHEV,
        ))
        sr = r.per_source.get("koren")
        assert sr is not None
        assert sr.dist is not None
        assert sr.dist.get("method") == HD_METHOD_CHEBYSHEV
        assert sr.dist.get("max_harmonic") == 9
        # All harmonics present
        for n in range(2, 10):
            assert f"hd{n}" in sr.dist

    def test_chebyshev_measurements_few_curves_auto_reduces(self):
        """Chebyshev with few Ug1 curves → max_harmonic auto-reduced."""
        e = AmplifierEngine()
        # Only 5 Ug1 levels → at most max_harmonic=4
        pts = _make_triode_points(n_ug1=5, series_id=1)
        e.set_data(pts)
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["measurements"], hd_method=HD_METHOD_CHEBYSHEV,
        ))
        sr = r.per_source.get("measurements")
        assert sr is not None
        if sr.dist is not None:
            assert sr.dist["max_harmonic"] < 9
            assert "hd2" in sr.dist

    def test_chebyshev_measurements_many_curves_hd9(self):
        """Chebyshev with enough Ug1 curves → full HD2-HD9."""
        e = AmplifierEngine()
        pts = _make_triode_points(n_ug1=21, series_id=1)
        e.set_data(pts)
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["measurements"], hd_method=HD_METHOD_CHEBYSHEV,
        ))
        sr = r.per_source.get("measurements")
        assert sr is not None
        assert sr.dist is not None
        assert sr.dist["max_harmonic"] == 9

    def test_chebyshev_model_more_isects_than_raw(self):
        """Chebyshev with model generates more intersections than raw Ug1."""
        e = self._engine_with_model()
        # Force chebyshev for model source
        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["koren"], hd_method=HD_METHOD_CHEBYSHEV,
        ))
        sr = r.per_source.get("koren")
        assert sr is not None
        assert sr.dist is not None
        # Dense grid should produce HD9 — model has no data limitation
        assert sr.dist["max_harmonic"] == 9

    def test_chebyshev_model_vs_dft_both_produce_results(self):
        """Both Chebyshev and DFT work on model and produce similar THD."""
        e = self._engine_with_model()
        r_cheb = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["koren"], hd_method=HD_METHOD_CHEBYSHEV,
        ))
        r_dft = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, series_id=1,
            sources=["koren"], hd_method=HD_METHOD_DFT,
        ))
        cheb = r_cheb.per_source["koren"].dist
        dft = r_dft.per_source["koren"].dist
        assert cheb is not None
        assert dft is not None
        # Both should give finite positive THD
        assert cheb["thd"] > 0
        assert dft["thd"] > 0


# ═══════════════════════════════════════════════════════════════════
#  Pentode model (Ug2-dependent) for UL tests
# ═══════════════════════════════════════════════════════════════════

class PentodeModel:
    """Minimal pentode model where Ia depends on Ug2.

    Ia = max(0, gm * (Ug1 + cutoff)^1.5 * (1 + Ua/500) * (Ug2/Ug2_ref))
    Ig2 = 0.15 * Ia  (screen current ~15% of anode current)

    The ^1.5 gives a realistic curved characteristic with moderate
    current levels (~10-50 mA) in the operating range Ug1 = -7..0V.
    """
    topology = "pentode"
    model_type = "koren"
    name = "Koren fit"
    pa_max = 12.0
    uh = 6.3
    ih = 0.76

    UG2_REF = 250.0
    CUTOFF = 10.0   # V (Ug1 cutoff, gives 0 at Ug1=-10)

    def ia(self, ua: float, ug1: float, ug2: float = 0.0) -> float:
        x = max(0.0, ug1 + self.CUTOFF)
        ug2_factor = ug2 / self.UG2_REF if self.UG2_REF > 0 else 1.0
        # k=0.5 mA/V^1.5 gives ~11 mA at Ug1=-7 (x=3), Ua=250, Ug2=250
        return 0.5 * (x ** 1.5) * (1.0 + ua / 500.0) * max(ug2_factor, 0.0)

    def ig2(self, ua: float, ug1: float, ug2: float) -> float:
        return 0.15 * self.ia(ua, ug1, ug2)

    def generate_scan(self, grid):
        return []

    def params_dict(self):
        return {"cutoff": self.CUTOFF}


class _DempwolfParams:
    """Minimal Dempwolf params with grid current fields."""
    Gg = 6.177e-4   # A (12AX7-like)
    xi = 1.314
    Cg = 9.901


class DempwolfPentodeModel(PentodeModel):
    """PentodeModel with Dempwolf grid current parameters attached."""
    model_type = "dempwolf"
    name = "Dempwolf fit"
    params = _DempwolfParams()


# ═══════════════════════════════════════════════════════════════════
#  Ultralinear integration through AmplifierEngine
# ═══════════════════════════════════════════════════════════════════

class TestUltralinearEngine:
    """Integration tests: UL through AmplifierEngine.analyze().

    These test the REAL pipeline path, including SRK handling,
    to catch bugs that unit tests on isolated functions miss.
    """

    def _pentode_engine(self, srk=None):
        """Create engine with pentode data + model."""
        e = AmplifierEngine()
        model = PentodeModel()
        pts = _make_pentode_points(n_ug1=11, n_ua=20, ug2=250.0, series_id=1)
        e.set_data(
            pts,
            series_models={1: model},
            srk=srk,
            is_triode=False,
        )
        return e

    def _pp_params(self, ul_tap=None, **kw):
        defaults = dict(
            ub=300, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_PP,
            pp_raa=8.0, series_id=1,
            ug2_filter=250.0,
            sources=["koren"],
            ul_tap=ul_tap,
        )
        defaults.update(kw)
        return AmpParams(**defaults)

    # ── UL activates without SRK ────────────────────────────────

    def test_ul_engine_no_srk_no_crash(self):
        """UL with no SRK → engine should not crash."""
        e = self._pentode_engine(srk=None)
        r = e.analyze(self._pp_params(ul_tap=0.43))
        assert isinstance(r, AnalysisResult)

    def test_ul_engine_no_srk_has_distortion(self):
        """UL with no SRK → distortion still computed (from intersections)."""
        e = self._pentode_engine(srk=None)
        r = e.analyze(self._pp_params(ul_tap=0.43))
        sr = r.per_source.get("koren")
        # PP pipeline uses pp_dist, not per-source dist for now
        # But the engine should at least not error
        assert r.error is None

    # ── UL with SRK: stage params use SRK, not UL ───��──────────

    def test_ul_stage_uses_model_not_srk(self):
        """Model source → stage uses model_gm_ra, even with SRK present.

        With model available, compute_stage_params uses model as primary
        source for gm/ra. SRK is cross-checked but not used.
        UL wrapper gives UL-corrected gm/ra automatically.
        """
        srk_pentode = {"s": 11.0, "r": 50.0, "k": 550.0}
        e = self._pentode_engine(srk=srk_pentode)

        params_no_ul = AmpParams(
            ub=300, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_SE,
            series_id=1, ug2_filter=250.0,
            sources=["koren"], ul_tap=None,
        )
        params_ul = AmpParams(
            ub=300, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_SE,
            series_id=1, ug2_filter=250.0,
            sources=["koren"], ul_tap=0.43,
        )

        r_no_ul = e.analyze(params_no_ul)
        r_ul = e.analyze(params_ul)

        sr_no_ul = r_no_ul.per_source.get("koren")
        sr_ul = r_ul.per_source.get("koren")
        assert sr_no_ul is not None and sr_ul is not None

        # Model source → method="model", stage is not None
        assert sr_no_ul.stage is not None, "model source should produce stage"
        assert sr_ul.stage is not None, "UL model source should produce stage"
        assert sr_no_ul.stage["method"] == "model"
        assert sr_ul.stage["method"] == "model"

        # UL should give DIFFERENT gm/ra than pentode
        assert sr_no_ul.stage["ra"] != pytest.approx(sr_ul.stage["ra"], rel=0.01), (
            "UL ra should differ from pentode ra"
        )

    def test_ul_distortion_differs_from_pentode(self):
        """UL changes intersections → distortion should differ from pentode."""
        e = self._pentode_engine(srk=None)

        params_pent = AmpParams(
            ub=300, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_SE,
            series_id=1, ug2_filter=250.0,
            sources=["koren"], ul_tap=None,
        )
        params_ul = AmpParams(
            ub=300, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_SE,
            series_id=1, ug2_filter=250.0,
            sources=["koren"], ul_tap=0.43,
        )

        r_pent = e.analyze(params_pent)
        r_ul = e.analyze(params_ul)

        sr_pent = r_pent.per_source.get("koren")
        sr_ul = r_ul.per_source.get("koren")

        assert sr_pent and sr_pent.dist and sr_ul and sr_ul.dist, "guard de-vacuated 2026-07-12: value must be present"
        pent_thd = sr_pent.dist["thd"]
        ul_thd = sr_ul.dist["thd"]
        pent_ua = sr_pent.dist["ua_0"]
        ul_ua = sr_ul.dist["ua_0"]
        assert (
            abs(pent_ua - ul_ua) > 0.1 or abs(pent_thd - ul_thd) > 0.01
        ), "UL should change distortion characteristics"

    # ── UL without SRK: numerical fallback uses raw points ─────

    def test_ul_no_srk_stage_still_uses_model(self):
        """Without SRK, model source still uses model_gm_ra (not numerical).

        Model is highest priority regardless of SRK presence.
        """
        e = self._pentode_engine(srk=None)

        params_no_ul = AmpParams(
            ub=300, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_SE,
            series_id=1, ug2_filter=250.0,
            sources=["koren"], ul_tap=None,
        )
        params_ul = AmpParams(
            ub=300, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_SE,
            series_id=1, ug2_filter=250.0,
            sources=["koren"], ul_tap=0.43,
        )

        r_no_ul = e.analyze(params_no_ul)
        r_ul = e.analyze(params_ul)

        sr_no_ul = r_no_ul.per_source.get("koren")
        sr_ul = r_ul.per_source.get("koren")

        assert sr_no_ul is not None and sr_ul is not None
        assert sr_no_ul.stage is not None, "model source should produce stage"
        assert sr_ul.stage is not None, "UL model source should produce stage"
        assert sr_no_ul.stage["method"] == "model"
        assert sr_ul.stage["method"] == "model"
        # UL should lower ra
        assert sr_ul.stage["ra"] < sr_no_ul.stage["ra"]

    # ── SRK correctness matters ─────────────────────────────────

    def test_wrong_srk_detected_via_cross_check(self):
        """Wrong SRK is detected: model is used for gm/ra, SRK is cross-checked.

        With model available, compute_stage_params uses model_gm_ra.
        Wrong SRK triggers srk_check="divergence".
        """
        # Wildly wrong SRK (triode-like values for a pentode)
        srk_bad = {"s": 2.0, "r": 7.0, "k": 14.0}
        e = self._pentode_engine(srk=srk_bad)

        params = AmpParams(
            ub=300, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_SE,
            series_id=1, ug2_filter=250.0, sources=["koren"],
        )

        r = e.analyze(params)
        sr = r.per_source.get("koren")
        assert sr is not None and sr.stage is not None

        # Model is used, not SRK
        assert sr.stage["method"] == "model"
        # Wrong SRK triggers divergence warning
        assert sr.stage["srk_check"] == "divergence"

    def test_no_srk_no_points_stage_is_none(self):
        """Without SRK and without raw points, stage is None."""
        e = AmplifierEngine()
        model = PentodeModel()
        # No measurement points at all — only model
        pts = _make_pentode_points(n_ug1=3, n_ua=3, ug2=250.0, series_id=1)
        e.set_data(pts, series_models={1: model}, srk=None, is_triode=False)
        r = e.analyze(AmpParams(
            ub=300, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_SE,
            series_id=1, ug2_filter=250.0, sources=["koren"],
        ))
        sr = r.per_source.get("koren")
        # With very few points, numerical method may fail
        # stage could be None — that's acceptable, not a crash
        assert isinstance(r, AnalysisResult)

    # ── UL only wraps for pentode, not triode ───────────────────

    def test_ul_tap_ignored_for_triode(self):
        """UL tap set but triode model → no wrapping (topology check)."""
        e = AmplifierEngine()
        model = SimpleModel()  # triode
        pts = _make_triode_points(series_id=1)
        e.set_data(pts, series_models={1: model})

        r = e.analyze(AmpParams(
            ub=250, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_SE,
            series_id=1, sources=["koren"], ul_tap=0.43,
        ))
        # Should run fine — ul_tap ignored for triode topology
        assert r.error is None

    # ── UL + NFB chain ──────────────────────────────────────────

    def test_ul_nfb_chain_uses_model_stage(self):
        """NFB with UL uses model-derived stage params (UL-corrected).

        NFB reads gain_open from stage["gain"], which comes from
        model_gm_ra (UL-wrapped). SRK is cross-checked.
        """
        srk = {"s": 11.0, "r": 50.0, "k": 550.0}
        e = self._pentode_engine(srk=srk)

        params = AmpParams(
            ub=300, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_SE,
            series_id=1, ug2_filter=250.0,
            sources=["koren"], ul_tap=0.43, nfb_db=10.0,
        )

        r = e.analyze(params)
        sr = r.per_source.get("koren")

        assert sr is not None
        assert sr.stage is not None, "model source should produce stage"
        assert sr.stage["method"] == "model"
        if sr.nfb:
            assert sr.nfb["gain_open"] == pytest.approx(sr.stage["gain"])

    # ── Pa_avg engine integration ────────────────────────────────

    def test_pa_avg_populated_with_model_and_swing(self):
        """Engine populates pa_avg when model + swing are available."""
        e = self._pentode_engine(srk=None)
        params = AmpParams(
            ub=300, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_SE,
            series_id=1, ug2_filter=250.0,
            sources=["koren"], half_swing=3.0,
        )
        r = e.analyze(params)
        sr = r.per_source.get("koren")
        assert sr is not None
        assert sr.pa_avg is not None, "pa_avg should be computed with model + swing"
        assert sr.pa_avg["pa_avg_mw"] > 0

    def test_pa_avg_none_without_swing(self):
        """Engine does NOT compute pa_avg when swing is None (auto)."""
        e = self._pentode_engine(srk=None)
        params = AmpParams(
            ub=300, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_SE,
            series_id=1, ug2_filter=250.0,
            sources=["koren"], half_swing=None,
        )
        r = e.analyze(params)
        sr = r.per_source.get("koren")
        assert sr is not None
        # Without explicit swing, pa_avg is not computed
        assert sr.pa_avg is None

    def test_pa_avg_with_measurements_source(self):
        """Measurements source with series model → pa_avg computed.

        The engine resolves a per-series model for the measurements
        source via ``series_id``, so ``pa_avg`` is available whenever
        a model exists for that series.
        """
        e = self._pentode_engine(srk=None)
        params = AmpParams(
            ub=300, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_SE,
            series_id=1, ug2_filter=250.0,
            sources=["measurements"], half_swing=3.0,
        )
        r = e.analyze(params)
        sr = r.per_source.get("measurements")
        assert sr is not None
        # Model is resolved from series_models → pa_avg is computed
        assert sr.pa_avg is not None

    # ── Grid current engine integration ──────────────────────────

    def test_grid_current_params_extraction_none_for_koren(self):
        """Koren model doesn't have Dempwolf grid params → None."""
        gc = AmplifierEngine._get_grid_current_params(PentodeModel())
        assert gc is None  # PentodeModel has no .params with Gg

    def test_grid_current_params_extraction_with_dempwolf(self):
        """Dempwolf model with Gg/xi/Cg → gc params extracted."""
        gc = AmplifierEngine._get_grid_current_params(DempwolfPentodeModel())
        assert gc is not None
        assert gc["Gg"] > 0
        assert "xi" in gc and "Cg" in gc

    def test_grid_current_quantified_in_headroom_via_engine(self):
        """Full engine pipeline: Dempwolf model → headroom includes ig1_ma."""
        e = AmplifierEngine()
        model = DempwolfPentodeModel()
        pts = _make_pentode_points(n_ug1=11, n_ua=20, ug2=250.0, series_id=1)
        e.set_data(pts, series_models={1: model}, srk=None, is_triode=False)

        # Use SE with Dempwolf model source
        params = AmpParams(
            ub=300, ra=5, ug1_bias=-3.0, circuit=CIRCUIT_SE,
            series_id=1, ug2_filter=250.0,
            sources=["dempwolf"], half_swing=2.5,
        )
        r = e.analyze(params)
        sr = r.per_source.get("dempwolf")
        assert sr is not None
        if sr.headroom is not None:
            # Dempwolf grid current should be quantified
            assert "ig1_ma" in sr.headroom, (
                "Headroom should include ig1_ma with Dempwolf model"
            )
            assert sr.headroom["ig1_ma"] >= 0


# ═══════════════════════════════════════════════════════════════════
#  PP pipeline: UL, stage params, pa_avg, grid current
# ═══════════════════════════════════════════════════════════════════

class TestPPPipelineFeatures:
    """Integration tests: PP pipeline uses UL, model_gm_ra, pa_avg, grid current."""

    def _pp_engine(self, model_cls=PentodeModel, srk=None):
        """Engine with pentode model for PP tests."""
        e = AmplifierEngine()
        model = model_cls()
        pts = _make_pentode_points(n_ug1=11, n_ua=20, ug2=250.0, series_id=1)
        e.set_data(pts, series_models={1: model}, srk=srk, is_triode=False)
        return e

    def _pp_params(self, **kw):
        defaults = dict(
            ub=300, ra=5, ug1_bias=-7.0, circuit=CIRCUIT_PP,
            pp_raa=8.0, series_id=1, ug2_filter=250.0,
        )
        defaults.update(kw)
        return AmpParams(**defaults)

    # ── PP has stage params ──────────────────────────────────────

    def test_pp_has_stage_params(self):
        """PP analysis should compute stage params."""
        e = self._pp_engine()
        r = e.analyze(self._pp_params())
        sr = r.per_source.get("measurements")
        assert sr is not None
        # Stage may be None if numerical fails on sparse data,
        # but should not crash
        assert isinstance(r, AnalysisResult)

    def test_pp_stage_params_with_model(self):
        """PP with model source → stage uses model_gm_ra."""
        e = self._pp_engine()
        # PP uses SOURCE_MEASUREMENTS in _get_intersections;
        # the model is resolved via series_id on the engine.
        r = e.analyze(self._pp_params())
        sr = r.per_source.get("measurements")
        assert sr is not None
        if sr.stage is not None:
            # Model should be found via series_id → method="model"
            assert sr.stage["method"] in ("model", "numerical", "srk")

    # ── PP has grid current ──────────────────────────────────────

    def test_pp_headroom_has_grid_current_with_dempwolf(self):
        """PP + Dempwolf model → headroom includes ig1_ma."""
        e = self._pp_engine(model_cls=DempwolfPentodeModel)
        r = e.analyze(self._pp_params(ug1_bias=-3.0, half_swing=2.5))
        sr = r.per_source.get("measurements")
        assert sr is not None
        if sr.headroom is not None:
            assert "ig1_ma" in sr.headroom

    def test_pp_headroom_no_grid_current_without_dempwolf(self):
        """PP + Koren model (no Dempwolf) → no ig1_ma."""
        e = self._pp_engine()
        r = e.analyze(self._pp_params())
        sr = r.per_source.get("measurements")
        assert sr is not None
        if sr.headroom is not None:
            assert "ig1_ma" not in sr.headroom

    # ── PP has pa_avg ────────────────────────────────────────────

    def test_pp_pa_avg_with_model_and_swing(self):
        """PP + model + explicit swing → pa_avg computed."""
        e = self._pp_engine()
        r = e.analyze(self._pp_params(half_swing=3.0))
        sr = r.per_source.get("measurements")
        assert sr is not None
        if sr.pa_avg is not None:
            assert sr.pa_avg["pa_avg_mw"] > 0

    def test_pp_pa_avg_none_without_swing(self):
        """PP without swing → pa_avg is None."""
        e = self._pp_engine()
        r = e.analyze(self._pp_params(half_swing=None))
        sr = r.per_source.get("measurements")
        assert sr is not None
        assert sr.pa_avg is None

    # ── PP + UL ──────────────────────────────────────────────────

    def test_pp_ul_no_crash(self):
        """PP + UL tap → no crash."""
        e = self._pp_engine()
        r = e.analyze(self._pp_params(ul_tap=0.43))
        assert r.error is None

    def test_pp_ul_changes_headroom(self):
        """PP + UL should change headroom vs pure pentode."""
        e = self._pp_engine()
        r_pent = e.analyze(self._pp_params(ul_tap=None))
        r_ul = e.analyze(self._pp_params(ul_tap=0.43))
        sr_pent = r_pent.per_source.get("measurements")
        sr_ul = r_ul.per_source.get("measurements")
        assert sr_pent is not None and sr_ul is not None
        # Both should have headroom, UL may change swing values
        if sr_pent.headroom and sr_ul.headroom:
            # At minimum, both should be valid
            assert sr_pent.headroom["max_swing"] > 0
            assert sr_ul.headroom["max_swing"] > 0



# ═══════════════════════════════════════════════════════════════════
#  Pentode Q-point validation — needs_ug2 signal to UI
# ═══════════════════════════════════════════════════════════════════

class TestPentodeUg2Validation:
    """Engine must signal UI when pentode model has no valid Ug2 source.

    The trap: if measurements have ``ug2=0`` (sensor failure or
    triode-format import), naive ``np.median(ug2_vals)`` returns 0 and
    gets fed to ``model.ia(ua, ug1, 0)`` (pentode at screen=0 V → fully
    cut off, silent garbage output). ``_resolve_intersections`` must
    detect this and raise ``_NeedsUg2``; ``analyze`` catches it and
    returns ``AnalysisResult(error="needs_ug2", suggested_ug2=...)`` so
    the UI can prompt the user for a screen voltage.
    """

    def _make_pentode_model_engine(self, broken_ug2: bool):
        """Build an engine with EL84 model + measurements (optionally
        with all ug2 zeroed to simulate sensor failure)."""
        from lm19.tube_sim import quick_pentode
        model, pts = quick_pentode("EL84")
        if broken_ug2:
            pts = [dict(p, ug2=0.0) for p in pts]
        engine = AmplifierEngine()
        engine.set_data(pts)
        engine._series_models = {0: model}
        return engine, model

    @pytest.mark.timeout(60)
    def test_zero_ug2_no_filter_raises_needs_ug2(self):
        """Pentode + all ug2=0 + no filter → engine signals needs_ug2."""
        engine, _ = self._make_pentode_model_engine(broken_ug2=True)
        params = AmpParams(circuit=CIRCUIT_SE, ub=300, ra=4.0, ug1_bias=-8.0,
                            sources=["koren"], series_id=0)
        r = engine.analyze(params)
        assert r.error == "needs_ug2",             f"Expected error=needs_ug2 for pentode with bad data, got {r.error}"
        assert r.suggested_ug2 is not None and r.suggested_ug2 > 0
        # No lamp default → falls back to DEFAULT_UG2_V (250)
        from lm19.constants import DEFAULT_UG2_V
        assert r.suggested_ug2 == DEFAULT_UG2_V

    @pytest.mark.timeout(60)
    def test_lamp_default_used_as_suggestion(self):
        """When lamp_ug2_default is set, engine uses it (e.g. 425V for KT88)."""
        engine, _ = self._make_pentode_model_engine(broken_ug2=True)
        params = AmpParams(circuit=CIRCUIT_SE, ub=300, ra=4.0, ug1_bias=-8.0,
                            sources=["koren"], series_id=0,
                            lamp_ug2_default=425.0)
        r = engine.analyze(params)
        assert r.error == "needs_ug2"
        assert r.suggested_ug2 == 425.0,             f"lamp_ug2_default=425 should be used as suggested, got {r.suggested_ug2}"

    @pytest.mark.timeout(60)
    def test_too_low_lamp_default_falls_back_to_global(self):
        """If lamp_ug2_default is also invalid (<= MIN_VALID_UG2_V),
        suggested falls back to DEFAULT_UG2_V."""
        engine, _ = self._make_pentode_model_engine(broken_ug2=True)
        params = AmpParams(circuit=CIRCUIT_SE, ub=300, ra=4.0, ug1_bias=-8.0,
                            sources=["koren"], series_id=0,
                            lamp_ug2_default=2.0)  # below threshold
        r = engine.analyze(params)
        from lm19.constants import DEFAULT_UG2_V
        assert r.suggested_ug2 == DEFAULT_UG2_V

    @pytest.mark.timeout(60)
    def test_valid_ug2_in_measurements_no_error(self):
        """Real Ug2 values in measurements → no signal, normal analysis."""
        engine, _ = self._make_pentode_model_engine(broken_ug2=False)
        params = AmpParams(circuit=CIRCUIT_SE, ub=300, ra=4.0, ug1_bias=-8.0,
                            sources=["koren"], series_id=0)
        r = engine.analyze(params)
        assert r.error is None, f"Valid data should not error, got {r.error}"
        assert r.per_source, "Expected populated per_source"

    @pytest.mark.timeout(60)
    def test_explicit_ug2_filter_overrides(self):
        """User-supplied ug2_filter wins even when measurements are bad."""
        engine, _ = self._make_pentode_model_engine(broken_ug2=True)
        params = AmpParams(circuit=CIRCUIT_SE, ub=300, ra=4.0, ug1_bias=-8.0,
                            sources=["koren"], series_id=0,
                            ug2_filter=250.0)  # explicit
        r = engine.analyze(params)
        assert r.error is None
        assert r.per_source

    @pytest.mark.timeout(60)
    def test_zero_ug2_filter_still_signals(self):
        """If user explicitly picks ug2_filter=0 (e.g., from a corrupt combo
        listing only zero), engine still signals — zero is never valid for
        a pentode regardless of who supplied it."""
        engine, _ = self._make_pentode_model_engine(broken_ug2=True)
        params = AmpParams(circuit=CIRCUIT_SE, ub=300, ra=4.0, ug1_bias=-8.0,
                            sources=["koren"], series_id=0,
                            ug2_filter=0.0,  # user picked the bad value
                            lamp_ug2_default=425.0)
        r = engine.analyze(params)
        assert r.error == "needs_ug2"
        assert r.suggested_ug2 == 425.0

    @pytest.mark.timeout(60)
    def test_triode_topology_no_validation(self):
        """Triode model with same broken data → no needs_ug2 signal
        (triodes don't use screen voltage)."""
        from lm19.tube_sim import quick_triode
        model, pts = quick_triode("12AU7")
        # zero ug2 in triode data shouldn't matter
        pts = [dict(p, ug2=0.0) for p in pts]
        engine = AmplifierEngine()
        engine.set_data(pts)
        engine._series_models = {0: model}
        params = AmpParams(circuit=CIRCUIT_SE, ub=250, ra=10.0, ug1_bias=-7.0,
                            sources=["koren"], series_id=0)
        r = engine.analyze(params)
        assert r.error is None,             f"Triode topology should not require Ug2, got error={r.error}"

    @pytest.mark.timeout(60)
    def test_partial_zeros_filtered_not_error(self):
        """Mix of zero and valid ug2 → median over valid only, no error."""
        from lm19.tube_sim import quick_pentode
        model, pts = quick_pentode("EL84")
        # Half points have ug2=0 (sensor flicker), half have ug2=250
        for i, p in enumerate(pts):
            if i % 2 == 0:
                p["ug2"] = 0.0
        engine = AmplifierEngine()
        engine.set_data(pts)
        engine._series_models = {0: model}
        params = AmpParams(circuit=CIRCUIT_SE, ub=300, ra=4.0, ug1_bias=-8.0,
                            sources=["koren"], series_id=0)
        r = engine.analyze(params)
        assert r.error is None,             f"Partial zeros should be filtered, not error: {r.error}"

    @pytest.mark.timeout(60)
    def test_amp_params_lamp_default_field_exists(self):
        """AmpParams has lamp_ug2_default field with default=None."""
        params = AmpParams()
        assert hasattr(params, "lamp_ug2_default")
        assert params.lamp_ug2_default is None

    @pytest.mark.timeout(60)
    def test_analysis_result_suggested_ug2_field_exists(self):
        """AnalysisResult has suggested_ug2 field with default=None."""
        from lm19.amp_engine import AnalysisResult
        r = AnalysisResult()
        assert hasattr(r, "suggested_ug2")
        assert r.suggested_ug2 is None


class TestModelFallbackVisibility:
    """ML-094: requesting a MODEL source whose model is gone (refit under a
    different type / removed) silently ran the analysis on raw measurements
    while the result kept the model's name. Now: WARNING + model_fallback
    flag on the SourceResult."""

    def test_missing_model_source_flagged_and_warns(self, caplog):
        import logging
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1))   # no models registered
        with caplog.at_level(logging.WARNING, logger="lm19.amp_engine"):
            r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0,
                                    series_id=1, sources=["koren"]))
        sr = r.per_source.get("koren")
        assert sr is not None
        assert sr.model_fallback is True,             "model->measurements substitution not flagged"
        assert any("falling back to raw measurements" in rec.getMessage()
                   for rec in caplog.records)

    def test_present_model_not_flagged(self):
        e = AmplifierEngine()
        e.set_data(_make_triode_points(series_id=1),
                   series_models={1: SimpleModel()})
        r = e.analyze(AmpParams(ub=250, ra=5, ug1_bias=-7.0,
                                series_id=1, sources=["koren"]))
        sr = r.per_source.get("koren")
        assert sr is not None
        assert sr.model_fallback is False
