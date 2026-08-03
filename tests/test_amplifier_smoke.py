"""Smoke tests for the amplifier analysis pipeline.

Run:  py -m pytest tests/test_amplifier_smoke.py -v

End-to-end tests: tube_sim → amplifier → verify full pipeline.
Covers all supported tube types, topologies, and edge cases.
"""

from pathlib import Path

import pytest
import numpy as np

# ML-148: paths anchored to the repo, not CWD — a pytest run
# from outside lm19_app must not FileNotFoundError.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

from lm19.amplifier import (
    ResistiveLoadLine, TransformerLoadLine,
    CathodeFollowerLoadLine, PushPullLoadLine,
    find_intersections, interp_intersection,
    compute_distortion, compute_imd,
    compute_headroom,
    sweep_amplitude, sweep_ra, sweep_bias,
    optimize_bias,
    compute_stage_params, compute_cf_stage_params,
    composite_characteristic, pp_distortion, sweep_pp_amplitude,
    select_analysis_points, get_available_series,
    AMPLIFIER_PRESETS,
    _numerical_gm_ra,
)
from lm19.tube_sim import (
    load_model, quick_triode, quick_pentode,
    ScanGrid, TubeModel, TRIODE_PRESETS, PENTODE_PRESETS,
)
from lm19.tube_params import list_tubes, lookup_tube
from lm19.constants import (
    TOPOLOGY_TRIODE,
)


# ── module local constants ──
# Physical sanity bounds: THD above this means the analyzer collapsed
# numerically; HD above this is unphysical for any real tube.
_THD_MAX_PCT = 70.0
_HD_MAX_PCT = 100.0


class TestFullPipelineTriode:
    """End-to-end: tube_sim → all analysis for triodes."""

    def test_12ax7_full_pipeline(self):
        model, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3

        dist = compute_distortion(isects, ug1_bias=-2.0)
        assert dist is not None
        assert 0 <= dist["hd2"] < 50
        assert 0 <= dist["thd"] < _THD_MAX_PCT
        assert dist["pout_mw"] > 0

        imd = compute_imd(isects, ug1_bias=-2.0)
        hr = compute_headroom(isects, ug1_bias=-2.0, pa_max=1.0, load_line=ll)
        assert hr is not None

        amp_data = sweep_amplitude(pts, ll, ug1_bias=-2.0, steps=10)
        assert len(amp_data) > 0

        stage = compute_stage_params(isects, ll, ug1_bias=-2.0, points=pts)
        assert stage is not None
        assert stage["gain"] > 10  # 12AX7 has mu~100

        opt = optimize_bias(pts, ll, target="min_thd")
        assert opt is not None

    def test_12au7_full_pipeline(self):
        model, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)

        dist = compute_distortion(isects, ug1_bias=-10.0)
        assert dist is not None
        assert 0 <= dist["thd"] < _THD_MAX_PCT
        assert 0 <= dist["hd2"] < _HD_MAX_PCT
        assert dist["pout_mw"] > 0

        ra_data = sweep_ra(pts, ub=250, ra_min=2.0, ra_max=30.0, steps=10)
        assert len(ra_data) > 0

        bias_data = sweep_bias(pts, ll, steps=10)
        assert len(bias_data) > 0

        opt = optimize_bias(pts, ll, target="balanced")
        assert opt is not None
        assert opt["rk_auto_bias"] > 0

    def test_6sn7_full_pipeline(self):
        model, pts = quick_triode("6SN7")
        ll = ResistiveLoadLine(ub=300, ra=15.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3

        dist = compute_distortion(isects, ug1_bias=-8.0)
        assert dist is not None
        assert 0 <= dist["thd"] < _THD_MAX_PCT
        assert 0 <= dist["hd2"] < _HD_MAX_PCT
        assert dist["pout_mw"] > 0


class TestFullPipelinePentode:
    """End-to-end: tube_sim → all analysis for pentodes."""

    def test_el84_full_pipeline(self):
        model, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3

        dist = compute_distortion(isects, ug1_bias=-7.0)
        assert dist is not None
        assert dist["pout_mw"] > 10
        assert 0 <= dist["thd"] < _THD_MAX_PCT
        assert 0 <= dist["hd3"] < _HD_MAX_PCT

        hr = compute_headroom(isects, ug1_bias=-7.0, pa_max=12.0, load_line=ll)
        assert hr is not None

        amp_data = sweep_amplitude(pts, ll, ug1_bias=-7.0, steps=10)
        assert len(amp_data) > 0

        opt = optimize_bias(pts, ll, target="min_thd")
        assert opt is not None

    def test_el34_full_pipeline(self):
        model, pts = quick_pentode("EL34")
        ll = ResistiveLoadLine(ub=400, ra=3.5)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3

        dist = compute_distortion(isects, ug1_bias=-20.0)
        assert dist is not None
        assert 0 <= dist["thd"] < _THD_MAX_PCT
        assert 0 <= dist["hd3"] < _HD_MAX_PCT
        assert dist["pout_mw"] > 0

    def test_el84_with_ug2_filter(self):
        model = load_model("EL84")
        grid = PENTODE_PRESETS["EL84_multi_ug2"]
        pts = model.generate_scan(grid)
        ll = ResistiveLoadLine(ub=300, ra=5.0)

        isects_250 = find_intersections(pts, ll, ug2_filter=250.0)
        assert len(isects_250) >= 2

        isects_200 = find_intersections(pts, ll, ug2_filter=200.0)
        assert len(isects_200) >= 2


class TestTriodeConnectedPentode:
    """Pentode in triode connection (Ug2 = Ua)."""

    def test_el34_triode_connected(self):
        model = load_model("EL34")
        grid = ScanGrid(
            ua=(0, 400, 10), ug1=(-30, 0, 3),
            ug2_track_ua=True, ug2_offset=0, uh=6.3, ih=1.5,
        )
        pts = model.generate_scan(grid)
        for p in pts:
            assert abs(p["ug2"] - p["ua"]) < 0.01

        ll = ResistiveLoadLine(ub=350, ra=3.5)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3

        dist = compute_distortion(isects)
        assert dist is not None
        assert dist["pout_mw"] > 0
        assert 0 <= dist["thd"] < _THD_MAX_PCT
        assert 0 <= dist["hd2"] < _HD_MAX_PCT


class TestTransformerLoadLine:
    """Smoke tests for transformer-coupled stages."""

    def test_transformer_dc_ac(self):
        tll = TransformerLoadLine(ub=300, ra_dc=0.1, ra_ac=5.0)
        ia_dc = tll.ia_at_ua_dc(200)
        assert ia_dc > 0

        ia_ac = tll.ia_at_ua_ac(200, q_ua=290, q_ia=10.0)
        assert ia_ac > 0

    def test_intersections_with_transformer(self):
        _, pts = quick_pentode("EL84")
        tll = TransformerLoadLine(ub=300, ra_dc=0.1, ra_ac=5.0)
        isects = find_intersections(pts, tll)
        assert len(isects) >= 3


class TestStageParamsSRKvsNumerical:
    """Compare SRK-based and numerical stage parameter estimation."""

    def test_both_methods_produce_results(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)

        srk = {"s": 2.2, "r": 7.7, "k": 17.0}
        result_srk = compute_stage_params(isects, ll, ug1_bias=-10.0, srk=srk)
        result_num = compute_stage_params(isects, ll, ug1_bias=-10.0, points=pts)

        assert result_srk is not None
        assert result_num is not None
        assert result_srk["method"] == "srk"
        assert result_num["method"] == "numerical"

        # Both should give positive gain and Zout
        assert result_srk["gain"] > 0
        assert result_num["gain"] > 0
        assert result_srk["zout"] > 0
        assert result_num["zout"] > 0


class TestDataSourcePipeline:
    """Smoke tests for data source selection in multi-series scenario."""

    def test_mixed_current_and_overlay(self):
        model, current = quick_triode("12AX7")
        model2, overlay = quick_triode("12AU7")

        all_pts = []
        for p in current:
            p2 = dict(p)
            p2["series_id"] = 0
            all_pts.append(p2)
        for p in overlay:
            p2 = dict(p)
            p2["series_id"] = 1
            all_pts.append(p2)

        # Default: should use current scan (series_id=0)
        selected = select_analysis_points(all_pts)
        assert all(p["series_id"] == 0 for p in selected)
        assert len(selected) == len(current)

        # Specific series
        selected_1 = select_analysis_points(all_pts, series_id=1)
        assert len(selected_1) == len(overlay)

        # Available series
        labels = {1: "12AU7 overlay"}
        sources = get_available_series(all_pts, labels)
        assert len(sources) == 2  # current + 1 overlay


class TestPresetsPipeline:
    """Verify presets correspond to real tubes."""

    def test_all_presets_produce_analysis(self):
        checked = 0
        for preset in AMPLIFIER_PRESETS:
            model = load_model(preset.tube)
            if model is None:
                continue

            if model.topology == TOPOLOGY_TRIODE:
                grid = TRIODE_PRESETS.get(preset.tube)
            else:
                grid = PENTODE_PRESETS.get(preset.tube)

            if grid is None:
                continue

            pts = model.generate_scan(grid)
            ll = ResistiveLoadLine(preset.ub, preset.ra)
            isects = find_intersections(pts, ll)

            # checked counter: at least some presets MUST exercise the
            # full path instead of all being silently skipped.
            if len(isects) >= 3:
                dist = compute_distortion(isects, ug1_bias=preset.ug1_bias)
                assert dist is not None, f"Preset {preset.name} produced no distortion"
                checked += 1
        assert checked >= 3, f"only {checked} presets exercised the path"


class TestNoiseRobustnessSmoke:
    """Full pipeline with noisy data."""

    def test_noisy_12au7_pipeline(self):
        model, pts_clean = quick_triode("12AU7")
        pts_noisy = model.add_noise(pts_clean, sigma_pct=1.0, seed=42)

        ll = ResistiveLoadLine(ub=250, ra=10.0)

        isects = find_intersections(pts_noisy, ll)
        assert len(isects) >= 3

        dist = compute_distortion(isects, ug1_bias=-10.0)
        assert dist is not None

        hr = compute_headroom(isects, ug1_bias=-10.0)
        assert hr is not None

        opt = optimize_bias(pts_noisy, ll, target="min_thd")
        assert opt is not None

    def test_noisy_el84_pipeline(self):
        model, pts_clean = quick_pentode("EL84")
        pts_noisy = model.add_noise(pts_clean, sigma_pct=1.0, seed=99)

        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts_noisy, ll)
        assert len(isects) >= 3

        dist = compute_distortion(isects, ug1_bias=-7.0)
        assert dist is not None


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_very_high_ra(self):
        _, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=500.0)
        isects = find_intersections(pts, ll)
        # Very high Ra may produce few intersections but shouldn't crash
        if len(isects) >= 3:
            dist = compute_distortion(isects)

    def test_very_low_ra(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=0.5)
        isects = find_intersections(pts, ll)
        if len(isects) >= 3:
            dist = compute_distortion(isects)

    def test_bias_at_extremes(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)

        # Bias at very negative end
        dist = compute_distortion(isects, ug1_bias=-19.0)
        # May be None if not enough data — that's ok

        # Bias near 0V
        dist = compute_distortion(isects, ug1_bias=-1.0)

    def test_empty_data_graceful(self):
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        assert find_intersections([], ll) == []
        assert compute_distortion([]) is None
        assert compute_imd([]) is None
        assert compute_headroom([], ug1_bias=-5.0) is None
        assert sweep_amplitude([], ll, ug1_bias=-5.0) == []
        assert sweep_ra([], ub=250) == []
        assert sweep_bias([], ll) == []
        assert optimize_bias([], ll) is None


# ═══════════════════════════════════════════════════════════════════════════
#  ALL tubes from tube_params.json — parametrized
# ═══════════════════════════════════════════════════════════════════════════

def _all_tube_names():
    """Yield (name, topology) for every tube with Koren params in the DB."""
    for name in list_tubes():
        ref = lookup_tube(name)
        if ref and ref.koren:
            yield name, ref.topology

_TUBE_IDS = [(n, t) for n, t in _all_tube_names()]


class TestAllTubesParametrized:
    """Run basic pipeline for every tube in tube_params.json."""

    @pytest.mark.parametrize("tube_name,topology", _TUBE_IDS,
                             ids=[t[0] for t in _TUBE_IDS])
    def test_generate_and_intersect(self, tube_name, topology):
        model = load_model(tube_name)
        assert model is not None, f"Failed to load {tube_name}"

        if topology == TOPOLOGY_TRIODE:
            grid = ScanGrid(ua=(0, 300, 15), ug1=(-10, 0, 1), uh=6.3, ih=0.3)
        else:
            grid = ScanGrid(
                ua=(0, 400, 15), ug1=(-20, 0, 2),
                ug2=(250, 250, 1), uh=6.3, ih=0.9,
            )

        pts = model.generate_scan(grid)
        assert len(pts) > 0
        assert all(p["ia"] >= 0 for p in pts)

        ll = ResistiveLoadLine(ub=250 if topology == TOPOLOGY_TRIODE else 350, ra=10.0)
        isects = find_intersections(pts, ll)

        if len(isects) >= 3:
            dist = compute_distortion(isects)
            assert dist is None or dist["thd"] >= 0, \
                f"{tube_name}: negative THD"

    @pytest.mark.parametrize("tube_name,topology", _TUBE_IDS,
                             ids=[t[0] for t in _TUBE_IDS])
    def test_full_sweep_no_crash(self, tube_name, topology):
        """Every tube: generate → intersect → sweep_amplitude without crash."""
        model = load_model(tube_name)
        assert not (model is None), "silent skip de-vacuated"

        if topology == TOPOLOGY_TRIODE:
            grid = ScanGrid(ua=(0, 300, 15), ug1=(-10, 0, 1), uh=6.3, ih=0.3)
            ll = ResistiveLoadLine(ub=250, ra=15.0)
        else:
            grid = ScanGrid(
                ua=(0, 400, 15), ug1=(-20, 0, 2),
                ug2=(250, 250, 1), uh=6.3, ih=0.9,
            )
            ll = ResistiveLoadLine(ub=350, ra=5.0)

        pts = model.generate_scan(grid)
        sweep_amplitude(pts, ll, ug1_bias=-5.0 if topology == TOPOLOGY_TRIODE else -10.0,
                        steps=5)
        sweep_ra(pts, ub=ll.ub, steps=5)


# ═══════════════════════════════════════════════════════════════════════════
#  TransformerLoadLine — full pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestTransformerFullPipeline:
    """Transformer-coupled stage through the complete analysis chain."""

    def test_pentode_transformer_distortion_headroom_sweep(self):
        # quick_pentode("EL84") has Ug1 step 1.5V — too coarse for the AC
        # load-line truncation typical of TransformerLoadLine
        # (AC isects start at ~bias, leaving small max_swing on negative
        # side, which the sparse-data guard correctly rejects).
        # Use real 6P1P pentode (19 Ug1 levels, step ~0.5–1V) instead.
        import json
        with open(_PROJECT_ROOT / "tests/spice_test_data/converted/pentode_6P1P_real.json") as f:
            pts = json.load(f)["points"]
        tll = TransformerLoadLine(ub=250, ra_dc=0.1, ra_ac=5.0)
        isects = find_intersections(pts, tll, ug2_filter=200)
        assert len(isects) >= 3

        bias = -10.0
        dist = compute_distortion(isects, ug1_bias=bias)
        assert dist is not None
        assert dist["pout_mw"] > 0
        assert 0 <= dist["thd"] < _THD_MAX_PCT
        assert 0 <= dist["hd3"] < _HD_MAX_PCT

        hr = compute_headroom(isects, ug1_bias=bias, pa_max=12.0, load_line=tll)
        assert hr is not None
        assert hr["max_swing"] > 0

        amp_data = sweep_amplitude(pts, tll, ug1_bias=bias, ug2_filter=200, steps=8)
        assert len(amp_data) > 0

    def test_12au7_transformer_optimize(self):
        _, pts = quick_triode("12AU7")
        tll = TransformerLoadLine(ub=250, ra_dc=0.2, ra_ac=10.0)
        opt = optimize_bias(pts, tll, target="min_thd")
        assert opt is not None
        assert opt["thd"] >= 0

    def test_transformer_stage_params(self):
        _, pts = quick_pentode("EL84")
        tll = TransformerLoadLine(ub=300, ra_dc=0.1, ra_ac=5.0)
        isects = find_intersections(pts, tll)
        result = compute_stage_params(isects, tll, ug1_bias=-7.0, points=pts)
        assert result is not None
        assert result["gain"] > 0
        assert result["zout"] > 0


# ═══════════════════════════════════════════════════════════════════════════
#  Headroom reason codes
# ═══════════════════════════════════════════════════════════════════════════

class TestHeadroomReasons:
    """Verify that headroom returns correct clip_neg / clip_pos reasons."""

    def test_triode_grid_current_at_0v(self):
        """Scan up to Ug1=0 → clip_pos should be 'grid_current'."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        hr = compute_headroom(isects, ug1_bias=-10.0)
        assert hr is not None
        assert hr["clip_pos"] == "grid_current"

    def test_triode_cutoff(self):
        """Bias near cutoff → clip_neg should be 'cutoff'."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        hr = compute_headroom(isects, ug1_bias=-5.0)
        assert hr is not None
        assert hr["clip_neg"] in ("cutoff", "data_limit")

    def test_pentode_headroom_reasons(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)
        hr = compute_headroom(isects, ug1_bias=-7.0)
        assert hr is not None
        assert hr["clip_pos"] in ("grid_current", "data_limit")
        assert hr["clip_neg"] in ("cutoff", "data_limit")

    def test_data_limit_narrow_scan(self):
        """Narrow Ug1 range → at least one side should be 'data_limit'."""
        model = load_model("12AU7")
        grid = ScanGrid(ua=(0, 300, 15), ug1=(-12, -8, 1), uh=12.6, ih=0.15)
        pts = model.generate_scan(grid)
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3, "guard de-vacuated 2026-07-12: value must be present"
        hr = compute_headroom(isects, ug1_bias=-10.0)
        if hr:
            assert "data_limit" in (hr["clip_neg"], hr["clip_pos"])


# ═══════════════════════════════════════════════════════════════════════════
#  Pa_max constraining headroom
# ═══════════════════════════════════════════════════════════════════════════

class TestPaMaxHeadroom:
    """Pa_max should reduce headroom when thermal limit is hit."""

    def test_pa_max_reduces_swing(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)

        hr_unlimited = compute_headroom(isects, ug1_bias=-7.0)
        hr_limited = compute_headroom(
            isects, ug1_bias=-7.0, pa_max=0.5, load_line=ll,
        )
        assert hr_unlimited is not None
        assert hr_limited is not None
        assert hr_limited["max_swing"] <= hr_unlimited["max_swing"]

    def test_pa_max_reason_code(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)
        hr = compute_headroom(isects, ug1_bias=-7.0, pa_max=0.1, load_line=ll)
        assert hr, "guard de-vacuated 2026-07-12: value must be present"
        has_pa_reason = hr["clip_neg"] == "pa_max" or hr["clip_pos"] == "pa_max"
        assert has_pa_reason or hr["max_swing"] == 0


# ═══════════════════════════════════════════════════════════════════════════
#  Pentode stage params: Zout ≈ Ra (since ra >> Ra)
# ═══════════════════════════════════════════════════════════════════════════

class TestPentodeStageParams:
    """For pentodes ra >> Ra, so Zout = ra || Ra ≈ Ra."""

    def test_el84_zout_close_to_ra(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)

        result = compute_stage_params(isects, ll, ug1_bias=-7.0, points=pts)
        assert result is not None
        assert result["ra"] > result["zout"]
        assert abs(result["zout"] - 5.0) < 2.0  # Zout ≈ Ra ± tolerance

    def test_el34_high_ra_numerical(self):
        """EL34 pentode: internal ra is very high, Zout dominated by load."""
        _, pts = quick_pentode("EL34")
        ll = ResistiveLoadLine(ub=400, ra=3.5)
        isects = find_intersections(pts, ll)

        result = compute_stage_params(isects, ll, ug1_bias=-20.0, points=pts)
        assert result, "guard de-vacuated 2026-07-12: value must be present"
        assert result["zout"] < result["ra"]
        assert result["zout"] < 10.0

    def test_pentode_srk_high_ra(self):
        """With SRK data where ra >> Ra, Zout should be close to Ra."""
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)
        srk = {"s": 11.0, "r": 40.0, "k": 440.0}
        result = compute_stage_params(isects, ll, ug1_bias=-7.0, srk=srk)
        assert result is not None
        expected_zout = (40.0 * 5.0) / (40.0 + 5.0)  # 4.44 kOhm
        assert abs(result["zout"] - expected_zout) < 0.01
        assert result["zout"] < 5.0


# ═══════════════════════════════════════════════════════════════════════════
#  IMD on pentode
# ═══════════════════════════════════════════════════════════════════════════

class TestIMDPentode:
    """IMD computation for pentode tubes."""

    def test_el84_imd(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 4

        result = compute_imd(isects, ug1_bias=-7.0)
        assert result is not None
        assert result["imd2"] >= 0
        assert result["imd3"] >= 0

    def test_el34_imd(self):
        _, pts = quick_pentode("EL34")
        ll = ResistiveLoadLine(ub=400, ra=3.5)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 4, "guard de-vacuated 2026-07-12: value must be present"
        result = compute_imd(isects, ug1_bias=-20.0)
        assert result is not None
        assert result["imd2"] >= 0

    def test_imd_with_half_swing(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)
        result = compute_imd(isects, ug1_bias=-7.0, half_swing=3.0)
        assert result, "guard de-vacuated 2026-07-12: value must be present"
        assert result["imd2"] >= 0


# ═══════════════════════════════════════════════════════════════════════════
#  optimize_bias with specific half_swing (thd_limit proxy)
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimizeBiasExtended:
    """optimize_bias with half_swing and different targets."""

    def test_optimize_with_fixed_swing(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        result = optimize_bias(pts, ll, half_swing=3.0, target="min_thd")
        assert result is not None
        assert result["thd"] >= 0

    def test_optimize_pentode_max_pout(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        result = optimize_bias(pts, ll, target="max_pout")
        assert result is not None
        assert result["pout_mw"] > 0

    def test_optimize_pentode_balanced(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        result = optimize_bias(pts, ll, target="balanced")
        assert result is not None

    def test_min_thd_gives_lower_thd_than_random(self):
        """min_thd target should produce lower THD than a random bias."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        opt = optimize_bias(pts, ll, target="min_thd")
        assert opt is not None

        isects = find_intersections(pts, ll)
        random_bias_dist = compute_distortion(isects, ug1_bias=-3.0)
        if random_bias_dist and random_bias_dist["thd"] > 0:
            assert opt["thd"] <= random_bias_dist["thd"] + 0.1


# ═══════════════════════════════════════════════════════════════════════════
#  sweep_amplitude with constrained max_swing
# ═══════════════════════════════════════════════════════════════════════════

class TestSweepAmplitudeExtended:
    """sweep_amplitude respects headroom and produces consistent results."""

    def test_pentode_sweep_produces_imd(self):
        """Pentode sweep should include imd2/imd3 fields."""
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-7.0, steps=8)
        assert len(results) > 0
        assert "imd2" in results[0]
        assert "imd3" in results[0]

    def test_sweep_thd_monotonic_tendency(self):
        """THD should generally increase with amplitude (may have local dips)."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-10.0, steps=15)
        assert len(results) >= 3, "guard de-vacuated 2026-07-12: value must be present"
        assert results[-1]["thd"] >= results[0]["thd"]


# ═══════════════════════════════════════════════════════════════════════════
#  plotting.py wrappers — smoke test (no crash)
# ═══════════════════════════════════════════════════════════════════════════

class TestPlottingWrappers:
    """Verify plotting.py thin wrappers call through without crash."""

    def test_find_load_line_intersections_wrapper(self):
        from app.plotting import PlotRenderer
        _, pts = quick_triode("12AX7")
        isects = PlotRenderer._find_load_line_intersections(pts, ub=250, ra=100.0)
        assert len(isects) >= 3

    def test_interp_intersection_wrapper(self):
        from app.plotting import PlotRenderer
        isects = [
            {"ug1": -4.0, "ua": 200, "ia": 2.0},
            {"ug1": -2.0, "ua": 180, "ia": 4.0},
        ]
        result = PlotRenderer._interp_intersection(isects, -3.0)
        assert result is not None

    def test_compute_5point_distortion_wrapper(self):
        from app.plotting import PlotRenderer
        _, pts = quick_triode("12AU7")
        isects = PlotRenderer._find_load_line_intersections(pts, ub=250, ra=10.0)
        result = PlotRenderer._compute_5point_distortion(isects, ug1_bias=-10.0)
        assert result is not None

    def test_compute_imd_wrapper(self):
        from app.plotting import PlotRenderer
        _, pts = quick_triode("12AU7")
        isects = PlotRenderer._find_load_line_intersections(pts, ub=250, ra=10.0)
        result = PlotRenderer.compute_imd(isects, ug1_bias=-10.0)
        assert len(isects) >= 4, "guard de-vacuated 2026-07-12: value must be present"
        assert result is not None

    def test_compute_ra_sweep_wrapper(self):
        from app.plotting import PlotRenderer
        _, pts = quick_triode("12AU7")
        renderer = PlotRenderer.__new__(PlotRenderer)
        results = renderer.compute_ra_sweep(
            pts, ub=250, ra_min=2.0, ra_max=20.0, steps=5,
        )
        assert len(results) > 0

    def test_compute_ra_sweep_applies_ug2_filter(self):
        """The Ra sweep must honour ug2_filter so multi-Ug2 pentode data isn't
        mixed across screen levels (HD/Pout differ from the unfiltered sweep)."""
        from app.plotting import PlotRenderer
        model = load_model("EL84")
        pts = model.generate_scan(PENTODE_PRESETS["EL84_multi_ug2"])
        renderer = PlotRenderer.__new__(PlotRenderer)
        res_unf = renderer.compute_ra_sweep(
            pts, ub=300, ra_min=2.0, ra_max=15.0, steps=8, ug1_bias=-7.0)
        res_250 = renderer.compute_ra_sweep(
            pts, ub=300, ra_min=2.0, ra_max=15.0, steps=8, ug1_bias=-7.0,
            ug2_filter=250.0)
        assert res_unf and res_250
        # Filtering to one Ug2 slice changes the computed HD/Pout.
        assert any(abs(a["hd2"] - b["hd2"]) > 1e-6
                   or abs(a["pout_mw"] - b["pout_mw"]) > 1e-3
                   for a, b in zip(res_unf, res_250))


# ═══════════════════════════════════════════════════════════════════════════
#  _numerical_gm_ra direct unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNumericalGmRa:
    """Direct tests for _numerical_gm_ra."""

    def test_triode_returns_plausible_gm_ra(self):
        _, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        isects = find_intersections(pts, ll)

        result = _numerical_gm_ra(pts, isects, ug1_bias=-2.0)
        assert result is not None
        assert result["gm"] > 0.5   # 12AX7 gm ~ 1.6 mA/V
        assert result["gm"] < 5.0
        assert result["ra"] > 20    # 12AX7 ra ~ 62 kOhm (numerical approx)
        assert result["mu"] > 30    # 12AX7 mu ~ 100 (numerical approx)

    def test_pentode_high_ra(self):
        """Pentode has very high internal ra."""
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)

        result = _numerical_gm_ra(pts, isects, ug1_bias=-7.0)
        assert result, "guard de-vacuated 2026-07-12: value must be present"
        assert result["gm"] > 3.0   # EL84 gm ~ 11 mA/V (numerical approx)
        assert result["ra"] > 5.0    # pentode ra is high (numerical approx on discrete grid)

    def test_none_on_empty_points(self):
        isects = [{"ug1": -5, "ua": 200, "ia": 5}]
        assert _numerical_gm_ra(None, isects, -5.0) is None
        assert _numerical_gm_ra([], isects, -5.0) is None

    def test_none_on_single_ug1(self):
        """Only one Ug1 value → can't compute gm."""
        pts = [
            {"ua": 100, "ug1": -5.0, "ia": 3.0},
            {"ua": 200, "ug1": -5.0, "ia": 5.0},
        ]
        isects = [{"ug1": -5.0, "ua": 150, "ia": 4.0}]
        assert _numerical_gm_ra(pts, isects, -5.0) is None

    def test_12au7_gm_matches_srk_order(self):
        """Numerical gm should be in the same order of magnitude as SRK gm."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)

        result = _numerical_gm_ra(pts, isects, ug1_bias=-10.0)
        assert result is not None
        # 12AU7 reference: gm ~ 2.2 mA/V
        assert 0.5 < result["gm"] < 10.0


# ═══════════════════════════════════════════════════════════════════════════
#  Cathode Follower — full pipeline smoke
# ═══════════════════════════════════════════════════════════════════════════

class TestCFFullPipeline:
    """End-to-end cathode follower analysis."""

    def test_12au7_cf_pipeline(self):
        _, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3

        dist = compute_distortion(isects, ug1_bias=-10.0)
        assert dist is not None

        hr = compute_headroom(isects, ug1_bias=-10.0)
        assert hr is not None

        amp_data = sweep_amplitude(pts, ll, ug1_bias=-10.0, steps=8)
        assert len(amp_data) > 0

        stage = compute_cf_stage_params(isects, ll, ug1_bias=-10.0, points=pts)
        assert stage is not None
        assert stage["gain"] < 1.0
        assert stage["zout"] < 2.0  # CF low Zout

    def test_6sn7_cf_with_srk(self):
        _, pts = quick_triode("6SN7")
        ll = CathodeFollowerLoadLine(ub=300, rk=5.0, rl=5.0)
        isects = find_intersections(pts, ll)
        srk = {"s": 2.6, "r": 7.7, "k": 20.0}
        stage = compute_stage_params(isects, ll, ug1_bias=-8.0, srk=srk)
        assert stage is not None
        assert stage["gain"] < 1.0
        assert stage["gain"] > 0.5

    def test_cf_optimize_bias(self):
        _, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        result = optimize_bias(pts, ll, target="min_thd")
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
#  Push-Pull — full pipeline smoke
# ═══════════════════════════════════════════════════════════════════════════

class TestPPFullPipeline:
    """End-to-end push-pull analysis."""

    def test_12au7_pp_matched_pipeline(self):
        _, pts = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)

        comp = composite_characteristic(pts, ug1_bias=-10.0)
        assert len(comp) >= 5

        dist = pp_distortion(pts, ll, ug1_bias=-10.0)
        assert dist is not None
        assert dist["hd2"] < 5.0  # matched: HD2 should be low
        assert dist["pout_mw"] > 0

        amp_data = sweep_pp_amplitude(pts, ll, ug1_bias=-10.0, steps=10)
        assert len(amp_data) > 0

    def test_el84_pp_matched(self):
        _, pts = quick_pentode("EL84")
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)

        dist = pp_distortion(pts, ll, ug1_bias=-7.0)
        assert dist is not None
        assert dist["pout_mw"] > 0
        assert 0 <= dist["thd"] < _THD_MAX_PCT
        assert 0 <= dist["hd3"] < _HD_MAX_PCT

    def test_el34_pp_matched(self):
        _, pts = quick_pentode("EL34")
        ll = PushPullLoadLine(ub=400, ra_aa=6.6)

        dist = pp_distortion(pts, ll, ug1_bias=-20.0)
        assert dist is not None
        assert 0 <= dist["thd"] < _THD_MAX_PCT
        assert 0 <= dist["hd3"] < _HD_MAX_PCT
        assert dist["pout_mw"] > 0

    def test_pp_mismatched_pair(self):
        """Using two different tubes as PP pair — balance error should be high."""
        _, pts_a = quick_triode("12AU7")
        _, pts_b = quick_triode("12AX7")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)

        comp = composite_characteristic(pts_a, pts_b)
        assert len(comp) >= 5, "guard de-vacuated 2026-07-12: value must be present"
        dist = pp_distortion(pts_a, ll, ug1_bias=-5.0, points_b=pts_b)
        if dist:
            assert dist["balance_error"] > 0

    def test_pp_produces_power(self):
        """PP should produce meaningful power output."""
        _, pts = quick_pentode("EL84")
        ll_pp = PushPullLoadLine(ub=300, ra_aa=8.0)

        dist_pp = pp_distortion(pts, ll_pp, ug1_bias=-7.0)
        assert dist_pp is not None
        assert dist_pp["pout_mw"] > 0
        assert dist_pp["thd"] >= 0

    def test_pp_noisy_data(self):
        """PP pipeline with noisy data should not crash."""
        model, pts_clean = quick_pentode("EL84")
        pts_noisy = model.add_noise(pts_clean, sigma_pct=1.0, seed=42)
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)
        dist = pp_distortion(pts_noisy, ll, ug1_bias=-7.0)
        assert dist is not None
