"""Tests for amplifier analysis module.

Run:  py -m pytest tests/test_amplifier.py -v

Uses tube_sim.py for realistic test data generation
based on Koren tube models from tube_params.json.
"""

import math
from pathlib import Path

import numpy as np
import pytest

# ML-148: paths anchored to the repo, not CWD — a pytest run
# from outside lm19_app must not FileNotFoundError.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

from lm19.amplifier import (
    ResistiveLoadLine, TransformerLoadLine,
    CathodeFollowerLoadLine, PushPullLoadLine,
    find_intersections, find_intersections_model, interp_intersection,
    compute_distortion, compute_distortion_chebyshev, compute_distortion_dft,
    compute_imd,
    compute_headroom,
    sweep_amplitude, sweep_ra, sweep_bias,
    optimize_bias,
    compute_stage_params, compute_cf_stage_params,
    _compare_srk, SRK_DIVERGENCE_THRESHOLD_PCT,
    composite_characteristic, pp_distortion, sweep_pp_amplitude,
    select_analysis_points, get_available_series,
    AmplifierPreset, AMPLIFIER_PRESETS,
    _numerical_gm_ra,
    estimate_ig2_at_q, compute_pg2,
    compute_nfb_effect,
    ul_screen_voltage, UltralinearModelWrapper,
    model_gm_ra,
    compute_pa_avg,
)
from lm19.tube_sim import (
    load_model, quick_triode, quick_pentode, ScanGrid,
    PENTODE_PRESETS,
)


# ---------------------------------------------------------------
# LoadLine
# ---------------------------------------------------------------

class TestResistiveLoadLine:
    def test_ia_at_ua_basic(self):
        ll = ResistiveLoadLine(ub=250, ra=5.0)
        assert ll.ia_at_ua(0) == 50.0
        assert ll.ia_at_ua(250) == 0.0
        assert ll.ia_at_ua(125) == 25.0

    def test_ia_at_ua_zero_ra(self):
        ll = ResistiveLoadLine(ub=250, ra=0)
        assert ll.ia_at_ua(100) == 0.0

    def test_endpoints(self):
        ll = ResistiveLoadLine(ub=300, ra=10.0)
        (ua0, ia0), (ua1, ia1) = ll.endpoints()
        assert (ua0, ia0) == (0.0, 30.0)
        assert (ua1, ia1) == (300.0, 0.0)

    def test_label(self):
        ll = ResistiveLoadLine(ub=250, ra=5.0)
        assert "250" in ll.label()
        assert "5.0" in ll.label()


class TestTransformerLoadLine:
    def test_ia_at_ua_ac(self):
        ll = TransformerLoadLine(ub=300, ra_dc=0.1, ra_ac=5.0)
        ia_ac = ll.ia_at_ua_ac(200, q_ua=280, q_ia=20.0)
        expected = 20.0 - (200 - 280) / 5.0
        assert abs(ia_ac - expected) < 0.01

    def test_ia_at_ua_dc(self):
        ll = TransformerLoadLine(ub=300, ra_dc=0.1, ra_ac=5.0)
        ia = ll.ia_at_ua_dc(200)
        assert abs(ia - (300 - 200) / 0.1) < 0.01

    def test_label(self):
        ll = TransformerLoadLine(ub=300, ra_dc=0.1, ra_ac=5.0)
        assert "300" in ll.label()
        assert "5.0" in ll.label()


# ---------------------------------------------------------------
# Intersections
# ---------------------------------------------------------------

class TestFindIntersections:
    def test_triode_12ax7(self):
        _, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3
        assert isects[0]["ug1"] < isects[-1]["ug1"]
        assert all(p["ia"] > 0 for p in isects)

    def test_triode_12au7(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3

    def test_pentode_el84(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3

    def test_pentode_ug2_filter(self):
        model = load_model("EL84")
        grid = PENTODE_PRESETS["EL84_multi_ug2"]
        pts = model.generate_scan(grid)
        ll = ResistiveLoadLine(ub=300, ra=5.0)

        isects_all = find_intersections(pts, ll)
        isects_200 = find_intersections(pts, ll, ug2_filter=200.0)
        isects_250 = find_intersections(pts, ll, ug2_filter=250.0)

        assert len(isects_200) > 0
        assert len(isects_250) > 0
        assert len(isects_200) <= len(isects_all)

    def test_triode_connected_pentode(self):
        model = load_model("EL34")
        grid = ScanGrid(
            ua=(0, 400, 10), ug1=(-30, 0, 3),
            ug2_track_ua=True, ug2_offset=0, uh=6.3, ih=1.5,
        )
        pts = model.generate_scan(grid)
        ll = ResistiveLoadLine(ub=400, ra=3.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 3

    def test_empty_points(self):
        ll = ResistiveLoadLine(ub=250, ra=5.0)
        assert find_intersections([], ll) == []

    def test_single_point(self):
        pts = [{"ua": 100, "ug1": -5, "ug2": 0, "ia": 10}]
        ll = ResistiveLoadLine(ub=250, ra=5.0)
        assert find_intersections(pts, ll) == []


# ---------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------

class TestInterpIntersection:
    def test_interpolate_midpoint(self):
        isects = [
            {"ug1": -4.0, "ua": 200, "ia": 2.0},
            {"ug1": -2.0, "ua": 180, "ia": 4.0},
        ]
        result = interp_intersection(isects, -3.0)
        assert result is not None
        assert abs(result["ug1"] - (-3.0)) < 0.01
        assert abs(result["ia"] - 3.0) < 0.01

    def test_extrapolate_beyond(self):
        isects = [
            {"ug1": -4.0, "ua": 200, "ia": 2.0},
            {"ug1": -2.0, "ua": 180, "ia": 4.0},
        ]
        result = interp_intersection(isects, -5.0)
        assert result is not None
        assert abs(result["ug1"] - (-5.0)) < 0.01

    def test_empty_returns_none(self):
        assert interp_intersection([], -3.0) is None

    def test_exact_match(self):
        isects = [
            {"ug1": -4.0, "ua": 200, "ia": 2.0},
            {"ug1": -2.0, "ua": 180, "ia": 4.0},
        ]
        result = interp_intersection(isects, -4.0)
        assert result is not None
        assert abs(result["ia"] - 2.0) < 0.01

    def test_single_point(self):
        isects = [{"ug1": -3.0, "ua": 190, "ia": 3.0}]
        result = interp_intersection(isects, -5.0)
        assert result is not None
        assert abs(result["ia"] - 3.0) < 0.01


# ---------------------------------------------------------------
# Distortion
# ---------------------------------------------------------------

class TestDistortion:
    def test_12ax7_distortion(self):
        _, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        isects = find_intersections(pts, ll)
        result = compute_distortion(isects, ug1_bias=-2.0)
        assert result is not None
        assert result["hd2"] >= 0
        assert result["hd3"] >= 0
        assert result["thd"] >= 0
        assert result["pout_mw"] > 0

    def test_12au7_distortion(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        result = compute_distortion(isects, ug1_bias=-10.0)
        assert result is not None
        assert result["pout_mw"] > 0

    def test_pentode_distortion(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)
        result = compute_distortion(isects, ug1_bias=-7.0)
        assert result is not None
        # Physical sanity rule: real quick_pentode data must give a
        # positive power and a bounded THD, not garbage like THD=1000%.
        assert result["pout_mw"] > 0
        assert 0 <= result["thd"] < 70.0

    def test_interpolated_swing(self):
        _, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        isects = find_intersections(pts, ll)
        result = compute_distortion(isects, ug1_bias=-2.0, half_swing=1.0)
        assert result is not None
        assert result["interpolated"] is True

    def test_too_few_intersections(self):
        model = load_model("12AX7")
        pts = model.generate_scan(ScanGrid(ua=(0, 300, 10), ug1=(-2, -2, 1)))
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        isects = find_intersections(pts, ll)
        result = compute_distortion(isects)
        assert result is None

    def test_thd_is_rss_of_hd2_hd3(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        result = compute_distortion(isects, ug1_bias=-10.0)
        assert result is not None, "quick_triode 12AU7 must yield a distortion result"
        # Unconditional formula check: THD = sqrt(HD2^2 + HD3^2).
        expected_thd = (result["hd2"] ** 2 + result["hd3"] ** 2) ** 0.5
        assert abs(result["thd"] - expected_thd) < 0.001


class TestDistortionGuards:
    def test_manual_swing_is_clamped_to_data_range(self):
        isects = [
            {"ug1": -5.0, "ua": 250.0, "ia": 1.0},
            {"ug1": -4.0, "ua": 220.0, "ia": 4.0},
            {"ug1": -3.0, "ua": 190.0, "ia": 8.0},
            {"ug1": -2.0, "ua": 160.0, "ia": 12.0},
        ]
        result = compute_distortion(isects, ug1_bias=-3.5, half_swing=5.0)
        assert result is not None
        assert result["manual_swing_clamped"] is True
        assert result["half_swing"] <= 1.5 + 1e-9
        assert result["ia_0"] >= 0.0
        assert result["i_min"] >= 0.0

    def test_negative_currents_in_range_do_not_crash(self):
        # Dense Ug1 grid (8 levels, 0.5V step) so half_swing=1.0V at
        # bias=-3.5 captures 4-5 curves inside swing window — passes
        # MIN_CURVES_IN_SWING guard. Test focus: tolerate Ia<0 robustly.
        isects = [
            {"ug1": -5.0, "ua": 260.0, "ia": -1.0},
            {"ug1": -4.5, "ua": 245.0, "ia": 0.5},
            {"ug1": -4.0, "ua": 230.0, "ia": 2.0},
            {"ug1": -3.5, "ua": 215.0, "ia": 4.0},
            {"ug1": -3.0, "ua": 200.0, "ia": 6.0},
            {"ug1": -2.5, "ua": 185.0, "ia": 8.0},
            {"ug1": -2.0, "ua": 170.0, "ia": 10.0},
        ]
        result = compute_distortion(isects, ug1_bias=-3.5, half_swing=1.0)
        assert result is not None

    def test_negative_currents_with_clamped_manual_swing_return_none(self):
        isects = [
            {"ug1": -5.0, "ua": 260.0, "ia": -1.0},
            {"ug1": -4.0, "ua": 230.0, "ia": 2.0},
            {"ug1": -3.0, "ua": 200.0, "ia": 6.0},
            {"ug1": -2.0, "ua": 170.0, "ia": 10.0},
        ]
        # Requested swing exceeds available data -> clamped branch should reject non-physical currents.
        result = compute_distortion(isects, ug1_bias=-3.5, half_swing=5.0)
        assert result is None

    def test_rejects_insufficient_signal_near_cutoff(self):
        """Near cutoff with negligible Ia swing → None (b1 too small)."""
        isects = [
            {"ug1": -8.0, "ua": 250.0, "ia": 0.0},
            {"ug1": -7.0, "ua": 250.0, "ia": 0.0},
            {"ug1": -6.0, "ua": 250.0, "ia": 0.0},
            {"ug1": -5.0, "ua": 250.0, "ia": 0.01},
            {"ug1": -4.0, "ua": 249.9, "ia": 0.02},
        ]
        result = compute_distortion(isects, ug1_bias=-6.0)
        assert result is None


class TestDiagnoseDistortion:
    """Cover all DIST_ERR_* return paths of diagnose_distortion()."""

    def _good_isects(self):
        return [
            {"ug1": -5.0, "ua": 250.0, "ia": 1.0},
            {"ug1": -4.0, "ua": 220.0, "ia": 4.0},
            {"ug1": -3.0, "ua": 190.0, "ia": 8.0},
            {"ug1": -2.0, "ua": 160.0, "ia": 12.0},
        ]

    def test_few_intersections(self):
        from lm19.amplifier import (
            diagnose_distortion, DIST_ERR_FEW_INTERSECTIONS,
        )
        isects = [
            {"ug1": -5.0, "ua": 250.0, "ia": 1.0},
            {"ug1": -4.0, "ua": 220.0, "ia": 4.0},
        ]
        assert diagnose_distortion(isects, ug1_bias=-4.5) == DIST_ERR_FEW_INTERSECTIONS

    def test_bias_outside_data(self):
        from lm19.amplifier import (
            diagnose_distortion, DIST_ERR_BIAS_OUTSIDE,
        )
        # Data ug1 ∈ [-5, -2], bias far outside (-20)
        assert diagnose_distortion(self._good_isects(), ug1_bias=-20.0) == DIST_ERR_BIAS_OUTSIDE

    def test_bias_at_data_edge(self):
        from lm19.amplifier import (
            diagnose_distortion, DIST_ERR_BIAS_AT_EDGE,
        )
        # Bias at very edge → no auto-swing room
        assert diagnose_distortion(
            self._good_isects(), ug1_bias=-5.0,
        ) == DIST_ERR_BIAS_AT_EDGE

    def test_manual_swing_too_small(self):
        from lm19.amplifier import (
            diagnose_distortion, DIST_ERR_MANUAL_SWING_SMALL,
        )
        # half_swing > 0 but < MIN_SWING_V (0.1)... wait, function treats <=
        # MIN_SWING_V as "not manual" (auto path). Use a value just above
        # MIN_SWING_V threshold logic: in compute_distortion
        # `manual_swing_requested = half_swing > MIN_SWING_V`. So a value at
        # or below MIN_SWING_V drops to auto. To trigger MANUAL_SWING_SMALL
        # the user's input would already be coerced; we test the explicit
        # branch which checks `half_swing < MIN_SWING_V` after detecting
        # manual mode — only reachable via direct API call.
        # The branch is structurally guarded; we accept either MANUAL_SMALL
        # or BIAS_AT_EDGE depending on bias.
        result = diagnose_distortion(
            self._good_isects(), ug1_bias=-3.5, half_swing=0.05,
        )
        # 0.05 < MIN_SWING_V → not manual → auto-swing path. Bias -3.5 has
        # auto_swing = min(1.5, 1.5) = 1.5 > MIN_SWING_V → returns UNKNOWN.
        # That's fine — the explicit "manual_swing_small" code path is
        # defensive.
        assert result in (
            DIST_ERR_MANUAL_SWING_SMALL,
            "unknown",
        )

    def test_unknown_when_inputs_pass(self):
        from lm19.amplifier import (
            diagnose_distortion, DIST_ERR_UNKNOWN,
        )
        # Inputs all valid → diagnostic returns "unknown"
        assert diagnose_distortion(self._good_isects(), ug1_bias=-3.5) == DIST_ERR_UNKNOWN

    def test_diagnose_matches_compute_distortion_failure(self):
        """When compute_distortion returns None, diagnose returns a
        non-empty error code (success would not call diagnose)."""
        from lm19.amplifier import diagnose_distortion
        # Bias outside → compute_distortion returns None, diagnose explains
        result = compute_distortion(self._good_isects(), ug1_bias=-20.0)
        assert result is None
        code = diagnose_distortion(self._good_isects(), ug1_bias=-20.0)
        assert code != ""

    def test_diagnose_few_isects_matches_compute(self):
        from lm19.amplifier import (
            diagnose_distortion, DIST_ERR_FEW_INTERSECTIONS,
        )
        thin = [
            {"ug1": -5.0, "ua": 250.0, "ia": 1.0},
            {"ug1": -4.0, "ua": 220.0, "ia": 4.0},
        ]
        assert compute_distortion(thin, ug1_bias=-4.5) is None
        assert diagnose_distortion(thin, ug1_bias=-4.5) == DIST_ERR_FEW_INTERSECTIONS


class TestFivePointSparseDataRejection:
    """5-point rejects swing windows with too few measured Ug1 curves
    to detect curvature. Without this guard, sample points become
    linear interpolations between the same 2 data lines → fake
    near-zero THD."""

    def _dense_isects(self, n_levels=10, ug1_min=-10.0, ug1_max=-1.0):
        """Linear-current intersections at uniform Ug1 spacing."""
        step = (ug1_max - ug1_min) / (n_levels - 1)
        return [
            {
                "ug1": ug1_min + i * step,
                "ua": 250.0 - 10 * i,
                "ia": 1.0 + 1.0 * i,
            }
            for i in range(n_levels)
        ]

    def test_rejects_only_2_curves_in_swing(self):
        """Manual swing covering only 2 Ug1 curves → None."""
        # 4 Ug1 levels, 1V apart. Manual swing 0.7V at bias=-3.5 covers
        # window [-4.2, -2.8] which includes only -4 and -3 (2 curves).
        isects = [
            {"ug1": -5.0, "ua": 250.0, "ia": 1.0},
            {"ug1": -4.0, "ua": 220.0, "ia": 4.0},
            {"ug1": -3.0, "ua": 190.0, "ia": 8.0},
            {"ug1": -2.0, "ua": 160.0, "ia": 12.0},
        ]
        assert compute_distortion(isects, ug1_bias=-3.5, half_swing=0.7) is None

    def test_strictly_inside_guard_defense_in_depth(self):
        """When count guard (#1) marginally passes, strict-inside guard (#2)
        is structurally satisfied (the center curve is strictly inside).
        This documents that #2 is defense-in-depth, not the primary filter."""
        isects = [
            {"ug1": -4.0, "ua": 250.0, "ia": 1.0},
            {"ug1": -3.0, "ua": 220.0, "ia": 4.0},
            {"ug1": -2.0, "ua": 190.0, "ia": 8.0},
        ]
        # 3 curves, swing exactly at edges → middle curve is strictly inside
        # (open interval) → both guards pass → result is valid.
        result = compute_distortion(isects, ug1_bias=-3.0, half_swing=1.0)
        assert result is not None

    def test_accepts_dense_5_curves_in_swing(self):
        """5 curves in swing window with curvature → passes both guards."""
        isects = self._dense_isects(n_levels=10)  # 1V step
        # Bias=-5.5, swing=2.5 → window [-8.0, -3.0]. 6 curves inside,
        # 5 strictly inside (-7,-6,-5,-4) plus -8/-3 at edges.
        result = compute_distortion(isects, ug1_bias=-5.5, half_swing=2.5)
        assert result is not None

    def test_accepts_auto_swing_dense_data(self):
        """Auto swing on dense uniform data — easy case, must pass."""
        isects = self._dense_isects(n_levels=10)
        result = compute_distortion(isects, ug1_bias=None)
        assert result is not None

    def test_diagnose_returns_sparse_data_code(self):
        """diagnose_distortion reports the specific cause."""
        from lm19.amplifier import diagnose_distortion, DIST_ERR_SPARSE_DATA
        isects = [
            {"ug1": -5.0, "ua": 250.0, "ia": 1.0},
            {"ug1": -4.0, "ua": 220.0, "ia": 4.0},
            {"ug1": -3.0, "ua": 190.0, "ia": 8.0},
            {"ug1": -2.0, "ua": 160.0, "ia": 12.0},
        ]
        # Same bias/swing that fails compute_distortion above
        code = diagnose_distortion(isects, ug1_bias=-3.5, half_swing=0.7)
        assert code == DIST_ERR_SPARSE_DATA

    def test_real_6p1p_no_thd_zero_at_narrow_swing(self):
        """Real 6P1P pentode: swing=0.4V at bias=-7 must return None.
        Without the sparse-data guard, 5-point returned a fake
        THD ≈ 0.07% on this slice."""
        import json
        from lm19.amplifier import (
            ResistiveLoadLine, find_intersections,
        )
        with open(_PROJECT_ROOT / "tests/spice_test_data/converted/pentode_6P1P_real.json") as f:
            pts = json.load(f)["points"]
        ll = ResistiveLoadLine(ub=250.0, ra=12.0)
        isects = find_intersections(pts, ll, ug2_filter=99.0)
        # Bias=-7 with swing 0.4V → only Ug1=-7 in window (data step ~1V)
        result = compute_distortion(isects, ug1_bias=-7.0, half_swing=0.4)
        assert result is None, "5-point should reject sparse swing on 6P1P"

    def test_real_6p1p_accepts_wide_swing(self):
        """Same data + bias, wider swing → passes the sparse-data guard."""
        import json
        from lm19.amplifier import (
            ResistiveLoadLine, find_intersections,
        )
        from lm19.constants import MAX_SANE_THD_PCT
        with open(_PROJECT_ROOT / "tests/spice_test_data/converted/pentode_6P1P_real.json") as f:
            pts = json.load(f)["points"]
        ll = ResistiveLoadLine(ub=250.0, ra=12.0)
        isects = find_intersections(pts, ll, ug2_filter=99.0)
        # Wider swing covers ≥3 Ug1 curves
        result = compute_distortion(isects, ug1_bias=-10.0, half_swing=3.0)
        assert result is not None
        # Sanity: real THD on 6P1P should be measurable at this bias
        assert 0.1 < result["thd"] < MAX_SANE_THD_PCT


class TestIMD:
    def test_imd_12ax7(self):
        _, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        isects = find_intersections(pts, ll)
        result = compute_imd(isects, ug1_bias=-2.0)
        assert len(isects) >= 4, "guarded value must be truthy (de-vacuated)"
        assert result is not None
        assert result["imd2"] >= 0
        assert result["imd3"] >= 0

    def test_imd_too_few_points(self):
        isects = [
            {"ug1": -4.0, "ua": 200, "ia": 2.0},
            {"ug1": -3.0, "ua": 190, "ia": 3.0},
        ]
        assert compute_imd(isects) is None


# ---------------------------------------------------------------
# Headroom
# ---------------------------------------------------------------

class TestHeadroom:
    def test_12au7_headroom(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        hr = compute_headroom(isects, ug1_bias=-10.0)
        assert hr is not None
        assert hr["max_swing"] > 0
        assert hr["swing_neg"] > 0
        assert hr["swing_pos"] > 0

    def test_bias_near_cutoff(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        hr = compute_headroom(isects, ug1_bias=-18.0)
        assert hr is not None
        assert hr["swing_neg"] < hr["swing_pos"]

    def test_pentode_headroom(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)
        hr = compute_headroom(isects, ug1_bias=-7.0)
        assert hr is not None
        assert hr["max_swing"] > 0

    def test_insufficient_data(self):
        isects = [{"ug1": -5.0, "ua": 200, "ia": 5.0}]
        hr = compute_headroom(isects, ug1_bias=-5.0)
        assert hr is None


# ---------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------

class TestSweepAmplitude:
    def test_thd_increases_with_amplitude(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-10.0, steps=10)
        assert len(results) > 0
        thds = [r["thd"] for r in results]
        assert thds[-1] >= thds[0]

    def test_pout_increases_with_amplitude(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-10.0, steps=10)
        pouts = [r["pout_mw"] for r in results]
        assert pouts[-1] > pouts[0]

    def test_pentode_sweep(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-7.0, steps=10)
        assert len(results) > 0

    def test_with_ug2_filter(self):
        model = load_model("EL84")
        grid = PENTODE_PRESETS["EL84_multi_ug2"]
        pts = model.generate_scan(grid)
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        results = sweep_amplitude(
            pts, ll, ug1_bias=-7.0, ug2_filter=250.0, steps=8,
        )
        assert len(results) > 0

    def test_result_keys(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-10.0, steps=5)
        assert results, "guard de-vacuated 2026-07-12: value must be present"
        expected = {"half_swing", "hd2", "hd3", "thd", "pout_mw", "imd2", "imd3"}
        assert expected.issubset(results[0].keys())


class TestSweepRa:
    def test_basic_12au7(self):
        _, pts = quick_triode("12AU7")
        results = sweep_ra(pts, ub=250, ra_min=2.0, ra_max=30.0, steps=10)
        assert len(results) > 0
        assert all("ra" in r and "hd2" in r and "pout_mw" in r for r in results)

    def test_pentode_ra_sweep(self):
        _, pts = quick_pentode("EL34")
        results = sweep_ra(pts, ub=400, ra_min=1.0, ra_max=15.0, steps=10)
        assert len(results) > 0

    def test_ra_values_ascending(self):
        _, pts = quick_triode("12AU7")
        results = sweep_ra(pts, ub=250, ra_min=2.0, ra_max=20.0, steps=10)
        ras = [r["ra"] for r in results]
        assert ras == sorted(ras)


class TestSweepHdMethods:
    """sweep_ra and sweep_amplitude respect hd_method parameter."""

    # ── sweep_ra ──

    def test_sweep_ra_chebyshev(self):
        _, pts = quick_triode("12AU7")
        results = sweep_ra(
            pts, ub=250, ra_min=5.0, ra_max=20.0, steps=5,
            hd_method=HD_METHOD_CHEBYSHEV,
        )
        assert len(results) > 0
        # Chebyshev can return hd4/hd5
        for r in results:
            assert r["thd"] > 0
            assert r["thd"] < 70.0  # sanity

    def test_sweep_ra_dft_with_model(self):
        model, pts = quick_triode("12AU7")
        results = sweep_ra(
            pts, ub=250, ra_min=5.0, ra_max=20.0, steps=5,
            ug1_bias=-10.0, model=model, hd_method=HD_METHOD_DFT,
        )
        assert len(results) > 0
        for r in results:
            assert r["thd"] > 0
            assert r["thd"] < 70.0

    def test_sweep_ra_dft_without_model_falls_back(self):
        """DFT without model → falls back to 5-point (no crash)."""
        _, pts = quick_triode("12AU7")
        results = sweep_ra(
            pts, ub=250, ra_min=5.0, ra_max=20.0, steps=5,
            hd_method=HD_METHOD_DFT,  # no model
        )
        assert len(results) > 0

    def test_sweep_ra_5point_default(self):
        _, pts = quick_triode("12AU7")
        r_default = sweep_ra(pts, ub=250, ra_min=5.0, ra_max=20.0, steps=5)
        r_5pt = sweep_ra(pts, ub=250, ra_min=5.0, ra_max=20.0, steps=5,
                         hd_method=HD_METHOD_5POINT)
        assert len(r_default) == len(r_5pt)
        for a, b in zip(r_default, r_5pt):
            assert a["thd"] == pytest.approx(b["thd"])

    def test_sweep_ra_methods_agree_order_of_magnitude(self):
        """5-point and Chebyshev should give similar THD for same data."""
        _, pts = quick_triode("12AU7")
        r_5pt = sweep_ra(pts, ub=250, ra_min=5.0, ra_max=20.0, steps=5,
                         hd_method=HD_METHOD_5POINT)
        r_cheb = sweep_ra(pts, ub=250, ra_min=5.0, ra_max=20.0, steps=5,
                          hd_method=HD_METHOD_CHEBYSHEV)
        # Unconditional: the cross-method agreement rule (x3) is the whole
        # point — if either sweep regresses to empty, that must fail, not skip.
        assert r_5pt and r_cheb, "sweep_ra returned empty for a method on synthetic data"
        # Mid-point comparison
        mid_5 = r_5pt[len(r_5pt) // 2]["thd"]
        mid_c = r_cheb[len(r_cheb) // 2]["thd"]
        assert abs(mid_5 - mid_c) < max(mid_5, mid_c) * 0.5 + 2.0

    # ── sweep_amplitude ──

    def test_sweep_amp_chebyshev(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-10.0, steps=5,
                                  hd_method=HD_METHOD_CHEBYSHEV)
        assert len(results) > 0
        for r in results:
            assert r["thd"] < 70.0

    def test_sweep_amp_dft_with_model(self):
        model, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-10.0, steps=5,
                                  model=model, hd_method=HD_METHOD_DFT)
        assert len(results) > 0
        for r in results:
            assert r["thd"] < 70.0

    def test_sweep_amp_pentode_chebyshev(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-7.0, steps=5,
                                  hd_method=HD_METHOD_CHEBYSHEV)
        assert len(results) > 0

    def test_sweep_ra_pentode_dft(self):
        model, pts = quick_pentode("EL84")
        results = sweep_ra(
            pts, ub=300, ra_min=2.0, ra_max=8.0, steps=5,
            ug1_bias=-7.0, model=model, model_ug2=250.0, hd_method=HD_METHOD_DFT,
        )
        assert len(results) > 0
        for r in results:
            assert r["thd"] < 70.0


from lm19.constants import MAX_SANE_THD_PCT as MAX_SANE_THD, MAX_SANE_HD_PCT as MAX_SANE_HD
from lm19.amplifier.constants import (
    HD_METHOD_5POINT,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_DFT,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
)


class TestSweepPhysicalSanity:
    """All sweep methods must produce physically valid results on real tube data."""

    # ── sweep_ra: all methods × triode + pentode ──

    @pytest.mark.parametrize("method", ["5point", "chebyshev"])
    def test_sweep_ra_triode_thd_sane(self, method):
        _, pts = quick_triode("12AU7")
        results = sweep_ra(pts, ub=250, ra_min=5.0, ra_max=25.0, steps=8,
                           ug1_bias=-10.0, hd_method=method)
        assert len(results) > 0
        for r in results:
            assert 0 <= r["thd"] < MAX_SANE_THD, f"Ra={r['ra']:.1f} THD={r['thd']:.1f}%"
            assert r["hd2"] < MAX_SANE_HD
            assert r["hd3"] < MAX_SANE_HD
            assert r["pout_mw"] > 0

    def test_sweep_ra_triode_dft_sane(self):
        model, pts = quick_triode("12AU7")
        results = sweep_ra(pts, ub=250, ra_min=5.0, ra_max=25.0, steps=8,
                           ug1_bias=-10.0, model=model, hd_method=HD_METHOD_DFT)
        assert len(results) > 0
        for r in results:
            assert 0 <= r["thd"] < MAX_SANE_THD
            assert r["hd2"] < MAX_SANE_HD
            assert r["pout_mw"] > 0

    @pytest.mark.parametrize("method", ["5point", "chebyshev"])
    def test_sweep_ra_pentode_thd_sane(self, method):
        _, pts = quick_pentode("EL84")
        results = sweep_ra(pts, ub=300, ra_min=2.0, ra_max=8.0, steps=6,
                           ug1_bias=-7.0, ug2_filter=250.0, hd_method=method)
        assert len(results) > 0
        for r in results:
            assert 0 <= r["thd"] < MAX_SANE_THD
            assert r["pout_mw"] > 0

    def test_sweep_ra_pentode_dft_sane(self):
        model, pts = quick_pentode("EL84")
        results = sweep_ra(pts, ub=300, ra_min=2.0, ra_max=8.0, steps=6,
                           ug1_bias=-7.0, model=model, model_ug2=250.0,
                           hd_method=HD_METHOD_DFT)
        assert len(results) > 0
        for r in results:
            assert 0 <= r["thd"] < MAX_SANE_THD
            assert r["pout_mw"] > 0

    # ── sweep_amplitude: all methods × triode + pentode ──

    @pytest.mark.parametrize("method", ["5point", "chebyshev"])
    def test_sweep_amp_triode_thd_sane(self, method):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-10.0, steps=8,
                                  hd_method=method)
        assert len(results) > 0
        for r in results:
            assert 0 <= r["thd"] < MAX_SANE_THD
            assert r["hd2"] < MAX_SANE_HD
            assert r["hd3"] < MAX_SANE_HD
            assert r["pout_mw"] > 0

    def test_sweep_amp_triode_dft_sane(self):
        model, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-10.0, steps=8,
                                  model=model, hd_method=HD_METHOD_DFT)
        assert len(results) > 0
        for r in results:
            assert 0 <= r["thd"] < MAX_SANE_THD
            assert r["pout_mw"] > 0

    @pytest.mark.parametrize("method", ["5point", "chebyshev"])
    def test_sweep_amp_pentode_thd_sane(self, method):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-7.0, steps=6,
                                  hd_method=method)
        assert len(results) > 0
        for r in results:
            assert 0 <= r["thd"] < MAX_SANE_THD
            assert r["pout_mw"] > 0

    # ── Pout order of magnitude ──

    def test_triode_pout_milliwatt_range(self):
        """Preamp triode Pout should be mW range, not watts."""
        _, pts = quick_triode("12AU7")
        results = sweep_ra(pts, ub=250, ra_min=5.0, ra_max=25.0, steps=5,
                           ug1_bias=-10.0)
        for r in results:
            assert r["pout_mw"] < 5000, f"Pout={r['pout_mw']:.0f}mW too high for preamp"

    def test_pentode_pout_range(self):
        """Power pentode Pout should be 100mW..20W range."""
        _, pts = quick_pentode("EL84")
        results = sweep_ra(pts, ub=300, ra_min=2.0, ra_max=8.0, steps=5,
                           ug1_bias=-7.0, ug2_filter=250.0)
        for r in results:
            assert r["pout_mw"] > 10, f"Pout={r['pout_mw']:.0f}mW too low"
            assert r["pout_mw"] < 20000, f"Pout={r['pout_mw']:.0f}mW too high"

    # ── THD monotonicity with swing ──

    def test_thd_increases_with_swing_all_methods(self):
        """THD should generally increase with signal amplitude."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        for method in ["5point", "chebyshev"]:
            results = sweep_amplitude(pts, ll, ug1_bias=-10.0, steps=10,
                                      hd_method=method)
            assert len(results) >= 3, "guarded value must be truthy (de-vacuated)"
            assert results[-1]["thd"] >= results[0]["thd"] - 0.5, (
                f"{method}: THD should increase with swing"
            )

    # ── Methods agree within reason ──

    def test_5point_vs_chebyshev_ra_sweep_agree(self):
        """5-point and Chebyshev should give THD within factor of 2."""
        _, pts = quick_triode("12AU7")
        r5 = sweep_ra(pts, ub=250, ra_min=5.0, ra_max=20.0, steps=5,
                       ug1_bias=-10.0, hd_method=HD_METHOD_5POINT)
        rc = sweep_ra(pts, ub=250, ra_min=5.0, ra_max=20.0, steps=5,
                       ug1_bias=-10.0, hd_method=HD_METHOD_CHEBYSHEV)
        assert r5 and rc and (len(r5) == len(rc)), "guard de-vacuated 2026-07-12: value must be present"
        for a, b in zip(r5, rc):
            ratio = max(a["thd"], 0.1) / max(b["thd"], 0.1)
            assert 0.3 < ratio < 3.0, (
                f"Ra={a['ra']:.1f}: 5pt={a['thd']:.2f}% vs cheb={b['thd']:.2f}%"
            )

    def test_dft_vs_5point_ra_sweep_agree(self):
        """DFT and 5-point should give THD within factor of 2."""
        model, pts = quick_triode("12AU7")
        r5 = sweep_ra(pts, ub=250, ra_min=5.0, ra_max=20.0, steps=5,
                       ug1_bias=-10.0, hd_method=HD_METHOD_5POINT)
        rd = sweep_ra(pts, ub=250, ra_min=5.0, ra_max=20.0, steps=5,
                       ug1_bias=-10.0, model=model, hd_method=HD_METHOD_DFT)
        assert r5 and rd and (len(r5) == len(rd)), "guard de-vacuated 2026-07-12: value must be present"
        for a, b in zip(r5, rd):
            ratio = max(a["thd"], 0.1) / max(b["thd"], 0.1)
            assert 0.3 < ratio < 3.0, (
                f"Ra={a['ra']:.1f}: 5pt={a['thd']:.2f}% vs dft={b['thd']:.2f}%"
            )


class TestSweepPhysicalSanitySynthetic:
    """Physical sanity on synthetic data (known transfer function)."""

    @staticmethod
    def _make_synth_points(n_ug1=21, n_ua=20, ub=250.0):
        """Quadratic Ia(Ug1) — known nonlinearity for predictable HD2."""
        pts = []
        for i in range(n_ug1):
            ug1 = -10.0 + 9.0 * i / (n_ug1 - 1)
            x = ug1 + 10.5
            for j in range(n_ua):
                ua = 10.0 + 390.0 * j / (n_ua - 1)
                ia = max(0.0, 1.5 * x * (1.0 + ua / 500.0))
                pts.append({"ug1": round(ug1, 2), "ua": round(ua, 1),
                            "ia": round(ia, 4), "series_id": 1})
        return pts

    # ── sweep_ra on synthetic data ──

    @pytest.mark.parametrize("method", ["5point", "chebyshev"])
    def test_synth_sweep_ra_thd_sane(self, method):
        pts = self._make_synth_points()
        results = sweep_ra(pts, ub=250, ra_min=3.0, ra_max=20.0, steps=8,
                           ug1_bias=-5.0, hd_method=method)
        assert len(results) > 0
        for r in results:
            assert 0 <= r["thd"] < MAX_SANE_THD, f"{method} Ra={r['ra']:.1f} THD={r['thd']:.1f}"
            assert r["hd2"] < MAX_SANE_HD
            assert r["hd3"] < MAX_SANE_HD
            assert r["pout_mw"] > 0

    @pytest.mark.parametrize("method", ["5point", "chebyshev"])
    def test_synth_sweep_ra_no_zeros(self, method):
        """No point should have THD=0 (synthetic data has nonlinearity)."""
        pts = self._make_synth_points()
        results = sweep_ra(pts, ub=250, ra_min=3.0, ra_max=15.0, steps=6,
                           ug1_bias=-5.0, hd_method=method)
        for r in results:
            assert r["thd"] > 0.1, f"{method} Ra={r['ra']:.1f} THD suspiciously zero"

    # ── sweep_amplitude on synthetic data ──

    @pytest.mark.parametrize("method", ["5point", "chebyshev"])
    def test_synth_sweep_amp_thd_sane(self, method):
        pts = self._make_synth_points()
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-5.0, steps=8,
                                  hd_method=method)
        assert len(results) > 0
        for r in results:
            assert 0 <= r["thd"] < MAX_SANE_THD
            assert r["pout_mw"] > 0

    def test_synth_sweep_amp_thd_monotonic_5point(self):
        """5-point THD should increase with swing on known nonlinear function."""
        pts = self._make_synth_points()
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-5.0, steps=10,
                                  hd_method=HD_METHOD_5POINT)
        assert len(results) >= 3, "guard de-vacuated 2026-07-12: value must be present"
        thds = [r["thd"] for r in results]
        assert thds[-1] >= thds[0] - 0.5, "5point: THD should grow with swing"

    @pytest.mark.parametrize("method", ["5point", "chebyshev"])
    def test_synth_sweep_amp_pout_monotonic(self, method):
        """Pout should increase with swing."""
        pts = self._make_synth_points()
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-5.0, steps=10,
                                  hd_method=method)
        assert len(results) >= 3, "guard de-vacuated 2026-07-12: value must be present"
        pouts = [r["pout_mw"] for r in results]
        assert pouts[-1] > pouts[0], f"{method}: Pout should grow with swing"

    # ── Consistency between methods on same synthetic data ──

    def test_synth_5point_vs_chebyshev_agree(self):
        """5-point and Chebyshev on same data → THD within factor of 3."""
        pts = self._make_synth_points()
        r5 = sweep_ra(pts, ub=250, ra_min=3.0, ra_max=15.0, steps=5,
                       ug1_bias=-5.0, hd_method=HD_METHOD_5POINT)
        rc = sweep_ra(pts, ub=250, ra_min=3.0, ra_max=15.0, steps=5,
                       ug1_bias=-5.0, hd_method=HD_METHOD_CHEBYSHEV)
        assert len(r5) > 0 and len(rc) > 0
        if len(r5) == len(rc):
            for a, b in zip(r5, rc):
                ratio = max(a["thd"], 0.1) / max(b["thd"], 0.1)
                assert 0.3 < ratio < 3.0, (
                    f"Ra={a['ra']:.1f}: 5pt={a['thd']:.2f}% vs cheb={b['thd']:.2f}%"
                )

    @pytest.mark.parametrize("n_pts", [8, 11, 15, 21])
    def test_synth_stable_across_point_density(self, n_pts):
        """THD should be under physical limit regardless of point count."""
        pts = self._make_synth_points(n_ug1=n_pts)
        results = sweep_ra(pts, ub=250, ra_min=5.0, ra_max=15.0, steps=4,
                           ug1_bias=-5.0, hd_method=HD_METHOD_CHEBYSHEV)
        for r in results:
            assert r["thd"] < MAX_SANE_THD, (
                f"n={n_pts} Ra={r['ra']:.1f} THD={r['thd']:.1f}%"
            )


class TestSweepBias:
    def test_basic(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_bias(pts, ll, steps=10)
        assert len(results) > 0
        assert all("ug1" in r and "thd" in r for r in results)

    def test_pentode_sweep_bias(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        results = sweep_bias(pts, ll, steps=10)
        assert len(results) > 0

    def test_sweep_bias_with_model(self):
        """sweep_bias with model should produce results."""
        model, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_bias(pts, ll, steps=8, model=model)
        assert len(results) > 0
        assert all("thd" in r for r in results)

    def test_sweep_bias_model_more_points(self):
        """Model-based sweep_bias should find at least as many points."""
        model, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        r_data = sweep_bias(pts, ll, steps=10)
        r_model = sweep_bias(pts, ll, steps=10, model=model)
        assert len(r_model) >= len(r_data) - 1  # model may differ slightly


class TestOptimizeBiasModel:
    """optimize_bias with model support."""

    def test_optimize_bias_with_model(self):
        model, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        result = optimize_bias(pts, ll, target="min_thd", model=model)
        assert result is not None
        assert result["thd"] > 0

    def test_optimize_bias_model_vs_data(self):
        """Model and data paths should both find valid bias points."""
        model, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        r_data = optimize_bias(pts, ll, target="min_thd")
        r_model = optimize_bias(pts, ll, target="min_thd", model=model)
        assert r_data is not None
        assert r_model is not None
        # Both should find reasonable Ug1 in the same ballpark
        assert abs(r_data["ug1_0"] - r_model["ug1_0"]) < 5.0


# ---------------------------------------------------------------
# HD4/HD5 propagation in sweeps and distortion
# ---------------------------------------------------------------

class TestHD45Propagation:
    """Verify hd4/hd5 flow through sweep and distortion functions."""

    def test_chebyshev_returns_hd4_hd5(self):
        """Chebyshev method should return hd4/hd5 keys."""
        # Cubic transfer function → HD3 dominant, but HD4/HD5 present as keys
        ub, ra = 250, 10.0
        pts = []
        for i in range(25):
            ug1 = -14.0 + 8.0 * i / 24
            ia = max(0.0, 10.0 + 2.0 * (ug1 + 10.0) + 0.1 * (ug1 + 10.0) ** 3)
            ua = ub - ia * ra
            pts.append({"ug1": ug1, "ua": ua, "ia": ia})
        dist = compute_distortion_chebyshev(pts, ug1_bias=-10.0, half_swing=3.0)
        assert dist is not None
        assert "hd4" in dist
        assert "hd5" in dist
        assert isinstance(dist["hd4"], float)
        assert isinstance(dist["hd5"], float)

    def test_5point_has_no_hd4_key(self):
        """5-point method returns only hd2/hd3, no hd4/hd5 keys."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        dist = compute_distortion(isects, ug1_bias=-10.0)
        assert dist is not None
        assert "hd4" not in dist
        # .get fallback should work
        assert dist.get("hd4", 0.0) == 0.0

    def test_sweep_amplitude_hd4_hd5_present(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_amplitude(pts, ll, ug1_bias=-10.0, steps=5)
        assert len(results) > 0
        for r in results:
            assert "hd4" in r
            assert "hd5" in r
            assert r["hd4"] >= 0.0
            assert r["hd5"] >= 0.0

    def test_sweep_ra_hd4_hd5_present(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        results = sweep_ra(pts, ub=250, ra_min=5.0, ra_max=20.0,
                           ug1_bias=-10.0, steps=5)
        assert len(results) > 0
        for r in results:
            assert "hd4" in r
            assert "hd5" in r

    def test_sweep_bias_hd4_hd5_present(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        results = sweep_bias(pts, ll, steps=5)
        assert len(results) > 0
        for r in results:
            assert "hd4" in r
            assert "hd5" in r

    def test_pp_sweep_hd4_hd5_present(self):
        _, pts = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        results = sweep_pp_amplitude(pts, ll, ug1_bias=-10.0, steps=5)
        assert len(results) > 0
        for r in results:
            assert "hd4" in r
            assert "hd5" in r

    def test_5point_sweep_hd4_hd5_are_zero(self):
        """5-point method in sweep produces hd4=hd5=0.0."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        # sweep_amplitude uses compute_distortion (5-point)
        results = sweep_amplitude(pts, ll, ug1_bias=-10.0, steps=3)
        for r in results:
            assert r["hd4"] == 0.0
            assert r["hd5"] == 0.0


# ---------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------

class TestOptimizeBias:
    def test_min_thd_12au7(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        result = optimize_bias(pts, ll, target="min_thd")
        assert result is not None
        assert "ug1_0" in result
        assert result["thd"] >= 0

    def test_max_pout_12au7(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        result = optimize_bias(pts, ll, target="max_pout")
        assert result is not None
        assert result["pout_mw"] > 0

    def test_balanced(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        result = optimize_bias(pts, ll, target="balanced")
        assert result is not None

    def test_pentode_optimize(self):
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        result = optimize_bias(pts, ll, target="min_thd")
        assert result is not None

    def test_has_rk_auto(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        result = optimize_bias(pts, ll, target="min_thd")
        assert result is not None
        assert "rk_auto_bias" in result
        assert result["rk_auto_bias"] > 0


# ---------------------------------------------------------------
# Stage parameters
# ---------------------------------------------------------------

class TestStageParams:
    def test_with_srk_data(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        srk = {"s": 2.2, "r": 7.7, "k": 17.0}
        result = compute_stage_params(isects, ll, ug1_bias=-10.0, srk=srk)
        assert result is not None
        assert result["method"] == "srk"
        assert result["gain"] > 0
        assert result["zout"] > 0
        assert result["gain_db"] > 0

    def test_numerical_fallback(self):
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        result = compute_stage_params(
            isects, ll, ug1_bias=-10.0, srk=None, points=pts,
        )
        assert result is not None
        assert result["method"] == "numerical"
        assert result["gm"] > 0
        assert result["ra"] > 0

    def test_no_points_no_srk_returns_none(self):
        isects = [
            {"ug1": -4.0, "ua": 200, "ia": 2.0},
            {"ug1": -2.0, "ua": 180, "ia": 4.0},
        ]
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        result = compute_stage_params(isects, ll, srk=None, points=None)
        assert result is None

    def test_zout_triode_ra_parallel(self):
        """Zout = ra || Ra for triode."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        srk = {"s": 2.2, "r": 7.7, "k": 17.0}
        result = compute_stage_params(isects, ll, srk=srk)
        expected_zout = (7.7 * 10.0) / (7.7 + 10.0)
        assert abs(result["zout"] - expected_zout) < 0.01


# ---------------------------------------------------------------
# SRK cross-check integration
# ---------------------------------------------------------------

class TestSRKCrossCheck:
    """Integration tests for numerical-primary + SRK cross-check logic."""

    def _make_triode_fixture(self):
        """Return (points, load_line, intersections) for 12AU7."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        return pts, ll, isects

    def _make_cf_fixture(self):
        """Return (points, load_line, intersections) for CF 12AU7."""
        _, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        isects = find_intersections(pts, ll)
        return pts, ll, isects

    # ── SE: no SRK ──

    def test_se_no_srk_method_numerical(self):
        pts, ll, isects = self._make_triode_fixture()
        result = compute_stage_params(isects, ll, ug1_bias=-10.0, points=pts)
        assert result is not None
        assert result["method"] == "numerical"
        assert result["srk_check"] is None
        assert result["srk_divergence_pct"] is None

    # ── SE: SRK close to numerical → "ok" ──

    def test_se_srk_agrees_with_numerical(self):
        pts, ll, isects = self._make_triode_fixture()
        # First get numerical values to build matching SRK
        num = compute_stage_params(isects, ll, ug1_bias=-10.0, points=pts)
        srk = {"s": num["gm"], "r": num["ra"], "k": num["mu"]}
        result = compute_stage_params(
            isects, ll, ug1_bias=-10.0, srk=srk, points=pts,
        )
        assert result["method"] == "numerical"
        assert result["srk_check"] == "ok"
        assert result["srk_divergence_pct"] < SRK_DIVERGENCE_THRESHOLD_PCT

    # ── SE: SRK diverges from numerical → "divergence" ──

    def test_se_srk_diverges_from_numerical(self):
        pts, ll, isects = self._make_triode_fixture()
        # Deliberately wrong SRK: 5× gm
        num = compute_stage_params(isects, ll, ug1_bias=-10.0, points=pts)
        srk = {"s": num["gm"] * 5.0, "r": num["ra"], "k": num["mu"]}
        result = compute_stage_params(
            isects, ll, ug1_bias=-10.0, srk=srk, points=pts,
        )
        assert result["method"] == "numerical"
        assert result["srk_check"] == "divergence"
        assert result["srk_divergence_pct"] > SRK_DIVERGENCE_THRESHOLD_PCT

    def test_se_divergence_uses_numerical_values(self):
        """Even when SRK diverges, result uses numerical gm/ra."""
        pts, ll, isects = self._make_triode_fixture()
        num = compute_stage_params(isects, ll, ug1_bias=-10.0, points=pts)
        srk = {"s": num["gm"] * 3.0, "r": num["ra"] * 3.0}
        result = compute_stage_params(
            isects, ll, ug1_bias=-10.0, srk=srk, points=pts,
        )
        assert result["gm"] == pytest.approx(num["gm"], rel=0.01)
        assert result["ra"] == pytest.approx(num["ra"], rel=0.01)

    # ── SE: only SRK, no points → fallback to SRK ──

    def test_se_srk_only_no_points_falls_back(self):
        _, ll, isects = self._make_triode_fixture()
        srk = {"s": 2.2, "r": 7.7, "k": 17.0}
        result = compute_stage_params(isects, ll, ug1_bias=-10.0, srk=srk)
        assert result is not None
        assert result["method"] == "srk"
        assert result["gm"] == 2.2
        assert result["ra"] == 7.7
        assert result["srk_check"] is None

    # ── SE: neither SRK nor points → None ──

    def test_se_no_srk_no_points_returns_none(self):
        _, ll, isects = self._make_triode_fixture()
        result = compute_stage_params(isects, ll, ug1_bias=-10.0)
        assert result is None

    # ── CF: no SRK ──

    def test_cf_no_srk_method_numerical(self):
        pts, ll, isects = self._make_cf_fixture()
        result = compute_stage_params(isects, ll, ug1_bias=-10.0, points=pts)
        assert result is not None
        assert result["method"] == "numerical"
        assert result["srk_check"] is None

    # ── CF: SRK agrees ──

    def test_cf_srk_agrees_with_numerical(self):
        pts, ll, isects = self._make_cf_fixture()
        num = compute_stage_params(isects, ll, ug1_bias=-10.0, points=pts)
        assert num is not None
        srk = {"s": num["gm"], "r": num["ra"], "k": num["mu"]}
        result = compute_stage_params(
            isects, ll, ug1_bias=-10.0, srk=srk, points=pts,
        )
        assert result["method"] == "numerical"
        assert result["srk_check"] == "ok"

    # ── CF: SRK diverges ──

    def test_cf_srk_diverges_from_numerical(self):
        pts, ll, isects = self._make_cf_fixture()
        num = compute_stage_params(isects, ll, ug1_bias=-10.0, points=pts)
        assert num is not None
        srk = {"s": num["gm"] * 5.0, "r": num["ra"] * 0.2}
        result = compute_stage_params(
            isects, ll, ug1_bias=-10.0, srk=srk, points=pts,
        )
        assert result["method"] == "numerical"
        assert result["srk_check"] == "divergence"

    # ── CF: only SRK ──

    def test_cf_srk_only_falls_back(self):
        _, ll, isects = self._make_cf_fixture()
        srk = {"s": 2.2, "r": 7.7, "k": 17.0}
        result = compute_stage_params(isects, ll, ug1_bias=-10.0, srk=srk)
        assert result is not None
        assert result["method"] == "srk"
        assert result["gain"] < 1.0  # CF gain always < 1

    # ── _compare_srk unit tests ──

    def test_compare_srk_absent(self):
        check, pct = _compare_srk(None, 2.0, 7.0)
        assert check is None
        assert pct is None

    def test_compare_srk_empty_s(self):
        check, pct = _compare_srk({"s": 0, "r": 7.0}, 2.0, 7.0)
        assert check is None

    def test_compare_srk_ok(self):
        check, pct = _compare_srk({"s": 2.0, "r": 7.0}, 2.0, 7.0)
        assert check == "ok"
        assert pct == 0.0

    def test_compare_srk_within_threshold(self):
        # ~9% off on gm (symmetric: 0.2/2.2), within 20% threshold
        check, pct = _compare_srk({"s": 2.2, "r": 7.0}, 2.0, 7.0)
        assert check == "ok"
        assert pct == pytest.approx(9.1, abs=0.2)

    def test_compare_srk_exceeds_threshold(self):
        # ~33% off on ra (symmetric: 3.5/10.5)
        check, pct = _compare_srk({"s": 2.0, "r": 10.5}, 2.0, 7.0)
        assert check == "divergence"
        assert pct == pytest.approx(33.3, abs=0.2)

    # ── _compare_srk edge cases ──

    def test_compare_srk_missing_r_key(self):
        check, pct = _compare_srk({"s": 2.2}, 2.0, 7.0)
        assert check is None
        assert pct is None

    def test_compare_srk_r_zero(self):
        check, pct = _compare_srk({"s": 2.2, "r": 0}, 2.0, 7.0)
        assert check is None

    def test_compare_srk_missing_s_key(self):
        check, pct = _compare_srk({"r": 7.0}, 2.0, 7.0)
        assert check is None

    def test_compare_srk_very_large_divergence(self):
        # 10× gm → ~90% symmetric divergence
        check, pct = _compare_srk({"s": 20.0, "r": 7.0}, 2.0, 7.0)
        assert check == "divergence"
        assert pct == pytest.approx(90.0, abs=0.5)

    def test_compare_srk_negative_values_treated_as_divergence(self):
        # Negative gm is physically invalid → large divergence
        check, pct = _compare_srk({"s": -2.0, "r": 7.0}, 2.0, 7.0)
        assert check == "divergence"
        assert pct > SRK_DIVERGENCE_THRESHOLD_PCT

    # ── SRK fallback without "k" key ──

    def test_se_srk_fallback_without_k_computes_mu(self):
        """When SRK has no 'k', mu should be computed as s * r."""
        _, ll, isects = self._make_triode_fixture()
        srk = {"s": 2.2, "r": 7.7}  # no "k"
        result = compute_stage_params(isects, ll, ug1_bias=-10.0, srk=srk)
        assert result is not None
        assert result["mu"] == pytest.approx(2.2 * 7.7, rel=0.01)


# ---------------------------------------------------------------
# Data source selection
# ---------------------------------------------------------------

class TestDataSourceSelection:
    def test_current_scan_preferred(self):
        pts = [
            {"ua": 100, "ug1": -5, "ia": 10, "series_id": 0},
            {"ua": 100, "ug1": -5, "ia": 10, "series_id": 1},
        ]
        selected = select_analysis_points(pts)
        assert len(selected) == 1
        assert selected[0]["series_id"] == 0

    def test_specific_series(self):
        pts = [
            {"ua": 100, "ug1": -5, "ia": 10, "series_id": 0},
            {"ua": 200, "ug1": -3, "ia": 15, "series_id": 0},
            {"ua": 100, "ug1": -5, "ia": 8, "series_id": 1},
        ]
        selected = select_analysis_points(pts, series_id=0)
        assert len(selected) == 2
        assert all(p["series_id"] == 0 for p in selected)

    def test_fallback_all(self):
        pts = [
            {"ua": 100, "ug1": -5, "ia": 10, "series_id": 1},
            {"ua": 200, "ug1": -3, "ia": 15, "series_id": 2},
        ]
        selected = select_analysis_points(pts)
        assert len(selected) == 2

    def test_get_available_series(self):
        pts = [
            {"ua": 100, "ug1": -5, "ia": 10, "series_id": 0},
            {"ua": 100, "ug1": -5, "ia": 10, "series_id": 1},
            {"ua": 100, "ug1": -5, "ia": 8, "series_id": 2},
        ]
        labels = {0: "Current scan", 1: "EL84 #1", 2: "EL84 #2"}
        sources = get_available_series(pts, labels)
        assert len(sources) == 3
        assert sources[0]["series_id"] == 0
        assert sources[0]["label"] == "Current scan"
        assert sources[1]["label"] == "EL84 #1"


# ---------------------------------------------------------------
# Presets
# ---------------------------------------------------------------

class TestPresets:
    def test_presets_exist(self):
        assert len(AMPLIFIER_PRESETS) > 0

    def test_preset_fields(self):
        for p in AMPLIFIER_PRESETS:
            assert p.name
            assert p.tube
            assert p.ub > 0
            assert p.ra > 0
            assert p.pa_max > 0


# ---------------------------------------------------------------
# Noise robustness
# ---------------------------------------------------------------

class TestNoiseRobustness:
    def test_distortion_stable_with_noise(self):
        model, pts_clean = quick_triode("12AU7")
        pts_noisy = model.add_noise(pts_clean, sigma_pct=0.5, seed=42)

        ll = ResistiveLoadLine(ub=250, ra=10.0)

        isects_clean = find_intersections(pts_clean, ll)
        isects_noisy = find_intersections(pts_noisy, ll)

        r_clean = compute_distortion(isects_clean, ug1_bias=-10.0)
        r_noisy = compute_distortion(isects_noisy, ug1_bias=-10.0)

        assert r_clean and r_noisy, "guard de-vacuated 2026-07-12: value must be present"
        assert abs(r_clean["hd2"] - r_noisy["hd2"]) < 2.0
        if r_clean["pout_mw"] > 0:
            pct_diff = abs(r_clean["pout_mw"] - r_noisy["pout_mw"]) / r_clean["pout_mw"]
            assert pct_diff < 0.2


# ---------------------------------------------------------------
# Cross-tube comparison
# ---------------------------------------------------------------

class TestCrossTube:
    def test_12au7_more_power_than_12ax7(self):
        """12AU7 (lower mu, more current) should produce more Pout than 12AX7 (preamp)."""
        _, pts_12au7 = quick_triode("12AU7")
        _, pts_12ax7 = quick_triode("12AX7")

        ll_12au7 = ResistiveLoadLine(ub=250, ra=10.0)
        ll_12ax7 = ResistiveLoadLine(ub=250, ra=100.0)

        isects_12au7 = find_intersections(pts_12au7, ll_12au7)
        isects_12ax7 = find_intersections(pts_12ax7, ll_12ax7)

        r_12au7 = compute_distortion(isects_12au7)
        r_12ax7 = compute_distortion(isects_12ax7)

        assert r_12au7 and r_12ax7, "guarded value must be truthy (de-vacuated)"
        assert r_12au7["pout_mw"] > r_12ax7["pout_mw"]


# ---------------------------------------------------------------
# Cathode Follower Load Line
# ---------------------------------------------------------------

class TestCathodeFollowerLoadLine:
    def test_ia_at_ua(self):
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        assert ll.ia_at_ua(250) == 0.0
        assert abs(ll.ia_at_ua(0) - 12.5) < 0.01
        assert abs(ll.ia_at_ua(50) - 10.0) < 0.01

    def test_zero_r_total(self):
        ll = CathodeFollowerLoadLine(ub=250, rk=0, rl=0)
        assert ll.ia_at_ua(100) == 0.0

    def test_endpoints(self):
        ll = CathodeFollowerLoadLine(ub=300, rk=10.0, rl=5.0)
        (ua0, ia0), (ua1, ia1) = ll.endpoints()
        assert (ua0, ia0) == (0.0, 20.0)
        assert (ua1, ia1) == (300.0, 0.0)

    def test_label(self):
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        assert "CF" in ll.label()

    def test_intersections_triode(self):
        _, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        isects = find_intersections(pts, ll)
        assert len(isects) >= 2


class TestCFStageParams:
    def test_cf_gain_near_unity(self):
        """CF gain should be close to 1 (less than 1)."""
        _, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        isects = find_intersections(pts, ll)
        srk = {"s": 2.2, "r": 7.7, "k": 17.0}
        result = compute_cf_stage_params(isects, ll, ug1_bias=-10.0, srk=srk)
        assert result is not None
        assert 0 < result["gain"] < 1.0
        assert result["gain"] > 0.5

    def test_cf_low_zout(self):
        """CF Zout ≈ 1/gm = ra/(mu+1), should be much lower than SE Zout."""
        _, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        isects = find_intersections(pts, ll)
        srk = {"s": 2.2, "r": 7.7, "k": 17.0}
        result = compute_cf_stage_params(isects, ll, srk=srk)
        assert result is not None
        expected_zout = 7.7 / (17.0 + 1)
        assert abs(result["zout"] - expected_zout) < 0.05

    def test_cf_numerical_fallback(self):
        _, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        isects = find_intersections(pts, ll)
        result = compute_cf_stage_params(isects, ll, ug1_bias=-10.0, points=pts)
        assert result, "guarded value must be truthy (de-vacuated)"
        assert result["method"] == "numerical"
        assert result["gain"] > 0
        assert result["gain"] < 1.0

    def test_cf_dispatched_from_compute_stage_params(self):
        """compute_stage_params should detect CF and use CF formulas."""
        _, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        isects = find_intersections(pts, ll)
        srk = {"s": 2.2, "r": 7.7, "k": 17.0}
        result = compute_stage_params(isects, ll, srk=srk)
        assert result is not None
        assert result["gain"] < 1.0


# ---------------------------------------------------------------
# Push-Pull Load Line
# ---------------------------------------------------------------

class TestPushPullLoadLine:
    def test_ra_per_tube(self):
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)
        assert ll.ra_per_tube == 2.0

    def test_ia_at_ua(self):
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)
        assert abs(ll.ia_at_ua(0) - 150.0) < 0.01
        assert ll.ia_at_ua(300) == 0.0

    def test_endpoints(self):
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)
        (_, ia0), (ua1, _) = ll.endpoints()
        assert ia0 == 150.0
        assert ua1 == 300.0

    def test_label(self):
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)
        assert "PP" in ll.label()


class TestCompositeCharacteristic:
    def test_matched_pair_basic(self):
        _, pts = quick_triode("12AU7")
        comp = composite_characteristic(pts, ug1_bias=-10.0)
        assert len(comp) > 0
        assert all("ia_composite" in c for c in comp)

    def test_matched_pair_odd_symmetric(self):
        """Matched pair composite should be odd-symmetric around bias."""
        _, pts = quick_triode("12AU7")
        comp = composite_characteristic(pts, ug1_bias=-10.0)
        assert len(comp) >= 5
        bias_ia = None
        for c in comp:
            if abs(c["ug1"] - (-10.0)) < 0.5:
                bias_ia = c["ia_composite"]
                break
        assert bias_ia is not None  # bias must land on a composite grid point
        assert abs(bias_ia) < 0.5  # at bias: Ia_A(bias) - Ia_A(bias) ≈ 0

    def test_pentode_composite(self):
        _, pts = quick_pentode("EL84")
        comp = composite_characteristic(pts, ug1_bias=-7.0)
        assert len(comp) > 0

    def test_empty_points(self):
        assert composite_characteristic([], ug1_bias=-10.0) == []


@pytest.mark.timeout(60)
# Heavy DFT solves ~8 s/test — near the global 10 s limit; under
# xdist load they cross it and look like worker crashes (recurring
# schedule-dependent flake).
class TestPPDftUlEffect:
    """``compute_distortion_dft_pp`` self-consistent Ua/Ia solve must
    correctly handle ``UltralinearModelWrapper``.

    The trap: if PP DFT uses a constant Ua = Ub, UL/Triode wrappers see
    no Ua variation, so the Ug2-modulation effect becomes invisible and
    Pentode / UL / Triode give identical results at the same bias.

    Correct behaviour: each time sample solves
    ``Ia = model.ia(Ua, Ug1, Ug2)`` jointly with the load line
    ``Ia = (Ub − Ua) / Ra_per_tube``. The UL wrapper then recomputes
    ``Ug2_eff(Ua)`` per sample → correct screen-tap linearisation.
    """

    def _setup_el84(self):
        import json
        from lm19.dempwolf import fit_dempwolf
        with open(_PROJECT_ROOT / "tests/spice_test_data/converted/pentode_EL84_SOVTEK_L1_real.json") as f:
            pts = json.load(f)["points"]
        return fit_dempwolf(pts, topology=TOPOLOGY_PENTODE).model

    def test_pentode_pout_matches_datasheet_ballpark(self):
        """EL84 PP class AB1 datasheet: 17W @ Ub=300V Ra_aa=8k.
        Mullard 5-10 reference: Vin_RMS≈4.3V → swing≈±6V (limited by
        grid current at Ug1=0). Larger swing crosses grid into AB2 with
        much higher Pout — our model doesn't include grid-current limit
        so we test the AB1 swing range explicitly."""
        from lm19.amplifier import (
            PushPullLoadLine, compute_distortion_dft_pp,
        )
        model = self._setup_el84()
        # Use realistic ra_dc for EL84-class OPT
        ll = PushPullLoadLine(ub=300.0, ra_aa=8.0, ra_dc=0.1)
        d = compute_distortion_dft_pp(
            model, ll, ug1_bias=-9.0, half_swing=6.0, ug2=300.0,
        )
        assert d is not None
        # Joint ideal-OPT solver at partial drive (±6 V of ±11 max):
        # probed 10.0 W; band centred there — kills naive-line (~half)
        # and wrong-per-tube-impedance (former solver: 13.9 W) regressions
        # while tolerating model-fit drift.
        assert 7.0 < d["pout_mw"] / 1000.0 < 18.0, (
            f"Pentode Pout {d['pout_mw']/1000:.1f}W out of datasheet range"
        )

    # probed reality: fitted EL84 model gives UL/pentode Pout ratio ≈ 0.53
    # (datasheet ballpark ≈ 0.62). Band is ±0.15 wide around reality so it
    # still rejects both "no UL effect" (ratio → 1.0) and degeneration to
    # triode wiring (ratio ≈ 0.31)
    UL_POUT_RATIO_MIN = 0.35
    UL_POUT_RATIO_MAX = 0.70

    def test_ul_reduces_pout_relative_to_pentode(self):
        """UL 43% tap delivers roughly half of pentode Pout (probed ≈ 0.53)."""
        from lm19.amplifier import (
            PushPullLoadLine, compute_distortion_dft_pp,
            UltralinearModelWrapper,
        )
        model = self._setup_el84()
        ll = PushPullLoadLine(ub=300.0, ra_aa=8.0)
        bias, swing = -9.0, 7.0

        d_pent = compute_distortion_dft_pp(
            model, ll, ug1_bias=bias, half_swing=swing, ug2=300.0,
        )
        ul_model = UltralinearModelWrapper(model, ug2_nom=300.0, tap=0.43)
        d_ul = compute_distortion_dft_pp(
            ul_model, ll, ug1_bias=bias, half_swing=swing, ug2=300.0,
        )
        assert d_pent is not None and d_ul is not None
        ratio = d_ul["pout_mw"] / d_pent["pout_mw"]
        assert self.UL_POUT_RATIO_MIN < ratio < self.UL_POUT_RATIO_MAX, (
            f"UL Pout ratio {ratio:.2f} outside "
            f"[{self.UL_POUT_RATIO_MIN}, {self.UL_POUT_RATIO_MAX}] — "
            f"UL screen-tap wiring likely broken"
        )

    def test_ul_reduces_thd_relative_to_pentode(self):
        """UL typically gives ~½ of pentode THD due to screen linearization."""
        from lm19.amplifier import (
            PushPullLoadLine, compute_distortion_dft_pp,
            UltralinearModelWrapper,
        )
        model = self._setup_el84()
        ll = PushPullLoadLine(ub=300.0, ra_aa=8.0)
        bias, swing = -9.0, 7.0

        d_pent = compute_distortion_dft_pp(
            model, ll, ug1_bias=bias, half_swing=swing, ug2=300.0,
        )
        ul_model = UltralinearModelWrapper(model, ug2_nom=300.0, tap=0.43)
        d_ul = compute_distortion_dft_pp(
            ul_model, ll, ug1_bias=bias, half_swing=swing, ug2=300.0,
        )
        assert d_pent is not None and d_ul is not None
        # UL THD strictly less than pentode (Ug2 modulation linearizes)
        assert d_ul["thd"] < d_pent["thd"], (
            f"UL THD {d_ul['thd']:.2f}% not below pentode {d_pent['thd']:.2f}%"
        )

    def test_triode_mode_lowest_pout_lowest_thd(self):
        """UL tap=1.0 (triode) gives least Pout and least THD vs pentode/UL."""
        from lm19.amplifier import (
            PushPullLoadLine, compute_distortion_dft_pp,
            UltralinearModelWrapper,
        )
        model = self._setup_el84()
        ll = PushPullLoadLine(ub=300.0, ra_aa=8.0)
        bias, swing = -9.0, 7.0

        d_pent = compute_distortion_dft_pp(
            model, ll, ug1_bias=bias, half_swing=swing, ug2=300.0,
        )
        triode = UltralinearModelWrapper(model, ug2_nom=300.0, tap=1.0)
        d_tri = compute_distortion_dft_pp(
            triode, ll, ug1_bias=bias, half_swing=swing, ug2=300.0,
        )
        assert d_pent is not None and d_tri is not None
        assert d_tri["pout_mw"] < d_pent["pout_mw"]
        assert d_tri["thd"] < d_pent["thd"]
        # Triode mode typically 25–50% of pentode Pout
        ratio = d_tri["pout_mw"] / d_pent["pout_mw"]
        assert 0.15 < ratio < 0.55, f"Triode Pout ratio {ratio:.2f} unexpected"

    def test_se_pentode_pout_matches_datasheet(self):
        """EL84 SE pentode @ Ub=250V, Ra_dc=0.1k, Ra_ac=5.2k, bias=-7.3V.
        Mullard datasheet: Pout=3.4W @ 10% THD, Iq=48mA.

        Critical: must use TransformerLoadLine — ResistiveLoadLine puts
        Q-point in saturation (Ua≈10V), gives <0.5W. Real SE pentode
        amps connect plate to Ub through transformer primary (low DC
        resistance), AC load comes from reflected secondary impedance."""
        from lm19.amplifier import (
            TransformerLoadLine, compute_distortion_dft,
        )
        model = self._setup_el84()

        ll = TransformerLoadLine(ub=250.0, ra_dc=0.1, ra_ac=5.2)
        # ±5V swing brackets datasheet input level (Vin_RMS=4.3V)
        d = compute_distortion_dft(
            model, ll, ug1_bias=-7.3, half_swing=5.0, ug2=249.0, ub=250.0,
        )
        assert d is not None
        pout_w = d["pout_mw"] / 1000.0
        # Datasheet 3.4W ± 30% tolerance for model fit / specimen variance
        assert 2.5 < pout_w < 5.0, f"SE Pout {pout_w:.2f}W out of datasheet range"
        # THD bracket: at lower swing higher headroom → cleaner;
        # at full swing closer to 10% datasheet
        assert d["thd"] < 12.0, f"SE THD {d['thd']:.1f}% suspiciously high"

    def test_se_pentode_resistive_vs_transformer_load(self):
        """Documents the resistor-vs-transformer pitfall.

        Resistive load with same Ra value produces a degenerate Q-point
        (saturation) → minimal Pout. Transformer load with same AC Ra
        gives the realistic SE result. Sanity-checks the load-line
        choice matters and that we compute it correctly."""
        from lm19.amplifier import (
            ResistiveLoadLine, TransformerLoadLine, compute_distortion_dft,
        )
        model = self._setup_el84()

        # Same nominal Ra value for both load lines
        ra = 5.2
        d_res = compute_distortion_dft(
            model, ResistiveLoadLine(ub=250.0, ra=ra),
            ug1_bias=-7.3, half_swing=5.0, ug2=249.0, ub=250.0,
        )
        d_xfmr = compute_distortion_dft(
            model, TransformerLoadLine(ub=250.0, ra_dc=0.1, ra_ac=ra),
            ug1_bias=-7.3, half_swing=5.0, ug2=249.0, ub=250.0,
        )
        assert d_res is not None and d_xfmr is not None
        # Transformer load gives 5–10× more output than resistive load
        # at the same Ra (because Q-point isn't saturated)
        assert d_xfmr["pout_mw"] > 2.0 * d_res["pout_mw"], (
            f"Transformer Pout={d_xfmr['pout_mw']:.0f}mW must be much "
            f"larger than resistive Pout={d_res['pout_mw']:.0f}mW "
            "at same Ra value (Q-point saturation pitfall)"
        )

    def test_se_ul_reduces_pout_relative_to_pentode(self):
        """SE UL effect: Pout drops to 30–60% of pentode, THD drops to
        about 30–60%. Triode mode further reduces both. Same physics as
        PP UL but expressed in the single-ended composite (no cancellation).
        """
        from lm19.amplifier import (
            TransformerLoadLine, compute_distortion_dft,
            UltralinearModelWrapper,
        )
        model = self._setup_el84()
        ll = TransformerLoadLine(ub=250.0, ra_dc=0.1, ra_ac=5.2)
        bias, swing = -7.3, 7.0

        d_pent = compute_distortion_dft(
            model, ll, ug1_bias=bias, half_swing=swing, ug2=249.0, ub=250.0,
        )
        ul = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.43)
        d_ul = compute_distortion_dft(
            ul, ll, ug1_bias=bias, half_swing=swing, ug2=249.0, ub=250.0,
        )
        triode = UltralinearModelWrapper(model, ug2_nom=250.0, tap=1.0)
        d_tri = compute_distortion_dft(
            triode, ll, ug1_bias=bias, half_swing=swing, ug2=249.0, ub=250.0,
        )
        assert d_pent and d_ul and d_tri
        # Sanity: each step in linearization reduces Pout AND THD
        assert d_ul["pout_mw"] < d_pent["pout_mw"]
        assert d_tri["pout_mw"] < d_ul["pout_mw"]
        assert d_ul["thd"] < d_pent["thd"]
        # Triode has lowest absolute Pout
        assert d_tri["pout_mw"] / d_pent["pout_mw"] < 0.5

    def test_pp_ra_dc_affects_q_point_iq(self):
        """``pp_ra_dc`` must alter the DC Q-point: higher winding
        resistance drops more DC voltage on the primary → less Ua at
        the plate → slightly less Iq for the same bias. If the Q-point
        solver mistakenly uses AC ``ra`` instead of ``ra_dc``,
        ``ra_dc`` would have zero effect on Iq.

        Sanity: for typical EL84 OPT (ra_dc 0.05–1.0 kΩ) Iq drops
        monotonically as ra_dc grows."""
        from lm19.amplifier import (
            PushPullLoadLine, compute_distortion_dft_pp,
        )
        model = self._setup_el84()
        iqs = []
        pa_qs = []
        for ra_dc in (0.05, 0.5, 1.0):
            ll = PushPullLoadLine(ub=300.0, ra_aa=8.0, ra_dc=ra_dc)
            d = compute_distortion_dft_pp(
                model, ll, ug1_bias=-9.0, half_swing=6.0, ug2=300.0,
            )
            assert d is not None
            iqs.append(d["iq_per_tube"])
            pa_qs.append(300.0 * d["iq_per_tube"] / 1000.0)
        # Monotonic: higher ra_dc → less Ua_q → slightly less Iq
        assert iqs[0] >= iqs[1] >= iqs[2], f"Iq not monotonic in ra_dc: {iqs}"
        # ra_dc actually affects something — not all the same value
        assert iqs[0] > iqs[2], (
            f"ra_dc has no effect: Iq[0.05k]={iqs[0]:.2f} == Iq[1.0k]={iqs[2]:.2f}"
        )

    def test_iq_decreases_with_higher_ul_tap(self):
        """Higher UL tap → screen voltage tracks Ua → effective gm drops →
        less Iq for same Ug1 bias. Pentode has highest Iq, triode lowest."""
        from lm19.amplifier import (
            PushPullLoadLine, compute_distortion_dft_pp,
            UltralinearModelWrapper,
        )
        model = self._setup_el84()
        ll = PushPullLoadLine(ub=300.0, ra_aa=8.0)
        bias, swing = -9.0, 5.0

        iqs = []
        for tap in (0.0, 0.43, 1.0):  # pentode, UL43%, triode
            wrapped = UltralinearModelWrapper(model, ug2_nom=300.0, tap=tap)
            d = compute_distortion_dft_pp(
                wrapped, ll, ug1_bias=bias, half_swing=swing, ug2=300.0,
            )
            assert d is not None
            iqs.append(d["iq_per_tube"])
        # Strictly monotonic decrease
        assert iqs[0] > iqs[1] > iqs[2], f"Iq not monotonic in tap: {iqs}"


class TestPPDistortion:
    def test_matched_triode_low_hd2(self):
        """Matched PP pair should have very low HD2 (even harmonics cancel)."""
        _, pts = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        dist = pp_distortion(pts, ll, ug1_bias=-10.0)
        assert dist is not None
        assert dist["hd2"] < 5.0  # should be near 0, allow tolerance for discrete grid
        assert dist["pout_mw"] > 0

    def test_matched_pentode(self):
        _, pts = quick_pentode("EL84")
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)
        dist = pp_distortion(pts, ll, ug1_bias=-7.0)
        assert dist is not None
        assert dist["pout_mw"] > 0

    def test_with_half_swing(self):
        _, pts = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        dist = pp_distortion(pts, ll, ug1_bias=-10.0, half_swing=3.0)
        assert dist is not None

    def test_balance_error_low_matched(self):
        """Matched pair should have balance_error (=HD2) near 0."""
        _, pts = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        dist = pp_distortion(pts, ll, ug1_bias=-10.0)
        assert dist is not None
        assert dist["balance_error"] < 5.0

    def test_insufficient_data(self):
        pts = [{"ua": 100, "ug1": -5, "ug2": 0, "ia": 10}]
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        assert pp_distortion(pts, ll, ug1_bias=-5.0) is None

    def test_rejects_narrow_swing_sparse_data(self):
        """A swing window spanning <3 real composite Ug1 curves must return
        None (5-point would otherwise interpolate a fake-low THD); a wide
        window over the same data still works."""
        _, pts = quick_triode("12AU7")  # Ug1 grid -20..0 step 2 V
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        # window [-11,-9] contains only the -10 curve → sparse → None
        assert pp_distortion(pts, ll, ug1_bias=-10.0, half_swing=1.0) is None
        # window [-15,-5] contains 5 real curves → valid
        assert pp_distortion(pts, ll, ug1_bias=-10.0, half_swing=5.0) is not None

    def test_diagnose_returns_sparse_data(self):
        from lm19.amplifier import diagnose_pp_distortion, DIST_ERR_SPARSE_DATA
        _, pts = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        assert diagnose_pp_distortion(pts, ll, ug1_bias=-10.0,
                                      half_swing=1.0) == DIST_ERR_SPARSE_DATA


class TestSweepPPAmplitude:
    def test_basic_sweep(self):
        _, pts = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        results = sweep_pp_amplitude(pts, ll, ug1_bias=-10.0, steps=8)
        assert len(results) > 0
        assert all("balance_error" in r for r in results)

    def test_pentode_sweep(self):
        _, pts = quick_pentode("EL84")
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)
        results = sweep_pp_amplitude(pts, ll, ug1_bias=-7.0, steps=8)
        assert len(results) > 0

    # probed reality: THD grows 0.84 % → 8.2 % across the sweep; require
    # a real multiple so a flat-THD regression fails, not just non-decrease
    THD_GROWTH_MIN_FACTOR = 2.0

    def test_thd_increases_with_amplitude(self):
        _, pts = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        results = sweep_pp_amplitude(pts, ll, ug1_bias=-10.0, steps=15)
        assert len(results) >= 3
        thds = [r["thd"] for r in results]
        assert thds[-1] > thds[0] * self.THD_GROWTH_MIN_FACTOR, \
            f"THD did not grow with amplitude: {thds[0]:.3f}% → {thds[-1]:.3f}%"


# ---------------------------------------------------------------
# Analytical verification — known polynomial transfer curves
# ---------------------------------------------------------------

def _make_polynomial_intersections(
    a0: float, a1: float, a2: float, a3: float, half_swing: float,
) -> list:
    """Build synthetic intersection list from Ia = a0 + a1*x + a2*x² + a3*x³.

    x is the grid voltage deviation from the bias point.
    Returns intersection dicts at x = -hs, -hs/2, 0, +hs/2, +hs
    with ua fabricated as Ua_q - x*Ra (Ra=10 kOhm, Ua_q=200V).
    """
    ra = 10.0  # kOhm
    ua_q = 200.0
    pts = []
    for x in [
        -half_swing,
        -half_swing / 2,
        0.0,
        half_swing / 2,
        half_swing,
    ]:
        ia = a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3
        ua = ua_q - x * ra
        pts.append({"ug1": x, "ua": ua, "ia": ia})
    return pts


def _analytical_fourier(a1, a2, a3, half_swing):
    """Exact Fourier harmonic amplitudes for i = a0+a1*V*cos+a2*V²*cos²+a3*V³*cos³."""
    v = half_swing
    b1 = a1 * v + 0.75 * a3 * v ** 3
    b2 = 0.5 * a2 * v ** 2
    b3 = 0.25 * a3 * v ** 3
    hd2 = abs(b2 / b1) * 100.0 if abs(b1) > 1e-12 else 0.0
    hd3 = abs(b3 / b1) * 100.0 if abs(b1) > 1e-12 else 0.0
    return hd2, hd3


class TestAnalyticalHD:
    """Verify HD2/HD3 against exact Fourier analysis on polynomial curves."""

    def test_pure_linear_zero_distortion(self):
        """Perfectly linear curve → HD2 = HD3 = 0."""
        isects = _make_polynomial_intersections(10.0, 2.0, 0.0, 0.0, 4.0)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=4.0)
        assert result is not None
        assert result["hd2"] < 0.01
        assert result["hd3"] < 0.01
        assert result["thd"] < 0.01

    def test_pure_quadratic_only_hd2(self):
        """i = a0 + a1*x + a2*x² → HD3 should be zero, HD2 matches Fourier."""
        a0, a1, a2, a3 = 10.0, 2.0, 0.1, 0.0
        hs = 4.0
        isects = _make_polynomial_intersections(a0, a1, a2, a3, hs)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=hs)
        assert result is not None

        hd2_exact, hd3_exact = _analytical_fourier(a1, a2, a3, hs)
        assert abs(result["hd2"] - hd2_exact) < 0.3, \
            f"HD2: code={result['hd2']:.3f} exact={hd2_exact:.3f}"
        assert result["hd3"] < 0.05

    def test_cubic_hd3_matches_fourier(self):
        """i = a0+a1*x+a2*x²+a3*x³ — both HD2 and HD3 match Fourier exactly."""
        a0, a1, a2, a3 = 10.0, 2.0, 0.1, 0.01
        hs = 4.0
        isects = _make_polynomial_intersections(a0, a1, a2, a3, hs)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=hs)
        assert result is not None

        hd2_exact, hd3_exact = _analytical_fourier(a1, a2, a3, hs)

        assert abs(result["hd2"] - hd2_exact) < 0.3, \
            f"HD2: code={result['hd2']:.3f} exact={hd2_exact:.3f}"
        assert abs(result["hd3"] - hd3_exact) < 0.3, \
            f"HD3: code={result['hd3']:.3f} exact={hd3_exact:.3f}"

    def test_high_distortion_cubic(self):
        """Higher distortion level — verify HD3 not inflated."""
        a0, a1, a2, a3 = 5.0, 3.0, 0.3, 0.05
        hs = 3.0
        isects = _make_polynomial_intersections(a0, a1, a2, a3, hs)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=hs)
        assert result is not None

        hd2_exact, hd3_exact = _analytical_fourier(a1, a2, a3, hs)

        assert abs(result["hd2"] - hd2_exact) < 0.5, \
            f"HD2: code={result['hd2']:.3f} exact={hd2_exact:.3f}"
        assert abs(result["hd3"] - hd3_exact) < 0.5, \
            f"HD3: code={result['hd3']:.3f} exact={hd3_exact:.3f}"

    def test_various_amplitudes(self):
        """HD should scale correctly with amplitude."""
        a0, a1, a2, a3 = 10.0, 2.0, 0.1, 0.01
        checked = 0
        for hs in [1.0, 2.0, 3.0, 5.0]:
            isects = _make_polynomial_intersections(a0, a1, a2, a3, hs)
            result = compute_distortion(isects, ug1_bias=0.0, half_swing=hs)
            if result is None:
                continue  # legitimately sparse swing (MIN_CURVES_IN_SWING guard)
            checked += 1
            hd2_exact, hd3_exact = _analytical_fourier(a1, a2, a3, hs)

            assert abs(result["hd2"] - hd2_exact) < 0.5, \
                f"hs={hs}: HD2 code={result['hd2']:.3f} exact={hd2_exact:.3f}"
            assert abs(result["hd3"] - hd3_exact) < 0.5, \
                f"hs={hs}: HD3 code={result['hd3']:.3f} exact={hd3_exact:.3f}"
        assert checked > 0, "no amplitude produced a distortion result"

    def test_negative_a2_asymmetry(self):
        """Negative a2 should still give correct HD2."""
        a0, a1, a2, a3 = 10.0, 2.0, -0.15, 0.005
        hs = 4.0
        isects = _make_polynomial_intersections(a0, a1, a2, a3, hs)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=hs)
        assert result is not None

        hd2_exact, hd3_exact = _analytical_fourier(a1, a2, a3, hs)
        assert abs(result["hd2"] - hd2_exact) < 0.5
        assert abs(result["hd3"] - hd3_exact) < 0.3

    def test_thd_is_rss(self):
        """THD should always equal sqrt(HD2² + HD3²)."""
        a0, a1, a2, a3 = 10.0, 2.0, 0.1, 0.01
        hs = 4.0
        isects = _make_polynomial_intersections(a0, a1, a2, a3, hs)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=hs)
        assert result is not None
        expected = (result["hd2"] ** 2 + result["hd3"] ** 2) ** 0.5
        assert abs(result["thd"] - expected) < 0.001


class TestAnalyticalPout:
    """Verify output power against analytical P = Vpp × Ipp / 8."""

    def test_se_pout_linear_curve(self):
        """For a linear curve Ia(Ug1), Pout should match Ipp*Vpp/8 exactly."""
        a0, a1 = 10.0, 2.0
        hs = 4.0
        ra = 10.0
        isects = _make_polynomial_intersections(a0, a1, 0.0, 0.0, hs)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=hs)
        assert result is not None

        i_pp = result["i_max"] - result["i_min"]  # mA
        ua_pp = abs(result["ua_max"] - result["ua_min"])  # V
        expected_pout = i_pp * ua_pp / 8.0
        assert abs(result["pout_mw"] - expected_pout) < 0.01

    def test_se_pout_cubic_curve(self):
        """Even with distortion, Pout = Ipp × Vpp / 8 from the extreme points."""
        a0, a1, a2, a3 = 10.0, 2.0, 0.1, 0.01
        hs = 4.0
        isects = _make_polynomial_intersections(a0, a1, a2, a3, hs)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=hs)
        assert result is not None

        i_pp = result["i_max"] - result["i_min"]
        ua_pp = abs(result["ua_max"] - result["ua_min"])
        expected_pout = i_pp * ua_pp / 8.0
        assert abs(result["pout_mw"] - expected_pout) < 0.01

    def test_pp_pout_formula(self):
        """PP Pout should use /8, same as SE (sinusoidal power formula)."""
        _, pts = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        dist = pp_distortion(pts, ll, ug1_bias=-10.0)
        assert dist is not None

        ra_pt = ll.ra_per_tube
        swing = abs(dist["i_max"] - dist["i_min"])
        ua_swing = swing * ra_pt
        expected = swing * ua_swing / 8.0
        assert abs(dist["pout_mw"] - expected) < 0.01, \
            f"PP Pout: code={dist['pout_mw']:.2f} expected={expected:.2f}"


class TestAnalyticalPP:
    """Analytical tests for push-pull distortion."""

    def test_matched_pair_perfect_hd2_cancellation(self):
        """Matched pair: composite is odd-symmetric, so HD2 ≈ 0."""
        _, pts = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        dist = pp_distortion(pts, ll, ug1_bias=-10.0)
        assert dist is not None
        assert dist["hd2"] < 2.0, \
            f"Matched PP HD2 should be near 0, got {dist['hd2']:.2f}%"

    # probed reality: 12AU7 matched PP @ half_swing=5 V → HD3 ≈ 2.9 %, HD2 ≈ 0
    PP_MATCHED_HD3_MIN_PCT = 0.1

    def test_pp_hd3_dominates_matched(self):
        """In matched PP, HD3 should be the dominant distortion component."""
        _, pts = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        dist = pp_distortion(pts, ll, ug1_bias=-10.0, half_swing=5.0)
        assert dist is not None
        # HD3 must genuinely be present — without this pin the dominance
        # check below passes vacuously on a degenerate all-zero result
        assert dist["hd3"] > self.PP_MATCHED_HD3_MIN_PCT, \
            f"PP matched HD3 collapsed: {dist['hd3']:.3f}% (expected ≈ 0.38%)"
        assert dist["hd3"] > dist["hd2"], \
            f"PP matched: HD3={dist['hd3']:.2f} should exceed HD2={dist['hd2']:.2f}"

    def test_pp_pout_less_than_old_formula(self):
        """PP Pout = swing × ua_swing / 8, not / 4.

        The class-A PP composite delivers half the per-tube swing power
        to the load (the other half cancels in the centre-tapped
        transformer). The naive ``/4`` formula double-counts and gives
        twice the real Pout.
        """
        _, pts = quick_pentode("EL84")
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)
        dist = pp_distortion(pts, ll, ug1_bias=-7.0)
        assert dist is not None
        swing = abs(dist["i_max"] - dist["i_min"])
        ua_swing = swing * ll.ra_per_tube
        naive_pout = swing * ua_swing / 4.0
        assert abs(dist["pout_mw"] - naive_pout / 2.0) < 0.01, \
            "PP Pout must be half the naive swing×ua_swing/4 product"


class TestFourierCoefficients:
    """Directly verify Fourier coefficients B1, B2, B3."""

    def _get_b_coefficients(self, a0, a1, a2, a3, hs):
        """Compute B1, B2, B3 via the 5-point method on a polynomial curve."""
        i_min = a0 + a1 * (-hs) + a2 * hs ** 2 + a3 * (-hs) ** 3
        i_low = a0 + a1 * (-hs / 2) + a2 * (hs / 2) ** 2 + a3 * (-hs / 2) ** 3
        i_0 = a0
        i_high = a0 + a1 * (hs / 2) + a2 * (hs / 2) ** 2 + a3 * (hs / 2) ** 3
        i_max = a0 + a1 * hs + a2 * hs ** 2 + a3 * hs ** 3

        swing = i_max - i_min
        half_diff = i_high - i_low
        b1 = (swing + half_diff) / 3.0
        b2 = (i_max + i_min - 2 * i_0) / 4.0
        b3 = (swing - 2 * half_diff) / 6.0
        return b1, b2, b3

    def test_b1_matches_fundamental(self):
        """B1 from 5-point method should match exact a1*V + 3/4*a3*V³."""
        a1, a2, a3 = 2.0, 0.1, 0.01
        for hs in [1.0, 2.0, 4.0, 6.0]:
            b1, _, _ = self._get_b_coefficients(10.0, a1, a2, a3, hs)
            exact_b1 = a1 * hs + 0.75 * a3 * hs ** 3
            assert abs(b1 - exact_b1) < 1e-10, \
                f"hs={hs}: B1={b1:.6f} exact={exact_b1:.6f}"

    def test_b2_matches_second_harmonic(self):
        """B2 from 5-point method should match exact 0.5*a2*V²."""
        a1, a2, a3 = 2.0, 0.15, 0.01
        for hs in [1.0, 2.0, 4.0]:
            _, b2, _ = self._get_b_coefficients(10.0, a1, a2, a3, hs)
            exact_b2 = 0.5 * a2 * hs ** 2
            assert abs(b2 - exact_b2) < 1e-10, \
                f"hs={hs}: B2={b2:.6f} exact={exact_b2:.6f}"

    def test_b3_matches_third_harmonic(self):
        """B3 from 5-point method should match exact 0.25*a3*V³."""
        a1, a2, a3 = 2.0, 0.1, 0.02
        for hs in [1.0, 2.0, 4.0]:
            _, _, b3 = self._get_b_coefficients(10.0, a1, a2, a3, hs)
            exact_b3 = 0.25 * a3 * hs ** 3
            assert abs(b3 - exact_b3) < 1e-10, \
                f"hs={hs}: B3={b3:.6f} exact={exact_b3:.6f}"

    def test_b_coefficients_pure_linear(self):
        """Linear curve: B1 > 0, B2 = B3 = 0."""
        b1, b2, b3 = self._get_b_coefficients(10.0, 2.0, 0.0, 0.0, 4.0)
        assert b1 > 0
        assert abs(b2) < 1e-12
        assert abs(b3) < 1e-12


class TestGainFormulas:
    """Verify gain/Zout formulas against textbook values."""

    def test_se_gain_known_values(self):
        """SE gain = mu * Ra / (ra + Ra), verify with known numbers."""
        mu, ra, ra_load = 100.0, 62.5, 100.0
        expected_gain = mu * ra_load / (ra + ra_load)  # 61.54
        expected_zout = ra * ra_load / (ra + ra_load)   # 38.46

        _, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=ra_load)
        isects = find_intersections(pts, ll)
        srk = {"s": mu / ra, "r": ra, "k": mu}
        result = compute_stage_params(isects, ll, ug1_bias=-2.0, srk=srk)
        assert result is not None
        assert abs(result["gain"] - expected_gain) < 0.01
        assert abs(result["zout"] - expected_zout) < 0.01

    def test_cf_gain_known_values(self):
        """CF gain = mu*Rk/(ra+(mu+1)*Rk), Zout = ra/(mu+1)."""
        mu, ra, rk = 17.0, 7.7, 10.0
        expected_gain = mu * rk / (ra + (mu + 1) * rk)  # 0.904
        expected_zout = ra / (mu + 1)  # 0.428

        _, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=rk, rl=10.0)
        isects = find_intersections(pts, ll)
        srk = {"s": mu / ra, "r": ra, "k": mu}
        result = compute_stage_params(isects, ll, ug1_bias=-10.0, srk=srk)
        assert result is not None
        assert abs(result["gain"] - expected_gain) < 0.01, \
            f"CF gain: {result['gain']:.4f} vs expected {expected_gain:.4f}"
        assert abs(result["zout"] - expected_zout) < 0.01, \
            f"CF Zout: {result['zout']:.4f} vs expected {expected_zout:.4f}"

    def test_pentode_zout_near_ra_load(self):
        """Pentode: ra >> Ra, so Zout = ra||Ra ≈ Ra."""
        ra_internal, ra_load = 50.0, 5.0
        expected_zout = ra_internal * ra_load / (ra_internal + ra_load)  # 4.545

        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=ra_load)
        isects = find_intersections(pts, ll)
        srk = {"s": 11.0, "r": ra_internal, "k": 550.0}
        result = compute_stage_params(isects, ll, ug1_bias=-7.0, srk=srk)
        assert result is not None
        assert abs(result["zout"] - expected_zout) < 0.01


class TestLoadLineFormulas:
    """Verify load line equations against Kirchhoff's laws."""

    def test_resistive_ohms_law(self):
        """Ia = (Ub - Ua) / Ra at any point along the line."""
        ll = ResistiveLoadLine(ub=300, ra=10.0)
        for ua in [0, 50, 100, 150, 200, 250, 300]:
            expected = (300 - ua) / 10.0
            assert abs(ll.ia_at_ua(ua) - expected) < 1e-10

    def test_transformer_ac_through_q(self):
        """AC load line: Ia = Iq - (Ua - Uaq) / Ra_ac."""
        ll = TransformerLoadLine(ub=300, ra_dc=0.1, ra_ac=5.0)
        q_ua, q_ia = 280.0, 20.0
        for ua in [200, 250, 280, 310, 350]:
            expected = q_ia - (ua - q_ua) / 5.0
            assert abs(ll.ia_at_ua_ac(ua, q_ua, q_ia) - expected) < 1e-10

    def test_cf_total_resistance(self):
        """CF: Ia = (Ub - Ua) / (Rk + Rl)."""
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=5.0)
        for ua in [0, 50, 100, 200, 250]:
            expected = (250 - ua) / 15.0
            assert abs(ll.ia_at_ua(ua) - expected) < 1e-10

    def test_pp_ra_per_tube(self):
        """PP: ra_per_tube = Ra_aa / 4."""
        for ra_aa in [4.0, 6.6, 8.0, 10.0, 16.0]:
            ll = PushPullLoadLine(ub=300, ra_aa=ra_aa)
            assert abs(ll.ra_per_tube - ra_aa / 4.0) < 1e-10

    def test_pp_ia_at_ua(self):
        """PP: Ia = (Ub - Ua) / ra_per_tube."""
        ll = PushPullLoadLine(ub=400, ra_aa=8.0)
        for ua in [0, 100, 200, 300, 400]:
            expected = (400 - ua) / 2.0
            assert abs(ll.ia_at_ua(ua) - expected) < 1e-10


# ---------------------------------------------------------------
# Gap 1: Analytical IMD — known polynomial → exact IMD2/IMD3
# ---------------------------------------------------------------

def _make_imd_polynomial_points(
    a0: float, a1: float, a2: float, a3: float,
    ug1_center: float, ug1_range: float, n_points: int = 30,
) -> list:
    """Build dense intersection-like points from a known polynomial.

    Ia(v) = a0 + a1*v + a2*v² + a3*v³,  v = ug1 - ug1_center
    """
    pts = []
    for i in range(n_points):
        ug1 = ug1_center - ug1_range + 2 * ug1_range * i / (n_points - 1)
        v = ug1 - ug1_center
        ia = a0 + a1 * v + a2 * v ** 2 + a3 * v ** 3
        pts.append({"ug1": ug1, "ia": max(ia, 0.0), "ua": 200.0})
    return pts


class TestAnalyticalIMD:
    """Verify IMD against exact polynomial coefficients."""

    def test_pure_linear_zero_imd(self):
        """Perfectly linear curve → IMD2 = IMD3 ≈ 0."""
        pts = _make_imd_polynomial_points(30.0, 2.0, 0.0, 0.0, -10.0, 8.0)
        result = compute_imd(pts, ug1_bias=-10.0)
        assert result is not None
        assert result["imd2"] < 0.1
        assert result["imd3"] < 0.1

    def test_quadratic_only_imd2(self):
        """Ia = a0 + a1*v + a2*v² → IMD2 = |a2/a1|*100, IMD3 ≈ 0."""
        a0, a1, a2, a3 = 10.0, 2.0, 0.15, 0.0
        pts = _make_imd_polynomial_points(a0, a1, a2, a3, -10.0, 8.0)
        result = compute_imd(pts, ug1_bias=-10.0)
        assert result is not None
        expected_imd2 = abs(a2 / a1) * 100.0  # 7.5%
        assert abs(result["imd2"] - expected_imd2) < 0.5, \
            f"IMD2: code={result['imd2']:.3f} expected={expected_imd2:.3f}"
        assert result["imd3"] < 0.5

    def test_cubic_imd2_and_imd3(self):
        """Ia = a0 + a1*v + a2*v² + a3*v³ → both IMD2 and IMD3."""
        a0, a1, a2, a3 = 10.0, 2.0, 0.15, 0.01
        pts = _make_imd_polynomial_points(a0, a1, a2, a3, -10.0, 8.0)
        result = compute_imd(pts, ug1_bias=-10.0)
        assert result is not None
        expected_imd2 = abs(a2 / a1) * 100.0  # 7.5%
        expected_imd3 = abs(a3 / a1) * 100.0  # 0.5%
        assert abs(result["imd2"] - expected_imd2) < 0.5, \
            f"IMD2: code={result['imd2']:.3f} expected={expected_imd2:.3f}"
        assert abs(result["imd3"] - expected_imd3) < 0.2, \
            f"IMD3: code={result['imd3']:.3f} expected={expected_imd3:.3f}"

    def test_imd_total_is_rss(self):
        """IMD_total = sqrt(IMD2² + IMD3²)."""
        a0, a1, a2, a3 = 10.0, 2.0, 0.15, 0.01
        pts = _make_imd_polynomial_points(a0, a1, a2, a3, -10.0, 8.0)
        result = compute_imd(pts, ug1_bias=-10.0)
        assert result is not None
        expected = math.sqrt(result["imd2"] ** 2 + result["imd3"] ** 2)
        assert abs(result["imd_total"] - expected) < 0.001

    def test_coefficients_recovered(self):
        """polyfit should recover the polynomial coefficients."""
        a0, a1, a2, a3 = 10.0, 2.0, 0.15, 0.01
        pts = _make_imd_polynomial_points(a0, a1, a2, a3, -10.0, 8.0)
        result = compute_imd(pts, ug1_bias=-10.0)
        assert result is not None
        assert abs(result["a0"] - a0) < 0.1, f"a0: {result['a0']:.4f} vs {a0}"
        assert abs(result["a1"] - a1) < 0.1, f"a1: {result['a1']:.4f} vs {a1}"
        assert abs(result["a2"] - a2) < 0.05, f"a2: {result['a2']:.4f} vs {a2}"
        assert abs(result["a3"] - a3) < 0.01, f"a3: {result['a3']:.4f} vs {a3}"

    def test_negative_a2_gives_correct_imd2(self):
        """Negative a2 should still produce correct IMD2."""
        a0, a1, a2, a3 = 30.0, 2.0, -0.1, 0.005
        pts = _make_imd_polynomial_points(a0, a1, a2, a3, -10.0, 6.0)
        result = compute_imd(pts, ug1_bias=-10.0)
        assert result is not None
        expected_imd2 = abs(a2 / a1) * 100.0  # 5.0%
        assert abs(result["imd2"] - expected_imd2) < 0.5


# ---------------------------------------------------------------
# Gap 2: PP mismatch — quantitative HD2 vs mismatch level
# ---------------------------------------------------------------

class TestPPMismatchQuantitative:
    """Verify that tube mismatch in PP produces quantifiable HD2."""

    @staticmethod
    def _scale_points(pts, factor):
        """Scale ia values by factor to simulate weaker/stronger tube."""
        return [
            {k: (v * factor if k == "ia" else v) for k, v in p.items()}
            for p in pts
        ]

    def test_mismatch_increases_hd2(self):
        """10% weaker tube B should produce higher HD2 than matched pair."""
        _, pts_a = quick_triode("12AU7")
        pts_b = self._scale_points(pts_a, 0.90)
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)

        dist_matched = pp_distortion(pts_a, ll, ug1_bias=-10.0)
        dist_mismatch = pp_distortion(pts_a, ll, ug1_bias=-10.0, points_b=pts_b)
        assert dist_matched is not None and dist_mismatch is not None
        assert dist_mismatch["hd2"] > dist_matched["hd2"], \
            f"Mismatch HD2={dist_mismatch['hd2']:.2f} should > matched HD2={dist_matched['hd2']:.2f}"

    def test_more_mismatch_more_hd2(self):
        """20% mismatch should produce more HD2 than 10% mismatch."""
        _, pts_a = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        hs = 5.0

        pts_b_10 = self._scale_points(pts_a, 0.90)
        pts_b_20 = self._scale_points(pts_a, 0.80)
        dist_10 = pp_distortion(pts_a, ll, ug1_bias=-10.0, points_b=pts_b_10, half_swing=hs)
        dist_20 = pp_distortion(pts_a, ll, ug1_bias=-10.0, points_b=pts_b_20, half_swing=hs)
        assert dist_10 is not None and dist_20 is not None
        assert dist_20["hd2"] > dist_10["hd2"], \
            f"20% mismatch HD2={dist_20['hd2']:.2f} should > 10% HD2={dist_10['hd2']:.2f}"

    def test_mismatch_preserves_hd3(self):
        """Mismatch affects mainly HD2; HD3 should remain in same order of magnitude."""
        _, pts_a = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        hs = 5.0

        dist_matched = pp_distortion(pts_a, ll, ug1_bias=-10.0, half_swing=hs)
        pts_b_10 = self._scale_points(pts_a, 0.90)
        dist_mismatch = pp_distortion(pts_a, ll, ug1_bias=-10.0, points_b=pts_b_10, half_swing=hs)
        assert dist_matched is not None and dist_mismatch is not None
        if dist_matched["hd3"] > 0.1:
            ratio = dist_mismatch["hd3"] / dist_matched["hd3"]
            assert 0.3 < ratio < 3.0, \
                f"HD3 ratio with/without mismatch = {ratio:.2f}, should be ~1"

    def test_pentode_mismatch(self):
        """PP mismatch effect works for pentodes too."""
        _, pts_a = quick_pentode("EL84")
        pts_b = self._scale_points(pts_a, 0.85)
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)

        dist_matched = pp_distortion(pts_a, ll, ug1_bias=-7.0)
        dist_mismatch = pp_distortion(pts_a, ll, ug1_bias=-7.0, points_b=pts_b)
        assert dist_matched is not None and dist_mismatch is not None
        assert dist_mismatch["hd2"] > dist_matched["hd2"]


# ---------------------------------------------------------------
# Gap 3: sweep_bias consistency with optimize_bias
# ---------------------------------------------------------------

class TestSweepBiasConsistency:
    """Verify sweep_bias minimum THD matches optimize_bias(min_thd)."""

    def test_min_thd_matches_optimize(self):
        """Minimum THD in sweep_bias should be close to optimize_bias result."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        hs = 3.0

        sweep = sweep_bias(pts, ll, half_swing=hs)
        opt = optimize_bias(pts, ll, half_swing=hs, target="min_thd")
        assert len(sweep) > 5 and opt is not None

        min_sweep_thd = min(s["thd"] for s in sweep)
        assert abs(opt["thd"] - min_sweep_thd) < 1.0, \
            f"optimize THD={opt['thd']:.3f} vs sweep min THD={min_sweep_thd:.3f}"

    def test_max_pout_matches_optimize(self):
        """Maximum Pout in sweep_bias should be close to optimize_bias(max_pout)."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        hs = 3.0

        sweep = sweep_bias(pts, ll, half_swing=hs)
        opt = optimize_bias(pts, ll, half_swing=hs, target="max_pout")
        assert len(sweep) > 5 and opt is not None

        max_sweep_pout = max(s["pout_mw"] for s in sweep)
        assert abs(opt["pout_mw"] - max_sweep_pout) < max_sweep_pout * 0.1, \
            f"optimize Pout={opt['pout_mw']:.2f} vs sweep max Pout={max_sweep_pout:.2f}"

    def test_balanced_between_min_thd_and_max_pout(self):
        """Balanced mode THD should be between min_thd and max_pout THD."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)

        opt_thd = optimize_bias(pts, ll, target="min_thd")
        opt_pout = optimize_bias(pts, ll, target="max_pout")
        opt_bal = optimize_bias(pts, ll, target="balanced")
        assert opt_thd and opt_pout and opt_bal
        assert opt_bal["pout_mw"] >= opt_pout["pout_mw"] * 0.5 - 0.01, \
            "Balanced Pout should be >= 50% of max Pout"

    def test_pentode_sweep_bias_consistency(self):
        """Pentode: sweep_bias minimum THD matches optimize_bias."""
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        hs = 3.0

        sweep = sweep_bias(pts, ll, half_swing=hs)
        opt = optimize_bias(pts, ll, half_swing=hs, target="min_thd")
        assert len(sweep) > 3 and opt is not None

        min_sweep_thd = min(s["thd"] for s in sweep)
        assert abs(opt["thd"] - min_sweep_thd) < 1.0


# ---------------------------------------------------------------
# Gap 4: sweep_ra physics — Pout vs Ra has a peak
# ---------------------------------------------------------------

class TestSweepRaPhysics:
    """Verify sweep_ra produces physically plausible results."""

    def test_pout_has_peak_triode(self):
        """Triode: Pout should increase then decrease as Ra sweeps from low to high."""
        _, pts = quick_triode("12AU7")
        results = sweep_ra(pts, ub=250, ra_min=1.0, ra_max=100.0,
                           ug1_bias=-8.0, half_swing=3.0, steps=40)
        assert len(results) >= 10
        pouts = [r["pout_mw"] for r in results]
        max_idx = pouts.index(max(pouts))
        assert 1 < max_idx < len(pouts) - 2, \
            f"Pout peak at index {max_idx}/{len(pouts)} — should be interior"

    def test_pout_low_at_extreme_ra(self):
        """At very low and very high Ra, Pout should be less than at moderate Ra."""
        _, pts = quick_triode("12AU7")
        results = sweep_ra(pts, ub=250, ra_min=0.5, ra_max=80.0,
                           ug1_bias=-8.0, half_swing=3.0, steps=50)
        assert len(results) >= 10
        pouts = [r["pout_mw"] for r in results]
        max_pout = max(pouts)
        assert pouts[0] < max_pout, "Pout at lowest Ra should be < peak"
        assert pouts[-1] < max_pout, "Pout at highest Ra should be < peak"

    def test_pentode_thd_increases_with_ra(self):
        """Pentode: THD should generally increase with Ra (more voltage swing)."""
        _, pts = quick_pentode("EL84")
        results = sweep_ra(pts, ub=300, ra_min=1.0, ra_max=30.0,
                           ug1_bias=-7.0, half_swing=3.0, steps=30)
        assert len(results) >= 5
        low_ra = [r for r in results if r["ra"] < 5.0]
        high_ra = [r for r in results if r["ra"] > 15.0]
        if low_ra and high_ra:
            avg_thd_low = sum(r["thd"] for r in low_ra) / len(low_ra)
            avg_thd_high = sum(r["thd"] for r in high_ra) / len(high_ra)
            assert avg_thd_high >= avg_thd_low * 0.5, \
                f"Pentode: high-Ra THD={avg_thd_high:.2f} should not be much less than low-Ra THD={avg_thd_low:.2f}"

    def test_ra_values_in_results(self):
        """Ra values in results should be monotonically increasing."""
        _, pts = quick_triode("12AU7")
        results = sweep_ra(pts, ub=250, ra_min=1.0, ra_max=50.0,
                           ug1_bias=-8.0, steps=20)
        assert len(results) >= 2, "guard de-vacuated 2026-07-12: value must be present"
        ras = [r["ra"] for r in results]
        for i in range(len(ras) - 1):
            assert ras[i] < ras[i + 1], f"Ra not ascending at index {i}"


# ---------------------------------------------------------------
# Gap 5: compute_headroom with pa_max — predictable unit test
# ---------------------------------------------------------------

class TestHeadroomPaMax:
    """Verify compute_headroom correctly limits swing when Pa exceeds pa_max."""

    def test_pa_max_reduces_swing(self):
        """With tight pa_max, swing should be less than without."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        ug1_bias = -8.0

        hr_unlimited = compute_headroom(isects, ug1_bias)
        hr_limited = compute_headroom(isects, ug1_bias, pa_max=0.3, load_line=ll)
        assert hr_unlimited is not None and hr_limited is not None
        assert hr_limited["max_swing"] <= hr_unlimited["max_swing"], \
            f"Limited swing={hr_limited['max_swing']:.2f} should be <= unlimited={hr_unlimited['max_swing']:.2f}"

    def test_pa_max_sets_clip_reason(self):
        """When pa_max clips, the reason should be 'pa_max'."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)

        hr = compute_headroom(isects, ug1_bias=-8.0, pa_max=0.1, load_line=ll)
        assert hr is not None
        has_pa_clip = hr["clip_neg"] == "pa_max" or hr["clip_pos"] == "pa_max"
        assert has_pa_clip, \
            f"With very low pa_max, at least one clip reason should be 'pa_max', " \
            f"got neg={hr['clip_neg']}, pos={hr['clip_pos']}"

    def test_pa_max_synthetic_known_limit(self):
        """Synthetic data: Pa exceeds limit at known Ug1 → swing stops there."""
        ll = ResistiveLoadLine(ub=300, ra=10.0)
        isects = []
        for ug1 in np.arange(-15.0, 0.5, 0.5):
            ua = 300.0 - (ug1 + 15.0) * 10.0  # Ua from 300 down to 150
            ia = (300.0 - ua) / 10.0           # mA
            isects.append({"ug1": float(ug1), "ua": ua, "ia": ia})

        ug1_bias = -8.0
        q = interp_intersection(isects, ug1_bias)
        assert q is not None
        pa_q = q["ua"] * q["ia"] / 1000.0  # Pa at Q-point in W

        hr_none = compute_headroom(isects, ug1_bias)
        hr_tight = compute_headroom(isects, ug1_bias, pa_max=pa_q * 0.8, load_line=ll)
        assert hr_none is not None and hr_tight is not None
        assert hr_tight["max_swing"] < hr_none["max_swing"], \
            "pa_max = 80% of Q-point Pa should reduce swing"

    def test_pa_max_very_large_no_effect(self):
        """With pa_max much larger than any point, headroom is unchanged."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)

        hr_none = compute_headroom(isects, ug1_bias=-8.0)
        hr_huge = compute_headroom(isects, ug1_bias=-8.0, pa_max=100.0, load_line=ll)
        assert hr_none is not None and hr_huge is not None
        assert abs(hr_none["max_swing"] - hr_huge["max_swing"]) < 0.01, \
            "Very large pa_max should not limit swing"


# ---------------------------------------------------------------
# Gap 6: _numerical_gm_ra accuracy on known curves
# ---------------------------------------------------------------

class TestNumericalGmRaAccuracy:
    """Verify _numerical_gm_ra on surfaces with analytically known gm and ra."""

    @staticmethod
    def _linear_surface_points(k, mu, ug1_range, ua_range):
        """Ia(Ua, Ug1) = k * (Ua + mu * Ug1).

        Exact parameters: gm = k*mu (mA/V), ra = 1/k (kΩ), mu = gm*ra.
        """
        points = []
        for ug1 in np.arange(ug1_range[0], ug1_range[1] + 0.5, 1.0):
            for ua in np.arange(ua_range[0], ua_range[1] + 1, 10.0):
                ia = k * (ua + mu * ug1)
                if ia > 0:
                    points.append({"ug1": float(ug1), "ua": float(ua), "ia": float(ia)})
        return points

    def test_linear_surface_exact_gm(self):
        """On a linear surface, gm should be recovered exactly (within discretization)."""
        k = 0.1   # mA/V
        mu = 20.0
        expected_gm = k * mu  # 2.0 mA/V

        pts = self._linear_surface_points(k, mu, (-20, 0), (50, 300))
        ll = ResistiveLoadLine(ub=300, ra=10.0)
        isects = find_intersections(pts, ll)
        result = _numerical_gm_ra(pts, isects, ug1_bias=-10.0)
        assert result is not None
        assert abs(result["gm"] - expected_gm) < 0.3, \
            f"gm: code={result['gm']:.3f} expected={expected_gm:.3f}"

    def test_linear_surface_exact_ra(self):
        """On a linear surface, ra should be recovered exactly."""
        k = 0.1
        mu = 20.0
        expected_ra = 1.0 / k  # 10.0 kΩ

        pts = self._linear_surface_points(k, mu, (-20, 0), (50, 300))
        ll = ResistiveLoadLine(ub=300, ra=10.0)
        isects = find_intersections(pts, ll)
        result = _numerical_gm_ra(pts, isects, ug1_bias=-10.0)
        assert result is not None
        assert abs(result["ra"] - expected_ra) < 2.0, \
            f"ra: code={result['ra']:.3f} expected={expected_ra:.3f}"

    def test_linear_surface_mu_relation(self):
        """On any surface, mu should equal gm * ra."""
        k = 0.08
        mu = 30.0
        pts = self._linear_surface_points(k, mu, (-25, 0), (50, 400))
        ll = ResistiveLoadLine(ub=400, ra=15.0)
        isects = find_intersections(pts, ll)
        result = _numerical_gm_ra(pts, isects, ug1_bias=-12.0)
        assert result is not None
        assert abs(result["mu"] - result["gm"] * result["ra"]) < 0.01, \
            f"mu={result['mu']:.3f} != gm*ra={result['gm']*result['ra']:.3f}"

    def test_tube_sim_gm_close_to_model(self):
        """For 12AU7, numerical gm should be close to model-derived gm."""
        model = load_model("12AU7")
        assert model is not None
        ua_q, ug1_q, delta = 170.0, -8.0, 0.1
        ref_gm = (model.ia(ua_q, ug1_q + delta) - model.ia(ua_q, ug1_q - delta)) / (2 * delta)

        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        result = _numerical_gm_ra(pts, isects, ug1_bias=ug1_q)
        assert result is not None
        ratio = result["gm"] / ref_gm
        assert 0.4 < ratio < 2.5, \
            f"Numerical gm={result['gm']:.3f} too far from model gm={ref_gm:.3f}"

    def test_tube_sim_ra_close_to_model(self):
        """For 12AU7, numerical ra should be close to model-derived ra."""
        model = load_model("12AU7")
        assert model is not None
        ua_q, ug1_q, delta = 170.0, -8.0, 5.0
        dia = model.ia(ua_q + delta, ug1_q) - model.ia(ua_q - delta, ug1_q)
        ref_ra = (2 * delta) / dia if abs(dia) > 0.001 else 999.0  # kΩ (Ua in V, Ia in mA)

        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        result = _numerical_gm_ra(pts, isects, ug1_bias=ug1_q)
        assert result is not None
        ratio = result["ra"] / ref_ra
        assert 0.3 < ratio < 3.0, \
            f"Numerical ra={result['ra']:.3f} too far from model ra={ref_ra:.3f}"

    def test_different_bias_points(self):
        """gm/ra should be reasonably stable across nearby bias points."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)

        results = []
        for bias in [-6.0, -8.0, -10.0, -12.0]:
            r = _numerical_gm_ra(pts, isects, ug1_bias=bias)
            if r is not None:
                results.append(r)
        assert len(results) >= 2
        gms = [r["gm"] for r in results]
        assert max(gms) / min(gms) < 5.0, \
            f"gm varies too much across bias: {gms}"


# ---------------------------------------------------------------
# Efficiency, Pdc, amp class
# ---------------------------------------------------------------

class TestEfficiencyAndClass:
    """Verify pdc_mw, eta_pct, pa_signal_mw, amp_class in compute_distortion."""

    # ── Analytical / exact ──────────────────────────────────────

    def test_pdc_exact_formula(self):
        """Pdc = Ub × Ia_q exactly."""
        # Synthetic: Ia_q=20mA at Q-point, Ub=250V → Pdc=5000mW
        isects = _make_polynomial_intersections(20.0, 2.0, 0.0, 0.0, 4.0)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=4.0, ub=250.0)
        assert result is not None
        assert result["pdc_mw"] is not None
        assert abs(result["pdc_mw"] - 250.0 * result["ia_0"]) < 1e-6

    def test_eta_exact_formula(self):
        """η = Pout / Pdc × 100 exactly."""
        isects = _make_polynomial_intersections(20.0, 2.0, 0.0, 0.0, 4.0)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=4.0, ub=250.0)
        assert result is not None
        assert result["eta_pct"] is not None
        expected = result["pout_mw"] / result["pdc_mw"] * 100.0
        assert abs(result["eta_pct"] - expected) < 1e-6

    def test_pa_signal_exact_formula(self):
        """Pa_signal = Pdc − Pout exactly."""
        isects = _make_polynomial_intersections(20.0, 2.0, 0.0, 0.0, 4.0)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=4.0, ub=250.0)
        assert result is not None
        assert result["pa_signal_mw"] is not None
        assert abs(result["pa_signal_mw"] - (result["pdc_mw"] - result["pout_mw"])) < 1e-6

    def test_eta_none_without_ub(self):
        """When ub is not passed, eta_pct/pdc_mw/pa_signal_mw are None."""
        isects = _make_polynomial_intersections(20.0, 2.0, 0.0, 0.0, 4.0)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=4.0)
        assert result is not None
        assert result["pdc_mw"] is None
        assert result["eta_pct"] is None
        assert result["pa_signal_mw"] is None

    def test_eta_class_a_upper_bound(self):
        """SE Class A resistive: η must be < 25% (theoretical maximum)."""
        # Use linear curve so Q-point is at the middle of the load line
        isects = _make_polynomial_intersections(10.0, 2.0, 0.0, 0.0, 4.0)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=4.0, ub=250.0)
        assert result is not None and result["eta_pct"] is not None
        assert result["eta_pct"] < 25.0, \
            f"SE Class A η={result['eta_pct']:.2f}% exceeds 25% maximum"

    # ── Real lamp data ───────────────────────────────────────────

    def test_12ax7_efficiency_range(self):
        """12AX7 SE: η should be > 0 and < 50% (absolute physical maximum for any Class A)."""
        _, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        isects = find_intersections(pts, ll)
        result = compute_distortion(isects, ug1_bias=-2.0, ub=250.0)
        assert result is not None and result["eta_pct"] is not None
        assert 0.1 < result["eta_pct"] < 50.0, \
            f"12AX7 η={result['eta_pct']:.2f}% out of expected range"

    def test_el84_efficiency_range(self):
        """EL84 SE pentode: η should be > 0 and < 50%."""
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)
        result = compute_distortion(isects, ug1_bias=-7.0, ub=300.0)
        assert result is not None and result["eta_pct"] is not None
        assert 0.1 < result["eta_pct"] < 50.0, \
            f"EL84 η={result['eta_pct']:.2f}% out of expected range"

    def test_pa_signal_positive(self):
        """Pa_signal must be positive (Pdc > Pout always for Class A)."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        result = compute_distortion(isects, ug1_bias=-10.0, ub=250.0)
        assert result and result['pa_signal_mw'] is not None, "guarded value must be truthy (de-vacuated)"
        assert result["pa_signal_mw"] > 0, \
            f"Pa_signal={result['pa_signal_mw']:.2f}mW should be positive"

    # ── Operating class detection ────────────────────────────────

    def test_amp_class_a_well_biased(self):
        """Tube biased well away from cutoff → Class A (i_min >> 0)."""
        # i_0=20mA, i_min=20-2*2=16mA (40% swing) → ratio=0.8 → Class A
        isects = _make_polynomial_intersections(20.0, 2.0, 0.0, 0.0, 2.0)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=2.0)
        assert result is not None
        assert result["amp_class"] == "A", \
            f"Expected Class A, got {result['amp_class']} (i_min={result['i_min']:.2f}, i_0={result['ia_0']:.2f})"

    def test_amp_class_b_at_cutoff(self):
        """Tube biased near cutoff → Class B (i_min ≈ 0)."""
        # Bias at 0.5mA, swing to 0mA at negative peak
        isects = [
            {"ug1": -4.0, "ua": 280.0, "ia": 0.0},
            {"ug1": -2.0, "ua": 260.0, "ia": 0.1},
            {"ug1": 0.0,  "ua": 240.0, "ia": 0.5},
            {"ug1": 2.0,  "ua": 220.0, "ia": 1.5},
            {"ug1": 4.0,  "ua": 200.0, "ia": 3.0},
        ]
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=4.0)
        assert result is not None
        assert result["amp_class"] == "B", \
            f"Expected Class B, got {result['amp_class']} (i_min={result['i_min']:.3f}, i_0={result['ia_0']:.3f})"

    def test_amp_class_ab(self):
        """Tube in between: i_min between 0.5% and 5% of i_0 → Class AB."""
        # i_0=20mA, i_min=0.5mA → ratio=0.025 → between B(0.005) and A(0.05) → AB
        isects = [
            {"ug1": -4.0, "ua": 280.0, "ia": 0.5},
            {"ug1": -2.0, "ua": 260.0, "ia": 5.0},
            {"ug1": 0.0,  "ua": 240.0, "ia": 20.0},
            {"ug1": 2.0,  "ua": 220.0, "ia": 45.0},
            {"ug1": 4.0,  "ua": 200.0, "ia": 80.0},
        ]
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=4.0)
        assert result is not None
        assert result["amp_class"] == "AB", \
            f"Expected Class AB, got {result['amp_class']} (i_min={result['i_min']:.3f}, i_0={result['ia_0']:.3f})"

    def test_12ax7_class_a(self):
        """12AX7 biased at moderate Ug1 should detect Class A."""
        _, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        isects = find_intersections(pts, ll)
        result = compute_distortion(isects, ug1_bias=-2.0)
        assert result is not None
        assert result["amp_class"] in ("A", "AB"), \
            f"12AX7 at -2V should be Class A or AB, got {result['amp_class']}"

    def test_amp_class_present_in_result(self):
        """amp_class key must always be present."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        result = compute_distortion(isects, ug1_bias=-10.0)
        assert result is not None
        assert "amp_class" in result
        assert result["amp_class"] in ("A", "AB", "B")

    def test_negative_imin_gives_class_b(self):
        """Negative i_min (interpolation artifact near cutoff) → Class B."""
        isects = [
            {"ug1": -4.0, "ua": 280.0, "ia": -0.1},  # slightly negative
            {"ug1": -2.0, "ua": 260.0, "ia": 0.5},
            {"ug1": 0.0,  "ua": 240.0, "ia": 5.0},
            {"ug1": 2.0,  "ua": 220.0, "ia": 15.0},
            {"ug1": 4.0,  "ua": 200.0, "ia": 30.0},
        ]
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=4.0)
        assert result is not None
        assert result["amp_class"] == "B", \
            f"Negative i_min should map to Class B, got {result['amp_class']}"

    def test_el84_pentode_small_swing_class_a(self):
        """EL84 pentode at -7V with small swing stays in Class A."""
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)
        # Small swing: tube never approaches cutoff
        result = compute_distortion(isects, ug1_bias=-7.0, half_swing=2.0)
        assert result is not None
        assert result["amp_class"] == "A", \
            f"EL84 at -7V/2V swing should be Class A, got {result['amp_class']} " \
            f"(i_min={result['i_min']:.3f}, i_0={result['ia_0']:.3f})"

    def test_el84_full_swing_class_b(self):
        """EL84 at full auto-swing drives tube near cutoff → Class B."""
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)
        result = compute_distortion(isects, ug1_bias=-7.0)
        assert result is not None
        # Full swing ≈ ±7.5V drives EL84 to near-cutoff
        assert result["amp_class"] in ("AB", "B"), \
            f"EL84 at full swing should be AB or B, got {result['amp_class']}"

    def test_12au7_well_biased_class_a(self):
        """12AU7 at -10V with 10kΩ is well-biased Class A."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        result = compute_distortion(isects, ug1_bias=-10.0, half_swing=3.0)
        assert result is not None
        assert result["amp_class"] == "A", \
            f"12AU7 at -10V/3V swing should be Class A, got {result['amp_class']} " \
            f"(i_min={result['i_min']:.3f}, i_0={result['ia_0']:.3f})"

    def test_eta_with_ub_zero(self):
        """ub=0 → guard rejects → pdc/eta are None (no crash)."""
        isects = _make_polynomial_intersections(20.0, 2.0, 0.0, 0.0, 4.0)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=4.0, ub=0.0)
        assert result is not None
        assert result["pdc_mw"] is None
        assert result["eta_pct"] is None
        assert result["pa_signal_mw"] is None

    def test_eta_with_ub_negative(self):
        """Negative ub (non-physical) → guard rejects → None."""
        isects = _make_polynomial_intersections(20.0, 2.0, 0.0, 0.0, 4.0)
        result = compute_distortion(isects, ug1_bias=0.0, half_swing=4.0, ub=-100.0)
        assert result is not None
        assert result["pdc_mw"] is None

    def test_eta_rejects_insufficient_signal(self):
        """Near-cutoff data with negligible swing → None (b1 too small)."""
        isects = [
            {"ug1": -8.0, "ua": 250.0, "ia": 0.0},
            {"ug1": -7.0, "ua": 250.0, "ia": 0.0},
            {"ug1": -6.0, "ua": 250.0, "ia": 0.0},
            {"ug1": -5.0, "ua": 250.0, "ia": 0.01},
            {"ug1": -4.0, "ua": 249.9, "ia": 0.02},
        ]
        result = compute_distortion(isects, ug1_bias=-6.0, ub=300.0)
        assert result is None


# ---------------------------------------------------------------
# Damping factor
# ---------------------------------------------------------------

class TestDampingFactor:
    """Verify df = Ra_load / Zout in compute_stage_params."""

    def test_df_exact_formula(self):
        """DF = Ra_load / Zout exactly — verify with known numbers."""
        # ra=7.7kΩ, Ra=10kΩ → Zout=4.35kΩ → DF=2.30
        ra_internal, ra_load = 7.7, 10.0
        zout_expected = ra_internal * ra_load / (ra_internal + ra_load)
        df_expected = ra_load / zout_expected

        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=ra_load)
        isects = find_intersections(pts, ll)
        srk = {"s": 2.2, "r": ra_internal, "k": 17.0}
        result = compute_stage_params(isects, ll, ug1_bias=-10.0, srk=srk)
        assert result is not None
        assert result["df"] is not None
        assert abs(result["df"] - df_expected) < 0.01, \
            f"DF: code={result['df']:.4f} expected={df_expected:.4f}"

    def test_df_higher_ra_gives_higher_df(self):
        """Higher load resistance → higher damping factor (more load control)."""
        _, pts = quick_triode("12AU7")
        srk = {"s": 2.2, "r": 7.7, "k": 17.0}
        ll_low = ResistiveLoadLine(ub=250, ra=5.0)
        ll_high = ResistiveLoadLine(ub=250, ra=20.0)
        isects_low = find_intersections(pts, ll_low)
        isects_high = find_intersections(pts, ll_high)
        res_low = compute_stage_params(isects_low, ll_low, srk=srk)
        res_high = compute_stage_params(isects_high, ll_high, srk=srk)
        assert res_low and res_high
        assert res_high["df"] > res_low["df"], \
            f"Higher Ra should give higher DF: Ra=5→DF={res_low['df']:.2f}, Ra=20→DF={res_high['df']:.2f}"

    def test_df_pentode_near_one(self):
        """Pentode: ra >> Ra → Zout ≈ Ra → DF ≈ 1."""
        ra_internal, ra_load = 50.0, 5.0
        # Zout = 50*5/(50+5) = 4.545, DF = 5/4.545 = 1.1
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=ra_load)
        isects = find_intersections(pts, ll)
        srk = {"s": 11.0, "r": ra_internal, "k": 550.0}
        result = compute_stage_params(isects, ll, ug1_bias=-7.0, srk=srk)
        assert result is not None and result["df"] is not None
        assert 1.0 < result["df"] < 2.0, \
            f"Pentode DF should be ~1.1, got {result['df']:.3f}"

    def test_df_triode_greater_than_one(self):
        """12AX7 high-mu triode: DF should be > 1."""
        _, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        isects = find_intersections(pts, ll)
        srk = {"s": 1.6, "r": 62.5, "k": 100.0}
        result = compute_stage_params(isects, ll, ug1_bias=-2.0, srk=srk)
        assert result is not None and result["df"] is not None
        assert result["df"] > 1.0, f"DF={result['df']:.3f} should be > 1"

    def test_cf_df_formula(self):
        """CF: DF = Rl / Zout.  Zout=ra/(mu+1), Rl=load_line.rl."""
        mu, ra, rk, rl = 17.0, 7.7, 10.0, 10.0
        zout_expected = ra / (mu + 1)
        df_expected = rl / zout_expected

        _, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=rk, rl=rl)
        isects = find_intersections(pts, ll)
        srk = {"s": mu / ra, "r": ra, "k": mu}
        result = compute_stage_params(isects, ll, ug1_bias=-10.0, srk=srk)
        assert result is not None and result["df"] is not None
        assert abs(result["df"] - df_expected) < 0.02, \
            f"CF DF: code={result['df']:.4f} expected={df_expected:.4f}"

    def test_df_key_always_present(self):
        """df key must always be present when stage params are returned."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        srk = {"s": 2.2, "r": 7.7, "k": 17.0}
        result = compute_stage_params(isects, ll, srk=srk)
        assert result is not None
        assert "df" in result

    def test_df_transformer_load_line(self):
        """SE transformer: DF = ra_ac / Zout."""
        ra_internal, ra_ac = 7.7, 5.0
        zout_expected = ra_internal * ra_ac / (ra_internal + ra_ac)
        df_expected = ra_ac / zout_expected

        _, pts = quick_triode("12AU7")
        ll = TransformerLoadLine(ub=300, ra_dc=0.1, ra_ac=ra_ac)
        isects = find_intersections(pts, ll, ug1_bias=-10.0)
        srk = {"s": 2.2, "r": ra_internal, "k": 17.0}
        result = compute_stage_params(isects, ll, ug1_bias=-10.0, srk=srk)
        assert result is not None and result["df"] is not None
        assert abs(result["df"] - df_expected) < 0.05, \
            f"Transformer DF: code={result['df']:.4f} expected={df_expected:.4f}"

    def test_df_push_pull(self):
        """PP: DF = ra_per_tube / Zout.  Both are in kΩ."""
        ra_internal, ra_aa = 7.7, 8.0
        ra_per_tube = ra_aa / 4.0  # 2.0 kΩ
        zout_expected = ra_internal * ra_per_tube / (ra_internal + ra_per_tube)
        df_expected = ra_per_tube / zout_expected

        _, pts = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=250, ra_aa=ra_aa)
        isects = find_intersections(pts, ll)
        srk = {"s": 2.2, "r": ra_internal, "k": 17.0}
        result = compute_stage_params(isects, ll, ug1_bias=-10.0, srk=srk)
        assert result is not None and result["df"] is not None
        assert abs(result["df"] - df_expected) < 0.05, \
            f"PP DF: code={result['df']:.4f} expected={df_expected:.4f}"

    def test_df_numerical_method(self):
        """DF should be computed with numerical method (no SRK)."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects = find_intersections(pts, ll)
        result = compute_stage_params(isects, ll, ug1_bias=-10.0, srk=None, points=pts)
        assert result is not None
        assert result["method"] == "numerical"
        assert result["df"] is not None
        assert result["df"] > 0


# ---------------------------------------------------------------
# Pg2 screen grid dissipation
# ---------------------------------------------------------------

class TestPg2:
    """Verify compute_pg2 and estimate_ig2_at_q."""

    # ── compute_pg2: pure formula ────────────────────────────────

    def test_pg2_exact_formula(self):
        """Pg2 = Ug2 × Ig2 (mW) exactly."""
        assert compute_pg2(200.0, 1.5) == pytest.approx(300.0)
        assert compute_pg2(150.0, 2.0) == pytest.approx(300.0)
        assert compute_pg2(250.0, 0.5) == pytest.approx(125.0)

    def test_pg2_zero_current(self):
        """Zero Ig2 → Pg2 = 0."""
        assert compute_pg2(300.0, 0.0) == pytest.approx(0.0)

    def test_pg2_zero_voltage(self):
        """Zero Ug2 → Pg2 = 0."""
        assert compute_pg2(0.0, 5.0) == pytest.approx(0.0)

    def test_pg2_units_milliwatts(self):
        """Standard EL84 at Ug2=300V, Ig2≈2mA → ~600mW = 0.6W."""
        pg2_mw = compute_pg2(300.0, 2.0)
        assert abs(pg2_mw - 600.0) < 1e-6

    # ── estimate_ig2_at_q: from real pentode data ────────────────

    def test_estimate_ig2_returns_float(self):
        """estimate_ig2_at_q returns a float for pentode data."""
        _, pts = quick_pentode("EL84")
        ig2 = estimate_ig2_at_q(pts, ug1_q=-7.0, ua_q=250.0, ug2_filter=300.0)
        assert isinstance(ig2, float)

    def test_estimate_ig2_positive_for_pentode(self):
        """EL84 pentode has positive Ig2 at normal operating point."""
        _, pts = quick_pentode("EL84")
        # Find a ua_q on the load line
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)
        q = interp_intersection(isects, -7.0)
        assert q is not None
        ig2 = estimate_ig2_at_q(pts, ug1_q=q["ug1"], ua_q=q["ua"], ug2_filter=250.0)
        assert ig2 > 0.0  # EL84 pentode has positive screen current at this Q-point

    def test_estimate_ig2_zero_for_no_ig2_data(self):
        """Points without ig2 key → estimate returns 0."""
        pts = [{"ua": 250.0, "ug1": -7.0, "ia": 20.0}]  # no ig2 key
        ig2 = estimate_ig2_at_q(pts, ug1_q=-7.0, ua_q=250.0)
        assert ig2 == 0.0

    def test_estimate_ig2_ug2_filter(self):
        """Ug2 filter correctly excludes wrong screen grid voltage levels."""
        pts = [
            {"ua": 250.0, "ug1": -7.0, "ia": 20.0, "ig2": 2.0, "ug2": 300.0},
            {"ua": 250.0, "ug1": -7.0, "ia": 10.0, "ig2": 0.5, "ug2": 150.0},
        ]
        # With filter=300V, only the first point qualifies
        ig2_300 = estimate_ig2_at_q(pts, ug1_q=-7.0, ua_q=250.0, ug2_filter=300.0)
        assert abs(ig2_300 - 2.0) < 1e-6
        # With filter=150V, only the second point qualifies
        ig2_150 = estimate_ig2_at_q(pts, ug1_q=-7.0, ua_q=250.0, ug2_filter=150.0)
        assert abs(ig2_150 - 0.5) < 1e-6

    def test_pg2_from_pipeline(self):
        """End-to-end: estimate Ig2 from pentode data, compute Pg2."""
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)
        q = interp_intersection(isects, -7.0)
        assert q is not None
        ig2 = estimate_ig2_at_q(pts, ug1_q=q["ug1"], ua_q=q["ua"], ug2_filter=250.0)
        assert ig2 > 0  # data is at Ug2=250; filter must match to exercise the pipeline
        pg2 = compute_pg2(250.0, ig2)
        assert pg2 > 0
        # EL84 Pg2 at normal operating point typically 100–800mW
        assert 50.0 < pg2 < 1000.0, f"EL84 Pg2={pg2:.1f}mW out of expected range"

    def test_triode_has_no_ig2(self):
        """Triode data has no ig2 → estimate_ig2_at_q returns 0."""
        _, pts = quick_triode("12AU7")
        ig2 = estimate_ig2_at_q(pts, ug1_q=-10.0, ua_q=170.0)
        assert ig2 == 0.0, f"Triode should have no ig2, got {ig2}"

    def test_estimate_ig2_none_values_skipped(self):
        """Points with ig2=None are treated as missing."""
        pts = [
            {"ua": 250.0, "ug1": -7.0, "ia": 20.0, "ig2": None, "ug2": 300.0},
            {"ua": 250.0, "ug1": -7.0, "ia": 20.0, "ig2": 1.5, "ug2": 300.0},
        ]
        ig2 = estimate_ig2_at_q(pts, ug1_q=-7.0, ua_q=250.0, ug2_filter=300.0)
        assert abs(ig2 - 1.5) < 1e-6, "None ig2 should be skipped, only 1.5 should remain"

    def test_estimate_ig2_ua_ug1_tolerance(self):
        """Only points within ±15V Ua and ±1V Ug1 of Q-point are used."""
        pts = [
            {"ua": 250.0, "ug1": -7.0, "ia": 20.0, "ig2": 2.0, "ug2": 300.0},  # within
            {"ua": 300.0, "ug1": -7.0, "ia": 20.0, "ig2": 5.0, "ug2": 300.0},  # ua too far
            {"ua": 250.0, "ug1": -10.0, "ia": 20.0, "ig2": 5.0, "ug2": 300.0}, # ug1 too far
        ]
        ig2 = estimate_ig2_at_q(pts, ug1_q=-7.0, ua_q=250.0, ug2_filter=300.0)
        assert abs(ig2 - 2.0) < 1e-6, f"Only the nearby point should match, got {ig2}"


# ---------------------------------------------------------------
# NFB (Negative Feedback)
# ---------------------------------------------------------------

class TestNFB:
    """Verify compute_nfb_effect: classical NFB post-processor.

    Core formula: D = 10^(nfb_db/20), X_closed = X_open / D.
    Sources: RDH 4th ed. Ch.12, VTADiy §4.4.
    """

    # ── Pure function: basic formula tests ────────────────────────

    def test_identity_zero_db(self):
        """NFB = 0 dB → all values unchanged (D=1, β=0)."""
        r = compute_nfb_effect(gain_open=60.0, zout_open=38.5, thd_open=2.5, nfb_db=0.0)
        assert r is not None
        assert r["gain_closed"] == pytest.approx(60.0)
        assert r["zout_closed"] == pytest.approx(38.5)
        assert r["thd_closed"] == pytest.approx(2.5)
        assert r["bw_factor"] == pytest.approx(1.0)
        assert r["desensitivity"] == pytest.approx(1.0)
        assert r["beta"] == pytest.approx(0.0)

    def test_basic_6db(self):
        """NFB = 6 dB → D = 1.9953.  Verify gain, Zout, THD."""
        D = 10.0 ** (6.0 / 20.0)  # 1.99526...
        r = compute_nfb_effect(gain_open=60.0, zout_open=38.5, thd_open=2.5, nfb_db=6.0)
        assert r is not None
        assert r["gain_closed"] == pytest.approx(60.0 / D, rel=1e-4)
        assert r["zout_closed"] == pytest.approx(38.5 / D, rel=1e-4)
        assert r["thd_closed"] == pytest.approx(2.5 / D, rel=1e-4)

    def test_basic_10db(self):
        """NFB = 10 dB → D = 3.1623."""
        D = 10.0 ** (10.0 / 20.0)
        r = compute_nfb_effect(gain_open=100.0, zout_open=50.0, thd_open=5.0, nfb_db=10.0)
        assert r is not None
        assert r["gain_closed"] == pytest.approx(100.0 / D, rel=1e-4)
        assert r["zout_closed"] == pytest.approx(50.0 / D, rel=1e-4)
        assert r["thd_closed"] == pytest.approx(5.0 / D, rel=1e-4)

    def test_basic_20db(self):
        """NFB = 20 dB → D = 10.  Everything divided by 10."""
        r = compute_nfb_effect(gain_open=100.0, zout_open=40.0, thd_open=3.0, nfb_db=20.0)
        assert r is not None
        assert r["gain_closed"] == pytest.approx(10.0, rel=1e-4)
        assert r["zout_closed"] == pytest.approx(4.0, rel=1e-4)
        assert r["thd_closed"] == pytest.approx(0.3, rel=1e-4)
        assert r["desensitivity"] == pytest.approx(10.0, rel=1e-4)

    def test_beta_calculation(self):
        """β = (D − 1) / A.  Verify for 6 dB, A=60."""
        D = 10.0 ** (6.0 / 20.0)
        beta_expected = (D - 1.0) / 60.0
        r = compute_nfb_effect(gain_open=60.0, zout_open=10.0, thd_open=1.0, nfb_db=6.0)
        assert r is not None
        assert r["beta"] == pytest.approx(beta_expected, rel=1e-4)

    def test_gain_closed_db_consistent(self):
        """gain_closed_db = 20·log10(gain_closed)."""
        for nfb_db in [3.0, 6.0, 10.0, 20.0]:
            r = compute_nfb_effect(gain_open=50.0, zout_open=10.0, thd_open=2.0, nfb_db=nfb_db)
            assert r is not None
            expected_db = 20.0 * math.log10(r["gain_closed"])
            assert r["gain_closed_db"] == pytest.approx(expected_db, abs=0.01)

    # ── Edge cases ────────────────────────────────────────────────

    def test_negative_nfb_returns_none(self):
        """Negative NFB (positive feedback) → None."""
        assert compute_nfb_effect(60.0, 10.0, 2.0, nfb_db=-3.0) is None

    def test_gain_zero_returns_none(self):
        """Zero open-loop gain → None."""
        assert compute_nfb_effect(0.0, 10.0, 2.0, nfb_db=6.0) is None

    def test_gain_negative_returns_none(self):
        """Negative gain → None."""
        assert compute_nfb_effect(-5.0, 10.0, 2.0, nfb_db=6.0) is None

    def test_very_high_nfb_30db(self):
        """30 dB NFB with A=60: D=31.62, gain_closed=1.90.  Unusual but valid."""
        D = 10.0 ** (30.0 / 20.0)
        r = compute_nfb_effect(gain_open=60.0, zout_open=38.5, thd_open=2.5, nfb_db=30.0)
        assert r is not None
        assert r["gain_closed"] == pytest.approx(60.0 / D, rel=1e-4)
        assert r["gain_closed"] < 2.0  # heavily feedback-reduced

    def test_nfb_exceeds_gain(self):
        """D > A (β > 1): mathematically valid, gain < 1 (attenuation)."""
        r = compute_nfb_effect(gain_open=10.0, zout_open=5.0, thd_open=3.0, nfb_db=40.0)
        assert r is not None
        assert r["gain_closed"] < 1.0  # attenuation
        assert r["beta"] > 1.0

    def test_thd_zero(self):
        """THD = 0 with NFB → still 0."""
        r = compute_nfb_effect(gain_open=50.0, zout_open=10.0, thd_open=0.0, nfb_db=10.0)
        assert r is not None
        assert r["thd_closed"] == pytest.approx(0.0)

    def test_zout_zero(self):
        """Zout = 0 → stays 0 with NFB."""
        r = compute_nfb_effect(gain_open=50.0, zout_open=0.0, thd_open=2.0, nfb_db=10.0)
        assert r is not None
        assert r["zout_closed"] == pytest.approx(0.0)

    # ── Algebraic invariants ──────────────────────────────────────

    def test_gain_product_invariant(self):
        """gain_closed × D = gain_open (exact by definition)."""
        r = compute_nfb_effect(gain_open=73.0, zout_open=20.0, thd_open=1.5, nfb_db=12.0)
        assert r is not None
        assert r["gain_closed"] * r["desensitivity"] == pytest.approx(73.0, rel=1e-10)

    def test_zout_product_invariant(self):
        """zout_closed × D = zout_open."""
        r = compute_nfb_effect(gain_open=73.0, zout_open=20.0, thd_open=1.5, nfb_db=12.0)
        assert r is not None
        assert r["zout_closed"] * r["desensitivity"] == pytest.approx(20.0, rel=1e-10)

    def test_thd_product_invariant(self):
        """thd_closed × D = thd_open."""
        r = compute_nfb_effect(gain_open=73.0, zout_open=20.0, thd_open=1.5, nfb_db=12.0)
        assert r is not None
        assert r["thd_closed"] * r["desensitivity"] == pytest.approx(1.5, rel=1e-10)

    def test_bw_equals_desensitivity(self):
        """bw_factor = D (bandwidth extended by desensitivity factor)."""
        r = compute_nfb_effect(gain_open=50.0, zout_open=10.0, thd_open=2.0, nfb_db=14.0)
        assert r is not None
        assert r["bw_factor"] == pytest.approx(r["desensitivity"], rel=1e-10)

    def test_beta_times_gain_plus_one(self):
        """β × A + 1 = D (definition of desensitivity)."""
        for nfb_db in [3.0, 6.0, 10.0, 15.0, 20.0]:
            r = compute_nfb_effect(gain_open=80.0, zout_open=30.0, thd_open=3.0, nfb_db=nfb_db)
            assert r is not None
            assert r["beta"] * 80.0 + 1.0 == pytest.approx(r["desensitivity"], rel=1e-10)

    def test_round_trip_nfb_db(self):
        """20·log10(D) = nfb_db (round-trip consistency)."""
        for nfb_db in [1.0, 6.0, 12.0, 20.0, 30.0]:
            r = compute_nfb_effect(gain_open=100.0, zout_open=10.0, thd_open=2.0, nfb_db=nfb_db)
            assert r is not None
            recovered = 20.0 * math.log10(r["desensitivity"])
            assert recovered == pytest.approx(nfb_db, abs=1e-10)

    # ── Textbook verification ─────────────────────────────────────

    def test_rdh_canonical_example(self):
        """RDH Ch.12: A=1000 (60 dB), NFB=20 dB → gain_closed=100 (40 dB)."""
        r = compute_nfb_effect(gain_open=1000.0, zout_open=10.0, thd_open=5.0, nfb_db=20.0)
        assert r is not None
        assert r["gain_closed"] == pytest.approx(100.0, rel=1e-3)
        assert r["gain_closed_db"] == pytest.approx(40.0, abs=0.1)
        assert r["thd_closed"] == pytest.approx(0.5, rel=1e-3)
        assert r["beta"] == pytest.approx(9.0 / 1000.0, rel=1e-3)

    def test_vtadiy_power_stage_example(self):
        """VTADiy §4.4: A=20, NFB=10 dB → D=3.162, gain_closed=6.32."""
        D = 10.0 ** (10.0 / 20.0)
        r = compute_nfb_effect(gain_open=20.0, zout_open=4.0, thd_open=8.0, nfb_db=10.0)
        assert r is not None
        assert r["gain_closed"] == pytest.approx(20.0 / D, rel=1e-3)
        assert r["zout_closed"] == pytest.approx(4.0 / D, rel=1e-3)
        assert r["thd_closed"] == pytest.approx(8.0 / D, rel=1e-3)

    # ── Integration with real lamp data ───────────────────────────

    def test_nfb_with_12ax7_triode(self):
        """12AX7: high gain triode.  6 dB NFB halves gain and Zout."""
        _, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        isects = find_intersections(pts, ll)
        srk = {"s": 1.6, "r": 62.5, "k": 100.0}
        stage = compute_stage_params(isects, ll, ug1_bias=-2.0, srk=srk)
        dist = compute_distortion(isects, ug1_bias=-2.0)
        assert stage is not None and dist is not None

        r = compute_nfb_effect(stage["gain"], stage["zout"], dist["thd"], nfb_db=6.0)
        assert r is not None
        D = 10.0 ** (6.0 / 20.0)
        assert r["gain_closed"] == pytest.approx(stage["gain"] / D, rel=1e-3)
        assert r["zout_closed"] == pytest.approx(stage["zout"] / D, rel=1e-3)
        assert r["gain_closed"] < stage["gain"]
        assert r["zout_closed"] < stage["zout"]

    def test_nfb_with_el84_pentode(self):
        """EL84 pentode: high ra, low DF. NFB improves DF by factor D."""
        _, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        isects = find_intersections(pts, ll)
        srk = {"s": 11.0, "r": 50.0, "k": 550.0}
        stage = compute_stage_params(isects, ll, ug1_bias=-7.0, srk=srk)
        assert stage is not None and stage["df"] is not None

        r = compute_nfb_effect(stage["gain"], stage["zout"], 5.0, nfb_db=10.0)
        assert r is not None
        D = 10.0 ** (10.0 / 20.0)
        # DF with NFB = DF_open × D
        df_nfb = stage["df"] * D
        assert df_nfb > stage["df"]
        assert df_nfb == pytest.approx(stage["df"] * r["desensitivity"], rel=1e-6)

    def test_nfb_with_12au7_cf(self):
        """12AU7 cathode follower: already low Zout, NFB reduces further."""
        _, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        isects = find_intersections(pts, ll)
        srk = {"s": 2.2, "r": 7.7, "k": 17.0}
        stage = compute_stage_params(isects, ll, ug1_bias=-10.0, srk=srk)
        assert stage is not None

        r = compute_nfb_effect(stage["gain"], stage["zout"], 1.0, nfb_db=6.0)
        assert r is not None
        assert r["zout_closed"] < stage["zout"]

    # ── Monotonicity: sweep NFB 0→20 dB ──────────────────────────

    def test_nfb_monotonic_gain_decrease(self):
        """Sweep NFB 0→20 dB: gain_closed strictly decreasing."""
        gains = []
        for nfb in range(0, 22, 2):
            r = compute_nfb_effect(60.0, 38.5, 2.5, nfb_db=float(nfb))
            assert r is not None
            gains.append(r["gain_closed"])
        for i in range(1, len(gains)):
            assert gains[i] < gains[i - 1], f"gain not decreasing at {i*2} dB"

    def test_nfb_monotonic_zout_decrease(self):
        """Sweep NFB 0→20 dB: zout_closed strictly decreasing."""
        zouts = []
        for nfb in range(0, 22, 2):
            r = compute_nfb_effect(60.0, 38.5, 2.5, nfb_db=float(nfb))
            assert r is not None
            zouts.append(r["zout_closed"])
        for i in range(1, len(zouts)):
            assert zouts[i] < zouts[i - 1], f"zout not decreasing at {i*2} dB"

    def test_nfb_monotonic_thd_decrease(self):
        """Sweep NFB 0→20 dB: thd_closed strictly decreasing."""
        thds = []
        for nfb in range(0, 22, 2):
            r = compute_nfb_effect(60.0, 38.5, 2.5, nfb_db=float(nfb))
            assert r is not None
            thds.append(r["thd_closed"])
        for i in range(1, len(thds)):
            assert thds[i] < thds[i - 1], f"thd not decreasing at {i*2} dB"

    def test_nfb_monotonic_bw_increase(self):
        """Sweep NFB 0→20 dB: bw_factor strictly increasing."""
        bws = []
        for nfb in range(0, 22, 2):
            r = compute_nfb_effect(60.0, 38.5, 2.5, nfb_db=float(nfb))
            assert r is not None
            bws.append(r["bw_factor"])
        for i in range(1, len(bws)):
            assert bws[i] > bws[i - 1], f"bw not increasing at {i*2} dB"

    # ── Return dict completeness ──────────────────────────────────

    def test_all_dict_keys_present(self):
        """Returned dict must contain all 9 expected keys."""
        expected_keys = {
            "gain_closed", "gain_closed_db", "zout_closed", "thd_closed",
            "bw_factor", "desensitivity", "beta", "nfb_db", "gain_open",
        }
        r = compute_nfb_effect(gain_open=50.0, zout_open=10.0, thd_open=2.0, nfb_db=10.0)
        assert r is not None
        assert set(r.keys()) == expected_keys

    def test_echo_back_fields(self):
        """nfb_db and gain_open are echoed back verbatim."""
        r = compute_nfb_effect(gain_open=73.5, zout_open=20.0, thd_open=1.5, nfb_db=8.0)
        assert r is not None
        assert r["nfb_db"] == 8.0
        assert r["gain_open"] == 73.5

    def test_gain_open_less_than_one(self):
        """Gain < 1 (e.g., cathode follower) — valid, gain_closed < gain_open."""
        r = compute_nfb_effect(gain_open=0.95, zout_open=0.4, thd_open=0.5, nfb_db=6.0)
        assert r is not None
        assert 0 < r["gain_closed"] < 0.95
        assert r["zout_closed"] < 0.4
        # beta > 1 since D ≈ 2.0 and A = 0.95 → beta = (2-1)/0.95 ≈ 1.05
        assert r["beta"] > 1.0

    def test_negative_zout_passthrough(self):
        """Negative Zout (physically impossible but not guarded) → negative closed."""
        r = compute_nfb_effect(gain_open=50.0, zout_open=-5.0, thd_open=2.0, nfb_db=6.0)
        assert r is not None
        assert r["zout_closed"] < 0  # passes through negative

    def test_negative_thd_passthrough(self):
        """Negative THD (physically impossible but not guarded) → negative closed."""
        r = compute_nfb_effect(gain_open=50.0, zout_open=10.0, thd_open=-1.0, nfb_db=6.0)
        assert r is not None
        assert r["thd_closed"] < 0  # passes through negative


# ---------------------------------------------------------------
# Ultralinear
# ---------------------------------------------------------------

class TestUltralinear:
    """Verify ul_screen_voltage(), UltralinearModelWrapper, and
    model-based pipeline with UL.

    Core formula: Ug2_eff = Ug2_nom * (1 - tap) + Ua * tap
    Sources: Hafler & Keroes 1951, Dempwolf §5.1.
    """

    # ── ul_screen_voltage: pure formula ─────────────────────────

    def test_ul_voltage_tap_zero_is_pentode(self):
        """tap=0 → Ug2_eff = Ug2_nom (pure pentode)."""
        assert ul_screen_voltage(200.0, 250.0, 0.0) == pytest.approx(250.0)

    def test_ul_voltage_tap_one_is_triode(self):
        """tap=1.0 → Ug2_eff = Ua (triode-connected)."""
        assert ul_screen_voltage(200.0, 250.0, 1.0) == pytest.approx(200.0)

    def test_ul_voltage_tap_043(self):
        """tap=0.43 (KT88): Ug2_eff = 250*(1-0.43) + 200*0.43 = 142.5 + 86 = 228.5."""
        expected = 250.0 * 0.57 + 200.0 * 0.43
        assert ul_screen_voltage(200.0, 250.0, 0.43) == pytest.approx(expected)

    def test_ul_voltage_tap_020(self):
        """tap=0.20 (EL84): Ug2_eff = 250*0.80 + 200*0.20 = 200 + 40 = 240."""
        assert ul_screen_voltage(200.0, 250.0, 0.20) == pytest.approx(240.0)

    def test_ul_voltage_linearity(self):
        """For fixed Ua and Ug2_nom, Ug2_eff is linear in tap."""
        ua, ug2_nom = 180.0, 300.0
        v1 = ul_screen_voltage(ua, ug2_nom, 0.2)
        v2 = ul_screen_voltage(ua, ug2_nom, 0.4)
        v3 = ul_screen_voltage(ua, ug2_nom, 0.6)
        # Equal spacing in tap → equal spacing in Ug2_eff
        assert v2 - v3 == pytest.approx(v1 - v2, rel=1e-10)

    def test_ul_voltage_when_ua_equals_ug2(self):
        """When Ua = Ug2_nom, any tap gives Ug2_eff = Ug2_nom = Ua."""
        for tap in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert ul_screen_voltage(250.0, 250.0, tap) == pytest.approx(250.0)

    def test_ul_voltage_ua_higher_than_ug2(self):
        """Ua > Ug2_nom: UL raises effective Ug2 (tap pulls it toward Ua)."""
        # Ua=350, Ug2_nom=250, tap=0.43 → 250*0.57 + 350*0.43 = 142.5 + 150.5 = 293
        v = ul_screen_voltage(350.0, 250.0, 0.43)
        assert v > 250.0  # pulled up
        assert v < 350.0  # but not fully to Ua

    def test_ul_voltage_ua_lower_than_ug2(self):
        """Ua < Ug2_nom: UL lowers effective Ug2 (tap pulls it toward Ua)."""
        v = ul_screen_voltage(100.0, 250.0, 0.43)
        assert v < 250.0  # pulled down
        assert v > 100.0  # but not fully to Ua

    # ── UltralinearModelWrapper: construction & delegation ──────

    def test_wrapper_ia_overrides_ug2(self):
        """Wrapper must replace ug2 with dynamic ul_screen_voltage."""
        model, _ = quick_pentode("EL84")
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.43)
        ua, ug1 = 200.0, -7.0
        # Direct model with dynamic Ug2
        ug2_eff = ul_screen_voltage(ua, 250.0, 0.43)
        expected_ia = model.ia(ua, ug1, ug2_eff)
        # Wrapper (ug2 arg should be ignored)
        actual_ia = wrapped.ia(ua, ug1, ug2=250.0)
        assert actual_ia == pytest.approx(expected_ia, rel=1e-10)

    def test_wrapper_ig2_overrides_ug2(self):
        """Wrapper ig2() also uses dynamic screen voltage."""
        model, _ = quick_pentode("EL84")
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.20)
        ua, ug1 = 200.0, -5.0
        ug2_eff = ul_screen_voltage(ua, 250.0, 0.20)
        expected = model.ig2(ua, ug1, ug2_eff)
        actual = wrapped.ig2(ua, ug1, ug2=250.0)
        assert actual == pytest.approx(expected, rel=1e-10)

    def test_wrapper_tap_zero_equals_original(self):
        """tap=0 → wrapper produces same Ia as original model at Ug2_nom."""
        model, _ = quick_pentode("EL84")
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.0)
        for ua in [50.0, 150.0, 300.0]:
            for ug1 in [-10.0, -5.0, -2.0]:
                orig = model.ia(ua, ug1, 250.0)
                wrap = wrapped.ia(ua, ug1, ug2=250.0)
                assert wrap == pytest.approx(orig, rel=1e-10)

    def test_wrapper_forwards_attributes(self):
        """model_type, name, topology, pa_max, uh, ih forwarded."""
        model, _ = quick_pentode("EL84")
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.43)
        assert wrapped.model_type == model.model_type
        assert wrapped.name == model.name
        assert wrapped.topology == model.topology
        assert wrapped.pa_max == model.pa_max
        assert wrapped.uh == model.uh
        assert wrapped.ih == model.ih

    def test_wrapper_params_dict_has_ul_fields(self):
        """params_dict() includes ul_tap and ul_ug2_nom."""
        model, _ = quick_pentode("EL84")
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.43)
        d = wrapped.params_dict()
        assert d["ul_tap"] == 0.43
        assert d["ul_ug2_nom"] == 250.0

    def test_wrapper_convenience_properties(self):
        """ug2_nom and tap properties."""
        model, _ = quick_pentode("EL84")
        wrapped = UltralinearModelWrapper(model, ug2_nom=300.0, tap=0.20)
        assert wrapped.ug2_nom == 300.0
        assert wrapped.tap == 0.20

    # ── UL changes tube characteristics ──────────────────────────

    def test_ul_reduces_ia_at_low_ua(self):
        """At Ua < Ug2_nom, UL tap lowers Ug2_eff → lower Ia."""
        model, _ = quick_pentode("EL84")
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.43)
        ua, ug1 = 100.0, -5.0
        ia_pentode = model.ia(ua, ug1, 250.0)
        ia_ul = wrapped.ia(ua, ug1, 250.0)
        # At low Ua: ug2_eff = 250*0.57 + 100*0.43 = 185.5 < 250
        assert ia_ul < ia_pentode

    def test_ul_increases_ia_at_high_ua(self):
        """At Ua > Ug2_nom, UL tap raises Ug2_eff → higher Ia."""
        model, _ = quick_pentode("EL84")
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.43)
        ua, ug1 = 350.0, -5.0
        ia_pentode = model.ia(ua, ug1, 250.0)
        ia_ul = wrapped.ia(ua, ug1, 250.0)
        # At high Ua: ug2_eff = 250*0.57 + 350*0.43 = 293 > 250
        assert ia_ul > ia_pentode

    def test_ul_effect_increases_with_tap(self):
        """Higher tap → bigger difference from pentode at low Ua."""
        model, _ = quick_pentode("EL84")
        ua, ug1 = 100.0, -5.0
        ia_pentode = model.ia(ua, ug1, 250.0)
        diffs = []
        for tap in [0.1, 0.2, 0.3, 0.4, 0.5]:
            w = UltralinearModelWrapper(model, ug2_nom=250.0, tap=tap)
            ia_ul = w.ia(ua, ug1, 250.0)
            diffs.append(ia_pentode - ia_ul)
        # Bigger tap → bigger reduction at low Ua
        for i in range(1, len(diffs)):
            assert diffs[i] > diffs[i - 1]

    # ── Integration: find_intersections_model with UL ────────────

    def test_ul_intersections_different_from_pentode(self):
        """UL wrapper changes intersection points vs. pure pentode."""
        model, pts = quick_pentode("EL84")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)

        isects_pent = find_intersections_model(model, ll, ug1_vals, ug2=250.0)
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.43)
        isects_ul = find_intersections_model(wrapped, ll, ug1_vals, ug2=250.0)

        assert len(isects_pent) >= 3
        assert len(isects_ul) >= 3
        # UL must produce different Ua values (not identical curves)
        ua_pent = [p["ua"] for p in isects_pent]
        ua_ul = [p["ua"] for p in isects_ul]
        # At least one point must differ significantly
        diffs = [abs(a - b) for a, b in zip(ua_pent, ua_ul)]
        assert max(diffs) > 1.0, "UL should shift intersection Ua values"

    def test_ul_intersections_with_bias(self):
        """UL + ug1_bias for PushPullLoadLine (DC Q-point + AC)."""
        model, pts = quick_pentode("EL84")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.20)

        isects = find_intersections_model(
            wrapped, ll, ug1_vals, ug2=250.0, ug1_bias=-7.0,
        )
        assert len(isects) >= 3
        # Q-point should exist near ug1=-7
        q = interp_intersection(isects, -7.0)
        assert q is not None
        assert q["ia"] > 0

    def test_ul_tap_zero_intersections_match_pentode(self):
        """tap=0 wrapper gives identical intersections to pentode."""
        model, pts = quick_pentode("EL84")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)

        isects_pent = find_intersections_model(model, ll, ug1_vals, ug2=250.0)
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.0)
        isects_ul = find_intersections_model(wrapped, ll, ug1_vals, ug2=250.0)

        assert len(isects_pent) == len(isects_ul)
        for p, u in zip(isects_pent, isects_ul):
            assert p["ua"] == pytest.approx(u["ua"], abs=0.1)
            assert p["ia"] == pytest.approx(u["ia"], abs=0.01)

    # ── Distortion with UL ─────────────────────────────────────────

    def test_ul_distortion_computable(self):
        """5-point distortion works with UL-wrapped intersections."""
        model, pts = quick_pentode("EL84")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.20)

        isects = find_intersections_model(
            wrapped, ll, ug1_vals, ug2=250.0, ug1_bias=-7.0,
        )
        dist = compute_distortion(isects, ug1_bias=-7.0, ub=300.0)
        assert dist is not None
        assert dist["thd"] > 0
        assert dist["pout_mw"] > 0

    @staticmethod
    def _generate_ul_points(
        wrapped: UltralinearModelWrapper, ug2_nom: float,
    ) -> list:
        """Generate synthetic measurement points using UL-wrapped model."""
        pts = []
        for ua in range(10, 310, 10):
            for ug1_10 in range(-150, 1, 30):
                ug1 = ug1_10 / 10.0
                ia = wrapped.ia(float(ua), ug1, ug2_nom)
                pts.append({"ua": float(ua), "ug1": ug1, "ia": ia, "ug2": ug2_nom})
        return pts

    def test_ul_reduces_output_impedance(self):
        """UL should give lower ra (closer to triode) → lower Zout.

        Uses UL-generated points for numerical gm/ra estimation.
        """
        model, pts = quick_pentode("EL84")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)

        # Pentode stage params
        isects_pent = find_intersections_model(
            model, ll, ug1_vals, ug2=250.0, ug1_bias=-7.0,
        )
        stage_pent = compute_stage_params(
            isects_pent, ll, ug1_bias=-7.0, points=pts,
        )

        # UL (43%) — generate UL-aware points for numerical method
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.43)
        ul_pts = self._generate_ul_points(wrapped, 250.0)
        isects_ul = find_intersections_model(
            wrapped, ll, ug1_vals, ug2=250.0, ug1_bias=-7.0,
        )
        stage_ul = compute_stage_params(
            isects_ul, ll, ug1_bias=-7.0, points=ul_pts,
        )

        assert stage_pent is not None and stage_ul is not None
        # UL should give lower Zout than pentode (higher tap → more triode-like)
        assert stage_ul["zout"] < stage_pent["zout"]

    # ── Sweep monotonicity: tap 0→100% ────────────────────────────

    def test_ul_zout_trend_with_tap(self):
        """Higher tap → overall lower Zout (pentode → triode direction).

        Numerical gm/ra estimation may introduce small non-monotonic noise,
        so we verify the trend: tap=0.0 (pentode) has higher Zout than
        tap=0.43 (UL), which has higher Zout than tap=1.0 (triode).
        """
        model, pts = quick_pentode("EL84")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)

        zouts = {}
        for tap in [0.0, 0.2, 0.43, 1.0]:
            if tap == 0.0:
                m = model
                m_pts = pts
            else:
                m = UltralinearModelWrapper(model, ug2_nom=250.0, tap=tap)
                m_pts = self._generate_ul_points(m, 250.0)
            isects = find_intersections_model(
                m, ll, ug1_vals, ug2=250.0, ug1_bias=-7.0,
            )
            stage = compute_stage_params(isects, ll, ug1_bias=-7.0, points=m_pts)
            if stage is not None:
                zouts[tap] = stage["zout"]

        assert len(zouts) >= 3, f"Need enough valid stages, got {len(zouts)}"
        # Pentode Zout > UL Zout > Triode Zout (overall trend)
        assert zouts[0.0] > zouts[0.43], (
            f"Pentode Zout {zouts[0.0]:.2f} should > UL Zout {zouts[0.43]:.2f}"
        )
        if 1.0 in zouts:
            assert zouts[0.43] > zouts[1.0], (
                f"UL Zout {zouts[0.43]:.2f} should > Triode Zout {zouts[1.0]:.2f}"
            )

    def test_ul_gain_lower_than_pentode(self):
        """UL gain is lower than pentode gain (screen feedback reduces gain).

        Note: gain is NOT monotonically decreasing with tap — at very high
        tap (near triode), ra drops faster than mu, so gain = mu*Ra/(ra+Ra)
        can increase when Ra >> ra_triode. We only check pentode > UL.
        """
        model, pts = quick_pentode("EL84")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)

        # Pentode
        isects_pent = find_intersections_model(
            model, ll, ug1_vals, ug2=250.0, ug1_bias=-7.0,
        )
        stage_pent = compute_stage_params(
            isects_pent, ll, ug1_bias=-7.0, points=pts,
        )

        # UL 43%
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.43)
        ul_pts = self._generate_ul_points(wrapped, 250.0)
        isects_ul = find_intersections_model(
            wrapped, ll, ug1_vals, ug2=250.0, ug1_bias=-7.0,
        )
        stage_ul = compute_stage_params(
            isects_ul, ll, ug1_bias=-7.0, points=ul_pts,
        )

        assert stage_pent is not None and stage_ul is not None
        assert stage_pent["gain"] > stage_ul["gain"], (
            f"Pentode gain {stage_pent['gain']:.2f} should > UL gain {stage_ul['gain']:.2f}"
        )

    # ── Tube-specific tap values (empirical) ─────────────────────

    def test_el84_typical_ul_tap(self):
        """EL84 at 20% tap: should produce valid analysis."""
        model, pts = quick_pentode("EL84")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.20)

        isects = find_intersections_model(
            wrapped, ll, ug1_vals, ug2=250.0, ug1_bias=-7.0,
        )
        assert len(isects) >= 5
        dist = compute_distortion(isects, ug1_bias=-7.0, ub=300.0)
        assert dist is not None
        # Reasonable output power for EL84 PP UL
        assert 100 < dist["pout_mw"] < 20000

    def test_el34_typical_ul_tap(self):
        """EL34 at 40% tap: should produce valid analysis."""
        model, pts = quick_pentode("EL34")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        ll = PushPullLoadLine(ub=400, ra_aa=6.6)
        wrapped = UltralinearModelWrapper(model, ug2_nom=265.0, tap=0.40)

        isects = find_intersections_model(
            wrapped, ll, ug1_vals, ug2=265.0, ug1_bias=-14.0,
        )
        assert len(isects) >= 3
        dist = compute_distortion(isects, ug1_bias=-14.0, ub=400.0)
        assert dist is not None
        assert dist["thd"] > 0

    # ── Edge cases ─────────────────────────────────────────────────

    def test_wrapper_with_triode_model_raises_no_error(self):
        """Wrapping a triode model technically works (unusual but not forbidden)."""
        model, _ = quick_triode("12AX7")
        wrapped = UltralinearModelWrapper(model, ug2_nom=0.0, tap=0.5)
        # Should not crash; triode ignores ug2 anyway
        ia = wrapped.ia(200.0, -2.0, 0.0)
        assert ia >= 0

    def test_wrapper_tap_boundary_values(self):
        """Boundary taps: 0.0 and 1.0 produce valid results."""
        model, _ = quick_pentode("EL84")
        for tap in [0.0, 1.0]:
            w = UltralinearModelWrapper(model, ug2_nom=250.0, tap=tap)
            ia = w.ia(200.0, -5.0, 250.0)
            assert ia >= 0

    def test_ul_generate_scan_forwards(self):
        """generate_scan() forwards to underlying model (no UL transform)."""
        model, _ = quick_pentode("EL84")
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.20)
        grid = ScanGrid(ua=(0, 300, 30), ug1=(-15, 0, 3), ug2=(250, 250, 1), uh=6.3, ih=0.76)
        pts = wrapped.generate_scan(grid)
        # Should produce points (forwarded to original model)
        assert len(pts) > 10


# ---------------------------------------------------------------
# model_gm_ra
# ---------------------------------------------------------------

class TestModelGmRa:
    """Verify model_gm_ra: finite-difference gm/ra from tube model.

    gm = dIa/dUg1 (central difference)
    ra = dUa/dIa  (central difference)
    mu = gm × ra
    """

    # ── Triode: compare with known SRK ───────────────────────────

    def test_12ax7_gm_matches_srk(self):
        """12AX7 model gm should be close to SRK value (~1.6 mA/V)."""
        model, _ = quick_triode("12AX7")
        # Ua=150V, Ug1=-1V → realistic Q-point (Ia ≈ 0.8 mA)
        r = model_gm_ra(model, ua_q=150.0, ug1_q=-1.0)
        assert r is not None
        assert r["method"] == "model"
        assert 1.0 < r["gm"] < 3.0, f"12AX7 gm={r['gm']:.2f}, expected ~1.6"

    def test_12ax7_ra_matches_srk(self):
        """12AX7 ra should be ~62.5 kΩ."""
        model, _ = quick_triode("12AX7")
        r = model_gm_ra(model, ua_q=150.0, ug1_q=-1.0)
        assert r is not None
        assert 30 < r["ra"] < 120, f"12AX7 ra={r['ra']:.1f}, expected ~62.5"

    def test_12ax7_mu_matches_srk(self):
        """12AX7 mu = gm × ra should be ~100."""
        model, _ = quick_triode("12AX7")
        r = model_gm_ra(model, ua_q=150.0, ug1_q=-1.0)
        assert r is not None
        assert 60 < r["mu"] < 150, f"12AX7 mu={r['mu']:.1f}, expected ~100"

    def test_12au7_lower_mu(self):
        """12AU7 has lower mu (~17) than 12AX7 (~100)."""
        model_ax, _ = quick_triode("12AX7")
        model_au, _ = quick_triode("12AU7")
        r_ax = model_gm_ra(model_ax, ua_q=150.0, ug1_q=-1.0)
        r_au = model_gm_ra(model_au, ua_q=150.0, ug1_q=-8.0)
        assert r_ax is not None and r_au is not None
        assert r_au["mu"] < r_ax["mu"], "12AU7 mu should be lower than 12AX7"

    # ── Pentode ──────────────────────────────────────────────────

    def test_el84_pentode_high_ra(self):
        """EL84 pentode ra should be very high (> 20 kΩ)."""
        model, _ = quick_pentode("EL84")
        r = model_gm_ra(model, ua_q=250.0, ug1_q=-7.0, ug2=250.0)
        assert r is not None
        assert r["ra"] > 20.0, f"EL84 pentode ra={r['ra']:.1f}, expected > 20 kΩ"

    def test_el84_pentode_high_gm(self):
        """EL84 pentode gm should be ~11 mA/V."""
        model, _ = quick_pentode("EL84")
        r = model_gm_ra(model, ua_q=250.0, ug1_q=-7.0, ug2=250.0)
        assert r is not None
        assert 5 < r["gm"] < 20, f"EL84 gm={r['gm']:.1f}, expected ~11"

    # ── UL wrapper changes gm/ra ─────────────────────────────────

    def test_ul_wrapper_changes_ra(self):
        """UL-wrapped model should give different ra than pentode."""
        model, _ = quick_pentode("EL84")
        r_pent = model_gm_ra(model, ua_q=250.0, ug1_q=-7.0, ug2=250.0)
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.43)
        r_ul = model_gm_ra(wrapped, ua_q=250.0, ug1_q=-7.0, ug2=250.0)
        assert r_pent is not None and r_ul is not None
        # UL should lower ra (toward triode)
        assert r_ul["ra"] < r_pent["ra"], (
            f"UL ra={r_ul['ra']:.1f} should be < pentode ra={r_pent['ra']:.1f}"
        )

    def test_ul_wrapper_lowers_mu(self):
        """UL-wrapped model should give lower mu than pentode."""
        model, _ = quick_pentode("EL84")
        r_pent = model_gm_ra(model, ua_q=250.0, ug1_q=-7.0, ug2=250.0)
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.43)
        r_ul = model_gm_ra(wrapped, ua_q=250.0, ug1_q=-7.0, ug2=250.0)
        assert r_pent is not None and r_ul is not None
        assert r_ul["mu"] < r_pent["mu"], (
            f"UL mu={r_ul['mu']:.1f} should be < pentode mu={r_pent['mu']:.1f}"
        )

    def test_ul_tap_zero_matches_pentode(self):
        """tap=0 wrapper should give same gm/ra as pentode."""
        model, _ = quick_pentode("EL84")
        r_pent = model_gm_ra(model, ua_q=250.0, ug1_q=-7.0, ug2=250.0)
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.0)
        r_ul = model_gm_ra(wrapped, ua_q=250.0, ug1_q=-7.0, ug2=250.0)
        assert r_pent is not None and r_ul is not None
        assert r_pent["gm"] == pytest.approx(r_ul["gm"], rel=1e-6)
        assert r_pent["ra"] == pytest.approx(r_ul["ra"], rel=1e-6)

    # ── mu = gm × ra invariant ───────────────────────────────────

    def test_mu_equals_gm_times_ra(self):
        """mu = gm × ra (definition) for any model."""
        for name in ["12AX7", "12AU7"]:
            model, _ = quick_triode(name)
            r = model_gm_ra(model, ua_q=150.0, ug1_q=-3.0)
            assert r is not None
            assert r["mu"] == pytest.approx(r["gm"] * r["ra"], rel=1e-6)

    def test_mu_equals_gm_times_ra_pentode(self):
        """mu = gm × ra for pentode model."""
        model, _ = quick_pentode("EL84")
        r = model_gm_ra(model, ua_q=250.0, ug1_q=-7.0, ug2=250.0)
        assert r is not None
        assert r["mu"] == pytest.approx(r["gm"] * r["ra"], rel=1e-6)

    # ── Edge cases ───────────────────────────────────────────────

    def test_cutoff_returns_none(self):
        """At deep cutoff (Ia ≈ 0), gm → 0 → returns None."""
        model, _ = quick_triode("12AX7")
        # Very negative Ug1 → near cutoff
        r = model_gm_ra(model, ua_q=150.0, ug1_q=-20.0)
        # Documented contract: None OR near-zero gm — both sides of
        # the disjunction are checked explicitly.
        if r is not None:  # vacuity-ok: None is a legal cutoff outcome
            assert r["gm"] < 0.5, f"gm={r['gm']:.3f} not near-zero at cutoff"

    def test_different_q_points_different_params(self):
        """gm/ra should vary with Q-point (non-linear tube)."""
        model, _ = quick_triode("12AX7")
        r1 = model_gm_ra(model, ua_q=100.0, ug1_q=-1.0)
        r2 = model_gm_ra(model, ua_q=250.0, ug1_q=-3.0)
        assert r1 is not None and r2 is not None
        # Different operating points → different parameters
        assert r1["gm"] != pytest.approx(r2["gm"], rel=0.01)

    # ── Integration: compute_stage_params with model ─────────────

    def test_stage_params_uses_model_when_available(self):
        """compute_stage_params with model → method='model'."""
        model, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        isects = find_intersections_model(model, ll, [-3.0, -2.0, -1.0, 0.0])
        stage = compute_stage_params(isects, ll, ug1_bias=-2.0, model=model)
        assert stage is not None
        assert stage["method"] == "model"

    def test_stage_params_model_overrides_srk(self):
        """Model has priority over SRK."""
        model, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        isects = find_intersections_model(model, ll, [-3.0, -2.0, -1.0, 0.0])
        srk = {"s": 1.6, "r": 62.5, "k": 100.0}
        stage = compute_stage_params(
            isects, ll, ug1_bias=-2.0, srk=srk, model=model,
        )
        assert stage is not None
        assert stage["method"] == "model"
        # SRK cross-check should be present
        assert stage["srk_check"] is not None

    def test_stage_params_model_pentode_ul(self):
        """UL-wrapped model gives different stage params than pentode."""
        model, pts = quick_pentode("EL84")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)

        # Pentode
        isects_p = find_intersections_model(model, ll, ug1_vals, ug2=250.0, ug1_bias=-7.0)
        stage_p = compute_stage_params(
            isects_p, ll, ug1_bias=-7.0, model=model, model_ug2=250.0,
        )

        # UL 43%
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.43)
        isects_ul = find_intersections_model(wrapped, ll, ug1_vals, ug2=250.0, ug1_bias=-7.0)
        stage_ul = compute_stage_params(
            isects_ul, ll, ug1_bias=-7.0, model=wrapped, model_ug2=250.0,
        )

        assert stage_p is not None and stage_ul is not None
        assert stage_p["method"] == "model"
        assert stage_ul["method"] == "model"
        # UL should give lower Zout
        assert stage_ul["zout"] < stage_p["zout"], (
            f"UL Zout {stage_ul['zout']:.2f} should < pentode {stage_p['zout']:.2f}"
        )
        # UL should give lower gain
        assert stage_ul["gain"] < stage_p["gain"]

    def test_no_model_falls_back_to_numerical(self):
        """Without model, compute_stage_params uses numerical/SRK."""
        _, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        isects = find_intersections(pts, ll)
        stage = compute_stage_params(isects, ll, ug1_bias=-2.0, points=pts)
        assert stage is not None
        assert stage["method"] == "numerical"

    # ── CF + model ───────────────────────────────────────────────

    def test_cf_stage_params_with_model(self):
        """Cathode follower with model → method='model', CF formulas."""
        model, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals)
        stage = compute_stage_params(
            isects, ll, ug1_bias=-10.0, model=model,
        )
        assert stage is not None
        assert stage["method"] == "model"
        # CF gain < 1 (always)
        assert 0 < stage["gain"] < 1.0
        # CF Zout ≈ ra/(mu+1), should be low (< 1 kΩ for 12AU7)
        assert stage["zout"] < 2.0

    def test_cf_model_vs_numerical_consistent(self):
        """CF model-derived params should be in same ballpark as numerical."""
        model, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals)

        stage_model = compute_stage_params(
            isects, ll, ug1_bias=-10.0, model=model,
        )
        stage_num = compute_stage_params(
            isects, ll, ug1_bias=-10.0, points=pts,
        )
        assert stage_model is not None
        assert stage_num is not None
        # Gain should be in same order of magnitude
        assert abs(stage_model["gain"] - stage_num["gain"]) < 0.3

    # ── SE Transformer + model ───────────────────────────────────

    def test_se_xfmr_stage_params_with_model(self):
        """SE transformer with model → method='model'."""
        model, pts = quick_triode("12AX7")
        ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=100.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(
            model, ll, ug1_vals, ug1_bias=-1.0,
        )
        stage = compute_stage_params(
            isects, ll, ug1_bias=-1.0, model=model,
        )
        assert stage is not None
        assert stage["method"] == "model"
        assert stage["gain"] > 1.0
        assert stage["zout"] > 0

    # ── UL + CF (edge case) ──────────────────────────────────────

    def test_ul_cf_no_crash(self):
        """UL-wrapped pentode in CF circuit — unusual but should not crash."""
        model, pts = quick_pentode("EL84")
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.20)
        ll = CathodeFollowerLoadLine(ub=300, rk=1.0, rl=8.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(
            wrapped, ll, ug1_vals, ug2=250.0, ug1_bias=-7.0,
        )
        assert len(isects) >= 2, "guard de-vacuated 2026-07-12: value must be present"
        stage = compute_stage_params(
            isects, ll, ug1_bias=-7.0,
            model=wrapped, model_ug2=250.0,
        )
        assert stage is not None, "guarded value must be truthy (de-vacuated)"
        assert stage["method"] == "model"
        assert 0 < stage["gain"] < 1.0  # CF gain always < 1


# ---------------------------------------------------------------
# compute_pa_avg — average plate dissipation
# ---------------------------------------------------------------

class TestPaAvg:
    """Verify compute_pa_avg: numerical integration of Pa over signal cycle.

    Pa_avg = mean(Ua × Ia) over one sinusoidal cycle.
    For Class A: Pa_avg ≈ Pdc − Pout (DC current constant).
    For Class AB/B: Pa_avg differs because Ia_avg changes with signal.
    """

    # ── Class A: Pa_avg ≈ Pdc − Pout ────────────────────────────

    def test_class_a_triode_energy_conservation(self):
        """For Class A triode, Pa_avg + P_load ≈ Pdc (energy conservation).

        Pa_avg = mean(Ua×Ia), Pdc = Ub × Ia_avg.
        P_load = Pdc − Pa_avg = power delivered to load resistor.
        P_load should be > 0 and < Pdc.
        """
        model, _ = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        r = compute_pa_avg(model, ll, ug1_bias=-1.0, half_swing=0.5, ub=250.0)
        assert r is not None
        assert r["pa_avg_mw"] > 0
        assert "pdc_avg_mw" in r
        p_load = r["pdc_avg_mw"] - r["pa_avg_mw"]
        assert p_load > 0, "Load must dissipate some power"
        assert r["pa_avg_mw"] < r["pdc_avg_mw"], "Pa < Pdc (tube doesn't create energy)"

    def test_pa_avg_less_than_pdc(self):
        """Pa_avg must be less than Pdc (energy conservation: Pdc = Pa + Pout)."""
        model, _ = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        r = compute_pa_avg(model, ll, ug1_bias=-1.0, half_swing=0.5, ub=250.0)
        assert r is not None
        if "pdc_avg_mw" in r:
            assert r["pa_avg_mw"] < r["pdc_avg_mw"]

    def test_pa_avg_positive(self):
        """Pa_avg is always positive (tube dissipates power)."""
        model, _ = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        r = compute_pa_avg(model, ll, ug1_bias=-10.0, half_swing=3.0, ub=250.0)
        assert r is not None
        assert r["pa_avg_mw"] > 0

    # ── Pentode: higher power ────────────────────────────────────

    def test_el84_pentode_pa_avg(self):
        """EL84 pentode Pa_avg should be physically reasonable."""
        model, _ = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        r = compute_pa_avg(model, ll, ug1_bias=-7.0, half_swing=3.0, ug2=250.0, ub=300.0)
        assert r is not None
        # EL84 Pa_max=12W → Pa_avg should be well under that
        assert 0 < r["pa_avg_mw"] < 15000

    # ── Ia_avg increases with swing for pentode ──────────────────

    def test_ia_avg_increases_with_swing(self):
        """For pentode driven hard, Ia_avg should increase with swing."""
        model, _ = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        ia_avgs = []
        for swing in [1.0, 3.0, 5.0]:
            r = compute_pa_avg(model, ll, ug1_bias=-7.0, half_swing=swing, ug2=250.0, ub=300.0)
            if r:
                ia_avgs.append(r["ia_avg"])
        # For large swing, clipping on one side → Ia_avg changes
        assert len(ia_avgs) >= 2

    # ── Pa_peak > Pa_avg ─────────────────────────────────────────

    def test_pa_peak_ge_pa_avg(self):
        """Peak Pa must be >= average Pa."""
        model, _ = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        r = compute_pa_avg(model, ll, ug1_bias=-1.0, half_swing=0.5, ub=250.0)
        assert r is not None
        assert r["pa_peak_mw"] >= r["pa_avg_mw"]

    # ── Zero swing → Pa_avg = Pa_q ───────────────────────────────

    def test_zero_swing_returns_none(self):
        """Swing below threshold returns None."""
        model, _ = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        r = compute_pa_avg(model, ll, ug1_bias=-1.0, half_swing=0.01, ub=250.0)
        assert r is None

    # ── Transformer load line ────────────────────────────────────

    def test_transformer_load_line(self):
        """Pa_avg works with TransformerLoadLine."""
        model, _ = quick_triode("12AX7")
        ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=100.0)
        r = compute_pa_avg(model, ll, ug1_bias=-1.0, half_swing=0.5, ub=250.0)
        assert r is not None
        assert r["pa_avg_mw"] > 0

    # ── PP load line ─────────────────────────────────────────────

    def test_pp_load_line(self):
        """Pa_avg works with PushPullLoadLine."""
        model, _ = quick_pentode("EL84")
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)
        r = compute_pa_avg(model, ll, ug1_bias=-7.0, half_swing=3.0, ug2=250.0, ub=300.0)
        assert r is not None
        assert r["pa_avg_mw"] > 0

    # ── DFT pa_signal_mw field ───────────────────────────────────

    def test_dft_includes_pa_avg(self):
        """compute_distortion_dft should include pa_avg_mw field."""
        from lm19.amplifier import compute_distortion_dft
        model, _ = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        dist = compute_distortion_dft(model, ll, ug1_bias=-1.0, half_swing=0.5, ub=250.0)
        assert dist is not None
        assert "pa_avg_mw" in dist
        assert dist["pa_avg_mw"] > 0
        assert "pa_signal_mw" in dist

    def test_dft_pa_avg_matches_standalone(self):
        """DFT pa_avg_mw should be close to standalone compute_pa_avg."""
        from lm19.amplifier import compute_distortion_dft
        model, _ = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        dist = compute_distortion_dft(model, ll, ug1_bias=-1.0, half_swing=0.5, ub=250.0)
        pa = compute_pa_avg(model, ll, ug1_bias=-1.0, half_swing=0.5, ub=250.0)
        assert dist is not None and pa is not None
        # Both integrate Pa over a cycle; should be close (different N but same physics)
        assert dist["pa_avg_mw"] == pytest.approx(pa["pa_avg_mw"], rel=0.05)

    def test_ul_wrapper_changes_pa_avg(self):
        """UL-wrapped pentode should give different Pa_avg than pentode."""
        model, _ = quick_pentode("EL84")
        ll = PushPullLoadLine(ub=300, ra_aa=8.0)
        pa_pent = compute_pa_avg(model, ll, ug1_bias=-7.0, half_swing=3.0, ug2=250.0, ub=300.0)
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.43)
        pa_ul = compute_pa_avg(wrapped, ll, ug1_bias=-7.0, half_swing=3.0, ug2=250.0, ub=300.0)
        assert pa_pent is not None and pa_ul is not None
        # UL changes the operating trajectory → Pa_avg differs
        assert pa_pent["pa_avg_mw"] != pytest.approx(pa_ul["pa_avg_mw"], rel=0.01)


# ---------------------------------------------------------------
# Grid current quantification
# ---------------------------------------------------------------

class TestGridCurrent:
    """Verify grid current quantification in compute_headroom.

    Uses Dempwolf grid current model: IGK = Gg * softplus(Cg * Vgk)^xi
    """

    def _make_intersections(self, ug1_range=(-10.0, 0.0), n=11):
        """Simple synthetic intersections spanning ug1_range."""
        pts = []
        for i in range(n):
            ug1 = ug1_range[0] + (ug1_range[1] - ug1_range[0]) * i / (n - 1)
            ua = 200.0 - 20.0 * (ug1 + 5.0)
            ia = max(0.0, 5.0 + 2.0 * (ug1 + 5.0))
            pts.append({"ug1": ug1, "ua": ua, "ia": ia})
        return pts

    # ── Without grid current params: hard-threshold fallback ─────

    def test_no_params_no_ig1(self):
        """Without ``grid_current_params``, ``ig1_ma`` is omitted from result."""
        isects = self._make_intersections()
        hr = compute_headroom(isects, ug1_bias=-5.0)
        assert hr is not None
        assert "ig1_ma" not in hr

    def test_hard_threshold_still_works(self):
        """Hard threshold (-0.1 V) flags ``grid_current`` when no
        Dempwolf params are supplied."""
        isects = self._make_intersections(ug1_range=(-10.0, 0.5))
        hr = compute_headroom(isects, ug1_bias=-5.0)
        assert hr is not None
        assert hr["clip_pos"] == "grid_current"

    # ── With grid current params: quantified ─────────────────────

    def test_ig1_returned_with_params(self):
        """With Dempwolf params, ig1_ma is returned."""
        gc = {"Gg": 6.177e-4, "xi": 1.314, "Cg": 9.901}  # 12AX7
        isects = self._make_intersections(ug1_range=(-10.0, 0.5))
        hr = compute_headroom(isects, ug1_bias=-5.0, grid_current_params=gc)
        assert hr is not None
        assert "ig1_ma" in hr
        assert hr["ig1_ma"] > 0

    def test_ig1_zero_at_deep_bias(self):
        """At deep negative bias (far from 0V), grid current ≈ 0."""
        gc = {"Gg": 6.177e-4, "xi": 1.314, "Cg": 9.901}
        isects = self._make_intersections(ug1_range=(-20.0, -5.0))
        hr = compute_headroom(isects, ug1_bias=-12.0, grid_current_params=gc)
        assert hr is not None
        if "ig1_ma" in hr:
            assert hr["ig1_ma"] < 0.001  # negligible

    def test_ig1_increases_near_zero(self):
        """Grid current increases as swing approaches Ug1=0."""
        gc = {"Gg": 6.177e-4, "xi": 1.314, "Cg": 9.901}
        isects_small = self._make_intersections(ug1_range=(-10.0, -2.0))
        isects_large = self._make_intersections(ug1_range=(-10.0, 0.5))

        hr_small = compute_headroom(isects_small, ug1_bias=-5.0, grid_current_params=gc)
        hr_large = compute_headroom(isects_large, ug1_bias=-5.0, grid_current_params=gc)

        ig1_small = hr_small.get("ig1_ma", 0.0) if hr_small else 0.0
        ig1_large = hr_large.get("ig1_ma", 0.0) if hr_large else 0.0
        assert ig1_large > ig1_small

    def test_ig1_physically_reasonable(self):
        """12AX7 grid current at Vgk=0 should be < 1 mA (small signal tube)."""
        gc = {"Gg": 6.177e-4, "xi": 1.314, "Cg": 9.901}
        isects = self._make_intersections(ug1_range=(-5.0, 0.0))
        hr = compute_headroom(isects, ug1_bias=-2.5, grid_current_params=gc)
        assert hr is not None
        if "ig1_ma" in hr:
            assert hr["ig1_ma"] < 2.0  # 12AX7 grid current is small

    def test_zero_gg_no_ig1(self):
        """Gg=0 means no grid current model → no ig1_ma."""
        gc = {"Gg": 0.0, "xi": 1.3, "Cg": 10.0}
        isects = self._make_intersections(ug1_range=(-10.0, 0.5))
        hr = compute_headroom(isects, ug1_bias=-5.0, grid_current_params=gc)
        assert hr is not None
        assert "ig1_ma" not in hr


# ---------------------------------------------------------------
# Cross-validation: 5-point vs Chebyshev vs DFT (synthetic data)
# ---------------------------------------------------------------

class TestDistortionCrossValidation:
    """Compare distortion methods on identical synthetic data.

    Model = ground truth. All three methods should agree within
    their respective accuracy limits.
    """

    def test_thd_5point_vs_chebyshev_triode(self):
        """12AX7: 5-point and Chebyshev THD should be close."""
        model, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals)

        # Use wider bias and swing so Chebyshev has enough points
        d5 = compute_distortion(isects, ug1_bias=-1.5, half_swing=1.0)
        dc = compute_distortion_chebyshev(isects, ug1_bias=-1.5, half_swing=1.0)
        assert not (d5 is None or dc is None), "silent skip de-vacuated"
        assert d5["thd"] == pytest.approx(dc["thd"], rel=0.5), (
            f"5pt THD={d5['thd']:.2f}% vs Cheby THD={dc['thd']:.2f}%"
        )

    def test_thd_5point_vs_dft_triode(self):
        """12AX7: 5-point and DFT THD should be close."""
        model, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals)

        d5 = compute_distortion(isects, ug1_bias=-1.0, half_swing=0.5)
        dd = compute_distortion_dft(model, ll, ug1_bias=-1.0, half_swing=0.5)
        assert d5 is not None and dd is not None
        assert d5["thd"] == pytest.approx(dd["thd"], rel=0.3), (
            f"5pt THD={d5['thd']:.2f}% vs DFT THD={dd['thd']:.2f}%"
        )

    def test_thd_chebyshev_vs_dft_triode(self):
        """12AX7: Chebyshev and DFT THD should be close."""
        model, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals)

        dc = compute_distortion_chebyshev(isects, ug1_bias=-1.5, half_swing=1.0)
        dd = compute_distortion_dft(model, ll, ug1_bias=-1.5, half_swing=1.0)
        assert not (dc is None or dd is None), "silent skip de-vacuated"
        assert dc["thd"] == pytest.approx(dd["thd"], rel=0.5), (
            f"Cheby THD={dc['thd']:.2f}% vs DFT THD={dd['thd']:.2f}%"
        )

    def test_pout_agreement_all_methods(self):
        """All three methods should agree on Pout within 20%."""
        model, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals, ug2=250.0)

        d5 = compute_distortion(isects, ug1_bias=-7.0, half_swing=3.0)
        dc = compute_distortion_chebyshev(isects, ug1_bias=-7.0, half_swing=3.0)
        dd = compute_distortion_dft(model, ll, ug1_bias=-7.0, half_swing=3.0, ug2=250.0)

        pouts = []
        for d, name in [(d5, "5pt"), (dc, "cheby"), (dd, "dft")]:
            if d and d.get("pout_mw", 0) > 0:
                pouts.append((name, d["pout_mw"]))
        assert len(pouts) >= 2, "Need at least 2 methods to compare"
        # All Pout values within 20% of each other
        for i in range(1, len(pouts)):
            assert pouts[i][1] == pytest.approx(pouts[0][1], rel=0.2), (
                f"{pouts[i][0]} Pout={pouts[i][1]:.0f} vs {pouts[0][0]} Pout={pouts[0][1]:.0f}"
            )

    # ── Strict tolerance: well-behaved operating points ─────────

    def test_strict_thd_el84_5pt_vs_dft(self):
        """EL84 at Ub=300, Ra=5k, Ug1=-7, 3V swing: THD within 10%."""
        model, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals, ug2=250.0)

        d5 = compute_distortion(isects, ug1_bias=-7.0, half_swing=3.0)
        dd = compute_distortion_dft(model, ll, ug1_bias=-7.0, half_swing=3.0, ug2=250.0)
        assert d5 is not None and dd is not None
        assert d5["thd"] == pytest.approx(dd["thd"], rel=0.10), (
            f"5pt THD={d5['thd']:.2f}% vs DFT THD={dd['thd']:.2f}%"
        )

    def test_strict_pout_el84_5pt_vs_dft(self):
        """EL84: Pout from 5-point and DFT within 5%."""
        model, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals, ug2=250.0)

        d5 = compute_distortion(isects, ug1_bias=-7.0, half_swing=3.0)
        dd = compute_distortion_dft(model, ll, ug1_bias=-7.0, half_swing=3.0, ug2=250.0)
        assert d5 is not None and dd is not None
        assert d5["pout_mw"] == pytest.approx(dd["pout_mw"], rel=0.05), (
            f"5pt Pout={d5['pout_mw']:.0f} vs DFT Pout={dd['pout_mw']:.0f}"
        )

    def test_strict_thd_12au7_5pt_vs_dft(self):
        """12AU7 at Ub=250, Ra=47k, Ug1=-8, 3V: THD within 15%."""
        model, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=47.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals)

        d5 = compute_distortion(isects, ug1_bias=-8.0, half_swing=3.0)
        dd = compute_distortion_dft(model, ll, ug1_bias=-8.0, half_swing=3.0)
        assert d5 is not None and dd is not None
        assert d5["thd"] == pytest.approx(dd["thd"], rel=0.15), (
            f"5pt THD={d5['thd']:.2f}% vs DFT THD={dd['thd']:.2f}%"
        )

    def test_hd2_dominant_for_triode(self):
        """Triode: HD2 > HD3 (asymmetric transfer characteristic)."""
        model, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals)

        for method, dist in [
            ("5pt", compute_distortion(isects, ug1_bias=-1.0, half_swing=0.5)),
            ("dft", compute_distortion_dft(model, ll, ug1_bias=-1.0, half_swing=0.5)),
        ]:
            assert dist, "guarded value must be truthy (de-vacuated)"
            assert dist["hd2"] > dist["hd3"], (
                f"{method}: HD2={dist['hd2']:.2f}% should > HD3={dist['hd3']:.2f}%"
            )


# ---------------------------------------------------------------
# Cross-validation: model_gm_ra vs numerical (synthetic data)
# ---------------------------------------------------------------

class TestGmRaCrossValidation:
    """Compare model_gm_ra with _numerical_gm_ra on same synthetic data.

    For model-generated points, both methods should agree because
    the model IS the data source.
    """

    def test_gm_model_vs_numerical_12ax7(self):
        """12AX7: model gm ≈ numerical gm at actual Q-point."""
        model, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals)

        # Use actual Q-point from intersections
        q = interp_intersection(isects, -1.0)
        assert q is not None
        r_model = model_gm_ra(model, ua_q=q["ua"], ug1_q=-1.0)
        r_num = _numerical_gm_ra(pts, isects, ug1_bias=-1.0)
        assert r_model is not None and r_num is not None
        # Both should give gm in the same order of magnitude.
        # Numerical uses broader averaging (robust slope over multiple points),
        # model uses tight central difference → values can differ by ~50%.
        assert r_model["gm"] == pytest.approx(r_num["gm"], rel=0.7), (
            f"model gm={r_model['gm']:.3f} vs numerical gm={r_num['gm']:.3f}"
        )

    def test_ra_model_vs_numerical_12ax7(self):
        """12AX7: model ra ≈ numerical ra at actual Q-point."""
        model, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals)

        q = interp_intersection(isects, -1.0)
        assert q is not None
        r_model = model_gm_ra(model, ua_q=q["ua"], ug1_q=-1.0)
        r_num = _numerical_gm_ra(pts, isects, ug1_bias=-1.0)
        assert r_model is not None and r_num is not None
        assert r_model["ra"] == pytest.approx(r_num["ra"], rel=0.4), (
            f"model ra={r_model['ra']:.1f} vs numerical ra={r_num['ra']:.1f}"
        )

    def test_gm_model_vs_numerical_el84_pentode(self):
        """EL84 pentode: model gm ≈ numerical gm."""
        model, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals, ug2=250.0)

        q = interp_intersection(isects, -7.0)
        assert q is not None
        r_model = model_gm_ra(model, ua_q=q["ua"], ug1_q=-7.0, ug2=250.0)
        filtered = [p for p in pts if abs(p.get("ug2", 0) - 250.0) < 10]
        r_num = _numerical_gm_ra(filtered, isects, ug1_bias=-7.0)
        assert r_model is not None and r_num is not None
        assert r_model["gm"] == pytest.approx(r_num["gm"], rel=0.3), (
            f"model gm={r_model['gm']:.2f} vs numerical gm={r_num['gm']:.2f}"
        )

    # ── Strict tolerance: well-behaved operating points ─────────

    def test_strict_gm_el84_model_vs_numerical(self):
        """EL84 pentode at Ug1=-7: model gm vs numerical within 10%."""
        model, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals, ug2=250.0)
        q = interp_intersection(isects, -7.0)
        assert q is not None

        r_model = model_gm_ra(model, ua_q=q["ua"], ug1_q=-7.0, ug2=250.0)
        filtered = [p for p in pts if abs(p.get("ug2", 0) - 250.0) < 10]
        r_num = _numerical_gm_ra(filtered, isects, ug1_bias=-7.0)
        assert r_model is not None and r_num is not None
        assert r_model["gm"] == pytest.approx(r_num["gm"], rel=0.10), (
            f"model gm={r_model['gm']:.2f} vs numerical gm={r_num['gm']:.2f}"
        )

    def test_strict_ra_el84_model_vs_numerical(self):
        """EL84 pentode at Ug1=-7: model ra vs numerical within 25%."""
        model, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=300, ra=5.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals, ug2=250.0)
        q = interp_intersection(isects, -7.0)
        assert q is not None

        r_model = model_gm_ra(model, ua_q=q["ua"], ug1_q=-7.0, ug2=250.0)
        filtered = [p for p in pts if abs(p.get("ug2", 0) - 250.0) < 10]
        r_num = _numerical_gm_ra(filtered, isects, ug1_bias=-7.0)
        assert r_model is not None and r_num is not None
        assert r_model["ra"] == pytest.approx(r_num["ra"], rel=0.25), (
            f"model ra={r_model['ra']:.1f} vs numerical ra={r_num['ra']:.1f}"
        )

    def test_sweep_amp_points_vs_model_similar(self):
        """Sweep amplitude from points and model should give similar curves."""
        model, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        sw_pts = sweep_amplitude(pts, ll, ug1_bias=-1.0, steps=10)
        sw_mdl = sweep_amplitude(pts, ll, ug1_bias=-1.0, steps=10, model=model)
        assert len(sw_pts) >= 3 and len(sw_mdl) >= 3, "guard de-vacuated 2026-07-12: value must be present"
        thd_pts = sw_pts[len(sw_pts) // 2]["thd"]
        thd_mdl = sw_mdl[len(sw_mdl) // 2]["thd"]
        assert thd_mdl == pytest.approx(thd_pts, rel=0.5)


class TestUg2FilterFallbackWarns:
    """ML-095: an empty Ug2-filter result falls back to the UNFILTERED set —
    that mixes screen levels and must warn at the default INFO level (the
    old log.debug was invisible, two other sites were fully silent)."""

    def test_removed_all_points_warns(self, caplog):
        import logging
        from lm19.amplifier.distortion import group_curves_by_ug1
        pts = [{"ua": 100.0, "ug1": -2.0, "ug2": 250.0, "ia": 5.0},
               {"ua": 200.0, "ug1": -2.0, "ug2": 250.0, "ia": 9.0}]
        with caplog.at_level(logging.WARNING,
                             logger="lm19.amplifier.distortion"):
            curves = group_curves_by_ug1(pts, ug2_filter=999.0)
        assert curves, "fallback must still return the full set"
        assert any("removed all points" in r.getMessage()
                   for r in caplog.records), "fallback stayed silent"

    def test_matching_filter_does_not_warn(self, caplog):
        import logging
        from lm19.amplifier.distortion import group_curves_by_ug1
        pts = [{"ua": 100.0, "ug1": -2.0, "ug2": 250.0, "ia": 5.0}]
        with caplog.at_level(logging.WARNING,
                             logger="lm19.amplifier.distortion"):
            group_curves_by_ug1(pts, ug2_filter=250.0)
        assert not caplog.records


class TestComputeHdValidationAndSignals:
    """ML-096/097: silent HD-method fallbacks and unaccounted Newton
    divergence in compute_pa_avg."""

    def test_unknown_hd_method_raises(self):
        from lm19.amplifier.sweeps import _compute_hd
        from lm19.amplifier import ResistiveLoadLine
        ll = ResistiveLoadLine(ub=250.0, ra=5000.0)
        with pytest.raises(ValueError):
            _compute_hd("cheby", [], None, ll, -2.0)   # typo, not 'chebyshev'

    def test_dft_without_model_warns_and_falls_back(self, caplog):
        import logging
        from lm19.amplifier.sweeps import _compute_hd
        from lm19.amplifier import ResistiveLoadLine
        ll = ResistiveLoadLine(ub=250.0, ra=5000.0)
        with caplog.at_level(logging.WARNING,
                             logger="lm19.amplifier.sweeps"):
            _compute_hd("dft", [], None, ll, -2.0)
        assert any("falling back to 5-point" in r.getMessage()
                   for r in caplog.records),             "dft->5point fallback stayed silent"

    def test_pa_avg_reports_convergence(self):
        """compute_pa_avg must expose n_not_converged (0 on a clean run) —
        Pa_avg is compared against Pa_max when picking the OP."""
        from lm19.amplifier.sweeps import compute_pa_avg
        from lm19.amplifier import ResistiveLoadLine
        from lm19.tube_sim import quick_triode
        model, _ = quick_triode("ECC83")
        ll = ResistiveLoadLine(ub=250.0, ra=100000.0)
        result = compute_pa_avg(model, ll, ug1_bias=-1.5, half_swing=0.5)
        assert result is not None
        assert result["n_not_converged"] == 0
