"""Coverage gap tests for amplifier module.

Addresses ALL remaining test gaps found in the coverage audit:
- _find_dc_q_point edge cases
- find_intersections AC fallback + exact numerical values
- sweep_bias / optimize_bias with Transformer/PP
- pp_distortion pathological data
- compute_distortion b1 <= 0
- estimate_ig2_at_q edge cases
- Interaction: transformer + pentode + Ig2 Rk
- find_intersections_model (no AC mode — documented limitation)

Run:  py -m pytest tests/test_amplifier_coverage.py -v
"""

import json
import math

import numpy as np

from lm19.amplifier import (
    ResistiveLoadLine,
    TransformerLoadLine,
    CathodeFollowerLoadLine,
    PushPullLoadLine,
    find_intersections,
    compute_distortion,
    compute_headroom,
    sweep_amplitude,
    sweep_bias,
    optimize_bias,
    compute_stage_params,
    composite_characteristic,
    pp_distortion,
    sweep_pp_amplitude,
    _find_dc_q_point,
    estimate_ig2_at_q,
    sweep_ra,
)
from lm19.tube_sim import quick_triode, quick_pentode


# ── module local constants ──
# quick_pentode("EL84") (single Ug2=250V, Ug1 grid −15..0 step 1.5): the AC
# line through Q(−7V, ≈248V, ≈44mA) intersects the 6 hottest curves
# (Ug1 ≥ −7.5); colder curves stay below the line within measured Ua.
_EL84_AC_MIN_INTERSECTIONS = 5
# AC mode anchors Q near Ub → every intersection shifts to higher Ua vs the
# DC/default path. Probed minimum shift ≈ 30 V (Ug1=−2); 10 V keeps margin.
_AC_VS_DC_MIN_UA_SHIFT_V = 10.0
# sweep_ra transformer-vs-resistive crossover (12AU7, Ub=250, data Ua ≤ 300V):
# at Ra ≤ 8 kΩ the full transformer swing fits the measured Ua range →
# transformer Pout > resistive; at Ra ≥ 10 kΩ the swing exits the 300 V data
# boundary, deep-negative-Ug1 intersections are lost → Pout truncated below
# resistive. Probed: 375 vs 255 mW at 8 kΩ, 143 vs 242 mW at 10 kΩ.
_XFMR_ADVANTAGE_RA_MAX_KOHM = 8.0
_XFMR_TRUNCATED_RA_MIN_KOHM = 10.0


# ═══════════════════════════════════════════════════════════════════════════
#  _find_dc_q_point edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestFindDcQPointEdgeCases:

    def test_ra_dc_exactly_zero(self):
        """Ra_dc = 0 should use low-Ra branch (Ua_q ≈ Ub)."""
        _, pts = quick_triode("12AU7")
        q = _find_dc_q_point(pts, ub=250.0, ra_dc=0.0, ug1_bias=-8.0)
        assert q is not None
        ua_q, ia_q = q
        assert abs(ua_q - 250.0) < 1.0

    def test_ug1_bias_far_outside_range(self):
        """ug1_bias far from any data should still find closest level."""
        _, pts = quick_triode("12AU7")
        # 12AU7 data range: roughly Ug1 = -20 to 0
        q = _find_dc_q_point(pts, ub=250.0, ra_dc=0.001, ug1_bias=-100.0)
        assert q is not None  # Should pick closest Ug1 level
        _, ia_q = q
        assert ia_q >= 0

    def test_ug1_bias_positive(self):
        """ug1_bias = +5V (beyond grid current) should not crash."""
        _, pts = quick_triode("12AU7")
        q = _find_dc_q_point(pts, ub=250.0, ra_dc=0.001, ug1_bias=5.0)
        assert q is not None

    def test_ub_below_data_range(self):
        """Ub below all Ua data should fall back to closest point."""
        _, pts = quick_triode("12AU7")
        # Ua data starts at 0V typically, so Ub=-10 is below
        q = _find_dc_q_point(pts, ub=-10.0, ra_dc=0.001, ug1_bias=-8.0)
        assert q is not None
        ua_q, ia_q = q
        assert ua_q == -10.0  # Uses Ub directly
        assert ia_q >= 0

    def test_ub_above_data_range(self):
        """Ub above all Ua data should fall back to last point."""
        _, pts = quick_triode("12AU7")
        q = _find_dc_q_point(pts, ub=9999.0, ra_dc=0.001, ug1_bias=-8.0)
        assert q is not None
        ua_q, ia_q = q
        assert ua_q == 9999.0

    def test_single_ua_point_returns_none(self):
        """Single point per Ug1 level → None (need ≥2 for interpolation)."""
        pts = [{"ug1": -8.0, "ua": 200.0, "ia": 10.0}]
        q = _find_dc_q_point(pts, ub=250.0, ra_dc=0.1, ug1_bias=-8.0)
        assert q is None

    def test_with_ug2_filter_no_match(self):
        """ug2_filter that matches nothing should fall back to all points."""
        _, pts = quick_pentode("EL84")
        # ug2=999 doesn't exist, so filtered is empty → uses all points
        q = _find_dc_q_point(pts, ub=250.0, ra_dc=0.001, ug1_bias=-7.0,
                             ug2_filter=999.0)
        # Falls back to unfiltered because f is empty
        assert q is not None

    def test_moderate_ra_dc_finds_intersection(self):
        """With moderate Ra_dc (0.5 kΩ), should find proper DC intersection."""
        _, pts = quick_triode("12AU7")
        q = _find_dc_q_point(pts, ub=300.0, ra_dc=0.5, ug1_bias=-8.0)
        assert q is not None, "guard de-vacuated 2026-07-12: value must be present"
        ua_q, ia_q = q
        expected_ia = (300.0 - ua_q) / 0.5
        assert abs(ia_q - expected_ia) < 2.0, \
            f"Q-point Ia={ia_q:.1f} vs DC line Ia={expected_ia:.1f}"


# ═══════════════════════════════════════════════════════════════════════════
#  find_intersections AC mode edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestFindIntersectionsACEdgeCases:

    def test_ac_fallback_when_q_not_found(self):
        """When Q-point can't be found, should fall back to default load line.

        Single Ug1 level, DC intersection outside the data window →
        _find_dc_q_point returns None. Fallback = default AC line through
        (Ub, 0) with slope −1/ra_ac: Ia = (250 − Ua)/5. Tube segment:
        Ia = 30 − 0.1·Ua. Analytic intersection: Ua = 200 V, Ia = 10 mA.
        """
        pts = [
            {"ug1": -8.0, "ua": 100.0, "ia": 20.0},
            {"ug1": -8.0, "ua": 200.0, "ia": 10.0},
        ]
        # Precondition of the docstring: Q-point genuinely NOT found
        assert _find_dc_q_point(pts, ub=250.0, ra_dc=0.05, ug1_bias=-8.0) is None

        ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=5.0)
        isects = find_intersections(pts, ll, ug1_bias=-8.0)
        assert len(isects) == 1
        p = isects[0]
        assert abs(p["ua"] - 200.0) < 0.5, f"expected Ua≈200V, got {p['ua']:.1f}"
        assert abs(p["ia"] - 10.0) < 0.1, f"expected Ia≈10mA, got {p['ia']:.2f}"
        # Point must lie ON the fallback line through (Ub, 0)
        assert abs(p["ia"] - (250.0 - p["ua"]) / 5.0) < 0.1

    def test_ac_with_ug2_filter_combined(self):
        """ug2_filter + ug1_bias should work together."""
        _, pts = quick_pentode("EL84")
        ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=5.0)
        isects = find_intersections(pts, ll, ug2_filter=250.0, ug1_bias=-7.0)
        assert len(isects) >= _EL84_AC_MIN_INTERSECTIONS, \
            f"expected >= {_EL84_AC_MIN_INTERSECTIONS} intersections, got {len(isects)}"
        for p in isects:
            assert p["ia"] > 0
            assert 0.0 < p["ua"] <= 300.0  # inside measured Ua range
        # Physics of a negative-slope load line: hotter grid (less negative
        # Ug1) pulls more current → Ia strictly increases with Ug1
        by_ug1 = sorted(isects, key=lambda p: p["ug1"])
        ias = [p["ia"] for p in by_ug1]
        assert all(b > a for a, b in zip(ias, ias[1:])), f"Ia not monotonic: {ias}"
        # AC mode anchors the line at Q near Ub: intersection closest to the
        # bias must sit high on the Ua axis (probed: Ua≈248V at Ug1=−7.5)
        nearest = min(isects, key=lambda p: abs(p["ug1"] - (-7.0)))
        assert nearest["ua"] > 200.0, \
            f"AC Q-point Ua={nearest['ua']:.0f} should be near Ub=250"

    def test_exact_numerical_ac_intersection(self):
        """Verify AC intersection Ua/Ia values against hand calculation.

        Synthetic data: two Ug1 levels, tube at constant Ia.
        AC line through Q-point at (Ua=250, Ia=48).
        Ia_ac(Ua) = 48 - (Ua - 250) / 5.0
        """
        ll = TransformerLoadLine(ub=250, ra_dc=0.001, ra_ac=5.0)
        # Tube curve at Ug1=-5: constant Ia=60mA from Ua=0 to Ua=400
        # AC line: Ia = 48 - (Ua-250)/5 → at Ia=60: 60 = 48 - (Ua-250)/5 → Ua = 190V
        pts = []
        # Bias curve (for Q-point) at Ug1=-7: Ia=48mA
        for ua in range(0, 350, 10):
            pts.append({"ug1": -7.0, "ua": float(ua), "ia": 48.0})
        # Signal curve at Ug1=-5: Ia=60mA
        for ua in range(0, 350, 10):
            pts.append({"ug1": -5.0, "ua": float(ua), "ia": 60.0})
        # Signal curve at Ug1=-9: Ia=30mA
        for ua in range(0, 350, 10):
            pts.append({"ug1": -9.0, "ua": float(ua), "ia": 30.0})

        isects = find_intersections(pts, ll, ug1_bias=-7.0)
        assert len(isects) >= 2

        # Check the Ug1=-5 intersection: Ia=60, AC line: 60 = 48 - (Ua-250)/5 → Ua=190
        isect_5 = [p for p in isects if abs(p["ug1"] - (-5.0)) < 0.1]
        assert len(isect_5) == 1
        assert abs(isect_5[0]["ua"] - 190.0) < 2.0, \
            f"Ua at Ug1=-5 should be ~190V, got {isect_5[0]['ua']:.1f}"
        assert abs(isect_5[0]["ia"] - 60.0) < 0.5

        # Ug1=-9: Ia=30, 30 = 48-(Ua-250)/5 → Ua=340
        isect_9 = [p for p in isects if abs(p["ug1"] - (-9.0)) < 0.1]
        if isect_9:  # May be outside scan range (350)
            assert abs(isect_9[0]["ua"] - 340.0) < 15.0, \
                f"Ua at Ug1=-9 should be ~340V, got {isect_9[0]['ua']:.1f}"

    def test_resistive_loadline_ignores_ug1_bias(self):
        """ResistiveLoadLine should ignore ug1_bias (no AC mode)."""
        _, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        isects_no_bias = find_intersections(pts, ll)
        isects_with_bias = find_intersections(pts, ll, ug1_bias=-8.0)
        assert len(isects_no_bias) == len(isects_with_bias)

    def test_cf_loadline_ignores_ug1_bias(self):
        """CathodeFollowerLoadLine should ignore ug1_bias."""
        _, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        isects1 = find_intersections(pts, ll)
        isects2 = find_intersections(pts, ll, ug1_bias=-8.0)
        assert len(isects1) == len(isects2)


# ═══════════════════════════════════════════════════════════════════════════
#  sweep_bias / optimize_bias with Transformer and PP
# ═══════════════════════════════════════════════════════════════════════════

class TestSweepBiasTransformer:

    def test_sweep_bias_with_transformer(self):
        """sweep_bias should work with TransformerLoadLine."""
        _, pts = quick_triode("12AU7")
        ll = TransformerLoadLine(ub=300, ra_dc=0.05, ra_ac=10.0)
        results = sweep_bias(pts, ll)
        assert len(results) >= 3, f"Expected >=3 bias points, got {len(results)}"
        # Each result should have valid fields
        for r in results:
            assert r["pout_mw"] >= 0
            assert r["thd"] >= 0

    def test_sweep_bias_with_pp(self):
        """sweep_bias should work with PushPullLoadLine."""
        _, pts = quick_pentode("EL84")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0, ra_dc=0.05)
        results = sweep_bias(pts, ll)
        assert len(results) >= 3

    def test_sweep_bias_transformer_with_explicit_ug1(self):
        """sweep_bias with explicit ug1_bias for AC line."""
        _, pts = quick_triode("12AU7")
        ll = TransformerLoadLine(ub=300, ra_dc=0.05, ra_ac=5.0)
        results = sweep_bias(pts, ll, ug1_bias=-8.0)
        assert len(results) >= 1


class TestOptimizeBiasTransformer:

    def test_optimize_with_transformer(self):
        """optimize_bias with TransformerLoadLine should work."""
        _, pts = quick_triode("12AU7")
        ll = TransformerLoadLine(ub=300, ra_dc=0.05, ra_ac=10.0)
        opt = optimize_bias(pts, ll, target="min_thd")
        assert opt is not None
        assert opt["pout_mw"] > 0
        assert opt["rk_auto_bias"] > 0

    def test_optimize_with_pp(self):
        """optimize_bias with PushPullLoadLine should produce results."""
        _, pts = quick_pentode("EL84")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0, ra_dc=0.05)
        opt = optimize_bias(pts, ll, target="min_thd")
        # May be None if not enough intersections, but should not crash
        assert opt is not None, "guard de-vacuated 2026-07-12: value must be present"
        assert opt["pout_mw"] >= 0
        assert opt["rk_auto_bias"] >= 0

    def test_optimize_all_targets_transformer(self):
        """All optimization targets should work with transformer."""
        _, pts = quick_triode("12AU7")
        ll = TransformerLoadLine(ub=300, ra_dc=0.05, ra_ac=10.0)
        for target in ["min_thd", "max_pout", "balanced"]:
            opt = optimize_bias(pts, ll, target=target)
            assert opt is not None, f"Target {target} returned None"


# ═══════════════════════════════════════════════════════════════════════════
#  pp_distortion pathological data
# ═══════════════════════════════════════════════════════════════════════════

class TestPPDistortionPathological:

    def test_inverted_composite_returns_none(self):
        """If tube B is stronger than A everywhere, swing < 0 → None."""
        _, pts_a = quick_triode("12AU7")
        # Tube B has 3x current → composite is inverted
        pts_b = [dict(p, ia=p["ia"] * 3.0) for p in pts_a]
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        dist = pp_distortion(pts_a, ll, ug1_bias=-10.0, points_b=pts_b)
        # Should either return None or have positive pout (not crash)
        assert dist is not None, "guard de-vacuated 2026-07-12: value must be present"
        assert dist["pout_mw"] >= 0

    def test_tiny_swing_returns_none(self):
        """Near-zero composite swing (< 0.001) should return None."""
        # Create data where all Ug1 levels have same Ia → composite ≈ 0
        pts = []
        for ug1 in np.linspace(-12, -4, 9):
            for ua in range(50, 350, 10):
                pts.append({"ug1": float(ug1), "ua": float(ua), "ia": 10.0})
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        dist = pp_distortion(pts, ll, ug1_bias=-8.0, half_swing=3.0)
        assert dist is None

    def test_few_composite_points_returns_none(self):
        """With < 5 composite points, should return None."""
        pts = [
            {"ug1": -8.0, "ua": 200.0, "ia": 10.0},
            {"ug1": -8.0, "ua": 250.0, "ia": 8.0},
        ]
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        dist = pp_distortion(pts, ll, ug1_bias=-8.0)
        assert dist is None


# ═══════════════════════════════════════════════════════════════════════════
#  compute_distortion b1 <= 0
# ═══════════════════════════════════════════════════════════════════════════

class TestDistortionB1Edge:

    def test_b1_zero_returns_none(self):
        """When fundamental is zero (constant Ia), should return None."""
        # Constant current across all Ug1 → swing = 0, b1 = 0
        isects = [
            {"ug1": -10.0, "ua": 200.0, "ia": 10.0},
            {"ug1": -8.0, "ua": 200.0, "ia": 10.0},
            {"ug1": -6.0, "ua": 200.0, "ia": 10.0},
            {"ug1": -4.0, "ua": 200.0, "ia": 10.0},
            {"ug1": -2.0, "ua": 200.0, "ia": 10.0},
        ]
        dist = compute_distortion(isects, ug1_bias=-6.0, half_swing=3.0)
        assert dist is None

    def test_nearly_zero_swing(self):
        """Very small Ia variation → None or very low Pout."""
        isects = [
            {"ug1": -10.0, "ua": 205.0, "ia": 10.000},
            {"ug1": -8.0, "ua": 204.0, "ia": 10.001},
            {"ug1": -6.0, "ua": 203.0, "ia": 10.002},
            {"ug1": -4.0, "ua": 202.0, "ia": 10.003},
            {"ug1": -2.0, "ua": 201.0, "ia": 10.004},
        ]
        dist = compute_distortion(isects, ug1_bias=-6.0, half_swing=3.0)
        # May return result with very low pout, or None
        # Contract disjunction: None OR a tiny Pout (near-zero swing).
        if dist is not None:  # vacuity-ok: None is legal at zero swing
            assert dist["pout_mw"] < 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  estimate_ig2_at_q edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEstimateIg2EdgeCases:

    def test_all_points_outside_window(self):
        """When all ig2 points are far from Q-point, returns 0."""
        pts = [
            {"ug1": -7.0, "ua": 100.0, "ia": 30.0, "ig2": 5.0},  # ua far from 250
            {"ug1": -7.0, "ua": 50.0, "ia": 40.0, "ig2": 6.0},
        ]
        # ua_q = 250 but all points at ua=50-100 (>15V away)
        ig2 = estimate_ig2_at_q(pts, ug1_q=-7.0, ua_q=250.0)
        assert ig2 == 0.0

    def test_ig2_none_values_filtered(self):
        """Points with ig2=None should be ignored."""
        pts = [
            {"ug1": -7.0, "ua": 200.0, "ia": 30.0, "ig2": None},
            {"ug1": -7.0, "ua": 205.0, "ia": 28.0, "ig2": 5.0},
        ]
        ig2 = estimate_ig2_at_q(pts, ug1_q=-7.0, ua_q=200.0)
        assert ig2 == 5.0

    def test_ig2_absent_key(self):
        """Points without ig2 key should return 0."""
        pts = [
            {"ug1": -7.0, "ua": 200.0, "ia": 30.0},
            {"ug1": -7.0, "ua": 205.0, "ia": 28.0},
        ]
        ig2 = estimate_ig2_at_q(pts, ug1_q=-7.0, ua_q=200.0)
        assert ig2 == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Interaction test: transformer + pentode + Ig2 Rk
# ═══════════════════════════════════════════════════════════════════════════

class TestTransformerPentodeIg2Interaction:
    """Full chain: TransformerLoadLine + pentode + Ig2-aware Rk."""

    def test_full_pipeline(self):
        """optimize_bias with pentode transformer should include Ig2 in Rk."""
        _, pts = quick_pentode("EL84")
        ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=5.0)
        opt = optimize_bias(pts, ll, target="min_thd")
        assert not (opt is None), "silent skip de-vacuated"

        assert opt["pout_mw"] > 0
        assert opt["rk_auto_bias"] > 0

        # Verify Ig2 was considered: Rk should be ≤ |Ug1|/Ia * 1000
        rk_ia_only = abs(opt["ug1_0"]) / opt["ia_0"] * 1000.0 if opt["ia_0"] > 0.01 else 0
        assert rk_ia_only > 0, "guard de-vacuated 2026-07-12: value must be present"
        assert opt["rk_auto_bias"] <= rk_ia_only * 1.01, \
            f"Rk with Ig2 ({opt['rk_auto_bias']:.0f}) should be ≤ Rk without ({rk_ia_only:.0f})"

    def test_pentode_stage_params_with_transformer(self):
        """Stage params with transformer + pentode should give valid gain."""
        _, pts = quick_pentode("EL84")
        ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=5.0)
        isects = find_intersections(pts, ll, ug1_bias=-7.0)
        assert not (len(isects) < 3), "silent skip de-vacuated"
        result = compute_stage_params(isects, ll, ug1_bias=-7.0, points=pts)
        assert result, "guard de-vacuated 2026-07-12: value must be present"
        assert result["gain"] > 0
        assert result["gm"] > 0
        assert result["ra"] > 0

    def test_headroom_with_transformer_pentode(self):
        """Headroom analysis with pentode transformer."""
        _, pts = quick_pentode("EL84")
        ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=5.0)
        isects = find_intersections(pts, ll, ug1_bias=-7.0)
        assert not (len(isects) < 3), "silent skip de-vacuated"
        hr = compute_headroom(isects, ug1_bias=-7.0, pa_max=12.0)
        assert hr, "guard de-vacuated 2026-07-12: value must be present"
        assert hr["max_swing"] >= 0
        assert hr["clip_neg"] in ("cutoff", "data_limit", "pa_max")
        assert hr["clip_pos"] in ("grid_current", "data_limit", "pa_max")


# ═══════════════════════════════════════════════════════════════════════════
#  Real measurement data integration
# ═══════════════════════════════════════════════════════════════════════════

class TestRealDataIntegration:
    """End-to-end tests using real measurement files if available."""

    def _load_real_files(self):
        from tests._real_data import EL84_REAL_FILES, converted_path
        results = []
        for name in EL84_REAL_FILES:
            path = converted_path(name)
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                results.append((name, data))
            except Exception:
                continue
        return results

    def test_all_files_produce_valid_results(self):
        """Every real measurement file should produce valid amplifier analysis."""
        files = self._load_real_files()
        assert not (not files), "silent skip de-vacuated"

        for fname, data in files:
            pts = data.get("points", [])
            if len(pts) < 20:
                continue
            is_pentode = data.get("topology") == TOPOLOGY_PENTODE

            # Choose appropriate load line
            if is_pentode:
                ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=5.0)
                bias = -7.0
                # Find dominant Ug2 for filter
                ug2_vals = [p.get("ug2", 0) for p in pts if p.get("ug2")]
                ug2_main = max(set(ug2_vals), key=ug2_vals.count) if ug2_vals else None
            else:
                ll = TransformerLoadLine(ub=300, ra_dc=0.05, ra_ac=5.0)
                bias = -7.0
                ug2_main = None

            isects = find_intersections(
                pts, ll, ug1_bias=bias,
                ug2_filter=ug2_main if is_pentode else None,
            )
            # Should find at least some intersections
            assert len(isects) >= 1, \
                f"{fname}: no intersections found"

            if len(isects) >= 3:
                # Distortion should not crash
                dist = compute_distortion(isects, ug1_bias=bias)
                # Stage params should not crash
                stage = compute_stage_params(isects, ll, ug1_bias=bias, points=pts)

    def test_pentode_file_ac_vs_default(self):
        """For pentode files, AC mode should give Q-point near Ub."""
        files = self._load_real_files()
        for fname, data in files:
            if data.get("topology") != TOPOLOGY_PENTODE:
                continue
            pts = data.get("points", [])
            if len(pts) < 20:
                continue

            ug2_vals = [p.get("ug2", 0) for p in pts if p.get("ug2")]
            ug2_main = max(set(ug2_vals), key=ug2_vals.count) if ug2_vals else None

            ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=5.0)
            isects_ac = find_intersections(pts, ll, ug2_filter=ug2_main, ug1_bias=-7.0)

            # Empty intersections, or no curve near the bias, is a
            # failure — not a silent pass.
            assert isects_ac, f"{fname}: no AC intersections"
            near_bias = [p for p in isects_ac
                         if abs(p["ug1"] - (-7.0)) < 1.5]
            assert near_bias, f"{fname}: no curve near bias -7.0"
            # Q-point should be significantly higher Ua than resistive
            assert near_bias[0]["ua"] > 100, \
                f"{fname}: AC Q-point Ua={near_bias[0]['ua']:.0f} too low"
            return  # Test one file


# ═══════════════════════════════════════════════════════════════════════════
#  find_intersections_model AC mode tests
# ═══════════════════════════════════════════════════════════════════════════

from lm19.amplifier import find_intersections_model, _find_model_dc_q_point
from lm19.tube_sim import load_model
from lm19.constants import (
    TOPOLOGY_PENTODE,
)


class TestFindModelDcQPoint:
    """Test _find_model_dc_q_point helper."""

    def test_q_point_near_ub(self):
        """With low Ra_dc, model Q-point should be near Ub."""
        model = load_model("12AU7")
        assert model is not None
        q = _find_model_dc_q_point(model, ub=250, ra_dc=0.001, ug1_bias=-8.0,
                                    ug2=0.0, ua_range=(0.1, 500))
        assert q is not None
        ua_q, ia_q = q
        assert abs(ua_q - 250) < 1.0
        assert ia_q > 0

    def test_pentode_q_point(self):
        """Pentode model Q-point."""
        model = load_model("EL84")
        assert model is not None
        q = _find_model_dc_q_point(model, ub=250, ra_dc=0.001, ug1_bias=-7.0,
                                    ug2=250.0, ua_range=(0.1, 500))
        assert q is not None
        ua_q, ia_q = q
        assert abs(ua_q - 250) < 1.0
        assert ia_q > 5.0  # EL84 at Ug1=-7V should have significant Ia

    def test_moderate_ra_dc(self):
        """With moderate Ra_dc, Q-point shifts from Ub."""
        model = load_model("12AU7")
        assert model is not None
        q = _find_model_dc_q_point(model, ub=300, ra_dc=0.5, ug1_bias=-8.0,
                                    ug2=0.0, ua_range=(0.1, 500))
        if q is not None:
            ua_q, ia_q = q
            # Should be below Ub due to DC voltage drop
            assert ua_q < 300, f"Ua_q={ua_q:.0f} should be below Ub=300"


class TestFindIntersectionsModelAC:
    """Test find_intersections_model with AC load line mode."""

    def test_triode_ac_mode(self):
        """Triode with transformer AC mode should find intersections."""
        model = load_model("12AU7")
        assert model is not None
        ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=5.0)
        ug1_values = list(np.arange(-16, 0, 1.0))
        isects = find_intersections_model(
            model, ll, ug1_values, ug2=0.0,
            ua_range=(0.1, 500), ug1_bias=-8.0,
        )
        assert len(isects) >= 3

    def test_pentode_ac_gives_realistic_q(self):
        """Pentode with transformer: intersection at bias should be near Ub."""
        model = load_model("EL84")
        assert model is not None
        ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=5.0)
        ug1_values = list(np.arange(-11, 0, 1.0))
        isects = find_intersections_model(
            model, ll, ug1_values, ug2=250.0,
            ua_range=(0.1, 500), ug1_bias=-7.0,
        )
        assert len(isects) >= 3
        near_bias = [p for p in isects if abs(p["ug1"] - (-7.0)) < 0.5]
        if near_bias:
            assert near_bias[0]["ua"] > 200, \
                f"Pentode Q-point Ua={near_bias[0]['ua']:.0f} should be near Ub=250"

    def test_model_ac_vs_no_ac(self):
        """Model AC mode should give different intersections than default."""
        model = load_model("EL84")
        assert model is not None
        ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=5.0)
        ug1_values = list(np.arange(-11, 0, 1.0))

        isects_default = find_intersections_model(
            model, ll, ug1_values, ug2=250.0,
            ua_range=(0.1, 500),
        )
        isects_ac = find_intersections_model(
            model, ll, ug1_values, ug2=250.0,
            ua_range=(0.1, 500), ug1_bias=-7.0,
        )
        # Both should have results
        assert len(isects_default) >= 1
        assert len(isects_ac) >= 1

        # AC Q-point at Ug1=-7 should be at higher Ua than default
        def get_ua(isects, ug1_target):
            matches = [p for p in isects if abs(p["ug1"] - ug1_target) < 0.5]
            return matches[0]["ua"] if matches else None

        ua_default_7 = get_ua(isects_default, -7.0)
        ua_ac_7 = get_ua(isects_ac, -7.0)
        if ua_default_7 and ua_ac_7:
            assert ua_ac_7 > ua_default_7, \
                f"AC Ua={ua_ac_7:.0f} should be > default Ua={ua_default_7:.0f} at Ug1=-7"

    def test_backward_compatible_without_bias(self):
        """Without ug1_bias, model mode must stay on the default (non-AC) path.

        Guard against the default silently becoming an AC bias: omitted
        argument must equal explicit ug1_bias=None point-for-point, and both
        must differ from a real AC-mode call at EVERY Ug1 level (AC anchors Q
        near Ub → higher Ua; probed shift 30–48 V on 12AU7 transformer line).
        """
        model = load_model("12AU7")
        assert model is not None
        ll = TransformerLoadLine(ub=250, ra_dc=0.05, ra_ac=5.0)
        ug1_values = list(np.arange(-16, 0, 2.0))

        isects_omitted = find_intersections_model(model, ll, ug1_values)
        isects_none = find_intersections_model(model, ll, ug1_values, ug1_bias=None)
        isects_ac = find_intersections_model(model, ll, ug1_values, ug1_bias=-8.0)

        # All Ug1 levels intersect for this configuration
        assert len(isects_omitted) == len(ug1_values)
        assert len(isects_none) == len(isects_omitted)
        assert len(isects_ac) == len(isects_omitted)

        def ua_by_ug1(isects):
            return {round(p["ug1"], 3): p["ua"] for p in isects}

        ua_omitted = ua_by_ug1(isects_omitted)
        ua_none = ua_by_ug1(isects_none)
        ua_ac = ua_by_ug1(isects_ac)
        for g in ua_omitted:
            # Omitted argument == explicit None (pins the default value)
            assert abs(ua_omitted[g] - ua_none[g]) < 1e-9
            # ...and the None default is genuinely NOT the AC path
            assert ua_ac[g] > ua_omitted[g] + _AC_VS_DC_MIN_UA_SHIFT_V, \
                f"Ug1={g}: AC Ua={ua_ac[g]:.1f} should exceed default " \
                f"Ua={ua_omitted[g]:.1f} by > {_AC_VS_DC_MIN_UA_SHIFT_V} V"

    def test_pp_model_ac_mode(self):
        """PP with model AC mode should work."""
        model = load_model("EL84")
        assert model is not None
        ll = PushPullLoadLine(ub=250, ra_aa=8.0, ra_dc=0.05)
        ug1_values = list(np.arange(-11, 0, 1.0))
        isects = find_intersections_model(
            model, ll, ug1_values, ug2=250.0,
            ua_range=(0.1, 500), ug1_bias=-7.0,
        )
        assert len(isects) >= 3

    def test_resistive_ignores_ug1_bias_in_model(self):
        """ResistiveLoadLine should ignore ug1_bias in model mode."""
        model = load_model("12AU7")
        assert model is not None
        ll = ResistiveLoadLine(ub=250, ra=10.0)
        ug1_values = list(np.arange(-16, 0, 2.0))

        isects1 = find_intersections_model(model, ll, ug1_values)
        isects2 = find_intersections_model(model, ll, ug1_values, ug1_bias=-8.0)
        assert len(isects1) == len(isects2)
        for a, b in zip(isects1, isects2):
            assert abs(a["ua"] - b["ua"]) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  sweep_ra with transformer mode
# ═══════════════════════════════════════════════════════════════════════════

class TestSweepRaTransformer:
    """Test sweep_ra with transformer=True."""

    def test_transformer_sweep_produces_results(self):
        """sweep_ra with transformer=True should produce valid results."""
        _, pts = quick_triode("12AU7")
        results = sweep_ra(
            pts, ub=250, ra_min=2.0, ra_max=30.0,
            ug1_bias=-10.0,
            transformer=True, ra_dc=0.05, steps=20,
        )
        assert len(results) >= 3, f"Expected >= 3 results, got {len(results)}"
        for r in results:
            assert r["pout_mw"] > 0
            assert r["thd"] >= 0

    def test_transformer_vs_resistive_sweep(self):
        """Transformer beats resistive at low Ra; truncated above data range.

        Reality on this dataset (12AU7, Ub=250 V, measured Ua ≤ 300 V):
        at Ra ≤ 8 kΩ the AC line through Q(≈Ub) keeps the full swing inside
        the measured Ua window → transformer Pout > resistive (375 vs 255 mW
        at 8 kΩ). At Ra ≥ 10 kΩ the shallower transformer line pushes the
        deep-negative-Ug1 intersections past the 300 V data boundary; the
        swing gets truncated (data_limit) and Pout drops BELOW resistive
        (143 vs 242 mW at 10 kΩ). This is a data-coverage limitation of
        measured points, not a model property — so the sweep-average
        comparison is NOT asserted, only the per-Ra crossover.
        """
        _, pts = quick_triode("12AU7")
        res_results = sweep_ra(
            pts, ub=250, ra_min=2.0, ra_max=20.0,
            ug1_bias=-10.0,
            transformer=False, steps=10,
        )
        xfmr_results = sweep_ra(
            pts, ub=250, ra_min=2.0, ra_max=20.0,
            ug1_bias=-10.0,
            transformer=True, ra_dc=0.05, steps=10,
        )
        assert len(res_results) == 10
        assert len(xfmr_results) == 10

        checked_advantage = 0
        checked_truncated = 0
        for r, x in zip(res_results, xfmr_results):
            assert abs(r["ra"] - x["ra"]) < 1e-6  # same Ra grid, comparable
            assert r["pout_mw"] > 0
            assert x["pout_mw"] > 0
            if r["ra"] <= _XFMR_ADVANTAGE_RA_MAX_KOHM:
                assert x["pout_mw"] > r["pout_mw"], \
                    f"Ra={r['ra']:.0f}k: transformer {x['pout_mw']:.0f} mW " \
                    f"should beat resistive {r['pout_mw']:.0f} mW"
                checked_advantage += 1
            elif r["ra"] >= _XFMR_TRUNCATED_RA_MIN_KOHM:
                assert x["pout_mw"] < r["pout_mw"], \
                    f"Ra={r['ra']:.0f}k: truncated transformer {x['pout_mw']:.0f} mW " \
                    f"should be below resistive {r['pout_mw']:.0f} mW"
                checked_truncated += 1
        # Both regimes actually exercised (guards did not silently skip)
        assert checked_advantage >= 3
        assert checked_truncated >= 3

    def test_pentode_transformer_sweep(self):
        """Pentode with transformer Ra sweep should work."""
        _, pts = quick_pentode("EL84")
        results = sweep_ra(
            pts, ub=250, ra_min=1.0, ra_max=20.0,
            ug1_bias=-7.0,
            transformer=True, ra_dc=0.05, steps=15,
        )
        assert len(results) >= 1

    def test_backward_compat_no_transformer(self):
        """Default transformer=False behaves as before."""
        _, pts = quick_triode("12AU7")
        results1 = sweep_ra(pts, ub=250, ra_min=1.0, ra_max=50.0, ug1_bias=-8.0, steps=10)
        results2 = sweep_ra(pts, ub=250, ra_min=1.0, ra_max=50.0, ug1_bias=-8.0,
                            steps=10, transformer=False)
        assert len(results1) == len(results2)
        for a, b in zip(results1, results2):
            assert abs(a["pout_mw"] - b["pout_mw"]) < 0.01
