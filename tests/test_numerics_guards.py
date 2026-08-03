"""Numerics guards (ML-138 / ML-139 / ML-140), physicality.

- ML-139: beyond tube B's data edge the mirrored composite branch
  decays to cutoff (edge-slope extrapolation, floor 0) instead of
  holding a phantom clamped current. Validated against the analytic
  truth (full model data) and on a real EL84 SOVTEK L1/L2 pair.
- ML-138: NaN/Inf from model.ia does not reach THD — all three model
  paths (SE-DFT, PP-DFT, PP-model-Chebyshev) return None + WARNING.
- ML-140: ra is fitted per-curve in a local window around Q; the
  expanding fallback is checked for sparse grids. The strict model
  cross-check lives in test_amplifier.py::TestGmRaCrossValidation
  (now discriminates both pooling and a non-local fit).

Run:  py -m pytest tests/test_numerics_guards.py -v
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

from i18n_setup import available_locales

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.amplifier import (
    PushPullLoadLine,
    ResistiveLoadLine,
    composite_characteristic,
    compute_distortion_chebyshev_pp,
    compute_distortion_dft,
    compute_distortion_dft_pp,
    compute_distortion_chebyshev_pp_model,
    find_intersections_model,
    pp_distortion,
)
from lm19.amplifier.distortion import _interp_transfer_cutoff
from lm19.amplifier.stage_params import _numerical_gm_ra
from lm19.amplifier.distortion import interp_intersection
from lm19.tube_sim import quick_triode
from tests._real_data import EL84_SOVTEK_L1_PENT, EL84_SOVTEK_L2_PENT, load_points
from lm19.amplifier.constants import (
    CIRCUIT_PP,
)
from lm19.amp_engine import (
    WARN_PP_B_EDGE_EXTRAPOLATED,
)

# ── module local constants ──
# Truncation cut for tube B: drop curves at/below this Ug1 so the mirror
# swings well beyond B's measured range in the composite.
_TRUNC_UG1_V = -12.0
_PP_BIAS_V = -10.0
# The fix must shrink the beyond-edge composite error vs the clamp by at
# least this factor on analytic truth (probed: ~raw clamp error is a
# constant phantom current, extrapolation decays to truth's cutoff tail).
_MIN_ERR_IMPROVEMENT = 2.0


# ═══════════════════════════════════════════════════════════════════
#  ML-139 unit: cutoff-side extrapolation semantics
# ═══════════════════════════════════════════════════════════════════

class TestInterpTransferCutoff:

    _UG1 = np.array([-12.0, -10.0, -8.0, -6.0])
    _IA = np.array([2.0, 6.0, 12.0, 20.0])

    def test_inside_range_matches_np_interp(self):
        t = np.array([-11.0, -7.0])
        out = _interp_transfer_cutoff(t, self._UG1, self._IA)
        assert out == pytest.approx(np.interp(t, self._UG1, self._IA))

    def test_below_edge_decays_not_clamps(self):
        """3/2-law tail, hand-derived for the fixture (i0=2, i1=6,
        u0=-12, u1=-10):
          r  = (2/6)^(2/3)              = 0.48075
          Vc = (u0 - r*u1)/(1 - r)      = -13.8518 V
          k  = i0/(u0 - Vc)^1.5         = 0.79365
          Ia(-12.5) = k*(1.3518)^1.5    = 1.2474 mA
        Discriminates the clamp (2.0), a linear tail (1.0) and an
        exponent mutation."""
        out = _interp_transfer_cutoff(
            np.array([-12.5]), self._UG1, self._IA)
        assert out[0] == pytest.approx(1.2474, abs=0.01)
        assert out[0] < 2.0, "clamp would hold the edge value"

    def test_deep_below_reaches_exact_zero(self):
        out = _interp_transfer_cutoff(
            np.array([-20.0, -40.0]), self._UG1, self._IA)
        assert out[0] == 0.0
        assert out[1] == 0.0

    def test_never_negative(self):
        t = np.linspace(-50.0, -6.0, 200)
        out = _interp_transfer_cutoff(t, self._UG1, self._IA)
        assert np.all(out >= 0.0)

    def test_continuous_at_edge(self):
        eps = 1e-6
        lo = _interp_transfer_cutoff(
            np.array([-12.0 - eps]), self._UG1, self._IA)[0]
        hi = _interp_transfer_cutoff(
            np.array([-12.0]), self._UG1, self._IA)[0]
        assert lo == pytest.approx(hi, abs=1e-4)

    def test_above_edge_still_clamps(self):
        """Documented conservative choice: above the most positive
        measured Ug1 the current keeps the edge value (grid current /
        rising region — upward extrapolation would be speculative)."""
        out = _interp_transfer_cutoff(
            np.array([-2.0]), self._UG1, self._IA)
        assert out[0] == pytest.approx(20.0)

    def test_zero_edge_current_stays_zero(self):
        """Degenerate: tube already cut off AT the data edge — the tail
        is exactly 0 beyond it (and must not divide by zero in the
        3/2 fit)."""
        ia0 = np.array([0.0, 6.0, 12.0, 20.0])
        out = _interp_transfer_cutoff(
            np.array([-15.0, -12.5]), self._UG1, ia0)
        assert out[0] == 0.0
        assert out[1] == 0.0

    def test_negative_edge_slope_degrades_to_clamp(self):
        """Non-physical data (Ia rising toward cutoff) must not
        extrapolate growth."""
        ia_weird = np.array([5.0, 3.0, 12.0, 20.0])
        out = _interp_transfer_cutoff(
            np.array([-15.0]), self._UG1, ia_weird)
        assert out[0] == pytest.approx(5.0)


# ═══════════════════════════════════════════════════════════════════
#  ML-139 physics: analytic truth (model data) + real EL84 pair
# ═══════════════════════════════════════════════════════════════════

def _truncate_b(points: List[dict], cut_ug1: float) -> List[dict]:
    return [p for p in points if p["ug1"] > cut_ug1]


def _composite_errors(points_a, points_b_full, points_b_trunc, bias):
    """Return (err_fixed, err_clamp) vs the full-data truth, summed over
    the composite points whose mirror falls beyond the truncated edge."""
    truth = composite_characteristic(points_a, points_b_full, ug1_bias=bias)
    fixed = composite_characteristic(points_a, points_b_trunc, ug1_bias=bias)
    assert truth and fixed

    # clamp reference = plain np.interp on the truncated transfer curve
    from lm19.amplifier.distortion import _build_transfer_curve
    ug1_b, ia_b = _build_transfer_curve(points_b_trunc)
    edge = float(ug1_b[0])

    err_fixed = 0.0
    err_clamp = 0.0
    n_beyond = 0
    truth_by_ug1 = {round(p["ug1"], 3): p for p in truth}
    for p in fixed:
        mirror = 2.0 * bias - p["ug1"]
        tp = truth_by_ug1.get(round(p["ug1"], 3))
        if tp is None or mirror >= edge:
            continue
        n_beyond += 1
        clamp_ia_b = float(np.interp(mirror, ug1_b, ia_b))
        clamp_comp = p["ia_a"] - clamp_ia_b
        err_fixed += abs(p["ia_composite"] - tp["ia_composite"])
        err_clamp += abs(clamp_comp - tp["ia_composite"])
    assert n_beyond >= 3, "precondition: mirror must cross the cut edge"
    return err_fixed, err_clamp


class TestCompositePhysics:

    def test_analytic_truth_model_data(self):
        """Koren 12AU7: truncate tube B below −12 V; the composite from
        truncated data must be ≥2× closer to the full-data truth than
        the old clamp in the beyond-edge region."""
        _, pts_a = quick_triode("12AU7")
        _, pts_b = quick_triode("12AU7")
        pts_b_trunc = _truncate_b(pts_b, _TRUNC_UG1_V)
        assert len(pts_b_trunc) < len(pts_b), "precondition: cut happened"
        err_fixed, err_clamp = _composite_errors(
            pts_a, pts_b, pts_b_trunc, _PP_BIAS_V)
        assert err_fixed < err_clamp / _MIN_ERR_IMPROVEMENT, (
            f"extrapolation must beat clamp ≥{_MIN_ERR_IMPROVEMENT}×: "
            f"fixed={err_fixed:.3f} clamp={err_clamp:.3f} (mA·points)")

    def test_real_el84_pair(self):
        """Real mismatched pair (SOVTEK L1 + L2): same truncation
        experiment on measured data."""
        pts_a = load_points(EL84_SOVTEK_L1_PENT)
        pts_b = load_points(EL84_SOVTEK_L2_PENT)
        ug1_min = min(p["ug1"] for p in pts_b)
        cut = ug1_min + 4.0   # drop the deepest ~4 V of B's grid range
        pts_b_trunc = _truncate_b(pts_b, cut)
        assert len(pts_b_trunc) < len(pts_b)
        bias = cut + 1.0      # mirror crosses the cut edge
        err_fixed, err_clamp = _composite_errors(
            pts_a, pts_b, pts_b_trunc, bias)
        assert err_fixed <= err_clamp, (
            f"real data: extrapolation must not be worse than clamp: "
            f"fixed={err_fixed:.3f} clamp={err_clamp:.3f}")

    def test_cross_method_on_truncated_real_pair(self):
        """5-point vs Chebyshev THD within ×3 on the truncated real
        composite (cross-method x3 rule)."""
        pts_a = load_points(EL84_SOVTEK_L1_PENT)
        pts_b = load_points(EL84_SOVTEK_L2_PENT)
        ug1_min = min(p["ug1"] for p in pts_b)
        pts_b_trunc = _truncate_b(pts_b, ug1_min + 4.0)
        bias = ug1_min + 5.0
        ll = PushPullLoadLine(300.0, 8.0)
        d5 = pp_distortion(pts_a, ll, ug1_bias=bias, points_b=pts_b_trunc,
                           ug2_filter=250.0)
        dch = compute_distortion_chebyshev_pp(
            pts_a, ll, ug1_bias=bias, points_b=pts_b_trunc,
            ug2_filter=250.0)
        assert d5 is not None and dch is not None
        assert d5["thd"] > 0 and dch["thd"] > 0
        ratio = max(d5["thd"], dch["thd"]) / min(d5["thd"], dch["thd"])
        assert ratio < 3.0, f"cross-method THD ratio {ratio:.2f}"


# ═══════════════════════════════════════════════════════════════════
#  ML-138: NaN from model.ia must not reach THD
# ═══════════════════════════════════════════════════════════════════

class _NanRegionModel:
    """Wraps a fitted model; ia() returns NaN above ua_nan_v (bad fit
    region reached mid-swing)."""

    def __init__(self, inner, ua_nan_v: float) -> None:
        self._inner = inner
        self._ua_nan_v = ua_nan_v

    def ia(self, ua: float, ug1: float, ug2: float = 0.0) -> float:
        if ua > self._ua_nan_v:
            return float("nan")
        return self._inner.ia(ua, ug1, ug2)

    def ia_array(self, ua, ug1, ug2=0.0):
        from lm19.tube_model_base import model_ia_array
        base = model_ia_array(self._inner, np.asarray(ua, dtype=float),
                              ug1, ug2)
        return np.where(np.asarray(ua) > self._ua_nan_v, np.nan, base)


@pytest.fixture(scope="module")
def triode_12au7():
    model, pts = quick_triode("12AU7")
    return model, pts


class TestNanModelGuards:

    def test_se_dft_returns_none_not_nan(self, triode_12au7, caplog):
        model, _ = triode_12au7
        # NaN region at ua > 200 V: the cutoff half-cycle drives ua
        # toward Ub=250 V on a resistive line — samples DO hit it.
        nan_model = _NanRegionModel(model, ua_nan_v=200.0)
        ll = ResistiveLoadLine(250.0, 10.0)
        with caplog.at_level(logging.WARNING,
                             logger="lm19.amplifier.distortion"):
            d = compute_distortion_dft(
                nan_model, ll, ug1_bias=-8.0, half_swing=5.0)
        assert d is None, "NaN model samples must fail loudly, not NaN THD"
        assert any("non-finite" in r.message for r in caplog.records)

    def test_se_dft_clean_model_still_works(self, triode_12au7):
        """Negative control: the guard must not reject finite models."""
        model, _ = triode_12au7
        ll = ResistiveLoadLine(250.0, 10.0)
        d = compute_distortion_dft(model, ll, ug1_bias=-8.0, half_swing=5.0)
        assert d is not None
        assert math.isfinite(d["thd"])

    def test_pp_dft_returns_none_not_nan(self, triode_12au7, caplog):
        model, _ = triode_12au7
        nan_model = _NanRegionModel(model, ua_nan_v=270.0)
        ll = PushPullLoadLine(250.0, 8.0)
        with caplog.at_level(logging.WARNING,
                             logger="lm19.amplifier.distortion"):
            d = compute_distortion_dft_pp(
                nan_model, ll, ug1_bias=-8.0, half_swing=6.0)
        assert d is None
        assert any("non-finite" in r.message for r in caplog.records)

    def test_pp_model_chebyshev_returns_none_not_nan(self, triode_12au7,
                                                     caplog):
        model, _ = triode_12au7
        nan_model = _NanRegionModel(model, ua_nan_v=270.0)
        ll = PushPullLoadLine(250.0, 8.0)
        with caplog.at_level(logging.WARNING,
                             logger="lm19.amplifier.distortion"):
            d = compute_distortion_chebyshev_pp_model(
                nan_model, ll, ug1_bias=-8.0, half_swing=6.0)
        assert d is None
        assert any("non-finite" in r.message for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════
#  ML-140: sparse-scan widening fallback
# ═══════════════════════════════════════════════════════════════════

class TestRaWindowFallback:

    def test_sparse_curve_widens_window(self, triode_12au7):
        """Per-curve: 2 points inside ±RA_WINDOW around Q + 3 far away —
        the widening ladder must reach the whole curve, not return None."""
        model, _ = triode_12au7
        ll = ResistiveLoadLine(250.0, 10.0)
        ug1_levels = [-9.0, -8.0, -7.0]
        isects = find_intersections_model(model, ll, ug1_levels)
        q = interp_intersection(isects, -8.0)
        assert q is not None
        ua_q = q["ua"]
        # 2 points per curve inside the Q window (gm is fine, the ra
        # slope is not) + 3 far points: the ladder must widen.
        pts = []
        for u in ug1_levels:
            for dua in (-10.0, 10.0, 180.0, 200.0, 220.0):
                ua = ua_q + dua
                pts.append({"ua": ua, "ug1": u,
                            "ia": max(0.0, model.ia(ua, u))})
        r = _numerical_gm_ra(pts, isects, ug1_bias=-8.0)
        assert r is not None, "sparse scan must fall back to wider window"
        assert r["ra"] > 0


class TestBExtrapVisibility:
    """ML-139 visibility (failure-visibility rule): the user must SEE when the
    analyzed window consumes extrapolated (non-measured) Ia_B values."""

    def _pair(self):
        pts_a = load_points(EL84_SOVTEK_L1_PENT)
        pts_b = load_points(EL84_SOVTEK_L2_PENT)
        return pts_a, pts_b

    def test_matched_depth_is_silent(self):
        """No-spam pin: equal-depth pair with auto swing keeps every
        sampled point data-backed — span 0, no flag."""
        pts_a, pts_b = self._pair()
        ll = PushPullLoadLine(300.0, 8.0)
        d = pp_distortion(pts_a, ll, ug1_bias=-8.0, points_b=pts_b,
                          ug2_filter=250.0)
        assert d["b_extrapolation_span_v"] == 0.0
        assert d["b_extrapolation_significant"] is False

    def test_truncated_b_flags_significant(self):
        pts_a, pts_b = self._pair()
        cut = min(p["ug1"] for p in pts_b) + 4.0
        pts_b_tr = _truncate_b(pts_b, cut)
        ll = PushPullLoadLine(300.0, 8.0)
        d = pp_distortion(pts_a, ll, ug1_bias=-8.0, points_b=pts_b_tr,
                          ug2_filter=250.0)
        assert d["b_extrapolation_span_v"] > 0
        assert d["b_edge_ia_fraction"] > 0.15
        assert d["b_extrapolation_significant"] is True

    def test_chebyshev_twin_carries_flags(self):
        pts_a, pts_b = self._pair()
        cut = min(p["ug1"] for p in pts_b) + 4.0
        pts_b_tr = _truncate_b(pts_b, cut)
        ll = PushPullLoadLine(300.0, 8.0)
        d = compute_distortion_chebyshev_pp(
            pts_a, ll, ug1_bias=-8.0, points_b=pts_b_tr, ug2_filter=250.0)
        assert d["b_extrapolation_span_v"] > 0
        assert d["b_extrapolation_significant"] is True

    def test_edge_near_cutoff_notice_without_warn(self):
        """Kills a 'significant = span > 0' mutation: B measured almost
        to cutoff — extrapolation active but the tail is negligible."""
        model, pts_a = quick_triode("12AU7")
        _, pts_b = quick_triode("12AU7")
        # 12AU7 @ ua_ref=250: cutoff ~ -11.6 V; edge -13 -> edge Ia ~ 0
        pts_b_tr = _truncate_b(pts_b, -13.0)
        ll = PushPullLoadLine(250.0, 8.0)
        d = pp_distortion(pts_a, ll, ug1_bias=-9.0, points_b=pts_b_tr,
                          half_swing=6.0)
        assert d is not None
        assert d["b_extrapolation_span_v"] > 0
        assert d["b_edge_ia_fraction"] < 0.15
        assert d["b_extrapolation_significant"] is False

    def test_criterion_is_signal_relative_not_grid_relative(self):
        """Formula discriminator for the significance denominator: the
        fraction must be ia(B edge) / b1_estimate of the ANALYZED window
        ((i_max−i_min)/2) — a grid-max denominator would depend on how
        far the positive Ug1 side was scanned (probed live: the same cut
        gave 0.36 on real scans ending at −1 V and 0.12 on the synthetic
        grid reaching 0 V)."""
        from lm19.tube_sim import quick_pentode
        from lm19.amplifier.distortion import _build_transfer_curve
        _, pts = quick_pentode("EL84")
        cut = min(p["ug1"] for p in pts) + 4.0
        pts_b_tr = [p for p in pts if p["ug1"] > cut]
        ll = PushPullLoadLine(300.0, 8.0)
        d = pp_distortion(pts, ll, ug1_bias=-8.0, points_b=pts_b_tr,
                          ug2_filter=250.0)
        assert d["b_extrapolation_span_v"] > 0
        ug1_b, ia_b = _build_transfer_curve(pts_b_tr, ug2_filter=250.0,
                                            ua_ref=ll.ub)
        expected = float(ia_b[0]) / ((d["i_max"] - d["i_min"]) / 2.0)
        assert d["b_edge_ia_fraction"] == pytest.approx(expected, rel=1e-6)

    def test_engine_appends_warn_code(self):
        from lm19.amp_engine import AmplifierEngine, AmpParams
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        a = [dict(p, series_id=1) for p in pts]
        # +5 V cut: the synthetic grid reaches ug1=0 where the transfer
        # max is far larger than in real scans (real ones stop at -1 V),
        # so a deeper cut is needed to push the edge fraction over the
        # warn threshold (probed: +4 -> 0.12, +5 -> 0.21).
        cut = min(p["ug1"] for p in pts) + 5.0
        b = [dict(p, series_id=2) for p in pts if p["ug1"] > cut]
        eng = AmplifierEngine()
        eng.set_data(a + b, is_triode=False)
        params = AmpParams(ub=300.0, ra=8.0, ug1_bias=-8.0, pa_max=12.0,
                           ug2_filter=250.0, circuit=CIRCUIT_PP,
                           pp_matched=False, pp_tube_b_sid=2)
        result = eng.analyze(params)
        assert result.pp_dist is not None
        assert result.pp_dist["b_extrapolation_significant"] is True
        sr = result.per_source["measurements"]
        codes = [w.get("code") for w in sr.warnings]
        assert WARN_PP_B_EDGE_EXTRAPOLATED in codes

    @pytest.mark.parametrize("locale", available_locales())
    def test_i18n_keys_exist(self, locale):
        import json
        root = Path(__file__).resolve().parents[1]
        data = json.loads((root / "locales" / f"{locale}.json")
                          .read_text(encoding="utf-8"))
        amp = data["amp"]
        assert "b_extrap_notice" in amp
        assert "warn_pp_b_edge_extrapolated" in amp


class TestRaPooledContamination:

    def test_pa_truncated_curves_do_not_poison_ra(self, triode_12au7):
        """THE ML-140 discriminator: Pa protection asymmetrically cuts
        the curves (hot curve keeps only low Ua, cold curve only high
        Ua). A pooled fit runs one line from the hot-low cluster (high
        Ia) to the cold-high cluster (low Ia) — a strongly NEGATIVE
        contaminated slope; per-curve slopes are all physical. On the
        symmetric full grid pooled≈per-curve (equal values away from the
        asymmetry — checklist trap)."""
        from lm19.amplifier.stage_params import model_gm_ra
        model, _ = triode_12au7
        ll = ResistiveLoadLine(250.0, 10.0)
        ug1_levels = [-10.0, -8.0, -6.0]
        isects = find_intersections_model(model, ll, ug1_levels)
        q = interp_intersection(isects, -8.0)
        assert q is not None
        ua_q = q["ua"]

        pts = []
        # hot curve (−6): Pa protection → only Ua BELOW ua_q
        for dua in (-40.0, -30.0, -20.0, -10.0):
            ua = ua_q + dua
            pts.append({"ua": ua, "ug1": -6.0,
                        "ia": max(0.0, model.ia(ua, -6.0))})
        # bias curve (−8): spans the window
        for dua in (-40.0, -20.0, 0.0, 20.0, 40.0):
            ua = ua_q + dua
            pts.append({"ua": ua, "ug1": -8.0,
                        "ia": max(0.0, model.ia(ua, -8.0))})
        # cold curve (−10): only Ua ABOVE ua_q
        for dua in (10.0, 20.0, 30.0, 40.0):
            ua = ua_q + dua
            pts.append({"ua": ua, "ug1": -10.0,
                        "ia": max(0.0, model.ia(ua, -10.0))})

        r_num = _numerical_gm_ra(pts, isects, ug1_bias=-8.0)
        assert r_num is not None
        r_model = model_gm_ra(model, ua_q=ua_q, ug1_q=-8.0)
        assert r_model is not None
        assert r_num["ra"] == pytest.approx(r_model["ra"], rel=0.35), (
            f"per-curve ra={r_num['ra']:.2f} must track model "
            f"ra={r_model['ra']:.2f} despite asymmetric truncation")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
