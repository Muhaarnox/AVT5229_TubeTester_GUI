"""Pre-validation physics fixes: transfer-at-supply,
dynamic AC-reach ceiling, fundamental-power / average-Pa fields.

1. ``_build_transfer_curve(ua_ref=Ub)`` — Ia interpolated AT the supply
   voltage (standard transfer-characteristic quick method) instead of the
   legacy mean over the whole Ua sweep, which understated triode Iq ~2.4×
   and skewed Pa checks / p_classA_w / amp_class on the PP data path.
2. ``find_intersections_model`` transformer/PP branches extend the Ua scan
   to the AC line's zero-current reach (Ua_q + Iq·Ra) — a fixed caller
   ceiling silently truncated deep-bias intersections.
3. ``pout_fund_mw`` (SE + PP DFT) — fundamental-only power for
   like-for-like comparison with external references (LTspice .four).
4. ``pa_avg_mw`` (PP DFT) — per-tube average dissipation under signal;
   informational field only, the Pa constraint still checks the Q-point.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.amplifier import (
    PushPullLoadLine,
    ResistiveLoadLine,
    TransformerLoadLine,
    compute_distortion_dft,
    compute_distortion_dft_pp,
    find_intersections_model,
    pp_distortion,
)
from lm19.amplifier.distortion import _build_transfer_curve
from lm19.tube_sim import load_model, quick_triode

# ── module local constants ──
_UB = 250.0
# Synthetic strongly-Ua-dependent curve: Ia = 0.1·Ua at every Ug1 level —
# average over Ua 0..300 gives 15 mA, Ia(250) gives 25 mA: cleanly separates
# the two definitions.
_SLOPE_MA_PER_V = 0.1


def _ua_dependent_points() -> list:
    pts = []
    for ug1 in (-12.0, -10.0, -8.0, -6.0):
        for ua in np.arange(0.0, 301.0, 50.0):
            pts.append({"ua": float(ua), "ug1": ug1,
                        "ia": _SLOPE_MA_PER_V * float(ua), "ug2": 0.0})
    return pts


class TestTransferAtSupply:
    def test_ua_ref_interpolates_at_supply(self) -> None:
        """With ua_ref the curve is Ia(ua_ref), NOT the sweep average —
        fails under a revert to averaging (25 vs 15 mA)."""
        _, ia = _build_transfer_curve(_ua_dependent_points(), ua_ref=_UB)
        assert np.allclose(ia, _SLOPE_MA_PER_V * _UB)

    def test_legacy_none_keeps_averaging(self) -> None:
        """API compat: ua_ref=None preserves the documented mean."""
        _, ia = _build_transfer_curve(_ua_dependent_points(), ua_ref=None)
        expected = _SLOPE_MA_PER_V * float(
            np.mean(np.arange(0.0, 301.0, 50.0)))
        assert np.allclose(ia, expected)

    def test_pp_iq_matches_true_supply_current_triode(self) -> None:
        """End-to-end: pp_distortion's iq_per_tube == Ia(Ub, bias) from
        the raw 12AU7 curves. The averaged definition gave ~0.41× of it —
        understating Pa = Ub·Iq by ~2.4× (the audit's flagged bias)."""
        _, pts = quick_triode("12AU7")
        ll = PushPullLoadLine(ub=_UB, ra_aa=8.0)
        d = pp_distortion(pts, ll, ug1_bias=-10.0, half_swing=5.0)
        assert d is not None
        lv: dict = {}
        for p in pts:
            lv.setdefault(round(p["ug1"], 1), []).append((p["ua"], p["ia"]))
        curve = sorted(lv[-10.0])
        true_iq = float(np.interp(
            _UB, [a for a, _ in curve], [b for _, b in curve]))
        assert d["iq_per_tube"] == pytest.approx(true_iq, rel=1e-6)
        assert true_iq > 5.0                    # sanity: tube conducts


class TestDynamicAcReach:
    def test_deep_bias_isects_beyond_fixed_ceiling(self) -> None:
        """Hot Iq × big Ra_ac: the AC line reaches Ua_q + Iq·Ra ≈ 700 V.
        The scan must extend past the caller's 500 V ceiling — under a
        revert (fixed linspace top) the family tops out at 500 V."""
        el84 = load_model("EL84")
        ll = TransformerLoadLine(ub=250.0, ra_dc=0.1, ra_ac=10.0)
        isects = find_intersections_model(
            el84, ll, [-12.0, -10.0, -8.0, -6.0, -4.0], ug2=250.0,
            ua_range=(1.0, 500.0), ug1_bias=-6.5)
        assert len(isects) >= 4
        assert max(p["ua"] for p in isects) > 500.0

    def test_resistive_range_untouched(self) -> None:
        """No ug1_bias / resistive line → the passed range is respected."""
        el84 = load_model("EL84")
        ll = ResistiveLoadLine(250.0, 5.0)
        isects = find_intersections_model(
            el84, ll, [-8.0, -6.0], ug2=250.0, ua_range=(1.0, 400.0))
        assert all(p["ua"] <= 400.0 for p in isects)


class TestFundamentalAndAvgPaFields:
    def test_se_fund_matches_peak_power_for_clean_sine(self) -> None:
        """Small swing → near-sinusoidal waveforms → the fundamental-only
        power must agree with the peak-based estimate within a few %."""
        el84 = load_model("EL84")
        ll = ResistiveLoadLine(300.0, 5.0)
        d = compute_distortion_dft(
            el84, ll, ug1_bias=-7.0, half_swing=1.0, ug2=250.0, ub=300.0)
        assert d is not None
        assert d["pout_fund_mw"] == pytest.approx(d["pout_mw"], rel=0.05)

    def test_se_fund_diverges_from_peak_under_distortion(self) -> None:
        """Hard drive: compression flattens the waveform peaks, so the
        peak-to-peak estimate UNDERSTATES the fundamental (probed ratio
        1.39 at THD 20%). The two must diverge measurably yet stay in a
        physical band — that divergence is exactly why the field exists
        (external references report the fundamental)."""
        el84 = load_model("EL84")
        ll = ResistiveLoadLine(300.0, 5.0)
        d = compute_distortion_dft(
            el84, ll, ug1_bias=-9.0, half_swing=8.0, ug2=250.0, ub=300.0)
        assert d is not None
        assert d["thd"] > 5.0                   # genuinely distorted
        ratio = d["pout_fund_mw"] / d["pout_mw"]
        assert 1.05 < ratio < 1.6, ratio

    def test_pp_fields_present_and_physical(self) -> None:
        el84 = load_model("EL84")
        ll = PushPullLoadLine(ub=300.0, ra_aa=8.0, ra_dc=0.1)
        d = compute_distortion_dft_pp(
            el84, ll, ug1_bias=-11.0, half_swing=9.0, ug2=300.0)
        assert d is not None
        # Compression → peak-based understates the fundamental (see the
        # SE test above); probed PP ratio 1.22 at this operating point.
        assert 0 < d["pout_fund_mw"] <= d["pout_mw"] * 1.4
        # Per-tube dissipation under signal: positive and bounded by the
        # supply × peak per-tube current (loose physical ceiling).
        assert 0 < d["pa_avg_mw"] < 300.0 * (d["iq_per_tube"] + d["i_max"])

    def test_pp_class_a_small_signal_pa_avg_near_quiescent(self) -> None:
        """Class A, tiny swing: average dissipation ≈ Q dissipation
        (Ub×Iq) — the defining property of class A."""
        el84 = load_model("EL84")
        ll = PushPullLoadLine(ub=300.0, ra_aa=8.0, ra_dc=0.1)
        d = compute_distortion_dft_pp(
            el84, ll, ug1_bias=-9.0, half_swing=0.5, ug2=300.0)
        assert d is not None
        pa_q = 300.0 * d["iq_per_tube"]
        assert d["pa_avg_mw"] == pytest.approx(pa_q, rel=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
