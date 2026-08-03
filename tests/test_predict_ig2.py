"""Noise-stability regression for ``_predict_ig2`` quadratic extrapolation.

Without a baseline-width guard the only check on the quadratic was
``_UA_DELTA_MIN = 0.01 V`` — a divide-by-zero test, not a noise
control. With three points spaced ~0.011 V apart, the curvature
coefficient ``a = (s2 - s1) / dua_total`` would amplify ±0.1 mA Ig2
noise into a prediction of ±15 000 mA — either firing a bogus
"ig2_predict" break or silently failing the predictive guard
(negative prediction passes the ``> ig2_limit`` check).

Guards in place:

- ``_QUADRATIC_MIN_BASELINE_V`` (0.5 V) — quadratic only used when
  the 3-point Ua span is wide enough; below threshold falls back to
  linear extrapolation.
- Negative quadratic prediction → linear fallback (Ig2 ≥ 0 physically).
- ``_PREDICT_MAX_OVER_LIMIT`` (5×) — prediction above this multiple
  of ``ig2_limit`` returns ``None``; the caller's ``ig2_est is not
  None`` guard then skips the predictive break.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from lm19.scan.refine import _predict_ig2
from lm19.scan.settings import (
    _PREDICT_MAX_OVER_LIMIT,
    _QUADRATIC_MIN_BASELINE_V,
    _UA_DELTA_MIN,
)


# ── helpers ──────────────────────────────────────────────────────────

def _pt(ua: float, ig2: float) -> Dict:
    return {"ua": ua, "ig2": ig2, "ia": 1.0, "ug1": 0.0, "ug2": 200.0}


def _linear_pred(pts: List[Dict], next_ua: float) -> float:
    """Manual linear extrapolation from last 2 points (for assertion)."""
    ua2, ig2_2 = pts[-1]["ua"], pts[-1]["ig2"]
    ua1, ig2_1 = pts[-2]["ua"], pts[-2]["ig2"]
    slope = (ig2_2 - ig2_1) / (ua2 - ua1)
    return ig2_2 + slope * (next_ua - ua2)


# ── Constant-relationship pin ────────────────────────────────────────

class TestConstantsHeldInvariant:
    """If someone bumps ``_UA_DELTA_MIN`` someday, the new baseline
    must still be ≥10× wider — otherwise the quadratic guard collapses
    back to a divide-by-zero check and the noise-amplification bug
    returns silently.
    """

    def test_baseline_at_least_10x_delta_min(self):
        assert _QUADRATIC_MIN_BASELINE_V >= 10 * _UA_DELTA_MIN, (
            f"_QUADRATIC_MIN_BASELINE_V ({_QUADRATIC_MIN_BASELINE_V}) must "
            f"be at least 10× _UA_DELTA_MIN ({_UA_DELTA_MIN}) to keep the "
            f"quadratic-fit noise amplification bounded. If the baseline "
            f"shrinks, the original bug returns: ±0.1 mA noise → ±15 A "
            f"prediction — bogus 'ig2_predict' breaks or missed real "
            f"Ig2 overshoots."
        )

    def test_max_over_limit_is_meaningful_multiplier(self):
        """Sanity range: 2-20×. Below 2× we'd reject realistic
        single-point overshoots; above 20× we let true blow-ups through."""
        assert 2.0 <= _PREDICT_MAX_OVER_LIMIT <= 20.0


# ── Narrow-baseline regression: the pathological case ────────────────

class TestNarrowBaselineFallsBackToLinear:
    """Pathological scenario: three points within ~0.011 V — the
    quadratic must be rejected and the linear fallback used."""

    def test_pathological_narrow_baseline_uses_linear_not_quadratic(self):
        """Three points with dua_total just above the 0.01 V
        divide-by-zero threshold. Without a wider noise-control
        baseline the quadratic engages and amplifies noise into a
        ~15 000 mA prediction; the 0.5 V baseline guard catches this.

        Spacing chosen so the divide-by-zero guard
        (``dua >= _UA_DELTA_MIN = 0.01``) passes but the noise-control
        guard (``dua_total >= _QUADRATIC_MIN_BASELINE_V = 0.5``) fails.
        """
        pts = [_pt(200.00, 5.00),
               _pt(200.05, 5.05),
               _pt(200.11, 5.07)]
        # dua=0.06 (>0.01 ✓), dua0=0.05 (>0.01 ✓), dua_total=0.11 (<0.5 ✗)
        result = _predict_ig2(pts, next_ua=195.0)
        # Expected: linear from last 2 points, then clamped to 0
        # slope = (5.07 - 5.05)/0.06 ≈ 0.333 mA/V
        # at 195: dt = -5.11 → 5.07 + 0.333*(-5.11) = 3.37 mA
        expected = max(_linear_pred(pts, 195.0), 0.0)
        assert result == pytest.approx(expected, abs=0.1)
        # Sanity: result is bounded — not the multi-thousand-mA blow-up
        # the unguarded quadratic produces on this data.
        assert 0 <= result <= 100, (
            f"Quadratic noise blow-up returned {result} mA — baseline "
            f"guard regressed."
        )

    def test_extreme_narrow_baseline_synthetic_noise(self):
        """Three points within 0.011 V with realistic ±0.1 mA Ig2
        noise. The unguarded quadratic produced predictions on the
        order of ±15 000 mA; the result must now be physically
        bounded (linear extrapolation, clamped) or ``None``."""
        pts = [_pt(200.000, 5.00),
               _pt(200.005, 5.05),
               _pt(200.011, 5.07)]
        # dua=0.006 (<0.01 ✗) — this trips the OLD delta_min guard
        # → returns None. That's also the correct "no prediction"
        # outcome: the data is too sparse to trust ANY extrapolation.
        result = _predict_ig2(pts, next_ua=195.0, ig2_limit=10.0)
        assert result is None, (
            f"With ua spacing of 0.005-0.006 V (below _UA_DELTA_MIN), "
            f"prediction must be None. Got {result}."
        )

    def test_baseline_at_threshold_uses_quadratic(self):
        """At exactly _QUADRATIC_MIN_BASELINE_V (0.5 V), quadratic
        should engage. Just below, linear. Boundary check."""
        # At threshold: 3 collinear points spaced 0.25V each → dua_total=0.5V
        pts = [_pt(200.0, 4.0), _pt(200.25, 5.0), _pt(200.5, 6.0)]
        result = _predict_ig2(pts, next_ua=200.75)
        # Truly linear data → quadratic gives same result as linear
        assert result == pytest.approx(7.0, abs=0.01)

    def test_baseline_below_threshold_uses_linear(self):
        """Just below threshold — quadratic rejected even with clean data."""
        # dua_total = 0.4V < 0.5V threshold
        pts = [_pt(200.0, 4.0), _pt(200.2, 5.0), _pt(200.4, 6.0)]
        result = _predict_ig2(pts, next_ua=200.6)
        # Same numerical result as quadratic for collinear data, but
        # we verify the fallback path was taken by checking that even
        # NON-collinear data here uses linear (next test does that).
        assert result == pytest.approx(7.0, abs=0.01)

    def test_narrow_baseline_with_curvature_uses_linear_not_quad(self):
        """Stronger check: if data has curvature but baseline is too
        narrow, we MUST get linear — quadratic would predict differently."""
        # 3 points spaced 0.1V (well below 0.5V threshold), strong curvature
        pts = [_pt(200.0, 1.0), _pt(200.1, 2.0), _pt(200.2, 5.0)]
        result = _predict_ig2(pts, next_ua=200.5)
        # Linear from last 2: slope = (5-2)/0.1 = 30 mA/V
        # At 200.5: 5 + 30*0.3 = 14
        # Quadratic (if it had fired): a = (30 - 10)/0.2 = 100 mA/V²
        #   prediction = 5 + 30*0.3 + 100*0.09 = 23 — clearly different
        assert result == pytest.approx(14.0, abs=0.5), (
            f"Narrow-baseline curved data: expected linear ≈14 mA, got "
            f"{result}. If ≈23, quadratic fired despite narrow baseline "
            f"— baseline guard regressed."
        )


# ── Sanity clamp: negative prediction ───────────────────────────────

class TestNegativePredictionClamp:
    """Quadratic with steep negative curvature can predict Ig2 < 0
    (unphysical: screen current can't go negative). V1 falls back
    to linear, which also gets clamped to 0 if itself negative."""

    def test_quadratic_negative_falls_back_to_linear_then_clamps(self):
        # Concave-down ig2 trajectory (decreasing curvature) — quadratic
        # extrapolates well below zero, while linear from last 2 is also
        # negative. Both should clamp to 0.
        pts = [_pt(200.0, 10.0), _pt(180.0, 5.0), _pt(160.0, 1.0)]
        # dua_total = 200 - 160 = 40 V → quadratic engages
        # s1 = (5-10)/(180-200) = 0.25 mA/V
        # s2 = (1-5)/(160-180) = 0.20 mA/V
        # a = (0.20 - 0.25)/(160-200) = 0.00125 mA/V²
        # at next_ua=140: dt = -20, ig2_2 + s2*dt + a*dt^2 = 1 + 0.20*(-20) + 0.00125*400 = -2.5
        # → negative → fall back to linear, slope=0.20, at 140: 1 + 0.2*(-20) = -3 → clamp to 0
        result = _predict_ig2(pts, next_ua=140.0)
        assert result == 0.0, (
            f"Negative quadratic must fall back through linear (also "
            f"negative) to 0.0 (Ig2 floor). Got {result}."
        )

    def test_linear_two_points_negative_clamps_to_zero(self):
        """2-point linear path also clamps negatives (V1 added max(_, 0))."""
        pts = [_pt(200.0, 5.0), _pt(180.0, 2.0)]
        # slope = (2-5)/(180-200) = 0.15 mA/V
        # at next_ua=140: 2 + 0.15*(140-180) = 2 + 0.15*(-40) = -4.0 → clamp to 0
        result = _predict_ig2(pts, next_ua=140.0)
        assert result == 0.0


# ── Sanity clamp: prediction far above ig2_limit ─────────────────────

class TestExcessivePredictionReturnsNone:
    """When the quadratic predicts >5× the Ig2 limit, the result is
    extrapolation blow-up — return None so the caller's ``is not None``
    check skips the bogus break."""

    def test_blow_up_above_5x_limit_returns_none(self):
        """Construct points where curvature drives a huge prediction.
        With ig2_limit=10 mA, threshold = 50 mA above which → None."""
        pts = [_pt(200.0, 1.0), _pt(180.0, 2.0), _pt(160.0, 8.0)]
        # s1=(2-1)/-20=-0.05, s2=(8-2)/-20=-0.30, a=(-0.30+0.05)/-40=0.00625
        # at next_ua=100: dt=-60, 8 + (-0.30)*(-60) + 0.00625*3600 = 8 + 18 + 22.5 = 48.5
        # 48.5 mA — just below 5×10=50 limit → keeps quadratic
        result = _predict_ig2(pts, next_ua=100.0, ig2_limit=10.0)
        assert result is not None
        assert result == pytest.approx(48.5, abs=0.5)

    def test_above_5x_limit_returns_none(self):
        """Push slightly further to cross the 5× ceiling."""
        pts = [_pt(200.0, 1.0), _pt(180.0, 2.0), _pt(160.0, 8.0)]
        # at next_ua=80, prediction ≈ 8 + (-0.3)*(-80) + 0.00625*6400 = 72 — above 50 ceiling
        result = _predict_ig2(pts, next_ua=80.0, ig2_limit=10.0)
        assert result is None, (
            f"Prediction ~72 mA with ig2_limit=10 mA exceeds "
            f"5×ig2_limit=50 mA — must return None. Got {result}."
        )

    def test_no_limit_no_clamp(self):
        """ig2_limit=0 (the default) disables the absolute clamp.
        Confirms the 2-argument call form still works when callers
        don't pass ``ig2_limit``."""
        pts = [_pt(200.0, 1.0), _pt(180.0, 2.0), _pt(160.0, 8.0)]
        # Same data as above test — without limit, no clipping
        result = _predict_ig2(pts, next_ua=80.0)  # no ig2_limit
        assert result is not None
        # 72 mA is the actual quadratic prediction
        assert result == pytest.approx(72.0, abs=2.0)


# ── Normal-spacing data: quadratic prediction unchanged ──────────────

class TestNormalSpacing:
    """Normal-spacing (5-25 V steps) is covered by the 7 cases above;
    this case pins the realistic scan-grid path so a future tweak to
    the quadratic logic can't silently break ordinary pentode
    down-sweeps."""

    def test_typical_pentode_down_sweep(self):
        """Realistic curve: Ig2 rises sharply as Ua approaches Ug2.
        Spacing 20 V → quadratic engages, prediction matches hand-fit."""
        pts = [_pt(280.0, 0.5), _pt(260.0, 1.5), _pt(240.0, 4.0)]
        # s1=(1.5-0.5)/-20=-0.05, s2=(4.0-1.5)/-20=-0.125
        # a=(s2-s1)/(240-280) = (-0.125+0.05)/-40 = 0.001875
        # at 220: dt=-20, 4 + (-0.125)*(-20) + 0.001875*400 = 4 + 2.5 + 0.75 = 7.25
        result = _predict_ig2(pts, next_ua=220.0, ig2_limit=20.0)
        assert result == pytest.approx(7.25, abs=0.1)
        # Sanity: prediction is below the 5×limit clamp ceiling
        assert result < 5 * 20.0


# ── Call-site integration: sweepers passes ig2_limit ─────────────────

class TestCallSitePassesIg2Limit:
    """Pin: ``sweepers._sweep_ug2_independent`` invokes ``_predict_ig2``
    with the ``ig2_limit`` kwarg. Without this, the absolute clamp is
    inert in production (the unit tests above pass, but the user-
    visible blow-up is not clipped)."""

    def test_sweepers_call_passes_ig2_limit(self):
        import inspect
        from lm19.scan import sweepers
        src = inspect.getsource(sweepers)
        # Look for "_predict_ig2(...ig2_limit=ctx.ig2_limit..." pattern.
        # Use a tolerant check that survives reformatting.
        assert "_predict_ig2(" in src, "Call site removed?"
        # Find the call and verify ig2_limit kwarg is present in args.
        # (Simple substring check — acceptable for a pin.)
        assert "ig2_limit=ctx.ig2_limit" in src, (
            "_sweep_ug2_independent must pass ig2_limit=ctx.ig2_limit "
            "to _predict_ig2 so the absolute-clamp guard is active in "
            "production. Without it, the unit-test guard exists but "
            "the call site provides no limit → no clamp."
        )
