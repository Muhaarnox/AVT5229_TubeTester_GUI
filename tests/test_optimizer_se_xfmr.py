"""#10 — se_xfmr measurements path: model-based Q-point load line.

A transformer-coupled SE stage sits at Ua ≈ Ub (near-zero DC winding drop)
and swings the anode ABOVE the supply on the AC line through the Q-point —
where scan data does not exist. The naive straight line from Ub put the
operating point far below the supply. With a fitted model the optimizer now
builds intersections from the model (correct Q-point + extrapolation past
measured Ua); without one it keeps the naive line and surfaces
``warning=OPT_WARN_SE_XFMR_NO_MODEL``.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.optimizer import OptimizerConstraints, optimize_measurements
from lm19.tube_sim import quick_pentode, quick_triode
from lm19.amplifier.constants import (
    CIRCUIT_CF,
    CIRCUIT_SE,
    CIRCUIT_SE_XFMR,
    HD_METHOD_DFT,
)
from lm19.optimizer import (
    OPT_WARN_DFT_NO_MODEL_FALLBACK,
    OPT_WARN_SE_XFMR_NO_MODEL,
)

# ── module local constants ──
# The DC winding drop is Iq*ra_dc ≈ 50 mA * 0.1 kΩ = 5 V → the Q-point of a
# transformer stage must sit within a few percent of the supply. The naive
# straight line from Ub intersects the bias curve FAR below Ub (typically
# 40-70% of it), so this bound cleanly separates the two behaviours.
_Q_POINT_MIN_FRACTION_OF_UB = 0.9
_MAX_SANE_THD_PCT = 70.0


def _xfmr_constraints(**over) -> OptimizerConstraints:
    base = dict(
        circuit=CIRCUIT_SE_XFMR, ra_dc=0.1,
        ug1_range=(-10.0, -4.0), ra_range=(2.0, 8.0),
        ug1_steps=6, ra_steps=6, pa_max_w=12.0,
    )
    base.update(over)
    return OptimizerConstraints(**base)


class TestXfmrModelIsects:
    """With a fitted model the Q-point lands at Ua ≈ Ub (#10 fix)."""

    def test_pentode_q_point_at_supply(self) -> None:
        model, pts = quick_pentode("EL84")
        c = _xfmr_constraints()
        r = optimize_measurements(pts, ub=300.0, constraints=c,
                                  ug2_filter=250.0, model=model)
        assert r.error is None
        assert r.best is not None
        # Transformer stage: Ua_q ≈ Ub. Naive line puts it far below —
        # this is the discriminating assertion (revert → fails).
        assert r.best.ua_0 >= 300.0 * _Q_POINT_MIN_FRACTION_OF_UB
        assert r.best.thd < _MAX_SANE_THD_PCT
        assert r.best.pout_mw > 0
        assert r.warning is None            # model present → no warning

    def test_triode_q_point_at_supply(self) -> None:
        model, pts = quick_triode("12AU7")
        c = _xfmr_constraints(
            ra_dc=0.05, ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=8, ra_steps=8, pa_max_w=2.75,
        )
        r = optimize_measurements(pts, ub=250.0, constraints=c, model=model)
        assert r.error is None
        assert r.best is not None
        assert r.best.ua_0 >= 250.0 * _Q_POINT_MIN_FRACTION_OF_UB
        assert r.best.thd < _MAX_SANE_THD_PCT

    def test_pentode_grid_not_starved(self) -> None:
        """The historical regression: correct AC line + data-only
        intersections → too few isects → no_valid_points on pentodes.
        The model path must not starve the grid."""
        model, pts = quick_pentode("EL84")
        c = _xfmr_constraints()
        r = optimize_measurements(pts, ub=300.0, constraints=c,
                                  ug2_filter=250.0, model=model)
        valid = [p for p in r.grid_points if p.valid]
        # A healthy sweep yields many feasible points, not a handful.
        assert len(valid) > 10

    def test_swing_phase_uses_per_ug1_cache(self) -> None:
        """Phase-2 swing sweep must find isects under the ug1-keyed cache
        (a stale (ub,ug2,ra)-only lookup would silently return None for
        every top point → zero swing variants)."""
        model, pts = quick_pentode("EL84")
        c = _xfmr_constraints(swing_steps=3)
        r = optimize_measurements(pts, ub=300.0, constraints=c,
                                  ug2_filter=250.0, model=model)
        assert r.best is not None
        with_swing = [p for p in r.grid_points if p.half_swing > 0]
        assert with_swing, "swing sweep produced no points — cache key broken"


class TestXfmrUaRangeExtension:
    """Pin for XFMR_UA_MAX_FACTOR: the mutation-testing audit showed a
    2.0→1.2 revert survives the whole suite. The transformer AC line
    swings the anode above Ub — the intersection family MUST contain
    points there, or the negative-bias curves silently vanish."""

    def test_model_ua_max_doubles_for_xfmr(self) -> None:
        from lm19.constants import MODEL_UA_MAX_DEFAULT_V
        from lm19.optimizer import (
            UA_MAX_FACTOR,
            XFMR_UA_MAX_FACTOR,
            _model_ua_max,
        )
        assert XFMR_UA_MAX_FACTOR == pytest.approx(2.0)
        assert _model_ua_max(300.0, "se_xfmr") == pytest.approx(600.0)
        assert _model_ua_max(300.0, "se") == pytest.approx(
            max(300.0 * UA_MAX_FACTOR, MODEL_UA_MAX_DEFAULT_V))

    def test_isects_reach_beyond_default_ceiling(self) -> None:
        """Discriminating M3 pin: the deep-bias intersections of the AC
        line live ABOVE both 1.2×Ub and the MODEL_UA_MAX_DEFAULT_V=500
        floor. A 2.0→1.2 revert caps the search at 500 V and this fails.
        (Grid-level assertions can't see the revert at Ub=300 — the 500 V
        floor masks it — hence the direct-range pin.)"""
        import lm19.optimizer as opt_mod
        from lm19.amplifier import (
            TransformerLoadLine,
            find_intersections_model,
        )

        model, _ = quick_pentode("EL84")
        ll = TransformerLoadLine(300.0, ra_dc=0.1, ra_ac=8.0)
        isects = find_intersections_model(
            model, ll, [-12.0, -10.0, -8.0, -6.0, -4.0], ug2=250.0,
            ua_range=(1.0, opt_mod._model_ua_max(300.0, "se_xfmr")),
            ug1_bias=-7.0)
        assert len(isects) >= 4
        # Q ≈ (300, ~48 mA) → the −12 V curve crosses the AC line well
        # past 500 V. Truncated range loses it entirely.
        assert max(p["ua"] for p in isects) > 500.0


class TestXfmrNoModelFallback:
    """Without a model: naive behaviour is kept and made visible."""

    def test_warning_surfaced(self) -> None:
        _, pts = quick_pentode("EL84")
        c = _xfmr_constraints()
        r = optimize_measurements(pts, ub=300.0, constraints=c,
                                  ug2_filter=250.0)
        assert r.best is not None           # still produces (naive) results
        assert r.warning == OPT_WARN_SE_XFMR_NO_MODEL

    def test_dft_fallback_warning_takes_priority(self) -> None:
        """warning is a single i18n slot — the hd-method warning wins."""
        _, pts = quick_pentode("EL84")
        c = _xfmr_constraints(hd_method=HD_METHOD_DFT)
        r = optimize_measurements(pts, ub=300.0, constraints=c,
                                  ug2_filter=250.0)
        assert r.warning == OPT_WARN_DFT_NO_MODEL_FALLBACK

    def test_non_xfmr_circuits_never_warn(self) -> None:
        _, pts = quick_pentode("EL84")
        for circuit in (CIRCUIT_SE, CIRCUIT_CF):
            c = _xfmr_constraints(circuit=circuit)
            r = optimize_measurements(pts, ub=300.0, constraints=c,
                                      ug2_filter=250.0)
            assert r.warning is None, circuit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
