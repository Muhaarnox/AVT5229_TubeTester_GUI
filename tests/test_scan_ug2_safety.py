"""Hardware-safety: keep Ua >= Ug2 across the three Ig2-spike-prone paths.

A conducting pentode whose Ua falls below Ug2 draws a screen-grid current spike
that can destroy the screen grid. Three scan paths must never leave the tube in
that state:
  #1 Ug2-track refine — order each step by direction (drop Ua → lower Ug2 FIRST).
  #2 independent Ug2 levels — raise Ua to the new Ug2 BEFORE raising Ug2 (the
     old ``if prev_ua < ug2`` guard read a frozen, stale scan-start prev_ua).
  #3 after a protection break zeroes Ug2 — re-establish Ug2 for the level's
     remaining Ug1 curves (do not measure them at Ug2≈0).

Each test records the ordered (param, value) settle calls and replays the
Ua/Ug2 state, asserting the invariant. The old code fails these.
"""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from lm19.scan.refine import _refine_curve_inline
from lm19.scan.sweepers import _SweepCtx, _sweep_ug2_independent
from lm19.scan.exceptions import _BreakSweep
from lm19.scan.settings import ScanRange, ScanSettings
from lm19.calibration import CalibrationData


def _refine_settings(**kw):
    d = dict(
        ua=ScanRange(0, 100, 20), ug1=ScanRange(-2, -2, 0),
        ug2=ScanRange(0, 0, 0), uh=6.3, ih=0.0,
        is_triode=False, ug2_track_ua=True, ug2_offset=0.0,
        refine_enabled=True, refine_max_depth=1,
        refine_min_step_ua=3.0, refine_onset_ma=0.5,
        refine_curvature_thr=0.15, refine_gradient_ratio=3.0,
        refine_ig2_delta_min=0.5, refine_delta_ia_thr=0.25,
    )
    d.update(kw)
    return ScanSettings(**d)


def _sweep_settings(**kw):
    d = dict(
        ua=ScanRange(0, 100, 50), ug1=ScanRange(-2, -2, 0),
        ug2=ScanRange(50, 100, 50), uh=6.3, ih=0.0,
        is_triode=False, ug2_track_ua=False,
        ua_settle_per_volt_s=0.0, ua_settle_base_s=0.0,
        ua_tolerance=1.0, ua_retries=1,
        ug1_settle_per_volt_s=0.0, ug1_settle_base_s=0.0,
        ug1_tolerance=0.1, ug1_retries=1,
        ug2_settle_per_volt_s=0.0, ug2_settle_base_s=0.0,
        ug2_tolerance=1.0, ug2_retries=1,
        ia_samples=1, calibration=CalibrationData(),
        refine_enabled=False,
    )
    d.update(kw)
    return ScanSettings(**d)


def _make_recorder(*, ua_values, ug1_values, settings, break_at_read=None):
    """Build a recording _SweepCtx. ``break_at_read`` makes read_point raise
    _BreakSweep once (after zeroing Ug2, as the firmware does on OVERIA)."""
    ops = []
    st = {"ua": 0.0, "ug1": 0.0, "ug2": 0.0}
    reads = {"n": 0}

    def settle_ua(v):
        ops.append(("ua", float(v))); st["ua"] = float(v); return float(v)

    def settle_ug1(v):
        ops.append(("ug1", float(v))); st["ug1"] = float(v); return float(v)

    def settle_ug2(v):
        ops.append(("ug2", float(v))); st["ug2"] = float(v); return float(v)

    def read_point():
        reads["n"] += 1
        if break_at_read is not None and reads["n"] == break_at_read:
            st["ug2"] = 0.0  # firmware OVERIA zeroes ug2set
            raise _BreakSweep()
        return {"ua": st["ua"], "ug1": st["ug1"], "ug2": st["ug2"],
                "ia": 5.0, "ig2": 0.0, "uh": 6.3, "ih": 0.3}

    ctx = _SweepCtx(
        settle_ua=settle_ua, settle_ug1=settle_ug1, settle_ug2=settle_ug2,
        read_point=read_point, stopped=lambda: False, progress=None, stop=None,
        settings=settings, pa_limit=0.0, pg2_limit=0.0, ig2_limit=0.0,
        ua_values=list(ua_values), ug1_values=list(ug1_values),
    )
    return ctx, ops


class TestTrackRefineOrdering(unittest.TestCase):
    """#1: Ug2-track refine never leaves Ua below Ug2."""

    @patch("time.sleep")
    def test_track_refine_keeps_ua_at_or_above_ug2(self, _):
        s = _refine_settings()
        ua_vals = [0, 20, 40, 60, 80, 100]
        # Onset at Ua~40 → refine produces midpoints; the first is a big drop
        # from the coarse top (Ua=100) where the old order spiked.
        pts = [
            {"ua": 0, "ug1": -2.0, "ug2": 0.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 20, "ug1": -2.0, "ug2": 20.0, "ia": 0.0, "ig2": 0.0},
            {"ua": 40, "ug1": -2.0, "ug2": 40.0, "ia": 50.0, "ig2": 0.0},
            {"ua": 60, "ug1": -2.0, "ug2": 60.0, "ia": 55.0, "ig2": 0.0},
            {"ua": 80, "ug1": -2.0, "ug2": 80.0, "ia": 57.0, "ig2": 0.0},
            {"ua": 100, "ug1": -2.0, "ug2": 100.0, "ia": 58.0, "ig2": 0.0},
        ]
        ops = []
        st = {"ua": 0.0, "ug2": 0.0}

        def settle_ua(ua):
            ops.append(("ua", ua)); st["ua"] = ua; return ua

        def settle_ug2(ug2):
            ops.append(("ug2", ug2)); st["ug2"] = ug2; return ug2

        def read_point():
            ia = {10: 0.0, 30: 25.0, 50: 52.0}.get(st["ua"], 1.0)
            return {"ua": st["ua"], "ug1": -2.0, "ug2": st["ug2"], "ia": ia,
                    "ig2": 0.0, "uh": 6.3, "ih": 0.3}

        result = _refine_curve_inline(
            pts, ua_vals, s, settle_ua, settle_ug2, read_point,
            None, None, 0.0, 0.0, ig2_limit=0.0)
        self.assertTrue(result, "refine should produce points for this curve")
        self.assertTrue(any(o == "ug2" for o, _ in ops),
                        "track refine should set Ug2")
        # Replay: the tube must never see Ua below Ug2 (offset 0 → Ug2 == Ua).
        cur_ua = float(pts[-1]["ua"])
        cur_ug2 = float(pts[-1]["ua"])
        for op, val in ops:
            if op == "ua":
                cur_ua = val
            else:
                cur_ug2 = val
            self.assertLessEqual(
                cur_ug2, cur_ua + 1e-6,
                f"Ig2 spike: Ug2={cur_ug2} > Ua={cur_ua} after {op}={val}\n{ops}")


class TestIndependentUg2Guard(unittest.TestCase):
    """#2: raising Ug2 to a new level is always preceded by raising Ua to it,
    regardless of the (stale) scan-start prev_ua."""

    @patch("time.sleep")
    def test_ua_raised_before_each_ug2_increase(self, _):
        s = _sweep_settings(ug2=ScanRange(50, 100, 50))  # levels 50, 100
        ctx, ops = _make_recorder(
            ua_values=[0, 50, 100], ug1_values=[-2.0], settings=s)
        # prev_ua HIGH (>= top Ug2): the OLD `if prev_ua < ug2` guard would skip
        # the safety raise for the Ug2=100 level, leaving Ug2 pushed above the
        # actual (low, post-down-sweep) Ua.
        _sweep_ug2_independent(ctx, prev_ua=100.0)
        for i, (p, v) in enumerate(ops):
            if p == "ug2" and v == 100.0:
                prior_ua = [val for op, val in ops[:i] if op == "ua"]
                self.assertTrue(prior_ua, "a Ua set must precede Ug2=100")
                self.assertGreaterEqual(
                    prior_ua[-1], 100.0,
                    f"Ua={prior_ua[-1]} must be >= Ug2=100 before the increase\n{ops}")
                break
        else:
            self.fail("Ug2=100 was never set")


class TestUg2ResettleAfterBreak(unittest.TestCase):
    """#3: after a protection break zeroes Ug2, the remaining Ug1 curves of the
    level re-establish Ug2 (not measure at Ug2≈0)."""

    @patch("time.sleep")
    def test_ug2_resettled_after_break(self, _):
        s = _sweep_settings(ug2=ScanRange(100, 100, 50), ua=ScanRange(0, 100, 50))
        ctx, ops = _make_recorder(
            ua_values=[0, 50, 100], ug1_values=[-2.0, -4.0],
            settings=s, break_at_read=1)  # break during the first Ug1 curve
        _sweep_ug2_independent(ctx, prev_ua=100.0)
        ug2_100 = [i for i, (p, v) in enumerate(ops) if p == "ug2" and v == 100.0]
        self.assertGreaterEqual(
            len(ug2_100), 2,
            "Ug2 must be re-settled after a protection break — else the level's "
            f"remaining Ug1 curves are measured at Ug2≈0\n{ops}")
        # The re-settle must be preceded by Ua >= Ug2 (safe order).
        resettle = ug2_100[-1]
        prior_ua = [v for p, v in ops[:resettle] if p == "ua"]
        self.assertTrue(prior_ua, "a Ua set must precede the Ug2 re-settle")
        self.assertGreaterEqual(
            prior_ua[-1], 100.0,
            f"Ua must be >= Ug2 before the Ug2 re-settle\n{ops}")


if __name__ == "__main__":
    unittest.main()
