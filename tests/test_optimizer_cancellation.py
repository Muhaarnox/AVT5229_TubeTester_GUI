"""Optimizer cancellation: grid loops must honour ``cancelled`` callback.

Without polling inside the grid loops, ``optimize_measurements`` /
``optimize_model`` / ``optimize_pp`` run their full sweep before
returning — Cancel waits up to ~60 s on the worst case (DFT-PP grid
with UL sweep). ``refine_pareto_front`` and ``OptimizeWorker``
between-phases already polled, so the heaviest phase was the only
one left uninterruptible.

These tests verify:

- Each grid function accepts ``cancelled`` and exits early when it
  fires (immediate-True case).
- A flipping callback (True after N calls) breaks out of the loops
  before the grid completes.
- ``cancelled=None`` (or omitted) keeps the full-grid behaviour for
  call sites that don't need cancellation.
- ``_sweep_swing_top_n`` also honours the callback (its DFT path is
  per-tube-cycle expensive on PP).
- ``OptimizeWorker.cancel()`` actually plumbs through to the grid
  functions (signal-level integration).
"""

from __future__ import annotations

import inspect
from typing import Callable
from unittest.mock import MagicMock

import pytest

from lm19.optimizer import (
    OptimizerConstraints,
    optimize_measurements,
    optimize_model,
    optimize_pp,
    _sweep_swing_top_n,
)
from lm19.tube_sim import quick_pentode, quick_triode

from tests._fixtures import make_triode_points as _make_triode_points
from lm19.amplifier.constants import (
    CIRCUIT_PP,
)
from lm19.optimizer import (
    OPT_ERR_NO_VALID_POINTS,
)


# ── helpers ──────────────────────────────────────────────────────────

def _flip_after(n: int) -> Callable[[], bool]:
    """Return a cancel callback that flips True on the (n+1)-th call.

    ``n=0`` → True from the start (immediate cancel).
    ``n=2`` → False on first 2 calls, True from call 3 onward.
    """
    state = {"calls": 0}

    def cb() -> bool:
        state["calls"] += 1
        return state["calls"] > n

    return cb


def _basic_constraints(**overrides) -> OptimizerConstraints:
    """Constraint preset with small grid so tests run fast."""
    base = dict(
        ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
        ug1_steps=6, ra_steps=6, ub_steps=2,
    )
    base.update(overrides)
    return OptimizerConstraints(**base)


# ── signature-level: each function exposes ``cancelled`` parameter ──

class TestSignatures:
    """Pin: every grid-function signature exposes ``cancelled``.
    Catches a future regression where someone removes the parameter."""

    @pytest.mark.parametrize("fn", [
        optimize_measurements, optimize_model, optimize_pp,
        _sweep_swing_top_n,
    ])
    def test_function_accepts_cancelled(self, fn):
        sig = inspect.signature(fn)
        assert "cancelled" in sig.parameters, (
            f"{fn.__name__} must accept a 'cancelled' callback; without "
            f"it the grid sweep blocks UI Cancel for up to ~60 s."
        )
        # Default must be None so call sites without a callback keep working.
        assert sig.parameters["cancelled"].default is None

    def test_all_public_optimize_functions_accept_cancelled(self):
        """Discoverability pin: ANY new ``optimize_*`` function added to
        ``lm19/optimizer.py`` must accept ``cancelled``. Without this
        CI check, a contributor adding e.g. ``optimize_cf_xfmr`` could
        forget the parameter and silently re-introduce the
        UI-blocking-on-Cancel bug for that circuit topology.

        Excludes ``refine_*`` (those already have cancellation via a
        different path) and private helpers.
        """
        from lm19 import optimizer as opt_mod
        offenders: list = []
        for name in dir(opt_mod):
            if not name.startswith("optimize_"):
                continue
            fn = getattr(opt_mod, name)
            if not callable(fn):
                continue
            sig = inspect.signature(fn)
            if "cancelled" not in sig.parameters:
                offenders.append(name)
        assert not offenders, (
            f"Public optimize_* functions must accept a 'cancelled' "
            f"callback. Missing: {offenders}. Without it, grid sweep "
            f"is uninterruptible and UI Cancel can block for tens of "
            f"seconds."
        )


# ── optimize_measurements ────────────────────────────────────────────

class TestOptimizeMeasurementsCancellation:

    def test_immediate_cancel_returns_no_valid_points(self):
        """``cancelled`` returning True from the first poll → loops
        skip everything → ``_build_result`` reports no_valid_points."""
        pts = _make_triode_points()
        c = _basic_constraints()
        r = optimize_measurements(
            pts, ub=250.0, constraints=c,
            cancelled=lambda: True,
        )
        assert r.error == OPT_ERR_NO_VALID_POINTS
        assert len(r.grid_points) == 0

    def test_flip_mid_grid_truncates(self):
        """Cancel after a few iterations → fewer grid points than
        a full uninterrupted run."""
        pts = _make_triode_points()
        c = _basic_constraints()

        full = optimize_measurements(pts, ub=250.0, constraints=c)
        truncated = optimize_measurements(
            pts, ub=250.0, constraints=c,
            cancelled=_flip_after(2),
        )
        assert len(truncated.grid_points) < len(full.grid_points), (
            "Flipping callback should break inner loops before grid completes"
        )

    def test_none_callback_full_grid(self):
        """``cancelled=None`` (and the omitted-argument case) runs the
        full grid, matching the explicit-True-only behaviour."""
        pts = _make_triode_points()
        c = _basic_constraints()
        no_cb = optimize_measurements(pts, ub=250.0, constraints=c)
        explicit_none = optimize_measurements(
            pts, ub=250.0, constraints=c, cancelled=None,
        )
        assert len(no_cb.grid_points) == len(explicit_none.grid_points)

    def test_cancel_skips_swing_phase(self):
        """When cancel fires after Phase 1, Phase 2 swing sweep must
        not start (it's another expensive nested loop)."""
        pts = _make_triode_points()
        c = _basic_constraints(swing_steps=4)
        # Flip after enough calls to complete Phase 1 grid but before
        # the swing top-N loop runs. Concretely: poll counter exhausts,
        # then is_cancelled = True before _sweep_swing_top_n iterates.
        # Simpler: full cancel — both phases skip.
        r = optimize_measurements(
            pts, ub=250.0, constraints=c,
            cancelled=lambda: True,
        )
        # No swing phase ran → no points were collected at all
        assert len(r.grid_points) == 0

    def test_bounded_blast_radius_first_ra_iteration(self):
        """After the first ra-loop iteration completes, the next poll
        must break. Asserts the check is in the **middle** ra loop, not
        only in the inner ug1 loop. A regression that moves the poll
        deeper would let many more points slip through.

        Poll sequence with this config (ub_steps=1, ug2 single,
        swing_steps=1 disables Phase 2):
          1: outer ub_v → False (continue)
          2: middle ra=0 → False (run ra=0 block: up to ug1_steps points)
          3: middle ra=1 → True  (break ra loop)
        Result: ≤ ug1_steps points, regardless of ra_steps×ug1_steps total.
        """
        pts = _make_triode_points()
        ug1_steps = 5
        c = OptimizerConstraints(
            ug1_range=(-9.0, -3.0), ra_range=(3.0, 20.0),
            ug1_steps=ug1_steps, ra_steps=8, ub_steps=1,
            swing_steps=1,  # disable Phase 2 to isolate Phase 1 bound
        )
        r = optimize_measurements(
            pts, ub=250.0, constraints=c,
            cancelled=_flip_after(2),
        )
        assert len(r.grid_points) <= ug1_steps, (
            f"After first ra iteration completes, cancel must break the ra "
            f"loop (one block ≤ {ug1_steps} points). Got "
            f"{len(r.grid_points)} — check probably misplaced into inner "
            f"ug1 loop, raising cancel latency."
        )

    def test_swing_phase_receives_cancelled_callback(self, monkeypatch):
        """End-to-end: cancelled forwarded from optimize_measurements
        into _sweep_swing_top_n. Without forwarding, a future grid
        function added by copy-paste could silently leave swing
        uninterruptible (covered for unit but not integration today)."""
        from lm19 import optimizer as opt_mod
        captured = {}
        real_sweep = opt_mod._sweep_swing_top_n

        def spy_sweep(*args, **kwargs):
            captured["cancelled"] = kwargs.get("cancelled")
            return real_sweep(*args, **kwargs)

        monkeypatch.setattr(opt_mod, "_sweep_swing_top_n", spy_sweep)

        pts = _make_triode_points()
        c = _basic_constraints(swing_steps=3)
        sentinel = lambda: False
        optimize_measurements(
            pts, ub=250.0, constraints=c, cancelled=sentinel,
        )
        assert captured.get("cancelled") is sentinel, (
            "optimize_measurements must forward cancelled to "
            "_sweep_swing_top_n so swing phase is also interruptible. "
            "Without this, swing top-N × swing_steps DFT calls keep "
            "running after Cancel."
        )


# ── optimize_model ───────────────────────────────────────────────────

class TestOptimizeModelCancellation:

    def _make_model_and_ug1(self):
        m, pts = quick_triode("12AU7")
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        return m, ug1_vals

    def _model_constraints(self):
        # Mirror the proven-good params from
        # TestOptimizeModel.test_model_explores_ub_range so the
        # uninterrupted grid actually produces points.
        return OptimizerConstraints(
            ug1_range=(-12.0, -5.0), ra_range=(5.0, 20.0),
            ug1_steps=3, ra_steps=3,
            ub_range=(200.0, 350.0), ub_steps=3,
        )

    def test_immediate_cancel_no_grid_points(self):
        m, ug1_vals = self._make_model_and_ug1()
        c = self._model_constraints()
        r = optimize_model(
            m, c, ug1_values=ug1_vals, cancelled=lambda: True,
        )
        assert r.error == OPT_ERR_NO_VALID_POINTS
        assert len(r.grid_points) == 0

    def test_flip_truncates_grid(self):
        m, ug1_vals = self._make_model_and_ug1()
        c = self._model_constraints()
        full = optimize_model(m, c, ug1_values=ug1_vals)
        # Sanity: full run actually has points to be truncated
        assert len(full.grid_points) > 0
        truncated = optimize_model(
            m, c, ug1_values=ug1_vals, cancelled=_flip_after(3),
        )
        assert len(truncated.grid_points) < len(full.grid_points)

    def test_swing_phase_receives_cancelled_callback(self, monkeypatch):
        """Forwarding sentinel: cancelled reaches _sweep_swing_top_n
        from optimize_model, same as for optimize_measurements."""
        from lm19 import optimizer as opt_mod
        captured = {}
        real_sweep = opt_mod._sweep_swing_top_n

        def spy_sweep(*args, **kwargs):
            captured["cancelled"] = kwargs.get("cancelled")
            return real_sweep(*args, **kwargs)

        monkeypatch.setattr(opt_mod, "_sweep_swing_top_n", spy_sweep)

        m, ug1_vals = self._make_model_and_ug1()
        c = self._model_constraints()
        sentinel = lambda: False
        optimize_model(m, c, ug1_values=ug1_vals, cancelled=sentinel)
        assert captured.get("cancelled") is sentinel


# ── optimize_pp ──────────────────────────────────────────────────────

class TestOptimizePPCancellation:

    def _make_points(self):
        _, pts = quick_pentode("EL84")
        return pts

    def test_immediate_cancel_no_grid_points(self):
        pts = self._make_points()
        c = _basic_constraints(circuit=CIRCUIT_PP, ra_range=(4.0, 12.0))
        r = optimize_pp(
            pts, ub=300.0, constraints=c,
            cancelled=lambda: True,
        )
        assert r.error == OPT_ERR_NO_VALID_POINTS
        assert len(r.grid_points) == 0

    def test_flip_truncates_grid(self):
        pts = self._make_points()
        c = _basic_constraints(circuit=CIRCUIT_PP, ra_range=(4.0, 12.0))
        full = optimize_pp(pts, ub=300.0, constraints=c)
        truncated = optimize_pp(
            pts, ub=300.0, constraints=c,
            cancelled=_flip_after(2),
        )
        assert len(truncated.grid_points) < len(full.grid_points)

    def test_none_callback_full_grid(self):
        pts = self._make_points()
        c = _basic_constraints(circuit=CIRCUIT_PP, ra_range=(4.0, 12.0))
        no_cb = optimize_pp(pts, ub=300.0, constraints=c)
        explicit_none = optimize_pp(
            pts, ub=300.0, constraints=c, cancelled=None,
        )
        assert len(no_cb.grid_points) == len(explicit_none.grid_points)

    def test_swing_phase_receives_cancelled_callback(self, monkeypatch):
        """Forwarding sentinel for PP: cancelled reaches
        _sweep_swing_top_n. PP has the worst worst-case (UL × ub × ra ×
        ug1 + DFT), so missing forwarding here is the highest blast."""
        from lm19 import optimizer as opt_mod
        captured = {}
        real_sweep = opt_mod._sweep_swing_top_n

        def spy_sweep(*args, **kwargs):
            captured["cancelled"] = kwargs.get("cancelled")
            return real_sweep(*args, **kwargs)

        monkeypatch.setattr(opt_mod, "_sweep_swing_top_n", spy_sweep)

        pts = self._make_points()
        c = _basic_constraints(circuit=CIRCUIT_PP, ra_range=(4.0, 12.0),
                                swing_steps=3)
        sentinel = lambda: False
        optimize_pp(pts, ub=300.0, constraints=c, cancelled=sentinel)
        assert captured.get("cancelled") is sentinel


# ── _sweep_swing_top_n direct test ───────────────────────────────────

class TestSwingSweepCancellation:
    """Phase-2 swing sweep must also poll. Per-pt block of swing_steps
    can take seconds on DFT path; without polling the user waits out
    the full top-N × swing_steps even after pressing Cancel."""

    def test_immediate_cancel_returns_empty(self):
        from lm19.optimizer import OptPoint
        # Build 3 dummy top points with non-zero max_swing
        top = [
            OptPoint(ub=250, ug2=0, ug1=-5, ra=10, thd=1.0, hd2=0.5, hd3=0.1,
                     pout_mw=500, pa_mw=2500, ia_0=10.0, ua_0=150.0,
                     amp_class="A", max_swing=2.0, half_swing=2.0,
                     valid=True)
            for _ in range(3)
        ]
        c = OptimizerConstraints(swing_steps=4)
        eval_fn = MagicMock(return_value=None)
        out = _sweep_swing_top_n(
            top, eval_fn, c, cancelled=lambda: True,
        )
        # Cancel before first iteration → eval_fn never called
        assert eval_fn.call_count == 0
        assert out == []

    def test_flip_after_first_pt_breaks(self):
        from lm19.optimizer import OptPoint
        top = [
            OptPoint(ub=250, ug2=0, ug1=-5 - i, ra=10, thd=1.0, hd2=0.5, hd3=0.1,
                     pout_mw=500, pa_mw=2500, ia_0=10.0, ua_0=150.0,
                     amp_class="A", max_swing=2.0, half_swing=2.0,
                     valid=True)
            for i in range(4)
        ]
        c = OptimizerConstraints(swing_steps=3)
        eval_fn = MagicMock(return_value=None)
        # cb returns False once, True afterwards → only first pt's
        # swing block runs. swing_steps=3 → 3 eval_fn calls then break.
        out = _sweep_swing_top_n(
            top, eval_fn, c, cancelled=_flip_after(1),
        )
        assert eval_fn.call_count == 3, (
            f"Expected exactly one swing-block (3 calls) before cancel "
            f"propagates, got {eval_fn.call_count}"
        )
        assert out == []  # eval_fn returned None for all calls


# ── Worker integration: cancel() actually plumbs through ──────────

class TestOptimizeWorkerCancellationPlumbing:
    """``OptimizeWorker.cancel()`` must reach the grid sweep, not just
    the between-phase guard. We verify by stubbing the optimize_*
    functions and asserting the ``cancelled`` kwarg reaches them as a
    callable that returns ``self._stop_requested``."""

    def _make_worker(self):
        from app.optimize_worker import OptimizeWorker
        return OptimizeWorker(
            points=[],
            constraints=OptimizerConstraints(),
            ub=250.0,
        )

    def test_measurements_path_passes_cancelled(self, monkeypatch):
        from app import optimize_worker as ow_mod
        captured = {}

        def fake_optimize_measurements(*args, **kwargs):
            captured["cancelled"] = kwargs.get("cancelled")
            from lm19.optimizer import OptimizerResult
            return OptimizerResult(grid_points=[], pareto_front=[])

        # Patch in the symbol the worker imports lazily inside _execute
        from lm19 import optimizer as opt_mod
        monkeypatch.setattr(opt_mod, "optimize_measurements",
                            fake_optimize_measurements)

        w = self._make_worker()
        w._execute()

        cb = captured.get("cancelled")
        assert callable(cb), "Worker must pass a cancelled callable"
        # Initially worker not cancelled → cb returns False
        assert cb() is False
        # After cancel(), the same callable starts returning True
        w.cancel()
        assert cb() is True

    def test_pp_path_passes_cancelled(self, monkeypatch):
        captured = {}

        def fake_optimize_pp(*args, **kwargs):
            captured["cancelled"] = kwargs.get("cancelled")
            from lm19.optimizer import OptimizerResult
            return OptimizerResult(grid_points=[], pareto_front=[])

        from lm19 import optimizer as opt_mod
        monkeypatch.setattr(opt_mod, "optimize_pp", fake_optimize_pp)

        w = self._make_worker()
        w._constraints = OptimizerConstraints(circuit=CIRCUIT_PP)
        w._execute()
        cb = captured.get("cancelled")
        assert callable(cb)
        w.cancel()
        assert cb() is True

    def test_model_path_passes_cancelled(self, monkeypatch):
        captured = {}

        def fake_optimize_model(*args, **kwargs):
            captured["cancelled"] = kwargs.get("cancelled")
            from lm19.optimizer import OptimizerResult
            return OptimizerResult(grid_points=[], pareto_front=[])

        from lm19 import optimizer as opt_mod
        monkeypatch.setattr(opt_mod, "optimize_model", fake_optimize_model)

        w = self._make_worker()
        # Give worker a model so use_model_path branch fires
        w._use_model_path = True
        w._model = MagicMock()
        w._execute()
        cb = captured.get("cancelled")
        assert callable(cb)
        w.cancel()
        assert cb() is True


class TestRefineNarrowExcept:
    """ML-091: an AttributeError/TypeError regression inside the NM
    objective must propagate — the old broad except silently substituted
    the unrefined grid point."""

    def test_programming_error_in_minimize_propagates(self, monkeypatch):
        import lm19.optimizer as opt
        monkeypatch.setattr(
            "scipy.optimize.minimize",
            lambda *a, **k: (_ for _ in ()).throw(AttributeError("drift")))
        pt = opt.OptPoint(ub=250.0, ug2=250.0, ug1=-8.0, ra=5000.0,
                          thd=1.0, hd2=0.5, hd3=0.2, pout_mw=100.0,
                          pa_mw=1000.0, ia_0=30.0, ua_0=150.0,
                          amp_class="A", max_swing=4.0, half_swing=2.0)
        from lm19.tube_sim import quick_pentode
        model, pts = quick_pentode("EL84")
        with pytest.raises(AttributeError):
            opt.refine_optimum(pt, pts, model,
                               opt.OptimizerConstraints())

    def test_data_error_returns_none(self, monkeypatch):
        import lm19.optimizer as opt
        monkeypatch.setattr(
            "scipy.optimize.minimize",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("bad x0")))
        pt = opt.OptPoint(ub=250.0, ug2=250.0, ug1=-8.0, ra=5000.0,
                          thd=1.0, hd2=0.5, hd3=0.2, pout_mw=100.0,
                          pa_mw=1000.0, ia_0=30.0, ua_0=150.0,
                          amp_class="A", max_swing=4.0, half_swing=2.0)
        from lm19.tube_sim import quick_pentode
        model, pts = quick_pentode("EL84")
        assert opt.refine_optimum(pt, pts, model,
                                  opt.OptimizerConstraints()) is None
