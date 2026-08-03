"""Tests for ``lm19.tube_model_base.check_fit_convergence`` helper.

The helper makes scipy ``least_squares`` non-convergence visible in
logs across the ~12 call sites in dempwolf / reefman / spice_export.
The fitters still return whatever scipy gave them — but a debugger
sees the WARNING and knows the parameters are noise vs. a genuine fit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

# ML-148: paths anchored to the repo, not CWD — a pytest run
# from outside lm19_app must not FileNotFoundError.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

from lm19.tube_model_base import (
    check_fit_convergence, compute_fit_quality, ConvergenceTracker,
    FIT_QUALITY_GREEN_PCT, FIT_QUALITY_YELLOW_PCT,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)


def _fake_result(success=True, status=2, cost=0.001, nfev=120):
    """Build a stand-in for scipy ``OptimizeResult``."""
    return SimpleNamespace(success=success, status=status, cost=cost,
                           nfev=nfev, x=[1.0, 2.0, 3.0])


class TestCheckFitConvergence:
    """``check_fit_convergence`` warns iff ``result.success`` is False."""

    def test_converged_result_silent(self, caplog):
        """Successful convergence emits no warning."""
        log = logging.getLogger("lm19.test")
        with caplog.at_level(logging.WARNING, logger="lm19.test"):
            check_fit_convergence(_fake_result(), "test", log)
        assert caplog.records == [], \
            f"Unexpected log records: {[r.message for r in caplog.records]}"

    def test_unconverged_logs_warning(self, caplog):
        """Failed convergence emits WARNING with phase + status + cost."""
        log = logging.getLogger("lm19.test")
        with caplog.at_level(logging.WARNING, logger="lm19.test"):
            check_fit_convergence(
                _fake_result(success=False, status=0, cost=42.0, nfev=5000),
                "phase1 (cathode)", log,
                tube="EL84", n_points=42,
            )
        assert len(caplog.records) == 1
        msg = caplog.records[0].message
        # Diagnostic must include phase, status, cost, and context
        assert "phase1 (cathode)" in msg
        assert "max_nfev exceeded" in msg
        assert "42" in msg  # cost
        assert "EL84" in msg
        assert "n_points=42" in msg

    def test_returns_result_unchanged(self):
        """Helper returns the same object so it can be chained."""
        r = _fake_result(success=False)
        log = logging.getLogger("lm19.test")
        out = check_fit_convergence(r, "phase", log)
        assert out is r

    def test_unknown_status_code_handled(self, caplog):
        """Unrecognized status codes don't crash the helper."""
        log = logging.getLogger("lm19.test")
        with caplog.at_level(logging.WARNING, logger="lm19.test"):
            check_fit_convergence(
                _fake_result(success=False, status=99, cost=1.0, nfev=1),
                "phase", log,
            )
        assert "status=99" in caplog.records[0].message


class TestFittersUseHelper:
    """Sanity: each fitter module imports + invokes ``check_fit_convergence``.

    Guards against a refactor that drops the helper call (which would
    silently restore the previous swallow-all behaviour).
    """

    @pytest.mark.parametrize("module", [
        "lm19.dempwolf",
        "lm19.reefman",
        "lm19.spice_export.koren",
    ])
    def test_module_imports_helper(self, module):
        import importlib
        m = importlib.import_module(module)
        assert hasattr(m, "check_fit_convergence"), (
            f"{module} must import check_fit_convergence "
            "(silent un-converged fits regression)"
        )

    @pytest.mark.parametrize("path,least_squares_count,convergence_check_count", [
        # dempwolf: 8 least_squares (phases 1-5 incl. phase-5 multi-start,
        # triode joint refine, phase-4 perturbed restarts, beam joint
        # refine); 7 wrapped via tracker.check (through the _check helper
        # that dispatches to ConvergenceTracker), 1 in the phase-4
        # perturbed-restart loop logged at DEBUG.
        ("lm19/dempwolf.py", 8, 7),
        # reefman: 3 least_squares; 1 in phase3 wrapped (the converged
        # signal that propagates to ModelFitResult), 2 in retry loops
        # with their own log.debug per non-converged start.
        ("lm19/reefman.py", 3, 1),
        # spice_export: 2 (triode + pentode), both wrapped.
        ("lm19/spice_export/koren.py", 2, 2),
    ])
    def test_module_call_counts(self, path, least_squares_count,
                                 convergence_check_count):
        """Pin down ``least_squares`` ↔ convergence-check relationship.

        Counts both forms of convergence reporting:
          - direct ``check_fit_convergence(...)`` call
          - dispatched via ``_check(tracker, ...)`` → tracker.check → helper

        Catches regressions where a new ``least_squares`` is added without
        any convergence reporting, OR an existing check is removed.
        """
        import re
        with open(_PROJECT_ROOT / path, "r", encoding="utf-8") as f:
            src = f.read()
        actual_ls = src.count("least_squares(")
        # Each least_squares call must be followed by a convergence
        # check. Count both forms (only call sites — exclude definitions,
        # imports, and docstrings):
        # 1. Direct: ``check_fit_convergence(result, ...)`` — call has at
        #    least one positional arg before the closing paren.
        direct_helper = len(re.findall(
            r"check_fit_convergence\(\s*\w", src,
        ))
        # 2. Via tracker dispatcher: ``_check(tracker, result, ...)``.
        # Exclude the ``def _check(tracker, ...)`` definition itself.
        tracker_check = len(re.findall(
            r"(?<!def )_check\(tracker,\s*\w", src,
        ))
        # Subtract any matches inside the helper's own def body (the
        # _check() dispatcher in dempwolf.py contains a fallback
        # ``check_fit_convergence(result, phase_name, log, **ctx)`` —
        # that's the helper's wiring, not a fit-phase call site).
        in_helper_def = src.count(
            "check_fit_convergence(result, phase_name, log, **ctx)"
        )
        actual_checks = direct_helper + tracker_check - in_helper_def
        assert actual_ls == least_squares_count, (
            f"{path}: {actual_ls} least_squares calls, expected "
            f"{least_squares_count} (update test if intentional)"
        )
        assert actual_checks == convergence_check_count, (
            f"{path}: {actual_checks} convergence checks "
            f"(direct={direct_helper} + via_tracker={tracker_check}), "
            f"expected {convergence_check_count} "
            f"(regression: silent un-converged fits?)"
        )

    @pytest.mark.parametrize("path", [
        "lm19/dempwolf.py",
        "lm19/reefman.py",
        "lm19/spice_export/koren.py",
    ])
    def test_no_silent_except_continue(self, path):
        """No fitter may use ``except Exception: continue``.

        That broad pattern silently swallows every error:
          - phase1/2 retry loops in reefman would absorb all errors per-start
          - the variant loop would wrap whole ``_fit_variant`` calls
          - dempwolf had no exception handling at all (different shape)

        Narrow excepts catching specific data-shape errors
        (``ValueError``, ``RuntimeError``, ``np.linalg.LinAlgError``,
        ``KeyError``, ``IndexError``) plus a per-call ``log.warning`` /
        ``log.debug`` are the right pattern. Programming errors like
        ``AttributeError`` / ``TypeError`` / ``NameError`` must still
        propagate to surface regressions.
        """
        import re
        with open(_PROJECT_ROOT / path, "r", encoding="utf-8") as f:
            src = f.read()
        bad = re.findall(r"except Exception:\s*\n\s*continue", src)
        assert not bad, (
            f"{path}: {len(bad)} occurrence(s) of 'except Exception: continue' "
            "(silent swallow regression). Use narrow except + log.warning."
        )


class TestReefmanVariantLevelExcept:
    """Reefman variant loop: catch data errors, propagate programming errors.

    The reefman top-level fit tries variants ``D`` and ``DE`` and picks
    the lowest-RMS one. If a variant's pipeline genuinely fails (all
    starts diverged → ``RuntimeError``), the loop should log a WARNING
    and try the next variant. But if the failure is a programming bug
    (``AttributeError`` from a regression), it must propagate so the
    user sees the real cause — not "Reefman fitting failed for both
    D and DE variants" with no diagnostic.
    """

    def _make_min_points(self):
        """Synthetic pentode points sufficient for fit_reefman to start."""
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        return pts

    def test_variant_runtime_error_logged_and_continued(self, caplog,
                                                         monkeypatch):
        """If both variants raise RuntimeError → final RuntimeError + 2 warnings."""
        import logging
        from lm19 import reefman

        def fake_fit_variant(*args, **kwargs):
            raise RuntimeError("Phase 1 (cathode) failed")

        monkeypatch.setattr(reefman, "_fit_variant", fake_fit_variant)
        pts = self._make_min_points()
        with caplog.at_level(logging.WARNING, logger="lm19.reefman"):
            with pytest.raises(RuntimeError,
                               match="failed for both D and DE variants"):
                reefman.fit_reefman(pts, topology=TOPOLOGY_PENTODE)
        # Both variants should have been attempted and warned about
        variant_warnings = [r for r in caplog.records
                            if "variant=" in r.message]
        assert len(variant_warnings) == 2, (
            f"Expected 2 variant warnings (D + DE), got {len(variant_warnings)}: "
            f"{[r.message for r in caplog.records]}"
        )
        for w in variant_warnings:
            assert "RuntimeError" in w.message
            assert "Phase 1 (cathode) failed" in w.message

    def test_variant_attribute_error_propagates(self, monkeypatch):
        """Programming error (AttributeError) must NOT be swallowed."""
        from lm19 import reefman

        def fake_fit_variant(*args, **kwargs):
            # Simulates a refactor regression: undefined attribute access
            raise AttributeError("'NoneType' object has no attribute 'foo'")

        monkeypatch.setattr(reefman, "_fit_variant", fake_fit_variant)
        pts = self._make_min_points()
        # Programming errors propagate up, not silently absorbed into
        # "fitting failed for both variants"
        with pytest.raises(AttributeError, match="NoneType.*has no attribute"):
            reefman.fit_reefman(pts, topology=TOPOLOGY_PENTODE)

    def test_variant_type_error_propagates(self, monkeypatch):
        """TypeError (programming error) must propagate too."""
        from lm19 import reefman

        def fake_fit_variant(*args, **kwargs):
            raise TypeError("unsupported operand")

        monkeypatch.setattr(reefman, "_fit_variant", fake_fit_variant)
        pts = self._make_min_points()
        with pytest.raises(TypeError, match="unsupported operand"):
            reefman.fit_reefman(pts, topology=TOPOLOGY_PENTODE)

    def test_converged_flag_propagates_when_phase3_succeeds(self,
                                                              monkeypatch):
        """Reefman fit_variant returns converged=True when scipy converged."""
        from lm19 import reefman
        from types import SimpleNamespace
        # Stub least_squares to return a clearly-converged result
        good_x = [20.0, 1.3, 200.0, 100.0, 1000.0,
                  2000.0, 0.001, 3.0, 0.1, 0.01]  # 10 vals, scipy slices
        call_count = [0]

        def fake_ls(*args, **kwargs):
            call_count[0] += 1
            n_x = len(args[1]) if len(args) > 1 else 5
            return SimpleNamespace(x=good_x[:n_x], cost=0.001, success=True,
                                    status=2, nfev=50)

        monkeypatch.setattr(reefman, "least_squares", fake_ls)
        pts = self._make_min_points()
        result = reefman.fit_reefman(pts, topology=TOPOLOGY_PENTODE)
        assert result.converged is True

    def test_variant_first_fails_second_succeeds(self, caplog, monkeypatch):
        """If D fails (data error) but DE succeeds → fit completes, log warns
        about D, returns DE result."""
        import logging
        import numpy as np
        from lm19 import reefman
        from lm19.tube_params import lookup_tube

        # Build a known-good DE result using real EL84 params
        ref = lookup_tube("EL84")
        good_params = np.array([ref.reefman.mu, ref.reefman.ex, ref.reefman.kg1,
                                 ref.reefman.kg2, ref.reefman.kp, ref.reefman.kvb,
                                 ref.reefman.A, ref.reefman.als, ref.reefman.be])

        def fake_fit_variant(va, vg1, vg2, ia, ig2, has_ig2, variant):
            if variant == "D":
                raise RuntimeError("Phase 2 (splitting) failed")
            return good_params, 1.5, True  # (params, rms_ia_mA, converged)

        monkeypatch.setattr(reefman, "_fit_variant", fake_fit_variant)
        pts = self._make_min_points()
        with caplog.at_level(logging.WARNING, logger="lm19.reefman"):
            result = reefman.fit_reefman(pts, topology=TOPOLOGY_PENTODE)
        # Got a fit (DE variant)
        assert result is not None
        assert result.params["type"].endswith("DE")
        # D variant warning logged
        d_warnings = [r for r in caplog.records
                      if "variant=D " in r.message and "RuntimeError" in r.message]
        assert len(d_warnings) == 1, \
            f"Expected 1 warning for D variant failure: {[r.message for r in caplog.records]}"



class TestComputeFitQuality:
    """Quality verdict from RMS error vs mean Ia (the "is this fit garbage?" signal)."""

    def test_good_fit_well_below_threshold(self):
        # 0.5 mA RMS on 50 mA mean → 1% → good (< GREEN 2%)
        rms_pct, verdict = compute_fit_quality(rms_error_mA=0.5, mean_ia_mA=50.0)
        assert verdict == "good"
        assert abs(rms_pct - 1.0) < 0.01

    def test_fair_fit_in_yellow_band(self):
        # 5 mA RMS on 50 mA mean → 10% — exactly on YELLOW boundary should
        # round to "poor" (strict <), 4 mA → 8% is fair.
        rms_pct, verdict = compute_fit_quality(rms_error_mA=4.0, mean_ia_mA=50.0)
        assert verdict == "fair"
        assert abs(rms_pct - 8.0) < 0.01

    def test_poor_fit_above_yellow(self):
        # 7.5 mA on 50 mA → 15% — poor
        rms_pct, verdict = compute_fit_quality(rms_error_mA=7.5, mean_ia_mA=50.0)
        assert verdict == "poor"

    def test_thresholds_at_boundaries(self):
        # GREEN < 2.0 — 1.999% → good, 2.0 → fair, 2.001 → fair
        _, v_below = compute_fit_quality(0.01999, 1.0)
        _, v_at = compute_fit_quality(0.02, 1.0)
        assert v_below == "good"
        assert v_at == "fair"
        # YELLOW < 10.0 — same logic
        _, v_below_y = compute_fit_quality(0.0999, 1.0)
        _, v_at_y = compute_fit_quality(0.10, 1.0)
        assert v_below_y == "fair"
        assert v_at_y == "poor"

    def test_zero_mean_returns_unknown(self):
        # mean_ia=0 (e.g. tube fully cut off in measurement window) → unknown
        rms_pct, verdict = compute_fit_quality(1.0, 0.0)
        assert verdict == "unknown"
        assert rms_pct == 0.0

    def test_negative_mean_treated_as_unknown(self):
        rms_pct, verdict = compute_fit_quality(1.0, -5.0)
        assert verdict == "unknown"


class TestConvergenceTracker:
    """ConvergenceTracker aggregates per-phase scipy convergence."""

    def test_all_converged_when_no_check(self):
        t = ConvergenceTracker()
        assert t.all_converged is True

    def test_stays_true_when_all_phases_converge(self, caplog):
        t = ConvergenceTracker()
        log = logging.getLogger("lm19.test_tracker")
        with caplog.at_level(logging.WARNING, logger="lm19.test_tracker"):
            t.check(_fake_result(success=True), "phase1", log)
            t.check(_fake_result(success=True), "phase2", log)
        assert t.all_converged is True
        assert caplog.records == []

    def test_flips_to_false_on_any_failure(self, caplog):
        t = ConvergenceTracker()
        log = logging.getLogger("lm19.test_tracker")
        with caplog.at_level(logging.WARNING, logger="lm19.test_tracker"):
            t.check(_fake_result(success=True), "phase1", log)
            t.check(_fake_result(success=False, status=0, cost=99.0,
                                  nfev=5000), "phase2 (knee)", log)
            t.check(_fake_result(success=True), "phase3", log)
        assert t.all_converged is False
        assert any("phase2 (knee)" in r.message for r in caplog.records)

    def test_returns_result_unchanged(self):
        t = ConvergenceTracker()
        log = logging.getLogger("lm19.test_tracker")
        r = _fake_result(success=False)
        out = t.check(r, "phase", log)
        assert out is r


class TestModelFitResultQualityFields:
    """Fitters populate converged + rms_pct + quality on real fits."""

    def test_dempwolf_triode_fit_populates_quality(self):
        from lm19.tube_sim import quick_triode
        from lm19.dempwolf import fit_dempwolf
        _, pts = quick_triode("12AU7")
        result = fit_dempwolf(pts, topology=TOPOLOGY_TRIODE)
        # Synthetic Koren data → Dempwolf fits well, should at least be fair
        assert result.quality in ("good", "fair")
        assert result.rms_pct > 0
        assert result.converged is True

    def test_dempwolf_pentode_fit_populates_quality(self):
        from lm19.tube_sim import quick_pentode
        from lm19.dempwolf import fit_dempwolf
        _, pts = quick_pentode("EL84")
        result = fit_dempwolf(pts, topology=TOPOLOGY_PENTODE)
        assert result.quality in ("good", "fair")
        assert result.converged is True

    def test_reefman_fit_populates_quality(self):
        from lm19.tube_sim import quick_pentode
        from lm19.reefman import fit_reefman
        _, pts = quick_pentode("EL84")
        result = fit_reefman(pts, topology=TOPOLOGY_PENTODE)
        assert result.quality in ("good", "fair")
        assert result.converged is True

    def test_dempwolf_passes_converged_false_when_phase_diverges(self,
                                                                   monkeypatch):
        """When _fit_phase1 returns a non-converged scipy result,
        ModelFitResult.converged must be False."""
        from lm19 import dempwolf
        from lm19.tube_sim import quick_triode
        _, pts = quick_triode("12AU7")

        original_ls = dempwolf.fit_dempwolf  # capture before patching

        def fake_ls(*args, **kwargs):
            # Return non-converged result with sane params (so fit completes)
            x0 = args[1] if len(args) > 1 else kwargs.get("x0", [50.0]*5)
            return SimpleNamespace(
                x=list(x0), cost=0.5, success=False, status=0, nfev=5000,
            )

        # Patch scipy.optimize.least_squares globally for this test
        import scipy.optimize
        monkeypatch.setattr(scipy.optimize, "least_squares", fake_ls)

        result = dempwolf.fit_dempwolf(pts, topology=TOPOLOGY_TRIODE)
        # At least one phase didn't converge → result.converged == False
        assert result.converged is False,             "non-converged scipy result must propagate to ModelFitResult.converged"


class TestFitQualityConstants:
    """Ensure threshold constants are sane."""

    def test_thresholds_ordered(self):
        assert 0 < FIT_QUALITY_GREEN_PCT < FIT_QUALITY_YELLOW_PCT
