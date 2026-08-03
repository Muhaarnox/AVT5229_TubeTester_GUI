"""Background worker for amplifier optimization.

Runs grid sweep + swing sweep + parallel Pareto refinement, emitting
progress signals for the UI. Inherits :class:`BaseWorker` for the
unified ``stop()`` / ``cleanup()`` / error-emit pattern shared with
``ScanWorker``, ``HealthWorker`` etc. (the worker doesn't need a
hardware ``client`` — it operates purely on in-memory data).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from PySide6.QtCore import Signal

from app.workers import BaseWorker
from lm19.amplifier.constants import (
    CIRCUIT_PP,
)

if TYPE_CHECKING:
    from lm19.optimizer import OptimizerConstraints, OptimizerResult
    from lm19.tube_model_base import TubeModelProtocol

log = logging.getLogger(__name__)


class OptimizeWorker(BaseWorker):
    """Run full optimization pipeline in background thread.

    Phases:
        1. Grid sweep (0–50%)
        2. Swing sweep (50–70%)   — handled inside optimize_*
        3. Pareto refinement (70–100%) — parallel scipy

    Signals:
        progress(current_pct, phase_text): 0–100 progress + label.
        finished_ok(OptimizerResult): successful result.
        finished_err(str): error message — emitted only when execution
            failed for a non-cancellation reason. ``BaseWorker.failed``
            is also available but ``finished_err`` is kept for the
            existing UI signal name.
    """

    progress = Signal(int, str)        # (percent 0-100, phase label)
    finished_ok = Signal(object)       # OptimizerResult
    finished_err = Signal(str)         # error text
    finished_cancelled = Signal()      # terminal signal on user cancel

    # Phase weight percentages
    _GRID_PCT = 50
    _SWING_PCT = 70
    _REFINE_PCT = 100

    def __init__(
        self,
        points: List[Dict],
        constraints: "OptimizerConstraints",
        ub: float,
        model: Optional["TubeModelProtocol"] = None,
        ug2_filter: Optional[float] = None,
        ug2_values: Optional[List[float]] = None,
        ug1_values: Optional[List[float]] = None,
        use_model_path: bool = False,
        points_b: Optional[List[Dict]] = None,
        parent=None,
    ) -> None:
        # client=None — optimizer operates on already-collected points.
        super().__init__(client=None, parent=parent)
        self._points = points
        self._constraints = constraints
        self._ub = ub
        self._model = model
        self._ug2_filter = ug2_filter
        self._ug2_values = ug2_values
        self._ug1_values = ug1_values
        self._use_model_path = use_model_path
        self._points_b = points_b

    # ── Aliases for UI binding + cancelled-callback shape ────────────
    # The UI binds ``opt_cancel_btn.clicked`` to ``worker.cancel`` and
    # the inner ``optimize_*`` calls receive ``cancelled=lambda: ...``,
    # so we expose ``cancel()`` and ``_cancelled`` as thin wrappers
    # over ``BaseWorker.stop()`` / ``_stop_requested``.

    def cancel(self) -> None:
        """Alias for :meth:`stop` — kept for UI button binding."""
        self.stop()

    @property
    def _cancelled(self) -> bool:
        """Alias for ``_stop_requested`` — read by inner cancellation lambdas."""
        return self._stop_requested

    def run(self) -> None:
        """Override BaseWorker.run() to suppress error emit on cancellation and
        emit a terminal ``finished_cancelled``.

        BaseWorker.run() always emits ``failed`` on exceptions; here a cancelled
        run can raise mid-sweep (e.g. scipy interrupted) and we don't want to
        surface that as an error. But ``_execute`` also *returns* silently at
        its ``if self._cancelled: return`` points — without a terminal signal
        the UI would stay stuck in the running state (neither finished_ok nor
        finished_err fires). ``finished_cancelled`` resolves that for both the
        return and the suppressed-error cancel paths.
        """
        try:
            self._execute()
        except Exception as exc:
            log.exception("OptimizeWorker failed")
            if not self._stop_requested:
                self.finished_err.emit(str(exc))
                return
        if self._stop_requested:
            self.finished_cancelled.emit()

    def _execute(self) -> None:
        from lm19.optimizer import (
            optimize_measurements,
            optimize_model,
            optimize_pp,
            refine_pareto_front,
        )

        # Phase 1+2: Grid + Swing sweep
        self.progress.emit(5, "opt_phase_grid")
        if self._cancelled:
            return

        # Cancellation lambda passed into all grid loop levels so Cancel
        # responds within a few evaluations even mid-row.
        _is_cancelled = lambda: self._cancelled

        # Live grid progress (5 → _GRID_PCT): without it the bar froze at
        # 5% for the entire phase-1 sweep. Only distinct percents are
        # emitted to avoid flooding the queued signal connection.
        _last_pct = [5]

        def _on_grid_progress(done: int, total: int) -> None:
            if total <= 0:
                return
            pct = 5 + int((self._GRID_PCT - 5) * done / total)
            if pct != _last_pct[0]:
                _last_pct[0] = pct
                self.progress.emit(pct, "opt_phase_grid")

        if self._constraints.circuit == CIRCUIT_PP:
            result = optimize_pp(
                self._points, ub=self._ub,
                constraints=self._constraints,
                points_b=self._points_b,
                ug2_filter=self._ug2_filter,
                model=self._model,
                cancelled=_is_cancelled,
                on_progress=_on_grid_progress,
            )
        elif self._use_model_path and self._model is not None:
            result = optimize_model(
                self._model, self._constraints,
                ug1_values=self._ug1_values,
                cancelled=_is_cancelled,
                on_progress=_on_grid_progress,
            )
        else:
            # Pass model so optimizer can use DFT in refine when hd_method
            # is auto/dft and model is available, even on measurements path.
            result = optimize_measurements(
                self._points, ub=self._ub,
                constraints=self._constraints,
                ug2_filter=self._ug2_filter,
                ug2_values=self._ug2_values,
                model=self._model,
                cancelled=_is_cancelled,
                on_progress=_on_grid_progress,
            )

        if self._cancelled:
            return

        self.progress.emit(self._SWING_PCT, "opt_phase_swing")

        if result.error:
            self.finished_err.emit(result.error)
            return

        # Phase 3: Refine Pareto front in parallel
        if result.pareto_front and not self._cancelled:
            self.progress.emit(self._SWING_PCT, "opt_phase_refine")

            def _on_refine_progress(current: int, total: int) -> None:
                if total > 0:
                    refine_pct = int(
                        self._SWING_PCT
                        + (self._REFINE_PCT - self._SWING_PCT) * current / total
                    )
                    self.progress.emit(refine_pct, f"opt_phase_refine_n|{current}|{total}")

            refine_warnings: list = []
            refined_front = refine_pareto_front(
                result.pareto_front,
                points=self._points,
                model=self._model,
                constraints=self._constraints,
                ug2_filter=self._ug2_filter,
                ug1_values=self._ug1_values,
                cancelled=lambda: self._cancelled,
                on_progress=_on_refine_progress,
                points_b=self._points_b,
                warnings_out=refine_warnings,
            )
            for w in refine_warnings:
                if w != result.warning and w not in result.warnings:
                    result.warnings.append(w)

            if self._cancelled:
                return

            result.refined_pareto = refined_front

            # Pick the best from refined front
            from lm19.optimizer import _score
            if refined_front:
                result.refined = min(
                    refined_front,
                    key=lambda p: _score(p, self._constraints),
                )

        if not self._cancelled:
            self.progress.emit(100, "opt_phase_done")
            self.finished_ok.emit(result)
