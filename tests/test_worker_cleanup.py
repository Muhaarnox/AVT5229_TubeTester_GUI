"""Tests for BaseWorker.cleanup, subclass cleanup, and HealthTab.shutdown."""

import os
from unittest.mock import MagicMock, patch

import pytest

# Headless Qt for tests that instantiate real QObject workers (signal-emit
# tests in TestStaleSignalIsolation). Other tests use MagicMock and don't
# touch Qt event loops.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.workers import (
    BaseWorker,
    ScanWorker,
    PreheatWorker,
    ParamPoller,
    CheckComWorker,
    ResetWorker,
    HealthWorker,
)


class _TestWorker(BaseWorker):
    """Concrete worker for testing."""

    def __init__(self):
        super().__init__(MagicMock())

    def _execute(self):
        pass


# ---------------------------------------------------------------------------
# BaseWorker.cleanup
# ---------------------------------------------------------------------------

class TestBaseWorkerCleanup:
    def test_cleanup_when_not_running(self):
        w = _TestWorker()
        w.cleanup()  # should not raise

    def test_cleanup_stops_and_waits(self):
        w = _TestWorker()
        w.isRunning = MagicMock(return_value=True)
        w.stop = MagicMock()
        w.wait = MagicMock()
        w.cleanup(timeout_ms=2000)
        w.stop.assert_called_once()
        w.wait.assert_called_once_with(2000)

    def test_cleanup_disconnects_signals(self):
        w = _TestWorker()
        receiver = MagicMock()
        w.failed.connect(receiver)
        w.cleanup()
        # After disconnect, emitting should not reach receiver
        w.failed.emit("test")
        receiver.assert_not_called()

    def test_cleanup_idempotent(self):
        """Calling cleanup twice should not raise."""
        w = _TestWorker()
        w.cleanup()
        w.cleanup()

    def test_cleanup_returns_true_when_stopped(self):
        w = _TestWorker()  # not running
        assert w.cleanup() is True

    def test_cleanup_returns_false_when_wait_times_out(self):
        """A worker that does not stop in time must report False so the caller
        keeps its reference — a live QThread freed by GC aborts the process."""
        w = _TestWorker()
        w.isRunning = MagicMock(return_value=True)
        w.stop = MagicMock()
        w.wait = MagicMock(return_value=False)
        assert w.cleanup() is False
        w.stop.assert_called_once()


# ---------------------------------------------------------------------------
class TestCleanupEmitsNoWarnings:
    """ML-092: on PySide6 a no-receiver disconnect() emits RuntimeWarning
    instead of raising — the old except branch was dead and 22 warnings
    spammed stderr, masking real ones. cleanup() must stay warning-free
    both with and without connected receivers."""

    def test_cleanup_unconnected_no_runtime_warnings(self):
        import warnings
        w = _TestWorker()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert w.cleanup() is True
        runtime = [x for x in rec if issubclass(x.category, RuntimeWarning)]
        assert not runtime, f"disconnect spam is back: {runtime[0].message}"

    def test_cleanup_connected_disconnects_without_warnings(self):
        import warnings
        w = _TestWorker()
        got = []
        w.failed.connect(got.append)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert w.cleanup() is True
        assert not [x for x in rec
                    if issubclass(x.category, RuntimeWarning)]
        w.failed.emit("late")           # disconnected — nothing arrives
        assert got == []


# Subclass cleanup — each worker type must support cleanup without error
# ---------------------------------------------------------------------------

class TestSubclassCleanup:
    """Verify cleanup() works on every worker subclass."""

    def _make_mock_client(self):
        client = MagicMock()
        client.is_open.return_value = True
        return client

    def test_scan_worker_cleanup(self):
        settings = MagicMock()
        w = ScanWorker(self._make_mock_client(), settings)
        receiver = MagicMock()
        w.progress.connect(receiver)
        w.finished.connect(receiver)
        w.comm_error.connect(receiver)
        w.failed.connect(receiver)
        w.cleanup()
        # All signals disconnected
        w.failed.emit("x")
        receiver.assert_not_called()

    def test_preheat_worker_cleanup(self):
        w = PreheatWorker(self._make_mock_client(), 6.3, 0.0, 30)
        receiver = MagicMock()
        w.progress.connect(receiver)
        w.finished.connect(receiver)
        w.cleanup()
        w.finished.emit()
        receiver.assert_not_called()

    def test_param_poller_cleanup(self):
        w = ParamPoller(self._make_mock_client(), interval_ms=500)
        receiver = MagicMock()
        w.updated.connect(receiver)
        w.cleanup()
        w.updated.emit({"ua": 0})
        receiver.assert_not_called()

    def test_check_com_worker_cleanup(self):
        w = CheckComWorker(self._make_mock_client())
        receiver = MagicMock()
        w.finished.connect(receiver)
        w.cleanup()
        w.finished.emit(42)
        receiver.assert_not_called()

    def test_reset_worker_cleanup(self):
        w = ResetWorker(self._make_mock_client(), 8.0)
        receiver = MagicMock()
        w.finished.connect(receiver)
        w.cleanup()
        w.finished.emit()
        receiver.assert_not_called()

    def test_health_worker_cleanup(self):
        w = HealthWorker(
            self._make_mock_client(),
            lamp=MagicMock(),
            app_config=MagicMock(),
            calibration=MagicMock(),
            lamp_id="L1",
            name="test",
            reference_mode="none",
            reference=None,
            emission_enabled=False,
            measurement_plan=None,
            warmup_s=0,
        )
        receiver = MagicMock()
        w.progress.connect(receiver)
        w.finished.connect(receiver)
        w.cleanup()
        w.finished.emit({})
        receiver.assert_not_called()


# ---------------------------------------------------------------------------
# HealthTab.shutdown
# ---------------------------------------------------------------------------

class TestHealthTabShutdown:
    """Test HealthTab.shutdown logic without creating real QWidgets."""

    def test_shutdown_no_worker(self):
        """shutdown() should not raise when worker is None."""
        from app.health_tab import HealthTab
        # Test the shutdown method logic directly via a mock object
        tab = MagicMock(spec=HealthTab)
        tab.worker = None
        # Call the real method on the mock
        HealthTab.shutdown(tab)

    def test_shutdown_with_running_worker(self):
        """shutdown() should call cleanup on a running worker."""
        from app.health_tab import HealthTab
        tab = MagicMock(spec=HealthTab)
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = True
        tab.worker = mock_worker
        HealthTab.shutdown(tab)
        mock_worker.cleanup.assert_called_once()

    def test_shutdown_with_stopped_worker(self):
        """shutdown() should not call cleanup if worker is not running."""
        from app.health_tab import HealthTab
        tab = MagicMock(spec=HealthTab)
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        tab.worker = mock_worker
        HealthTab.shutdown(tab)
        mock_worker.cleanup.assert_not_called()


# ---------------------------------------------------------------------------
# OptimizeWorker — must follow the BaseWorker pattern
# ---------------------------------------------------------------------------

class TestOptimizeWorkerInheritsBaseWorker:
    """``OptimizeWorker`` must inherit ``BaseWorker``.

    Pins the unified ``stop()`` / ``_stop_requested`` semantics, the
    ``cleanup()`` helper that disconnects signals on completion, and
    the standard error-emit pattern. Without ``cleanup``, signal
    connections accumulate across repeated optimization runs because
    dropping the Python reference (``self._opt_worker = None``) does
    not sever Qt slot connections.
    """

    def _make_worker(self):
        from app.optimize_worker import OptimizeWorker
        from lm19.optimizer import OptimizerConstraints
        return OptimizeWorker(points=[], constraints=OptimizerConstraints(),
                              ub=250.0)

    def test_inherits_base_worker(self):
        from app.optimize_worker import OptimizeWorker
        assert issubclass(OptimizeWorker, BaseWorker), \
            "OptimizeWorker must inherit BaseWorker for the unified " \
            "stop/cleanup/failed-signal pattern"

    def test_has_cleanup_method(self):
        w = self._make_worker()
        assert hasattr(w, "cleanup") and callable(w.cleanup)

    def test_inherits_failed_signal(self):
        """BaseWorker.failed is available on the worker (in addition to
        finished_err which is kept for the existing UI binding)."""
        w = self._make_worker()
        assert hasattr(w, "failed")
        assert hasattr(w, "finished_err")
        assert hasattr(w, "finished_ok")

    def test_cancel_aliases_stop(self):
        """``cancel()`` is an alias for ``stop()`` so the UI binding
        ``opt_cancel_btn.clicked.connect(worker.cancel)`` works
        without changes — internally it sets the same flag as
        ``stop()``."""
        w = self._make_worker()
        assert w._stop_requested is False
        w.cancel()
        assert w._stop_requested is True
        # Property alias also flips
        assert w._cancelled is True

    def test_stop_method_works(self):
        """The BaseWorker stop() method directly flips _stop_requested."""
        w = self._make_worker()
        w.stop()
        assert w._stop_requested is True
        assert w._cancelled is True  # alias view of same flag

    def test_cleanup_disconnects_finished_ok(self):
        """After cleanup, the receiver is no longer notified on emit."""
        w = self._make_worker()
        receiver = MagicMock()
        w.finished_ok.connect(receiver)
        w.cleanup()
        # After cleanup, signal should be disconnected — emitting now
        # must not call the receiver. (Qt allows emitting on disconnected
        # signal silently.)
        w.finished_ok.emit({"dummy": True})
        receiver.assert_not_called()

    def test_cleanup_safe_when_not_running(self):
        """cleanup() on a fresh (never started) worker doesn't raise."""
        w = self._make_worker()
        w.cleanup()  # would raise on stop()/wait() if mishandled

    def test_run_suppresses_error_when_cancelled(self, monkeypatch):
        """If _execute raises after cancel, finished_err must NOT fire
        (cancellation isn't a user-visible error)."""
        from app.optimize_worker import OptimizeWorker
        w = self._make_worker()
        # Replace _execute with one that raises after cancel
        def fake_execute(self_=w):
            self_.stop()
            raise RuntimeError("cancelled mid-sweep")
        monkeypatch.setattr(w, "_execute", fake_execute)
        receiver = MagicMock()
        w.finished_err.connect(receiver)
        w.run()
        receiver.assert_not_called()

    def test_run_emits_error_when_not_cancelled(self, monkeypatch):
        """A genuine exception (not from cancel) → finished_err fires."""
        w = self._make_worker()
        def fake_execute(self_=w):
            raise ValueError("real bug")
        monkeypatch.setattr(w, "_execute", fake_execute)
        receiver = MagicMock()
        w.finished_err.connect(receiver)
        w.run()
        receiver.assert_called_once()
        args, _ = receiver.call_args
        assert "real bug" in args[0]

    def test_run_emits_cancelled_when_stopped(self, monkeypatch):
        """On cancel, _execute returns silently at a cancel point — run() must
        emit the terminal ``finished_cancelled`` so the UI is not stuck in the
        running state (neither finished_ok nor finished_err fires)."""
        w = self._make_worker()

        def fake_execute(self_=w):
            return  # simulate the `if self._cancelled: return` path

        monkeypatch.setattr(w, "_execute", fake_execute)
        w.stop()
        cancelled = MagicMock()
        ok = MagicMock()
        w.finished_cancelled.connect(cancelled)
        w.finished_ok.connect(ok)
        w.run()
        cancelled.assert_called_once()
        ok.assert_not_called()

    def test_run_cancelled_on_suppressed_error(self, monkeypatch):
        """A mid-cancel exception is suppressed AND still terminates with
        finished_cancelled (not finished_err)."""
        w = self._make_worker()

        def fake_execute(self_=w):
            raise RuntimeError("scipy interrupted")

        monkeypatch.setattr(w, "_execute", fake_execute)
        w.stop()
        cancelled = MagicMock()
        err = MagicMock()
        w.finished_cancelled.connect(cancelled)
        w.finished_err.connect(err)
        w.run()
        cancelled.assert_called_once()
        err.assert_not_called()

    def test_run_no_cancelled_on_normal_completion(self, monkeypatch):
        """Normal (uncancelled) completion must NOT emit finished_cancelled."""
        w = self._make_worker()

        def fake_execute(self_=w):
            self_.finished_ok.emit({"dummy": True})

        monkeypatch.setattr(w, "_execute", fake_execute)
        cancelled = MagicMock()
        w.finished_cancelled.connect(cancelled)
        w.run()
        cancelled.assert_not_called()


# ---------------------------------------------------------------------------
# _CompareWorker and SrkWorker — same BaseWorker pattern
# ---------------------------------------------------------------------------

class TestCompareWorkerInheritsBaseWorker:
    """``_CompareWorker`` (model dialog "Compare all" feature) inherits
    ``BaseWorker`` for the same reason as ``OptimizeWorker``: unified
    stop/cleanup pattern."""

    def _make_worker(self):
        from app.model_dialog import _CompareWorker
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        return _CompareWorker(pts, "pentode")

    def test_inherits_base_worker(self):
        from app.model_dialog import _CompareWorker
        assert issubclass(_CompareWorker, BaseWorker)

    def test_has_cleanup(self):
        w = self._make_worker()
        assert hasattr(w, "cleanup") and callable(w.cleanup)

    def test_inherits_failed_signal(self):
        """``failed`` from BaseWorker is available (no separate signal)."""
        w = self._make_worker()
        assert hasattr(w, "failed")
        assert hasattr(w, "finished_ok")  # specific to compare
        assert hasattr(w, "progress")

    def test_cancel_aliases_stop(self):
        w = self._make_worker()
        assert w._stop_requested is False
        w.cancel()
        assert w._stop_requested is True
        assert w._cancelled is True  # property alias

    def test_run_suppresses_error_when_cancelled(self, monkeypatch):
        from app.model_dialog import _CompareWorker
        w = self._make_worker()
        def fake_execute(self_=w):
            self_.stop()
            raise RuntimeError("cancelled mid-fit")
        monkeypatch.setattr(w, "_execute", fake_execute)
        receiver = MagicMock()
        w.failed.connect(receiver)
        w.run()
        receiver.assert_not_called()


class TestHealthWorkerStopSuppression:
    """``HealthWorker.run`` suppresses the ``failed`` signal on a user stop:
    ``run_health_test`` raises 'Health test stopped' when stopped mid-SRK (so
    no partial result is emitted as complete), which must NOT surface as an
    error dialog (#669)."""

    def _make_worker(self):
        from app.workers import HealthWorker
        return HealthWorker(
            MagicMock(), lamp=MagicMock(), app_config=MagicMock(),
            calibration=MagicMock(), lamp_id="L1", name="t",
            reference_mode="none", reference=None, emission_enabled=False,
            measurement_plan=None, warmup_s=0)

    def test_run_suppresses_failed_when_stopped(self, monkeypatch):
        w = self._make_worker()

        def boom(self_=w):
            raise RuntimeError("Health test stopped")

        monkeypatch.setattr(w, "_execute", boom)
        w._stop_requested = True
        spy = MagicMock()
        w.failed.connect(spy)
        w.run()
        spy.assert_not_called()

    def test_run_emits_failed_when_not_stopped(self, monkeypatch):
        w = self._make_worker()

        def boom(self_=w):
            raise ValueError("real bug")

        monkeypatch.setattr(w, "_execute", boom)
        spy = MagicMock()
        w.failed.connect(spy)
        w.run()
        spy.assert_called_once()


class TestHealthStopCommonDrain:
    """``HealthTab._stop_common`` must drain the worker via ``cleanup()->bool``
    and keep the reference if it does not stop in time — a live QThread freed
    by GC aborts the process, and resetting outputs under a live worker races
    its Ua/Ug1 commands (#669 / _stop_common wait race)."""

    def _tab(self):
        # Plain mock (not spec=HealthTab): _stop_common touches ctx-callback
        # attributes (set_poller_active etc.) assigned in __init__, which a
        # class-spec mock would not provide.
        return MagicMock()

    def test_keeps_hung_worker(self):
        from app.health_tab import HealthTab
        tab = self._tab()
        w = MagicMock()
        w.isRunning.return_value = True
        w.cleanup.return_value = False  # did not stop in time
        tab.worker = w
        HealthTab._stop_common(tab, keep_heater=True)
        w.cleanup.assert_called_once_with(timeout_ms=2000)
        assert tab.worker is w  # retained, not nulled

    def test_nulls_drained_worker(self):
        from app.health_tab import HealthTab
        tab = self._tab()
        w = MagicMock()
        w.isRunning.return_value = True
        w.cleanup.return_value = True
        tab.worker = w
        HealthTab._stop_common(tab, keep_heater=True)
        assert tab.worker is None


class TestSrkWorkerInheritsBaseWorker:
    """``SrkWorker`` (S/R/K corner measurement) inherits ``BaseWorker``
    so the ``failed`` signal lives on the base class, not duplicated
    on the subclass."""

    def _make_worker(self):
        from app.srk_widget import SrkWorker
        from lm19.scan import SrkSettings
        client = MagicMock()
        settings = SrkSettings(ua_min=100, ua_max=100,
                                ug1_min=-2, ug1_max=-2, ug2=0)
        return SrkWorker(client, settings, repeats=1)

    def test_inherits_base_worker(self):
        from app.srk_widget import SrkWorker
        assert issubclass(SrkWorker, BaseWorker)

    def test_has_cleanup(self):
        w = self._make_worker()
        assert hasattr(w, "cleanup")

    def test_failed_signal_inherited(self):
        """``failed`` is BaseWorker's, not duplicated locally."""
        w = self._make_worker()
        assert hasattr(w, "failed")
        # The signal is defined on BaseWorker, not redeclared on SrkWorker.
        # Verify by checking class-level vars.
        from app.srk_widget import SrkWorker
        # SrkWorker should NOT have its own ``failed`` class attribute —
        # it inherits BaseWorker's. (If a future refactor re-adds it,
        # signals get shadowed and BaseWorker.cleanup may miss them.)
        assert "failed" not in vars(SrkWorker), (
            "SrkWorker should not redeclare 'failed' — inherit from BaseWorker"
        )

    def test_stop_alias_property_works(self):
        """``_stop`` is preserved as a property aliasing _stop_requested."""
        w = self._make_worker()
        assert w._stop is False
        w.stop()
        assert w._stop is True
        assert w._stop_requested is True

    def test_client_passed_through(self):
        """SrkWorker still accepts a serial client (unlike OptimizeWorker)."""
        client = MagicMock()
        from app.srk_widget import SrkWorker
        from lm19.scan import SrkSettings
        settings = SrkSettings(ua_min=100, ua_max=100,
                                ug1_min=-2, ug1_max=-2, ug2=0)
        w = SrkWorker(client, settings)
        assert w.client is client


# ---------------------------------------------------------------------------
# Controllers must cleanup() stale workers before reassignment
# ---------------------------------------------------------------------------

class TestStaleWorkerCleanupOnRestart:
    """``ScanController.start_scan`` / ``start_preheat`` / ``reset_outputs``
    and ``ConnectionManager.check_com`` reassign their worker attribute on
    every invocation. Without an explicit ``cleanup()`` on the prior
    instance, queued cross-thread signals from the just-exited worker keep
    firing the controller's slots — contaminating the new worker's UI
    state. ``cleanup()`` disconnects all signals via the MRO walk in
    ``BaseWorker.cleanup``.

    Worst case (without cleanup): a stale ``comm_error`` from old scan
    pops a dialog, user responds, and ``respond_comm_error`` routes the
    answer to the *new* worker because ``self._scan_worker`` already
    points there.
    """

    def _make_scan_controller(self):
        from app.scan_controller import ScanController
        from lm19.app_config import AppConfig
        return ScanController(AppConfig())

    def _make_connection_manager(self):
        from app.connection_manager import ConnectionManager
        from lm19.app_config import AppConfig
        return ConnectionManager(AppConfig())

    def test_start_scan_cleans_up_stale_worker(self):
        ctrl = self._make_scan_controller()
        stale = MagicMock()
        stale.isRunning.return_value = False  # exited but reference still held
        ctrl._scan_worker = stale
        with patch("app.scan_controller.ScanWorker") as MockScan, \
             patch("app.scan_controller.scan_point_count", return_value=0):
            MockScan.return_value = MagicMock()
            ctrl.start_scan(MagicMock(), MagicMock())
        stale.cleanup.assert_called_once()

    def test_start_scan_no_cleanup_when_no_prior_worker(self):
        """First-ever call must not crash on the absent attribute."""
        ctrl = self._make_scan_controller()
        assert ctrl._scan_worker is None
        with patch("app.scan_controller.ScanWorker") as MockScan, \
             patch("app.scan_controller.scan_point_count", return_value=0):
            MockScan.return_value = MagicMock()
            ctrl.start_scan(MagicMock(), MagicMock())  # would AttributeError if buggy

    def test_start_scan_returns_early_when_still_running(self):
        """Existing ``isRunning(): return`` guard is preserved — must
        NOT cleanup the still-running worker (would block UI on wait)."""
        ctrl = self._make_scan_controller()
        running = MagicMock()
        running.isRunning.return_value = True
        ctrl._scan_worker = running
        with patch("app.scan_controller.ScanWorker") as MockScan:
            ctrl.start_scan(MagicMock(), MagicMock())
            MockScan.assert_not_called()
        running.cleanup.assert_not_called()

    def test_start_preheat_cleans_up_stale_worker(self):
        ctrl = self._make_scan_controller()
        stale = MagicMock()
        stale.isRunning.return_value = False
        ctrl._preheat_worker = stale
        with patch("app.scan_controller.PreheatWorker") as MockPreheat:
            MockPreheat.return_value = MagicMock()
            ctrl.start_preheat(MagicMock(), 6.3, 0.0, 30)
        stale.cleanup.assert_called_once()

    def test_reset_outputs_cleans_up_stale_worker(self):
        ctrl = self._make_scan_controller()
        client = MagicMock()
        client.is_open.return_value = True
        stale = MagicMock()
        stale.isRunning.return_value = False
        ctrl._reset_worker = stale
        with patch("app.scan_controller.ResetWorker") as MockReset:
            MockReset.return_value = MagicMock()
            ctrl.reset_outputs(client)
        stale.cleanup.assert_called_once()

    def test_check_com_cleans_up_stale_worker(self):
        cm = self._make_connection_manager()
        # check_com gates on is_connected — fake a live client
        cm._client = MagicMock()
        cm._client.is_open.return_value = True
        stale = MagicMock()
        stale.isRunning.return_value = False
        cm._check_com_worker = stale
        with patch("app.connection_manager.CheckComWorker") as MockCheck:
            MockCheck.return_value = MagicMock()
            cm.check_com()
        stale.cleanup.assert_called_once()


class TestStaleSignalIsolation:
    """End-to-end behavioural check: a stale signal emit on the old
    worker (after restart) must NOT propagate through the controller's
    upstream signal. Uses real worker QObjects so the Qt-disconnect
    path is exercised, not just MagicMock ``cleanup`` calls.
    """

    def test_scan_progress_from_stale_worker_does_not_propagate(self):
        from app.scan_controller import ScanController
        from lm19.app_config import AppConfig

        # QApplication is required for Signal/slot machinery in tests that
        # actually emit. We use the existing instance if one exists.
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])

        ctrl = ScanController(AppConfig())

        spy = MagicMock()
        ctrl.scan_progress.connect(spy)

        # Inject a real ScanWorker as "previous" without starting it.
        # Spoof isRunning() so start_scan thinks it has exited.
        first = ScanWorker(MagicMock(), MagicMock())
        first.isRunning = MagicMock(return_value=False)
        ctrl._scan_worker = first

        # Manually wire the signal as start_scan would have, so we can
        # simulate "queued signal still hooked to controller slot".
        first.progress.connect(ctrl._on_scan_progress)

        # Sanity: emit BEFORE the restart — slot fires, spy called.
        first.progress.emit({"ua": 100})
        assert spy.call_count == 1, "baseline: emit before restart should propagate"
        spy.reset_mock()

        # Restart — controller must cleanup() first, breaking the slot link.
        with patch("app.scan_controller.ScanWorker") as MockScan, \
             patch("app.scan_controller.scan_point_count", return_value=0):
            MockScan.return_value = MagicMock()
            ctrl.start_scan(MagicMock(), MagicMock())

        # Now the supposedly-queued stale emit fires AFTER restart.
        # Without cleanup it would still call _on_scan_progress.
        first.progress.emit({"ua": 999})
        assert spy.call_count == 0, (
            "stale signal from old worker leaked through after restart "
            "— cleanup-before-reassign regressed"
        )


class TestAllWorkersUseBaseWorker:
    """Pin: every ``QThread`` subclass in ``app/`` should inherit BaseWorker.

    Catches a future regression where someone adds a new direct-QThread
    worker bypassing the unified cleanup/stop pattern.
    """

    def test_no_direct_qthread_workers(self):
        import re
        import glob
        from pathlib import Path
        from PySide6.QtCore import QThread
        # Walk all app/*.py and find class X(QThread): patterns
        # — should be empty (everything goes through BaseWorker now).
        # Exception: BaseWorker itself in workers.py.
        # Anchor from this file (not CWD) so the pin can't pass vacuously.
        app_root = Path(__file__).resolve().parents[1] / "app"
        files = sorted(glob.glob(str(app_root / "**" / "*.py"), recursive=True))
        assert files, f"No app/*.py found under {app_root} — pin would pass vacuously"
        offenders = []
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                src = fh.read()
            # Match `class Foo(QThread)` but not `class BaseWorker(QThread)`
            for m in re.finditer(r"^class (\w+)\(QThread\)", src, re.MULTILINE):
                cls_name = m.group(1)
                if cls_name == "BaseWorker":
                    continue
                offenders.append((f, cls_name))
        assert not offenders, (
            f"Workers must inherit BaseWorker, not QThread directly. "
            f"Direct QThread subclasses found: {offenders}"
        )


# ---------------------------------------------------------------------------
# closeEvent / dialog-close must stop in-memory compute workers
# ---------------------------------------------------------------------------

class TestMainWindowCloseEventOptWorker:
    """``MainWindow.closeEvent`` must stop a running ``OptimizeWorker`` (via
    ``_shutdown_opt_worker``). It is a ``QThread`` child of the window with no
    serial client; if an optimization is still running at close, a live
    ``QThread`` freed by GC calls ``qFatal`` and aborts the process. The other
    workers are drained via collaborators (scan_ctrl/health_tab/srk/conn_mgr)
    — only the opt worker was previously unhandled.
    """

    def test_shutdown_opt_worker_cleans_up_and_nulls(self):
        from app.main_window import MainWindow
        win = MagicMock()
        opt = MagicMock()
        opt.cleanup.return_value = True
        win._opt_worker = opt
        MainWindow._shutdown_opt_worker(win)
        opt.cleanup.assert_called_once()
        assert win._opt_worker is None

    def test_shutdown_opt_worker_keeps_ref_when_not_drained(self):
        """cleanup()==False → keep the reference; a live QThread freed by GC
        aborts the process."""
        from app.main_window import MainWindow
        win = MagicMock()
        opt = MagicMock()
        opt.cleanup.return_value = False
        win._opt_worker = opt
        MainWindow._shutdown_opt_worker(win)
        opt.cleanup.assert_called_once()
        assert win._opt_worker is opt

    def test_shutdown_opt_worker_noop_when_none(self):
        from app.main_window import MainWindow
        win = MagicMock()
        win._opt_worker = None
        MainWindow._shutdown_opt_worker(win)  # must not raise

    def test_shutdown_opt_worker_noop_when_attr_absent(self):
        """``_opt_worker`` is created lazily on first optimization — closeEvent
        before any optimization must not AttributeError."""
        from app.main_window import MainWindow

        class _Bare:
            pass
        bare = _Bare()
        MainWindow._shutdown_opt_worker(bare)  # getattr(..., None) guards


class TestModelDialogCompareWorkerClose:
    """``ModelDialog.done()`` must stop a running ``_CompareWorker`` (via
    ``_stop_compare_worker``) so closing the dialog mid-compare (Esc / window
    ✕ / accepting button) does not leave a live ``QThread`` child that aborts
    the process when freed by GC.
    """

    def test_stop_compare_worker_cleans_up_and_nulls(self):
        from app.model_dialog import ModelDialog
        dlg = MagicMock()
        w = MagicMock()
        w.cleanup.return_value = True
        dlg._compare_worker = w
        ModelDialog._stop_compare_worker(dlg)
        w.cleanup.assert_called_once()
        assert dlg._compare_worker is None

    def test_stop_compare_worker_keeps_ref_when_not_drained(self):
        from app.model_dialog import ModelDialog
        dlg = MagicMock()
        w = MagicMock()
        w.cleanup.return_value = False
        dlg._compare_worker = w
        ModelDialog._stop_compare_worker(dlg)
        assert dlg._compare_worker is w

    def test_stop_compare_worker_noop_when_none(self):
        from app.model_dialog import ModelDialog
        dlg = MagicMock()
        dlg._compare_worker = None
        ModelDialog._stop_compare_worker(dlg)  # must not raise

    def test_done_stops_compare_worker_real_dialog(self):
        """Real-dialog wiring: ``done()`` (the QDialog hook for accept /
        reject / window-✕ / Esc) must invoke ``_stop_compare_worker`` AND
        still close via ``super().done()``. Catches a wrong override
        signature (which would silently NOT override) and a missing super
        call (dialog would never close)."""
        from PySide6.QtWidgets import QApplication
        from app.model_dialog import ModelDialog
        QApplication.instance() or QApplication([])
        dlg = ModelDialog()
        finished_spy = MagicMock()
        dlg.finished.connect(finished_spy)  # QDialog.done(r) emits finished(r)
        w = MagicMock()
        w.cleanup.return_value = True
        dlg._compare_worker = w
        dlg.done(0)  # QDialog.DialogCode.Rejected
        w.cleanup.assert_called_once()
        assert dlg._compare_worker is None
        finished_spy.assert_called_once_with(0)  # super().done(0) actually ran


# ---------------------------------------------------------------------------
# reset_outputs must drain a running preheat worker (heater re-assert race)
# ---------------------------------------------------------------------------

class TestResetOutputsPreheatDrain:
    """``reset_outputs`` must drain a still-running preheat worker BEFORE
    zeroing the heater. ``PreheatWorker`` writes Uh/Ih setpoints between ramp
    steps; without draining it can re-assert the heater after ``ResetWorker``
    zeroes it (the firmware serializes single commands, not the two logical
    sequences). Only applies when ``reset_heater=True`` — keep-heater resets
    (post-scan) must leave a warming preheat alone.
    """

    def _make_scan_controller(self):
        from app.scan_controller import ScanController
        from lm19.app_config import AppConfig
        return ScanController(AppConfig())

    def _running_preheat(self):
        w = MagicMock()
        w.isRunning.return_value = True
        w.cleanup.return_value = True
        return w

    def _open_client(self):
        client = MagicMock()
        client.is_open.return_value = True
        return client

    def test_drains_running_preheat_when_reset_heater(self):
        ctrl = self._make_scan_controller()
        pre = self._running_preheat()
        ctrl._preheat_worker = pre
        with patch("app.scan_controller.ResetWorker") as MockReset:
            MockReset.return_value = MagicMock()
            ctrl.reset_outputs(self._open_client(), reset_heater=True)
        pre.cleanup.assert_called_once()
        assert ctrl._preheat_worker is None

    def test_does_not_drain_preheat_when_reset_heater_false(self):
        ctrl = self._make_scan_controller()
        pre = self._running_preheat()
        ctrl._preheat_worker = pre
        with patch("app.scan_controller.ResetWorker") as MockReset:
            MockReset.return_value = MagicMock()
            ctrl.reset_outputs(self._open_client(), reset_heater=False)
        pre.cleanup.assert_not_called()
        assert ctrl._preheat_worker is pre

    def test_ignores_stopped_preheat(self):
        ctrl = self._make_scan_controller()
        pre = MagicMock()
        pre.isRunning.return_value = False
        ctrl._preheat_worker = pre
        with patch("app.scan_controller.ResetWorker") as MockReset:
            MockReset.return_value = MagicMock()
            ctrl.reset_outputs(self._open_client(), reset_heater=True)
        pre.cleanup.assert_not_called()

    def test_keeps_preheat_ref_when_not_drained(self):
        """cleanup()==False → keep the live preheat QThread reference, but the
        reset MUST still proceed: zeroing outputs is the safety priority, the
        un-drained preheat is only logged (failure-visibility)."""
        ctrl = self._make_scan_controller()
        pre = MagicMock()
        pre.isRunning.return_value = True
        pre.cleanup.return_value = False
        ctrl._preheat_worker = pre
        with patch("app.scan_controller.ResetWorker") as MockReset:
            reset_inst = MagicMock()
            MockReset.return_value = reset_inst
            ctrl.reset_outputs(self._open_client(), reset_heater=True)
        assert ctrl._preheat_worker is pre  # live QThread retained, not GC'd
        MockReset.assert_called_once()       # reset still proceeds (safety)
        reset_inst.start.assert_called_once()


class TestPreheatWorkerStopGate:
    """PreheatWorker must not (re-)assert the heater once a stop is flagged —
    otherwise, after an emergency synchronous zero, the in-flight ramp could
    re-energize Uh/Ih (audit finding #2). The warmup-hold loop never
    writes the heater (read-only), so only the ramp branches need the gate.
    """

    def _heater_writes(self, client):
        return [c for c in client.set_param.call_args_list
                if c.args and c.args[0] in ("Uh", "Ih")]

    def test_no_heater_write_when_stopped_ramp_up(self):
        from app.workers import PreheatWorker
        client = MagicMock()
        client.get_param.return_value = 0  # start below target → ramp up
        w = PreheatWorker(client, target_uh=6.3, target_ih=0.0, warmup_s=5)
        w._stop_requested = True
        w._execute()
        assert self._heater_writes(client) == []

    def test_no_heater_write_when_stopped_ramp_down(self):
        from app.workers import PreheatWorker
        from lm19.protocol import encode_uh
        client = MagicMock()
        client.get_param.return_value = encode_uh(7.0)  # start above target
        w = PreheatWorker(client, target_uh=6.3, target_ih=0.0, warmup_s=5)
        w._stop_requested = True
        w._execute()
        assert self._heater_writes(client) == []

    def test_heater_written_when_not_stopped(self):
        """Sanity: with no stop, the ramp DOES drive the heater (gate is not
        over-eager)."""
        from app.workers import PreheatWorker
        client = MagicMock()
        client.get_param.return_value = 0  # start below target → ramp up
        w = PreheatWorker(client, target_uh=6.3, target_ih=0.0, warmup_s=5)
        # Avoid real sleeping in the ramp + warmup loops.
        w.msleep = MagicMock()
        w._stop_requested = False
        w._execute()
        assert self._heater_writes(client), "ramp should drive the heater"


class TestConnectionManagerZombieRetention:
    """A poller/worker that does not drain within its cleanup timeout must be
    RETAINED (live QThread freed by GC aborts the process). `_stop_poller`
    keeps it, but `_start_poller` then reassigns `_param_poller` and `shutdown`
    nulled it unconditionally — both dropped the only reference. The zombie now
    goes into `_zombies` instead.
    """

    def _cm(self):
        from app.connection_manager import ConnectionManager
        from lm19.app_config import AppConfig
        return ConnectionManager(AppConfig())

    def test_start_poller_stashes_undrained_poller(self):
        cm = self._cm()
        cm._client = MagicMock()
        hung = MagicMock()
        hung.cleanup.return_value = False  # _stop_poller retains it (non-None)
        cm._param_poller = hung
        with patch("app.connection_manager.ParamPoller") as MockP:
            MockP.return_value = MagicMock()
            cm._start_poller()
        assert hung in cm._zombies          # retained, not clobbered
        assert cm._param_poller is not hung  # fresh poller created

    def test_start_poller_drops_drained_poller(self):
        cm = self._cm()
        cm._client = MagicMock()
        ok = MagicMock()
        ok.cleanup.return_value = True
        cm._param_poller = ok
        with patch("app.connection_manager.ParamPoller") as MockP:
            MockP.return_value = MagicMock()
            cm._start_poller()
        assert ok not in cm._zombies  # cleanly drained → not a zombie

    def test_shutdown_stashes_undrained_worker(self):
        cm = self._cm()
        hung = MagicMock()
        hung.cleanup.return_value = False
        cm._param_poller = hung
        cm._check_com_worker = None
        cm._client = None
        cm.shutdown()
        assert hung in cm._zombies
        assert cm._param_poller is None

    def test_shutdown_drops_drained_worker(self):
        cm = self._cm()
        ok = MagicMock()
        ok.cleanup.return_value = True
        cm._param_poller = ok
        cm._check_com_worker = None
        cm._client = None
        cm.shutdown()
        assert ok not in cm._zombies
        assert cm._param_poller is None


class TestPreheatProgressStaleGuard:
    """``_on_preheat_progress`` must ignore stale progress events delivered
    after the preheat worker was drained on Stop (``scan_ctrl.preheat_worker``
    is None) — otherwise a queued event flips the status back to "Warmup
    remaining" after the UI already showed "Stopped".
    """

    def test_ignores_progress_when_worker_drained(self):
        from app.main_window import MainWindow
        win = MagicMock()
        win.scan_ctrl.preheat_worker = None
        MainWindow._on_preheat_progress(win, 6.3, 0.5, 10)
        win.preheat_progress.setValue.assert_not_called()
        win.preheat_status.setText.assert_not_called()

    def test_processes_progress_when_worker_active(self):
        from app.main_window import MainWindow
        win = MagicMock()
        win.scan_ctrl.preheat_worker = MagicMock()  # active
        win.preheat_seconds.value.return_value = 30
        MainWindow._on_preheat_progress(win, 6.3, 0.5, 10)
        win.preheat_progress.setValue.assert_called()


class TestCompareFailedVisibility:
    """ML-088: the _CompareWorker.failed signal was never connected — a
    compare error left the status stuck on "Fitting X" with no message."""

    @staticmethod
    def _ensure_qapp():
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])

    def test_on_compare_all_connects_failed(self, monkeypatch):
        """Wiring pin: starting a compare must connect failed to the
        handler. Spy is installed BEFORE _on_compare_all so Qt binds the
        connection to it (start is stubbed so no thread runs)."""
        self._ensure_qapp()
        from app.model_dialog import ModelDialog, _CompareWorker
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        dlg = ModelDialog(points=pts, is_triode=False)
        monkeypatch.setattr(_CompareWorker, "start", lambda self: None)
        failed = []
        monkeypatch.setattr(dlg, "_on_compare_failed",
                            lambda msg: failed.append(msg))
        dlg._on_compare_all()
        try:
            worker = dlg._compare_worker
            assert worker is not None
            worker.failed.emit("boom")
            assert failed == ["boom"],                 "failed signal not wired to _on_compare_failed"
        finally:
            dlg._stop_compare_worker()
            dlg.deleteLater()

    def test_on_compare_failed_sets_status(self):
        self._ensure_qapp()
        from app.model_dialog import ModelDialog
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        dlg = ModelDialog(points=pts, is_triode=False)
        try:
            dlg._on_compare_failed("boom")
            assert "boom" in dlg._compare_status.text()
        finally:
            dlg.deleteLater()
