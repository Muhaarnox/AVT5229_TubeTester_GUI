"""Tests for ScanController."""

from unittest.mock import MagicMock, patch

import pytest

from app.scan_controller import ScanController
from lm19.app_config import AppConfig


@pytest.fixture
def app_config():
    cfg = MagicMock(spec=AppConfig)
    cfg.ug1_after_stop = 8.0
    cfg.ug1_settle_s = 0.3
    return cfg


@pytest.fixture
def ctrl(app_config):
    return ScanController(app_config)


class TestScanControllerInit:
    def test_initial_state(self, ctrl):
        assert ctrl.is_scanning is False
        assert ctrl.scan_worker is None
        assert ctrl.preheat_worker is None
        assert ctrl.reset_worker is None
        assert ctrl.preheat_done is False
        assert ctrl.scan_start_time is None


class TestScanLifecycle:
    @patch("app.scan_controller.ScanWorker")
    @patch("app.scan_controller.scan_point_count", return_value=100)
    def test_start_scan(self, mock_count, MockWorker, ctrl):
        mock_worker = MockWorker.return_value
        mock_worker.isRunning.return_value = False
        client = MagicMock()
        settings = MagicMock()

        started = []
        ctrl.scan_started.connect(lambda: started.append(True))

        ctrl.start_scan(client, settings)

        assert ctrl.is_scanning is True
        assert ctrl.scan_total_points == 100
        assert ctrl.scan_start_time is not None
        MockWorker.assert_called_once_with(client, settings)
        mock_worker.start.assert_called_once()
        assert started == [True]

    @patch("app.scan_controller.ScanWorker")
    @patch("app.scan_controller.scan_point_count", return_value=10)
    def test_stop_scan(self, mock_count, MockWorker, ctrl):
        mock_worker = MockWorker.return_value
        mock_worker.isRunning.return_value = False
        ctrl.start_scan(MagicMock(), MagicMock())

        ctrl.stop_scan()
        mock_worker.stop.assert_called_once()
        assert ctrl.reset_on_finish is True

    @patch("app.scan_controller.ScanWorker")
    @patch("app.scan_controller.scan_point_count", return_value=10)
    def test_scan_finished_resets_state(self, mock_count, MockWorker, ctrl):
        mock_worker = MockWorker.return_value
        mock_worker.isRunning.return_value = False
        ctrl.start_scan(MagicMock(), MagicMock())

        finished = []
        ctrl.scan_finished.connect(lambda pts: finished.append(pts))

        # Simulate finished signal
        ctrl._on_scan_finished([{"ua": 100, "ia": 5}])
        assert ctrl.is_scanning is False
        assert len(finished) == 1


class TestPreheat:
    @patch("app.scan_controller.PreheatWorker")
    def test_start_preheat(self, MockWorker, ctrl):
        mock_worker = MockWorker.return_value
        mock_worker.isRunning.return_value = False
        client = MagicMock()

        started = []
        ctrl.preheat_started.connect(lambda: started.append(True))

        ctrl.start_preheat(client, 6.3, 0.0, 30)
        assert ctrl.preheat_done is False
        MockWorker.assert_called_once_with(client, 6.3, 0.0, 30)
        mock_worker.start.assert_called_once()
        assert started == [True]

    @patch("app.scan_controller.PreheatWorker")
    def test_preheat_finished(self, MockWorker, ctrl):
        mock_worker = MockWorker.return_value
        mock_worker.isRunning.return_value = False
        ctrl.start_preheat(MagicMock(), 6.3, 0.0, 30)

        ctrl._on_preheat_finished()
        assert ctrl.preheat_done is True


class TestReset:
    @patch("app.scan_controller.ResetWorker")
    def test_reset_outputs(self, MockWorker, ctrl):
        mock_worker = MockWorker.return_value
        mock_worker.isRunning.return_value = False
        client = MagicMock()
        client.is_open.return_value = True

        ctrl.reset_outputs(client, reset_heater=True, reset_order=["Ug2", "Ug1", "Ua", "Uh"])
        MockWorker.assert_called_once()
        mock_worker.start.assert_called_once()

    def test_reset_no_client(self, ctrl):
        # Should not raise
        ctrl.reset_outputs(None)


class TestStopAll:
    @patch("app.scan_controller.ScanWorker")
    @patch("app.scan_controller.scan_point_count", return_value=10)
    def test_stop_all_stops_scan(self, mock_count, MockWorker, ctrl):
        mock_worker = MockWorker.return_value
        mock_worker.isRunning.return_value = True
        ctrl.start_scan(MagicMock(), MagicMock())

        ctrl.stop_all()
        mock_worker.stop.assert_called()
        assert ctrl.is_scanning is False


class TestShutdown:
    def test_shutdown_when_idle(self, ctrl):
        # Should not raise
        ctrl.shutdown()

    def test_shutdown_keeps_running_worker_reference(self, ctrl):
        """A worker that won't stop keeps its ref so a live QThread is not
        freed by GC (which would abort the process)."""
        w = MagicMock()
        w.isRunning.return_value = True
        w.wait.return_value = False
        ctrl._scan_worker = w
        ctrl.shutdown()
        w.stop.assert_called()
        assert ctrl._scan_worker is w

    def test_shutdown_drops_stopped_worker_reference(self, ctrl):
        w = MagicMock()
        w.isRunning.side_effect = [True, False]  # running, then stopped
        ctrl._scan_worker = w
        ctrl.shutdown()
        assert ctrl._scan_worker is None
