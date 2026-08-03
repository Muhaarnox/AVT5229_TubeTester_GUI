"""Tests for ConnectionManager."""

from unittest.mock import MagicMock, patch

import pytest

from app.connection_manager import ConnectionManager
from lm19.app_config import AppConfig


@pytest.fixture
def app_config():
    cfg = MagicMock(spec=AppConfig)
    cfg.serial_timeout_s = 1.0
    cfg.serial_write_timeout_s = 1.0
    cfg.read_param_timeout_s = 0.5
    cfg.read_lcd_timeout_s = 0.5
    cfg.serial_set_param_delay_s = 0.05
    cfg.live_poll_ms = 500
    cfg.live_poll_during_test = True
    return cfg


@pytest.fixture
def conn_mgr(app_config):
    return ConnectionManager(app_config)


class TestConnectionManagerInit:
    def test_initial_state(self, conn_mgr):
        assert conn_mgr.client is None
        assert conn_mgr.is_connected is False

    def test_set_poller_active_when_no_poller(self, conn_mgr):
        # Should not raise when no poller exists
        conn_mgr.set_poller_active(True)
        conn_mgr.set_poller_active(False)

    def test_stop_poller_drops_ref_when_cleanup_succeeds(self, conn_mgr):
        poller = MagicMock()
        poller.cleanup.return_value = True
        conn_mgr._param_poller = poller
        conn_mgr._stop_poller()
        poller.cleanup.assert_called_once()
        assert conn_mgr._param_poller is None

    def test_stop_poller_keeps_ref_when_cleanup_fails(self, conn_mgr):
        """A poller that won't stop must not be orphaned (live QThread GC abort)."""
        poller = MagicMock()
        poller.cleanup.return_value = False
        conn_mgr._param_poller = poller
        conn_mgr._stop_poller()
        assert conn_mgr._param_poller is poller


class TestConnectDisconnect:
    @patch("app.connection_manager.ParamPoller")
    @patch("app.connection_manager.LM19Serial")
    def test_connect_emits_signal(self, MockSerial, MockPoller, conn_mgr):
        mock_client = MockSerial.return_value
        mock_client.is_open.return_value = True
        mock_poller = MockPoller.return_value
        mock_poller.isRunning.return_value = False

        signals = []
        conn_mgr.connected.connect(lambda port: signals.append(port))

        conn_mgr.connect_port("COM3")

        assert conn_mgr.client is mock_client
        assert conn_mgr.is_connected is True
        mock_client.open.assert_called_once()
        assert signals == ["COM3"]
        MockPoller.assert_called_once()
        mock_poller.start.assert_called_once()

    @patch("app.connection_manager.ParamPoller")
    @patch("app.connection_manager.LM19Serial")
    def test_disconnect_emits_signal(self, MockSerial, MockPoller, conn_mgr):
        mock_client = MockSerial.return_value
        mock_client.is_open.return_value = True
        mock_poller = MockPoller.return_value
        mock_poller.isRunning.return_value = False

        conn_mgr.connect_port("COM3")

        signals = []
        conn_mgr.disconnected.connect(lambda: signals.append(True))
        conn_mgr.disconnect()

        assert conn_mgr.client is None
        assert conn_mgr.is_connected is False
        assert signals == [True]
        mock_client.close.assert_called_once()

    @patch("app.connection_manager.ParamPoller")
    @patch("app.connection_manager.LM19Serial")
    def test_set_poller_active_delegates(self, MockSerial, MockPoller, conn_mgr,
                                         app_config):
        """With live polling paused during tests, both directions delegate."""
        app_config.live_poll_during_test = False
        mock_poller = MockPoller.return_value
        mock_poller.isRunning.return_value = False
        conn_mgr.connect_port("COM3")

        conn_mgr.set_poller_active(False)
        mock_poller.set_active.assert_called_with(False)

        conn_mgr.set_poller_active(True)
        mock_poller.set_active.assert_called_with(True)

    @patch("app.connection_manager.ParamPoller")
    @patch("app.connection_manager.LM19Serial")
    def test_pause_ignored_when_polling_during_test(
            self, MockSerial, MockPoller, conn_mgr, app_config):
        """live_poll_during_test: a pause request must not reach the poller.

        Subsystems (scan, health) still call set_poller_active(False) on
        start — the config decides whether that request is honoured, so
        the live panel keeps reading through a measurement.
        """
        app_config.live_poll_during_test = True
        mock_poller = MockPoller.return_value
        mock_poller.isRunning.return_value = False
        conn_mgr.connect_port("COM3")

        conn_mgr.set_poller_active(False)
        assert not any(call.args == (False,)
                       for call in mock_poller.set_active.call_args_list), \
            "pause reached the poller despite live_poll_during_test"

        # Resume must still delegate — a paused poller (port reconnect,
        # config off) has to be resumable.
        conn_mgr.set_poller_active(True)
        mock_poller.set_active.assert_called_with(True)

    @patch("app.connection_manager.ParamPoller")
    @patch("app.connection_manager.LM19Serial")
    def test_poller_interval_comes_from_config(
            self, MockSerial, MockPoller, conn_mgr, app_config):
        """Call-site pin: the interval is read from config, not hardcoded."""
        app_config.live_poll_ms = 250
        MockPoller.return_value.isRunning.return_value = False
        conn_mgr.connect_port("COM3")
        assert MockPoller.call_args.kwargs["interval_ms"] == 250


class TestShutdown:
    @patch("app.connection_manager.ParamPoller")
    @patch("app.connection_manager.LM19Serial")
    def test_shutdown_stops_workers_and_closes(self, MockSerial, MockPoller, conn_mgr):
        mock_client = MockSerial.return_value
        mock_client.is_open.return_value = True
        mock_poller = MockPoller.return_value
        mock_poller.isRunning.return_value = True

        conn_mgr.connect_port("COM3")
        conn_mgr.shutdown()

        # shutdown() now drains via cleanup() (stop+wait+disconnect) and keeps
        # the reference only if the thread does not drain — see
        # TestConnectionManagerZombieRetention. cleanup() returning truthy here
        # (MagicMock) means the worker drained, so the reference is dropped.
        mock_poller.cleanup.assert_called_once_with(timeout_ms=1500)
        mock_client.close.assert_called()
        assert conn_mgr.client is None

    def test_shutdown_when_not_connected(self, conn_mgr):
        # Should not raise
        conn_mgr.shutdown()


class TestCheckCom:
    @patch("app.connection_manager.CheckComWorker")
    @patch("app.connection_manager.ParamPoller")
    @patch("app.connection_manager.LM19Serial")
    def test_check_com_emits_ok(self, MockSerial, MockPoller, MockCheckWorker, conn_mgr):
        mock_client = MockSerial.return_value
        mock_client.is_open.return_value = True
        MockPoller.return_value.isRunning.return_value = False
        MockCheckWorker.return_value.isRunning.return_value = False

        conn_mgr.connect_port("COM3")

        results = []
        conn_mgr.check_ok.connect(lambda v: results.append(v))
        conn_mgr.check_com()

        MockCheckWorker.assert_called_once_with(mock_client)
        MockCheckWorker.return_value.start.assert_called_once()

    def test_check_com_when_not_connected(self, conn_mgr):
        # Should not raise
        conn_mgr.check_com()
