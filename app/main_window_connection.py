"""Connection / live-params mixin for MainWindow.

Owns connection lifecycle, COM check, IO stats, live-params handling
and the emergency-reset flow.

Method groups:
  - port lifecycle: ``_refresh_ports``, ``_toggle_connection``,
    ``_on_connected``, ``_on_disconnected``, ``_apply_trace_state``,
    ``_toggle_trace``
  - COM check: ``_check_com``, ``_on_check_com_ok``, ``_on_check_com_fail``
  - IO stats / live: ``_on_io_stats``, ``_update_live_params``,
    ``_on_live_params_error``, ``_update_live_from_point``,
    ``_sync_pa_limits``
  - emergency stop: ``_on_reset_all_clicked``

Helpers _set_connection_led / _set_io_activity / _flash_io_activity /
_set_write_controls_locked stay in MainWindowBuilders mixin (created
during top-status-bar construction so they sit in the same module as
their widgets — splitting them into this mixin would be net negative).

Host class (MainWindow) provides via __init__:
  - self.conn_mgr (ConnectionManager), self.scan_ctrl (ScanController),
    self.live_panel (LivePanel), self.srk (SrkController),
    self.health_tab (HealthTab), self.manual_tab (ManualTab)
  - self.client (property → conn_mgr.client)
  - self.app_config, self.calibration
  - widget refs: port_combo, connect_btn, status_label, tx_count_label,
    rx_count_label, tx_activity_label, rx_activity_label, conn_led,
    run_btn, stop_btn,
    preheat_start_btn, preheat_stop_btn, preheat_status, pa_max_input,
    pa_over_pct, pig2_over_pct, trace_checkbox
  - state: _emergency_lock, _is_triode, _pig2_max_val, _prev_tx, _prev_rx
  - helpers (provided by other mixins or host):
    _set_connection_led, _set_io_activity, _flash_io_activity,
    _set_write_controls_locked, _reset_outputs_after_scan,
    _start_preheat, _stop_preheat
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List

from PySide6.QtWidgets import QMessageBox
from serial import SerialException
from serial.tools import list_ports

from app.main_window_builders import _IO_COUNT_ZERO
from app.ui_theme import COLOR_GREEN, COLOR_ORANGE, COLOR_RED
from lm19.protocol import encode_ug1
from i18n_setup import t

log = logging.getLogger(__name__)


# ── module local constants ──
# Idle age (s) after which the RX counter tooltip reports how long ago the
# last byte arrived: a dead link and an idle one look identical on the bare
# counter, so the age is the only cue that traffic stopped coming back.
_RX_STALE_S = 5


class MainWindowConnection:
    """Mixin: COM/connection lifecycle + live-params updates."""

    def _refresh_ports(self) -> None:
        self.port_combo.clear()
        for port in list_ports.comports():
            self.port_combo.addItem(port.device)
        if self.app_config.default_com_port:
            index = self.port_combo.findText(self.app_config.default_com_port)
            if index >= 0:
                self.port_combo.setCurrentIndex(index)

    def _toggle_connection(self) -> None:
        if self.conn_mgr.is_connected:
            # Refuse to close the port under a live worker — disconnect stops
            # polling and frees the client, leaving the outputs energized with
            # no way to zero them. The ownership arbiter already tracks every
            # write subsystem; it just was not wired into this path.
            busy = self._hw_busy_reason()
            if busy:
                log.warning("Disconnect blocked — hardware busy: %s", busy)
                QMessageBox.warning(self, t('msg.COM'), t('msg.Hw_busy'))
                return
            self.conn_mgr.disconnect()
            return
        port = self.port_combo.currentText()
        if not port:
            QMessageBox.warning(self, t('msg.COM'), t('msg.Select_COM_port'))
            return
        try:
            self.conn_mgr.connect_port(port)
            self._apply_trace_state()
        except Exception as exc:
            log.exception("COM port connection failed")
            QMessageBox.critical(self, t('msg.COM'), str(exc))

    def _on_connected(self, port: str) -> None:
        """Handle successful connection."""
        self.status_label.setText(t('conn.Connected', port=port))
        self._set_connection_led(COLOR_GREEN)
        self.connect_btn.setText(t('conn.Disconnect'))
        self._emergency_lock = False
        self._set_write_controls_locked(False)

    def _on_disconnected(self) -> None:
        """Handle disconnection."""
        self.status_label.setText(t('conn.Disconnected'))
        self.tx_count_label.setText(_IO_COUNT_ZERO)
        self.rx_count_label.setText(_IO_COUNT_ZERO)
        self.rx_count_label.setToolTip(t('conn.Rx_bytes'))
        self._set_connection_led(COLOR_RED)
        self._set_io_activity(self.tx_activity_label, active=False)
        self._set_io_activity(self.rx_activity_label, active=False)
        self.connect_btn.setText(t('conn.Connect'))
        self._emergency_lock = False
        self._set_write_controls_locked(False)

    def _check_com(self) -> None:
        if not self.conn_mgr.is_connected:
            QMessageBox.warning(self, t('msg.COM'), t('msg.Connect_first'))
            return
        self.conn_mgr.check_com()

    def _on_check_com_ok(self, value: int) -> None:
        self.status_label.setText(t('conn.OK_Ua', value=value))
        self._set_connection_led(COLOR_GREEN)

    def _on_check_com_fail(self, message: str) -> None:
        self.status_label.setText(t('conn.Invalid', message=message))
        self._set_connection_led(COLOR_ORANGE)

    def _on_io_stats(self, tx: int, rx: int, last_rx) -> None:
        """Handle IO stats from ConnectionManager."""
        self._flash_io_activity(
            self._prev_tx is not None and tx > self._prev_tx,
            self._prev_rx is not None and rx > self._prev_rx,
        )
        self._prev_tx = tx
        self._prev_rx = rx
        self.tx_count_label.setText(str(tx))
        self.rx_count_label.setText(str(rx))
        delta = max(0, int(time.time() - last_rx)) if last_rx else 0
        if last_rx and delta >= _RX_STALE_S:
            self.rx_count_label.setToolTip(
                t('conn.Rx_bytes_stale', delta=delta))
        else:
            self.rx_count_label.setToolTip(t('conn.Rx_bytes'))

    def _update_live_params(self, data: Dict) -> None:
        self._sync_pa_limits()
        self.live_panel.update_values(data, self.calibration)
        # Update Manual tab live readings and Ia chart
        if hasattr(self, 'manual_tab'):
            self.manual_tab.update_live_params(data)
        if hasattr(self, "health_tab"):
            self.health_tab.update_live_params(data)

    def _on_live_params_error(self, message: str) -> None:
        self.status_label.setText(t('conn.COM_error', message=message))
        self._set_connection_led(COLOR_ORANGE)

    def _emergency_zero_outputs(self, reset_heater: bool = True) -> List[str]:
        """Drive outputs to the safe state *immediately* and synchronously.

        Emergency stop must not wait for workers to wind down before zeroing.
        Outputs are written here on the UI thread; each ``set_param`` takes the
        serial lock, so they are serialised against any in-flight worker
        command. Ug1 goes to the safe negative bias (cuts the tube off) — NOT
        0, which would open the tube fully. Workers are flagged to stop before
        this runs, so they issue no further setpoints (they check stop before
        each write); the scan finish handler re-confirms the reset.

        ``reset_heater=False`` keeps Uh/Ih at their last setpoints — used on
        window close, where killing the HV is the safety goal but cycling the
        heater would needlessly stress the tube (a warm heater is harmless).

        Returns the channels that FAILED to zero — the tube may still be
        under voltage on those. A caller with a live UI must show them to
        the operator (ML-124/125 mirror for the synchronous path).
        """
        client = self.client
        if not client or not client.is_open():
            return []
        ug1_after_stop = self.app_config.ug1_after_stop
        # Safe order: screen off, grid to cutoff bias, anode off, then heater.
        steps = [
            ("Ug2", 0),
            ("Ug1", encode_ug1(ug1_after_stop)),
            ("Ua", 0),
        ]
        if reset_heater:
            steps += [("Uh", 0), ("Ih", 0)]
        failed: List[str] = []
        for name, value in steps:
            try:
                client.set_param(name, value)
            except (OSError, ValueError, RuntimeError, SerialException) as exc:
                log.warning("Emergency zero of %s failed: %s", name, exc)
                failed.append(name)
        return failed

    def _on_reset_all_clicked(self) -> None:
        if not self.client or not self.client.is_open():
            QMessageBox.warning(self, t('msg.COM'), t('msg.Connect_first'))
            return
        self._emergency_lock = True
        self._set_write_controls_locked(True)
        self.status_label.setText(t('conn.Emergency_stop'))
        self._set_connection_led(COLOR_RED)

        # Flag every write/measurement worker to stop — NO blocking wait.
        # Emergency must act now; workers check their stop flag before each
        # setpoint and wind down on their own. stop_scan() also sets
        # reset_on_finish so the scan finish handler re-confirms the reset.
        self.scan_ctrl.stop_scan()
        self.scan_ctrl.stop_preheat()
        if self.srk.worker and self.srk.worker.isRunning():
            self.srk.worker.stop()
        if (hasattr(self, "health_tab") and self.health_tab.worker
                and self.health_tab.worker.isRunning()):
            self.health_tab.worker.stop()

        # Zero outputs immediately and synchronously — the definitive action.
        failed = self._emergency_zero_outputs()
        if failed:
            # The operator MUST know the stop did not complete: the tube may
            # still be under voltage on these channels (failure-visibility rule).
            QMessageBox.critical(
                self, t('msg.Emergency_stop'),
                t('msg.Emergency_zero_failed', channels=", ".join(failed)),
            )

        self.scan_in_progress = False
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.preheat_start_btn.setEnabled(False)
        self.preheat_stop_btn.setEnabled(False)
        self.preheat_status.setText(t('heat.Stopped'))

        # Live must continue after emergency stop.
        self.conn_mgr.set_poller_active(True)

    def _apply_trace_state(self) -> None:
        if not self.conn_mgr.is_connected:
            return
        from pathlib import Path

        trace_path = Path(__file__).resolve().parents[1] / "logs" / "com_trace.log"
        self.conn_mgr.apply_trace(self.trace_checkbox.isChecked(), trace_path)

    def _toggle_trace(self) -> None:
        self._apply_trace_state()

    def _update_live_from_point(self, point: Dict) -> None:
        self._sync_pa_limits()
        self.live_panel.update_from_point(point)

    def _sync_pa_limits(self) -> None:
        """Push current Pa / Pg2 limits to both live panels."""
        pa_max = self.pa_max_input.value()
        pa_over = self.pa_over_pct.value()
        pg2_max = self._pig2_max_val if self._pig2_max_val else 0.0
        pg2_over = self.pig2_over_pct.value()
        self.live_panel.set_pa_limits(pa_max, pa_over)
        self.live_panel.set_pg2_limits(pg2_max, pg2_over)
        if hasattr(self, 'manual_tab'):
            self.manual_tab.live_panel.set_pa_limits(pa_max, pa_over)
            self.manual_tab.live_panel.set_pg2_limits(pg2_max, pg2_over)
