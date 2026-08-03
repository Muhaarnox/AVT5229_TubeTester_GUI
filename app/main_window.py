import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

import pyqtgraph as pg
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QMainWindow,
    QMessageBox,
)
# QFileDialog & list_ports kept exposed on this module's namespace because
# tests monkeypatch ``app.main_window.QFileDialog.getSaveFileName`` and
# ``app.main_window.list_ports.comports``. Runtime code reaches them
# through the mixin modules; this re-export keeps the test monkeypatch
# surface stable.
from PySide6.QtWidgets import QFileDialog  # noqa: F401  (test monkeypatch)
from serial.tools import list_ports  # noqa: F401  (test monkeypatch)

from lm19.app_config import load_app_config, load_calibration
from lm19.config import LampConfig, find_lamp, load_lamps, load_device_limits
from lm19.constants import DEFAULT_UG2_V
from lm19.protocol import (
    LM19Serial,
    encode_ih,
    encode_uh,
    setup_param_debug,
)
from app.amp_control_panel import AmpControlPanel
from app.compare_tab import CompareTab
from app.connection_manager import ConnectionManager
from app.export_manager import export_csv, export_pdf, export_spice, export_utd
from app.main_window_builders import MainWindowBuilders
from app.main_window_connection import MainWindowConnection
from app.main_window_scan import MainWindowScan
from app.main_window_settings import MainWindowSettings
from app.scan_controller import ScanController
from app.srk_widget import SrkController
from app.ui_theme import SERIES_PALETTE
from i18n_setup import t
from lm19.amplifier.constants import (
    CIRCUIT_PP,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)
from lm19.version import APP_VERSION
from lm19.tube_model_base import (
    MODEL_TYPE_DEMPWOLF,
    MODEL_TYPE_KOREN,
    MODEL_TYPE_REEFMAN,
)


class MainWindow(MainWindowBuilders, MainWindowConnection,
                 MainWindowSettings, MainWindowScan, QMainWindow):
    # UI layout constants
    LAMP_COMBO_MIN_W = 240
    LAMP_COMBO_MAX_W = 500
    UG2_DISPLAY_MIN_W = 100
    UG2_DISPLAY_MAX_W = 200
    UG2_CALC_MIN_W = 80
    UG2_CALC_MAX_W = 150
    SECTION_SPACING = 15
    CONTROL_SPACING = 10
    TIGHT_SPACING = 4
    NARROW_SPIN_W = 60
    MEDIUM_SPIN_W = 80
    WIDE_SPIN_W = 90
    CMAP_COMBO_W = 80
    # How far each side of the Measure splitter may be squeezed: below
    # these the controls stop compressing and the handle stops.
    LEFT_PANEL_MIN_W = 320
    RIGHT_PANEL_MIN_W = 400

    # --- Compatibility properties for code that accesses self.client / self.param_poller ---
    @property
    def client(self) -> Optional[LM19Serial]:
        return self.conn_mgr.client if hasattr(self, "conn_mgr") else None

    @property
    def param_poller(self):
        return self.conn_mgr._param_poller if hasattr(self, "conn_mgr") else None

    @property
    def scan_worker(self):
        return self.scan_ctrl.scan_worker if hasattr(self, "scan_ctrl") else None

    @property
    def preheat_worker(self):
        return self.scan_ctrl.preheat_worker if hasattr(self, "scan_ctrl") else None

    @property
    def reset_worker(self):
        return self.scan_ctrl.reset_worker if hasattr(self, "scan_ctrl") else None

    @property
    def scan_in_progress(self) -> bool:
        return self.scan_ctrl.scan_in_progress if hasattr(self, "scan_ctrl") else False

    @scan_in_progress.setter
    def scan_in_progress(self, value: bool) -> None:
        if hasattr(self, "scan_ctrl"):
            self.scan_ctrl.scan_in_progress = value

    @property
    def reset_on_finish(self) -> bool:
        return self.scan_ctrl.reset_on_finish if hasattr(self, "scan_ctrl") else False

    @reset_on_finish.setter
    def reset_on_finish(self, value: bool) -> None:
        if hasattr(self, "scan_ctrl"):
            self.scan_ctrl.reset_on_finish = value

    @property
    def preheat_done(self) -> bool:
        return self.scan_ctrl.preheat_done if hasattr(self, "scan_ctrl") else False

    @preheat_done.setter
    def preheat_done(self, value: bool) -> None:
        if hasattr(self, "scan_ctrl"):
            self.scan_ctrl.preheat_done = value

    @property
    def scan_total_points(self) -> int:
        return self.scan_ctrl.scan_total_points if hasattr(self, "scan_ctrl") else 0

    @scan_total_points.setter
    def scan_total_points(self, value: int) -> None:
        if hasattr(self, "scan_ctrl"):
            self.scan_ctrl.scan_total_points = value

    def __init__(self) -> None:
        super().__init__()
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        # Default PlotItem margins (1,1,1,1) leave no gap at the right
        # edge when right axis is hidden → plot looks clipped.
        PLOT_RIGHT_MARGIN = 4
        _orig_pi_init = pg.PlotItem.__init__

        def _patched_pi_init(self_pi, *args, **kwargs):
            _orig_pi_init(self_pi, *args, **kwargs)
            l, t_m, r, b = self_pi.layout.getContentsMargins()
            self_pi.layout.setContentsMargins(l, t_m, max(r, PLOT_RIGHT_MARGIN), b)

        pg.PlotItem.__init__ = _patched_pi_init
        self.setWindowTitle(t('app.Window_title', version=APP_VERSION))
        self.resize(1200, 800)

        self.lamps: List[LampConfig] = []
        self._emergency_lock = False
        self._prev_tx: Optional[int] = None
        self._prev_rx: Optional[int] = None
        self.srk = SrkController(self)
        self.srk.label_changed.connect(self._on_srk_label_changed)
        self.srk.measure_btn_enabled.connect(self._on_srk_measure_btn_enabled)
        self.srk.show_points_btn_enabled.connect(self._on_srk_show_points_enabled)
        self.srk.reset_requested.connect(self._reset_outputs_after_scan)
        self.srk.measurement_ready.connect(self._save_and_display_measurement)
        self.app_config = load_app_config()
        self.calibration = load_calibration()
        # Setup parameter debug logging from config
        if self.app_config.debug_params:
            from pathlib import Path
            log_path = Path(__file__).resolve().parents[1] / self.app_config.debug_params_file
            setup_param_debug(True, str(log_path))
        self._is_triode = False

        # Connection manager — owns serial client and poller
        self.conn_mgr = ConnectionManager(self.app_config, parent=self)
        self.conn_mgr.connected.connect(self._on_connected)
        self.conn_mgr.disconnected.connect(self._on_disconnected)
        self.conn_mgr.live_params.connect(self._update_live_params)
        self.conn_mgr.live_error.connect(self._on_live_params_error)
        self.conn_mgr.check_ok.connect(self._on_check_com_ok)
        self.conn_mgr.check_fail.connect(self._on_check_com_fail)
        self.conn_mgr.io_stats_updated.connect(self._on_io_stats)
        self.srk.poller_active.connect(self.conn_mgr.set_poller_active)

        # Scan controller — owns scan, preheat, reset workers
        self.scan_ctrl = ScanController(self.app_config, parent=self)
        self.scan_ctrl.scan_started.connect(self._on_scan_started)
        self.scan_ctrl.scan_progress.connect(self._on_point)
        self.scan_ctrl.scan_finished.connect(self._on_scan_finished)
        self.scan_ctrl.scan_failed.connect(self._on_scan_failed)
        self.scan_ctrl.scan_comm_error.connect(self._on_scan_comm_error)
        self.scan_ctrl.preheat_started.connect(self._on_preheat_started)
        self.scan_ctrl.preheat_progress.connect(self._on_preheat_progress)
        self.scan_ctrl.preheat_finished.connect(self._on_preheat_finished)
        self.scan_ctrl.preheat_failed.connect(self._on_preheat_failed)
        self.scan_ctrl.reset_finished.connect(self._on_reset_finished)
        self.scan_ctrl.reset_failed.connect(self._on_reset_failed)

        self._measure_splitter_fitted = False
        self._build_ui()
        # Working line live layer — after the UI is built: needs
        # self.plot / plot_mgr / the panel.
        self._wire_working_line()

        from app.import_controller import ImportController
        self.import_ctrl = ImportController(
            parent_widget=self,
            compare_tab=self.compare_tab,
            tabs=self.tabs,
            get_lamps=lambda: self.lamps,
        )

        self._build_menu()
        self._load_lamps()
        self._refresh_ports()


    def _load_lamps(self) -> None:
        try:
            self.lamps = load_lamps()
        except Exception as exc:
            log.exception("Failed to load lamp config")
            QMessageBox.warning(self, t('msg.Config'), str(exc))
            self.lamps = []

        self._device_limits = load_device_limits()
        self.lamp_panel.set_lamps(self.lamps)

        if self.lamps:
            self._apply_lamp(self.lamps[0].tube_type)
            self.ia_max_input.setValue(self.app_config.plot_ia_max)
            if hasattr(self, "health_tab"):
                self.health_tab.refresh_lamps()
        else:
            self.status_label.setText(t('conn.No_lamp_data'))
            self.compare_tab.clear()

    def _update_ia_axis(self, value: float) -> None:
        self.plot_renderer.apply_ia_axis(value)

    def _auto_ia_axis(self) -> None:
        if not self.plot_mgr.points:
            return
        max_ia = max((p.get("ia", 0.0) for p in self.plot_mgr.points), default=0.0)
        max_ia = max(10.0, max_ia * 1.1)
        self.ia_max_input.setValue(max_ia)

    def _auto_ua_axis(self) -> None:
        if not self.plot_mgr.points:
            return
        ua_vals = [p.get("ua", 0.0) for p in self.plot_mgr.points if "ua" in p]
        if not ua_vals:
            return
        ua_min = min(ua_vals)
        ua_max = max(ua_vals)
        span = ua_max - ua_min
        pad = max(5.0, span * 0.05)
        self.plot_renderer.apply_ua_axis(ua_min - pad, ua_max + pad)

    def _auto_ug1_step(self, start: float, stop: float) -> float:
        span = abs(stop - start)
        if span <= 0:
            return 0.5
        target_count = 9
        raw_step = span / target_count
        if raw_step <= 0:
            return 0.5
        magnitude = 10 ** int(math.floor(math.log10(raw_step)))
        bases = [0.5, 1.0, 2.0]
        candidates = []
        for scale in [magnitude / 10, magnitude, magnitude * 10]:
            for base in bases:
                candidates.append(base * scale)
        best = candidates[0]
        best_score = float("inf")
        for step in candidates:
            count = span / step
            score = abs(count - target_count)
            if score < best_score:
                best_score = score
                best = step
        return max(0.5, round(best, 2))



    def _start_preheat(self) -> None:
        if self._emergency_lock:
            QMessageBox.warning(self, t('msg.Reset_all_title'), t('msg.Emergency_write_block'))
            return
        if not self.client or not self.client.is_open():
            QMessageBox.warning(self, t('msg.COM'), t('msg.Connect_first'))
            return
        if not self.preheat_enabled.isChecked():
            return
        tube_type = self.lamp_combo.currentText()
        lamp = find_lamp(self.lamps, tube_type)
        if not lamp:
            return
        # Preheat writes Uh/Ih — it is a hardware-owning start path like scan /
        # SRK / health and must respect the ownership arbiter. Without this a
        # user could start preheat during a Health/SRK/scan run, driving the
        # heater concurrently with another subsystem's measurements.
        busy = self._hw_busy_reason(exclude="preheat")
        if busy:
            log.warning("Preheat blocked — hardware busy: %s", busy)
            QMessageBox.warning(self, t('msg.Reset_all_title'), t('msg.Hw_busy'))
            return
        target_uh = self.uh_input.value()
        target_ih = self.ih_input.value() if target_uh <= 0 else 0.0
        warmup_s = int(self.preheat_seconds.value())
        self.scan_ctrl.start_preheat(self.client, target_uh, target_ih, warmup_s,
                                     calibration=self.calibration)

    def _on_preheat_started(self) -> None:
        self.preheat_start_btn.setEnabled(False)
        self.preheat_status.setText(t('heat.Preheating'))
        self._update_preheat_live_label(None, None)

    def _stop_preheat(self) -> None:
        self.scan_ctrl.stop_preheat()
        self.preheat_start_btn.setEnabled(True)
        self.preheat_status.setText(t('heat.Stopped'))
        self._reset_outputs_after_scan(reset_heater=True, reset_order=["Ug2", "Ug1", "Ua", "Uh", "Ih"])
        self.conn_mgr.set_poller_active(True)

    def _on_preheat_progress(self, uh: float, ih: float, remaining: int) -> None:
        # Ignore stale progress delivered after the preheat worker was drained
        # on Stop (reset_outputs clears _preheat_worker) — otherwise a queued
        # event would flip the status back to "Warmup remaining" after we
        # already showed "Stopped".
        if self.scan_ctrl.preheat_worker is None:
            return
        total = max(1, int(self.preheat_seconds.value()))
        elapsed = total - remaining
        percent = int(min(100, (elapsed / total) * 100))
        self.preheat_progress.setValue(percent)
        self.preheat_live.setText(t('heat.Uh_Ih_live', uh=f"{uh:.2f}", ih=f"{ih:.3f}"))
        if remaining > 0:
            self.preheat_status.setText(t('heat.Warmup_remaining', remaining=remaining))
        self._update_preheat_live_label(uh, ih)

    def _on_preheat_finished(self) -> None:
        self.preheat_progress.setValue(100)
        self.preheat_status.setText(t('heat.Warmup_ready'))
        self.preheat_start_btn.setEnabled(True)
        self.preheat_done = True
        self._update_preheat_live_label(None, None)
        self.conn_mgr.set_poller_active(True)

    def _on_preheat_failed(self, message: str) -> None:
        self.preheat_status.setText(t('heat.Preheat_error'))
        self.preheat_start_btn.setEnabled(True)
        self.conn_mgr.set_poller_active(True)
        QMessageBox.critical(self, t('msg.Preheat'), message)
        self._update_preheat_live_label(None, None)

    def _hw_busy_reason(self, exclude: str = "") -> Optional[str]:
        """Return a short token of the subsystem currently driving the device
        (``"scan"``/``"preheat"``/``"SRK"``/``"health"``) or ``None`` if free.

        ``exclude`` skips the caller's own subsystem so a start-path is blocked
        only by *other* owners. Single arbiter for hardware ownership: the
        firmware serialises individual commands but not logical sequences, so
        two subsystems writing at once corrupt each other's measurements and
        protection checks.
        """
        if exclude != "scan" and self.scan_ctrl.is_scanning:
            return "scan"
        pw = self.scan_ctrl.preheat_worker
        if exclude != "preheat" and pw is not None and pw.isRunning():
            return "preheat"
        rw = self.scan_ctrl.reset_worker
        if exclude != "reset" and rw is not None and rw.isRunning():
            return "reset"
        if exclude != "SRK" and self.srk.worker is not None and self.srk.worker.isRunning():
            return "SRK"
        ht = getattr(self, "health_tab", None)
        hw = getattr(ht, "worker", None) if ht is not None else None
        if exclude != "health" and hw is not None and hw.isRunning():
            return "health"
        return None

    def _can_send_heater(self) -> bool:
        if self._emergency_lock:
            return False
        if not self.preheat_enabled.isChecked():
            return False
        if not self.client or not self.client.is_open():
            return False
        if self.scan_worker and self.scan_worker.isRunning():
            return False
        if self.preheat_worker and self.preheat_worker.isRunning():
            return False
        if self.reset_worker and self.reset_worker.isRunning():
            # A live post-scan reset is zeroing the outputs — a heater
            # command now would race it (ML-122).
            return False
        return True

    def _on_uh_changed(self, value: float) -> None:
        if not self._can_send_heater():
            return
        # Working-point command → SET feedforward (plan B). Zero means
        # "heater off" — raw literal: apply_set(0) = offset would command
        # a non-zero voltage. The An selector below stays raw too.
        if value <= 0.0:
            self.client.set_param("Uh", 0)
            return
        self.client.set_param(
            "Uh", encode_uh(self.calibration.apply_set("uh", value)))

    def _on_ih_changed(self, value: float) -> None:
        if not self._can_send_heater():
            return
        if value <= 0.0:
            self.client.set_param("Ih", 0)
            return
        self.client.set_param(
            "Ih", encode_ih(self.calibration.apply_set("ih", value)))

    def _on_anode_changed(self, an_id: int) -> None:
        if self.lamp_panel.anode() != an_id:
            (self.lamp_panel.anode_2 if an_id == 2 else self.lamp_panel.anode_1).setChecked(True)
        self.live_panel.set_an(an_id)
        if self._emergency_lock:
            return
        if self.client and self.client.is_open():
            if not (self.scan_worker and self.scan_worker.isRunning()):
                self.client.set_param("An", an_id)

    def _reset_outputs_after_scan(
        self, reset_heater: bool = True, reset_order: Optional[List[str]] = None
    ) -> None:
        dropped = self.scan_ctrl.reset_outputs(
            self.client, reset_heater, reset_order)
        if dropped == "port_closed":
            # ML-090: no way to zero the outputs — the operator must know
            # the tube keeps its last setpoints (failure-visibility rule).
            QMessageBox.critical(
                self, t('msg.Reset_all_title'),
                t('msg.Reset_dropped_port_closed'))
        elif dropped == "reset_busy":
            # Another reset is already zeroing the outputs — non-critical,
            # but the differing request (e.g. heater flag) was ignored.
            self.status_label.setText(t('conn.Reset_busy'))

    def _on_reset_finished(self) -> None:
        self.conn_mgr.update_io_stats()

    def _on_reset_failed(self, message: str) -> None:
        self.status_label.setText(t('conn.Reset_error', message=message))

    def _update_preheat_live_label(self, uh: Optional[float], ih: Optional[float]) -> None:
        if uh is None:
            text = self.preheat_live.text().split("[")[0].strip()
        else:
            text = f"Uh: {uh:.2f} V  Ih: {ih:.3f} A"
        self.preheat_live.setText(f"{text}  [{self.preheat_status.text()}]")

    def _on_compare_show_main_plot(self, selected_points: list,
                                    series_labels: dict, series_colors: dict,
                                    series_ug2_track: dict = None,
                                    scan_meta: dict = None) -> None:
        """Handle 'Show on main plot' signal from CompareTab.

        The selection replaces every overlay series while the current scan
        (sid 0) and its bookkeeping stay (``replace_overlay_series``).
        ``scan_meta`` (single shown measurement) feeds the PDF report's
        scan-settings section — a loaded measurement must not tell the
        user to "run a scan first".
        """
        if scan_meta:
            self._last_scan_meta = scan_meta
        pm = self.plot_mgr
        pm.replace_overlay_series(selected_points, series_labels,
                                  series_colors, series_ug2_track)
        if hasattr(self, "working_line"):
            self.working_line.invalidate()
        pm.refresh_lamp_combos()
        pm.refresh_ug2_combos(pm.points)
        pm.invalidate_cache()
        pm.render_all()
        self.tabs.setCurrentIndex(0)

    def _on_amp_update(self) -> None:
        """AmpControlPanel settings changed → run engine → render."""
        if not hasattr(self, "amp_engine") or not self.amp_engine.has_data:
            self.amp_control_panel.show_results(t("amp.no_data"))
            return

        params = self.amp_control_panel.params_snapshot()
        # Pull Ug2 filter from main UI if pentode
        ug2_filter = self._get_amp_ug2_filter()
        if ug2_filter is not None:
            params.ug2_filter = ug2_filter
        self.amp_control_panel.set_ug2_display(ug2_filter)
        # Provide lamp config Ug2 as a fallback "suggested" if engine
        # needs to ask the user for one. This is the per-tube nominal
        # screen voltage from lamps.json — much closer to reality than
        # the global DEFAULT_UG2_V (e.g. 425V for KT88 vs 250V default).
        lamp = find_lamp(self.lamps, self.lamp_combo.currentText())
        if lamp is not None and not lamp.is_triode and lamp.ug2 > 0:
            params.lamp_ug2_default = lamp.ug2

        result = self.amp_engine.analyze(params)

        # Engine asks UI to obtain explicit Ug2 (pentode model but no
        # valid Ug2 in measurements). Show input dialog with lamp's
        # nominal Ug2 pre-filled; on confirm, retry analysis.
        if result.error == "needs_ug2":
            self._handle_needs_ug2_dialog(result, params)
            return

        self.working_line.apply_full_result(result.working_line)
        self._last_amp_result = result  # PDF amp-report source
        self._mark_amp_verify_stale_if_needed(result.params)
        html = self.amplifier_tab.render(result)
        self.amp_control_panel.show_results(html)
        self._set_ui_warnings(
            "analysis", self.amplifier_tab.collect_warnings(result))

    def _handle_needs_ug2_dialog(self, result, params) -> None:
        """Prompt the user to specify Ug2 for pentode analysis.

        Engine raised needs_ug2 because a pentode model was selected but
        the measurement points don't carry a valid screen voltage (sensor
        failure, mislabelled triode-mode export, etc.). We pre-fill with
        the lamp's nominal Ug2 from lamps.json, accept the user's value,
        then re-run the analysis with ``params.ug2_filter`` set.
        """
        suggested = result.suggested_ug2 or DEFAULT_UG2_V
        from PySide6.QtWidgets import QInputDialog
        value, ok = QInputDialog.getDouble(
            self,
            t("amp.Needs_Ug2_title"),
            t("amp.Needs_Ug2_prompt", suggested=f"{suggested:.0f}"),
            float(suggested),
            10.0,    # min: physically reasonable lower bound for pentode screen
            2000.0,  # max: well above typical maximum (KT88 max ~600V)
            0,       # decimals: integer volts
        )
        if not ok:
            self.amp_control_panel.show_results(t("amp.Needs_Ug2_cancelled"))
            return
        params.ug2_filter = float(value)
        self.amp_control_panel.set_ug2_display(value)
        result = self.amp_engine.analyze(params)
        self.working_line.apply_full_result(result.working_line)
        self._last_amp_result = result  # PDF amp-report source
        self._mark_amp_verify_stale_if_needed(result.params)
        html = self.amplifier_tab.render(result)
        self.amp_control_panel.show_results(html)
        self._set_ui_warnings(
            "analysis", self.amplifier_tab.collect_warnings(result))

    def _get_amp_ug2_filter(self) -> 'Optional[float]':
        """Get Ug2 filter from ug2_calc_combo for pentode analysis."""
        combo = getattr(self, "ug2_calc_combo", None)
        if combo and combo.count() > 0:
            try:
                return float(combo.currentText())
            except (ValueError, TypeError):
                pass
        return None

    def _on_amp_auto_q(self) -> None:
        """Auto Q: sweep bias, find min THD, update Ug1 spinbox."""
        if not self.amp_engine.has_data:
            return
        from lm19 import amplifier

        params = self.amp_control_panel.params_snapshot()
        points = amplifier.select_analysis_points(
            self.amp_engine._all_points, series_id=params.series_id,
        )
        ll = self.amp_engine._make_load_line(params)

        # Use model if available for more accurate sweep
        model = None
        model_ug2 = 0.0
        if self.amp_engine._series_models and params.series_id is not None:
            model = self.amp_engine._series_models.get(params.series_id)
        ug2_filter = self._get_amp_ug2_filter()
        if ug2_filter is not None:
            model_ug2 = ug2_filter

        result = amplifier.optimize_bias(
            points, ll, ug2_filter=ug2_filter, target="min_thd",
            model=model, model_ug2=model_ug2,
        )
        if result:
            self.amp_control_panel.ug1_spin.setValue(result["ug1_0"])
            log.info("Auto Q: Ug1=%.1f V, THD=%.2f%%", result["ug1_0"], result["thd"])
            self._on_amp_update()

    def _on_amp_optimize_ra(self) -> None:
        """Optimize Ra: sweep Ra, find min THD, update Ra spinbox."""
        if not self.amp_engine.has_data:
            return
        from lm19 import amplifier

        params = self.amp_control_panel.params_snapshot()
        points = amplifier.select_analysis_points(
            self.amp_engine._all_points, series_id=params.series_id,
        )
        cfg = self.app_config
        ra_min_factor = cfg.amp_opt_ra_min_factor if cfg else 0.1
        ra_max_factor = cfg.amp_opt_ra_max_factor if cfg else 10.0
        ra_min_abs = cfg.amp_opt_ra_min_abs_kohm if cfg else 0.5
        ra_max_abs = cfg.amp_opt_ra_max_abs_kohm if cfg else 200.0
        ra_steps = cfg.amp_opt_ra_steps if cfg else 100

        # Use model if available for more accurate sweep
        model = None
        model_ug2 = 0.0
        if self.amp_engine._series_models and params.series_id is not None:
            model = self.amp_engine._series_models.get(params.series_id)
        ug2_filter = self._get_amp_ug2_filter()
        if ug2_filter is not None:
            model_ug2 = ug2_filter

        ra_data = amplifier.sweep_ra(
            points, params.ub,
            ra_min=max(ra_min_abs, params.ra * ra_min_factor),
            ra_max=min(ra_max_abs, params.ra * ra_max_factor),
            ug1_bias=params.ug1_bias,
            half_swing=params.half_swing,
            ug2_filter=ug2_filter,
            steps=ra_steps,
            model=model, model_ug2=model_ug2,
        )
        if ra_data:
            best = min(ra_data, key=lambda d: d["thd"])
            self.amp_control_panel.ra_spin.setValue(best["ra"])
            log.info("Optimize Ra: %.1f kΩ, THD=%.2f%%", best["ra"], best["thd"])
            self._on_amp_update()

    def _on_amp_ra_clicked(self, ra_val: float) -> None:
        """Click on Ra plot → set Ra in spinbox."""
        self.amp_control_panel.ra_spin.setValue(round(ra_val, 1))

    def _on_amp_optimizer_toggled(self, enabled: bool) -> None:
        """Pareto toggle → switch left plot page."""
        if enabled:
            self.amplifier_tab.show_pareto()
        else:
            self.amplifier_tab.clear_pareto()

    def _on_amp_optimize_full(self) -> None:
        """Run full multi-parameter optimization in background thread."""
        if not hasattr(self, "amp_engine") or not self.amp_engine.has_data:
            return
        # Prevent double-start
        if hasattr(self, "_opt_worker") and self._opt_worker is not None:
            if self._opt_worker.isRunning():
                return
        from lm19 import amplifier
        from app.optimize_worker import OptimizeWorker

        constraints = self.amp_control_panel.optimizer_constraints()
        params = self.amp_control_panel.params_snapshot()

        points = amplifier.select_analysis_points(
            self.amp_engine._all_points, series_id=params.series_id,
        )
        if not points:
            self.amp_control_panel.set_optimizer_status(t("amp.opt_no_data"))
            return

        # Choose model or measurements path
        model = None
        use_model_path = False
        if self.amp_engine._series_models and params.series_id is not None:
            model = self.amp_engine._series_models.get(params.series_id)

        ug2_filter = self._get_amp_ug2_filter()
        ug1_vals = sorted({round(p["ug1"], 1) for p in points})

        ug2_available = sorted({
            round(p.get("ug2", 0.0), 0)
            for p in points if "ug2" in p and p["ug2"] > 0
        }) or None

        if model is not None and constraints.ub_range is not None:
            use_model_path = True

        # PP: get tube B data if not matched
        points_b = None
        if constraints.circuit == CIRCUIT_PP and not params.pp_matched:
            if params.pp_tube_b_sid is not None:
                points_b = amplifier.select_analysis_points(
                    self.amp_engine._all_points, series_id=params.pp_tube_b_sid,
                )

        self._opt_worker = OptimizeWorker(
            points=points,
            constraints=constraints,
            ub=params.ub,
            model=model,
            ug2_filter=ug2_filter,
            ug2_values=ug2_available,
            ug1_values=ug1_vals,
            use_model_path=use_model_path,
            points_b=points_b,
            parent=self,
        )
        self._opt_worker.progress.connect(self._on_opt_progress)
        self._opt_worker.finished_ok.connect(self._on_opt_finished)
        self._opt_worker.finished_err.connect(self._on_opt_error)
        self._opt_worker.finished_cancelled.connect(self._on_opt_cancelled)

        self.amp_control_panel.set_optimizer_running(True)
        self._opt_worker.start()

    def _on_opt_cancel_clicked(self) -> None:
        """Stable Cancel slot, wired once at build time (ML-003).

        The old per-run ``clicked.connect(worker.cancel)`` stacked one
        connection per optimization run: finished workers stayed reachable
        from the button forever and every Cancel click fanned out to all
        of them.
        """
        worker = getattr(self, "_opt_worker", None)
        if worker is not None:
            worker.cancel()

    def _on_opt_progress(self, pct: int, phase_key: str) -> None:
        """Update progress bar from worker signal."""
        # Parse phase key — may contain "|current|total" for refine step
        if "|" in phase_key:
            parts = phase_key.split("|")
            phase_text = t(f"amp.{parts[0]}", current=parts[1], total=parts[2])
        else:
            phase_text = t(f"amp.{phase_key}")
        self.amp_control_panel.set_optimizer_progress(pct, phase_text)

    def _on_opt_finished(self, result: object) -> None:
        """Worker finished successfully — show results."""
        self.amp_control_panel.set_optimizer_running(False)
        # Disconnect worker signals + drop reference. Without cleanup()
        # the C++ worker object holds connections from this slot, which
        # accumulate over repeated optimizations.
        if self._opt_worker is not None:
            self._opt_worker.cleanup()
        self._opt_worker = None

        self.amplifier_tab.render_pareto(result)
        self.amp_control_panel.opt_pareto_btn.blockSignals(True)
        self.amp_control_panel.opt_pareto_btn.setChecked(True)
        self.amp_control_panel.opt_pareto_btn.blockSignals(False)

        # Status
        n_valid = sum(1 for p in result.grid_points if p.valid)
        best = result.refined or result.best
        n_pareto = len(result.refined_pareto) if result.refined_pareto else len(result.pareto_front)
        status = t(
            "amp.opt_done",
            n_total=len(result.grid_points),
            n_valid=n_valid,
            n_pareto=n_pareto,
        )
        if best:
            status += f"  THD={best.thd:.2f}%  Pout={best.pout_mw / 1000:.3f}W"
            status += f"\nUb={best.ub:.0f}V  Ra={best.ra:.1f}kΩ  Ug1={best.ug1:.1f}V"
            if best.half_swing > 0:
                status += f"  Swing={best.half_swing:.1f}V"
            if best.ug2 > 0:
                status += f"  Ug2={best.ug2:.0f}V"
            # Pa vs the constraint + amp class + UL tap: without these the
            # best point is not reproducible/judgeable from the status line
            # (a 43% UL optimum looked identical to a pentode one).
            status += (f"\nPa={best.pa_mw / 1000:.2f}W"
                       f"  Class {best.amp_class}")
            if getattr(best, "ul_tap", 0.0) > 0:
                status += f"  UL={best.ul_tap * 100:.0f}%"
        # Show actual HD method used (may differ between grid and refined)
        method_label = best.hd_method if best else "?"
        status += f"\n{t('amp.opt_method_used', method=method_label)}"
        # Surface non-fatal warnings (e.g. dft requested but no model
        # fitted, refine-phase degradations) — in the status text AND the
        # status-bar indicator (failure-visibility rule).
        warn_codes = []
        if getattr(result, "warning", None):
            warn_codes.append(result.warning)
        warn_codes.extend(getattr(result, "warnings", []))
        warn_texts = [t(f"amp.opt_warn_{code}") for code in warn_codes]
        for text in warn_texts:
            status += f"\n⚠ {text}"
        self._set_ui_warnings("optimizer", warn_texts)
        self.amp_control_panel.set_optimizer_status(status)

        # Store result for Apply / Top-N actions and enable buttons
        self._last_opt_result = result if best is not None else None
        self.amp_control_panel.set_optimizer_result_available(best is not None)

    def _on_opt_error(self, error: str) -> None:
        """Worker finished with error."""
        self.amp_control_panel.set_optimizer_running(False)
        if self._opt_worker is not None:
            self._opt_worker.cleanup()
        self._opt_worker = None
        # ④: known codes get a translated explanation with advice;
        # unknown text (worker exceptions) stays raw inside the generic
        # message — never hide the original cause.
        known = t(f"amp.opt_err_{error}")
        if known != f"amp.opt_err_{error}":
            self.amp_control_panel.set_optimizer_status(known)
        else:
            self.amp_control_panel.set_optimizer_status(
                t("amp.opt_error", error=error),
            )

    def _on_opt_cancelled(self) -> None:
        """Worker cancelled by the user — reset the UI out of the running state
        (neither finished_ok nor finished_err fires on cancel) and drop the
        worker. Without this the optimizer panel stays stuck showing Cancel +
        progress forever."""
        self.amp_control_panel.set_optimizer_running(False)
        if self._opt_worker is not None:
            self._opt_worker.cleanup()
        self._opt_worker = None

    def _on_amp_pareto_clicked(
        self, ub: float, ug2: float, ug1: float, ra: float, swing: float,
        ul_tap: float = 0.0,
    ) -> None:
        """Click on Pareto point → apply Ub, Ug2, Ug1, Ra, Swing, UL tap."""
        self._apply_opt_point_to_params(ub, ug2, ug1, ra, swing, ul_tap)
        applied = t("amp.opt_applied",
                    ub=f"{ub:.0f}", ra=f"{ra:.1f}",
                    ug1=f"{ug1:.1f}", swing=f"{swing:.1f}")
        # The tap IS applied (4a) — saying so was the missing half.
        if ul_tap > 0:
            applied += f"  UL={ul_tap * 100:.0f}%"
        self.amp_control_panel.append_optimizer_status(applied)

    def _apply_opt_point_to_params(
        self, ub: float, ug2: float, ug1: float, ra: float, swing: float,
        ul_tap: float = 0.0,
    ) -> None:
        """Apply optimizer point values to manual analysis spinboxes."""
        self.amp_control_panel.ub_spin.setValue(round(ub, 0))
        self.amp_control_panel.ug1_spin.setValue(round(ug1, 1))
        self.amp_control_panel.ra_spin.setValue(round(ra, 1))
        if swing > 0:
            self.amp_control_panel.swing_spin.setValue(round(swing, 1))
        # ML-029: the point's UL tap is part of the operating point — a
        # stale tap spin silently reproduces DIFFERENT physics (UL 43%
        # vs pentode: ~0.62× Pout, ~0.4× THD). Zero is applied too: an
        # optimum found at tap 0 must not inherit a non-zero spin.
        self.amp_control_panel.ul_tap_spin.setValue(round(ul_tap * 100.0, 1))
        # Set Ug2 in main UI combo if pentode
        if ug2 > 0:
            combo = getattr(self, "ug2_calc_combo", None)
            if combo:
                target = str(int(round(ug2)))
                idx = combo.findText(target)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def _on_amp_apply_best(self) -> None:
        """Apply best point from last optimizer result to manual params."""
        result = getattr(self, "_last_opt_result", None)
        if result is None:
            return
        best = result.refined or result.best
        if best is None:
            return
        self._apply_opt_point_to_params(
            best.ub, best.ug2, best.ug1, best.ra, best.half_swing,
        )
        self.amp_control_panel.append_optimizer_status(
            t("amp.opt_applied",
              ub=f"{best.ub:.0f}", ra=f"{best.ra:.1f}",
              ug1=f"{best.ug1:.1f}", swing=f"{best.half_swing:.1f}")
        )

    def _on_amp_show_top_n(self) -> None:
        """Open Top-N candidates dialog from last optimizer result."""
        result = getattr(self, "_last_opt_result", None)
        if result is None:
            return
        from app.optimizer_top_n_dialog import OptimizerTopNDialog
        # Prefer refined Pareto (more accurate via DFT) when available
        candidates = list(result.refined_pareto) if result.refined_pareto \
            else list(result.pareto_front)
        if not candidates:
            return
        dlg = OptimizerTopNDialog(candidates, parent=self)
        if dlg.exec() == dlg.DialogCode.Accepted and dlg.selected_point is not None:
            pt = dlg.selected_point
            self._apply_opt_point_to_params(
                pt.ub, pt.ug2, pt.ug1, pt.ra, pt.half_swing,
                getattr(pt, "ul_tap", 0.0) or 0.0,
            )
            applied = t("amp.opt_applied",
                        ub=f"{pt.ub:.0f}", ra=f"{pt.ra:.1f}",
                        ug1=f"{pt.ug1:.1f}", swing=f"{pt.half_swing:.1f}")
            if getattr(pt, "ul_tap", 0.0) > 0:
                applied += f"  UL={pt.ul_tap * 100:.0f}%"
            self.amp_control_panel.append_optimizer_status(applied)

    @staticmethod
    def _sync_spin_pair(spin_a: 'QDoubleSpinBox', spin_b: 'QDoubleSpinBox') -> None:
        """Wire two spinboxes for bidirectional sync.

        Uses a recursion guard instead of blockSignals so that
        downstream slots (e.g. _rerender_2d) still fire on the
        receiving spinbox.
        """
        syncing = [False]  # mutable guard shared by both closures

        def a_to_b(val: float) -> None:
            if syncing[0]:
                return
            syncing[0] = True
            spin_b.setValue(val)
            syncing[0] = False

        def b_to_a(val: float) -> None:
            if syncing[0]:
                return
            syncing[0] = True
            spin_a.setValue(val)
            syncing[0] = False

        spin_a.valueChanged.connect(a_to_b)
        spin_b.valueChanged.connect(b_to_a)

    def _measure_srk(self) -> None:
        """Start a manual SRK measurement via the controller."""
        if self._emergency_lock:
            QMessageBox.warning(self, t('msg.Reset_all_title'), t('msg.Emergency_write_block'))
            return
        busy = self._hw_busy_reason(exclude="SRK")
        if busy:
            log.warning("SRK measurement blocked — hardware busy: %s", busy)
            QMessageBox.warning(self, t('msg.Reset_all_title'), t('msg.Hw_busy'))
            return
        self.srk.measure(
            self.client, self._zone_dict(), self.app_config, self.calibration,
            self.ug2_track_radio.isChecked(), self.ug2_offset.value(),
            self.srk_repeats.value(),
            is_triode=self._is_triode,
            ug1_sweep=self.srk_sweep_cb.isChecked(),
            uh=self.uh_input.value(), ih=self.ih_input.value(),
        )

    # ------------------------------------------------------------------
    # PDF export
    # ------------------------------------------------------------------

    def _export_pdf(self) -> None:
        """Export measurement report as PDF."""
        lamp = find_lamp(self.lamps, self.lamp_combo.currentText())
        export_pdf(
            parent=self,
            points=self._get_exportable_points(),
            tube_type=self.lamp_combo.currentText(),
            lamp_id=self.lamp_panel.lamp_id(),
            lamp=lamp,
            srk_results=self.srk.srk_results,
            plot_renderer=self.plot_renderer,
            plot_widget=self.plot,
            transfer_widget=self.transfer_plot,
            mfg_date=self.lamp_panel.mfg_date(),
            config=self.app_config,
            scan_meta=getattr(self, "_last_scan_meta", None),
        )

    def _export_amp_pdf(self) -> None:
        """Export the last amplifier analysis as a PDF report."""
        from app.amp_report_pdf import (
            AMP_REPORT_SECTIONS,
            build_amp_header_lines,
            generate_amp_pdf_report,
            render_spectrum_pixmap,
        )
        from app.export_manager import render_plot_pixmap
        from app.report_options_dialog import ask_report_options

        result = getattr(self, "_last_amp_result", None)
        if result is None or not result.per_source:
            QMessageBox.warning(self, t("msg.Export_PDF"), t("amp.no_data"))
            return
        tab = self.amplifier_tab
        dist = next((sr.dist for sr in result.per_source.values()
                     if sr.dist), None)
        available = {
            "amp_results": "",
            "amp_ltspice": ("" if getattr(self, "_last_amp_verify", None)
                            else "report.Na_no_verify"),
            "amp_spectrum": "" if dist else "report.Na_no_spectrum",
            "amp_plot_thd_sweep": ("" if tab._amp_sweep_data
                                   else "report.Na_no_sweep"),
            "amp_plot_ra_sweep": ("" if tab._ra_sweep_data
                                  else "report.Na_no_sweep"),
            "amp_plot_pareto": ("" if tab._pareto_data
                                else "report.Na_no_pareto"),
        }
        opts = ask_report_options(
            self, available, self.app_config,
            specs=AMP_REPORT_SECTIONS, session_key="amp")
        if opts is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, t("msg.Export_PDF"), "", t("msg.PDF_filter"))
        if not path:
            return
        from app.report_options_dialog import resolve_report_language
        from i18n_setup import locale_override

        try:
            # the DOCUMENT text follows the configured report language;
            # the panel keeps the UI language (report-language contract). The
            # THD/Ra/Pareto images are WYSIWYG screenshots of UI plots;
            # the spectrum is generated fresh inside the override.
            with locale_override(resolve_report_language(self.app_config)):
                images = []
                if "amp_spectrum" in opts.sections and dist:
                    pm = render_spectrum_pixmap(dist)
                    if pm is not None:
                        images.append((t("report.Amp_spectrum_title"), pm))
                if "amp_plot_thd_sweep" in opts.sections:
                    images.append((t("amp.thd_vs_amplitude"),
                                   render_plot_pixmap(tab.thd_pout_plot)))
                if "amp_plot_ra_sweep" in opts.sections:
                    images.append((t("amp.hd_vs_ra"),
                                   render_plot_pixmap(tab.hd_ra_plot)))
                if "amp_plot_pareto" in opts.sections:
                    images.append((t("amp.pareto_title"),
                                   render_plot_pixmap(tab.pareto_plot)))
                results_html = (tab.format_results_html(result)
                                if "amp_results" in opts.sections else "")
                verify_html = ""
                verify = getattr(self, "_last_amp_verify", None)
                if "amp_ltspice" in opts.sections and verify is not None:
                    from app.amp_report_pdf import build_verify_table_html
                    verify_html = build_verify_table_html(
                        verify, self._amp_engine_reference())
                    if getattr(self, "_amp_verify_stale", False):
                        verify_html = self._verify_stale_prefix() + verify_html
                generate_amp_pdf_report(
                    path,
                    tube_type=self.lamp_combo.currentText(),
                    header_lines=build_amp_header_lines(
                        self.lamp_panel.lamp_id(), result.params,
                        data_label=self._amp_data_label(result.params)),
                    results_html=results_html,
                    images=images,
                    verify_html=verify_html,
                )
            QMessageBox.information(
                self, t("msg.Export_PDF"), t("msg.PDF_saved", path=path))
        except Exception as exc:
            log.exception("Amplifier PDF export failed")
            QMessageBox.critical(
                self, t("msg.Export_PDF"),
                t("msg.PDF_error", error=str(exc)))

    # ------------------------------------------------------------------
    # LTspice verification of the amplifier analysis
    # ------------------------------------------------------------------

    def _amp_data_label(self, params) -> str:
        """Human label of the data the analysis actually ran on —
        mirrors ``select_analysis_points`` resolution order."""
        labels = self.amp_engine._series_labels
        sid = params.series_id
        if sid is not None and any(p.get("series_id") == sid
                                   for p in self.amp_engine._all_points):
            return labels.get(sid) or f"series {sid}"
        if any(p.get("series_id") == 0
               for p in self.amp_engine._all_points):
            return t("report.Amp_data_current")
        return t("report.Amp_data_all")

    def _on_amp_verify_ltspice(self) -> None:
        """Start LTspice verification of the last amplifier analysis."""
        from app.workers import AmpVerifyWorker
        from lm19 import amplifier
        from lm19.ltspice_verify import (
            VerifyRequest,
            ltspice_available,
            resolve_verify_workdir,
        )

        result = getattr(self, "_last_amp_result", None)
        if result is None or not result.per_source or result.params is None:
            QMessageBox.warning(self, t("report.Verify_btn"),
                                t("amp.no_data"))
            return
        from lm19.ltspice_raw import LTSPICE_EXE
        ltspice_exe = self.app_config.ltspice_exe or LTSPICE_EXE
        if not ltspice_available(ltspice_exe):
            QMessageBox.warning(self, t("report.Verify_btn"),
                                t("report.Verify_no_ltspice",
                                  path=ltspice_exe))
            return
        params = result.params
        # The SAME series the analysis ran on — the plot may hold several
        # lamps (Compare overlays) and the panel selects one of them.
        points = amplifier.select_analysis_points(
            self.amp_engine._all_points, series_id=params.series_id,
        )
        if not points:
            QMessageBox.warning(self, t("report.Verify_btn"),
                                t("amp.no_data"))
            return
        dist = next((sr.dist for sr in result.per_source.values()
                     if sr.dist), None)
        half_swing = (dist or {}).get("half_swing") or params.half_swing or 0.0
        if half_swing <= 0:
            QMessageBox.warning(self, t("report.Verify_btn"),
                                t("report.Verify_needs_swing"))
            return
        topology = self._current_ug2_mode()
        ug2 = params.ug2_filter
        if topology != TOPOLOGY_TRIODE and not ug2:
            screens = sorted(p.get("ug2", 0.0) for p in points
                             if p.get("ug2", 0.0) > 0)
            ug2 = screens[len(screens) // 2] if screens else None
            if ug2 is None:
                QMessageBox.warning(self, t("report.Verify_btn"),
                                    t("report.Verify_needs_ug2"))
                return
        model_type = next(
            (s for s in params.sources
             if s in (MODEL_TYPE_KOREN, MODEL_TYPE_DEMPWOLF, MODEL_TYPE_REEFMAN)),
            MODEL_TYPE_KOREN)
        # Analysis on a model source → verify THAT model, not a refit
        # (same model_type lookup the engine uses to resolve the source).
        loaded_model = None
        if model_type in params.sources:
            loaded_model = next(
                (m for m in self.amp_engine._series_models.values()
                 if getattr(m, "model_type", "") == model_type), None)
        # An explicit fitter choice in the combo overrides Auto: the user
        # asked for THAT fitter → always a fresh fit (never the loaded
        # model, even of the same type — the basis line says which).
        fitter_choice = (self.amp_control_panel.verify_fitter_combo
                         .currentData() or "")
        if fitter_choice:
            model_type = fitter_choice
            loaded_model = None
        request = VerifyRequest(
            circuit=params.circuit,
            tube_type=self.lamp_combo.currentText(),
            topology=topology, points=points, model_type=model_type,
            ub=params.ub, ra_kohm=params.ra, ug1_bias=params.ug1_bias,
            half_swing=float(half_swing), ug2=ug2,
            ul_tap=float(params.ul_tap or 0.0),
            amp_sweep=self.amp_control_panel.verify_sweep_cb.isChecked(),
            imd=self.amp_control_panel.verify_imd_cb.isChecked(),
            data_label=self._amp_data_label(params),
            model=loaded_model,
        )
        # Worker reattach canon (ML-002): cleanup the old worker BEFORE
        # reassignment; a stuck thread keeps its reference and is visible.
        old = getattr(self, "_amp_verify_worker", None)
        if old is not None:
            if not old.cleanup():
                QMessageBox.warning(self, t("report.Verify_btn"),
                                    t("report.Verify_worker_stuck"))
                return
            self._amp_verify_worker = None
        workdir = resolve_verify_workdir(
            self.app_config.ltspice_verify_dir,
            Path(__file__).resolve().parents[1])
        worker = AmpVerifyWorker(request, str(workdir),
                                 ltspice_exe=ltspice_exe)
        worker.progress.connect(self._on_amp_verify_progress)
        worker.finished_result.connect(self._on_amp_verify_done)
        worker.failed.connect(self._on_amp_verify_failed)
        self._amp_verify_worker = worker
        self._last_amp_verify_params = params  # staleness reference
        self._amp_verify_stale = False
        self.amp_control_panel.set_verify_running(True)
        self.amp_control_panel.verify_status_label.setText(
            t("report.Verify_running", stage="start"))
        worker.start()

    @staticmethod
    def _verify_stale_prefix() -> str:
        """⚠ line for a verification that predates the current analysis
        parameters — built with the ACTIVE locale (panel: UI language;
        PDF: inside the report-language override)."""
        from app.ui_theme import AMP_WARNING_HTML_COLOR
        return (f'<span style="color:{AMP_WARNING_HTML_COLOR}">⚠ '
                f"{t('report.Verify_stale')}</span><br>")

    def _mark_amp_verify_stale_if_needed(self, params) -> None:
        """A verification belongs to the analysis it ran for — once the
        parameters change, the stored table (panel AND future PDF) gets a
        visible ⚠ prefix instead of silently posing as current."""
        html = getattr(self, "_last_amp_verify_html", "")
        if not html or getattr(self, "_amp_verify_stale", False):
            return
        if params == getattr(self, "_last_amp_verify_params", params):
            return
        self._last_amp_verify_html = self._verify_stale_prefix() + html
        self._amp_verify_stale = True
        self.amp_control_panel.show_verify_results(self._last_amp_verify_html)

    def _on_amp_verify_cancel(self) -> None:
        """Stable cancel slot (ML-003): acts on the current worker."""
        worker = getattr(self, "_amp_verify_worker", None)
        if worker is not None:
            worker.stop()

    def _on_amp_verify_progress(self, stage: str) -> None:
        self.amp_control_panel.verify_status_label.setText(
            t("report.Verify_running", stage=stage))

    def _on_amp_verify_done(self, verify_result) -> None:
        from app.amp_report_pdf import build_verify_table_html

        self.amp_control_panel.set_verify_running(False)
        html = build_verify_table_html(verify_result,
                                       self._amp_engine_reference())
        # the raw result is kept so the PDF can REBUILD the table in the
        # configured report language (panel html stays UI-language)
        self._last_amp_verify = verify_result
        self._last_amp_verify_html = html
        self.amp_control_panel.show_verify_results(html)

    def _on_amp_verify_failed(self, message: str) -> None:
        self.amp_control_panel.set_verify_running(False)
        self.amp_control_panel.verify_status_label.setText("")
        QMessageBox.critical(self, t("report.Verify_btn"),
                             t("report.Verify_failed", error=message))

    def _amp_engine_reference(self) -> Dict:
        """Engine-side numbers for the comparison table, with their basis.

        Prefers a model source (tightest comparison against the fitted
        ``.sub``) and falls back to measurements; the basis label always
        names the source and HD method (method-visibility rule).
        """
        result = getattr(self, "_last_amp_result", None)
        if result is None:
            return {}
        name = None
        source = None
        for key, sr in result.per_source.items():
            if sr.dist is None:
                continue
            if source is None or key != "measurements":
                name, source = key, sr
                if key != "measurements":
                    break
        if source is None or source.dist is None:
            return {}
        dist = source.dist
        return {
            "basis": f"{name}/{dist.get('method') or source.method_used or '?'}",
            "thd": dist.get("thd"),
            "hd2": dist.get("hd2"),
            "hd3": dist.get("hd3"),
            "pout_fund_mw": dist.get("pout_fund_mw") or dist.get("pout_mw"),
            "pout_is_fund": dist.get("pout_fund_mw") is not None,
            "imd": source.imd,
            "sweep_amp": source.sweep_amp,
        }

    # ------------------------------------------------------------------
    # SPICE model export
    # ------------------------------------------------------------------

    def _get_exportable_points(self) -> List[Dict]:
        """Return points for export: scan data preferred, all plot data as fallback."""
        scan_pts = [p for p in self.plot_mgr.points if p.get("series_id") == 0]
        if scan_pts:
            return scan_pts
        # No scan data — use all points on the plot (e.g. from Compare overlay)
        return list(self.plot_mgr.points)

    def _current_ug2_mode(self) -> str:
        """Canonical Ug2-mode string for the current UI state.

        Single source of truth for "triode" / "triode_connected" / "pentode" —
        used by scan-metadata writers and SPICE export so the fitter sees the
        real topology (a triode-connected scan must not be fit as a pentode).
        """
        if self._is_triode:
            return TOPOLOGY_TRIODE
        return (TOPOLOGY_TRIODE_CONNECTED if self.ug2_track_radio.isChecked()
                else TOPOLOGY_PENTODE)

    def _export_spice(self) -> None:
        """Export SPICE model fitted to measured data."""
        # Gather amplifier parameters from UI for schematic generation
        amp_params = None
        if hasattr(self, "amp_control_panel"):
            p = self.amp_control_panel.params_snapshot()
            amp_params = {
                "ub": p.ub,
                "ra_ohm": f"{p.ra}k" if p.ra >= 1 else f"{int(p.ra * 1000)}",
                "rk_ohm": "1.5k",  # default, not in AmpParams directly
                "ra_dc_ohm": f"{int(p.ra_dc * 1000)}",
                "ug2": getattr(p, "ug2_filter", 250) or 250,
                "ra_aa_ohm": f"{p.pp_raa}k" if p.pp_raa >= 1 else f"{int(p.pp_raa * 1000)}",
            }
        export_spice(
            parent=self,
            points=self._get_exportable_points(),
            tube_type=self.lamp_combo.currentText(),
            plot_mgr=self.plot_mgr,
            topology=self._current_ug2_mode(),
            amp_params=amp_params,
            all_points=list(self.plot_mgr.points),
            series_labels=self.plot_mgr.series_labels,
            mfg_date=self.lamp_panel.mfg_date(),
            series_models=self.plot_mgr.series_models,
        )

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def _export_csv(self) -> None:
        """Export current plot data as CSV."""
        srk_data = None
        if self.srk.srk_results:
            valid = [r for r in self.srk.srk_results if r.get("s") is not None]
            if valid:
                srk_data = {
                    "s": sum(r["s"] for r in valid) / len(valid),
                    "r": sum(r["r"] for r in valid) / len(valid),
                    "k": sum(r["k"] for r in valid) / len(valid),
                }
        export_csv(
            parent=self,
            points=self._get_exportable_points(),
            tube_type=self.lamp_combo.currentText(),
            lamp_id=self.lamp_panel.lamp_id(),
            name=self.measurement_name_edit.text().strip(),
            mfg_date=self.lamp_panel.mfg_date(),
            srk=srk_data,
            is_triode=self._is_triode,
        )

    # ------------------------------------------------------------------
    # uTracer .utd export
    # ------------------------------------------------------------------

    def _export_utd(self) -> None:
        """Export current plot data as uTracer .utd file."""
        export_utd(
            parent=self,
            points=self._get_exportable_points(),
            tube_type=self.lamp_combo.currentText(),
        )

    # ------------------------------------------------------------------
    # Operating point calculator
    # ------------------------------------------------------------------

    def _show_op_calculator(self) -> None:
        """Navigate to Amplifier tab (replaces the old dialog)."""
        idx = self.plot_tabs.indexOf(self.amplifier_tab)
        if idx >= 0:
            self.plot_tabs.setCurrentIndex(idx)
            self.tabs.setCurrentIndex(0)
            if not self.load_line_cb.isChecked():
                self.load_line_cb.setChecked(True)

    # ------------------------------------------------------------------
    # Model dialog
    # ------------------------------------------------------------------

    def _show_model_dialog(self) -> None:
        """Open ModelDialog and add the resulting model as a series."""
        from app.model_dialog import ModelDialog

        scan_settings = {
            "ua_start": self.ua_start.value(),
            "ua_stop": self.ua_stop.value(),
            "ua_step": self.ua_step.value(),
            "ug1_start": self.ug1_start.value(),
            "ug1_stop": self.ug1_stop.value(),
            "ug1_step": self.ug1_step.value(),
            "ug2_track_ua": self.ug2_track_radio.isChecked(),
            "ug2_offset": self.ug2_offset.value(),
            "ug2_start": self.ug2_start.value(),
            "ug2_stop": self.ug2_stop.value(),
            "ug2_step": self.ug2_step.value(),
            "uh": self.uh_input.value(),
            "ih": self.ih_input.value(),
        }

        # Use selected series for fit
        calc_combo = self.lamp_calc_combo
        selected_sid = calc_combo.currentData()
        if selected_sid is not None:
            fit_points = [
                p for p in self.plot_mgr.points
                if p.get("series_id", 0) == selected_sid
            ]
            series_name = calc_combo.currentText()
        else:
            # No series in combo (empty plot)
            fit_points = []
            series_name = ""

        dlg = ModelDialog(
            self,
            points=fit_points,
            scan_settings=scan_settings,
            is_triode=self._is_triode,
            series_name=series_name,
            ia_dead_thr=self.app_config.ia_dead_threshold,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # ML-087: the fit verdict used to be written into the dialog's own
        # label one instant before accept() closed it — show it where the
        # user can actually read it (failure-visibility rule). It rides the same
        # indicator as the fit alerts and survives until the next fit
        # replaces it, so a verdict is never missed by looking away.
        self._set_ui_warnings(
            "fit_verdict", [dlg.fit_verdict] if dlg.fit_verdict else [])
        self._set_ui_warnings("model_fit", dlg.fit_alerts)

        results = dlg.results_multi()
        if not results:
            return

        for model, grid, label in results:
            sid = self.plot_mgr.allocate_series_id()

            # Generate scan points and tag with series_id
            pts = model.generate_scan(grid)
            for p in pts:
                p["series_id"] = sid

            self.plot_mgr.points.extend(pts)
            self.plot_mgr.series_models[sid] = model
            self.plot_mgr.series_grids[sid] = grid
            self.plot_mgr.series_labels[sid] = label
            self.plot_mgr.series_colors[sid] = SERIES_PALETTE[
                sid % len(SERIES_PALETTE)
            ]
            self.plot_mgr.series_ug2_track[sid] = (
                model.topology == TOPOLOGY_TRIODE or grid.ug2_track_ua
            )

        self.plot_mgr.invalidate_cache()
        self.plot_mgr.refresh_lamp_combos()
        self.plot_mgr.refresh_ug2_combos(self.plot_mgr.points)
        self.plot_mgr.render_all()

    # Import — delegated to ImportController (see app/import_controller.py)

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def _shutdown_opt_worker(self) -> None:
        """Stop a running OptimizeWorker before the window closes.

        ``OptimizeWorker`` is a ``QThread`` child of this window operating on
        in-memory data (no serial client). If an optimization is still running
        at close, a live ``QThread`` freed by GC calls ``qFatal`` and aborts
        the process. ``cleanup()`` stops it (``cancel`` == ``stop``, polled
        inside the grid loops) and disconnects signals; keep the reference if
        it did not drain in time (``BaseWorker.cleanup`` contract).
        """
        if getattr(self, "_opt_worker", None) is not None:
            if self._opt_worker.cleanup():
                self._opt_worker = None

    def showEvent(self, event) -> None:
        """First show only: size the Measure control column to its content.

        Guarded by a flag — later shows (restore from minimised, tab
        switches) must not undo a width the user chose by dragging.
        """
        super().showEvent(event)
        if not self._measure_splitter_fitted:
            self._measure_splitter_fitted = True
            self._fit_measure_splitter()

    def closeEvent(self, event) -> None:
        self.scan_ctrl.shutdown()
        if hasattr(self, "health_tab"):
            self.health_tab.shutdown()
        if self.srk and self.srk.worker:
            self.srk.worker.cleanup()
        self._shutdown_opt_worker()
        # All hardware-owning workers are now stopped → drive the HV outputs
        # to the safe state synchronously while the port is still open.
        # Closing mid-scan (or with a working point set) otherwise leaves the
        # tube energized: the firmware holds the last Ua/Ug2 setpoints, and
        # the scan finish handler's reset never runs during teardown. Keep
        # the heater (reset_heater=False) — see _emergency_zero_outputs.
        self._emergency_zero_outputs(reset_heater=False)
        self.conn_mgr.shutdown()
        super().closeEvent(event)
