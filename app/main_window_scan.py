"""Scan-controller integration mixin for MainWindow.

Owns scan/measurement orchestration so the host class stays focused
on lifecycle and the scan flow lives in one place.

Method groups:
  - run/stop: ``_run_scan``, ``_stop_scan``, ``_on_scan_started``
  - per-point: ``_on_point``, ``_update_scan_time_format``
  - completion: ``_on_scan_finished``, ``_save_and_display_measurement``,
    ``_show_scan_summary_dialog``, ``_update_quality_label``
  - error paths: ``_on_scan_failed``, ``_on_scan_comm_error``

Host class (MainWindow) provides via __init__:
  - self.scan_ctrl (ScanController), self.conn_mgr (ConnectionManager),
    self.plot_mgr (PlotManager), self.live_panel (LivePanel),
    self.srk (SrkController), self.compare_tab (CompareTab)
  - self.client (property → conn_mgr.client)
  - self.app_config, self.calibration, self.lamps, self._device_limits
  - widget refs from MainWindowBuilders mixin: run_btn, stop_btn,
    scan_progress_bar, ua/ug1/ug2 spinboxes, anode_group, ug2_track_radio,
    pa_max_input/pa_over_pct/pig2_over_pct, srk_repeats/srk_sweep_cb/
    srk_measure_separately, refine_cb, ia_samples_spin, plot, quality_label,
    uh_input/ih_input, preheat_enabled, ug2_offset
  - state: _emergency_lock, _is_triode, _pig2_max_val, _scan_summary,
    scan_total_points, scan_in_progress (property), preheat_done (property)
  - helpers: _auto_ug1_step, _update_live_from_point, _reset_outputs_after_scan,
    _build_scan_metadata, _zone_dict, _start_preheat
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List

from PySide6.QtWidgets import QMessageBox

from app.ui_theme import COLOR_MID_GRAY, QUALITY_COLORS, STYLE_BOLD_LABEL_SM
from app.save_recovery import save_with_recovery, suggested_filename
from i18n_setup import t
from lm19.scan.events import (CURVE_STATUS_COMPLETED,
                              CURVE_STATUS_USER_STOP)
from lm19.config import find_lamp
from lm19.measurements import save_measurement
from lm19.quality import compute_quality
from lm19.scan import ScanRange, ScanSettings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ScanRunContext:
    """Hardware-relevant settings of the RUNNING scan, frozen at start.

    The finish path (after-scan SRK corner measurement, SRK computed
    from scan data) must see what THIS run commanded — the scan controls
    stay live and the user routinely arms the next run mid-scan.  A live
    read at finish would make the SRK command the screen in the wrong
    mode and file S/R/K measured at a different operating condition into
    this measurement.
    """
    is_triode: bool
    ug2_track: bool
    ug2_offset: float
    # Gated heater values (0.0 with preheat disabled) — exactly what the
    # scan commanded, which is what the SRK heater-loss check expects.
    uh: float
    ih: float


class MainWindowScan:
    """Mixin: scan controller integration + measurement save/display."""

    def _run_scan(self) -> None:
        if self._emergency_lock:
            QMessageBox.warning(self, t('msg.Reset_all_title'), t('msg.Emergency_write_block'))
            return
        if not self.client or not self.client.is_open():
            QMessageBox.warning(self, t('msg.COM'), t('msg.Connect_first'))
            return
        if self.scan_worker and self.scan_worker.isRunning():
            return
        busy = self._hw_busy_reason(exclude="scan")
        if busy:
            log.warning("Scan blocked — hardware busy: %s", busy)
            QMessageBox.warning(self, t('msg.Reset_all_title'), t('msg.Hw_busy'))
            return

        # Warn on the values that will actually be COMMANDED (ML-123): with
        # 'Enable preheat' unchecked the scan runs at uh=ih=0 regardless of
        # the spinboxes, and reading the raw spinboxes here skipped the
        # cold-scan warning in exactly the case it exists for.
        preheat_on = self.preheat_enabled.isChecked()
        uh_val = self.uh_input.value() if preheat_on else 0.0
        ih_val = self.ih_input.value() if preheat_on else 0.0
        if (
            uh_val < self.app_config.heater_zero_warn_uh_v
            and ih_val < self.app_config.heater_zero_warn_ih_a
        ):
            reply = QMessageBox.question(
                self,
                t('msg.Heater_warning'),
                t('msg.Heater_zero_confirm'),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self.conn_mgr.set_poller_active(False)
        # Keep imported points (series_id != 0), clear only scan points
        self.plot_mgr.points = [p for p in self.plot_mgr.points if p.get("series_id", 0) != 0]
        if hasattr(self, "working_line"):
            self.working_line.invalidate()
        self.plot_mgr.current_curve_points = []
        self.plot_mgr.labeled_ug1 = set()
        self.scan_in_progress = True
        # Stamp the plot with the Ug2 mode THIS run measures in, so the
        # curves keep their grouping when the next run is armed in the
        # other mode (see PlotManager.set_scan_ug2_track).
        ug2_track = self.ug2_track_radio.isChecked()
        self.plot_mgr.set_scan_ug2_track(self._is_triode or ug2_track)
        self.plot_mgr.invalidate_cache()
        self.plot_mgr.render_2d_only()

        ug1_step = self.ug1_step.value()
        if ug1_step <= 0:
            ug1_step = self._auto_ug1_step(self.ug1_start.value(), self.ug1_stop.value())
            self.ug1_step.setValue(ug1_step)
        # Freeze the settings this run is made with — AFTER the auto Ug1
        # step above lands in its spinbox, so the saved measurement
        # records the step actually swept and not the "0 = auto" input.
        self._freeze_scan_metadata()
        self._scan_run_ctx = _ScanRunContext(
            is_triode=self._is_triode,
            ug2_track=ug2_track,
            ug2_offset=self.ug2_offset.value(),
            uh=uh_val,
            ih=ih_val,
        )
        # Same gated values the warning above was based on (single source).
        uh_value = uh_val
        ih_value = ih_val
        an_value = self.anode_group.checkedId()
        self.live_panel.set_an(an_value)
        pa_over = self.pa_over_pct.value()
        pa_max_w = self.pa_max_input.value()
        pig2_over = self.pig2_over_pct.value()
        pig2_max = self._pig2_max_val if self._pig2_max_val else 0.0
        if not self._is_triode and not pig2_max:
            log.warning(
                "Pg2 protection disabled — Pig2_max not set for this pentode; "
                "set it in lamps.json to enable screen-grid protection"
            )
        ig2_hw_max = self._device_limits.get("ig2_max", 0.0)
        ig2_max_ma = ig2_hw_max * (self.app_config.ig2_hw_margin_pct / 100.0)
        settings = ScanSettings(
            ua=ScanRange(self.ua_start.value(), self.ua_stop.value(), self.ua_step.value()),
            ug1=ScanRange(self.ug1_start.value(), self.ug1_stop.value(), ug1_step),
            ug2=ScanRange(self.ug2_start.value(), self.ug2_stop.value(), self.ug2_step.value()),
            uh=uh_value,
            ih=ih_value,
            an=an_value,
            is_triode=self._is_triode,
            ug2_track_ua=ug2_track,
            ug2_offset=self.ug2_offset.value(),
            calibration=self.calibration,
            pa_max_w=pa_max_w,
            pa_over_pct=pa_over,
            pig2_max_w=pig2_max,
            pig2_over_pct=pig2_over,
            ua_settle_per_volt_s=self.app_config.scan_ua_settle_per_volt_s,
            ua_settle_base_s=self.app_config.scan_ua_settle_base_s,
            ua_tolerance=self.app_config.scan_ua_tolerance,
            ua_retries=self.app_config.scan_ua_retries,
            ug1_settle_per_volt_s=self.app_config.scan_ug1_settle_per_volt_s,
            ug1_settle_base_s=self.app_config.scan_ug1_settle_base_s,
            ug1_tolerance=self.app_config.scan_ug1_tolerance,
            ug1_retries=self.app_config.scan_ug1_retries,
            ug2_settle_per_volt_s=self.app_config.scan_ug2_settle_per_volt_s,
            ug2_settle_base_s=self.app_config.scan_ug2_settle_base_s,
            ug2_tolerance=self.app_config.scan_ug2_tolerance,
            ug2_retries=self.app_config.scan_ug2_retries,
            ia_samples=self.ia_samples_spin.value(),
            ia_outlier_ratio=self.app_config.scan_ia_outlier_ratio,
            ia_outlier_reread_samples=(
                self.app_config.scan_ia_outlier_reread_samples),
            refine_enabled=self.refine_cb.isChecked(),
            refine_max_depth=self.app_config.scan_refine_max_depth,
            refine_min_step_ua=self.app_config.scan_refine_min_step_ua,
            refine_onset_ma=self.app_config.scan_refine_onset_ma,
            refine_curvature_thr=self.app_config.scan_refine_curvature_thr,
            refine_gradient_ratio=self.app_config.scan_refine_gradient_ratio,
            refine_ig2_delta_min=self.app_config.scan_refine_ig2_delta_min,
            refine_delta_ia_thr=self.app_config.scan_refine_delta_ia_thr,
            down_max_step_v=self.app_config.scan_down_max_step_v,
            comm_retries=self.app_config.scan_comm_retries,
            ig2_max_ma=ig2_max_ma if not self._is_triode else 0.0,
        )
        self.scan_progress_bar.setValue(0)
        self.scan_progress_bar.setFormat("%p%")
        self.live_panel.clear_protection()
        log.info("Starting scan: lamp=%s id=%s",
                 self._app_ctx.get_current_tube_type(),
                 self._app_ctx.get_current_lamp_id())
        self.scan_ctrl.start_scan(self.client, settings)

    def _run_ctx_for_finish(self) -> _ScanRunContext:
        """Run context of the scan being finished (see _ScanRunContext).

        Defensive twin of ``_scan_meta_for_save``: without a recorded
        context (never the case for a scan started by ``_run_scan``) the
        current UI state is used — loudly, since it may describe the run
        armed next rather than the one that produced the points.
        """
        ctx = getattr(self, "_scan_run_ctx", None)
        if ctx is not None:
            return ctx
        log.warning(
            "No scan-start run context — using the CURRENT UI state for "
            "the after-scan SRK; it may not match the finished run")
        preheat_on = self.preheat_enabled.isChecked()
        return _ScanRunContext(
            is_triode=self._is_triode,
            ug2_track=self.ug2_track_radio.isChecked(),
            ug2_offset=self.ug2_offset.value(),
            uh=self.uh_input.value() if preheat_on else 0.0,
            ih=self.ih_input.value() if preheat_on else 0.0,
        )

    def _on_scan_started(self) -> None:
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._scan_summary = None  # reset — will be set by progress event

    def _stop_scan(self) -> None:
        # Defer the output reset to _on_scan_finished. stop_scan() only flags
        # the worker, which keeps running (settle / verify-retry) for up to one
        # point. Resetting outputs now would race the still-live worker, whose
        # verify-retry could re-assert Ua/Ug2 *after* the reset and leave the
        # lamp energized. The deferred path (reset_on_finish) fires once the
        # worker has actually finished.
        self.scan_ctrl.stop_scan()

    def _on_point(self, point: Dict) -> None:
        pm = self.plot_mgr
        event = point.get("event")
        if event:
            # --- Control events (no measurement data) ---
            if event == "curve_done":
                pm.refresh_ug2_combos(pm.points)
                pm.refresh_lamp_combos()
                if self.scan_in_progress:
                    pm.render_curve_incremental(point)
                    pm.current_curve_points = []
                    pm.update_scan_marker()
                else:
                    pm.invalidate_cache()
                    pm.render_all()
            elif event == "refine_count":
                pass  # scan_ctrl already updated scan_total_points
            elif event == "protection":
                self._update_live_from_point(point)
                self.live_panel.show_protection(point.get("param", ""))
            elif event == "hw_protection":
                self.live_panel.show_protection()
                self.scan_progress_bar.setFormat(t("msg.Hw_error_reset"))
            elif event == "heater_restoring":
                self.scan_progress_bar.setFormat(t("msg.Heater_restoring"))
            elif event == "hw_protection_cleared":
                self.live_panel.clear_protection()
                self._update_scan_time_format()
            elif event == "scan_summary":
                self._scan_summary = point
            return  # all events exit here — never fall through to data path
        # --- Measurement data point ---
        point.setdefault("series_id", 0)
        pm.points.append(point)
        pm.current_curve_points.append(point)
        if self.scan_total_points > 0:
            percent = min(100, int(len(pm.points) * 100 / self.scan_total_points))
            self.scan_progress_bar.setValue(percent)
            self._update_scan_time_format()
        self.plot.plot([point["ua"]], [point["ia"]], pen=None, symbol="o", symbolSize=5)
        self._update_live_from_point(point)
        self.conn_mgr.update_io_stats()

    def _update_scan_time_format(self) -> None:
        """Update progress bar format with elapsed / ETA times."""
        if self.scan_ctrl.scan_start_time is None:
            self.scan_progress_bar.setFormat("%p%")
            return
        elapsed = time.monotonic() - self.scan_ctrl.scan_start_time
        elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
        percent = self.scan_progress_bar.value()
        if percent > 0:
            eta = elapsed * (100 - percent) / percent
            eta_str = f"{int(eta // 60)}:{int(eta % 60):02d}"
            self.scan_progress_bar.setFormat(f"%p% — {elapsed_str} / ~{eta_str}")
        else:
            self.scan_progress_bar.setFormat(f"%p% — {elapsed_str}")

    def _on_scan_failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.scan_in_progress = False
        self.plot_mgr.renderer.resume_markers()
        self.scan_progress_bar.setValue(0)
        self.scan_progress_bar.setFormat("%p%")
        log.warning("Scan failed: %s", message)

        # Partial scan points collected before the error (exclude imported)
        partial_points = [p for p in self.plot_mgr.points if p.get("series_id") == 0]

        if partial_points:
            reply = QMessageBox.question(
                self,
                t('msg.Scan_error'),
                t('msg.Scan_failed_partial', message=message, count=len(partial_points)),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                scan_meta = self._scan_meta_for_save()
                self._last_scan_meta = scan_meta  # PDF scan-settings section
                zone = self._zone_dict()
                lamp = find_lamp(self.lamps, scan_meta["tube_type"])
                self.srk.compute_from_scan(
                    partial_points, zone, scan_meta, lamp=lamp,
                    is_triode=self._run_ctx_for_finish().is_triode)
                return

        self._reset_outputs_after_scan(reset_heater=False, reset_order=["Ug2", "Ug1", "Ua", "Uh"])
        self.conn_mgr.set_poller_active(True)
        if not partial_points:
            QMessageBox.critical(self, t('msg.Scan'), message)

    def _on_scan_comm_error(self, message: str, attempt: int) -> None:
        """Show Retry / Ignore / Abort dialog when comm error occurs during scan."""
        collected = len([p for p in self.plot_mgr.points if p.get("series_id") == 0])
        box = QMessageBox(
            QMessageBox.Warning,
            t('msg.Scan_error'),
            t('msg.Scan_comm_retry', message=message, attempt=attempt,
              count=collected),
            parent=self,
        )
        btn_retry = box.addButton(QMessageBox.Retry)
        btn_ignore = box.addButton(QMessageBox.Ignore)
        btn_abort = box.addButton(QMessageBox.Abort)
        box.setDefaultButton(btn_retry)
        box.exec()
        clicked = box.clickedButton()
        if clicked == btn_retry:
            decision = "retry"
        elif clicked == btn_abort:
            decision = "abort"
        else:
            decision = "skip"
        self.scan_ctrl.respond_comm_error(decision)

    def _on_scan_finished(self, points: List[Dict]) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.scan_in_progress = False
        self.plot_mgr.renderer.resume_markers()
        self.scan_progress_bar.setValue(100)
        if self.scan_ctrl.scan_start_time is not None:
            elapsed = time.monotonic() - self.scan_ctrl.scan_start_time
            elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
            self.scan_progress_bar.setFormat(f"100% — {elapsed_str}")
        else:
            self.scan_progress_bar.setFormat("%p%")

        # Zero outputs now that the worker has actually finished, when the
        # user stopped the scan (reset deferred from _stop_scan) or there is
        # no data. Doing it here — not in _stop_scan — prevents a live
        # verify-retry from re-asserting HV after the reset.
        if self.scan_ctrl.reset_on_finish or not points:
            self._reset_outputs_after_scan(reset_heater=False, reset_order=["Ug2", "Ug1", "Ua", "Uh"])
            self.conn_mgr.set_poller_active(True)
        if not points:
            return

        # Preserve compare overlay (series_id != 0) alongside new scan data
        for p in points:
            p.setdefault("series_id", 0)
        compare_overlay = [p for p in self.plot_mgr.points if p.get("series_id", 0) != 0]
        self.plot_mgr.points = list(points) + compare_overlay
        if hasattr(self, "working_line"):
            self.working_line.invalidate()

        scan_meta = self._scan_meta_for_save()
        self._last_scan_meta = scan_meta  # PDF scan-settings section
        zone = self._zone_dict()

        # Everything describing the finished RUN comes from the frozen
        # context — the widgets may already be armed for the next one.
        # SRK's own options (repeats, Ug1 sweep) stay live: they describe
        # the SRK measurement about to happen, not the finished scan.
        ctx = self._run_ctx_for_finish()
        if self.srk_measure_separately.isChecked():
            # Measure SRK at corner points after scan.
            # Heater-loss check expects what the SCAN actually commanded:
            # ctx.uh/ih are the preheat-gated values of the run (ML-123),
            # so a cold scan is not false-tripped as HeaterLostError.
            self.srk.measure_after_scan(
                self.client, zone, self.app_config, self.calibration,
                ctx.ug2_track, ctx.ug2_offset,
                self.srk_repeats.value(),
                pending_points=points, pending_meta=scan_meta,
                is_triode=ctx.is_triode,
                ug1_sweep=self.srk_sweep_cb.isChecked(),
                uh=ctx.uh,
                ih=ctx.ih,
            )
        else:
            # Compute SRK from scan data — the lamp comes from the frozen
            # metadata, so re-selecting the combo mid-scan cannot compare
            # these points against another tube's reference.
            lamp = find_lamp(self.lamps, scan_meta["tube_type"])
            self.srk.compute_from_scan(points, zone, scan_meta, lamp=lamp,
                                       is_triode=ctx.is_triode)

    def _save_and_display_measurement(self, measurement: Dict, points: List[Dict]) -> None:
        """Save measurement and update UI."""
        tube_type = measurement["tube_type"]
        lamp_id = measurement["lamp_id"]
        # A save failure is recoverable: Retry / Save As / keep
        # in-memory. The in-memory compare entry + plot + summary below run
        # either way. (Serialization bugs raise TypeError — a programming
        # error left to propagate, per the failure-visibility policy.)
        save_with_recovery(
            self,
            lambda: save_measurement(tube_type, lamp_id, measurement),
            measurement,
            suggested_filename(tube_type, lamp_id,
                               str(measurement.get("timestamp", ""))),
        )
        entry = {
            "lamp_type": tube_type,
            "lamp_id": lamp_id,
            "timestamp": measurement["timestamp"],
            "name": measurement.get("name", ""),
            "mfg_date": measurement.get("mfg_date", ""),
            "points": measurement.get("points", []),
            "data": measurement,
        }
        self.compare_tab.add_entry(entry)

        # Compute and display quality score
        lamp = find_lamp(self.lamps, tube_type)
        srk = measurement.get("srk")
        scan_pts = measurement.get("points", [])
        self._update_quality_label(scan_pts, lamp, srk)

        try:
            self.plot_mgr.refresh_lamp_combos()
            self.plot_mgr.refresh_ug2_combos(self.plot_mgr.points)
            self.plot_mgr.invalidate_cache()
            self.plot_mgr.render_all()
        except Exception:
            log.exception("Failed to render plots after measurement")

        # Show scan summary dialog — always after save, details depend on
        # whether any curves were incomplete.
        if getattr(self, "_scan_summary", None) is not None:
            self._show_scan_summary_dialog(self._scan_summary)
            self._scan_summary = None

    def _show_scan_summary_dialog(self, summary: Dict) -> None:
        """Show scan completion dialog with per-curve outcomes."""
        curves = summary.get("curves", [])
        total_points = summary.get("total_points", 0)
        duration_s = summary.get("duration_s", 0.0)
        dur_str = f"{int(duration_s // 60)}:{int(duration_s % 60):02d}"

        # ML-108/109: settle/outlier degradation counters — shown even on
        # an otherwise clean scan (failure-visibility rule: the log alone is not
        # a user-facing channel).
        extras = []
        oot = summary.get("settle_out_of_tolerance", 0)
        if oot:
            extras.append(t("msg.Scan_summary_out_of_tolerance", count=oot))
        rereads = summary.get("ia_outlier_rereads", 0)
        if rereads:
            extras.append(t("msg.Scan_summary_outlier_rereads", count=rereads))
        unstable = summary.get("ia_unstable_points", 0)
        if unstable:
            extras.append(t("msg.Scan_summary_unstable_points", count=unstable))

        incomplete = [c for c in curves if c.get("status") != CURVE_STATUS_COMPLETED]
        if not incomplete:
            text = t("msg.Scan_summary_short",
                     count=total_points, duration=dur_str)
        else:
            header = t("msg.Scan_summary_issues",
                       count=total_points, duration=dur_str)
            lines = [header]
            for c in incomplete:
                status_key = f"msg.Scan_status_{c.get('status', CURVE_STATUS_USER_STOP)}"
                reason = t(status_key)
                lines.append(t("msg.Scan_curve_line",
                               ug1=f"{c.get('ug1', 0):.1f}",
                               ug2=f"{c.get('ug2', 0):.0f}",
                               points=c.get("points", 0),
                               reason=reason))
            heater = summary.get("heater_lost")
            if heater:
                lines.append("")
                lines.append(t("msg.Scan_heater_lost", message=heater))
            text = "\n".join(lines)

        if extras:
            text += "\n\n" + "\n".join(extras)

        QMessageBox.information(self, t("msg.Scan_summary_title"), text)

    def _update_quality_label(self, points: List[Dict], lamp, srk: Dict = None) -> None:
        """Update quality verdict label from measurement data."""
        if not lamp or not points:
            self.quality_label.setText("")
            return
        report = compute_quality(points, lamp, srk)
        if report.verdict == "N/A":
            self.quality_label.setText("")
            return
        ia_str = f"{report.ia_pct:.0f}" if report.ia_pct is not None else "—"
        s_str = f"{report.s_pct:.0f}" if report.s_pct is not None else "—"
        text = t('msg.Quality_verdict', verdict=report.verdict, ia=ia_str, s=s_str)
        color = QUALITY_COLORS.get(report.verdict, COLOR_MID_GRAY)
        self.quality_label.setStyleSheet(f"{STYLE_BOLD_LABEL_SM} color: {color};")
        self.quality_label.setText(text)
