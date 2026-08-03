import logging
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Signal

log = logging.getLogger(__name__)
from PySide6.QtGui import QBrush, QColor, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lm19.analysis import Zone, compute_k, compute_r, compute_s, compute_sr_zone, count_in_zone
from lm19.config import LampConfig
from lm19.protocol import LM19Serial
from lm19.scan import SrkSettings, _srk_ug1_values, measure_srk
from app.workers import BaseWorker
from i18n_setup import t


class SrkWorker(BaseWorker):
    """Run SRK (S, R, K) corner-method measurement.

    Inherits :class:`BaseWorker` for unified ``stop()`` /
    ``_stop_requested`` semantics, the ``cleanup()`` helper, and the
    standard ``failed(str)`` signal.
    """

    progress = Signal(int, int)       # (done, total) across all repeats
    finished = Signal(object)         # list of {"s", "r", "k", "points"} per repeat
    # ``failed = Signal(str)`` is inherited from BaseWorker.

    def __init__(self, client: LM19Serial, settings: SrkSettings, repeats: int = 1):
        super().__init__(client=client)
        self.settings = settings
        self.repeats = repeats

    # ── Alias for callers that read ``_stop`` directly ───────────────
    @property
    def _stop(self) -> bool:
        """Read-only alias for ``BaseWorker._stop_requested``."""
        return self._stop_requested

    def _execute(self) -> None:
        results = []
        pts_per_repeat = len(_srk_ug1_values(self.settings)) * 2  # 2 Ua values
        total_points = pts_per_repeat * self.repeats
        done_points = 0
        for i in range(self.repeats):
            if self._stop_requested:
                break
            s, r, k, points, uncertainty = measure_srk(
                self.client,
                self.settings,
                progress=lambda done, total, base=done_points: self.progress.emit(base + done, total_points),
                stop=lambda: self._stop_requested,
            )
            done_points += pts_per_repeat
            if s is not None and r is not None and k is not None:
                results.append({"s": s, "r": r, "k": k, "points": points, "uncertainty": uncertainty})
            else:
                results.append({"s": None, "r": None, "k": None, "points": points, "uncertainty": uncertainty})
        if not results or all(r["s"] is None for r in results):
            self.failed.emit(t('srk.Not_enough_data'))
        else:
            self.finished.emit(results)


class SrkResultsDialog(QDialog):
    def __init__(self, results: List[Dict], parent: QWidget = None, is_triode: bool = False):
        super().__init__(parent)
        self.setWindowTitle(t('srk.SRK_Results'))
        self.resize(500, 350)
        layout = QVBoxLayout(self)

        valid = [r for r in results if r["s"] is not None]
        n_rows = len(results) + (1 if len(valid) > 1 else 0)

        # --- SRK results table ---
        srk_table = QTableWidget(n_rows, 4)
        srk_table.setHorizontalHeaderLabels([
            t("srk.col_index"),
            t("srk.col_s"),
            t("srk.col_r"),
            t("srk.col_k"),
        ])
        srk_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        srk_table.setEditTriggers(QTableWidget.NoEditTriggers)

        def _fmt(val: float, rel: Optional[float]) -> str:
            # ML-091: the measured uncertainty must reach the user —
            # a shaky measurement is indistinguishable from a solid one
            # without it.
            if rel is None:
                return f"{val:.2f}"
            return f"{val:.2f} ±{rel * 100:.0f}%"

        for row, r in enumerate(results):
            unc = r.get("uncertainty") or {}
            srk_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            if r["s"] is not None:
                srk_table.setItem(
                    row, 1, QTableWidgetItem(_fmt(r["s"], unc.get("s"))))
                srk_table.setItem(
                    row, 2, QTableWidgetItem(_fmt(r["r"], unc.get("r"))))
                srk_table.setItem(
                    row, 3, QTableWidgetItem(_fmt(r["k"], unc.get("k"))))
            else:
                for c in range(1, 4):
                    srk_table.setItem(row, c, QTableWidgetItem("—"))

        s_avg = r_avg = k_avg = None
        if len(valid) > 1:
            avg_row = len(results)
            s_avg = sum(r["s"] for r in valid) / len(valid)
            r_avg = sum(r["r"] for r in valid) / len(valid)
            k_avg = sum(r["k"] for r in valid) / len(valid)
            avg_label = QTableWidgetItem(t("srk.avg"))
            avg_label.setBackground(QBrush(QColor(220, 240, 255)))
            srk_table.setItem(avg_row, 0, avg_label)
            for c, val in enumerate([s_avg, r_avg, k_avg], 1):
                item = QTableWidgetItem(f"{val:.2f}")
                item.setBackground(QBrush(QColor(220, 240, 255)))
                srk_table.setItem(avg_row, c, item)

        layout.addWidget(srk_table)

        # --- Raw points table ---
        all_points: List[Dict] = []
        for r in results:
            all_points.extend(r.get("points", []))
        if all_points:
            if is_triode:
                pts_table = QTableWidget(len(all_points), 3)
                pts_table.setHorizontalHeaderLabels([
                    t("srk.col_ua_v"),
                    t("srk.col_ug1_v"),
                    t("srk.col_ia_ma"),
                ])
            else:
                pts_table = QTableWidget(len(all_points), 4)
                pts_table.setHorizontalHeaderLabels([
                    t("srk.col_ua_v"),
                    t("srk.col_ug1_v"),
                    t("srk.col_ug2_v"),
                    t("srk.col_ia_ma"),
                ])
            pts_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            pts_table.setEditTriggers(QTableWidget.NoEditTriggers)
            for row, p in enumerate(all_points):
                pts_table.setItem(row, 0, QTableWidgetItem(f"{p['ua']:.1f}"))
                pts_table.setItem(row, 1, QTableWidgetItem(f"{p['ug1']:.2f}"))
                if is_triode:
                    pts_table.setItem(row, 2, QTableWidgetItem(f"{p['ia']:.3f}"))
                else:
                    pts_table.setItem(row, 2, QTableWidgetItem(f"{p['ug2']:.1f}"))
                    pts_table.setItem(row, 3, QTableWidgetItem(f"{p['ia']:.3f}"))
            layout.addWidget(pts_table)

        # --- Buttons ---
        def copy_to_clipboard():
            lines = [f"{t('srk.col_index')}\t{t('common.S')}\t{t('common.R')}\t{t('common.K')}"]
            for i, r in enumerate(results):
                if r["s"] is not None:
                    lines.append(f"{i+1}\t{r['s']:.2f}\t{r['r']:.2f}\t{r['k']:.2f}")
                else:
                    lines.append(f"{i+1}\t—\t—\t—")
            if s_avg is not None:
                lines.append(f"{t('srk.avg')}\t{s_avg:.2f}\t{r_avg:.2f}\t{k_avg:.2f}")
            lines.append("")
            if is_triode:
                lines.append(f"{t('common.Ua')}\t{t('common.Ug1')}\t{t('common.Ia')}")
                for p in all_points:
                    lines.append(f"{p['ua']:.1f}\t{p['ug1']:.2f}\t{p['ia']:.3f}")
            else:
                lines.append(f"{t('common.Ua')}\t{t('common.Ug1')}\t{t('common.Ug2')}\t{t('common.Ia')}")
                for p in all_points:
                    lines.append(f"{p['ua']:.1f}\t{p['ug1']:.2f}\t{p['ug2']:.1f}\t{p['ia']:.3f}")
            QGuiApplication.clipboard().setText("\n".join(lines))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        copy_btn = buttons.addButton(t('srk.Copy'), QDialogButtonBox.ActionRole)
        copy_btn.clicked.connect(copy_to_clipboard)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class SrkController(QObject):
    """Manages SRK measurement lifecycle and state.

    Communicates with the host (MainWindow) via signals to keep UI concerns separated.
    """

    # UI update signals
    label_changed = Signal(str)
    measure_btn_enabled = Signal(bool)
    show_points_btn_enabled = Signal(bool)
    # Request output reset: (reset_heater, order)
    reset_requested = Signal(bool, object)
    # Measurement is ready to save: (measurement_dict, points_list)
    measurement_ready = Signal(object, object)
    # Request poller active state change
    poller_active = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[SrkWorker] = None
        self.srk_points: List[Dict] = []
        self.srk_results: List[Dict] = []
        self._after_scan = False
        self._pending_points: Optional[List[Dict]] = None
        self._pending_meta: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_settings(zone: Dict, app_config, calibration,
                       ug2_track: bool, ug2_offset: float,
                       is_triode: bool = False,
                       ug1_sweep: bool = False,
                       uh: float = 0.0, ih: float = 0.0) -> SrkSettings:
        """Build SrkSettings from zone values and app config.

        *ug1_sweep*: when True and ``app_config.srk_ug1_step > 0``,
        enables Ug1 mini-scan for higher S precision.
        """
        return SrkSettings(
            ua_min=zone["ua_min"],
            ua_max=zone["ua_max"],
            ug1_min=zone["ug1_min"],
            ug1_max=zone["ug1_max"],
            ug2=zone["ug2"],
            samples=app_config.srk_samples,
            settle_s=app_config.srk_settle_s,
            calibration=calibration,
            ug1_verify_tolerance=app_config.ug1_verify_tolerance,
            verify_retries=app_config.srk_verify_retries,
            ua_tolerance=app_config.srk_ua_tolerance,
            ug2_tolerance=app_config.srk_ug2_tolerance,
            settle_per_volt_s=app_config.srk_settle_per_volt_s,
            settle_base_s=app_config.srk_settle_base_s,
            is_triode=is_triode,
            ug2_track_ua=ug2_track,
            ug2_offset=ug2_offset,
            ug1_step=app_config.srk_ug1_step if ug1_sweep else 0.0,
            uh=uh, ih=ih,
        )

    @staticmethod
    def _average_uncertainty(valid: List[Dict]) -> Dict[str, float]:
        """Mean relative uncertainty across repeats (keys with data only)."""
        out: Dict[str, float] = {}
        for key in ("s", "r", "k"):
            vals = [r["uncertainty"][key] for r in valid
                    if r.get("uncertainty") and r["uncertainty"].get(key)
                    is not None]
            if vals:
                out[key] = sum(vals) / len(vals)
        return out

    @staticmethod
    def format_srk(s, r, k, suffix: str = "", lamp: Optional[LampConfig] = None) -> str:
        """Format SRK label with optional reference values from lamp config."""
        if s is not None and r is not None and k is not None:
            text = t("srk.label_srk", s=f"{s:.2f}", r=f"{r:.2f}", k=f"{k:.2f}")
        else:
            text = t("srk.label_srk_none")
        if lamp and (lamp.s > 0 or lamp.r > 0 or lamp.k > 0):
            rs = f"{lamp.s:.1f}" if lamp.s > 0 else "—"
            rr = f"{lamp.r:.1f}" if lamp.r > 0 else "—"
            rk = f"{lamp.k:.1f}" if lamp.k > 0 else "—"
            text += t("srk.label_srk_delta", s=rs, r=rr, k=rk)
        if suffix:
            text += f"\n{suffix}"
        return text

    # ------------------------------------------------------------------
    # Manual SRK measurement
    # ------------------------------------------------------------------

    def _release_worker(self) -> bool:
        """Detach the previous worker before starting a new one.

        Returns ``True`` when starting is safe: the old thread is stopped
        and its signals disconnected (``BaseWorker.cleanup()`` canon).
        Returns ``False`` when the previous worker is still running or
        would not stop — the reference is RETAINED (a live ``QThread``
        freed by GC aborts the process; the former ``wait(500)``-and-drop
        here ignored the wait result) and no new measurement must start:
        two SrkWorkers would double-command the hardware.
        """
        if self.worker is None:
            return True
        if self.worker.isRunning():
            return False
        if not self.worker.cleanup():
            # stopped→running race or a stuck thread: keep the reference.
            log.warning("SRK worker did not stop — keeping the reference")
            return False
        self.worker = None
        return True

    def measure(self, client: LM19Serial, zone: Dict, app_config, calibration,
                ug2_track: bool, ug2_offset: float, repeats: int,
                is_triode: bool = False,
                ug1_sweep: bool = False,
                uh: float = 0.0, ih: float = 0.0) -> None:
        """Start a manual (standalone) SRK measurement."""
        if not client or not client.is_open():
            QMessageBox.warning(None, t('msg.COM'), t('msg.Connect_first'))
            return
        if not self._release_worker():
            return

        self._after_scan = False
        settings = self.build_settings(zone, app_config, calibration,
                                       ug2_track, ug2_offset,
                                       is_triode, ug1_sweep=ug1_sweep,
                                       uh=uh, ih=ih)

        self.measure_btn_enabled.emit(False)
        self.label_changed.emit(t('srk.Measuring'))

        self.worker = SrkWorker(client, settings, repeats=repeats)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    # ------------------------------------------------------------------
    # SRK measurement after scan
    # ------------------------------------------------------------------

    def measure_after_scan(self, client: LM19Serial, zone: Dict, app_config,
                           calibration,
                           ug2_track: bool, ug2_offset: float, repeats: int,
                           pending_points: List[Dict], pending_meta: Dict,
                           is_triode: bool = False,
                           ug1_sweep: bool = False,
                           uh: float = 0.0, ih: float = 0.0) -> None:
        """Start SRK measurement as part of a scan workflow."""
        self._pending_points = pending_points
        self._pending_meta = pending_meta
        self._after_scan = True

        settings = self.build_settings(zone, app_config, calibration,
                                       ug2_track, ug2_offset,
                                       is_triode, ug1_sweep=ug1_sweep,
                                       uh=uh, ih=ih)

        if not self._release_worker():
            # Previous SRK still owns the hardware (arbiter makes this
            # rare, not impossible). Don't race it — take the same
            # graceful path a failed measurement takes: SRK computed
            # from the scan data, pending points still delivered.
            log.warning("SRK after scan skipped — previous worker busy; "
                        "falling back to scan-data SRK")
            self._on_after_scan_failed("previous SRK worker still running")
            return

        self.measure_btn_enabled.emit(False)
        self.label_changed.emit(t('srk.Measuring_SRK'))

        self.worker = SrkWorker(client, settings, repeats=repeats)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_after_scan_failed)
        self.worker.start()

    def compute_from_scan(self, points: List[Dict], zone: Dict,
                          scan_meta: Dict, lamp: Optional[LampConfig] = None,
                          is_triode: bool = False) -> None:
        """Compute SRK from scan data (no separate measurement)."""
        self.reset_requested.emit(False, ["Ug2", "Ug1", "Ua", "Uh"])
        try:
            z = Zone(
                ua_min=zone["ua_min"],
                ua_max=zone["ua_max"],
                ug1_min=zone["ug1_min"],
                ug1_max=zone["ug1_max"],
                ug2=zone["ug2"],
                is_triode=is_triode,
                ug2_track_ua=zone.get("ug2_track_ua", False),
                ug2_offset=zone.get("ug2_offset", 0),
            )
            n_in_zone = count_in_zone(points, z)
            if n_in_zone == 0:
                s = r = k = None
                self.label_changed.emit(self.format_srk(None, None, None, t('srk.No_points_in_zone'), lamp))
            else:
                s, r, expanded = compute_sr_zone(points, z)
                k = compute_k(s, r)
                suffix = t("srk.pt_suffix", count=n_in_zone)
                if expanded:
                    suffix += " ~"
                self.label_changed.emit(self.format_srk(s, r, k, suffix, lamp))

            measurement = dict(scan_meta)
            measurement["zone"] = {
                "ua_min": z.ua_min,
                "ua_max": z.ua_max,
                "ug1_min": z.ug1_min,
                "ug1_max": z.ug1_max,
            }
            measurement["srk"] = {"s": s, "r": r, "k": k}
            measurement["srk_method"] = "computed"
            measurement["points"] = points
            self.measurement_ready.emit(measurement, points)
        finally:
            self.poller_active.emit(True)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_progress(self, done: int, total: int) -> None:
        self.label_changed.emit(t('srk.Measuring_progress', done=done, total=total))

    def _on_finished(self, results: List[Dict]) -> None:
        valid = [r for r in results if r["s"] is not None]
        if not valid:
            self.label_changed.emit(self.format_srk(None, None, None))
            self.srk_results = results
            self.measure_btn_enabled.emit(True)
            self.show_points_btn_enabled.emit(False)
            self.reset_requested.emit(False, ["Ug2", "Ug1", "Ua"])
            return

        s_avg = sum(r["s"] for r in valid) / len(valid)
        r_avg = sum(r["r"] for r in valid) / len(valid)
        k_avg = sum(r["k"] for r in valid) / len(valid)
        unc_avg = self._average_uncertainty(valid)

        suffix = ""
        if unc_avg:
            suffix = t("srk.label_uncertainty",
                       s=f"{unc_avg.get('s', 0) * 100:.0f}",
                       r=f"{unc_avg.get('r', 0) * 100:.0f}",
                       k=f"{unc_avg.get('k', 0) * 100:.0f}")
        self.label_changed.emit(
            self.format_srk(s_avg, r_avg, k_avg, suffix=suffix))
        self.srk_results = results
        self.srk_points = []
        for r in results:
            self.srk_points.extend(r.get("points", []))
        self.measure_btn_enabled.emit(True)
        self.show_points_btn_enabled.emit(True)
        self.reset_requested.emit(False, ["Ug2", "Ug1", "Ua"])

        if self._after_scan:
            self._after_scan = False
            if self._pending_points:
                try:
                    meta = self._pending_meta or {}
                    measurement = dict(meta)
                    measurement["srk"] = {"s": s_avg, "r": r_avg, "k": k_avg}
                    if unc_avg:
                        # ML-091: persist the measured uncertainty (extra
                        # key — backward compatible, no schema bump).
                        measurement["srk"]["uncertainty"] = unc_avg
                    measurement["srk_method"] = "measured"
                    measurement["srk_results"] = [
                        {"s": r["s"], "r": r["r"], "k": r["k"],
                         "uncertainty": r.get("uncertainty")}
                        for r in results
                    ]
                    measurement["srk_points"] = self.srk_points
                    measurement["points"] = self._pending_points
                    self.measurement_ready.emit(measurement, self._pending_points)
                finally:
                    self._pending_points = None
                    self._pending_meta = None
                    self.poller_active.emit(True)

    def _on_failed(self, msg: str) -> None:
        self.label_changed.emit(self.format_srk(None, None, None))
        self.measure_btn_enabled.emit(True)
        self.reset_requested.emit(False, ["Ug2", "Ug1", "Ua"])
        QMessageBox.warning(None, t('msg.Measure_SRK'), msg)

    def _on_after_scan_failed(self, msg: str) -> None:
        self._after_scan = False
        self.measure_btn_enabled.emit(True)
        self.reset_requested.emit(False, ["Ug2", "Ug1", "Ua"])
        self.poller_active.emit(True)

        # Fallback: compute SRK from scan data
        if self._pending_points:
            meta = self._pending_meta or {}
            zone = meta.get("_zone", {})
            self.compute_from_scan(self._pending_points, zone, meta,
                                   is_triode=zone.get("is_triode", False))
            self._pending_points = None
            self._pending_meta = None

        QMessageBox.warning(None, t('msg.Measure_SRK'),
                            t('srk.SRK_failed', message=msg))

    def show_results_dialog(self, parent=None, is_triode: bool = False) -> None:
        """Show SRK results dialog if results are available."""
        if not self.srk_results:
            return
        dialog = SrkResultsDialog(self.srk_results, parent=parent, is_triode=is_triode)
        dialog.exec()
