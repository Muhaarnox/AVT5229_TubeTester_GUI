"""ModelDialog — add a model curve to the plot.

Supports two modes:
  - Reference: load a model from tube_params.json by name
  - Fit: fit a model to the current measured data

Grid settings control the Ua/Ug1/Ug2 ranges for the generated scan.
Grid source: "From data" derives ranges from series points,
"From scan settings" uses current scan panel values.
Spinboxes are always editable for manual correction.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Signal

from app.workers import BaseWorker
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ui_theme import (
    COLOR_GREEN, COLOR_ORANGE, COLOR_RED, MARGIN, MODEL_COMPARE_BEST_BG,
)


# Fit-quality verdict → label color (matches lm19.tube_model_base thresholds)
_QUALITY_COLORS = {
    "good": COLOR_GREEN,
    "fair": COLOR_ORANGE,
    "poor": COLOR_RED,
    "unknown": COLOR_ORANGE,
}
from app.widget_factory import make_double_spinbox
from i18n_setup import t
from lm19.constants import (
    DEFAULT_UG2_V,
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)
from lm19.quality import detect_dead_data, clean_dead_points
from lm19.tube_model_base import (
    MODEL_REGISTRY,
    TubeModelProtocol,
    list_all_tubes,
)
from lm19.tube_sim import ScanGrid
import lm19.dempwolf  # noqa: F401 — register Dempwolf in MODEL_REGISTRY
import lm19.reefman   # noqa: F401 — register Reefman (Derk/DerkE) in MODEL_REGISTRY

log = logging.getLogger(__name__)

_GRID_FROM_DATA = "data"
_GRID_FROM_SCAN = "scan"

# Green highlight for best values in compare table



class _CompareWorker(BaseWorker):
    """Run all fitters in a background thread.

    Inherits :class:`BaseWorker` for unified ``stop()`` /
    ``_stop_requested`` semantics, the ``cleanup()`` helper, and the
    standard error-emit pattern. The ``cancel()`` / ``_cancelled``
    aliases keep the existing UI binding working.
    """

    progress = Signal(int, int, str)   # current, total, model_label
    finished_ok = Signal(list)         # List[CompareRow]

    def __init__(self, points, topology, parent=None):
        # client=None — fitters operate on already-collected points.
        super().__init__(client=None, parent=parent)
        self._points = points
        self._topology = topology

    # ── Aliases for UI bindings ──────────────────────────────────────
    def cancel(self):
        """Alias for :meth:`stop` used by the dialog's Cancel button."""
        self.stop()

    @property
    def _cancelled(self) -> bool:
        return self._stop_requested

    def run(self):
        try:
            self._execute()
        except Exception as exc:
            log.exception("CompareWorker failed")
            if not self._stop_requested:
                # No dedicated error signal here — surface via BaseWorker.failed
                self.failed.emit(str(exc))

    def _execute(self):
        from lm19.model_compare import compare_all_models

        rows = compare_all_models(
            self._points,
            self._topology,
            cancelled=lambda: self._stop_requested,
            on_progress=lambda cur, tot, lbl: self.progress.emit(cur, tot, lbl),
        )
        if not self._stop_requested:
            self.finished_ok.emit(rows)


class ModelDialog(QDialog):
    """Dialog for adding a model curve to the plot."""

    def __init__(
        self,
        parent=None,
        *,
        points: Optional[List[Dict]] = None,
        scan_settings: Optional[Dict] = None,
        is_triode: bool = False,
        series_name: str = "",
        ia_dead_thr: float = 0.30,
    ):
        super().__init__(parent)
        self._points = points or []
        self._scan_settings = scan_settings or {}
        self._is_triode = is_triode
        self._ia_dead_thr = ia_dead_thr
        self._ug2_track = (
            not is_triode and self._scan_settings.get("ug2_track_ua", False)
        )
        self._series_name = series_name
        self._result_model: Optional[TubeModelProtocol] = None
        # ML-087: verdict of the last fit + degraded-fit alerts — read by
        # the parent AFTER exec() (fit_info dies with the dialog).
        self.fit_verdict: str = ""
        self.fit_alerts: List[str] = []
        self._result_grid: Optional[ScanGrid] = None
        self._result_label: str = ""
        self._compare_worker: Optional[_CompareWorker] = None
        self._compare_rows: List = []
        self._compare_checkboxes: List[QCheckBox] = []
        self._multi_result: List[Tuple[TubeModelProtocol, ScanGrid, str]] = []

        self.setWindowTitle(t("model.Title"))
        self.setMinimumSize(620, 520)

        layout = QVBoxLayout(self)
        layout.addLayout(self._build_model_type_row())
        layout.addWidget(self._build_mode_group())
        layout.addWidget(self._build_grid_group())
        self._build_compare_section(layout)
        self._build_dialog_buttons(layout)
        self._connect_and_init()

    def _build_model_type_row(self) -> QHBoxLayout:
        """Build the model type combo row."""
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel(t("model.Model_type")))
        self.model_type_combo = QComboBox()
        self.model_type_combo.setToolTip(t("model.Model_type_tip"))
        for key, entry in MODEL_REGISTRY.items():
            self.model_type_combo.addItem(entry.label, userData=key)
        type_row.addWidget(self.model_type_combo)
        return type_row

    def _build_mode_group(self) -> QGroupBox:
        """Build the Reference/Fit radio group with tube combo."""
        mode_group = QGroupBox()
        mode_layout = QVBoxLayout(mode_group)
        self.ref_radio = QRadioButton(t("model.Reference"))
        self.ref_radio.setToolTip(t("model.Reference_tip"))
        self.fit_radio = QRadioButton(t("model.Fit"))
        self.fit_radio.setToolTip(t("model.Fit_tip"))
        self.fit_radio.setChecked(True)
        btn_group = QButtonGroup(self)
        btn_group.addButton(self.ref_radio)
        btn_group.addButton(self.fit_radio)
        mode_layout.addWidget(self.ref_radio)

        # Reference tube combo
        ref_row = QHBoxLayout()
        self.tube_combo = QComboBox()
        self.tube_combo.setToolTip(t("model.Ref_tube_tip"))
        self.tube_combo.setMinimumWidth(200)
        ref_row.addWidget(self.tube_combo)
        mode_layout.addLayout(ref_row)

        mode_layout.addWidget(self.fit_radio)

        # Fit info label
        self.fit_info = QLabel("")
        self.fit_info.setWordWrap(True)
        mode_layout.addWidget(self.fit_info)

        return mode_group

    def _build_grid_group(self) -> QGroupBox:
        """Build the grid settings group with all spinboxes."""
        grid_group = QGroupBox(t("model.Grid"))
        grid_layout = QFormLayout(grid_group)

        # Grid source selector
        self.grid_source_combo = QComboBox()
        self.grid_source_combo.setToolTip(t("model.Grid_source_tip"))
        if self._points:
            self.grid_source_combo.addItem(
                t("model.Grid_from_data"), userData=_GRID_FROM_DATA,
            )
        self.grid_source_combo.addItem(
            t("model.Grid_from_scan"), userData=_GRID_FROM_SCAN,
        )
        grid_layout.addRow(self.grid_source_combo)

        # Spinboxes
        self.ua_start = QDoubleSpinBox()
        self.ua_stop = QDoubleSpinBox()
        self.ua_step = QDoubleSpinBox()
        self.ug1_start = QDoubleSpinBox()
        self.ug1_stop = QDoubleSpinBox()
        self.ug1_step = QDoubleSpinBox()
        for _sb in (self.ua_start, self.ua_stop, self.ua_step):
            _sb.setToolTip(t("model.Ua_grid_tip"))
        for _sb in (self.ug1_start, self.ug1_stop, self.ug1_step):
            _sb.setToolTip(t("model.Ug1_grid_tip"))

        for sb in (self.ua_start, self.ua_stop):
            sb.setRange(0, 1000)
            sb.setDecimals(0)
            sb.setSuffix(" V")
        self.ua_step.setRange(1, 100)
        self.ua_step.setDecimals(0)
        self.ua_step.setSuffix(" V")

        for sb in (self.ug1_start, self.ug1_stop):
            sb.setRange(-200, 0)
            sb.setDecimals(1)
            sb.setSuffix(" V")
        self.ug1_step.setRange(0.1, 50)
        self.ug1_step.setDecimals(1)
        self.ug1_step.setSuffix(" V")

        ua_row = QHBoxLayout()
        ua_row.addWidget(self.ua_start)
        ua_row.addWidget(QLabel("–"))
        ua_row.addWidget(self.ua_stop)
        ua_row.addWidget(QLabel(t("scan.Step")))
        ua_row.addWidget(self.ua_step)
        grid_layout.addRow("Ua:", ua_row)

        ug1_row = QHBoxLayout()
        ug1_row.addWidget(self.ug1_start)
        ug1_row.addWidget(QLabel("–"))
        ug1_row.addWidget(self.ug1_stop)
        ug1_row.addWidget(QLabel(t("scan.Step")))
        ug1_row.addWidget(self.ug1_step)
        grid_layout.addRow("Ug1:", ug1_row)

        # Ug2 row (pentodes/tetrodes only)
        if self._ug2_track:
            # Triode-connected: single offset spinbox
            self.ug2_offset_spin = make_double_spinbox(
                min_val=-100, max_val=100,
                value=self._scan_settings.get("ug2_offset", 0.0),
                decimals=1, suffix=" V",
                tooltip_key="model.Ug2_offset_tip",
            )
            self._ug2_label = QLabel("Ug2 = Ua +")
            grid_layout.addRow(self._ug2_label, self.ug2_offset_spin)
        else:
            # Pentode: start–stop, step (like Ua/Ug1)
            self.ug2_start_spin = QDoubleSpinBox()
            self.ug2_stop_spin = QDoubleSpinBox()
            self.ug2_step_spin = QDoubleSpinBox()
            for _sb in (self.ug2_start_spin, self.ug2_stop_spin,
                        self.ug2_step_spin):
                _sb.setToolTip(t("model.Ug2_grid_tip"))
            for sb in (self.ug2_start_spin, self.ug2_stop_spin):
                sb.setRange(0, 500)
                sb.setDecimals(0)
                sb.setSuffix(" V")
            self.ug2_step_spin.setRange(1, 200)
            self.ug2_step_spin.setDecimals(0)
            self.ug2_step_spin.setSuffix(" V")
            ug2_row = QHBoxLayout()
            ug2_row.addWidget(self.ug2_start_spin)
            ug2_row.addWidget(QLabel("–"))
            ug2_row.addWidget(self.ug2_stop_spin)
            ug2_row.addWidget(QLabel(t("scan.Step")))
            ug2_row.addWidget(self.ug2_step_spin)
            self._ug2_label = QLabel("Ug2:")
            grid_layout.addRow(self._ug2_label, ug2_row)
        # Hide for true triodes
        if self._is_triode:
            self._ug2_label.setVisible(False)
            if self._ug2_track:
                self.ug2_offset_spin.setVisible(False)
            else:
                for sb in (self.ug2_start_spin, self.ug2_stop_spin,
                           self.ug2_step_spin):
                    sb.setVisible(False)

        return grid_group

    def _build_compare_section(self, layout: QVBoxLayout) -> None:
        """Build the compare button row and results table."""
        compare_row = QHBoxLayout()
        self._compare_btn = QPushButton(t("model.Compare_all"))
        self._compare_btn.setToolTip(t("model.Compare_all_tip"))
        self._compare_btn.setEnabled(bool(self._points))
        self._cancel_btn = QPushButton(t("model.Compare_cancel"))
        self._cancel_btn.setToolTip(t("model.Compare_cancel_tip"))
        self._cancel_btn.setVisible(False)
        self._compare_status = QLabel("")
        compare_row.addWidget(self._compare_btn)
        compare_row.addWidget(self._cancel_btn)
        compare_row.addWidget(self._compare_status)
        compare_row.addStretch()
        layout.addLayout(compare_row)

        self._compare_table = QTableWidget()
        self._compare_table.setVisible(False)
        self._compare_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._compare_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._compare_table.verticalHeader().setVisible(False)
        layout.addWidget(self._compare_table)

        self._add_selected_btn = QPushButton(t("model.Add_selected"))
        self._add_selected_btn.setToolTip(t("model.Add_selected_tip"))
        self._add_selected_btn.setVisible(False)
        self._add_selected_btn.clicked.connect(self._on_add_selected)
        layout.addWidget(self._add_selected_btn)

    def _build_dialog_buttons(self, layout: QVBoxLayout) -> None:
        """Build the OK/Cancel button box."""
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(t("model.Add"))
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _connect_and_init(self) -> None:
        """Connect signals and set initial state."""
        self.model_type_combo.currentIndexChanged.connect(self._on_model_type_changed)
        self.ref_radio.toggled.connect(self._on_mode_changed)
        self.grid_source_combo.currentIndexChanged.connect(self._fill_grid_from_source)
        self._compare_btn.clicked.connect(self._on_compare_all)
        self._cancel_btn.clicked.connect(self._on_compare_cancel)

        self._populate_tubes()
        self._fill_grid_from_source()
        self._on_mode_changed()

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _grid_values_from_data(self) -> Dict:
        """Derive grid values from series points."""
        if not self._points:
            return self._grid_values_from_scan()
        ua_vals = [p["ua"] for p in self._points]
        ug1_vals = [p["ug1"] for p in self._points]
        vals = {
            "ua_start": min(ua_vals),
            "ua_stop": max(ua_vals),
            "ua_step": max(1.0, (max(ua_vals) - min(ua_vals)) / 30),
            "ug1_start": min(ug1_vals),
            "ug1_stop": max(ug1_vals),
            "ug1_step": max(0.5, (max(ug1_vals) - min(ug1_vals)) / 10),
        }
        if self._ug2_track:
            offsets = [p.get("ug2", 0) - p["ua"] for p in self._points
                       if "ug2" in p]
            vals["ug2_offset"] = float(np.median(offsets)) if offsets else 0.0
        else:
            ug2_raw = sorted({p.get("ug2", 0) for p in self._points
                              if "ug2" in p})
            if ug2_raw:
                vals["ug2_start"] = min(ug2_raw)
                vals["ug2_stop"] = max(ug2_raw)
                if len(ug2_raw) >= 2:
                    diffs = [ug2_raw[i+1] - ug2_raw[i]
                             for i in range(len(ug2_raw) - 1)]
                    vals["ug2_step"] = max(1.0, float(np.median(diffs)))
                else:
                    vals["ug2_step"] = 50.0
            else:
                vals["ug2_start"] = DEFAULT_UG2_V
                vals["ug2_stop"] = DEFAULT_UG2_V
                vals["ug2_step"] = 50.0
        return vals

    def _grid_values_from_scan(self) -> Dict:
        """Get grid values from scan settings."""
        s = self._scan_settings
        vals = {
            "ua_start": s.get("ua_start", 0),
            "ua_stop": s.get("ua_stop", 300),
            "ua_step": s.get("ua_step", 10),
            "ug1_start": s.get("ug1_start", -20),
            "ug1_stop": s.get("ug1_stop", 0),
            "ug1_step": s.get("ug1_step", 2),
        }
        if self._ug2_track:
            vals["ug2_offset"] = s.get("ug2_offset", 0.0)
        else:
            vals["ug2_start"] = s.get("ug2_start", DEFAULT_UG2_V)
            vals["ug2_stop"] = s.get("ug2_stop", DEFAULT_UG2_V)
            vals["ug2_step"] = s.get("ug2_step", 50.0)
        return vals

    def _fill_grid_from_source(self) -> None:
        """Fill spinboxes from selected grid source."""
        source = self.grid_source_combo.currentData()
        if source == _GRID_FROM_DATA:
            vals = self._grid_values_from_data()
        else:
            vals = self._grid_values_from_scan()
        self.ua_start.setValue(vals["ua_start"])
        self.ua_stop.setValue(vals["ua_stop"])
        self.ua_step.setValue(vals["ua_step"])
        self.ug1_start.setValue(vals["ug1_start"])
        self.ug1_stop.setValue(vals["ug1_stop"])
        self.ug1_step.setValue(vals["ug1_step"])
        if not self._is_triode:
            if self._ug2_track:
                self.ug2_offset_spin.setValue(vals["ug2_offset"])
            else:
                self.ug2_start_spin.setValue(vals["ug2_start"])
                self.ug2_stop_spin.setValue(vals["ug2_stop"])
                self.ug2_step_spin.setValue(vals["ug2_step"])

    def _build_grid(self) -> ScanGrid:
        """Build ScanGrid from current spinbox values."""
        ua = (self.ua_start.value(), self.ua_stop.value(), self.ua_step.value())
        ug1 = (self.ug1_start.value(), self.ug1_stop.value(), self.ug1_step.value())

        # Ug2 from spinboxes (pentodes/tetrodes only)
        ug2 = None
        ug2_track_ua = False
        ug2_offset = 0.0
        if not self._is_triode:
            if self._ug2_track:
                ug2_track_ua = True
                ug2_offset = self.ug2_offset_spin.value()
            else:
                ug2 = (
                    self.ug2_start_spin.value(),
                    self.ug2_stop_spin.value(),
                    self.ug2_step_spin.value(),
                )

        s = self._scan_settings
        return ScanGrid(
            ua=ua, ug1=ug1, ug2=ug2,
            ug2_track_ua=ug2_track_ua, ug2_offset=ug2_offset,
            uh=s.get("uh", 6.3), ih=s.get("ih", 0.3),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _populate_tubes(self) -> None:
        """Fill tube_combo from current model_type registry."""
        model_type = self.model_type_combo.currentData()
        if model_type is None:
            return
        self.tube_combo.clear()
        try:
            tubes = list_all_tubes(model_type)
            for name in sorted(tubes):
                self.tube_combo.addItem(name)
        except KeyError:
            pass

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_model_type_changed(self) -> None:
        self._populate_tubes()

    def _on_mode_changed(self) -> None:
        is_ref = self.ref_radio.isChecked()
        self.tube_combo.setEnabled(is_ref)
        self.fit_info.setVisible(not is_ref)
        if not is_ref:
            if not self._points:
                self.fit_info.setText(t("model.No_data"))
            elif self._series_name:
                self.fit_info.setText(
                    t("model.Fit_source", name=self._series_name)
                )
            else:
                self.fit_info.setText("")

    def _on_accept(self) -> None:
        """Validate and build result on OK."""
        model_type = self.model_type_combo.currentData()
        if model_type is None:
            self.reject()
            return

        entry = MODEL_REGISTRY.get(model_type)
        if entry is None:
            self.reject()
            return

        grid = self._build_grid()

        if self.ref_radio.isChecked():
            tube_name = self.tube_combo.currentText()
            if not tube_name:
                self.reject()
                return
            model = entry.loader(tube_name)
            if model is None:
                self.fit_info.setVisible(True)
                self.fit_info.setText(t("model.Not_found", name=tube_name))
                return
            label = t("model.Label_ref", name=tube_name)
            self._result_model = model
            self._result_grid = grid
            self._result_label = label
            self.accept()
        else:
            # Fit mode
            if not self._points:
                self.fit_info.setText(t("model.No_data"))
                return
            topology = self._get_topology()

            # Dead-data detection & optional cleanup
            fit_points = self._points
            report = detect_dead_data(
                fit_points, ia_thr=self._ia_dead_thr, topology=topology,
            )
            if report.has_dead_data:
                msg = self._format_dead_data_msg(report)
                reply = QMessageBox.warning(
                    self,
                    t("dead_data.Title"),
                    msg + "\n\n" + t("dead_data.Clean_before_fit"),
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No
                    | QMessageBox.StandardButton.Cancel,
                )
                if reply == QMessageBox.StandardButton.Cancel:
                    return
                if reply == QMessageBox.StandardButton.Yes:
                    fit_points = clean_dead_points(
                        fit_points, report=report,
                    )
                    if not fit_points:
                        self.fit_info.setText(t("model.No_data"))
                        return

            self.fit_info.setText(t("model.Fitting"))
            try:
                fit_result = entry.fitter(fit_points, topology)
            except (ValueError, RuntimeError, KeyError,
                    np.linalg.LinAlgError) as e:
                # ML-086: narrow except — programming errors (Attribute/
                # TypeError from a refactor) propagate with a traceback
                # instead of showing an empty str(e) in fit_info.
                log.exception("Fit failed")
                self.fit_info.setText(str(e) or type(e).__name__)
                return
            self._result_model = fit_result.model
            self._result_grid = grid
            fit_source = self._series_name
            if fit_source:
                fit_desc = f"{fit_source}, {entry.label}"
            else:
                fit_desc = entry.label
            self._result_label = t(
                "model.Label_fit",
                type=fit_desc,
                rms=f"{fit_result.rms_error:.2f}",
            )
            # Format fit info with quality verdict (color) + convergence badge.
            # Quality buckets (compute_fit_quality): good < 2%, fair < 10%,
            # poor >= 10% — surfaces silently-bad fits to the user instead
            # of just a raw RMS number with no context.
            quality = getattr(fit_result, "quality", "unknown")
            rms_pct = getattr(fit_result, "rms_pct", 0.0)
            converged = getattr(fit_result, "converged", True)
            verdict = t(f"model.Quality_{quality}")
            color = _QUALITY_COLORS.get(quality, COLOR_ORANGE)
            badge = "" if converged else f"  ⚠ {t('model.Partial_fit')}"
            text = (
                f"{t('model.Fit_done')}: "
                f"{t('model.RMS')} = {fit_result.rms_error:.2f} mA "
                f"({rms_pct:.1f}% {t('model.of_mean_Ia')}) — {verdict}"
                f"{badge}"
            )
            # Fitter degradation signals (few Ug2 levels, triode-form
            # fallback, knee masking) — failure-visibility rule: the log warning
            # alone never reaches the user.
            for w in getattr(fit_result, "warnings", []):
                code = w.get("code", "")
                params = {k: v for k, v in w.items() if k != "code"}
                line = t(f"model.warn_{code}", **params)
                if line == f"model.warn_{code}":
                    line = code  # missing i18n key must not hide the signal
                text += f"\n⚠ {line}"
            self.fit_info.setText(text)
            self.fit_info.setStyleSheet(f"color: {color};")
            # ML-087: accept() closes the dialog the same instant this
            # label is set — the verdict (and the fitter ⚠ lines) must
            # survive the dialog. The parent shows fit_verdict in the
            # status bar and feeds fit_alerts to the ⚠ indicator.
            self.fit_verdict = text.replace("\n", "  ")
            if quality == "poor" or not converged or getattr(
                    fit_result, "warnings", []):
                self.fit_alerts.append(text)
            self.accept()

    @staticmethod
    def _format_dead_data_msg(report) -> str:
        """Build a human-readable dead-data warning message."""
        lines = [
            t("dead_data.Summary",
              dead=str(report.dead_points),
              total=str(report.total_points),
              pct=f"{report.dead_pct:.0f}"),
        ]
        if report.dead_ug2_levels:
            levels_str = ", ".join(f"{v:.0f} V" for v in report.dead_ug2_levels)
            lines.append(t("dead_data.Dead_levels", levels=levels_str))
        if report.partial_ug2_levels:
            parts = []
            for ug2, ug1 in report.partial_ug2_levels:
                parts.append(f"Ug2={ug2:.0f} V (Ug1>={ug1:.1f} V)")
            lines.append(t("dead_data.Partial_levels", levels=", ".join(parts)))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Compare All
    # ------------------------------------------------------------------

    def _get_topology(self) -> str:
        if self._is_triode:
            return TOPOLOGY_TRIODE
        elif self._ug2_track:
            return TOPOLOGY_TRIODE_CONNECTED
        return TOPOLOGY_PENTODE

    def _on_compare_all(self) -> None:
        """Start comparing all models in a background thread."""
        if not self._points:
            return

        topology = self._get_topology()

        # Dead data cleanup (auto-clean for compare)
        fit_points = self._points
        report = detect_dead_data(
            fit_points, ia_thr=self._ia_dead_thr, topology=topology)
        if report.has_dead_data:
            fit_points = clean_dead_points(fit_points, report=report)
        if not fit_points:
            self._compare_status.setText(t("model.No_data"))
            return

        self._compare_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)
        self._compare_status.setText("")
        self._compare_table.setVisible(False)

        self._compare_worker = _CompareWorker(fit_points, topology, self)
        self._compare_worker.progress.connect(self._on_compare_progress)
        self._compare_worker.finished_ok.connect(self._on_compare_done)
        # ML-088: failed was never connected — a compare error left the
        # status stuck on "Fitting X" with no message at all.
        self._compare_worker.failed.connect(self._on_compare_failed)
        self._compare_worker.finished.connect(self._on_compare_thread_done)
        self._compare_worker.start()

    def _on_compare_cancel(self) -> None:
        if self._compare_worker:
            self._compare_worker.cancel()

    def _on_compare_progress(self, current: int, total: int, label: str):
        if label:
            self._compare_status.setText(
                t("model.Compare_fitting",
                  name=label, n=str(current + 1), total=str(total)))

    def _on_compare_done(self, rows: list) -> None:
        self._compare_rows = rows
        self._fill_compare_table(rows)
        self._compare_table.setVisible(True)
        self._add_selected_btn.setVisible(True)
        self._compare_status.setText(t("model.Compare_done"))

    def _on_compare_failed(self, message: str) -> None:
        self._compare_status.setText(
            t("model.Compare_failed", error=message))
        log.warning("Model compare failed: %s", message)

    def _on_compare_thread_done(self) -> None:
        self._compare_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        # Disconnect signals via cleanup() so receivers aren't held
        # alive across repeated compares.
        if self._compare_worker is not None:
            self._compare_worker.cleanup()
        self._compare_worker = None

    def _stop_compare_worker(self) -> None:
        """Stop a running compare worker before the dialog closes.

        Without this, closing the dialog mid-compare (Esc / window ✕ / a
        button that accepts) leaves a live ``QThread`` child that calls
        ``qFatal`` when freed by GC. ``cleanup()`` stops it and disconnects
        signals; keep the reference if it did not drain.
        """
        if self._compare_worker is not None:
            if self._compare_worker.cleanup():
                self._compare_worker = None

    def done(self, result: int) -> None:
        """QDialog routes accept()/reject()/close through done() — stop the
        compare worker here so every close path drains it."""
        self._stop_compare_worker()
        super().done(result)

    def _fill_compare_table(self, rows: list) -> None:
        """Populate the QTableWidget with CompareRow data."""
        is_pentode = not self._is_triode and not self._ug2_track

        # Column definitions: (key, header, visible)
        cols = [
            ("check",   "",                      True),
            ("model",   t("model.col_Model"),    True),
            ("params",  t("model.col_Params"),   True),
            ("rms_ia",  t("model.col_RMS_Ia"),   True),
            ("max_ia",  t("model.col_Max_Ia"),   True),
            ("rms_ig2", t("model.col_RMS_Ig2"),  is_pentode),
            ("max_ig2", t("model.col_Max_Ig2"),  is_pentode),
            ("rms_gm",  t("model.col_RMS_gm"),   True),
            ("spice",   t("model.col_SPICE"),    True),
            ("status",  t("model.col_Status"),   True),
        ]
        visible_cols = [c for c in cols if c[2]]

        table = self._compare_table
        table.setColumnCount(len(visible_cols))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels([c[1] for c in visible_cols])

        # Checkbox column
        self._compare_checkboxes = []
        check_col = next(i for i, (k, _, _) in enumerate(visible_cols) if k == "check")
        for r, row in enumerate(rows):
            cb = QCheckBox()
            cb.setToolTip(t("model.Compare_check_tip"))
            cb.setEnabled(row.status == "OK" and row.fit_result is not None)
            self._compare_checkboxes.append(cb)
            w = QWidget()
            lay = QHBoxLayout(w)
            lay.addWidget(cb)
            lay.setContentsMargins(MARGIN, 0, 0, 0)
            table.setCellWidget(r, check_col, w)

        def _fmt(val, suffix=""):
            if val is None:
                return "—"
            return f"{val:.2f}{suffix}"

        # Fill cells
        for r, row in enumerate(rows):
            values = {
                "model":   row.label,
                "params":  str(row.n_params) if row.n_params else "—",
                "rms_ia":  _fmt(row.rms_ia),
                "max_ia":  _fmt(row.max_ia),
                "rms_ig2": _fmt(row.rms_ig2),
                "max_ig2": _fmt(row.max_ig2),
                "rms_gm":  _fmt(row.rms_gm),
                "spice":   "✓" if row.spice_support else "",
                "status":  row.status,
            }
            for c, (key, _, vis) in enumerate(visible_cols):
                if key == "check":
                    continue
                item = QTableWidgetItem(values[key])
                table.setItem(r, c, item)

        # Highlight best (minimum) in numeric columns
        numeric_keys = {"rms_ia", "max_ia", "rms_ig2", "max_ig2", "rms_gm"}
        for c, (key, _, _) in enumerate(visible_cols):
            if key not in numeric_keys:
                continue
            # Find min value among OK rows
            best_val = float("inf")
            best_row = -1
            for r, row in enumerate(rows):
                if row.status != "OK":
                    continue
                val = getattr(row, key, None)
                if val is not None and val < best_val:
                    best_val = val
                    best_row = r
            if best_row >= 0:
                item = table.item(best_row, c)
                if item:
                    item.setBackground(QBrush(MODEL_COMPARE_BEST_BG))

        table.resizeColumnsToContents()
        header = table.horizontalHeader()
        header.setStretchLastSection(True)

    # ------------------------------------------------------------------
    # Public result
    # ------------------------------------------------------------------

    def _on_add_selected(self) -> None:
        """Add all checked models from the compare table."""
        grid = self._build_grid()
        selected: List[Tuple[TubeModelProtocol, ScanGrid, str]] = []
        for i, cb in enumerate(self._compare_checkboxes):
            if not cb.isChecked():
                continue
            row = self._compare_rows[i]
            if row.fit_result is None or row.fit_result.model is None:
                continue
            label = t(
                "model.Label_fit",
                type=row.label,
                rms=f"{row.rms_ia:.2f}" if row.rms_ia is not None else "?",
            )
            selected.append((row.fit_result.model, grid, label))
        if not selected:
            return
        self._multi_result = selected
        self.accept()

    def result(self) -> Optional[Tuple[TubeModelProtocol, ScanGrid, str]]:
        """Return (model, grid, label) or None if cancelled."""
        if self._multi_result:
            return self._multi_result[0]
        if self._result_model is None:
            return None
        return (self._result_model, self._result_grid, self._result_label)

    def results_multi(self) -> List[Tuple[TubeModelProtocol, ScanGrid, str]]:
        """Return list of (model, grid, label) for multi-select from compare."""
        if self._multi_result:
            return self._multi_result
        single = self.result()
        return [single] if single else []
