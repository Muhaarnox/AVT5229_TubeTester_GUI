"""Export helpers for PDF report, SPICE model, CSV and uTracer .utd."""

import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from app.ui_theme import STYLE_MUTED_SMALL
from app.widget_factory import make_double_spinbox
from lm19.app_config import AppConfig
from lm19.config import LampConfig
from lm19.constants import DEFAULT_UB_V, DEFAULT_UG2_V
from lm19.tube_model_base import MODEL_TYPE_KOREN
from lm19.quality import compute_quality
from i18n_setup import t
from lm19.amplifier.constants import (
    CIRCUIT_CF,
    CIRCUIT_PP,
    CIRCUIT_SE,
    CIRCUIT_SE_XFMR,
)

# ── module local constants ──
_PLOT_EXPORT_WIDTH_PX = 1600  # print-quality plot render width for PDF


def render_plot_pixmap(widget) -> "QPixmap":
    """Render a pyqtgraph PlotWidget at print resolution (WYSIWYG theme).

    Falls back to a screen-resolution ``grab()`` — loudly — when the
    pyqtgraph exporter is unavailable or fails (e.g. a non-plot widget).
    """
    from PySide6.QtGui import QPixmap

    try:
        from pyqtgraph.exporters import ImageExporter

        exporter = ImageExporter(widget.plotItem)
        exporter.parameters()["width"] = _PLOT_EXPORT_WIDTH_PX
        exporter.parameters()["background"] = widget.backgroundBrush().color()
        image = exporter.export(toBytes=True)
        if image is not None and not image.isNull():
            return QPixmap.fromImage(image)
        log.warning("Plot high-res export returned a null image — "
                    "falling back to grab()")
    except Exception:
        log.warning("Plot high-res export failed — falling back to grab()",
                    exc_info=True)
    return widget.grab()


def render_points_pixmap(points: List[Dict], title: str = "") -> "QPixmap":
    """Print-resolution I-V family of ONE measurement's points.

    Used by the Compare separate-PDF path so each lamp's report carries
    its own curves (the shared compare-plot screenshot showed every
    selected lamp in every PDF).
    """
    return render_plot_pixmap(_points_curves_widget(points, title))


def _points_curves_widget(points: List[Dict], title: str = ""):
    """Offscreen pyqtgraph widget with per-(Ug1,Ug2) curves (testable)."""
    import pyqtgraph as pg

    from app.ui_theme import SERIES_PALETTE

    widget = pg.PlotWidget(title=title)
    widget.resize(_PLOT_EXPORT_WIDTH_PX // 2, _PLOT_EXPORT_WIDTH_PX // 3)
    widget.setLabel("bottom", "Ua, V")
    widget.setLabel("left", "Ia, mA")
    curves: Dict = {}
    for p in points:
        key = (round(p.get("ug1", 0.0), 2), round(p.get("ug2", 0.0), 1))
        curves.setdefault(key, []).append(p)
    for i, key in enumerate(sorted(curves)):
        pts = sorted(curves[key], key=lambda p: p.get("ua", 0.0))
        widget.plot(
            [p.get("ua", 0.0) for p in pts],
            [p.get("ia", 0.0) for p in pts],
            pen=SERIES_PALETTE[i % len(SERIES_PALETTE)],
        )
    return widget


def render_group_overlay_pixmap(
    members: List[tuple], title: str = "",
) -> "QPixmap":
    """Print-resolution overlay of a matched GROUP's lamps only.

    ``members`` = [(label, points), …]; one color per lamp (all its
    curves), legend with the labels. The certificate must not show
    other checked lamps' curves — a plain compare-plot screenshot would.
    """
    return render_plot_pixmap(_group_overlay_widget(members, title))


def _group_overlay_widget(members: List[tuple], title: str = ""):
    """Offscreen widget behind render_group_overlay_pixmap (testable)."""
    import pyqtgraph as pg

    from app.ui_theme import SERIES_PALETTE

    widget = pg.PlotWidget(title=title)
    widget.resize(_PLOT_EXPORT_WIDTH_PX // 2, _PLOT_EXPORT_WIDTH_PX // 3)
    widget.setLabel("bottom", "Ua, V")
    widget.setLabel("left", "Ia, mA")
    widget.addLegend()
    for i, (label, points) in enumerate(members):
        color = SERIES_PALETTE[i % len(SERIES_PALETTE)]
        curves: Dict = {}
        for p in points:
            key = (round(p.get("ug1", 0.0), 2), round(p.get("ug2", 0.0), 1))
            curves.setdefault(key, []).append(p)
        first = True
        for key in sorted(curves):
            pts = sorted(curves[key], key=lambda p: p.get("ua", 0.0))
            widget.plot(
                [p.get("ua", 0.0) for p in pts],
                [p.get("ia", 0.0) for p in pts],
                pen=color, name=(str(label) if first else None),
            )
            first = False
    return widget


def export_pdf(
    parent: QWidget,
    points: List[Dict],
    tube_type: str,
    lamp_id: str,
    lamp: Optional[LampConfig],
    srk_results: List[Dict],
    plot_renderer,
    plot_widget,
    transfer_widget=None,
    mfg_date: str = "",
    config: Optional[AppConfig] = None,
    scan_meta: Optional[Dict] = None,
) -> None:
    """Export measurement report as PDF.

    Args:
        parent: parent widget for dialogs.
        points: all measurement points.
        tube_type: lamp type name.
        lamp_id: unique lamp identifier.
        lamp: lamp configuration (optional).
        srk_results: list of SRK measurement dicts.
        plot_renderer: PlotRenderer instance (for analysis data).
        plot_widget: 2D plot widget to render as image.
        transfer_widget: transfer plot widget (optional).
        mfg_date: manufacturing date "YYYY-MM" ("" = unknown).
        config: AppConfig for report defaults; None = silent full export
            (no options dialog — legacy/test path).
        scan_meta: measurement metadata for the scan-settings section.
    """
    from app.report import generate_pdf_report
    from app.report_options_dialog import ReportOptions, ask_report_options

    if not points:
        QMessageBox.warning(parent, t("msg.Export_PDF"), t("msg.Op_no_data"))
        return

    srk_data = _average_srk(srk_results)
    analysis = getattr(plot_renderer, "_load_line_analysis", None)
    available = {
        "nominal": "" if lamp else "report.Na_no_lamp",
        "scan_settings": "" if scan_meta else "report.Na_no_meta",
        "srk": "" if srk_data else "report.Na_no_srk",
        "quality": "" if lamp else "report.Na_no_quality",
        "distortion": "" if analysis else "report.Na_no_analysis",
        "plot_curves": "" if plot_widget is not None else "report.Na_no_image",
        "plot_transfer": ("" if transfer_widget is not None
                          else "report.Na_no_image"),
    }
    if config is None:
        opts = ReportOptions(
            sections={sid for sid, reason in available.items() if not reason},
            language="en")
    else:
        opts = ask_report_options(parent, available, config)
        if opts is None:
            return

    path, _ = QFileDialog.getSaveFileName(
        parent, t("msg.Export_PDF"), "", t("msg.PDF_filter"),
    )
    if not path:
        return
    try:
        quality = compute_quality(points, lamp, srk_data) if lamp else None

        plot_pixmap = (render_plot_pixmap(plot_widget)
                       if "plot_curves" in opts.sections else None)
        transfer_pixmap = (render_plot_pixmap(transfer_widget)
                           if transfer_widget is not None
                           and "plot_transfer" in opts.sections else None)

        generate_pdf_report(
            path=path,
            tube_type=tube_type,
            lamp_id=lamp_id,
            lamp_config=lamp,
            points=points,
            srk=srk_data,
            quality=quality,
            analysis=analysis,
            plot_image=plot_pixmap,
            transfer_image=transfer_pixmap,
            mfg_date=mfg_date,
            sections=opts.sections,
            language=opts.language,
            scan_meta=scan_meta,
        )
        QMessageBox.information(
            parent, t("msg.Export_PDF"), t("msg.PDF_saved", path=path),
        )
    except Exception as exc:
        log.exception("PDF export failed")
        QMessageBox.critical(
            parent, t("msg.Export_PDF"), t("msg.PDF_error", error=str(exc)),
        )


_SPICE_MODEL_CHOICES = [
    ("koren", "Koren"),
    ("dempwolf", "Dempwolf v2"),
    ("reefman", "Reefman (Derk/DerkE)"),
]


class SpiceExportDialog(QDialog):
    """Dialog for SPICE export options: model selection + test schematic."""

    def __init__(self, parent=None, series_labels: Optional[Dict] = None,
                 loaded_models: Optional[Dict] = None):
        """``loaded_models``: sid → model_type of an already-fitted model
        for that series — enables the "export loaded model" checkbox."""
        super().__init__(parent)
        self.setWindowTitle(t("msg.Spice_export"))
        self.setMinimumWidth(320)
        self._series_labels = series_labels or {}
        self._loaded_models = loaded_models or {}

        layout = QVBoxLayout(self)

        # --- Model selection ---
        model_group = QGroupBox(t("msg.Spice_select_model"))
        model_layout = QVBoxLayout(model_group)
        self._model_radios = []
        for i, (key, label) in enumerate(_SPICE_MODEL_CHOICES):
            rb = QRadioButton(label)
            if i == 0:
                rb.setChecked(True)
            rb._model_key = key
            self._model_radios.append(rb)
            model_layout.addWidget(rb)
        self.cb_use_loaded = QCheckBox(t("msg.Spice_use_loaded"))
        self.cb_use_loaded.setToolTip(t("tip.Spice_use_loaded"))
        self.cb_use_loaded.toggled.connect(self._on_use_loaded_toggled)
        model_layout.addWidget(self.cb_use_loaded)
        layout.addWidget(model_group)

        # --- Test schematic checkbox ---
        self.cb_test_schematic = QCheckBox(t("msg.Spice_generate_test_asc"))
        self.cb_test_schematic.setChecked(True)
        self.cb_test_schematic.setToolTip(t("tip.Spice_test_asc"))
        layout.addWidget(self.cb_test_schematic)

        # --- Amplifier schematic ---
        amp_group = QGroupBox(t("msg.Spice_amp_schematic"))
        amp_layout = QVBoxLayout(amp_group)
        self.cb_amp_schematic = QCheckBox(t("msg.Spice_generate_amp_asc"))
        self.cb_amp_schematic.setChecked(False)
        amp_layout.addWidget(self.cb_amp_schematic)

        self.amp_circuit_combo = QComboBox()
        self.amp_circuit_combo.addItem("SE Resistive", CIRCUIT_SE)
        self.amp_circuit_combo.addItem("SE Transformer", CIRCUIT_SE_XFMR)
        self.amp_circuit_combo.addItem("Cathode Follower", CIRCUIT_CF)
        self.amp_circuit_combo.addItem("Push-Pull", CIRCUIT_PP)
        amp_layout.addWidget(self.amp_circuit_combo)

        # Transformer / PP parameters
        xfmr_row = QFormLayout()
        self._rload_label = QLabel("R load:")
        self.rload_combo = QComboBox()
        for ohm in ["4", "8", "16"]:
            self.rload_combo.addItem(f"{ohm} Ω", ohm)
        self.rload_combo.setCurrentIndex(1)  # 8 Ohm default
        xfmr_row.addRow(self._rload_label, self.rload_combo)

        self._flow_label = QLabel("f low:")
        self.flow_spin = make_double_spinbox(
            min_val=5.0, max_val=100.0, value=20.0,
            suffix=" Hz", tooltip_key="tip.Spice_flow",
        )
        xfmr_row.addRow(self._flow_label, self.flow_spin)

        self._xfmr_widgets = [self._rload_label, self.rload_combo,
                              self._flow_label, self.flow_spin]
        amp_layout.addLayout(xfmr_row)

        # Data source selection (always visible when series available)
        source_row = QFormLayout()
        self._source_label = QLabel(t("msg.Spice_source") + ":")
        self.pp_tube_a_combo = QComboBox()
        self.pp_tube_a_combo.setToolTip(t("tip.Spice_pp_tube_a"))
        for sid, label in self._series_labels.items():
            self.pp_tube_a_combo.addItem(label, sid)
        source_row.addRow(self._source_label, self.pp_tube_a_combo)

        # Tube B — PP only
        self._pp_label_b = QLabel("Tube B:")
        self.pp_tube_b_combo = QComboBox()
        self.pp_tube_b_combo.setToolTip(t("tip.Spice_pp_tube_b"))
        for sid, label in self._series_labels.items():
            self.pp_tube_b_combo.addItem(label, sid)
        self.pp_tube_b_combo.insertItem(0, t("msg.Spice_pp_matched"), None)
        self.pp_tube_b_combo.setCurrentIndex(0)
        source_row.addRow(self._pp_label_b, self.pp_tube_b_combo)
        amp_layout.addLayout(source_row)

        # Source combo visible when multiple series; Tube B only for PP
        has_series = len(self._series_labels) > 1
        self._source_widgets = [self._source_label, self.pp_tube_a_combo]
        self._pp_widgets = [self._pp_label_b, self.pp_tube_b_combo]
        if not has_series:
            for w in self._source_widgets:
                w.setVisible(False)
        self.amp_circuit_combo.currentIndexChanged.connect(self._on_circuit_changed)
        self.pp_tube_a_combo.currentIndexChanged.connect(
            self._update_loaded_availability)
        self._on_circuit_changed()
        self._update_loaded_availability()

        layout.addWidget(amp_group)

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def model_type(self) -> str:
        for rb in self._model_radios:
            if rb.isChecked():
                return rb._model_key
        return MODEL_TYPE_KOREN

    @property
    def generate_test_schematic(self) -> bool:
        return self.cb_test_schematic.isChecked()

    @property
    def generate_amp_schematic(self) -> bool:
        return self.cb_amp_schematic.isChecked()

    @property
    def amp_circuit(self) -> str:
        return self.amp_circuit_combo.currentData() or CIRCUIT_SE

    @property
    def r_load(self) -> str:
        return self.rload_combo.currentData() or "8"

    @property
    def f_low(self) -> float:
        return self.flow_spin.value()

    @property
    def pp_tube_a_sid(self) -> Optional[int]:
        return self.pp_tube_a_combo.currentData()

    @property
    def pp_tube_b_sid(self) -> Optional[int]:
        """None = matched pair (same as tube A)."""
        return self.pp_tube_b_combo.currentData()

    def source_sid(self) -> int:
        """Series id of the selected data source (0 = current scan)."""
        sid = self.pp_tube_a_combo.currentData()
        return 0 if sid is None else sid

    @property
    def use_loaded_model(self) -> bool:
        return self.cb_use_loaded.isChecked()

    def _update_loaded_availability(self) -> None:
        """The checkbox lives only where a fitted model actually exists
        for the selected source series. With exactly ONE model overall it
        stays reachable even when the source combo is hidden (models
        attach to overlay series, never to the current scan's sid 0)."""
        model_type = self._loaded_models.get(self.source_sid())
        if (model_type is None and len(self._loaded_models) == 1
                and len(self._series_labels) <= 1):
            # source combo hidden → no explicit choice was possible; the
            # single fitted model is unambiguous. With a visible combo a
            # modelless source keeps the checkbox off (no silent swap).
            model_type = next(iter(self._loaded_models.values()))
        self.cb_use_loaded.setEnabled(model_type is not None)
        if model_type is None:
            self.cb_use_loaded.setChecked(False)

    def _on_use_loaded_toggled(self, checked: bool) -> None:
        # exporting as-is → the fitter choice is moot; grey it out
        for rb in self._model_radios:
            rb.setEnabled(not checked)

    def _on_circuit_changed(self) -> None:
        circuit = self.amp_circuit_combo.currentData()
        is_pp = circuit == CIRCUIT_PP
        has_xfmr = circuit in (CIRCUIT_SE_XFMR, CIRCUIT_PP)
        for w in self._pp_widgets:
            w.setVisible(is_pp)
        for w in self._xfmr_widgets:
            w.setVisible(has_xfmr)
        # Source label: "Tube A:" for PP, "Source:" for others
        self._source_label.setText(
            "Tube A:" if is_pp else t("msg.Spice_source") + ":"
        )


def export_spice(
    parent: QWidget,
    points: List[Dict],
    tube_type: str,
    plot_mgr=None,
    topology: Optional[str] = None,
    amp_params: Optional[Dict] = None,
    all_points: Optional[List[Dict]] = None,
    series_labels: Optional[Dict] = None,
    mfg_date: str = "",
    series_models: Optional[Dict] = None,
) -> None:
    """Export SPICE model fitted to measured data.

    Args:
        parent: parent widget for dialogs.
        points: current scan points (without series_id).
        tube_type: lamp type name.
        plot_mgr: PlotManager instance — for setting model overlay after fit.
        topology: force topology ("triode"/"pentode") or None for auto.
        amp_params: optional amplifier parameters from UI (ub, ra, rk, ug2, etc.)
    """
    from lm19.spice_export import export_spice_from_model, fit_and_export_spice

    if not points:
        QMessageBox.warning(parent, t("msg.Spice_export"), t("msg.Spice_no_data"))
        return

    loaded_types = {sid: getattr(m, "model_type", "?")
                    for sid, m in (series_models or {}).items()}
    dlg = SpiceExportDialog(parent, series_labels=series_labels,
                            loaded_models=loaded_types)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return

    model_type = dlg.model_type
    want_test_asc = dlg.generate_test_schematic
    want_amp_asc = dlg.generate_amp_schematic
    amp_circuit = dlg.amp_circuit
    r_load = dlg.r_load
    f_low = dlg.f_low

    default_name = tube_type.replace(" ", "_")
    path, _ = QFileDialog.getSaveFileName(
        parent, t("msg.Spice_export"), default_name, t("msg.Spice_filter"),
    )
    if not path:
        return

    # Use filename (without extension) as subcircuit name.
    # This lets the user name tubes individually (e.g. "12AU7_tube1.sub")
    # so two identical tubes get distinct .SUBCKT names.
    from pathlib import Path as _Path
    spice_name = _Path(path).stem

    # Select points by series_id from dialog (for any circuit type)
    pp_tube_a_sid = dlg.pp_tube_a_sid
    pp_tube_b_sid = dlg.pp_tube_b_sid
    if pp_tube_a_sid is not None and all_points:
        filtered = [p for p in all_points if p.get("series_id") == pp_tube_a_sid]
        if filtered:
            points = filtered
        else:
            # ML-081: the selected series has NO points — fitting the
            # full unfiltered set silently would produce a model of the
            # wrong tube. Abort visibly instead.
            log.warning("SPICE export aborted — selected tube A series "
                        "(sid=%s) has no points", pp_tube_a_sid)
            QMessageBox.warning(
                parent, t("msg.Spice_export"),
                t("msg.Spice_series_empty", sid=str(pp_tube_a_sid)))
            return

    loaded_model = None
    if dlg.use_loaded_model:
        loaded_model = (series_models or {}).get(dlg.source_sid())
        if (loaded_model is None and len(series_models or {}) == 1
                and len(series_labels or {}) <= 1):
            # single fitted model, source combo hidden → it is the one
            loaded_model = next(iter(series_models.values()))

    try:
        if loaded_model is not None:
            # as-is export: the exact model the plots/analysis use
            result = export_spice_from_model(
                path, loaded_model, tube_type=spice_name,
                mfg_date=mfg_date)
        else:
            result = fit_and_export_spice(
                path, spice_name, points, topology=topology,
                model_type=model_type, mfg_date=mfg_date)

        # Generate test schematic if requested
        asc_path = None
        if want_test_asc:
            from lm19.ltspice_asc import generate_test_schematic
            asc_path = generate_test_schematic(
                path, spice_name, points, result.model_type)

        topo_label = t("msg.Spice_model_" + result.model_type)
        algo_label = result.algorithm.capitalize()
        if loaded_model is not None:
            # exported as-is: there are no fresh fit statistics to show
            msg = t("msg.Spice_loaded_ok",
                    path=path, model=topo_label, algorithm=algo_label)
        else:
            msg = t("msg.Spice_fit_ok",
                    path=path, model=topo_label, algorithm=algo_label,
                    rms=f"{result.rms_error:.2f}",
                    max=f"{result.max_error:.2f}",
                    n=str(result.n_points))
        # ML-111: degraded exports must not look like clean ones —
        # append the fitter's warning codes as ⚠ lines.
        for code in getattr(result, "warnings", []):
            msg += "\n⚠ " + t(f"msg.Spice_warn_{code}")
        if asc_path:
            msg += "\n" + t("msg.Spice_test_asc_saved", path=asc_path)

        # Generate amplifier schematic if requested
        amp_asc_path = None
        if want_amp_asc:
            from lm19.ltspice_asc import generate_amp_schematic
            ap = amp_params or {}
            # For PP mismatched: pass tube B name and sub file
            tube_b_name = ""
            sub_b_file = ""
            if amp_circuit == CIRCUIT_PP and pp_tube_b_sid is not None:
                tube_b_name = spice_name + "_B"
                sub_b_file = spice_name + "_B.sub"

            amp_asc_path = generate_amp_schematic(
                path, spice_name, points, result.model_type,
                circuit=amp_circuit,
                ub=ap.get("ub", DEFAULT_UB_V),
                ra_ohm=ap.get("ra_ohm", ""),
                rk_ohm=ap.get("rk_ohm", ""),
                ra_dc_ohm=ap.get("ra_dc_ohm", ""),
                ug2=ap.get("ug2", DEFAULT_UG2_V),
                ra_aa_ohm=ap.get("ra_aa_ohm", ""),
                r_load=r_load,
                f_low=f_low,
                tube_name_b=tube_b_name,
                sub_file_b=sub_b_file,
            )
        if amp_asc_path:
            msg += "\n" + t("msg.Spice_amp_asc_saved", path=amp_asc_path)

        # PP mismatched: export tube B as separate .sub
        if amp_circuit == CIRCUIT_PP and pp_tube_b_sid is not None and all_points:
            points_b = [p for p in all_points if p.get("series_id") == pp_tube_b_sid]
            if not points_b:
                # ML-082: the .asc generated above references
                # <name>_B.sub — without points the file is never
                # written and the schematic ships with a dead include.
                log.warning("PP mismatched: tube B series (sid=%s) has no "
                            "points — %s_B.sub NOT created, the amp "
                            "schematic references a missing file",
                            pp_tube_b_sid, spice_name)
                msg += "\n⚠ " + t("msg.Spice_pp_tube_b_missing",
                                  name=spice_name + "_B.sub")
            if points_b:
                path_b = str(_Path(path).with_stem(spice_name + "_B"))
                name_b = spice_name + "_B"
                # the checkbox covers BOTH tubes: an as-is tube A with a
                # refitted tube B would silently mismatch the pair
                model_b = ((series_models or {}).get(pp_tube_b_sid)
                           if dlg.use_loaded_model else None)
                if model_b is not None:
                    export_spice_from_model(
                        path_b, model_b, tube_type=name_b,
                        mfg_date=mfg_date)
                    msg += "\n" + t("msg.Spice_pp_tube_b_saved_loaded",
                                    path=path_b)
                else:
                    result_b = fit_and_export_spice(
                        path_b, name_b, points_b, topology=topology,
                        model_type=model_type, mfg_date=mfg_date,
                    )
                    msg += "\n" + t("msg.Spice_pp_tube_b_saved", path=path_b,
                                    rms=f"{result_b.rms_error:.2f}")

        QMessageBox.information(parent, t("msg.Spice_export"), msg)
    except Exception as exc:
        log.exception("SPICE export failed")
        QMessageBox.critical(
            parent, t("msg.Spice_export"), t("msg.Spice_error", error=str(exc)),
        )


def _average_srk(srk_results: List[Dict]) -> Optional[Dict]:
    """Compute average S, R, K from a list of measurement results."""
    valid = [r for r in srk_results if r.get("s") is not None]
    if not valid:
        return None
    return {
        "s": sum(r["s"] for r in valid) / len(valid),
        "r": sum(r["r"] for r in valid) / len(valid),
        "k": sum(r["k"] for r in valid) / len(valid),
    }


# ======================================================================
# CSV export
# ======================================================================

class CsvOptionsDialog(QDialog):
    """Dialog for CSV export options."""

    def __init__(self, parent=None, *, multi: bool = False):
        super().__init__(parent)
        self.setWindowTitle(t("csv.Export_CSV"))
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)

        # --- Format ---
        fmt_group = QGroupBox(t("csv.Format"))
        fmt_layout = QVBoxLayout(fmt_group)
        self.rb_flat = QRadioButton(t("csv.Flat_table"))
        self.rb_matrix = QRadioButton(t("csv.Matrix"))
        self.rb_flat.setChecked(True)
        self.rb_flat.setToolTip(t("tip.CSV_flat"))
        self.rb_matrix.setToolTip(t("tip.CSV_matrix"))
        fmt_layout.addWidget(self.rb_flat)
        fmt_layout.addWidget(self.rb_matrix)
        layout.addWidget(fmt_group)

        # Matrix parameter selector (only visible for matrix)
        self.matrix_param_label = QLabel(t("csv.Matrix_param"))
        self.matrix_param_combo = QComboBox()
        self.matrix_param_combo.addItems(["Ia", "Ig2", "Pa"])
        self.matrix_param_label.setVisible(False)
        self.matrix_param_combo.setVisible(False)
        self.rb_matrix.toggled.connect(self._on_format_changed)
        fmt_layout.addWidget(self.matrix_param_label)
        fmt_layout.addWidget(self.matrix_param_combo)

        # --- Separator ---
        sep_group = QGroupBox(t("csv.Separator"))
        sep_layout = QVBoxLayout(sep_group)
        self.rb_semi = QRadioButton(t("csv.Semicolon"))
        self.rb_comma = QRadioButton(t("csv.Comma"))
        self.rb_tab = QRadioButton(t("csv.Tab"))
        self.rb_semi.setChecked(True)
        sep_layout.addWidget(self.rb_semi)
        sep_layout.addWidget(self.rb_comma)
        sep_layout.addWidget(self.rb_tab)
        layout.addWidget(sep_group)

        # --- Options ---
        self.cb_computed = QCheckBox(t("csv.Include_computed"))
        self.cb_computed.setChecked(True)
        self.cb_computed.setToolTip(t("tip.CSV_computed"))
        layout.addWidget(self.cb_computed)

        # Computed columns are only for flat format
        self.rb_matrix.toggled.connect(
            lambda checked: self.cb_computed.setEnabled(not checked)
        )

        # --- Multi-mode options ---
        if multi:
            multi_group = QGroupBox(t("csv.Multi_mode"))
            multi_layout = QVBoxLayout(multi_group)
            self.rb_single_file = QRadioButton(t("csv.All_in_one"))
            self.rb_separate = QRadioButton(t("csv.Separate_files"))
            self.rb_single_file.setChecked(True)
            multi_layout.addWidget(self.rb_single_file)
            multi_layout.addWidget(self.rb_separate)
            layout.addWidget(multi_group)
        else:
            self.rb_single_file = None
            self.rb_separate = None

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_format_changed(self, matrix_checked: bool) -> None:
        self.matrix_param_label.setVisible(matrix_checked)
        self.matrix_param_combo.setVisible(matrix_checked)

    @property
    def separator(self) -> str:
        if self.rb_comma.isChecked():
            return ","
        if self.rb_tab.isChecked():
            return "\t"
        return ";"

    @property
    def is_matrix(self) -> bool:
        return self.rb_matrix.isChecked()

    @property
    def matrix_parameter(self) -> str:
        return self.matrix_param_combo.currentText()

    @property
    def include_computed(self) -> bool:
        return self.cb_computed.isChecked()

    @property
    def separate_files(self) -> bool:
        if self.rb_separate is not None:
            return self.rb_separate.isChecked()
        return False


def _csv_file_filter(sep: str) -> str:
    """Return file dialog filter appropriate for separator."""
    if sep == "\t":
        return t("csv.TSV_filter")
    return t("csv.CSV_filter")


def _csv_default_ext(sep: str) -> str:
    return ".tsv" if sep == "\t" else ".csv"


def export_csv(
    parent: QWidget,
    points: List[Dict],
    tube_type: str,
    lamp_id: str = "",
    name: str = "",
    timestamp: str = "",
    mfg_date: str = "",
    srk: Optional[Dict] = None,
    scan_info: str = "",
    is_triode: bool = False,
) -> None:
    """Show CSV options dialog and export single measurement."""
    from lm19.csv_export import format_csv, format_matrix, write_csv

    if not points:
        QMessageBox.warning(parent, t("csv.Export_CSV"), t("msg.Op_no_data"))
        return

    dlg = CsvOptionsDialog(parent, multi=False)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return

    sep = dlg.separator
    filt = _csv_file_filter(sep)
    path, _ = QFileDialog.getSaveFileName(parent, t("csv.Export_CSV"), "", filt)
    if not path:
        return

    # Ensure proper extension
    ext = _csv_default_ext(sep)
    if not path.lower().endswith(ext):
        path += ext

    try:
        if dlg.is_matrix:
            content = format_matrix(
                points,
                tube_type=tube_type,
                lamp_id=lamp_id,
                timestamp=timestamp,
                mfg_date=mfg_date,
                separator=sep,
                parameter=dlg.matrix_parameter,
                is_triode=is_triode,
            )
        else:
            content = format_csv(
                points,
                tube_type=tube_type,
                lamp_id=lamp_id,
                name=name,
                timestamp=timestamp,
                mfg_date=mfg_date,
                srk=srk,
                scan_info=scan_info,
                separator=sep,
                include_computed=dlg.include_computed,
                is_triode=is_triode,
            )
        write_csv(path, content)
        QMessageBox.information(
            parent, t("csv.Export_CSV"), t("csv.CSV_saved", path=path),
        )
    except Exception as exc:
        log.exception("CSV export failed")
        QMessageBox.critical(
            parent, t("csv.Export_CSV"), t("csv.CSV_error", error=str(exc)),
        )


def export_csv_multi(
    parent: QWidget,
    entries: List[Dict],
    is_triode: bool = False,
) -> None:
    """Show CSV options dialog and export multiple measurements.

    Args:
        parent: parent widget.
        entries: list of dicts with lamp_type, lamp_id, name, timestamp, points.
        is_triode: omit Ug2/Ig2 columns for true triodes.
    """
    from lm19.csv_export import format_csv, format_matrix, format_multi_csv, write_csv
    import os

    if not entries:
        QMessageBox.warning(parent, t("csv.Export_CSV"), t("msg.Op_no_data"))
        return

    dlg = CsvOptionsDialog(parent, multi=True)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return

    sep = dlg.separator
    ext = _csv_default_ext(sep)

    try:
        if dlg.separate_files:
            # Export each entry as individual file — pick a folder
            folder = QFileDialog.getExistingDirectory(
                parent, t("csv.Select_folder"),
            )
            if not folder:
                return

            count = 0
            for entry in entries:
                pts = entry.get("points", [])
                if not pts:
                    continue
                lt = entry.get("lamp_type", "unknown")
                lid = entry.get("lamp_id", "")
                nm = entry.get("name", "")
                ts = entry.get("timestamp", "")
                data = entry.get("data")
                mfg = str(
                    entry.get("mfg_date")
                    or (data.get("mfg_date") if isinstance(data, dict) else "")
                    or ""
                )
                srk = None
                if isinstance(data, dict):
                    srk = data.get("srk")

                safe = "".join(
                    c if c.isalnum() or c in "_-." else "_"
                    for c in f"{lt}_{lid}_{nm}"
                ).strip("_")
                fname = (safe or "measurement") + ext
                fpath = os.path.join(folder, fname)
                # Avoid overwrite
                if os.path.exists(fpath):
                    base, e = os.path.splitext(fpath)
                    n = 1
                    while os.path.exists(fpath):
                        fpath = f"{base}_{n}{e}"
                        n += 1

                if dlg.is_matrix:
                    content = format_matrix(
                        pts, tube_type=lt, lamp_id=lid, timestamp=ts,
                        mfg_date=mfg,
                        separator=sep, parameter=dlg.matrix_parameter,
                        is_triode=is_triode,
                    )
                else:
                    content = format_csv(
                        pts, tube_type=lt, lamp_id=lid, name=nm,
                        timestamp=ts, mfg_date=mfg, srk=srk, separator=sep,
                        include_computed=dlg.include_computed,
                        is_triode=is_triode,
                    )
                write_csv(fpath, content)
                count += 1

            QMessageBox.information(
                parent, t("csv.Export_CSV"),
                t("csv.CSV_multi_saved", count=count, path=folder),
            )
        else:
            # All in one file
            filt = _csv_file_filter(sep)
            path, _ = QFileDialog.getSaveFileName(
                parent, t("csv.Export_CSV"), "", filt,
            )
            if not path:
                return
            if not path.lower().endswith(ext):
                path += ext

            if dlg.is_matrix:
                # Matrix: concatenate blocks per entry
                parts = []
                for entry in entries:
                    pts = entry.get("points", [])
                    if not pts:
                        continue
                    lt = entry.get("lamp_type", "")
                    lid = entry.get("lamp_id", "")
                    ts = entry.get("timestamp", "")
                    d = entry.get("data")
                    mfg = str(
                        entry.get("mfg_date")
                        or (d.get("mfg_date") if isinstance(d, dict) else "")
                        or ""
                    )
                    parts.append(format_matrix(
                        pts, tube_type=lt, lamp_id=lid, timestamp=ts,
                        mfg_date=mfg,
                        separator=sep, parameter=dlg.matrix_parameter,
                        is_triode=is_triode,
                    ))
                content = "\n".join(parts)
            else:
                content = format_multi_csv(
                    entries, separator=sep,
                    include_computed=dlg.include_computed,
                    is_triode=is_triode,
                )
            write_csv(path, content)
            QMessageBox.information(
                parent, t("csv.Export_CSV"), t("csv.CSV_saved", path=path),
            )
    except Exception as exc:
        log.exception("CSV multi-export failed")
        QMessageBox.critical(
            parent, t("csv.Export_CSV"), t("csv.CSV_error", error=str(exc)),
        )


# ======================================================================
# uTracer .utd export
# ======================================================================

class UtdExportDialog(QDialog):
    """Dialog for uTracer .utd export options."""

    def __init__(self, parent=None, *, n_ua: int = 0, n_ug1: int = 0):
        super().__init__(parent)
        self.setWindowTitle(t("utd.Export_title"))
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)

        # --- Format ---
        fmt_group = QGroupBox(t("utd.Format"))
        fmt_layout = QVBoxLayout(fmt_group)
        self.rb_output = QRadioButton(t("utd.Output_curves"))
        self.rb_transfer = QRadioButton(t("utd.Transfer_curves"))
        self.rb_output.setChecked(True)
        self.rb_output.setToolTip(t("tip.UTD_output"))
        self.rb_transfer.setToolTip(t("tip.UTD_transfer"))
        fmt_layout.addWidget(self.rb_output)
        fmt_layout.addWidget(self.rb_transfer)

        self.matrix_info = QLabel()
        self._n_ua = n_ua
        self._n_ug1 = n_ug1
        self._update_matrix_info()
        fmt_layout.addWidget(self.matrix_info)
        self.rb_output.toggled.connect(lambda _: self._update_matrix_info())
        layout.addWidget(fmt_group)

        # --- Warning ---
        warn = QLabel(t("utd.Vs_Vh_warning"))
        warn.setWordWrap(True)
        warn.setStyleSheet(STYLE_MUTED_SMALL)
        layout.addWidget(warn)

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_matrix_info(self) -> None:
        if self.rb_output.isChecked():
            rows, cols = self._n_ua, self._n_ug1
            self.matrix_info.setText(
                t("utd.Matrix_info", rows=rows, cols=cols,
                  x="Va", step="Vg"))
        else:
            rows, cols = self._n_ug1, self._n_ua
            self.matrix_info.setText(
                t("utd.Matrix_info", rows=rows, cols=cols,
                  x="Vg", step="Va"))

    @property
    def fmt(self) -> str:
        return "output" if self.rb_output.isChecked() else "transfer"


def export_utd(
    parent: QWidget,
    points: List[Dict],
    tube_type: str,
) -> None:
    """Show UTD options dialog and export measurement as .utd file."""
    from lm19.utracer_export import format_utd, suggest_filename

    if not points:
        QMessageBox.warning(parent, t("utd.Export_title"), t("msg.Op_no_data"))
        return

    n_ua = len(set(round(p.get("ua", 0.0), 1) for p in points))
    n_ug1 = len(set(round(p.get("ug1", 0.0), 2) for p in points))

    dlg = UtdExportDialog(parent, n_ua=n_ua, n_ug1=n_ug1)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return

    default_name = suggest_filename(tube_type, points)
    path, _ = QFileDialog.getSaveFileName(
        parent, t("utd.Export_title"), default_name,
        t("msg.UTD_filter"),
    )
    if not path:
        return
    if not path.lower().endswith(".utd"):
        path += ".utd"

    try:
        utd_stats: Dict[str, int] = {}
        content = format_utd(points, fmt=dlg.fmt, stats=utd_stats)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        msg = t("utd.Export_saved", path=path)
        holes = utd_stats.get("utd_matrix_holes", 0)
        if holes:
            # ML-117: fabricated Ia=0.0 cells must be visible to the user.
            msg += "\n⚠ " + t("utd.Export_holes", count=str(holes))
        QMessageBox.information(parent, t("utd.Export_title"), msg)
    except Exception as exc:
        log.exception("UTD export failed")
        QMessageBox.critical(
            parent, t("utd.Export_title"),
            t("utd.Export_error", error=str(exc)),
        )
