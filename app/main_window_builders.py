"""UI builder mixin for MainWindow.

Holds the 20 ``_build_*`` UI construction methods so the host class
stays focused on lifecycle/wiring and the layout code lives in one
place.

The mixin assumes its host class:
  - is a QMainWindow subclass
  - provides ``self.app_config`` / ``self.calibration`` / ``self.lamps``
    (set in ``MainWindow.__init__`` before ``_build_ui`` runs)
  - exposes the slot methods used by widget signals
    (``_apply_lamp``, ``_run_scan``, ``_stop_scan``, ``_start_preheat``,
    ``_stop_preheat``, ``_check_com``, ``_toggle_connection``, etc.)
  - composes the controller entry points
    (``ConnectionManager``, ``ScanController``, ``PlotManager``,
    ``LampPanel``, ``LivePanel``, ``HealthTab``, ``CompareTab``,
    ``CalibrationTab``, …).

Does not define ``__init__`` — relies on the host class for setup.
Widget refs created here (``self.run_btn``, ``self.zone_ua_min``, …)
are read directly by downstream slots via ``self.<widget>``.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict

import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.amp_control_panel import AmpControlPanel
from app.amplifier_tab import AmplifierTab
from app.calibration_tab import CalibrationTab
from app.checkable_combo import CheckableComboBox
from app.compare_tab import CompareTab
from app.health_tab import HealthTab
from app.lamp_panel import DEFAULT_LAMP_ID, LampPanel
from app.live_panel import LivePanel
from app.manual_tab import ManualTab
from app.plot_manager import (
    PlotManager,
    TRANSFER_VIEW_ALL,
    TRANSFER_VIEW_CUSTOM,
    TRANSFER_VIEW_DATASHEET,
    TRANSFER_VIEW_LOADLINE,
)
from app.plotting import PlotRenderer
from app.ui_theme import (
    AMP_WARNING_HTML_COLOR,
    COLOR_ACCENT_BLUE,
    COLOR_RED,
    MARGIN,
    SPACING_NORMAL,
    STYLE_BOLD_LABEL_SM,
    apply_no_margin,
    apply_tight,
    connection_led_stylesheet,
    io_activity_stylesheet,
)
from app.widget_factory import make_double_spinbox, make_int_spinbox
from i18n_setup import t
from lm19.config import find_lamp
from lm19.constants import DEFAULT_UB_V, TOPOLOGY_PENTODE


# ── module local constants ──
# Cap for the warning-indicator tooltip so a pathological startup does not
# produce a screen-sized tooltip; the click dialog always shows everything.
_WARNBAR_TOOLTIP_MAX_LINES = 12
# Startup log-bridge cap — a corrupt measurements dir warns once per file;
# the indicator keeps the first N and reports how many were suppressed.
_STARTUP_WARNINGS_MAX = 30
# Idle text of the TX/RX byte counters (no link → nothing transferred).
_IO_COUNT_ZERO = "0"


class MainWindowBuilders:
    """Mixin: build the entire MainWindow widget tree.

    All _build_* methods construct widgets and attach them to
    self (e.g. self.run_btn, self.zone_ua_min). The host
    class (MainWindow) calls self._build_ui() from its
    __init__ after collaborators ("app_config, calibration, conn_mgr,
    scan_ctrl, lamp_panel, live_panel") are set up.
    """

    def _build_warning_indicator(self) -> None:
        """Top-status-bar ⚠ indicator (failure-visibility rule: every failure must
        reach the user — the indicator aggregates per-category warning
        lists; empty → hidden). Click shows the full grouped list.

        Creates the widget only; ``_build_top_status_bar`` places it, so
        the window carries no QMainWindow status bar at all."""
        self._ui_warnings: Dict[str, list] = {}
        btn = QToolButton(self)
        btn.setAutoRaise(True)
        btn.setStyleSheet(f"color: {AMP_WARNING_HTML_COLOR}; font-weight: bold;")
        btn.clicked.connect(self._show_ui_warnings)
        btn.hide()
        self.warning_indicator = btn

    def _set_ui_warnings(self, category: str, items: list) -> None:
        """Replace *category*'s warnings; refresh the indicator."""
        if items:
            self._ui_warnings[category] = list(items)
        else:
            self._ui_warnings.pop(category, None)
        total = sum(len(v) for v in self._ui_warnings.values())
        if total == 0:
            self.warning_indicator.hide()
            return
        self.warning_indicator.setText(f"⚠ {total}")
        lines = [line for v in self._ui_warnings.values() for line in v]
        shown = lines[:_WARNBAR_TOOLTIP_MAX_LINES]
        if len(lines) > _WARNBAR_TOOLTIP_MAX_LINES:
            shown.append("…")
        self.warning_indicator.setToolTip("\n".join(shown))
        self.warning_indicator.show()

    def set_startup_warnings(self, messages: list) -> None:
        """Feed WARNING+ log records collected during startup into the
        indicator (capped — a corrupt measurements dir can warn per file)."""
        capped = list(messages[:_STARTUP_WARNINGS_MAX])
        if len(messages) > _STARTUP_WARNINGS_MAX:
            capped.append(
                t("warnbar.More_suppressed",
                  n=len(messages) - _STARTUP_WARNINGS_MAX))
        self._set_ui_warnings("startup", capped)

    def _show_ui_warnings(self) -> None:
        parts = []
        for category, items in self._ui_warnings.items():
            parts.append(t(f"warnbar.category_{category}"))
            parts.extend(f"  ⚠ {line}" for line in items)
        QMessageBox.warning(
            self, t("warnbar.Dialog_title"),
            "\n".join(parts) or t("warnbar.No_warnings"),
        )

    def _build_ui(self) -> None:
        self.top_status_bar = self._build_top_status_bar()
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_measure_tab(), t('tab.Measure'))
        self._init_plot_manager()
        self.tabs.addTab(self._build_manual_tab(), t('tab.Manual'))
        from app.app_context import AppContext

        self._app_ctx = AppContext(
            get_client=lambda: self.client,
            get_write_locked=lambda: self._emergency_lock,
            get_app_config=lambda: self.app_config,
            get_calibration=lambda: self.calibration,
            get_lamps=lambda: self.lamps,
            get_current_tube_type=lambda: self.lamp_combo.currentText() if hasattr(self, "lamp_combo") else "",
            get_current_lamp_id=lambda: self.lamp_panel.lamp_id()
            if hasattr(self, "lamp_panel")
            else DEFAULT_LAMP_ID,
            set_poller_active=self.conn_mgr.set_poller_active,
            get_hw_busy=self._hw_busy_reason,
            get_preheat_enabled=lambda: self.preheat_enabled.isChecked() if hasattr(self, "preheat_enabled") else False,
            get_preheat_done=lambda: bool(getattr(self, "preheat_done", False)),
            request_start_preheat=lambda: self._start_preheat(),
            request_stop_all=lambda: self._reset_outputs_after_scan(
                reset_heater=True, reset_order=["Ug2", "Ug1", "Ua", "Uh", "Ih"]
            ),
            request_stop_keep_heater=lambda: self._reset_outputs_after_scan(
                reset_heater=False, reset_order=["Ug2", "Ug1", "Ua"]
            ),
        )
        self.health_tab = HealthTab(self._app_ctx)
        self.health_tab.on_load_to_manual = self._health_load_to_manual
        self.tabs.addTab(self.health_tab, t('tab.Health'))
        self.compare_tab = CompareTab(
            marker_lock_px=self.app_config.marker_lock_px,
            ug1_cluster_thr=self.app_config.ug1_cluster_threshold,
            ug2_cluster_thr=self.app_config.ug2_cluster_threshold,
            ia_dead_thr=self.app_config.ia_dead_threshold,
            match_algorithm=self.app_config.compare_matching_algorithm,
            get_app_config=lambda: self.app_config,
        )
        self.compare_tab.show_on_main_plot.connect(self._on_compare_show_main_plot)
        self.tabs.addTab(self.compare_tab, t('tab.Compare'))
        self.calibration_tab = CalibrationTab(
            get_client=lambda: self.client,
            get_calibration=lambda: self.calibration,
            set_calibration=lambda cal: setattr(self, 'calibration', cal),
            app_config=self.app_config,
            get_write_locked=lambda: self._emergency_lock,
            get_hw_busy=self._hw_busy_reason,
        )
        self.tabs.addTab(self.calibration_tab, t('tab.Calibration'))
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        root_layout.setSpacing(SPACING_NORMAL)
        root_layout.addWidget(self.top_status_bar)
        root_layout.addWidget(self.tabs, 1)
        self.setCentralWidget(root)

    def _build_top_status_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        self.conn_led = QLabel()
        self.conn_led.setFixedSize(10, 10)
        self.conn_led.setToolTip(t('conn.Connection'))
        self._set_connection_led(COLOR_RED)

        self.status_label = QLabel(t('conn.Disconnected'))
        self.tx_activity_label = QLabel(t('common.TX'))
        self.rx_activity_label = QLabel(t('common.RX'))
        self._set_io_activity(self.tx_activity_label, active=False)
        self._set_io_activity(self.rx_activity_label, active=False)
        # Each byte counter sits next to its own flashing indicator, which
        # already names the direction — the counter itself stays caption-less.
        self.tx_count_label = QLabel(_IO_COUNT_ZERO)
        self.tx_count_label.setToolTip(t('conn.Tx_bytes'))
        self.rx_count_label = QLabel(_IO_COUNT_ZERO)
        self.rx_count_label.setToolTip(t('conn.Rx_bytes'))
        self.reset_all_btn = QPushButton(t('conn.Reset_all'))
        self.reset_all_btn.clicked.connect(self._on_reset_all_clicked)
        self._build_warning_indicator()

        layout.addWidget(self.reset_all_btn)
        layout.addSpacing(self.CONTROL_SPACING)
        layout.addWidget(self.conn_led)
        layout.addWidget(self.status_label)
        layout.addSpacing(self.CONTROL_SPACING)
        layout.addWidget(self.tx_activity_label)
        layout.addWidget(self.tx_count_label)
        layout.addSpacing(self.CONTROL_SPACING)
        layout.addWidget(self.rx_activity_label)
        layout.addWidget(self.rx_count_label)
        layout.addStretch(1)
        layout.addWidget(self.warning_indicator)
        return bar

    def _set_connection_led(self, color: str) -> None:
        self.conn_led.setStyleSheet(connection_led_stylesheet(color))

    def _set_io_activity(self, label: QLabel, active: bool) -> None:
        label.setStyleSheet(io_activity_stylesheet(active))

    def _flash_io_activity(self, tx_changed: bool, rx_changed: bool) -> None:
        if tx_changed:
            self._set_io_activity(self.tx_activity_label, active=True)
            QTimer.singleShot(180, lambda: self._set_io_activity(self.tx_activity_label, active=False))
        if rx_changed:
            self._set_io_activity(self.rx_activity_label, active=True)
            QTimer.singleShot(180, lambda: self._set_io_activity(self.rx_activity_label, active=False))

    # --- Named slots replacing lambdas for SRK signals ---

    def _on_srk_label_changed(self, text: str) -> None:
        self.srk_label.setText(text)

    def _on_srk_measure_btn_enabled(self, enabled: bool) -> None:
        self.measure_srk_btn.setEnabled(enabled)

    def _on_srk_show_points_enabled(self, enabled: bool) -> None:
        self.show_srk_points_btn.setEnabled(enabled)

    # --- Named slots replacing lambdas for plot_mgr render calls ---

    def _rerender_2d(self, *_args) -> None:
        self.plot_mgr.render_2d_only()

    def _rerender_2d_pa(self, *_args) -> None:
        self.plot_mgr.render_2d_and_pa()

    def _rerender_line_width(self, *_args) -> None:
        self.plot_mgr.render_line_width_plots()

    def _on_right_heatmap_changed(self) -> None:
        """Switch right heatmap between Rp and µ via QStackedWidget."""
        mode = self.right_heatmap_combo.currentData()
        idx = 0 if mode == "rp" else 1
        self.right_heatmap_stack.setCurrentIndex(idx)
        self.plot_renderer.set_right_heatmap_mode(mode)

    def _on_heatmap_colormap_changed(self) -> None:
        """Apply selected colormap to all heatmap ImageItems and re-render."""
        name = self.heatmap_cmap_combo.currentText()
        cmap = pg.colormap.get(name)
        if cmap is None:
            return
        lut = cmap.getLookupTable()
        for img in (self.contour_image, self.gm_image, self.rp_image,
                    self.mu_image, self.pa_map_image):
            if img is not None:
                img.setLookupTable(lut)
                img.update()
        # Updates renderer.cmap AND re-luts the colorbars' gradients.
        self.plot_renderer.update_heatmap_colorbars_cmap(cmap)
        # Re-render 2D to apply new cmap to Ug2 color-mode pen colors
        self.plot_mgr.render_2d_only()

    def _on_heatmap_lock_toggled(self, checked: bool) -> None:
        """Freeze/unfreeze the heatmap color scale (capture-on-lock)."""
        self.plot_renderer.set_heatmap_scale_locked(checked)
        if not checked:
            # Back to autoLevels — re-render the maps so the scale is
            # honest immediately, not on the next data change.
            self.plot_mgr.render_slice_plots()

    def _toggle_legend(self) -> None:
        self.plot_mgr.toggle_legend()

    def _clear_all_plots(self) -> None:
        from lm19.analysis import get_available_series
        from app.clear_dialog import ClearSeriesDialog

        pm = self.plot_mgr
        series_info = get_available_series(pm.points, pm.series_labels)
        if not series_info:
            pm.clear_all()
            return

        dlg = ClearSeriesDialog(series_info, parent=self)
        if dlg.exec() != ClearSeriesDialog.DialogCode.Accepted:
            return

        if dlg.is_remove_all:
            pm.clear_all()
        else:
            pm.clear_series(dlg.selected_series_ids)

    def _show_ra_sweep(self) -> None:
        self.plot_mgr.show_ra_sweep(self, self._amp_params_for_line())

    def _amp_params_for_line(self):
        """AmpParams for the live layer/dialogs: panel snapshot + the
        main-UI Ug2 filter (same recipe as Analyze, no dialogs)."""
        params = self.amp_control_panel.params_snapshot()
        ug2_filter = self._get_amp_ug2_filter()
        if ug2_filter is not None:
            params.ug2_filter = ug2_filter
        return params

    def _wire_working_line(self) -> None:
        """Live layer of the working line."""
        from app.working_line import WorkingLineController
        self.working_line = WorkingLineController(
            self.plot, self.amp_engine, self._amp_params_for_line,
            info_label=self.load_line_info,
            renderer=self.plot_renderer,
        )
        self.plot_mgr.working_line_reattach = self.working_line.reattach
        self.amp_control_panel.connect_params_changed(
            self.working_line.schedule)
        self.ug2_calc_combo.currentIndexChanged.connect(
            self.working_line.schedule)
        self.load_line_cb.toggled.connect(self.working_line.set_visible)

    def _show_srk_results(self) -> None:
        self.srk.show_results_dialog(self, is_triode=self._is_triode)

    def _set_write_controls_locked(self, locked: bool) -> None:
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(not locked)
        if hasattr(self, "stop_btn"):
            self.stop_btn.setEnabled(not locked)
        if hasattr(self, "preheat_start_btn"):
            self.preheat_start_btn.setEnabled(not locked)
        if hasattr(self, "preheat_stop_btn"):
            self.preheat_stop_btn.setEnabled(not locked)
        if hasattr(self, "measure_srk_btn"):
            self.measure_srk_btn.setEnabled(not locked)
        if hasattr(self, "save_manual_btn"):
            self.save_manual_btn.setEnabled(not locked)
        if hasattr(self, "manual_tab"):
            write_buttons = [
                "ua_btn", "ug1_btn", "ug2_btn", "uh_btn", "ih_btn",
                "apply_all_btn", "reset_hv_btn", "reset_all_btn", "save_btn",
            ]
            for name in write_buttons:
                btn = getattr(self.manual_tab, name, None)
                if btn is not None:
                    btn.setEnabled(not locked)
        if hasattr(self, "health_tab"):
            for name in ("quick_btn", "stop_btn", "save_type_ref_btn", "set_active_btn"):
                btn = getattr(self.health_tab, name, None)
                if btn is not None:
                    btn.setEnabled(not locked)

    def _init_plot_manager(self) -> None:
        """Create PlotManager with references to the plot-related widgets."""
        self._pig2_max_val = None  # set by _apply_lamp
        self._nominal_s = None

        widgets = {
            "ug2_mode_series": self.ug2_mode_series,
            "ug2_track_radio": self.ug2_track_radio,
            "plot_line_width": self.plot_line_width,
            "overlay_pen_style": self.overlay_pen_style,
            "zone_ua_min": self.zone_ua_min,
            "zone_ua_max": self.zone_ua_max,
            "zone_ug1_min": self.zone_ug1_min,
            "zone_ug1_max": self.zone_ug1_max,
            "pa_max_cb": self.pa_max_cb,
            "pa_max_input": self.pa_max_input,
            "pg2_max_cb": self.pg2_max_cb,
            "pg2_max_input": self.pg2_max_input,
            "ua_max_cb": self.ua_max_cb,
            "ua_max_input": self.ua_max_input,
            "ia_max_limit_cb": self.ia_max_limit_cb,
            "ia_max_limit_input": self.ia_max_limit_input,
            "load_line_info": self.load_line_info,
            "legend_toggle_btn": self.legend_toggle_btn,
            "app_config": self.app_config,
            "pig2_max_val": None,
            "nominal_s": None,
            "srk_data": None,
            "curves_y_combo": self.curves_y_combo,
            "curves_x_combo": self.curves_x_combo,
            "lamp_display_combo": self.lamp_display_combo,
            "lamp_calc_combo": self.lamp_calc_combo,
            "_source_label": self._source_label,
            "_calc_lamp_label": self._calc_lamp_label,
            "ug2_display_combo": self.ug2_display_combo,
            "ug2_calc_combo": self.ug2_calc_combo,
            "_ug2_display_label": self._ug2_display_label,
            "_ug2_calc_label": self._ug2_calc_label,
            "transfer_view_combo": self.transfer_view_combo,
            "transfer_ua_combo": self.transfer_ua_combo,
        }

        # Create AmplifierTab and replace the placeholder
        self.amplifier_tab = AmplifierTab()
        idx = self.plot_tabs.indexOf(self._amplifier_tab_placeholder)
        if idx >= 0:
            self.plot_tabs.removeTab(idx)
            self.plot_tabs.insertTab(idx, self.amplifier_tab, t('tab.Amplifier'))
            self.plot_tabs.setTabToolTip(idx, t('tip.Tab_Amplifier'))
        del self._amplifier_tab_placeholder

        widgets["amplifier_tab"] = self.amplifier_tab
        widgets["amp_control_panel"] = self.amp_control_panel

        # AmplifierEngine: pure computation, no Qt
        from lm19.amp_engine import AmplifierEngine
        self.amp_engine = AmplifierEngine()
        widgets["amp_engine"] = self.amp_engine

        # Wire AmpControlPanel → engine → tab
        self.amp_control_panel.settings_changed.connect(self._on_amp_update)
        self.amp_control_panel.auto_q_requested.connect(self._on_amp_auto_q)
        self.amp_control_panel.ra_optimize_requested.connect(self._on_amp_optimize_ra)
        self.amp_control_panel.optimize_requested.connect(self._on_amp_optimize_full)
        self.amp_control_panel.opt_pareto_btn.toggled.connect(self._on_amp_optimizer_toggled)
        self.amp_control_panel.opt_cancel_btn.clicked.connect(self._on_opt_cancel_clicked)
        self.amp_control_panel.optimizer_apply_best.connect(self._on_amp_apply_best)
        self.amp_control_panel.optimizer_show_top_n.connect(self._on_amp_show_top_n)
        self.amp_control_panel.export_pdf_requested.connect(self._export_amp_pdf)
        self.amp_control_panel.verify_requested.connect(self._on_amp_verify_ltspice)
        self.amp_control_panel.verify_cancel_requested.connect(self._on_amp_verify_cancel)
        self.amplifier_tab.ra_clicked.connect(self._on_amp_ra_clicked)
        self.amplifier_tab.pareto_clicked.connect(self._on_amp_pareto_clicked)
        self.amplifier_tab.set_line_width_spin(self.plot_line_width)

        # Bidirectional sync: plot_options ↔ AmpControlPanel

        self.plot_mgr = PlotManager(self.plot_renderer, widgets)

        self.curves_y_combo.currentIndexChanged.connect(
            lambda: self.plot_mgr.render_curves_only())
        self.curves_x_combo.currentIndexChanged.connect(
            lambda: self.plot_mgr.render_curves_only())
        self.lamp_display_combo.selectionChanged.connect(
            self.plot_mgr.on_display_filter_changed)
        self.lamp_calc_combo.currentIndexChanged.connect(
            lambda: self.plot_mgr.render_slice_plots())
        self.ug2_display_combo.selectionChanged.connect(
            self.plot_mgr.on_ug2_display_changed)
        self.ug2_calc_combo.currentIndexChanged.connect(
            self.plot_mgr.on_ug2_calc_changed)
        self.transfer_ua_combo.selectionChanged.connect(
            self.plot_mgr.on_ua_display_changed)
        self.transfer_view_combo.currentIndexChanged.connect(
            lambda _idx: self.plot_mgr.on_transfer_view_changed())

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu(t('menu.File'))
        reload_config_action = menu.addAction(t('menu.Reload_lamp_config'))
        reload_config_action.triggered.connect(self._load_lamps)
        menu.addSeparator()
        pdf_action = menu.addAction(t('msg.Export_PDF'))
        pdf_action.setToolTip(t('tip.Export_PDF'))
        pdf_action.triggered.connect(self._export_pdf)
        spice_action = menu.addAction(t('msg.Spice_export'))
        spice_action.setToolTip(t('tip.Export_SPICE'))
        spice_action.triggered.connect(self._export_spice)
        csv_action = menu.addAction(t('csv.Export_CSV'))
        csv_action.setToolTip(t('tip.Export_CSV'))
        csv_action.triggered.connect(self._export_csv)
        utd_action = menu.addAction(t('utd.Export_utd'))
        utd_action.setToolTip(t('tip.Export_UTD'))
        utd_action.triggered.connect(self._export_utd)
        menu.addSeparator()
        import_menu = menu.addMenu(t('menu.Import'))
        import_utd = import_menu.addAction(t('menu.Import_uTracer'))
        import_utd.triggered.connect(self.import_ctrl.import_utracer)
        import_csv = import_menu.addAction(t('menu.Import_CSV'))
        import_csv.triggered.connect(self.import_ctrl.import_csv)
        import_ctd = import_menu.addAction(t('menu.Import_CurveTraceData'))
        import_ctd.triggered.connect(self.import_ctrl.import_curvetracedata)
        import_etr = import_menu.addAction(t('menu.Import_eTracer'))
        import_etr.triggered.connect(self.import_ctrl.import_etracer)
        menu.addSeparator()
        op_calc_action = menu.addAction(t('msg.Op_calculator'))
        op_calc_action.setToolTip(t('tip.Op_calculator'))
        op_calc_action.triggered.connect(self._show_op_calculator)

        settings_menu = self.menuBar().addMenu(t('menu.Settings'))
        load_action = settings_menu.addAction(t('menu.Load_scan_settings'))
        load_action.triggered.connect(self._load_scan_settings)
        save_action = settings_menu.addAction(t('menu.Save_scan_settings'))
        save_action.triggered.connect(self._save_scan_settings)

    def _build_measure_tab(self) -> QWidget:
        root = QWidget()
        layout = QHBoxLayout(root)

        # Left panel: QTabWidget with Measure + Amplifier tabs
        self.left_tabs = QTabWidget()
        self.left_tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Tab 1: Measure controls
        measure_left = QWidget()
        left = QVBoxLayout(measure_left)
        apply_no_margin(left)

        left.addWidget(self._build_connection_group())
        self.live_panel = LivePanel(
            title=t('live.Live_parameters'), sep=": ", layout_mode="compact"
        )
        left.addWidget(self.live_panel)
        left.addWidget(self._build_lamp_group())
        left.addWidget(self._build_preheat_group())
        left.addWidget(self._build_ranges_group())
        left.addWidget(self._build_zone_group())
        left.addWidget(self._build_actions_group())
        left.addStretch(1)
        # Explicit minimum, so the column COMPRESSES under the splitter
        # instead of nailing it: Qt honours an explicit minimumWidth over
        # the layout's own minimumSizeHint (the widest group in here).
        measure_left.setMinimumWidth(self.LEFT_PANEL_MIN_W)
        self.left_tabs.addTab(measure_left, t('tab.Measure'))

        # Tab 2: Amplifier control panel (scrollable)
        self.amp_control_panel = AmpControlPanel()
        amp_scroll = QScrollArea()
        amp_scroll.setWidget(self.amp_control_panel)
        amp_scroll.setWidgetResizable(True)
        amp_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.left_tabs.addTab(amp_scroll, t('tab.Amplifier'))

        right_panel = QWidget()
        right = QVBoxLayout(right_panel)
        apply_no_margin(right)

        self.plot_tabs = QTabWidget()
        self._build_plot_tabs()

        self.plot_renderer = PlotRenderer(
            self.plot,
            self.contour_plot,
            self.contour_image,
            self.transfer_plot,
            gm_plot=self.gm_plot,
            gm_image=self.gm_image,
            rp_plot=self.rp_plot,
            rp_image=self.rp_image,
            mu_plot=self.mu_plot,
            mu_image=self.mu_image,
            pa_map_plot=self.pa_map_plot,
            pa_map_image=self.pa_map_image,
            curves_plot=self.curves_plot,
            marker_lock_px=self.app_config.marker_lock_px,
            ua_cluster_thr=self.app_config.ua_cluster_threshold,
            ug1_cluster_thr=self.app_config.ug1_cluster_threshold,
            ug2_cluster_thr=self.app_config.ug2_cluster_threshold,
        )
        self.plot_renderer.configure_base()
        self.plot_renderer.configure_base_extended()

        # Same rule as the control column: an explicit minimum lets this
        # group compress, instead of its widest row (~1067 px) becoming the
        # splitter's right-hand stop.
        plot_options = self._build_plot_options_group()
        plot_options.setMinimumWidth(self.RIGHT_PANEL_MIN_W)
        right.addWidget(plot_options)

        # Amplifier tab placeholder — fully initialized after _init_plot_manager
        self._amplifier_tab_placeholder = QWidget()
        self.plot_tabs.addTab(self._amplifier_tab_placeholder, t('tab.Amplifier'))
        self.plot_tabs.setTabToolTip(
            self.plot_tabs.indexOf(self._amplifier_tab_placeholder),
            t('tip.Tab_Amplifier'),
        )
        self.load_line_info = QLabel("")
        self.load_line_info.setStyleSheet(
            f"color: {COLOR_ACCENT_BLUE}; {STYLE_BOLD_LABEL_SM}"
        )
        self.load_line_info.setWordWrap(True)
        self.load_line_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.load_line_info.setVisible(False)
        right.addWidget(self.load_line_info)
        right.addWidget(self.plot_tabs)

        # Splitter, not a fixed pair: the control column used to take
        # whatever its widest child asked for, with no way to give the
        # plots more room on a narrow screen.
        self.measure_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.measure_splitter.addWidget(self.left_tabs)
        self.measure_splitter.addWidget(right_panel)
        self.measure_splitter.setStretchFactor(0, 0)
        self.measure_splitter.setStretchFactor(1, 1)
        # Collapsing either side to zero looks like a broken window and is
        # only undone by finding a 5 px handle; resizing stays free.
        self.measure_splitter.setChildrenCollapsible(False)
        self._measure_page = measure_left
        layout.addWidget(self.measure_splitter)
        return root

    def _fit_measure_splitter(self, total: int = 0) -> None:
        """Open the control column at the width its content asks for, but
        never wider than what leaves the plot side its own minimum.

        *total* is the width to divide, defaulting to the splitter's own.
        Must run after the first show: a splitter discards sizes set
        before it has laid out once.
        """
        total = total or self.measure_splitter.width()
        right_min = self.measure_splitter.widget(1).minimumSizeHint().width()
        natural = self._measure_page.sizeHint().width()
        room = total - right_min - self.measure_splitter.handleWidth()
        width = max(self.LEFT_PANEL_MIN_W, min(natural, room))
        self.measure_splitter.setSizes([width, total - width])

    def _health_load_to_manual(self, conditions: Dict) -> None:
        """Load conditions from a health measurement into the Manual tab."""
        if hasattr(self, "manual_tab"):
            self.manual_tab.load_from_conditions(conditions)
            self.tabs.setCurrentWidget(self.manual_tab)

    def _build_plot_tabs(self) -> None:
        """Populate self.plot_tabs with all plot/analysis tabs."""
        self.plot = pg.PlotWidget(title=t('plot.Ia_vs_Ua'))
        self.plot_tabs.addTab(self.plot, t('tab.2D'))
        self.plot_tabs.setTabToolTip(
            self.plot_tabs.indexOf(self.plot),
            t('tip.Tab_2D'),
        )

        # Transfer tab: control row (view preset + Ua slice filter) + plot
        transfer_tab_widget = QWidget()
        transfer_tab_layout = QVBoxLayout(transfer_tab_widget)
        transfer_tab_layout.setContentsMargins(0, 0, 0, 0)

        transfer_ctrl = QHBoxLayout()
        transfer_ctrl.setContentsMargins(4, 2, 4, 2)
        self.transfer_view_combo = QComboBox()
        for mode, key in (
            (TRANSFER_VIEW_ALL, 'plot.Transfer_view_all'),
            (TRANSFER_VIEW_DATASHEET, 'plot.Transfer_view_datasheet'),
            (TRANSFER_VIEW_LOADLINE, 'plot.Transfer_view_loadline'),
            (TRANSFER_VIEW_CUSTOM, 'plot.Transfer_view_custom'),
        ):
            self.transfer_view_combo.addItem(t(key), userData=mode)
        self.transfer_view_combo.setCurrentIndex(
            self.transfer_view_combo.findData(TRANSFER_VIEW_DATASHEET))
        self.transfer_view_combo.setToolTip(t('tip.Transfer_view'))
        self.transfer_ua_combo = CheckableComboBox(
            placeholder=t('common.Ua'))
        self.transfer_ua_combo.setMinimumWidth(self.UG2_DISPLAY_MIN_W)
        self.transfer_ua_combo.setMaximumWidth(self.UG2_DISPLAY_MAX_W)
        transfer_ctrl.addWidget(QLabel(
            t('common.Label_colon', label=t('plot.Transfer_view'))))
        transfer_ctrl.addWidget(self.transfer_view_combo)
        transfer_ctrl.addWidget(QLabel(
            t('common.Label_colon', label=t('common.Ua'))))
        transfer_ctrl.addWidget(self.transfer_ua_combo)
        transfer_ctrl.addStretch(1)
        transfer_tab_layout.addLayout(transfer_ctrl)

        self.transfer_plot = pg.PlotWidget(title=t('plot.Ia_vs_Ug1'))
        transfer_tab_layout.addWidget(self.transfer_plot, 1)
        self.plot_tabs.addTab(transfer_tab_widget, t('tab.Transfer'))
        self.plot_tabs.setTabToolTip(
            self.plot_tabs.indexOf(transfer_tab_widget),
            t('tip.Tab_Transfer'),
        )

        self.contour_plot = pg.PlotWidget(title=t('plot.Contour_title'))
        self.contour_image = pg.ImageItem()
        colormap = pg.colormap.get('viridis')
        self.contour_image.setLookupTable(colormap.getLookupTable())
        self.contour_plot.addItem(self.contour_image)
        self.plot_tabs.addTab(self.contour_plot, t('tab.Contour'))
        self.plot_tabs.setTabToolTip(
            self.plot_tabs.indexOf(self.contour_plot),
            t('tip.Tab_Contour'),
        )

        # Gm/Rp/µ tab: Gm always left, right = stacked Rp / µ with combo
        gm_rp_widget = QWidget()
        gm_rp_outer = QVBoxLayout(gm_rp_widget)
        gm_rp_outer.setContentsMargins(0, 0, 0, 0)
        gm_rp_outer.setSpacing(2)

        # Combo selector for right heatmap (pushed to right half)
        combo_row = QHBoxLayout()
        combo_row.setContentsMargins(0, 0, 0, 0)
        combo_row.addStretch(1)
        self.right_heatmap_combo = QComboBox()
        self.right_heatmap_combo.addItem(t('tip.Right_heatmap_Rp'), userData="rp")
        self.right_heatmap_combo.addItem(t('tip.Right_heatmap_Mu'), userData="mu")
        self.right_heatmap_combo.setFixedWidth(60)
        self.right_heatmap_combo.setToolTip(t('tip.Right_heatmap_combo'))
        self.right_heatmap_combo.currentIndexChanged.connect(
            self._on_right_heatmap_changed)
        combo_row.addWidget(self.right_heatmap_combo)
        gm_rp_outer.addLayout(combo_row)

        # Two plots side by side: Gm (left) + stacked Rp/µ (right)
        gm_rp_plots = QHBoxLayout()
        gm_rp_plots.setContentsMargins(0, 0, 0, 0)

        self.gm_plot = pg.PlotWidget(title=t('plot.Gm_title'))
        self.gm_image = pg.ImageItem()
        self.gm_image.setLookupTable(colormap.getLookupTable())
        self.gm_plot.addItem(self.gm_image)

        self.rp_plot = pg.PlotWidget(title=t('plot.Rp_title'))
        self.rp_image = pg.ImageItem()
        self.rp_image.setLookupTable(colormap.getLookupTable())
        self.rp_plot.addItem(self.rp_image)

        self.mu_plot = pg.PlotWidget(title=t('plot.Mu_title'))
        self.mu_image = pg.ImageItem()
        self.mu_image.setLookupTable(colormap.getLookupTable())
        self.mu_plot.addItem(self.mu_image)

        self.right_heatmap_stack = QStackedWidget()
        self.right_heatmap_stack.addWidget(self.rp_plot)   # index 0
        self.right_heatmap_stack.addWidget(self.mu_plot)   # index 1
        # Prevent stack from shrinking on page switch
        self.right_heatmap_stack.setSizePolicy(
            self.gm_plot.sizePolicy())

        gm_rp_plots.addWidget(self.gm_plot, 1)
        gm_rp_plots.addWidget(self.right_heatmap_stack, 1)
        gm_rp_outer.addLayout(gm_rp_plots, 1)

        self.plot_tabs.addTab(gm_rp_widget, t('tab.Gm_Rp'))
        self.plot_tabs.setTabToolTip(
            self.plot_tabs.indexOf(gm_rp_widget),
            t('tip.Tab_Gm_Rp'),
        )

        # Pa map tab
        self.pa_map_plot = pg.PlotWidget(title=t('plot.Pa_map_title'))
        self.pa_map_image = pg.ImageItem()
        self.pa_map_image.setLookupTable(colormap.getLookupTable())
        self.pa_map_plot.addItem(self.pa_map_image)
        self.plot_tabs.addTab(self.pa_map_plot, t('tab.Pa_map'))
        self.plot_tabs.setTabToolTip(
            self.plot_tabs.indexOf(self.pa_map_plot),
            t('tip.Tab_Pa_map'),
        )

        # Curves tab (Gm/Rp/mu/Ia/Ig2 as line curves)
        curves_tab_widget = QWidget()
        curves_tab_layout = QVBoxLayout(curves_tab_widget)
        curves_tab_layout.setContentsMargins(0, 0, 0, 0)

        curves_ctrl = QHBoxLayout()
        curves_ctrl.setContentsMargins(4, 2, 4, 2)
        self.curves_y_combo = QComboBox()
        self.curves_y_combo.addItems(["Gm", "Rp", "mu", "Ia", "Ig2", "Pa", "Pig2", "Ia/Ig2"])
        _y_tips = {
            "Gm": t('tip.Curves_Y_Gm'), "Rp": t('tip.Curves_Y_Rp'),
            "mu": t('tip.Curves_Y_mu'), "Ia": t('tip.Curves_Y_Ia'),
            "Ig2": t('tip.Curves_Y_Ig2'), "Pa": t('tip.Curves_Y_Pa'),
            "Pig2": t('tip.Curves_Y_Pig2'), "Ia/Ig2": t('tip.Curves_Y_Ia_Ig2'),
        }
        for i in range(self.curves_y_combo.count()):
            key = self.curves_y_combo.itemText(i)
            self.curves_y_combo.setItemData(i, _y_tips.get(key, ""), Qt.ItemDataRole.ToolTipRole)
        self.curves_y_combo.setToolTip(t('tip.Curves_Y_Gm'))
        self.curves_y_combo.currentIndexChanged.connect(
            lambda idx: self.curves_y_combo.setToolTip(
                self.curves_y_combo.itemData(idx, Qt.ItemDataRole.ToolTipRole) or ""))
        self.curves_y_combo.setMaximumWidth(self.MEDIUM_SPIN_W)
        self.curves_x_combo = QComboBox()
        self.curves_x_combo.addItems(["Ua", "Ug1", "Ug2"])
        _x_tips = {
            "Ua": t('tip.Curves_X_Ua'), "Ug1": t('tip.Curves_X_Ug1'),
            "Ug2": t('tip.Curves_X_Ug2'),
        }
        for i in range(self.curves_x_combo.count()):
            key = self.curves_x_combo.itemText(i)
            self.curves_x_combo.setItemData(i, _x_tips.get(key, ""), Qt.ItemDataRole.ToolTipRole)
        self.curves_x_combo.setToolTip(t('tip.Curves_X_Ua'))
        self.curves_x_combo.currentIndexChanged.connect(
            lambda idx: self.curves_x_combo.setToolTip(
                self.curves_x_combo.itemData(idx, Qt.ItemDataRole.ToolTipRole) or ""))
        self.curves_x_combo.setMaximumWidth(self.MEDIUM_SPIN_W)
        curves_ctrl.addWidget(QLabel(t('common.Label_colon', label=t('common.Y'))))
        curves_ctrl.addWidget(self.curves_y_combo)
        curves_ctrl.addWidget(QLabel(t('common.Label_colon', label=t('common.X'))))
        curves_ctrl.addWidget(self.curves_x_combo)
        curves_ctrl.addStretch(1)
        curves_tab_layout.addLayout(curves_ctrl)

        self.curves_plot = pg.PlotWidget(title=t('plot.Curves_title'))
        curves_tab_layout.addWidget(self.curves_plot, 1)
        self.plot_tabs.addTab(curves_tab_widget, t('tab.Curves'))
        self.plot_tabs.setTabToolTip(
            self.plot_tabs.indexOf(curves_tab_widget),
            t('tip.Tab_Curves'),
        )

    def _build_manual_tab(self) -> QWidget:
        self.manual_tab = ManualTab(
            get_client=lambda: self.client,
            get_app_config=lambda: self.app_config,
            get_calibration=lambda: self.calibration,
            get_write_locked=lambda: self._emergency_lock,
            get_hw_busy=self._hw_busy_reason,
            on_add_to_main_plot=self._on_manual_add_to_main_plot,
            on_save=self._on_manual_save,
            get_current_lamp=lambda: find_lamp(self.lamps, self.lamp_combo.currentText()),
        )
        return self.manual_tab

    def _on_manual_add_to_main_plot(self, manual_points: List[Dict]) -> None:
        """Callback when Manual tab adds points to main plot."""
        pm = self.plot_mgr
        next_series_id = max((p.get("series_id", 0) for p in pm.points), default=0) + 1
        for p in manual_points:
            pt = dict(p)
            pt["series_id"] = next_series_id
            pm.points.append(pt)
        pm.series_labels[next_series_id] = f"Manual ({len(manual_points)} pts)"
        pm.refresh_lamp_combos()
        pm.refresh_ug2_combos(pm.points)
        pm.invalidate_cache()
        pm.render_all()
        self.tabs.setCurrentIndex(0)

    def _on_save_manual_btn(self) -> None:
        """Save button on S/R/K zone panel — delegates to manual save."""
        include = self.manual_tab.include_points_cb.isChecked()
        points = list(self.manual_tab.manual_points) if include else []
        self._on_manual_save(points, include)

    def _on_manual_save(self, points: List[Dict], include_points: bool) -> None:
        """Save manual measurement using already-measured SRK results."""
        # Extract SRK averages from existing results
        results = self.srk.srk_results
        valid = [r for r in results if r.get("s") is not None]
        if valid:
            s_avg = sum(r["s"] for r in valid) / len(valid)
            r_avg = sum(r["r"] for r in valid) / len(valid)
            k_avg = sum(r["k"] for r in valid) / len(valid)
        else:
            s_avg = r_avg = k_avg = None

        # Build scan section from manual points
        if points:
            ua_vals = [p["ua"] for p in points]
            ug1_vals = [p["ug1"] for p in points]
            ug2_vals = [p["ug2"] for p in points]
            scan_section = {
                "ua": {"start": min(ua_vals), "stop": max(ua_vals), "step": 0},
                "ug1": {"start": min(ug1_vals), "stop": max(ug1_vals), "step": 0},
                "ug2": {"start": min(ug2_vals), "stop": max(ug2_vals), "step": 0},
                "uh": points[-1].get("uh", 0),
                "ih": points[-1].get("ih", 0),
            }
        else:
            scan_section = {
                "ua": {"start": 0, "stop": 0, "step": 0},
                "ug1": {"start": 0, "stop": 0, "step": 0},
                "ug2": {"start": 0, "stop": 0, "step": 0},
                "uh": 0,
                "ih": 0,
            }

        tube_type = self.lamp_combo.currentText()
        lamp_id = self.lamp_panel.lamp_id()
        name = self.measurement_name_edit.text().strip()
        if not name:
            name = "manual"

        srk_points = self.srk.srk_points

        # Add topology and ug2_mode to scan section
        scan_section["ug2_track_ua"] = self.ug2_track_radio.isChecked()
        scan_section["ug2_mode"] = self._current_ug2_mode()

        measurement = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "tube_type": tube_type,
            "lamp_id": lamp_id,
            "name": name,
            "topology": getattr(self, '_topology', TOPOLOGY_PENTODE),
            "source": "manual",
            "scan": scan_section,
            "zone": {
                "ua_min": self.zone_ua_min.value(),
                "ua_max": self.zone_ua_max.value(),
                "ug1_min": self.zone_ug1_min.value(),
                "ug1_max": self.zone_ug1_max.value(),
            },
            "srk": {"s": s_avg, "r": r_avg, "k": k_avg},
            "srk_method": "measured" if valid else "none",
            "srk_results": [{"s": r["s"], "r": r["r"], "k": r["k"]} for r in results],
            "srk_points": srk_points,
            "points": points,
        }

        self._save_and_display_measurement(measurement, points)
        n_pts = len(points)
        QMessageBox.information(
            self, t('msg.Save'),
            t('msg.Saved_points', count=n_pts)
        )

    def _build_connection_group(self) -> QGroupBox:
        box = QGroupBox(t('conn.Connection'))
        layout = QVBoxLayout(box)
        apply_tight(layout)
        row1 = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.setToolTip(t("tip.Port_combo"))
        self.refresh_ports_btn = QPushButton(t('conn.Refresh'))
        self.refresh_ports_btn.setToolTip(t('tip.Refresh_ports'))
        self.refresh_ports_btn.clicked.connect(self._refresh_ports)
        self.test_port_btn = QPushButton(t('conn.Check_COM'))
        self.test_port_btn.setToolTip(t('tip.Check_COM'))
        self.test_port_btn.clicked.connect(self._check_com)
        self.trace_checkbox = QCheckBox(t('conn.Trace_COM'))
        self.trace_checkbox.setToolTip(t('tip.Trace_COM'))
        self.trace_checkbox.stateChanged.connect(self._toggle_trace)
        self.connect_btn = QPushButton(t('conn.Connect'))
        self.connect_btn.setToolTip(t('tip.Connect'))
        self.connect_btn.clicked.connect(self._toggle_connection)
        self.port_combo.setMaximumWidth(self.port_combo.fontMetrics().horizontalAdvance('M') * 10 + 30)
        row1.addWidget(self.port_combo)
        row1.addWidget(self.refresh_ports_btn)
        row1.addWidget(self.test_port_btn)
        row1.addWidget(self.connect_btn)
        row1.addWidget(self.trace_checkbox)
        layout.addLayout(row1)
        return box


    def _build_lamp_group(self) -> LampPanel:
        self.lamp_panel = LampPanel(
            name_label=t("lamp.Measurement_name"),
            show_name=True,
        )
        self.lamp_panel.tube_changed.connect(self._apply_lamp)
        self.lamp_panel.anode_changed.connect(self._on_anode_changed)

        self.lamp_combo = self.lamp_panel.tube_combo
        self.lamp_id_edit = self.lamp_panel.lamp_id_edit
        self.measurement_name_edit = self.lamp_panel.name_edit
        self.socket_label = self.lamp_panel.socket_label
        self.mode_label = self.lamp_panel.mode_label
        self.anode_group = self.lamp_panel.anode_group
        self.anode_1 = self.lamp_panel.anode_1
        self.anode_2 = self.lamp_panel.anode_2
        return self.lamp_panel

    def _build_ranges_group(self) -> QGroupBox:
        box = QGroupBox(t('scan.Scan_ranges'))
        layout = QFormLayout(box)
        apply_tight(layout)
        self._build_ua_ug1_ranges(layout)
        self._build_ug2_mode_and_range(layout)
        self._build_power_dissipation_row(layout)
        return box

    def _build_ua_ug1_ranges(self, layout: QFormLayout) -> None:
        self.ua_start = QDoubleSpinBox()
        self.ua_stop = QDoubleSpinBox()
        self.ua_step = QDoubleSpinBox()
        self.ug1_start = QDoubleSpinBox()
        self.ug1_stop = QDoubleSpinBox()
        self.ug1_step = QDoubleSpinBox()
        self.ug2_start = QDoubleSpinBox()
        self.ug2_stop = QDoubleSpinBox()
        self.ug2_step = QDoubleSpinBox()

        for w in (self.ua_start, self.ua_stop, self.ua_step, self.ug2_start, self.ug2_stop, self.ug2_step):
            w.setDecimals(1)
            w.setRange(0, 300)
        for w in (self.ug1_start, self.ug1_stop):
            w.setDecimals(1)
            w.setRange(-50, 0)
        self.ug1_step.setDecimals(1)
        self.ug1_step.setRange(0, 20)

        self.ua_step.setValue(10)
        self.ug1_step.setValue(1)
        self.ug2_step.setValue(10)
        self.ua_start.setToolTip(t("tip.Scan_ua_start"))
        self.ua_stop.setToolTip(t("tip.Scan_ua_stop"))
        self.ua_step.setToolTip(t("tip.Scan_ua_step"))
        self.ug1_start.setToolTip(t("tip.Scan_ug1_start"))
        self.ug1_stop.setToolTip(t("tip.Scan_ug1_stop"))
        self.ug1_step.setToolTip(t("tip.Scan_ug1_step"))
        self.ug2_start.setToolTip(t("tip.Scan_ug2_start"))
        self.ug2_stop.setToolTip(t("tip.Scan_ug2_stop"))
        self.ug2_step.setToolTip(t("tip.Scan_ug2_step"))

        ua_row = QHBoxLayout()
        ua_row.addWidget(QLabel(t('scan.Min')))
        ua_row.addWidget(self.ua_start)
        ua_row.addWidget(QLabel(t('scan.Max')))
        ua_row.addWidget(self.ua_stop)
        ua_row.addWidget(QLabel(t('scan.Step')))
        ua_row.addWidget(self.ua_step)
        layout.addRow(t("common.Label_unit_colon", label=t("common.Ua"), unit=t("common.V")), ua_row)

        ug1_row = QHBoxLayout()
        ug1_row.addWidget(QLabel(t('scan.Min')))
        ug1_row.addWidget(self.ug1_start)
        ug1_row.addWidget(QLabel(t('scan.Max')))
        ug1_row.addWidget(self.ug1_stop)
        ug1_row.addWidget(QLabel(t('scan.Step')))
        ug1_row.addWidget(self.ug1_step)
        layout.addRow(t("common.Label_unit_colon", label=t("common.Ug1"), unit=t("common.V")), ug1_row)

    def _build_ug2_mode_and_range(self, layout: QFormLayout) -> None:
        self.ug2_scan_mode_group = QButtonGroup(self)
        self.ug2_sweep_radio = QRadioButton(t('scan.Sweep'))
        self.ug2_track_radio = QRadioButton(t('scan.Track_Ua'))
        self.ug2_sweep_radio.setToolTip(t('tip.Sweep_Ug2'))
        self.ug2_track_radio.setToolTip(t('tip.Track_Ug2'))
        self.ug2_sweep_radio.setChecked(True)
        self.ug2_scan_mode_group.addButton(self.ug2_sweep_radio, 0)
        self.ug2_scan_mode_group.addButton(self.ug2_track_radio, 1)
        self.ug2_offset = make_double_spinbox(
            min_val=-100, max_val=100, value=0,
            decimals=1,
            tooltip_key="tip.Scan_ug2_offset",
        )
        self.ug2_offset.setEnabled(False)
        self.ug2_scan_mode_group.idClicked.connect(self._on_ug2_scan_mode_changed)

        self.pa_over_pct = make_double_spinbox(
            min_val=0, max_val=200, value=self.app_config.scan_pa_over_pct,
            decimals=0, suffix=" %", fixed_width=65,
            tooltip_key="tip.Pa_over",
        )

        self.pig2_over_pct = make_double_spinbox(
            min_val=0, max_val=200, value=self.app_config.scan_pig2_over_pct,
            decimals=0, suffix=" %", fixed_width=65,
            tooltip_key="tip.Pg2_over",
        )

        ug2_mode_row = QHBoxLayout()
        ug2_mode_row.addWidget(self.ug2_sweep_radio)
        ug2_mode_row.addWidget(self.ug2_track_radio)
        ug2_mode_row.addWidget(self.ug2_offset)
        self._ug2_offset_v_label = QLabel(t('common.V'))
        ug2_mode_row.addWidget(self._ug2_offset_v_label)
        ug2_mode_row.addStretch(1)
        self._ug2_mode_label = QLabel(t('scan.Ug2_mode'))
        layout.addRow(self._ug2_mode_label, ug2_mode_row)

        ug2_row = QHBoxLayout()
        self._ug2_range_min_label = QLabel(t('scan.Min'))
        self._ug2_range_max_label = QLabel(t('scan.Max'))
        self._ug2_range_step_label = QLabel(t('scan.Step'))
        ug2_row.addWidget(self._ug2_range_min_label)
        ug2_row.addWidget(self.ug2_start)
        ug2_row.addWidget(self._ug2_range_max_label)
        ug2_row.addWidget(self.ug2_stop)
        ug2_row.addWidget(self._ug2_range_step_label)
        ug2_row.addWidget(self.ug2_step)
        self._ug2_range_label = QLabel(
            t('common.Label_unit_colon', label=t('common.Ug2'), unit=t('common.V'))
        )
        layout.addRow(self._ug2_range_label, ug2_row)

    def _build_power_dissipation_row(self, layout: QFormLayout) -> None:
        p_row = QHBoxLayout()
        self._pa_over_label = QLabel(t('scan.Pa_over'))
        p_row.addWidget(self._pa_over_label)
        p_row.addWidget(self.pa_over_pct)
        self._pg2_over_label = QLabel(t('scan.Pg2_over'))
        p_row.addWidget(self._pg2_over_label)
        p_row.addWidget(self.pig2_over_pct)
        p_row.addStretch(1)
        layout.addRow(p_row)

    def _set_ug2_visibility(self, visible: bool) -> None:
        """Show/hide all Ug2-related controls (for triode vs pentode)."""
        # Scan range row: Ug2 start/stop/step
        for w in (self._ug2_range_label, self._ug2_range_min_label,
                  self._ug2_range_max_label, self._ug2_range_step_label,
                  self.ug2_start, self.ug2_stop, self.ug2_step):
            w.setVisible(visible)
        # Scan mode row: Sweep/Track radio, offset, Pg2 over (keep Pa_over visible)
        for w in (self._ug2_mode_label, self.ug2_sweep_radio,
                  self.ug2_track_radio, self.ug2_offset,
                  self._ug2_offset_v_label):
            w.setVisible(visible)
        # Zone: Ug2 label and spinbox
        self._zone_ug2_label.setVisible(visible)
        self.zone_ug2.setVisible(visible)
        # Plot options: Ug2 mode radios (display/calc combos stay visible always)
        for w in (self._plot_ug2_mode_label, self.ug2_mode_series,
                  self.ug2_mode_color):
            w.setVisible(visible)
    def _on_ug2_scan_mode_changed(self, mode_id: int) -> None:
        is_sweep = mode_id == 0
        self.ug2_start.setEnabled(is_sweep)
        self.ug2_stop.setEnabled(is_sweep)
        self.ug2_step.setEnabled(is_sweep)
        self.ug2_offset.setEnabled(not is_sweep)
        self.zone_ug2.setEnabled(is_sweep)

    def _build_plot_options_group(self) -> QGroupBox:
        box = QGroupBox(t('plot.Plot_options'))
        vlayout = QVBoxLayout(box)
        vlayout.addLayout(self._build_plot_row_selectors())
        vlayout.addLayout(self._build_plot_row_axes())
        vlayout.addLayout(self._build_plot_row_limits())
        vlayout.addLayout(self._build_plot_row_loadline())
        return box

    def _build_plot_row_selectors(self) -> QHBoxLayout:
        """Row 0: lamp & Ug2 display/calc selectors."""
        row = QHBoxLayout()
        self._source_label = QLabel(t('plot.Lamps'))
        self.lamp_display_combo = CheckableComboBox(placeholder=t('plot.All'))
        self.lamp_display_combo.setMinimumWidth(self.LAMP_COMBO_MIN_W)
        self.lamp_display_combo.setMaximumWidth(self.LAMP_COMBO_MAX_W)
        self._calc_lamp_label = QLabel(t('plot.Lamp_calc'))
        self.lamp_calc_combo = QComboBox()
        self.lamp_calc_combo.setToolTip(t("tip.Lamp_calc_combo"))
        self.lamp_calc_combo.setMinimumWidth(self.LAMP_COMBO_MIN_W)
        self.lamp_calc_combo.setMaximumWidth(self.LAMP_COMBO_MAX_W)

        self._ug2_display_label = QLabel(t('common.Label_colon', label=t('common.Ug2')))
        self.ug2_display_combo = CheckableComboBox(placeholder=t('common.Ug2'))
        self.ug2_display_combo.setMinimumWidth(self.UG2_DISPLAY_MIN_W)
        self.ug2_display_combo.setMaximumWidth(self.UG2_DISPLAY_MAX_W)
        self._ug2_calc_label = QLabel(t('plot.Ug2_calc'))
        self.ug2_calc_combo = QComboBox()
        self.ug2_calc_combo.setToolTip(t("tip.Ug2_calc_combo"))
        self.ug2_calc_combo.setMinimumWidth(self.UG2_CALC_MIN_W)
        self.ug2_calc_combo.setMaximumWidth(self.UG2_CALC_MAX_W)

        row.addWidget(self._source_label)
        row.addWidget(self.lamp_display_combo)
        row.addWidget(self._ug2_display_label)
        row.addWidget(self.ug2_display_combo)
        row.addSpacing(self.SECTION_SPACING)
        row.addWidget(self._calc_lamp_label)
        row.addWidget(self.lamp_calc_combo)
        row.addWidget(self._ug2_calc_label)
        row.addWidget(self.ug2_calc_combo)
        row.addStretch(1)
        self.clear_plot_btn = QPushButton(t('plot.Clear'))
        self.clear_plot_btn.setToolTip(t('tip.Clear_plot'))
        self.clear_plot_btn.clicked.connect(self._clear_all_plots)
        row.addWidget(self.clear_plot_btn)
        return row

    def _build_plot_row_axes(self) -> QHBoxLayout:
        """Row 1: what is plotted and in which axes — Ug2 mode, Ia axis,
        auto-range buttons, legend. Drawing style lives in row 3."""
        row = QHBoxLayout()
        self.ug2_mode_group = QButtonGroup(self)
        self.ug2_mode_series = QRadioButton(t('plot.Ug2_as_series'))
        self.ug2_mode_series.setToolTip(t('tip.Ug2_as_series'))
        self.ug2_mode_color = QRadioButton(t('plot.Ug2_as_color'))
        self.ug2_mode_color.setToolTip(t('tip.Ug2_as_color'))
        self.ug2_mode_color.setChecked(True)
        self.ug2_mode_group.addButton(self.ug2_mode_series)
        self.ug2_mode_group.addButton(self.ug2_mode_color)
        self.ug2_mode_series.toggled.connect(self._rerender_2d)
        self.ug2_mode_color.toggled.connect(self._rerender_2d)

        self.legend_toggle_btn = QPushButton(t('plot.Hide_legend'))
        self.legend_toggle_btn.setToolTip(t('tip.Legend_toggle'))
        self.legend_toggle_btn.clicked.connect(self._toggle_legend)
        self.ia_max_input = make_double_spinbox(
            min_val=1, max_val=2000, value=self.app_config.plot_ia_max,
            decimals=0,
            tooltip_key="tip.Ia_max_axis",
            on_change=self._update_ia_axis,
        )
        self.ia_max_auto_btn = QPushButton(t('plot.Auto_Ia'))
        self.ia_max_auto_btn.setToolTip(t('tip.Auto_Ia'))
        self.ia_max_auto_btn.clicked.connect(self._auto_ia_axis)

        self._plot_ug2_mode_label = QLabel(t('plot.Ug2_mode'))
        row.addWidget(self._plot_ug2_mode_label)
        row.addWidget(self.ug2_mode_series)
        row.addWidget(self.ug2_mode_color)
        row.addSpacing(self.CONTROL_SPACING)
        row.addWidget(QLabel(t('plot.Ia_max')))
        row.addWidget(self.ia_max_input)
        row.addWidget(self.ia_max_auto_btn)
        row.addSpacing(self.CONTROL_SPACING)
        self.ua_auto_btn = QPushButton(t('plot.Auto_U'))
        self.ua_auto_btn.setToolTip(t('tip.Auto_U'))
        self.ua_auto_btn.clicked.connect(self._auto_ua_axis)
        row.addWidget(self.ua_auto_btn)
        row.addSpacing(self.CONTROL_SPACING)
        row.addWidget(self.legend_toggle_btn)
        row.addStretch(1)
        return row

    def _build_plot_row_limits(self) -> QHBoxLayout:
        """Row 2: Pa_max, Ua_max, Ia_max limit checkboxes with spinboxes."""
        row = QHBoxLayout()
        self.pa_max_cb = QCheckBox(t('plot.Pa_max'))
        self.pa_max_cb.setChecked(False)
        self.pa_max_cb.setToolTip(t('tip.Pa_max'))
        self.pa_max_cb.toggled.connect(self._rerender_2d_pa)
        self.pa_max_input = make_double_spinbox(
            min_val=0.1, max_val=999.0, value=12.5,
            step=0.5, decimals=1, suffix=" W",
            fixed_width=self.MEDIUM_SPIN_W,
            tooltip_key="tip.Pa_max_value",
            on_change=self._rerender_2d_pa,
        )
        self.pa_max_input.setEnabled(False)
        self.pa_max_cb.toggled.connect(self.pa_max_input.setEnabled)
        row.addWidget(self.pa_max_cb)
        row.addWidget(self.pa_max_input)
        row.addSpacing(self.CONTROL_SPACING)
        self.pg2_max_cb = QCheckBox(t('plot.Pg2_max'))
        self.pg2_max_cb.setChecked(False)
        self.pg2_max_cb.setToolTip(t('tip.Pg2_max'))
        self.pg2_max_cb.toggled.connect(self._rerender_2d_pa)
        self.pg2_max_input = make_double_spinbox(
            min_val=0.1, max_val=99.0, value=2.0,
            step=0.5, decimals=1, suffix=" W",
            fixed_width=self.MEDIUM_SPIN_W,
            tooltip_key="tip.Pg2_max_value",
            on_change=self._rerender_2d_pa,
        )
        self.pg2_max_input.setEnabled(False)
        self.pg2_max_cb.toggled.connect(self.pg2_max_input.setEnabled)
        row.addWidget(self.pg2_max_cb)
        row.addWidget(self.pg2_max_input)
        row.addSpacing(self.CONTROL_SPACING)
        self.ua_max_cb = QCheckBox(t('plot.Ua_max'))
        self.ua_max_cb.setChecked(False)
        self.ua_max_cb.setToolTip(t('tip.Ua_max'))
        self.ua_max_cb.toggled.connect(self._rerender_2d)
        self.ua_max_input = make_double_spinbox(
            min_val=1, max_val=1000, value=DEFAULT_UB_V,
            step=10, decimals=0, suffix=" V",
            fixed_width=self.MEDIUM_SPIN_W,
            tooltip_key="tip.Ua_max_value",
            on_change=self._rerender_2d,
        )
        self.ua_max_input.setEnabled(False)
        self.ua_max_cb.toggled.connect(self.ua_max_input.setEnabled)
        row.addWidget(self.ua_max_cb)
        row.addWidget(self.ua_max_input)
        row.addSpacing(self.CONTROL_SPACING)
        self.ia_max_limit_cb = QCheckBox(t('plot.Ia_max_limit'))
        self.ia_max_limit_cb.setChecked(False)
        self.ia_max_limit_cb.setToolTip(t('tip.Ia_max_limit'))
        self.ia_max_limit_cb.toggled.connect(self._rerender_2d)
        self.ia_max_limit_input = make_double_spinbox(
            min_val=1, max_val=2000, value=75,
            step=5, decimals=0, suffix=" mA",
            fixed_width=self.WIDE_SPIN_W,
            tooltip_key="tip.Ia_max_value",
            on_change=self._rerender_2d,
        )
        self.ia_max_limit_input.setEnabled(False)
        self.ia_max_limit_cb.toggled.connect(self.ia_max_limit_input.setEnabled)
        row.addWidget(self.ia_max_limit_cb)
        row.addWidget(self.ia_max_limit_input)
        row.addSpacing(self.CONTROL_SPACING)
        self.model_btn = QPushButton(t('plot.Model'))
        self.model_btn.setToolTip(t('tip.Model_dialog'))
        self.model_btn.clicked.connect(self._show_model_dialog)
        row.addWidget(self.model_btn)
        row.addStretch(1)
        return row

    def _build_plot_row_loadline(self) -> QHBoxLayout:
        """Row 3: working line + how the plots are drawn (curve width,
        overlay pen, heatmap palette). Row 1 stayed crowded while this one
        held two controls, and the drawing-style group is the part of it
        that never answers "what is plotted", only "how"."""
        row = QHBoxLayout()
        # The line source is the amp panel (WorkingLineController);
        # the plot keeps the visibility checkbox and the distortion
        # dialog button. The Ub/Ra/Ug1/Sw spins are removed (they
        # duplicated the panel; every tick did a FULL re-render —
        # baseline bench: 446 ms/tick).
        self.load_line_cb = QCheckBox(t('plot.Load_line'))
        self.load_line_cb.setChecked(False)
        self.load_line_cb.setToolTip(t('tip.Load_line'))
        self.ra_sweep_btn = QPushButton(t('plot.Ra_sweep'))
        self.ra_sweep_btn.setToolTip(t('tip.Ra_sweep'))
        self.ra_sweep_btn.setEnabled(False)
        self.load_line_cb.toggled.connect(self.ra_sweep_btn.setEnabled)
        self.ra_sweep_btn.clicked.connect(self._show_ra_sweep)

        row.addWidget(self.load_line_cb)
        row.addSpacing(self.TIGHT_SPACING)
        row.addWidget(self.ra_sweep_btn)
        row.addSpacing(self.SECTION_SPACING)

        row.addWidget(QLabel(t('plot.Line')))
        self.plot_line_width = make_double_spinbox(
            min_val=0.5, max_val=5.0, value=PlotRenderer.DEFAULT_LINE_WIDTH,
            step=0.5, fixed_width=self.NARROW_SPIN_W,
            tooltip_key="tip.Line_width",
            on_change=self._rerender_line_width,
        )
        row.addWidget(self.plot_line_width)
        row.addSpacing(self.CONTROL_SPACING)
        row.addWidget(QLabel(t('plot.Overlay_style')))
        self.overlay_pen_style = QComboBox()
        self.overlay_pen_style.addItems(["—", "- -", "···", "-·-"])
        self.overlay_pen_style.setCurrentIndex(0)
        self.overlay_pen_style.setFixedWidth(self.NARROW_SPIN_W)
        self.overlay_pen_style.setToolTip(t('tip.Overlay_style'))
        self.overlay_pen_style.currentIndexChanged.connect(self._rerender_2d)
        row.addWidget(self.overlay_pen_style)
        row.addSpacing(self.SECTION_SPACING)
        row.addWidget(QLabel(t('plot.Heatmap_colormap')))
        self.heatmap_cmap_combo = QComboBox()
        self.heatmap_cmap_combo.addItems([
            "viridis", "plasma", "inferno", "magma", "cividis",
        ])
        self.heatmap_cmap_combo.setCurrentIndex(0)
        self.heatmap_cmap_combo.setFixedWidth(self.CMAP_COMBO_W)
        self.heatmap_cmap_combo.setToolTip(t('tip.Heatmap_colormap'))
        self.heatmap_cmap_combo.currentIndexChanged.connect(
            self._on_heatmap_colormap_changed)
        row.addWidget(self.heatmap_cmap_combo)
        # Lock the heatmap color scale: autoLevels re-stretches the
        # palette per render, making two scans incomparable by color.
        self.heatmap_lock_cb = QCheckBox(t('plot.Heatmap_lock'))
        self.heatmap_lock_cb.setChecked(False)
        self.heatmap_lock_cb.setToolTip(t('tip.Heatmap_lock'))
        self.heatmap_lock_cb.toggled.connect(self._on_heatmap_lock_toggled)
        row.addWidget(self.heatmap_lock_cb)
        row.addStretch(1)
        return row

    def _build_preheat_group(self) -> QGroupBox:
        box = QGroupBox(t('heat.Heater_preheat'))
        layout = QFormLayout(box)
        apply_tight(layout)
        self.preheat_target = QLabel(t('heat.Recommended'))
        self.preheat_live = QLabel(t('heat.Live_Uh_Ih'))
        self.preheat_seconds = make_int_spinbox(
            min_val=0, max_val=3600, value=120,
            tooltip_key="tip.Warmup_seconds",
        )
        # keyboard_tracking=False: on_change commands the heater — typing
        # "12.6" must not send the intermediate 1 and 12 to the tube (ML-122)
        self.uh_input = make_double_spinbox(
            min_val=0, max_val=20, value=0,
            step=0.1, decimals=1,
            tooltip_key="tip.Heater_uh_target",
            on_change=self._on_uh_changed,
            keyboard_tracking=False,
        )
        self.ih_input = make_double_spinbox(
            min_val=0, max_val=5, value=0,
            step=0.1, decimals=1,
            tooltip_key="tip.Heater_ih_target",
            on_change=self._on_ih_changed,
            keyboard_tracking=False,
        )
        self.preheat_enabled = QCheckBox(t('heat.Enable_preheat'))
        self.preheat_enabled.setToolTip(t('tip.Enable_preheat'))
        self.preheat_enabled.setChecked(True)
        self.preheat_progress = QProgressBar()
        self.preheat_progress.setRange(0, 100)
        self.preheat_status = QLabel(t('heat.Not_started'))
        self.preheat_start_btn = QPushButton(t('heat.Start_preheat'))
        self.preheat_start_btn.setToolTip(t('tip.Start_preheat'))
        self.preheat_start_btn.clicked.connect(self._start_preheat)
        self.preheat_stop_btn = QPushButton(t('heat.Stop'))
        self.preheat_stop_btn.setToolTip(t('tip.Stop_preheat'))
        self.preheat_stop_btn.clicked.connect(self._stop_preheat)

        target_row = QHBoxLayout()
        target_row.addWidget(self.preheat_target)
        target_row.addSpacing(self.CONTROL_SPACING)
        target_row.addWidget(QLabel(t('common.Label_unit_colon', label=t('common.Ih'), unit=t('common.A'))))
        target_row.addWidget(self.ih_input)
        target_row.addSpacing(self.CONTROL_SPACING)
        target_row.addWidget(QLabel(t('common.Label_unit_colon', label=t('common.Uh'), unit=t('common.V'))))
        target_row.addWidget(self.uh_input)
        target_row.addSpacing(self.CONTROL_SPACING)
        target_row.addWidget(QLabel(t('heat.Warmup_s')))
        target_row.addWidget(self.preheat_seconds)
        target_row.addStretch(1)
        layout.addRow(t('heat.Target'), target_row)
        self.preheat_live.setText(f"{self.preheat_live.text()}  [{self.preheat_status.text()}]")
        layout.addRow(self.preheat_live)
        btns = QHBoxLayout()
        btns.addWidget(self.preheat_enabled)
        btns.addWidget(self.preheat_start_btn)
        btns.addWidget(self.preheat_stop_btn)
        btns.addWidget(self.preheat_progress, 1)
        layout.addRow(btns)
        return box

    def _build_zone_group(self) -> QGroupBox:
        box = QGroupBox(t('zone.SRK_zone'))
        layout = QFormLayout(box)
        apply_tight(layout)
        self.zone_ua_min = QDoubleSpinBox()
        self.zone_ua_max = QDoubleSpinBox()
        self.zone_ug1_min = QDoubleSpinBox()
        self.zone_ug1_max = QDoubleSpinBox()
        self.zone_ug2 = QDoubleSpinBox()
        for w in (self.zone_ua_min, self.zone_ua_max, self.zone_ug2):
            w.setDecimals(1)
            w.setRange(0, 300)
        for w in (self.zone_ug1_min, self.zone_ug1_max):
            w.setDecimals(1)
            w.setRange(-50, 0)
        self.zone_ug2.setValue(100)
        self.zone_ua_min.setToolTip(t("tip.Zone_ua_min"))
        self.zone_ua_max.setToolTip(t("tip.Zone_ua_max"))
        self.zone_ug1_min.setToolTip(t("tip.Zone_ug1_min"))
        self.zone_ug1_max.setToolTip(t("tip.Zone_ug1_max"))
        self.zone_ug2.setToolTip(t("tip.Zone_ug2"))

        self.srk_label = QLabel(t("srk.label_srk_none"))
        self.quality_label = QLabel("")
        self.quality_label.setStyleSheet(STYLE_BOLD_LABEL_SM)
        ua_row = QHBoxLayout()
        ua_row.addWidget(QLabel(t('scan.Min')))
        ua_row.addWidget(self.zone_ua_min)
        ua_row.addWidget(QLabel(t('scan.Max')))
        ua_row.addWidget(self.zone_ua_max)
        layout.addRow(t("common.Label_unit_colon", label=t("common.Ua"), unit=t("common.V")), ua_row)

        ug1_row = QHBoxLayout()
        ug1_row.addWidget(QLabel(t('scan.Min')))
        ug1_row.addWidget(self.zone_ug1_min)
        ug1_row.addWidget(QLabel(t('scan.Max')))
        ug1_row.addWidget(self.zone_ug1_max)
        layout.addRow(t("common.Label_unit_colon", label=t("common.Ug1"), unit=t("common.V")), ug1_row)

        # Ug2 + N + Measure checkbox on one row
        ug2_row = QHBoxLayout()
        self._zone_ug2_label = QLabel(t('common.Label_colon', label=t('common.Ug2')))
        ug2_row.addWidget(self._zone_ug2_label)
        ug2_row.addWidget(self.zone_ug2)
        ug2_row.addSpacing(8)
        ug2_row.addWidget(QLabel(t('common.Label_colon', label=t('common.N'))))
        self.srk_repeats = make_int_spinbox(
            min_val=1, max_val=50, value=1,
            fixed_width=45,
            tooltip_key="tip.SRK_repeats",
        )
        ug2_row.addWidget(self.srk_repeats)
        ug2_row.addSpacing(8)
        # Option to measure SRK separately after scan
        self.srk_measure_separately = QCheckBox(t('zone.Measure'))
        self.srk_measure_separately.setToolTip(t('tip.SRK_measure_separately'))
        ug2_row.addWidget(self.srk_measure_separately)
        ug2_row.addSpacing(8)
        # Option to sweep Ug1 with step for higher precision
        self.srk_sweep_cb = QCheckBox(t('zone.Sweep'))
        self.srk_sweep_cb.setToolTip(t('tip.SRK_sweep'))
        ug2_row.addWidget(self.srk_sweep_cb)
        ug2_row.addStretch(1)
        layout.addRow(ug2_row)

        # SRK label + quality + buttons on one row
        srk_row = QHBoxLayout()
        srk_row.addWidget(self.srk_label)
        srk_row.addSpacing(8)
        srk_row.addWidget(self.quality_label)
        srk_row.addStretch(1)
        self.measure_srk_btn = QPushButton(t('zone.Measure'))
        self.measure_srk_btn.setFixedWidth(70)
        self.measure_srk_btn.setToolTip(t("tip.Zone_measure"))
        self.measure_srk_btn.clicked.connect(self._measure_srk)
        srk_row.addWidget(self.measure_srk_btn)
        self.show_srk_points_btn = QPushButton(t('zone.More'))
        self.show_srk_points_btn.setFixedWidth(30)
        self.show_srk_points_btn.setToolTip(t('zone.Show_points'))
        self.show_srk_points_btn.clicked.connect(self._show_srk_results)
        self.show_srk_points_btn.setEnabled(False)
        srk_row.addWidget(self.show_srk_points_btn)
        self.save_manual_btn = QPushButton(t('zone.Save'))
        self.save_manual_btn.setFixedWidth(50)
        self.save_manual_btn.setToolTip(t('tip.SRK_save_manual'))
        self.save_manual_btn.clicked.connect(self._on_save_manual_btn)
        srk_row.addWidget(self.save_manual_btn)
        layout.addRow(srk_row)
        return box

    def _build_actions_group(self) -> QGroupBox:
        box = QGroupBox(t('action.Actions'))
        layout = QHBoxLayout(box)
        layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        layout.setSpacing(SPACING_NORMAL)
        self.run_btn = QPushButton(t('action.Run_scan'))
        self.run_btn.setToolTip(t('tip.Run_scan'))
        self.run_btn.clicked.connect(self._run_scan)
        self.stop_btn = QPushButton(t('action.Stop'))
        self.stop_btn.setToolTip(t('tip.Stop_scan'))
        self.stop_btn.clicked.connect(self._stop_scan)
        self.stop_btn.setEnabled(True)
        self.scan_progress_bar = QProgressBar()
        self.scan_progress_bar.setRange(0, 100)
        self.scan_progress_bar.setValue(0)
        self.scan_progress_bar.setMinimumWidth(120)
        self.scan_progress_bar.setAlignment(Qt.AlignCenter)
        self.ia_samples_spin = make_int_spinbox(
            min_val=1, max_val=10, value=self.app_config.scan_ia_samples,
            fixed_width=45,
            tooltip_key="tip.Samples",
        )
        self.refine_cb = QCheckBox(t('action.Refine'))
        self.refine_cb.setChecked(self.app_config.scan_refine_enabled)
        self.refine_cb.setToolTip(t('tip.Refine'))
        layout.addWidget(self.run_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.scan_progress_bar, 1)
        layout.addWidget(QLabel(t('action.Samples')))
        layout.addWidget(self.ia_samples_spin)
        layout.addWidget(self.refine_cb)
        return box
