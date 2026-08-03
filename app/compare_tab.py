import contextlib
import json
import logging
import math
import re
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.widget_factory import make_double_spinbox, make_int_spinbox
from lm19.app_config import AppConfig
from lm19.io_utils import make_unique_path, write_json
from lm19.measurements import (
    list_measurement_entries, measurement_filename,
    get_ug2_mode, is_entry_triode, is_ug2_track_mode,
)
from lm19.quality import (
    compute_matching, compute_matching_curves, compute_aging_trend,
    detect_dead_data, clean_dead_points,
)
from app.plotting import PlotRenderer
from lm19.curve_data import _cluster_nominal, _nominal_key
from lm19.label_formats import format_label
from app.export_manager import export_csv_multi, export_utd, export_spice, export_pdf
from app.ui_theme import (
    COLOR_IA, COLOR_IG2, COLOR_MID_GRAY, SERIES_PALETTE, STYLE_BOLD_LABEL,
    HEALTH_MATCH_GROUP_COLORS, HEALTH_MATCH_INACTIVE_FG,
    HEALTH_MATCH_UNMATCHED_BG, apply_tight,
    HEALTH_MATCH_DELTA_EXCELLENT, HEALTH_MATCH_DELTA_GOOD,
    HEALTH_MATCH_DELTA_FAIR, HEALTH_MATCH_DELTA_POOR,
    DELTA_QUALITY_COLOR_MAP, DELTA_QUALITY_HEX_MAP,
)
from lm19.tube_matching import (
    CurveMatchResult, CurveDistanceInfo, MatchGroup, MatchCancelled,
    match_curves, build_curve_distance_matrix,
    delta_quality, WARN_OVERLAP_POINTS,
)
from i18n_setup import t
from lm19.constants import UG1_CLUSTER_THR, UG2_CLUSTER_THR, UA_ROUND, UG1_ROUND, UG2_ROUND
from lm19.plot_style import COLOR_ZONE, DEFAULT_LINE_WIDTH, DEFAULT_GRID_ALPHA
from lm19.constants import (
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
)

# Column indices for Sel and Grp (appended after existing base columns).
# Order: Show(0) Type(1) ID(2) Name(3) Mfg(4) An(5) Mode(6) SRK(7) Timestamp(8) Color(9) Sel(10) Grp(11)
_COL_MFG = 4
_COL_AN = 5
_COL_MODE = 6
_COL_SRK = 7
_COL_TS = 8
_COL_COLOR = 9
_COL_SEL = 10
_COL_GRP = 11
_NUM_COLS = 12


@contextlib.contextmanager
def _suspended_resize_to_contents(table):
    """Temporarily flip ResizeToContents columns to Interactive.

    ``QHeaderView.ResizeToContents`` recomputes a column's width on
    every ``setItem``/``setText`` that changes its display value — and
    the recompute scans **all** rows in that column. During bulk
    inserts or per-row updates this turns the operation O(N²) and is
    visible as a freeze on Match/Clear with 50+ entries. We flip the
    affected columns to ``Interactive`` for the duration of the bulk
    operation, then restore — Qt does a single final recompute on
    restore instead of one per row.
    """
    hh = table.horizontalHeader()
    rtc = QHeaderView.ResizeMode.ResizeToContents
    interactive = QHeaderView.ResizeMode.Interactive
    auto_cols = [c for c in range(table.columnCount())
                 if hh.sectionResizeMode(c) == rtc]
    for c in auto_cols:
        hh.setSectionResizeMode(c, interactive)
    try:
        yield auto_cols
    finally:
        for c in auto_cols:
            hh.setSectionResizeMode(c, rtc)


class CompareTab(QWidget):
    """Compare tab for viewing and comparing multiple measurements."""

    # Emitted when user clicks "Show on main plot".
    # Carries (points, series_labels, series_colors, series_ug2_track,
    # scan_meta) — scan_meta is the measurement dict when exactly ONE
    # entry is shown (feeds the PDF scan-settings section), else None.
    show_on_main_plot = Signal(object, object, object, object, object)

    def __init__(self, parent=None, marker_lock_px: int = 15,
                 ug1_cluster_thr: float = UG1_CLUSTER_THR,
                 ug2_cluster_thr: float = UG2_CLUSTER_THR,
                 ia_dead_thr: float = 0.30,
                 match_algorithm: str = "optimal",
                 get_app_config: Optional[Callable[[], AppConfig]] = None):
        super().__init__(parent)
        self._ug1_cluster_thr = ug1_cluster_thr
        self._ug2_cluster_thr = ug2_cluster_thr
        self._ia_dead_thr = ia_dead_thr
        self._get_app_config = get_app_config or (lambda: AppConfig())
        self.compare_entries: List[Dict] = []
        self._compare_sort_column = -1
        self._compare_sort_order = Qt.AscendingOrder
        self._compare_points_data: List[Dict] = []
        # ML-033: Show-checks and user colors survive re-renders. Keyed by
        # id(entry); rebuilt FRESH from the table on every render (no
        # cross-render accumulation — a recycled id() must not inherit a
        # deleted entry's state).
        self._rendered_entries: List[Dict] = []
        self._entry_checks: Dict[int, Qt.CheckState] = {}
        self._entry_colors: Dict[int, str] = {}
        self._compare_legend = None
        self._marker_lock_px = marker_lock_px
        self._match_result: Optional[CurveMatchResult] = None
        # Match info keyed by id(entry): (grp_num, delta, n_points, low_overlap)
        self._match_grp_info: Dict[int, Tuple[int, float, int, bool]] = {}
        self._match_unmatched_ids: set = set()
        # Entries that participated in the last match (for summary/export)
        self._match_entries: List[Dict] = []
        # Re-entrancy guard: the progress dialog calls ``processEvents`` so
        # any other queued event (Match button re-click, Clear, dropdown
        # change) can fire mid-loop. Modality protects us only once the
        # dialog is visible — and it has a 500 ms minimumDuration. The
        # flag closes that window: a second ``_run_curve_matching`` call
        # returns immediately while the first is still in progress.
        self._match_running: bool = False
        self._build_ui()
        # Apply saved algorithm preset to the dropdown built above.
        idx = self.match_algorithm_combo.findData(match_algorithm)
        if idx >= 0:
            self.match_algorithm_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Public API (called by MainWindow)
    # ------------------------------------------------------------------

    def add_entry(self, entry: Dict) -> None:
        """Append a measurement entry and refresh the table."""
        self.compare_entries.append(entry)
        self._render_table(self.compare_entries)

    def clear(self) -> None:
        """Remove all entries and clear the table."""
        self.compare_entries.clear()
        self.table.setRowCount(0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Top-level UI assembly. Builders below populate ``self.*``
        attributes and return their root layout/widget; this method just
        wires everything into the final splitter layout."""
        layout = QVBoxLayout(self)

        controls, match_group_btn = self._build_toolbar()
        match_row = self._build_match_row(match_group_btn)
        filter_row = self._build_filter_row()
        self._build_table()
        self._build_details_panel()
        self._build_plot()
        self._build_match_summary()
        self._build_ug2_panel()

        # Left column: toolbar rows + table (stacked vertically)
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addLayout(controls)
        left_layout.addLayout(match_row)
        left_layout.addLayout(filter_row)
        left_layout.addWidget(self.table, 1)

        # Top splitter: left column | Details (full height)
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(left_col)
        top_splitter.addWidget(self.details)
        top_splitter.setChildrenCollapsible(True)
        top_splitter.setHandleWidth(3)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 0)
        top_splitter.setSizes([700, 260])
        top_splitter.setCollapsible(0, False)
        top_splitter.setCollapsible(1, True)

        # Plot + summary + ug2 horizontal layout
        self._plot_summary_layout = QHBoxLayout()
        self._plot_summary_layout.addWidget(self._match_summary)
        self._plot_summary_layout.addWidget(self.plot, 1)
        self._plot_summary_layout.addWidget(self._ug2_panel)
        plot_container = QWidget()
        plot_container.setLayout(self._plot_summary_layout)

        # Main vertical splitter: top (toolbar+table+details) | bottom (plot)
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top_splitter)
        splitter.addWidget(plot_container)
        splitter.setChildrenCollapsible(True)
        splitter.setHandleWidth(3)
        splitter.setSizes([200, 400])
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, True)
        layout.addWidget(splitter)

    # ── _build_ui sub-builders ──────────────────────────────────────

    def _build_toolbar(self) -> Tuple[QHBoxLayout, QToolButton]:
        """Build the top button row (load/save/match/export + line width).

        Returns (controls_layout, match_group_btn). The match_group_btn is
        re-used by ``_build_match_row`` so it sits in the row below.
        """
        controls = QHBoxLayout()
        reload_btn = QPushButton(t('compare.Reload_measurements'))
        reload_btn.setToolTip(t('tip.Compare_reload'))
        reload_btn.clicked.connect(self._load_measurements)
        load_btn = QPushButton(t('compare.Load_external'))
        load_btn.setToolTip(t('tip.Compare_load_ext'))
        load_btn.clicked.connect(self._load_external_measurement)
        load_dir_btn = QPushButton(t('compare.Load_folder'))
        load_dir_btn.setToolTip(t('tip.Compare_load_folder'))
        load_dir_btn.clicked.connect(self._load_external_folder)
        show_main_btn = QPushButton(t('compare.Show_on_main_plot'))
        show_main_btn.setToolTip(t('tip.Compare_show_main'))
        show_main_btn.clicked.connect(self._show_selected_on_main_plot)
        save_btn = QPushButton(t('compare.Save_selected'))
        save_btn.setToolTip(t('tip.Compare_save'))
        save_btn.clicked.connect(self._save_selected_measurements)
        remove_btn = QPushButton(t('compare.Remove_selected'))
        remove_btn.setToolTip(t('tip.Compare_remove'))
        remove_btn.clicked.connect(self._remove_rows)
        clean_btn = QPushButton(t('dead_data.Clean_btn'))
        clean_btn.setToolTip(t('dead_data.Title'))
        clean_btn.clicked.connect(self._clean_dead_data)

        self.line_width = make_double_spinbox(
            min_val=0.5, max_val=5.0, value=DEFAULT_LINE_WIDTH,
            step=0.5, fixed_width=60,
            tooltip_key='tip.Compare_line_width',
            on_change=self._plot_selected,
        )

        matching_btn = QPushButton(t('compare.Show_matching'))
        matching_btn.setToolTip(t('tip.Compare_matching'))
        matching_btn.clicked.connect(self._show_matching)
        aging_btn = QPushButton(t('compare.Aging_trend'))
        aging_btn.setToolTip(t('tip.Compare_aging'))
        aging_btn.clicked.connect(self._show_aging_trend)
        export_btn = QToolButton()
        export_btn.setText(t('compare.Export'))
        export_btn.setToolTip(t('tip.Compare_export'))
        export_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export_menu = QMenu(export_btn)
        export_menu.addAction(t('csv.Export_CSV'), self._export_csv)
        export_menu.addAction(t('utd.Export_utd'), self._export_utd)
        export_menu.addAction(t('msg.Spice_export'), self._export_spice)
        export_menu.addAction(t('msg.Export_PDF'), self._export_pdf)
        export_btn.setMenu(export_menu)

        controls.addWidget(reload_btn)
        controls.addWidget(load_btn)
        controls.addWidget(load_dir_btn)
        controls.addWidget(show_main_btn)
        controls.addWidget(save_btn)
        controls.addWidget(remove_btn)
        controls.addWidget(clean_btn)

        # Match Groups dropdown + clear button (lifted into match row below)
        match_group_btn = QToolButton()
        match_group_btn.setText(t('compare.Match_groups'))
        match_group_btn.setToolTip(t('tip.Compare_match_groups'))
        match_group_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        match_menu = QMenu(match_group_btn)
        match_menu.addAction(t('compare.Match_all'), lambda: self._run_curve_matching("all"))
        match_menu.addAction(t('compare.Match_visible'), lambda: self._run_curve_matching("visible"))
        match_menu.addAction(t('compare.Match_selected'), lambda: self._run_curve_matching("selected"))
        match_group_btn.setMenu(match_menu)
        self.match_clear_btn = QPushButton(t('compare.Match_clear'))
        self.match_clear_btn.setToolTip(t('tip.Compare_match_clear'))
        self.match_clear_btn.setEnabled(False)
        self.match_clear_btn.clicked.connect(self._clear_curve_match)

        controls.addWidget(matching_btn)
        controls.addWidget(aging_btn)
        controls.addWidget(export_btn)
        controls.addStretch(1)
        controls.addWidget(QLabel(t('plot.Line')))
        controls.addWidget(self.line_width)
        return controls, match_group_btn

    def _build_match_row(self, match_group_btn: QToolButton) -> QHBoxLayout:
        """Build the curve-matching settings row (mode/class/size/Δ/min-pts)."""
        match_row = QHBoxLayout()
        match_row.addWidget(QLabel(t('health.match_Tube_mode')))
        self.match_mode_combo = QComboBox()
        self.match_mode_combo.addItem(t('health.All'), "all")
        self.match_mode_combo.addItem("Pent", TOPOLOGY_PENTODE)
        self.match_mode_combo.addItem("TriC", TOPOLOGY_TRIODE_CONNECTED)
        self.match_mode_combo.addItem("Tri", TOPOLOGY_TRIODE)
        self.match_mode_combo.setToolTip(t('tip.Health_match_tube_mode'))
        match_row.addWidget(self.match_mode_combo)
        match_row.addWidget(QLabel(t('compare.match_Amp_class')))
        self.match_amp_class_combo = QComboBox()
        self.match_amp_class_combo.addItem(t('compare.match_Class_A'), "class_a")
        self.match_amp_class_combo.addItem(t('compare.match_Class_AB'), "class_ab")
        self.match_amp_class_combo.addItem(t('compare.match_Class_B'), "class_b")
        self.match_amp_class_combo.setCurrentIndex(1)  # AB default
        self.match_amp_class_combo.setToolTip(t('tip.Compare_match_amp_class'))
        match_row.addWidget(self.match_amp_class_combo)
        match_row.addWidget(QLabel(t('health.match_Group_size')))
        self.match_group_size_combo = QComboBox()
        self.match_group_size_combo.setEditable(True)
        self.match_group_size_combo.setToolTip(t('tip.Health_match_group_size'))
        for n in range(2, 9):
            self.match_group_size_combo.addItem(str(n), n)
        self.match_group_size_combo.setCurrentIndex(0)
        self.match_group_size_combo.setFixedWidth(50)
        match_row.addWidget(self.match_group_size_combo)

        match_row.addWidget(QLabel(t('health.match_Algorithm')))
        self.match_algorithm_combo = QComboBox()
        self.match_algorithm_combo.addItem(t('health.match_Algo_greedy'), "greedy")
        self.match_algorithm_combo.addItem(t('health.match_Algo_optimal'), "optimal")
        self.match_algorithm_combo.setToolTip(t('tip.Health_match_algorithm'))
        match_row.addWidget(self.match_algorithm_combo)

        match_row.addWidget(QLabel(t('health.match_Max_delta')))
        self.match_max_delta_spin = make_double_spinbox(
            min_val=0.0, max_val=100.0, value=30.0,
            decimals=1, step=0.5, fixed_width=60,
            tooltip_key='tip.Health_match_max_delta',
        )
        match_row.addWidget(self.match_max_delta_spin)

        match_row.addWidget(QLabel(t('compare.match_Min_pts')))
        self.match_min_overlap_spin = make_int_spinbox(
            min_val=3, max_val=100, value=10,
            fixed_width=55, tooltip_key='tip.Compare_match_min_pts',
        )
        match_row.addWidget(self.match_min_overlap_spin)

        match_row.addWidget(match_group_btn)
        match_row.addWidget(self.match_clear_btn)
        match_row.addStretch(1)
        return match_row

    def _build_filter_row(self) -> QHBoxLayout:
        """Build the table-filter row (type / search / group)."""
        filter_row = QHBoxLayout()
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItem(t('health.All'), "all")
        self.filter_type_combo.setToolTip(t('tip.Compare_filter_type'))
        # Resize to longest item so long tube type names are not truncated.
        self.filter_type_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.filter_search = QLineEdit()
        self.filter_search.setPlaceholderText(t('compare.filter_Search_placeholder'))
        self.filter_search.setToolTip(t('tip.Compare_filter_search'))
        self.filter_search.setClearButtonEnabled(True)
        self.filter_group_combo = QComboBox()
        self.filter_group_combo.addItem(t('health.All'), "all")
        self.filter_group_combo.setToolTip(t('tip.Compare_filter_group'))
        self.filter_group_combo.setVisible(False)
        filter_row.addWidget(QLabel(t('compare.filter_Type')))
        filter_row.addWidget(self.filter_type_combo)
        filter_row.addWidget(self.filter_search, 1)
        filter_row.addWidget(self.filter_group_combo)
        self.filter_type_combo.currentIndexChanged.connect(self._apply_compare_filters)
        self.filter_search.textChanged.connect(self._apply_compare_filters)
        self.filter_group_combo.currentIndexChanged.connect(self._apply_compare_filters)
        return filter_row

    def _build_table(self) -> None:
        """Build the main measurements table (populates ``self.table``)."""
        self.table = QTableWidget(0, _NUM_COLS)
        self.table.setHorizontalHeaderLabels([
            t('compare.col_Show'), t('compare.col_Lamp_Type'), t('compare.col_Lamp_ID'),
            t('compare.col_Name'),
            t('common.col_Mfg'),
            t('health.col_An'), t('health.col_Mode'),
            t('compare.col_SRK'), t('compare.col_Timestamp'),
            t('compare.col_Color'),
            t('compare.col_Sel'), t('compare.col_Grp'),
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        # ID and Name stretch to fill remaining space
        header.setSectionResizeMode(2, QHeaderView.Stretch)   # ID
        header.setSectionResizeMode(3, QHeaderView.Stretch)   # Name
        header.setSectionResizeMode(_COL_AN, QHeaderView.Fixed)
        header.setSectionResizeMode(_COL_COLOR, QHeaderView.Fixed)
        header.setSectionResizeMode(_COL_SEL, QHeaderView.Fixed)
        header.setSectionResizeMode(_COL_GRP, QHeaderView.ResizeToContents)
        header.setSortIndicatorShown(True)
        header.sectionClicked.connect(self._sort_table)
        self.table.setColumnWidth(_COL_AN, 28)
        self.table.setColumnWidth(_COL_COLOR, 40)
        self.table.setColumnWidth(_COL_SEL, 28)
        self.table.setColumnHidden(_COL_SEL, True)
        self.table.setColumnHidden(_COL_GRP, True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMinimumWidth(0)
        self.table.setMinimumHeight(95)
        self.table.cellDoubleClicked.connect(self._edit_color)
        self.table.cellClicked.connect(self._on_compare_cell_clicked)
        self.table.itemChanged.connect(self._plot_selected)
        self.table.itemSelectionChanged.connect(self._update_details)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_compare_context_menu)

    def _build_details_panel(self) -> None:
        """Build the right-side details QGroupBox (populates ``self.details``)."""
        self.details = QGroupBox(t('compare.Details'))
        details_layout = QVBoxLayout(self.details)
        details_layout.setContentsMargins(5, 5, 5, 5)
        details_layout.setSpacing(4)

        details_main = QWidget()
        details_main_form = QFormLayout(details_main)
        details_main_form.setContentsMargins(0, 0, 0, 0)

        self.detail_lamp = QLabel("—")
        self.detail_name = QLabel("—")
        self.detail_points = QLabel("—")
        self.detail_ua = QLabel("—")
        self.detail_ug1 = QLabel("—")
        self.detail_ug2 = QLabel("—")
        self.detail_heater = QLabel("—")
        self.detail_zone = QLabel("—")
        self.detail_srk = QLabel("—")
        details_main_form.addRow(t('compare.Lamp'), self.detail_lamp)
        details_main_form.addRow(t('compare.Name'), self.detail_name)
        details_main_form.addRow(t('compare.Points'), self.detail_points)
        details_main_form.addRow(t("common.Label_colon", label=t("common.Ua")), self.detail_ua)
        details_main_form.addRow(t("common.Label_colon", label=t("common.Ug1")), self.detail_ug1)
        self._detail_ug2_label = QLabel(t("common.Label_colon", label=t("common.Ug2")))
        details_main_form.addRow(self._detail_ug2_label, self.detail_ug2)
        details_main_form.addRow(t('compare.Heater'), self.detail_heater)
        details_main_form.addRow(t('compare.Zone'), self.detail_zone)
        details_main_form.addRow(t("compare.SRK"), self.detail_srk)

        details_desc = QWidget()
        details_desc_layout = QVBoxLayout(details_desc)
        details_desc_layout.setContentsMargins(0, 0, 0, 0)
        details_desc_layout.setSpacing(2)
        details_desc_layout.addWidget(QLabel(t('compare.Description')))
        self.detail_description = QPlainTextEdit()
        self.detail_description.setReadOnly(True)
        self.detail_description.setPlaceholderText("—")
        self.detail_description.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.detail_description.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.detail_description.setMinimumHeight(22)  # about one text line
        details_desc_layout.addWidget(self.detail_description)

        details_splitter = QSplitter(Qt.Vertical)
        details_splitter.addWidget(details_main)
        details_splitter.addWidget(details_desc)
        details_splitter.setChildrenCollapsible(False)
        details_splitter.setCollapsible(0, False)
        details_splitter.setCollapsible(1, False)
        # Keep details block compact; give extra height to Description.
        details_splitter.setStretchFactor(0, 0)
        details_splitter.setStretchFactor(1, 1)
        details_splitter.setSizes([260, 28])
        details_layout.addWidget(details_splitter)

        self.details.setMaximumWidth(16777215)
        self.details.setMinimumWidth(0)
        self.details.setMinimumHeight(0)
        self.details.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        details_main.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        details_desc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        details_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _build_plot(self) -> None:
        """Build the Compare plot widget + curve marker."""
        self.plot = pg.PlotWidget(title=t('plot.Compare_Ia_vs_Ua'))
        self.plot.setLabel("left", t('plot.Ia_mA'))
        self.plot.setLabel("bottom", t('plot.Ua_V'))
        self.plot.showGrid(x=True, y=True, alpha=DEFAULT_GRID_ALPHA)

        from app.curve_marker import CurveMarker
        from lm19.curve_data import FIELDS_COMPARE
        self._marker = CurveMarker(self.plot, fields=FIELDS_COMPARE,
                                   lock_px=self._marker_lock_px)

    def _build_match_summary(self) -> None:
        """Build the curve-match summary panel (hidden until matching runs)."""
        self._match_summary = QWidget()
        summary_layout = QVBoxLayout(self._match_summary)
        apply_tight(summary_layout)
        self._match_summary_text = QTextEdit()
        self._match_summary_text.setReadOnly(True)
        self._match_summary_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        summary_layout.addWidget(self._match_summary_text, 1)
        btn_row = QHBoxLayout()
        self._match_copy_btn = QPushButton(t('compare.match_Copy_groups'))
        self._match_export_btn = QPushButton(t('compare.match_Export_CSV'))
        self._match_cert_btn = QPushButton(t('report.Cert_btn'))
        self._match_cert_btn.setToolTip(t('report.Tip_cert_btn'))
        self._match_copy_btn.clicked.connect(self._copy_curve_match_groups)
        self._match_export_btn.clicked.connect(self._export_curve_match_csv)
        self._match_cert_btn.clicked.connect(self._export_match_certificate)
        btn_row.addWidget(self._match_copy_btn)
        btn_row.addWidget(self._match_export_btn)
        btn_row.addWidget(self._match_cert_btn)
        summary_layout.addLayout(btn_row)
        self._match_summary.setFixedWidth(220)
        self._match_summary.setVisible(False)

    def _build_ug2_panel(self) -> None:
        """Build the Ug2 filter panel (right of plot)."""
        self._ug2_panel = QGroupBox("Ug2")
        self._ug2_panel.setFixedWidth(80)
        ug2_panel_layout = QVBoxLayout(self._ug2_panel)
        apply_tight(ug2_panel_layout)
        ug2_btn_row = QHBoxLayout()
        ug2_all_btn = QPushButton(t('health.All'))
        ug2_all_btn.setFixedHeight(20)
        ug2_all_btn.clicked.connect(self._ug2_check_all)
        ug2_none_btn = QPushButton("—")
        ug2_none_btn.setFixedHeight(20)
        ug2_none_btn.clicked.connect(self._ug2_check_none)
        ug2_btn_row.addWidget(ug2_all_btn)
        ug2_btn_row.addWidget(ug2_none_btn)
        ug2_panel_layout.addLayout(ug2_btn_row)
        self._ug2_scroll = QScrollArea()
        self._ug2_scroll.setWidgetResizable(True)
        self._ug2_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._ug2_container = QWidget()
        self._ug2_layout = QVBoxLayout(self._ug2_container)
        apply_tight(self._ug2_layout)
        self._ug2_layout.addStretch(1)
        self._ug2_scroll.setWidget(self._ug2_container)
        ug2_panel_layout.addWidget(self._ug2_scroll, 1)
        # State: nominal Ug2 → checked (True/False), persists across updates
        self._ug2_checked: Dict[float, bool] = {}

    # ------------------------------------------------------------------
    # Table rendering
    # ------------------------------------------------------------------

    def _render_table(self, entries: List[Dict]) -> None:
        # ML-033: any re-render used to reset Show checks and user colors
        # to defaults (only the sort path preserved them, with its own
        # copy of the logic). Snapshot current state, restore in the body.
        self._snapshot_row_state()
        # Suspend ResizeToContents columns for the bulk insert — see
        # ``_suspended_resize_to_contents`` for why this matters.
        self.table.blockSignals(True)
        with _suspended_resize_to_contents(self.table):
            self._render_table_body(entries)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()

    def _snapshot_row_state(self) -> None:
        """Capture Show checks + color cells for entries currently shown."""
        self._entry_checks = {}
        self._entry_colors = {}
        for row, entry in enumerate(self._rendered_entries):
            if row >= self.table.rowCount():
                break
            eid = id(entry)
            show_item = self.table.item(row, 0)
            if show_item is not None:
                self._entry_checks[eid] = show_item.checkState()
            color_item = self.table.item(row, _COL_COLOR)
            if color_item is not None and color_item.text():
                self._entry_colors[eid] = color_item.text()

    def _render_table_body(self, entries: List[Dict]) -> None:
        """Bulk-insert rows into ``self.table``. Always invoked under
        ``_suspended_resize_to_contents`` from ``_render_table``."""
        self.table.setRowCount(0)
        palette = SERIES_PALETTE
        types_seen: set = set()
        for idx, entry in enumerate(entries):
            row = self.table.rowCount()
            self.table.insertRow(row)

            show_item = QTableWidgetItem()
            show_item.setFlags(show_item.flags() | Qt.ItemIsUserCheckable)
            show_item.setCheckState(
                self._entry_checks.get(id(entry), Qt.Unchecked))
            self.table.setItem(row, 0, show_item)

            lamp_type = entry.get("lamp_type", "")
            types_seen.add(lamp_type)
            self.table.setItem(row, 1, QTableWidgetItem(lamp_type))
            self.table.setItem(row, 2, QTableWidgetItem(entry["lamp_id"]))
            self.table.setItem(row, 3, QTableWidgetItem(entry.get("name", "")))

            # An column — check scan.an (new) and conditions.an (health)
            data = entry.get("data", {})
            an = data.get("scan", {}).get("an")
            if an is None:
                an = data.get("conditions", {}).get("an")
            an_item = QTableWidgetItem(str(an) if an else "")
            an_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, _COL_AN, an_item)

            # Mode column
            mode = get_ug2_mode(entry)
            mode_short = {
                TOPOLOGY_PENTODE: t('compare.mode_pent'),
                TOPOLOGY_TRIODE_CONNECTED: t('compare.mode_tric'),
                TOPOLOGY_TRIODE: t('compare.mode_tri'),
            }.get(mode, mode)
            self.table.setItem(row, _COL_MODE, QTableWidgetItem(mode_short))

            # SRK column
            srk = entry.get("data", {}).get("srk", {})
            if srk:
                s = srk.get('s') or 0
                r = srk.get('r') or 0
                k = srk.get('k') or 0
                srk_text = t("compare.SRK_compact", s=f"{s:.2f}", r=f"{r:.2f}", k=f"{k:.2f}")
            else:
                srk_text = "—"
            self.table.setItem(row, _COL_SRK, QTableWidgetItem(srk_text))

            # Timestamp column
            self.table.setItem(row, _COL_TS, QTableWidgetItem(entry["timestamp"]))

            # Mfg date column (YYYY-MM; empty -> em-dash, sorts as text).
            # mfg_date may live at entry root or inside entry["data"]
            # depending on origin (compare add_entry vs list_measurement_entries).
            _d = entry.get("data") or {}
            mfg_val = str(
                entry.get("mfg_date")
                or (_d.get("mfg_date") if isinstance(_d, dict) else "")
                or ""
            )
            mfg_item = QTableWidgetItem(mfg_val if mfg_val else "—")
            if mfg_val:
                mfg_item.setData(Qt.ItemDataRole.EditRole, mfg_val)
            self.table.setItem(row, _COL_MFG, mfg_item)

            # Color column — user-picked color wins over the palette
            stored_color = self._entry_colors.get(id(entry))
            color = (QColor(stored_color) if stored_color
                     else QColor(palette[idx % len(palette)]))
            color_item = QTableWidgetItem(color.name())
            color_item.setBackground(QBrush(color))
            self.table.setItem(row, _COL_COLOR, color_item)

            # Sel and Grp columns (empty by default)
            sel_item = QTableWidgetItem("")
            sel_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, _COL_SEL, sel_item)
            grp_item = QTableWidgetItem("")
            grp_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, _COL_GRP, grp_item)

            self.table.setRowHeight(row, 22)

        self._rendered_entries = list(entries)
        self._update_type_filter(types_seen)

    def _update_type_filter(self, types: set) -> None:
        """Populate lamp type filter combo."""
        self.filter_type_combo.blockSignals(True)
        current = self.filter_type_combo.currentData()
        self.filter_type_combo.clear()
        self.filter_type_combo.addItem(t('health.All'), "all")
        for tp in sorted(types):
            if tp:
                self.filter_type_combo.addItem(tp, tp)
        # Restore selection if still available
        idx = self.filter_type_combo.findData(current)
        if idx >= 0:
            self.filter_type_combo.setCurrentIndex(idx)
        self.filter_type_combo.blockSignals(False)
        # Recompute geometry so the combo fits the longest tube type label.
        self.filter_type_combo.adjustSize()

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def _sort_table(self, column: int) -> None:
        """Sort compare table by clicked column header (except Color, Sel)."""
        if column in (_COL_COLOR, _COL_SEL):
            return

        if self._compare_sort_column == column:
            self._compare_sort_order = (
                Qt.DescendingOrder if self._compare_sort_order == Qt.AscendingOrder
                else Qt.AscendingOrder
            )
        else:
            self._compare_sort_column = column
            self._compare_sort_order = Qt.AscendingOrder

        reverse = self._compare_sort_order == Qt.DescendingOrder

        # ML-033: check/color preservation now lives in _render_table;
        # snapshot HERE too so the Show-column sort key sees the current
        # (pre-sort) check states.
        self._snapshot_row_state()
        if column == 0:
            # Coerce to bool — Qt.CheckState enums are not orderable on PySide6
            # (sorting >=2 rows by the Show column would raise TypeError).
            # Unchecked (False) sorts before Checked (True) on ascending.
            key = lambda e: (self._entry_checks.get(id(e), Qt.Unchecked)
                             == Qt.CheckState.Checked)
        elif column == 1:
            key = lambda e: e.get("lamp_type", "").lower()
        elif column == 2:
            key = lambda e: e.get("lamp_id", "").lower()
        elif column == 3:
            key = lambda e: (e.get("name") or "").lower()
        elif column == _COL_AN:
            # Match the displayed value (scan.an → conditions.an) AND keep the
            # key a str — an codes can be strings ('Au'), so an int default 0
            # would raise TypeError when sorted against them.
            def _an_sort_key(e):
                data = e.get("data", {})
                an = data.get("scan", {}).get("an")
                if an is None:
                    an = data.get("conditions", {}).get("an")
                return str(an) if an else ""
            key = _an_sort_key
        elif column == _COL_MODE:
            key = lambda e: get_ug2_mode(e)
        elif column == _COL_SRK:
            key = lambda e: e.get("data", {}).get("srk", {}).get("s", 0) or 0
        elif column == _COL_TS:
            key = lambda e: e.get("timestamp", "")
        elif column == _COL_MFG:
            # Empty mfg sorts last on ascending (sentinel "~" > any "YYYY-MM").
            # On descending, reverse=True puts empties first — acceptable.
            # mfg_date may live at entry root or inside entry["data"] depending
            # on origin (compare add_entry vs list_measurement_entries).
            def _mfg_sort_key(e):
                d = e.get("data") or {}
                val = (e.get("mfg_date") or
                       (d.get("mfg_date") if isinstance(d, dict) else "") or "")
                return val or "~"
            key = _mfg_sort_key
        elif column == _COL_GRP:
            # Sort by group number (from _match_grp_info keyed by id(entry))
            gi = self._match_grp_info
            key = lambda e: gi.get(id(e), (9999, 0, 0, False))[0]
        else:
            return

        self.compare_entries.sort(key=key, reverse=reverse)
        self._render_table(self.compare_entries)

        self.table.horizontalHeader().setSortIndicator(column, self._compare_sort_order)

        # Re-apply match coloring if active
        if self._match_result is not None:
            self._apply_curve_match_to_table()

    # ------------------------------------------------------------------
    # Details panel
    # ------------------------------------------------------------------

    @staticmethod
    def _format_scan_range(scan: Dict, key: str) -> str:
        """Format a Ua/Ug1/Ug2 scan range as ``"start..stop / step V"``.

        Returns ``"—"`` when the axis dict is empty/missing.
        """
        rng = scan.get(key, {})
        if not rng:
            return "—"
        return t(
            "compare.Range_scan_v",
            start=rng.get("start", 0),
            stop=rng.get("stop", 0),
            step=rng.get("step", 0),
        )

    def _clear_details_panel(self) -> None:
        """Reset all details labels to ``—`` (empty selection state)."""
        for w in (
            self.detail_lamp, self.detail_name, self.detail_points,
            self.detail_ua, self.detail_ug1, self.detail_ug2,
            self.detail_heater, self.detail_zone, self.detail_srk,
        ):
            w.setText("—")
        self.detail_description.setPlainText("—")

    def _update_details_ug2(self, entry: Dict, scan: Dict) -> None:
        """Update Ug2 row: hide for triode, show track/range/dash otherwise."""
        is_entry_triode = self._get_ug2_mode(entry) == TOPOLOGY_TRIODE
        self._detail_ug2_label.setVisible(not is_entry_triode)
        self.detail_ug2.setVisible(not is_entry_triode)
        if is_entry_triode:
            return
        if scan.get("ug2_track_ua", False):
            self.detail_ug2.setText(t('compare.Track_Ua'))
        else:
            self.detail_ug2.setText(self._format_scan_range(scan, "ug2"))

    def _update_details(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows or rows[0].row() >= len(self.compare_entries):
            self._clear_details_panel()
            return

        row = rows[0].row()
        entry = self.compare_entries[row]
        data = entry.get("data", {})
        scan = data.get("scan", {})
        zone = data.get("zone", {})
        srk = data.get("srk", {})
        points = entry.get("points", [])

        self.detail_lamp.setText(f"{entry.get('lamp_type', '—')} / {entry.get('lamp_id', '—')}")
        self.detail_name.setText(entry.get("name", "—") or "—")
        desc = str(data.get("description", "")).strip()
        self.detail_description.setPlainText(desc or "—")
        self.detail_points.setText(str(len(points)))

        self.detail_ua.setText(self._format_scan_range(scan, "ua"))
        self.detail_ug1.setText(self._format_scan_range(scan, "ug1"))
        self._update_details_ug2(entry, scan)

        uh = scan.get("uh", 0)
        ih = scan.get("ih", 0)
        if uh > 0:
            self.detail_heater.setText(t("compare.Heater_Uh", value=uh))
        elif ih > 0:
            self.detail_heater.setText(t("compare.Heater_Ih", value=ih))
        else:
            self.detail_heater.setText("—")

        if zone:
            self.detail_zone.setText(
                t(
                    "compare.Zone_line",
                    ua_min=zone.get("ua_min", 0),
                    ua_max=zone.get("ua_max", 0),
                    ug1_min=zone.get("ug1_min", 0),
                    ug1_max=zone.get("ug1_max", 0),
                )
            )
        else:
            self.detail_zone.setText("—")

        if srk:
            s = float(srk.get("s") or 0)
            r = float(srk.get("r") or 0)
            k = float(srk.get("k") or 0)
            self.detail_srk.setText(t("compare.SRK_line", s=f"{s:.2f}", r=f"{r:.2f}", k=f"{k:.2f}"))
        else:
            self.detail_srk.setText("—")

    # ------------------------------------------------------------------
    # Color editing
    # ------------------------------------------------------------------

    def _edit_color(self, row: int, column: int) -> None:
        if column != _COL_COLOR:
            return
        current = self.table.item(row, _COL_COLOR)
        if not current:
            return
        color = QColorDialog.getColor(QColor(current.text()), self, t('compare.Select_color'))
        if not color.isValid():
            return
        current.setText(color.name())
        current.setBackground(QBrush(color))
        self._plot_selected()

    # ------------------------------------------------------------------
    # Topology / Ug2 mode detection
    # ------------------------------------------------------------------

    @staticmethod
    def _get_ug2_mode(entry: Dict) -> str:
        return get_ug2_mode(entry)

    @staticmethod
    def _is_triode(entry: Dict) -> bool:
        return is_entry_triode(entry)

    @staticmethod
    def _is_ug2_track_mode(entry: Dict, points: List[Dict]) -> bool:
        return is_ug2_track_mode(entry, points)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_selected(self, _item=None) -> None:
        self.plot.clear()
        self._compare_points_data = []
        self._marker.reattach()

        # Reuse ONE legend across replots: pyqtgraph's addLegend()
        # returns the existing legend object WITHOUT re-attaching it,
        # so a remove-then-addLegend cycle leaves the legend detached
        # from the scene (invisible) from the second replot onward.
        # plot.clear() above already dropped entries of removed curves;
        # clear() sweeps the remaining [],[] legend stubs.
        if self._compare_legend is None:
            self._compare_legend = self.plot.addLegend()
        else:
            try:
                self._compare_legend.clear()
            except RuntimeError as exc:
                # Qt wrapper already deleted (C++ side gone) — a visible
                # deviation, so WARNING; recreate from scratch.
                log.warning("Compare legend lost, recreating: %s", exc)
                self.plot.getPlotItem().legend = None
                self._compare_legend = self.plot.addLegend()

        line_width = self.line_width.value()
        selected_data = self._collect_checked_entries()

        # Build Ug2→line-style map from all sweep-mode data (clustered)
        all_ug2_raw: list = []
        for _entry, points, _color, _lamp_id, is_track in selected_data:
            if not is_track and not self._is_triode(_entry):
                for p in points:
                    all_ug2_raw.append(round(p.get("ug2", 0.0), UG2_ROUND))
        all_ug2_noms = _cluster_nominal(sorted(set(all_ug2_raw)),
                                        threshold=self._ug2_cluster_thr)

        line_styles = [Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine, Qt.DashDotDotLine]
        ug2_sorted = sorted(all_ug2_noms)
        ug2_style_map = {ug2: line_styles[i % len(line_styles)] for i, ug2 in enumerate(ug2_sorted)}

        # Update Ug2 filter panel checkboxes
        self._update_ug2_panel(ug2_sorted)
        # Set of enabled Ug2 levels for filtering
        ug2_enabled = {ug2 for ug2 in ug2_sorted if self._ug2_checked.get(ug2, True)}

        # Legend semantics: LAMPS by color (solid pen, one entry per
        # entry) + Ug2 levels by line STYLE in neutral gray. The old
        # legend listed only Ug2 entries colored with the first lamp
        # that happened to hit that level — conflating both axes and
        # leaving the lamps identifiable only via the table.
        label_counts: Dict[str, int] = {}
        for entry, _pts, color, lamp_id, _tr in selected_data:
            label = entry.get("name") or lamp_id or "?"
            n = label_counts.get(label, 0) + 1
            label_counts[label] = n
            if n > 1:
                # Two checked measurements sharing a name (e.g. two runs
                # of one lamp) each get their own plot color — skipping
                # the duplicate would leave the second color unnamed.
                label = f"{label} ({n})"
            self.plot.plot([], [], pen=pg.mkPen(color, width=line_width),
                           name=label)
        if len(ug2_sorted) > 1:
            for ug2 in ug2_sorted:
                # A level unchecked in the filter panel draws no curves
                # — listing its style would advertise absent data. The
                # style MAP still covers all levels, so styles stay
                # stable while toggling.
                if ug2 not in ug2_enabled:
                    continue
                self.plot.plot(
                    [], [],
                    pen=pg.mkPen(COLOR_ZONE, width=line_width,
                                 style=ug2_style_map[ug2]),
                    name=format_label("ug2", ug2))

        labeled_ug1: Dict[str, set] = {}

        for _entry, points, color, _lamp_id, is_track in selected_data:
            if color not in labeled_ug1:
                labeled_ug1[color] = set()

            entry_triode = self._is_triode(_entry)
            if is_track or entry_triode:
                self._plot_entry_track(points, color, line_width, labeled_ug1[color])
            else:
                self._plot_entry_sweep(
                    points, color, line_width, ug2_style_map,
                    labeled_ug1[color],
                    ug2_enabled=ug2_enabled,
                    global_ug2_noms=all_ug2_noms,
                )

        # Filter compare_points_data by enabled Ug2 levels
        if ug2_enabled is not None and ug2_sorted:
            filtered = []
            for p in self._compare_points_data:
                # ML-035: track-mode curves are DRAWN without the Ug2
                # filter (their Ug2 varies with Ua) — filtering their
                # points here starved the snap-to-curve marker.
                if p.get("_is_triode") or p.get("_is_track"):
                    filtered.append(p)
                else:
                    ug2_raw = round(p.get("ug2", 0.0), UG2_ROUND)
                    ug2_nom = _nominal_key(ug2_raw, all_ug2_noms)
                    if ug2_nom in ug2_enabled:
                        filtered.append(p)
            self._compare_points_data = filtered

        # Build CurveData for snap-to-curve marker
        self._marker.set_curves(self._build_compare_curves())

    def _build_compare_curves(self) -> list:
        """Build CurveData from compare points grouped by (lamp, ug1, ug2)."""
        from lm19.curve_data import build_compare_curves
        return build_compare_curves(self._compare_points_data,
                                    ug1_cluster_thr=self._ug1_cluster_thr,
                                    ug2_cluster_thr=self._ug2_cluster_thr)

    def _ug1_nominals(self, points: List[Dict]) -> list:
        """Build Ug1 cluster nominals from a set of points."""
        raw = sorted({round(p.get("ug1", 0.0), UG1_ROUND) for p in points})
        return _cluster_nominal(raw, threshold=self._ug1_cluster_thr)

    def _ug2_nominals(self, points: List[Dict]) -> list:
        """Build Ug2 cluster nominals from a set of points."""
        raw = sorted({round(p.get("ug2", 0.0), UG2_ROUND) for p in points})
        return _cluster_nominal(raw, threshold=self._ug2_cluster_thr)

    # ------------------------------------------------------------------
    # Ug2 filter panel
    # ------------------------------------------------------------------

    def _update_ug2_panel(self, ug2_sorted: List[float]) -> None:
        """Sync Ug2 checkbox list with current levels, preserving state."""
        # Register new levels as checked by default
        for ug2 in ug2_sorted:
            if ug2 not in self._ug2_checked:
                self._ug2_checked[ug2] = True

        # Rebuild checkboxes (block signals to avoid replot loop)
        while self._ug2_layout.count() > 1:
            item = self._ug2_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        from lm19.label_formats import format_label
        for ug2 in ug2_sorted:
            cb = QCheckBox(format_label("ug2_short", ug2))
            cb.setChecked(self._ug2_checked.get(ug2, True))
            # Connect AFTER setChecked to avoid triggering replot
            cb.toggled.connect(lambda checked, u=ug2: self._on_ug2_toggled(u, checked))
            insert_pos = self._ug2_layout.count() - 1  # before stretch
            self._ug2_layout.insertWidget(insert_pos, cb)

    def _on_ug2_toggled(self, ug2: float, checked: bool) -> None:
        """Handle Ug2 checkbox toggle — update state and replot."""
        if self._ug2_checked.get(ug2) == checked:
            return  # no change — avoid replot loop during panel rebuild
        self._ug2_checked[ug2] = checked
        self._plot_selected()

    def _ug2_check_all(self) -> None:
        """Check all Ug2 levels."""
        for ug2 in self._ug2_checked:
            self._ug2_checked[ug2] = True
        self._plot_selected()

    def _ug2_check_none(self) -> None:
        """Uncheck all Ug2 levels."""
        for ug2 in self._ug2_checked:
            self._ug2_checked[ug2] = False
        self._plot_selected()

    def _collect_checked_entries(self) -> List[tuple]:
        """Return list of (entry, points, color, lamp_id, is_track) for checked rows."""
        result = []
        entry_counter = 0
        for row in range(self.table.rowCount()):
            show_item = self.table.item(row, 0)
            if not show_item or show_item.checkState() != Qt.Checked:
                continue
            if row >= len(self.compare_entries):
                continue
            entry = self.compare_entries[row]
            points = entry.get("points", [])
            if not points:
                continue
            color_item = self.table.item(row, _COL_COLOR)
            color = color_item.text() if color_item else COLOR_IA
            lamp_id = entry.get("lamp_id", "")
            is_track = self._is_ug2_track_mode(entry, points)
            result.append((entry, points, color, lamp_id, is_track))

            entry_is_triode = self._is_triode(entry)
            entry_name = entry.get("name") or lamp_id or f"#{entry_counter}"
            for p in points:
                self._compare_points_data.append({
                    **p,
                    "lamp_id": lamp_id,
                    "lamp_type": entry.get("lamp_type", ""),
                    "_is_triode": entry_is_triode,
                    "_is_track": is_track,
                    "_entry_idx": entry_counter,
                    "_entry_name": entry_name,
                })
            entry_counter += 1
        return result

    def _plot_entry_track(self, points: List[Dict], color: str,
                          line_width: float, labeled_ug1: set) -> None:
        """Plot a single entry in track mode (Ug2 follows Ua), grouped by Ug1."""
        ug1_noms = self._ug1_nominals(points)
        groups: Dict[float, List[Dict]] = {}
        for p in points:
            ug1 = _nominal_key(round(p.get("ug1", 0.0), UG1_ROUND), ug1_noms)
            groups.setdefault(ug1, []).append(p)

        for ug1, group_points in groups.items():
            sorted_points = sorted(group_points, key=lambda p: p["ua"])
            if len(sorted_points) < 2:
                continue
            xs = [p["ua"] for p in sorted_points]
            ys = [p["ia"] for p in sorted_points]
            pen = pg.mkPen(color, width=line_width)
            self.plot.plot(xs, ys, pen=pen, symbol="o", symbolSize=4, symbolBrush=color)

            if ug1 not in labeled_ug1 and xs and ys:
                label = pg.TextItem(format_label("ug1_short", ug1),
                                    color=color, anchor=(0, 0.5))
                label.setPos(xs[-1], ys[-1])
                self.plot.addItem(label)
                labeled_ug1.add(ug1)

    def _plot_entry_sweep(self, points: List[Dict], color: str,
                          line_width: float, ug2_style_map: dict,
                          labeled_ug1: set, is_triode: bool = False,
                          ug2_enabled: Optional[set] = None,
                          global_ug2_noms: Optional[list] = None) -> None:
        """Plot a single entry in sweep mode, grouped by (Ug1, Ug2)."""
        ug1_noms = self._ug1_nominals(points)
        ug2_noms = self._ug2_nominals(points) if not is_triode else [0.0]
        groups: Dict[tuple, List[Dict]] = {}
        for p in points:
            ug1 = _nominal_key(round(p.get("ug1", 0.0), UG1_ROUND), ug1_noms)
            if is_triode:
                key = (ug1,)
            else:
                key = (ug1, _nominal_key(round(p.get("ug2", 0.0), UG2_ROUND), ug2_noms))
            groups.setdefault(key, []).append(p)

        for key, group_points in groups.items():
            ug1 = key[0]
            ug2 = key[1] if len(key) > 1 else 0
            # Filter membership and line style operate on GLOBAL cluster
            # nominals: a per-entry nominal inside the cluster threshold
            # of a global level (e.g. 251.5 vs 250.0) must land in the
            # same bucket — otherwise the curve is silently dropped by
            # the Ug2 filter and its style diverges from the legend.
            if global_ug2_noms and not is_triode:
                ug2 = _nominal_key(ug2, global_ug2_noms)
            # Skip Ug2 levels disabled in filter panel
            if ug2_enabled is not None and ug2 not in ug2_enabled and not is_triode:
                continue
            sorted_points = sorted(group_points, key=lambda p: p["ua"])
            if len(sorted_points) < 2:
                continue
            xs = [p["ua"] for p in sorted_points]
            ys = [p["ia"] for p in sorted_points]
            style = ug2_style_map.get(ug2, Qt.SolidLine)
            pen = pg.mkPen(color, width=line_width, style=style)

            # Data curves carry NO legend name — legend entries (lamps
            # by color, Ug2 by neutral-gray style) are built once in
            # _plot_selected with clean semantics.
            self.plot.plot(xs, ys, pen=pen, symbol="o", symbolSize=4, symbolBrush=color)

            if ug1 not in labeled_ug1 and xs and ys:
                label = pg.TextItem(format_label("ug1_short", ug1),
                                    color=color, anchor=(0, 0.5))
                label.setPos(xs[-1], ys[-1])
                self.plot.addItem(label)
                labeled_ug1.add(ug1)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _load_measurements(self) -> None:
        self.compare_entries = list_measurement_entries()
        self._render_table(self.compare_entries)

    def _save_selected_measurements(self) -> None:
        target = QFileDialog.getExistingDirectory(self, t('compare.Select_export_folder'), "")
        if not target:
            return

        saved = 0
        for row in range(self.table.rowCount()):
            show_item = self.table.item(row, 0)
            if not show_item or show_item.checkState() != Qt.Checked:
                continue
            if row >= len(self.compare_entries):
                continue
            entry = self.compare_entries[row]
            data = entry.get("data")
            if not isinstance(data, dict):
                continue
            filename = measurement_filename(data)
            path = make_unique_path(Path(target) / filename)
            write_json(path, data)
            saved += 1

        QMessageBox.information(self, t('msg.Export'), t('msg.Saved_measurements', count=saved))

    def _export_csv(self) -> None:
        """Export checked measurements as CSV via options dialog."""
        entries = self._get_checked_entries()
        if not entries:
            QMessageBox.warning(self, t('csv.Export_CSV'), t('msg.Op_no_data'))
            return
        all_triode = all(self._is_triode(e) for e in entries)
        export_csv_multi(parent=self, entries=entries, is_triode=all_triode)

    # ------------------------------------------------------------------
    # Multi-mode helper
    # ------------------------------------------------------------------

    @staticmethod
    def _ask_multi_mode(parent, title: str) -> Optional[str]:
        """Ask user: combined or separate export. Returns 'combined', 'separate', or None."""
        dlg = QDialog(parent)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(320)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(t('compare.Multi_export_prompt')))
        rb_combined = QRadioButton(t('compare.Multi_combined'))
        rb_separate = QRadioButton(t('compare.Multi_separate'))
        rb_combined.setChecked(True)
        layout.addWidget(rb_combined)
        layout.addWidget(rb_separate)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return "separate" if rb_separate.isChecked() else "combined"

    def _batch_export_dispatch(
        self,
        title: str,
        on_single: Callable[[Dict], None],
        on_combined: Callable[[List[Dict]], None],
        on_separate: Callable[[List[Dict]], None],
    ) -> None:
        """Shared pattern: get checked entries, check empty, ask mode, dispatch."""
        entries = self._get_checked_entries()
        if not entries:
            QMessageBox.warning(self, title, t('msg.Op_no_data'))
            return
        if len(entries) == 1:
            on_single(entries[0])
            return
        mode = self._ask_multi_mode(self, title)
        if mode is None:
            return
        if mode == "combined":
            if not self._check_combined_homogeneity(entries, title):
                return
            on_combined(entries)
        else:
            on_separate(entries)

    def _check_combined_homogeneity(self, entries: List[Dict],
                                    title: str) -> bool:
        """ML-079 (hybrid policy): a combined export
        blindly concatenates points and labels them with entries[0]'s
        type/mode. Mixed ug2_mode → BLOCK (physically incompatible
        surfaces, no valid merge exists); mixed lamp_type → warn with the
        breakdown + Yes/No (deliberate analog merges like 6P14P ~ EL84
        stay possible). Returns True to proceed."""
        from collections import Counter

        def _fmt(counter: Counter) -> str:
            return ", ".join(f"{k} ({n})" for k, n in counter.most_common())

        modes = Counter(self._get_ug2_mode(e) for e in entries)
        if len(modes) > 1:
            log.warning("Combined export blocked — mixed Ug2 modes: %s",
                        _fmt(modes))
            QMessageBox.warning(
                self, title,
                t('msg.Combined_mixed_modes', modes=_fmt(modes)))
            return False
        types = Counter(str(e.get("lamp_type", "unknown")) for e in entries)
        if len(types) > 1:
            log.warning("Combined export mixes lamp types: %s", _fmt(types))
            reply = QMessageBox.question(
                self, title,
                t('msg.Combined_mixed_types',
                  types=_fmt(types),
                  name=str(entries[0].get("lamp_type", "unknown"))),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return reply == QMessageBox.StandardButton.Yes
        return True

    # ------------------------------------------------------------------
    # UTD export
    # ------------------------------------------------------------------

    def _export_utd(self) -> None:
        """Export checked measurements as uTracer .utd files."""
        def _single(e: Dict) -> None:
            export_utd(parent=self, points=e.get("points", []),
                       tube_type=e.get("lamp_type", "unknown"))

        def _combined(entries: List[Dict]) -> None:
            all_points: List[Dict] = []
            for e in entries:
                all_points.extend(e.get("points", []))
            export_utd(parent=self, points=all_points,
                       tube_type=entries[0].get("lamp_type", "unknown"))

        self._batch_export_dispatch(
            t('utd.Export_title'), _single, _combined, self._export_utd_separate)

    def _batch_export(
        self,
        entries: List[Dict],
        title: str,
        ext: str,
        export_fn,
    ) -> None:
        """Common batch-export loop: pick folder, export each entry, report."""
        folder = QFileDialog.getExistingDirectory(self, title)
        if not folder:
            return
        count = 0
        errors: List[str] = []
        for e in entries:
            pts = e.get("points", [])
            if not pts:
                continue
            tube = e.get("lamp_type", "unknown")
            lamp_id = e.get("lamp_id", "")
            fname = self._safe_stem(tube, lamp_id) + ext
            fpath = str(make_unique_path(Path(folder) / fname))
            try:
                export_fn(e, pts, fpath)
                count += 1
            except Exception as exc:
                log.exception("Batch export failed for %s", fname)
                errors.append(f"{fname}: {exc}")
        msg = t('compare.Export_batch_done', count=count, path=folder)
        if errors:
            msg += "\n\n" + "\n".join(errors)
        QMessageBox.information(self, title, msg)

    def _export_utd_separate(self, entries: List[Dict]) -> None:
        """Export each entry as a separate .utd file into a folder."""
        from lm19.utracer_export import format_utd
        from app.export_manager import UtdExportDialog

        n_ua = n_ug1 = 0
        for e in entries:
            pts = e.get("points", [])
            n_ua = max(n_ua, len(set(round(p.get("ua", 0), UA_ROUND) for p in pts)))
            n_ug1 = max(n_ug1, len(set(round(p.get("ug1", 0), 2) for p in pts)))

        dlg = UtdExportDialog(self, n_ua=n_ua, n_ug1=n_ug1)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        chosen_fmt = dlg.fmt

        def _do_export(_e, pts, fpath):
            content = format_utd(pts, fmt=chosen_fmt)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)

        self._batch_export(entries, t('utd.Export_title'), ".utd", _do_export)

    @staticmethod
    def _safe_stem(tube: str, lamp_id: str) -> str:
        s = f"{tube}_{lamp_id}" if lamp_id else tube
        return "".join(c if c.isalnum() or c in "_-." else "_" for c in s).strip("_") or "export"

    # ------------------------------------------------------------------
    # SPICE export
    # ------------------------------------------------------------------

    def _export_spice(self) -> None:
        """Export SPICE model from checked measurements."""
        def _mfg_of(e: Dict) -> str:
            d = e.get("data") or {}
            return str(
                e.get("mfg_date")
                or (d.get("mfg_date") if isinstance(d, dict) else "")
                or ""
            )

        def _single(e: Dict) -> None:
            export_spice(parent=self, points=e.get("points", []),
                         tube_type=e.get("lamp_type", "unknown"),
                         topology=self._get_ug2_mode(e),
                         mfg_date=_mfg_of(e))

        def _combined(entries: List[Dict]) -> None:
            all_points: List[Dict] = []
            for e in entries:
                all_points.extend(e.get("points", []))
            if not all_points:
                QMessageBox.warning(self, t('msg.Spice_export'), t('msg.Spice_no_data'))
                return
            # entries[0] is the representative (matches tube_type/mfg_date below)
            export_spice(parent=self, points=all_points,
                         tube_type=entries[0].get("lamp_type", "unknown"),
                         topology=self._get_ug2_mode(entries[0]),
                         mfg_date=_mfg_of(entries[0]))

        self._batch_export_dispatch(
            t('msg.Spice_export'), _single, _combined, self._export_spice_separate)

    def _export_spice_separate(self, entries: List[Dict]) -> None:
        """Fit and save a separate .sub for each entry into a folder."""
        from lm19.spice_export import fit_and_export_spice
        from app.export_manager import _ask_spice_model

        model_type = _ask_spice_model(self)
        if model_type is None:
            return

        def _do_export(e, pts, fpath):
            d = e.get("data") or {}
            mfg = str(
                e.get("mfg_date")
                or (d.get("mfg_date") if isinstance(d, dict) else "")
                or ""
            )
            fit_and_export_spice(fpath, e.get("lamp_type", "unknown"), pts,
                                 topology=self._get_ug2_mode(e),
                                 model_type=model_type, mfg_date=mfg)

        self._batch_export(entries, t('msg.Spice_export'), ".sub", _do_export)

    # ------------------------------------------------------------------
    # PDF export
    # ------------------------------------------------------------------

    def _export_pdf(self) -> None:
        """Export PDF report for checked measurements."""
        def _mfg_of(e: Dict) -> str:
            d = e.get("data") or {}
            return str(
                e.get("mfg_date")
                or (d.get("mfg_date") if isinstance(d, dict) else "")
                or ""
            )

        def _single(e: Dict) -> None:
            data = e.get("data", {})
            srk_list = [data["srk"]] if isinstance(data, dict) and data.get("srk") else []
            export_pdf(
                parent=self, points=e.get("points", []),
                tube_type=e.get("lamp_type", "unknown"),
                lamp_id=e.get("lamp_id", ""),
                lamp=None, srk_results=srk_list,
                plot_renderer=None, plot_widget=self.plot,
                mfg_date=_mfg_of(e),
                config=self._get_app_config(),
                scan_meta=data if isinstance(data, dict) else None,
            )

        def _combined(entries: List[Dict]) -> None:
            all_points: List[Dict] = []
            srk_list: List[Dict] = []
            for e in entries:
                all_points.extend(e.get("points", []))
                data = e.get("data", {})
                if isinstance(data, dict) and data.get("srk"):
                    srk_list.append(data["srk"])
            export_pdf(
                parent=self, points=all_points,
                tube_type=entries[0].get("lamp_type", "unknown"),
                lamp_id=entries[0].get("lamp_id", ""),
                lamp=None, srk_results=srk_list,
                plot_renderer=None, plot_widget=self.plot,
                mfg_date=_mfg_of(entries[0]),
                config=self._get_app_config(),
                # merged points from several lamps — no single scan meta
                scan_meta=None,
            )

        self._batch_export_dispatch(
            t('msg.Export_PDF'), _single, _combined, self._export_pdf_separate)

    def _export_match_certificate(self) -> None:
        """Matched pair/quad PDF certificate for a curve-match group."""
        from app.export_manager import render_group_overlay_pixmap
        from app.match_certificate import (
            CERT_SECTIONS,
            build_compare_cert_fragments,
            generate_certificate_pdf,
            pick_match_group,
        )
        from app.report_options_dialog import ask_report_options
        from i18n_setup import translator_for

        result = self._match_result
        group = pick_match_group(self, result)
        if group is None:
            return
        available = {"cert_conditions": "", "cert_metrics": "",
                     "cert_plot": ""}
        opts = ask_report_options(self, available, self._get_app_config(),
                                  specs=CERT_SECTIONS, session_key="cert")
        if opts is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, t("report.Cert_btn"), "", t("msg.PDF_filter"))
        if not path:
            return
        tr = translator_for(opts.language)
        entries = self._match_entries or []
        indices = [i for i in ((r.entry or {}).get("_index")
                               for r in group.records)
                   if i is not None and i < len(entries)]
        first = entries[indices[0]] if indices else {}
        tube_type = str(first.get("lamp_type", "") or "?")
        amp_class = self.match_amp_class_combo.currentData() or "class_ab"
        # the plot shows ONLY the group's lamps — a compare-plot
        # screenshot would leak every other checked lamp into the document
        members = [(entries[i].get("lamp_id", ""),
                    entries[i].get("points", [])) for i in indices]
        try:
            generate_certificate_pdf(
                path,
                fragments=build_compare_cert_fragments(
                    group, tube_type=tube_type, entries=entries,
                    pair_info=result.pair_info, amp_class=amp_class,
                    sections=opts.sections, tr=tr),
                image=(render_group_overlay_pixmap(members)
                       if "cert_plot" in opts.sections and members
                       else None),
                image_caption=tr("report.Sec_cert_plot"),
                tr=tr)
            QMessageBox.information(self, t("report.Cert_btn"),
                                    t("msg.PDF_saved", path=path))
        except Exception as exc:
            log.exception("Certificate export failed")
            QMessageBox.critical(self, t("report.Cert_btn"),
                                 t("msg.PDF_error", error=str(exc)))

    def _export_pdf_separate(self, entries: List[Dict]) -> None:
        """Generate a separate PDF report for each entry into a folder."""
        from app.export_manager import render_points_pixmap
        from app.report import generate_pdf_report
        from app.report_options_dialog import ask_report_options

        def _data_of(e: Dict) -> Optional[Dict]:
            data = e.get("data")
            return data if isinstance(data, dict) else None

        has_srk = any((_data_of(e) or {}).get("srk") for e in entries)
        available = {
            "nominal": "report.Na_no_lamp",
            "scan_settings": "",
            "srk": "" if has_srk else "report.Na_no_srk",
            "quality": "report.Na_no_quality",
            "distortion": "report.Na_no_analysis",
            "plot_curves": "",
            "plot_transfer": "report.Na_no_image",
        }
        opts = ask_report_options(self, available, self._get_app_config())
        if opts is None:
            return

        def _do_export(e, pts, fpath):
            data = e.get("data", {})
            srk = data.get("srk") if isinstance(data, dict) else None
            mfg = str(
                e.get("mfg_date")
                or (data.get("mfg_date") if isinstance(data, dict) else "")
                or ""
            )
            # per-lamp offscreen render: each PDF carries ITS OWN curves,
            # not a screenshot of every selected lamp
            plot_pixmap = (
                render_points_pixmap(pts, title=str(e.get("lamp_id", "")))
                if "plot_curves" in opts.sections else None)
            generate_pdf_report(
                path=fpath,
                tube_type=e.get("lamp_type", "unknown"),
                lamp_id=e.get("lamp_id", ""),
                points=pts,
                srk=srk,
                plot_image=plot_pixmap,
                mfg_date=mfg,
                sections=opts.sections,
                language=opts.language,
                scan_meta=_data_of(e),
            )

        self._batch_export(entries, t('msg.Export_PDF'), ".pdf", _do_export)

    def _show_selected_on_main_plot(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.warning(self, t('msg.Plot'), t('msg.Select_row_for_plot'))
            return
        selected_points: List[Dict] = []
        series_labels: Dict[int, str] = {}
        series_colors: Dict[int, str] = {}
        series_ug2_track: Dict[int, bool] = {}
        series_counter = 1  # 0 is reserved for current scan

        shown_metas: List[Optional[Dict]] = []
        for sel in rows:
            row = sel.row()
            if row >= len(self.compare_entries):
                continue
            entry = self.compare_entries[row]
            points = entry.get("points", [])
            if not points:
                continue
            data = entry.get("data")
            shown_metas.append(data if isinstance(data, dict) else None)

            entry_label = entry.get("name") or entry.get("lamp_id") or t("compare.Entry_fallback", index=row + 1)
            lamp_type = entry.get("lamp_type", "")
            lamp_id = entry.get("lamp_id", "")
            color_item = self.table.item(row, _COL_COLOR)
            entry_color = color_item.text() if color_item else None

            ug2_mode = self._get_ug2_mode(entry)
            is_track = ug2_mode in (TOPOLOGY_TRIODE, TOPOLOGY_TRIODE_CONNECTED)

            sid = series_counter
            series_labels[sid] = entry_label
            if entry_color:
                series_colors[sid] = entry_color
            series_ug2_track[sid] = is_track
            for p in points:
                pt = dict(p)
                pt["series_id"] = sid
                pt["lamp_type"] = lamp_type
                pt["lamp_id"] = lamp_id
                selected_points.append(pt)
            series_counter += 1

        if not selected_points:
            QMessageBox.warning(self, t('msg.Plot'), t('msg.No_points_in_selection'))
            return
        scan_meta = shown_metas[0] if len(shown_metas) == 1 else None
        self.show_on_main_plot.emit(
            selected_points, series_labels, series_colors, series_ug2_track,
            scan_meta)

    def _remove_rows(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()}, reverse=True)
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self.compare_entries):
                self.compare_entries.pop(row)
        self._render_table(self.compare_entries)

    @staticmethod
    def _build_dead_data_confirm_msg(entry: Dict, report) -> str:
        """Build the per-entry confirmation message shown in the dead-data dialog."""
        name = entry.get("name", entry.get("lamp_id", ""))
        msg_lines = [f"{name}:"]
        if report.dead_ug2_levels:
            levels_str = ", ".join(f"{v:.0f} V" for v in report.dead_ug2_levels)
            msg_lines.append(t("dead_data.Dead_levels", levels=levels_str))
        if report.partial_ug2_levels:
            parts = [f"Ug2={ug2:.0f} V (Ug1>={ug1:.1f} V)"
                     for ug2, ug1 in report.partial_ug2_levels]
            msg_lines.append(t("dead_data.Partial_levels", levels=", ".join(parts)))
        msg_lines.append(
            t("dead_data.Summary",
              dead=str(report.dead_points),
              total=str(report.total_points),
              pct=f"{report.dead_pct:.0f}"))
        return "\n".join(msg_lines) + "\n\n" + t(
            "dead_data.Clean_confirm",
            dead=str(report.dead_points),
            total=str(report.total_points),
        )

    @staticmethod
    def _persist_cleaned_entry(entry: Dict, cleaned: List[Dict]) -> bool:
        """Write cleaned points back to the entry + its source JSON file.

        Returns True when the on-disk file was overwritten (used only for
        debug counters; missing path / failed write returns False).
        """
        entry["points"] = cleaned
        data = entry.get("data")
        if isinstance(data, dict):
            data["points"] = cleaned

        file_path = entry.get("path")
        if not file_path:
            # External entries carry no source file — the cleanup is
            # in-memory only; the caller surfaces this (ML-080).
            log.warning("Cleaned entry %r has no source path — not persisted",
                        entry.get("name", entry.get("lamp_id", "?")))
            return False
        try:
            p = Path(file_path)
            if p.exists() and isinstance(data, dict):
                write_json(p, data)
                return True
        except (OSError, TypeError, ValueError) as exc:
            # Narrow except (failure-visibility pr.1): programming errors
            # must propagate, disk/data errors are reported to the caller.
            log.warning("Failed to overwrite %s: %s", file_path, exc)
        return False

    def _process_dead_entry(self, entry: Dict) -> Tuple[int, int, bool]:
        """Detect/clean dead points for one entry.

        Returns (removed, remaining, persisted). ``persisted`` is False when
        points were removed but the change did NOT reach the disk (no source
        path / write failure) — the caller must surface that (ML-080), else
        the cleanup silently reverts on the next reload.

        Pops a confirmation dialog if dead points are present. ``remaining``
        always reflects the post-action point count (or pre-clean count if
        the user declined).
        """
        points = entry.get("points", [])
        if not points:
            return 0, 0, True

        topology = get_ug2_mode(entry)
        report = detect_dead_data(
            points, ia_thr=self._ia_dead_thr, topology=topology,
        )
        if not report.has_dead_data:
            return 0, len(points), True

        confirm_msg = self._build_dead_data_confirm_msg(entry, report)
        reply = QMessageBox.question(
            self, t('dead_data.Title'), confirm_msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return 0, len(points), True

        cleaned = clean_dead_points(points, report=report)
        removed = len(points) - len(cleaned)
        persisted = self._persist_cleaned_entry(entry, cleaned)
        return removed, len(cleaned), persisted

    def _clean_dead_data(self) -> None:
        """Detect and remove dead points from checked measurements, overwriting files."""
        checked = self._get_checked_entries()
        if not checked:
            QMessageBox.information(
                self, t('dead_data.Title'), t('dead_data.No_dead_data'))
            return

        total_removed = 0
        total_remaining = 0
        unsaved: List[str] = []

        for entry in checked:
            removed, remaining, persisted = self._process_dead_entry(entry)
            total_removed += removed
            total_remaining += remaining
            if removed > 0 and not persisted:
                unsaved.append(
                    str(entry.get("name", entry.get("lamp_id", "?"))))

        if total_removed > 0:
            self._render_table(self.compare_entries)
            self._plot_selected()
            msg = t('dead_data.Clean_done',
                    removed=str(total_removed),
                    remaining=str(total_remaining))
            if unsaved:
                # ML-080: in-memory-only cleanups revert on the next reload
                # — the user must know which entries did not hit the disk.
                msg += chr(10) + t('dead_data.Clean_not_persisted',
                                   names=", ".join(unsaved))
            QMessageBox.information(self, t('dead_data.Title'), msg)
        else:
            QMessageBox.information(
                self, t('dead_data.Title'), t('dead_data.No_dead_data'))

    def _load_external_measurement(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            t('compare.Open_measurement_JSON'),
            "",
            t('msg.JSON_filter'),
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            log.exception("Failed to load compare file %s", path)
            QMessageBox.warning(self, t('msg.Load'), t('msg.Failed_to_load', error=exc))
            return

        entries: List[Dict] = []
        if isinstance(data, dict):
            entries.append(self._normalize_external_entry(data, path))
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    entries.append(self._normalize_external_entry(item, path, idx))
        else:
            QMessageBox.warning(self, t('msg.Load'), t('msg.Unsupported_JSON'))
            return

        self.compare_entries.extend(entries)
        self._render_table(self.compare_entries)

    def _load_external_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, t('compare.Open_measurements_folder'), "")
        if not folder:
            return

        entries: List[Dict] = []
        for file_path in Path(folder).glob("*.json"):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                log.warning("Failed to load %s", file_path, exc_info=True)
                continue
            if isinstance(data, dict):
                entries.append(self._normalize_external_entry(data, str(file_path)))
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        entries.append(self._normalize_external_entry(item, str(file_path), idx))
        if not entries:
            QMessageBox.warning(self, t('msg.Load'), t('msg.No_JSON_in_folder'))
            return
        self.compare_entries.extend(entries)
        self._render_table(self.compare_entries)

    @staticmethod
    def _normalize_external_entry(data: Dict, path: str, index: Optional[int] = None) -> Dict:
        file_stem = Path(path).stem
        name = str(data.get("name", "")) or file_stem
        if index is not None:
            name = f"{name} #{index + 1}"
        return {
            "lamp_type": str(data.get("tube_type", t("import.Default_tube_type"))),
            "lamp_id": str(data.get("lamp_id", file_stem)),
            "timestamp": str(data.get("timestamp", "")),
            "name": name,
            "points": data.get("points", []),
            "data": data,
        }

    def add_imported_entries(self, entries: List[Dict]) -> None:
        """Programmatically add pre-built entries (from Import menu)."""
        self.compare_entries.extend(entries)
        self._render_table(self.compare_entries)

    # ------------------------------------------------------------------
    # Matching dialog
    # ------------------------------------------------------------------

    def _get_checked_entries(self) -> List[Dict]:
        """Return entries whose Show checkbox is checked."""
        checked = []
        for row in range(self.table.rowCount()):
            show_item = self.table.item(row, 0)
            if show_item and show_item.checkState() == Qt.Checked:
                if row < len(self.compare_entries):
                    checked.append(self.compare_entries[row])
        return checked

    def _show_matching(self) -> None:
        """Show matching delta dialog for exactly 2 checked measurements."""
        checked = self._get_checked_entries()
        if len(checked) != 2:
            QMessageBox.warning(self, t('plot.Matching_delta'),
                                t('msg.Select_two_for_matching'))
            return

        pts_a = checked[0].get("points", [])
        pts_b = checked[1].get("points", [])
        name_a = f"{checked[0].get('lamp_type', '')} {checked[0].get('lamp_id', '')}"
        name_b = f"{checked[1].get('lamp_type', '')} {checked[1].get('lamp_id', '')}"

        # Compute matching score
        match_result = compute_matching(pts_a, pts_b,
                                        ug1_cluster_thr=self._ug1_cluster_thr)
        if match_result is None:
            QMessageBox.warning(self, t('plot.Matching_delta'),
                                t('msg.Not_enough_matching_data'))
            return

        # Compute delta curves
        curves = compute_matching_curves(pts_a, pts_b,
                                         ug1_cluster_thr=self._ug1_cluster_thr)
        if not curves:
            QMessageBox.warning(self, t('plot.Matching_delta'),
                                t('msg.Not_enough_matching_data'))
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(t('msg.Matching_title', a=name_a, b=name_b))
        dlg.resize(700, 500)
        layout = QVBoxLayout(dlg)

        # Score label
        delta_pct = 100.0 - match_result.match_pct
        score_text = t('msg.Matching_score',
            pct=f"{match_result.match_pct:.1f}",
            delta=f"{delta_pct:.1f}",
            mean=f"{match_result.mean_delta:.2f}",
            max=f"{match_result.max_delta:.2f}",
            rms=f"{match_result.rms_delta:.2f}",
            n=str(match_result.n_points),
        )
        score_label = QLabel(score_text)
        score_label.setStyleSheet(STYLE_BOLD_LABEL)
        layout.addWidget(score_label)

        # Absolute delta plot
        abs_plot = pg.PlotWidget(title=t('plot.Delta_Ia_mA'))
        abs_plot.showGrid(x=True, y=True, alpha=DEFAULT_GRID_ALPHA)
        PlotRenderer.render_matching_dialog(abs_plot, curves, mode="absolute",
                                            line_width=self.line_width.value())
        layout.addWidget(abs_plot)

        # Percentage delta plot
        pct_plot = pg.PlotWidget(title=t('plot.Delta_Ia_pct'))
        pct_plot.showGrid(x=True, y=True, alpha=DEFAULT_GRID_ALPHA)
        PlotRenderer.render_matching_dialog(pct_plot, curves, mode="percent",
                                            line_width=self.line_width.value())
        layout.addWidget(pct_plot)

        dlg.exec()

    # ------------------------------------------------------------------
    # Aging trend dialog
    # ------------------------------------------------------------------

    def _show_aging_trend(self) -> None:
        """Show aging trend for checked measurements of the same lamp."""
        checked = self._get_checked_entries()
        if len(checked) < 2:
            QMessageBox.warning(self, t('compare.Aging_trend'),
                                t('msg.Same_lamp_for_aging'))
            return

        # Verify same lamp type and ID
        lamp_types = {e.get("lamp_type", "") for e in checked}
        lamp_ids = {e.get("lamp_id", "") for e in checked}
        if len(lamp_types) > 1 or len(lamp_ids) > 1:
            QMessageBox.warning(self, t('compare.Aging_trend'),
                                t('msg.Same_lamp_for_aging'))
            return

        measurements = [e.get("data", {}) for e in checked]
        trend = compute_aging_trend(measurements)

        if not trend:
            QMessageBox.warning(self, t('compare.Aging_trend'),
                                t('msg.No_aging_data'))
            return

        lamp_label = f"{list(lamp_types)[0]} / {list(lamp_ids)[0]}"
        dlg = QDialog(self)
        dlg.setWindowTitle(t('msg.Aging_title', lamp=lamp_label))
        dlg.resize(700, 500)
        layout = QVBoxLayout(dlg)

        # X axis: measurement index (or timestamp)
        x_vals = list(range(len(trend)))
        x_labels = [ap.timestamp[:10] if ap.timestamp else str(i) for i, ap in enumerate(trend)]

        # Ia at operating point
        ia_vals = [ap.ia_at_op for ap in trend]
        s_vals = [ap.s for ap in trend]

        has_ia = any(v is not None for v in ia_vals)
        has_s = any(v is not None for v in s_vals)

        if has_ia:
            ia_plot = pg.PlotWidget(title=t("compare.Ia_at_op"))
            ia_plot.setLabel("left", t("plot.Ia_mA"))
            ia_plot.setLabel("bottom", t("compare.Measurement"))
            ia_plot.showGrid(x=True, y=True, alpha=DEFAULT_GRID_ALPHA)
            xs = [i for i, v in enumerate(ia_vals) if v is not None]
            ys = [v for v in ia_vals if v is not None]
            ia_plot.plot(xs, ys, pen=pg.mkPen(COLOR_IG2, width=2),
                        symbol="o", symbolSize=8, symbolBrush=COLOR_IG2)
            # Add timestamp labels
            for i, lbl_text in enumerate(x_labels):
                if i in xs:
                    lbl = pg.TextItem(lbl_text, color=COLOR_MID_GRAY, anchor=(0.5, 1.2))
                    lbl.setFont(pg.QtGui.QFont("", 7))
                    idx_in_xs = xs.index(i)
                    lbl.setPos(i, ys[idx_in_xs])
                    ia_plot.addItem(lbl)
            layout.addWidget(ia_plot)

        if has_s:
            s_plot = pg.PlotWidget(title=t("compare.S_transconductance"))
            s_plot.setLabel("left", t("compare.S_mA_V"))
            s_plot.setLabel("bottom", t("compare.Measurement"))
            s_plot.showGrid(x=True, y=True, alpha=DEFAULT_GRID_ALPHA)
            xs = [i for i, v in enumerate(s_vals) if v is not None]
            ys = [v for v in s_vals if v is not None]
            s_plot.plot(xs, ys, pen=pg.mkPen(COLOR_IA, width=2),
                       symbol="o", symbolSize=8, symbolBrush=COLOR_IA)
            for i, lbl_text in enumerate(x_labels):
                if i in xs:
                    lbl = pg.TextItem(lbl_text, color=COLOR_MID_GRAY, anchor=(0.5, 1.2))
                    lbl.setFont(pg.QtGui.QFont("", 7))
                    idx_in_xs = xs.index(i)
                    lbl.setPos(i, ys[idx_in_xs])
                    s_plot.addItem(lbl)
            layout.addWidget(s_plot)

        if not has_ia and not has_s:
            layout.addWidget(QLabel(t('msg.No_aging_data')))

        dlg.exec()

    # ------------------------------------------------------------------
    # Curve-based matching (group / find similar)
    # ------------------------------------------------------------------

    def _get_matching_entries(self, source: str) -> List[Dict]:
        """Return entries for matching based on source mode."""
        if source == "selected":
            rows = {idx.row() for idx in self.table.selectedIndexes()}
            return [self.compare_entries[r] for r in sorted(rows)
                    if r < len(self.compare_entries)]
        if source == "visible":
            out = []
            for row in range(self.table.rowCount()):
                if not self.table.isRowHidden(row) and row < len(self.compare_entries):
                    out.append(self.compare_entries[row])
            return out
        return list(self.compare_entries)  # "all"

    def _match_curves_with_progress(self, entries, labels, **match_kwargs):
        """Run :func:`match_curves` behind a modal QProgressDialog.

        Shared by group matching and Find similar (ML-142: the latter
        used to call match_curves with no progress/Cancel). Returns the
        result, or ``None`` if the user cancelled.

        QProgressDialog avoids a worker thread: ``setValue`` internally
        calls ``processEvents`` so the dialog repaints and Cancel
        responds while the heavy loop runs on the main thread. The
        window-modal dialog blocks Match/Clear/dropdown for the whole
        run; ``_match_running`` is defensive backup for the brief
        pre-show window.
        """
        dlg = QProgressDialog(
            t('compare.match_Progress'), t('compare.match_Cancel'),
            0, max(1, len(entries)), self,
        )
        dlg.setWindowTitle(t('compare.Match_groups'))
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        # ``%v / %m  %p%`` = value, max, percent (Qt placeholders).
        _bar = dlg.findChild(QProgressBar)
        if _bar is not None:
            _bar.setFormat("%v / %m  %p%")
        dlg.setValue(0)
        dlg.show()

        def _on_progress(done: int, total: int) -> bool:
            dlg.setMaximum(total)
            dlg.setValue(done)
            return not dlg.wasCanceled()

        self._match_running = True
        try:
            return match_curves(entries, labels, progress=_on_progress,
                                **match_kwargs)
        except MatchCancelled:
            return None
        finally:
            dlg.close()
            self._match_running = False

    def _run_curve_matching(self, source: str = "all") -> None:
        """Run curve-based matching and update table/summary."""
        # See ``_match_running`` docstring above for why this guard is here.
        if self._match_running:
            return
        entries = self._get_matching_entries(source)

        # Filter by mode if selected
        mode_filter = self.match_mode_combo.currentData()
        if mode_filter and mode_filter != "all":
            entries = [e for e in entries
                       if get_ug2_mode(e) == mode_filter]

        if len(entries) < 2:
            QMessageBox.information(self, t('compare.Match_groups'),
                                    t('compare.match_No_data'))
            return

        labels = [
            f"{e.get('lamp_type', '')} {e.get('lamp_id', '')}".strip()
            for e in entries
        ]

        # Read settings from match row
        group_size_text = self.match_group_size_combo.currentText()
        try:
            group_size = max(2, int(group_size_text))
        except ValueError:
            group_size = 2
        max_delta = self.match_max_delta_spin.value()
        min_overlap = self.match_min_overlap_spin.value()
        amp_class = self.match_amp_class_combo.currentData() or "class_ab"
        algorithm = self.match_algorithm_combo.currentData() or "optimal"

        result = self._match_curves_with_progress(
            entries, labels, mode="groups",
            group_size=group_size, max_delta=max_delta,
            min_overlap=min_overlap, amp_class=amp_class,
            algorithm=algorithm)
        if result is None:   # cancelled
            return

        self._match_result = result
        self._build_match_grp_info(result, entries)

        # Show columns
        self.table.setColumnHidden(_COL_SEL, False)
        self.table.setColumnHidden(_COL_GRP, False)
        self.match_clear_btn.setEnabled(True)

        self._apply_curve_match_to_table()
        self._update_curve_match_summary()
        self._update_curve_group_filter()
        self._match_summary.setVisible(True)

    def _build_match_grp_info(self, result: CurveMatchResult,
                              entries: List[Dict]) -> None:
        """Build match lookup dicts keyed by id(entry) — stable across sorts."""
        self._match_grp_info.clear()
        self._match_unmatched_ids.clear()
        self._match_entries = entries

        for g in result.groups:
            indices = [(r.entry or {}).get("_index") for r in g.records]
            indices = [i for i in indices if i is not None]
            # Get pair info
            n_pts = 0
            low_ov = False
            if result.mode == "similar" and result.anchor_idx is not None:
                for idx in indices:
                    pk = (min(result.anchor_idx, idx), max(result.anchor_idx, idx))
                    if pk in result.pair_info:
                        n_pts = result.pair_info[pk].n_points
                        low_ov = result.pair_info[pk].low_overlap
            elif len(indices) == 2:
                pk = (min(indices[0], indices[1]), max(indices[0], indices[1]))
                if pk in result.pair_info:
                    n_pts = result.pair_info[pk].n_points
                    low_ov = result.pair_info[pk].low_overlap

            for idx in indices:
                if idx < len(entries):
                    eid = id(entries[idx])
                    self._match_grp_info[eid] = (g.number, g.delta, n_pts, low_ov)

        # Anchor entry gets grp_num=0 so it sorts first
        if result.anchor_idx is not None and result.anchor_idx < len(entries):
            eid = id(entries[result.anchor_idx])
            self._match_grp_info[eid] = (0, 0.0, 0, False)

        for idx in result.unmatched:
            if idx < len(entries):
                self._match_unmatched_ids.add(id(entries[idx]))

    def _apply_curve_match_to_table(self) -> None:
        """Update Sel/Grp columns and row colors from curve match result."""
        result = self._match_result
        if result is None:
            return

        # Per-row setText on the Grp column (ResizeToContents) triggers a
        # full-column width recalc on every iteration → O(N²). Suspend
        # RTC for the whole apply loop and let Qt recompute once at the
        # end. Same fix as ``_render_table``.
        self.table.blockSignals(True)
        amp_class = self.match_amp_class_combo.currentData() or "class_ab"

        with _suspended_resize_to_contents(self.table):
            self._apply_curve_match_to_table_body(result, amp_class)

        self.table.blockSignals(False)

    def _apply_curve_match_to_table_body(self, result, amp_class: str) -> None:
        """Inner loop of ``_apply_curve_match_to_table``; always runs under
        ``_suspended_resize_to_contents``."""
        for row in range(self.table.rowCount()):
            if row >= len(self.compare_entries):
                continue

            entry = self.compare_entries[row]
            eid = id(entry)

            # ML-036: wash the previous match's fill FIRST — a row that
            # left its group otherwise keeps the stale coloring forever.
            for c in range(self.table.columnCount()):
                item = self.table.item(row, c)
                if item and c != _COL_COLOR:
                    item.setBackground(QBrush())

            sel_item = self.table.item(row, _COL_SEL)
            if sel_item is None:
                sel_item = QTableWidgetItem()
                sel_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, _COL_SEL, sel_item)

            grp_item = self.table.item(row, _COL_GRP)
            if grp_item is None:
                grp_item = QTableWidgetItem()
                grp_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, _COL_GRP, grp_item)

            info = self._match_grp_info.get(eid)
            if info is not None:
                grp_num, delta, n_pts, low_ov = info
                sel_item.setText("●")
                if grp_num == 0:
                    # Anchor entry in similar mode
                    grp_item.setText("★")
                else:
                    pts_str = f"{n_pts}{t('compare.match_Points_suffix')}"
                    warn = " ⚠" if low_ov else ""
                    if result.mode == "similar":
                        star = "★" if grp_num == 1 else ""
                        grp_item.setText(f"#{grp_num} (Δ={delta:.1f}%, {pts_str}){warn}{star}")
                    else:
                        star = "★" if grp_num == 1 else ""
                        grp_item.setText(f"{grp_num} (Δ={delta:.1f}%, {pts_str}){warn}{star}")
                grp_item.setData(Qt.ItemDataRole.UserRole + 1, delta)

                q = delta_quality(delta, amp_class=amp_class)
                grp_item.setForeground(QBrush(
                    DELTA_QUALITY_COLOR_MAP.get(q, HEALTH_MATCH_DELTA_POOR)))

                if grp_num > 0:
                    bg = HEALTH_MATCH_GROUP_COLORS[
                        (grp_num - 1) % len(HEALTH_MATCH_GROUP_COLORS)]
                    for c in range(self.table.columnCount()):
                        item = self.table.item(row, c)
                        if item and c != _COL_COLOR:
                            item.setBackground(QBrush(bg))
            elif eid in self._match_unmatched_ids:
                sel_item.setText("●")
                grp_item.setText("—")
                for c in range(self.table.columnCount()):
                    item = self.table.item(row, c)
                    if item and c != _COL_COLOR:
                        item.setBackground(QBrush(HEALTH_MATCH_UNMATCHED_BG))
            else:
                sel_item.setText("")
                grp_item.setText("")

    def _update_curve_match_summary(self) -> None:
        """Populate the match summary panel."""
        result = self._match_result
        amp_class = self.match_amp_class_combo.currentData() or "class_ab"

        if result is None:
            self._match_summary_text.setHtml("")
            return

        entries = self._match_entries
        parts = []

        n_groups = len(result.groups)
        n_unmatched = len(result.unmatched)
        header = (f"<b>{t('compare.match_Summary')}: {n_groups} × 2</b>"
                  + (f", {t('compare.match_Unmatched_label')}: {n_unmatched}"
                     if n_unmatched else ""))
        parts.append(header)

        for g in result.groups:
            indices = [(r.entry or {}).get("_index") for r in g.records]
            indices = [i for i in indices if i is not None]
            names = []
            n_pts = 0
            low_ov = False
            for idx in indices:
                if idx < len(entries):
                    e = entries[idx]
                    names.append(e.get('lamp_id', '?'))
                    info = self._match_grp_info.get(id(e))
                    if info:
                        n_pts = info[2]
                        low_ov = info[3]
            star = " ★" if g.number == 1 else ""
            warn = " ⚠" if low_ov else ""
            q = delta_quality(g.delta, amp_class=amp_class)
            fg = DELTA_QUALITY_HEX_MAP.get(q, DELTA_QUALITY_HEX_MAP["poor"])
            parts.append(
                f'<span style="color:{fg}">'
                f"<b>{t('compare.match_Group_prefix')} {g.number}{star}</b> "
                f"Δ={g.delta:.1f}% ({n_pts}{t('compare.match_Points_suffix')})"
                f"{warn}</span><br/>"
                f"&nbsp;&nbsp;{' + '.join(names)}")

        if result.unmatched:
            um_names = []
            for idx in result.unmatched:
                if idx < len(entries):
                    um_names.append(entries[idx].get('lamp_id', '?'))
            fg = HEALTH_MATCH_INACTIVE_FG.name()
            parts.append(
                f'<span style="color:{fg}">— '
                f"{t('compare.match_Unmatched_label')}: {', '.join(um_names)}"
                f"</span>")

        self._match_summary_text.setHtml("<br/>".join(parts))

    def _update_curve_group_filter(self) -> None:
        """Populate group filter combo from curve match result."""
        result = self._match_result
        self.filter_group_combo.blockSignals(True)
        self.filter_group_combo.clear()
        self.filter_group_combo.addItem(t('health.All'), "all")
        if result and result.groups:
            entries = self._match_entries
            for g in result.groups:
                indices = [(r.entry or {}).get("_index") for r in g.records]
                names = []
                for idx in indices:
                    if idx is not None and idx < len(entries):
                        names.append(entries[idx].get('lamp_id', '?'))
                self.filter_group_combo.addItem(
                    f"Group {g.number} ({', '.join(names)})", f"g{g.number}")
            if result.unmatched:
                self.filter_group_combo.addItem(
                    t('compare.filter_Unmatched'), "unmatched")
        self.filter_group_combo.blockSignals(False)
        self.filter_group_combo.setVisible(
            result is not None and bool(result.groups))

    def _clear_curve_match(self) -> None:
        """Clear matching results and restore normal table view."""
        # Don't race with a Match in progress: it would re-render the
        # table after we've reset state. User should cancel via the
        # progress dialog if they want to abort.
        if self._match_running:
            return
        self._match_result = None
        self._match_grp_info.clear()
        self._match_unmatched_ids.clear()
        self._match_entries = []
        self.table.setColumnHidden(_COL_SEL, True)
        self.table.setColumnHidden(_COL_GRP, True)
        self.match_clear_btn.setEnabled(False)
        self._match_summary.setVisible(False)
        self._update_curve_group_filter()
        self._render_table(self.compare_entries)

    def _apply_compare_filters(self) -> None:
        """Hide rows that don't match type, regex, or group filters."""
        type_filter = self.filter_type_combo.currentData()
        pattern_text = self.filter_search.text().strip()
        try:
            regex = re.compile(pattern_text, re.IGNORECASE) if pattern_text else None
        except re.error:
            regex = None
        group_filter = self.filter_group_combo.currentData()

        for row in range(self.table.rowCount()):
            if row >= len(self.compare_entries):
                continue
            entry = self.compare_entries[row]
            eid = id(entry)
            visible = True

            # Type filter
            if type_filter and type_filter != "all":
                if entry.get("lamp_type", "") != type_filter:
                    visible = False

            # Regex filter on lamp_id and name
            if visible and regex:
                lamp_id = str(entry.get("lamp_id", ""))
                name = str(entry.get("name", ""))
                if not (regex.search(lamp_id) or regex.search(name)):
                    visible = False

            # Group filter
            if visible and group_filter and group_filter != "all":
                info = self._match_grp_info.get(eid)
                if group_filter == "unmatched":
                    if eid not in self._match_unmatched_ids:
                        visible = False
                elif group_filter.startswith("g"):
                    grp_num = int(group_filter[1:])
                    if info is None or info[0] != grp_num:
                        visible = False

            self.table.setRowHidden(row, not visible)

    def _on_compare_context_menu(self, pos) -> None:
        """Context menu for compare table: Find similar."""
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        if not rows:
            return
        menu = QMenu(self)
        if len(rows) == 1:
            find_action = menu.addAction(t('compare.ctx_Find_similar'))
            find_action.triggered.connect(
                lambda: self._find_similar_curve(rows[0]))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _find_similar_curve(self, row: int) -> None:
        """Find similar curves to the entry at the given row."""
        if row >= len(self.compare_entries):
            return

        # Auto-detect mode from anchor entry
        anchor = self.compare_entries[row]
        anchor_mode = get_ug2_mode(anchor)
        # Set combo to match anchor mode
        idx = self.match_mode_combo.findData(anchor_mode)
        if idx >= 0:
            self.match_mode_combo.setCurrentIndex(idx)

        # Filter entries by same mode
        entries = []
        anchor_idx_in_filtered = 0
        for j, e in enumerate(self.compare_entries):
            if get_ug2_mode(e) == anchor_mode:
                if j == row:
                    anchor_idx_in_filtered = len(entries)
                entries.append(e)

        labels = [
            f"{e.get('lamp_type', '')} {e.get('lamp_id', '')}".strip()
            for e in entries
        ]

        amp_class = self.match_amp_class_combo.currentData() or "class_ab"
        # ML-142: honor the Max Δ / Min pts settings from the Match row
        # (they were silently ignored) and run behind the same progress+
        # Cancel dialog as group matching.
        result = self._match_curves_with_progress(
            entries, labels, mode="similar",
            anchor_idx=anchor_idx_in_filtered,
            max_delta=self.match_max_delta_spin.value(),
            min_overlap=self.match_min_overlap_spin.value(),
            amp_class=amp_class)
        if result is None:   # cancelled
            return
        self._match_result = result
        self._build_match_grp_info(result, entries)

        self.table.setColumnHidden(_COL_SEL, False)
        self.table.setColumnHidden(_COL_GRP, False)
        self.match_clear_btn.setEnabled(True)

        self._apply_curve_match_to_table()
        self._update_curve_match_summary()
        self._update_curve_group_filter()
        self._match_summary.setVisible(True)

    def _on_compare_cell_clicked(self, row: int, col: int) -> None:
        """Handle click on Sel column (no-op for now, reserved)."""
        pass

    def _copy_curve_match_groups(self) -> None:
        """Copy curve match groups to clipboard."""
        result = self._match_result
        if not result:
            return
        from PySide6.QtWidgets import QApplication
        entries = self._match_entries
        lines = []
        for g in result.groups:
            indices = [(r.entry or {}).get("_index") for r in g.records]
            names = [f"{entries[i].get('lamp_type', '')} {entries[i].get('lamp_id', '')}"
                     for i in indices if i is not None and i < len(entries)]
            lines.append(f"Group {g.number}: {' + '.join(names)} Δ={g.delta:.1f}%")
        if result.unmatched:
            um = [f"{entries[i].get('lamp_type', '')} {entries[i].get('lamp_id', '')}"
                  for i in result.unmatched if i < len(entries)]
            lines.append(f"Unmatched: {', '.join(um)}")
        QApplication.clipboard().setText("\n".join(lines))

    def _export_curve_match_csv(self) -> None:
        """Export curve match groups to CSV."""
        result = self._match_result
        if not result:
            return
        import csv
        path, _ = QFileDialog.getSaveFileName(
            self, t('compare.match_Export_CSV'), "", "CSV (*.csv)")
        if not path:
            return
        entries = self._match_entries
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Group", "Lamp Type", "Lamp ID", "Name", "Δ (%)",
                         "Points", "Low Overlap"])
            for g in result.groups:
                indices = [(r.entry or {}).get("_index") for r in g.records]
                indices = [i for i in indices if i is not None]
                n_pts = 0
                low_ov = False
                for i in indices:
                    if i < len(entries):
                        info = self._match_grp_info.get(id(entries[i]))
                        if info:
                            n_pts = info[2]
                            low_ov = info[3]
                            break
                for i in indices:
                    if i < len(entries):
                        e = entries[i]
                        w.writerow([
                            g.number, e.get("lamp_type", ""),
                            e.get("lamp_id", ""), e.get("name", ""),
                            f"{g.delta:.1f}", n_pts,
                            "⚠" if low_ov else "",
                        ])
            for i in result.unmatched:
                if i < len(entries):
                    e = entries[i]
                    w.writerow([
                        "—", e.get("lamp_type", ""),
                        e.get("lamp_id", ""), e.get("name", ""),
                        "", "", "",
                    ])
