import datetime as dt
import csv
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.app_context import AppContext

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QSplitter,
    QFileDialog,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ui_theme import HEALTH_HISTORY_HIGHLIGHT_COLS
from app.widget_factory import FormattedNumericItem as _FormattedNumericItem
from app.widget_factory import TitleRowButtonGroupBox
from app.health_history import (
    FILTER_ALL,
    entry_matches_filter,
    verdict_color,
    table_to_tsv,
    extract_row_values,
    populate_history_table,
    build_match_entry_info,
    build_match_active,
    build_matching_conditions,
    format_match_groups_text,
    format_match_csv_rows,
    COL_SEL,
    COL_GRP,
)
from app.lamp_panel import LampPanel, check_heater_level
from app.match_panel import MatchPanel
from app.ui_theme import (
    apply_tight, apply_no_margin, MARGIN, SPACING_NORMAL,
    HEALTH_STEP_OP, HEALTH_STEP_SRK, HEALTH_STEP_EMISSION_100, HEALTH_STEP_EMISSION_80,
    HEALTH_VERDICT_STRONG_BG, HEALTH_VERDICT_GOOD_BG, HEALTH_VERDICT_WEAK_BG, HEALTH_VERDICT_REPLACE_BG,
    HEALTH_HISTORY_LAMP_ID_WIDTH, HEALTH_HISTORY_NAME_WIDTH, HEALTH_HISTORY_REF_WIDTH,
    HEALTH_HISTORY_CONDITION_COLS,
    HEALTH_STEPS_COL_WIDTHS,
    HEALTH_SPLITTER_SIZES, HEALTH_MIN_SECTION_SIZE,
    HEALTH_PROGRESS_OP, HEALTH_PROGRESS_SRK, HEALTH_PROGRESS_UH80, HEALTH_PROGRESS_SRK_SPAN,
    HEALTH_PREHEAT_POLL_MS,
    HEALTH_PREHEAT_TIMEOUT_FACTOR,
    HEALTH_PREHEAT_TIMEOUT_MARGIN_S,
    HEALTH_MATCH_GROUP_COLORS, HEALTH_MATCH_INACTIVE_FG, HEALTH_MATCH_UNMATCHED_BG,
    HEALTH_MATCH_DELTA_EXCELLENT, HEALTH_MATCH_DELTA_GOOD,
    HEALTH_MATCH_DELTA_FAIR, HEALTH_MATCH_DELTA_POOR,
    DELTA_QUALITY_COLOR_MAP,
    STYLE_INPUT_ERROR, STYLE_INPUT_WARN, STYLE_STATUS_ERROR, STYLE_STATUS_WARN, STYLE_STATUS_OK,
)
from app.workers import HealthWorker
from app.health_protection_dialog import show_health_protection_dialog
from app.live_panel import LivePanel
from app.save_recovery import save_with_recovery, suggested_filename
from i18n_setup import t
from lm19.config import LampConfig, find_lamp
from lm19.config import DEFAULT_LIMITS
from lm19.protocol import UA_RESOLUTION_V, UG1_RESOLUTION_V, UG2_RESOLUTION_V
from lm19.health import (
    clamp_delta_ua, clamp_delta_ug1, clamp_delta_ug2,
    compute_shifted_r_center, compute_shifted_sg2_center,
    BIAS_SERVO_DISABLED, BIAS_SERVO_NO_REFERENCE, BIAS_SERVO_OK,
    BIAS_SERVO_UNREACHABLE, STEP_BIAS_SERVO, STEP_BIAS_SERVO_OP,
    STEP_BIAS_SERVO_RESTORE, KNEE_CONF_LOW,
    EMISSION_VERDICT_EXHAUSTED, EMISSION_VERDICT_NA,
    EMISSION_VERDICT_NORMAL, EMISSION_VERDICT_WEAKENED,
    HEALTH_VERDICT_GOOD, HEALTH_VERDICT_NA, HEALTH_VERDICT_ORDER,
    HEALTH_VERDICT_REPLACE, HEALTH_VERDICT_STRONG, HEALTH_VERDICT_WEAK,
)
from lm19.health_measurements import list_health_entries, save_health_measurement, delete_health_measurement
from lm19.tube_matching import (
    match_tubes, select_measurements, MatchResult, _conditions_key, _extract_record,
    _conditions_compatible,
    default_weights_for_mode, delta_quality, DEFAULT_MATCHING_PROTOCOL,
    ANCHOR_ERR_INCOMPATIBLE, ANCHOR_ERR_NOT_FOUND,
)
from lm19.label_formats import format_label
from lm19.constants import EPS_COARSE, HEATER_NEAR_ZERO_V, HEATER_NEAR_ZERO_A
from lm19.health_refs import (
    build_reference_from_measurement,
    get_active_type_ref,
    list_type_refs,
    load_personal_baseline,
    load_type_ref,
    resolve_reference,
    save_personal_baseline,
    save_type_ref,
    set_active_type_ref,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
)
from lm19.constants import (
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)

# Column indices — imported from app.health_history
_COL_SEL = COL_SEL
_COL_GRP = COL_GRP

# ── module local constants ──
# Emission-verdict code -> i18n key. Every code in EMISSION_VERDICTS must
# appear here, and every key must exist in every locale (pinned).
EMISSION_VERDICT_KEYS = {
    EMISSION_VERDICT_NORMAL: "health.Emission_verdict_normal",
    EMISSION_VERDICT_WEAKENED: "health.Emission_verdict_weakened",
    EMISSION_VERDICT_EXHAUSTED: "health.Emission_verdict_exhausted",
    EMISSION_VERDICT_NA: "health.Emission_verdict_na",
}


def emission_verdict_text(code: object) -> str:
    """Translate an emission-verdict code; unknown codes degrade to N/A."""
    key = EMISSION_VERDICT_KEYS.get(str(code))
    if key is None:
        log.warning("Unknown emission verdict code: %r", code)
        return t(EMISSION_VERDICT_KEYS[EMISSION_VERDICT_NA])
    return t(key)


# Composite verdict code -> i18n key. The codes are persisted and
# filtered on as strings, so they stay English in the data; only the
# rendering is localized.
HEALTH_VERDICT_KEYS = {
    HEALTH_VERDICT_STRONG: "quality.Strong",
    HEALTH_VERDICT_GOOD: "quality.Good",
    HEALTH_VERDICT_WEAK: "quality.Weak",
    HEALTH_VERDICT_REPLACE: "quality.Replace",
    HEALTH_VERDICT_NA: "quality.NA",
}


def health_verdict_text(code: object) -> str:
    """Translate a composite verdict; unknown codes degrade to N/A."""
    key = HEALTH_VERDICT_KEYS.get(str(code))
    if key is None:
        log.warning("Unknown health verdict code: %r", code)
        return t(HEALTH_VERDICT_KEYS[HEALTH_VERDICT_NA])
    return t(key)


def populate_verdict_filter_combo(combo: QComboBox) -> None:
    """Fill the history verdict filter: translated labels, raw codes as data.

    Best-first reads naturally in a filter. The stored data stays the raw
    code so the filter keeps working across a locale change, and so it
    can be compared against what ``_verdict`` produced.
    """
    combo.clear()
    combo.addItem(t("health.All"), FILTER_ALL)
    for code in reversed(HEALTH_VERDICT_ORDER):
        combo.addItem(health_verdict_text(code), code)


# Bias-servo status -> i18n key. Same bijection requirement as above.
BIAS_SERVO_STATUS_KEYS = {
    BIAS_SERVO_OK: "health.Bias_servo_ok",
    BIAS_SERVO_DISABLED: "health.Bias_servo_disabled",
    BIAS_SERVO_NO_REFERENCE: "health.Bias_servo_no_reference",
    BIAS_SERVO_UNREACHABLE: "health.Bias_servo_unreachable",
}


def bias_servo_status_text(code: object) -> str:
    """Translate a bias-servo status; unknown codes degrade to 'disabled'."""
    key = BIAS_SERVO_STATUS_KEYS.get(str(code))
    if key is None:
        log.warning("Unknown bias servo status: %r", code)
        return t(BIAS_SERVO_STATUS_KEYS[BIAS_SERVO_DISABLED])
    return t(key)


# Anchor-error code -> i18n key. Same bijection requirement as above
# (registry: lm19.tube_matching.MATCH_ANCHOR_ERRORS).
ANCHOR_ERROR_KEYS = {
    ANCHOR_ERR_NOT_FOUND: "health.match_Anchor_missing",
    ANCHOR_ERR_INCOMPATIBLE: "health.match_Anchor_incompatible",
}


class HealthTab(QWidget):
    def __init__(
        self,
        ctx: "AppContext",
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.ctx = ctx
        # Delegate attribute access so existing code (self.get_client() etc.) keeps working.
        self.get_client = ctx.get_client
        self.get_write_locked = ctx.get_write_locked
        self.get_hw_busy = ctx.get_hw_busy
        self.get_calibration = ctx.get_calibration
        self.get_app_config = ctx.get_app_config
        self.get_lamps = ctx.get_lamps
        self.get_current_tube_type = ctx.get_current_tube_type
        self.get_current_lamp_id = ctx.get_current_lamp_id
        self.set_poller_active = ctx.set_poller_active
        self.get_preheat_enabled = ctx.get_preheat_enabled
        self.get_preheat_done = ctx.get_preheat_done
        self.request_start_preheat = ctx.request_start_preheat
        self.request_stop_all = ctx.request_stop_all
        self.request_stop_keep_heater = ctx.request_stop_keep_heater
        self.on_load_to_manual: Optional[Callable[[Dict], None]] = None

        self.worker: Optional[HealthWorker] = None
        self.last_measurement: Optional[Dict] = None
        self._ignore_filter_signal = False
        self._pending_start_after_preheat = False
        self._preheat_wait_timer: Optional[QTimer] = None
        self._preheat_start_ts: float = 0.0
        self._preheat_warmup_s: int = 0
        self._reset_requested = False
        self._history_entries: List[Dict] = []
        # ML-143: cache of the last disk read, reused by the filter path
        # so Lamp ID keystrokes / filter changes don't re-read all JSON.
        self._all_history_entries: List[Dict] = []
        self._match_result: Optional[MatchResult] = None
        # Rows participating in the current match, by row identity
        # (lamp_id, timestamp) — feeds the ● marker, dimming and
        # "Hide inactive". Rebuilt from every match result.
        self._match_active: Set[Tuple[str, str]] = set()
        # User's measurement picks (B3): (lamp_id, an) -> timestamp pinned
        # via the Sel column. Survives recalculation — it is an INPUT of
        # the match (entries are pre-filtered to honour it), never rebuilt
        # from the result.
        self._match_pick: Dict[Tuple[str, int], str] = {}
        self._match_anchor_lamp_id: Optional[str] = None
        # When set, "Find similar" anchors on this SPECIFIC measurement (by
        # timestamp) instead of the lamp's latest/best.
        self._match_anchor_timestamp: Optional[str] = None
        # Anode system of the clicked entry — a twin-triode's An2 row must
        # not anchor on the lamp's An1 record.
        self._match_anchor_an: Optional[int] = None
        self._build_ui()
        self.refresh_lamps()

    def shutdown(self) -> None:
        """Stop running worker and disconnect signals. Call from closeEvent."""
        if self.worker and self.worker.isRunning():
            self.worker.cleanup()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        root.setSpacing(SPACING_NORMAL)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        right = QWidget()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes(HEALTH_SPLITTER_SIZES)

        left_layout = QVBoxLayout(left)
        apply_no_margin(left_layout)

        self._left_tabs = QTabWidget()
        measure_widget = QWidget()
        measure_layout = QVBoxLayout(measure_widget)
        apply_no_margin(measure_layout)
        self._build_left_panel(measure_layout)
        self._left_tabs.addTab(measure_widget, t("health.match_tab_Measure"))

        self.match_panel = MatchPanel()
        # Preselect saved pair-matching algorithm from health.json
        self.match_panel.set_algorithm(self.get_app_config().health_matching_algorithm)
        self.match_panel.set_protocol(self.get_app_config().health_matching_protocol)
        self._left_tabs.addTab(self.match_panel, t("health.match_tab_Match"))

        left_layout.addWidget(self._left_tabs)

        right_layout = QVBoxLayout(right)
        apply_no_margin(right_layout)
        self._build_right_panel(right_layout)

        root.addWidget(splitter, 1)
        self._connect_signals()

    # --- Left panel sub-builders ---

    def _build_left_panel(self, layout: QVBoxLayout) -> None:
        self.lamp_panel = LampPanel(name_label=t("health.Name"), show_name=True)
        self.tube_combo = self.lamp_panel.tube_combo
        self.lamp_id_edit = self.lamp_panel.lamp_id_edit
        self.name_edit = self.lamp_panel.name_edit
        self.anode_group = self.lamp_panel.anode_group

        health_opts_box = self._build_health_opts()
        plan_box = self._build_plan_box()

        self.live_panel = LivePanel(title=t("health.Live"), sep=": ", layout_mode="compact")
        self.live_state = QLabel(t("health.State_idle"))
        actions_box = self._build_actions_box()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setToolTip(t("tip.Health_progress"))
        self.progress_label = QLabel(t("health.Progress_idle"))
        result_box = self._build_result_box()

        layout.addWidget(self.lamp_panel)
        layout.addWidget(health_opts_box)
        layout.addWidget(plan_box)
        layout.addWidget(self.live_panel)
        layout.addWidget(actions_box)
        # One status row instead of three stacked ones: state and phase
        # labels keep their natural width, the bar absorbs the leftover.
        status_row = QHBoxLayout()
        status_row.setSpacing(SPACING_NORMAL)
        status_row.addWidget(self.live_state)
        status_row.addWidget(self.progress, 1)
        status_row.addWidget(self.progress_label)
        layout.addLayout(status_row)
        layout.addWidget(result_box)
        layout.addStretch(1)

    def _build_health_opts(self) -> QGroupBox:
        box = QGroupBox(t("health.Setup"))
        form = QFormLayout(box)
        apply_tight(form)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(t("health.RefMode_datasheet"), "datasheet")
        self.mode_combo.addItem(t("health.RefMode_type"), "type")
        self.mode_combo.addItem(t("health.RefMode_personal"), "personal")
        self.mode_combo.setToolTip(t("tip.Health_reference_mode"))
        self.type_ref_combo = QComboBox()
        self.type_ref_combo.setToolTip(t("tip.Health_type_ref"))
        self.emission_enabled = QCheckBox(t("health.Emission_enabled"))
        self.emission_enabled.setChecked(True)
        self.emission_enabled.setToolTip(t("tip.Health_emission_enabled"))
        self.auto_preheat_cb = QCheckBox(t("health.Auto_preheat_if_needed"))
        self.auto_preheat_cb.setChecked(True)
        self.auto_preheat_cb.setToolTip(t("tip.Health_auto_preheat"))
        self.baseline_info = QLabel(t("health.Baseline_none"))
        self.ref_info = QLabel(t("health.Reference_none"))
        self.ref_info.setWordWrap(True)

        mode_row = QHBoxLayout()
        mode_row.addWidget(self.mode_combo)
        mode_row.addWidget(QLabel(t("health.Type_ref")))
        mode_row.addWidget(self.type_ref_combo, 1)

        cb_row = QHBoxLayout()
        cb_row.addWidget(self.emission_enabled)
        cb_row.addWidget(self.auto_preheat_cb)
        cb_row.addStretch(1)

        info_row = QHBoxLayout()
        info_row.addWidget(self.baseline_info)
        info_row.addWidget(self.ref_info, 1)

        form.addRow(t("health.Reference_mode"), mode_row)
        form.addRow("", cb_row)
        form.addRow("", info_row)
        return box

    def _build_plan_box(self) -> QGroupBox:
        """Build "Plan settings" group box.

        Layout/widget construction lives in ``app/health_plan_builder.py``.
        Here we just unpack the returned ``PlanWidgets`` dataclass back to
        ``self.<name>`` attributes so the ~100 ``self.plan_*`` reads
        scattered through this file keep working unchanged.
        """
        from app.health_plan_builder import build_plan_box
        box, w = build_plan_box(self, on_ug2_mode_toggled=self._on_ug2_mode_toggled)
        # Unpack widget refs to flat attrs (preserves API for ~100 callers).
        self.plan_ua_target = w.plan_ua_target
        self.plan_ug1_target = w.plan_ug1_target
        self.plan_ug2_target = w.plan_ug2_target
        self.plan_delta_ua = w.plan_delta_ua
        self.plan_delta_ug1 = w.plan_delta_ug1
        self.plan_delta_ug2 = w.plan_delta_ug2
        self.plan_emission_ratio = w.plan_emission_ratio
        self.plan_emission_mode = w.plan_emission_mode
        self.plan_bias_servo = w.plan_bias_servo
        self.plan_points = w.plan_points
        self.plan_repeats = w.plan_repeats
        self.plan_reset_btn = w.plan_reset_btn
        self.plan_validation_label = w.plan_validation_label
        self.ug2_mode_group = w.ug2_mode_group
        self.ug2_independent_radio = w.ug2_independent_radio
        self.ug2_track_radio = w.ug2_track_radio
        self.ug2_offset = w.ug2_offset
        self._plan_ug2_label = w.plan_ug2_label
        self._plan_delta_ug2_label = w.plan_delta_ug2_label
        self._plan_delta_ug2_unit = w.plan_delta_ug2_unit
        self._plan_emission_label = w.plan_emission_label
        self._ug2_offset_label = w.ug2_offset_label
        self._ug2_mode_prefix_label = w.ug2_mode_prefix_label
        return box

    def _build_actions_box(self) -> QGroupBox:
        box = QGroupBox(t("health.Actions"))
        row = QHBoxLayout(box)
        row.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        row.setSpacing(SPACING_NORMAL)
        self.quick_btn = QPushButton(t("health.Quick_test"))
        self.quick_btn.setToolTip(t("tip.Health_quick_test"))
        self.stop_btn = QPushButton(t("health.Stop"))
        self.stop_btn.setToolTip(t("tip.Health_stop_keep_heater"))
        self.stop_all_btn = QPushButton(t("health.Stop_heater_off"))
        self.stop_all_btn.setToolTip(t("tip.Health_stop_heater_off"))
        self.save_type_ref_btn = QPushButton(t("health.Save_as_type_ref"))
        self.save_type_ref_btn.setToolTip(t("tip.Health_save_type_ref"))
        self.set_active_btn = QPushButton(t("health.Set_active"))
        self.set_active_btn.setToolTip(t("tip.Health_set_active"))
        for w in (self.quick_btn, self.stop_btn, self.stop_all_btn,
                  self.save_type_ref_btn, self.set_active_btn):
            row.addWidget(w)
        row.addStretch(1)
        return box

    def _build_result_box(self) -> QGroupBox:
        # Copy lives in the title row next to the box name: the header
        # space is otherwise empty, so the grid does not spend a row on
        # the button.
        self.result_copy_btn = QPushButton(t("health.Copy_result"))
        self.result_copy_btn.setToolTip(t("tip.Health_copy_result"))
        self.result_copy_btn.clicked.connect(self._copy_result_to_clipboard)
        box = TitleRowButtonGroupBox(t("health.Result"), self.result_copy_btn)
        grid = QGridLayout(box)
        apply_tight(grid)
        self.result_index = QLabel(t("health.Index_none"))
        self.result_verdict = QLabel(t("health.Verdict_none"))
        self.result_delta = QLabel(t("health.Delta_none"))
        self.result_pct = QLabel(t("health.Result_pct_none"))
        self.result_ia_abs = QLabel(t("health.Result_ia_none"))
        self.result_srk = QLabel(t("health.Result_srk_none"))
        self.result_sg2 = QLabel(t("health.Result_sg2_none"))
        self.result_bias = QLabel("")
        self.result_bias.setToolTip(t("tip.Health_result_bias"))
        self.result_bias.setVisible(False)
        self.result_emission = QLabel(t("health.Emission_none"))

        self.result_index.setToolTip(t("tip.Health_result_index"))
        self.result_verdict.setToolTip(t("tip.Health_result_verdict"))
        self.result_delta.setToolTip(t("tip.Health_result_delta"))
        self.result_pct.setToolTip(t("tip.Health_result_pct"))
        self.result_ia_abs.setToolTip(t("tip.Health_result_ia"))
        self.result_srk.setToolTip(t("tip.Health_result_srk"))
        self.result_sg2.setToolTip(t("tip.Health_result_sg2"))
        self.result_emission.setToolTip(t("tip.Health_result_emission"))

        grid.addWidget(self.result_index, 0, 0)
        grid.addWidget(self.result_verdict, 0, 1)
        grid.addWidget(self.result_delta, 0, 2)
        grid.addWidget(self.result_pct, 1, 0, 1, 3)
        grid.addWidget(self.result_ia_abs, 2, 0)
        grid.addWidget(self.result_srk, 2, 1, 1, 2)
        # Servo details live on their own row — inside the Ia cell they
        # visually merge with the S/R/K column. Hidden for plan-bias
        # runs, so it costs no height then.
        grid.addWidget(self.result_bias, 3, 0, 1, 3)
        grid.addWidget(self.result_sg2, 4, 0, 1, 2)
        grid.addWidget(self.result_emission, 4, 2)
        return box

    # --- Right panel sub-builders ---

    def _build_right_panel(self, layout: QVBoxLayout) -> None:
        filter_row = QHBoxLayout()
        self.filter_lamp_combo = QComboBox()
        self.filter_lamp_combo.addItem(t("health.All"), "all")
        self.filter_lamp_combo.setToolTip(t("tip.Health_history_filter"))
        # Resize to longest item so long lamp IDs are not truncated. The
        # default AdjustToContentsOnFirstShow freezes the width after the
        # first populate, before real IDs are loaded.
        self.filter_lamp_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.filter_search = QLineEdit()
        self.filter_search.setPlaceholderText(t("health.filter_Search_placeholder"))
        self.filter_search.setToolTip(t("tip.Health_filter_search"))
        self.filter_search.setClearButtonEnabled(True)
        self.filter_mode_combo = QComboBox()
        self.filter_mode_combo.addItem(t("health.All"), "all")
        self.filter_mode_combo.addItem("Pent", TOPOLOGY_PENTODE)
        self.filter_mode_combo.addItem("TriC", TOPOLOGY_TRIODE_CONNECTED)
        self.filter_mode_combo.addItem("Tri", TOPOLOGY_TRIODE)
        self.filter_mode_combo.setToolTip(t("tip.Health_filter_mode"))
        self.filter_verdict_combo = QComboBox()
        populate_verdict_filter_combo(self.filter_verdict_combo)
        self.filter_verdict_combo.setToolTip(t("tip.Health_filter_verdict"))
        self.filter_group_combo = QComboBox()
        self.filter_group_combo.addItem(t("health.All"), "all")
        self.filter_group_combo.setToolTip(t("tip.Health_filter_group"))
        self.filter_group_combo.setVisible(False)
        self.show_conditions_chk = QCheckBox(t("health.Show_conditions"))
        self.show_conditions_chk.setToolTip(t("tip.Health_show_conditions"))
        self.show_conditions_chk.setChecked(False)
        self.reload_table_btn = QPushButton(t("health.Reload"))
        self.reload_table_btn.setToolTip(t("tip.Health_reload"))
        self.copy_history_btn = QPushButton(t("health.Copy_history"))
        self.copy_history_btn.setToolTip(t("tip.Health_copy_history"))
        self.export_csv_btn = QPushButton(t("health.Export_CSV"))
        self.export_csv_btn.setToolTip(t("tip.Health_export_csv"))
        filter_row.addWidget(QLabel(t("health.Lamp_ID_filter")))
        filter_row.addWidget(self.filter_lamp_combo)
        filter_row.addWidget(self.filter_search, 1)
        filter_row.addWidget(self.filter_mode_combo)
        filter_row.addWidget(self.filter_verdict_combo)
        filter_row.addWidget(self.filter_group_combo)
        filter_row.addWidget(self.show_conditions_chk)
        filter_row.addWidget(self.copy_history_btn)
        filter_row.addWidget(self.export_csv_btn)
        filter_row.addWidget(self.reload_table_btn)

        self.table = QTableWidget(0, 25)
        self.table.setHorizontalHeaderLabels([
            t("health.col_Timestamp"),
            t("health.col_Lamp_ID"), t("health.col_Name"),
            t("common.col_Mfg"),
            t("health.col_An"), t("health.col_Mode"),
            t("common.Ua"), t("common.Ug1"), t("common.Ug2"),
            t("health.col_Dbias"),
            t("health.col_Index"),
            t("health.Pct_Ia"), t("health.Pct_S"), t("health.Pct_R"), t("health.Pct_K"),
            t("common.Ia"),
            t("health.S_short"), t("health.R_short"), t("health.K_short"),
            "\u00b5g2",
            t("health.col_Emission"), t("health.col_Reserve"), t("health.col_Ref"),
            t("health.match_col_Sel"), t("health.match_col_Grp"),
        ])
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_history_context_menu)
        self.table.setToolTip(t("tip.Health_history_table"))
        hh = self.table.horizontalHeader()
        hh.setMinimumSectionSize(HEALTH_MIN_SECTION_SIZE)
        # Free-text columns: explicit width, user-resizable.
        # All other columns: auto-size to content (header width sets the
        # minimum, so short headers won't truncate long values).
        _RTC = QHeaderView.ResizeMode.ResizeToContents
        _INTERACTIVE = QHeaderView.ResizeMode.Interactive
        _text_col_widths = {
            1: HEALTH_HISTORY_LAMP_ID_WIDTH,  # Lamp ID
            2: HEALTH_HISTORY_NAME_WIDTH,     # Name
            22: HEALTH_HISTORY_REF_WIDTH,     # Ref
        }
        for col in range(self.table.columnCount()):
            if col in _text_col_widths:
                hh.setSectionResizeMode(col, _INTERACTIVE)
                self.table.setColumnWidth(col, _text_col_widths[col])
            elif col in (_COL_SEL, _COL_GRP):
                # Fixed narrow widths for match markers (24/64 px).
                hh.setSectionResizeMode(col, _INTERACTIVE)
                self.table.setColumnWidth(col, 24 if col == _COL_SEL else 64)
            else:
                hh.setSectionResizeMode(col, _RTC)
        # Match columns hidden by default; visually move to front
        hh.moveSection(_COL_SEL, 0)  # Sel appears as visual column 0
        hh.moveSection(_COL_GRP, 1)  # Grp appears as visual column 1
        self.table.setColumnHidden(_COL_SEL, True)
        self.table.setColumnHidden(_COL_GRP, True)
        # OP-condition columns (Ua/Ug1/Ug2) hidden by default — they
        # rarely vary across a tube's measurements; toggled via the
        # filter-row "Show conditions" checkbox.
        for col in HEALTH_HISTORY_CONDITION_COLS:
            self.table.setColumnHidden(col, True)
        self.show_conditions_chk.toggled.connect(self._on_show_conditions_toggled)

        steps_box = self._build_steps_box()

        layout.addLayout(filter_row)
        tables_splitter = QSplitter(Qt.Orientation.Vertical)
        tables_splitter.addWidget(self.table)
        tables_splitter.addWidget(steps_box)
        tables_splitter.setSizes([400, 250])
        tables_splitter.setChildrenCollapsible(False)
        layout.addWidget(tables_splitter, 1)

    def _build_steps_box(self) -> QGroupBox:
        box = QGroupBox(t("health.Steps"))
        layout = QVBoxLayout(box)
        apply_tight(layout)
        actions = QHBoxLayout()
        self.copy_points_btn = QPushButton(t("health.Copy_points"))
        self.copy_points_btn.setToolTip(t("tip.Health_copy_points"))
        actions.addWidget(self.copy_points_btn)
        # Return-to-live: visible only while a test runs AND the table
        # shows a stored measurement the user clicked in the history.
        self.steps_live_btn = QPushButton(t("health.Steps_live_btn"))
        self.steps_live_btn.setToolTip(t("tip.Health_steps_live_btn"))
        self.steps_live_btn.setVisible(False)
        actions.addWidget(self.steps_live_btn)
        actions.addStretch(1)
        layout.addLayout(actions)
        self._steps_columns = [
            t("health.col_Step"), t("common.Ua"), t("common.Ug1"), t("common.Ug2"), t("common.Uh"), t("common.Ih"),
            t("common.Ia"), t("common.Ig2"), t("common.Pa"), t("common.Pg2"), t("health.col_Details"),
        ]
        self.steps_table = QTableWidget(0, len(self._steps_columns))
        self.steps_table.setHorizontalHeaderLabels(self._steps_columns)
        self.steps_table.setToolTip(t("tip.Health_steps_table"))
        self.steps_table.setSortingEnabled(False)
        self.steps_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.steps_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        sh = self.steps_table.horizontalHeader()
        sh.setMinimumSectionSize(HEALTH_MIN_SECTION_SIZE)
        for col, w in enumerate(HEALTH_STEPS_COL_WIDTHS):
            self.steps_table.setColumnWidth(col, w)
        sh.setStretchLastSection(True)
        layout.addWidget(self.steps_table)
        # Live-view state: every live point of the current run is buffered
        # here — the table is a VIEW over this buffer (or over a stored
        # measurement in history mode), never the only copy of the data.
        self._live_points: List[Dict] = []
        self._planned_steps: List[Dict] = []
        self._steps_view_live: bool = True
        self._test_active: bool = False
        return box

    # --- Signals ---

    def _connect_signals(self) -> None:
        self.lamp_panel.tube_changed.connect(self._on_tube_changed)
        self.lamp_panel.lamp_id_changed.connect(self._on_lamp_id_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.type_ref_combo.currentIndexChanged.connect(self._update_reference_info)
        self.emission_enabled.toggled.connect(self._on_emission_toggled)
        self.plan_reset_btn.clicked.connect(self._reset_plan_defaults)
        for _w in (
            self.plan_ua_target,
            self.plan_ug1_target,
            self.plan_ug2_target,
            self.plan_delta_ua,
            self.plan_delta_ug1,
            self.plan_delta_ug2,
            self.plan_points,
            self.plan_repeats,
            self.plan_emission_ratio,
            self.ug2_offset,
        ):
            _w.valueChanged.connect(lambda *_: self._refresh_planned_info())
            _w.valueChanged.connect(lambda *_: self._validate_plan())
        self.reload_table_btn.clicked.connect(self.reload_history)
        # ML-143: filter dropdown only re-applies the filter over cached
        # entries — no disk re-read (the disk set is unchanged).
        self.filter_lamp_combo.currentIndexChanged.connect(self._apply_history_filter)
        self.filter_search.textChanged.connect(self._apply_table_filters)
        self.filter_mode_combo.currentIndexChanged.connect(self._apply_table_filters)
        self.filter_verdict_combo.currentIndexChanged.connect(self._apply_table_filters)
        self.filter_group_combo.currentIndexChanged.connect(self._apply_table_filters)
        self.table.itemSelectionChanged.connect(self._on_history_selection_changed)
        self.copy_points_btn.clicked.connect(self._copy_points_to_clipboard)
        self.steps_live_btn.clicked.connect(self._render_live_steps)
        self.copy_history_btn.clicked.connect(self._copy_history_to_clipboard)
        self.export_csv_btn.clicked.connect(self._export_history_csv)
        self.quick_btn.clicked.connect(self._start_test)
        self.stop_btn.clicked.connect(self._stop_keep_heater)
        self.stop_all_btn.clicked.connect(self._stop_and_heater_off)
        self.save_type_ref_btn.clicked.connect(self._save_as_type_ref)
        self.set_active_btn.clicked.connect(self._set_active_type_ref)
        # Match panel signals
        self.match_panel.calculate_requested.connect(self._run_matching)
        self.match_panel.clear_requested.connect(self._clear_match_result)
        self.match_panel.hide_inactive_cb.toggled.connect(self._apply_match_visibility)
        self.match_panel.copy_btn.clicked.connect(self._copy_match_groups)
        self.match_panel.export_btn.clicked.connect(self._export_match_csv)
        self.match_panel.certificate_btn.clicked.connect(
            self._export_match_certificate)
        self.table.cellClicked.connect(self._on_table_cell_clicked)

    # ---------- Public ----------
    def refresh_lamps(self) -> None:
        lamps = self.get_lamps() or []
        current = self.tube_combo.currentText() or self.get_current_tube_type() or ""
        self.lamp_panel.set_lamps(lamps, current)
        if self.get_current_lamp_id():
            self.lamp_id_edit.setText(self.get_current_lamp_id())
        self._on_tube_changed(self.tube_combo.currentText())
        try:
            self.match_panel.load_from_config(self.get_app_config())
        except (KeyError, ValueError, OSError):
            # Best-effort load: missing keys / corrupt JSON / file-system
            # issues should not block the Health tab from opening.
            # Programming errors (AttributeError, TypeError) propagate.
            pass

    def update_live_params(self, data: Dict) -> None:
        try:
            self.live_panel.update_values(data, self.get_calibration())
        except (KeyError, ValueError, TypeError) as exc:
            # Data errors from a transient comm tick: missing dict key,
            # malformed protocol value, decode of None. Programming
            # errors (AttributeError on a removed widget, NameError)
            # propagate so refactor regressions surface immediately.
            log.debug("health live param update failed: %s: %s",
                      type(exc).__name__, exc)
            return

    # ---------- Internal ----------
    def _on_ug2_mode_toggled(self, _checked: bool) -> None:
        track = self.ug2_track_radio.isChecked()
        self.plan_ug2_target.setEnabled(not track)
        self.plan_delta_ug2.setEnabled(not track)
        self._plan_delta_ug2_label.setVisible(not track)
        self.plan_delta_ug2.setVisible(not track)
        self._plan_delta_ug2_unit.setVisible(not track)
        self._refresh_planned_info()
        self._validate_plan()

    def _set_plan_defaults_from_lamp(self, lamp: LampConfig) -> None:
        cfg = self.get_app_config()
        target_ug2 = 0.0 if lamp.is_triode else float(lamp.ug2)
        pct = cfg.health_delta_pct / 100.0
        delta_ua = clamp_delta_ua(lamp.ua * pct,
                                  cfg.health_delta_ua_min_v, cfg.health_delta_ua_max_v)
        delta_ug1 = clamp_delta_ug1(abs(lamp.ug1) * pct,
                                    cfg.health_delta_ug1_min_v, cfg.health_delta_ug1_max_v)
        pct_ug2 = cfg.health_delta_ug2_pct / 100.0
        delta_ug2 = clamp_delta_ug2(target_ug2 * pct_ug2,
                                    cfg.health_delta_ug2_min_v, cfg.health_delta_ug2_max_v) \
            if target_ug2 > 0 else clamp_delta_ug2(0, cfg.health_delta_ug2_min_v, cfg.health_delta_ug2_max_v)
        self.plan_ua_target.setValue(float(lamp.ua))
        self.plan_ug1_target.setValue(float(lamp.ug1))
        self.plan_ug2_target.setValue(float(target_ug2))
        self.plan_delta_ua.setValue(delta_ua)
        self.plan_delta_ug1.setValue(delta_ug1)
        self.plan_delta_ug2.setValue(delta_ug2)
        self.plan_points.setValue(5)
        self.plan_repeats.setValue(int(cfg.health_ia_samples))
        self.plan_emission_ratio.setValue(float(cfg.health_emission_uh_ratio))
        mode_idx = self.plan_emission_mode.findData(str(cfg.health_emission_mode_default))
        self.plan_emission_mode.setCurrentIndex(max(0, mode_idx))
        self.plan_bias_servo.setChecked(bool(cfg.health_bias_servo_enabled_default))
        self.ug2_independent_radio.setChecked(True)
        self.ug2_offset.setValue(0.0)

    def _reset_plan_defaults(self) -> None:
        lamp = find_lamp(self.get_lamps() or [], self.tube_combo.currentText().strip())
        if not lamp:
            return
        self._set_plan_defaults_from_lamp(lamp)
        self._refresh_planned_info()

    def _collect_measurement_plan(self) -> Dict:
        lamp = find_lamp(self.get_lamps() or [], self.tube_combo.currentText().strip())
        if not lamp:
            return {}
        ug2_track = self.ug2_track_radio.isChecked() and not lamp.is_triode
        target_ug2 = 0.0 if lamp.is_triode else float(self.plan_ug2_target.value())
        return {
            "an": int(self.lamp_panel.anode()),
            "ug2_track_ua": ug2_track,
            "ug2_offset": float(self.ug2_offset.value()) if ug2_track else 0.0,
            "op": {
                "ua": float(self.plan_ua_target.value()),
                "ug1": float(self.plan_ug1_target.value()),
                "ug2": target_ug2,
                "uh": float(lamp.uh),
                "ih": float(lamp.ih),
            },
            "srk": {
                "delta_ua": float(self.plan_delta_ua.value()),
                "delta_ug1": float(self.plan_delta_ug1.value()),
                "delta_ug2": float(self.plan_delta_ug2.value()),
                "points": int(self.plan_points.value()),
                "repeats": int(self.plan_repeats.value()),
            },
            "emission": {
                "uh_ratio": float(self.plan_emission_ratio.value()),
                "mode": str(self.plan_emission_mode.currentData()),
            },
            "bias_servo": {
                "enabled": bool(self.plan_bias_servo.isChecked()),
            },
        }

    def _clear_plan_styles(self) -> None:
        for w in (
            self.plan_ua_target,
            self.plan_ug1_target,
            self.plan_ug2_target,
            self.plan_delta_ua,
            self.plan_delta_ug1,
            self.plan_delta_ug2,
            self.plan_points,
            self.plan_repeats,
            self.plan_emission_ratio,
        ):
            w.setStyleSheet("")

    def _validate_plan(self) -> bool:
        self._clear_plan_styles()
        errors: List[str] = []
        warnings: List[str] = []
        ua = float(self.plan_ua_target.value())
        delta_ua = float(self.plan_delta_ua.value())
        delta_ug1 = float(self.plan_delta_ug1.value())
        ua_max = DEFAULT_LIMITS["ua_max"]

        if ua - delta_ua < 0:
            errors.append(t("health.Plan_err_ua_delta"))
            self.plan_ua_target.setStyleSheet(STYLE_INPUT_ERROR)
            self.plan_delta_ua.setStyleSheet(STYLE_INPUT_ERROR)
        if delta_ua < UA_RESOLUTION_V:
            errors.append(t("health.Plan_err_delta_ua_small"))
            self.plan_delta_ua.setStyleSheet(STYLE_INPUT_ERROR)
        if delta_ug1 < UG1_RESOLUTION_V:
            errors.append(t("health.Plan_err_delta_ug1_small"))
            self.plan_delta_ug1.setStyleSheet(STYLE_INPUT_ERROR)
        ug1 = float(self.plan_ug1_target.value())
        # S-measurement sweeps Ug1 ± δUg1; if |Ug1| ≤ δ the top of the sweep
        # reaches/crosses 0 V into positive grid → garbage (and grid current).
        if abs(ug1) <= delta_ug1:
            errors.append(t("health.Plan_err_ug1_delta"))
            self.plan_ug1_target.setStyleSheet(STYLE_INPUT_ERROR)
            self.plan_delta_ug1.setStyleSheet(STYLE_INPUT_ERROR)

        r_center, r_method = compute_shifted_r_center(ua, delta_ua, ua_max)
        if r_method == "shifted_op":
            shift_v = round(ua - r_center)
            warnings.append(t("health.Plan_warn_shifted_r", shift=shift_v,
                              center=int(r_center), low=int(r_center - delta_ua),
                              high=int(r_center + delta_ua)))
            self.plan_ua_target.setStyleSheet(STYLE_INPUT_WARN)
            self.plan_delta_ua.setStyleSheet(STYLE_INPUT_WARN)

        lamp = find_lamp(self.get_lamps() or [], self.tube_combo.currentText().strip())
        is_pentode_mode = lamp and not lamp.is_triode and not self.ug2_track_radio.isChecked()
        if is_pentode_mode:
            ug2 = float(self.plan_ug2_target.value())
            delta_ug2 = float(self.plan_delta_ug2.value())
            ug2_max = DEFAULT_LIMITS["ug2_max"]
            sg2_center, sg2_method = compute_shifted_sg2_center(ug2, delta_ug2, ug2_max)
            if sg2_method == "shifted_op":
                shift_v = round(ug2 - sg2_center)
                warnings.append(t("health.Plan_warn_shifted_sg2", shift=shift_v,
                                  center=int(sg2_center)))
                self.plan_ug2_target.setStyleSheet(STYLE_INPUT_WARN)

        if errors:
            self.plan_validation_label.setStyleSheet(STYLE_STATUS_ERROR)
            self.plan_validation_label.setText("\n".join(errors))
            if not (self.worker and self.worker.isRunning()):
                self.quick_btn.setEnabled(False)
            return False

        if warnings:
            self.plan_validation_label.setStyleSheet(STYLE_STATUS_WARN)
            self.plan_validation_label.setText(t("health.Plan_valid") + "\n" + "\n".join(warnings))
        else:
            self.plan_validation_label.setStyleSheet(STYLE_STATUS_OK)
            self.plan_validation_label.setText(t("health.Plan_valid"))
        if not (self.worker and self.worker.isRunning()):
            self.quick_btn.setEnabled(True)
        return True

    def _plan_for_reference(self, plan: Dict) -> Dict:
        if not isinstance(plan, dict):
            return {}
        op = plan.get("op") if isinstance(plan.get("op"), dict) else {}
        srk = plan.get("srk") if isinstance(plan.get("srk"), dict) else {}
        emission = plan.get("emission") if isinstance(plan.get("emission"), dict) else {}
        return {
            "op": {
                "ua": op.get("ua"),
                "ug1": op.get("ug1"),
                "ug2": op.get("ug2"),
                "uh": op.get("uh"),
                "ih": op.get("ih"),
            },
            "srk": {
                "ua_min": srk.get("ua_min"),
                "ua_max": srk.get("ua_max"),
                "ug1_min": srk.get("ug1_min"),
                "ug1_max": srk.get("ug1_max"),
                "samples": srk.get("samples"),
                "ug1_step": srk.get("ug1_step"),
            },
            "emission": {
                "enabled": emission.get("enabled"),
                "uh_ratio": emission.get("uh_ratio"),
                "mode": emission.get("mode"),
            },
            "bias_servo": {
                "enabled": (plan.get("bias_servo") or {}).get("enabled")
                if isinstance(plan.get("bias_servo"), dict) else None,
            },
        }

    def _on_tube_changed(self, tube_type: str) -> None:
        self._reload_type_refs(tube_type)
        self._update_baseline_info(tube_type, self.lamp_panel.lamp_id())
        self._update_reference_info()
        lamp = find_lamp(self.get_lamps() or [], tube_type)
        if lamp:
            self.lamp_panel.apply_lamp(lamp)
            self.live_panel.set_nominal_heater(lamp.uh, lamp.ih)
            self._set_plan_defaults_from_lamp(lamp)
            self._apply_plan_visibility(lamp)
        self._refresh_planned_info()
        self._validate_plan()
        self.reload_history()

    def _apply_plan_visibility(self, lamp: LampConfig) -> None:
        is_pentode = not lamp.is_triode
        self._plan_ug2_label.setVisible(is_pentode)
        self.plan_ug2_target.setVisible(is_pentode)
        for w in (self._ug2_mode_prefix_label, self.ug2_independent_radio,
                  self.ug2_track_radio, self._ug2_offset_label, self.ug2_offset):
            w.setVisible(is_pentode)
        is_independent_pentode = is_pentode and not self.ug2_track_radio.isChecked()
        for w in (self._plan_delta_ug2_label, self.plan_delta_ug2, self._plan_delta_ug2_unit):
            w.setVisible(is_independent_pentode)

    def _on_lamp_id_changed(self, *_args) -> None:
        tube_type = self.tube_combo.currentText().strip()
        lamp_id = self.lamp_panel.lamp_id()
        self._update_baseline_info(tube_type, lamp_id)
        # ML-143: the measurement's Lamp ID doesn't change which files are
        # on disk — only re-apply the display filter, don't re-read all
        # JSON on every keystroke.
        self._apply_history_filter()

    def _on_emission_toggled(self, checked: bool) -> None:
        self._plan_emission_label.setVisible(checked)
        self.plan_emission_ratio.setVisible(checked)
        self._refresh_planned_info()
        self._validate_plan()

    def _on_mode_changed(self) -> None:
        mode = self.mode_combo.currentData()
        self.type_ref_combo.setEnabled(mode == "type")
        self.set_active_btn.setEnabled(mode == "type")
        self._update_reference_info()
        self._refresh_planned_info()
        self._validate_plan()

    def _reload_type_refs(self, tube_type: str) -> None:
        refs = list_type_refs(tube_type)
        self.type_ref_combo.clear()
        for ref in refs:
            ref_id = str(ref.get("id", ""))
            label = str(ref.get("label", ref_id))
            if ref.get("active"):
                label += t("health.Active_suffix")
            self.type_ref_combo.addItem(label, ref_id)
        self.type_ref_combo.setEnabled(self.mode_combo.currentData() == "type")

    def _update_baseline_info(self, tube_type: str, lamp_id: str) -> None:
        baseline = load_personal_baseline(tube_type, lamp_id)
        if baseline:
            ts = baseline.get("timestamp", "")
            self.baseline_info.setText(t("health.Baseline_found", timestamp=ts))
            self.name_edit.setText(str(baseline.get("name", f"{tube_type}_{lamp_id}")))
        else:
            self.baseline_info.setText(t("health.Baseline_not_found"))
            if not self.name_edit.text().strip():
                self.name_edit.setText(f"{tube_type}_{lamp_id}")

    def _resolve_reference(self, lamp: LampConfig) -> Dict:
        mode = str(self.mode_combo.currentData())
        ref_id = self.type_ref_combo.currentData()
        lamp_id = self.lamp_panel.lamp_id()
        return resolve_reference(
            mode,
            lamp.tube_type,
            lamp_id,
            ref_id,
            lamp,
            datasheet_label=t("health.RefMode_datasheet"),
            type_median_label=t("health.Type_median"),
        )

    def _validate_test_inputs(self) -> Optional[Tuple[Any, str, str, Any]]:
        """Run pre-flight checks for ``_start_test``.

        Returns ``(lamp, lamp_id, name, client)`` on success. Returns
        ``None`` and shows a warning dialog if any check fails.
        """
        if not self._validate_plan():
            QMessageBox.warning(self, t("msg.Error"), t("health.Plan_invalid_start"))
            return None
        if self.get_write_locked():
            QMessageBox.warning(self, t("msg.Reset_all_title"), t("msg.Emergency_write_block"))
            return None
        busy = self.get_hw_busy() if self.get_hw_busy else None
        if busy:
            log.warning("Health test blocked — hardware busy: %s", busy)
            QMessageBox.warning(self, t("msg.Reset_all_title"), t("msg.Hw_busy"))
            return None

        tube_type = self.tube_combo.currentText().strip()
        lamp_id = self.lamp_panel.lamp_id()
        name = self.name_edit.text().strip() or f"{tube_type}_{lamp_id}"
        lamp = find_lamp(self.get_lamps() or [], tube_type)
        if not lamp:
            QMessageBox.warning(self, t("msg.Lamp"), t("msg.No_lamp_selected"))
            return None

        client = self.get_client()
        if not client or not client.is_open():
            QMessageBox.warning(self, t("msg.COM"), t("msg.Connect_first"))
            return None
        return lamp, lamp_id, name, client

    def _check_heater_for_test(self, client, lamp) -> bool:
        """Confirm heater is hot enough to start a test.

        Returns True when the test should proceed. Returns False when
        the user aborted, or when a preheat was launched (the test will
        be auto-resumed on preheat completion).
        """
        cfg = self.get_app_config()
        ratio = float(getattr(cfg, "health_preheat_required_ratio", 0.75))
        now_value, required, unit = check_heater_level(
            client, lamp, ratio, calibration=self.get_calibration())
        nominal = lamp.uh if lamp.uh > 0 else lamp.ih
        near_zero = HEATER_NEAR_ZERO_V if unit == "V" else HEATER_NEAR_ZERO_A

        if now_value < near_zero:
            if self.auto_preheat_cb.isChecked() and callable(self.request_start_preheat):
                self._launch_preheat_then_test(lamp)
                return False
            reply = QMessageBox.question(
                self,
                t("msg.Preheat"),
                t("health.Heater_off_start_preheat",
                  now=f"{now_value:.2f}", unit=unit),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes and callable(self.request_start_preheat):
                self._launch_preheat_then_test(lamp)
            return False
        if now_value < required:
            pct = int(now_value / nominal * 100) if nominal > 0 else 0
            reply = QMessageBox.question(
                self,
                t("msg.Preheat"),
                t("health.Heater_low_confirm",
                  now=f"{now_value:.2f}", required=f"{required:.2f}",
                  unit=unit, pct=f"{pct}"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return False
        return True

    def _launch_health_worker(self, client, lamp, lamp_id: str, name: str, ref) -> None:
        """Reset progress UI, instantiate HealthWorker, wire signals, start."""
        self.set_poller_active(False)
        self.quick_btn.setEnabled(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_label.setText(t("health.Progress_starting"))
        self.live_state.setText(t("health.State_running"))
        # Fresh run: empty the live buffer and force the table back to
        # the live view whatever the user was reading before.
        self._live_points = []
        self._test_active = True
        self._steps_view_live = True
        self.steps_live_btn.setVisible(False)
        self._refresh_planned_info()

        self.worker = HealthWorker(
            client=client,
            lamp=lamp,
            app_config=self.get_app_config(),
            calibration=self.get_calibration(),
            lamp_id=lamp_id,
            name=name,
            reference_mode=str(self.mode_combo.currentData()),
            reference=ref,
            emission_enabled=self.emission_enabled.isChecked(),
            measurement_plan=self._collect_measurement_plan(),
            warmup_s=int(lamp.warmup_s),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.protection_triggered.connect(self._on_protection_triggered)
        self.worker.start()

    def _start_test(self) -> None:
        if self._pending_start_after_preheat:
            return

        validated = self._validate_test_inputs()
        if validated is None:
            return
        lamp, lamp_id, name, client = validated

        if not self._check_heater_for_test(client, lamp):
            return

        if self.worker and self.worker.isRunning():
            # Normally unreachable (quick_btn is disabled during a run);
            # reachable when a zombie thread was retained by
            # _cleanup_after_test — the refusal must be visible (ML-002).
            QMessageBox.warning(self, t("health.Quick_test"),
                                t("health.Worker_stuck_msg"))
            return
        if self.worker:
            # ML-002 (worker reattach canon): cleanup() joins the thread and
            # disconnects signals — queued cross-thread emissions of the old
            # worker must not land in slots and contaminate the new run.
            if not self.worker.cleanup():
                # Thread hung: keep the reference (a live QThread freed by
                # GC aborts the process) and refuse the restart — visibly.
                QMessageBox.warning(self, t("health.Quick_test"),
                                    t("health.Worker_stuck_msg"))
                return
            self.worker = None

        ref = self._resolve_reference(lamp)
        self._update_reference_info(ref)
        self._launch_health_worker(client, lamp, lamp_id, name, ref)

    def _launch_preheat_then_test(self, lamp) -> None:
        self._pending_start_after_preheat = True
        self._preheat_start_ts = time.monotonic()
        self._preheat_warmup_s = max(1, int(lamp.warmup_s))
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_label.setText(t("health.Preheat_auto_starting"))
        self.live_state.setText(t("health.State_wait_preheat"))
        if callable(self.request_start_preheat):
            self.request_start_preheat()
        self._start_preheat_wait_loop()

    def _start_preheat_wait_loop(self) -> None:
        if self._preheat_wait_timer is None:
            self._preheat_wait_timer = QTimer(self)
            self._preheat_wait_timer.setInterval(HEALTH_PREHEAT_POLL_MS)
            self._preheat_wait_timer.timeout.connect(self._check_preheat_ready)
        self._preheat_wait_timer.start()

    def _stop_preheat_wait_loop(self) -> None:
        if self._preheat_wait_timer and self._preheat_wait_timer.isActive():
            self._preheat_wait_timer.stop()

    def _check_preheat_ready(self) -> None:
        if not self._pending_start_after_preheat:
            self._stop_preheat_wait_loop()
            return
        if callable(self.get_preheat_done) and bool(self.get_preheat_done()):
            self._pending_start_after_preheat = False
            self._stop_preheat_wait_loop()
            self.progress.setValue(100)
            self.progress_label.setText(t("health.Preheat_auto_ready"))
            self._start_test()
            return
        elapsed = time.monotonic() - self._preheat_start_ts
        total = max(1, self._preheat_warmup_s)
        # ML-083: without a deadline a dead preheat worker or a heater
        # fault left Quick test waiting forever with no signal.
        deadline = (total * HEALTH_PREHEAT_TIMEOUT_FACTOR
                    + HEALTH_PREHEAT_TIMEOUT_MARGIN_S)
        if elapsed > deadline:
            log.warning("Preheat wait timed out after %.0f s "
                        "(warmup %d s, deadline %.0f s) — test aborted",
                        elapsed, total, deadline)
            self._cleanup_after_test()
            self.live_state.setText(t("health.State_error"))
            self.progress_label.setText(
                t("health.Preheat_timeout", elapsed=int(elapsed)))
            QMessageBox.warning(
                self, t("msg.Error"),
                t("health.Preheat_timeout_msg", elapsed=int(elapsed)))
            return
        pct = int(min(99, max(0, elapsed / total * 100)))
        remaining = max(0, int(total - elapsed))
        self.progress.setValue(pct)
        self.progress_label.setText(
            t("health.Preheat_progress", elapsed=int(elapsed), remaining=remaining, total=total)
        )

    def _on_progress(self, evt: Dict) -> None:
        event = evt.get("event")
        if event == "anode_sync":
            if bool(evt.get("confirmed", False)):
                try:
                    actual_an = int(evt.get("actual_an"))
                    self.live_panel.set_an(actual_an)
                except (TypeError, ValueError):
                    # int(None) → TypeError; int("foo") → ValueError.
                    # Both mean firmware sent unexpected payload — skip
                    # this update silently.
                    pass
            else:
                self.progress_label.setText(t("health.An_readback_failed"))
            return
        if event == "step":
            step = str(evt.get("step", ""))
            self.progress_label.setText(t("health.Progress_step", step=step))
            mapping = {"op": HEALTH_PROGRESS_OP, "srk": HEALTH_PROGRESS_SRK, "uh80": HEALTH_PROGRESS_UH80}
            self.progress.setValue(mapping.get(step, self.progress.value()))
            return
        if event == "srk_progress":
            done = int(evt.get("done", 0))
            total = int(evt.get("total", 1))
            pct = int(min(100, max(0, (done / max(1, total)) * 100)))
            self.progress.setValue(HEALTH_PROGRESS_OP + int(pct * HEALTH_PROGRESS_SRK_SPAN))
            self.progress_label.setText(t("health.Progress_step", step=t("health.SRK_progress", done=done, total=total)))
            return
        if event == "live_point":
            pt = evt.get("point") or {}
            self._live_points.append(dict(pt))
            self._update_live_from_point(pt)
            # In history mode the point only lands in the buffer — the
            # user is reading a stored measurement, the table must not
            # jump; the Live button brings everything buffered back.
            if self._steps_view_live:
                self._render_live_steps()
            return
        if event == "op_ramp":
            step_idx = int(evt.get("step_idx", 0))
            total = int(evt.get("total_steps", 1))
            ug1 = float(evt.get("ug1", 0.0))
            target_ug1 = float(evt.get("target_ug1", 0.0))
            pct = int(min(100, max(0, (step_idx / max(1, total)) * 100)))
            # OP-ramp progresses inside the OP phase budget — show it as a
            # fraction of HEALTH_PROGRESS_OP so the bar advances visibly
            # without overshooting into the SRK range.
            self.progress.setValue(int(pct * HEALTH_PROGRESS_OP / 100))
            self.progress_label.setText(
                t("health.Progress_op_ramp",
                  step=step_idx, total=total,
                  ug1=f"{ug1:.1f}", target=f"{target_ug1:.1f}")
            )
            # Reuse live-point update so Ia/Pa/Ig2/etc. show the ramp value.
            live_pt = {
                "ua": evt.get("ua"),
                "ug1": ug1,
                "ug2": evt.get("ug2"),
                "uh": evt.get("uh"),
                "ih": evt.get("ih"),
                "ia": evt.get("ia_ma"),
                "ig2": evt.get("ig2_ma") if evt.get("ig2_ma") is not None else 0.0,
            }
            self._update_live_from_point(live_pt)
            return
        if event == "bias_servo":
            # Live readings arrive via the live_point the probe emits just
            # before this event; here only the phase label is updated.
            self.progress_label.setText(t(
                "health.Progress_bias_servo",
                i=int(evt.get("iteration", 0)),
                max=int(evt.get("max_iterations", 0)),
            ))
            return
        if event == "bias_servo_accept":
            # The accepted probe already sits in the buffer under a plain
            # probe tag (its live_point predates the acceptance): retag
            # the last servo row and attach the shift so the live view
            # shows the OP row without waiting for the final re-render.
            for p in reversed(self._live_points):
                if str(p.get("step", "")).startswith(STEP_BIAS_SERVO):
                    p["step"] = STEP_BIAS_SERVO_OP
                    p["bias_shift_v"] = evt.get("bias_shift_v")
                    break
            if self._steps_view_live:
                self._render_live_steps()
            return
        if event == "emission_sweep":
            ratio = evt.get("ratio")
            pct = (int(round(float(ratio) * 100.0))
                   if isinstance(ratio, (int, float)) else 0)
            self.progress_label.setText(t(
                "health.Progress_em_sweep",
                i=int(evt.get("step_idx", 0)),
                max=int(evt.get("total_steps", 0)),
                pct=pct,
            ))
            return
        if event == "uh80_stabilizing":
            elapsed = int(evt.get("elapsed_s", 0))
            max_s = int(evt.get("t_max_s", 1))
            eta = int(evt.get("eta_s", 0))
            pct = int(min(100, max(0, (elapsed / max(1, max_s)) * 100)))
            self.progress.setValue(pct)
            self.progress_label.setText(t("health.Progress_uh80_eta", eta=eta, max=max_s))
            s = self.live_panel._sep
            ia = evt.get("ia_ma")
            if isinstance(ia, (int, float)):
                self.live_panel.lbl_ia.setText(f"{t('common.Ia')}{s}{format_label('ia_unit', float(ia))}")
            ig2 = evt.get("ig2_ma")
            if isinstance(ig2, (int, float)):
                self.live_panel.lbl_ig2.setText(f"{t('common.Ig2')}{s}{format_label('ig2_unit', float(ig2))}")
            uh = evt.get("uh")
            if isinstance(uh, (int, float)):
                # _update_uh_label adds the off-nominal badge — this is the
                # Uh80 phase, exactly where the heater sits below nominal.
                self.live_panel._update_uh_label(float(uh))
            ih = evt.get("ih")
            if isinstance(ih, (int, float)):
                self.live_panel._update_ih_label(float(ih))
            ua = evt.get("ua")
            if isinstance(ia, (int, float)) and isinstance(ua, (int, float)):
                # ML-041: Ua comes from the event now — parsing the UI's own
                # label text back into a number broke on any format change.
                pa_w = float(ua) * float(ia) / 1000.0
                self.live_panel._update_pa_label(pa_w)

    def _update_live_from_point(self, point: Dict) -> None:
        if not isinstance(point, dict):
            return
        self.live_panel.update_from_point(point)

    def _stop_common(self, keep_heater: bool) -> None:
        if self.worker and self.worker.isRunning():
            # cleanup() stops + waits + disconnects signals. If it does NOT
            # drain within the timeout, keep the reference (a live QThread freed
            # by GC aborts the process) and warn — do not silently reset outputs
            # and re-enable controls while a live worker can still command
            # Ua/Ug1 under the reset.
            if self.worker.cleanup(timeout_ms=2000):
                self.worker = None
            else:
                log.warning("Health worker did not stop within 2 s — keeping "
                            "reference; outputs reset is best-effort")
        self._pending_start_after_preheat = False
        self._stop_preheat_wait_loop()
        self._reset_outputs_safe(keep_heater=keep_heater)
        self.set_poller_active(True)
        self.quick_btn.setEnabled(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.live_state.setText(t("health.State_idle"))
        if keep_heater:
            self.progress_label.setText(t("health.Stopped_keep_heater"))
        else:
            self.progress_label.setText(t("health.Stopped_heater_off"))

    def _stop_keep_heater(self) -> None:
        self._stop_common(keep_heater=True)

    def _stop_and_heater_off(self) -> None:
        self._stop_common(keep_heater=False)

    def _cleanup_after_test(self) -> None:
        self._pending_start_after_preheat = False
        self._test_active = False
        self.steps_live_btn.setVisible(False)
        self._stop_preheat_wait_loop()
        if self.worker:
            # ML-002: cleanup() instead of bare wait() — the old code
            # ignored the wait() result and dropped a possibly-live QThread
            # to GC (process abort), leaving its signals connected.
            if not self.worker.cleanup():
                # Zombie: keep the reference (BaseWorker contract) and make
                # the degradation visible in the tab; a late finish still
                # lands in _on_finished and retries this cleanup.
                self.live_state.setText(t("health.State_error"))
                self.progress_label.setText(t("health.Worker_stuck"))
            else:
                self.worker = None
        else:
            self.worker = None
        self.quick_btn.setEnabled(True)
        self.set_poller_active(True)
        self._reset_outputs_safe(keep_heater=True)
        self._validate_plan()
        self.progress.setRange(0, 100)

    def _on_finished(self, measurement: Dict) -> None:
        self._cleanup_after_test()
        self.live_state.setText(t("health.State_done"))
        self.progress.setValue(100)
        self.progress_label.setText(t("health.Progress_completed"))

        tube_type = str(measurement.get("tube_type", self.tube_combo.currentText()))
        lamp_id = str(measurement.get("lamp_id", self.lamp_panel.lamp_id()))
        mfg_date = self.lamp_panel.mfg_date()
        if mfg_date:
            measurement["mfg_date"] = mfg_date
        # ML-084: a failed save is recoverable (Retry /
        # Save As / keep in-memory); the in-memory result below stays
        # visible either way — the label above already says "Completed".
        save_with_recovery(
            self,
            lambda: save_health_measurement(tube_type, lamp_id, measurement),
            measurement,
            suggested_filename(tube_type, lamp_id,
                               str(measurement.get("timestamp", ""))),
        )
        self.last_measurement = measurement
        self._update_result(measurement)
        self._update_steps_from_points(measurement.get("measurement_points") or [])
        self.reload_history()

        # Autocreate personal baseline on first successful measurement.
        baseline = load_personal_baseline(tube_type, lamp_id)
        if not baseline:
            baseline_payload = {
                "tube_type": tube_type,
                "lamp_id": lamp_id,
                "name": measurement.get("name", ""),
                "timestamp": measurement.get("timestamp", dt.datetime.now().isoformat(timespec="seconds")),
                "source": "measured",
                "conditions": measurement.get("conditions", {}),
                "measurement_plan": self._plan_for_reference(measurement.get("measurement_plan", {})),
                "reference": build_reference_from_measurement(measurement),
            }
            save_personal_baseline(tube_type, lamp_id, baseline_payload)
            self._update_baseline_info(tube_type, lamp_id)

    def _on_failed(self, message: str) -> None:
        self._cleanup_after_test()
        self.live_state.setText(t("health.State_error"))
        self.progress_label.setText(t("health.Progress_failed"))
        QMessageBox.critical(self, t("msg.Error"), message)

    def _on_protection_triggered(self, payload) -> None:
        """Pa/Pg2 safety limit tripped during OP-approach ramp.

        Worker has already restored Ug1 to safe lock before emitting; we
        finish the same cleanup as a normal failure and show a dedicated
        diagnostic dialog instead of the generic error message.
        """
        self._cleanup_after_test()
        self.live_state.setText(t("health.State_error"))
        self.progress_label.setText(
            t("health.Progress_protection", kind=payload.kind.upper())
        )
        show_health_protection_dialog(self, payload)

    def _reset_outputs_safe(self, keep_heater: bool = False) -> None:
        if self._reset_requested:
            return
        self._reset_requested = True
        try:
            cb = self.request_stop_keep_heater if keep_heater else self.request_stop_all
            if callable(cb):
                cb()
        finally:
            self._reset_requested = False

    def _save_as_type_ref(self) -> None:
        if not self.last_measurement:
            QMessageBox.warning(self, t("msg.Save"), t("health.No_result_to_save_ref"))
            return
        m = self.last_measurement
        tube_type = str(m.get("tube_type", self.tube_combo.currentText()))
        name = str(m.get("name", "")).strip() or f"{tube_type}_ref"
        ref_id = f"{name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ref_id = ref_id.replace(" ", "_")
        ref_payload = {
            "id": ref_id,
            "label": name,
            "tube_type": tube_type,
            "active": get_active_type_ref(tube_type) is None,
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "source": "measured",
            "conditions": m.get("conditions", {}),
            "measurement_plan": self._plan_for_reference(m.get("measurement_plan", {})),
            "reference": build_reference_from_measurement(m),
        }
        save_type_ref(tube_type, ref_id, ref_payload)
        self._reload_type_refs(tube_type)
        QMessageBox.information(self, t("msg.Save"), t("health.Type_ref_saved"))

    def _set_active_type_ref(self) -> None:
        tube_type = self.tube_combo.currentText().strip()
        ref_id = self.type_ref_combo.currentData()
        if not tube_type or not ref_id:
            return
        set_active_type_ref(tube_type, str(ref_id))
        self._reload_type_refs(tube_type)
        self._update_reference_info()

    def _update_reference_info(self, ref: Optional[Dict] = None) -> None:
        if not isinstance(ref, dict):
            ref = None
        if ref is None:
            lamp = find_lamp(self.get_lamps() or [], self.tube_combo.currentText())
            if lamp:
                ref = self._resolve_reference(lamp)
        if not ref:
            self.ref_info.setText(t("health.Reference_none"))
            return
        label = ref.get("label", ref.get("id", "datasheet"))
        src = ref.get("source", "unknown")
        ts = ref.get("timestamp", "")
        self.ref_info.setText(t("health.Reference_line", label=label, source=src, timestamp=ts))

    def _update_result(self, measurement: Dict) -> None:
        h = measurement.get("health", {})
        m = h.get("metrics", {})
        srk = measurement.get("srk", {})
        raw = h.get("raw", {})

        index = h.get("index")
        self.result_index.setText(
            format_label("index_pct", index) if isinstance(index, (int, float)) else t("health.Index_none")
        )
        self.result_verdict.setText(
            t("health.Result_verdict_line",
              verdict=health_verdict_text(h.get("verdict", HEALTH_VERDICT_NA)))
        )
        self.result_delta.setText(t("health.Delta_none"))

        parts = [
            self._fmt_pct(t("health.Pct_Ia"), m.get("ia_pct")),
            self._fmt_pct(t("health.Pct_S"), m.get("s_pct")),
            self._fmt_pct(t("health.Pct_R"), m.get("r_pct")),
            self._fmt_pct(t("health.Pct_K"), m.get("k_pct")),
        ]
        self.result_pct.setText("  ".join(parts))

        ia_op = raw.get("ia_op")
        self.result_ia_abs.setText(
            format_label("ia", ia_op) if isinstance(ia_op, (int, float))
            else t("health.Result_ia_none"))
        # Servo details go to their own row — merged into the Ia cell
        # they visually run into the S/R/K column.
        servo = h.get("bias_servo") or {}
        servo_status = servo.get("status", BIAS_SERVO_DISABLED)
        bias_text = ""
        if servo_status == BIAS_SERVO_OK:
            # S was measured at the reference current — say at which bias,
            # because the bias shift is itself the diagnostic.
            bias_text = t(
                "health.Result_bias_line",
                ug1=self._fmt_num(servo.get("ug1"), 2),
                shift=self._fmt_num(m.get("bias_shift_v"), 2),
            ).strip()
            plan_pct = m.get("ia_plan_pct")
            if isinstance(plan_pct, (int, float)):
                # The wear figure ia_pct no longer carries (it reads
                # ~100 at the reference current by construction).
                bias_text += t("health.Result_bias_plan",
                               pct=self._fmt_num(plan_pct, 0))
        elif servo_status != BIAS_SERVO_DISABLED:
            bias_text = t("health.Result_bias_failed",
                          status=bias_servo_status_text(servo_status)).strip()
        self.result_bias.setText(bias_text)
        self.result_bias.setVisible(bool(bias_text))

        em_ratio = m.get("emission_ratio")
        if isinstance(em_ratio, (int, float)):
            em_text = t(
                "health.Result_em_line",
                emission=format_label("emission", em_ratio),
                verdict=emission_verdict_text(m.get("emission_verdict")),
            )
            reserve = m.get("emission_reserve_pct")
            if isinstance(reserve, (int, float)):
                sweep = h.get("emission_sweep") or {}
                key = ("health.Result_em_reserve_min" if sweep.get("knee_below_range")
                       else "health.Result_em_reserve")
                em_text += t(key, pct=self._fmt_num(reserve, 0))
                # A knee estimated from a single falling point (or a
                # clamped/degenerate fit) is a bracket, not a fit —
                # the number must not read as solid.
                if m.get("emission_knee_confidence") == KNEE_CONF_LOW:
                    em_text += t("health.Result_em_knee_lowconf")
                # Interpretation bands on hover — "reserve 15%" alone
                # does not say whether that is good or bad.
                self.result_emission.setToolTip(t("health.Reserve_tooltip"))
            else:
                self.result_emission.setToolTip(t("tip.Health_result_emission"))
            # A ratio measured far below the tube's current capability
            # cannot see a depleted cathode — qualify it rather than let
            # "reserve normal" stand unconditionally.
            if m.get("emission_low_sensitivity"):
                sens = m.get("emission_sensitivity_ratio")
                em_text += t("health.Result_em_low_sens",
                             pct=self._fmt_num((sens or 0.0) * 100.0, 0))
            self.result_emission.setText(em_text)
        else:
            self.result_emission.setText(t("health.Result_em_none"))
        s = srk.get("s")
        r = srk.get("r")
        k = srk.get("k")
        sg2 = srk.get("sg2")
        mu_g1g2 = srk.get("mu_g1g2")
        unc = srk.get("uncertainty") or {}
        s_err = self._fmt_err(s, unc.get("s_rel"))
        r_err = self._fmt_err(r, unc.get("r_rel"))
        k_err = self._fmt_err(k, unc.get("k_rel"))
        self.result_srk.setText(t("health.Result_srk_line", s=s_err, r=r_err, k=k_err))
        sg2_err = self._fmt_err(sg2, unc.get("sg2_rel"))
        mu_str = format_label("mu", mu_g1g2) if isinstance(mu_g1g2, (int, float)) else "\u2014"
        if sg2 is not None:
            self.result_sg2.setText(t("health.Result_sg2_line", sg2=sg2_err, mu=mu_str))
        else:
            self.result_sg2.setText(t("health.Result_sg2_none"))

    def _copy_result_to_clipboard(self) -> None:
        lines = [
            self.result_index.text(),
            self.result_verdict.text(),
            self.result_pct.text(),
            self.result_ia_abs.text(),
            self.result_bias.text(),
            self.result_srk.text(),
            self.result_sg2.text(),
            self.result_emission.text(),
        ]
        QApplication.clipboard().setText("\n".join(l for l in lines if l))

    def _fmt_pct(self, label: str, v: Optional[float]) -> str:
        return f"{label}: {format_label('pct', v)}" if isinstance(v, (int, float)) else f"{label}: —"

    def _fmt_num(self, v: Optional[float], digits: int) -> str:
        if not isinstance(v, (int, float)):
            return "—"
        return f"{v:.{digits}f}"

    def _verdict_color(self, index: object) -> Optional[QColor]:
        cfg = self.get_app_config()
        return verdict_color(
            index,
            cfg.health_verdict_strong_min,
            cfg.health_verdict_good_min,
            cfg.health_verdict_weak_min,
        )

    def _fmt_err(self, v: Optional[float], rel: Optional[float]) -> str:
        if not isinstance(v, (int, float)):
            return "—"
        if isinstance(rel, (int, float)):
            return format_label("err_abs", v, error=abs(v * rel))
        return format_label("err_plain", v)

    def _fmt_target(self, value: Optional[float], digits: int = 2) -> str:
        if not isinstance(value, (int, float)):
            return "—"
        return f"{float(value):.{digits}f}"

    def _build_planned_steps(self, lamp: LampConfig) -> List[Dict]:
        """Thin wrapper — see ``lm19.health_planning.compute_planned_steps``."""
        from lm19.health_planning import compute_planned_steps
        return compute_planned_steps(
            plan=self._collect_measurement_plan(),
            lamp=lamp,
            emission_enabled=self.emission_enabled.isChecked(),
            uh_ratio_default=self.get_app_config().health_emission_uh_ratio,
        )

    def _refresh_planned_info(self) -> None:
        lamp = find_lamp(self.get_lamps() or [], self.tube_combo.currentText().strip())
        if not lamp:
            self.steps_table.setRowCount(0)
            return
        steps = self._build_planned_steps(lamp)
        self._show_steps_plan(steps)

    def _plan_step_color(self, step_name: str) -> Optional[QColor]:
        _map = {
            t("health.step_op"): HEALTH_STEP_OP,
            t("health.step_emission_100"): HEALTH_STEP_EMISSION_100,
            t("health.step_emission_80"): HEALTH_STEP_EMISSION_80,
        }
        c = _map.get(step_name)
        if c is None and step_name and step_name[:2] in ("S-", "S+", "S1", "S2", "S3", "R-", "R+", "Sg"):
            return HEALTH_STEP_SRK
        return c

    def _show_steps_plan(self, steps: List[Dict]) -> None:
        """Fill steps_table with planned voltages; current columns show '—'."""
        self._planned_steps = list(steps)
        self.steps_table.setRowCount(len(steps))
        for row_idx, step in enumerate(steps):
            step_name = step.get("step", "")
            vals = [
                step_name,
                step.get("ua"), step.get("ug1"), step.get("ug2"),
                step.get("uh"), step.get("ih"),
                None, None, None, None,
                step.get("details", ""),
            ]
            color = self._plan_step_color(step_name)
            self._fill_row(row_idx, vals, color)

    @staticmethod
    def _fmt_cell(col: int, val: float) -> str:
        if col == 5:  # Ih
            return f"{val:.3f}"
        if col in (8, 9):  # Pa, Pg2
            return f"{val:.3f}"
        return f"{val:.2f}"

    def _point_float(self, v: object) -> Optional[float]:
        if isinstance(v, (int, float)):
            return float(v)
        return None

    def _point_details(self, p: Dict) -> str:
        """Details text for a measured point, composed at the UI border
        from the STRUCTURED fields the point carries (ref_ia,
        bias_shift_v) — the saved JSON stays locale-independent, and the
        live and history views render identically."""
        step = str(p.get("step", ""))
        if step == STEP_BIAS_SERVO_RESTORE:
            return t("health.Detail_servo_restore")
        if step == STEP_BIAS_SERVO_OP:
            shift = p.get("bias_shift_v")
            if isinstance(shift, (int, float)):
                return t("health.Detail_servo_op", shift=f"{shift:+.2f}")
        if step.startswith(STEP_BIAS_SERVO):
            ref = p.get("ref_ia")
            ia = p.get("ia")
            if isinstance(ref, (int, float)) and isinstance(ia, (int, float)):
                return t("health.Detail_servo_probe",
                         ref=f"{ref:.1f}", delta=f"{ia - ref:+.1f}")
        return str(p.get("details", "") or "")

    def _point_row_vals(self, p: Dict) -> List:
        ua = self._point_float(p.get("ua"))
        ug1 = self._point_float(p.get("ug1"))
        ug2 = self._point_float(p.get("ug2"))
        uh = self._point_float(p.get("uh"))
        ih = self._point_float(p.get("ih"))
        ia = self._point_float(p.get("ia"))
        ig2 = self._point_float(p.get("ig2"))
        pa = (ua * ia / 1000.0) if ua is not None and ia is not None else None
        pg2 = (ug2 * ig2 / 1000.0) if ug2 is not None and ig2 is not None else None
        return [
            p.get("step", ""),
            ua, ug1, ug2, uh, ih, ia, ig2, pa, pg2,
            self._point_details(p),
        ]

    @staticmethod
    def _step_color(step_name: str) -> Optional[QColor]:
        # The ACCEPTED servo point shares the OP colour: it is the
        # measuring point of the run. Intermediate probes and the
        # restore-to-plan row stay uncoloured — approach trajectory,
        # not results.
        _map = {"op": HEALTH_STEP_OP,
                STEP_BIAS_SERVO_OP: HEALTH_STEP_OP,
                "emission_100": HEALTH_STEP_EMISSION_100,
                "emission_80": HEALTH_STEP_EMISSION_80}
        c = _map.get(step_name)
        if c is None and (step_name.startswith("srk_") or step_name.startswith("srk_sg2_")):
            return HEALTH_STEP_SRK
        return c

    def _fill_row(self, row_idx: int, vals: List, color: Optional[QColor] = None) -> None:
        for col, val in enumerate(vals):
            item = QTableWidgetItem()
            if isinstance(val, (int, float)):
                item.setData(Qt.ItemDataRole.EditRole, float(val))
                item.setText(self._fmt_cell(col, val))
            else:
                item.setText(str(val) if val not in (None, "") else "—")
            if color is not None:
                item.setBackground(color)
            self.steps_table.setItem(row_idx, col, item)

    def _update_steps_from_points(self, points: List[Dict]) -> None:
        """Populate steps_table from completed measurement points (result/history)."""
        self.steps_table.setRowCount(len(points))
        for row_idx, p in enumerate(points):
            vals = self._point_row_vals(p)
            color = self._step_color(str(p.get("step", "")))
            self._fill_row(row_idx, vals, color)

    @staticmethod
    def _consumes_plan_slot(p: Dict) -> bool:
        """Single source for the plan-cursor predicate: bias-servo rows
        (probes, the accepted OP, the restore) are trajectory points the
        plan never contained — they must not eat plan slots, or every
        probe would shift the preview by one row."""
        return not str(p.get("step", "")).startswith(STEP_BIAS_SERVO)

    def _remaining_plan(self) -> List[Dict]:
        """Planned rows not yet consumed by the live run.

        Adaptive emission-descent points beyond the grid merely exhaust
        the tail early, which is harmless for a preview.
        """
        consumed = sum(1 for p in self._live_points
                       if self._consumes_plan_slot(p))
        return self._planned_steps[consumed:]

    def _render_live_steps(self) -> None:
        """Hybrid live view: measured points on top — under their OWN
        step tags (a servo probe is a ``bias_servo`` row, never a
        mislabelled plan row) — and the not-yet-reached tail of the
        plan below."""
        self._steps_view_live = True
        self.steps_live_btn.setVisible(False)
        live = self._live_points
        tail = self._remaining_plan()
        self.steps_table.setRowCount(len(live) + len(tail))
        # Plan cursor mirrors _remaining_plan's predicate: every
        # non-servo point consumes one plan row and inherits its
        # localized name and Details; servo probes (and descent points
        # past the plan) keep their raw step tag.
        plan_cursor = 0
        for row_idx, p in enumerate(live):
            vals = self._point_row_vals(p)
            if self._consumes_plan_slot(p):
                if plan_cursor < len(self._planned_steps):
                    plan_row = self._planned_steps[plan_cursor]
                    vals[0] = plan_row.get("step", vals[0])
                    if not vals[10]:
                        vals[10] = plan_row.get("details", "")
                plan_cursor += 1
            vals[0] = f"✓ {vals[0]}"
            self._fill_row(row_idx, vals,
                           self._step_color(str(p.get("step", ""))))
        for i, step in enumerate(tail):
            step_name = step.get("step", "")
            vals = [
                step_name,
                step.get("ua"), step.get("ug1"), step.get("ug2"),
                step.get("uh"), step.get("ih"),
                None, None, None, None,
                step.get("details", ""),
            ]
            self._fill_row(len(live) + i, vals,
                           self._plan_step_color(step_name))
        if live:
            self.steps_table.scrollToItem(
                self.steps_table.item(len(live) - 1, 0))

    def _on_history_context_menu(self, pos) -> None:
        selected_rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        if not selected_rows:
            return
        entries = []
        for row in selected_rows:
            item = self.table.item(row, 0)
            if item:
                entry = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(entry, dict) and entry.get("_file_path"):
                    entries.append(entry)
        if not entries:
            return

        menu = QMenu(self)
        single = len(entries) == 1
        if single:
            repeat_action = QAction(t("health.ctx_Repeat_test"), menu)
            menu.addAction(repeat_action)
            manual_action = QAction(t("health.ctx_Load_to_manual"), menu)
            menu.addAction(manual_action)
            find_similar_action = QAction(t("health.ctx_Find_similar"), menu)
            menu.addAction(find_similar_action)
            find_similar_meas_action = QAction(t("health.ctx_Find_similar_meas"), menu)
            menu.addAction(find_similar_meas_action)
            menu.addSeparator()
        else:
            repeat_action = manual_action = find_similar_action = None
            find_similar_meas_action = None
        delete_action = QAction(t("health.ctx_Delete_selected", count=len(entries)), menu)
        menu.addAction(delete_action)
        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if action is None:
            return
        if single and action is repeat_action:
            self._load_plan_from_measurement(entries[0])
            return
        if single and action is manual_action:
            self._send_to_manual(entries[0])
            return
        if single and action is find_similar_action:
            self._start_find_similar(entries[0])
            return
        if single and action is find_similar_meas_action:
            self._start_find_similar(entries[0], this_measurement=True)
            return
        if action is not delete_action:
            return

        lines = []
        for e in entries:
            ts = e.get("timestamp", "?")
            lid = e.get("lamp_id", "?")
            name = e.get("name", "")
            idx_val = (e.get("health") or {}).get("index")
            idx_str = f", Index {idx_val:.0f}" if isinstance(idx_val, (int, float)) else ""
            lines.append(f"  • {ts}  {lid}  {name}{idx_str}")
        detail = "\n".join(lines)

        reply = QMessageBox.question(
            self,
            t("health.ctx_Delete_title"),
            t("health.ctx_Delete_confirm", count=len(entries)) + "\n\n" + detail,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        for e in entries:
            delete_health_measurement(e["_file_path"])
        self.reload_history()

    def _load_plan_from_measurement(self, entry: Dict) -> None:
        """Fill health plan fields from a saved measurement's measurement_plan."""
        plan = entry.get("measurement_plan") or {}
        op = plan.get("op") or {}
        srk = plan.get("srk") or {}
        emission = plan.get("emission") or {}

        # Select tube type if available
        tube_type = entry.get("tube_type", "")
        if tube_type:
            idx = self.tube_combo.findText(tube_type)
            if idx >= 0:
                self.tube_combo.setCurrentIndex(idx)

        # Lamp ID and name
        lamp_id = entry.get("lamp_id", "")
        if lamp_id:
            self.lamp_id_edit.setText(lamp_id)
        name = entry.get("name", "")
        if name:
            self.name_edit.setText(name)

        # Operating point
        if "ua" in op:
            self.plan_ua_target.setValue(float(op["ua"]))
        if "ug1" in op:
            self.plan_ug1_target.setValue(float(op["ug1"]))
        if "ug2" in op:
            self.plan_ug2_target.setValue(float(op["ug2"]))

        # SRK deltas
        if "delta_ua" in srk:
            self.plan_delta_ua.setValue(float(srk["delta_ua"]))
        if "delta_ug1" in srk:
            self.plan_delta_ug1.setValue(float(srk["delta_ug1"]))
        if "delta_ug2" in srk:
            self.plan_delta_ug2.setValue(float(srk["delta_ug2"]))
        if "points" in srk:
            self.plan_points.setValue(int(srk["points"]))
        if "repeats" in srk:
            self.plan_repeats.setValue(int(srk["repeats"]))

        # Emission
        if "uh_ratio" in emission:
            self.plan_emission_ratio.setValue(float(emission["uh_ratio"]))
        em_enabled = emission.get("enabled")
        if em_enabled is not None:
            self.emission_enabled.setChecked(bool(em_enabled))
        if "mode" in emission:
            mode_idx = self.plan_emission_mode.findData(str(emission["mode"]))
            if mode_idx >= 0:
                self.plan_emission_mode.setCurrentIndex(mode_idx)

        # Bias servo
        servo = plan.get("bias_servo") or {}
        if "enabled" in servo:
            self.plan_bias_servo.setChecked(bool(servo["enabled"]))

        # Ug2 mode
        ug2_track = plan.get("ug2_track_ua", False)
        if ug2_track:
            self.ug2_track_radio.setChecked(True)
            self.ug2_offset.setValue(float(plan.get("ug2_offset", 0.0)))
        else:
            self.ug2_independent_radio.setChecked(True)

        # Anode
        an = plan.get("an")
        if an is not None:
            btn = self.anode_group.button(int(an))
            if btn:
                btn.setChecked(True)

        self._refresh_planned_info()
        self._validate_plan()

    def _send_to_manual(self, entry: Dict) -> None:
        """Send measurement conditions to Manual tab."""
        conditions = entry.get("conditions") or {}
        if not conditions:
            return
        if self.on_load_to_manual:
            self.on_load_to_manual(conditions)

    def _on_history_selection_changed(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        item = self.table.item(row, 0)
        if not item:
            return
        entry = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(entry, dict):
            self._update_steps_from_points(entry.get("measurement_points") or [])
            # The table now shows a STORED measurement. During a running
            # test the live points keep landing in the buffer, and the
            # Live button is the way back to the process view.
            self._steps_view_live = False
            self.steps_live_btn.setVisible(self._test_active)

    @staticmethod
    @staticmethod
    def _table_to_tsv(table, selected_only=False):
        return table_to_tsv(table, selected_only)

    def _copy_points_to_clipboard(self) -> None:
        lines, count = self._table_to_tsv(self.steps_table, selected_only=True)
        if not count:
            lines, count = self._table_to_tsv(self.steps_table, selected_only=False)
        if not count:
            return
        QApplication.clipboard().setText("\n".join(lines))
        QMessageBox.information(
            self,
            t("msg.Save"),
            t("health.Points_copied", count=count),
        )

    def _history_table_to_lines(self, selected_only: bool = True) -> tuple[list[str], int]:
        return self._table_to_tsv(self.table, selected_only=selected_only)

    def _copy_history_to_clipboard(self) -> None:
        lines, count = self._history_table_to_lines(selected_only=True)
        if not count:
            lines, count = self._history_table_to_lines(selected_only=False)
        if not count:
            return
        QApplication.clipboard().setText("\n".join(lines))
        QMessageBox.information(
            self, t("msg.Save"),
            t("health.History_copied", count=count),
        )

    def _export_history_csv(self) -> None:
        lines, count = self._history_table_to_lines(selected_only=True)
        if not count:
            lines, count = self._history_table_to_lines(selected_only=False)
        if not count:
            return
        tube = self.tube_combo.currentText().strip() or "health"
        default_name = f"{tube}_health_history.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, t("health.Export_CSV_title"), default_name,
            t("health.Export_CSV_filter"),
        )
        if not path:
            return
        # ML-042: proper CSV — a cell containing ';' (free-text Name) or a
        # quote must be quoted, not split into extra columns. The tab→';'
        # string replace did exactly that. Narrow except: only I/O errors
        # are user-fixable here.
        rows = [line.split("\t") for line in lines]
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f, delimiter=";",
                                    quoting=csv.QUOTE_MINIMAL)
                writer.writerows(rows)
            QMessageBox.information(
                self, t("msg.Save"),
                t("health.CSV_exported", count=count, path=path),
            )
        except OSError as exc:
            log.exception("Health CSV export failed")
            QMessageBox.critical(self, t("msg.Error"), str(exc))

    def _on_show_conditions_toggled(self, checked: bool) -> None:
        """Show/hide the OP-condition columns (Ua/Ug1/Ug2) on demand."""
        for col in HEALTH_HISTORY_CONDITION_COLS:
            self.table.setColumnHidden(col, not checked)

    def reload_history(self, *_args) -> None:
        """Re-read all history JSON from disk, then re-apply the filter.

        ML-143: the disk read (``list_health_entries`` — every JSON of
        the tube type) is separated from the display filter. This full
        path is for events that actually change the on-disk set: tube
        type change, a saved measurement, the explicit Reload button.
        Field edits that only change what's DISPLAYED (Lamp ID keystroke,
        filter dropdown) call :meth:`_apply_history_filter`, which reuses
        the cached entries and never touches the disk.
        """
        if self._ignore_filter_signal:
            return
        tube_type = self.tube_combo.currentText().strip()
        if not tube_type:
            self._all_history_entries = []
            self.table.setRowCount(0)
            return
        self._all_history_entries = list_health_entries(tube_type)
        self._apply_history_filter()

    def _apply_history_filter(self, *_args) -> None:
        """Rebuild the lamp filter + table from the cached entries.

        No disk I/O — operates on ``self._all_history_entries`` populated
        by the last :meth:`reload_history`.
        """
        if self._ignore_filter_signal:
            return
        all_entries = self._all_history_entries

        # Update lamp filter values from available entries.
        lamp_ids = sorted({str(e.get("lamp_id", "")) for e in all_entries if e.get("lamp_id")})
        current_filter = self.filter_lamp_combo.currentData()
        self._ignore_filter_signal = True
        self.filter_lamp_combo.clear()
        self.filter_lamp_combo.addItem(t("health.All"), "all")
        for lid in lamp_ids:
            self.filter_lamp_combo.addItem(lid, lid)
        idx = self.filter_lamp_combo.findData(current_filter)
        self.filter_lamp_combo.setCurrentIndex(max(0, idx))
        # Recompute geometry so the combo fits the longest lamp_id label.
        self.filter_lamp_combo.adjustSize()
        self._ignore_filter_signal = False

        filter_lamp = self.filter_lamp_combo.currentData()
        rows = [e for e in all_entries if filter_lamp in (None, "all") or str(e.get("lamp_id", "")) == str(filter_lamp)]
        self._history_entries = rows

        populate_history_table(self.table, rows, color_fn=self._verdict_color)

        # Update tube mode combo from available entries
        self._update_match_mode_combo()

        # Reapply match decorations if a match result is active
        if self._match_result is not None:
            self._apply_match_to_table()

    # ── Tube Matching ────────────────────────────────────────────────

    def _start_find_similar(self, entry: Dict, *, this_measurement: bool = False) -> None:
        """Switch to Match tab in 'similar' mode for the given entry.

        ``this_measurement=False`` (default) anchors on the lamp's latest/best
        measurement; ``True`` anchors on the SPECIFIC clicked measurement.
        """
        lamp_id = str(entry.get("lamp_id", ""))
        cond = entry.get("conditions") or {}
        self._match_anchor_lamp_id = lamp_id
        self._match_anchor_timestamp = (
            str(entry.get("timestamp", "")) if this_measurement else None)
        self._match_anchor_an = int(cond.get("an", 1))
        self.match_panel.set_similar_mode(lamp_id)
        self._left_tabs.setCurrentWidget(self.match_panel)
        # Auto-detect tube mode from the anchor entry
        anchor_mode = cond.get("ug2_mode", TOPOLOGY_PENTODE)
        config = self.match_panel.get_config()
        config["tube_mode"] = anchor_mode
        self._run_matching(config)

    def _get_matching_entries(self, source: str = "all") -> List[Dict]:
        """Return entries for matching based on source.

        source: "all" = all history entries,
                "filtered" = visible (non-hidden) rows,
                "selected" = selected rows in table.
        """
        if source == "selected":
            selected_rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
            entries = []
            for row in selected_rows:
                item = self.table.item(row, 0)
                if item:
                    entry = item.data(Qt.ItemDataRole.UserRole)
                    if isinstance(entry, dict):
                        entries.append(entry)
            return entries

        if source == "filtered":
            entries = []
            for row in range(self.table.rowCount()):
                if self.table.isRowHidden(row):
                    continue
                item = self.table.item(row, 0)
                if item:
                    entry = item.data(Qt.ItemDataRole.UserRole)
                    if isinstance(entry, dict):
                        entries.append(entry)
            return entries

        # "all"
        return list(self._history_entries)

    def _apply_match_picks(
        self,
        entries: List[Dict],
        keep: Optional[Tuple[Optional[str], Optional[str]]] = None,
    ) -> List[Dict]:
        """Honour the user's measurement picks by pre-filtering *entries*.

        For each pinned (lamp_id, an) -> timestamp, the lamp's OTHER
        entries sharing the picked entry's conditions key are dropped, so
        ``match_tubes``'s latest/best selector trivially picks the pinned
        one. Entries in other pools stay — a pick acts within its pool
        only. A stale pick (entry no longer present) is ignored with a
        warning. *keep* — (lamp_id, timestamp) of an explicit
        similar-mode anchor, never dropped.
        """
        if not self._match_pick:
            return entries
        out = list(entries)
        for (lid, an), ts in self._match_pick.items():
            picked = next(
                (e for e in out
                 if str(e.get("lamp_id", "")) == lid
                 and int((e.get("conditions") or {}).get("an", 1)) == an
                 and str(e.get("timestamp", "")) == ts),
                None)
            if picked is None:
                log.warning(
                    "Measurement pick (%s, an%d) -> %s no longer matches "
                    "any entry — ignored", lid, an, ts)
                continue
            key = _conditions_key(picked)
            out = [
                e for e in out
                if e is picked
                or (keep is not None
                    and (str(e.get("lamp_id", "")),
                         str(e.get("timestamp", ""))) == keep)
                or str(e.get("lamp_id", "")) != lid
                or int((e.get("conditions") or {}).get("an", 1)) != an
                or _conditions_key(e) != key
            ]
        return out

    def _run_matching(self, config: Dict) -> None:
        """Execute matching with given config and update table."""
        source = config.get("source", "all")
        entries = self._get_matching_entries(source)
        source_count = len(entries)
        if not entries:
            empty = MatchResult(mode="groups", groups=[], unmatched=[])
            self._match_result = empty
            self.match_panel.set_result(empty)
            return

        # Determine tube_mode for conditions filter
        tube_mode = config.get("tube_mode") or self.match_panel.tube_mode_combo.currentData()
        if not tube_mode:
            tube_mode = (entries[0].get("conditions") or {}).get("ug2_mode", TOPOLOGY_PENTODE)

        mode = config.get("mode", "groups")
        anchor_id = self._match_anchor_lamp_id if mode == "similar" else None
        anchor_ts = self._match_anchor_timestamp if mode == "similar" else None
        anchor_an = self._match_anchor_an if mode == "similar" else None

        # B3 measurement picks: pre-filter entries so the core's
        # latest/best selector picks the pinned measurements. An explicit
        # "this measurement" anchor is exempt — the click names an exact
        # entry and must win over a pick on the same lamp.
        keep = (anchor_id, anchor_ts) if anchor_ts is not None else None
        entries = self._apply_match_picks(entries, keep=keep)

        conditions = build_matching_conditions(entries, tube_mode)
        cond_entries = [e for e in entries
                        if (e.get("conditions") or {}).get("ug2_mode", TOPOLOGY_PENTODE) == tube_mode]

        app_cfg = self.get_app_config()
        protocol = config.get("protocol", DEFAULT_MATCHING_PROTOCOL)
        result = match_tubes(
            entries,
            mode=mode,
            group_size=config.get("group_size", 2),
            use=config.get("use", "latest"),
            anode=config.get("anode", "each"),
            weights=config.get("weights"),
            max_delta=config.get("max_delta", 0.0),
            conditions=conditions,
            anchor_lamp_id=anchor_id,
            anchor_timestamp=anchor_ts,
            anchor_an=anchor_an,
            algorithm=config.get("algorithm", "greedy"),
            protocol=protocol,
            max_iq_imbalance_pct=app_cfg.health_matching_max_iq_imbalance_pct,
            bias_adjust_range_pct=app_cfg.health_matching_bias_adjust_range_pct,
        )
        self._match_result = result

        # Row-identity membership (includes the similar-mode anchor);
        # the lamp count deduplicates twin-anode records of one lamp.
        self._match_active = build_match_active(result)
        matched_count = len({lid for lid, _ts in self._match_active})
        source_labels = {
            "all": "",
            "filtered": t("health.match_Source_filtered", count=source_count),
            "selected": t("health.match_Source_selected", count=source_count),
        }
        # The pool the match ACTUALLY ran on (similar mode may switch to
        # the anchor's conditions — possibly another ug2_mode, so the
        # count runs over ALL pick-filtered entries, keyed by the used
        # tuple): the label must not promise lamps the pool never
        # admitted.
        used = result.conditions_used or conditions
        if used is not None:
            pool_total = len([
                e for e in entries
                if _conditions_compatible(_conditions_key(e), used, protocol)])
        else:
            pool_total = len(cond_entries)
        self.match_panel.set_info(
            matched_count, pool_total, source_labels.get(source, ""))

        # Update conditions display
        if used:
            self.match_panel.set_conditions(
                used[0], used[1], used[2], used[3], servo=used[4])

        self.match_panel.set_result(result)
        if result.anchor_error is not None:
            key = ANCHOR_ERROR_KEYS.get(result.anchor_error)
            if key is None:
                log.warning("Unknown anchor error code: %r",
                            result.anchor_error)
                key = ANCHOR_ERROR_KEYS[ANCHOR_ERR_NOT_FOUND]
            self.match_panel.summary_info.setText(t(key))
        self._update_group_filter(result)
        self._apply_match_to_table()

    def _update_group_filter(self, result: Optional[MatchResult]) -> None:
        """Populate group filter combo from match result."""
        self.filter_group_combo.blockSignals(True)
        self.filter_group_combo.clear()
        self.filter_group_combo.addItem(t("health.All"), "all")
        if result and result.groups:
            for g in result.groups:
                ids = ", ".join(r.lamp_id for r in g.records)
                grp = t("health.match_Group_prefix")
                self.filter_group_combo.addItem(
                    f"{grp} {g.number} ({ids})", f"g{g.number}")
            if result.unmatched:
                self.filter_group_combo.addItem(
                    t("health.filter_Unmatched"), "unmatched")
        self.filter_group_combo.blockSignals(False)
        self.filter_group_combo.setVisible(result is not None and bool(result.groups))

    def _apply_match_to_table(self) -> None:
        """Update Sel/Grp columns and row colors based on match result."""
        result = self._match_result
        if result is None:
            # Clear match columns
            self.table.setColumnHidden(_COL_SEL, True)
            self.table.setColumnHidden(_COL_GRP, True)
            return

        # Show match columns
        self.table.setColumnHidden(_COL_SEL, False)
        self.table.setColumnHidden(_COL_GRP, False)

        entry_info = build_match_entry_info(result)

        self.table.blockSignals(True)
        self.table.setSortingEnabled(False)

        for row in range(self.table.rowCount()):
            item0 = self.table.item(row, 0)
            if not item0:
                continue
            entry = item0.data(Qt.ItemDataRole.UserRole)
            if not isinstance(entry, dict):
                continue

            lamp_id = str(entry.get("lamp_id", ""))
            timestamp = str(entry.get("timestamp", ""))
            key = (lamp_id, timestamp)

            # ML-043: wash the PREVIOUS match's row colors first —
            # repeated Calculate runs accumulated stale group fills and
            # dimmed foregrounds. Restore the verdict tint populate applies.
            verdict_bg = self._verdict_color(
                (entry.get("health") or {}).get("index"))
            for col in range(self.table.columnCount()):
                ci = self.table.item(row, col)
                if ci is None:
                    continue
                ci.setData(Qt.ItemDataRole.ForegroundRole, None)
                if col in HEALTH_HISTORY_HIGHLIGHT_COLS:
                    continue  # headline cells keep their own bg
                if verdict_bg is not None:
                    ci.setBackground(verdict_bg)
                else:
                    ci.setData(Qt.ItemDataRole.BackgroundRole, None)

            # Sel column: ● if this row participates in the match
            sel_item = QTableWidgetItem()
            is_active = (lamp_id, timestamp) in self._match_active
            sel_item.setText("●" if is_active else "")
            sel_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, _COL_SEL, sel_item)

            # Grp column: group number or Δ. ML-044: sortItems compares
            # items via __lt__ — a plain text item sorted "10" before "2".
            # _FormattedNumericItem freezes the display and sorts by the
            # numeric key (Δ in similar mode, group number in groups
            # mode); unmatched/blank rows get +inf → last on ascending.
            info = entry_info.get(key)
            grp_sort_key = float("inf")
            grp_text = ""
            anchor = result.anchor
            is_anchor = (result.mode == "similar" and anchor is not None
                         and lamp_id == anchor.lamp_id
                         and timestamp == anchor.timestamp)
            if is_anchor:
                # The reference row of "Find similar": mark it and sort it
                # to the top — without a marker it reads as an inactive row.
                grp_text = "★"
                grp_sort_key = -1.0
            elif info is not None:
                grp_num, delta = info
                best_num = result.groups[0].number if result.groups else 0
                if result.mode == "similar":
                    if grp_num > 0:
                        star = "\u2605" if grp_num == 1 else ""
                        grp_text = f"#{grp_num} ({delta:.1f}){star}"
                        grp_sort_key = float(delta)
                    else:
                        grp_text = "—"
                else:
                    if grp_num > 0:
                        star = "\u2605" if grp_num == best_num else ""
                        grp_text = f"{grp_num} ({delta:.1f}){star}"
                        grp_sort_key = float(grp_num)
                    else:
                        grp_text = "—"
            grp_item = _FormattedNumericItem(grp_sort_key, grp_text)
            if info is not None:
                grp_item.setData(Qt.ItemDataRole.UserRole + 1,
                                 float(info[1]) if result.mode == "similar"
                                 else float(info[0]))
            grp_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if info is not None and info[0] > 0:
                dq = delta_quality(info[1])
                dq_color = DELTA_QUALITY_COLOR_MAP.get(dq, HEALTH_MATCH_DELTA_POOR)
                grp_item.setForeground(dq_color)
            self.table.setItem(row, _COL_GRP, grp_item)

            # Row coloring
            if info is not None and is_active:
                grp_num = info[0]
                if grp_num > 0:
                    color = HEALTH_MATCH_GROUP_COLORS[
                        (grp_num - 1) % len(HEALTH_MATCH_GROUP_COLORS)
                    ]
                else:
                    color = HEALTH_MATCH_UNMATCHED_BG
                for col in range(self.table.columnCount()):
                    ci = self.table.item(row, col)
                    if ci:
                        ci.setBackground(color)
            elif not is_active and self._match_active:
                # Dim inactive rows
                for col in range(self.table.columnCount()):
                    ci = self.table.item(row, col)
                    if ci:
                        ci.setForeground(HEALTH_MATCH_INACTIVE_FG)

        self.table.blockSignals(False)
        self.table.setSortingEnabled(True)
        # Sort by Grp column
        self.table.sortItems(_COL_GRP, Qt.SortOrder.AscendingOrder)
        self._apply_match_visibility()

    def _apply_match_visibility(self) -> None:
        """Reapply all table filters (delegates to unified filter)."""
        self._apply_table_filters()

    def _on_table_cell_clicked(self, row: int, col: int) -> None:
        """Sel-column click: pin/unpin this measurement for its lamp+anode.

        The pick lands in ``_match_pick`` (an INPUT of the match — entries
        are pre-filtered to honour it) and the match re-runs. Clicking the
        already-pinned row unpins it (back to latest/best).
        """
        if col != _COL_SEL or self._match_result is None:
            return
        item0 = self.table.item(row, 0)
        if not item0:
            return
        entry = item0.data(Qt.ItemDataRole.UserRole)
        if not isinstance(entry, dict):
            return
        lamp_id = str(entry.get("lamp_id", ""))
        if not lamp_id:
            return
        ts = str(entry.get("timestamp", ""))
        an = int((entry.get("conditions") or {}).get("an", 1))
        pick_key = (lamp_id, an)
        if self._match_pick.get(pick_key) == ts:
            del self._match_pick[pick_key]
        else:
            self._match_pick[pick_key] = ts
        self._run_matching(self.match_panel.get_config())

    def _apply_table_filters(self) -> None:
        """Hide rows that don't match search regex, mode, or verdict filters."""
        pattern_text = self.filter_search.text().strip()
        try:
            regex = re.compile(pattern_text, re.IGNORECASE) if pattern_text else None
        except re.error:
            regex = None

        mode_filter = self.filter_mode_combo.currentData()
        verdict_filter = self.filter_verdict_combo.currentData()
        group_filter = self.filter_group_combo.currentData()
        cfg = self.get_app_config()
        verdict_thr = {
            "strong": cfg.health_verdict_strong_min,
            "good": cfg.health_verdict_good_min,
            "weak": cfg.health_verdict_weak_min,
        }
        hide_inactive = bool(
            self._match_active
            and self.match_panel.hide_inactive_cb.isChecked()
        )

        for row in range(self.table.rowCount()):
            item0 = self.table.item(row, 0)
            if not item0:
                continue
            entry = item0.data(Qt.ItemDataRole.UserRole)
            if not isinstance(entry, dict):
                continue
            visible = entry_matches_filter(
                entry,
                regex=regex,
                mode_filter=mode_filter,
                verdict_filter=verdict_filter,
                verdict_thresholds=verdict_thr,
                group_filter=group_filter,
                match_result=self._match_result,
                match_active=self._match_active,
                hide_inactive=hide_inactive,
            )
            self.table.setRowHidden(row, not visible)

    def _update_match_mode_combo(self) -> None:
        """Populate tube mode combo from current history entries."""
        entries = self._history_entries
        if not entries:
            return
        mode_lamps: Dict[str, set] = {}
        for e in entries:
            m = (e.get("conditions") or {}).get("ug2_mode", TOPOLOGY_PENTODE)
            mode_lamps.setdefault(m, set()).add(str(e.get("lamp_id", "")))
        mode_lamp_counts = {m: len(lids) for m, lids in mode_lamps.items()}
        default_mode = (entries[0].get("conditions") or {}).get("ug2_mode", TOPOLOGY_PENTODE)
        self.match_panel.set_available_modes(mode_lamp_counts, default_mode)

    def _clear_match_result(self) -> None:
        """Clear matching state and restore table appearance."""
        self._match_result = None
        self._match_active.clear()
        self._match_pick.clear()
        self._match_anchor_lamp_id = None
        self._match_anchor_timestamp = None
        self._match_anchor_an = None
        self.match_panel.set_result(None)
        self._update_group_filter(None)
        self.table.setColumnHidden(_COL_SEL, True)
        self.table.setColumnHidden(_COL_GRP, True)
        # Restore original colors on next reload
        self.reload_history()

    def _copy_match_groups(self) -> None:
        """Copy match groups to clipboard as text."""
        result = self._match_result
        if not result:
            return
        lines = format_match_groups_text(
            result,
            group_label=t("health.match_Group_prefix"),
            unmatched_label=t("health.match_Unmatched_label"),
        )
        QApplication.clipboard().setText("\n".join(lines))

    def _export_match_certificate(self) -> None:
        """Generate a matched pair/quad PDF certificate (PDF-plan stage 3)."""
        from app.match_certificate import (
            CERT_SECTIONS,
            build_health_cert_fragments,
            generate_certificate_pdf,
            pick_match_group,
        )
        from app.report_options_dialog import ask_report_options
        from i18n_setup import translator_for

        group = pick_match_group(self, self._match_result)
        if group is None:
            return
        available = {"cert_conditions": "", "cert_metrics": "",
                     "cert_plot": "report.Na_no_image"}
        opts = ask_report_options(self, available, self.get_app_config(),
                                  specs=CERT_SECTIONS, session_key="cert")
        if opts is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, t("report.Cert_btn"), "", t("msg.PDF_filter"))
        if not path:
            return
        tr = translator_for(opts.language)
        tube_type = str((group.records[0].entry or {}).get("tube_type", "")
                        or self.tube_combo.currentText())
        try:
            generate_certificate_pdf(
                path,
                fragments=build_health_cert_fragments(
                    group, tube_type=tube_type, sections=opts.sections,
                    tr=tr),
                tr=tr)
            QMessageBox.information(self, t("report.Cert_btn"),
                                    t("msg.PDF_saved", path=path))
        except Exception as exc:
            log.exception("Certificate export failed")
            QMessageBox.critical(self, t("report.Cert_btn"),
                                 t("msg.PDF_error", error=str(exc)))

    def _export_match_csv(self) -> None:
        """Export match groups to CSV."""
        result = self._match_result
        if not result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, t("health.match_Export_CSV"), "",
            "CSV (*.csv);;All files (*)",
        )
        if not path:
            return
        import csv
        grp = t("health.match_Group_prefix")
        rows = format_match_csv_rows(result, group_label=grp)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([grp, "Lamp ID", "Mfg", "Ia (mA)", "S (mS)",
                             "R (kΩ)", "Δ (%)", "δIq (mA)"])
            writer.writerows(rows)
