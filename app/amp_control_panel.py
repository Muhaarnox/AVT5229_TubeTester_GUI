"""Amplifier control panel — left-panel widget for amplifier analysis.

Contains collapsible sections: Circuit, Source & Data, Parameters,
Optimizer, and Results. Emits settings_changed when any control changes.
Provides params_snapshot() to read all settings as AmpParams,
and show_results() to display analysis results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.checkable_combo import CheckableComboBox
from app.ui_theme import STYLE_MUTED_ITALIC
from app.widget_factory import make_double_spinbox, make_int_spinbox
from i18n_setup import t
from lm19.amp_engine import AmpParams, MIN_HALF_SWING_V
from lm19.amplifier.constants import (
    CIRCUIT_CF,
    CIRCUIT_PP,
    CIRCUIT_SE,
    CIRCUIT_SE_XFMR,
    HD_METHOD_5POINT,
    HD_METHOD_AUTO,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_DFT,
)

if TYPE_CHECKING:
    from lm19.amp_engine import AnalysisResult

log = logging.getLogger(__name__)


# ── module local constants ──
# Optimizer Ra grid derived from the CURRENT Ra spin (ML-014): fixed lower
# bound; upper = Ra × span factor with a floor so a small Ra still gets a
# meaningful sweep. kΩ.
_OPT_RA_MIN_KOHM = 0.5
_OPT_RA_SPAN_FACTOR = 5.0
_OPT_RA_MAX_FLOOR_KOHM = 50.0
RESULTS_MIN_HEIGHT = 180  # minimum height for results scroll area

# ═══════════════════════════════════════════════════════════════════════════
#  Collapsible GroupBox helper
# ═══════════════════════════════════════════════════════════════════════════


def _collapsible_group(title: str, *, collapsed: bool = False) -> QGroupBox:
    """Create a checkable QGroupBox that hides contents when unchecked."""
    gb = QGroupBox(title)
    gb.setCheckable(True)
    gb.setChecked(not collapsed)
    gb.toggled.connect(lambda checked: _toggle_group_content(gb, checked))
    # Defer initial collapse — children aren't added yet at this point.
    # _finalize_collapsible() must be called after populating the group.
    gb.setProperty("_collapsed_init", collapsed)
    return gb


def _finalize_collapsible(gb: QGroupBox) -> None:
    """Apply initial collapsed state after children have been added."""
    if gb.property("_collapsed_init"):
        _toggle_group_content(gb, False)


# Dynamic property holding a child's pre-collapse visibility (ML-028)
_COLLAPSE_SAVED_VISIBLE = "_collapse_saved_visible"


def _toggle_group_content(gb: QGroupBox, visible: bool) -> None:
    """Show/hide the group's children, PRESERVING state-driven hiding.

    ML-028: a blanket ``setVisible(True)`` on expand resurrected widgets
    that circuit/method state logic had hidden (e.g. PP-only rows while
    an SE circuit is selected). Collapse records each child's explicit
    hidden flag; expand restores it. Residual: a state change made WHILE
    the group is collapsed still wins only after the next state update.
    """
    for child in gb.findChildren(QWidget):
        if child.parent() != gb:
            continue
        if visible:
            saved = child.property(_COLLAPSE_SAVED_VISIBLE)
            child.setVisible(True if saved is None else bool(saved))
        else:
            # isHidden() = explicit flag (isVisible() is False for any
            # child of a not-yet-shown window — offscreen tests included).
            child.setProperty(_COLLAPSE_SAVED_VISIBLE, not child.isHidden())
            child.setVisible(False)


# ═══════════════════════════════════════════════════════════════════════════
#  AmpControlPanel
# ═══════════════════════════════════════════════════════════════════════════


class AmpControlPanel(QWidget):
    """Left-panel control widget for amplifier analysis.

    Sections:
      1. Circuit — topology + circuit-specific controls
      2. Source & Data — series selector, data source multi-select, HD method
      3. Parameters — Ub, Ra, Ug1, Swing, PaMax, NFB + AutoQ/OptRa
      4. Optimizer — multi-parameter optimization (collapsed by default)
      5. Results — scrollable RichText label

    Signals:
        settings_changed: emitted when any parameter changes (debounced).
        auto_q_requested: emitted when Auto Q button is clicked.
        ra_optimize_requested: emitted when Opt Ra button is clicked.
        optimize_requested: emitted when optimizer Run button is clicked.
    """

    settings_changed = Signal()
    auto_q_requested = Signal()
    ra_optimize_requested = Signal()
    optimize_requested = Signal()     # full multi-param optimization
    optimizer_apply_best = Signal()   # apply optimizer's best point to manual params
    optimizer_show_top_n = Signal()   # open Top-N candidates dialog
    export_pdf_requested = Signal()   # export the analysis as a PDF report
    verify_requested = Signal()       # run LTspice verification
    verify_cancel_requested = Signal()  # cancel a running verification

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._block_signals = False
        self._build_ui()

    # ── Build UI ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(4)

        root.addWidget(self._build_circuit_section())
        root.addWidget(self._build_source_section())
        root.addWidget(self._build_params_section())
        root.addWidget(self._build_optimizer_section())
        root.addWidget(self._build_results_section())
        root.addStretch(1)

    # ── Section 1: Circuit ────────────────────────────────────────

    def _build_circuit_section(self) -> QGroupBox:
        gb = _collapsible_group(t("amp.section_circuit"))
        gb.setToolTip(t("amp.section_circuit_tip"))
        layout = QVBoxLayout(gb)
        layout.setSpacing(4)

        self.circuit_combo = QComboBox()
        self.circuit_combo.addItem(t("amp.circuit_se"), CIRCUIT_SE)
        self.circuit_combo.addItem(t("amp.circuit_se_xfmr"), CIRCUIT_SE_XFMR)
        self.circuit_combo.addItem(t("amp.circuit_cf"), CIRCUIT_CF)
        self.circuit_combo.addItem(t("amp.circuit_pp"), CIRCUIT_PP)
        self.circuit_combo.setToolTip(t("amp.circuit_type_tip"))
        self.circuit_combo.currentIndexChanged.connect(self._on_circuit_changed)
        layout.addWidget(self.circuit_combo)

        # SE Transformer: Ra DC
        self._xfmr_widget = QWidget()
        xfmr_lay = QFormLayout(self._xfmr_widget)
        xfmr_lay.setContentsMargins(0, 0, 0, 0)
        self.xfmr_ra_dc_spin = make_double_spinbox(
            min_val=0.0, max_val=5.0, value=0.05,
            step=0.01, decimals=3, suffix=" kΩ",
            tooltip_key="amp.ra_dc_tip",
            on_change=self._on_setting_changed,
        )
        xfmr_lay.addRow(t("amp.ra_dc"), self.xfmr_ra_dc_spin)
        layout.addWidget(self._xfmr_widget)

        # CF: Rk, Rl
        self._cf_widget = QWidget()
        cf_lay = QFormLayout(self._cf_widget)
        cf_lay.setContentsMargins(0, 0, 0, 0)
        self.cf_rk_spin = make_double_spinbox(
            min_val=0.1, max_val=200.0, value=10.0,
            step=1.0, suffix=" kΩ",
            tooltip_key="amp.cf_rk_tip",
            on_change=self._on_setting_changed,
        )
        cf_lay.addRow(t("amp.rk"), self.cf_rk_spin)

        self.cf_rl_spin = make_double_spinbox(
            min_val=0.1, max_val=200.0, value=10.0,
            step=1.0, suffix=" kΩ",
            tooltip_key="amp.cf_rl_tip",
            on_change=self._on_setting_changed,
        )
        cf_lay.addRow(t("amp.rl"), self.cf_rl_spin)
        layout.addWidget(self._cf_widget)

        # PP: Raa, Matched, Tube B
        self._pp_widget = QWidget()
        pp_lay = QVBoxLayout(self._pp_widget)
        pp_lay.setContentsMargins(0, 0, 0, 0)
        pp_lay.setSpacing(4)

        pp_form = QFormLayout()
        pp_form.setContentsMargins(0, 0, 0, 0)
        self.pp_raa_spin = make_double_spinbox(
            min_val=0.5, max_val=100.0, value=8.0,
            step=0.5, suffix=" kΩ",
            tooltip_key="amp.pp_raa_tip",
            on_change=self._on_setting_changed,
        )
        pp_form.addRow(t("amp.ra_aa"), self.pp_raa_spin)

        # PP transformer half-primary winding DC resistance (each half).
        # Influences DC Q-point: Ua_q ≈ Ub − Iq · ra_dc.
        self.pp_ra_dc_spin = make_double_spinbox(
            min_val=0.0, max_val=5.0, value=0.1,
            step=0.01, decimals=3, suffix=" kΩ",
            tooltip_key="amp.pp_ra_dc_tip",
            on_change=self._on_setting_changed,
        )
        pp_form.addRow(t("amp.pp_ra_dc"), self.pp_ra_dc_spin)

        self.ul_tap_spin = make_double_spinbox(
            min_val=0, max_val=100, value=0,
            step=5, suffix=" %",
            tooltip_key="amp.ul_tap_tip",
            on_change=self._on_setting_changed,
        )
        pp_form.addRow(t("amp.ul_tap"), self.ul_tap_spin)

        # ── UL sweep for optimizer (presets / custom range) ──
        # Mode selector: off / presets / custom / both
        self.ul_sweep_mode_combo = QComboBox()
        self.ul_sweep_mode_combo.addItem(t("amp.ul_sweep_off"), "off")
        self.ul_sweep_mode_combo.addItem(t("amp.ul_sweep_presets"), "presets")
        self.ul_sweep_mode_combo.addItem(t("amp.ul_sweep_custom"), "custom")
        self.ul_sweep_mode_combo.addItem(t("amp.ul_sweep_both"), "presets_custom")
        self.ul_sweep_mode_combo.setToolTip(t("amp.ul_sweep_mode_tip"))
        self.ul_sweep_mode_combo.currentIndexChanged.connect(self._on_ul_sweep_mode_changed)
        pp_form.addRow(t("amp.ul_sweep") + ":", self.ul_sweep_mode_combo)

        # Horizontal preset checkboxes (visible in presets / both mode)
        self._ul_preset_specs: List[Tuple[float, str]] = [
            (0.0,  "ul_pentode_tip"),
            (0.20, "ul_acrosound_tip"),
            (0.35, "ul_6l6_tip"),
            (0.43, "ul_williamson_tip"),
            (0.50, "ul_quad_tip"),
            (1.0,  "ul_triode_tip"),
        ]
        self.ul_preset_cbs: Dict[float, QCheckBox] = {}
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        for tap, tip_key in self._ul_preset_specs:
            cb = QCheckBox(f"{int(round(tap * 100))}%")
            cb.setChecked(True)
            cb.setToolTip(t(f"amp.{tip_key}"))
            cb.toggled.connect(self._on_setting_changed)
            self.ul_preset_cbs[tap] = cb
            preset_row.addWidget(cb)
        preset_row.addStretch(1)
        self._ul_presets_label = QLabel(t("amp.ul_sweep_presets_label") + ":")
        pp_form.addRow(self._ul_presets_label, preset_row)
        self._ul_presets_row_layout = preset_row

        # Custom range (visible in custom / both mode)
        self.ul_sweep_min_spin = make_double_spinbox(
            min_val=0, max_val=100, value=0, suffix=" %",
        )
        self.ul_sweep_max_spin = make_double_spinbox(
            min_val=0, max_val=100, value=100, suffix=" %",
        )
        self.ul_sweep_steps_spin = make_int_spinbox(
            min_val=2, max_val=21, value=11,
        )
        custom_row = QHBoxLayout()
        custom_row.addWidget(self.ul_sweep_min_spin)
        custom_row.addWidget(QLabel("–"))
        custom_row.addWidget(self.ul_sweep_max_spin)
        custom_row.addWidget(QLabel(t("amp.ul_sweep_steps") + ":"))
        custom_row.addWidget(self.ul_sweep_steps_spin)
        custom_row.addStretch(1)
        self._ul_custom_label = QLabel(t("amp.ul_sweep_custom_label") + ":")
        pp_form.addRow(self._ul_custom_label, custom_row)
        self._ul_custom_row_layout = custom_row
        for sp in (self.ul_sweep_min_spin, self.ul_sweep_max_spin):
            sp.valueChanged.connect(self._on_setting_changed)
        self.ul_sweep_steps_spin.valueChanged.connect(self._on_setting_changed)

        # Initial visibility (mode=off → both rows hidden)
        self._update_ul_sweep_visibility("off")

        pp_lay.addLayout(pp_form)

        self.pp_matched_btn = QPushButton(t("amp.pp_matched"))
        self.pp_matched_btn.setCheckable(True)
        self.pp_matched_btn.setChecked(True)
        self.pp_matched_btn.setToolTip(t("amp.pp_matched_tip"))
        self.pp_matched_btn.toggled.connect(self._on_pp_matched_toggled)
        pp_lay.addWidget(self.pp_matched_btn)

        self._pp_tube_b_label = QLabel(t("amp.pp_tube_b") + ":")
        pp_lay.addWidget(self._pp_tube_b_label)
        self.pp_tube_b_combo = QComboBox()
        self.pp_tube_b_combo.setToolTip(t("amp.pp_tube_b_tip"))
        self.pp_tube_b_combo.currentIndexChanged.connect(self._on_setting_changed)
        pp_lay.addWidget(self.pp_tube_b_combo)
        self._pp_tube_b_label.setVisible(False)
        self.pp_tube_b_combo.setVisible(False)

        layout.addWidget(self._pp_widget)

        self._update_circuit_widgets()
        return gb

    # ── Section 2: Source & Data ──────────────────────────────────

    def _build_source_section(self) -> QGroupBox:
        gb = _collapsible_group(t("amp.section_source"))
        gb.setToolTip(t("amp.section_source_tip"))
        layout = QFormLayout(gb)
        layout.setSpacing(4)

        self.source_combo = QComboBox()
        self.source_combo.setToolTip(t("amp.source_tip"))
        self.source_combo.currentIndexChanged.connect(self._on_setting_changed)
        layout.addRow(t("amp.source") + ":", self.source_combo)

        self.data_source_combo = CheckableComboBox(placeholder=t("amp.data_source"))
        self.data_source_combo.setToolTip(t("amp.data_source_tip"))
        self._populate_data_source_combo()
        self.data_source_combo.selectionChanged.connect(self._on_setting_changed)
        layout.addRow(t("amp.data_source") + ":", self.data_source_combo)

        self.opt_mode_label = QLabel(t("amp.opt_mode_measurements"))
        self.opt_mode_label.setStyleSheet(STYLE_MUTED_ITALIC)
        self.ug2_info_label = QLabel("")
        self.ug2_info_label.setStyleSheet(STYLE_MUTED_ITALIC)
        info_row = QHBoxLayout()
        info_row.addWidget(self.opt_mode_label)
        info_row.addWidget(self.ug2_info_label)
        info_row.addStretch(1)
        layout.addRow("", info_row)

        self.hd_method_combo = QComboBox()
        self.hd_method_combo.addItem(t("amp.hd_auto"), HD_METHOD_AUTO)
        self.hd_method_combo.addItem(t("amp.hd_chebyshev"), HD_METHOD_CHEBYSHEV)
        self.hd_method_combo.addItem(t("amp.hd_dft"), HD_METHOD_DFT)
        self.hd_method_combo.addItem(t("amp.hd_5point"), HD_METHOD_5POINT)
        self.hd_method_combo.setToolTip(t("amp.hd_method_tip"))
        self.hd_method_combo.currentIndexChanged.connect(self._on_setting_changed)
        layout.addRow(t("amp.hd_method") + ":", self.hd_method_combo)

        self.show_hd45_cb = QCheckBox(t("amp.show_hd45"))
        self.show_hd45_cb.setChecked(False)
        self.show_hd45_cb.setToolTip(t("amp.show_hd45_tip"))
        self.show_hd45_cb.toggled.connect(self._on_setting_changed)

        self.show_gzp_cb = QCheckBox(t("amp.show_gzp"))
        self.show_gzp_cb.setChecked(False)
        self.show_gzp_cb.setToolTip(t("amp.show_gzp_tip"))
        self.show_gzp_cb.toggled.connect(self._on_setting_changed)

        cb_row = QHBoxLayout()
        cb_row.addWidget(self.show_hd45_cb)
        cb_row.addWidget(self.show_gzp_cb)
        cb_row.addStretch(1)
        layout.addRow("", cb_row)

        return gb

    # ── Section 3: Parameters ─────────────────────────────────────

    def _build_params_section(self) -> QGroupBox:
        gb = _collapsible_group(t("amp.section_params"))
        gb.setToolTip(t("amp.section_params_tip"))
        layout = QVBoxLayout(gb)
        layout.setSpacing(4)

        # Two-column grid: left (Ub, Ug1, Swing) + right (Ra, Pa max)
        self.ub_spin = make_double_spinbox(
            min_val=1.0, max_val=1000.0, value=250.0,
            step=10.0, suffix=" V",
            tooltip_key="amp.ub_tip",
            on_change=self._on_setting_changed,
        )

        self.ra_spin = make_double_spinbox(
            min_val=0.1, max_val=999.0, value=5.0,
            step=0.5, suffix=" kΩ",
            tooltip_key="amp.ra_tip",
            on_change=self._on_setting_changed,
        )

        self.ug1_spin = make_double_spinbox(
            min_val=-50.0, max_val=0.0, value=-6.0,
            step=0.5, suffix=" V",
            tooltip_key="amp.ug1_tip",
            on_change=self._on_setting_changed,
        )

        self.swing_spin = make_double_spinbox(
            min_val=0.0, max_val=50.0, value=0.0,
            step=0.1, suffix=" V",
            tooltip_key="amp.swing_tip",
            on_change=self._on_setting_changed,
        )

        self.pa_max_spin = make_double_spinbox(
            min_val=0.1, max_val=100.0, value=12.5,
            step=0.5, suffix=" W",
            tooltip_key="amp.pa_max_tip",
            on_change=self._on_setting_changed,
        )

        row1 = QHBoxLayout()
        row1.addWidget(QLabel(t("amp.ub")))
        row1.addWidget(self.ub_spin)
        row1.addWidget(QLabel(t("amp.ra")))
        row1.addWidget(self.ra_spin)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel(t("amp.ug1")))
        row2.addWidget(self.ug1_spin)
        row2.addWidget(QLabel(t("amp.swing")))
        row2.addWidget(self.swing_spin)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel(t("amp.pa_max")))
        row3.addWidget(self.pa_max_spin)

        self.nfb_check = QCheckBox(t("amp.nfb_label"))
        self.nfb_check.setToolTip(t("amp.nfb_enable_tip"))
        self.nfb_check.toggled.connect(self._on_nfb_toggled)
        row3.addWidget(self.nfb_check)

        self.nfb_spin = make_double_spinbox(
            min_val=0.0, max_val=30.0, value=6.0,
            step=1.0, suffix=" dB",
            tooltip_key="amp.nfb_spin_tip",
            on_change=self._on_setting_changed,
        )
        self.nfb_spin.setEnabled(False)
        row3.addWidget(self.nfb_spin)

        layout.addLayout(row3)

        # Buttons
        btn_row = QHBoxLayout()
        self.auto_q_btn = QPushButton(t("amp.auto_q"))
        self.auto_q_btn.setToolTip(t("amp.auto_q_tip"))
        self.auto_q_btn.clicked.connect(self.auto_q_requested.emit)
        btn_row.addWidget(self.auto_q_btn)

        self.opt_ra_btn = QPushButton(t("amp.opt_ra"))
        self.opt_ra_btn.setToolTip(t("amp.opt_ra_tip"))
        self.opt_ra_btn.clicked.connect(self.ra_optimize_requested.emit)
        btn_row.addWidget(self.opt_ra_btn)

        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        return gb

    # ── Section 4: Optimizer ─────────────────────────────────────

    def _build_optimizer_section(self) -> QGroupBox:
        gb = _collapsible_group(t("amp.section_optimizer"), collapsed=True)
        # ML-076: optimizer_constraints() gates the Ub/Ug2 ranges on this
        # group's CHECKED state — isVisible() also went False for a hidden
        # window / offscreen run, silently dropping user-set ranges.
        self.opt_group = gb
        gb.setToolTip(t("amp.section_optimizer_tip"))
        layout = QFormLayout(gb)
        layout.setSpacing(4)

        self.opt_target_combo = QComboBox()
        self.opt_target_combo.addItem(t("amp.opt_min_thd"), "min_thd")
        self.opt_target_combo.addItem(t("amp.opt_max_pout"), "max_pout")
        self.opt_target_combo.addItem(t("amp.opt_balanced"), "balanced")
        self.opt_target_combo.setToolTip(t("amp.opt_target_tip"))
        # Per-item tooltips so user gets explanation when hovering over
        # the dropdown items themselves, not just the closed combobox.
        for i, key in enumerate(("opt_min_thd_tip",
                                 "opt_max_pout_tip",
                                 "opt_balanced_tip")):
            self.opt_target_combo.setItemData(i, t(f"amp.{key}"), Qt.ItemDataRole.ToolTipRole)
        self.opt_target_combo.currentIndexChanged.connect(self._on_target_changed)
        layout.addRow(t("amp.opt_target") + ":", self.opt_target_combo)

        # Balanced weight — visible only when target=balanced. Controls
        # the trade-off in score = THD − w · log10(Pout_mW). Higher w =
        # prefer more output power; lower = prefer lower THD.
        self.opt_balanced_weight_label = QLabel(t("amp.opt_balanced_weight") + ":")
        # Range up to 20 — formula score = THD − w·log₁₀(Pout_mW). Since
        # log10 grows slowly (Pout 100 mW → log = 2; Pout 10 W → log = 4),
        # for typical distortion of a few %, w needs to reach ~5–10 before
        # the choice of "best" actually flips toward higher Pout. 0.5 is
        # only a mild Pout influence.
        self.opt_balanced_weight_spin = make_double_spinbox(
            min_val=0.0, max_val=20.0, value=0.5,
            step=0.5, decimals=2,
            tooltip_key="amp.opt_balanced_weight_tip",
        )
        layout.addRow(self.opt_balanced_weight_label, self.opt_balanced_weight_spin)
        # Hidden by default — initial target is min_thd
        self.opt_balanced_weight_label.setVisible(False)
        self.opt_balanced_weight_spin.setVisible(False)

        self.opt_pout_min_spin = make_double_spinbox(
            min_val=0.0, max_val=50.0, value=0.0,
            step=0.1, suffix=" W",
            tooltip_key="amp.opt_pout_min_tip",
        )
        layout.addRow(t("amp.opt_pout_min") + ":", self.opt_pout_min_spin)

        # THD cap: with target=max_pout this yields the datasheet-style
        # "Pout at X% THD" answer (swing sweep + refine push Pout up to
        # the cap boundary). 0 = off.
        self.opt_thd_max_spin = make_double_spinbox(
            min_val=0.0, max_val=100.0, value=0.0,
            step=0.5, decimals=1, suffix=" %",
            tooltip_key="amp.opt_thd_max_tip",
        )
        layout.addRow(t("amp.opt_thd_max") + ":", self.opt_thd_max_spin)

        # PP class-A power threshold: P_A = Iq² × Ra_aa / 8 (PP only)
        self.opt_class_a_mode_combo = QComboBox()
        self.opt_class_a_mode_combo.addItem(t("amp.class_a_off"), "off")
        self.opt_class_a_mode_combo.addItem(t("amp.class_a_absolute"), "absolute")
        self.opt_class_a_mode_combo.addItem(t("amp.class_a_percent"), "percent")
        self.opt_class_a_mode_combo.setToolTip(t("amp.class_a_mode_tip"))
        self.opt_class_a_mode_combo.currentIndexChanged.connect(self._on_class_a_mode_changed)
        self.opt_class_a_value_spin = make_double_spinbox(
            min_val=0.0, max_val=100.0, value=0.0,
            step=0.5, suffix=" W",
            tooltip_key="amp.class_a_value_tip",
        )
        self.opt_class_a_value_spin.setEnabled(False)
        class_a_row = QHBoxLayout()
        class_a_row.addWidget(self.opt_class_a_mode_combo)
        class_a_row.addWidget(self.opt_class_a_value_spin)
        self.opt_class_a_label = QLabel(t("amp.class_a_label") + ":")
        layout.addRow(self.opt_class_a_label, class_a_row)

        self.opt_ub_min_spin = make_double_spinbox(
            min_val=50.0, max_val=500.0, value=200.0,
            step=10.0, suffix=" V",
        )
        self.opt_ub_max_spin = make_double_spinbox(
            min_val=50.0, max_val=500.0, value=350.0,
            step=10.0, suffix=" V",
        )
        ub_row = QHBoxLayout()
        ub_row.addWidget(self.opt_ub_min_spin)
        ub_row.addWidget(QLabel("–"))
        ub_row.addWidget(self.opt_ub_max_spin)
        self.opt_ub_label = QLabel(t("amp.opt_ub_range") + ":")
        layout.addRow(self.opt_ub_label, ub_row)

        self.opt_ug2_min_spin = make_double_spinbox(
            min_val=50.0, max_val=500.0, value=150.0,
            step=10.0, suffix=" V",
        )
        self.opt_ug2_max_spin = make_double_spinbox(
            min_val=50.0, max_val=500.0, value=300.0,
            step=10.0, suffix=" V",
        )
        ug2_row = QHBoxLayout()
        ug2_row.addWidget(self.opt_ug2_min_spin)
        ug2_row.addWidget(QLabel("–"))
        ug2_row.addWidget(self.opt_ug2_max_spin)
        self.opt_ug2_label = QLabel(t("amp.opt_ug2_range") + ":")
        layout.addRow(self.opt_ug2_label, ug2_row)

        self.opt_run_btn = QPushButton(t("amp.opt_run"))
        self.opt_run_btn.setToolTip(t("amp.opt_run_tip"))
        self.opt_run_btn.clicked.connect(self.optimize_requested.emit)

        self.opt_cancel_btn = QPushButton(t("amp.opt_cancel"))
        self.opt_cancel_btn.setToolTip(t("amp.opt_cancel_tip"))
        self.opt_cancel_btn.setVisible(False)

        self.opt_pareto_btn = QPushButton(t("amp.opt_pareto"))
        self.opt_pareto_btn.setCheckable(True)
        self.opt_pareto_btn.setChecked(False)
        self.opt_pareto_btn.setToolTip(t("amp.opt_enable_tip"))

        # Post-run actions: enabled after optimize_measurements/_pp/_model
        # finishes. Disabled again at next Run.
        self.opt_apply_btn = QPushButton(t("amp.opt_apply_best"))
        self.opt_apply_btn.setToolTip(t("amp.opt_apply_best_tip"))
        self.opt_apply_btn.setEnabled(False)
        self.opt_apply_btn.clicked.connect(self.optimizer_apply_best.emit)

        self.opt_top_n_btn = QPushButton(t("amp.opt_top_n"))
        self.opt_top_n_btn.setToolTip(t("amp.opt_top_n_tip"))
        self.opt_top_n_btn.setEnabled(False)
        self.opt_top_n_btn.clicked.connect(self.optimizer_show_top_n.emit)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.opt_run_btn)
        btn_row.addWidget(self.opt_cancel_btn)
        btn_row.addWidget(self.opt_pareto_btn)
        btn_row.addWidget(self.opt_apply_btn)
        btn_row.addWidget(self.opt_top_n_btn)
        btn_row.addStretch(1)
        # Span both columns of QFormLayout — no label space wasted on the
        # left, buttons get full width.
        layout.addRow(btn_row)

        self.opt_progress = QProgressBar()
        self.opt_progress.setRange(0, 100)
        self.opt_progress.setVisible(False)
        self.opt_progress.setTextVisible(True)
        layout.addRow(self.opt_progress)

        self.opt_status_label = QLabel("")
        self.opt_status_label.setWordWrap(True)
        self.opt_status_label.setTextFormat(Qt.TextFormat.RichText)
        self.opt_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addRow(self.opt_status_label)

        _finalize_collapsible(gb)
        return gb

    def optimizer_constraints(self) -> "OptimizerConstraints":
        """Read optimizer settings from UI controls."""
        from lm19.optimizer import OptimizerConstraints

        # ML-076: gate on the collapsible's checked STATE (the user's
        # explicit expand), not on isVisible() — visibility is also False
        # when the panel/window is hidden or offscreen, which silently
        # dropped explicitly configured ranges.
        ranges_enabled = self.opt_group.isChecked()

        ub_range = None
        if ranges_enabled:
            ub_min = self.opt_ub_min_spin.value()
            ub_max = self.opt_ub_max_spin.value()
            if ub_max > ub_min:
                ub_range = (ub_min, ub_max)

        ug2_range = None
        if ranges_enabled:
            ug2_min = self.opt_ug2_min_spin.value()
            ug2_max = self.opt_ug2_max_spin.value()
            if ug2_max > ug2_min:
                ug2_range = (ug2_min, ug2_max)

        circuit = self.circuit_combo.currentData() or CIRCUIT_SE
        class_a_mode = self.opt_class_a_mode_combo.currentData() or "off"
        hd_method = self.hd_method_combo.currentData() or HD_METHOD_5POINT
        return OptimizerConstraints(
            target=self.opt_target_combo.currentData() or "min_thd",
            pa_max_w=self.pa_max_spin.value(),
            pout_min_w=self.opt_pout_min_spin.value(),
            thd_max_pct=self.opt_thd_max_spin.value(),
            balanced_weight=self.opt_balanced_weight_spin.value(),
            hd_method=hd_method,
            class_a_power_mode=class_a_mode,
            class_a_power_value=self.opt_class_a_value_spin.value(),
            circuit=circuit,
            ra_dc=self.xfmr_ra_dc_spin.value(),
            cf_rk=self.cf_rk_spin.value(),
            cf_rl=self.cf_rl_spin.value(),
            pp_raa=self.pp_raa_spin.value(),
            pp_ra_dc=self.pp_ra_dc_spin.value(),
            ug1_range=(self.ug1_spin.minimum(), self.ug1_spin.maximum()),
            ra_range=(_OPT_RA_MIN_KOHM,
                      max(self.ra_spin.value() * _OPT_RA_SPAN_FACTOR,
                          _OPT_RA_MAX_FLOOR_KOHM)),
            ub_range=ub_range,
            ug2_range=ug2_range,
            ul_tap_mode=self.ul_sweep_mode_combo.currentData() or "off",
            ul_tap_manual=self.ul_tap_spin.value() / 100.0,
            ul_tap_presets=tuple(t for t, _ in self._ul_preset_specs),
            ul_tap_presets_enabled=tuple(
                self.ul_preset_cbs[t].isChecked() for t, _ in self._ul_preset_specs
            ),
            ul_tap_range=(
                self.ul_sweep_min_spin.value() / 100.0,
                self.ul_sweep_max_spin.value() / 100.0,
            ),
            ul_tap_steps=self.ul_sweep_steps_spin.value(),
        )

    @property
    def optimizer_enabled(self) -> bool:
        return self.opt_pareto_btn.isChecked()

    def set_ug2_display(self, ug2: Optional[float]) -> None:
        """Update Ug2 info label in Source & Data section."""
        if ug2 is not None and ug2 > 0:
            self.ug2_info_label.setText(f"Ug2={ug2:.0f}V")
        else:
            self.ug2_info_label.setText("")

    def set_optimizer_model_mode(self, has_model: bool) -> None:
        """Update optimizer indicator based on data source.

        Model mode: continuous Ub/Ug2 sweep.
        Measurements mode: Ub from grid, Ug2 from available data values.
        """
        if has_model:
            self.opt_mode_label.setText(t("amp.opt_mode_model"))
        else:
            self.opt_mode_label.setText(t("amp.opt_mode_measurements"))

    def set_optimizer_status(self, text: str) -> None:
        self.opt_status_label.setText(text)

    def append_optimizer_status(self, line: str) -> None:
        """Append a transient line (e.g. 'Applied: ...') below current status
        without losing the optimization result info (HD method, warnings)."""
        current = self.opt_status_label.text()
        if current and line:
            self.opt_status_label.setText(f"{current}\n{line}")
        else:
            self.opt_status_label.setText(line or current)

    def set_optimizer_running(self, running: bool) -> None:
        """Toggle UI between running/idle states."""
        self.opt_run_btn.setVisible(not running)
        self.opt_cancel_btn.setVisible(running)
        self.opt_progress.setVisible(running)
        if running:
            self.opt_progress.setValue(0)
            self.opt_status_label.setText("")
            # Disable post-run actions while a new run is in progress
            self.set_optimizer_result_available(False)
        else:
            self.opt_progress.setVisible(False)

    def set_optimizer_result_available(self, available: bool) -> None:
        """Toggle Apply best / Top-N buttons based on result availability."""
        self.opt_apply_btn.setEnabled(available)
        self.opt_top_n_btn.setEnabled(available)

    def set_optimizer_progress(self, pct: int, phase: str) -> None:
        """Update progress bar value and phase text."""
        self.opt_progress.setValue(pct)
        self.opt_progress.setFormat(f"{phase}  %p%")

    # ── Section 5: Results ────────────────────────────────────────

    def _build_results_section(self) -> QGroupBox:
        gb = _collapsible_group(t("amp.section_results"))
        gb.setToolTip(t("amp.section_results_tip"))
        layout = QVBoxLayout(gb)
        layout.setContentsMargins(2, 2, 2, 2)

        self.results_label = QLabel(t("amp.no_data"))
        self.results_label.setWordWrap(True)
        self.results_label.setTextFormat(Qt.TextFormat.RichText)
        self.results_label.setStyleSheet("QLabel { padding: 4px; }")
        self.results_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        scroll = QScrollArea()
        scroll.setWidget(self.results_label)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(RESULTS_MIN_HEIGHT)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(scroll)

        self.export_pdf_btn = QPushButton(t("report.Amp_export_btn"))
        self.export_pdf_btn.setToolTip(t("report.Tip_amp_export"))
        self.export_pdf_btn.clicked.connect(self.export_pdf_requested)
        layout.addWidget(self.export_pdf_btn)

        verify_row = QHBoxLayout()
        self.verify_btn = QPushButton(t("report.Verify_btn"))
        self.verify_btn.setToolTip(t("report.Tip_verify_btn"))
        self.verify_btn.clicked.connect(self.verify_requested)
        verify_row.addWidget(self.verify_btn)
        self.verify_cancel_btn = QPushButton(t("report.Verify_cancel_btn"))
        self.verify_cancel_btn.setToolTip(t("report.Tip_verify_cancel"))
        self.verify_cancel_btn.setEnabled(False)
        self.verify_cancel_btn.clicked.connect(self.verify_cancel_requested)
        verify_row.addWidget(self.verify_cancel_btn)
        self.verify_fitter_combo = QComboBox()
        self.verify_fitter_combo.setToolTip(t("report.Tip_verify_fitter"))
        self.verify_fitter_combo.addItem(t("report.Verify_fitter_auto"), "")
        # same fitter registry the SPICE-export dialog offers
        from app.export_manager import _SPICE_MODEL_CHOICES
        for key, label in _SPICE_MODEL_CHOICES:
            self.verify_fitter_combo.addItem(label, key)
        verify_row.addWidget(self.verify_fitter_combo)
        verify_row.addStretch(1)
        layout.addLayout(verify_row)

        # run options on their own row — the buttons row is crowded
        verify_opts_row = QHBoxLayout()
        self.verify_sweep_cb = QCheckBox(t("report.Verify_sweep_cb"))
        self.verify_sweep_cb.setToolTip(t("report.Tip_verify_sweep"))
        verify_opts_row.addWidget(self.verify_sweep_cb)
        self.verify_imd_cb = QCheckBox(t("report.Verify_imd_cb"))
        self.verify_imd_cb.setToolTip(t("report.Tip_verify_imd"))
        verify_opts_row.addWidget(self.verify_imd_cb)
        verify_opts_row.addStretch(1)
        layout.addLayout(verify_opts_row)

        self.verify_status_label = QLabel("")
        self.verify_status_label.setWordWrap(True)
        self.verify_status_label.setTextFormat(Qt.TextFormat.RichText)
        self.verify_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.verify_status_label)

        return gb

    # ── Public API ────────────────────────────────────────────────

    def connect_params_changed(self, slot) -> None:
        """Working-line live-layer subscription: controls that affect
        params_snapshot() trigger the slot; debounce is on the
        listener side (WorkingLineController)."""
        for w in (self.ub_spin, self.ra_spin, self.ug1_spin,
                  self.swing_spin, self.xfmr_ra_dc_spin,
                  self.cf_rk_spin, self.cf_rl_spin,
                  self.pp_raa_spin, self.pp_ra_dc_spin,
                  self.ul_tap_spin, self.pa_max_spin):
            w.valueChanged.connect(slot)
        for c in (self.circuit_combo, self.hd_method_combo,
                  self.source_combo, self.pp_tube_b_combo):
            c.currentIndexChanged.connect(slot)
        # Changing the series (source_combo), the source set
        # (data_source_combo) and the matched/mismatched pair ALSO
        # changes params_snapshot — without these connects the live
        # line kept stale data (found by the call-site != function
        # checklist).
        self.data_source_combo.selectionChanged.connect(slot)
        self.pp_matched_btn.toggled.connect(slot)

    def params_snapshot(self) -> AmpParams:
        """Read all controls into an AmpParams dataclass."""
        circuit = self.circuit_combo.currentData() or CIRCUIT_SE
        swing = self.swing_spin.value()

        # Data source → sources list
        sources = self._selected_sources()

        # HD method
        hd_method = self.hd_method_combo.currentData() or HD_METHOD_AUTO

        # Series ID
        series_id = self.source_combo.currentData()

        # NFB
        nfb_db = self.nfb_spin.value() if self.nfb_check.isChecked() else None

        # Ultralinear tap (PP pentode only; 0% = disabled)
        ul_tap_pct = self.ul_tap_spin.value()
        ul_tap = ul_tap_pct / 100.0 if circuit == CIRCUIT_PP and ul_tap_pct > 0 else None

        # PP Tube B
        pp_tube_b_sid = None
        if not self.pp_matched_btn.isChecked():
            pp_tube_b_sid = self.pp_tube_b_combo.currentData()

        return AmpParams(
            ub=self.ub_spin.value(),
            ra=self.ra_spin.value(),
            ug1_bias=self.ug1_spin.value(),
            half_swing=swing if swing > MIN_HALF_SWING_V else None,
            circuit=circuit,
            pa_max=self.pa_max_spin.value(),
            sources=sources,
            hd_method=hd_method,
            series_id=series_id,
            nfb_db=nfb_db,
            ul_tap=ul_tap,
            ra_dc=self.xfmr_ra_dc_spin.value(),
            cf_rk=self.cf_rk_spin.value(),
            cf_rl=self.cf_rl_spin.value(),
            pp_raa=self.pp_raa_spin.value(),
            pp_ra_dc=self.pp_ra_dc_spin.value(),
            pp_matched=self.pp_matched_btn.isChecked(),
            pp_tube_b_sid=pp_tube_b_sid,
            show_hd45=self.show_hd45_cb.isChecked(),
            show_gzp=self.show_gzp_cb.isChecked(),
        )

    def show_results(self, html: str) -> None:
        """Display results HTML in the results label."""
        self.results_label.setText(html)

    def show_verify_results(self, html: str) -> None:
        """Display the LTspice-verification comparison table."""
        self.verify_status_label.setText(html)

    def set_verify_running(self, running: bool) -> None:
        """Toggle the verify/cancel buttons for a worker run."""
        self.verify_btn.setEnabled(not running)
        self.verify_cancel_btn.setEnabled(running)

    def set_series_items(
        self, labels: Dict[int, str], *, current_sid: Optional[int] = None,
    ) -> None:
        """Populate the measurement series combo."""
        self._block_signals = True
        self.source_combo.clear()
        for sid, label in labels.items():
            self.source_combo.addItem(label, sid)
        if current_sid is not None:
            idx = self.source_combo.findData(current_sid)
            if idx >= 0:
                self.source_combo.setCurrentIndex(idx)
        self._block_signals = False

    def set_pp_tube_b_items(self, labels: Dict[int, str]) -> None:
        """Populate the PP Tube B series combo."""
        self._block_signals = True
        self.pp_tube_b_combo.clear()
        for sid, label in labels.items():
            self.pp_tube_b_combo.addItem(label, sid)
        self._block_signals = False

    def set_available_models(self, model_labels: Dict[str, str]) -> None:
        """Update data source combo with available models.

        Args:
            model_labels: {source_key: display_label}, e.g.
                          {"koren": "Koren fit", "dempwolf": "Dempwolf fit"}
        """
        self._populate_data_source_combo(model_labels)

    def selected_series_id(self) -> Optional[int]:
        """Return currently selected measurement series ID."""
        return self.source_combo.currentData()

    # ── Internal ──────────────────────────────────────────────────

    def _populate_data_source_combo(
        self, model_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Rebuild data_source_combo items as CheckableComboBox.

        New items are checked by default. Previously unchecked items
        stay unchecked.
        """
        # Remember which keys were explicitly unchecked
        prev_unchecked: set = set()
        if hasattr(self, "data_source_combo") and self.data_source_combo.model().rowCount() > 0:
            all_keys = set()
            checked = set(self.data_source_combo.checked_keys())
            for row in range(self.data_source_combo.model().rowCount()):
                item = self.data_source_combo.model().item(row)
                all_keys.add(item.data(Qt.ItemDataRole.UserRole))
            prev_unchecked = all_keys - checked

        items = [("measurements", t("amp.source_measurements"))]
        if model_labels:
            for key, label in model_labels.items():
                items.append((key, label))

        # Check all except previously unchecked
        checked_keys = [key for key, _ in items if key not in prev_unchecked]
        self.data_source_combo.set_string_items(items, checked_keys=checked_keys)

    def _selected_sources(self) -> List[str]:
        """Return selected data source(s) as list."""
        keys = self.data_source_combo.checked_keys()
        return keys if keys else ["measurements"]

    def _on_circuit_changed(self) -> None:
        self._update_circuit_widgets()
        self._on_setting_changed()

    def _update_circuit_widgets(self) -> None:
        """Show/hide circuit-specific controls."""
        circuit = self.circuit_combo.currentData() or CIRCUIT_SE
        self._xfmr_widget.setVisible(circuit == CIRCUIT_SE_XFMR)
        self._cf_widget.setVisible(circuit == CIRCUIT_CF)
        self._pp_widget.setVisible(circuit == CIRCUIT_PP)
        # Update optimizer status for circuit type
        if hasattr(self, "opt_run_btn"):
            self.opt_run_btn.setEnabled(True)

    def _on_pp_matched_toggled(self, checked: bool) -> None:
        self._pp_tube_b_label.setVisible(not checked)
        self.pp_tube_b_combo.setVisible(not checked)
        self._on_setting_changed()

    def _on_ul_sweep_mode_changed(self) -> None:
        mode = self.ul_sweep_mode_combo.currentData() or "off"
        self._update_ul_sweep_visibility(mode)
        self._on_setting_changed()

    def _update_ul_sweep_visibility(self, mode: str) -> None:
        """Show preset checkboxes for presets/both, custom range for custom/both."""
        show_presets = mode in ("presets", "presets_custom")
        show_custom = mode in ("custom", "presets_custom")
        # Toggle row visibility via QFormLayout label widgets
        self._ul_presets_label.setVisible(show_presets)
        for cb in self.ul_preset_cbs.values():
            cb.setVisible(show_presets)
        self._ul_custom_label.setVisible(show_custom)
        self.ul_sweep_min_spin.setVisible(show_custom)
        self.ul_sweep_max_spin.setVisible(show_custom)
        self.ul_sweep_steps_spin.setVisible(show_custom)

    def _on_target_changed(self) -> None:
        """Show balanced_weight controls only when target=balanced."""
        is_balanced = self.opt_target_combo.currentData() == "balanced"
        self.opt_balanced_weight_label.setVisible(is_balanced)
        self.opt_balanced_weight_spin.setVisible(is_balanced)

    def _on_class_a_mode_changed(self) -> None:
        """Update spinbox state/range/suffix based on class-A mode."""
        mode = self.opt_class_a_mode_combo.currentData() or "off"
        if mode == "off":
            self.opt_class_a_value_spin.setEnabled(False)
        elif mode == "absolute":
            self.opt_class_a_value_spin.setEnabled(True)
            self.opt_class_a_value_spin.setRange(0.0, 100.0)
            self.opt_class_a_value_spin.setSingleStep(0.5)
            self.opt_class_a_value_spin.setSuffix(" W")
        else:  # percent
            self.opt_class_a_value_spin.setEnabled(True)
            self.opt_class_a_value_spin.setRange(0.0, 100.0)
            self.opt_class_a_value_spin.setSingleStep(5.0)
            self.opt_class_a_value_spin.setSuffix(" %")

    def _on_nfb_toggled(self, checked: bool) -> None:
        self.nfb_spin.setEnabled(checked)
        self._on_setting_changed()

    def _on_setting_changed(self) -> None:
        if not self._block_signals:
            self.settings_changed.emit()
