"""Match panel widget for Tube Health tab — settings & summary for tube matching."""

import logging
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.widget_factory import make_double_spinbox

from app.ui_theme import (
    apply_tight, apply_no_margin,
    HEALTH_MATCH_GROUP_SIZE_MIN, HEALTH_MATCH_GROUP_SIZE_MAX,
    HEALTH_MATCH_MAX_DELTA_LIMIT,
    HEALTH_MATCH_DELTA_EXCELLENT, HEALTH_MATCH_DELTA_GOOD,
    HEALTH_MATCH_DELTA_FAIR, HEALTH_MATCH_DELTA_POOR,
    DELTA_QUALITY_HEX_MAP,
)
from i18n_setup import t
from lm19.app_config import AppConfig
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)
from lm19.tube_matching import (
    DEFAULT_WEIGHTS_PENTODE, MatchResult,
    default_weights_for_mode, delta_quality,
    DELTA_EXCELLENT, DELTA_GOOD, DELTA_FAIR,
    MATCHING_PROTOCOL_STRICT, MATCHING_PROTOCOL_SHARED,
    MATCHING_PROTOCOL_INDIVIDUAL, MATCHING_PROTOCOLS,
)

# Registry code → i18n label key. The completeness pin derives the combo
# contents from MATCHING_PROTOCOLS, so a protocol added to the registry
# without a row here fails CI instead of silently missing from the UI.
PROTOCOL_ITEMS = (
    (MATCHING_PROTOCOL_STRICT, "health.match_Protocol_strict"),
    (MATCHING_PROTOCOL_SHARED, "health.match_Protocol_shared"),
    (MATCHING_PROTOCOL_INDIVIDUAL, "health.match_Protocol_individual"),
)

log = logging.getLogger(__name__)


class MatchPanel(QWidget):
    """Left-panel widget for the Match tab.

    Emits ``calculate_requested`` with config dict when the user clicks Calculate.
    """

    calculate_requested = Signal(dict)
    clear_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._result: Optional[MatchResult] = None
        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        apply_no_margin(layout)

        # Info
        self.info_label = QLabel()
        layout.addWidget(self.info_label)

        # Settings group
        settings_box = QGroupBox(t("health.match_tab_Match"))
        form = QVBoxLayout(settings_box)
        apply_tight(form)

        # Conditions
        self.conditions_label = QLabel("—")
        form.addWidget(self.conditions_label)

        # Grid: 3 rows × 4 cols (label, control, label, control)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        # Row 0: Tube mode | Match mode
        self.tube_mode_combo = QComboBox()
        self.tube_mode_combo.setToolTip(t("tip.Health_match_tube_mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(t("health.match_Mode_groups"), "groups")
        self.mode_combo.addItem(t("health.match_Mode_similar"), "similar")
        self.mode_combo.setToolTip(t("tip.Health_match_mode"))
        grid.addWidget(QLabel(t("health.match_Tube_mode")), 0, 0)
        grid.addWidget(self.tube_mode_combo, 0, 1)
        grid.addWidget(QLabel(t("health.match_Mode")), 0, 2)
        grid.addWidget(self.mode_combo, 0, 3)

        # Row 1: Use | Anode
        self.use_combo = QComboBox()
        self.use_combo.addItem(t("health.match_Use_latest"), "latest")
        self.use_combo.addItem(t("health.match_Use_best"), "best")
        self.use_combo.setToolTip(t("tip.Health_match_use"))
        self.anode_combo = QComboBox()
        self.anode_combo.addItem(t("health.match_Anode_each"), "each")
        self.anode_combo.addItem(t("health.match_Anode_combined"), "combined")
        self.anode_combo.setToolTip(t("tip.Health_match_anode"))
        grid.addWidget(QLabel(t("health.match_Use")), 1, 0)
        grid.addWidget(self.use_combo, 1, 1)
        grid.addWidget(QLabel(t("health.match_Anode")), 1, 2)
        grid.addWidget(self.anode_combo, 1, 3)

        # Row 2: Max Δ | Size
        self.max_delta_spin = make_double_spinbox(
            min_val=0.0, max_val=HEALTH_MATCH_MAX_DELTA_LIMIT,
            value=0.0, decimals=1, step=0.5,
            tooltip_key="tip.Health_match_max_delta",
        )
        # setSpecialValueText not in factory — special "—" sentinel for value=min.
        self.max_delta_spin.setSpecialValueText("—")
        self._group_size_label = QLabel(t("health.match_Group_size"))
        self.group_size_combo = QComboBox()
        self.group_size_combo.setEditable(True)
        self.group_size_combo.setToolTip(t("tip.Health_match_group_size"))
        for n in range(HEALTH_MATCH_GROUP_SIZE_MIN, HEALTH_MATCH_GROUP_SIZE_MAX + 1):
            self.group_size_combo.addItem(str(n), n)
        self.group_size_combo.setCurrentIndex(0)
        grid.addWidget(QLabel(t("health.match_Max_delta")), 2, 0)
        grid.addWidget(self.max_delta_spin, 2, 1)
        grid.addWidget(self._group_size_label, 2, 2)
        grid.addWidget(self.group_size_combo, 2, 3)

        # Row 3a: Pair algorithm (only relevant when group_size == 2)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItem(t("health.match_Algo_greedy"), "greedy")
        self.algorithm_combo.addItem(t("health.match_Algo_optimal"), "optimal")
        self.algorithm_combo.setToolTip(t("tip.Health_match_algorithm"))
        grid.addWidget(QLabel(t("health.match_Algorithm")), 3, 0)
        grid.addWidget(self.algorithm_combo, 3, 1, 1, 3)

        # Row 4: Matching protocol — how the target amplifier biases its
        # tubes (decides pool, metric and pairwise gates in match_tubes).
        self.protocol_combo = QComboBox()
        for code, label_key in PROTOCOL_ITEMS:
            self.protocol_combo.addItem(t(label_key), code)
        self.protocol_combo.setToolTip(t("tip.Health_match_protocol"))
        grid.addWidget(QLabel(t("health.match_Protocol")), 4, 0)
        grid.addWidget(self.protocol_combo, 4, 1, 1, 3)

        form.addLayout(grid)

        # Row 3: Weights Ia[ ] S[ ] R[ ]
        row3 = QHBoxLayout()
        self.weight_ia_spin = self._make_weight_spin(DEFAULT_WEIGHTS_PENTODE["ia"])
        self.weight_ia_spin.setToolTip(t("tip.Health_match_weight_ia"))
        self.weight_s_spin = self._make_weight_spin(DEFAULT_WEIGHTS_PENTODE["s"])
        self.weight_s_spin.setToolTip(t("tip.Health_match_weight_s"))
        self.weight_r_spin = self._make_weight_spin(DEFAULT_WEIGHTS_PENTODE["r"])
        self.weight_r_spin.setToolTip(t("tip.Health_match_weight_r"))
        row3.addWidget(QLabel(t("health.match_Weights")))
        row3.addWidget(QLabel("Ia"))
        row3.addWidget(self.weight_ia_spin)
        row3.addWidget(QLabel("S"))
        row3.addWidget(self.weight_s_spin)
        row3.addWidget(QLabel("R"))
        row3.addWidget(self.weight_r_spin)
        row3.addStretch(1)
        form.addLayout(row3)

        # Row 4: [All] [Visible] [Selected] [Clear] ☐ Hide inactive
        row4 = QHBoxLayout()
        self.calc_all_btn = QPushButton(t("health.match_Calc_all"))
        self.calc_all_btn.setToolTip(t("tip.Health_match_calculate"))
        self.calc_filtered_btn = QPushButton(t("health.match_Calc_filtered"))
        self.calc_filtered_btn.setToolTip(t("tip.Health_match_calc_filtered"))
        self.calc_selected_btn = QPushButton(t("health.match_Calc_selected"))
        self.calc_selected_btn.setToolTip(t("tip.Health_match_calc_selected"))
        self.clear_btn = QPushButton(t("health.match_Clear"))
        self.clear_btn.setToolTip(t("tip.Health_match_clear"))
        self.clear_btn.setEnabled(False)
        self.hide_inactive_cb = QCheckBox(t("health.match_Hide_inactive"))
        self.hide_inactive_cb.setToolTip(t("tip.Health_match_hide_inactive"))
        row4.addWidget(self.calc_all_btn)
        row4.addWidget(self.calc_filtered_btn)
        row4.addWidget(self.calc_selected_btn)
        row4.addWidget(self.clear_btn)
        row4.addWidget(self.hide_inactive_cb)
        row4.addStretch(1)
        form.addLayout(row4)

        layout.addWidget(settings_box)

        # Summary
        summary_box = QGroupBox(t("health.match_Summary"))
        summary_layout = QVBoxLayout(summary_box)
        apply_tight(summary_layout)
        self.summary_info = QLabel("—")
        self.summary_info.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        summary_layout.addWidget(self.summary_info)

        # Scrollable group list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._groups_container = QWidget()
        self._groups_layout = QVBoxLayout(self._groups_container)
        apply_tight(self._groups_layout)
        self._groups_layout.addStretch(1)
        scroll.setWidget(self._groups_container)
        summary_layout.addWidget(scroll, 1)
        layout.addWidget(summary_box, 1)

        # Action buttons
        btn_row = QHBoxLayout()
        self.copy_btn = QPushButton(t("health.match_Copy_groups"))
        self.copy_btn.setToolTip(t("tip.Health_match_copy_groups"))
        self.export_btn = QPushButton(t("health.match_Export_CSV"))
        self.export_btn.setToolTip(t("tip.Health_match_export_csv"))
        self.certificate_btn = QPushButton(t("report.Cert_btn"))
        self.certificate_btn.setToolTip(t("report.Tip_cert_btn"))
        self.copy_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.certificate_btn.setEnabled(False)
        btn_row.addWidget(self.copy_btn)
        btn_row.addWidget(self.export_btn)
        btn_row.addWidget(self.certificate_btn)
        layout.addLayout(btn_row)

        # Signals
        self.calc_all_btn.clicked.connect(lambda: self._on_calculate("all"))
        self.calc_filtered_btn.clicked.connect(lambda: self._on_calculate("filtered"))
        self.calc_selected_btn.clicked.connect(lambda: self._on_calculate("selected"))
        self.clear_btn.clicked.connect(self.clear_requested)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.tube_mode_combo.currentIndexChanged.connect(self._on_tube_mode_changed)
        self.protocol_combo.currentIndexChanged.connect(self._on_protocol_changed)
        self._on_mode_changed()
        self._on_protocol_changed()

    def _make_weight_spin(self, default: float) -> QDoubleSpinBox:
        return make_double_spinbox(
            min_val=0.0, max_val=1.0, value=default,
            decimals=2, step=0.05, fixed_width=58,
        )

    # ── Mode toggle ──────────────────────────────────────────────────

    def _on_mode_changed(self) -> None:
        is_groups = self.mode_combo.currentData() == "groups"
        self.group_size_combo.setVisible(is_groups)
        self._group_size_label.setVisible(is_groups)

    def _on_tube_mode_changed(self) -> None:
        """Update default weights when tube mode changes."""
        mode = self.tube_mode_combo.currentData()
        if mode:
            w = default_weights_for_mode(mode)
            self.weight_ia_spin.setValue(w["ia"])
            self.weight_s_spin.setValue(w["s"])
            self.weight_r_spin.setValue(w["r"])

    def _on_protocol_changed(self) -> None:
        """individual_bias zeroes the Ia weight inside match_tubes — grey
        the spin out so the UI does not promise a knob that has no effect."""
        individual = (self.protocol_combo.currentData()
                      == MATCHING_PROTOCOL_INDIVIDUAL)
        self.weight_ia_spin.setEnabled(not individual)
        self.weight_ia_spin.setToolTip(
            t("tip.Health_match_weight_ia_ignored") if individual
            else t("tip.Health_match_weight_ia"))

    # ── Calculate ────────────────────────────────────────────────────

    def _on_calculate(self, source: str = "all") -> None:
        config = self.get_config()
        config["source"] = source
        self.calculate_requested.emit(config)

    def _get_group_size(self) -> int:
        """Parse group size from editable combo, clamp to valid range."""
        data = self.group_size_combo.currentData()
        if data is not None:
            return int(data)
        try:
            val = int(self.group_size_combo.currentText())
            return max(HEALTH_MATCH_GROUP_SIZE_MIN,
                       min(HEALTH_MATCH_GROUP_SIZE_MAX, val))
        except (ValueError, TypeError):
            return HEALTH_MATCH_GROUP_SIZE_MIN

    def get_config(self) -> Dict[str, Any]:
        return {
            "mode": self.mode_combo.currentData(),
            "tube_mode": self.tube_mode_combo.currentData() or None,
            "use": self.use_combo.currentData(),
            "anode": self.anode_combo.currentData(),
            "group_size": self._get_group_size(),
            "max_delta": self.max_delta_spin.value(),
            "algorithm": self.algorithm_combo.currentData() or "greedy",
            "protocol": (self.protocol_combo.currentData()
                         or MATCHING_PROTOCOL_STRICT),
            "weights": {
                "ia": self.weight_ia_spin.value(),
                "s": self.weight_s_spin.value(),
                "r": self.weight_r_spin.value(),
            },
        }

    def set_algorithm(self, algorithm: str) -> None:
        """Pre-select pair-matching algorithm from saved settings."""
        idx = self.algorithm_combo.findData(algorithm)
        if idx >= 0:
            self.algorithm_combo.setCurrentIndex(idx)

    def set_protocol(self, protocol: str) -> None:
        """Pre-select matching protocol from saved settings."""
        idx = self.protocol_combo.findData(protocol)
        if idx >= 0:
            self.protocol_combo.setCurrentIndex(idx)

    # ── Load defaults from config ────────────────────────────────────

    def load_from_config(self, cfg: AppConfig) -> None:
        """Load default values from AppConfig."""
        self.weight_ia_spin.setValue(cfg.health_matching_weight_ia)
        self.weight_s_spin.setValue(cfg.health_matching_weight_s)
        self.weight_r_spin.setValue(cfg.health_matching_weight_r)
        self.max_delta_spin.setValue(cfg.health_matching_max_delta)
        idx = self.group_size_combo.findData(cfg.health_matching_group_size)
        if idx >= 0:
            self.group_size_combo.setCurrentIndex(idx)
        else:
            self.group_size_combo.setCurrentText(str(cfg.health_matching_group_size))
        # Use
        idx = self.use_combo.findData(cfg.health_matching_use)
        if idx >= 0:
            self.use_combo.setCurrentIndex(idx)
        # Anode
        idx = self.anode_combo.findData(cfg.health_matching_anode)
        if idx >= 0:
            self.anode_combo.setCurrentIndex(idx)

    # ── Update conditions display ────────────────────────────────────

    def set_conditions(self, ua: float, ug1: float, ug2: float,
                       ug2_mode: str = "", servo: bool = False) -> None:
        """Show the pool's operating point INCLUDING the bias-servo flag —
        servo and fixed-bias pools at the same Ua/Ug1/Ug2 are different
        pools and must not display identically."""
        text = f"Ua={ua:.0f}  Ug1={ug1:.1f}  Ug2={ug2:.0f}"
        if ug2_mode:
            text += f"  {self._MODE_LABELS.get(ug2_mode, ug2_mode)}"
        if servo:
            text += "  " + t("health.match_Cond_servo")
            if self.protocol_combo.currentData() == MATCHING_PROTOCOL_STRICT:
                # Every servo record sits at the reference current, so
                # under strict the Ia term barely discriminates — point
                # at the protocols built for servo pools.
                text += "\n" + t("health.match_Servo_pool_hint")
        self.conditions_label.setText(text)

    # ── Update available tube modes ──────────────────────────────────

    _MODE_LABELS = {TOPOLOGY_PENTODE: "Pent",
                    TOPOLOGY_TRIODE_CONNECTED: "TriC",
                    TOPOLOGY_TRIODE: "Tri"}

    def set_available_modes(self, mode_counts: Dict[str, int],
                            default_mode: str = "") -> None:
        """Populate tube mode combo from {ug2_mode: lamp_count} dict.

        Preserves current selection if still available.
        """
        prev = self.tube_mode_combo.currentData()
        # ML-045: bulk repopulation fires currentIndexChanged on every
        # addItem — _on_tube_mode_changed then STOMPS user-edited weights
        # even when the effective mode did not change. blockSignals is the
        # correct pattern for clear+addItem×N (same as refresh_*_combos);
        # a REAL mode transition is re-dispatched manually below.
        self.tube_mode_combo.blockSignals(True)
        try:
            self.tube_mode_combo.clear()
            for mode, count in sorted(mode_counts.items()):
                label = self._MODE_LABELS.get(mode, mode)
                self.tube_mode_combo.addItem(f"{label} ({count})", mode)

            # Restore previous or set default
            target = prev if prev and self.tube_mode_combo.findData(prev) >= 0 else default_mode
            idx = self.tube_mode_combo.findData(target)
            if idx >= 0:
                self.tube_mode_combo.setCurrentIndex(idx)
        finally:
            self.tube_mode_combo.blockSignals(False)
        if self.tube_mode_combo.currentData() != prev:
            self._on_tube_mode_changed()

    # ── Update info label ────────────────────────────────────────────

    def set_info(self, lamp_count: int, total_count: int,
                 source: str = "") -> None:
        text = t("health.match_Lamps_found", count=lamp_count, total=total_count)
        if source:
            text = f"{text}  [{source}]"
        self.info_label.setText(text)

    # ── Set "similar to" mode with anchor ────────────────────────────

    def set_similar_mode(self, anchor_lamp_id: str) -> None:
        idx = self.mode_combo.findData("similar")
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

    # ── Update summary from result ───────────────────────────────────

    def set_result(self, result: Optional[MatchResult]) -> None:
        self._result = result
        has_result = result is not None and bool(result.groups or result.unmatched)
        self.copy_btn.setEnabled(has_result)
        self.export_btn.setEnabled(has_result)
        self.certificate_btn.setEnabled(has_result)
        self.clear_btn.setEnabled(result is not None)
        self._clear_groups_list()

        if result is None or not result.groups:
            self.summary_info.setText(
                t("health.match_No_data") if result is None else "\u2014")
            return

        # Info line
        if result.mode == "similar":
            best_d = result.groups[0].delta if result.groups else 0
            self.summary_info.setText(
                t("health.match_Summary_similar",
                  count=len(result.groups), best=f"{best_d:.1f}"))
        else:
            size = len(result.groups[0].records) if result.groups else 0
            self.summary_info.setText(
                t("health.match_Summary_groups",
                  count=len(result.groups), size=size,
                  unmatched=len(result.unmatched)))

        # Group detail list
        best_num = result.groups[0].number if result.groups else 0
        for g in result.groups:
            star = " \u2605" if g.number == best_num else ""
            dq = delta_quality(g.delta)
            dq_color = DELTA_QUALITY_HEX_MAP.get(
                dq, DELTA_QUALITY_HEX_MAP["poor"])
            if result.mode == "similar":
                header = f"#{g.number}  \u0394={g.delta:.1f}%{star}"
            else:
                grp = t("health.match_Group_prefix")
                header = f"{grp} {g.number}  \u0394={g.delta:.1f}%{star}"
            # shared_bias protocol: predicted quiescent-current imbalance
            # of the group in a common-bias amplifier (method visibility \u2014
            # the gate that shaped the selection shows its number).
            if g.iq_imbalance_ma is not None:
                header += "  " + t("health.match_dIq",
                                   ma=f"{g.iq_imbalance_ma:.1f}")
            self._add_group_label(
                f"<b><span style='color:{dq_color}'>{header}</span></b>")
            for rec in g.records:
                self._add_group_label(
                    f"  {rec.lamp_id}  Ia:{rec.ia:.1f}  S:{rec.s:.2f}", indent=8)

        if result.unmatched:
            un = t("health.match_Unmatched_label")
            self._add_group_label(f"<b>\u2014 {un} \u2014</b>")
            ids = ", ".join(r.lamp_id for r in result.unmatched)
            self._add_group_label(f"  {ids}", indent=8)

    def _add_group_label(self, text: str, indent: int = 0) -> None:
        label = QLabel(text)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if indent:
            label.setIndent(indent)
        # Insert before the trailing stretch
        self._groups_layout.insertWidget(
            self._groups_layout.count() - 1, label)

    def _clear_groups_list(self) -> None:
        """Remove all widgets from groups container except the stretch."""
        while self._groups_layout.count() > 1:
            item = self._groups_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
