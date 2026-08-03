"""Reusable builder for the Health tab "Plan settings" QGroupBox.

The builder returns a ``(QGroupBox, PlanWidgets)`` tuple. ``PlanWidgets``
is a dataclass holding references to every widget the rest of HealthTab
needs to read/write; HealthTab unpacks it back into flat ``self.<name>``
attributes so call sites can read them as ``self.plan_*``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget,
)

from app.ui_theme import apply_tight
from app.widget_factory import make_double_spinbox, make_int_spinbox
from i18n_setup import t
from lm19.health import EMISSION_MODE_SINGLE, EMISSION_MODE_SWEEP
from lm19.protocol import (
    UA_RESOLUTION_V, UG1_RESOLUTION_V, UG2_RESOLUTION_V,
)

# ── module local constants ──
# Ordered so the combo index maps to a code without a parallel list; the
# codes themselves come from the owning registry in lm19/health.py.
EMISSION_MODE_ITEMS = (
    (EMISSION_MODE_SINGLE, "health.Emission_mode_single"),
    (EMISSION_MODE_SWEEP, "health.Emission_mode_sweep"),
)


@dataclass
class PlanWidgets:
    """Widget refs returned by ``build_plan_box``.

    HealthTab unpacks each field to ``self.<name>`` to preserve the
    ``self.plan_*`` attribute API used throughout health_tab.py.
    """
    # Operating-point targets
    plan_ua_target: QDoubleSpinBox
    plan_ug1_target: QDoubleSpinBox
    plan_ug2_target: QDoubleSpinBox
    # Δ deltas for SRK measurement
    plan_delta_ua: QDoubleSpinBox
    plan_delta_ug1: QDoubleSpinBox
    plan_delta_ug2: QDoubleSpinBox
    # SRK sweep settings
    plan_emission_ratio: QDoubleSpinBox
    plan_points: QSpinBox
    plan_repeats: QSpinBox
    # Action button + validation feedback
    plan_reset_btn: QPushButton
    plan_validation_label: QLabel
    # Ug2 mode selector (independent vs. track)
    ug2_mode_group: QButtonGroup
    ug2_independent_radio: QRadioButton
    ug2_track_radio: QRadioButton
    ug2_offset: QDoubleSpinBox
    # Emission mode + bias servo
    plan_emission_mode: QComboBox
    plan_bias_servo: QCheckBox
    # Label refs (for show/hide on triode lamps)
    plan_ug2_label: QLabel
    plan_delta_ug2_label: QLabel
    plan_delta_ug2_unit: QLabel
    plan_emission_label: QLabel
    ug2_offset_label: QLabel
    ug2_mode_prefix_label: QLabel


def build_plan_box(
    parent: QWidget,
    *,
    on_ug2_mode_toggled: Callable[[bool], None],
) -> tuple[QGroupBox, PlanWidgets]:
    """Build the "Plan settings" QGroupBox.

    Args:
        parent: Owner widget (used as parent for the QButtonGroup so
            radio buttons get the right Qt cleanup chain).
        on_ug2_mode_toggled: callback for ``ug2_track_radio.toggled``.
            HealthTab uses this to refresh planned-step preview when the
            mode flips between independent and track.

    Returns:
        ``(group_box, widgets)`` — caller wires the box into a layout
        and unpacks the widget refs into instance attributes.
    """
    box = QGroupBox(t("health.Plan_settings"))
    plan_layout = QVBoxLayout(box)
    apply_tight(plan_layout)

    # ── Operating-point target spinboxes ──────────────────────────
    plan_ua_target = make_double_spinbox(
        min_val=0.0, max_val=1000.0, value=0.0,
        step=1.0, decimals=1, tooltip_key="tip.Health_plan_ua_target",
    )
    plan_ug1_target = make_double_spinbox(
        min_val=-100.0, max_val=0.0, value=0.0,
        step=0.1, decimals=2, tooltip_key="tip.Health_plan_ug1_target",
    )
    plan_ug2_target = make_double_spinbox(
        min_val=0.0, max_val=1000.0, value=0.0,
        step=1.0, decimals=1, tooltip_key="tip.Health_plan_ug2_target",
    )

    # ── Δ deltas ──
    plan_delta_ua = make_double_spinbox(
        min_val=UA_RESOLUTION_V * 5, max_val=100.0, value=0.0,
        step=UA_RESOLUTION_V, decimals=0,
        tooltip_key="tip.Health_plan_delta_ua",
    )
    plan_delta_ug1 = make_double_spinbox(
        min_val=UG1_RESOLUTION_V * 2, max_val=10.0, value=0.0,
        step=UG1_RESOLUTION_V, decimals=2,
        tooltip_key="tip.Health_plan_delta_ug1",
    )
    plan_delta_ug2 = make_double_spinbox(
        min_val=UG2_RESOLUTION_V * 5, max_val=100.0, value=0.0,
        step=UG2_RESOLUTION_V, decimals=0,
        tooltip_key="tip.Health_plan_delta_ug2",
    )
    plan_emission_ratio = make_double_spinbox(
        min_val=0.1, max_val=1.0, value=0.0,
        step=0.01, decimals=2,
        tooltip_key="tip.Health_plan_emission_ratio",
    )

    plan_points = make_int_spinbox(
        min_val=5, max_val=21, value=5, step=2,
        tooltip_key="tip.Health_plan_points",
    )
    plan_repeats = make_int_spinbox(
        min_val=1, max_val=50, value=5,
        tooltip_key="tip.Health_plan_repeats",
    )

    plan_emission_mode = QComboBox()
    for code, key in EMISSION_MODE_ITEMS:
        plan_emission_mode.addItem(t(key), code)
    plan_emission_mode.setToolTip(t("tip.Health_plan_emission_mode"))

    plan_bias_servo = QCheckBox(t("health.Plan_bias_servo"))
    plan_bias_servo.setToolTip(t("tip.Health_plan_bias_servo"))

    plan_reset_btn = QPushButton(t("health.Plan_apply_from_lamp"))
    plan_reset_btn.setToolTip(t("tip.Health_plan_apply_from_lamp"))
    plan_validation_label = QLabel("")
    plan_validation_label.setWordWrap(True)

    # ── Operating-point row ──
    op_row = QHBoxLayout()
    op_row.addWidget(QLabel(t("health.OP_short")))
    op_row.addWidget(QLabel(t("common.Ua")))
    op_row.addWidget(plan_ua_target)
    op_row.addWidget(QLabel(t("common.Ug1")))
    op_row.addWidget(plan_ug1_target)
    plan_ug2_label = QLabel(t("common.Ug2"))
    op_row.addWidget(plan_ug2_label)
    op_row.addWidget(plan_ug2_target)
    op_row.addStretch(1)
    plan_layout.addLayout(op_row)

    # ── Ug2 mode selector + offset ──
    ug2_mode_group = QButtonGroup(parent)
    ug2_independent_radio = QRadioButton(t("health.Ug2_independent"))
    ug2_track_radio = QRadioButton(t("health.Ug2_track_ua"))
    ug2_independent_radio.setChecked(True)
    ug2_mode_group.addButton(ug2_independent_radio, 0)
    ug2_mode_group.addButton(ug2_track_radio, 1)
    ug2_offset = make_double_spinbox(
        min_val=-100, max_val=100, value=0,
        decimals=1, tooltip_key="tip.Health_ug2_offset",
    )
    # setEnabled isn't covered by the factory; the radio toggle wires it up
    # below and the initial state must be disabled (independent mode default).
    ug2_offset.setEnabled(False)
    ug2_offset_label = QLabel(t("health.Ug2_offset"))
    ug2_track_radio.toggled.connect(ug2_offset.setEnabled)
    ug2_track_radio.toggled.connect(on_ug2_mode_toggled)
    ug2_independent_radio.setToolTip(t("tip.Health_ug2_independent"))
    ug2_track_radio.setToolTip(t("tip.Health_ug2_track_ua"))
    ug2_mode_row = QHBoxLayout()
    ug2_mode_prefix_label = QLabel(t("health.Ug2_mode"))
    ug2_mode_row.addWidget(ug2_mode_prefix_label)
    ug2_mode_row.addWidget(ug2_independent_radio)
    ug2_mode_row.addWidget(ug2_track_radio)
    ug2_mode_row.addWidget(ug2_offset_label)
    ug2_mode_row.addWidget(ug2_offset)
    ug2_mode_row.addStretch(1)
    plan_layout.addLayout(ug2_mode_row)

    # ── Δ row ──
    delta_row = QHBoxLayout()
    delta_row.addWidget(QLabel(t("health.Delta_symbol")))
    delta_row.addWidget(QLabel(t("common.Ua")))
    delta_row.addWidget(plan_delta_ua)
    delta_row.addWidget(QLabel(t("common.V")))
    delta_row.addSpacing(8)
    delta_row.addWidget(QLabel(t("common.Ug1")))
    delta_row.addWidget(plan_delta_ug1)
    delta_row.addWidget(QLabel(t("common.V")))
    delta_row.addSpacing(8)
    plan_delta_ug2_label = QLabel(t("common.Ug2"))
    delta_row.addWidget(plan_delta_ug2_label)
    delta_row.addWidget(plan_delta_ug2)
    plan_delta_ug2_unit = QLabel(t("common.V"))
    delta_row.addWidget(plan_delta_ug2_unit)
    delta_row.addStretch(1)
    plan_layout.addLayout(delta_row)

    # ── SRK row (points + repeats + emission ratio) ──
    srk_row = QHBoxLayout()
    srk_row.addWidget(QLabel(t("health.SRK_short")))
    srk_row.addWidget(QLabel(t("health.Plan_points")))
    srk_row.addWidget(plan_points)
    srk_row.addSpacing(8)
    srk_row.addWidget(QLabel(t("common.N")))
    srk_row.addWidget(plan_repeats)
    srk_row.addSpacing(8)
    plan_emission_label = QLabel(t("health.Plan_emission_ratio"))
    srk_row.addWidget(plan_emission_label)
    srk_row.addWidget(plan_emission_ratio)
    srk_row.addStretch(1)
    plan_layout.addLayout(srk_row)

    # ── Accuracy row (emission mode + bias servo) ──
    accuracy_row = QHBoxLayout()
    plan_emission_mode_label = QLabel(t("health.Plan_emission_mode"))
    accuracy_row.addWidget(plan_emission_mode_label)
    accuracy_row.addWidget(plan_emission_mode)
    accuracy_row.addSpacing(8)
    accuracy_row.addWidget(plan_bias_servo)
    accuracy_row.addStretch(1)
    plan_layout.addLayout(accuracy_row)

    # ── Action row (apply-from-lamp + validation feedback) ──
    plan_actions_row = QHBoxLayout()
    plan_actions_row.addWidget(plan_reset_btn)
    plan_actions_row.addWidget(plan_validation_label, 1)
    plan_layout.addLayout(plan_actions_row)

    widgets = PlanWidgets(
        plan_ua_target=plan_ua_target,
        plan_ug1_target=plan_ug1_target,
        plan_ug2_target=plan_ug2_target,
        plan_delta_ua=plan_delta_ua,
        plan_delta_ug1=plan_delta_ug1,
        plan_delta_ug2=plan_delta_ug2,
        plan_emission_ratio=plan_emission_ratio,
        plan_emission_mode=plan_emission_mode,
        plan_bias_servo=plan_bias_servo,
        plan_points=plan_points,
        plan_repeats=plan_repeats,
        plan_reset_btn=plan_reset_btn,
        plan_validation_label=plan_validation_label,
        ug2_mode_group=ug2_mode_group,
        ug2_independent_radio=ug2_independent_radio,
        ug2_track_radio=ug2_track_radio,
        ug2_offset=ug2_offset,
        plan_ug2_label=plan_ug2_label,
        plan_delta_ug2_label=plan_delta_ug2_label,
        plan_delta_ug2_unit=plan_delta_ug2_unit,
        plan_emission_label=plan_emission_label,
        ug2_offset_label=ug2_offset_label,
        ug2_mode_prefix_label=ug2_mode_prefix_label,
    )
    return box, widgets
