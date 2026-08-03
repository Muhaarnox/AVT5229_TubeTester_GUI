"""Dialog shown when ``HealthProtectionError`` trips during OP-ramp.

Renders the structured ``HealthProtectionPayload`` into a single readable
message with every measured parameter, both the datasheet limit and the
configured safety limit, the ramp step where the trip happened, and a
short list of likely causes. Receives the payload from
``HealthWorker.protection_triggered``.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QWidget

from i18n_setup import t
from lm19.scan.exceptions import HealthProtectionPayload
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)


def _fmt_mode(payload: HealthProtectionPayload) -> str:
    """Render ``topology`` + ``ug2_mode`` as a single user-facing string."""
    key = {
        (TOPOLOGY_TRIODE, TOPOLOGY_TRIODE): "health.Protect_mode_triode",
        (TOPOLOGY_PENTODE, TOPOLOGY_TRIODE_CONNECTED):
            "health.Protect_mode_triode_connected",
        (TOPOLOGY_PENTODE, TOPOLOGY_PENTODE): "health.Protect_mode_pentode",
    }.get((payload.topology, payload.ug2_mode), "health.Protect_mode_unknown")
    return t(key)


def show_health_protection_dialog(
    parent: QWidget, payload: HealthProtectionPayload,
) -> None:
    """Display a critical-style QMessageBox describing the trip.

    ``payload`` is the ``HealthProtectionPayload`` from
    ``HealthProtectionError``. The dialog is modal and blocks until the
    user acknowledges; the caller is responsible for any cleanup before
    invoking this.
    """
    if payload.kind == "pa":
        title_key = "health.Protect_title_pa"
        kind_label = t("health.Protect_kind_pa")
    else:
        title_key = "health.Protect_title_pg2"
        kind_label = t("health.Protect_kind_pg2")

    datasheet = (
        f"{payload.datasheet_max_w:.2f} W"
        if payload.datasheet_max_w is not None
        else t("health.Protect_datasheet_na")
    )

    lines = []
    if payload.ug1_restore_failed:
        # The post-trip Ug1 safe-lock restore failed — the tube may still
        # be conducting. This must be the FIRST thing the operator reads.
        lines += [t("health.Protect_restore_failed"), ""]
    lines += [
        t("health.Protect_intro", kind=kind_label),
        "",
        t("health.Protect_step_line",
          step=payload.step_idx, total=payload.total_steps,
          start=f"{payload.start_ug1:.1f}",
          target=f"{payload.target_ug1:.1f}"),
        "",
        t("health.Protect_measured_header"),
        f"  Ua  = {payload.ua:.1f} V        Ia  = {payload.ia_ma:.2f} mA",
        f"  Ug1 = {payload.ug1:.2f} V       Ig2 = {payload.ig2_ma:.2f} mA",
        f"  Ug2 = {payload.ug2:.1f} V",
        "",
        t("health.Protect_power_line",
          kind=kind_label,
          measured=f"{payload.measured_w:.2f}",
          limit=f"{payload.limit_w:.2f}",
          pct=f"{payload.safety_pct:.0f}",
          datasheet=datasheet),
        "",
        t("health.Protect_tube_line",
          tube=payload.tube_type, lamp_id=payload.lamp_id,
          mode=_fmt_mode(payload)),
        "",
        t("health.Protect_causes_header"),
        t("health.Protect_cause_emission"),
        t("health.Protect_cause_datasheet"),
        t("health.Protect_cause_op_too_high"),
        t("health.Protect_cause_short"),
    ]
    text = "\n".join(lines)

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle(t(title_key))
    box.setText(t("health.Protect_headline", kind=kind_label))
    box.setInformativeText(text)
    box.setTextInteractionFlags(Qt.TextSelectableByMouse)
    box.setStandardButtons(QMessageBox.Ok)
    box.exec()
