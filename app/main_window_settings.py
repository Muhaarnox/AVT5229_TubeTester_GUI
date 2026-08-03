"""Settings & lamp-application mixin for MainWindow.

Owns zone/metadata builders, lamp-config application and
scan-settings persistence.

Methods:
  - ``_zone_dict`` — collect zone spinbox values into a dict for
    ``SrkController``
  - ``_build_scan_metadata`` — full measurement metadata for
    ``save_measurement``
  - ``_apply_lamp`` — push ``LampConfig`` (Ua/Ug1/Ug2 ranges, Pa,
    load line) into UI controls
  - ``_apply_plot_ranges`` — push ``lamp.ranges`` to
    ``PlotRenderer.apply_ranges``
  - ``_save_scan_settings`` — JSON dump of current UI state to a
    user-chosen file
  - ``_load_scan_settings`` — restore JSON state into UI controls

Host class (MainWindow) provides via __init__:
  - self.lamps, self.app_config, self.lamp_panel (LampPanel),
    self.plot_renderer (PlotRenderer), self.plot_mgr (PlotManager),
    self.live_panel, self.amp_control_panel, self.compare_tab
  - state: _is_triode, _topology, _pig2_max_val
  - widget refs from MainWindowBuilders mixin (zone_*, ua/ug1/ug2 spinboxes,
    measurement_name_edit, lamp_combo, srk_label, preheat_*, pa_max_input,
    pg2_max_input, ua_max_input, ia_max_input, ia_max_limit_input,
    load_line_*, ug2_track_radio, ug2_offset, ug2_mode_*, ia_samples_spin,
    ug2_calc_combo, plot_line_width, anode_group)
  - helpers: _set_ug2_visibility, _update_preheat_live_label
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Dict

from PySide6.QtWidgets import QFileDialog, QMessageBox

from app.srk_widget import SrkController
from i18n_setup import t
from lm19.config import LampConfig, find_lamp
from lm19.constants import TOPOLOGY_PENTODE
from lm19.io_utils import write_json
from lm19.schema import (
    SETTINGS_SCHEMA_VERSION,
    _check_schema_version,
    stamp_schema_version,
)

log = logging.getLogger(__name__)

# ── module local constants ──
# Fallback Pa_max (W) applied when the selected lamp has no configured
# value, so a previously selected (possibly higher-power) lamp's limit is
# never silently reused for the protection check. Matches the pa_max_input
# spinbox construction default.
DEFAULT_PA_MAX_W = 12.5

# Top-level keys recognised by ``_load_scan_settings``. Any other key
# at the top level is reported as ignored — catches typos in
# hand-edited files. ``_schema_version`` is always allowed.
_SETTINGS_KNOWN_KEYS: frozenset = frozenset({
    "_schema_version", "lamp_type", "scan", "preheat", "plot",
})
_SETTINGS_LABEL = "settings"


class MainWindowSettings:
    """Mixin: lamp-config application + scan-settings persistence."""

    def _zone_dict(self) -> Dict:
        """Collect zone spinbox values into a dict for SrkController."""
        return {
            "ua_min": self.zone_ua_min.value(),
            "ua_max": self.zone_ua_max.value(),
            "ug1_min": self.zone_ug1_min.value(),
            "ug1_max": self.zone_ug1_max.value(),
            "ug2": self.zone_ug2.value(),
            "is_triode": self._is_triode,
            "ug2_track_ua": self.ug2_track_radio.isChecked(),
            "ug2_offset": self.ug2_offset.value(),
        }

    def _scan_meta_live_fields(self) -> Dict:
        """Fields that label the MEASUREMENT rather than describe the run.

        The user routinely types the name / lamp id and tunes the SRK
        zone while the scan is running, so these are read again at save
        time (see :meth:`_scan_meta_for_save`) instead of being frozen at
        start. The zone additionally has to match the one the scan flow
        passes to the SRK computation, which is read live there.
        """
        name = self.measurement_name_edit.text().strip()
        out: Dict = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "lamp_id": self.lamp_panel.lamp_id(),
            "name": name or "scan",
            "_zone": self._zone_dict(),
        }
        mfg_date = self.lamp_panel.mfg_date()
        if mfg_date:
            out["mfg_date"] = mfg_date
        return out

    def _build_scan_metadata(self) -> Dict:
        """Build measurement metadata dict from current UI state."""
        live = self._scan_meta_live_fields()
        meta: Dict = {
            "timestamp": live["timestamp"],
            "tube_type": self.lamp_combo.currentText(),
            "lamp_id": live["lamp_id"],
            "name": live["name"],
            "topology": getattr(self, '_topology', TOPOLOGY_PENTODE),
            "scan": {
                "ua": {"start": self.ua_start.value(), "stop": self.ua_stop.value(), "step": self.ua_step.value()},
                "ug1": {"start": self.ug1_start.value(), "stop": self.ug1_stop.value(), "step": self.ug1_step.value()},
                "ug2": {"start": self.ug2_start.value(), "stop": self.ug2_stop.value(), "step": self.ug2_step.value()},
                "uh": self.uh_input.value(),
                "ih": self.ih_input.value(),
                "an": self.anode_group.checkedId(),
                "ug2_track_ua": self.ug2_track_radio.isChecked(),
                "ug2_mode": self._current_ug2_mode(),
            },
            "_zone": live["_zone"],
        }
        if "mfg_date" in live:
            meta["mfg_date"] = live["mfg_date"]
        return meta

    def _freeze_scan_metadata(self) -> None:
        """Snapshot the settings the starting run is made with.

        The scan controls stay live for the whole run (a scan takes tens
        of minutes and the next one is routinely armed meanwhile), so
        reading them at save time records the run being ARMED, not the
        one that produced the points. ``ug2_mode`` is the worst of them:
        on load an explicit flag outranks the Ug2 ≈ Ua + offset
        auto-detection, so a mis-stamped triode-connected file is
        regrouped per point and falls apart into symbols.

        Call this only once the run's derived values are settled — the
        auto Ug1 step is computed and written back into its spinbox, and
        freezing before that records step 0 instead of the step actually
        swept.
        """
        self._pending_scan_meta = self._build_scan_metadata()

    def _scan_meta_for_save(self) -> Dict:
        """Metadata of the run that produced the points (see the freeze).

        Everything describing the RUN comes from the snapshot; the
        labelling fields come from the UI as it stands now
        (:meth:`_scan_meta_live_fields`) — freezing those would discard a
        name or lamp id typed while the scan was running.
        """
        frozen = getattr(self, "_pending_scan_meta", None)
        if frozen is None:
            log.warning(
                "No scan-start metadata snapshot — saving the CURRENT UI "
                "state instead; recorded scan settings may not match the "
                "run that produced these points")
            return self._build_scan_metadata()
        meta = copy.deepcopy(frozen)
        live = self._scan_meta_live_fields()
        meta.update(live)
        if "mfg_date" not in live:
            # Cleared while the scan ran: absence means "no data", and an
            # empty value must not be written (measurement format rule).
            meta.pop("mfg_date", None)
        return meta

    def _apply_lamp(self, tube_type: str) -> None:
        lamp = find_lamp(self.lamps, tube_type)
        if not lamp:
            return
        self._is_triode = lamp.is_triode
        self._topology = lamp.topology
        self._set_ug2_visibility(not lamp.is_triode)
        self.lamp_panel.apply_lamp(lamp)
        self.ua_start.setValue(lamp.ranges["ua"].min)
        self.ua_stop.setValue(lamp.ranges["ua"].max)
        # Per-lamp screen cap BEFORE the values (ML-131): Qt clamps the
        # current value on setMaximum, so the order matters when switching
        # from a lower-cap lamp to a higher-cap one.
        self.ug2_start.setMaximum(lamp.limits["ug2_max"])
        self.ug2_stop.setMaximum(lamp.limits["ug2_max"])
        self.ug2_start.setValue(lamp.ranges["ug2"].min)
        self.ug2_stop.setValue(lamp.ranges["ug2"].max)

        ug1_min = lamp.ranges["ug1"].min
        ug1_max = lamp.ranges["ug1"].max
        if ug1_min >= 0 and ug1_max >= 0:
            self.ug1_start.setValue(-ug1_max)
            self.ug1_stop.setValue(-ug1_min)
        else:
            self.ug1_start.setValue(ug1_min)
            self.ug1_stop.setValue(ug1_max)

        # Switching the lamp selection must not push heater commands to the
        # device; block signals so _on_uh_changed/_on_ih_changed don't fire.
        # Per-lamp heater caps (ML-130) applied BEFORE the values and inside
        # the signal block: setMaximum may clamp the current value, and that
        # clamp must neither command the heater nor clip the new nominal.
        self.uh_input.blockSignals(True)
        self.ih_input.blockSignals(True)
        self.uh_input.setMaximum(lamp.limits["uh_max"])
        self.ih_input.setMaximum(lamp.limits["ih_max"])
        self.uh_input.setValue(lamp.uh)
        self.ih_input.setValue(lamp.ih)
        self.uh_input.blockSignals(False)
        self.ih_input.blockSignals(False)

        self.zone_ua_min.setValue(lamp.ua - 10)
        self.zone_ua_max.setValue(lamp.ua + 10)
        self.zone_ug1_min.setValue(lamp.ug1)
        self.zone_ug1_max.setValue(lamp.ug1 + 0.4)
        self.zone_ug2.setValue(lamp.ug2)

        self.srk_label.setText(SrkController.format_srk(None, None, None, lamp=lamp))
        self.live_panel.set_an(self.lamp_panel.anode())
        # Nominal heater for the off-nominal live badge (both live panels).
        self.live_panel.set_nominal_heater(lamp.uh, lamp.ih)
        if hasattr(self, "manual_tab"):
            self.manual_tab.live_panel.set_nominal_heater(lamp.uh, lamp.ih)
            # The new lamp changes what counts as an off-nominal heater
            # setpoint, so the inline markers must be re-judged.
            self.manual_tab.refresh_heater_setpoint_warnings()

        heater_mode = "Uh" if lamp.uh > 0 else "Ih"
        target = lamp.uh if lamp.uh > 0 else lamp.ih
        unit = "V" if lamp.uh > 0 else "A"
        self.preheat_target.setText(f"{heater_mode} {target:.2f} {unit}")
        self.preheat_seconds.setValue(max(0, int(lamp.warmup_s)))
        self.preheat_done = False
        self.preheat_status.setText(t('heat.Not_started'))
        self._update_preheat_live_label(None, None)
        self.preheat_progress.setValue(0)

        # Auto-fill Pa_max / Pg2_max / Ua_max / Ia_max / Load line from lamp config
        if lamp.pa_max is not None:
            self.pa_max_input.setValue(lamp.pa_max)
            self.amp_control_panel.pa_max_spin.setValue(lamp.pa_max)
        else:
            # No lamp-specific Pa_max: reset to a defined default instead of
            # leaving the previous lamp's value, and make the gap visible.
            log.warning(
                "Pa_max not set for lamp '%s' — using default %.1f W; set it "
                "manually before scanning a power tube",
                lamp.tube_type, DEFAULT_PA_MAX_W,
            )
            self.pa_max_input.setValue(DEFAULT_PA_MAX_W)
            self.amp_control_panel.pa_max_spin.setValue(DEFAULT_PA_MAX_W)
        if lamp.pig2_max is not None:
            self.pg2_max_input.setValue(lamp.pig2_max)
        if lamp.ua_max_limit is not None:
            self.ua_max_input.setValue(lamp.ua_max_limit)
        if lamp.ia_max_limit is not None:
            self.ia_max_limit_input.setValue(lamp.ia_max_limit)
        # Lamp defaults go to the amp panel — the single source of
        # working-line parameters.
        if lamp.ra is not None and lamp.ra > 0:
            self.amp_control_panel.ra_spin.setValue(lamp.ra)
        if lamp.ua > 0:
            self.amp_control_panel.ub_spin.setValue(lamp.ua)
        if lamp.ug1 != 0:
            self.amp_control_panel.ug1_spin.setValue(lamp.ug1)

        # Store pig2_max, nominal_s, and triode state for PlotManager
        self._pig2_max_val = lamp.pig2_max
        if hasattr(self, 'plot_mgr'):
            self.plot_mgr.w["pig2_max_val"] = lamp.pig2_max
            self.plot_mgr.w["nominal_s"] = lamp.s if lamp.s > 0 else None
            self.plot_mgr.set_triode(lamp.is_triode)

        # Hide Pg2 controls for triodes (no screen grid)
        pg2_visible = not lamp.is_triode
        self.pg2_max_cb.setVisible(pg2_visible)
        self.pg2_max_input.setVisible(pg2_visible)

        self._apply_plot_ranges(lamp)

    def _apply_plot_ranges(self, lamp: LampConfig) -> None:
        ua_min = 0.0
        ua_max = lamp.ranges["ua"].max
        ia_max = self.ia_max_input.value() if hasattr(self, "ia_max_input") else self.app_config.plot_ia_max
        ug1_min = lamp.ranges["ug1"].min
        ug1_max = lamp.ranges["ug1"].max
        if ug1_min >= 0 and ug1_max >= 0:
            ug1_min, ug1_max = -ug1_max, -ug1_min

        self.plot_renderer.apply_ranges(ua_min, ua_max, ia_max, ug1_min, ug1_max)

    def _save_scan_settings(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            t('msg.Save_scan_settings'),
            "",
            t('msg.JSON_filter'),
        )
        if not path:
            return
        data: Dict = {
            "lamp_type": self.lamp_combo.currentText(),
            "scan": {
                "ua": {
                    "start": self.ua_start.value(),
                    "stop": self.ua_stop.value(),
                    "step": self.ua_step.value(),
                },
                "ug1": {
                    "start": self.ug1_start.value(),
                    "stop": self.ug1_stop.value(),
                    "step": self.ug1_step.value(),
                },
                "ug2": {
                    "start": self.ug2_start.value(),
                    "stop": self.ug2_stop.value(),
                    "step": self.ug2_step.value(),
                },
                "uh": self.uh_input.value(),
                "ih": self.ih_input.value(),
                "pa_over_pct": self.pa_over_pct.value(),
                "pig2_over_pct": self.pig2_over_pct.value(),
                "ia_samples": self.ia_samples_spin.value(),
                "ug2_scan_mode": "track" if self.ug2_track_radio.isChecked() else "sweep",
                "ug2_offset": self.ug2_offset.value(),
            },
            "preheat": {
                "enabled": self.preheat_enabled.isChecked(),
                "warmup_s": self.preheat_seconds.value(),
            },
            "plot": {
                "ia_max": self.ia_max_input.value(),
                "ug2_mode": "series" if self.ug2_mode_series.isChecked() else "color",
                "ug2_calc": self.ug2_calc_combo.currentText(),
                "line_width": self.plot_line_width.value(),
            },
        }
        stamp_schema_version(data, SETTINGS_SCHEMA_VERSION)
        try:
            write_json(Path(path), data)
        except OSError as exc:
            # ML-085: the load side already surfaces failures with a dialog
            # — a read-only/locked target on save was silent (traceback to
            # stderr only), user believed the settings were saved.
            log.exception("Failed to save scan settings to %s", path)
            QMessageBox.warning(self, t('msg.Save_scan_settings'),
                                t('msg.Save_failed', error=str(exc)))

    def _load_scan_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            t('msg.Load_scan_settings'),
            "",
            t('msg.JSON_filter'),
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            log.exception("Failed to load measurement file %s", path)
            QMessageBox.warning(self, t('msg.Load'), t('msg.Failed_to_load', error=exc))
            return

        file_version = _check_schema_version(
            data, SETTINGS_SCHEMA_VERSION, _SETTINGS_LABEL, str(path),
        )
        unknown_keys = sorted(set(data.keys()) - _SETTINGS_KNOWN_KEYS)
        if unknown_keys:
            log.warning(
                "settings (%s): %d unknown top-level key(s) ignored: %s. "
                "Possible typo or stale field from older format.",
                path, len(unknown_keys), unknown_keys,
            )

        lamp_type = data.get("lamp_type")
        if lamp_type:
            index = self.lamp_combo.findText(lamp_type)
            if index >= 0:
                self.lamp_combo.setCurrentIndex(index)

        scan = data.get("scan", {})
        ua = scan.get("ua", {})
        ug1 = scan.get("ug1", {})
        ug2 = scan.get("ug2", {})
        self.ua_start.setValue(ua.get("start", self.ua_start.value()))
        self.ua_stop.setValue(ua.get("stop", self.ua_stop.value()))
        self.ua_step.setValue(ua.get("step", self.ua_step.value()))
        self.ug1_start.setValue(ug1.get("start", self.ug1_start.value()))
        self.ug1_stop.setValue(ug1.get("stop", self.ug1_stop.value()))
        self.ug1_step.setValue(ug1.get("step", self.ug1_step.value()))
        self.ug2_start.setValue(ug2.get("start", self.ug2_start.value()))
        self.ug2_stop.setValue(ug2.get("stop", self.ug2_stop.value()))
        self.ug2_step.setValue(ug2.get("step", self.ug2_step.value()))
        # Loading a settings file must not fire heater commands to the device
        # (same guard as _apply_lamp): block signals around the setValue calls.
        self.uh_input.blockSignals(True)
        self.ih_input.blockSignals(True)
        self.uh_input.setValue(scan.get("uh", self.uh_input.value()))
        self.ih_input.setValue(scan.get("ih", self.ih_input.value()))
        self.uh_input.blockSignals(False)
        self.ih_input.blockSignals(False)
        if "pa_over_pct" in scan:
            self.pa_over_pct.setValue(float(scan["pa_over_pct"]))
        if "pig2_over_pct" in scan:
            self.pig2_over_pct.setValue(float(scan["pig2_over_pct"]))
        if "ia_samples" in scan:
            self.ia_samples_spin.setValue(int(scan["ia_samples"]))
        if "ug2_scan_mode" in scan:
            if scan["ug2_scan_mode"] == "track":
                self.ug2_track_radio.setChecked(True)
            else:
                self.ug2_sweep_radio.setChecked(True)
        if "ug2_offset" in scan:
            self.ug2_offset.setValue(float(scan["ug2_offset"]))

        preheat = data.get("preheat", {})
        if "enabled" in preheat:
            self.preheat_enabled.setChecked(bool(preheat.get("enabled")))
        if "warmup_s" in preheat:
            self.preheat_seconds.setValue(int(preheat.get("warmup_s", self.preheat_seconds.value())))

        plot = data.get("plot", {})
        if "ia_max" in plot:
            self.ia_max_input.setValue(float(plot.get("ia_max", self.ia_max_input.value())))
        if plot.get("ug2_mode") == "color":
            self.ug2_mode_color.setChecked(True)
        elif plot.get("ug2_mode") == "series":
            self.ug2_mode_series.setChecked(True)
        if "ug2_calc" in plot:
            calc_text = str(plot.get("ug2_calc"))
            idx = self.ug2_calc_combo.findText(calc_text)
            if idx >= 0:
                self.ug2_calc_combo.setCurrentIndex(idx)
        elif "ug2_slice" in plot:
            calc_text = str(plot.get("ug2_slice"))
            idx = self.ug2_calc_combo.findText(calc_text)
            if idx >= 0:
                self.ug2_calc_combo.setCurrentIndex(idx)
        if "line_width" in plot:
            self.plot_line_width.setValue(float(plot.get("line_width", self.plot_line_width.value())))

        # Surface the load result so the user sees what actually loaded
        # (and what was ignored) — a silently partial restore reads as a
        # full one.
        applied_top = sum(
            1 for k in ("lamp_type", "scan", "preheat", "plot") if k in data
        )
        msg = t(
            'msg.Settings_loaded',
            version=file_version,
            applied=applied_top,
            ignored=len(unknown_keys),
        )
        QMessageBox.information(self, t('menu.Load_scan_settings'), msg)

