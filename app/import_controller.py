"""Import controller — handles uTracer, CSV, CurveTraceData, and eTracer import workflows."""

import datetime as dt
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PySide6.QtWidgets import QFileDialog, QMessageBox, QTabWidget, QWidget

from app.import_dialog import ImportMetaDialog, CsvImportDialog
from i18n_setup import t
from lm19.config import LampConfig, find_lamp
from lm19.measurements import save_imported_measurement
from lm19.utracer_import import parse_utd, utd_to_lm19_points, guess_meta_from_filename
from lm19.csv_import import parse_csv
from lm19.curvetracedata_import import (
    parse_curvetracedata_dat,
    dat_to_lm19_points,
    guess_meta_from_dat_filename,
)
from lm19.etracer_import import (
    parse_etracer_csv,
    etracer_to_lm19_points,
    guess_meta_from_etracer,
    extract_heater_from_etd,
)
from lm19.import_helpers import (
    import_topology_payload,
    build_utd_description,
    build_csv_description,
    build_ctd_description,
    build_etracer_description,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)

log = logging.getLogger(__name__)


class ImportController:
    """Handles all measurement import workflows.

    Receives references to shared UI objects it needs; does NOT
    inherit from QObject — it's a plain helper.
    """

    def __init__(
        self,
        parent_widget: QWidget,
        compare_tab,
        tabs: QTabWidget,
        get_lamps: Callable[[], List[LampConfig]],
    ) -> None:
        self._parent = parent_widget
        self._compare_tab = compare_tab
        self._tabs = tabs
        self._get_lamps = get_lamps

    def _warn_skipped(self, count: int) -> None:
        """Surface parser data-loss counters to the user (failure-visibility rule:
        the parser's log WARNING alone is not a user-facing channel)."""
        if count > 0:
            QMessageBox.warning(
                self._parent, t('menu.Import'),
                t('msg.Import_rows_skipped', count=count))

    # ------------------------------------------------------------------
    # Public import actions (connect to menu triggers)
    # ------------------------------------------------------------------

    def import_utracer(self) -> None:
        """Import uTracer .utd measurement file."""
        path, _ = QFileDialog.getOpenFileName(
            self._parent, t('menu.Import_uTracer'), "", t('msg.UTD_filter'),
        )
        if not path:
            return
        try:
            stats: Dict = {}
            parsed = parse_utd(path, stats=stats)
        except Exception as exc:
            log.exception("Failed to import uTracer file %s", path)
            QMessageBox.warning(self._parent, t('menu.Import'), t('msg.Failed_to_load', error=exc))
            return
        self._warn_skipped(int(stats.get("short_rows", 0)))

        src_stem = Path(path).stem
        meta = guess_meta_from_filename(path)
        meta_defaults = self._import_defaults_from_stem(
            src_stem,
            guessed_type=str(meta.get("tube_type", src_stem)),
            guessed_vs=float(meta.get("vs", 0.0)),
        )
        meta_defaults["description"] = build_utd_description(
            path, parsed, float(meta.get("vs", 0.0)),
        )
        points = utd_to_lm19_points(parsed, vs=meta.get("vs", 0.0))
        if not points:
            QMessageBox.warning(self._parent, t('menu.Import'), t('msg.Import_no_points'))
            return

        dlg = ImportMetaDialog(self._parent, defaults=meta_defaults, point_count=len(points))
        if dlg.exec() != ImportMetaDialog.DialogCode.Accepted:
            return

        points = utd_to_lm19_points(parsed, vs=dlg.ug2, vh=dlg.uh)
        self._finalize_import(dlg, points, "uTracer", "utd", path, src_stem)

    def import_csv(self) -> None:
        """Import CSV/TSV measurement file."""
        path, _ = QFileDialog.getOpenFileName(
            self._parent, t('menu.Import_CSV'), "", t('msg.CSV_import_filter'),
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            log.exception("Failed to import CSV file %s", path)
            QMessageBox.warning(self._parent, t('menu.Import'), t('msg.Failed_to_load', error=exc))
            return

        csv_dlg = CsvImportDialog(self._parent, text=text)
        if csv_dlg.exec() != CsvImportDialog.DialogCode.Accepted:
            return

        stats: Dict = {}
        points = parse_csv(text, csv_dlg.column_mapping, csv_dlg.separator,
                           stats=stats)
        if not points:
            QMessageBox.warning(self._parent, t('menu.Import'), t('msg.Import_no_points'))
            return
        self._warn_skipped(int(stats.get("skipped_rows", 0)))

        stem = Path(path).stem
        meta_defaults = self._import_defaults_from_stem(stem)
        meta_defaults["description"] = build_csv_description(path, text)

        meta_dlg = ImportMetaDialog(self._parent, defaults=meta_defaults, point_count=len(points))
        if meta_dlg.exec() != ImportMetaDialog.DialogCode.Accepted:
            return

        ug2_val, uh_val = meta_dlg.ug2, meta_dlg.uh
        for p in points:
            if ug2_val > 0 and p["ug2"] == 0.0:
                p["ug2"] = ug2_val
            if uh_val > 0 and p["uh"] == 0.0:
                p["uh"] = uh_val

        self._finalize_import(meta_dlg, points, "CSV", "csv", path, stem)

    def import_curvetracedata(self) -> None:
        """Import CurveTraceData .dat measurement file."""
        path, _ = QFileDialog.getOpenFileName(
            self._parent, t('menu.Import_CurveTraceData'), "", t('msg.CTD_filter'),
        )
        if not path:
            return
        try:
            parsed = parse_curvetracedata_dat(path)
        except Exception as exc:
            log.exception("Failed to import CurveTraceData file %s", path)
            QMessageBox.warning(self._parent, t('menu.Import'), t('msg.Failed_to_load', error=exc))
            return

        src_stem = Path(path).stem
        sample_name = str(parsed.get("sample_name", ""))
        guessed = guess_meta_from_dat_filename(path, sample_name=sample_name)
        meta_defaults = self._import_defaults_from_stem(
            src_stem,
            guessed_type=str(guessed.get("tube_type", src_stem)),
            guessed_vs=0.0,
        )
        meta_defaults["lamp_id"] = str(guessed.get("lamp_id", src_stem))
        meta_defaults["name"] = str(guessed.get("name", src_stem))
        meta_defaults["description"] = build_ctd_description(path, parsed)

        points = dat_to_lm19_points(parsed)
        if not points:
            QMessageBox.warning(self._parent, t('menu.Import'), t('msg.Import_no_points'))
            return

        meta_dlg = ImportMetaDialog(self._parent, defaults=meta_defaults, point_count=len(points))
        if meta_dlg.exec() != ImportMetaDialog.DialogCode.Accepted:
            return

        for p in points:
            p["ug2"] = meta_dlg.ug2
            p["uh"] = meta_dlg.uh

        self._finalize_import(meta_dlg, points, "CurveTraceData", "curvetracedata", path, src_stem)

    def import_etracer(self) -> None:
        """Import eTracer CSV measurement file."""
        path, _ = QFileDialog.getOpenFileName(
            self._parent, t('menu.Import_eTracer'), "", t('msg.eTracer_filter'),
        )
        if not path:
            return
        try:
            stats: Dict = {}
            parsed = parse_etracer_csv(path, stats=stats)
        except Exception as exc:
            log.exception("Failed to import eTracer file %s", path)
            QMessageBox.warning(self._parent, t('menu.Import'), t('msg.Failed_to_load', error=exc))
            return
        self._warn_skipped(int(stats.get("nan_points", 0)))

        src_stem = Path(path).stem
        guessed = guess_meta_from_etracer(path, parsed)
        topology = str(guessed.get("topology", TOPOLOGY_TRIODE))
        ug2_mode = topology  # triode / triode_connected / pentode

        meta_defaults = self._import_defaults_from_stem(
            src_stem,
            guessed_type=str(guessed.get("tube_type", src_stem)),
            guessed_vs=0.0,
        )
        meta_defaults["lamp_id"] = str(guessed.get("lamp_id", src_stem))
        meta_defaults["name"] = str(guessed.get("name", src_stem))
        meta_defaults["ug2_mode"] = ug2_mode
        meta_defaults["description"] = build_etracer_description(path, parsed)

        # Try companion .etd file for heater voltage (fallback if lamp not in config)
        if meta_defaults.get("vh", 0.0) == 0.0:
            etd_vh = extract_heater_from_etd(path, parsed.get("etd_file", ""))
            if etd_vh is not None and etd_vh > 0:
                meta_defaults["vh"] = etd_vh

        points = etracer_to_lm19_points(parsed)
        if not points:
            QMessageBox.warning(self._parent, t('menu.Import'), t('msg.Import_no_points'))
            return

        meta_dlg = ImportMetaDialog(self._parent, defaults=meta_defaults, point_count=len(points))
        if meta_dlg.exec() != ImportMetaDialog.DialogCode.Accepted:
            return

        # Re-convert with user-supplied heater voltage
        points = etracer_to_lm19_points(parsed, vh=meta_dlg.uh)

        # For triode mode, override Ug2 from dialog if user changed it
        if ug2_mode == TOPOLOGY_TRIODE:
            for p in points:
                p["ug2"] = meta_dlg.ug2

        self._finalize_import(meta_dlg, points, "eTracer", "etracer", path, src_stem)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _import_defaults_from_stem(
        self,
        stem: str,
        *,
        guessed_type: Optional[str] = None,
        guessed_vs: float = 0.0,
    ) -> Dict:
        tube_type = (guessed_type or stem or t("import.Default_tube_type")).strip()
        defaults: Dict = {
            "tube_type": tube_type,
            "lamp_id": stem or t("import.Default_lamp_id"),
            "name": stem or t("import.Default_name"),
            "vs": float(guessed_vs or 0.0),
            "vh": 0.0,
            "ug2_mode": TOPOLOGY_PENTODE,
        }
        lamp = find_lamp(self._get_lamps(), tube_type)
        if lamp is not None:
            defaults["vh"] = float(lamp.uh)
            defaults["ug2_mode"] = (TOPOLOGY_TRIODE
                                    if lamp.topology == TOPOLOGY_TRIODE
                                    else TOPOLOGY_PENTODE)
        return defaults

    def _finalize_import(
        self,
        dlg: ImportMetaDialog,
        points: List[Dict],
        source: str,
        source_key: str,
        source_file: str,
        source_stem: str,
    ) -> None:
        """Common tail for all import workflows: save and display."""
        imported_measurement = {
            "tube_type": dlg.tube_type,
            "lamp_id": dlg.lamp_id,
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "name": dlg.name,
            "description": dlg.description,
            "points": points,
            "source": source,
            "source_file": source_file,
        }
        imported_measurement.update(import_topology_payload(dlg.ug2_mode))
        try:
            save_imported_measurement(
                dlg.tube_type, dlg.lamp_id, imported_measurement,
                source=source_key, source_stem=source_stem,
            )
        except Exception as exc:
            log.exception("Failed to auto-save imported %s measurement", source)
            QMessageBox.warning(
                self._parent, t('menu.Import'),
                t('msg.Import_autosave_failed', error=exc),
            )
        entry = {
            "lamp_type": dlg.tube_type,
            "lamp_id": dlg.lamp_id,
            "timestamp": imported_measurement["timestamp"],
            "name": dlg.name,
            "points": points,
            "data": imported_measurement,
        }
        self._compare_tab.add_imported_entries([entry])
        self._tabs.setCurrentWidget(self._compare_tab)
