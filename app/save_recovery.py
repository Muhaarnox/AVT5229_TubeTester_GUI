"""Save-with-recovery dialog for measurement persistence (scan + health).

A failed save used to be a dead end: ``QMessageBox.critical`` and the
result lived only in memory until the next test or the app close. The
recovery loop gives the operator real options:

- **Retry** — run the same save again (disk space freed / dir unlocked);
- **Save As…** — write the measurement JSON to any reachable location
  (the payload already carries its ``_schema_version``: ``save_*`` stamp
  it in-place before the first write attempt);
- **Close** — keep the result in-memory for this session only.

Used by the scan save (``main_window_scan``) and the health save
(``health_tab``) — the two places a finished measurement can be lost.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from i18n_setup import t
from lm19.io_utils import write_json

log = logging.getLogger(__name__)

# _ask_recovery outcomes
RETRY = "retry"
SAVE_AS = "save_as"
CLOSE = "close"


def _ask_recovery(parent: Optional[QWidget], error: str) -> str:
    """Show the recovery dialog. Separated from the loop for testability."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(t("msg.Save"))
    box.setText(t("msg.Save_failed", error=error))
    retry_btn = box.addButton(t("msg.Save_retry"),
                              QMessageBox.ButtonRole.AcceptRole)
    save_as_btn = box.addButton(t("msg.Save_as"),
                                QMessageBox.ButtonRole.ActionRole)
    box.addButton(t("msg.Save_keep_memory"),
                  QMessageBox.ButtonRole.RejectRole)
    box.exec()
    clicked = box.clickedButton()
    if clicked is retry_btn:
        return RETRY
    if clicked is save_as_btn:
        return SAVE_AS
    return CLOSE


def save_with_recovery(
    parent: Optional[QWidget],
    save_fn: Callable[[], Path],
    payload: Dict,
    suggested_name: str,
) -> Optional[Path]:
    """Run ``save_fn`` until it succeeds or the operator gives up.

    Returns the written path, or ``None`` when the result stays in-memory
    only (the caller keeps displaying it for the session either way).
    Only ``OSError`` enters the recovery loop — serialization bugs raise
    ``TypeError`` and propagate (failure-visibility principle 1).
    """
    while True:
        try:
            return save_fn()
        except OSError as exc:
            log.exception("Save failed")
            error = str(exc)
        # Recovery loop for this failure.
        while True:
            choice = _ask_recovery(parent, error)
            if choice == CLOSE:
                log.warning("Measurement NOT saved — kept in-memory only")
                return None
            if choice == RETRY:
                break  # outer loop calls save_fn again
            # SAVE_AS
            path_str, _ = QFileDialog.getSaveFileName(
                parent, t("msg.Save"), suggested_name, t("msg.JSON_filter"))
            if not path_str:
                continue  # back to the recovery dialog — still unsaved
            try:
                p = Path(path_str)
                write_json(p, payload)
                log.info("Measurement saved via Save As to %s", p)
                return p
            except OSError as exc2:
                log.exception("Save As failed")
                error = str(exc2)


def suggested_filename(tube_type: str, lamp_id: str, timestamp: str) -> str:
    """Windows-safe default name for the Save As dialog."""
    raw = f"{tube_type}_{lamp_id}_{timestamp}.json"
    return "".join("-" if c in ':<>"|?*' else c for c in raw)
