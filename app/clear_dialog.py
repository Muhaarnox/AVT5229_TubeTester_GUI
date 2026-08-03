"""Dialog for selective series removal from plots."""
from __future__ import annotations

from typing import Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from i18n_setup import t


class ClearSeriesDialog(QDialog):
    """Let the user pick which series to remove from the main plot.

    Parameters
    ----------
    series_info : list of dicts
        Each dict has ``series_id``, ``label``, ``n_points``.
    parent : QWidget, optional
    """

    def __init__(self, series_info: List[Dict], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t('clear_dlg.Title'))
        self.setMinimumWidth(340)

        self._checks: List[tuple[QCheckBox, int]] = []  # (checkbox, series_id)
        self._remove_all = False

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(t('clear_dlg.Prompt')))

        # One checkbox per series — unchecked by default
        for info in series_info:
            sid = info["series_id"]
            label = info["label"]
            n_pts = info["n_points"]
            cb = QCheckBox(f"{label}  ({n_pts} {t('clear_dlg.Points')})")
            cb.setChecked(False)
            layout.addWidget(cb)
            self._checks.append((cb, sid))

        if not series_info:
            layout.addWidget(QLabel(t('clear_dlg.No_data')))

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self._remove_sel_btn = QPushButton(t('clear_dlg.Remove_selected'))
        self._remove_sel_btn.clicked.connect(self._on_remove_selected)
        self._remove_sel_btn.setEnabled(False)
        btn_row.addWidget(self._remove_sel_btn)

        remove_all_btn = QPushButton(t('clear_dlg.Remove_all'))
        remove_all_btn.clicked.connect(self._on_remove_all)
        btn_row.addWidget(remove_all_btn)

        cancel_btn = QPushButton(t('clear_dlg.Cancel'))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)

        # Enable "Remove selected" only when something is checked
        for cb, _ in self._checks:
            cb.toggled.connect(self._update_remove_btn)

    # ── Results ─────────────────────────────────────────────────────

    @property
    def is_remove_all(self) -> bool:
        """True if user clicked 'Remove All'."""
        return self._remove_all

    @property
    def selected_series_ids(self) -> List[int]:
        """Series IDs the user checked for removal."""
        return [sid for cb, sid in self._checks if cb.isChecked()]

    # ── Slots ───────────────────────────────────────────────────────

    def _update_remove_btn(self) -> None:
        any_checked = any(cb.isChecked() for cb, _ in self._checks)
        self._remove_sel_btn.setEnabled(any_checked)

    def _on_remove_selected(self) -> None:
        self._remove_all = False
        self.accept()

    def _on_remove_all(self) -> None:
        self._remove_all = True
        self.accept()
