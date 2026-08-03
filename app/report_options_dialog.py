"""PDF report options dialog: section checkboxes, language, presets.

Shared by every PDF export entry point (main window, Compare tab). The
section list comes from :data:`app.report.REPORT_SECTIONS` (source of
truth); sections whose data is unavailable are greyed out with the
reason in the tooltip instead of being hidden (failure visibility).

Dialog choices are remembered for the session in a module-level store;
permanent defaults live in ``config/app.json`` (``report_language``,
``report_sections``, ``report_ask``) — config files are read-only for
the UI, same pattern as the matching-algorithm dropdowns.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set, Tuple

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.report import (
    BRIEF_SECTION_IDS,
    REPORT_SECTIONS,
    ReportSection,
    sections_from_config,
)
from i18n_setup import available_locales, t

log = logging.getLogger(__name__)

# Session memory for dialog choices, per report kind ("scan", "amp", …) —
# config stays read-only. Each entry: {"sections": Set[str], "ask": bool}.
_SESSION: Dict[str, Dict[str, object]] = {}


def resolve_report_language(config) -> str:
    """Report language from ``app.json: report_language`` ONLY (no UI
    selector by design). "" → English (documented default); a
    code without a ``locales/<code>.json`` file → English with a loud
    WARNING (a config typo must not silently switch the document language).
    """
    code = (getattr(config, "report_language", "") or "").strip()
    if not code:
        return "en"
    if code not in available_locales():
        log.warning(
            "report_language '%s' has no locales/%s.json — "
            "falling back to English", code, code)
        return "en"
    return code


def reset_session() -> None:
    """Forget session choices (tests / locale switch)."""
    _SESSION.clear()


@dataclass(frozen=True)
class ReportOptions:
    """User's choice for one PDF export."""

    sections: Set[str]
    language: str


class ReportOptionsDialog(QDialog):
    """Checkbox-per-section options dialog for PDF export."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        available: Dict[str, str],
        sections: Set[str],
        specs: Tuple[ReportSection, ...] = REPORT_SECTIONS,
    ) -> None:
        """``available``: sid → "" when the section has data, else the
        i18n key of the reason it is unavailable (greyed out).
        ``specs`` selects the section list (scan report by default).
        The report language is NOT chosen here — it comes from
        ``app.json: report_language`` (see ``resolve_report_language``).
        """
        super().__init__(parent)
        self.setWindowTitle(t("report.Dlg_title"))
        self.setMinimumWidth(360)
        self._specs = specs

        layout = QVBoxLayout(self)

        group = QGroupBox(t("report.Dlg_sections"))
        group_lay = QVBoxLayout(group)
        self._boxes: Dict[str, QCheckBox] = {}
        for spec in specs:
            cb = QCheckBox(t(spec.label_key))
            reason = available.get(spec.sid, "")
            if reason:
                cb.setEnabled(False)
                cb.setChecked(False)
                cb.setToolTip(t(reason))
            else:
                cb.setChecked(spec.sid in sections)
                cb.setToolTip(t("report.Tip_section"))
            self._boxes[spec.sid] = cb
            group_lay.addWidget(cb)
        layout.addWidget(group)

        presets = QHBoxLayout()
        btn_full = QPushButton(t("report.Dlg_preset_full"))
        btn_full.setToolTip(t("report.Tip_preset_full"))
        btn_full.setAutoDefault(False)
        btn_full.clicked.connect(
            lambda: self._apply_preset({s.sid for s in self._specs}))
        presets.addWidget(btn_full)
        if specs is REPORT_SECTIONS:
            btn_brief = QPushButton(t("report.Dlg_preset_brief"))
            btn_brief.setToolTip(t("report.Tip_preset_brief"))
            btn_brief.setAutoDefault(False)
            btn_brief.clicked.connect(
                lambda: self._apply_preset(set(BRIEF_SECTION_IDS)))
            presets.addWidget(btn_brief)
        presets.addStretch(1)
        layout.addLayout(presets)

        self.dont_ask_cb = QCheckBox(t("report.Dlg_dont_ask"))
        self.dont_ask_cb.setToolTip(t("report.Tip_dont_ask"))
        layout.addWidget(self.dont_ask_cb)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _apply_preset(self, ids: Set[str]) -> None:
        for sid, cb in self._boxes.items():
            if cb.isEnabled():
                cb.setChecked(sid in ids)

    def selected_sections(self) -> Set[str]:
        """Checked AND available sections only — a disabled checkbox can
        never smuggle its section into the PDF."""
        return {sid for sid, cb in self._boxes.items()
                if cb.isEnabled() and cb.isChecked()}

    def dont_ask(self) -> bool:
        return self.dont_ask_cb.isChecked()


def ask_report_options(
    parent: Optional[QWidget],
    available: Dict[str, str],
    config,
    *,
    dialog_cls: Callable[..., ReportOptionsDialog] = ReportOptionsDialog,
    specs: Tuple[ReportSection, ...] = REPORT_SECTIONS,
    session_key: str = "scan",
) -> Optional[ReportOptions]:
    """Resolve report options for one export; ``None`` = user cancelled.

    Honors the per-``session_key`` "don't ask again" flag and
    ``config.report_ask``; in the silent path the remembered/config
    sections are still clipped to what is actually available. The
    language always comes from the config (``resolve_report_language``).
    """
    sess = _SESSION.setdefault(session_key, {})
    if specs is REPORT_SECTIONS:
        default_secs = sections_from_config(config.report_sections)
    else:
        default_secs = {s.sid for s in specs if s.default_on}
    remembered: Set[str] = sess.get("sections", default_secs)
    language = resolve_report_language(config)
    ask = bool(sess.get("ask", config.report_ask))

    if not ask:
        usable = {sid for sid in remembered if not available.get(sid, "")}
        return ReportOptions(sections=usable, language=language)

    dlg = dialog_cls(parent, available=available,
                     sections=remembered, specs=specs)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    opts = ReportOptions(sections=dlg.selected_sections(),
                         language=language)
    sess["sections"] = set(opts.sections)
    sess["ask"] = not dlg.dont_ask()
    return opts
