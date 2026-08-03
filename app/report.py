"""PDF report generation for tube test measurements.

Three layers, split on purpose:

- :data:`REPORT_SECTIONS` — the source-of-truth list of toggleable report
  sections (the options dialog and the config CSV both derive from it);
- :func:`build_report_lines` — a pure function producing the ordered
  ``(role, text)`` body lines (unit-testable — QtPdf cannot extract text
  back out of QPdfWriter output);
- :func:`_render_report` — dumb painter layout of lines + plot images.

The report language is independent from the UI language: callers pass a
locale code and the text goes through ``i18n_setup.translator_for``.
"""

import datetime as dt
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import QMarginsF, QRect, Qt
from PySide6.QtGui import (
    QFont,
    QFontMetrics,
    QPageLayout,
    QPageSize,
    QPainter,
    QPdfWriter,
    QPixmap,
)

from i18n_setup import translator_for
from lm19.version import APP_VERSION
from lm19.amplifier.constants import (
    HD_METHOD_5POINT,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_DFT,
)

if TYPE_CHECKING:
    from lm19.config import LampConfig
    from lm19.quality import QualityReport

log = logging.getLogger(__name__)

# HD method code → i18n label key for the distortion block (method
# visibility: with auto-routing the numbers are NOT necessarily 5-point,
# the report must say which method produced them). An unknown code is
# printed raw rather than hidden.
_HD_METHOD_LABEL_KEYS = {
    HD_METHOD_5POINT: "plot.Wl_method_5point",
    HD_METHOD_CHEBYSHEV: "plot.Wl_method_chebyshev",
    HD_METHOD_DFT: "plot.Wl_method_dft",
}

# ── module local constants ──
_REPORT_DPI = 150             # QPdfWriter render resolution
_PAGE_MARGIN_MM = 15.0        # A4 margins (QPageLayout default unit is mm)
_LINE_HEIGHT_IN = 0.2         # body line height, inches
_FONT_FAMILY = "Arial"
_TITLE_FONT_PT = 16
_BODY_FONT_PT = 10
_SMALL_FONT_PT = 8
_TITLE_GAP = 1.5              # line-height multiples after the title line
_SECTION_GAP = 0.3            # small vertical gap between text sections
_PLOT_GAP = 1.3               # gap between last text line and first image
_IMAGE_MIN_LINES = 3          # min free lines to start an image on this page
_FOOTER_RAISE = 0.5           # footer baseline offset from page bottom

# Line roles produced by build_report_lines: role → advance multiplier.
# "title" uses the title font; "gap" draws nothing (spacer).
_ROLE_ADVANCE = {"title": _TITLE_GAP, "body": 1.0, "gap": _SECTION_GAP}

ReportLine = Tuple[str, str]  # (role, text)


@dataclass(frozen=True)
class ReportSection:
    """One toggleable report section (dialog checkbox + config CSV token)."""

    sid: str        # stable id used in config CSV and the ``sections`` set
    label_key: str  # i18n key of the dialog checkbox label
    default_on: bool = True


# Source of truth for the options dialog, the config CSV and the render
# gating — new sections are added HERE (completeness pins enumerate this).
REPORT_SECTIONS: Tuple[ReportSection, ...] = (
    ReportSection("nominal", "report.Sec_nominal"),
    ReportSection("scan_settings", "report.Sec_scan_settings"),
    ReportSection("srk", "report.Sec_srk"),
    ReportSection("quality", "report.Sec_quality"),
    ReportSection("distortion", "report.Sec_distortion"),
    ReportSection("plot_curves", "report.Sec_plot_curves"),
    ReportSection("plot_transfer", "report.Sec_plot_transfer"),
)
SECTION_IDS = frozenset(s.sid for s in REPORT_SECTIONS)
# One-page summary preset for the options dialog.
BRIEF_SECTION_IDS = frozenset({"nominal", "srk", "quality", "plot_curves"})


def sections_from_config(csv: str) -> Set[str]:
    """Enabled section ids from the ``report_sections`` config CSV.

    Empty/blank string means "section defaults". Unknown tokens are
    dropped loudly (config typos must not silently shrink the report).
    """
    if not csv.strip():
        return {s.sid for s in REPORT_SECTIONS if s.default_on}
    ids = {tok.strip() for tok in csv.split(",") if tok.strip()}
    unknown = ids - SECTION_IDS
    if unknown:
        log.warning("report_sections: unknown section ids ignored: %s",
                    sorted(unknown))
    return ids & SECTION_IDS


def _fmt_range(rng: Dict) -> str:
    """``{"start": 0, "stop": 300, "step": 10}`` → ``"0…300/10"``."""
    return (f"{rng.get('start', 0):g}…{rng.get('stop', 0):g}"
            f"/{rng.get('step', 0):g}")


def build_report_lines(
    tube_type: str,
    lamp_id: str,
    timestamp: str,
    lamp_config: Optional["LampConfig"] = None,
    points: Optional[List[Dict]] = None,
    srk: Optional[Dict] = None,
    quality: Optional["QualityReport"] = None,
    analysis: Optional[Dict] = None,
    mfg_date: str = "",
    *,
    sections: Optional[Set[str]] = None,
    scan_meta: Optional[Dict] = None,
    tr: Optional[Callable[..., str]] = None,
) -> List[ReportLine]:
    """Build the ordered text-body lines of the report (pure content model).

    ``sections=None`` means "all sections" (data presence still gates each
    block); ``tr=None`` renders in English.
    """
    tr = tr or translator_for("en")
    secs = SECTION_IDS if sections is None else sections

    lines: List[ReportLine] = []
    lines.append(("title", tr("report.Title", tube=tube_type)))
    lines.append(("body", tr("report.Lamp_line", id=lamp_id, date=timestamp)))
    if mfg_date:
        lines.append(("body", tr("report.Manufactured", date=mfg_date)))

    if scan_meta and "scan_settings" in secs:
        name = str(scan_meta.get("name") or "")
        ts = str(scan_meta.get("timestamp") or "")
        if name or ts:
            lines.append(("body", tr("report.Meas_line",
                                     name=name or "—", ts=ts or "—")))
        sc = scan_meta.get("scan") or {}
        mode = str(sc.get("ug2_mode") or scan_meta.get("topology") or "?")
        ua, ug1, ug2 = sc.get("ua"), sc.get("ug1"), sc.get("ug2")
        if ua and ug1:
            scan_line = tr("report.Scan_line", mode=mode,
                           ua=_fmt_range(ua), ug1=_fmt_range(ug1))
            if mode == "pentode" and ug2:
                scan_line += tr("report.Scan_ug2_part", ug2=_fmt_range(ug2))
            lines.append(("body", scan_line))
        if sc.get("uh") is not None and sc.get("ih") is not None:
            lines.append(("body", tr("report.Heater_line",
                                     uh=f"{sc['uh']:g}", ih=f"{sc['ih']:g}")))

    if lamp_config and "nominal" in secs:
        common = dict(ua=f"{lamp_config.ua:.0f}", ug1=f"{lamp_config.ug1:.1f}",
                      ia=f"{lamp_config.ia:.0f}", uh=f"{lamp_config.uh:.1f}")
        if getattr(lamp_config, 'is_triode', False):
            lines.append(("body", tr("report.Nominal_triode", **common)))
        else:
            lines.append(("body", tr("report.Nominal_pentode", **common,
                                     ug2=f"{lamp_config.ug2:.0f}")))
        if lamp_config.pa_max:
            lines.append(("body", tr("report.Ref_params",
                                     pa=f"{lamp_config.pa_max:.1f}",
                                     s=f"{lamp_config.s:.1f}",
                                     r=f"{lamp_config.r:.1f}",
                                     k=f"{lamp_config.k:.0f}")))

    lines.append(("gap", ""))

    if srk and "srk" in secs:
        unc = srk.get("uncertainty") or {}

        def _fmt_srk(key: str) -> str:
            value = srk.get(key)
            if value is None:
                return "—"
            text = f"{value:.2f}"
            rel = unc.get(key)
            if rel is not None:
                text += f"±{rel * 100.0:.0f}%"
            return text

        lines.append(("body", tr("report.Measured", s=_fmt_srk("s"),
                                 r=_fmt_srk("r"), k=_fmt_srk("k"))))

    if quality and quality.verdict != "N/A" and "quality" in secs:
        ia_str = f"{quality.ia_pct:.0f}%" if quality.ia_pct is not None else "—"
        s_str = f"{quality.s_pct:.0f}%" if quality.s_pct is not None else "—"
        # Verdicts come from lm19.quality as English tokens; translate
        # when a key exists, keep the raw token visible when it doesn't
        # (a new verdict must not vanish from the report).
        verdict_key = f"report.Verdict_{quality.verdict}"
        verdict_disp = tr(verdict_key)
        if verdict_disp == verdict_key:
            verdict_disp = quality.verdict
        lines.append(("body", tr("report.Quality", verdict=verdict_disp,
                                 ia=ia_str, s=s_str)))

    # The analysis dict comes from the working-line controller; a missing
    # key means schema drift — skip the block loudly rather than crash
    # mid-render or print garbage (failure visibility).
    if analysis and "distortion" in secs:
        keys = ("hd2", "hd3", "pout_mw", "ua_0", "ia_0", "ug1_0")
        missing = sorted(k for k in keys if analysis.get(k) is None)
        if missing:
            log.warning(
                "PDF report: analysis dict missing keys %s — "
                "distortion block skipped", missing,
            )
        else:
            method_code = analysis.get("method")
            method_key = _HD_METHOD_LABEL_KEYS.get(method_code)
            method_label = tr(method_key) if method_key else str(
                method_code or "?")
            lines.append(("body", tr("report.Distortion",
                                     hd2=f"{analysis['hd2']:.2f}",
                                     hd3=f"{analysis['hd3']:.2f}",
                                     pout=f"{analysis['pout_mw']:.0f}",
                                     method=method_label)))
            lines.append(("body", tr("report.Q_point",
                                     ua=f"{analysis['ua_0']:.0f}",
                                     ia=f"{analysis['ia_0']:.1f}",
                                     ug1=f"{analysis['ug1_0']:.1f}")))

    n_pts = len(points) if points else 0
    lines.append(("body", tr("report.Scan_points", n=n_pts)))
    return lines


def generate_pdf_report(
    path: str,
    tube_type: str,
    lamp_id: str,
    lamp_config: Optional["LampConfig"] = None,
    points: Optional[List[Dict]] = None,
    srk: Optional[Dict] = None,
    quality: Optional["QualityReport"] = None,
    analysis: Optional[Dict] = None,
    plot_image: Optional[QPixmap] = None,
    transfer_image: Optional[QPixmap] = None,
    mfg_date: str = "",
    *,
    sections: Optional[Set[str]] = None,
    language: str = "en",
    scan_meta: Optional[Dict] = None,
) -> None:
    """Generate a PDF report with tube test results.

    Uses Qt's QPdfWriter for zero-dependency PDF generation.

    Args:
        path: output PDF file path
        tube_type: tube type name
        lamp_id: lamp identifier
        lamp_config: LampConfig with nominal values
        points: measurement points
        srk: dict with s, r, k values
        quality: QualityReport object
        analysis: distortion analysis dict from load line
        plot_image: QPixmap of the 2D plot
        transfer_image: QPixmap of the transfer plot
        mfg_date: manufacturing date "YYYY-MM" ("" = unknown, line omitted)
        sections: enabled section ids (None = all; see REPORT_SECTIONS)
        language: report text locale code (independent from the UI locale)
        scan_meta: measurement metadata dict (timestamp/name/topology/scan)

    Raises:
        OSError: when the output file cannot be opened for writing
            (file locked by a viewer, directory missing, no permission).
    """
    tr = translator_for(language or "en")
    secs = SECTION_IDS if sections is None else sections

    writer = QPdfWriter(path)
    page_layout = QPageLayout(
        QPageSize(QPageSize.PageSizeId.A4),
        QPageLayout.Orientation.Portrait,
        QMarginsF(_PAGE_MARGIN_MM, _PAGE_MARGIN_MM,
                  _PAGE_MARGIN_MM, _PAGE_MARGIN_MM),
    )
    writer.setPageLayout(page_layout)
    writer.setResolution(_REPORT_DPI)

    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = build_report_lines(
        tube_type, lamp_id, timestamp, lamp_config,
        points, srk, quality, analysis, mfg_date,
        sections=sections, scan_meta=scan_meta, tr=tr,
    )

    painter = QPainter(writer)
    # QPainter.begin() fails *silently* (returns False, no exception) on a
    # locked file or missing directory — without this check callers would
    # report success over a 0-byte file.
    if not painter.isActive():
        raise OSError(
            f"Cannot write PDF to '{path}' (file locked or path unavailable)"
        )
    try:
        _render_report(
            painter, writer, lines, timestamp,
            plot_image if "plot_curves" in secs else None,
            transfer_image if "plot_transfer" in secs else None,
            tr,
        )
    finally:
        painter.end()


def _render_report(
    painter: QPainter,
    writer: QPdfWriter,
    lines: List[ReportLine],
    timestamp: str,
    plot_image: Optional[QPixmap],
    transfer_image: Optional[QPixmap],
    tr: Optional[Callable[..., str]] = None,
) -> None:
    """Lay the content lines and plot images out onto the painter."""
    tr = tr or translator_for("en")
    dpi = writer.resolution()
    page_w = writer.width()
    page_h = writer.height()

    y = 0
    line_h = int(dpi * _LINE_HEIGHT_IN)

    title_font = QFont(_FONT_FAMILY, _TITLE_FONT_PT, QFont.Weight.Bold)
    body_font = QFont(_FONT_FAMILY, _BODY_FONT_PT)
    small_font = QFont(_FONT_FAMILY, _SMALL_FONT_PT)
    page_no = 1

    def _draw_footer() -> None:
        painter.setFont(small_font)
        painter.drawText(
            0, page_h - int(line_h * _FOOTER_RAISE),
            tr("report.Footer", date=timestamp, page=page_no,
               version=APP_VERSION),
        )

    def _draw_image(image: QPixmap) -> None:
        """Draw a pixmap scaled to page width; page-break when it does not
        fit (ML-089: the transfer plot used to be silently dropped)."""
        nonlocal y, page_no
        available = page_h - y - line_h  # reserve the footer line
        if available <= line_h * _IMAGE_MIN_LINES:
            _draw_footer()
            writer.newPage()
            page_no += 1
            y = 0
            available = page_h - y - line_h
        img_w = page_w
        img_h = int(img_w * image.height() / max(image.width(), 1))
        if img_h > available:
            img_h = available
            img_w = int(img_h * image.width() / max(image.height(), 1))
        painter.drawPixmap(QRect(0, y, img_w, img_h), image)
        y += img_h + int(line_h * _SECTION_GAP)

    for role, text in lines:
        if role != "gap":
            font = title_font if role == "title" else body_font
            painter.setFont(font)
            fm = QFontMetrics(font, writer)
            painter.drawText(
                0, y + line_h,
                fm.elidedText(text, Qt.TextElideMode.ElideRight, page_w),
            )
        y += int(line_h * _ROLE_ADVANCE[role])

    y += int(line_h * (_PLOT_GAP - 1.0))

    if plot_image and not plot_image.isNull():
        _draw_image(plot_image)
    if transfer_image and not transfer_image.isNull():
        _draw_image(transfer_image)

    _draw_footer()
