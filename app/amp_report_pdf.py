"""Amplifier analysis PDF report.

Reuses the Amplifier tab's HTML results (``app.amplifier_report``) as the
text body and lays it out through ``QTextDocument`` (automatic pagination,
tables/colors for free), with the tab plots embedded as print-resolution
images. Report text follows the UI language — the HTML formatter is bound
to the global ``t()``; a language selector is deliberately not offered.

The section list :data:`AMP_REPORT_SECTIONS` plugs into the shared
``ReportOptionsDialog`` (``session_key="amp"``).
"""

import datetime as dt
import logging
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QSizeF, QUrl
from PySide6.QtGui import QPageLayout, QPageSize, QPdfWriter, QPixmap, QTextDocument
from PySide6.QtCore import QMarginsF

from app.report import _PAGE_MARGIN_MM, _REPORT_DPI, ReportSection
from app.ui_theme import COLOR_IA, COLOR_MUTED_TEXT
from i18n_setup import t
from lm19.version import APP_VERSION

log = logging.getLogger(__name__)

# ── module local constants ──
_SPECTRUM_MIN_HARMONICS = 2       # need at least HD2+HD3 to draw a spectrum
_SPECTRUM_PIXMAP_W = 600          # offscreen widget size before hi-res export
_SPECTRUM_PIXMAP_H = 380
_SPECTRUM_BAR_WIDTH = 0.6
_MAX_SPECTRUM_HARMONIC = 9        # HD2..HD9 — what the engine can produce

# Source of truth for the amplifier-report options dialog.
AMP_REPORT_SECTIONS: Tuple[ReportSection, ...] = (
    ReportSection("amp_results", "report.Sec_amp_results"),
    ReportSection("amp_ltspice", "report.Sec_amp_ltspice"),
    ReportSection("amp_spectrum", "report.Sec_amp_spectrum"),
    ReportSection("amp_plot_thd_sweep", "report.Sec_amp_thd_sweep"),
    ReportSection("amp_plot_ra_sweep", "report.Sec_amp_ra_sweep"),
    ReportSection("amp_plot_pareto", "report.Sec_amp_pareto"),
)
AMP_SECTION_IDS = frozenset(s.sid for s in AMP_REPORT_SECTIONS)

# Δ-column formatting guards
_DELTA_MIN_REF = 1e-9


def _fmt_num(value, decimals: int = 2) -> str:
    return "—" if value is None else f"{value:.{decimals}f}"


def _fmt_delta(sim, ref) -> str:
    if sim is None or ref is None or abs(ref) < _DELTA_MIN_REF:
        return "—"
    return f"{(sim - ref) / ref * 100.0:+.0f}%"


def build_verify_table_html(verify_result, engine_ref: Dict) -> str:
    """Engine-vs-LTspice comparison table (HTML for panel and report).

    Every column is labelled with its basis (method-visibility rule);
    the engine side may be absent (``engine_ref={}``) — the simulation
    numbers are then shown alone.
    """
    runs = verify_result.runs
    if not runs and not verify_result.warnings:
        return ""
    parts: List[str] = [f"<b>{escape(t('report.Verify_table_title'))}</b>"]
    for w in verify_result.warnings:
        parts.append(f'<span style="color:{COLOR_MUTED_TEXT}">⚠ '
                     f"{escape(str(w))}</span>")
    if verify_result.fit_rms_ma is not None:
        parts.append(
            f'<span style="color:{COLOR_MUTED_TEXT}">'
            f"{escape(t('report.Verify_fit_line', rms=f'{verify_result.fit_rms_ma:.2f}'))}"
            "</span>")
    if verify_result.workdir:
        # where the .sub/.cir/.log live — the user can re-run them
        # manually in the LTspice GUI
        parts.append(
            f'<span style="color:{COLOR_MUTED_TEXT}">'
            f"{escape(t('report.Verify_workdir_line', path=verify_result.workdir))}"
            "</span>")

    if runs:
        base = runs[-1]  # full-drive run (sweep runs ascending fractions)
        ref_basis = escape(str(engine_ref.get("basis", "—")))
        sim_basis = escape(verify_result.basis)
        rows = [
            ("THD %", engine_ref.get("thd"), base.thd_pct),
            ("HD2 %", engine_ref.get("hd2"), base.hd_pct.get(2)),
            ("HD3 %", engine_ref.get("hd3"), base.hd_pct.get(3)),
            ("Pout mW", engine_ref.get("pout_fund_mw"), base.pout_fund_mw),
        ]
        cells = [
            "<table border='1' cellspacing='0' cellpadding='3'>",
            f"<tr><th>{escape(t('report.Verify_col_metric'))}</th>"
            f"<th>{ref_basis}</th><th>{sim_basis}</th>"
            f"<th>{escape(t('report.Verify_col_delta'))}</th></tr>",
        ]
        for label, ref_v, sim_v in rows:
            cells.append(
                f"<tr><td>{label}</td><td>{_fmt_num(ref_v)}</td>"
                f"<td>{_fmt_num(sim_v)}</td>"
                f"<td>{_fmt_delta(sim_v, ref_v)}</td></tr>")
        if base.ia_avg_ma is not None:
            cells.append(f"<tr><td>Ia avg mA</td><td>—</td>"
                         f"<td>{_fmt_num(base.ia_avg_ma, 1)}</td><td>—</td></tr>")
        cells.append("</table>")
        parts.append("".join(cells))
        if engine_ref and not engine_ref.get("pout_is_fund", True):
            parts.append(f'<span style="color:{COLOR_MUTED_TEXT}">'
                         f"{escape(t('report.Verify_pout_peak_note'))}</span>")

    if len(runs) > 1:
        sweep_ref = engine_ref.get("sweep_amp") or []

        def _engine_thd_at(hs: float):
            best, best_d = None, None
            for row in sweep_ref:
                row_hs = row.get("half_swing")
                if row_hs is None:
                    continue
                d = abs(row_hs - hs)
                if best_d is None or d < best_d:
                    best, best_d = row.get("thd"), d
            return best

        cells = [f"<br><b>{escape(t('report.Verify_sweep_title'))}</b>",
                 "<table border='1' cellspacing='0' cellpadding='3'>",
                 "<tr><th>V half-swing</th><th>THD % (engine)</th>"
                 "<th>THD % (LTspice)</th></tr>"]
        for run in runs:
            cells.append(
                f"<tr><td>{run.half_swing:.2f}</td>"
                f"<td>{_fmt_num(_engine_thd_at(run.half_swing))}</td>"
                f"<td>{_fmt_num(run.thd_pct)}</td></tr>")
        cells.append("</table>")
        parts.append("".join(cells))

    if verify_result.imd is not None:
        engine_imd = engine_ref.get("imd") or {}
        parts.append(
            "IMD2: "
            f"{_fmt_num(engine_imd.get('imd2'), 1)} / "
            f"{_fmt_num(verify_result.imd.get('imd2'), 1)}%   IMD3: "
            f"{_fmt_num(engine_imd.get('imd3'))} / "
            f"{_fmt_num(verify_result.imd.get('imd3'))}%")
        parts.append(f'<span style="color:{COLOR_MUTED_TEXT}">'
                     f"{escape(t('report.Verify_imd_caveat'))}</span>")

    return "<br>".join(parts)


def build_amp_header_lines(lamp_id: str, params,
                           data_label: str = "") -> List[str]:
    """Header lines for the amplifier report (pure, testable).

    ``params`` is the ``AmpParams`` the analysis actually ran with — the
    circuit/Ub/Ra/bias/method the numbers belong to (method-visibility);
    ``data_label`` names the analyzed series (the plot may hold several
    lamps — the reader must know whose numbers these are).
    """
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [t("report.Lamp_line", id=lamp_id, date=timestamp)]
    if data_label:
        lines.append(t("report.Amp_data", label=data_label))
    if params is not None:
        lines.append(t(
            "report.Amp_params",
            circuit=str(getattr(params, "circuit", "?")),
            ub=f"{getattr(params, 'ub', 0):g}",
            ra=f"{getattr(params, 'ra', 0):g}",
            bias=f"{getattr(params, 'ug1_bias', 0):g}",
            method=str(getattr(params, 'hd_method', "?")),
        ))
        ug2 = getattr(params, "ug2_filter", None)
        if ug2:
            lines.append(t("report.Amp_ug2", ug2=f"{ug2:g}"))
    return lines


def render_spectrum_pixmap(dist: Dict) -> Optional[QPixmap]:
    """Bar chart of the harmonic spectrum (HD2..HD9, %) from a distortion
    dict; ``None`` when fewer than two harmonics are present."""
    harmonics = [(n, dist.get(f"hd{n}"))
                 for n in range(2, _MAX_SPECTRUM_HARMONIC + 1)]
    known = [(n, float(v)) for n, v in harmonics if v is not None]
    if len(known) < _SPECTRUM_MIN_HARMONICS:
        return None

    import pyqtgraph as pg
    from app.export_manager import render_plot_pixmap

    widget = pg.PlotWidget(title=t("report.Amp_spectrum_title"))
    widget.resize(_SPECTRUM_PIXMAP_W, _SPECTRUM_PIXMAP_H)
    widget.setLabel("left", "%")
    widget.setLabel("bottom", "HD n")
    bars = pg.BarGraphItem(
        x=[n for n, _ in known], height=[v for _, v in known],
        width=_SPECTRUM_BAR_WIDTH, brush=COLOR_IA,
    )
    widget.addItem(bars)
    widget.getAxis("bottom").setTicks(
        [[(n, str(n)) for n, _ in known]])
    return render_plot_pixmap(widget)


def generate_amp_pdf_report(
    path: str,
    *,
    tube_type: str,
    header_lines: List[str],
    results_html: str,
    images: List[Tuple[str, QPixmap]],
    verify_html: str = "",
) -> None:
    """Write the amplifier report PDF.

    Args:
        path: output PDF file path.
        tube_type: tube name for the title.
        header_lines: plain-text header lines (escaped here).
        results_html: HTML fragment of the results panel ("" = omitted).
        images: (caption, pixmap) pairs, drawn full page width in order.
        verify_html: LTspice-verification block ("" = omitted).

    Raises:
        OSError: when the output file cannot be written (locked by a
            viewer, missing directory) — QTextDocument.print_, like
            QPainter, swallows write errors silently.
    """
    from app.pdf_doc import print_html_pdf

    parts: List[str] = [
        f"<h2>{escape(t('report.Amp_title', tube=tube_type))}</h2>",
    ]
    parts.extend(f"<p>{escape(line)}</p>" for line in header_lines)
    if results_html:
        parts.append("<hr>")
        parts.append(results_html)
    if verify_html:
        parts.append("<hr>")
        parts.append(verify_html)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer_text = t("report.Footer", date=timestamp, page=1,
                    version=APP_VERSION)
    footer = (f'<p style="color:{COLOR_MUTED_TEXT}">'
              f"{escape(footer_text)}</p>")
    print_html_pdf(path, parts, images, tail=footer)
