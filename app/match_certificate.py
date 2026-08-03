"""Matched pair/quad PDF certificate (stage 3 of the PDF-reports plan).

Both matching UIs hand a ``MatchGroup`` here: the Health Match panel
(Ia/S/R at the operating point) and the Compare tab (full-curve match).
No free-text fields by design; the certificate language
comes from the shared report-options dialog (``translator_for``), so a
document for a buyer can be printed in any available locale.
"""

import datetime as dt
import logging
from html import escape
from typing import Callable, Dict, List, Optional, Tuple

from PySide6.QtGui import QPixmap

from app.report import ReportSection
from app.ui_theme import COLOR_MUTED_TEXT
from i18n_setup import translator_for
from lm19.tube_matching import MatchGroup, delta_quality
from lm19.version import APP_VERSION

log = logging.getLogger(__name__)

# Source of truth for the certificate options dialog (session_key="cert").
CERT_SECTIONS: Tuple[ReportSection, ...] = (
    ReportSection("cert_conditions", "report.Sec_cert_conditions"),
    ReportSection("cert_metrics", "report.Sec_cert_metrics"),
    ReportSection("cert_plot", "report.Sec_cert_plot"),
)
CERT_SECTION_IDS = frozenset(s.sid for s in CERT_SECTIONS)


def pick_match_group(parent, result) -> Optional[MatchGroup]:
    """Select a ≥2-tube group from a match result; ``None`` = no groups
    (warned visibly) or user cancel. Single-record "groups" (similar-mode
    ranking rows) are not certificate material."""
    from PySide6.QtWidgets import QInputDialog, QMessageBox

    from i18n_setup import t

    groups = [g for g in (result.groups if result else [])
              if len(g.records) >= 2]
    if not groups:
        QMessageBox.warning(parent, t("report.Cert_btn"),
                            t("report.Cert_no_groups"))
        return None
    if len(groups) == 1:
        return groups[0]
    labels = [t("report.Cert_group_label", n=g.number,
                delta=f"{g.delta:.2f}",
                ids=" + ".join(r.lamp_id for r in g.records))
              for g in groups]
    item, ok = QInputDialog.getItem(parent, t("report.Cert_btn"),
                                    t("report.Cert_pick_group"),
                                    labels, 0, False)
    if not ok:
        return None
    return groups[labels.index(item)]


def _quality_label(delta: float, tr: Callable[..., str],
                   amp_class: Optional[str] = None) -> str:
    return tr(f"report.Cert_q_{delta_quality(delta, amp_class)}")


def _head(tube_type: str, group: MatchGroup, quality: str,
          tr: Callable[..., str]) -> List[str]:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [
        f"<h2>{escape(tr('report.Cert_title', tube=tube_type))}</h2>",
        f"<p>{escape(tr('report.Cert_date', date=timestamp))}<br>"
        f"{escape(tr('report.Cert_group_line', n=group.number, size=len(group.records), delta=f'{group.delta:.2f}', quality=quality))}"
        "</p>",
    ]


def build_health_cert_fragments(
    group: MatchGroup,
    *,
    tube_type: str,
    sections: Optional[frozenset] = None,
    tr: Optional[Callable[..., str]] = None,
) -> List[str]:
    """Certificate body for a Health match group (Ia/S/R at the OP)."""
    tr = tr or translator_for("en")
    secs = CERT_SECTION_IDS if sections is None else sections
    parts = _head(tube_type, group,
                  _quality_label(group.delta, tr), tr)
    parts.append(f'<p style="color:{COLOR_MUTED_TEXT}">'
                 f"{escape(tr('report.Cert_basis_health'))}</p>")

    entry0 = group.records[0].entry or {}
    conditions = entry0.get("conditions") or {}
    if "cert_conditions" in secs and conditions:
        # health measurements store the mode as conditions["ug2_mode"]
        mode = (conditions.get("ug2_mode") or conditions.get("mode")
                or entry0.get("ug2_mode") or "?")
        parts.append("<p>" + escape(tr(
            "report.Cert_conditions",
            ua=f"{conditions.get('ua', 0):g}",
            ug1=f"{conditions.get('ug1', 0):g}",
            ug2=f"{conditions.get('ug2', 0):g}",
            mode=str(mode),
        )) + "</p>")

    if "cert_metrics" in secs:
        rows = ["<table border='1' cellspacing='0' cellpadding='3'>",
                f"<tr><th>{escape(tr('report.Cert_th_lamp'))}</th>"
                f"<th>{escape(tr('report.Cert_th_an'))}</th>"
                f"<th>{escape(tr('report.Cert_th_ts'))}</th>"
                f"<th>{escape(tr('report.Cert_th_mfg'))}</th>"
                "<th>Ia mA</th><th>S mA/V</th><th>R kΩ</th>"
                f"<th>{escape(tr('report.Cert_th_index'))}</th></tr>"]
        for rec in group.records:
            entry = rec.entry or {}
            index = (f"{rec.index:.0f}" if rec.index is not None else "—")
            rows.append(
                f"<tr><td>{escape(rec.lamp_id)}</td><td>{rec.an}</td>"
                f"<td>{escape(rec.timestamp)}</td>"
                f"<td>{escape(str(entry.get('mfg_date', '') or '—'))}</td>"
                f"<td>{rec.ia:.1f}</td><td>{rec.s:.2f}</td>"
                f"<td>{rec.r:.2f}</td><td>{index}</td></tr>")
        rows.append("</table>")
        parts.append("".join(rows))
        # shared_bias protocol: the buyer of a common-bias amp cares about
        # exactly this number — the current split the pair will settle at.
        if group.iq_imbalance_ma is not None:
            parts.append("<p>" + escape(tr(
                "report.Cert_iq_imbalance",
                ma=f"{group.iq_imbalance_ma:.1f}")) + "</p>")
    return parts


def build_compare_cert_fragments(
    group: MatchGroup,
    *,
    tube_type: str,
    entries: List[Dict],
    pair_info: Dict,
    amp_class: Optional[str] = None,
    sections: Optional[frozenset] = None,
    tr: Optional[Callable[..., str]] = None,
) -> List[str]:
    """Certificate body for a Compare curve-match group.

    ``entries`` is the entry list the match ran on (records carry
    ``entry["_index"]`` into it); ``pair_info`` is
    ``CurveMatchResult.pair_info``.
    """
    tr = tr or translator_for("en")
    secs = CERT_SECTION_IDS if sections is None else sections
    quality = _quality_label(group.delta, tr, amp_class)
    parts = _head(tube_type, group, quality, tr)
    cls = f" [{amp_class}]" if amp_class else ""
    parts.append(f'<p style="color:{COLOR_MUTED_TEXT}">'
                 f"{escape(tr('report.Cert_basis_compare', cls=cls))}</p>")
    parts.append("<p>" + escape(tr(
        "report.Cert_match_line",
        pct=f"{100.0 - group.delta:.1f}", quality=quality)) + "</p>")

    indices = [(r.entry or {}).get("_index") for r in group.records]
    indices = [i for i in indices if i is not None and i < len(entries)]

    if "cert_conditions" in secs and indices:
        data = entries[indices[0]].get("data") or {}
        mode = (data.get("scan") or {}).get("ug2_mode") \
            or data.get("topology") or "?"
        parts.append("<p>" + escape(tr("report.Cert_mode_line",
                                       mode=str(mode))) + "</p>")

    if "cert_metrics" in secs:
        rows = ["<table border='1' cellspacing='0' cellpadding='3'>",
                f"<tr><th>{escape(tr('report.Cert_th_lamp'))}</th>"
                f"<th>{escape(tr('report.Cert_th_name'))}</th>"
                f"<th>{escape(tr('report.Cert_th_mfg'))}</th>"
                f"<th>{escape(tr('report.Cert_th_ts'))}</th></tr>"]
        for idx in indices:
            e = entries[idx]
            data = e.get("data") or {}
            mfg = e.get("mfg_date") or data.get("mfg_date") or "—"
            rows.append(
                f"<tr><td>{escape(str(e.get('lamp_id', '')))}</td>"
                f"<td>{escape(str(e.get('name', '')))}</td>"
                f"<td>{escape(str(mfg))}</td>"
                f"<td>{escape(str(e.get('timestamp', '')))}</td></tr>")
        rows.append("</table>")
        parts.append("".join(rows))

        if len(indices) > 2:
            pair_rows = [f"<p>{escape(tr('report.Cert_pairwise'))}</p>",
                         "<table border='1' cellspacing='0' cellpadding='3'>"]
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    key = (min(indices[a], indices[b]),
                           max(indices[a], indices[b]))
                    info = pair_info.get(key)
                    if info is None:
                        continue
                    pct = f"{100.0 - info.distance:.1f}"
                    pair_rows.append(
                        f"<tr><td>{escape(str(entries[indices[a]].get('lamp_id', '')))}"
                        f" + {escape(str(entries[indices[b]].get('lamp_id', '')))}</td>"
                        f"<td>{pct}%</td></tr>")
            pair_rows.append("</table>")
            parts.append("".join(pair_rows))
    return parts


def generate_certificate_pdf(
    path: str,
    *,
    fragments: List[str],
    image: Optional[QPixmap] = None,
    image_caption: str = "",
    tr: Optional[Callable[..., str]] = None,
) -> None:
    """Write the certificate PDF (shared HTML printer, footer appended).

    Raises:
        OSError: when the file cannot be written (see ``pdf_doc``).
    """
    from app.pdf_doc import print_html_pdf

    tr = tr or translator_for("en")
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer_text = tr("report.Footer", date=timestamp, page=1,
                     version=APP_VERSION)
    footer = (f'<p style="color:{COLOR_MUTED_TEXT}">'
              f"{escape(footer_text)}</p>")
    images = [(image_caption, image)] if image is not None else []
    print_html_pdf(path, fragments, images, tail=footer)
