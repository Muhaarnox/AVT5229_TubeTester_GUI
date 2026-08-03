"""Shared QTextDocument → PDF printer for HTML-bodied reports.

Single source for the amplifier report and the matched-tubes certificate:
A4/`_REPORT_DPI` page setup, image embedding at page width, and the
failure-visibility probes (``QTextDocument.print_`` swallows write errors
exactly like ``QPainter`` — see ``app/report.py``).
"""

import logging
from html import escape
from pathlib import Path
from typing import List, Tuple

from PySide6.QtCore import QMarginsF, QSizeF, QUrl
from PySide6.QtGui import (
    QPageLayout,
    QPageSize,
    QPdfWriter,
    QPixmap,
    QTextDocument,
)

from app.report import _PAGE_MARGIN_MM, _REPORT_DPI
from app.ui_theme import COLOR_MUTED_TEXT

log = logging.getLogger(__name__)


def print_html_pdf(
    path: str,
    fragments: List[str],
    images: List[Tuple[str, QPixmap]],
    tail: str = "",
) -> None:
    """Write *fragments* + *images* (+ *tail*, e.g. a footer) as a PDF.

    Raises:
        OSError: when the output file cannot be written (locked by a
            viewer, missing directory) — detected by a pre-flight open
            probe and a post-write size check, since Qt raises nothing.
    """
    with open(path, "ab"):
        pass

    writer = QPdfWriter(path)
    writer.setPageLayout(QPageLayout(
        QPageSize(QPageSize.PageSizeId.A4),
        QPageLayout.Orientation.Portrait,
        QMarginsF(_PAGE_MARGIN_MM, _PAGE_MARGIN_MM,
                  _PAGE_MARGIN_MM, _PAGE_MARGIN_MM),
    ))
    writer.setResolution(_REPORT_DPI)

    doc = QTextDocument()
    doc.setPageSize(QSizeF(writer.width(), writer.height()))

    parts = list(fragments)
    for i, (caption, pixmap) in enumerate(images):
        url = QUrl(f"pdfdoc://img{i}")
        doc.addResource(QTextDocument.ResourceType.ImageResource,
                        url, pixmap.toImage())
        parts.append(
            f'<p style="color:{COLOR_MUTED_TEXT}">{escape(caption)}</p>'
            f'<img src="{url.toString()}" width="{writer.width()}">')
    if tail:
        parts.append(tail)

    doc.setHtml("".join(parts))
    doc.print_(writer)

    if Path(path).stat().st_size == 0:
        raise OSError(f"PDF write produced an empty file: '{path}'")
