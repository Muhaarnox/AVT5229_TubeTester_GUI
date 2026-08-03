"""Tests for PDF report generation (app.report).

Three layers:
- smoke: generator runs end-to-end and produces a non-trivial PDF file;
- content pins: ``build_report_lines`` is pure — its output is asserted
  directly (QtPdf cannot extract text back out of QPdfWriter output);
- layout/call-site pins: elision, pagination, footer, failure visibility,
  and export_pdf/CompareTab argument marshalling (spy pins).
"""

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtGui import QPdfWriter, QPixmap
from PySide6.QtPdf import QPdfDocument
from PySide6.QtWidgets import QApplication, QWidget

from app.report import _render_report, build_report_lines, generate_pdf_report
from lm19.quality import QualityReport
from lm19.tube_sim import quick_pentode, quick_triode
from lm19.constants import (
    TOPOLOGY_PENTODE,
)


# ── Module local constants ──
_MIN_PDF_SIZE_BYTES = 1000  # any real PDF >> 1 KB; 0-byte file = silent failure
_TS = "2026-07-12 10:00:00"  # fixed timestamp for pure content pins


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _LampConfigStub:
    """Minimal lamp-config-shaped object accepted by generate_pdf_report."""

    def __init__(self, *, is_triode: bool = False) -> None:
        self.is_triode = is_triode
        self.ua = 250.0
        self.ug1 = -2.0
        self.ug2 = 250.0
        self.ia = 8.0
        self.uh = 6.3
        self.pa_max = 12.0
        self.s = 4.5
        self.r = 50.0
        self.k = 1500.0


def _make_pixmap() -> QPixmap:
    """Small non-null QPixmap that PDF generator can embed."""
    pm = QPixmap(120, 80)
    pm.fill()
    return pm


# ----------------------------------------------------------------------
# Golden path
# ----------------------------------------------------------------------


def test_generate_pdf_basic_pentode(qapp, tmp_path):
    """Pentode flow with all fields populated produces a valid PDF file."""
    _, points = quick_pentode("EL84")
    out = tmp_path / "report.pdf"
    generate_pdf_report(
        path=str(out),
        tube_type="EL84",
        lamp_id="L01",
        lamp_config=_LampConfigStub(is_triode=False),
        points=points,
        srk={"s": 11.0, "r": 38.0, "k": 1500.0},
        quality=QualityReport(ia_pct=98.0, s_pct=102.0, r_pct=95.0,
                              verdict="Good"),
        analysis={"hd2": 1.5, "hd3": 0.4, "pout_mw": 1800.0,
                  "ua_0": 250.0, "ia_0": 8.0, "ug1_0": -7.0},
        plot_image=_make_pixmap(),
        transfer_image=_make_pixmap(),
    )
    assert out.exists()
    assert out.stat().st_size > _MIN_PDF_SIZE_BYTES
    assert out.read_bytes().startswith(b"%PDF")


def test_generate_pdf_basic_triode(qapp, tmp_path):
    """Triode flow renders the triode-tagged nominal line without crashing."""
    _, points = quick_triode("12AX7")
    out = tmp_path / "triode.pdf"
    generate_pdf_report(
        path=str(out),
        tube_type="12AX7",
        lamp_id="T1",
        lamp_config=_LampConfigStub(is_triode=True),
        points=points,
        srk={"s": 1.6, "r": 62.5, "k": 100.0},
        quality=QualityReport(ia_pct=95.0, s_pct=97.0, r_pct=None,
                              verdict="Good"),
        analysis=None,
        plot_image=_make_pixmap(),
    )
    assert out.exists()
    assert out.stat().st_size > _MIN_PDF_SIZE_BYTES


# ----------------------------------------------------------------------
# Edge cases — every optional field omitted independently
# ----------------------------------------------------------------------


def test_generate_pdf_no_lamp_config(qapp, tmp_path):
    out = tmp_path / "no_cfg.pdf"
    generate_pdf_report(
        path=str(out),
        tube_type="ECC83",
        lamp_id="X",
        lamp_config=None,
        points=[{"ua": 100.0, "ug1": -1.0, "ia": 1.5}],
    )
    assert out.exists() and out.stat().st_size > _MIN_PDF_SIZE_BYTES


def test_generate_pdf_no_points(qapp, tmp_path):
    """`points=None` and `points=[]` both yield 'Scan points: 0' line."""
    for points in (None, []):
        out = tmp_path / f"empty_{points!r}.pdf"
        generate_pdf_report(
            path=str(out),
            tube_type="EL84",
            lamp_id="X",
            lamp_config=_LampConfigStub(),
            points=points,
        )
        assert out.exists() and out.stat().st_size > _MIN_PDF_SIZE_BYTES


def test_generate_pdf_partial_srk(qapp, tmp_path):
    """SRK with some None values should render '—' placeholders."""
    out = tmp_path / "partial_srk.pdf"
    generate_pdf_report(
        path=str(out),
        tube_type="EL84",
        lamp_id="X",
        lamp_config=_LampConfigStub(),
        srk={"s": 11.0, "r": None, "k": None},
    )
    assert out.exists() and out.stat().st_size > _MIN_PDF_SIZE_BYTES


def test_generate_pdf_quality_na_skipped(qapp, tmp_path):
    """Quality with verdict='N/A' is rendered as if absent (skipped)."""
    out = tmp_path / "na_quality.pdf"
    generate_pdf_report(
        path=str(out),
        tube_type="EL84",
        lamp_id="X",
        lamp_config=_LampConfigStub(),
        quality=QualityReport(None, None, None, "N/A"),
    )
    assert out.exists() and out.stat().st_size > _MIN_PDF_SIZE_BYTES


def test_generate_pdf_no_plot_images(qapp, tmp_path):
    """Missing plot images must not raise; PDF still generated."""
    out = tmp_path / "no_imgs.pdf"
    generate_pdf_report(
        path=str(out),
        tube_type="EL84",
        lamp_id="X",
        lamp_config=_LampConfigStub(),
        plot_image=None,
        transfer_image=None,
    )
    assert out.exists() and out.stat().st_size > _MIN_PDF_SIZE_BYTES


def test_generate_pdf_null_pixmap(qapp, tmp_path):
    """A constructed-but-null QPixmap must be skipped (isNull check)."""
    out = tmp_path / "null_pm.pdf"
    generate_pdf_report(
        path=str(out),
        tube_type="EL84",
        lamp_id="X",
        lamp_config=_LampConfigStub(),
        plot_image=QPixmap(),
        transfer_image=QPixmap(),
    )
    assert out.exists() and out.stat().st_size > _MIN_PDF_SIZE_BYTES


def test_generate_pdf_only_required_fields(qapp, tmp_path):
    """Bare-minimum call (everything optional → None) still produces PDF."""
    out = tmp_path / "minimal.pdf"
    generate_pdf_report(
        path=str(out),
        tube_type="EL84",
        lamp_id="L1",
    )
    assert out.exists() and out.stat().st_size > _MIN_PDF_SIZE_BYTES


# ----------------------------------------------------------------------
# Content pins — build_report_lines is pure, assert its output directly
# ----------------------------------------------------------------------


def _text(lines) -> str:
    return "\n".join(t for _, t in lines)


class TestBuildReportLines:
    def test_title_first_with_tube_type(self):
        # Native-script tube name ("6P14P-ER") — non-ASCII survives as-is.
        soviet_tube = "6\u041f14\u041f-\u0415\u0420"
        lines = build_report_lines(soviet_tube, "L01", _TS)
        assert lines[0] == ("title", f"Tube Test Report: {soviet_tube}")

    def test_lamp_id_and_timestamp_line(self):
        # Non-ASCII lamp id ("Lampa-42") — free-text field survives as-is.
        nonascii_lamp_id = "\u041b\u0430\u043c\u043f\u0430-42"
        lines = build_report_lines("EL84", nonascii_lamp_id, _TS)
        assert lines[1] == ("body", f"Lamp ID: {nonascii_lamp_id}    Date: {_TS}")

    def test_mfg_date_line_present_when_given(self):
        lines = build_report_lines("EL84", "X", _TS, mfg_date="1962-05")
        assert ("body", "Manufactured: 1962-05") in lines

    def test_mfg_date_line_absent_when_empty(self):
        lines = build_report_lines("EL84", "X", _TS, mfg_date="")
        assert "Manufactured" not in _text(lines)

    def test_pentode_nominal_has_ug2_and_no_triode_tag(self):
        lines = build_report_lines(
            "EL84", "X", _TS, lamp_config=_LampConfigStub(is_triode=False))
        txt = _text(lines)
        assert "Ug2=250V" in txt
        assert "[triode]" not in txt

    def test_triode_nominal_tagged_and_ug2_free(self):
        lines = build_report_lines(
            "12AX7", "X", _TS, lamp_config=_LampConfigStub(is_triode=True))
        txt = _text(lines)
        assert "[triode]" in txt
        assert "Ug2=" not in txt

    def test_pa_max_reference_line_gated(self):
        cfg = _LampConfigStub()
        with_pa = _text(build_report_lines("EL84", "X", _TS, lamp_config=cfg))
        assert "Pa_max=12.0W" in with_pa and "S=4.5mA/V" in with_pa
        cfg.pa_max = 0.0
        without = _text(build_report_lines("EL84", "X", _TS, lamp_config=cfg))
        assert "Pa_max" not in without

    def test_partial_srk_dash_placeholders(self):
        lines = build_report_lines(
            "EL84", "X", _TS, srk={"s": 11.0, "r": None, "k": None})
        txt = _text(lines)
        assert "S=11.00 mA/V" in txt
        assert "R=— kΩ" in txt and "K=—" in txt

    def test_no_srk_no_measured_line(self):
        assert "Measured:" not in _text(build_report_lines("EL84", "X", _TS))

    def test_quality_percent_slots_not_swapped(self):
        q = QualityReport(ia_pct=98.0, s_pct=102.0, r_pct=None, verdict="Good")
        txt = _text(build_report_lines("EL84", "X", _TS, quality=q))
        assert "Quality: Good  (Ia: 98%  S: 102% of nominal)" in txt

    def test_quality_na_block_skipped(self):
        q = QualityReport(None, None, None, "N/A")
        assert "Quality" not in _text(
            build_report_lines("EL84", "X", _TS, quality=q))

    def test_analysis_values_in_right_slots(self):
        from lm19.amplifier.constants import HD_METHOD_CHEBYSHEV
        analysis = {"hd2": 1.5, "hd3": 0.4, "pout_mw": 1800.0,
                    "ua_0": 250.0, "ia_0": 8.0, "ug1_0": -7.0,
                    "method": HD_METHOD_CHEBYSHEV}
        txt = _text(build_report_lines("EL84", "X", _TS, analysis=analysis))
        # Method visibility: with auto-routing the numbers are not
        # necessarily 5-point — the report must say which method.
        assert "Distortion: HD2=1.50%  HD3=0.40%  Pout=1800mW — Chebyshev" \
            in txt
        assert "Q-point: Ua=250V  Ia=8.0mA  Ug1=-7.0V" in txt

    def test_analysis_method_label_twins_all_codes(self):
        """Twins: every resolved method code maps to ITS OWN label — a
        swapped mapping would label DFT numbers '5-point' and vice
        versa, and only the chebyshev pin above would stay green."""
        from lm19.amplifier.constants import (
            HD_METHOD_5POINT,
            HD_METHOD_CHEBYSHEV,
            HD_METHOD_DFT,
        )
        expected = {HD_METHOD_5POINT: "5-point",
                    HD_METHOD_CHEBYSHEV: "Chebyshev",
                    HD_METHOD_DFT: "DFT (model)"}
        for code, label in expected.items():
            analysis = {"hd2": 1.5, "hd3": 0.4, "pout_mw": 1800.0,
                        "ua_0": 250.0, "ia_0": 8.0, "ug1_0": -7.0,
                        "method": code}
            txt = _text(build_report_lines("EL84", "X", _TS,
                                           analysis=analysis))
            assert f"Pout=1800mW — {label}" in txt, code

    def test_method_label_map_covers_registry(self):
        """Completeness from the source-of-truth registry: a new HD
        method must get a label key, or it would print raw forever
        (auto is a UI routing value, never a resolved method_used)."""
        from app.report import _HD_METHOD_LABEL_KEYS
        from lm19.amplifier.constants import HD_METHOD_AUTO, HD_METHODS
        assert set(_HD_METHOD_LABEL_KEYS) == HD_METHODS - {HD_METHOD_AUTO}

    def test_analysis_method_missing_prints_question_mark(self):
        """A missing method code stays VISIBLE as '?' — not silently
        dropped (an unlabeled number would claim more than we know)."""
        analysis = {"hd2": 1.5, "hd3": 0.4, "pout_mw": 1800.0,
                    "ua_0": 250.0, "ia_0": 8.0, "ug1_0": -7.0}
        txt = _text(build_report_lines("EL84", "X", _TS, analysis=analysis))
        assert "Pout=1800mW — ?" in txt

    def test_analysis_unknown_method_printed_raw(self):
        """An unknown code is printed raw rather than mislabeled."""
        analysis = {"hd2": 1.5, "hd3": 0.4, "pout_mw": 1800.0,
                    "ua_0": 250.0, "ia_0": 8.0, "ug1_0": -7.0,
                    "method": "quantum"}
        txt = _text(build_report_lines("EL84", "X", _TS, analysis=analysis))
        assert "Pout=1800mW — quantum" in txt

    def test_analysis_missing_key_skips_block_loudly(self, caplog):
        analysis = {"hd2": 1.5, "pout_mw": 1800.0,
                    "ua_0": 250.0, "ia_0": 8.0, "ug1_0": -7.0}  # no hd3
        with caplog.at_level(logging.WARNING, logger="app.report"):
            txt = _text(build_report_lines("EL84", "X", _TS, analysis=analysis))
        assert "Distortion" not in txt and "Q-point" not in txt
        assert any("hd3" in r.message for r in caplog.records)

    @pytest.mark.parametrize("points,expected", [
        ([{"ua": 1.0}, {"ua": 2.0}, {"ua": 3.0}], "Scan points: 3"),
        (None, "Scan points: 0"),
        ([], "Scan points: 0"),
    ])
    def test_scan_point_count(self, points, expected):
        assert expected in _text(
            build_report_lines("EL84", "X", _TS, points=points))


# ----------------------------------------------------------------------
# Layout pins — elision, pagination, footer
# ----------------------------------------------------------------------


class _RecordingPainter:
    """Duck-typed painter capturing drawn text (QPainter is not spyable)."""

    def __init__(self) -> None:
        self.texts: list = []

    def setFont(self, font) -> None:
        pass

    def drawText(self, x, y, text) -> None:
        self.texts.append(text)

    def drawPixmap(self, rect, pixmap) -> None:
        pass


def _tall_pixmap() -> QPixmap:
    pm = QPixmap(300, 2400)
    pm.fill()
    return pm


class TestRenderLayout:
    def test_long_lamp_id_elided_not_overflowed(self, qapp, tmp_path):
        long_id = "W" * 300
        writer = QPdfWriter(str(tmp_path / "elide.pdf"))
        lines = build_report_lines("EL84", long_id, _TS)
        rec = _RecordingPainter()
        _render_report(rec, writer, lines, _TS, None, None)
        id_lines = [t for t in rec.texts if t.startswith("Lamp ID:")]
        assert id_lines, "lamp-id line was not drawn at all"
        assert long_id not in id_lines[0]
        assert "…" in id_lines[0]

    def test_two_tall_images_paginate(self, qapp, tmp_path):
        """ML-089 branch: second image must open a new page, not vanish."""
        out = tmp_path / "two_pages.pdf"
        generate_pdf_report(
            path=str(out), tube_type="EL84", lamp_id="X",
            plot_image=_tall_pixmap(), transfer_image=_tall_pixmap(),
        )
        doc = QPdfDocument()
        assert doc.load(str(out)) == QPdfDocument.Error.None_
        assert doc.pageCount() == 2

    def test_footer_stamped_on_every_page(self, qapp, tmp_path):
        writer = QPdfWriter(str(tmp_path / "footer.pdf"))
        lines = build_report_lines("EL84", "X", _TS)
        rec = _RecordingPainter()
        _render_report(rec, writer, lines, _TS,
                       _tall_pixmap(), _tall_pixmap())
        footers = [t for t in rec.texts if "Generated by LM19" in t]
        assert [f.rsplit(" ", 1)[-1] for f in footers] == ["1", "2"]

    def test_single_page_single_footer(self, qapp, tmp_path):
        writer = QPdfWriter(str(tmp_path / "one_footer.pdf"))
        lines = build_report_lines("EL84", "X", _TS)
        rec = _RecordingPainter()
        _render_report(rec, writer, lines, _TS, None, None)
        footers = [t for t in rec.texts if "Generated by LM19" in t]
        assert len(footers) == 1 and footers[0].endswith("p. 1")


# ----------------------------------------------------------------------
# Failure visibility — write errors must not report success
# ----------------------------------------------------------------------


class TestWriteFailureVisibility:
    def test_missing_directory_raises_oserror(self, qapp, tmp_path):
        target = tmp_path / "no_such_dir" / "report.pdf"
        with pytest.raises(OSError, match="Cannot write PDF"):
            generate_pdf_report(path=str(target), tube_type="EL84",
                                lamp_id="X")
        assert not target.exists()

    def test_export_pdf_write_error_reaches_user(self, qapp, tmp_path,
                                                 monkeypatch):
        """export_pdf must show critical (not success) when writing fails."""
        import app.export_manager as em
        import app.report as report_mod

        def _boom(**kwargs):
            raise OSError("boom-disk")

        monkeypatch.setattr(report_mod, "generate_pdf_report", _boom)
        monkeypatch.setattr(
            em.QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (str(tmp_path / "o.pdf"), "")))
        infos, criticals = [], []
        monkeypatch.setattr(em.QMessageBox, "information",
                            staticmethod(lambda *a, **k: infos.append(a)))
        monkeypatch.setattr(em.QMessageBox, "critical",
                            staticmethod(lambda *a, **k: criticals.append(a)))
        w = QWidget()
        em.export_pdf(parent=None, points=[{"ua": 1.0}], tube_type="EL84",
                      lamp_id="X", lamp=None, srk_results=[],
                      plot_renderer=None, plot_widget=w)
        assert not infos, "success dialog shown despite write failure"
        assert criticals and "boom-disk" in str(criticals[0][2])


# ----------------------------------------------------------------------
# Call-site spy pins — export_pdf marshalling (call-site ≠ function)
# ----------------------------------------------------------------------


class TestExportPdfCallsite:
    def test_marshals_all_arguments(self, qapp, tmp_path, monkeypatch):
        import app.export_manager as em
        import app.report as report_mod

        captured: dict = {}
        monkeypatch.setattr(report_mod, "generate_pdf_report",
                            lambda **kw: captured.update(kw))
        monkeypatch.setattr(
            em.QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (str(tmp_path / "out.pdf"), "")))
        infos = []
        monkeypatch.setattr(em.QMessageBox, "information",
                            staticmethod(lambda *a, **k: infos.append(a)))

        pts = [{"ua": 10.0 * i, "ug1": -1.0, "ia": float(i)}
               for i in range(3)]
        analysis = {"hd2": 1.0, "hd3": 2.0, "pout_mw": 3.0,
                    "ua_0": 4.0, "ia_0": 5.0, "ug1_0": -6.0}
        plot_w, transfer_w = QWidget(), QWidget()
        plot_w.resize(100, 60)
        transfer_w.resize(100, 60)

        em.export_pdf(
            parent=None, points=pts, tube_type="EL84", lamp_id="L77",
            lamp=None,
            srk_results=[{"s": 1.0, "r": 2.0, "k": 3.0},
                         {"s": 3.0, "r": 6.0, "k": 9.0}],
            plot_renderer=SimpleNamespace(_load_line_analysis=analysis),
            plot_widget=plot_w, transfer_widget=transfer_w,
            mfg_date="1962-05",
        )
        assert captured["path"] == str(tmp_path / "out.pdf")
        assert captured["tube_type"] == "EL84"
        assert captured["lamp_id"] == "L77"
        assert captured["points"] is pts
        assert captured["mfg_date"] == "1962-05"
        # asymmetric inputs → averaging (not first/last picking) is pinned
        assert captured["srk"] == {"s": 2.0, "r": 4.0, "k": 6.0}
        assert captured["analysis"] is analysis
        assert captured["quality"] is None  # lamp=None → no quality
        assert not captured["plot_image"].isNull()
        assert not captured["transfer_image"].isNull()
        assert infos, "success dialog missing"

    def test_no_points_warns_and_skips_dialog(self, qapp, monkeypatch):
        import app.export_manager as em

        warns, saves = [], []
        monkeypatch.setattr(em.QMessageBox, "warning",
                            staticmethod(lambda *a, **k: warns.append(a)))
        monkeypatch.setattr(
            em.QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: saves.append(a) or ("", "")))
        em.export_pdf(parent=None, points=[], tube_type="EL84", lamp_id="X",
                      lamp=None, srk_results=[], plot_renderer=None,
                      plot_widget=QWidget())
        assert warns and not saves

    def test_cancelled_dialog_generates_nothing(self, qapp, monkeypatch):
        import app.export_manager as em
        import app.report as report_mod

        calls = []
        monkeypatch.setattr(report_mod, "generate_pdf_report",
                            lambda **kw: calls.append(kw))
        monkeypatch.setattr(em.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: ("", "")))
        em.export_pdf(parent=None, points=[{"ua": 1.0}], tube_type="EL84",
                      lamp_id="X", lamp=None, srk_results=[],
                      plot_renderer=None, plot_widget=QWidget())
        assert calls == []


# ----------------------------------------------------------------------
# Call-site spy pins — CompareTab PDF paths
# ----------------------------------------------------------------------


class TestComparePdfCallsites:
    @staticmethod
    def _entry(lamp_id: str, *, entry_mfg: str = "",
               data_mfg: str = "", srk=None) -> dict:
        data: dict = {"topology": TOPOLOGY_PENTODE}
        if data_mfg:
            data["mfg_date"] = data_mfg
        if srk:
            data["srk"] = srk
        return {
            "lamp_type": "EL84", "lamp_id": lamp_id, "name": f"m_{lamp_id}",
            "mfg_date": entry_mfg,
            "points": [{"ua": 50.0, "ug1": -2.0, "ia": 5.0}],
            "data": data,
        }

    def test_single_marshals_saved_srk_and_data_mfg(self, qapp, monkeypatch):
        import app.compare_tab as ct
        tab = ct.CompareTab()
        captured: dict = {}
        monkeypatch.setattr(ct, "export_pdf",
                            lambda **kw: captured.update(kw))
        srk = {"s": 5.0, "r": 6.0, "k": 7.0}
        entry = self._entry("A1", data_mfg="1961-03", srk=srk)
        monkeypatch.setattr(tab, "_get_checked_entries", lambda: [entry])
        tab._export_pdf()
        assert captured["srk_results"] == [srk]
        assert captured["lamp"] is None
        assert captured["mfg_date"] == "1961-03"  # data-level fallback
        assert captured["points"] is entry["points"]
        assert captured["lamp_id"] == "A1"
        assert captured["scan_meta"] is entry["data"]
        assert captured["config"] is not None

    def test_separate_unique_paths_and_entry_mfg_priority(
            self, qapp, tmp_path, monkeypatch):
        """Colliding filenames must get _1/_2 suffixes (make_unique_path),
        and entry-level mfg_date must win over the saved-data one."""
        import app.compare_tab as ct
        import app.report as report_mod
        import app.report_options_dialog as rod
        from app.report import SECTION_IDS
        tab = ct.CompareTab()
        monkeypatch.setattr(
            rod, "ask_report_options",
            lambda *a, **k: rod.ReportOptions(sections=set(SECTION_IDS),
                                              language="en"))

        recorded: list = []

        def _fake_gen(**kw):
            recorded.append(kw)
            Path(kw["path"]).write_bytes(b"%PDF stub")

        monkeypatch.setattr(report_mod, "generate_pdf_report", _fake_gen)
        monkeypatch.setattr(
            ct.QFileDialog, "getExistingDirectory",
            staticmethod(lambda *a, **k: str(tmp_path)))
        monkeypatch.setattr(ct.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))
        rendered_pts: list = []
        import app.export_manager as em
        monkeypatch.setattr(
            em, "render_points_pixmap",
            lambda pts, title="": (rendered_pts.append(pts),
                                   QPixmap(10, 10))[1])
        (tmp_path / "EL84_A1.pdf").write_bytes(b"%PDF existing")

        entries = [
            self._entry("A1", entry_mfg="1970-01", data_mfg="1960-01"),
            self._entry("A1", data_mfg="1955-11"),
        ]
        tab._export_pdf_separate(entries)

        paths = [Path(kw["path"]).name for kw in recorded]
        assert paths == ["EL84_A1_1.pdf", "EL84_A1_2.pdf"]
        assert recorded[0]["mfg_date"] == "1970-01"  # entry wins over data
        assert recorded[1]["mfg_date"] == "1955-11"  # data-level fallback
        # dialog choices reach every per-entry generate call
        assert recorded[0]["sections"] == set(SECTION_IDS)
        assert recorded[0]["language"] == "en"
        assert recorded[0]["scan_meta"] is entries[0]["data"]
        # per-lamp plots: each PDF renders ITS OWN points, not a shared
        # screenshot of the whole compare plot
        assert rendered_pts[0] is entries[0]["points"]
        assert rendered_pts[1] is entries[1]["points"]
        assert recorded[0]["plot_image"] is not recorded[1]["plot_image"]
