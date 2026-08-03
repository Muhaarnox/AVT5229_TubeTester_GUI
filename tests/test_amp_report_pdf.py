"""Amplifier-analysis PDF report.

Pins:
- build_amp_header_lines — method visibility (circuit/Ub/Ra/bias/
  HD method from the actual analysis AmpParams, the Ug2 line only
  when filtered);
- render_spectrum_pixmap — spectrum from dist (print width), None
  when < 2 harmonics;
- generate_amp_pdf_report — valid PDF, size grows with content,
  OSError on a missing directory (QTextDocument.print_ is silent);
- generalized ask_report_options: specs=AMP_REPORT_SECTIONS,
  session_key isolation (scan quiet mode does not mute amp);
- ratchet: every render(result) in main_window saves _last_amp_result.
"""

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtGui import QPixmap
from PySide6.QtPdf import QPdfDocument
from PySide6.QtWidgets import QApplication

from app.amp_report_pdf import (
    AMP_REPORT_SECTIONS,
    AMP_SECTION_IDS,
    build_amp_header_lines,
    generate_amp_pdf_report,
    render_spectrum_pixmap,
)
from app.report_options_dialog import (
    ReportOptions,
    ask_report_options,
    reset_session,
)
from lm19.amp_engine import AmpParams
from lm19.app_config import AppConfig
from lm19.amplifier.constants import (
    CIRCUIT_PP,
    HD_METHOD_DFT,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _fresh_session():
    reset_session()
    yield
    reset_session()


class TestAmpHeaderLines:
    def test_params_reach_header_with_method_visibility(self):
        params = AmpParams(ub=300.0, ra=8.0, ug1_bias=-14.7,
                           circuit=CIRCUIT_PP, hd_method=HD_METHOD_DFT, ug2_filter=300.0)
        lines = build_amp_header_lines("L42", params)
        joined = "\n".join(lines)
        assert "Lamp ID: L42" in joined
        assert "Circuit: pp" in joined
        assert "Ub=300V" in joined and "Ra=8kΩ" in joined
        assert "Bias=-14.7V" in joined
        assert "HD: dft" in joined
        assert "Ug2=300V" in joined

    def test_no_ug2_filter_no_ug2_line(self):
        params = AmpParams(ug2_filter=None)
        assert "Ug2=" not in "\n".join(build_amp_header_lines("X", params))

    def test_none_params_only_lamp_line(self):
        lines = build_amp_header_lines("X", None)
        assert len(lines) == 1 and "Lamp ID: X" in lines[0]

    def test_data_label_named_in_header(self):
        lines = build_amp_header_lines("X", None,
                                       data_label="EL84 saved L42")
        assert any("Data: EL84 saved L42" in ln for ln in lines)


class TestSpectrumPixmap:
    def test_spectrum_rendered_at_print_width(self, qapp):
        from app.export_manager import _PLOT_EXPORT_WIDTH_PX
        dist = {"hd2": 1.5, "hd3": 0.4, "hd5": 0.1}
        pm = render_spectrum_pixmap(dist)
        assert pm is not None and pm.width() == _PLOT_EXPORT_WIDTH_PX

    def test_single_harmonic_returns_none(self, qapp):
        assert render_spectrum_pixmap({"hd2": 1.0}) is None

    def test_no_harmonics_returns_none(self, qapp):
        assert render_spectrum_pixmap({"thd": 5.0}) is None


def _small_pixmap() -> QPixmap:
    pm = QPixmap(400, 300)
    pm.fill()
    return pm


class TestGenerateAmpPdf:
    def test_golden_path_produces_valid_pdf(self, qapp, tmp_path):
        out = tmp_path / "amp.pdf"
        generate_amp_pdf_report(
            str(out), tube_type="EL84",
            header_lines=["Lamp ID: L1", "Circuit: se"],
            results_html="<b>THD=1.2%</b><br>Pout=2W",
            images=[("THD sweep", _small_pixmap())],
        )
        assert out.read_bytes().startswith(b"%PDF")
        doc = QPdfDocument()
        assert doc.load(str(out)) == QPdfDocument.Error.None_
        assert doc.pageCount() >= 1

    def test_image_actually_embedded(self, qapp, tmp_path):
        """A dropped addResource/img-tag mutation must shrink the file."""
        with_img = tmp_path / "with.pdf"
        without = tmp_path / "without.pdf"
        kwargs = dict(tube_type="EL84", header_lines=["h"],
                      results_html="x")
        generate_amp_pdf_report(str(with_img),
                                images=[("cap", _small_pixmap())], **kwargs)
        generate_amp_pdf_report(str(without), images=[], **kwargs)
        assert with_img.stat().st_size > without.stat().st_size * 1.5

    def test_results_html_included(self, qapp, tmp_path):
        short = tmp_path / "short.pdf"
        long = tmp_path / "long.pdf"
        kwargs = dict(tube_type="EL84", header_lines=[], images=[])
        generate_amp_pdf_report(str(short), results_html="", **kwargs)
        generate_amp_pdf_report(
            str(long),
            results_html="THD line<br>" * 400, **kwargs)
        # 400 lines paginate: the body demonstrably reaches the document
        doc = QPdfDocument()
        assert doc.load(str(long)) == QPdfDocument.Error.None_
        assert doc.pageCount() > 1
        doc2 = QPdfDocument()
        assert doc2.load(str(short)) == QPdfDocument.Error.None_
        assert doc2.pageCount() == 1

    def test_missing_directory_raises_oserror(self, qapp, tmp_path):
        target = tmp_path / "nope" / "amp.pdf"
        with pytest.raises(OSError):
            generate_amp_pdf_report(str(target), tube_type="X",
                                    header_lines=[], results_html="",
                                    images=[])
        assert not target.exists()


class TestAmpSectionsDialogIntegration:
    def test_silent_path_uses_amp_defaults(self, qapp):
        cfg = AppConfig(report_ask=False,
                        report_sections="srk,plot_curves")  # scan CSV ignored
        opts = ask_report_options(
            None, {sid: "" for sid in AMP_SECTION_IDS}, cfg,
            dialog_cls=None, specs=AMP_REPORT_SECTIONS, session_key="amp")
        assert opts.sections == set(AMP_SECTION_IDS)

    def test_silent_path_clips_unavailable(self, qapp):
        cfg = AppConfig(report_ask=False)
        available = {sid: "" for sid in AMP_SECTION_IDS}
        available["amp_plot_pareto"] = "report.Na_no_pareto"
        opts = ask_report_options(
            None, available, cfg,
            dialog_cls=None, specs=AMP_REPORT_SECTIONS, session_key="amp")
        assert "amp_plot_pareto" not in opts.sections

    def test_session_keys_are_isolated(self, qapp):
        """Scan-session "don't ask" must not silence the amp dialog."""

        class _ScanFake:
            def __init__(self, parent, **kw):
                pass

            def exec(self):
                from PySide6.QtWidgets import QDialog
                return QDialog.DialogCode.Accepted

            def selected_sections(self):
                return {"srk"}

            def language(self):
                return "en"

            def dont_ask(self):
                return True  # silences the SCAN session only

        cfg = AppConfig()
        ask_report_options(None, {}, cfg, dialog_cls=_ScanFake,
                           session_key="scan")

        amp_dialog_opened = []

        class _AmpFake(_ScanFake):
            def __init__(self, parent, **kw):
                amp_dialog_opened.append(True)

            def dont_ask(self):
                return False

        ask_report_options(None, {}, cfg, dialog_cls=_AmpFake,
                           specs=AMP_REPORT_SECTIONS, session_key="amp")
        assert amp_dialog_opened, "amp dialog silenced by scan session"


class TestVerifyTableHtml:
    @staticmethod
    def _result(n_runs: int = 1, imd=None, warnings=()):
        from lm19.ltspice_verify import VerifyResult, VerifyRun
        runs = [VerifyRun(half_swing=9.0 * (i + 1) / n_runs, thd_pct=10.0,
                          hd_pct={2: 1.0, 3: 2.0}, pout_fund_mw=1000.0,
                          ia_avg_ma=30.0)
                for i in range(n_runs)]
        # synthetic marker path — never touches the disk; the pin below
        # only checks that the workdir string REACHES the HTML table
        return VerifyResult(runs=runs, imd=imd, basis="LTspice/koren fit",
                            fit_rms_ma=0.42, warnings=list(warnings),
                            workdir=r"C:\verify\run42")

    _REF = {"basis": "measurements/chebyshev", "thd": 8.0, "hd2": 1.1,
            "hd3": 1.9, "pout_fund_mw": 900.0, "pout_is_fund": True,
            "imd": {"imd2": 3.0, "imd3": 0.5},
            "sweep_amp": [{"half_swing": 4.5, "thd": 5.5},
                          {"half_swing": 9.0, "thd": 9.5}]}

    def test_both_bases_labelled_and_delta_signed(self):
        from app.amp_report_pdf import build_verify_table_html
        html = build_verify_table_html(self._result(), self._REF)
        assert "measurements/chebyshev" in html
        assert "LTspice/koren fit" in html
        assert "+25%" in html          # THD 10 vs 8 — sim above engine
        assert "+11%" in html          # Pout 1000 vs 900
        assert "0.42" in html          # fit RMS honesty line
        assert r"C:\verify\run42" in html  # manual re-run path visible

    def test_sweep_table_pairs_engine_by_nearest_swing(self):
        from app.amp_report_pdf import build_verify_table_html
        html = build_verify_table_html(self._result(n_runs=2), self._REF)
        assert "4.50" in html and "5.50" in html   # engine THD @ 4.5 V
        assert "9.00" in html and "9.50" in html

    def test_imd_block_carries_caveat(self):
        from app.amp_report_pdf import build_verify_table_html
        from i18n_setup import t
        html = build_verify_table_html(
            self._result(imd={"imd2": 24.3, "imd3": 24.9}), self._REF)
        assert "24.3" in html and "3.0" in html    # sim and engine IMD2
        assert t("report.Verify_imd_caveat") in html

    def test_warnings_visible_and_empty_result_empty(self):
        from app.amp_report_pdf import build_verify_table_html
        html = build_verify_table_html(self._result(warnings=["cancelled"]),
                                       {})
        assert "cancelled" in html and "⚠" in html
        from lm19.ltspice_verify import VerifyResult
        assert build_verify_table_html(VerifyResult(), {}) == ""


class TestLastAmpResultRatchet:
    def test_every_render_callsite_stores_last_result(self):
        """Each ``self.amplifier_tab.render(result)`` in main_window.py
        must be preceded (≤3 lines) by storing ``_last_amp_result`` —
        the amp PDF export reads it."""
        src = (Path(__file__).resolve().parents[1]
               / "app" / "main_window.py").read_text(encoding="utf-8")
        lines = src.splitlines()
        sites = [i for i, l in enumerate(lines)
                 if "self.amplifier_tab.render(result)" in l]
        assert sites, "render call-sites vanished — rewire the amp PDF export"
        for i in sites:
            window = "\n".join(lines[max(0, i - 3):i])
            assert "_last_amp_result" in window, (
                f"main_window.py:{i + 1}: analysis result not stored "
                "for the amp PDF report")
