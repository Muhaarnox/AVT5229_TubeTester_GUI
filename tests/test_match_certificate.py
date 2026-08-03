"""Matched pair/quad certificate (PDF plan).

Pins:
- health fragments: header/conditions/metrics table from TubeRecords
  (non-symmetric data — a column swap is caught), section gating
  (on/off twins), localized rendering via translator_for;
- compare fragments: match % = 100-delta, quality tier by amp_class,
  lamp rows from entries by _index, pairwise table for a quad;
- pick_match_group: no groups >=2 -> visible warning (None), one
  group -> no dialog, dialog cancel -> None;
- generate_certificate_pdf: valid PDF, OSError on a missing
  directory (QTextDocument.print_ is silent);
- UI: buttons exist and are wired to slots (HealthTab/CompareTab),
  Match-panel button enablement follows the result.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtPdf import QPdfDocument
from PySide6.QtWidgets import QApplication

from app.pdf_doc import print_html_pdf
from app.match_certificate import (
    CERT_SECTION_IDS,
    CERT_SECTIONS,
    build_compare_cert_fragments,
    build_health_cert_fragments,
    generate_certificate_pdf,
    pick_match_group,
)
from i18n_setup import available_locales, translator_for
from lm19.tube_matching import (
    CurveDistanceInfo,
    MatchGroup,
    MatchResult,
    TubeRecord,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
)


def _localized_cert_locales() -> list:
    """Non-default locales that actually localize certificates
    (sentinel: report.Cert_title differs from the en rendering).
    Derived from locales/*.json — no language is hardcoded."""
    en_title = translator_for("en")("report.Cert_title", tube="X")
    out = [loc for loc in available_locales()
           if loc != "en"
           and translator_for(loc)("report.Cert_title",
                                   tube="X") != en_title]
    assert out, "no localized cert locale found — pin would vanish"
    return out


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def QPixmap_stub():
    from PySide6.QtGui import QPixmap
    pm = QPixmap(8, 8)
    pm.fill()
    return pm


def _health_group() -> MatchGroup:
    """Asymmetric pair: distinct ia/s/r per record pin column wiring."""
    # conditions use the REAL health-measurement key: ug2_mode (lm19/health.py)
    entry_a = {"tube_type": "EL84", "mfg_date": "1962-05",
               "conditions": {"ua": 250.0, "ug1": -7.0, "ug2": 250.0,
                              "an": 1, "ug2_mode": TOPOLOGY_PENTODE}}
    entry_b = {"tube_type": "EL84", "mfg_date": "",
               "conditions": {"ua": 250.0, "ug1": -7.0, "ug2": 250.0,
                              "an": 2, "ug2_mode": TOPOLOGY_PENTODE}}
    rec_a = TubeRecord(lamp_id="L01", timestamp="2026-07-10T10:00:00",
                       an=1, ia=36.1, s=11.3, r=38.0, index=92.0,
                       entry=entry_a)
    rec_b = TubeRecord(lamp_id="L07", timestamp="2026-07-11T09:00:00",
                       an=2, ia=35.2, s=11.0, r=40.0, index=88.0,
                       entry=entry_b)
    return MatchGroup(number=3, records=[rec_a, rec_b], delta=0.42)


def _text(fragments) -> str:
    return "\n".join(fragments)


class TestHealthCertFragments:
    def test_header_and_metrics_columns(self):
        txt = _text(build_health_cert_fragments(
            _health_group(), tube_type="EL84"))
        assert "Matched Tubes Certificate: EL84" in txt
        assert "Group 3, 2 tubes" in txt and "0.42" in txt
        # both lamps, values in the right cells (asymmetric data)
        assert "L01" in txt and "L07" in txt
        assert "36.1" in txt and "11.30" in txt and "38.00" in txt
        assert "35.2" in txt and "11.00" in txt and "40.00" in txt
        assert "1962-05" in txt          # mfg present for A, dash for B
        assert "2026-07-10T10:00:00" in txt

    def test_conditions_section_twin(self):
        on = _text(build_health_cert_fragments(
            _health_group(), tube_type="EL84"))
        assert "Ua=250V" in on and "Ug1=-7V" in on and "pentode" in on
        off = _text(build_health_cert_fragments(
            _health_group(), tube_type="EL84",
            sections=CERT_SECTION_IDS - {"cert_conditions"}))
        assert "Ua=250V" not in off

    def test_metrics_section_twin(self):
        off = _text(build_health_cert_fragments(
            _health_group(), tube_type="EL84",
            sections=CERT_SECTION_IDS - {"cert_metrics"}))
        assert "36.1" not in off

    def test_shared_group_prints_iq_imbalance(self):
        group = _health_group()
        group.iq_imbalance_ma = 4.5
        txt = _text(build_health_cert_fragments(group, tube_type="EL84"))
        expected = translator_for("en")("report.Cert_iq_imbalance", ma="4.5")
        assert expected != "report.Cert_iq_imbalance" and expected in txt

    def test_strict_group_has_no_iq_line(self):
        # Default group (iq_imbalance_ma=None) must not fabricate the line.
        txt = _text(build_health_cert_fragments(
            _health_group(), tube_type="EL84"))
        assert "δIq" not in txt

    def test_iq_line_rides_the_metrics_section(self):
        group = _health_group()
        group.iq_imbalance_ma = 4.5
        off = _text(build_health_cert_fragments(
            group, tube_type="EL84",
            sections=CERT_SECTION_IDS - {"cert_metrics"}))
        assert "δIq" not in off

    @pytest.mark.parametrize("locale", _localized_cert_locales())
    def test_iq_key_localized(self, locale):
        # Read the locale JSON directly: translator_for would fall back
        # to en and hide a key missing from one locale.
        import json
        from pathlib import Path
        import i18n_setup
        data = json.loads((Path(i18n_setup._LOCALES_DIR)
                           / f"{locale}.json").read_text("utf-8"))
        assert "Cert_iq_imbalance" in data["report"], locale
        assert "%{ma}" in data["report"]["Cert_iq_imbalance"], locale

    @pytest.mark.parametrize("locale", _localized_cert_locales())
    def test_localized_certificate(self, locale):
        tr = translator_for(locale)
        txt = _text(build_health_cert_fragments(
            _health_group(), tube_type="EL84", tr=tr))
        # Expected renderings come from the locale file, not literals.
        title = tr("report.Cert_title", tube="EL84")
        assert title != "report.Cert_title" and title in txt
        group_prefix = tr("report.Cert_group_line", n="3", size="2",
                          delta="", quality="").split("   ")[0]
        assert group_prefix in txt
        assert tr("report.Cert_q_excellent") in txt     # delta=0.42 tier


def _compare_setup():
    entries = [
        {"lamp_type": "EL84", "lamp_id": "A1", "name": "run_a",
         "mfg_date": "1970-01", "timestamp": "2026-07-01T10:00:00",
         "data": {"topology": TOPOLOGY_PENTODE,
                  "scan": {"ug2_mode": TOPOLOGY_PENTODE}}},
        {"lamp_type": "EL84", "lamp_id": "B2", "name": "run_b",
         "mfg_date": "", "timestamp": "2026-07-02T10:00:00",
         "data": {"topology": TOPOLOGY_PENTODE,
                  "scan": {"ug2_mode": TOPOLOGY_PENTODE}}},
        {"lamp_type": "EL84", "lamp_id": "C3", "name": "run_c",
         "mfg_date": "", "timestamp": "2026-07-03T10:00:00",
         "data": {}},
    ]
    pair_info = {
        (0, 1): CurveDistanceInfo(distance=3.0, n_points=40,
                                  low_overlap=False),
        (0, 2): CurveDistanceInfo(distance=4.5, n_points=38,
                                  low_overlap=False),
        (1, 2): CurveDistanceInfo(distance=6.0, n_points=42,
                                  low_overlap=False),
    }
    return entries, pair_info


def _compare_group(indices, delta) -> MatchGroup:
    records = [TubeRecord(lamp_id=f"idx{i}", timestamp="", an=0,
                          ia=0, s=0, r=0, entry={"_index": i})
               for i in indices]
    return MatchGroup(number=1, records=records, delta=delta)


class TestCompareCertFragments:
    def test_match_percent_and_class_tier(self):
        # δ=6.0 discriminates the class: class_b → excellent (≤8),
        # generic Health thresholds → fair (>5) — amp_class must be used
        entries, pair_info = _compare_setup()
        txt = _text(build_compare_cert_fragments(
            _compare_group([0, 1], 6.0), tube_type="EL84",
            entries=entries, pair_info=pair_info, amp_class="class_b"))
        assert "94.0%" in txt                 # 100 − δ
        assert "excellent" in txt             # class_b tier, NOT generic
        assert "[class_b]" in txt             # basis carries the class
        assert "A1" in txt and "B2" in txt and "run_a" in txt
        assert "1970-01" in txt
        assert "pentode" in txt               # scan mode line

    def test_quad_pairwise_table(self):
        entries, pair_info = _compare_setup()
        # shuffled record order: pair keys MUST be (min,max)-normalized
        txt = _text(build_compare_cert_fragments(
            _compare_group([2, 0, 1], 6.0), tube_type="EL84",
            entries=entries, pair_info=pair_info, amp_class="class_ab"))
        assert "Pairwise match:" in txt
        assert "97.0%" in txt and "95.5%" in txt and "94.0%" in txt

    def test_pair_has_no_pairwise_table(self):
        entries, pair_info = _compare_setup()
        txt = _text(build_compare_cert_fragments(
            _compare_group([0, 1], 3.0), tube_type="EL84",
            entries=entries, pair_info=pair_info))
        assert "Pairwise match:" not in txt


class TestPickMatchGroup:
    def test_no_groups_warns_and_returns_none(self, qapp, monkeypatch):
        import app.match_certificate as mc
        warns = []
        monkeypatch.setattr(
            "PySide6.QtWidgets.QMessageBox.warning",
            staticmethod(lambda *a, **k: warns.append(a)))
        result = MatchResult(mode="groups", groups=[], unmatched=[])
        assert mc.pick_match_group(None, result) is None
        assert warns

    def test_single_record_groups_not_offered(self, qapp, monkeypatch):
        """Similar-mode ranking rows (1 record) are not certificates."""
        warns = []
        monkeypatch.setattr(
            "PySide6.QtWidgets.QMessageBox.warning",
            staticmethod(lambda *a, **k: warns.append(a)))
        lone = MatchGroup(number=1, records=[_health_group().records[0]],
                          delta=0.1)
        result = MatchResult(mode="similar", groups=[lone], unmatched=[])
        assert pick_match_group(None, result) is None

    def test_single_group_returned_without_dialog(self, qapp, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("dialog shown for a single group")

        monkeypatch.setattr("PySide6.QtWidgets.QInputDialog.getItem",
                            staticmethod(_boom))
        group = _health_group()
        result = MatchResult(mode="groups", groups=[group], unmatched=[])
        assert pick_match_group(None, result) is group

    def test_dialog_cancel_returns_none(self, qapp, monkeypatch):
        monkeypatch.setattr(
            "PySide6.QtWidgets.QInputDialog.getItem",
            staticmethod(lambda *a, **k: ("", False)))
        g1, g2 = _health_group(), _health_group()
        result = MatchResult(mode="groups", groups=[g1, g2], unmatched=[])
        assert pick_match_group(None, result) is None


class TestGenerateCertificatePdf:
    def test_golden_path(self, qapp, tmp_path):
        out = tmp_path / "cert.pdf"
        generate_certificate_pdf(
            str(out),
            fragments=build_health_cert_fragments(
                _health_group(), tube_type="EL84"))
        assert out.read_bytes().startswith(b"%PDF")
        doc = QPdfDocument()
        assert doc.load(str(out)) == QPdfDocument.Error.None_
        assert doc.pageCount() >= 1

    def test_missing_directory_raises(self, qapp, tmp_path):
        with pytest.raises(OSError):
            generate_certificate_pdf(
                str(tmp_path / "nope" / "c.pdf"),
                fragments=["<p>x</p>"])


class TestPrintHtmlPdf:
    def test_golden_and_missing_dir(self, qapp, tmp_path):
        out = tmp_path / "doc.pdf"
        print_html_pdf(str(out), ["<p>x</p>"], [], tail="<p>f</p>")
        assert out.read_bytes().startswith(b"%PDF")
        with pytest.raises(OSError):
            print_html_pdf(str(tmp_path / "no" / "d.pdf"), ["<p>x</p>"], [])

    def test_tail_is_rendered(self, qapp, tmp_path):
        """A dropped-tail mutation must lose the second page."""
        out = tmp_path / "tailed.pdf"
        print_html_pdf(str(out), ["<p>x</p>"], [],
                       tail="<p>tail line</p>" * 300)
        doc = QPdfDocument()
        assert doc.load(str(out)) == QPdfDocument.Error.None_
        assert doc.pageCount() > 1


class TestCertificateUiWiring:
    def test_health_button_wired(self, qapp, monkeypatch):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_health_tab_logic import _make_health_tab
        tab = _make_health_tab()
        assert tab.match_panel.certificate_btn.toolTip()
        assert not tab.match_panel.certificate_btn.isEnabled()
        # no match result → visible warning through the real slot
        warns = []
        monkeypatch.setattr(
            "PySide6.QtWidgets.QMessageBox.warning",
            staticmethod(lambda *a, **k: warns.append(a)))
        tab.match_panel.certificate_btn.setEnabled(True)
        tab.match_panel.certificate_btn.click()
        assert warns, "certificate click did not reach the slot"

    def test_compare_button_wired(self, qapp, monkeypatch):
        from app.compare_tab import CompareTab
        tab = CompareTab()
        assert tab._match_cert_btn.toolTip()
        warns = []
        monkeypatch.setattr(
            "PySide6.QtWidgets.QMessageBox.warning",
            staticmethod(lambda *a, **k: warns.append(a)))
        tab._match_cert_btn.click()
        assert warns, "certificate click did not reach the slot"

    def test_compare_cert_plot_covers_group_only(self, qapp, tmp_path,
                                                 monkeypatch):
        """UX fix: the certificate plot must NOT leak other checked
        lamps — only the matched group's own curves are rendered."""
        import app.compare_tab as ct
        import app.export_manager as em
        import app.match_certificate as mc
        import app.report_options_dialog as rod

        tab = ct.CompareTab()
        entries, pair_info = _compare_setup()   # 3 entries, group = 0+1
        entries[0]["points"] = [{"ua": 1.0, "ug1": -1.0, "ia": 1.0}]
        entries[1]["points"] = [{"ua": 2.0, "ug1": -1.0, "ia": 2.0}]
        entries[2]["points"] = [{"ua": 3.0, "ug1": -1.0, "ia": 3.0}]
        from lm19.tube_matching import CurveMatchResult
        tab._match_entries = entries
        tab._match_result = CurveMatchResult(
            mode="groups", groups=[_compare_group([0, 1], 3.0)],
            unmatched=[2], pair_info=pair_info)

        monkeypatch.setattr(
            rod, "ask_report_options",
            lambda *a, **k: rod.ReportOptions(
                sections={"cert_metrics", "cert_plot"}, language="en"))
        monkeypatch.setattr(
            ct.QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (str(tmp_path / "c.pdf"), "")))
        monkeypatch.setattr(ct.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))
        rendered = {}
        monkeypatch.setattr(
            em, "render_group_overlay_pixmap",
            lambda members, title="": rendered.update(members=members)
            or QPixmap_stub())
        monkeypatch.setattr(mc, "generate_certificate_pdf",
                            lambda *a, **k: None)

        tab._export_match_certificate()
        assert rendered["members"] == [
            ("A1", entries[0]["points"]),
            ("B2", entries[1]["points"]),
        ]  # C3 is checked/visible but NOT in the group — must be absent

    def test_sections_source_of_truth(self):
        assert {s.sid for s in CERT_SECTIONS} == CERT_SECTION_IDS
        assert len(CERT_SECTIONS) == len(CERT_SECTION_IDS)
