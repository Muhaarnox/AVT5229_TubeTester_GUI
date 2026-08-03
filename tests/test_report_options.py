"""PDF reports: sections, language, options dialog, print render.

Pins:
- section model: source-of-truth enumeration REPORT_SECTIONS gates
  build_report_lines (each section has an on/off twin);
- config CSV parsing (unknown tokens dropped loudly);
- language: locales/*.json auto-discovery (a dropped-in file appears
  without code changes), translator isolation from the global t();
- options dialog: grey-out with reason, presets, disabled checkbox can
  never smuggle a section in, session don't-ask flow;
- export_pdf call-site: chosen sections/language reach the generator,
  unavailable sections are clipped even when the config CSV names them;
- render_plot_pixmap: print-resolution export + loud grab() fallback.
"""

import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtGui import QPixmap
from PySide6.QtPdf import QPdfDocument
from PySide6.QtWidgets import QApplication, QDialog, QWidget

import i18n_setup
from i18n_setup import available_locales, t, translator_for
from app.report import (
    BRIEF_SECTION_IDS,
    REPORT_SECTIONS,
    SECTION_IDS,
    build_report_lines,
    generate_pdf_report,
    sections_from_config,
)
from app.report_options_dialog import (
    ReportOptions,
    ReportOptionsDialog,
    ask_report_options,
    reset_session,
)
from lm19.app_config import AppConfig
from lm19.quality import QualityReport
from lm19.constants import (
    TOPOLOGY_PENTODE,
)

_TS = "2026-07-12 10:00:00"


def _localized_report_locales() -> list:
    """Non-default locales that actually localize reports (sentinel:
    report.Title differs from the en rendering). Derived from
    locales/*.json — no language is hardcoded; a locale relying on the
    en fallback for reports is excluded by design (fallback is legal
    and carries nothing to discriminate)."""
    en_title = translator_for("en")("report.Title", tube="X")
    out = [loc for loc in available_locales()
           if loc != "en"
           and translator_for(loc)("report.Title", tube="X") != en_title]
    assert out, "no localized report locale found — pins would vanish"
    return out

# Full data kit: every text section has data → present unless toggled off.
_SCAN_META = {
    "timestamp": "2026-07-10T15:30:00",
    "name": "morning_run",
    "topology": TOPOLOGY_PENTODE,
    "scan": {
        "ua": {"start": 0.0, "stop": 300.0, "step": 10.0},
        "ug1": {"start": -20.0, "stop": 0.0, "step": 2.0},
        "ug2": {"start": 250.0, "stop": 250.0, "step": 0.0},
        "uh": 6.3, "ih": 0.76,
        "ug2_mode": TOPOLOGY_PENTODE,
    },
}
_ANALYSIS = {"hd2": 1.5, "hd3": 0.4, "pout_mw": 1800.0,
             "ua_0": 250.0, "ia_0": 8.0, "ug1_0": -7.0}

# Marker substring per text section — the completeness assert below forces
# this map to grow whenever REPORT_SECTIONS gains a new text section.
_SECTION_MARKERS = {
    "nominal": "Nominal:",
    "scan_settings": "Scan:",
    "srk": "Measured:",
    "quality": "Quality:",
    "distortion": "Distortion:",
}
_PLOT_SECTIONS = {"plot_curves", "plot_transfer"}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _fresh_session():
    reset_session()
    yield
    reset_session()


class _LampStub:
    is_triode = False
    ua, ug1, ug2, ia, uh = 250.0, -2.0, 250.0, 8.0, 6.3
    pa_max, s, r, k = 12.0, 4.5, 50.0, 1500.0


def _full_lines(sections=None, tr=None):
    return build_report_lines(
        "EL84", "L01", _TS, lamp_config=_LampStub(),
        points=[{"ua": 1.0}], srk={"s": 11.0, "r": 38.0, "k": 1500.0},
        quality=QualityReport(98.0, 102.0, 95.0, "Good"),
        analysis=_ANALYSIS, mfg_date="1962-05",
        sections=sections, scan_meta=_SCAN_META, tr=tr,
    )


def _text(lines) -> str:
    return "\n".join(txt for _, txt in lines)


# ----------------------------------------------------------------------
# Section model
# ----------------------------------------------------------------------


class TestSectionModel:
    def test_marker_map_covers_every_text_section(self):
        """Ratchet: adding a section to REPORT_SECTIONS must extend the
        marker map (and thereby the on/off twins below)."""
        text_sids = SECTION_IDS - _PLOT_SECTIONS
        assert set(_SECTION_MARKERS) == text_sids

    def test_section_ids_unique(self):
        sids = [s.sid for s in REPORT_SECTIONS]
        assert len(sids) == len(set(sids))

    def test_brief_preset_is_subset(self):
        assert BRIEF_SECTION_IDS < SECTION_IDS

    @pytest.mark.parametrize("sid", sorted(_SECTION_MARKERS))
    def test_section_present_when_enabled(self, sid):
        assert _SECTION_MARKERS[sid] in _text(_full_lines(sections=None))

    @pytest.mark.parametrize("sid", sorted(_SECTION_MARKERS))
    def test_section_absent_when_disabled(self, sid):
        secs = set(SECTION_IDS) - {sid}
        assert _SECTION_MARKERS[sid] not in _text(_full_lines(sections=secs))

    def test_sections_from_config_defaults(self):
        assert sections_from_config("") == {
            s.sid for s in REPORT_SECTIONS if s.default_on}

    def test_sections_from_config_csv_subset(self):
        assert sections_from_config(" srk, plot_curves ") == {
            "srk", "plot_curves"}

    def test_sections_from_config_unknown_dropped_loudly(self, caplog):
        with caplog.at_level(logging.WARNING, logger="app.report"):
            out = sections_from_config("srk,bogus_section")
        assert out == {"srk"}
        assert any("bogus_section" in r.message for r in caplog.records)


class TestSrkUncertaintyAndVerdict:
    def test_uncertainty_suffixes_per_value(self):
        """Asymmetric ± per metric — a swapped key would show up."""
        lines = build_report_lines(
            "EL84", "X", _TS,
            srk={"s": 11.0, "r": 38.0, "k": 1500.0,
                 "uncertainty": {"s": 0.02, "r": 0.031, "k": 0.04}})
        txt = _text(lines)
        assert "S=11.00±2%" in txt
        assert "R=38.00±3%" in txt
        assert "K=1500.00±4%" in txt

    def test_no_uncertainty_no_suffix(self):
        txt = _text(build_report_lines(
            "EL84", "X", _TS, srk={"s": 11.0, "r": 38.0, "k": 1500.0}))
        assert "±" not in txt

    @pytest.mark.parametrize("locale", _localized_report_locales())
    def test_verdict_translated_in_localized_report(self, locale):
        q = QualityReport(98.0, 102.0, None, "Good")
        tr = translator_for(locale)
        txt = _text(build_report_lines("EL84", "X", _TS, quality=q,
                                       tr=tr))
        # Expected rendering comes from the locale file, not literals.
        verdict_loc = tr("report.Verdict_Good")
        assert verdict_loc != "report.Verdict_Good"     # key resolved
        # Split on either paren form: CJK locales use fullwidth brackets.
        quality_prefix = re.split(r"[(（]", tr(
            "report.Quality", verdict=verdict_loc,
            ia="", s=""))[0].rstrip()
        assert quality_prefix in txt
        assert "Good" not in txt

    def test_unknown_verdict_stays_visible(self):
        """A new engine verdict without an i18n key must not vanish."""
        q = QualityReport(98.0, 102.0, None, "Mystery")
        txt = _text(build_report_lines("EL84", "X", _TS, quality=q))
        assert "Mystery" in txt


class TestPointsCurvesWidget:
    def test_groups_and_sorted_ua(self, qapp):
        from app.export_manager import _points_curves_widget
        # two Ug1 levels, interleaved and unsorted in Ua on purpose
        pts = [
            {"ua": 200.0, "ug1": -2.0, "ia": 5.0},
            {"ua": 100.0, "ug1": -4.0, "ia": 1.0},
            {"ua": 100.0, "ug1": -2.0, "ia": 3.0},
            {"ua": 200.0, "ug1": -4.0, "ia": 2.0},
        ]
        w = _points_curves_widget(pts, title="L01")
        items = w.plotItem.listDataItems()
        assert len(items) == 2                     # one curve per Ug1
        for item in items:
            x, _y = item.getData()
            assert list(x) == sorted(x)            # Ua ascending

    def test_render_at_print_width(self, qapp):
        from app.export_manager import render_points_pixmap
        pm = render_points_pixmap(
            [{"ua": 1.0, "ug1": -1.0, "ia": 1.0},
             {"ua": 2.0, "ug1": -1.0, "ia": 2.0}])
        from app.export_manager import _PLOT_EXPORT_WIDTH_PX
        assert pm.width() == _PLOT_EXPORT_WIDTH_PX


class TestGroupOverlayWidget:
    def test_one_color_per_lamp_with_legend(self, qapp):
        """UX fix: the certificate plot shows ONLY the group's lamps —
        one color per lamp, all its curves in that color."""
        from app.export_manager import _group_overlay_widget
        members = [
            ("L01", [{"ua": 100.0, "ug1": -2.0, "ia": 3.0},
                     {"ua": 200.0, "ug1": -2.0, "ia": 5.0},
                     {"ua": 100.0, "ug1": -4.0, "ia": 1.0},
                     {"ua": 200.0, "ug1": -4.0, "ia": 2.0}]),
            ("L07", [{"ua": 100.0, "ug1": -2.0, "ia": 2.9},
                     {"ua": 200.0, "ug1": -2.0, "ia": 4.8}]),
        ]
        w = _group_overlay_widget(members)
        items = w.plotItem.listDataItems()
        assert len(items) == 3          # 2 curves lamp A + 1 curve lamp B
        pens = [item.opts["pen"] for item in items]
        assert pens[0] == pens[1]       # lamp A curves share one color
        assert pens[0] != pens[2]       # lamp B differs
        names = [item.opts.get("name") for item in items]
        assert names.count("L01") == 1 and names.count("L07") == 1


class TestScanSettingsContent:
    def test_ranges_and_heater(self):
        txt = _text(_full_lines())
        assert "morning_run" in txt and "2026-07-10T15:30:00" in txt
        assert "Ua 0…300/10" in txt
        assert "Ug1 -20…0/2" in txt
        assert "Ug2 250…250/0" in txt          # pentode mode → Ug2 part
        assert "Uh=6.3" in txt and "Ih=0.76" in txt

    def test_track_mode_has_no_ug2_part(self):
        meta = json.loads(json.dumps(_SCAN_META))
        meta["scan"]["ug2_mode"] = "triode_connected"
        lines = build_report_lines("EL84", "X", _TS,
                                   scan_meta=meta, sections=None)
        txt = _text(lines)
        assert "triode_connected" in txt
        assert "Ug2 250" not in txt


# ----------------------------------------------------------------------
# Language
# ----------------------------------------------------------------------


class TestReportLanguage:
    def test_available_locales_discovers_shipped_files(self):
        locs = available_locales()
        assert "en" in locs              # default locale must ship
        assert len(locs) >= 2            # at least one translation ships

    def test_new_locale_file_is_picked_up_automatically(self, tmp_path,
                                                        monkeypatch):
        """Dropping xx.json into locales/ must surface the language with
        zero code changes (config-only language contract)."""
        src = i18n_setup._LOCALES_DIR
        shutil.copy(src / "en.json", tmp_path / "en.json")
        (tmp_path / "xx.json").write_text(
            json.dumps({"report": {"Title": "XX-Report: %{tube}"}}),
            encoding="utf-8")
        monkeypatch.setattr(i18n_setup, "_LOCALES_DIR", tmp_path)
        i18n_setup._locale_data.cache_clear()
        try:
            assert "xx" in available_locales()
            tr = translator_for("xx")
            assert tr("report.Title", tube="EL84") == "XX-Report: EL84"
            # missing keys fall back to English
            assert tr("report.Scan_points", n=3) == "Scan points: 3"
        finally:
            i18n_setup._locale_data.cache_clear()

    @pytest.mark.parametrize("locale", _localized_report_locales())
    def test_localized_report_lines(self, locale):
        tr = translator_for(locale)
        txt = _text(_full_lines(tr=tr))
        # Expected renderings come from the locale file, not literals.
        title = tr("report.Title", tube="EL84")
        assert title != "report.Title" and title in txt
        measured_prefix = tr("report.Measured", s="11.00", r="",
                             k="").split("    ")[0]
        assert measured_prefix in txt
        # Split on either paren form: CJK locales use fullwidth brackets.
        quality_prefix = re.split(r"[(（]", tr(
            "report.Quality", verdict=tr("report.Verdict_Good"),
            ia="", s=""))[0].rstrip()
        assert quality_prefix in txt        # verdict translated

    @pytest.mark.parametrize("locale", _localized_report_locales())
    def test_translator_does_not_touch_global_t(self, locale):
        translator_for(locale)("report.Title", tube="X")
        assert t("report.Title", tube="Y") == "Tube Test Report: Y"

    @pytest.mark.parametrize("locale", _localized_report_locales())
    def test_locale_override_scoped_and_restored(self, locale):
        from i18n_setup import locale_override
        loc_title = translator_for(locale)("report.Title", tube="X")
        assert loc_title != t("report.Title", tube="X")  # differs from en
        with locale_override(locale):
            assert t("report.Title", tube="X") == loc_title
        assert t("report.Title", tube="X").startswith("Tube Test")

    @pytest.mark.parametrize("locale", _localized_report_locales())
    def test_locale_override_restores_on_exception(self, locale):
        from i18n_setup import locale_override
        with pytest.raises(RuntimeError):
            with locale_override(locale):
                raise RuntimeError("boom")
        assert t("report.Title", tube="X").startswith("Tube Test")


# ----------------------------------------------------------------------
# Options dialog
# ----------------------------------------------------------------------


def _dlg(qapp, available=None, sections=None):
    return ReportOptionsDialog(
        None,
        available=available or {},
        sections=SECTION_IDS if sections is None else sections,
    )


class TestReportOptionsDialog:
    def test_unavailable_section_disabled_with_reason(self, qapp):
        dlg = _dlg(qapp, available={"distortion": "report.Na_no_analysis"})
        cb = dlg._boxes["distortion"]
        assert not cb.isEnabled() and not cb.isChecked()
        assert cb.toolTip() == t("report.Na_no_analysis")

    def test_disabled_checkbox_cannot_smuggle_section(self, qapp):
        dlg = _dlg(qapp, available={"distortion": "report.Na_no_analysis"})
        dlg._boxes["distortion"].setChecked(True)  # bypass the UI guard
        assert "distortion" not in dlg.selected_sections()

    def test_presets(self, qapp):
        dlg = _dlg(qapp, sections=set())
        for btn_ids, expected in (
            ({s.sid for s in REPORT_SECTIONS}, SECTION_IDS),
            (set(BRIEF_SECTION_IDS), BRIEF_SECTION_IDS),
        ):
            dlg._apply_preset(btn_ids)
            assert dlg.selected_sections() == set(expected)

    def test_preset_skips_disabled_boxes(self, qapp):
        dlg = _dlg(qapp, available={"srk": "report.Na_no_srk"},
                   sections=set())
        dlg._apply_preset({s.sid for s in REPORT_SECTIONS})
        assert "srk" not in dlg.selected_sections()

    def test_no_language_selector_in_ui(self, qapp):
        """Contract: the report language comes from the
        config ONLY — the dialog must not offer a selector."""
        from PySide6.QtWidgets import QComboBox
        dlg = _dlg(qapp)
        assert not dlg.findChildren(QComboBox)
        assert not hasattr(dlg, "language_combo")

    def test_every_interactive_widget_has_tooltip(self, qapp):
        from PySide6.QtWidgets import QAbstractButton, QComboBox
        dlg = _dlg(qapp)
        victims = dlg.findChildren(QAbstractButton) + dlg.findChildren(QComboBox)
        missing = [w for w in victims
                   if not w.toolTip() and w.text() not in ("OK", "Cancel")]
        assert not missing, [w.text() for w in missing]


class _FakeDialog:
    """Duck-typed stand-in capturing ask_report_options wiring."""

    accepted = True
    result_sections = {"srk"}
    result_dont_ask = False
    instances: list = []

    def __init__(self, parent, *, available, sections, specs=None):
        self.available = available
        self.init_sections = sections
        self.specs = specs
        type(self).instances.append(self)

    def exec(self):
        return (QDialog.DialogCode.Accepted if type(self).accepted
                else QDialog.DialogCode.Rejected)

    def selected_sections(self):
        return set(type(self).result_sections)

    def dont_ask(self):
        return type(self).result_dont_ask


@pytest.fixture()
def fake_dialog():
    _FakeDialog.accepted = True
    _FakeDialog.result_sections = {"srk"}
    _FakeDialog.result_dont_ask = False
    _FakeDialog.instances = []
    return _FakeDialog


class TestAskReportOptions:
    def test_config_ask_false_skips_dialog_and_clips(self, qapp):
        cfg = AppConfig(report_ask=False,
                        report_sections="srk,distortion,plot_curves")
        opts = ask_report_options(
            None, {"distortion": "report.Na_no_analysis"}, cfg,
            dialog_cls=None)  # any instantiation would raise
        assert opts == ReportOptions(sections={"srk", "plot_curves"},
                                     language="en")

    def test_dialog_result_remembered_for_session(self, qapp, fake_dialog):
        loc = _localized_report_locales()[0]
        cfg = AppConfig(report_language=loc)
        first = ask_report_options(None, {}, cfg, dialog_cls=fake_dialog)
        # language comes from the CONFIG, never from the dialog
        assert first.sections == {"srk"} and first.language == loc
        second = ask_report_options(None, {}, cfg, dialog_cls=fake_dialog)
        assert second.sections == {"srk"}
        # dialog opened both times (don't-ask not set), preseeded from session
        assert len(fake_dialog.instances) == 2
        assert fake_dialog.instances[1].init_sections == {"srk"}

    def test_dont_ask_makes_next_export_silent(self, qapp, fake_dialog):
        fake_dialog.result_dont_ask = True
        cfg = AppConfig()
        ask_report_options(None, {}, cfg, dialog_cls=fake_dialog)
        opts = ask_report_options(None, {}, cfg, dialog_cls=None)
        assert opts.sections == {"srk"} and opts.language == "en"
        assert len(fake_dialog.instances) == 1


class TestResolveReportLanguage:
    def test_configured_language_used(self):
        from app.report_options_dialog import resolve_report_language
        loc = _localized_report_locales()[0]
        assert resolve_report_language(AppConfig(report_language=loc)) == loc

    def test_empty_config_defaults_to_english(self, caplog):
        from app.report_options_dialog import resolve_report_language
        with caplog.at_level(logging.WARNING,
                             logger="app.report_options_dialog"):
            assert resolve_report_language(AppConfig()) == "en"
        assert not caplog.records          # documented default — no noise

    def test_unknown_language_falls_back_loudly(self, caplog):
        """Contract: a config typo → English + log WARNING."""
        from app.report_options_dialog import resolve_report_language
        with caplog.at_level(logging.WARNING,
                             logger="app.report_options_dialog"):
            out = resolve_report_language(AppConfig(report_language="xx_np"))
        assert out == "en"
        assert any("xx_np" in r.message for r in caplog.records)

    def test_cancel_returns_none_and_keeps_session(self, qapp, fake_dialog):
        fake_dialog.accepted = False
        cfg = AppConfig(report_sections="quality")
        assert ask_report_options(None, {}, cfg,
                                  dialog_cls=fake_dialog) is None
        fake_dialog.accepted = True
        nxt = ask_report_options(None, {}, cfg, dialog_cls=fake_dialog)
        assert fake_dialog.instances[-1].init_sections == {"quality"}
        assert nxt is not None


# ----------------------------------------------------------------------
# Call-site: chosen options must reach the generator
# ----------------------------------------------------------------------


class TestExportPdfOptionsCallsite:
    def test_sections_and_language_reach_generator(self, qapp, tmp_path,
                                                   monkeypatch):
        import app.export_manager as em
        import app.report as report_mod

        captured: dict = {}
        monkeypatch.setattr(report_mod, "generate_pdf_report",
                            lambda **kw: captured.update(kw))
        monkeypatch.setattr(
            em.QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (str(tmp_path / "o.pdf"), "")))
        monkeypatch.setattr(em.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))

        doc_locale = _localized_report_locales()[0]
        cfg = AppConfig(report_ask=False, report_language=doc_locale,
                        report_sections="srk,distortion,plot_curves")
        em.export_pdf(
            parent=None, points=[{"ua": 1.0}], tube_type="EL84",
            lamp_id="X", lamp=None, srk_results=[{"s": 1.0, "r": 2.0,
                                                  "k": 3.0}],
            plot_renderer=None,  # analysis unavailable
            plot_widget=QWidget(), config=cfg, scan_meta=None,
        )
        # distortion clipped by availability despite the config CSV
        assert captured["sections"] == {"srk", "plot_curves"}
        assert captured["language"] == doc_locale
        assert captured["transfer_image"] is None

    def test_cancelled_options_dialog_aborts_export(self, qapp, monkeypatch):
        import app.export_manager as em
        import app.report_options_dialog as rod

        monkeypatch.setattr(rod, "ask_report_options",
                            lambda *a, **k: None)
        saves = []
        monkeypatch.setattr(
            em.QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: saves.append(a) or ("", "")))
        em.export_pdf(parent=None, points=[{"ua": 1.0}], tube_type="EL84",
                      lamp_id="X", lamp=None, srk_results=[],
                      plot_renderer=None, plot_widget=QWidget(),
                      config=AppConfig())
        assert not saves, "file dialog shown after options cancel"


# ----------------------------------------------------------------------
# Image gating + print-resolution render
# ----------------------------------------------------------------------


def _tall_pixmap() -> QPixmap:
    pm = QPixmap(300, 2400)
    pm.fill()
    return pm


class TestImageSections:
    def test_transfer_section_off_drops_second_page(self, qapp, tmp_path):
        for sections, pages in ((SECTION_IDS, 2),
                                (SECTION_IDS - {"plot_transfer"}, 1)):
            out = tmp_path / f"img_{pages}.pdf"
            generate_pdf_report(
                path=str(out), tube_type="EL84", lamp_id="X",
                plot_image=_tall_pixmap(), transfer_image=_tall_pixmap(),
                sections=set(sections),
            )
            doc = QPdfDocument()
            assert doc.load(str(out)) == QPdfDocument.Error.None_
            assert doc.pageCount() == pages


class TestScanMetaWiringRatchet:
    def test_scan_flow_stores_last_scan_meta(self):
        """Each ``scan_meta = self._scan_meta_for_save()`` site in the
        scan flow must immediately store ``self._last_scan_meta`` — the
        PDF scan-settings section reads it at export time.

        The save paths go through the start-of-run snapshot, never
        ``_build_scan_metadata()`` directly: the scan controls stay live
        during a run, so a fresh read would record the run armed next.
        """
        src = (Path(__file__).resolve().parents[1]
               / "app" / "main_window_scan.py").read_text(encoding="utf-8")
        lines = src.splitlines()
        assert "self._build_scan_metadata()" not in src, (
            "scan flow must take metadata from the start-of-run snapshot "
            "(_scan_meta_for_save), not rebuild it at save time")
        sites = [i for i, l in enumerate(lines)
                 if "self._scan_meta_for_save()" in l and "scan_meta =" in l]
        assert sites, "scan-metadata call-sites vanished — rewire the PDF section"
        for i in sites:
            nxt = next(l for l in lines[i + 1:] if l.strip())
            assert "_last_scan_meta" in nxt, (
                f"main_window_scan.py:{i + 1}: scan_meta not stored "
                "for the PDF scan-settings section")


class TestRenderPlotPixmap:
    def test_plotwidget_renders_at_print_width(self, qapp):
        import pyqtgraph as pg
        from app.export_manager import _PLOT_EXPORT_WIDTH_PX, render_plot_pixmap
        w = pg.PlotWidget()
        w.resize(400, 300)
        w.plotItem.plot([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        pm = render_plot_pixmap(w)
        assert pm.width() == _PLOT_EXPORT_WIDTH_PX

    def test_non_plot_widget_falls_back_loudly(self, qapp, caplog):
        from app.export_manager import render_plot_pixmap
        w = QWidget()
        w.resize(120, 90)
        with caplog.at_level(logging.WARNING, logger="app.export_manager"):
            pm = render_plot_pixmap(w)
        assert not pm.isNull() and pm.width() != 1600
        assert any("falling back to grab" in r.message
                   for r in caplog.records)
