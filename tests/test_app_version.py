"""Application version — constant, window title, PDF footers.

``t()`` leaves an unfilled ``%{version}`` in the output when a call site
forgets the kwarg, so the template alone proves nothing: every consumer
is pinned at its own call site (the version must appear in the produced
text, and no raw placeholder may survive).

Pins:
- APP_VERSION shape and single-source-of-truth (the literal lives in
  exactly one module);
- every locale carries the ``%{version}`` placeholder in both consuming
  keys (read from the JSON files directly — ``translator_for`` would
  hide a dropped key behind the en fallback);
- window title renders the version;
- all three PDF documents (scan report, matched-tubes certificate,
  amplifier report) render it in their footer.
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from i18n_setup import translator_for
from lm19.version import APP_VERSION

_ROOT = Path(__file__).resolve().parents[1]
_LOCALES_DIR = _ROOT / "locales"

# Keys that interpolate the app version, and where each lives.
_VERSION_KEYS = (("app", "Window_title"), ("report", "Footer"))

# Source trees scanned by the single-source ratchet (tests may name the
# version freely; production code must import it).
_SOURCE_DIRS = ("app", "lm19", "tools")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class TestVersionConstant:

    def test_version_is_dotted_numeric(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+", APP_VERSION), APP_VERSION

    def test_version_literal_declared_once(self):
        """The version string must not be duplicated anywhere else —
        a second copy silently goes stale on the next release bump."""
        owner = _ROOT / "lm19" / "version.py"
        offenders = []
        for sub in _SOURCE_DIRS:
            for path in (_ROOT / sub).rglob("*.py"):
                if path == owner:
                    continue
                if f'"{APP_VERSION}"' in path.read_text(encoding="utf-8"):
                    offenders.append(str(path.relative_to(_ROOT)))
        assert not offenders, (
            f"APP_VERSION literal {APP_VERSION!r} duplicated in: {offenders}")


class TestLocaleTemplates:
    """Placeholder presence is read from the locale FILES: a key dropped
    from one translation would otherwise fall back to en and render a
    correct version in a test while the real locale shows none."""

    @pytest.mark.parametrize(
        "locale_path", sorted(_LOCALES_DIR.glob("*.json")),
        ids=lambda p: p.stem)
    def test_every_locale_interpolates_version(self, locale_path: Path):
        data = json.loads(locale_path.read_text(encoding="utf-8"))
        for section, key in _VERSION_KEYS:
            template = data.get(section, {}).get(key)
            assert template is not None, (locale_path.stem, section, key)
            assert "%{version}" in template, (locale_path.stem, section, key)

    def test_rendered_text_carries_version_in_every_locale(self):
        for locale_path in sorted(_LOCALES_DIR.glob("*.json")):
            tr = translator_for(locale_path.stem)
            for section, key in _VERSION_KEYS:
                text = tr(f"{section}.{key}", version=APP_VERSION,
                          date="2026-01-01", page=1)
                assert APP_VERSION in text, (locale_path.stem, key, text)
                assert "%{" not in text, (locale_path.stem, key, text)


class TestWindowTitle:

    def test_title_shows_version(self, qapp, monkeypatch):
        from app.main_window import MainWindow

        monkeypatch.setattr("app.main_window.list_ports.comports", lambda: [])
        window = MainWindow()
        try:
            title = window.windowTitle()
            assert APP_VERSION in title, title
            assert "%{" not in title, title
        finally:
            window.close()


class TestPdfFooters:
    """One pin per PDF producer: a forgotten ``version=`` kwarg at any of
    the three call sites leaves ``%{version}`` in that document only."""

    def test_scan_report_footer(self, qapp, tmp_path, monkeypatch):
        import app.report as report_mod

        rendered: list = []
        real_translator = report_mod.translator_for

        def _spy_translator(locale: str):
            inner = real_translator(locale)

            def _tr(key: str, **kwargs) -> str:
                text = inner(key, **kwargs)
                if key == "report.Footer":
                    rendered.append(text)
                return text

            return _tr

        monkeypatch.setattr(report_mod, "translator_for", _spy_translator)
        report_mod.generate_pdf_report(
            str(tmp_path / "report.pdf"), "EL84", "L1",
            points=[{"ua": 250.0, "ug1": -7.0, "ia": 48.0}],
        )
        assert rendered, "footer never rendered"
        for text in rendered:
            assert APP_VERSION in text, text
            assert "%{" not in text, text

    def test_certificate_footer(self, qapp, tmp_path, monkeypatch):
        import app.pdf_doc as pdf_doc
        from app.match_certificate import generate_certificate_pdf

        captured: dict = {}
        monkeypatch.setattr(
            pdf_doc, "print_html_pdf",
            lambda path, fragments, images, tail="": captured.update(tail=tail))
        generate_certificate_pdf(str(tmp_path / "cert.pdf"),
                                 fragments=["<p>body</p>"])
        assert APP_VERSION in captured["tail"], captured
        assert "%{" not in captured["tail"], captured

    def test_amp_report_footer(self, qapp, tmp_path, monkeypatch):
        import app.pdf_doc as pdf_doc
        from app.amp_report_pdf import generate_amp_pdf_report

        captured: dict = {}
        monkeypatch.setattr(
            pdf_doc, "print_html_pdf",
            lambda path, fragments, images, tail="": captured.update(tail=tail))
        generate_amp_pdf_report(str(tmp_path / "amp.pdf"), tube_type="EL84",
                                header_lines=["line"], results_html="<p>x</p>",
                                images=[])
        assert APP_VERSION in captured["tail"], captured
        assert "%{" not in captured["tail"], captured
