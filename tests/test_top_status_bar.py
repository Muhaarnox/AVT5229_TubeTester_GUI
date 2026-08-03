"""Pins for the single (top) status bar of MainWindow.

The window used to carry two status strips: the custom top bar (LED,
connection text, TX/RX) and a lazily created QMainWindow status bar that
held the warning indicator plus two transient messages. The bottom strip
was blank in normal operation, so the indicator moved up and the two
messages got permanent homes: the model-fit verdict became a warning
category, the settings-load summary became a dialog.

Pins cover the layout contract (nothing left in the bottom bar, no
unnamed widget in the top one), the per-direction byte counters and both
relocated messages.
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QPushButton,
    QStatusBar,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow
from app.main_window_builders import _IO_COUNT_ZERO
from app.main_window_connection import MainWindowConnection, _RX_STALE_S
from i18n_setup import available_locales, t

pytestmark = [pytest.mark.smoke_ui]


# ── Module local helpers ──

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Warning categories are declared at the call sites, so the list of keys
# the dialog needs is derived from the sources rather than retyped here.
_CATEGORY_CALL = re.compile(r"_set_ui_warnings\(\s*\n?\s*[\"'](\w+)[\"']")
_STARTUP_CATEGORY = "startup"
# Fixed clock for the stale-tooltip boundary (real time() would make the
# threshold cases race).
_FROZEN_NOW = 1_700_000_000.0


class _Port:
    def __init__(self, device: str) -> None:
        self.device = device


class _CountersHost(MainWindowConnection):
    """Minimal host wearing the connection mixin's IO-stats slots."""

    def __init__(self) -> None:
        self.status_label = QLabel()
        self.tx_activity_label = QLabel()
        self.rx_activity_label = QLabel()
        self.tx_count_label = QLabel()
        self.rx_count_label = QLabel()
        self.connect_btn = QPushButton()
        self._prev_tx = None
        self._prev_rx = None
        self._emergency_lock = True
        self.led_colors: list = []
        self.locked: list = []

    def _flash_io_activity(self, tx_changed: bool, rx_changed: bool) -> None:
        pass

    def _set_io_activity(self, label: QLabel, active: bool) -> None:
        pass

    def _set_connection_led(self, color: str) -> None:
        self.led_colors.append(color)

    def _set_write_controls_locked(self, locked: bool) -> None:
        self.locked.append(locked)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def host(qapp):
    return _CountersHost()


@pytest.fixture
def frozen_time(monkeypatch):
    monkeypatch.setattr("app.main_window_connection.time.time",
                        lambda: _FROZEN_NOW)
    return _FROZEN_NOW


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(
        "app.main_window.list_ports.comports",
        lambda: [_Port("COM1")],
    )
    w = MainWindow()
    yield w
    w.close()


# ── Layout contract ──────────────────────────────────────────────────

class TestBottomBarIsGone:

    def test_no_qmainwindow_status_bar_exists(self, window):
        """QMainWindow.statusBar() builds the strip on first access, so any
        surviving call materialises the empty bar again."""
        assert window.findChild(QStatusBar) is None


class TestTopBarComposition:

    @staticmethod
    def _layout_widgets(window) -> set:
        layout = window.top_status_bar.layout()
        out = set()
        for i in range(layout.count()):
            w = layout.itemAt(i).widget()
            if w is not None:
                out.add(id(w))
        return out

    def test_top_bar_holds_exactly_the_named_widgets(self, window):
        """Both directions: nothing expected is missing (the indicator did
        not stay behind in the status bar) and nothing unnamed rides in
        the bar."""
        expected = {
            "reset_all_btn", "conn_led", "status_label",
            "tx_activity_label", "tx_count_label",
            "rx_activity_label", "rx_count_label",
            "warning_indicator",
        }
        named = {id(getattr(window, name)) for name in expected}
        assert named == self._layout_widgets(window)

    def test_indicator_is_last_in_the_row(self, window):
        """Permanent-indicator position: after the stretch, at the right
        end — anywhere else and it drifts with the connection text."""
        layout = window.top_status_bar.layout()
        last = layout.itemAt(layout.count() - 1).widget()
        assert last is window.warning_indicator

    def test_indicator_starts_hidden(self, window):
        assert window.warning_indicator.isHidden()

    def test_counters_start_at_zero_without_caption(self, window):
        assert window.tx_count_label.text() == _IO_COUNT_ZERO
        assert window.rx_count_label.text() == _IO_COUNT_ZERO


# ── Byte counters ────────────────────────────────────────────────────

class TestByteCounters:

    def test_each_count_lands_on_its_own_label(self, host):
        """Asymmetric values: swapping the two assignments must be visible."""
        host._on_io_stats(128, 96, None)
        assert host.tx_count_label.text() == "128"
        assert host.rx_count_label.text() == "96"

    def test_counters_carry_digits_only(self, host):
        """The flashing TX/RX indicator is the caption; the counter is not."""
        host._on_io_stats(128, 96, None)
        assert host.tx_count_label.text().isdigit()
        assert host.rx_count_label.text().isdigit()

    def test_tooltip_stays_fresh_without_any_rx(self, host):
        host._on_io_stats(4, 0, None)
        assert host.rx_count_label.toolTip() == t('conn.Rx_bytes')

    def test_tooltip_fresh_just_below_threshold(self, host, frozen_time):
        host._on_io_stats(4, 2, frozen_time - (_RX_STALE_S - 1))
        assert host.rx_count_label.toolTip() == t('conn.Rx_bytes')

    def test_tooltip_stale_at_threshold(self, host, frozen_time):
        """Boundary equality: the age is reported at exactly the threshold."""
        host._on_io_stats(4, 2, frozen_time - _RX_STALE_S)
        assert host.rx_count_label.toolTip() == t(
            'conn.Rx_bytes_stale', delta=_RX_STALE_S)

    def test_tooltip_reports_the_actual_age(self, host, frozen_time):
        host._on_io_stats(4, 2, frozen_time - 42)
        assert host.rx_count_label.toolTip() == t(
            'conn.Rx_bytes_stale', delta=42)

    def test_tooltip_returns_to_fresh_after_traffic_resumes(
            self, host, frozen_time):
        """Negative space: the stale note must not stick once RX is live."""
        host._on_io_stats(4, 2, frozen_time - 30)
        assert host.rx_count_label.toolTip() != t('conn.Rx_bytes')
        host._on_io_stats(5, 3, frozen_time)
        assert host.rx_count_label.toolTip() == t('conn.Rx_bytes')


class TestDisconnectResetsCounters:

    def test_counters_and_tooltip_reset(self, host, frozen_time):
        host._on_io_stats(128, 96, frozen_time - 30)
        host._on_disconnected()
        assert host.tx_count_label.text() == _IO_COUNT_ZERO
        assert host.rx_count_label.text() == _IO_COUNT_ZERO
        assert host.rx_count_label.toolTip() == t('conn.Rx_bytes')


# ── Relocated messages ───────────────────────────────────────────────

class _FakeModelDialog:
    """Stand-in for ModelDialog: accepts immediately, yields no series."""

    verdict = "RMS 0.42 mA, quality good"
    alerts: list = []

    def __init__(self, *args, **kwargs) -> None:
        self.fit_verdict = self.verdict
        self.fit_alerts = list(self.alerts)

    def exec(self) -> int:
        return QDialog.DialogCode.Accepted

    def results_multi(self) -> list:
        return []


class TestFitVerdictReachesIndicator:

    @staticmethod
    def _spy(window, monkeypatch) -> dict:
        seen: dict = {}
        monkeypatch.setattr(
            window, "_set_ui_warnings",
            lambda category, items: seen.__setitem__(category, list(items)))
        return seen

    def test_verdict_is_pushed_to_the_indicator(self, window, monkeypatch):
        """Call-site pin: the dialog's verdict must reach the indicator —
        a unit pin on _set_ui_warnings cannot prove the caller passes it."""
        monkeypatch.setattr("app.model_dialog.ModelDialog", _FakeModelDialog)
        seen = self._spy(window, monkeypatch)
        window._show_model_dialog()
        assert seen["fit_verdict"] == [_FakeModelDialog.verdict]

    def test_empty_verdict_clears_the_category(self, window, monkeypatch):
        """A fit without a verdict must not leave the previous one on
        screen labelled as current."""
        class _NoVerdict(_FakeModelDialog):
            verdict = ""

        monkeypatch.setattr("app.model_dialog.ModelDialog", _NoVerdict)
        seen = self._spy(window, monkeypatch)
        window._show_model_dialog()
        assert seen["fit_verdict"] == []

    def test_alerts_still_reach_the_indicator(self, window, monkeypatch):
        class _WithAlerts(_FakeModelDialog):
            alerts = ["ia_out_of_range"]

        monkeypatch.setattr("app.model_dialog.ModelDialog", _WithAlerts)
        seen = self._spy(window, monkeypatch)
        window._show_model_dialog()
        assert seen["model_fit"] == ["ia_out_of_range"]


class TestWarningCategoriesAreTranslated:

    @staticmethod
    def _categories() -> set:
        found = {_STARTUP_CATEGORY}
        for path in sorted((_PROJECT_ROOT / "app").rglob("*.py")):
            src = path.read_text(encoding="utf-8")
            found.update(_CATEGORY_CALL.findall(src))
        return found

    def test_every_used_category_has_a_key(self):
        """Completeness from the source of truth (the call sites), not from
        a hand-kept list: a raw key renders as 'warnbar.category_x' in the
        dialog and nobody notices."""
        cats = self._categories()
        assert len(cats) >= 4, "category scan found nothing to check"
        missing = [c for c in sorted(cats)
                   if t(f"warnbar.category_{c}") == f"warnbar.category_{c}"]
        assert missing == []

    def test_translated_locales_cover_every_category(self):
        """Locale-neutral: a locale that ships the warnbar section must
        carry every category in it.

        Read from the JSON, NOT through ``translator_for`` — the resolver
        falls back to en, so a missing key still renders as English text
        and a fallback-aware check cannot tell it from a real translation.
        """
        cats = sorted(self._categories())
        checked = []
        for loc in available_locales():
            if loc == "en":
                continue
            data = json.loads(
                (_PROJECT_ROOT / "locales" / f"{loc}.json").read_text(
                    encoding="utf-8"))
            section = data.get("warnbar")
            if not section:
                continue  # locale rides the en fallback for this section
            checked.append(loc)
            missing = [c for c in cats
                       if f"category_{c}" not in section]
            assert missing == [], f"{loc}: {missing}"
        assert checked, "no translated locale exercised"


class TestSettingsLoadUsesDialog:

    def test_load_summary_reaches_a_dialog(self, window, monkeypatch,
                                           tmp_path):
        """Call-site pin: with the status bar gone, the load summary must
        be shown some other way — dropping the call leaves the user with a
        silently partial restore."""
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"_schema_version": 1,
                        "plot": {"line_width": 2.0},
                        "bogus_key": 1}),
            encoding="utf-8")
        monkeypatch.setattr(
            "app.main_window_settings.QFileDialog.getOpenFileName",
            lambda *a, **k: (str(path), ""))
        shown: list = []
        monkeypatch.setattr(
            "app.main_window_settings.QMessageBox.information",
            lambda parent, title, text, *a, **k: shown.append((title, text)))

        window._load_scan_settings()

        assert len(shown) == 1
        title, text = shown[0]
        assert text == t('msg.Settings_loaded', version=1, applied=1,
                         ignored=1)
        assert title == t('menu.Load_scan_settings')
