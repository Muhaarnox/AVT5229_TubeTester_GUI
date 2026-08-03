"""Failure-visibility rule: every failure must reach the user via the UI —
log alone is not a user-facing channel.

Pins the full warning pipeline built for it:
- ``SourceResult.warnings`` population in the amp engine (5 codes),
- the ⚠ block in the results-panel HTML (``format_warnings_html``),
- ``collect_warnings`` (status-bar feed, multi-source prefixes),
- optimizer refine ``warnings_out`` (nm_not_converged / refine_failed),
- the status-bar indicator (``_set_ui_warnings`` / startup cap),
- emergency-stop failed-channel visibility,
- ``HealthProtectionPayload.ug1_restore_failed`` → dialog line,
- fitter warnings → ``ModelFitResult.warnings``,
- i18n keys exist in ALL locales for every warning code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from i18n_setup import available_locales

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import i18n_setup
from lm19.amplifier.constants import (
    CIRCUIT_SE,
    HD_METHOD_DFT,
)
from lm19.amp_engine import (
    WARN_DFT_NO_MODEL,
    WARN_MODEL_FALLBACK,
    WARN_PA_AVG_NOT_CONVERGED,
)
from lm19.amp_engine import (
    ENGINE_WARNING_CODES,
    WARN_UG2_FILTER_EMPTY,
)
from lm19.optimizer import (
    OPT_WARNING_CODES,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
)
from lm19.tube_model_base import (
    MODEL_WARN_REEFMAN_FEW_UG2,
)
from lm19.tube_model_base import (
    MODEL_WARNING_CODES,
)

i18n_setup.setup("en")


# ── Engine: SourceResult.warnings population ─────────────────────────

class TestEngineSourceWarnings:

    @pytest.fixture(scope="class")
    def engine_and_points(self):
        from lm19.amp_engine import AmplifierEngine
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        eng = AmplifierEngine()
        eng.set_data(pts, is_triode=False)
        return eng, pts

    def _params(self, **kw):
        from lm19.amp_engine import AmpParams
        base = dict(ub=300.0, ra=7.0, ug1_bias=-7.0, pa_max=12.0,
                    ug2_filter=250.0)
        base.update(kw)
        return AmpParams(**base)

    def test_clean_run_has_no_warnings(self, engine_and_points):
        eng, _ = engine_and_points
        result = eng.analyze(self._params())
        sr = result.per_source["measurements"]
        assert sr.warnings == []

    def test_empty_ug2_filter_warns(self, engine_and_points):
        eng, _ = engine_and_points
        result = eng.analyze(self._params(ug2_filter=999.0))
        sr = result.per_source["measurements"]
        codes = {w["code"] for w in sr.warnings}
        assert WARN_UG2_FILTER_EMPTY in codes

    def test_dft_without_model_warns(self, engine_and_points):
        eng, _ = engine_and_points
        result = eng.analyze(self._params(hd_method=HD_METHOD_DFT))
        sr = result.per_source["measurements"]
        codes = {w["code"] for w in sr.warnings}
        assert WARN_DFT_NO_MODEL in codes

    def test_model_fallback_warns(self, engine_and_points):
        from lm19.amp_engine import AmpParams
        eng, _ = engine_and_points
        result = eng.analyze(self._params(sources=["koren"]))
        sr = result.per_source["koren"]
        assert sr.model_fallback is True
        codes = {w["code"] for w in sr.warnings}
        assert WARN_MODEL_FALLBACK in codes

    def test_pa_avg_warning_helper(self):
        from lm19.amp_engine import AmplifierEngine, SourceResult
        sr = SourceResult(pa_avg={"pa_avg_mw": 1000.0, "n_not_converged": 3})
        AmplifierEngine._append_pa_avg_warning(sr)
        assert {"code": WARN_PA_AVG_NOT_CONVERGED, "n": 3} in sr.warnings
        sr2 = SourceResult(pa_avg={"pa_avg_mw": 1000.0, "n_not_converged": 0})
        AmplifierEngine._append_pa_avg_warning(sr2)
        assert sr2.warnings == []


# ── Report: ⚠ block + collect_warnings ───────────────────────────────

class TestReportWarningsBlock:

    def test_format_warnings_html_renders_lines(self):
        from app.amplifier_report import format_warnings_html
        lines = format_warnings_html(
            [{"code": WARN_DFT_NO_MODEL},
             {"code": WARN_PA_AVG_NOT_CONVERGED, "n": 3}])
        assert len(lines) == 2
        assert all("⚠" in line for line in lines)

    def test_unknown_code_stays_visible(self):
        from app.amplifier_report import warning_text
        assert warning_text({"code": "brand_new_code"}) == "brand_new_code"

    def test_source_results_html_contains_block(self):
        from app.amplifier_report import format_source_results
        from lm19.amp_engine import SourceResult
        sr = SourceResult(warnings=[{"code": WARN_DFT_NO_MODEL}])
        html = format_source_results(sr, pa_max=12.0, circuit=CIRCUIT_SE)
        assert "⚠" in html

    def test_collect_warnings_prefixes_sources(self, qapp):
        from app.amplifier_tab import AmplifierTab
        from lm19.amp_engine import AnalysisResult, SourceResult
        tab = AmplifierTab()
        result = AnalysisResult()
        result.per_source["measurements"] = SourceResult(
            warnings=[{"code": WARN_DFT_NO_MODEL}])
        result.per_source["koren"] = SourceResult(
            warnings=[{"code": WARN_MODEL_FALLBACK, "source": "koren"}])
        out = tab.collect_warnings(result)
        assert len(out) == 2
        assert any(text.startswith("[measurements]") for text in out)


# ── Optimizer: refine warnings_out ───────────────────────────────────

class TestRefineWarningsOut:

    def _pt(self):
        from lm19.optimizer import OptPoint
        return OptPoint(ub=250.0, ug2=250.0, ug1=-8.0, ra=5.0,
                        thd=1.0, hd2=0.5, hd3=0.2, pout_mw=100.0,
                        pa_mw=1000.0, ia_0=30.0, ua_0=150.0,
                        amp_class="A", max_swing=4.0, half_swing=2.0)

    def test_refine_failed_recorded(self, monkeypatch):
        import lm19.optimizer as opt
        from lm19.tube_sim import quick_pentode
        monkeypatch.setattr(
            "scipy.optimize.minimize",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("bad x0")))
        model, pts = quick_pentode("EL84")
        warnings_out: list = []
        r = opt.refine_optimum(self._pt(), pts, model,
                               opt.OptimizerConstraints(),
                               warnings_out=warnings_out)
        assert r is None
        assert warnings_out == ["refine_failed"]

    def test_nm_not_converged_recorded(self, monkeypatch):
        import lm19.optimizer as opt
        from lm19.tube_sim import quick_pentode

        class FakeResult:
            success = False
            fun = 1.0  # finite score, below PENALTY_SCORE
            nit = 400
            x = np.array([250.0, 250.0, -8.0, 5.0, 2.0])

        monkeypatch.setattr("scipy.optimize.minimize",
                            lambda *a, **k: FakeResult())
        model, pts = quick_pentode("EL84")
        warnings_out: list = []
        opt.refine_optimum(self._pt(), pts, model,
                           opt.OptimizerConstraints(),
                           warnings_out=warnings_out)
        assert warnings_out == ["nm_not_converged"]


# ── Status-bar indicator ─────────────────────────────────────────────

class TestWarningIndicator:

    @pytest.fixture()
    def host(self, qapp):
        """Minimal QMainWindow wearing the builders-mixin indicator API."""
        from PySide6.QtWidgets import QMainWindow
        from app.main_window_builders import MainWindowBuilders

        class Host(MainWindowBuilders, QMainWindow):
            pass

        w = Host()
        w._build_warning_indicator()
        return w

    def test_hidden_when_empty(self, host):
        host._set_ui_warnings("analysis", [])
        assert host.warning_indicator.isHidden()

    def test_counts_across_categories(self, host):
        host._set_ui_warnings("analysis", ["a", "b"])
        host._set_ui_warnings("optimizer", ["c"])
        assert host.warning_indicator.text() == "⚠ 3"
        assert not host.warning_indicator.isHidden()
        # replacing a category recounts; clearing both hides
        host._set_ui_warnings("analysis", [])
        assert host.warning_indicator.text() == "⚠ 1"
        host._set_ui_warnings("optimizer", [])
        assert host.warning_indicator.isHidden()

    def test_startup_cap(self, host):
        from app.main_window_builders import _STARTUP_WARNINGS_MAX
        msgs = [f"warn {i}" for i in range(_STARTUP_WARNINGS_MAX + 10)]
        host.set_startup_warnings(msgs)
        shown = host._ui_warnings["startup"]
        assert len(shown) == _STARTUP_WARNINGS_MAX + 1  # + "…and N more"


# ── Emergency stop: failed channels reach the operator ───────────────

class TestEmergencyZeroVisibility:

    def test_returns_failed_channels(self, qapp):
        from serial import SerialException
        from app.main_window_connection import MainWindowConnection

        class Host(MainWindowConnection):
            def __init__(self):
                self.client = MagicMock()
                self.client.is_open.return_value = True
                self.app_config = MagicMock(ug1_after_stop=-40.0)

        host = Host()

        def set_param(name, value):
            if name == "Ua":
                raise SerialException("port gone")

        host.client.set_param.side_effect = set_param
        failed = host._emergency_zero_outputs()
        assert failed == ["Ua"]

    def test_all_ok_returns_empty(self, qapp):
        from app.main_window_connection import MainWindowConnection

        class Host(MainWindowConnection):
            def __init__(self):
                self.client = MagicMock()
                self.client.is_open.return_value = True
                self.app_config = MagicMock(ug1_after_stop=-40.0)

        assert Host()._emergency_zero_outputs() == []


# ── Health: restore-failed flag reaches the dialog ───────────────────

def _make_payload(**kw):
    from lm19.scan.exceptions import HealthProtectionPayload
    base = dict(kind="pa", ua=250.0, ug1=-7.0, ug2=250.0, ia_ma=60.0,
                ig2_ma=5.0, measured_w=15.0, limit_w=12.0,
                datasheet_max_w=12.0, safety_pct=100.0, step_idx=3,
                total_steps=5, start_ug1=-40.0, target_ug1=-7.0,
                tube_type="EL84", lamp_id="L1", topology=TOPOLOGY_PENTODE,
                ug2_mode=TOPOLOGY_PENTODE)
    base.update(kw)
    return HealthProtectionPayload(**base)


class TestHealthRestoreFlag:

    def test_payload_default_false(self):
        assert _make_payload().ug1_restore_failed is False

    def test_dialog_shows_restore_failure_first(self, qapp, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        import app.health_protection_dialog as hpd
        captured = {}
        monkeypatch.setattr(QMessageBox, "exec", lambda self: None)
        orig = QMessageBox.setInformativeText
        monkeypatch.setattr(
            QMessageBox, "setInformativeText",
            lambda self, text: captured.setdefault("text", text))
        hpd.show_health_protection_dialog(
            None, _make_payload(ug1_restore_failed=True))
        marker = i18n_setup.t("health.Protect_restore_failed")
        assert captured["text"].startswith(marker)
        captured.clear()
        hpd.show_health_protection_dialog(None, _make_payload())
        assert marker not in captured["text"]


# ── Fitters: ModelFitResult.warnings ─────────────────────────────────

class TestFitterWarnings:

    def test_reefman_few_ug2_levels(self):
        from lm19.reefman import fit_reefman
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        # quick_pentode has a single Ug2 level — already < 3
        result = fit_reefman(pts, topology=TOPOLOGY_PENTODE)
        codes = {w["code"] for w in result.warnings}
        assert MODEL_WARN_REEFMAN_FEW_UG2 in codes

    def test_dempwolf_no_ig2_masking_warns(self):
        from lm19.dempwolf import fit_dempwolf
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        no_ig2 = [{k: v for k, v in p.items() if k != "ig2"} for p in pts]
        result = fit_dempwolf(no_ig2, topology=TOPOLOGY_PENTODE)
        codes = {w["code"] for w in result.warnings}
        assert codes & {"dempwolf_no_ig2_mask", "dempwolf_no_ig2_full"}, codes


# ── i18n: every warning code has a key in every locale ───────────────

class TestWarningI18nKeys:

    # From the owner registries (manual shadow lists removed; the old
    # manual OPT_CODES knew only 5 of the 9 codes)
    AMP_CODES = tuple(sorted(ENGINE_WARNING_CODES))
    OPT_CODES = tuple(sorted(OPT_WARNING_CODES))
    MODEL_CODES = tuple(sorted(MODEL_WARNING_CODES))
    WARNBAR = ("Dialog_title", "No_warnings", "category_analysis",
               "category_optimizer", "category_startup", "More_suppressed")

    @pytest.mark.parametrize("locale", available_locales())
    def test_all_codes_translated(self, locale):
        data = json.loads(
            (PROJECT_ROOT / "locales" / f"{locale}.json")
            .read_text(encoding="utf-8"))
        for code in self.AMP_CODES:
            assert f"warn_{code}" in data["amp"], (locale, code)
        for code in self.OPT_CODES:
            assert f"opt_warn_{code}" in data["amp"], (locale, code)
        for code in self.MODEL_CODES:
            assert f"warn_{code}" in data["model"], (locale, code)
        for key in self.WARNBAR:
            assert key in data["warnbar"], (locale, key)
        assert "Emergency_zero_failed" in data["msg"], locale
        assert "Protect_restore_failed" in data["health"], locale


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
