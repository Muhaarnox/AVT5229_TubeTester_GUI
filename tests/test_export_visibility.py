"""Export/reports failure visibility.

Pins:
- ML-114/115: Koren fitters return ``converged`` → ``ModelFitResult``;
- ML-116: Koren pentode without Ig2 → ``koren_kg2_unfitted`` warning;
- ML-111: same condition on the SPICE path → ``SpiceFitResult.warnings``;
- ML-101: dempwolf triode without grid-region points →
  ``dempwolf_grid_defaults`` warning;
- ML-079: combined-export homogeneity (mixed modes BLOCK, mixed types
  warn + Yes/No — deliberate hybrid policy);
- ML-117: .utd matrix holes counted and surfaced via ``stats``;
- ML-087: fit verdict/alerts survive the dialog's accept();
- i18n keys exist in ALL locales for every new code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from i18n_setup import available_locales

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import i18n_setup
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)
from lm19.tube_model_base import (
    MODEL_TYPE_KOREN,
    MODEL_WARN_DEMPWOLF_GRID_DEFAULTS,
    MODEL_WARN_KOREN_KG2_UNFITTED,
)

i18n_setup.setup("en")


def _pentode_points(with_ig2: bool) -> List[Dict]:
    from lm19.tube_sim import quick_pentode
    _, pts = quick_pentode("EL84")
    if with_ig2:
        return pts
    return [{k: v for k, v in p.items() if k != "ig2"} for p in pts]


# ── ML-114/115: converged propagation ────────────────────────────────

class TestKorenConvergedPropagation:

    def test_scipy_fitters_return_converged(self):
        import numpy as np
        from lm19.spice_export import _fit_koren_scipy
        from lm19.tube_sim import quick_triode
        _, pts = quick_triode("12AU7")
        ua = np.array([p["ua"] for p in pts])
        ug1 = np.array([p["ug1"] for p in pts])
        ia = np.array([p["ia"] for p in pts]) / 1000.0
        out = _fit_koren_scipy(ua, ug1, ia)
        assert len(out) == 3
        assert isinstance(out[2], bool)

    def test_fit_koren_result_carries_converged(self):
        from lm19.tube_sim import fit_koren, quick_triode
        _, pts = quick_triode("12AU7")
        result = fit_koren(pts, topology=TOPOLOGY_TRIODE)
        assert result.converged is True

    def test_forced_false_propagates(self, monkeypatch):
        """Mutation-audit-style pin: a hardwired ``converged=True``
        (the pre-fix behaviour) is indistinguishable from propagation on
        clean data — force the fitter to report False and require it to
        reach the result."""
        import lm19.spice_export as se
        from lm19.tube_sim import fit_koren, quick_triode
        orig = se._fit_koren_scipy
        monkeypatch.setattr(
            se, "_fit_koren_scipy",
            lambda *a, **k: (*orig(*a, **k)[:2], False))
        _, pts = quick_triode("12AU7")
        result = fit_koren(pts, topology=TOPOLOGY_TRIODE)
        assert result.converged is False


# ── ML-116: kg2 unfitted warning ─────────────────────────────────────

class TestKorenKg2Unfitted:

    def test_no_ig2_warns(self):
        from lm19.tube_sim import fit_koren
        result = fit_koren(_pentode_points(with_ig2=False),
                           topology=TOPOLOGY_PENTODE)
        codes = {w["code"] for w in result.warnings}
        assert MODEL_WARN_KOREN_KG2_UNFITTED in codes

    def test_with_ig2_no_warning(self):
        from lm19.tube_sim import fit_koren
        result = fit_koren(_pentode_points(with_ig2=True),
                           topology=TOPOLOGY_PENTODE)
        codes = {w["code"] for w in result.warnings}
        assert MODEL_WARN_KOREN_KG2_UNFITTED not in codes


# ── ML-111: SPICE export path warning ────────────────────────────────

class TestSpiceExportKg2Warning:

    def test_export_without_ig2_flags_result(self, tmp_path):
        from lm19.spice_export import fit_and_export_spice
        path = str(tmp_path / "EL84.sub")
        result = fit_and_export_spice(
            path, "EL84", _pentode_points(with_ig2=False),
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_KOREN)
        assert "kg2_unfitted" in result.warnings

    def test_export_with_ig2_clean(self, tmp_path):
        from lm19.spice_export import fit_and_export_spice
        path = str(tmp_path / "EL84.sub")
        result = fit_and_export_spice(
            path, "EL84", _pentode_points(with_ig2=True),
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_KOREN)
        assert result.warnings == []


# ── ML-101: dempwolf grid defaults warning ───────────────────────────

class TestDempwolfGridDefaults:

    def test_no_grid_region_warns(self):
        from lm19.dempwolf import fit_dempwolf
        from lm19.tube_sim import quick_triode
        _, pts = quick_triode("12AU7")
        # mimic a real LM19 scan: no positive-grid region (Ug1 > -1 V)
        pts = [p for p in pts if p["ug1"] <= -1.0]
        assert len(pts) >= 20
        result = fit_dempwolf(pts, topology=TOPOLOGY_TRIODE)
        codes = {w["code"] for w in result.warnings}
        assert MODEL_WARN_DEMPWOLF_GRID_DEFAULTS in codes


# ── ML-079: combined homogeneity (hybrid) ────────────────────────────

class TestCombinedHomogeneity:

    @pytest.fixture()
    def tab(self, qapp):
        from app.compare_tab import CompareTab
        return CompareTab()

    @staticmethod
    def _entry(lamp_type: str, ug2_mode: str) -> Dict:
        return {"lamp_type": lamp_type,
                "data": {"scan": {"ug2_mode": ug2_mode}},
                "points": [{"ua": 1.0, "ug1": -1.0, "ia": 1.0}]}

    def test_mixed_modes_blocked(self, tab, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        shown = {}
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a: shown.setdefault("warned", True)))
        ok = tab._check_combined_homogeneity(
            [self._entry("EL84", "pentode"),
             self._entry("EL84", "triode_connected")], "T")
        assert ok is False and shown.get("warned")

    def test_mixed_types_asks(self, tab, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        asked = {}

        def fake_question(*a, **k):
            asked["yes"] = True
            return QMessageBox.StandardButton.No

        monkeypatch.setattr(QMessageBox, "question",
                            staticmethod(fake_question))
        ok = tab._check_combined_homogeneity(
            [self._entry("EL84", "pentode"),
             self._entry("6P14P", "pentode")], "T")
        assert ok is False and asked.get("yes")

    def test_homogeneous_passes_silently(self, tab, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a: pytest.fail("no dialog expected")))
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: pytest.fail("no dialog expected")))
        ok = tab._check_combined_homogeneity(
            [self._entry("EL84", "pentode"),
             self._entry("EL84", "pentode")], "T")
        assert ok is True


# ── ML-117: .utd holes counter ───────────────────────────────────────

class TestUtdHoles:

    def test_ragged_grid_counts_holes(self):
        from lm19.utracer_export import format_utd
        # 2×2 grid with one missing corner → 1 hole
        pts = [
            {"ua": 100.0, "ug1": -2.0, "ia": 10.0, "ig2": 0.0},
            {"ua": 200.0, "ug1": -2.0, "ia": 20.0, "ig2": 0.0},
            {"ua": 100.0, "ug1": -4.0, "ia": 5.0, "ig2": 0.0},
        ]
        stats: Dict[str, int] = {}
        format_utd(pts, fmt="output", stats=stats)
        assert stats["utd_matrix_holes"] == 1

    def test_full_grid_no_holes(self):
        from lm19.utracer_export import format_utd
        pts = [
            {"ua": 100.0, "ug1": -2.0, "ia": 10.0, "ig2": 0.0},
            {"ua": 200.0, "ug1": -2.0, "ia": 20.0, "ig2": 0.0},
            {"ua": 100.0, "ug1": -4.0, "ia": 5.0, "ig2": 0.0},
            {"ua": 200.0, "ug1": -4.0, "ia": 12.0, "ig2": 0.0},
        ]
        stats: Dict[str, int] = {}
        format_utd(pts, fmt="output", stats=stats)
        assert stats["utd_matrix_holes"] == 0


# ── ML-087: verdict survives accept ──────────────────────────────────

class TestFitVerdictSurvivesAccept:

    def test_fit_sets_verdict_and_alerts(self, qapp, monkeypatch):
        from app.model_dialog import ModelDialog
        from lm19.tube_sim import quick_triode
        _, pts = quick_triode("12AU7")
        dlg = ModelDialog(points=pts, is_triode=True)
        accepted = {}
        monkeypatch.setattr(ModelDialog, "accept",
                            lambda self: accepted.setdefault("yes", True))
        dlg.fit_radio.setChecked(True)
        # koren is index 0 in MODEL_REGISTRY
        dlg._on_accept()
        assert accepted.get("yes")
        assert dlg.fit_verdict  # non-empty one-line verdict
        # clean synthetic triode fit → no degradation alerts
        assert dlg.fit_alerts == []


# ── i18n coverage for all new codes ──────────────────────────────────

class TestExportI18nKeys:

    MODEL_CODES = (MODEL_WARN_KOREN_KG2_UNFITTED,
                   MODEL_WARN_DEMPWOLF_GRID_DEFAULTS)
    MSG_KEYS = ("Spice_warn_kg2_unfitted", "Spice_series_empty",
                "Spice_pp_tube_b_missing", "Combined_mixed_modes",
                "Combined_mixed_types")

    @pytest.mark.parametrize("locale", available_locales())
    def test_keys_exist(self, locale):
        data = json.loads(
            (PROJECT_ROOT / "locales" / f"{locale}.json")
            .read_text(encoding="utf-8"))
        for code in self.MODEL_CODES:
            assert f"warn_{code}" in data["model"], (locale, code)
        for key in self.MSG_KEYS:
            assert key in data["msg"], (locale, key)
        assert "Export_holes" in data["utd"], locale
        assert "Test_set_failed" in data["cal"], locale


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
