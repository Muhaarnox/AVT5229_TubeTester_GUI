"""export_spice_from_model — .sub export from a loaded model, no refit.

Pins:
- byte equivalence with the fit path (Koren triode/pentode): a model
  rebuilt from fit parameters exports into the same file — .sub
  content is single-source (same generators);
- 'no refit' discriminator: a tweaked model parameter lands in .sub
  as-is and distinguishes the output from a fresh fit;
- honest header: without fit_info there are no Fit quality lines
  (stats are not invented), backend = "loaded model (no refit)";
- Dempwolf/Reefman: a model from the lm19 fitter exports, structure
  parameters reach the text (a dispatcher argument swap is caught);
- verification: req.model -> refit is NOT called, basis "loaded";
- SPICE dialog: the checkbox exists only when the selected source has
  a model, radio buttons grey out, the choice reaches the export path.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.spice_export import export_spice_from_model, fit_and_export_spice
from lm19.tube_params import KorenParams
from lm19.tube_sim import TubeModel, quick_pentode, quick_triode
from lm19.amplifier.constants import (
    CIRCUIT_SE,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)
from lm19.tube_model_base import (
    MODEL_TYPE_DEMPWOLF,
    MODEL_TYPE_KOREN,
)


@pytest.fixture(scope="module")
def triode_points():
    _, pts = quick_triode("12AU7")
    return pts


@pytest.fixture(scope="module")
def pentode_points_no_ig2():
    """Without ig2 the fit path emits no Ig2-RMS header line — the
    byte-equivalence below then needs no line juggling."""
    _, pts = quick_pentode("EL84")
    return [{k: v for k, v in p.items() if k != "ig2"} for p in pts]


def _koren_model(result, topology: str) -> TubeModel:
    p = result.params
    return TubeModel(
        name="T1", topology=topology,
        koren=KorenParams(mu=p["mu"], ex=p["ex"], kg1=p["kg1"],
                          kp=p["kp"], kvb=p["kvb"], kg2=p.get("kg2")))


def _fit_info(result) -> dict:
    return {"rms_error": result.rms_error, "max_error": result.max_error,
            "n_points": result.n_points, "backend": "scipy"}


class TestKorenEquivalence:
    def test_triode_byte_identical(self, triode_points, tmp_path):
        fit_sub = tmp_path / "fit.sub"
        r = fit_and_export_spice(str(fit_sub), "T1", triode_points,
                                 topology=TOPOLOGY_TRIODE)
        model_sub = tmp_path / "model.sub"
        export_spice_from_model(str(model_sub), _koren_model(r, "triode"),
                                tube_type="T1", fit_info=_fit_info(r))
        assert model_sub.read_bytes() == fit_sub.read_bytes()

    def test_pentode_byte_identical(self, pentode_points_no_ig2, tmp_path):
        fit_sub = tmp_path / "fit.sub"
        r = fit_and_export_spice(str(fit_sub), "T1", pentode_points_no_ig2,
                                 topology=TOPOLOGY_PENTODE)
        model_sub = tmp_path / "model.sub"
        export_spice_from_model(str(model_sub), _koren_model(r, "pentode"),
                                tube_type="T1", fit_info=_fit_info(r))
        assert model_sub.read_bytes() == fit_sub.read_bytes()

    def test_tweaked_model_exports_as_is(self, triode_points, tmp_path):
        """The whole point of the API: NO refit — a hand-tweaked parameter
        must land in the .sub verbatim."""
        r = fit_and_export_spice(str(tmp_path / "fit.sub"), "T1",
                                 triode_points, topology=TOPOLOGY_TRIODE)
        model = _koren_model(r, "triode")
        model.koren.mu = 123.4567
        out = tmp_path / "tweaked.sub"
        export_spice_from_model(str(out), model, tube_type="T1")
        text = out.read_text(encoding="utf-8")
        assert "MU=123.4567" in text

    def test_no_fit_info_header_is_honest(self, triode_points, tmp_path):
        r = fit_and_export_spice(str(tmp_path / "fit.sub"), "T1",
                                 triode_points, topology=TOPOLOGY_TRIODE)
        out = tmp_path / "m.sub"
        export_spice_from_model(str(out), _koren_model(r, "triode"),
                                tube_type="T1")
        text = out.read_text(encoding="utf-8")
        assert "Fit quality" not in text
        assert "Data points used" not in text
        assert "loaded model (no refit)" in text

    def test_pentode_without_kg2_refused(self, tmp_path):
        model = TubeModel(name="X", topology=TOPOLOGY_PENTODE,
                          koren=KorenParams(mu=10, ex=1.4, kg1=500,
                                            kp=100, kvb=300, kg2=None))
        with pytest.raises(RuntimeError, match="kg2"):
            export_spice_from_model(str(tmp_path / "x.sub"), model)
        assert not (tmp_path / "x.sub").exists()

    def test_unknown_model_type_refused(self, tmp_path):
        class Alien:
            model_type = "alien"
            topology = TOPOLOGY_TRIODE
            name = "A"

        with pytest.raises(RuntimeError, match="alien"):
            export_spice_from_model(str(tmp_path / "a.sub"), Alien())


class TestDempwolfReefmanFromModel:
    def test_dempwolf_params_reach_sub(self, tmp_path):
        from lm19.dempwolf import fit_dempwolf
        _, pts = quick_pentode("EL84")
        fr = fit_dempwolf(pts, topology=TOPOLOGY_PENTODE)
        out = tmp_path / "dw.sub"
        export_spice_from_model(str(out), fr.model, tube_type="EL84")
        text = out.read_text(encoding="utf-8")
        assert ".SUBCKT EL84" in text
        # struct actually wired through (argument swap would break this)
        assert f"{fr.model.dempwolf.G:.6e}" in text
        assert f"{fr.model.dempwolf.mu:.4f}" in text

    def test_dempwolf_tweak_is_exported_verbatim(self, tmp_path):
        from lm19.dempwolf import fit_dempwolf
        _, pts = quick_pentode("EL84")
        fr = fit_dempwolf(pts, topology=TOPOLOGY_PENTODE)
        fr.model.dempwolf.mu = 77.7777
        out = tmp_path / "dw2.sub"
        export_spice_from_model(str(out), fr.model, tube_type="EL84")
        assert "77.7777" in out.read_text(encoding="utf-8")


    def test_dempwolf_triode_reports_triode_model_type(self, tmp_path,
                                                       triode_points):
        """Twin of the pentode branch: the exported kind must say triode.

        ``model_type`` drives the overlay label and the caller's branch on
        screen-grid data, so a pentode/triode swap here is invisible in the
        .sub text yet wrong everywhere the result object travels.
        """
        from lm19.dempwolf import fit_dempwolf
        fr = fit_dempwolf(triode_points, topology=TOPOLOGY_TRIODE)
        out = tmp_path / "dw_triode.sub"
        res = export_spice_from_model(str(out), fr.model, tube_type="12AU7")
        assert res.model_type == TOPOLOGY_TRIODE
        assert res.algorithm == MODEL_TYPE_DEMPWOLF

    def test_reefman_variant_and_params_reach_sub(self, tmp_path):
        from lm19.reefman import fit_reefman
        _, pts = quick_pentode("EL84")
        rr = fit_reefman(pts, topology=TOPOLOGY_PENTODE)
        out = tmp_path / "rf.sub"
        export_spice_from_model(str(out), rr.model, tube_type="EL84")
        text = out.read_text(encoding="utf-8")
        assert ".SUBCKT EL84" in text
        assert rr.model.reefman.type in text     # variant label honest
        assert f"{rr.model.reefman.mu:.4f}" in text


class TestVerificationUsesLoadedModel:
    def test_no_refit_and_loaded_basis(self, tmp_path, monkeypatch):
        import lm19.ltspice_verify as lv
        from lm19.ltspice_verify import VerifyRequest, run_verification

        _, pts = quick_triode("12AU7")
        r = fit_and_export_spice(str(tmp_path / "f.sub"), "T1", pts,
                                 topology=TOPOLOGY_TRIODE)
        model = _koren_model(r, "triode")
        model.koren.mu = 55.5555

        def _no_refit(*a, **k):
            raise AssertionError("fit_and_export_spice called despite model")

        monkeypatch.setattr("lm19.spice_export.fit_and_export_spice",
                            _no_refit)
        monkeypatch.setattr(lv, "ltspice_available", lambda *a: True)
        req = VerifyRequest(circuit=CIRCUIT_SE, tube_type="T1", topology=TOPOLOGY_TRIODE,
                            points=pts, model=model)
        # stop immediately after the .sub export — offline pin
        vr = run_verification(req, workdir=str(tmp_path / "wd"),
                              stop=lambda: True)
        assert "loaded koren model" in vr.basis
        assert vr.fit_rms_ma is None            # no refit → no fresh RMS
        sub = (tmp_path / "wd" / "verify_model.sub").read_text(
            encoding="utf-8")
        assert "MU=55.5555" in sub              # the tweak, not a refit


class TestSpiceDialogLoadedModel:
    @pytest.fixture(scope="class")
    def qapp(self):
        from PySide6.QtWidgets import QApplication
        return QApplication.instance() or QApplication([])

    def test_checkbox_disabled_without_model(self, qapp):
        from app.export_manager import SpiceExportDialog
        dlg = SpiceExportDialog(loaded_models={})
        assert not dlg.cb_use_loaded.isEnabled()
        assert not dlg.use_loaded_model

    def test_single_model_reachable_without_source_combo(self, qapp):
        """UX fix: models attach to overlay series (never sid 0) — with
        exactly one model and the source combo hidden the checkbox must
        still be usable."""
        from app.export_manager import SpiceExportDialog
        dlg = SpiceExportDialog(series_labels={},
                                loaded_models={5: "koren"})
        assert dlg.cb_use_loaded.isEnabled()

    def test_checkbox_follows_selected_source(self, qapp):
        from app.export_manager import SpiceExportDialog
        dlg = SpiceExportDialog(
            series_labels={1: "lamp A", 2: "lamp B"},
            loaded_models={1: "dempwolf"})
        dlg.pp_tube_a_combo.setCurrentIndex(0)   # sid 1 — has a model
        assert dlg.cb_use_loaded.isEnabled()
        dlg.cb_use_loaded.setChecked(True)
        assert all(not rb.isEnabled() for rb in dlg._model_radios)
        dlg.pp_tube_a_combo.setCurrentIndex(1)   # sid 2 — no model
        assert not dlg.cb_use_loaded.isEnabled()
        assert not dlg.use_loaded_model          # auto-unchecked
        assert all(rb.isEnabled() for rb in dlg._model_radios)

    def test_export_path_uses_loaded_model(self, qapp, tmp_path,
                                           monkeypatch):
        import app.export_manager as em
        import lm19.spice_export as spx
        from PySide6.QtWidgets import QDialog

        sentinel = object()

        class _FakeDlg:
            def __init__(self, parent=None, series_labels=None,
                         loaded_models=None):
                self.loaded_models = loaded_models

            def exec(self):
                return QDialog.DialogCode.Accepted

            model_type = MODEL_TYPE_KOREN
            generate_test_schematic = False
            generate_amp_schematic = False
            amp_circuit = "se"
            r_load = "8"
            f_low = 20.0
            pp_tube_a_sid = None
            pp_tube_b_sid = None
            use_loaded_model = True

            def source_sid(self):
                return 0

        captured = {}

        def _fake_export(path, model, tube_type=None, mfg_date=""):
            captured["model"] = model
            Path(path).write_text("; stub .sub", encoding="utf-8")
            from lm19.spice_export import SpiceFitResult
            return SpiceFitResult(model_type=TOPOLOGY_TRIODE, algorithm=MODEL_TYPE_KOREN,
                                  params={}, rms_error=None, max_error=None,
                                  n_points=0, path=path)

        def _no_refit(*a, **k):
            raise AssertionError("refit despite use_loaded_model")

        monkeypatch.setattr(em, "SpiceExportDialog", _FakeDlg)
        monkeypatch.setattr(spx, "export_spice_from_model", _fake_export)
        monkeypatch.setattr(spx, "fit_and_export_spice", _no_refit)
        monkeypatch.setattr(
            em.QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (str(tmp_path / "m.sub"), "")))
        monkeypatch.setattr(em.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))

        em.export_spice(parent=None, points=[{"ua": 1.0, "ug1": -1.0,
                                              "ia": 1.0}],
                        tube_type="T1", series_models={0: sentinel})
        assert captured["model"] is sentinel

        # single model under an overlay sid, source combo at default 0:
        # the fallback must still find it (UX fix)
        captured.clear()
        em.export_spice(parent=None, points=[{"ua": 1.0, "ug1": -1.0,
                                              "ia": 1.0}],
                        tube_type="T1", series_models={5: sentinel})
        assert captured["model"] is sentinel

    def test_no_silent_model_swap_with_visible_combo(self, qapp, tmp_path,
                                                     monkeypatch):
        """With a VISIBLE source combo a modelless selection must refit,
        never silently borrow the single model of another series
        (defense-in-depth mirror of the dialog-side guard)."""
        import app.export_manager as em
        import lm19.spice_export as spx
        from PySide6.QtWidgets import QDialog

        class _FakeDlg:
            def __init__(self, parent=None, series_labels=None,
                         loaded_models=None):
                pass

            def exec(self):
                return QDialog.DialogCode.Accepted

            model_type = MODEL_TYPE_KOREN
            generate_test_schematic = False
            generate_amp_schematic = False
            amp_circuit = "se"
            r_load = "8"
            f_low = 20.0
            pp_tube_a_sid = None
            pp_tube_b_sid = None
            use_loaded_model = True   # stale checkbox state

            def source_sid(self):
                return 0              # modelless source selected

        fitted, from_model = [], []

        def _fake_fit(path, *a, **k):
            fitted.append(path)
            Path(path).write_text("; fit stub", encoding="utf-8")
            from lm19.spice_export import SpiceFitResult
            return SpiceFitResult(model_type=TOPOLOGY_TRIODE, algorithm=MODEL_TYPE_KOREN,
                                  params={}, rms_error=1.0, max_error=2.0,
                                  n_points=3, path=path)

        monkeypatch.setattr(em, "SpiceExportDialog", _FakeDlg)
        monkeypatch.setattr(spx, "fit_and_export_spice", _fake_fit)
        monkeypatch.setattr(spx, "export_spice_from_model",
                            lambda *a, **k: from_model.append(a))
        monkeypatch.setattr(
            em.QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (str(tmp_path / "s.sub"), "")))
        monkeypatch.setattr(em.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))

        em.export_spice(parent=None,
                        points=[{"ua": 1.0, "ug1": -1.0, "ia": 1.0}],
                        tube_type="T1",
                        series_labels={1: "lamp A", 5: "model B"},
                        series_models={5: object()})
        assert fitted and not from_model

    def test_pp_mismatched_exports_both_tubes_as_is(self, qapp, tmp_path,
                                                    monkeypatch):
        """The checkbox covers BOTH tubes of a mismatched pair — an as-is
        tube A with a silently refitted tube B would mismatch the pair."""
        import app.export_manager as em
        import lm19.spice_export as spx
        from PySide6.QtWidgets import QDialog

        model_a, model_b = object(), object()

        class _FakeDlg:
            def __init__(self, parent=None, series_labels=None,
                         loaded_models=None):
                pass

            def exec(self):
                return QDialog.DialogCode.Accepted

            model_type = MODEL_TYPE_KOREN
            generate_test_schematic = False
            generate_amp_schematic = False
            amp_circuit = "pp"
            r_load = "8"
            f_low = 20.0
            pp_tube_a_sid = 1
            pp_tube_b_sid = 5
            use_loaded_model = True

            def source_sid(self):
                return 1

        exported = []

        def _fake_export(path, model, tube_type=None, mfg_date=""):
            exported.append((Path(path).name, model))
            Path(path).write_text("; stub", encoding="utf-8")
            from lm19.spice_export import SpiceFitResult
            return SpiceFitResult(model_type=TOPOLOGY_PENTODE, algorithm=MODEL_TYPE_KOREN,
                                  params={}, rms_error=None, max_error=None,
                                  n_points=0, path=path)

        def _no_refit(*a, **k):
            raise AssertionError("refit despite use_loaded_model")

        monkeypatch.setattr(em, "SpiceExportDialog", _FakeDlg)
        monkeypatch.setattr(spx, "export_spice_from_model", _fake_export)
        monkeypatch.setattr(spx, "fit_and_export_spice", _no_refit)
        monkeypatch.setattr(
            em.QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (str(tmp_path / "pair.sub"), "")))
        monkeypatch.setattr(em.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))

        all_points = [
            {"ua": 1.0, "ug1": -1.0, "ia": 1.0, "series_id": 1},
            {"ua": 2.0, "ug1": -1.0, "ia": 2.0, "series_id": 5},
        ]
        em.export_spice(parent=None, points=all_points[:1], tube_type="T1",
                        all_points=all_points,
                        series_models={1: model_a, 5: model_b})
        assert exported == [("pair.sub", model_a),
                            ("pair_B.sub", model_b)]
