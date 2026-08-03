"""Tests for tools/compare_models.py CLI helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from lm19.constants import (
    TOPOLOGY_PENTODE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_compare_models_module():
    """Load compare_models.py as a module (it's a script, not a package)."""
    script_path = PROJECT_ROOT / "tools" / "compare_models.py"
    spec = importlib.util.spec_from_file_location("compare_models_under_test", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cm():
    return _load_compare_models_module()


class TestSelectTubeDirs:
    @pytest.fixture
    def root_with_dirs(self, tmp_path: Path) -> Path:
        for name in ["EL84", "6S19P", "ECC83", "6P14P"]:
            (tmp_path / name).mkdir()
        (tmp_path / "README.txt").write_text("ignore me")  # not a dir
        return tmp_path

    def test_no_filter_returns_empty(self, cm, root_with_dirs: Path) -> None:
        # Explicit --all path lives in main(); _select_tube_dirs only filters.
        # Empty filter must NOT silently return everything, to avoid accidental
        # full-database scans when user forgets the argument.
        result = cm._select_tube_dirs(root_with_dirs, [])
        assert result == []

    def test_files_are_ignored(self, cm, root_with_dirs: Path) -> None:
        result = cm._select_tube_dirs(root_with_dirs, ["EL84"])
        assert all(p.is_dir() for p in result)
        assert "README.txt" not in [p.name for p in result]

    def test_single_filter_matches_exact_name(self, cm, root_with_dirs: Path) -> None:
        result = cm._select_tube_dirs(root_with_dirs, ["EL84"])
        assert [p.name for p in result] == ["EL84"]

    def test_filter_is_case_insensitive(self, cm, root_with_dirs: Path) -> None:
        result = cm._select_tube_dirs(root_with_dirs, ["el84"])
        assert [p.name for p in result] == ["EL84"]

    def test_filter_is_substring_match(self, cm, root_with_dirs: Path) -> None:
        # "6" matches "6S19P" and "6P14P"
        result = cm._select_tube_dirs(root_with_dirs, ["6"])
        assert sorted(p.name for p in result) == ["6P14P", "6S19P"]

    def test_multiple_filters_union(self, cm, root_with_dirs: Path) -> None:
        result = cm._select_tube_dirs(root_with_dirs, ["EL84", "6S19P"])
        assert sorted(p.name for p in result) == ["6S19P", "EL84"]

    def test_filter_no_match_returns_empty(self, cm, root_with_dirs: Path) -> None:
        result = cm._select_tube_dirs(root_with_dirs, ["NONEXISTENT"])
        assert result == []

    def test_empty_root(self, cm, tmp_path: Path) -> None:
        # Even with a filter, empty root yields nothing.
        result = cm._select_tube_dirs(tmp_path, ["EL84"])
        assert result == []


class TestListAvailableTubeTypes:
    def test_lists_only_directories_sorted(self, cm, tmp_path: Path) -> None:
        for name in ["EL84", "6S19P", "ECC83"]:
            (tmp_path / name).mkdir()
        (tmp_path / "stray.json").write_text("{}")
        result = cm._list_available_tube_types(tmp_path)
        assert result == ["6S19P", "ECC83", "EL84"]

    def test_empty_root(self, cm, tmp_path: Path) -> None:
        assert cm._list_available_tube_types(tmp_path) == []


class TestReefmanRowProduction:
    """ML-137: the local Derk/DerkE copy is removed — the Reefman row
    is computed by the PRODUCTION fitter lm19.reefman.fit_reefman
    (what the application user actually gets)."""

    def test_row_uses_production_fit_reefman(self, cm, monkeypatch) -> None:
        import lm19.reefman as rf

        class _FakeModel:
            def params_dict(self):
                return {"type": "BTetrodeDE"}

        class _FakeResult:
            model = _FakeModel()
            rms_error = 1.25
            max_error = 4.5
            rms_ig2 = 0.75
            n_points = 321

        seen = {}

        def fake_fit(points, topology):
            seen["topology"] = topology
            seen["n"] = len(points)
            return _FakeResult()

        monkeypatch.setattr(rf, "fit_reefman", fake_fit)
        row = cm.fit_reefman_row([{"ua": 1.0}] * 5)
        assert seen == {"topology": TOPOLOGY_PENTODE, "n": 5}
        assert row == {"model": "Reefman (DE)", "rms_ia": 1.25,
                       "max_ia": 4.5, "rms_ig2": 0.75, "n_points": 321}

    def test_rms_ig2_none_becomes_zero_and_variant_d(self, cm,
                                                     monkeypatch) -> None:
        import lm19.reefman as rf

        class _FakeModel:
            def params_dict(self):
                return {"type": "BTetrodeD"}

        class _FakeResult:
            model = _FakeModel()
            rms_error = 2.0
            max_error = 3.0
            rms_ig2 = None          # optional branch (no Ig2 data)
            n_points = 10

        monkeypatch.setattr(rf, "fit_reefman",
                            lambda points, topology: _FakeResult())
        row = cm.fit_reefman_row([{"ua": 1.0}])
        assert row["model"] == "Reefman (D)"
        assert row["rms_ig2"] == 0.0

    def test_no_local_model_math_ratchet(self) -> None:
        """Local model math must not return to tools."""
        src = (PROJECT_ROOT / "tools" / "compare_models.py").read_text(
            encoding="utf-8")
        for banned in ("_koren_current", "_derk_ia_ig2", "_derke_ia_ig2",
                       "least_squares", "def fit_derk", "def fit_derke",
                       "def _fit_reefman"):
            assert banned not in src, f"local fitter math is back: {banned}"

    def test_run_comparison_prints_production_row(
        self, cm, monkeypatch, tmp_path: Path, capsys,
    ) -> None:
        """Call site: run_comparison prints the fit_reefman_row line."""
        import json

        import lm19.dempwolf as dw
        import lm19.tube_sim as ts

        def boom(points, topology):
            raise RuntimeError("skip fitter")

        monkeypatch.setattr(ts, "fit_koren", boom)
        monkeypatch.setattr(dw, "fit_dempwolf", boom)
        monkeypatch.setattr(cm, "fit_reefman_row", lambda pts: {
            "model": "Reefman (D)", "rms_ia": 0.5, "max_ia": 1.0,
            "rms_ig2": 0.25, "n_points": 42})
        f = tmp_path / "m.json"
        f.write_text(json.dumps({"topology": TOPOLOGY_PENTODE,
                                 "points": [{"ua": 1.0}]}),
                     encoding="utf-8")
        cm.run_comparison(str(f))
        out = capsys.readouterr().out
        assert "Reefman (D)" in out
        assert "Winner (Ia RMS): Reefman (D)" in out

    @pytest.mark.timeout(120)
    def test_production_fit_on_real_6p1p(self, cm) -> None:
        """Real 6P1P scan (2827 points): the production fit is physical.
        Probed: Reefman (D), rms_ia ~1.01 mA, ~1.6 s."""
        import json
        data = json.loads(
            (PROJECT_ROOT / "tests" / "spice_test_data" / "converted" /
             "pentode_6P1P_real.json").read_text(encoding="utf-8"))
        row = cm.fit_reefman_row(data["points"])
        assert row["model"] in ("Reefman (D)", "Reefman (DE)")
        assert 0.0 < row["rms_ia"] < 5.0     # order of magnitude, not a snapshot
        assert row["rms_ia"] < row["max_ia"]
        assert row["rms_ig2"] > 0.0          # Ig2 data present -> metric alive
        assert row["n_points"] > 1000
