"""Tools guards (ML-072 / ML-074 / ML-075 + bootstrap guard).

- extract_lamps.py: BOOTSTRAP-ONLY — running with an existing
  lamps.json is forbidden without --force-overwrite; placeholder
  types are filtered unconditionally; manual lamps are carried over.
- revise_lamp_params_from_tdsl.py: _to_float on negative ranges;
  ug2/ig2 None semantics against clobbering the nominal.

Run:  py -m pytest tests/test_tools_lamp_sources.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from lm19.constants import (
    TOPOLOGY_PENTODE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_tool(name: str):
    script_path = PROJECT_ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def xl():
    return _load_tool("extract_lamps")


@pytest.fixture(scope="module")
def tdsl():
    return _load_tool("revise_lamp_params_from_tdsl")


# Firmware names are built from per-char tokens — exactly 9 chars.
_FIRMWARE_SNIPPET = """
lamprom[FLAMP] = {
{'E','C','C','8','3','_','G','1','1', 63, 30, 20, 250, 12, 0, 0, 16, 625, 1000},
{'P','w','r','S','u','p','p','l','y', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{'F','o','r','F','u','t','U','s','e', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
};
"""


@pytest.fixture
def fake_paths(xl, tmp_path, monkeypatch):
    fw = tmp_path / "TTesterLCD.c"
    fw.write_text(_FIRMWARE_SNIPPET, encoding="utf-8")
    out = tmp_path / "lamps.json"
    monkeypatch.setattr(xl, "FIRMWARE", fw)
    monkeypatch.setattr(xl, "OUT", out)
    monkeypatch.setattr(xl, "LIMITS", tmp_path / "lamp_limits.json")
    return out


def _lamp_types(out: Path) -> list[str]:
    data = json.loads(out.read_text(encoding="utf-8"))
    return [x["type"] for x in data["lamps"]]


# ═══════════════════════════════════════════════════════════════════
#  ML-072 + bootstrap guard: extract_lamps.py
# ═══════════════════════════════════════════════════════════════════

class TestExtractLampsGuard:

    def test_initial_export_writes_without_flag(self, xl, fake_paths):
        assert xl.main([]) == 0
        assert fake_paths.exists()
        assert "ECC83" in _lamp_types(fake_paths)

    def test_placeholders_filtered_unconditionally(self, xl, fake_paths):
        """Conditionality discriminator (ML-072): the existing file has
        NO manual 6P18P — the old PwrSupply filter was dead in this
        branch, and ForFutUse was never filtered."""
        xl.main([])
        types = _lamp_types(fake_paths)
        assert "PwrSupply" not in types
        assert "ForFutUse" not in types

    def test_rerun_refused_without_flag(self, xl, fake_paths, capsys):
        """Bootstrap-only: a re-run without --force-overwrite is
        forbidden, the file stays untouched, a warning is printed."""
        xl.main([])
        before = fake_paths.read_text(encoding="utf-8")
        fake_paths.write_text(before.replace("ECC83", "ECC83X"),
                              encoding="utf-8")  # manual edit
        marked = fake_paths.read_text(encoding="utf-8")
        assert xl.main([]) == 2
        assert fake_paths.read_text(encoding="utf-8") == marked, \
            "refused run must not touch the curated file"
        err = capsys.readouterr().err
        assert "REFUSED" in err and "force-overwrite" in err

    def test_force_overwrite_carries_manual_lamps(self, xl, fake_paths):
        """ML-072 variant B: a manually added lamp (absent from the
        firmware, like GU-29) survives forced regen; a placeholder not."""
        manual = {"type": "GU-29", "socket": "J", "topology": TOPOLOGY_PENTODE,
                  "uh": 12.6, "Pa_max": 40.0}
        fake_paths.write_text(json.dumps(
            {"lamps": [manual, {"type": "PwrSupply"}]},
            ensure_ascii=False), encoding="utf-8")
        assert xl.main(["--force-overwrite"]) == 0
        data = json.loads(fake_paths.read_text(encoding="utf-8"))
        by_type = {x["type"]: x for x in data["lamps"]}
        assert "GU-29" in by_type
        assert by_type["GU-29"]["Pa_max"] == 40.0
        assert "PwrSupply" not in by_type
        assert "ECC83" in by_type

    def test_manual_6p18p_still_replaces_firmware(self, xl, fake_paths,
                                                  monkeypatch):
        """6P18P mechanism regression: a manual record replaces the
        firmware one and is not duplicated by the carry-over."""
        fw = xl.FIRMWARE
        snippet = fw.read_text(encoding="utf-8").replace(
            "{'E','C','C','8','3','_','G','1','1'",
            "{'6','P','1','8','P','_','B','0','2'")
        fw.write_text(snippet, encoding="utf-8")
        curated = {"type": "6P18P", "socket": "B", "uh": 6.3,
                   "Pa_max": 12.0}
        fake_paths.write_text(json.dumps({"lamps": [curated]}),
                              encoding="utf-8")
        assert xl.main(["--force-overwrite"]) == 0
        data = json.loads(fake_paths.read_text(encoding="utf-8"))
        entries = [x for x in data["lamps"] if x["type"] == "6P18P"]
        assert len(entries) == 1, "manual 6P18P must not be duplicated"
        assert entries[0]["Pa_max"] == 12.0, "manual curation must win"

    def test_help_does_not_execute(self, xl, fake_paths):
        """Incident guard: the script used to have no argparse, and
        --help EXECUTED it, overwriting the config."""
        xl.main([])
        before = fake_paths.read_text(encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            xl.main(["--help"])
        assert exc.value.code == 0
        assert fake_paths.read_text(encoding="utf-8") == before


# ═══════════════════════════════════════════════════════════════════
#  ML-074: _to_float on ranges with negative numbers
# ═══════════════════════════════════════════════════════════════════

class TestToFloat:

    @pytest.mark.parametrize("text,expected", [
        ("-2--4", -3.0),    # range of two negatives — old crash
        ("-2-4", 1.0),      # negative first — old crash
        ("2--4", -1.0),     # negative second
        ("2-4", 3.0),       # plain range
        ("2 - 4", 3.0),     # with spaces
        ("-5", -5.0),       # BOUNDARY: single negative — NOT a range
        ("5", 5.0),
        ("1,5", 1.5),       # comma decimal
    ])
    def test_table(self, tdsl, text, expected):
        assert tdsl._to_float(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["", "abc", "3-", "-"])
    def test_unparseable_returns_none(self, tdsl, text):
        assert tdsl._to_float(text) is None


# ═══════════════════════════════════════════════════════════════════
#  ML-075: ug2/ig2 None semantics (do not clobber the nominal with 0)
# ═══════════════════════════════════════════════════════════════════

class TestExtractNominal:

    _BASE_ROW = {"VaV": "250", "Vg1V": "-2", "IamA": "10",
                 "SmA/V": "5", "RaΩ": "10000"}

    def test_pentode_missing_vg2_stays_none(self, tdsl):
        """ML-075 discriminator: the old `ug2 or 0.0` yielded 0.0 which
        passed the `if v is not None` merge filter and clobbered the
        nominal."""
        nom = tdsl._extract_nominal(dict(self._BASE_ROW), "pentode")
        assert nom is not None
        assert nom["ug2"] is None
        assert nom["ig2"] is None   # twin

    def test_pentode_present_vg2_parsed(self, tdsl):
        row = dict(self._BASE_ROW, **{"Vg2V": "250", "Ig2mA": "2.5"})
        nom = tdsl._extract_nominal(row, "pentode")
        assert nom["ug2"] == pytest.approx(250.0)
        assert nom["ig2"] == pytest.approx(2.5)

    def test_triode_literal_zero_preserved(self, tdsl):
        """The triode branch stores a literal 0.0 — semantics, not a skip."""
        nom = tdsl._extract_nominal(dict(self._BASE_ROW), "triode")
        assert nom["ug2"] == 0.0
        assert nom["ig2"] == 0.0

    def test_none_survives_merge_filter(self, tdsl):
        """Contract with the main() merge loop (`if v is not None`):
        None from the builder must preserve the existing nominal."""
        nom = tdsl._extract_nominal(dict(self._BASE_ROW), "pentode")
        lamp = {"ug2": 250.0, "ig2": 2.0}
        for k, v in nom.items():
            if v is not None:
                lamp[k] = v
        assert lamp["ug2"] == 250.0
        assert lamp["ig2"] == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
