"""Focused tests for ``lm19.io_utils`` helpers.

The helper consolidates a 7-line collision-counter loop that was
duplicated across 4 save / export call sites (``save_measurement``,
``save_imported_measurement``, ``save_health_measurement``,
``compare_tab._export_to_directory``).  These tests cover the
behaviour contract; the call sites themselves are exercised by
their own integration tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from lm19 import calibration, health_measurements, health_refs, measurements
from lm19.io_utils import make_unique_path, write_json


class TestMakeUniquePath:

    def test_path_does_not_exist_returns_same(self, tmp_path: Path):
        """No collision → return original path unchanged (fast path)."""
        p = tmp_path / "scan.json"
        result = make_unique_path(p)
        assert result == p
        # Helper does NOT create the file (caller writes after)
        assert not result.exists()

    def test_single_collision_returns_one_suffix(self, tmp_path: Path):
        """First collision → ``_1`` suffix, original suffix preserved."""
        p = tmp_path / "scan.json"
        p.write_text("first")
        result = make_unique_path(p)
        assert result.name == "scan_1.json"
        assert result.parent == tmp_path
        assert not result.exists()  # caller will create

    def test_multiple_collisions_increments_counter(self, tmp_path: Path):
        """Existing _1, _2 → returns _3."""
        (tmp_path / "scan.json").write_text("0")
        (tmp_path / "scan_1.json").write_text("1")
        (tmp_path / "scan_2.json").write_text("2")
        result = make_unique_path(tmp_path / "scan.json")
        assert result.name == "scan_3.json"

    def test_gap_in_counter_picks_next_after_highest(self, tmp_path: Path):
        """``scan.json`` and ``scan_5.json`` exist, but ``scan_1`` does NOT
        → counter starts at 1, finds gap, returns ``scan_1.json``.

        Documents the contract: the helper finds the FIRST free slot,
        not the slot after the highest. All 4 call sites depend on
        this byte-for-byte."""
        (tmp_path / "scan.json").write_text("0")
        (tmp_path / "scan_5.json").write_text("5")
        result = make_unique_path(tmp_path / "scan.json")
        assert result.name == "scan_1.json"

    def test_non_json_suffix_preserved(self, tmp_path: Path):
        """Helper isn't json-only — preserves whatever suffix the caller
        passes. Pins this contract so call sites with ``.csv``, ``.utd``
        etc. keep working."""
        p = tmp_path / "data.csv"
        p.write_text("col1,col2")
        result = make_unique_path(p)
        assert result.name == "data_1.csv"

    def test_compound_stem_with_dots(self, tmp_path: Path):
        """``a.b.json`` → stem is ``a.b``, suffix is ``.json``."""
        p = tmp_path / "lamp.v2.json"
        p.write_text("first")
        result = make_unique_path(p)
        assert result.name == "lamp.v2_1.json"

    def test_no_extension(self, tmp_path: Path):
        """Path without extension still works (suffix is empty string)."""
        p = tmp_path / "data"
        p.write_text("first")
        result = make_unique_path(p)
        assert result.name == "data_1"

    def test_directory_collision_uses_dir_suffix(self, tmp_path: Path):
        """If a directory exists at the target path, helper still
        treats it as 'exists' and appends counter. Edge case but
        consistent with ``Path.exists()`` semantics."""
        (tmp_path / "scan.json").mkdir()  # weirdly, a directory there
        result = make_unique_path(tmp_path / "scan.json")
        assert result.name == "scan_1.json"


class TestMakeUniquePathInvariant:
    """Behavioural invariant: returned path never points to an existing
    file/dir, regardless of how cluttered the directory is."""

    def test_returned_path_never_exists(self, tmp_path: Path):
        # Pre-populate a noisy directory
        for i in range(20):
            (tmp_path / f"scan_{i}.json").write_text(str(i))
        (tmp_path / "scan.json").write_text("base")
        result = make_unique_path(tmp_path / "scan.json")
        assert not result.exists(), (
            f"Helper returned {result} which already exists. "
            f"This is the contract violation that would cause "
            f"silent overwrites in save_measurement."
        )


def _assert_lf_json(path: Path, expected: object) -> None:
    """Assert the file is LF-terminated UTF-8 JSON holding *expected*.

    ``\\r`` absence discriminates against text-mode writes only where the
    platform translates (Windows); the trailing-newline and encoding
    assertions discriminate everywhere.
    """
    raw = path.read_bytes()
    assert b"\r" not in raw, f"{path.name}: CR found — written in text mode"
    assert raw.endswith(b"\n"), f"{path.name}: no trailing newline"
    assert json.loads(raw.decode("utf-8")) == expected


class TestWriteJson:
    """Contract of the single JSON writer."""

    def test_lf_and_trailing_newline(self, tmp_path: Path):
        p = tmp_path / "a.json"
        write_json(p, {"a": 1, "b": [1, 2]})
        _assert_lf_json(p, {"a": 1, "b": [1, 2]})

    def test_indent_produces_multiple_lines(self, tmp_path: Path):
        """Without indent the CR/LF question is moot — a one-line file has
        no interior line ending. Pin that the writer really is multi-line."""
        p = tmp_path / "a.json"
        write_json(p, {"a": 1, "b": 2})
        assert p.read_bytes().count(b"\n") >= 4

    def test_non_ascii_kept_verbatim(self, tmp_path: Path):
        """``ensure_ascii=False`` default: a native-script value survives as
        characters, not ``\\uXXXX`` escapes."""
        value = "\u0141\u00f3d\u017a"  # native-script sample
        p = tmp_path / "a.json"
        write_json(p, {"name": value})
        assert value in p.read_text(encoding="utf-8")
        _assert_lf_json(p, {"name": value})

    def test_ensure_ascii_opt_in(self, tmp_path: Path):
        p = tmp_path / "a.json"
        write_json(p, {"name": "\u00e9"}, ensure_ascii=True)
        assert "\\u00e9" in p.read_text(encoding="ascii")

    def test_overwrites_longer_previous_content(self, tmp_path: Path):
        """Truncating write, not a partial patch — a shorter payload must
        not leave a tail of the previous one."""
        p = tmp_path / "a.json"
        write_json(p, {"k": "x" * 500})
        write_json(p, {"k": 1})
        _assert_lf_json(p, {"k": 1})


class TestJsonSaveCallSitesUseHelper:
    """Call-site pins: the helper respecting LF proves nothing about the
    savers actually calling it. Each save path is exercised end-to-end and
    its file inspected as bytes.

    Not every write_json call site is listed here — the AST ratchet
    ``test_code_quality.py::TestJsonWritesUseHelper`` is what makes the
    set complete; these pin the paths a user's data actually travels.
    """

    _MEASUREMENT = {
        "timestamp": "2026-01-02T03:04:05",
        "tube_type": "EL84",
        "lamp_id": "L1",
        "name": "pin",
        "points": [{"ua": 250.0, "ug1": -7.0, "ia": 48.0}],
    }

    def test_save_measurement_writes_lf(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(measurements, "_root", lambda: tmp_path)
        path = measurements.save_measurement("EL84", "L1", dict(self._MEASUREMENT))
        raw = path.read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")

    def test_save_imported_measurement_writes_lf(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(measurements, "_root", lambda: tmp_path)
        path = measurements.save_imported_measurement(
            "EL84", "L1", dict(self._MEASUREMENT), source="utd", source_stem="s1")
        raw = path.read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")

    def test_save_health_measurement_writes_lf(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(health_measurements, "_root", lambda: tmp_path)
        path = health_measurements.save_health_measurement(
            "EL84", "L1", dict(self._MEASUREMENT))
        raw = path.read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")

    def test_health_ref_savers_write_lf(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        type_path = health_refs.save_type_ref("EL84", "ref1", {"emission_ratio": 0.9})
        pers_path = health_refs.save_personal_baseline("EL84", "L1", {"ia": 48.0})
        for p in (type_path, pers_path):
            raw = p.read_bytes()
            assert b"\r" not in raw and raw.endswith(b"\n"), p

    def test_set_active_type_ref_rewrite_keeps_lf(self, tmp_path: Path, monkeypatch):
        """The rewrite path (load → flag → save) is a separate call site
        from save_type_ref and flipped EOL independently."""
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        health_refs.save_type_ref("EL84", "ref1", {"emission_ratio": 0.9})
        health_refs.save_type_ref("EL84", "ref2", {"emission_ratio": 1.1})
        health_refs.set_active_type_ref("EL84", "ref2")
        for p in sorted((tmp_path / "config" / "health_refs" / "type" / "EL84").glob("*.json")):
            raw = p.read_bytes()
            assert b"\r" not in raw and raw.endswith(b"\n"), p

    def test_calibration_save_writes_lf(self, tmp_path: Path):
        """Atomic tmp+replace path: the temp file is what gets written, so
        the helper has to be used there, not on the final name."""
        path = tmp_path / "calibration.json"
        calibration.CalibrationData().save(path)
        raw = path.read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")
        assert not path.with_suffix(".json.tmp").exists()
