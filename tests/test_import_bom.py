"""BOM-handling regression: importers must skip the UTF-8 BOM.

Excel-exported CSV and Windows tools routinely add ``\\xef\\xbb\\xbf``
to the start of the file. With plain ``encoding="utf-8"`` the BOM
survives as ``\\ufeff`` in the first character of line 0, so header
parsing (``"Va" in header.split()`` / ``"%" check`` /
``"# ETRACER"`` prefix) silently picks the wrong format branch with
no error. ``utf-8-sig`` strips the BOM if present and is a no-op
otherwise.

Tests verify byte-for-byte equivalence of parse output with/without
BOM for one representative sample per importer.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


_BOM = b"\xef\xbb\xbf"


def _write_tmp(content: str, suffix: str, *, with_bom: bool) -> str:
    """Write *content* to a tmp file, optionally with UTF-8 BOM prefix."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    data = content.encode("utf-8")
    if with_bom:
        data = _BOM + data
    os.write(fd, data)
    os.close(fd)
    return path


def _cleanup(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# ── Sample contents (minimal, focused on header parse) ─────────────

# Output-format sample (Va on X-axis). BOM ends up in lines[0] as
# "\\ufeffVa (V) Ia (mA)" → header.split()[:3][0] == "\\ufeffVa".
# Without the utf-8-sig fix, ``"Va" in header.split()[:3]`` returns
# False → is_output flips to False → parser takes the transfer branch
# and produces a structurally wrong result. This is the case where
# the BOM bug actually bites.
_UTRACER_OUTPUT = """\
Va (V) Ia (mA)
 Vg = -4 V  Vg = -3 V  Vg = -2 V
5.0 0.05 0.12 0.25
25.0 0.45 0.89 1.56
50.0 0.82 1.55 2.65
"""

_CURVETRACE_DAT = """\
% * Sample: PCC88_5_Vh=6.3V
% * Date / time: 2025-01-29 22:34:13.446470
0.00 0.02500 0.1 0.00007 0 -0.000 -1.000 -0.153 -0.000 0 NA
5.00 0.02500 5.1 0.00063 0 -0.000 -1.000 -0.130 -0.000 0 NA
10.00 0.02500 10.1 0.00158 0 -0.000 -1.000 -0.101 -0.000 0 NA
"""

_ETRACER_CSV = """\
# ETRACER_CSV_FORMAT_VERSION:2.0
# Each curve-set contains 6 rows: [HV1_V HV1_I HV2_V HV2_I NEGV SWEEP_SOURCE]
# ETD_FILE:test/triode.etd
# NEGV:ON NEGV_SWEEP:ON NEGV_SETTING: [0.0:10.0:5.0]
# HV2:OFF HV2_LINK:OFF HV2_SETTING:[0.0:0.0:0.0]
0,50.0,100.0,150.0,nan
0,1.5,5.0,10.0,nan
0,0.5,0.5,0.5,nan
0,0.01,0.01,0.01,nan
0,-0.00,-0.00,-0.00,nan
0,1.00,1.00,1.00,nan
"""

_ETD_HEATER = """\
HEATER_V = 6.3
HEATER_I = 0.3
"""


# ── utracer_import ──────────────────────────────────────────────────

class TestUtracerBom:
    """``parse_utd`` must skip BOM. Without the utf-8-sig fix the BOM
    ended up as ``\\ufeffVg`` in lines[0] →
    ``"Va" in header.split()[:3]`` returned False → ``is_output``
    defaulted to False → wrong format detected."""

    def test_output_format_parses_with_bom(self):
        """Output-format file (Va on X) must parse as ``"output"``
        with a BOM prefix. This is the case where a BOM-corrupted
        header silently mis-classifies the format."""
        from lm19.utracer_import import parse_utd
        path = _write_tmp(_UTRACER_OUTPUT, ".utd", with_bom=True)
        try:
            result = parse_utd(path)
        finally:
            _cleanup(path)
        assert result["format"] == "output", (
            f"Expected format='output' (Va on X-axis), got "
            f"{result['format']!r}. BOM prefix corrupted the "
            f"``\"Va\" in header.split()[:3]`` discriminator."
        )
        assert result["x_name"] == "Va"
        assert result["step_name"] == "Vg"
        assert result["step_values"] == [-4.0, -3.0, -2.0]

    def test_with_and_without_bom_match(self):
        """Byte-for-byte equivalence of parsed structure."""
        from lm19.utracer_import import parse_utd
        no_bom = _write_tmp(_UTRACER_OUTPUT, ".utd", with_bom=False)
        with_bom = _write_tmp(_UTRACER_OUTPUT, ".utd", with_bom=True)
        try:
            r_plain = parse_utd(no_bom)
            r_bom = parse_utd(with_bom)
        finally:
            _cleanup(no_bom)
            _cleanup(with_bom)
        # Compare top-level scalars
        for key in ("format", "x_name", "step_name", "step_values",
                    "x_values", "has_is"):
            assert r_plain[key] == r_bom[key], (
                f"{key}: plain={r_plain[key]!r} vs bom={r_bom[key]!r}"
            )


# ── curvetracedata_import ───────────────────────────────────────────

class TestCurvetracedataBom:
    """``parse_curvetracedata_dat`` extracts ``sample_name`` via
    ``% * Sample:`` regex on the header. Without the utf-8-sig fix
    the BOM-prefixed first line was ``\\ufeff% * Sample: ...`` —
    regex anchored at start-of-line still matched (the ``%`` isn't
    at position 0 anymore). The check this test guards: result is
    non-empty AND sample_name is correctly extracted."""

    def test_parses_with_bom(self):
        from lm19.curvetracedata_import import parse_curvetracedata_dat
        path = _write_tmp(_CURVETRACE_DAT, ".dat", with_bom=True)
        try:
            result = parse_curvetracedata_dat(path)
        finally:
            _cleanup(path)
        assert result["sample_name"] == "PCC88_5_Vh=6.3V"
        assert len(result["points"]) == 3

    def test_with_and_without_bom_match(self):
        from lm19.curvetracedata_import import parse_curvetracedata_dat
        no_bom = _write_tmp(_CURVETRACE_DAT, ".dat", with_bom=False)
        with_bom = _write_tmp(_CURVETRACE_DAT, ".dat", with_bom=True)
        try:
            r_plain = parse_curvetracedata_dat(no_bom)
            r_bom = parse_curvetracedata_dat(with_bom)
        finally:
            _cleanup(no_bom)
            _cleanup(with_bom)
        assert r_plain["sample_name"] == r_bom["sample_name"]
        assert len(r_plain["points"]) == len(r_bom["points"])
        for a, b in zip(r_plain["points"], r_bom["points"]):
            assert a == b


# ── etracer_import ──────────────────────────────────────────────────

class TestEtracerBom:
    """``parse_etracer_csv`` reads ``# ETRACER_CSV_FORMAT_VERSION:`` on
    line 0 to validate the file format. With BOM prefix it became
    ``\\ufeff# ETRACER...`` — startswith("#") still true, but the
    ``# ETRACER_CSV`` discriminator at col 0 broke parsing assumptions."""

    def test_parses_with_bom(self):
        from lm19.etracer_import import parse_etracer_csv
        path = _write_tmp(_ETRACER_CSV, ".csv", with_bom=True)
        try:
            result = parse_etracer_csv(path)
        finally:
            _cleanup(path)
        # version is parsed from line 0 — the field most affected by a
        # leading BOM. Without the utf-8-sig fix the BOM-prefixed
        # line 0 was "\\ufeff# ETRACER_CSV_FORMAT_VERSION:2.0" →
        # version regex may still match but the BOM character could
        # leak into downstream prefix checks.
        assert result["version"] == "2.0"
        assert len(result["curves"]) >= 1

    def test_with_and_without_bom_match(self):
        from lm19.etracer_import import parse_etracer_csv
        no_bom = _write_tmp(_ETRACER_CSV, ".csv", with_bom=False)
        with_bom = _write_tmp(_ETRACER_CSV, ".csv", with_bom=True)
        try:
            r_plain = parse_etracer_csv(no_bom)
            r_bom = parse_etracer_csv(with_bom)
        finally:
            _cleanup(no_bom)
            _cleanup(with_bom)
        # All top-level scalars must match
        for key in ("version", "etd_file", "hv2_on", "hv2_link",
                    "negv_setting", "hv2_setting"):
            assert r_plain[key] == r_bom[key], (
                f"{key}: plain={r_plain[key]!r} vs bom={r_bom[key]!r}"
            )
        # curves list must match in length and shape
        assert len(r_plain["curves"]) == len(r_bom["curves"])


class TestEtracerHeaterBom:
    """``extract_heater_from_etd`` opens the ``.etd`` companion file
    and reads ``HEATER_V`` from it (line 305 in etracer_import.py).
    Same BOM concern."""

    def test_heater_extraction_with_bom(self, tmp_path):
        """Place ETD next to a fake CSV; verify heater parses through BOM."""
        from lm19.etracer_import import extract_heater_from_etd
        # extract_heater_from_etd looks at a path RELATIVE to csv_path.
        etd_filename = "rig.etd"
        etd_path = tmp_path / etd_filename
        etd_path.write_bytes(_BOM + _ETD_HEATER.encode("utf-8"))
        # csv_path can be any path in same dir
        csv_path = str(tmp_path / "fake.csv")
        result = extract_heater_from_etd(csv_path, etd_filename)
        assert result == 6.3, (
            f"Expected 6.3 V from ETD with BOM, got {result}. "
            f"BOM in regex search context broke HEATER_V= match."
        )

    def test_heater_extraction_without_bom(self, tmp_path):
        """BOM-less ETD parses identically to the BOM-prefixed case."""
        from lm19.etracer_import import extract_heater_from_etd
        etd_filename = "rig.etd"
        etd_path = tmp_path / etd_filename
        etd_path.write_text(_ETD_HEATER, encoding="utf-8")
        csv_path = str(tmp_path / "fake.csv")
        assert extract_heater_from_etd(csv_path, etd_filename) == 6.3
