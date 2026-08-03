"""Tests for LTSpice test schematic generation.

Run:  py -m pytest tests/test_ltspice_asc.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.ltspice_asc import generate_test_schematic, _extract_sweep_params
from lm19.tube_sim import quick_pentode, quick_triode
from lm19.amplifier.constants import (
    CIRCUIT_PP,
    CIRCUIT_SE,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)
from lm19.tube_model_base import (
    MODEL_TYPE_DEMPWOLF,
    MODEL_TYPE_KOREN,
    MODEL_TYPE_REEFMAN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def triode_data():
    _model, pts = quick_triode("12AX7")
    return pts


@pytest.fixture(scope="module")
def pentode_data():
    _model, pts = quick_pentode("EL84")
    return pts


@pytest.fixture
def triode_sub(tmp_path, triode_data):
    """Create a dummy .sub file and return its path."""
    sub = tmp_path / "12AX7.sub"
    sub.write_text("; dummy subcircuit\n.SUBCKT 12AX7 A G K\n.ENDS 12AX7\n")
    return sub


@pytest.fixture
def pentode_sub(tmp_path, pentode_data):
    sub = tmp_path / "EL84.sub"
    sub.write_text("; dummy\n.SUBCKT EL84 A G K G2\n.ENDS EL84\n")
    return sub


# ---------------------------------------------------------------------------
# Sweep parameter extraction
# ---------------------------------------------------------------------------

class TestExtractSweepParams:

    def test_triode_basic(self, triode_data):
        params = _extract_sweep_params(triode_data, "triode")
        assert "ua_max" in params
        assert "ug1_start" in params
        assert "ug1_stop" in params
        assert "ug1_step" in params
        assert float(params["ua_max"]) > 0
        assert float(params["ug1_start"]) <= float(params["ug1_stop"])
        assert "ug2" not in params

    def test_pentode_has_ug2(self, pentode_data):
        params = _extract_sweep_params(pentode_data, "pentode")
        assert "ug2" in params
        assert float(params["ug2"]) > 10

    def test_step_positive(self, triode_data):
        params = _extract_sweep_params(triode_data, "triode")
        assert float(params["ua_step"]) >= 1
        assert float(params["ug1_step"]) > 0


# ---------------------------------------------------------------------------
# Triode schematic generation
# ---------------------------------------------------------------------------

class TestTriodeSchematic:

    def test_generates_asc_file(self, triode_sub, triode_data, tmp_path):
        result = generate_test_schematic(
            str(triode_sub), "12AX7", triode_data, "triode")
        assert result is not None
        asc_path = Path(result)
        assert asc_path.exists()
        assert asc_path.suffix == ".asc"

    def test_generates_asy_file(self, triode_sub, triode_data, tmp_path):
        generate_test_schematic(
            str(triode_sub), "12AX7", triode_data, "triode")
        asy_path = tmp_path / "12AX7.asy"
        assert asy_path.exists()

    def test_asc_contains_dc_sweep(self, triode_sub, triode_data):
        result = generate_test_schematic(
            str(triode_sub), "12AX7", triode_data, "triode")
        content = Path(result).read_text(encoding="utf-8")
        assert ".dc V1" in content
        assert "V2" in content

    def test_asc_contains_include(self, triode_sub, triode_data):
        result = generate_test_schematic(
            str(triode_sub), "12AX7", triode_data, "triode")
        content = Path(result).read_text(encoding="utf-8")
        assert ".include 12AX7.sub" in content

    def test_asc_contains_tube_symbol(self, triode_sub, triode_data):
        result = generate_test_schematic(
            str(triode_sub), "12AX7", triode_data, "triode")
        content = Path(result).read_text(encoding="utf-8")
        assert "SYMBOL 12AX7" in content

    def test_asc_no_unsubstituted_placeholders(self, triode_sub, triode_data):
        result = generate_test_schematic(
            str(triode_sub), "12AX7", triode_data, "triode")
        content = Path(result).read_text(encoding="utf-8")
        # No remaining {placeholder} patterns (except LTSpice {} expressions)
        import re
        remaining = re.findall(r"\{[a-z_]+\}", content)
        assert not remaining, f"Unsubstituted placeholders: {remaining}"

    def test_no_pentode_elements(self, triode_sub, triode_data):
        result = generate_test_schematic(
            str(triode_sub), "12AX7", triode_data, "triode")
        content = Path(result).read_text(encoding="utf-8")
        assert "V3" not in content

    def test_asy_has_3_pins(self, triode_sub, triode_data, tmp_path):
        generate_test_schematic(
            str(triode_sub), "12AX7", triode_data, "triode")
        asy = (tmp_path / "12AX7.asy").read_text(encoding="utf-8")
        assert "PinName A" in asy
        assert "PinName G" in asy
        assert "PinName K" in asy
        assert "PinName G2" not in asy


# ---------------------------------------------------------------------------
# Pentode schematic generation
# ---------------------------------------------------------------------------

class TestPentodeSchematic:

    def test_generates_asc_file(self, pentode_sub, pentode_data, tmp_path):
        result = generate_test_schematic(
            str(pentode_sub), "EL84", pentode_data, "pentode")
        assert result is not None
        assert Path(result).exists()

    def test_generates_asy_file(self, pentode_sub, pentode_data, tmp_path):
        generate_test_schematic(
            str(pentode_sub), "EL84", pentode_data, "pentode")
        asy_path = tmp_path / "EL84.asy"
        assert asy_path.exists()

    def test_asc_has_v3_ug2(self, pentode_sub, pentode_data):
        result = generate_test_schematic(
            str(pentode_sub), "EL84", pentode_data, "pentode")
        content = Path(result).read_text(encoding="utf-8")
        assert "V3" in content

    def test_asc_contains_include(self, pentode_sub, pentode_data):
        result = generate_test_schematic(
            str(pentode_sub), "EL84", pentode_data, "pentode")
        content = Path(result).read_text(encoding="utf-8")
        assert ".include EL84.sub" in content

    def test_asc_no_unsubstituted_placeholders(self, pentode_sub, pentode_data):
        result = generate_test_schematic(
            str(pentode_sub), "EL84", pentode_data, "pentode")
        content = Path(result).read_text(encoding="utf-8")
        import re
        remaining = re.findall(r"\{[a-z_]+\}", content)
        assert not remaining, f"Unsubstituted placeholders: {remaining}"

    def test_asy_has_4_pins(self, pentode_sub, pentode_data, tmp_path):
        generate_test_schematic(
            str(pentode_sub), "EL84", pentode_data, "pentode")
        asy = (tmp_path / "EL84.asy").read_text(encoding="utf-8")
        assert "PinName A" in asy
        assert "PinName G" in asy
        assert "PinName K" in asy
        assert "PinName G2" in asy


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestAsySubcktPinConsistency:
    """Verify .asy SpiceOrder matches .SUBCKT pin order for all models.

    In LTSpice, SpiceOrder N in the .asy maps to the Nth pin in .SUBCKT.
    If these don't match, wires connect to wrong electrodes.
    """

    def _parse_asy_pin_order(self, asy_text: str) -> list:
        """Extract pin names sorted by SpiceOrder from .asy content."""
        import re
        pins = {}
        current_name = None
        for line in asy_text.splitlines():
            m = re.match(r"PINATTR\s+PinName\s+(\S+)", line)
            if m:
                current_name = m.group(1)
            m = re.match(r"PINATTR\s+SpiceOrder\s+(\d+)", line)
            if m and current_name:
                pins[int(m.group(1))] = current_name
                current_name = None
        return [pins[k] for k in sorted(pins)]

    def _parse_subckt_pin_order(self, sub_text: str) -> list:
        """Extract pin names from .SUBCKT line."""
        for line in sub_text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith(".SUBCKT"):
                no_comment = stripped.split(";")[0].strip()
                parts = no_comment.split()
                return parts[2:]  # skip .SUBCKT and name
        return []

    def test_triode_asy_matches_koren_subckt(self, tmp_path, triode_data):
        """Triode .asy SpiceOrder must match Koren .SUBCKT pin order."""
        from lm19.spice_export import fit_and_export_spice
        sub = tmp_path / "tri.sub"
        fit_and_export_spice(str(sub), "12AX7", triode_data,
                             topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_KOREN)
        generate_test_schematic(str(sub), "12AX7", triode_data, "triode")

        asy_pins = self._parse_asy_pin_order(
            (tmp_path / "12AX7.asy").read_text(encoding="utf-8"))
        sub_pins = self._parse_subckt_pin_order(
            sub.read_text(encoding="utf-8"))

        assert asy_pins == sub_pins, (
            f"Triode .asy pins {asy_pins} != .SUBCKT pins {sub_pins}")

    def test_pentode_asy_matches_koren_subckt(self, tmp_path, pentode_data):
        from lm19.spice_export import fit_and_export_spice
        sub = tmp_path / "pent.sub"
        fit_and_export_spice(str(sub), "EL84", pentode_data,
                             topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_KOREN)
        generate_test_schematic(str(sub), "EL84", pentode_data, "pentode")

        asy_pins = self._parse_asy_pin_order(
            (tmp_path / "EL84.asy").read_text(encoding="utf-8"))
        sub_pins = self._parse_subckt_pin_order(
            sub.read_text(encoding="utf-8"))

        assert asy_pins == sub_pins, (
            f"Pentode .asy pins {asy_pins} != .SUBCKT pins {sub_pins}")

    def test_dempwolf_triode_asy_matches_subckt(self, tmp_path, triode_data):
        from lm19.spice_export import fit_and_export_spice
        sub = tmp_path / "dw_tri.sub"
        fit_and_export_spice(str(sub), "12AX7", triode_data,
                             topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_DEMPWOLF)
        generate_test_schematic(str(sub), "12AX7", triode_data, "triode")

        asy_pins = self._parse_asy_pin_order(
            (tmp_path / "12AX7.asy").read_text(encoding="utf-8"))
        sub_pins = self._parse_subckt_pin_order(
            sub.read_text(encoding="utf-8"))

        assert asy_pins == sub_pins, (
            f"Dempwolf triode .asy {asy_pins} != .SUBCKT {sub_pins}")

    def test_dempwolf_pentode_asy_matches_subckt(self, tmp_path, pentode_data):
        from lm19.spice_export import fit_and_export_spice
        sub = tmp_path / "dw_pent.sub"
        fit_and_export_spice(str(sub), "EL84", pentode_data,
                             topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_DEMPWOLF)
        generate_test_schematic(str(sub), "EL84", pentode_data, "pentode")

        asy_pins = self._parse_asy_pin_order(
            (tmp_path / "EL84.asy").read_text(encoding="utf-8"))
        sub_pins = self._parse_subckt_pin_order(
            sub.read_text(encoding="utf-8"))

        assert asy_pins == sub_pins, (
            f"Dempwolf pentode .asy {asy_pins} != .SUBCKT {sub_pins}")

    def test_reefman_pentode_asy_matches_subckt(self, tmp_path, pentode_data):
        from lm19.spice_export import fit_and_export_spice
        sub = tmp_path / "rf_pent.sub"
        fit_and_export_spice(str(sub), "EL84", pentode_data,
                             topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_REEFMAN)
        generate_test_schematic(str(sub), "EL84", pentode_data, "pentode")

        asy_pins = self._parse_asy_pin_order(
            (tmp_path / "EL84.asy").read_text(encoding="utf-8"))
        sub_pins = self._parse_subckt_pin_order(
            sub.read_text(encoding="utf-8"))

        assert asy_pins == sub_pins, (
            f"Reefman pentode .asy {asy_pins} != .SUBCKT {sub_pins}")


# ---------------------------------------------------------------------------
# Sweep parameter values
# ---------------------------------------------------------------------------

class TestSweepParamValues:
    """Verify sweep parameters have sensible values from real data."""

    def test_triode_ua_max_reasonable(self, triode_data):
        params = _extract_sweep_params(triode_data, "triode")
        ua_max = float(params["ua_max"])
        assert 50 <= ua_max <= 1000, f"ua_max={ua_max} out of reasonable range"

    def test_triode_ug1_range_negative(self, triode_data):
        params = _extract_sweep_params(triode_data, "triode")
        ug1_start = float(params["ug1_start"])
        ug1_stop = float(params["ug1_stop"])
        assert ug1_start <= 0, f"ug1_start={ug1_start} should be <= 0"
        assert ug1_stop <= 0, f"ug1_stop={ug1_stop} should be <= 0"
        assert ug1_start <= ug1_stop, "ug1_start should be <= ug1_stop"

    def test_pentode_ug2_reasonable(self, pentode_data):
        params = _extract_sweep_params(pentode_data, "pentode")
        ug2 = float(params["ug2"])
        assert 50 <= ug2 <= 500, f"ug2={ug2} out of reasonable range"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_special_chars_in_name(self, tmp_path):
        """Tube names with special chars get sanitized."""
        sub = tmp_path / "6P3S_E.sub"
        sub.write_text("; dummy\n.SUBCKT 6P3S_E A G K\n.ENDS\n")
        pts = [{"ua": v, "ug1": -2.0, "ia": 5.0}
               for v in range(10, 310, 10)]
        result = generate_test_schematic(
            str(sub), "6P3S-E", pts, "triode")
        assert result is not None
        content = Path(result).read_text(encoding="utf-8")
        assert "6P3S_E" in content  # dash replaced with underscore


# ---------------------------------------------------------------------------
# ML-061: amp schematic export must overwrite a stale .asy
# ---------------------------------------------------------------------------

class TestAmpAsyOverwrite:

    def test_stale_asy_overwritten(self, triode_sub, triode_data, tmp_path):
        """A leftover 3-pin symbol from an earlier export must not survive
        a re-export - it breaks the schematic pinout (ML-061).
        generate_test_schematic always overwrote; only the amp path had
        the exists() guard."""
        from lm19.ltspice_asc import _TEMPLATES_DIR, generate_amp_schematic
        stale = tmp_path / "12AX7.asy"
        stale.write_text("STALE 3-PIN SYMBOL", encoding="utf-8")
        result = generate_amp_schematic(
            str(triode_sub), "12AX7", triode_data, "triode",
            circuit=CIRCUIT_SE, ra_ohm="47k", rk_ohm="1.5k")
        assert result is not None
        expected = (_TEMPLATES_DIR / "triode.asy").read_text(encoding="utf-8")
        assert stale.read_text(encoding="utf-8") == expected

    def test_stale_tube_b_asy_overwritten(self, triode_sub, triode_data,
                                          tmp_path):
        """Mutation-audit: restoring the exists() guard only on
        the tube-B copy survived the single-tube pin."""
        from lm19.ltspice_asc import _TEMPLATES_DIR, generate_amp_schematic
        sub_b = tmp_path / "12AU7.sub"
        sub_b.write_text(triode_sub.read_text(encoding="utf-8"),
                         encoding="utf-8")
        stale_b = tmp_path / "12AU7.asy"
        stale_b.write_text("STALE 3-PIN SYMBOL", encoding="utf-8")
        result = generate_amp_schematic(
            str(triode_sub), "12AX7", triode_data, "triode",
            circuit=CIRCUIT_PP, ra_aa_ohm="8k",
            tube_name_b="12AU7", sub_file_b=str(sub_b))
        assert result is not None
        expected = (_TEMPLATES_DIR / "triode.asy").read_text(encoding="utf-8")
        assert stale_b.read_text(encoding="utf-8") == expected
