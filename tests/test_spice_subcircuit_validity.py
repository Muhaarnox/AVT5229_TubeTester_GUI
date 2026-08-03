"""Structural validity tests for SPICE subcircuit netlists.

Ensures that all generated .sub files have consistent pin naming,
valid internal node references, and (for Koren) electrically correct
round-trip behavior.

These tests are designed to catch errors during pin renaming /
unification across models (Koren, Dempwolf, Reefman).

Run:  py -m pytest tests/test_spice_subcircuit_validity.py -v
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.spice_export import (
    fit_and_export_spice,
    _generate_triode_subcircuit,
    _generate_pentode_subcircuit,
)
from lm19.tube_sim import quick_pentode, quick_triode
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
# Helpers — .sub parser
# ---------------------------------------------------------------------------

def _parse_subckt(content: str):
    """Parse a .sub file and return (pins, internal_nodes, v_refs).

    Returns:
        pins: list of pin names from .SUBCKT line
        internal_nodes: set of node names used in E/G/R/C/D statements
            (left-hand side connections, e.g. "7", "0", "8")
        v_refs: set of (node_a, node_b) tuples from V(x,y) expressions
    """
    pins = []
    v_refs = set()

    for line in content.splitlines():
        stripped = line.strip()

        # Parse .SUBCKT line
        if stripped.upper().startswith(".SUBCKT"):
            # .SUBCKT NAME pin1 pin2 ... [; comment]
            no_comment = stripped.split(";")[0].strip()
            parts = no_comment.split()
            # parts[0] = ".SUBCKT", parts[1] = name, rest = pins
            pins = parts[2:]

    # Find all V(x,y) references in VALUE={...} expressions
    v_refs = set(re.findall(r"V\((\w+),(\w+)\)", content))

    return pins, v_refs


def _get_internal_nodes(content: str) -> set:
    """Extract internal node numbers from E/RE statements (e.g. E1 7 0)."""
    nodes = set()
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";") or stripped.startswith("*"):
            continue
        if stripped.startswith("+"):
            continue
        # Match patterns like "E1 7 0", "RE1 7 0", "E2 8 0"
        m = re.match(r"(?:E|RE)\w*\s+(\w+)\s+(\w+)", stripped)
        if m:
            nodes.add(m.group(1))
            nodes.add(m.group(2))
    return nodes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pentode_points():
    _model, pts = quick_pentode("EL84")
    return pts


@pytest.fixture(scope="module")
def triode_points():
    _model, pts = quick_triode("12AX7")
    return pts


# ---------------------------------------------------------------------------
# 1. V() reference validity — all V(x,y) must use known pins or internals
# ---------------------------------------------------------------------------

class TestVoltageRefValidity:
    """Every V(x,y) in a VALUE= expression must reference either
    a pin from .SUBCKT or a known internal node (7, 8, 9, 10, ...)."""

    # Known internal nodes used across models (E1→7, E2→8, etc.)
    KNOWN_INTERNAL = {"0", "5", "7", "8", "9", "10", "11", "12", "17"}

    def _check_vrefs(self, content: str, label: str):
        pins, v_refs = _parse_subckt(content)
        assert pins, f"{label}: no pins found in .SUBCKT"

        valid = set(pins) | self.KNOWN_INTERNAL
        bad = []
        for a, b in v_refs:
            if a not in valid:
                bad.append(f"V({a},{b}) — '{a}' unknown")
            if b not in valid:
                bad.append(f"V({a},{b}) — '{b}' unknown")

        assert not bad, (
            f"{label}: invalid V() references:\n"
            + "\n".join(f"  {x}" for x in bad)
            + f"\n  Valid nodes: {sorted(valid)}"
        )

    def test_koren_triode_vrefs(self):
        content = _generate_triode_subcircuit(
            "TEST", "TEST", 100, 1.4, 1060, 600, 300,
            0.5, 1.0, 50, "test")
        self._check_vrefs(content, "Koren triode")

    def test_koren_pentode_vrefs(self):
        content = _generate_pentode_subcircuit(
            "TEST", "TEST", 8, 1.35, 890, 60, 24, 4200,
            5.0, 15.0, 30, "test")
        self._check_vrefs(content, "Koren pentode")

    def test_dempwolf_pentode_vrefs(self, tmp_path, pentode_points):
        out = tmp_path / "dw_pent.sub"
        fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_DEMPWOLF)
        self._check_vrefs(out.read_text(encoding="utf-8"),
                          "Dempwolf pentode")

    def test_dempwolf_triode_vrefs(self, tmp_path, triode_points):
        out = tmp_path / "dw_tri.sub"
        fit_and_export_spice(
            str(out), "12AX7", triode_points,
            topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_DEMPWOLF)
        self._check_vrefs(out.read_text(encoding="utf-8"),
                          "Dempwolf triode")

    def test_reefman_pentode_vrefs(self, tmp_path, pentode_points):
        out = tmp_path / "rf_pent.sub"
        fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_REEFMAN)
        self._check_vrefs(out.read_text(encoding="utf-8"),
                          "Reefman pentode")


# ---------------------------------------------------------------------------
# 2. Pin order uniformity — all models must use the same pin convention
# ---------------------------------------------------------------------------

# Expected pin conventions (will enforce after unification):
TRIODE_PINS = {"A", "G", "K"}
PENTODE_PINS = {"A", "G", "K", "G2"}


class TestPinUniformity:
    """All models should use the same pin naming convention.

    After unification:
      triode  → A G K
      pentode → A G K G2

    Before unification, Dempwolf/Reefman use numbered pins (1 2 3 4),
    so these tests are expected to FAIL until the code is updated.
    """

    def test_koren_triode_pins(self):
        content = _generate_triode_subcircuit(
            "T", "T", 100, 1.4, 1060, 600, 300,
            0.5, 1.0, 50, "test")
        pins, _ = _parse_subckt(content)
        assert set(pins) == TRIODE_PINS, f"Koren triode pins: {pins}"

    def test_koren_pentode_pins(self):
        content = _generate_pentode_subcircuit(
            "T", "T", 8, 1.35, 890, 60, 24, 4200,
            5.0, 15.0, 30, "test")
        pins, _ = _parse_subckt(content)
        assert set(pins) == PENTODE_PINS, f"Koren pentode pins: {pins}"

    def test_dempwolf_pentode_pins(self, tmp_path, pentode_points):
        out = tmp_path / "dw.sub"
        fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_DEMPWOLF)
        pins, _ = _parse_subckt(out.read_text(encoding="utf-8"))
        assert set(pins) == PENTODE_PINS, f"Dempwolf pentode pins: {pins}"

    def test_dempwolf_triode_pins(self, tmp_path, triode_points):
        out = tmp_path / "dw.sub"
        fit_and_export_spice(
            str(out), "12AX7", triode_points,
            topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_DEMPWOLF)
        pins, _ = _parse_subckt(out.read_text(encoding="utf-8"))
        assert set(pins) == TRIODE_PINS, f"Dempwolf triode pins: {pins}"

    def test_reefman_pentode_pins(self, tmp_path, pentode_points):
        out = tmp_path / "rf.sub"
        fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_REEFMAN)
        pins, _ = _parse_subckt(out.read_text(encoding="utf-8"))
        assert set(pins) == PENTODE_PINS, f"Reefman pentode pins: {pins}"


# ---------------------------------------------------------------------------
# 3. Koren round-trip — subcircuit params reproduce Python model currents
# ---------------------------------------------------------------------------

def _parse_params(content: str) -> dict:
    """Extract PARAMS: values from subcircuit text."""
    params = {}
    for m in re.finditer(r"(\w+)=([\d.eE+\-]+)", content):
        key = m.group(1).upper()
        try:
            params[key] = float(m.group(2))
        except ValueError:
            pass
    return params


class TestKorenRoundTrip:
    """Generate .sub, parse params back, compute Ia with Python model,
    and verify it matches the original fit — ensuring the subcircuit
    parameters are written correctly and consistently."""

    def test_triode_round_trip(self, tmp_path, triode_points):
        from lm19.spice_export import _koren_ia

        out = tmp_path / "rt_tri.sub"
        result = fit_and_export_spice(
            str(out), "12AX7", triode_points,
            topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_KOREN)
        content = out.read_text(encoding="utf-8")
        p = _parse_params(content)

        # Verify all params present
        for key in ("MU", "EX", "KG1", "KP", "KVB"):
            assert key in p, f"Missing param {key} in subcircuit"

        # Verify params match SpiceFitResult
        assert abs(p["MU"] - result.params["mu"]) < 0.01
        assert abs(p["EX"] - result.params["ex"]) < 0.001
        assert abs(p["KG1"] - result.params["kg1"]) < 0.1
        assert abs(p["KP"] - result.params["kp"]) < 0.1
        assert abs(p["KVB"] - result.params["kvb"]) < 0.1

        # Compute Ia from parsed params and compare with Python model
        pts = [pt for pt in triode_points if pt["ia"] > 0.1][:20]
        ua = np.array([pt["ua"] for pt in pts])
        ug1 = np.array([pt["ug1"] for pt in pts])

        ia_python = _koren_ia(ua, ug1, p["MU"], p["EX"], p["KG1"],
                              p["KP"], p["KVB"])
        ia_result = _koren_ia(ua, ug1, result.params["mu"],
                              result.params["ex"], result.params["kg1"],
                              result.params["kp"], result.params["kvb"])

        # Should be identical (same params, just parsed back)
        np.testing.assert_allclose(ia_python, ia_result, atol=1e-9,
                                   err_msg="Triode round-trip param mismatch")

    def test_pentode_round_trip(self, tmp_path, pentode_points):
        from lm19.spice_export import _koren_ia_pentode

        out = tmp_path / "rt_pent.sub"
        result = fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_KOREN)
        content = out.read_text(encoding="utf-8")
        p = _parse_params(content)

        for key in ("MU", "EX", "KG1", "KG2", "KP", "KVB"):
            assert key in p, f"Missing param {key} in subcircuit"

        assert abs(p["MU"] - result.params["mu"]) < 0.01
        assert abs(p["KG2"] - result.params["kg2"]) < 0.1

        pts = [pt for pt in pentode_points
               if pt["ia"] > 0.1 and pt.get("ug2", 0) > 10][:20]
        ua = np.array([pt["ua"] for pt in pts])
        ug1 = np.array([pt["ug1"] for pt in pts])
        ug2 = np.array([pt["ug2"] for pt in pts])

        ia_python = _koren_ia_pentode(ua, ug1, ug2, p["MU"], p["EX"],
                                      p["KG1"], p["KP"], p["KVB"])
        ia_result = _koren_ia_pentode(ua, ug1, ug2, result.params["mu"],
                                      result.params["ex"], result.params["kg1"],
                                      result.params["kp"], result.params["kvb"])

        np.testing.assert_allclose(ia_python, ia_result, atol=1e-9,
                                   err_msg="Pentode round-trip param mismatch")


# ---------------------------------------------------------------------------
# 4. G/E/R/C/D connections use only valid nodes
# ---------------------------------------------------------------------------

class TestNetlistConnections:
    """Verify that all component connections (G1 A K, RCP A K, C1 G K, etc.)
    use valid pin names or known internal nodes."""

    # Pattern: component_type name node1 node2 [node3...] [VALUE=...]
    _COMPONENT_RE = re.compile(
        r"^([GRCDE]\w*)\s+(\w+)\s+(\w+)"
        r"(?:\s+(\w+))?"  # optional 3rd node
    )

    KNOWN_INTERNAL = {"0", "5", "7", "8", "9", "10", "11", "12", "17"}

    def _check_connections(self, content: str, label: str):
        pins, _ = _parse_subckt(content)
        valid = set(pins) | self.KNOWN_INTERNAL

        bad = []
        in_subckt = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith(".SUBCKT"):
                in_subckt = True
                continue
            if stripped.upper().startswith(".ENDS"):
                in_subckt = False
                continue
            if not in_subckt:
                continue
            if not stripped or stripped.startswith(";") or stripped.startswith("*"):
                continue
            if stripped.startswith("+") or stripped.startswith("."):
                continue

            m = self._COMPONENT_RE.match(stripped)
            if m:
                comp = m.group(1)
                for g in (2, 3, 4):
                    node = m.group(g)
                    if node and node not in valid:
                        # Check it's not a keyword like VALUE, DX, etc.
                        if node.upper() not in ("VALUE", "DX", "1G", "1MEG"):
                            bad.append(f"{comp}: node '{node}' unknown")

        assert not bad, (
            f"{label}: invalid component connections:\n"
            + "\n".join(f"  {x}" for x in bad)
            + f"\n  Valid: {sorted(valid)}"
        )

    def test_koren_triode_connections(self):
        content = _generate_triode_subcircuit(
            "T", "T", 100, 1.4, 1060, 600, 300,
            0.5, 1.0, 50, "test")
        self._check_connections(content, "Koren triode")

    def test_koren_pentode_connections(self):
        content = _generate_pentode_subcircuit(
            "T", "T", 8, 1.35, 890, 60, 24, 4200,
            5.0, 15.0, 30, "test")
        self._check_connections(content, "Koren pentode")

    def test_dempwolf_pentode_connections(self, tmp_path, pentode_points):
        out = tmp_path / "dw.sub"
        fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_DEMPWOLF)
        self._check_connections(out.read_text(encoding="utf-8"),
                                "Dempwolf pentode")

    def test_dempwolf_triode_connections(self, tmp_path, triode_points):
        out = tmp_path / "dw.sub"
        fit_and_export_spice(
            str(out), "12AX7", triode_points,
            topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_DEMPWOLF)
        self._check_connections(out.read_text(encoding="utf-8"),
                                "Dempwolf triode")

    def test_reefman_pentode_connections(self, tmp_path, pentode_points):
        out = tmp_path / "rf.sub"
        fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_REEFMAN)
        self._check_connections(out.read_text(encoding="utf-8"),
                                "Reefman pentode")
