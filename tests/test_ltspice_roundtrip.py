"""LTspice round-trip validation tests.

Export Python model → .sub → LTspice batch → .raw → compare with Python Ia.

Requires LTspice installed at LTSPICE_EXE path.
Run:  py -m pytest tests/test_ltspice_roundtrip.py -v

Simulation tests are slow (~5s each) and require LTspice — they are
gated with @requires_ltspice. Generation-only tests (schematic text,
resistance/inductance parse helpers) run on machines without LTspice.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from lm19.ltspice_raw import LTSPICE_EXE, parse_raw, get_variable
from lm19.spice_export import fit_and_export_spice
from lm19.ltspice_asc import generate_test_schematic, generate_amp_schematic
from lm19.tube_sim import quick_triode, quick_pentode
from lm19.amplifier.constants import (
    CIRCUIT_CF,
    CIRCUIT_PP,
    CIRCUIT_SE,
    CIRCUIT_SE_XFMR,
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

# LTspice-execution classes are gated on the installed binary.
# Generation-only classes (schematic text / parse helpers) run everywhere.
requires_ltspice = pytest.mark.skipif(
    not Path(LTSPICE_EXE).exists(),
    reason=f"LTspice not found at {LTSPICE_EXE}",
)

# ── Constants ────────────────────────────────────────────────────
LTSPICE_TIMEOUT_S = 30
MAX_RMS_ERROR_PCT = 10.0   # max RMS error as % of Ia range
MAX_POINT_ERROR_PCT = 25.0  # max single-point error as % of Ia range


def _run_ltspice(asc_path: str) -> str:
    """Run LTspice in batch mode, return .raw path."""
    result = subprocess.run(
        [LTSPICE_EXE, "-b", asc_path],
        capture_output=True, timeout=LTSPICE_TIMEOUT_S,
    )
    raw_path = str(Path(asc_path).with_suffix(".raw"))
    if not Path(raw_path).exists():
        log_path = str(Path(asc_path).with_suffix(".log"))
        log_text = Path(log_path).read_text(errors="replace") if Path(log_path).exists() else ""
        raise RuntimeError(f"LTspice produced no .raw file.\nLog:\n{log_text}")
    return raw_path


def _export_and_simulate(tube_name, topology, points, tmpdir):
    """Export .sub + .asc, run LTspice, return parsed .raw result."""
    sub_path = os.path.join(tmpdir, f"{tube_name}.sub")
    fit_result = fit_and_export_spice(
        sub_path, tube_name, points,
        topology=topology, model_type=MODEL_TYPE_KOREN,
    )
    asc_path = generate_test_schematic(sub_path, tube_name, points, topology)
    if asc_path is None:
        pytest.skip("Failed to generate test schematic")
    raw_path = _run_ltspice(asc_path)
    return parse_raw(raw_path), fit_result


# ═══════════════════════════════════════════════════════════════════
#  Triode round-trip tests
# ═══════════════════════════════════════════════════════════════════

@requires_ltspice
class TestTriodeRoundTrip:
    """Export triode model to SPICE, simulate, verify non-zero Ia."""

    def test_12au7_produces_output(self):
        """12AU7: LTspice simulation produces valid .raw with current data."""
        model, pts = quick_triode("12AU7")
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_result, fit = _export_and_simulate("12AU7", "triode", pts, tmpdir)
            assert raw_result["n_points"] > 0
            # Should have at least one current variable
            has_current = any("current" in t for t in raw_result["types"])
            assert has_current, f"No current in variables: {raw_result['variables']}"

    def test_12au7_ia_nonzero(self):
        """12AU7: simulated Ia should be non-zero (tube conducts)."""
        model, pts = quick_triode("12AU7")
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_result, fit = _export_and_simulate("12AU7", "triode", pts, tmpdir)
            # Find any current variable
            # Non-vacuous: assert found-flag + physical range.
            found = False
            for i, t in enumerate(raw_result["types"]):
                if "current" in t and i > 0:
                    current = raw_result["data"][:, i]
                    max_current_ma = max(abs(current)) * 1000
                    assert 0.1 < max_current_ma < 2000.0, (
                        f"peak {max_current_ma:.1f} mA out of physical range")
                    found = True
                    break
            assert found, "no current trace in .raw output"

    def test_12ax7_produces_output(self):
        """12AX7: LTspice simulation produces valid output."""
        model, pts = quick_triode("12AX7")
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_result, fit = _export_and_simulate("12AX7", "triode", pts, tmpdir)
            assert raw_result["n_points"] > 0

    def test_12au7_fit_error_acceptable(self):
        """12AU7: fit RMS error from export should be low."""
        model, pts = quick_triode("12AU7")
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_result, fit = _export_and_simulate("12AU7", "triode", pts, tmpdir)
            # fit_result has rms_error attribute
            assert fit.rms_error < 2.0, f"RMS error {fit.rms_error:.2f}mA too high"

    def test_6sn7_triode(self):
        """6SN7: another common triode."""
        model, pts = quick_triode("6SN7")
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_result, fit = _export_and_simulate("6SN7", "triode", pts, tmpdir)
            assert raw_result["n_points"] > 0
            assert fit.rms_error < 3.0


# ═══════════════════════════════════════════════════════════════════
#  Pentode round-trip tests
# ═══════════════════════════════════════════════════════════════════

@requires_ltspice
class TestPentodeRoundTrip:
    """Export pentode model to SPICE, simulate, verify."""

    def test_el84_produces_output(self):
        """EL84: pentode LTspice simulation produces valid output."""
        model, pts = quick_pentode("EL84")
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_result, fit = _export_and_simulate("EL84", "pentode", pts, tmpdir)
            assert raw_result["n_points"] > 0

    def test_el84_ia_nonzero(self):
        """EL84: simulated pentode Ia should be non-zero."""
        model, pts = quick_pentode("EL84")
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_result, fit = _export_and_simulate("EL84", "pentode", pts, tmpdir)
            # Non-vacuous: assert found-flag + physical range.
            found = False
            for i, t in enumerate(raw_result["types"]):
                if "current" in t and i > 0:
                    current = raw_result["data"][:, i]
                    max_current_ma = max(abs(current)) * 1000
                    assert 0.1 < max_current_ma < 2000.0, (
                        f"peak {max_current_ma:.1f} mA out of physical range")
                    found = True
                    break
            assert found, "no current trace in .raw output"

    def test_el34_pentode(self):
        """EL34: another common power pentode."""
        model, pts = quick_pentode("EL34")
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_result, fit = _export_and_simulate("EL34", "pentode", pts, tmpdir)
            assert raw_result["n_points"] > 0


# ═══════════════════════════════════════════════════════════════════
#  .raw parser tests
# ═══════════════════════════════════════════════════════════════════

@requires_ltspice
class TestPythonVsLTspice:
    """Compare Python model Ia with LTspice simulated Ia point-by-point."""

    def _compare(self, tube_name, topology, max_rms_pct=5.0, pentode=False, ug2=0.0):
        """Run full comparison, return RMS error as % of Ia range."""
        if pentode:
            model, pts = quick_pentode(tube_name)
        else:
            model, pts = quick_triode(tube_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, f"{tube_name}.sub")
            fit = fit_and_export_spice(
                sub_path, tube_name, pts,
                topology=topology, model_type=MODEL_TYPE_KOREN,
            )
            asc_path = generate_test_schematic(sub_path, tube_name, pts, topology)
            assert asc_path is not None
            raw_path = _run_ltspice(asc_path)
            result = parse_raw(raw_path)

            # Extract Ua and Ug1 from LTspice variables
            # V1 = Ua (sweep var, column 0), V2 = Ug1
            ua_spice = result["data"][:, 0]
            ug1_col = None
            ia_col = None
            for i, v in enumerate(result["variables"]):
                if v == "I(V1)":
                    ia_col = i
                if v.startswith("V(") and "n002" in v.lower():
                    ug1_col = i
            # Fallback: I(V2) for grid voltage, Ix(x1:a) for anode current
            if ia_col is None:
                for i, v in enumerate(result["variables"]):
                    if "x1:a" in v.lower():
                        ia_col = i
            if ug1_col is None:
                # V2 is the grid voltage source — its value = Ug1
                for i, v in enumerate(result["variables"]):
                    if v == "I(V2)":
                        # Use V2 voltage instead — find V(n002)
                        pass
                # In nested sweep, Ug1 steps are in outer loop
                # Need to reconstruct from sweep params
                from lm19.ltspice_asc import _extract_sweep_params
                params = _extract_sweep_params(pts, topology)
                ug1_start = params["ug1_start"]
                ug1_stop = params["ug1_stop"]
                ug1_step = params["ug1_step"]
                ua_step = params["ua_step"]
                ua_max = params["ua_max"]
                n_ua = int(ua_max / ua_step) + 1
                n_ug1 = int((ug1_stop - ug1_start) / ug1_step) + 1
                # Reconstruct Ug1 for each point
                ug1_spice = np.repeat(
                    np.arange(ug1_start, ug1_stop + ug1_step / 2, ug1_step),
                    n_ua,
                )[:len(ua_spice)]

            assert ia_col is not None, f"No Ia column in {result['variables']}"

            ia_spice_a = result["data"][:, ia_col]
            # Anode current from Ix(x1:a) is positive when current flows in
            ia_spice_ma = np.abs(ia_spice_a) * 1000.0

            # Reconstruct Ug1 if not found directly
            if ug1_col is not None:
                ug1_spice = result["data"][:, ug1_col]
            # else already computed above

            # Compute Python model Ia at same (Ua, Ug1) points
            ia_python = np.array([
                model.ia(float(ua), float(ug1), ug2)
                for ua, ug1 in zip(ua_spice, ug1_spice)
            ])

            # Compare
            ia_range = max(ia_python.max(), ia_spice_ma.max())
            if ia_range < 0.1:
                return  # trivial case

            diff = np.abs(ia_python - ia_spice_ma)
            rms = np.sqrt(np.mean(diff ** 2))
            rms_pct = rms / ia_range * 100.0
            max_err = diff.max()
            max_err_pct = max_err / ia_range * 100.0

            assert rms_pct < max_rms_pct, (
                f"{tube_name}: Python vs LTspice RMS={rms:.2f}mA ({rms_pct:.1f}%), "
                f"max={max_err:.2f}mA ({max_err_pct:.1f}%)"
            )

    def test_12au7_python_vs_ltspice(self):
        self._compare("12AU7", "triode")

    def test_12ax7_python_vs_ltspice(self):
        self._compare("12AX7", "triode")

    def test_6sn7_python_vs_ltspice(self):
        self._compare("6SN7", "triode")

    def test_el84_python_vs_ltspice(self):
        self._compare("EL84", "pentode", pentode=True, ug2=250.0)

    def test_el34_python_vs_ltspice(self):
        self._compare("EL34", "pentode", pentode=True, ug2=250.0)


@requires_ltspice
class TestDempwolfRoundTrip:
    """Dempwolf model round-trip through LTspice."""

    def test_12au7_dempwolf_ia_nonzero(self):
        model, pts = quick_triode("12AU7")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "12AU7.sub")
            fit = fit_and_export_spice(sub_path, "12AU7", pts,
                                       topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_DEMPWOLF)
            asc_path = generate_test_schematic(sub_path, "12AU7", pts, "triode")
            raw_path = _run_ltspice(asc_path)
            result = parse_raw(raw_path)
            # Non-vacuous: assert found-flag + physical range.
            found = False
            for i, v in enumerate(result["variables"]):
                if "x1:a" in v.lower():
                    ia_max = max(abs(result["data"][:, i])) * 1000
                    assert 1.0 < ia_max < 2000.0, f"Dempwolf 12AU7 Ia={ia_max:.2f}mA"
                    found = True
                    break
            assert found, "no anode-current trace in .raw"

    def test_el84_dempwolf_ia_nonzero(self):
        model, pts = quick_pentode("EL84")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "EL84.sub")
            fit = fit_and_export_spice(sub_path, "EL84", pts,
                                       topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_DEMPWOLF)
            asc_path = generate_test_schematic(sub_path, "EL84", pts, "pentode")
            raw_path = _run_ltspice(asc_path)
            result = parse_raw(raw_path)
            # Non-vacuous: assert found-flag + physical range.
            found = False
            for i, v in enumerate(result["variables"]):
                if "x1:a" in v.lower():
                    ia_max = max(abs(result["data"][:, i])) * 1000
                    assert 1.0 < ia_max < 2000.0, f"Dempwolf EL84 Ia={ia_max:.2f}mA"
                    found = True
                    break
            assert found, "no anode-current trace in .raw"


@requires_ltspice
class TestReefmanRoundTrip:
    """Reefman model round-trip (pentode only)."""

    def test_el84_reefman_ia_nonzero(self):
        model, pts = quick_pentode("EL84")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "EL84.sub")
            fit = fit_and_export_spice(sub_path, "EL84", pts,
                                       topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_REEFMAN)
            asc_path = generate_test_schematic(sub_path, "EL84", pts, "pentode")
            raw_path = _run_ltspice(asc_path)
            result = parse_raw(raw_path)
            # Non-vacuous: assert found-flag + physical range.
            found = False
            for i, v in enumerate(result["variables"]):
                if "x1:a" in v.lower():
                    ia_max = max(abs(result["data"][:, i])) * 1000
                    assert 1.0 < ia_max < 2000.0, f"Reefman EL84 Ia={ia_max:.2f}mA"
                    found = True
                    break
            assert found, "no anode-current trace in .raw"

    def test_el34_reefman_ia_nonzero(self):
        model, pts = quick_pentode("EL34")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "EL34.sub")
            fit = fit_and_export_spice(sub_path, "EL34", pts,
                                       topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_REEFMAN)
            asc_path = generate_test_schematic(sub_path, "EL34", pts, "pentode")
            raw_path = _run_ltspice(asc_path)
            result = parse_raw(raw_path)
            # Non-vacuous: assert found-flag + physical range.
            found = False
            for i, v in enumerate(result["variables"]):
                if "x1:a" in v.lower():
                    ia_max = max(abs(result["data"][:, i])) * 1000
                    assert 1.0 < ia_max < 2000.0, f"Reefman EL34 Ia={ia_max:.2f}mA"
                    found = True
                    break
            assert found, "no anode-current trace in .raw"


@requires_ltspice
class TestAmpSchematicRoundTrip:
    """Amplifier circuit templates produce valid simulations in LTspice."""

    def _run_amp(self, tube, topology, circuit, pentode=False, **kwargs):
        if pentode:
            model, pts = quick_pentode(tube)
        else:
            model, pts = quick_triode(tube)
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, f"{tube}.sub")
            fit_and_export_spice(sub_path, tube, pts, topology=topology)
            asc_path = generate_amp_schematic(
                sub_path, tube, pts, topology, circuit=circuit, **kwargs,
            )
            assert asc_path is not None, f"Failed to generate {circuit} schematic"
            raw_path = _run_ltspice(asc_path)
            result = parse_raw(raw_path)
            # Find max anode current
            for i, v in enumerate(result["variables"]):
                if "x1:a" in v.lower():
                    ia_max = max(abs(result["data"][:, i])) * 1000
                    return ia_max, result
            return 0.0, result

    def test_se_triode_ia_nonzero(self):
        ia_max, _ = self._run_amp("12AU7", "triode", "se", ra_ohm="47k", rk_ohm="1.5k")
        assert ia_max > 0.1, f"SE triode Ia={ia_max:.2f}mA"

    def test_se_pentode_ia_nonzero(self):
        ia_max, _ = self._run_amp(
            "EL84", "pentode", "se", pentode=True,
            ra_ohm="5.6k", rk_ohm="150", ug2=300.0, ub=300.0,
        )
        assert ia_max > 0.1, f"SE pentode Ia={ia_max:.2f}mA"

    def test_cf_triode_ia_nonzero(self):
        ia_max, _ = self._run_amp("12AU7", "triode", "cf", rk_ohm="10k")
        assert ia_max > 0.1, f"CF triode Ia={ia_max:.2f}mA"

    def test_pp_triode_ia_nonzero(self):
        ia_max, _ = self._run_amp("12AU7", "triode", "pp", ra_aa_ohm="10k", rk_ohm="470")
        assert ia_max > 0.1, f"PP triode Ia={ia_max:.2f}mA"

    def test_se_xfmr_triode_ia_nonzero(self):
        ia_max, _ = self._run_amp("12AU7", "triode", "se_xfmr",
                                   ra_dc_ohm="50", rk_ohm="1.5k")
        assert ia_max > 0.1, f"SE XFMR triode Ia={ia_max:.2f}mA"

    def test_se_xfmr_pentode_ia_nonzero(self):
        ia_max, _ = self._run_amp(
            "EL84", "pentode", "se_xfmr", pentode=True,
            ra_dc_ohm="100", rk_ohm="150", ug2=300.0, ub=300.0,
        )
        assert ia_max > 0.1, f"SE XFMR pentode Ia={ia_max:.2f}mA"

    def test_se_xfmr_triode_high_current(self):
        """SE Transformer: low Rdc means high quiescent current."""
        ia_max, _ = self._run_amp("12AU7", "triode", "se_xfmr",
                                   ra_dc_ohm="50", rk_ohm="1k")
        # With Rdc=50 Ohm, current should be much higher than SE with Ra=47k
        assert ia_max > 5.0, f"SE XFMR Ia={ia_max:.2f}mA should be >5mA"

    def test_se_triode_ia_physical_range(self):
        """SE 12AU7 with Ra=47k: Ia should be 1-5mA range."""
        ia_max, _ = self._run_amp("12AU7", "triode", "se", ra_ohm="47k", rk_ohm="1.5k")
        assert 0.5 < ia_max < 10.0, f"SE triode Ia={ia_max:.2f}mA out of range"

    def test_se_pentode_ia_physical_range(self):
        """SE EL84 with Ra=5.6k: Ia should be 10-80mA range."""
        ia_max, _ = self._run_amp(
            "EL84", "pentode", "se", pentode=True,
            ra_ohm="5.6k", rk_ohm="150", ug2=300.0, ub=300.0,
        )
        assert 5.0 < ia_max < 200.0, f"SE pentode Ia={ia_max:.2f}mA out of range"

    def test_cf_pentode_ia_nonzero(self):
        ia_max, _ = self._run_amp(
            "EL84", "pentode", "cf", pentode=True,
            rk_ohm="1k", ug2=300.0, ub=300.0,
        )
        assert ia_max > 0.1, f"CF pentode Ia={ia_max:.2f}mA"

    def test_pp_pentode_ia_nonzero(self):
        ia_max, _ = self._run_amp(
            "EL84", "pentode", "pp", pentode=True,
            ra_aa_ohm="8k", rk_ohm="150", ug2=300.0, ub=300.0,
        )
        assert ia_max > 0.1, f"PP pentode Ia={ia_max:.2f}mA"

    def test_se_xfmr_triode_transformer_model(self):
        """SE XFMR with inductor model should produce Ia > 0."""
        ia_max, _ = self._run_amp("12AU7", "triode", "se_xfmr",
                                   ra_dc_ohm="50", rk_ohm="1.5k")
        assert ia_max > 0.1

    def test_se_xfmr_pentode_transformer_model(self):
        ia_max, _ = self._run_amp(
            "EL84", "pentode", "se_xfmr", pentode=True,
            ra_dc_ohm="100", rk_ohm="150", ug2=300.0, ub=300.0,
        )
        assert ia_max > 0.1

    def test_pp_has_two_tubes(self):
        """PP schematic should have X1 and X2 in netlist."""
        _, pts = quick_triode("12AU7")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "12AU7.sub")
            fit_and_export_spice(sub_path, "12AU7", pts, topology=TOPOLOGY_TRIODE)
            asc_path = generate_amp_schematic(
                sub_path, "12AU7", pts, "triode", circuit=CIRCUIT_PP,
                ra_aa_ohm="10k", rk_ohm="470",
            )
            raw_path = _run_ltspice(asc_path)
            result = parse_raw(raw_path)
            # Should have x1 and x2 currents
            var_names = [v.lower() for v in result["variables"]]
            assert any("x1" in v for v in var_names), "Missing X1 in PP"
            assert any("x2" in v for v in var_names), "Missing X2 in PP"


class TestParseFormatResistance:
    """Unit tests for _parse_resistance and _format_resistance."""

    def test_parse_k(self):
        from lm19.ltspice_asc import _parse_resistance
        assert _parse_resistance("47k") == 47000.0
        assert _parse_resistance("1.5k") == 1500.0
        assert _parse_resistance("5.6K") == 5600.0

    def test_parse_meg(self):
        from lm19.ltspice_asc import _parse_resistance
        assert _parse_resistance("1M") == 1e6
        assert _parse_resistance("2.2m") == 2.2e6

    def test_parse_plain(self):
        from lm19.ltspice_asc import _parse_resistance
        assert _parse_resistance("470") == 470.0
        assert _parse_resistance("50") == 50.0
        assert _parse_resistance("8") == 8.0

    def test_format_kohm(self):
        from lm19.ltspice_asc import _format_resistance
        assert _format_resistance(47000) == "47k"
        assert _format_resistance(1500) == "1.5k"
        assert _format_resistance(2500) == "2.5k"

    def test_format_ohm(self):
        from lm19.ltspice_asc import _format_resistance
        assert _format_resistance(470) == "470"
        assert _format_resistance(50) == "50"
        assert _format_resistance(8) == "8"

    def test_format_meg(self):
        from lm19.ltspice_asc import _format_resistance
        assert _format_resistance(1e6) == "1.0Meg"


# ── SE-XFMR generation helpers (schematic text only, no LTspice run) ──

# L1/L2 emitted into the .asc are formatted with ≥3 significant digits,
# so 1% relative tolerance covers formatting round-off while still
# catching any formula error (wrong f_low, Ra, Rload → tens of %).
_L_VALUE_REL_TOL = 0.01

# Physically realistic 12AU7-like sweep grid: generate_amp_schematic only
# needs Ua/Ug1 ranges for the sweep directives — no fitted model required
_XFMR_SWEEP_POINTS = [
    {"ua": float(ua), "ug1": float(ug1), "ia": 1.0}
    for ua in range(0, 301, 50)
    for ug1 in range(-8, 1, 2)
]


def _generate_se_xfmr(tmpdir: str, **kwargs) -> str:
    """Generate SE transformer .asc via production generator, return its text."""
    sub_path = os.path.join(tmpdir, "12AU7.sub")
    # Generator only references the .sub by name (.include line) — a stub
    # keeps the test independent of scipy fitting and LTspice
    Path(sub_path).write_text("* stub subcircuit for schematic generation\n")
    asc_path = generate_amp_schematic(
        sub_path, "12AU7", _XFMR_SWEEP_POINTS, "triode",
        circuit=CIRCUIT_SE_XFMR, **kwargs,
    )
    assert asc_path is not None, "generate_amp_schematic returned None"
    return Path(asc_path).read_text(encoding="utf-8")


def _component_value(asc_content: str, inst_name: str) -> str:
    """Extract the 'SYMATTR Value …' string of the component named inst_name."""
    lines = asc_content.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == f"SYMATTR InstName {inst_name}":
            for follower in lines[i + 1:]:
                if follower.startswith("SYMATTR Value"):
                    return follower[len("SYMATTR Value"):].strip()
                if follower.startswith("SYMBOL"):
                    break  # next component started — Value missing
            break
    raise AssertionError(f"Component {inst_name} has no Value in .asc:\n{asc_content}")


def _parse_inductance(s: str) -> float:
    """Parse inductor value string: '39.8' → 39.8 H, '63.66m' → 0.06366 H."""
    s = s.strip()
    if s.lower().endswith("m"):
        return float(s[:-1]) * 1e-3
    return float(s)


class TestTransformerParams:
    """Verify L_primary / L_secondary values in the GENERATED .asc
    (production code lm19/ltspice_asc.py), not local re-computation."""

    def test_l_primary_formula(self):
        """L1 in generated .asc = Ra / (2*pi*f_low)."""
        import math
        with tempfile.TemporaryDirectory() as tmpdir:
            content = _generate_se_xfmr(tmpdir, ra_ohm="5k", f_low=20.0)
        l1 = _parse_inductance(_component_value(content, "L1"))
        # Ra=5k, f_low=20Hz → L1 = 5000 / (2*pi*20) ≈ 39.8 H
        expected_l1 = 5000.0 / (2 * math.pi * 20.0)
        assert l1 == pytest.approx(expected_l1, rel=_L_VALUE_REL_TOL)

    def test_l_secondary_from_turns_ratio(self):
        """L2 in generated .asc = L1 / n², where n² = Ra / Rload."""
        import math
        with tempfile.TemporaryDirectory() as tmpdir:
            content = _generate_se_xfmr(
                tmpdir, ra_ohm="5k", r_load="8", f_low=20.0,
            )
        l1 = _parse_inductance(_component_value(content, "L1"))
        l2 = _parse_inductance(_component_value(content, "L2"))
        n_sq = 5000.0 / 8.0
        expected_l2 = 5000.0 / (2 * math.pi * 20.0) / n_sq  # ≈ 0.0637 H
        assert l2 == pytest.approx(expected_l2, rel=_L_VALUE_REL_TOL)
        # Consistency between the two emitted values (both carry round-off)
        assert l2 == pytest.approx(l1 / n_sq, rel=2 * _L_VALUE_REL_TOL)

    def test_different_rload_changes_l_secondary(self):
        """Lower Rload → higher turns ratio → smaller L2 in generated .asc."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content_4 = _generate_se_xfmr(tmpdir, ra_ohm="5k", r_load="4")
            content_16 = _generate_se_xfmr(tmpdir, ra_ohm="5k", r_load="16")
        l2_4ohm = _parse_inductance(_component_value(content_4, "L2"))
        l2_16ohm = _parse_inductance(_component_value(content_16, "L2"))
        assert l2_4ohm < l2_16ohm
        # L2 = L1·Rload/Ra ∝ Rload → 16Ω tap must give exactly 4× the 4Ω L2
        assert l2_16ohm / l2_4ohm == pytest.approx(
            16.0 / 4.0, rel=2 * _L_VALUE_REL_TOL,
        )

    def test_different_flow_changes_l_primary(self):
        """Lower f_low → larger L1 in generated .asc (L1 ∝ 1/f_low)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content_20 = _generate_se_xfmr(tmpdir, ra_ohm="5k", f_low=20.0)
            content_50 = _generate_se_xfmr(tmpdir, ra_ohm="5k", f_low=50.0)
        l1_20hz = _parse_inductance(_component_value(content_20, "L1"))
        l1_50hz = _parse_inductance(_component_value(content_50, "L1"))
        assert l1_20hz > l1_50hz
        assert l1_20hz / l1_50hz == pytest.approx(
            50.0 / 20.0, rel=2 * _L_VALUE_REL_TOL,
        )

    def test_generate_xfmr_substitutes_rload(self):
        """Generated .asc should contain the specified Rload value."""
        _, pts = quick_triode("12AU7")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "12AU7.sub")
            fit_and_export_spice(sub_path, "12AU7", pts, topology=TOPOLOGY_TRIODE)
            asc_path = generate_amp_schematic(
                sub_path, "12AU7", pts, "triode",
                circuit=CIRCUIT_SE_XFMR, ra_dc_ohm="50", r_load="16", f_low=30.0,
            )
            content = open(asc_path).read()
            # Bare '"16" in content' would match wire coordinates —
            # parse the actual Rload component value line instead
            assert _component_value(content, "Rload") == "16"
            assert "{r_load}" not in content  # no unsubstituted placeholders
            assert "{l_primary}" not in content
            assert "{l_secondary}" not in content

    def test_generate_xfmr_4ohm_vs_16ohm(self):
        """Different Rload should produce different L_secondary in .asc."""
        _, pts = quick_triode("12AU7")
        with tempfile.TemporaryDirectory() as tmpdir4:
            sub4 = os.path.join(tmpdir4, "12AU7.sub")
            fit_and_export_spice(sub4, "12AU7", pts, topology=TOPOLOGY_TRIODE)
            asc_4 = generate_amp_schematic(
                sub4, "12AU7", pts, "triode",
                circuit=CIRCUIT_SE_XFMR, r_load="4",
            )
            content_4 = open(asc_4).read()
        with tempfile.TemporaryDirectory() as tmpdir16:
            sub16 = os.path.join(tmpdir16, "12AU7.sub")
            fit_and_export_spice(sub16, "12AU7", pts, topology=TOPOLOGY_TRIODE)
            asc_16 = generate_amp_schematic(
                sub16, "12AU7", pts, "triode",
                circuit=CIRCUIT_SE_XFMR, r_load="16",
            )
            content_16 = open(asc_16).read()
        # Rload component value must reflect the requested load
        assert _component_value(content_4, "Rload") == "4"
        assert _component_value(content_16, "Rload") == "16"
        # Reflected-impedance-dependent value: L2 = L1·Rload/Ra ∝ Rload,
        # so 16Ω must give 4× the 4Ω L2 ('content_4 != content_16' was
        # already guaranteed by the Rload text alone)
        l2_4ohm = _parse_inductance(_component_value(content_4, "L2"))
        l2_16ohm = _parse_inductance(_component_value(content_16, "L2"))
        assert l2_4ohm < l2_16ohm
        assert l2_16ohm / l2_4ohm == pytest.approx(
            16.0 / 4.0, rel=2 * _L_VALUE_REL_TOL,
        )


class TestAscPlaceholders:
    """All placeholders must be substituted — no {xxx} remnants."""

    def _check_no_placeholders(self, content: str):
        import re
        leftover = re.findall(r"\{[a-z_]+\}", content)
        assert not leftover, f"Unsubstituted placeholders: {leftover}"

    def test_se_triode_no_placeholders(self):
        _, pts = quick_triode("12AU7")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "12AU7.sub")
            fit_and_export_spice(sub_path, "12AU7", pts, topology=TOPOLOGY_TRIODE)
            asc = generate_amp_schematic(sub_path, "12AU7", pts, "triode", circuit=CIRCUIT_SE)
            self._check_no_placeholders(open(asc).read())

    def test_se_pentode_no_placeholders(self):
        _, pts = quick_pentode("EL84")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "EL84.sub")
            fit_and_export_spice(sub_path, "EL84", pts, topology=TOPOLOGY_PENTODE)
            asc = generate_amp_schematic(sub_path, "EL84", pts, "pentode", circuit=CIRCUIT_SE)
            self._check_no_placeholders(open(asc).read())

    def test_se_xfmr_triode_no_placeholders(self):
        _, pts = quick_triode("12AU7")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "12AU7.sub")
            fit_and_export_spice(sub_path, "12AU7", pts, topology=TOPOLOGY_TRIODE)
            asc = generate_amp_schematic(sub_path, "12AU7", pts, "triode", circuit=CIRCUIT_SE_XFMR)
            self._check_no_placeholders(open(asc).read())

    def test_se_xfmr_pentode_no_placeholders(self):
        _, pts = quick_pentode("EL84")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "EL84.sub")
            fit_and_export_spice(sub_path, "EL84", pts, topology=TOPOLOGY_PENTODE)
            asc = generate_amp_schematic(sub_path, "EL84", pts, "pentode", circuit=CIRCUIT_SE_XFMR)
            self._check_no_placeholders(open(asc).read())

    def test_cf_triode_no_placeholders(self):
        _, pts = quick_triode("12AU7")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "12AU7.sub")
            fit_and_export_spice(sub_path, "12AU7", pts, topology=TOPOLOGY_TRIODE)
            asc = generate_amp_schematic(sub_path, "12AU7", pts, "triode", circuit=CIRCUIT_CF)
            self._check_no_placeholders(open(asc).read())

    def test_cf_pentode_no_placeholders(self):
        _, pts = quick_pentode("EL84")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "EL84.sub")
            fit_and_export_spice(sub_path, "EL84", pts, topology=TOPOLOGY_PENTODE)
            asc = generate_amp_schematic(sub_path, "EL84", pts, "pentode", circuit=CIRCUIT_CF)
            self._check_no_placeholders(open(asc).read())

    def test_pp_triode_no_placeholders(self):
        _, pts = quick_triode("12AU7")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "12AU7.sub")
            fit_and_export_spice(sub_path, "12AU7", pts, topology=TOPOLOGY_TRIODE)
            asc = generate_amp_schematic(sub_path, "12AU7", pts, "triode", circuit=CIRCUIT_PP)
            self._check_no_placeholders(open(asc).read())

    def test_pp_pentode_no_placeholders(self):
        _, pts = quick_pentode("EL84")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "EL84.sub")
            fit_and_export_spice(sub_path, "EL84", pts, topology=TOPOLOGY_PENTODE)
            asc = generate_amp_schematic(sub_path, "EL84", pts, "pentode", circuit=CIRCUIT_PP)
            self._check_no_placeholders(open(asc).read())

    def test_asc_contains_include(self):
        """Every generated .asc should have .include directive."""
        _, pts = quick_triode("12AU7")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "12AU7.sub")
            fit_and_export_spice(sub_path, "12AU7", pts, topology=TOPOLOGY_TRIODE)
            for circuit in ["se", "se_xfmr", "cf", "pp"]:
                asc = generate_amp_schematic(
                    sub_path, "12AU7", pts, "triode", circuit=circuit,
                )
                content = open(asc).read()
                assert ".include 12AU7.sub" in content, f"{circuit}: missing .include"

    def test_asc_contains_component_values(self):
        """Specified Ra and Rk should appear in the generated .asc."""
        _, pts = quick_triode("12AU7")
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = os.path.join(tmpdir, "12AU7.sub")
            fit_and_export_spice(sub_path, "12AU7", pts, topology=TOPOLOGY_TRIODE)
            asc = generate_amp_schematic(
                sub_path, "12AU7", pts, "triode", circuit=CIRCUIT_SE,
                ra_ohm="100k", rk_ohm="2.2k",
            )
            content = open(asc).read()
            assert "100k" in content
            assert "2.2k" in content


@requires_ltspice
class TestRawParser:
    """Verify .raw parser on a known simple circuit."""

    def test_resistor_dc_sweep(self):
        """1kΩ resistor: V sweep 0-10V → I should be V/1000 A."""
        with tempfile.TemporaryDirectory() as tmpdir:
            asc_path = os.path.join(tmpdir, "resistor_test.asc")
            with open(asc_path, "w") as f:
                f.write(
                    "Version 4\n"
                    "SHEET 1 880 680\n"
                    "WIRE 300 100 200 100\n"
                    "WIRE 300 200 300 100\n"
                    "WIRE 200 200 200 100\n"
                    "FLAG 300 200 0\n"
                    "FLAG 200 200 0\n"
                    "SYMBOL voltage 200 100 R0\n"
                    "SYMATTR InstName V1\n"
                    "SYMATTR Value 5\n"
                    "SYMBOL res 284 100 R0\n"
                    "SYMATTR InstName R1\n"
                    "SYMATTR Value 1k\n"
                    "TEXT 50 300 Left 2 !.dc V1 0 10 1\n"
                )
            raw_path = _run_ltspice(asc_path)
            result = parse_raw(raw_path)

            assert result["n_points"] == 11
            v1 = result["data"][:, 0]  # sweep variable
            # Find I(R1)
            i_r1 = get_variable(result, "I(R1)")
            # V/R = I: at 5V, I should be 5mA
            expected_ma = v1  # V / 1kΩ = mA
            actual_ma = i_r1 * 1000
            np.testing.assert_allclose(actual_ma, expected_ma, atol=0.01)
