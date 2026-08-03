"""Unit tests for SPICE model fitting and export.

Datasets: "Next-Tube datasets", "loadline_plotter datasets",
"pypsucurvetrace / curvetracedata" — see SOURCES_INDEX.md.

Run:  py -m pytest tests/test_spice_export.py -v
  or: py -m unittest tests.test_spice_export -v

Covers:
  - Koren triode model equation verification
  - Koren pentode model equation verification
  - Triode fitting on Tuparam + loadline_plotter datasets
  - Pentode fitting on Tuparam datasets
  - SPICE subcircuit output format validation
  - Full pipeline (fit_and_export_spice) end-to-end

Test data: tests/spice_test_data/converted/*.json
"""

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Ensure the app root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.spice_export import (
    _koren_ia,
    _koren_ia_pentode,
    _koren_ig2_pentode,
    _fit_koren_scipy,
    _fit_koren_numpy,
    _fit_pentode_scipy,
    _fit_pentode_numpy,
    _generate_triode_subcircuit,
    _generate_pentode_subcircuit,
    fit_and_export_spice,
    SpiceFitResult,
    _HAS_SCIPY,
)
from lm19.tube_params import TubeRefParams, TubeCaps, KorenParams
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "spice_test_data" / "converted"


def _load(filename):
    """Load a JSON test data file."""
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


def _triode_arrays(data):
    """Extract ua, ug1, ia (Amps) arrays from triode JSON data."""
    pts = data["points"]
    ua = np.array([p["ua"] for p in pts], dtype=float)
    ug1 = np.array([p["ug1"] for p in pts], dtype=float)
    ia = np.array([p["ia"] for p in pts], dtype=float) / 1000.0  # mA → A
    return ua, ug1, ia


def _pentode_arrays(data):
    """Extract ua, ug1, ug2, ia (Amps), ig2 arrays from pentode JSON data."""
    pts = data["points"]
    ua = np.array([p["ua"] for p in pts], dtype=float)
    ug1 = np.array([p["ug1"] for p in pts], dtype=float)
    ug2 = np.array([p["ug2"] for p in pts], dtype=float)
    ia = np.array([p["ia"] for p in pts], dtype=float) / 1000.0  # mA → A
    ig2_raw = np.array([p.get("ig2") or 0.0 for p in pts], dtype=float) / 1000.0
    has_ig2 = bool(np.any(ig2_raw > 1e-5))
    return ua, ug1, ug2, ia, ig2_raw if has_ig2 else None


def _rms_pct(ia_pred, ia_meas):
    """Compute RMS error as percentage of max measured Ia."""
    diff = ia_pred - ia_meas
    rms = float(np.sqrt(np.mean(diff ** 2)))
    ia_max = float(np.max(np.abs(ia_meas)))
    if ia_max < 1e-9:
        return 100.0
    return 100.0 * rms / ia_max


# ======================================================================
# 1. Model equation verification
# ======================================================================

class TestKorenTriodeModel(unittest.TestCase):
    """Verify triode model against hand-calculated values from SPICE_KOREN_MODELS.md."""

    def test_12ax7_point1(self):
        """12AX7: Ua=250, Ug1=-2 → Ia ≈ 0.95 mA (from doc section 14)."""
        # Published params: mu=100, Ex=1.4, Kg1=1060, Kp=600, Kvb=300
        ia = _koren_ia(
            np.array([250.0]), np.array([-2.0]),
            mu=100.0, ex=1.4, kg1=1060.0, kp=600.0, kvb=300.0,
        )
        ia_mA = float(ia[0]) * 1000.0
        self.assertAlmostEqual(ia_mA, 0.95, delta=0.15,
                               msg=f"12AX7 @250V/-2V: expected ~0.95 mA, got {ia_mA:.3f}")

    def test_12ax7_point2(self):
        """12AX7: Ua=100, Ug1=-1 → Ia ≈ 0.10 mA."""
        ia = _koren_ia(
            np.array([100.0]), np.array([-1.0]),
            mu=100.0, ex=1.4, kg1=1060.0, kp=600.0, kvb=300.0,
        )
        ia_mA = float(ia[0]) * 1000.0
        self.assertAlmostEqual(ia_mA, 0.10, delta=0.05,
                               msg=f"12AX7 @100V/-1V: expected ~0.10 mA, got {ia_mA:.3f}")

    def test_12au7_typical(self):
        """12AU7: Ua=250, Ug1=0 → reasonable current (5-50 mA range)."""
        ia = _koren_ia(
            np.array([250.0]), np.array([0.0]),
            mu=21.5, ex=1.3, kg1=1180.0, kp=84.0, kvb=300.0,
        )
        ia_mA = float(ia[0]) * 1000.0
        self.assertTrue(5.0 < ia_mA < 50.0,
                        msg=f"12AU7 @250V/0V: got {ia_mA:.2f} mA, expected 5-50 range")

    def test_cutoff_region(self):
        """Deep cutoff should give Ia ≈ 0."""
        ia = _koren_ia(
            np.array([100.0]), np.array([-50.0]),
            mu=100.0, ex=1.4, kg1=1060.0, kp=600.0, kvb=300.0,
        )
        ia_mA = float(ia[0]) * 1000.0
        self.assertLess(ia_mA, 0.001, msg=f"Cutoff: expected ~0, got {ia_mA:.6f}")

    def test_zero_plate_voltage(self):
        """Ua ≈ 0 should give Ia ≈ 0 (no negative current)."""
        ia = _koren_ia(
            np.array([0.0]), np.array([0.0]),
            mu=100.0, ex=1.4, kg1=1060.0, kp=600.0, kvb=300.0,
        )
        self.assertGreaterEqual(float(ia[0]), 0.0)

    def test_vectorized(self):
        """Model should handle array inputs."""
        ua = np.array([100.0, 200.0, 300.0])
        ug1 = np.array([0.0, -1.0, -2.0])
        ia = _koren_ia(ua, ug1, 100.0, 1.4, 1060.0, 600.0, 300.0)
        self.assertEqual(ia.shape, (3,))
        self.assertTrue(np.all(ia >= 0))
        # Ia should generally increase with Ua for these conditions
        # (not strictly monotone due to grid bias, but first > last is unlikely)


class TestKorenPentodeModel(unittest.TestCase):
    """Verify pentode model against hand-calculated values."""

    def test_6550_reference_point(self):
        """6550: Ua=300, Ug2=300, Ug1=0 → Ia ≈ 454 mA (from doc section 14)."""
        ia = _koren_ia_pentode(
            np.array([300.0]), np.array([0.0]), np.array([300.0]),
            mu=7.9, ex=1.35, kg1=890.0, kp=60.0, kvb=24.0,
        )
        ia_mA = float(ia[0]) * 1000.0
        self.assertAlmostEqual(ia_mA, 454.0, delta=30.0,
                               msg=f"6550 @300/300/0: expected ~454 mA, got {ia_mA:.1f}")

    def test_6550_ig2(self):
        """6550: Ug2=300, Ug1=0 → Ig2 ≈ 32 mA (from doc section 14)."""
        ig2 = _koren_ig2_pentode(
            np.array([0.0]), np.array([300.0]),
            mu=7.9, ex=1.35, kg2=4200.0,
        )
        ig2_mA = float(ig2[0]) * 1000.0
        self.assertAlmostEqual(ig2_mA, 32.0, delta=5.0,
                               msg=f"6550 Ig2 @300/0: expected ~32 mA, got {ig2_mA:.1f}")

    def test_pentode_knee(self):
        """Low Ua (knee region) should give reduced current via arctan."""
        ia_high = _koren_ia_pentode(
            np.array([300.0]), np.array([0.0]), np.array([300.0]),
            mu=7.9, ex=1.35, kg1=890.0, kp=60.0, kvb=24.0,
        )
        ia_low = _koren_ia_pentode(
            np.array([20.0]), np.array([0.0]), np.array([300.0]),
            mu=7.9, ex=1.35, kg1=890.0, kp=60.0, kvb=24.0,
        )
        # At Ua=20V the arctan factor should be much less than at 300V
        self.assertGreater(float(ia_high[0]), float(ia_low[0]) * 2.0,
                           msg="arctan knee: Ia@300V should be >> Ia@20V")

    def test_pentode_cutoff(self):
        """Deep negative grid should cut off pentode."""
        ia = _koren_ia_pentode(
            np.array([300.0]), np.array([-100.0]), np.array([300.0]),
            mu=7.9, ex=1.35, kg1=890.0, kp=60.0, kvb=24.0,
        )
        ia_mA = float(ia[0]) * 1000.0
        self.assertLess(ia_mA, 1.0, msg=f"Pentode cutoff: expected ~0, got {ia_mA:.3f}")

    def test_ig2_cutoff(self):
        """Deep negative grid should give near-zero Ig2."""
        ig2 = _koren_ig2_pentode(
            np.array([-100.0]), np.array([300.0]),
            mu=7.9, ex=1.35, kg2=4200.0,
        )
        ig2_mA = float(ig2[0]) * 1000.0
        self.assertLess(ig2_mA, 0.01)

    def test_pentode_vectorized(self):
        """Pentode model should handle arrays."""
        ua = np.array([100.0, 200.0, 400.0])
        ug1 = np.array([0.0, -10.0, -30.0])
        ug2 = np.array([300.0, 300.0, 300.0])
        ia = _koren_ia_pentode(ua, ug1, ug2, 7.9, 1.35, 890.0, 60.0, 24.0)
        self.assertEqual(ia.shape, (3,))
        self.assertTrue(np.all(ia >= 0))


# ======================================================================
# 2. Triode fitting tests
# ======================================================================

class TestTriodeFittingScipy(unittest.TestCase):
    """Test triode model fitting with scipy on real datasets."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_SCIPY:
            raise unittest.SkipTest("scipy not available")

    def _fit_triode(self, filename, max_rms_pct=8.0):
        """Helper: fit triode and check RMS < threshold."""
        data = _load(filename)
        ua, ug1, ia = _triode_arrays(data)
        # Filter low current
        mask = ia > 0.0001  # 0.1 mA
        ua, ug1, ia = ua[mask], ug1[mask], ia[mask]
        if len(ua) < 4:
            self.skipTest(f"Too few points after filter: {len(ua)}")
        params, cost, _ = _fit_koren_scipy(ua, ug1, ia)
        mu, ex, kg1, kp, kvb = params
        ia_pred = _koren_ia(ua, ug1, mu, ex, kg1, kp, kvb)
        rms = _rms_pct(ia_pred, ia)
        self.assertLess(rms, max_rms_pct,
                        msg=f"{filename}: RMS {rms:.1f}% > {max_rms_pct}%")
        return params, rms

    # --- Tuparam gold standard (with expected params) ---

    def test_12ax7_mitch_params(self):
        """12AX7A (Tom Mitchell): fitted params should be close to Tuparam expected."""
        data = _load("triode_12AX7AMitch_tuparam.json")
        expected = data["expected_params"]
        ua, ug1, ia = _triode_arrays(data)
        params, cost, _ = _fit_koren_scipy(ua, ug1, ia)
        mu, ex, kg1, kp, kvb = params
        # Allow ±25% tolerance (different optimizer, different initial guess)
        self.assertAlmostEqual(mu, expected["mu"], delta=expected["mu"] * 0.25,
                               msg=f"mu: {mu:.2f} vs expected {expected['mu']}")
        self.assertAlmostEqual(ex, expected["ex"], delta=expected["ex"] * 0.25,
                               msg=f"ex: {ex:.3f} vs expected {expected['ex']}")

    def test_7025_params(self):
        """7025 (Sylvania, VCT=0.5): fitted params should be reasonable."""
        data = _load("triode_7025_tuparam.json")
        expected = data["expected_params"]
        ua, ug1, ia = _triode_arrays(data)
        params, cost, _ = _fit_koren_scipy(ua, ug1, ia)
        mu, ex, kg1, kp, kvb = params
        # mu should be in the 80-130 range (expected 103.44)
        self.assertTrue(70 < mu < 150,
                        msg=f"7025 mu={mu:.1f}, expected ~103")

    # --- Tuparam datasets (accuracy check) ---

    def test_12at7(self):
        self._fit_triode("triode_12AT7_tuparam.json")

    def test_12au7(self):
        self._fit_triode("triode_12AU7_tuparam.json")

    def test_12au7a(self):
        self._fit_triode("triode_12AU7A_tuparam.json")

    def test_12ax7_syl(self):
        self._fit_triode("triode_12AX7ASYL_tuparam.json")

    def test_6dj8(self):
        self._fit_triode("triode_6DJ8Mitch_tuparam.json")

    def test_6sn7(self):
        self._fit_triode("triode_6SN7Sylv_tuparam.json")

    # --- Loadline_plotter datasets (many points, looser tolerance) ---

    def test_ecc83_datasheet(self):
        self._fit_triode("triode_ecc83_datasheet.json", max_rms_pct=10.0)

    def test_ecc82_datasheet(self):
        self._fit_triode("triode_ecc82_datasheet.json", max_rms_pct=10.0)

    def test_ecc81_datasheet(self):
        self._fit_triode("triode_ecc81_datasheet.json", max_rms_pct=10.0)

    def test_e88cc_datasheet(self):
        self._fit_triode("triode_e88cc_datasheet.json", max_rms_pct=10.0)

    def test_ecc85_datasheet(self):
        self._fit_triode("triode_ecc85_datasheet.json", max_rms_pct=10.0)

    def test_12ay7_datasheet(self):
        self._fit_triode("triode_12ay7_datasheet.json", max_rms_pct=10.0)


class TestTriodeFittingNumpy(unittest.TestCase):
    """Test triode fitting with numpy-only fallback."""

    def _fit_triode_np(self, filename, max_rms_pct=12.0):
        """Fit using numpy fallback, allow slightly looser tolerance."""
        data = _load(filename)
        ua, ug1, ia = _triode_arrays(data)
        mask = ia > 0.0001
        ua, ug1, ia = ua[mask], ug1[mask], ia[mask]
        if len(ua) < 4:
            self.skipTest(f"Too few points: {len(ua)}")
        params, cost, _ = _fit_koren_numpy(ua, ug1, ia)
        mu, ex, kg1, kp, kvb = params
        ia_pred = _koren_ia(ua, ug1, mu, ex, kg1, kp, kvb)
        rms = _rms_pct(ia_pred, ia)
        self.assertLess(rms, max_rms_pct,
                        msg=f"{filename}: numpy RMS {rms:.1f}% > {max_rms_pct}%")
        return params, rms

    def test_12ax7_mitch_numpy(self):
        """12AX7A with numpy fallback should still give reasonable fit."""
        self._fit_triode_np("triode_12AX7AMitch_tuparam.json")

    def test_ecc83_numpy(self):
        """ECC83 datasheet with numpy fallback."""
        self._fit_triode_np("triode_ecc83_datasheet.json", max_rms_pct=15.0)

    def test_ecc82_numpy(self):
        self._fit_triode_np("triode_ecc82_datasheet.json", max_rms_pct=15.0)

    def test_12au7_numpy(self):
        self._fit_triode_np("triode_12AU7_tuparam.json")


# ======================================================================
# 3. Pentode fitting tests
# ======================================================================

class TestPentodeFittingScipy(unittest.TestCase):
    """Test pentode model fitting with scipy on real datasets."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_SCIPY:
            raise unittest.SkipTest("scipy not available")

    def _fit_pentode(self, filename, max_rms_pct=10.0):
        """Helper: fit pentode and check RMS."""
        data = _load(filename)
        ua, ug1, ug2, ia, ig2 = _pentode_arrays(data)
        # Filter
        mask = (ia > 0.0001) & (ug2 > 10)
        ua, ug1, ug2, ia = ua[mask], ug1[mask], ug2[mask], ia[mask]
        if ig2 is not None:
            ig2 = ig2[mask]
        if len(ua) < 4:
            self.skipTest(f"Too few points: {len(ua)}")
        params, cost, _ = _fit_pentode_scipy(ua, ug1, ug2, ia, ig2)
        mu, ex, kg1, kp, kvb, kg2 = params
        ia_pred = _koren_ia_pentode(ua, ug1, ug2, mu, ex, kg1, kp, kvb)
        rms = _rms_pct(ia_pred, ia)
        self.assertLess(rms, max_rms_pct,
                        msg=f"{filename}: RMS {rms:.1f}% > {max_rms_pct}%")
        return params, rms

    # --- 6550 gold standard ---

    def test_6550_params(self):
        """6550 (Tung-Sol): fitted params should match Tuparam expected."""
        data = _load("pentode_6550_tuparam.json")
        expected = data["expected_params"]
        ua, ug1, ug2, ia, ig2 = _pentode_arrays(data)
        mask = (ia > 0.0001) & (ug2 > 10)
        params, cost, _ = _fit_pentode_scipy(
            ua[mask], ug1[mask], ug2[mask], ia[mask], None)
        mu, ex, kg1, kp, kvb, kg2 = params
        # mu should be in 5-15 range (expected 8.45, published 7.9)
        self.assertTrue(4.0 < mu < 20.0,
                        msg=f"6550 mu={mu:.2f}, expected ~8.45")
        # ex should be in 1.0-2.0 range (expected 1.247)
        self.assertTrue(0.8 < ex < 2.5,
                        msg=f"6550 ex={ex:.3f}, expected ~1.247")

    def test_6550_rms(self):
        self._fit_pentode("pentode_6550_tuparam.json")

    # --- Other pentode datasets ---

    def test_6550a(self):
        self._fit_pentode("pentode_6550A_tuparam.json")

    def test_6550c(self):
        self._fit_pentode("pentode_6550C_tuparam.json")

    def test_6l6gb(self):
        self._fit_pentode("pentode_6L6GB_tuparam.json")

    def test_6l6gc(self):
        self._fit_pentode("pentode_6L6GC_tuparam.json")

    def test_el34(self):
        self._fit_pentode("pentode_EL34_tuparam.json")

    def test_kt88(self):
        self._fit_pentode("pentode_KT88_tuparam.json")


class TestPentodeFittingNumpy(unittest.TestCase):
    """Test pentode fitting with numpy-only fallback."""

    def _fit_pentode_np(self, filename, max_rms_pct=15.0):
        data = _load(filename)
        ua, ug1, ug2, ia, ig2 = _pentode_arrays(data)
        mask = (ia > 0.0001) & (ug2 > 10)
        ua, ug1, ug2, ia = ua[mask], ug1[mask], ug2[mask], ia[mask]
        if len(ua) < 4:
            self.skipTest(f"Too few points: {len(ua)}")
        params, cost, _ = _fit_pentode_numpy(ua, ug1, ug2, ia, None)
        mu, ex, kg1, kp, kvb, kg2 = params
        ia_pred = _koren_ia_pentode(ua, ug1, ug2, mu, ex, kg1, kp, kvb)
        rms = _rms_pct(ia_pred, ia)
        self.assertLess(rms, max_rms_pct,
                        msg=f"{filename}: numpy RMS {rms:.1f}% > {max_rms_pct}%")

    def test_6550_numpy(self):
        self._fit_pentode_np("pentode_6550_tuparam.json")

    def test_kt88_numpy(self):
        self._fit_pentode_np("pentode_KT88_tuparam.json")

    def test_6l6gb_numpy(self):
        self._fit_pentode_np("pentode_6L6GB_tuparam.json")


# ======================================================================
# 4. Subcircuit generation tests
# ======================================================================

class TestTriodeSubcircuit(unittest.TestCase):
    """Verify generated triode SPICE subcircuit format."""

    def setUp(self):
        self.sub = _generate_triode_subcircuit(
            safe_name="TEST_12AX7", tube_type="12AX7",
            mu=100.0, ex=1.4, kg1=1060.0, kp=600.0, kvb=300.0,
            rms_error=0.5, max_error=1.2, n_points=50,
            backend="scipy",
        )

    def test_starts_with_comment(self):
        self.assertTrue(self.sub.startswith(";"))

    def test_has_subckt_header(self):
        self.assertIn(".SUBCKT TEST_12AX7 A G K", self.sub)

    def test_has_ends(self):
        self.assertIn(".ENDS TEST_12AX7", self.sub)

    def test_has_params_line(self):
        self.assertIn("MU=100.0000", self.sub)
        self.assertIn("EX=1.4000", self.sub)
        self.assertIn("KG1=1060.0000", self.sub)
        self.assertIn("KP=600.0000", self.sub)
        self.assertIn("KVB=300.0000", self.sub)

    def test_has_e1_equation(self):
        self.assertIn("E1 7 0 VALUE=", self.sub)
        self.assertIn("SQRT(KVB+V(A,K)*V(A,K))", self.sub)

    def test_has_g1_source(self):
        self.assertIn("G1 A K VALUE=", self.sub)
        self.assertIn("PWR(V(7),EX)", self.sub)

    def test_has_convergence_resistor(self):
        self.assertIn("RCP A K 1G", self.sub)

    def test_no_pentode_elements(self):
        """Triode subcircuit should not have G2 pin or ATAN."""
        self.assertNotIn("G2", self.sub.split(".ENDS")[0].split("SUBCKT")[1])
        self.assertNotIn("ATAN", self.sub)

    def test_fit_quality_comment(self):
        self.assertIn("RMS error = 0.50 mA", self.sub)
        self.assertIn("Max error = 1.20 mA", self.sub)


class TestPentodeSubcircuit(unittest.TestCase):
    """Verify generated pentode SPICE subcircuit format."""

    def setUp(self):
        self.sub = _generate_pentode_subcircuit(
            safe_name="TEST_6550", tube_type="6550",
            mu=7.9, ex=1.35, kg1=890.0, kp=60.0, kvb=24.0, kg2=4200.0,
            rms_error=5.0, max_error=15.0, n_points=30,
            backend="scipy",
        )

    def test_has_4pin_header(self):
        self.assertIn(".SUBCKT TEST_6550 A G K G2", self.sub)

    def test_has_ends(self):
        self.assertIn(".ENDS TEST_6550", self.sub)

    def test_has_all_params(self):
        self.assertIn("MU=7.9000", self.sub)
        self.assertIn("KG2=4200.0000", self.sub)
        self.assertIn("KVB=24.0000", self.sub)

    def test_has_e1_pentode_equation(self):
        """Pentode E1 uses V(G2,K)/KP and V(G,K)/V(G2,K)."""
        self.assertIn("V(G2,K)/KP", self.sub)
        self.assertIn("V(G,K)/V(G2,K)", self.sub)

    def test_has_atan_knee(self):
        """Pentode G1 should include ATAN(V(A,K)/KVB)."""
        self.assertIn("ATAN(V(A,K)/KVB)", self.sub)

    def test_has_g2_screen_current(self):
        """Should have G2 screen current source."""
        self.assertIn("G2 G2 K VALUE=", self.sub)
        self.assertIn("EXP(EX*(LOG(", self.sub)

    def test_has_convergence_resistors(self):
        self.assertIn("RCP A K 1G", self.sub)
        self.assertIn("R2 G2 K 1G", self.sub)

    def test_equations_match_koren_lib(self):
        """Pentode equations should match Koren's Tube.lib format."""
        # E1: {V(G2,K)/KP*LOG(1+EXP((1/MU+V(G,K)/V(G2,K))*KP))}
        self.assertIn("1/MU+V(G,K)/V(G2,K)", self.sub)
        # G1: {(PWR(V(7),EX)+PWRS(V(7),EX))/KG1*ATAN(V(A,K)/KVB)}
        self.assertIn("PWR(V(7),EX)+PWRS(V(7),EX))/KG1*ATAN", self.sub)
        # G2: {(EXP(EX*(LOG((V(G2,K)/MU)+V(G,K)))))/KG2}
        self.assertIn("V(G2,K)/MU)+V(G,K)", self.sub)


class TestTriodeSubcircuitWithRef(unittest.TestCase):
    """Verify triode subcircuit generation with ref params (caps, RGI, VCT)."""

    def setUp(self):
        self.ref = TubeRefParams(
            name="12AX7", aliases=["ECC83"],
            topology=TOPOLOGY_TRIODE,
            koren=KorenParams(mu=100.0, ex=1.4, kg1=1060.0, kp=600.0, kvb=300.0),
            caps=TubeCaps(ccg=1.6, cgp=1.7, ccp=0.46),
            rgi=2000, vct=0.5,
            source="Test",
        )
        self.sub = _generate_triode_subcircuit(
            safe_name="X_12AX7", tube_type="12AX7",
            mu=100.0, ex=1.4, kg1=1060.0, kp=600.0, kvb=300.0,
            rms_error=0.3, max_error=0.8, n_points=40,
            backend="scipy", ref=self.ref,
        )

    def test_has_vct_param(self):
        """VCT should appear in PARAMS line."""
        self.assertIn("VCT=0.50", self.sub)

    def test_has_capacitance_elements(self):
        """C1, C2, C3 for interelectrode capacitances."""
        self.assertIn("C1 G K", self.sub)
        self.assertIn("C2 G A", self.sub)
        self.assertIn("C3 A K", self.sub)

    def test_cap_values_in_comments(self):
        """Capacitance pF values in comments."""
        self.assertIn("1.6 pF", self.sub)
        self.assertIn("1.7 pF", self.sub)
        self.assertIn("0.46 pF", self.sub)

    def test_has_grid_stopper(self):
        """Grid stopper resistor R1 and diode D3."""
        self.assertIn("R1 G 5", self.sub)
        self.assertIn("D3 5 K DX", self.sub)
        self.assertIn(".MODEL DX D", self.sub)

    def test_has_rgi_value(self):
        self.assertIn("2000", self.sub)

    def test_source_in_comment(self):
        self.assertIn("Reference data source: Test", self.sub)


class TestPentodeSubcircuitWithRef(unittest.TestCase):
    """Verify pentode subcircuit generation with ref params."""

    def setUp(self):
        self.ref = TubeRefParams(
            name="EL34", aliases=[],
            topology=TOPOLOGY_PENTODE,
            koren=KorenParams(mu=11.0, ex=1.35, kg1=650.0, kp=60.0, kvb=24.0, kg2=4200.0),
            caps=TubeCaps(ccg1=15.0, ccg2=8.0, cpg1=0.7, cg1g2=0.5, ccp=8.0),
            rgi=1000, vct=0.0,
        )
        self.sub = _generate_pentode_subcircuit(
            safe_name="X_EL34", tube_type="EL34",
            mu=11.0, ex=1.35, kg1=650.0, kp=60.0, kvb=24.0, kg2=4200.0,
            rms_error=5.0, max_error=12.0, n_points=30,
            backend="scipy", ig2_rms=2.5, ref=self.ref,
        )

    def test_has_pentode_capacitance_params(self):
        """Pentode subcircuit must read the PENTODE cap fields, not triode."""
        self.assertIn("CCG1=15.0P", self.sub)
        self.assertIn("CCG2=8.0P", self.sub)
        self.assertIn("CPG1=0.7P", self.sub)
        self.assertIn("CG1G2=0.5P", self.sub)
        self.assertIn("CCP=8.0P", self.sub)
        # Lock out the triode-field regression (ccg/cgp instead of ccg1/cpg1).
        self.assertNotIn("CCG=", self.sub)

    def test_has_cap_elements(self):
        self.assertIn("C1 G K", self.sub)
        self.assertIn("C4 G2 K", self.sub)
        self.assertIn("C5 G2 G", self.sub)
        self.assertIn("C2 A G", self.sub)
        self.assertIn("C3 A K", self.sub)

    def test_has_ig2_rms_comment(self):
        """Ig2 fit quality should appear in header comments."""
        self.assertIn("Ig2", self.sub)
        self.assertIn("2.50 mA", self.sub)

    def test_has_grid_stopper(self):
        self.assertIn("R1 G 5", self.sub)
        self.assertIn("D3 5 K DX", self.sub)

    def test_convergence_resistors_present(self):
        self.assertIn("RCP A K 1G", self.sub)
        self.assertIn("R2 G2 K 1G", self.sub)


class TestSubcircuitWithoutRef(unittest.TestCase):
    """Subcircuit without ref should NOT include caps/rgi/vct elements."""

    def test_triode_no_caps(self):
        sub = _generate_triode_subcircuit(
            safe_name="BARE", tube_type="BARE",
            mu=50.0, ex=1.3, kg1=500.0, kp=200.0, kvb=100.0,
            rms_error=1.0, max_error=2.0, n_points=20,
            backend="numpy", ref=None,
        )
        self.assertNotIn("C1 G K", sub)
        self.assertNotIn("D3 5 K", sub)
        self.assertNotIn("VCT=", sub)

    def test_pentode_no_caps(self):
        sub = _generate_pentode_subcircuit(
            safe_name="BARE_P", tube_type="BARE_P",
            mu=8.0, ex=1.35, kg1=890.0, kp=60.0, kvb=24.0, kg2=4500.0,
            rms_error=5.0, max_error=10.0, n_points=15,
            backend="numpy", ref=None,
        )
        self.assertNotIn("C1 G K", sub)
        self.assertNotIn("D3 5 K", sub)
        self.assertNotIn("CCG=", sub)


# ======================================================================
# 5. Full pipeline tests
# ======================================================================

class TestFullPipelineTriode(unittest.TestCase):
    """End-to-end test of fit_and_export_spice for triodes."""

    def test_ecc83_full_pipeline(self):
        """ECC83 datasheet (85 points) → fit_and_export_spice → .sub file."""
        data = _load("triode_ecc83_datasheet.json")
        points = data["points"]
        # Simulate the format expected by fit_and_export_spice
        # (points already have ua, ug1, ia in mA)

        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = fit_and_export_spice(
                path=tmp_path,
                tube_type="ECC83",
                points=points,
                topology=TOPOLOGY_TRIODE,
            )

            # Check result type
            self.assertIsInstance(result, SpiceFitResult)
            self.assertEqual(result.model_type, "triode")
            self.assertIn("mu", result.params)
            self.assertIn("kvb", result.params)
            self.assertNotIn("kg2", result.params)
            self.assertGreater(result.n_points, 20)
            self.assertLess(result.rms_error, 2.0,
                            msg=f"ECC83 pipeline RMS={result.rms_error:.2f} mA")

            # Check file was written and is valid SPICE
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn(".SUBCKT", content)
            self.assertIn(".ENDS", content)
            self.assertIn("ECC83", content)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_ecc82_full_pipeline(self):
        """ECC82 datasheet → full pipeline."""
        data = _load("triode_ecc82_datasheet.json")

        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = fit_and_export_spice(
                tmp_path, "ECC82", data["points"], topology=TOPOLOGY_TRIODE)
            self.assertEqual(result.model_type, "triode")
            self.assertGreater(result.n_points, 20)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestFullPipelinePentode(unittest.TestCase):
    """End-to-end test of fit_and_export_spice for pentodes."""

    def _run_pentode_pipeline(self, filename, tube_type):
        """Helper: run full pipeline on a pentode dataset."""
        data = _load(filename)
        points = data["points"]
        # Ensure all points have ug2 and ig2 keys
        for p in points:
            if "ug2" not in p:
                self.skipTest(f"{filename}: missing ug2")

        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = fit_and_export_spice(
                tmp_path, tube_type, points, topology=TOPOLOGY_PENTODE)
            self.assertIsInstance(result, SpiceFitResult)
            self.assertEqual(result.model_type, "pentode")
            self.assertIn("kg2", result.params)
            self.assertGreater(result.params["kg2"], 0)

            with open(tmp_path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn(".SUBCKT", content)
            self.assertIn("A G K G2", content)
            self.assertIn("ATAN", content)
            self.assertIn(".ENDS", content)
            return result
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_6550_pipeline(self):
        """6550 (11 points) → full pentode pipeline."""
        result = self._run_pentode_pipeline("pentode_6550_tuparam.json", "6550")
        self.assertLess(result.rms_error, 50.0,
                        msg=f"6550 RMS={result.rms_error:.1f} mA")

    def test_6550a_pipeline(self):
        result = self._run_pentode_pipeline("pentode_6550A_tuparam.json", "6550A")
        self.assertLess(result.rms_error, 60.0)

    def test_6l6gb_pipeline(self):
        result = self._run_pentode_pipeline("pentode_6L6GB_tuparam.json", "6L6GB")
        self.assertLess(result.rms_error, 30.0)

    def test_6l6gc_pipeline(self):
        result = self._run_pentode_pipeline("pentode_6L6GC_tuparam.json", "6L6GC")
        self.assertLess(result.rms_error, 30.0)

    def test_kt88_pipeline(self):
        result = self._run_pentode_pipeline("pentode_KT88_tuparam.json", "KT88")
        self.assertLess(result.rms_error, 50.0)


class TestPipelineTopologyDetection(unittest.TestCase):
    """Test topology auto-detection in the pipeline."""

    def test_auto_detect_triode(self):
        """Pipeline should auto-detect triode from tube_params.json."""
        data = _load("triode_ecc83_datasheet.json")
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            # topology=None → should auto-detect from tube_params.json
            result = fit_and_export_spice(
                tmp_path, "12AX7", data["points"], topology=None)
            self.assertEqual(result.model_type, "triode")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_triode_connected_uses_triode_model(self):
        """topology='triode_connected' should use triode model."""
        data = _load("triode_ecc83_datasheet.json")
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = fit_and_export_spice(
                tmp_path, "ECC83", data["points"],
                topology=TOPOLOGY_TRIODE_CONNECTED)
            self.assertEqual(result.model_type, "triode")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestPipelineUnknownTube(unittest.TestCase):
    """Test pipeline with unknown tube type (no ref in tube_params.json)."""

    def test_unknown_tube_defaults_to_triode(self):
        """topology=None + unknown tube → should default to triode model."""
        data = _load("triode_ecc83_datasheet.json")
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = fit_and_export_spice(
                tmp_path, "UNKNOWN_TUBE_XYZ", data["points"], topology=None)
            self.assertEqual(result.model_type, "triode")
            self.assertGreater(result.n_points, 10)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_unknown_tube_explicit_pentode(self):
        """Unknown tube + topology='pentode' → should use pentode model."""
        data = _load("pentode_6550_tuparam.json")
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = fit_and_export_spice(
                tmp_path, "UNKNOWN_PENTODE_XYZ", data["points"],
                topology=TOPOLOGY_PENTODE)
            self.assertEqual(result.model_type, "pentode")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestPipelineEdgeCases(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_too_few_points_raises(self):
        """Pipeline should raise RuntimeError for <10 valid points."""
        points = [
            {"ua": 100, "ug1": 0.0, "ia": 1.0},
            {"ua": 200, "ug1": -1.0, "ia": 0.5},
        ]
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with self.assertRaises(RuntimeError):
                fit_and_export_spice(tmp_path, "TEST", points, topology=TOPOLOGY_TRIODE)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_all_zero_ia_raises(self):
        """All-zero Ia should raise (no valid points after filter)."""
        points = [{"ua": 100 + i * 10, "ug1": -5.0, "ia": 0.0}
                  for i in range(20)]
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with self.assertRaises(RuntimeError):
                fit_and_export_spice(tmp_path, "TEST", points, topology=TOPOLOGY_TRIODE)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_pentode_missing_ug2_raises(self):
        """Pentode with missing ug2 should raise."""
        points = [{"ua": 100 + i * 10, "ug1": 0.0, "ia": 50.0}
                  for i in range(20)]
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with self.assertRaises((RuntimeError, KeyError)):
                fit_and_export_spice(tmp_path, "TEST", points, topology=TOPOLOGY_PENTODE)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ======================================================================
# 6. Fitter branch coverage (ref_koren, Ig2 data, initial guess edge cases)
# ======================================================================

class TestFitterWithRefKoren(unittest.TestCase):
    """Test fitters when ref_koren initial guess is explicitly provided."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_SCIPY:
            raise unittest.SkipTest("scipy not available")

    def test_triode_with_ref_koren(self):
        """Triode fitter with KorenParams seed should converge."""
        data = _load("triode_ecc83_datasheet.json")
        ua, ug1, ia = _triode_arrays(data)
        mask = ia > 0.0001
        ua, ug1, ia = ua[mask], ug1[mask], ia[mask]
        ref = KorenParams(mu=100.0, ex=1.4, kg1=1060.0, kp=600.0, kvb=300.0)
        params, cost, _ = _fit_koren_scipy(ua, ug1, ia, ref_koren=ref)
        ia_pred = _koren_ia(ua, ug1, *params)
        rms = _rms_pct(ia_pred, ia)
        self.assertLess(rms, 10.0, msg=f"With ref_koren: RMS={rms:.1f}%")

    def test_pentode_with_ref_koren_kg2_none(self):
        """Pentode fitter with ref_koren.kg2=None → should fallback to 4500."""
        data = _load("pentode_6550_tuparam.json")
        ua, ug1, ug2, ia, ig2 = _pentode_arrays(data)
        mask = (ia > 0.0001) & (ug2 > 10)
        # ref with kg2=None → _make_pentode_initial_guess uses 4500.0
        ref = KorenParams(mu=8.0, ex=1.35, kg1=890.0, kp=60.0, kvb=24.0, kg2=None)
        params, cost, _ = _fit_pentode_scipy(
            ua[mask], ug1[mask], ug2[mask], ia[mask], None, ref_koren=ref)
        mu = params[0]
        self.assertTrue(3.0 < mu < 30.0, msg=f"mu={mu:.2f}")


class TestPentodeFitterWithIg2Data(unittest.TestCase):
    """Test pentode fitter when real Ig2 measurements are provided."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_SCIPY:
            raise unittest.SkipTest("scipy not available")

    def test_synthetic_ig2_data(self):
        """Pentode fitter with synthetic Ig2 data should use combined residual."""
        # Generate synthetic data from known params
        mu, ex, kg1, kp, kvb, kg2 = 8.0, 1.35, 890.0, 60.0, 24.0, 4200.0
        ua = np.array([50, 100, 200, 300, 400, 50, 100, 200, 300, 400,
                        50, 100, 200, 300, 400], dtype=float)
        ug1 = np.array([0, 0, 0, 0, 0, -10, -10, -10, -10, -10,
                         -20, -20, -20, -20, -20], dtype=float)
        ug2 = np.full(15, 300.0)

        ia_true = _koren_ia_pentode(ua, ug1, ug2, mu, ex, kg1, kp, kvb)
        ig2_true = _koren_ig2_pentode(ug1, ug2, mu, ex, kg2)

        # Add small noise
        rng = np.random.RandomState(42)
        ia_noisy = ia_true + rng.normal(0, 0.001, len(ia_true))
        ig2_noisy = ig2_true + rng.normal(0, 0.0005, len(ig2_true))
        ia_noisy = np.maximum(ia_noisy, 0)
        ig2_noisy = np.maximum(ig2_noisy, 0)

        # Fit WITH Ig2 data
        params_with, cost_with, _ = _fit_pentode_scipy(
            ua, ug1, ug2, ia_noisy, ig2_noisy)
        # Fit WITHOUT Ig2 data
        params_without, cost_without, _ = _fit_pentode_scipy(
            ua, ug1, ug2, ia_noisy, None)

        # Both should produce reasonable mu
        self.assertTrue(3 < params_with[0] < 20,
                        msg=f"mu with ig2={params_with[0]:.2f}")
        self.assertTrue(3 < params_without[0] < 20,
                        msg=f"mu without ig2={params_without[0]:.2f}")

        # With Ig2 data, kg2 should be closer to true value
        kg2_with = params_with[5]
        kg2_without = params_without[5]
        self.assertAlmostEqual(kg2_with, 4200.0, delta=2000.0,
                               msg=f"kg2 with ig2={kg2_with:.0f}")


class TestSubcircuitEdgeCases(unittest.TestCase):
    """Test subcircuit generation edge cases."""

    def test_pentode_rgi_zero_fallback(self):
        """Pentode with rgi=0 in ref → should use 1000 as fallback."""
        ref = TubeRefParams(
            name="TEST", aliases=[], topology=TOPOLOGY_PENTODE,
            caps=TubeCaps(ccg=10.0, cgp=0.5, ccp=5.0),
            rgi=0, vct=0.0,
        )
        sub = _generate_pentode_subcircuit(
            safe_name="TEST_P", tube_type="TEST_P",
            mu=8.0, ex=1.35, kg1=890.0, kp=60.0, kvb=24.0, kg2=4500.0,
            rms_error=5.0, max_error=10.0, n_points=20,
            backend="scipy", ref=ref,
        )
        # rgi=0 is falsy → fallback to 1000 in line 440
        self.assertIn("RGI=1000", sub)
        # rgi=0 is falsy → should NOT have R1/D3 (line 463: `if ref and ref.rgi`)
        self.assertNotIn("R1 G 5", sub)

    def test_triode_ref_no_caps(self):
        """Triode with ref but caps=None → no cap ELEMENTS, but RGI declared.

        The {RGI} grid-stopper is emitted whenever rgi is truthy, so RGI must be
        declared in PARAMS even without caps (else: undefined-parameter netlist).
        """
        ref = TubeRefParams(
            name="TEST", aliases=[], topology=TOPOLOGY_TRIODE,
            caps=None, rgi=2000, vct=0.0,
        )
        sub = _generate_triode_subcircuit(
            safe_name="TEST_T", tube_type="TEST_T",
            mu=50.0, ex=1.3, kg1=500.0, kp=200.0, kvb=100.0,
            rms_error=1.0, max_error=2.0, n_points=20,
            backend="scipy", ref=ref,
        )
        self.assertNotIn("C1 G K", sub)  # no cap elements without caps
        # Grid stopper present AND its RGI parameter declared (the #4 fix).
        self.assertIn("R1 G 5", sub)
        self.assertIn("D3 5 K DX", sub)
        self.assertIn("RGI=2000", sub)
        # Structural invariant: any {RGI} reference must have a declaration.
        if "{RGI}" in sub:
            self.assertIn("RGI=", sub)

    def test_pentode_ref_no_caps_declares_rgi(self):
        """Pentode with rgi but caps=None (KT120 case) → RGI must be declared."""
        ref = TubeRefParams(
            name="KT120", aliases=[], topology=TOPOLOGY_PENTODE,
            caps=None, rgi=1000, vct=0.0,
        )
        sub = _generate_pentode_subcircuit(
            safe_name="KT120", tube_type="KT120",
            mu=8.0, ex=1.35, kg1=890.0, kp=60.0, kvb=24.0, kg2=4500.0,
            rms_error=5.0, max_error=10.0, n_points=20,
            backend="scipy", ref=ref,
        )
        self.assertIn("R1 G 5", sub)       # grid stopper referenced
        self.assertIn("RGI=1000", sub)     # ...and declared (the #4 fix)
        self.assertNotIn("C1 G K", sub)    # no cap elements without caps
        if "{RGI}" in sub:
            self.assertIn("RGI=", sub)

    def test_triode_ref_no_source(self):
        """Triode with ref.source=None → no 'Reference data source' comment."""
        ref = TubeRefParams(
            name="TEST", aliases=[], topology=TOPOLOGY_TRIODE,
            caps=None, rgi=0, vct=0.0,
            source=None,
        )
        sub = _generate_triode_subcircuit(
            safe_name="T", tube_type="T",
            mu=50.0, ex=1.3, kg1=500.0, kp=200.0, kvb=100.0,
            rms_error=1.0, max_error=2.0, n_points=10,
            backend="numpy", ref=ref,
        )
        self.assertNotIn("Reference data source", sub)

    def test_pentode_no_ig2_rms_comment(self):
        """Pentode subcircuit with ig2_rms=None → no Ig2 quality line."""
        sub = _generate_pentode_subcircuit(
            safe_name="P", tube_type="P",
            mu=8.0, ex=1.35, kg1=890.0, kp=60.0, kvb=24.0, kg2=4500.0,
            rms_error=5.0, max_error=10.0, n_points=15,
            backend="scipy", ig2_rms=None, ref=None,
        )
        self.assertNotIn("Fit quality (Ig2)", sub)


# ======================================================================
# 7. Cross-validation with published parameters
# ======================================================================

class TestPublishedParamsComparison(unittest.TestCase):
    """Compare our fitted parameters to Koren's published tube.lib values."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_SCIPY:
            raise unittest.SkipTest("scipy not available")

    def _check_param_range(self, name, fitted, published, tolerance=0.5):
        """Check fitted param is within tolerance factor of published."""
        lo = published * (1.0 - tolerance)
        hi = published * (1.0 + tolerance)
        self.assertTrue(lo <= fitted <= hi,
                        msg=f"{name}: fitted={fitted:.2f}, "
                            f"published={published:.2f}, "
                            f"tolerance=±{tolerance*100:.0f}%")

    def test_12au7_vs_published(self):
        """Fit 12AU7 Sylvania data and compare mu to published."""
        data = _load("triode_12AU7_tuparam.json")
        pub = data.get("published_params", {})
        if not pub:
            self.skipTest("No published params")
        ua, ug1, ia = _triode_arrays(data)
        params, _, _ = _fit_koren_scipy(ua, ug1, ia)
        mu = params[0]
        # Published mu=21.5, allow ±50% (fitting from few points)
        self._check_param_range("mu", mu, pub["mu"], tolerance=0.5)

    def test_6550_pentode_vs_published(self):
        """Fit 6550 pentode data and compare mu to published."""
        data = _load("pentode_6550_tuparam.json")
        pub = data.get("published_params", {})
        if not pub:
            self.skipTest("No published params")
        ua, ug1, ug2, ia, ig2 = _pentode_arrays(data)
        mask = (ia > 0.0001) & (ug2 > 10)
        params, _, _ = _fit_pentode_scipy(
            ua[mask], ug1[mask], ug2[mask], ia[mask], None)
        mu = params[0]
        # Published mu=7.9, allow ±50%
        self._check_param_range("mu", mu, pub["mu"], tolerance=0.5)


# ======================================================================
# Run
# ======================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
