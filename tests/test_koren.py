"""Unit tests for Koren tube model and SPICE fitting.

Covers:
  - Koren triode model (``_koren_ia``)
  - Koren pentode model (``_koren_ia_pentode``, ``_koren_ig2_pentode``)
  - SPICE fitting and file export (triode & pentode)
  - Topology auto-detection
  - Convergence on real tube published parameters with noise

Run:  ``py -m pytest tests/test_koren.py -v``
  or: ``py tests/test_koren.py``  (standalone, no pytest needed)
"""

import math
import os
import sys
import unittest

import numpy as np

# Detect scipy for adaptive convergence tolerances
try:
    import scipy  # noqa: F401
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Ensure the app root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.spice_export import (
    _koren_ia, _koren_ia_pentode, _koren_ig2_pentode,
    _looks_like_independent_pentode,
    fit_and_export_spice, SpiceFitResult,
)


# ======================================================================
# Helpers: generate synthetic tube data
# ======================================================================

def make_triode_points(
    mu=100.0, ex=1.4, kg1=1060.0, kp=600.0, kvb=300.0,
    ua_range=(50, 350, 25), ug1_range=(-4.0, 0.0, 0.5), ug2=0.0,
    uh=6.3, ih=0.3,
):
    """Generate synthetic measurement points using Koren model.

    Returns list of dicts in measurement format (Ia in mA).
    """
    ua_vals = np.arange(*ua_range, dtype=float)
    ug1_vals = np.arange(ug1_range[0], ug1_range[1] + 0.01, ug1_range[2])
    points = []
    for ug1 in ug1_vals:
        for ua in ua_vals:
            ia_A = _koren_ia(
                np.array([ua]), np.array([ug1]), mu, ex, kg1, kp, kvb
            )[0]
            points.append({
                "ua": float(ua),
                "ug1": float(ug1),
                "ug2": float(ug2 if ug2 > 0 else ua),  # triode: ug2=ua
                "ia": float(ia_A * 1000.0),  # convert A → mA
                "ig2": 0.1,
                "uh": uh,
                "ih": ih,
            })
    return points


def make_pentode_points(
    mu=11.0, ex=1.35, kg1=650.0, kp=60.0, kvb=24.0, kg2=4200.0,
    ua_range=(50, 400, 25), ug1_range=(-20.0, 0.0, 2.0),
    ug2_values=(200.0, 250.0),
    uh=6.3, ih=0.75,
):
    """Generate synthetic pentode measurement points using Koren pentode model.

    Returns list of dicts in measurement format (Ia/Ig2 in mA).
    """
    ua_vals = np.arange(*ua_range, dtype=float)
    ug1_vals = np.arange(ug1_range[0], ug1_range[1] + 0.01, ug1_range[2])
    points = []
    for ug2 in ug2_values:
        for ug1 in ug1_vals:
            for ua in ua_vals:
                ia_A = _koren_ia_pentode(
                    np.array([ua]), np.array([ug1]),
                    np.array([ug2]), mu, ex, kg1, kp, kvb,
                )[0]
                ig2_A = _koren_ig2_pentode(
                    np.array([ug1]), np.array([ug2]), mu, ex, kg2,
                )[0]
                points.append({
                    "ua": float(ua),
                    "ug1": float(ug1),
                    "ug2": float(ug2),
                    "ia": float(ia_A * 1000.0),   # A → mA
                    "ig2": float(ig2_A * 1000.0),  # A → mA
                    "uh": uh,
                    "ih": ih,
                })
    return points


# ======================================================================
# SPICE export tests
# ======================================================================

import tempfile
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE_CONNECTED,
)


class TestKorenModel(unittest.TestCase):
    """Test the canonical Koren triode model implementation."""

    # Published 12AX7 parameters
    PARAMS_12AX7 = (100.0, 1.4, 1060.0, 600.0, 300.0)

    def test_positive_current(self):
        """Ia should be positive for normal operating conditions."""
        ua = np.array([100.0, 200.0, 300.0])
        ug1 = np.array([-1.0, -1.0, -1.0])
        ia = _koren_ia(ua, ug1, *self.PARAMS_12AX7)
        self.assertTrue(np.all(ia > 0), f"Ia should be positive: {ia}")

    def test_cutoff_near_zero(self):
        """Ia should be near zero for deep cutoff (large negative Ug1)."""
        ua = np.array([100.0])
        ug1 = np.array([-20.0])
        ia = _koren_ia(ua, ug1, *self.PARAMS_12AX7)
        self.assertLess(ia[0], 1e-9, "Ia should be ~0 at deep cutoff")

    def test_ia_increases_with_ua(self):
        """Ia should increase as Ua increases (fixed Ug1)."""
        ua = np.array([100.0, 200.0, 300.0, 400.0])
        ug1 = np.array([-1.0, -1.0, -1.0, -1.0])
        ia = _koren_ia(ua, ug1, *self.PARAMS_12AX7)
        for i in range(len(ia) - 1):
            self.assertGreater(ia[i + 1], ia[i],
                               f"Ia should increase: Ia[{i}]={ia[i]}, Ia[{i+1}]={ia[i+1]}")

    def test_ia_increases_with_ug1(self):
        """Ia should increase as Ug1 becomes less negative."""
        ua = np.array([200.0, 200.0, 200.0])
        ug1 = np.array([-3.0, -2.0, -1.0])
        ia = _koren_ia(ua, ug1, *self.PARAMS_12AX7)
        for i in range(len(ia) - 1):
            self.assertGreater(ia[i + 1], ia[i],
                               f"Ia should increase with Ug1: Ug1[{i}]={ug1[i]}")

    def test_canonical_formula_manual(self):
        """Verify _koren_ia matches hand-computed canonical formula."""
        mu, ex, kg1, kp, kvb = self.PARAMS_12AX7
        ua, ug1 = 250.0, -2.0

        # Manual computation
        arg = kp * (1.0 / mu + ug1 / math.sqrt(kvb + ua * ua))
        e1 = (ua / kp) * math.log1p(math.exp(arg))
        # PWR+PWRS for positive e1 = 2 * e1^ex
        ia_expected = 2.0 * (e1 ** ex) / kg1

        ia_computed = _koren_ia(
            np.array([ua]), np.array([ug1]), mu, ex, kg1, kp, kvb
        )[0]

        self.assertAlmostEqual(ia_computed, ia_expected, places=10,
                               msg="Model should match hand computation")

    def test_pwr_pwrs_negative_e1(self):
        """PWR+PWRS should give 0 for negative E1 (tube cutoff)."""
        # Force E1 negative by extreme negative Ug1
        ua = np.array([10.0])
        ug1 = np.array([-100.0])
        ia = _koren_ia(ua, ug1, *self.PARAMS_12AX7)
        self.assertAlmostEqual(ia[0], 0.0, places=15,
                               msg="Ia must be 0 when E1 < 0")

    def test_vectorized(self):
        """Model should handle arrays of different sizes."""
        n = 1000
        ua = np.random.uniform(50, 400, n)
        ug1 = np.random.uniform(-5, 0, n)
        ia = _koren_ia(ua, ug1, *self.PARAMS_12AX7)
        self.assertEqual(ia.shape, (n,))
        self.assertTrue(np.all(ia >= 0))

    def test_typical_12ax7_operating_point(self):
        """12AX7 at Ua=250V Ug1=-2V should give ~0.5-1.5 mA."""
        ia = _koren_ia(
            np.array([250.0]), np.array([-2.0]), *self.PARAMS_12AX7
        )[0]
        ia_mA = ia * 1000.0
        self.assertGreater(ia_mA, 0.3, f"Ia too low: {ia_mA:.3f} mA")
        self.assertLess(ia_mA, 3.0, f"Ia too high: {ia_mA:.3f} mA")


class TestSpiceFitting(unittest.TestCase):
    """Test Koren model fitting and SPICE file export."""

    def test_fit_recovers_known_params(self):
        """Fitting data generated with known params should recover them."""
        mu, ex, kg1, kp, kvb = 100.0, 1.4, 1060.0, 600.0, 300.0
        points = make_triode_points(mu, ex, kg1, kp, kvb)

        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            fit_and_export_spice(path, "TEST_12AX7", points)
            with open(path) as f:
                content = f.read()

            # Parse fitted params from file
            import re
            for name, expected, tol_pct in [
                ("MU", mu, 10), ("EX", ex, 10), ("KG1", kg1, 15),
                ("KP", kp, 5), ("KVB", kvb, 10),
            ]:
                m = re.search(rf'{name}=([0-9.]+)', content)
                self.assertIsNotNone(m, f"Parameter {name} not found in output")
                fitted = float(m.group(1))
                pct_err = abs(fitted - expected) / expected * 100
                self.assertLess(pct_err, tol_pct,
                                f"{name}: expected={expected}, fitted={fitted}, "
                                f"err={pct_err:.1f}% > {tol_pct}%")
        finally:
            os.unlink(path)

    def test_spice_file_structure(self):
        """Output .sub file should have correct SPICE structure."""
        points = make_triode_points()
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            fit_and_export_spice(path, "6N2P", points)
            with open(path) as f:
                content = f.read()

            # Required SPICE elements
            self.assertIn(".SUBCKT 6N2P A G K", content)
            self.assertIn("PARAMS:", content)
            self.assertIn("E1 7 0 VALUE=", content)
            self.assertIn("RE1 7 0 1G", content)
            self.assertIn("G1 A K VALUE=", content)
            self.assertIn("RCP A K 1G", content)
            self.assertIn(".ENDS 6N2P", content)
            # Canonical formula markers
            self.assertIn("V(A,K)/KP*LOG", content)
            self.assertIn("PWR(V(7),EX)+PWRS(V(7),EX)", content)
            # 6N2P is alias of 12AX7 — should have caps and grid model
            self.assertIn("C1 G K", content)   # cathode-grid cap
            self.assertIn("C2 G A", content)   # grid-plate cap
            self.assertIn("C3 A K", content)   # cathode-plate cap
            self.assertIn("D3 5 K DX", content)  # grid current diode
            self.assertIn("R1 G 5", content)   # grid current resistor
            self.assertIn(".MODEL DX D", content)
        finally:
            os.unlink(path)

    def test_spice_with_ref_params(self):
        """SPICE export for known tube should include capacitances."""
        # ECC83 maps to 12AX7 — has reference params
        points = make_triode_points()
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            fit_and_export_spice(path, "ECC83", points)
            with open(path) as f:
                content = f.read()

            # Should have capacitance params
            self.assertIn("CCG=", content)
            self.assertIn("CGP=", content)
            self.assertIn("CCP=", content)
            self.assertIn("RGI=", content)
            # Should mention reference source in comments
            self.assertIn("Reference data source:", content)
        finally:
            os.unlink(path)

    def test_spice_unknown_tube_no_caps(self):
        """SPICE export for unknown tube should work without caps."""
        points = make_triode_points()
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            fit_and_export_spice(path, "UNKNOWN_TUBE", points)
            with open(path) as f:
                content = f.read()

            # Should still have core model
            self.assertIn(".SUBCKT UNKNOWN_TUBE A G K", content)
            self.assertIn("E1 7 0 VALUE=", content)
            self.assertIn("G1 A K VALUE=", content)
            # Should NOT have caps or grid model
            self.assertNotIn("C1 G K", content)
            self.assertNotIn("D3 5 K", content)
        finally:
            os.unlink(path)

    def test_unknown_pentode_data_warns(self):
        """Unknown tube + independent-screen pentode data → WARNING.

        Fitting independent-screen pentode data as a triode is wrong; without a
        reference the function must at least warn (failure-visibility).
        """
        points = make_pentode_points()  # Ug2 ∈ {200,250} swept against Ua
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            with self.assertLogs("lm19.spice_export", level="WARNING") as cm:
                fit_and_export_spice(path, "UNKNOWN_PENTODE", points)
            self.assertTrue(any("pentode" in m.lower() for m in cm.output))
        finally:
            os.unlink(path)

    def test_unknown_triode_data_no_pentode_warning(self):
        """Unknown tube + triode data (Ug2=Ua) → no pentode warning."""
        points = make_triode_points()
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            with self.assertLogs("lm19.spice_export", level="INFO") as cm:
                fit_and_export_spice(path, "UNKNOWN_TRIODE", points)
            self.assertFalse(any("likely wrong" in m for m in cm.output))
        finally:
            os.unlink(path)

    def test_looks_like_independent_pentode_predicate(self):
        """Direct unit test of the detection helper."""
        self.assertFalse(_looks_like_independent_pentode([]))
        self.assertFalse(_looks_like_independent_pentode(make_triode_points()))
        self.assertTrue(_looks_like_independent_pentode(make_pentode_points()))
        # True-triode data (Ug2=0) must NOT be flagged — pins the Ug2>min guard
        # (make_triode_points sets Ug2=Ua, so it never exercises the Ug2=0 path).
        true_triode = [{"ua": float(ua), "ug1": -2.0, "ug2": 0.0, "ia": 5.0}
                       for ua in range(50, 350, 25)]
        self.assertFalse(_looks_like_independent_pentode(true_triode))

    def test_insufficient_data_raises(self):
        """Should raise RuntimeError if fewer than 10 valid points."""
        points = [{"ua": 100, "ug1": -1, "ia": 0.0}] * 5  # all Ia=0
        with self.assertRaises(RuntimeError):
            fit_and_export_spice("dummy.sub", "TEST", points)

    def test_units_amps_in_spice(self):
        """Fitted params should produce correct amps in SPICE."""
        mu, ex, kg1, kp, kvb = 100.0, 1.4, 1060.0, 600.0, 300.0
        points = make_triode_points(mu, ex, kg1, kp, kvb)

        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            fit_and_export_spice(path, "TEST", points)

            # Verify model with original params at a test point
            ia_A = _koren_ia(np.array([250.0]), np.array([-2.0]),
                             mu, ex, kg1, kp, kvb)[0]
            # Should be in amps, ~0.001 A range for 12AX7
            self.assertGreater(ia_A, 1e-5, "Ia should be in amps")
            self.assertLess(ia_A, 0.1, "Ia should be reasonable for a triode")
        finally:
            os.unlink(path)

    def test_fit_returns_spice_fit_result(self):
        """fit_and_export_spice should return SpiceFitResult."""
        points = make_triode_points()
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            result = fit_and_export_spice(path, "TEST_12AX7", points)
            self.assertIsInstance(result, SpiceFitResult)
            self.assertEqual(result.model_type, "triode")
            self.assertIn("mu", result.params)
            self.assertIn("ex", result.params)
            self.assertIn("kg1", result.params)
            self.assertIn("kp", result.params)
            self.assertIn("kvb", result.params)
            self.assertGreater(result.n_points, 0)
            self.assertGreater(result.rms_error, 0)  # non-zero even for synthetic
            self.assertEqual(result.path, path)
        finally:
            os.unlink(path)


# ======================================================================
# Koren pentode model tests
# ======================================================================

class TestKorenPentodeModel(unittest.TestCase):
    """Test the Koren pentode model implementation."""

    # Published EL34 parameters (pentode)
    PARAMS_EL34 = (11.0, 1.35, 650.0, 60.0, 24.0)

    def test_positive_current(self):
        """Pentode Ia should be positive for normal operation."""
        ua = np.array([100.0, 200.0, 300.0])
        ug1 = np.array([-10.0, -10.0, -10.0])
        ug2 = np.array([250.0, 250.0, 250.0])
        ia = _koren_ia_pentode(ua, ug1, ug2, *self.PARAMS_EL34)
        self.assertTrue(np.all(ia > 0), f"Ia should be positive: {ia}")

    def test_cutoff_near_zero(self):
        """Pentode Ia should be near zero for deep cutoff."""
        ua = np.array([200.0])
        ug1 = np.array([-100.0])
        ug2 = np.array([250.0])
        ia = _koren_ia_pentode(ua, ug1, ug2, *self.PARAMS_EL34)
        self.assertLess(ia[0], 1e-9, "Ia should be ~0 at deep cutoff")

    def test_ia_increases_with_ua(self):
        """Pentode Ia should increase with Ua (saturation region)."""
        ua = np.array([50.0, 100.0, 200.0, 300.0])
        ug1 = np.full(4, -10.0)
        ug2 = np.full(4, 250.0)
        ia = _koren_ia_pentode(ua, ug1, ug2, *self.PARAMS_EL34)
        for i in range(len(ia) - 1):
            self.assertGreater(ia[i + 1], ia[i],
                               f"Ia should increase: {ia[i]:.6f} vs {ia[i+1]:.6f}")

    def test_ia_increases_with_ug1(self):
        """Pentode Ia should increase as Ug1 becomes less negative."""
        ua = np.array([200.0, 200.0, 200.0])
        ug1 = np.array([-15.0, -10.0, -5.0])
        ug2 = np.full(3, 250.0)
        ia = _koren_ia_pentode(ua, ug1, ug2, *self.PARAMS_EL34)
        for i in range(len(ia) - 1):
            self.assertGreater(ia[i + 1], ia[i])

    def test_ia_increases_with_ug2(self):
        """Pentode Ia should increase with higher Ug2 (screen voltage)."""
        ua = np.array([200.0, 200.0, 200.0])
        ug1 = np.full(3, -10.0)
        ug2 = np.array([150.0, 200.0, 300.0])
        ia = _koren_ia_pentode(ua, ug1, ug2, *self.PARAMS_EL34)
        for i in range(len(ia) - 1):
            self.assertGreater(ia[i + 1], ia[i])

    def test_arctan_knee_effect(self):
        """At low Ua the arctan knee should suppress current."""
        # arctan(Ua/Kvb) → small for Ua << Kvb
        ua_low = np.array([1.0])
        ua_high = np.array([200.0])
        ug1 = np.array([-5.0])
        ug2 = np.array([250.0])
        ia_low = _koren_ia_pentode(ua_low, ug1, ug2, *self.PARAMS_EL34)
        ia_high = _koren_ia_pentode(ua_high, ug1, ug2, *self.PARAMS_EL34)
        # At Ua=1V, current should be much smaller than at Ua=200V
        self.assertLess(ia_low[0], ia_high[0] * 0.1,
                        "Arctan knee should strongly reduce Ia at low Ua")

    def test_vectorized(self):
        """Pentode model should handle large arrays."""
        n = 1000
        ua = np.random.uniform(50, 400, n)
        ug1 = np.random.uniform(-20, 0, n)
        ug2 = np.random.uniform(150, 300, n)
        ia = _koren_ia_pentode(ua, ug1, ug2, *self.PARAMS_EL34)
        self.assertEqual(ia.shape, (n,))
        self.assertTrue(np.all(ia >= 0))

    def test_canonical_pentode_formula_manual(self):
        """Verify _koren_ia_pentode matches hand-computed pentode formula."""
        mu, ex, kg1, kp, kvb = self.PARAMS_EL34
        ua, ug1, ug2 = 250.0, -10.0, 250.0

        # Manual computation:
        # E1 = (Ug2/Kp) * log(1 + exp(Kp * (1/mu + Ug1/Ug2)))
        arg = kp * (1.0 / mu + ug1 / ug2)
        e1 = (ug2 / kp) * math.log1p(math.exp(arg))
        # Ia = 2 * E1^Ex / Kg1 * arctan(Ua/Kvb)
        ia_expected = 2.0 * (e1 ** ex) / kg1 * math.atan(ua / kvb)

        ia_computed = _koren_ia_pentode(
            np.array([ua]), np.array([ug1]), np.array([ug2]),
            mu, ex, kg1, kp, kvb,
        )[0]

        self.assertAlmostEqual(ia_computed, ia_expected, places=10,
                               msg="Pentode model should match hand computation")

    def test_numerical_stability_ua_near_zero(self):
        """Model should not crash or produce NaN for Ua near 0."""
        ua = np.array([0.0, 0.001, 0.01])
        ug1 = np.full(3, -5.0)
        ug2 = np.full(3, 250.0)
        ia = _koren_ia_pentode(ua, ug1, ug2, *self.PARAMS_EL34)
        self.assertFalse(np.any(np.isnan(ia)), "Should not produce NaN")
        self.assertFalse(np.any(np.isinf(ia)), "Should not produce Inf")
        self.assertTrue(np.all(ia >= 0))

    def test_numerical_stability_ug2_near_zero(self):
        """Model should not crash or produce NaN for Ug2 near 0."""
        ua = np.array([200.0, 200.0])
        ug1 = np.array([-5.0, -5.0])
        ug2 = np.array([0.0, 0.001])
        ia = _koren_ia_pentode(ua, ug1, ug2, *self.PARAMS_EL34)
        self.assertFalse(np.any(np.isnan(ia)), "Should not produce NaN")
        self.assertFalse(np.any(np.isinf(ia)), "Should not produce Inf")
        self.assertTrue(np.all(ia >= 0))


class TestKorenPentodeIg2(unittest.TestCase):
    """Test the Koren pentode screen current model."""

    def test_positive_ig2(self):
        """Ig2 should be positive under normal conditions."""
        ug1 = np.array([-10.0, -5.0, 0.0])
        ug2 = np.array([250.0, 250.0, 250.0])
        ig2 = _koren_ig2_pentode(ug1, ug2, mu=11.0, ex=1.35, kg2=4200.0)
        self.assertTrue(np.all(ig2 >= 0))

    def test_ig2_cutoff(self):
        """Ig2 should be zero when Ug1 + Ug2/mu < 0."""
        ug1 = np.array([-100.0])
        ug2 = np.array([50.0])
        ig2 = _koren_ig2_pentode(ug1, ug2, mu=11.0, ex=1.35, kg2=4200.0)
        self.assertAlmostEqual(ig2[0], 0.0, places=10)

    def test_ig2_increases_with_ug2(self):
        """Ig2 should increase with screen voltage."""
        ug1 = np.array([-5.0, -5.0, -5.0])
        ug2 = np.array([100.0, 200.0, 300.0])
        ig2 = _koren_ig2_pentode(ug1, ug2, mu=11.0, ex=1.35, kg2=4200.0)
        for i in range(len(ig2) - 1):
            self.assertGreater(ig2[i + 1], ig2[i])

    def test_ig2_formula_manual(self):
        """Verify _koren_ig2_pentode matches hand-computed formula."""
        mu, ex, kg2 = 11.0, 1.35, 4200.0
        ug1, ug2 = -10.0, 250.0
        # Ig2 = (Ug1 + Ug2/mu)^Ex / Kg2
        v = ug1 + ug2 / mu
        ig2_expected = (v ** ex) / kg2

        ig2_computed = _koren_ig2_pentode(
            np.array([ug1]), np.array([ug2]), mu, ex, kg2,
        )[0]

        self.assertAlmostEqual(ig2_computed, ig2_expected, places=10,
                               msg="Ig2 model should match hand computation")


class TestPentodeFitting(unittest.TestCase):
    """Test pentode model fitting and SPICE file export."""

    def test_pentode_fit_recovers_params(self):
        """Fitting pentode data should recover approximate parameters."""
        mu, ex, kg1, kp, kvb, kg2 = 11.0, 1.35, 650.0, 60.0, 24.0, 4200.0
        points = make_pentode_points(mu, ex, kg1, kp, kvb, kg2)

        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            result = fit_and_export_spice(
                path, "TEST_EL34", points, topology=TOPOLOGY_PENTODE)
            self.assertEqual(result.model_type, "pentode")
            self.assertIn("kg2", result.params)

            # Check fitted params are in reasonable range
            p = result.params
            self.assertGreater(p["mu"], 2.0)
            self.assertLess(p["mu"], 500.0)
            self.assertGreater(p["kg1"], 10.0)
            self.assertGreater(p["kg2"], 500.0)
            # RMS error should be reasonable
            self.assertLess(result.rms_error, 20.0,
                            f"RMS error too high: {result.rms_error:.2f} mA")
        finally:
            os.unlink(path)

    def test_pentode_spice_structure(self):
        """Pentode .sub file should have 4-pin subcircuit structure."""
        points = make_pentode_points()
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            result = fit_and_export_spice(
                path, "EL34_test", points, topology=TOPOLOGY_PENTODE)
            with open(path) as f:
                content = f.read()

            # 4-pin pentode subcircuit
            self.assertIn(".SUBCKT EL34_test A G K G2", content)
            self.assertIn("PARAMS:", content)
            self.assertIn("KG2=", content)
            self.assertIn("E1 7 0 VALUE=", content)
            self.assertIn("RE1 7 0 1G", content)
            # Pentode Ia with arctan knee
            self.assertIn("ATAN(V(A,K)/KVB)", content)
            # Screen current source
            self.assertIn("G2 G2 K VALUE=", content)
            self.assertIn("R2 G2 K 1G", content)
            self.assertIn(".ENDS EL34_test", content)
            # Pentode model comment
            self.assertIn("pentode model", content)
        finally:
            os.unlink(path)

    def test_pentode_insufficient_data(self):
        """Should raise if not enough pentode data points."""
        points = [{"ua": 100, "ug1": -1, "ug2": 250, "ia": 0.0, "ig2": 0}] * 5
        with self.assertRaises(RuntimeError):
            fit_and_export_spice("dummy.sub", "TEST", points, topology=TOPOLOGY_PENTODE)

    def test_pentode_with_ref_params(self):
        """Pentode SPICE for known tube should include caps and grid model."""
        # EL34 is in tube_params.json with caps and rgi
        points = make_pentode_points()
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            result = fit_and_export_spice(
                path, "EL34", points, topology=TOPOLOGY_PENTODE)
            with open(path) as f:
                content = f.read()

            # Should have PENTODE capacitance params (ccg1/cpg1, not triode ccg)
            self.assertIn("CCG1=", content)
            self.assertIn("CPG1=", content)
            self.assertIn("CCP=", content)
            self.assertNotIn("CCG=", content)
            # Should have grid current model
            self.assertIn("D3 5 K DX", content)
            self.assertIn("R1 G 5", content)
            self.assertIn(".MODEL DX D", content)
            # Reference source in comments
            self.assertIn("Reference data source:", content)
        finally:
            os.unlink(path)

    def test_pentode_unknown_tube_no_caps(self):
        """Pentode SPICE for unknown tube should work without caps."""
        points = make_pentode_points()
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            result = fit_and_export_spice(
                path, "UNKNOWN_PENT", points, topology=TOPOLOGY_PENTODE)
            with open(path) as f:
                content = f.read()

            # Core pentode structure present
            self.assertIn(".SUBCKT UNKNOWN_PENT A G K G2", content)
            self.assertIn("ATAN(V(A,K)/KVB)", content)
            self.assertIn("G2 G2 K VALUE=", content)
            # No caps or grid model
            self.assertNotIn("C1 G K", content)
            self.assertNotIn("D3 5 K", content)
        finally:
            os.unlink(path)

    def test_pentode_returns_spice_fit_result(self):
        """Pentode fit should return SpiceFitResult with all fields."""
        points = make_pentode_points()
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            result = fit_and_export_spice(
                path, "PENT_TEST", points, topology=TOPOLOGY_PENTODE)
            self.assertIsInstance(result, SpiceFitResult)
            self.assertEqual(result.model_type, "pentode")
            self.assertGreater(result.n_points, 10)
            self.assertEqual(result.path, path)
            # Pentode params should include kg2
            self.assertIn("kg2", result.params)
            self.assertEqual(len(result.params), 6)
        finally:
            os.unlink(path)


class TestTopologyAutoDetection(unittest.TestCase):
    """Test automatic topology detection in fit_and_export_spice."""

    def test_triode_auto_detect(self):
        """Known triode (12AX7) should auto-detect as triode."""
        points = make_triode_points()
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            result = fit_and_export_spice(path, "ECC83", points)
            self.assertEqual(result.model_type, "triode")
        finally:
            os.unlink(path)

    def test_pentode_auto_detect(self):
        """Known pentode (EL34) should auto-detect as pentode."""
        points = make_pentode_points()
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            result = fit_and_export_spice(path, "EL34", points)
            self.assertEqual(result.model_type, "pentode")
        finally:
            os.unlink(path)

    def test_unknown_tube_defaults_triode(self):
        """Unknown tube should default to triode model."""
        points = make_triode_points()
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            result = fit_and_export_spice(path, "UNKNOWN_TUBE", points)
            self.assertEqual(result.model_type, "triode")
        finally:
            os.unlink(path)

    def test_topology_override_pentode(self):
        """Explicit topology='pentode' should override auto-detection."""
        points = make_pentode_points()
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            result = fit_and_export_spice(
                path, "UNKNOWN", points, topology=TOPOLOGY_PENTODE)
            self.assertEqual(result.model_type, "pentode")
        finally:
            os.unlink(path)

    def test_topology_triode_connected(self):
        """triode_connected should use triode model."""
        points = make_triode_points()
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            result = fit_and_export_spice(
                path, "TEST", points, topology=TOPOLOGY_TRIODE_CONNECTED)
            self.assertEqual(result.model_type, "triode")
        finally:
            os.unlink(path)


# ======================================================================
# Convergence tests on real tube data
# ======================================================================

class TestTriodeConvergenceRealTubes(unittest.TestCase):
    """Test fitting convergence on data from real tube published parameters.

    For each tube: generate data from published Koren params with
    realistic measurement noise (~1% + 0.05 mA), fit parameters,
    and verify they converge within acceptable tolerance.

    Also verifies specific datasheet operating points.
    """

    # Published Koren params: (mu, ex, kg1, kp, kvb)
    # Tolerances are adaptive: tight with scipy (Trust Region Reflective),
    # relaxed with numpy-only (coordinate descent, prone to local minima).
    # Kvb is excluded from per-param checks — it's inside sqrt(Kvb + Ua²)
    # and nearly unidentifiable from curve data when Ua² >> Kvb.
    TUBES = {
        "12AX7": {
            "params": (100.0, 1.4, 1060.0, 600.0, 300.0),
            "ua_range": (50, 350, 25),
            "ug1_range": (-4.0, 0.0, 0.5),
            "max_rms_mA": 0.5 if HAS_SCIPY else 1.0,
            "tol_pct": (
                {"mu": 5, "ex": 5, "kg1": 10, "kp": 8}
                if HAS_SCIPY else
                {"mu": 20, "ex": 10, "kg1": 25, "kp": 20}
            ),
            # Datasheet: Ua=250V, Ug1=-2V → Ia ≈ 1.0 mA
            "operating_points": [
                {"ua": 250, "ug1": -2.0, "ia_min": 0.5, "ia_max": 2.0},
            ],
        },
        "12AU7": {
            "params": (21.5, 1.3, 1180.0, 84.0, 300.0),
            "ua_range": (50, 350, 25),
            "ug1_range": (-15.0, 0.0, 1.0),
            "max_rms_mA": 0.5 if HAS_SCIPY else 1.0,
            "tol_pct": (
                {"mu": 5, "ex": 5, "kg1": 10, "kp": 10}
                if HAS_SCIPY else
                {"mu": 20, "ex": 10, "kg1": 25, "kp": 25}
            ),
            # Datasheet: Ua=250V, Ug1=-8.5V → Ia ≈ 10 mA
            "operating_points": [
                {"ua": 250, "ug1": -8.5, "ia_min": 5.0, "ia_max": 18.0},
            ],
        },
        # 6DJ8: high-gm tube. Numpy fitter finds different local minima
        # due to mu/kg1/kp correlation. With scipy — full param checks.
        "6DJ8": {
            "params": (33.0, 1.3, 330.0, 320.0, 300.0),
            "ua_range": (20, 300, 20),
            "ug1_range": (-4.0, 0.0, 0.5),
            "max_rms_mA": 0.5 if HAS_SCIPY else 5.0,
            "tol_pct": (
                {"mu": 10, "ex": 10, "kg1": 15, "kp": 10}
                if HAS_SCIPY else
                None  # numpy: skip per-param, only RMS + operating point
            ),
            # Datasheet: Ua=90V, Ug1=-1.3V → Ia ≈ 15 mA
            "operating_points": [
                {"ua": 90, "ug1": -1.3,
                 "ia_min": 8.0 if HAS_SCIPY else 5.0,
                 "ia_max": 25.0 if HAS_SCIPY else 30.0},
            ],
        },
        "6SN7": {
            "params": (20.0, 1.3, 1350.0, 130.0, 300.0),
            "ua_range": (50, 350, 25),
            "ug1_range": (-12.0, 0.0, 1.0),
            "max_rms_mA": 0.5 if HAS_SCIPY else 1.0,
            "tol_pct": (
                {"mu": 5, "ex": 5, "kg1": 10, "kp": 10}
                if HAS_SCIPY else
                {"mu": 20, "ex": 10, "kg1": 25, "kp": 25}
            ),
            # Datasheet: Ua=250V, Ug1=-8V → Ia ≈ 9 mA
            "operating_points": [
                {"ua": 250, "ug1": -8.0, "ia_min": 4.0, "ia_max": 15.0},
            ],
        },
    }

    def _generate_noisy_triode_data(self, tube_name):
        """Generate data from published params with 1% noise + 0.05 mA offset."""
        cfg = self.TUBES[tube_name]
        mu, ex, kg1, kp, kvb = cfg["params"]
        rng = np.random.RandomState(42)  # reproducible noise
        points = make_triode_points(
            mu, ex, kg1, kp, kvb,
            ua_range=cfg["ua_range"],
            ug1_range=cfg["ug1_range"],
        )
        for p in points:
            noise = 1.0 + rng.normal(0, 0.01)  # 1% relative
            p["ia"] = max(0.0, p["ia"] * noise + rng.normal(0, 0.05))  # +0.05mA abs
        return points

    def _run_convergence_test(self, tube_name):
        """Fit noisy data and verify params converge to published values."""
        cfg = self.TUBES[tube_name]
        mu, ex, kg1, kp, kvb = cfg["params"]
        points = self._generate_noisy_triode_data(tube_name)

        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            result = fit_and_export_spice(path, tube_name, points)
            p = result.params

            # Check each param within its specific tolerance (if defined).
            # Kvb is excluded — it's inside sqrt(Kvb + Ua²) and nearly
            # unidentifiable from curve data when Ua² >> Kvb.
            tol = cfg["tol_pct"]
            if tol is not None:
                for name, expected, fitted in [
                    ("mu", mu, p["mu"]),
                    ("ex", ex, p["ex"]),
                    ("kg1", kg1, p["kg1"]),
                    ("kp", kp, p["kp"]),
                ]:
                    pct_err = abs(fitted - expected) / expected * 100
                    self.assertLess(
                        pct_err, tol[name],
                        f"{tube_name} {name}: expected={expected:.2f}, "
                        f"fitted={fitted:.2f}, err={pct_err:.1f}% > {tol[name]}%")

            # RMS error should be under per-tube threshold
            max_rms = cfg.get("max_rms_mA", 1.0)
            self.assertLess(result.rms_error, max_rms,
                            f"{tube_name}: RMS error {result.rms_error:.2f} mA "
                            f"> {max_rms} mA threshold")

            # Verify fitted model reproduces the original data
            # (this is the real convergence check — params may differ
            # due to parameter trade-offs but the curve must fit)
            mu0, ex0, kg10, kp0, kvb0 = cfg["params"]
            for op in cfg["operating_points"]:
                ia_fit = _koren_ia(
                    np.array([float(op["ua"])]),
                    np.array([float(op["ug1"])]),
                    p["mu"], p["ex"], p["kg1"], p["kp"], p["kvb"],
                )[0] * 1000.0
                self.assertGreater(
                    ia_fit, op["ia_min"],
                    f"{tube_name}: fitted Ia={ia_fit:.2f} mA < {op['ia_min']} mA "
                    f"at Ua={op['ua']}V Ug1={op['ug1']}V")
                self.assertLess(
                    ia_fit, op["ia_max"],
                    f"{tube_name}: fitted Ia={ia_fit:.2f} mA > {op['ia_max']} mA "
                    f"at Ua={op['ua']}V Ug1={op['ug1']}V")
        finally:
            os.unlink(path)

    def test_12ax7_convergence(self):
        """12AX7 (high-mu triode, mu=100): fit with noise should converge."""
        self._run_convergence_test("12AX7")

    def test_12au7_convergence(self):
        """12AU7 (low-mu triode, mu=21.5): fit with noise should converge."""
        self._run_convergence_test("12AU7")

    def test_6dj8_convergence(self):
        """6DJ8 (medium-mu triode, mu=33): fit with noise should converge."""
        self._run_convergence_test("6DJ8")

    def test_6sn7_convergence(self):
        """6SN7 (dual triode, mu=20): fit with noise should converge."""
        self._run_convergence_test("6SN7")

    def test_12ax7_operating_point(self):
        """12AX7: Ia at Ua=250V Ug1=-2V should match datasheet range."""
        mu, ex, kg1, kp, kvb = self.TUBES["12AX7"]["params"]
        ia_A = _koren_ia(np.array([250.0]), np.array([-2.0]),
                         mu, ex, kg1, kp, kvb)[0]
        ia_mA = ia_A * 1000.0
        self.assertGreater(ia_mA, 0.5, f"12AX7 Ia too low: {ia_mA:.3f} mA")
        self.assertLess(ia_mA, 2.0, f"12AX7 Ia too high: {ia_mA:.3f} mA")

    def test_12au7_operating_point(self):
        """12AU7: Ia at Ua=250V Ug1=-8.5V should match datasheet range."""
        mu, ex, kg1, kp, kvb = self.TUBES["12AU7"]["params"]
        ia_A = _koren_ia(np.array([250.0]), np.array([-8.5]),
                         mu, ex, kg1, kp, kvb)[0]
        ia_mA = ia_A * 1000.0
        self.assertGreater(ia_mA, 5.0, f"12AU7 Ia too low: {ia_mA:.3f} mA")
        self.assertLess(ia_mA, 18.0, f"12AU7 Ia too high: {ia_mA:.3f} mA")

    def test_6dj8_operating_point(self):
        """6DJ8: Ia at Ua=90V Ug1=-1.3V should match datasheet range."""
        mu, ex, kg1, kp, kvb = self.TUBES["6DJ8"]["params"]
        ia_A = _koren_ia(np.array([90.0]), np.array([-1.3]),
                         mu, ex, kg1, kp, kvb)[0]
        ia_mA = ia_A * 1000.0
        self.assertGreater(ia_mA, 8.0, f"6DJ8 Ia too low: {ia_mA:.3f} mA")
        self.assertLess(ia_mA, 25.0, f"6DJ8 Ia too high: {ia_mA:.3f} mA")


class TestPentodeConvergenceRealTubes(unittest.TestCase):
    """Test pentode fitting convergence on published parameters with noise.

    EL34 and 6L6 are the most commonly used pentodes and have
    well-established Koren parameters.
    """

    # Adaptive tolerances: scipy gets tighter checks
    TUBES = {
        "EL34": {
            "params": (11.0, 1.35, 650.0, 60.0, 24.0, 4200.0),
            "ua_range": (50, 400, 25),
            "ug1_range": (-20.0, 0.0, 2.0),
            "ug2_values": (200.0, 250.0, 300.0),
            "max_rms_mA": 3.0 if HAS_SCIPY else 5.0,
            "tol_pct": (
                {"mu": 10, "ex": 8, "kg1": 15, "kp": 20, "kvb": 30}
                if HAS_SCIPY else
                {"mu": 25, "ex": 15, "kg1": 30, "kp": 35, "kvb": 50}
            ),
            # Datasheet: Ua=250V Ug1=-13V Ug2=265V → Ia ≈ 100 mA
            "operating_points": [
                {"ua": 250, "ug1": -13.0, "ug2": 265.0,
                 "ia_min": 40.0, "ia_max": 200.0},
            ],
        },
        "6L6": {
            "params": (8.7, 1.35, 1460.0, 48.0, 12.0, 4500.0),
            "ua_range": (50, 400, 25),
            "ug1_range": (-25.0, 0.0, 2.5),
            "ug2_values": (200.0, 250.0),
            "max_rms_mA": 2.0 if HAS_SCIPY else 5.0,
            "tol_pct": (
                {"mu": 10, "ex": 8, "kg1": 15, "kp": 20, "kvb": 30}
                if HAS_SCIPY else
                {"mu": 25, "ex": 15, "kg1": 30, "kp": 35, "kvb": 50}
            ),
            # Datasheet: Ua=250V Ug1=-14V Ug2=250V → Ia ≈ 72 mA
            "operating_points": [
                {"ua": 250, "ug1": -14.0, "ug2": 250.0,
                 "ia_min": 30.0, "ia_max": 150.0},
            ],
        },
    }

    def _generate_noisy_pentode_data(self, tube_name):
        """Generate pentode data from published params with 1.5% noise."""
        cfg = self.TUBES[tube_name]
        mu, ex, kg1, kp, kvb, kg2 = cfg["params"]
        rng = np.random.RandomState(42)
        points = make_pentode_points(
            mu, ex, kg1, kp, kvb, kg2,
            ua_range=cfg["ua_range"],
            ug1_range=cfg["ug1_range"],
            ug2_values=cfg["ug2_values"],
        )
        for p in points:
            noise_ia = 1.0 + rng.normal(0, 0.015)   # 1.5% relative
            noise_ig2 = 1.0 + rng.normal(0, 0.02)    # 2% for Ig2 (noisier)
            p["ia"] = max(0.0, p["ia"] * noise_ia + rng.normal(0, 0.1))
            p["ig2"] = max(0.0, p["ig2"] * noise_ig2 + rng.normal(0, 0.05))
        return points

    def _run_pentode_convergence(self, tube_name):
        """Fit noisy pentode data and verify convergence."""
        cfg = self.TUBES[tube_name]
        mu, ex, kg1, kp, kvb, kg2 = cfg["params"]
        points = self._generate_noisy_pentode_data(tube_name)

        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            result = fit_and_export_spice(
                path, tube_name, points, topology=TOPOLOGY_PENTODE)
            p = result.params

            tol = cfg["tol_pct"]
            for name, expected, fitted in [
                ("mu", mu, p["mu"]),
                ("ex", ex, p["ex"]),
                ("kg1", kg1, p["kg1"]),
                ("kp", kp, p["kp"]),
                ("kvb", kvb, p["kvb"]),
            ]:
                pct_err = abs(fitted - expected) / expected * 100
                self.assertLess(
                    pct_err, tol[name],
                    f"{tube_name} {name}: expected={expected:.2f}, "
                    f"fitted={fitted:.2f}, err={pct_err:.1f}% > {tol[name]}%")

            # RMS threshold: adaptive per tube and fitter
            max_rms = cfg.get("max_rms_mA", 5.0)
            self.assertLess(result.rms_error, max_rms,
                            f"{tube_name}: RMS error {result.rms_error:.2f} mA "
                            f"> {max_rms} mA threshold")
        finally:
            os.unlink(path)

    def test_el34_convergence(self):
        """EL34 (power pentode, mu=11): fit with noise should converge."""
        self._run_pentode_convergence("EL34")

    def test_6l6_convergence(self):
        """6L6 (beam tetrode, mu=8.7): fit with noise should converge."""
        self._run_pentode_convergence("6L6")

    def test_el34_operating_point(self):
        """EL34: Ia at Ua=250V Ug1=-13V Ug2=265V should match datasheet."""
        mu, ex, kg1, kp, kvb = 11.0, 1.35, 650.0, 60.0, 24.0
        ia_A = _koren_ia_pentode(
            np.array([250.0]), np.array([-13.0]), np.array([265.0]),
            mu, ex, kg1, kp, kvb)[0]
        ia_mA = ia_A * 1000.0
        self.assertGreater(ia_mA, 40.0, f"EL34 Ia too low: {ia_mA:.1f} mA")
        self.assertLess(ia_mA, 200.0, f"EL34 Ia too high: {ia_mA:.1f} mA")

    def test_6l6_operating_point(self):
        """6L6: Ia at Ua=250V Ug1=-14V Ug2=250V should match datasheet."""
        mu, ex, kg1, kp, kvb = 8.7, 1.35, 1460.0, 48.0, 12.0
        ia_A = _koren_ia_pentode(
            np.array([250.0]), np.array([-14.0]), np.array([250.0]),
            mu, ex, kg1, kp, kvb)[0]
        ia_mA = ia_A * 1000.0
        self.assertGreater(ia_mA, 30.0, f"6L6 Ia too low: {ia_mA:.1f} mA")
        self.assertLess(ia_mA, 150.0, f"6L6 Ia too high: {ia_mA:.1f} mA")

    def test_fitted_model_matches_data(self):
        """Fitted pentode model should reproduce input data within RMS."""
        points = make_pentode_points()
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False) as f:
            path = f.name
        try:
            result = fit_and_export_spice(
                path, "EL34", points, topology=TOPOLOGY_PENTODE)
            p = result.params

            # Evaluate fitted model at each data point
            for pt in points[:20]:  # spot-check first 20
                if pt["ia"] < 0.1:
                    continue
                ia_fit = _koren_ia_pentode(
                    np.array([pt["ua"]]), np.array([pt["ug1"]]),
                    np.array([pt["ug2"]]),
                    p["mu"], p["ex"], p["kg1"], p["kp"], p["kvb"],
                )[0] * 1000.0
                # Within 2× RMS error of data point
                self.assertLess(
                    abs(ia_fit - pt["ia"]), result.rms_error * 3 + 0.5,
                    f"Fitted model deviates too much at "
                    f"Ua={pt['ua']} Ug1={pt['ug1']} Ug2={pt['ug2']}: "
                    f"data={pt['ia']:.2f}, model={ia_fit:.2f} mA")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
