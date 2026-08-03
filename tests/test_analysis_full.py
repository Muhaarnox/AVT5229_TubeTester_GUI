"""Tests for analysis, quality scoring, matching, aging, topology and plotting helpers.

Covers:
  - analysis: S/R/K computation, zone filtering (triode/pentode)
  - quality: tube scoring, matching, aging trend
  - topology: is_triode across config, analysis, csv_export, quality
  - plotting: cluster_nominal, nominal_key static helpers
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.analysis import Zone, compute_s, compute_r, compute_k, compute_sr_zone, count_in_zone
from lm19.quality import (
    compute_quality, compute_matching, compute_matching_curves,
    compute_aging_trend, QualityReport, MatchResult, AgingPoint,
)
from lm19.csv_export import format_csv, format_matrix, format_multi_csv
from app.plotting import PlotRenderer
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)


# ======================================================================
# Helpers
# ======================================================================

class _FakeLampConfig:
    """Minimal LampConfig stand-in for testing."""
    def __init__(self, ua=250.0, ug1=-8.5, ug2=170.0, ia=48.0,
                 s=11.0, r=None, k=None):
        self.ua = ua
        self.ug1 = ug1
        self.ug2 = ug2
        self.ia = ia
        self.s = s
        self.r = r or 0
        self.k = k or 0


# ======================================================================
# Analysis tests
# ======================================================================

class TestAnalysis(unittest.TestCase):
    """Test S/R/K computation."""

    def _make_zone_points(self):
        """Create points with known linear relationship for S/R/K."""
        # S = dIa/dUg1 = 10 mA/V, R = dUa/dIa = 1/0.05 = 20 kOhm
        # Ia = 10 * Ug1 + 0.05 * Ua + const
        # At Ua=200, Ug1=-2: Ia = 10*(-2) + 0.05*200 + 30 = 20
        points = []
        for ua in [180, 200, 220]:
            for ug1 in [-3.0, -2.0, -1.0]:
                ia = 10.0 * ug1 + 0.05 * ua + 30.0
                points.append({
                    "ua": float(ua), "ug1": float(ug1),
                    "ug2": 200.0, "ia": ia,
                })
        return points

    def test_compute_s(self):
        """S (transconductance) = dIa/dUg1 should be ~10 mA/V."""
        points = self._make_zone_points()
        zone = Zone(ua_min=170, ua_max=230, ug1_min=-3.5, ug1_max=-0.5,
                    ug2=200.0)
        s = compute_s(points, zone)
        self.assertIsNotNone(s)
        self.assertAlmostEqual(s, 10.0, places=1,
                               msg=f"S should be ~10 mA/V, got {s}")

    def test_compute_r(self):
        """R (plate resistance) = 1/(dIa/dUa) should be ~20 kOhm."""
        points = self._make_zone_points()
        zone = Zone(ua_min=170, ua_max=230, ug1_min=-3.5, ug1_max=-0.5,
                    ug2=200.0)
        r = compute_r(points, zone)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 20.0, places=0,
                               msg=f"R should be ~20 kOhm, got {r}")

    def test_compute_k(self):
        """K = S * R (amplification factor)."""
        s = 10.0
        r = 20.0
        k = compute_k(s, r)
        self.assertEqual(k, 200.0)

    def test_compute_k_none(self):
        """K should be None if S or R is None."""
        self.assertIsNone(compute_k(None, 20.0))
        self.assertIsNone(compute_k(10.0, None))

    def test_zone_filtering(self):
        """Only points inside the zone should be counted."""
        points = [
            {"ua": 200, "ug1": -2.0, "ug2": 170.0, "ia": 5.0},
            {"ua": 200, "ug1": -2.0, "ug2": 170.0, "ia": 6.0},
            {"ua": 500, "ug1": -2.0, "ug2": 170.0, "ia": 20.0},  # outside
        ]
        zone = Zone(ua_min=100, ua_max=300, ug1_min=-3, ug1_max=-1,
                    ug2=170.0)
        self.assertEqual(count_in_zone(points, zone), 2)

    def test_zone_triode_mode(self):
        """In triode mode (ug2_track_ua), ug2 should match ua+offset."""
        points = [
            {"ua": 200, "ug1": -2.0, "ug2": 200.0, "ia": 5.0},  # ug2=ua
            {"ua": 200, "ug1": -2.0, "ug2": 170.0, "ia": 5.0},  # wrong ug2
        ]
        zone = Zone(ua_min=100, ua_max=300, ug1_min=-3, ug1_max=-1,
                    ug2_track_ua=True, ug2_offset=0.0)
        self.assertEqual(count_in_zone(points, zone), 1)

    def test_empty_points(self):
        """S/R/K should return None for empty data."""
        zone = Zone(ua_min=0, ua_max=500, ug1_min=-10, ug1_max=0, ug2=0)
        self.assertIsNone(compute_s([], zone))
        self.assertIsNone(compute_r([], zone))


class TestComputeSrZone(unittest.TestCase):
    """Test multivariate S/R computation via compute_sr_zone."""

    @staticmethod
    def _grid_points(s_true=10.0, inv_r_true=0.05, offset=30.0,
                     ua_vals=(180, 200, 220), ug1_vals=(-3.0, -2.0, -1.0),
                     ug2=200.0):
        """Generate a 2-D grid: Ia = s_true·Ug1 + inv_r_true·Ua + offset."""
        points = []
        for ua in ua_vals:
            for ug1 in ug1_vals:
                ia = s_true * ug1 + inv_r_true * ua + offset
                points.append({"ua": float(ua), "ug1": float(ug1),
                                "ug2": ug2, "ia": ia})
        return points

    def test_grid_basic(self):
        """2-D grid with S=10, R=20: both should be recovered exactly."""
        points = self._grid_points(s_true=10.0, inv_r_true=0.05)
        zone = Zone(ua_min=170, ua_max=230, ug1_min=-3.5, ug1_max=-0.5,
                    ug2=200.0)
        s, r, expanded = compute_sr_zone(points, zone)
        self.assertIsNotNone(s)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(s, 10.0, places=4)
        self.assertAlmostEqual(r, 20.0, places=4)
        self.assertFalse(expanded)

    def test_single_ug1_returns_none_s(self):
        """Only one Ug1 level in zone — S must be None, R still valid."""
        points = [
            {"ua": 240.0, "ug1": -7.055, "ug2": 239.0, "ia": 48.7},
            {"ua": 249.0, "ug1": -7.055, "ug2": 249.0, "ia": 54.7},
            {"ua": 260.0, "ug1": -7.060, "ug2": 260.0, "ia": 60.9},
        ]
        zone = Zone(ua_min=230, ua_max=270, ug1_min=-7.3, ug1_max=-6.9,
                    ug2_track_ua=True, ug2_offset=0.0)
        s, r, expanded = compute_sr_zone(points, zone, auto_expand=False)
        self.assertIsNone(s, "S must be None when Ug1 spread < threshold")
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 20.0 / 12.2, places=1)
        self.assertFalse(expanded)

    def test_single_ua_returns_none_r(self):
        """Only one Ua level — R must be None, S still valid."""
        points = [
            {"ua": 200.0, "ug1": -3.0, "ug2": 200.0, "ia": 10.0},
            {"ua": 200.0, "ug1": -2.0, "ug2": 200.0, "ia": 20.0},
            {"ua": 200.0, "ug1": -1.0, "ug2": 200.0, "ia": 30.0},
        ]
        zone = Zone(ua_min=190, ua_max=210, ug1_min=-3.5, ug1_max=-0.5,
                    ug2=200.0)
        s, r, expanded = compute_sr_zone(points, zone, auto_expand=False)
        self.assertAlmostEqual(s, 10.0, places=4)
        self.assertIsNone(r, "R must be None when Ua spread < threshold")
        self.assertFalse(expanded)

    def test_empty_returns_none(self):
        """No points → (None, None, False)."""
        zone = Zone(ua_min=0, ua_max=500, ug1_min=-10, ug1_max=0, ug2=0)
        s, r, expanded = compute_sr_zone([], zone)
        self.assertIsNone(s)
        self.assertIsNone(r)
        self.assertFalse(expanded)

    def test_one_point_returns_none(self):
        """Single point → (None, None, False)."""
        points = [{"ua": 200.0, "ug1": -2.0, "ug2": 200.0, "ia": 10.0}]
        zone = Zone(ua_min=100, ua_max=300, ug1_min=-3, ug1_max=-1, ug2=200.0)
        s, r, expanded = compute_sr_zone([], zone)
        self.assertIsNone(s)
        self.assertIsNone(r)
        self.assertFalse(expanded)

    def test_triode_mode(self):
        """is_triode=True: ug2 check skipped, S/R still correct."""
        points = self._grid_points(s_true=15.0, inv_r_true=0.1, ug2=999.0)
        zone = Zone(ua_min=170, ua_max=230, ug1_min=-3.5, ug1_max=-0.5,
                    ug2=200.0, is_triode=True)
        s, r, _ = compute_sr_zone(points, zone)
        self.assertAlmostEqual(s, 15.0, places=4)
        self.assertAlmostEqual(r, 10.0, places=4)

    def test_consistent_with_univariate_single_axis(self):
        """When Ua is constant, result matches compute_s; same for R."""
        points_s = [
            {"ua": 200.0, "ug1": -3.0, "ug2": 200.0, "ia": 5.0},
            {"ua": 200.0, "ug1": -2.0, "ug2": 200.0, "ia": 15.0},
            {"ua": 200.0, "ug1": -1.0, "ug2": 200.0, "ia": 25.0},
        ]
        zone = Zone(ua_min=190, ua_max=210, ug1_min=-3.5, ug1_max=-0.5,
                    ug2=200.0)
        s_zone, _, _ = compute_sr_zone(points_s, zone)
        s_uni = compute_s(points_s, zone)
        self.assertAlmostEqual(s_zone, s_uni, places=6)

    def test_large_grid_accuracy(self):
        """Larger grid (6×5) with known S=8, R=25 — regression must match."""
        points = self._grid_points(
            s_true=8.0, inv_r_true=0.04, offset=20.0,
            ua_vals=range(200, 260, 10),
            ug1_vals=[-5.0, -4.0, -3.0, -2.0, -1.0],
        )
        zone = Zone(ua_min=195, ua_max=255, ug1_min=-5.5, ug1_max=-0.5,
                    ug2=200.0)
        s, r, _ = compute_sr_zone(points, zone)
        self.assertAlmostEqual(s, 8.0, places=4)
        self.assertAlmostEqual(r, 25.0, places=4)

    def test_noisy_grid(self):
        """Grid with small noise: S and R should be close to true values."""
        import random
        random.seed(42)
        s_true, inv_r_true = 12.0, 0.02
        points = []
        for ua in range(200, 260, 10):
            for ug1 in [-6.0, -5.0, -4.0, -3.0]:
                ia = s_true * ug1 + inv_r_true * ua + 100.0
                ia += random.gauss(0, 0.05)
                points.append({"ua": float(ua), "ug1": float(ug1),
                                "ug2": 200.0, "ia": ia})
        zone = Zone(ua_min=195, ua_max=255, ug1_min=-6.5, ug1_max=-2.5,
                    ug2=200.0)
        s, r, _ = compute_sr_zone(points, zone)
        self.assertAlmostEqual(s, 12.0, delta=0.5)
        self.assertAlmostEqual(r, 50.0, delta=2.0)

    def test_real_bug_case(self):
        """Single-Ug1-level scan zone — divergent behaviour pin.

        ``compute_s`` treats the tiny Ug1 noise (~0.005 V) as signal and
        returns ≈ −1840 mA/V (wildly off). ``compute_sr_zone`` correctly
        recognises there's only one Ug1 level and returns ``S = None``
        while keeping R from the Ua sweep.
        """
        points = [
            {"ua": 240.0, "ug1": -7.055, "ug2": 239.0, "ia": 48.7},
            {"ua": 249.0, "ug1": -7.055, "ug2": 249.0, "ia": 54.7},
            {"ua": 260.0, "ug1": -7.060, "ug2": 260.0, "ia": 60.9},
        ]
        zone = Zone(ua_min=230, ua_max=270, ug1_min=-7.3, ug1_max=-6.9,
                    ug2_track_ua=True, ug2_offset=0.0)
        s_legacy = compute_s(points, zone)
        self.assertIsNotNone(s_legacy,
                             "compute_s returns a value for single-Ug1 zones "
                             "(treats Ug1 noise as signal)")
        self.assertGreater(abs(s_legacy), 1000,
                           "compute_s magnitude is wildly off here because "
                           "of Ug1 noise — this is what compute_sr_zone fixes")

        s, r, _ = compute_sr_zone(points, zone, auto_expand=False)
        self.assertIsNone(s, "compute_sr_zone must return None for S "
                             "when only one Ug1 level is present")
        self.assertIsNotNone(r, "R should still be computable from Ua variation")

    # -- auto_expand tests ------------------------------------------------

    def test_expand_ug1_interpolation(self):
        """Zone with narrow Ug1 but neighboring curves → expand, S recovered."""
        s_true, inv_r = 10.0, 0.04
        points = self._grid_points(
            s_true=s_true, inv_r_true=inv_r, offset=50.0,
            ua_vals=(200, 220, 240),
            ug1_vals=(-8.0, -7.0, -6.0),
        )
        zone = Zone(ua_min=190, ua_max=250, ug1_min=-7.3, ug1_max=-6.7,
                    ug2=200.0)
        s, r, expanded = compute_sr_zone(points, zone, auto_expand=True)
        self.assertTrue(expanded, "Zone should have been expanded")
        self.assertIsNotNone(s)
        self.assertAlmostEqual(s, s_true, places=2)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 1.0 / inv_r, places=1)

    def test_no_expand_when_zone_at_edge(self):
        """Zone at the edge of data (no levels below) → no expansion."""
        points = self._grid_points(
            s_true=10.0, inv_r_true=0.04, offset=50.0,
            ua_vals=(200, 220, 240),
            ug1_vals=(-8.0, -7.0, -6.0),
        )
        zone = Zone(ua_min=190, ua_max=250, ug1_min=-8.3, ug1_max=-7.7,
                    ug2=200.0)
        s, r, expanded = compute_sr_zone(points, zone, auto_expand=True)
        self.assertFalse(expanded, "No levels below zone → no expansion")
        self.assertIsNone(s)

    def test_no_expand_when_zone_above_all(self):
        """Zone above all data levels → no expansion."""
        points = self._grid_points(
            s_true=10.0, inv_r_true=0.04, offset=50.0,
            ua_vals=(200, 220, 240),
            ug1_vals=(-8.0, -7.0, -6.0),
        )
        zone = Zone(ua_min=190, ua_max=250, ug1_min=-5.3, ug1_max=-4.7,
                    ug2=200.0)
        s, r, expanded = compute_sr_zone(points, zone, auto_expand=True)
        self.assertFalse(expanded)

    def test_expand_disabled(self):
        """auto_expand=False → no expansion even when interpolation possible."""
        points = self._grid_points(
            s_true=10.0, inv_r_true=0.04, offset=50.0,
            ua_vals=(200, 220, 240),
            ug1_vals=(-8.0, -7.0, -6.0),
        )
        zone = Zone(ua_min=190, ua_max=250, ug1_min=-7.3, ug1_max=-6.7,
                    ug2=200.0)
        s, r, expanded = compute_sr_zone(points, zone, auto_expand=False)
        self.assertFalse(expanded)
        self.assertIsNone(s)

    def test_expand_ua_interpolation(self):
        """Zone with narrow Ua but neighboring Ua values → expand, R recovered."""
        s_true, inv_r = 10.0, 0.04
        points = self._grid_points(
            s_true=s_true, inv_r_true=inv_r, offset=50.0,
            ua_vals=(180, 200, 220),
            ug1_vals=(-3.0, -2.0, -1.0),
        )
        # Triode mode: tests multivariate _try_expand_axis for Ua
        zone = Zone(ua_min=199, ua_max=201, ug1_min=-3.5, ug1_max=-0.5,
                    ug2=200.0, is_triode=True)
        s, r, expanded = compute_sr_zone(points, zone, auto_expand=True)
        self.assertTrue(expanded)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 1.0 / inv_r, places=1)

    def test_expand_real_case_triode_connected(self):
        """Simulates real EL84 triode-connected scan with ug1_step=1V."""
        s_true = 11.0
        points = []
        for ua in range(230, 271, 10):
            for ug1 in [-8.0, -7.0, -6.0, -5.0]:
                ia = s_true * ug1 + 0.03 * ua + 120.0
                points.append({"ua": float(ua), "ug1": float(ug1),
                                "ug2": float(ua), "ia": ia})
        zone = Zone(ua_min=230, ua_max=270, ug1_min=-7.3, ug1_max=-6.9,
                    ug2_track_ua=True, ug2_offset=0.0)
        s, r, expanded = compute_sr_zone(points, zone, auto_expand=True)
        self.assertTrue(expanded)
        self.assertIsNotNone(s)
        self.assertAlmostEqual(s, s_true, places=1)

    def test_no_expand_when_spread_sufficient(self):
        """Zone already has good spread → no expansion."""
        points = self._grid_points(s_true=10.0, inv_r_true=0.05)
        zone = Zone(ua_min=170, ua_max=230, ug1_min=-3.5, ug1_max=-0.5,
                    ug2=200.0)
        s, r, expanded = compute_sr_zone(points, zone, auto_expand=True)
        self.assertFalse(expanded, "Sufficient spread → no expansion needed")
        self.assertIsNotNone(s)
        self.assertIsNotNone(r)


# ======================================================================
# Pentode per-level S/R tests
# ======================================================================

class TestPentodePerLevelSR(unittest.TestCase):
    """Pentode S/R must use per-level univariate approach, not multivariate.

    For pentodes the Ug1->Ia effect is ~100x stronger than Ua->Ia.
    Multivariate regression on unbalanced grids produces garbage R.
    The fix: S via univariate Ia-vs-Ug1, R via per-Ug1-level Ia-vs-Ua median.
    """

    @staticmethod
    def _pentode_grid(s_true=11.0, inv_r_true=0.03, offset=80.0,
                      ua_vals=None, ug1_vals=None, ug2=250.0):
        """Generate pentode-like scan data: Ia = S*Ug1 + (1/R)*Ua + offset."""
        if ua_vals is None:
            ua_vals = list(range(50, 310, 10))
        if ug1_vals is None:
            ug1_vals = [-11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0]
        points = []
        for ug1 in ug1_vals:
            for ua in ua_vals:
                ia = s_true * ug1 + inv_r_true * ua + offset
                if ia < 0:
                    ia = 0.0
                points.append({"ua": float(ua), "ug1": float(ug1),
                                "ug2": ug2, "ia": ia})
        return points

    def test_pentode_balanced_grid(self):
        """Balanced pentode grid: S and R recovered exactly."""
        points = self._pentode_grid(s_true=11.0, inv_r_true=0.03, offset=120.0,
                                    ua_vals=list(range(100, 310, 10)),
                                    ug1_vals=[-8.0, -7.0, -6.0, -5.0])
        zone = Zone(ua_min=190, ua_max=260, ug1_min=-8.5, ug1_max=-4.5,
                    ug2=250.0)
        s, r, _ = compute_sr_zone(points, zone)
        self.assertAlmostEqual(s, 11.0, places=2)
        self.assertAlmostEqual(r, 1.0 / 0.03, places=1)  # 33.3 kOhm

    def test_pentode_unbalanced_grid(self):
        """Unbalanced grid (missing high-Ua points at low-Ug1): R still correct.

        This reproduces the real-world scenario where protection cuts off
        high-current points, leaving an unbalanced Ug1 x Ua grid.
        """
        points = []
        ug2 = 250.0
        s_true, inv_r_true = 11.0, 0.03
        for ug1 in [-8.0, -7.0, -6.0]:
            # More negative Ug1 → lower Ia → full Ua range available
            ua_max = 300 if ug1 <= -7.0 else 250  # -6.0 limited
            for ua in range(50, ua_max + 1, 10):
                ia = s_true * ug1 + inv_r_true * ua + 80.0
                if ia < 0:
                    ia = 0.0
                points.append({"ua": float(ua), "ug1": float(ug1),
                                "ug2": ug2, "ia": ia})

        zone = Zone(ua_min=240, ua_max=260, ug1_min=-7.3, ug1_max=-6.7,
                    ug2=250.0)
        s, r, expanded = compute_sr_zone(points, zone)
        self.assertIsNotNone(s)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(s, 11.0, delta=0.5)
        # R should be close to 33.3 kOhm (a faulty Ua-fit can return 200+)
        self.assertAlmostEqual(r, 1.0 / inv_r_true, delta=5.0)
        self.assertLess(r, 60.0, "R must not be wildly inflated")

    def test_pentode_narrow_zone_expands_ua(self):
        """Narrow zone Ua (20V) with sparse points: expands to get 3 pts."""
        s_true, inv_r_true = 11.0, 0.03
        # Large Ua step (20V) → only 1-2 points per level in a 20V zone
        points = self._pentode_grid(s_true=s_true, inv_r_true=inv_r_true,
                                    ua_vals=list(range(50, 310, 20)),
                                    ug1_vals=[-8.0, -7.0, -6.0])
        zone = Zone(ua_min=240, ua_max=260, ug1_min=-7.3, ug1_max=-6.7,
                    ug2=250.0)
        s, r, expanded = compute_sr_zone(points, zone)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 1.0 / inv_r_true, delta=2.0)

    def test_pentode_r_per_level_median(self):
        """Per-level R uses median: one outlier level doesn't corrupt result."""
        points = []
        ug2 = 250.0
        inv_r = 0.03
        for ug1 in [-9.0, -8.0, -7.0, -6.0, -5.0]:
            for ua in range(100, 301, 10):
                ia = 11.0 * ug1 + inv_r * ua + 80.0
                if ia < 0:
                    ia = 0.0
                points.append({"ua": float(ua), "ug1": float(ug1),
                                "ug2": ug2, "ia": ia})

        zone = Zone(ua_min=190, ua_max=260, ug1_min=-9.5, ug1_max=-4.5,
                    ug2=250.0)
        s, r, _ = compute_sr_zone(points, zone)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 1.0 / inv_r, delta=2.0)

    def test_pentode_vs_triode_connected_different_paths(self):
        """Pentode and triode-connected zones on matching data: both work."""
        s_true, inv_r_true = 11.0, 0.05

        # Pentode data (fixed ug2=250)
        pts_p = []
        for ua in range(180, 261, 10):
            for ug1 in [-8.0, -7.0, -6.0]:
                ia = s_true * ug1 + inv_r_true * ua + 80.0
                pts_p.append({"ua": float(ua), "ug1": float(ug1),
                               "ug2": 250.0, "ia": ia})
        zone_p = Zone(ua_min=175, ua_max=265, ug1_min=-8.5, ug1_max=-5.5,
                      ug2=250.0)
        s_p, r_p, _ = compute_sr_zone(pts_p, zone_p)

        # Triode-connected data (ug2 = ua)
        pts_t = []
        for ua in range(180, 261, 10):
            for ug1 in [-8.0, -7.0, -6.0]:
                ia = s_true * ug1 + inv_r_true * ua + 80.0
                pts_t.append({"ua": float(ua), "ug1": float(ug1),
                               "ug2": float(ua), "ia": ia})
        zone_t = Zone(ua_min=175, ua_max=265, ug1_min=-8.5, ug1_max=-5.5,
                      ug2_track_ua=True, ug2_offset=0.0)
        s_t, r_t, _ = compute_sr_zone(pts_t, zone_t)

        # Both should recover S and R
        self.assertAlmostEqual(s_p, s_true, delta=0.5)
        self.assertAlmostEqual(s_t, s_true, delta=0.5)
        self.assertAlmostEqual(r_p, 1.0 / inv_r_true, delta=2.0)
        self.assertAlmostEqual(r_t, 1.0 / inv_r_true, delta=2.0)

    def test_pentode_triode_connected_unchanged(self):
        """Triode-connected pentode still uses multivariate (not per-level)."""
        s_true, inv_r_true = 11.0, 0.5  # triode-like R = 2 kOhm
        points = []
        for ua in [200, 220, 240, 260]:
            for ug1 in [-8.0, -7.0, -6.0]:
                ia = s_true * ug1 + inv_r_true * ua + 80.0
                points.append({"ua": float(ua), "ug1": float(ug1),
                                "ug2": float(ua), "ia": ia})
        zone = Zone(ua_min=195, ua_max=265, ug1_min=-8.5, ug1_max=-5.5,
                    ug2_track_ua=True, ug2_offset=0.0)
        s, r, _ = compute_sr_zone(points, zone)
        self.assertAlmostEqual(s, s_true, places=2)
        self.assertAlmostEqual(r, 1.0 / inv_r_true, places=2)

    def test_pentode_zone_enough_points_no_expand(self):
        """Zone with enough points per level: no expansion beyond zone."""
        from lm19.analysis import _expand_for_r_level
        # 5 points in zone, step 10V
        pts = [{"ua": float(ua), "ug1": -7.0, "ia": 40.0 + 0.03 * ua}
               for ua in range(200, 310, 10)]
        selected = _expand_for_r_level(pts, ua_min=220, ua_max=280)
        uas = [p["ua"] for p in selected]
        self.assertTrue(all(220 <= ua <= 280 for ua in uas))
        self.assertGreaterEqual(len(selected), 3)

    def test_pentode_expand_avoids_knee(self):
        """Downward expansion stops when slope indicates knee region."""
        from lm19.analysis import _expand_for_r_level
        # Linear region: slope ~0.03 mA/V (R=33k)
        # Knee region (Ua<100): slope ~0.5 mA/V
        pts = []
        for ua in range(50, 300, 10):
            if ua < 100:
                ia = 0.5 * ua  # steep knee
            else:
                ia = 0.03 * ua + 47.0  # flat pentode region
            pts.append({"ua": float(ua), "ug1": -7.0, "ia": ia})
        # Zone 200-220: only 3 points (200,210,220) — enough, no expand
        sel = _expand_for_r_level(pts, ua_min=200, ua_max=220)
        self.assertEqual(len(sel), 3)
        # Zone 250-260: only 2 points — expand up first, then maybe down
        sel2 = _expand_for_r_level(pts, ua_min=250, ua_max=260)
        self.assertGreaterEqual(len(sel2), 3)
        # Should NOT include knee points (ua < 100)
        self.assertTrue(all(p["ua"] >= 100 for p in sel2))

    def test_pentode_expand_prefers_upward(self):
        """Expansion adds points above zone before trying below."""
        from lm19.analysis import _expand_for_r_level
        pts = [{"ua": float(ua), "ug1": -7.0, "ia": 40.0 + 0.03 * ua}
               for ua in range(100, 310, 10)]
        # Zone with only 1 point (250)
        selected = _expand_for_r_level(pts, ua_min=245, ua_max=255)
        self.assertGreaterEqual(len(selected), 3)
        # First expansion should be upward (260, 270)
        uas = sorted(p["ua"] for p in selected)
        above_zone = [ua for ua in uas if ua > 255]
        self.assertGreaterEqual(len(above_zone), 1,
                                "Should expand upward first")

    def test_pentode_expand_only_below(self):
        """No points above zone: expansion goes downward with knee check."""
        from lm19.analysis import _expand_for_r_level
        # Data only up to 260, zone at top edge 250-260: only 2 pts (250, 260)
        pts = [{"ua": float(ua), "ug1": -7.0, "ia": 40.0 + 0.03 * ua}
               for ua in range(100, 270, 10)]
        selected = _expand_for_r_level(pts, ua_min=250, ua_max=260)
        self.assertGreaterEqual(len(selected), 3)
        # Must have picked a point below 250 (e.g. 240)
        below_zone = [p["ua"] for p in selected if p["ua"] < 250]
        self.assertGreaterEqual(len(below_zone), 1)

    def test_pentode_zero_points_in_zone(self):
        """Zone Ua misses all data points: R still computed via expansion."""
        s_true, inv_r_true = 11.0, 0.03
        # Data at Ua = 100,120,...,300 — no point at 245 or 255
        points = self._pentode_grid(s_true=s_true, inv_r_true=inv_r_true,
                                    offset=120.0,
                                    ua_vals=list(range(100, 310, 20)),
                                    ug1_vals=[-8.0, -7.0, -6.0])
        # Zone 245-255: zero Ua points inside
        zone = Zone(ua_min=245, ua_max=255, ug1_min=-8.5, ug1_max=-5.5,
                    ug2=250.0)
        s, r, _ = compute_sr_zone(points, zone)
        # R should still be computed via expansion
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 1.0 / inv_r_true, delta=5.0)

    def test_pentode_expand_adds_one_point(self):
        """Zone has 2 points (one short of min): expand adds exactly 1."""
        from lm19.analysis import _expand_for_r_level
        # Step 10V, zone 240-260 captures 240,250,260 = 3 pts — enough
        # Shrink zone to 245-255: only 250 = 1 pt, needs 2 more
        # Widen to 240-255: 240,250 = 2 pts, needs 1 more
        pts = [{"ua": float(ua), "ug1": -7.0, "ia": 40.0 + 0.03 * ua}
               for ua in range(100, 310, 10)]
        selected = _expand_for_r_level(pts, ua_min=240, ua_max=255)
        # 240 and 250 in zone, should add 260 from above
        self.assertEqual(len(selected), 3)
        uas = sorted(p["ua"] for p in selected)
        self.assertAlmostEqual(uas[0], 240)
        self.assertAlmostEqual(uas[1], 250)
        self.assertAlmostEqual(uas[2], 260)

    def test_pentode_real_measurement_data(self):
        """Smoke test on real EL84 pentode scan data (converted fixture)."""
        from tests._real_data import load_points
        points = load_points("pentode_EL84_ER_L1_real.json")
        if not points:
            self.skipTest("EL84 ER L1 fixture not available")

        # Zone snapshot taken from the original scan's `_zone` metadata; the
        # converted fixture intentionally strips run-specific metadata, so
        # the operating window is pinned here.
        zone = Zone(
            ua_min=240.0, ua_max=260.0,
            ug1_min=-7.3, ug1_max=-6.9,
            ug2=250.0, is_triode=False,
            ug2_track_ua=False, ug2_offset=0.0,
        )
        s, r, _ = compute_sr_zone(points, zone)

        # EL84 pentode: S ≈ 11 mA/V, R ≈ 15-40 kOhm
        self.assertIsNotNone(s)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(s, 11.0, delta=2.0)
        self.assertGreater(r, 10.0)
        self.assertLess(r, 60.0)
        # Old bug produced R > 200 kOhm
        self.assertLess(r, 100.0, "R must not be wildly inflated (old bug)")


# ======================================================================
# Quality tests
# ======================================================================

class TestQuality(unittest.TestCase):
    """Test tube quality scoring."""

    def test_quality_strong(self):
        """Tube with 120% Ia and 115% S should be 'Strong'."""
        lamp = _FakeLampConfig(ua=250, ug1=-8.5, ia=48.0, s=11.0)
        points = [{"ua": 250, "ug1": -8.5, "ug2": 170, "ia": 57.6}]  # 120%
        srk = {"s": 12.65, "r": 20.0, "k": 0.5}  # 115%
        q = compute_quality(points, lamp, srk)
        self.assertEqual(q.verdict, "Strong")

    def test_quality_good(self):
        """Tube with 95% Ia and 90% S should be 'Good'."""
        lamp = _FakeLampConfig(ua=250, ug1=-8.5, ia=48.0, s=11.0)
        points = [{"ua": 250, "ug1": -8.5, "ug2": 170, "ia": 45.6}]  # 95%
        srk = {"s": 9.9, "r": 20.0, "k": 0.5}  # 90%
        q = compute_quality(points, lamp, srk)
        self.assertEqual(q.verdict, "Good")

    def test_quality_weak(self):
        """Tube with 60% Ia should be 'Weak'."""
        lamp = _FakeLampConfig(ua=250, ug1=-8.5, ia=48.0, s=11.0)
        points = [{"ua": 250, "ug1": -8.5, "ug2": 170, "ia": 28.8}]  # 60%
        srk = {"s": 6.6, "r": 20.0, "k": 0.5}  # 60%
        q = compute_quality(points, lamp, srk)
        self.assertEqual(q.verdict, "Weak")

    def test_quality_replace(self):
        """Tube with 30% Ia should be 'Replace'."""
        lamp = _FakeLampConfig(ua=250, ug1=-8.5, ia=48.0, s=11.0)
        points = [{"ua": 250, "ug1": -8.5, "ug2": 170, "ia": 14.4}]  # 30%
        srk = {"s": 3.3, "r": 20.0, "k": 0.5}  # 30%
        q = compute_quality(points, lamp, srk)
        self.assertEqual(q.verdict, "Replace")

    def test_quality_no_config(self):
        """No lamp config should return N/A."""
        q = compute_quality([], None)
        self.assertEqual(q.verdict, "N/A")


class TestMatching(unittest.TestCase):
    """Test tube matching analysis."""

    def test_identical_tubes(self):
        """Identical tubes should match 100%."""
        points = [
            {"ua": 100, "ug1": -1.0, "ia": 5.0},
            {"ua": 200, "ug1": -1.0, "ia": 12.0},
            {"ua": 100, "ug1": -2.0, "ia": 3.0},
            {"ua": 200, "ug1": -2.0, "ia": 8.0},
        ]
        result = compute_matching(points, points)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.mean_delta, 0.0, places=5)
        self.assertAlmostEqual(result.match_pct, 100.0, places=1)

    def test_different_tubes(self):
        """Different tubes should have non-zero delta."""
        points_a = [
            {"ua": 100, "ug1": -1.0, "ia": 5.0},
            {"ua": 200, "ug1": -1.0, "ia": 12.0},
        ]
        points_b = [
            {"ua": 100, "ug1": -1.0, "ia": 6.0},
            {"ua": 200, "ug1": -1.0, "ia": 14.0},
        ]
        result = compute_matching(points_a, points_b)
        self.assertIsNotNone(result)
        self.assertGreater(result.mean_delta, 0)
        self.assertLess(result.match_pct, 100.0)

    def test_no_common_ug1(self):
        """No common Ug1 should return None."""
        a = [{"ua": 100, "ug1": -1.0, "ia": 5.0}]
        b = [{"ua": 100, "ug1": -5.0, "ia": 3.0}]
        self.assertIsNone(compute_matching(a, b))

    def test_matching_curves(self):
        """Matching curves should return data for common Ug1."""
        points_a = [
            {"ua": 100, "ug1": -1.0, "ia": 5.0},
            {"ua": 200, "ug1": -1.0, "ia": 12.0},
        ]
        points_b = [
            {"ua": 100, "ug1": -1.0, "ia": 6.0},
            {"ua": 200, "ug1": -1.0, "ia": 13.0},
        ]
        curves = compute_matching_curves(points_a, points_b)
        self.assertEqual(len(curves), 1)
        self.assertEqual(curves[0]["ug1"], -1.0)
        self.assertGreater(len(curves[0]["ua_values"]), 0)


class TestAgingTrend(unittest.TestCase):
    """Test aging trend extraction."""

    def test_sorted_by_timestamp(self):
        """Results should be sorted by timestamp."""
        measurements = [
            {"timestamp": "2026-02-03", "srk": {"s": 10, "r": 20, "k": 200},
             "points": [], "name": "C"},
            {"timestamp": "2026-02-01", "srk": {"s": 11, "r": 22, "k": 242},
             "points": [], "name": "A"},
            {"timestamp": "2026-02-02", "srk": {"s": 10.5, "r": 21, "k": 220},
             "points": [], "name": "B"},
        ]
        trend = compute_aging_trend(measurements)
        self.assertEqual(len(trend), 3)
        self.assertEqual(trend[0].timestamp, "2026-02-01")
        self.assertEqual(trend[1].timestamp, "2026-02-02")
        self.assertEqual(trend[2].timestamp, "2026-02-03")

    def test_srk_extraction(self):
        """S/R/K should be extracted from measurement data."""
        measurements = [
            {"timestamp": "2026-01-01", "srk": {"s": 11.2, "r": 15.0, "k": 168.0},
             "points": [], "name": "Test"},
        ]
        trend = compute_aging_trend(measurements)
        self.assertEqual(len(trend), 1)
        self.assertAlmostEqual(trend[0].s, 11.2)
        self.assertAlmostEqual(trend[0].r, 15.0)

    def test_ia_at_operating_point(self):
        """Should find Ia nearest to nominal operating point."""
        lamp = _FakeLampConfig(ua=250, ug1=-8.5, ug2=170, ia=48)
        measurements = [
            {"timestamp": "2026-01-01", "srk": {},
             "points": [
                 {"ua": 250, "ug1": -8.5, "ug2": 170, "ia": 45.0},
                 {"ua": 100, "ug1": -2.0, "ug2": 170, "ia": 10.0},
             ], "name": "Test"},
        ]
        trend = compute_aging_trend(measurements, lamp)
        self.assertAlmostEqual(trend[0].ia_at_op, 45.0)

    def test_empty_measurements(self):
        """Empty list should return empty trend."""
        self.assertEqual(compute_aging_trend([]), [])


# ======================================================================
# Topology / is_triode tests
# ======================================================================

class TestTopology(unittest.TestCase):
    """Test topology field and is_triode across modules."""

    # -- config.py --

    def test_load_lamps_has_topology(self):
        """Every lamp in lamps.json should have a topology field."""
        from lm19.config import load_lamps
        lamps = load_lamps()
        for lamp in lamps:
            self.assertIn(lamp.topology, ("triode", "pentode"),
                          f"{lamp.tube_type}: topology={lamp.topology!r}")

    def test_triode_is_triode(self):
        """ECC83 (12AX7) should be a triode."""
        from lm19.config import load_lamps, find_lamp
        lamps = load_lamps()
        lamp = find_lamp(lamps, "ECC83")
        self.assertIsNotNone(lamp)
        self.assertEqual(lamp.topology, "triode")
        self.assertTrue(lamp.is_triode)

    def test_pentode_is_not_triode(self):
        """EL84 should be a pentode."""
        from lm19.config import load_lamps, find_lamp
        lamps = load_lamps()
        lamp = find_lamp(lamps, "EL84")
        self.assertIsNotNone(lamp)
        self.assertEqual(lamp.topology, "pentode")
        self.assertFalse(lamp.is_triode)

    def test_topology_default_pentode(self):
        """Missing topology should default to pentode."""
        from lm19.config import LampConfig
        cfg = LampConfig(
            tube_type="TEST", socket="", anodes=1, warmup_s=120, topology=TOPOLOGY_PENTODE,
            uh=6.3, ih=0, ug1=-2, ua=250, ia=10, ug2=0, ig2=0,
            s=0, r=0, k=0, ranges={}, limits={},
        )
        self.assertFalse(cfg.is_triode)

    # -- analysis.py --

    def test_in_zone_triode_skips_ug2(self):
        """Zone with is_triode=True should accept any ug2 value."""
        points = [
            {"ua": 200, "ug1": -2.0, "ug2": 999.0, "ia": 5.0},
            {"ua": 200, "ug1": -2.0, "ug2": 0.0, "ia": 6.0},
        ]
        zone = Zone(ua_min=100, ua_max=300, ug1_min=-3, ug1_max=-1,
                    ug2=200.0, is_triode=True)
        # Both points should be in zone (ug2 check skipped)
        self.assertEqual(count_in_zone(points, zone), 2)

    def test_in_zone_pentode_checks_ug2(self):
        """Zone without is_triode should filter by ug2."""
        points = [
            {"ua": 200, "ug1": -2.0, "ug2": 200.0, "ia": 5.0},  # match
            {"ua": 200, "ug1": -2.0, "ug2": 999.0, "ia": 6.0},  # no match
        ]
        zone = Zone(ua_min=100, ua_max=300, ug1_min=-3, ug1_max=-1,
                    ug2=200.0, is_triode=False)
        self.assertEqual(count_in_zone(points, zone), 1)

    # -- csv_export.py --

    def test_csv_triode_no_ug2_columns(self):
        """Triode CSV should omit Ug2, Ig2, Pg2 columns."""
        points = [{"ua": 100, "ug1": -1, "ia": 5, "uh": 6.3, "ih": 0.3}]
        csv = format_csv(points, separator=";", is_triode=True)
        lines = [l for l in csv.strip().split("\n") if not l.startswith("#")]
        header = lines[0]
        self.assertNotIn("Ug2", header)
        self.assertNotIn("Ig2", header)
        self.assertNotIn("Pg2", header)
        self.assertIn("Ua", header)
        self.assertIn("Ia", header)

    def test_csv_pentode_has_ug2_columns(self):
        """Pentode CSV should include Ug2, Ig2 columns."""
        points = [{"ua": 100, "ug1": -1, "ug2": 170, "ia": 5,
                    "ig2": 1, "uh": 6.3, "ih": 0.3}]
        csv = format_csv(points, separator=";", is_triode=False)
        lines = [l for l in csv.strip().split("\n") if not l.startswith("#")]
        self.assertIn("Ug2", lines[0])
        self.assertIn("Ig2", lines[0])

    def test_matrix_triode_no_ug2_header(self):
        """Triode matrix should not mention Ug2 in section header."""
        points = [
            {"ua": 100, "ug1": -1, "ia": 5},
            {"ua": 200, "ug1": -1, "ia": 12},
            {"ua": 100, "ug1": -2, "ia": 3},
            {"ua": 200, "ug1": -2, "ia": 8},
        ]
        csv = format_matrix(points, separator=";", is_triode=True)
        self.assertNotIn("Ug2=", csv)

    def test_multi_csv_triode(self):
        """Multi CSV in triode mode should omit Ug2/Ig2 columns."""
        entries = [{"lamp_type": "ECC83", "lamp_id": "L1", "name": "T",
                     "points": [{"ua": 100, "ug1": -1, "ia": 5,
                                  "uh": 6.3, "ih": 0.3}]}]
        csv = format_multi_csv(entries, separator=";", is_triode=True)
        lines = [l for l in csv.strip().split("\n") if not l.startswith("#")]
        self.assertNotIn("Ug2", lines[0])
        self.assertNotIn("Ig2", lines[0])

    # -- quality.py --

    def test_quality_triode(self):
        """Quality scoring for triode should work without Ug2."""
        from lm19.config import LampConfig
        lamp = LampConfig(
            tube_type="TEST_TRIODE", socket="G", anodes=1, warmup_s=120, topology=TOPOLOGY_TRIODE,
            uh=6.3, ih=0, ug1=-2, ua=250, ia=10, ug2=0, ig2=0,
            s=16, r=625, k=999, ranges={}, limits={},
        )
        points = [{"ua": 250, "ug1": -2.0, "ug2": 0, "ia": 10.0}]
        srk = {"s": 16.0, "r": 625.0, "k": 999.0}
        q = compute_quality(points, lamp, srk)
        self.assertEqual(q.verdict, "Good")  # 100% Ia, 100% S → avg 100%
        self.assertAlmostEqual(q.ia_pct, 100.0, places=0)
        self.assertAlmostEqual(q.s_pct, 100.0, places=0)


# ======================================================================
# PlotRenderer static math methods
# ======================================================================

class TestClusterNominal(unittest.TestCase):
    """``cluster_nominal``: direct call via ``lm19.plotting.grids`` and
    via the ``PlotRenderer._cluster_nominal`` static-method delegate."""

    def test_empty(self):
        from lm19.plotting.grids import cluster_nominal
        self.assertEqual(cluster_nominal([]), [])
        self.assertEqual(PlotRenderer._cluster_nominal([]), [])

    def test_no_merge(self):
        """Values far apart should remain separate."""
        from lm19.plotting.grids import cluster_nominal
        vals = [1.0, 2.0, 3.0]
        result = cluster_nominal(vals, threshold=0.1)
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_merge_close(self):
        """Values within threshold should merge into one representative."""
        from lm19.plotting.grids import cluster_nominal
        vals = [1.49, 1.50, 1.51]
        result = cluster_nominal(vals, threshold=0.03)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 1.49)

    def test_mixed(self):
        """Mix of close and far values."""
        from lm19.plotting.grids import cluster_nominal
        vals = [1.0, 1.01, 2.0, 2.005, 3.0]
        result = cluster_nominal(vals, threshold=0.02)
        self.assertEqual(len(result), 3)

    def test_unsorted_input(self):
        """Should sort before processing."""
        from lm19.plotting.grids import cluster_nominal
        vals = [3.0, 1.0, 2.0]
        result = cluster_nominal(vals, threshold=0.02)
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_single_value(self):
        from lm19.plotting.grids import cluster_nominal
        result = cluster_nominal([5.0])
        self.assertEqual(result, [5.0])


class TestNominalKey(unittest.TestCase):
    """``nominal_key``: direct call via ``lm19.plotting.grids`` and via
    the ``PlotRenderer._nominal_key`` static-method delegate."""

    def test_exact_match(self):
        from lm19.plotting.grids import nominal_key
        nominals = [1.0, 2.0, 3.0]
        self.assertEqual(nominal_key(2.0, nominals), 2.0)

    def test_nearest(self):
        from lm19.plotting.grids import nominal_key
        nominals = [1.0, 2.0, 3.0]
        self.assertEqual(nominal_key(1.8, nominals), 2.0)

    def test_empty_nominals(self):
        """Should return the value itself when no nominals."""
        from lm19.plotting.grids import nominal_key
        self.assertAlmostEqual(nominal_key(5.5, []), 5.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestUnifiedRSignPolicy:
    """ML-098: physical Rp is positive — a negative dIa/dUa slope (noise)
    must yield None with a WARNING on every R path; one path used to
    return a negative R silently."""

    def test_compute_r_negative_slope_none_and_warns(self, caplog):
        import logging
        from lm19.analysis import Zone, compute_r
        zone = Zone(ua_min=0, ua_max=300, ug1_min=-10, ug1_max=0,
                    is_triode=True)
        pts = [{"ua": 100.0, "ug1": -2.0, "ia": 10.0, "ug2": 0.0},
               {"ua": 200.0, "ug1": -2.0, "ia": 5.0, "ug2": 0.0}]  # Ia falls
        with caplog.at_level(logging.WARNING, logger="lm19.analysis"):
            r = compute_r(pts, zone)
        assert r is None, "negative R returned silently"
        assert any("negative" in rec.getMessage()
                   for rec in caplog.records)

    def test_compute_srk_direct_negative_slope_none(self):
        from lm19.analysis import compute_srk_direct
        s, r, k = compute_srk_direct(
            s_ug1=[-3.0, -2.0], s_ia=[5.0, 8.0],
            r_ua=[100.0, 200.0], r_ia=[10.0, 4.0])   # Ia falls with Ua
        assert r is None
        assert k is None

    def test_positive_slope_unchanged(self):
        import pytest
        from lm19.analysis import compute_srk_direct
        s, r, k = compute_srk_direct(
            s_ug1=[-3.0, -2.0], s_ia=[5.0, 8.0],
            r_ua=[100.0, 200.0], r_ia=[4.0, 10.0])
        assert r == pytest.approx((200.0 - 100.0) / (10.0 - 4.0))
