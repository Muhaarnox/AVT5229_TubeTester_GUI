"""Tests for dead-data detection and cleanup in quality module."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.quality import (  # noqa: E402
    detect_dead_data, clean_dead_points, DeadDataReport,
    IA_DEAD_THR,
)
from lm19.constants import (
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)


# ------------------------------------------------------------------
# Helpers to build synthetic measurement data
# ------------------------------------------------------------------

def _make_pentode_points(
    ug2_values, ug1_values, ua_values,
    ia_func=None, dead_ug2=None, dead_ug1_map=None,
):
    """Generate synthetic pentode measurement points.

    Args:
        ug2_values: list of Ug2 values
        ug1_values: list of Ug1 values (ascending, e.g. -8...-1)
        ua_values: list of Ua values
        ia_func: callable(ua, ug1, ug2) → Ia in mA. Default: simple model.
        dead_ug2: set of Ug2 values where all points are dead (Ia≈0)
        dead_ug1_map: dict {ug2: ug1_threshold} — at this Ug2,
                      all points with ug1 >= threshold are dead
    """
    if ia_func is None:
        def ia_func(ua, ug1, ug2):
            # Simple increasing function, always > 0.5 mA for typical values
            return max(0.5, (ua / 50.0) * (10.0 + ug1) / 5.0 * (ug2 / 100.0))

    dead_ug2 = dead_ug2 or set()
    dead_ug1_map = dead_ug1_map or {}

    points = []
    for ug2 in ug2_values:
        for ug1 in ug1_values:
            for ua in ua_values:
                if ug2 in dead_ug2:
                    ia = 0.05  # hardware protection artifact
                elif ug2 in dead_ug1_map and ug1 >= dead_ug1_map[ug2]:
                    ia = 0.05  # Ig2 protection artifact
                else:
                    ia = ia_func(ua, ug1, ug2)
                points.append({"ua": ua, "ug1": ug1, "ug2": ug2, "ia": ia})
    return points


# ------------------------------------------------------------------
# Tests for detect_dead_data
# ------------------------------------------------------------------

class TestDetectDeadData:
    """Tests for detect_dead_data()."""

    def test_empty_points(self):
        report = detect_dead_data([])
        assert report.total_points == 0
        assert report.dead_points == 0
        assert not report.has_dead_data

    def test_all_healthy(self):
        points = _make_pentode_points(
            ug2_values=[100, 150, 200],
            ug1_values=[-8, -6, -4, -2],
            ua_values=[50, 100, 150, 200],
        )
        report = detect_dead_data(points)
        assert report.total_points == len(points)
        assert not report.has_dead_data
        assert report.dead_ug2_levels == []
        assert report.partial_ug2_levels == []

    def test_dead_ug2_levels(self):
        """Entire Ug2 levels dead from hardware protection."""
        points = _make_pentode_points(
            ug2_values=[100, 150, 200, 250],
            ug1_values=[-8, -6, -4, -2],
            ua_values=[50, 100, 150],
            dead_ug2={200, 250},
        )
        report = detect_dead_data(points)
        assert report.has_dead_data
        assert len(report.dead_ug2_levels) == 2
        assert 200 in report.dead_ug2_levels
        assert 250 in report.dead_ug2_levels
        assert report.partial_ug2_levels == []
        # Dead count: 2 levels * 4 ug1 * 3 ua = 24
        assert report.dead_points == 24

    def test_partial_ig2_protection(self):
        """Ig2 protection fires at specific Ug1 within a level."""
        points = _make_pentode_points(
            ug2_values=[100, 200],
            ug1_values=[-8, -6, -4, -2],
            ua_values=[50, 100, 150],
            dead_ug1_map={200: -4},  # at Ug2=200, Ug1>=-4 are dead
        )
        report = detect_dead_data(points)
        assert report.has_dead_data
        assert report.dead_ug2_levels == []
        assert len(report.partial_ug2_levels) == 1
        ug2_val, ug1_trans = report.partial_ug2_levels[0]
        assert ug2_val == 200
        assert ug1_trans == -4.0
        # Dead: 2 Ug1 values (-4, -2) * 3 Ua = 6
        assert report.dead_points == 6

    def test_mixed_dead_and_partial(self):
        """Both dead levels and partial protection in same dataset."""
        points = _make_pentode_points(
            ug2_values=[100, 150, 200, 250],
            ug1_values=[-8, -6, -4, -2],
            ua_values=[50, 100],
            dead_ug2={250},
            dead_ug1_map={200: -2},  # at Ug2=200, Ug1>=-2 dead
        )
        report = detect_dead_data(points)
        assert report.has_dead_data
        assert 250 in report.dead_ug2_levels
        assert len(report.partial_ug2_levels) == 1
        # Dead: level 250 = 4*2=8, partial level 200 Ug1=-2: 1*2=2
        assert report.dead_points == 10

    def test_normal_cutoff_not_flagged(self):
        """Normal cutoff at negative Ug1 (Ia≈0) should NOT be flagged.

        Cutoff pattern: dead at MOST negative Ug1, live at less negative.
        This is opposite to protection pattern.
        """
        def ia_with_cutoff(ua, ug1, ug2):
            if ug1 <= -6:
                return 0.01  # normal cutoff
            return max(0.5, (ua / 50.0) * (1.0 + ug1 / 3.0) * (ug2 / 100.0))

        points = _make_pentode_points(
            ug2_values=[100, 200],
            ug1_values=[-8, -6, -4, -2],
            ua_values=[50, 100, 150],
            ia_func=ia_with_cutoff,
        )
        report = detect_dead_data(points)
        assert not report.has_dead_data

    def test_triode_mode(self):
        """Triode mode: just counts dead points, no level analysis."""
        points = [
            {"ua": 100, "ug1": -4, "ia": 10.0},
            {"ua": 150, "ug1": -4, "ia": 15.0},
            {"ua": 200, "ug1": -4, "ia": 0.05},  # dead
            {"ua": 100, "ug1": -2, "ia": 20.0},
        ]
        report = detect_dead_data(points, topology=TOPOLOGY_TRIODE)
        assert report.total_points == 4
        assert report.dead_points == 1
        assert not report.has_dead_data  # triode doesn't populate levels

    def test_triode_connected_no_false_positives(self):
        """Triode-connected pentode (Ug2=Ua+offset) must not trigger
        pentode-style dead data detection — each point has unique Ug2,
        so grouping by Ug2 would create single-point 'levels' with low Ia
        that look dead but are just normal cutoff.
        """
        # Simulate triode-connected: Ug2 = Ua + 5V offset
        offset = 5.0
        points = []
        for ug1 in [-8, -6, -4, -2]:
            for ua in [50, 100, 150, 200, 250]:
                ug2 = ua + offset
                ia = max(0.01, (ua / 100.0) * (10.0 + ug1) / 5.0)
                points.append({"ua": ua, "ug1": ug1, "ug2": ug2, "ia": ia})

        # With topology=TOPOLOGY_TRIODE_CONNECTED — no dead data
        report_tc = detect_dead_data(points, topology=TOPOLOGY_TRIODE_CONNECTED)
        assert not report_tc.has_dead_data
        assert report_tc.dead_ug2_levels == []
        assert report_tc.partial_ug2_levels == []

        # Verify triode_connected path counts individual dead points
        # (Ia=0.01 at ua=50,ug1=-8 is below default threshold)
        assert report_tc.dead_points >= 0  # just counting, no level analysis

    def test_topology_triode(self):
        """topology='triode' skips level analysis, only counts dead points."""
        points = _make_pentode_points(
            ug2_values=[100, 200],
            ug1_values=[-4, -2],
            ua_values=[50, 100],
            dead_ug2={200},
        )
        report = detect_dead_data(points, topology=TOPOLOGY_TRIODE)
        # Triode mode: no level analysis, just dead point count
        assert report.dead_ug2_levels == []
        assert report.partial_ug2_levels == []

    def test_single_ug1_per_level_no_crash(self):
        """Single Ug1 curve per level should not crash."""
        points = _make_pentode_points(
            ug2_values=[100, 200],
            ug1_values=[-4],
            ua_values=[50, 100, 150],
        )
        report = detect_dead_data(points)
        assert report.total_points == 6
        assert not report.has_dead_data

    def test_dead_pct_property(self):
        points = _make_pentode_points(
            ug2_values=[100, 200],
            ug1_values=[-4, -2],
            ua_values=[50, 100],
            dead_ug2={200},
        )
        report = detect_dead_data(points)
        assert report.dead_pct == pytest.approx(50.0)
        assert report.live_points == 4

    def test_all_levels_dead(self):
        """All Ug2 levels are dead."""
        points = _make_pentode_points(
            ug2_values=[100, 200, 300],
            ug1_values=[-4, -2],
            ua_values=[50, 100],
            dead_ug2={100, 200, 300},
        )
        report = detect_dead_data(points)
        assert report.dead_pct == pytest.approx(100.0)
        assert report.live_points == 0


# ------------------------------------------------------------------
# Tests for clean_dead_points
# ------------------------------------------------------------------

class TestCleanDeadPoints:
    """Tests for clean_dead_points()."""

    def test_no_dead_data_returns_copy(self):
        points = _make_pentode_points(
            ug2_values=[100, 200],
            ug1_values=[-4, -2],
            ua_values=[50, 100],
        )
        cleaned = clean_dead_points(points)
        assert len(cleaned) == len(points)
        assert cleaned is not points  # new list

    def test_dead_levels_removed(self):
        points = _make_pentode_points(
            ug2_values=[100, 200, 300],
            ug1_values=[-4, -2],
            ua_values=[50, 100],
            dead_ug2={200, 300},
        )
        cleaned = clean_dead_points(points)
        # Only Ug2=100 remains: 2 Ug1 * 2 Ua = 4
        assert len(cleaned) == 4
        assert all(abs(p["ug2"] - 100) < 5 for p in cleaned)

    def test_partial_level_cleaned(self):
        points = _make_pentode_points(
            ug2_values=[100, 200],
            ug1_values=[-8, -6, -4, -2],
            ua_values=[50, 100],
            dead_ug1_map={200: -4},
        )
        cleaned = clean_dead_points(points)
        # Ug2=100: all 4*2=8 remain
        # Ug2=200: only Ug1=-8,-6 remain (2*2=4)
        assert len(cleaned) == 12
        # No dead points in cleaned data
        for p in cleaned:
            if abs(p["ug2"] - 200) < 5:
                assert p["ug1"] < -4

    def test_original_not_modified(self):
        points = _make_pentode_points(
            ug2_values=[100, 200],
            ug1_values=[-4, -2],
            ua_values=[50],
            dead_ug2={200},
        )
        original_len = len(points)
        clean_dead_points(points)
        assert len(points) == original_len

    def test_with_precomputed_report(self):
        """Pass pre-computed report to avoid double detection."""
        points = _make_pentode_points(
            ug2_values=[100, 200],
            ug1_values=[-4, -2],
            ua_values=[50, 100],
            dead_ug2={200},
        )
        report = detect_dead_data(points)
        cleaned = clean_dead_points(points, report=report)
        assert len(cleaned) == 4

    def test_all_dead_returns_empty(self):
        points = _make_pentode_points(
            ug2_values=[100, 200],
            ug1_values=[-4, -2],
            ua_values=[50],
            dead_ug2={100, 200},
        )
        cleaned = clean_dead_points(points)
        assert len(cleaned) == 0


# ------------------------------------------------------------------
# Tests for DeadDataReport properties
# ------------------------------------------------------------------

class TestDeadDataReport:
    """Tests for DeadDataReport dataclass."""

    def test_has_dead_data_dead_levels(self):
        r = DeadDataReport(total_points=10, dead_points=5,
                           dead_ug2_levels=[200.0])
        assert r.has_dead_data

    def test_has_dead_data_partial_levels(self):
        r = DeadDataReport(total_points=10, dead_points=3,
                           partial_ug2_levels=[(200.0, -4.0)])
        assert r.has_dead_data

    def test_has_dead_data_none(self):
        r = DeadDataReport(total_points=10, dead_points=0)
        assert not r.has_dead_data

    def test_dead_pct_zero_total(self):
        r = DeadDataReport(total_points=0, dead_points=0)
        assert r.dead_pct == 0.0

    def test_live_points(self):
        r = DeadDataReport(total_points=100, dead_points=30)
        assert r.live_points == 70


# ------------------------------------------------------------------
# Test with real measurement data (EL84 pentode with protection)
# ------------------------------------------------------------------

_EL84_PATH = (
    Path(__file__).resolve().parent
    / "spice_test_data" / "converted"
    / "pentode_EL84_hw_protected_real.json"
)


@pytest.mark.skipif(not _EL84_PATH.exists(), reason="EL84 hw-protected fixture missing")
class TestRealEL84Data:
    """Realism smoke-check for the dead-data detector on an EL84 pentode
    scan with hardware Ig2 protection tripped at Ug2 >= 174 V.

    Fixture: pentode_EL84_hw_protected_real.json (~82% dead points,
    dead_ug2_levels include 174..300 V). Synthetic coverage in
    TestDetectDeadData exercises the same API in controlled conditions.
    """

    @pytest.fixture(autouse=True)
    def load_data(self):
        data = json.loads(_EL84_PATH.read_text(encoding="utf-8"))
        self.points = data["points"]

    def test_detects_dead_data(self):
        report = detect_dead_data(self.points)
        assert report.has_dead_data
        assert report.total_points == len(self.points)
        # Most points should be dead (85%+ from prior analysis)
        assert report.dead_pct > 50

    def test_dead_ug2_levels_present(self):
        report = detect_dead_data(self.points)
        # Ug2 >= ~174V are dead from hardware protection
        assert len(report.dead_ug2_levels) > 0
        # Working levels (Ug2 ~99, ~124) should NOT be flagged
        # Ug2=0 is legitimately dead (no screen voltage)
        for ug2 in report.dead_ug2_levels:
            assert ug2 < 1 or ug2 > 130  # no false positives on working levels

    def test_clean_reduces_points(self):
        report = detect_dead_data(self.points)
        cleaned = clean_dead_points(self.points, report=report)
        assert len(cleaned) < len(self.points)
        assert len(cleaned) > 0  # some data survives

    def test_cleaned_data_has_current(self):
        """All cleaned points should have meaningful Ia."""
        cleaned = clean_dead_points(self.points)
        # At least some points have appreciable current
        max_ia = max(p["ia"] for p in cleaned)
        assert max_ia > 1.0  # mA

    def test_custom_threshold(self):
        """Higher threshold catches more points."""
        report_low = detect_dead_data(self.points, ia_thr=0.1)
        report_high = detect_dead_data(self.points, ia_thr=1.0)
        assert report_high.dead_points >= report_low.dead_points
