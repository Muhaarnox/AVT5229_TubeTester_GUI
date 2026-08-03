"""Tests for LivePanel off-nominal heater badge + tooltip.

The badge makes a non-standard heater visible: a stuck-reduced heater (e.g.
80 % after an interrupted emission Uh80 phase) otherwise sits just above the
75 % auto-preheat gate and every subsequent measurement is silently low.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app.live_panel import (
    LivePanel, HEATER_NOMINAL_TOLERANCE_PCT, _PA_OVER, _PA_NORMAL, _PA_WARN,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class TestPaOverloadAtZeroOverPct:
    """pa_over_pct=0 is an EXACT limit, not 'protection off' — the Pa/Pg2 label
    must still turn red when the dissipation exceeds the max."""

    def test_pa_red_when_over_max_at_zero_over_pct(self, qapp):
        p = LivePanel()
        p.set_pa_limits(pa_max=12.0, pa_over_pct=0)
        p._update_pa_label(13.0)
        # Unfixed: the over_pct>0 clause routes to the else branch (_PA_NORMAL).
        assert p.lbl_pa.styleSheet() == _PA_OVER
        assert "12.0" in p.lbl_pa.text()  # the limit is shown

    def test_pa_normal_when_under_max_at_zero_over_pct(self, qapp):
        p = LivePanel()
        p.set_pa_limits(pa_max=12.0, pa_over_pct=0)
        p._update_pa_label(5.0)
        assert p.lbl_pa.styleSheet() == _PA_NORMAL

    def test_pg2_red_when_over_max_at_zero_over_pct(self, qapp):
        p = LivePanel()
        p.set_pg2_limits(pg2_max=2.0, pg2_over_pct=0)
        p._update_pg2_label(3.0)
        assert p.lbl_pg2.styleSheet() == _PA_OVER


class TestPaWarnTierAboveZeroOverPct:
    """The gate change must NOT break the WARN (orange) tolerance band at
    over_pct>0: max < pa_w <= limit is orange, above limit is red."""

    def test_warn_tier_between_max_and_limit(self, qapp):
        p = LivePanel()
        p.set_pa_limits(pa_max=12.0, pa_over_pct=10)   # limit = 13.2 W
        p._update_pa_label(13.0)                        # 12 < 13 <= 13.2 → orange
        assert p.lbl_pa.styleSheet() == _PA_WARN
        p._update_pa_label(14.0)                        # > 13.2 → red
        assert p.lbl_pa.styleSheet() == _PA_OVER
        p._update_pa_label(11.0)                        # < 12 → normal
        assert p.lbl_pa.styleSheet() == _PA_NORMAL


class TestHeaterBadge:
    def test_no_nominal_means_no_badge(self, qapp):
        """Without a nominal reference (0), the heater never gets a badge."""
        p = LivePanel()
        p._update_uh_label(5.0)
        assert "⚠" not in p.lbl_uh.text()
        assert p.lbl_uh.toolTip() == ""

    def test_off_nominal_shows_badge_and_tooltip(self, qapp):
        p = LivePanel()
        p.set_nominal_heater(6.3, 0.0)
        p._update_uh_label(5.0)  # 5.0/6.3 ≈ 79 % → 21 % deviation → off-nominal
        assert "⚠" in p.lbl_uh.text()
        assert "79%" in p.lbl_uh.text()
        tip = p.lbl_uh.toolTip()
        assert tip and "6.3" in tip and "5.0" in tip

    def test_within_tolerance_has_no_badge(self, qapp):
        p = LivePanel()
        p.set_nominal_heater(6.3, 0.0)
        p._update_uh_label(6.2)  # ~1.6 % deviation < tolerance
        assert "⚠" not in p.lbl_uh.text()
        assert p.lbl_uh.toolTip() == ""

    def test_above_nominal_also_flags(self, qapp):
        """'Differs from standard' is symmetric — an over-voltage flags too."""
        p = LivePanel()
        p.set_nominal_heater(6.3, 0.0)
        p._update_uh_label(7.2)  # ~114 % → off-nominal
        assert "⚠" in p.lbl_uh.text()
        assert "114%" in p.lbl_uh.text()

    def test_current_heater_badges_on_ih(self, qapp):
        """Current-heated lamp (uh nominal 0, ih nominal > 0): badge is on Ih,
        and the always-~0 Uh never flags (no reference)."""
        p = LivePanel()
        p.set_nominal_heater(0.0, 0.3)
        p._update_ih_label(0.24)  # 80 % → off-nominal
        assert "⚠" in p.lbl_ih.text()
        assert "80%" in p.lbl_ih.text()
        p._update_uh_label(0.0)
        assert "⚠" not in p.lbl_uh.text()

    def test_badge_clears_when_back_to_nominal(self, qapp):
        p = LivePanel()
        p.set_nominal_heater(6.3, 0.0)
        p._update_uh_label(5.0)
        assert "⚠" in p.lbl_uh.text()
        p._update_uh_label(6.3)  # restored
        assert "⚠" not in p.lbl_uh.text()
        assert p.lbl_uh.toolTip() == ""

    def test_tolerance_constant_is_sane(self):
        # A meaningful, small-but-not-trivial tolerance (catches 80 %, ignores
        # verify jitter). Pin so a careless change is noticed.
        assert 1.0 <= HEATER_NOMINAL_TOLERANCE_PCT <= 15.0

    def test_heater_off_pct_helper(self, qapp):
        p = LivePanel()
        assert p._heater_off_pct(5.0, 0.0) is None    # no nominal
        assert p._heater_off_pct(6.3, 6.3) is None    # exactly nominal
        assert p._heater_off_pct(5.0, 6.3) == 79      # off-nominal → clamped int %

    def test_non_finite_reading_does_not_crash(self, qapp):
        """A NaN / inf heater reading must not raise out of the live-update slot
        (round() would). It degrades to no badge rather than crashing — the
        badge is meant to surface anomalies, not blow up the UI callback."""
        p = LivePanel()
        p.set_nominal_heater(6.3, 0.0)
        for bad in (float("nan"), float("inf"), float("-inf")):
            p._update_uh_label(bad)            # must not raise
            assert "⚠" not in p.lbl_uh.text()
            assert p.lbl_uh.toolTip() == ""
            # also via the point path (where a corrupt JSON point could arrive)
            p.update_from_point({"uh": bad, "ih": 0.0, "ua": 0, "ug1": 0,
                                 "ug2": 0, "ia": 0.0, "ig2": 0.0})

    def test_partial_point_keeps_last_heater_reading(self, qapp):
        """A point without uh/ih must not blank the heater labels.

        Partial points arrive during a health test (OP-ramp steps carry no
        heater channel). Defaulting the missing key to 0.0 renders "Uh: 0 V"
        with an off-nominal badge — indistinguishable from a dead heater.
        """
        p = LivePanel()
        p.set_nominal_heater(6.3, 0.76)
        p.update_from_point({"uh": 6.3, "ih": 0.76, "ua": 250, "ug1": -7.0,
                             "ug2": 250, "ia": 48.0, "ig2": 5.0})
        assert "6.3" in p.lbl_uh.text()
        assert "⚠" not in p.lbl_uh.text()

        p.update_from_point({"ua": 250, "ug1": -6.0, "ug2": 250,
                             "ia": 52.0, "ig2": 5.5})  # no uh/ih
        assert "6.3" in p.lbl_uh.text(), "heater voltage was blanked"
        assert "0.76" in p.lbl_ih.text(), "heater current was blanked"
        assert "⚠" not in p.lbl_uh.text()
        assert "⚠" not in p.lbl_ih.text()

    def test_point_with_heater_still_updates_labels(self, qapp):
        """The optional-key guard must not freeze a genuine heater update."""
        p = LivePanel()
        p.set_nominal_heater(6.3, 0.76)
        p.update_from_point({"uh": 6.3, "ih": 0.76, "ua": 0, "ug1": 0,
                             "ug2": 0, "ia": 0.0, "ig2": 0.0})
        p.update_from_point({"uh": 5.0, "ih": 0.60, "ua": 0, "ug1": 0,
                             "ug2": 0, "ia": 0.0, "ig2": 0.0})
        assert "5.0" in p.lbl_uh.text()
        assert "⚠" in p.lbl_uh.text()   # off-nominal badge follows
        assert "0.6" in p.lbl_ih.text()
        assert "⚠" in p.lbl_ih.text()
        # An explicit zero IS a reading — a real heater-off must show through.
        p.update_from_point({"uh": 0.0, "ih": 0.0, "ua": 0, "ug1": 0,
                             "ug2": 0, "ia": 0.0, "ig2": 0.0})
        assert "0.0" in p.lbl_uh.text()

    def test_negative_reading_shows_zero_percent_not_negative(self, qapp):
        """A glitch negative reading flags as off-nominal but displays '0 %',
        not a confusing negative percentage."""
        p = LivePanel()
        p.set_nominal_heater(6.3, 0.0)
        p._update_uh_label(-0.5)
        assert "⚠" in p.lbl_uh.text()
        assert "0%" in p.lbl_uh.text()
        assert "-" not in p.lbl_uh.text().split("(")[-1]  # no negative in the badge
