"""Tests for health history pure logic (app.health_history module).

These functions are used by HealthHistoryManager and MatchPanel.
"""

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Sibling test modules (shared _FakeClient/_cfg/_lamp harness) are
# imported by name; make the tests directory importable regardless of
# the pytest import mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.health_history import (
    entry_matches_filter,
    build_matching_conditions,
    build_match_active,
    build_match_entry_info,
)
from lm19.tube_matching import MatchResult, MatchGroup, TubeRecord
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)


# ===========================================================================
# Tests
# ===========================================================================


class TestEntryMatchesFilter:
    """Test the pure filter predicate."""

    def _entry(self, lamp_id="L1", name="scan", ug2_mode=TOPOLOGY_PENTODE,
               index=75.0, timestamp="2024-01-01"):
        return {
            "lamp_id": lamp_id,
            "name": name,
            "timestamp": timestamp,
            "conditions": {"ug2_mode": ug2_mode},
            "health": {"index": index},
        }

    def test_no_filters_all_visible(self):
        assert entry_matches_filter(self._entry()) is True

    def test_regex_matches_lamp_id(self):
        assert entry_matches_filter(
            self._entry(lamp_id="EL34-A"),
            regex=re.compile("EL34", re.IGNORECASE),
        ) is True

    def test_regex_matches_name(self):
        assert entry_matches_filter(
            self._entry(name="morning test"),
            regex=re.compile("morning", re.IGNORECASE),
        ) is True

    def test_regex_no_match(self):
        assert entry_matches_filter(
            self._entry(lamp_id="L1", name="scan"),
            regex=re.compile("xyz"),
        ) is False

    def test_mode_filter_match(self):
        assert entry_matches_filter(
            self._entry(ug2_mode=TOPOLOGY_PENTODE),
            mode_filter="pentode",
        ) is True

    def test_mode_filter_no_match(self):
        assert entry_matches_filter(
            self._entry(ug2_mode=TOPOLOGY_TRIODE),
            mode_filter="pentode",
        ) is False

    def test_mode_filter_all_passes(self):
        assert entry_matches_filter(
            self._entry(ug2_mode=TOPOLOGY_TRIODE),
            mode_filter="all",
        ) is True

    def test_verdict_strong(self):
        assert entry_matches_filter(
            self._entry(index=90),
            verdict_filter="Strong",
            verdict_thresholds={"strong": 85, "good": 65, "weak": 40},
        ) is True

    def test_verdict_good(self):
        assert entry_matches_filter(
            self._entry(index=70),
            verdict_filter="Good",
            verdict_thresholds={"strong": 85, "good": 65, "weak": 40},
        ) is True

    def test_verdict_weak(self):
        assert entry_matches_filter(
            self._entry(index=50),
            verdict_filter="Weak",
            verdict_thresholds={"strong": 85, "good": 65, "weak": 40},
        ) is True

    def test_verdict_replace(self):
        assert entry_matches_filter(
            self._entry(index=20),
            verdict_filter="Replace",
            verdict_thresholds={"strong": 85, "good": 65, "weak": 40},
        ) is True

    def test_verdict_mismatch(self):
        assert entry_matches_filter(
            self._entry(index=90),
            verdict_filter="Weak",
            verdict_thresholds={"strong": 85, "good": 65, "weak": 40},
        ) is False

    def test_verdict_none_index(self):
        entry = self._entry()
        entry["health"]["index"] = None
        assert entry_matches_filter(
            entry,
            verdict_filter="Strong",
            verdict_thresholds={"strong": 85, "good": 65, "weak": 40},
        ) is False

    def test_combined_regex_and_mode(self):
        assert entry_matches_filter(
            self._entry(lamp_id="EL34", ug2_mode=TOPOLOGY_PENTODE),
            regex=re.compile("EL34"),
            mode_filter="pentode",
        ) is True

    def test_combined_regex_matches_but_mode_fails(self):
        assert entry_matches_filter(
            self._entry(lamp_id="EL34", ug2_mode=TOPOLOGY_TRIODE),
            regex=re.compile("EL34"),
            mode_filter="pentode",
        ) is False


class TestBuildMatchingConditions:
    def _entry(self, ua=250.0, ug1=-8.0, ug2=250.0, ug2_mode=TOPOLOGY_PENTODE):
        return {
            "conditions": {
                "ua": ua, "ug1": ug1, "ug2": ug2,
                "ug2_mode": ug2_mode,
            }
        }

    def test_basic(self):
        entries = [self._entry(ua=250, ug1=-8, ug2=250)]
        result = build_matching_conditions(entries, "pentode")
        assert result == (250.0, -8.0, 250.0, "pentode", False)

    def test_no_matching_mode(self):
        entries = [self._entry(ug2_mode=TOPOLOGY_TRIODE)]
        result = build_matching_conditions(entries, "pentode")
        assert result is None

    def test_uses_first_entry(self):
        entries = [
            self._entry(ua=200, ug1=-5, ug2=200),
            self._entry(ua=300, ug1=-10, ug2=300),
        ]
        result = build_matching_conditions(entries, "pentode")
        assert result == (200.0, -5.0, 200.0, "pentode", False)

    def test_rounds_values(self):
        entries = [self._entry(ua=250.123, ug1=-8.456, ug2=249.789)]
        result = build_matching_conditions(entries, "pentode")
        assert result == (250.1, -8.5, 249.8, "pentode", False)


class TestBuildMatchEntryInfo:
    def test_empty_result(self):
        result = MatchResult(mode="groups", groups=[], unmatched=[])
        assert build_match_entry_info(result) == {}

    def test_groups_have_number_and_delta(self):
        rec1 = TubeRecord(lamp_id="L1", timestamp="t1", an=1, ia=50, s=8, r=20)
        rec2 = TubeRecord(lamp_id="L2", timestamp="t2", an=1, ia=55, s=8.5, r=21)
        g = MatchGroup(number=1, records=[rec1, rec2], delta=3.5)
        result = MatchResult(mode="groups", groups=[g], unmatched=[])
        info = build_match_entry_info(result)
        assert info[("L1", "t1")] == (1, 3.5)
        assert info[("L2", "t2")] == (1, 3.5)

    def test_unmatched_has_zero_group(self):
        rec = TubeRecord(lamp_id="L3", timestamp="t3", an=1, ia=30, s=5, r=15)
        result = MatchResult(mode="groups", groups=[], unmatched=[rec])
        info = build_match_entry_info(result)
        assert info[("L3", "t3")] == (0, 0.0)

    def test_multiple_groups(self):
        r1 = TubeRecord(lamp_id="L1", timestamp="t1", an=1, ia=50, s=8, r=20)
        r2 = TubeRecord(lamp_id="L2", timestamp="t2", an=1, ia=55, s=8.5, r=21)
        r3 = TubeRecord(lamp_id="L3", timestamp="t3", an=1, ia=60, s=9, r=22)
        g1 = MatchGroup(number=1, records=[r1, r2], delta=2.0)
        g2 = MatchGroup(number=2, records=[r3], delta=0.0)
        result = MatchResult(mode="groups", groups=[g1, g2], unmatched=[])
        info = build_match_entry_info(result)
        assert info[("L1", "t1")][0] == 1
        assert info[("L3", "t3")][0] == 2


class TestBuildMatchActive:
    def test_empty_result(self):
        result = MatchResult(mode="groups", groups=[], unmatched=[])
        assert build_match_active(result) == set()

    def test_with_groups_and_unmatched(self):
        rec1 = TubeRecord(lamp_id="L1", timestamp="t1", an=1,
                          ia=50, s=8.0, r=20.0)
        rec2 = TubeRecord(lamp_id="L2", timestamp="t2", an=1,
                          ia=55, s=8.5, r=21.0)
        rec3 = TubeRecord(lamp_id="L3", timestamp="t3", an=1,
                          ia=30, s=5.0, r=15.0)
        group = MatchGroup(number=1, records=[rec1, rec2], delta=5.0)
        result = MatchResult(mode="groups", groups=[group], unmatched=[rec3])

        active = build_match_active(result)
        assert active == {("L1", "t1"), ("L2", "t2"), ("L3", "t3")}

    def test_twin_anode_rows_coexist(self):
        """Both records of one lamp stay active — the old lamp_id-keyed
        dict evicted one of them (last-wins), dimming a row that DID
        participate in the match."""
        rec1 = TubeRecord(lamp_id="L1", timestamp="t1", an=1,
                          ia=50, s=8.0, r=20.0)
        rec2 = TubeRecord(lamp_id="L1", timestamp="t2", an=2,
                          ia=55, s=8.5, r=21.0)
        result = MatchResult(mode="groups", groups=[], unmatched=[rec1, rec2])
        active = build_match_active(result)
        assert ("L1", "t1") in active and ("L1", "t2") in active

    def test_similar_anchor_included(self):
        anchor = TubeRecord(lamp_id="A", timestamp="ta", an=1,
                            ia=44, s=11.0, r=30.0)
        cand = TubeRecord(lamp_id="B", timestamp="tb", an=1,
                          ia=43, s=11.0, r=30.0)
        result = MatchResult(
            mode="similar",
            groups=[MatchGroup(number=1, records=[cand], delta=1.0)],
            unmatched=[], anchor=anchor)
        assert ("A", "ta") in build_match_active(result)


# ===========================================================================
# Plan box construction smoke tests
# ===========================================================================
#
# Construct a real ``HealthTab`` against a stub ``AppContext`` and verify
# the plan-box widgets exist with the right ranges/types. These guard
# the ``self.plan_*`` attribute API that ~100 call sites in
# ``health_tab.py`` read.

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _make_health_tab():
    """Construct a HealthTab with stubbed AppContext for widget inspection."""
    from PySide6.QtWidgets import (
        QApplication, QButtonGroup, QDoubleSpinBox, QSpinBox, QPushButton,
        QLabel, QRadioButton,
    )
    QApplication.instance() or QApplication([])

    from app.app_context import AppContext
    from app.health_tab import HealthTab
    from lm19.app_config import AppConfig
    from lm19.calibration import CalibrationData

    cfg = AppConfig()
    cal = CalibrationData()
    ctx = AppContext(
        get_client=lambda: None,
        get_write_locked=lambda: False,
        get_app_config=lambda: cfg,
        get_calibration=lambda: cal,
        get_lamps=lambda: [],
        get_current_tube_type=lambda: "",
        get_current_lamp_id=lambda: "",
        set_poller_active=lambda _b: None,
    )
    tab = HealthTab(ctx)
    return tab


class TestHealthTabPlanBoxConstruction:
    """Smoke tests for ``_build_plan_box`` widget contract.

    These guard the plan_widget API (``self.plan_*`` attributes) used by
    ``_collect_measurement_plan``, ``_validate_plan``, and ~100 other
    call sites in health_tab.py.
    """

    def test_plan_target_spinboxes_exist(self):
        """plan_ua_target / plan_ug1_target / plan_ug2_target are spinboxes."""
        from PySide6.QtWidgets import QDoubleSpinBox
        tab = _make_health_tab()
        assert isinstance(tab.plan_ua_target, QDoubleSpinBox)
        assert isinstance(tab.plan_ug1_target, QDoubleSpinBox)
        assert isinstance(tab.plan_ug2_target, QDoubleSpinBox)

    def test_plan_target_ranges(self):
        """Target ranges match physical limits (Ua/Ug2 positive, Ug1 negative)."""
        tab = _make_health_tab()
        assert tab.plan_ua_target.minimum() == 0.0
        assert tab.plan_ua_target.maximum() == 1000.0
        assert tab.plan_ug1_target.minimum() == -100.0
        assert tab.plan_ug1_target.maximum() == 0.0
        assert tab.plan_ug2_target.minimum() == 0.0
        assert tab.plan_ug2_target.maximum() == 1000.0

    def test_plan_delta_spinboxes_exist(self):
        from PySide6.QtWidgets import QDoubleSpinBox
        tab = _make_health_tab()
        assert isinstance(tab.plan_delta_ua, QDoubleSpinBox)
        assert isinstance(tab.plan_delta_ug1, QDoubleSpinBox)
        assert isinstance(tab.plan_delta_ug2, QDoubleSpinBox)

    def test_plan_emission_ratio_range(self):
        tab = _make_health_tab()
        assert tab.plan_emission_ratio.minimum() == pytest.approx(0.1)
        assert tab.plan_emission_ratio.maximum() == pytest.approx(1.0)

    def test_plan_points_spinbox(self):
        from PySide6.QtWidgets import QSpinBox
        tab = _make_health_tab()
        assert isinstance(tab.plan_points, QSpinBox)
        assert tab.plan_points.minimum() == 5
        assert tab.plan_points.maximum() == 21
        assert tab.plan_points.singleStep() == 2
        assert tab.plan_points.value() == 5

    def test_plan_repeats_spinbox(self):
        from PySide6.QtWidgets import QSpinBox
        tab = _make_health_tab()
        assert isinstance(tab.plan_repeats, QSpinBox)
        assert tab.plan_repeats.minimum() == 1
        assert tab.plan_repeats.maximum() == 50
        assert tab.plan_repeats.value() == 5

    def test_plan_reset_button_exists(self):
        from PySide6.QtWidgets import QPushButton
        tab = _make_health_tab()
        assert isinstance(tab.plan_reset_btn, QPushButton)

    def test_plan_validation_label_exists(self):
        from PySide6.QtWidgets import QLabel
        tab = _make_health_tab()
        assert isinstance(tab.plan_validation_label, QLabel)
        assert tab.plan_validation_label.wordWrap()

    def test_ug2_mode_group_setup(self):
        """ug2 independent/track radios are mutually exclusive (button group)."""
        from PySide6.QtWidgets import QButtonGroup, QRadioButton
        tab = _make_health_tab()
        assert isinstance(tab.ug2_mode_group, QButtonGroup)
        assert isinstance(tab.ug2_independent_radio, QRadioButton)
        assert isinstance(tab.ug2_track_radio, QRadioButton)
        # Independent is default
        assert tab.ug2_independent_radio.isChecked()
        assert not tab.ug2_track_radio.isChecked()

    def test_ug2_offset_disabled_in_independent_mode(self):
        """ug2_offset is disabled until ug2_track radio is selected."""
        from PySide6.QtWidgets import QDoubleSpinBox
        tab = _make_health_tab()
        assert isinstance(tab.ug2_offset, QDoubleSpinBox)
        assert tab.ug2_offset.minimum() == -100.0
        assert tab.ug2_offset.maximum() == 100.0
        assert tab.ug2_offset.value() == 0.0
        # Disabled because ug2_independent_radio is checked
        assert not tab.ug2_offset.isEnabled()

    def test_ug2_offset_enables_when_track_selected(self):
        """Switching to ug2_track_radio enables ug2_offset."""
        tab = _make_health_tab()
        tab.ug2_track_radio.setChecked(True)
        assert tab.ug2_offset.isEnabled()


class TestValidatePlanUg1ZeroCrossing:
    """_validate_plan must reject an S-sweep whose top crosses 0 V (positive
    grid). The S-measurement sweeps Ug1 ± δUg1 around the bias."""

    def test_ug1_delta_crosses_zero_rejected(self):
        from i18n_setup import t
        tab = _make_health_tab()
        tab.plan_ua_target.setValue(200.0)
        tab.plan_delta_ua.setValue(25.0)
        tab.plan_ug1_target.setValue(-3.0)
        tab.plan_delta_ug1.setValue(4.0)   # -3 + 4 = +1 > 0 → crosses zero
        assert tab._validate_plan() is False
        assert t("health.Plan_err_ug1_delta") in tab.plan_validation_label.text()

    def test_ug1_delta_within_bounds_ok(self):
        from i18n_setup import t
        tab = _make_health_tab()
        tab.plan_ua_target.setValue(200.0)
        tab.plan_delta_ua.setValue(25.0)
        tab.plan_ug1_target.setValue(-8.0)
        tab.plan_delta_ug1.setValue(2.0)   # -8 + 2 = -6 < 0 → ok
        # No errors at all for a fully valid plan, and specifically no Ug1 error.
        assert tab._validate_plan() is True
        assert t("health.Plan_err_ug1_delta") not in tab.plan_validation_label.text()

    def test_ug1_delta_exactly_zero_endpoint_rejected(self):
        """Sweep top exactly 0 V (grid-conduction onset) is rejected (<= boundary)."""
        from i18n_setup import t
        tab = _make_health_tab()
        tab.plan_ua_target.setValue(200.0)
        tab.plan_delta_ua.setValue(25.0)
        tab.plan_ug1_target.setValue(-2.0)
        tab.plan_delta_ug1.setValue(2.0)   # -2 + 2 = 0 → endpoint at 0 V
        assert tab._validate_plan() is False
        assert t("health.Plan_err_ug1_delta") in tab.plan_validation_label.text()


class TestFindSimilarAnchor:
    """'Find similar' (latest) vs 'Find similar (this measurement)' (specific)."""

    def test_this_measurement_sets_anchor_timestamp(self):
        tab = _make_health_tab()
        captured = []
        tab._run_matching = lambda cfg: captured.append(cfg)  # spy, skip real run
        entry = {"lamp_id": "L1", "timestamp": "2026-01-01T10:00:00",
                 "conditions": {"ug2_mode": TOPOLOGY_PENTODE}}
        tab._start_find_similar(entry, this_measurement=True)
        assert tab._match_anchor_lamp_id == "L1"
        assert tab._match_anchor_timestamp == "2026-01-01T10:00:00"
        assert captured  # matching was triggered

    def test_default_find_similar_clears_anchor_timestamp(self):
        tab = _make_health_tab()
        tab._run_matching = lambda cfg: None
        entry = {"lamp_id": "L2", "timestamp": "2026-02-02T00:00:00",
                 "conditions": {"ug2_mode": TOPOLOGY_PENTODE}}
        # First pin a specific measurement, then plain Find similar must clear it.
        tab._start_find_similar(entry, this_measurement=True)
        assert tab._match_anchor_timestamp == "2026-02-02T00:00:00"
        tab._start_find_similar(entry)  # default = latest/best
        assert tab._match_anchor_timestamp is None


class TestMatchAnchorUi:
    """Anchor plumbing between the history table and match_tubes."""

    def _entry(self, lid, ts, an=1, servo=False, ia=44.0, s=11.0, r=30.0,
               index=90.0):
        e = {
            "lamp_id": lid, "timestamp": ts, "name": "",
            "conditions": {"ua": 250.0, "ug1": -7.3, "ug2": 250.0,
                           "an": an, "ug2_mode": TOPOLOGY_PENTODE},
            "srk": {"s": s, "r": r, "ia_op": ia},
            "health": {"raw": {"ia_op": ia}, "index": index},
        }
        if servo:
            e["conditions"]["bias_servo"] = True
            e["health"]["raw"]["ia_plan_ma"] = ia - 5.0
            e["health"]["metrics"] = {"bias_shift_v": -0.5}
        return e

    def _spy_tab(self, monkeypatch, entries, result=None):
        from lm19.tube_matching import MatchResult
        import app.health_tab as HT
        tab = _make_health_tab()
        seen = {}

        def spy(spy_entries, **kw):
            seen.update(kw)
            return result if result is not None else MatchResult(
                mode="similar", groups=[], unmatched=[])

        monkeypatch.setattr(HT, "match_tubes", spy)
        tab._history_entries = list(entries)
        return tab, seen

    def test_start_find_similar_forwards_anchor_kwargs(self, monkeypatch):
        # Call-site spy for the WHOLE anchor triple: a unit pin on
        # match_tubes cannot prove the caller forwards it (and anchor_an
        # was a dead parameter before — no caller ever passed it).
        e = self._entry("L1", "2026-01-01T10:00:00", an=2)
        tab, seen = self._spy_tab(monkeypatch, [e])
        tab._start_find_similar(e, this_measurement=True)
        assert seen.get("anchor_lamp_id") == "L1"
        assert seen.get("anchor_timestamp") == "2026-01-01T10:00:00"
        assert seen.get("anchor_an") == 2
        assert seen.get("mode") == "similar"

    def test_default_find_similar_forwards_an_without_timestamp(
            self, monkeypatch):
        e = self._entry("L1", "2026-01-01T10:00:00", an=2)
        tab, seen = self._spy_tab(monkeypatch, [e])
        tab._start_find_similar(e)
        assert seen.get("anchor_an") == 2
        assert seen.get("anchor_timestamp") is None

    def test_anchor_error_reaches_the_summary(self, monkeypatch):
        from i18n_setup import t
        from lm19.tube_matching import ANCHOR_ERR_NOT_FOUND, MatchResult
        e = self._entry("L1", "2026-01-01T10:00:00")
        res = MatchResult(mode="similar", groups=[], unmatched=[],
                          anchor_error=ANCHOR_ERR_NOT_FOUND)
        tab, _seen = self._spy_tab(monkeypatch, [e], result=res)
        tab._start_find_similar(e)
        assert (tab.match_panel.summary_info.text()
                == t("health.match_Anchor_missing"))

    def test_anchor_error_registry_locale_bijection(self):
        import json
        from pathlib import Path
        from i18n_setup import available_locales, _LOCALES_DIR
        from app.health_tab import ANCHOR_ERROR_KEYS
        from lm19.tube_matching import MATCH_ANCHOR_ERRORS
        assert set(ANCHOR_ERROR_KEYS) == set(MATCH_ANCHOR_ERRORS)
        needed = [k.split(".", 1) for k in ANCHOR_ERROR_KEYS.values()]
        needed += [("health", "match_Cond_servo"),
                   ("health", "match_Servo_pool_hint")]
        for loc in available_locales():
            data = json.loads(
                (Path(_LOCALES_DIR) / f"{loc}.json").read_text("utf-8"))
            for section, key in needed:
                assert key in data.get(section, {}), (
                    f"{loc}.json missing {section}.{key}")

    def test_anchor_row_marked_and_kept_active(self, monkeypatch):
        from PySide6.QtCore import Qt
        from lm19.tube_matching import MatchGroup, MatchResult, TubeRecord
        import app.health_tab as HT
        from app.health_tab import _COL_GRP
        ea = self._entry("A", "2026-01-02T00:00:00")
        eb = self._entry("B", "2026-01-01T00:00:00")
        monkeypatch.setattr(HT, "list_health_entries",
                            lambda tube_type: [ea, eb])
        tab = _make_health_tab()
        tab.tube_combo.addItem("EL84")
        tab.tube_combo.setCurrentText("EL84")
        tab.reload_history()

        rec_a = TubeRecord(lamp_id="A", timestamp="2026-01-02T00:00:00",
                           an=1, ia=44.0, s=11.0, r=30.0)
        rec_b = TubeRecord(lamp_id="B", timestamp="2026-01-01T00:00:00",
                           an=1, ia=43.0, s=11.0, r=30.0)
        res = MatchResult(
            mode="similar",
            groups=[MatchGroup(number=1, records=[rec_b], delta=1.0)],
            unmatched=[], anchor=rec_a)
        monkeypatch.setattr(HT, "match_tubes", lambda *a, **k: res)
        tab._start_find_similar(ea)

        # The anchor row is an active pool member (undimmed, countable) …
        assert ("A", "2026-01-02T00:00:00") in tab._match_active
        # … and its Grp cell carries the reference marker.
        grp_texts = {}
        for row in range(tab.table.rowCount()):
            entry = tab.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            cell = tab.table.item(row, _COL_GRP)
            grp_texts[entry["lamp_id"]] = cell.text() if cell else ""
        assert grp_texts["A"] == "★"
        assert grp_texts["B"].startswith("#1")

    def test_pool_total_counts_protocol_compatible_only(self, monkeypatch):
        from i18n_setup import t
        from lm19.tube_matching import (
            MatchResult, TubeRecord, _conditions_key)
        servo1 = self._entry("S1", "2026-01-03T00:00:00", servo=True)
        servo2 = self._entry("S2", "2026-01-02T00:00:00", servo=True)
        fixed = self._entry("F1", "2026-01-01T00:00:00")
        rec = TubeRecord(lamp_id="S1", timestamp="2026-01-03T00:00:00",
                         an=1, ia=44.0, s=11.0, r=30.0)
        res = MatchResult(mode="similar", groups=[], unmatched=[],
                          anchor=rec,
                          conditions_used=_conditions_key(servo1))
        tab, _seen = self._spy_tab(monkeypatch, [servo1, servo2, fixed],
                                   result=res)
        tab._start_find_similar(servo1)
        # Under strict the fixed entry is not in the servo pool: the label
        # must say 2, not 3 — it must not promise lamps the pool never
        # admitted.
        assert (tab.match_panel.info_label.text()
                == t("health.match_Lamps_found", count=1, total=2))
        # Call-site pin for the conditions label: the servo flag of the
        # ACTUAL pool must reach set_conditions (a panel-level unit test
        # cannot prove the caller forwards it).
        assert (t("health.match_Cond_servo")
                in tab.match_panel.conditions_label.text())


class TestMatchPanelConditionsLabel:
    """The conditions label must expose the servo flag of the pool and
    hint at the servo-aware protocols under strict."""

    def test_servo_marker_and_strict_hint(self):
        from i18n_setup import t
        from lm19.tube_matching import (
            MATCHING_PROTOCOL_SHARED, MATCHING_PROTOCOL_STRICT)
        tab = _make_health_tab()
        panel = tab.match_panel
        panel.set_protocol(MATCHING_PROTOCOL_STRICT)
        panel.set_conditions(250.0, -7.3, 250.0, "pentode", servo=True)
        text = panel.conditions_label.text()
        assert t("health.match_Cond_servo") in text
        assert t("health.match_Servo_pool_hint") in text
        # shared protocol IS servo-aware — marker stays, hint goes.
        panel.set_protocol(MATCHING_PROTOCOL_SHARED)
        panel.set_conditions(250.0, -7.3, 250.0, "pentode", servo=True)
        text = panel.conditions_label.text()
        assert t("health.match_Cond_servo") in text
        assert t("health.match_Servo_pool_hint") not in text

    def test_fixed_pool_shows_no_servo_marker(self):
        from i18n_setup import t
        tab = _make_health_tab()
        panel = tab.match_panel
        panel.set_conditions(250.0, -7.3, 250.0, "pentode", servo=False)
        assert t("health.match_Cond_servo") not in panel.conditions_label.text()


class TestMeasurementPicks:
    """B3: a Sel-column pick is an INPUT of the match — entries are
    pre-filtered so the core's latest/best selector picks the pinned
    measurement, and the pick survives recalculation (the original bug:
    the click was overwritten by the very recalculation it triggered)."""

    def _entry(self, lid, ts, an=1, ia=44.0, ua=250.0, s=11.0,
               ug2_mode=TOPOLOGY_PENTODE):
        return {
            "lamp_id": lid, "timestamp": ts, "name": "",
            "conditions": {"ua": ua, "ug1": -7.3, "ug2": 250.0,
                           "an": an, "ug2_mode": ug2_mode},
            "srk": {"s": s, "r": 30.0, "ia_op": ia},
            "health": {"raw": {"ia_op": ia}, "index": 90.0},
        }

    def _l1_ia(self, result):
        for g in result.groups:
            for r in g.records:
                if r.lamp_id == "L1":
                    return r.ia
        for r in result.unmatched:
            if r.lamp_id == "L1":
                return r.ia
        return None

    def _tab(self, entries):
        tab = _make_health_tab()
        tab._history_entries = list(entries)
        return tab

    def test_pick_reaches_the_match_and_survives_recalc(self):
        # Distinct ia values (48 vs 40) make the outcome observable.
        entries = [
            self._entry("L1", "2026-01-02T00:00:00", ia=48.0),
            self._entry("L1", "2026-01-01T00:00:00", ia=40.0),
            self._entry("L2", "2026-01-01T12:00:00", ia=41.0),
        ]
        tab = self._tab(entries)
        cfg = tab.match_panel.get_config()
        cfg["source"] = "all"
        # Negative control: without a pick, latest wins.
        tab._run_matching(cfg)
        assert self._l1_ia(tab._match_result) == 48.0
        # Pin the OLDER measurement; two consecutive runs must both
        # honour it — the second run is the survival pin.
        tab._match_pick[("L1", 1)] = "2026-01-01T00:00:00"
        tab._run_matching(cfg)
        assert self._l1_ia(tab._match_result) == 40.0
        tab._run_matching(cfg)
        assert tab._match_pick == {("L1", 1): "2026-01-01T00:00:00"}
        assert self._l1_ia(tab._match_result) == 40.0

    def test_sel_click_toggles_the_pick(self, monkeypatch):
        from PySide6.QtCore import Qt
        import app.health_tab as HT
        from app.health_tab import _COL_SEL
        entries = [
            self._entry("L1", "2026-01-02T00:00:00", ia=48.0),
            self._entry("L1", "2026-01-01T00:00:00", ia=40.0),
            self._entry("L2", "2026-01-01T12:00:00", ia=41.0),
        ]
        monkeypatch.setattr(HT, "list_health_entries",
                            lambda tube_type: list(entries))
        tab = _make_health_tab()
        tab.tube_combo.addItem("EL84")
        tab.tube_combo.setCurrentText("EL84")
        tab.reload_history()
        cfg = tab.match_panel.get_config()
        cfg["source"] = "all"
        tab._run_matching(cfg)

        def _row_of(ts):
            for row in range(tab.table.rowCount()):
                e = tab.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
                if isinstance(e, dict) and e.get("timestamp") == ts:
                    return row
            raise AssertionError(f"row {ts} not found")

        # First click pins the older measurement and re-runs the match…
        tab._on_table_cell_clicked(_row_of("2026-01-01T00:00:00"), _COL_SEL)
        assert tab._match_pick == {("L1", 1): "2026-01-01T00:00:00"}
        assert self._l1_ia(tab._match_result) == 40.0
        # …second click on the same row unpins (rows re-sort per run,
        # so the row is located again by identity).
        tab._on_table_cell_clicked(_row_of("2026-01-01T00:00:00"), _COL_SEL)
        assert tab._match_pick == {}
        assert self._l1_ia(tab._match_result) == 48.0

    def test_pick_scoped_to_its_conditions_pool(self):
        # The pick must drop only same-pool siblings: the lamp's entry at
        # ANOTHER operating point stays (a pick acts within its pool).
        entries = [
            self._entry("L1", "2026-01-03T00:00:00", ia=48.0),           # pool A sibling
            self._entry("L1", "2026-01-01T00:00:00", ia=40.0),           # picked (pool A)
            self._entry("L1", "2026-01-02T00:00:00", ia=60.0, ua=300.0),  # pool B
        ]
        tab = self._tab(entries)
        tab._match_pick[("L1", 1)] = "2026-01-01T00:00:00"
        out = tab._apply_match_picks(list(entries))
        stamps = {e["timestamp"] for e in out}
        assert "2026-01-01T00:00:00" in stamps      # the pick itself
        assert "2026-01-02T00:00:00" in stamps      # other pool untouched
        assert "2026-01-03T00:00:00" not in stamps  # same-pool sibling dropped

    def test_stale_pick_ignored_with_warning(self, caplog):
        import logging
        entries = [self._entry("L1", "2026-01-02T00:00:00", ia=48.0)]
        tab = self._tab(entries)
        tab._match_pick[("L1", 1)] = "2099-01-01T00:00:00"
        with caplog.at_level(logging.WARNING):
            out = tab._apply_match_picks(list(entries))
        assert out == entries
        assert any("pick" in r.message.lower() for r in caplog.records)

    def test_this_measurement_anchor_wins_over_pick(self):
        # An explicit similar-mode anchor names an exact entry: it must
        # not be dropped by a pick on the same lamp.
        e_a = self._entry("L1", "2026-01-01T00:00:00", ia=40.0)
        e_b = self._entry("L1", "2026-01-02T00:00:00", ia=48.0)
        e_c = self._entry("L2", "2026-01-01T12:00:00", ia=41.0)
        tab = self._tab([e_b, e_a, e_c])
        tab._match_pick[("L1", 1)] = "2026-01-01T00:00:00"   # pick A
        tab._start_find_similar(e_b, this_measurement=True)   # anchor B
        assert tab._match_result.anchor is not None
        assert tab._match_result.anchor.timestamp == "2026-01-02T00:00:00"

    def test_twin_anode_anchor_and_candidate_both_active(self):
        # The collision fix at tab level: anchor L1-An1 and its ranked
        # twin candidate L1-An2 are BOTH active rows (the lamp_id-keyed
        # dict used to evict one of them).
        e1 = self._entry("L1", "2026-01-04T00:00:00", an=1, ia=44.0)
        e2 = self._entry("L1", "2026-01-03T00:00:00", an=2, ia=43.5)
        e3 = self._entry("L2", "2026-01-02T00:00:00", an=1, ia=43.0)
        tab = self._tab([e1, e2, e3])
        tab._start_find_similar(e1)
        assert ("L1", "2026-01-04T00:00:00") in tab._match_active
        assert ("L1", "2026-01-03T00:00:00") in tab._match_active
        assert ("L2", "2026-01-02T00:00:00") in tab._match_active

    def test_lamp_count_deduplicates_twin_anodes(self, monkeypatch):
        from i18n_setup import t
        e1 = self._entry("L1", "2026-01-04T00:00:00", an=1, ia=44.0)
        e2 = self._entry("L1", "2026-01-03T00:00:00", an=2, ia=43.5)
        e3 = self._entry("L2", "2026-01-02T00:00:00", an=1, ia=43.0)
        tab = self._tab([e1, e2, e3])
        cfg = tab.match_panel.get_config()
        cfg["source"] = "all"
        tab._run_matching(cfg)
        # 3 active rows, 2 lamps — the label counts lamps.
        assert (tab.match_panel.info_label.text()
                == t("health.match_Lamps_found", count=2, total=3))

    def test_pool_total_follows_anchor_mode_not_panel(self):
        from i18n_setup import t
        from lm19.constants import TOPOLOGY_TRIODE_CONNECTED
        # A specific anchor may sit in ANOTHER ug2_mode than the panel's
        # tube-mode combo: the "of M" total must count the anchor's pool
        # (2 TC entries), not "0 of the panel's mode".
        tc_a = self._entry("L1", "2026-01-04T00:00:00", ia=44.0,
                           ug2_mode=TOPOLOGY_TRIODE_CONNECTED)
        tc_b = self._entry("L2", "2026-01-03T00:00:00", ia=43.0,
                           ug2_mode=TOPOLOGY_TRIODE_CONNECTED)
        p_c = self._entry("L3", "2026-01-02T00:00:00", ia=42.0)
        p_d = self._entry("L4", "2026-01-01T00:00:00", ia=41.0)
        tab = self._tab([tc_a, tc_b, p_c, p_d])
        tab._match_anchor_lamp_id = "L1"
        tab._match_anchor_timestamp = "2026-01-04T00:00:00"
        tab._match_anchor_an = 1
        cfg = tab.match_panel.get_config()
        cfg.update({"source": "all", "mode": "similar",
                    "tube_mode": TOPOLOGY_PENTODE})
        tab._run_matching(cfg)
        assert (tab.match_panel.info_label.text()
                == t("health.match_Lamps_found", count=2, total=2))


class TestHideInactiveFilter:
    """entry_matches_filter hide_inactive branch — previously untested;
    membership is by ROW identity (lamp, timestamp)."""

    def _entry(self, lid="L1", ts="t1"):
        return {"lamp_id": lid, "timestamp": ts, "name": "",
                "conditions": {}, "health": {}}

    def test_active_row_visible(self):
        assert entry_matches_filter(
            self._entry("L1", "t1"), match_active={("L1", "t1")},
            hide_inactive=True) is True

    def test_same_lamp_other_measurement_hidden(self):
        # Discriminates a lamp-only membership check: the lamp IS active
        # via another measurement, but THIS row is not.
        assert entry_matches_filter(
            self._entry("L1", "t2"), match_active={("L1", "t1")},
            hide_inactive=True) is False

    def test_flag_off_shows_everything(self):
        assert entry_matches_filter(
            self._entry("L1", "t2"), match_active={("L1", "t1")},
            hide_inactive=False) is True

    def test_empty_active_set_hides_nothing(self):
        assert entry_matches_filter(
            self._entry("L1", "t2"), match_active=set(),
            hide_inactive=True) is True


class TestMatchPanelAlgorithm:
    """Pair-matching algorithm dropdown — UI + config wiring."""

    def test_algorithm_combo_exists_with_two_options(self):
        tab = _make_health_tab()
        assert hasattr(tab.match_panel, "algorithm_combo")
        # 2 algorithms: greedy + optimal
        assert tab.match_panel.algorithm_combo.count() == 2

    def test_default_algorithm_is_greedy(self):
        tab = _make_health_tab()
        # AppConfig default is "greedy"; MatchPanel.set_algorithm picked it up
        assert tab.match_panel.algorithm_combo.currentData() == "greedy"

    def test_get_config_includes_algorithm(self):
        tab = _make_health_tab()
        cfg = tab.match_panel.get_config()
        assert cfg.get("algorithm") == "greedy"

    def test_set_algorithm_changes_selection(self):
        tab = _make_health_tab()
        tab.match_panel.set_algorithm("optimal")
        assert tab.match_panel.algorithm_combo.currentData() == "optimal"
        # And the new value appears in get_config
        assert tab.match_panel.get_config().get("algorithm") == "optimal"

    def test_set_algorithm_ignores_unknown(self):
        """Unknown algorithm string from corrupt config → keep current selection."""
        tab = _make_health_tab()
        initial = tab.match_panel.algorithm_combo.currentData()
        tab.match_panel.set_algorithm("xyz_nonsense")
        # Selection unchanged — defensive against bad config values
        assert tab.match_panel.algorithm_combo.currentData() == initial

    def test_appconfig_default_matches_match_panel_default(self):
        """AppConfig().health_matching_algorithm must be a valid MatchPanel option.

        Catches regressions where the global default drifts out of sync
        with the UI dropdown options.
        """
        from lm19.app_config import AppConfig
        tab = _make_health_tab()
        default = AppConfig().health_matching_algorithm
        idx = tab.match_panel.algorithm_combo.findData(default)
        assert idx >= 0, (
            f"AppConfig default {default!r} is not a MatchPanel option"
        )


class TestMatchPanelProtocol:
    """Matching-protocol dropdown — UI + config wiring + call-site."""

    def test_combo_covers_the_registry(self):
        # Completeness from the source of truth: every protocol in the
        # registry is a combo item and nothing else is.
        from lm19.tube_matching import MATCHING_PROTOCOLS
        tab = _make_health_tab()
        combo = tab.match_panel.protocol_combo
        items = {combo.itemData(i) for i in range(combo.count())}
        assert items == set(MATCHING_PROTOCOLS)
        assert combo.count() == len(MATCHING_PROTOCOLS)

    def test_registry_labels_exist_in_every_locale(self):
        # Locale pins read locales/*.json directly — translator_for would
        # fall back to en and hide a key missing from one locale.
        import json
        from pathlib import Path
        from i18n_setup import available_locales, _LOCALES_DIR
        from app.match_panel import PROTOCOL_ITEMS
        from lm19.tube_matching import MATCHING_PROTOCOLS
        assert tuple(code for code, _k in PROTOCOL_ITEMS) == MATCHING_PROTOCOLS
        for loc in available_locales():
            data = json.loads(
                (Path(_LOCALES_DIR) / f"{loc}.json").read_text("utf-8"))
            for _code, label_key in PROTOCOL_ITEMS:
                section, key = label_key.split(".", 1)
                assert key in data.get(section, {}), (
                    f"{loc}.json missing {label_key}")
            assert "%{ma}" in data["health"]["match_dIq"], loc

    def test_default_protocol_is_strict_from_config(self):
        from lm19.tube_matching import MATCHING_PROTOCOL_STRICT
        from lm19.app_config import AppConfig
        tab = _make_health_tab()
        assert (tab.match_panel.protocol_combo.currentData()
                == MATCHING_PROTOCOL_STRICT)
        default = AppConfig().health_matching_protocol
        assert tab.match_panel.protocol_combo.findData(default) >= 0

    def test_get_config_includes_protocol(self):
        from lm19.tube_matching import MATCHING_PROTOCOL_SHARED
        tab = _make_health_tab()
        tab.match_panel.set_protocol(MATCHING_PROTOCOL_SHARED)
        assert (tab.match_panel.get_config().get("protocol")
                == MATCHING_PROTOCOL_SHARED)

    def test_set_protocol_ignores_unknown(self):
        tab = _make_health_tab()
        initial = tab.match_panel.protocol_combo.currentData()
        tab.match_panel.set_protocol("xyz_nonsense")
        assert tab.match_panel.protocol_combo.currentData() == initial

    def test_individual_greys_out_ia_weight(self):
        from i18n_setup import t
        from lm19.tube_matching import (
            MATCHING_PROTOCOL_INDIVIDUAL, MATCHING_PROTOCOL_STRICT)
        tab = _make_health_tab()
        assert tab.match_panel.weight_ia_spin.isEnabled()
        tab.match_panel.set_protocol(MATCHING_PROTOCOL_INDIVIDUAL)
        assert not tab.match_panel.weight_ia_spin.isEnabled()
        # The tooltip must explain WHY the knob is dead...
        assert (tab.match_panel.weight_ia_spin.toolTip()
                == t("tip.Health_match_weight_ia_ignored"))
        tab.match_panel.set_protocol(MATCHING_PROTOCOL_STRICT)
        assert tab.match_panel.weight_ia_spin.isEnabled()
        # ...and swap back when the knob comes alive again.
        assert (tab.match_panel.weight_ia_spin.toolTip()
                == t("tip.Health_match_weight_ia"))

    def test_run_matching_passes_protocol_and_gates(self, monkeypatch):
        # Call-site spy: the panel's protocol and BOTH config thresholds
        # must reach match_tubes (a unit pin on match_tubes alone cannot
        # prove the caller forwards them).
        from lm19.tube_matching import (
            MATCHING_PROTOCOL_SHARED, MatchResult)
        import app.health_tab as HT
        tab = _make_health_tab()
        seen = {}

        def spy(entries, **kw):
            seen.update(kw)
            return MatchResult(mode="groups", groups=[], unmatched=[])

        monkeypatch.setattr(HT, "match_tubes", spy)
        tab._history_entries = [
            {"lamp_id": "L1", "timestamp": "2026-01-01T00:00:00",
             "conditions": {"ua": 250.0, "ug1": -7.3, "ug2": 250.0,
                            "an": 1, "ug2_mode": TOPOLOGY_PENTODE},
             "srk": {"s": 11.0, "r": 30.0, "ia_op": 46.0},
             "health": {"raw": {"ia_op": 46.0}, "index": 90.0}},
        ]
        cfg = tab.match_panel.get_config()
        cfg["protocol"] = MATCHING_PROTOCOL_SHARED
        cfg["source"] = "all"
        tab._run_matching(cfg)
        assert seen.get("protocol") == MATCHING_PROTOCOL_SHARED
        app_cfg = tab.get_app_config()
        assert (seen.get("max_iq_imbalance_pct")
                == app_cfg.health_matching_max_iq_imbalance_pct)
        assert (seen.get("bias_adjust_range_pct")
                == app_cfg.health_matching_bias_adjust_range_pct)

    def test_group_header_shows_iq_imbalance(self):
        # shared_bias groups carry δIq — the summary header must print it.
        from lm19.tube_matching import MatchGroup, MatchResult, TubeRecord
        tab = _make_health_tab()
        rec = TubeRecord(lamp_id="A", timestamp="", an=1,
                         ia=44.0, s=11.0, r=30.0)
        rec2 = TubeRecord(lamp_id="B", timestamp="", an=1,
                          ia=46.0, s=11.0, r=30.0)
        result = MatchResult(
            mode="groups",
            groups=[MatchGroup(number=1, records=[rec, rec2], delta=1.0,
                               iq_imbalance_ma=2.0)],
            unmatched=[])
        tab.match_panel.set_result(result)
        layout = tab.match_panel._groups_layout
        texts = []
        for i in range(layout.count()):
            w = layout.itemAt(i).widget()
            if w is not None:
                texts.append(w.text())
        joined = "\n".join(texts)
        assert "2.0" in joined and "δIq" in joined


class TestStepsLiveView:
    """Live/history mode of the measurement-points table.

    Invariants: live points render under their OWN step tags (per-index
    writes into planned rows would make servo probes vanish from the
    live table), and clicking a history entry mid-test must never
    hijack the table for good — the Live button returns the buffered
    process view."""

    def _live_pt(self, step, ia=40.0, ug1=-7.3):
        return {"event": "live_point",
                "point": {"ua": 250.0, "ug1": ug1, "ug2": 250.0,
                          "uh": 6.3, "ih": 0.3, "ia": ia, "ig2": 4.0,
                          "step": step}}

    def _running_tab(self):
        tab = _make_health_tab()
        tab._test_active = True
        tab._steps_view_live = True
        tab._live_points = []
        tab._planned_steps = [
            {"step": "OP", "ua": 250.0, "ug1": -7.3, "ug2": 250.0},
            {"step": "S-1", "ua": 250.0, "ug1": -8.1, "ug2": 250.0},
        ]
        return tab

    def _select_history_entry(self, tab):
        from PySide6.QtWidgets import QTableWidgetItem
        from PySide6.QtCore import Qt
        item = QTableWidgetItem("x")
        item.setData(Qt.ItemDataRole.UserRole, {
            "measurement_points": [
                {"ua": 200.0, "ug1": -5.0, "ug2": 200.0, "ia": 30.0,
                 "ig2": 3.0, "step": "op"}]})
        tab.table.setRowCount(1)
        tab.table.setItem(0, 0, item)
        tab.table.setCurrentCell(0, 0)  # fires itemSelectionChanged

    def test_servo_probes_render_under_their_own_tags(self):
        tab = self._running_tab()
        tab._on_progress(self._live_pt("op"))
        tab._on_progress(self._live_pt("bias_servo", ia=44.0))
        names = [tab.steps_table.item(r, 0).text()
                 for r in range(tab.steps_table.rowCount())]
        # The planned point inherits the PLAN's localized row name...
        assert names[0] == "✓ OP", names
        # ...while the unplanned servo probe keeps its own raw tag.
        assert any("bias_servo" in n for n in names), names

    def test_planned_live_row_inherits_plan_details(self):
        tab = self._running_tab()
        tab._planned_steps[0]["details"] = "plan says so"
        tab._on_progress(self._live_pt("op"))
        assert tab.steps_table.item(0, 10).text() == "plan says so"

    def test_servo_probe_details_show_the_deviation(self):
        from i18n_setup import t
        tab = self._running_tab()
        pt = self._live_pt("bias_servo", ia=44.0)
        pt["point"]["ref_ia"] = 48.0
        tab._on_progress(pt)
        expected = t("health.Detail_servo_probe", ref="48.0", delta="-4.0")
        assert tab.steps_table.item(0, 10).text() == expected

    def test_history_view_renders_servo_details_identically(self):
        from i18n_setup import t
        tab = _make_health_tab()
        tab._update_steps_from_points([
            {"ua": 250.0, "ug1": -6.4, "ia": 44.0, "ig2": 4.0,
             "step": "bias_servo", "ref_ia": 48.0},
            {"ua": 250.0, "ug1": -6.4, "ia": 47.9, "ig2": 4.0,
             "step": "bias_servo_op", "bias_shift_v": -0.9},
            {"ua": 250.0, "ug1": -7.3, "ia": 40.0, "ig2": 4.0,
             "step": "bias_servo_restore"},
        ])
        assert (tab.steps_table.item(0, 10).text()
                == t("health.Detail_servo_probe", ref="48.0", delta="-4.0"))
        assert (tab.steps_table.item(1, 10).text()
                == t("health.Detail_servo_op", shift="-0.90"))
        assert (tab.steps_table.item(2, 10).text()
                == t("health.Detail_servo_restore"))

    def test_plain_point_details_stay_empty(self):
        tab = _make_health_tab()
        tab._update_steps_from_points([
            {"ua": 250.0, "ug1": -8.1, "ia": 30.0, "ig2": 3.0,
             "step": "srk_s1"}])
        assert tab.steps_table.item(0, 10).text() == "—"

    def test_servo_probes_do_not_consume_plan_slots(self):
        tab = self._running_tab()
        tab._on_progress(self._live_pt("op"))
        tab._on_progress(self._live_pt("bias_servo", ia=44.0))
        tab._on_progress(self._live_pt("bias_servo", ia=46.0))
        # 3 live rows + the S-1 plan row still previewed (only "op"
        # consumed a plan slot).
        assert tab.steps_table.rowCount() == 4
        assert "S-1" in tab.steps_table.item(3, 0).text()

    def test_history_click_switches_and_live_button_returns(self):
        tab = self._running_tab()
        tab._on_progress(self._live_pt("op"))
        self._select_history_entry(tab)
        assert tab._steps_view_live is False
        assert not tab.steps_live_btn.isHidden()
        # Live points arriving now must NOT repaint the history view...
        tab._on_progress(self._live_pt("bias_servo", ia=44.0))
        assert tab.steps_table.rowCount() == 1
        # ...but they land in the buffer and return with the button.
        tab.steps_live_btn.click()
        assert tab._steps_view_live is True
        assert tab.steps_live_btn.isHidden()
        names = [tab.steps_table.item(r, 0).text()
                 for r in range(tab.steps_table.rowCount())]
        assert any("bias_servo" in n for n in names), names

    def test_history_click_while_idle_shows_no_live_button(self):
        tab = _make_health_tab()
        tab._test_active = False
        self._select_history_entry(tab)
        assert tab.steps_live_btn.isHidden()

    def test_new_run_resets_buffer_and_forces_live_view(self, monkeypatch):
        # Call-site pin on _launch_health_worker: a stale buffer or a
        # sticky history mode would leak the previous run into the new
        # one. The worker itself is stubbed out.
        import app.health_tab as HT

        class _WorkerStub:
            def __init__(self, **kw):
                self.progress = MagicMock()
                self.finished = MagicMock()
                self.failed = MagicMock()
                self.protection_triggered = MagicMock()

            def start(self):
                pass

        monkeypatch.setattr(HT, "HealthWorker", _WorkerStub)
        tab = _make_health_tab()
        tab._live_points = [{"step": "op"}]
        tab._steps_view_live = False
        tab._test_active = False
        tab.steps_live_btn.setVisible(True)
        lamp = MagicMock()
        lamp.warmup_s = 60
        tab._launch_health_worker(client=MagicMock(), lamp=lamp,
                                  lamp_id="L1", name="n", ref=None)
        assert tab._live_points == []
        assert tab._test_active is True
        assert tab._steps_view_live is True
        assert tab.steps_live_btn.isHidden()

    def test_cleanup_hides_the_live_button(self):
        tab = self._running_tab()
        self._select_history_entry(tab)
        assert not tab.steps_live_btn.isHidden()
        tab._cleanup_after_test()
        assert tab._test_active is False
        assert tab.steps_live_btn.isHidden()

    def test_refresh_planned_info_populates_the_plan_buffer(self):
        # Call-site pin: _planned_steps is filled by exactly one
        # production path (_refresh_planned_info -> _show_steps_plan).
        # Every other live-view test sets the field by hand, so without
        # this pin dropping the store would silently empty the plan tail
        # and the plan-name inheritance in the real flow.
        from test_health_logic import _lamp
        lamp = _lamp()
        tab = _make_health_tab()
        tab.get_lamps = lambda: [lamp]
        tab.tube_combo.addItem(lamp.tube_type)
        tab.tube_combo.setCurrentIndex(tab.tube_combo.count() - 1)
        tab._planned_steps = []
        tab._refresh_planned_info()
        assert tab._planned_steps, "plan buffer must be populated"
        assert tab.steps_table.rowCount() == len(tab._planned_steps)

    def test_accept_event_retags_the_last_servo_row(self):
        from i18n_setup import t
        from lm19.health import STEP_BIAS_SERVO_OP
        tab = self._running_tab()
        tab._on_progress(self._live_pt("op"))
        pt = self._live_pt("bias_servo", ia=44.0)
        pt["point"]["ref_ia"] = 48.0
        tab._on_progress(pt)
        tab._on_progress({"event": "bias_servo_accept", "ug1": -7.3,
                          "ia_ma": 44.0, "bias_shift_v": -0.9})
        # The buffered probe became the OP row...
        assert tab._live_points[-1]["step"] == STEP_BIAS_SERVO_OP
        # ...and the rendered row shows the OP tag and the shift.
        name = tab.steps_table.item(1, 0).text()
        assert STEP_BIAS_SERVO_OP in name
        assert (tab.steps_table.item(1, 10).text()
                == t("health.Detail_servo_op", shift="-0.90"))

    def test_accept_event_in_history_mode_updates_buffer_only(self):
        from lm19.health import STEP_BIAS_SERVO_OP
        tab = self._running_tab()
        tab._on_progress(self._live_pt("bias_servo", ia=44.0))
        self._select_history_entry(tab)
        tab._on_progress({"event": "bias_servo_accept", "ug1": -7.3,
                          "ia_ma": 44.0, "bias_shift_v": -0.9})
        assert tab.steps_table.rowCount() == 1, "history view untouched"
        assert tab._live_points[-1]["step"] == STEP_BIAS_SERVO_OP

    def test_full_chain_details_from_a_real_run(self):
        # Producer-to-UI chain in ONE test: the dict run_health_test
        # actually saves renders the servo Details — a key rename on
        # either side cannot stay green behind matching synthetic
        # fixtures.
        from i18n_setup import t
        from lm19.health import (
            run_health_test, BIAS_SERVO_OK, STEP_BIAS_SERVO_OP)
        from lm19.calibration import CalibrationData
        from test_health_logic import _FakeClient, _cfg, _lamp
        client = _FakeClient()
        client.state["Ua"] = 250
        client.state["Ug2"] = 250
        m = run_health_test(
            client=client, lamp=_lamp(), cfg=_cfg(),
            calibration=CalibrationData(), lamp_id="L1", name="chain",
            reference_mode="type",
            reference={"reference": {"ia": 34.0, "s": 11.0,
                                     "r": 40.0, "k": 19.0}},
            emission_enabled=False, warmup_s=1,
            measurement_plan={
                "op": {"ua": 250.0, "ug1": -7.0, "ug2": 250.0,
                       "uh": 6.3, "ih": 0.0},
                "srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5,
                        "repeats": 1},
                "bias_servo": {"enabled": True},
            },
        )
        servo = m["health"]["bias_servo"]
        assert servo["status"] == BIAS_SERVO_OK
        tab = _make_health_tab()
        tab._update_steps_from_points(m["measurement_points"])
        details_by_step = {}
        for row in range(tab.steps_table.rowCount()):
            step = tab.steps_table.item(row, 0).text()
            details_by_step.setdefault(step, []).append(
                tab.steps_table.item(row, 10).text())
        op_details = details_by_step.get(STEP_BIAS_SERVO_OP, [])
        shift = m["health"]["metrics"]["bias_shift_v"]
        expected = t("health.Detail_servo_op", shift=f"{shift:+.2f}")
        assert op_details == [expected]


class TestHistoryConditionsToggle:
    """History table OP-condition columns (Ua/Ug1/Ug2) hidden by default,
    revealed via the ``Show conditions`` checkbox in the filter row."""

    def test_condition_columns_hidden_by_default(self):
        from app.ui_theme import HEALTH_HISTORY_CONDITION_COLS
        tab = _make_health_tab()
        for col in HEALTH_HISTORY_CONDITION_COLS:
            assert tab.table.isColumnHidden(col), (
                f"col {col} (Ua/Ug1/Ug2) must be hidden by default"
            )

    def test_checkbox_unchecked_by_default(self):
        tab = _make_health_tab()
        assert not tab.show_conditions_chk.isChecked()

    def test_toggle_shows_condition_columns(self):
        from app.ui_theme import HEALTH_HISTORY_CONDITION_COLS
        tab = _make_health_tab()
        tab.show_conditions_chk.setChecked(True)
        for col in HEALTH_HISTORY_CONDITION_COLS:
            assert not tab.table.isColumnHidden(col), (
                f"col {col} must be visible when checkbox is on"
            )

    def test_toggle_hides_again(self):
        from app.ui_theme import HEALTH_HISTORY_CONDITION_COLS
        tab = _make_health_tab()
        tab.show_conditions_chk.setChecked(True)
        tab.show_conditions_chk.setChecked(False)
        for col in HEALTH_HISTORY_CONDITION_COLS:
            assert tab.table.isColumnHidden(col)


class TestHistoryDiskVsFilter:
    """ML-143: the history disk read (list_health_entries — every JSON of
    the tube type) must be separated from the display filter. Lamp ID
    keystrokes and filter-dropdown changes only re-apply the cached
    filter; they must NOT re-read the disk."""

    def _tab_with_spy(self, monkeypatch):
        import app.health_tab as HT
        calls = {"n": 0}
        entries = [
            {"lamp_id": "A", "tube_type": "EL84", "timestamp": "2026-01-01",
             "conditions": {"ug2_mode": TOPOLOGY_PENTODE}, "health": {"index": 80}},
            {"lamp_id": "B", "tube_type": "EL84", "timestamp": "2026-01-02",
             "conditions": {"ug2_mode": TOPOLOGY_PENTODE}, "health": {"index": 70}},
        ]

        def spy(tube_type):
            calls["n"] += 1
            return list(entries)
        monkeypatch.setattr(HT, "list_health_entries", spy)
        tab = _make_health_tab()
        # tube_combo is not editable — inject and select an item.
        tab.tube_combo.addItem("EL84")
        tab.tube_combo.setCurrentText("EL84")
        return tab, calls

    def test_reload_history_reads_disk(self, monkeypatch):
        tab, calls = self._tab_with_spy(monkeypatch)
        calls["n"] = 0
        tab.reload_history()
        assert calls["n"] == 1
        assert len(tab._all_history_entries) == 2
        assert tab.table.rowCount() == 2, (
            "reload must also RENDER the cached entries")

    def test_lamp_id_keystroke_does_not_read_disk(self, monkeypatch):
        tab, calls = self._tab_with_spy(monkeypatch)
        tab.reload_history()          # prime the cache
        calls["n"] = 0
        # simulate typing "E", "L", "8", "4" into the Lamp ID field
        for _ in range(4):
            tab._on_lamp_id_changed()
        assert calls["n"] == 0, (
            "Lamp ID keystrokes must not re-read the history from disk")

    def test_filter_dropdown_does_not_read_disk(self, monkeypatch):
        """Exercises the WIRING, not just the method: changing the filter
        combo index must route through _apply_history_filter (no disk),
        not reload_history. A direct method call would miss a wrong
        signal connection (checklist: call-site != function)."""
        tab, calls = self._tab_with_spy(monkeypatch)
        tab.reload_history()          # primes cache + populates combo
        assert tab.filter_lamp_combo.count() >= 2
        calls["n"] = 0
        # user picks a specific lamp in the filter dropdown
        idx = tab.filter_lamp_combo.findData("A")
        assert idx >= 0
        tab.filter_lamp_combo.setCurrentIndex(idx)
        assert calls["n"] == 0, (
            "changing the filter dropdown must not re-read the disk")

    def test_filter_still_narrows_rows(self, monkeypatch):
        """Negative control: the split must not break filtering."""
        tab, calls = self._tab_with_spy(monkeypatch)
        tab.reload_history()
        idx = tab.filter_lamp_combo.findData("A")
        assert idx >= 0
        tab.filter_lamp_combo.setCurrentIndex(idx)
        tab._apply_history_filter()
        assert all(e.get("lamp_id") == "A" for e in tab._history_entries)


class TestSaveFailureVisibility:
    """ML-084: a failed save_health_measurement must not silently discard
    the finished measurement — the label already says "Completed"; the
    user gets QMessageBox.critical and the in-memory result stays."""

    def test_on_finished_save_oserror_shows_dialog_keeps_result(
            self, monkeypatch):
        from unittest.mock import MagicMock
        tab = _make_health_tab()
        shown = []
        monkeypatch.setattr(
            "app.health_tab.save_health_measurement",
            MagicMock(side_effect=OSError("disk full")))
        monkeypatch.setattr(
            "app.save_recovery._ask_recovery",
            lambda *a, **k: (shown.append(a), "close")[1])
        # keep the test hermetic: no reads/writes of user data dirs
        monkeypatch.setattr(tab, "reload_history", lambda: None)
        monkeypatch.setattr(
            "app.health_tab.load_personal_baseline", lambda *a: {"stub": 1})
        measurement = {"tube_type": "EL84", "lamp_id": "L1",
                       "timestamp": "t", "name": "n",
                       "measurement_points": [], "conditions": {}}
        tab._on_finished(measurement)   # must not raise
        assert shown, "save failure did not reach the user"
        assert tab.last_measurement is measurement


class TestOpRampLiveHeater:
    """Call-site pin: the op_ramp handler must forward the heater channel.

    The event carries uh/ih (pinned in test_health_protection.py), but a
    handler that rebuilds a partial point without them leaves the live
    panel showing the previous value at best — and, before the optional-key
    guard in LivePanel, a false "Uh = 0". The function-level pin on
    LivePanel does not prove that its caller passes the heater through.
    """

    @staticmethod
    def _op_ramp_event(**overrides):
        evt = {
            "event": "op_ramp",
            "step_idx": 3, "total_steps": 17,
            "ug1": -10.0, "target_ug1": -7.0, "start_ug1": -24.0,
            "ua": 250.0, "ug2": 250.0,
            "uh": 6.3, "ih": 0.76,
            "ia_ma": 20.0, "ig2_ma": 2.0,
            "pa_w": 5.0, "pg2_w": 0.5,
        }
        evt.update(overrides)
        return evt

    def test_op_ramp_updates_heater_labels_from_event(self):
        tab = _make_health_tab()
        tab.live_panel.set_nominal_heater(6.3, 0.76)
        # Start from a clearly different reading so a dropped forward
        # (label frozen at 1.0 V) and a correct one (6.3 V) diverge.
        tab.live_panel._update_uh_label(1.0)
        tab.live_panel._update_ih_label(0.10)

        tab._on_progress(self._op_ramp_event())

        assert "6.3" in tab.live_panel.lbl_uh.text()
        assert "0.76" in tab.live_panel.lbl_ih.text()

    def test_op_ramp_without_heater_keeps_previous_reading(self):
        """Defence in depth: a heater-less event must not blank the labels."""
        tab = _make_health_tab()
        tab.live_panel.set_nominal_heater(6.3, 0.76)
        tab.live_panel._update_uh_label(6.3)
        tab.live_panel._update_ih_label(0.76)

        evt = self._op_ramp_event()
        del evt["uh"]
        del evt["ih"]
        tab._on_progress(evt)

        assert "6.3" in tab.live_panel.lbl_uh.text()
        assert "0.76" in tab.live_panel.lbl_ih.text()
        assert "⚠" not in tab.live_panel.lbl_uh.text()

    def test_live_point_event_still_updates_heater(self):
        """Twin path: full measurement points keep driving the labels."""
        tab = _make_health_tab()
        tab.live_panel.set_nominal_heater(6.3, 0.76)
        tab.live_panel._update_uh_label(1.0)
        tab._on_progress({
            "event": "live_point",
            "point": {"ua": 250.0, "ug1": -7.0, "ug2": 250.0,
                      "uh": 6.3, "ih": 0.76, "ia": 48.0, "ig2": 5.0},
        })
        assert "6.3" in tab.live_panel.lbl_uh.text()


class TestResultCopyButtonInTitleRow:
    """Copy sits in the Result group-box title row - the grid must
    not spend a row on it, and the button must actually land next
    to the title text after a resize."""

    def _box(self):
        import app.health_tab  # noqa: F401  (register widgets)
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        tab = _make_health_tab()
        btn = tab.result_copy_btn
        box = btn.parent()
        return tab, box, btn

    def test_button_parent_is_the_result_box(self):
        from app.widget_factory import TitleRowButtonGroupBox
        tab, box, btn = self._box()
        assert isinstance(box, TitleRowButtonGroupBox)
        assert box.title() # the Result box, not some container

    def test_button_not_in_the_grid(self):
        tab, box, btn = self._box()
        assert box.layout().indexOf(btn) == -1

    def test_button_positioned_after_the_title_on_resize(self):
        from app.widget_factory import TITLE_ROW_BUTTON_GAP_PX
        tab, box, btn = self._box()
        # Hidden widgets defer resize delivery, and a child's show() is
        # inert while its parent is hidden — the tab must be shown, same
        # as the first real frame in the application.
        tab.show()
        box.resize(420, 130)
        rect = box.title_label_rect()
        assert btn.x() == rect.right() + TITLE_ROW_BUTTON_GAP_PX
        assert btn.y() >= 0
        assert btn.y() <= rect.bottom()

    def test_click_still_copies_the_result(self):
        from PySide6.QtWidgets import QApplication
        tab, box, btn = self._box()
        tab.result_index.setText("Index: 78%")
        btn.click()
        assert "Index: 78%" in QApplication.clipboard().text()


class TestStatusRowSingleLine:
    """State + progress bar + phase label share one row — three
    stacked rows collapsed into one."""

    def test_all_three_on_one_line(self):
        tab = _make_health_tab()
        tab.show()
        ls, pb, pl = tab.live_state, tab.progress, tab.progress_label
        assert ls.y() == pb.y() == pl.y()
        assert ls.x() < pb.x() < pl.x()

    def test_bar_absorbs_the_stretch(self):
        # The bar must be the stretching element: labels keep natural
        # width, otherwise long phase texts squeeze the bar to nothing.
        tab = _make_health_tab()
        tab.show()
        assert tab.progress.width() > tab.live_state.width()
        assert tab.progress.width() > tab.progress_label.width()
