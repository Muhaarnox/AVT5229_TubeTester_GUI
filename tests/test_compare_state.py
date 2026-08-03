"""Compare tab: state survives re-render.

Pins:
- ML-033: Show checkmarks and user colors survive ANY re-render
  (add entry, sort, extend), not just sorting;
- ML-034: An sorting reads the same path as the renderer
  (scan.an -> conditions.an) and does not crash on string codes;
- ML-035: track-curve points stay in the snap marker under an active
  Ug2 filter (curves draw unfiltered — the marker must see them);
- ML-036: a repeated Match washes the previous result's row fill.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest
from PySide6.QtCore import Qt
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE_CONNECTED,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _entry(lamp_id: str, an=None, track: bool = False,
           n_pts: int = 12) -> Dict:
    pts: List[Dict] = []
    for i in range(n_pts):
        ua = 50.0 + 20.0 * i
        pts.append({
            "ua": ua, "ug1": -2.0 - (i % 3), "ia": 5.0 + i,
            "ug2": (ua + 20.0) if track else 250.0,
        })
    return {
        "lamp_type": "EL84", "lamp_id": lamp_id, "name": f"run_{lamp_id}",
        "timestamp": f"2026-07-0{(hash(lamp_id) % 8) + 1}T10:00:00",
        "points": pts,
        "data": {"topology": TOPOLOGY_PENTODE,
                 "scan": ({"an": an} if an is not None else {})
                 | ({"ug2_mode": TOPOLOGY_TRIODE_CONNECTED} if track
                    else {"ug2_mode": TOPOLOGY_PENTODE})},
    }


@pytest.fixture()
def tab(qapp):
    from app.compare_tab import CompareTab
    return CompareTab()


# -- ML-142: Find similar via progress + settings ---------------------

class TestFindSimilarProgressAndSettings:
    """ML-142: Find similar must run behind the shared progress+Cancel
    dialog and honor the Match-row Max Δ / Min pts (both were ignored)."""

    def test_find_similar_routes_through_progress_helper(self, tab, monkeypatch):
        seen = {}

        def fake_helper(entries, labels, **kw):
            seen["kw"] = kw
            seen["n"] = len(entries)
            from lm19.tube_matching import CurveMatchResult
            return CurveMatchResult(mode="similar", groups=[], unmatched=[],
                                    pair_info={}, anchor_idx=kw.get("anchor_idx"))
        monkeypatch.setattr(tab, "_match_curves_with_progress", fake_helper)
        for lid in ("L1", "L2", "L3"):
            tab.add_entry(_entry(lid))
        tab.match_max_delta_spin.setValue(7.5)
        tab.match_min_overlap_spin.setValue(9)
        tab._find_similar_curve(0)
        assert seen, "find similar must call the progress helper"
        assert seen["kw"]["mode"] == "similar"
        assert seen["kw"]["max_delta"] == 7.5
        assert seen["kw"]["min_overlap"] == 9

    def test_helper_forwards_live_progress_callback(self, tab, monkeypatch):
        """Mutation-audit: the earlier pin mocked the helper
        itself, so the helper's OWN wiring (progress → match_curves) was
        unpinned — progress=None would kill Cancel silently."""
        seen = {}

        def fake_match_curves(entries, labels, progress=None, **kw):
            seen["progress"] = progress
            from lm19.tube_matching import CurveMatchResult
            return CurveMatchResult(mode="similar", groups=[], unmatched=[],
                                    pair_info={}, anchor_idx=0)
        monkeypatch.setattr("app.compare_tab.match_curves", fake_match_curves)
        entries = [_entry("L1"), _entry("L2")]
        labels = ["L1", "L2"]
        tab._match_curves_with_progress(entries, labels, mode="similar",
                                        anchor_idx=0)
        assert callable(seen.get("progress")), (
            "helper must forward a live progress callback (Cancel support)")
        assert seen["progress"](1, 2) in (True, False)

    def test_find_similar_passes_filtered_anchor_idx(self, tab, monkeypatch):
        """Mutation-audit (degenerate-data checklist): the
        original pin clicked row 0, where anchor_idx == default 0 —
        dropping the argument survived. Row 2 with a track-mode entry in
        between makes filtered index (1) differ from the row (2)."""
        seen = {}

        def fake_helper(entries, labels, **kw):
            seen["kw"] = kw
            from lm19.tube_matching import CurveMatchResult
            return CurveMatchResult(mode="similar", groups=[], unmatched=[],
                                    pair_info={}, anchor_idx=kw.get("anchor_idx"))
        monkeypatch.setattr(tab, "_match_curves_with_progress", fake_helper)
        tab.add_entry(_entry("L1"))
        tab.add_entry(_entry("L2", track=True))   # different ug2_mode
        tab.add_entry(_entry("L3"))
        tab._find_similar_curve(2)                # L3: filtered idx = 1
        assert seen["kw"].get("anchor_idx") == 1, (
            "anchor must be the index WITHIN the mode-filtered entries")

    def test_find_similar_cancel_is_graceful(self, tab, monkeypatch):
        """Helper returns None on Cancel — must not crash or set result."""
        monkeypatch.setattr(tab, "_match_curves_with_progress",
                            lambda *a, **k: None)
        for lid in ("L1", "L2"):
            tab.add_entry(_entry(lid))
        tab._match_result = None
        tab._find_similar_curve(0)
        assert tab._match_result is None


# ── ML-033 ───────────────────────────────────────────────────────────

class TestStateSurvivesRerender:

    def test_check_and_color_survive_add(self, tab):
        from app.compare_tab import _COL_COLOR
        tab.add_entry(_entry("L1"))
        tab.table.item(0, 0).setCheckState(Qt.CheckState.Checked)
        color_item = tab.table.item(0, _COL_COLOR)
        color_item.setText("#123456")
        # any re-render: adding a second entry
        tab.add_entry(_entry("L2"))
        assert tab.table.item(0, 0).checkState() == Qt.CheckState.Checked
        assert tab.table.item(0, _COL_COLOR).text() == "#123456"
        assert tab.table.item(1, 0).checkState() == Qt.CheckState.Unchecked

    def test_state_follows_entry_through_sort(self, tab):
        tab.add_entry(_entry("B"))
        tab.add_entry(_entry("A"))
        tab.table.item(0, 0).setCheckState(Qt.CheckState.Checked)  # lamp B
        tab._sort_table(2)  # sort by Lamp ID → A first
        ids = [tab.compare_entries[r]["lamp_id"] for r in range(2)]
        assert ids == ["A", "B"]
        checked = {ids[r]: tab.table.item(r, 0).checkState()
                   for r in range(2)}
        assert checked["B"] == Qt.CheckState.Checked
        assert checked["A"] == Qt.CheckState.Unchecked


# ── ML-034 ───────────────────────────────────────────────────────────

class TestAnSortMatchesRender:

    def test_sort_order_uses_render_path(self, tab):
        """The key must read scan.an -> conditions.an (like the renderer)
        and order by the DISPLAYED value — a key reading the wrong path
        gives all entries the same key and a stable (original) order
        (mutation lesson RV2)."""
        from app.compare_tab import _COL_AN
        tab.add_entry(_entry("L1", an="Au"))          # scan.an = "Au"
        health = _entry("L2")
        health["data"]["conditions"] = {"an": 2}       # health-style path
        tab.add_entry(health)
        tab.add_entry(_entry("L3"))                    # no an anywhere
        tab._sort_table(_COL_AN)
        order = [tab.compare_entries[r]["lamp_id"] for r in range(3)]
        # ascending by str key: "" < "2" < "Au"
        assert order == ["L3", "L2", "L1"], order
        # and the cells mirror the same path (no crash on mixed str/int)
        texts = [tab.table.item(r, _COL_AN).text() for r in range(3)]
        assert texts == ["", "2", "Au"]


# ── ML-035 ───────────────────────────────────────────────────────────

class TestMarkerKeepsTrackPoints:

    def test_track_points_survive_ug2_filter(self, tab):
        tab.add_entry(_entry("SW", track=False))   # sweep 250 V
        tab.add_entry(_entry("TR", track=True))    # track: ug2 = ua + 20
        for row in range(2):
            tab.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
        tab._plot_selected()
        # Ug2-filter panel exists with the sweep's 250 V level; disable it
        assert tab._ug2_checked, "ug2 panel must have levels"
        for k in list(tab._ug2_checked):
            tab._ug2_checked[k] = False
        tab._plot_selected()
        kinds = {p.get("lamp_id") for p in tab._compare_points_data}
        assert "TR" in kinds, \
            "track-curve points must survive the Ug2 filter (drawn unfiltered)"
        assert "SW" not in kinds, \
            "sweep points at the disabled level must be filtered"


# ── ML-036 ───────────────────────────────────────────────────────────

class TestMatchWashesStaleFill:

    def test_regrouped_row_loses_old_background(self, tab, qapp):
        from lm19.tube_matching import CurveMatchResult
        from app.compare_tab import _COL_GRP
        tab.add_entry(_entry("L1"))
        tab.add_entry(_entry("L2"))
        e1, e2 = tab.compare_entries

        # first match: both in group 1
        tab._match_result = CurveMatchResult(
            mode="groups", groups=[[0, 1]], unmatched=[], pair_info={})
        tab._match_entries = [e1, e2]
        tab._match_grp_info = {id(e1): (1, 1.0, 30, False),
                               id(e2): (1, 1.0, 30, False)}
        tab._match_unmatched_ids = set()
        tab._apply_curve_match_to_table()
        bg_before = tab.table.item(0, 2).background().color().name()

        # second match: NOTHING matched — fills must wash off
        tab._match_result = CurveMatchResult(
            mode="groups", groups=[], unmatched=[0, 1], pair_info={})
        tab._match_grp_info = {}
        tab._match_unmatched_ids = set()  # not even flagged unmatched
        tab._apply_curve_match_to_table()
        item = tab.table.item(0, 2)
        assert item.background().color().name() != bg_before or \
            item.background().style() == Qt.BrushStyle.NoBrush, \
            "stale group fill must be washed on re-match"
        assert tab.table.item(0, _COL_GRP).text() == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
