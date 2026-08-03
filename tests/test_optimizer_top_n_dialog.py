"""Tests for OptimizerTopNDialog — Top-N picker dialog.

Run:  py -m pytest tests/test_optimizer_top_n_dialog.py -v
"""

from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.optimizer_top_n_dialog import OptimizerTopNDialog
from lm19.optimizer import OptPoint
from lm19.amplifier.constants import (
    HD_METHOD_5POINT,
    HD_METHOD_CHEBYSHEV_PP,
)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_pt(thd=2.0, pout_mw=1000.0, ub=250.0, ug2=0.0, ug1=-7.0,
             ra=10.0, swing=4.0, p_classA_w=0.0, hd_method=HD_METHOD_5POINT):
    return OptPoint(
        ub=ub, ug2=ug2, ug1=ug1, ra=ra,
        thd=thd, hd2=thd * 0.7, hd3=thd * 0.3,
        pout_mw=pout_mw, pa_mw=2000.0,
        ia_0=10.0, ua_0=200.0, amp_class="A",
        max_swing=swing, half_swing=swing,
        p_classA_w=p_classA_w, hd_method=hd_method,
    )


class TestOptimizerTopNDialog:

    def test_table_populated_with_candidates(self):
        candidates = [
            _make_pt(thd=1.0, pout_mw=500.0),
            _make_pt(thd=2.0, pout_mw=1500.0),
            _make_pt(thd=3.0, pout_mw=2500.0),
        ]
        dlg = OptimizerTopNDialog(candidates)
        assert dlg.table.rowCount() == 3
        # First row should display rank "1"
        assert dlg.table.item(0, 0).text() == "1"

    def test_pa_column_hidden_for_se_only(self):
        """SE/CF candidates have p_classA_w=0 → P_A column hidden."""
        candidates = [_make_pt(p_classA_w=0.0) for _ in range(3)]
        dlg = OptimizerTopNDialog(candidates)
        # Column index 8 is P_A (per dialog implementation)
        assert dlg.table.isColumnHidden(8)

    def test_pa_column_visible_for_pp(self):
        """At least one PP point with p_classA_w>0 → P_A column visible."""
        candidates = [
            _make_pt(p_classA_w=0.0),
            _make_pt(p_classA_w=2.5),
        ]
        dlg = OptimizerTopNDialog(candidates)
        assert not dlg.table.isColumnHidden(8)

    def test_numeric_column_sorts_numerically(self):
        """Ra column {9,10,2} must sort 2,9,10 (numeric), not lexicographic
        10,2,9 (plain QTableWidgetItem compares the DisplayRole string)."""
        candidates = [_make_pt(ra=9.0), _make_pt(ra=10.0), _make_pt(ra=2.0)]
        dlg = OptimizerTopNDialog(candidates)
        RA_COL = 4  # #,Ub,Ug2,Ug1,Ra,...
        dlg.table.sortByColumn(RA_COL, Qt.SortOrder.AscendingOrder)
        vals = [float(dlg.table.item(r, RA_COL).text().split()[0])
                for r in range(dlg.table.rowCount())]
        assert vals == [2.0, 9.0, 10.0]

    def test_rank_column_sorts_numerically(self):
        """# column must sort 1..12, not lexicographic '1','10','11','12','2'."""
        dlg = OptimizerTopNDialog([_make_pt() for _ in range(12)])
        dlg.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        ranks = [int(dlg.table.item(r, 0).text())
                 for r in range(dlg.table.rowCount())]
        assert ranks == list(range(1, 13))

    def test_mixed_numeric_and_dash_column_sort_does_not_crash(self):
        """A column mixing numeric cells and '—' dash cells (ug2=0) must sort
        without crashing — FormattedNumericItem.__lt__ vs a plain dash item must
        not recurse (super().__lt__ re-dispatches into the override on PySide6)."""
        cands = [_make_pt(ug2=0.0), _make_pt(ug2=250.0),
                 _make_pt(ug2=0.0), _make_pt(ug2=150.0)]
        dlg = OptimizerTopNDialog(cands)
        for col in (5, 8, 10):  # Swing, P_A, UL — mixed dash columns: no crash
            dlg.table.sortByColumn(col, Qt.SortOrder.AscendingOrder)  # RecursionError on unfixed
        # Sort Ug2 (col 2, mixed) LAST and verify numeric ordering of non-dash rows.
        dlg.table.sortByColumn(2, Qt.SortOrder.AscendingOrder)
        ug2 = [dlg.table.item(r, 2).text() for r in range(dlg.table.rowCount())]
        nums = [float(t.split()[0]) for t in ug2 if t != "—"]
        assert nums == sorted(nums)

    def test_no_selection_no_apply(self):
        """Clicking Apply without selection doesn't accept."""
        dlg = OptimizerTopNDialog([_make_pt()])
        dlg.table.clearSelection()
        dlg._accept_selection()
        # Should not have set selected_point (no row selected)
        assert dlg.selected_point is None

    def test_first_row_preselected(self):
        """Dialog opens with first row selected."""
        candidates = [_make_pt(thd=t) for t in (1.0, 2.0, 3.0)]
        dlg = OptimizerTopNDialog(candidates)
        rows = dlg.table.selectionModel().selectedRows()
        assert len(rows) == 1
        assert rows[0].row() == 0

    def test_accept_selection_returns_chosen_point(self):
        """Selecting row 1 and accepting yields second candidate."""
        candidates = [
            _make_pt(thd=1.0, ub=200.0),
            _make_pt(thd=2.0, ub=300.0),
        ]
        dlg = OptimizerTopNDialog(candidates)
        dlg.table.selectRow(1)
        dlg._accept_selection()
        assert dlg.selected_point is not None
        assert dlg.selected_point.ub == 300.0

    def test_method_column_shows_label(self):
        """hd_method label appears in last column."""
        dlg = OptimizerTopNDialog([_make_pt(hd_method=HD_METHOD_CHEBYSHEV_PP)])
        # Method column is at index 9
        assert dlg.table.item(0, 9).text() == HD_METHOD_CHEBYSHEV_PP

    def test_empty_candidates_handled_gracefully(self):
        """No candidates → empty table, no crash."""
        dlg = OptimizerTopNDialog([])
        assert dlg.table.rowCount() == 0
        assert dlg.selected_point is None

    def test_swing_zero_displayed_as_dash(self):
        """Half-swing 0 → "—" (no swing optimized)."""
        dlg = OptimizerTopNDialog([_make_pt(swing=0.0)])
        # Swing is column 5
        assert dlg.table.item(0, 5).text() == "—"

    def test_ug2_zero_displayed_as_dash_for_triode(self):
        """Triode (ug2=0) → "—" in Ug2 column."""
        dlg = OptimizerTopNDialog([_make_pt(ug2=0.0)])
        # Ug2 is column 2
        assert dlg.table.item(0, 2).text() == "—"

    def test_ul_tap_column_hidden_when_all_zero(self):
        """All candidates with ul_tap=0 → UL column hidden."""
        from app.optimizer_top_n_dialog import _UL_COL_INDEX
        dlg = OptimizerTopNDialog([_make_pt(), _make_pt()])
        assert dlg.table.isColumnHidden(_UL_COL_INDEX)

    def test_ul_tap_column_visible_when_any_nonzero(self):
        """Even one candidate with ul_tap>0 → UL column visible."""
        from app.optimizer_top_n_dialog import _UL_COL_INDEX
        # Need ul_tap kwarg in OptPoint constructor
        from lm19.optimizer import OptPoint
        pt_pent = _make_pt()
        pt_ul = OptPoint(
            ub=250.0, ug2=0.0, ug1=-7.0, ra=10.0,
            thd=2.0, hd2=1.4, hd3=0.6,
            pout_mw=1000.0, pa_mw=2000.0,
            ia_0=10.0, ua_0=200.0, amp_class="A",
            max_swing=4.0, half_swing=4.0,
            ul_tap=0.43,
        )
        dlg = OptimizerTopNDialog([pt_pent, pt_ul])
        assert not dlg.table.isColumnHidden(_UL_COL_INDEX)

    def test_ul_tap_column_renders_percentage(self):
        """UL tap 0.43 displays as '43 %'."""
        from lm19.optimizer import OptPoint
        from app.optimizer_top_n_dialog import _UL_COL_INDEX
        pt = OptPoint(
            ub=250.0, ug2=0.0, ug1=-7.0, ra=10.0,
            thd=2.0, hd2=1.4, hd3=0.6,
            pout_mw=1000.0, pa_mw=2000.0,
            ia_0=10.0, ua_0=200.0, amp_class="A",
            max_swing=4.0, half_swing=4.0,
            ul_tap=0.43,
        )
        dlg = OptimizerTopNDialog([pt])
        text = dlg.table.item(0, _UL_COL_INDEX).text()
        assert "43" in text and "%" in text
