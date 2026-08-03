"""Pins for the Measure tab's control column / plot splitter.

The column used to be added with stretch 0, so it took exactly the width
its widest group asked for and the user had no way to give the plots more
room. It is now a QSplitter, and both sides COMPRESS under the handle:
each carries an explicit minimum width, which Qt honours over the layout's
own minimumSizeHint. No scroll areas — the controls themselves give way.
"""

import os
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QScrollArea, QSplitter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow

pytestmark = [pytest.mark.smoke_ui]


# ── Module local constants ──

# Width the splitter is dragged to when testing the stops; far outside
# the allowed range on purpose, so the clamp is what decides.
_DRAG_FAR_CLOSED = 50
_DRAG_FAR_OPEN = 5000
# Slack for frame/scrollbar/tab-widget chrome around the page.
_CHROME_SLACK_PX = 60
# Narrowest screen the layout must stay usable on: both sides' minimum
# widths have to fit inside it, or the handle has nowhere to travel and
# the control column is squeezed to its floor from the start.
_MIN_SUPPORTED_WINDOW_W = 1280


class _Port:
    def __init__(self, device: str) -> None:
        self.device = device


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    from unittest.mock import patch

    with patch("app.main_window.list_ports.comports",
               return_value=[_Port("COM1")]):
        w = MainWindow()
    w.resize(1600, 900)
    w.show()
    qapp.processEvents()
    yield w
    w.close()


def _measure_page(window):
    """The controls page — a plain widget, deliberately not scrolled."""
    page = window.left_tabs.widget(0)
    assert not isinstance(page, QScrollArea), (
        "the control column must compress, not scroll")
    return page


class TestSplitterWiring:

    def test_splitter_holds_controls_and_plots(self, window):
        sp = window.measure_splitter
        assert isinstance(sp, QSplitter)
        assert sp.orientation() == Qt.Orientation.Horizontal
        assert sp.count() == 2
        assert sp.widget(0) is window.left_tabs
        assert sp.widget(1) is window.plot_tabs.parentWidget()

    def test_extra_window_width_goes_to_the_plots(self, window):
        """Stretch factors, pinned by behaviour: on a wider window the
        control column keeps its width and the plots grow."""
        sp = window.measure_splitter
        before = sp.sizes()
        window.resize(window.width() + 400, window.height())
        QApplication.processEvents()
        after = sp.sizes()
        assert after[0] == before[0]
        assert after[1] > before[1]

    def test_neither_side_can_be_collapsed(self, window):
        assert not window.measure_splitter.childrenCollapsible()


class TestSplitterTravel:

    def test_column_opens_wide_enough_for_its_content(self, window):
        """A splitter distributes by size hints on its first layout, and
        at construction time the operating-point line is still empty, so
        without the explicit fit the column opens too narrow to show it."""
        page = _measure_page(window)
        assert (window.measure_splitter.sizes()[0]
                >= page.sizeHint().width())

    def test_narrow_screen_keeps_the_plot_side_usable(self, window):
        """On a screen too small for both natural widths the column must
        give way — taking its full natural width would leave the plots
        below their own minimum."""
        sp = window.measure_splitter
        page = _measure_page(window)
        right_min = sp.widget(1).minimumSizeHint().width()
        total = page.sizeHint().width() + right_min - 200
        window._fit_measure_splitter(total=total)
        QApplication.processEvents()
        assert sp.sizes()[0] < page.sizeHint().width()
        assert sp.sizes()[0] >= MainWindow.LEFT_PANEL_MIN_W

    def test_column_shrinks_below_its_content(self, window):
        """The point of the splitter: without an explicit minimum the
        page's own minimumSizeHint (its widest group) is the left stop and
        the handle barely moves."""
        sp = window.measure_splitter
        page = _measure_page(window)
        sp.setSizes([_DRAG_FAR_CLOSED, _DRAG_FAR_OPEN])
        QApplication.processEvents()
        assert sp.sizes()[0] < page.sizeHint().width()

    def test_column_stops_at_the_floor(self, window):
        sp = window.measure_splitter
        sp.setSizes([_DRAG_FAR_CLOSED, _DRAG_FAR_OPEN])
        QApplication.processEvents()
        assert sp.sizes()[0] >= MainWindow.LEFT_PANEL_MIN_W

    def test_both_minimums_fit_a_small_screen(self, window):
        """The bug this pins: the plot side's minimum was the width of the
        Plot options rows (~1067 px). Together with the column floor it
        exceeded a 1366-wide window, so the handle could not move and the
        column opened squeezed to its floor — "the splitter does nothing"."""
        sp = window.measure_splitter
        total = (sp.widget(0).minimumSizeHint().width()
                 + sp.widget(1).minimumSizeHint().width()
                 + sp.handleWidth())
        assert total <= _MIN_SUPPORTED_WINDOW_W

    def test_first_show_fits_the_column_only_once(self, window):
        """A later show (restore from minimised) must not undo a width the
        user dragged to."""
        sp = window.measure_splitter
        sp.setSizes([MainWindow.LEFT_PANEL_MIN_W + 40, _DRAG_FAR_OPEN])
        QApplication.processEvents()
        chosen = sp.sizes()[0]
        window.hide()
        window.show()
        QApplication.processEvents()
        assert sp.sizes()[0] == chosen

    def test_column_can_be_widened(self, window):
        sp = window.measure_splitter
        start = sp.sizes()[0]
        sp.setSizes([start + 200, sp.sizes()[1] - 200])
        QApplication.processEvents()
        assert sp.sizes()[0] > start


class TestControlsCompressRatherThanScroll:

    def test_no_scroll_area_wraps_either_side(self, window):
        """Deliberate layout policy: dragging the handle squeezes the
        controls; it must not park them behind scrollbars."""
        sp = window.measure_splitter
        # The Amplifier control panel brought its own scroll area long
        # before the splitter existed; the Measure page and the plot
        # column must not gain one.
        assert not isinstance(window.left_tabs.widget(0), QScrollArea)
        right = sp.widget(1)
        direct = [right.layout().itemAt(i).widget()
                  for i in range(right.layout().count())]
        assert not any(isinstance(w, QScrollArea) for w in direct)

    def test_page_actually_narrows_with_the_handle(self, window):
        """Compression is real: the page follows the splitter instead of
        keeping its natural width behind a viewport."""
        sp = window.measure_splitter
        page = _measure_page(window)
        sp.setSizes([_DRAG_FAR_CLOSED, _DRAG_FAR_OPEN])
        QApplication.processEvents()
        assert page.width() <= sp.sizes()[0] + _CHROME_SLACK_PX
        assert page.width() < page.sizeHint().width()
