"""Layout pins for the Measure tab's "Plot options" group.

The group is four rows built by separate ``_build_plot_row_*`` methods,
so a control can be constructed, wired and still never reach a layout —
invisible, with every unit pin on it still green. These pins derive the
placement from the group's own widget tree and assert it against the
intended per-row composition.

Row semantics: 0 = which data, 1 = what is plotted and in which axes,
2 = limit overlays, 3 = working line + drawing style.
"""

import inspect
import os
import re
import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QPushButton,
    QRadioButton,
    QWidget,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow
from app.main_window_builders import MainWindowBuilders
from i18n_setup import t

pytestmark = [pytest.mark.smoke_ui]


# ── Module local constants ──

# Interactive widget types the group is built from; captions (plain
# QLabel) carry no state and are deliberately not tracked.
_CONTROL_TYPES = (QCheckBox, QComboBox, QPushButton, QRadioButton,
                  QAbstractSpinBox)

# Intended composition, one entry per row of the group.
_ROWS = (
    {"lamp_display_combo", "ug2_display_combo", "lamp_calc_combo",
     "ug2_calc_combo", "clear_plot_btn"},
    {"ug2_mode_series", "ug2_mode_color", "ia_max_input",
     "ia_max_auto_btn", "ua_auto_btn", "legend_toggle_btn"},
    {"pa_max_cb", "pa_max_input", "pg2_max_cb", "pg2_max_input",
     "ua_max_cb", "ua_max_input", "ia_max_limit_cb",
     "ia_max_limit_input", "model_btn"},
    {"load_line_cb", "ra_sweep_btn", "plot_line_width",
     "overlay_pen_style", "heatmap_cmap_combo", "heatmap_lock_cb"},
)

# Crowding budget for a single row. Row 1 once held 10 controls and read
# as a wall; the drawing-style block moved to row 3. Raising this number
# is a UI decision, not a refactor side effect.
_MAX_CONTROLS_PER_ROW = 9

# The row builders themselves are the source of truth for what the group
# creates — a hand-kept list would only repeat the author's blind spots.
_ROW_BUILDERS = tuple(sorted(
    name for name in dir(MainWindowBuilders)
    if name.startswith("_build_plot_row_")))
_SELF_ASSIGN = re.compile(r"self\.(\w+)\s*=\s*(?!=)")


class _Port:
    def __init__(self, device: str) -> None:
        self.device = device


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def window(qapp):
    from unittest.mock import patch

    with patch("app.main_window.list_ports.comports",
               return_value=[_Port("COM1")]):
        w = MainWindow()
    yield w
    w.close()


def _plot_options_box(window) -> QGroupBox:
    title = t('plot.Plot_options')
    for box in window.findChildren(QGroupBox):
        if box.title() == title:
            return box
    raise AssertionError(f"no group box titled {title!r}")


def _names_by_id(window) -> dict:
    """Widget id → attribute name it is reachable through."""
    return {id(v): k for k, v in vars(window).items()
            if isinstance(v, QWidget)}


def _row_control_names(window, row_index: int) -> set:
    box = _plot_options_box(window)
    row = box.layout().itemAt(row_index).layout()
    assert row is not None, f"row {row_index} is not a layout"
    names = _names_by_id(window)
    out = set()
    for i in range(row.count()):
        w = row.itemAt(i).widget()
        if isinstance(w, _CONTROL_TYPES):
            assert id(w) in names, (
                f"row {row_index} holds an interactive widget that is not "
                f"reachable as a window attribute: {w!r}")
            out.add(names[id(w)])
    return out


class TestPlotOptionsRows:

    def test_group_has_the_expected_row_count(self, window):
        assert _plot_options_box(window).layout().count() == len(_ROWS)

    @pytest.mark.parametrize("index", range(len(_ROWS)))
    def test_row_composition(self, window, index):
        assert _row_control_names(window, index) == _ROWS[index]

    def test_one_expectation_per_row_builder(self, window):
        """_ROWS must track the builders — a fifth row added without an
        entry here would go unpinned."""
        assert len(_ROW_BUILDERS) == len(_ROWS)

    def test_every_control_is_placed_exactly_once(self, window):
        """Source of truth is the builders' own ``self.x = ...`` lines, not
        the group's widget tree: a control that is built and wired but never
        added to a layout has no parent, so it would not show up in the tree
        at all — and every unit pin on it still passes."""
        built = set()
        for meth in _ROW_BUILDERS:
            src = inspect.getsource(getattr(MainWindowBuilders, meth))
            for name in _SELF_ASSIGN.findall(src):
                if isinstance(getattr(window, name, None), _CONTROL_TYPES):
                    built.add(name)
        assert built, "no controls found in the row builders"
        placed: list = []
        for index in range(len(_ROWS)):
            placed.extend(_row_control_names(window, index))
        assert sorted(placed) == sorted(set(placed)), "control placed twice"
        assert built == set(placed)

    def test_no_row_is_overcrowded(self, window):
        """The complaint that started this layout: one row carried the
        whole group while the last one held two controls."""
        sizes = {i: len(_ROWS[i]) for i in range(len(_ROWS))}
        too_full = {i: n for i, n in sizes.items()
                    if n > _MAX_CONTROLS_PER_ROW}
        assert too_full == {}

    def test_style_controls_sit_with_the_working_line(self, window):
        """The moved block, pinned by name: width / overlay pen / palette
        answer "how it is drawn", so they belong to the last row."""
        last = _row_control_names(window, len(_ROWS) - 1)
        assert {"plot_line_width", "overlay_pen_style",
                "heatmap_cmap_combo", "heatmap_lock_cb"} <= last
        for index in range(len(_ROWS) - 1):
            assert not (_row_control_names(window, index) & last)
