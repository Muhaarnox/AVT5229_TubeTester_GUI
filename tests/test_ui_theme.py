"""Tests for app.ui_theme color functions.

Run:  py -m pytest tests/test_ui_theme.py -v
"""

from app.ui_theme import (
    COLOR_LEVEL_BAD,
    COLOR_LEVEL_BAD_TEXT,
    COLOR_LEVEL_GOOD,
    COLOR_LEVEL_GOOD_TEXT,
    COLOR_LEVEL_WARN,
    COLOR_LEVEL_WARN_TEXT,
    level_color_high_good,
    level_color_low_good,
)


class TestLevelColorTextVariants:
    """level_color_*() returns TEXT-mode colors (darker, for HTML on white).

    Pastel plot colors (#ffe66d et al.) have poor contrast on QLabel's
    light background — the result_label in amp_control_panel renders HTML
    on light Qt-default background. We use darker variants for legibility.
    """

    def test_text_colors_distinct_from_plot_colors(self):
        """Sanity: TEXT variants differ from PLOT variants."""
        assert COLOR_LEVEL_GOOD_TEXT != COLOR_LEVEL_GOOD
        assert COLOR_LEVEL_WARN_TEXT != COLOR_LEVEL_WARN
        assert COLOR_LEVEL_BAD_TEXT != COLOR_LEVEL_BAD

    def test_low_good_returns_text_variants(self):
        # Below good_max → good
        assert level_color_low_good(0.5, good_max=1.0, warn_max=5.0) == COLOR_LEVEL_GOOD_TEXT
        # Between good_max and warn_max → warn
        assert level_color_low_good(3.0, good_max=1.0, warn_max=5.0) == COLOR_LEVEL_WARN_TEXT
        # Above warn_max → bad
        assert level_color_low_good(10.0, good_max=1.0, warn_max=5.0) == COLOR_LEVEL_BAD_TEXT

    def test_high_good_returns_text_variants(self):
        # Above good_min → good
        assert level_color_high_good(5.0, good_min=2.0, warn_min=0.5) == COLOR_LEVEL_GOOD_TEXT
        # Between warn_min and good_min → warn
        assert level_color_high_good(1.0, good_min=2.0, warn_min=0.5) == COLOR_LEVEL_WARN_TEXT
        # Below warn_min → bad
        assert level_color_high_good(0.1, good_min=2.0, warn_min=0.5) == COLOR_LEVEL_BAD_TEXT

    def test_low_good_boundary_warn(self):
        """value == good_max falls into warn (strict <)."""
        assert level_color_low_good(1.0, good_max=1.0, warn_max=5.0) == COLOR_LEVEL_WARN_TEXT

    def test_high_good_boundary_warn(self):
        """value == good_min falls into warn (strict >)."""
        assert level_color_high_good(2.0, good_min=2.0, warn_min=0.5) == COLOR_LEVEL_WARN_TEXT


class TestTextColorContrast:
    """Sanity: TEXT colors must be perceptibly darker than plot colors,
    so they remain readable on light QLabel background."""

    @staticmethod
    def _luminance(hex_color: str) -> float:
        h = hex_color.lstrip("#")
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        # Approximate relative luminance (sRGB linear coefficients)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def test_text_warn_darker_than_plot_warn(self):
        assert self._luminance(COLOR_LEVEL_WARN_TEXT) < self._luminance(COLOR_LEVEL_WARN)

    def test_text_good_darker_than_plot_good(self):
        assert self._luminance(COLOR_LEVEL_GOOD_TEXT) < self._luminance(COLOR_LEVEL_GOOD)

    def test_text_bad_darker_than_plot_bad(self):
        assert self._luminance(COLOR_LEVEL_BAD_TEXT) < self._luminance(COLOR_LEVEL_BAD)

    def test_text_colors_have_minimum_contrast_on_white(self):
        """Each TEXT variant must have luminance < 0.5 → readable on white."""
        for color in (COLOR_LEVEL_GOOD_TEXT, COLOR_LEVEL_WARN_TEXT, COLOR_LEVEL_BAD_TEXT):
            assert self._luminance(color) < 0.5, f"{color} too bright for white bg"
