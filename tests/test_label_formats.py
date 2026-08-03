"""Unit tests for lm19/label_formats.py.

``format_label`` is the single source of truth for value display across 8+
render/UI modules — an unnoticed format regression would ripple to every graph
label. This module was previously untested (blind zone, audit-tests).
"""
import string

from lm19.label_formats import LABEL_FORMATS, format_label


class TestFormatLabel:
    def test_known_kinds_basic(self):
        assert format_label("ug1", -1.5) == "Ug1 -1.5 V"
        assert format_label("ua", 250.0) == "Ua 250 V"
        assert format_label("ia", 12.5) == "Ia 12.50 mA"
        assert format_label("ih", 0.3) == "Ih 0.300 A"
        assert format_label("ra", 8.0) == "Ra=8.0kΩ"

    def test_negative_and_zero(self):
        assert format_label("ug1", 0.0) == "Ug1 0.0 V"
        assert format_label("ug1_short", -2.0) == "-2.0V"
        assert format_label("pa", 0.0) == "Pa 0.00 W"

    def test_multikey_formats(self):
        assert format_label("hd", hd2=2.3, hd3=0.1) == "HD2=2.3%  HD3=0.1%"
        assert format_label("pa_limit", 12.0, limit=12.5) == "12.0/12.5 W"
        assert (format_label("q_point", ug1=-1.5, ua=250.0, ia=10.0)
                == "Q: Ug1=-1.5V  Ua=250V  Ia=10.0mA")

    def test_unknown_kind_falls_back_to_value(self):
        assert format_label("nonexistent", 42.0) == "42.0"

    def test_value_only_precision_pinned(self):
        """Pin the precision of representative value-only formats so a
        regression (e.g. .2f -> .0f) is caught, not just non-emptiness."""
        assert format_label("ua_value", 250.0) == "250"     # .0f
        assert format_label("ug1_value", -1.5) == "-1.50"   # .2f
        assert format_label("ia_value", 12.5) == "12.50"    # .2f
        assert format_label("ih_value", 0.3) == "0.300"     # .3f
        assert format_label("mu", 12.3) == "12.3"           # .1f
        assert format_label("pct", 42.0) == "42%"           # .0f

    def test_every_value_only_format_renders(self):
        """Every format whose only placeholder is {value} must render a
        non-empty string for a plain float (catches a malformed format string)."""
        checked = 0
        for kind, fmt in LABEL_FORMATS.items():
            fields = {fn for _, fn, _, _ in string.Formatter().parse(fmt) if fn}
            if fields <= {"value"}:
                out = format_label(kind, 1.5)
                assert isinstance(out, str) and out, f"{kind} rendered empty"
                checked += 1
        assert checked >= 1, "no value-only formats exercised (de-vacuated)"
