"""Unit tests for lm19.csv_import module.

Run:  py -m pytest tests/test_csv_import.py -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.csv_import import (
    detect_separator,
    detect_columns,
    parse_csv,
    skip_comment_lines,
)


# ======================================================================
# Sample CSV content
# ======================================================================

CSV_LM19_FLAT = """\
# LM19 Tube Tester Export
# Tube: 6P14P  ID: L1
# Points: 3
#
Ua;Ug1;Ug2;Ia;Ig2;Uh;Ih
130.0;-19.00;130.0;4.41;0.04;6.30;0.78
150.0;-19.00;150.0;6.12;0.08;6.30;0.78
170.0;-19.00;170.0;8.55;0.12;6.30;0.78
"""

CSV_COMMA = """\
Ua,Ug1,Ug2,Ia,Ig2,Uh,Ih
130.0,-19.00,130.0,4.41,0.04,6.30,0.78
150.0,-19.00,150.0,6.12,0.08,6.30,0.78
"""

CSV_TAB = """\
Ua\tUg1\tUg2\tIa\tIg2\tUh\tIh
130.0\t-19.00\t130.0\t4.41\t0.04\t6.30\t0.78
"""

CSV_UTRACER_STYLE = """\
Va,Vg,Ia,Is
100.0,-10.0,0.12,0.01
200.0,-10.0,0.45,0.03
300.0,-10.0,0.89,0.05
100.0,-8.0,0.23,0.02
200.0,-8.0,0.67,0.04
"""

CSV_PARTIAL_HEADERS = """\
Voltage;Current;Something
100.0;0.12;foo
200.0;0.45;bar
"""


class TestDetectSeparator(unittest.TestCase):
    """Auto-detection of CSV separator."""

    def test_semicolon(self):
        self.assertEqual(detect_separator(CSV_LM19_FLAT), ";")

    def test_comma(self):
        self.assertEqual(detect_separator(CSV_COMMA), ",")

    def test_tab(self):
        self.assertEqual(detect_separator(CSV_TAB), "\t")

    def test_empty(self):
        # Fallback to comma
        result = detect_separator("")
        self.assertEqual(result, ",")


class TestDetectColumns(unittest.TestCase):
    """Auto-mapping of column headers to LM19 keys."""

    def test_lm19_native(self):
        headers = ["Ua", "Ug1", "Ug2", "Ia", "Ig2", "Uh", "Ih"]
        mapping = detect_columns(headers)
        self.assertEqual(mapping, {
            0: "ua", 1: "ug1", 2: "ug2",
            3: "ia", 4: "ig2", 5: "uh", 6: "ih",
        })

    def test_utracer_names(self):
        headers = ["Va", "Vg", "Ia", "Is"]
        mapping = detect_columns(headers)
        self.assertEqual(mapping, {
            0: "ua", 1: "ug1", 2: "ia", 3: "ig2",
        })

    def test_with_units(self):
        headers = ["Va (V)", "Ia (mA)", "Is (mA)"]
        mapping = detect_columns(headers)
        self.assertEqual(mapping, {0: "ua", 1: "ia", 2: "ig2"})

    def test_unknown_headers(self):
        headers = ["Voltage", "Current", "Something"]
        mapping = detect_columns(headers)
        self.assertEqual(mapping, {})

    def test_case_insensitive(self):
        headers = ["ua", "UG1", "Ia"]
        mapping = detect_columns(headers)
        self.assertIn(0, mapping)
        self.assertIn(1, mapping)
        self.assertIn(2, mapping)


class TestSkipComments(unittest.TestCase):
    """Comment and empty line filtering."""

    def test_comments_removed(self):
        lines = [
            "# comment",
            "# another",
            "",
            "data line 1",
            "data line 2",
        ]
        result = skip_comment_lines(lines)
        self.assertEqual(result, ["data line 1", "data line 2"])

    def test_no_comments(self):
        lines = ["a", "b", "c"]
        result = skip_comment_lines(lines)
        self.assertEqual(result, ["a", "b", "c"])


class TestParseCsvFlat(unittest.TestCase):
    """Parsing LM19 flat CSV export."""

    def test_lm19_flat(self):
        mapping = {0: "ua", 1: "ug1", 2: "ug2", 3: "ia", 4: "ig2", 5: "uh", 6: "ih"}
        points = parse_csv(CSV_LM19_FLAT, mapping, ";")
        self.assertEqual(len(points), 3)

        p0 = points[0]
        self.assertAlmostEqual(p0["ua"], 130.0)
        self.assertAlmostEqual(p0["ug1"], -19.0)
        self.assertAlmostEqual(p0["ug2"], 130.0)
        self.assertAlmostEqual(p0["ia"], 4.41)
        self.assertAlmostEqual(p0["ig2"], 0.04)
        self.assertAlmostEqual(p0["uh"], 6.3)
        self.assertAlmostEqual(p0["ih"], 0.78)


class TestParseCsvUtracer(unittest.TestCase):
    """Parsing CSV exported from uTracer via Excel."""

    def test_utracer_csv(self):
        mapping = {0: "ua", 1: "ug1", 2: "ia", 3: "ig2"}
        points = parse_csv(CSV_UTRACER_STYLE, mapping, ",")
        self.assertEqual(len(points), 5)

        p0 = points[0]
        self.assertAlmostEqual(p0["ua"], 100.0)
        self.assertAlmostEqual(p0["ug1"], -10.0)
        self.assertAlmostEqual(p0["ia"], 0.12)
        self.assertAlmostEqual(p0["ig2"], 0.01)
        # Unmapped fields default to 0
        self.assertAlmostEqual(p0["ug2"], 0.0)
        self.assertAlmostEqual(p0["uh"], 0.0)
        self.assertAlmostEqual(p0["ih"], 0.0)


class TestPartialMapping(unittest.TestCase):
    """Handling of partially recognized column names."""

    def test_no_recognized_columns(self):
        mapping = {}  # nothing mapped
        points = parse_csv(CSV_PARTIAL_HEADERS, mapping, ";")
        self.assertEqual(len(points), 0)  # no valid points

    def test_single_column_mapped(self):
        # Only map column 1 as ia
        mapping = {1: "ia"}
        points = parse_csv(CSV_PARTIAL_HEADERS, mapping, ";")
        # Should parse 2 data rows with ia filled
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0]["ia"], 0.12)
        self.assertAlmostEqual(points[0]["ua"], 0.0)  # not mapped


class TestParseCsvBadCell(unittest.TestCase):
    """A mapped, non-empty cell that won't parse must drop the whole row and
    surface a warning — never leave a silent 0.0 (failure-visibility)."""

    _CSV = "Ua;Ug1;Ia\n130.0;-19.0;4.41\nabc;-19.0;6.12\n"
    _MAP = {0: "ua", 1: "ug1", 2: "ia"}

    def test_bad_mapped_cell_dropped(self):
        points = parse_csv(self._CSV, self._MAP, ";")
        # The 'abc' ua row is dropped; only the clean row survives.
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0]["ua"], 130.0)
        # No surviving point carries a silent 0.0 in the mapped ua slot.
        self.assertTrue(all(p["ua"] != 0.0 for p in points))

    def test_bad_cell_logs_warning(self):
        with self.assertLogs("lm19.csv_import", level="WARNING") as cm:
            parse_csv(self._CSV, self._MAP, ";")
        self.assertTrue(any("skipped" in m for m in cm.output))

    def test_empty_cell_is_not_bad(self):
        # An empty mapped cell is an absent column, not corruption — no drop.
        text = "Ua;Ug1;Ia\n130.0;;4.41\n"
        points = parse_csv(text, self._MAP, ";")
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0]["ua"], 130.0)


class TestParseCsvComma(unittest.TestCase):
    """Comma-separated variant."""

    def test_comma_csv(self):
        mapping = {0: "ua", 1: "ug1", 2: "ug2", 3: "ia", 4: "ig2", 5: "uh", 6: "ih"}
        points = parse_csv(CSV_COMMA, mapping, ",")
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0]["ua"], 130.0)


class TestParseCsvTab(unittest.TestCase):
    """Tab-separated variant."""

    def test_tab_csv(self):
        mapping = {0: "ua", 1: "ug1", 2: "ug2", 3: "ia", 4: "ig2", 5: "uh", 6: "ih"}
        points = parse_csv(CSV_TAB, mapping, "\t")
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0]["ua"], 130.0)


class TestParseCsvCommaDecimal(unittest.TestCase):
    """European locales (de_DE, fr_FR, ru_RU, …) export CSVs with comma
    decimals (``12,5`` instead of ``12.5``). Without ``replace(',', '.')``
    in the parser, ``float('12,5')`` raises ``ValueError`` → field
    silently skipped → mostly-empty point dicts.

    Auto-detect of the field separator is a separate concern (covered
    by ``TestDetectSeparator``); these tests assume the user manually
    selected the correct separator via the import dialog.
    """

    def test_regional_semicolon_separator_with_comma_decimal(self):
        """Regional format: ``;`` field separator + ``,`` decimal."""
        text = (
            "Ua;Ug1;Ug2;Ia;Ig2;Uh;Ih\n"
            "130,5;-19,00;130,0;4,41;0,04;6,30;0,78\n"
            "150,0;-19,00;150,0;6,12;0,08;6,30;0,78\n"
        )
        mapping = {0: "ua", 1: "ug1", 2: "ug2", 3: "ia", 4: "ig2", 5: "uh", 6: "ih"}
        points = parse_csv(text, mapping, ";")
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0]["ua"], 130.5)
        self.assertAlmostEqual(points[0]["ug1"], -19.0)
        self.assertAlmostEqual(points[0]["ia"], 4.41)
        self.assertAlmostEqual(points[1]["ua"], 150.0)
        self.assertAlmostEqual(points[1]["ia"], 6.12)

    def test_tab_separator_with_comma_decimal(self):
        """Some Excel exports use tab + comma decimal — also valid."""
        text = "Ua\tUg1\tIa\n130,5\t-2,5\t4,41\n"
        mapping = {0: "ua", 1: "ug1", 2: "ia"}
        points = parse_csv(text, mapping, "\t")
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0]["ua"], 130.5)
        self.assertAlmostEqual(points[0]["ug1"], -2.5)
        self.assertAlmostEqual(points[0]["ia"], 4.41)

    def test_dot_decimal_unchanged(self):
        """Regression: English '.'-decimal CSV still parses identically.
        replace(',', '.') is a no-op when no comma is present in field."""
        text = (
            "Ua,Ug1,Ia\n"
            "130.5,-19.0,4.41\n"
            "150.0,-19.0,6.12\n"
        )
        mapping = {0: "ua", 1: "ug1", 2: "ia"}
        points = parse_csv(text, mapping, ",")
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0]["ua"], 130.5)
        self.assertAlmostEqual(points[1]["ia"], 6.12)


if __name__ == "__main__":
    unittest.main()
