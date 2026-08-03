"""Unit tests for CSV export functions (format_csv, format_matrix, format_multi_csv, write_csv).

Run:  py -m pytest tests/test_csv_export_full.py -v
  or: py tests/test_csv_export_full.py
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.csv_export import format_csv, format_matrix, format_multi_csv, write_csv


class TestCsvExport(unittest.TestCase):
    """Test CSV export functions."""

    SAMPLE_POINTS = [
        {"ua": 100.0, "ug1": -1.0, "ug2": 170.0, "ia": 5.0,
         "ig2": 1.0, "uh": 6.3, "ih": 0.3},
        {"ua": 200.0, "ug1": -1.0, "ug2": 170.0, "ia": 12.0,
         "ig2": 2.0, "uh": 6.3, "ih": 0.3},
        {"ua": 100.0, "ug1": -2.0, "ug2": 170.0, "ia": 3.0,
         "ig2": 0.5, "uh": 6.3, "ih": 0.3},
        {"ua": 200.0, "ug1": -2.0, "ug2": 170.0, "ia": 8.0,
         "ig2": 1.2, "uh": 6.3, "ih": 0.3},
    ]

    def test_flat_csv_header(self):
        """Flat CSV should contain header comments and column names."""
        csv = format_csv(self.SAMPLE_POINTS, tube_type="6P14P",
                         lamp_id="L1", separator=";")
        self.assertIn("# LM19 Tube Tester Export", csv)
        self.assertIn("# Tube: 6P14P", csv)
        self.assertIn("Ua;Ug1;Ug2;Ia;Ig2;Uh;Ih;Pa;Pg2;Ik", csv)

    def test_flat_csv_row_count(self):
        """Should have one data row per point."""
        csv = format_csv(self.SAMPLE_POINTS, separator=";")
        lines = [l for l in csv.strip().split("\n")
                 if not l.startswith("#")]
        # First line is header, rest are data
        self.assertEqual(len(lines), 1 + len(self.SAMPLE_POINTS))

    def test_flat_csv_computed_columns(self):
        """Pa, Pg2, Ik should be computed correctly."""
        csv = format_csv(self.SAMPLE_POINTS, separator=";",
                         include_computed=True)
        # First data point: Ua=100, Ia=5 → Pa=100*5/1000=0.5W
        lines = [l for l in csv.strip().split("\n")
                 if not l.startswith("#")]
        data_line = lines[1]  # first data row
        cols = data_line.split(";")
        pa = float(cols[7])   # Pa column
        self.assertAlmostEqual(pa, 0.500, places=3)

    def test_flat_csv_no_computed(self):
        """Without computed columns, should have 7 columns."""
        csv = format_csv(self.SAMPLE_POINTS, separator=",",
                         include_computed=False)
        lines = [l for l in csv.strip().split("\n")
                 if not l.startswith("#")]
        header_cols = lines[0].split(",")
        self.assertEqual(len(header_cols), 7)
        self.assertNotIn("Pa", lines[0])

    def test_flat_csv_srk_in_header(self):
        """S/R/K values should appear in comments."""
        srk = {"s": 11.2, "r": 15.3, "k": 0.73}
        csv = format_csv(self.SAMPLE_POINTS, srk=srk)
        self.assertIn("S: 11.20 mA/V", csv)
        self.assertIn("R: 15.30 kOhm", csv)

    def test_comma_separator(self):
        """Comma separator should work."""
        csv = format_csv(self.SAMPLE_POINTS, separator=",")
        lines = [l for l in csv.strip().split("\n")
                 if not l.startswith("#")]
        self.assertIn(",", lines[0])
        self.assertNotIn(";", lines[0])

    def test_tab_separator(self):
        """Tab separator should work."""
        csv = format_csv(self.SAMPLE_POINTS, separator="\t")
        lines = [l for l in csv.strip().split("\n")
                 if not l.startswith("#")]
        self.assertIn("\t", lines[0])

    def test_matrix_format(self):
        """Matrix should have Ua columns and Ug1 rows."""
        csv = format_matrix(self.SAMPLE_POINTS, separator=";",
                            parameter="Ia")
        self.assertIn("Ua:", csv)
        self.assertIn("Ug1=", csv)
        self.assertIn("Ug2=170.0V", csv)

    def test_matrix_correct_values(self):
        """Matrix cells should contain correct Ia values."""
        csv = format_matrix(self.SAMPLE_POINTS, separator=";",
                            parameter="Ia")
        # Point: Ua=200, Ug1=-1 → Ia=12.0
        self.assertIn("12.000", csv)
        # Point: Ua=100, Ug1=-2 → Ia=3.0
        self.assertIn("3.000", csv)

    def test_matrix_empty_points(self):
        """Empty points should return empty string."""
        csv = format_matrix([], parameter="Ia")
        self.assertEqual(csv, "")

    def test_matrix_pa_parameter(self):
        """Matrix with Pa parameter should compute Ua*Ia/1000."""
        csv = format_matrix(self.SAMPLE_POINTS, separator=";",
                            parameter="Pa")
        # Ua=200, Ug1=-1, Ia=12 → Pa=200*12/1000=2.4W
        self.assertIn("2.400", csv)

    def test_multi_csv(self):
        """Multi-measurement CSV should have Series column."""
        entries = [
            {"lamp_type": "6P14P", "lamp_id": "L1", "name": "Test1",
             "points": self.SAMPLE_POINTS[:2]},
            {"lamp_type": "6P14P", "lamp_id": "L2", "name": "Test2",
             "points": self.SAMPLE_POINTS[2:]},
        ]
        csv = format_multi_csv(entries, separator=";")
        self.assertIn("Series;Lamp_Type;Lamp_ID;Mfg", csv)
        self.assertIn("Test1", csv)
        self.assertIn("Test2", csv)

    def test_mfg_date_in_flat_header(self):
        """mfg_date kwarg should add a Manufactured comment line."""
        csv = format_csv(self.SAMPLE_POINTS, tube_type="6P14P", lamp_id="L1",
                         mfg_date="1972-05", separator=";")
        self.assertIn("# Manufactured: 1972-05", csv)

    def test_mfg_date_omitted_no_header_line(self):
        """No mfg_date -> no Manufactured comment, no blank line."""
        csv = format_csv(self.SAMPLE_POINTS, tube_type="6P14P", lamp_id="L1",
                         separator=";")
        self.assertNotIn("Manufactured", csv)

    def test_mfg_date_in_matrix_header(self):
        csv = format_matrix(self.SAMPLE_POINTS, tube_type="6P14P", lamp_id="L1",
                            mfg_date="1965-11", separator=";")
        self.assertIn("# Manufactured: 1965-11", csv)

    def test_multi_csv_mfg_column_always_present(self):
        """Mfg column appears in multi-CSV header regardless of data presence."""
        entries = [
            {"lamp_type": "T1", "lamp_id": "L1", "name": "A",
             "points": self.SAMPLE_POINTS[:1]},
            {"lamp_type": "T1", "lamp_id": "L2", "name": "B",
             "mfg_date": "1980-03", "points": self.SAMPLE_POINTS[1:2]},
        ]
        csv = format_multi_csv(entries, separator=";")
        lines = csv.strip().split("\n")
        header = next(l for l in lines if l.startswith("Series"))
        self.assertIn("Mfg", header.split(";"))

    def test_multi_csv_mfg_value_from_entry_root(self):
        entries = [
            {"lamp_type": "T1", "lamp_id": "L1", "name": "A",
             "mfg_date": "1972-05", "points": self.SAMPLE_POINTS[:1]},
        ]
        csv = format_multi_csv(entries, separator=";")
        data_rows = [l for l in csv.strip().split("\n")
                     if not l.startswith("#") and not l.startswith("Series")]
        self.assertTrue(any("1972-05" in r for r in data_rows))

    def test_multi_csv_mfg_value_from_entry_data(self):
        """Should also pick mfg_date from entry['data'] (compare flow)."""
        entries = [
            {"lamp_type": "T1", "lamp_id": "L1", "name": "A",
             "data": {"mfg_date": "1968-09"},
             "points": self.SAMPLE_POINTS[:1]},
        ]
        csv = format_multi_csv(entries, separator=";")
        self.assertIn("1968-09", csv)

    def test_multi_csv_empty_mfg_cell(self):
        """Entry without mfg_date should produce empty Mfg cell (not crash)."""
        entries = [
            {"lamp_type": "T1", "lamp_id": "L1", "name": "A",
             "points": self.SAMPLE_POINTS[:1]},
        ]
        csv = format_multi_csv(entries, separator=";")
        data_rows = [l for l in csv.strip().split("\n")
                     if not l.startswith("#") and not l.startswith("Series")]
        # Series;Lamp_Type;Lamp_ID;Mfg;... — Mfg cell is the 4th, empty between ;;
        self.assertTrue(any(";;" in r for r in data_rows))

    def test_multi_csv_row_count(self):
        """Total data rows should equal sum of all points."""
        entries = [
            {"lamp_type": "A", "lamp_id": "1", "name": "M1",
             "points": self.SAMPLE_POINTS[:2]},
            {"lamp_type": "A", "lamp_id": "2", "name": "M2",
             "points": self.SAMPLE_POINTS[2:]},
        ]
        csv = format_multi_csv(entries, separator=";")
        lines = [l for l in csv.strip().split("\n")
                 if not l.startswith("#")]
        self.assertEqual(len(lines), 1 + 4)  # header + 4 data rows

    def test_write_csv_file(self):
        """write_csv should create a file with correct content."""
        content = "header\ndata\n"
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            write_csv(path, content)
            with open(path, encoding="utf-8") as f:
                result = f.read()
            self.assertEqual(result, content)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
