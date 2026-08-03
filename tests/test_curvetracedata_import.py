"""Unit tests for lm19.curvetracedata_import module.

Run:  py -m pytest tests/test_curvetracedata_import.py -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.curvetracedata_import import (
    parse_curvetracedata_dat,
    dat_to_lm19_points,
    guess_meta_from_dat_filename,
)


SAMPLE_DAT = """\
% * Sample: PCC88_5_Vh=6.3V
% * Date / time: 2025-01-29 22:34:13.446470
% Column 1:  PSU1 nominal voltage setting (V)
% Column 2:  PSU1 nominal current setting (A)
% Column 3:  PSU1 voltage measurement (V)
% Column 4:  PSU1 current measurement (I)
% Column 5:  PSU1 limiter flag
% Column 6:  PSU2 nominal voltage setting (V)
% Column 7:  PSU2 nominal current setting (A)
% Column 8:  PSU2 voltage measurement (V)
% Column 9:  PSU2 current measurement (I)
% Column 10: PSU2 limiter flag
0.00 0.02500 0.1 0.00007 0 -0.000 -1.000 -0.153 -0.000 0 NA
5.00 0.02500 5.1 0.00063 0 -0.000 -1.000 -0.130 -0.000 0 NA
10.00 0.02500 10.1 0.00158 0 -0.000 -1.000 -0.101 -0.000 0 NA
65.00 0.02500 59.3 0.02001 1 -0.000 -1.000 -0.000 -0.000 0 NA
"""


def _write_tmp(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".dat")
    os.write(fd, content.encode("utf-8"))
    os.close(fd)
    return path


class TestParseCurveTraceData(unittest.TestCase):
    def test_parse_dat(self):
        path = _write_tmp(SAMPLE_DAT)
        try:
            result = parse_curvetracedata_dat(path)
        finally:
            os.unlink(path)

        self.assertEqual(result["sample_name"], "PCC88_5_Vh=6.3V")
        self.assertTrue(result["date_str"].startswith("2025-01-29"))
        # limiter=1 row should be skipped
        self.assertEqual(len(result["points"]), 3)
        self.assertAlmostEqual(result["points"][0]["ua"], 0.1)
        self.assertAlmostEqual(result["points"][1]["ua"], 5.1)
        self.assertAlmostEqual(result["points"][2]["ug1"], -0.101)

    def test_parse_empty_raises(self):
        path = _write_tmp("% no data\n")
        try:
            with self.assertRaises(ValueError):
                parse_curvetracedata_dat(path)
        finally:
            os.unlink(path)

    def test_duplicate_ua_ug1_rows_are_deduplicated(self):
        content = """\
% * Sample: EL34_1
10.00 0.02500 10.1 0.00158 0 -0.000 -1.000 -1.500 -0.000 0 NA
10.00 0.02500 10.1 0.00160 0 -0.000 -1.000 -1.500 -0.000 0 NA
"""
        path = _write_tmp(content)
        try:
            result = parse_curvetracedata_dat(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(result["points"]), 1)


class TestDatToLm19Points(unittest.TestCase):
    def test_convert_points(self):
        parsed = {
            "points": [
                {"ua": 100.2, "ug1": -1.50, "ia_A": 0.01234},
                {"ua": 120.2, "ug1": -1.00, "ia_A": 0.01500},
            ]
        }
        points = dat_to_lm19_points(parsed, vs=250.0, vh=6.3)
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0]["ua"], 100.2)
        self.assertAlmostEqual(points[0]["ug1"], -1.5)
        self.assertAlmostEqual(points[0]["ia"], 12.34)  # A -> mA
        self.assertAlmostEqual(points[0]["ug2"], 250.0)
        self.assertAlmostEqual(points[0]["uh"], 6.3)
        self.assertAlmostEqual(points[0]["ig2"], 0.0)
        self.assertAlmostEqual(points[0]["ih"], 0.0)


class TestGuessMeta(unittest.TestCase):
    def test_guess_from_data_folder(self):
        meta = guess_meta_from_dat_filename(
            "D:/raw/curvetracedata/PCC88/data/PCC88_5_Vh=6.3V.dat",
            sample_name="PCC88_5_Vh=6.3V",
        )
        self.assertEqual(meta["tube_type"], "PCC88")
        self.assertEqual(meta["lamp_id"], "PCC88_5_Vh=6.3V")
        self.assertEqual(meta["name"], "PCC88_5_Vh=6.3V")

    def test_guess_from_stem(self):
        meta = guess_meta_from_dat_filename("EL34_12.dat")
        self.assertEqual(meta["tube_type"], "EL34")
        self.assertEqual(meta["lamp_id"], "EL34_12")

    def test_guess_multi_part_tube_from_stem(self):
        meta = guess_meta_from_dat_filename("801_VT62_12.dat")
        self.assertEqual(meta["tube_type"], "801_VT62")


if __name__ == "__main__":
    unittest.main()
