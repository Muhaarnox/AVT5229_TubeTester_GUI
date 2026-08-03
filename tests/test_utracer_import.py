"""Unit tests for lm19.utracer_import module.

Run:  py -m pytest tests/test_utracer_import.py -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.utracer_import import parse_utd, utd_to_lm19_points, guess_meta_from_filename


# ======================================================================
# Sample .utd content strings
# ======================================================================

TRANSFER_SINGLE_VA = """\
Vg (V) Ia (mA) 
 Va = 325 V 
-50 4.626 
-46 5.495 
-42 6.669 
-38 8.23 
-34 10.291 
-30 12.992 
"""

TRANSFER_MULTI_VA = """\
Vg (V) Ia (mA) 
 Va = 100 V  Va = 200 V  Va = 300 V
-10 0.12  0.45  0.89
-8  0.23  0.67  1.23
-6  0.45  1.12  2.01
"""

OUTPUT_TRIODE = """\
Va (V) Ia (mA) 
 Vg = -4 V  Vg = -3 V  Vg = -2 V  Vg = -1 V  Vg = 0 V
5.2 0.00 0.01 0.05 0.15 0.35
25.4 0.08 0.22 0.52 1.05 1.82
50.3 0.25 0.55 1.12 2.05 3.25
"""

OUTPUT_PENTODE = """\
Va (V) Ia (mA) Is (mA) 
 Vg = -4 V  Vg = -4 V  Vg = -3 V  Vg = -3 V  Vg = -2 V  Vg = -2 V
5.0 0.05 0.02 0.12 0.04 0.25 0.08
25.0 0.45 0.08 0.89 0.15 1.56 0.25
50.0 0.82 0.12 1.55 0.22 2.65 0.38
"""


def _write_tmp(content: str) -> str:
    """Write content to a temporary .utd file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".utd")
    os.write(fd, content.encode("utf-8"))
    os.close(fd)
    return path


class TestParseUtdTransfer(unittest.TestCase):
    """Parsing transfer curves (Vg on X)."""

    def test_single_va(self):
        path = _write_tmp(TRANSFER_SINGLE_VA)
        try:
            result = parse_utd(path)
        finally:
            os.unlink(path)

        self.assertEqual(result["format"], "transfer")
        self.assertEqual(result["x_name"], "Vg")
        self.assertEqual(result["step_name"], "Va")
        self.assertFalse(result["has_is"])
        self.assertIsNone(result["is_matrix"])
        self.assertEqual(result["step_values"], [325.0])
        self.assertEqual(len(result["x_values"]), 6)
        self.assertAlmostEqual(result["x_values"][0], -50.0)
        self.assertAlmostEqual(result["ia_matrix"][0][0], 4.626)
        self.assertAlmostEqual(result["ia_matrix"][5][0], 12.992)

    def test_multi_va(self):
        path = _write_tmp(TRANSFER_MULTI_VA)
        try:
            result = parse_utd(path)
        finally:
            os.unlink(path)

        self.assertEqual(result["format"], "transfer")
        self.assertEqual(result["step_values"], [100.0, 200.0, 300.0])
        self.assertEqual(len(result["x_values"]), 3)
        # Row 0 (Vg=-10): Ia at Va=100 -> 0.12, Va=200 -> 0.45, Va=300 -> 0.89
        self.assertAlmostEqual(result["ia_matrix"][0][0], 0.12)
        self.assertAlmostEqual(result["ia_matrix"][0][1], 0.45)
        self.assertAlmostEqual(result["ia_matrix"][0][2], 0.89)


class TestParseUtdOutput(unittest.TestCase):
    """Parsing output curves (Va on X), triode (Ia only)."""

    def test_triode(self):
        path = _write_tmp(OUTPUT_TRIODE)
        try:
            result = parse_utd(path)
        finally:
            os.unlink(path)

        self.assertEqual(result["format"], "output")
        self.assertEqual(result["x_name"], "Va")
        self.assertEqual(result["step_name"], "Vg")
        self.assertFalse(result["has_is"])
        self.assertEqual(result["step_values"], [-4.0, -3.0, -2.0, -1.0, 0.0])
        self.assertEqual(len(result["x_values"]), 3)
        self.assertAlmostEqual(result["x_values"][0], 5.2)
        # Row 0 (Va=5.2): Ia at Vg=-4 -> 0.00, Vg=0 -> 0.35
        self.assertAlmostEqual(result["ia_matrix"][0][0], 0.0)
        self.assertAlmostEqual(result["ia_matrix"][0][4], 0.35)


class TestParseUtdPentode(unittest.TestCase):
    """Parsing output curves with Ia + Is (pentode)."""

    def test_pentode(self):
        path = _write_tmp(OUTPUT_PENTODE)
        try:
            result = parse_utd(path)
        finally:
            os.unlink(path)

        self.assertEqual(result["format"], "output")
        self.assertTrue(result["has_is"])
        self.assertEqual(result["step_values"], [-4.0, -3.0, -2.0])
        self.assertIsNotNone(result["is_matrix"])
        # Row 0 (Va=5.0): Ia at Vg=-4 -> 0.05, Is at Vg=-4 -> 0.02
        self.assertAlmostEqual(result["ia_matrix"][0][0], 0.05)
        self.assertAlmostEqual(result["is_matrix"][0][0], 0.02)
        # Row 2 (Va=50.0): Ia at Vg=-2 -> 2.65, Is at Vg=-2 -> 0.38
        self.assertAlmostEqual(result["ia_matrix"][2][2], 2.65)
        self.assertAlmostEqual(result["is_matrix"][2][2], 0.38)


class TestUtdToLm19Points(unittest.TestCase):
    """Conversion of parsed data to LM19 point format."""

    def test_transfer_to_points(self):
        path = _write_tmp(TRANSFER_SINGLE_VA)
        try:
            parsed = parse_utd(path)
        finally:
            os.unlink(path)

        points = utd_to_lm19_points(parsed, vs=250.0, vh=6.3)
        self.assertEqual(len(points), 6)  # 6 Vg × 1 Va

        p0 = points[0]
        self.assertAlmostEqual(p0["ua"], 325.0)  # Va is stepping
        self.assertAlmostEqual(p0["ug1"], -50.0)  # Vg is running
        self.assertAlmostEqual(p0["ug2"], 250.0)  # user-supplied Vs
        self.assertAlmostEqual(p0["ia"], 4.626)
        self.assertAlmostEqual(p0["ig2"], 0.0)
        self.assertAlmostEqual(p0["uh"], 6.3)
        self.assertAlmostEqual(p0["ih"], 0.0)

    def test_output_to_points(self):
        path = _write_tmp(OUTPUT_TRIODE)
        try:
            parsed = parse_utd(path)
        finally:
            os.unlink(path)

        points = utd_to_lm19_points(parsed, vs=0.0, vh=12.6)
        # 3 Va values × 5 Vg values = 15 points
        self.assertEqual(len(points), 15)

        p0 = points[0]
        self.assertAlmostEqual(p0["ua"], 5.2)    # Va is running
        self.assertAlmostEqual(p0["ug1"], -4.0)  # Vg is stepping
        self.assertAlmostEqual(p0["uh"], 12.6)

    def test_pentode_ig2(self):
        path = _write_tmp(OUTPUT_PENTODE)
        try:
            parsed = parse_utd(path)
        finally:
            os.unlink(path)

        points = utd_to_lm19_points(parsed, vs=200.0, vh=6.3)
        # 3 Va × 3 Vg = 9 points
        self.assertEqual(len(points), 9)
        # First point: Vg=-4, Va=5.0
        self.assertAlmostEqual(points[0]["ig2"], 0.02)
        self.assertAlmostEqual(points[0]["ug2"], 200.0)


class TestGuessMetaFromFilename(unittest.TestCase):
    """Extraction of metadata from filename."""

    def test_tube_with_vs(self):
        meta = guess_meta_from_filename("EL84_250.utd")
        self.assertEqual(meta["tube_type"], "EL84")
        self.assertAlmostEqual(meta["vs"], 250.0)
        self.assertEqual(meta["lamp_id"], "EL84_250")

    def test_tube_no_vs(self):
        meta = guess_meta_from_filename("ECC81.utd")
        self.assertEqual(meta["tube_type"], "ECC81")
        self.assertNotIn("vs", meta)

    def test_full_path(self):
        meta = guess_meta_from_filename("/data/tubes/pf86_200.utd")
        self.assertEqual(meta["tube_type"], "pf86")
        self.assertAlmostEqual(meta["vs"], 200.0)

    def test_numeric_only_name(self):
        meta = guess_meta_from_filename("42.utd")
        self.assertEqual(meta["tube_type"], "42")
        self.assertEqual(meta["lamp_id"], "42")


class TestEdgeCases(unittest.TestCase):
    """Edge cases and error handling."""

    def test_empty_file(self):
        path = _write_tmp("")
        try:
            with self.assertRaises(ValueError):
                parse_utd(path)
        finally:
            os.unlink(path)

    def test_too_short(self):
        path = _write_tmp("Va (V) Ia (mA)\n")
        try:
            with self.assertRaises(ValueError):
                parse_utd(path)
        finally:
            os.unlink(path)

    def test_trailing_spaces(self):
        content = "Vg (V) Ia (mA)   \n Va = 100 V   \n-5 1.23   \n-3 2.45   \n"
        path = _write_tmp(content)
        try:
            result = parse_utd(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(result["x_values"]), 2)

    def test_empty_trailing_lines(self):
        content = TRANSFER_SINGLE_VA + "\n\n\n"
        path = _write_tmp(content)
        try:
            result = parse_utd(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(result["x_values"]), 6)


if __name__ == "__main__":
    unittest.main()
