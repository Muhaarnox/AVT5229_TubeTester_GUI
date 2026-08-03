"""Unit tests for lm19.utracer_export module.

Run:  py -m pytest tests/test_utracer_export.py -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.utracer_export import (
    format_utd, detect_best_format, suggest_filename,
    _fmt_voltage, _fmt_current,
)
from lm19.utracer_import import parse_utd, utd_to_lm19_points


# ======================================================================
# Helper: build LM19 points from simple grid
# ======================================================================

def _make_output_triode_points():
    """3 Va × 5 Vg triode output data (no Ig2)."""
    points = []
    va_vals = [5.2, 25.4, 50.3]
    vg_vals = [-4.0, -3.0, -2.0, -1.0, 0.0]
    ia_data = [
        [0.00, 0.01, 0.05, 0.15, 0.35],
        [0.08, 0.22, 0.52, 1.05, 1.82],
        [0.25, 0.55, 1.12, 2.05, 3.25],
    ]
    for vi, va in enumerate(va_vals):
        for gi, vg in enumerate(vg_vals):
            points.append({
                "ua": va, "ug1": vg, "ug2": 0.0,
                "ia": ia_data[vi][gi], "ig2": 0.0,
                "uh": 12.6, "ih": 0.0,
            })
    return points


def _make_output_pentode_points():
    """3 Va × 3 Vg pentode output data (with Ig2)."""
    points = []
    va_vals = [5.0, 25.0, 50.0]
    vg_vals = [-4.0, -3.0, -2.0]
    ia_data = [
        [0.05, 0.12, 0.25],
        [0.45, 0.89, 1.56],
        [0.82, 1.55, 2.65],
    ]
    is_data = [
        [0.02, 0.04, 0.08],
        [0.08, 0.15, 0.25],
        [0.12, 0.22, 0.38],
    ]
    for vi, va in enumerate(va_vals):
        for gi, vg in enumerate(vg_vals):
            points.append({
                "ua": va, "ug1": vg, "ug2": 200.0,
                "ia": ia_data[vi][gi], "ig2": is_data[vi][gi],
                "uh": 6.3, "ih": 0.0,
            })
    return points


def _make_transfer_points():
    """6 Vg × 1 Va transfer data."""
    points = []
    vg_vals = [-50, -46, -42, -38, -34, -30]
    ia_vals = [4.626, 5.495, 6.669, 8.23, 10.291, 12.992]
    for vg, ia in zip(vg_vals, ia_vals):
        points.append({
            "ua": 325.0, "ug1": float(vg), "ug2": 250.0,
            "ia": ia, "ig2": 0.0,
            "uh": 6.3, "ih": 0.0,
        })
    return points


def _make_transfer_multi_va_points():
    """3 Vg × 3 Va transfer data."""
    points = []
    vg_vals = [-10.0, -8.0, -6.0]
    va_vals = [100.0, 200.0, 300.0]
    ia_data = [
        [0.12, 0.45, 0.89],
        [0.23, 0.67, 1.23],
        [0.45, 1.12, 2.01],
    ]
    for gi, vg in enumerate(vg_vals):
        for vi, va in enumerate(va_vals):
            points.append({
                "ua": va, "ug1": vg, "ug2": 0.0,
                "ia": ia_data[gi][vi], "ig2": 0.0,
                "uh": 6.3, "ih": 0.0,
            })
    return points


# ======================================================================
# Tests: format detection
# ======================================================================

class TestDetectFormat(unittest.TestCase):

    def test_output_more_ua(self):
        pts = _make_output_triode_points()  # 3 Va, 5 Vg
        self.assertEqual(detect_best_format(pts), "transfer")

    def test_transfer_more_ug1(self):
        pts = _make_transfer_points()  # 1 Va, 6 Vg
        self.assertEqual(detect_best_format(pts), "transfer")

    def test_equal_counts(self):
        pts = _make_output_pentode_points()  # 3 Va, 3 Vg
        self.assertEqual(detect_best_format(pts), "output")

    def test_empty(self):
        self.assertEqual(detect_best_format([]), "output")


# ======================================================================
# Tests: format output triode
# ======================================================================

class TestFormatOutputTriode(unittest.TestCase):

    def test_header(self):
        pts = _make_output_triode_points()
        content = format_utd(pts, fmt="output")
        lines = content.splitlines()
        self.assertIn("Va", lines[0])
        self.assertIn("Ia", lines[0])
        self.assertNotIn("Is", lines[0])

    def test_step_values(self):
        pts = _make_output_triode_points()
        content = format_utd(pts, fmt="output")
        lines = content.splitlines()
        self.assertIn("Vg = -4 V", lines[1])
        self.assertIn("Vg = 0 V", lines[1])

    def test_data_rows(self):
        pts = _make_output_triode_points()
        content = format_utd(pts, fmt="output")
        lines = content.splitlines()
        # 3 data rows (Va values)
        data_lines = [l for l in lines[2:] if l.strip()]
        self.assertEqual(len(data_lines), 3)
        self.assertTrue(data_lines[0].startswith("5.2"))

    def test_matrix_dimensions(self):
        pts = _make_output_triode_points()
        content = format_utd(pts, fmt="output")
        lines = content.splitlines()
        data_lines = [l for l in lines[2:] if l.strip()]
        # Each row: 1 Va + 5 Ia values = 6 tokens
        tokens = data_lines[0].split()
        self.assertEqual(len(tokens), 6)


# ======================================================================
# Tests: format output pentode
# ======================================================================

class TestFormatOutputPentode(unittest.TestCase):

    def test_header_has_is(self):
        pts = _make_output_pentode_points()
        content = format_utd(pts, fmt="output")
        lines = content.splitlines()
        self.assertIn("Is", lines[0])

    def test_interleaved_columns(self):
        pts = _make_output_pentode_points()
        content = format_utd(pts, fmt="output")
        lines = content.splitlines()
        data_lines = [l for l in lines[2:] if l.strip()]
        # Each row: 1 Va + 3*(Ia+Is) = 7 tokens
        tokens = data_lines[0].split()
        self.assertEqual(len(tokens), 7)

    def test_step_values_doubled(self):
        pts = _make_output_pentode_points()
        content = format_utd(pts, fmt="output")
        lines = content.splitlines()
        step_line = lines[1]
        # Each Vg appears twice (for Ia and Is columns)
        self.assertEqual(step_line.count("Vg = -4 V"), 2)
        self.assertEqual(step_line.count("Vg = -3 V"), 2)
        self.assertEqual(step_line.count("Vg = -2 V"), 2)


# ======================================================================
# Tests: format transfer curves
# ======================================================================

class TestFormatTransfer(unittest.TestCase):

    def test_header(self):
        pts = _make_transfer_points()
        content = format_utd(pts, fmt="transfer")
        lines = content.splitlines()
        self.assertIn("Vg", lines[0])
        self.assertIn("Ia", lines[0])

    def test_step_values(self):
        pts = _make_transfer_points()
        content = format_utd(pts, fmt="transfer")
        lines = content.splitlines()
        self.assertIn("Va = 325 V", lines[1])

    def test_multi_va_transfer(self):
        pts = _make_transfer_multi_va_points()
        content = format_utd(pts, fmt="transfer")
        lines = content.splitlines()
        self.assertIn("Va = 100 V", lines[1])
        self.assertIn("Va = 200 V", lines[1])
        self.assertIn("Va = 300 V", lines[1])
        data_lines = [l for l in lines[2:] if l.strip()]
        self.assertEqual(len(data_lines), 3)  # 3 Vg values


# ======================================================================
# Tests: suggest_filename
# ======================================================================

class TestSuggestFilename(unittest.TestCase):

    def test_pentode_with_ug2(self):
        pts = _make_output_pentode_points()
        name = suggest_filename("EL84", pts)
        self.assertEqual(name, "EL84_200.utd")

    def test_triode_no_ug2(self):
        pts = _make_output_triode_points()
        name = suggest_filename("ECC81", pts)
        self.assertEqual(name, "ECC81.utd")


# ======================================================================
# Tests: round-trip export → import
# ======================================================================

def _write_tmp(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".utd")
    os.write(fd, content.encode("utf-8"))
    os.close(fd)
    return path


class TestRoundTripOutputTriode(unittest.TestCase):
    """Export triode output → parse → convert → compare."""

    def test_round_trip(self):
        original = _make_output_triode_points()
        content = format_utd(original, fmt="output")

        path = _write_tmp(content)
        try:
            parsed = parse_utd(path)
        finally:
            os.unlink(path)

        self.assertEqual(parsed["format"], "output")
        self.assertFalse(parsed["has_is"])
        self.assertEqual(len(parsed["step_values"]), 5)
        self.assertEqual(len(parsed["x_values"]), 3)

        reimported = utd_to_lm19_points(parsed, vs=0.0, vh=12.6)
        self.assertEqual(len(reimported), len(original))

        orig_sorted = sorted(original, key=lambda p: (p["ua"], p["ug1"]))
        re_sorted = sorted(reimported, key=lambda p: (p["ua"], p["ug1"]))
        for o, r in zip(orig_sorted, re_sorted):
            self.assertAlmostEqual(o["ua"], r["ua"], places=1)
            self.assertAlmostEqual(o["ug1"], r["ug1"], places=1)
            self.assertAlmostEqual(o["ia"], r["ia"], places=2)


class TestRoundTripOutputPentode(unittest.TestCase):
    """Export pentode output → parse → convert → compare."""

    def test_round_trip(self):
        original = _make_output_pentode_points()
        content = format_utd(original, fmt="output")

        path = _write_tmp(content)
        try:
            parsed = parse_utd(path)
        finally:
            os.unlink(path)

        self.assertEqual(parsed["format"], "output")
        self.assertTrue(parsed["has_is"])
        self.assertEqual(len(parsed["step_values"]), 3)
        self.assertEqual(len(parsed["x_values"]), 3)

        reimported = utd_to_lm19_points(parsed, vs=200.0, vh=6.3)
        self.assertEqual(len(reimported), len(original))

        orig_sorted = sorted(original, key=lambda p: (p["ua"], p["ug1"]))
        re_sorted = sorted(reimported, key=lambda p: (p["ua"], p["ug1"]))
        for o, r in zip(orig_sorted, re_sorted):
            self.assertAlmostEqual(o["ua"], r["ua"], places=1)
            self.assertAlmostEqual(o["ug1"], r["ug1"], places=1)
            self.assertAlmostEqual(o["ia"], r["ia"], places=2)
            self.assertAlmostEqual(o["ig2"], r["ig2"], places=2)


class TestRoundTripTransfer(unittest.TestCase):
    """Export transfer → parse → convert → compare."""

    def test_round_trip_single_va(self):
        original = _make_transfer_points()
        content = format_utd(original, fmt="transfer")

        path = _write_tmp(content)
        try:
            parsed = parse_utd(path)
        finally:
            os.unlink(path)

        self.assertEqual(parsed["format"], "transfer")
        self.assertEqual(len(parsed["step_values"]), 1)
        self.assertEqual(len(parsed["x_values"]), 6)

        reimported = utd_to_lm19_points(parsed, vs=250.0, vh=6.3)
        self.assertEqual(len(reimported), len(original))

        orig_sorted = sorted(original, key=lambda p: (p["ug1"], p["ua"]))
        re_sorted = sorted(reimported, key=lambda p: (p["ug1"], p["ua"]))
        for o, r in zip(orig_sorted, re_sorted):
            self.assertAlmostEqual(o["ua"], r["ua"], places=1)
            self.assertAlmostEqual(o["ug1"], r["ug1"], places=1)
            self.assertAlmostEqual(o["ia"], r["ia"], places=2)

    def test_round_trip_multi_va(self):
        original = _make_transfer_multi_va_points()
        content = format_utd(original, fmt="transfer")

        path = _write_tmp(content)
        try:
            parsed = parse_utd(path)
        finally:
            os.unlink(path)

        self.assertEqual(parsed["format"], "transfer")
        self.assertEqual(len(parsed["step_values"]), 3)
        self.assertEqual(len(parsed["x_values"]), 3)

        reimported = utd_to_lm19_points(parsed, vs=0.0, vh=6.3)
        self.assertEqual(len(reimported), len(original))

        orig_sorted = sorted(original, key=lambda p: (p["ug1"], p["ua"]))
        re_sorted = sorted(reimported, key=lambda p: (p["ug1"], p["ua"]))
        for o, r in zip(orig_sorted, re_sorted):
            self.assertAlmostEqual(o["ua"], r["ua"], places=1)
            self.assertAlmostEqual(o["ug1"], r["ug1"], places=1)
            self.assertAlmostEqual(o["ia"], r["ia"], places=2)


class TestEdgeCases(unittest.TestCase):

    def test_empty_points_raises(self):
        with self.assertRaises(ValueError):
            format_utd([])

    def test_single_point(self):
        pts = [{"ua": 100.0, "ug1": -2.0, "ug2": 0.0,
                "ia": 5.0, "ig2": 0.0, "uh": 6.3, "ih": 0.0}]
        content = format_utd(pts, fmt="output")
        lines = content.splitlines()
        self.assertIn("Va", lines[0])
        data_lines = [l for l in lines[2:] if l.strip()]
        self.assertEqual(len(data_lines), 1)


# ======================================================================
# Tests: _fmt_voltage / _fmt_current helpers
# ======================================================================

class TestFmtVoltage(unittest.TestCase):

    def test_integer(self):
        self.assertEqual(_fmt_voltage(0.0), "0")
        self.assertEqual(_fmt_voltage(-4.0), "-4")
        self.assertEqual(_fmt_voltage(100.0), "100")

    def test_one_decimal(self):
        self.assertEqual(_fmt_voltage(5.2), "5.2")
        self.assertEqual(_fmt_voltage(-3.5), "-3.5")

    def test_two_decimals(self):
        self.assertEqual(_fmt_voltage(-3.75), "-3.75")
        self.assertEqual(_fmt_voltage(0.25), "0.25")
        self.assertEqual(_fmt_voltage(-1.25), "-1.25")


class TestFmtCurrent(unittest.TestCase):

    def test_zero(self):
        self.assertEqual(_fmt_current(0.0), "0.000")

    def test_below_threshold(self):
        self.assertEqual(_fmt_current(0.0004), "0.000")
        self.assertEqual(_fmt_current(-0.0004), "0.000")

    def test_normal(self):
        self.assertEqual(_fmt_current(1.234), "1.234")
        self.assertEqual(_fmt_current(0.05), "0.050")

    def test_negative(self):
        self.assertEqual(_fmt_current(-0.5), "-0.500")

    def test_consistent_decimals(self):
        """All non-zero outputs should have 3 decimal places."""
        for val in [0.001, 0.1, 1.0, 10.0, 100.5]:
            result = _fmt_current(val)
            self.assertEqual(len(result.split(".")[-1]), 3,
                             f"_fmt_current({val}) = {result!r}")


# ======================================================================
# Tests: negative ig2
# ======================================================================

class TestNegativeIg2(unittest.TestCase):
    """Negative ig2 (leakage, secondary emission) should preserve Is column."""

    def test_negative_ig2_triggers_is_column(self):
        pts = [
            {"ua": 50.0, "ug1": -2.0, "ug2": 200.0,
             "ia": 5.0, "ig2": -0.05, "uh": 6.3, "ih": 0.0},
            {"ua": 100.0, "ug1": -2.0, "ug2": 200.0,
             "ia": 10.0, "ig2": -0.03, "uh": 6.3, "ih": 0.0},
        ]
        content = format_utd(pts, fmt="output")
        self.assertIn("Is", content.splitlines()[0])

    def test_negative_ig2_round_trip(self):
        pts = [
            {"ua": 50.0, "ug1": -2.0, "ug2": 200.0,
             "ia": 5.0, "ig2": -0.1, "uh": 6.3, "ih": 0.0},
            {"ua": 100.0, "ug1": -2.0, "ug2": 200.0,
             "ia": 10.0, "ig2": -0.05, "uh": 6.3, "ih": 0.0},
        ]
        content = format_utd(pts, fmt="output")
        path = _write_tmp(content)
        try:
            parsed = parse_utd(path)
        finally:
            os.unlink(path)
        self.assertTrue(parsed["has_is"])
        reimported = utd_to_lm19_points(parsed, vs=200.0)
        for o, r in zip(
            sorted(pts, key=lambda p: p["ua"]),
            sorted(reimported, key=lambda p: p["ua"]),
        ):
            self.assertAlmostEqual(o["ig2"], r["ig2"], places=2)


# ======================================================================
# Tests: sparse grid
# ======================================================================

class TestSparseGrid(unittest.TestCase):
    """Missing (ua, ug1) combinations → filled with 0.0."""

    def test_sparse_fills_zeros(self):
        pts = [
            {"ua": 50.0, "ug1": -4.0, "ug2": 0.0,
             "ia": 1.0, "ig2": 0.0, "uh": 6.3, "ih": 0.0},
            {"ua": 100.0, "ug1": -2.0, "ug2": 0.0,
             "ia": 5.0, "ig2": 0.0, "uh": 6.3, "ih": 0.0},
        ]
        content = format_utd(pts, fmt="output")
        path = _write_tmp(content)
        try:
            parsed = parse_utd(path)
        finally:
            os.unlink(path)
        # 2 Va × 2 Vg = 4 points, but only 2 had data
        self.assertEqual(len(parsed["x_values"]), 2)
        self.assertEqual(len(parsed["step_values"]), 2)
        reimported = utd_to_lm19_points(parsed)
        self.assertEqual(len(reimported), 4)
        # Missing combos should have ia=0
        by_key = {(round(p["ua"], 1), round(p["ug1"], 2)): p for p in reimported}
        self.assertAlmostEqual(by_key[(50.0, -2.0)]["ia"], 0.0)
        self.assertAlmostEqual(by_key[(100.0, -4.0)]["ia"], 0.0)


# ======================================================================
# Tests: lossy fields (ug2, uh not preserved in .utd)
# ======================================================================

class TestLossyFields(unittest.TestCase):
    """Ug2 and Uh are NOT stored in .utd — verify this is handled."""

    def test_ug2_not_in_file(self):
        pts = _make_output_pentode_points()  # ug2=200
        content = format_utd(pts, fmt="output")
        # Vs is not anywhere in the .utd content
        self.assertNotIn("200", content.splitlines()[0])
        self.assertNotIn("200", content.splitlines()[1])

    def test_reimport_with_wrong_vs(self):
        pts = _make_output_pentode_points()  # ug2=200
        content = format_utd(pts, fmt="output")
        path = _write_tmp(content)
        try:
            parsed = parse_utd(path)
        finally:
            os.unlink(path)
        # If user supplies wrong Vs, all ug2 will be wrong
        reimported = utd_to_lm19_points(parsed, vs=0.0)
        for p in reimported:
            self.assertAlmostEqual(p["ug2"], 0.0)

    def test_uh_always_from_parameter(self):
        pts = _make_transfer_points()  # uh=6.3
        content = format_utd(pts, fmt="transfer")
        path = _write_tmp(content)
        try:
            parsed = parse_utd(path)
        finally:
            os.unlink(path)
        reimported = utd_to_lm19_points(parsed, vs=250.0, vh=12.6)
        for p in reimported:
            self.assertAlmostEqual(p["uh"], 12.6)


# ======================================================================
# Tests: pentode transfer round-trip
# ======================================================================

class TestRoundTripPentodeTransfer(unittest.TestCase):
    """Pentode data exported as transfer curves with Is columns."""

    def test_pentode_transfer_round_trip(self):
        pts = _make_output_pentode_points()
        content = format_utd(pts, fmt="transfer")
        lines = content.splitlines()
        self.assertIn("Vg", lines[0])
        self.assertIn("Is", lines[0])

        path = _write_tmp(content)
        try:
            parsed = parse_utd(path)
        finally:
            os.unlink(path)
        self.assertEqual(parsed["format"], "transfer")
        self.assertTrue(parsed["has_is"])
        reimported = utd_to_lm19_points(parsed, vs=200.0)
        self.assertEqual(len(reimported), len(pts))


# ======================================================================
# Tests: auto-detect format (fmt=None)
# ======================================================================

class TestAutoDetectIntegration(unittest.TestCase):

    def test_auto_detect_through_format_utd(self):
        """format_utd(pts) without explicit fmt should not raise."""
        pts = _make_output_triode_points()
        content = format_utd(pts)
        self.assertTrue(content.startswith("V"))  # Va or Vg header

    def test_auto_detect_transfer(self):
        pts = _make_transfer_points()  # 6 Vg, 1 Va → transfer
        content = format_utd(pts)
        self.assertIn("Vg", content.splitlines()[0])

    def test_auto_detect_output(self):
        """Equal counts → output (>= rule)."""
        pts = _make_output_pentode_points()  # 3 Va, 3 Vg → output
        content = format_utd(pts)
        self.assertIn("Va", content.splitlines()[0])


# ======================================================================
# Tests: suggest_filename edge cases
# ======================================================================

class TestSuggestFilenameEdge(unittest.TestCase):

    def test_empty_tube_type(self):
        pts = [{"ua": 100, "ug1": -2, "ug2": 0, "ia": 5,
                "ig2": 0, "uh": 6.3, "ih": 0}]
        name = suggest_filename("", pts)
        self.assertEqual(name, ".utd")

    def test_empty_points(self):
        name = suggest_filename("ECC81", [])
        self.assertEqual(name, "ECC81.utd")

    def test_mixed_ug2_uses_max(self):
        pts = [
            {"ua": 100, "ug1": -2, "ug2": 150, "ia": 5,
             "ig2": 0.1, "uh": 6.3, "ih": 0},
            {"ua": 100, "ug1": -4, "ug2": 250, "ia": 3,
             "ig2": 0.1, "uh": 6.3, "ih": 0},
        ]
        name = suggest_filename("EL84", pts)
        self.assertEqual(name, "EL84_250.utd")

    def test_small_ug2_rounds_to_zero(self):
        """ug2 < 0.5 rounds to 0 → treated as triode."""
        pts = [{"ua": 100, "ug1": -2, "ug2": 0.3, "ia": 5,
                "ig2": 0, "uh": 6.3, "ih": 0}]
        name = suggest_filename("ECC81", pts)
        self.assertEqual(name, "ECC81.utd")


# ======================================================================
# Tests: Vg with 0.25V steps (regression for _fmt_voltage bug)
# ======================================================================

class TestFractionalVgSteps(unittest.TestCase):
    """Verify that 0.25V Vg steps survive round-trip."""

    def test_025v_steps_round_trip(self):
        pts = []
        for ua in [50.0, 100.0]:
            for vg in [-3.0, -2.75, -2.5, -2.25, -2.0]:
                pts.append({"ua": ua, "ug1": vg, "ug2": 0.0,
                            "ia": abs(vg) * ua / 100, "ig2": 0.0,
                            "uh": 6.3, "ih": 0.0})
        content = format_utd(pts, fmt="output")
        # Verify step line preserves .25 / .75
        self.assertIn("-2.75", content)
        self.assertIn("-2.25", content)

        path = _write_tmp(content)
        try:
            parsed = parse_utd(path)
        finally:
            os.unlink(path)
        self.assertEqual(parsed["step_values"], [-3.0, -2.75, -2.5, -2.25, -2.0])


if __name__ == "__main__":
    unittest.main()
