"""Unit tests for lm19.etracer_import module.

Run:  py -m pytest tests/test_etracer_import.py -v
"""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.etracer_import import (
    parse_etracer_csv,
    detect_topology,
    etracer_to_lm19_points,
    guess_meta_from_etracer,
    extract_heater_from_etd,
    _SWEEP_NEGV,
)

# ── Paths to real sample files ──
_SAMPLES_DIR = os.path.join(
    os.path.dirname(__file__),
    "spice_test_data", "raw", "etracer_samples",
)


def _sample(name: str) -> str:
    return os.path.join(_SAMPLES_DIR, name)


# ── Synthetic test data ──

TRIODE_CSV = """\
# ETRACER_CSV_FORMAT_VERSION:2.0
# Each curve-set contains 6 rows: [HV1_V HV1_I HV2_V HV2_I NEGV SWEEP_SOURCE]
# SWEEP_SOURCE types: 0:NONE 1:NEGV 2:HV2
# Each row starts with a curve-set sequential number starting from 0 followed by data pts.
# ETD_FILE:test/triode.etd
# NEGV:ON NEGV_SWEEP:ON NEGV_SETTING: [0.0:10.0:5.0]
# HV2:OFF HV2_LINK:OFF HV2_SETTING:[0.0:0.0:0.0]
0,50.0,100.0,150.0,nan
0,1.5,5.0,10.0,nan
0,0.5,0.5,0.5,nan
0,0.01,0.01,0.01,nan
0,-0.00,-0.00,-0.00,nan
0,1.00,1.00,1.00,nan
1,50.0,100.0,150.0,200.0,nan
1,0.5,2.0,5.0,9.0,nan
1,0.5,0.5,0.5,0.5,nan
1,0.01,0.01,0.01,0.01,nan
1,-5.00,-5.00,-5.00,-5.00,nan
1,1.00,1.00,1.00,1.00,nan
"""

TRIODE_CONNECTED_CSV = """\
# ETRACER_CSV_FORMAT_VERSION:2.0
# Each curve-set contains 6 rows: [HV1_V HV1_I HV2_V HV2_I NEGV SWEEP_SOURCE]
# SWEEP_SOURCE types: 0:NONE 1:NEGV 2:HV2
# ETD_FILE:test/pentode_triode.etd
# NEGV:ON NEGV_SWEEP:ON NEGV_SETTING: [0.0:10.0:5.0]
# HV2:ON HV2_LINK:ON HV2_SETTING:[0.0:0.0:0.0]
0,50.0,100.0,150.0,nan
0,10.0,30.0,60.0,nan
0,50.5,100.3,150.2,nan
0,1.5,3.0,5.0,nan
0,-0.00,-0.00,-0.00,nan
0,1.00,1.00,1.00,nan
"""

PENTODE_CSV = """\
# ETRACER_CSV_FORMAT_VERSION:2.0
# Each curve-set contains 6 rows: [HV1_V HV1_I HV2_V HV2_I NEGV SWEEP_SOURCE]
# SWEEP_SOURCE types: 0:NONE 1:NEGV 2:HV2
# ETD_FILE:test/pentode.etd
# NEGV:ON NEGV_SWEEP:OFF NEGV_SETTING: [0.0:0.0:0.0]
# HV2:ON HV2_LINK:OFF HV2_SETTING:[100.0:300.0:100.0]
0,50.0,100.0,150.0,nan
0,5.0,12.0,18.0,nan
0,250.0,250.0,250.0,nan
0,1.0,2.5,3.5,nan
0,-10.00,-10.00,-10.00,nan
0,2.00,2.00,2.00,nan
1,50.0,100.0,150.0,nan
1,3.0,8.0,14.0,nan
1,200.0,200.0,200.0,nan
1,0.8,1.8,3.0,nan
1,-10.00,-10.00,-10.00,nan
1,2.00,2.00,2.00,nan
"""

# Fixed-screen pentode: HV2 on, not linked, NOT swept (SWEEP_SOURCE=1=NEGV).
# detect_topology() still returns "pentode"; the screen Ug2/Ig2 must be kept.
PENTODE_FIXED_SCREEN_CSV = """\
# ETRACER_CSV_FORMAT_VERSION:2.0
# Each curve-set contains 6 rows: [HV1_V HV1_I HV2_V HV2_I NEGV SWEEP_SOURCE]
# SWEEP_SOURCE types: 0:NONE 1:NEGV 2:HV2
# ETD_FILE:test/pentode_fixed.etd
# NEGV:ON NEGV_SWEEP:ON NEGV_SETTING: [-10.0:-6.0:2.0]
# HV2:ON HV2_LINK:OFF HV2_SETTING:[250.0:250.0:0.0]
0,250.0,250.0,250.0,nan
0,5.0,8.0,12.0,nan
0,250.0,250.0,250.0,nan
0,1.0,2.5,3.5,nan
0,-10.00,-8.00,-6.00,nan
0,1.00,1.00,1.00,nan
"""


def _write_tmp(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.write(fd, content.encode("utf-8"))
    os.close(fd)
    return path


class TestParseEtracerCsv(unittest.TestCase):
    """Tests for parse_etracer_csv() on synthetic data."""

    def test_parse_triode(self) -> None:
        path = _write_tmp(TRIODE_CSV)
        try:
            result = parse_etracer_csv(path)
        finally:
            os.unlink(path)

        self.assertEqual(result["version"], "2.0")
        self.assertFalse(result["hv2_on"])
        self.assertFalse(result["hv2_link"])
        self.assertEqual(result["negv_setting"], (0.0, 10.0, 5.0))
        self.assertEqual(len(result["curves"]), 2)

        c0 = result["curves"][0]
        self.assertEqual(c0["curve_idx"], 0)
        self.assertEqual(len(c0["hv1_v"]), 3)  # nan stripped
        self.assertAlmostEqual(c0["hv1_v"][1], 100.0)
        self.assertAlmostEqual(c0["hv1_i"][1], 5.0)
        self.assertAlmostEqual(c0["sweep_src"], 1.0)

        c1 = result["curves"][1]
        self.assertEqual(len(c1["hv1_v"]), 4)
        self.assertAlmostEqual(c1["negv"][0], -5.0)

    def test_parse_triode_connected(self) -> None:
        path = _write_tmp(TRIODE_CONNECTED_CSV)
        try:
            result = parse_etracer_csv(path)
        finally:
            os.unlink(path)

        self.assertTrue(result["hv2_on"])
        self.assertTrue(result["hv2_link"])
        self.assertEqual(len(result["curves"]), 1)
        c = result["curves"][0]
        # HV2 linked: HV2_V should track HV1_V
        self.assertAlmostEqual(c["hv2_v"][0], 50.5, places=1)
        self.assertAlmostEqual(c["hv2_v"][1], 100.3, places=1)

    def test_parse_pentode(self) -> None:
        path = _write_tmp(PENTODE_CSV)
        try:
            result = parse_etracer_csv(path)
        finally:
            os.unlink(path)

        self.assertTrue(result["hv2_on"])
        self.assertFalse(result["hv2_link"])
        self.assertEqual(len(result["curves"]), 2)
        c0 = result["curves"][0]
        self.assertAlmostEqual(c0["sweep_src"], 2.0)
        self.assertAlmostEqual(c0["hv2_v"][0], 250.0)

    def test_empty_file_raises(self) -> None:
        path = _write_tmp("# just a comment\n")
        try:
            with self.assertRaises(ValueError):
                parse_etracer_csv(path)
        finally:
            os.unlink(path)

    def test_incomplete_curve_skipped(self) -> None:
        """A curve-set with fewer than 6 rows should be skipped."""
        csv = TRIODE_CSV + "99,1.0,2.0,nan\n99,0.5,1.0,nan\n"
        path = _write_tmp(csv)
        try:
            result = parse_etracer_csv(path)
        finally:
            os.unlink(path)
        # Only the 2 complete curves from TRIODE_CSV
        self.assertEqual(len(result["curves"]), 2)


class TestDetectTopology(unittest.TestCase):
    def test_triode(self) -> None:
        parsed = {"hv2_on": False, "hv2_link": False, "curves": []}
        self.assertEqual(detect_topology(parsed), "triode")

    def test_triode_connected(self) -> None:
        parsed = {"hv2_on": True, "hv2_link": True, "curves": []}
        self.assertEqual(detect_topology(parsed), "triode_connected")

    def test_pentode_by_sweep(self) -> None:
        parsed = {"hv2_on": True, "hv2_link": False,
                  "curves": [{"sweep_src": 2.0}]}
        self.assertEqual(detect_topology(parsed), "pentode")

    def test_pentode_fixed_screen(self) -> None:
        parsed = {"hv2_on": True, "hv2_link": False,
                  "curves": [{"sweep_src": 1.0}]}
        self.assertEqual(detect_topology(parsed), "pentode")


class TestEtracerToLm19Points(unittest.TestCase):
    """Tests for etracer_to_lm19_points() on synthetic data."""

    def test_triode_points(self) -> None:
        path = _write_tmp(TRIODE_CSV)
        try:
            parsed = parse_etracer_csv(path)
        finally:
            os.unlink(path)

        points = etracer_to_lm19_points(parsed, vh=5.0)
        # 3 + 4 = 7 points
        self.assertEqual(len(points), 7)
        p0 = points[0]
        self.assertAlmostEqual(p0["ua"], 50.0)
        self.assertAlmostEqual(p0["ug1"], 0.0)
        self.assertAlmostEqual(p0["ia"], 1.5)
        self.assertAlmostEqual(p0["ug2"], 0.0)
        self.assertAlmostEqual(p0["ig2"], 0.0)
        self.assertAlmostEqual(p0["uh"], 5.0)

        # Curve 1 has Ug1 = -5V
        p3 = points[3]
        self.assertAlmostEqual(p3["ug1"], -5.0)

    def test_triode_connected_points(self) -> None:
        path = _write_tmp(TRIODE_CONNECTED_CSV)
        try:
            parsed = parse_etracer_csv(path)
        finally:
            os.unlink(path)

        points = etracer_to_lm19_points(parsed)
        self.assertEqual(len(points), 3)
        # Ug2 should track Ua (HV2 linked)
        self.assertAlmostEqual(points[0]["ug2"], 50.5, places=1)
        self.assertAlmostEqual(points[0]["ig2"], 1.5)

    def test_pentode_points(self) -> None:
        path = _write_tmp(PENTODE_CSV)
        try:
            parsed = parse_etracer_csv(path)
        finally:
            os.unlink(path)

        points = etracer_to_lm19_points(parsed)
        self.assertEqual(len(points), 6)
        # Pentode: Ug2 from HV2_V, Ug1 from NEGV
        p0 = points[0]
        self.assertAlmostEqual(p0["ug1"], -10.0)
        self.assertAlmostEqual(p0["ug2"], 250.0)
        self.assertAlmostEqual(p0["ig2"], 1.0)
        self.assertAlmostEqual(p0["ia"], 5.0)

    def test_pentode_fixed_screen_keeps_screen(self) -> None:
        """Fixed-screen pentode (HV2 on, not swept) must retain Ug2/Ig2.

        Before the fix the non-HV2-sweep branch zeroed Ug2/Ig2 for any topology
        other than triode_connected — discarding the recorded screen data.
        """
        path = _write_tmp(PENTODE_FIXED_SCREEN_CSV)
        try:
            parsed = parse_etracer_csv(path)
        finally:
            os.unlink(path)

        self.assertEqual(detect_topology(parsed), "pentode")
        self.assertEqual(parsed["curves"][0]["sweep_src"], _SWEEP_NEGV)

        points = etracer_to_lm19_points(parsed)
        self.assertEqual(len(points), 3)
        # Screen voltage/current preserved, not zeroed.
        self.assertAlmostEqual(points[0]["ug2"], 250.0)
        self.assertAlmostEqual(points[0]["ig2"], 1.0)
        self.assertAlmostEqual(points[0]["ug1"], -10.0)
        self.assertTrue(all(p["ug2"] == 250.0 for p in points))
        self.assertTrue(any(p["ig2"] > 0 for p in points))

    def test_all_points_have_required_keys(self) -> None:
        path = _write_tmp(TRIODE_CSV)
        try:
            parsed = parse_etracer_csv(path)
        finally:
            os.unlink(path)

        required = {"ua", "ug1", "ug2", "ia", "ig2", "uh", "ih"}
        for p in etracer_to_lm19_points(parsed):
            self.assertTrue(required.issubset(p.keys()), f"Missing keys in {p}")

    def test_ia_positive_for_valid_points(self) -> None:
        """Physical sanity: Ia should be non-negative."""
        path = _write_tmp(TRIODE_CSV)
        try:
            parsed = parse_etracer_csv(path)
        finally:
            os.unlink(path)

        for p in etracer_to_lm19_points(parsed):
            self.assertGreaterEqual(p["ia"], 0.0, f"Negative Ia: {p}")


class TestGuessMetaFromEtracer(unittest.TestCase):
    def test_triode_meta(self) -> None:
        path = _write_tmp(TRIODE_CSV)
        try:
            parsed = parse_etracer_csv(path)
            meta = guess_meta_from_etracer(path, parsed)
        finally:
            os.unlink(path)
        self.assertEqual(meta["topology"], "triode")

    def test_etd_file_extraction(self) -> None:
        parsed = {
            "etd_file": "D:/Audio/eTracer/tubecfg/EL34_Quintet_triode.etd",
            "hv2_on": True, "hv2_link": True, "curves": [],
        }
        meta = guess_meta_from_etracer("/fake/EL34_triode.csv", parsed)
        self.assertEqual(meta["tube_type"], "EL34")
        self.assertEqual(meta["topology"], "triode_connected")


class TestExtractHeaterFromEtd(unittest.TestCase):
    """Tests for extract_heater_from_etd()."""

    def test_finds_companion_etd(self) -> None:
        """When .etd file exists next to CSV, extract HEATER_V."""
        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "test.csv")
        etd_path = os.path.join(tmpdir, "test.etd")
        try:
            with open(csv_path, "w") as f:
                f.write("dummy")
            with open(etd_path, "w") as f:
                f.write("[HEATER]\nHEATER_V = 6.3\nHEATER_I = 0.3\n")
            vh = extract_heater_from_etd(csv_path, f"C:/fake/path/{os.path.basename(etd_path)}")
            self.assertAlmostEqual(vh, 6.3)
        finally:
            os.unlink(csv_path)
            os.unlink(etd_path)
            os.rmdir(tmpdir)

    def test_returns_none_when_no_etd(self) -> None:
        path = _write_tmp(TRIODE_CSV)
        try:
            vh = extract_heater_from_etd(path, "nonexistent.etd")
            self.assertIsNone(vh)
        finally:
            os.unlink(path)

    def test_returns_none_for_empty_filename(self) -> None:
        self.assertIsNone(extract_heater_from_etd("/fake/path.csv", ""))

    @unittest.skipUnless(os.path.isdir(_SAMPLES_DIR), "eTracer samples not found")
    def test_real_etd_10Y(self) -> None:
        """Real ETD file should return correct heater voltage."""
        csv_path = _sample("10Y.csv")
        # The ETD_FILE header points to 10Y_VT25.etd which exists in samples
        vh = extract_heater_from_etd(csv_path, "10Y_VT25.etd")
        if vh is not None:
            self.assertAlmostEqual(vh, 7.5)


# ── Tests on real sample files ──

@unittest.skipUnless(
    os.path.isdir(_SAMPLES_DIR),
    "eTracer sample files not found",
)
class TestRealEtracerFiles(unittest.TestCase):
    """Integration tests using real eTracer CSV samples from vt52.com."""

    def test_10Y_triode(self) -> None:
        path = _sample("10Y.csv")
        parsed = parse_etracer_csv(path)
        self.assertEqual(parsed["version"], "2.0")
        self.assertFalse(parsed["hv2_on"])
        self.assertEqual(detect_topology(parsed), "triode")
        self.assertGreater(len(parsed["curves"]), 3)

        points = etracer_to_lm19_points(parsed, vh=7.5)
        self.assertGreater(len(points), 20)
        # Physical sanity: all Ia >= 0
        for p in points:
            self.assertGreaterEqual(p["ia"], 0.0)
        # First curve at Ug1=0 should have significant current
        curve0_pts = [p for p in points if abs(p["ug1"]) < 0.1]
        if curve0_pts:
            max_ia = max(p["ia"] for p in curve0_pts)
            self.assertGreater(max_ia, 10.0, "10Y at Ug1=0 should draw >10mA")

    def test_D3a_triode(self) -> None:
        path = _sample("D3a.csv")
        parsed = parse_etracer_csv(path)
        self.assertEqual(detect_topology(parsed), "triode")

        points = etracer_to_lm19_points(parsed)
        self.assertGreater(len(points), 50)
        # D3a has small grid steps (0.5V) — many curves expected
        self.assertGreater(len(parsed["curves"]), 8)

    def test_EL34_triode_connected(self) -> None:
        path = _sample("EL34_triode.csv")
        parsed = parse_etracer_csv(path)
        self.assertTrue(parsed["hv2_on"])
        self.assertTrue(parsed["hv2_link"])
        self.assertEqual(detect_topology(parsed), "triode_connected")

        points = etracer_to_lm19_points(parsed)
        self.assertGreater(len(points), 30)
        # Ug2 should approximately track Ua (triode connected)
        for p in points:
            if p["ua"] > 10:
                self.assertGreater(p["ug2"], 0.0,
                                   "Ug2 should be >0 in triode-connected mode")

    def test_KT66_triode_connected(self) -> None:
        path = _sample("KT66.csv")
        parsed = parse_etracer_csv(path)
        self.assertEqual(detect_topology(parsed), "triode_connected")

        points = etracer_to_lm19_points(parsed)
        self.assertGreater(len(points), 20)

    def test_meta_guess_10Y(self) -> None:
        path = _sample("10Y.csv")
        parsed = parse_etracer_csv(path)
        meta = guess_meta_from_etracer(path, parsed)
        self.assertEqual(meta["tube_type"], "10Y")
        self.assertEqual(meta["topology"], "triode")

    def test_meta_guess_EL34(self) -> None:
        path = _sample("EL34_triode.csv")
        parsed = parse_etracer_csv(path)
        meta = guess_meta_from_etracer(path, parsed)
        # Should strip _triode suffix
        self.assertEqual(meta["tube_type"], "EL34")

    def test_all_samples_parse_without_error(self) -> None:
        """Smoke test: every CSV in samples dir should parse."""
        for fname in os.listdir(_SAMPLES_DIR):
            if not fname.endswith(".csv"):
                continue
            path = _sample(fname)
            with self.subTest(file=fname):
                parsed = parse_etracer_csv(path)
                self.assertGreater(len(parsed["curves"]), 0,
                                   f"{fname} has no curves")
                points = etracer_to_lm19_points(parsed)
                self.assertGreater(len(points), 0,
                                   f"{fname} produced no points")


if __name__ == "__main__":
    unittest.main()
