"""Unit tests for misc lm19 modules: lcd_parse, measurements, config.

Run:  py -m pytest tests/test_lm19.py -v

For the full test suite split by module:
  test_koren.py          — Koren model, fitting, convergence
  test_analysis_full.py  — analysis, quality, matching, aging, topology
  test_csv_export_full.py — CSV export
  test_loadline.py       — load line, interpolation, distortion, IMD
  test_curve_marker.py   — curve marker, data builders, compare
  test_scan_full.py      — scan settle, verify, Pg2, refine
  test_srk.py            — SRK measurement, sweep, real tubes
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the app root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ======================================================================
# LCD parse tests
# ======================================================================

from lm19.lcd_parse import parse_lcd_line
from lm19.constants import (
    TOPOLOGY_PENTODE,
)


class TestLcdParse(unittest.TestCase):
    """Test LCD dump parsing."""

    def test_full_line(self):
        """Parse a typical 62-character LCD dump line."""
        line = "TH ECC83     6.30 300  2.50 250 01.20 170 00.50 16.2 0625 0160"
        result = parse_lcd_line(line)
        self.assertEqual(result["type"], "TH")
        self.assertEqual(result["name"], "ECC83")
        self.assertEqual(result["uh"], "6.30")
        self.assertEqual(result["ua"], "250")
        self.assertEqual(result["s"], "16.2")

    def test_short_line_padded(self):
        """Short line should be padded to 62 chars."""
        line = "TH TEST"
        result = parse_lcd_line(line)
        self.assertEqual(result["type"], "TH")

    def test_field_stripping(self):
        """Fields should have whitespace stripped."""
        line = " T  Test      6.30 300   2.50 250 01.20 170 00.50 16.2 0625 0160"
        result = parse_lcd_line(line)
        self.assertEqual(result["type"], "T")
        self.assertEqual(result["name"], "Test")


# ======================================================================
# Measurements save/load tests
# ======================================================================


class TestMeasurements(unittest.TestCase):
    """Test measurement save/load logic."""

    def test_sanitize(self):
        from lm19.measurements import _sanitize
        self.assertEqual(_sanitize("ECC83"), "ECC83")
        self.assertEqual(_sanitize("test/bad:name"), "test_bad_name")
        self.assertEqual(_sanitize(""), "unknown")
        self.assertEqual(_sanitize("  "), "unknown")

    def test_measurement_filename(self):
        from lm19.measurements import _measurement_filename
        fname = _measurement_filename("L001", "2026-01-15", "test_run")
        self.assertTrue(fname.endswith(".json"))
        self.assertIn("L001", fname)
        self.assertIn("2026-01-15", fname)
        self.assertIn("test_run", fname)

    def test_save_and_load(self):
        """Save and load a measurement using temp directory."""
        import lm19.measurements as mmod
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mmod, '_root', return_value=Path(tmpdir)):
                measurement = {
                    "tube_type": "ECC83",
                    "lamp_id": "L001",
                    "timestamp": "2026-02-15T12:00:00",
                    "name": "test",
                    "points": [{"ua": 250, "ug1": -2.0, "ia": 1.2}],
                }
                path = mmod.save_measurement("ECC83", "L001", measurement)
                self.assertTrue(path.exists())

                loaded = mmod.load_measurements("ECC83", "L001")
                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded[0]["lamp_id"], "L001")
                self.assertEqual(len(loaded[0]["points"]), 1)

    def test_list_lamp_ids(self):
        """list_lamp_ids should return saved lamp IDs."""
        import lm19.measurements as mmod
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mmod, '_root', return_value=Path(tmpdir)):
                m1 = {"lamp_id": "L001", "timestamp": "t1", "name": "a", "points": []}
                m2 = {"lamp_id": "L002", "timestamp": "t2", "name": "b", "points": []}
                mmod.save_measurement("ECC83", "L001", m1)
                mmod.save_measurement("ECC83", "L002", m2)
                ids = mmod.list_lamp_ids("ECC83")
                self.assertIn("L001", ids)
                self.assertIn("L002", ids)

    def test_unreadable_file_does_not_kill_listing(self):
        """ML-117: one non-UTF8 / locked file must not take down the whole
        listing — the old except caught only JSONDecodeError, so
        UnicodeDecodeError/OSError crashed load_measurements /
        list_lamp_ids / list_measurement_entries because of one file."""
        import lm19.measurements as mmod
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mmod, '_root', return_value=Path(tmpdir)):
                m = {"lamp_id": "L001", "timestamp": "t1", "name": "a",
                     "points": []}
                saved = mmod.save_measurement("ECC83", "L001", m)
                # plant a non-UTF8 file next to the valid one
                broken = saved.parent / "broken.json"
                broken.write_bytes(bytes([0xFF, 0xFE, 0x00]) + b"broken")

                loaded = mmod.load_measurements("ECC83", "L001")
                self.assertEqual(len(loaded), 1)
                self.assertIn("L001", mmod.list_lamp_ids("ECC83"))
                entries = mmod.list_measurement_entries()
                self.assertTrue(any(e.get("lamp_id") == "L001"
                                    for e in entries))

    def test_duplicate_save_no_overwrite(self):
        """Saving same measurement twice should not overwrite."""
        import lm19.measurements as mmod
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mmod, '_root', return_value=Path(tmpdir)):
                m = {"lamp_id": "L001", "timestamp": "t1", "name": "a", "points": []}
                p1 = mmod.save_measurement("ECC83", "L001", m)
                p2 = mmod.save_measurement("ECC83", "L001", m)
                self.assertNotEqual(p1, p2)
                self.assertTrue(p1.exists())
                self.assertTrue(p2.exists())

    def test_save_imported_measurement_filename_no_timestamp(self):
        """Imported files should use import_<source> naming without timestamp."""
        import lm19.measurements as mmod
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mmod, '_root', return_value=Path(tmpdir)):
                m = {
                    "tube_type": "ECC83",
                    "lamp_id": "L001",
                    "timestamp": "2026-02-22T12:00:00",
                    "name": "from_file",
                    "points": [],
                }
                p = mmod.save_imported_measurement(
                    "ECC83",
                    "L001",
                    m,
                    source="csv",
                    source_stem="my_data",
                )
                self.assertEqual(p.name, "L001__import_csv__my_data.json")

    def test_save_imported_measurement_no_overwrite(self):
        """Saving imported measurement twice should add numeric suffix."""
        import lm19.measurements as mmod
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mmod, '_root', return_value=Path(tmpdir)):
                m = {"lamp_id": "L001", "name": "n", "points": []}
                p1 = mmod.save_imported_measurement(
                    "ECC83", "L001", m, source="utd", source_stem="el84_250"
                )
                p2 = mmod.save_imported_measurement(
                    "ECC83", "L001", m, source="utd", source_stem="el84_250"
                )
                self.assertNotEqual(p1, p2)
                self.assertTrue(p2.name.endswith("_1.json"))


# ======================================================================
# Config / LampConfig tests
# ======================================================================


class TestLampConfig(unittest.TestCase):
    """Test LampConfig loading and basic properties."""

    def test_load_lamps_non_empty(self):
        from lm19.config import load_lamps
        lamps = load_lamps()
        self.assertGreater(len(lamps), 0)

    def test_find_lamp(self):
        from lm19.config import load_lamps, find_lamp
        lamps = load_lamps()
        lamp = find_lamp(lamps, "ECC83")
        self.assertIsNotNone(lamp)
        self.assertEqual(lamp.topology, "triode")

    def test_all_lamps_have_socket(self):
        """Every lamp should have a socket field (possibly empty string)."""
        from lm19.config import load_lamps
        lamps = load_lamps()
        for lamp in lamps:
            self.assertIsNotNone(lamp.socket,
                                 f"{lamp.tube_type} missing socket")

    def test_pentode_ranges_include_ug2(self):
        """Pentode lamps should have ug2 in ranges."""
        from lm19.config import load_lamps
        lamps = load_lamps()
        pentodes = [l for l in lamps if l.topology == TOPOLOGY_PENTODE]
        if pentodes:
            lamp = pentodes[0]
            self.assertIn("ug2", lamp.ranges,
                          f"{lamp.tube_type}: pentode missing ug2 range")


if __name__ == "__main__":
    unittest.main(verbosity=2)
