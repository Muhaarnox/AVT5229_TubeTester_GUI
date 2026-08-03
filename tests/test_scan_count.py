"""Tests for scan_point_count."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from scan_test_helpers import ScanSettings, ScanRange, scan_point_count


class TestScanPointCount(unittest.TestCase):
    """Tests for scan_point_count."""

    def test_triode_count(self):
        s = ScanSettings(
            ua=ScanRange(0, 300, 10),
            ug1=ScanRange(-6, 0, 1),
            ug2=ScanRange(0, 0, 0),
            uh=6.3, ih=0.0,
            is_triode=True,
        )
        # 31 ua × 7 ug1
        self.assertEqual(scan_point_count(s), 31 * 7)

    def test_pentode_count(self):
        s = ScanSettings(
            ua=ScanRange(0, 300, 10),
            ug1=ScanRange(-6, 0, 1),
            ug2=ScanRange(200, 250, 50),
            uh=6.3, ih=0.0,
            is_triode=False,
        )
        # 31 ua × 7 ug1 × 2 ug2
        self.assertEqual(scan_point_count(s), 31 * 7 * 2)

    def test_track_mode_count(self):
        s = ScanSettings(
            ua=ScanRange(0, 300, 10),
            ug1=ScanRange(-6, 0, 1),
            ug2=ScanRange(0, 0, 0),
            uh=6.3, ih=0.0,
            is_triode=False,
            ug2_track_ua=True,
        )
        # track: same as triode (no ug2 sweep)
        self.assertEqual(scan_point_count(s), 31 * 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
