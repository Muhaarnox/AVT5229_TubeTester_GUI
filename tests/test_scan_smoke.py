"""Smoke tests for scan grid counting logic."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.scan import ScanRange, ScanSettings, scan_point_count

pytestmark = [pytest.mark.smoke_scan]


@pytest.mark.smoke
def test_scan_point_count_modes_smoke():
    # ua: 0,10,20 -> 3 ; ug1: -2,-1,0 -> 3 ; ug2: 100,200 -> 2
    base = ScanSettings(
        ua=ScanRange(0, 20, 10),
        ug1=ScanRange(-2, 0, 1),
        ug2=ScanRange(100, 200, 100),
        uh=6.3,
        ih=0.7,
    )

    assert scan_point_count(base) == 18
    assert scan_point_count(ScanSettings(**{**base.__dict__, "is_triode": True})) == 9
    assert scan_point_count(ScanSettings(**{**base.__dict__, "ug2_track_ua": True})) == 9
