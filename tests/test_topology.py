"""Tests for topology-detection helpers in ``lm19.measurements``.

Covers ``get_ug2_mode``, ``is_entry_triode``, and
``is_ug2_track_mode`` — the pure logic that CompareTab uses to
classify imported measurements.
"""

import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.measurements import get_ug2_mode, is_entry_triode as is_triode, is_ug2_track_mode  # noqa: E402
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)


# ---------------------------------------------------------------------------
# Tests: get_ug2_mode
# ---------------------------------------------------------------------------

class TestGetUg2Mode:
    def test_explicit_ug2_mode_in_scan(self):
        entry = {"data": {"scan": {"ug2_mode": TOPOLOGY_TRIODE_CONNECTED}}}
        assert get_ug2_mode(entry) == TOPOLOGY_TRIODE_CONNECTED

    def test_explicit_ug2_mode_triode(self):
        entry = {"data": {"scan": {"ug2_mode": TOPOLOGY_TRIODE}}}
        assert get_ug2_mode(entry) == TOPOLOGY_TRIODE

    def test_explicit_ug2_mode_pentode(self):
        entry = {"data": {"scan": {"ug2_mode": TOPOLOGY_PENTODE}}}
        assert get_ug2_mode(entry) == TOPOLOGY_PENTODE

    def test_topology_triode(self):
        entry = {"data": {"topology": TOPOLOGY_TRIODE, "scan": {}}}
        assert get_ug2_mode(entry) == TOPOLOGY_TRIODE

    def test_topology_triode_no_scan(self):
        entry = {"data": {"topology": TOPOLOGY_TRIODE}}
        assert get_ug2_mode(entry) == TOPOLOGY_TRIODE

    def test_ug2_track_ua_true(self):
        entry = {"data": {"scan": {"ug2_track_ua": True}}}
        assert get_ug2_mode(entry) == TOPOLOGY_TRIODE_CONNECTED

    def test_ug2_track_ua_false(self):
        entry = {"data": {"scan": {"ug2_track_ua": False}}}
        assert get_ug2_mode(entry) == TOPOLOGY_PENTODE

    def test_default_pentode(self):
        entry = {"data": {"scan": {}}}
        assert get_ug2_mode(entry) == TOPOLOGY_PENTODE

    def test_empty_data(self):
        entry = {"data": {}}
        assert get_ug2_mode(entry) == TOPOLOGY_PENTODE

    def test_no_data_key(self):
        entry = {}
        assert get_ug2_mode(entry) == TOPOLOGY_PENTODE

    def test_ug2_mode_takes_priority_over_topology(self):
        entry = {"data": {"topology": TOPOLOGY_TRIODE, "scan": {"ug2_mode": TOPOLOGY_PENTODE}}}
        assert get_ug2_mode(entry) == TOPOLOGY_PENTODE

    def test_ug2_mode_takes_priority_over_track_ua(self):
        entry = {"data": {"scan": {"ug2_mode": TOPOLOGY_PENTODE, "ug2_track_ua": True}}}
        assert get_ug2_mode(entry) == TOPOLOGY_PENTODE

    def test_autodetect_from_points_no_metadata(self):
        """No scan metadata, but points show ug2 = ua → triode_connected."""
        entry = {
            "data": {"scan": {}},
            "points": [
                {"ua": 50, "ug2": 50},
                {"ua": 100, "ug2": 100},
                {"ua": 150, "ug2": 150},
                {"ua": 200, "ug2": 200},
            ],
        }
        assert get_ug2_mode(entry) == TOPOLOGY_TRIODE_CONNECTED

    def test_autodetect_from_points_with_offset(self):
        """No metadata, points show ug2 = ua + 5 → triode_connected."""
        entry = {
            "data": {"scan": {}},
            "points": [
                {"ua": 50, "ug2": 55},
                {"ua": 100, "ug2": 105},
                {"ua": 200, "ug2": 205},
            ],
        }
        assert get_ug2_mode(entry) == TOPOLOGY_TRIODE_CONNECTED

    def test_autodetect_from_points_adc_noise(self):
        """Simulated ADC noise (±2V) should still detect triode_connected."""
        entry = {
            "data": {"scan": {}},
            "points": [
                {"ua": 50, "ug2": 50.5},
                {"ua": 100, "ug2": 99.3},
                {"ua": 150, "ug2": 151.2},
                {"ua": 200, "ug2": 199.0},
                {"ua": 250, "ug2": 252.0},
            ],
        }
        assert get_ug2_mode(entry) == TOPOLOGY_TRIODE_CONNECTED

    def test_autodetect_pentode_not_tracking(self):
        """Independent Ug2 values → pentode, not triode_connected."""
        entry = {
            "data": {"scan": {}},
            "points": [
                {"ua": 50, "ug2": 200},
                {"ua": 100, "ug2": 200},
                {"ua": 150, "ug2": 200},
                {"ua": 50, "ug2": 250},
                {"ua": 100, "ug2": 250},
            ],
        }
        assert get_ug2_mode(entry) == TOPOLOGY_PENTODE

    def test_autodetect_no_points_key(self):
        """No points in entry → fallback to pentode."""
        entry = {"data": {"scan": {}}}
        assert get_ug2_mode(entry) == TOPOLOGY_PENTODE


# ---------------------------------------------------------------------------
# Tests: is_triode
# ---------------------------------------------------------------------------

class TestIsTriode:
    def test_true_for_triode_mode(self):
        entry = {"data": {"topology": TOPOLOGY_TRIODE}}
        assert is_triode(entry) is True

    def test_false_for_pentode(self):
        entry = {"data": {"scan": {"ug2_mode": TOPOLOGY_PENTODE}}}
        assert is_triode(entry) is False

    def test_false_for_triode_connected(self):
        entry = {"data": {"scan": {"ug2_mode": TOPOLOGY_TRIODE_CONNECTED}}}
        assert is_triode(entry) is False


# ---------------------------------------------------------------------------
# Tests: is_ug2_track_mode
# ---------------------------------------------------------------------------

class TestIsUg2TrackMode:
    def test_explicit_true(self):
        entry = {"data": {"scan": {"ug2_track_ua": True}}}
        assert is_ug2_track_mode(entry, []) is True

    def test_explicit_false(self):
        entry = {"data": {"scan": {"ug2_track_ua": False}}}
        assert is_ug2_track_mode(entry, []) is False

    def test_empty_points(self):
        entry = {"data": {"scan": {}}}
        assert is_ug2_track_mode(entry, []) is False

    def test_single_point(self):
        entry = {"data": {"scan": {}}}
        points = [{"ua": 100, "ug2": 100}]
        assert is_ug2_track_mode(entry, points) is False

    def test_autodetect_perfect_track(self):
        entry = {"data": {"scan": {}}}
        points = [
            {"ua": 100, "ug2": 100},
            {"ua": 150, "ug2": 150},
            {"ua": 200, "ug2": 200},
        ]
        assert is_ug2_track_mode(entry, points) is True

    def test_autodetect_track_with_offset(self):
        entry = {"data": {"scan": {}}}
        points = [
            {"ua": 100, "ug2": 110},
            {"ua": 150, "ug2": 160},
            {"ua": 200, "ug2": 210},
        ]
        assert is_ug2_track_mode(entry, points) is True

    def test_autodetect_not_track_different_counts(self):
        entry = {"data": {"scan": {}}}
        points = [
            {"ua": 100, "ug2": 200},
            {"ua": 100, "ug2": 250},
            {"ua": 150, "ug2": 200},
            {"ua": 150, "ug2": 250},
            {"ua": 200, "ug2": 200},
        ]
        # unique ua: {100, 150, 200} (3), unique ug2: {200, 250} (2) → different count → False
        assert is_ug2_track_mode(entry, points) is False

    def test_autodetect_not_track_inconsistent_offset(self):
        entry = {"data": {"scan": {}}}
        points = [
            {"ua": 100, "ug2": 110},
            {"ua": 150, "ug2": 200},
            {"ua": 200, "ug2": 210},
        ]
        assert is_ug2_track_mode(entry, points) is False

    def test_autodetect_within_tolerance(self):
        entry = {"data": {"scan": {}}}
        points = [
            {"ua": 100, "ug2": 110.0},
            {"ua": 150, "ug2": 160.9},
            {"ua": 200, "ug2": 210.5},
        ]
        assert is_ug2_track_mode(entry, points) is True

    def test_autodetect_just_outside_tolerance(self):
        """Offset varies by more than _UG2_TRACK_TOL (5V) → not tracking."""
        entry = {"data": {"scan": {}}}
        points = [
            {"ua": 100, "ug2": 110.0},   # offset = 10
            {"ua": 150, "ug2": 166.0},    # offset = 16, Δ=6 > 5V
        ]
        assert is_ug2_track_mode(entry, points) is False

    def test_explicit_flag_overrides_points(self):
        entry = {"data": {"scan": {"ug2_track_ua": False}}}
        points = [
            {"ua": 100, "ug2": 100},
            {"ua": 200, "ug2": 200},
        ]
        assert is_ug2_track_mode(entry, points) is False

    def test_no_data_key(self):
        entry = {}
        points = [
            {"ua": 100, "ug2": 100},
            {"ua": 200, "ug2": 200},
        ]
        assert is_ug2_track_mode(entry, points) is True

    def test_duplicate_ua_values_merged(self):
        entry = {"data": {"scan": {}}}
        points = [
            {"ua": 100, "ug1": -5, "ug2": 110},
            {"ua": 100, "ug1": -3, "ug2": 110},
            {"ua": 200, "ug1": -5, "ug2": 210},
            {"ua": 200, "ug1": -3, "ug2": 210},
        ]
        assert is_ug2_track_mode(entry, points) is True

    def test_autodetect_adc_noise(self):
        """Real-world ADC noise: ug2 readings vary ±2V from nominal.

        Old algorithm compared counts of unique Ua vs Ug2 — this fails
        when noise splits one nominal Ug2 into multiple distinct values.
        """
        entry = {"data": {"scan": {}}}
        points = [
            {"ua": 10, "ug1": -8, "ug2": 9.3},
            {"ua": 10, "ug1": -6, "ug2": 9.7},
            {"ua": 10, "ug1": -4, "ug2": 10.0},
            {"ua": 20, "ug1": -8, "ug2": 20.0},
            {"ua": 20, "ug1": -6, "ug2": 19.5},
            {"ua": 50, "ug1": -8, "ug2": 50.0},
            {"ua": 50, "ug1": -6, "ug2": 51.2},
            {"ua": 100, "ug1": -8, "ug2": 99.0},
            {"ua": 100, "ug1": -6, "ug2": 100.5},
        ]
        assert is_ug2_track_mode(entry, points) is True
