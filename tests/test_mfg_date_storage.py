"""Tests for mfg_date round-trip through measurements & health storage."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19 import measurements, health_measurements  # noqa: E402


# ---------------------------------------------------------------------------
# Scan measurements
# ---------------------------------------------------------------------------

class TestScanMfgDate:
    def test_save_and_load_with_mfg_date(self, tmp_path, monkeypatch):
        monkeypatch.setattr(measurements, "_root", lambda: tmp_path)
        m = {
            "timestamp": "2026-05-11T12:00:00",
            "tube_type": "EL84",
            "lamp_id": "L1",
            "name": "scan",
            "mfg_date": "1972-05",
            "points": [{"ua": 250, "ug1": -7, "ug2": 250, "ia": 48, "ig2": 5}],
        }
        measurements.save_measurement("EL84", "L1", m)
        loaded = measurements.load_measurements("EL84", "L1")
        assert len(loaded) == 1
        assert loaded[0]["mfg_date"] == "1972-05"

    def test_save_without_mfg_date_loads_without_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr(measurements, "_root", lambda: tmp_path)
        m = {
            "timestamp": "2026-05-11T12:00:00",
            "tube_type": "EL84",
            "lamp_id": "L1",
            "name": "scan",
            "points": [{"ua": 250, "ug1": -7, "ug2": 250, "ia": 48, "ig2": 5}],
        }
        measurements.save_measurement("EL84", "L1", m)
        loaded = measurements.load_measurements("EL84", "L1")
        assert "mfg_date" not in loaded[0]

    def test_list_measurement_entries_includes_mfg_date(self, tmp_path, monkeypatch):
        monkeypatch.setattr(measurements, "_root", lambda: tmp_path)
        m = {
            "timestamp": "2026-05-11T12:00:00",
            "tube_type": "EL84",
            "lamp_id": "L1",
            "name": "scan",
            "mfg_date": "1965-11",
            "points": [{"ua": 250, "ug1": -7, "ug2": 250, "ia": 48, "ig2": 5}],
        }
        measurements.save_measurement("EL84", "L1", m)
        entries = measurements.list_measurement_entries()
        assert len(entries) == 1
        assert entries[0]["mfg_date"] == "1965-11"

    def test_list_measurement_entries_empty_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(measurements, "_root", lambda: tmp_path)
        m = {
            "timestamp": "2026-05-11T12:00:00",
            "tube_type": "EL84",
            "lamp_id": "L1",
            "name": "scan",
            "points": [{"ua": 250, "ug1": -7, "ug2": 250, "ia": 48, "ig2": 5}],
        }
        measurements.save_measurement("EL84", "L1", m)
        entries = measurements.list_measurement_entries()
        assert entries[0]["mfg_date"] == ""


# ---------------------------------------------------------------------------
# Health measurements
# ---------------------------------------------------------------------------

class TestHealthMfgDate:
    def test_save_and_load_with_mfg_date(self, tmp_path, monkeypatch):
        monkeypatch.setattr(health_measurements, "_root", lambda: tmp_path)
        m = {
            "timestamp": "2026-05-11T12:00:00",
            "tube_type": "EL84",
            "lamp_id": "L1",
            "name": "main",
            "mfg_date": "1980-03",
            "health": {"index": 87.0},
        }
        health_measurements.save_health_measurement("EL84", "L1", m)
        loaded = health_measurements.load_health_measurements("EL84", "L1")
        assert len(loaded) == 1
        assert loaded[0]["mfg_date"] == "1980-03"

    def test_list_health_entries_propagates_mfg_date(self, tmp_path, monkeypatch):
        monkeypatch.setattr(health_measurements, "_root", lambda: tmp_path)
        m = {
            "timestamp": "2026-05-11T12:00:00",
            "tube_type": "EL84",
            "lamp_id": "L1",
            "name": "main",
            "mfg_date": "1980-03",
            "health": {"index": 87.0},
        }
        health_measurements.save_health_measurement("EL84", "L1", m)
        entries = health_measurements.list_health_entries("EL84")
        assert len(entries) == 1
        # Health entries are raw data dicts — mfg_date at root.
        assert entries[0].get("mfg_date") == "1980-03"


# ---------------------------------------------------------------------------
# Backwards compat: old files without mfg_date
# ---------------------------------------------------------------------------

class TestBackwardsCompat:
    def test_legacy_scan_file_loads_cleanly(self, tmp_path, monkeypatch):
        """A pre-mfg_date JSON file must still load without errors."""
        monkeypatch.setattr(measurements, "_root", lambda: tmp_path)
        legacy = {
            "timestamp": "2020-01-01T00:00:00",
            "tube_type": "EL84",
            "lamp_id": "L_old",
            "name": "legacy",
            "points": [{"ua": 200, "ug1": -6, "ug2": 200, "ia": 40, "ig2": 4}],
        }
        measurements.save_measurement("EL84", "L_old", legacy)
        entries = measurements.list_measurement_entries()
        assert entries[0]["mfg_date"] == ""
        assert entries[0]["lamp_id"] == "L_old"
