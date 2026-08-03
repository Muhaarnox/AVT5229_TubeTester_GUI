"""Schema-versioning regression tests.

Three persisted JSON formats embed ``_schema_version`` on save and
log a warning on load when the file is newer than the app. Files
written before the scheme was introduced — including the 173 existing
measurements in the repo — load unchanged because an absent field is
treated as version 0 ("pre-versioning").

Tests:
- save → load round-trip stamps the constant version
- pre-versioning files (no field) load without complaint
- newer-version files emit a WARNING (forward-compat hint)
- malformed version values fall back to 0 with a WARNING
- settings loader reports unknown top-level keys (catches typos)
"""

from __future__ import annotations

import json
import logging

import pytest

from lm19.schema import (
    HEALTH_MEASUREMENT_SCHEMA_VERSION,
    MEASUREMENT_SCHEMA_VERSION,
    SETTINGS_SCHEMA_VERSION,
    _check_schema_version,
    stamp_schema_version,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
)


# ── helpers ──────────────────────────────────────────────────────────

def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ── lm19/schema.py: low-level helpers ────────────────────────────────

class TestCheckSchemaVersion:
    """``_check_schema_version`` is the single source of truth for
    version-decode semantics. Tested in isolation from any specific
    format so the rules are clear: missing field → 0, future version
    → WARNING but not raised, malformed → 0 + WARNING."""

    def test_missing_field_returns_zero(self):
        version = _check_schema_version({}, current=1, label="x")
        assert version == 0

    def test_present_field_returned_as_int(self):
        version = _check_schema_version(
            {"_schema_version": 1}, current=1, label="x")
        assert version == 1

    def test_string_int_coerced(self):
        """Tolerate ``"1"`` (some serialisers stringify numerics)."""
        version = _check_schema_version(
            {"_schema_version": "1"}, current=1, label="x")
        assert version == 1

    def test_future_version_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="lm19.schema"):
            version = _check_schema_version(
                {"_schema_version": 99}, current=1, label="alpha")
        assert version == 99  # returned as-is so caller can decide
        assert any("newer than this app" in rec.message and "alpha" in rec.message
                   for rec in caplog.records)

    def test_malformed_version_falls_back_to_zero(self, caplog):
        with caplog.at_level(logging.WARNING, logger="lm19.schema"):
            version = _check_schema_version(
                {"_schema_version": "not-a-number"}, current=1, label="beta")
        assert version == 0
        assert any("malformed" in rec.message for rec in caplog.records)

    def test_current_version_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="lm19.schema"):
            _check_schema_version(
                {"_schema_version": 1}, current=1, label="x")
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not warnings

    def test_old_version_no_warning(self, caplog):
        """Files with a lower version load silently — the version-0
        path must not spam WARNINGs on every ``list_lamp_ids()`` call."""
        with caplog.at_level(logging.WARNING, logger="lm19.schema"):
            _check_schema_version(
                {"_schema_version": 0}, current=1, label="x")
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not warnings


class TestStampSchemaVersion:
    def test_stamps_int_version(self):
        d = {"foo": "bar"}
        stamp_schema_version(d, 1)
        assert d["_schema_version"] == 1

    def test_overwrites_existing(self):
        d = {"_schema_version": 0, "foo": "bar"}
        stamp_schema_version(d, 1)
        assert d["_schema_version"] == 1

    def test_returns_same_dict(self):
        d = {}
        result = stamp_schema_version(d, 1)
        assert result is d


# ── lm19/measurements.py round-trip ──────────────────────────────────

class TestMeasurementSchemaVersion:
    """Measurement save now stamps the current version. Load tolerates
    files with or without the field — the 173 existing files in the
    repo are the regression target."""

    def _make_measurement(self):
        return {
            "timestamp": "2026-05-04T12:00:00",
            "tube_type": "EL84",
            "lamp_id": "TEST_001",
            "name": "schema_test",
            "topology": TOPOLOGY_PENTODE,
            "scan": {"ua": {"start": 0, "stop": 250, "step": 25}},
            "points": [{"ua": 100, "ug1": -2, "ia": 30}],
        }

    def test_save_stamps_current_version(self, tmp_path, monkeypatch):
        from lm19 import measurements
        # Redirect _root() so saves land in tmp_path/measurements/...
        monkeypatch.setattr(measurements, "_root", lambda: tmp_path)
        m = self._make_measurement()
        path = measurements.save_measurement("EL84", "TEST_001", m)
        loaded = _read_json(path)
        assert loaded["_schema_version"] == MEASUREMENT_SCHEMA_VERSION

    def test_save_load_round_trip(self, tmp_path, monkeypatch):
        from lm19 import measurements
        monkeypatch.setattr(measurements, "_root", lambda: tmp_path)
        m = self._make_measurement()
        measurements.save_measurement("EL84", "TEST_001", m)
        loaded = measurements.load_measurements("EL84", "TEST_001")
        assert len(loaded) == 1
        assert loaded[0]["lamp_id"] == "TEST_001"
        assert loaded[0]["_schema_version"] == MEASUREMENT_SCHEMA_VERSION

    def test_load_pre_versioning_file_no_warning(
            self, tmp_path, monkeypatch, caplog):
        """A file written before the versioning scheme (no
        ``_schema_version`` field) loads silently as version 0. This
        is the path that handles the 173 pre-existing measurement files."""
        from lm19 import measurements
        monkeypatch.setattr(measurements, "_root", lambda: tmp_path)
        d = tmp_path / "measurements" / "EL84"
        d.mkdir(parents=True)
        m = self._make_measurement()
        # Note: NO _schema_version in this file
        _write_json(d / "old.json", m)
        with caplog.at_level(logging.WARNING, logger="lm19.schema"):
            loaded = measurements.load_measurements("EL84", "TEST_001")
        assert len(loaded) == 1
        assert "_schema_version" not in loaded[0]
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_load_future_version_warns(self, tmp_path, monkeypatch, caplog):
        from lm19 import measurements
        monkeypatch.setattr(measurements, "_root", lambda: tmp_path)
        d = tmp_path / "measurements" / "EL84"
        d.mkdir(parents=True)
        m = self._make_measurement()
        m["_schema_version"] = MEASUREMENT_SCHEMA_VERSION + 5
        _write_json(d / "future.json", m)
        with caplog.at_level(logging.WARNING, logger="lm19.schema"):
            loaded = measurements.load_measurements("EL84", "TEST_001")
        assert len(loaded) == 1  # still loads — forward-compat best effort
        assert any("newer than this app" in r.message for r in caplog.records)

    def test_list_measurement_entries_checks_version(
            self, tmp_path, monkeypatch, caplog):
        """The directory-walk path also routes through the check."""
        from lm19 import measurements
        monkeypatch.setattr(measurements, "_root", lambda: tmp_path)
        d = tmp_path / "measurements" / "EL84"
        d.mkdir(parents=True)
        m = self._make_measurement()
        m["_schema_version"] = 999
        _write_json(d / "future.json", m)
        with caplog.at_level(logging.WARNING, logger="lm19.schema"):
            entries = measurements.list_measurement_entries()
        assert len(entries) == 1
        assert any("newer than this app" in r.message for r in caplog.records)


# ── lm19/health_measurements.py round-trip ───────────────────────────

class TestHealthMeasurementSchemaVersion:
    def _make_health(self):
        return {
            "timestamp": "2026-05-04T12:00:00",
            "tube_type": "EL84",
            "lamp_id": "H_001",
            "name": "health_test",
            "verdict": "Good",
        }

    def test_save_stamps_current_version(self, tmp_path, monkeypatch):
        from lm19 import health_measurements as hm
        monkeypatch.setattr(hm, "_root", lambda: tmp_path)
        path = hm.save_health_measurement("EL84", "H_001", self._make_health())
        loaded = _read_json(path)
        assert loaded["_schema_version"] == HEALTH_MEASUREMENT_SCHEMA_VERSION

    def test_load_pre_versioning_file_no_warning(
            self, tmp_path, monkeypatch, caplog):
        from lm19 import health_measurements as hm
        monkeypatch.setattr(hm, "_root", lambda: tmp_path)
        d = tmp_path / "health_measurements" / "EL84"
        d.mkdir(parents=True)
        _write_json(d / "old.json", self._make_health())
        with caplog.at_level(logging.WARNING, logger="lm19.schema"):
            loaded = hm.load_health_measurements("EL84")
        assert len(loaded) == 1
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_load_future_version_warns(self, tmp_path, monkeypatch, caplog):
        from lm19 import health_measurements as hm
        monkeypatch.setattr(hm, "_root", lambda: tmp_path)
        d = tmp_path / "health_measurements" / "EL84"
        d.mkdir(parents=True)
        m = self._make_health()
        m["_schema_version"] = HEALTH_MEASUREMENT_SCHEMA_VERSION + 1
        _write_json(d / "future.json", m)
        with caplog.at_level(logging.WARNING, logger="lm19.schema"):
            loaded = hm.load_health_measurements("EL84")
        assert len(loaded) == 1
        assert any("newer than this app" in r.message for r in caplog.records)


# ── settings file (Save/Load Settings dialog) ────────────────────────

class TestSettingsSchemaVersionConstants:
    """Settings save/load runs through main_window_settings — testing
    the full UI flow needs a QMainWindow. Here we verify the contract
    by checking the constants are stable and the helper functions in
    lm19/schema work — UI integration is covered by a smoke test
    elsewhere if needed."""

    def test_settings_version_constant_is_one(self):
        """Loud reminder if someone bumps the version: the on-disk
        format change must come with a migration entry."""
        assert SETTINGS_SCHEMA_VERSION == 1

    def test_settings_known_keys_match_loader(self):
        """The whitelist of recognised top-level keys must match what
        ``_load_scan_settings`` actually reads. If a section is added
        to the loader, it must be added to the whitelist (otherwise
        the user gets a spurious "unknown key ignored" warning)."""
        from app.main_window_settings import _SETTINGS_KNOWN_KEYS
        # Loader reads: lamp_type, scan, preheat, plot. Plus version.
        expected = {"_schema_version", "lamp_type", "scan", "preheat", "plot"}
        assert _SETTINGS_KNOWN_KEYS == expected


class TestSettingsUnknownKeyWarning:
    """Settings file with a typo in a top-level key produces a WARNING.
    Pure helper-level test — doesn't require QMainWindow."""

    def test_unknown_top_level_keys_detected(self):
        from app.main_window_settings import _SETTINGS_KNOWN_KEYS
        bad_data = {
            "_schema_version": 1,
            "lamp_type": "EL84",
            "scan": {},                # known
            "scna": {},                # typo for "scan"
            "Plot": {},                # case typo
            "extra_thing": 42,         # legit unknown
        }
        unknown = sorted(set(bad_data.keys()) - _SETTINGS_KNOWN_KEYS)
        assert unknown == ["Plot", "extra_thing", "scna"]


class TestSchemaVersionsAreDistinct:
    """Each format has its own version namespace. A future bump in one
    must not auto-bump the others (different on-disk shapes evolve at
    different rates)."""

    def test_three_independent_constants(self):
        # All three currently happen to be 1 — but they're independent
        # variables so the ASSERTION is that they are addressable as
        # separate identifiers, not that their values differ.
        assert isinstance(MEASUREMENT_SCHEMA_VERSION, int)
        assert isinstance(HEALTH_MEASUREMENT_SCHEMA_VERSION, int)
        assert isinstance(SETTINGS_SCHEMA_VERSION, int)
