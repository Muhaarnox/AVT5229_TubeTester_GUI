"""Error-handling tests for config / refs / measurement loaders.

Verifies that corrupted or missing files don't crash and return
safe defaults.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lm19.config import load_device_limits, DEFAULT_LIMITS
from lm19.health_refs import (
    list_type_refs,
    load_type_ref,
    load_personal_baseline,
)


# ---------------------------------------------------------------------------
# config.py — load_device_limits
# ---------------------------------------------------------------------------

class TestLoadDeviceLimitsErrorHandling:
    def test_corrupted_device_json(self, tmp_path, monkeypatch):
        """Corrupted device.json should return DEFAULT_LIMITS without crashing."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "device.json").write_text("{bad json!!!", encoding="utf-8")
        monkeypatch.setattr("lm19.config._resolve_paths", lambda: tmp_path)
        result = load_device_limits()
        assert result == DEFAULT_LIMITS

    def test_missing_device_json(self, tmp_path, monkeypatch):
        """Missing device.json should return DEFAULT_LIMITS."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        # No device.json file
        monkeypatch.setattr("lm19.config._resolve_paths", lambda: tmp_path)
        result = load_device_limits()
        assert result == DEFAULT_LIMITS

    def test_valid_device_json_merges(self, tmp_path, monkeypatch):
        """Valid device.json should merge into DEFAULT_LIMITS."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "device.json").write_text(
            json.dumps({"ua_max": 999}), encoding="utf-8"
        )
        monkeypatch.setattr("lm19.config._resolve_paths", lambda: tmp_path)
        result = load_device_limits()
        assert result["ua_max"] == 999
        # Other defaults still present
        for key in DEFAULT_LIMITS:
            assert key in result


# ---------------------------------------------------------------------------
# health_refs.py — corrupted ref files
# ---------------------------------------------------------------------------

class TestHealthRefsErrorHandling:
    def _setup_refs_dir(self, tmp_path, monkeypatch):
        """Create a config/health_refs/type/<tube>/ directory structure."""
        refs_dir = tmp_path / "config" / "health_refs" / "type" / "6L6"
        refs_dir.mkdir(parents=True)
        monkeypatch.setattr("lm19.health_refs._root", lambda: tmp_path)
        return refs_dir

    def _setup_personal_dir(self, tmp_path, monkeypatch):
        """Create a config/health_refs/personal/<tube>/ directory structure."""
        personal_dir = tmp_path / "config" / "health_refs" / "personal" / "6L6"
        personal_dir.mkdir(parents=True)
        monkeypatch.setattr("lm19.health_refs._root", lambda: tmp_path)
        return personal_dir

    def test_list_type_refs_skips_corrupted(self, tmp_path, monkeypatch):
        """Corrupted JSON in type refs should be skipped, not crash."""
        refs_dir = self._setup_refs_dir(tmp_path, monkeypatch)
        (refs_dir / "good.json").write_text(
            json.dumps({"ref_id": "good", "active": True, "timestamp": "2024-01-01"}),
            encoding="utf-8",
        )
        (refs_dir / "bad.json").write_text("NOT JSON{{{", encoding="utf-8")
        result = list_type_refs("6L6")
        # Should get 1 valid ref, bad one skipped
        valid_ids = [r.get("ref_id") for r in result]
        assert "good" in valid_ids
        assert len(result) == 1

    def test_load_type_ref_corrupted_returns_none(self, tmp_path, monkeypatch):
        """Corrupted type ref file should return None."""
        refs_dir = self._setup_refs_dir(tmp_path, monkeypatch)
        (refs_dir / "broken.json").write_text("{{{invalid", encoding="utf-8")
        result = load_type_ref("6L6", "broken")
        assert result is None

    def test_load_type_ref_missing_returns_none(self, tmp_path, monkeypatch):
        """Missing type ref file should return None."""
        self._setup_refs_dir(tmp_path, monkeypatch)
        result = load_type_ref("6L6", "nonexistent")
        assert result is None

    def test_load_personal_baseline_corrupted_returns_none(self, tmp_path, monkeypatch):
        """Corrupted personal baseline should return None."""
        personal_dir = self._setup_personal_dir(tmp_path, monkeypatch)
        (personal_dir / "L1.json").write_text("bad json!", encoding="utf-8")
        result = load_personal_baseline("6L6", "L1")
        assert result is None

    def test_load_personal_baseline_missing_returns_none(self, tmp_path, monkeypatch):
        """Missing personal baseline should return None."""
        self._setup_personal_dir(tmp_path, monkeypatch)
        result = load_personal_baseline("6L6", "L1")
        assert result is None
