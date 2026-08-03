"""Tests for configurable data directory resolution."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19 import data_paths, health_measurements, health_refs, measurements  # noqa: E402
from lm19.config import LampConfig, LampRange  # noqa: E402
from lm19.constants import TOPOLOGY_PENTODE  # noqa: E402


def _make_lamp() -> LampConfig:
    """Datasheet values deliberately differ from the stored ref below, so a
    ref the resolver cannot reach shows up as different numbers, not just a
    different label."""
    return LampConfig(
        tube_type="EL84", socket="B", anodes=1, warmup_s=120,
        topology=TOPOLOGY_PENTODE, uh=6.3, ih=0.76, ug1=-7.3, ua=250.0,
        ia=36.0, ug2=250.0, ig2=5.5, s=9.0, r=40.0, k=19.0,
        ranges={"ua": LampRange(0, 250, 10)}, limits={},
    )


def _write_app_json(anchor: Path, payload: dict) -> None:
    cfg_dir = anchor / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "app.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture(autouse=True)
def _reset_warning_cache():
    data_paths._reset_warning_cache()
    yield
    data_paths._reset_warning_cache()


class TestResolveDataDir:
    def test_default_when_key_missing(self, tmp_path: Path) -> None:
        _write_app_json(tmp_path, {"some_other_key": "value"})
        result = data_paths.resolve_data_dir(tmp_path, "measurements_dir", "measurements")
        assert result == (tmp_path / "measurements").resolve()
        assert result.exists()

    def test_default_when_value_empty(self, tmp_path: Path) -> None:
        _write_app_json(tmp_path, {"measurements_dir": ""})
        result = data_paths.resolve_data_dir(tmp_path, "measurements_dir", "measurements")
        assert result == (tmp_path / "measurements").resolve()

    def test_default_when_app_json_missing(self, tmp_path: Path) -> None:
        result = data_paths.resolve_data_dir(tmp_path, "measurements_dir", "measurements")
        assert result == (tmp_path / "measurements").resolve()
        assert result.exists()

    def test_relative_path_resolves_against_anchor(self, tmp_path: Path) -> None:
        _write_app_json(tmp_path, {"measurements_dir": "data/raw"})
        result = data_paths.resolve_data_dir(tmp_path, "measurements_dir", "measurements")
        assert result == (tmp_path / "data" / "raw").resolve()
        assert result.exists()

    def test_absolute_path_used_as_is(self, tmp_path: Path) -> None:
        target = tmp_path / "elsewhere" / "scans"
        _write_app_json(tmp_path, {"measurements_dir": str(target)})
        result = data_paths.resolve_data_dir(tmp_path, "measurements_dir", "measurements")
        assert result == target.resolve()
        assert result.exists()

    def test_auto_create_logs_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="lm19.data_paths"):
            data_paths.resolve_data_dir(tmp_path, "measurements_dir", "measurements")
        assert any("auto-created" in r.getMessage() for r in caplog.records)

    def test_warning_fires_once_per_path(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="lm19.data_paths"):
            data_paths.resolve_data_dir(tmp_path, "measurements_dir", "measurements")
            data_paths.resolve_data_dir(tmp_path, "measurements_dir", "measurements")
        warns = [r for r in caplog.records if "auto-created" in r.getMessage()]
        assert len(warns) == 1

    def test_no_warning_when_dir_exists(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        (tmp_path / "measurements").mkdir()
        with caplog.at_level(logging.WARNING, logger="lm19.data_paths"):
            data_paths.resolve_data_dir(tmp_path, "measurements_dir", "measurements")
        assert not any("auto-created" in r.getMessage() for r in caplog.records)

    def test_corrupt_json_falls_back_to_default(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "app.json").write_text("{ broken", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="lm19.data_paths"):
            result = data_paths.resolve_data_dir(tmp_path, "measurements_dir", "measurements")
        assert result == (tmp_path / "measurements").resolve()
        assert any("Failed to read" in r.getMessage() for r in caplog.records)


class TestMeasurementsIntegration:
    def test_save_uses_configured_relative_path(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(measurements, "_root", lambda: tmp_path)
        _write_app_json(tmp_path, {"measurements_dir": "custom_meas"})

        point = {"ua": 250.0, "ug1": -8.0, "ia": 20.0}
        path = measurements.save_measurement(
            "EL84",
            "L1",
            {
                "timestamp": "2026-05-12T10:00:00",
                "tube_type": "EL84",
                "lamp_id": "L1",
                "name": "rel-test",
                "points": [point],
            },
        )
        assert (tmp_path / "custom_meas" / "EL84").exists()
        assert path.is_file()
        assert "custom_meas" in str(path)

    def test_save_uses_absolute_path(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(measurements, "_root", lambda: tmp_path)
        external = tmp_path / "external_root"
        _write_app_json(tmp_path, {"measurements_dir": str(external)})

        path = measurements.save_measurement(
            "6N3P",
            "L9",
            {
                "timestamp": "2026-05-12T10:00:00",
                "tube_type": "6N3P",
                "lamp_id": "L9",
                "name": "abs-test",
                "points": [{"ua": 100.0, "ug1": -1.0, "ia": 5.0}],
            },
        )
        assert external.exists()
        assert "external_root" in str(path)


class TestHealthMeasurementsIntegration:
    def test_save_uses_configured_relative_path(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(health_measurements, "_root", lambda: tmp_path)
        _write_app_json(tmp_path, {"health_measurements_dir": "custom_health"})

        path = health_measurements.save_health_measurement(
            "EL84",
            "L1",
            {
                "timestamp": "2026-05-12T10:00:00",
                "tube_type": "EL84",
                "lamp_id": "L1",
                "name": "h-rel",
                "health": {"index": 80.0},
            },
        )
        assert (tmp_path / "custom_health" / "EL84").exists()
        assert "custom_health" in str(path)

    def test_save_uses_absolute_path(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(health_measurements, "_root", lambda: tmp_path)
        external = tmp_path / "external_health"
        _write_app_json(tmp_path, {"health_measurements_dir": str(external)})

        path = health_measurements.save_health_measurement(
            "EL84",
            "L1",
            {
                "timestamp": "2026-05-12T10:00:00",
                "tube_type": "EL84",
                "lamp_id": "L1",
                "name": "h-abs",
                "health": {"index": 80.0},
            },
        )
        assert external.exists()
        assert "external_health" in str(path)


class TestHealthRefsIntegration:
    """Call-site pins: both ref halves must route through the config key.

    ``type`` and ``personal`` are twins with separate dir helpers — a fix
    applied to one says nothing about the other, so each is pinned on its
    own. The custom-path cases are what discriminate: with the key honoured
    the file lands under the configured root, with the old hardcoded
    ``<anchor>/config/health_refs`` it does not.
    """

    def test_default_keeps_legacy_layout_when_key_missing(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Installs predating the key keep their existing refs."""
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        _write_app_json(tmp_path, {"some_other_key": "value"})

        type_path = health_refs.save_type_ref("EL84", "nos_a", {"id": "nos_a"})
        pers_path = health_refs.save_personal_baseline("EL84", "L1", {"lamp_id": "L1"})

        assert type_path == tmp_path / "config" / "health_refs" / "type" / "EL84" / "nos_a.json"
        assert pers_path == tmp_path / "config" / "health_refs" / "personal" / "EL84" / "L1.json"

    def test_type_ref_uses_configured_relative_path(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        _write_app_json(tmp_path, {"health_refs_dir": "custom_refs"})

        path = health_refs.save_type_ref(
            "EL84", "nos_a", {"id": "nos_a", "reference": {"ia": 48.0}})

        assert path == tmp_path / "custom_refs" / "type" / "EL84" / "nos_a.json"
        assert not (tmp_path / "config" / "health_refs").exists()
        # Reads follow writes — the whole module shares one root.
        assert [r["id"] for r in health_refs.list_type_refs("EL84")] == ["nos_a"]

    def test_type_ref_uses_absolute_path(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        external = tmp_path / "external_refs"
        _write_app_json(tmp_path, {"health_refs_dir": str(external)})

        health_refs.save_type_ref("EL84", "nos_a", {"id": "nos_a"})

        assert (external / "type" / "EL84" / "nos_a.json").is_file()
        assert health_refs.load_type_ref("EL84", "nos_a") == {"id": "nos_a"}

    def test_personal_baseline_uses_configured_relative_path(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        _write_app_json(tmp_path, {"health_refs_dir": "custom_refs"})

        path = health_refs.save_personal_baseline(
            "EL84", "L1", {"lamp_id": "L1", "reference": {"ia": 45.0}})

        assert path == tmp_path / "custom_refs" / "personal" / "EL84" / "L1.json"
        assert not (tmp_path / "config" / "health_refs").exists()
        loaded = health_refs.load_personal_baseline("EL84", "L1")
        assert loaded is not None and loaded["reference"]["ia"] == 45.0

    def test_personal_baseline_uses_absolute_path(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        external = tmp_path / "external_refs"
        _write_app_json(tmp_path, {"health_refs_dir": str(external)})

        health_refs.save_personal_baseline("EL84", "L1", {"lamp_id": "L1"})

        assert (external / "personal" / "EL84" / "L1.json").is_file()

    def test_resolve_reference_reads_configured_root(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The scoring base itself must come from the configured root.

        Pinned separately from the save helpers: resolution walks its own
        lookup chain, and a relocated root that resolution cannot see would
        silently fall back to the datasheet — a shifted health-score base
        rather than a visible failure.
        """
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        _write_app_json(tmp_path, {"health_refs_dir": "custom_refs"})
        health_refs.save_type_ref(
            "EL84", "nos_a",
            {"id": "nos_a", "active": True, "reference": {"ia": 48.0, "s": 11.0}})

        ref = health_refs.resolve_reference(
            "type", "EL84", "L1", None, _make_lamp())

        assert ref["id"] == "nos_a"
        assert ref["reference"]["ia"] == 48.0
        assert ref["reference"]["s"] == 11.0

    def test_defaults_preserved_when_keys_absent(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(health_measurements, "_root", lambda: tmp_path)
        # No app.json at all
        path = health_measurements.save_health_measurement(
            "EL84",
            "L1",
            {
                "timestamp": "2026-05-12T10:00:00",
                "tube_type": "EL84",
                "lamp_id": "L1",
                "name": "h-default",
                "health": {"index": 80.0},
            },
        )
        assert (tmp_path / "health_measurements" / "EL84").exists()
        assert "health_measurements" in str(path)
