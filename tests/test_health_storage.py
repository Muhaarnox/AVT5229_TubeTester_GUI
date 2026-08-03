"""Smoke/unit tests for Tube Health storage helpers."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19 import health_measurements, health_refs  # noqa: E402


@pytest.mark.smoke
def test_health_measurements_save_and_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(health_measurements, "_root", lambda: tmp_path)

    m1 = {
        "timestamp": "2026-02-22T12:40:00",
        "tube_type": "EL84",
        "lamp_id": "L1",
        "name": "main",
        "health": {"index": 87.0},
    }
    m2 = {
        "timestamp": "2026-02-22T12:45:00",
        "tube_type": "EL84",
        "lamp_id": "L2",
        "name": "pairB",
        "health": {"index": 91.0},
    }

    health_measurements.save_health_measurement("EL84", "L1", m1)
    health_measurements.save_health_measurement("EL84", "L2", m2)

    all_el84 = health_measurements.load_health_measurements("EL84")
    assert len(all_el84) == 2
    # Newest first
    assert all_el84[0]["lamp_id"] == "L2"

    only_l1 = health_measurements.load_health_measurements("EL84", "L1")
    assert len(only_l1) == 1
    assert only_l1[0]["name"] == "main"

    lamp_ids = health_measurements.list_health_lamp_ids("EL84")
    assert lamp_ids == ["L1", "L2"]


@pytest.mark.smoke
def test_save_duplicate_name_gets_counter(tmp_path, monkeypatch):
    monkeypatch.setattr(health_measurements, "_root", lambda: tmp_path)
    m = {"timestamp": "2026-03-01T10:00:00", "tube_type": "6P18P",
         "lamp_id": "L1", "name": "dup"}
    p1 = health_measurements.save_health_measurement("6P18P", "L1", m)
    p2 = health_measurements.save_health_measurement("6P18P", "L1", m)
    assert p1 != p2
    assert p1.exists()
    assert p2.exists()
    assert "_1" in p2.stem


@pytest.mark.smoke
def test_list_health_entries_has_file_path(tmp_path, monkeypatch):
    monkeypatch.setattr(health_measurements, "_root", lambda: tmp_path)
    m = {"timestamp": "2026-03-01T10:00:00", "tube_type": "EL84",
         "lamp_id": "L1", "name": "fp_test", "health": {"index": 80}}
    health_measurements.save_health_measurement("EL84", "L1", m)
    entries = health_measurements.list_health_entries("EL84")
    assert len(entries) == 1
    assert "_file_path" in entries[0]
    assert entries[0]["_file_path"].endswith(".json")


@pytest.mark.smoke
def test_list_health_entries_empty_root(tmp_path, monkeypatch):
    monkeypatch.setattr(health_measurements, "_root", lambda: tmp_path)
    entries = health_measurements.list_health_entries("NonExistent")
    assert entries == []


@pytest.mark.smoke
def test_delete_health_measurement(tmp_path, monkeypatch):
    monkeypatch.setattr(health_measurements, "_root", lambda: tmp_path)
    m = {"timestamp": "2026-03-01T11:00:00", "tube_type": "EL84",
         "lamp_id": "L3", "name": "del_test"}
    saved_path = health_measurements.save_health_measurement("EL84", "L3", m)
    assert saved_path.exists()

    result = health_measurements.delete_health_measurement(str(saved_path))
    assert result is True
    assert not saved_path.exists()

    all_entries = health_measurements.load_health_measurements("EL84")
    assert len(all_entries) == 0


@pytest.mark.smoke
def test_delete_nonexistent_returns_false(tmp_path):
    result = health_measurements.delete_health_measurement(str(tmp_path / "no_such_file.json"))
    assert result is False


@pytest.mark.smoke
def test_delete_non_json_refused(tmp_path):
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("hello")
    result = health_measurements.delete_health_measurement(str(bad_file))
    assert result is False
    assert bad_file.exists()


@pytest.mark.smoke
def test_load_ignores_corrupt_json(tmp_path, monkeypatch):
    monkeypatch.setattr(health_measurements, "_root", lambda: tmp_path)
    tube_dir = tmp_path / "health_measurements" / "EL84"
    tube_dir.mkdir(parents=True)
    (tube_dir / "good.json").write_text(
        '{"timestamp":"2026-03-01","tube_type":"EL84","lamp_id":"L1","name":"ok"}')
    (tube_dir / "bad.json").write_text('NOT JSON{{{')
    entries = health_measurements.list_health_entries("EL84")
    assert len(entries) == 1
    assert entries[0]["name"] == "ok"


@pytest.mark.smoke
def test_health_refs_type_active_and_personal(tmp_path, monkeypatch):
    monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)

    ref_a = {
        "id": "nos_a",
        "label": "NOS A",
        "tube_type": "EL84",
        "active": True,
        "timestamp": "2026-02-20T10:11:12",
        "reference": {"ia": 48.5, "s": 10.9},
    }
    ref_b = {
        "id": "batch_b",
        "label": "Batch B",
        "tube_type": "EL84",
        "active": False,
        "timestamp": "2026-02-22T09:00:00",
        "reference": {"ia": 47.9, "s": 10.6},
    }
    health_refs.save_type_ref("EL84", "nos_a", ref_a)
    health_refs.save_type_ref("EL84", "batch_b", ref_b)

    refs = health_refs.list_type_refs("EL84")
    assert len(refs) == 2

    active = health_refs.get_active_type_ref("EL84")
    assert active is not None
    assert active["id"] == "nos_a"

    health_refs.set_active_type_ref("EL84", "batch_b")
    active2 = health_refs.get_active_type_ref("EL84")
    assert active2 is not None
    assert active2["id"] == "batch_b"

    baseline = {
        "tube_type": "EL84",
        "lamp_id": "L1",
        "timestamp": "2026-02-21T09:00:00",
        "name": "EL84_L1_main",
        "reference": {"ia": 47.2, "s": 10.1},
    }
    health_refs.save_personal_baseline("EL84", "L1", baseline)
    loaded = health_refs.load_personal_baseline("EL84", "L1")
    assert loaded is not None
    assert loaded["name"] == "EL84_L1_main"
