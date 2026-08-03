"""Tests for health reference resolution and payload construction.

Covers ``lm19.health_refs.resolve_reference`` and
``build_reference_from_measurement`` — pure helpers used by
``HealthTab`` for reference lookup and persistence.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19 import health_refs  # noqa: E402
from lm19.health_refs import resolve_reference as _resolve_reference  # noqa: E402
from lm19.health_refs import build_reference_from_measurement as _build_reference_from_measurement  # noqa: E402
from lm19.config import LampConfig, LampRange  # noqa: E402
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_lamp(topology=TOPOLOGY_PENTODE) -> LampConfig:
    return LampConfig(
        tube_type="EL84",
        socket="B",
        anodes=1,
        warmup_s=120,
        topology=topology,
        uh=6.3,
        ih=0.76,
        ug1=-7.3,
        ua=250.0,
        ia=48.0,
        ug2=250.0 if topology != TOPOLOGY_TRIODE else 0.0,
        ig2=5.5 if topology != TOPOLOGY_TRIODE else 0.0,
        s=11.0,
        r=40.0,
        k=19.0,
        ranges={"ua": LampRange(0, 250, 10), "ug1": LampRange(-20, 0, 1), "ug2": LampRange(0, 250, 10)},
        limits={},
    )


# ---------------------------------------------------------------------------
# Tests: _resolve_reference
# ---------------------------------------------------------------------------

class TestResolveReferenceDatasheet:
    def test_datasheet_mode_pentode(self):
        lamp = _make_lamp("pentode")
        ref = _resolve_reference("datasheet", "EL84", "L1", None, lamp)
        assert ref["source"] == "datasheet"
        assert ref["reference"]["ia"] == 48.0
        assert ref["reference"]["s"] == 11.0
        assert ref["reference"]["r"] == 40.0
        assert ref["reference"]["k"] == 19.0
        assert ref["reference"]["rh"] == pytest.approx(6.3 / 0.76)
        assert ref["reference"]["screen_ratio"] == pytest.approx(5.5 / 48.0)

    def test_datasheet_mode_triode(self):
        lamp = _make_lamp("triode")
        ref = _resolve_reference("datasheet", "EL84", "L1", None, lamp)
        assert ref["source"] == "datasheet"
        assert ref["reference"]["screen_ratio"] == pytest.approx(0.0)  # ig2=0, ia=48 → 0/48
        assert ref["reference"]["rh"] == pytest.approx(6.3 / 0.76)

    def test_datasheet_triode_zero_ia(self):
        lamp = _make_lamp("triode")
        lamp.ia = 0.0
        ref = _resolve_reference("datasheet", "EL84", "L1", None, lamp)
        assert ref["reference"]["screen_ratio"] is None

    def test_datasheet_zero_ih(self):
        lamp = _make_lamp("pentode")
        lamp.ih = 0.0
        ref = _resolve_reference("datasheet", "EL84", "L1", None, lamp)
        assert ref["reference"]["rh"] is None


class TestResolveReferencePersonal:
    def test_personal_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        baseline = {
            "tube_type": "EL84", "lamp_id": "L1",
            "reference": {"ia": 47.0, "s": 10.5},
            "source": "measured",
        }
        health_refs.save_personal_baseline("EL84", "L1", baseline)
        lamp = _make_lamp()
        ref = _resolve_reference("personal", "EL84", "L1", None, lamp)
        assert ref["reference"]["ia"] == 47.0
        assert ref["source"] == "measured"

    def test_personal_not_found_falls_back_to_type(self, tmp_path, monkeypatch):
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        type_ref = {
            "id": "batch1", "active": True,
            "reference": {"ia": 46.0, "s": 10.2},
        }
        health_refs.save_type_ref("EL84", "batch1", type_ref)
        lamp = _make_lamp()
        ref = _resolve_reference("personal", "EL84", "L1", None, lamp)
        assert ref["id"] == "batch1"
        assert ref["reference"]["ia"] == 46.0

    def test_personal_and_type_both_missing_falls_to_datasheet(self, tmp_path, monkeypatch):
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        lamp = _make_lamp()
        ref = _resolve_reference("personal", "EL84", "L1", None, lamp)
        assert ref["source"] == "datasheet"


class TestResolveReferenceType:
    def test_type_by_ref_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        ref_data = {
            "id": "specific_ref", "active": False,
            "reference": {"ia": 49.0, "s": 11.5},
        }
        health_refs.save_type_ref("EL84", "specific_ref", ref_data)
        lamp = _make_lamp()
        ref = _resolve_reference("type", "EL84", "L1", "specific_ref", lamp)
        assert ref["id"] == "specific_ref"
        assert ref["reference"]["ia"] == 49.0

    def test_type_active_ref_when_no_ref_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        health_refs.save_type_ref("EL84", "a", {"id": "a", "active": True, "reference": {"ia": 50.0}})
        health_refs.save_type_ref("EL84", "b", {"id": "b", "active": False, "reference": {"ia": 51.0}})
        lamp = _make_lamp()
        ref = _resolve_reference("type", "EL84", "L1", None, lamp)
        assert ref["id"] == "a"

    def test_type_median_when_no_active(self, tmp_path, monkeypatch):
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        health_refs.save_type_ref("EL84", "r1", {
            "id": "r1", "active": False, "timestamp": "2026-01-01",
            "reference": {"ia": 45.0, "s": 10.0, "r": 38.0},
        })
        health_refs.save_type_ref("EL84", "r2", {
            "id": "r2", "active": False, "timestamp": "2026-01-02",
            "reference": {"ia": 47.0, "s": 11.0, "r": 42.0},
        })
        health_refs.save_type_ref("EL84", "r3", {
            "id": "r3", "active": False, "timestamp": "2026-01-03",
            "reference": {"ia": 50.0, "s": 12.0, "r": 40.0},
        })
        lamp = _make_lamp()
        ref = _resolve_reference("type", "EL84", "L1", None, lamp)
        assert ref["source"] == "type_median"
        assert ref["reference"]["ia"] == pytest.approx(47.0)
        assert ref["reference"]["s"] == pytest.approx(11.0)
        assert ref["reference"]["r"] == pytest.approx(40.0)

    def test_type_median_skips_non_numeric(self, tmp_path, monkeypatch):
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        health_refs.save_type_ref("EL84", "r1", {
            "id": "r1", "active": False, "timestamp": "2026-01-01",
            "reference": {"ia": 45.0, "s": None},
        })
        health_refs.save_type_ref("EL84", "r2", {
            "id": "r2", "active": False, "timestamp": "2026-01-02",
            "reference": {"ia": 50.0, "s": 10.0},
        })
        health_refs.save_type_ref("EL84", "r3", {
            "id": "r3", "active": False, "timestamp": "2026-01-03",
            "reference": {"ia": 48.0, "s": 11.0},
        })
        lamp = _make_lamp()
        ref = _resolve_reference("type", "EL84", "L1", None, lamp)
        assert ref["source"] == "type_median"
        # median of [45, 48, 50] = 48; s has only 2 numeric [10, 11] → median picks [1]=11
        assert ref["reference"]["ia"] == pytest.approx(48.0)
        assert ref["reference"]["s"] == pytest.approx(11.0)
        assert "k" not in ref["reference"]

    def test_type_empty_falls_to_datasheet(self, tmp_path, monkeypatch):
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        lamp = _make_lamp()
        ref = _resolve_reference("type", "EL84", "L1", None, lamp)
        assert ref["source"] == "datasheet"

    def test_type_ref_id_not_found_tries_active(self, tmp_path, monkeypatch):
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        health_refs.save_type_ref("EL84", "real", {
            "id": "real", "active": True,
            "reference": {"ia": 48.0},
        })
        lamp = _make_lamp()
        ref = _resolve_reference("type", "EL84", "L1", "nonexistent", lamp)
        assert ref["id"] == "real"


# ---------------------------------------------------------------------------
# Tests: _build_reference_from_measurement
# ---------------------------------------------------------------------------

class TestBuildReferenceFromMeasurement:
    def test_full_measurement(self):
        measurement = {
            "health": {
                "raw": {"ia_op": 47.5},
                "metrics": {"emission_ratio": 0.82},
            },
            "srk": {"s": 10.8, "r": 38.5, "k": 18.0},
        }
        ref = _build_reference_from_measurement(measurement)
        assert ref["ia"] == pytest.approx(47.5)
        assert ref["s"] == pytest.approx(10.8)
        assert ref["r"] == pytest.approx(38.5)
        assert ref["k"] == pytest.approx(18.0)
        assert ref["emission_ratio"] == pytest.approx(0.82)

    def test_empty_measurement(self):
        ref = _build_reference_from_measurement({})
        assert ref["ia"] is None
        assert ref["s"] is None
        assert ref["r"] is None
        assert ref["k"] is None
        assert ref["emission_ratio"] is None

    def test_partial_measurement_no_srk(self):
        measurement = {
            "health": {
                "raw": {"ia_op": 45.0},
                "metrics": {},
            },
        }
        ref = _build_reference_from_measurement(measurement)
        assert ref["ia"] == pytest.approx(45.0)
        assert ref["s"] is None
        assert ref["emission_ratio"] is None

    def test_measurement_with_none_health(self):
        measurement = {"health": None, "srk": None}
        ref = _build_reference_from_measurement(measurement)
        assert ref["ia"] is None
        assert ref["s"] is None

    def test_measurement_srk_only(self):
        measurement = {"srk": {"s": 12.0, "r": 42.0, "k": 20.0}}
        ref = _build_reference_from_measurement(measurement)
        assert ref["ia"] is None
        assert ref["s"] == pytest.approx(12.0)
        assert ref["r"] == pytest.approx(42.0)
        assert ref["k"] == pytest.approx(20.0)
        assert ref["emission_ratio"] is None


class TestUnreadableRefWarns:
    """ML-102: a corrupt reference silently shifted the scoring base — the
    skip must be visible at the default INFO log level (WARNING, not
    debug), mirroring health_measurements.py."""

    def test_unreadable_type_ref_warns(self, tmp_path, monkeypatch, caplog):
        import logging
        monkeypatch.setattr(health_refs, "_root", lambda: tmp_path)
        bad = health_refs._type_dir("EL84") / "bad.json"
        bad.write_text("{broken", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="lm19.health_refs"):
            refs = health_refs.list_type_refs("EL84")
        assert refs == []
        assert any(r.levelno >= logging.WARNING
                   and "bad.json" in r.getMessage()
                   for r in caplog.records),             "unreadable ref skipped without a WARNING"
