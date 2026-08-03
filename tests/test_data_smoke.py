"""Smoke tests for data import/export and persistence flows."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.curvetracedata_import import dat_to_lm19_points, parse_curvetracedata_dat
from lm19.measurements import load_measurements, save_measurement
from lm19.utracer_import import parse_utd, utd_to_lm19_points

pytestmark = [pytest.mark.smoke_data]


@pytest.mark.smoke
def test_measurements_save_load_roundtrip_smoke(monkeypatch, tmp_path):
    from lm19 import measurements as m

    monkeypatch.setattr(m, "_root", lambda: tmp_path)
    payload = {
        "timestamp": "2026-02-25T12:00:00",
        "tube_type": "EL84",
        "lamp_id": "L1",
        "name": "smoke",
        "points": [{"ua": 150.0, "ug1": -5.0, "ug2": 150.0, "ia": 10.0, "ig2": 0.2, "uh": 6.3, "ih": 0.7}],
    }

    path = save_measurement("EL84", "L1", payload)
    loaded = load_measurements("EL84", "L1")

    assert path.exists()
    assert len(loaded) == 1
    assert loaded[0]["name"] == "smoke"
    assert loaded[0]["points"][0]["ia"] == 10.0


@pytest.mark.smoke
def test_utracer_import_minimal_output_matrix_smoke(tmp_path):
    content = (
        "Va (V) Ia (mA)\n"
        " Vg = -4 V  Vg = -2 V\n"
        "10 0.10 0.30\n"
        "20 0.20 0.50\n"
    )
    file_path = tmp_path / "mini.utd"
    file_path.write_text(content, encoding="utf-8")

    parsed = parse_utd(str(file_path))
    points = utd_to_lm19_points(parsed, vs=250.0, vh=6.3)

    assert parsed["format"] == "output"
    assert len(points) == 4
    assert {"ua", "ug1", "ug2", "ia", "ig2", "uh", "ih"}.issubset(points[0].keys())


@pytest.mark.smoke
def test_curvetracedata_import_minimal_dat_smoke(tmp_path):
    content = (
        "% * Sample: EL34_1\n"
        "% * Date / time: 2026-02-25 12:00:00\n"
        "0.00 0.02500 0.1 0.00010 0 -0.000 -1.000 -1.000 -0.000 0 NA\n"
        "10.00 0.02500 10.0 0.00200 0 -0.000 -1.000 -1.200 -0.000 0 NA\n"
    )
    file_path = tmp_path / "mini.dat"
    file_path.write_text(content, encoding="utf-8")

    parsed = parse_curvetracedata_dat(str(file_path))
    points = dat_to_lm19_points(parsed, vs=250.0, vh=6.3)

    assert parsed["sample_name"] == "EL34_1"
    assert len(points) == 2
    assert points[1]["ia"] > points[0]["ia"]
