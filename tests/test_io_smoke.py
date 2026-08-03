"""Smoke tests for import/export critical I/O paths."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.csv_export import format_csv
from lm19.csv_import import detect_columns, detect_separator, parse_csv


@pytest.mark.smoke
def test_csv_import_minimal_valid_file_roundtrip_to_points():
    csv_text = (
        "Ua;Ug1;Ug2;Ia;Ig2;Uh;Ih\n"
        "150.0;-5.0;150.0;12.0;0.2;6.3;0.7\n"
        "200.0;-5.0;200.0;18.0;0.3;6.3;0.7\n"
    )
    sep = detect_separator(csv_text)
    headers = csv_text.splitlines()[0].split(sep)
    mapping = detect_columns(headers)
    points = parse_csv(csv_text, mapping, sep)

    assert sep == ";"
    assert len(points) == 2
    assert points[0]["ua"] == 150.0
    assert points[1]["ia"] == 18.0


@pytest.mark.smoke
def test_export_spice_or_csv_minimal_dataset_creates_nonempty_output():
    points = [
        {"ua": 120.0, "ug1": -4.0, "ug2": 120.0, "ia": 7.5, "ig2": 0.1, "uh": 6.3, "ih": 0.7},
        {"ua": 180.0, "ug1": -4.0, "ug2": 180.0, "ia": 10.2, "ig2": 0.2, "uh": 6.3, "ih": 0.7},
    ]

    out = format_csv(points, tube_type="EL84", lamp_id="L1", is_triode=False)

    assert out
    assert "LM19 Tube Tester Export" in out
    assert "Ua;Ug1;Ug2;Ia;Ig2;Uh;Ih" in out
