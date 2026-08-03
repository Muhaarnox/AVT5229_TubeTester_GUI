"""Smoke tests for fast SPICE export fit flows."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.spice_export import fit_and_export_spice
from lm19.tube_sim import quick_pentode, quick_triode
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)

pytestmark = [pytest.mark.smoke_spice]


@pytest.mark.smoke
def test_spice_export_fit_minimal_triode_smoke(tmp_path):
    out_path = tmp_path / "triode.sub"
    _model, points = quick_triode("12AX7")
    result = fit_and_export_spice(str(out_path), "12AX7", points, topology=TOPOLOGY_TRIODE)

    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8")
    assert result.model_type == TOPOLOGY_TRIODE
    assert result.n_points >= 10


@pytest.mark.smoke
def test_spice_export_fit_minimal_pentode_smoke(tmp_path):
    out_path = tmp_path / "pentode.sub"
    _model, points = quick_pentode("EL84")
    result = fit_and_export_spice(str(out_path), "EL84", points, topology=TOPOLOGY_PENTODE)

    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8")
    assert result.model_type == TOPOLOGY_PENTODE
    assert result.n_points >= 10
