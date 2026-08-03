"""Smoke tests for core analysis pipeline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.amplifier import (
    ResistiveLoadLine,
    compute_distortion,
    compute_headroom,
    find_intersections,
    optimize_bias,
)
from lm19.tube_sim import quick_pentode, quick_triode

pytestmark = [pytest.mark.smoke_analysis]


@pytest.mark.smoke
def test_triode_pipeline_quick_smoke_generate_intersect_distortion():
    _, points = quick_triode("12AU7")
    load_line = ResistiveLoadLine(ub=250, ra=10.0)
    intersections = find_intersections(points, load_line)
    dist = compute_distortion(intersections, ug1_bias=-10.0)

    assert len(intersections) >= 3
    assert dist is not None
    assert dist["thd"] >= 0


@pytest.mark.smoke
def test_pentode_pipeline_quick_smoke_generate_intersect_headroom():
    _, points = quick_pentode("EL84")
    load_line = ResistiveLoadLine(ub=300, ra=5.0)
    intersections = find_intersections(points, load_line)
    headroom = compute_headroom(intersections, ug1_bias=-7.0, pa_max=12.0, load_line=load_line)

    assert len(intersections) >= 3
    assert headroom is not None
    assert headroom["max_swing"] >= 0


@pytest.mark.smoke
def test_optimize_bias_smoke_returns_valid_result():
    _, points = quick_triode("12AX7")
    load_line = ResistiveLoadLine(ub=250, ra=100.0)
    result = optimize_bias(points, load_line, target="min_thd")

    assert result is not None
    assert result["thd"] >= 0
