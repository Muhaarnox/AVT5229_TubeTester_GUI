"""Tests for tools/refine_efficiency.py benchmark helper.

Primary regression pin (#24): run_scan_benchmark must actually RUN — the old
`patch("lm19.scan.time.sleep")` target crashed on entry (the lm19.scan package
never imports time), so the whole benchmark died before producing a number.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))


def _load_module():
    """Load refine_efficiency.py as a module (it's a script, not a package)."""
    script_path = PROJECT_ROOT / "tools" / "refine_efficiency.py"
    spec = importlib.util.spec_from_file_location(
        "refine_efficiency_under_test", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rf():
    return _load_module()


class TestRunScanBenchmark:
    """The benchmark scan must run end-to-end without raising."""

    def test_grid_scan_runs_and_returns_points(self, rf) -> None:
        # Under the old broken patch target this raised AttributeError before
        # collecting a single point — assert it now returns real data.
        points = rf.run_scan_benchmark(50, refine=False, ug2=200)
        assert isinstance(points, list)
        assert len(points) > 0
        p = points[0]
        assert {"ua", "ug1", "ia"} <= set(p)

    def test_refine_never_drops_points(self, rf) -> None:
        # Refinement only inserts points between grid points — at a coarse
        # step it must collect at least as many as the plain grid.
        grid = rf.run_scan_benchmark(50, refine=False, ug2=200)
        refined = rf.run_scan_benchmark(50, refine=True, ug2=200)
        assert len(refined) >= len(grid)

    def test_scan_is_physically_sane(self, rf) -> None:
        # Ia must be non-negative and within the model's 0..100 mA range.
        points = rf.run_scan_benchmark(25, refine=False, ug2=200)
        ias = [p["ia"] for p in points]
        assert all(ia >= 0.0 for ia in ias)
        assert max(ias) <= 100.0 + 1e-6
        assert max(ias) > 0.0          # the tube actually conducts somewhere


class TestTubeModel:
    """Physical sanity of the synthetic pentode model."""

    def test_cutoff_below_minus_10(self, rf) -> None:
        assert rf.tube_ia_ma(200, -10, 200) == 0.0
        assert rf.tube_ia_ma(200, -12, 200) == 0.0

    def test_zero_screen_gives_zero(self, rf) -> None:
        # Division guard: Ug2=0 must not raise and yields no current.
        assert rf.tube_ia_ma(200, -4, 0) == 0.0

    def test_monotonic_increasing_in_ua(self, rf) -> None:
        lo = rf.tube_ia_ma(50, -4, 200)
        mid = rf.tube_ia_ma(150, -4, 200)
        hi = rf.tube_ia_ma(300, -4, 200)
        assert lo < mid < hi

    def test_more_open_grid_gives_more_current(self, rf) -> None:
        closed = rf.tube_ia_ma(200, -8, 200)
        open_ = rf.tube_ia_ma(200, -2, 200)
        assert open_ > closed


class TestApproxError:
    def test_too_few_points_is_inf(self, rf) -> None:
        pts = [{"ua": 100, "ug1": -4, "ug2": 200, "ia": 50}]
        assert rf.approx_error(pts, [100, 150], -4, 200) == float("inf")

    def test_dense_true_curve_has_small_error(self, rf) -> None:
        import numpy as np
        ua_grid = np.arange(10, 301, 10.0)
        pts = [{"ua": float(ua), "ug1": -4, "ug2": 200,
                "ia": rf.tube_ia_ma(float(ua), -4, 200)} for ua in ua_grid]
        err = rf.approx_error(pts, ua_grid, -4, 200)
        assert err < 1.0          # sampling the true curve → near-zero RMS

    def test_no_overlap_grid_is_inf(self, rf) -> None:
        # All eval points outside the measured ua span → no comparison.
        pts = [{"ua": 100, "ug1": -4, "ug2": 200, "ia": 40},
               {"ua": 120, "ug1": -4, "ug2": 200, "ia": 45}]
        assert rf.approx_error(pts, [10, 300], -4, 200) == float("inf")
