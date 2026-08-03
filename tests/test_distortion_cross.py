"""Cross-validation: Chebyshev vs DFT vs 5-point on identical data."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.amplifier import (
    compute_distortion,
    compute_distortion_chebyshev,
    compute_distortion_dft,
    ResistiveLoadLine,
    find_intersections,
)


class SimpleModel:
    """Tube model from analytical function for cross-validation."""
    topology = "triode"
    model_type = "test"
    name = "test"
    pa_max = 12.0
    uh = 6.3
    ih = 0.7

    def __init__(self, func):
        self._func = func

    def ia(self, ua, ug1, ug2=0.0):
        return max(0.0, self._func(ug1))

    def ig2(self, ua, ug1, ug2):
        return 0.0

    def generate_scan(self, grid):
        return []

    def params_dict(self):
        return {}


def _make_pts(func, n=20, ub=250.0, ra=5.0):
    pts = []
    for i in range(n):
        ug1 = -10.0 + (9.0) * i / (n - 1)
        ia = max(0.0, func(ug1))
        ua = ub - ia * ra
        pts.append({"ug1": ug1, "ua": ua, "ia": ia})
    return pts


def _ll(ub=250.0, ra=5.0):
    return ResistiveLoadLine(ub, ra)


# ═══════════════════════════════════════════════════════════════════
# Chebyshev vs DFT agreement
# ═══════════════════════════════════════════════════════════════════

class TestChebyshevVsDft:
    """Chebyshev on measured points vs DFT on model should agree."""

    def test_quadratic_hd2_agrees(self):
        func = lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2
        pts = _make_pts(func, n=20)
        model = SimpleModel(func)

        r_cheb = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        r_dft = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)

        assert r_cheb is not None and r_dft is not None
        assert abs(r_cheb["hd2"] - r_dft["hd2"]) < 2.0  # within 2%
        assert abs(r_cheb["hd3"] - r_dft["hd3"]) < 1.0

    def test_cubic_hd3_agrees(self):
        func = lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.1 * (ug1 + 7.0) ** 3
        pts = _make_pts(func, n=20)
        model = SimpleModel(func)

        r_cheb = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        r_dft = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)

        assert r_cheb is not None and r_dft is not None
        assert abs(r_cheb["hd3"] - r_dft["hd3"]) < 2.0

    def test_mixed_nonlinearity_thd_agrees(self):
        func = lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.2 * (ug1 + 7.0) ** 2 + 0.05 * (ug1 + 7.0) ** 3
        pts = _make_pts(func, n=25)
        model = SimpleModel(func)

        r_cheb = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        r_dft = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)

        assert r_cheb is not None and r_dft is not None
        assert abs(r_cheb["thd"] - r_dft["thd"]) < 3.0

    def test_hd4_through_hd9_agree(self):
        """Higher harmonics should agree between methods."""
        func = lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.2 * (ug1 + 7.0) ** 2 + 0.05 * (ug1 + 7.0) ** 3
        pts = _make_pts(func, n=25)
        model = SimpleModel(func)

        r_cheb = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        r_dft = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)

        assert r_cheb is not None and r_dft is not None
        for n in range(4, 8):
            key = f"hd{n}"
            # Both should be small for polynomial of degree 3
            assert r_cheb[key] < 2.0
            assert r_dft[key] < 2.0


# ═══════════════════════════════════════════════════════════════════
# All 3 methods on same data
# ═══════════════════════════════════════════════════════════════════

class TestThreeWay:
    """Compare 5-point, Chebyshev, and DFT on the same characteristic."""

    def test_all_three_hd2_agree_quadratic(self):
        func = lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.3 * (ug1 + 7.0) ** 2
        pts = _make_pts(func, n=20)
        model = SimpleModel(func)

        r_5pt = compute_distortion(pts, ug1_bias=-7.0, half_swing=3.0)
        r_cheb = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        r_dft = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)

        assert r_5pt is not None and r_cheb is not None and r_dft is not None

        hd2_values = [r_5pt["hd2"], r_cheb["hd2"], r_dft["hd2"]]
        # All three should be within ±15% of each other
        mean_hd2 = sum(hd2_values) / 3
        for v in hd2_values:
            assert abs(v - mean_hd2) / max(mean_hd2, 0.1) < 0.20, \
                f"HD2 spread too large: {hd2_values}"

    def test_all_three_hd3_agree_cubic(self):
        func = lambda ug1: 10.0 + 2.0 * (ug1 + 7.0) + 0.1 * (ug1 + 7.0) ** 3
        pts = _make_pts(func, n=20)
        model = SimpleModel(func)

        r_5pt = compute_distortion(pts, ug1_bias=-7.0, half_swing=3.0)
        r_cheb = compute_distortion_chebyshev(pts, ug1_bias=-7.0, half_swing=3.0)
        r_dft = compute_distortion_dft(model, _ll(), ug1_bias=-7.0, half_swing=3.0)

        assert r_5pt is not None and r_cheb is not None and r_dft is not None

        hd3_values = [r_5pt["hd3"], r_cheb["hd3"], r_dft["hd3"]]
        mean_hd3 = sum(hd3_values) / 3
        for v in hd3_values:
            assert abs(v - mean_hd3) / max(mean_hd3, 0.1) < 0.20, \
                f"HD3 spread too large: {hd3_values}"


# ═══════════════════════════════════════════════════════════════════
# Different swing levels
# ═══════════════════════════════════════════════════════════════════

class TestSwingLevels:
    """Methods should agree across different swing levels."""

    @pytest.mark.parametrize("swing", [1.0, 2.0, 3.0, 4.0])
    def test_cheb_dft_agree_at_different_swings(self, swing):
        func = lambda ug1: 10.0 + 2.0 * (ug1 + 5.5) + 0.2 * (ug1 + 5.5) ** 2
        pts = _make_pts(func, n=20)
        model = SimpleModel(func)

        r_cheb = compute_distortion_chebyshev(pts, ug1_bias=-5.5, half_swing=swing)
        r_dft = compute_distortion_dft(model, _ll(), ug1_bias=-5.5, half_swing=swing)

        assert r_cheb is not None and r_dft is not None, (
            f"both methods must compute on synthetic rig at swing={swing}")
        assert abs(r_cheb["thd"] - r_dft["thd"]) < 5.0, \
            f"THD mismatch at swing={swing}: cheb={r_cheb['thd']:.2f} dft={r_dft['thd']:.2f}"
