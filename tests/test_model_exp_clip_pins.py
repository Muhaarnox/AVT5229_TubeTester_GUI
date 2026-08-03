# -*- coding: utf-8 -*-
"""exp-clip regression pins for the Koren and Reefman model kernels.

Twin of tests/test_dempwolf_paper_pins.py::TestNoInRangeSaturation:
the old np.clip(arg, -50, 50) in
spice_export/koren.py SATURATED the Kp-normalized softplus in-range —
3 real benchmark datasets were affected (10_VT25 max_arg=77, 801_VT62
62, 6N5P 234 at fitted kp=1000) — while the SPICE export (unclipped)
diverged from Python exactly there. Reefman's +80 cap was the same bug
class (no dataset observed in the zone; fixed for consistency).

Reference values below are computed via the exact overflow-safe softplus
identity ln(1+e^x) = max(x,0) + ln(1+e^-|x|), independent of the kernels.
"""

import numpy as np
import pytest

from lm19.spice_export.koren import (
    _koren_ia,
    _koren_ia_pentode,
    _generate_triode_subcircuit,
    _generate_pentode_subcircuit,
)
from lm19.reefman import _koren_cathode
from lm19.spice_export.reefman import _generate_reefman_subcircuit
from lm19.tube_params import ReefmanParams


def _softplus_exact(arg: float) -> float:
    """ln(1 + exp(arg)) without overflow — reference, not the kernel."""
    return max(arg, 0.0) + float(np.log1p(np.exp(-abs(arg))))


# ---------------------------------------------------------------------------
# Koren triode — 6N5P-like fit (kp=1000) hit arg=234 under the old clip
# ---------------------------------------------------------------------------

class TestKorenTriodeNoSaturation:
    MU, EX, KG1, KP, KVB = 24.5, 1.4, 500.0, 1000.0, 300.0

    def _ia(self, ua, ug1):
        return float(_koren_ia(np.array([ua]), np.array([ug1]),
                               self.MU, self.EX, self.KG1, self.KP,
                               self.KVB)[0])

    def test_exact_in_former_clip_zone(self):
        """At (Ua=50, Vg=+2) arg ~ 79 — between the old 50 and new 700."""
        ua, ug1 = 50.0, 2.0
        arg = self.KP * (1.0 / self.MU + ug1 / np.sqrt(self.KVB + ua * ua))
        assert 50 < arg < 700  # guard: the point must sit in the zone
        e1 = (ua / self.KP) * _softplus_exact(arg)
        ia_ref = 2.0 * e1 ** self.EX / self.KG1
        assert self._ia(ua, ug1) == pytest.approx(ia_ref, rel=1e-12)

    def test_gm_monotone_through_former_onset(self):
        """Old clip flattened Ia above Vg ~ +0.5 at this setpoint."""
        ia = [self._ia(50.0, vg) for vg in (0.0, 0.5, 1.0, 2.0)]
        assert ia[0] < ia[1] < ia[2] < ia[3]
        # non-degeneracy: the last two points lie past the old onset and
        # must still show real slope, not epsilon jitter
        assert ia[3] > 1.05 * ia[2]

    def test_finite_at_pathological_arg(self):
        """arg > 700 clips instead of overflowing to inf."""
        assert np.isfinite(self._ia(0.02, 30.0))


# ---------------------------------------------------------------------------
# Koren pentode — arg = Kp*(1/mu + Vg/Vg2) crosses 50 near Vg=0 for
# high-Kp/low-mu parameter sets
# ---------------------------------------------------------------------------

class TestKorenPentodeNoSaturation:
    MU, EX, KG1, KP, KVB = 10.0, 1.35, 300.0, 600.0, 30.0

    def _ia(self, ua, ug1, ug2):
        return float(_koren_ia_pentode(np.array([ua]), np.array([ug1]),
                                       np.array([ug2]), self.MU, self.EX,
                                       self.KG1, self.KP, self.KVB)[0])

    def test_exact_in_former_clip_zone(self):
        """At Vg=0, Vg2=250: arg = Kp/mu = 60 — inside the former zone."""
        ua, ug1, ug2 = 250.0, 0.0, 250.0
        arg = self.KP * (1.0 / self.MU + ug1 / ug2)
        assert 50 < arg < 700
        e1 = (ug2 / self.KP) * _softplus_exact(arg)
        ia_ref = 2.0 * e1 ** self.EX / self.KG1 * np.arctan(ua / self.KVB)
        assert self._ia(ua, ug1, ug2) == pytest.approx(ia_ref, rel=1e-12)

    def test_gm_monotone_through_former_onset(self):
        """Former onset at Vg = (50/Kp - 1/mu)*Vg2 ~ -4.2 V @ Vg2=250."""
        ia = [self._ia(250.0, vg, 250.0) for vg in (-6.0, -4.0, -2.0, 0.0)]
        assert ia[0] < ia[1] < ia[2] < ia[3]
        assert ia[3] > 1.05 * ia[2]


# ---------------------------------------------------------------------------
# Reefman — the old +80 cap was the same class (denominator softens it)
# ---------------------------------------------------------------------------

class TestReefmanNoSaturation:
    MU, EX, KP, KVB = 5.0, 1.35, 500.0, 100.0

    def test_exact_in_former_clip_zone(self):
        """At Vg1=0, Vg2=250: arg = Kp/mu = 100 — past the old +80 cap."""
        vg2, vg1 = 250.0, 0.0
        arg = self.KP * (1.0 / self.MU
                         + vg1 / np.sqrt(self.KVB + vg2 * vg2))
        assert 80 < arg < 700
        e1 = (vg2 / self.KP) * _softplus_exact(arg)
        ip_ref = e1 ** self.EX
        got = float(_koren_cathode(np.array([vg2]), np.array([vg1]),
                                   self.MU, self.EX, self.KP, self.KVB)[0])
        assert got == pytest.approx(ip_ref, rel=1e-12)


# ---------------------------------------------------------------------------
# Headroom inline grid-current softplus (lm19/amplifier/sweeps.py) — the
# same class hid there as min(arg, 50); reachable only with positive
# ug1_bias (A2 analysis), where Vgk at the positive peak = 2*bias
# ---------------------------------------------------------------------------

class TestHeadroomGridCurrentClip:
    def test_ig1_exact_in_former_clip_zone(self):
        from lm19.amplifier import compute_headroom
        gg, xi, cg = 6e-4, 1.3, 10.0
        # ug1_max >= -0.1 => swing_pos = |bias| => Vgk_at_pos = 2*bias = 6 V
        pts = [{"ug1": vg, "ua": 300.0 - 10.0 * vg, "ia": 20.0 + 3.0 * vg}
               for vg in np.arange(-10.0, 8.1, 1.0)]
        hr = compute_headroom(
            pts, ug1_bias=3.0,
            grid_current_params={"Gg": gg, "xi": xi, "Cg": cg})
        assert hr is not None and hr.get("ig1_ma") is not None
        arg = cg * 6.0  # = 60: between the old cap 50 and the new 700
        assert 50 < arg < 700
        ig1_ref = gg * (_softplus_exact(arg) / cg) ** xi * 1000.0
        assert hr["ig1_ma"] == pytest.approx(ig1_ref, rel=1e-12)


# ---------------------------------------------------------------------------
# SPICE export guards — MIN(...,700) mirrors the Python clips
# ---------------------------------------------------------------------------

def _e1_lines(content):
    return [ln for ln in content.splitlines() if "LOG(1+EXP" in ln]


class TestSpiceExportGuards:
    def test_koren_triode_e1_guarded(self):
        content = _generate_triode_subcircuit(
            "TESTTRI", "12AX7", 100.0, 1.4, 1060.0, 600.0, 300.0,
            rms_error=0.1, max_error=0.2, n_points=100, backend="scipy")
        lines = _e1_lines(content)
        assert lines
        assert all("MIN(" in ln and ",700)" in ln for ln in lines)

    def test_koren_pentode_e1_guarded(self):
        content = _generate_pentode_subcircuit(
            "TESTPENT", "EL84", 19.0, 1.35, 692.0, 300.0, 20.0, 4500.0,
            rms_error=0.5, max_error=1.0, n_points=100, backend="scipy")
        lines = _e1_lines(content)
        assert lines
        assert all("MIN(" in ln and ",700)" in ln for ln in lines)

    @pytest.mark.parametrize("rtype,mu_b", [
        ("BTetrodeD", None),          # plain E1 only
        ("PenthodeVD", 6.0),          # vmu adds the E11 twin
    ])
    def test_reefman_e1_guarded(self, rtype, mu_b):
        rp = ReefmanParams(
            type=rtype, mu=23.0, ex=1.35, kg1=1060.0, kg2=4500.0,
            kp=300.0, kvb=25.0, als=0.5, be=0.02, A=1e-4,
            mu_b=mu_b, ex_b=1.5 if mu_b else None,
            svar=0.08 if mu_b else 0.0)
        content = _generate_reefman_subcircuit(
            "TESTREEF", "EL84", rp,
            rms_error=0.5, max_error=1.0, n_points=100)
        lines = _e1_lines(content)
        expected = 2 if mu_b else 1  # vmu emits E1 + E11
        assert len(lines) == expected
        assert all("MIN(" in ln and ",700)" in ln for ln in lines)
