# -*- coding: utf-8 -*-
"""Paper-equation pins for the Reefman (Derk/DerkE) model.

The
reference implementations below are INDEPENDENT re-implementations from
the primary source — D. Reefman, "Spice models for vacuum tubes using
the uTracer", Theory.pdf, Jan 2016
(external_sources/theory/reefman_utracer_spice_theory.pdf):

  eq. (14)-(15)  Koren current Ip = E1^x, E1 with sqrt(kVB + Vg2^2)
  eq. (23)/(25)  Derk screen/anode currents, f(Va) = 1/(1+beta*Va)
  eq. (27)       alpha = 1 - kg1/kg2*(1+alpha_s)  (Ia(0) = 0 constraint)
  eq. (28)/(30)  DerkE variant, f(Va) = exp(-(beta*Va)^(3/2))
  eq. (33)       variable-mu blend of two Koren currents
  eq. (42)-(46)  secondary emission Psec = S*Va*(1+tanh(-ap*(Va-Vco))),
                 subtracted from Ia AND ADDED to Ig2 (LM19 fix;
                 NB Reefman's own TubeLib.inc omits the +Psec term in
                 its G2 sources — LM19 follows the paper)

Do NOT refactor them to call lm19.reefman helpers — independence from
the implementation is what makes these pins discriminating.
"""

import math

import numpy as np
import pytest

from lm19.reefman import _koren_cathode, _koren_cathode_vmu, _derk_ia_ig2
from lm19.spice_export.reefman import _generate_reefman_subcircuit
from lm19.tube_params import ReefmanParams
from lm19.constants import (
    TOPOLOGY_PENTODE,
)


# ---------------------------------------------------------------------------
# Reference implementations (independent of lm19.reefman)
# ---------------------------------------------------------------------------

def _softplus_exact(x: float) -> float:
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def _ip_ref(vg2, vg1, mu, ex, kp, kvb):
    """eq. (14)-(15): Ip = E1^x for E1 > 0, else 0."""
    e1 = (vg2 / kp) * _softplus_exact(
        kp * (1.0 / mu + vg1 / math.sqrt(kvb + vg2 * vg2)))
    return max(e1, 0.0) ** ex


def _reefman_ref(va, vg1, vg2, p):
    """eq. (23)-(46) verbatim: returns (Ia, Ig2) in A, pre-clamp."""
    if p.mu_b is not None and p.type in ("PenthodeVD", "PenthodeVDE"):
        ip = ((1.0 - p.svar) * _ip_ref(vg2, vg1, p.mu, p.ex, p.kp, p.kvb)
              + p.svar * _ip_ref(vg2, vg1, p.mu_b, p.ex_b, p.kp, p.kvb))
    else:
        ip = _ip_ref(vg2, vg1, p.mu, p.ex, p.kp, p.kvb)
    va_s = max(va, 0.001)
    alpha = 1.0 - (p.kg1 / p.kg2) * (1.0 + p.als)
    if p.type in ("BTetrodeDE", "PenthodeDE", "PenthodeVDE"):
        f = math.exp(-((p.be * va_s) ** 1.5))
    else:
        f = 1.0 / (1.0 + p.be * va_s)
    ia = ip * (1.0 / p.kg1 - 1.0 / p.kg2 + p.A * va_s / p.kg1
               - (alpha / p.kg1 + p.als / p.kg2) * f)
    ig2 = ip / p.kg2 * (1.0 + p.als * f)
    if p.type in ("BTetrodeD", "BTetrodeDE") and p.Sc > 0:
        vco = vg2 / max(p.lam, 0.001) - p.nu * vg1 - p.w
        psec = (p.Sc / p.kg2) * va_s * (1.0 + math.tanh(-p.ap * (va_s - vco)))
        ia -= ip * psec
        ig2 += ip * psec
    return ia, ig2


def _got(va, vg1, vg2, p):
    ia, ig2 = _derk_ia_ig2(np.array([va], float), np.array([vg1], float),
                           np.array([vg2], float), p)
    return float(ia[0]), float(ig2[0])


def _make(rtype, **over):
    base = dict(type=rtype, mu=23.0, ex=1.35, kg1=1060.0, kg2=4500.0,
                kp=300.0, kvb=25.0, als=3.0, be=0.02, A=1e-4)
    base.update(over)
    return ReefmanParams(**base)


def _random_grid(seed=3, n=400):
    rng = np.random.default_rng(seed)
    return (rng.uniform(1, 400, n), rng.uniform(-30, 0, n),
            rng.uniform(50, 300, n))


# ---------------------------------------------------------------------------
# 1. Cathode Koren current — eq. (14)-(15)
# ---------------------------------------------------------------------------

class TestCathodePaperEquations:
    def test_matches_eq_14_15_on_random_grid(self):
        mu, ex, kp, kvb = 23.0, 1.35, 300.0, 25.0
        va, vg1, vg2 = _random_grid()
        got = _koren_cathode(vg2, vg1, mu, ex, kp, kvb)
        for i in range(len(vg2)):
            ref = _ip_ref(vg2[i], vg1[i], mu, ex, kp, kvb)
            assert got[i] == pytest.approx(ref, rel=1e-9, abs=1e-300)

    def test_kvb_in_denominator_matters_at_low_vg2(self):
        """E1 uses sqrt(kvb + Vg2^2), not plain Vg2 (Koren pentode form) —
        discriminates at Vg2 comparable to sqrt(kvb)."""
        mu, ex, kp = 23.0, 1.35, 300.0
        got = float(_koren_cathode(np.array([20.0]), np.array([-1.0]),
                                   mu, ex, kp, 2000.0)[0])
        wrong = _ip_ref(20.0, -1.0, mu, ex, kp, 0.0)  # plain-Vg2 twin
        right = _ip_ref(20.0, -1.0, mu, ex, kp, 2000.0)
        assert got == pytest.approx(right, rel=1e-9)
        assert abs(got - wrong) > 0.05 * abs(right)  # non-degenerate

    def test_vmu_blend_eq_33(self):
        mu, ex, kp, kvb = 40.0, 1.3, 300.0, 25.0
        mu_b, ex_b, svar = 8.0, 1.5, 0.08
        va, vg1, vg2 = _random_grid(seed=4, n=100)
        got = _koren_cathode_vmu(vg2, vg1, mu, ex, kp, kvb,
                                 mu_b, ex_b, svar)
        for i in range(len(vg2)):
            ref = ((1.0 - svar) * _ip_ref(vg2[i], vg1[i], mu, ex, kp, kvb)
                   + svar * _ip_ref(vg2[i], vg1[i], mu_b, ex_b, kp, kvb))
            assert got[i] == pytest.approx(ref, rel=1e-9, abs=1e-300)


# ---------------------------------------------------------------------------
# 2. Derk / DerkE splitting — eq. (23)/(25)/(27)/(28)/(30)
# ---------------------------------------------------------------------------

class TestDerkSplitting:
    @pytest.mark.parametrize("rtype", ["PenthodeD", "PenthodeDE",
                                       "PenthodeVD", "PenthodeVDE"])
    def test_matches_paper_on_random_grid(self, rtype):
        over = {}
        if rtype.startswith("PenthodeV"):
            over = dict(mu=40.0, mu_b=8.0, ex_b=1.5, svar=0.08)
        p = _make(rtype, **over)
        va, vg1, vg2 = _random_grid()
        for i in range(0, len(va), 4):
            ia_ref, ig2_ref = _reefman_ref(va[i], vg1[i], vg2[i], p)
            ia, ig2 = _got(va[i], vg1[i], vg2[i], p)
            assert ia == pytest.approx(max(ia_ref, 0.0), rel=1e-9,
                                       abs=1e-15)
            assert ig2 == pytest.approx(max(ig2_ref, 0.0), rel=1e-9,
                                        abs=1e-15)

    def test_alpha_constraint_forces_ia_zero_at_va_zero(self):
        """eq. (26)-(27): alpha is derived so that Ia(Va=0) = 0."""
        for rtype in ("PenthodeD", "PenthodeDE"):
            p = _make(rtype, A=0.0)
            ia, ig2 = _got(1e-3, -5.0, 250.0, p)
            assert ig2 > 1e-5           # screen carries the current
            assert ia < 1e-3 * ig2      # anode is pinched off

    def test_derke_knee_sharper_than_derk(self):
        """f(Va) forms must differ: exp(-(be*Va)^1.5) decays past the knee
        much faster than 1/(1+be*Va) — pin at be*Va = 3."""
        va, vg1, vg2 = 150.0, -5.0, 250.0
        ia_d, _ = _got(va, vg1, vg2, _make("PenthodeD", A=0.0))
        ia_de, _ = _got(va, vg1, vg2, _make("PenthodeDE", A=0.0))
        # DerkE: f = exp(-3^1.5) ~ 0.0055; Derk: f = 0.25 -> Ia_DE > Ia_D
        assert ia_de > ia_d * 1.05

    def test_high_va_limit_ig2(self):
        """eq. (23): Ig2(Va->inf) -> Ip/kg2 (f -> 0). Va = 1e9 keeps the
        residual als*f = 3/(be*Va) far below the 1e-4 tolerance."""
        p = _make("PenthodeD", A=0.0)
        _, ig2 = _got(1e9, -5.0, 250.0, p)
        ip = _ip_ref(250.0, -5.0, p.mu, p.ex, p.kp, p.kvb)
        assert ig2 == pytest.approx(ip / p.kg2, rel=1e-4)


# ---------------------------------------------------------------------------
# 3. Secondary emission — eq. (42)-(46) + the +Psec Ig2 fix
# ---------------------------------------------------------------------------

class TestSecondaryEmission:
    SE = dict(Sc=0.05, ap=0.2, w=5.0, nu=2.0, lam=1.2)

    @pytest.mark.parametrize("rtype", ["BTetrodeD", "BTetrodeDE"])
    def test_matches_paper_including_ig2(self, rtype):
        """Unclamped equality: eq. (44)/(46) permit Ia < 0 (dynatron)."""
        p = _make(rtype, **self.SE)
        va, vg1, vg2 = _random_grid(seed=5)
        for i in range(0, len(va), 4):
            ia_ref, ig2_ref = _reefman_ref(va[i], vg1[i], vg2[i], p)
            ia, ig2 = _got(va[i], vg1[i], vg2[i], p)
            assert ia == pytest.approx(ia_ref, rel=1e-9, abs=1e-15)
            assert ig2 == pytest.approx(max(ig2_ref, 0.0), rel=1e-9,
                                        abs=1e-15)

    def test_dynatron_reversal_not_clamped(self):
        """Deep kink with TubeLib-scale Sc: the net anode current
        physically reverses. Paper eq. (44), TubeLib.inc G1 and our SPICE
        G1 all yield Ia < 0 there (LTspice gave -1.4 mA at this exact
        point) — the old np.maximum(ia, 0) clamp silently
        diverged from all three."""
        p = _make("BTetrodeD", Sc=0.03, ap=0.2, w=5.0, nu=2.0, lam=1.2)
        ia_ref, ig2_ref = _reefman_ref(130.0, -10.0, 250.0, p)
        assert ia_ref < 0  # non-degenerate: the point really reverses
        ia, ig2 = _got(130.0, -10.0, 250.0, p)
        assert ia == pytest.approx(ia_ref, rel=1e-9)
        assert ig2 == pytest.approx(ig2_ref, rel=1e-9)

    def test_space_current_independent_of_sc(self):
        """eq. (43)-(46): Psec moves current from anode to screen —
        Ia + Ig2 must not depend on Sc (constant space current).
        Sc = 0.002 keeps Ia > 0 at every point so the physical >=0
        clamp stays inert (larger Sc drives Ia negative below the
        crossover and the clamp legitimately breaks the sum)."""
        p0 = _make("BTetrodeD")
        p1 = _make("BTetrodeD", Sc=0.002, ap=0.2, w=5.0, nu=2.0, lam=1.2)
        for va, vg1, vg2 in ((60.0, -10.0, 250.0), (120.0, -20.0, 200.0),
                             (200.0, -5.0, 250.0)):
            ia0, ig20 = _got(va, vg1, vg2, p0)
            ia1, ig21 = _got(va, vg1, vg2, p1)
            assert ia1 < ia0            # anode loses current...
            assert ig21 > ig20          # ...the screen gains it
            assert ia1 + ig21 == pytest.approx(ia0 + ig20, rel=1e-12)

    def test_ig2_hump_in_kink_region(self):
        """Fig. 14 (EL500): measured Ig2 shows a hump where secondaries
        arrive; above the crossover (tanh -> -1) Psec dies out."""
        p = _make("BTetrodeD", Sc=0.2, ap=0.2, w=5.0, nu=2.0, lam=1.2)
        p0 = _make("BTetrodeD")
        vg1, vg2 = -10.0, 250.0
        vco = vg2 / p.lam - p.nu * vg1 - p.w   # ~223 V
        _, ig2_kink = _got(vco * 0.6, vg1, vg2, p)
        _, ig2_kink0 = _got(vco * 0.6, vg1, vg2, p0)
        assert ig2_kink > 1.2 * ig2_kink0      # hump present
        _, ig2_hi = _got(vco * 2.5, vg1, vg2, p)
        _, ig2_hi0 = _got(vco * 2.5, vg1, vg2, p0)
        assert ig2_hi == pytest.approx(ig2_hi0, rel=0.02)  # dies out

    def test_crossover_shifts_with_grid_voltage(self):
        """eq. (42): Vco = Vg2/lam - nu*Vg1 - w, nu > 0 — the Psec hump
        centre moves to higher Va at more negative Vg1."""
        p = _make("BTetrodeD", Sc=0.2, ap=0.2, w=5.0, nu=4.0, lam=1.2)
        p0 = _make("BTetrodeD")
        va_c = np.linspace(50, 400, 300)

        def hump_va(vg1):
            extra = [_got(v, vg1, 250.0, p)[1] - _got(v, vg1, 250.0, p0)[1]
                     for v in va_c]
            return va_c[int(np.argmax(extra))]

        assert hump_va(-25.0) > hump_va(-5.0)


# ---------------------------------------------------------------------------
# 4. Model wrappers and SPICE G2 guard
# ---------------------------------------------------------------------------

class TestWrappersAndSpice:
    def test_model_ia_ig2_in_ma(self):
        from lm19.reefman import ReefmanModel
        p = _make("BTetrodeD", **TestSecondaryEmission.SE)
        m = ReefmanModel(name="z", topology=TOPOLOGY_PENTODE, reefman=p)
        ia_a, ig2_a = _got(120.0, -10.0, 250.0, p)
        assert m.ia(120.0, -10.0, 250.0) == pytest.approx(ia_a * 1e3,
                                                          rel=1e-12)
        assert m.ig2(120.0, -10.0, 250.0) == pytest.approx(ig2_a * 1e3,
                                                           rel=1e-12)

    def test_spice_g2_carries_psec_for_beam_tetrodes(self):
        """The .sub G2 source must add Ip*V(9) (Psec node) — Theory.pdf
        eq. (43)/(45); TubeLib.inc omits it, LM19 follows the paper."""
        for rtype in ("BTetrodeD", "BTetrodeDE"):
            rp = _make(rtype, **TestSecondaryEmission.SE)
            content = _generate_reefman_subcircuit(
                "ZR", "ZONE", rp, rms_error=0.1, max_error=0.2, n_points=10)
            g2 = [ln for ln in content.splitlines()
                  if ln.startswith("G2 G2 K")]
            assert len(g2) == 1
            assert "*V(9)" in g2[0], "G2 must carry the +Ip*Psec term"

    def test_spice_g2_no_psec_for_pure_pentodes(self):
        rp = _make("PenthodeD")
        content = _generate_reefman_subcircuit(
            "ZR", "ZONE", rp, rms_error=0.1, max_error=0.2, n_points=10)
        g2 = [ln for ln in content.splitlines() if ln.startswith("G2 G2 K")]
        assert len(g2) == 1 and "V(9)" not in g2[0]
