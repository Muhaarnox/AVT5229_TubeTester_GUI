# -*- coding: utf-8 -*-
"""Paper-equation pins for the Dempwolf Extended v2 model.

The reference
implementations below are INDEPENDENT re-implementations from the
primary sources:

  - Dempwolf & Zoelzer, DAFx-11, eq. (10)-(12) + Table 1 (tube RSD-1)
    (external_sources/theory/dempwolf_dafx11_triode_model.pdf)
  - docs/DEMPWOLF_EXTENDED_MODEL.md par. 3.3 / 14.5 (v2 equations)
  - Reefman, Theory.pdf par. 6.2 (Vco physics: kink shifts with Vg)

Do NOT refactor them to call lm19.dempwolf helpers — their independence
from the implementation is what makes these pins discriminating.
"""

import numpy as np
import pytest

from lm19.tube_params import DempwolfParams
from lm19.dempwolf import (
    dempwolf_v2,
    dempwolf_v2_ia_vec,
    _eval_pentode_vec,
    _cathode_current,
)
from lm19.spice_export.dempwolf import (
    _generate_dempwolf_pentode_subcircuit,
    _generate_dempwolf_triode_subcircuit,
)


# ---------------------------------------------------------------------------
# Reference implementations (independent of lm19.dempwolf)
# ---------------------------------------------------------------------------

# DAFx-11 Table 1, tube RSD-1 (Ig0 = 8.025e-8 A intentionally omitted —
# see DEMPWOLF_EXTENDED_MODEL.md par. 2.1 note).
_MU, _G, _GAM, _C = 103.2, 2.242e-3, 1.26, 3.40
_GG, _XI, _CG = 6.177e-4, 1.314, 9.901

# Clamps documented in DEMPWOLF_EXTENDED_MODEL.md par. 9.5 / 14.5.3.
_V_MIN_REF = 0.01
_KVB_MIN_REF = 0.1
_CLIP_REF = 700.0


def _ik_paper(va, vg):
    """DAFx-11 eq. (10): Ik = G * (ln(1 + exp(C*(Va/mu + Vg))) / C)^gamma."""
    return _G * (np.log1p(np.exp(_C * (va / _MU + vg))) / _C) ** _GAM


def _ig_paper(vg):
    """DAFx-11 eq. (11) sans Ig0: Ig = Gg * (ln(1+exp(Cg*Vg))/Cg)^xi."""
    return _GG * (np.log1p(np.exp(_CG * vg)) / _CG) ** _XI


def _v2_doc(vpk, vgk, vg2k, q):
    """docs par. 14.5.2-14.5.4 pentode/beam/varmu equations, verbatim."""
    vpk_s = max(vpk, _V_MIN_REF)
    vg2_s = max(vg2k, _V_MIN_REF)
    arg = np.clip(q.C * (1.0 / q.mu + vgk / vg2_s), -_CLIP_REF, _CLIP_REF)
    sp = (vg2_s / q.C) * np.log1p(np.exp(arg))
    ik = q.G * max(sp, 0.0) ** q.gamma
    if q.mu_b is not None and q.gamma_b is not None and q.svar > 0:
        arg_b = np.clip(q.C * (1.0 / q.mu_b + vgk / vg2_s),
                        -_CLIP_REF, _CLIP_REF)
        sp_b = (vg2_s / q.C) * np.log1p(np.exp(arg_b))
        ik_b = q.G * max(sp_b, 0.0) ** q.gamma_b
        ik = (1.0 - q.svar) * ik + q.svar * ik_b
    ik = ik * (1.0 + q.A * vpk_s)
    igk = q.Gg * max(
        np.log1p(np.exp(np.clip(q.Cg * vgk, -_CLIP_REF, _CLIP_REF))) / q.Cg,
        0.0) ** q.xi
    i_thr = max(ik - igk, 0.0)
    v_eff = max(vg2_s / q.mu + vgk, 0.0)
    kvb_eff = max(q.Kvb + q.Kvb1 * v_eff, _KVB_MIN_REF)
    alpha = (1.0 - q.fg2) * (2.0 / np.pi) * np.arctan(
        (vpk_s / kvb_eff) ** q.Kn)
    ipk = i_thr * alpha
    ig2 = i_thr * (1.0 - alpha)
    if q.sigma > 0 and vg2_s > _V_MIN_REF:
        vco = vg2_s / q.lam - q.nu * vgk - q.w
        x = max(1.0 - vpk_s / max(vco, _V_MIN_REF), 0.0)
        i_sec = q.sigma * i_thr * (vpk_s / vg2_s) * x * np.exp(-q.Ks * x)
        ipk -= i_sec
        ig2 += i_sec
    return ipk, ig2, igk, ik


def _make_triode_paper(kvb_t):
    return DempwolfParams(mu=_MU, G=_G, gamma=_GAM, C=_C,
                          Gg=_GG, xi=_XI, Cg=_CG, Kvb_t=kvb_t)


def _make_beam():
    # nu=7.8 mirrors the real 6P1P fit; lam/w chosen off-default so the
    # Vco formula is exercised in full (non-degenerate data rule).
    return DempwolfParams(mu=8.7, G=2.8e-3, gamma=1.35, C=625.0,
                          Gg=6e-4, xi=1.3, Cg=10.0,
                          Kvb=24.0, Kvb1=0.3, Kn=1.2, fg2=0.08, A=2e-4,
                          sigma=3.0, Ks=4.0, lam=1.2, nu=7.8, w=5.0)


def _make_pentode():
    return DempwolfParams(mu=11.0, G=3e-3, gamma=1.35, C=750.0,
                          Gg=6e-4, xi=1.3, Cg=10.0,
                          Kvb=30.0, Kvb1=0.5, Kn=1.0, fg2=0.06, A=1e-4)


def _make_varmu():
    return DempwolfParams(mu=40.0, G=2e-3, gamma=1.3, C=400.0,
                          Gg=6e-4, xi=1.3, Cg=10.0,
                          Kvb=20.0, Kvb1=0.2, Kn=1.0, fg2=0.05, A=0.0,
                          mu_b=12.0, gamma_b=1.5, svar=0.08)


_PENTODE_SETS = [("beam", _make_beam), ("pentode", _make_pentode),
                 ("varmu", _make_varmu)]


def _random_triode_grid():
    rng = np.random.default_rng(1)
    return rng.uniform(20, 300, 400), rng.uniform(-5, 3, 400)


def _random_pentode_grid():
    rng = np.random.default_rng(2)
    return (rng.uniform(1, 400, 500), rng.uniform(-30, 2, 500),
            rng.uniform(50, 300, 500))


# ---------------------------------------------------------------------------
# 1. Triode core vs DAFx-11
# ---------------------------------------------------------------------------

class TestTriodePaperEquations:
    def test_matches_dafx11_on_random_grid(self):
        """Kvb_t -> 0 turns v_grid_eff into plain Vg: eq. (10)-(12) exact."""
        p = _make_triode_paper(kvb_t=1e-12)
        va_g, vg_g = _random_triode_grid()
        for va, vg in zip(va_g, vg_g):
            ia_ref = max(_ik_paper(va, vg) - _ig_paper(vg), 0.0)
            ipk, _, igk = dempwolf_v2(float(va), float(vg), None, p=p)
            assert ipk == pytest.approx(ia_ref, rel=1e-9, abs=1e-15)
            assert igk == pytest.approx(_ig_paper(vg), rel=1e-9, abs=1e-15)

    def test_datasheet_scale_at_250v_minus2v(self):
        """RSD-1 12AX7 at (250 V, -2 V) ~ 0.9 mA (paper fig. 6 ballpark)."""
        p = _make_triode_paper(kvb_t=1e-12)
        ia, _, _ = dempwolf_v2(250.0, -2.0, None, p=p)
        assert 0.5e-3 < ia < 2.5e-3

    def test_mu_and_gm_recovered_from_derivatives(self):
        """Finite-difference mu_eff must recover the input mu (~103)."""
        p = _make_triode_paper(kvb_t=1e-12)
        ia0, _, _ = dempwolf_v2(250.0, -2.0, None, p=p)
        d = 1e-3
        dva = (dempwolf_v2(250.0 + d, -2.0, None, p=p)[0] - ia0) / d
        dvg = (dempwolf_v2(250.0, -2.0 + d, None, p=p)[0] - ia0) / d
        assert 80 < dvg / dva < 130
        assert 0.8 < dvg * 1e3 < 3.0  # gm in mA/V

    def test_region_a_matches_doc_3_3(self):
        """v_grid_eff = Vg*Va/sqrt(Kvb_t + Va^2) with Kvb_t = 300."""
        p = _make_triode_paper(kvb_t=300.0)
        va_g, vg_g = _random_triode_grid()
        for va, vg in zip(va_g, vg_g):
            vge = vg * va / np.sqrt(300.0 + va * va)
            ik = _G * (np.log1p(np.exp(_C * (va / _MU + vge))) / _C) ** _GAM
            ia_ref = max(ik - _ig_paper(vg), 0.0)
            got = dempwolf_v2(float(va), float(vg), None, p=p)[0]
            assert got == pytest.approx(ia_ref, rel=1e-9, abs=1e-15)


# ---------------------------------------------------------------------------
# 2. Pentode / beam / varmu v2 vs doc par. 14.5
# ---------------------------------------------------------------------------

class TestPentodeV2Equations:
    @pytest.mark.parametrize("label,make", _PENTODE_SETS)
    def test_scalar_matches_independent_reimpl(self, label, make):
        q = make()
        va_p, vg_p, vg2_p = _random_pentode_grid()
        for va, vg, vg2 in zip(va_p, vg_p, vg2_p):
            ref = _v2_doc(va, vg, vg2, q)
            got = dempwolf_v2(float(va), float(vg), float(vg2), p=q)
            scale = max(abs(ref[0]), abs(ref[1]), 1e-9)
            assert abs(got[0] - ref[0]) <= 1e-9 * scale
            assert abs(got[1] - ref[1]) <= 1e-9 * scale
            assert abs(got[2] - ref[2]) <= 1e-9 * scale

    @pytest.mark.parametrize("label,make", _PENTODE_SETS)
    def test_current_conservation_exact(self, label, make):
        """Ia + Ig2 + Ig1 == Ik*(1 + A*Va) wherever i_through isn't clamped
        (doc par. 14.5.7) — including through secondary emission."""
        q = make()
        va_p, vg_p, vg2_p = _random_pentode_grid()
        checked = 0
        for va, vg, vg2 in zip(va_p, vg_p, vg2_p):
            ik_tot = _v2_doc(va, vg, vg2, q)[3]
            igk_ref = _v2_doc(va, vg, vg2, q)[2]
            if ik_tot - igk_ref <= 0:
                continue
            ia, ig2, igk = dempwolf_v2(float(va), float(vg), float(vg2), p=q)
            assert abs(ia + ig2 + igk - ik_tot) <= 1e-12 * max(ik_tot, 1e-12)
            checked += 1
        assert checked > 400  # non-degenerate: the clamp path must be rare


# ---------------------------------------------------------------------------
# 3. Kernel parity: scalar vs vectorized vs fitting kernel
# ---------------------------------------------------------------------------

class TestKernelParity:
    def test_triode_ia_vec_mirrors_scalar(self):
        p = _make_triode_paper(kvb_t=300.0)
        va_g, vg_g = _random_triode_grid()
        vec = dempwolf_v2_ia_vec(va_g, vg_g, None, p=p)
        sc = np.array([dempwolf_v2(float(a), float(g), None, p=p)[0]
                       for a, g in zip(va_g, vg_g)])
        assert np.array_equal(vec, sc)

    @pytest.mark.parametrize("label,make", _PENTODE_SETS)
    def test_pentode_ia_vec_mirrors_scalar(self, label, make):
        q = make()
        va_p, vg_p, vg2_p = _random_pentode_grid()
        vec = dempwolf_v2_ia_vec(va_p, vg_p, vg2_p, p=q)
        sc = np.array([dempwolf_v2(float(a), float(g), float(s), p=q)[0]
                       for a, g, s in zip(va_p, vg_p, vg2_p)])
        assert np.array_equal(vec, sc)

    def test_ia_vec_mirrors_scalar_at_vg2_edge(self):
        """Both ends of the vg2 <= _V_MIN boundary: the normalized-form mask
        and the secondary-emission gate must match the scalar branches.

        vg = +0.5 is load-bearing: at vg2 <= _V_MIN the scalar falls back to
        the PLAIN softplus form, and only a positive grid keeps both forms
        out of deep cutoff — at vg = -3 they both flush to ~0 A and a
        dropped mask is invisible (mutation-audit pin m7)."""
        q = _make_beam()
        vg2_edge = np.array([0.0, 0.005, 0.0099, 0.01, 0.0101, 0.02, 1.0])
        for vg_val in (-3.0, 0.5):
            va = np.full_like(vg2_edge, 150.0)
            vg = np.full_like(vg2_edge, vg_val)
            vec = dempwolf_v2_ia_vec(va, vg, vg2_edge, p=q)
            sc = np.array([dempwolf_v2(float(a), float(g), float(s), p=q)[0]
                           for a, g, s in zip(va, vg, vg2_edge)])
            assert np.array_equal(vec, sc), f"vg={vg_val}"
        # non-degeneracy guard: at vg=+0.5 the plain form must carry real
        # current (otherwise this pin silently degenerates again)
        sc_pos = dempwolf_v2(150.0, 0.5, 0.005, p=q)[0]
        assert sc_pos > 1e-5

    @pytest.mark.parametrize("label,make",
                             [("beam", _make_beam), ("pentode", _make_pentode)])
    def test_eval_pentode_vec_matches_scalar(self, label, make):
        """The fitting kernel must equal the shipped model on real-scan
        domain (vg2 >> _V_MIN) — it is the fit-quality metric."""
        q = make()
        va_p, vg_p, vg2_p = _random_pentode_grid()
        ia_fit, ig2_fit = _eval_pentode_vec(va_p, vg_p, vg2_p, q)
        sc = [dempwolf_v2(float(a), float(g), float(s), p=q)
              for a, g, s in zip(va_p, vg_p, vg2_p)]
        assert np.array_equal(ia_fit, np.array([r[0] for r in sc]))
        assert np.array_equal(ig2_fit, np.array([r[1] for r in sc]))


# ---------------------------------------------------------------------------
# 4. Asymptotics
# ---------------------------------------------------------------------------

class TestAsymptotics:
    def test_ig2_fraction_tends_to_fg2(self):
        """Va -> inf (A=0, sigma=0): alpha -> 1-fg2, so Ig2/Ik -> fg2 —
        the v2 fix for the v1 'screen current vanishes' bug (doc 14.1)."""
        q = _make_pentode()
        q = DempwolfParams(**{**q.__dict__, "A": 0.0})
        ia, ig2, _ = dempwolf_v2(1e7, -5.0, 250.0, p=q)
        assert ig2 / (ia + ig2) == pytest.approx(q.fg2, abs=1e-4)

    def test_all_current_to_screen_at_va_zero(self):
        ia, ig2, _ = dempwolf_v2(0.0, -5.0, 250.0, p=_make_pentode())
        assert ia < 0.02 * ig2

    def test_deep_cutoff_no_current(self):
        ia, ig2, _ = dempwolf_v2(250.0, -200.0, 250.0, p=_make_pentode())
        assert ia < 1e-9 and ig2 < 1e-9


# ---------------------------------------------------------------------------
# 5. Secondary emission physics (Reefman par. 6.2)
# ---------------------------------------------------------------------------

def _ia_curve(q, vg, va_c):
    return np.array([dempwolf_v2(float(a), vg, 250.0, p=q)[0] for a in va_c])


class TestSecondaryEmission:
    VA_C = np.linspace(5, 300, 200)

    def test_drawdown_present_iff_sigma_positive(self):
        q = _make_beam()
        ia = _ia_curve(q, -12.0, self.VA_C)
        dd = float(np.max(np.maximum.accumulate(ia) - ia))
        q0 = DempwolfParams(**{**q.__dict__, "sigma": 0.0})
        ia0 = _ia_curve(q0, -12.0, self.VA_C)
        dd0 = float(np.max(np.maximum.accumulate(ia0) - ia0))
        assert dd > 1e-4          # a real dynatron valley (>0.1 mA)
        assert dd0 < dd / 20      # none without secondary emission

    def test_kink_shifts_up_at_more_negative_grid(self):
        """Vco = Vg2/lam - nu*Vg - w with nu > 0: more negative Vg pushes
        the valley to higher Va (Reefman par. 6.2 item 2)."""
        q = _make_beam()

        def valley_va(vg):
            ia = _ia_curve(q, vg, self.VA_C)
            return self.VA_C[int(np.argmin(ia - np.maximum.accumulate(ia)))]

        assert valley_va(-16.0) > valley_va(-8.0)


# ---------------------------------------------------------------------------
# 6. Grid current smoothness (the model's core selling point vs Koren)
# ---------------------------------------------------------------------------

class TestGridCurrentSmoothness:
    def test_derivative_continuous_at_vg_zero(self):
        p = _make_triode_paper(kvb_t=1e-12)
        eps = 1e-6
        ref = (_ig_paper(eps) - _ig_paper(-eps)) / (2 * eps)
        got = (dempwolf_v2(250.0, eps, None, p=p)[2]
               - dempwolf_v2(250.0, -eps, None, p=p)[2]) / (2 * eps)
        assert got == pytest.approx(ref, rel=1e-6)


# ---------------------------------------------------------------------------
# 7. No in-range softplus saturation (_EXP_CLIP regression pins)
# ---------------------------------------------------------------------------

class TestNoInRangeSaturation:
    """The old _EXP_CLIP = 50 saturated the Kp-normalized softplus for
    hand-curated high-C params (C=750, mu=11: arg > 50 for Vg > -6 V at
    Vg2 = 250) — gm collapsed to ~0 near Vg = 0 while the (unclipped)
    SPICE export diverged from Python exactly there."""

    def test_high_c_pentode_gm_alive_near_vg0(self):
        q = DempwolfParams(mu=11.0, G=3e-3, gamma=1.35, C=750.0,
                           Gg=6e-4, xi=1.3, Cg=10.0,
                           Kvb=24.0, Kvb1=0.0, Kn=1.0, fg2=0.05, A=0.0)
        ia = {vg: dempwolf_v2(250.0, vg, 250.0, p=q)[0]
              for vg in (0.0, -2.0, -5.0)}
        # monotone in Vg (clip=50 made this flat / slightly inverted)
        assert ia[0.0] > ia[-2.0] > ia[-5.0]
        gm = (ia[0.0] - ia[-2.0]) / 2.0 * 1e3  # mA/V
        assert gm > 3.0, f"gm={gm:.2f} mA/V — softplus saturated?"

    def test_normalized_softplus_exact_between_50_and_700(self):
        """ln(1+e^x) = x + ln(1+e^-x) exactly; both boundary ends pinned."""
        mu, c, va = 11.0, 750.0, 250.0
        for vg in (-8.0, -2.0, 0.0, 2.0):  # arg ~ 44, 62, 68, 74
            arg = c * (1.0 / mu + vg / va)
            exact = (va / c) * (max(arg, 0.0) + np.log1p(np.exp(-abs(arg))))
            got = _cathode_current(va, vg, mu, 1.0, 1.0, c, normalized=True)
            assert got == pytest.approx(exact, rel=1e-12)

    def test_finite_at_pathological_args(self):
        """arg > 700 must clip, not overflow to inf/nan (scalar and vec)."""
        q = DempwolfParams(mu=5.0, G=3e-3, gamma=1.35, C=2000.0,
                           Gg=6e-4, xi=1.3, Cg=10.0,
                           Kvb=24.0, Kvb1=0.0, Kn=1.0, fg2=0.05, A=0.0)
        ia, ig2, igk = dempwolf_v2(300.0, 3.0, 20.0, p=q)
        assert np.isfinite(ia) and np.isfinite(ig2) and np.isfinite(igk)
        vec = dempwolf_v2_ia_vec(np.array([300.0]), np.array([3.0]),
                                 np.array([20.0]), p=q)
        assert np.isfinite(vec).all()


# ---------------------------------------------------------------------------
# 8. SPICE export guards (mirror of _EXP_CLIP in the .sub text)
# ---------------------------------------------------------------------------

class TestSpiceExportGuards:
    def test_pentode_e1_has_exp_guard(self):
        content = _generate_dempwolf_pentode_subcircuit(
            "TESTPENT", "EL84", _make_pentode(),
            rms_error=1.0, max_error=2.0, n_points=100)
        e1 = [ln for ln in content.splitlines() if "LOG(1+EXP" in ln]
        assert e1, "E1 softplus line missing"
        assert all("MIN(" in ln and ",700)" in ln for ln in e1), \
            "E1 EXP argument must be guarded with MIN(...,700)"

    def test_triode_e1_has_exp_guard(self):
        content = _generate_dempwolf_triode_subcircuit(
            "TESTTRI", "12AX7", _make_triode_paper(kvb_t=300.0),
            rms_error=1.0, max_error=2.0, n_points=100)
        joined = " ".join(content.splitlines())
        assert "LOG(1+EXP(MIN(" in joined and ",700)))" in joined, \
            "triode E1 EXP argument must be guarded with MIN(...,700)"
