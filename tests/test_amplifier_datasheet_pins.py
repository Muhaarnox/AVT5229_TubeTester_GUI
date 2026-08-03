"""Absolute datasheet pins for transformer stages (EL84).

A mutation/vacuity audit showed the project's documented datasheet claims
(EL84 PP 17 W @ ~9% THD etc.) had no absolute assertion behind them — the
only absolute PP pin ran at a hot bias with no THD check. These tests pin
Pout AND THD at the DOCUMENTED datasheet operating points, with tolerances
wide enough for model error but narrow enough to kill physics regressions
(AC-line sign, Q-point placement, ra_per_tube, UL wiring all land far
outside the bands).

References (external_sources/lamp_datasheets, LAMP_DATASHEETS.md;
true operating rows verified against the primary source,
docs/KOREN_KNEE_RESEARCH.md):
  - Philips EL84 PP (Jan-1969 p.4): Va=Vg2=300 V, Iq≈36 mA/tube,
    Ra-a=8 kΩ, Vi=10 Vrms → Wo=17 W @ dtot=4% (class B row: fixed
    bias −14.7 V, same drive/output).
  - Philips EL84 SE:  Va=Vg2=250 V, Ia≈48 mA (bias ≈ −6.5 V),
    Ra=5.2 kΩ → 5.7 W @ 10% THD.
Measured model values (knee-recalibrated Koren EL84 reference kvb=20,
joint ideal-OPT PP solver): PP at the historical pin
conditions (−11 V, ±9 V): pout_mw 14.7 W @ 9.7%; SE 5.71 W @ 12.2%
(datasheet-exact power). At the TRUE datasheet row the reference is
pinned by TestEl84ReferenceCalibration below.
(History: the former fixed-Ra_aa/4 per-tube solve gave PP 15.8 W —
the joint solve models the impedance kink exactly; the rev-32
EL34-template reference (kvb=48) gave PP 11.8 W @ 11.3% and SE
5.0 W @ 12.7% — the soft knee, since recalibrated.)
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.amplifier import (
    PushPullLoadLine,
    TransformerLoadLine,
    compute_distortion_dft,
    compute_distortion_dft_pp,
)
from lm19.tube_sim import load_model

# ── module local constants ──
# Datasheet anchors ± tolerance. ±40% on Pout covers model error and
# the condition mismatch (these two pins run at the historical ±9 V /
# ±6.5 V drives, below the datasheet Vi=10 Vrms row) while still
# failing on physics regressions (naive load line halves Pout; AC sign
# flip / wrong per-tube impedance push it far out of band). The
# reference calibration itself is pinned tightly by
# TestEl84ReferenceCalibration at the true datasheet row.
_PP_DATASHEET_POUT_W = 17.0
_SE_DATASHEET_POUT_W = 5.7
_POUT_TOL = 0.40
_PP_THD_BAND_PCT = (5.0, 13.0)     # calibrated model: 9.7% @ ±9 V
_SE_THD_BAND_PCT = (7.0, 18.0)     # calibrated model: 12.2%
# TestEl84ReferenceCalibration bands — sized to kill the plausible
# wrong parameter sets measured in KOREN_KNEE_RESEARCH.md:
#   rev-32 EL34-template (kvb=48): knee rms 27%, P1 15.1 W, THD 5.24%
#   kg1 unrescaled (650 @ kvb=20): grid anchor +6.4%
#   published SMB-2008 (kvb=17.9): grid anchor +11%, knee rms 11%
_REF_GRID_ANCHOR_MA = 36.1         # Philips p.4: Iq=36 mA @ (300,-11,300)
_REF_GRID_ANCHOR_REL = 0.05
_REF_KNEE_RMS_MAX_PCT = 10.0       # calibrated: 5.4%; kvb=48: 27%
_REF_CLASSB_P1_BAND_W = (16.0, 21.0)   # calibrated: 18.4; kvb=48: 15.1
_REF_CLASSB_THD_BAND_PCT = (3.0, 5.0)  # calibrated: 4.02; kvb=48: 5.24


@pytest.fixture(scope="module")
def el84():
    model = load_model("EL84")
    assert model is not None
    return model


class TestEl84PpDatasheetPin:
    def test_pp_pentode_pout_and_thd_at_datasheet_bias(self, el84) -> None:
        """Va=Vg2=300, Iq≈36 mA (bias −11 V), Ra-a=8k, ±9 V drive.
        NB: the datasheet 17 W row is at Vi=10 Vrms (±14.1 V) and is
        pinned separately by TestEl84ReferenceCalibration; this pin
        keeps the historical ±9 V regression band. Calibrated model
        (kvb=20): pout_mw 14.7 W @ 9.7%."""
        ll = PushPullLoadLine(ub=300.0, ra_aa=8.0, ra_dc=0.1)
        d = compute_distortion_dft_pp(
            el84, ll, ug1_bias=-11.0, half_swing=9.0, ug2=300.0)
        assert d is not None
        # Bias sanity: this IS the datasheet operating point.
        assert d["iq_per_tube"] == pytest.approx(36.0, rel=0.25)
        pout_w = d["pout_mw"] / 1000.0
        assert (_PP_DATASHEET_POUT_W * (1 - _POUT_TOL)
                <= pout_w
                <= _PP_DATASHEET_POUT_W * (1 + _POUT_TOL)), pout_w
        assert _PP_THD_BAND_PCT[0] <= d["thd"] <= _PP_THD_BAND_PCT[1], d["thd"]


class TestEl84SeXfmrDatasheetPin:
    def test_se_xfmr_pout_and_thd_at_datasheet_point(self, el84) -> None:
        """Va=Vg2=250, Ia≈48 mA (bias −6.5 V), Ra=5.2k → datasheet
        5.7 W @ 10%. Calibrated model: 5.71 W @ 12.2% (power
        datasheet-exact). Runs the full transformer pipeline (DC
        Q-point + AC line through it) — integration-level kill for
        AC-line regressions that unit tests of ia_at_ua_ac cannot see
        (production uses internal closures)."""
        ll = TransformerLoadLine(ub=250.0, ra_dc=0.1, ra_ac=5.2)
        d = compute_distortion_dft(
            el84, ll, ug1_bias=-6.5, half_swing=6.5, ug2=250.0, ub=250.0)
        assert d is not None
        assert d["ia_0"] == pytest.approx(48.0, rel=0.25)   # datasheet Iq
        # Q-point at the supply — the transformer signature. The naive
        # straight line from Ub puts it far below.
        assert d["ua_0"] == pytest.approx(250.0, rel=0.12)
        pout_w = d["pout_mw"] / 1000.0
        assert (_SE_DATASHEET_POUT_W * (1 - _POUT_TOL)
                <= pout_w
                <= _SE_DATASHEET_POUT_W * (1 + _POUT_TOL)), pout_w
        assert _SE_THD_BAND_PCT[0] <= d["thd"] <= _SE_THD_BAND_PCT[1], d["thd"]

    def test_se_xfmr_swing_exceeds_supply(self, el84) -> None:
        """The anode waveform must swing ABOVE Ub (the transformer
        physics the whole #10/M3 story is about) — ua_max from the DFT
        solve, not from a unit formula."""
        ll = TransformerLoadLine(ub=250.0, ra_dc=0.1, ra_ac=5.2)
        d = compute_distortion_dft(
            el84, ll, ug1_bias=-6.5, half_swing=6.5, ug2=250.0, ub=250.0)
        assert d is not None
        assert d["ua_max"] > 250.0 * 1.3      # well above the supply


class TestPpJointSolveKink:
    """Acceptance pins for the class-AB per-tube impedance kink.

    Textbook (external_sources/theory/vactube_push_pull_impedance.html):
    while BOTH tubes conduct each one sees Ra_aa/2 (the partner's
    antiphase current doubles the flux); once the partner cuts off the
    active tube sees Ra_aa/4. The former solver used a fixed Ra_aa/4
    line per tube — these tests discriminate it (it gives Ra_aa/4 in
    class A too)."""

    _RA_PT = 2.0            # kΩ = Ra_aa/4 for Ra_aa = 8k
    _IMPEDANCE_REL_TOL = 0.15

    def _solve(self, el84, bias: float, swing: float, n: int = 64):
        import numpy as np
        from lm19.amplifier.distortion import _pp_joint_solve_vec
        from lm19.tube_model_base import model_ia_array

        ug2 = 300.0
        ua_q = 300.0
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        ug1_a = bias + swing * np.sin(t)
        ug1_b = bias - swing * np.sin(t)

        def ia_fn(ua, g):
            return model_ia_array(el84, ua, g, ug2)

        ia_a, ia_b, ua_a, _ua_b, n_div, _ = _pp_joint_solve_vec(
            ia_fn, ia_fn, ug1_a, ug1_b, ua_q, ua_q, self._RA_PT, 40)
        assert n_div == 0
        return ia_a, ia_b, ua_a, ua_q

    def test_class_a_per_tube_impedance_is_zaa_over_2(self, el84) -> None:
        """Small signal at hot bias — both tubes conduct all cycle →
        per-tube dUa/dIa must be 2×Ra_per_tube = Ra_aa/2. The former
        fixed-line solver gives exactly Ra_per_tube here → fails."""
        import numpy as np
        ia_a, ia_b, ua_a, _ = self._solve(el84, bias=-9.0, swing=0.5)
        assert float(np.min(ia_b)) > 1.0        # genuinely class A
        # Impedance the SOLVER actually applied: anode-voltage excursion
        # per unit of own-current excursion (not reconstructed from the
        # constraint — that would be circular).
        ratio = ((np.max(ua_a) - np.min(ua_a))
                 / (np.max(ia_a) - np.min(ia_a)))
        assert ratio == pytest.approx(
            2.0 * self._RA_PT, rel=self._IMPEDANCE_REL_TOL)

    def test_class_b_region_per_tube_impedance_is_zaa_over_4(self, el84) -> None:
        """Deep bias, big swing — where the partner is cut off the
        active tube must see exactly Ra_per_tube = Ra_aa/4:
        v = Ra_pt × ia_a when ia_b = 0 (circuit equation)."""
        import numpy as np
        ia_a, ia_b, ua_a, ua_q = self._solve(el84, bias=-14.0, swing=10.0)
        # "Off" = below 0.01 mA: the Koren softplus tail never reaches
        # exactly 0; its contribution to v is < ra·0.01 mA = 0.02 V.
        solo = (ia_b < 0.01) & (ia_a > 5.0)     # partner off, tube A hot
        assert solo.sum() > 5                    # class-B region exists
        # Solver-applied per-tube line in the solo region: (ua_q − ua)/ia.
        ratio = (ua_q - ua_a[solo]) / ia_a[solo]
        assert np.allclose(ratio, self._RA_PT, rtol=0.02)


class TestOtherPentodeReferencesSanity:
    """Coarse sanity pins of the EL34/6L6/KT88 references (see
    KOREN_KNEE_RESEARCH.md, other-tubes section): the sets are genuine
    Koren publications, uniformly 'hot' at +4..+15% against datasheet
    printed tables (bogey spread; knee shapes are healthy — NOT the
    EL84 disease), hence the wide band (+-25%): the pin only catches
    gross tube_params.json corruption (a typo, a parameter swap, a
    lost multiplier), not specimen drift."""

    _GRID_TOL_REL = 0.25

    @pytest.fixture(scope="class")
    def research(self):
        import importlib.util
        path = os.path.join(os.path.dirname(__file__), "..", "tools",
                            "koren_knee_research.py")
        spec = importlib.util.spec_from_file_location(
            "koren_knee_research_sanity", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @pytest.mark.parametrize("tube", ["EL34", "6L6", "KT88"])
    def test_grid_anchors_within_bogey_scatter(self, tube,
                                               research) -> None:
        import numpy as np
        from lm19.tube_model_base import model_ia_array
        from lm19.tube_sim import load_model
        model = load_model(tube)
        assert model is not None
        for ua, g1, g2, want in research._OTHER_TUBES[tube]["grid_anchors"]:
            got = float(model_ia_array(model, np.array([ua]),
                                       np.array([g1]), g2)[0])
            assert got == pytest.approx(want, rel=self._GRID_TOL_REL), (
                f"{tube} Ia({ua},{g1},{g2}) = {got:.1f}, ds {want}")


class TestEl84ReferenceCalibration:
    """Pins the knee-recalibrated EL84 reference (A1,
    docs/KOREN_KNEE_RESEARCH.md): kvb 48→20 + kg1 650→692.

    Anchors are imported from tools/koren_knee_research.py — the single
    source of truth for the Philips Jan-1969 readings (p.8 Vg2=300
    curve family ±5-8 mA; p.4 operating rows). Bands are sized to kill
    every plausible wrong parameter set measured in the research:
    rev-32 template (kvb=48), kg1 left unrescaled, published SMB-2008.
    Before this pin the reference was UNPINNED — the dry run showed
    both kvb=48 and kvb=20 passing the wide ±40% bands above.
    """

    @pytest.fixture(scope="class")
    def research(self):
        import importlib.util
        path = os.path.join(os.path.dirname(__file__), "..", "tools",
                            "koren_knee_research.py")
        spec = importlib.util.spec_from_file_location(
            "koren_knee_research_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_grid_anchor_preserved(self, el84, research) -> None:
        """Ia(300,−11,300) = 36 mA (Philips p.4 Iq). Kills kg1 left
        unrescaled (650 @ kvb=20 → +6.4%) and the published SMB-2008
        set (+11%) — the calibration must not trade the grid region
        for the knee."""
        import numpy as np
        from lm19.tube_model_base import model_ia_array
        ua, g1, g2, _ = research._GRID_ANCHOR
        ia = float(model_ia_array(el84, np.array([ua]),
                                  np.array([g1]), g2)[0])
        assert ia == pytest.approx(_REF_GRID_ANCHOR_MA,
                                   rel=_REF_GRID_ANCHOR_REL)

    def test_knee_anchors_match_datasheet_curves(self, el84,
                                                 research) -> None:
        """RMS over the 7 Philips p.8 knee anchors < 10%. The rev-32
        set (kvb=48) sits at 27% — a silent revert fails loudly."""
        import math
        errs = research._knee_errors(el84)
        rms = math.sqrt(sum(e * e for e in errs) / len(errs))
        assert rms < _REF_KNEE_RMS_MAX_PCT, f"knee rms {rms:.1f}%"

    def test_class_b_true_datasheet_row(self, el84, research) -> None:
        """The TRUE Philips p.4 class-B row (bias −14.7, Vi=10 Vrms,
        Ra-aa=8k): Wo=17 W @ dtot=4%. Calibrated model: P1=18.4 W @
        4.02%. Fundamental power (pout_fund_mw), NOT the peak-based
        pout_mw — the metric mismatch was half of the historical
        «11.8 vs 17 W» confusion."""
        r = research._pp_row(el84, research._DS_BIAS_B,
                             research._VI_PEAK_V)
        lo, hi = _REF_CLASSB_P1_BAND_W
        assert lo < r["p1_w"] < hi, f"P1 {r['p1_w']:.2f} W"
        lo, hi = _REF_CLASSB_THD_BAND_PCT
        assert lo < r["thd"] < hi, f"THD {r['thd']:.2f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
