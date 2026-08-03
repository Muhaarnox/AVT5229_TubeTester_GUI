"""Dedup guards (ML-132 / ML-133 / ML-135), maintainability.

- ML-132: compute_pa_avg is solved by the shared _newton_solve_vec
  (the last inline copy of the DFT-Newton with magic
  0.5/1e-6/+-1.0/30 was removed). Equivalence is proven against a
  FROZEN copy of the old scalar loop — bit-exact, including the
  n_not_converged counter on a non-converging scenario.
- ML-133: the gm/ra priority (model -> numerical -> SRK) lives in
  _resolve_tube_params; both call sites (SE and the CF twin) must
  go through it and produce the same resolution.
- ML-135: cluster_nominal/nominal_key — single implementation in
  lm19/plotting/grids.py; curve_data imports the same one
  (identity), the ratchet bans new private copies.

Run:  py -m pytest tests/test_dedup_guards.py -v
"""

from __future__ import annotations

import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lm19.amplifier.constants import (
    FIXED_POINT_CONVERGENCE_MA,
    MIN_SWING_V,
    N_PA_SAMPLES,
)
from lm19.amplifier.distortion import (
    _DFT_NEWTON_MAX_ITER_SE,
    _find_model_dc_q_point,
)
from lm19.amplifier.loadlines import (
    CathodeFollowerLoadLine,
    PushPullLoadLine,
    ResistiveLoadLine,
    TransformerLoadLine,
)
from lm19.amplifier.sweeps import compute_pa_avg
from lm19.amplifier.stage_params import (
    _resolve_tube_params,
    compute_cf_stage_params,
    compute_stage_params,
)
from lm19.constants import DEFAULT_UB_V, MODEL_UA_MAX_DEFAULT_V, MODEL_UA_MIN_V


# ═══════════════════════════════════════════════════════════════════
#  Fake models (both ia and ia_array use bit-identical arithmetic:
#  d·sqrt(d) — math.sqrt/np.sqrt are IEEE-correctly-rounded, unlike **1.5)
# ═══════════════════════════════════════════════════════════════════

class _SpaceChargeModel:
    """Koren-like triode: ia = k*max(ug1 + ua/mu, 0)^(3/2) (mA)."""

    def __init__(self, k: float = 2.0, mu: float = 10.0) -> None:
        self.k = k
        self.mu = mu

    def ia(self, ua: float, ug1: float, ug2: float = 0.0) -> float:
        d = max(ug1 + ua / self.mu, 0.0)
        return self.k * d * math.sqrt(d)

    def ia_array(self, ua: np.ndarray, ug1: np.ndarray,
                 ug2: float = 0.0) -> np.ndarray:
        d = np.maximum(np.asarray(ug1, dtype=float)
                       + np.asarray(ua, dtype=float) / self.mu, 0.0)
        return self.k * d * np.sqrt(d)


class _SpaceChargeScalarOnly:
    """Same law but WITHOUT ia_array — pins the model_ia_array
    fallback (does not inherit _SpaceChargeModel to skip the method)."""

    def __init__(self, k: float = 2.0, mu: float = 10.0) -> None:
        self.k = k
        self.mu = mu

    def ia(self, ua: float, ug1: float, ug2: float = 0.0) -> float:
        d = max(ug1 + ua / self.mu, 0.0)
        return self.k * d * math.sqrt(d)


class _StepModel:
    """Ia discontinuity: the load line has no root -> Newton fails to
    converge deterministically (oscillates around the jump)."""

    def ia(self, ua: float, ug1: float, ug2: float = 0.0) -> float:
        return 200.0 if ua >= 100.0 else 0.0


class _LinearModel:
    """ia = i0 + gm*(ug1-ug1_ref) + slope*(ua-ua_ref): model_gm_ra
    yields exactly gm and ra = 1/slope (central differences are linear)."""

    def __init__(self, gm: float = 5.0, slope: float = 0.1,
                 i0: float = 10.0, ug1_ref: float = -2.0,
                 ua_ref: float = 200.0) -> None:
        self.gm = gm
        self.slope = slope
        self.i0 = i0
        self.ug1_ref = ug1_ref
        self.ua_ref = ua_ref

    def ia(self, ua: float, ug1: float, ug2: float = 0.0) -> float:
        return max(0.0, self.i0 + self.gm * (ug1 - self.ug1_ref)
                   + self.slope * (ua - self.ua_ref))


# ═══════════════════════════════════════════════════════════════════
#  ML-132: frozen scalar reference (pre-ML-132 body, verbatim)
# ═══════════════════════════════════════════════════════════════════

def _pa_avg_scalar_reference(model, load_line, ug1_bias: float,
                             half_swing: float, ug2: float = 0.0,
                             ub: Optional[float] = None) -> Optional[Dict]:
    """Frozen copy of the compute_pa_avg body BEFORE ML-132 (scalar
    Newton with inline magic). Do not edit — equivalence reference."""
    if half_swing < MIN_SWING_V:
        return None

    ll_fn = load_line.ia_at_ua
    if isinstance(load_line, (TransformerLoadLine, PushPullLoadLine)):
        if isinstance(load_line, TransformerLoadLine):
            ra_dc, ra_ac = load_line.ra_dc, load_line.ra_ac
        else:
            ra_dc, ra_ac = load_line.ra_dc, load_line.ra_per_tube
        q = _find_model_dc_q_point(
            model, load_line.ub, ra_dc, ug1_bias, ug2,
            (MODEL_UA_MIN_V, MODEL_UA_MAX_DEFAULT_V),
        )
        if q is not None:
            q_ua, q_ia = q

            def ll_fn(ua, _q_ua=q_ua, _q_ia=q_ia, _ra=ra_ac):
                if _ra <= 0:
                    return _q_ia
                return _q_ia - (ua - _q_ua) / _ra

    pa_sum = 0.0
    ia_sum = 0.0
    pa_peak = 0.0
    ia_peak = 0.0

    eps = load_line.endpoints()
    ua_guess = (eps[0][0] + eps[1][0]) / 2.0 if eps else (ub or DEFAULT_UB_V)

    not_converged = 0
    for i in range(N_PA_SAMPLES):
        theta = 2.0 * math.pi * i / N_PA_SAMPLES
        ug1_i = ug1_bias + half_swing * math.sin(theta)

        ua = ua_guess
        converged = False
        for _ in range(30):
            ia_m = model.ia(ua, ug1_i, ug2)
            ia_l = ll_fn(ua)
            err = ia_m - ia_l
            if abs(err) < FIXED_POINT_CONVERGENCE_MA:
                converged = True
                break
            ua_test = ua + 0.5
            err2 = model.ia(ua_test, ug1_i, ug2) - ll_fn(ua_test)
            deriv = (err2 - err) / 0.5
            if abs(deriv) > 1e-6:
                ua -= err / deriv
            else:
                ua += 1.0 if err > 0 else -1.0
            ua = max(0.0, ua)
        if not converged:
            not_converged += 1

        ia_val = max(0.0, model.ia(ua, ug1_i, ug2))
        pa_val = ua * ia_val

        pa_sum += pa_val
        ia_sum += ia_val
        if pa_val > pa_peak:
            pa_peak = pa_val
        if ia_val > ia_peak:
            ia_peak = ia_val

    return {
        "pa_avg_mw": pa_sum / N_PA_SAMPLES,
        "ia_avg": ia_sum / N_PA_SAMPLES,
        "pa_peak_mw": pa_peak,
        "ia_peak": ia_peak,
        "n_not_converged": not_converged,
    }


_EQ_KEYS = ("pa_avg_mw", "ia_avg", "pa_peak_mw", "ia_peak",
            "n_not_converged")


class TestPaAvgSharedNewton:
    """ML-132: the shared solver is bit-identical to the old loop."""

    def _assert_equal(self, model, ll, bias: float, swing: float,
                      ug2: float = 0.0, ub: Optional[float] = None) -> Dict:
        ref = _pa_avg_scalar_reference(model, ll, bias, swing, ug2, ub)
        new = compute_pa_avg(model, ll, ug1_bias=bias, half_swing=swing,
                             ug2=ug2, ub=ub)
        assert ref is not None and new is not None
        for k in _EQ_KEYS:
            assert new[k] == ref[k], f"{k}: {new[k]!r} != {ref[k]!r}"
        return new

    def test_resistive_class_a_bit_identical(self):
        model = _SpaceChargeModel()
        ll = ResistiveLoadLine(ub=300.0, ra=10.0)
        r = self._assert_equal(model, ll, bias=-8.0, swing=3.0, ub=300.0)
        assert r["n_not_converged"] == 0
        assert r["pa_avg_mw"] > 0

    def test_resistive_overdrive_bit_identical(self):
        # Deep overdrive: part of the period in cutoff (ia=0) — the
        # kick/clamp path must match as well.
        model = _SpaceChargeModel()
        ll = ResistiveLoadLine(ub=300.0, ra=10.0)
        self._assert_equal(model, ll, bias=-25.0, swing=20.0, ub=300.0)

    def test_transformer_twin_bit_identical(self):
        model = _SpaceChargeModel()
        ll = TransformerLoadLine(ub=250.0, ra_dc=0.2, ra_ac=5.0)
        self._assert_equal(model, ll, bias=-10.0, swing=5.0, ub=250.0)

    def test_pp_twin_bit_identical(self):
        model = _SpaceChargeModel()
        ll = PushPullLoadLine(ub=300.0, ra_aa=8.0)
        self._assert_equal(model, ll, bias=-12.0, swing=6.0, ub=300.0)

    def test_fallback_model_without_ia_array_bit_identical(self):
        model = _SpaceChargeScalarOnly()
        assert not hasattr(model, "ia_array")
        ll = ResistiveLoadLine(ub=300.0, ra=10.0)
        self._assert_equal(model, ll, bias=-8.0, swing=3.0, ub=300.0)

    def test_nonconverged_counter_matches_reference(self, caplog):
        # Ia discontinuity -> no root -> no sample converges; counter
        # and WARNING (ML-096) must match the reference.
        model = _StepModel()
        ll = ResistiveLoadLine(ub=300.0, ra=3.0)
        with caplog.at_level(logging.WARNING, logger="lm19.amplifier.sweeps"):
            r = self._assert_equal(model, ll, bias=-5.0, swing=2.0, ub=300.0)
        assert r["n_not_converged"] > 0
        assert isinstance(r["n_not_converged"], int)
        assert any("did not converge" in rec.message
                   for rec in caplog.records)

    def test_call_site_uses_shared_solver(self, monkeypatch):
        """Dedup discriminator: reverting to the scalar loop keeps
        equivalence — only a call-site spy catches it."""
        import lm19.amplifier.sweeps as sw
        seen: Dict = {}
        orig = sw._newton_solve_vec

        def spy(ia_fn, ll_fn, ug1_arr, ua_init, max_iter):
            seen["max_iter"] = max_iter
            seen["n"] = int(np.asarray(ug1_arr).shape[0])
            return orig(ia_fn, ll_fn, ug1_arr, ua_init, max_iter)

        monkeypatch.setattr(sw, "_newton_solve_vec", spy)
        r = compute_pa_avg(_SpaceChargeModel(),
                           ResistiveLoadLine(ub=300.0, ra=10.0),
                           ug1_bias=-8.0, half_swing=3.0, ub=300.0)
        assert r is not None
        assert seen.get("max_iter") == _DFT_NEWTON_MAX_ITER_SE
        assert seen.get("n") == N_PA_SAMPLES


# ═══════════════════════════════════════════════════════════════════
#  ML-133: the single gm/ra resolver and both of its call sites
# ═══════════════════════════════════════════════════════════════════

def _make_intersections() -> List[Dict]:
    return [
        {"ug1": -1.0, "ua": 205.0, "ia": 12.0},
        {"ug1": -3.0, "ua": 195.0, "ia": 8.0},
    ]


def _make_numerical_points(gm: float = 2.0, slope: float = 0.05) -> List[Dict]:
    """Linear law -> _numerical_gm_ra yields exactly gm and ra=1/slope."""
    pts: List[Dict] = []
    for ug1 in (-1.0, -2.0, -3.0):
        for ua in range(160, 250, 10):
            ia = 10.0 + gm * (ug1 + 2.0) + slope * (float(ua) - 200.0)
            pts.append({"ua": float(ua), "ug1": ug1, "ia": ia})
    return pts


class TestResolveTubeParams:
    """Priority model -> numerical -> SRK; the sources give DIFFERENT
    values (degenerate data would hide a wrong routing)."""

    _SRK = {"s": 1.0, "r": 40.0}          # gm=1, ra=40 — unlike others
    _MODEL = _LinearModel(gm=5.0, slope=0.1)   # gm=5, ra=10
    # points: gm=2, ra=20

    def test_model_wins_over_numerical_and_srk(self):
        r = _resolve_tube_params(_make_intersections(), -2.0, self._SRK,
                                 _make_numerical_points(), self._MODEL, 0.0)
        assert r is not None
        assert r["method"] == "model"
        assert r["gm"] == pytest.approx(5.0, rel=1e-6)
        assert r["ra"] == pytest.approx(10.0, rel=1e-6)

    def test_numerical_when_no_model(self):
        r = _resolve_tube_params(_make_intersections(), -2.0, self._SRK,
                                 _make_numerical_points(), None, 0.0)
        assert r is not None
        assert r["method"] == "numerical"
        assert r["gm"] == pytest.approx(2.0, rel=1e-6)
        assert r["ra"] == pytest.approx(20.0, rel=1e-6)

    def test_srk_fallback_with_and_without_k(self):
        r = _resolve_tube_params(_make_intersections(), -2.0,
                                 {"s": 1.0, "r": 40.0, "k": 77.0},
                                 None, None, 0.0)
        assert r is not None
        assert (r["method"], r["gm"], r["ra"], r["mu"]) == \
            ("srk", 1.0, 40.0, 77.0)
        assert r["srk_check"] is None      # SRK is the source — no cross-check
        r2 = _resolve_tube_params(_make_intersections(), -2.0,
                                  {"s": 1.0, "r": 40.0}, None, None, 0.0)
        assert r2 is not None and r2["mu"] == pytest.approx(40.0)

    def test_none_when_no_source(self):
        assert _resolve_tube_params(_make_intersections(), -2.0,
                                    None, None, None, 0.0) is None
        # Falsy s — SRK inapplicable (boundary: 0 does not count as data).
        assert _resolve_tube_params(_make_intersections(), -2.0,
                                    {"s": 0.0, "r": 40.0},
                                    None, None, 0.0) is None

    def test_srk_cross_check_divergence_propagates_to_both_twins(self):
        """Twins: SE and CF must propagate divergence identically."""
        srk_off = {"s": 15.0, "r": 10.0}   # gm x3 vs model -> divergence
        inter = _make_intersections()
        se = compute_stage_params(inter, ResistiveLoadLine(ub=300.0, ra=10.0),
                                  ug1_bias=-2.0, srk=srk_off,
                                  model=self._MODEL)
        cf = compute_cf_stage_params(inter,
                                     CathodeFollowerLoadLine(
                                         ub=300.0, rk=10.0, rl=0.0),
                                     ug1_bias=-2.0, srk=srk_off,
                                     model=self._MODEL)
        assert se is not None and cf is not None
        assert se["method"] == cf["method"] == "model"
        assert se["srk_check"] == cf["srk_check"] == "divergence"
        assert se["srk_divergence_pct"] == cf["srk_divergence_pct"] > 0

    def test_both_call_sites_route_through_resolver(self, monkeypatch):
        """Call site vs function: an inline block returning in EITHER
        of the two functions is visible only through a resolver spy."""
        import lm19.amplifier.stage_params as sp
        calls: List[str] = []
        orig = sp._resolve_tube_params

        def spy(*args, **kwargs):
            calls.append("hit")
            return orig(*args, **kwargs)

        monkeypatch.setattr(sp, "_resolve_tube_params", spy)
        inter = _make_intersections()
        se = compute_stage_params(inter, ResistiveLoadLine(ub=300.0, ra=10.0),
                                  ug1_bias=-2.0, model=self._MODEL)
        assert se is not None and len(calls) == 1
        cf = compute_cf_stage_params(inter,
                                     CathodeFollowerLoadLine(
                                         ub=300.0, rk=10.0, rl=0.0),
                                     ug1_bias=-2.0, model=self._MODEL)
        assert cf is not None and len(calls) == 2

    def test_twins_agree_on_resolution(self):
        """Same inputs -> same gm/ra/mu/method in SE and CF."""
        inter = _make_intersections()
        pts = _make_numerical_points()
        se = compute_stage_params(inter, ResistiveLoadLine(ub=300.0, ra=10.0),
                                  ug1_bias=-2.0, points=pts)
        cf = compute_cf_stage_params(inter,
                                     CathodeFollowerLoadLine(
                                         ub=300.0, rk=10.0, rl=0.0),
                                     ug1_bias=-2.0, points=pts)
        assert se is not None and cf is not None
        for k in ("gm", "ra", "mu", "method"):
            assert se[k] == cf[k], k


# ═══════════════════════════════════════════════════════════════════
#  ML-135: single implementation of cluster_nominal / nominal_key
# ═══════════════════════════════════════════════════════════════════

class TestClusterNominalSingleSource:

    def test_curve_data_aliases_are_grids_functions(self):
        from lm19 import curve_data
        from lm19.plotting import grids
        assert curve_data._cluster_nominal is grids.cluster_nominal
        assert curve_data._nominal_key is grids.nominal_key

    def test_single_definition_ratchet(self):
        """Exactly one def cluster_nominal/nominal_key in the project
        (lm19/, app/, tools/) — new private copies fail the test."""
        pat = re.compile(
            r"^\s*def\s+_?(cluster_nominal|nominal_key)\s*\(", re.M)
        offenders: List[str] = []
        for sub in ("lm19", "app", "tools"):
            for py in (PROJECT_ROOT / sub).rglob("*.py"):
                for m in pat.finditer(py.read_text(encoding="utf-8")):
                    offenders.append(f"{py.relative_to(PROJECT_ROOT)}"
                                     f":{m.group(0).strip()}")
        assert sorted(offenders) == [
            "lm19\\plotting\\grids.py:def cluster_nominal(",
            "lm19\\plotting\\grids.py:def nominal_key(",
        ], offenders

    def test_shuffled_input_clusters_identically(self):
        """Unsorted input (sorted != input order) — a mutation without
        internal sorting is visible."""
        from lm19.curve_data import _cluster_nominal, _nominal_key
        shuffled = [5.03, 0.0, 5.0, 0.02, 10.1, 10.08]
        noms = _cluster_nominal(shuffled, threshold=0.1)
        assert noms == [0.0, 5.0, 10.08]
        assert _nominal_key(5.01, noms) == 5.0
        assert _nominal_key(10.2, noms) == 10.08
