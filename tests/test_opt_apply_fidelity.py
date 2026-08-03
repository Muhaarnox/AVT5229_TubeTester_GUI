"""An applied optimizer point is reproduced faithfully.

Pins:
- ML-029 wiring: both Apply paths set the ul_tap spin (including
  zero); the ``pareto_clicked`` signal carries the tap;
- physics: tap 0 vs 0.43 at the same operating point is a documented
  divergence (UL 43% ~ 0.62x Pout, 0.4x THD vs pentode);
- ML-050: ``_analyze_pp`` routes hd_method (DFT/Chebyshev/5-point);
  UL tap really affects the PP panel;
- acceptance: the engine reproduces the optimizer point's THD/Pout
  under the applied parameters;
- physicality audit: UL-ratio pin, CF sanity (THD < MAX_SANE_THD, a
  cathode-follower physical property), mode-matrix smoke,
  cross-method agreement (x3 rule).
- ML-028: expanding a collapsible does not revive state-hidden widgets.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lm19.constants import MAX_SANE_THD_PCT
from lm19.amplifier.constants import (
    CIRCUIT_CF,
    CIRCUIT_PP,
    CIRCUIT_SE,
    HD_METHOD_5POINT,
    HD_METHOD_CHEBYSHEV,
    HD_METHOD_DFT,
    HD_METHOD_DFT_PP,
)
from lm19.amp_engine import (
    WARN_DFT_NO_MODEL,
)

# ── module local constants ──
# Documented UL-43% physics vs pentode (EL84 PP reference):
# Pout ratio ≈ 0.62, THD ratio ≈ 0.4 — pinned as ratios (robust to
# absolute calibration of the model fit).
UL_TAP_WILLIAMSON = 0.43
# At datasheet bias/swing the handbook quotes ~0.62/[~0.4]; the pin
# guards DIRECTION and order of magnitude, not the handbook decimals.
# On the knee-recalibrated EL84 reference (kvb=20,
# KOREN_KNEE_RESEARCH.md) the THD ratio at this rig is ~0.90 — the
# sharper knee weakens the UL-vs-pentode THD contrast; hi=0.93 still
# kills the "UL wrapper dropped" mutation (ratio 1.0).
UL_POUT_RATIO_RANGE = (0.45, 0.85)
UL_THD_RATIO_RANGE = (0.15, 0.93)
CROSS_METHOD_FACTOR = 3.0  # project rule: methods agree within x3


@pytest.fixture(scope="module")
def el84():
    from lm19.tube_sim import quick_pentode
    model, pts = quick_pentode("EL84")
    return model, pts


def _pp_params(**kw):
    from lm19.amp_engine import AmpParams
    base = dict(ub=300.0, ra=8.0, ug1_bias=-11.0, half_swing=9.0,
                circuit=CIRCUIT_PP, pp_raa=8.0, pa_max=12.0, ug2_filter=250.0,
                series_id=0)
    base.update(kw)
    return AmpParams(**base)


# ── ML-029: wiring pins ──────────────────────────────────────────────

class TestApplyUlTapWiring:

    @pytest.fixture()
    def win(self, qapp):
        """Slim host with the two widgets the apply path touches."""
        from unittest.mock import MagicMock
        from app.main_window import MainWindow
        host = MainWindow.__new__(MainWindow)
        host.amp_control_panel = MagicMock()
        return host

    def test_apply_sets_tap_spin(self, win):
        win._apply_opt_point_to_params(300.0, 250.0, -11.0, 8.0, 9.0,
                                       ul_tap=UL_TAP_WILLIAMSON)
        win.amp_control_panel.ul_tap_spin.setValue.assert_called_with(43.0)

    def test_apply_zero_tap_resets_spin(self, win):
        """Mirror: an optimum at tap=0 does not inherit a nonzero spin."""
        win._apply_opt_point_to_params(300.0, 250.0, -11.0, 8.0, 9.0,
                                       ul_tap=0.0)
        win.amp_control_panel.ul_tap_spin.setValue.assert_called_with(0.0)

    def test_pareto_signal_carries_tap(self, qapp):
        from app.amplifier_tab import AmplifierTab
        from lm19.optimizer import OptPoint
        tab = AmplifierTab()
        pt = OptPoint(ub=300.0, ug2=250.0, ug1=-11.0, ra=8.0,
                      thd=3.5, hd2=1.0, hd3=0.5, pout_mw=11000.0,
                      pa_mw=10000.0, ia_0=36.0, ua_0=300.0,
                      amp_class="AB", max_swing=9.0, half_swing=9.0,
                      ul_tap=UL_TAP_WILLIAMSON)
        received: List[float] = []
        tab.pareto_clicked.connect(
            lambda *args: received.extend(args))
        tab.pareto_clicked.emit(pt.ub, pt.ug2, pt.ug1, pt.ra,
                                pt.half_swing, pt.ul_tap)
        assert received[-1] == pytest.approx(UL_TAP_WILLIAMSON)

    def test_topn_apply_passes_tap(self, win, qapp, monkeypatch):
        from unittest.mock import MagicMock
        from lm19.optimizer import OptimizerResult, OptPoint
        pt = OptPoint(ub=300.0, ug2=250.0, ug1=-11.0, ra=8.0,
                      thd=3.5, hd2=1.0, hd3=0.5, pout_mw=11000.0,
                      pa_mw=10000.0, ia_0=36.0, ua_0=300.0,
                      amp_class="AB", max_swing=9.0, half_swing=9.0,
                      ul_tap=UL_TAP_WILLIAMSON)
        win._last_opt_result = OptimizerResult(pareto_front=[pt])
        import app.optimizer_top_n_dialog as tnd
        fake = MagicMock()
        fake.exec.return_value = 1
        fake.DialogCode.Accepted = 1
        fake.selected_point = pt
        monkeypatch.setattr(tnd, "OptimizerTopNDialog",
                            lambda *a, **k: fake)
        applied = {}
        win._apply_opt_point_to_params = (
            lambda ub, ug2, ug1, ra, swing, ul_tap=0.0:
            applied.setdefault("tap", ul_tap))
        win._on_amp_show_top_n()
        assert applied.get("tap") == pytest.approx(UL_TAP_WILLIAMSON)


# -- ML-050 + physics: tap affects the PP panel -----------------------

class TestEnginePpDftRouting:

    def _analyze(self, model, pts, **params_kw):
        from lm19.amp_engine import AmplifierEngine
        eng = AmplifierEngine()
        eng.set_data(pts, is_triode=False, series_models={0: model})
        return eng.analyze(_pp_params(**params_kw))

    def test_dft_method_used(self, el84):
        model, pts = el84
        result = self._analyze(model, pts, hd_method=HD_METHOD_DFT)
        sr = result.per_source["measurements"]
        assert sr.method_used == HD_METHOD_DFT
        assert result.pp_dist is not None
        assert result.pp_dist.get("method") == HD_METHOD_DFT_PP

    def test_chebyshev_method_used(self, el84):
        model, pts = el84
        result = self._analyze(model, pts, hd_method=HD_METHOD_CHEBYSHEV)
        assert result.per_source["measurements"].method_used == HD_METHOD_CHEBYSHEV

    def test_dft_without_model_falls_back_visibly(self, el84):
        from lm19.amp_engine import AmplifierEngine
        _, pts = el84
        eng = AmplifierEngine()
        eng.set_data(pts, is_triode=False)  # no model
        result = eng.analyze(_pp_params(hd_method=HD_METHOD_DFT))
        sr = result.per_source["measurements"]
        assert sr.method_used == HD_METHOD_5POINT
        assert {"code": WARN_DFT_NO_MODEL} in sr.warnings

    def test_ul_tap_changes_pp_panel_numbers(self, el84):
        """Core of ML-029+050: tap 0 vs 0.43 at the same operating
        point — the PP-panel numbers must differ in the documented
        direction (lower Pout, lower THD)."""
        model, pts = el84
        pent = self._analyze(model, pts, hd_method=HD_METHOD_DFT, ul_tap=None)
        ul = self._analyze(model, pts, hd_method=HD_METHOD_DFT,
                           ul_tap=UL_TAP_WILLIAMSON)
        assert pent.pp_dist and ul.pp_dist
        pout_ratio = ul.pp_dist["pout_mw"] / pent.pp_dist["pout_mw"]
        thd_ratio = ul.pp_dist["thd"] / pent.pp_dist["thd"]
        lo, hi = UL_POUT_RATIO_RANGE
        assert lo < pout_ratio < hi, f"Pout ratio {pout_ratio:.2f}"
        lo, hi = UL_THD_RATIO_RANGE
        assert lo < thd_ratio < hi, f"THD ratio {thd_ratio:.2f}"


# -- Acceptance: the engine reproduces the optimizer numbers ----------

class TestAppliedPointReproducesOptimizer:

    def test_engine_matches_optpoint(self, el84):
        from lm19.amp_engine import AmplifierEngine
        from lm19.optimizer import OptimizerConstraints, optimize_pp
        model, pts = el84
        c = OptimizerConstraints(
            circuit=CIRCUIT_PP, pp_raa=8.0, pa_max_w=12.0, hd_method=HD_METHOD_DFT,
            ug1_range=(-12.0, -10.0), ra_range=(8.0, 8.0),
            ug1_steps=3, ra_steps=1, swing_steps=3,
            ul_tap_mode="off", ul_tap_manual=UL_TAP_WILLIAMSON,
        )
        r = optimize_pp(pts, ub=300.0, constraints=c, model=model,
                        ug2_filter=250.0)
        best = r.best
        assert best is not None and best.ul_tap == pytest.approx(
            UL_TAP_WILLIAMSON)

        eng = AmplifierEngine()
        eng.set_data(pts, is_triode=False, series_models={0: model})
        result = eng.analyze(_pp_params(
            hd_method=HD_METHOD_DFT, ub=best.ub, pp_raa=best.ra,
            ug1_bias=best.ug1, half_swing=best.half_swing,
            ul_tap=best.ul_tap))
        assert result.pp_dist is not None
        # Same code path (compute_distortion_dft_pp) -> close match
        assert result.pp_dist["thd"] == pytest.approx(best.thd, rel=0.05)
        assert result.pp_dist["pout_mw"] == pytest.approx(
            best.pout_mw, rel=0.05)


# -- Audit: CF sanity + mode-matrix + cross-method --------------------

class TestOptimizerPhysicalSanityMatrix:

    def test_cf_thd_sane_and_below_se(self):
        """CF with ~100% local NFB: THD well below SE at the same point."""
        from lm19.optimizer import OptimizerConstraints, optimize_measurements
        from lm19.tube_sim import quick_triode
        _, pts = quick_triode("12AU7")
        base = dict(ug1_range=(-10.0, -6.0), ra_range=(10.0, 10.0),
                    ug1_steps=3, ra_steps=1)
        r_cf = optimize_measurements(
            pts, ub=250.0, constraints=OptimizerConstraints(
                circuit=CIRCUIT_CF, cf_rk=10.0, cf_rl=10.0, **base))
        r_se = optimize_measurements(
            pts, ub=250.0, constraints=OptimizerConstraints(
                circuit=CIRCUIT_SE, **base))
        assert r_cf.best is not None and r_se.best is not None
        assert 0.0 < r_cf.best.thd < MAX_SANE_THD_PCT
        assert r_cf.best.thd < r_se.best.thd

    @pytest.mark.parametrize("hd_method", ["5point", "chebyshev", "dft"])
    @pytest.mark.parametrize("circuit,extra", [
        ("se", {}),
        ("se_xfmr", {"ra_dc": 0.05}),
        ("cf", {"cf_rk": 10.0, "cf_rl": 10.0}),
    ])
    def test_mode_matrix_sanity(self, circuit, extra, hd_method):
        """Each circuit x hd_method: THD and Pout are physical, not junk."""
        from lm19.optimizer import OptimizerConstraints, optimize_measurements
        from lm19.tube_sim import quick_triode
        model, pts = quick_triode("12AU7")
        c = OptimizerConstraints(
            circuit=circuit, hd_method=hd_method, **extra,
            ug1_range=(-10.0, -6.0), ra_range=(8.0, 14.0),
            ug1_steps=3, ra_steps=2)
        r = optimize_measurements(pts, ub=250.0, constraints=c,
                                  model=model)
        assert r.best is not None, f"{circuit}/{hd_method}: no best"
        assert 0.0 < r.best.thd < MAX_SANE_THD_PCT, \
            f"{circuit}/{hd_method}: THD={r.best.thd}"
        assert r.best.pout_mw > 0

    def test_cross_method_agreement(self):
        """Project rule: different methods agree on THD within x3."""
        from lm19.optimizer import OptimizerConstraints, optimize_measurements
        from lm19.tube_sim import quick_triode
        model, pts = quick_triode("12AU7")
        thds: Dict[str, float] = {}
        for method in (HD_METHOD_5POINT, HD_METHOD_CHEBYSHEV, HD_METHOD_DFT):
            c = OptimizerConstraints(
                circuit=CIRCUIT_SE, hd_method=method,
                ug1_range=(-8.0, -8.0), ra_range=(10.0, 10.0),
                ug1_steps=1, ra_steps=1)
            r = optimize_measurements(pts, ub=250.0, constraints=c,
                                      model=model)
            assert r.best is not None, method
            thds[method] = r.best.thd
        lo, hi = min(thds.values()), max(thds.values())
        assert hi / max(lo, 1e-9) < CROSS_METHOD_FACTOR, thds


# -- ML-028: collapsible preserves state visibility -------------------

class TestCollapsiblePreservesStateVisibility:

    def test_expand_restores_hidden_flags(self, qapp):
        from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout
        from app.amp_control_panel import (
            _finalize_collapsible, _collapsible_group,
            _toggle_group_content,
        )
        gb = _collapsible_group("g", collapsed=False)
        lay = QVBoxLayout(gb)
        shown = QLabel("shown", gb)
        state_hidden = QLabel("state-hidden", gb)
        lay.addWidget(shown)
        lay.addWidget(state_hidden)
        _finalize_collapsible(gb)
        state_hidden.hide()  # state-driven hiding (e.g. SE circuit)
        _toggle_group_content(gb, False)  # collapse
        _toggle_group_content(gb, True)   # expand
        assert not shown.isHidden()
        assert state_hidden.isHidden(), \
            "expand must NOT resurrect state-hidden widgets"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
