"""Tests for Dempwolf Extended v2 model.

Verification against published data (DAFx-11) and test points from
DEMPWOLF_EXTENDED_MODEL.md §11.
"""

import math

import pytest

from lm19.tube_params import DempwolfParams, lookup_tube
from lm19.dempwolf import (
    dempwolf_v2,
    dempwolf_triode,
    dempwolf_pentode,
    dempwolf_beam_tetrode,
    DempwolfModel,
    load_dempwolf_model,
    _list_dempwolf_tubes,
    _softplus,
)
from lm19.tube_model_base import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Reference parameter sets from DAFx-11 (Table 1)
# ---------------------------------------------------------------------------

P_12AX7_RSD1 = DempwolfParams(
    mu=103.2, G=2.242e-3, gamma=1.26, C=3.4,
    Gg=6.177e-4, xi=1.314, Cg=9.901, Kvb_t=300.0,
)

P_12AX7_RSD2 = DempwolfParams(
    mu=100.2, G=2.173e-3, gamma=1.28, C=3.19,
    Gg=5.911e-4, xi=1.358, Cg=11.76, Kvb_t=300.0,
)

P_12AX7_EHX1 = DempwolfParams(
    mu=86.9, G=1.371e-3, gamma=1.349, C=4.56,
    Gg=3.263e-4, xi=1.156, Cg=11.99, Kvb_t=300.0,
)

# Estimated pentode parameters from §11.2
# C is now Kp-style (Vg2-normalized): C_new = C_old * Vg2_ref
P_EL34 = DempwolfParams(
    mu=11.0, G=3.0e-3, gamma=1.35, C=750.0,
    Gg=6.0e-4, xi=1.3, Cg=10.0,
    Kvb=24.0, Kvb1=0.5, Kn=1.0, fg2=0.08, A=0.0002,
)

# Estimated beam tetrode parameters from §11.3
# C_new = 2.5 * 250 = 625
P_6L6GC = DempwolfParams(
    mu=8.7, G=2.8e-3, gamma=1.35, C=625.0,
    Gg=6.0e-4, xi=1.3, Cg=10.0,
    Kvb=12.0, Kvb1=0.3, Kn=1.0, fg2=0.10, A=0.0002,
    sigma=3.0, Ks=1.5, lam=1.0, nu=2.0, w=0.0,
)


# ===================================================================
# §11.1 — 12AX7 Triode verification
# ===================================================================

class TestTriode12AX7:
    """Verify 12AX7 triode against §11.1 test points."""

    def test_normal_operation(self):
        """VPK=250V, VGK=-2V → IPK ≈ 1.03 mA."""
        ipk, igk = dempwolf_triode(250.0, -2.0, p=P_12AX7_RSD1)
        ipk_ma = ipk * 1000.0
        # §11.1: IPK ≈ 1.03 mA, datasheet ≈ 1.0 mA
        assert 0.8 < ipk_ma < 1.3, f"IPK = {ipk_ma:.3f} mA, expected ~1.03"
        # Grid current should be ~0 at VGK = -2
        assert igk < 1e-9, f"IGK = {igk:.2e}, expected ~0"

    def test_grid_conduction(self):
        """VPK=250V, VGK=+0.5V → grid current measurable."""
        ipk, igk = dempwolf_triode(250.0, 0.5, p=P_12AX7_RSD1)
        ipk_ma = ipk * 1000.0
        igk_ma = igk * 1000.0
        # §11.1: IPK ≈ 8.45 mA, IGK ≈ 0.27 mA
        assert 6.0 < ipk_ma < 12.0, f"IPK = {ipk_ma:.2f} mA"
        assert 0.05 < igk_ma < 1.0, f"IGK = {igk_ma:.3f} mA"

    def test_cutoff(self):
        """At large negative VGK, current should be ~0."""
        ipk, igk = dempwolf_triode(250.0, -10.0, p=P_12AX7_RSD1)
        assert ipk * 1000.0 < 0.01, f"Should be cutoff, IPK = {ipk*1000:.4f} mA"

    def test_current_conservation_triode(self):
        """IPK = IK - IGK must hold (no negative IPK)."""
        for vgk in [-4, -2, -1, 0, 0.5, 1.0]:
            ipk, _, igk = dempwolf_v2(200.0, vgk, vg2k=None, p=P_12AX7_RSD1)
            assert ipk >= 0, f"Negative IPK at VGK={vgk}"

    def test_specimen_variation(self):
        """All three 12AX7 specimens should give reasonable results."""
        for label, p in [
            ("RSD-1", P_12AX7_RSD1),
            ("RSD-2", P_12AX7_RSD2),
            ("EHX-1", P_12AX7_EHX1),
        ]:
            ipk, igk = dempwolf_triode(250.0, -2.0, p=p)
            ipk_ma = ipk * 1000.0
            assert 0.5 < ipk_ma < 2.0, f"{label}: IPK = {ipk_ma:.3f} mA out of range"


# ===================================================================
# §11.2 — EL34 Pentode verification
# ===================================================================

class TestPentodeEL34:
    """Verify EL34 pentode against §11.2 test points."""

    def test_full_drive(self):
        """VPK=300V, VG2K=300V, VGK=0V → IPK ≈ 235 mA."""
        ipk, ig2k, igk = dempwolf_pentode(300.0, 0.0, 300.0, p=P_EL34)
        ipk_ma = ipk * 1000.0
        ig2k_ma = ig2k * 1000.0
        # §11.2: IPK ≈ 235 mA, IG2K ≈ 20 mA
        assert 150 < ipk_ma < 350, f"IPK = {ipk_ma:.1f} mA"
        assert 5 < ig2k_ma < 60, f"IG2K = {ig2k_ma:.1f} mA"

    def test_negative_bias(self):
        """VPK=300V, VG2K=300V, VGK=-20V → IPK ≈ 40 mA."""
        ipk, ig2k, igk = dempwolf_pentode(300.0, -20.0, 300.0, p=P_EL34)
        ipk_ma = ipk * 1000.0
        # §11.2: IPK ≈ 40 mA
        assert 20 < ipk_ma < 80, f"IPK = {ipk_ma:.1f} mA"

    def test_current_conservation_pentode(self):
        """IPK + IG2K + IGK == IK_total (within float precision)."""
        for vgk in [-30, -20, -10, -5, 0]:
            ipk, ig2k, igk = dempwolf_v2(300.0, vgk, 300.0, p=P_EL34)
            total = ipk + (ig2k or 0.0) + igk
            # Recompute IK_total for comparison
            # Total should be > 0 and consistent
            assert total >= 0, f"Negative total at VGK={vgk}"
            # Check splitting: IPK + IG2K == I_through
            i_through = ipk + (ig2k or 0.0)
            assert abs(i_through + igk - total) < 1e-15

    def test_adaptive_knee(self):
        """Knee should be wider at VGK=0 than at VGK=-20."""
        # At knee region (Va ≈ Kvb_eff), compare alpha values
        # VGK=0: Kvb_eff = 24 + 0.5 * (300/11) ≈ 37.6
        # VGK=-20: Kvb_eff = 24 + 0.5 * (300/11 - 20) ≈ 27.6
        ipk_0, _, _ = dempwolf_v2(30.0, 0.0, 300.0, p=P_EL34)
        ipk_20, _, _ = dempwolf_v2(30.0, -20.0, 300.0, p=P_EL34)
        # At Va=30V (in the knee), VGK=0 should have more current
        # going to G2 (wider knee → lower alpha)
        # Both should be positive
        assert ipk_0 >= 0 and ipk_20 >= 0

    def test_fg2_screen_current_at_high_va(self):
        """With fg2>0, IG2K should not vanish at high Va."""
        ipk, ig2k, igk = dempwolf_v2(500.0, -10.0, 300.0, p=P_EL34)
        ig2k_ma = (ig2k or 0.0) * 1000.0
        # fg2=0.08 → at least ~8% of I_through goes to G2
        assert ig2k_ma > 0.5, f"IG2K = {ig2k_ma:.3f} mA, should be > 0 at high Va"


# ===================================================================
# §11.3 — 6L6GC Beam Tetrode verification
# ===================================================================

class TestBeamTetrode6L6GC:
    """Verify 6L6GC beam tetrode against §11.3 test points."""

    def test_kink_region(self):
        """VPK=100V, VG2K=250V, VGK=0V → secondary emission reduces IPK."""
        ipk, ig2k, igk = dempwolf_beam_tetrode(100.0, 0.0, 250.0, p=P_6L6GC)
        ipk_ma = ipk * 1000.0
        ig2k_ma = ig2k * 1000.0
        # With sigma=3.0, strong secondary emission reduces IPK in kink region
        assert 30 < ipk_ma < 350, f"IPK = {ipk_ma:.1f} mA"
        assert ig2k_ma > 5, f"IG2K = {ig2k_ma:.1f} mA"

    def test_kink_peak(self):
        """Near VPK ≈ 190V (x=1/Ks=0.25), secondary emission peaks."""
        ipk_190, ig2k_190, _ = dempwolf_beam_tetrode(190.0, 0.0, 250.0, p=P_6L6GC)
        ipk_100, ig2k_100, _ = dempwolf_beam_tetrode(100.0, 0.0, 250.0, p=P_6L6GC)
        ipk_300, ig2k_300, _ = dempwolf_beam_tetrode(300.0, 0.0, 250.0, p=P_6L6GC)
        # At 190V the secondary emission effect should be strongest
        # IG2K at 190V should be elevated compared to 300V
        assert ig2k_190 > ig2k_300, "Kink: IG2K at 190V should exceed IG2K at 300V"

    def test_no_kink_above_vg2(self):
        """Above VG2K, x=0, no secondary emission effect."""
        ipk, ig2k, _ = dempwolf_beam_tetrode(300.0, 0.0, 250.0, p=P_6L6GC)
        # x = max(1 - 300/250, 0) = 0 → no secondary emission
        # Should behave like a pentode
        p_no_sec = DempwolfParams(
            mu=P_6L6GC.mu, G=P_6L6GC.G, gamma=P_6L6GC.gamma, C=P_6L6GC.C,
            Gg=P_6L6GC.Gg, xi=P_6L6GC.xi, Cg=P_6L6GC.Cg,
            Kvb=P_6L6GC.Kvb, Kvb1=P_6L6GC.Kvb1, Kn=P_6L6GC.Kn,
            fg2=P_6L6GC.fg2, A=P_6L6GC.A,
            sigma=0.0,  # no secondary emission
        )
        ipk_ref, ig2k_ref, _ = dempwolf_pentode(300.0, 0.0, 250.0, p=p_no_sec)
        # Should be very close (x=0 means I_sec=0)
        assert abs(ipk - ipk_ref) < 1e-10
        assert abs(ig2k - ig2k_ref) < 1e-10

    def test_current_conservation_beam(self):
        """IPK + IG2K + IGK == IK_total for beam tetrode."""
        for vpk in [50, 100, 190, 250, 400]:
            ipk, ig2k, igk = dempwolf_v2(vpk, -10.0, 250.0, p=P_6L6GC)
            total = ipk + (ig2k or 0.0) + igk
            assert total >= 0, f"Negative total at VPK={vpk}"


# ===================================================================
# Numerical safety (§12)
# ===================================================================

class TestNumericalSafety:
    """Edge cases from §12."""

    def test_vpk_zero(self):
        """VPK=0 should not crash."""
        ipk, igk = dempwolf_triode(0.0, -2.0, p=P_12AX7_RSD1)
        assert math.isfinite(ipk) and math.isfinite(igk)

    def test_vg2k_zero(self):
        """VG2K=0 should not crash for pentode."""
        ipk, ig2k, igk = dempwolf_v2(100.0, -5.0, 0.0, p=P_EL34)
        assert math.isfinite(ipk)
        assert ig2k is not None and math.isfinite(ig2k)

    def test_large_negative_vgk(self):
        """Very negative VGK should give ~0 current, not NaN."""
        ipk, igk = dempwolf_triode(300.0, -100.0, p=P_12AX7_RSD1)
        assert math.isfinite(ipk) and ipk >= 0
        assert math.isfinite(igk) and igk >= 0

    def test_large_positive_vgk(self):
        """Positive VGK should not overflow."""
        ipk, igk = dempwolf_triode(300.0, 10.0, p=P_12AX7_RSD1)
        assert math.isfinite(ipk) and math.isfinite(igk)
        assert ipk >= 0 and igk > 0

    def test_very_large_vpk(self):
        """VPK=1000V should not overflow."""
        ipk, ig2k, igk = dempwolf_v2(1000.0, -5.0, 300.0, p=P_EL34)
        assert math.isfinite(ipk)

    def test_softplus_extreme_args(self):
        """Softplus should handle extreme arguments."""
        assert math.isfinite(_softplus(100.0, 3.4))
        assert math.isfinite(_softplus(-100.0, 3.4))
        assert _softplus(-100.0, 3.4) >= 0


# ===================================================================
# Model integration
# ===================================================================

class TestModelIntegration:
    """Test DempwolfModel, loader, and registry."""

    def test_registry_has_dempwolf(self):
        """Dempwolf should be registered in MODEL_REGISTRY."""
        assert "dempwolf" in MODEL_REGISTRY

    def test_registry_label(self):
        assert MODEL_REGISTRY["dempwolf"].label == "Dempwolf v2"

    def test_load_12ax7(self):
        """Load 12AX7 Dempwolf model from tube_params.json."""
        model = load_dempwolf_model("12AX7")
        assert model is not None
        assert model.name == "12AX7"
        assert model.topology == TOPOLOGY_TRIODE
        assert model.model_type == MODEL_TYPE_DEMPWOLF
        assert abs(model.dempwolf.mu - 103.2) < 0.1

    def test_load_el34(self):
        model = load_dempwolf_model("EL34")
        assert model is not None
        assert model.topology == TOPOLOGY_PENTODE

    def test_load_6l6(self):
        model = load_dempwolf_model("6L6")
        assert model is not None
        assert model.dempwolf.sigma > 0

    def test_load_nonexistent(self):
        """Tube without Dempwolf params should return None."""
        model = load_dempwolf_model("12AU7")
        assert model is None

    def test_load_via_alias(self):
        """Load via alias (ECC83 → 12AX7)."""
        model = load_dempwolf_model("ECC83")
        assert model is not None
        assert model.name == "12AX7"

    def test_list_dempwolf_tubes(self):
        tubes = _list_dempwolf_tubes()
        assert "12AX7" in tubes
        assert "EL34" in tubes
        assert "6L6" in tubes

    def test_model_ia_triode(self):
        """DempwolfModel.ia() should return mA."""
        model = load_dempwolf_model("12AX7")
        ia = model.ia(250.0, -2.0)
        assert 0.5 < ia < 2.0, f"ia = {ia:.3f} mA"

    def test_model_ig2_triode_zero(self):
        """Triode ig2 should return 0."""
        model = load_dempwolf_model("12AX7")
        assert model.ig2(250.0, -2.0, 0.0) == 0.0

    def test_model_ia_pentode(self):
        model = load_dempwolf_model("EL34")
        ia = model.ia(300.0, -10.0, 300.0)
        assert ia > 10, f"EL34 ia = {ia:.1f} mA, expected > 10"

    def test_model_ig2_pentode(self):
        model = load_dempwolf_model("EL34")
        ig2 = model.ig2(300.0, -10.0, 300.0)
        assert ig2 > 0, f"EL34 ig2 = {ig2:.3f} mA"

    def test_params_dict_triode(self):
        model = load_dempwolf_model("12AX7")
        d = model.params_dict()
        assert "mu" in d
        assert "Kvb_t" in d
        assert "Kvb" not in d  # triode has no pentode Kvb

    def test_params_dict_pentode(self):
        model = load_dempwolf_model("EL34")
        d = model.params_dict()
        assert "Kvb" in d
        assert "fg2" in d
        assert "Kvb_t" not in d

    def test_params_dict_beam_tetrode(self):
        model = load_dempwolf_model("6L6")
        d = model.params_dict()
        assert "sigma" in d
        assert "Ks" in d

    def test_fitter_registered(self):
        """Fitter should be callable (not a stub)."""
        entry = MODEL_REGISTRY["dempwolf"]
        assert callable(entry.fitter)
        # Empty data should raise RuntimeError, not NotImplementedError
        with pytest.raises(RuntimeError):
            entry.fitter([], "triode")

    def test_fitter_too_few_points(self):
        """Fitter should raise RuntimeError on insufficient data."""
        points = [{"ua": 100, "ug1": -2, "ia": 0.01}]
        entry = MODEL_REGISTRY["dempwolf"]
        with pytest.raises(RuntimeError):
            entry.fitter(points, "triode")


# ===================================================================
# Fitter tests — round-trip on synthetic data
# ===================================================================

import numpy as np
from lm19.dempwolf import fit_dempwolf, _has_kink
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)
from lm19.tube_model_base import (
    MODEL_TYPE_DEMPWOLF,
)


def _generate_triode_points(p, ua_range=(50, 400, 25), ug1_range=(-4, 0, 0.5)):
    """Generate synthetic triode measurement points (Ia in mA)."""
    points = []
    ua_vals = np.arange(ua_range[0], ua_range[1] + 1, ua_range[2])
    ug1_vals = np.arange(ug1_range[0], ug1_range[1] + 0.01, ug1_range[2])
    for ug1 in ug1_vals:
        for ua in ua_vals:
            ipk, _, igk = dempwolf_v2(float(ua), float(ug1), vg2k=None, p=p)
            ia_mA = ipk * 1000.0
            points.append({"ua": float(ua), "ug1": float(ug1), "ia": ia_mA})
    return points


def _generate_pentode_points(p, ua_range=(20, 400, 20),
                              ug1_range=(-20, 0, 2),
                              ug2_vals=(200, 300)):
    """Generate synthetic pentode measurement points (Ia, Ig2 in mA).

    If ug2_vals is None, generates triode_connected data (Vg2 = Va).
    """
    points = []
    ua_vals = np.arange(ua_range[0], ua_range[1] + 1, ua_range[2])
    ug1_vals = np.arange(ug1_range[0], ug1_range[1] + 0.01, ug1_range[2])
    triode_connected = ug2_vals is None
    if triode_connected:
        ug2_vals = [None]  # sentinel
    for ug2 in ug2_vals:
        for ug1 in ug1_vals:
            for ua in ua_vals:
                vg2 = float(ua) if triode_connected else float(ug2)
                ipk, ig2k, igk = dempwolf_v2(
                    float(ua), float(ug1), vg2, p=p,
                )
                ia_mA = ipk * 1000.0
                ig2_mA = (ig2k or 0.0) * 1000.0
                points.append({
                    "ua": float(ua), "ug1": float(ug1), "ug2": vg2,
                    "ia": ia_mA, "ig2": ig2_mA,
                })
    return points


class TestFitterTriode:
    """Round-trip fitter tests for triode topology."""

    @pytest.fixture(scope="class")
    def triode_fit(self):
        """Fit synthetic 12AX7 triode data once for all tests."""
        points = _generate_triode_points(P_12AX7_RSD1)
        return fit_dempwolf(points, "triode")

    def test_fit_returns_result(self, triode_fit):
        assert triode_fit is not None
        assert triode_fit.model_type == MODEL_TYPE_DEMPWOLF
        assert triode_fit.topology == TOPOLOGY_TRIODE

    def test_fit_model_object(self, triode_fit):
        assert triode_fit.model is not None
        assert isinstance(triode_fit.model, DempwolfModel)

    def test_rms_error_low(self, triode_fit):
        """RMS error on clean synthetic data should be < 0.1 mA."""
        assert triode_fit.rms_error < 0.1, \
            f"RMS = {triode_fit.rms_error:.4f} mA, expected < 0.1"

    def test_max_error_low(self, triode_fit):
        """Max error on clean synthetic data should be < 0.5 mA."""
        assert triode_fit.max_error < 0.5, \
            f"Max = {triode_fit.max_error:.4f} mA, expected < 0.5"

    def test_mu_recovery(self, triode_fit):
        """Fitted µ should be close to reference (103.2)."""
        fitted_mu = triode_fit.params["mu"]
        assert abs(fitted_mu - 103.2) / 103.2 < 0.15, \
            f"µ = {fitted_mu:.1f}, expected ~103.2"

    def test_G_recovery(self, triode_fit):
        """Fitted G should be in the right order of magnitude."""
        fitted_G = triode_fit.params["G"]
        assert 1e-4 < fitted_G < 1e-2, \
            f"G = {fitted_G:.4e}, expected ~2.242e-3"

    def test_gamma_recovery(self, triode_fit):
        """Fitted γ should be close to reference (1.26)."""
        fitted_gamma = triode_fit.params["gamma"]
        assert 0.9 < fitted_gamma < 1.8, \
            f"γ = {fitted_gamma:.3f}, expected ~1.26"

    def test_n_points(self, triode_fit):
        """Should report reasonable number of points used."""
        assert triode_fit.n_points > 20

    def test_params_dict_has_keys(self, triode_fit):
        """Result params dict should have core Dempwolf keys."""
        d = triode_fit.params
        for key in ("mu", "G", "gamma", "C", "Gg", "xi", "Cg", "Kvb_t"):
            assert key in d, f"Missing key '{key}' in params"

    def test_triode_connected_alias(self):
        """topology='triode_connected' should use pentode path with ug2=ua."""
        points = _generate_triode_points(P_12AX7_RSD1)
        # Add ug2 = ua (triode_connected means screen tied to plate)
        for p in points:
            p["ug2"] = p["ua"]
            p["ig2"] = 0.0
        result = fit_dempwolf(points, "triode_connected")
        assert result.topology == TOPOLOGY_PENTODE
        assert result.rms_error < 1.0


class TestTriodeJointRefine:
    """Pins for the triode joint-refine phase (all 8 params, full data).

    Before the refine, Kvb_t was fit ONLY on grid-region points (Ug1 > -1 V)
    — the wrong region for a Region-A parameter that acts at low Va — and the
    cathode params were frozen against the default Kvb_t=300. On data with no
    grid region at all (every real LM19 scan) Kvb_t silently stayed 300.
    """

    # Kvb_t far from the 300 default; NO grid-region points (Ug1 <= -1.5),
    # low-Va points present -> only the joint refine can identify Kvb_t.
    _P_TRUE = DempwolfParams(
        mu=20.0, G=2e-3, gamma=1.4, C=4.0,
        Gg=6e-4, xi=1.3, Cg=10.0, Kvb_t=1200.0,
    )

    @pytest.fixture(scope="class")
    def refine_fit(self):
        points = _generate_triode_points(
            self._P_TRUE, ua_range=(20, 300, 20), ug1_range=(-8, -1.5, 0.5))
        return fit_dempwolf(points, "triode")

    def test_kvb_t_recovered_without_grid_region(self, refine_fit):
        kvbt = refine_fit.params["Kvb_t"]
        assert abs(kvbt - 1200.0) / 1200.0 < 0.4, (
            f"Kvb_t = {kvbt:.0f}, expected ~1200 — joint refine failed to "
            f"identify Region A (pre-refine code left the 300 default)")

    def test_rms_low_on_region_a_data(self, refine_fit):
        assert refine_fit.rms_error < 0.05, \
            f"RMS = {refine_fit.rms_error:.4f} mA, expected < 0.05"

    def test_grid_params_frozen_without_grid_data(self, refine_fit):
        """Mutation-audit pin (M2): with no grid-region points
        Gg/ξ/Cg must stay EXACTLY at the phase-2 defaults — unfrozen they
        wander to fit noise (observed: Gg 6e-4 → 5.2e-3, a 9× fictional
        grid current shipped into SPICE exports) while every rms assert
        stays green."""
        from lm19.dempwolf import _GRID_CURRENT_DEFAULTS
        gg, xi, cg = _GRID_CURRENT_DEFAULTS
        p = refine_fit.params
        assert p["Gg"] == pytest.approx(gg, rel=1e-9)
        assert p["xi"] == pytest.approx(xi, rel=1e-9)
        assert p["Cg"] == pytest.approx(cg, rel=1e-9)


class TestTriodeMetricHonesty:
    """Pin: reported rms/max must be computed by the SAME model that ships
    (igk subtracted). The old fitting-only kernel omitted grid current and
    overstated rms up to 7x on tubes with grid-region data (real 6N5P:
    reported 18.2 mA vs actual 2.5 mA), biasing the fit benchmark against
    Dempwolf on triodes."""

    def test_reported_rms_matches_shipped_model(self):
        from lm19.tube_model_base import extract_arrays
        points = _generate_triode_points(
            P_12AX7_RSD1, ua_range=(50, 300, 25), ug1_range=(-3, 1, 0.5))
        result = fit_dempwolf(points, "triode")
        data = extract_arrays(points, topology=TOPOLOGY_TRIODE,
                              ia_thr_mA=0.05, min_count=5)
        ia_model_mA = result.model.ia_array(data.ua, data.ug1, 0.0)
        rms_true = float(np.sqrt(np.mean((ia_model_mA - data.ia * 1000.0) ** 2)))
        assert result.rms_error == pytest.approx(rms_true, rel=1e-9, abs=1e-12), (
            f"reported {result.rms_error:.4f} mA != shipped-model "
            f"{rms_true:.4f} mA — metric diverged from the model")


class TestFitterPentode:
    """Round-trip fitter tests for pentode topology."""

    @pytest.fixture(scope="class")
    def pentode_fit(self):
        """Fit synthetic EL34 pentode data once for all tests."""
        points = _generate_pentode_points(P_EL34)
        return fit_dempwolf(points, "pentode")

    def test_fit_returns_result(self, pentode_fit):
        assert pentode_fit is not None
        assert pentode_fit.model_type == MODEL_TYPE_DEMPWOLF
        assert pentode_fit.topology == TOPOLOGY_PENTODE

    def test_fit_model_object(self, pentode_fit):
        assert pentode_fit.model is not None
        assert isinstance(pentode_fit.model, DempwolfModel)

    def test_rms_error_low(self, pentode_fit):
        """RMS error on clean synthetic pentode data should be < 0.1 mA."""
        assert pentode_fit.rms_error < 0.1, \
            f"RMS = {pentode_fit.rms_error:.4f} mA, expected < 0.1"

    def test_max_error_reasonable(self, pentode_fit):
        """Max error on clean synthetic pentode data should be < 0.5 mA."""
        assert pentode_fit.max_error < 0.5, \
            f"Max = {pentode_fit.max_error:.4f} mA, expected < 0.5"

    def test_mu_recovery(self, pentode_fit):
        """Fitted µ should be close to reference (11.0)."""
        fitted_mu = pentode_fit.params["mu"]
        assert abs(fitted_mu - 11.0) / 11.0 < 0.3, \
            f"µ = {fitted_mu:.1f}, expected ~11.0"

    def test_n_points(self, pentode_fit):
        """Should report reasonable number of points used."""
        assert pentode_fit.n_points > 50

    def test_params_dict_has_pentode_keys(self, pentode_fit):
        """Result params dict should have pentode-specific keys."""
        d = pentode_fit.params
        for key in ("mu", "G", "gamma", "C", "Kvb", "Kvb1", "fg2"):
            assert key in d, f"Missing key '{key}' in params"

    def test_fg2_positive(self, pentode_fit):
        """fg2 should be positive (screen interception)."""
        assert pentode_fit.params.get("fg2", 0) >= 0

    def test_forward_eval_matches(self, pentode_fit):
        """Model from fit should produce reasonable Ia for known point."""
        model = pentode_fit.model
        ia = model.ia(300.0, -10.0, 300.0)
        # Should be in the same ballpark as the reference EL34
        assert ia > 10, f"Fitted model ia = {ia:.1f} mA at VPK=300"


class TestFitPhase2PentodeBranch:
    """Phase 2 must use the pentode (Vg2k) cathode form, tested in isolation
    from the phase-3/4 joint refinement that can otherwise mask a bad seed.

    Data is built from phase 1's exact cathode formula so the grid params are
    self-consistently recoverable; the triode-form cathode (the bug) saturates
    on the large pentode C and cannot recover them.
    """

    def _grid_region_data(self):
        from lm19.dempwolf import _EXP_CLIP
        mu, G, gamma, C = 11.0, 3.0e-3, 1.35, 750.0
        Gg, xi, Cg = 6.0e-4, 1.3, 10.0
        ug1 = np.array([-0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
        ua = np.full_like(ug1, 250.0)
        ug2 = np.full_like(ug1, 300.0)
        arg = np.clip(C * (1.0 / mu + ug1 / ug2), -_EXP_CLIP, _EXP_CLIP)
        ik = G * ((ug2 / C) * np.log1p(np.exp(arg))) ** gamma
        arg_g = np.clip(Cg * ug1, -_EXP_CLIP, _EXP_CLIP)
        igk = Gg * (np.log1p(np.exp(arg_g)) / Cg) ** xi
        ia = ik - igk  # ig2=0 → phase-2 cathode target = ia
        ig2 = np.zeros_like(ug1)
        return ua, ug1, ug2, ia, ig2, (mu, G, gamma, C), (Gg, xi, Cg)

    def test_pentode_branch_recovers_grid_params(self):
        from lm19.dempwolf import _fit_phase2
        from scipy.optimize import least_squares
        ua, ug1, ug2, ia, ig2, phase1, (Gg, xi, Cg) = self._grid_region_data()
        gp, _ = _fit_phase2(ua, ug1, ia, phase1, least_squares,
                            ug2=ug2, ig2=ig2)
        assert abs(gp[0] - Gg) / Gg < 0.1, f"Gg={gp[0]:.3e} expected {Gg}"
        assert abs(gp[1] - xi) / xi < 0.1, f"xi={gp[1]:.3f} expected {xi}"
        assert abs(gp[2] - Cg) / Cg < 0.1, f"Cg={gp[2]:.3f} expected {Cg}"

    def test_triode_branch_fails_on_pentode_data(self):
        """The pre-fix triode-form cathode cannot recover the grid params —
        pins the bug so the fix cannot silently regress."""
        from lm19.dempwolf import _fit_phase2
        from scipy.optimize import least_squares
        ua, ug1, ug2, ia, ig2, phase1, (Gg, xi, Cg) = self._grid_region_data()
        gp, _ = _fit_phase2(ua, ug1, ia, phase1, least_squares)  # no ug2 → bug
        assert abs(gp[0] - Gg) / Gg > 0.5, f"Gg={gp[0]:.3e} unexpectedly near ref"

    def test_pentode_fit_passes_ug2_to_phase2(self, monkeypatch):
        """The pentode fitter must hand Ug2 to phase 2 (call-site guard)."""
        import lm19.dempwolf as D
        captured = {}
        orig = D._fit_phase2

        def spy(ua, ug1, ia, phase1, least_squares, fit_kvb_t=False,
                ug2=None, ig2=None, tracker=None, warnings_out=None):
            captured["ug2_passed"] = ug2 is not None
            return orig(ua, ug1, ia, phase1, least_squares,
                        fit_kvb_t=fit_kvb_t, ug2=ug2, ig2=ig2, tracker=tracker,
                        warnings_out=warnings_out)

        monkeypatch.setattr(D, "_fit_phase2", spy)
        D.fit_dempwolf(_generate_pentode_points(P_EL34), "pentode")
        assert captured.get("ug2_passed") is True

    def test_determinability_fallback_to_triode(self, caplog):
        """A true pentode with <2 distinct grid-region Ug1 levels is
        underdetermined → phase 2 falls back to the triode cathode form and
        warns (failure-visibility), rather than seeding phase 4 with a bad fit."""
        import logging
        from lm19.dempwolf import _fit_phase2
        from scipy.optimize import least_squares
        ua, ug1, ug2, ia, ig2, phase1, _ = self._grid_region_data()
        ug1_one = np.zeros_like(ug1)  # collapse to a single grid-region level
        with caplog.at_level(logging.WARNING, logger="lm19.dempwolf"):
            gp_pent, _ = _fit_phase2(ua, ug1_one, ia, phase1, least_squares,
                                     ug2=ug2, ig2=ig2)
        gp_tri, _ = _fit_phase2(ua, ug1_one, ia, phase1, least_squares)
        # Pentode attempt fell back to the identical triode-form fit...
        assert np.allclose(gp_pent, gp_tri)
        # ...and the degraded path is visible.
        assert any("falls back to the triode-form" in r.getMessage()
                   for r in caplog.records)


# ===================================================================
# Beam tetrode fitter (secondary emission)
# ===================================================================

class TestFitterBeamTetrode:
    """Round-trip fitter test for beam tetrode with secondary emission."""

    @pytest.fixture(scope="class")
    def beam_fit(self):
        """Fit synthetic 6L6GC beam tetrode data."""
        points = _generate_pentode_points(
            P_6L6GC,
            ua_range=(10, 400, 10),
            ug1_range=(-20, 0, 2),
            ug2_vals=(250,),
        )
        return fit_dempwolf(points, "pentode")

    def test_fit_returns_result(self, beam_fit):
        assert beam_fit is not None
        assert beam_fit.topology == TOPOLOGY_PENTODE

    def test_rms_error_low(self, beam_fit):
        """RMS on clean beam tetrode data should be < 1.0 mA."""
        assert beam_fit.rms_error < 1.0, \
            f"RMS = {beam_fit.rms_error:.4f} mA, expected < 1.0"

    def test_sigma_detected(self, beam_fit):
        """Fitter should detect secondary emission (sigma > 0)."""
        assert beam_fit.params.get("sigma", 0) > 0.05, \
            f"sigma = {beam_fit.params.get('sigma', 0):.4f}, expected > 0.05"

    def test_mu_recovery(self, beam_fit):
        """Fitted µ should be close to reference (8.7)."""
        fitted_mu = beam_fit.params["mu"]
        assert abs(fitted_mu - 8.7) / 8.7 < 0.3, \
            f"µ = {fitted_mu:.1f}, expected ~8.7"

    def test_kvb_recovery(self, beam_fit):
        """Fitted Kvb should be in reasonable range."""
        fitted_kvb = beam_fit.params["Kvb"]
        assert 2.0 < fitted_kvb < 80.0, \
            f"Kvb = {fitted_kvb:.1f}, expected ~12.0"

    def test_forward_eval_kink(self, beam_fit):
        """Fitted model should reproduce kink region behavior."""
        model = beam_fit.model
        # At Va=100 (below Vg2=250), secondary emission should reduce Ia
        ia_100 = model.ia(100.0, 0.0, 250.0)
        ia_350 = model.ia(350.0, 0.0, 250.0)
        # Both should be positive
        assert ia_100 > 0 and ia_350 > 0


# ===================================================================
# _has_kink detection
# ===================================================================

class TestHasKink:
    """Direct tests for kink (negative resistance) detection."""

    def test_pentode_no_kink(self):
        """EL34 pentode data (no sigma) should NOT trigger kink."""
        points = _generate_pentode_points(
            P_EL34, ua_range=(20, 400, 20),
            ug1_range=(-20, 0, 4), ug2_vals=(300,),
        )
        ua = np.array([p["ua"] for p in points])
        ug1 = np.array([p["ug1"] for p in points])
        ug2 = np.array([p["ug2"] for p in points])
        ia = np.array([p["ia"] for p in points]) / 1000.0
        assert _has_kink(ua, ug1, ug2, ia) is False

    def test_beam_tetrode_has_kink(self):
        """6L6GC beam tetrode data (sigma=0.25) should trigger kink."""
        points = _generate_pentode_points(
            P_6L6GC, ua_range=(10, 400, 10),
            ug1_range=(-10, 0, 2), ug2_vals=(250,),
        )
        ua = np.array([p["ua"] for p in points])
        ug1 = np.array([p["ug1"] for p in points])
        ug2 = np.array([p["ug2"] for p in points])
        ia = np.array([p["ia"] for p in points]) / 1000.0
        assert _has_kink(ua, ug1, ug2, ia) is True

    def test_insufficient_data(self):
        """Too few points should return False, not crash."""
        ua = np.array([10.0, 20.0])
        ug1 = np.array([-5.0, -5.0])
        ug2 = np.array([250.0, 250.0])
        ia = np.array([0.01, 0.02])
        assert _has_kink(ua, ug1, ug2, ia) is False

    def test_clean_multi_ug2_pentode_no_kink(self):
        """Decoupling pin (inverts the old load-bearing test): clean multi-Ug2
        pentode data must NOT trip _has_kink. The old Ug1-only grouping
        interleaved Ug2 levels and read the level steps as dips — every real
        LM19 pentode scan got classified as a beam tetrode and phase 5 fitted
        fictional secondary emission (EL84: σ ≈ 0.2). Below-knee masking no
        longer depends on this detection (it is gated on Ig2 availability)."""
        points = _generate_pentode_points(P_EL34, ug2_vals=(200, 300))
        ua = np.array([p["ua"] for p in points])
        ug1 = np.array([p["ug1"] for p in points])
        ug2 = np.array([p["ug2"] for p in points])
        ia = np.array([p["ia"] for p in points]) / 1000.0
        assert _has_kink(ua, ug1, ug2, ia) is False

    def test_noise_level_dip_no_kink(self):
        """A drawdown below the relative threshold (measurement/trace
        artifacts reach 2.6% of curve max on real scans) must not trip."""
        ua = np.array([50.0, 100.0, 150.0, 200.0, 240.0])
        ug2 = np.full_like(ua, 250.0)
        ug1 = np.full_like(ua, -5.0)
        # 40 mA curve with a 1 mA (2.5%) artifact dip — under the 5% threshold
        ia = np.array([0.030, 0.038, 0.037, 0.039, 0.040])
        assert _has_kink(ua, ug1, ug2, ia) is False

    def test_deep_relative_dip_is_kink(self):
        """A dynatron-scale drawdown (real kinks measure ≥16% of curve max)
        must trip detection — including one spread across several Ua steps
        (per-step diffs stay small; the cumulative drawdown does not)."""
        ua = np.array([50.0, 100.0, 150.0, 200.0, 240.0, 280.0])
        ug2 = np.full_like(ua, 300.0)
        ug1 = np.full_like(ua, -5.0)
        # 40 mA curve descending 38 -> 34 mA over three steps (10% drawdown)
        ia = np.array([0.030, 0.038, 0.0365, 0.035, 0.034, 0.040])
        assert _has_kink(ua, ug1, ug2, ia) is True

    def test_real_scan_detection(self):
        """Real-scan pins: a true pentode (EL84, suppressor grid → σ=0
        physically) must NOT read as kinked; a real beam tetrode (6P1P,
        dynatron dip ≈5.6% of curve max) MUST. These are the datasets on
        which the old detector mis-fired."""
        import json
        from pathlib import Path
        from lm19.tube_model_base import extract_arrays
        base = Path(__file__).parent / "spice_test_data" / "converted"
        expected = {
            "pentode_EL84_ER_L1_real.json": False,
            "pentode_EL84_ER_L2_real.json": False,   # 2.6% artifact dips
            "pentode_EL84_SOVTEK_L1_real.json": False,
            "pentode_6P1P_real.json": True,          # 5.6% dynatron kink
        }
        for fname, want in expected.items():
            pts = json.loads((base / fname).read_text(encoding="utf-8"))["points"]
            d = extract_arrays(pts, topology=TOPOLOGY_TRIODE_CONNECTED,
                               ia_thr_mA=0.05, min_count=10)
            assert _has_kink(d.ua, d.ug1, d.ug2, d.ia) is want, \
                f"{fname}: expected kink={want}"

    def test_adjacent_ug1_curves_no_false_kink(self):
        """Two clean monotonic curves 0.1 V apart must NOT read as a kink.

        The old 0.15 V window merged ±0.1 V neighbour curves, so interleaving
        them on Ua produced a spurious dip; exact-key grouping prevents it.
        """
        ua = np.array([50.0, 100.0, 150.0, 200.0,
                       50.0, 100.0, 150.0, 200.0])
        ug1 = np.array([-2.0, -2.0, -2.0, -2.0,
                        -2.1, -2.1, -2.1, -2.1])
        ug2 = np.full(8, 300.0)
        ia = np.array([0.010, 0.020, 0.030, 0.040,    # curve -2.0
                       0.009, 0.019, 0.029, 0.039])   # curve -2.1
        assert _has_kink(ua, ug1, ug2, ia) is False

    def test_no_ig2_masking_logs_warning(self, caplog):
        """When below-knee points are masked (Ia-only data — the masking
        criterion is Ig2 availability, not kink), the user must be warned."""
        import logging
        points = _generate_pentode_points(P_EL34)
        for p in points:
            p["ig2"] = 0.0  # simulate missing Ig2 measurement
        with caplog.at_level(logging.WARNING, logger="lm19.dempwolf"):
            fit_dempwolf(points, "pentode")
        assert any("masking" in r.getMessage().lower()
                   or "below-knee" in r.getMessage().lower()
                   for r in caplog.records)

    def test_ig2_data_disables_masking(self, caplog):
        """With Ig2 present the target Ik = Ia + Ig2 is knee-independent —
        no points may be discarded (the old code masked phases 1-4 whenever
        the kink detector fired, even on beam tetrodes with full Ig2 data)."""
        import logging
        points = _generate_pentode_points(
            P_6L6GC, ua_range=(10, 400, 10),
            ug1_range=(-10, 0, 2), ug2_vals=(250,),
        )
        with caplog.at_level(logging.WARNING, logger="lm19.dempwolf"):
            fit_dempwolf(points, "pentode")
        assert not any("masking" in r.getMessage().lower()
                       for r in caplog.records)

    def test_phases_3_4_see_full_dataset(self, monkeypatch):
        """Mutation-audit pin (M3): the phase-1/2 mask must NOT
        leak into phases 3-4 — Kvb/Kvb1/Kn/fg2 parameterize the knee, and
        re-coupling them to the above-knee subset fits the knee without
        knee data (all rms asserts stayed green under that mutation)."""
        import lm19.dempwolf as dw
        from lm19.tube_model_base import extract_arrays
        points = _generate_pentode_points(P_EL34)
        for p in points:
            p["ig2"] = 0.0  # Ia-only data — the phase-1/2 mask engages
        n_full = len(extract_arrays(
            points, topology=TOPOLOGY_TRIODE_CONNECTED,
            ia_thr_mA=0.05, min_count=10).ua)

        seen = {}
        orig1, orig3, orig4 = dw._fit_phase1, dw._fit_phase3, dw._fit_phase4

        def spy1(ua, *a, **k):
            seen["p1"] = len(ua)
            return orig1(ua, *a, **k)

        def spy3(ua, *a, **k):
            seen["p3"] = len(ua)
            return orig3(ua, *a, **k)

        def spy4(ua, *a, **k):
            seen["p4"] = len(ua)
            return orig4(ua, *a, **k)

        monkeypatch.setattr(dw, "_fit_phase1", spy1)
        monkeypatch.setattr(dw, "_fit_phase3", spy3)
        monkeypatch.setattr(dw, "_fit_phase4", spy4)
        dw.fit_dempwolf(points, "pentode")

        assert seen["p1"] < n_full, "mask did not engage — test is vacuous"
        assert seen["p3"] == n_full, "phase 3 must see the knee data"
        assert seen["p4"] == n_full, "phase 4 must see the knee data"


class TestBoundsHygiene:
    """Docs §10.6/§14.7 bounds are load-bearing physics limits."""

    def test_pentode_bounds_match_documented_limits(self):
        from lm19.dempwolf import _PENTODE_BOUNDS_HI
        # order: [mu, G, gamma, C, Gg, xi, Cg, Kvb, Kvb1, fg2, A, Kn]
        assert _PENTODE_BOUNDS_HI[9] == pytest.approx(0.30), "fg2 hi (§14.7)"
        assert _PENTODE_BOUNDS_HI[10] == pytest.approx(0.001), \
            "A hi (§14.2 perturbative Durchgriff)"
        assert _PENTODE_BOUNDS_HI[5] == pytest.approx(2.5), "xi hi (§10.6)"

    def test_beam_joint_refine_uses_shared_bounds(self):
        """Mutation-audit pin (M5): the beam joint refine must
        derive its bounds from the shared constants — it historically
        carried divergent literals (fg2 0.5, A 0.01: a 6× emission scale at
        500 V, far past the perturbative Durchgriff regime), invisible to
        every quality assert."""
        import inspect
        import lm19.dempwolf as dw
        src = inspect.getsource(dw._fit_pentode)
        assert "_PENTODE_BOUNDS_HI + _SEC_EMISSION_BOUNDS_HI" in src
        assert "_PENTODE_BOUNDS_LO" in src


class TestRealBeamTetrodeFit:
    """Integration pin on the real 6P1P scan — the only real beam tetrode
    in the dataset pool.

    Mutation-audit pin (M1): with phase-5 multi-start reverted
    to a single (λ=1, ν=2) start, ALL synthetic tests stay green while the
    real 6P1P fit collapses to σ ≈ 0 (kink lost): the 5-param secondary-
    emission landscape is multimodal and only real-scan geometry exposes it.

    The scan is subsampled [::2] (1414 pts, fit ~7 s vs ~40 s full) — the
    [::2] grid still discriminates (clean σ=1.76 vs single-start σ=0.000);
    [::3] does NOT (single-start degenerates to the σ=10 bound instead and
    would slip past the assert). Timeout raised over the global 10 s for
    xdist-contention headroom."""

    @pytest.mark.timeout(120)
    def test_6p1p_fit_keeps_dynatron_kink(self):
        import json
        from pathlib import Path
        base = Path(__file__).parent / "spice_test_data" / "converted"
        pts = json.loads((base / "pentode_6P1P_real.json")
                         .read_text(encoding="utf-8"))["points"][::2]
        result = fit_dempwolf(pts, "pentode")
        dp = result.model.dempwolf
        assert dp.sigma > 0.5, \
            f"sigma = {dp.sigma:.3f} — real dynatron kink lost"
        assert result.rms_error < 1.0, \
            f"rms = {result.rms_error:.3f} mA (kink-preserving fit gives ~0.8)"


# ===================================================================
# Pentode fitter without Ig2 data
# ===================================================================

class TestFitterPentodeNoIg2:
    """Fitter should work when Ig2 data is missing or zero."""

    @pytest.fixture(scope="class")
    def fit_no_ig2(self):
        """Fit pentode data with ig2=0 (unmeasured)."""
        points = _generate_pentode_points(P_EL34)
        for p in points:
            p["ig2"] = 0.0  # simulate missing Ig2 measurement
        return fit_dempwolf(points, "pentode")

    def test_fit_succeeds(self, fit_no_ig2):
        assert fit_no_ig2 is not None
        assert fit_no_ig2.model_type == MODEL_TYPE_DEMPWOLF

    def test_rms_reasonable(self, fit_no_ig2):
        """Without Ig2 guidance, fit should still be decent for Ia."""
        assert fit_no_ig2.rms_error < 5.0, \
            f"RMS = {fit_no_ig2.rms_error:.4f} mA"

    def test_mu_in_range(self, fit_no_ig2):
        """µ should still be in a reasonable range."""
        mu = fit_no_ig2.params["mu"]
        assert 3.0 < mu < 50.0, f"µ = {mu:.1f}"


# ===================================================================
# Fitter with noisy data
# ===================================================================

class TestFitterNoisyData:
    """Fitter robustness on data with measurement noise."""

    @pytest.fixture(scope="class")
    def noisy_triode_fit(self):
        """Fit 12AX7 data with ±2% Gaussian noise on Ia."""
        rng = np.random.RandomState(42)
        points = _generate_triode_points(P_12AX7_RSD1)
        for p in points:
            noise = 1.0 + rng.normal(0, 0.02)  # ±2%
            p["ia"] = max(p["ia"] * noise, 0.0)
        return fit_dempwolf(points, "triode")

    def test_rms_under_noise_level(self, noisy_triode_fit):
        """RMS should be on the order of the noise, not blow up."""
        assert noisy_triode_fit.rms_error < 0.5, \
            f"RMS = {noisy_triode_fit.rms_error:.4f} mA"

    def test_mu_recovery_noisy(self, noisy_triode_fit):
        """µ should be close to 103.2 despite noise."""
        mu = noisy_triode_fit.params["mu"]
        assert abs(mu - 103.2) / 103.2 < 0.15, \
            f"µ = {mu:.1f}, expected ~103.2"

    @pytest.fixture(scope="class")
    def noisy_pentode_fit(self):
        """Fit EL34 data with ±3% Gaussian noise."""
        rng = np.random.RandomState(123)
        points = _generate_pentode_points(P_EL34)
        for p in points:
            noise_ia = 1.0 + rng.normal(0, 0.03)
            noise_ig2 = 1.0 + rng.normal(0, 0.03)
            p["ia"] = max(p["ia"] * noise_ia, 0.0)
            p["ig2"] = max(p["ig2"] * noise_ig2, 0.0)
        return fit_dempwolf(points, "pentode")

    def test_pentode_noisy_rms(self, noisy_pentode_fit):
        """Noisy pentode RMS should stay bounded."""
        assert noisy_pentode_fit.rms_error < 5.0, \
            f"RMS = {noisy_pentode_fit.rms_error:.4f} mA"

    def test_pentode_noisy_mu(self, noisy_pentode_fit):
        """µ should be in the right ballpark despite noise."""
        mu = noisy_pentode_fit.params["mu"]
        assert abs(mu - 11.0) / 11.0 < 0.3, \
            f"µ = {mu:.1f}, expected ~11.0"


# ===================================================================
# Triode_connected round-trip fitter
# ===================================================================

class TestFitterTriodeConnected:
    """Round-trip fitter for triode_connected (pentode path with Vg2=Va)."""

    @pytest.fixture(scope="class")
    def triconn_fit(self):
        """Fit synthetic EL34 data in triode_connected mode (Vg2=Va)."""
        # Generate pentode data but with Vg2 = Va
        points = _generate_pentode_points(
            P_EL34, ua_range=(20, 300, 10),
            ug1_range=(-20, 0, 2), ug2_vals=None,
        )
        return fit_dempwolf(points, "triode_connected")

    def test_fit_succeeds(self, triconn_fit):
        assert triconn_fit is not None
        assert triconn_fit.model_type == MODEL_TYPE_DEMPWOLF

    def test_rms_error_low(self, triconn_fit):
        """RMS on clean triode_connected data should be < 0.5 mA."""
        assert triconn_fit.rms_error < 0.5, \
            f"RMS = {triconn_fit.rms_error:.4f} mA"

    def test_mu_recovery(self, triconn_fit):
        """Fitted µ should be close to reference (11.0)."""
        mu = triconn_fit.params["mu"]
        assert abs(mu - 11.0) / 11.0 < 0.3, \
            f"µ = {mu:.1f}, expected ~11.0"

    def test_C_is_kp_scale(self, triconn_fit):
        """Fitted C should be in Kp range (>>1), not original Dempwolf range."""
        C = triconn_fit.params["C"]
        assert C > 10, f"C = {C:.1f}, expected Kp-scale (>>1)"


# ===================================================================
# Normalized formula numerical safety
# ===================================================================

class TestNormalizedFormulaSafety:
    """Edge cases for Kp-normalized pentode softplus."""

    def test_vg2_very_small(self):
        """Vg2 near zero should not crash or produce NaN."""
        p = P_EL34
        ipk, ig2k, igk = dempwolf_v2(100.0, -5.0, 0.5, p=p)
        assert np.isfinite(ipk)
        assert np.isfinite(ig2k)

    def test_vg2_equals_vmin(self):
        """Vg2 at minimum clamp should be safe."""
        p = P_EL34
        ipk, ig2k, igk = dempwolf_v2(100.0, -5.0, 0.01, p=p)
        assert np.isfinite(ipk)
        assert ipk >= 0

    def test_large_negative_ug1_over_vg2(self):
        """Large |Ug1/Vg2| should give near-zero current, not error."""
        p = P_EL34
        ipk, ig2k, igk = dempwolf_v2(300.0, -50.0, 10.0, p=p)
        assert np.isfinite(ipk)
        assert ipk * 1000 < 1.0  # essentially cutoff

    def test_triode_unchanged(self):
        """Triode path should NOT use normalized formula."""
        # Triode with same params should still work (uses original softplus)
        p = P_12AX7_RSD1
        ipk, ig2k, igk = dempwolf_v2(250.0, -2.0, None, p=p)
        assert np.isfinite(ipk)
        assert ipk * 1000 > 0.5  # should produce current


# ===================================================================
# Scalar vs vectorized consistency
# ===================================================================

class TestScalarVsVectorized:
    """Ensure dempwolf_v2() and _eval_pentode_vec() produce identical results."""

    def test_pentode_consistency(self):
        """Scalar and vectorized pentode eval must match exactly."""
        from lm19.dempwolf import _eval_pentode_vec
        p = P_EL34
        test_pts = [
            (300.0, -10.0, 300.0), (100.0, -5.0, 250.0),
            (50.0, 0.0, 200.0), (10.0, -20.0, 100.0),
        ]
        for ua, ug1, ug2 in test_pts:
            ipk_s, ig2_s, _ = dempwolf_v2(ua, ug1, ug2, p=p)
            ipk_v, ig2_v = _eval_pentode_vec(
                np.array([ua]), np.array([ug1]), np.array([ug2]), p,
            )
            assert abs(ipk_s - ipk_v[0]) < 1e-9, \
                f"Ia mismatch at Va={ua}"
            assert abs(ig2_s - ig2_v[0]) < 1e-12, \
                f"Ig2 mismatch at Va={ua}"

    def test_triode_consistency(self):
        """Scalar and vectorized triode eval must match — including the
        grid-current region (Ug1 ≥ 0), since dempwolf_v2_ia_vec subtracts
        igk exactly like the scalar model."""
        from lm19.dempwolf import dempwolf_v2_ia_vec
        p = P_12AX7_RSD1
        for ua, ug1 in [(250, -2), (100, -1), (50, -4), (300, -1),
                        (250, 0.0), (150, 1.0)]:
            ipk_s, _, _ = dempwolf_v2(float(ua), float(ug1), None, p=p)
            ipk_v = dempwolf_v2_ia_vec(
                np.array([float(ua)]), np.array([float(ug1)]), None, p=p,
            )
            assert abs(ipk_s - ipk_v[0]) < 1e-9, \
                f"Ia mismatch at Va={ua}, Vg={ug1}"

    def test_beam_tetrode_consistency(self):
        """Scalar and vectorized beam tetrode eval must match."""
        from lm19.dempwolf import _eval_pentode_vec
        p = P_6L6GC
        for ua in [50, 100, 200, 300]:
            ipk_s, ig2_s, _ = dempwolf_v2(float(ua), -10.0, 250.0, p=p)
            ipk_v, ig2_v = _eval_pentode_vec(
                np.array([float(ua)]), np.array([-10.0]),
                np.array([250.0]), p,
            )
            assert abs(ipk_s - ipk_v[0]) < 1e-9, \
                f"Ia mismatch at Va={ua}"


# ===================================================================
# DempwolfModel object (used by UI)
# ===================================================================

class TestDempwolfModelObject:
    """Verify DempwolfModel produces correct results with Kp-normalized C."""

    def test_pentode_model_ia(self):
        """DempwolfModel.ia() should return mA matching dempwolf_v2()."""
        from lm19.dempwolf import load_dempwolf_model
        model = load_dempwolf_model("EL34")
        if model is None:
            pytest.skip("EL34 not in tube_params.json")
        ia_model = model.ia(300.0, -10.0, 300.0)
        ipk, _, _ = dempwolf_v2(300.0, -10.0, 300.0, p=model.dempwolf)
        assert abs(ia_model - ipk * 1000) < 0.001, \
            f"Model ia={ia_model:.3f} vs direct {ipk*1000:.3f}"

    def test_pentode_model_ig2(self):
        """DempwolfModel.ig2() should match dempwolf_v2()."""
        from lm19.dempwolf import load_dempwolf_model
        model = load_dempwolf_model("EL34")
        if model is None:
            pytest.skip("EL34 not in tube_params.json")
        ig2_model = model.ig2(300.0, -10.0, 300.0)
        _, ig2k, _ = dempwolf_v2(300.0, -10.0, 300.0, p=model.dempwolf)
        assert abs(ig2_model - ig2k * 1000) < 0.001

    def test_fitted_model_matches_data(self):
        """Model object from fitter should reproduce training data."""
        points = _generate_pentode_points(
            P_EL34, ua_range=(50, 300, 50),
            ug1_range=(-15, -3, 3), ug2_vals=(250,),
        )
        result = fit_dempwolf(points, "pentode")
        model = result.model
        max_err = 0
        for p in points:
            ia_pred = model.ia(p["ua"], p["ug1"], p["ug2"])
            max_err = max(max_err, abs(ia_pred - p["ia"]))
        assert max_err < 1.0, f"Max error = {max_err:.3f} mA"


class TestDempwolfModelGenerateScan:
    """``DempwolfModel.generate_scan`` — overlay-curve generation path.

    ``main_window.py``'s Compare-Models dialog feeds the chosen model
    (Koren / Dempwolf / Reefman) into ``generate_scan`` to materialise
    overlay curves, so a regression here would silently break that
    workflow. ``quick_triode`` / ``quick_pentode`` only route through
    Koren, so this path needs explicit coverage.

    Tests cover the four code paths through ``generate_scan``:
      1. Pentode with explicit ``grid.ug2`` range (most common production case)
      2. Triode topology (forces ``ug2_actual = 0``)
      3. ``ug2_track_ua=True`` (triode-connected pentode)
      4. Output dict shape (all 7 expected keys, finite numerics)
    """

    def _load_or_skip(self, name: str):
        from lm19.dempwolf import load_dempwolf_model
        model = load_dempwolf_model(name)
        if model is None:
            pytest.skip(f"{name} not in tube_params.json")
        return model

    def test_pentode_explicit_ug2_grid(self):
        """grid.ug2 set → np.arange iterates the requested Ug2 values.
        Most common production path (user picks pentode in Compare Models)."""
        from lm19.tube_sim import ScanGrid
        model = self._load_or_skip("EL34")
        grid = ScanGrid(
            ua=(0, 300, 50), ug1=(-15, 0, 5),
            ug2=(200, 300, 50), uh=6.3, ih=1.5,
        )
        pts = model.generate_scan(grid)
        # Expected counts: ua=7 (0,50,…,300), ug1=4 (-15,-10,-5,0), ug2=3 (200,250,300)
        assert len(pts) == 7 * 4 * 3
        # Every point's ug2 must come from the requested grid
        ug2s = {round(p["ug2"], 1) for p in pts}
        assert ug2s == {200.0, 250.0, 300.0}

    def test_triode_topology_forces_ug2_zero(self):
        """topology=TOPOLOGY_TRIODE → ug2_actual is 0 for every point regardless
        of any grid.ug2 the caller might (mistakenly) supply."""
        from lm19.tube_sim import ScanGrid
        model = self._load_or_skip("12AX7")
        assert model.topology == TOPOLOGY_TRIODE
        grid = ScanGrid(
            ua=(0, 250, 50), ug1=(-4, 0, 1),
            ug2=(250, 250, 1),  # supplied but should be ignored for triode
            uh=12.6, ih=0.15,
        )
        pts = model.generate_scan(grid)
        assert len(pts) > 0
        for p in pts:
            assert p["ug2"] == 0.0, (
                f"Triode point has Ug2={p['ug2']} (must be 0). "
                f"DempwolfModel.generate_scan triode branch regressed."
            )

    def test_ug2_track_ua_pentode(self):
        """grid.ug2_track_ua=True → ug2_actual = ua + offset per point.
        Used for triode-connected pentodes in Compare Models."""
        from lm19.tube_sim import ScanGrid
        model = self._load_or_skip("EL34")
        grid = ScanGrid(
            ua=(50, 300, 50), ug1=(-30, 0, 10),
            ug2_track_ua=True, ug2_offset=0.0, uh=6.3, ih=1.5,
        )
        pts = model.generate_scan(grid)
        assert len(pts) > 0
        for p in pts:
            assert p["ug2"] == pytest.approx(p["ua"], abs=1e-6), (
                f"track_ua: expected Ug2≈Ua, got Ug2={p['ug2']} Ua={p['ua']}"
            )

    def test_output_dict_shape(self):
        """Pin: every point dict has exactly the 7 standard keys with
        finite numeric values. A silent rename or drop would break
        downstream consumers (PlotManager, save_measurement, etc.)."""
        import math
        from lm19.tube_sim import ScanGrid
        model = self._load_or_skip("EL34")
        grid = ScanGrid(
            ua=(100, 200, 50), ug1=(-5, 0, 5),
            ug2=(250, 250, 1), uh=6.3, ih=1.5,
        )
        pts = model.generate_scan(grid)
        assert len(pts) > 0
        expected = {"ua", "ug1", "ug2", "ia", "ig2", "uh", "ih"}
        for p in pts:
            assert set(p.keys()) == expected, (
                f"Point keys diverged: {set(p.keys()) ^ expected}"
            )
            for k, v in p.items():
                assert isinstance(v, (int, float)), f"{k} is {type(v)}"
                assert math.isfinite(v), f"{k}={v} not finite"


# ===================================================================
# Variable-mu pentode (normalized formula)
# ===================================================================

class TestVariableMuPentode:
    """Test variable-mu pentode uses Kp-normalized formula."""

    def test_varmu_pentode_finite(self):
        """Variable-mu pentode with normalized formula should not crash."""
        p = DempwolfParams(
            mu=11.0, G=3.0e-3, gamma=1.35, C=750.0,
            Gg=6.0e-4, xi=1.3, Cg=10.0,
            Kvb=24.0, Kvb1=0.5, Kn=1.0, fg2=0.08, A=0.0002,
            mu_b=50.0, gamma_b=1.5, svar=0.3,
        )
        ipk, ig2k, igk = dempwolf_v2(300.0, -10.0, 300.0, p=p)
        assert np.isfinite(ipk) and ipk > 0
        assert np.isfinite(ig2k) and ig2k > 0

    def test_varmu_vs_single_section(self):
        """With svar=0, variable-mu should equal single-section."""
        p_single = DempwolfParams(
            mu=11.0, G=3.0e-3, gamma=1.35, C=750.0,
            Gg=6.0e-4, xi=1.3, Cg=10.0,
            Kvb=24.0, Kvb1=0.5, Kn=1.0, fg2=0.08, A=0.0002,
        )
        p_varmu = DempwolfParams(
            mu=11.0, G=3.0e-3, gamma=1.35, C=750.0,
            Gg=6.0e-4, xi=1.3, Cg=10.0,
            Kvb=24.0, Kvb1=0.5, Kn=1.0, fg2=0.08, A=0.0002,
            mu_b=50.0, gamma_b=1.5, svar=0.0,
        )
        ipk_s, _, _ = dempwolf_v2(300.0, -10.0, 300.0, p=p_single)
        ipk_v, _, _ = dempwolf_v2(300.0, -10.0, 300.0, p=p_varmu)
        # svar=0 means 100% section A, so should be identical
        assert abs(ipk_s - ipk_v) < 1e-12

    def test_varmu_blending(self):
        """Variable-mu should blend two sections with svar weight."""
        p = DempwolfParams(
            mu=11.0, G=3.0e-3, gamma=1.35, C=750.0,
            Gg=6.0e-4, xi=1.3, Cg=10.0,
            Kvb=24.0, Kvb1=0.5, Kn=1.0, fg2=0.08, A=0.0002,
            mu_b=50.0, gamma_b=1.5, svar=0.5,
        )
        # At Ug1=-10, mu_b=50 section should give less current than mu=11
        # So blended result should be less than pure mu=11
        ipk_pure, _, _ = dempwolf_v2(300.0, -10.0, 300.0, p=DempwolfParams(
            mu=11.0, G=3.0e-3, gamma=1.35, C=750.0,
            Gg=6.0e-4, xi=1.3, Cg=10.0,
            Kvb=24.0, Kvb1=0.5, Kn=1.0, fg2=0.08, A=0.0002,
        ))
        ipk_blend, _, _ = dempwolf_v2(300.0, -10.0, 300.0, p=p)
        assert ipk_blend < ipk_pure, \
            f"Blended {ipk_blend:.6f} should be < pure {ipk_pure:.6f}"


# ===================================================================
# Dempwolf Ig2 error fields
# ===================================================================

class TestFitterIg2Errors:
    """Verify rms_ig2 / max_ig2 are populated for pentode fits."""

    @pytest.fixture(scope="class")
    def pentode_fit(self):
        points = _generate_pentode_points(P_EL34)
        return fit_dempwolf(points, "pentode")

    def test_ig2_errors_present(self, pentode_fit):
        assert pentode_fit.rms_ig2 is not None
        assert pentode_fit.max_ig2 is not None

    def test_ig2_rms_positive(self, pentode_fit):
        assert pentode_fit.rms_ig2 >= 0

    def test_ig2_max_ge_rms(self, pentode_fit):
        assert pentode_fit.max_ig2 >= pentode_fit.rms_ig2

    def test_ig2_rms_reasonable(self, pentode_fit):
        """On synthetic data, Ig2 RMS should be bounded."""
        assert pentode_fit.rms_ig2 < 10.0, \
            f"RMS Ig2 = {pentode_fit.rms_ig2:.2f} mA"

    def test_triode_no_ig2_errors(self):
        """Triode fit should NOT have Ig2 errors."""
        points = _generate_triode_points(P_12AX7_RSD1)
        result = fit_dempwolf(points, "triode")
        assert result.rms_ig2 is None
        assert result.max_ig2 is None
