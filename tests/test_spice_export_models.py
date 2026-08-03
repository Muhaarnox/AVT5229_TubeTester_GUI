"""Tests for SPICE export of Dempwolf and Reefman models."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.spice_export import fit_and_export_spice
from lm19.tube_sim import quick_pentode, quick_triode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def pentode_points():
    _model, pts = quick_pentode("EL84")
    return pts


@pytest.fixture
def triode_points():
    _model, pts = quick_triode("12AX7")
    return pts


# ---------------------------------------------------------------------------
# Reefman SPICE export
# ---------------------------------------------------------------------------

class TestReefmanSpiceExport:

    def test_pentode_creates_file(self, tmp_path, pentode_points):
        out = tmp_path / "reefman.sub"
        result = fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_REEFMAN)
        assert out.exists()
        assert result.rms_error > 0
        assert result.n_points >= 10

    def test_pentode_subcircuit_structure(self, tmp_path, pentode_points):
        out = tmp_path / "reefman.sub"
        fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_REEFMAN)
        content = out.read_text(encoding="utf-8")

        assert ".SUBCKT EL84 A G K G2" in content
        assert "PARAMS:" in content
        assert "MU=" in content
        assert "KG1=" in content
        assert "KG2=" in content
        assert "Ookg1mOokG2=" in content
        assert "alkg1palskg2=" in content
        assert "be=" in content
        assert "als=" in content
        # Cathode current: Koren-style E1
        assert "E1 7 0 VALUE=" in content
        assert "LOG(1+EXP" in content
        # Splitting E2
        assert "E2 8 0 VALUE" in content
        # Current sources
        assert "G1 A K VALUE" in content
        assert "G2 G2 K VALUE" in content
        assert "RCP A K 1G" in content
        assert ".ENDS EL84" in content
        # Grid current
        assert "D3 5 K DX" in content
        assert ".MODEL DX D" in content

    def test_reefman_header_info(self, tmp_path, pentode_points):
        out = tmp_path / "reefman.sub"
        fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_REEFMAN)
        content = out.read_text(encoding="utf-8")

        assert "Reefman" in content
        assert "LM19 Tube Tester" in content
        assert "RMS" in content

    def test_reefman_rejects_triode(self, tmp_path, triode_points):
        out = tmp_path / "reefman.sub"
        with pytest.raises(RuntimeError, match="pentode"):
            fit_and_export_spice(
                str(out), "12AX7", triode_points,
                topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_REEFMAN)

    def test_splitting_type_in_content(self, tmp_path, pentode_points):
        out = tmp_path / "reefman.sub"
        fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_REEFMAN)
        content = out.read_text(encoding="utf-8")
        # Should contain either Derk or DerkE splitting description
        assert "Derk" in content


# ---------------------------------------------------------------------------
# Dempwolf SPICE export — pentode
# ---------------------------------------------------------------------------

class TestDempwolfPentodeSpiceExport:

    def test_pentode_creates_file(self, tmp_path, pentode_points):
        out = tmp_path / "dempwolf.sub"
        result = fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_DEMPWOLF)
        assert out.exists()
        assert result.rms_error > 0
        assert result.n_points >= 10

    def test_pentode_subcircuit_structure(self, tmp_path, pentode_points):
        out = tmp_path / "dempwolf.sub"
        fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_DEMPWOLF)
        content = out.read_text(encoding="utf-8")

        assert ".SUBCKT EL84 A G K G2" in content
        assert "PARAMS:" in content
        assert "MU=" in content
        assert "G=" in content
        assert "GAMMA=" in content
        assert "KVB=" in content
        assert "KN=" in content
        assert "FG2=" in content
        # Cathode softplus
        assert "E1 7 0 VALUE=" in content
        assert "LOG(1+EXP" in content
        # Cathode emission E2
        assert "E2 8 0 VALUE=" in content
        # Kvb_eff E4
        assert "E4 10 0 VALUE=" in content
        # Alpha E5 with arctan
        assert "E5 11 0 VALUE=" in content
        assert "ATAN" in content
        # Current sources
        assert "G1 A K VALUE=" in content
        assert "G2 G2 K VALUE=" in content
        assert ".ENDS EL84" in content

    def test_dempwolf_header_info(self, tmp_path, pentode_points):
        out = tmp_path / "dempwolf.sub"
        fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_DEMPWOLF)
        content = out.read_text(encoding="utf-8")

        assert "Dempwolf" in content
        assert "LM19 Tube Tester" in content
        assert "softplus" in content
        assert "arctan" in content

    def test_twopi_param(self, tmp_path, pentode_points):
        out = tmp_path / "dempwolf.sub"
        fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_DEMPWOLF)
        content = out.read_text(encoding="utf-8")
        assert "TWOPI=0.6366197724" in content


# ---------------------------------------------------------------------------
# Dempwolf SPICE export — triode
# ---------------------------------------------------------------------------

class TestDempwolfTriodeSpiceExport:

    def test_triode_creates_file(self, tmp_path, triode_points):
        out = tmp_path / "dempwolf_tri.sub"
        result = fit_and_export_spice(
            str(out), "12AX7", triode_points,
            topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_DEMPWOLF)
        assert out.exists()
        assert result.rms_error > 0
        assert result.model_type == TOPOLOGY_TRIODE

    def test_triode_subcircuit_structure(self, tmp_path, triode_points):
        out = tmp_path / "dempwolf_tri.sub"
        fit_and_export_spice(
            str(out), "12AX7", triode_points,
            topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_DEMPWOLF)
        content = out.read_text(encoding="utf-8")

        assert ".SUBCKT 12AX7 A G K" in content
        assert "PARAMS:" in content
        assert "MU=" in content
        assert "GAMMA=" in content
        assert "KVBT=" in content
        assert "E1 7 0 VALUE=" in content
        assert "G1 A K VALUE=" in content
        assert "RCP A K 1G" in content
        assert ".ENDS 12AX7" in content

    def test_triode_connected_uses_triode(self, tmp_path, pentode_points):
        """triode_connected topology should produce triode subcircuit."""
        out = tmp_path / "dempwolf_tc.sub"
        result = fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_TRIODE_CONNECTED, model_type=MODEL_TYPE_DEMPWOLF)
        content = out.read_text(encoding="utf-8")
        assert result.model_type == TOPOLOGY_TRIODE
        assert ".SUBCKT" in content
        assert "A G K" in content


# ---------------------------------------------------------------------------
# Koren as the default SPICE model
# ---------------------------------------------------------------------------

class TestKorenSpiceDefault:

    def test_default_model_is_koren(self, tmp_path, pentode_points):
        out = tmp_path / "koren.sub"
        result = fit_and_export_spice(
            str(out), "EL84", pentode_points, topology=TOPOLOGY_PENTODE)
        content = out.read_text(encoding="utf-8")
        assert "Koren" in content
        assert result.model_type == TOPOLOGY_PENTODE

    def test_explicit_koren(self, tmp_path, pentode_points):
        out = tmp_path / "koren.sub"
        result = fit_and_export_spice(
            str(out), "EL84", pentode_points,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_KOREN)
        assert out.exists()
        assert result.rms_error > 0


# ---------------------------------------------------------------------------
# Low-mu power triode tests (6S19P, 6C33C)
# ---------------------------------------------------------------------------

import json
import numpy as np


def _make_low_mu_triode_points(mu=3.0, ia_max_ma=150.0, n_ug1=12, n_ua=15):
    """Synthetic data for a low-mu power triode (mu ~ 2-5, Ia up to 200mA)."""
    pts = []
    for i in range(n_ug1):
        ug1 = -30.0 + 28.0 * i / (n_ug1 - 1)  # -30 to -2V
        for j in range(n_ua):
            ua = 5.0 + 180.0 * j / (n_ua - 1)  # 5 to 185V
            # Simplified triode: Ia ~ (ua/mu + ug1)^1.5 scaled
            v_eff = ua / mu + ug1
            ia = ia_max_ma * max(0.0, v_eff / 50.0) ** 1.5 if v_eff > 0 else 0.0
            pts.append({"ua": round(ua, 1), "ug1": round(ug1, 1),
                        "ia": round(ia, 2)})
    return pts


class TestLowMuTriodeSynthetic:
    """Dempwolf fitter must handle low-mu power triodes (mu < 5)."""

    def test_dempwolf_low_mu_synthetic(self, tmp_path):
        """Synthetic low-mu triode: Dempwolf should fit with reasonable RMS."""
        pts = _make_low_mu_triode_points(mu=3.0, ia_max_ma=150.0)
        out = tmp_path / "low_mu.sub"
        result = fit_and_export_spice(
            str(out), "LowMu", pts, topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_DEMPWOLF)
        assert result.rms_error < 20.0, f"RMS={result.rms_error:.1f}mA too high"
        mu = result.params.get("mu", 0)
        assert 1.5 < mu < 8.0, f"mu={mu:.1f} should be near 3"

    def test_koren_low_mu_synthetic(self, tmp_path):
        """Synthetic low-mu triode: Koren should also work."""
        pts = _make_low_mu_triode_points(mu=3.0, ia_max_ma=150.0)
        out = tmp_path / "low_mu_k.sub"
        result = fit_and_export_spice(
            str(out), "LowMu", pts, topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_KOREN)
        assert result.rms_error < 15.0
        mu = result.params.get("mu", 0)
        assert 1.5 < mu < 8.0

    def test_dempwolf_very_low_mu(self, tmp_path):
        """mu=2.5 (like 6C33C): Dempwolf should still converge."""
        pts = _make_low_mu_triode_points(mu=2.5, ia_max_ma=300.0)
        out = tmp_path / "very_low_mu.sub"
        result = fit_and_export_spice(
            str(out), "VLowMu", pts, topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_DEMPWOLF)
        assert result.rms_error < 50.0, f"RMS={result.rms_error:.1f}mA"
        mu = result.params.get("mu", 0)
        assert mu < 10.0, f"mu={mu:.1f} should be low"

    @pytest.mark.parametrize("mu,ia_max", [(3.0, 150), (5.0, 80), (2.5, 300)])
    def test_dempwolf_mu_range(self, tmp_path, mu, ia_max):
        """Dempwolf should handle range of low-mu triodes."""
        pts = _make_low_mu_triode_points(mu=mu, ia_max_ma=ia_max)
        out = tmp_path / f"mu{mu}.sub"
        result = fit_and_export_spice(
            str(out), f"Mu{mu}", pts, topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_DEMPWOLF)
        ia_range = max(p["ia"] for p in pts)
        rms_pct = result.rms_error / max(ia_range, 1.0) * 100
        assert rms_pct < 15.0, f"mu={mu}: RMS={result.rms_error:.1f}mA ({rms_pct:.0f}%)"


class TestLowMuTriodeRealData:
    """Dempwolf fitter on real low-mu triode measurements."""

    @pytest.fixture
    def pts_6s19p(self):
        path = Path(__file__).parent / "spice_test_data" / "converted" / \
               "triode_6S19P_real.json"
        if not path.exists():
            pytest.skip("6S19P test data not available")
        with open(path, encoding="utf-8") as f:
            return json.load(f)["points"]

    @pytest.fixture
    def pts_6c33c(self):
        path = Path(__file__).parent / "spice_test_data" / "converted" / \
               "triode_6C33C_curvetracedata.json"
        if not path.exists():
            pytest.skip("6C33C test data not available")
        with open(path) as f:
            return json.load(f)["points"]

    def test_6s19p_dempwolf(self, tmp_path, pts_6s19p):
        """6S19P (mu~3): Dempwolf should fit with RMS < 10mA."""
        out = tmp_path / "6S19P.sub"
        result = fit_and_export_spice(
            str(out), "6S19P", pts_6s19p, topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_DEMPWOLF)
        assert result.rms_error < 10.0, f"6S19P Dempwolf RMS={result.rms_error:.1f}mA"
        assert 1.5 < result.params["mu"] < 6.0

    def test_6s19p_koren(self, tmp_path, pts_6s19p):
        """6S19P: Koren reference fit."""
        out = tmp_path / "6S19P_k.sub"
        result = fit_and_export_spice(
            str(out), "6S19P", pts_6s19p, topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_KOREN)
        assert result.rms_error < 5.0
        assert 1.5 < result.params["mu"] < 6.0

    def test_6c33c_dempwolf(self, tmp_path, pts_6c33c):
        """6C33C (mu~2.7, Ia up to 500mA): Dempwolf should converge."""
        out = tmp_path / "6C33C.sub"
        result = fit_and_export_spice(
            str(out), "6C33C", pts_6c33c, topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_DEMPWOLF)
        assert result.rms_error < 50.0, f"6C33C Dempwolf RMS={result.rms_error:.1f}mA"
        assert 1.5 < result.params["mu"] < 6.0

    def test_6c33c_koren(self, tmp_path, pts_6c33c):
        """6C33C: Koren reference fit."""
        out = tmp_path / "6C33C_k.sub"
        result = fit_and_export_spice(
            str(out), "6C33C", pts_6c33c, topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_KOREN)
        assert result.rms_error < 20.0
        assert 1.5 < result.params["mu"] < 6.0

    def test_6s19p_dempwolf_vs_koren_comparable(self, tmp_path, pts_6s19p):
        """Dempwolf RMS should be within 3x of Koren on same data."""
        out_k = tmp_path / "k.sub"
        out_d = tmp_path / "d.sub"
        rk = fit_and_export_spice(str(out_k), "k", pts_6s19p, topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_KOREN)
        rd = fit_and_export_spice(str(out_d), "d", pts_6s19p, topology=TOPOLOGY_TRIODE, model_type=MODEL_TYPE_DEMPWOLF)
        ratio = rd.rms_error / max(rk.rms_error, 0.01)
        assert ratio < 3.0, f"Dempwolf {rd.rms_error:.1f}mA vs Koren {rk.rms_error:.1f}mA (ratio {ratio:.1f}x)"


# ---------------------------------------------------------------------------
# Reefman on real pentode data
# ---------------------------------------------------------------------------

from lm19.reefman import fit_reefman
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)
from lm19.tube_model_base import (
    MODEL_TYPE_DEMPWOLF,
    MODEL_TYPE_KOREN,
    MODEL_TYPE_REEFMAN,
)


@pytest.mark.timeout(60)
class TestReefmanRealData:
    """Reefman fitter on real pentode measurements with sufficient Ug2 levels."""

    @pytest.fixture
    def pts_el84_er_pent(self):
        path = Path(__file__).parent / "spice_test_data" / "converted" / \
               "pentode_EL84_ER_L1_real.json"
        if not path.exists():
            pytest.skip("EL84 ER pentode data not available")
        with open(path, encoding="utf-8") as f:
            return json.load(f)["points"]

    @pytest.fixture
    def pts_el84_sov_pent(self):
        path = Path(__file__).parent / "spice_test_data" / "converted" / \
               "pentode_EL84_SOVTEK_L1_real.json"
        if not path.exists():
            pytest.skip("EL84 SOVTEK pentode data not available")
        with open(path, encoding="utf-8") as f:
            return json.load(f)["points"]

    @pytest.fixture
    def pts_6p1p_pent(self):
        path = Path(__file__).parent / "spice_test_data" / "converted" / \
               "pentode_6P1P_real.json"
        if not path.exists():
            pytest.skip("6P1P pentode data not available")
        with open(path, encoding="utf-8") as f:
            return json.load(f)["points"]

    def test_el84_er_reefman_rms(self, pts_el84_er_pent):
        """EL84 ER pentode: Reefman RMS should be < 5 mA on real data."""
        result = fit_reefman(pts_el84_er_pent, "pentode")
        assert result.rms_error < 5.0, f"EL84 ER Reefman RMS={result.rms_error:.1f}mA"

    def test_el84_sovtek_reefman_rms(self, pts_el84_sov_pent):
        """EL84 SOVTEK pentode: Reefman should fit well."""
        result = fit_reefman(pts_el84_sov_pent, "pentode")
        assert result.rms_error < 5.0, f"EL84 SOVTEK Reefman RMS={result.rms_error:.1f}mA"

    def test_6p1p_reefman_rms(self, pts_6p1p_pent):
        """6P1P pentode: Reefman should fit well."""
        result = fit_reefman(pts_6p1p_pent, "pentode")
        assert result.rms_error < 5.0, f"6P1P Reefman RMS={result.rms_error:.1f}mA"

    def test_reefman_vs_koren_pentode(self, tmp_path, pts_el84_er_pent):
        """On real pentode data, Reefman should be comparable to Koren."""
        rr = fit_reefman(pts_el84_er_pent, "pentode")
        rk = fit_and_export_spice(
            str(tmp_path / "k.sub"), "k", pts_el84_er_pent,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_KOREN,
        )
        # Reefman should not be more than 3x worse
        ratio = rr.rms_error / max(rk.rms_error, 0.01)
        assert ratio < 3.0, f"Reefman {rr.rms_error:.1f}mA vs Koren {rk.rms_error:.1f}mA"

    def test_reefman_vs_dempwolf_pentode(self, pts_6p1p_pent):
        """On real pentode data, Reefman and Dempwolf should be comparable."""
        from lm19.dempwolf import fit_dempwolf
        rr = fit_reefman(pts_6p1p_pent, "pentode")
        rd = fit_dempwolf(pts_6p1p_pent, "pentode")
        # Both should be reasonable, ratio within 3x
        ratio = rr.rms_error / max(rd.rms_error, 0.01)
        assert ratio < 3.0, f"Reefman {rr.rms_error:.1f}mA vs Dempwolf {rd.rms_error:.1f}mA"

    def test_all_three_pentode_physical(self, tmp_path, pts_el84_er_pent):
        """All 3 fitters should produce physical results on real pentode data."""
        from lm19.dempwolf import fit_dempwolf
        from lm19.constants import MAX_SANE_THD_PCT
        ia_max = max(p["ia"] for p in pts_el84_er_pent)

        rk = fit_and_export_spice(
            str(tmp_path / "k.sub"), "k", pts_el84_er_pent,
            topology=TOPOLOGY_PENTODE, model_type=MODEL_TYPE_KOREN,
        )
        rd = fit_dempwolf(pts_el84_er_pent, "pentode")
        rr = fit_reefman(pts_el84_er_pent, "pentode")

        for name, rms in [("Koren", rk.rms_error), ("Dempwolf", rd.rms_error),
                          ("Reefman", rr.rms_error)]:
            rms_pct = rms / ia_max * 100
            assert rms_pct < 10.0, f"{name}: RMS={rms:.1f}mA ({rms_pct:.0f}% of Ia_max)"
