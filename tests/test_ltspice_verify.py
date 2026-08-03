"""LTspice verification of the amplifier analysis.

Offline pins: the netlist builder for all 4 circuits (engine
idealization: fixed bias, ideal choke/transformer), the .four table
parser (authentic LTspice 26 format), IMD from a synthetic two-tone
(exact physics: attenuated sidebands are recovered), average current
from .raw, worker stop plumbing.

Integration (@requires_ltspice, real LTspice): SE and PP EL84 —
physical ranges; PP also discriminates coupled half-winding polarity
(a K sign error zeroes Pout); cancel returns a partial result.
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.ltspice_raw import LTSPICE_EXE
from lm19.ltspice_verify import (
    IMD_F1_HZ,
    IMD_F2_HZ,
    LtspiceVerifyError,
    VerifyRequest,
    build_verify_netlist,
    imd_from_signal,
    parse_four_table,
    read_ltspice_log,
    run_verification,
    subckt_name_of,
    tube_current_avg_ma,
)
from lm19.amplifier.constants import (
    CIRCUIT_PP,
    CIRCUIT_SE,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)
from lm19.tube_model_base import (
    MODEL_TYPE_KOREN,
)

requires_ltspice = pytest.mark.skipif(
    not Path(LTSPICE_EXE).exists(),
    reason=f"LTspice not found at {LTSPICE_EXE}",
)

_NL_ARGS = dict(sub_path="C:/tmp/m.sub", sub_name="EL84", ub=300.0,
                ra_ohm=5600.0, bias_v=-11.0, amp_v=9.0, ug2_v=300.0)


class TestNetlistBuilder:
    def test_se_pentode(self):
        n = build_verify_netlist("se", pentode=True, **_NL_ARGS)
        assert '.include "C:/tmp/m.sub"' in n
        assert "Ra ub out 5600" in n
        assert "XU out g 0 g2 EL84" in n          # pin order A G K G2
        assert "SINE(-11 9 1000)" in n            # bias/amp/f0 in right slots
        assert "Vg2 g2 0 300" in n
        assert ".four 1000 9 5 V(out)" in n
        assert ".tran 0 0.015 0.01 2e-06" in n

    def test_se_triode_has_no_screen(self):
        n = build_verify_netlist("se", pentode=False, **_NL_ARGS)
        assert "XU out g 0 EL84" in n             # triode pin order A G K
        assert "Vg2" not in n and "g2" not in n

    def test_cf_screen_rides_on_cathode(self):
        n = build_verify_netlist("cf", pentode=True, **_NL_ARGS)
        assert "XU ub g out g2 EL84" in n         # anode to supply
        assert "Rk out 0 5600" in n
        assert "Vg2 g2 out 300" in n              # Vg2k constant, NOT g2-0

    def test_se_xfmr_parafeed_idealization(self):
        n = build_verify_netlist("se_xfmr", pentode=False, **_NL_ARGS)
        assert "Lfeed ub a" in n
        assert "Cc a out" in n
        assert "Rl out 0 5600" in n
        assert "XU a g 0 EL84" in n

    def test_pp_coupled_halves_and_antiphase(self):
        n = build_verify_netlist("pp", pentode=True, **_NL_ARGS)
        assert "K1 L1 L2 1" in n
        assert "Rl a1 a2 5600" in n               # Ra_aa across full primary
        assert "XA a1 ga 0 g2a EL84" in n
        assert "XB a2 gb 0 g2b EL84" in n
        assert "Vg2a g2a 0 300" in n and "Vg2b g2b 0 300" in n
        assert "SINE(-11 9 1000 0 0 180)" in n    # tube B in anti-phase
        assert "Bout out 0 V=V(a1)-V(a2)" in n

    def test_ul_tap_screen_follows_own_anode(self):
        """UL law V=Ug2·(1−tap)+tap·V(anode); in PP each screen must track
        ITS OWN anode (a crossed reference is the plausible wiring bug)."""
        n = build_verify_netlist("pp", pentode=True, ul_tap=0.43, **_NL_ARGS)
        assert "Bg2a g2a 0 V=171+0.43*V(a1)" in n
        assert "Bg2b g2b 0 V=171+0.43*V(a2)" in n
        assert "Vg2a" not in n                    # fixed supply replaced
        n_se = build_verify_netlist("se", pentode=True, ul_tap=0.43,
                                    **_NL_ARGS)
        assert "Bg2 g2 0 V=171+0.43*V(out)" in n_se

    def test_ul_tap_zero_keeps_fixed_screen(self):
        n = build_verify_netlist("se", pentode=True, ul_tap=0.0, **_NL_ARGS)
        assert "Vg2 g2 0 300" in n and "Bg2" not in n

    def test_ul_tap_on_cf_refused(self):
        with pytest.raises(LtspiceVerifyError, match="cathode follower"):
            build_verify_netlist("cf", pentode=True, ul_tap=0.43, **_NL_ARGS)

    def test_two_tone_replaces_four_directive(self):
        tt = {"f1": 60.0, "a1": 4.0, "f2": 7000.0, "a2": 1.0}
        n = build_verify_netlist("se", pentode=False, two_tone=tt, **_NL_ARGS)
        assert ".four" not in n
        assert "SINE(-11 4 60)" in n
        assert "SINE(0 1 7000)" in n

    def test_pentode_without_ug2_refused(self):
        args = dict(_NL_ARGS, ug2_v=None)
        with pytest.raises(LtspiceVerifyError, match="Ug2"):
            build_verify_netlist("se", pentode=True, **args)

    def test_unknown_circuit_refused(self):
        with pytest.raises(LtspiceVerifyError, match="unknown circuit"):
            build_verify_netlist("otl", pentode=False, **_NL_ARGS)


# Authentic LTspice 26 log block (live run, EL84 SE @ ±9 V).
_FOUR_BLOCK = (
    "Fourier components of V(out)\nN-Period=4\nDC component:138.198\n\n"
    "Harmonic\tFrequency\t Fourier \tNormalized\t Phase  \tNormalized\n"
    " Number \t  [Hz]   \tComponent\t Component\t[degree]\tPhase [deg]\n"
    "    1   \t 1.000e+03\t 1.485e+02\t 1.000e+00\t  -89.99\t    0.00\n"
    "    2   \t 2.000e+03\t 2.047e+01\t 1.379e-01\t -179.96\t  -89.97\n"
    "    3   \t 3.000e+03\t 1.659e+01\t 1.118e-01\t  -89.97\t    0.02\n"
    "    9   \t 9.000e+03\t 2.000e-01\t 1.347e-03\t  -89.80\t    0.19\n"
    "Partial Harmonic Distortion: 18.149194%\n"
    "Total Harmonic Distortion:   18.149307%\n"
)


class TestFourParser:
    def test_authentic_block(self):
        out = parse_four_table(_FOUR_BLOCK)
        assert out["fund_v"] == pytest.approx(148.5)
        assert out["thd_pct"] == pytest.approx(18.149307)
        assert out["hd_pct"][2] == pytest.approx(13.79)
        assert out["hd_pct"][3] == pytest.approx(11.18)
        assert out["hd_pct"][9] == pytest.approx(0.1347)

    def test_partial_line_not_mistaken_for_total(self):
        assert parse_four_table(_FOUR_BLOCK)["thd_pct"] != pytest.approx(
            18.149194)

    def test_missing_table_raises(self):
        with pytest.raises(LtspiceVerifyError, match="Fourier"):
            parse_four_table("simulation aborted\n")


class TestSubcktName:
    def test_reads_actual_name(self, tmp_path):
        p = tmp_path / "verify_model.sub"
        p.write_text("* comment\n.SUBCKT EL84 A G K G2\n.ENDS\n")
        assert subckt_name_of(p) == "EL84"   # NOT the file stem

    def test_missing_subckt_raises(self, tmp_path):
        p = tmp_path / "empty.sub"
        p.write_text("* nothing here\n")
        with pytest.raises(LtspiceVerifyError, match="SUBCKT"):
            subckt_name_of(p)


class TestLogEncoding:
    def test_utf16_log(self, tmp_path):
        p = tmp_path / "a.log"
        p.write_bytes("Total: ok".encode("utf-16-le"))
        assert "Total" in read_ltspice_log(p)

    def test_cp1252_log(self, tmp_path):
        p = tmp_path / "b.log"
        p.write_bytes(b"Total Harmonic Distortion: 1.5%")
        assert "1.5%" in read_ltspice_log(p)


class TestImdFromSignal:
    def test_recovers_planted_sidebands(self):
        """Physically honest synthetic: carrier at f2 with distinct 2nd/3rd
        order sidebands; the analyzer must read back the planted ratios."""
        f1, f2 = IMD_F1_HZ, IMD_F2_HZ
        a2, s2, s3 = 10.0, 0.5, 0.2          # IMD2=5%, IMD3=2% — asymmetric
        t = np.linspace(0.0, 8 / f1, 240_000)
        v = (a2 * np.sin(2 * np.pi * f2 * t)
             + s2 * np.sin(2 * np.pi * (f2 - f1) * t)
             + s2 * np.sin(2 * np.pi * (f2 + f1) * t)
             + s3 * np.sin(2 * np.pi * (f2 - 2 * f1) * t)
             + s3 * np.sin(2 * np.pi * (f2 + 2 * f1) * t))
        imd = imd_from_signal(t, v)
        assert imd["imd2"] == pytest.approx(5.0, rel=0.02)
        assert imd["imd3"] == pytest.approx(2.0, rel=0.02)

    def test_window_too_short_raises(self):
        t = np.linspace(0.0, 0.5 / IMD_F1_HZ, 1000)
        with pytest.raises(LtspiceVerifyError, match="2 low-tone"):
            imd_from_signal(t, np.sin(t))


class TestTubeCurrentAvg:
    def test_time_weighted_average(self):
        # non-uniform grid: plain mean would give a different answer
        t = np.array([0.0, 1.0, 3.0])
        ia = np.array([0.0, 0.010, 0.010])
        raw = {"variables": ["time", "Ix(xu:A)"],
               "data": np.column_stack([t, ia])}
        # trapezoid: (0.005*1 + 0.010*2)/3 = 0.008(3) A → 8.33 mA
        assert tube_current_avg_ma(raw) == pytest.approx(8.3333, rel=1e-3)

    def test_no_tube_variable_returns_none(self):
        raw = {"variables": ["time", "V(out)"],
               "data": np.zeros((3, 2))}
        assert tube_current_avg_ma(raw) is None


class TestLtspiceExeOverride:
    def test_configured_exe_checked_instead_of_default(self, tmp_path):
        from lm19.ltspice_verify import ltspice_available
        missing = tmp_path / "nope" / "LTspice.exe"
        assert ltspice_available(str(missing)) is False
        present = tmp_path / "MyLTspice.exe"
        present.write_bytes(b"MZ")
        assert ltspice_available(str(present)) is True

    def test_run_verification_reports_configured_path(self, tmp_path):
        req = VerifyRequest(circuit=CIRCUIT_SE, tube_type="X", topology=TOPOLOGY_TRIODE,
                            points=[])
        with pytest.raises(LtspiceVerifyError,
                           match=r"nope[/\\]LTspice\.exe"):
            run_verification(req, workdir=str(tmp_path),
                             ltspice_exe=str(tmp_path / "nope"
                                             / "LTspice.exe"))


class TestResolveVerifyWorkdir:
    def test_empty_config_uses_system_temp(self, tmp_path):
        from lm19.ltspice_verify import resolve_verify_workdir
        wd = resolve_verify_workdir("", tmp_path)
        assert wd.exists() and "lm19_verify_" in wd.name
        assert tmp_path not in wd.parents   # NOT under the anchor

    def test_relative_config_resolves_against_anchor(self, tmp_path):
        from lm19.ltspice_verify import resolve_verify_workdir
        wd = resolve_verify_workdir("ltspice_runs", tmp_path)
        assert wd.parent == tmp_path / "ltspice_runs"
        assert wd.name.startswith("verify_")

    def test_absolute_config_used_as_is_with_collision_suffix(
            self, tmp_path):
        from lm19.ltspice_verify import resolve_verify_workdir
        first = resolve_verify_workdir(str(tmp_path), tmp_path / "other")
        first.mkdir()
        second = resolve_verify_workdir(str(tmp_path), tmp_path / "other")
        assert first.parent == tmp_path and second.parent == tmp_path
        assert first != second              # same-second collision suffixed


class TestWorkerPlumbing:
    def test_stop_flag_reaches_run_verification(self, monkeypatch):
        from app.workers import AmpVerifyWorker
        import lm19.ltspice_verify as lv

        captured = {}

        def fake_run(req, *, workdir, ltspice_exe=None, stop=None,
                     progress=None):
            captured["stop"] = stop
            return lv.VerifyResult()

        monkeypatch.setattr(lv, "run_verification", fake_run)
        req = VerifyRequest(circuit=CIRCUIT_SE, tube_type="X", topology=TOPOLOGY_TRIODE,
                            points=[])
        worker = AmpVerifyWorker(req, "C:/tmp")
        emitted = []
        worker.finished_result.connect(lambda r: emitted.append(r))
        worker._execute()
        assert emitted, "finished_result not emitted"
        assert captured["stop"]() is False
        worker.stop()
        assert captured["stop"]() is True   # live view of _stop_requested


# ----------------------------------------------------------------------
# Integration — real LTspice runs
# ----------------------------------------------------------------------


@requires_ltspice
@pytest.mark.timeout(300)
class TestLiveVerification:
    @pytest.fixture(scope="class")
    def el84_points(self):
        from lm19.tube_sim import quick_pentode
        _, pts = quick_pentode("EL84")
        return pts

    def test_se_el84_physical_ranges(self, el84_points, tmp_path):
        req = VerifyRequest(
            circuit=CIRCUIT_SE, tube_type="EL84", topology=TOPOLOGY_PENTODE,
            points=el84_points, model_type=MODEL_TYPE_KOREN, ub=300.0, ra_kohm=5.6,
            ug1_bias=-11.0, half_swing=5.0, ug2=300.0)
        vr = run_verification(req, workdir=str(tmp_path / "se"))
        run = vr.runs[0]
        assert 3.0 < run.thd_pct < 40.0
        assert 200.0 < run.pout_fund_mw < 6000.0
        assert run.ia_avg_ma is not None and 5.0 < run.ia_avg_ma < 80.0
        assert run.hd_pct[2] > 0.5      # SE: strong 2nd harmonic
        assert vr.fit_rms_ma is not None
        # bulky waveforms deleted after parsing; text artifacts kept
        wd = Path(vr.workdir)
        assert not list(wd.glob("*.raw")), "parsed .raw files must be deleted"
        assert list(wd.glob("*.cir")) and list(wd.glob("*.log"))
        # relocatable for manual re-runs: bare-filename include, not absolute
        cir_text = next(wd.glob("*.cir")).read_text(encoding="ascii")
        assert '.include "verify_model.sub"' in cir_text

    def test_pp_el84_matches_datasheet_scale(self, el84_points, tmp_path):
        """PP EL84 300V/8k/−11V/±9V: P1 ≈ 17–18 W @ THD ≈ 9–10% (engine
        reference 17.6 W @ 9.7). A wrong coupling polarity of the primary
        halves collapses Pout to ~0 — this pin discriminates it."""
        req = VerifyRequest(
            circuit=CIRCUIT_PP, tube_type="EL84", topology=TOPOLOGY_PENTODE,
            points=el84_points, model_type=MODEL_TYPE_KOREN, ub=300.0, ra_kohm=8.0,
            ug1_bias=-11.0, half_swing=9.0, ug2=300.0)
        vr = run_verification(req, workdir=str(tmp_path / "pp"))
        run = vr.runs[0]
        assert 14_000.0 < run.pout_fund_mw < 21_000.0
        assert 6.0 < run.thd_pct < 13.0
        assert run.hd_pct[2] < 1.0      # matched pair: even harmonics cancel

    def test_pp_ul43_matches_reference(self, el84_points, tmp_path):
        """UL 43% tap, same OP: project reference (DFT engine)
        is P1 ≈ 7.8 W @ THD ≈ 2.1% — far below the pentode connection.
        A screen NOT following its anode reproduces pentode numbers."""
        req = VerifyRequest(
            circuit=CIRCUIT_PP, tube_type="EL84", topology=TOPOLOGY_PENTODE,
            points=el84_points, model_type=MODEL_TYPE_KOREN, ub=300.0, ra_kohm=8.0,
            ug1_bias=-11.0, half_swing=9.0, ug2=300.0, ul_tap=0.43)
        vr = run_verification(req, workdir=str(tmp_path / "ul"))
        run = vr.runs[0]
        assert 5_500.0 < run.pout_fund_mw < 11_000.0
        assert run.thd_pct < 5.0

    def test_cancel_returns_partial_with_warning(self, el84_points,
                                                 tmp_path):
        req = VerifyRequest(
            circuit=CIRCUIT_SE, tube_type="EL84", topology=TOPOLOGY_PENTODE,
            points=el84_points, model_type=MODEL_TYPE_KOREN, ub=300.0, ra_kohm=5.6,
            ug1_bias=-11.0, half_swing=5.0, ug2=300.0)
        vr = run_verification(req, workdir=str(tmp_path / "cancel"),
                              stop=lambda: True)
        assert vr.runs == []
        assert "cancelled" in vr.warnings
