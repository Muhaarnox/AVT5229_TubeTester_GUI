"""Part 4 of the optimizer physical validation: independent LTspice .tran
cross-check of THD and fundamental power.

The SPICE subcircuit is generated from the SAME Koren reference
parameters the Python model uses (no fitting in between), the stage is
driven by a sine in LTspice's transient solver, and ``.four`` extracts
THD + the fundamental — a completely independent computation of the same
physics. Agreement validates the whole chain model → load line → DFT
solve → THD/Pout, including the PP joint ideal-OPT solver (the coupled
inductors in LTspice reproduce the per-tube impedance kink natively).

Gated on the installed LTspice binary like test_ltspice_roundtrip.py.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.amplifier import (
    PushPullLoadLine,
    ResistiveLoadLine,
    compute_distortion_dft,
    compute_distortion_dft_pp,
)
from lm19.ltspice_raw import LTSPICE_EXE
from lm19.spice_export.koren import _generate_pentode_subcircuit
from lm19.tube_sim import load_model

requires_ltspice = pytest.mark.skipif(
    not Path(LTSPICE_EXE).exists(),
    reason=f"LTspice not found at {LTSPICE_EXE}",
)

# ── module local constants ──
_LTSPICE_TIMEOUT_S = 120
_F_HZ = 1000.0
# Agreement tolerances Python-DFT vs LTspice .four on the same equations:
# residual differences come from the transient solver (timestep, trap
# integration, .four windowing) vs our steady-state solve.
_THD_REL_TOL = 0.10            # 10% relative on THD
_THD_ABS_TOL_PCT = 0.3         # or 0.3 THD points absolute, whichever wider
_PFUND_REL_TOL = 0.05          # 5% on fundamental power
# PP ideal-OPT emulation: half-primary inductance and the impedance ratio
# 4·L_half/L_sec = Zaa/RL (K=1 coupled inductors).
_PP_L_HALF_H = 100.0
_PP_RL_OHM = 8.0


def _el84_sub(tmpdir: str) -> tuple:
    """Write an EL84 .sub from the REFERENCE Koren params (no fit)."""
    model = load_model("EL84")
    k = model.koren
    text = _generate_pentode_subcircuit(
        "EL84VAL", "EL84", k.mu, k.ex, k.kg1, k.kp, k.kvb, k.kg2,
        rms_error=0.0, max_error=0.0, n_points=0, backend="reference",
    )
    sub_path = os.path.join(tmpdir, "el84val.sub")
    Path(sub_path).write_text(text)
    return model, sub_path


def _run_ltspice_net(net_path: str) -> str:
    """Run a netlist in batch mode, return the .log text (.four output)."""
    subprocess.run([LTSPICE_EXE, "-b", net_path],
                   capture_output=True, timeout=_LTSPICE_TIMEOUT_S)
    log_path = str(Path(net_path).with_suffix(".log"))
    raw = Path(log_path).read_bytes()
    # LTspice writes UTF-16LE logs on Windows.
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16")
    return raw.decode("cp1252", errors="replace")


def _parse_four(log_text: str) -> tuple:
    """Extract (fundamental_amplitude_A, thd_pct) from a .four block."""
    m = re.search(r"Total Harmonic Distortion:\s*([\d.eE+-]+)%", log_text)
    assert m, f"no THD in LTspice log:\n{log_text[-2000:]}"
    thd_pct = float(m.group(1))
    m1 = re.search(
        r"^\s*1\s+\S+\s+([\d.eE+-]+)", log_text, flags=re.MULTILINE)
    assert m1, "no fundamental row in .four table"
    return float(m1.group(1)), thd_pct


@requires_ltspice
class TestSeTranAgreement:
    """SE resistive EL84 stage: Python DFT vs LTspice transient."""

    _UB, _RA_K, _UG2 = 300.0, 5.0, 250.0
    _BIAS, _SWING = -7.0, 4.0

    def _ltspice(self, tmpdir: str) -> tuple:
        _, sub = _el84_sub(tmpdir)
        net = os.path.join(tmpdir, "se_tran.net")
        Path(net).write_text(f"""* SE EL84 tran validation
.include {sub}
VB N1 0 DC {self._UB}
RA N1 A {self._RA_K * 1000:.0f}
XV1 A G 0 SC EL84VAL
VSC SC 0 DC {self._UG2}
VG G 0 SINE({self._BIAS} {self._SWING} {_F_HZ:.0f})
.options plotwinsize=0
.tran 0 10m 0 1u
.four {_F_HZ:.0f} 9 I(RA)
.end
""")
        return _parse_four(_run_ltspice_net(net))

    def test_thd_and_fundamental_power_agree(self) -> None:
        model = load_model("EL84")
        d = compute_distortion_dft(
            model, ResistiveLoadLine(self._UB, self._RA_K),
            ug1_bias=self._BIAS, half_swing=self._SWING,
            ug2=self._UG2, ub=self._UB)
        assert d is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            i1_a, thd_lt = self._ltspice(tmpdir)
        # THD: relative band with an absolute floor.
        tol = max(_THD_REL_TOL * thd_lt, _THD_ABS_TOL_PCT)
        assert abs(d["thd"] - thd_lt) < tol, (d["thd"], thd_lt)
        # Fundamental power: P1 = I1²·Ra/2 on both sides.
        p1_lt_mw = (i1_a ** 2) * (self._RA_K * 1000.0) / 2.0 * 1000.0
        assert d["pout_fund_mw"] == pytest.approx(
            p1_lt_mw, rel=_PFUND_REL_TOL), (d["pout_fund_mw"], p1_lt_mw)


@requires_ltspice
class TestPpTranAgreement:
    """PP EL84 with an ideal center-tapped OPT (K=1 coupled inductors):
    LTspice natively reproduces the per-tube impedance kink the joint
    solver models — agreement here is the independent acceptance of
    ``_pp_joint_solve_vec``."""

    _UB, _ZAA_K, _UG2 = 300.0, 8.0, 300.0
    _BIAS, _SWING = -11.0, 9.0

    def _ltspice(self, tmpdir: str) -> tuple:
        _, sub = _el84_sub(tmpdir)
        # 4·L_half / L_sec = Zaa / RL
        l_sec = 4.0 * _PP_L_HALF_H * _PP_RL_OHM / (self._ZAA_K * 1000.0)
        net = os.path.join(tmpdir, "pp_tran.net")
        Path(net).write_text(f"""* PP EL84 tran validation (ideal OPT)
.include {sub}
VB NB 0 DC {self._UB}
L1 NB A1 {_PP_L_HALF_H}
L2 A2 NB {_PP_L_HALF_H}
L3 S1 0 {l_sec}
K1 L1 L2 L3 1
RL S1 0 {_PP_RL_OHM}
XV1 A1 GA 0 SC EL84VAL
XV2 A2 GB 0 SC EL84VAL
VSC SC 0 DC {self._UG2}
VGA GA 0 SINE({self._BIAS} {self._SWING} {_F_HZ:.0f})
VGB GB 0 SINE({self._BIAS} -{self._SWING} {_F_HZ:.0f})
.options plotwinsize=0
.tran 0 10m 0 1u
.four {_F_HZ:.0f} 9 I(RL)
.end
""")
        return _parse_four(_run_ltspice_net(net))

    def test_pp_thd_and_power_agree(self) -> None:
        model = load_model("EL84")
        d = compute_distortion_dft_pp(
            model, PushPullLoadLine(self._UB, ra_aa=self._ZAA_K, ra_dc=0.1),
            ug1_bias=self._BIAS, half_swing=self._SWING, ug2=self._UG2)
        assert d is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            i1_a, thd_lt = self._ltspice(tmpdir)
        tol = max(_THD_REL_TOL * thd_lt, _THD_ABS_TOL_PCT)
        assert abs(d["thd"] - thd_lt) < tol, (d["thd"], thd_lt)
        # Fundamental power at the secondary load == composite fundamental
        # power (ideal transformer): P1 = I1²·RL/2.
        p1_lt_mw = (i1_a ** 2) * _PP_RL_OHM / 2.0 * 1000.0
        assert d["pout_fund_mw"] == pytest.approx(
            p1_lt_mw, rel=_PFUND_REL_TOL), (d["pout_fund_mw"], p1_lt_mw)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
