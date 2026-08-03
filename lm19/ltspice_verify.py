"""LTspice verification of the amplifier analysis (stage 2, slice B).

Builds a purpose-built netlist that reproduces the engine's exact
idealization — fixed grid bias (V-source, no Rk auto-bias), fixed screen
supply, ideal DC feed (huge choke / perfectly-coupled transformer halves)
— fits the current measurement points into a ``.sub`` (the same
``fit_and_export_spice`` path the SPICE export uses), runs an LTspice
batch ``.tran`` with a sine drive and reads THD/HD2..HD9/fundamental
power back from the ``.four`` table in the ``.log`` file.

Sine is the correct probe: every engine distortion method models a
sinusoidal Ug1 drive on a static load line — by design;
optional extras are an amplitude sweep and a two-tone SMPTE IMD run
(numpy FFT over the ``.raw`` output).

No Qt imports — the UI drives this through ``app.workers.AmpVerifyWorker``.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from lm19.io_utils import make_unique_path
from lm19.ltspice_raw import LTSPICE_EXE, get_variable, parse_raw
from lm19.amplifier.constants import (
    CIRCUIT_CF,
    CIRCUIT_PP,
    CIRCUIT_SE,
    CIRCUIT_SE_XFMR,
)
from lm19.constants import (
    TOPOLOGY_TRIODE,
)
from lm19.tube_model_base import (
    MODEL_TYPE_KOREN,
)

log = logging.getLogger(__name__)

# ── module local constants ──
VERIFY_F0_HZ = 1000.0          # mid-band: ideal-feed elements transparent
VERIFY_N_HARMONICS = 9         # HD2..HD9 — matches engine chebyshev/dft
_SETTLE_CYCLES = 10            # transient settling before measurement
_MEASURE_CYCLES = 5            # .four averaging window, whole cycles
_STEPS_PER_CYCLE = 500         # .tran max step = period / this
_IDEAL_FEED_XL_FACTOR = 100.0  # choke/primary reactance × load R at f0
_POLL_INTERVAL_S = 0.2         # stop-callback poll while LTspice runs
LTSPICE_RUN_TIMEOUT_S = 120.0
SWEEP_FRACTIONS = (0.25, 0.5, 0.75, 1.0)   # amplitude-sweep drive levels
# SMPTE two-tone IMD: 60 Hz + 7 kHz mixed 4:1 by amplitude
IMD_F1_HZ = 60.0
IMD_F2_HZ = 7000.0
IMD_A1_FRACTION = 0.8          # of the total half-swing (4:1 ratio)
IMD_A2_FRACTION = 0.2
_IMD_SETTLE_CYCLES_F1 = 3      # settle in low-tone periods
_IMD_MEASURE_CYCLES_F1 = 8     # FFT window: whole 60 Hz periods
_MIN_FUND_V = 1e-6             # guard against degenerate .four output


class LtspiceVerifyError(RuntimeError):
    """LTspice missing, crashed, timed out, or produced unparsable output."""


class _Cancelled(Exception):
    """Internal: stop() raised between/inside runs."""


def ltspice_available(exe: Optional[str] = None) -> bool:
    """``exe`` — configured override (``app.json: ltspice_exe``);
    empty/None falls back to the standard install path."""
    return Path(exe or LTSPICE_EXE).exists()


def resolve_verify_workdir(config_dir: str, anchor: Path) -> Path:
    """Working directory for one verification run.

    ``config_dir`` (``app.json: ltspice_verify_dir``) empty → a fresh
    system temp directory. Otherwise a per-run ``verify_<timestamp>``
    subdirectory under it (relative paths resolve against *anchor*, the
    ``lm19_app`` root — same rule as ``measurements_dir``); same-second
    runs get the ``_1`` collision suffix.
    """
    if not config_dir.strip():
        return Path(tempfile.mkdtemp(prefix="lm19_verify_"))
    base = Path(config_dir)
    if not base.is_absolute():
        base = anchor / base
    base.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("verify_%Y%m%d_%H%M%S")
    return make_unique_path(base / stamp)


@dataclass
class VerifyRequest:
    """One verification job — mirrors the analysis the user just ran."""

    circuit: str                  # "se" | "se_xfmr" | "cf" | "pp"
    tube_type: str
    topology: str                 # fitter topology (triode/pentode/…)
    points: List[Dict]
    model_type: str = MODEL_TYPE_KOREN
    ub: float = 250.0
    ra_kohm: float = 5.0          # se/cf/se_xfmr: load; pp: Ra_aa
    ug1_bias: float = -7.0
    half_swing: float = 5.0
    ug2: Optional[float] = None   # required for pentode topologies
    ul_tap: float = 0.0           # 0 = fixed screen; else UL fraction 0..1
    amp_sweep: bool = False
    imd: bool = False
    data_label: str = ""          # which series the points came from
    # Loaded model object (TubeModelProtocol): when set, its parameters
    # are exported as-is (no refit) — the exact model the analysis used.
    model: Optional[object] = None


@dataclass
class VerifyRun:
    """One sine run at one drive level."""

    half_swing: float
    thd_pct: float
    hd_pct: Dict[int, float]
    pout_fund_mw: float
    ia_avg_ma: Optional[float]


@dataclass
class VerifyResult:
    runs: List[VerifyRun] = field(default_factory=list)
    imd: Optional[Dict] = None    # {"imd2": pct, "imd3": pct}
    basis: str = ""               # what LTspice actually simulated
    fit_rms_ma: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    workdir: str = ""             # netlists/logs kept for traceability


# ── netlist construction (pure, unit-tested) ─────────────────────────


def _sine(bias: float, amp: float, freq: float, phase_deg: float = 0.0) -> str:
    if phase_deg:
        return f"SINE({bias:g} {amp:g} {freq:g} 0 0 {phase_deg:g})"
    return f"SINE({bias:g} {amp:g} {freq:g})"


def build_verify_netlist(
    circuit: str,
    *,
    sub_path: str,
    sub_name: str,
    pentode: bool,
    ub: float,
    ra_ohm: float,
    bias_v: float,
    amp_v: float,
    ug2_v: Optional[float] = None,
    ul_tap: float = 0.0,
    f0: float = VERIFY_F0_HZ,
    two_tone: Optional[Dict] = None,
) -> str:
    """Build the verification netlist for *circuit*.

    The circuits reproduce the engine's load-line idealizations exactly:
    fixed bias from a V-source, grounded cathode (CF excepted), ideal DC
    feed. ``ul_tap > 0`` replaces the fixed screen supply with a
    behavioral source ``V = Ug2·(1−tap) + tap·V(anode)`` per tube — the
    exact ``UltralinearModelWrapper`` law. ``two_tone={"f1","a1","f2",
    "a2"}`` replaces the single sine. The output node is always ``out``.
    """
    if pentode and ug2_v is None:
        raise LtspiceVerifyError("pentode verification requires Ug2")

    lines = [f"* LM19 amplifier verification ({circuit})",
             f'.include "{sub_path}"']

    def _screen(node: str, anode: str) -> None:
        """Fixed screen supply, or the UL law tied to this tube's anode."""
        if ul_tap > 0.0:
            lines.append(
                f"B{node} {node} 0 "
                f"V={ug2_v * (1.0 - ul_tap):g}+{ul_tap:g}*V({anode})")
        else:
            lines.append(f"V{node} {node} 0 {ug2_v:g}")

    def _grid(node: str, phase: float = 0.0) -> None:
        if two_tone is not None:
            mid = f"{node}m"
            lines.append(f"Vg1{node} {node} {mid} "
                         + _sine(bias_v, two_tone['a1'], two_tone['f1'], phase))
            lines.append(f"Vg2{node} {mid} 0 "
                         + _sine(0.0, two_tone['a2'], two_tone['f2'], phase))
        else:
            lines.append(f"Vg{node} {node} 0 "
                         + _sine(bias_v, amp_v, f0, phase))

    def _tube(ref: str, anode: str, grid: str, cathode: str,
              g2: str = "g2") -> None:
        if pentode:
            lines.append(f"X{ref} {anode} {grid} {cathode} {g2} {sub_name}")
        else:
            lines.append(f"X{ref} {anode} {grid} {cathode} {sub_name}")

    ideal_l = _IDEAL_FEED_XL_FACTOR * ra_ohm / (2 * math.pi * f0)

    if circuit == CIRCUIT_SE:
        lines.append(f"Vub ub 0 {ub:g}")
        lines.append(f"Ra ub out {ra_ohm:g}")
        _tube("U", "out", "g", "0")
        _grid("g")
        if pentode:
            _screen("g2", "out")
    elif circuit == CIRCUIT_CF:
        if ul_tap > 0.0:
            # the CF anode sits at AC ground — a UL tap has nothing to
            # follow; the engine never applies UL to CF either
            raise LtspiceVerifyError("UL tap is not applicable to a "
                                     "cathode follower")
        lines.append(f"Vub ub 0 {ub:g}")
        _tube("U", "ub", "g", "out")
        lines.append(f"Rk out 0 {ra_ohm:g}")
        _grid("g")
        if pentode:
            # screen rides on the cathode: constant Vg2k, as the model sees
            lines.append(f"Vg2 g2 out {ug2_v:g}")
    elif circuit == CIRCUIT_SE_XFMR:
        # Parafeed idealization: huge choke = DC feed at Ua_q ≈ Ub, the AC
        # load line of slope Ra runs through it — the engine's exact model.
        ideal_c = _IDEAL_FEED_XL_FACTOR / (2 * math.pi * f0 * ra_ohm)
        lines.append(f"Vub ub 0 {ub:g}")
        lines.append(f"Lfeed ub a {ideal_l:g}")
        lines.append(f"Cc a out {ideal_c:g}")
        lines.append(f"Rl out 0 {ra_ohm:g}")
        _tube("U", "a", "g", "0")
        _grid("g")
        if pentode:
            _screen("g2", "a")
    elif circuit == CIRCUIT_PP:
        # Ideal center-tapped output transformer: perfectly-coupled primary
        # halves, Ra_aa across the whole primary. The class-AB kink emerges
        # from the circuit itself (same physics the joint-solve models).
        lines.append(f"Vub ub 0 {ub:g}")
        lines.append(f"L1 a1 ub {ideal_l:g}")
        lines.append(f"L2 ub a2 {ideal_l:g}")
        lines.append("K1 L1 L2 1")
        lines.append(f"Rl a1 a2 {ra_ohm:g}")
        _tube("A", "a1", "ga", "0", g2="g2a")
        _tube("B", "a2", "gb", "0", g2="g2b")
        _grid("ga")
        _grid("gb", phase=180.0)
        if pentode:
            # each screen follows ITS OWN anode in UL mode
            _screen("g2a", "a1")
            _screen("g2b", "a2")
        lines.append("Bout out 0 V=V(a1)-V(a2)")
        lines.append("Rdum out 0 1G")
    else:
        raise LtspiceVerifyError(f"unknown circuit '{circuit}'")

    if two_tone is not None:
        f_lo = two_tone["f1"]
        t_start = _IMD_SETTLE_CYCLES_F1 / f_lo
        t_end = t_start + _IMD_MEASURE_CYCLES_F1 / f_lo
        max_step = 1.0 / (two_tone["f2"] * _STEPS_PER_CYCLE)
    else:
        t_start = _SETTLE_CYCLES / f0
        t_end = t_start + _MEASURE_CYCLES / f0
        max_step = 1.0 / (f0 * _STEPS_PER_CYCLE)
        lines.append(f".four {f0:g} {VERIFY_N_HARMONICS} "
                     f"{_MEASURE_CYCLES} V(out)")

    lines.append(f".tran 0 {t_end:g} {t_start:g} {max_step:g}")
    lines.append(".options plotwinsize=0")
    lines.append(".end")
    return "\n".join(lines) + "\n"


# ── LTspice log parsing (pure, unit-tested) ──────────────────────────


_SUBCKT_RE = re.compile(r"^\.SUBCKT\s+(\S+)", re.MULTILINE | re.IGNORECASE)


def subckt_name_of(sub_path: Path) -> str:
    """The actual ``.SUBCKT`` name inside a generated ``.sub`` file.

    ``fit_and_export_spice`` names the subcircuit after the (sanitized)
    tube, not the file stem — instantiate what is really there.
    """
    match = _SUBCKT_RE.search(sub_path.read_text(encoding="utf-8",
                                                 errors="replace"))
    if match is None:
        raise LtspiceVerifyError(f"no .SUBCKT line found in {sub_path}")
    return match.group(1)


def read_ltspice_log(path: Path) -> str:
    """LTspice writes .log as UTF-16-LE on modern builds, cp1252 on old."""
    raw = path.read_bytes()
    if b"\x00" in raw[:200]:
        return raw.decode("utf-16-le", errors="replace")
    return raw.decode("cp1252", errors="replace")


_FOUR_ROW_RE = re.compile(
    r"^\s*(\d+)\s+[\d.eE+-]+\s+([\d.eE+-]+)\s+([\d.eE+-]+)", re.MULTILINE)
_THD_RE = re.compile(r"Total Harmonic Distortion:\s*([\d.eE+-]+)\s*%")


# .four is SPICE's DFT of the simulated waveform — "Power Electronics
# News — QSPICE FFT Analysis" and "AES E-Library — Distortion Analysis
# Using SPICE", see SOURCES_INDEX.md
def parse_four_table(log_text: str) -> Dict:
    """Extract fundamental amplitude, THD%% and HDn%% from a .four block."""
    thd_m = _THD_RE.search(log_text)
    rows = _FOUR_ROW_RE.findall(log_text)
    fund = None
    hd: Dict[int, float] = {}
    for num, component, normalized in rows:
        n = int(num)
        if n == 1:
            fund = float(component)
        elif 2 <= n <= VERIFY_N_HARMONICS:
            hd[n] = float(normalized) * 100.0
    if fund is None or fund < _MIN_FUND_V or thd_m is None:
        raise LtspiceVerifyError(
            "no usable Fourier table in LTspice log (simulation failed?)")
    return {"fund_v": fund, "thd_pct": float(thd_m.group(1)), "hd_pct": hd}


def tube_current_avg_ma(raw: Dict) -> Optional[float]:
    """Time-weighted average anode current of the first tube, mA.

    LTspice ``.meas`` over subcircuit pin currents fails on current
    builds ("Measurement FAILed"), so the average is integrated from the
    adaptive-step ``.raw`` waveform instead.
    """
    for i, var in enumerate(raw["variables"]):
        low = var.lower()
        if low.startswith("ix(x") and low.endswith(":a)"):
            t = np.abs(raw["data"][:, 0])
            ia = raw["data"][:, i]
            span = t[-1] - t[0]
            if span <= 0:
                return None
            return abs(float(np.trapezoid(ia, t))) / span * 1000.0
    return None


# ── two-tone IMD from the .raw waveform (pure, unit-tested) ──────────


def imd_from_signal(t: np.ndarray, v: np.ndarray,
                    f1: float = IMD_F1_HZ, f2: float = IMD_F2_HZ) -> Dict:
    """SMPTE IMD from a time-domain two-tone response.

    Resamples to a uniform grid (LTspice steps are adaptive), FFTs whole
    ``f1`` periods and reads the sideband amplitudes around ``f2``:
    IMD2 = mean(f2±f1)/A(f2), IMD3 = mean(f2±2·f1)/A(f2), in %.
    """
    duration = t[-1] - t[0]
    n_periods = int(duration * f1)
    if n_periods < 2:
        raise LtspiceVerifyError("IMD window shorter than 2 low-tone periods")
    window = n_periods / f1
    n_samples = 2 ** int(math.ceil(math.log2(window * f2 * 16)))
    tu = np.linspace(t[-1] - window, t[-1], n_samples, endpoint=False)
    vu = np.interp(tu, t, v)
    # Hann window: the grid is whole f1-periods but generally NOT whole
    # f2-periods — rectangular-window leakage would smear the carrier
    # into the sidebands and overstate IMD.
    hann = np.hanning(n_samples)
    spec = np.abs(np.fft.rfft((vu - vu.mean()) * hann)) / (hann.sum() / 2)
    df = 1.0 / window

    def _amp(freq: float) -> float:
        idx = int(round(freq / df))
        lo, hi = max(idx - 2, 0), min(idx + 3, len(spec))
        return float(spec[lo:hi].max())

    a2 = _amp(f2)
    if a2 <= 0:
        raise LtspiceVerifyError("no carrier at f2 in IMD response")
    imd2 = (_amp(f2 - f1) + _amp(f2 + f1)) / 2.0 / a2 * 100.0
    imd3 = (_amp(f2 - 2 * f1) + _amp(f2 + 2 * f1)) / 2.0 / a2 * 100.0
    return {"imd2": imd2, "imd3": imd3}


# ── batch runner ─────────────────────────────────────────────────────


def _cleanup_raw(cir_path: Path) -> None:
    """Delete the parsed waveform files: an IMD run's ``.raw`` weighs
    ~29 MB and %TEMP% would silently accumulate one per verification.
    The netlist, ``.log`` and ``.sub`` (a few KB) stay for traceability."""
    for suffix in (".raw", ".op.raw"):
        target = cir_path.with_suffix(suffix)
        try:
            target.unlink(missing_ok=True)
        except OSError:
            log.warning("could not delete %s", target, exc_info=True)


def _run_ltspice_batch(cir_path: Path, *, exe: str,
                       stop: Optional[Callable[[], bool]],
                       timeout_s: float = LTSPICE_RUN_TIMEOUT_S) -> None:
    proc = subprocess.Popen(
        [exe, "-b", str(cir_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + timeout_s
    try:
        while proc.poll() is None:
            if stop is not None and stop():
                raise _Cancelled()
            if time.monotonic() > deadline:
                raise LtspiceVerifyError(
                    f"LTspice timed out after {timeout_s:g}s on {cir_path.name}")
            time.sleep(_POLL_INTERVAL_S)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(5)
    if proc.returncode != 0:
        raise LtspiceVerifyError(
            f"LTspice exited with code {proc.returncode} on {cir_path.name}")


def run_verification(
    req: VerifyRequest,
    *,
    workdir: str,
    ltspice_exe: Optional[str] = None,
    stop: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> VerifyResult:
    """Fit → netlist → LTspice → parsed numbers, per requested run set.

    Cancellation (``stop``) is polled between runs and while LTspice is
    executing; a cancelled job returns the completed runs with a
    ``"cancelled"`` warning instead of raising (partial result stays
    visible).
    """
    exe = ltspice_exe or LTSPICE_EXE
    if not ltspice_available(exe):
        raise LtspiceVerifyError(f"LTspice not found at {exe}")

    from lm19.spice_export import export_spice_from_model, fit_and_export_spice

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    result = VerifyResult(workdir=str(wd))

    if progress:
        progress("fit")
    sub_path = wd / "verify_model.sub"
    if req.model is not None:
        # the exact model the analysis used — no refit
        fit = export_spice_from_model(str(sub_path), req.model,
                                      tube_type=req.tube_type,)
        result.basis = (
            f"LTspice .tran/.four @ {VERIFY_F0_HZ:g} Hz, loaded "
            f"{getattr(req.model, 'model_type', '?')} model")
    else:
        fit = fit_and_export_spice(str(sub_path), req.tube_type, req.points,
                                   topology=req.topology,
                                   model_type=req.model_type)
        result.fit_rms_ma = getattr(fit, "rms_error", None)
        result.basis = (f"LTspice .tran/.four @ {VERIFY_F0_HZ:g} Hz, "
                        f"fresh {req.model_type} fit")
    for w in getattr(fit, "warnings", None) or []:
        result.warnings.append(str(w))
    if req.data_label:
        result.basis += f" [{req.data_label}]"

    pentode = req.topology != TOPOLOGY_TRIODE
    ra_ohm = req.ra_kohm * 1000.0
    amps = ([f * req.half_swing for f in SWEEP_FRACTIONS]
            if req.amp_sweep else [req.half_swing])

    sub_name = subckt_name_of(sub_path)

    def _netlist(**kw) -> str:
        # bare filename: LTspice resolves .include against the netlist's
        # directory, so the whole folder stays relocatable for manual
        # re-runs (an absolute path would break on move/copy)
        return build_verify_netlist(
            req.circuit, sub_path=sub_path.name, sub_name=sub_name,
            pentode=pentode, ub=req.ub, ra_ohm=ra_ohm,
            bias_v=req.ug1_bias, ug2_v=req.ug2, ul_tap=req.ul_tap, **kw)

    try:
        for i, amp in enumerate(amps):
            if stop is not None and stop():
                raise _Cancelled()
            if progress:
                progress(f"run {i + 1}/{len(amps)}")
            cir = wd / f"verify_run{i}.cir"
            cir.write_text(_netlist(amp_v=amp), encoding="ascii")
            _run_ltspice_batch(cir, exe=exe, stop=stop)
            four = parse_four_table(read_ltspice_log(cir.with_suffix(".log")))
            try:
                ia_avg = tube_current_avg_ma(
                    parse_raw(str(cir.with_suffix(".raw"))))
            except (OSError, ValueError, KeyError) as exc:
                log.warning("verification: .raw current read failed: %s", exc)
                ia_avg = None
            _cleanup_raw(cir)
            # power reference: full load for se/cf/se_xfmr, Ra_aa for pp
            pout_mw = four["fund_v"] ** 2 / (2.0 * ra_ohm) * 1000.0
            result.runs.append(VerifyRun(
                half_swing=amp,
                thd_pct=four["thd_pct"],
                hd_pct=four["hd_pct"],
                pout_fund_mw=pout_mw,
                ia_avg_ma=ia_avg,
            ))

        if req.imd:
            if stop is not None and stop():
                raise _Cancelled()
            if progress:
                progress("imd")
            cir = wd / "verify_imd.cir"
            two_tone = {"f1": IMD_F1_HZ, "a1": IMD_A1_FRACTION * req.half_swing,
                        "f2": IMD_F2_HZ, "a2": IMD_A2_FRACTION * req.half_swing}
            cir.write_text(_netlist(amp_v=0.0, two_tone=two_tone),
                           encoding="ascii")
            _run_ltspice_batch(cir, exe=exe, stop=stop)
            raw = parse_raw(str(cir.with_suffix(".raw")))
            t = raw["data"][:, 0]
            v = get_variable(raw, "V(out)")
            result.imd = imd_from_signal(np.abs(t), v)
            _cleanup_raw(cir)
    except _Cancelled:
        log.warning("LTspice verification cancelled by user (%d runs done)",
                    len(result.runs))
        result.warnings.append("cancelled")

    return result
