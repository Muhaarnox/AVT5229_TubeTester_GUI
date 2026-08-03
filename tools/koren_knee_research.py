# -*- coding: utf-8 -*-
"""Research tool: EL84 Koren-knee quality (see docs/KOREN_KNEE_RESEARCH.md).

Question: EL84 PP gave 11.8 W against the "17 W datasheet" figure —
the Koren model knee was suspected of understating power.

Primary source: Philips EL84 datasheet, January 1969
(external_sources/lamp_datasheets/EL84_philips.pdf, pages 4 and 8):
  - Class B  PP: Va=Vg2=300, Vg1=-14.7 fixed, Ra-aa=8k, Vi=10 Vrms
                 -> Ia 2x7.5 -> 2x46 mA, Wo=17 W @ dtot=4%
  - Class AB PP: Va=Vg2=300, Rk=130 common, Ra-aa=8k, Vi=10 Vrms
                 -> Ia 2x36 -> 2x46 mA, Wo=17 W @ dtot=4%
    (bias slides: Vk = 130*(2*Ia+2*Ig2) ~ 14.8 V at full signal)
  - Ia(Va) family at Vg2=300 (p.8 bottom) — knee anchors read off
    the chart manually, +/-5-8 mA (_KNEE_ANCHORS_MA below).

Findings (details in docs/KOREN_KNEE_RESEARCH.md):
  1. "11.8 vs 17 W" compared different things: peak-based pout_mw
     instead of the P1 fundamental, +/-9 V drive instead of the
     datasheet +/-14.14 V, and a "~9-10%" THD reference instead of
     the tabulated 4%.
  2. At the true conditions the reference Koren gives 15.1 W @ 5.2%
     — the real knee deficit is ~11% of power, not ~30%.
  3. The deficit is fixable WITHIN the Koren form: the published
     kvb=48 is ~3x too high; kvb~16-24 (+kg1 rescale preserving the
     grid region) lands both the knee anchors and the 17 W @ 4% row.

Run (from lm19_app/): py tools/koren_knee_research.py [--section NAME]
Sections: ref, drive, kvb, data, fit, others (default: all).
Read-only: prints tables, writes nothing.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from lm19.amplifier import PushPullLoadLine, compute_distortion_dft_pp
from lm19.tube_model_base import model_ia_array
from lm19.tube_sim import load_model

# ── module local constants ──
# Knee anchors: Philips p.8, Vg2=300 V family, read off the chart
# (150-400 dpi render), tolerance +/-5-8 mA.
_KNEE_ANCHORS_MA: List[Tuple[float, float, float]] = [
    # (ug1 V, ua V, Ia mA per datasheet)
    (0.0, 40.0, 143.0),
    (0.0, 60.0, 163.0),
    (0.0, 100.0, 180.0),
    (-2.0, 40.0, 107.0),
    (-2.0, 60.0, 124.0),
    (-2.0, 80.0, 133.0),
    (-6.0, 60.0, 78.0),
]
# Grid region: the class-AB PP operating point from the p.4 table
# (Iq=36 mA at -11 V) — the published Koren set hits it exactly; any
# modification must preserve it.
_GRID_ANCHOR = (300.0, -11.0, 300.0, 36.1)  # (ua, ug1, ug2, Ia mA)
# Datasheet PP row (class B, p.4): target for the integral check.
_DS_POUT_W = 17.0
_DS_DTOT_PCT = 4.0
_DS_IA_FULL_MA = 46.0
_DS_BIAS_B = -14.7
_VI_PEAK_V = 10.0 * math.sqrt(2.0)  # Vi=10 Vrms
_PP_LL = dict(ub=300.0, ra_aa=8.0, ra_dc=0.1)
_DATA_DIR = _PROJECT_ROOT / "tests" / "spice_test_data" / "converted"
_EL84_REAL = (
    "pentode_EL84_ER_L1_real.json",
    "pentode_EL84_ER_L2_real.json",
    "pentode_EL84_SOVTEK_L1_real.json",
    "pentode_EL84_SOVTEK_L2_real.json",
    "pentode_EL84_hw_protected_real.json",
)


def _pp_row(model, bias: float, swing: float) -> Dict[str, float]:
    ll = PushPullLoadLine(**_PP_LL)
    d = compute_distortion_dft_pp(model, ll, ug1_bias=bias,
                                  half_swing=swing, ug2=300.0)
    if d is None:
        return {}
    p1_w = d["pout_fund_mw"] / 1000.0
    ia_avg = (d["pa_avg_mw"] / 1000.0 + p1_w / 2.0) / _PP_LL["ub"] * 1000.0
    return dict(pout_w=d["pout_mw"] / 1000.0, p1_w=p1_w, thd=d["thd"],
                iq=d["iq_per_tube"], ia_avg=ia_avg)


def _knee_errors(model) -> List[float]:
    errs = []
    for g1, ua, want in _KNEE_ANCHORS_MA:
        got = float(model_ia_array(model, np.array([ua]),
                                   np.array([g1]), 300.0)[0])
        errs.append(100.0 * (got - want) / want)
    return errs


def section_ref() -> None:
    """Reference Koren: grid anchor + true datasheet conditions."""
    el84 = load_model("EL84")
    print("=== ref: published Koren set vs Philips p.4 ===")
    for g1, want in ((-11.0, 36.0), (_DS_BIAS_B, 7.5)):
        ia = float(model_ia_array(el84, np.array([300.0]),
                                  np.array([g1]), 300.0)[0])
        print(f"  Ia(300,{g1:6.1f},300) = {ia:6.1f} mA  (datasheet {want})")
    r = _pp_row(el84, _DS_BIAS_B, _VI_PEAK_V)
    print(f"  class B full drive: P1={r['p1_w']:.2f} W (ds {_DS_POUT_W}), "
          f"THD={r['thd']:.2f}% (ds {_DS_DTOT_PCT}), "
          f"Ia_avg={r['ia_avg']:.1f} mA (ds {_DS_IA_FULL_MA})")
    errs = _knee_errors(el84)
    rms = math.sqrt(sum(e * e for e in errs) / len(errs))
    print(f"  knee anchors err%: "
          + " ".join(f"{e:+.0f}" for e in errs) + f"  rms={rms:.1f}%")


def section_drive() -> None:
    """Decompose "11.8 vs 17 W": metric + drive (old pin conditions)."""
    el84 = load_model("EL84")
    print("=== drive/metric decomposition (bias -11, Ra-aa=8k) ===")
    print(f"{'swing V':>8} {'pout_mw W':>10} {'P1 W':>6} {'THD %':>6}")
    for sw in (9.0, 11.0, _VI_PEAK_V):
        r = _pp_row(el84, -11.0, sw)
        print(f"{sw:8.2f} {r['pout_w']:10.2f} {r['p1_w']:6.2f} "
              f"{r['thd']:6.2f}")
    print("  (the old pin compared pout_mw@+/-9V=11.8 with the 17 W"
          " row; P1@+/-14.14V is 20.8 but THD 17.5% w/o bias slide)")


def section_kvb() -> None:
    """kvb knife: knee sharpness within the Koren form, kg1 rescale."""
    print("=== kvb experiment (kg1 rescaled to keep grid anchor) ===")
    print(f"{'kvb':>5} {'kg1':>6} {'knee rms%':>9} {'P1 W':>6} {'THD%':>5} "
          f"{'Ia_avg':>6}")
    ua_a, g1_a, g2_a, want_a = _GRID_ANCHOR
    for kvb in (48.0, 24.0, 20.0, 16.0, 12.0):
        m = load_model("EL84")
        k0 = m.koren
        m.koren = dataclasses.replace(k0, kvb=kvb)
        got = float(model_ia_array(m, np.array([ua_a]),
                                   np.array([g1_a]), g2_a)[0])
        m.koren = dataclasses.replace(k0, kvb=kvb,
                                      kg1=k0.kg1 * got / want_a)
        errs = _knee_errors(m)
        rms = math.sqrt(sum(e * e for e in errs) / len(errs))
        r = _pp_row(m, _DS_BIAS_B, _VI_PEAK_V)
        print(f"{kvb:5.0f} {m.koren.kg1:6.0f} {rms:9.1f} {r['p1_w']:6.2f} "
              f"{r['thd']:5.2f} {r['ia_avg']:6.1f}")
    print(f"  target: knee rms -> 0, P1={_DS_POUT_W}, THD={_DS_DTOT_PCT}, "
          f"Ia_avg={_DS_IA_FULL_MA}")


def _load_points(name: str) -> List[dict]:
    d = json.loads((_DATA_DIR / name).read_text(encoding="utf-8"))
    return d["points"] if isinstance(d, dict) else d


def section_data() -> None:
    """Knee-region coverage by the real LM19 EL84 scans."""
    print("=== real scan knee coverage ===")
    for name in _EL84_REAL:
        pts = _load_points(name)
        ug2s = sorted({round(p.get("ug2", 0.0)) for p in pts})
        ia_max = max(p["ia"] for p in pts)
        knee = [p for p in pts if p["ua"] <= 80 and p["ia"] >= 60]
        print(f"  {name}: n={len(pts)} ug2_max={max(ug2s)} "
              f"ia_max={ia_max:.0f} knee_pts={len(knee)}")
    print("  (scans stay below Ug2=250 and Ik~65 mA — the knee at "
          "Vg2=300 is an extrapolation for ANY fit of this data)")


def section_fit() -> None:
    """fit_koren / fit_dempwolf on a real scan: kvb, in-sample knee."""
    from lm19.tube_sim import fit_koren
    from lm19.dempwolf import fit_dempwolf

    name = "pentode_EL84_ER_L2_real.json"
    pts = _load_points(name)
    print(f"=== fitters on {name} (in-sample knee) ===")
    knee_pts = [p for p in pts if p["ua"] <= 80 and p["ia"] >= 40]
    print(f"  knee sample: {len(knee_pts)} pts (ua<=80, ia>=40)")

    results = {}
    kr = fit_koren(pts, topology="pentode")
    results["koren"] = kr
    print(f"  fit_koren: kvb={kr.model.koren.kvb:.1f} "
          f"rms={kr.rms_error:.2f} mA")
    dr = fit_dempwolf(pts, topology="pentode")
    results["dempwolf"] = dr
    print(f"  fit_dempwolf: rms={dr.rms_error:.2f} mA")

    print(f"  {'point':>28} {'meas':>6} {'koren':>6} {'dempw':>6}")
    for p in sorted(knee_pts, key=lambda q: -q["ia"])[:8]:
        row = []
        for key in ("koren", "dempwolf"):
            m = results[key].model
            got = float(model_ia_array(
                m, np.array([p["ua"]]), np.array([p["ug1"]]),
                p.get("ug2", 0.0))[0])
            row.append(got)
        print(f"  ua={p['ua']:5.1f} ug1={p['ug1']:5.1f} "
              f"ug2={p.get('ug2', 0):5.0f} {p['ia']:6.1f} "
              f"{row[0]:6.1f} {row[1]:6.1f}")

    # mean in-sample bias over the knee points
    for key in ("koren", "dempwolf"):
        m = results[key].model
        errs = []
        for p in knee_pts:
            got = float(model_ia_array(
                m, np.array([p["ua"]]), np.array([p["ug1"]]),
                p.get("ug2", 0.0))[0])
            errs.append(100.0 * (got - p["ia"]) / p["ia"])
        mean = sum(errs) / len(errs)
        rms = math.sqrt(sum(e * e for e in errs) / len(errs))
        print(f"  {key}: knee in-sample bias {mean:+.1f}% rms {rms:.1f}%")


# -- verification of the other pentode references ---------------------
# Anchors from primary sources (external_sources/lamp_datasheets/):
#   EL34_philips.pdf (Jan-1969): p.2 class A table (printed numbers),
#     p.6 Vg2=250 family (chart reading +/-8-10 mA).
#   6L6_tungsol.pdf (Tung-Sol 1960): p.2-3 tables (printed numbers
#     ONLY — the 14 merging curves on p.5 cannot be read reliably).
#   KT88_jj.pdf (JJ, modern production): p.1 typical characteristics,
#     p.2 Ug2=300 family (+/-8-15 mA). NB: Koren fitted the GEC era;
#     JJ specimens may differ from the 1957 bogey.
_OTHER_TUBES: Dict[str, Dict] = {
    "EL34": {
        # (ua, ug1, ug2, Ia mA) — printed table points
        "grid_anchors": [(250.0, -13.5, 265.0, 100.0),
                         (250.0, -14.5, 245.0, 70.0)],
        # (ug1, ua, Ia mA) at the ug2 below
        "knee_ug2": 250.0,
        "knee_anchors": [(0.0, 40.0, 172.0), (0.0, 60.0, 205.0),
                         (0.0, 100.0, 247.0), (-5.0, 60.0, 138.0),
                         (-5.0, 100.0, 183.0), (-10.0, 100.0, 112.0)],
        # SE class A row p.2: Vb=265 (ra_dc drops ~15 V at 100 mA),
        # Va=250, Ra~=2k, Vi=8.7 Vrms -> Wo=11 W @ 10%
        "se_row": dict(ub=265.0, ra_dc=0.15, ra_ac=2.0, bias=-13.5,
                       swing=8.7 * math.sqrt(2.0), ug2=265.0,
                       wo_w=11.0, dtot_pct=10.0),
    },
    "6L6": {
        "grid_anchors": [(250.0, -14.0, 250.0, 72.0),
                         (300.0, -12.5, 200.0, 48.0),
                         (350.0, -18.0, 250.0, 54.0)],
        "knee_ug2": None,
        "knee_anchors": [],
        # PP class AB1 row (Tung-Sol p.3): 450/400/-37, Raa=5.6k,
        # drive 70 V peak grid-to-grid (+/-35 per grid) -> 55 W @ 1.8%
        "pp_row": dict(ub=450.0, ug2=400.0, bias=-37.0, swing=35.0,
                       raa=5.6, wo_w=55.0, dtot_pct=1.8,
                       iq_ma=58.0),
    },
    "KT88": {
        "grid_anchors": [(250.0, -15.0, 250.0, 140.0)],
        "knee_ug2": 300.0,
        "knee_anchors": [(0.0, 60.0, 300.0), (-5.0, 100.0, 312.0),
                         (-5.0, 300.0, 345.0), (-10.0, 100.0, 245.0),
                         (-15.0, 100.0, 185.0), (-20.0, 100.0, 140.0),
                         (-25.0, 100.0, 105.0), (-30.0, 100.0, 78.0)],
    },
}


def section_others() -> None:
    """Verify the EL34/6L6/KT88 references against datasheet anchors."""
    from lm19.amplifier import (TransformerLoadLine,
                                compute_distortion_dft)

    for name, spec in _OTHER_TUBES.items():
        m = load_model(name)
        print(f"=== {name} (koren: {m.koren}) ===")
        for ua, g1, g2, want in spec["grid_anchors"]:
            got = float(model_ia_array(m, np.array([ua]),
                                       np.array([g1]), g2)[0])
            print(f"  grid Ia({ua:.0f},{g1:5.1f},{g2:.0f}) = {got:6.1f} "
                  f"(ds {want:5.1f})  {100*(got-want)/want:+5.1f}%")
        if spec["knee_anchors"]:
            errs = []
            for g1, ua, want in spec["knee_anchors"]:
                got = float(model_ia_array(m, np.array([ua]),
                                           np.array([g1]),
                                           spec["knee_ug2"])[0])
                errs.append(100.0 * (got - want) / want)
            rms = math.sqrt(sum(e * e for e in errs) / len(errs))
            print(f"  knee (ug2={spec['knee_ug2']:.0f}): "
                  + " ".join(f"{e:+.0f}" for e in errs)
                  + f"  rms={rms:.1f}%")
        row = spec.get("se_row")
        if row:
            ll = TransformerLoadLine(ub=row["ub"], ra_dc=row["ra_dc"],
                                     ra_ac=row["ra_ac"])
            d = compute_distortion_dft(m, ll, ug1_bias=row["bias"],
                                       half_swing=row["swing"],
                                       ug2=row["ug2"], ub=row["ub"])
            if d:
                print(f"  SE row: pout={d['pout_mw']/1000:.2f} W "
                      f"THD={d['thd']:.1f}% Iq={d['ia_0']:.0f} mA "
                      f"(ds {row['wo_w']} W @ {row['dtot_pct']}%)")
        row = spec.get("pp_row")
        if row:
            ll = PushPullLoadLine(ub=row["ub"], ra_aa=row["raa"],
                                  ra_dc=0.1)
            d = compute_distortion_dft_pp(m, ll, ug1_bias=row["bias"],
                                          half_swing=row["swing"],
                                          ug2=row["ug2"])
            if d:
                p1 = d["pout_fund_mw"] / 1000.0
                print(f"  PP row: P1={p1:.1f} W (peak-based "
                      f"{d['pout_mw']/1000:.1f}) THD={d['thd']:.1f}% "
                      f"Iq={d['iq_per_tube']:.0f} mA "
                      f"(ds {row['wo_w']} W @ {row['dtot_pct']}%, "
                      f"Iq {row['iq_ma']})")
        print()


_SECTIONS = dict(ref=section_ref, drive=section_drive, kvb=section_kvb,
                 data=section_data, fit=section_fit, others=section_others)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--section", choices=sorted(_SECTIONS),
                    help="run one section (default: all)")
    args = ap.parse_args(argv)
    names = [args.section] if args.section else list(_SECTIONS)
    for n in names:
        _SECTIONS[n]()
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
