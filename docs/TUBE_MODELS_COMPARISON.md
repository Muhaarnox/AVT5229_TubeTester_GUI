# Vacuum Tube SPICE Models — Comparative Reference

Overview of published vacuum tube models relevant to the LM19 project.
Based on Koren (1996-2003), Reefman/Derk (2016), and Dempwolf (2011).

**Sources:**
- N. Koren, "Improved vacuum tube models for SPICE", *Glass Audio*, 1996
- D. Reefman, "Spice models for vacuum tubes using the uTracer", 2016
  (dos4ever.com/uTracer3/Theory.pdf)
- S. Dempwolf et al., "A physically-motivated triode model for circuit simulations", 2011
  (DAFx-11). The original covers triodes; the extension to pentodes/beam
  tetrodes (v2) is our own work.
- utMax — GUI companion for uTracer (bmamps.com)
- White Cottage valve modelling reference (whitecottage.org.uk)

---

## Table of Contents

1. [Quick Comparison Table](#1-quick-comparison-table)
2. [Koren Model](#2-koren-model)
3. [Reefman Derk Model](#3-reefman-derk-model-true-pentodes)
4. [Reefman DerkE Model](#4-reefman-derke-model-beam-tetrodes)
5. [Dempwolf Extended v2](#5-dempwolf-extended-v2)
6. [Secondary Emission](#6-secondary-emission)
7. [Variable-Mu Pentodes](#7-variable-mu-pentodes)
8. [Hexodes and Heptodes](#8-hexodes-and-heptodes)
9. [Parameter Comparison](#9-parameter-comparison)
10. [Model Selection Guide](#10-model-selection-guide)
11. [Implementation Status in LM19](#11-implementation-status-in-lm19)

---

## 1. Quick Comparison Table

| Feature | Koren | Derk | DerkE | Dempwolf v2 |
|---|---|---|---|---|
| **Triode Ia** | Yes | (inherited) | (inherited) | Yes |
| **Pentode Ia** | arctan(Va/Kvb) | Physical split 1/(1+βVa) | Physical split exp(-(βVa)^1.5) | Softplus + arctan-split |
| **Ig2 depends on Va** | No | Yes | Yes | Yes |
| **Constant space-charge current** | No | Yes (core principle) | Yes | No (fitted separately) |
| **Secondary emission** | No | Yes (Psec tanh) | Yes (Psec tanh) | Yes (σ·x·exp, beam) |
| **Variable-mu** | No | Yes (two Koren currents) | Yes | Yes (dual µ) |
| **Hexode/Heptode** | Partial | Yes | — | No |
| **Triode-strapped pentode** | Poor | Good (by design) | Good | Good |
| **Parameters (pentode)** | 6 | 8-10 | 8-10 | 12 |
| **Parameters (with sec. em.)** | — | 13-15 | 13-15 | 17 |
| **SPICE subcircuit** | Yes | Yes (ExtractModel) | Yes (ExtractModel) | Yes (`spice_export.py`) |
| **Implemented in LM19** | Yes | Yes (`reefman.py`) | Yes (`reefman.py`) | Yes |

---

## 2. Koren Model

### 2.1 Triode

```
E1 = (Va / Kp) · ln(1 + exp(Kp · (1/µ + Vg1 / √(Kvb + Va²))))

Ia = (PWR(E1,Ex) + PWRS(E1,Ex)) / Kg1 = 2·E1^Ex / Kg1  (for E1 > 0)
```

**Parameters (5):** µ, Ex, Kg1, Kp, Kvb

| Parameter | Meaning | Typical values |
|---|---|---|
| µ | Amplification factor | 10–100 |
| Ex | Langmuir–Child exponent | 1.2–1.8 |
| Kg1 | Grid perveance | 100–5000 |
| Kp | Current compression factor | 20–600 |
| Kvb | Knee voltage / island effect | 1–300 |

### 2.2 Pentode

```
E1 = (Vg2 / Kp) · ln(1 + exp(Kp · (1/µ + Vg1/Vg2)))

Ia = 2·E1^Ex / Kg1 · arctan(Va / Kvb)     (for E1 > 0)

Ig2 = (Vg1 + Vg2/µ)^(3/2) / Kg2     for Vg1 + Vg2/µ > 0   [original Koren]

Note: the LM19 implementation uses the exponent Ex instead of 3/2.
```

**Parameters (6):** µ, Ex, Kg1, Kp, Kvb, Kg2

**Key problems of the Koren pentode (noted by Reefman):**
1. Ig2 does not depend on Va — fails to describe current redistribution in the knee region
2. A triode-strapped pentode does NOT behave like the triode model of the same tube
3. Kp and Kg1 are correlated (our own Hessian analysis) — mutually interchangeable
4. Kvb has little effect on the triode fit (small eigenvector)

---

## 3. Reefman Derk Model (True Pentodes)

A model for **true pentodes** (EF86, PF86, EL34, etc.) with a wire
suppressor grid g3.

### 3.1 Koren Current (shared base for Derk/DerkE)

Reefman defines the "Koren current" — the cathode current without scaling
coefficients:

```
E1,p = (Vg2 / Kp) · ln(1 + exp(Kp · (1/µ + Vg1 / √(Kvb + Vg2²))))

Ip,Koren = ½ · E1,p^Ex · (1 + sgn(E1,p))
```

**Important:** unlike the original Koren, `√(Kvb + Vg2²)` is used here
instead of the simplified `Vg2`. This ensures correct behavior in
triode-strapped operation (Va = Vg2).

### 3.2 Constant Space-Charge Current

The core physical principle: **the total cathode current Ic = Ia + Ig2 does
not depend on Va** at fixed Vg2 (for small-signal pentodes). For power
pentodes/tetrodes there is a weak linear dependence:

```
Ic(Va) = Ia(Va) + Ig2(Va) = (Ip,Koren / Kg1) · (1 + A·Va − α/(1 + β·Va))
```

where A is the slope of the linear space-charge growth (usually ≈ 0).

### 3.3 Screen and Anode Currents (Derk)

For a **true pentode**, the screen current falls off inversely with Va:

```
Ig2(Va) = (Ip,Koren / Kg2) · (1 + αs / (1 + β·Va))
```

From the constant space-charge principle, the anode current is:

```
Ia(Va) = Ip,Koren · (1/Kg1 - 1/Kg2 + A·Va/Kg1 − (1/(1+β·Va)) · (α/Kg1 + αs/Kg2))
```

The **boundary condition** Ia(Va=0) = 0 gives:

```
α = 1 − (Kg1/Kg2) · (1 + αs)
```

That is, **α is not a free parameter** — it is computed from Kg1, Kg2, αs.

### 3.4 Derk Parameters

| Parameter | Meaning | Typical values |
|---|---|---|
| µ | Amplification factor | 10–100 |
| Ex | Current exponent | 1.2–1.5 |
| Kg1 | Anode perveance | 100–5000 |
| Kg2 | Screen grid perveance | 300–4000 |
| Kp | Compression factor | 0.5–2.0 (!) |
| Kvb | Island effect | 300–800 |
| A | Space-charge slope | 0–0.1 |
| β | Knee steepness (onset) | 0.01–0.5 |
| αs | Screen current scale | 0.1–1.0 |

---

## 4. Reefman DerkE Model (Beam Tetrodes)

A model for **beam tetrodes** (6L6, KT88, EL500) and pentodes with beam-like
behavior (EF80). Differs from Derk in the shape of the scaling function.

### 4.1 Screen and Anode Currents (DerkE)

Screen current:

```
Ig2(Va) = (Ip,Koren / Kg2) · (1 + αs · exp(−(β·Va)^(3/2)))
```

Anode current:

```
Ia(Va) = Ip,Koren · (1/Kg1 - 1/Kg2 + A·Va/Kg1 − exp(−(β·Va)^(3/2)) · (α/Kg1 + αs/Kg2))
```

The same boundary condition: `α = 1 − (Kg1/Kg2)·(1 + αs)`.

### 4.2 Physical Difference: Derk vs DerkE

| Characteristic | Derk: 1/(1+βVa) | DerkE: exp(-(βVa)^1.5) |
|---|---|---|
| Knee shape | Smooth, soft | Sharp, abrupt |
| Typical of | EF86, PF86, EL34 | EF80, 6L6, KT88, EL500 |
| Physical origin | Suppressor grid g3 | Beam-forming plates |

Reefman notes: the EL34 is a **true pentode** (Derk), not a beam tetrode.
A DerkE fit for the EL34 gives worse results.

---

## 5. Dempwolf Extended v2

Our own implementation based on the work of S. Dempwolf et al. (DAFx-11).
Details in [DEMPWOLF_EXTENDED_MODEL.md](DEMPWOLF_EXTENDED_MODEL.md).

### 5.1 Triode

```
Vg_eff = Vg1 · Va / √(Kvb_t + Va²)   (island effect, as in Koren)
arg = C · (Va/µ + Vg_eff)
Softplus: sp = ln(1 + exp(arg)) / C

Ik = G · sp^γ                         (cathode current)
Ia = Ik − Igk                         (anode = cathode − grid)
Igk = Gg · (ln(1+exp(Cg·Vgk))/Cg)^ξ  (grid current for Vg1 > 0)
```

### 5.2 Pentode (Kp-normalized)

```
V_eff = 1/µ + Vg1/Vg2
Softplus: sp = (Vg2/C) · ln(1 + exp(C · V_eff))    ← Kp normalization

Ik = G · sp^γ · (1 + A·Va)       (cathode current with Durchgriff, usually A ≈ 0)
```

Anode/screen current split via arctan-split (analogous to the Koren knee):

```
Kvb_eff = max(Kvb + Kvb1 · max(Vg2/µ + Vg1, 0), Kvb_min)
α = (1 − fg2) · (2/π) · arctan((Va / Kvb_eff)^Kn)

Ia  = (Ik − Igk) · α
Ig2 = (Ik − Igk) · (1 − α)
```

For beam tetrodes there is an additional secondary-emission term
(σ, Ks, λ, ν, w), see §6.2.

### 5.3 Comparison with Derk

| | Derk/DerkE | Dempwolf v2 |
|---|---|---|
| Knee | g(Va) — analytic | arctan((Va/Kvb)^Kn) |
| Ia(Va=0) = 0 | Ensured by the α constraint | Ensured by arctan→0 |
| Ia + Ig2 = const(Va) | Yes (by design) | Approximately |
| Triode-strapped | Correct (Kvb + Vg2²) | Correct (Kp-norm) |

---

## 6. Secondary Emission

### 6.1 Reefman (Derk/DerkE)

Added to the screen current:

```
Psec = S · Va · (1 + tanh(−ap · (Va − Vco)))

Vco = Vg2/λ − ν·Vg1 − ω
```

| Parameter | Meaning |
|---|---|
| S | Secondary emission intensity |
| ap | Crossover zone width |
| λ | Vco dependence on Vg2 |
| ν | Vco dependence on Vg1 |
| ω | Vco offset constant |

**Physics:** at Va < Vco, secondary electrons emitted from the anode are
attracted to the screen (Ig2 rises, Ia falls). At Va > Vco they return to
the anode. The linear S·Va dependence reflects the growing energy of the
primary electrons.

With secondary emission, the Derk formulas become:

```
Ig2(Va) = (Ip,K / Kg2) · (1 + αs/(1+βVa) + Psec)

Ia(Va) = Ip,K · (1/Kg1 - 1/Kg2 + A·Va/Kg1 − Psec/Kg2 − (1/(1+βVa))·(α/Kg1 + αs/Kg2))
```

**Important for:** EL500/PL504, 6L6, KT88 at high power, and also for true
pentodes (EL34) at Va < Vg2 in class AB/B operation.

> **LM19 port (audit 2026-07-12):** `lm19/reefman.py::_derk_ia_ig2`
> implements the formulas above in full — including the `+Psec` term in
> Ig2. NB: Reefman's own TubeLib.inc omits this term in its G2 sources
> (his library diverges from his own paper; the Ig2 hump in Fig. 14 of
> Theory.pdf is exactly Psec) — LM19 follows the paper, and our SPICE G2
> carries `+Ip·V(9)`. Ia is deliberately NOT clamped to 0: dynatron
> reversal is physical (the paper, TubeLib G1, and our SPICE agree;
> LTspice cross-check confirms the sign), while the old clamp created
> gradient-dead zones at trial Kg1 > Kg2 and stalled the fitter on sparse
> datasets (6P3S rms 75→3.4 mA, GU50 173→8.3). Pins:
> `tests/test_reefman_paper_pins.py`.

### 6.2 Dempwolf v2 (beam tetrodes)

A different formula, based on the crossover voltage:

```
Vco = Vg2/λ − ν·Vg1 − w
x = max(1 − Va/Vco, 0)
Isec = σ · I_through · (Va/Vg2) · x · exp(−Ks · x)

Ia  = Ia_primary − Isec
Ig2 = Ig2_base + Isec
```

| Parameter | Meaning |
|---|---|
| σ (sigma) | Secondary emission intensity |
| Ks | Decay rate |
| λ, ν, w | Crossover voltage Vco parameters |

Difference from Reefman: `x·exp(-Ks·x)` is used (peaking at Va≈Vco) instead
of `tanh` (a step-like transition). Active only at Va < Vco.

---

## 7. Variable-Mu Pentodes

Implemented in Reefman and Dempwolf v2.

### 7.1 Reefman: Two Parallel Pentodes

```
Ip,Koren_v = (1 − svar) · Ip,Koren_a + svar · Ip,Koren_b
```

where a and b have different µ and Ex but share the same Kp, Kvb. Typically:
- µa/µb ≈ 3.5
- svar ≈ 0.05–0.1

### 7.2 Dempwolf v2: Dual µ

Triode:
```
sp_a = ln(1 + exp(C · (Va/µ_a + Vg_eff))) / C
sp_b = ln(1 + exp(C · (Va/µ_b + Vg_eff))) / C
```

Pentode (Kp-normalized):
```
sp_a = (Vg2/C) · ln(1 + exp(C · (1/µ_a + Vg1/Vg2)))
sp_b = (Vg2/C) · ln(1 + exp(C · (1/µ_b + Vg1/Vg2)))
```

```
Ik = (1 − svar) · G · sp_a^γ_a + svar · G · sp_b^γ_b
```

Details in DEMPWOLF_EXTENDED_MODEL.md, §6.4 (v1) and §14.5.4 (v2).

---

## 8. Hexodes and Heptodes

Implemented **only** in Reefman. A "pentode + stacked pentode" model:

```
Ia = [pentode part (g1, g2)] × [triode/pentode part (g3, g4)] / Kg1'
```

The constant space-charge principle is preserved. Applicable to the ECH81,
and to the EF80 with g3 modulation.

---

## 9. Parameter Comparison

### 9.1 Triode

| Parameter | Koren | Dempwolf v2 | Analogy |
|---|---|---|---|
| µ | µ (10-100) | µ (10-100) | Same meaning |
| Ex | Ex (1.2-1.8) | γ (1.2-2.5) | Current exponent |
| Kg1 | Kg1 (100-5000) | 1/G | Perveance (inverse) |
| Kp | Kp (20-600) | C (3-5 triode) | Cutoff steepness |
| Kvb | Kvb (1-300) | Kvb_t (triode) | Island effect |
| — | — | Gg, Cg, ξ | Grid current (absent in Koren) |

### 9.2 Pentode (additional)

| Parameter | Koren | Derk/DerkE | Dempwolf v2 |
|---|---|---|---|
| Screen current | Kg2 | Kg2, αs, β | fg2, (1−α) split |
| Knee shape | arctan(Va/Kvb) | 1/(1+βVa) or exp(-(βVa)^1.5) | arctan((Va/Kvb)^Kn) |
| Space charge slope | — | A | A (Durchgriff) |
| Sec. emission | — | S, ap, λ, ν, ω | σ, Ks, λ, ν, w |
| Cutoff sharpness | Kp (20-600) | Kp (0.5-2.0!) | C (Kp-equiv, 50-750) |

**Caution:** Koren's and Reefman's Kp differ by orders of magnitude!
Koren Kp ≈ 20–600, Reefman Kp ≈ 0.5–2.0. This is because Reefman uses
`√(Kvb + Vg2²)` in the denominator instead of `Vg2`.

---

## 10. Model Selection Guide

```
                          Tube type
                              │
                ┌─────────────┼─────────────┐
                │             │             │
             Triode       Pentode       Heptode
                │             │             │
          Koren /      ┌──────┴──────┐   Reefman
          Dempwolf v2  │             │   (only)
                     True         Beam
                    pentode      tetrode
                       │             │
                ┌──────┼──┐     ┌────┤
                │      │  │     │    │
             Koren  Demp Derk Demp  DerkE
                    v2        v2
                       │        │      │
                  Sec.em.?  Sec.em.? Sec.em.?
                       │        │      │
                  Derk+Psec  Demp   DerkE
                             beam   +Psec
```

**Recommendations:**
- **Triode:** Koren or Dempwolf v2 — both are good; Dempwolf is slightly more accurate for grid current
- **Pentode (general case):** Dempwolf v2 (best RMS on EL84 data) or Derk
- **Small-signal pentode (EF86):** Derk — smooth knee; Dempwolf v2 — arctan-split
- **Beam tetrode (6L6, KT88):** DerkE — sharp knee; Dempwolf v2 — beam tetrode mode
- **EL34:** Derk (not DerkE!) — it is a true pentode; Dempwolf v2 also applies
- **Power tubes with sec. emission:** Derk/DerkE + Psec or Dempwolf v2 beam tetrode mode (different formulas)
- **Variable-mu (EF89, 6K7):** Reefman var-mu or Dempwolf v2 var-mu
- **Hexode/Heptode (ECH81):** Reefman only

---

## 11. Implementation Status in LM19

| Model | Status | File | Tests |
|---|---|---|---|
| **Koren Triode** | Implemented | `lm19/tube_sim.py` | 65 tests |
| **Koren Pentode** | Implemented | `lm19/tube_sim.py` | 65 tests |
| **Dempwolf v2 Triode** | Implemented | `lm19/dempwolf.py` | 94 + 28 paper-pins |
| **Dempwolf v2 Pentode** | Implemented | `lm19/dempwolf.py` | 94 + 28 paper-pins |
| **Dempwolf v2 Beam Tetrode** | Implemented | `lm19/dempwolf.py` | 94 + 28 paper-pins |
| **Dempwolf v2 Variable-mu** | Implemented | `lm19/dempwolf.py` | 94 + 28 paper-pins |
| **Reefman Derk (Pentode)** | Implemented | `lm19/reefman.py` | 53 + 19 paper-pins |
| **Reefman DerkE (Beam Tet.)** | Implemented | `lm19/reefman.py` | 53 + 19 paper-pins |
| **Reefman Psec (sec.em.)** | Implemented (BTetrodeD/DE, +Psec in Ig2 2026-07-12) | `lm19/reefman.py` | 53 + 19 paper-pins |
| Hexode/Heptode | Not implemented | — | — |

### Comparison Results on Real Data (EL84 / 6P14P)

Test: 13 measurement files from `measurements/EL84/`, pentode mode.
Script: `tools/compare_models.py`.

| Model | Wins (best RMS Ia) | Typical RMS Ia (mA) | Note |
|---|---|---|---|
| **Dempwolf v2** | **11 of 13** | 0.3 – 4.2 | Multi-phase fit, mature |
| **Koren** | 2 of 13 | 0.4 – 10.7 | Mature fit |
| Derk | 0 | 3.7 – 11.9 | Historical local copy (see below) |
| DerkE | 0 | 3.7 – 11.9 | Historical local copy (see below) |

**Historical caveat (resolved, ML-137 / 2026-07-05):** the Derk/DerkE
numbers in the table above were produced by a minimal local implementation
that lived in compare_models.py and had drifted away from the production
fitter. The script now calls the production `lm19.reefman.fit_reefman`
(phased fit, retry starts, adaptive bounds; it picks the better of D/DE
itself and prints a single "Reefman (D|DE)" row) — on regeneration the
table will reflect what the application user actually gets. The canonical
benchmark across all datasets is `python tools/fit_benchmark.py`.

### Potential Implementation Candidates

1. **Hexode/Heptode (ECH81)** — supported only by Reefman. A narrow niche.
2. **Cohen-Hélie** — of academic interest; a physically-motivated triode
   model (12-13 parameters), but with no pentode extension and no widely
   available implementation.

## Sources

Materials this document rests on. The full registry of the project
external sources (all entries, statuses, local copies) — `SOURCES_INDEX.md`.

### Reefman — Spice Models for Vacuum Tubes Using the uTracer (Theory.pdf, 2016)
- url: <https://www.dos4ever.com/uTracer3/Theory.pdf>
- type: theory
- role here: primary source for Derk/DerkE model equations
- note: 50-page paper by Derk Reefman (2016-01-24). Defines Derk (true
  pentode) and DerkE (beam tetrode) models with physically-motivated current
  splitting, constant space charge principle, secondary emission (Psec),
  variable-mu pentodes, and hexode/heptode models. Key equations: Ip,Koren
  with √(Kvb+Vg2²), Derk Ig2 = Ip/Kg2·(1+αs/(1+βVa)), DerkE Ig2 =
  Ip/Kg2·(1+αs·exp(-(βVa)^1.5)). Companion software: ExtractModel, utMax.
  Several ideas adopted into Dempwolf Extended v2.

### uTracer page 14 — ExtractModel and SPICE models
- url: <https://www.dos4ever.com/uTracer3/uTracer3_pag14.html>
- type: documentation
- role here: SPICE subcircuit reference
- note: Ronald Dekker's uTracer page with SPICE subcircuits for
  Koren/Derk/DerkE models. Equations in Theory.pdf.

### TubeLib.inc — ExtractModel SPICE library
- url: <https://www.dos4ever.com/uTracer3/TubeLib.inc>
- type: data
- role here: reference parameters read directly from the .inc
- note: 213 SPICE subcircuits (164 unique tubes) fitted by ExtractModel
  (Reefman, Jan 2016). Subcircuit types (9): TriodeK (98), BTetrodeD (23),
  BTetrodeDE (18), DiodeK (31), PenthodeD (8), PenthodeDE (4), PenthodeVD
  (3), PenthodeVDE (4), HepthodeD (10). Contains Derk/DerkE parameters for
  EL84, EL34, 6L6GC, KT88, EF86, etc.

### White Cottage — Valve Modelling
- url: <https://whitecottage.org.uk/guitar-and-audio/valve-modelling/>
- type: theory
- role here: extended Derk/DerkE equations with Vg1-dependent knee
- note: Detailed presentation of Derk/DerkE equations with extended knee
  functions: g(Va) = exp(-(β(1-αVg1)Va)^γ), h(Va) = exp(-(ρ(1-τVg1)Va)^θ).
  Adds Vg1-dependent knee parameters (α, γ, ρ, τ, θ) not in original Reefman
  paper.
