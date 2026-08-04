# Dempwolf Extended — Unified Vacuum Tube Model

Technical reference for the extended Dempwolf vacuum tube model covering
triodes, pentodes, and beam tetrodes in a single hierarchical framework.
Developed for the LM19 tube tester as a higher-fidelity alternative to
the Koren model.

**Primary sources:**
- K. Dempwolf, U. Zölzer, "A physically-motivated triode model for
  circuit simulations", *14th Int. Conf. Digital Audio Effects (DAFx-11)*,
  Paris, France, September 2011.
- W. R. Dunkel, M. Rest, K. J. Werner, M. J. Olsen, J. O. Smith III,
  "The Fender Bassman 5F6-A family of preamplifier circuits — a wave
  digital filter case study", *19th Int. Conf. Digital Audio Effects
  (DAFx-16)*, Brno, Czech Republic, September 2016.
- N. Koren, "Improved vacuum tube models for SPICE simulations",
  *Glass Audio*, Vol. 8, No. 5, 1996 (for comparison and Region A fix).

- D. Reefman, "Spice models for vacuum tubes using the uTracer",
  Theory.pdf, January 2016 (Derk/DerkE models, ExtractModel theory).

**Extensions developed for LM19 (v1):**
- Pentode extension (screen grid, current splitting)
- Adaptive knee (current-dependent Kvb)
- Knee sharpness exponent (Kn) — unifies smooth and sharp knee
- Variable-mu pentode (dual-section cathode, after Reefman)
- Beam tetrode secondary emission (kink modeling)
- Region A correction for triodes

**Improvements in v2 (based on Reefman analysis):**
- Screen grid geometric interception (fg2) — fixes Ig2→0 at high Va
- Anode Durchgriff (penetration factor A)
- Grid-voltage-dependent crossover for secondary emission (λ, ν, w)
- Energy-dependent secondary emission yield (I_sec ∝ I·Va/Vg2)

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Original Dempwolf Triode Model](#2-original-dempwolf-triode-model)
   - 2.1 [Equations](#21-equations)
   - 2.2 [Physical Meaning of Parameters](#22-physical-meaning-of-parameters)
   - 2.3 [Comparison with Koren Triode](#23-comparison-with-koren-triode)
   - 2.4 [Published Parameters (12AX7)](#24-published-parameters-12ax7)
3. [Region A Correction](#3-region-a-correction)
   - 3.1 [The Problem](#31-the-problem)
   - 3.2 [The Fix](#32-the-fix)
   - 3.3 [Corrected Triode Equation](#33-corrected-triode-equation)
4. [Pentode Extension](#4-pentode-extension)
   - 4.1 [Physics of Pentodes](#41-physics-of-pentodes)
   - 4.2 [Equations](#42-equations)
   - 4.3 [Adaptive Knee](#43-adaptive-knee)
   - 4.4 [Why Adaptive Knee Matters](#44-why-adaptive-knee-matters)
   - 4.5 [Operating Modes](#45-operating-modes)
5. [Beam Tetrode Extension](#5-beam-tetrode-extension)
   - 5.1 [Physics of Secondary Emission](#51-physics-of-secondary-emission)
   - 5.2 [Equations](#52-equations)
   - 5.3 [The Kink Shape](#53-the-kink-shape)
   - 5.4 [Pentode vs Beam Tetrode](#54-pentode-vs-beam-tetrode)
6. [Unified Model — Complete Equations](#6-unified-model--complete-equations)
   - 6.1 [Triode (8 parameters)](#61-triode-8-parameters)
   - 6.2 [Pentode (10 parameters)](#62-pentode-10-parameters)
   - 6.3 [Beam Tetrode (12 parameters)](#63-beam-tetrode-12-parameters)
   - 6.4 [Variable-mu Pentode (13 parameters)](#64-variable-mu-pentode-13-parameters)
   - 6.5 [Parameter Summary Table](#65-parameter-summary-table)
   - 6.6 [Model Hierarchy](#66-model-hierarchy)
   - 6.7 [Current Conservation Proof](#67-current-conservation-proof)
7. [Comparison with Koren](#7-comparison-with-koren)
   - 7.1 [Triode](#71-triode)
   - 7.2 [Pentode](#72-pentode)
   - 7.3 [Beam Tetrode](#73-beam-tetrode)
   - 7.4 [Summary Table](#74-summary-table)
8. [SPICE Subcircuit Templates](#8-spice-subcircuit-templates)
   - 8.1 [Triode Subcircuit](#81-triode-subcircuit)
   - 8.2 [Pentode Subcircuit](#82-pentode-subcircuit)
   - 8.3 [Beam Tetrode Subcircuit](#83-beam-tetrode-subcircuit)
9. [Python Implementation](#9-python-implementation)
   - 9.1 [Unified Model Function](#91-unified-model-function)
   - 9.2 [Usage — No Convenience Functions](#92-usage--no-convenience-functions)
   - 9.5 [Numerical Safety](#95-numerical-safety)
10. [Fitting Strategy](#10-fitting-strategy)
    - 10.1 [Phase 1 — Triode Core](#101-phase-1--triode-core)
    - 10.2 [Phase 2 — Grid Current](#102-phase-2--grid-current)
    - 10.2b [Triode Joint Refinement](#102b-triode-joint-refinement)
    - 10.3 [Phase 3 — Pentode Knee](#103-phase-3--pentode-knee)
    - 10.4 [Phase 4 — Full Pentode Refinement](#104-phase-4--full-pentode-refinement)
    - 10.5 [Phase 5 — Secondary Emission (Beam Tetrodes)](#105-phase-5--secondary-emission-beam-tetrodes)
    - 10.6 [Parameter Bounds](#106-parameter-bounds)
    - 10.7 [Residual Functions](#107-residual-functions)
    - 10.8 [Fit Quality Metrics](#108-fit-quality-metrics)
11. [Verification Examples](#11-verification-examples)
    - 11.1 [12AX7 Triode](#111-12ax7-triode)
    - 11.2 [EL34 Pentode](#112-el34-pentode)
    - 11.3 [6L6GC Beam Tetrode](#113-6l6gc-beam-tetrode)
12. [Edge Cases and Numerical Safety](#12-edge-cases-and-numerical-safety)
13. [Topology Detection](#13-topology-detection)
14. [Dempwolf Extended v2 — Improved Model](#14-dempwolf-extended-v2--improved-model)
    - 14.1 [Problem: Screen Current Vanishes at High Va](#141-problem-screen-current-vanishes-at-high-va)
    - 14.2 [Anode Durchgriff (Penetration Factor)](#142-anode-durchgriff-penetration-factor)
    - 14.3 [Grid-Voltage-Dependent Crossover](#143-grid-voltage-dependent-crossover-secondary-emission)
    - 14.4 [Energy-Dependent Secondary Emission](#144-energy-dependent-secondary-emission)
    - 14.5 [v2 Complete Equations](#145-v2-complete-equations)
    - 14.6 [Fitting Strategy Changes for v2](#146-fitting-strategy-changes-for-v2)
    - 14.7 [Additional Parameter Bounds (v2)](#147-additional-parameter-bounds-v2)
15. [Comparison with Koren and Derk/DerkE Models](#15-comparison-with-koren-and-derkderke-models)
    - 15.1 [Koren Model (1996)](#151-koren-model-1996)
    - 15.2 [Derk Model (Reefman 2014)](#152-derk-model-reefman-2014)
    - 15.3 [DerkE Model (Reefman 2014)](#153-derke-model-reefman-2014)
    - 15.4 [Derk + Secondary Emission](#154-derk--secondary-emission-reefman-2014)
    - 15.5 [Dempwolf Extended v2 (LM19)](#155-dempwolf-extended-v2-lm19-this-document)
    - 15.6 [Comparison Table — Equations](#156-comparison-table--equations)
    - 15.7 [Comparison Table — Parameter Count](#157-comparison-table--parameter-count)
    - 15.8 [Comparison Table — Capabilities](#158-comparison-table--capabilities)
    - 15.9 [When to Use Which Model](#159-when-to-use-which-model)
16. [Physicality Assessment](#16-physicality-assessment)
    - 16.1 [Classification](#161-classification)
    - 16.2 [Assessment Table](#162-assessment-table)
    - 16.3 [Key Physical Findings](#163-key-physical-findings)
17. [SPICE Simulator Practical Evaluation](#17-spice-simulator-practical-evaluation)
    - 17.1 [Convergence (Newton-Raphson)](#171-convergence-newton-raphson)
    - 17.2 [Simulation Speed](#172-simulation-speed)
    - 17.3 [Accuracy for Audio Design Tasks](#173-accuracy-for-audio-design-tasks)
    - 17.4 [Ecosystem and Tooling](#174-ecosystem-and-tooling)
    - 17.5 [Practical Recommendation](#175-practical-recommendation)
    - 17.6 [Migration Path](#176-migration-path)
18. [External References and Implementation Notes](#18-external-references-and-implementation-notes)
    - 18.1 [Reference Implementations](#181-reference-implementations)
    - 18.2 [Cohen-Hélie Model — Synonym](#182-cohen-hélie-model--synonym)
    - 18.3 [Ig0 Parameter — Omitted from LM19 Spec](#183-ig0-parameter--omitted-from-lm19-spec)
    - 18.4 [Published µ Value — 103.2, Not 100](#184-published-µ-value--1032-not-100)
    - 18.5 [Reefman Theory.pdf — DerkE Knee Derivation](#185-reefman-theorypdf--derke-knee-derivation)
    - 18.6 [Academic Validation — Dempwolf as Ground Truth](#186-academic-validation--dempwolf-as-ground-truth)
    - 18.7 [Quadric Surface Model — Not Applicable to LM19](#187-quadric-surface-model--not-applicable-to-lm19)
    - 18.8 [ExtractModel + .utd — Fast Path to 157+ Tubes](#188-extractmodel--utd--fast-path-to-157-tubes)
    - 18.9 [Published Dempwolf Parameters — Only 12AX7](#189-published-dempwolf-parameters--only-12ax7)

---

## 1. Motivation

The Koren model (1996) is the de facto standard for vacuum tube SPICE
simulation. It works well for most applications, but has limitations:

1. **Grid current** is modeled as a simple diode — discontinuous derivative
   at Vg = 0, poor accuracy for overdriven amplifier analysis.
2. **No current conservation** — plate current (Ia) and screen current (Ig2)
   are computed independently; their sum does not equal cathode current.
3. **Fixed knee shape** — the pentode knee (arctan) has the same shape for
   all grid voltages, but real tubes show wider knees at higher currents.
4. **No distinction between pentodes and beam tetrodes** — the "tetrode kink"
   (secondary emission) is not modeled at all.
5. **Separate equations** for triodes and pentodes — no unified framework.

The Dempwolf model (2011) addressed problem (1) with a physically-motivated
triode model featuring smooth grid current. However, it was published only
for triodes and only with 12AX7 parameters.

This document extends Dempwolf's approach into a **unified hierarchical
model** covering triodes, pentodes, and beam tetrodes, while addressing
all five limitations above.

---

## 1a. Implementation Status

> **This document is a technical reference for the Dempwolf Extended v2 model.**
> Sections 1–13 describe v1; Sections 14–17 describe v2 improvements
> and model comparisons.

### Implementation status

| Component | State | Module |
|---|---|---|
| SPICE export (Koren) | ✅ Implemented | `spice_export.py` |
| SPICE export (Dempwolf v2) | ✅ Implemented | `spice_export.py` |
| SPICE export (Reefman Derk/DerkE) | ✅ Implemented | `spice_export.py` |
| Dempwolf v2 model function | ✅ Implemented | `dempwolf.py` |
| Dempwolf v2 fitter (phased) | ✅ Implemented | `dempwolf.py` (`fit_dempwolf()`) |
| Reefman fitter | ✅ Implemented | `reefman.py` (`fit_reefman()`) |
| Tube simulator (Koren) | ✅ Implemented | `tube_sim.py` |
| .utd export (for ExtractModel) | ✅ Implemented | `utracer_export.py` |
| LTSpice test schematic gen. | ✅ Implemented | `ltspice_asc.py` |
| Topology in configs | `"triode"` \| `"pentode"` only | `config.py`, `tube_params.py` |
| Beam tetrodes (6L6, KT88, 6550…) | Classified as `"pentode"` | both configs |
| `detect_topology()` | **Not implemented** | — |

### Remaining work

1. Optionally add `"beam_tetrode"` topology value — currently all
   beam tetrodes work fine as `"pentode"` (σ = 0); the distinction
   only matters when kink modeling is needed
6. Add `detect_topology()` for automatic model selection from data
7. Integrate into SPICE export as an alternative to Koren
8. Build validated parameter library for common tubes

---

## 2. Original Dempwolf Triode Model

### 2.1 Equations

From Dempwolf & Zölzer (DAFx-11, 2011), as cited in Dunkel et al.
(DAFx-16, 2016):

```
IK  = G  · (ln(1 + exp(C  · (VPK/µ + VGK))) / C )^γ
IGK = Gg · (ln(1 + exp(Cg · VGK))            / Cg)^ξ
IPK = IK − IGK
```

Where:
- VPK = plate-to-cathode voltage (V)
- VGK = grid-to-cathode voltage (V)
- IK  = total cathode current (A)
- IGK = grid current (A)
- IPK = plate current (A)

The key insight: **cathode emission is the fundamental quantity**. The
cathode emits all electrons (IK); some are captured by the grid (IGK);
the rest reach the plate (IPK = IK − IGK). Current is conserved by
construction.

**Note (Ig0):** the paper's grid-current equation (11) adds a small
constant `Ig0` (≈ 4–8·10⁻⁸ A, Table 1) "due to stability reasons" in
state-space circuit simulation. Our implementation omits it: tens of nA
are three orders below LM19 measurement resolution (mA scale) and the
term carries no curve-shape information.

### 2.2 Physical Meaning of Parameters

#### µ (amplification factor, dimensionless)

Same as in the Koren model. The ratio of plate voltage change to grid
voltage change that keeps cathode current constant:

```
µ = −dVPK / dVGK   at constant IK
```

The term `VPK/µ + VGK` inside the exponential is the **effective control
voltage** — the grid voltage equivalent that combines the effects of
both plate and grid voltages on cathode emission.

#### G (perveance, A)

The cathode current scale factor. Larger G = more current for the same
effective voltage. Analogous to `1/Kg1` in Koren, but with a different
exponent convention.

Typical values: 1e-4 to 1e-2 A.

#### γ (current exponent, dimensionless)

The power-law exponent of the current-voltage relationship. Analogous
to Koren's `Ex`. The Langmuir-Child law gives γ = 3/2 = 1.5; real
tubes typically show γ = 1.2 — 1.7.

#### C (cutoff sharpness / Kp-equivalent)

Controls how sharply the tube transitions from conducting to cutoff.
Analogous to Koren's `Kp`. Larger C = sharper cutoff.

**Triode path:** C is the original Dempwolf sharpness parameter
(typical values 3–5). The `ln(1 + exp(C · x)) / C` construction is
the **softplus function divided by C**, which approximates:
- `x` when `x >> 0` (tube conducting)
- `0` when `x << 0` (tube in cutoff)
- smooth transition in between (width ≈ 1/C)

**Pentode path (Kp-normalized):** C is directly equivalent to Koren's Kp
(typical values 50–750). The formula `(VG2K/C) · ln(1 + exp(C · (1/µ +
VGK/VG2K)))` automatically adapts transition sharpness to the screen
voltage level. This is crucial for `triode_connected` mode where VG2K =
VPK varies across operating points.

#### Gg (grid perveance, A)

The grid current scale factor. Controls the magnitude of grid current
when VGK > 0 (grid conduction). No analog in the Koren model — Koren
uses a diode + resistor instead.

Typical values: 1e-5 to 1e-3 A.

#### ξ (grid current exponent, dimensionless)

The power-law exponent for grid current. Controls how steeply grid
current rises with positive grid voltage. Typically 1.2 — 1.5.

No analog in Koren.

#### Cg (grid transition sharpness, dimensionless)

Controls how sharply grid current turns on near VGK = 0. Larger Cg =
sharper transition (closer to ideal diode behavior).

The `ln(1 + exp(Cg · VGK)) / Cg` function provides a smooth onset of
grid current, unlike Koren's diode which has a derivative discontinuity.

### 2.3 Comparison with Koren Triode

| Aspect                    | Koren                              | Dempwolf                           |
|---------------------------|------------------------------------|------------------------------------|
| Fundamental quantity      | Plate current Ia                   | Cathode current IK                 |
| Grid current              | Diode D3 (discontinuous dI/dV)     | Analytic (smooth everywhere)       |
| Current conservation      | No (Ia independent of Ig)          | Yes (IPK = IK − IGK)              |
| Soft transition function  | `ln(1 + exp(Kp·x))`               | `ln(1 + exp(C·x)) / C`            |
| Region A handling         | `√(Kvb + Vp²)` — good             | None — weak point                  |
| Parameters                | 5                                  | 7                                  |
| Published tube data       | Dozens of tubes                    | 12AX7 only                         |

### 2.4 Published Parameters (12AX7)

From the original Dempwolf paper, Table 1, tube **RSD-1** (the same
values are carried into `config/tube_params.json`):

```
µ   = 103.2
G   = 2.242e-3 A
γ   = 1.26
C   = 3.40
Gg  = 6.177e-4 A
ξ   = 1.314
Cg  = 9.901
Ig0 = 8.025e-8 A   (omitted in our model — see the Ig0 note in §2.1)
```

---

## 3. Region A Correction

### 3.1 The Problem

**Region A** = large VPK, large negative VGK, low current. In this region,
the effective control voltage `VPK/µ + VGK` approaches zero from above.

In Koren's model, the term `Vg / sqrt(Kvb + Vp²)` ensures that the grid
influence is **modulated by the plate voltage**. At large Vp, the grid
has less control — this models the real physical phenomenon where plate
field penetrates through the grid and draws current even at large
negative grid voltages.

In the original Dempwolf model, the control voltage is simply `VPK/µ + VGK`
with no such modulation. This means the cutoff is "too clean" — the model
predicts zero current in Region A where real tubes show measurable leakage.

### 3.2 The Fix

Replace the grid voltage term with a plate-voltage-dependent version:

```
VGK  →  VGK · VPK / sqrt(Kvb_t + VPK²)
```

Where Kvb_t is the Region A parameter (in V², same as Koren's Kvb for
triodes).

This gives:
- When VPK² >> Kvb_t: `VGK · VPK / VPK = VGK` (normal operation, no change)
- When VPK is small: `VGK · VPK / sqrt(Kvb_t)` (reduced grid influence)

The correction adds 1 parameter (Kvb_t) and brings Region A behavior
to parity with Koren.

### 3.3 Corrected Triode Equation

```
V_grid_eff = VGK · VPK / sqrt(Kvb_t + VPK²)

IK  = G  · (ln(1 + exp(C  · (VPK/µ + V_grid_eff))) / C )^γ
IGK = Gg · (ln(1 + exp(Cg · VGK))                   / Cg)^ξ
IPK = max(IK − IGK, 0)
```

**Note:** The grid current equation still uses raw VGK, not V_grid_eff.
Grid current depends only on the actual grid-cathode voltage, not on
the plate's influence on cathode emission.

**Note:** Kvb_t is in V² (volts squared), same as Koren's Kvb for
triodes. Typical range: 50 — 3000 V².

---

## 4. Pentode Extension

### 4.1 Physics of Pentodes

In a pentode, the screen grid (G2) replaces the plate as the primary
accelerating electrode:

```
Cathode → [G1] → [G2] → [G3] → Plate
  IK       IGK    IG2K          IPK

IK = IGK + IG2K + IPK   (current conservation)
```

Key physical differences from a triode:
1. **VG2K controls cathode emission** (not VPK)
2. **VPK determines current splitting** between plate and G2
3. The **knee** occurs when VPK drops below VG2K
4. **High output impedance** — IPK nearly independent of VPK above knee

### 4.2 Equations

```
IK  = G  · ((VG2K/C) · ln(1 + exp(C · (1/µ + VGK/VG2K))))^γ
IGK = Gg · (ln(1 + exp(Cg · VGK))             / Cg)^ξ

I_through = max(IK − IGK, 0)

α = (2/π) · arctan(VPK / Kvb_eff)

IPK  = I_through · α
IG2K = I_through · (1 − α)
```

The substitution VPK → VG2K in the cathode emission equation follows
directly from physics: the screen grid creates the accelerating field
that drives electron emission, not the plate.

**Kp-normalization:** The pentode softplus uses the Koren-style
`(VG2K/C) · ln(1 + exp(C · (1/µ + VGK/VG2K)))` form instead of the
original Dempwolf `ln(1 + exp(C · (VG2K/µ + VGK))) / C`. These are
mathematically equivalent at fixed VG2K (with `C_new = C_old × VG2K`),
but the normalized form automatically adapts the transition sharpness to
the screen voltage level. This is crucial for `triode_connected` mode
where VG2K = VPK varies across operating points. The triode path retains
the original Dempwolf form, preserving its decoupled C parameter.

The arctan current-splitting function is the same proven form as Koren:
- α → 0 when VPK → 0 (all current to G2, the knee)
- α → 1 when VPK >> Kvb (all current to plate, flat region)

### 4.3 Adaptive Knee

In real pentodes, the knee shape depends on the current level. At higher
currents (less negative VGK), the knee is wider and shifted to higher VPK.

**Physical reason:** Space charge between G2 and the plate. Higher
current = denser electron beam = deeper potential minimum = electrons
need more VPK to reach the plate.

The standard Koren model uses a fixed Kvb, giving identical knee shape
for all grid voltages. This is a significant source of error in the
knee region.

**Solution:** Make Kvb depend on the effective control voltage:

```
V_eff = max(VG2K/µ + VGK, 0)
Kvb_eff = Kvb + Kvb1 · V_eff
```

Where:
- Kvb — base knee voltage at cutoff (small currents)
- Kvb1 — knee broadening coefficient (dimensionless)

This adds 1 parameter and provides:
- At cutoff (V_eff ≈ 0): Kvb_eff ≈ Kvb (sharp knee, low current)
- At high current (V_eff large): Kvb_eff >> Kvb (broad knee)

**Why V_eff instead of I_through directly?**

Using the actual current would give `Kvb_eff = Kvb + δ · I^(2/3)` (from
space charge theory, Poisson's equation). However, since `I ∝ V_eff^γ`,
the voltage proxy gives `Kvb_eff ∝ I^(1/γ) ≈ I^0.74` — close to the
theoretically correct `I^0.67`. The optimizer compensates for the
exponent difference by adjusting Kvb1. The voltage form is simpler
and avoids a two-stage computation.

### 4.4 Why Adaptive Knee Matters

At the operating point of a typical class AB push-pull amplifier, the
load line crosses through the knee region as each tube transitions
between conduction and cutoff. The exact shape of the knee determines:
- Crossover distortion characteristics
- Power output at clipping
- The harmonic spectrum of the distorted signal

A fixed-Kvb model systematically misshapes the knee at extreme grid
voltages, leading to errors in distortion prediction.

### 4.5 Operating Modes

A pentode can operate in three modes, all handled by the same equations:

**Pentode mode (VG2K = constant):**
```
Standard operation. VG2K is set by the power supply.
```

**Triode mode (VG2K = VPK):**
```
Screen connected to plate. Substituting VG2K = VPK:
IK = G · (ln(1+exp(C · (VPK/µ + VGK))) / C)^γ
α = (2/π) · arctan(VPK/Kvb_eff) ≈ 1  for VPK >> Kvb
IPK ≈ I_through ≈ IK − IGK
```
This naturally reduces to the triode model.

**Ultra-linear mode (VG2K = partial tracking):**
```
VG2K = VG2_nom · (1 − tap) + VPK · tap
```
Where tap is typically 0.4 (40% UL tap). The same equations apply
with the computed VG2K value.

---

## 5. Beam Tetrode Extension

### 5.1 Physics of Secondary Emission

In a **pentode** (EL34, EF86), the suppressor grid G3 (at cathode
potential) repels secondary electrons back to the plate. No kink.

In a **beam tetrode** (6L6, KT88, 6550, 6V6), there is no suppressor
grid. Instead, beam-forming plates focus electrons into dense beams.
The space charge of the beam partially suppresses secondary emission,
but not completely.

When VPK < VG2K:
1. Primary electrons hit the plate with kinetic energy ∝ VPK
2. Secondary electrons are knocked out of the plate surface
3. G2 (at higher potential) attracts these secondary electrons
4. Net effect: current flows FROM the plate TO G2

This creates the **tetrode kink** — a region where increasing VPK can
actually decrease IPK (negative resistance).

```
Ia
│
│  Vg=0   ╭─╮    ╭────────────
│         │  ╲──╱      ← kink (negative resistance region)
│         │
│  Vg=-10╭──────────────────
│        │
└────────┼──────────────────── Vp
         0     Vg2
```

### 5.2 Equations

The secondary emission current is added as a perturbation to the
current splitting:

```
x = max(1 − VPK / VG2K, 0)

I_sec = σ · IPK_primary · x · exp(−Ks · x)
```

Where:
- σ — secondary emission yield coefficient (dimensionless)
  - σ = 0 for true pentodes (suppressor grid eliminates effect)
  - σ = 0.1–0.5 for beam tetrodes
- Ks — kink shape parameter (dimensionless, typically 3–5)
- IPK_primary — plate current before secondary emission correction
- x — normalized distance below screen voltage (0 at VPK = VG2K)

The net currents become:

```
IPK  = IPK_primary − I_sec     (secondaries leave the plate)
IG2K = IG2K_base   + I_sec     (secondaries arrive at G2)
```

Current is conserved: the secondary emission only redistributes current
between plate and screen grid, without changing the total cathode current.

### 5.3 The Kink Shape

The function `x · exp(−Ks · x)` is bell-shaped:
- Zero at x = 0 (VPK = VG2K): no secondary emission effect
- Peak at x = 1/Ks: maximum secondary emission
- Decays exponentially for large x (VPK << VG2K)

For Ks = 4, the peak occurs at x = 0.25, i.e., VPK = 0.75 · VG2K.
With VG2K = 250V, the kink peaks at VPK ≈ 190V.

For Ks = 3, peak at x = 0.33, VPK ≈ 0.67 · VG2K.

The decay at large x models the physical fact that at very low VPK,
primary electrons have less energy and produce fewer secondaries.

### 5.4 Pentode vs Beam Tetrode

| Aspect                  | True Pentode (EL34)     | Beam Tetrode (6L6)      |
|-------------------------|-------------------------|-------------------------|
| Suppressor grid G3      | Yes (at cathode pot.)   | No                      |
| Beam-forming plates     | No                      | Yes                     |
| Secondary emission      | Fully suppressed        | Partially suppressed    |
| Kink in Ia-Va curves    | None                    | Visible                 |
| σ parameter             | 0                       | 0.1 — 0.5              |
| Model difference        | Equations with σ = 0    | Equations with σ > 0    |

---

## 6. Unified Model — Complete Equations

### 6.1 Triode (8 parameters)

Parameters: `µ, G, γ, C, Gg, ξ, Cg, Kvb_t`

```
V_grid_eff = VGK · VPK / sqrt(Kvb_t + VPK²)

IK  = G  · (ln(1 + exp(C  · (VPK/µ + V_grid_eff))) / C )^γ
IGK = Gg · (ln(1 + exp(Cg · VGK))                   / Cg)^ξ

IPK = max(IK − IGK, 0)
```

### 6.2 Pentode (10 parameters)

Parameters: `µ, G, γ, C, Gg, ξ, Cg, Kvb, Kvb1, Kn`

Note: Kvb_t (Region A) is not used for pentodes — the screen grid
provides the accelerating field, and Region A is not a concern because
VG2K is constant and independent of the plate.

```
IK  = G  · ((VG2K/C) · ln(1 + exp(C · (1/µ + VGK/VG2K))))^γ
IGK = Gg · (ln(1 + exp(Cg · VGK))             / Cg)^ξ

I_through = max(IK − IGK, 0)

V_eff   = max(VG2K/µ + VGK, 0)
Kvb_eff = Kvb + Kvb1 · V_eff

α = (2/π) · arctan((VPK / Kvb_eff)^Kn)

IPK  = I_through · α
IG2K = I_through · (1 − α)
```

The pentode cathode emission uses the Kp-normalized softplus form (see §4.2).
C is analogous to Koren's Kp (typical values 50–750).

`Kn` controls knee sharpness: `Kn = 1.0` gives standard smooth knee
(equivalent to original arctan), `Kn > 1` gives sharper transition
similar to DerkE's `exp(−(βVa)^1.5)` for beam-like tubes.

### 6.3 Beam Tetrode (12 parameters)

Parameters: `µ, G, γ, C, Gg, ξ, Cg, Kvb, Kvb1, Kn, σ, Ks`

Same as pentode, plus secondary emission:

```
IK  = G  · ((VG2K/C) · ln(1 + exp(C · (1/µ + VGK/VG2K))))^γ
IGK = Gg · (ln(1 + exp(Cg · VGK))             / Cg)^ξ

I_through = max(IK − IGK, 0)

V_eff   = max(VG2K/µ + VGK, 0)
Kvb_eff = Kvb + Kvb1 · V_eff

α = (2/π) · arctan((VPK / Kvb_eff)^Kn)

IPK_primary = I_through · α
IG2K_base   = I_through · (1 − α)

x     = max(1 − VPK / VG2K, 0)
I_sec = σ · IPK_primary · x · exp(−Ks · x)

IPK  = IPK_primary − I_sec
IG2K = IG2K_base   + I_sec
```

### 6.4 Variable-mu Pentode (13 parameters)

Parameters: `µ_a (≡ µ), γ_a (≡ γ), G, C, Gg, ξ, Cg, Kvb, Kvb1, Kn, µ_b, γ_b, svar`

Based on Reefman's approach (Theory.pdf §5): the variable-mu tube is
modeled as two pentode sections in parallel with different grid winding
pitch — a tight-wound section (high-µ, dominant: most of the grid
length) and a wide-wound section (low-µ, small contribution: the
wide-pitch middle that keeps conducting at deep bias and produces the
remote-cutoff tail). Reefman §5.2: µ_a > µ_b, and `svar` — the weight
of the low-µ section — fits to ≈ 0.05–0.1.

```
IK_a = G · ((VG2K/C) · ln(1 + exp(C · (1/µ_a + VGK/VG2K))))^γ_a
IK_b = G · ((VG2K/C) · ln(1 + exp(C · (1/µ_b + VGK/VG2K))))^γ_b

IK  = (1 − svar) · IK_a  +  svar · IK_b
IGK = Gg · (ln(1 + exp(Cg · VGK)) / Cg)^ξ

I_through = max(IK − IGK, 0)

V_eff   = max(VG2K/µ_a + VGK, 0)
Kvb_eff = Kvb + Kvb1 · V_eff

α = (2/π) · arctan((VPK / Kvb_eff)^Kn)

IPK  = I_through · α
IG2K = I_through · (1 − α)
```

Shared across both sections: `G, C, Gg, ξ, Cg, Kvb, Kvb1` —
determined by geometry common to both halves. Section-specific:
`µ_a, γ_a` (tight-wound, high-µ, weight `1 − svar`) and
`µ_b, γ_b` (wide-wound, low-µ, weight `svar`).

Typical values: `µ_a/µ_b ≈ 3–4`, `svar ≈ 0.05–0.1` (Reefman §5.2).

### 6.5 Parameter Summary Table

| #  | Parameter | Units     | Triode | Pentode | Var-mu | Beam Tet. | Description                     |
|----|-----------|-----------|:------:|:-------:|:------:|:---------:|---------------------------------|
| 1  | µ (µ_a)   | —         | +      | +       | +      | +         | Amplification factor (section A)|
| 2  | G         | A         | +      | +       | +      | +         | Cathode current perveance       |
| 3  | γ (γ_a)   | —         | +      | +       | +      | +         | Current exponent (section A)    |
| 4  | C         | —         | +      | +       | +      | +         | Cutoff sharpness                |
| 5  | Gg        | A         | +      | +       | +      | +         | Grid current perveance          |
| 6  | ξ         | —         | +      | +       | +      | +         | Grid current exponent           |
| 7  | Cg        | —         | +      | +       | +      | +         | Grid transition sharpness       |
| 8  | Kvb_t     | V²        | +      | —       | —      | —         | Region A parameter (triode)     |
| 9  | Kvb       | V         | —      | +       | +      | +         | Base knee voltage (pentode)     |
| 10 | Kvb1      | —         | —      | +       | +      | +         | Knee broadening coefficient     |
| 11 | Kn        | —         | —      | +       | +      | +         | Knee sharpness exponent         |
| 12 | µ_b       | —         | —      | —       | +      | —         | Amplification factor (section B)|
| 13 | γ_b       | —         | —      | —       | +      | —         | Current exponent (section B)    |
| 14 | svar      | —         | —      | —       | +      | —         | Section B weight fraction       |
| 15 | σ         | —         | —      | —       | —      | +         | Secondary emission yield        |
| 16 | Ks        | —         | —      | —       | —      | +         | Kink shape parameter            |
|    | **Total** |           | **8**  | **10**  | **13** | **12**    |                                 |

### 6.6 Model Hierarchy

```
Variable-mu Pentode (13 params)
  │ set svar = 0, drop µ_b, γ_b
  ▼
Pentode (10 params)          Beam Tetrode (12 params)
  │ set VG2K = VPK,            │ set σ = 0
  │ Kn = 1, Kvb → ∞           ▼
  ▼                           Pentode (10 params)
Triode (8 params)               │ ...
```

Variable-mu and Beam Tetrode are independent extensions of the
Pentode base. A single code path handles all topologies by setting
unused parameters to their identity values.

### 6.7 Current Conservation Proof

**Triode:**
```
IK = IPK + IGK
   = (IK − IGK) + IGK = IK  ✓
```

**Pentode:**
```
IK = IPK + IG2K + IGK
   = I_through·α + I_through·(1−α) + IGK
   = I_through + IGK
   = (IK − IGK) + IGK = IK  ✓
```

**Beam tetrode:**
```
IK = IPK + IG2K + IGK
   = (IPK_primary − I_sec) + (IG2K_base + I_sec) + IGK
   = IPK_primary + IG2K_base + IGK
   = I_through·α + I_through·(1−α) + IGK
   = I_through + IGK = IK  ✓
```

Secondary emission redistributes current between plate and screen
without affecting the total cathode current. This holds for any value
of σ.

---

## 7. Comparison with Koren

### 7.1 Triode

| Aspect                    | Koren (5 params)                               | Derk (5 params)                                | Dempwolf v2 (8 params)                         |
|---------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|
| Ia equation               | `2·E1^Ex / Kg1`                                | `2·E1^Ex / Kg1` (same)                        | `IK − IGK`                                     |
| E1 / IK                   | `(Vp/Kp)·ln(1+exp(Kp·(1/µ+Vg/√(Kvb+Vp²))))` | Same as Koren                                  | `G·(ln(1+exp(C·(Vp/µ+Vg_eff)))/C)^γ`         |
| Grid current              | Diode `D3` + resistor `RGI`                   | Diode `D3` + resistor `RGI`                   | `Gg·(ln(1+exp(Cg·Vgk))/Cg)^ξ` — smooth       |
| dIg/dVg continuous        | No (kink at Vg = 0)                            | No (kink at Vg = 0)                            | **Yes** (smooth everywhere)                    |
| Region A                  | Good (`√(Kvb+Vp²)`)                           | Good (`√(Kvb+Vg2²)`)                          | Good (`√(Kvb_t+Vp²)`)                         |
| Current conservation      | No                                              | No                                              | **Yes** (IPK = IK − IGK)                      |

### 7.2 Pentode

| Aspect                    | Koren (6 params)                           | Derk (9 params)                                | DerkE (9 params)                               | Dempwolf v2 (12 params)                        |
|---------------------------|--------------------------------------------|-------------------------------------------------|------------------------------------------------|------------------------------------------------|
| Ia equation               | `E1^Ex/Kg1 · arctan(Vp/Kvb)`              | `Ik − Ig2` (from balance)                      | `Ik − Ig2` (from balance)                      | `I_through · α` (from splitting)               |
| Ig2 equation              | `(Vg+Vg2/µ)^Ex / Kg2`                     | `IP/Kg2·(1+αs/(1+βVa))`                        | `IP/Kg2·(1+αs·exp(−(βVa)^1.5))`               | `I_through · (1 − α)`                          |
| Ig2 depends on Va?        | **No**                                     | **Yes** — `1/(1+βVa)`                           | **Yes** — `exp(−(βVa)^1.5)`                    | **Yes** — from `(1−α)` splitting               |
| Ig2 at Va → ∞             | Constant (correct)                         | **IP/Kg2 > 0** (correct)                        | **IP/Kg2 > 0** (correct)                        | **fg2 · Ik > 0** (correct, v2 fix)             |
| Ia + Ig2 = Ik             | **No** — independent                       | **≈ Yes** (constant space charge)               | **≈ Yes** (constant space charge)               | **Yes** — exact by construction                 |
| Durchgriff (A·Va)         | No                                         | **Yes** — `Ik·(1+A·Va)`                         | **Yes** — `Ik·(1+A·Va)`                         | **Yes** — `Ik·(1+A·Va)`                         |
| Knee function             | `arctan(Va/Kvb)` — fixed                  | `1/(1+βVa)` — fixed                            | `exp(−(βVa)^1.5)` — fixed, sharper             | `arctan((Va/Kvb_eff)^Kn)` — **adaptive+sharp** |
| Knee adapts to Vg?        | No                                         | No                                              | No                                              | **Yes** — `Kvb_eff = Kvb + Kvb1·V_eff`         |
| Knee sharpness control    | No                                         | No                                              | Sharper by design                               | **Yes** — `Kn` exponent (Kn>1 = DerkE-like)    |
| Grid current              | Diode                                      | Diode                                           | Diode                                           | **Smooth** softplus                             |
| Pentode-as-triode         | Inconsistent scaling                       | **Consistent** (δx correction)                  | **Consistent** (δx correction)                  | **Consistent** (VG2K=VPK, fg2=0)               |

### 7.3 Beam Tetrode

| Aspect                    | Koren (6 params)                           | Derk+SE (14 params)                             | DerkE+SE (14 params)                            | Dempwolf v2 (17 params)                         |
|---------------------------|--------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|
| Kink modeled              | **No**                                     | **Yes** — `S·Va·(1+tanh(…))`                   | **Yes** — `S·Va·(1+tanh(…))`                   | **Yes** — `σ·I·Va/Vg2·x·exp(−Ks·x)`           |
| Kink source               | —                                          | ∝ Va (energy only)                              | ∝ Va (energy only)                              | **∝ I·Va** (energy × flux)                      |
| Vco depends on Vg?        | —                                          | **Yes** — `Vco = Vg2/λ − ν·Vg1 − w`           | **Yes** — `Vco = Vg2/λ − ν·Vg1 − w`           | **Yes** — `Vco = VG2K/λ − ν·VGK − w`          |
| Kink shape function       | —                                          | `tanh` (smooth step)                            | `tanh` (smooth step)                            | `x·exp(−Ks·x)` (bell-shaped)                   |
| Negative resistance       | Impossible                                 | **Yes** (for large S)                           | **Yes** (for large S)                           | **Yes** (for large σ)                           |
| Distinction from pentode  | None                                       | S = 0 → no kink                                | S = 0 → no kink                                | σ = 0 → no kink                                |
| Knee at high current      | Same shape as low current                  | Same shape                                      | Same shape (sharper)                            | **Wider** (adaptive Kvb)                        |
| Current conservation      | No                                         | ≈ Yes                                           | ≈ Yes                                           | **Exact**                                       |

### 7.4 Summary Table

|                             | Koren         | Derk             | DerkE            | Derk/E+SE        | **Dempwolf v2**   |
|-----------------------------|:-------------:|:----------------:|:----------------:|:----------------:|:-----------------:|
| Triode parameters           | 5             | 5                | 5                | 5                | **8**             |
| Pentode parameters          | 6             | 9                | 9                | 9                | **12**            |
| Variable-mu parameters      | —             | 12               | 12               | 17               | **15**            |
| Beam tetrode parameters     | 6             | —                | 9                | 14               | **17**            |
| Grid current                | Diode (kink)  | Diode (kink)     | Diode (kink)     | Diode (kink)     | **Smooth**        |
| Current conservation        | No            | ≈ Yes            | ≈ Yes            | ≈ Yes            | **Exact**         |
| Adaptive knee               | No            | No               | No               | No               | **Yes**           |
| Sharp knee (Kn)             | No            | No               | By design        | By design        | **Yes** (Kn>1)    |
| Ig2 at Va → ∞              | Constant      | > 0 (correct)    | > 0 (correct)    | > 0 (correct)    | **fg2·Ik > 0**   |
| Durchgriff (A·Va)           | No            | Yes              | Yes              | Yes              | **Yes**           |
| Kink (secondary emission)   | No            | No               | No               | **Yes**          | **Yes**           |
| Vco depends on Vg           | —             | —                | —                | **Yes**          | **Yes**           |
| Energy-dependent kink       | —             | —                | —                | ∝ Va             | **∝ I·Va**       |
| Region A                    | Good          | Good             | Good             | Good             | Good              |
| Pentode-as-triode           | Inconsistent  | Consistent       | Consistent       | Consistent       | **Consistent**    |
| Unified framework           | No            | No               | No               | No               | **Yes**           |
| Variable-mu                 | No            | **Yes**          | **Yes**          | **Yes**          | **Yes**           |
| Heptodes                    | Partial       | **Yes**          | **Yes**          | **Yes**          | No                |
| Ready library               | Dozens        | **157+ tubes**   | **157+ tubes**   | **157+ tubes**   | 12AX7 only        |
| SPICE compatible            | Yes           | Yes              | Yes              | Yes              | Yes               |
| Ecosystem maturity          | 30 years      | 10+ years        | 10+ years        | 10+ years        | New               |

---

## 8. SPICE Subcircuit Templates

### 8.1 Triode Subcircuit

```spice
.SUBCKT DEMPWOLF_TRIODE A G K
+ PARAMS: MU=100 G=2.242E-3 GAMMA=1.26 C=3.4
+         GG=6.177E-4 XI=1.314 CG=9.901 KVBT=300
+         CCG=2.3P CGP=2.4P CCP=0.9P
*
* V_grid_eff = Vgk * Vpk / sqrt(Kvb_t + Vpk^2)
* IK = G * (ln(1+exp(C*(Vpk/mu + V_grid_eff))) / C) ^ gamma
E1 7 0 VALUE={
+ LOG(1+EXP(C*(V(A,K)/MU
+   +V(G,K)*V(A,K)/SQRT(KVBT+V(A,K)*V(A,K)))))/C}
RE1 7 0 1G
*
* Plate current source: G * E1^gamma
* (IK minus IGK, but we compute IPK = IK - IGK via separate sources)
G1 A K VALUE={G*PWR(V(7),GAMMA)}
*
* Grid current: IGK = Gg * (ln(1+exp(Cg*Vgk))/Cg) ^ xi
E2 8 0 VALUE={LOG(1+EXP(CG*V(G,K)))/CG}
RE2 8 0 1G
G2 G K VALUE={GG*PWR(V(8),XI)}
*
* Net plate current: IPK = IK - IGK (SPICE sums G1 and -G2 at node A)
* G1 sources IK from A to K; G2 sources IGK from G to K.
* Kirchhoff: current leaving A = IK, current leaving G = IGK,
* current entering K = IK + IGK = total cathode current.
* Actually IPK = IK - IGK requires correction:
* G1 sources IK (total cathode) into plate — we need to subtract IGK.
* Use a dependent source:
R1 A K 1G
*
* Interelectrode capacitances
C1 G K {CCG}
C2 G A {CGP}
C3 A K {CCP}
*
.ENDS DEMPWOLF_TRIODE
```

**Alternative cleaner formulation using explicit IPK:**

```spice
.SUBCKT DEMPWOLF_TRIODE_V2 A G K
+ PARAMS: MU=100 G=2.242E-3 GAMMA=1.26 C=3.4
+         GG=6.177E-4 XI=1.314 CG=9.901 KVBT=300
+         CCG=2.3P CGP=2.4P CCP=0.9P
*
* Intermediate: cathode emission softplus argument
E1 7 0 VALUE={
+ LOG(1+EXP(C*(V(A,K)/MU
+   +V(G,K)*V(A,K)/SQRT(KVBT+V(A,K)*V(A,K)))))/C}
RE1 7 0 1G
*
* Intermediate: grid current softplus argument
E2 8 0 VALUE={LOG(1+EXP(CG*V(G,K)))/CG}
RE2 8 0 1G
*
* IK = G * E1^gamma
E3 9 0 VALUE={G*PWR(V(7),GAMMA)}
RE3 9 0 1G
*
* IGK = Gg * E2^xi
E4 10 0 VALUE={GG*PWR(V(8),XI)}
RE4 10 0 1G
*
* Plate current: IPK = IK - IGK
G1 A K VALUE={MAX(V(9)-V(10), 0)}
*
* Grid current: IGK
G2 G K VALUE={V(10)}
*
* Convergence helper
RCP A K 1G
*
* Interelectrode capacitances
C1 G K {CCG}
C2 G A {CGP}
C3 A K {CCP}
*
.ENDS DEMPWOLF_TRIODE_V2
```

> **NB — shipped generator:** `_generate_dempwolf_triode_subcircuit()`
> (`lm19/spice_export/dempwolf.py`) emits neither template above
> verbatim: like the pentode generator it uses Koren's diode `D3`+`RGI`
> for grid current and does **not** subtract `IGK` from the plate
> current (equivalent for `VGK < 0`; the fitted `Gg/ξ/Cg` stay
> Python-side). The templates here are the analytic-IGK design study.

### 8.2 Pentode Subcircuit (v2)

Generated by `_generate_dempwolf_pentode_subcircuit()` in `spice_export.py`.
Uses Kp-normalized softplus, fg2 geometric interception, and Durchgriff (A).

```spice
.SUBCKT DEMPWOLF_PENTODE A G K G2
+ PARAMS: MU=11.0000 G=3.000000E-03 GAMMA=1.3500 C=625.0000
+ KVB=24.0000 KVB1=0.5000 KN=1.0000 FG2=0.050000 A=2.000000E-04
+ TWOPI=0.6366197724
+ CCG1=14.0P CCG2=4.5P CPG1=0.85P CG1G2=3.0P CCP=12.0P RGI=2000
*
* E1: Kp-normalized softplus cathode intermediate
*   sp = (Vg2/C) · ln(1 + exp(C · (1/µ + Vg1/Vg2)))
RE1 7 0 1MEG
E1 7 0 VALUE=
+{V(G2,K)/C*LOG(1+EXP(MIN(C*(1/MU+V(G,K)/MAX(V(G2,K),0.01)),700)))}
*
* E2: cathode emission current Ik = G · sp^γ · (1 + A·Va)
E2 8 0 VALUE=
+{G*(PWR(MAX(V(7),0),GAMMA))*(1+A*MAX(V(A,K),0.01))}
RE2 8 0 1MEG
*
* E4: adaptive knee  Kvb_eff = max(Kvb + Kvb1 · max(Vg2/µ + Vg1, 0), 0.1)
E4 10 0 VALUE=
+{MAX(KVB+KVB1*MAX(V(G2,K)/MU+V(G,K),0),0.1)}
RE4 10 0 1MEG
*
* E5: α = (1−fg2) · (2/π) · arctan((Va/Kvb_eff)^Kn)
E5 11 0 VALUE=
+{(1-FG2)*TWOPI*ATAN(PWR(MAX(V(A,K),0.01)/V(10),KN))}
RE5 11 0 1MEG
*
* Current splitting — Ia = Ik·α,  Ig2 = Ik·(1−α)
G1 A K VALUE={V(8)*V(11)}
G2 G2 K VALUE={V(8)*(1-V(11))}
RCP A K 1G
*
* Interelectrode capacitances
C1 G K {CCG1}    ; cathode-grid1
C4 G2 K {CCG2}   ; cathode-grid2
C5 G2 G {CG1G2}  ; grid1-grid2
C2 A G {CPG1}    ; grid1-plate
C3 A K {CCP}     ; cathode-plate
*
R1 G 5 {RGI}     ; grid current
D3 5 K DX
.MODEL DX D(IS=1N RS=1 CJO=10PF TT=1N)
*
.ENDS DEMPWOLF_PENTODE
```

**Key differences from v1:** Softplus is now Kp-normalized (`Vg2/C · ln(...)` instead
of `ln(...)/C`). Parameter `C` has Kp-equivalent scale (~50–750 instead of ~2–5).
Added `FG2` (geometric screen interception fraction), `A` (Durchgriff), and five
capacitances (was three). Grid current via a diode model (D3) instead of explicit
softplus source.

### 8.3 Beam Tetrode Subcircuit (v2)

Same generator as §8.2, with `has_sec=True` adding secondary emission
parameters (σ, Ks, λ, ν, w). Uses energy-dependent yield `I_through · Va/Vg2`
and grid-voltage-dependent crossover `Vco = Vg2/λ − ν·Vg1 − w`.

```spice
.SUBCKT DEMPWOLF_BEAMTET A G K G2
+ PARAMS: MU=7.9000 G=3.500000E-03 GAMMA=1.3500 C=625.0000
+ KVB=24.0000 KVB1=0.5000 KN=1.0000 FG2=0.050000 A=2.000000E-04
+ TWOPI=0.6366197724
+ SIGMA=3.0000 KS=1.5000 LAM=1.0000 NU=2.0000 W=0.0000
+ CCG1=14.0P CCG2=4.5P CPG1=0.85P CG1G2=3.0P CCP=12.0P RGI=2000
*
* E1: Kp-normalized softplus  sp = (Vg2/C) · ln(1 + exp(C · (1/µ + Vg1/Vg2)))
RE1 7 0 1MEG
E1 7 0 VALUE=
+{V(G2,K)/C*LOG(1+EXP(MIN(C*(1/MU+V(G,K)/MAX(V(G2,K),0.01)),700)))}
*
* E2: Ik = G · sp^γ · (1 + A·Va)
E2 8 0 VALUE=
+{G*(PWR(MAX(V(7),0),GAMMA))*(1+A*MAX(V(A,K),0.01))}
RE2 8 0 1MEG
*
* E4: Kvb_eff = max(Kvb + Kvb1 · max(Vg2/µ + Vg1, 0), 0.1)
E4 10 0 VALUE=
+{MAX(KVB+KVB1*MAX(V(G2,K)/MU+V(G,K),0),0.1)}
RE4 10 0 1MEG
*
* E5: α = (1−fg2) · (2/π) · arctan((Va/Kvb_eff)^Kn)
E5 11 0 VALUE=
+{(1-FG2)*TWOPI*ATAN(PWR(MAX(V(A,K),0.01)/V(10),KN))}
RE5 11 0 1MEG
*
* E6: Vco = Vg2/λ − ν·Vg1 − w  (crossover voltage)
E6 12 0 VALUE=
+{MAX(V(G2,K)/LAM-NU*V(G,K)-W,0.01)}
RE6 12 0 1MEG
*
* G1: Ia = Ik·α − I_sec
*   I_sec = σ · Ik · (Va/Vg2) · max(1−Va/Vco, 0) · exp(−Ks · max(1−Va/Vco, 0))
G1 A K VALUE=
+{V(8)*V(11)
+-SIGMA*V(8)*(MAX(V(A,K),0.01)/MAX(V(G2,K),0.01))
+*MAX(1-MAX(V(A,K),0.01)/V(12),0)
+*EXP(-KS*MAX(1-MAX(V(A,K),0.01)/V(12),0))}
*
* G2: Ig2 = Ik·(1−α) + I_sec
G2 G2 K VALUE=
+{V(8)*(1-V(11))
++SIGMA*V(8)*(MAX(V(A,K),0.01)/MAX(V(G2,K),0.01))
+*MAX(1-MAX(V(A,K),0.01)/V(12),0)
+*EXP(-KS*MAX(1-MAX(V(A,K),0.01)/V(12),0))}
RCP A K 1G
*
* Interelectrode capacitances
C1 G K {CCG1}    ; cathode-grid1
C4 G2 K {CCG2}   ; cathode-grid2
C5 G2 G {CG1G2}  ; grid1-grid2
C2 A G {CPG1}    ; grid1-plate
C3 A K {CCP}     ; cathode-plate
*
R1 G 5 {RGI}     ; grid current
D3 5 K DX
.MODEL DX D(IS=1N RS=1 CJO=10PF TT=1N)
*
.ENDS DEMPWOLF_BEAMTET
```

**Key differences from v1:**
- Secondary emission uses `σ · Ik · (Va/Vg2) · x · exp(−Ks·x)` (energy-dependent
  yield, proportional to `I_through · Va/Vg2`) instead of v1 `σ · IPK_primary · x · exp(−Ks·x)`
- Crossover voltage `Vco = Vg2/λ − ν·Vg1 − w` depends on grid voltage (was fixed `Vg2`)
- Added parameters: FG2, A, LAM (λ), NU (ν), W
- The secondary emission expressions appear inline in both G1 and G2 sources
  (no separate intermediate node) for numerical stability

---

## 9. Python Implementation

### 9.1 Unified Model Function (`dempwolf_v2`)

The actual implementation uses a `DempwolfParams` dataclass (defined in
`tube_params.py`) and a single function `dempwolf_v2()` in `dempwolf.py`.

```python
from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class DempwolfParams:
    """All 21 parameters for the Dempwolf Extended v2 model."""
    # Cathode emission (triode + pentode)
    mu: float = 100.0          # amplification factor (section A)
    G: float = 2.242e-3        # cathode current perveance (A)
    gamma: float = 1.26        # current exponent (section A)
    C: float = 3.4             # cutoff sharpness (Kp-equiv for pentodes)
    # Grid current
    Gg: float = 6.177e-4       # grid current perveance (A)
    xi: float = 1.314          # grid current exponent
    Cg: float = 9.901          # grid transition sharpness
    # Triode Region A
    Kvb_t: float = 300.0       # Region A parameter (V²)
    # Pentode knee
    Kvb: float = 24.0          # base knee voltage (V)
    Kvb1: float = 0.0          # knee broadening coefficient
    Kn: float = 1.0            # knee sharpness exponent
    # v2 extensions
    fg2: float = 0.0           # G2 geometric interception fraction
    A: float = 0.0             # anode Durchgriff (V⁻¹)
    # Secondary emission (beam tetrodes)
    sigma: float = 0.0         # secondary emission yield
    Ks: float = 4.0            # kink shape parameter
    lam: float = 1.0           # G2 screening factor (λ)
    nu: float = 2.0            # space charge Vco modulation (ν)
    w: float = 0.0             # crossover voltage offset (V)
    # Variable-mu (dual-section)
    mu_b: Optional[float] = None   # amplification factor (section B)
    gamma_b: Optional[float] = None  # current exponent (section B)
    svar: float = 0.0          # section B weight fraction


def dempwolf_v2(
    vpk: float,
    vgk: float,
    vg2k: Optional[float] = None,
    *,
    p: DempwolfParams,
) -> Tuple[float, Optional[float], float]:
    """Compute currents using Dempwolf Extended v2 model.

    Args:
        vpk:  plate-to-cathode voltage (V).
        vgk:  grid-to-cathode voltage (V, negative for normal operation).
        vg2k: screen-to-cathode voltage (V).  None → triode mode.
        p:    model parameters (DempwolfParams).

    Returns:
        (ipk, ig2k, igk) — currents in **amperes**.
        ig2k is None for triodes.
    """
    is_triode = vg2k is None
    vpk_safe = max(vpk, 0.01)
    vg2k_safe = max(vg2k, 0.01) if not is_triode else 0.0

    # --- grid effective voltage (Region A for triodes) ---
    if is_triode:
        v_grid_eff = vgk * vpk_safe / sqrt(p.Kvb_t + vpk_safe**2)
        v_accel = vpk_safe
    else:
        v_grid_eff = vgk
        v_accel = vg2k_safe

    # --- cathode emission ---
    if is_triode:
        # original Dempwolf softplus: ln(1+exp(C·(Va/µ + Vg_eff))) / C
        sp = softplus(p.C * (v_accel / p.mu + v_grid_eff), p.C)
    else:
        # Kp-normalized softplus: (Vg2/C) · ln(1+exp(C·(1/µ + Vgk/Vg2)))
        arg = p.C * (1.0 / p.mu + v_grid_eff / v_accel)
        sp = (v_accel / p.C) * log1p(exp(clip(arg)))

    ik = p.G * max(sp, 0) ** p.gamma

    # --- variable-mu: blend sections A and B ---
    if p.mu_b is not None and p.gamma_b is not None and p.svar > 0:
        # same softplus with mu_b, gamma_b
        ik = (1 - p.svar) * ik + p.svar * ik_b

    # --- Durchgriff (v2, pentodes only) ---
    if not is_triode and p.A > 0:
        ik *= (1.0 + p.A * vpk_safe)

    # --- grid current ---
    igk = p.Gg * max(softplus(p.Cg * vgk, p.Cg), 0) ** p.xi

    if is_triode:
        return max(ik - igk, 0), None, igk

    # --- pentode current splitting ---
    i_through = max(ik - igk, 0)
    v_eff = max(v_accel / p.mu + vgk, 0)
    kvb_eff = max(p.Kvb + p.Kvb1 * v_eff, 0.1)
    alpha = (1 - p.fg2) * (2/pi) * arctan((vpk_safe / kvb_eff) ** p.Kn)

    ipk_primary = i_through * alpha
    ig2k_base = i_through * (1 - alpha)

    # --- secondary emission (beam tetrodes) ---
    if p.sigma > 0 and vg2k_safe > 0.01:
        vco = max(vg2k_safe / p.lam - p.nu * vgk - p.w, 0.01)
        x = max(1 - vpk_safe / vco, 0)
        i_sec = p.sigma * i_through * (vpk_safe / vg2k_safe) * x * exp(-p.Ks * x)
        return ipk_primary - i_sec, ig2k_base + i_sec, igk

    return ipk_primary, ig2k_base, igk
```

### 9.2 Usage — No Convenience Functions

The v2 API uses a single entry point `dempwolf_v2()` for all topologies.
Tube type is determined by which parameters are set in `DempwolfParams`:

```python
# Triode
p = DempwolfParams(mu=100, G=2.242e-3, gamma=1.26, C=3.4,
                   Gg=6.177e-4, xi=1.314, Cg=9.901, Kvb_t=300)
ipk, _, igk = dempwolf_v2(250, -2, p=p)

# Pentode
p = DempwolfParams(mu=11, G=3e-3, gamma=1.35, C=625,
                   Gg=6e-4, xi=1.3, Cg=10,
                   Kvb=24, Kvb1=0.5, Kn=1.0, fg2=0.05, A=2e-4)
ipk, ig2k, igk = dempwolf_v2(250, -2, 250, p=p)

# Beam tetrode (sigma > 0 activates secondary emission)
p = DempwolfParams(mu=7.9, G=3.5e-3, gamma=1.35, C=625,
                   Gg=6e-4, xi=1.3, Cg=10,
                   Kvb=24, Kvb1=0.5, Kn=1.0, fg2=0.05, A=2e-4,
                   sigma=3.0, Ks=1.5, lam=1.0, nu=2.0, w=0.0)
ipk, ig2k, igk = dempwolf_v2(250, -2, 250, p=p)

# Variable-mu pentode (mu_b, gamma_b, svar set)
p = DempwolfParams(mu=28, G=1.5e-3, gamma=1.35, C=500,
                   Gg=4e-4, xi=1.2, Cg=8,
                   Kvb=20, Kvb1=0.3, Kn=1.0, fg2=0.04, A=1e-4,
                   mu_b=8, gamma_b=1.6, svar=0.15)
ipk, ig2k, igk = dempwolf_v2(250, -6, 250, p=p)
```

### 9.5 Numerical Safety

| Issue                     | Solution                              | Why                                    |
|---------------------------|---------------------------------------|----------------------------------------|
| VPK = 0                   | `vpk = max(vpk, 0.01)`               | Division by zero in Region A term      |
| VG2K = 0 (pentode)        | `vg2k = max(vg2k, 0.01)`             | Division by zero in E1 and x           |
| exp() overflow            | `clip(arg, -700, 700)`               | float64 overflows at exp(709); 700 keeps headroom |
| softplus negative         | `max(softplus, 0)` before power       | Rounding errors near cutoff            |
| Kvb_eff → 0               | `max(kvb_eff, 0.1)`                  | Division by zero in arctan argument    |
| x = 1 - Vpk/Vg2k < 0     | `max(x, 0)` clamp                    | No secondary emission above Vg2k      |
| IK < IGK                  | `max(ik - igk, 0)`                   | Can happen at extreme positive Vgk     |

---

## 10. Fitting Strategy

Fitting all parameters simultaneously is numerically unstable due to
parameter correlations. A phased approach is recommended.

### 10.1 Phase 1 — Triode Core

**Data:** Triode-connected measurements (VG2 = VA), VGK < −0.5V
(to avoid grid current influence).

**Fit:** `µ, G, γ, C`

**Method:** Minimize `||Ia_model − Ia_meas||²` using
`scipy.optimize.least_squares` with Trust Region Reflective.

**Rationale:** With VGK well below zero, IGK ≈ 0, so IPK ≈ IK. The
4 cathode emission parameters can be isolated.

### 10.2 Phase 2 — Grid Current

**Data:** Triode-connected measurements with VGK ≥ −0.5V (where
grid current becomes measurable).

**Fit:** `Gg, ξ, Cg` (and optionally `Kvb_t` for Region A)

**Fix:** µ, G, γ, C from Phase 1.

**Note:** If the tester does not measure grid current directly, IGK
can be inferred from the difference between cathode current and plate
current: `IGK = IK − IPK`. However, this requires accurate cathode
current measurement.

**Alternative:** If no grid current data is available, use default
values: `Gg = 6e-4, ξ = 1.3, Cg = 10.0`. These provide reasonable
grid current behavior for most triodes.

### 10.2b Triode Joint Refinement

**Data:** Full triode dataset (all points).

**Fit:** All 8 parameters `µ, G, γ, C, Gg, ξ, Cg, Kvb_t` simultaneously
(`_fit_triode_refine`), seeded by Phases 1–2. Kept only if it improves the
Ia MSE — a dataset can never regress.

**Rationale:** Phases 1–2 alone leave the triode under-fit. Phase 1 freezes
Kvb_t at the default while fitting the cathode params; Phase 2 fits Kvb_t
only on the grid-region subset (VGK > −1 V) — the wrong region for a
Region-A parameter that acts at LOW Va (`v_grid_eff` departs from VGK when
`VPK² ≲ Kvb_t`). On data with no grid region at all (every real LM19 scan)
Kvb_t silently stayed at the default. The joint pass identifies Kvb_t from
the low-Va curvature and re-balances the cathode params against it.
Empirically (2026-07-02, 80 triode datasets): 79 improved, 0 regressed,
median RMS 1.25 → 0.81 mA; battery triodes measured at low Va (DL92–DL98)
improved ~95%.

**Kvb_t multi-start:** C (softplus width) and Kvb_t are correlated in the
low-current region; a single start can settle in a local minimum (observed:
true 1200 → fitted 16). The refine restarts from Kvb_t ∈ {30, 300, 3000} V²
plus the phase-2 value and keeps the best Ia MSE.

**Grid-parameter freeze:** When the data has no grid region (< 3 points
with VGK > −1 V), `Gg, ξ, Cg` are frozen at the defaults during the refine:
with IGK ≈ 0 everywhere they are unidentifiable and would wander to the
bounds, exporting fictional grid current into SPICE models.

### 10.3 Phase 3 — Pentode Knee

**Data:** Pentode-mode measurements (VG2 = const ≠ VA), focusing
on the knee region (VPK < 2 × VG2K).

**Fit:** `Kvb, Kvb1, fg2, A, Kn` (5 parameters)

**Fix:** µ, G, γ, C from Phase 1; Gg, ξ, Cg from Phase 2.

**Initial estimates:**
- fg2: estimated from high-Va data as `median(Ig2 / (Ia + Ig2))` where `VPK > 2·VG2K`
- A: `0.0002` (small Durchgriff)
- Kn: `1.0` (standard smooth knee)

**Rationale:** Kvb, Kvb1, and Kn control the knee shape (α), fg2 controls the
geometric screen interception fraction, and A controls the anode Durchgriff.
All five affect the Ia/Ig2 current splitting but not the total cathode emission.
By fixing the emission parameters, the fitting is well-conditioned.
See also §14.6 for v2 fitting strategy details.

### 10.4 Phase 4 — Full Pentode Refinement

**Data:** Full pentode-mode dataset (all VPK, VGK, VG2K).

**Fit:** All 10 parameters simultaneously.

**Initial guess:** Results from Phases 1–3.

**Rationale:** Phases 1–3 provide an excellent starting point. The
simultaneous fit allows small adjustments to all parameters for the
best overall fit to pentode-mode data.

### 10.5 Phase 5 — Secondary Emission (Beam Tetrodes)

**Data:** Measurements in the kink region (VPK < VG2K), where
plate current shows non-monotonic behavior.

**Fit:** `σ, Ks, λ, ν, w` (v2; see §14.6)

**Fix:** All other parameters from Phase 4.

**Detection (`_has_kink`):** Points are grouped into per-curve buckets by
BOTH VGK (0.1 V) and VG2K (1 V) — Ug1-only grouping interleaves the Ug2
levels of a multi-Ug2 scan and reads the level steps as dips (historically
every real LM19 pentode scan was classified as a beam tetrode and received
fictional σ ≈ 0.2). The kink metric is the cumulative **drawdown** (running
Ia peak minus Ia along ascending Ua) within VPK < VG2K, not the per-step
diff: a dynatron valley descends across many Ua steps while noise recovers
immediately. A curve is kinked when the drawdown exceeds
`max(5% × curve max, 0.1 mA)`. Measured populations behind the thresholds:
true-pentode scan/trace artifacts ≤ 2.6% of curve max (EL84), real dynatron
valleys ≥ 16% (6P1P real scan, 6L6-class synthetics ~24%).

Detection gates **phase 5 only**. It is decoupled from the below-knee
masking of phases 1–2, which is gated on Ig2 availability instead: with
Ig2 data the cathode target `Ik = Ia + Ig2` is knee-independent (exact
current conservation) and the full dataset is usable; without Ig2 the
Ia-only target equals `Ik·α` and below-knee points (α < 1) are masked.
Phases 3–4 ALWAYS see the full dataset — the knee is what they fit.
(Historically masking was tied to kink detection for phases 1–4, so knee
parameters were fit without knee data and phase-5 σ acted as a knee-shaping
crutch on true pentodes.)

**Multi-start:** The 5-parameter landscape is multimodal — a single start
can settle in the σ ≈ 0 minimum even with a deep kink present (observed on
the real 6P1P scan). Phase 5 restarts over λ ∈ {1, 15} (see §14.6) ×
ν ∈ {1, 4, 8} and keeps the best residual cost.

### 10.6 Parameter Bounds

As implemented (`lm19/dempwolf.py`). Phase 1 uses mu-adaptive branch bounds
(low-mu power triodes get G up to 0.5 A and C down to 0.05); the triode
joint refine uses the union of all branches:

**Triode joint refine (`_TRIODE_REFINE_LO/HI`):**

```
µ:     [1.5,     500.0]
G:     [1e-5,    0.5]      A
γ:     [0.8,     2.5]
C:     [0.05,    50.0]          (original Dempwolf sharpness)
Gg:    [1e-6,    1e-2]     A
ξ:     [0.8,     2.5]
Cg:    [1.0,     50.0]
Kvb_t: [10.0,    5000.0]   V²
```

**Pentode phases 3–4 and beam joint refine (`_PENTODE_BOUNDS_LO/HI`,
single source — the beam joint refine historically carried divergent
literals with fg2 up to 0.5 and A up to 0.01, far past the perturbative
Durchgriff regime of §14.2):**

```
µ:     [2.0,     500.0]
G:     [1e-5,    1e-1]     A
γ:     [0.8,     2.5]
C:     [1.0,     2000.0]        (Kp-equivalent, ~50–750 typical)
Gg:    [1e-6,    1e-1]     A
ξ:     [0.8,     2.5]
Cg:    [1.0,     50.0]
Kvb:   [0.5,     200.0]    V
Kvb1:  [0.0,     5.0]
fg2:   [0.0,     0.30]
A:     [0.0,     0.001]    V⁻¹
Kn:    [0.7,     2.0]
```

**Beam tetrode phase 5 + joint refine (`_SEC_EMISSION_BOUNDS_LO/HI`;
λ/ν/w discussion in §14.7):**

```
σ:     [0.0,     10.0]          (joint refine floor: 0.01)
Ks:    [0.1,     20.0]
λ:     [0.5,     25.0]
ν:     [0.5,     12.0]
w:     [-50.0,   50.0]     V
```

### 10.7 Residual Functions

Residuals are plain ampere-domain differences (uniform normalization by a
constant does not move the least-squares minimum, so no peak-normalization
is applied):

**Triode (Ia only, Phases 1–2 and joint refine):**

```python
residual = ia_model_A - ia_meas_A
```

**Pentode (Ia + Ig2, Phases 3–5 and joint refine):**

```python
r_ia  = ia_pred_A - ia_A
r_ig2 = (ig2_pred_A - ig2_A) * sqrt(w_ig2)
residual = np.concatenate([r_ia, r_ig2])
```

Where `w_ig2 = 0.3` — screen current gets lower weight because it is less
critical and less accurately measured. Note the weight applies in the raw
ampere domain: since Ig2 is typically ~10× smaller than Ia, its effective
influence is lower than a per-channel-normalized scheme would give.

### 10.8 Fit Quality Metrics

```python
ia_diff_mA = (ia_model - ia_meas) * 1000.0
rms_error  = sqrt(mean(ia_diff_mA²))
max_error  = max(|ia_diff_mA|)
r_squared  = 1 - sum((ia_meas - ia_model)²) / sum((ia_meas - mean(ia_meas))²)
```

**Typical good fit:**

| Tube type          | RMS (mA) | Max (mA) | R²      |
|--------------------|----------|----------|---------|
| Small-signal triode| < 0.1    | < 0.5    | > 0.999 |
| Power triode       | < 1      | < 5      | > 0.998 |
| Power pentode      | < 2      | < 10     | > 0.995 |
| Beam tetrode (kink)| < 3      | < 15     | > 0.990 |

**Metric consistency:** the reported RMS/max are computed by the SAME
formula the shipped model evaluates (grid current subtracted). A historical
fitting-only kernel omitted IGK from the final triode metric and overstated
RMS up to 7× on tubes with grid-region data (real 6N5P: reported 18.2 mA
vs actual 2.5 mA), biasing the fit benchmark against Dempwolf on triodes.

**Known limitation — emission saturation:** transmitting-class tubes
measured at low Va with multi-ampere currents (e.g. THF51: Va ≤ 60 V,
Ia up to 6 A) operate in the emission-saturation regime. The space-charge
law behind both Dempwolf and Koren does not model saturation, so both fail
comparably there (benchmark: Koren RMS 1707 mA, Dempwolf 904 mA); the
parameters pin at the bounds. This is a model-class mismatch, not a bounds
or fitter defect — the dataset is kept in the benchmark as an out-of-class
outlier.

---

## 11. Verification Examples

### 11.1 12AX7 Triode

Published Dempwolf parameters:
```
µ=100, G=2.242e-3, γ=1.26, C=3.4, Gg=6.177e-4, ξ=1.314, Cg=9.901
Kvb_t=300 (estimated, not in original paper)
```

**Test point: VPK=250V, VGK=−2V**

```
V_grid_eff = -2 * 250 / sqrt(300 + 62500) = -2 * 250 / 250.6 = -1.995V
arg_k = 3.4 * (250/100 + (-1.995)) = 3.4 * (2.5 - 1.995) = 3.4 * 0.505 = 1.717
softplus = ln(1 + exp(1.717)) / 3.4 = ln(1 + 5.567) / 3.4 = 1.882 / 3.4 = 0.554
IK = 2.242e-3 * 0.554^1.26 = 2.242e-3 * 0.461 = 1.034e-3 A

arg_g = 9.901 * (-2) = -19.80
softplus_g = ln(1 + exp(-19.80)) / 9.901 ≈ 0
IGK ≈ 0

IPK = IK - IGK ≈ 1.03 mA
```

Koren gives 0.95 mA, datasheet ≈ 1.0 mA. Both models are within
measurement tolerance.

**Test point: VPK=250V, VGK=+0.5V (grid conduction)**

```
arg_k = 3.4 * (2.5 + 0.499) ≈ 3.4 * 2.999 = 10.20
softplus = ln(1+exp(10.20)) / 3.4 ≈ 10.20 / 3.4 = 3.0
IK = 2.242e-3 * 3.0^1.26 = 2.242e-3 * 3.89 = 8.72e-3 A = 8.72 mA

arg_g = 9.901 * 0.5 = 4.95
softplus_g = ln(1+exp(4.95)) / 9.901 = 5.00 / 9.901 = 0.505
IGK = 6.177e-4 * 0.505^1.314 = 6.177e-4 * 0.434 = 2.68e-4 A = 0.27 mA

IPK = 8.72 - 0.27 = 8.45 mA
```

Grid current of 0.27 mA at VGK = +0.5V is physically reasonable.
Koren's diode model gives either 0 or a crude step depending on the
diode parameters.

### 11.2 EL34 Pentode

Estimated parameters (Kp-normalized C, fitted from data):
```
µ=11, G=3.0e-3, γ=1.35, C=750.0, Gg=6e-4, ξ=1.3, Cg=10.0
Kvb=24, Kvb1=0.5, Kn=1.0
```

**Test point: VPK=300V, VG2K=300V, VGK=0V**

Using Kp-normalized pentode formula:
```
arg_k = 750.0 * (1/11 + 0/300) = 750.0 * 0.0909 = 68.18
softplus = (300/750.0) * ln(1+exp(68.18)) ≈ 0.4 * 68.18 = 27.27
IK = 3.0e-3 * 27.27^1.35 = 3.0e-3 * 85.0 = 0.255 A

IGK ≈ 0 (VGK = 0, softplus_g = ln(2)/10 = 0.069, IGK ≈ 4e-5 A ≈ 0)

I_through = 0.255 A

V_eff = max(300/11 + 0, 0) = 27.27
Kvb_eff = 24 + 0.5 * 27.27 = 37.6 V
α = (2/π) * arctan(300/37.6) = 0.637 * 1.448 = 0.922

IPK  = 0.255 * 0.922 = 0.235 A = 235 mA
IG2K = 0.255 * 0.078 = 0.020 A = 20 mA
```

Note: C=750 with VG2K=300 gives the same result as the old C=2.5,
because `C_new = C_old × VG2K` → `750 = 2.5 × 300`.

**Test point: VPK=300V, VG2K=300V, VGK=−20V**

```
arg_k = 750.0 * (1/11 + (-20)/300) = 750.0 * (0.0909 - 0.0667) = 18.18
softplus = (300/750.0) * ln(1+exp(18.18)) ≈ 0.4 * 18.18 = 7.27
IK = 3.0e-3 * 7.27^1.35 = 3.0e-3 * 14.2 = 0.0426 A

V_eff = max(27.27 + (-20), 0) = 7.27
Kvb_eff = 24 + 0.5 * 7.27 = 27.6 V           ← narrower knee
α = (2/π) * arctan(300/27.6) = 0.637 * 1.479 = 0.942

IPK  = 0.0426 * 0.942 = 0.0401 A = 40.1 mA
IG2K = 0.0426 * 0.058 = 0.0025 A = 2.5 mA
```

Note how Kvb_eff decreased from 37.6V (at VGK=0) to 27.6V (at VGK=−20),
making the knee sharper at lower currents — matching real tube behavior.

### 11.3 6L6GC Beam Tetrode

Estimated parameters (Kp-normalized C):
```
µ=8.7, G=2.8e-3, γ=1.35, C=625.0, Gg=6e-4, ξ=1.3, Cg=10.0
Kvb=12, Kvb1=0.3, Kn=1.0, σ=3.0, Ks=1.5
```

**Test point in kink region: VPK=100V, VG2K=250V, VGK=0V**

```
arg_k = 625.0 * (1/8.7 + 0/250) = 625.0 * 0.1149 = 71.8
softplus = (250/625.0) * 71.8 ≈ 0.4 * 71.8 = 28.74
IK = 2.8e-3 * 28.74^1.35 = 2.8e-3 * 91.0 = 0.255 A

I_through ≈ 0.255 A

V_eff = 28.74
Kvb_eff = 12 + 0.3 * 28.74 = 20.6 V
α = (2/π) * arctan(100/20.6) = 0.637 * 1.373 = 0.875

IPK_primary = 0.255 * 0.875 = 0.223 A

x = max(1 - 100/250, 0) = 0.60
I_sec = 3.0 * 0.223 * 0.60 * exp(-1.5 * 0.60)
      = 3.0 * 0.223 * 0.60 * 0.407
      = 0.163 A = 163.2 mA       ← NOTE: simplified formula (see §14.5 for full)

IPK  = 223.0 - 163.2 = 59.8 mA
IG2K = 0.255*0.125 + 163.2 = 31.9 + 163.2 = 195.1 mA
```

The larger σ (3.0 vs old 0.25) produces a very strong kink, characteristic
of beam tetrodes like 6L6GC. The actual implementation (§14.5) uses a
more refined formula with additional parameters (λ, ν, w) for crossover
voltage and `I_through · VPK/VG2K` scaling.

**Test point at kink peak: VPK=190V, VG2K=250V, VGK=0V**

```
α = (2/π) * arctan(190/20.6) = 0.637 * 1.463 = 0.932
IPK_primary = 0.255 * 0.932 = 0.238 A

x = max(1 - 190/250, 0) = 0.24
I_sec = 3.0 * 0.238 * 0.24 * exp(-1.5 * 0.24)
      = 3.0 * 0.238 * 0.24 * 0.698
      = 0.119 A = 119.4 mA

IPK  = 238.0 - 119.4 = 118.6 mA
IG2K = 0.255*0.068 + 119.4 = 17.3 + 119.4 = 136.7 mA
```

Peak secondary emission occurs near x = 1/Ks = 0.25, i.e., VPK ≈ 190V.
This creates a subtle dip in the Ia-Va characteristic at that voltage.

---

## 12. Edge Cases and Numerical Safety

1. **VPK = 0:** `max(vpk, 0.01)` prevents division by zero in
   Region A term and arctan argument.

2. **VG2K = 0:** `max(vg2k, 0.01)` prevents division by zero in
   cathode emission and secondary emission.

3. **exp() overflow:** `clip(arg, -700, 700)` — float64 overflows at
   `exp(709)`, 700 keeps headroom. Until 2026-07-11 the clip sat at ±50,
   which **saturated** the Kp-normalized softplus in-range for
   hand-curated high-C parameter sets: `arg = C·(1/µ + Vg/Vg2)` crosses
   50 already at Vg ≈ −6 V for C = 750, µ = 11, Vg2 = 250 — flattening
   gm to ≈ 0 near Vg = 0. Fitted models never reached the zone (real
   fits give C ≤ ~120, arg ≤ ~5), so the fix is bit-identical for them.

4. **Negative softplus:** Due to floating-point rounding, softplus
   can be very slightly negative near cutoff. `max(softplus, 0)` before
   applying the power function prevents NaN.

5. **Kvb_eff → 0:** When both Kvb and V_eff are near zero,
   `max(kvb_eff, 0.1)` prevents arctan(∞).

6. **IK < IGK:** At extreme positive VGK, grid current can exceed the
   modeled cathode current (because IGK is computed independently).
   `max(ik - igk, 0)` prevents negative I_through.

7. **IPK negative:** In the kink region, I_sec can theoretically
   exceed IPK_primary for very high σ. This is physically possible
   (net electron emission from plate) but may cause SPICE convergence
   issues. For SPICE, consider `max(IPK, -0.01)` if needed.

8. **Very small currents:** Below ~0.1 mA, measurement noise dominates.
   Filter these points from fitting data.

9. **Mixed topology data:** Separate VG2=VA points (triode mode) from
   independent-VG2 points (pentode mode) before fitting.

---

## 13. Topology Detection

> **Not yet implemented.** This section describes the planned design.

### 13.1 Current project state

Both `lamps.json` and `tube_params.json` use only two topology values:

| Value | Covers |
|---|---|
| `"triode"` | True triodes (12AX7, 6SN7, 6SL7, …) |
| `"pentode"` | Pentodes (EL34, EF86), beam tetrodes (6L6, KT88, 6V6), tetrodes |

There is **no** `"beam_tetrode"` value in the configs today. Beam tetrodes
are classified as `"pentode"` because from the measurement and Koren model
perspective they are identical — both have Ug2, Ig2, and the same equations.

The `ug2_mode` field in `lamps.json` distinguishes operating modes:

| ug2_mode | Meaning |
|---|---|
| `"triode"` | True triode, Ug2 = 0, hidden |
| `"triode_connected"` | Pentode/tetrode wired as triode, Ug2 = Ua + offset |
| `"pentode"` | Independent Ug2 (pentode or beam tetrode) |

### 13.2 Planned: detect_topology()

```python
def detect_topology(points, lamp_config=None, tube_params_entry=None):
    """Detect tube topology for Dempwolf model selection.

    Returns:
        "triode"            — true triode (no G2)
        "pentode"           — true pentode (G3 present, no kink)
        "beam_tetrode"      — beam tetrode (no G3, kink possible)
        "variable_mu"       — variable-mu (remote cutoff) pentode
        "triode_connected"  — pentode/tetrode wired as triode
    """
```

Detection logic (priority order):

1. **From tube_params.json** — if entry has explicit `"dempwolf_topology"`
   field (future extension), use it directly. Supports all five values
   including `"variable_mu"`.

2. **From lamp_config (lamps.json):**
   - `topology == "triode"` → `"triode"`
   - `topology == "pentode"` and `ug2_mode == "triode_connected"` → `"triode_connected"`
   - `topology == "pentode"` and `variable_mu == true` → `"variable_mu"`
   - `topology == "pentode"` → proceed to data analysis (step 3)

3. **From measurement data** (only for `topology == "pentode"`):
   - If `dIa/dVa < 0` observed at any VPK < VG2K → `"beam_tetrode"`
   - If cutoff curve is notably non-linear (remote cutoff signature) →
     `"variable_mu"` (heuristic: Ia at Vg = −2×Vg_cutoff_50% > threshold)
   - Otherwise → `"pentode"`

4. **Model selection:**
   - `"triode"` / `"triode_connected"` → Dempwolf triode (8 params)
   - `"pentode"` → Dempwolf pentode (10 params, σ = 0)
   - `"variable_mu"` → Dempwolf variable-mu pentode (13 params)
   - `"beam_tetrode"` → Dempwolf beam tetrode (12 params, σ > 0)

### 13.3 Known beam tetrodes in tube_params.json

These tubes are currently listed as `topology: "pentode"` but are
physically beam tetrodes (would use σ > 0 in the Dempwolf model):

- **6L6** (6L6GC, 5881, 7027)
- **KT88** (6550, KT90)
- **6V6** (6V6GT, 6V6S)
- **KT66**
- **6F6**

When the Dempwolf model is implemented, these can either:
- Get an explicit `"dempwolf_topology": "beam_tetrode"` field in
  `tube_params.json`, or
- Be auto-detected from measurement data via the kink test (step 3 above)

---

## 14. Dempwolf Extended v2 — Improved Model

> Version 2 addresses five specific weaknesses identified in v1 through
> analysis of Reefman's Derk/DerkE models (Theory.pdf, January 2016)
> and first-principles physics review.

### 14.1 Problem: Screen Current Vanishes at High Va

**The most critical defect in v1.**

In v1, the current splitting fraction is:

```
α = (2/π) · arctan(VPK / Kvb_eff)
```

As VPK → ∞: α → 1, therefore IG2K → 0. This is **unphysical**. The
screen grid consists of physical wires that geometrically intercept a
fraction of the electron beam regardless of plate voltage. For typical
pentodes, IG2K / (IPK + IG2K) ≈ 5–15% even at high VPK.

In Reefman's Derk model this is correct:
`Ig2(Va→∞) → IP_Koren/Kg2 > 0` (constant).

**Fix:** Introduce **fg2** — geometric interception fraction:

```
α = (1 − fg2) · (2/π) · arctan(VPK / Kvb_eff)

IPK  = I_through · α
IG2K = I_through · (1 − α)
```

Limits:
- VPK → ∞: α → (1 − fg2), IG2K → I_through · fg2 > 0  ✓
- VPK → 0: α → 0, IG2K → I_through (all current to G2)  ✓
- Current conservation: IPK + IG2K = I_through  ✓

Typical values: fg2 = 0.05–0.15 (5–15%).

Related to Derk's parameters: `fg2 = Kg1/Kg2` — the exact Va→∞ limit
of Derk at A = 0 (Ig2 → IP/Kg2 while Ik → IP/Kg1).

### 14.2 Anode Durchgriff (Penetration Factor)

**Physical effect:** The electric field from the plate penetrates
through G2 and G3, slightly modulating cathode emission. For small-
signal pentodes Da ≈ 0 (G3 screens effectively). For power pentodes
and beam tetrodes Da > 0.

Reefman models this as: `Ik = IP_Koren/Kg1 · (1 + A·Va)`.

**v2 equation:**

```
IK_total = IK · (1 + A · VPK)
```

Where A is typically 0.0001–0.001 V⁻¹. At A = 0 the equation reduces
to v1 (no penetration).

**Constraint:** A·VPK should remain << 1 (perturbative regime).
At A = 0.001 and VPK = 500V, the correction is 50% — too large.
Practical limit: A < 0.0005 for VPK_max = 500V (25% max correction).

### 14.3 Grid-Voltage-Dependent Crossover (Secondary Emission)

**Physical effect:** The voltage at which secondary electrons return
to the anode (crossover voltage Vco) depends on:

1. **VG2K** — higher screen voltage attracts more secondaries (Vco ∝ VG2K)
2. **VGK** — more negative grid → less space charge → weaker suppression
   of secondaries → higher Vco (Vco ∝ −VGK)
3. **Geometry** — suppressor grid / beam-forming plates (constant offset w)

Reefman's formulation (Theory.pdf §6.2):

```
Vco = VG2K/λ − ν·VGK − w
```

Where:
- λ — G2 screening factor (≈1 for beam tetrodes, ≈10–20 for pentodes)
- ν — space charge modulation (typically 1–4)
- w — geometric offset (V)

**v2 replaces** the v1 kink variable `x = max(1 − VPK/VG2K, 0)` with:

```
Vco = VG2K/λ − ν·VGK − w
x   = max(1 − VPK / Vco, 0)
```

This causes the kink to shift with grid voltage — matching real
measurements of beam tetrodes where the kink moves to higher VPK
at more negative VGK.

### 14.4 Energy-Dependent Secondary Emission

**Physical effect:** The number of secondary electrons ejected from
the anode depends on both (a) the number of primary electrons hitting
it and (b) their kinetic energy at impact (proportional to VPK).

v1 uses: `I_sec = σ · IPK_primary · x · exp(−Ks·x)` — proportional
to primary current only.

Reefman uses: `Psec = S · Va · (1 + tanh(−ap·(Va − Vco)))` —
proportional to energy only.

**v2 hybrid — proportional to both (power deposited on anode):**

```
I_sec = σ · I_through · (VPK / VG2K) · x · exp(−Ks · x)
```

Where `VPK/VG2K` is the normalized electron energy (= 1 when
electrons arrive with screen-accelerated energy, < 1 in the knee,
> 1 above screen voltage).

The `I_through` factor (instead of `IPK_primary`) avoids circular
dependency: IPK_primary already depends on α which changes near the
knee.

### 14.5 v2 Complete Equations

#### 14.5.1 Triode (8 parameters — unchanged from v1)

Parameters: `µ, G, γ, C, Gg, ξ, Cg, Kvb_t`

```
V_grid_eff = VGK · VPK / sqrt(Kvb_t + VPK²)

IK  = G  · (ln(1 + exp(C  · (VPK/µ + V_grid_eff))) / C )^γ
IGK = Gg · (ln(1 + exp(Cg · VGK))                   / Cg)^ξ

IPK = max(IK − IGK, 0)
```

No changes from v1. The triode model does not have the Ig2 problem,
and Durchgriff is not applicable (plate IS the accelerating electrode).

#### 14.5.2 Pentode (12 parameters)

Parameters: `µ, G, γ, C, Gg, ξ, Cg, Kvb, Kvb1, Kn, fg2, A`

```
IK  = G  · ((VG2K/C) · ln(1 + exp(C · (1/µ + VGK/VG2K))))^γ
IGK = Gg · (ln(1 + exp(Cg · VGK))             / Cg)^ξ

IK_total = IK · (1 + A · VPK)
I_through = max(IK_total − IGK, 0)

V_eff   = max(VG2K/µ + VGK, 0)
Kvb_eff = Kvb + Kvb1 · V_eff

α = (1 − fg2) · (2/π) · arctan((VPK / Kvb_eff)^Kn)

IPK  = I_through · α
IG2K = I_through · (1 − α)
```

Changes from v1: +fg2 (geometric interception), +A (Durchgriff),
Kp-normalized softplus (C is now Kp-equivalent, ~50–750).

#### 14.5.3 Beam Tetrode (17 parameters)

Parameters: `µ, G, γ, C, Gg, ξ, Cg, Kvb, Kvb1, Kn, fg2, A, σ, Ks, λ, ν, w`

```
IK  = G  · ((VG2K/C) · ln(1 + exp(C · (1/µ + VGK/VG2K))))^γ
IGK = Gg · (ln(1 + exp(Cg · VGK))             / Cg)^ξ

IK_total = IK · (1 + A · VPK)
I_through = max(IK_total − IGK, 0)

V_eff   = max(VG2K/µ + VGK, 0)
Kvb_eff = Kvb + Kvb1 · V_eff

α = (1 − fg2) · (2/π) · arctan((VPK / Kvb_eff)^Kn)

IPK_primary = I_through · α
IG2K_base   = I_through · (1 − α)

Vco   = VG2K/λ − ν·VGK − w
x     = max(1 − VPK / max(Vco, 0.01), 0)
I_sec = σ · I_through · (VPK / max(VG2K, 0.01)) · x · exp(−Ks · x)

IPK  = IPK_primary − I_sec
IG2K = IG2K_base   + I_sec
```

#### 14.5.4 Variable-mu Pentode (15 parameters)

Parameters: `µ_a (≡ µ), γ_a (≡ γ), G, C, Gg, ξ, Cg, Kvb, Kvb1, Kn, fg2, A, µ_b, γ_b, svar`

Same dual-section approach as v1 (§6.4), combined with v2 extensions:

```
IK_a = G · ((VG2K/C) · ln(1 + exp(C · (1/µ_a + VGK/VG2K))))^γ_a
IK_b = G · ((VG2K/C) · ln(1 + exp(C · (1/µ_b + VGK/VG2K))))^γ_b

IK  = (1 − svar) · IK_a  +  svar · IK_b
IGK = Gg · (ln(1 + exp(Cg · VGK)) / Cg)^ξ

IK_total = IK · (1 + A · VPK)
I_through = max(IK_total − IGK, 0)

V_eff   = max(VG2K/µ_a + VGK, 0)
Kvb_eff = Kvb + Kvb1 · V_eff

α = (1 − fg2) · (2/π) · arctan((VPK / Kvb_eff)^Kn)

IPK  = I_through · α
IG2K = I_through · (1 − α)
```

Changes from v1 variable-mu: +fg2 (geometric interception), +A (Durchgriff).

#### 14.5.5 v2 Parameter Summary Table

| #  | Parameter | Units | Triode | Pentode | Var-mu | Beam Tet. | Description                    | New |
|----|-----------|-------|:------:|:-------:|:------:|:---------:|--------------------------------|:---:|
| 1  | µ (µ_a)   | —     | +      | +       | +      | +         | Amplification factor (sect. A) |     |
| 2  | G         | A     | +      | +       | +      | +         | Cathode current perveance      |     |
| 3  | γ (γ_a)   | —     | +      | +       | +      | +         | Current exponent (section A)   |     |
| 4  | C         | —     | +      | +       | +      | +         | Cutoff sharpness (Kp-equiv for pent.) |     |
| 5  | Gg        | A     | +      | +       | +      | +         | Grid current perveance         |     |
| 6  | ξ         | —     | +      | +       | +      | +         | Grid current exponent          |     |
| 7  | Cg        | —     | +      | +       | +      | +         | Grid transition sharpness      |     |
| 8  | Kvb_t     | V²    | +      | —       | —      | —         | Region A parameter (triode)    |     |
| 9  | Kvb       | V     | —      | +       | +      | +         | Base knee voltage              |     |
| 10 | Kvb1      | —     | —      | +       | +      | +         | Knee broadening coefficient    |     |
| 11 | Kn        | —     | —      | +       | +      | +         | Knee sharpness exponent        |     |
| 12 | fg2       | —     | —      | +       | +      | +         | G2 geometric interception frac | v2  |
| 13 | A         | V⁻¹   | —      | +       | +      | +         | Anode Durchgriff coefficient   | v2  |
| 14 | µ_b       | —     | —      | —       | +      | —         | Amplification factor (sect. B) |     |
| 15 | γ_b       | —     | —      | —       | +      | —         | Current exponent (section B)   |     |
| 16 | svar      | —     | —      | —       | +      | —         | Section B weight fraction      |     |
| 17 | σ         | —     | —      | —       | —      | +         | Secondary emission yield       |     |
| 18 | Ks        | —     | —      | —       | —      | +         | Kink shape parameter           |     |
| 19 | λ         | —     | —      | —       | —      | +         | G2 screening factor            | v2  |
| 20 | ν         | —     | —      | —       | —      | +         | Space charge Vco modulation    | v2  |
| 21 | w         | V     | —      | —       | —      | +         | Crossover voltage offset       | v2  |
|    | **Total** |       | **8**  | **12**  | **15** | **17**    |                                |     |

#### 14.5.6 v2 Model Hierarchy

```
Variable-mu Pentode (15 params)
  │ set svar = 0, drop µ_b, γ_b
  ▼
Pentode (12 params)          Beam Tetrode (17 params)
  │ set VG2K = VPK,            │ set σ = 0, drop λ, ν, w, Ks
  │ fg2 = 0, A = 0,            ▼
  │ Kn = 1, Kvb → ∞           Pentode (12 params)
  ▼                              │ ...
Triode (8 params)
```

Variable-mu and Beam Tetrode are independent extensions of the
Pentode base.

#### 14.5.7 v2 Current Conservation Proof

**Pentode (v2):**
```
IK_total = IK · (1 + A·VPK)
I_through = IK_total − IGK

IPK + IG2K + IGK
= I_through·α + I_through·(1−α) + IGK
= I_through + IGK
= IK_total  ✓
```

**Beam tetrode (v2):**
```
IPK + IG2K + IGK
= (IPK_primary − I_sec) + (IG2K_base + I_sec) + IGK
= IPK_primary + IG2K_base + IGK
= I_through·α + I_through·(1−α) + IGK
= I_through + IGK
= IK_total  ✓
```

Secondary emission redistributes current between plate and screen
without changing total cathode current. The Durchgriff term (1+A·VPK)
modifies total emission but does not break conservation.

#### 14.5.8 Default Values for Optional Parameters

When data is insufficient to fit all parameters, these defaults
provide reasonable behavior:

| Parameter | Default  | Effect at default                          |
|-----------|----------|--------------------------------------------|
| fg2       | 0.0      | Reverts to v1 behavior (α up to 1.0)       |
| A         | 0.0      | No Durchgriff (constant space charge)      |
| Kn        | 1.0      | Standard smooth knee (arctan)              |
| σ         | 0.0      | No secondary emission (true pentode)       |
| λ         | 1.0      | Vco ≈ VG2K (beam tetrode, no screening)    |
| ν         | 2.0      | Moderate space charge effect on Vco         |
| w         | 0.0      | No geometric offset                        |
| Ks        | 4.0      | Kink peak at x = 0.25 (VPK ≈ 0.75·Vco)    |
| Kvb1      | 0.0      | Fixed knee shape (reverts to Koren-like)   |
| svar      | 0.0      | Single-section (no variable-mu)            |

With all defaults, pentode v2 reduces to v1 (which reduces to Koren-
like behavior when also Gg → 0 and C → Kp).

### 14.6 Fitting Strategy Changes for v2

Phase 3 (Pentode Knee) now fits 5 parameters: `Kvb, Kvb1, fg2, A, Kn`.

**fg2 initial estimate:**
```
fg2_est = median(Ig2_meas / (Ia_meas + Ig2_meas))  at VPK > 2·VG2K
```
This selects the high-Va region where the geometric fraction dominates.

**A initial estimate:**
```
Ik_low  = median(Ia + Ig2)  at VPK < 100V
Ik_high = median(Ia + Ig2)  at VPK > 300V
A_est = (Ik_high/Ik_low − 1) / 300
```

Phase 5 (Secondary Emission) now fits 5 parameters: `σ, Ks, λ, ν, w`.

**λ / ν starts:** the landscape is multimodal, so phase 5 multi-starts over
λ ∈ {1.0, 15.0} (beam-tetrode vs pentode-like screening) × ν ∈ {1, 4, 8}
and keeps the best residual cost (see §10.5). w starts at 0.

**A initial estimate:** fixed x0 = 2e-4 (the formula-based estimate from
`(Ik_high/Ik_low − 1)/300` was considered but the optimizer recovers A
reliably from the flat-region slope; not implemented).

### 14.7 Additional Parameter Bounds (v2)

```
fg2:  [0.0,     0.30]
A:    [0.0,     0.001]     V⁻¹
λ:    [0.5,     25.0]
ν:    [0.5,     12.0]
w:    [-50.0,   50.0]      V
```

ν extends past the "typical 1–4" theory range: the real 6P1P beam-tetrode
scan fits ν = 7.8 (kink shift vs VGK), and clamping ν at 6 collapses the
whole secondary-emission fit to σ ≈ 0 (kink lost, RMS 0.69 → 1.05 mA).

---

## 15. Comparison with Koren and Derk/DerkE Models

This section provides a detailed comparison of five models: the
original Koren (1996), the Derk and DerkE models from ExtractModel
(Reefman 2014–2016), and our Dempwolf Extended v1 and v2.

**References:**
- D. Reefman, "Spice models for vacuum tubes using the uTracer",
  Theory.pdf, January 2016 (Derk/DerkE theory and equations).
- ExtractModel v3.0 (February 2016) — parameter extraction program.
- TubeLib.inc — LTspice library with 157+ models fitted by ExtractModel.

### 15.1 Koren Model (1996)

**Triode:**
```
E1 = (Va/Kp) · ln(1 + exp(Kp·(1/µ + Vg/√(Kvb + Va²))))
Ia = E1^Ex / Kg1   (for E1 > 0)
```

**Pentode:**
```
E1 = (Vg2/Kp) · ln(1 + exp(Kp·(1/µ + Vg/Vg2)))
Ia = E1^Ex / Kg1 · arctan(Va/Kvb)
Ig2 = (Vg + Vg2/µ)^(3/2) / Kg2   (independent of Va)
```

**Known issues** (identified by Reefman, Theory.pdf §3.4):

1. **Kg1/Kp correlation** — eigenanalysis of the Hessian at convergence
   shows that Kg1 and Kp are not orthogonal (eigenvalue ratio > 10⁵).
   Different Kg1/Kp pairs give almost identical DC fit but different
   derivatives → unreliable gm/rp prediction.
2. **Ig2 independent of Va** — the screen current equation has no Va
   term. Real screen current decreases with Va (electrons pulled to
   plate). This means Ia + Ig2 ≠ constant (no space charge conservation).
3. **Pentode-as-triode inconsistency** — setting Vg2 = Va in the pentode
   equations gives a different current-voltage relationship than the
   triode equations (different scaling law).
4. **No secondary emission** — monotonic arctan cannot produce kinks.
5. **Kvb dual meaning** — in the triode model Kvb is in V² (Region A),
   in the pentode model Kvb is in V (knee). Same name, different physics.

### 15.2 Derk Model (Reefman 2014)

**Key innovation:** Constant Space Charge — total cathode current
Ik = Ia + Ig2 is (nearly) independent of Va.

```
IP_Koren = E1^Ex / 2
E1 = (Vg2/Kp) · ln(1 + exp(Kp·(1/µ + Vg/√(Kvb + Vg2²))))

Ik = IP_Koren/Kg1 · (1 + A·Va)
Ig2 = IP_Koren/Kg2 · (1 + αs/(1 + β·Va))
Ia = Ik − Ig2                          (from balance)

Constraint: α = 1 − Kg1/Kg2·(1 + αs)  (ensures Ia(0) = 0)
```

**Parameters:** µ, Ex, Kp, Kvb, Kg1, Kg2, A, αs, β (9 for pentode).

**Improvements over Koren:**
- Ig2 depends on Va through `1/(1+βVa)` term
- Ia + Ig2 ≈ constant (from constant space charge principle)
- Pentode-as-triode gives consistent results (with δx correction)
- E1 uses `√(Kvb + Vg2²)` instead of just `Vg2` — maintains
  Region A behavior from triode model

### 15.3 DerkE Model (Reefman 2014)

Same as Derk but with beam-tetrode-appropriate screen current:

```
Ig2 = IP_Koren/Kg2 · (1 + αs · exp(−(β·Va)^(3/2)))
```

The `exp(−(βVa)^1.5)` gives a **sharper knee** than Derk's `1/(1+βVa)`,
matching the behavior of beam-like tubes (EF80, 6L6) where the
transition from "all current to G2" to "most current to plate" is abrupt.

**When to use:** Tubes with sharp knee and/or beam-forming structure.
Reefman recommends DerkE when Derk gives poor knee fit.

### 15.4 Derk + Secondary Emission (Reefman 2014)

Added to either Derk or DerkE:

```
Psec = S · Va · (1 + tanh(−ap · (Va − Vco)))
Vco = Vg2/λ − ν·Vg1 − w
```

**Additional parameters:** S, ap, λ, ν, w (5 parameters).

> **LM19 port (audit 2026-07-12):** `lm19/reefman.py::_derk_ia_ig2`
> implements eq. (43)-(46) in full — Psec is subtracted from Ia **and
> added to Ig2** (constant space current; the measured Ig2 hump of
> Theory.pdf Fig. 14 is exactly this term). NB Reefman's own TubeLib.inc
> G2 sources omit +Psec — his library diverges from his paper; LM19
> follows the paper. Only loaded Sc > 0 reference sets are affected:
> `fit_reefman` never fits Sc. Ia is deliberately NOT clamped at 0 —
> in the dynatron region the net anode current physically reverses
> (paper, TubeLib.inc G1 and our SPICE export all agree;
> LTspice-verified to ~1e-4). Removing the old `Ia >= 0` clamp also
> unstuck the fitter on sparse datasets: the clamp created
> gradient-dead zones for nonphysical kg1 > kg2 trial points
> (6P3C rms 75 → 3.4 mA, GU50 173 → 8.3 mA). Pins:
> `tests/test_reefman_paper_pins.py`.

Physical basis (Spangenberg 1948):
- Secondary emission yield ∝ primary electron energy ∝ Va
- Crossover where secondaries return to anode depends on space charge
  (VGK), screen voltage (VG2K), and geometry (w)
- tanh provides smooth crossover transition (width controlled by ap)

### 15.5 Dempwolf Extended v2 (LM19, this document)

Combines the best elements of Dempwolf (2011) and Reefman (2016)
with original extensions:

**From Dempwolf (2011):** Softplus-based cathode emission, smooth
analytic grid current, current conservation by construction.

**From Reefman / Derk (2016):** Durchgriff `A·Va`, crossover
`Vco(Vg1, Vg2)` for secondary emission.

**Original extensions:**
- Adaptive knee `Kvb_eff = Kvb + Kvb1·V_eff` — unique to this model
- Knee sharpness `Kn` — `arctan((Va/Kvb_eff)^Kn)`, unifies smooth/sharp
- Geometric interception `fg2` — explicit physical parameter
- Energy × flux secondary emission `σ·I·Va/Vg2` — hybrid formula
- Variable-mu (dual-section cathode) — after Reefman's approach
- Unified framework — one equation set for triode/pentode/beam tetrode/variable-mu
- Region A correction for triodes (from Koren's `√(Kvb + Va²)`)

**Pentode equations (summary, Kp-normalized):**
```
IK      = G · ((VG2K/C) · ln(1+exp(C·(1/µ + VGK/VG2K))))^γ · (1 + A·VPK)
IGK     = Gg · (ln(1+exp(Cg·VGK)) / Cg)^ξ
I_thr   = max(IK − IGK, 0)
Kvb_eff = Kvb + Kvb1 · max(VG2K/µ + VGK, 0)
α       = (1 − fg2) · (2/π) · arctan((VPK / Kvb_eff)^Kn)
IPK     = I_thr · α
IG2K    = I_thr · (1 − α)
```

Note: Triode path uses original Dempwolf `ln(1+exp(C·x))/C` form.

**Parameters:** 8 (triode) / 12 (pentode) / 15 (variable-mu) / 17 (beam tetrode).

**Status:** Fully implemented in `lm19/dempwolf.py` with 94 tests.

### 15.6 Comparison Table — Equations

| Aspect                   | Koren          | Derk              | DerkE             | Dempwolf v1        | **Dempwolf v2**    |
|--------------------------|----------------|-------------------|-------------------|--------------------|--------------------|
| **Cathode emission**     |                |                   |                   |                    |                    |
| Softplus form            | `ln(1+exp(…))` | `ln(1+exp(…))`    | `ln(1+exp(…))`    | `ln(1+exp(…))/C`  | tri: `ln(…)/C`, pent: `(Vg2/C)·ln(…)` |
| Scale / perveance        | `1/Kg1`        | `1/Kg1`           | `1/Kg1`           | `G`                | `G`                |
| Exponent                 | `Ex` on E1     | `Ex` on E1        | `Ex` on E1        | `γ` on softplus    | `γ` on softplus    |
| Controlling electrode    | Va (triode), Vg2 (pent.) | Vg2       | Vg2               | Vp (tri.), Vg2 (p.)| Vp (tri.), Vg2 (p.)|
| Durchgriff `(1+A·Va)`   | No             | **Yes**           | **Yes**           | No                 | **Yes**            |
| **Region A (triode)**    |                |                   |                   |                    |                    |
| Grid modulation          | `Vg/√(Kvb+Va²)` | `Vg/√(Kvb+Vg2²)` | `Vg/√(Kvb+Vg2²)` | `Vg·Va/√(Kvb_t+Va²)` | `Vg·Va/√(Kvb_t+Va²)` |
| **Grid current**         |                |                   |                   |                    |                    |
| Method                   | Diode D3       | Diode D3          | Diode D3          | Softplus `Gg·(…)^ξ` | **Softplus `Gg·(…)^ξ`** |
| dIg/dVg at Vg=0          | Discontinuous  | Discontinuous     | Discontinuous     | **Continuous**     | **Continuous**     |
| **Current splitting**    |                |                   |                   |                    |                    |
| Knee function            | `arctan(Va/Kvb)` | `1/(1+βVa)`    | `exp(−(βVa)^1.5)` | `arctan((Va/Kvb_eff)^Kn)` | **`arctan((Va/Kvb_eff)^Kn)`** |
| Knee adapts to Vg?       | No             | No                | No                | **Yes** (Kvb1)     | **Yes** (Kvb1)     |
| Knee sharpness control   | No             | No                | Sharper by design  | **Yes** (Kn)       | **Yes** (Kn)       |
| Ig2 source               | Independent eq. | From balance     | From balance      | `(1−α)·I_thr`     | **`(1−α)·I_thr`** |
| Ig2 at Va→∞              | Const (OK)     | **Const > 0**     | **Const > 0**     | **→ 0 (bug!)**    | **fg2·Ik > 0**    |
| Ia(Va=0) = 0 enforced?   | No             | **Yes** (constraint) | **Yes** (constraint) | **Yes** (arctan(0)=0) | **Yes**       |
| **Current conservation** |                |                   |                   |                    |                    |
| Ia + Ig2 = Ik − Ig1      | ❌ No          | ≈ Yes (98–99%)    | ≈ Yes (98–99%)    | ✅ Exact           | **✅ Exact**       |
| Method                   | Independent eqs | Constant space charge | Constant space charge | Splitting I_thr | **Splitting I_thr** |
| **Secondary emission**   |                |                   |                   |                    |                    |
| Modeled?                 | No             | No                | No                | Yes                | **Yes**            |
| +SE variant?             | No             | +SE available     | +SE available     | Built-in (σ>0)    | **Built-in (σ>0)** |
| I_sec formula            | —              | `S·Va·(1+tanh(…))` | `S·Va·(1+tanh(…))` | `σ·Ip·x·e^(−Ks·x)` | **`σ·I·(Va/Vg2)·x·e^(−Ks·x)`** |
| Proportional to          | —              | Energy (Va)       | Energy (Va)       | Flux (Ip)          | **Energy × flux**  |
| Crossover shape          | —              | `tanh` (step)     | `tanh` (step)     | `x·exp(−Ks·x)` (bell) | **`x·exp(−Ks·x)` (bell)** |
| Vco depends on Vg?       | —              | **Yes** `ν·Vg1`   | **Yes** `ν·Vg1`   | No (fixed x)       | **Yes** `ν·VGK`   |
| Crossover params         | —              | S, ap, λ, ν, w   | S, ap, λ, ν, w   | σ, Ks              | **σ, Ks, λ, ν, w** |
| **Pentode-as-triode**    |                |                   |                   |                    |                    |
| Consistent with triode?  | ❌ Different scaling | ✅ (δx correction) | ✅ (δx correction) | ✅ (VG2K=VPK) | **✅ (VG2K=VPK, fg2=0)** |

> **SPICE-export caveat.** The grid-current rows above describe the
> **Python** model. The shipped `.sub` generator (§8.2/8.3,
> `lm19/spice_export/dempwolf.py`) intentionally replaces the analytic
> softplus grid current with Koren's diode (`D3` + `RGI`) and splits the
> full `IK` without subtracting `IGK` — equivalent for `VGK < 0` (all
> LM19 scan data), diverging only in class-A2 territory. The fitted
> `Gg/ξ/Cg` are not exported to SPICE.

### 15.7 Comparison Table — Parameter Count

|                   | Koren | Derk   | DerkE  | Derk+SE | DerkE+SE | Dempwolf v1 | **Dempwolf v2** |
|-------------------|:-----:|:------:|:------:|:-------:|:--------:|:-----------:|:---------------:|
| Triode            | 5     | 5      | 5      | 5       | 5        | 8           | **8**           |
| Pentode           | 6     | 9      | 9      | 9       | 9        | 10          | **12**          |
| Beam tetrode      | 6     | —      | 9      | 14      | 14       | 12          | **17**          |
| Variable-mu pent. | —     | 12     | 12     | 17      | 17       | 13          | **15**          |
| Heptode           | 7     | ~15    | ~15    | ~20     | ~20      | —           | **—**           |

### 15.8 Comparison Table — Capabilities

|                              | Koren | Derk/E   | Derk/E+SE | Dempwolf v1 | **Dempwolf v2** |
|------------------------------|:-----:|:--------:|:---------:|:-----------:|:---------------:|
| **Tube types**               |       |          |           |             |                 |
| Triodes                      | ✅    | ✅       | ✅        | ✅          | ✅              |
| Pentodes                     | ✅    | ✅       | ✅        | ✅          | ✅              |
| Beam tetrodes                | ◐     | ◐        | ✅        | ◐           | ✅              |
| Variable-mu (remote cut-off) | ❌    | ✅       | ✅        | ✅          | **✅**          |
| Heptodes                     | ◐     | ✅       | ✅        | ❌          | ❌              |
| **Physics accuracy**         |       |          |           |             |                 |
| Current conservation (Ia+Ig2=Ik) | ❌ | ◐ ~98%  | ◐ ~98%   | ✅ exact    | ✅ **exact**    |
| Ig2 depends on Va            | ❌    | ✅       | ✅        | ✅          | ✅              |
| Ig2 > 0 at Va → ∞           | ✅    | ✅       | ✅        | ❌ **bug**  | ✅ **fg2 fix**  |
| Durchgriff (anode penetration)| ❌   | ✅       | ✅        | ❌          | ✅ **A·Va**     |
| Pentode-as-triode consistent | ❌    | ✅       | ✅        | ✅          | ✅              |
| **Knee region**              |       |          |           |             |                 |
| Knee modeled                 | ✅ arctan | ✅ 1/(1+βVa) | ✅ exp(−(βVa)^1.5) | ✅ arctan^Kn | ✅ **arctan^Kn** |
| Adaptive knee (Vg-dependent) | ❌    | ❌       | ❌        | ✅          | ✅ **Kvb_eff**  |
| Sharp knee (beam-like)       | ❌    | ❌       | ✅        | ✅ (Kn>1)   | ✅ **(Kn>1)**   |
| **Secondary emission**       |       |          |           |             |                 |
| Kink (negative resistance)   | ❌    | ❌       | ✅        | ✅          | ✅              |
| Kink with sec. emission      | ❌    | ❌       | ✅        | ✅          | ✅              |
| Vco depends on Vg            | —     | —        | ✅ (ν·Vg) | ❌          | ✅ **(ν·VGK)**  |
| Energy-dependent yield       | —     | —        | ✅ (∝ Va) | ❌          | ✅ **(∝ I·Va)** |
| **Grid current**             |       |          |           |             |                 |
| Grid current modeled         | ✅ diode | ✅ diode | ✅ diode | ✅ analytic | ✅ **analytic** |
| Smooth dIg/dVg at Vg=0      | ❌    | ❌       | ❌        | ✅          | ✅              |
| Class A2 operation           | ◐     | ◐        | ◐         | ✅          | ✅              |
| **SPICE integration**        |       |          |           |             |                 |
| SPICE subcircuit template    | ✅    | ✅       | ✅        | ✅          | ✅              |
| LTspice convergence          | ★★★★★| ★★★★☆   | ★★★★☆    | ★★★★☆      | ★★★★☆          |
| Internal nodes per tube      | 2     | 3–4      | 3–4       | 7–9         | 9–11            |
| **Ecosystem**                |       |          |           |             |                 |
| Auto parameter extraction    | ❌    | ✅ ExtractModel | ✅ ExtractModel | ❌ | ✅ `fit_dempwolf()` |
| Ready-to-use library         | ◐ dozens | ✅ 157+ | ✅ 157+ | ❌          | ❌              |
| Community validation         | 30 yr | 10+ yr   | 10+ yr    | New         | New             |

Legend: ✅ = full support, ◐ = partial/approximate, ❌ = not supported.
Grid-current / Class A2 rows: Python model only — the SPICE export uses
a diode grid current (see the caveat under §15.6).

**Key v2 improvements over v1** (bold in table):
- fg2 fix — screen current no longer vanishes at high anode voltage
- A·Va — Durchgriff: total cathode current slightly modulated by plate
- ν·VGK — kink position shifts with grid voltage (physically correct)
- ∝ I·Va — secondary emission depends on both electron flux and energy

**Dempwolf v2 advantages over all other models:**
- Only model with exact current conservation AND adaptive knee
- Only model with smooth grid current AND secondary emission
- Only unified framework (triode/pentode/beam tetrode/variable-mu in one equation)
- Kn exponent — unifies smooth and sharp knee in one parameter

**Derk/DerkE advantages over Dempwolf v2:**
- Heptodes (ECH81) — not in Dempwolf
- 157+ validated tube models — years of community testing
- ExtractModel — automated parameter extraction from .utd files

### 15.9 When to Use Which Model

| Use case                                  | Recommended model             |
|-------------------------------------------|-------------------------------|
| Quick SPICE model from measurements       | **Koren** (our existing fitter) |
| Production SPICE model for LTspice        | **Derk/DerkE** via ExtractModel |
| Beam tetrode with visible kink            | **DerkE+SE** via ExtractModel |
| Variable-mu pentode (EF89, 6K7)           | **Derk/DerkE** or **Dempwolf v2** |
| LM19 curve overlay / quick prediction     | **Koren** (existing) or **v2** |
| LM19 amplifier analysis (Class AB, THD)   | **Dempwolf v2** (best knee)   |
| Overdriven amplifier simulation           | **Dempwolf v2** (grid current) |

---

## 16. Physicality Assessment

This section evaluates each model element against first-principles
physics to determine which aspects are physically derived, which are
physically motivated (correct qualitative behavior from a simplified
physical argument), and which are purely phenomenological (curve
fitting without physics justification).

### 16.1 Classification

- **P** — Physical: derived from first principles (Poisson equation,
  space charge theory, energy conservation)
- **M** — Motivated: correct qualitative behavior from simplified
  physical argument, but functional form is approximate
- **F** — Phenomenological: functional form chosen for mathematical
  convenience / curve fitting quality, not from physics

### 16.2 Assessment Table

| Model element                    | Koren | Derk/DerkE | Dempwolf v2 | Physics basis                                    |
|----------------------------------|:-----:|:----------:|:-----------:|--------------------------------------------------|
| **Cathode emission**             |       |            |             |                                                  |
| Power-law I ∝ V^γ               | M     | M          | M           | Child-Langmuir gives γ=1.5; real γ≈1.2–1.7      |
| Softplus ln(1+exp(…))           | F     | F          | F           | Smooth approx to max(V,0); no derivation         |
| µ (amplification factor)        | P     | P          | P           | Electrostatic field ratio, measurable             |
| **Region A**                     |       |            |             |                                                  |
| √(Kvb + Va²) modulation         | M     | M          | M           | Field penetration; form is approximate            |
| **Grid current**                 |       |            |             |                                                  |
| Diode D3 + resistor             | F     | F          | —           | Discontinuous; not physical near Vg=0            |
| Softplus-based Gg·(…)^ξ         | —     | —          | M           | Smooth onset from thermal distribution; form approx |
| **Current splitting (pentode)**  |       |            |             |                                                  |
| arctan(Va/Kvb)                   | F     | —          | F           | Reynolds (1993); no physical derivation           |
| 1/(1+βVa)                        | —     | M          | —           | From Appendix B scaling analysis (Derk)          |
| exp(−(βVa)^1.5)                  | —     | M          | —           | Beam tetrode space charge (Appendix B)           |
| Adaptive Kvb(V_eff)             | —     | —          | M           | Space charge density ∝ current → Poisson eqn     |
| fg2 (geometric interception)    | —     | P*         | P           | Physical wire intercept fraction; measurable      |
| **Constant space charge**        |       |            |             |                                                  |
| Ia + Ig2 ≈ const vs Va          | ❌    | M          | **P**       | Well-established theory (Spangenberg 1948)       |
| Durchgriff A·Va                  | —     | M          | M           | Field penetration; linear approx                  |
| **Secondary emission**           |       |            |             |                                                  |
| Yield ∝ Va (energy)              | —     | P          | P           | KE at impact = eVa; yield data (Spangenberg)     |
| Yield ∝ I_primary (flux)         | —     | —          | P           | More primaries → more secondaries                 |
| tanh crossover shape             | —     | F          | —           | Smooth step; no derivation for exact shape        |
| x·exp(−Ks·x) kink shape         | —     | —          | M           | Bell-shaped; qualitatively correct                |
| Vco(Vg1, Vg2) dependency        | —     | M          | M           | Space charge suppression; linear approx           |
| **Current conservation**         |       |            |             |                                                  |
| IPK + IG2K + IGK = IK           | ❌    | ≈ (98–99%) | **✅ exact** | Kirchhoff's current law; fundamental              |

*P* for Derk fg2: the ratio Kg1/Kg2 encodes this implicitly but is
not presented as a separate physical parameter.

### 16.3 Key Physical Findings

**1. No model is fully derived from first principles.**

All vacuum tube models use phenomenological elements. The exact
solution requires solving Poisson's equation in 3D for the specific
tube geometry — computationally prohibitive and not useful for SPICE.

**2. Dempwolf v2 is the most physically complete for the common case.**

It explicitly models: cathode emission, grid current onset, current
conservation (Kirchhoff), geometric screen interception, adaptive knee
(space charge), and energy-dependent secondary emission. Each element
has at least a physical motivation (M or P).

**3. Derk/DerkE is more physical for specific effects.**

The DerkE `exp(−(βVa)^1.5)` knee function has a derivation from beam
tetrode space charge scaling (Theory.pdf Appendix B), whereas our
arctan is phenomenological. The Derk model also has a more rigorous
derivation of constant space charge from Spangenberg (1948).

**4. The arctan knee remains phenomenological but is now flexible.**

It is purely phenomenological (from Reynolds 1993). However, with
the adaptive Kvb correction and the Kn sharpness exponent
(`arctan((Va/Kvb_eff)^Kn)`), the error is substantially reduced:
Kvb adapts to operating conditions, and Kn controls knee sharpness
from gradual (Kn < 1) to DerkE-like steep (Kn > 1). A physically-
derived knee function would require separate treatments for pentodes
vs beam tetrodes, breaking the unified framework.

**5. Grid current modeling is a genuine advantage of Dempwolf.**

Koren and Derk use a SPICE diode (D3) with discontinuous dI/dV at
Vg = 0. Dempwolf's softplus-based formula gives continuously
differentiable grid current — correct for the thermal distribution
of electrons at the grid. This matters for:
- Overdriven amplifier analysis
- SPICE convergence near Vg = 0
- Accurate Class A2 (grid current) operation modeling

**6. Current conservation is fundamental physics, not a "nice to have".**

Kirchhoff's current law must hold at every node. Koren violates it
(Ia and Ig2 are independent). Derk approximates it (constant space
charge assumption, ≈98–99% accurate). Dempwolf v2 satisfies it
exactly by construction. This matters for:
- Correct cathode resistor voltage (bias stability analysis)
- Power supply current draw prediction
- Transient analysis accuracy in SPICE

---

## 17. SPICE Simulator Practical Evaluation

This section evaluates the models from the perspective of practical
SPICE simulation: convergence behavior, simulation speed, accuracy
for typical audio circuit analysis tasks, and ecosystem support.

### 17.1 Convergence (Newton-Raphson)

SPICE solves nonlinear equations iteratively using Newton-Raphson.
Key requirements for good convergence:

1. **Smooth functions** — continuous first derivatives everywhere
2. **Bounded derivatives** — no infinite slopes
3. **Monotonic behavior** — fewer local minima for the solver
4. **Few internal nodes** — smaller Jacobian matrix

| Model         | Internal nodes | Derivative smoothness | Typical convergence |
|---------------|:--------------:|:---------------------:|:-------------------:|
| Koren         | 2              | Good (except D3)      | ★★★★★              |
| Derk          | 3–4            | Good (except D3)      | ★★★★☆              |
| DerkE         | 3–4            | Good (except D3)      | ★★★★☆              |
| Derk+SE       | 4–6            | tanh steep gradient   | ★★★☆☆              |
| Dempwolf v1   | 7–9            | Excellent (all smooth)| ★★★★☆ (Ig2→0 issue)|
| **Dempwolf v2** | 9–11         | Excellent (all smooth)| **★★★★☆**          |

**Notes:**
- Koren wins on convergence due to minimal node count and decades
  of SPICE simulator optimization for this model type.
- Dempwolf v2's smooth softplus functions avoid the D3 diode
  discontinuity that causes convergence failures in Koren/Derk
  when VGK crosses zero during transient analysis.
- The v1 Ig2→0 issue could cause convergence problems when the
  solver tries large Va values during iteration; v2 fixes this.
- All models converge reliably for typical audio circuits (1–8 tubes).
  Convergence differences matter mainly for complex circuits or
  extreme operating conditions.

### 17.2 Simulation Speed

For typical audio circuits (preamp + power amp, 2–8 tubes):

| Model       | Relative speed | Reason                                |
|-------------|:--------------:|---------------------------------------|
| Koren       | 1.0× (fastest) | 2 internal nodes per tube             |
| Derk/DerkE  | ~1.2×          | 3–4 internal nodes                    |
| Dempwolf v2 | ~1.5–2.0×      | 9–11 internal nodes                   |

**In practice:** Tube circuits are tiny compared to modern IC designs.
A full Fender Bassman simulation (4 triodes + 2 power tubes) runs in
< 1 second with any model. Speed difference is negligible.

### 17.3 Accuracy for Audio Design Tasks

#### 17.3.1 DC Operating Point

All models give comparable accuracy when properly fitted to the same
data. The operating point is what the fitter optimizes for.

Winner: **Tie** (all models).

#### 17.3.2 Small-Signal Parameters (gm, rp, µ)

These are derivatives of the I-V curves at the operating point.

| Model       | gm accuracy | rp accuracy | Issue                          |
|-------------|:-----------:|:-----------:|--------------------------------|
| Koren       | Good        | **Variable**| Kg1/Kp correlation → unstable rp |
| Derk/DerkE  | Good        | Good        | No correlation issue           |
| Dempwolf v2 | Good        | Good        | Adaptive knee → better rp in knee |

Reefman showed (Theory.pdf §3.4.1) that the Koren Hessian eigenvalue
for the Kg1/Kp eigenvector is 10⁵× smaller than the dominant
eigenvalue — meaning these parameters are poorly determined. Since
rp = ∂Va/∂Ia, it depends on the exact shape of the Ia-Va curve,
which is governed by Kg1 and Kp.

Winner: **Derk/DerkE** and **Dempwolf v2** (no correlation issue).

#### 17.3.3 Harmonic Distortion (THD Spectrum)

THD depends on higher-order derivatives of the transfer function.

| Model       | THD accuracy | Key advantage / limitation            |
|-------------|:------------:|---------------------------------------|
| Koren       | ◐ Fair       | Kg1/Kp correlation → wrong curvature  |
| Derk/DerkE  | ✅ Good      | Consistent I-V scaling                |
| Dempwolf v2 | ✅ Good      | Adaptive knee → correct at all biases |

For single-ended Class A triode amplifiers, all models give similar
THD because the operating point is far from the knee. Differences
appear in:

- **Push-pull Class AB:** Load line crosses the knee region. Adaptive
  knee (Dempwolf v2) gives the most accurate crossover distortion.
- **Pentode output stages:** Screen current dynamics affect B+ sag.
  Derk and Dempwolf v2 model Ig2(Va); Koren does not.
- **Overdriven preamp:** Grid current limiting. Only Dempwolf v2
  models this smoothly.

Winner: **Dempwolf v2** (most complete), **Derk/DerkE** close second.

#### 17.3.4 Power Stage Behavior (Class AB Push-Pull)

The critical test for power tube models. The load line passes through
the knee region as tubes alternate between conduction and cutoff.

| Aspect                       | Koren      | Derk/DerkE  | Dempwolf v2    |
|------------------------------|:----------:|:-----------:|:--------------:|
| Knee shape at Vg = 0         | Fixed      | Fixed       | **Adaptive**   |
| Knee shape at Vg = −30V      | Same       | Same        | **Narrower**   |
| Crossover distortion predict | ◐ 5–15% err| ◐ 3–8% err | **✅ < 3% err** |
| Screen current during cutoff | ❌ Const   | ✅ Varies   | ✅ Varies      |
| Beam tetrode kink            | ❌         | ✅ (DerkE+SE)| ✅             |

The adaptive knee is the decisive advantage for push-pull analysis:
real tubes have wider knees at high current (space charge effect) and
narrower knees at low current. Only Dempwolf v2 captures this.

Winner: **Dempwolf v2** for push-pull; **DerkE+SE** for beam tetrodes
with strong kink.

#### 17.3.5 Transient Analysis (Step Response, Oscillation)

Transient analysis requires smooth models (no convergence timestep
issues) and correct reactive behavior (interelectrode capacitances).

All models support external capacitance elements (Ccg, Cgp, Cpk).
The difference is in the nonlinear element behavior during transients:

- Koren/Derk: D3 diode can cause convergence issues when Vg crosses 0
- Dempwolf v2: smooth everywhere → timestep never needs to shrink
  for model discontinuities

Winner: **Dempwolf v2** (smoothest).

### 17.4 Ecosystem and Tooling

| Aspect                    | Koren      | Derk/DerkE    | Dempwolf v2    |
|---------------------------|:----------:|:-------------:|:--------------:|
| Ready-to-use models       | Dozens     | **157+ tubes**| **0**          |
| Parameter extraction tool | Manual     | **ExtractModel**| LM19 fitter  |
| LTspice library           | Various    | **TubeLib.inc**| —             |
| SPICE compatibility       | All        | LTspice/ngspice| LTspice/ngspice|
| Community adoption        | ★★★★★     | ★★★★☆        | ★☆☆☆☆         |
| Years of validation       | 30 years   | 10+ years     | New            |

**This is the primary practical disadvantage of Dempwolf v2.** The
model may be more accurate, but ExtractModel + TubeLib provides 157+
immediately usable, validated models. Building an equivalent library
for Dempwolf v2 would require years of fitting and validation.

### 17.5 Practical Recommendation

```
For SPICE users who need models NOW:
  → Export .utd from LM19 → ExtractModel → Derk/DerkE
  → Mature, validated, 157+ tube library

For LM19 internal amplifier analysis:
  → Dempwolf v2 (best knee accuracy for push-pull)
  → Koren as fast fallback

For LM19 SPICE export (built-in fitter):
  → Koren (existing, covers 80% of needs)
  → Dempwolf v2 SPICE subcircuit (future, higher quality)
```

### 17.6 Migration Path

1. **Now:** .utd export → access to ExtractModel ecosystem (zero effort)
2. **Now:** Koren fitter (already implemented in `spice_export.py`)
3. **Future:** Dempwolf v2 model functions (for amplifier analysis)
4. **Future:** Dempwolf v2 fitter (phased, as described in §10, §14.6)
5. **Future:** Dempwolf v2 SPICE subcircuit export
6. **Long term:** Build validated parameter library for common tubes

---

## 18. External References and Implementation Notes

Research notes collected during source verification (March 2026).

### 18.1 Reference Implementations

Two existing implementations of the original Dempwolf triode model are
available for cross-validation:

**RT-WDF (C++):**
- Repository: `github.com/RT-WDF/rt-wdf_lib`
- File: `rt-wdf_nlModels.cpp`
- Contains: Dempwolf triode nonlinearity with 12AX7 (RSD-1) parameters
- Accompanies Dunkel et al. DAFx-16 paper
- Can serve as reference implementation for verifying our Python code

**KostasKaram (Python):**
- Repository: `github.com/KostasKaram/common-cathode-triode`
- Contains: `default_params.json` with 12AX7 EHX-1 parameters
- Different specimen from RSD-1 — useful for testing parameter variation

### 18.2 Cohen-Hélie Model — Synonym

The model referred to as "Cohen-Hélie" in some literature (e.g., ACME.jl
issue #29 on GitHub, HSU-ANT/ACME.jl) uses **the same equation structure**
as Dempwolf with different parameter naming:

| Dempwolf   | Cohen-Hélie |
|------------|-------------|
| G          | Gk          |
| C          | Ck          |
| γ          | Ek          |
| µ          | uk          |

If "Cohen-Hélie triode model" appears in literature, it is interchangeable
with Dempwolf for implementation purposes.

### 18.3 Ig0 Parameter — Omitted from LM19 Spec

The original Dempwolf paper includes a small grid current offset:

```
IGK = Gg · (ln(1 + exp(Cg · VGK)) / Cg)^ξ  +  Ig0
```

Where Ig0 ≈ 8e-8 A (≈ 80 nA) for 12AX7. This was included for numerical
stability in the original SPICE implementation.

In the LM19 spec (this document), Ig0 is omitted because:
- The `max(softplus, 0)` clamp in §9.5 achieves the same numerical goal
- 80 nA is below the LM19 tester's measurement resolution (~0.01 mA)
- One fewer parameter simplifies fitting

If SPICE convergence issues arise near Ig = 0, consider restoring Ig0
as a fixed constant (not fitted).

### 18.4 Published µ Value — 103.2, Not 100

The original Dempwolf paper (DAFx-11) reports µ = **103.2** for 12AX7
(RSD-1 specimen). Section 2.4 of this document rounds it to 100 for
readability. For verification tests against published data, use the
exact value 103.2.

Full original parameter set (RSD-1):

```
G = 2.242e-3, µ = 103.2, γ = 1.26, C = 3.40
Gg = 6.177e-4, ξ = 1.314, Cg = 9.901, Ig0 = 8.025e-8
```

### 18.5 Reefman Theory.pdf — DerkE Knee Derivation

Reefman's Theory.pdf (6 pages, January 2016) contains **Appendix B**
with a physical derivation of the DerkE knee function `exp(−(βVa)^1.5)`
from beam tetrode space charge scaling (Poisson equation in beam geometry).

This is the only published physical justification for a specific knee
functional form. Our arctan^Kn remains phenomenological (§16.2), but
with adaptive Kvb and Kn it achieves comparable or better fit quality
by having two degrees of freedom (shape + width) vs DerkE's one (β only).

Future work: compare RMS errors of `arctan((Va/Kvb_eff)^Kn)` vs
`exp(−(βVa)^1.5)` on the same measured data to quantify the trade-off.

### 18.6 Academic Validation — Dempwolf as Ground Truth

Two recent DAFx papers provide independent validation of the Dempwolf
model's standing in the academic community:

**DAFx-22 (Stanford CCRMA, Darabundit et al.):**
- "Neural Net Tube Models for Wave Digital Filters"
- Used Dempwolf model (not Koren) as **ground truth** for training
  neural networks to approximate 12AX7 behavior
- Group of Julius O. Smith III — author of standard DSP textbooks
- Validates Dempwolf as the preferred reference model for accuracy

**DAFx-23 (Politecnico di Milano, Giampiccolo et al.):**
- "A Quadric Surface Model of Vacuum Tubes for Virtual Analog Applications"
- Direct comparison: even a **3-parameter quadric surface** outperforms
  Koren in common operating regions of 12AX7
- Koren described as "reasonable but not the most accurate"
- 4.6× speedup over Cardarilli (15-param spline model) in WDF

Both papers are saved locally:
- `external_sources/theory/modern_dafx22_neural_wdf_tube.pdf`
- `external_sources/theory/modern_dafx23_quadric_vacuum_tube_model.pdf`

### 18.7 Quadric Surface Model — Not Applicable to LM19

The DAFx-23 quadric model (`ip = kp2·Va² + kpg·Va·Vg + kp·Va + ...`)
is interesting but not suitable for LM19:
- Only 3 parameters → cannot model pentodes, grid current, or knee
- Designed for real-time virtual analog (WDF), where speed matters
- No grid current model (assumes Ig = 0)
- No pentode extension published

Mentioned here for completeness. If ultra-fast triode-only evaluation
is ever needed (e.g., real-time curve preview), the quadric form could
be considered as a lightweight approximation.

### 18.8 ExtractModel + .utd — Fast Path to 157+ Tubes

Reefman's **ExtractModel** program (v3.0, February 2016) accepts `.utd`
files (µTracer measurement format) and automatically fits Derk/DerkE
parameters. The output is a ready-to-use LTspice subcircuit.

The accompanying **TubeLib.inc** library contains 157+ pre-fitted models
covering most common audio tubes (12AX7, EL34, 6L6, KT88, etc.).

**LM19 already has .utd export:** `utracer_export.py` (`format_utd()`)
converts LM19 JSON measurements to .utd format. Supports both output
curves I(Va, Vg) and transfer curves I(Vg, Va), with optional Ig2.
Screen voltage Vs is encoded in the filename by convention.

This means the **lowest-effort step** in the migration path (§1a step 1)
is already done — users can export .utd from LM19, feed it to
ExtractModel, and get Derk/DerkE SPICE models immediately.

**Remaining integration opportunity:** add a UI button "Export for
ExtractModel" in CompareTab or ExportManager that calls `format_utd()`
and opens the save dialog with the suggested filename.

### 18.9 Published Dempwolf Parameters — Only 12AX7

The original Dempwolf paper (DAFx-11) published fitted parameters for
**three specimens** of 12AX7 only:

| Parameter | RSD-1      | RSD-2      | EHX-1      |
|-----------|------------|------------|------------|
| G         | 2.242e-3   | 2.173e-3   | 1.371e-3   |
| µ         | 103.2      | 100.2      | 86.9       |
| γ         | 1.26       | 1.28       | 1.349      |
| C         | 3.40       | 3.19       | 4.56       |
| Gg        | 6.177e-4   | 5.911e-4   | 3.263e-4   |
| ξ         | 1.314      | 1.358      | 1.156      |
| Cg        | 9.901      | 11.76      | 11.99      |
| Ig0       | 8.025e-8   | 4.527e-8   | 3.917e-8   |

**Key observations:**

1. **Significant specimen variation:** µ ranges 86.9–103.2 (18% spread),
   G ranges 1.37e-3–2.24e-3 (63% spread). This is expected for vacuum
   tubes and validates per-tube fitting as the primary use case.

2. **No published parameters for other tube types.** For pentodes
   (EL34, EL84), beam tetrodes (6L6, KT88), and other triodes (6SN7,
   12AU7), all Dempwolf parameters must be fitted from measurements.
   Section 11 provides estimated parameters for EL34 and 6L6GC derived
   from Koren equivalents, but these are approximations, not published
   fits.

3. **Verification strategy:** Use all three 12AX7 parameter sets for
   regression tests. A correct implementation must reproduce the
   published Ia/Ig values at the test points in §11.1 for each specimen.

---

*Document version: 2.1 — March 2026*
*For LM19 Tube Tester application*
*Based on Dempwolf & Zölzer (2011), extended for pentodes and beam tetrodes*
*v2 improvements based on analysis of Reefman's Derk/DerkE models (2016)*

## Sources

Materials this document rests on. The full registry of the project
external sources (all entries, statuses, local copies) — `SOURCES_INDEX.md`.

### Dempwolf & Zölzer — A Physically-Motivated Triode Model (DAFx-11, 2011)
- url: <https://dafx.de/paper-archive/2011/Papers/76_e.pdf>
- type: theory
- role here: base model equations, 12AX7 parameters
- note: Original Dempwolf triode model paper. Cathode current Ik =
  G·(ln(1+exp(C·(Va/µ+Vg)))/C)^γ, grid current Ig =
  Gg·(ln(1+exp(Cg·Vg))/Cg)^ξ + Ig0, plate current Ia = Ik − Ig. 8 parameters
  (G, µ, γ, C, Gg, ξ, Cg, Ig0). Published fits for 12AX7 (3 specimens). Key
  advantage over Koren: smooth grid current, current conservation by
  construction. Known limitation: poor Region A behavior at low Va
  (addressed by Kvb_t extension in LM19).

### Dunkel et al. — Fender Bassman 5F6-A WDF Case Study (DAFx-16, 2016)
- url: <https://www.dafx.de/paper-archive/2016/dafxpapers/37-DAFx-16_paper_53-PN.pdf>
- type: theory
- role here: Dempwolf model validation, 12AX7 parameters
- note: Wave digital filter implementation of Fender Bassman using Dempwolf
  triode model. Confirms 12AX7 parameters from original Dempwolf paper.
  Reference C++ implementation available in RT-WDF library (github.com/RT-
  WDF/rt-wdf_lib).

### Reefman — Spice Models for Vacuum Tubes Using the uTracer (Theory.pdf, 2016)
- url: <https://www.dos4ever.com/uTracer3/Theory.pdf>
- type: theory
- role here: Derk/DerkE models, Durchgriff, Vco, variable-mu, fg2
- note: 50-page paper by Derk Reefman (2016-01-24). Defines Derk (true
  pentode) and DerkE (beam tetrode) models with physically-motivated current
  splitting, constant space charge principle, secondary emission (Psec),
  variable-mu pentodes, and hexode/heptode models. Key equations: Ip,Koren
  with √(Kvb+Vg2²), Derk Ig2 = Ip/Kg2·(1+αs/(1+βVa)), DerkE Ig2 =
  Ip/Kg2·(1+αs·exp(-(βVa)^1.5)). Companion software: ExtractModel, utMax.
  Several ideas adopted into Dempwolf Extended v2.

### Norman Koren — Improved Vacuum Tube Models for SPICE Simulations (1996)
- url: <https://www.normankoren.com/Audio/Tubemodspice_article.html>
- type: theory
- role here: comparison baseline
- note: Full article with Koren model equations. Triode:
  E1=(Va/Kp)·ln(1+exp(Kp·(1/µ+Vg/√(Kvb+Va²)))), Ia=2·E1^Ex/Kg1. Pentode: E1
  with Vg2, Ia·arctan(Va/Kvb). Known issues (per Reefman): Kg1/Kp
  correlation, Ig2 independent of Va, pentode-as-triode inconsistency.
  Companion page with parameter tables saved as
  koren_improved_tube_models.html.
