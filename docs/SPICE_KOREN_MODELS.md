# Koren Triode & Pentode SPICE Models — Full Reference

Technical and theoretical reference for the vacuum tube SPICE models used
in LM19 tube tester. Based on Norman Koren's original publications (1996-2003).

**Primary sources:**
- N. Koren, "Improved vacuum tube models for SPICE simulations",
  *Glass Audio*, Vol. 8, No. 5, 1996.
  Part 1: normankoren.com/Audio/Tubemodspice_article.html
  Part 2: normankoren.com/Audio/Tubemodspice_article_2.html
- N. Koren, "Finding SPICE tube model parameters" (Tuparam),
  normankoren.com/Audio/Tube_params.html
- S. Reynolds, "Vacuum-tube models for PSPICE simulations",
  *Glass Audio*, Vol. 5, No. 4, 1993.
- W. M. Leach, Jr., "SPICE models for vacuum-tube amplifiers",
  *J. Audio Eng. Soc.*, Vol. 43, No. 3, 1995.

---

## Table of Contents

1. [Historical Background](#1-historical-background)
2. [Classical Equations (Reynolds/Leach)](#2-classical-equations-reynoldsleach)
3. [Koren Triode Model](#3-koren-triode-model)
   - 3.1 [Equation](#31-equation)
   - 3.2 [Physical Meaning of Each Parameter](#32-physical-meaning-of-each-parameter)
   - 3.3 [How the Equation Works (Step by Step)](#33-how-the-equation-works-step-by-step)
   - 3.4 [Reduction to Classical Form](#34-reduction-to-classical-form)
   - 3.5 [Region A Problem and Solution](#35-region-a-problem-and-solution)
4. [Koren Pentode Model](#4-koren-pentode-model)
   - 4.1 [Equation — Plate Current](#41-equation--plate-current)
   - 4.2 [Equation — Screen Current](#42-equation--screen-current)
   - 4.3 [Physical Meaning of Parameters](#43-physical-meaning-of-parameters)
   - 4.4 [How the Pentode Equation Differs from Triode](#44-how-the-pentode-equation-differs-from-triode)
   - 4.5 [Operating Modes (Pentode / Ultra-Linear / Triode)](#45-operating-modes-pentode--ultra-linear--triode)
5. [Comparison Table: Triode vs Pentode](#5-comparison-table-triode-vs-pentode)
6. [SPICE Subcircuit Templates](#6-spice-subcircuit-templates)
   - 6.1 [Triode Subcircuit](#61-triode-subcircuit)
   - 6.2 [Pentode Subcircuit](#62-pentode-subcircuit)
   - 6.3 [SPICE Functions: PWR, PWRS, LOG, EXP](#63-spice-functions-pwr-pwrs-log-exp)
7. [Published Parameters](#7-published-parameters)
8. [Python Implementation](#8-python-implementation)
   - 8.1 [Triode Model Function](#81-triode-model-function)
   - 8.2 [Pentode Model Function](#82-pentode-model-function)
   - 8.3 [Pentode Screen Current Function](#83-pentode-screen-current-function)
   - 8.4 [Numerical Safety](#84-numerical-safety)
9. [Fitting Strategy](#9-fitting-strategy)
   - 9.1 [Data Requirements](#91-data-requirements)
   - 9.2 [Triode Fitting](#92-triode-fitting)
   - 9.3 [Pentode Fitting — Two-Phase Approach](#93-pentode-fitting--two-phase-approach)
   - 9.4 [Residual Function](#94-residual-function)
   - 9.5 [Parameter Bounds](#95-parameter-bounds)
   - 9.6 [Kg2 — Special Handling](#96-kg2--special-handling)
10. [Finding Parameters by Trial and Error](#10-finding-parameters-by-trial-and-error)
11. [Fit Quality Metrics](#11-fit-quality-metrics)
12. [Topology Detection and Mode Selection](#12-topology-detection-and-mode-selection)
13. [Edge Cases and Numerical Safety](#13-edge-cases-and-numerical-safety)
14. [Verification Examples](#14-verification-examples)

---

## 1. Historical Background

SPICE (Simulation Program with Integrated Circuit Emphasis) was developed at
UC Berkeley for electronic circuit simulation. It contains built-in models for
transistors and passive devices, but **none for vacuum tubes**.

The classical approach (Reynolds 1993, Leach 1995) models vacuum tubes as
voltage-controlled current sources based on the **Langmuir-Child's Law**
(the "three-halves power law"), derived from fundamental physics of electron
flow in a vacuum.

Norman Koren (1996) improved these models by introducing a **soft transition
function** `log(1 + exp(x))` that eliminates the discontinuity at cutoff and
provides much better accuracy in the critical "Region A" (large plate voltage,
large negative grid voltage, low current) where real tubes deviate most from
the idealized equations.

**Why Region A matters:** A typical triode load line (e.g., 12AX7 with 350V
supply, 150 kOhm plate resistor) crosses Region A. Class AB push-pull
amplifiers operate extensively in this region as each tube moves between
conduction and cutoff. Accurate distortion analysis is impossible without
a good model here.

---

## 2. Classical Equations (Reynolds/Leach)

For context, here are the older equations that Koren improved upon.

### Triode (Leach)

```
Ip = (Eg + Ep/mu)^(3/2) / Kg1     when (Eg + Ep/mu) >= 0
   = 0                              otherwise
```

**Problem:** Assumes the grid has perfect control over plate current.
At large Ep with large negative Eg, the model predicts zero current, but
real tubes show measurable leakage current (Region A).

### Pentode (Reynolds/Leach)

```
Ip = 2 * (Eg + Eg2/mu)^(3/2) / Kg1 * arctan(Ep/Kvb)
                                     when (Eg + Eg2/mu) >= 0
   = 0                              otherwise
```

```
Ig2 = (Eg + Eg2/mu)^(3/2) / Kg2    when (Eg + Eg2/mu) >= 0
    = 0                              otherwise
```

**Problem:** Same discontinuity at cutoff, plus the knee shape
(`arctan(Ep/Kvb)`) is not well modeled for all grid voltage levels.

---

## 3. Koren Triode Model

### 3.1 Equation

```
E1 = (Ep / Kp) * ln(1 + exp(Kp * (1/mu + Eg / sqrt(Kvb + Ep²))))

Ia = (PWR(E1, Ex) + PWRS(E1, Ex)) / Kg1
```

Where:
- `PWR(x, n)  = |x|^n`
- `PWRS(x, n) = sign(x) * |x|^n`
- The sum `PWR + PWRS = 2x^n` for x > 0, and `= 0` for x <= 0

**Simplified (for E1 > 0, which is the normal operating region):**

```
Ia = 2 * E1^Ex / Kg1
```

**For E1 <= 0 (tube in cutoff):**

```
Ia = 0
```

### 3.2 Physical Meaning of Each Parameter

#### mu (amplification factor, dimensionless)

The voltage gain of the tube in open-circuit conditions. Defined as:

```
mu = -dEp / dEg   at constant Ip
```

In other words: how much must the plate voltage change to compensate for
a 1V change in grid voltage while keeping plate current constant.

- **Low-mu triodes:** 12AU7 (mu ~ 21), 6SN7 (mu ~ 20)
- **Medium-mu:** 12AT7 (mu ~ 60), 6SL7 (mu ~ 70)
- **High-mu:** 12AX7 (mu ~ 100), 6SF5 (mu ~ 100)

mu is approximately equal to the manufacturer's published specification, but
the fitting process may adjust it slightly for best overall curve fit.

**In the equation:** `1/mu` is the DC bias term in the E1 argument. It sets
the baseline at which the grid voltage has no effect and only plate voltage
drives current.

#### Ex (current exponent, dimensionless)

The exponent of the current-voltage relationship. Classical
Langmuir-Child's law gives Ex = 3/2 = 1.5. In practice, Ex varies:

- **Typical range:** 1.0 — 2.0 (most commonly 1.2 — 1.7)
- **Low Ex (1.2-1.3):** More linear curves (12AU7, 6DJ8)
- **High Ex (1.4-1.5):** More pronounced curvature (12AX7)

Ex controls the "shape" of the curves — how steeply plate current rises
with effective grid voltage. Higher Ex = steeper rise = more curvature =
potentially more harmonic distortion. The textbook value 3/2 (1.5) is an
average; individual tube types deviate.

**In the equation:** `E1^Ex` is the power law that converts the effective
control voltage E1 into current.

#### Kg1 (cathode current scale, dimensionless in SPICE context)

A scaling factor inversely proportional to plate current. Larger Kg1 = less
current for the same E1.

- **Small signal tubes:** Kg1 = 1000 — 2000 (12AX7: 1060, 12AU7: 1180)
- **Power tubes (triode mode):** Kg1 = 500 — 1500 (EL34: 650, 6550: 890)

**How to estimate:** At a known operating point, compute E1 and solve:
`Kg1 = 2 * E1^Ex / Ia`

**In the equation:** `2 * E1^Ex / Kg1` — Kg1 simply scales the output
current. Doubling Kg1 halves the plate current.

#### Kp (pinch-off parameter, dimensionless)

Controls the plate current in **Region A** — the region of large plate
voltage and large negative grid voltage where the classical model fails.

- **High-mu triodes:** Kp = 300 — 900 (12AX7: 600, 6SL7: 400)
- **Low-mu triodes:** Kp = 60 — 300 (12AU7: 84, 6DJ8: 320)
- **Power pentodes (triode mode):** Kp = 30 — 120 (6550: 48, EL34: 60)

Higher Kp means the grid has stronger control: the tube cuts off more
sharply and Region A leakage is lower.

**In the equation:** Kp appears in two places:
1. **Inside the exp:** `Kp * (1/mu + Eg / sqrt(Kvb + Ep²))` — amplifies
   the control voltage, making the soft-max transition sharper.
2. **Outside as divisor:** `Ep / Kp` — normalizes E1 so that the plate
   voltage contribution is inversely proportional to Kp.

The combination ensures that the transition from conducting to cutoff is
smooth (controlled by the `log(1 + exp(...))` function) and that the
leakage in Region A is correctly modeled.

**Finding Kp:** This is the hardest parameter to determine. Koren
recommends finding it by trial and error by matching the tube curves in
Region A. Triode-mode curves must be used for pentodes.

#### Kvb (knee voltage parameter)

**For triodes: Kvb is in units of V² (volts squared).**

It controls the curvature in the "knee" region where plate current first
begins to flow (low Ep, high Eg). It also modulates how the grid voltage
influence depends on plate voltage.

- **Small-signal triodes:** Kvb = 100 — 500 (12AX7: 300, 12AU7: 300)
- **Power tubes:** Kvb = 10 — 50 (EL34: 24, 6550: 24)

**In the equation:** The term `Eg / sqrt(Kvb + Ep²)` means:
- When `Ep² >> Kvb`: the grid influence is `Eg / Ep` (standard mu behavior)
- When `Ep` is small: `sqrt(Kvb)` dominates, softening the grid's influence

This gives the characteristic "rounding" of curves near the knee.

**For pentodes:** Kvb has a different meaning — see Section 4.

### 3.3 How the Equation Works (Step by Step)

Given Ep = 250V, Eg = -2V for a 12AX7 (mu=100, Ex=1.4, Kg1=1060, Kp=600,
Kvb=300):

**Step 1: Compute the argument of the exponent**

```
arg = Kp * (1/mu + Eg / sqrt(Kvb + Ep²))
    = 600 * (1/100 + (-2) / sqrt(300 + 62500))
    = 600 * (0.01 + (-2) / 250.6)
    = 600 * (0.01 - 0.00798)
    = 600 * 0.00202
    = 1.21
```

**Step 2: Apply the soft-max function**

```
log(1 + exp(1.21)) = log(1 + 3.353) = log(4.353) = 1.472
```

For arg >> 1: `log(1 + exp(arg)) ≈ arg` (tube is conducting)
For arg << -1: `log(1 + exp(arg)) ≈ 0` (tube is in cutoff)

**Step 3: Compute E1**

```
E1 = (Ep / Kp) * log(1 + exp(arg))
   = (250 / 600) * 1.472
   = 0.417 * 1.472
   = 0.613
```

**Step 4: Compute plate current**

```
Ia = 2 * E1^Ex / Kg1
   = 2 * 0.613^1.4 / 1060
   = 2 * 0.505 / 1060
   = 0.000953 A
   ≈ 0.95 mA
```

This matches published 12AX7 data very well (datasheet ~1.0 mA at this point).

### 3.4 Reduction to Classical Form

When `Ep² >> Kvb` and `Kp * (1/mu + Eg/Ep) >> 1`:

```
sqrt(Kvb + Ep²) ≈ Ep
log(1 + exp(x)) ≈ x    (for x >> 1)

E1 ≈ (Ep / Kp) * Kp * (1/mu + Eg / Ep)
   = Ep/mu + Eg

Ia ≈ 2 * (Ep/mu + Eg)^Ex / Kg1
```

This is exactly the classical Langmuir-Child equation (1) with the
3/2 exponent generalized to Ex.

### 3.5 Region A Problem and Solution

**Region A** = large Ep, large negative Eg, low current.

In the classical model, `Eg + Ep/mu` can be exactly zero or negative,
giving Ip = 0 with a hard cutoff. Real tubes always have some leakage.

**Koren's solution:** The `log(1 + exp(x))` function is a **smooth
approximation of max(x, 0)**:
- For x >> 0: `log(1 + exp(x)) ≈ x` (same as classical)
- For x = 0: `log(1 + exp(0)) = log(2) = 0.693`
- For x < 0: `log(1 + exp(x)) ≈ exp(x) > 0` (small but nonzero)

This means E1 is always > 0 when Ep > 0, so there is always some plate
current — exactly matching real tube behavior.

The `Ep / Kp` factor outside the log ensures that E1 scales proportionally
to plate voltage, so the leakage current increases with Ep (as observed
in real tubes).

---

## 4. Koren Pentode Model

### 4.1 Equation — Plate Current

```
E1 = (Eg2 / Kp) * ln(1 + exp(Kp * (1/mu + Eg / Eg2)))

Ia = (PWR(E1, Ex) + PWRS(E1, Ex)) / Kg1 * arctan(Ep / Kvb)
   = 2 * E1^Ex / Kg1 * arctan(Ep / Kvb)       when E1 > 0
   = 0                                          when E1 <= 0
```

### 4.2 Equation — Screen Current

The screen current uses a simpler form (from the original Reynolds/Leach
model). Koren kept it because:

1. Screen current is less critical to tube performance than plate current.
2. Good data on screen current is scarce.
3. The model should be kept simple for evaluation versions of SPICE.

**Mathematical form:**

```
Ig2 = (Eg + Eg2/mu)^(3/2) / Kg2    when (Eg + Eg2/mu) > 0
    = 0                              otherwise
```

**SPICE listing form** (from Koren's tube.lib, using EX instead of 3/2):

```
G2 = exp(Ex * ln(Eg2/mu + Eg)) / Kg2
```

This is equivalent to `(Eg2/mu + Eg)^Ex / Kg2`, using the identity
`x^n = exp(n * ln(x))`. The SPICE `LOG()` function is natural logarithm.
When the argument is negative or zero, SPICE clamps internally; in Python
we use `np.maximum(v, 0)` explicitly.

**Note:** Koren himself did not optimize Kg2 in his Tuparam program.
He states: "KG2 does not need to be estimated with great accuracy to obtain
good results in SPICE simulations." It is typically estimated from a single
published screen current data point: `Kg2 = (Eg + Eg2/mu)^Ex / Ig2`.

### 4.3 Physical Meaning of Parameters

All parameters from the triode model (mu, Ex, Kg1, Kp) have the **same
physical meaning** in the pentode model. The key differences are:

#### mu (amplification factor) — same meaning

In a pentode, mu is the amplification factor measured in **triode mode**
(screen grid connected to plate). This is the "true" mu of the tube's
control grid structure.

- **Beam tetrodes / pentodes:** mu = 5 — 30 (6550: 7.9, EL34: 11, KT88: 8.8)
- Much lower than triodes because the screen grid shields the plate

Koren recommends: "For pentodes, the best estimates of MU, EX, and KG1 are
obtained from triode-mode curves."

#### Ex (current exponent) — same meaning

Typically 1.3 — 1.4 for power pentodes. Usually the same whether measured
in triode or pentode mode.

#### Kg1 (plate current scale) — same meaning

Inversely proportional to plate current. Power pentodes have lower Kg1
(more current) than signal triodes.

#### Kp (pinch-off parameter) — same meaning

Controls the transition to cutoff. For pentodes, Kp is typically lower
(30-120) than for high-mu triodes (300-900), because the screen grid
already provides strong control.

#### Kvb (knee voltage, in Volts — NOT V²)

**THIS IS THE KEY DIFFERENCE FROM THE TRIODE MODEL.**

In the pentode equation, Kvb appears in `arctan(Ep / Kvb)`, where it
directly sets the plate voltage at which the "knee" of the pentode
characteristic occurs.

- **Meaning:** The plate voltage at which the arctan function reaches
  about 50% of its maximum (since `arctan(1) = pi/4 ≈ 0.785`, which
  is about half of `pi/2 ≈ 1.571`).
- **Typical values:** 10 — 100 V (6550: 24V, EL34: 24V, 6L6GC: 12V)
- **Low Kvb:** Sharp knee (beam tetrodes like 6L6)
- **High Kvb:** Gradual knee (true pentodes like EF86)

**The arctan function models the pentode characteristic shape:**

```
arctan(Ep / Kvb):
  Ep = 0:         arctan(0) = 0           → no current (knee)
  Ep = Kvb:       arctan(1) = 0.785       → ~50% of max
  Ep = 5*Kvb:     arctan(5) = 1.373       → ~87% of max
  Ep >> Kvb:      arctan(∞) = π/2 = 1.571 → flat (constant Ip region)
```

This perfectly models the pentode/tetrode behavior:
- Below the knee: plate current rises steeply with plate voltage
- Above the knee: plate current is nearly constant (high output impedance)
- The transition is smooth and physics-based

#### Kg2 (screen current scale, pentode only)

Inversely proportional to screen grid current. Similar role to Kg1 but
for the screen grid.

- **Typical values:** 2000 — 10000 (6550: 4200, EL34: 4200, 6L6: 4500)
- Not fitted with high precision (per Koren)
- Usually estimated from a single published Ig2 data point

### 4.4 How the Pentode Equation Differs from Triode

The fundamental difference is **which electrode controls the current**.

**In a real pentode/tetrode:**
- The **screen grid (g2)** acts as the primary accelerating electrode
- The plate merely collects electrons that pass through the screen grid
- The plate voltage has little effect on current (high output impedance)
- The "knee" occurs when the plate voltage drops below the screen voltage

Koren's pentode equation reflects this physics:

| Element              | Triode                            | Pentode                        |
|----------------------|-----------------------------------|--------------------------------|
| E1 numerator         | `Ep / Kp`                         | `Eg2 / Kp`                    |
| Grid influence term  | `Eg / sqrt(Kvb + Ep²)`           | `Eg / Eg2`                    |
| Knee modeling        | Built into E1 via sqrt()          | Separate `arctan(Ep / Kvb)`   |
| Controlling voltage  | Plate voltage (Ep)                | Screen voltage (Eg2)           |
| Ep in E1             | Primary role                      | No role (only in arctan)       |
| Kvb units            | V² (modulates grid influence)     | V (plate knee position)        |
| Screen current       | N/A                               | Separate equation with Kg2     |
| Total parameters     | 5                                 | 6                              |
| Pin count            | 3 (A, G, K)                       | 4 (A, G, K, G2)               |

**Why Eg2 replaces Ep in E1:**

In a pentode, the screen grid voltage determines the electric field that
accelerates electrons from the cathode. The plate is "behind" the screen
and has minimal influence on total cathode current. So the E1 formula uses
Eg2 instead of Ep:
- `Eg2 / Kp` scales E1 by screen voltage (not plate)
- `Eg / Eg2` is the grid influence relative to screen (not plate)

**Why arctan(Ep/Kvb) is separate:**

The plate voltage determines **what fraction** of the total cathode current
reaches the plate (vs returns to the screen grid). This is modeled by
the arctan factor:
- At low Ep (< Kvb): most electrons return to g2 → low Ip
- At high Ep (>> Kvb): most electrons reach the plate → Ip ≈ maximum

### 4.5 Operating Modes (Pentode / Ultra-Linear / Triode)

A pentode can operate in three modes depending on how Eg2 is connected:

#### Pentode mode (Eg2 = constant)

```
Eg2 = Eg2_nominal (fixed voltage from power supply)
```

Standard operation. High output impedance, high gain, high distortion.
The Koren pentode equation is used directly.

#### Ultra-Linear mode (Eg2 = partial tracking)

```
Eg2 = Eg2_nominal * (1 - tap) + Ep * tap
```

Where `tap` is typically 0.4 (40% ultra-linear tap on the output
transformer). This is a compromise between pentode and triode modes:
lower output impedance and distortion than pentode, higher power than
triode.

In SPICE, this is modeled by setting the E2 voltage source:
```spice
E2 VALUE = {V(VG2NOM) * 0.6 + V(1P) * 0.4}
```

#### Triode mode (Eg2 = Ep)

```
Eg2 = Ep  (screen grid connected directly to plate)
```

The pentode now behaves as a triode. Koren notes that substituting
Eg2 = Ep into the pentode equation gives:

```
E1 = (Ep / Kp) * ln(1 + exp(Kp * (1/mu + Eg / Ep)))
Ia = 2 * E1^Ex / Kg1 * arctan(Ep / Kvb)
```

This is close to (but not identical to) the triode equation (4). For
Ep >> Kvb, `arctan(Ep/Kvb) ≈ π/2`, and the remaining equation is similar
to the triode form with `Eg / Ep` instead of `Eg / sqrt(Kvb + Ep²)`.

**This is why Koren recommends fitting mu, Ex, Kg1, Kp from triode-mode
curves first** — these 4 parameters should give a good fit in both triode
and pentode modes. Then Kvb and Kg2 are adjusted to fit pentode-mode data.

---

## 5. Comparison Table: Triode vs Pentode

| Aspect                 | Koren Triode                                     | Koren Pentode                                     |
|------------------------|--------------------------------------------------|---------------------------------------------------|
| **E1 formula**         | `(Ep/Kp) * ln(1+exp(Kp*(1/mu+Eg/sqrt(Kvb+Ep²))))` | `(Eg2/Kp) * ln(1+exp(Kp*(1/mu+Eg/Eg2)))`       |
| **Ia formula**         | `2*E1^Ex/Kg1`                                    | `2*E1^Ex/Kg1 * arctan(Ep/Kvb)`                   |
| **Ig2 formula**        | N/A                                              | `(Eg+Eg2/mu)^Ex / Kg2`                           |
| **Parameters**         | mu, Ex, Kg1, Kp, Kvb (5)                        | mu, Ex, Kg1, Kg2, Kp, Kvb (6)                    |
| **Kvb meaning**        | V² — modulates grid influence                    | V — plate knee location                           |
| **Kvb range**          | 5 — 3000 V²                                     | 1 — 500 V                                        |
| **Controlling voltage**| Plate (Ep)                                       | Screen grid (Eg2)                                 |
| **SPICE pins**         | A, G, K                                          | A, G, K, G2                                       |
| **Tube types**         | Triodes (12AX7, 6SN7, 6SL7...)                  | Pentodes, beam tetrodes (EL34, 6L6, KT88...)     |
| **Classic reduces to** | `2*(Ep/mu + Eg)^Ex / Kg1`                       | `2*(Eg2/mu + Eg)^Ex / Kg1 * arctan(Ep/Kvb)`     |
| **Region A fix**       | Yes — `log(1+exp())` soft transition             | Yes — same mechanism via E1                       |
| **Output impedance**   | Low (Ep controls current)                        | High (Ep only in arctan)                          |

---

## 6. SPICE Subcircuit Templates

### 6.1 Triode Subcircuit

Directly from Koren's tube.lib (12AX7 example):

```spice
.SUBCKT 12AX7 A G K
+ PARAMS: MU=100 EX=1.4 KG1=1060 KP=600 KVB=300
+ CCG=2.3P CGP=2.4P CCP=0.9P RGI=2000
*
* E1: intermediate Koren voltage
* The ln(1+exp()) function is the smooth max(x,0) that fixes Region A
E1 7 0 VALUE={V(A,K)/KP*LOG(1+EXP(KP*(1/MU+V(G,K)/SQRT(KVB+V(A,K)*V(A,K)))))}
RE1 7 0 1G
*
* G1: anode current source
* PWR(x,n)+PWRS(x,n) = 2*x^n for x>0, 0 for x<=0
G1 A K VALUE={(PWR(V(7),EX)+PWRS(V(7),EX))/KG1}
RCP A K 1G         ; plate-cathode leakage for SPICE convergence
*
* Interelectrode capacitances (from datasheets + socket estimate)
C1 G K {CCG}       ; cathode-grid
C2 G A {CGP}       ; grid-plate (Miller capacitance)
C3 A K {CCP}       ; cathode-plate
*
* Grid current model (conduction when Vg > 0)
R1 G 5 {RGI}       ; grid stopper resistor
D3 5 K DX           ; grid-cathode diode
.MODEL DX D(IS=1N RS=1 CJO=10PF TT=1N)
*
.ENDS 12AX7
```

**Circuit explanation:**
- `E1` — voltage-controlled voltage source computing the intermediate
  variable E1. Breaks the long Ia equation into two manageable parts.
- `RE1` — 1 Gigaohm dummy resistor to prevent floating node 7.
- `G1` — voltage-controlled current source for plate current.
- `RCP` — 1 Gigaohm plate-cathode resistor for SPICE convergence.
- `C1, C2, C3` — interelectrode capacitances for AC/frequency analysis.
- `R1, D3` — grid current model: when Eg > 0 (grid conduction), current
  flows through R1 and diode D3 to cathode.

### 6.2 Pentode Subcircuit

Directly from Koren's tube.lib (6550 example):

```spice
.SUBCKT 6550 A G K G2
+ PARAMS: MU=7.9 EX=1.35 KG1=890 KG2=4200 KP=60 KVB=24
+ CCG=14P CPG1=0.85P CCP=12P RGI=1000
*
* E1: intermediate Koren voltage (screen-grid controlled)
E1 7 0 VALUE={V(G2,K)/KP*LOG(1+EXP((1/MU+V(G,K)/V(G2,K))*KP))}
RE1 7 0 1G
*
* G1: anode current source (with arctan knee)
G1 A K VALUE={(PWR(V(7),EX)+PWRS(V(7),EX))/KG1*ATAN(V(A,K)/KVB)}
*
* G2: screen current source
G2 G2 K VALUE={(EXP(EX*(LOG((V(G2,K)/MU)+V(G,K)))))/KG2}
*
RCP A K 1G
*
* Interelectrode capacitances
C1 G K {CCG}        ; cathode-grid1
C2 A G {CPG1}       ; grid1-plate
C3 A K {CCP}        ; cathode-plate
*
* Grid current model
R1 G 5 {RGI}
D3 5 K DX
.MODEL DX D(IS=1N RS=1 CJO=10PF TT=1N)
*
.ENDS 6550
```

**Key differences from triode:**
- **4 terminals:** A (anode), G (control grid), K (cathode), G2 (screen)
- **E1 uses `V(G2,K)` instead of `V(A,K)`** — screen controls current
- **G1 has `*ATAN(V(A,K)/KVB)`** — the arctan knee factor
- **G2 source** — separate screen current equation
- **KG2 parameter** — screen current scale

### 6.3 SPICE Functions: PWR, PWRS, LOG, EXP

| Function   | Definition                     | Notes                          |
|------------|--------------------------------|--------------------------------|
| `PWR(x,n)` | `|x|^n`                       | Always positive                |
| `PWRS(x,n)`| `sign(x) * |x|^n`             | Preserves sign                 |
| `LOG(x)`   | Natural logarithm `ln(x)`     | NOT log base 10                |
| `EXP(x)`   | `e^x`                         |                                |
| `ATAN(x)`  | `arctan(x)` in radians         | Range: `-π/2` to `+π/2`       |
| `SQRT(x)`  | `√x`                          |                                |

**PWR + PWRS trick:** The sum `PWR(x,n) + PWRS(x,n)` equals:
- `2 * x^n` when x > 0 (both terms are positive)
- `0` when x <= 0 (`|x|^n - |x|^n = 0`)

This is a clever SPICE idiom for implementing `max(x, 0)^n * 2` without
an IF statement (which many SPICE versions don't support in behavioral
sources).

---

## 7. Published Parameters

From Norman Koren's Table 1 and tube.lib:

### Triodes

| Tube   | mu    | Ex   | Kg1   | Kp   | Kvb   | CCG   | CGP   | CCP   | RGI  |
|--------|-------|------|-------|------|-------|-------|-------|-------|------|
| 12AX7  | 100   | 1.4  | 1060  | 600  | 300   | 2.3p  | 2.4p  | 0.9p  | 2k   |
| 12AU7  | 21.5  | 1.3  | 1180  | 84   | 300   | 2.3p  | 2.2p  | 1.0p  | 2k   |
| 6DJ8   | 28    | 1.3  | 330   | 320  | 300   | 2.3p  | 2.1p  | 0.7p  | 2k   |

### Pentodes (beam tetrodes)

| Tube   | mu   | Ex   | Kg1  | Kg2  | Kp  | Kvb | CCG  | CPG1  | CCP  | RGI |
|--------|------|------|------|------|-----|-----|------|-------|------|-----|
| 6L6GC  | 8.7  | 1.35 | 1460 | 4500 | 48  | 12  | 14p  | 0.85p | 12p  | 1k  |
| 6550   | 7.9  | 1.35 | 890  | 4200 | 60  | 24  | 14p  | 0.85p | 12p  | 1k  |
| KT88   | 8.8  | 1.35 | 730  | 4200 | 32  | 16  | 14p  | 0.85p | 12p  | 1k  |
| EL34   | 11   | 1.35 | 650  | 4200 | 60  | 24  | 14p  | 0.85p | 12p  | 1k  |

**Notes:**
- Capacitances include ~0.7pF for tube socket with 2" leads (adjacent pins)
  and ~0.5pF for non-adjacent pins.
- Pentode parameters (mu, Ex, Kg1, Kp) are derived from **triode-mode**
  curves, which gives the best fit.
- Kg2 is estimated from published screen current data at a single
  operating point.

---

## 8. Python Implementation

### 8.1 Triode Model Function

```python
def _koren_ia_triode(ua, ug1, mu, ex, kg1, kp, kvb):
    """Koren triode plate current (vectorized, amps).

    Args:
        ua:  anode voltage array (V), positive
        ug1: grid voltage array (V), negative for normal operation
        mu, ex, kg1, kp, kvb: Koren model parameters

    Returns:
        Ia in amperes (A), same shape as ua/ug1.
    """
    ua = np.maximum(ua, 0.01)                    # avoid division by zero
    arg = kp * (1.0 / mu + ug1 / np.sqrt(kvb + ua * ua))
    arg = np.clip(arg, -50, 50)                  # prevent exp overflow
    e1 = (ua / kp) * np.log1p(np.exp(arg))       # log1p = log(1+x), better precision
    abs_e1 = np.maximum(np.abs(e1), 0.0)
    pwr = np.power(abs_e1, ex)                    # |E1|^Ex
    pwrs = np.sign(e1) * pwr                      # sign(E1)*|E1|^Ex
    ia = (pwr + pwrs) / kg1                        # 2*E1^Ex/Kg1 for E1>0
    return np.maximum(ia, 0.0)
```

### 8.2 Pentode Model Function

```python
def _koren_ia_pentode(ua, ug1, ug2, mu, ex, kg1, kp, kvb):
    """Koren pentode plate current (vectorized, amps).

    Key differences from triode:
    - E1 scaled by Ug2/Kp (not Ua/Kp)
    - Grid term is Ug1/Ug2 (not Ug1/sqrt(Kvb + Ua^2))
    - Additional arctan(Ua/Kvb) knee factor

    Args:
        ua:  anode voltage array (V)
        ug1: control grid voltage array (V), negative
        ug2: screen grid voltage array (V), positive
        mu, ex, kg1, kp, kvb: Koren model parameters

    Returns:
        Ia in amperes (A).
    """
    ua = np.maximum(ua, 0.01)
    ug2 = np.maximum(ug2, 0.01)                   # prevent division by zero

    arg = kp * (1.0 / mu + ug1 / ug2)
    arg = np.clip(arg, -50, 50)
    e1 = (ug2 / kp) * np.log1p(np.exp(arg))

    abs_e1 = np.maximum(np.abs(e1), 0.0)
    pwr = np.power(abs_e1, ex)
    pwrs = np.sign(e1) * pwr
    ia = (pwr + pwrs) / kg1 * np.arctan(ua / kvb)
    return np.maximum(ia, 0.0)
```

### 8.3 Pentode Screen Current Function

```python
def _koren_ig2_pentode(ug1, ug2, mu, ex, kg2):
    """Koren pentode screen grid current (vectorized, amps).

    Uses the simplified Reynolds/Leach form:
    Ig2 = (Ug1 + Ug2/mu)^Ex / Kg2

    Note: uses np.maximum instead of exp(ex*log(x)) to avoid
    log of negative numbers.

    Args:
        ug1: control grid voltage array (V), negative
        ug2: screen grid voltage array (V), positive
        mu:  amplification factor
        ex:  current exponent
        kg2: screen current scale

    Returns:
        Ig2 in amperes (A).
    """
    v = ug1 + ug2 / mu                           # effective control voltage
    v_pos = np.maximum(v, 0.0)                    # clamp to non-negative
    ig2 = np.power(v_pos, ex) / kg2
    return ig2
```

### 8.4 Numerical Safety

| Issue                   | Solution                                   | Why                                    |
|-------------------------|--------------------------------------------|----------------------------------------|
| Ua = 0                  | `ua = max(ua, 0.01)`                       | Prevents E1=0 and numerical instability|
| Ug2 = 0 (pentode)      | `ug2 = max(ug2, 0.01)`                     | Prevents division by zero in E1        |
| exp() overflow          | `arg = clip(arg, -50, 50)`                 | `exp(50) ≈ 5e21`, safe for float64    |
| log of negative         | Use `np.maximum(v, 0)` before power        | Pentode Ig2: v can be negative         |
| arctan normalization    | None needed — absorbed into Kg1             | `arctan(∞) = π/2 ≈ 1.571`             |
| Very small E1           | `np.maximum(abs(e1), 0.0)` before power    | Prevents negative base in power()      |

---

## 9. Fitting Strategy

### 9.1 Data Requirements

**Triode:**
- Input: arrays of Ua, Ug1, Ia (in mA, converted to A before fitting)
- Filter: Ia > 0.1 mA (below this is noise/cutoff)
- Minimum: 10 points (5 parameters)
- Better: 30+ points from a full scan

**Pentode:**
- Input: arrays of Ua, Ug1, Ug2, Ia, Ig2 (in mA, converted to A)
- Filter: Ia > 0.1 mA AND Ug2 > 10V
- Minimum: 15 points (6 parameters)
- Better: 50+ points with varying Ug2

### 9.2 Triode Fitting

Current implementation uses:
1. **scipy.optimize.least_squares** (preferred) — Trust Region Reflective
   method, bounded optimization, up to 5000 function evaluations
2. **numpy fallback** — Coordinate descent with golden-section line search,
   30 outer iterations, 5 params x 40 line search steps

Initial guess: from tube_params.json reference values, or estimated from
voltage ratios with 12AX7 defaults.

### 9.3 Pentode Fitting — Two-Phase Approach

Fitting all 6 pentode parameters simultaneously can be unstable because
some parameters are correlated. Koren's recommended approach:

**Phase 1 — Triode mode (if available):**

When the scan includes triode-connected data (Ug2 = Ua, "triode_connected"
mode), fit the triode model to get mu, Ex, Kg1, Kp. These parameters
are the same in both modes, and the triode equation is better conditioned
for fitting.

**Phase 2 — Pentode mode:**

Using Phase 1 results (or tube_params.json references) as initial guess,
fit all 6 parameters to pentode-mode data (Ug2 ≠ Ua).

**If no triode-mode data:** Fit all 6 parameters directly, relying on
good initial guess from tube_params.json.

### 9.4 Residual Function

**Triode (single objective):**

```python
residual = ia_pred - ia_meas
```

**Pentode (combined Ia + Ig2 objective):**

```python
r_ia  = ia_pred  - ia_meas          # plate current residuals (main)
r_ig2 = (ig2_pred - ig2_meas) * w   # screen current (lower weight)
residual = np.concatenate([r_ia, r_ig2])
```

Where `w` = 0.3-0.5 (Ig2 weight). Lower weight because screen current
is less critical and less accurately measured.

If Ig2 data is not available, only Ia residuals are used and Kg2 is taken
from the reference database.

### 9.5 Parameter Bounds

**Triode:**

```
     mu:   [2.0,    500.0]
     Ex:   [0.8,    3.5]
     Kg1:  [10.0,   50000.0]
     Kp:   [20.0,   3000.0]
     Kvb:  [5.0,    3000.0]   (V²)
```

**Pentode:**

```
     mu:   [2.0,    500.0]
     Ex:   [0.8,    3.5]
     Kg1:  [10.0,   50000.0]
     Kg2:  [500.0,  20000.0]
     Kp:   [10.0,   3000.0]
     Kvb:  [1.0,    500.0]    (V)
```

Note: Kvb bounds differ significantly between triode (V²) and pentode (V).

### 9.6 Kg2 — Special Handling

Per Koren: "KG2 does not need to be estimated with great accuracy."

Strategy:
1. If Ig2 measurement data available → fit Kg2 with lower weight
2. If no Ig2 data → use Kg2 from tube_params.json
3. If no reference → default Kg2 = 4500

---

## 10. Finding Parameters by Trial and Error

Koren's manual procedure (useful for understanding parameter sensitivity):

1. **mu:** Read from datasheet, or measure from plate curves:
   `mu = -dEp/dEg` at constant Ip. Choose a region far from Region A.

2. **Ex and Kg1:** Adjust until curves at low negative Eg (e.g., Vg = 0
   or -0.5V for 12AX7) match published data. Kg1 is inversely proportional
   to current. Ex controls curvature (1.3 = more linear, 1.5 = more curved).

3. **Kp:** Trial and error to match Region A (large Ep, large negative Eg,
   low current). For pentodes, use triode-mode curves.

4. **Kvb (triode):** Adjust to match the knee region at positive grid
   voltage. Rarely published, so often an educated guess.

5. **Kvb (pentode):** Proportional to the knee visible in pentode-mode
   curves. Match the knee at Eg = 0 (where the load line typically passes).

6. **Kg2 (pentode):** Set from a single published screen current data
   point: `Kg2 = (Eg + Eg2/mu)^Ex / Ig2`.

---

## 11. Fit Quality Metrics

For both models:

```python
ia_pred = model(ua, ug1, ...)
diff_mA = (ia_pred - ia_meas) * 1000.0
rms_error = sqrt(mean(diff_mA²))
max_error = max(|diff_mA|)
```

**Typical good fit:**
- Small-signal triode (12AX7): RMS < 0.1 mA, Max < 0.5 mA
- Power triode: RMS < 1 mA, Max < 5 mA
- Power pentode: RMS < 2 mA, Max < 10 mA

For pentode, additionally:
```python
ig2_pred = _koren_ig2_pentode(ug1, ug2, mu, ex, kg2)
ig2_diff_mA = (ig2_pred - ig2_meas) * 1000.0
ig2_rms = sqrt(mean(ig2_diff_mA²))
```

---

## 12. Topology Detection and Mode Selection

```python
def fit_and_export_spice(path, tube_type, points, topology=None):
    """
    topology:
      "triode"            → triode model (3-pin)
      "pentode"           → pentode model (4-pin)
      "triode_connected"  → pentode wired as triode → triode model
      None                → auto-detect from tube_params.json
    """
```

Auto-detection logic:
1. Look up tube in tube_params.json → get topology field
2. If not found, check measurement data:
   - If all Ug2 ≈ 0 or Ug2 = Ua → triode
   - If Ug2 is independent and > 0 → pentode
3. For "triode_connected": use triode model (Ug2 = Ua, triode equations
   are more appropriate)

---

## 13. Edge Cases and Numerical Safety

1. **Division by zero:** Ua=0 → `max(ua, 0.01)`, Ug2=0 → `max(ug2, 0.01)`
2. **Exp overflow:** `clip(arg, -50, 50)` → `exp(50) ≈ 5.18e21` (safe)
3. **Log domain:** Use `np.maximum(v, 0)` + `np.power()` instead of
   `exp(n*log(v))` for negative v
4. **arctan range:** `arctan(Ua/Kvb)` ranges 0..π/2; the factor 2/π is
   absorbed into Kg1 during fitting, so no explicit normalization needed
5. **Mixed topology data:** Separate Ug2=Ua points from independent-Ug2
   points before fitting
6. **Missing Ig2:** Use reference Kg2 or default 4500
7. **Very few points:** Minimum 10 (triode) or 15 (pentode); raise
   RuntimeError if insufficient

---

## 14. Verification Examples

### 12AX7 (Triode)

```
Parameters: mu=100, Ex=1.4, Kg1=1060, Kp=600, Kvb=300

Ua=250V, Ug1=-2V:
  arg = 600*(0.01 + (-2)/sqrt(300+62500)) = 600*0.00202 = 1.21
  e1  = (250/600)*ln(1+exp(1.21)) = 0.417*1.472 = 0.613
  Ia  = 2*0.613^1.4/1060 = 2*0.505/1060 = 0.95 mA ✓

Ua=100V, Ug1=-1V:
  arg = 600*(0.01 + (-1)/sqrt(300+10000)) = 600*0.000147 = 0.088
  e1  = (100/600)*ln(1+exp(0.088)) = 0.167*0.738 = 0.123
  Ia  = 2*0.123^1.4/1060 = 2*0.053/1060 = 0.10 mA ✓
```

### 6550 (Pentode)

```
Parameters: mu=7.9, Ex=1.35, Kg1=890, Kg2=4200, Kp=60, Kvb=24

Ua=300V, Ug2=300V, Ug1=0:
  arg = 60*(1/7.9 + 0/300) = 60*0.1266 = 7.59
  e1  = (300/60)*ln(1+exp(7.59)) = 5.0*7.59 = 37.97
  arctan(300/24) = arctan(12.5) = 1.491
  Ia  = 2*37.97^1.35/890*1.491
      = 2*135.6/890*1.491
      = 0.454 A = 454 mA

Published: Ia ≈ 455 mA at this point — excellent match.

Screen current at same point:
  Ig2 = (0 + 300/7.9)^1.35 / 4200
      = 37.97^1.35 / 4200
      = 135.6 / 4200
      = 0.032 A = 32 mA
```

### EL34 (Pentode in triode mode)

```
Parameters: mu=11, Ex=1.35, Kg1=650, Kp=60, Kvb=24

Triode mode (Ug2=Ua): Ua=250V, Ug1=-10V:
  Using pentode equation with Ug2=Ua=250:
  arg = 60*(1/11 + (-10)/250) = 60*(0.0909 - 0.04) = 60*0.0509 = 3.05
  e1  = (250/60)*ln(1+exp(3.05)) = 4.167*3.097 = 12.90
  arctan(250/24) = arctan(10.42) = 1.475
  Ia  = 2*12.90^1.35/650*1.475
      = 2*31.4/650*1.475
      = 0.142 A = 142 mA
```

---

*Document version: 2.0 — February 2026*
*For LM19 Tube Tester application*

## Sources

Materials this document rests on. The full registry of the project
external sources (all entries, statuses, local copies) — `SOURCES_INDEX.md`.

### Norman Koren (model theory and Tuparam data)
- url: <https://www.normankoren.com/Audio/Tube_params.html>
- type: theory
- note: Core equations and fitting reference for SPICE export/tests.

### Norman Koren — Improved Vacuum Tube Models for SPICE Simulations (1996)
- url: <https://www.normankoren.com/Audio/Tubemodspice_article.html>
- type: theory
- note: Full article with Koren model equations. Triode:
  E1=(Va/Kp)·ln(1+exp(Kp·(1/µ+Vg/√(Kvb+Va²)))), Ia=2·E1^Ex/Kg1. Pentode: E1
  with Vg2, Ia·arctan(Va/Kvb). Known issues (per Reefman): Kg1/Kp
  correlation, Ig2 independent of Va, pentode-as-triode inconsistency.
  Companion page with parameter tables saved as
  koren_improved_tube_models.html.
