# Amplifier Calculations Reference

How LM19 computes amplifier parameters — formulas, methods, and code mapping.

---

## Units Convention

All internal calculations use:
- **Voltage:** V (Volts)
- **Current:** mA (milliamperes)
- **Resistance:** kΩ (kilohms)
- **Power:** mW (milliwatts); displayed as W where > 1000 mW
- **Distortion:** % (percent of fundamental)

Key conversion: `V × mA = mW` (no factor needed).

---

## 1. Load Lines

The load line defines the relationship between anode voltage and current
for a given circuit topology.

### SE Resistive

```
Ia = (Ub − Ua) / Ra
```

- `Ub` — supply voltage (V)
- `Ua` — anode voltage (V)
- `Ra` — load resistance (kΩ)

Code: `ResistiveLoadLine.ia_at_ua()` in `amplifier.py`

### SE Transformer

Two load lines: DC (through winding resistance) and AC (through reflected load).

```
DC:  Ia = (Ub − Ua) / Ra_dc        (Q-point lies on this line)
AC:  Ia = Ia_q − (Ua − Ua_q) / Ra_ac   (signal swings along this line)
```

- `Ra_dc` — primary winding DC resistance (kΩ), typically 0.02–0.5 kΩ
- `Ra_ac` — reflected impedance (kΩ) = `n² × Rload`
- `Ua_q, Ia_q` — quiescent operating point

Code: `TransformerLoadLine` in `amplifier.py`

### Cathode Follower

Load is in the cathode circuit:

```
Ia = (Ub − Ua) / (Rk + Rl)
```

- `Rk` — cathode bias resistor (kΩ)
- `Rl` — AC load resistor (kΩ)

Code: `CathodeFollowerLoadLine` in `amplifier.py`

### Push-Pull

Center-tapped transformer, each tube sees 1/4 of anode-to-anode impedance:

```
Ra_per_tube = Ra_aa / 4
Ia = (Ub − Ua) / Ra_per_tube
```

Code: `PushPullLoadLine` in `amplifier.py`

---

## 2. Harmonic Distortion Analysis

Three methods available, selectable in UI via HD Method dropdown.

### 2a. 5-Point Method (default)

Based on Radiotron Designer's Handbook (Langford-Smith, 1953).

Samples the Ia(Ug1) transfer characteristic at 5 points spaced at
0°, ±60°, ±90° of a sinusoidal signal cycle:

```
i_0        = Ia at Ug1_bias                    (center)
i_max      = Ia at Ug1_bias + half_swing       (positive peak)
i_min      = Ia at Ug1_bias − half_swing       (negative peak)
i_high_half = Ia at Ug1_bias + half_swing/2    (60° point)
i_low_half  = Ia at Ug1_bias − half_swing/2    (60° point)
```

Fourier coefficients:

```
B1 = (i_max − i_min + i_high_half − i_low_half) / 3     (fundamental)
B2 = (i_max + i_min − 2·i_0) / 4                        (2nd harmonic)
B3 = (i_max − i_min − 2·(i_high_half − i_low_half)) / 6 (3rd harmonic)
```

Distortion:

```
HD2 = |B2 / B1| × 100%
HD3 = |B3 / B1| × 100%
THD = √(HD2² + HD3²)
```

Output power:

```
Pout = (Ia_pp × Ua_pp) / 8    [mW]
```

where `Ia_pp = i_max − i_min` (mA), `Ua_pp = |Ua_min − Ua_max|` (V).

**Strengths:** Fast, no fitting needed, works with sparse data (≥ 5 intersections).
**Limitations:** Only HD2 + HD3. Linear interpolation between measured points.

Guard: `B1 < MIN_B1_MA (0.01 mA)` → result rejected (prevents garbage at very low swing).

Code: `compute_distortion()` in `amplifier.py`

### 2b. Chebyshev Polynomial Method

Decomposes Ia(Ug1) along the load line into Chebyshev polynomial series.
Key identity: `Tn(cos θ) = cos(nθ)` — coefficients directly give harmonic amplitudes.

```
1. Normalize Ug1 → u ∈ [−1, +1]:  u = (Ug1 − bias) / half_swing
2. Fit:  Ia(u) ≈ Σ cn·Tn(u)      (Chebyshev polynomial fit)
3. HDn = |cn / c1| × 100%         (nth harmonic distortion)
4. THD = √(Σ HDn²)                (n = 2..max_harmonic)
```

Overfit protection: `max_harmonic ≤ 2·n_points / 3` (`CHEBYSHEV_OVERFIT_RATIO = 3`).
Minimum: `max_harmonic ≥ 3` (HD2 + HD3 at least).

**Strengths:** HD2–HD9 from measurement data. No sampling artifacts.
**Limitations:** Needs ≥ 7 points for HD2–HD4. Can overfit with few points.
**Best for:** Measured data with 10–30 intersection points.

Code: `compute_distortion_chebyshev()` in `amplifier.py`

### 2c. DFT Method (model only)

Generates a synthetic sinusoidal drive, computes Ia(t) using the tube model
+ load line, then applies FFT to extract harmonics.

```
1. Ug1(t) = bias + half_swing × cos(2π·t/N)    (N = 1024 samples)
2. For each t: solve Ua where model.ia(Ua, Ug1) = load_line.ia(Ua)
   (fixed-point iteration with Newton-Raphson refinement)
3. FFT:  spectrum = rfft(Ia(t))
4. HDn = |spectrum[n]| / |spectrum[1]| × 100%
5. THD = √(Σ HDn²)
```

**Strengths:** HD2–HD9, infinite resolution, no interpolation artifacts.
**Limitations:** Requires a fitted model (Koren/Dempwolf/Reefman). Slower.
**Best for:** Fitted models — most accurate.

Code: `compute_distortion_dft()` in `amplifier.py`

### Method selection (auto mode)

```
source = "measurements" → Chebyshev
source = model name     → DFT (requires model; if unavailable → 5-point)
Chebyshev returns None  → fallback to 5-point
DFT without model       → fallback to 5-point
```

Code: `_resolve_hd_method()` in `amp_engine.py` (selects method by source name).
Dispatcher: `_compute_hd()` in `amplifier.py` (handles fallback).

---

## 3. Intermodulation Distortion (IMD)

Polynomial fit of Ia(Ug1) around the operating point:

```
Ia(x) = a0 + a1·x + a2·x² + a3·x³     where x = Ug1 − Ug1_bias

IMD2 = |a2 / a1| × 100%    (2nd-order nonlinearity)
IMD3 = |a3 / a1| × 100%    (3rd-order nonlinearity)
IMD_total = √(IMD2² + IMD3²)
```

This is a **lumped nonlinearity** measure, not a true two-tone IMD test.
It indicates the "distortion makeup" — relative strength of even vs odd nonlinearity.

Code: `compute_imd()` in `amplifier.py`

---

## 4. Stage Parameters

### Small-signal parameters (gm, ra, μ)

Three sources, in priority order:

1. **Model** (highest accuracy): central finite differences on `model.ia()`:
   ```
   gm = ΔIa / ΔUg1  at constant Ua    (Δ = 0.05V)
   ra = ΔUa / ΔIa   at constant Ug1   (Δ = 1.0V)
   μ  = gm × ra
   ```

2. **Numerical** (from measurements): robust regression (MAD outlier rejection)
   on measurement points near Q-point (within ±15V Ua window).

3. **SRK** (fallback): from manual S/R/K measurements.
   Cross-check: warns if SRK diverges > 20% from numerical values.

### Voltage gain

**SE (common cathode):**
```
Gain = μ × Ra / (ra + Ra)
Gain_dB = 20 × log₁₀(Gain)
```

**Cathode Follower:**
```
Gain = μ × Rk / (ra + (μ + 1) × Rk)    (≈ 1 for large Rk)
```

### Output impedance

**SE:**
```
Zout = ra × Ra / (ra + Ra)    (ra ∥ Ra)
```

**CF:**
```
Zout = ra / (μ + 1)    ≈ 1/gm
```

### Damping Factor

```
DF = Ra_load / Zout
```

Higher DF = better load control (less output impedance relative to load).

### Auto-bias cathode resistor

```
Rk = |Ug1_bias| / Ik
Ik = Ia + Ig2     (pentode: screen current adds to cathode current)
```

Code: `compute_stage_params()`, `model_gm_ra()`, `_numerical_gm_ra()` in `amplifier.py`

---

## 5. Negative Feedback (NFB)

Classical linear feedback theory. User specifies NFB amount in dB.

```
D = 10^(NFB_dB / 20)              (desensitivity factor)
β = (D − 1) / Gain_open           (feedback fraction)

Gain_closed = Gain_open / D
Zout_closed = Zout_open / D
THD_closed  = THD_open / D
BW_factor   = D                   (bandwidth extended by D)
DF_closed   = DF_open × D
```

**Example:** 6 dB NFB → D = 2.0 → gain halved, THD halved, Zout halved.

**Limitations:** Assumes linear small-signal feedback. At very high NFB (> 20 dB),
THD reduction is optimistic. Does not model phase margin or loop stability.

Code: `compute_nfb_effect()` in `amplifier.py`

---

## 6. Headroom (Maximum Swing)

Determines the maximum symmetric signal amplitude before clipping.
Four limits are evaluated:

1. **Cutoff:** Ia → 0 on the negative swing (tube stops conducting)
2. **Grid current:** Ug1 → 0 V on the positive swing (grid starts drawing current)
3. **Pa_max:** Ua × Ia exceeds thermal dissipation limit at any point
4. **Data limit:** ran out of measured Ug1 range

Grid current is modeled via Dempwolf parameters when available:
```
Ig1 = Gg × softplus(Cg × Ug1)^ξ     [Amperes]
softplus(x) = ln(1 + e^x)
```

Otherwise a hard threshold at Ug1 = −0.1V is used.

Result: `max_swing = min(swing_neg, swing_pos)` (V).

Code: `compute_headroom()` in `amplifier.py`

---

## 7. Average Plate Dissipation

Numerical integration of Pa = Ua × Ia over one signal cycle (64 samples).
Accurate for all operating classes (A, AB, B).

```
Pa_avg = (1/N) × Σ Ua(θ) × Ia(θ)      for θ = 0..2π

Pa at Q-point (no signal):   Pa_q = Ua_q × Ia_q
Pa under signal (Class A):   Pa_signal ≈ Pa_q − Pout
Pa under signal (Class AB):  Pa_signal computed numerically (Pa_q − Pout is inaccurate)
```

For each sample θ, Ug1(θ) = bias + swing × sin(θ), then Ua is found by solving
the load line equation with Newton-Raphson iteration.

Code: `compute_pa_avg()` in `amplifier.py`

---

## 8. Push-Pull Analysis

### Composite characteristic

Mirrors tube B's transfer curve around the bias point and subtracts:

```
Ia_composite(Ug1) = Ia_A(Ug1) − Ia_B(2·bias − Ug1)
```

For **matched pair:** composite is odd-symmetric → even harmonics (HD2) cancel.
For **mismatched pair:** HD2 residual = balance error.

### PP distortion

5-point method applied to the composite characteristic:
```
B1, B2, B3 → HD2, HD3, THD    (same formulas as SE)
balance_error = HD2            (measure of tube mismatch)
```

### PP output power

```
Ua_swing = Ia_pp × Ra_per_tube    (V)
Pout = (Ia_pp × Ua_swing) / 8    (mW)
```

where `Ra_per_tube = Ra_aa / 4` for center-tapped transformer.

Code: `composite_characteristic()`, `pp_distortion()` in `amplifier.py`

---

## 9. Optimizer

Multi-parameter search for optimal operating point.

### HD method (`OptimizerConstraints.hd_method`)

User selects the harmonic distortion method via UI combobox. The
optimizer uses different methods for grid (fast pre-selection) and
refine (final accurate ranking) phases:

| `hd_method` | Grid phase | Refine top-N (Pareto) |
|---|---|---|
| `5point` | 5-point | 5-point |
| `chebyshev` | Chebyshev | Chebyshev |
| `dft` | DFT (requires model) | DFT |
| `auto` | Chebyshev | DFT (if model) else Chebyshev |

`auto` is the sensible default: fast Chebyshev grid catches sparse-data
degeneracies (returns None when sample density is insufficient), then
DFT refines the Pareto candidates for ground-truth THD when a tube
model is fitted.

If `dft` is requested but no model is fitted, optimizer falls back to
Chebyshev and sets `result.warning = "dft_no_model_fallback"`.

**5-point sparse-data guard:** before extracting harmonics,
`compute_distortion()` requires ≥ `MIN_CURVES_IN_SWING = 3` actual
measurement Ug1 curves to fall within the swing window, with at least
one strictly inside. Otherwise the 5 sample points become linear
interpolations between only 2 data lines, b2/b3 ≈ 0 algebraically, and
THD reports as fake-near-zero. Diagnostic code: `DIST_ERR_SPARSE_DATA`.

**Push-pull HD methods:** PP supports the same 4 methods as
SE. Implemented as separate functions in `amplifier.py`:

| `hd_method` | PP function | Source |
|---|---|---|
| `5point` | `pp_distortion()` | composite + 5-point |
| `chebyshev` | `compute_distortion_chebyshev_pp()` | composite + Chebyshev fit |
| `dft` | `compute_distortion_dft_pp()` | model-based DFT on composite |
| `auto` | grid Chebyshev, refine DFT (if model) | dispatcher |

For matched pair the composite Ia(Ug1) is mathematically odd-symmetric
around bias → even Chebyshev coefficients ≈ 0 → HD2,HD4 ≈ 0 (correct
physics, not artifact). DFT on the model is the smoothest variant and
typically gives the lowest THD for matched pair (no measurement noise);
Chebyshev/5-point on real measurements may report higher THD due to
data discretization. All three labels appear as `OptPoint.hd_method`:
`5point` / `chebyshev_pp` / `dft_pp`.

**Self-consistent Ua/Ia solve in `compute_distortion_dft_pp`:** at
each time sample Newton-iterates on tube curve
`Ia = model.ia(Ua, Ug1, Ug2)` ∩ load line
`Ia = (Ub − Ua)/Ra_per_tube`. Critical for
`UltralinearModelWrapper` / Triode mode — without iteration Ua stays
≈ Ub constant and the UL screen-tap formula
`Ug2_eff = Ug2_nom·(1−tap) + Ua·tap` gives `Ug2_eff = Ub` regardless of
tap → UL effect invisible. With the proper solve, varying Ua during the
signal cycle modulates Ug2 → captures correct ultralinear linearisation
(lower THD, lower Pout vs pentode) and triode mode (much lower Pout,
lowest THD).

**EL84 PP physical sanity vs canonical references:**

| Mode | Iq | Pout @ ±9V | THD | vs Mullard 5-10 / Brimar / Philips |
|---|---|---|---|---|
| Pentode (Ug2=300V) | 60mA | 17.6W | 8.8% | datasheet ~17W @ 5–10% ✅ |
| UL 43% (Williamson) | 42mA | 11.0W | 3.5% | reference ~13W @ 1–3% ≈ |
| Triode (UL tap=1.0) | 30mA | 5.5W | 2.0% | reference ~5W @ 1–2% ✅ |

Ratios match published Aiken «UL myths» rule of thumb: UL Pout = 60–90%
of Pentode, Triode = 25–50% of Pentode. Lower ratios in our model arise
from the `Ra_per_tube = Ra_aa/4` class-A approximation (impedance
doubling at AB cutoff not modelled).

### Resistive vs Transformer load — pitfall for output stages

For SE/PP **output pentode amplifiers** the plate is wired to Ub through
the **primary of an output transformer**, not a resistor. Mixing these
up gives nonsense Q-points:

|  | ResistiveLoadLine | TransformerLoadLine |
|---|---|---|
| DC behaviour | Ia × Ra drops on resistor | Ia × Ra_dc (~50–500 Ω) — negligible |
| Q-point Ua | `Ub − Ia·Ra` (low if Ra·Ia ≳ Ub) | `Ub − Ia·Ra_dc ≈ Ub` |
| AC swing | Same as DC (resistor is non-reactive) | Reflected `n²·Rload`, separate slope |

For EL84 SE pentode at Ub=250V, Ia_q=48mA, "Ra"=5.2k:
- **ResistiveLoadLine**: Ua_q = 250 − 48·5.2 = **10 V** → tube saturates → ~0.5 W achievable
- **TransformerLoadLine** (ra_dc=0.1, ra_ac=5.2): Ua_q ≈ **245 V**, full ±5 V swing → **3–5 W** matching Mullard datasheet

When to use:
- **Resistive**: driver/voltage-amp stages where the output drives a high-impedance grid (Ra typically 50–250 kΩ, Ia low, voltage drop budget is fine).
- **Transformer**: every output power stage feeding a loudspeaker, headphone, or low-impedance load.
- **Cathode follower**: separate `CathodeFollowerLoadLine` (cathode resistor + plate to Ub direct).

**EL84 SE pentode physical sanity** (TransformerLoadLine, ra_dc=0.1, ra_ac=5.2):

| Mode | Pout @ ±5V | THD | vs Mullard SE pentode datasheet (3.4W @ 10%) |
|---|---|---|---|
| Pentode | 4.1 W | 5.8% | ✅ within 25% |
| UL 43% | 2.5 W (60%) | 3.6% (60%) | ✅ matches Williamson SE |
| Triode | 0.95 W (23%) | 3.5% (60%) | ✅ matches SE triode reference |

### Three phases

1. **Grid sweep** — coarse search over Ub × Ug2 × Ug1 × Ra. Ub is a
   *virtual* analysis parameter: the load line `Ia = (Ub − Ua)/Ra`
   shifts over the same measured I-V family, so Ub-sweep works on
   raw measurements too (not only fitted models) and applies to all
   circuit topologies (SE / SE-XFMR / CF / PP). Ub varies when
   `constraints.ub_range != None`; otherwise fixed at the supplied
   `ub`. Ug2 varies in model path (continuous range) or iterates over
   discrete data values in measurements path.

2. **Swing sweep** — top-20 points re-evaluated at multiple swing levels
   (30%–100% of max swing, `DEFAULT_SWING_STEPS = 5`). Intersections
   cached per `(Ub, Ug2, Ra)` to avoid recomputation.

3. **Pareto refinement** — scipy Nelder-Mead on up to 8 Pareto front
   points in parallel (`ThreadPoolExecutor`, 4 workers). The vector
   includes Ub when `ub_range != None`, Ug2 when model + `ug2_range`,
   plus Ug1, Ra, and swing.

### Optimization targets

| Target | Score function |
|--------|---------------|
| `min_thd` | THD (lower = better) |
| `max_pout` | −Pout_mW (more negative = more power) |
| `balanced` | THD − 0.5 × log₁₀(Pout_mW) (trade-off) |

### Pareto front

Set of points where you can't improve THD without reducing Pout (and vice versa).
Computed by sorting valid points by THD ascending, keeping those with
increasing Pout.

### Constraints

- `pa_max_w` — anode dissipation limit (W)
- `pout_min_w` — minimum output power (W)
- `amp_class` — "A" / "AB" / "any"
- `class_a_power_mode` + `class_a_power_value` — PP class-A power threshold (see below)

### Class A power threshold (PP only)

In a push-pull amplifier, class-A operation persists only while both
tubes conduct. The transition to class AB occurs when the off-going
tube reaches cutoff at signal current swing `ΔI = Iq` (per-tube
quiescent current). The corresponding output power:

```
P_A = Iq² × Ra_aa / 8         (Iq in A, Ra_aa in Ω, P_A in W)
```

Derivation: in PP class-A, each tube sees `Ra_aa/4`. AC voltage on the
full primary at boundary: `V_pk = ΔI × Ra_aa/2 = Iq × Ra_aa/2`. Output
power: `P = V_pk² / (2·Ra_aa) = Iq² × Ra_aa / 8`.

Sanity: 6L6 triode PP, Iq=109 mA, Ra_aa=5 kΩ → P_A = 7.4 W (matches
the empirical "7.5 W class A before cutoff").

The optimizer accepts the threshold in two modes:
- `absolute` — minimum P_A in watts (`class_a_power_value` is W)
- `percent` — fraction of point's max output (`class_a_power_value` × Pout/100)
- `off` — no filter

Filter applies to PP only; for SE/SE-XFMR/CF the field `p_classA_w` in
`OptPoint` stays at 0.

Sources: Aiken «Last Word on Class A», sound-au.com Class A.

### Circuit support

All 4 circuit types supported via `_make_load_line()`:
SE, SE Transformer, Cathode Follower, Push-Pull.
PP uses `pp_distortion()` instead of `compute_distortion()`.

Code: `optimizer.py`, `optimize_worker.py`

---

## 10. SPICE Export

### Subcircuit (.sub)

Three model types: Koren, Dempwolf, Reefman. Fitted to measurements
using scipy (or numpy fallback). Subcircuit name = filename stem.

### Amplifier schematics (.asc)

10 templates: SE triode/pentode, SE Transformer triode/pentode,
CF triode/pentode, PP triode/pentode, test triode/pentode.

Transformer model:
```
L_primary   = Ra / (2π × f_low)       (H)
L_secondary = L_primary / n²          (H)
n²          = Ra / Rload
K1 L1 L2 0.95                         (coupling coefficient)
```

PP mismatched: two .sub files, two .include, different SYMBOL names for X1/X2.

### Round-trip validation

`tests/test_ltspice_roundtrip.py` exports model → runs LTspice batch →
parses .raw → compares Ia point-by-point with Python model. RMS < 5%.

Code: `spice_export.py`, `ltspice_asc.py`, `ltspice_raw.py`

---

## 11. Sweeps

### THD vs Amplitude (`sweep_amplitude`)

Sweeps half_swing from 0.5V to max headroom. At each step computes
THD, HD2, HD3, Pout, IMD using selected HD method.

### HD vs Ra (`sweep_ra`)

Sweeps Ra from min to max. At each Ra: computes intersections, distortion,
gain, Zout, Pa. Supports transformer mode (AC load line).

### THD vs Bias (`sweep_bias`)

Sweeps Ug1 across available range. Finds optimal bias for given target.

Code: `sweep_amplitude()`, `sweep_ra()`, `sweep_bias()` in `amplifier.py`

---

## UI → Calculation → Result Mapping

How user actions in the interface trigger calculations and where results appear.

### Controls → Parameters

| UI Control | Location | Sets Parameter | Used By |
|-----------|----------|---------------|---------|
| **Ub** spinbox | Parameters section / Plot Options | Supply voltage | Load line, Pdc, Pa, optimizer |
| **Ra** spinbox | Parameters section / Plot Options | Load resistance | Load line, Gain, Zout, sweep_ra |
| **Ug1** spinbox | Parameters section / Plot Options | Bias point | Q-point, distortion, headroom |
| **Swing** spinbox | Parameters section / Plot Options | Signal amplitude | Half_swing for distortion (0 = auto max) |
| **Pa max** spinbox | Parameters section | Thermal limit | Headroom, optimizer constraint |
| **NFB** checkbox + spin | Parameters section | Feedback (dB) | compute_nfb_effect → Gain/Zout/THD/BW |
| **Circuit** combo | Circuit section | se / se_xfmr / cf / pp | Load line type, stage param formulas |
| **Ra_dc** spinbox | Circuit (SE Xfmr) | Winding DCR | TransformerLoadLine DC path |
| **Rk, Rl** spinboxes | Circuit (CF) | Cathode/load R | CathodeFollowerLoadLine, CF gain |
| **Ra_aa** spinbox | Circuit (PP) | Anode-to-anode | PushPullLoadLine, Ra_per_tube = Ra_aa/4 |
| **Matched** button | Circuit (PP) | Same/different tubes | composite_characteristic points_b |
| **Source** combo | Source & Data | Series ID | Which measurement data to analyze |
| **HD method** combo | Source & Data | 5point/chebyshev/dft | _compute_hd dispatcher |
| **Show HD4/5** checkbox | Parameters section | Display HD4/HD5 | UI display only (always computed by Chebyshev/DFT) |
| **Show Gain/Zout/Pa** checkbox | Parameters section | Extra curves on Ra plot | sweep_ra includes stage params |
| **Auto Q** button | Parameters section | — | sweep_bias → min THD → sets Ug1 |
| **Opt Ra** button | Parameters section | — | sweep_ra → min THD → sets Ra |
| **Run** (Optimizer) | Optimizer section | — | optimize_* → Pareto front |
| **Pareto** toggle | Optimizer section | — | Switch left plot to Pareto view |

### Calculation Pipeline

When any parameter changes (`settings_changed` signal):

```
amp_control_panel.params_snapshot()
    ↓
amp_engine.analyze(params)
    ├── _make_load_line(params)     → LoadLine object
    ├── _get_intersections()        → load line × tube curves
    ├── _resolve_hd_method()        → "5point" / "chebyshev" / "dft"
    ├── compute_distortion*()       → HD2, HD3, THD, Pout
    ├── compute_imd()               → IMD2, IMD3
    ├── compute_headroom()          → max_swing, clip limits
    ├── compute_stage_params()      → Gain, Zout, DF, gm, ra, μ
    ├── compute_nfb_effect()        → adjusted Gain/Zout/THD (if NFB enabled)
    ├── compute_pa_avg()            → Pa under signal (if model available)
    ├── sweep_amplitude()           → left plot data (THD vs Swing)
    └── sweep_ra()                  → right plot data (HD vs Ra)
    ↓
amplifier_tab.render(result)
    ├── Left plot:  THD & Pout vs Amplitude (or Pareto)
    ├── Right plot: HD vs Ra (+ Gain/Zout/Pa if enabled)
    └── Results HTML: Q-point, distortion, power, stage params
```

### Where Results Appear

| Result | Where Displayed | Format |
|--------|----------------|--------|
| Q-point (Ua, Ia, Ug1) | Results section, Load Line info bar | `Q: Ua=160V Ia=1.91mA Ug1=-8.0V` |
| HD2, HD3, THD | Results section, left plot (curves) | `HD2=6.3% HD3=0.0% THD=6.3%` |
| HD4, HD5 | Results section (if > 0.1%), left plot (if checkbox) | Shown when Chebyshev/DFT |
| Pout | Results section, left plot (right Y axis) | `Pout=0.023W` (always Watts) |
| IMD2, IMD3 | Results section, Load Line info bar | `IMD2=4.5% IMD3=0.5%` |
| Gain, Zout | Results section | `Gain=11.5 (21.2dB) Zout=11.8kΩ` |
| DF (Damping Factor) | Results section | `DF=4.0` |
| Rk (auto-bias) | Results section | `Rk=4.2kΩ` |
| Pa, Pa_avg | Results section, Ra plot (if checkbox) | `Pa=0.31W Pa_avg=0.28W` |
| Headroom | Results section | `Swing_max=8.0V (cutoff)` |
| NFB effect | Results section (if NFB enabled) | `NFB 6dB → Gain=5.8 Zout=5.9kΩ THD=3.2% DF=8.0 BW×2.0` |
| Chebyshev notice | Results section | `Chebyshev: HD2–HD6 (need more Ug1 curves for higher harmonics)` |
| Method tag | Results section | `[numerical]` or `[model]` or `[SRK ⚠ +25%]` |
| Sweep curves | Left plot (THD vs Swing), Right plot (HD vs Ra) | Interactive: hover for tooltip |
| Pareto front | Left plot (after optimizer) | Click to apply parameters |
| Optimizer status | Optimizer section | `Grid: 400 pts, 180 valid, 12 Pareto. THD=3.2% Pout=0.42W` |

---

## Worked Example: 12AU7 SE Resistive

**Given:** Ub = 250V, Ra = 47kΩ, Ug1 = −8V, Swing = 4V
(Koren model, `quick_triode("12AU7")`)

**Step 1: Q-point**
- Load line: Ia = (250 − Ua) / 47 mA
- At Ug1 = −8V: Ua = 160V, Ia = 1.91mA
- Pa_q = 160 × 1.91 = 306 mW (within Pa_max = 2.75W)

**Step 2: 5-point sampling** (half_swing = 4V)

| Point | Ug1 | Ia | Ua |
|-------|-----|-----|-----|
| i_min | −12V | 1.05 mA | 201V |
| i_low_half | −10V | 1.45 mA | 182V |
| i_0 (center) | −8V | 1.91 mA | 160V |
| i_high_half | −6V | 2.43 mA | 136V |
| i_max | −4V | 3.01 mA | 108V |

**Step 3: Fourier coefficients**
- swing = 3.01 − 1.05 = 1.96 mA
- half_diff = 2.43 − 1.45 = 0.98 mA
- B1 = (1.96 + 0.98) / 3 = 0.98 mA
- B2 = (3.01 + 1.05 − 2 × 1.91) / 4 = 0.06 mA
- B3 = (1.96 − 2 × 0.98) / 6 = 0.00 mA

**Step 4: Distortion**
- HD2 = |0.06 / 0.98| × 100 = 6.3%
- HD3 ≈ 0.0%
- THD = 6.3%

**Step 5: Output power**
- Ia_pp = 1.96 mA, Ua_pp = |201 − 108| = 93V
- Pout = (1.96 × 93) / 8 = 22.7 mW

**Step 6: Stage parameters** (from model at Q-point)
- gm = 0.98 mA/V, ra = 15.7 kΩ, μ = 15.4
- Gain = 15.4 × 47 / (15.7 + 47) = 11.5 (21.2 dB)
- Zout = 15.7 × 47 / (15.7 + 47) = 11.8 kΩ

---

## Physical Sanity Limits

All results are checked against physical limits (constants in `lm19/constants.py`):

| Metric | Sanity limit | Meaning |
|--------|-------------|---------|
| THD | < 50% (`MAX_SANE_THD_PCT`) | Hard clipping ≈ 48%; normal < 20% |
| HDn (individual) | < 100% (`MAX_SANE_HD_PCT`) | Harmonic > fundamental = math error |
| Pout | > 0 | Negative power = computation error |
| Ia_q | > 0 | Tube must conduct at Q-point |
| B1 | > 0.01 mA (`MIN_B1_MA`) | Negligible signal = garbage HD |

---

## Code → Formula Cross-Reference

| Formula | Function | File |
|---------|----------|------|
| B1, B2, B3 | `compute_distortion` | amplifier.py |
| Chebyshev HDn | `compute_distortion_chebyshev` | amplifier.py |
| DFT HDn | `compute_distortion_dft` | amplifier.py |
| IMD2, IMD3 | `compute_imd` | amplifier.py |
| Gain, Zout, DF | `compute_stage_params` | amplifier.py |
| CF Gain, Zout | `compute_stage_params` (CF branch) | amplifier.py |
| gm, ra (model) | `model_gm_ra` | amplifier.py |
| gm, ra (data) | `_numerical_gm_ra` | amplifier.py |
| NFB effects | `compute_nfb_effect` | amplifier.py |
| Headroom | `compute_headroom` | amplifier.py |
| Pa average | `compute_pa_avg` | amplifier.py |
| PP composite | `composite_characteristic` | amplifier.py |
| PP distortion | `pp_distortion` | amplifier.py |
| Optimizer score | `_score` | optimizer.py |
| Load lines | `ResistiveLoadLine` etc. | amplifier.py |
| Transformer L | `generate_amp_schematic` | ltspice_asc.py |

## Sources

Materials this document rests on. The full registry of the project
external sources (all entries, statuses, local copies) — `SOURCES_INDEX.md`.

### Aiken Amps — The Last Word on Class A
- url: <https://www.aikenamps.com/index.php/the-last-word-on-class-a>
- type: theory
- role here: §«Class A power threshold»
- note: Definition used for the optimizer's class-A constraint: class A =
  plate current in every output device flows for the full 360 deg of the
  cycle at full unclipped output. Key claims consumed by this project: PP
  class A gives exactly 2x the power of SE class A under the same plate
  voltage / bias / effective load; the bias METHOD (fixed vs cathode) does
  not determine the class; dissipation is maximal at idle in class A and
  does not rise with signal (unlike AB/B). Basis for `P_A = Iq^2*Ra_aa/8`.
  Index entry added retroactively 2026-08-02 (downloaded 2026-04-26,
  unregistered).

### Rod Elliott (ESP) — Class-A Amplifiers Explained
- url: <https://sound-au.com/class-a.htm>
- type: theory
- role here: §«Class A power threshold»
- note: Second, independent statement of the class-A boundary used to cross-
  check the Aiken definition: the amplifier stays in class A only while the
  signal current stays below the idle current, so the class-A power ceiling
  is fixed by Iq and the load, and an amplifier leaves class A long before
  clipping. Efficiency ceiling (25% SE / 50% PP theoretical) and heat-at-
  idle consequences documented there too. Index entry added retroactively
  2026-08-02 (downloaded 2026-04-26, unregistered).
