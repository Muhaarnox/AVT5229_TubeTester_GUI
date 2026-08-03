# Amplifier Analysis — User Guide

Quick guide to using the **Amplifier** tab for designing tube amplifier stages
based on measured (or loaded) VA characteristics.

---

## Quick Start

1. **Scan a tube** (or load a saved measurement / import CSV).
2. In **Plot Options** (below the plot area), check **Load Line** checkbox.
3. Set **Ub** (supply voltage) and **Ra** (load resistance).
4. Switch to the **Amplifier** tab in the left panel — controls and results appear.
5. Click the **Amplifier** tab in the plot area — sweep plots appear.

> Parameters (Ub, Ra, Ug1, Swing) are bidirectionally synced between
> Plot Options and the Amplifier control panel — change in one place,
> both update automatically.

---

## Interface Overview

The amplifier controls live in the **left panel** (tab "Amplifier"), while the
plots occupy the main area (tab "Amplifier" in the plot tabs).

```
 Left Panel                          Main Plot Area
+---------------------------+  +-----------------------------------+
| [Measure] [Amplifier]     |  |  [2D] [Transfer] ... [Amplifier]  |
|                           |  |                                   |
| Circuit  [Single-Ended v] |  |  THD & Pout        Distortion     |
|          [Auto Q][Opt.Ra] |  |  vs Amplitude       vs Ra         |
|  (SE Xfmr: +Ra_dc)        |  |  (left plot)       (right plot)   |
|  (CF: +Rk +Rl)            |  |  Left plot switches to Pareto     |
|  (PP: +Ra_aa [Matched]    |  |  when Optimizer runs.             |
|       +UL tap [43]%       |  +-----------------------------------+
|       +UL sweep [Presets])|
|                           |  Parameters Ub/Ra/Ug1/Swing are
| Source                    |  bidirectionally synced with
|  Source  [Scan 1 (247) v] |  Plot Options controls.
|  Data    [x]Meas [x]Koren |
|  HD method  [Auto      v] |  The working line of the selected
|  [ ]Show HD4/HD5          |  circuit is drawn live on the 2D
|  [ ]Show Gain/Zout/Pa     |  plot as these controls change.
|                           |
| Parameters                |
|   Ub [250] Ra [5.0]       |
|   Ug1 [-7] Swing [0]      |
|   Pa max [12.5] NFB[ ]6dB |
|                           |
| Optimizer (collapsed)     |
|   Target [Min THD v]      |
|   Pout min [0] THD max [0]|
|   Min class A power [Off] |
|   Ub range / Ug2 range    |
|   [Run][Cancel] [Pareto]  |
|   [Top candidates...]     |
|   [====progress bar====]  |
|                           |
| Results                   |
|  Q: Ua=189V Ia=3.2mA      |
|  THD=1.8% Pout=42mW       |
|  Gain=38.2 Zout=43.5k     |
|  [Verify in LTspice][Auto]|
|  [ ]Sweep  [ ]IMD         |
+---------------------------+
```

**HD method**, **UL tap / UL sweep**, **THD max**, **Min class A power** and
**Verify in LTspice** are covered in their own sections below.

---

## Step-by-step Workflows

### Workflow 1: Find Optimal Bias (Auto Q)

**Goal:** Minimize distortion at a given supply voltage and load.

1. Set **Ub** and **Ra** in Plot Options.
2. Open the **Amplifier** tab.
3. Click **Auto Q**.
4. The system sweeps all available Ug1 values and picks the one with minimum THD.
5. The Ug1 control updates automatically — all plots refresh.

The results panel shows the optimal Q-point: Ua, Ia, Ug1, THD, Pout,
and the auto-bias cathode resistor value Rk.

### Workflow 2: Find Optimal Load Resistance (Optimize Ra)

**Goal:** Find the Ra that gives minimum distortion for the current bias.

1. Set **Ub** and **Ug1** (bias point).
2. Click **Optimize Ra**.
3. The system sweeps Ra from 10% to 1000% of the current value
   and picks the one with minimum THD.
4. The Ra control updates automatically.

Look at the **Distortion vs Ra** plot (right) — the white vertical line
shows your current Ra. The THD curve shows how distortion varies.

### Workflow 3: Evaluate Headroom

**Goal:** How large a signal can the tube handle before clipping?

The results panel always shows:
```
Headroom: ±X.X V (neg: cutoff, pos: grid current)
```

- **neg: cutoff** — the tube reaches Ia ≈ 0 (signal clipped at bottom)
- **pos: grid_current** — Ug1 reaches 0V (grid starts drawing current)
- **pos: data_limit** — ran out of measured data (scan wider Ug1 range)

The headroom determines the maximum undistorted signal swing.

### Workflow 4: Design Power Stage

**Goal:** Maximize output power at acceptable distortion.

1. Set Ub to your supply voltage, Ra to your output transformer impedance.
2. Click **Auto Q** to find optimal bias.
3. Look at the **THD & Pout vs Amplitude** plot (left):
   - Yellow line (THD%) — grows with signal level.
   - Purple dashed line (Pout) — output power.
   - Horizontal gray dashed line — 1% THD reference.
4. Find where THD crosses your acceptable limit (e.g., 3%) and read Pout.
5. Adjust **Swing** in Plot Options to set the desired operating point.
6. If requested swing is outside measured data range, Amplifier now clamps it
   and shows a notice in the results panel.

### Workflow 5: SE Transformer Analysis

**Goal:** Analyse a single-ended stage with a transformer-coupled load (typical for
output stages like EL84, 6L6, 300B).

1. Select **SE Transformer** from the circuit type dropdown.
2. Set **Ra** to the reflected primary impedance (kΩ) — this is the AC load.
3. Set **Ra_dc** to the DC resistance of the primary winding (kΩ, typically 0.02–0.5).
4. Set **Ub** and **Ug1** as usual.

**How it differs from resistive SE:**

- **DC Q-point** sits near Ua ≈ Ub (the winding has low DC resistance), so the
  tube runs at much higher plate voltage than a resistive load would give.
- **AC load line** passes through the Q-point with slope −1/Ra and can extend
  Ua **beyond Ub** — the transformer primary stores energy and swings the plate
  voltage higher than the supply.
- **Result:** significantly higher output power and different distortion profile
  compared to a resistive load of the same value.

> **Pentode output stages almost always use transformer coupling.**
> Selecting "SE Transformer" instead of "Single-Ended" gives a physically correct
> analysis. The "Single-Ended" (resistive) mode underestimates Pout for pentodes.

**Ra sweep:** The Distortion vs Ra plot sweeps Ra as reflected transformer impedance,
using the AC load line at each step.

### Workflow 6: Cathode Follower Analysis

**Goal:** Evaluate a cathode follower stage (gain ≈ 1, low output impedance).

> Numbering shifted: former Workflows 5–7 are now 6–8.

1. Select **Cathode Follower** from the circuit type dropdown.
2. Set **Rk** (cathode resistor) and **Rl** (load) — these appear automatically.
3. Set **Ub** and **Ug1** in Plot Options.
4. The analysis shows:
   - **Gain** — should be slightly less than 1.0 (e.g., 0.85–0.95).
   - **Zout** — very low, approximately 1/gm (typically 0.3–1 kΩ).
   - **THD** — typically lower than SE due to strong negative feedback.

> **Note:** The Ra sweep plot is disabled for CF since the cathode load
> is set by Rk/Rl. Use the THD vs Amplitude plot to evaluate linearity.

### Workflow 7: Push-Pull Analysis

**Goal:** Evaluate a push-pull output stage with even harmonic cancellation.

1. Select **Push-Pull** from the circuit type dropdown.
2. Set **Ra_aa** (anode-to-anode impedance) — appears automatically.
3. **"Matched pair"** button (default: ON) — uses the same tube data mirrored.
4. To use real second tube data:
   - Load a second measurement via the Compare tab → "Show on main plot".
   - Uncheck **"Matched pair"**.
   - Select the second dataset in **Tube B** (shown on the Source row).

**What to look for:**
- **HD2** should be very low (< 1%) for a matched pair.
- **Balance error** shows how well the pair is matched (0% = perfect).
- **HD3** dominates in PP — this is the main distortion component.
- **Pout** is calculated as `(Ia_pp × Ua_pp) / 8` (standard sinusoidal power formula).

### Workflow 8: Use with Saved Measurements

The Amplifier tab works with any data source:

1. **Current scan** — active measurement from the tester.
2. **Overlays** — measurements loaded via Compare tab → "Show on main plot".
3. **Imports** — CSV or uTracer data imported via File → Import.

Use the **Source** dropdown to pick which dataset to analyze.
The control is always visible and shows point count for each source.

### Workflow 9: Multi-Parameter Optimizer (Pareto)

**Goal:** Find optimal combination of Ub, Ug2, Ug1, Ra, and Swing simultaneously.

1. Open the **Optimizer** section in the Amplifier panel (click to expand).
2. Set **Target**: Min THD, Max Pout, or Balanced.
3. Optionally add constraints: **Pout min** (W), **THD max** (%) and — for
   push-pull — **Min class A power**. See *Optimizer constraints* below.
4. For model mode, set **Ub range** and **Ug2 range** if needed.
   For push-pull, **UL sweep** decides whether the screen tap is searched too.
5. Click **Run**. Progress bar shows three phases:
   - Grid sweep (Ug1 × Ra, optionally Ub × Ug2)
   - Swing sweep (top-N points re-evaluated at multiple amplitudes)
   - Pareto refinement (scipy Nelder-Mead in parallel on front points)
6. The left plot switches to **Pareto front** (THD vs Pout trade-off):
   - Gray dots = all valid grid points
   - Green dashed line = grid Pareto front
   - Cyan solid line = refined Pareto front
   - Yellow star = best grid point
   - Cyan star = best refined point
7. **Hover** over Pareto points to see THD, Pout, Ub, Ra, Ug1, Swing, Class, Pa.
8. **Click** on any Pareto point — all parameters apply automatically
   (Ub, Ra, Ug1, Swing, Ug2, and the UL tap for push-pull).
9. Toggle **Pareto** button to switch between Pareto and THD vs Amplitude plots.
10. **Top candidates...** opens the ranked list if you would rather apply a
    runner-up than the single best point.
11. Click **Cancel** to abort a running optimization.

> The optimizer runs in a background thread — the UI stays responsive.
> Model mode provides continuous Ub/Ug2 sweep; measurement mode uses
> discrete Ug2 values from data.

### Workflow 10: Export to LTspice

**Goal:** Generate a ready-to-simulate LTspice schematic with the fitted tube model.

1. Design your amplifier using the Amplifier tab (set Ub, Ra, Ug1, circuit type).
2. Go to **File → Export SPICE Model**.
3. Select model type (Koren / Dempwolf / Reefman).
4. Check **"Generate amplifier circuit"** and select circuit type:
   - **SE Resistive** — simple RC-coupled stage
   - **SE Transformer** — transformer-coupled output (set Rload and f_low)
   - **Cathode Follower** — unity-gain buffer
   - **Push-Pull** — select Tube A and Tube B sources (matched or mismatched)
5. Choose filename (e.g. `12AU7_V1.sub`) — the name becomes the subcircuit name.
6. Click OK → files are generated:
   - `.sub` — SPICE subcircuit
   - `.asy` — LTspice symbol
   - `*_amp.asc` — amplifier schematic (or `*_se_xfmr.asc`, `*_cf.asc`, `*_pp.asc`)
7. Open the `.asc` file in LTspice — circuit is ready with your tube and components.
8. Run DC sweep (default), or uncomment `.tran` / `.ac` for transient/frequency analysis.

> **Transformer parameters:** Rload = speaker impedance (4/8/16Ω).
> f_low = low-frequency -3dB point (20Hz for hi-fi, 50Hz for guitar).
> Primary inductance is computed automatically: L = Ra / (2π × f_low).

> **PP mismatched pair:** Export Tube A and Tube B from different measurement
> series. The schematic will reference two different `.sub` models.

> To check the analysis itself rather than export it, use **Verify in LTspice**
> in the Results section — it simulates the circuit as the engine idealizes it
> and puts engine vs LTspice side by side. See *Tips*.

---

## What Each Parameter Means

| Parameter | Where | Description |
|-----------|-------|-------------|
| **Ub** | Plot Options / Amplifier panel | Supply voltage (V). X-intercept of the load line. Synced bidirectionally. |
| **Ra** | Plot Options / Amplifier panel | Anode load resistance (kΩ). For SE resistive: slope of load line. For SE Transformer: reflected impedance (Ra_ac). Synced bidirectionally. |
| **Ug1** | Plot Options / Amplifier panel | Quiescent grid bias (V, negative). The operating point. Synced bidirectionally. |
| **Swing** | Plot Options / Amplifier panel | Half-swing amplitude (V). 0 = auto (maximum symmetric). Manual values are clamped to available measured range. Synced bidirectionally. |
| **Pa max** | Amplifier panel | Maximum anode dissipation (W). For thermal limit check. |
| **NFB** | Amplifier panel (Parameters row) | Negative feedback (dB). Enable checkbox + value. Recalculates Gain, Zout, THD, DF, BW. |
| **Rk** | Amplifier panel (CF) | Cathode resistor (kΩ). Sets DC bias and feedback. |
| **Rl** | Amplifier panel (CF) | AC load resistance (kΩ). External load on the cathode. |
| **Ra_dc** | Amplifier panel (SE Xfmr) | DC resistance of transformer primary winding (kΩ). Typical: 0.02–0.5 kΩ. |
| **Ra_aa** | Amplifier panel (PP) | Anode-to-anode impedance (kΩ). In class A each tube sees Ra_aa/2 (the partner's antiphase current doubles the swing); once the partner cuts off, the surviving tube sees Ra_aa/4. That impedance halving is the class-A→AB kink visible on the PP working line. |
| **Matched pair** | Amplifier panel (PP) | Assume identical tubes (mirror curves). |
| **HD method** | Amplifier panel (Source) | How harmonics are computed — see *Choosing the HD method*. |
| **UL tap** | Amplifier panel (PP) | Ultralinear screen tap, % — see *Ultralinear mode*. |
| **THD max** | Amplifier panel (Optimizer) | Distortion ceiling for the optimizer, % (0 = off). |
| **Min class A power** | Amplifier panel (Optimizer, PP) | Lower bound on the class-A region, W or % of Pout. |

---

## Choosing the HD method

The **HD method** combo (Source section) selects how harmonic distortion is
computed. It applies to the results panel, both sweeps and the optimizer.

| Method | Harmonics | Best for | Notes |
|---|---|---|---|
| **Auto** | — | Default | Chebyshev for measurements, DFT when a fitted model is the source. |
| **Chebyshev** | HD2..HD{2n/3} | Measured curves | Polynomial decomposition of the transfer curve. ~5× slower than 5-point. |
| **DFT** | HD2..HD9 | Fitted models | FFT of a synthesised waveform; needs a model. Without one it falls back to 5-point and says so. |
| **5-point** | HD2, HD3 only | Quick looks | Classic Radiotron method. Fast, but blind to HD4+. |

Practical rules:

- **5-point needs at least 3 measured Ug1 curves inside the swing window.**
  On a narrow swing it would otherwise report an impossibly clean number
  (a real 1.85 % case reads as 0.07 %); the engine rejects such windows
  instead, and the panel shows a sparse-data notice.
- **UL tap only affects Chebyshev/DFT with a model.** With 5-point on measured
  data the tap cannot influence the composite curve, so the panel reports
  `ul_tap_ignored_by_method` rather than silently ignoring it.
- Cross-checking is cheap: different methods should agree within ~×3 on THD.
  A larger spread means the swing window or the data is marginal.

---

## Ultralinear mode (PP pentode)

**UL tap %** taps the output-transformer primary for the screen grid, so the
screen follows the anode: `Ug2_eff = Ug2_nom × (1 − tap) + Ua × tap`.
0 % is pure pentode, 100 % is triode-strapped; classic values sit in between.

Typical result on an EL84 pair (bias −11 V, ±9 V drive) — power and THD both
fall as the tap rises, which is the whole trade (fundamental power P1):

| Mode | P1 | THD |
|---|---:|---:|
| Pentode (0 %) | 17.6 W | 9.7 % |
| UL 43 % (Williamson) | 7.8 W | 2.1 % |
| Triode-strapped (100 %) | 3.2 W | 0.55 % |

**UL sweep** (below UL tap) lets the optimizer search taps instead of fixing one:

- **Off** — use the single **UL tap** value.
- **Presets** — historical taps (0 % pentode, 20 % Acrosound, 35 % 6L6,
  43 % Williamson, 50 % Quad, 100 % triode); enable/disable each with its
  checkbox.
- **Custom range** — a min..max range plus **Steps**.
- **Presets + range** — both, de-duplicated.

The tap of the winning point is applied along with everything else when you
click a Pareto point, including tap = 0.

---

## Optimizer constraints

Beyond **Target** and **Pout min**:

- **THD max** (%, 0 = off) — the datasheet-style question "how much power at
  X % distortion?". Points exactly at the cap pass. Candidates that fail only
  the cap are not thrown away: distortion falls with swing, so the swing-sweep
  phase gets a chance to bring them under the limit. If nothing in the whole
  grid fits, you get a specific "no points within THD cap" message advising you
  to raise the limit, not the generic no-solution error.
- **Min class A power** (PP only) — `P_A = Iq² × Ra_aa / 8`, the power at which
  the off-going tube reaches cutoff. Modes: **Off**, **Absolute** (watts),
  **% of Pout**. Use it to keep a design honestly class A up to some level.
- **Top candidates...** — opens the ranked table of the best points, so you can
  apply a runner-up instead of the single best.

---

## Understanding the Plots

### Left Plot: THD & Pout vs Amplitude

X-axis: signal half-swing (V) — how far Ug1 swings from the bias point.

- **HD2%** (red) — second harmonic distortion. Dominant in single-ended stages.
- **HD3%** (teal) — third harmonic distortion.
- **THD%** (yellow, thick) — total harmonic distortion (RSS of HD2 + HD3).
- **Pout** (purple, dashed, right Y-axis) — output power in mW.

White dotted vertical line = your current Swing setting (if non-zero).
Gray horizontal dashed line = 1% THD reference level.

**How to read:** Follow the yellow THD curve — as amplitude increases,
distortion grows. Find the power level at your acceptable THD limit.

### Right Plot: Distortion vs Ra

X-axis: load resistance Ra (kΩ).

Same color coding as the left plot. Shows how distortion and power change
when you vary the load resistance at fixed bias and amplitude.

White dotted vertical line = your current Ra setting.

**How to read:** The THD minimum shows the optimal Ra. Higher Ra → more gain
but possibly more distortion. Lower Ra → more power but earlier clipping.

---

## Understanding the Results Panel

```
Q-point: Ua=189V  Ia=3.2mA  Ug1=-2.0V
HD2=1.8%  HD3=0.3%  THD=1.8%  Pout=42mW
IMD2=2.1%  IMD3=0.4%
Headroom: ±1.8V (neg: cutoff, pos: grid current)
Gain=38.2 (31.6 dB)  Zout=43.5 kΩ  [srk]
Rk=625Ω (auto-bias)  Pa=0.6W / 12.5W max
```

| Line | Meaning |
|------|---------|
| **Q-point** | DC operating point: anode voltage, current, grid bias |
| **HD2/HD3/THD** | Harmonic distortion at the current swing |
| **Pout** | Output power in milliwatts |
| **IMD** | Intermodulation distortion — polynomial nonlinearity coefficients (a2/a1, a3/a1). When Swing is set, only points within the swing window are used for the fit. |
| **Swing limited** | Requested Swing exceeded measured range, analysis used the clamped value |
| **Insufficient signal** | Near-cutoff / too small signal span, THD line may be suppressed |
| **Headroom** | Max symmetric signal swing before clipping |
| **Gain** | Voltage amplification factor (absolute and dB) |
| **Zout** | Output impedance of the stage (kΩ) |
| **[srk]** or **[numerical]** | How Gain/Zout was computed |
| **Rk** | Cathode resistor for auto-bias (Ω). Formula: Rk = \|Ug1\| / Ik × 1000. For pentodes Ik = Ia + Ig2 (screen current included). |
| **Pa** | Anode dissipation at Q-point vs maximum rated |

### Color Coding

- **Green** — within safe / good range
- **Yellow** — approaching limits
- **Red** — exceeding limits or high distortion

---

## Tips

- **Pentodes:** If the tube has multiple Ug2 values in the scan,
  select the desired Ug2 in the main Plot Options → Ug2 slice dropdown.
  The Amplifier tab respects this filter.

- **Gain accuracy:** For best Gain/Zout values, run an **SRK measurement**
  (S, R, K button on the Measure tab). The `[srk]` method is more accurate
  than `[numerical]` which estimates from curve slopes.

- **Wide scan:** For meaningful headroom analysis, scan a wide Ug1 range
  (from cutoff to near 0V). Narrow scans limit the available headroom data.

- **Copy results:** Click the 📋 button to copy the results text
  to the clipboard for pasting into reports or notes.

- **Export PDF:** The `Export PDF` button in the Results section writes
  the analysis as an A4 report: the results text as shown, the actual
  analysis parameters (circuit, Ub, Ra, bias, HD method, data series),
  a harmonic-spectrum bar chart and the THD/Ra/Pareto plots at print
  resolution. A section dialog picks what goes in; the document
  language follows `config/app.json:report_language`.

- **Verify in LTspice:** The `Verify in LTspice` button simulates the
  analyzed circuit in LTspice batch mode (1 kHz sine at the analysis
  drive; UL tap honored) and shows an "Engine vs LTspice" table with
  the basis of each column labelled. The fitter combo next to the button
  picks the simulated model: `Auto (as analyzed)` uses the analysis
  source (its loaded model, or a Koren fit for measurements); an
  explicit Koren/Dempwolf/Reefman choice always runs a fresh fit of that
  type — for pentodes Dempwolf or Reefman usually fit better than Koren.
  Optional check-boxes add a 25/50/75/100% drive sweep and a two-tone
  SMPTE IMD run. Requires
  LTspice (path override: `config/app.json:ltspice_exe`); working files
  stay in `ltspice_verify_dir` (default: system temp) for manual
  re-runs — the netlists are relocatable. The result becomes the
  `LTspice verification` section of the PDF report once it has run.
  If you change the analysis parameters afterwards, the table gets a
  visible ⚠ "verified with different parameters" mark (panel and PDF) —
  re-run the verification to refresh it.

- **Triode-connected pentodes:** Set Ug2 mode to "Track Ua" on the Measure tab.
  The Amplifier tab handles this topology correctly.

- **Pentode output stages:** Always use **SE Transformer** (not plain Single-Ended)
  for pentode output stages. With a resistive load line, the Q-point is forced into
  the knee region and Pout is severely underestimated. The transformer mode places
  the Q-point near Ua = Ub (correct for real amplifiers) and allows signal swing
  beyond the supply voltage.

- **Rk for pentodes:** The auto-bias resistor Rk accounts for screen current (Ig2)
  in pentode mode: Rk = |Ug1| / (Ia + Ig2). This is lower than Rk computed from
  plate current alone.

---

## Troubleshooting

| Message | Cause | Fix |
|---------|-------|-----|
| "No data" | No measurement points | Run a scan or load a measurement |
| "Enable Load Line" | Load Line checkbox is off | Check the Load Line checkbox in Plot Options |
| "Not enough intersections (need ≥ 3)" | Load line misses the curves | Adjust Ub/Ra so the line crosses at least 3 Ug1 curves, or scan wider range |
| "Swing limited by available data" | Requested manual Swing is outside measured Ug1 range | Use wider scan range or accept clamped swing |
| "Insufficient signal for reliable THD" | Operating point is near cutoff or span is too small | Move bias from cutoff, increase data range/swing headroom |
| "Gain: N/A" | No SRK data and insufficient raw points for numerical estimation | Run SRK measurement |
| Plots empty but results show data | Very narrow sweep range | Increase Ug1 scan range or check Ra value |
