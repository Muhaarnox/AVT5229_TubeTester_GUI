# LM19 Tube Tester — Features Guide

This document describes all graphs, analysis tools, and export features available
in the LM19 tube tester application.

---

## Table of Contents

- [Graphs](#graphs)
  - [2D — Ia(Ua)](#2d--iaua)
  - [Transfer — Ia(Ug1)](#transfer--iaug1)
  - [Contour — Ia(Ua, Ug1)](#contour--iaua-ug1)
  - [Gm/Rp — Heatmaps](#gmrp--heatmaps)
  - [Pa Map — Power Dissipation](#pa-map--power-dissipation)
  - [Curves — Universal Parameter Plot](#curves--universal-parameter-plot)
  - [Real-time Ia/Ig2](#real-time-iaig2)
  - [Compare Plot](#compare-plot)
- [Overlays & Limit Zones](#overlays--limit-zones)
  - [Pa Max Hyperbola](#pa-max-hyperbola)
  - [Ua Max Zone](#ua-max-zone)
  - [Ia Max Zone](#ia-max-zone)
  - [Load Line](#load-line)
- [Analysis Tools](#analysis-tools)
  - [S/R/K Measurement](#srk-measurement)
  - [5-Point Distortion (HD2/HD3)](#5-point-distortion-hd2hd3)
  - [Ra Sweep Analysis](#ra-sweep-analysis)
  - [IMD (Intermodulation Distortion)](#imd-intermodulation-distortion)
  - [Tube Quality Score](#tube-quality-score)
  - [Amplifier Analysis Tab](#amplifier-analysis-tab)
  - [Tube Simulator (tube_sim.py)](#tube-simulator-tube_simpy)
- [Comparison & Matching](#comparison--matching)
  - [Matching Delta Analysis](#matching-delta-analysis)
  - [Aging Trend](#aging-trend)
- [Export](#export)
  - [PDF Report](#pdf-report)
  - [SPICE Model Export](#spice-model-export)
  - [CSV Export](#csv-export)
  - [uTracer Export](#utracer-export)
  - [JSON Measurements](#json-measurements)
  - [Compare Tab — Multi-Export](#compare-tab--multi-export)
- [Import](#import)
  - [CSV Import](#csv-import)
  - [uTracer Import](#utracer-import)
  - [CurveTraceData Import](#curvetracedata-import)
  - [eTracer Import](#etracer-import)
- [Manual Tab](#manual-tab)
- [Tube Health](#tube-health)
  - [Quick Health Test](#quick-health-test)
  - [SRK Measurement (Health)](#srk-measurement-health)
  - [Emission Test](#emission-test)
  - [Reference Sources](#reference-sources)
  - [Plan Settings](#plan-settings)
  - [Steps / Measurement Points Table](#steps--measurement-points-table)
  - [Health History](#health-history)
  - [Tube Matching](#tube-matching)
  - [Data Storage](#data-storage)
- [Calibration](#calibration)
  - [Calibration Tab](#calibration-tab)
  - [Calibration Wizard](#calibration-wizard)
  - [Manual Edit](#manual-edit)
  - [Live Readings](#live-readings)
  - [Calibration Data](#calibration-data)
- [Scan Settings](#scan-settings)
- [Scan Stabilization](#scan-stabilization)
  - [Dynamic Settle](#dynamic-settle)
  - [Verification with Retry](#verification-with-retry)
  - [Per-Parameter Configuration](#per-parameter-configuration)
  - [Ia/Ig2 Averaging (Samples)](#iaig2-averaging-samples)
  - [Pg2 (Screen Grid Power) Protection](#pg2-screen-grid-power-protection)
  - [Safe Down-Sweep (Pentode Mode)](#safe-down-sweep-pentode-mode)
  - [Adaptive Refine (Two-Pass Scan)](#adaptive-refine-two-pass-scan)
  - [Communication Error Recovery](#communication-error-recovery)
  - [Heater Alive Check](#heater-alive-check)

---

## Graphs

### 2D — Ia(Ua)

**Tab:** `2D` on the main Measure screen.

Output characteristics of the tube. Shows anode current (Ia, mA) vs anode
voltage (Ua, V). Each curve represents a fixed grid voltage (Ug1). Color
encodes the screen grid voltage (Ug2) using the viridis colormap.

**Ug2 display modes:**
- **Ug2 as color** — continuous colormap by Ug2 value (default).
- **Ug2 families** — each Ug2 value as a discrete color with legend.

**Ug2 display filter:** A checkable combo box (`ug2_display_combo`) selects
which Ug2 values are visible on the 2D, Transfer, and Curves plots. Toggling
checkboxes is instant — all curves are pre-rendered and their visibility is
switched without recomputation or re-drawing. The filter has no effect on
Contour, Gm/Rp, and Pa map plots (these use the separate `ug2_calc_combo`
for single-slice selection).

**Overlays:** Pa max hyperbola, Ua/Ia max zones, load line (see below).

### Transfer — Ia(Ug1)

**Tab:** `Transfer` on the main Measure screen.

Transfer characteristics. Shows anode current (Ia, mA) vs grid voltage
(Ug1, V) for each anode voltage. Each curve is a fixed Ua.

The slope of each curve is the transconductance S = dIa/dUg1 (mA/V).

**Rendering:** Transfer uses the same unified `render_curves` pipeline as
Curves, via `target_plot=transfer_plot`, `x_param="Ug1"`, `y_param="Ia"`.
The old `_ensure_transfer_cache` and `render_transfer` are removed. Data
is grouped by `series_id` (like Curves), not by `(lamp_type, lamp_id)` —
fixing overlay artifacts when the same lamp appears in different modes
(e.g. triode-connected vs pentode sweep).

**Load-line intersections:** When the load line is active, `render_curves`
draws the operating-point intersections conditionally when X=Ug1, Y=Ia.
They appear on both Transfer and Curves plots in that axis configuration.

**Color coding:**
- Single measurement — color by Ug2 (viridis).
- Multiple measurements (overlay from Compare) — color by source.

**Ug2 display filter:** Shares the same Ug2 display toggle as the 2D plot.
Transfer items are stored in `_ug2_transfer_items`; toggling Ug2
checkboxes instantly shows/hides the corresponding curves without
recomputation.

**View presets & Ua filter:** The `View` combo selects which Ua slices
are shown: All / Datasheet (a few spread slices) / Load line (slices
near Ub) / Custom (checkable Ua combo). With the working line active,
accents are applied: the slice nearest Ub is labeled `≈Ub`, curves
outside the signal swing are dimmed, and the dynamic curve gets a halo.

**PP composite:** For push-pull analysis a folded composite transfer
curve Ia_comp(Vg) = Ia_A(Vg) − Ia_B(2·bias−Vg) is drawn (solid direct
branch + dashed mirrored branch) — the gap between branches visualizes
even-harmonic cancellation.

**Use cases:**
- Visual evaluation of transconductance (S).
- Comparing linearity at different operating points.
- Tube matching assessment via overlay.

### Contour — Ia(Ua, Ug1)

**Tab:** `Contour` on the main Measure screen.

Color intensity map of Ia across the Ua/Ug1 plane for a selected Ug2 slice.
Bright areas = high current, dark = low current.
When load line is active, Q-point and swing endpoints are shown as markers.

**Controls:**
- `Ug2 slice` — select which Ug2 value to display. "Auto" picks the most
  common one.

### Gm/Rp/µ — Heatmaps

**Tab:** `Gm/Rp/µ` on the main Measure screen.

Left plot: **Gm** — transconductance map: Gm = dIa/dUg1 (mA/V).
Right plot: **Rp** or **µ** — switchable via combo above the plot.

- **Rp** — plate resistance: Rp = dUa/dIa (kΩ).
- **µ** — amplification factor: µ = Gm × Rp (dimensionless).

Color intensity indicates the magnitude. Axes: Ua (V) vs Ug1 (V).
Tooltip shows the primary value + Ia at cursor position.
µ tooltip additionally shows Gm and Rp components.
When load line is active, Q-point and swing path are overlaid.

**Colormap selector** — ComboBox on row 1 to switch heatmap color scheme
for all heatmaps: viridis (default), plasma, inferno, magma, cividis.
The colormap change also re-luts every heatmap colorbar and the 2D Ug2
colorbar.

**Colorbars & Lock scale** — every heatmap (Contour, Gm, Rp, µ, Pa)
shows a value colorbar with its unit label; bars follow autoLevels on
each render and hide when the map has no data. The **Lock scale**
checkbox (axes row) freezes the current levels of all maps, making two
scans directly color-comparable; unchecking re-renders with autoLevels.
Note: Rp/µ bars reflect the display-clipped values (p95×1.5 / p97×1.3
contrast clip) — hover tooltips report the unclipped computed values.

### Pa Map — Power Dissipation

**Tab:** `Pa map` on the main Measure screen.

Heatmap of anode power dissipation Pa = Ua × Ia (W). Bright areas indicate
high dissipation. If a Pa_max limit is set, a contour line marks the safe
operating boundary. Tooltip shows Pa + Ia at cursor.
When load line is active, Q-point and swing path are overlaid.

**Use cases:**
- Visualizing the safe operating area.
- Identifying hot spots in the operating space.
- Checking Pa at Q-point vs Pa_max limit.

### Curves — Universal Parameter Plot

**Tab:** `Curves` on the main Measure screen.

A single large plot with selectable Y-axis parameter and X-axis variable.
Replaces the former dedicated Ig2 tab and extends functionality with
derived parameters and flexible axis combinations.

**Y-axis options:** Gm, Rp, μ (mu), Ia, Ig2, Pa, Pig2, Ia/Ig2 ratio.
**X-axis options:** Ua, Ug1, Ug2 (Ug2 only for raw parameters).

**Load-line intersections:** When X=Ug1 and Y=Ia, load-line operating-point
intersections are drawn on the Curves plot (same logic as Transfer).

**Multi-Ug2 support:** Shares the Ug2 display toggle with 2D and Transfer.
All Ug2 groups are pre-rendered with distinct color palettes; toggling Ug2
checkboxes instantly shows/hides individual groups without recomputation.
When the Y/X axis selection changes, Curves are fully re-rendered and the
Ug2 visibility filter is re-applied.

**Use cases:**
- Viewing Ig2(Ua), Ig2(Ug2), and all other screen grid characteristics.
- Analyzing transconductance (Gm), plate resistance (Rp), and μ across
  the operating space.
- Checking plate dissipation (Pa) and screen dissipation (Pig2) limits.
- Evaluating Ia/Ig2 ratio for pentode safety analysis.

### Real-time Ia/Ig2

**Panel:** Available on the Manual tab.

Live chart of Ia and Ig2 as a function of time (seconds). Updated
continuously while the device is connected. Configurable line colors
and widths.

### Compare Plot

**Tab:** `Compare` tab.

Overlay of multiple saved measurements on a single Ia(Ua) plot. Each
measurement is a separate color (customizable by double-clicking the Color
column). Hovering over curves shows a tooltip with Ua/Ia values.

**Legend:** color and line style are independent axes with separate
entries — one solid entry per lamp in its color (duplicate measurement
names get " (2)" suffixes), plus neutral-gray entries per Ug2 level
showing its line style (only for levels enabled in the Ug2 filter
panel, and only when there is more than one level). Measurements whose
Ug2 setpoints differ within the cluster threshold (default 2 V) are
drawn, styled, and filtered as one Ug2 level.

**Controls:**
- `Show on main plot` — sends selected measurements to the main Measure tab
  as overlay curves.
- `Export ▾` — dropdown menu with CSV, uTracer, SPICE, and PDF export for
  checked measurements (see [Compare Tab — Multi-Export](#compare-tab--multi-export)).
- `Line width` spinner — adjusts curve thickness.

---

## Overlays & Limit Zones

These overlays are available on the 2D plot via checkboxes in the Plot
options panel.

### Pa Max Hyperbola

Draws the hyperbola Pa = Ua × Ia = const on the 2D plot. The red shaded
area above the curve is the danger zone. The value (W) is auto-filled from
the lamp configuration.

### Ua Max Zone

Draws a vertical red line at the maximum anode voltage. The area to the right
is shaded red. Value auto-filled from lamp configuration.

### Ia Max Zone

Draws a horizontal red line at the maximum anode current. The area above is
shaded red. Value auto-filled from lamp configuration.

### Load Line

Draws the DC load line from (Ua=Ub, Ia=0) to (Ua=0, Ia=Ub/Ra). Intersections
with the Ia(Ua) curves show the operating point for each Ug1.

**Parameters:**
- **Ub** — supply voltage (V). Auto-filled from nominal Ua.
- **Ra** — load resistance (kΩ). Auto-filled from lamp config.
- **Ug1₀** — quiescent grid bias (V) for distortion analysis.
- **Swing** — half-swing amplitude (V). 0 = auto (max symmetric).

When active, the info bar below the plot shows the Q-point parameters
(Ua, Ia, Ug1), HD2/HD3 distortion percentages, and output power.

**Q-point on all plots:** When load line is active, the Q-point marker (red dot)
and swing endpoints (orange triangles) are shown on all heatmaps (Contour,
Gm, Rp, µ, Pa map). A connecting line shows the signal path through the
operating space. On the Transfer plot, the swing range is shown as a shaded
region between Ug1 min/max with a dashed Q-point line.

### Clear Dialog

The **Clear** button opens a dialog listing all data series on the plot
(current scan + overlays from Compare). Each series shows its name and
point count. The user can check individual series for removal or click
**Remove All** to clear everything. Unchecked series remain on the plot.
Series removal is instant (direct item removal, no full re-render).

### Series Visibility Toggle

The **Lamp display** combo (checkable) controls which series are visible on
the 2D plot. Toggling a checkbox instantly shows/hides the series via
`setVisible()` — no re-render needed. The snap-to-curve marker automatically
filters to only visible series.

---

## Analysis Tools

### S/R/K Measurement

**Panel:** `S/R/K zone` on the main Measure screen.

Measures the tube's key small-signal parameters at the operating point:

| Parameter | Meaning | Unit |
|-----------|---------|------|
| **S** | Transconductance (Steepness) dIa/dUg1 | mA/V |
| **R** | Internal plate resistance dUa/dIa | kΩ |
| **K** | Amplification factor (µ = S × R) | — |

**Three methods:**
1. **From scan data** — computed by multivariate linear regression
   (`Ia = a + S·Ug1 + (1/R)·Ua`) in the S/R/K zone (default).
   Uses `compute_sr_zone()` which correctly separates Ug1 and Ua effects
   even on 2-D scan grids.  When the zone is too narrow to contain
   multiple Ug1 (or Ua) levels, auto-expansion includes neighboring
   curves — but only when the zone lies strictly between existing data
   levels (interpolation, not extrapolation).  The `~` indicator in the
   status line signals that zone expansion was applied.
2. **Measured separately (classic)** — measures at 4 corner points after scan
   (checkbox "Measure" in the zone panel).
3. **Measured with sweep (mini-scan)** — sweeps Ug1 from min to max with a
   configurable step, measuring at 2 Ua values for each Ug1 (checkboxes
   "Measure" + "Sweep" in the zone panel).

**Measurement implementation** (methods 2 & 3): `measure_srk()` uses the same
`_set_param_with_settle` and `_read_measurement_point` helpers as `run_scan`,
ensuring consistent settle/verify/averaging behaviour across all measurement
modes.  Each parameter (Ua, Ug1, Ug2) is set with dynamic settle time and
automatic retry, and the result is verified against a strict tolerance —
`SrkVerifyError` is raised on failure. After settle, a full 7-field point
(ua, ug1, ug2, ia, ig2, uh, ih) is read via `_read_measurement_point` with
N-sample Ia/Ig2 averaging.  In track mode, Ua is raised before Ug2 to avoid
transient Pg2 spikes.

When `ug1_step > 0` (sweep mode), Ug1 is stepped through the full zone
range instead of using only the two corner values. This produces many more
data points for the linear regression, significantly improving precision.

**Controls:**
- Zone boundaries: Ua min/max, Ug1 min/max, Ug2.
- **N** — number of measurement repeats for averaging.
- **Measure** checkbox — measure SRK separately (not from scan data).
- **Sweep** checkbox — enable Ug1 mini-scan (step from `app.json`).
- **...** button — shows detailed results dialog.
- **Save** button — saves current SRK + manual points as a measurement.

#### Sweep Mode Precision Analysis

The sweep mode (`srk_ug1_step > 0`) improves S/R/K measurement precision
by replacing the classic 4-point measurement with a multi-point linear
regression across a dense Ug1 grid.

**Hardware foundation (ATmega16):**

| Component | Resolution | Notes |
|-----------|-----------|-------|
| Ug1 ADC (ch7) | 10-bit, ~0.036V/LSB | 64-sample hardware averaging |
| Ug1 charge pump | Comparator feedback | ug1set = round(val × 1024 / Vref) |
| Ug1 protocol | 0.01V (hundredths) | Raw value 0–2400 = 0.00–24.00V |
| Ia ADC (ch4) | 10-bit, 2 ranges | 64-sample hardware averaging |
| Ia protocol | 0.01 mA (IA_HW_SCALE) | Software N-sample averaging + calibration |
| Ua PWM | ~0.97V/step (Timer1) | 10-bit duty cycle |

**Step size vs. reliability:**

| Ug1 step | ADC LSB diff | Zone 0.48V → points | Reliability |
|----------|-------------|---------------------|-------------|
| 0.04V | ≥1 LSB (min 1, avg 1.2) | 13 Ug1 × 2 Ua = 26 | Good (min reliable) |
| 0.06V | ≥1 LSB (min 1, avg 1.75) | 9 Ug1 × 2 Ua = 18 | Good |
| 0.08V | ≥2 LSB always | 7 Ug1 × 2 Ua = 14 | Ironclad |
| 0.4V (classic) | ~11 LSB | 2 Ug1 × 2 Ua = 4 | N/A |

**Precision comparison (S = dIa/dUg1):**

| Method | Points | Relative error σ(S)/S | Notes |
|--------|--------|-----------------------|-------|
| Classic 4-point | 4 | ~5–10% | Single pair of Ug1, error dominated by ADC noise |
| Sweep 0.08V step | 14 | ~1.5–2% | Regression averages noise across 7 Ug1 |
| Sweep 0.04V step | 26 | < 1% | Most points, best regression fit |

**Why 0.04V is the minimum reliable step:**

The ATmega16 charge pump for Ug1 uses a comparator-based feedback loop:
the firmware computes `ug1set = round(val × 1024 / Vref)` and drives the
charge pump until the ADC reading matches `ug1set`.  With Vref ≈ 3.6V and
a 10-bit ADC, one LSB ≈ 0.036V.  Two Ug1 values that produce the same
`ug1set` are physically indistinguishable — the hardware will settle to
the same voltage.  At 0.04V step, each step changes `ug1set` by at least
1 (verified across the full 0–24V range).  At < 0.04V, some steps may
produce identical `ug1set` values and thus identical physical voltages.

**Why sweep helps precision:**

S = dIa/dUg1 is computed by linear regression.  With 4 points, one ADC
noise spike shifts the slope significantly.  With 26 points, regression
averages noise across many measurements, and any single outlier has
minimal effect.  The statistical improvement follows ~1/√N: 26 points
give ~2.5× better precision than 4 points from noise averaging alone,
plus the regression fit is more robust.

**Error sources:**

1. **ADC quantization** — ±0.5 LSB on Ia (~0.005 mA at 20mA range).
2. **Charge pump ripple** — ~0.02V p-p on Ug1 (averaged by 64 HW samples,
   effective noise < 0.005V).
3. **Tube non-linearity** — linear regression assumes S ≈ const in zone;
   smaller zone reduces non-linearity error but fewer points.
4. **Settle time** — dynamic settle (per_volt_s + base_s) ensures voltage
   stabilization before measurement.  Insufficient settle → measurement
   at wrong Ug1 → systematic error in S.

**Recommended configuration:**

```json
{
  "srk_ug1_step": 0.04,
  "srk_samples": 5,
  "srk_settle_per_volt_s": 0.5,
  "srk_settle_base_s": 1.0
}
```

With these settings and a typical zone (ΔUg1 = 0.48V), the measurement
takes ~45 seconds (26 points × ~1.7s per point) vs ~7s for classic mode.

### 5-Point Distortion (HD2/HD3)

Computed automatically when the load line is active. Uses the 5-point method:

Given the load line intersections at Ug1 values from -swing to +swing around
the bias point Ug1₀, computes:
- **HD2** — 2nd harmonic distortion (%).
- **HD3** — 3rd harmonic distortion (%).
- **Pout** — output power (mW).
- **THD** — total harmonic distortion (%).

Results are shown in the load line info bar.

### Ra Sweep Analysis

**Button:** `Ra sweep` (next to load line parameters).

Opens a dialog showing how HD2, HD3, and output power change as the load
resistance Ra is swept from a fraction to several multiples of the nominal Ra.

**Outputs:**
- Top chart: HD2 and HD3 vs Ra (kΩ).
- Bottom chart: Output power (Pout) vs Ra (kΩ).
- Status bar: Ra values for minimum THD and maximum Pout.

Helps find the optimal Ra for best distortion/power trade-off.

### IMD (Intermodulation Distortion)

Computed alongside the load line analysis when data is available.

Uses polynomial fitting on the Ia(Ug1) transfer characteristic along the
load line to estimate:
- **IMD2** — 2nd-order intermodulation products (%).
- **IMD3** — 3rd-order intermodulation products (%).
- **IMD total** — combined IMD (%).

Shown in the load line info bar when the load line is active.

### Tube Quality Score

**Display:** Next to S/R/K label in the zone panel.

Compares measured parameters to the tube's datasheet nominal values:
- **Ia %** — measured Ia at operating point / nominal Ia × 100%.
- **S %** — measured transconductance / nominal S × 100%.

**Verdicts:**
| Score (avg) | Verdict | Color |
|-------------|---------|-------|
| ≥ 110% | Strong | Blue |
| ≥ 80% | Good | Green |
| ≥ 50% | Weak | Orange |
| < 50% | Replace | Red |

Computed automatically after each scan completes and measurement is saved.

### Amplifier Analysis Tab

**Controls:** "Amplifier" tab in the left panel (AmpControlPanel).
**Plots:** "Amplifier" tab in the main plot area (AmplifierTab).
**Menu:** `File → Operating point` navigates to this tab.

Integrated amplifier design analysis that replaces the old Operating Point dialog.
Activate by enabling the Load Line checkbox in Plot Options.

**UI structure:**
- Left panel "Amplifier" tab — collapsible sections: Circuit, Source & Data,
  Parameters (includes NFB), Optimizer (collapsed by default), Results.
- Main area "Amplifier" tab — two plots via QStackedWidget:
  - Page 0: THD vs Amplitude + Distortion vs Ra (default)
  - Page 1: Pareto front (shown after optimizer runs)
- Parameters Ub/Ra/Ug1/Swing are bidirectionally synced with Plot Options.
- Computation engine (AmplifierEngine) is pure Python, no Qt dependency.
- All three plots have crosshair + tooltip (QLabel on viewport, like CurveMarker).

**Features:**
- **THD & Pout vs Amplitude** — sweep signal level, showing how distortion
  grows with output power. The most practical graph for amplifier design.
- **Distortion vs Ra** — sweep load resistance to find the optimal Ra.
  Supports HD4/HD5 display and Gain/Zout/Pa curves via checkboxes.
- **Auto Q** — automatically finds the bias point that minimizes THD.
  Supports model path for more accurate DFT-based sweep.
- **Optimize Ra** — finds the load resistance that minimizes distortion.
- **Multi-parameter Optimizer** — grid sweep + swing sweep + parallel Pareto
  refinement. Searches Ub × Ug2 × Ug1 × Ra × Swing space. Background thread
  with progress bar and Cancel. Click on Pareto point to apply parameters.
- **THD-cap constraint** — "THD max %" spinbox (0 = off). Combined with the
  Max Pout target it answers the datasheet question "Pout at X% THD": points
  over the cap stay eligible for the swing sweep (THD falls with reduced
  swing), and refinement pushes output power up to the cap boundary. If the
  cap alone rejects everything, a dedicated error suggests raising the limit.
- **HD method selection** — 5-point, Chebyshev, or DFT. Applies to both
  sweeps (amplitude and Ra) and single-point analysis.
- **Results panel** — Q-point parameters, HD2/HD3/THD (+ HD4/HD5 if available),
  Pout in Watts, IMD, headroom, voltage gain, output impedance,
  auto-bias cathode resistor Rk, SRK cross-check.
- **Data source selection** — analyze current scan, overlay measurements,
  or imported data. Mode indicator shows measurements vs fitted model.
- **Ug2 filtering** — correct pentode analysis with screen grid voltage filtering.
  Ug2 displayed in Source & Data section and Q-point results.

**Circuit topologies:**
- **Single-Ended** — resistive or transformer load, full SE analysis.
- **Cathode Follower** — gain ≈ 1, very low Zout (≈ 1/gm).
  Uses CF-specific formulas: Gain = μ·Rk / (ra + (μ+1)·Rk), Zout = ra/(μ+1).
- **Push-Pull** — composite characteristic, even harmonic cancellation,
  balance error analysis, PP-specific power calculation (Pout = Ia_pp × Ua_pp / 4).
  Supports matched pair (mirror) and mismatched pair (from overlay data).
  Beyond tube B's measured Ug1 range the composite extrapolates Ia(B) to
  cutoff (normal for class AB). When the analyzed swing consumes
  extrapolated values, the panel shows a notice with the span; if tube B's
  data edge is still far from cutoff (edge current > 15% of the analyzed
  signal amplitude) a ⚠ warning recommends rescanning tube B to deeper
  Ug1. The tail follows the space-charge 3/2 law fitted from the two
  edge points.

**Stage parameters:**
- SE: Voltage gain Av = μ × Ra / (ra + Ra), Zout = ra ∥ Ra.
- CF: Gain ≈ 1, Zout ≈ 1/gm.
- PP: HD2 cancellation, balance_error metric.
- Priority: model gm/ra → numerical → SRK (with cross-check).

**Load line models:**
- Resistive: Ia = (Ub − Ua) / Ra
- Transformer (AC/DC): separate DC and AC load paths.
- Cathode Follower: Ia = (Ub − Ua) / (Rk + Rl).
- Push-Pull: each tube sees Ra_aa / 4 effective load.

### Tube Simulator (tube_sim.py)

Test and demo utility that generates realistic measurement data from
theoretical Koren tube models. Uses reference parameters for 46 tubes
from tube_params.json.

- Supports triodes, pentodes, triode-connected pentodes.
- Gaussian noise injection for robustness testing.
- Used by the test suite (~250+ amplifier/optimizer tests) for reproducible results.

---

## Comparison & Matching

### Matching Delta Analysis

**Button:** `Matching` on the Compare tab.

Select exactly 2 measurements (checkboxes), then click "Matching".

Opens a dialog showing:
- **Delta Ia curves** — difference in anode current between the two tubes at
  each Ua point, grouped by Ug1. Shows both absolute (mA) and percentage plots.
- **Match score** — quantitative matching:
  - Mean Δ, Max Δ, RMS Δ in mA.
  - Match % (100% = identical curves).

**Use cases:**
- Push-pull pair matching.
- Quality control — comparing tube to a reference.

### Curve-Based Tube Matching (Compare Tab)

**Match settings row** (always visible below toolbar):
Mode, Class, Size, Max Δ, Min pts, [Match Groups ▾], [Clear].

Group measurements into matched pairs by comparing full Ia(Ua) curves.
Distance metric: `100 - match_pct` from `compute_matching()`.
`compute_matching()` groups by (Ug1, Ug2) — correct for multi-Ug2 pentode
scans.  Same `lamp_id` → distance = ∞ (tube cannot match itself).

**Amp class weighting** (Class combo):

| Class | Weight | Delta thresholds (excellent/good/fair) |
|-------|--------|---------------------------------------|
| A | Ia (linear) | ≤2% / ≤5% / ≤10% |
| AB (default) | √Ia | ≤5% / ≤10% / ≤20% |
| B / Guitar | uniform | ≤8% / ≤15% / ≤30% |

Class A weights the operating point heavily (knee irrelevant).
Class AB balances operating region and knee/cutoff.
Class B treats entire curve equally (crossover region critical).

**Three source modes:**
- **All** — match all loaded measurements.
- **Visible** — match only rows passing current filters.
- **Selected** — match only highlighted rows.

**Mode filter:** Pent / TriC / Tri / All — prevents mixing pentode and
triode-connected measurements (different curve shapes, incomparable).

**Table columns** (in order): Show, Type, ID, Name, Mfg, An, Mode, SRK,
Timestamp, Color, Sel, Grp.  An is saved in `scan.an` since 2026-04-09;
older files show empty An.  Mfg is the manufacturing date (YYYY-MM, sorts
chronologically); empty cell rendered as `—`.  ID and Name stretch; An is
fixed 28px.  Sel (●) and Grp columns appear after matching.  Rows
colour-coded by group.  Grp shows group number, Δ%, point count, and ⚠
for low overlap.  Colour thresholds depend on amp class.

**Partial overlap handling:** Curves with different Ua/Ug1 ranges are
compared only on the overlapping region:
- ≥30 common points: normal comparison.
- 10–29 common points: valid but flagged ⚠ (low overlap).
- <10 common points: incomparable (distance = ∞, not grouped).

**Summary panel:** Left of plot after Calculate.  QTextEdit with coloured
HTML — groups with Δ, quality tier, point count, and member IDs.
Text is selectable.  Copy/Export buttons below.

**Ug2 filter panel:** Right of plot (always visible).  Checkboxes for
each clustered Ug2 level across all checked entries.  All/None buttons.
State persists across entry add/remove.  Curves with < 2 points skipped.

**Filters:** Type (lamp type combo), Regex (ID/Name search), Group
(appears after matching). Context menu: right-click → "Find similar".

**Logic module:** `lm19/tube_matching.py` (`match_curves()`,
`build_curve_distance_matrix()`, `group_by_distance_matrix()`).
**Matching engine:** `lm19/quality.py` (`compute_matching()` with
`amp_class` parameter).

### Aging Trend

**Button:** `Aging trend` on the Compare tab.

Select multiple measurements of the **same tube** (same type + ID), then click
"Aging trend".

Opens a dialog showing:
- **Ia at operating point** over time.
- **S (transconductance)** over time.
- Both plotted against measurement timestamps.

**Use cases:**
- Monitoring tube degradation.
- Predicting remaining useful life.
- Deciding when to replace a tube.

---

## Export

### PDF Report

**Menu:** `File → Export PDF` (also Compare tab `Export ▾ → PDF`).

An options dialog opens first (`app/report_options_dialog.py`):

- **Sections** — checkboxes for each report section (source of truth:
  `REPORT_SECTIONS` in `app/report.py`): nominal parameters, scan
  settings, measured S/R/K, quality verdict, distortion analysis, the
  I-V curves plot and the transfer plot. Sections whose data is missing
  are greyed out with the reason in the tooltip (e.g. "Enable the
  working line to get distortion analysis").
- **Presets** — `Full` / `Brief` (one-page summary).
- **Don't ask again** — export silently with the same settings until
  restart. Permanent defaults live in `config/app.json`:
  `report_sections` (CSV of section ids), `report_ask`.

The report/certificate language is set by `config/app.json:
report_language` ONLY (no UI selector): `""` = English; a code with a
`locales/<code>.json` file switches the document language (dropping in a
new translation file needs zero code changes); an unknown code falls
back to English with a WARNING in the log.

Dialog choices are remembered for the session; config files stay
read-only for the UI.

Report content:
- Tube type, lamp ID, export date, manufacturing date.
- Scan settings of the plotted scan (mode, Ua/Ug1/Ug2 ranges, heater).
- Nominal values from lamp configuration.
- Measured S/R/K values.
- Quality verdict.
- Distortion analysis (if the working line was active).
- Number of scan points.
- Plots rendered at print resolution (1600 px wide) via the pyqtgraph
  `ImageExporter` (WYSIWYG theme; falls back to a screen grab loudly if
  the exporter fails). Each plot goes on a new page when it doesn't fit;
  every page carries a footer with a page number.

A write failure (file locked by a PDF viewer, missing directory) raises
and reaches the user as an error dialog — never a false "PDF saved".
Uses Qt's built-in `QPdfWriter` — no external dependencies.

### Amplifier Analysis PDF Report

**Button:** `Export PDF` in the Amplifier panel's Results section.

Exports the last analysis (`app/amp_report_pdf.py`, QTextDocument-based):
the results text rebuilt in the configured report language
(`report_language` — the panel itself stays in the UI language), a header
with the actual analysis parameters (circuit, Ub, Ra, bias, HD method,
Ug2, data series), and print-resolution images: harmonic spectrum bar
chart (HD2–HD9), THD vs amplitude, HD vs Ra, and the optimizer Pareto
plot (the three sweep plots are WYSIWYG screenshots of the UI plots; the
spectrum is generated in the document language). The LTspice-verification
table is likewise rebuilt in the document language, including the stale
⚠ marker. The same options dialog applies (own section set, remembered
separately from the scan report); sections without data are greyed out
with the reason.

### LTspice Verification

**Button:** `Verify in LTspice` in the Amplifier panel's Results section
(requires LTspice at the standard install path and measurement data).

Simulates the analyzed circuit in LTspice batch mode
(`lm19/ltspice_verify.py`). The fitter combo next to the button selects
the simulated model: `Auto (as analyzed)` follows the analysis source;
an explicit Koren/Dempwolf/Reefman choice always runs a fresh fit of
that type (the table basis names it). By default a model fit is
exported to a `.sub`,
a purpose-built netlist reproduces the engine's exact idealization
(fixed V-source bias, fixed screen, ideal choke / perfectly-coupled PP
output transformer), and a 1 kHz sine at the analysis drive is run
through `.tran`/`.four`. The comparison table (THD, HD2/HD3, fundamental
Pout, average Ia) labels the basis of both columns. Options:

- **Amplitude sweep** — additional runs at 25/50/75% drive, paired with
  the engine's THD-vs-drive sweep by nearest swing.
- **Two-tone IMD** — SMPTE 60 Hz + 7 kHz (4:1); the engine IMD is a
  polynomial estimate, so the table carries an accuracy caveat.

UL tap is honored: the fixed screen supply is replaced by a behavioral
source `V = Ug2·(1−tap) + tap·V(anode)` per tube (the exact
`UltralinearModelWrapper` law; in PP each screen follows its own anode).

When the analysis ran on a model source, the verification exports THAT
loaded model as-is (`export_spice_from_model`, no refit) — the table
basis says `loaded <type> model`. Otherwise a fresh fit of the analyzed
series' points is used and labelled accordingly.

Runs on a background worker with Cancel (partial results stay visible).
The result becomes the `LTspice verification` section of the amplifier
PDF report (greyed out until a verification has been run). If the
analysis parameters change after a verification, the stored table gets a
visible ⚠ "verified with different parameters" prefix — in the panel and
in any PDF it is exported into. Working files live in
`%TEMP%\lm19_verify_*` (one directory per run; override:
`ltspice_verify_dir`); the bulky `.raw` waveforms are deleted after
parsing, the netlist/`.log`/`.sub` stay for traceability.

### Matched Pair/Quad Certificate

**Buttons:** `Certificate…` in the Health tab's Match panel and in the
Compare tab's match row (enabled once a Match has produced groups).

Generates a PDF certificate for one matched group
(`app/match_certificate.py`): title with the tube type, group number and
δ with a quality tier, the matching basis (Ia/S/R at the operating point
for Health; full-curve match % with the amplifier class for Compare), an
operating-point / scan-mode line, a per-lamp table (IDs, measurement
timestamps, manufacturing dates, and Ia/S/R/Index for Health), a pairwise
match table for quads, and — on the Compare side — a print-resolution
overlay of the GROUP's lamps only (one color per lamp, legend with the
IDs; other checked lamps never leak into the document). The shared report-options dialog picks the
sections; the certificate language follows `config/app.json:
report_language`. There are no free-text fields by design.

### SPICE Model Export

**Menu:** `File → SPICE export`.

Fits a vacuum tube model to the measured data and exports a SPICE subcircuit
file (`.sub`). A dialog (`SpiceExportDialog`) lets the user choose the model
algorithm and optionally generate an LTSpice test schematic.

**Topology auto-detection:** The model type is selected automatically based
on `tube_params.json`. For pentodes in triode connection (`ug2_mode =
"triode_connected"`), the triode model is used.

**Model algorithms:**

| Algorithm | Triode | Pentode | Best for | Source |
|-----------|--------|---------|----------|--------|
| **Koren** | 5 params (µ, Ex, Kg1, Kp, Kvb) | 6 params (+Kg2) | Triodes (~91%) | Norman Koren, 1996 |
| **Dempwolf** | v2 + grid current | v2 + current splitting | Pentodes (~80%) | Dempwolf et al. |
| **Reefman** | — | Derk/DerkE | Pentodes (alt.) | D. Reefman |

All models use unified pin naming: `A, G, K` (triode) or `A, G, K, G2`
(pentode), ensuring compatibility with a single LTSpice symbol.

Fitters use adaptive bounds based on estimated µ — low-mu power triodes
(6S19P, 6C33C, µ < 8) get wider parameter ranges. Reefman requires ≥ 3 Ug2
levels for reliable current splitting.

Benchmark: `py tools/fit_benchmark.py [filter] [--real] [--no-ref]` —
compare all three fitters on 86+ test datasets.

**Koren triode model** (5 parameters, 3-pin: A, G, K):

| Parameter | Description |
|-----------|-------------|
| **mu** (µ) | Amplification factor |
| **Ex** | Current exponent (replaces classical 3/2) |
| **Kg1** | Cathode current scale |
| **Kp** | Pinch-off (Region A control) |
| **Kvb** | Knee voltage factor (V²) |

**Koren pentode model** (6 parameters, 4-pin: A, G, K, G2):

All triode parameters plus:

| Parameter | Description |
|-----------|-------------|
| **Kg2** | Screen current scale |

The pentode equation uses `arctan(Ua/Kvb)` for the characteristic knee
and a separate screen current source `Ig2 = (Ug1 + Ug2/mu)^Ex / Kg2`.

**Fitting method:**
- **Primary:** `scipy.optimize.least_squares` (Trust Region Reflective),
  bounded optimization, 5000–8000 evaluations. Best convergence.
- **Fallback:** Coordinate descent with golden-section search (numpy only).
  Used automatically if scipy is not installed.

Initial guess values are loaded from `tube_params.json` reference database
(22 triodes, 12 pentodes with Koren parameters).

**The exported `.sub` file includes:**
- Comment header with fit quality (RMS / max error in mA).
- Fitted parameter values.
- Model SPICE equations (E1/G1 for triode, E1/G1/G2 for pentode).
- Interelectrode capacitances (if available in reference database).
- Grid current model (R + diode).
- Convergence resistors (1 GΩ).

Compatible with LTspice, PSpice, ngspice, and most SPICE simulators.

**LTSpice test schematic generation:**

When enabled in the export dialog, generates alongside the `.sub` file:
- `.asc` — test schematic with DC sweep (V1 = Ua, V2 = Ug1, V3 = Ug2 for
  pentode). Sweep parameters are extracted from measurement data.
- `.asy` — tube symbol (rectangle with labeled pins matching `.SUBCKT` order).

The user can open the `.asc` file in LTSpice and immediately run a DC sweep
to verify the fitted model visually. Templates are in `config/templates/`.

**Amplifier schematic export:**

When enabled in the export dialog, generates a complete amplifier circuit `.asc`
with the fitted tube model, ready for simulation in LTSpice.

Circuit types (4 × triode/pentode = 8 templates):
- **SE Resistive** — Ra load, Rk cathode bias. DC sweep by default,
  `.tran` and `.ac` as commented alternatives.
- **SE Transformer** — inductor-coupled output with `K1 L1 L2 0.95` coupling.
  L_primary computed from Ra and f_low: `L = Ra / (2π × f_low)`.
  L_secondary from turns ratio: `L2 = L1 × Rload / Ra`.
  Parameters: Rload (4/8/16 Ω speaker), f_low (20 Hz default).
- **Cathode Follower** — anode direct to Ub, Rk in cathode.
- **Push-Pull** — two tube instances (X1, X2) with per-tube Ra.
  Supports matched pair (same model) and mismatched pair (two different
  `.sub` models with separate `.include` directives). Tube A and Tube B
  selected from available measurement series in the export dialog.

All templates use parameters from the Amplifier control panel (Ub, Ra, Rk,
Ra_dc, Ug2, Ra_aa). The subcircuit name is derived from the save filename,
allowing multiple exports of the same tube type with distinct names.

**LTSpice round-trip validation:** automated tests verify that exported models
produce identical Ia values when simulated in LTSpice batch mode (Koren,
Dempwolf, Reefman × triode/pentode). Parser: `lm19/ltspice_raw.py`.

**Technical documentation:** See `docs/SPICE_KOREN_MODELS.md` for Koren model
reference, `docs/TUBE_MODELS_COMPARISON.md` for algorithm comparison.

### CSV Export

**Menu:** `File → Export CSV`.

Exports the current measurement data (or multiple measurements from Compare tab)
as a CSV or TSV file.

**Options dialog (CsvOptionsDialog):**

| Option | Values | Notes |
|--------|--------|-------|
| **Format** | Flat table (default) / Matrix | Matrix = parameter grid (Ua × Ug1) |
| **Separator** | Semicolon (default) / Comma / Tab | Tab produces .tsv |
| **Include computed** | Pa, Pg2, Ik columns | Flat format only |

**Flat table format:**
- Comment header with metadata (tube type, lamp ID, timestamp, SRK values).
- Columns: `Ua`, `Ug1`, `[Ug2]`, `Ia`, `[Ig2]`, `Uh`, `Ih`, optional `Pa`, `Pg2`, `Ik`.
- Triode mode omits Ug2 and Ig2 columns.

**Matrix format:**
- Parameter selector: Ia, Ig2, or Pa.
- One block per Ug2 level.
- Header row: Ua values.
- Each row: `Ug1=<value>` followed by the parameter at each Ua.
- Convenient for spreadsheet analysis and plotting.

**Multi-export** (from Compare tab):
- "All in one file" — single CSV with extra `Series`, `Lamp_Type`, `Lamp_ID`, `Mfg` columns.
- "Separate files" — one file per measurement in a selected folder.

**Manufacturing date (`mfg_date`)** is an optional per-measurement field
(`YYYY-MM`). When present:
- Flat / matrix CSV: extra header comment `# Manufactured: YYYY-MM`.
- Multi-CSV: dedicated `Mfg` column (always present in schema; empty cell
  when source measurement has no date).
- PDF report: header line `Manufactured: YYYY-MM` (hidden when empty).
- SPICE `.sub`: comment `; Manufactured: YYYY-MM` near the top.
- uTracer `.utd`: not exported (foreign format, schema fixed).

### uTracer Export

**Menu:** `File → uTracer (.utd)...`

Exports the current measurement as a `.utd` file compatible with uTracer3 GUI,
ExtractModel, Load Line Tool, and utMax.

**Options dialog (UtdExportDialog):**

| Option | Values | Notes |
|--------|--------|-------|
| **Curve format** | Output (default) / Transfer | Output = Ia(Va) with Vg stepping; Transfer = Ia(Vg) with Va stepping |

The dialog shows the resulting matrix dimensions (rows × columns) and a note
that Vs (screen voltage) and Vh (heater) are not stored in the `.utd` format —
Vs is encoded in the filename by convention (e.g. `EL84_250.utd`).

**Output format details:**

- **Header (line 1):** `Va (V) Ia (mA)` for triodes, `Va (V) Ia (mA) Is (mA)` for pentodes.
- **Step line (line 2):** stepping parameter values, e.g. ` Vg = -4 V  Vg = -2 V  Vg = 0 V`.
  For pentodes, each step value appears twice (Ia + Is columns).
- **Data rows (lines 3+):** running variable value followed by current values.
  Voltages use adaptive precision (integer, 1dp, or 2dp). Currents use 3 decimal places.

**Screen current (Ig2/Is):** Automatically included when any point has
`|ig2| > 0.001 mA`. Negative Ig2 (secondary emission) is preserved.

**Sparse grids:** Missing (Ua, Ug1) combinations are filled with 0.000 mA.

**Filename suggestion:** `<tube_type>_<Vs>.utd` for pentodes (e.g. `EL84_250.utd`),
`<tube_type>.utd` for triodes.

**Round-trip compatibility:** Files exported by LM19 can be re-imported via
`File → Import → uTracer (.utd)...` with full fidelity for Ia, Ig2, Ua, and Ug1
values. Vs and Vh must be re-entered on import (not stored in `.utd`).

**See also:** `docs/UTRACER_FORMAT.md` for the file format specification.

### JSON Measurements

All measurements are automatically saved as JSON files in:
```
measurements/<tube_type>/<lamp_id>__<timestamp>__<name>.json
```

Each file contains:
- Metadata: timestamp, tube_type, lamp_id, name, optional `mfg_date` (YYYY-MM).
- Scan settings: ua/ug1/ug2 ranges, uh, ih.
- Zone parameters.
- S/R/K values and individual results.
- Full array of measurement points.

The scan controls stay editable while a scan is running, so the file
distinguishes two kinds of fields:

- **Run description** (tube type, topology, Ug2 mode, voltage ranges,
  heater setpoints) is captured when the scan **starts**. Re-arming the
  controls for the next run mid-scan does not rewrite what the finished
  measurement says about itself. The after-scan SRK measurement likewise
  runs with the started scan's Ug2 mode and heater expectations.
- **Measurement labels** (name, lamp ID, `mfg_date`, SRK zone) are read
  when the scan **finishes** — typing them in while the scan runs is the
  normal workflow, and the zone always matches the one the S/R/K values
  were computed at. The timestamp marks completion (file names and
  history sort on it).

### Compare Tab — Multi-Export

**Button:** `Export ▾` dropdown on the Compare tab toolbar.

All four export formats are available directly from the Compare tab for
checked measurements:

| Action | Format | Extension |
|--------|--------|-----------|
| CSV | Flat table or matrix | `.csv` / `.tsv` |
| uTracer (.utd) | uTracer measurement matrix | `.utd` |
| SPICE export | Koren triode/pentode model | `.sub` |
| Export PDF | Measurement report | `.pdf` |

**Single measurement checked:** Opens the standard export dialog (same as
from the File menu).

**Multiple measurements checked:** A dialog asks how to export:

- **Combined** — merge all points into one file and open the standard export
  dialog. Useful for aggregate analysis or a single SPICE model fitted to
  combined data.
- **Separate** — select a destination folder, then each measurement is
  exported as an individual file with auto-generated names
  (`<tube_type>_<lamp_id>.<ext>`). Files are never overwritten — a `_1`, `_2`
  suffix is added if a file already exists. Errors per file are collected and
  reported at the end.

**Format-specific notes for separate export:**

- **UTD:** A single format dialog (output/transfer) is shown once and applied
  to all files.
- **SPICE:** Each file gets its own model fit (auto-detected topology).
- **PDF:** Each PDF carries a print-resolution render of ITS OWN lamp's
  curves (one curve per Ug1/Ug2 level), plus the shared report-options
  dialog (sections, language).
- **CSV:** Uses the existing CSV multi-export dialog with "All in one file"
  or "Separate file per measurement" options.

---

## Import

### CSV Import

**Menu:** `File → Import → Import CSV`.

Imports measurement data from CSV or TSV files produced by other tube testers,
spreadsheets, or LM19's own CSV export.

**Workflow:**
1. Select a `.csv` or `.tsv` file.
2. **CsvImportDialog** opens:
   - Auto-detects separator (tab, semicolon, or comma).
   - Shows a preview table (header + first 10 data rows).
   - Each column has a dropdown to map it to an LM19 field (`Ua`, `Ug1`,
     `Ug2`, `Ia`, `Ig2`, `Uh`, `Ih`) or `skip`.
   - Column mapping is auto-filled from header names (supports aliases:
     `Va` → Ua, `Vg` → Ug1, `Vs` → Ug2, `Is` → Ig2, `Vh` → Uh, etc.).
3. **ImportMetaDialog** opens for metadata not in the file:
   - Tube type (text).
   - Lamp ID (text).
   - Name (text) — default is the source file name (stem).
   - Description (multiline text) — prefilled from import metadata when available.
   - Ug2 voltage (V) — for files without a Ug2 column.
   - Uh heater voltage (V) — for files without a Uh column.
4. Imported points are added to the **Compare tab** as a new measurement entry.
   From there they can be overlaid on the main plot and analyzed.
5. The imported measurement is auto-saved to `measurements/<tube_type>/` using:
   `<lamp_id>__import_csv__<source_stem>.json` (no timestamp in file name).
   On collision, `_1`, `_2`, ... suffixes are appended.

**Supported sources:**
- LM19 CSV export (round-trip).
- uTracer data exported via Excel/spreadsheet.
- Generic tube tester CSV with voltage/current columns.
- Comment lines (starting with `#`) are automatically skipped.

### uTracer Import

**Menu:** `File → Import → Import uTracer`.

Imports `.utd` files from the uTracer tube tester software ("Save Measurement
Matrix" format).

**Supported formats:**
- **Output characteristics** — X = Va (anode voltage), stepping by Vg (grid voltage).
  Each step produces an Ia(Va) curve at a fixed Vg.
- **Transfer characteristics** — X = Vg, stepping by Va.
  Automatically converted to LM19's standard output format.
- **Screen current (Is)** — imported as Ig2 when present in the file.

**Workflow:**
1. Select a `.utd` file.
2. **ImportMetaDialog** opens for:
   - Tube type — auto-guessed from filename (e.g. `EL84_250.utd` → type `EL84`).
   - Lamp ID — default is source file name (stem).
   - Name — default is source file name (stem).
   - Description — prefilled with parsed .utd metadata (format, Is presence, dimensions, filename).
   - Ug2 (V) — guessed from filename suffix (e.g. `_250` → Ug2=250V).
   - Uh (V) — not stored in .utd format, prefilled from lamp config if tube type is recognized.
3. Points are added to the Compare tab.
4. The imported measurement is auto-saved to `measurements/<tube_type>/` using:
   `<lamp_id>__import_utd__<source_stem>.json` (no timestamp in file name).
   On collision, `_1`, `_2`, ... suffixes are appended.

**See also:** `docs/UTRACER_FORMAT.md` for the .utd file format specification.

### CurveTraceData Import

**Menu:** `File → Import → CurveTraceData (.dat)...`

Imports raw `.dat` files from pypsucurvetrace/curvetracedata.

**Workflow:**
1. Select a `.dat` file.
2. LM19 parses measured columns from CurveTraceData:
   - Ua from PSU1 measured voltage.
   - Ia from PSU1 measured current (A → mA conversion).
   - Ug1 from PSU2 measured voltage.
   - Rows with limiter flag (`PSU1 limiter = 1`) are skipped.
3. **ImportMetaDialog** opens with prefilled values:
   - Tube type — guessed from path (`.../<tube>/data/*.dat`) or filename.
   - Lamp ID — from sample name/file stem.
   - Name — from sample name/file stem.
   - Description — prefilled from `.dat` metadata (`Sample`, `Date / time`, parsed point count).
   - Ug2 (V), Uh (V) — user-adjustable.
4. Points are added to the Compare tab and auto-saved to:
   `<lamp_id>__import_curvetracedata__<source_stem>.json`
   (collision-safe via `_1`, `_2`, ... suffix).

### eTracer Import

**Menu:** `File → Import → eTracer (.csv)...`

Imports CSV v2.0 files exported by eTracer curve tracer (Essues Technologies).

**Format:** Groups of 6 rows per curve-set: HV1_V, HV1_I, HV2_V, HV2_I, NEGV, SWEEP_SOURCE.
Header comments contain format version, ETD config path, sweep settings.

**Workflow:**
1. Select a `.csv` file exported by eTracer software.
2. LM19 parses:
   - Ua/Ia from HV1 rows, Ug1 from NEGV row.
   - Ug2/Ig2 from HV2 rows (for pentode and triode-connected modes).
   - Topology auto-detected: `triode` (HV2 off), `triode_connected` (HV2 linked), `pentode` (HV2 swept).
   - Trailing `nan` values stripped; incomplete curve-sets skipped.
3. **Heater voltage** resolved by priority:
   - Lamp config (if tube type is known).
   - Companion `.etd` file (if present next to CSV — extracts `HEATER_V`).
   - User input in ImportMetaDialog.
4. **ImportMetaDialog** opens with prefilled values:
   - Tube type — from CSV filename, cleaned of `_triode`/`_pentode` suffixes. Falls back to ETD header for filenames with spaces.
   - Topology/ug2_mode — auto-detected from HV2 settings.
   - Description — format version, curve count, sweep ranges.
5. Points are added to the Compare tab and auto-saved to:
   `<lamp_id>__import_etracer__<source_stem>.json`

---

## Manual Tab

The Manual tab allows direct control of tube parameters:

| Control | Description |
|---------|-------------|
| **Ua, Ug1, Ug2** | Set voltages individually or all at once |
| **Uh, Ih** | Set heater voltage/current |
| **Anode (An) + Set** | Independent local anode selector on Manual tab (1/2) with explicit Set |
| **From Lamp** | Load nominal parameters from selected lamp |
| **Reset HV** | Reset Ua, Ug1, Ug2 to 0 (keeps heater) |
| **Take Point** | Capture a single measurement point |
| **Add to Main Plot** | Send manual points to the 2D plot |
| **Save** | Measure S/R/K and save as JSON |

The real-time chart shows Ia and Ig2 over time with configurable colors.
`Apply All` sends parameters in safe order with anode first:
`An -> Ug1 -> Ug2 -> Ua -> Uh -> Ih`.

The heater is guarded at two levels, because `Apply All` commands the
heater and the HV in one shot — on a cold or starved tube the anode
voltage lands on an unemissive cathode.

**Inline marker (advisory, 5 %).** A warning glyph appears next to the Uh
or Ih field whenever that setpoint is off the selected lamp's rated heater
by more than `HEATER_NOMINAL_TOLERANCE_PCT` — the same band that drives
the off-nominal badge in the live panel. The tooltip names the lamp and
its rating. A value dialled into the channel the lamp is not heated
through is flagged too. The marker is re-judged on every spinbox edit and
whenever the selected lamp changes.

**Apply All gate (blocking, `app.json: manual_heater_tolerance_pct`,
10 %).** Before commanding anything, `Apply All` checks two independent
things and reports every failing one on its own line of a single dialog:

1. the setpoint against the lamp's rated heater — a 1 V setpoint held
   perfectly steady still starves a 6.3 V cathode, and an over-driven
   setpoint cooks it;
2. the live reading against that setpoint — a deliberate reduced-heater
   experiment is legitimate, but only once the tube has settled there.

Both comparisons are two-sided. Zero setpoints (Apply All would switch
the heater off) and a wrong heater channel for the lamp get their own
lines. Declining applies **nothing at all**, and a heater that cannot be
read blocks the whole action rather than putting HV on an unknown
filament. The individual `Set` buttons are deliberately left ungated -
they are the expert path, and auto-apply would pop a dialog on every
spinbox edit.
Common UI terms (`Ua`, `Ug1`, `Ug2`, `Uh`, `Ih`, `An`, `N`, `X`, `Y`, units)
are now centralized in the `common` i18n dictionary and reused across tabs.

---

## Tube Health

**Tab:** `Tube Health` — separate tab for quick assessment of tube wear
and remaining useful life without a full curve scan.

The tab uses a two-panel layout with `QSplitter` (default ratio ~1:2).
The left panel contains setup, plan, live display, actions, and results.
The right panel contains the history table and the unified steps/measurement
points table.

### Quick Health Test

**Button:** `Quick Test` in the Actions group.

Performs a fast multi-point measurement at the tube's operating point to
evaluate key health metrics:

| Metric | Meaning | Source |
|--------|---------|--------|
| **Ia%** | Anode current vs reference (%) | OP measurement |
| **S%** | Transconductance vs reference (%) | Ug1 sweep (S phase) |
| **R%** | Plate resistance vs reference (%) | Ua sweep (R phase) |
| **K%** | Amplification factor vs reference (%) | K = S × R |
| **Rh** | Heater resistance score | Uh / Ih at OP |
| **Ig2/Ia** | Screen grid ratio (pentodes) | OP measurement |
| **Emission** | Cathode emission reserve | Ia(80%Uh) / Ia(100%Uh) |

**Composite Health Index** — weighted average of individual scores
(shipped `config/health.json` weights):

- Ia 35%, S 40%, Rh 10%, Emission 15% (if enabled).
- Screen weight is **0** by default: the ratio is still measured and
  stored, it just does not enter the index until the weight is raised.
- Missing metrics are excluded and weights renormalized (configurable).
  For lamps with `ih = 0` in `lamps.json` there is no `Rh` reference, so
  the index runs on Ia/S/Emission renormalized over 0.90.
- R% and K% are measured and stored but carry no weight.

**Verdict scale:**

| Index | Verdict | Colour |
|-------|---------|--------|
| ≥ 90% | Strong | Blue |
| ≥ 75% | Good | Green |
| ≥ 55% | Weak | Orange |
| < 55% | Replace | Red |

Thresholds are configurable in `config/health.json`.

**Test sequence:**

1. (Optional) Auto-preheat check — verifies heater voltage/current is at
   least `preheat_required_ratio` (default 75%) of nominal.
2. **OP phase — soft-start with Pa/Pg2 protection** (see below).
2b. **Bias servo** (optional, `Bias to reference Ia` in the plan) —
   walk Ug1 in `bias_servo_step_v` steps until Ia crosses the reference
   current, then bisect inside the last step (never wider — the bracket
   is at most one step), and leave the tube there so every later phase
   measures at that point. Transconductance is a function of anode
   current, so S taken at a bias where Ia has drifted is not comparable
   with a reference measured elsewhere on the same curve. The stepped
   approach bounds overshoot to one step's worth of current (S·step);
   walking up additionally stops at `bias_servo_pa_ceiling_pct` of the
   Pa trip limit, so the protection remains a backstop rather than the
   expected stop. Bisection gives up once the bracket falls below
   `bias_servo_ug1_floor_v` — the hardware cannot set finer. Each probe
   still passes the same Pa/Pg2 check as the OP ramp, the excursion is
   bounded by `bias_servo_max_shift_v`, and a trip restores the safe
   lock. The excursion limit is per-tube:
   `bias_servo_shift_margin × |ref − Ia| / lamp.s`, capped by
   `bias_servo_max_shift_v` — one park-wide constant was too small for
   a 6L6 and far too wide for a 12AX7; the applied limit is stored as
   `bias_servo.shift_limit_v`. The required bias itself is reported as
   `bias_shift_v` — a shifted bias with intact transconductance points at
   grid contamination rather than at wear.
   With the servo on, `Ia%` reads ~100 by construction (measured at the
   reference current), so the plan-point deficit is preserved separately:
   `metrics.ia_plan_pct` / `raw.ia_plan_ma` — shown as a suffix in the
   Result Ia line and in the history Δbias tooltip. In the points table
   the servo rows carry three step tags: `bias_servo` (intermediate
   probes), `bias_servo_op` (the ONE accepted measuring point, coloured
   like the OP row) and `bias_servo_restore` (back-to-plan after a
   failure). SRK is always measured around a single point — the accepted
   one, or the plan bias when the servo is off or failed. If the reference current is
   unreachable inside the allowed excursion, the plan bias is restored
   and the status is shown next to Ia instead of silently measuring at
   the wrong point.
   Saved `conditions` keep the PLAN bias plus a `bias_servo` flag — the
   servo outcome is per-tube data (`health.bias_servo.ug1`), so servo
   runs of differently worn tubes stay matchable against each other,
   while servo and fixed-bias runs are never mixed in one match pool.
   A failed servo run was measured at the plan bias and therefore
   counts as a fixed-bias run.
3. **S phase** — sweep Ug1 around OP (central difference or multi-point
   linear regression for transconductance S).
4. **R phase** — measure at Ua ± δ for plate resistance R via central
   difference.
5. **Restore OP** — return Ua and Ug2 to nominal values.
6. **Emission phase** (if enabled) — measure Ia at 100% Uh, reduce to
   configurable ratio (default 80%), wait for stabilization, measure Ia
   at reduced Uh, restore Uh to nominal.
7. **Scoring** — compute all metrics, index, and verdict.
8. **Auto-save** — result saved to `health_measurements/`.

**Soft-start OP approach with Pa/Pg2 protection:**

To avoid frying a worn or defective tube on operator error, the OP
phase does **not** jump straight to the target bias. Instead:

1. Ug1 is set to the safe-lock value (`app.json:ug1_after_stop`,
   default −24 V) — the tube is biased fully closed.
2. Ua is set to target (lamp closed, Ia ≈ 0 — anode voltage is safe to
   apply now).
3. Ug2 is set to target for pentodes (lamp still closed, no Pg2 yet).
4. **Ug1 ramp**: Ug1 is stepped from safe lock toward target in 1 V
   increments (`health.json:op_ug1_ramp_step_v`). After every step:
   - A fast single-shot read of Ua, Ug1, Ug2, Ia, Ig2.
   - Compare `Pa = Ua × Ia` against `lamp.pa_max × pa_safety_pct/100`
     (default 135% of the datasheet maximum).
   - Compare `Pg2 = Ug2 × Ig2` against `lamp.pig2_max × pig2_safety_pct/100`
     (pentodes only).
   - If either limit is exceeded, Ug1 is immediately driven back to
     the safe lock and the test aborts with a detailed dialog showing
     the trip point (Ua, Ug1, Ug2, Ia, Ig2, measured power, limit,
     datasheet value, ramp step, tube ID, mode).

This catches: shorted / leaky tubes, tubes with very high emission,
incorrectly set operating points, and incorrect Pa_max / Pig2_max in
the lamp config. It does **not** protect during the S/R/Sg2 sweep
extrema — those are very brief and bounded by δ from a known-good
OP. Hardware Er bits remain the last-resort safety net for them.

The ramp is configurable via `config/health.json`: `op_ramp_enabled`
(toggle, default `true`), `op_ug1_ramp_step_v` (step size, default
1 V), `pa_safety_pct` (default 135%) and `pig2_safety_pct` (default
120%). Tubes without `Pa_max` / `Pig2_max` in `config/lamps.json`
have their respective check skipped.

**Stop controls:**

- `Stop (keep heater)` — aborts the test but keeps Uh at current level
  for quicker subsequent tests.
- `Stop / Heater Off` — aborts and sets Uh = 0 for safety.

After test completion (or failure), output voltages (Ua, Ug1, Ug2) are
reset to zero.

### SRK Measurement (Health)

The health test measures S, R, K directly using configurable delta
voltages around the operating point, inspired by the µTracer Quick Test
methodology.

**S (transconductance):**

- `Points = 5` (default): two-point central difference at Ug1 ± δVg.
  S = ΔIa / ΔUg1.
- `Points > 5`: multi-point Ug1 sweep across [Ug1 − δ .. Ug1 + δ],
  S computed by linear regression.

**R (plate resistance):**

- Always two-point central difference at Ua ± δVa.
  R = ΔUa / ΔIa (kΩ).

**K (amplification factor):** K = S × R.

**Sg2 (screen grid transconductance, pentode mode only):**

- Two-point central difference at Ug2 ± δVg2 (at OP Ua and Ug1;
  the anode is restored to the operating point after the R-phase
  sweep before Sg2 is measured). Sg2 = ΔIa / ΔUg2 (mA/V).
- Only measured when `ug2_mode == "pentode"` (independent Ug2).
  Skipped for triodes and triode-connected pentodes.
- Default δVg2 = 5% of Ug2 (`delta_ug2_pct` in `health.json`),
  separate from the 10% used for Ua and Ug1.

**µ(g1→g2) (screen amplification factor):** µ = S / Sg2.

In datasheets this is the commonly listed "µ" for pentodes
(typically 10–30), distinct from the full µ(g1→a) = S × R which
reaches hundreds and is noisy due to high Rp.

**N (repeats):** Each point is read N times (default 5) with robust
averaging (median + outlier trimming) to reduce noise.

**Uncertainty estimation:** Relative uncertainty of S, R, K, Sg2 is computed
via error propagation from instrument precision (Ua ±1 V, Ug1 ±0.04 V,
Ug2 ±1 V, Ia noise) and displayed as `value ± abs_error` in the result card.
Implementation: `analysis.estimate_srk_uncertainty()`, `analysis.estimate_sg2_uncertainty()`.

All SRK math (linear regression, `compute_srk_direct`, `compute_sg2_direct`,
`compute_mu_g1g2`, uncertainty estimation) is in `lm19/analysis.py`, shared
between the health test and the main scan's `measure_srk`.

### Emission Test

**Checkbox:** `Emission enabled` in the Setup group (default: on).

Evaluates cathode emission reserve by comparing anode current at nominal
vs reduced heater voltage:

1. Measure `Ia100` at `Uh = 100%` nominal.
2. Set `Uh` to `emission_uh_ratio × Uh_nom` (default 80%).
3. Wait for Ia to stabilize (dynamic criterion: `|dIa/dt|` below
   threshold over a sliding window).
4. Measure `Ia80` after stabilization.
5. Compute `EmissionRatio = Ia80 / Ia100`.
6. Restore Uh to nominal.

`emission_score` normalizes the ratio against `reference.emission_ratio`
when the active reference carries one (personal baseline / type ref /
type median), otherwise against `health.json: emission_ratio_nominal`
(0.90). The base actually used is stored as `metrics.emission_ratio_ref`.

**Stabilization parameters** (all configurable in `health.json`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emission_stable_warmup_ratio` | 0.50 | Fraction of warmup_s for t_max |
| `emission_stable_min_s` | 20 | Minimum stabilization time |
| `emission_stable_max_s` | 120 | Maximum stabilization time |
| `emission_stable_slope_threshold_ma_per_s` | 0.01 | dIa/dt threshold |
| `emission_stable_window_points` | 5 | Sliding window size |
| `emission_sample_period_s` | 2.0 | Ia read interval during wait |

During stabilization, the Live Panel updates in real-time: Ia, Ig2,
Uh, Ih, and Pa are refreshed on every sample. The progress bar shows
elapsed time, ETA, and max wait.  If t_max is reached without
stabilization, the result is flagged as `low_confidence`.

**Deep emission (sweep mode):** instead of one reduced-heater point, the
heater walks a descending grid (`emission_uh_sweep_steps` points down to
`emission_uh_sweep_min_ratio`), stabilizing at each. Ia(Uh) rides the
space-charge plateau while emission is abundant and collapses once it
becomes temperature limited; the transition is the **knee**. As the
cathode's emitting material is consumed the knee migrates toward higher
heater voltage, which makes its position a more direct wear measure than
any single-point ratio.

The knee is located the classic Miram way — as the **intersection of two
fitted lines**: the plateau line and the emission-branch line. The
plateau is fitted, not assumed flat: a real tube's plateau sags several
percent by 80% heater (cathode temperature shifts the contact
potential), and measuring the drop from the extrapolated plateau LINE —
rather than from Ia at nominal — keeps that sag from reading as a knee.
When the configured grid does not bracket the falling branch, the sweep
**descends adaptively**: it keeps stepping down at the grid pace until
the branch holds two points, bounded by the absolute floor
`emission_uh_sweep_abs_min_ratio` (default 50% — brief starved operation
is standard Miram practice, but below that the thinned space-charge
cloud stops shielding the cathode from ion bombardment, and Ia is too
small to measure; the last step clamps to the floor). Outcomes:

- two or more falling points — true two-line knee, confidence `ok`;
- exactly one — the knee is bracketed but its slope is not: bracket
  midpoint, confidence `low` (⚠ on the Em line);
- none down to the floor — `below_range`: reserve reported as a lower
  bound (`>N%`), knee unknown — the honest answer for a fresh cathode;
- no plateau at all (the curve is one steep line) — the tube is
  emission-limited at nominal already: reserve 0, confidence `low`.

The report carries the whole curve, `uh_knee`,
`emission_reserve_pct = (Uh_nom − Uh_knee)/Uh_nom`, `knee_confidence`
and the fitted `plateau_slope_ma_per_v` (a diagnostic in its own right).
The configured `emission_uh_ratio` always stays on the grid, so
`EmissionRatio` remains comparable with single-point measurements and
stored baselines. A wall-clock budget (`emission_sweep_max_total_s`)
truncates the curve, flags it and lowers confidence rather than running
unbounded.

**Emission sensitivity:** the ratio only discriminates when the tube is
deep in space-charge limitation at the test point. `Ik = Ia + Ig2` is
compared against `lamp.ia_max`; below `emission_min_ik_ratio` the result
is flagged `emission_low_sensitivity` and the panel qualifies the verdict
— a low-current operating point returns ~1.0 even from a depleted
cathode.

**Emission verdict (standalone):** absolute cathode-reserve scale, shown
in the result line next to the ratio. Unlike `emission_score` it is not
divided by any reference, so it stays honest when the reference itself
came from a worn tube. Thresholds: `emission_ratio_good_min` /
`emission_ratio_weak_min` in `health.json`.

| EmissionRatio | Code | Status |
|---------------|------|--------|
| ≥ 0.70 | `normal` | Reserve normal |
| 0.50 – 0.70 | `weakened` | Reserve weakened |
| < 0.50 | `exhausted` | Cathode exhausted (replace) |
| — | `na` | Emission test disabled / no data |

### What the Health test does NOT cover

A `Strong` verdict means "the metrics this instrument can measure look
good", not "the tube is sound in every respect". The hardware cannot
measure, and the test therefore never checks:

| Not measured | Why | Usual symptom it would catch |
|---|---|---|
| Grid current `Ig1` / grid leakage | The grid is driven from a low-impedance source; the classic method compares Ia at two different `Rg1` values, which needs a switchable grid resistor | Runaway bias in a fixed-bias amp, "the tube red-plates after 20 minutes" |
| Heater-cathode leakage | No isolated Uk/f source, and the check is only valid at full anode temperature anyway | Hum, noise |
| Inter-electrode shorts as such | Only visible indirectly, through their effect on Ia/Ig2 | Crackle, intermittent faults |
| Cathode interface resistance | Shows up as gm falling with frequency; a DC instrument cannot see it | "Sleepy" tube after long idle-biased service |
| Microphony, gas (as a direct measurement) | Needs mechanical excitation / a gas test circuit | Ringing, blue glow, drift |

The one indirect handle the instrument does have on the thermal failure
family is time at the operating point: several of these faults (thermal
runaway, gas, grid emission) only appear once the anode reaches
equilibrium, which is minutes rather than the seconds a Quick test
spends there. Treat a Quick-test `Strong` on an unknown used tube as
"worth putting in a test amp", not as a clean bill of health.

### Reference Sources

Three reference modes for comparison:

1. **Datasheet** (default) — nominal values from `config/lamps.json`
   (Ia, S, R, K, Uh, Ih).
2. **Type Reference** — user-saved measurement of a known-good tube of
   this type. Multiple references per type are supported; one is marked
   `active`. Stored in `<health_refs_dir>/type/<tube_type>/<ref_id>.json`
   (`health_refs_dir` in `config/app.json`, default `config/health_refs`).
3. **Personal Baseline** — first health measurement of a specific
   `lamp_id`, used to track individual tube degradation. Stored in
   `<health_refs_dir>/personal/<tube_type>/<lamp_id>.json`.

**Priority fallback:** Personal Baseline → Type Reference → Datasheet.

**UI controls:**

- `Reference Mode` radio/combo — select Datasheet / Type / Personal.
- `Type Reference` dropdown — appears when Type mode is selected,
  lists all saved type references for the current tube.
- `Set Active` button — marks the selected type reference as the
  active one for future tests.
- `Save as Type Reference` button — saves the current test result as
  a new type reference.

### Plan Settings

**Group:** `Plan Settings` on the left panel.

Editable parameters for the health test measurement plan:

| Control | Range | Default | Description |
|---------|-------|---------|-------------|
| **OP: Ua** | 0–1000 V | from lamp config | Operating point anode voltage |
| **OP: Ug1** | −100–0 V | from lamp config | Operating point grid voltage |
| **OP: Ug2** | 0–1000 V | from lamp config | Screen voltage (pentodes) |
| **δ Ua** | 5–100 V, step 1 V | 10% of Ua, rounded to 1 V | Delta for R measurement |
| **δ Ug1** | 0.08–10 V, step 0.04 V | 10% of |Ug1|, rounded to 0.04 V | Delta for S measurement |
| **Points** | 5–21, step 2 | 5 | Total SRK measurement points |
| **N** | 1–50 | 5 | Repeats per measurement point |
| **Emission Uh ratio** | 0.10–1.00 | 0.80 | Uh ratio for emission test |

**Ug2 mode** (pentodes only):

- `Independent` — fixed Ug2.
- `Ug2 = Ua + offset` — Ug2 tracks Ua (triode-connected mode) with
  configurable offset.

Both Ug2 mode controls and Emission Uh ratio are hidden for triodes
and when emission is disabled, respectively.

**`Apply from lamp`** button resets all plan parameters to defaults
calculated from the current lamp configuration.

**Validation:** The plan is validated in real-time. Warnings are shown
for deltas below instrument precision (Ua < 5 V, Ug1 < 0.08 V),
voltages out of device range, or if Ug1 ± δ falls outside [−24, 0] V.

### Steps / Measurement Points Table

**Panel:** `Steps / Measurement Points` on the right panel (below
the history table).

A single unified table that serves three purposes across the test
lifecycle:

| Phase | Behaviour |
|-------|-----------|
| **Plan** | Shows planned steps (OP, S±, R±, Emission) with target voltages; current columns display "—" |
| **Live** | Updates rows in real-time as each measurement completes; auto-scrolls to the current step |
| **Result / History** | Filled from saved `measurement_points` when a history entry is selected |

**Columns:** Step, Ua, Ug1, Ug2, Uh, Ih, Ia, Ig2, Pa, Pg2, Details.

**Colour coding:** OP = green, SRK = blue, Emission 100% = light orange,
Emission 80% = darker orange.

**`Copy points`** button copies selected rows (or all) to the clipboard
as tab-separated text for pasting into a spreadsheet.

### Health History

**Table:** Right panel, full-height.

Shows all health measurements for the selected tube type.

**Columns:** Timestamp, Lamp ID, Name, Mfg, An, Mode, Ua, Ug1, Ug2,
Index, Ia%, S%, R%, K%, Ia, S, R, K, µg2, Emission, Reference, Sel, Grp.

**Headline columns Index and Ia** use bold font and a distinct light
background so the two most-scanned results stand out. The verdict row
tint colours every other cell but skips those two, so they remain
visually pinned even on Strong/Good/Weak/Replace-coloured rows.

Ua/Ug1/Ug2 show the operating point at which the measurement was taken
(from `conditions` in the JSON), so two rows for the same tube can be
compared apples-to-apples when bias differs. These three columns are
**hidden by default** — they rarely vary across a tube's measurement
history — and revealed by the `Show conditions` checkbox in the filter
row above the table.

Timestamp is displayed as `YY-MM-DD HH:MM` for compactness; the stored
full ISO string is kept on the sort key so chronological order is
preserved. Mfg holds the manufacturing date (`YYYY-MM` when entered in
the Lamp panel), displayed as `YY-MM`; full year is shown as a hover
tooltip to disambiguate vintage tubes (1955 vs 2055). Sortable
chronologically (lex order matches date order); empty cell shows `—`.

Sel and Grp columns are hidden until a matching operation is run. When
shown they appear at visual columns 0 and 1 (leftmost) regardless of
logical column index.

**Column widths:** Free-text columns (Lamp ID, Name, Reference) have
fixed widths from `ui_theme.py` (`HEALTH_HISTORY_LAMP_ID_WIDTH` etc.)
and are user-resizable. All other columns auto-size to content
(`QHeaderView.ResizeMode.ResizeToContents`) — the header width is the
minimum, so short headers never truncate longer values.

µg2 shows µ(g1→g2) = S / Sg2 for pentode mode measurements (structural
parameter, typically 10–30). Sg2 is saved in the JSON but not shown.

Index and percentage columns (Ia%, S%, R%, K%) are displayed as
integers.  Scan mode is abbreviated (Tri / Pent / TriC).  Rows are
colour-coded by verdict (Strong=blue, Good=green, Weak=orange,
Replace=red).

**Toolbar** (above the table):

- **Lamp ID filter** — show all or a specific lamp.
- **Copy** — copies selected rows (or all) to clipboard as TSV.
- **Export CSV** — saves selected rows (or all) to a `;`-delimited
  CSV file (UTF-8 BOM, opens correctly in Excel).
- **Reload** — refresh from disk.

**Features:**

- Sortable by clicking column headers (default: timestamp descending).
- Clicking a row loads its result into the Result card and fills the
  steps table with its measurement points.
- **Live vs stored view:** during a running test the steps table is a
  live view — planned points fill in under their planned names and
  Details, while bias-servo probes and adaptive-descent points (which
  the plan could not predict) appear under their own step tags, with
  the not-yet-reached tail of the plan previewed below. Servo rows
  carry live Details: each probe shows its deviation from the
  reference current ("→ ref 48.0 mA (Δ −3.8)"), so the convergence is
  visible row by row, and the moment a probe is accepted a dedicated
  event retags it as the OP row with its "Δbias" — no waiting for the
  test to finish. Clicking a history row mid-test switches the table
  to that stored measurement; live points keep accumulating in a buffer
  and the **⟵ Live** button (visible only in this situation) brings the
  process view back, nothing lost. Starting a new test always resets to
  the live view.
- History and Steps tables are separated by a vertical `QSplitter`
  (resizable by dragging).

### Tube Matching

Find similar tubes and group them into matched pairs/quads for push-pull
amplifiers.

**UI:** Left panel has two tabs — **Measure** (standard health controls) and
**Match** (matching settings and summary).  History table gains two columns:
**Sel** (●) marks every row that participates in the match (both anode
systems of a twin tube can be active at once), **Grp** shows group number
(or Δ in "Similar to" mode; the anchor row shows **★**).

**Two modes:**

For large sets the O(N²) pair-comparison loop is shown in a modal
**QProgressDialog with Cancel** (Compare tab) so the UI stays responsive
without spawning a worker thread. The dialog appears only if the run
takes longer than 500 ms; cancelling raises ``MatchCancelled`` which
the tab catches and discards — the previous match result is preserved.

- **Groups** — partition all tubes into pairs (2) or quads (4) of maximally
  similar tubes. Pair selection is configurable via the **Algorithm** dropdown
  in the Match panel:
  - **Tightest-first (greedy)** — default for Health tab. Picks the globally
    closest pair, locks both, repeats. Stops when `Max Δ` is exceeded. Best
    for **"select N best pairs from a box of tubes, return the rest"**:
    never sacrifices a tight pair to balance the overall sum.
  - **Optimal sum (Hungarian)** — minimises total distance across all pairs
    using `scipy.linear_sum_assignment`. Best when every tube must end up
    in some pair within `Max Δ`. **Default for Compare tab** (preserves
    legacy behaviour; user can switch to greedy via the Algorithm dropdown).
  Groups of size > 2 (quads, etc.) always use greedy chunking by average
  distance regardless of the algorithm setting. The two tabs persist the
  choice independently (`config/health.json:matching_algorithm` for Health,
  `config/app.json:compare_matching_algorithm` for Compare).
- **Similar to…** — rank all tubes by distance from a selected anchor tube.
  Two context-menu items in the history table: **"Find similar"** anchors
  on the clicked lamp's latest/best protocol-eligible measurement (under
  the individual-bias protocol that is its latest *servo* run);
  **"Find similar (this measurement)"** anchors on the exact clicked
  entry. In both cases the candidate pool and the protocol gates are
  built from the **anchor's own operating conditions** (including the
  bias-servo flag), not from the newest history entry. An anchor that
  cannot rank is a visible error in the summary ("no usable measurement
  in this pool" / "not eligible under the current protocol") — never a
  silently substituted lamp. Candidates cut by a protocol gate or by
  `Max Δ` appear in the **Unmatched** list instead of vanishing.

**Distance metric:** Normalised weighted Euclidean distance on absolute
values (Ia mA, S mS, R kΩ) — not percentages, so results are independent
of reference source.  Weights configurable.  Pentode default: Ia 0.5, S 0.5,
R 0.0 (Rp irrelevant for pentodes).  Triode default: Ia 0.4, S 0.3, R 0.3.
Weights auto-switch when tube mode changes.

**Matching protocol** (dropdown in the Match panel; default from
`config/health.json:matching_protocol`): what "matched" means depends on
how the target amplifier biases its tubes (Aiken / Apex / TubeSound
sources in `external_sources/theory/`).

- **Strict** (default) — bias-servo and fixed-bias measurements never
  mix: exact conditions match, the previous behaviour.
- **Shared bias amp** — one bias adjustment feeds both tubes, so they
  must draw the same current at the *same* grid voltage. Servo and
  fixed runs mix via the **plan-point current** (a servo run's op Ia
  sits at the reference by construction and says nothing about wear).
  Pairs whose predicted quiescent-current imbalance
  **δIq = |ia_plan₁ − ia_plan₂|** exceeds
  `matching_max_iq_imbalance_pct` (default 10 %) of the pair's mean
  current become incomparable **before** selection — the algorithm
  pairs the next-best candidates instead of dropping tubes. Each
  group's δIq shows in its summary header, in Copy groups text, in the
  CSV export (a permanent "δIq (mA)" column, dash outside this
  protocol) and as a dedicated line in the pair certificate.
- **Individual bias amp** — each tube has its own bias pot: the
  operator dials DC balance regardless of bias spread, so Ia carries
  no weight (the Ia spin greys out) and matching ranks S (and R) at
  the reference current. Only servo runs qualify, and each tube's
  |Δbias| must fit `matching_bias_adjust_range_pct` (default 30 % of
  the plan bias — percent, not volts, so the limit scales from a −2 V
  high-mu triode to a −60 V transmitting tube).

Gate thresholds are config-only (no UI spins); 0 disables a gate.

**Settings:**

| Setting | Options | Default |
|---------|---------|---------|
| Use | Latest / Best (Index) | Latest |
| Anode | Each / Combined (avg An1+An2) | Each |
| Group size | 2–8 | 2 |
| Max Δ | 0 (no limit) – 100% | 0 |

**Selection:** When a tube has multiple measurements, one is auto-selected
(latest or best by Index).  Click the **Sel** column cell of a row to
**pin** that measurement for its lamp + anode — the match recalculates
with the pinned one; clicking the pinned row again unpins it (back to
latest/best). A pin acts within its own operating-point pool only, and
an explicit "Find similar (this measurement)" anchor always wins over a
pin on the same lamp. Pins survive recalculations and reset on Clear.

**Conditions filter:** Only measurements at the same operating point
(Ua, Ug1, Ug2, ug2_mode, bias-servo flag) are compared. Auto-detected
from the most recent entry (Similar mode: from the anchor's own entry).
The conditions label shows the pool actually used, including a
"servo pool" marker — and, under the Strict protocol, a hint that a
servo pool degenerates the Ia term (every tube sits at the reference
current; consider the shared/individual protocols). The "N of M" counter
counts lamps and protocol-compatible entries of that pool.

**Output:** Groups displayed in the table with colour-coded rows (8 alternating
group colours).  Δ values colour-coded by quality: green ≤2%, blue ≤5%,
orange ≤10%, red >10%.  Copy to clipboard or export to CSV.

**Calculate source:** Three buttons — All (all entries), Visible (filtered
rows), Selected (highlighted rows).  Enables workflows like: filter by regex
`ER_`, then match only those tubes.

**Table filters:** Regex search (Lamp ID / Name), Mode (Pent/TriC/Tri),
Verdict (Strong/Good/Weak/Replace), Group (appears after matching).

**Logic module:** `lm19/tube_matching.py`.  **UI widget:** `app/match_panel.py`.
**Config:** `config/health.json` keys `matching_*`.

### Data Storage

**Health measurements:** `health_measurements/<tube_type>/<lamp_id>__<timestamp>__<name>.json`

Each file contains: metadata (timestamp, tube_type, lamp_id, name),
health scores and verdict, SRK values with uncertainty, raw measurements
(Ia at OP, Ia100, Ia80), stabilization details, conditions (voltages),
measurement plan, and full array of measurement points.

**Type references:** `<health_refs_dir>/type/<tube_type>/<ref_id>.json`

**Personal baselines:** `<health_refs_dir>/personal/<tube_type>/<lamp_id>.json`

`health_refs_dir` defaults to `config/health_refs` and is set in
`config/app.json`; relative paths resolve against `lm19_app/`, absolute
paths are used as-is (same rules as `measurements_dir`). Both subtrees
move together — see CONFIG_REFERENCE.md.

Both reference types store: identification, conditions, reference values
(Ia, S, R, K, Rh, screen_ratio, emission_ratio), timestamp, and
active/source flags.

**Configuration:** `config/health.json` — all health-specific parameters.
See `docs/CONFIG_REFERENCE.md` for the full parameter list.

---

## Calibration

Software calibration corrects systematic errors in the device's ADC (READ)
and DAC/PWM (SET) paths. Each channel has independent gain and offset
coefficients: `corrected = raw × gain + offset`.

### Calibration Tab

**Tab:** `Calibration` — dedicated tab with three areas:

1. **Live Readings** table — real-time raw and calibrated values for all channels.
2. **Coefficients** table — all channel calibration coefficients (gain, offset,
   calibrated_at) for READ and SET directions.
3. **Manual Edit** section — edit coefficients, test values, and preview.

**Actions:**
- **Wizard** — launches the step-by-step calibration wizard for the selected channel.
- **Reset** — resets the selected channel to defaults (gain=1, offset=0).
- **Reset All** — resets all channels.
- **Save** — writes current coefficients to `config/calibration.json`.
- **Discard** — reverts to last saved state.

### Calibration Wizard

Two-point calibration wizard for a single channel. The user selects a channel
in the coefficients table and clicks "Wizard".

**Voltage channels** (Ua, Ug1, Ug2, Uh, Ih) — unified READ + SET wizard:

1. **Prep page** — connect multimeter to the channel output.
2. **Low point** — device sets ~17% of range, user enters multimeter reading.
3. **High point** — device sets ~83% of range, user enters multimeter reading.
4. **Result page** — computes the READ correction from the multimeter;
   the SET correction is **auto-derived from READ** + the observed DAC
   transfer (`derive_set_two_point`), marked "auto" in the UI. Applying
   SET requires applying READ (no basis otherwise). Quality metrics shown
   for both.

The wizard commands its measurement points **raw** (no SET correction) —
it characterizes the uncalibrated transfer; applying feedforward during
calibration would compound corrections.

Calibration points are derived from `DEFAULT_LIMITS` (e.g. Ua: 50 V / 250 V).

**Current channels** (Ia low, Ia high, Ig2) — ammeter method:

1. **Prep page** — connect load resistor and ammeter, load recommendations shown.
2. **Low point** — device sets source voltage, user reads ammeter.
3. **High point** — higher source voltage, user reads ammeter.
4. **Result page** — computes READ correction, shows quality.

Ia has two hardware ranges (20 mA / 200 mA) calibrated independently
(`ia_low_read`, `ia_high_read`). Range selection is automatic based on
`IA_RANGE_THRESHOLD` (17 mA).

**Wizard features:**
- Live polling (~1 Hz) showing instantaneous device reading.
- N-point averaged measurement with configurable N (1–100), sigma/stats.
- Editable source voltage for current channels.
- Quality metadata (point spread, residuals, meter accuracy) saved per channel.
- Safe reset of all touched channels on wizard close (Finish or Cancel).

### Manual Edit

Two-column layout for the selected channel:

| Coefficients (left) | Test (right) |
|---------------------|--------------|
| Gain spinbox | Test value spinbox + unit |
| Offset spinbox + unit | **Set** button (with calibration) |
| Meter ±% spinbox | **Set raw** button (without calibration) |
| | Preview: `raw avg → calibrated` |
| | N spinbox (rolling average samples) |

- **Apply** saves gain, offset, and meter accuracy. `*_set` rows are
  read-only — SET coefficients are auto-derived by the wizard.
- **Apply to both Ia ranges** checkbox — copies coefficients to both `ia_low` and `ia_high`.
- **Meter ±%** — editable for READ channels, disabled for SET (inherits from READ).
- **Preview** uses rolling average of live readings with the edited coefficients.
- **Set** applies SET calibration; **Set raw** bypasses it (raw encode only).

### Feedforward (plan B)

SET calibration is applied to **every working-point command** in the
application (scan, SRK, health, manual tab, preheat, heater restore):
the command is pre-corrected once (`apply_set`) before the settle loop,
and the verify compares the raw device reading against
`read_inverse(target)` — the reading expected when the physical value
equals the target. Shutdown/zeroing commands and the Ug1 safe-lock stay
**raw** by design (`apply_set(0) = offset` could command a non-zero
voltage). Domain convention: caller code is physical; the
`_set_param_with_settle` pipe is raw; `_set_param_calibrated`
(lm19/scan/io.py) converts at the boundary. See
`docs/CALIBRATION_PLAN.md` for the full design.

### Live Readings

Real-time table showing for each channel:
- Raw value (protocol units).
- Calibrated value (physical units).
- Delta (calibrated − raw decoded).

Precision: Ua, Ug2 displayed as `.1f`; Ug1, Uh `.2f`; Ih `.3f`; Ia, Ig2 `.2f`.

The Live Readings and Coefficients tables are separated by a horizontal splitter.

### Calibration Data

**File:** `config/calibration.json` (version 2).

**Channels** (13 total):
- READ: `ua_read`, `ug1_read`, `ug2_read`, `uh_read`, `ih_read`,
  `ia_low_read`, `ia_high_read`, `ig2_read`.
- SET: `ua_set`, `ug1_set`, `ug2_set`, `uh_set`, `ih_set`.

Each channel stores: `gain`, `offset`, `calibrated_at` (ISO timestamp or null),
`quality` (dict with point spread, residuals, meter accuracy, or null).

**Top-level fields:**
- `version` — schema version (currently 2).
- `meter_accuracy_pct` — external instrument accuracy (±% of reading) per channel.

**Migration:** v1 `ia_read` is automatically migrated to `ia_low_read` + `ia_high_read`.

**Load validation:** any stored fit outside sanity bounds
(`fit_within_bounds`) is reset to default with a `WARNING` — with
feedforward a bad coefficient would drive real commands.

**Fallback:** If the file is missing or corrupt, defaults are used (all gain=1,
offset=0). A `WARNING` is logged.

**Configuration** (in `config/app.json`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `cal_measure_samples` | int | 10 | Default N for wizard averaging |
| `cal_measure_interval_ms` | int | 200 | Interval between wizard samples, ms |

---

## Scan Settings

Scan settings can be saved/loaded from JSON files via the File menu.

**Saved parameters:**
- Lamp type.
- Scan ranges: Ua, Ug1, Ug2 (min/max/step).
- Heater: Uh, Ih.
- Pa over-exceedance percentage.
- Ia/Ig2 samples (averaging count).
- Preheat: enabled, warmup seconds.
- Plot: Ia max, Ug2 mode, line width.

---

## Scan Stabilization

### Firmware Timing (from `TTesterLCD.c`)

The scan engine's settle parameters are derived from the actual firmware
timing. All numbers below come from the ATmega16 source code.

#### ADC Cycle

The ADC runs in free-running mode (ADATE=1) with prescaler 128:

| Parameter | Value | Source |
|-----------|-------|--------|
| Crystal | 16 MHz | `KWARC` |
| ADC prescaler | 128 | `ADPS2:1:0 = 111` |
| ADC clock | 125 kHz | 16 MHz / 128 |
| Single conversion | 104 µs | 13 ADC clocks / 125 kHz |
| Channels per round | 14 | `adc()` ISR cases 0–13: 7 measurement + 7 Ug1 comparator |
| One round | **1.46 ms** | 14 × 104 µs |
| Samples per average | 64 | `LPROB = 63` (0-based) |
| **Full averaging cycle** | **~93 ms** | 64 × 1.46 ms |

The `adc()` ISR accumulates 64 rounds of samples into `s*adc` accumulators.
On the 64th round (`probki == LPROB`), accumulators are copied to output
registers (`m*adc`) and reset to zero. The output registers are what the
UART READ command returns. This means:

- After a voltage change, the **worst-case** stale data lasts one full
  cycle (~93 ms). In practice, a stale reading is a weighted mix: if the
  voltage changed at round K, the output contains (64−K) old + K new samples.
- After waiting **≥ 93 ms** with stable voltage, the output is guaranteed
  to contain only post-change samples.

#### Ua/Ug2 PWM Ramp

Ua and Ug2 are controlled by Timer1 PWM. The firmware ramps `PWMUA` and
`PWMUG2` by **±1 per ADC round** (in `adc()` case 8 and case 12):

```c
if( PWMUA < uaset ) PWMUA++;
if( PWMUA > uaset ) PWMUA--;
```

| Parameter | Value |
|-----------|-------|
| Ramp rate | ±1 PWM step / 1.46 ms |
| Ua resolution | ~1 V/step (protocol: `Ua` raw ≈ volts) |
| **Ramp speed** | **~1.46 ms/V** |
| Configured `settle_per_volt_s` | **2.5 ms/V** (71% margin over ramp) |
| 10 V step ramp time | ~15 ms |
| 100 V step ramp time | ~146 ms |
| 300 V step ramp time | ~438 ms |

Important: the ramp is **linear**, not instant. During the ramp, the tube
passes through all intermediate voltage levels. For Ua, this means the tube
temporarily operates in potentially unsafe zones (e.g. redistribution zone
at Ua ≈ Ug2) during large jumps.

#### Ug1 Charge Pump

Ug1 uses a comparator-based charge pump with capacitor storage. The firmware
alternates between charging (`CLKUG1SET`) and discharging (`CLKUG1RST`)
every other ADC conversion (cases 1, 3, 5, 7, 9, 11, 13). The settling
speed depends on the RC time constant of the external circuit, not the
PWM ramp rate.

#### Combined Timing Budget

For a typical measurement step (10 V Ua change):

| Phase | Duration | Breakdown |
|-------|----------|-----------|
| DAC ramp | ~15 ms | 10 PWM steps × 1.46 ms |
| ADC refresh | ~93 ms | One full 64-sample cycle |
| **Minimum clean delay** | **~108 ms** | Ramp + one ADC cycle |
| Configured settle | 175 ms | 0.0025 × 10 + 0.15 |
| **Margin** | **67 ms** | 175 − 108 = 62% headroom |

For a large jump (100 V, e.g. soft landing):

| Phase | Duration | Breakdown |
|-------|----------|-----------|
| DAC ramp | ~146 ms | 100 PWM steps × 1.46 ms |
| ADC refresh | ~93 ms | One full 64-sample cycle |
| **Minimum clean delay** | **~239 ms** | Ramp + one ADC cycle |
| Configured settle | 400 ms | 0.0025 × 100 + 0.15 |
| **Margin** | **161 ms** | 400 − 239 = 67% headroom |

For a very large jump (300 V, start of scan):

| Phase | Duration | Breakdown |
|-------|----------|-----------|
| DAC ramp | ~438 ms | 300 PWM steps × 1.46 ms |
| ADC refresh | ~93 ms | One full 64-sample cycle |
| **Minimum clean delay** | **~531 ms** | Ramp + one ADC cycle |
| Configured settle | 900 ms | 0.0025 × 300 + 0.15 |
| **Margin** | **369 ms** | 900 − 531 = 69% headroom |

The `settle_per_volt_s = 0.0025` (2.5 ms/V) exceeds the ramp rate (1.46 ms/V)
by ~71%, and `settle_base_s = 0.15` adds time for at least one full ADC
cycle (93 ms) plus margin. Together they guarantee stable readings, including
time for tube transient processes (redistribution zone settling).

### Dynamic Settle

Instead of a fixed delay, settle time is calculated per-parameter based on
the actual voltage change:

```
settle = |target - previous| × settle_per_volt_s + settle_base_s
```

This means small steps (e.g. 10 V Ua) settle in ~175 ms, while large jumps
(e.g. 300 V at curve start) get ~900 ms — always enough for the PWM ramp
to complete, ADC values to refresh, and tube transients to settle.

### Verification with Retry

After the settle delay, the real value is read back from the device and
compared against the target within a tolerance. If outside tolerance, the
command is re-sent and the base settle delay is applied again, up to a
configurable number of retries.

### Per-Parameter Configuration

Each parameter has independent settings in `config/app.json`:

| Parameter | settle_per_volt_s | settle_base_s | tolerance | retries | Notes |
|-----------|-------------------|---------------|-----------|---------|-------|
| **Ua** | 0.0025 | 0.15 s | 1.0 V | 2 | PWM ramp ~1.46 ms/V |
| **Ug1** | 0.02 | 0.15 s | 0.1 V | 2 | RC circuit, tighter tolerance |
| **Ug2** | 0.0025 | 0.15 s | 1.0 V | 1 | PWM ramp, same as Ua |

- **Ua/Ug2**: PWM output ramped by firmware at ±1 step per ADC round
  (1.46 ms). `settle_per_volt_s = 0.0025` (2.5 ms/V) covers the ramp rate
  (1.46 ms/V) with 71% margin. The extra headroom accounts for tube
  transient processes (e.g. redistribution zone settling in pentodes).
  The `settle_base_s = 0.15` ensures at least one full ADC averaging
  cycle (93 ms) plus headroom.
- **Ug1**: Comparator + capacitor circuit. Faster than PWM but the capacitor
  needs charging time. `settle_per_volt_s = 0.02` (~20 ms/V) accounts for
  the RC time constant. Tolerance is 0.1 V (tighter, since Ug1 strongly
  affects transconductance accuracy).

**SRK measurement** uses separate (more conservative) settle parameters
from `config/app.json`:

| Parameter | app.json key | Default | Scan equivalent | Notes |
|-----------|-------------|---------|-----------------|-------|
| Settle per V (Ug1) | `srk_settle_per_volt_s` | 0.5 | 0.02 | 25× slower for accuracy |
| Settle base | `srk_settle_base_s` | 1.0 | 0.15 | Longer minimum wait |
| Settle (Ua) | `srk_settle_s` | 1.0 | — | Fixed per Ua attempt |
| Ua tolerance | `srk_ua_tolerance` | 2.0 V | 1.0 V | Slightly relaxed |
| Ug2 tolerance | `srk_ug2_tolerance` | 2.0 V | 1.0 V | Slightly relaxed |
| Verify retries | `srk_verify_retries` | 3 | 2 | Extra retry for SRK |
| Samples (Ia avg) | `srk_samples` | 5 | 2 | More averaging |
| Ug1 step (sweep) | `srk_ug1_step` | 0.04 V | — | 0 = classic 4-point |

All these values can be changed in `config/app.json` to trade speed for
accuracy (or vice versa).  Lower settle rates → faster but less precise;
higher samples → slower but lower noise.

### Ia/Ig2 Averaging (Samples)

The **Samples** spinbox in the Actions bar controls how many times Ia and
Ig2 are read per measurement point.  Individual Ia and Ig2 samples are
collected separately, then processed with **robust averaging**.

| Samples | Averaging method | Extra time per point |
|---------|-----------------|---------------------|
| 1 | Single read (fastest) | 0 ms |
| 2 | Simple mean | ~25 ms |
| 3–4 | Median | ~50–75 ms |
| 5+ | Trimmed mean (drop min & max) | ~100+ ms |

The default value is **3** (loaded from `scan_ia_samples` in
`config/scan.json`).

#### Outlier Detection & Re-reading

When `ia_outlier_ratio > 0` (default **2.0**) and the number of samples
is ≥ 3, the scan checks whether `max(Ia) / min(Ia)` exceeds the ratio
threshold.  The check is armed by the *largest* sample of the point:
points whose whole sample set stays below `_IA_OUTLIER_FLOOR` (0.5 mA)
are near-zero noise where the ratio carries no information, but a point
that collapses from tens of mA to ~0 on one sample (intermittent
contact) is reported — including a sample of exactly 0, where the ratio
is undefined and treated as an outlier outright.  If an outlier is
detected (e.g. due to ADC range switching near 17 mA), an additional
batch of N samples is read automatically.  The combined set of 2N samples
is then processed with robust averaging (trimmed mean for ≥ 5 total
samples), effectively filtering out the spike.

Each detection logs two WARNING lines: the operating point it happened
at (`Ia outlier at Ua=… Ug1=… Ug2=…: spread …`) and what robust
averaging made of it (`Ia outlier resolved: samples […] mA, trimmed
mean of 6 → Ia=… mA`).  The sample list keeps the read order, so a
spike in the first sample (setpoint still settling) is distinguishable
from a sporadic one.

Configuration in `config/scan.json`: `ia_outlier_ratio` (0 = disabled)
and `ia_outlier_reread_samples` (extra samples per detection; 0 = warn
and count only, no re-read).  Mind that the resulting pool size selects
the averaging rule — 3–4 samples give a median, 5+ a trimmed mean, and
a trimmed mean drops only one extreme per side, so two correlated
spikes leave one of them in the result.

**Precision impact:** Each hardware ADC reading is already averaged over
64 samples (~93 ms cycle).  Software N-sample averaging adds a second
layer.  Effective Ia noise after both stages:

| Samples (N) | Effective Ia noise (20mA range) | Notes |
|-------------|-------------------------------|-------|
| 1 | ~0.01 mA | 64 HW avg only |
| 3 | ~0.006 mA | Default scan (median) |
| 5 | ~0.005 mA | Default SRK (srk_samples) |

The hardware 64-sample averaging dominates noise reduction.  Software
samples primarily help with occasional ADC outliers and charge pump
timing misalignment.  For SRK measurements, 5 samples are recommended
to ensure stable S/R/K values.

### Measurement Timing

#### Time per point breakdown

Each measurement point consists of three phases:

| Phase | Duration | Depends on |
|-------|----------|------------|
| **Settle** | 20–900 ms | Voltage change × settle_per_volt_s + base_s |
| **Verify** | ~25 ms | UART roundtrip (?:Param; → response) |
| **Read** | 25–125 ms | N × ~25 ms per sample (Ia + Ig2 reads) |

**Settle detail:** The settle time varies per parameter and voltage step:

| Scenario | Ua settle | Ug1 settle | Total settle |
|----------|----------|-----------|-------------|
| Small Ua step (10 V) | 0.0025×10 + 0.15 = 175 ms | — | ~175 ms |
| Large Ua jump (300 V) | 0.0025×300 + 0.15 = 900 ms | — | ~900 ms |
| Ug1 step (0.5 V) | — | 0.02×0.5 + 0.15 = 160 ms | ~160 ms |
| Ug1 step (2 V, SRK) | — | 0.5×2 + 1.0 = 2000 ms | ~2000 ms |
| New Ug1 curve (6 V) | — | 0.02×6 + 0.15 = 270 ms | ~270 ms |

Note: SRK settle uses longer per-volt rates (`srk_settle_per_volt_s = 0.5`)
than scan (`scan_ug1_settle_per_volt_s = 0.02`) for higher accuracy.

#### Typical scan durations

Times below are calculated for the **default `app.json` settings**:

```
scan_ua_settle_per_volt_s = 0.0025   scan_ua_settle_base_s = 0.15
scan_ug1_settle_per_volt_s = 0.02    scan_ug1_settle_base_s = 0.15
scan_ug2_settle_per_volt_s = 0.0025  scan_ug2_settle_base_s = 0.15
scan_ia_samples = 3
```

| Tube type | Grid | Points | Approx. time | Calculation |
|-----------|------|--------|-------------|-------------|
| Triode (ECC83) | 31 Ua × 7 Ug1 | 217 | ~50–70 s | 217 × 0.2s + 7 × 0.27s + UART |
| Triode (fine) | 61 Ua × 13 Ug1 | 793 | ~3–4 min | 793 × 0.2s + 13 × 0.27s |
| Pentode (6P18P) | 31 Ua × 7 Ug1 × 2 Ug2 | 434 | ~2–3 min | + Ug2 settle per level |
| + Adaptive refine | +10–30% | +50–130 | +15–45 s | Same per-point time |

**Per-point time at these settings** (10V Ua step, 3 samples):
- Ua settle: 0.0025 × 10 + 0.15 = **175 ms**
- Verify: ~**25 ms**
- Read (3 samples): ~**50 ms**
- **Total: ~250 ms/point** + ~270 ms per Ug1 curve change (0.02 × 6V + 0.15)

#### SRK measurement durations

Times below are calculated for the **default SRK `app.json` settings**:

```
srk_settle_per_volt_s = 0.5    srk_settle_base_s = 1.0
srk_settle_s = 1.0             srk_samples = 5
srk_ua_tolerance = 2.0         srk_ug1_step = 0.04
```

| Mode | Points | Time/point | Total | Calculation |
|------|--------|-----------|-------|-------------|
| Classic (4-point) | 4 | ~1.7 s | ~7 s | 1.0s Ua settle + 0.5×0.04+1.0=1.02s Ug1 + 0.125s read |
| Sweep 0.08V (ΔUg1=0.48V) | 14 | ~1.7 s | ~24 s | 7 Ug1 × 2 Ua |
| Sweep 0.04V (ΔUg1=0.48V) | 26 | ~1.7 s | ~45 s | 13 Ug1 × 2 Ua |
| Sweep 0.04V (ΔUg1=2.0V) | 102 | ~1.2 s | ~2 min | 51 Ug1 × 2 Ua, small Ug1 steps |

**Per-point time at these settings** (Ua ΔV=100V, Ug1 Δ=0.04V, 5 samples):
- Ua settle: 0 × 100 + **1.0 s** (fixed `srk_settle_s`)
- Ug1 settle: 0.5 × 0.04 + 1.0 = **1.02 s**
- Verify (Ua + Ug1 + Ug2): ~**75 ms**
- Read (5 samples): ~**125 ms**
- **Total: ~1.7 s/point** (Ug1 and Ua settle overlap with different points)

**Why SRK is slower per point:** `srk_settle_per_volt_s` is 25× larger
than scan's (0.5 vs 0.02), `srk_settle_base_s` is 6.7× larger (1.0 vs
0.15), and 5 vs 2 samples add ~75 ms.  This is deliberate — SRK accuracy
depends on each individual point being as precise as possible.

#### Speed optimization tips

All timing parameters are configurable in `config/app.json` (see
[Per-Parameter Configuration](#per-parameter-configuration) above for the
full list of scan and SRK settle/verify/retry keys).

| What to change | Key in `app.json` | Default → Fast | Effect on time | Trade-off |
|----------------|-------------------|---------------|---------------|-----------|
| Ua zone range | UI spinboxes | ±10V | Fewer points | Less coverage |
| Use scan for S | "Measure" checkbox off | — | +0 s | Lower precision (~5–10%) |
| Ug1 step | `srk_ug1_step` | 0.04 → 0.08 | ÷2 points | σ(S) 1.5% vs < 1% |
| Ia samples | `srk_samples` | 5 → 3 | −50 ms/pt | Slightly higher noise |
| Ug1 settle rate | `srk_settle_per_volt_s` | 0.5 → 0.2 | ÷2.5 Ug1 settle | Less accuracy on slow tubes |
| Base settle | `srk_settle_base_s` | 1.0 → 0.5 | −500 ms/pt | Less safe for large steps |
| Scan Ua settle | `scan_ua_settle_per_volt_s` | 0.002 → 0.001 | −50% Ua settle | May fail verify |
| Scan Ug1 settle | `scan_ug1_settle_per_volt_s` | 0.02 → 0.01 | −50% Ug1 settle | May fail verify |
| Scan samples | `scan_ia_samples` | 3 → 1 | −50 ms/pt | No robust averaging, ~40% more Ia noise |

**Example:** With `srk_settle_per_volt_s = 0.2`, `srk_settle_base_s = 0.5`,
`srk_samples = 3`, `srk_ug1_step = 0.08`: SRK sweep in a 0.48V zone takes
~10 s (14 pts × ~0.7 s) instead of ~45 s — 4.5× faster, with σ(S)/S ≈ 2%.

### Pg2 (Screen Grid Power) Protection

Pentodes and tetrodes can suffer screen grid overheating when anode voltage
is low relative to screen voltage — most of the cathode current is captured
by the screen grid instead of the anode.

**Where Pg2 is highest:**
- Independent Ug2 sweep with high Ug2 (200–300 V) and low Ua (0–50 V).
- Pg2 = Ug2 × Ig2 can easily exceed the rated limit (e.g. 2.5 W for 6P18P).

**How protection works:**

For **independent Ug2 sweep** — before the Ug1 loop begins, the scan
engine evaluates the Pg2 boundary **once per Ug2 level** at the
worst-case Ug1 (closest to 0 V = maximum cathode current = highest Ig2).
This yields a conservative safe Ua index that is then applied to **all**
Ug1 curves at that Ug2 level.

The boundary search uses a two-phase algorithm starting from an
estimated safe point (Ua ≈ Ug2 × 0.5):
1. **Probe the estimate.** If safe — search **downward** toward lower Ua
   to find the exact transition point.
2. If the estimate itself exceeds the limit — search **upward** toward
   higher Ua to find the first safe point.

Points below the safe Ua are skipped entirely (voltage is never applied).
The search always approaches the danger zone from the **safe side**, so
the tube is never exposed to extreme Pg2 values. If no safe point exists,
the entire Ua sweep for that Ug2 level is skipped.

For **triode-connected** mode — Ug2 tracks Ua, so Pg2 is not monotonic.
A post-measurement check skips individual points where Pg2 exceeds the
limit (`continue`, not `break`).

For **true triode** mode — Ug2 = 0, Pg2 protection is not needed.

**Pg2 safety in refine pass:**
The Pg2 safe boundary from the coarse pass is cached per Ug2 level.
During the refine pass (independent sweep), midpoints below the safe
boundary are automatically excluded — the tube is never driven into the
high-Pg2 zone, even if the refine analysis marks boundary-adjacent
intervals.

**Configuration / UI:**
- `Pig2_max` — per-lamp in `config/lamps.json` (managed via `config/lamp_limits.json` + lamp extractor).
- `Pg2 over %` — UI spinbox in the Ug2 mode row (visible for pentodes only).
  Default value from `scan_pig2_over_pct` in `config/app.json` (default 20%).
  Set 0 to disable.
- Pg2 limit = Pig2_max × (1 + pig2_over_pct / 100).

### Safe Down-Sweep (Pentode Mode)

In independent-Ug2 pentode mode, the scan engine uses a bidirectional sweep
for each (Ug2, Ug1) pair: UP from `Ua ≈ Ug2` to `Ua_max`, then DOWN from
`Ua ≈ Ug2` toward `Ua = 0`. The DOWN sweep enters the **redistribution zone**
(`Ua < Ug2`) where screen grid current Ig2 rises steeply and is very sensitive
to voltage changes. Two mechanisms prevent Ig2 spikes and premature sweep
termination:

**1. Soft Landing.** Before the DOWN sweep begins, the scan engine explicitly
settles Ua at `ua_values[start_idx]` (the grid point closest to Ug2). This
breaks a potentially large voltage jump from the UP sweep end (high Ua) into
two smaller steps and ensures the tube stabilizes at the Ua ≈ Ug2 boundary
before descending.

**2. Step Bisection.** If the native Ua grid step exceeds `down_max_step_v`
(default 25 V), intermediate measurement points are automatically inserted
so that no single Ua change in the DOWN sweep exceeds the configured limit.
For example, with `ua_step = 50 V` and `down_max_step_v = 25`, each 50 V
interval is split into two 25 V sub-steps, each measured. This makes the
descent through the steep Ig2 gradient smoother and gives the firmware
averaging cycle time to produce representative readings.

Intermediate points are **settle-only**: the engine moves to them, reads
Ig2/Pg2 for safety checks, but does **not** save them as measurement data.
Only original grid points appear in the scan results. This avoids artifacts
from firmware averaging lag: although the ADC cycle is only ~93 ms, the
settle time between small intermediate steps (e.g. 12.5 V → 0.002×12.5 +
0.15 = 175 ms) may not fully account for the tube's transient behavior
in the redistribution zone, producing slightly stale current readings.

**3. Predictive Ig2 Check.** Before moving to the next Ua in the DOWN sweep,
the engine extrapolates Ig2 from already measured points:
- **2 points** → linear extrapolation;
- **3+ points** → quadratic extrapolation (captures the convex Ig2 rise
  in the redistribution zone more accurately than linear).

If the predicted Ig2 exceeds `ig2_limit × 0.8`, the sweep is stopped
**before** the tube is moved to the dangerous operating point. This avoids
triggering hardware protection and the associated Ig2 spike.

| Config key | Default | Description |
|---|---:|---|
| `scan_down_max_step_v` (`config/scan.json`: `down_max_step_v`) | `25` | Maximum Ua step (V) in the pentode DOWN sweep. Set `0` to disable bisection. |

### Adaptive Refine (Two-Pass Scan)

When enabled (via the **Refine** checkbox in the Actions bar, or
`scan_refine_enabled` in `config/app.json`), the scan engine performs a
second pass to add measurement points in regions where the coarse grid
missed important curve features.

**How it works:**
1. **Coarse pass** — normal scan with the configured Ua step.
2. **Analysis** — five physics-based criteria are evaluated on every
   consecutive pair/triple of measured points.  Triggered intervals are
   collected across **all Ug1 curves** into a unified set.
3. **Bisection** — each flagged interval is recursively bisected up to
   `refine_max_depth` times, respecting `refine_min_step_ua`.
4. **Refine pass** — new midpoint Ua values are measured for **every**
   Ug1 curve, producing a consistent (non-uniform) grid.

**Criteria:**

| # | Name | Triggers when | Tube types |
|---|------|---------------|------------|
| C1 | Onset | Ia crosses `onset_ma` threshold (tube starts conducting) | All |
| C2 | Curvature | Normalised second derivative exceeds `curvature_thr` | All (pentode knee, triode onset) |
| C3 | Gradient ratio | Slope changes by more than `gradient_ratio`× between neighbours | All |
| C4 | Ig2 kink | Ig2 gradient reverses sign (secondary emission) | Pentode only |
| C5 | Ia jump | Single-step Ia change exceeds `delta_ia_thr` fraction of range | All |

For **independent Ug2 sweep**, analysis and refine are performed
per Ug2 level.  For **triode** and **triode-connected** modes,
all points are analysed as a single group.

**Configuration (in `config/app.json`):**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `scan_refine_enabled` | Enable two-pass adaptive refine | `false` |
| `scan_refine_max_depth` | Max bisection depth (1 = halve, 2 = quarter) | `2` |
| `scan_refine_min_step_ua` | Minimum Ua step after bisection (V) | `3` |
| `scan_refine_onset_ma` | C1: "zero current" threshold (mA) | `0.5` |
| `scan_refine_curvature_thr` | C2: curvature threshold (0–1, normalised) | `0.15` |
| `scan_refine_gradient_ratio` | C3: slope change ratio | `3.0` |
| `scan_refine_ig2_delta_min` | C4: min Ig2 delta for kink (mA) | `0.5` |
| `scan_refine_delta_ia_thr` | C5: Ia jump threshold (fraction of range) | `0.25` |

### Communication Error Recovery

UART communication can occasionally produce garbled responses (e.g. partial
parameter reads due to electrical noise or timing issues). Instead of
aborting the entire scan, the engine uses a two-tier recovery strategy:

**Tier 1 — Silent auto-retry.** When `_read_measurement_point` raises a
`ValueError` or `RuntimeError`, the serial input buffer is flushed and the
read is retried up to `comm_retries` times (default 2, configurable in
`config/scan.json`). This handles transient glitches transparently.

**Tier 2 — User prompt.** If auto-retries are exhausted, a dialog is shown
with three options:

| Button | Action |
|--------|--------|
| **Retry** | Flush buffer, reset retry counter, try again |
| **Ignore** | Skip this measurement point, continue scan |
| **Abort** | Stop the scan (partial results can be saved) |

The dialog shows the error message, attempt count, and number of points
collected so far.

**Implementation details:**
- `LM19Serial.flush_input()` clears the pyserial input buffer after errors.
- `ScanWorker` uses a `threading.Event` to block the worker thread while
  waiting for the user's response in the UI thread.
- The `stop()` method wakes the event, so the Stop button works even while
  the error dialog is open.
- If the user chooses Abort, the existing partial-save logic in
  `_on_scan_failed` handles offering to keep collected points.

**Configuration (in `config/scan.json`):**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `comm_retries` | Number of silent auto-retries before asking user | `2` |

---

### Heater Alive Check

During a scan, the engine monitors heater voltage (Uh) and current (Ih) after
every measurement point. If the heater readings drop below safe thresholds,
the scan stops immediately and emits a `heater_lost` progress event so the
UI can warn the operator.

**Detection logic (`_check_heater`):**

| Condition | Threshold | Meaning |
|-----------|-----------|---------|
| `settings.uh > 0` and `point["uh"] < 0.5 V` | `_UH_LOST_THRESHOLD = 0.5 V` | Voltage-heated tube lost filament |
| `settings.ih > 0` and `point["ih"] < 0.02 A` | `_IH_LOST_THRESHOLD = 0.02 A` | Current-heated tube lost filament |

Only the configured heater channel is checked (voltage or current, not both).
If neither `uh` nor `ih` is set in settings, the check is skipped entirely.

**How it works:**
1. `_read_point()` reads the measurement and passes it to `_check_heater()`.
2. If the check fails, it sets an internal `heater_lost_msg` flag and returns
   `None` (as if the point was skipped).
3. All scan loops check `_stopped()` which combines the user stop request with
   the heater-lost flag, so every nested loop breaks quickly.
4. After exiting the scan loops, a `{"event": "heater_lost", "message": ...}`
   event is emitted via the `progress` callback.
5. Points collected before the failure are preserved and returned normally.

**Typical causes of heater loss:**
- Tube removed from socket during scan
- Broken filament (tube failure)
- Bad socket contact

---

## Snap-to-Curve Marker

Available on: **2D**, **Transfer**, **Curves**, and **Compare** plots.

A visual marker that snaps to the nearest curve and slides along it as
you move the mouse. Shows precise interpolated values between measurement
points.

**Visual elements:**
- **Crosshair** — dashed horizontal + vertical lines through the marker position.
- **Marker dot** — bright red dot on the curve.
- **Tooltip** — shows interpolated values (Ia, Ua, Ug1, Ug2, Pa) near the marker.

**Interaction:**
- **Mouse move** — marker automatically snaps to the nearest curve (free mode), interpolating Y between data points.
- **Left click near curve** — lock to that curve (marker slides along it, won't jump to others). Tooltip shows *locked*. Proximity threshold is configurable via `marker_lock_px` in `config/app.json` (default 15 pixels).
- **Left click near curve while locked** — cycle to the next curve at this X position (sorted by Y distance). Allows switching between overlapping curves from different lamps.
- **Left click away from curve while locked** — unlock (back to free auto-snap mode).
- **Left click away from any curve while unlocked** — nothing happens (no accidental lock).

**Mixed overlay support:**
- When overlaying measurements with different modes (e.g. triode-connected + pentode sweep), the marker detects Ug2-tracking per series independently, so curves from both modes are visible.
- Measurements from different entries are distinguished by `_entry_idx`, so even lamps with the same type/id produce separate marker curves.

**Supported plots and tooltip fields:**

| Plot | X axis | Y axis | Extra fields |
|------|--------|--------|--------------|
| 2D | Ua (V) | Ia (mA) | Ug1, Ug2, Pa |
| Transfer | Ug1 (V) | Ia (mA) | Ua |
| Curves | configurable | configurable | Ua, Ug1, Ug2, Ia, Ig2, Pa |
| Compare | Ua (V) | Ia (mA) | Ug1, Ug2, Source |

---

## Operating-Point Swing Block (Amplifier tab)

For every computed operating point the results panel shows, besides the
Q-point and THD spectrum:

- **Ua swing** — peak-to-peak anode voltage with min…max (for SE
  transformer stages the anode swings above the supply);
- **Ia range** — min…max anode current: the cutoff margin that explains
  the amp class (i_min → 0 means class AB/B operation);
- **Grid drive** — required driver amplitude, ± volts and Vpp;
- **P1 (fundamental)** — DFT only: the output power an external meter
  would read (the peak-based Pout overstates it under compression);
- **Iq per tube** — Push-Pull only: the quiescent current the chosen
  bias produces in each tube.

The optimizer status line additionally reports Pa, amp class and the UL
tap of the best point, and the same tap is shown when a Pareto/Top-N
point is applied.

## Warnings Indicator (top status bar)

Every degraded result is surfaced in the UI (project rule 2026-07-04) —
the log records everything, but the interface is the primary channel:

- **⚠ N indicator** at the right end of the main-window top status bar
  aggregates warnings from the last analysis run, the last optimizer run,
  the last model fit (its verdict and its alerts) and application startup
  (broken config/calibration/data files). Click it for the full grouped
  list; it hides itself when there is nothing to report. The window has
  no bottom status bar — it stayed blank in normal operation, so the
  indicator lives in the top bar and one-off notices (settings load) use
  a dialog.
- **Amplifier tab results panel** prepends orange ⚠ lines when the
  analysis is degraded: a requested model was not found (fell back to
  raw measurements), the Ug2 filter matched no points (levels mixed),
  DFT was requested without a model (5-point used), or Newton samples
  failed to converge (THD/Pa avg may be inaccurate near clipping).
- **Optimizer status** lists refine-phase degradations: scipy missing,
  refinement failed, or Nelder-Mead not fully converged.
- **Model dialog** shows fitter warnings under the fit summary
  (too few Ug2 levels for Reefman splitting, underdetermined grid fit,
  below-knee masking without Ig2 data).
- **Emergency stop** reports channels that FAILED to zero in a critical
  dialog — the tube may still be under voltage on those.
- **Tube Health protection dialog** warns first of all when the post-trip
  Ug1 restore failed (the tube may still be conducting).
- **Scan summary dialog** appends ⚠ lines when setpoints failed to settle
  within tolerance (points measured off the requested operating point) or
  Ia was unstable and re-read with robust averaging (check tube contact) —
  even when every curve completed.
- **SRK results** carry the measured uncertainty: ±% per repeat in the
  results table, a summary line under the S/R/K label, and an
  ``uncertainty`` block in the saved measurement JSON.
- **Combined export** (Compare tab) validates homogeneity: measurements
  with different Ug2 modes cannot be combined (physically incompatible);
  different tube types ask for confirmation (deliberate analog merges
  like 6P14P ≈ EL84 stay possible). The SPICE "Saved" dialog also warns
  when Kg2 was exported unfitted (no Ig2 data), and the .utd export
  reports grid cells written as Ia=0.0 (curves cut by protection).

---

## Keyboard / UI Tips

- **Double-click** a color cell in Compare table to change the curve color.
- **Ug2 slice** dropdown updates automatically when new data arrives.
- **Auto Ia / Auto U** buttons rescale axes to fit current data.
- All limit values (Pa max, Ua max, Ia max, load line) are auto-filled from
  the lamp configuration when you select a tube type.
- Hover over any control to see its tooltip with detailed description.

---

*Generated for LM19 Tube Tester v1.0*
